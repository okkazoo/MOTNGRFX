"""Three-layer shot route (tools/app/layered_shots.py, nodes_fusion/layers.py, tools/fusion/layers_recipe.py, 16_app/LAYERS.md).
No model, no GPU. Graph tests read the base graphs on disk (app/graphs/); compose tests are tiny synthetic numpy frames; the node tests need a ComfyUI
checkout (skipped otherwise: MOGRAPH_COMFYUI_PATH).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_layered_shots.py -q"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.app import layered_shots as LS  # noqa: E402
from tools.fusion import compose as C  # noqa: E402
from tools.fusion import layers_recipe as LR  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import window_cut as W  # noqa: E402

GRAPHS = ROOT / "app" / "graphs"
DESIGN = ["fusion_design_qwen", "fusion_design_qwen_auto"]
REQ = {"words": "RISE\nHOLD\nGO", "scene": "a lone runner in a concrete tunnel", "music": "kinetic and emotional, 160 bpm",
       "length": 12, "look": "auto", "seed": 6, "quality": "preview", "format": "9:16"}


@pytest.fixture(autouse=True)
def _restore_backend():
    prev = C.B
    yield
    C.B = prev


def _load(name):
    p = GRAPHS / f"{name}.json"
    if not p.is_file():
        pytest.skip(f"{p} not built (python tools/app/build_graphs.py)")
    return json.loads(p.read_text(encoding="utf-8")), json.loads((GRAPHS / f"{name}.app.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan():
    owned = dict(REQ, owners=json.dumps([{"render": "handoff", "material": "carved ice"}] * 3))          # the default owner is flat since 2026-10-29: these tests are about the text pass of LTX-owned words
    from tools.fusion import owner as OW, user_style as US
    p = PR.plan_piece(owned, synth_music(160, 12))
    US.stamp(p, "fill the letters with carved ice")                     # since 2026-11-01 letters are solid (words flat) unless the user asks for a material: the typed text lands on the piece, the owner pass runs again
    return OW.apply_to_plan(p)


# ============================================================================================ graph: the three layers per shot
@pytest.mark.parametrize("base", DESIGN + ["fusion"])
def test_converted_graph_has_clean_matte_and_text_chains_per_shot(base):
    g0, s0 = _load(base)
    g, s = LS.convert(g0, s0)
    assert LS.check(g) == []
    ks = LS._slots(g)
    assert ks == [1, 2, 3, 4, 5, 6]
    for k in ks:
        assert not any(n.startswith(f"c{k}.") for n in g), "the old union-control chain is gone"
        # 1. CLEAN: plain image to video, base model, no IC guide, prompt without the letter clauses
        clean = [n for n in g if n.startswith(f"l{k}.")]
        assert {g[n]["class_type"] for n in clean}.isdisjoint({"LTXAddVideoICLoRAGuide", "LTXVCropGuides"})
        assert all(g[n]["inputs"]["model"] == ["1", 0] for n in clean if g[n]["class_type"] == "CFGGuider")
        assert g[f"l{k}.6"]["inputs"]["text"] == [f"lay_prompt{k}", 0] and g[f"lay_prompt{k}"]["inputs"]["prompt"] == [f"slot{k}", 0]
        assert g[f"l{k}.dec"]["class_type"] == "MographDecodeClip" and g[f"l{k}.dec"]["inputs"]["index"] == k
        # 2. MATTE: Alpha-Gen is fed by the gather, which builds its reference from the CLEAN clip
        assert g[f"a{k}.17"]["inputs"]["input"] == ["gather", k]
        # 3. TEXT: union-control guide on the type control pass, started from the (gated) clean latent
        tx = [n for n in g if n.startswith(f"t{k}.")]
        guide = g[f"t{k}.32"]
        assert guide["class_type"] == "LTXAddVideoICLoRAGuide" and guide["inputs"]["strength"] == [f"slot{k}", 6]
        assert guide["inputs"]["image"] == [f"t{k}.20", 0] and g[f"t{k}.17"]["inputs"]["input"] == [f"slot{k}", 3]      # the type control slice
        assert g[f"t{k}.31"]["inputs"]["latent"] == [f"lay_gate{k}", 0]
        assert g[f"lay_gate{k}"]["inputs"] == {"latent": [f"l{k}.40", 0], "after": ["lay_clean", 0]}
        assert g[f"t{k}.39"]["inputs"]["sigmas"] == ["lay_sig", 0]
        assert all(g[n]["inputs"]["model"] == ["5", 0] for n in tx if g[n]["class_type"] == "CFGGuider")
        assert g[f"gather"]["inputs"][f"text_{k}"] == [f"t{k}.dec", 0]
        if k > 1:
            assert g[f"slot{k}"]["inputs"]["prev"] == [f"l{k - 1}.dec", 0], "the next shot starts from the previous CLEAN clip"
    assert g["gather"]["class_type"] == "MographLayerGather" and g["gather"]["inputs"]["clean"] == ["lay_clean", 0]
    assert g["lay_sig"]["inputs"]["sigmas"] == "0.975, 0.909375, 0.725, 0.421875, 0.0"
    assert s["layers"]["v2v_steps"] == 4 and "three-layer" in s["title"]
    assert "archive" not in g0 or "archive" in g, "the archive node survives the pruning"


def test_the_input_graph_is_not_modified_and_the_conversion_is_deterministic():
    g0, s0 = _load(DESIGN[0])
    snap = copy.deepcopy((g0, s0))
    a = LS.convert(g0, s0)
    b = LS.convert(g0, s0)
    assert (g0, s0) == snap and a == b


def test_v2v_start_selects_the_tail_of_the_distilled_schedule():
    assert LS.v2v_sigmas(0.909375) == "0.909375, 0.725, 0.421875, 0.0" and LS.steps_of(0.909375) == 3
    assert LS.v2v_sigmas(0.975) == "0.975, 0.909375, 0.725, 0.421875, 0.0" and LS.steps_of(0.975) == 4
    with pytest.raises(ValueError):
        LS.v2v_sigmas(0.5)
    g0, s0 = _load(DESIGN[0])
    g, _ = LS.convert(g0, s0, v2v_start=0.975)
    assert g["lay_sig"]["inputs"]["sigmas"].startswith("0.975,")


def test_passes_the_repo_graph_checker():
    from tools.app import build_graphs as BG
    for base in DESIGN:
        g0, s0 = _load(base)
        g, s = LS.convert(g0, s0)
        bad, _notes = BG.check(g, s)
        assert bad == [], bad
    g0, s0 = _load("fusion")                                   # the two-stage graph converts and validates too
    g, s = LS.convert(g0, s0)
    assert LS.check(g) == [] and s["layers"]["two_stage"] is True
    assert g["t1.57"]["inputs"]["sigmas"] == ["56", 0], "stage 2 of the text pass keeps the refine schedule"
    assert g["l1.50"]["inputs"]["samples"] == ["l1.40", 0], "the clean chain feeds the upsampler directly (no crop guides)"


def test_checker_finds_a_broken_layered_graph():
    g0, s0 = _load(DESIGN[0])
    g, _ = LS.convert(g0, s0)
    bad = copy.deepcopy(g)
    bad["l1.36"]["inputs"]["model"] = ["5", 0]                       # clean shot with the union-control LoRA
    bad["t2.39"]["inputs"]["sigmas"] = ["38", 0]                     # text pass with the full schedule
    bad["a3.17"]["inputs"]["input"] = ["lay_clean", 0]
    msgs = " | ".join(LS.check(bad))
    assert "l1.36" in msgs and "t2.39" in msgs and "a3.17" in msgs


@pytest.mark.skipif(not any((ROOT / "app" / "graphs" / f"{b}.json").exists() for b in ("fusion_design_qwen", "fusion_design_qwen_auto")),
                    reason="dev fixture not shipped in the public export: app/graphs/fusion_design_qwen.json")
def test_write_and_check_roundtrip(tmp_path):
    assert LS.main(["--out-dir", str(tmp_path)]) == 0
    assert LS.main(["--out-dir", str(tmp_path), "--check"]) == 0
    p = tmp_path / "fusion_design_qwen_layers.json"
    assert json.loads(p.read_text(encoding="utf-8"))["gather"]["class_type"] == "MographLayerGather"
    p.write_text("{}", encoding="utf-8")
    assert LS.main(["--out-dir", str(tmp_path), "--check"]) == 1      # stale file is reported


# ============================================================================================ guide == 0: no V2V pass
def test_guide_zero_means_no_text_pass(plan):
    assert plan["technique"] == "behind"
    assert all(LS.wants_text(plan, s["id"]) for s in plan["shots"]), "behind: depth control at guide 0.4 -> a text pass per shot"
    p0 = copy.deepcopy(plan)
    for sh in p0["shots"]:
        sh["control"] = {"pass": "depth", "guide": 0.0}
    assert not any(LS.wants_text(p0, s["id"]) for s in p0["shots"])
    p1 = copy.deepcopy(plan)
    p1["shots"][1]["control"] = {"pass": "depth", "guide": 0}
    assert [LS.wants_text(p1, s["id"]) for s in p1["shots"]][:3] == [True, False, True]
    p2 = copy.deepcopy(plan)
    p2["layers"] = {"text": False}                                    # the piece opts out
    assert not LS.wants_text(p2, "s1")
    u8 = np.zeros((9, 4, 4, 3), np.uint8)
    c = LS.layer_clip({"u8": u8, "sid": "s1", "start": 0, "fps": 24}, None)
    assert c["u8"] is u8 and c["subject"] is u8 and c["textbg"] is False   # no text pass: the background IS the clean clip


def test_layer_clip_background_is_the_text_clip_and_subject_the_clean_clip():
    torch = pytest.importorskip("torch")
    clean = {"u8": torch.full((9, 4, 4, 3), 10, dtype=torch.uint8), "sid": "s2", "start": 5, "fps": 24}
    text = {"u8": torch.full((9, 4, 4, 3), 200, dtype=torch.uint8), "sid": "s2", "start": 5, "fps": 24}
    c = LS.layer_clip(clean, text)
    assert int(c["u8"][0, 0, 0, 0]) == 200 and int(c["subject"][0, 0, 0, 0]) == 10 and c["textbg"] is True
    assert (c["sid"], c["start"], c["fps"]) == ("s2", 5, 24)
    short = {"u8": torch.full((5, 4, 4, 3), 200, dtype=torch.uint8), "sid": "s2", "start": 5, "fps": 24}
    assert LS.layer_clip(clean, short)["u8"].shape[0] == 9            # a short text clip holds its last frame


def test_clean_prompt_drops_only_the_letter_clauses():
    built_in = ("a cathedral at golden hour, glowing letters built into the scene as part of the world, a slow dolly in, cinematic")
    assert LS.clean_prompt(built_in) == "a cathedral at golden hour, a slow dolly in, cinematic"
    behind = "a runner, the subject stands perfectly still while the air moves around it, the camera pans left, cinematic"
    assert LS.clean_prompt(behind) == behind
    assert "strands" not in LS.clean_prompt("a lake, long flowing strands draw out and trace the lines, calm")


# ============================================================================================ compose: the subject is the CLEAN clip
H, Wd, N = 96, 64, 4
BG = np.array([0.8, 0.2, 0.2], np.float32)          # the text-rendered background (red)
CLEAN = np.array([0.2, 0.2, 0.8], np.float32)       # the clean clip (blue)
SUBJ = np.zeros((H, Wd), np.float32)
SUBJ[20:60, 10:40] = 1.0
TYPE = np.zeros((H, Wd), np.float32)
TYPE[40:70, 20:55] = 1.0
P_SUBJ, P_OVERLAP, P_TYPE, P_BG = (30, 15), (50, 30), (65, 45), (5, 5)


def u8(c):
    return np.broadcast_to(np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8), (N, H, Wd, 3)).copy()


def _type_dir(tmp):
    d = Path(tmp) / "matte"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        cv2.imwrite(str(d / f"matte_{i:05d}.png"), (TYPE * 255).astype(np.uint8))
    return str(tmp)


def _piece(ops):
    return {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [Wd, H]}, "shots": [{"id": "s1", "frames": N, "matte": True}],
            "compose": {"crossfade": 0.0, "default": {"ops": ops}}}


SANDWICH_OPS = [
    {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.5}},
    {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_rgb": [1.0, 1.0, 1.0], "type_a": "$type_matte", "matte": "$matte",
                                                "matte_threshold": 0.5, "matte_soft": 0.08}},
]


def _run(ops, subject):
    with tempfile.TemporaryDirectory() as tmp:
        td = _type_dir(tmp)
        out, st = RE.run_compose(_piece(ops), "s1", type_dir=td, footage=u8(BG), matte=np.broadcast_to(SUBJ, (N, H, Wd)).copy(), backend="numpy",
                                 threads=0, subject=subject)
    return np.asarray(out, np.uint8)[0].astype(np.float32) / 255.0


def test_compose_puts_the_clean_subject_between_the_type_and_the_camera():
    out = _run(SANDWICH_OPS, u8(CLEAN))
    sub = CLEAN * 0.5                                                  # the subject is graded like the plate, from the CLEAN clip
    assert np.allclose(out[P_SUBJ], sub, atol=0.03), out[P_SUBJ]       # subject pixel: clean (blue), NOT the background (red)
    assert np.allclose(out[P_OVERLAP], sub, atol=0.03)                 # subject over the type where they overlap (type BETWEEN subject and background)
    assert np.allclose(out[P_TYPE], 1.0, atol=0.03)                    # type over the background outside the subject
    assert np.allclose(out[P_BG], BG * 0.5, atol=0.03)                 # the background keeps the text-rendered clip
    single = _run(SANDWICH_OPS, None)                                  # default: ONE clip, the subject is the background's own pixels
    assert np.allclose(single[P_SUBJ], BG * 0.5, atol=0.03) and np.allclose(single[P_OVERLAP], BG * 0.5, atol=0.03)
    assert np.allclose(single[P_TYPE], 1.0, atol=0.03) and np.allclose(single[P_BG], out[P_BG], atol=1e-6)


def test_single_clip_recipe_is_unchanged_by_the_layer_support(plan):
    with tempfile.TemporaryDirectory() as tmp:
        for k in ("matte", "line", "beauty", "depth"):
            os.makedirs(os.path.join(tmp, k))
        a = RE.build_recipe(plan, "s1", type_dir=tmp)
        b = RE.build_recipe(plan, "s1", type_dir=tmp, layered=False)
    assert a == b and "subject" not in a["inputs"] and "layers" not in a
    assert not LR.uses_subject(a["ops"])


def test_layerized_planner_recipe_reads_the_subject_everywhere_it_takes_one(plan):
    """the real `behind` recipe of the planner (sandwich, front, wipe, window treatments): every subject-taking op reads the clean clip."""
    with tempfile.TemporaryDirectory() as tmp:
        for k in ("matte", "line", "beauty", "depth"):
            os.makedirs(os.path.join(tmp, k))
        rec = RE.build_recipe(plan, "s1", type_dir=tmp, layered=True)     # raises on any undefined reference
        base = RE.build_recipe(plan, "s1", type_dir=tmp)
    assert rec["inputs"]["subject"] == dict(src="subject", kind="rgb")
    assert rec["layers"]["changed"] >= 4 and rec["layers"]["skipped"] == []
    ops = rec["ops"]
    by_out = {}
    for o in ops:
        by_out.setdefault(o.get("out", "image"), []).append(o)

    def roots(name, seen=None):
        """the root input a buffer derives from through the clonable plate chain: 'footage' | 'subject' | None."""
        o = by_out[name][-1]
        a = o["args"]
        for key in ("frame", "img", "a", "footage"):
            v = a.get(key)
            if isinstance(v, str) and v.startswith("$"):
                r = v[1:]
                return r if r in ("footage", "subject") else (roots(r) if r in by_out else None)
        return None
    sandwiches = [o for o in ops if o["op"] == "sandwich" and not o["args"].get("invert") and o["args"].get("type_opacity") != 0.0]
    assert sandwiches
    for o in sandwiches:
        fg = o["args"]["fg"][1:]
        assert fg.endswith(tuple(f"__s{i}" for i in range(200))) and roots(fg) == "subject", o
    for o in ops:
        if o["op"] == "knockout" and "fg_matte" in o["args"]:
            assert roots(o["args"]["fg"][1:]) == "subject"
        if o["op"] == "upscale_matte":
            assert o["args"]["guide"] == "$subject"
    # the wipe's brighten blend re-inserts the subject into the graded plate instead of pasting the background's own subject pixels
    cf = [o for o in ops if o["op"] == "crossfade" and str(o["args"].get("b", "")).endswith(tuple(f"__L{i}" for i in range(200)))]
    assert cf, "the wipe crossfade blends a subject-corrected plate"
    # everything that is not a subject-taking op is untouched: same op sequence, only added clones and fg / guide / b arguments
    names = [o["op"] for o in base["ops"]]
    kept = [o["op"] for o in ops if "__s" not in str(o.get("out", "")) and "__L" not in str(o.get("out", ""))]
    assert kept == names


def test_chained_inverted_sandwich_keeps_the_background_over_the_type():
    """invert=True: the region the matte marks stays UNDER the type and the background is drawn over it, so fg stays the BACKGROUND plate."""
    ops = [{"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 1.0}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_a": "$type_matte", "type_rgb": [1, 1, 1], "matte": "$matte"}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$image", "fg": "$foot", "type_a": "$type_matte", "type_rgb": [1, 1, 1], "matte": "$matte",
                                                      "invert": True}}]
    out, rep = LR.layerize_ops(ops)
    sw = [o for o in out if o["op"] == "sandwich"]
    assert sw[0]["args"]["fg"].startswith("$foot__s") and sw[1]["args"]["fg"] == "$foot" and rep["changed"] == 1


def test_stateful_plate_chain_is_reported_not_guessed():
    ops = [{"op": "flow_wake", "out": "foot", "args": {"type_a": "$footage", "flow": "$flow", "matte": "$flow_m"}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_a": "$type_matte", "type_rgb": [1, 1, 1], "matte": "$matte"}}]
    out, rep = LR.layerize_ops(ops)
    assert out == ops and rep["changed"] == 0 and rep["skipped"] and "flow_wake" in rep["skipped"][0]


def test_depth_occlude_takes_covering_pixels_from_the_subject():
    C.set_backend("numpy")
    depth = np.zeros((H, Wd), np.float32)
    depth[20:60, 10:40] = 0.9                                          # the subject is near
    kw = dict(type_rgb=(1.0, 1.0, 1.0), type_a=TYPE, plane=0.5, softness=0.02)
    foot, subj = np.broadcast_to(BG, (H, Wd, 3)).copy(), np.broadcast_to(CLEAN, (H, Wd, 3)).copy()
    old = C.depth_occlude(foot, depth, **kw)
    new = C.depth_occlude(foot, depth, fg=subj, **kw)
    assert np.allclose(old[P_OVERLAP], BG, atol=1e-4) and np.allclose(new[P_OVERLAP], CLEAN, atol=1e-4)     # covered by the subject
    assert np.allclose(new[P_TYPE], 1.0, atol=1e-4) and np.allclose(new[P_BG], BG, atol=1e-4)
    ops = [{"op": "depth_occlude", "out": "image", "args": {"footage": "$footage", "footage_depth": "$s1_depth", "type_rgb": [1, 1, 1], "type_a": "$type_matte"}}]
    assert LR.layerize_ops(ops)[0][0]["args"]["fg"] == "$subject"


DEPTH_OPS = [{"op": "depth_occlude", "out": "image", "args": {"footage": "$footage", "footage_depth": "$depth", "type_rgb": [1.0, 1.0, 1.0], "type_a": "$type_matte",
                                                              "plane": 0.55, "softness": 0.02}}]


def test_depth_occlude_subject_matte_keeps_the_subject_in_front_even_where_depth_calls_it_far():
    """batch 3060 v007 r01: the depth map called the small climber FAR (cover 0), no Alpha-Gen matte existed: the type was drawn over him. With the matte the subject stays on top."""
    C.set_backend("numpy")
    depth = np.zeros((H, Wd), np.float32)                              # depth sees nothing near
    kw = dict(type_rgb=(1.0, 1.0, 1.0), type_a=TYPE, plane=0.5, softness=0.02)
    foot, subj = np.broadcast_to(BG, (H, Wd, 3)).copy(), np.broadcast_to(CLEAN, (H, Wd, 3)).copy()
    old = C.depth_occlude(foot, depth, fg=subj, **kw)
    new = C.depth_occlude(foot, depth, fg=subj, subject_matte=SUBJ, **kw)
    assert np.allclose(old[P_OVERLAP], 1.0, atol=1e-4)                 # type over the subject (the bug)
    assert np.allclose(new[P_OVERLAP], CLEAN, atol=1e-4) and np.allclose(new[P_SUBJ], CLEAN, atol=1e-4)      # the matte puts the clean subject on top
    assert np.allclose(new[P_TYPE], 1.0, atol=1e-4) and np.allclose(new[P_BG], BG, atol=1e-4)
    cover = np.full((H, Wd), 0.0, np.float32)                          # a guard-cleared cover (0) cannot take the subject out of the front either
    assert np.allclose(C.depth_occlude(foot, depth, fg=subj, subject_matte=SUBJ, cover=cover, **kw)[P_OVERLAP], CLEAN, atol=1e-4)


def test_layered_depth_occlude_shot_asks_for_the_matte_and_the_recipe_wires_it():
    p = _piece(DEPTH_OPS)
    assert RE.shot_needs(p, "s1")["matte"] is False and RE.shot_needs(p, "s1", layered=True)["matte"] is True      # single clip: unchanged; layered: Alpha-Gen is requested
    off = copy.deepcopy(p)
    off["layers"] = {"subject_matte": False}
    assert RE.shot_needs(off, "s1", layered=True)["matte"] is False
    with tempfile.TemporaryDirectory() as tmp:
        td = _type_dir(tmp)
        lay = RE.build_recipe(p, "s1", type_dir=td, layered=True)
        one = RE.build_recipe(p, "s1", type_dir=td)
    do = [o for o in lay["ops"] if o["op"] == "depth_occlude"][0]["args"]
    assert do["fg"] == "$subject" and do["subject_matte"] == "s1_matte".join(["$", ""]) and lay["inputs"]["s1_matte"]["src"] == "matte"
    assert "subject_matte" not in [o for o in one["ops"] if o["op"] == "depth_occlude"][0]["args"]


def test_window_cut_subject_comes_from_the_clean_layers():
    C.set_backend("numpy")
    red, cyan, blue, green = (np.broadcast_to(np.array(c, np.float32), (H, Wd, 3)).copy() for c in ((0.8, 0.2, 0.2), (0.2, 0.8, 0.9), (0.1, 0.1, 0.9), (0.1, 0.9, 0.1)))
    letters = TYPE
    kw = dict(mode="swap", exit="io", t_open=0.5, t_cut=2.0, t_end=3.0, matte_a=SUBJ, matte_b=np.zeros_like(SUBJ))
    plain = W.window_cut(letters, red, cyan, t=1.0, **kw)
    layered = W.window_cut(letters, red, cyan, t=1.0, a_subject=blue, b_subject=green, **kw)
    assert np.allclose(plain[P_SUBJ], red, atol=1e-4) and np.allclose(layered[P_SUBJ], blue, atol=1e-4)      # A is the plate: A's clean subject steps out
    after = W.window_cut(letters, red, cyan, t=2.5, mode="swap", exit="io", t_open=0.5, t_cut=2.0, t_end=3.0, matte_a=np.zeros_like(SUBJ), matte_b=SUBJ,
                         a_subject=blue, b_subject=green)
    assert np.allclose(after[P_SUBJ], green, atol=1e-4)                                                    # after the swap B is the plate: B's clean subject
    ops = [{"op": "window_cut", "out": "image", "args": {"type_a": "$wm", "a": "$footage", "b": "$next", "matte_a": "$matte", "matte_b": "$next_matte", "t": "$t"}}]
    a = LR.layerize_ops(ops)[0][0]["args"]
    assert a["a_subject"] == "$subject" and a["b_subject"] == "$next_subject"
    assert W.other_shot_ref("next_subject", "s1", ["s1", "s2"]) == ("s2", "subject")
    assert W.other_shot_ref("shot:s2:subject", "s1", ["s1", "s2"]) == ("s2", "subject")
    assert W.other_shot_ref("next_matte", "s1", ["s1", "s2"]) == ("s2", "matte")           # the old refs are unchanged
    got = W.collect_others({"shots": [{"id": "s1"}, {"id": "s2"}], "compose": {"default": {"ops": [
        {"op": "window_cut", "out": "image", "args": {"a": "$footage", "b": "$next", "matte_b": "$next_matte"}}]}}}, "s1",
        lambda s: "F" + s, lambda s: "M" + s, subject_of=lambda s: "S" + s)
    assert got == {"s2": {"footage": "Fs2", "matte": "Ms2", "subject": "Ss2"}}


# ============================================================================================ nodes (need a ComfyUI checkout)
def _comfy():
    for c in (os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if c and os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            try:
                import comfy_api.latest  # noqa: F401
                import folder_paths  # noqa: F401
            except Exception as e:   # noqa: BLE001
                pytest.skip(f"comfy_api import failed: {e}")
            return c
    pytest.skip("no ComfyUI checkout found (set MOGRAPH_COMFYUI_PATH)")


def _out(no):
    return no.args if hasattr(no, "args") else tuple(no)


def test_gather_node_builds_layers_and_cuts_references_from_the_clean_clip(plan):
    _comfy()
    torch = pytest.importorskip("torch")
    try:
        from nodes_fusion import layers as L
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"nodes_fusion.layers does not import here: {e}")
    piece = copy.deepcopy(plan)
    ids = list(piece["layout"])
    for s in piece["shots"]:
        s["control"] = {"pass": "depth", "guide": 0.4 if s["id"] != ids[1] else 0.0}
    h, w = 64, 36
    mk = lambda v, sid: {"u8": torch.full((9, h, w, 3), v, dtype=torch.uint8), "sid": sid, "start": int(piece["layout"][sid]["start"]), "fps": 24}   # noqa: E731
    clean = [mk(40, sid) for sid in ids]
    texts = {f"text_{sid[1:]}": mk(220, sid) for sid in ids if sid != ids[1]}
    # lazy gating: the text clips are requested only for shots with a text pass (guide > 0), never the guide-0 shot
    req = L.MographLayerGather.check_lazy_status(piece=piece, request={}, clean=clean, **{f"text_{k}": None for k in range(1, 7)})
    assert set(req) == {f"text_{sid[1:]}" for sid in ids if sid != ids[1]}
    assert f"text_{ids[1][1:]}" not in req
    out = _out(L.MographLayerGather.execute(piece, {}, None, clean, **texts))
    clips = out[0]
    by = {c["sid"]: c for c in clips}
    assert int(by[ids[0]]["u8"][0, 0, 0, 0]) == 220 and int(by[ids[0]]["subject"][0, 0, 0, 0]) == 40 and by[ids[0]]["textbg"] is True
    assert int(by[ids[1]]["u8"][0, 0, 0, 0]) == 40 and by[ids[1]]["textbg"] is False       # guide 0: the background IS the clean clip, no V2V pass
    need = RE.shot_needs(piece, ids[0])
    assert need["matte"]
    alpha = out[1 + int(ids[0][1:]) - 1]
    assert alpha.shape[0] == 9 and abs(float(alpha.mean()) - 40 / 255.0) < 0.01, "the Alpha-Gen reference is the CLEAN clip, not the text clip"
    # the barrier and the gate
    assert len(_out(L.MographLayerClean.execute(piece, **{f"clean_{c['sid'][1:]}": c for c in clean}))[0]) == len(clean)
    lat = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    assert _out(L.MographLayerGate.execute(lat, clean))[0] is lat
    assert _out(L.MographLayerPrompt.execute("a, glowing letters built into the scene as part of the world, b"))[0] == "a, b"


def test_gather_depth_refs_are_real_size_for_the_plan_piece_without_cuts(plan):
    """2026-10-30: the gather gets the PLAN piece (no compose.cuts: the design runs after the depth); the depth refs must be real (> 64 px), never the 16x16 placeholder."""
    _comfy()
    torch = pytest.importorskip("torch")
    try:
        from nodes_fusion import layers as L
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"nodes_fusion.layers does not import here: {e}")
    piece = copy.deepcopy(plan)
    piece.get("compose", {}).pop("cuts", None)
    piece["plan"] = dict(piece.get("plan") or {}, piece_system={"cuts": {"kinds": ["depth_pivot", "orbit:push"]}})
    ids = list(piece["layout"])
    assert len(ids) >= 2
    mk = lambda v, sid: {"u8": torch.full((9, 64, 36, 3), v, dtype=torch.uint8), "sid": sid, "start": int(piece["layout"][sid]["start"]), "fps": 24}   # noqa: E731
    clean = [mk(40, sid) for sid in ids]
    texts = {f"text_{sid[1:]}": mk(220, sid) for sid in ids if LS.wants_text(piece, sid, False)}
    out = _out(L.MographLayerGather.execute(piece, {}, None, clean, **texts))
    for sid in ids:
        d = out[1 + L.MAX_SHOTS + int(sid[1:]) - 1]
        assert min(d.shape[1:3]) >= 64, (sid, tuple(d.shape))


def test_compose_node_passes_the_subject_layer(plan):
    _comfy()
    import inspect
    try:
        from nodes_fusion import compose as NC
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"nodes_fusion.compose does not import here: {e}")
    src = inspect.getsource(NC.compose_clips)
    assert 'subject=clip.get("subject")' in src and "subject_of=" in src
