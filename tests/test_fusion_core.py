"""Core pipeline of the app: footage flow in every plan (flow_wake), 9:16 / 16:9 sizes, no dead tail, optional Qwen nodes, graphs.
Run: python -m pytest tests/test_fusion_core.py -q       (CPU only; the node test needs a ComfyUI checkout, else it skips)"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.fusion import compose  # noqa: E402
from tools.fusion import footage_flow as FF  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

WORDS = "AFTER HOURS / THE CITY KEEPS DREAMING"
SCENES = {"built_in": "", "behind": "a lone figure standing in tall wheat", "traced": "a contour landscape of ridges"}


def synth_music(bpm: float, length: float, first: float = 0.2) -> dict:
    iv = 60.0 / bpm
    bar = 4 * iv
    beats = [first + i * iv for i in range(int(length / iv) + 40)]
    down = beats[::4]
    cut = min([d for d in down if d <= length + bar / 2], key=lambda d: abs(d - length))
    nf = math.ceil(cut * 24 - 1e-6)
    return dict(cut=round(cut, 4), nframes=nf, bpm=bpm, bar_len=bar, key="F minor", beats=beats, downbeats=down,
                chord_changes=[], curves={"onset": [0.15] * (nf + 1)}, music_end=cut + 1, fade=0.5, fps=24)


def _req(**kw):
    return dict({"words": WORDS, "scene": "", "music": "", "length": 12, "look": "auto", "seed": 5, "quality": "draft",
                 "format": "9:16"}, **kw)


@pytest.fixture(scope="module")
def plans():
    out = {}
    for tech, scene in SCENES.items():
        for fmt in ("9:16", "16:9"):
            out[(tech, fmt)] = pr.plan_piece(_req(scene=scene, format=fmt), synth_music(100, 12))
    return out


def _chains(piece):
    comp = piece["compose"]
    return [comp["default"]] + list((comp.get("shots") or {}).values())


# ----------------------------------------------------------------------------------------------- format
def test_request_format_and_sizes():
    assert pr.normalize_request({"words": "A"})["format"] == "9:16"
    assert pr.normalize_request({"words": "A", "format": "16:9"})["format"] == "16:9"
    assert pr.normalize_request({"words": "A", "format": "landscape"})["format"] == "16:9"
    with pytest.raises(ValueError):
        pr.normalize_request({"words": "A", "format": "1:1"})
    p = pr.sizes_for("final", "9:16")
    assert (p["control"], p["shot"], p["type"], p["compose"]) == ([576, 1024], [1152, 2048], [1088, 1920], [1080, 1920])
    l = pr.sizes_for("final", "16:9")
    assert (l["control"], l["shot"], l["type"], l["compose"]) == ([1024, 576], [2048, 1152], [1920, 1088], [1920, 1080])
    d = pr.sizes_for("draft", "16:9")
    assert (d["type"], d["compose"]) == ([1024, 576], [960, 540])
    assert d["control"] == d["shot"] == d["type"] == [1024, 576]                      # draft: one LTX stage, type aligned with the shot
    assert pr.sizes_for("draft", "9:16")["shot"] == [576, 1024]
    for fmt in ("9:16", "16:9"):
        for q in ("draft", "final"):
            s = pr.sizes_for(q, fmt)
            assert all(v % 64 == 0 for v in s["control"] + s["type"]) and all(v % (64 if q == "draft" else 128) == 0 for v in s["shot"]), s


def test_ltx_ic_plan_size_draft_is_single_stage():
    from tools.fusion import ltx_ic
    for kind in ("control", "motiontrack"):
        assert ltx_ic.plan_size(kind, (576, 1024), True) == ((576, 1024), False)
        assert ltx_ic.plan_size(kind, (1024, 576), True) == ((1024, 576), False)
        assert ltx_ic.plan_size(kind, (544, 960), True) == ((576, 1024), False)           # 544x960 breaks the ref-downscale rule
        assert ltx_ic.plan_size(kind, (1152, 2048), False) == ((1152, 2048), True)
        assert ltx_ic.plan_size(kind, (1088, 1920), False) == ((1152, 2048), True)        # 1088 is not a multiple of 128
        assert ltx_ic.plan_size(kind, (1920, 1088), False) == ((2048, 1152), True)
    g = ltx_ic.build("control", ref_video="r.mp4", size=(576, 1024), two_stage=False)    # no stage 2 nodes in a draft graph
    assert "53" not in g and "9" not in g and g["60"]["inputs"]["samples"] == ["41", 2]
    g2 = ltx_ic.build("control", ref_video="r.mp4", size=(1152, 2048), two_stage=True)
    assert "53" in g2 and g2["10"]["inputs"]["value"] == 576 and g2["27"]["inputs"]["value"] == 1152
    with pytest.raises(ValueError):
        ltx_ic.build("control", ref_video="r.mp4", size=(544, 960))


def test_ltx_graph_i2v_single_stage_draft():
    from tools import ltx_graph
    full = ltx_graph.build("i2v", "p", "a.png", seconds=12, width=1088, height=1920)
    one = ltx_graph.build("i2v", "p", "a.png", seconds=12, width=544, height=960, two_stage=False)
    assert "398:348" in full and "398:368" in full and full["398:374"]["inputs"]["samples"] == ["398:369", 0]
    for nid in ltx_graph.I2V_STAGE2:
        assert nid not in one, nid
    assert one["398:374"]["inputs"]["samples"] == ["398:367", 0] and one["398:358"]["inputs"]["samples"] == ["398:367", 1]
    assert one["398:353"]["inputs"]["expression"] == "a" and one["398:372"]["inputs"]["value"] == 544
    ids = set(one)
    assert all(v[0] in ids for n in one.values() for v in n["inputs"].values()
               if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str))


@pytest.mark.parametrize("tech", list(SCENES))
def test_plan_follows_the_format(plans, tech):
    p, l = plans[(tech, "9:16")], plans[(tech, "16:9")]
    assert p["format"] == "9:16" and l["format"] == "16:9"
    assert p["sizes"]["compose"] == [540, 960] and l["sizes"]["compose"] == [960, 540]
    assert l["stills"][0]["size"] == l["sizes"]["type"] and "horizontal" in l["stills"][0]["prompt"]
    assert "vertical" in p["stills"][0]["prompt"]
    assert all(s["size"] == [1024, 576] for s in l["shots"])                            # draft plan: single stage 1024x576
    assert pr.format_of(l) == "16:9" and pr.format_of({"sizes": {"compose": [1080, 1920]}}) == "9:16"
    # literal sizes in the recipe (behind: upscale_matte size) follow the compose size
    for ch in _chains(l):
        for op in ch["ops"]:
            if isinstance(op.get("args", {}).get("size"), list):
                assert op["args"]["size"] == l["sizes"]["compose"]
    # timing does not depend on the format: the same holds, the same words
    assert p["plan"]["flow_wake"]["holds"] == l["plan"]["flow_wake"]["holds"]


def test_landscape_scene_builds_and_fits():
    """the type scene builds at 1024x576 with no timeline notes, for modules of every family."""
    for look in ("lanes", "hollow_tubes", "beat_bars", "spectrum", "fly_into", "voxel_orbit", "particles"):
        piece = pr.plan_piece(_req(look=look, format="16:9"), synth_music(100, 12))
        chk = pr.dry_run(piece)
        assert chk["notes"] == [], (look, chk["notes"])
        assert chk["words"][0]["start"] >= 0.0


# ----------------------------------------------------------------------------------------------- flow in every recipe
@pytest.mark.parametrize("tech", list(SCENES))
def test_every_recipe_gets_flow_wake(plans, tech):
    piece = plans[(tech, "9:16")]
    assert piece["technique"] == tech
    assert piece["plan"]["flow_wake"]["chains"] >= 1
    known = set(compose._ops_registry())
    for ch in _chains(piece):
        ops = ch["ops"]
        names = [o["op"] for o in ops]
        assert "flow_wake" in names and set(names) <= known | {"trace_edges"}
        first_wake = names.index("flow_wake")
        for o in ops:
            if o["op"] == "flow_wake":
                a = o["args"]
                assert a["flow"] == "$flow" and a["matte"] == "$flow_m" and a["type_a"] in ("$type_matte", "$type_line")
                assert a["frame_idx"] == "$frame_idx" and a["u"] == "$u"
        outs = {o["out"] for o in ops if o["op"] == "flow_wake"}
        keys = [o["args"]["state_key"] for o in ops if o["op"] == "flow_wake"]
        assert len(keys) == len(set(keys))
        # nobody downstream reads the undisplaced alpha any more
        later = json.dumps([o["args"] for o in ops[first_wake + len(outs):]])
        for src in ("$type_matte", "$type_line"):
            if "$w" + src.split("_")[1][0] in later:
                assert f'"{src}"' not in later
        assert any(f'"${n}"' in later for n in outs)
    for s in piece["shots"]:
        assert s["flow"] is True
    rec = RE.build_recipe(piece, "s1", type_dir=str(ROOT / "nonexistent"), curves=synth_music(100, 12)["curves"])
    assert rec["inputs"]["flow"] == dict(src="flow", kind="flow", native=True)
    assert rec["inputs"]["flow_m"] == dict(src="flowmatte", kind="gray", native=True)
    assert RE.shot_needs_flow(piece, "s1")
    # still no alpha / depth demands of its own: the flow matte is derived from the flow
    assert RE.shot_needs(piece, "s1")["matte"] == (tech == "behind")


def test_settle_follows_the_word_holds(plans):
    piece = plans[("built_in", "9:16")]
    holds = piece["plan"]["flow_wake"]["holds"]
    assert len(holds) == len(piece["type"]["words"])
    op = next(o for o in _chains(piece)[0]["ops"] if o["op"] == "flow_wake")
    keys = op["args"]["settle"]["timeline"]
    assert all(b[0] > a[0] for a, b in zip(keys, keys[1:])), keys                     # strictly increasing
    st = op["args"]["settle"]
    for a, b in holds:                                                                  # settled inside every hold
        t = (a + b) / 2
        assert compose._timeline(st, t) == pytest.approx(pr.SETTLE_HOLD, abs=0.01)         # fully settled (2026-10-29)
    assert compose._timeline(st, 0.2) < 0.01 or holds[0][0] < 0.5
    assert pr.settle_timeline([(1.0, 1.05)])["timeline"] == [[0.0, 0.0]]              # too short a hold is ignored
    merged = pr.settle_timeline([(1.0, 2.0), (2.1, 3.0)])["timeline"]
    assert [k[1] for k in merged] == [0.0, 0.0, pr.SETTLE_HOLD, pr.SETTLE_HOLD, 0.0]
    assert pr.SETTLE_HOLD == 1.0


def test_weave_is_idempotent_and_keeps_hand_pieces_alone(plans):
    piece = copy.deepcopy(plans[("built_in", "9:16")])
    n = len(piece["compose"]["default"]["ops"])
    pr.weave_flow(piece, None, None)
    assert len(piece["compose"]["default"]["ops"]) == n
    p06 = ROOT / "15_fusion" / "pieces" / "P06.json"
    if p06.exists():
        hand = pr.finalize_piece(json.loads(p06.read_text(encoding="utf-8")))
        assert not any(o["op"] == "flow_wake" for ch in _chains(hand) for o in ch["ops"])


# ----------------------------------------------------------------------------------------------- tail / length
@pytest.mark.parametrize("bpm,length", [(100, 12), (122, 12), (80, 8), (120, 16), (90, 24), (140, 8)])
def test_no_dead_tail_and_exact_length(bpm, length):
    music = synth_music(bpm, length)
    piece = pr.plan_piece(_req(), music)
    chk = pr.dry_run(piece)
    cut = float(music["cut"])
    assert chk["notes"] == []
    assert chk["last_end"] <= cut - 0.3 + 1e-6
    assert cut - chk["last_end"] <= pr.TAIL_MAX_S + 1e-6, (bpm, length, cut, chk["last_end"])
    so = piece["signoff"]
    assert so["t0"] <= chk["last_end"] and cut - so["t0"] >= pr.SIGNOFF_MIN_S - 1e-6      # the type layer is never empty
    lay_end = max(v["start"] + v["frames"] for v in piece["layout"].values())
    assert 0 <= lay_end - music["nframes"] < 8                                           # footage covers the piece exactly (8k+1 shots)
    assert all(s["frames"] % 8 == 1 for s in piece["shots"])                              # LTX lengths (the layout holds the cut spans)


def test_nodry_plan_still_has_wake_and_signoff():
    music = synth_music(100, 12)
    piece = pr.plan_piece(_req(), music, dry=False)
    assert piece["plan"]["flow_wake"]["chains"] == 1
    assert piece["signoff"]["t0"] <= music["cut"] - pr.SIGNOFF_MIN_S + 1e-6


# ----------------------------------------------------------------------------------------------- flow + compose
def _shift_clip(n=10, h=96, w=64, dx=3):
    rng = np.random.default_rng(1)
    base = (rng.random((h, w * 2, 3)) * 255).astype(np.uint8)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    return np.stack([base[:, i * dx // 2: i * dx // 2 + w] for i in range(n)])


def test_flow_is_fp16_and_follows_the_motion():
    clip = _shift_clip()
    fl = FF.flow_for(clip, method="dis", verbose=False)
    assert fl.dtype == np.float16 and fl.shape == (clip.shape[0], clip.shape[1] // 2, clip.shape[2] // 2, 2)
    fx = float(np.median(fl[:-1, ..., 0].astype(np.float32)))
    assert fx < -0.2, fx                                       # content moves left (frame i+1 is shifted right in the source)
    assert abs(float(np.median(fl[:-1, ..., 1].astype(np.float32)))) < 0.3


def test_flow_matte_ignores_global_drift():
    fl = np.zeros((4, 40, 30, 2), np.float16)
    fl[..., 0] = 3.0                                           # camera pan: everything moves 3 px
    src = RE.FlowMatteSource(fl)
    assert float(src._decode(1).max()) == 0.0
    fl[:, 10:20, 8:16, 0] = 9.0                                # something moves 6 px faster than the pan
    m = RE.FlowMatteSource(fl)._decode(1)
    assert m[12, 10] > 0.95 and m[30, 25] == 0.0 and m.dtype == np.float32


def _fake_layers(tmp, n, size):
    w, h = size
    import cv2
    for name in ("matte", "line"):
        d = Path(tmp) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            im = np.zeros((h, w), np.uint8)
            im[h // 3: 2 * h // 3, w // 6: 5 * w // 6] = 255
            cv2.imwrite(str(d / f"{name}_{i:05d}.png"), im)
    return str(tmp)


@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
def test_compose_uses_the_flow_with_warmup(tmp_path, fmt):
    music = synth_music(100, 8)
    piece = pr.plan_piece(_req(format=fmt, length=8), music, dry=False)
    size = tuple(piece["sizes"]["compose"])
    n = 24
    piece = copy.deepcopy(piece)
    piece["layout"] = {"s1": {"start": 0, "frames": n, "xf": 0}}
    piece["shots"] = [dict(piece["shots"][0], frames=n)]
    w, h = size
    layers = _fake_layers(tmp_path, n, (w // 2, h // 2))
    rng = np.random.default_rng(3)
    foot = (rng.random((n, h // 2, w // 2, 3)) * 200).astype(np.uint8)
    flow = np.zeros((n, h // 4, w // 4, 2), np.float16)
    fh, fw = flow.shape[1:3]
    flow[:, fh // 4: 3 * fh // 4, fw // 8: 7 * fw // 8, 0] = 6.0       # something moves across the type
    depth = np.full((n, h // 2, w // 2), 0.3, np.float32)
    curves = {k: [0.2] * 400 for k in ("rms", "sub", "low", "mid", "high", "onset")}
    # the fake type layers sit under the wake influence; frames 12..24 rendered alone (warm-up) match the full run
    full, st = RE.run_compose(piece, "s1", type_dir=layers, footage=foot, depth=depth, flow=flow, curves=curves, cut=8.0,
                              backend="numpy")
    assert full.shape == (n, h, w, 3)
    for ch in _chains(piece):
        for o in ch["ops"]:
            if o["op"] == "flow_wake":
                o["args"]["gate"] = 0.0
    off, _ = RE.run_compose(piece, "s1", type_dir=layers, footage=foot, depth=depth, flow=flow, curves=curves, cut=8.0,
                            backend="numpy")
    diff = (full.float() - off.float()).abs().mean().item()
    assert diff > 0.05, diff                                    # the wake changes the picture
    with pytest.raises(ValueError):                             # a recipe that reads $flow without flow frames says so
        RE.run_compose(piece, "s1", type_dir=layers, footage=foot, depth=depth, curves=curves, cut=8.0, backend="numpy")


def _comfy():
    for c in (os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"),
              str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
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


def test_footage_flow_node_attaches_fp16_flow(plans):
    _comfy()
    import torch
    from nodes_fusion import footage as FT
    piece = plans[("built_in", "9:16")]
    u8 = torch.from_numpy(_shift_clip(n=8, h=128, w=64))
    clips = [{"u8": u8, "sid": "s1", "start": 0, "fps": 24}, {"u8": u8, "sid": "s2", "start": 90, "fps": 24}]
    out = FT.MographFootageFlow.execute(piece, clips, "dis", "raft_large").args[0]
    assert [c["sid"] for c in out] == ["s1", "s2"] and "flow" not in clips[0]            # input not mutated
    for c in out:
        assert c["u8"] is u8 and c["flow"].dtype == torch.float16 and tuple(c["flow"].shape) == (8, 64, 32, 2)
    # a piece whose recipe never reads the flow gets the clips back untouched
    plain = copy.deepcopy(piece)
    for ch in _chains(plain):
        ch["ops"] = [o for o in ch["ops"] if o["op"] != "flow_wake"]
    out2 = FT.MographFootageFlow.execute(plain, clips, "dis", "raft_large").args[0]
    assert all("flow" not in c for c in out2)
    # sizes follow the format
    land = pr.plan_piece(_req(format="16:9"), synth_music(100, 12), dry=False)
    assert FT.size_of(land, "control", (576, 1024), 2) == (512, 288)
    assert FT.size_of(plans[("built_in", "9:16")], "control", (576, 1024), 2) == (288, 512)
    assert FT.size_of({}, "compose", (1080, 1920)) == (1080, 1920)
    assert FT.size_of({"format": "16:9"}, "compose", (1080, 1920)) == (1920, 1080)


def test_gather_skips_refs_the_recipe_does_not_read(plans):
    """the still preview's gather (the only footage without a clip chain): float refs only for the chains the recipe reads, the rest a 16x16 placeholder."""
    _comfy()
    import torch
    from nodes_fusion import footage as FT
    piece = dict(plans[("built_in", "9:16")], still_preview=True)                         # depth only
    req = {"seed": 1, "quality": "draft"}
    first = torch.zeros((1, 64, 64, 3))
    out = FT.MographFootageGather.execute(piece, req, first=first).args
    a1, d1 = out[1], out[1 + FT.MAX_SHOTS]
    assert tuple(a1.shape) == (1, 16, 16, 3)                                              # no matte needed: placeholder
    assert d1.shape[0] == 1 and tuple(d1.shape[1:3]) == (960, 540)                        # depth: ONE frame of the shot's still
    assert out[0][0]["u8"].shape[0] == piece["layout"]["s1"]["frames"]
    land = dict(pr.plan_piece(_req(format="16:9"), synth_music(100, 12), dry=False), still_preview=True)
    out = FT.MographFootageGather.execute(land, req, first=first).args
    assert tuple(out[1 + FT.MAX_SHOTS].shape[1:3]) == (540, 960)
    assert out[0][0]["u8"].shape[1:3] == (540, 960)


# ----------------------------------------------------------------------------------------------- graphs
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bg():
    return _load("build_graphs_t", "tools/app/build_graphs.py")


@pytest.fixture(scope="module")
def built(bg):
    return {d: bg.make(d) for d in (False,)}


def test_graphs_have_flow_between_gather_and_compose(built):
    for draft, (g, side) in built.items():
        flow = [n for n in g.values() if n["class_type"] == "MographFootageFlow"]
        assert len(flow) == 1
        fid = next(i for i, n in g.items() if n["class_type"] == "MographFootageFlow")
        assert g[fid]["inputs"]["clips"][0] == "gather"
        comp = next(n for n in g.values() if n["class_type"] == "MographCompose")
        assert comp["inputs"]["clips"][0] == fid
        assert not any("masktrack" in n["class_type"].lower() or "motiontrack" in n["class_type"].lower() for n in g.values())


def test_format_input_and_sizes_are_linked(built):
    for draft, (g, side) in built.items():
        fmt = [i for i, n in g.items() if (n.get("_meta") or {}).get("title") == "IN format"]
        assert len(fmt) == 1 and g[fmt[0]]["inputs"]["value"] == "9:16"
        assert g["request"]["inputs"]["format"] == [fmt[0], 0]
        f = next(f for f in side["fields"] if f["id"] == "format")
        assert f["kind"] == "select" and f["default"] == "9:16" and [o["value"] for o in f["options"]] == ["9:16", "16:9"]
    g = built[False][0]
    sz = g["sizes"]["inputs"]["piece"]
    assert sz == ["plan", 0]
    for oid, o in (("10", 0), ("11", 1), ("21", 2), ("22", 3), ("27", 4), ("28", 5), ("a.10", 0), ("a.11", 1)):
        assert g[oid]["inputs"]["value"] == ["sizes", o], oid
    assert g["still_lat"]["inputs"]["width"] == ["sizes", 6] and g["still_lat"]["inputs"]["height"] == ["sizes", 7]


def test_moss_is_the_default_still(built, bg):
    g = built[False][0]
    assert g["still_unet"]["inputs"]["unet_name"] == "moss_image_sft_int8_convrot.safetensors"
    moss = json.loads((ROOT / "workflows" / "api" / "moss_t2i.json").read_text(encoding="utf-8"))
    assert g["still_clip"]["inputs"]["clip_name"] == moss["2"]["inputs"]["clip_name"]
    assert g["still_vae"]["inputs"]["vae_name"] == moss["3"]["inputs"]["vae_name"]


def test_video_depth_anything_is_small_only(built, bg):
    g = copy.deepcopy(built[False][0])
    assert bg.check_licences(g) == []
    vda = next(n for n in g.values() if n["class_type"] == "LoadVideoDepthAnythingModel")
    vda["inputs"]["model"] = "video_depth_anything_vitl.pth"
    assert bg.check_licences(g)


def test_qwen_is_optional_in_one_graph(built, bg):
    g, side = built[False]
    spec = side["optional_nodes"][0]
    assert spec["class_type"] == "DesignLLM_Advanced"
    classes = {n["class_type"] for n in g.values()}
    full, rep = bg.optional_nodes.prune_optional(g, side, classes)
    assert rep == [] and full == g
    pruned, rep = bg.optional_nodes.prune_optional(g, side, classes - {"DesignLLM_Advanced"})
    assert rep and set(rep[0]["removed"]) == set(spec["titles"])
    assert not any(n["class_type"] in ("DesignLLM_Advanced", "MographFusionBriefPrompt") for n in pruned.values())
    for n in pruned.values():
        for v in n["inputs"].values():
            assert not (isinstance(v, list) and len(v) == 2 and v[0] not in pruned), n
    assert "brief" not in pruned["plan"]["inputs"] and "brief" not in pruned["t2m_enc"]["inputs"]
    assert bg.check_reachability(pruned) == []
    assert g["plan"]["inputs"]["brief"] == ["qwen", 0]                    # the template itself keeps the Qwen chain
    assert bg.optional_nodes.prune_optional(g, side, None)[1] == []        # unknown server: nothing removed
    assert not (ROOT / "app" / "graphs" / "fusion_noqwen.json").exists()


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion.json").exists(), reason="dev fixture not shipped in the public export: app/graphs/fusion.json")
def test_route_prunes_by_installed_classes(built, bg):
    routes = _load("app_routes_t", "app_routes.py")
    g, side = built[False]
    classes = {n["class_type"] for n in g.values()}
    got = routes.load_graph("fusion", classes - {"DesignLLM_Advanced"})
    assert got and got["pruned"] and "qwen" not in got["graph"]
    got = routes.load_graph("fusion", classes)
    assert got["pruned"] == [] if "pruned" in got else True
    assert "qwen" in routes.load_graph("fusion", None)["graph"]


def test_music_is_exact_length_in_both_graphs(built):
    for draft, (g, _) in built.items():
        assert "cut" in g and "ace_audio" not in g["cut"]["inputs"]
        route = g["route"]["inputs"]
        assert route["user_audio"] == ["cut", 1] and route["user_music"] == ["cut", 0]
        # approved route: ACE XL base text-to-music -> exact-length cut + measured grid -> tame highs (music first, then the plan)
        assert route["score_audio"] == ["t2m_tame", 0] and route["score_music"] == ["t2m_cut", 1]
        assert g["master"]["inputs"]["audio"] == ["route", 0] and g["master"]["inputs"]["mono_safe"] is True
        for node in ("plan", "type", "compose"):
            assert g[node]["inputs"]["music"] == ["route", 1], node
    g = built[False][0]
    ks = g["t2m_ks"]["inputs"]
    assert ks["steps"] == 50 and ks["cfg"] == 6.0 and ks["sampler_name"] == "euler" and ks["scheduler"] == "simple" and ks["denoise"] == 1.0
    assert g["t2m_lat"]["inputs"]["seconds"] == ["t2m_enc", 1]                  # length + 4 s from the encode node
    assert g["t2m_unet"]["inputs"]["unet_name"] == "acestep_v1.5_xl_base_bf16.safetensors"
    assert g["t2m_ks"]["inputs"]["model"] == ["t2m_live", 0] and g["t2m_live"]["inputs"]["model"] == ["t2m_shift", 0] and g["t2m_shift"]["inputs"]["shift"] == 3       # live waveform patch between the shift and the sampler
    assert not any(f["id"] == "restyle" for f in built[False][1]["fields"])


def _upstream(g: dict, start: str) -> set:
    seen, stack = set(), [start]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack += [v[0] for v in g[n]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in g]
    return seen


def test_music_path_has_no_turbo_and_no_score_skeleton(built):
    """BUILT.md Music: skeleton + ACE audio-to-audio ("circus / tinker toy") and turbo are rejected; every ACE model in the graph is the XL base."""
    g = built[False][0]
    for start in ("master", "plan", "route", "assemble"):
        for nid in _upstream(g, start):
            n = g[nid]
            assert n["class_type"] not in ("MographScoreSkeleton", "MographMusicFit", "MographMusicStyleEncode"), (start, nid)
            assert "turbo" not in str(n["inputs"].get("unet_name", "")).lower(), (start, nid)
    assert not any(n["class_type"] == "MographScoreSkeleton" for n in g.values())
    ace = [n["inputs"]["unet_name"] for n in g.values() if n["class_type"] == "UNETLoader" and "ace" in n["inputs"]["unet_name"].lower()]
    assert ace == ["acestep_v1.5_xl_base_bf16.safetensors"]
    assert all("sft" not in a and "turbo" not in a for a in ace)
    # the music (and so its measured grid) exists before the plan: plan -> route -> T2M cut -> sampler -> encode
    assert {"t2m_cut", "t2m_ks", "t2m_enc", "t2m_tame"} <= _upstream(g, "plan") | _upstream(g, "master")
    assert "plan" not in _upstream(g, "t2m_cut")


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion.json").exists(), reason="dev fixture not shipped in the public export: app/graphs/fusion.json")
def test_graphs_on_disk_are_current(bg):
    for draft in (False,):
        g, side = bg.make(draft)
        name = "fusion"
        assert (ROOT / "app" / "graphs" / f"{name}.json").read_text(encoding="utf-8") == bg.dump(g)
        assert (ROOT / "app" / "graphs" / f"{name}.app.json").read_text(encoding="utf-8") == bg.dump(side)
        problems, _ = bg.check(g, side)
        assert problems == [], problems


# ----------------------------------------------------------------------------------------------- type framing at the holds, both formats
HOLD_SETS = {"short": WORDS, "long": "ELECTROENCEPHALOGRAPHY", "four": "ONE / TWO TWO / THREE THREE THREE / FOUR FOUR FOUR FOUR"}
SAFE = 0.05                                 # every glyph stays >= 5 % of the frame away from each edge while a word is held
HOLD_Q = (0.25, 0.5, 0.75)                  # sampled inside the hold (the hit itself still settles the camera orbit)
_HOLDS: dict = {}


def _alpha_at_holds(name: str, fmt: str) -> dict:
    """plan the words in `fmt`, build the real type scene (beat_orbit camera) at the layers size and rasterize the matte at the hold
    of every word -> {'size', 'piece', 'holds': [(text, lines, frame, alpha HxW float)]}. Cached per (set, format)."""
    key = (name, fmt)
    if key in _HOLDS:
        return _HOLDS[key]
    from mograph.render import cam3d
    from tools.fusion import raster
    from tools.fusion import spec as SP
    piece = pr.plan_piece(_req(format=fmt, words=HOLD_SETS[name]), synth_music(100, 12))
    size = tuple(piece["sizes"]["type"])
    world = (cam3d.W, cam3d.H)
    try:
        sc = SP.build_scene(copy.deepcopy(piece["type"]), size=size)
        holds = []
        for w in sc.info["words"]:
            for q in HOLD_Q:
                t = w["hit"] + q * (w["release"] - w["hit"])
                G = raster.rasterize(sc.seq.layers(t), sc.chain.at(t), size, S=1)
                holds.append((w["text"], w["lines"], int(round(t * sc.fps)), np.asarray(G.A, np.float32)))
    finally:
        cam3d.set_world(*world)
    _HOLDS[key] = dict(size=size, piece=piece, holds=holds)
    return _HOLDS[key]


def _bbox(a: np.ndarray, thr: float = 0.04):
    ys, xs = np.where(a > thr)
    assert len(xs), "empty type alpha at a hold"
    h, w = a.shape
    return xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h


@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
@pytest.mark.parametrize("name", list(HOLD_SETS))
def test_type_alpha_stays_inside_the_safe_area_at_every_hold(name, fmt):
    r = _alpha_at_holds(name, fmt)
    w, h = r["size"]
    assert (w > h) == (fmt == "16:9") and (r["piece"]["sizes"]["compose"][0] > r["piece"]["sizes"]["compose"][1]) == (w > h)
    for text, lines, f, a in r["holds"]:
        x0, x1, y0, y1 = _bbox(a)
        assert x0 >= SAFE and 1 - x1 >= SAFE and y0 >= SAFE and 1 - y1 >= SAFE, (name, fmt, text, lines, f, (x0, x1, y0, y1))
        if fmt == "16:9":                                       # laid out for a wide frame, not the portrait stack squeezed in
            one_word = len(str(text).split()) == 1                      # words are never split across lines (2026-10-29): a single long word is one line, its box is as wide as the frame allows
            assert len(lines) <= 3 and ((x1 - x0) * (y1 - y0) >= 0.18 or one_word), (name, text, lines, (x0, x1, y0, y1))
            assert x1 - x0 >= 0.4, (name, text, lines, x1 - x0)


@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
def test_composed_hold_frame_keeps_the_type_in_frame(tmp_path, fmt):
    """the composed frame (flat footage, the real recipe incl. flow_wake / glow / vignette): the pixels the type changes sit inside the
    safe area, and the type layers need no crop to reach the compose size."""
    r = _alpha_at_holds("short", fmt)
    piece = copy.deepcopy(r["piece"])
    size = tuple(RE.compose_size(piece))
    assert abs(r["size"][0] / r["size"][1] - size[0] / size[1]) < 0.01          # layers and compose share the aspect: fit_cover is a no-op crop
    text, lines, f, a = r["holds"][1]                                              # first word, middle of its hold
    assert f < piece["layout"]["s1"]["frames"]
    import cv2
    w, h = size
    T = int(piece["layout"]["s1"]["frames"])
    foot = np.full((T, h // 2, w // 2, 3), 60, np.uint8)
    depth = np.full((T, h // 2, w // 2), 0.3, np.uint8).astype(np.float32)
    flow = np.zeros((T, h // 4, w // 4, 2), np.float16)
    curves = {k: [0.2] * 400 for k in ("rms", "sub", "low", "mid", "high", "onset")}

    def layers(d, alpha):
        for p in ("matte", "line"):
            (d / p).mkdir(parents=True)
            cv2.imwrite(str(d / p / "x.png"), np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
            for i in range(f + 1):
                cv2.imwrite(str(d / p / f"{p}_{i:05d}.png"), np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
            (d / p / "x.png").unlink()
        return str(d)

    frames = []
    for tag, al in (("on", a), ("off", np.zeros_like(a))):
        d = layers(tmp_path / tag, al)
        out, _ = RE.run_compose(piece, "s1", type_dir=d, footage=foot, depth=depth, flow=flow, curves=curves, cut=12.0,
                                backend="numpy", frames=(f, f + 1))
        out = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
        assert out.shape[1:3] == (h, w)
        frames.append(out[0].astype(np.float32))
    diff = np.abs(frames[0] - frames[1]).max(axis=2)
    assert diff.max() > 40
    ys, xs = np.where(diff > 0.35 * diff.max())
    x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
    assert x0 >= SAFE and 1 - x1 >= SAFE and y0 >= SAFE and 1 - y1 >= SAFE, (fmt, text, lines, (x0, x1, y0, y1))
