"""The open-model design step of the app graph (tools/fusion/design_step.py, nodes_fusion/design.py, graph fusion_design_qwen).
No model, no GPU: the model's reply is faked. The node tests need a ComfyUI checkout (skip otherwise).
Run: python -m pytest tests/test_design_nodes.py -q"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from test_fusion_core import synth_music  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

REQ = {"words": "STILL\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING", "scene": "a lone runner in a concrete tunnel", "music": "kinetic and emotional, 160 bpm",
       "length": 12, "look": "auto", "seed": 6, "quality": "preview", "format": "9:16"}


@pytest.fixture(scope="module")
def music():
    return synth_music(160, 12)


@pytest.fixture(scope="module")
def plan(music):
    return pr.plan_piece(REQ, music)


@pytest.fixture(scope="module")
def reply(plan, music):
    """what a model would answer if it agreed with the rules: valid by construction."""
    return DS.reply_from_piece(plan, music)


def _clips(piece, n=12):
    """fake generated footage: a few frames per shot, the same shape as MographFootageFlow's clips (uint8 [T,H,W,3])."""
    rng = np.random.default_rng(1)
    h, w = piece["sizes"]["compose"][1], piece["sizes"]["compose"][0]
    lay = pr.layout(piece)
    return [{"u8": rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8), "sid": s["id"], "start": lay[s["id"]]["start"], "fps": 24} for s in piece["shots"]]


def _mattes(piece, n=12):
    return {s["id"]: np.repeat((np.arange(n)[:, None, None, None] % 2 * 0.5 + 0.25) * np.ones((n, 64, 36, 1), np.float32), 3, axis=3) for s in piece["shots"]}


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


def _assert_valid(piece_json: str, plan: dict, music: dict):
    """a piece the rest of the graph can run: JSON, locked fields intact, liked fonts, the type engine accepts it, compose reads only what exists."""
    from mograph import fonts_liked as fl
    p = json.loads(piece_json)
    for k in ("audio", "fps", "stills", "shots", "type_passes", "sizes", "quality", "format"):
        assert p[k] == plan[k], k
    assert [w["text"] for w in p["type"]["words"]] == [w["text"] for w in plan["type"]["words"]]
    assert p["type"]["seed"] == plan["type"]["seed"]
    for k in ("size", "crossfade", "crf"):
        assert p["compose"][k] == plan["compose"][k], k
    assert p["signoff"]["off"] is True
    for w in p["type"]["words"]:
        fl.require_liked(w["font"])
    q = copy.deepcopy(p)
    pr.fill_type_timing(q, music)
    chk = pr.dry_run(q)
    assert len(chk["words"]) == len(p["type"]["words"])
    RE.need_passes(p)
    have = DS._piece_needs(plan)
    for sid, need in DS._piece_needs(p).items():
        for kind in ("matte", "depth"):
            assert not need[kind] or have[sid][kind], (sid, kind)
    assert any(o["op"] == "flow_wake" for o in p["compose"]["default"]["ops"])
    return p


# ------------------------------------------------------------------------------------------------ package
def test_package_builds_from_a_fixture(plan, music):
    pk = DS.package(REQ, music, plan, _clips(plan), _mattes(plan), "DESIGN_GUIDE_open_model", 6)
    pr_ = pk["prompt"]
    guide = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    assert guide.strip()[:200] in pr_                                                     # the guide, pasted
    for w in plan["type"]["words"]:
        assert w["text"] in pr_                                                           # the (planned) words
    assert f"tempo {music['bpm']:.1f} bpm" in pr_ and "downbeats (s)" in pr_              # music analysis as text
    assert plan["shots"][0]["prompt"][:40] in pr_                                         # shot list
    assert "9:16" in pr_ and "360x640" in pr_                                             # frame of the PREVIEW tier
    assert "One image is attached" in pr_ and "matte sheet" in pr_
    n = len(plan["shots"])
    assert len(pk["images"]) == n + 1 and len({i.shape for i in pk["images"]}) == 1       # contact sheets + matte sheet, one batch size
    assert pk["sheet"].ndim == 3 and pk["sheet"].shape[0] >= pk["images"][0].shape[0]
    locked = json.loads(pk["locked"])
    assert locked["words"] == [w["text"] for w in plan["type"]["words"]] and locked["shots"] == plan["shots"]
    assert locked["audio"] == plan["audio"] and locked["compose_locked"]["crossfade"] == plan["compose"]["crossfade"]
    assert pk["techniques"][0] == plan["technique"]
    # the allowed techniques follow what was generated: a `behind` plan has mattes, so `behind` stays on the list
    assert "behind" in pk["techniques"] and "built_in" not in pk["techniques"] or plan["technique"] != "behind"
    # no matte sheet when the plan made no mattes
    pk2 = DS.package(REQ, music, plan, _clips(plan), None, None, 6)
    assert len(pk2["images"]) == n and "matte sheet" not in pk2["prompt"]


def test_techniques_follow_the_footage(music):
    built = pr.plan_piece(dict(REQ, scene=""), music, dry=False)           # no subject word in the scene: built_in = depth only, no matte
    if built["technique"] != "built_in":
        pytest.skip("planner did not pick built_in for this scene")
    allowed = DS.allowed_techniques(built, music)
    assert allowed[0] == "built_in" and "behind" not in allowed and "behind_sheer" not in allowed


def test_package_node(plan, music):
    _comfy()
    import torch
    from nodes_fusion import design as D
    clips = [dict(c, u8=torch.from_numpy(c["u8"])) for c in _clips(plan)]
    imgs = {f"matte_{k + 1}": torch.from_numpy(v) for k, v in enumerate(_mattes(plan).values())}
    out = D.MographDesignPackage.execute(REQ, music, plan, clips, "DESIGN_GUIDE_open_model", 6, **imgs).args
    prompt, images, locked, sheet = out
    assert isinstance(prompt, str) and images.dtype == torch.float32 and images.shape[0] == len(plan["shots"]) + 1 and images.shape[-1] == 3
    assert sheet.shape[0] == 1 and 0.0 <= float(sheet.min()) and float(sheet.max()) <= 1.0
    assert json.loads(locked)["words"]
    s = D.MographDesignPackage.define_schema()
    assert [o.display_name for o in s.outputs][:3] == ["prompt", "images", "locked"]
    # matte / depth are requested before the model loads, only for what the plan's recipe reads and only when linked
    need = D.MographDesignPackage.check_lazy_status(REQ, music, plan, clips, "DESIGN_GUIDE_open_model", 6)
    want = {f"{kind}_{sid[1:]}" for sid in RE.piece_layout(plan)[0] for kind, v in RE.shot_needs(plan, sid).items() if v}
    assert set(need) == want and need


# ------------------------------------------------------------------------------------------------ parse
def test_parse_valid_reply_is_ok(plan, music, reply):
    r = reply
    r["compose"] = dict(r["compose"], plate_gain=0.9, grain=0.03)
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "PROMPT")
    assert out["ok"] is True and out["repair_prompt"] == ""
    p = _assert_valid(out["piece_json"], plan, music)
    assert "type = model, compose = model" in out["report"]
    assert any(o.get("args", {}).get("gain") == 0.9 for o in p["compose"]["default"]["ops"])      # the model's compose values are in
    assert p["design"]["src"] == {"type": "model", "compose": "model"}


def test_parse_reply_in_a_fence_with_prose_is_ok(plan, music, reply):
    text = "Here is my design:\n```json\n" + json.dumps(reply) + "\n```\nHope it helps."
    assert DS.parse(text, json.dumps(DS.make_locked(plan)), plan, music, "P")["ok"] is True


def test_parse_broken_reply_gives_ok_false_and_a_repair_prompt(plan, music):
    out = DS.parse("I cannot do that, sorry.", json.dumps(DS.make_locked(plan)), plan, music, "THE DESIGN PROMPT")
    assert out["ok"] is False
    assert out["repair_prompt"].startswith("THE DESIGN PROMPT") and "no parseable JSON" in out["repair_prompt"] and "I cannot do that" in out["repair_prompt"]
    _assert_valid(out["piece_json"], plan, music)                                    # piece_json is valid anyway (the rules' plan)
    assert json.loads(out["piece_json"])["type"]["words"] == json.loads(DS.parse("x", "", plan, music)["piece_json"])["type"]["words"]


def test_parse_bad_font_keeps_the_rest_of_the_model_type(plan, music, reply):
    bad = copy.deepcopy(reply)
    bad["type"]["words"][0]["font"] = "Comic Sans.ttf"
    bad["type"]["mood"] = "tender"
    out = DS.parse(json.dumps(bad), json.dumps(DS.make_locked(plan)), plan, music, "P")
    assert out["ok"] is False and "not in the liked font list" in out["repair_prompt"]
    p = _assert_valid(out["piece_json"], plan, music)
    assert p["type"]["mood"] == "tender" and p["design"]["src"]["type"].startswith("model fields")


def test_reply_missing_type_falls_back_to_the_rules(plan, music, reply):
    """Qwen3.5-9B twice returned no `type` object at all."""
    r = copy.deepcopy(reply)
    r.pop("type")
    r["compose"]["vignette"] = 0.31
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")
    assert out["ok"] is False and "missing object `type`" in out["repair_prompt"]
    p = _assert_valid(out["piece_json"], plan, music)
    assert p["type"]["words"] == plan["type"]["words"] and p["type"]["palette"] == plan["type"]["palette"]       # the rules' type
    assert p["design"]["src"] == {"type": "rules", "compose": "model"}                                            # the model's compose survived
    assert any(o.get("args", {}).get("strength") == 0.31 for o in p["compose"]["default"]["ops"])
    # ... and the repair step keeps a valid piece when the second reply is no better either
    for second in (None, "", "nonsense", json.dumps({"concept": "x"}), json.dumps({"compose": "oops", "type": [1, 2]})):
        fixed = DS.repair(out["piece_json"], out["ok"], second, plan, music, json.dumps(DS.make_locked(plan)))
        _assert_valid(fixed["piece_json"], plan, music)


def test_empty_object_and_wrong_shapes_fall_back_completely(plan, music):
    lk = json.dumps(DS.make_locked(plan))
    for text in ("{}", '{"type": {}, "compose": {}}', '{"type": "driving", "compose": 3}', "[]", '{"type": {"words": "STILL"}}'):
        out = DS.parse(text, lk, plan, music, "P")
        assert out["ok"] is False, text
        _assert_valid(out["piece_json"], plan, music)


def test_repair_reply_fixes_what_was_wrong(plan, music, reply):
    lk = json.dumps(DS.make_locked(plan))
    first = DS.parse("nope", lk, plan, music, "P")
    assert first["ok"] is False
    fixed = DS.repair(first["piece_json"], first["ok"], json.dumps(reply), plan, music, lk)
    p = _assert_valid(fixed["piece_json"], plan, music)
    assert p["design"]["src"] == {"type": "model", "compose": "model"} and "repair: type = model" in fixed["report"]


def _with_contour_morph(reply):
    """the model names contour_morph for the gaps (type.transitions) and for one word (word.transition)."""
    r = copy.deepcopy(reply)
    n = len(r["type"]["words"])
    r["type"]["transitions"] = ["contour_morph"] * (n - 1)
    r["type"]["words"][1]["transition"] = "contour_morph"
    return r


def test_parse_accepts_a_reply_naming_contour_morph(plan, music, reply):
    """regression (GPU render 2026-11-01): the type engine rejected 'contour_morph' (KeyError unknown module) before design_step._morph_names_to_null ran."""
    lk = json.dumps(DS.make_locked(plan))
    out = DS.parse(json.dumps(_with_contour_morph(reply)), lk, plan, music, "P")
    assert "KeyError" not in out["report"] and "unknown module" not in out["report"]
    assert out["ok"] is True and "type = model" in out["report"]
    p = _assert_valid(out["piece_json"], plan, music)
    assert p["design"]["src"]["type"] == "model"
    assert p["plan"]["contour_morph_asked"] == list(range(len(p["type"]["words"]) - 1))
    assert "contour_morph" not in json.dumps(p["type"])                           # the engine sees null gaps, the morph plays from plan.contour_morph_asked


def test_repair_accepts_a_reply_naming_contour_morph(plan, music, reply):
    lk = json.dumps(DS.make_locked(plan))
    first = DS.parse("nope", lk, plan, music, "P")
    assert first["ok"] is False
    fixed = DS.repair(first["piece_json"], first["ok"], json.dumps(_with_contour_morph(reply)), plan, music, lk)
    assert "KeyError" not in fixed["report"] and "unknown module" not in fixed["report"]
    p = _assert_valid(fixed["piece_json"], plan, music)
    assert p["design"]["src"]["type"] == "model" and p["plan"]["contour_morph_asked"]


def test_repair_passes_a_good_piece_through(plan, music, reply):
    lk = json.dumps(DS.make_locked(plan))
    first = DS.parse(json.dumps(reply), lk, plan, music, "P")
    assert first["ok"]
    same = DS.repair(first["piece_json"], True, None, plan, music, lk)
    assert same["piece_json"] == first["piece_json"] and "not needed" in same["report"]


def test_repair_survives_garbage_piece_json(plan, music):
    out = DS.repair("not json at all", False, None, plan, music)
    _assert_valid(out["piece_json"], plan, music)


# ------------------------------------------------------------------------------------------------ the lazy input
def test_repair_reply_is_lazy_and_only_requested_when_ok_is_false(plan, music):
    _comfy()
    from nodes_fusion import design as D
    R = D.MographDesignRepair
    s = R.define_schema()
    lazy = {i.id: getattr(i, "lazy", None) for i in s.inputs}
    assert lazy["repair_reply"] is True and not any(v for k, v in lazy.items() if k != "repair_reply")
    assert R.check_lazy_status(piece_json="{}", ok=True, repair_reply=None, piece=plan) == []
    assert R.check_lazy_status(piece_json="{}", ok=False, repair_reply=None, piece=plan) == ["repair_reply"]
    assert R.check_lazy_status(piece_json="{}", ok=False, repair_reply="{...}", piece=plan) == []
    # execute with ok = true never touches the (unevaluated, None) reply
    first = DS.parse("nope", "", plan, music)
    good = DS.parse(json.dumps(DS.reply_from_piece(plan, music)), "", plan, music)
    out = R.execute(good["piece_json"], True, plan, None, music, "").args
    assert out[0] == good["piece_json"]
    out = R.execute(first["piece_json"], False, plan, None, music, "").args           # false and no reply arrived: still a valid piece
    _assert_valid(out[0], plan, music)


def test_parse_and_repair_nodes_return_the_documented_outputs(plan, music, reply):
    _comfy()
    from nodes_fusion import design as D
    lk = json.dumps(DS.make_locked(plan))
    a = D.MographDesignParse.execute(json.dumps(reply), lk, plan, music, "P").args
    assert a[1] is True and a[2] == "" and isinstance(a[0], str) and "parse:" in a[3]
    b = D.MographDesignParse.execute("junk", lk, plan, music, "P").args
    assert b[1] is False and b[2].startswith("P")
    assert [o.display_name for o in D.MographDesignParse.define_schema().outputs][:3] == ["piece_json", "ok", "repair_prompt"]
    assert [o.display_name for o in D.MographDesignRepair.define_schema().outputs][0] == "piece_json"
    import nodes_fusion
    assert "design" in nodes_fusion.LOADED and {n.define_schema().node_id for n in D.NODES} <= {"MographDesignPackage", "MographDesignParse", "MographDesignRepair"}


# ------------------------------------------------------------------------------------------------ the graph
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bg():
    return _load("build_graphs_design_t", "tools/app/build_graphs.py")


@pytest.fixture(scope="module")
def design_graph(bg):
    return bg.make(False, None, "moss", True)


def test_design_graph_passes_the_checker(bg, design_graph):
    g, side = design_graph
    problems, _ = bg.check(g, side)
    assert problems == [], problems
    assert bg.check_reachability(g) == []


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion_design_qwen.json").exists(), reason="dev fixture not shipped in the public export: app/graphs/fusion_design_qwen.json")
def test_design_graph_on_disk_is_current(bg, design_graph):
    g, side = design_graph
    assert (ROOT / "app" / "graphs" / "fusion_design_qwen.json").read_text(encoding="utf-8") == bg.dump(g)
    assert (ROOT / "app" / "graphs" / "fusion_design_qwen.app.json").read_text(encoding="utf-8") == bg.dump(side)


def test_design_graph_chain_and_models(design_graph):
    g, side = design_graph
    cls = {n["class_type"] for n in g.values()}
    # NO CLAUDE / no paid nodes anywhere
    assert not [c for c in cls if any(x in c.lower() for x in ("claude", "anthropic", "openai", "gemini", "kling", "seedream", "api"))]
    assert not any("brief" in k for k in g)                                     # the design replaces the brief: no second Qwen call before the footage
    qw = {k: v for k, v in g.items() if v["class_type"] == "DesignLLM_Advanced"}
    assert set(qw) == {"design_qwen", "design_qwen2"}
    for v in qw.values():
        i = v["inputs"]
        assert i["model_name"] == "design_llm_8b" and i["quantization"].startswith("4-bit") and i["keep_model_loaded"] is False
    assert g["design_qwen"]["inputs"]["image"] == ["design_pkg", 3] and "image" not in g["design_qwen2"]["inputs"]
    # the chain, in order
    assert g["design_pkg"]["inputs"]["clips"] == ["flow", 0] and g["flow"]["inputs"]["clips"] == ["gather", 0]
    assert g["design_qwen"]["inputs"]["custom_prompt"] == ["design_pkg", 0]
    assert g["design_parse"]["inputs"]["reply"] == ["design_qwen", 0]
    assert g["design_qwen2"]["inputs"]["custom_prompt"] == ["design_parse", 2]
    assert g["design_fix"]["inputs"]["repair_reply"] == ["design_qwen2", 0] and g["design_fix"]["inputs"]["ok"] == ["design_parse", 1]
    assert g["design_piece"]["inputs"]["piece_json"] == ["design_fix", 0]
    assert g["type_final"]["inputs"]["piece"] == ["design_piece", 0]
    assert g["compose"]["inputs"]["piece"] == ["design_piece", 0] and g["compose"]["inputs"]["type"] == ["type_final", 0]
    assert g["assemble"]["inputs"]["piece"] == ["design_piece", 0] and g["out_final"]["inputs"]["video"] == ["assemble", 0]
    assert g["out_final"]["_meta"]["title"] == "OUT final" and g["out_final"]["class_type"] == "SaveVideo"
    # the repair model has exactly one consumer: the lazy input (so it never runs when the first reply passed)
    users = [(k, n) for k, v in g.items() for n, x in v["inputs"].items() if x == ["design_qwen2", 0]]
    assert users == [("design_fix", "repair_reply")]
    # the first type passes (LTX control video) are the rules' plan; PreviewAny never reads the repair model
    assert g["type"]["inputs"]["piece"] == ["plan", 0]
    assert not any(v["class_type"] == "PreviewAny" and v["inputs"]["source"][0] == "design_qwen2" for v in g.values())


def test_design_graph_is_single_stage_preview(design_graph):
    g, side = design_graph
    assert g["quality"]["inputs"]["value"] == "preview"
    stage2 = {"9", "55", "56"} | {f"c{k}.{i}" for k in range(1, 7) for i in (50, 51, 52, 53, 54, 57, 58)}
    assert not (stage2 & set(g))                                                 # no stage-2 nodes
    assert not any(v["class_type"] == "LatentUpscaleModelLoader" for v in g.values())
    assert g["c1.dec"]["inputs"]["samples"] == ["c1.41", 2]
    # one sampler pass per shot: SamplerCustomAdvanced c<k>.39 only
    samplers = [k for k, v in g.items() if v["class_type"] == "SamplerCustomAdvanced" and k.startswith("c")]
    assert sorted(samplers) == sorted(f"c{k}.39" for k in range(1, 7))
    assert g["10"]["inputs"]["value"] == ["sizes", 0] and g["27"]["inputs"]["value"] == ["sizes", 0]      # first frame fitted to the single stage
    assert [f for f in side["fields"] if f["id"] == "quality"][0]["default"] == "still"       # the app shows Preview (still) and Final; the graph literal above stays the internal "preview" tier


def test_design_graph_request_is_p06s(design_graph):
    g, _ = design_graph
    p06 = json.loads((ROOT / "15_fusion" / "pieces" / "P06_v003.json").read_text(encoding="utf-8")) if (ROOT / "15_fusion" / "pieces" / "P06_v003.json").is_file() else None
    if p06 is None:
        pytest.skip("P06_v003.json is gitignored and not on this machine")
    assert g["words"]["inputs"]["value"].split("\n") == [w["text"] for w in p06["type"]["words"]]
    assert g["scene"]["inputs"]["value"] == p06["stills"][0]["prompt"]
    assert str(p06["audio"]["bpm"]) + " bpm" in g["music"]["inputs"]["value"]


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion.json").exists(), reason="dev fixture not shipped in the public export: app/graphs/fusion.json")
def test_the_fusion_graph_is_untouched(bg):
    g, side = bg.make(False)
    assert (ROOT / "app" / "graphs" / "fusion.json").read_text(encoding="utf-8") == bg.dump(g)
    assert "MographDesignPackage" not in {n["class_type"] for n in g.values()}


# ------------------------------------------------------------------------------------------------ fonts: characterful, at least three faces
from mograph import fonts_liked as FL  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

PLAIN = "BigShouldersDisplay[wght].ttf"


def _fw(fonts, holds=None, wf=None):
    return [{"text": f"W{i}", "font": f, "hold_s": (holds or [2.0] * len(fonts))[i], "placement": {"wf": (wf or [0.8] * len(fonts))[i]}}
            for i, f in enumerate(fonts)]


def _roles(t):
    return [FL.role_of(w["font"]) for w in t["words"]]


def test_fonts_block_is_grouped_expressive_first_and_shuffled_by_seed():
    b1, b1b, b2 = QP.fonts_block(1), QP.fonts_block(1), QP.fonts_block(2)
    assert b1 == b1b and b1 != b2
    rows = [r for r in b1.split("\n") if r.startswith("- ")]
    assert len(rows) == len(FL.LIKED)                                           # every liked font, nothing else
    roles = [r.split("| role ")[1].split(" |")[0] for r in rows]
    order = [r for i, r in enumerate(roles) if i == 0 or r != roles[i - 1]]
    assert order == [r for r in QP.ROLE_ORDER if r in roles]                    # one block per role, pixel / ornamental / decon / blackletter first
    assert all("|" in r and r.split("| role ")[1].count("|") >= 1 for r in rows)  # a character note on every line
    assert {r.split(" | ")[0][2:] for r in rows} == {e["file"] for e in FL.LIKED}
    pix = lambda b: [r for r in b.split("\n") if "role pixel" in r]               # noqa: E731
    assert sorted(pix(b1)) == sorted(pix(b2)) and pix(b1) != pix(b2)               # same fonts, another order
    assert QP.fonts_block() == QP.fonts_block(None)                                 # no seed = file order, still grouped


def test_plain_fonts_everywhere_get_diversified():
    t = {"words": _fw([PLAIN, PLAIN, PLAIN], holds=[1.6, 3.0, 2.0])}
    fixes = []
    QP.diversify_fonts(t, fixes, 7)
    assert len({w["font"] for w in t["words"]}) >= 3 and len(set(_roles(t))) >= 2 and set(_roles(t)) & set(QP.EXPRESSIVE)
    assert FL.role_of(t["words"][1]["font"]) in QP.EXPRESSIVE                        # the hero (longest hold) is the expressive one
    assert t["words"][1]["font"] != PLAIN and len(fixes) >= 2 and all("(code fix)" in f for f in fixes)
    assert all(FL.is_liked(w["font"]) for w in t["words"])
    again = {"words": _fw([PLAIN, PLAIN, PLAIN], holds=[1.6, 3.0, 2.0])}
    QP.diversify_fonts(again, [], 7)
    assert again == t                                                                  # seeded: same seed, same picks
    other = {"words": _fw([PLAIN, PLAIN, PLAIN], holds=[1.6, 3.0, 2.0])}
    QP.diversify_fonts(other, [], 8)
    assert other != t


def test_only_the_hero_is_swapped_when_three_plain_faces_are_already_there():
    fonts = [PLAIN, "FacadeVF[wght,wdth].ttf", "SpaceGrotesk-Bold.ttf"]
    fonts = [FL.require_liked(f)["file"] for f in fonts]
    t = {"words": _fw(fonts, holds=[1.6, 1.6, 3.5])}
    fixes = []
    QP.diversify_fonts(t, fixes, 3)
    assert len(fixes) == 1 and "key word" in fixes[0]
    assert [w["font"] for w in t["words"][:2]] == fonts[:2] and FL.role_of(t["words"][2]["font"]) in QP.EXPRESSIVE


def test_a_varied_expressive_design_is_left_alone():
    fonts = [FL.require_liked(k)["file"] for k in ("Doto", "Jacquard24", "Anybody", "BungeeInline")]
    t = {"words": _fw(fonts)}
    fixes = []
    QP.diversify_fonts(t, fixes, 1)
    assert fixes == [] and [w["font"] for w in t["words"]] == fonts


def test_no_upper_limit_one_face_per_word_is_fine():
    fonts = [e["file"] for e in FL.LIKED if e["role"] in QP.EXPRESSIVE][:6]
    t = {"words": _fw(fonts)}
    fixes = []
    QP.diversify_fonts(t, fixes, 1)
    assert fixes == [] and len({w["font"] for w in t["words"]}) == 6


def test_fewer_than_three_words_need_one_face_per_word():
    t = {"words": _fw([PLAIN, PLAIN])}
    QP.diversify_fonts(t, [], 4)
    assert t["words"][0]["font"] != t["words"][1]["font"] and set(_roles(t)) & set(QP.EXPRESSIVE)
    one = {"words": _fw([PLAIN])}
    QP.diversify_fonts(one, [], 4)
    assert FL.role_of(one["words"][0]["font"]) in QP.EXPRESSIVE


def test_same_face_in_a_row_at_a_similar_scale_gets_a_contrasting_role():
    a, b, c = (FL.require_liked(k)["file"] for k in ("Doto", "Jacquard24", "Anybody"))
    t = {"words": _fw([a, b, b, c, a], holds=[2, 2, 2, 2, 2.5], wf=[0.8, 0.8, 0.8, 0.8, 0.8])}
    fixes = []
    QP.diversify_fonts(t, fixes, 5)
    assert len(fixes) == 1 and "twice in a row" in fixes[0]
    assert all(t["words"][i]["font"] != t["words"][i - 1]["font"] for i in range(1, 5))
    assert FL.contrast_axes(FL.role_of(t["words"][1]["font"]), FL.role_of(t["words"][2]["font"]))
    # a different scale is a contrast of its own
    t2 = {"words": _fw([a, b, b, c], wf=[0.8, 0.9, 0.4, 0.8])}
    f2 = []
    QP.diversify_fonts(t2, f2, 5)
    assert f2 == []


def test_swaps_never_pick_the_barcode_face_or_a_font_outside_liked_json():
    for seed in range(40):
        t = {"words": _fw([PLAIN] * 5)}
        QP.diversify_fonts(t, [], seed)
        assert all(FL.is_liked(w["font"]) and FL.require_liked(w["font"])["key"] not in QP.NOT_PICKED for w in t["words"])


def test_swapped_word_loses_axes_that_belonged_to_the_old_font():
    t = {"words": _fw([PLAIN] * 3)}
    t["words"][2]["axes"] = {"wght": 700}
    fixes = []
    QP.diversify_fonts(t, fixes, 2)
    assert "axes" not in t["words"][2] and any("axes dropped" in f for f in fixes)


def test_parse_applies_the_font_rules_and_logs_them(plan, music, reply):
    r = copy.deepcopy(reply)
    for w in r["type"]["words"]:
        w["font"] = PLAIN
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")
    assert out["ok"] is True                                                           # code fixes are not errors
    p = _assert_valid(out["piece_json"], plan, music)
    faces = [w["font"] for w in p["type"]["words"]]
    assert len(set(faces)) >= min(3, len(faces)) and set(_roles(p["type"])) & set(QP.EXPRESSIVE)
    assert "swapped to" in out["report"] and "(code fix)" in out["report"]


def test_package_prompt_lists_fonts_expressive_first_and_asks_for_three_faces(plan, music):
    p1 = DS.package(REQ, music, plan, _clips(plan), None, None, 1)["prompt"]
    p2 = DS.package(REQ, music, plan, _clips(plan), None, None, 2)["prompt"]
    i = p1.index("=== FONTS")
    sec = p1[i:p1.index("=== OUTPUT SCHEMA")]
    assert "at least three different faces" in sec and "role pixel" in sec.split("role heavy")[0] and sec.index("role pixel") < sec.index("role tall")
    assert sec != p2[p2.index("=== FONTS"):p2.index("=== OUTPUT SCHEMA")]                 # the design seed shuffles the list
    assert "at least three faces" in p1 or "at least three different faces" in p1


def test_guide_section_6_asks_for_character():
    g = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    s6 = g[g.index("## 6. Fonts"):g.index("## 7. Camera")]
    assert "at least three different faces" in s6 and "CHARACTERFUL" in s6 and "pixel, ornamental, decon or blackletter" in s6
    assert "Pick the main face for readability" not in g


# ------------------------------------------------------------------------------------------------ the 3060 / GPU settings of the graph
def test_vae_decode_tiles_follow_the_tier():
    _comfy()
    from nodes_fusion import footage as FT
    assert FT.decode_tiles({"quality": "preview"}) == (512, 64, 64, 8)
    assert FT.decode_tiles({"quality": "final"}) == (512, 64, 128, 32) and FT.decode_tiles({}) == (512, 64, 128, 32)


def test_design_graph_gpu_settings(design_graph):
    g, _ = design_graph
    assert g["flow"]["inputs"]["method"] == "auto" and g["flow"]["inputs"]["model"] == "raft_large"       # RAFT on CUDA, DIS only without CUDA
    assert g["compose"]["inputs"]["backend"] == "auto"                                                    # torch on CUDA
    assert g["assemble"]["inputs"]["encoder"] == "auto"                                                   # NVENC when ffmpeg + the driver have it
    assert g["type"]["inputs"]["procs"] == 10 and g["type_final"]["inputs"]["procs"] == 10               # clamped to os.cpu_count() by the node
    assert g["vda"]["inputs"]["model"] == "video_depth_anything_vits.pth"
    assert all(g[f"d{k}.proc"]["inputs"]["precision"] == "fp16" for k in range(1, 7))
    _, side = design_graph
    assert "--use-sage-attention" in side["server_flags"]["comfyui"] and "--mmap-torch-files" in side["server_flags"]["comfyui"]


# ------------------------------------------------------------------------------------------------ text as window (design_options + window_cut)
def test_prompt_and_schema_carry_the_window_choices(plan, music):
    p = DS.package(REQ, music, plan, _clips(plan), _mattes(plan), None, 3)["prompt"]
    assert "=== TEXT AS WINDOW ===" in p and "BOUNDARIES" in p and "after s1 (into s2)" in p
    assert '"window_source"' in p and '"cuts": [' in p and "window_swap" in p
    assert p.index("=== DESIGN GUIDE") < p.index("=== TEXT AS WINDOW") < p.index("=== THE PIECE")


def test_a_window_cut_reaches_the_piece_and_its_font_survives_the_font_rules(plan, music, reply):
    plan = pr.plan_piece(dict(REQ, words="RUN\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING"), music)       # a window word is never cut across lines (round-3 QC): only a short word reaches the window's cap height on one line
    plan["plan"]["cut_kinds"] = ["window_cut", "orbit:push"]               # the piece's drawn cut kinds (a model's kind outside the draw is replaced): window_cut is drawn here
    reply = DS.reply_from_piece(plan, music)
    r = copy.deepcopy(reply)
    r["compose"]["cuts"] = [{"after": "s1", "cut": "window_cut", "word": 0}]
    r["type"]["words"][0]["hold_s"] = 5.2      # words are timed to their shots now: a window word is the one that must hold ACROSS the cut (its hold/exit are not fitted to the shot)
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")
    assert out["ok"] is True
    p = _assert_valid(out["piece_json"], plan, music)
    assert p["compose"]["cuts"][0]["cut"] == "window_cut" and p["compose"]["cuts"][0]["after"] == "s1"
    from tools.fusion import design_options as DO
    from tools.fusion import window_cut
    assert DO.is_fat(p["type"]["words"][0]["font"], p["type"]["words"][0].get("axes"))[0]                 # the window word keeps a fat face
    faces = [w["font"] for w in p["type"]["words"]]
    assert len(set(faces)) >= 3 and set(_roles(p["type"])) & set(QP.EXPRESSIVE)                           # ... and the piece still has 3 faces + an expressive one
    assert window_cut.owned_boundaries(p)[0] == 0                                                          # (later place changes now carry planned 3D cuts, owned boundaries too)
    parts, cfl = RE.assemble_parts(p)
    assert cfl[0] == 0 or cfl[0] == [0] or not cfl[0]                                                      # the owned boundary is a hard cut, no crossfade
    assert window_cut.other_refs(p, "s1")                                                                  # s1's compose reads the next shot ($next ...)
    assert "s1" in p["compose"].get("shots", {})


def test_a_window_swap_with_no_word_across_the_cut_is_downgraded_and_logged(plan, music, reply):
    plan = copy.deepcopy(plan)
    plan["plan"]["cut_kinds"] = ["window_swap", "orbit:fly"]               # the piece's drawn cut kinds: window_swap is drawn, fails here, and a drawn 3D kind takes the place change
    r = copy.deepcopy(reply)
    r["compose"]["cuts"] = [{"after": "s2", "cut": "window_swap", "word": 0, "exit": "bogus"}]
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")
    p = _assert_valid(out["piece_json"], plan, music)
    from tools.fusion import design_options as DO
    assert p["compose"]["cuts"][0]["cut"] in ("hard",) + tuple(DO.SPECIAL) and p["compose"]["cuts"][0]["cut"] != "window_swap" and "downgraded to hard" in out["report"]      # a place change is never left plain: the 3D cut is planned there


def test_window_source_word_gets_a_fat_face_the_font_rules_leave_alone(plan, music, reply):
    r = copy.deepcopy(reply)
    r["type"]["words"][1]["window_source"] = "next"
    r["type"]["words"][1]["font"] = "FacadeVF[wght,wdth].ttf"
    out = DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")
    p = _assert_valid(out["piece_json"], plan, music)
    from tools.fusion import design_options as DO
    w = p["type"]["words"][1]
    assert w.get("window_source") in ("next", "s3") and DO.is_fat(w["font"], w.get("axes"))[0]
    assert "not a fat face" in out["report"]


def test_compose_requests_the_neighbour_matte_of_a_window_cut(plan, music, reply):
    _comfy()
    import torch
    from nodes_fusion import compose as NC
    r = copy.deepcopy(reply)
    r["compose"]["cuts"] = [{"after": "s1", "cut": "window_cut", "word": 0}]
    p = json.loads(DS.parse(json.dumps(r), json.dumps(DS.make_locked(plan)), plan, music, "P")["piece_json"])
    clips = [dict(c, u8=torch.from_numpy(c["u8"])) for c in _clips(p)[:2]]                                # only s1 and s2 are asked for
    from tools.fusion import window_cut
    need = NC.MographCompose.check_lazy_status(piece=p, type={}, clips=clips)
    refs_matte = {f"matte_{tgt[1:]}" for _r, tgt, kd in window_cut.other_refs(p, "s1") if kd == "matte"}
    assert refs_matte <= set(need)


def test_t2m_music_has_bare_second_chord_changes(plan, music):
    """3060 run: the t2m grid stores chord_changes as bare seconds (not {t, from, to}); the package / parse used to crash on `c["t"]`."""
    m = dict(music, chord_changes=[3.0, 6.0])
    pk = DS.package({"words": REQ["words"]}, m, plan, _clips(plan), _mattes(plan), None, 6)
    assert "chord" in pk["prompt"]
    out = DS.parse(json.dumps(DS.reply_from_piece(plan, m)), pk["locked"], plan, m, pk["prompt"])
    assert out["ok"] in (True, False) and out["piece_json"]
