"""STILL PREVIEW (app quality `still`, 16_app/STILL_PREVIEW.md) and the removal of the stand-in footage. CPU only; the node tests skip when no ComfyUI checkout is importable.
Run: python -m pytest tests/test_still_preview.py -q"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_plan import synth_music  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402


# ------------------------------------------------------------------------------------------------------------------------ ComfyUI (node tests)
@pytest.fixture(scope="module")
def nodes():
    for cand in (os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if cand and os.path.isfile(os.path.join(cand, "comfy_api", "latest", "__init__.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import comfy_api.latest  # noqa: F401
        import folder_paths  # noqa: F401
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"comfy_api not importable: {e}")
    import nodes_fusion.footage as footage
    import nodes_fusion.layers as layers
    import nodes_fusion.plan as plan
    return type("N", (), {"footage": footage, "layers": layers, "plan": plan})


# ------------------------------------------------------------------------------------------------------------------------ the quality value, sizes, the piece
def test_still_is_a_draft_size_run_with_a_flag():
    assert SP.split_quality("still") == ("draft", True) and SP.split_quality(" Still ") == ("draft", True)
    for q in ("preview", "draft", "final", ""):
        assert SP.split_quality(q)[1] is False
    assert SP.split_quality("final") == (SP.APP_FINAL_TIER, False) and SP.APP_FINAL_TIER == "draft", "the app's Final renders at the draft tier until the two-stage final has run as a whole piece"
    assert SP.split_quality("preview") == ("preview", False) and SP.split_quality("draft") == ("draft", False), "the internal tiers stay reachable (batch drivers, tests)"
    assert SP.label("r01_gemma_ltx") == "r01_gemma_ltx_still" and SP.label("r01_gemma_ltx_still") == "r01_gemma_ltx_still"


def test_the_apps_final_renders_at_the_draft_sizes_and_one_constant_switches_it(monkeypatch):
    q, flag = SP.split_quality("final")
    assert (q, flag) == ("draft", False) and PR.sizes_for(q)["compose"] == [540, 960] and PR.sizes_for(q)["shot"] == [576, 1024]
    monkeypatch.setattr(SP, "APP_FINAL_TIER", "final")
    assert SP.split_quality("final") == ("final", False) and PR.sizes_for(SP.split_quality("final")[0])["shot"] == [1152, 2048]
    with pytest.raises(ValueError, match="draft"):
        SP.upgraded_piece({"quality": "draft"})                    # the stills of a Preview are draft-size: no upgrade to a two-stage final


def test_the_still_preview_sizes_are_the_draft_sizes_so_a_draft_render_re_uses_the_stills():
    q, flag = SP.split_quality("still")
    req = PR.normalize_request({"words": "HELLO\nWORLD", "scene": "a quiet harbour at dawn", "music": "", "length": 12, "look": "auto", "seed": 3, "quality": q, "format": "9:16"})
    piece = SP.prepare_piece(PR.plan_piece(req, synth_music(100.0, 12.0)), {"words": "HELLO\nWORLD", "length": 12, "seed": 3, "format": "9:16"}, flag)
    assert piece["quality"] == "draft" and piece[SP.FLAG] is True
    s = piece["sizes"]
    assert (s["control"], s["shot"], s["type"], s["compose"]) == ([576, 1024], [576, 1024], [576, 1024], [540, 960]), s       # the still size = sizes.type (MographPieceSizes still_w / still_h)
    assert s == PR.sizes_for("draft")
    assert piece[SP.REQUEST_KEY] == {"words": "HELLO\nWORLD", "length": 12, "seed": 3, "format": "9:16"}
    assert all("render" not in w or w["render"] == "flat" for w in piece["type"]["words"]), "a draft draws every word flat: no LTX letters in a still preview"


def test_the_request_and_plan_nodes_carry_the_flag(nodes):
    req = nodes.plan.MographRequest.execute("HELLO\nWORLD", "", "", 12, "auto", 3, "still", format="9:16").args[0]
    assert req["quality"] == "draft" and req[SP.FLAG] is True and req["raw"]["words"] == "HELLO\nWORLD"
    piece = nodes.plan.MographPiecePlan.execute(req, synth_music(100.0, 12.0)).args[0]
    assert piece[SP.FLAG] is True and piece["quality"] == "draft" and piece[SP.REQUEST_KEY]["seed"] == 3
    sz = nodes.plan.MographPieceSizes.execute(piece).args
    assert (sz[6], sz[7]) == (576, 1024), "the stills are rendered at the draft still size even in the still preview"
    fin = nodes.plan.MographRequest.execute("HELLO\nWORLD", "", "", 12, "auto", 3, "final", format="9:16").args[0]
    assert fin["quality"] == SP.APP_FINAL_TIER and SP.FLAG not in fin, "the app's Final is the APP_FINAL_TIER tier, not a still preview"
    plain = nodes.plan.MographRequest.execute("HELLO\nWORLD", "", "", 12, "auto", 3, "draft", format="9:16").args[0]
    assert SP.FLAG not in plain and plain["quality"] == "draft"
    assert nodes.plan.MographPiecePlan.execute(plain, synth_music(100.0, 12.0)).args[0].get(SP.FLAG) is None


def test_the_flag_survives_the_design_step():
    from tools.fusion import design_step as DS
    q, flag = SP.split_quality("still")
    req = PR.normalize_request({"words": "BREAK\nCLAIM", "scene": "x", "music": "", "length": 12, "look": "auto", "seed": 5, "quality": q, "format": "9:16"})
    music = synth_music(100.0, 12.0)
    piece = SP.prepare_piece(PR.plan_piece(req, music), {"words": "BREAK\nCLAIM", "seed": 5}, flag)
    locked = DS.make_locked(piece)
    assert locked[SP.FLAG] is True and locked[SP.REQUEST_KEY] == {"words": "BREAK\nCLAIM", "seed": 5}
    out = DS.apply_locked(PR.plan_piece(req, music), locked)
    assert out[SP.FLAG] is True and out[SP.REQUEST_KEY]["seed"] == 5


# ------------------------------------------------------------------------------------------------------------------------ the held footage
def test_hold_u8_is_the_same_picture_on_every_frame_at_the_compose_size():
    img = np.linspace(0, 1, 64 * 48 * 3, dtype=np.float32).reshape(64, 48, 3)
    u8 = SP.hold_u8(img, 7, (54, 96))
    assert u8.shape == (7, 96, 54, 3) and u8.dtype == np.uint8
    assert all(np.array_equal(u8[0], u8[i]) for i in range(7))


def test_a_shot_takes_the_row_of_its_own_still():
    piece = {"stills": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "C"}, {"id": "s3", "first": "last:s2"}, {"id": "s4"}]}
    assert [SP.shot_row(piece, f"s{i}", 3) for i in (1, 2, 3, 4)] == [0, 2, 0, 0]
    assert SP.shot_row(piece, "s2", 1) == 0                      # a batch of one row is always row 0


def _piece3():
    return {"quality": "draft", SP.FLAG: True, "format": "9:16", "sizes": {"compose": [54, 96], "control": [64, 112]},
            "stills": [{"id": "A"}, {"id": "B"}],
            "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "B"}],
            "layout": {"s1": {"start": 0, "frames": 6, "xf": 0}, "s2": {"start": 6, "frames": 10, "xf": 0}}}


def _first_batch():
    import torch
    rows = np.zeros((2, 64, 48, 3), np.float32)
    rows[0, ..., 0] = 0.8                                          # still A: red
    rows[1, ..., 2] = 0.8                                          # still B: blue
    return torch.from_numpy(rows)


def test_still_outputs_hold_each_shots_own_still_and_cut_the_references_from_it(nodes, monkeypatch):
    FT = nodes.footage
    monkeypatch.setattr(FT.RE, "shot_needs", lambda piece, sid, layered=False: {"matte": sid == "s1", "depth": True})
    clips, alpha, depth = FT.still_outputs(_piece3(), _first_batch())
    assert [c["sid"] for c in clips] == ["s1", "s2"] and [c["u8"].shape[0] for c in clips] == [6, 10] and [c["start"] for c in clips] == [0, 6]
    assert all(c["still"] is True for c in clips)
    r1, r2 = clips[0]["u8"][0].numpy().astype(int), clips[1]["u8"][0].numpy().astype(int)
    assert r1[..., 0].mean() > 150 and r1[..., 2].mean() < 5 and r2[..., 2].mean() > 150 and r2[..., 0].mean() < 5, "shot 2 holds still B, not still A"
    assert (clips[0]["u8"] == clips[0]["u8"][0]).all()
    assert alpha[0].shape[0] == SP.ALPHA_FRAMES == 9 and tuple(alpha[0].shape[1:3]) == (112, 64) and float(alpha[0][..., 0].mean()) > 0.5
    assert tuple(alpha[1].shape) == (1, 16, 16, 3), "s2 reads no matte: placeholder, the Alpha-Gen chain of s2 never runs"
    assert depth[0].shape[0] == 1 and depth[1].shape[0] == 1, "depth is ONE frame of the still"
    assert tuple(alpha[2].shape) == (1, 16, 16, 3) and tuple(depth[5].shape) == (1, 16, 16, 3), "slots outside the layout stay placeholders"


def test_the_gather_nodes_ask_for_no_ltx_clip_in_a_still_preview(nodes):
    FT, LY = nodes.footage, nodes.layers
    piece = _piece3()
    clips, cleans = {"clip_1": None, "clip_2": None}, {"clean_1": None, "clean_2": None}
    texts = {"text_1": None, "text_2": None, "plate_1": None, "plate_2": None}
    assert FT.MographFootageGather.check_lazy_status(piece, {}, None, **clips) == []
    assert LY.MographLayerClean.check_lazy_status(piece, **cleans) == []
    assert LY.MographLayerGather.check_lazy_status(piece, {}, None, None, True, **texts) == []
    plain = dict(piece)
    plain.pop(SP.FLAG)
    assert set(FT.MographFootageGather.check_lazy_status(plain, {}, None, **clips)) == {"clip_1", "clip_2"}, "a draft / final piece still asks for its clips"
    assert LY.MographLayerClean.check_lazy_status(plain, **cleans) == ["clean_1", "clean_2"]
    assert {"text_1", "text_2"} <= set(LY.MographLayerGather.check_lazy_status(plain, {}, None, None, False, **texts))


def test_the_gather_nodes_hold_the_stills_instead_of_failing_or_faking_footage(nodes):
    FT, LY = nodes.footage, nodes.layers
    first = _first_batch()
    out = FT.MographFootageGather.execute(_piece3(), {}, first).args
    assert len(out[0]) == 2 and out[0][0]["still"] is True
    out = LY.MographLayerGather.execute(_piece3(), {}, first, None, False).args
    assert len(out[0]) == 2 and out[0][1]["still"] is True and "subject" not in out[0][0] and not out[0][0].get("textbg"), "no layers in a still preview: graphic type only"
    plain = _piece3()
    plain.pop(SP.FLAG)
    with pytest.raises(ValueError, match="no clip arrived"):
        FT.MographFootageGather.execute(plain, {}, first)                               # no stand-in: a missing clip is an error, not a Ken Burns move
    with pytest.raises(ValueError, match="no clean clips"):
        LY.MographLayerGather.execute(plain, {}, first, None, False)
    with pytest.raises(ValueError, match="stills"):
        FT.MographFootageGather.execute(_piece3(), {}, None)


def test_the_matte_of_a_still_preview_is_frame_zero_held_over_the_shot(nodes):
    import torch
    nine = torch.rand(9, 8, 6, 3)
    held = nodes.footage.hold_first(nine, SP.hold_len(_piece3(), "s2"))
    assert tuple(held.shape) == (10, 8, 6, 3) and all(torch.equal(held[i], nine[0]) for i in range(10))
    assert SP.hold_len(_piece3(), "s1") == 6 and SP.hold_len(_piece3(), "s9", 3) == 3


def test_a_held_still_gets_zero_flow_not_raft(nodes, monkeypatch):
    import torch
    FT = nodes.footage
    monkeypatch.setattr(FT.RE, "shot_needs_flow", lambda piece, sid: True)
    monkeypatch.setattr(FT, "clip_flow", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RAFT / DIS must not run over a held still")))
    clips, _, _ = FT.still_outputs(_piece3(), _first_batch())
    out = FT.MographFootageFlow.execute(_piece3(), clips, "auto", "raft_large").args[0]
    assert tuple(out[0]["flow"].shape) == (6, 48, 27, 2) and out[0]["flow"].dtype == torch.float16 and float(out[0]["flow"].abs().max()) == 0.0


# ------------------------------------------------------------------------------------------------------------------------ no stand-in code path remains
def test_no_stand_in_code_path_remains():
    bad = re.compile(r"_is_standin|_is_draft|\bcache_path\b|\bcache_fits\b|mograph_cache|ken_burns|\bluma_matte\b|\bluma_depth\b|MographDraftMatte|MographDraftDepth|MographAudioOrDefault|palette_gradient|seed_cache")
    for rel in ("nodes_fusion/footage.py", "nodes_fusion/layers.py", "nodes_fusion/inputs.py", "nodes_fusion/_util.py", "tools/app/build_graphs.py", "tools/app/llm_arms.py",
                "tools/app/layered_shots.py", "tools/app/run_batch.py"):
        hits = [l.strip()[:90] for l in (ROOT / rel).read_text(encoding="utf-8").splitlines() if bad.search(l) and not l.strip().startswith("#")]
        assert not hits, (rel, hits)
    assert not (ROOT / "tools" / "app" / "seed_cache.py").exists() and not (ROOT / "tools" / "app" / "test_footage.py").exists()


def test_the_cpu_draft_graph_builder_option_is_gone():
    from tools.app import build_graphs as BG
    for fn in (BG.build_graph, BG.build_sidecar):
        with pytest.raises(ValueError, match="removed"):
            fn(True)
    assert not {"MographDraftMatte", "MographDraftDepth", "MographAudioOrDefault"} & set(BG.EXTRA_CLASSES)


# ------------------------------------------------------------------------------------------------------------------------ the built graph (graph inspection)
@pytest.fixture(scope="module")
def graphs():
    from tools.app import build_graphs as BG
    from tools.app import layered_shots as LS
    from tools.app import llm_arms as LA
    g, side = BG.make(False, None, "moss", True, True)                      # fusion_design_qwen_auto
    g, side = LS.convert(g, side)                                           # _layers (clean + text + plate chains)
    g = BG.insert_pivot_cut(g)                                              # _pivot
    g, side = LA.convert(g, side, "gemma31b_qat")                           # the default arm (24 GB tier: plates on)
    return {"g": g, "side": side, "BG": BG}


LAZY_BY_STILL = {"MographLayerClean": re.compile(r"^clean_\d$"), "MographLayerGather": re.compile(r"^(text|plate)_\d$"), "MographFootageGather": re.compile(r"^clip_\d$")}
OUT_CLASSES = ("PreviewImage", "SaveVideo", "SaveImage", "PreviewAny", "MographArchive", "MographSaveFinalUpscaled")


def executed(g: dict, still: bool) -> set[str]:
    """the nodes ComfyUI would run for this graph: everything upstream of an output node, following lazy inputs only where the node asks for them. A still preview: the three
    gather barriers ask for no clip (nodes_fusion check_lazy_status); every other lazy input (mattes, depth, the stills) is followed (conservative: more nodes than really run)."""
    seen, stack = set(), [n for n, v in g.items() if v["class_type"] in OUT_CLASSES]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        cls = g[n]["class_type"]
        for name, v in g[n]["inputs"].items():
            if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in g):
                continue
            if still and cls in LAZY_BY_STILL and LAZY_BY_STILL[cls].match(name):
                continue
            stack.append(v[0])
    return seen


def test_a_still_preview_runs_no_ltx_video_chain(graphs):
    g = graphs["g"]
    run = executed(g, still=True)
    chains = [n for n in run if re.match(r"^[ltcp]\d\.", n)]
    assert chains == [], f"clean (l), text (t), union-control (c) or plate (p) LTX chains would run: {chains[:6]}"
    samplers = sorted(n for n in run if g[n]["class_type"] == "SamplerCustomAdvanced")
    assert samplers == [f"a{k}.37" for k in range(1, 7)], f"the only LTX sampler of a still preview is the Alpha-Gen matte (9 frames of the still): {samplers}"
    assert not [n for n in run if g[n]["class_type"] in ("LTXVImgToVideoInplace", "MographShotSlot", "MographLayerPrompt", "MographLayerGate", "MographPlatePrompt", "MographPlateRef")]
    assert "type" not in run, "the first type pass only cuts the LTX control video: it does not run"
    assert {"gather", "type_final", "compose", "pivot", "flow", "vda"} <= run and all(f"d{k}.proc" in run for k in range(1, 7)), "depth (VDA Small), the designed type and the 3D cuts run as normal"


def test_a_normal_run_still_runs_the_ltx_chains(graphs):
    g = graphs["g"]
    run = executed(g, still=False)
    assert "l1.39" in run and "t1.39" in run and "slot1" in run and "type" in run, "the simulation must see the clean + text chains of a draft / final run"


def test_mattes_and_depth_come_from_the_shots_own_still(graphs):
    g = graphs["g"]
    gather = g["gather"]
    assert gather["class_type"] == "MographLayerGather" and gather["inputs"]["first"] == ["first_set", 0], "the gather holds the row of each place's still"
    for k in range(1, 7):
        assert g[f"a{k}.17"]["inputs"]["input"] == ["gather", k], "Alpha-Gen reference = the gather's alpha_ref (9 copies of the shot's still)"
        assert g[f"d{k}.proc"]["inputs"]["images"] == ["gather", 6 + k], "VDA Small reads the gather's depth_ref (the shot's still, 1 frame)"
        assert g[f"a{k}.dec"]["inputs"]["gray"] is True and g[f"a{k}.dec"]["inputs"]["piece"] == ["plan", 0]
        assert g["compose"]["inputs"][f"matte_{k}"] == [f"a{k}.dec", 1] and g["pivot"]["inputs"][f"matte_{k}"] == [f"a{k}.dec", 1], "the 3D cuts and compose read the held matte"
    assert g["first_set"]["class_type"] == "MographFirstFrames" and g["still_lat"]["inputs"]["width"] == ["sizes", 6], "stills at the sizes node's still size (= sizes.type)"


def test_the_quality_option_and_the_app_default(graphs):
    side, g = graphs["side"], graphs["g"]
    q = next(f for f in side["fields"] if f["id"] == "quality")
    assert q["options"] == [{"value": "still", "label": "Preview"}, {"value": "final", "label": "Final"}], "the app shows two choices; Draft is an internal tier"
    assert q["default"] == "final" and g["quality"]["inputs"]["value"] == "preview", "the tier default is Final (= the draft tier, as before); the graph's own IN quality literal stays (batch drivers)"
    BG = graphs["BG"]
    plain = BG.build_sidecar(False, "moss", True, True)
    qd = next(f for f in plain["fields"] if f["id"] == "quality")
    assert [o["value"] for o in qd["options"]] == ["still", "final"] and qd["default"] == "still"
    assert [o["value"] for o in next(f for f in BG.build_sidecar(False, "moss", False, False)["fields"] if f["id"] == "quality")["options"]] == ["still", "final"]


def test_the_graph_files_check_clean(graphs):
    BG = graphs["BG"]
    g, side = BG.make(False, None, "moss", True, True)
    problems, _ = BG.check(g, side)
    assert problems == [], problems
    assert not BG.check_reachability(graphs["g"])


# ------------------------------------------------------------------------------------------------------------------------ the lock: upgrade to the real render
def _saved_piece():
    q, flag = SP.split_quality("still")
    raw = {"words": "BREAK\nCLAIM", "scene": "a harbour at dawn", "music": "slow, warm", "length": 12, "look": "auto", "seed": 7, "format": "9:16", "shots": "cut", "owners": "[]"}
    req = PR.normalize_request(dict(raw, quality=q))
    piece = SP.prepare_piece(PR.plan_piece(req, synth_music(100.0, 12.0)), raw, flag)
    piece["type"]["words"][0]["font"] = "DESIGNED-BY-THE-MODEL"           # a design decision of the preview run
    return piece


def test_the_upgrade_graph_re_uses_the_stills_and_the_design_and_calls_no_model(graphs):
    from tools.app import upgrade_still as US
    g0 = graphs["g"]
    piece = _saved_piece()
    names = {"stills": {k: f"u_still_{k}.png" for k in range(1, 7)}, "audio": "u_music.wav"}
    g = US.upgrade_graph(g0, {"piece": piece}, names)
    assert g0["quality"]["inputs"]["value"] == "preview" and "concept_final" in g0, "the input graph is untouched"
    assert g["quality"]["inputs"]["value"] == "final", "the app's Final (MographRequest maps it to SP.APP_FINAL_TIER)"
    assert g["still_dec"] == {"inputs": {"image": "u_still_1.png"}, "class_type": "LoadImage", "_meta": g["still_dec"]["_meta"]} and g["still2_dec"]["inputs"]["image"] == "u_still_2.png"
    assert g["audio"]["inputs"]["audio"] == "u_music.wav"
    for cls in ("MographGGUFVL", "DesignLLM_Advanced", "MographConceptPrompt", "MographDesignPackage", "MographDesignParse", "MographDesignRepair", "MographDesignRepair"):
        assert not [n for n, v in g.items() if v["class_type"] == cls], f"{cls}: no model call, no concept, no design step in the upgrade"
    assert not [n for n in g if re.match(r"^still\d?_(ks|txt|lat|unet|clip)$", n) or n in ("still_unet", "still_clip")], "no still model: the saved PNGs"
    ins = g["request"]["inputs"]
    assert (ins["words"], ins["scene"], ins["music"], ins["look"], ins["shots"], ins["owners"]) == ("BREAK\nCLAIM", "a harbour at dawn", "slow, warm", "auto", "cut", "[]")
    assert g["length"]["inputs"]["value"] == 12 and g["seed"]["inputs"]["value"] == 7 and g["format"]["inputs"]["value"] == "9:16"
    for nid in ("plan", "design_piece"):
        assert g[nid]["class_type"] == "MographPieceFromJSON"
        p = json.loads(g[nid]["inputs"]["piece_json"])
        assert p["quality"] == SP.APP_FINAL_TIER and SP.FLAG not in p and p["type"]["words"][0]["font"] == "DESIGNED-BY-THE-MODEL", "the same design, the flag off"
        assert p["stills"] == piece["stills"] and p["shots"] == piece["shots"]
    assert not [n for n in g if n.startswith("concept") or n.startswith("design_q") or n == "design_pkg" or n == "previous"]
    run = executed(g, still=False)
    assert "l1.39" in run and "t1.39" in run, "the upgrade is the real LTX render again (clean + text chains), and the Alpha-Gen / depth chains over the whole shot"
    assert not graphs["BG"].check_reachability(g)


def test_only_a_draft_re_uses_the_draft_size_stills():
    from tools.app import upgrade_still as US
    with pytest.raises(ValueError, match="Final"):
        US.upgrade_graph({}, {"piece": {}}, {}, "draft")
    with pytest.raises(ValueError):
        SP.upgraded_piece({"quality": "draft"}, "preview")
    assert US.upgrade_part("r01_gemma_ltx_still") == "r01_gemma_ltx_up540" and US.upgrade_part("r02_gemma_still") == "r02_gemma_ltx_up540" and US.upgrade_part("r03", compose_h=360) == "r03_ltx_up360"


def test_an_upgrade_never_writes_under_the_name_of_its_source_or_of_an_earlier_upgrade():
    """user rule: nothing is overwritten, every composition and every generation (stills, clips, mattes, depth, design json) has a unique name."""
    from tools.app import upgrade_still as US
    src = "r01_gemma_ltx_still"
    first = US.upgrade_part(src)
    second = US.upgrade_part(src, taken={first})
    third = US.upgrade_part(src, taken={first, second})
    assert len({src, first, second, third}) == 4 and second == first + "b" and third == first + "c"
    target = "COMPETITIONS/motion-graphic/app/v012"
    stems = [US.out_stem(target, p) for p in (src, first, second, third)]
    assert len(set(stems)) == 4, "every ingredient (<stem>_still.png, _sK_clip.mp4, _sK_matte/, _sK_depth, _piece.json, _music.wav) is written under a different stem"
    assert stems[0] == "motion-graphic_app_v012_r01_gemma_ltx_still" and stems[1] == "motion-graphic_app_v012_r01_gemma_ltx_up540"
    assert US.upgrade_part("r01_gemma_ltx_up540") != "r01_gemma_ltx_up540"


def test_the_saved_run_is_found_and_checked(tmp_path):
    from tools.app import upgrade_still as US
    d = tmp_path / "x_r01_gemma_ltx_still_out_shots"
    d.mkdir()
    piece = _saved_piece()
    (d / "x_r01_gemma_ltx_still_piece.json").write_text(json.dumps(piece), encoding="utf-8")
    (d / "x_r01_gemma_ltx_still_still.png").write_bytes(b"png")
    (d / "x_r01_gemma_ltx_still_still_1.png").write_bytes(b"png")
    (d / "x_r01_gemma_ltx_still_still_2.png").write_bytes(b"png")
    with pytest.raises(FileNotFoundError, match="music"):
        US.load_saved(str(d))
    (d / "x_r01_gemma_ltx_still_music.wav").write_bytes(b"wav")
    saved = US.load_saved(str(d))
    assert sorted(saved["stills"]) == [1, 2] and saved["stem"] == "x_r01_gemma_ltx_still" and saved["piece"][SP.FLAG] is True
    notflag = copy.deepcopy(piece)
    notflag.pop(SP.FLAG)
    (d / "x_r01_gemma_ltx_still_piece.json").write_text(json.dumps(notflag), encoding="utf-8")
    with pytest.raises(ValueError, match="not a still preview"):
        US.load_saved(str(d))


def test_the_upgrade_dry_run_writes_the_graph_and_uploads_nothing(tmp_path, graphs):
    from tools.app import upgrade_still as US
    d = tmp_path / "vdir" / "out" / "x_r01_gemma_ltx_still_out_shots"
    d.mkdir(parents=True)
    st = "x_r01_gemma_ltx_still"
    (d / f"{st}_piece.json").write_text(json.dumps(_saved_piece()), encoding="utf-8")
    for n in ("still", "still_1", "still_2"):
        (d / f"{st}_{n}.png").write_bytes(b"png")
    (d / f"{st}_music.wav").write_bytes(b"wav")
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps(graphs["g"]), encoding="utf-8")
    gp.with_suffix(".app.json").write_text(json.dumps(graphs["side"]), encoding="utf-8")
    up = []
    res = US.run(dict(target="P/I/v001", part="r01_gemma_ltx_still", iid="1", port=1, graph=str(gp), out=str(tmp_path / "o"), dry_run=True),
                 uploader=lambda *a, **k: up.append(a), resolve=lambda t: str(tmp_path / "vdir"), log=lambda *a: None)
    assert res["dry_run"] and res["part"] == "r01_gemma_ltx_up540" and up == []
    assert all(res["part"] in n for n in res["names"]["stills"].values()) and res["part"] in res["names"]["audio"], "uploaded input files carry the new part: no overwrite of the preview's or an earlier upgrade's"
    res2 = US.run(dict(target="P/I/v001", part="r01_gemma_ltx_still", iid="1", port=1, graph=str(gp), out=str(tmp_path / "o"), dry_run=True),
                  uploader=lambda *a, **k: None, resolve=lambda t: str(tmp_path / "vdir"), log=lambda *a: None)
    assert res2["part"] == "r01_gemma_ltx_up540", "nothing was written to out/, so the label is still free"
    (d.parent / "x_r01_gemma_ltx_up540_out_shots").mkdir()
    res3 = US.run(dict(target="P/I/v001", part="r01_gemma_ltx_still", iid="1", port=1, graph=str(gp), out=str(tmp_path / "o"), dry_run=True),
                  uploader=lambda *a, **k: None, resolve=lambda t: str(tmp_path / "vdir"), log=lambda *a: None)
    assert res3["part"] == "r01_gemma_ltx_up540b", "a second upgrade of the same preview gets its own label"
    assert json.loads(Path(res["graph"]).read_text(encoding="utf-8"))["quality"]["inputs"]["value"] == "final"
    assert "upgrade of a still preview" in json.loads(Path(res["graph"]).with_suffix(".app.json").read_text(encoding="utf-8"))["title"]


def test_the_batch_driver_labels_a_still_preview_part():
    src = (ROOT / "tools" / "app" / "run_batch.py").read_text(encoding="utf-8")
    assert 'SP.SUFFIX if a.get("quality") == SP.QUALITY' in src and '"--upgrade"' in src and '"--quality"' in src
