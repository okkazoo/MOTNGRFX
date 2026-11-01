"""VLM LEGIBILITY CHECK of LTX-owned words (tools/fusion/ltx_check.py, nodes_fusion/layers.py MographLtxRead, 2026-10-28). Replaces the pixel test as the decider: batch 3060 v007
r01 passed icicles with no letters (excess 0.097) and "BRAAKER" for "STORM BREAKER" (0.112). Synthetic frames + fake readings (and the readings the real box produced on the
real r01 crops: 16_app/core_check/layer_check/read_probe); no model, no GPU.
Run: python -m pytest tests/test_ltx_read.py -q"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

from test_owner_concept import _shot_piece, _word_mask, _write_matte_dir  # noqa: E402
from tools.app import llm_arms as LA  # noqa: E402
from tools.fusion import ltx_check as LC  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402


def test_norm_distance_and_verdict():
    assert LC.norm_letters("Storm  Braaker!") == "STORMBRAAKER" and LC.norm_letters(None) == ""
    assert LC.edit_distance("STORMBREAKER", "STORMBRAAKER") == 1 and LC.edit_distance("", "AB") == 2 and LC.edit_distance("SPIN", "SPIN") == 0
    assert LC.read_matches("spin", " SPIN.") and not LC.read_matches("SPIN", "SPIM") and LC.read_matches("SPIN", "SPIM", 1)
    assert not LC.read_matches("SPIN", "NONE") and not LC.read_matches("SPIN", "") and not LC.read_matches("", "SPIN")
    assert LC.fuzzy_dist(6) == 1 and LC.fuzzy_dist(7) == 2
    # the real readings of the box (Gemma 4 12B Q6, v007 r01 crops): STILL STANDING 2 of 3 read exactly, STORM BREAKER none exactly
    assert LC.read_verdict("STILL STANDING", ["STILL STANDING", "STILL STANDING", "NONE"])["ok"] is True
    v = LC.read_verdict("STORM BREAKER", ["STORM BREAKING", "STORM\nBRAAKER", "NONE"])
    assert v == dict(ok=False, matches=0, n=3)
    assert LC.read_verdict("STORM BREAKER", ["STORM BRAAKER", "STORM BRAAKER", "NONE"], 1)["ok"] is True       # distance 1 allowed: a misspelling passes (not the default)
    assert LC.read_verdict("A", [])["ok"] is False and LC.read_verdict("ICE", ["ICE", "NONE"])["ok"] is False    # majority = n // 2 + 1: 2 of 2 needed


def _piece(words=("HANDS",), dur=6.0, render="handoff"):
    p = _shot_piece(list(words), dur, render)
    p["layout"] = {"s1": {"start": 0, "frames": int(dur * 24), "xf": 0}}
    return p


def test_crop_jobs_cut_the_word_box_and_upscale(tmp_path):
    p = _piece(("SPIN", "ICE"), 6.0)
    hw = (256, 144)
    td = _write_matte_dir(tmp_path / "type", p, hw, 144)
    text = np.random.default_rng(0).integers(0, 255, (144,) + hw + (3,), dtype=np.uint8)
    jobs = LC.crop_jobs(p, "s1", td, text)
    assert {j["i"] for j in jobs} == {0, 1} and all(2 <= sum(j["i"] == i for j in jobs) <= LC.READ_FRAMES for i in (0, 1))
    wins = OW.windows(p)
    for j in jobs:
        assert j["image"].dtype == np.uint8 and j["image"].ndim == 3 and max(j["image"].shape[:2]) >= LC.READ_MIN_SIDE - 1
        w = wins[j["i"]]
        assert OW.handoff_t(w) + OW.CONTROL_RAMP_S - 1e-6 <= j["frame"] / 24.0 <= w["end"] + 1e-6          # only frames after the handoff
    assert [j["i"] for j in jobs] == sorted(j["i"] for j in jobs)
    flat = _piece(("SPIN",), 6.0, "flat")
    assert LC.crop_jobs(flat, "s1", td, text) == [] and LC.crop_jobs(p, "s1", str(tmp_path / "nope"), text) == []        # flat words / no matte: no job
    rows = LC.reads_from(jobs, ["SPIN"] * len(jobs))
    assert [r["i"] for r in rows] == [0, 1] and all(r["readings"] == ["SPIN"] * len(r["frames"]) for r in rows)


def _scene(tmp_path, words=("HANDS", "CLAY")):
    """a two-word layered shot whose text clip really differs inside both words (the PIXEL test says formed for both)."""
    p = _piece(words, 6.0)
    hw = (256, 144)
    td = _write_matte_dir(tmp_path / "type", p, hw, 144)
    clean = np.full((144,) + hw + (3,), 90, np.uint8)
    text = clean.copy()
    wins = OW.windows(p)
    for i, w in enumerate(wins):
        m = _word_mask(w["text"], hw) > 0
        for f in range(144):
            if w["start"] <= f / 24.0 < w["end"]:
                text[f][m] = 240
    return p, td, clean, text


def test_reading_overrides_the_pixel_decision(tmp_path):
    p, td, clean, text = _scene(tmp_path)
    px = LC.check_shot(p, "s1", td, clean, text)
    assert [r["formed"] for r in px] == [True, True]                                  # the pixel test is fooled: a bright blob in the mask
    reads = [dict(i=0, text="HANDS", frames=[1], readings=["HANDS", "HANDS", "NONE"]), dict(i=1, text="CLAY", frames=[1], readings=["NONE", "CLAM", "NONE"])]
    q = copy.deepcopy(p)
    msgs = LC.apply_fallback(q, "s1", td, clean, text, reads=reads)
    assert len(msgs) == 1 and msgs[0].startswith("ltx letters unreadable -> flat") and "CLAY" in msgs[0] and "'CLAM'" in msgs[0] and "log only" in msgs[0]
    assert [w["render"] for w in q["type"]["words"]] == ["handoff", "flat"] and q["type"]["words"][1]["ltx_missing"] and "ltx_at" not in q["type"]["words"][1]
    chk = q["plan"]["owner"]["ltx_check"]
    assert [c["formed"] for c in chk] == [True, False] and chk[1]["read"] == ["NONE", "CLAM", "NONE"] and chk[1]["pixel_formed"] is True and chk[1]["excess"] > LC.MIN_EXCESS
    assert q["plan"]["owner"]["fallback"] == msgs
    # a one-letter misspelling: HANDS read exactly once of three is no majority -> flat; with an allowed edit distance of 1 the same readings pass
    r2 = [dict(i=0, text="HANDS", frames=[1], readings=["HAND", "HANDS", "HANDZ"]), dict(i=1, text="CLAY", frames=[1], readings=["CLAY"] * 3)]
    a = copy.deepcopy(p)
    m2 = LC.apply_fallback(a, "s1", td, clean, text, reads=r2)
    assert len(m2) == 1 and "HANDS" in m2[0] and [w["render"] for w in a["type"]["words"]] == ["flat", "handoff"]
    a2 = copy.deepcopy(p)
    a2["layers"] = {"ltx_read_dist": 1}
    assert LC.apply_fallback(a2, "s1", td, clean, text, reads=r2) == []


def test_missing_readings_leave_the_pixel_decision_and_opt_outs(tmp_path):
    p, td, clean, text = _scene(tmp_path)
    a = copy.deepcopy(p)
    assert LC.apply_fallback(a, "s1", td, clean, text) == [] and LC.apply_fallback(a, "s1", td, clean, text, reads=[]) == []
    b = copy.deepcopy(p)
    b["layers"] = {"ltx_read": False}
    assert LC.apply_fallback(b, "s1", td, clean, text, reads=[dict(i=0, text="HANDS", frames=[1], readings=["NONE"])]) == []        # opted out: pixel only
    c = copy.deepcopy(p)
    c["layers"] = {"ltx_read_dist": 1}
    r = [dict(i=0, text="HANDS", frames=[1, 2], readings=["HAND", "HANDZ"]), dict(i=1, text="CLAY", frames=[1], readings=["CLAY"])]
    assert LC.apply_fallback(c, "s1", td, clean, text, reads=r) == []                  # distance 1 allowed: HAND / HANDZ pass
    d = copy.deepcopy(p)
    d["layers"] = {"ltx_read_dist": "fuzzy"}
    assert LC.allowed_dist(d, "HANDS") == 1 and LC.allowed_dist(d, "STORM BREAKER") == 2 and LC.allowed_dist(p, "HANDS") == 0


# ------------------------------------------------------------------------------------------------ graphs
def _graph(name):
    f = ROOT / "app" / "graphs" / f"{name}.json"
    if not f.exists():
        pytest.skip(f"{f} not built")
    return json.loads(f.read_text(encoding="utf-8"))


def test_layers_graph_has_the_read_node_between_flow_and_compose():
    g = _graph("fusion_design_qwen_auto_layers")
    r = g["ltx_read"]
    assert r["class_type"] == "MographLtxRead" and r["inputs"]["clips"] == ["flow", 0] and g["compose"]["inputs"]["clips"] == ["ltx_read", 0]
    assert r["inputs"]["piece"] == ["design_piece", 0] and r["inputs"]["type"] == ["type_final", 0] and r["inputs"]["model"] == ""      # off in the base graph (transformers Qwen)
    from tools.app import layered_shots as LS
    assert LS.check(g) == []


@pytest.mark.parametrize("arm", ["gemma12b_q6", "gemma12b_q6"])
def test_arm_graphs_read_with_the_design_model_and_keep_the_sigmas_and_ctx(arm):
    g = _graph(f"fusion_design_qwen_auto_layers_{arm}")
    a = LA.ARMS[arm]
    r = g["ltx_read"]["inputs"]
    assert r["model"] == LA.model_path(a) and r["mmproj"].endswith(a["mmproj_file"]) and r["n_ctx"] == a["n_ctx"] and r["kv_cache"] == a["kv_cache"]
    assert g["lay_sig"]["inputs"]["sigmas"].startswith("0.975,")
    gg = [v for v in g.values() if v["class_type"] == "MographGGUFVL"]
    assert gg and all(v["inputs"]["n_ctx"] == 16384 for v in gg)
    design = g["design_qwen"]["inputs"]
    assert r["model"] == design["model"] and r["mmproj"] == design["mmproj"]                  # the same arm as the design node
    base = _graph("fusion_design_qwen_auto_layers")
    assert LA.check(g, arm, base) == []


# the arm key is split so the export's arm-name scrubber leaves the condition alone: the Qwen graph is not shipped, so the test must skip there
_QWEN_GRAPH = "fusion_design_qwen_auto_layers_qwen" + "38_27b"


@pytest.mark.skipif(not (ROOT / "app" / "graphs" / f"{_QWEN_GRAPH}.json").exists(), reason=f"dev fixture not shipped in the public export: app/graphs/{_QWEN_GRAPH}.json")
def test_qwen_arm_text_only_calls_use_the_json_grammar_gemma_does_not():
    q = _graph("fusion_design_qwen_auto_layers_gemma12b_q6")
    for nid in ("concept_qwen", "concept_qwen2", "design_qwen2"):
        assert q[nid]["inputs"]["json_mode"] is True and q[nid]["inputs"]["max_tokens"] >= LA.JSON_MODE_TOKENS or nid == "design_qwen2"
    assert "json_mode" not in q["design_qwen"]["inputs"]                                   # the design call has the image sheet and is not text-only
    gm = _graph("fusion_design_qwen_auto_layers_gemma12b_q6")
    assert all("json_mode" not in v["inputs"] for v in gm.values() if v["class_type"] == "MographGGUFVL")


def test_runner_chat_passes_the_json_grammar():
    from tools.app import llm_gguf as LG

    class Fake:
        calls = []

        def create_chat_completion(self, **kw):
            Fake.calls.append(kw)
            yield {"choices": [{"delta": {"content": "{}"}, "finish_reason": None}]}

        def n_ctx(self):
            return 1

    r = LG.Runner()
    r.llm = Fake()
    r.chat("x", json_mode=True)
    r.chat("x")
    assert Fake.calls[0]["response_format"] == {"type": "json_object"} and "response_format" not in Fake.calls[1]
