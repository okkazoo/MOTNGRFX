"""CONTOUR_MORPH = the default word-to-word move on a shot with a subject matte (tools/fusion/contour_morph.py, ship #8, gallery r2b1) + the brush `var` line scale.
Tiny synthetic frames (numpy). Run: python -m pytest tests/test_contour_morph.py -q"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_hold_legibility as THL  # noqa: E402
from tools.fusion import contour_morph as CM  # noqa: E402
from tools.fusion import contour_sweep as CS  # noqa: E402
from tools.fusion import explore_effects as EE  # noqa: E402
from tools.fusion import explore_ops as XO  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W = THL.H, THL.W
NF = 120                                           # a 5 s shot
HOLDS = [(0.3, 2.4), (3.9, 4.9)]                   # gap 1.5 s
CHK = {"words": [{"hit": a, "release": b, "end": b + 0.4} for a, b in HOLDS], "notes": []}


def _piece(matte=True, transitions=None, designed_words=None):
    p = THL._piece(THL._old_ops(), holds=HOLDS)
    p["shots"] = [{"id": "s1", "frames": NF, "matte": matte}]
    p["layout"] = {"s1": {"start": 0, "frames": NF, "xf": 0}}
    p["plan"]["piece_system"] = {"seed": 4, "palette": {"bg": [20, 20, 24], "fg": [235, 230, 220], "accents": [[200, 60, 40]]}, "effects": {}}
    p["type"] = {"bpm": 90.0, "words": [{"text": "AB", "t0": 0.0, "t1": 2.8}, {"text": "CD", "t0": 3.3, "t1": 5.0}]}
    if transitions is not None:
        p["type"]["transitions"] = transitions
    return p


def test_plan_applies_on_a_matte_shot_with_a_gap_of_a_second_or_more():
    ev = CM.plan(_piece(), HOLDS, {"bpm": 90.0})
    assert len(ev) == 1 and ev[0]["shot"] == "s1" and ev[0]["word"] == 0
    p = ev[0]["params"]
    assert p["rel"] == 2.4 and p["land"] == 3.9 and p["mid"] == pytest.approx(3.15)
    assert 2.4 - 0.3 > p["lag_a"] / 24 and ev[0]["t0"] == 2.4 and ev[0]["t1"] >= 3.9 + 0.6 - 1e-6       # B reads >= max(1 beat, 0.6 s) after it lands


def test_plan_keeps_the_existing_behaviour_without_matte_or_with_a_short_or_huge_gap():
    assert CM.plan(_piece(matte=False), HOLDS) == []
    assert CM.plan(_piece(), [(0.3, 2.4), (3.3, 4.9)]) == []                       # 0.9 s gap
    assert CM.plan(_piece(), [(0.3, 1.0), (3.9, 4.9)]) == []                       # 2.9 s gap: the engine's own transition plays
    assert CM.plan(_piece(), [(0.3, 0.5), (1.9, 3.0)]) == []                       # word A's hold is too short to sample its landed letters
    p = _piece()
    p["compose"]["contour_morph"] = False
    assert CM.plan(p, HOLDS) == []


def test_a_model_chosen_transition_is_kept_but_null_or_contour_morph_morphs():
    assert CM.plan(_piece(transitions=["braid"]), HOLDS, designed=True) == []
    assert len(CM.plan(_piece(transitions=[None]), HOLDS, designed=True)) == 1
    assert len(CM.plan(_piece(transitions=["contour_morph"]), HOLDS, designed=True)) == 1
    assert len(CM.plan(_piece(transitions=["braid"]), HOLDS, designed=False)) == 1        # the rules planner's assignment is only a default
    p = _piece()
    p["type"]["words"][0]["transition"] = {"kind": "pan", "dir": "down"}
    assert CM.plan(p, HOLDS, designed=False) == []                                   # a pan owns the gap


def test_ops_use_the_liked_line_scale_and_go_straight_to_the_word():
    ev = CM.plan(_piece(), HOLDS, {"bpm": 90.0})[0]
    ops = CM.build_ops(dict(colours=EE.colours(_piece()), seed=3, plated=False, params=ev["params"]))
    assert [o["op"] for o in ops] == ["sdf_morph", "sdf_morph", "mask_gate", "split_region", "mask_mul", "mask_max", "knockout", "brush_edge"]
    be = ops[-1]["args"]
    assert be["var"] == 1.25 and be["guard"] is False and CM.LINE_W == 0.006
    tl = be["width_px"]["timeline"]
    assert tl[0][1] == pytest.approx(0.006 * 360.0) and tl[-1][1] < 0.5 * tl[0][1]            # 0.006 x width, thinning away as the word lands
    assert ops[0]["args"]["a"].startswith("$type_matte@-") and ops[1]["args"]["b"].startswith("$type_matte@")
    kn = ops[6]["args"]
    assert kn["type_a"] == "$cmwin" and kn["min_contrast"] == 0.22 and kn["fg_matte"] == "$fxm"
    # one continuous line, no free strokes: the stroke follows the morphing shape (a -> contour -> b) all the time, then the letters fill
    wipe = ops[3]["args"]["pos"]["timeline"]
    morph2 = ops[1]["args"]["k"]["timeline"]
    assert wipe[0][0] == pytest.approx([x for x, k in morph2 if k == 1.0][0], abs=1e-3)       # the letters fill only once the line IS the letter outline
    assert wipe[-1][0] <= ev["params"]["land"] + 0.11 and be["opacity"]["timeline"][0][1] == 0.0


def test_brush_edge_var_widens_the_thick_thin_band_and_default_is_unchanged():
    from tools.fusion import contour_sweep as cs
    seen = []
    orig = cs.edge_alpha

    def spy(m, frame, seed=0, width_px=None, params=None, section=None, guard=True):
        seen.append(dict(params or {}))
        return orig(m, frame, seed=seed, width_px=width_px, params=params, section=section, guard=guard)
    cs.edge_alpha = spy
    try:
        img = np.zeros((80, 60, 3), np.float32)
        m = np.zeros((80, 60), np.float32)
        m[20:60, 15:45] = 1.0
        XO.brush_edge(img, m, frame_idx=3, guard=False)
        XO.brush_edge(img, m, frame_idx=3, guard=False, var=1.25)
    finally:
        cs.edge_alpha = orig
    assert "width_range" not in seen[0]                                       # var 1.0 = byte-identical to before
    lo, hi = seen[1]["width_range"]
    assert lo < 0.3 < 1.6 < hi


def test_weave_gates_it_before_the_tail_idempotently_and_never_strips_an_explore_look():
    p = _piece(transitions=[None])
    p["plan"]["piece_system"]["effects"] = {"explore": ["chronophoto_echoes"]}
    EE.weave(p, {"bpm": 90.0}, CHK, (W, H))
    n_fx = len([o for o in p["compose"]["default"]["ops"] if o.get("fx")])
    CM.weave(p, {"bpm": 90.0}, CHK, (W, H), designed=True)
    ops = p["compose"]["default"]["ops"]
    cm = [o for o in ops if o.get(CM.FLAG)]
    assert p["plan"]["contour_morph"]["words"] in ([0], [])
    if p["plan"]["contour_morph"]["words"]:                                      # not under the explore section: both live in the chain
        assert cm and ops[-1]["op"] == "grade" and len([o for o in ops if o.get("fx")]) == n_fx
        assert sum(1 for o in cm if o["op"] == "crossfade") == 1
        CM.weave(p, {"bpm": 90.0}, CHK, (W, H), designed=True)                  # idempotent
        assert len([o for o in p["compose"]["default"]["ops"] if o.get(CM.FLAG)]) == len(cm)
    p2 = _piece(transitions=[None])
    CM.weave(p2, {"bpm": 90.0}, CHK, (W, H))
    assert p2["plan"]["contour_morph"]["words"] == [0] and any(o.get(CM.FLAG) for o in p2["compose"]["default"]["ops"])
    p2["compose"]["contour_morph"] = False
    CM.weave(p2, {"bpm": 90.0}, CHK, (W, H))
    assert not any(o.get(CM.FLAG) for o in p2["compose"]["default"]["ops"])


def test_an_explore_section_on_the_same_gap_wins():
    p = _piece(transitions=[None])
    p["plan"]["explore"] = {"placed": [dict(name="echo_trail", shot="s1", t0=2.0, t1=4.0, word=0)]}
    assert CM.plan(p, HOLDS) == []


def test_the_sweep_leaves_the_gap_to_the_morph():
    p = _piece(transitions=["braid"])
    p["compose"]["cuts"] = []
    base = CS.plan(p, music={"bpm": 90.0}, chk=CHK, holds=HOLDS)
    assert any(e["type"] == "word" for e in base["events"])
    p["plan"]["contour_morph"] = {"words": [0]}
    assert not any(e["type"] == "word" for e in CS.plan(p, music={"bpm": 90.0}, chk=CHK, holds=HOLDS)["events"])


def test_the_menu_and_the_design_step_know_the_name():
    from tools.fusion import design_step as DS
    from tools.fusion import qwen_piece as QP
    assert "contour_morph" in QP.TRANSITIONS
    out = {"type": {"transitions": ["braid", "contour_morph", None], "words": [{"text": "A", "transition": "contour_morph"}, {"text": "B", "transition": {"kind": "pan", "dir": "up"}}]}}
    DS._morph_names_to_null(out)
    assert out["type"]["transitions"] == ["braid", None, None] and "transition" not in out["type"]["words"][0] and out["type"]["words"][1]["transition"]["kind"] == "pan"
    from tools.fusion import design_salvage as SV
    assert SV.SYN_TRANSITION["contour"] == "contour_morph"


def _write_two_words(tmp):
    """type layers for 5 s: word A (three bars) until 3.0 s, word B (two wide bars) from then on."""
    a = np.zeros((H, W), np.float32)
    for x0 in (10, 38, 66):
        cv2.rectangle(a, (x0, 50), (x0 + 20, 110), 1.0, -1)
    b = np.zeros((H, W), np.float32)
    cv2.rectangle(b, (12, 40), (44, 120), 1.0, -1)
    cv2.rectangle(b, (54, 40), (86, 120), 1.0, -1)
    for k, fn in (("beauty", lambda m: np.repeat(m[..., None], 3, 2) * 0.9), ("matte", lambda m: m), ("line", lambda m: m)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        for i in range(NF):
            m = a if i < 72 else b
            u8 = np.clip(fn(m) * 255 + 0.5, 0, 255).astype(np.uint8)
            if u8.ndim == 3:
                u8 = u8[..., ::-1]
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), u8)
    return a, b


def test_renders_the_morph_and_the_landed_word_reads(tmp_path):
    a, b = _write_two_words(tmp_path / "t")
    p = _piece(transitions=[None])
    CM.weave(p, {"bpm": 90.0}, CHK, (W, H), designed=True)
    assert p["plan"]["contour_morph"]["words"] == [0]
    matte = np.zeros((NF, H, W), np.float32)
    matte[:, 20:70, 60:90] = 1.0                                                   # a subject that stays right of centre
    foot = np.full((NF, H, W, 3), 0.55, np.float32)
    foot = np.clip(foot * 255 + 0.5, 0, 255).astype(np.uint8)
    u8, _st = RE.run_compose(p, "s1", type_dir=str(tmp_path / "t"), footage=foot, matte=matte, backend="numpy", threads=0, frames=(66, 104))
    f = np.asarray(u8, np.uint8).astype(np.float32) / 255.0
    assert np.isfinite(f).all() and f.shape[1:3] == (H, W)
    land = f[100 - 66]                                                               # 4.17 s: B landed 0.27 s ago, still inside its section
    inside, outside = land[b > 0.5].mean(), land[(b < 0.5) & (matte[0] < 0.5)].mean()
    assert inside > outside + 0.15                                                   # B's letters are windows onto the shot on a darker ground: they read
    mid = f[75 - 66]                                                                 # 3.12 s: mid-morph, the picture is the fx (dark ground), not the old chain
    base = copy.deepcopy(p)
    base["compose"]["default"]["ops"] = [o for o in p["compose"]["default"]["ops"] if not o.get(CM.FLAG)]
    ub, _ = RE.run_compose(base, "s1", type_dir=str(tmp_path / "t"), footage=foot, matte=matte, backend="numpy", threads=0, frames=(75, 76))
    assert float(np.abs(mid - np.asarray(ub[0], np.float32) / 255.0).mean()) > 0.02


def test_a_specific_text_look_keeps_its_own_transitions_auto_gets_the_morph():
    for look in ("strand", "mech", "ring", "particles"):
        p = _piece()
        p["plan"]["look"] = look
        assert CM.plan(p, HOLDS, {"bpm": 90.0}) == [], look                           # the family's own word-to-word transition plays
        assert CM.plan(p, HOLDS, {"bpm": 90.0}, designed=True) == [], look
        CM.weave(p, {"bpm": 90.0}, CHK, (W, H))
        assert not any(o.get(CM.FLAG) for o in p["compose"]["default"]["ops"])
    p = _piece()
    p["plan"]["look"] = "auto"
    assert len(CM.plan(p, HOLDS, {"bpm": 90.0})) == 1
    p = _piece()
    assert len(CM.plan(p, HOLDS, {"bpm": 90.0})) == 1                                # no look recorded = auto


def test_a_named_contour_morph_wins_over_a_specific_look():
    from tools.fusion import design_step as DS
    p = _piece(transitions=["contour_morph"])
    p["plan"]["look"] = "strand"
    assert len(CM.plan(p, HOLDS, {"bpm": 90.0}, designed=True)) == 1                  # named in type.transitions
    q = _piece(transitions=["contour_morph"])
    q["plan"]["look"] = "slab"
    DS._morph_names_to_null(q)                                                       # what design_step._finish does before the engine sees the reply
    assert q["type"]["transitions"] == [None] and q["plan"]["contour_morph_asked"] == [0]
    assert len(CM.plan(q, HOLDS, {"bpm": 90.0}, designed=True)) == 1
    r = _piece(transitions=["braid"])
    r["plan"]["look"] = "slab"
    assert CM.plan(r, HOLDS, {"bpm": 90.0}, designed=True) == []
