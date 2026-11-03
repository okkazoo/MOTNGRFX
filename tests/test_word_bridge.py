"""WORD BRIDGE = the CONTOUR LEG of a library word-to-word transition (tools/fusion/word_bridge.py, 2026-11-01 v2): A's letter outlines -> the subject's live
contour -> exactly where the LIBRARY module's own strands are at the hand-over (read from the type pass); compose never draws a pattern of its own.
Synthetic type pass on disk (manifest + matte PNGs), CPU numpy. Run: python -m pytest tests/test_word_bridge.py -q"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402
from tools.fusion import word_bridge as WB  # noqa: E402

W, H, FPS, NF = 320, 180, 24, 120
A_HIT, A_REL, A_END, B_HIT, B_REL = 0.3, 1.5, 2.0, 3.6, 4.6
TR_START, TR_DUR = 2.0, 1.2                       # the library transition (type pass): hand-over Th = 2.0 + 0.4 x 1.2 = 2.48 s


def _letters(boxes):
    m = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
        cv2.rectangle(m, (x0 + 8, y0 + 10), (x1 - 8, y1 - 10), 0, -1)      # a counter: two outlines per letter
    return m


def _strands():
    """what the library module draws mid-way (thin lines away from both words)."""
    m = np.zeros((H, W), np.uint8)
    for k in range(6):
        cv2.line(m, (200, 20 + 6 * k), (300, 40 + 6 * k), 255, 2)
    return m


A_BOXES = [(20, 60, 60, 120), (75, 60, 115, 120)]
B_BOXES = [(150, 55, 190, 125), (205, 55, 245, 125), (260, 55, 300, 125)]


@pytest.fixture()
def type_dir(tmp_path):
    d = tmp_path / "type"
    (d / "matte").mkdir(parents=True)
    ma, mb, ms = _letters(A_BOXES), _letters(B_BOXES), _strands()
    for f in range(NF):
        t = f / FPS
        m = ma if t < A_END else (ms if t < B_HIT - 0.4 else mb)
        cv2.imwrite(str(d / "matte" / f"t_{f + 1:05d}.png"), m)
    man = {"fps": FPS, "spec": {"beats": [i * 0.5 for i in range(12)]},
           "info": {"words": [{"text": "AB", "hit": A_HIT, "release": A_REL, "end": A_END},
                              {"text": "CDE", "hit": B_HIT, "release": B_REL, "end": B_REL + 0.4}],
                    "transitions": [{"module": "spiral_around", "start": TR_START, "dur": TR_DUR}]}}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return str(d)


def _piece(contour=None, matte=True, transitions=None, flag=None, seed=3):
    ops = [{"op": "sandwich", "out": "image", "args": {"bg": "$footage", "type_rgb": "$type_beauty", "type_a": "$type_matte", "matte": "$matte" if matte else None}},
           {"op": "grade", "out": "image", "args": {"img": "$image"}}]
    p = {"fps": FPS, "shots": [{"id": "s1", "frames": NF, "matte": matte}], "layout": {"s1": {"start": 0, "frames": NF, "xf": 0}},
         "type": {"seed": seed, "words": [{"text": "AB"}, {"text": "CDE"}]}, "compose": {"default": {"ops": ops}}, "plan": {"piece_system": {"seed": seed}}}
    if transitions is not None:
        p["type"]["transitions"] = transitions
    if contour is not None:
        p["compose"]["word_bridge_contour"] = contour
    if flag is not None:
        p["compose"]["word_bridge"] = flag
    return p


def _dist_to_outline(pts, mask):
    edge = cv2.Canny(mask, 50, 150)
    dt = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    p = np.clip(np.round(pts.reshape(-1, 2)).astype(int), 0, [W - 1, H - 1])
    return dt[p[:, 1], p[:, 0]]


def test_never_a_pattern_of_its_own():
    """the invented compose patterns (R16b e3 / e7, R17) are gone for good."""
    assert not hasattr(WB, "pattern") and not hasattr(WB, "flight")


def test_plan_takes_the_leg_only_with_a_subject_matte(type_dir):
    ev = WB.plan(_piece(contour="subject"), type_dir)
    assert len(ev) == 1 and ev[0]["kind"] == "spiral_around" and ev[0]["contour"]["src"] == "subject"
    assert ev[0]["rel"] == A_REL and abs(ev[0]["th"] - (TR_START + WB.HANDOVER * TR_DUR)) < 1e-3
    assert [p[:2] for p in ev[0]["phases"]] == [["a", "c"], ["c", "c"], ["c", "p"]]
    assert WB.plan(_piece(contour="subject", matte=False), type_dir) == []           # no matte: the type pass alone (connected library transition)
    assert WB.plan(_piece(contour="subject", flag=False), type_dir) == []
    assert WB.plan(_piece(contour=False), type_dir) == []
    assert WB.plan(_piece(contour="subject", transitions=["cut"]), type_dir) == []
    picks = [bool(WB.plan(_piece(seed=s), type_dir)) for s in range(40)]             # ON by default, seeded about half the time
    assert 8 < sum(picks) < 32


def test_leg_starts_on_a_and_hands_over_to_the_library_strands(type_dir):
    ev = WB.plan(_piece(contour="subject"), type_dir)[0]
    K, A, P, rank = WB.keys_for(ev, type_dir + "/matte", W, H)
    s_a, _ = WB.state_at(ev, ev["s0"], A, P, rank)
    s_p, _ = WB.state_at(ev, ev["th"], A, P, rank)
    assert _dist_to_outline(s_a, _letters(A_BOXES)).max() <= 2.5                     # the lines start ON A's letters
    assert np.percentile(_dist_to_outline(s_p, _strands()), 95) <= 2.5                # ... and end ON the library module's own strands at Th
    prev = None
    for f in range(int(ev["rel"] * FPS), int(ev["th"] * FPS) + 1):                     # continuous
        cur, _ = WB.state_at(ev, f / FPS, A, P, rank)
        if prev is not None:
            assert np.abs(cur - prev).max() < 0.25 * W
        prev = cur


def test_contour_leg_hugs_the_subject(type_dir):
    ev = WB.plan(_piece(contour="subject"), type_dir)[0]
    sub = np.zeros((H, W), np.uint8)
    cv2.ellipse(sub, (W // 2, H // 2), (60, 75), 0, 0, 360, 255, -1)
    a_, b_, t0, t1 = ev["phases"][1]
    t = 0.5 * (t0 + t1)
    fr = np.full((H, W, 3), 0.3, np.float32)
    out = WB.word_bridge(fr.copy(), footage=fr, matte=sub.astype(np.float32) / 255.0, frame_idx=t * FPS, events=[ev], sid="s1", type_dir=type_dir + "/matte")
    drawn = np.abs(out - fr).max(-1) > 0.05
    ring = cv2.dilate(cv2.Canny(sub, 50, 150), np.ones((7, 7), np.uint8)) > 0
    assert drawn.sum() > 200 and (drawn & ring).sum() / drawn.sum() > 0.9
    small = np.zeros((H, W), np.uint8)                                                  # a far-away subject: straight to the strands, no tiny contour drawing
    cv2.circle(small, (W // 2, H // 2), 6, 255, -1)
    ev2 = dict(ev, word=7)
    out2 = WB.word_bridge(fr.copy(), footage=fr, matte=small.astype(np.float32) / 255.0, frame_idx=t * FPS, events=[ev2], sid="s1", type_dir=type_dir + "/matte")
    d2 = np.abs(out2 - fr).max(-1) > 0.05
    near = cv2.dilate(small, np.ones((9, 9), np.uint8)) > 0
    assert d2.sum() > 100 and (d2 & near).sum() / d2.sum() < 0.2


def test_gate_ramps_and_nothing_after_the_hand_over(type_dir):
    ev = WB.plan(_piece(contour="subject"), type_dir)[0]
    spec = {"win": WB.gate_windows([ev])}
    one = np.ones((4, 4), np.float32)
    assert PC.apply_gate(one, spec, ev["rel"] - 0.01).mean() == 1.0
    assert 0.0 < PC.apply_gate(one, spec, ev["rel"] + 0.5 * WB.FADE_A_S).mean() < 1.0
    assert PC.apply_gate(one, spec, 0.5 * (ev["rel"] + ev["th"])).mean() == 0.0
    assert 0.0 < PC.apply_gate(one, spec, ev["th"] - 0.5 * WB.FADE_IN_S).mean() < 1.0
    assert PC.apply_gate(one, spec, ev["th"]).mean() == 1.0                              # the library strands carry on from Th
    fr = np.full((H, W, 3), 0.2, np.float32)
    late = WB.word_bridge(fr.copy(), footage=fr, frame_idx=(ev["th"] + WB.OUT_S + 0.05) * FPS, events=[ev], sid="s1", type_dir=type_dir + "/matte")
    assert np.allclose(late, fr)                                                         # compose draws nothing on the library's way to B


def test_recipe_inserts_one_op_before_the_tail(type_dir):
    p = _piece(contour="subject")
    ops = WB.recipe_ops(p, p["compose"]["default"]["ops"], "s1", type_dir)
    assert [o["op"] for o in ops] == ["sandwich", "word_bridge", "grade"]
    assert WB.recipe_ops(p, ops, "s1", type_dir) == ops
    assert "word_bridge" in C._ops_registry()
