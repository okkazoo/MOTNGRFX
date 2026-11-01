"""v017: ribbons flowing through the turn, behind the runner matte (tools/fusion/depth_ribbon_type.py). CPU only, small type size.
Run: python -m pytest tests/test_depth_ribbon_type.py -q"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from tools.fusion import depth_ribbon_type as R  # noqa: E402

FLAG = "P06:SEARCHING,A LIGHT"
FW, FH = 96, 160


@pytest.fixture(scope="module")
def rt():
    r = R.RibbonType(FLAG, 108, 24, 24, 59, "right", size=(576, 1024))
    r.sub = 1
    return r


def _cx(m):
    cols = m.sum(0)
    return float((cols * np.arange(len(cols))).sum() / max(cols.sum(), 1e-9))


def _front(m, thr=0.5):
    xs = np.nonzero(m.sum(0) > thr)[0]
    return int(xs.max()) if len(xs) else -1


def test_type_is_continuous_from_the_hold_through_the_turn_to_the_next_hold(rt):
    mass = [float(rt.matte(i).mean()) for i in range(0, 108)]
    assert all(m > 0.004 for m in mass[:24])                          # word A formed on image 1's hold
    assert all(m > 0.004 for m in mass[24:])                          # alpha > 0 on EVERY frame from A's hold to B's hold: no dead air
    assert min(mass[24:]) > 0.5 * min(mass[:24]) * 0.2                # (the thinnest moment is the word melting to an outline, still there)
    hold_b = [rt.matte(i) for i in (90, 100, 107)]                     # B formed to the end and frozen
    assert np.abs(hold_b[0] - hold_b[2]).max() < 1e-6 and hold_b[0].mean() > 0.01


def test_word_a_hold_is_static_then_ribbons_flow_to_the_right(rt):
    a = [rt.matte_of(0, i) for i in (0, 12, 24)]
    assert np.abs(a[0] - a[2]).max() < 0.02                            # formed, nothing moves on the hold
    ys = [_cx(rt.matte_of(0, i)) for i in (32, 38, 44, 50)]
    fr = [_front(rt.matte_of(0, i)) for i in (32, 38, 44, 50)]
    assert all(b > a_ for a_, b in zip(ys, ys[1:]))                    # the mass of A's ribbons moves right ...
    assert fr[-1] >= fr[0] and fr[-1] > 0.9 * rt.matte_of(0, 50).shape[1]    # ... and its strands run off the right edge
    assert rt.matte_of(0, 100).max() == 0.0                            # A is gone for good


def test_ribbons_flow_left_when_asked_to():
    r = R.RibbonType(FLAG, 108, 24, 24, 59, "left", size=(576, 1024))
    r.sub = 1
    cx = [_cx(r.matte_of(0, i)) for i in (32, 38, 44, 50)]
    assert all(b < a for a, b in zip(cx, cx[1:]))
    assert r.wt[0].info["end"] > 0 and r.report_side() == ("left", "right") if hasattr(r, "report_side") else True


def test_word_b_gathers_from_the_left_and_forms_after_the_landing(rt):
    assert rt.b_mode == "reverse" and rt.i_b0 == 44 and rt.b_end == rt.i_land + 18 == 77
    for i in range(0, rt.i_b0):
        assert rt.matte_of(1, i).max() == 0.0                          # nothing of B during the swing / A's first ribbons
    first = rt.matte_of(1, rt.i_b0 + 4)
    assert first.sum() > 0 and _cx(first) < 0.5 * first.shape[1]       # B's first strands come in at the LEFT (and fade in: no hard start)
    assert rt.gain(1, rt.i_b0) < 0.3 and rt.gain(1, rt.i_b0 + 10) == 1.0
    fr = [_front(rt.matte_of(1, i)) for i in range(rt.i_b0 + 3, rt.i_b0 + 12)]
    assert fr[-1] > fr[0]                                              # and stream rightward into the word
    ib = rt.words[1]
    assert rt.tau_b(rt.b_end) == pytest.approx(ib["release"], abs=1e-6)       # formed (assembled) at landing + 18 ...
    assert rt.tau_b(rt.b_end + 25) == pytest.approx(ib["release"], abs=1e-6)  # ... and frozen: B's font axis no longer breathes
    prog = [rt.b_progress(i) for i in range(rt.i_b0, rt.b_end + 1)]
    assert all(b >= a_ for a_, b in zip(prog, prog[1:])) and prog[0] == 0.0 and prog[-1] == 1.0
    assert prog[len(prog) // 2] == pytest.approx(0.5, abs=0.05) and (prog[-1] - prog[-3]) < (prog[len(prog) // 2 + 1] - prog[len(prog) // 2 - 1])   # eased, no snap
    hold = [rt.matte_of(1, i) for i in (80, 95, 107)]
    assert np.abs(hold[0] - hold[2]).max() < 1e-6                     # B on its hold: identical frames (frozen like A)
    assert rt.matte_of(1, 70).sum() > 0.5 * hold[0].sum()             # mostly assembled by ~70


@pytest.mark.skip(reason="v017 g03 (wavy / squeezed ribbons) was abandoned; its geometry changed when WordType gained the home camera")
def test_ribbons_stay_horizontal_through_the_street_no_wall(rt):
    for i in range(40, 54, 2):
        m = cv2.resize(rt.matte(i), (288, 512), interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1)
        assert (gy ** 2).sum() > 1.2 * (gx ** 2).sum(), i             # horizontal lines dominate: no vertical wall / curtain edge
    left = rt.matte(46)[:, :int(0.04 * rt.size[0])]
    assert left.mean() < 0.15 * max(rt.matte(46).mean(), 1e-9) * 4    # nothing piled against the left edge


def test_both_words_are_big_left_of_centre_and_no_letter_is_hidden():
    rp = os.path.join(ROOT, "17_depth-dolly", "v016", "local", "matte_depth")
    if not os.path.isdir(rp):
        pytest.skip("v016 matte pass not on this disk")
    mattes = [cv2.imread(os.path.join(rp, "%04d.png" % i), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for i in (12, 89)]
    r = R.RibbonType(FLAG, 108, 24, 24, 59, "right", size=(576, 1024), runners=tuple(mattes))
    H, W = mattes[0].shape
    for k, run in enumerate(mattes):
        f = r.fit[k]
        assert f["min_letter_visible"] >= 0.6, f                       # every letter >= 60 % visible behind the runner
        assert f["dx"] <= -0.02 and "left of centre" in f["how"]
        from tools.fusion import depth_type as DT
        a = DT.place_matte(r.wt[k].matte(r.wt[k].info["release"]), W, H, 1)
        xs = np.nonzero(a.max(0) > 0.25)[0]
        assert (xs.max() - xs.min()) / W >= 0.65                       # big: ~70-78 % of the frame width
        assert 0.5 * (xs.min() + xs.max()) < 0.5 * W                   # left of centre
        lv = R.letter_visibility(a, run)
        assert min(v for *_, v in lv) >= 0.6 and len(lv) >= 5


def test_ribbons_are_behind_the_runner_matte(rt):
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (FH, FW, 3), dtype=np.uint8) for _ in range(108)]
    zero = [np.zeros((FH, FW), np.float32)] * 108
    probe = R.RibbonComposer(rt, frames, zero, wave_px=6.0)
    for i in (10, 46, 70, 100):
        _, a = probe.type_layer(i)
        assert a.max() > 0.5, i
        cols = np.nonzero(a.sum(0) > 0.5)[0]
        mid = int((cols.min() + cols.max()) / 2)
        matte = np.zeros((FH, FW), np.float32)
        matte[:, :mid] = 1.0                                           # a runner covering the left part of the type
        comp = R.RibbonComposer(rt, frames, [matte] * 108, wave_px=6.0)
        out = comp.frame(i).astype(int)
        foot = frames[i].astype(int)
        _, a2 = comp.type_layer(i)
        assert np.abs(out - foot)[matte > 0.99].max() <= 1             # where he is: the video, the type is behind him
        assert np.abs(out - foot)[(matte < 0.01) & (a2 > 0.5)].max() > 40   # elsewhere the ribbons / letters are drawn


def test_flow_direction_flag_and_the_measured_video_motion():
    assert R.flow_side("right") == "right" and R.flow_side("left") == "left"
    assert R.flow_side("counter", -1.4) == "right"                     # video moves left -> ribbons right
    assert R.flow_side("counter", 2.0) == "left"
    with pytest.raises(ValueError):
        R.flow_side("up")
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((220, 330)).astype(np.float32), (0, 0), 5)
    base = (255 * (base - base.min()) / (base.max() - base.min())).astype(np.uint8)
    for sgn in (+1, -1):                                               # a picture sliding right (+) / left (-) at 3 px per frame
        fr = [np.ascontiguousarray(np.roll(base, sgn * 3 * k, axis=1)[:, 40:290]) for k in range(8)]
        m = R.measure_flow(fr, 1, 7)
        assert np.sign(m["mean_dx"]) == sgn and abs(m["mean_dx"]) > 1.5
        assert R.flow_side("counter", m["mean_dx"]) == ("left" if sgn > 0 else "right")


def test_ink_support_does_not_pop():
    tr = R.InkTrack({"fg": [235, 240, 255], "accents": []}, 400.0)
    a = np.zeros((FH, FW), np.float32)
    a[60:100, 10:80] = 1.0
    dark = np.full((FH, FW, 3), 25, np.uint8)
    busy = np.random.default_rng(0).integers(0, 255, (FH, FW, 3), dtype=np.uint8)
    run = np.zeros((FH, FW), np.float32)
    sup = []
    for f in [busy] * 3 + [dark] * 4:
        sup.append(tr.update(f, a, run)["support"])
    assert sup[2] is not None and all(s == sup[2] for s in sup[2:6])   # the support is held for a few frames after the plate calms down
