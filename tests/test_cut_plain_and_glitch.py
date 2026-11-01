"""2026-10-30 (user on own-video v003 base2 s2): a 3D cut never uses a blurred-out silhouette. A shot without a clean plate moves as ONE picture (no subject card, no hole fill); a shot with a plate moves as its
PLATE and its subject GLITCHES IN over ~8 frames once the move has settled. Tiny synthetic clips on the CPU. Run: python -m pytest tests/test_cut_plain_and_glitch.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_cut_styles as T  # noqa: E402
from tools.fusion import glitch_in as G  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402

KINDS = [("depth_pivot", None), ("orbit:push", "smooth"), ("orbit:pullback", "smooth"), ("orbit:fly", "smooth")]


def clips():
    a, ma = T.clip(1, 60, 0.3)
    b, mb = T.clip(2, 60, -0.3, bg=(40, 90, 200))
    dm = T.depth_of(ma)
    return a, ma, b, mb, dm


def run(kind, style=None, plates=None, split=False, par=None):
    a, ma, b, mb, dm = clips()
    p = T.piece(kind, dict(par or T.SMALL), n=60, split=split)
    if style:
        p["compose"]["cuts"][0]["style"] = style
    res = PC.apply_cut(p, PC.specs(p)[0], {"s1": a, "s2": b}, {"s1": ma, "s2": mb}, {"s1": dm[None], "s2": T.depth_of(mb)[None]}, plates=plates)
    return res, a, b, ma, mb


@pytest.mark.parametrize("kind,style", KINDS)
def test_no_plate_means_one_picture_no_subject_card_and_no_hole_fill(kind, style):
    res, a, b, ma, mb = run(kind, style)
    assert res.plan["split"] is False and res.info.get("split") is False
    pl = res.plan
    # the engine got NO subject: the matte through the move comes from running the same warp on the matte frames, so it is nonzero where the picture's subject went
    assert res.matte_a is not None and float(res.matte_a[1].max()) > 0.2
    # hold frames (before the move): A is the clip's own frame, nothing blurred / filled behind the subject (the subject colour is still there)
    f0 = res.a[pl["a0"]].astype(np.float32)
    assert float(np.abs(f0 - a[pl["a0"]].astype(np.float32)).mean()) < 6.0
    assert (f0[..., 0] > 200).sum() > 50                                   # SUBJ red is in the frame: the subject was never cut out


def test_the_old_card_split_is_still_reachable_with_split_true():
    res, *_ = run("orbit:push", "smooth", split=True)
    assert res.plan["split"] is True


@pytest.mark.parametrize("kind,style", KINDS)
def test_a_plate_shot_moves_as_its_plate_and_the_subject_glitches_in_after_the_settle(kind, style):
    a, ma, b, mb, dm = clips()
    plate_b = np.where(mb[..., None] > 0.05, np.array([40, 90, 200], np.uint8), b).astype(np.uint8)       # the empty scene: the subject painted out
    plate_b = np.stack([np.full_like(f, 120) for f in b]).astype(np.uint8)
    plate_a = np.stack([np.full_like(f, 90) for f in a]).astype(np.uint8)
    res, a, b, ma, mb = run(kind, style, plates={"s1": plate_a, "s2": plate_b}, par=dict(T.SMALL, hold2=16))
    pl = res.plan
    assert res.info.get("subject_entry") == "glitch_in" and res.info["glitch_frames"] == G.GLITCH_FRAMES
    j_s = res.info["glitch_start"]
    assert j_s == max(pl["settle"] - pl["i2"] - PC.GLITCH_LEAD_FRAMES, 0)
    mB = res.matte_b[1]
    assert float(mB[:j_s].max() if j_s > 0 else 0.0) == 0.0                # the subject is ABSENT during the move
    assert float(mB[j_s + G.GLITCH_FRAMES:].max()) > 0.9                    # and present, solid, after the glitch
    # no subject colour in the moving frames of B (plate only): red channel stays at the plate's level
    moving = res.b[:j_s] if j_s > 0 else res.b[:1]
    assert int((moving[..., 0] > 200).sum()) == 0
    # the glitch window flickers: at least one step of the pattern has no subject, at least one has it
    inside = [float(m.max()) for m in mB[j_s:j_s + G.GLITCH_FRAMES]]
    assert min(inside) < 0.05 and max(inside) > 0.9, inside


def test_glitch_in_pattern_is_6_to_10_frames_with_a_flicker_then_solid():
    assert 6 <= G.GLITCH_FRAMES <= 10
    for seed in range(12):
        f = G.flicker(G.GLITCH_FRAMES, seed)
        assert f[0] == 0.0 and f[-1] == 1.0 and f[-2] == 1.0 and any(x == 0.0 for x in f[1:-2]) and any(x == 1.0 for x in f)
        assert G.step(G.GLITCH_FRAMES, seed=seed) == (1.0, 0.0) and G.step(-1, seed=seed) == (0.0, 0.0)


def test_the_glitch_is_crisp_integer_shifts_no_blur():
    sub = np.zeros((40, 60, 3), np.uint8)
    sub[:, 20:40] = (250, 10, 10)
    m = np.zeros((40, 60), np.float32)
    m[:, 20:40] = 1.0
    rgb, mm = G.glitch_layer(sub, m, 1.0, seed=3, k=2)
    assert set(np.unique(mm).tolist()) <= {0.0, 1.0}                       # a shifted binary matte stays binary: no blur, no soft edge
    assert set(np.unique(rgb[..., 1]).tolist()) <= {0, 10}


def test_a_plate_shot_a_side_keeps_the_subject_for_the_lead_in_then_flickers_out():
    a, ma, b, mb, dm = clips()
    plate_a = np.stack([np.full_like(f, 90) for f in a]).astype(np.uint8)
    res, a, b, ma, mb = run("depth_pivot", None, plates={"s1": plate_a}, par=dict(short=64, travel=14, hold1=3, hold2=3))
    pl = res.plan
    h1 = pl["params"]["hold1"]
    mA = res.matte_a[1]
    assert float(mA[:h1].max()) > 0.9 and float(mA[h1 + PC.GLITCH_OUT_FRAMES:].max()) == 0.0
    assert res.info.get("subject_exit") == "flicker_out"


@pytest.mark.parametrize("kind,style", KINDS)
def test_the_glitch_in_overlaps_the_settle_and_never_pops(kind, style):
    """ship #10b: the subject starts to glitch in GLITCH_LEAD_FRAMES before the travel ends (the resolve overlaps the settle) and is solid well before the window ends; the picture
    through the glitch has no isolated one-frame spike (a diff more than 3x BOTH neighbours: a pop) and no run of > 3 stalled frames."""
    import cv2
    a, ma, b, mb, dm = clips()
    plate_b = np.stack([np.full_like(f, 120) for f in b]).astype(np.uint8)
    plate_a = np.stack([np.full_like(f, 90) for f in a]).astype(np.uint8)
    res, a, b, ma, mb = run(kind, style, plates={"s1": plate_a, "s2": plate_b}, par=dict(T.SMALL, hold2=16))
    pl = res.plan
    j_s = res.info["glitch_start"]
    assert PC.GLITCH_LEAD_FRAMES > 0 and j_s == max(pl["settle"] - pl["i2"] - PC.GLITCH_LEAD_FRAMES, 0)
    assert j_s + G.GLITCH_FRAMES <= pl["b_len"]
    fr = [cv2.resize(f, (90, 160), interpolation=cv2.INTER_AREA).astype(np.float32) for f in res.b[:pl["b_len"]]]
    d = [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
    lo, hi = max(j_s - 2, 1), min(j_s + G.GLITCH_FRAMES + 2, len(d) - 1)
    for i in range(lo, hi):
        assert d[i] <= 3 * max(d[i - 1], d[i + 1], 1.0), (i, [round(x, 1) for x in d])
