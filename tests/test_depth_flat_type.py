"""v016: flat type + the transition's own runner matte (tools/fusion/depth_flat_type.py, PivotScene.render_matte_t).
CPU only, synthetic layers. Run: python -m pytest tests/test_depth_flat_type.py -q"""
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

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402
from test_depth_orbit import _layer, W, H  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    layers = [_layer(s, p) for s in (1, 2)]
    op = do.OrbitParams(mode="pivot", splat="billboard", max_chunks=30, bg_chunks=12, in_angle=45, out_angle=40, hold1=24, hold2=48)
    return layers, do.PivotScene(layers, op)


# ----------------------------------------------------------------------------------------------- the matte pass
def test_matte_pass_home_frames_equal_the_source_matte(scene):
    layers, s = scene
    m0 = s.render_matte_t(0.0)
    m1 = s.render_matte_t(float(s.op.travel))
    assert m0.shape == (H, W) and m0.dtype == np.float32
    assert np.abs(m0 - layers[0].subj.astype(np.float32)).max() <= 1.0 / 255 + 1e-6
    assert np.abs(m1 - layers[1].subj.astype(np.float32)).max() <= 1.0 / 255 + 1e-6


def test_matte_pass_follows_the_runner_pixels_mid_turn(scene):
    layers, s = scene
    t = 8.0
    q = s.q(t)
    assert 0.0 < q < 0.47                                                 # image 1 only (image 2 not yet in)
    m = s.render_matte_t(t, s.nsub_at(t))
    with_r = s.render_t(t, s.nsub_at(t)).astype(int)
    c = s.cards[id(s.rl1)]
    full = c["tex"].copy()
    tex = full.reshape(2, H, W, 4).copy()
    tex[0, ..., 3] = 0.0                                                  # the colour frame without the runner card
    c["tex"] = tex.reshape(-1, 4)
    try:
        without = s.render_t(t, s.nsub_at(t)).astype(int)
    finally:
        c["tex"] = full
    runner_colour = np.abs(with_r - without).sum(-1) > 24                  # where the runner's colour is in the frame
    mm = m > 0.5
    iou = (runner_colour & mm).sum() / max((runner_colour | mm).sum(), 1)
    assert mm.sum() > 20 and iou > 0.8
    # the flag is back off: the colour render is unchanged by having run a matte pass
    assert np.array_equal(s.render_t(t, s.nsub_at(t)).astype(int), with_r)


def test_matte_pass_is_blurred_where_the_colour_is_blurred(scene):
    _, s = scene
    soft = lambda m: float(((m > 0.03) & (m < 0.97)).sum())                # noqa: E731
    home = soft(s.render_matte_t(0.0))
    t = 8.0
    moving = soft(s.render_matte_t(t, s.nsub_at(t)))
    sharp = soft(s.render_matte_t(t, 1) if s.blur_mode() != "continuous" else s.render_matte_t(t, 1))
    assert moving > 1.5 * max(home, 1.0)                                   # the motion-blurred runner edge is soft
    assert moving >= sharp - 1e-6


def test_matte_pass_has_no_hole_fill_and_fades_with_the_images(scene):
    _, s = scene
    for t in (8.0, 20.0, 30.0):
        m = s.render_matte_t(t, s.nsub_at(t))
        assert m.min() >= 0.0 and m.max() <= 1.0
        assert (m < 0.01).mean() > 0.3                                     # background cards are black, holes are 0


# ----------------------------------------------------------------------------------------------- flat type on the pan transition
from tools.fusion import depth_flat_type as ft_mod  # noqa: E402

FRAME_W, FRAME_H = 96, 160


@pytest.fixture(scope="module")
def flat():
    ft = ft_mod.FlatType("P06:SEARCHING,A LIGHT", 108, 24, 24, 77, "left", 0.5, 4.0, size=(576, 1024))
    return ft


def _mass(m):
    return float(m.sum()) / m.size


def _cx(m):
    xs = np.nonzero(m.sum(0) > 0.5)[0]
    return float((m.sum(0) * np.arange(m.shape[1])).sum() / max(m.sum(), 1e-9)) if len(xs) else float("nan")


def test_word_a_is_formed_on_the_hold_and_exits_to_the_right(flat):
    f = flat
    hold = [f.matte(i, 96, step_px=1e9) for i in (0, 8, 23)]
    assert all(_mass(m) > 0.005 for m in hold)                           # SEARCHING is ONE line now (spec: lines break only at spaces, 2026-10-29), so a thin dotted word is ~0.009 of the frame (was 0.02+ as SEA / RCH / ING)
    assert np.abs(hold[0] - hold[2]).max() < 0.02                         # nothing moves on the hold (camera at rest, word formed; 5/255 engine jitter)
    cx = [_cx(f.matte(i, 96, step_px=1e9)) for i in (24, 30, 34, 38)]
    assert all(b > a for a, b in zip(cx, cx[1:]))                         # the picture slides to the RIGHT
    assert _mass(f.matte(46, 96, step_px=1e9)) < 0.002                    # off the right edge before image 2 is visible (frame 41 +)
    assert f.pan["dir"] == "left"                                         # camera dir left = content right


def test_word_b_enters_from_the_left_only_after_the_landing(flat):
    f = flat
    landing = 24 + 36 - 1
    for i in range(46, landing + 1):                                      # invisible until the depth video has landed
        assert _mass(f.matte(i, 96, step_px=1e9)) < 0.002, i
    seen = [i for i in range(landing + 1, 108) if _mass(f.matte(i, 96, step_px=1e9)) > 0.0003]
    assert seen and seen[0] <= landing + 9                                # its first strands come in right after the landing
    first = f.matte(seen[0], 96, step_px=1e9)
    xs = np.nonzero(first.sum(0) > 0.5)[0]
    assert xs.max() < first.shape[1] * 0.35                               # first seen at the LEFT edge
    cx = [_cx(f.matte(i, 96, step_px=1e9)) for i in (70, 72, 74, 76)]
    assert all(b > a for a, b in zip(cx, cx[1:]))                         # and moves right into place
    end = [f.matte(i, 96, step_px=1e9) for i in (80, 100, 107)]
    assert _mass(end[0]) > 0.02 and np.abs(end[1] - end[2]).max() < 0.02  # formed and still: readable to the end


def test_type_stays_flat_2d_shapes_do_not_change_under_the_pan(flat):
    f = flat
    a, b = f.matte(26, 96, step_px=1e9), f.matte(30, 96, step_px=1e9)
    ys, xs = np.nonzero(a > 0.5)
    ys2, xs2 = np.nonzero(b > 0.5)
    assert abs((ys.max() - ys.min()) - (ys2.max() - ys2.min())) <= 2      # same height: no perspective, no scale
    dx = int(round(xs2.min() - xs.min()))
    assert dx > 0
    sh = np.roll(a, dx, axis=1)
    inter = ((sh > 0.5) & (b > 0.5)).sum()
    assert inter / max(((sh > 0.5) | (b > 0.5)).sum(), 1) > 0.93          # b is a pure horizontal translation of a


def test_runner_matte_is_in_front_of_the_flat_type(flat):
    f = flat
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(108)]
    probe = ft_mod.Composer(f, frames, [np.zeros((FRAME_H, FRAME_W), np.float32)] * 108)
    pm, a = probe.type_layer(10)
    cols = np.nonzero(a.sum(0) > 0.5)[0]
    mid = int((cols.min() + cols.max()) / 2)
    matte = np.zeros((FRAME_H, FRAME_W), np.float32)
    matte[:, :mid] = 1.0                                                  # a runner covering the left half of the word
    comp = ft_mod.Composer(f, frames, [matte] * 108)
    out = comp.frame(10).astype(int)
    foot = frames[10].astype(int)
    _, a = comp.type_layer(10)
    inside = (matte > 0.99) & (a > 0.5)
    outside = (matte < 0.01) & (a > 0.5)
    assert inside.sum() > 20 and outside.sum() > 20
    assert np.abs(out - foot)[matte > 0.99].max() <= 1                    # where the runner is: the video, the letters are behind him
    assert np.abs(out - foot)[outside].max() > 40                         # elsewhere the letters are drawn
    assert comp.report[0]["hidden_by_runner"] <= 0.5


def test_pan_entry_win_defaults_are_unchanged():
    from mograph.modules.transitions.pan import Pan
    p = Pan(dir="left")
    assert p.entry_win == (0.0, 0.65) and p.exit_win == (0.35, 1.0)


def test_static_axes_follow_the_piece_not_the_pan_default():
    assert ft_mod._static_axes({"ELSH": [30, 70, "beat"], "ROND": 0}) == {"ELSH": 50.0, "ROND": 0.0}
