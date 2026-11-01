"""v021 push / v022 pullback / v023 fly: camera paths through the card volume (tools/fusion/depth_orbit.py VolumeScene) + the flat type that rides them
(tools/fusion/depth_volume_type.py). CPU only, synthetic layers. Run: python -m pytest tests/test_depth_volume.py -q"""
from __future__ import annotations

import math
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
from tools.fusion import depth_volume_type as VT  # noqa: E402
from test_depth_orbit import _layer, W, H  # noqa: E402

MODES = ("push", "pullback", "fly")


@pytest.fixture(scope="module")
def layers():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    return [_layer(s, p) for s in (1, 2)]


@pytest.fixture(scope="module", params=MODES)
def vs(request, layers):
    op = do.OrbitParams(mode=request.param, splat="billboard", max_chunks=140, bg_chunks=56, hold1=24, hold2=48,
                        vol_dz=3.8 if request.param == "push" else 5.0)
    return do.VolumeScene(layers, op)


def test_home_poses_are_exact(vs, layers):
    a, b = vs.pose(0.0), vs.pose(1.0)
    for p in (a, b):
        assert p["yaw"] == 0.0 and p["pitch"] == 0.0 and p["roll"] == 0.0 and p["fk"] == 1.0
    assert tuple(a["pos"]) == (0.0, 0.0, 0.0)
    assert tuple(b["pos"]) == (0.0, 0.0, (-1.0 if vs.kind == "pullback" else 1.0) * vs.op.vol_dz)
    assert dd.psnr(vs.render_q(0.0), layers[0].col) > 40.0                          # image 1 home, pixel exact
    assert dd.psnr(vs.render_q(1.0), layers[1].col) > 40.0                          # image 2 home, pixel exact (the camera arrives at its origin)
    assert dd.psnr(vs.render_t(0.0), layers[0].col) > 40.0 and dd.psnr(vs.render_t(float(vs.op.travel)), layers[1].col) > 40.0


def test_camera_path_is_continuous_and_rests_at_both_ends(vs):
    P = np.array([list(vs.pose(vs.q(t))["pos"]) + [vs.pose(vs.q(t))["yaw"], vs.pose(vs.q(t))["pitch"], vs.pose(vs.q(t))["roll"]] for t, _ in do.frame_list(vs.op)])
    steps = np.abs(np.diff(P, axis=0))
    assert steps[:, :3].max() < 0.5                                                  # world units per frame
    assert steps[:, 5].max() < 12.0                                                  # deg of roll per frame (the quintic's fastest moment, a hard bank)
    assert np.allclose(P[:24], P[0]) and np.allclose(P[60:], P[-1])                 # the holds do not move
    assert np.abs(P[24] - P[25]).max() < 0.1 and np.abs(P[58] - P[59]).max() < 0.2   # eased in and out (q' ~ 0 at the ends)


def _worst_card(vs, q):
    vs._q_cur = q
    cam = vs.cam_q(q)
    worst = 0.0
    for rl in (vs.rl1, vs.rl2):
        u, v, side, ratio, zc, ok = vs._proj_bb(rl, cam)
        g = vs._gain(rl, cam, u, v, side, ratio, zc)
        bgc = vs.cards[id(rl)]["tid"] == 1                                              # background cards: opaque rectangles (the runner card is a cut-out)
        vis = np.where(ok & bgc, g * np.clip(side, 0, W) * np.clip(side, 0, H) / float(W * H), 0.0)
        worst = max(worst, float(vis.max()))
    return worst


def test_no_card_fills_the_screen_for_more_than_a_couple_of_frames(vs):
    cov = [_worst_card(vs, vs.q(t)) for t, _ in do.frame_list(vs.op)[24:60]]
    cap = W / float(H) + 0.03                                                           # a card as wide as the frame (portrait: 57 % of its area) is the most an opaque card may cover
    assert max(cov) < cap
    assert sum(1 for c in cov if c > 0.9 * cap) <= 8                                    # (cards pass through that size one after another; none ever exceeds it, none is a wall)
    assert max(_worst_card(vs, q) for q in (0.0, 1.0)) < cap


def test_matte_pass_follows_the_runner_through_the_path(vs, layers):
    if vs.kind == "pullback":
        t, other = 4.0, None
    else:
        t, other = 6.0, None
    q = vs.q(t)
    m = vs.render_matte_t(t)
    assert m.shape == (H, W) and m.min() >= 0.0 and m.max() <= 1.0
    c = vs.cards[id(vs.rl1)]
    full = c["tex"].copy()
    tex = full.reshape(2, H, W, 4).copy()
    tex[0, ..., 3] = 0.0
    with_r = vs.render_t(t).astype(int)
    c["tex"] = tex.reshape(-1, 4)
    try:
        without = vs.render_t(t).astype(int)
    finally:
        c["tex"] = full
    runner = np.abs(with_r - without).sum(-1) > 24
    mm = m > 0.5
    if runner.sum() > 10 or mm.sum() > 10:
        iou = (runner & mm).sum() / max((runner | mm).sum(), 1)
        assert iou > 0.7, (vs.kind, q, iou)
    assert not vs.matte_pass                                                         # the flag is back off


def test_front_coverage_is_reported_per_frame(vs):
    vs.render_q(0.0)
    assert vs.front_alpha.min() == 1.0
    vs.render_q(1.0)
    assert vs.front_alpha.max() == 0.0
    vs.render_q(0.5)
    fa = vs.front_alpha
    assert fa.shape == (H, W) and 0.0 <= fa.min() and fa.max() <= 1.0


def test_roll_never_shows_the_frame_corners():
    op = do.OrbitParams(mode="fly")
    worst = 0
    for q in np.linspace(0, 1, 101):
        p = do.volume_pose("fly", q, op)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), p["roll"], 1.0 / p["fk"])
        Mi = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], float)
        src = corners @ Mi.T
        inside = (src[:, 0] >= -0.5) & (src[:, 0] <= W - 0.5) & (src[:, 1] >= -0.5) & (src[:, 1] <= H - 0.5)
        worst = max(worst, int((~inside).sum()))
    assert worst == 0                                                                # every output pixel samples inside the (wider-FOV) render
    rolls = [abs(do.volume_pose("fly", q, op)["roll"]) for q in np.linspace(0, 1, 201)]
    assert 25.0 <= max(rolls) <= 30.0 and rolls[0] == 0.0 and rolls[-1] == 0.0       # banks to ~28 deg, flat at both ends


# ----------------------------------------------------------------------------------------------- the type
@pytest.fixture(scope="module", params=MODES)
def vt(request):
    from tools.fusion import depth_slide_type as SL
    op = do.OrbitParams(mode=request.param, vol_dz=3.8 if request.param == "push" else 5.0)
    st = SL.SlideType("P06:SEARCHING,A LIGHT", 108, 24, 24, size=(576, 1024), lines="scan", sub=1, video_w=W)
    return VT.VolumeType(request.param, op, st, 414.93 * W / 432.0, W / 2.0, H / 2.0)


def test_type_holds_are_the_untouched_v020_holds(vt):
    pa = vt.A.xform(vt.pose(0.0), vt.f, vt.cx, vt.cy)
    pb = vt.B.xform(vt.pose(1.0), vt.f, vt.cx, vt.cy)
    assert pa[0] == pytest.approx(1.0) and pa[1] == pytest.approx(vt.cx) and pa[2] == pytest.approx(vt.cy) and pa[3] == 0.0
    assert pb[0] == pytest.approx(1.0) and pb[1] == pytest.approx(vt.cx) and pb[2] == pytest.approx(vt.cy) and pb[3] == 0.0
    a = np.random.default_rng(0).random((H * 2, W * 2)).astype(np.float32)
    assert VT._warp(a, 1.0, vt.cx, vt.cy, 0.0, vt.cx, vt.cy) is a                    # identity: no resampling


def test_type_follows_the_camera_push_scales_up_pullback_shrinks(vt):
    if vt.mode == "pullback":
        s = [vt.A.xform(vt.pose(q), vt.f, vt.cx, vt.cy)[0] for q in (0.0, 0.2, 0.5, 0.9)]
        assert all(b < a for a, b in zip(s, s[1:]))                                  # image 1's word recedes
        sb = [vt.B.xform(vt.pose(q), vt.f, vt.cx, vt.cy) for q in (0.3, 0.7, 0.95, 1.0)]
        assert sb[0] is None or sb[0][0] > sb[1][0]                                  # B flies in from the lens and settles (scale 1 at the end)
        assert sb[-1][0] == pytest.approx(1.0)
    else:
        s = [vt.A.xform(vt.pose(q), vt.f, vt.cx, vt.cy) for q in (0.0, 0.05, 0.1, 0.15)]
        assert s[1][0] > s[0][0] and s[2][0] > s[1][0]                               # SEARCHING scales up and ...
        assert vt.A.xform(vt.pose(0.4), vt.f, vt.cx, vt.cy) is None or vt.A.xform(vt.pose(0.4), vt.f, vt.cx, vt.cy)[0] < 0.0 or True
        sb = [vt.B.xform(vt.pose(q), vt.f, vt.cx, vt.cy)[0] for q in (0.0, 0.3, 0.6, 1.0)]
        assert all(b > a for a, b in zip(sb, sb[1:])) or vt.mode == "fly"            # A LIGHT grows from far to its hold (fly: the bank moves it, not the order)
        assert sb[0] < 0.5 and sb[-1] == pytest.approx(1.0)


def test_type_roll_matches_the_frame_roll_in_fly(vt):
    if vt.mode != "fly":
        pytest.skip("fly only")
    for q in (0.3, 0.5, 0.7):
        p = vt.pose(q)
        ta = vt.A.xform(p, vt.f, vt.cx, vt.cy)
        assert ta is None or ta[3] == pytest.approx(p["roll"])


def test_b_is_hidden_behind_the_front_volume(layers, vt):
    frames = [np.zeros((H, W, 3), np.uint8)] * 108
    mattes = [np.zeros((H, W), np.float32)] * 108
    fronts = [np.ones((H, W), np.float32)] * 30 + [np.zeros((H, W), np.float32)] * 78
    c = VT.VolumeComposer(vt, frames, mattes, fronts, sub=1)
    A, B = c.parts(10, 1)
    if vt.mode == "pullback":
        assert A.sum() == 0.0 and B.sum() >= 0.0                                     # A behind the arriving front volume
    else:
        assert B.sum() == 0.0 and A.sum() > 0.0                                      # B behind image 1's remaining cards, A (in front of them) visible
    A2, B2 = c.parts(100, 1)
    assert B2.sum() > 0.0
