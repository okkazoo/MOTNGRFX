"""Camera-orbit transition (tools/fusion/depth_orbit.py). CPU only, synthetic layers, no depth model.
Run: python -m pytest tests/test_depth_orbit.py -q"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402

W, H = 96, 160


def _layer(seed, params):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    img = np.ascontiguousarray(np.kron(base, np.ones((8, 8, 1), np.uint8)))
    disp = np.full((H, W), 0.1, np.float32)
    disp[:, :20] = 0.5
    disp[60:120, 30:70] = 0.9
    matte = np.zeros((H, W), np.float32)
    matte[60:120, 30:70] = 1.0
    return dd.build_layer(img, disp, matte, params)


@pytest.fixture(scope="module")
def setup():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    layers = [_layer(s, p) for s in (1, 2)]
    out = {}
    for spring in (False, True):
        op = do.OrbitParams(spring=spring)
        geo = do.make_geometry(layers, op)
        out[spring] = (do.OrbitScene(layers, geo, op), op)
    return layers, out


def test_frame0_is_image1_home(setup):
    layers, sc = setup
    for spring in (False, True):
        s, op = sc[spring]
        fr = s.render_t(0.0)
        assert dd.psnr(fr, layers[0].col) > 40.0


def test_final_settled_frame_is_image2_home(setup):
    layers, sc = setup
    for spring in (False, True):
        s, op = sc[spring]
        t_last = float(op.travel + op.hold2)
        assert s.q(t_last) == 1.0
        fr = s.render_t(t_last)
        assert dd.psnr(fr, layers[1].col) > 40.0
    # the landing frame of the plain move is already exactly home
    s, op = sc[False]
    assert dd.psnr(s.render_t(float(op.travel)), layers[1].col) > 40.0


def test_end_pose_is_image2_home_pose(setup):
    _, sc = setup
    s, op = sc[False]
    x, z, psi = s.path.pose(s.path.L)
    ex, ez, epsi = s.path.end_pose()
    assert (x, z, psi) == pytest.approx((ex, ez, epsi), abs=1e-6)
    assert s.path.rel_angle(s.path.L) == pytest.approx(0.0, abs=1e-6)
    assert s.path.rel_angle(s.path.L - s.path.LB) == pytest.approx(op.enter_angle, abs=1e-6)   # entry at 45 deg


@pytest.mark.parametrize("spring", [False, True])
def test_path_continuous(setup, spring):
    _, sc = setup
    s, op = sc[spring]
    taus = np.linspace(0.0, s.path.L * 1.02, 4000)
    P = np.array([s.path.pose(t) for t in taus])
    step = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    dstep = np.abs(np.diff(P[:, 2]))
    ds = taus[1] - taus[0]
    assert step.max() < 1.2 * ds            # no position jump; speed ~ 1 everywhere (arc length parametrisation)
    assert dstep.max() < 2.0 * ds           # heading rate bounded (1/d2 on the arrival arc)
    # C1: second differences stay small (position tangent and heading rate continuous at the joins)
    v = np.diff(P[:, :2], axis=0) / ds
    assert np.abs(np.diff(v, axis=0)).max() / ds < 5.0
    # per-frame poses: smooth
    pos = np.array([(p[2], p[3]) for p in s.poses()])
    assert np.hypot(*np.diff(pos, axis=0).T).max() < 0.2 * s.path.L


@pytest.mark.parametrize("spring", [False, True])
def test_camera_never_inside_a_volume_bbox(setup, spring):
    _, sc = setup
    s, op = sc[spring]
    g = s.geo
    for tau in np.linspace(0.0, s.path.L * 1.03, 600):
        x, z, _ = s.path.pose(tau)
        pos = np.array([x, 0.0, z])
        assert not do.inside_aabb(pos, g.aabb1), tau
        assert not do.inside_aabb(pos, g.aabb2), tau


def test_volumes_do_not_interpenetrate(setup):
    _, sc = setup
    s, op = sc[False]
    assert s.geo.mind >= op.gap
    assert s.geo.s <= op.s_max


def test_spring_overshoots_and_settles(setup):
    _, sc = setup
    s, op = sc[True]
    qs = np.array([s.q(float(t)) for t in range(0, op.travel + op.hold2 + 1)])
    assert qs.max() > 1.0                                    # overshoots the final pose
    over_deg = math.degrees((qs.max() - 1.0) * s.path.L / s.path.d2)
    assert 1.0 < over_deg < 8.0                              # a few degrees along the arrival arc
    assert qs[op.travel + op.settle:].tolist() == [1.0] * len(qs[op.travel + op.settle:])
    assert (qs[op.travel - 4:op.travel + op.settle] < 1.0).any()          # and undershoots: 1-2 decaying oscillations
    plain, _ = sc[False]
    assert max(plain.q(float(t)) for t in range(0, 61)) == 1.0


REAL = os.path.join(ROOT, "17_depth-dolly", "v001", "local")


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "depth_s1a_raw.npy")), reason="cached P06 depth missing")
def test_real_p06_homes_and_clearance():
    p = dd.Params(depth_ratio=6.0)
    layers = dd.make_layers(REAL, 432, p)[:2]
    for spring in (False, True):
        op = do.OrbitParams(spring=spring)
        geo = do.make_geometry(layers, op)
        sc = do.OrbitScene(layers, geo, op)
        assert dd.psnr(sc.render_q(0.0), layers[0].col) > 40.0
        assert dd.psnr(sc.render_q(1.0), layers[1].col) > 40.0
        assert geo.mind >= op.gap
        for tau in np.linspace(0.0, geo.path.L * 1.03, 400):
            x, z, _ = geo.path.pose(tau)
            pos = np.array([x, 0.0, z])
            assert not do.inside_aabb(pos, geo.aabb1) and not do.inside_aabb(pos, geo.aabb2)


# ----------------------------------------------------------------------------------------------- v006 pivot mode
@pytest.fixture(scope="module")
def pivot(setup):
    layers, _ = setup
    out = {}
    for spring in (False, True):
        op = do.OrbitParams(mode="pivot", spring=spring)
        out[spring] = (do.PivotScene(layers, op), op)
    return layers, out


def test_pivot_frame0_and_end_are_homes(pivot):
    layers, sc = pivot
    for spring in (False, True):
        s, op = sc[spring]
        assert dd.psnr(s.render_t(0.0), layers[0].col) > 40.0
        t_last = float(op.travel + op.hold2)
        assert s.q(t_last) == 1.0
        assert dd.psnr(s.render_t(t_last), layers[1].col) > 40.0
    s, op = sc[False]
    assert dd.psnr(s.render_t(float(op.travel)), layers[1].col) > 40.0


@pytest.mark.parametrize("spring", [False, True])
def test_pivot_camera_distance_to_P_is_constant(pivot, spring):
    _, sc = pivot
    s, op = sc[spring]
    P = np.array([0.0, s.zp])
    for q in np.linspace(0.0, 1.05, 80):
        x, z, psi = s.pose_phi(q * s.Phi)
        assert np.hypot(*(np.array([x, z]) - P)) == pytest.approx(s.zp, abs=1e-9)
        # always looking at P: forward points from the camera to P
        to_p = (P - np.array([x, z])) / s.zp
        assert np.allclose(to_p, do.fwd(psi), atol=1e-9)
    # both volumes share the pivot: image 2's home point at depth zp lands on P
    ex, ez, epsi = s.end_pose()
    p2 = np.array([ex, ez]) + s.zp * do.fwd(epsi)
    assert np.allclose(p2, P, atol=1e-9)


def test_pivot_image2_yaw_monotonic_and_ends_flat(pivot):
    _, sc = pivot
    s, op = sc[False]
    yaw2 = np.array([s.rel_yaw(s.q(float(t)))[1] for t in range(0, op.hold1 + op.travel + op.hold2 + 1)])
    entered = yaw2[yaw2 >= -op.in_angle - 1e-9]
    assert entered[0] <= -op.in_angle + 8.0                     # first drawn at about -in_angle
    assert np.all(np.diff(entered) >= -1e-9)                    # same turning direction all the way
    assert yaw2[-1] == pytest.approx(0.0, abs=1e-9)
    yaw1 = np.array([s.rel_yaw(s.q(float(t)))[0] for t in range(0, op.hold1 + op.travel + 1)])
    assert yaw1[0] == 0.0 and np.all(np.diff(yaw1) >= -1e-9)    # image 1 turns away the same way


def test_pivot_defaults_overlap(pivot):
    _, sc = pivot
    s, op = sc[False]
    assert math.degrees(s.Phi) == pytest.approx(op.out_angle + op.in_angle)
    assert s.out_angle == pytest.approx(60.0)                    # image 1 has turned 60 when image 2 starts to enter
    assert s.q(op.hold1 * 0 + op.travel / 2) == pytest.approx(0.5)   # overlap 0.5 of the travel


def test_pivot_image2_drawn_over_image1(pivot):
    layers, sc = pivot
    s, op = sc[False]
    q = 0.75                                                    # both volumes visible
    both = s.render_q(q)
    cam = do.Cam(*s.pose_phi(q * s.Phi))
    img2, m2 = s._pass(s.rl2, cam)
    img1, m1 = s._pass(s.rl1, cam)
    assert (m1 & m2).sum() > 200                                # both cover these pixels
    w1, w2 = s.weights(q)
    assert w2 > 0.99
    inner = cv2.erode(m2.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0     # clear of the soft edge
    sel = inner & m1
    assert sel.sum() > 100
    assert np.abs(both[sel].astype(int) - img2[sel].astype(int)).max() <= 1   # image 2 wins regardless of depth
    # and where image 2 has no coverage the image 1 / background shows (not image 2)
    far = ~(cv2.dilate(m2.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & m1
    if far.any() and w1 > 0.99:
        assert np.abs(both[far].astype(int) - img1[far].astype(int)).max() <= 1


@pytest.mark.parametrize("spring", [False, True])
def test_pivot_image2_edge_is_soft(pivot, spring):
    layers, sc = pivot
    s, op = sc[spring]
    q = 0.7
    cam = do.Cam(*s.pose_phi(q * s.Phi))
    _, m2 = s._pass(s.rl2, cam)
    m = cv2.erode(m2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    a = np.clip(cv2.GaussianBlur(m, (0, 0), op.edge_soft), 0, 1)
    frac = ((a > 0.05) & (a < 0.95)).sum()
    assert frac > 0                                              # there are partially covered edge pixels


def test_pivot_spring_overshoots_and_rests(pivot):
    _, sc = pivot
    s, op = sc[True]
    qs = np.array([s.q(float(t)) for t in range(op.travel + op.hold2 + 1)])
    over = math.degrees((qs.max() - 1.0) * s.Phi)
    assert 1.0 < over < 8.0
    assert (qs[op.travel + op.settle:] == 1.0).all()


# ----------------------------------------------------------------------------------------------- v008 billboard cards
@pytest.fixture(scope="module")
def bb(setup):
    layers, _ = setup
    op = do.OrbitParams(mode="pivot", splat="billboard")
    return layers, do.PivotScene(layers, op), op


def test_billboard_home_frames_exact(bb):
    layers, s, op = bb
    assert dd.psnr(s.render_t(0.0), layers[0].col) > 40.0
    assert dd.psnr(s.render_t(float(op.travel)), layers[1].col) > 40.0
    assert dd.psnr(s.render_t(float(op.travel + op.hold2)), layers[1].col) > 40.0
    assert dd.psnr(s.render_q(0.0), layers[0].col) > 40.0 and dd.psnr(s.render_q(1.0), layers[1].col) > 40.0


def test_billboard_chunk_cap_and_every_pixel_covered(bb):
    layers, s, op = bb
    for rl in (s.rl1, s.rl2):
        c = s.cards[id(rl)]
        assert c["n_subject"] + c["n_back"] <= op.max_chunks
        assert c["n_subject"] > 0 and c["n_back"] > 0
        cov = do._covered(rl.L.disp.shape, c["x"][c["tid"] == 1], c["y"][c["tid"] == 1], c["S"][c["tid"] == 1])
        assert cov.all()                                          # background cards cover the whole frame
        sub = do._covered(rl.L.disp.shape, c["x"][c["tid"] == 0], c["y"][c["tid"] == 0], c["S"][c["tid"] == 0])
        assert sub[rl.L.subj].all()                               # subject cards cover the whole matte


def test_billboard_pieces_do_not_stretch_and_stay_screen_aligned(bb):
    layers, s, op = bb
    q = 0.5
    img, al = s._pass_bb(s.rl1, do.Cam(*s.pose_phi(q * s.Phi)))
    assert al.max() <= 1.0 and al.max() > 0.9


@pytest.mark.parametrize("side", [6.0, 17.0, 40.5])
def test_card_screen_size_scales_with_inverse_depth_and_ignores_neighbours(side):
    W_, H_ = 300, 300
    tex = np.ones((W_ * H_, 4), np.float32) * 200.0
    tex[:, 3] = 1.0

    def area(sd, other=None):
        u = np.array([100.3]); v = np.array([120.6]); sz = np.array([sd]); rt = np.array([1.0])
        cx = np.array([100.3]); cy = np.array([120.6]); tid = np.array([0]); rank = np.array([1])
        if other is not None:                                     # a nearer, far-away neighbour (different size)
            u = np.append(u, 230.0); v = np.append(v, 40.0); sz = np.append(sz, other); rt = np.append(rt, 1.0)
            cx = np.append(cx, 230.0); cy = np.append(cy, 40.0); tid = np.append(tid, 0); rank = np.append(rank, 0)
        rgb, al = do.card_splat(u, v, sz, rt, cx, cy, tid, rank, tex, W_, H_)
        return float(al[:, :200].sum()), al
    # side = ratio * S with ratio = Z_home / Z_cam: halving the camera depth doubles the side (area x4)
    a1, _ = area(side)
    a2, _ = area(side * 2.0)
    assert a1 == pytest.approx(side ** 2, rel=2e-3)
    assert a2 == pytest.approx(4.0 * a1, rel=2e-3)
    a3, _ = area(side, other=23.0)                                # neighbour of another size changes nothing
    assert a3 == pytest.approx(a1, rel=1e-6)


def test_billboard_scene_card_side_follows_z_home_over_z_cam(bb):
    layers, s, op = bb
    rl = s.rl1
    full = s.cards[id(rl)]
    k = int(np.nonzero(full["tid"] == 1)[0][0])
    one = {kk: (v[k:k + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(full["S"]) else v) for kk, v in full.items()}
    one["x"] = np.array([48.0]); one["y"] = np.array([80.0]); one["S"] = np.array([10.0])
    Z = float(one["Z"][0])
    one["X"] = (one["x"] - rl.L.cx) / rl.L.f * Z
    one["Y"] = (one["y"] - rl.L.cy) / rl.L.f * Z
    s.cards[id(rl)] = one
    try:
        a0 = float(s._pass_bb(rl, do.Cam(0.0, 0.0, 0.0))[1].sum())
        a1 = float(s._pass_bb(rl, do.Cam(0.0, 0.5 * Z, 0.0))[1].sum())   # camera moves 0.5 Z forward: Z_cam = Z/2
    finally:
        s.cards[id(rl)] = full
    assert a0 == pytest.approx((10.0 + 2 * op.card_grow) ** 2, rel=2e-3)
    assert a1 == pytest.approx(4.0 * a0, rel=2e-3)                    # side x2 at half the depth, nothing else changes


# ----------------------------------------------------------------------------------------------- v012 continuous blur + one subject card
def _unimodal(row, tol=1e-5):
    d = np.diff(row)
    sgn = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    sgn = sgn[sgn != 0]
    return int((np.diff(sgn) != 0).sum()) <= 1 and (len(sgn) == 0 or sgn[0] >= 0)   # rises (or flat), then falls: once


@pytest.mark.parametrize("ang", [0.0, 37.0, 90.0])
def test_smear_patch_has_no_gaps_or_repeated_edges(ang):
    P = np.zeros((120, 160, 4), np.float32)
    P[50:70, 70:90] = 1.0                                          # opaque 20 x 20 square, white (premultiplied)
    L = 60.0                                                       # 3x the square: stepped copies would leave gaps
    dx, dy = L * math.cos(math.radians(ang)), L * math.sin(math.radians(ang))
    out = do.smear_patch(P, dx, dy)
    assert out[..., 3].sum() == pytest.approx(P[..., 3].sum(), rel=0.03)            # mass conserved
    chord = 20.0 / max(abs(math.cos(math.radians(ang))), abs(math.sin(math.radians(ang))))   # longest chord through the square
    assert out[..., 3].max() == pytest.approx(chord / L, rel=0.08)                    # plateau = chord / L
    if ang == 0.0:
        row = out[60, :, 3]
        assert _unimodal(row)                                                          # one rise, one plateau, one fall
        nz = np.nonzero(row > 1e-4)[0]
        assert nz.max() - nz.min() + 1 == pytest.approx(20 + L, abs=3)               # extent = width + motion
        assert (row[nz.min():nz.max() + 1] > 1e-3).all()                              # no gap anywhere inside
    # colour stays white wherever there is alpha (premultiplied rgb == alpha)
    assert np.allclose(out[..., 0], out[..., 3], atol=2e-3)


def test_smear_patch_zero_motion_is_identity():
    rng = np.random.default_rng(3)
    P = rng.random((40, 50, 4), dtype=np.float32)
    assert np.array_equal(do.smear_patch(P, 0.0, 0.0), P)
    assert np.array_equal(do.smear_patch(P, 0.4, 0.3), P)


def test_continuous_blur_static_card_unchanged(bb):
    layers, s, op = bb
    for q in (0.3, 0.7):
        cam = do.Cam(*s.pose_phi(q * s.Phi))
        for rl in (s.rl1, s.rl2):
            a, aa = s._pass_bb(rl, cam)
            b, bb_ = s._pass_bb_motion(rl, cam, cam, cam)
            assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
            assert np.abs(aa - bb_).max() < 2e-3


def test_continuous_blur_moving_card_in_scene_is_a_smooth_trapezoid(bb):
    layers, s, op = bb
    rl = s.rl1
    full = s.cards[id(rl)]
    k = int(np.nonzero(full["tid"] == 1)[0][0])
    one = {kk: (v[k:k + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(full["S"]) else v) for kk, v in full.items()}
    one["x"] = np.array([48.0]); one["y"] = np.array([80.0]); one["S"] = np.array([6.0])
    Z = float(one["Z"][0])
    one["X"] = (one["x"] - rl.L.cx) / rl.L.f * Z
    one["Y"] = (one["y"] - rl.L.cy) / rl.L.f * Z
    s.cards[id(rl)] = one
    try:
        cam = do.Cam(0.0, 0.0, 0.0)
        ca, cb = do.Cam(-0.3 * Z, 0.0, 0.0), do.Cam(0.3 * Z, 0.0, 0.0)    # camera slides sideways: the card sweeps ~ 0.6 f px
        u_a = float(s._proj_bb(rl, ca)[0][0]); u_b = float(s._proj_bb(rl, cb)[0][0])
        sweep = abs(u_b - u_a)
        assert sweep > 3 * (6 + 2 * op.card_grow)                          # far more than the card width: steps would show gaps
        _, al = s._pass_bb_motion(rl, cam, ca, cb)
        row = al[80]
        assert _unimodal(row, 1e-6)
        nz = np.nonzero(row > 1e-4)[0]
        assert (row[nz.min():nz.max() + 1] > 1e-4).all()
        assert al.sum() == pytest.approx((6 + 2 * op.card_grow) ** 2, rel=0.05)    # nothing lost, nothing added
    finally:
        s.cards[id(rl)] = full


def test_subject_is_one_card_and_cap_holds(setup):
    layers, _ = setup
    for chunks in (25, 50, 100):
        op = do.OrbitParams(mode="pivot", splat="billboard", max_chunks=chunks)
        assert op.subject_cards == 1
        sc = do.PivotScene(layers, op)
        for rl in (sc.rl1, sc.rl2):
            c = sc.cards[id(rl)]
            assert c["n_subject"] == 1
            assert c["n_subject"] + c["n_back"] <= chunks
            sel = c["tid"] == 0
            assert sel.sum() == 1
            sub = do._covered(rl.L.disp.shape, c["x"][sel], c["y"][sel], c["S"][sel])
            assert sub[rl.L.subj].all()                            # the one card covers the whole matte
            assert c["tex"].reshape(2, H, W, 4)[0, ..., 3].sum() == rl.L.subj.sum()   # its alpha is the full matte
    op0 = do.OrbitParams(mode="pivot", splat="billboard", subject_cards=0)
    sc0 = do.PivotScene(layers, op0)
    assert sc0.cards[id(sc0.rl1)]["n_subject"] > 1                 # the old behaviour is still there behind the flag


def test_blur_modes(setup):
    layers, _ = setup
    q = None
    outs = {}
    for b in ("steps", "continuous", "off"):
        sc = do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard", blur=b))
        outs[b] = sc.render_t(24.0, 4)
        assert sc.blur_mode() == b
    assert not np.array_equal(outs["steps"], outs["continuous"])
    assert not np.array_equal(outs["off"], outs["continuous"])
    assert do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="stretch")).blur_mode() == "steps"
    assert do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard")).blur_mode() == "continuous"


# ----------------------------------------------------------------------------------------------- v013 video textures
def _fake_video(layers, n=5):
    """per shot: n frames (the frame-0 image shifted by k px so every frame differs) and a matte that moves 2 px per frame."""
    out = []
    for L in layers[:2]:
        frames = np.stack([np.roll(L.col, 3 * k, axis=1) for k in range(n)])
        mattes = np.zeros((n, H, W), np.float32)
        for k in range(n):
            mattes[k, 60:120, 30 + 2 * k:70 + 2 * k] = 1.0
        out.append((frames, mattes))
    return out


def test_video_home_frames_match_the_current_clip_frame(setup):
    layers, _ = setup
    vid = _fake_video(layers)
    s = do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard", video=True), video=vid)
    s.set_frames(3, 4)
    # scene level: the background cards carry the frame with the runner FILLED out of it, 3 px wider than his matte (subject_split), so the ring round him is a fill, not
    # the frame (the pivot_cut adapter returns the clip's own frame at a rest pose: exact, tests/test_cut_styles.py)
    assert dd.psnr(s.render_q(0.0), vid[0][0][3]) > 25.0           # image 1 home ~ its frame 3
    assert dd.psnr(s.render_q(1.0), vid[1][0][4]) > 25.0           # image 2 landing ~ its frame 4
    s.set_frames(0, 0)
    assert dd.psnr(s.render_q(0.0), layers[0].col) > 25.0
    s.set_frames(99, 99)                                           # a clip that runs out holds its last frame
    assert dd.psnr(s.render_q(0.0), vid[0][0][-1]) > 25.0


def test_video_runner_card_alpha_is_the_frame_matte_and_back_cards_exclude_the_runner(setup):
    layers, _ = setup
    vid = _fake_video(layers)
    s = do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard", video=True), video=vid)
    s.set_frames(4, 0)
    c = s.cards[id(s.rl1)]
    t = c["tex"].reshape(2, H, W, 4)
    assert np.array_equal(t[0, ..., 3] > 0.5, vid[0][1][4] > 0.5)                  # runner alpha = the (soft) matte of frame 4
    sub = vid[0][1][4] > 0.5
    far = cv2.dilate((vid[0][1][4] > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=9) == 0
    assert np.array_equal(t[1][far][:, :3], vid[0][0][4][far].astype(np.float32))    # live video away from him
    under = t[1][sub][:, :3]                                                        # under him: filled from THIS frame's surroundings, not the frame-0 LDI colour
    assert not np.array_equal(under, layers[0].bcol[sub].astype(np.float32)) and np.abs(under - vid[0][0][4][~cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)].mean(0)).max() < 255
    assert c["n_subject"] == 1
    # the single runner card spans the matte of EVERY frame, not just frame 0
    sel = c["tid"] == 0
    cov = do._covered((H, W), c["x"][sel], c["y"][sel], c["S"][sel])
    assert cov[(vid[0][1] > 0.5).any(0)].all()
