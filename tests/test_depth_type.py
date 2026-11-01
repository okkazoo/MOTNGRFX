"""P06 type planes in the depth-orbit pivot scene (tools/fusion/depth_type.py). CPU only, synthetic layers, small type matte.
Run: python -m pytest tests/test_depth_type.py -q"""
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
from tools.fusion import depth_type as dt  # noqa: E402
from test_depth_orbit import _layer, W, H  # noqa: E402

FLAG = "P06:SEARCHING,A LIGHT"


@pytest.fixture(scope="module")
def scene():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    layers = [_layer(s, p) for s in (1, 2)]
    op = do.OrbitParams(mode="pivot", splat="billboard", max_chunks=30, in_angle=45, out_angle=40)
    s = do.PivotScene(layers, op)
    s.typ = dt.TypeLayer(s, FLAG, matte_size=(192, 320), mode="front", style="v014")      # v014 behaviour (front, q-retimed, centred)
    return s


@pytest.fixture(scope="module")
def scene2():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    layers = [_layer(s, p) for s in (1, 2)]
    op = do.OrbitParams(mode="pivot", splat="billboard", max_chunks=30, in_angle=45, out_angle=40, hold1=24, hold2=48)
    s = do.PivotScene(layers, op)
    s.typ = dt.TypeLayer(s, FLAG, matte_size=(192, 320))                                   # v015: behind_subject + organic
    return s


def test_parse_flag():
    assert dt.parse_flag("P06:SEARCHING,A LIGHT") == ("P06", ["SEARCHING", "A LIGHT"])
    for bad in ("SEARCHING,A LIGHT", "P06:ONLY", "P06:A,B,C", ""):
        with pytest.raises(ValueError):
            dt.parse_flag(bad)


def test_plane_home_pose_is_the_frame_rectangle(scene):
    pl = dt.Plane(scene.rl1.L, 1.7, 0.0)
    d = pl.project(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    assert d == pytest.approx(np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]]), abs=1e-6)


def test_plane_corners_move_exactly_like_cards(scene):
    """the corner of a type plane projects to the same pixel as a card placed at that 3D point of the image's volume."""
    s = scene
    for which, rl in enumerate((s.rl1, s.rl2)):
        pl = s.typ.planes[which]
        for q in (0.0, 0.3, 0.6, 0.9, 1.0):
            cam = do.Cam(*s.pose_phi(q * s.Phi))
            d = s.typ.corners(which, q)
            full = s.cards[id(rl)]
            keep = {"X": pl.P[:, 0], "Y": pl.P[:, 1], "Z": pl.P[:, 2], "S": np.ones(4)}
            s.cards[id(rl)] = keep
            try:
                u, v, *_ = s._proj_bb(rl, cam)
            finally:
                s.cards[id(rl)] = full
            assert d is not None
            assert d[:, 0] == pytest.approx(u, abs=1e-4) and d[:, 1] == pytest.approx(v, abs=1e-4)


def test_home_poses_equal_flat_overlay(scene):
    """yaw 0: at the home pose the word covers the frame 1:1 (the sharp frame = the 2x texture area-downscaled)."""
    s = scene
    for which, q in ((0, 0.0), (1, 1.0)):
        lay = s.rls[which].L if hasattr(s, "rls") else (s.rl1, s.rl2)[which].L
        s.typ.planes[which] = dt.Plane(lay, s.typ.planes[which].zt, 0.0)
        tau = s.typ.tau(which, q)
        tex = s.typ.texture(which, tau)
        ref = cv2.resize(tex, (W, H), interpolation=cv2.INTER_AREA)
        pm, a = s.typ.layer(which, q)
        assert a == pytest.approx(ref[..., 3], abs=1e-4)
        assert pm == pytest.approx(ref[..., :3], abs=1e-2)
        assert a.max() > 0.5
    # restore the piece's yaw
    for which, (rl, wt) in enumerate(zip((s.rl1, s.rl2), s.typ.words)):
        s.typ.planes[which] = dt.Plane(rl.L, s.typ.planes[which].zt, wt.yaw)


def test_plane_turns_in_perspective_with_its_image(scene):
    s = scene
    d = s.typ.corners(0, 0.3)                                     # image 1 has turned: the plane is no longer a rectangle
    left, right = np.linalg.norm(d[3] - d[0]), np.linalg.norm(d[2] - d[1])
    assert abs(left - right) / max(left, right) > 0.05
    d1 = s.typ.corners(1, 0.6)                                    # image 2 swinging in: also foreshortened
    assert abs(np.linalg.norm(d1[3] - d1[0]) - np.linalg.norm(d1[2] - d1[1])) > 1.0
    # the word's yaw about its own axis: at yaw 0 the home view is a rectangle, at +16 the right edge is nearer (longer)
    pl16 = dt.Plane(s.rl1.L, 1.7, 16.0)
    h = pl16.project(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    assert np.linalg.norm(h[2] - h[1]) > np.linalg.norm(h[3] - h[0])


def test_retiming_form_out_on_image1_form_in_on_image2(scene):
    t = scene.typ
    i0, i1 = t.words[0].info, t.words[1].info
    assert t.tau(0, 0.0) == i0["hit"] and t.tau(0, t.q_out0) == i0["hit"]               # formed on image 1's hold
    qs = np.linspace(t.q_out0, t.q_split, 20)
    taus = [t.tau(0, q) for q in qs]
    assert taus[0] == i0["hit"] and taus[-1] == pytest.approx(i0["end"])                 # the hold pose is the release pose (static)
    assert all(b >= a for a, b in zip(taus[1:], taus[2:])) and taus[1] > i0["release"]
    assert t.layer(0, 0.0) is not None and t.layer(0, t.q_split) is None                  # gone when image 2 starts to enter
    assert t.tau(1, 0.0) is None and t.tau(1, t.q_split - 0.01) is None
    qs = np.linspace(t.q_split, t.q_in1, 20)
    taus = [t.tau(1, q) for q in qs]
    assert all(b >= a for a, b in zip(taus, taus[1:])) and taus[0] == pytest.approx(i1["start"]) and taus[-1] == pytest.approx(i1["hit"])
    assert t.tau(1, 1.0) == i1["hit"] and t.tau(1, 0.97) == i1["hit"]                   # fully formed before the landing
    assert t.layer(1, 1.0)[1].max() > 0.5                                               # readable in the final hold
    assert t.layer(1, 0.0) is None                                                      # nothing of image 2 on image 1's hold


def test_motion_blur_uses_subposes_and_keeps_the_energy(scene):
    t = scene.typ
    sh = scene.op.shutter
    tm = 18.0
    q = scene.q(tm)
    assert t.sub_qs(1, q, tm) and len(t.sub_qs(1, q, tm)) > 1
    assert t.sub_qs(1, 0.0, 0.0) == [0.0] and t.sub_qs(1, 1.0, 40.0) == [1.0]
    sharp = t.layer(1, q)
    blur = t.layer(1, q, tm)
    assert sharp is not None and blur is not None
    assert blur[1].sum() == pytest.approx(sharp[1].sum(), rel=0.3)
    assert blur[1].max() <= sharp[1].max() + 1e-6


def test_ink_is_legible_on_dark_and_on_light_backgrounds():
    pal = {"fg": [235, 240, 255], "accents": [[90, 170, 255], [255, 170, 60]]}
    a = np.zeros((H, W), np.float32)
    a[60:100, 20:76] = 1.0
    dark = np.full((H, W, 3), 25, np.uint8)
    light = np.full((H, W, 3), 235, np.uint8)
    ink_d = dt.choose_ink(dark, a, pal, cap_norm=400)
    ink_l = dt.choose_ink(light, a, pal, cap_norm=400)
    assert ink_d["name"] == "fg" and ink_d["support"] is None and ink_d["share_ok"] > 0.99
    assert ink_l["y_ink"] < 0.3 and ink_l["share_ok"] > 0.99                                 # went dark, not the palette white
    mid = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)               # busy plate: support instead
    ink_m = dt.choose_ink(mid, a, pal, cap_norm=400)
    assert ink_m["support"] in ("outline", "shadow", "scrim") or ink_m["share_ok"] > 0.8
    tex = dt.premult_tex(a, dict(ink_m, support="scrim"))
    assert tex[..., 3].max() <= 1.0 + 1e-6 and tex[10, 5, 3] > 0.0                           # the scrim reaches beside the letters


def test_scene_hook_draws_the_words_and_leaves_the_image_without_the_flag(scene):
    s = scene
    typ = s.typ
    try:
        s.typ = None
        base = {q: s.render_q(q) for q in (0.0, 0.85, 1.0)}
        s.typ = typ
        for q in (0.0, 0.85, 1.0):
            fr = s.render_q(q)
            assert fr.dtype == np.uint8 and fr.shape == base[q].shape
            assert np.abs(fr.astype(int) - base[q].astype(int)).max() > 40            # the word is on the frame
        s.typ = None
        assert np.array_equal(s.render_q(0.0), base[0.0])
        blur = s.render_q(s.q(18.0), blur_t=18.0)                                     # continuous path with type: runs, same shape
        s.typ = typ
        assert s.render_q(s.q(18.0), blur_t=18.0).shape == blur.shape
    finally:
        s.typ = typ


def test_bg_chunks_caps_the_background_cards_only():
    p = dd.Params(depth_ratio=6.0)
    p.guided_r = 0
    layers = [_layer(s, p) for s in (1, 2)]
    base = do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard", max_chunks=50))
    cap = do.PivotScene(layers, do.OrbitParams(mode="pivot", splat="billboard", max_chunks=50, bg_chunks=8))
    for rl in (base.rl1, base.rl2):
        b, c = base.cards[id(rl)], cap.cards[id((cap.rl1, cap.rl2)[(base.rl1, base.rl2).index(rl)])]
        assert c["n_back"] <= 8 < b["n_back"] and c["n_subject"] == b["n_subject"]
        assert c["ref"] == b["ref"]


# ----------------------------------------------------------------------------------------------- v015: behind the runner, organic timing
def test_defaults_are_behind_subject_organic(scene2):
    t = scene2.typ
    assert (t.mode, t.style) == ("behind_subject", "organic") and scene2._sandwich_on()
    assert t.report[0]["mode"] == "behind_subject"
    for which, rl in enumerate((scene2.rl1, scene2.rl2)):                                  # between the runner card and the background
        zs = float(np.median(rl.L.grids[0][2][rl.L.subj]))
        assert t.planes[which].zt > zs


def test_runner_matte_occludes_the_type(scene2):
    s = scene2
    typ = s.typ
    for which, q, tt in ((0, 0.0, 0.0), (1, 1.0, s.op.travel + 40.0)):
        s.t_cur = tt
        with_t = s.render_q(q, blur_t=tt).astype(int)
        s.typ = None
        try:
            base = s.render_q(q).astype(int)
        finally:
            s.typ = typ
        rl = (s.rl1, s.rl2)[which]
        runner = cv2.erode(rl.L.subj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        assert np.abs(with_t - base)[runner].max() <= 1                                      # where he is, the frame is the runner, not the letters
        assert np.abs(with_t - base)[~rl.L.subj].max() > 40                                  # elsewhere the word is there
    # front mode puts the type over everything (v014)
    s.typ = dt.TypeLayer(s, FLAG, matte_size=(192, 320), mode="front", style="v014")
    try:
        s.t_cur = 0.0
        f_front = s.render_q(0.0).astype(int)
    finally:
        s.typ = None
    try:
        base = s.render_q(0.0).astype(int)
    finally:
        s.typ = typ
    assert np.abs(f_front - base)[s.rl1.L.subj].max() > 40                                   # front: the letters cover him


def test_word1_is_carried_in_the_direction_of_the_turn(scene2):
    s = scene2
    t = s.typ
    turn = t.turn
    base0 = t.corners(0, 0.0, 0.0).mean(0)
    for tt in (10.0, 14.0, 18.0, 22.0):
        q = s.q(tt)
        with_e = t.corners(0, q, tt, extras=True).mean(0)
        no_e = t.corners(0, q, tt, extras=False).mean(0)
        assert (with_e[0] - base0[0]) * turn > 0                                              # on-screen motion along the turn
        assert (with_e[0] - no_e[0]) * turn > 0                                               # and the follow-through adds to the image's own
    assert t.report[0]["side"] == ("right" if turn > 0 else "left")                           # strands stream the same way
    assert t.report[1]["side"] == ("left" if turn > 0 else "right")                           # and arrive from the opposite side of the screen
    assert t.extras(0, 0.0) == (0.0, 0.0)                                                     # image 1's hold: no extra motion
    dx = np.array([t.extras(0, tt)[0] for tt in np.arange(0, 36, 0.5)]) * turn
    assert dx[-1] > dx[4] >= 0 and dx.max() < 1.12 * 0.9 * t.planes[0].width                  # spring: little overshoot


def test_word1_is_gone_before_image2_is_half_in(scene2):
    s = scene2
    t = s.typ
    assert t.tau(0, 0.0, 0.0) == t.words[0].info["hit"]                                       # formed on the hold
    assert t.tau(0, t.q_out1, 20.0) == pytest.approx(t.words[0].info["end"])
    half_in = (s.out_angle + 0.5 * s.op.in_angle) / math.degrees(s.Phi)                       # image 2 half way (22.5 of 45 deg)
    assert t.q_out1 <= half_in and t.layer(0, t.q_out1, None, 20.0, weight=False) is None


def test_word2_forms_only_after_the_landing(scene2):
    s = scene2
    t = s.typ
    land = s.op.travel
    for tt in (0.0, 10.0, 25.0, land - 1.0):                                                  # invisible during the swing
        assert t.tau(1, s.q(tt), tt) is None
        assert t.layer(1, s.q(tt), None, tt, weight=False) is None
    a_land = t.layer(1, 1.0, None, land, weight=False)
    assert a_land is None or a_land[1].max() < 0.05                                           # nothing at the landing frame itself
    a_mid = t.layer(1, 1.0, None, land + 8.0, weight=False)
    assert a_mid is not None and a_mid[1].max() > 0.0                                         # forming after the landing
    i1 = t.words[1].info
    assert t.tau(1, 1.0, land + t.form_frames) == pytest.approx(i1["hit"])                    # formed 18 frames (0.75 s) later
    for tt in (land + 18.0, land + 30.0, land + 48.0):                                        # readable hold >= 1.25 s (30 frames)
        assert t.tau(1, 1.0, tt) == pytest.approx(i1["hit"])
        assert t.layer(1, 1.0, None, tt, weight=False)[1].max() > 0.5
    taus = [t.tau(1, 1.0, land + k) for k in range(0, 19)]                                    # eased arrival: no hard stop
    steps = np.diff(taus)
    assert steps[0] > steps[-1] >= 0
    assert abs(t.extras(1, land)[0]) > 0 and abs(t.extras(1, land + 60.0)[0]) < 0.02 * t.planes[1].width   # settles through a spring


def test_holds_and_timeline(scene2):
    op = scene2.op
    fl = do.frame_list(op)
    assert op.hold1 == 24 and op.travel == 36 and op.hold2 == 48 and len(fl) == 24 + 36 + 48
    assert sum(1 for t, _ in fl if t == 0.0) == 24                                            # image 1 holds ~1 s (24 frames)
    assert sum(1 for t, _ in fl if t >= op.travel + 18) >= 30                                 # after word 2 has formed: >= 1.25 s read
    i2 = next(i for i, (t, _) in enumerate(fl) if scene2.weights(scene2.q(t))[1] > 0.0)
    assert len(fl) - 1 - i2 < 73                                                              # image 2's 73-frame clip does not run out


def test_spring_response_overshoots_a_little_and_settles():
    ts = np.arange(0, 120, 0.125)
    x = dt.spring_response(np.ones_like(ts), ts, omega=0.3, zeta=0.7)
    assert 1.0 < x.max() < 1.08 and abs(x[-1] - 1.0) < 1e-3
    y = dt.spring_response(np.ones_like(ts), ts, omega=0.3, zeta=1.0)
    assert y.max() <= 1.0 + 1e-6


def test_placement_beside_the_runner_and_ink_measured_without_him(scene2):
    t = scene2.typ
    for which, rl in enumerate((scene2.rl1, scene2.rl2)):
        ys, xs = np.nonzero(rl.L.subj)
        runner_left = 0.5 * (xs.min() + xs.max()) < W / 2
        assert (t.centers[which][0] > W / 2) == runner_left                                   # the word is on the side the runner leaves free
        assert 0.0 <= t.report[which]["hidden_by_runner"] <= 0.5          # (the 96 px test frame leaves little room; real frames: < 0.07)
    a = np.zeros((H, W), np.float32)
    a[60:100, 20:76] = 1.0
    img = np.full((H, W, 3), 235, np.uint8)
    img[:, :48] = 20                                                                          # the dark half is background, the bright half is the runner
    hide = np.zeros((H, W), np.float32)
    hide[:, 48:] = 1.0
    pal = {"fg": [235, 240, 255], "accents": []}
    assert dt.choose_ink(img, a, pal, 400, hide=hide)["y_ink"] > 0.5                          # measured without him: dark plate -> light ink
    assert dt.choose_ink(img, a, pal, 400)["share_ok"] <= 1.0
