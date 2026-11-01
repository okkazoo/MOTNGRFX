"""Depth-parallax dolly transitions (tools/fusion/depth_dolly.py). CPU only, synthetic layers, no depth model.
Run: python -m pytest tests/test_depth_dolly.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402

W, H = 96, 160


def _texture(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8, 1), np.uint8))
    return np.ascontiguousarray(img)


def _layer(seed, params):
    """Synthetic image: far wall (disp 0.1) with a near block (disp 0.9) in the middle and a subject card."""
    img = _texture(seed)
    disp = np.full((H, W), 0.1, np.float32)
    disp[:, :20] = 0.5
    disp[60:120, 30:70] = 0.9
    matte = np.zeros((H, W), np.float32)
    matte[60:120, 30:70] = 1.0
    return dd.build_layer(img, disp, matte, params)


@pytest.fixture(scope="module")
def params():
    p = dd.Params()
    p.guided_r = 0                      # synthetic blocks: keep the depth exactly as written
    return p


@pytest.fixture(scope="module")
def layers(params):
    return [_layer(s, params) for s in (1, 2, 3)]


def test_home_camera_reproduces_source(layers, params):
    for d in dd.DIRS:
        sc = dd.Scene(layers, d, params)
        out = sc.render_c(0.0)
        assert dd.psnr(out, layers[0].col) > 30.0, d


def test_near_moves_more_than_far(layers, params):
    sc = dd.Scene(layers, "right", params)
    L = layers[0]
    C0, C1 = sc.camera(0.0), sc.camera(0.2)
    u0, _, _ = L.project(C0, sc.offset(0))
    u1, _, _ = L.project(C1, sc.offset(0))
    near = abs(float(np.mean(u1[90, 40:60] - u0[90, 40:60])))     # near block
    far = abs(float(np.mean(u1[20, 40:90] - u0[20, 40:90])))      # far wall
    assert near > 1.5 * far > 0.0
    # direction: camera moves right -> content moves left
    assert np.all(u1[90, 40:60] < u0[90, 40:60])


def test_incoming_final_frame_equals_source(layers, params):
    for d in ("left", "down"):
        sc = dd.Scene(layers, d, params)
        assert dd.psnr(sc.render_c(1.0), layers[1].col) > 30.0, d
        assert dd.psnr(sc.render_c(2.0), layers[2].col) > 30.0, d


def test_outgoing_gone_and_incoming_placed_mid_travel(layers, params):
    """Near the end of travel image 1 is (almost) entirely replaced by image 2; the far content has left."""
    sc = dd.Scene(layers, "right", params)
    late = sc.render_c(0.999)
    assert dd.psnr(late, layers[1].col) > 20.0
    early = sc.render_c(0.001)
    assert dd.psnr(early, layers[0].col) > 20.0


def test_timeline_and_easing(params):
    tl = dd.timeline(params, 3)
    assert tl[0][0] == 0.0 and tl[-1][0] == 2.0
    cs = [c for c, _ in tl]
    assert cs == sorted(cs)                                   # monotone
    assert dd.smootherstep(0.0) == 0.0 and dd.smootherstep(1.0) == 1.0
    assert dd.smootherstep(0.5) == pytest.approx(0.5)
    n_hold = sum(1 for _, k in tl if k == "hold")
    assert n_hold == 3 * params.ease_hold


def test_unique_path(tmp_path):
    p = tmp_path / "a.png"
    assert dd.unique_path(str(p)) == str(p)
    p.write_bytes(b"x")
    assert dd.unique_path(str(p)).endswith("a_g02.png")
    (tmp_path / "a_g02.png").write_bytes(b"x")
    assert dd.unique_path(str(p)).endswith("a_g03.png")


# ------------------------------------------------------------------------------------ v003: per-depth motion
def _p3(**kw):
    p = dd.Params()
    p.guided_r = 0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_lag0_matches_rigid_camera():
    """lag=0 and no spring: fixed camera + per-point offsets == the old one-camera move."""
    p = _p3(lag=0.0, spring=False, zbias=0.0)
    ls = [_layer(s, p) for s in (1, 2, 3)]
    for d in ("right", "down"):
        sc = dd.Scene(ls, d, p)
        for t in (6, 10, 20):
            c = float(dd.smootherstep(t / p.ease_travel))
            assert dd.psnr(sc.render_t(0, float(t)), sc.render_c(c)) > 40.0, (d, t)


def test_exit_far_starts_later_entry_reversed():
    p = _p3()
    ls = [_layer(s, p) for s in (1, 2, 3)]
    sc = dd.Scene(ls, "right", p)
    L = ls[0]
    assert L.wfar[0][90, 40] < 0.2 and L.wfar[0][20, 60] > 0.8          # near block / far wall
    for t in (4.0, 8.0, 12.0):
        o = sc.point_offset("out", t)(L, 0)[0]
        near, far = abs(float(o[90, 40])), abs(float(o[20, 60]))
        assert far < near, t                                           # far starts later -> has moved less
        i = sc.point_offset("in", t)(L, 0)[0]
        near_i, far_i = abs(float(i[90, 40])), abs(float(i[20, 60]))
        assert far_i < near_i, t                                       # far already closer to home (offset smaller)
    # far content of the leaving image has not started at t = 4 (delay = lag*T*w ~ 13 frames)
    assert abs(float(sc.point_offset("out", 4.0)(L, 0)[0][20, 60])) < 1e-6


def test_spring_overshoots_then_rests():
    p = _p3()
    ls = [_layer(s, p) for s in (1, 2, 3)]
    sc = dd.Scene(ls, "right", p)
    fs = sc.f * sc.span                                                # px of near content per span
    u = np.arange(0, p.ease_travel + p.settle + 6, 0.25)
    x = dd.spring_curve(u, sc.cfg)
    assert x[0] == 0.0 and x.max() > 1.0
    assert (x.max() - 1.0) * fs == pytest.approx(p.overshoot * sc.W, rel=0.02)     # ~35 px at 432 wide
    assert x.min() == 0.0 and (x[u > p.ease_travel + 3] < 1.0).any()                # undershoot after the peak
    end = p.ease_travel + p.settle - 1
    assert np.all(np.abs(x[u >= end] - 1.0) == 0.0)
    assert np.abs(x[u >= end - 1.0] - 1.0).max() * fs < 0.25
    assert dd.rest_time(p) == dd.window(p) + p.settle - 1


def test_settled_hold_is_home():
    p = _p3()
    ls = [_layer(s, p) for s in (1, 2, 3)]
    sc = dd.Scene(ls, "down", p)
    fl = dd.frames(p, 3, loop=False)
    last = fl[-1]
    assert last.kind == "hold" and sc.home_of(last) == 2
    assert dd.psnr(sc.render_t(last.seg, last.t), ls[2].col) > 30.0
    # the frame right before the rest frame is still in the spring, within 0.25 px of home
    assert fl[-(p.ease_hold - p.settle + 2)].kind == "settle"


def test_loop_first_frame_equals_frame_after_last():
    p = _p3()
    ls = [_layer(s, p) for s in (1, 2, 3)]
    ls = ls + [ls[0]]
    sc = dd.Scene(ls, "right", p)
    fl = dd.frames(p, len(ls), loop=True)
    first, last = fl[0], fl[-1]
    assert sc.home_of(first) == 0 and sc.home_of(last) == 3 and ls[3] is ls[0]
    assert last.kind == "hold" and first.kind == "hold"
    assert np.array_equal(sc.render_t(first.seg, first.t), sc.render_t(last.seg, last.t))
    # the hold is split across the loop point: end hold + start hold == one ordinary hold; spring at rest at the end
    n_start = next(i for i, f in enumerate(fl) if f.t > 0)
    n_end = sum(1 for f in fl if f.seg == 2 and f.t >= dd.window(p))
    assert n_start + n_end == p.ease_hold and n_end >= p.settle


# ------------------------------------------------------------------------------------ v004: random patch delays
def test_random_delays_per_patch():
    p = _p3(delay_mode="random", cell=60.0, seed=3)           # 60 px at 432 wide = ~13 px on the 96 px test layers
    ls = [_layer(s, p) for s in (1, 2)]
    sc = dd.Scene(ls, "right", p)
    L = ls[0]
    cell = p.cell * L.W / 432.0
    labs = {}
    for mode, off in (("out", 17), ("in", 91)):
        lab = dd.patch_labels(L.H, L.W, cell, 3 + off, L.subj)
        labs[mode] = lab
        w = sc.phase(L, 0, mode)
        assert w.min() >= 0.0 and w.max() <= 1.0
        ids = np.unique(lab)
        assert len(ids) > 30
        for v in ids:                                             # same patch -> one delay
            assert len(np.unique(w[lab == v])) == 1
        per = np.array([w[lab == v][0] for v in ids])
        assert len(np.unique(per)) > 0.9 * len(ids)               # different patches -> different delays
        # the subject is cut out of the cells: subject and background pixels never share a patch
        assert not set(np.unique(lab[L.subj])) & set(np.unique(lab[~L.subj]))
    # leaving and forming images use independent draws
    assert not np.allclose(sc.phase(L, 0, "out"), 1.0 - sc.phase(L, 0, "in"))
    # delay is unrelated to depth: the far wall alone already spans many delays
    assert len(np.unique(sc.phase(L, 0, "out")[:50, 40:90])) > 3


def test_random_mode_settles_to_exact_home():
    p = _p3(delay_mode="random", cell=12.0)
    ls = [_layer(s, p) for s in (1, 2)]
    sc = dd.Scene(ls, "right", p)
    # mid-travel, different patches are at different positions along D
    o = sc.point_offset("in", 12.0)(ls[1], 0)[0]
    assert len(np.unique(np.round(o, 4))) > 10
    fl = dd.frames(p, 2, loop=False)
    last = fl[-1]
    assert last.kind == "hold" and sc.home_of(last) == 1
    assert np.array_equal(sc.render_t(last.seg, last.t), sc.render_t(last.seg, last.t + 50))
    assert dd.psnr(sc.render_t(last.seg, last.t), ls[1].col) > 30.0
