"""v004 compose capabilities (camera track, contour_lines, solid vs sheer sandwich, signoff off, `rise` motion tracks).
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_fusion_v004_compose.py -q      (CPU, synthetic data)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import mask_tracks as MT  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

torch = C.torch
BACKENDS = ["numpy"] + (["torch"] if torch is not None else [])
W, H = 72, 128


@pytest.fixture(autouse=True)
def _restore_backend():
    prev = C.B
    yield
    C.B = prev


def piece_with(ops, signoff=None, n=6):
    p = {"fps": 24, "shots": [{"id": "s1", "frames": n}], "sizes": {"compose": [W, H]},
         "compose": {"default": {"ops": ops}}, "type_passes": ["matte"]}
    if signoff is not None:
        p["signoff"] = signoff
    return p


def bar_footage(n):
    f = np.zeros((n, H, W, 3), np.uint8)
    f[:, :, 20:25, :] = 255
    return f


def bar_x(img):
    g = np.asarray(img, np.float32)[..., 0]
    xs = np.arange(g.shape[1], dtype=np.float32)
    return float((g.sum(0) * xs).sum() / max(g.sum(), 1e-6))


# ============================================================================================ 1. camera track
def test_camera_track_constant_drift_scales_to_compose_size():
    n = 8
    flow = np.zeros((n, H // 2, W // 2, 2), np.float16)
    flow[..., 0], flow[..., 1] = 2.0, -1.0
    tr = RE.camera_track(flow, (W, H))
    assert tr.shape == (n, 2) and np.allclose(tr[0], 0)
    scale = W / (W // 2)                                   # 2.0
    assert np.allclose(np.diff(tr[:, 0]), 2.0 * scale, atol=1e-4)
    assert np.allclose(np.diff(tr[:, 1]), -1.0 * (H / (H // 2)), atol=1e-4)


def test_camera_track_smoothing_kills_jitter():
    rng = np.random.default_rng(0)
    n = 40
    flow = np.zeros((n, 16, 8, 2), np.float32)
    flow[..., 0] = 1.0 + rng.normal(0, 0.8, (n, 1, 1))
    raw = np.diff(RE.camera_track(flow, (8, 16), smooth=1)[:, 0])
    sm = np.diff(RE.camera_track(flow, (8, 16), smooth=5)[:, 0])
    assert sm.std() < 0.7 * raw.std()


@pytest.mark.parametrize("be", BACKENDS)
def test_cam_ops_move_type_right_by_flow_times_scale(be):
    n = 5                                                  # stays clear of the reflected border (bar x 20..24, shift <= 16)
    flow = np.zeros((n, H // 2, W // 2, 2), np.float16)
    flow[..., 0] = 2.0                                     # +2 px/frame to the right at flow res
    ops = RE.cam_ops(layers=("footage",), gain=1.0)
    assert ops[0]["op"] == "translate" and ops[0]["out"] == "cam_footage"
    piece = piece_with(ops, n=n)
    piece["compose"]["default"]["output"] = "cam_footage"
    with RE.use_backend(be):
        rec = RE.build_recipe(piece, "s1")
        assert "flow" in rec["inputs"] and RE.shot_needs_flow(piece, "s1")
        srcs = RE.open_sources(rec, footage=bar_footage(n), flow=flow)
        out, _ = RE.compose_frames(rec, srcs)
    xs = [bar_x(out[i].numpy() if hasattr(out[i], "numpy") else out[i]) for i in range(n)]
    d = np.diff(xs)
    assert np.allclose(d, 2.0 * (W / (W // 2)), atol=0.15), d       # +4 px per frame, positive = right
    xs0 = bar_x(bar_footage(1)[0])
    assert abs(xs[0] - xs0) < 0.05                                   # frame 0 untouched


def test_cam_arg_gain_and_axis_in_resolve():
    ctx = {"__cam__flow": np.array([10.0, -4.0], np.float32)}
    r = RE.resolve({"cam": "x", "gain": 0.5, "ref": "$flow"}, ctx, {}, 0.0, 0, 24)
    assert r == pytest.approx(5.0)
    assert RE.resolve({"cam": "y", "ref": "$flow"}, ctx, {}, 0.0, 0, 24) == pytest.approx(-4.0)


# ============================================================================================ 2. contour_lines
def disk_depth(cx=36, cy=64, r=22, hh=H, ww=W):
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - d / 50.0, 0, 1).astype(np.float32), (d < r).astype(np.float32)


def test_contour_iso_outline_scribble_shapes_and_range():
    depth, matte = disk_depth()
    for kw in (dict(mode="iso", src=depth, levels=5), dict(mode="outline", src=matte),
               dict(mode="scribble", src=matte, frame_idx=0)):
        a = C.contour_lines(width=1.0, min_len=4.0, **kw)
        assert a.shape == (H, W) and a.dtype == np.float32
        assert 0.0 <= a.min() and a.max() <= 1.0 and a.max() > 0.9 and (a > 0.5).sum() > 20, kw["mode"]


def test_contour_outline_sits_on_the_matte_edge():
    _, matte = disk_depth(r=22)
    a = C.contour_lines(matte, mode="outline", width=1.0, smooth=0.0, min_len=4.0)
    ys, xs = np.nonzero(a > 0.5)
    rad = np.sqrt((xs - 36) ** 2 + (ys - 64) ** 2)
    assert abs(float(np.median(rad)) - 22) < 1.5


def test_contour_tracks_the_subject_and_is_stable():
    _, m1 = disk_depth(cx=30)
    _, m2 = disk_depth(cx=34)
    a1 = C.contour_lines(m1, mode="outline", width=1.0, min_len=4.0)
    a2 = C.contour_lines(m2, mode="outline", width=1.0, min_len=4.0)
    cx = lambda a: float((np.nonzero(a > 0.5)[1]).mean())  # noqa: E731
    assert cx(a2) - cx(a1) == pytest.approx(4.0, abs=0.6)
    a1b = C.contour_lines(m1, mode="outline", width=1.0, min_len=4.0)
    assert np.array_equal(a1, a1b)                                    # deterministic


def test_contour_reveal_draws_a_fraction_of_the_arc():
    _, m = disk_depth()
    full = C.contour_lines(m, mode="outline", width=1.0, min_len=4.0)
    half = C.contour_lines(m, mode="outline", width=1.0, min_len=4.0, reveal=0.5)
    none = C.contour_lines(m, mode="outline", width=1.0, min_len=4.0, reveal=0.0)
    r = (half > 0.5).sum() / (full > 0.5).sum()
    assert 0.35 < r < 0.65 and none.max() == 0.0


def test_contour_iso_phase_flows_and_loops():
    depth, _ = disk_depth()
    a0 = C.contour_lines(depth, mode="iso", levels=4, phase=0.0, width=1.0, min_len=4.0)
    a5 = C.contour_lines(depth, mode="iso", levels=4, phase=0.5, width=1.0, min_len=4.0)
    a1 = C.contour_lines(depth, mode="iso", levels=4, phase=1.0, width=1.0, min_len=4.0)
    assert np.abs(a0 - a5).mean() > 0.002                               # the lines moved
    assert np.abs(a0 - a1).mean() < np.abs(a0 - a5).mean() * 0.5        # phase 1 == phase 0 shifted by one level: near-loop


def test_contour_scribble_boils_every_n_frames():
    _, m = disk_depth()
    f = lambda i: C.contour_lines(m, mode="scribble", frame_idx=i, boil=3, width=1.0, min_len=4.0, seed=2)  # noqa: E731
    assert np.array_equal(f(0), f(2))                                   # same boil bucket
    assert not np.array_equal(f(0), f(3))                               # re-jittered
    ring = C.contour_lines(m, mode="outline", width=1.0, min_len=4.0)
    assert not np.array_equal(f(0) > 0.5, ring > 0.5)                   # hand-drawn, not the plain outline


def test_contour_bad_mode_and_output_size():
    _, m = disk_depth()
    with pytest.raises(ValueError):
        C.contour_lines(m, mode="nope")
    a = C.contour_lines(m, mode="outline", size=[W * 2, H * 2], width=2.0, min_len=4.0)
    assert a.shape == (H * 2, W * 2)


@pytest.mark.parametrize("be", BACKENDS)
def test_contour_lines_in_a_recipe_feeds_type_from_edges_and_over(be):
    n = 3
    depth = np.stack([disk_depth()[0]] * n)
    ops = [{"op": "contour_lines", "out": "cl", "args": {"src": "$footage", "mode": "outline", "width": 1, "min_len": 4,
                                                           "reveal": {"timeline": [[0, 0.0], [0.1, 1.0]]},
                                                           "frame_idx": "$frame_idx"}},
           {"op": "over", "out": "image", "args": {"base": "$footage", "fg": [1.0, 0.5, 0.2], "alpha": "$cl"}}]
    foot = np.stack([(disk_depth()[1][..., None] * 255).repeat(3, -1).astype(np.uint8)] * n)
    piece = piece_with(ops, n=n)
    with RE.use_backend(be):
        rec = RE.build_recipe(piece, "s1")
        out, _ = RE.compose_frames(rec, RE.open_sources(rec, footage=foot))
    o = np.asarray(out[2]) if not hasattr(out[2], "numpy") else out[2].numpy()
    assert (o[..., 0] > 200).any() and ((o[..., 2] < 120) & (o[..., 0] > 200)).sum() > 10    # orange line over the disk
    # type_from_edges takes the lines as `edges`
    cl = C.contour_lines(disk_depth()[1], mode="outline", width=1.0, min_len=4.0)
    ta = np.zeros((H, W), np.float32)
    ta[50:80, 20:50] = 1
    tfe = C.type_from_edges(cl, ta, grow=6.0, line_width=1.0)
    assert tfe.shape == (H, W) and tfe.max() > 0.5 and tfe[:20].max() == 0.0


# ============================================================================================ 3. solid vs sheer sandwich
def _sand(**kw):
    foot = np.full((H, W, 3), 0.2, np.float32)
    rgb = np.full((H, W, 3), 0.9, np.float32)
    ta = np.ones((H, W), np.float32)
    return C.sandwich(foot, rgb, ta, kw.pop("matte"), **kw)


@pytest.mark.parametrize("be", BACKENDS)
def test_sandwich_solid_vs_sheer(be):
    C.set_backend(be)
    grey = np.full((H, W), 0.7, np.float32)
    sheer = _sand(matte=grey)
    solid = _sand(matte=grey, matte_threshold=0.5, matte_soft=0.08)
    assert sheer[0, 0, 0] == pytest.approx(0.9 + (0.2 - 0.9) * 0.7, abs=1e-4)      # 70 % subject over the type: see-through
    assert solid[0, 0, 0] == pytest.approx(0.2, abs=1e-4)                           # fully the subject
    low = _sand(matte=np.full((H, W), 0.3, np.float32), matte_threshold=0.5, matte_soft=0.08)
    assert low[0, 0, 0] == pytest.approx(0.9, abs=1e-4)                             # background: type shows fully
    none = _sand(matte=grey, matte_threshold=None)
    assert np.allclose(none, sheer)                                                  # None = old behaviour


@pytest.mark.parametrize("be", BACKENDS)
def test_sandwich_front_region_and_two_layer_chain(be):
    C.set_backend(be)
    foot = np.full((H, W, 3), 0.2, np.float32)
    m = np.zeros((H, W), np.float32)
    m[:, :36] = 1.0                                                                  # left half = subject
    ta = np.ones((H, W), np.float32)
    rgb = np.full((H, W, 3), 0.9, np.float32)
    behind = C.sandwich(foot, rgb, ta, m, matte_threshold=0.5)
    assert behind[10, 10, 0] == pytest.approx(0.2, abs=1e-4) and behind[10, 60, 0] == pytest.approx(0.9, abs=1e-4)
    front = np.zeros((H, W), np.float32)
    front[:64] = 1.0                                                                 # top half: type in front of the subject
    split = C.sandwich(foot, rgb, ta, m, matte_threshold=0.5, front=front)
    assert split[10, 10, 0] == pytest.approx(0.9, abs=1e-4)                          # subject, top: type in front
    assert split[100, 10, 0] == pytest.approx(0.2, abs=1e-4)                         # subject, bottom: type behind
    # two sandwich ops in one chain: type A behind the subject, type B in front of it only
    taA = np.zeros((H, W), np.float32)
    taA[:, 30:44] = 1.0
    taB = np.zeros((H, W), np.float32)
    taB[:, 30:44] = 1.0
    rgbB = np.full((H, W, 3), 0.6, np.float32)
    s1 = C.sandwich(foot, rgb, taA, m, matte_threshold=0.5)
    s2 = C.sandwich(s1, rgbB, taB, m, matte_threshold=0.5, invert=True)
    assert s2[10, 33, 0] == pytest.approx(0.6, abs=1e-4)           # on the subject (x<36): B in front
    assert s2[10, 40, 0] == pytest.approx(0.9, abs=1e-4)           # on the background: A, B hidden behind it
    assert s2[10, 5, 0] == pytest.approx(0.2, abs=1e-4)            # untouched subject


def test_behind_recipes_solid_and_sheer():
    rd = ROOT / "tools" / "fusion" / "recipes"
    solid = json.loads((rd / "behind.json").read_text(encoding="utf-8"))
    sheer = json.loads((rd / "behind_sheer.json").read_text(encoding="utf-8"))
    sa = [o for o in solid["compose"]["default"]["ops"] if o["op"] == "sandwich"][0]["args"]
    ha = [o for o in sheer["compose"]["default"]["ops"] if o["op"] == "sandwich"][0]["args"]
    assert sa["matte_threshold"] == 0.5 and sa["matte_soft"] == 0.08
    assert "matte_threshold" not in ha and ha["matte_gamma"] == 0.75
    assert sheer["technique"] == "behind_sheer"


# ============================================================================================ 4. signoff off
def test_signoff_off_skips_the_signoff_op():
    so = dict(text="MOTNGRFX", t0=0.1, px=30)
    ops = [{"op": "grade", "out": "image", "args": {"frame": "$footage", "gain": 1.0}}]
    on = RE.build_recipe(piece_with(ops, signoff=so, n=24), "s1", cut=1.0)
    off = RE.build_recipe(piece_with(ops, signoff=dict(so, off=True), n=24), "s1", cut=1.0)
    assert any(o["op"] == "over" for o in on["ops"]) and "signoff" in on["inputs"]
    assert not any(o["op"] == "over" for o in off["ops"]) and "signoff" not in off["inputs"]


# ============================================================================================ 5. rise tracks
def _matte_seq(F=40, h=128, w=72):
    m = np.zeros((F, h, w), np.float32)
    cv2.circle(m[0], (36, 80), 14, 1.0, -1)
    for i in range(1, F):
        m[i] = m[0]
    return m


def test_rise_tracks_go_up_and_stay_inside():
    assert "rise" in MT.STYLES
    m = _matte_seq()
    tr, meta = MT.build_tracks(m, "rise", n_points=20, start_frame=4, distance=60.0, margin=6.0)
    assert tr.shape[1] == 40 and meta["style"] == "rise"
    assert (tr[:, -1, 1] < tr[:, 0, 1] - 1.0).mean() > 0.6                        # most threads end higher (y smaller)
    dy = np.diff(tr[:, :, 1], axis=1)
    assert dy.max() < 0.51 and dy.min() < -0.5                                    # never fall (sway only moves x); they climb
    assert tr[..., 0].min() >= 6.0 - 1e-6 and tr[..., 0].max() <= 72 - 6.0 + 1e-6
    assert tr[..., 1].min() >= 6.0 - 1e-6 and tr[..., 1].max() <= 128 - 6.0 + 1e-6
    rain, _ = MT.build_tracks(m, "rain", n_points=20, start_frame=4, distance=60.0, margin=6.0)
    assert (rain[:, -1, 1] > rain[:, 0, 1]).mean() > 0.6                          # rain goes the other way


def test_rise_is_in_the_ltx_mask_style_tables():
    from tools.fusion import ltx_mask
    assert "rise" in ltx_mask.GRID_STYLES and "rise" in ltx_mask.STYLE_PROMPTS
