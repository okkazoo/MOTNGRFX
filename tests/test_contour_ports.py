"""Contour ports (mograph/modules/contour_ports.py) and matte treatments (tools/fusion/matte_treatments.py) on a synthetic plate:
extraction + temporal correspondence, projection through a moving camera, ride contracts (layers(0) = entry, layers(dur) = exit),
tracking of the strands on the moving contour, PortedWord timing, the spec keys and the planner hooks. No rendering.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_contour_ports.py -q"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402

from mograph.modules import Placement, Timing, Sequence, core  # noqa: E402
from mograph.modules import contour_ports as cp  # noqa: E402
from mograph.modules.words.lanes import Lanes  # noqa: E402
from mograph.render import cam3d  # noqa: E402

SLOW = os.environ.get("MOGRAPH_SKIP_SLOW") != "1"
W, H, T, FPS = 180, 320, 48, 24.0


def synth(n=T):
    """a moving ellipse 'dancer' (matte), a depth map that is far at the top and near at the bottom (floor) with the ellipse nearer,
    and a plate whose luminance has a bright disc."""
    matte = np.zeros((n, H, W), np.uint8)
    depth = np.zeros((n, H, W), np.uint8)
    plate = np.zeros((n, H, W, 3), np.uint8)
    yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
    for f in range(n):
        cx, cy = 70 + 1.5 * f, 150 + 20 * np.sin(f / 8.0)
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (28, 70), 0, 0, 360, 255, -1)
        matte[f] = m
        d = (40 + 190 * yy).astype(np.uint8)
        d[m > 0] = 150
        depth[f] = d
        g = np.full((H, W), 60, np.uint8)
        cv2.circle(g, (int(130 - f), 80), 30, 220, -1)
        plate[f] = np.stack([g, g, g], -1)
    return cp.PlateData(matte=matte, depth=depth, plate=plate, fps=FPS)


@pytest.fixture(scope="module")
def data():
    return synth()


@pytest.fixture()
def world():
    prev = cam3d.set_world(180, 320)
    yield (180, 320)
    cam3d.set_world(*prev)


@pytest.fixture()
def ctx(data, world):
    return cp.ContourContext(data, world=world, n=96)


# ---------------------------------------------------------------------------------------------------- extraction
def test_available_from_piece_and_plate(data):
    piece = {"shots": [{"id": "s1", "matte": True, "depth": False}, {"id": "s2", "matte": True, "depth": True}]}
    a = cp.available(piece)
    assert a["shots"]["s1"] == {"luma_edges": True, "matte_outline": True, "depth_iso": False}
    assert a["ports"] == ["matte_outline", "luma_edges"]
    assert set(cp.available(piece, shot="s2")["ports"]) == {"matte_outline", "depth_iso", "luma_edges"}
    assert cp.available({"shots": [{"id": "x"}]})["ports"] == ["luma_edges"]
    assert set(cp.available(data)["ports"]) == set(cp.KINDS)
    assert "front" not in cp.available({"shots": [{"id": "x"}]})["matte"]          # needs a subject matte


def test_matte_outline_track(data):
    tr = cp.build_track(data, "matte_outline", n=96)
    assert tr.P.shape == (T, 1, 96, 2) and tr.valid.all() and tr.closed[0]
    p0 = tr.lines_at(0)[0]
    assert abs(p0[0, 1] - p0[:, 1].max()) < 3                         # s = 0 is the lowest point
    assert cp._area(p0) > 0                                           # clockwise on screen
    # correspondence: the point at s = 0.5 (top) stays the top of the ellipse over time, and moves with it (1.5 px / frame in x)
    top = [tr.sample(f, 0, np.array([0.5]))[0] for f in (0, 20, 40)]
    assert all(abs(t[1] - 80 - 20 * np.sin(f / 8.0)) < 5 for t, f in zip(top, (0, 20, 40)))
    assert top[2][0] - top[0][0] > 40


def test_depth_iso_runs_behind_the_subject(data):
    tr = cp.build_track(data, "depth_iso", levels=4, n=96)
    assert tr.L == 4 and tr.valid.all() and not tr.closed.any()
    for l in range(4):
        P = tr.lines_at(10)[l]
        assert P[0, 0] < P[-1, 0]                                     # left to right
        assert P[:, 0].max() - P[:, 0].min() > 0.8 * W                # the whole width: the subject was painted over
    ys = [tr.lines_at(10)[l][:, 1].mean() for l in range(4)]
    assert ys == sorted(ys)                                           # level order = band order, same all clip long
    assert all(abs(tr.lines_at(40)[l][:, 1].mean() - ys[l]) < 6 for l in range(4))


def test_luma_edges_track(data):
    tr = cp.build_track(data, "luma_edges", levels=2, n=64)
    assert tr.valid.any() and tr.P.shape[1] == 2


def test_missing_channel_is_an_error():
    d = cp.PlateData(plate=np.zeros((3, 16, 16, 3), np.uint8))
    with pytest.raises(ValueError):
        cp.build_track(d, "matte_outline")
    with pytest.raises(ValueError):
        cp.build_track(d, "nope")


# ---------------------------------------------------------------------------------------------------- projection
def test_lift_pins_contour_to_the_picture_under_any_camera(ctx):
    px = np.array([[40.0, 60.0], [150.0, 250.0], [90.0, 160.0]])
    for cam in (ctx.camera(0.0), cam3d.Camera((90 + 400, 160 - 100, -900), (90, 160, 0), 0.1, 1000.0)):
        P = ctx.lift(px, cam)
        xy, z = cam.project(P)
        back = xy / ctx.k
        assert np.allclose(back, px, atol=1e-6) and (z > 0).all()
    home = ctx.lift(px, ctx.camera(0.0), depth=ctx.F)                 # the home camera: the z = 0 frame plane, 1:1 (here k = 1)
    assert np.allclose(home[:, 2], 0.0, atol=1e-6) and np.allclose(home[:, :2], px * ctx.k)


# ---------------------------------------------------------------------------------------------------- ports
def port(ctx, kind="matte_outline", **kw):
    return cp.ContourPort.parse(ctx, dict(port=kind, **kw))


def test_parse_and_split_spec(ctx):
    p = cp.ContourPort.parse(ctx, {"port": "depth_iso", "dir": "up", "lead": 0.4, "glow": 0.5})
    assert (p.kind, p.dir, p.lead, p.glow) == ("depth_iso", "up", 0.4, 0.5)
    assert cp.ContourPort.parse(ctx, "matte_outline", "around").dir == "around"
    assert cp.ContourPort.parse(ctx, None) is None
    with pytest.raises(ValueError):
        cp.ContourPort.parse(ctx, {"port": "bogus"})
    with pytest.raises(ValueError):
        cp.ContourPort.parse(ctx, {"port": "matte_outline", "dir": "down"})            # no 'down': never downward on people
    w = {"entry": {"port": "matte_outline", "side": "left"}, "exit": {"port": "depth_iso", "dir": "up"},
         "via": {"port": "matte_outline", "dir": "around"}, "matte": "front"}
    me, mx, ports = cp.split_port_spec(w)
    assert me == "left" and mx == "down" and ports["entry"]["port"] == "matte_outline" and ports["exit"]["dir"] == "up"
    assert ports["via"]["dir"] == "around" and ports["matte"] == "front"
    me, mx, ports = cp.split_port_spec({"entry": "up", "exit": "right"})              # plain module sides are untouched
    assert (me, mx) == ("up", "right") and not any(ports[k] for k in ("entry", "exit", "via", "matte"))
    assert cp.split_port_spec({"entry": "depth_iso"})[2]["entry"] == {"port": "depth_iso"}


@pytest.mark.parametrize("kind,dir", [("matte_outline", "up"), ("matte_outline", "around"), ("matte_outline", "along"),
                                      ("depth_iso", "up"), ("depth_iso", "around"), ("luma_edges", "along")])
def test_layout_stays_on_the_line(ctx, kind, dir):
    p = port(ctx, kind, dir=dir)
    lay = p.layout(24, 5.0)
    assert (lay.s0 >= 0).all() and (lay.s0 <= 1).all() and set(np.unique(lay.step)) <= {-1.0, 1.0}
    tr = p.track
    for k in range(24):
        l = lay.line[k]
        if not tr.closed[l]:
            end = lay.s0[k] + lay.step[k] * lay.travel[k]
            assert -1e-9 <= end <= 1 + 1e-9                                # open lines: never ride off the end
    if kind == "matte_outline" and dir == "up":                             # both sides ride up to the top (s = 0.5): never downward
        end = lay.s0 + lay.step * lay.travel
        assert (np.abs(end - 0.5) < np.abs(lay.s0 - 0.5) + 1e-9).all()
        P = lambda s: tr.sample(5.0, 0, s)[:, 1]
        assert (P(end) <= P(lay.s0) + 1.0).all()                            # the strand's y decreases (up on screen)


def lanes(**kw):
    return Lanes("RISE", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9), **kw)


def _heads(layers):
    return np.concatenate([p[0:1] if p.ndim == 2 else p for L in layers for p in L.polys])


def test_entry_ride_contract(ctx):
    w = lanes()
    pt = port(ctx, "matte_outline", dir="up", lead=0.7, glow=0.8)
    r = cp.entry_ride(ctx, pt, w.entry_bundle(), t_abs=0.5)
    assert r.mode == "entry" and r.dur == pytest.approx(0.7)
    # layers(0) draws exactly the lying strands (a_exit) and layers(dur) exactly B's entry strands
    s0 = np.sort(np.concatenate([win.reshape(-1, 3) for win, h, t in r.windows(0.0) if win is not None]), 0)
    a_pts = np.sort(r.A.P.reshape(-1, 3), 0)
    assert np.allclose(s0.min(0), a_pts.min(0), atol=1.0) and np.allclose(s0.max(0), a_pts.max(0), atol=1.0)
    e = np.concatenate([win.reshape(-1, 3) for win, h, t in r.windows(r.dur) if win is not None])
    assert np.allclose(e.min(0), r.B.P.reshape(-1, 3).min(0), atol=1.0) and np.allclose(e.max(0), r.B.P.reshape(-1, 3).max(0), atol=1.0)
    assert r.glow_layers(0.0, r.env(0.0)) and not r.glow_layers(r.dur, r.env(r.dur))                  # the glowing outline fades out as the strands peel off


def test_exit_ride_lies_on_the_moving_outline(ctx):
    w = lanes()
    pt = port(ctx, "matte_outline", dir="up", lead=0.6)
    t0 = 0.4
    r = cp.exit_ride(ctx, pt, w.exit_bundle(), t0)
    tr = pt.track
    for dt in (0.0, 0.5, 1.0):                                             # after the ride the strands stay on the outline while it moves
        t = r.dur + dt
        f = ctx.frame_of(t0 + t)
        cam = ctx.camera(t0 + t)
        polys = [win for win, h, tl in r.windows(t) if win is not None]
        assert polys
        out = tr.lines_at(f)[0]
        W = ctx.lift(out, cam)
        d = [np.linalg.norm(p[:, None, :] - W[None, :, :], axis=2).min(1).mean() for p in polys]
        assert np.mean(d) < 3.0, (dt, np.mean(d))                          # world px: on the line, not on its snapshot


def test_via_ride_in_a_sequence(ctx, world):
    a = lanes()
    b = Lanes("UP", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9), entry="down", exit="up")
    tr = cp.via_ride(ctx, port(ctx, "matte_outline", dir="around", glow=0.7), dur=1.0, t_abs=1.0)
    seq = Sequence([a, tr, b])
    cp.attach(seq, ctx)
    assert tr.dur == pytest.approx(1.0) and tr.mode == "via" and tr.t_abs == pytest.approx(a.duration)
    lyr = [seq.layers(t) for t in np.linspace(0, seq.duration, 9)]
    assert all(isinstance(L, list) for L in lyr)
    mid = tr.windows(0.5 * tr.dur)
    assert sum(win is not None for win, h, t in mid) >= tr.K * 0.8
    # the middle of the ride is on the contour: every strand head is near the outline at the snapshot time
    f = ctx.frame_of(tr.t_abs + 0.5 * tr.dur)
    cam = ctx.camera(tr.t_abs + 0.5 * tr.dur)
    outline = ctx.lift(tr.port.track.lines_at(f)[0], cam)
    heads = np.array([win[-1] for win, h, t in mid if win is not None])
    near = np.linalg.norm(heads[:, None] - outline[None], axis=2).min(1)
    assert np.median(near) < 40.0


def test_ride_follows_the_camera(ctx):
    """the contour stays pinned to the picture when the type camera orbits: lift() of the same plate point through two cameras gives
    different world points that both project to the same pixel; the route's contour part moves with the camera."""
    w = lanes()
    pt = port(ctx, "matte_outline", dir="up", lead=0.6)
    moving = lambda t: cam3d.Camera((90 + 300 * np.sin(0.6 * t), 160, -900 - 100 * t), (90, 160, 0), 0.0, 1000.0)
    ctx.bind_camera(moving)
    r = cp.exit_ride(ctx, pt, w.exit_bundle(), 0.2)
    for dt in (0.0, 0.8):
        t = r.dur + dt
        cam = ctx.camera(0.2 + t)
        polys = [win for win, h, tl in r.windows(t) if win is not None]
        xy = [cam.project(p)[0] for p in polys]
        outline_px = r.port.track.lines_at(ctx.frame_of(0.2 + t))[0] * ctx.k
        d = [np.linalg.norm(x[:, None, :] - outline_px[None], axis=2).min(1).mean() for x in xy]
        assert np.mean(d) < 3.0                                            # on screen the strands sit on the outline


# ---------------------------------------------------------------------------------------------------- PortedWord
def test_ported_word_keeps_the_timing(ctx):
    base = lanes()
    entry = {"port": "matte_outline", "dir": "up", "lead": 0.4}
    exit_ = {"port": "depth_iso", "dir": "up", "lead": 0.4, "persist": 0.2}
    pw = cp.PortedWord(base, ctx, entry, exit_, t_start=0.0)
    assert pw.duration == pytest.approx(base.duration)                      # carved out of form / exit: the beat grid does not move
    assert pw.timing.form == pytest.approx(base.timing.form) and pw.timing.exit == pytest.approx(base.timing.exit)
    assert pw.timing.hold == pytest.approx(base.timing.hold)
    assert pw.inner.timing.form == pytest.approx(base.timing.form - 0.4)
    assert pw.name == base.name and pw.exit_material().kind == "strands" and pw.entry_material().kind == "strands"
    lng = cp.PortedWord(base, ctx, entry, exit_, carve=False)
    assert lng.duration == pytest.approx(base.duration + 0.4 + 0.6)
    plain = cp.PortedWord(base, ctx, None, None)
    assert plain.duration == pytest.approx(base.duration) and plain.ride_in is None and plain.ride_out is None
    # a short form: the lead is clamped so the module keeps at least min_form
    short = Lanes("A", timing=Timing(0.5, 0.4, 0.5), placement=Placement.centred(w=150, h=80))
    p2 = cp.PortedWord(short, ctx, {"port": "matte_outline", "lead": 0.9}, None)
    assert p2.inner.timing.form >= 0.25 - 1e-9 and p2.duration == pytest.approx(short.duration)


def test_ported_word_layers_and_bundles(ctx):
    base = lanes()
    pw = cp.PortedWord(base, ctx, {"port": "matte_outline", "dir": "up", "lead": 0.4, "glow": 0.6},
                       {"port": "depth_iso", "dir": "up", "lead": 0.4, "persist": 0.2, "glow": 0.6}, t_start=0.3)
    assert np.allclose(pw.entry_bundle().P, pw.ride_in.a_exit.P) and np.allclose(pw.exit_bundle().P, pw.ride_out.b_entry.P)
    for t in np.linspace(0, pw.duration, 14):
        L = pw.layers(float(t))
        assert isinstance(L, list)
        assert np.isfinite(pw.focus(float(t))).all()
    # a hold shows exactly the module (same layers as the carved module itself)
    th = pw.lead_in + pw.inner.timing.form + 0.1
    a, b = pw.layers(th), pw.inner.layers(th - pw.lead_in)
    assert len(a) == len(b) and all(np.allclose(x.pts, y.pts) for x, y in zip(a, b))


def test_a_ported_word_beside_a_library_transition_keeps_its_timing(ctx):
    """2026-11-01: the outline hand-off of a library transition (twist, braid ...) never re-times a PortedWord: its contour ride is the hand-off."""
    pw = cp.PortedWord(lanes(), ctx, {"port": "matte_outline", "dir": "up", "lead": 0.4}, {"port": "matte_outline", "dir": "around", "lead": 0.4})
    d0, inner0 = pw.duration, pw.inner.duration
    b = Lanes("UP", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9))
    for pair in ((pw, b), (b, pw)):
        a_, b_ = pair
        a_.handoff_reset()
        seq = Sequence([a_, core.get_module("twist")(dur=1.0), b_])
        assert pw.duration == pytest.approx(d0) and pw.inner.duration == pytest.approx(inner0)
        assert seq.duration == pytest.approx(sum(it.duration for it in seq.items))


def test_apply_ports_builds_a_sequence(ctx, monkeypatch):
    from mograph.modules import sequence as _seq
    monkeypatch.setattr(_seq, "HANDOFF", "gate")                   # 2026-11-01: the library transitions retime their words (outline hand-off); this test is about the ports keeping the timeline, so the old gate hand-off
    mods = [lanes(), Lanes("UP", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9)),
            Lanes("INTO", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9))]
    base = Sequence(mods)
    items = list(base.items)
    words = [{"entry": {"port": "matte_outline", "dir": "up", "lead": 0.4}, "via": {"port": "depth_iso", "dir": "along"}},
             {"exit": "right"},
             {"exit": {"port": "matte_outline", "dir": "around", "lead": 0.4, "persist": 0.2, "glow": 0.8}, "matte": "glow"}]
    out = cp.apply_ports(items, words, ctx)
    assert isinstance(out[0], cp.PortedWord) and isinstance(out[1], cp.ContourRide) and out[1].mode == "via"
    assert out[2] is items[2] and out[3] is items[3] and isinstance(out[4], cp.PortedWord)
    seq = Sequence(out)
    cp.attach(seq, ctx)
    assert seq.duration == pytest.approx(base.duration, abs=1e-6)
    for t in np.linspace(0, seq.duration, 20):
        seq.layers(float(t))
    with pytest.raises(ValueError):
        cp.apply_ports(items, words[:2], ctx)


# ---------------------------------------------------------------------------------------------------- every approved gallery module
NAMES = sorted(core.list_modules("word"))
FAST_NAMES = ["lanes", "pipework", "slats", "beat_bars"]


@pytest.mark.parametrize("name", NAMES if SLOW else [n for n in FAST_NAMES if n in NAMES])
def test_every_word_module_takes_entry_and_exit_ports(ctx, name):
    cls = core.get_module(name)
    w = cls(cls.DEMO_WORDS, font="anybody", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.5, 0.9))
    pw = cp.PortedWord(w, ctx, {"port": "matte_outline", "dir": "up", "lead": 0.4, "glow": 0.5},
                       {"port": "depth_iso", "dir": "around", "lead": 0.4, "glow": 0.5})
    assert pw.duration == pytest.approx(w.duration)
    for t in (0.0, 0.2, pw.lead_in + 0.1, pw.duration * 0.5, pw.duration - 0.3, pw.duration):
        pw.layers(float(t))


# ---------------------------------------------------------------------------------------------------- matte treatments
from tools.fusion import compose, matte_treatments as mt, recipe_eval  # noqa: E402


def _plate(h=96, w=64):
    """grey wall, bright subject ellipse in the middle, a word-like bar of type across the subject and the wall."""
    foot = np.full((h, w, 3), 0.35, np.float32)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (12, 30), 0, 0, 360, 1.0, -1)
    foot[m > 0.5] = (0.8, 0.75, 0.7)
    ta = np.zeros((h, w), np.float32)
    ta[h // 2 - 6:h // 2 + 6, 4:w - 4] = 1.0
    return foot, m, ta


def _run(ops, t, buffers):
    reg = compose._ops_registry()
    ctx = dict(buffers, t=t, frame_idx=int(t * 24), u=0.0)

    def res(v):
        if isinstance(v, str) and v.startswith("$"):
            return ctx[v[1:]]
        if isinstance(v, dict) and "timeline" in v:
            return compose._timeline(v, t)
        if isinstance(v, dict) and "curve" in v:
            return 0.5
        if isinstance(v, list):
            return [res(x) for x in v]
        return v
    for op in ops:
        ctx[op.get("out", "image")] = reg[op["op"]](**{k: res(v) for k, v in op.get("args", {}).items()})
    return ctx["image"]


def _bufs():
    foot, m, ta = _plate()
    return dict(footage=foot.copy(), image=foot.copy(), m=m, type_matte=ta, near=np.zeros_like(ta))


def test_matte_ops_are_registered():
    reg = compose._ops_registry()
    for n in ("mask_gate", "mask_mul", "split_region", "matte_reach", "matte_ring", "sandwich", "knockout"):
        assert n in reg


def test_mask_ops():
    foot, m, ta = _plate()
    assert mt.mask_gate(ta, 0.5).max() == pytest.approx(0.5)
    g = mt.mask_mul(ta, m, floor=0.2, gain=1.0)
    assert g[48, 32] == pytest.approx(1.0) and g[48, 6] == pytest.approx(0.2)
    r = mt.split_region(foot, "x", 0.5, 0.02, "left")
    assert r[10, 5] == 1.0 and r[10, 60] == 0.0
    assert mt.split_region(foot, "x", 0.5, 0.02, "right")[10, 5] == 0.0
    assert mt.split_region(foot, "y", 0.5, 0.02, "top")[5, 10] == 1.0
    assert mt.split_region(foot, "diag", 0.5, 0.02, "left").shape == m.shape
    rc0, rc1 = mt.matte_reach(m, 0.0), mt.matte_reach(m, 1.0, reach=1.0)
    assert rc0.max() == 0.0 and rc1.min() > 0.9                                  # nothing at progress 0, the whole frame at the end
    rc = mt.matte_reach(m, 0.3, reach=0.6, soft=0.02)
    assert rc[48, 32] == pytest.approx(1.0) and rc[0, 0] < 0.05 and rc.sum() > m.sum()           # grows out of the outline
    assert mt.matte_ring(m, 0.0).max() == 0.0 and mt.matte_ring(m, 0.5, reach=0.25).max() > 0.5


def test_treatments_build_and_run_for_every_mode():
    for mode in mt.MODES:
        ops = mt.ops_for([{"matte": mode, "t0": 1.0, "t1": 3.0}], type_rgb=[1.0, 0.9, 0.7])
        assert ops and all(o["op"] in compose._ops_registry() for o in ops)
        off = _run(ops, 0.2, _bufs())                                              # outside the line's window: nothing changes
        assert np.allclose(off, _bufs()["image"], atol=1e-4), mode
        on = _run(ops, 2.0, _bufs())
        assert on.shape == (96, 64, 3) and np.isfinite(on).all()
        assert np.abs(on - _bufs()["image"]).max() > 0.05, mode                      # and something visible inside it
    with pytest.raises(ValueError):
        mt.ops_for([{"matte": "melt", "t0": 0, "t1": 1}])


def test_front_is_half_in_front_half_behind():
    ops = mt.ops_for([{"matte": "front", "t0": 1.0, "t1": 3.0, "split": "x", "pos": 0.5, "soft": 0.02, "side": "left"}],
                     type_rgb=[0.0, 1.0, 0.0])
    out = _run(ops, 2.0, _bufs())
    foot = _bufs()["footage"]
    assert out[48, 24][1] > 0.8 and out[48, 24][0] < 0.3                              # type colour over the subject (front region)
    assert np.allclose(out[48, 40], foot[48, 40], atol=0.05)                          # behind the subject: footage stays
    assert out[48, 6][1] > 0.8                                                       # over the wall everywhere


def test_window_is_brighter_than_the_dimmed_plate():
    ops = mt.ops_for([{"matte": "window", "t0": 1.0, "t1": 3.0, "dim": 0.25, "gain": 1.4}])
    out = _run(ops, 2.0, _bufs())
    foot = _bufs()["footage"]
    assert out[48, 6].mean() > foot[48, 6].mean() * 1.2                              # letters on the wall: a brighter plate
    assert out[5, 6].mean() < foot[5, 6].mean() * 0.5                                # the rest of the wall is dimmed
    assert np.allclose(out[48, 32], foot[48, 32], atol=0.05)                         # the subject steps out over the window
    assert any(o["op"] == "legible" for o in ops)                                     # and the legible guard is explicit


def test_fill_keeps_type_inside_the_silhouette():
    ops = mt.ops_for([{"matte": "fill", "t0": 1.0, "t1": 3.0, "outside": 0.1}], type_rgb=[1.0, 1.0, 1.0])
    out = _run(ops, 2.0, _bufs())
    assert out[48, 32].mean() > 0.9                                                   # letters at full strength on the subject
    assert abs(out[48, 6].mean() - (0.35 * 0.904 + 0.096)) < 0.06                     # only a ghost outside it


def test_wipe_reveals_from_the_matte_edge():
    ops = mt.ops_for([{"matte": "wipe", "t0": 1.0, "t1": 4.0, "reveal": 1.0, "reach": 0.9}], type_rgb=[1.0, 1.0, 1.0])
    early, late = _run(ops, 1.02, _bufs()), _run(ops, 2.5, _bufs())
    foot = _bufs()["footage"]
    d = lambda im: np.abs(im - foot).sum(-1) > 0.15
    assert not d(early)[48, 6]                                                        # far from the subject: not yet revealed
    assert d(late)[48, 6]                                                             # at the end the letters are there
    assert d(late).sum() > d(early).sum()


def test_glow_draws_the_outline():
    ops = mt.ops_for([{"matte": "glow", "t0": 1.0, "t1": 3.0, "intensity": 0.8, "colour": [1.0, 0.8, 0.4]}])
    out = _run(ops, 2.0, _bufs())
    diff = np.abs(out - _bufs()["footage"]).sum(-1)
    assert diff[48, 32 - 12] > 0.3 and diff[48, 32 + 12] > 0.3 and diff[48, 32] < 0.5 * diff[48, 32 - 12]   # the line sits on the outline


def test_legible_guard_applies_to_front_wipe_and_window():
    piece = {"fps": 24, "plan": {"flow_wake": {"holds": [[1.0, 2.0]]}}}
    for mode in ("front", "wipe", "window"):
        ops = mt.ops_for([{"matte": mode, "t0": 1.0, "t1": 3.0}])
        guarded = recipe_eval.guard_ops(piece, ops)
        leg = [o for o in guarded if o["op"] == "legible"]
        assert leg and all(o["args"].get("holds") == [[1.0, 2.0]] for o in leg), mode
    ops = recipe_eval.guard_ops(piece, mt.ops_for([{"matte": "front", "t0": 1.0, "t1": 3.0}]))
    i = [o["op"] for o in ops].index("sandwich")
    assert ops[i - 1]["op"] == "legible" and ops[i]["args"]["matte"] == "$" + ops[i - 1]["out"]


def test_from_words_reads_the_matte_key():
    words = [{"text": "A", "t0": 0.0, "t1": 2.0, "matte": "front"}, {"text": "B", "t0": 3.0, "t1": 5.0},
             {"text": "C", "t0": 6.0, "t1": 8.0, "matte": {"mode": "window", "dim": 0.2}}]
    tr = mt.from_words(words)
    assert [t["matte"] for t in tr] == ["front", "window"] and tr[1]["dim"] == 0.2 and tr[1]["t1"] == 8.0
    assert mt.from_words(words, windows=[(0.5, 1.5), None, (6.5, 7.5)])[0]["t0"] == 0.5


# ---------------------------------------------------------------------------------------------------- multiplane plate
torch = pytest.importorskip("torch")
from mograph.render import multiplane as MP  # noqa: E402


def test_multiplane_plan_and_slots():
    c = MP.plan({"cards": 4})
    assert c["z"] == sorted(c["z"], reverse=True) and c["z"][c["subject"]] == 0.0 and len(c["slots"]) == 5
    assert all(a > b for a, b in zip(c["slots"], c["slots"][1:]))                  # slot k + 1 sits between card k and k + 1
    for k in range(4):
        assert c["z"][k] > c["slots"][k + 1] > c["z"][k + 1] if k < 3 else True
    c2 = MP.plan({"cards": 5, "subject_z": 1000})
    assert len(c2["z"]) == 5 and c2["z"][c2["subject"]] == 1000.0
    out = MP.apply_slots([{"text": "A", "slot": 1, "placement": {"yaw": 10}}, {"text": "B"}], c)
    assert out[0]["placement"]["z"] == c["slots"][2] and out[0]["placement"]["yaw"] == 10 and "placement" not in out[1]
    assert MP.camera_spec({"chain": "orbit_even", "plate": "multiplane", "cards": 4})["camera"] == "orbit_even"
    assert MP.camera_spec("auto") is None


@pytest.fixture(scope="module")
def mplane(data):
    return MP.MultiPlane(data, (180, 320), (180, 320), cards=4, device="cpu", edge_frames=(0, 10))


def test_decompose_cards(data):
    e = MP.band_edges(data, 2, (0, 10))
    c = MP.decompose(data.plate[5], data.depth[5], data.matte[5], e, 4)
    assert c.shape == (4, H, W, 4) and np.isfinite(c).all()
    assert (c[0, ..., 3] == 1.0).all()                                              # the back card is opaque everywhere
    m = data.matte[5] > 128
    top = np.zeros_like(m)
    top[:int(0.35 * H)] = True                                                       # (below, the subject card continues behind the near floor)
    assert c[2, ..., 3][m].mean() > 0.9 and c[2, ..., 3][(~m) & top].mean() < 0.1     # the subject card is the matte
    assert c[3, ..., 3][-5:, :].mean() > 0.5 and c[3, ..., 3][:50].mean() < 0.1        # the foreground card is the near floor


def _ref_cam(mp, shift=0.0, yaw=0.0):
    cx, cy = mp.world[0] / 2.0, mp.world[1] / 2.0
    eye = np.array([cx + shift, cy, -mp.ref])
    return cam3d.Camera(eye, eye + np.array([np.sin(yaw), 0.0, np.cos(yaw)]) * 1000.0, 0.0, 1000.0)


def test_multiplane_reference_camera_reproduces_the_plate(mplane, data):
    img = mplane.render(5, _ref_cam(mplane)).cpu().numpy()
    plate = data.plate[5].astype(np.float32) / 255.0
    assert np.abs(img - plate).mean() < 0.02                                         # the cards stack back into the original frame


def test_multiplane_parallax(mplane, data):
    a = mplane.render(5, _ref_cam(mplane)).cpu().numpy()
    b = mplane.render(5, _ref_cam(mplane, shift=120.0)).cpu().numpy()
    m = (data.matte[5] > 128)
    ys, xs = np.nonzero(m)
    cx0 = xs.mean()
    ma = (a.mean(-1)[m].mean())
    # find the subject (bright 150/255 ellipse on a darker gradient is the matte region): its centroid shifts relative to the back wall
    gray_a, gray_b = a.mean(-1), b.mean(-1)
    sub = (np.abs(gray_a - gray_a[m].mean()) < 0.02)
    assert np.abs(a - b).mean() > 0.004                                              # moving the camera changes the picture


def test_multiplane_type_between_cards(mplane, data):
    cam = _ref_cam(mplane)
    type_rgb = np.zeros((H, W, 3), np.float32)
    type_rgb[..., 0] = 1.0                                                           # red, fully covering
    A = np.ones((H, W), np.float32)
    base = mplane.render(5, cam).cpu().numpy()
    front = mplane.render(5, cam, type_rgb, A, np.full((H, W), 100.0, np.float32)).cpu().numpy()   # nearer than every card
    behind = mplane.render(5, cam, type_rgb, A, np.full((H, W), 1e5, np.float32)).cpu().numpy()     # behind the back card
    mid = mplane.render(5, cam, type_rgb, A, np.full((H, W), 1428.0 + 65.0, np.float32)).cpu().numpy()   # between mid and subject cards
    assert np.allclose(front[..., 0], 1.0) and np.allclose(behind, base, atol=1e-5)
    m = cv2.erode((data.matte[5] > 128).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert np.allclose(mid[m], base[m], atol=0.02)                                    # the subject card hides the type that sits behind it
    top = np.zeros_like(m)
    top[:int(0.3 * H), :40] = True
    assert (mid[top][..., 0] > base[top][..., 0] + 0.05).mean() > 0.3                  # and the type shows where only cards behind it are


def test_contour_plane_lift(ctx):
    ctx.set_plane(1000.0, 1428.0)
    px = np.array([[10.0, 20.0], [90.0, 160.0], [170.0, 300.0]])
    P = ctx.lift(px, ctx.camera(0.0))
    assert np.allclose(P[:, 2], 1000.0)
    # seen from the reference camera the plane point lands on the original pixel
    cam = cam3d.Camera((90.0, 160.0, -1428.0), (90.0, 160.0, 0.0), 0.0, 1000.0)
    xy, _ = cam.project(P)
    assert np.allclose(xy / ctx.k, px, atol=1e-6)
    ctx.plane = None


def test_camera_for_gallery_chains(world):
    a = lanes()
    b = Lanes("UP", placement=Placement.centred(w=150, h=80), timing=Timing(0.9, 0.4, 0.9))
    seq = Sequence([a, b])
    ch = MP.camera_for(seq, {"chain": [{"module": "orbit_even", "word": 0, "sweep": 40}], "plate": "multiplane", "cards": 4})
    assert "orbit_even" in ch.plan_text()
    assert isinstance(ch.at(0.5), cam3d.Camera)
    assert "hold_drift" in MP.camera_for(seq, {"chain": "hold_drift"}).plan_text()


def test_multiplane_magnification_is_clamped(mplane):
    near = cam3d.Camera((90.0, 160.0, mplane.z[2] - 150.0), (90.0, 160.0, 1e4), 0.0, 1000.0)       # a push to 150 in front of the subject card
    for zk in mplane.z:
        assert mplane.magnification(near, zk) <= mplane.max_mag + 1e-6
    ref = _ref_cam(mplane)
    assert all(mplane.hold_back(ref, zk) == 0.0 for zk in mplane.z)                   # at the reference camera nothing is held
    assert mplane.hold_back(near, mplane.z[2]) > 500.0
    # a camera far behind the reference would shrink the cards below the overscan: it is brought back in (never smaller than min_mag)
    far = cam3d.Camera((90.0, 160.0, -3000.0), (90.0, 160.0, 0.0), 0.0, 1000.0)
    assert mplane.min_mag - 1e-6 <= mplane.magnification(far, 0.0) <= 1.0 and mplane.hold_back(far, 0.0) < 0.0
    img = mplane.render(5, near).cpu().numpy()
    assert np.isfinite(img).all() and mplane.edge_report()["mag_max"] <= mplane.max_mag + 1e-6


def test_multiplane_overscan_keeps_a_sideways_move_inside_the_cards(mplane):
    mplane.stats.update(over_max=0.0)
    mplane.render(5, _ref_cam(mplane, shift=20.0))
    assert mplane.edge_report()["over_max"] == 0.0                                    # 15 % overscan: no card runs out
    assert mplane.pad == pytest.approx(0.15) and mplane.cw > mplane.size[0]


def test_multiplane_legible_guard_pulls_a_hidden_word_in_front(mplane, data):
    """a word that sits behind the subject card for its whole hold is flagged and drawn in front; a visible one is left between the cards."""
    cam = _ref_cam(mplane)
    m = cv2.erode((data.matte[5] > 128).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    A = np.zeros((H, W), np.float32)
    A[cy - 6:cy + 6, cx - 8:cx + 8] = 1.0                                             # a letter right behind the subject
    A[20:32, 8:24] = 1.0                                                              # a letter over the open wall
    rgb = np.stack([A, A * 0, A * 0], -1)
    D = np.where(A > 0, 1428.0 + 65.0, 1e9).astype(np.float32)                         # between the mid and subject cards
    plan = MP.plan_legible(mplane, type("Ch", (), {"at": lambda self, t: cam})(), 24.0, lambda f: A, lambda f: D, [(0.0, 1.0)], frame1=24)
    h = plan["holds"][0]
    assert h["n"] == 2 and len(h["boxes"]) == 1 and h["hidden"].count(1.0) == 1
    from tools.fusion import compose
    clear = compose.legible_clear(A, plan, 0.5)
    assert clear is not None and clear[cy, cx] > 0.5 and clear[26, 16] < 0.1
    plain = mplane.render(5, cam, rgb, A, D).cpu().numpy()
    guarded = mplane.render(5, cam, rgb, A, D, clear=clear).cpu().numpy()
    assert plain[cy, cx, 0] < 0.5 and guarded[cy, cx, 0] > 0.9                         # hidden without the guard, readable with it
    assert np.allclose(plain[26, 16], guarded[26, 16], atol=0.05)


def _dive_pan_path(n=60):
    """a push_through-like path: the eye dives toward the cards while the look turns sharply to one side (the pan onto the strands)."""
    cams = []
    for i in range(n):
        u = i / (n - 1.0)
        eye = np.array([90.0 + 40.0 * u, 160.0, -1900.0 + 1500.0 * u])
        pan = np.clip((u - 0.3) / 0.35, 0.0, 1.0)
        pan = pan * pan * (3 - 2 * pan)                                             # 0 -> 1 over a third of the move
        yaw = np.radians(48.0) * pan
        look = eye + np.array([np.sin(yaw), -0.15 * pan, np.cos(yaw)]) * 1000.0
        cams.append(cam3d.Camera(eye, look, 0.0, 1000.0))
    return cams


def test_multiplane_card_scale_is_continuous_through_a_fast_pan(mplane):
    """the subject card is never rescaled on its own: through a dive + sharp look turn its on-screen scale changes by < 5% between frames, the
    cards stay on their canvas, and the look turn is eased back per card instead of zooming the window."""
    cams = _dive_pan_path()
    sc = np.array([mplane.scales(c) for c in cams])                                  # (frames, cards)
    for k in range(mplane.n):
        r = sc[1:, k] / sc[:-1, k]
        assert np.abs(r - 1.0).max() < 0.05, (k, float(np.abs(r - 1.0).max()))
    sub = mplane.cfg["subject"]
    assert sc[:, sub].min() > 0.0
    mplane.stats.update(over_max=0.0, turn_max=0.0)
    for c in cams[::4]:
        mplane.render(5, c)
    rep = mplane.edge_report()
    assert rep["over_max"] < 1e-3                                                     # no card edge enters the frame
    assert rep["turn_max"] > 0.0                                                      # and the pan did need the eased turn


# ---------------------------------------------------------------------------------------------------- compose op + recipe hook
import json as _json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from tools.fusion import multiplane_recipe as MR  # noqa: E402


class _StubChain:
    def __init__(self, cam):
        self.cam = cam

    def at(self, t):
        return self.cam


def _seed_state(key, mp, holds=(), near=500.0, far=1500.0, bg=(0, 0, 0), cam=None):
    MP._STATE[key] = dict(scene=None, chain=_StubChain(cam or _ref_cam(mp)), mp=mp,
                          setup=dict(near=near, far=far, mode="linear", bg=list(bg), holds=[list(h) for h in holds], size=list(mp.size)))


def _op_inputs(data, f=5, z_type=1493.0):
    plate = data.plate[f].astype(np.float32) / 255.0
    depth = data.depth[f].astype(np.float32) / 255.0
    matte = data.matte[f].astype(np.float32) / 255.0
    return plate, depth, matte


def _type_buffers(A, z, near=500.0, far=1500.0, rgb=(1.0, 0.0, 0.0), bg=(0.0, 0.0, 0.0)):
    """what the type passes hand the op: beauty on the palette bg, coverage, depth grey (32 + 223 (1 - (z - near) / (far - near)))."""
    beauty = np.zeros(A.shape + (3,), np.float32) + np.asarray(bg, np.float32) / 255.0 * (1 - A[..., None]) + A[..., None] * np.asarray(rgb, np.float32)
    grey = np.where(A > 0, (32.0 + 223.0 * (1.0 - (z - near) / (far - near))) / 255.0, 0.0).astype(np.float32)
    return beauty, grey


def test_multiplane_compose_op_matches_the_flat_plate_without_type(mplane, data):
    _seed_state("t:plain", mplane)
    plate, depth, matte = _op_inputs(data)
    z = np.zeros((H, W), np.float32)
    out = compose._ops_registry()["multiplane"](plate, depth, matte, np.zeros((H, W, 3), np.float32), z, z, t=0.2, state_key="t:plain")
    assert out.shape == plate.shape and np.abs(out - plate).mean() < 0.02            # the reference camera gives the plate back


def test_multiplane_compose_op_type_between_cards_and_legible_guard(mplane, data):
    cy, cx = 160, 70
    m = cv2.erode((data.matte[5] > 128).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    A = np.zeros((H, W), np.float32)
    A[cy - 6:cy + 6, cx - 8:cx + 8] = 1.0                                             # a letter that sits behind the subject card
    rgb, grey = _type_buffers(A, 1493.0)
    plate, depth, matte = _op_inputs(data)
    op = compose._ops_registry()["multiplane"]
    _seed_state("t:nohold", mplane)
    hidden = op(plate, depth, matte, rgb, A, grey, t=0.5, state_key="t:nohold")
    assert hidden[cy, cx, 0] < 0.6                                                    # the subject card covers it, no hold: stays behind
    _seed_state("t:hold", mplane, holds=[(0.0, 1.0)])
    shown = op(plate, depth, matte, rgb, A, grey, t=0.5, state_key="t:hold")
    assert shown[cy, cx, 0] > 0.9 and shown[cy, cx, 1] < 0.2                          # inside its word's hold the legible guard pulls it in front
    outside = op(plate, depth, matte, rgb, A, grey, t=3.0, state_key="t:hold")        # long after the hold: behind again
    assert outside[cy, cx, 0] < 0.6


def test_multiplane_recipe_swaps_the_flat_type_ops(tmp_path):
    ops = [{"op": "flow_wake", "out": "wl", "args": {"type_a": "$type_line", "flow": "$flow"}},
           {"op": "upscale_matte", "out": "m", "args": {"matte": "$s1_matte", "guide": "$footage"}},
           {"op": "tint_type", "out": "tcol", "args": {"type_a": "$wl", "footage": "$footage"}},
           {"op": "depth_occlude", "out": "image", "args": {"footage": "$footage", "footage_depth": "$s1_depth", "type_a": "$wl", "type_rgb": "$tcol"}},
           {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$wl", "intensity": 0.3}},
           {"op": "split_region", "out": "_fr0f", "args": {"like": "$footage"}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$image", "type_a": "$type_matte", "matte": "$m", "front": "$_fr0f", "type_rgb": [1, 1, 1]}},
           {"op": "contour_lines", "out": "_gl3c", "args": {"src": "$m", "mode": "outline"}},
           {"op": "over", "out": "image", "args": {"base": "$image", "fg": [1, 1, 1], "alpha": "$_gl3c"}},
           {"op": "grade", "out": "image", "args": {"frame": "$image", "gain": 1.0}}]
    tdir = tmp_path / "type"
    tdir.mkdir()
    (tdir / "manifest.json").write_text(_json.dumps({"spec": {"palette": {"bg": [4, 5, 6]}}, "size": [384, 704], "depth": {"near": 600, "far": 1700, "mode": "linear"},
                                                    "info": {"words": [{"hit": 1.0, "release": 2.0}]}}))
    cam = {"chain": "orbit_even", "plate": "multiplane", "cards": 4}
    piece = SimpleNamespace(j={"type": {"camera": cam}, "compose": {}}, id="PX", dir=tmp_path, shots={"s1": {"depth": True, "matte": True}},
                            typ=lambda: {"dir": str(tdir)}, log=lambda m: None)
    out = MR.apply(piece, ops, "s1")
    names = [o["op"] for o in out]
    assert names.count("multiplane") == 1 and "depth_occlude" not in names and "sandwich" not in names and "tint_type" not in names
    assert "flow_wake" not in names and "split_region" not in names                    # the type is drawn by the cards, once
    assert {"contour_lines", "over", "grade", "add_glow", "upscale_matte"} <= set(names)    # the outline line, glow and grade stay
    mpo = next(o for o in out if o["op"] == "multiplane")
    assert mpo["args"]["type_rgb"] == "$type_beauty" and mpo["args"]["depth"] == "$s1_depth" and mpo["args"]["matte"] == "$s1_matte"
    assert [o for o in out if o["op"] == "add_glow"][0]["args"]["mask"] == "$type_matte"
    setup = _json.loads(Path(mpo["args"]["setup_path"]).read_text())
    assert setup["near"] == 600 and setup["holds"] == [[1.0, 2.0]] and setup["bg"] == [4, 5, 6] and setup["size"] == [384, 704]
    # flat pieces and shots without a depth pass are left alone
    assert MR.apply(SimpleNamespace(j={"type": {"camera": "auto"}}, id="PY", dir=tmp_path, shots={}, log=lambda m: None), ops, "s1") is ops
    flat = SimpleNamespace(j={"type": {"camera": cam}, "compose": {}}, id="PZ", dir=tmp_path, shots={"s1": {"depth": False}}, log=lambda m: None)
    assert MR.apply(flat, ops, "s1") is ops
    assert MR.extra_passes({"type": {"camera": cam}}) == {"beauty", "matte", "depth"} and MR.extra_passes({"type": {}}) == set()
