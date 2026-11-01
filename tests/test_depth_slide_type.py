"""v020: one eased move - A slides off left, straight lines travel left -> right, B slides in from the right and forms from them
(tools/fusion/depth_slide_type.py). CPU only. Run: python -m pytest tests/test_depth_slide_type.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.append(p)

from tools.fusion import depth_orbit as do  # noqa: E402
from tools.fusion import depth_slide_type as S  # noqa: E402

FLAG = "P06:SEARCHING,A LIGHT"
W, H = 216, 381                                                    # half-size video grid: shifts scale with W / 432


@pytest.fixture(scope="module")
def st():
    k = W / 432.0
    return S.SlideType(FLAG, 108, 24, 24, size=(576, 1024), xa=830.0 * k, xb=850.0 * k, sub=1, video_w=W, lines="lanes", centre=False, b_font=None, b_hf=None, b_k=1.0, dy_b=0.0, gate=(0.6, 1.0))      # g01: the lanes strands


@pytest.fixture(scope="module")
def comp(st):
    frames = [np.zeros((H, W, 3), np.uint8)] * 108
    return S.SlideComposer(st, frames, [np.zeros((H, W), np.float32)] * 108)


def _cx(a):
    c = a.sum(0)
    return float((c * np.arange(len(c))).sum() / max(c.sum(), 1e-9))


def test_slide_progress_is_the_camera_ease_q(st, comp):
    qs = np.array([st.q(i) for i in range(24, 60)])
    ref = np.array([float(do.dd.smootherstep((st.t_of(i)) / 36.0)) for i in range(24, 60)])
    assert np.abs(qs - ref).max() < 1e-9                                           # the pivot's own quintic over the 36-frame travel
    ids = [i for i in range(25, 32)]                                               # while A is fully on screen
    cxa = np.array([_cx(comp.parts(i, 1)[0]) for i in ids])                        # A's measured screen x
    qa = np.array([st.q(i) for i in ids])
    assert np.corrcoef(cxa, qa)[0, 1] < -0.999                                     # x is linear in q (correlation -1: it moves left with the ease)
    slope = np.polyfit(qa, cxa, 1)[0]
    assert slope == pytest.approx(-st.xa, rel=0.02)
    ids_b = [i for i in range(51, 59)]
    cxb = np.array([_cx(comp.parts(i, 1)[1]) for i in ids_b])
    qb = np.array([st.q(i) for i in ids_b])
    assert np.corrcoef(cxb, qb)[0, 1] < -0.999                                     # B's x is linear in (1 - q): it slides in with the same ease


def test_a_leaves_left_b_arrives_from_the_right_and_rests(st, comp):
    xs = [_cx(comp.parts(i, 1)[0]) for i in range(24, 40)]
    assert all(b < a for a, b in zip(xs, xs[1:]))                                  # A's screen x decreases during the exit
    assert comp.parts(46, 1)[0].sum() < 0.02 * comp.parts(0, 1)[0].sum()           # and A is off the screen by q ~ 0.6
    xb = [_cx(comp.parts(i, 1)[1]) for i in range(48, 60)]
    assert all(b < a for a, b in zip(xb, xb[1:])) and xb[0] > xb[-1] + 0.1 * W     # B's x decreases into place from the right
    hold = _cx(comp._placed("B", st._hold(st.wb), 1))
    assert xb[-1] == pytest.approx(hold, abs=1.0) and _cx(comp.parts(100, 1)[1]) == pytest.approx(hold, abs=0.5)    # at rest exactly its hold position


def test_lines_move_left_to_right_as_distinct_segments_shorter_than_the_frame(st, comp):
    lc = [_cx(comp.parts(i, 1)[2]) for i in range(34, 48, 2)]
    assert all(b > a for a, b in zip(lc, lc[1:])) or lc[-1] > lc[0] + 0.1 * W      # the lines' mass moves +x
    L = comp.parts(38, 1)[2]
    rows = np.nonzero(L.max(1) > 0.3)[0]
    assert len(rows) > 0
    groups = 1 + int((np.diff(rows) > 3).sum())
    assert groups >= 8                                                             # a moderate number of distinct rows
    longest = 0
    for r in rows:
        on = (L[r] > 0.3).astype(int)
        d = np.diff(np.r_[0, on, 0])
        runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
        longest = max(longest, int(runs.max()) if len(runs) else 0)
    assert 0 < longest < 0.6 * W                                                   # segments (both ends visible), not full-width streaks
    assert st.n_strands_shown < 60 and st.n_strands_all > st.n_strands_shown       # the module's strands, thinned to a readable count


def test_type_or_lines_on_screen_every_frame_from_the_exit_to_the_hold(st, comp):
    tot = [sum(float(x.mean()) for x in comp.parts(i, 1)) for i in range(24, 78)]
    assert min(tot) > 0.004, [24 + i for i, t in enumerate(tot) if t <= 0.004]


def test_b_forms_after_the_landing(st, comp):
    qs = [st.q(i) for i in range(24, 80)]
    g = [st.gate_of(q) for q in qs]
    assert max(g[:30]) < 0.999 and st.i_land == 59                                 # still forming before the landing
    assert st.gate_of(st.q(st.i_land)) == pytest.approx(1.0) and st.gate_of(st.q(70)) == 1.0
    B = comp.parts(100, 1)[1]
    assert comp.parts(40, 1)[1].sum() < 0.02 * B.sum()                              # not there at the turn's middle
    la, lb = comp.parts(64, 1)[2].sum(), comp.parts(100, 1)[2].sum()
    assert abs(la - lb) < 0.15 * max(la, lb)                                       # the lines have finished drawing the outline by ~landing + 5
    assert comp.parts(100, 1)[0].sum() == pytest.approx(comp.parts(0, 1)[0].sum(), rel=1.0)


def test_runner_matte_is_in_front_of_the_type(st):
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(108)]
    probe = S.SlideComposer(st, frames, [np.zeros((H, W), np.float32)] * 108)
    for i in (10, 100):
        _, a = probe.type_layer(i)
        cols = np.nonzero(a.sum(0) > 0.5)[0]
        mid = int((cols.min() + cols.max()) / 2)
        matte = np.zeros((H, W), np.float32)
        matte[:, :mid] = 1.0
        c = S.SlideComposer(st, frames, [matte] * 108)
        out, foot = c.frame(i).astype(int), frames[i].astype(int)
        _, a2 = c.type_layer(i)
        assert np.abs(out - foot)[matte > 0.99].max() <= 1
        assert np.abs(out - foot)[(matte < 0.01) & (a2 > 0.5)].max() > 40


# ----------------------------------------------------------------------------------------------- g02: the words' own scanline rows are the lines
import cv2  # noqa: E402


@pytest.fixture(scope="module")
def sc():
    k = W / 432.0
    st2 = S.SlideType(FLAG, 108, 24, 24, size=(576, 1024), xa=830.0 * k, xb=850.0 * k, sub=1, video_w=W, lines="scan", centre=False, b_font=None, b_hf=None, b_k=1.0, dy_b=0.0, gate=(0.6, 1.0))
    frames = [np.zeros((H, W, 3), np.uint8)] * 108
    return S.ScanComposer(st2, frames, [np.zeros((H, W), np.float32)] * 108, rows=36, seg=0.34, thick=5.0 * k)


def test_scan_lines_are_straight_horizontal_segments_shorter_than_the_frame(sc):
    L = sc.parts(38, 1)[2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats((L > 0.5).astype(np.uint8))
    comps = [stats[j] for j in range(1, n) if stats[j][4] > 12]
    assert len(comps) >= 15                                                      # a moderate number of distinct lines
    for x, y, w, h, area in comps:
        assert h <= sc.thick + 2 and w >= 4 * h                                  # angle 0: a bar, not a streak / hook / ring
        assert w < 0.7 * W                                                       # shorter than the frame: both ends visible (or entering)
    assert (np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1)) ** 2).sum() > 5 * (np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0)) ** 2).sum()


def test_scan_lines_move_plus_x_on_the_camera_ease(sc):
    st = sc.rt
    rows = [r for r in range(sc.rows) if sc.A.ext[r] is not None]
    for r in rows[::5]:
        hs = [sc.a_head_tail(r, st.q(i))[0] for i in range(25, 60)]
        assert all(b >= a - 1e-9 for a, b in zip(hs, hs[1:])) and hs[-1] > W   # A's segments run right and leave the frame
        qs = [st.q(i) for i in range(25, 50)]
        assert np.corrcoef([sc.a_head_tail(r, q)[0] for q in qs], qs)[0, 1] > 0.99     # on the camera ease q
    for r in [r for r in range(sc.rows) if sc.B.ext[r] is not None][::4]:
        s0, s1 = sc.b_row(r)
        hs = [sc.b_head_tail(r, float(i), st.q(i))[0] for i in range(int(s0), int(s1) + 1)]
        pk = int(np.argmax(hs))
        assert all(b > a for a, b in zip(hs[:pk + 1], hs[1:pk + 1])) and pk >= 0.6 * (len(hs) - 1), r      # B's segments move +x until they reach the row end ...
        assert hs[-1] > hs[pk] - 0.25 * (hs[pk] - hs[0])                         # ... which itself comes in from the right on q, so the head settles back a little onto it
    c = [_cx(sc.parts(i, 1)[2]) for i in (30, 34, 38)]
    assert c[-1] > c[0]


def test_each_landed_run_matches_the_b_mask_row_span_and_b_shows_only_where_landed(sc):
    st = sc.rt
    mask = sc.b_mask
    for r in range(sc.rows):
        if sc.B.ext[r] is None:
            continue
        lo, hi = int(np.floor(sc.ys[r] - 0.5 * sc.pitch)), int(np.ceil(sc.ys[r] + 0.5 * sc.pitch))
        cols = np.nonzero((mask[max(lo, 0):hi + 1] > 0.3).any(0))[0]
        assert sc.B.ext[r] == (float(cols.min()), float(cols.max()))             # the row's x-extent IS the mask's
        s0, s1 = sc.b_row(r)
        q = st.q(int(s1))
        h, t, e, land = sc.b_head_tail(r, s1, q)
        assert e == pytest.approx(1.0)
        assert h == pytest.approx(sc.B.ext[r][1] + (1 - q) * st.xb) and t == pytest.approx(sc.B.ext[r][0] + (1 - q) * st.xb)   # lands exactly on the run extent (on the sliding end)
    i = 52
    B = sc.parts(i, 1)[1]
    landed = [r for r in range(sc.rows) if sc.B.ext[r] is not None and sc.b_row(r)[1] <= i]
    pending = [r for r in range(sc.rows) if sc.B.ext[r] is not None and sc.b_row(r)[1] > i + 1]
    assert landed and pending
    for r in pending:
        assert B[int(sc.ys[r]) - 1:int(sc.ys[r]) + 2].sum() == 0.0               # B has nothing on a row whose segment has not landed
    assert any(B[int(sc.ys[r])].sum() > 0 for r in landed)                       # and has pixels on the landed ones
    assert sc.parts(40, 1)[1].sum() == 0.0                                       # nothing of B before the first run lands


def test_b_forms_from_the_rows_complete_just_after_the_landing(sc):
    st = sc.rt
    full = sc.b_mask
    assert np.abs(sc.parts(72, 1)[1] - full).max() < 0.02                        # the solid glyph is complete (= the hold matte) and at rest
    assert sc.parts(46, 1)[1].sum() < 0.15 * full.sum()                          # still forming at q = 0.75
    last = max(sc.b_row(r)[1] for r in range(sc.rows) if sc.B.ext[r] is not None) + sc.rev_dur
    assert st.i_land <= last <= st.i_land + 8                                    # complete at / just after the landing


def test_stream_is_continuous_from_a_exit_into_b_entry_no_sparse_stretch(sc):
    tot = [sum(float(x.mean()) for x in sc.parts(i, 1)) for i in range(24, 78)]
    assert min(tot) > 0.03, [24 + i for i, t in enumerate(tot) if t <= 0.03]
    ln = [float(sc.parts(i, 1)[2].mean()) for i in range(26, 58)]
    assert min(ln) > 0.004                                                       # lines on screen on every frame of the turn (A's hand over to B's)


def test_scan_runner_matte_is_in_front(sc):
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(108)]
    probe = S.ScanComposer(sc.rt, frames, [np.zeros((H, W), np.float32)] * 108, thick=2.5)
    for i in (10, 44, 100):
        _, a = probe.type_layer(i)
        cols = np.nonzero(a.sum(0) > 0.5)[0]
        mid = int((cols.min() + cols.max()) / 2)
        matte = np.zeros((H, W), np.float32)
        matte[:, :mid] = 1.0
        c = S.ScanComposer(sc.rt, frames, [matte] * 108, thick=2.5)
        out, foot = c.frame(i).astype(int), frames[i].astype(int)
        _, a2 = c.type_layer(i)
        assert np.abs(out - foot)[matte > 0.99].max() <= 1
        assert np.abs(out - foot)[(matte < 0.01) & (a2 > 0.5)].max() > 40


# ----------------------------------------------------------------------------------------------- g03/g04: the circuit_board library transition, centred words, Tourney
@pytest.fixture(scope="module")
def cb():
    k = W / 432.0
    st3 = S.SlideType(FLAG, 108, 24, 24, size=(576, 1024), xa=830.0 * k, xb=850.0 * k, sub=1, video_w=W, lines="circuit", centre=True,
                      b_font="Tourney[wdth,wght].ttf", b_axes={"wght": 400, "wdth": 100}, b_hf=0.72, b_k=0.85, dy_b=-0.18, gate=(0.5, 1.0))
    frames = [np.zeros((H, W, 3), np.uint8)] * 108
    return S.ModuleComposer(st3, frames, [np.zeros((H, W), np.float32)] * 108)


def _tr(cb):
    return next(it for it, _a, _b in cb.rt.cs.sc.seq.timeline if getattr(it, "name", "") == "circuit_board")


def _segments(P):
    d = np.diff(P[:, :2], axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    return ang[L > 1e-6], L[L > 1e-6]


def test_circuit_board_is_a_library_module_with_a_capped_strand_count(cb):
    tr = _tr(cb)
    assert tr.name == "circuit_board" and cb.rt.cs.tr["module"] == "circuit_board"
    assert 12 <= tr.K <= 30                                                        # restrained: the module carries <= 30 strands (a dotted face has hundreds)
    assert cb.rt.lines_mode == "circuit"


def test_traces_are_orthogonal_with_45_degree_corners_and_travel_net_plus_x(cb):
    tr = _tr(cb)
    lens, orth = 0.0, 0.0
    net = []
    for P in tr.pts:
        P = np.asarray(P, float)
        ang, L = _segments(P)
        d = np.minimum.reduce([np.abs(ang - a) for a in (0.0, 45.0, 90.0, 135.0, 180.0)])
        lens += L.sum()
        orth += L[d < 3.0].sum()
        net.append(P[-1, 0] - P[0, 0])
    assert orth / lens > 0.65                                                      # circuit look: most of the route is runs at 0 / 90 and 45-degree corners (the rest = the lane fan at the two gates)
    a0 = []
    for P in tr.pts:
        ang, L = _segments(np.asarray(P, float))
        m = (np.minimum(np.abs(ang - 0.0), np.abs(ang - 180.0)) < 3.0) | (np.abs(ang - 90.0) < 3.0)
        a0.append(L[m].sum() / L.sum())
    assert np.median(a0) > 0.5                                                     # mostly orthogonal runs (the rest are the corner cuts)
    assert np.mean(net) > 0.4 * 540.0                                              # net travel LEFT -> RIGHT across the frame (world px, 540 wide)
    assert min(net) > 0.0


def test_traces_criss_cross(cb):
    tr = _tr(cb)
    segs = []
    for k, P in enumerate(tr.pts):
        P = np.asarray(P, float)[:, :2]
        for a, b in zip(P[:-1], P[1:]):
            segs.append((k, a, b))

    def cross(p, q, r, s):
        def o(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return o(p, q, r) * o(p, q, s) < 0 and o(r, s, p) * o(r, s, q) < 0
    pairs = set()
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if segs[i][0] != segs[j][0] and cross(segs[i][1], segs[i][2], segs[j][1], segs[j][2]):
                pairs.add((segs[i][0], segs[j][0]))
    assert len(pairs) >= 5                                                         # some pairs of traces cross over each other


def test_words_are_centred_and_b_is_tourney(cb):
    st = cb.rt
    for wt in (st.wa, st.wb):
        a = S.DT.place_matte(st._hold(wt), W, H, 1)
        xs = np.nonzero(a.max(0) > 0.25)[0]
        assert abs(0.5 * (xs.min() + xs.max()) - W / 2.0) < 0.03 * W               # bbox centre within 3 % of the frame centre
    assert st.words[0]["font"] == "doto" and st.words[1]["font"] == "tourney"


def test_coverage_ceiling_and_continuity_with_the_circuit_lines(cb):
    tot, ln = [], []
    for i in range(24, 78):
        A, B, L = cb.parts(i, 1)
        tot.append(float(np.clip(np.maximum(np.maximum(A, B), L), 0, 1).mean()))
        ln.append(float(L.mean()))
    assert max(tot) < 0.15 and max(ln) < 0.15                                       # never a sheet: the type / strands cover < 15 % of the frame
    assert min(tot) > 0.01, [24 + i for i, t in enumerate(tot) if t <= 0.01]       # type or strands on screen on every frame 24-77


def test_circuit_lines_move_plus_x_on_the_camera_ease(cb):
    st = cb.rt
    c = [_cx(cb.parts(i, 1)[2]) for i in (30, 33, 36, 39)]
    assert c[-1] > c[0]
    taus = [st.cs.tau_of_q(st.q(i)) for i in range(24, 60)]
    assert all(b >= a - 1e-9 for a, b in zip(taus, taus[1:]))                       # module time follows q monotonically
    qs = np.array([st.q(i) for i in range(24, 60)])
    cum = np.interp(taus, st.cs._clock[0], st.cs._clock[1])
    assert np.corrcoef(cum, qs)[0, 1] > 0.9999                                      # the module's (visible) progress is linear in q: it eases with the camera


def test_circuit_runner_matte_is_in_front(cb):
    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(108)]
    for i in (10, 38, 100):
        c0 = S.ModuleComposer(cb.rt, frames, [np.zeros((H, W), np.float32)] * 108)
        _, a = c0.type_layer(i)
        cols = np.nonzero(a.sum(0) > 0.5)[0]
        mid = int((cols.min() + cols.max()) / 2)
        matte = np.zeros((H, W), np.float32)
        matte[:, :mid] = 1.0
        c = S.ModuleComposer(cb.rt, frames, [matte] * 108)
        out, foot = c.frame(i).astype(int), frames[i].astype(int)
        _, a2 = c.type_layer(i)
        assert np.abs(out - foot)[matte > 0.99].max() <= 1
        assert np.abs(out - foot)[(matte < 0.01) & (a2 > 0.5)].max() > 40


def test_defaults_are_the_g04_pick():
    import inspect
    d = inspect.signature(S.SlideType.__init__).parameters
    assert (d["lines"].default, d["centre"].default, d["b_font"].default, d["b_hf"].default, d["b_k"].default, d["dy_b"].default,
            d["gate"].default, d["k_strands"].default) == ("circuit", True, "Tourney[wdth,wght].ttf", 0.72, 0.85, -0.18, (0.5, 1.0), 24)
