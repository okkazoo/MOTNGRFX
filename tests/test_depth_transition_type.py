"""v018: the P06 words in their own layout handed over by a library transition, behind the runner matte (tools/fusion/depth_transition_type.py).
CPU only, small type size. Run: python -m pytest tests/test_depth_transition_type.py -q"""
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

from tools.fusion import depth_transition_type as T  # noqa: E402

FLAG = "P06:SEARCHING,A LIGHT"
SIZE = (576, 1024)
FW, FH = 96, 160


@pytest.fixture(scope="module")
def tw():
    return T.TransitionType(FLAG, 108, "twist", size=SIZE, sub=1)


def _mass(m):
    return float(m.mean())


def _cx(m):
    c = m.sum(0)
    return float((c * np.arange(len(c))).sum() / max(c.sum(), 1e-9)) / m.shape[1]


def test_layout_is_the_piece_own_layout_like_the_reference(tw):
    assert tw.pa["lines"] == ["SEA", "RCH", "ING"] and tw.pb["lines"] == ["A", "LIG", "HT"]      # the piece's own lines
    assert tw.pa["font"] == "doto" and tw.pb["font"] == "geistpixel"
    m = tw.matte(10)
    xs, ys = np.nonzero(m.max(0) > 0.3)[0], np.nonzero(m.max(1) > 0.3)[0]
    w, h = (xs.max() - xs.min()) / m.shape[1], (ys.max() - ys.min()) / m.shape[0]
    assert 0.62 <= w <= 0.80 and h >= 0.5                                       # big: ~72 % of the width, most of the height (reference)
    assert 0.50 <= 0.5 * (xs.min() + xs.max()) / m.shape[1] <= 0.66              # right of centre ('right two-thirds')
    assert tw.sc.chain is None                                                   # no camera chain: static type camera (cam3d home), no beat_orbit


def test_a_is_formed_and_frozen_on_the_hold_b_formed_and_frozen_at_the_end(tw):
    a = [tw.matte(i) for i in (0, 12, 24)]
    assert _mass(a[0]) > 0.02 and np.abs(a[0] - a[2]).max() < 1e-6
    end = [tw.matte(i) for i in (tw.i_hit, 90, 107)]
    assert _mass(end[0]) > 0.02 and np.abs(end[0] - end[2]).max() < 1e-6          # B frozen from the frame it is formed (axes do not breathe)
    assert _cx(end[0]) < 0.5 < _cx(a[0])                                         # A right of centre, B left of centre (mirrored placement)


def test_hand_off_runs_left_to_right_a_leaves_right_b_arrives_from_the_left(tw):
    assert tw.pa and tw.sc.info["words"][0]["lines"]
    early = [_cx(tw.matte(i)) for i in (24, 26, 28)]                             # A's strands leave toward the right (its exit ends at frame ~29)
    assert early[-1] > early[0] > 0.5 and tw.matte(28).sum() > 0
    pre = tw.matte(tw.i_hit - 6)                                                 # B's strands come in on the left side
    assert pre.sum() > 0 and _cx(pre) < 0.5
    assert tw.tau(tw.i_hit) == pytest.approx(tw.pb["hit"]) and tw.tau(tw.i_hit + 20) == pytest.approx(tw.pb["hit"])
    assert tw.tau(10) == tw.pa["release"]


def test_timing_window_matches_the_depth_video(tw):
    # A's exit starts with the turn (frame 24), the transition runs through the turn + swing (landing = frame 59), B is formed 18 frames after the landing
    t0 = (tw.tr["start"] - tw.pa["release"]) * 24 + tw.i_start
    t1 = (tw.tr["start"] + tw.tr["dur"] - tw.pa["release"]) * 24 + tw.i_start
    assert 24 <= t0 <= 32 and t1 >= 59 and abs((tw.pb["hit"] - tw.pa["release"]) * 24 + tw.i_start - 77) <= 1.0


@pytest.mark.parametrize("name", T.TRANSITIONS)
def test_every_library_transition_builds_and_draws(name):
    t = T.TransitionType(FLAG, 108, name, size=SIZE, sub=1)
    assert t.tr["module"] == name
    assert _mass(t.matte(10)) > 0.02                                             # A formed
    assert _mass(t.matte(t.i_hit)) > 0.02                                        # B formed
    assert max(_mass(t.matte(i)) for i in range(26, 70, 6)) > 0.0                # something of the hand-off is drawn


def test_runner_matte_is_in_front_of_the_flat_type(tw):
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (FH, FW, 3), dtype=np.uint8) for _ in range(108)]
    zero = [np.zeros((FH, FW), np.float32)] * 108
    probe = T.TransitionComposer(tw, frames, zero)
    for i in (10, 100):
        _, a = probe.type_layer(i)
        cols = np.nonzero(a.sum(0) > 0.5)[0]
        mid = int((cols.min() + cols.max()) / 2)
        matte = np.zeros((FH, FW), np.float32)
        matte[:, :mid] = 1.0                                                     # a runner covering the left half of the word
        comp = T.TransitionComposer(tw, frames, [matte] * 108)
        out, foot = comp.frame(i).astype(int), frames[i].astype(int)
        _, a2 = comp.type_layer(i)
        assert np.abs(out - foot)[matte > 0.99].max() <= 1                       # where he is: the video; the type is behind him
        assert np.abs(out - foot)[(matte < 0.01) & (a2 > 0.5)].max() > 40       # elsewhere the type is drawn


def test_no_wave_no_squeeze_the_type_is_what_the_engine_drew(tw):
    frames = [np.zeros((FH, FW, 3), np.uint8)] * 108
    comp = T.TransitionComposer(tw, frames, [np.zeros((FH, FW), np.float32)] * 108)
    from tools.fusion import depth_type as DT
    for i in (10, 44):
        ref = DT.place_matte(tw.matte(i), FW, FH, 2)
        assert np.array_equal(comp.alpha(i), ref)


# ----------------------------------------------------------------------------------------------- v019: engine camera pinned to home on the holds, no empty frame
V019 = ("ripple", "fan", "twist")


@pytest.fixture(scope="module", params=V019)
def eng(request):
    t = T.TransitionType(FLAG, 108, request.param, size=SIZE, sub=1, camera="engine")
    t.fit_clock()
    return t


def test_no_empty_frame_between_the_turn_and_the_landed_word(eng):
    m = [_mass(eng.matte(i)) for i in range(24, 78)]
    assert min(m) > 5.0e-4, (eng.transition, [i + 24 for i, x in enumerate(m) if x <= 5.0e-4])      # type on screen on EVERY frame 24-77


def test_holds_are_framed_exactly_like_v018(eng):
    home = T.TransitionType(FLAG, 108, eng.transition, size=SIZE, sub=1, camera="home")
    for i in (0, 12, 24):                                                          # SEARCHING on image 1's hold
        assert np.abs(eng.matte(i) - home.matte(i)).max() < 1e-6
    for i in (eng.i_hit, 90, 107):                                                 # A LIGHT on its hold
        assert np.abs(eng.matte(i) - home.matte(i)).max() < 1e-6
    assert not np.array_equal(eng.matte(48), home.matte(48))                       # the camera does move inside the window


def test_camera_move_only_inside_the_window_and_eases_out_of_and_into_home(eng):
    c = eng.cam
    assert c.t1 > c.t0 and c.weight(c.t0 - 0.01) == 0.0 and c.weight(c.t1 + 0.01) == 0.0
    L = c.t1 - c.t0
    ws = [c.weight(c.t0 + u * L) for u in np.linspace(0.02, 0.98, 25)]
    assert ws[0] < 0.1 and ws[-1] < 0.1 and max(ws) == pytest.approx(1.0, abs=1e-6)       # eased in and out
    assert all(b >= a - 1e-9 for a, b in zip(ws[:8], ws[1:9])) and all(b <= a + 1e-9 for a, b in zip(ws[-9:], ws[-8:]))
    home = c.home
    for t in (0.2, 1.0, c.t0 - 0.05, c.t1 + 0.05, 5.0):
        k = c.at(t)
        assert np.allclose(k.eye, home.eye) and np.allclose(k.target, home.target) and k.F == home.F
    mid = c.at(c.t0 + 0.5 * L)
    assert not np.allclose(mid.eye, home.eye) or not np.allclose(mid.target, home.target)


def test_direction_stays_left_to_right_at_the_gates(eng):
    early = [_cx(eng.matte(i)) for i in (24, 26, 28)]
    assert early[-1] > early[0] > 0.5                                              # A's strands leave toward the right
    pre = eng.matte(eng.i_hit - 5)
    assert pre.sum() > 0 and _cx(pre) < 0.6                                        # B's arrive on the left side of the frame


def test_word_b_nudge_keeps_every_letter_at_least_60_percent_visible():
    rp = os.path.join(ROOT, "17_depth-dolly", "v016", "local", "matte_depth")
    if not os.path.isdir(rp):
        pytest.skip("v016 matte pass not on this disk")
    import cv2
    mattes = [cv2.imread(os.path.join(rp, "%04d.png" % i), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for i in range(108)]
    t = T.TransitionType(FLAG, 108, "twist", size=SIZE, sub=1, camera="engine")
    dx, dy, mv = T.fit_b(t, mattes)
    assert mv is not None and mv >= 0.6 and dx <= -0.02                            # left of centre, nobody > 40 % hidden on the whole hold
    t2 = T.TransitionType(FLAG, 108, "twist", size=SIZE, sub=1, camera="engine", b_dx=dx, b_dy=dy)
    from tools.fusion import depth_type as DT
    from tools.fusion import depth_ribbon_type as RB
    H, W = mattes[0].shape
    a = DT.place_matte(t2.matte(t2.i_hit), W, H, 1)
    for i in (77, 92, 107):
        assert min(v for *_, v in RB.letter_visibility(a, mattes[i])) >= 0.6
