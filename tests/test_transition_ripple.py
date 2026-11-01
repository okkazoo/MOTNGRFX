"""ripple transition (mograph/modules/transitions/ripple.py): registration, hand-off contract, duration range, seeded slow glitch,
a real two-word chain. Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_transition_ripple.py -q"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mograph.modules import Placement, Timing, Sequence, core  # noqa: E402
from mograph.modules.core import TransitionModule, get_module  # noqa: E402
from mograph.modules.words.lanes import Lanes  # noqa: E402
from mograph.render.cam3d import Camera  # noqa: E402


def make(dur=2.4, **kw):
    a, b = TransitionModule.demo_bundles()
    return get_module("ripple")(a, b, dur=dur, **kw)


def pts(layers):
    return [np.round(p, 4).tolist() for L in layers for p in L.polys]


def test_registered_and_suits():
    assert "ripple" in core.list_modules("transition")
    cls = get_module("ripple")
    assert cls.SUITS["brief"] and cls.description and cls.DEMO_DUR > 0


def test_hand_off_contract():
    tr = make()
    a, b = tr.a_exit, tr.b_entry
    first = pts(tr.layers(0.0))
    want = pts(core.strand_layers([a.P[k] for k in range(a.N)], list(a.col), list(a.w), a.tube, a.rim, hi_off=np.array([-0.8, -0.8, 0.0])))
    assert first == want
    last = pts(tr.layers(tr.dur))
    wantb = pts(core.strand_layers([b.P[k] for k in range(b.N)], list(b.col), list(b.w), b.tube, b.rim, hi_off=np.array([-0.8, -0.8, 0.0])))
    assert last == wantb


def test_every_duration_runs_and_alphas_are_valid():
    for dur in (1.2, 1.8, 2.4, 3.2):
        tr = make(dur)
        assert tr.duration == dur
        for u in np.linspace(0, 1, 13):
            for L in tr.layers(u * dur):
                assert 0.0 <= L.alpha <= 1.0 and np.isfinite(L.pts).all()
        assert len(tr.layers(0.45 * dur)) > 0           # something is on screen mid-way
        assert tr.layers(0.9 * dur) is not None


def test_everything_left_before_b_settles():
    """the last letter and its last echo ring are gone (q = 1) by 95 % of the duration."""
    for rings in (2, 3):
        tr = make(2.4, rings=rings)
        assert max(tr.delay) + tr.span + 0.06 * rings <= 0.95
        assert all(tr._qs(0.95, g, lag=0.06 * rings) >= 1.0 for g in range(len(tr.groups)))


def test_glitch_is_slow_seeded_and_deterministic():
    for dur in (1.2, 2.4, 3.2):
        a, b = make(dur, seed=5), make(dur, seed=5)
        assert a.events == b.events
        assert len(a.events) <= max(1, round(dur / 1.1)) + 1       # low frequency
        for (t0, t1, bands), nxt in zip(a.events, a.events[1:] + [None]):
            assert 0 < t0 < t1 < dur and 0.15 <= t1 - t0 <= 0.31   # held for several frames
            assert nxt is None or nxt[0] >= t1 + 0.29               # not flickery: gaps between events
    assert make(2.4, seed=5).events != make(2.4, seed=6).events


def test_letters_expand_staggered():
    tr = make(2.4, rings=2)
    assert len(tr.groups) >= 2
    assert tr.delay == sorted(tr.delay) or tr.delay[0] < tr.delay[-1]
    s0 = [tr._scale(tr._qs(0.3, gi), tr.S[gi]) for gi in range(len(tr.groups))]
    assert s0[0] > s0[-1] >= 1.0                         # the first letter leads


def test_frames_render_and_differ():
    tr = make(2.0)
    seq = Sequence([tr], bare=True)
    cam = Camera.home()
    f0, f1, f2 = (seq.frame(t, cam, S=1) for t in (0.0, 0.8, 1.4))
    assert f0.shape == (540, 960, 3)
    assert not np.array_equal(f0, f1) and not np.array_equal(f1, f2)


def test_in_a_real_two_word_sequence():
    tm = Timing(0.8, 0.3, 0.5)
    w0 = Lanes("ONE", placement=Placement.at((160, 200, 0), w=380, h=150), timing=tm, n_strands=18)
    w1 = Lanes("TWO", placement=Placement.at((1300, 300, 0), w=380, h=150), timing=tm, n_strands=18)
    seq = Sequence([w0, get_module("ripple")(), w1])
    tr = [it for it in seq.items if it.kind == "transition"][0]
    assert tr.name == "ripple" and tr.duration > 0
    for u in (0.0, 0.3, 0.6, 1.0):
        assert tr.layers(u * tr.duration) is not None
