"""Transition engine (mograph/modules/engine.py): rules on synthetic gaps, variety, determinism, analyse on real modules,
Sequence.auto. No rendering. Run: python -m pytest tests/test_modules_engine.py -q"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mograph.modules import Placement, Timing, Sequence, engine, core  # noqa: E402
from mograph.modules.engine import GapInfo, analyse, choose, rank  # noqa: E402
from mograph.modules.words.lanes import Lanes  # noqa: E402
from mograph.modules.transitions.twist import Twist  # noqa: E402


def top(gap, energy=None):
    return rank(gap, energy)[0][0]


def gap(**kw):
    return GapInfo(**kw)


# ------------------------------------------------------------------ rules (each triggers where expected)
def test_none_side_forces_cut():
    for kinds in (("none", "strands"), ("strands", "none"), ("none", "none")):
        g = gap(a_kind=kinds[0], b_kind=kinds[1], dist=700)
        assert [r[0] for r in rank(g)] == ["cut"]
        ch = choose(g, [], seed=1)
        assert ch.name == "cut" and ch.forced and ch.tcls is core.get_module("cut")


def test_big_heading_change_is_spiral():
    g = gap(dist=500, heading=150, turn_out=70, turn_in=80, plane_angle=40)
    assert top(g) == "spiral_around"


def test_passing_a_word_is_spiral():
    g = gap(dist=700, heading=5, obstacle=0.0, plane_angle=3)
    assert top(g) == "spiral_around"
    assert choose(g, [], 0, names=["spiral_around"]).params["around"] == "word"


def test_long_calm_same_plane_is_braid():
    g = gap(dist=1000, heading=4, turn_out=3, turn_in=3, plane_angle=3, energy=0.2)
    assert top(g) == "braid"
    assert top(gap(dist=1000, heading=4, plane_angle=3, energy=0.95)) != "braid"      # not when it is energetic
    assert top(gap(dist=1000, heading=4, plane_angle=70, energy=0.2)) != "braid"      # not across a fold


def test_orthogonal_same_plane_is_circuit():
    g = gap(dist=450, heading=90, turn_out=45, turn_in=45, plane_angle=5, ortho=1.0, energy=0.6, mood="tense")
    assert top(g) == "circuit_board"
    folded = gap(dist=450, heading=90, turn_out=45, turn_in=45, plane_angle=60, ortho=1.0, energy=0.6)
    assert top(folded) == "circuit_board"
    assert top(gap(dist=450, heading=40, plane_angle=5, ortho=0.0, energy=0.6)) != "circuit_board"   # off the lattice


def test_wider_target_is_fan():
    g = gap(dist=500, heading=20, turn_out=10, turn_in=10, plane_angle=5, size_ratio=2.5)
    assert top(g) == "fan"
    assert top(gap(dist=500, heading=20, plane_angle=5, size_ratio=1.0)) != "fan"


def test_generic_medium_gap_is_twist():
    g = gap(dist=450, heading=60, turn_out=30, turn_in=30, plane_angle=30, ortho=0.2, size_ratio=1.0)
    assert top(g) == "twist"


def test_tiny_gap_at_high_energy_can_cut():
    assert top(gap(dist=30, heading=10, energy=1.0)) == "cut"
    assert top(gap(dist=30, heading=10, energy=0.3)) != "cut"


def test_params_follow_the_gap():
    near = choose(gap(dist=250, plane_angle=30, heading=60), [], 0, names=["twist"])
    far = choose(gap(dist=1500, plane_angle=30, heading=60), [], 0, names=["twist"])
    assert far.params["turns"] > near.params["turns"]
    lo = choose(gap(a_strands=18, b_strands=18), [], 0, names=["circuit_board"])
    hi = choose(gap(a_strands=45, b_strands=45), [], 0, names=["circuit_board"])
    assert lo.params["lanes"] < hi.params["lanes"]
    small = choose(gap(b_size=(200, 80), heading=100, plane_angle=30), [], 0, names=["spiral_around"])
    big = choose(gap(b_size=(800, 340), heading=100, plane_angle=30), [], 0, names=["spiral_around"])
    assert small.params["radius"] < big.params["radius"]
    f1 = choose(gap(size_ratio=1.3), [], 0, names=["fan"]).params["angle"]
    f2 = choose(gap(size_ratio=3.0), [], 0, names=["fan"]).params["angle"]
    assert f2 > f1
    short = choose(gap(dist=1000, avail=2.4), [], 0, names=["braid"])
    long_ = choose(gap(dist=1000, avail=5.0, dur_auto=3.2), [], 0, names=["braid"])
    assert short.params["cycles"] < long_.params["cycles"]


def test_handedness_alternates():
    hist, hands = [], []
    for i in range(5):
        ch = choose(gap(index=i, dist=400, heading=60, plane_angle=30), hist, 4, names=["twist"])
        hist.append(ch)
        hands.append(ch.params["handed"])
    assert all(hands[i] == -hands[i + 1] for i in range(4)), hands


def test_duration_beat_and_avail():
    g = gap(dist=500, heading=60, plane_angle=30, dur_auto=2.0, beat=0.5)
    d = choose(g, [], 0, names=["twist"]).dur
    assert abs(d / 0.5 - round(d / 0.5)) < 1e-6
    capped = choose(gap(dist=500, heading=60, plane_angle=30, dur_auto=3.2, avail=1.4), [], 0, names=["twist"]).dur
    assert capped <= 1.4 + 1e-6
    fast = choose(gap(dist=500, heading=60, plane_angle=30, dur_auto=2.5, energy=1.0), [], 0, names=["twist"]).dur
    slow = choose(gap(dist=500, heading=60, plane_angle=30, dur_auto=2.5, energy=0.0), [], 0, names=["twist"]).dur
    assert fast < slow


def test_camera_hint_is_a_plain_dict():
    for n in ("twist", "circuit_board", "spiral_around", "braid", "fan", "cut"):
        ch = choose(gap(index=2, dist=800, heading=100, plane_angle=30, size_ratio=2.0, energy=0.3), [], 0, names=[n])
        h = ch.camera_hint
        assert type(h) is dict and {"move", "follow", "hold", "window"} <= set(h)
        assert h["window"] == ("trans", 2)


# ------------------------------------------------------------------ variety + determinism
CHAIN6 = [gap(index=0, dist=600, heading=10, plane_angle=5, ortho=0.3, size_ratio=1.0, energy=0.4),
          gap(index=1, dist=500, heading=100, turn_out=50, turn_in=50, plane_angle=40, ortho=0.9, size_ratio=1.0, energy=0.4),
          gap(index=2, dist=900, heading=15, plane_angle=5, ortho=0.2, size_ratio=1.0, energy=0.4),
          gap(index=3, dist=450, heading=30, plane_angle=10, size_ratio=2.2, energy=0.4),
          gap(index=4, dist=500, heading=140, turn_out=60, turn_in=90, plane_angle=20, energy=0.4),
          gap(index=5, dist=420, heading=60, plane_angle=35, ortho=0.4, energy=0.4)]


def run_chain(seed, gaps=CHAIN6):
    hist = []
    for g in gaps:
        hist.append(choose(g, hist, seed))
    return hist


def test_variety_across_a_six_word_chain():
    for seed in range(8):
        names = [c.name for c in run_chain(seed)]
        assert len(set(names)) >= 3, (seed, names)
        assert all(a != b for a, b in zip(names, names[1:])), (seed, names)


def test_five_gap_chain_with_five_names_never_repeats():
    """an unused transition is always preferred (hard), whatever the scores say; the soft penalty only applies once all are used."""
    pool = [n for n in core.list_modules("transition") if n != "cut"]
    assert len(pool) >= 5, pool
    pool = pool[:5]
    for seed in range(12):
        hist = []
        for g in CHAIN6[:5]:
            hist.append(choose(g, hist, seed, names=pool))
        names = [c.name for c in hist]
        assert sorted(names) == sorted(pool), (seed, names)
    hist = []                                            # a 6th gap with every name used falls back to the soft penalty
    for g in CHAIN6:
        hist.append(choose(g, hist, 3, names=pool))
    assert hist[-1].name in pool and hist[-1].name != hist[-2].name


def test_variety_penalty_on_identical_gaps():
    """six identical gaps: the engine must not repeat the previous transition."""
    base = dict(dist=450, heading=60, turn_out=30, turn_in=30, plane_angle=30, ortho=0.5, size_ratio=1.3, energy=0.5)
    for seed in range(8):
        hist = []
        for i in range(6):
            hist.append(choose(gap(index=i, **base), hist, seed))
        names = [c.name for c in hist]
        assert len(set(names)) >= 3 and all(a != b for a, b in zip(names, names[1:])), (seed, names)


def test_deterministic_per_seed_and_seed_changes_things():
    a = [(c.name, c.params, c.dur) for c in run_chain(5)]
    b = [(c.name, c.params, c.dur) for c in run_chain(5)]
    assert a == b
    seen = {tuple(c.name for c in run_chain(s)) for s in range(10)}
    assert len(seen) >= 3


# ------------------------------------------------------------------ real modules
def mods(tm=None):
    tm = tm or Timing(0.8, 0.3, 0.5)
    return [Lanes("ONE", placement=Placement.at((160, 200, 0), w=380, h=150), timing=tm, n_strands=18),
            Lanes("TWO", placement=Placement.at((1300, 300, 0), w=380, h=150), timing=tm, n_strands=18),
            Lanes("THREE", placement=Placement.at((1700, 420, 40), yaw=45, w=620, h=250), timing=tm, n_strands=18, entry="up")]


def test_analyse_real_modules():
    ws = mods()
    g = analyse(ws[0], ws[1], others=ws, index=0, energy=0.3, bpm=120)
    assert g.a_kind == g.b_kind == "strands" and g.a_strands == 18
    assert g.plane == "same" and g.heading < 1.0 and g.dist > 300
    assert abs(np.linalg.norm(np.array(g.b_gate) - np.array(g.a_gate)) - g.dist) < 1e-6
    assert g.beat == 0.5 and g.energy == 0.3 and g.size_ratio == 1.0
    g2 = analyse(ws[1], ws[2], others=ws, index=1)
    assert g2.plane == "fold" and 40 < g2.plane_angle < 50 and g2.heading > 40 and g2.size_ratio > 1.5
    d = g2.to_dict()
    assert d["a_kind"] == "strands" and isinstance(d["a_gate"], list)


def test_sequence_auto_builds_and_keeps_explicit():
    ws = mods()
    seq = Sequence.auto([ws[0], Twist(turns=3), ws[1], ws[2]], seed=2, energy=0.4, bpm=120)
    trs = [it for it in seq.items if it.kind == "transition"]
    assert len(trs) == 2 and trs[0].name == "twist" and trs[0].params["turns"] == 3
    assert len(seq.engine) == 1 and len(seq.gaps) == 1            # only the free gap was decided
    assert hasattr(trs[1], "camera_hint") and trs[1].name == seq.engine[0].name and trs[1].engine is seq.engine[0]
    n1 = [c.name for c in Sequence.auto(mods(), seed=2, energy=0.4).engine]
    n2 = [c.name for c in Sequence.auto(mods(), seed=2, energy=0.4).engine]
    assert n1 == n2 and len(n1) == 2


def test_flicker_side_becomes_cut_in_a_chain():
    tm = Timing(0.8, 0.3, 0.5)
    a = Lanes("ONE", placement=Placement.at((160, 200, 0), w=380, h=150), timing=tm, n_strands=18, form_out="flicker")
    b = Lanes("TWO", placement=Placement.at((900, 300, 0), w=380, h=150), timing=tm, n_strands=18)
    seq = Sequence.auto([a, b], seed=1)
    assert seq.engine[0].name == "cut" and [it.name for it in seq.items if it.kind == "transition"] == ["cut"]


def test_total_duration_budget():
    ws = mods()
    seq = Sequence.auto(ws, seed=1, energy=0.5, duration=9.0)
    budget = (9.0 - sum(w.duration for w in ws)) / 2
    tr = [it.duration for it in seq.items if it.kind == "transition"]
    assert all(d <= max(budget, 0.8) + 1e-6 for d in tr), (tr, budget)


def test_sequence_with_engine_choices_pickles():
    """Sequence.render pickles the sequence for the process pool; the Choice objects it carries must survive that."""
    import pickle
    seq = Sequence.auto(mods(), seed=2, energy=0.4)
    seq2 = pickle.loads(pickle.dumps(seq))
    assert [c.name for c in seq2.engine] == [c.name for c in seq.engine]
    assert seq2.engine[0][1] == seq.engine[0][1] and seq2.engine[0].gap.dist == seq.engine[0].gap.dist
