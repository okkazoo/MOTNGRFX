"""GLYPH-OUTLINE HAND-OFF of the library transitions (mograph/modules/io/outline.py + mograph/modules/sequence.outline_handoff, 2026-11-01): in a library gap
the module's own strands start on word A's glyph outlines and end on word B's; the old gate hand-off stays reachable. Run: python -m pytest tests/test_outline_handoff.py -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mograph.modules.core import get_module, Placement, Timing  # noqa: E402
import mograph.modules.words  # noqa: E402,F401
import mograph.modules.transitions  # noqa: E402,F401
from mograph.modules import sequence as SQ  # noqa: E402

PATH_KINDS = ("braid", "circuit_board", "fan", "spiral_around", "twist")


def _seq(kind, mode="outline", **tp):
    prev, SQ.HANDOFF = SQ.HANDOFF, mode
    try:
        tm = Timing(1.2, 0.8, 0.9)
        A = get_module("lanes")(words="GLIDE", placement=Placement((260.0, 190.0, 0.0), (1, 0, 0), (0, 1, 0), 330, 150), timing=tm)
        B = get_module("lanes")(words="RIDE", placement=Placement((700.0, 350.0, 0.0), (1, 0, 0), (0, 1, 0), 300, 150), timing=tm)
        tr = get_module(kind)(dur=2.0, **tp)
        return SQ.Sequence([A, tr, B]), A, tr, B
    finally:
        SQ.HANDOFF = prev


def _dist_to_glyphs(word, P3):
    """distance (px) of world points to the word's glyph outlines (placement plane)."""
    pl = word.placement
    rel = P3 - pl.origin
    q = np.stack([rel @ pl.u, rel @ pl.v], 1)
    sh = word.hold_shape()
    segs = []
    for c in sh.cons:
        c2 = np.vstack([c, c[:1]])
        segs.append((c2[:-1], c2[1:]))
    best = np.full(len(q), np.inf)
    for a, b in segs:
        ab = b - a
        tt = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
        d = np.linalg.norm(q[:, None, :] - (a[None] + tt[..., None] * ab[None]), axis=-1).min(1)
        best = np.minimum(best, d)
    return best


@pytest.mark.parametrize("kind", PATH_KINDS)
def test_strands_start_on_a_and_end_on_b(kind):
    seq, A, tr, B = _seq(kind)
    assert type(A.form_out).__name__ == "Outline" and type(B.form_in).__name__ == "Outline"
    off_a = 0.06 * A.hold_shape().cap
    off_b = 0.06 * B.hold_shape().cap
    first = np.array([R.at(0.0) for R in tr.routes])                    # every strand's route: the first point is on A's outline (eased off by the hand-off offset) ...
    last = np.array([R.at(R.L) for R in tr.routes])                     # ... the last point on B's
    assert np.percentile(_dist_to_glyphs(A, first), 95) <= off_a + 1.5
    assert np.percentile(_dist_to_glyphs(B, last), 95) <= off_b + 1.5
    assert len(tr.layers(0.0)) and len(tr.layers(tr.duration))


def test_exit_and_entry_contracts():
    seq, A, tr, B = _seq("spiral_around")
    ex = A.exit_bundle()
    en = B.entry_bundle()
    assert np.allclose(ex.P, tr.a_exit.P) and np.allclose(en.P, tr.b_entry.P)
    assert len(A.layers(A.duration)) and len(B.layers(0.0))
    assert A.form_out.material().bundle.N == tr.a_exit.N


def test_old_gate_handoff_is_reachable():
    seq, A, tr, B = _seq("braid", mode="gate")
    assert A.form_out.is_lines and B.form_in.is_lines
    seq, A, tr, B = _seq("braid", handoff="gate")
    assert A.form_out.is_lines and B.form_in.is_lines
    seq, A, tr, B = _seq("cut")                                         # only the library kinds take the outline hand-off
    assert A.form_out.is_lines


def test_ripple_takes_the_outline_too():
    seq, A, tr, B = _seq("ripple")
    assert type(A.form_out).__name__ == "Outline"
    assert np.percentile(_dist_to_glyphs(A, tr.a_exit.P[:, 0]), 95) <= 0.06 * A.hold_shape().cap + 1.5


def _extent(layers):
    pts = [L.pts for L in layers if len(L.pts)]
    if not pts:
        return 0.0
    P = np.concatenate(pts)
    return float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))


@pytest.mark.parametrize("kind", PATH_KINDS + ("ripple",))
def test_never_a_small_word_mid_transition(kind):
    """user 2026-11-01: "it should just be from the outline of the text its transforming from not forming a smaller word inbetween": through A's exit
    window and B's form window the word-shaped cluster stays at >= 0.85 of its hold size (the strands unravel from / re-form into the FULL-SIZE outline)."""
    seq, A, tr, B = _seq(kind)
    hold_a = _extent(A.layers(A.timing.form + 0.5 * A.timing.hold))
    hold_b = _extent(B.layers(B.timing.form + 0.5 * B.timing.hold))
    ta0 = A.timing.form + A.timing.hold
    for q in np.linspace(0.0, 1.0, 9):
        assert _extent(A.layers(ta0 + q * A.timing.exit)) >= 0.85 * hold_a
        assert _extent(B.layers(q * B.timing.form)) >= 0.85 * hold_b


def test_flock_staggers_and_lands_on_time():
    seq, A, tr, B = _seq("braid")
    d = np.array([s.delay for s in tr.slides])
    e = np.array([s.delay + s.T for s in tr.slides])
    assert d.max() - d.min() > 0.05 * tr.dur and e.max() <= tr.dur + 1e-6        # some lead, some lag; all have arrived by the end (B on its beat)
    seq2, A2, tr2, B2 = _seq("braid")
    assert np.allclose(d, [s.delay for s in tr2.slides])                            # deterministic


def test_staircase_exit_has_no_tiny_copy():
    m = get_module("staircase")(words="STAIR")
    for (Lbig, s2_t0, s2_t1, Lend), R in zip(m.keys_out, m.R_out):
        assert abs((s2_t1 - s2_t0) - min(m.Lb, s2_t1)) < 1e-6                      # the window goes straight from the big arc up the stairs (no tiny arc)


def test_library_module_gets_the_time_hits_stay():
    """R16: the engine left the library module 0.4 s between a long exit and a long form-in. With the outline hand-off A's exit and B's form-in shrink
    to the outline fade / ease and the module gets the rest; A's release and B's hit do not move."""
    out = {}
    for mode in ("gate", "outline"):
        prev, SQ.HANDOFF = SQ.HANDOFF, mode
        try:
            A = get_module("lanes")(words="GLIDE", timing=Timing(1.2, 0.8, 0.9))
            B = get_module("lanes")(words="RIDE", timing=Timing(1.6, 0.8, 0.9))
            s = SQ.Sequence([A, get_module("braid")(dur=0.4), B])
        finally:
            SQ.HANDOFF = prev
        (a, a0, _), (t, t0, t1), (b, b0, _) = s.timeline
        out[mode] = (a0 + a.timing.form + a.timing.hold, b0 + b.timing.form, t1 - t0)
    assert abs(out["gate"][0] - out["outline"][0]) < 1e-9 and abs(out["gate"][1] - out["outline"][1]) < 1e-9
    assert out["outline"][2] > out["gate"][2] + 1.0


def test_a_second_sequence_of_the_same_words_keeps_the_hits():
    """tools/fusion/spec.py builds Sequence.auto on the words, then the real Sequence with the planned transitions: the hits must still not move."""
    A = get_module("lanes")(words="GLIDE", timing=Timing(1.2, 0.8, 0.9))
    B = get_module("lanes")(words="RIDE", timing=Timing(1.6, 0.8, 0.9))
    SQ.Sequence([A, get_module("braid")(dur=0.4), B])
    s = SQ.Sequence([A, get_module("braid")(dur=0.4), B])
    (a, a0, _), (t, t0, t1), (b, b0, _) = s.timeline
    assert abs(a0 + a.timing.form + a.timing.hold - 2.0) < 1e-9 and abs(b0 + b.timing.form - 4.9) < 1e-9 and t1 - t0 > 1.4
    s2 = SQ.Sequence([A, get_module("cut")(dur=0.4), B])                           # a cut afterwards: the words are back to their own styles and timing
    assert A.form_out.is_lines and abs(A.timing.exit - 0.9) < 1e-9 and abs(B.timing.form - 1.6) < 1e-9
