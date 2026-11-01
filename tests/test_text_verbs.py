"""Letter-level text verbs: scramble (in) and row_slide (in / out)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import plan as P  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from render_helpers import base_plan, box, text_layer  # noqa: E402

FPS = 20


def _plan(layer, duration=1.0):
    p = base_plan(duration, FPS)
    p["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
    p["scenes"] = [{"id": "s0", "t0": 0.0, "t1": duration,
                    "background": {"type": "color", "color": "palette:0"},
                    "layers": [layer], "transition_out": {"type": "cut", "dur": 0.0}}]
    return p


def _frames(plan, fmt="9:16"):
    return [f.astype(np.int32) for f in render_frames(plan, {"images": []}, fmt)]


def _ink(f):
    """Fraction of pixels lit by the light text on the dark ground."""
    return float((f.mean(axis=2) > 128).mean())


def test_new_verbs_listed_and_validate():
    assert "scramble" in P.VERBS["text_in"] and "row_slide" in P.VERBS["text_in"]
    assert "row_slide" in P.VERBS["text_out"]
    lay = text_layer("SEA", 0.0, 1.0, **{"in": {"verb": "row_slide", "dur": 0.4, "rows": 3, "scale": 1.6},
                                         "out": {"verb": "row_slide", "dur": 0.2, "fade": True}})
    assert P.validate_plan(_plan(lay), {"images": []}) == []
    lay = text_layer("SEA", 0.0, 1.0, **{"in": {"verb": "scramble", "dur": 0.4}})
    assert P.validate_plan(_plan(lay), {"images": []}) == []


def test_scramble_resolves_to_the_plain_text():
    words = "BETWEEN HERE"
    scr = _frames(_plan(text_layer(words, 0.0, 1.0, "scramble", "cut", **{"in": {"verb": "scramble", "dur": 0.5}})))
    ref = _frames(_plan(text_layer(words, 0.0, 1.0, "cut", "cut")))
    # mid-scramble: glyphs are drawn (not empty) but differ from the resolved text, and change frame to frame
    assert _ink(scr[3]) > 0.002
    assert np.abs(scr[3] - ref[3]).mean() > 0.5
    assert np.abs(scr[3] - scr[4]).mean() > 0.2
    # after the in-phase the layer is exactly the plain text
    assert np.array_equal(scr[12], ref[12])


def test_scramble_is_deterministic():
    lay = text_layer("SCRAMBLE", 0.0, 1.0, "scramble", "cut", **{"in": {"verb": "scramble", "dur": 0.6}})
    a, b = _frames(_plan(lay)), _frames(_plan(lay))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_row_slide_rows_align_and_overflow():
    def lay(rows, scale, **kw):
        return text_layer("SEA", 0.0, 1.0, **{"in": {"verb": "row_slide", "dur": 0.4, "rows": rows,
                                                     "scale": scale, **kw}, "out": {"verb": "cut"}},
                          box=box([0.1, 0.4, 0.8, 0.2], [0.2, 0.35, 0.6, 0.3]))
    one = _frames(_plan(lay(1, 1.0)))
    three = _frames(_plan(lay(3, 1.0)))
    giant = _frames(_plan(lay(3, 2.2)))
    # aligned (after the in-phase): three rows carry ~3x the ink of one; the centre row matches plain text
    assert _ink(three[12]) > 2.4 * _ink(one[12])
    plain = _frames(_plan(text_layer("SEA", 0.0, 1.0, "cut", "cut", box=box([0.1, 0.4, 0.8, 0.2], [0.2, 0.35, 0.6, 0.3]))))
    assert np.abs(one[12] - plain[12]).mean() < 1.0
    # still sliding before alignment, then held still (no drift)
    assert np.abs(three[2] - three[12]).mean() > 1.0
    assert np.array_equal(three[11], three[13])          # (the clip's final fade starts after frame 13)
    # scale > 1 overflows the box: lit pixels touch both side edges of the frame
    g = giant[12].mean(axis=2) > 128
    assert g[:, :3].any() and g[:, -3:].any()


def test_row_slide_drift_track_and_fade_out():
    lay = text_layer("and the sea", 0.0, 1.0, case="as_is",
                     **{"in": {"verb": "row_slide", "dur": 0.4, "rows": 1, "travel": 0.05, "track": 0.5,
                               "drift": 0.02, "fade": True, "stagger": 0.6},
                        "out": {"verb": "row_slide", "dur": 0.3, "rows": 1, "travel": 0.0, "track": 0.6,
                                "fade": True}})
    plan = _plan(lay, duration=2.0)
    plan["style"]["motion"] = "smooth"                 # the calm ease this is meant for
    fr = _frames(plan)
    assert _ink(fr[0]) < _ink(fr[10])                     # fades in
    assert np.abs(fr[10] - fr[12]).mean() > 0.05          # drift: the hold never freezes
    assert _ink(fr[19]) < 0.5 * _ink(fr[12])              # tracks out and fades
