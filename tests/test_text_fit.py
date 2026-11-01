"""Display auto-fit (balanced line breaks in tall boxes), per-format text, and the letter verbs / bold-type
options with multi-line text."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import plan as P  # noqa: E402
from mograph.geometry import frame_size  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from mograph.render.common import Ctx  # noqa: E402
from mograph.render.text import (FONTS, TextLayer, _bind_short, auto_fit_on, balanced_lines,  # noqa: E402
                                 fit_text, layer_text)
from render_helpers import base_plan, text_layer  # noqa: E402

FPS = 20
NOIMG = {"images": []}
# tall in 9:16 (pixel h/w ~ 1.06 -> auto-fit on), a wide band in 16:9 (h/w ~ 0.21 -> off)
TALL = {"9:16": [0.08, 0.25, 0.84, 0.5], "16:9": [0.1, 0.35, 0.8, 0.3]}
ANTON = FONTS["Big Shoulders Display"]        # was Anton: liked tall face of the same role


def _plan(layers, duration=1.0, bg=None, display="Big Shoulders Display"):
    p = base_plan(duration, FPS)
    p["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
    p["style"]["fonts"] = {"display": display, "body": "Space Grotesk"}
    p["scenes"] = [{"id": "s0", "t0": 0.0, "t1": duration,
                    "background": bg or {"type": "color", "color": "palette:0"},
                    "layers": layers, "transition_out": {"type": "cut", "dur": 0.0}}]
    return p


def _layer_obj(layer, fmt="9:16", display="Big Shoulders Display"):
    plan = _plan([layer], display=display)
    W, H = frame_size(fmt, "draft")
    return TextLayer(layer, Ctx(plan, NOIMG, fmt, W, H), 0)


def _lines(tl):
    return [" ".join(words) for words, _ in tl.lay["lines"]]


def _frames(plan, analysis=NOIMG, fmt="9:16"):
    return [f.astype(np.int32) for f in render_frames(plan, analysis, fmt)]


def _ink(f):
    return float((f.mean(axis=2) > 128).mean())


def _bands(mask, min_gap=2):
    """Vertical runs of rows that contain any True pixel (one per rendered text line)."""
    rows = mask.any(axis=1)
    bands, start, gap = [], None, 0
    for y, r in enumerate(rows):
        if r:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                bands.append((start, y - gap))
                start, gap = None, 0
    if start is not None:
        bands.append((start, len(rows) - 1))
    return bands


def _lit(f):
    return f.mean(axis=2) > 128


# ----------------------------------------------------------------------------- line-breaking choice

def test_short_words_bind_to_the_next_word():
    assert _bind_short("THE CITY OF LIGHTS".split()) == ["THE CITY", "OF LIGHTS"]
    assert _bind_short("and the sea.".split()) == ["and the sea."]
    assert _bind_short("rock and".split()) == ["rock and"]               # a trailing short word joins the last unit
    assert _bind_short("between here".split()) == ["between", "here"]


def test_balanced_lines_choice():
    # tall box: two short words stack, which renders bigger than one line
    two = balanced_lines("between here", ANTON, 900, 1000, 1.0)
    assert two == "between\nhere"
    one = fit_text("between here", ANTON, 900, 1000, 1.0, wrap=False)
    stacked = fit_text(two, ANTON, 900, 1000, 1.0, wrap=False)
    assert stacked["size"] > 1.3 * one["size"]
    # never breaks after a short word; balanced halves
    assert balanced_lines("THE CITY OF LIGHTS", ANTON, 900, 1200, 1.0) == "THE CITY\nOF LIGHTS"
    assert balanced_lines("and the sea.", ANTON, 900, 1000, 1.0) == "and the sea."
    assert balanced_lines("Somewhere", ANTON, 900, 1000, 1.0) == "Somewhere"
    # the chosen line count is the one that renders largest (1..4 lines)
    txt = "MAKE IT MOVE RIGHT NOW"
    got = balanced_lines(txt, ANTON, 900, 1400, 1.0)
    assert 2 <= got.count("\n") + 1 <= 4
    best = fit_text(got, ANTON, 900, 1400, 1.0, wrap=False)["size"]
    w = txt.split()
    for cuts in ([2], [1, 3], [1, 2, 4], [3]):
        edges = [0] + cuts + [len(w)]
        alt = "\n".join(" ".join(w[a:b]) for a, b in zip(edges, edges[1:]))
        assert best >= 0.97 * fit_text(alt, ANTON, 900, 1400, 1.0, wrap=False)["size"], alt
    # balanced, not greedy: line widths within a factor of ~2
    f_lines = got.split("\n")
    assert max(len(s) for s in f_lines) <= 2 * min(len(s) for s in f_lines) + 2


def test_auto_fit_gate():
    lay = text_layer("NEON HEART", 0.0, 1.0, box=TALL)
    assert auto_fit_on(lay, "NEON HEART", 900, 1000)
    assert not auto_fit_on(lay, "NEON HEART", 900, 300)                    # wide box: plain fit
    assert not auto_fit_on(lay, "NEON\nHEART", 900, 1000)                  # explicit breaks kept
    assert not auto_fit_on(dict(lay, fit="none"), "NEON HEART", 900, 1000)
    body = dict(lay, role="caption", font="body")
    assert not auto_fit_on(body, "NEON HEART", 900, 1000)                  # not a display layer
    assert auto_fit_on(dict(body, fit="auto"), "NEON HEART", 900, 1000)    # opted in


def test_text_layer_stacks_in_portrait_only():
    lay = text_layer("NEON HEART", 0.0, 1.0, "cut", "cut", box=TALL)
    port = _layer_obj(lay, "9:16")
    land = _layer_obj(lay, "16:9")
    assert _lines(port) == ["NEON", "HEART"]
    assert _lines(land) == ["NEON HEART"]
    off = _layer_obj(dict(lay, fit="none"), "9:16")
    assert off.size <= port.size
    # explicit "\n" respected (no re-break into "A B / C D")
    exp = _layer_obj(text_layer("A\nB C D", 0.0, 1.0, "cut", "cut", box=TALL), "9:16")
    assert _lines(exp) == ["A", "B C D"]
    # the stacked portrait type fills most of the frame width
    W = frame_size("9:16", "draft")[0]
    assert port.lay["width"] > 0.7 * W


# ----------------------------------------------------------------------------- per-format text

def test_per_format_text_override():
    txt = {"9:16": "NEON\nHEART", "16:9": "NEON HEART"}
    lay = text_layer(txt, 0.0, 1.0, "cut", "cut", fit="none")
    assert layer_text(lay, "9:16") == "NEON\nHEART" and layer_text(lay, "16:9") == "NEON HEART"
    assert P.validate_plan(_plan([lay]), NOIMG) == []
    assert _lines(_layer_obj(lay, "9:16")) == ["NEON", "HEART"]
    assert _lines(_layer_obj(lay, "16:9")) == ["NEON HEART"]
    f9 = _frames(_plan([lay]))[5]
    f16 = _frames(_plan([lay]), fmt="16:9")[5]
    assert len(_bands(_lit(f9))) == 2 and len(_bands(_lit(f16))) == 1
    # validation
    for bad in ({"9:16": "NEON"}, {"9:16": "NEON", "16:9": ""}, {"9:16": "A", "16:9": "B", "1:1": "C"}, 7):
        errs = P.validate_plan(_plan([text_layer(bad, 0.0, 1.0)]), NOIMG)
        assert any(".text" in e for e in errs), bad
    assert any(".fit" in e for e in P.validate_plan(_plan([text_layer("X", 0.0, 1.0, fit="shrink")]), NOIMG))


# ----------------------------------------------------------------------------- verbs with multi-line text

def _two_lines(f):
    b = _bands(_lit(f))
    return len(b) == 2 and b[0][1] < b[1][0]


def test_row_slide_multiline():
    lay = text_layer("NEON HEART", 0.0, 1.0, "row_slide", "cut", box=TALL,
                     **{"in": {"verb": "row_slide", "dur": 0.4, "rows": 1, "travel": 0.3}})
    fr = _frames(_plan([lay]))
    ref = _frames(_plan([text_layer("NEON HEART", 0.0, 1.0, "cut", "cut", box=TALL)]))
    assert _two_lines(fr[12]) and _two_lines(ref[12])
    assert np.abs(fr[2] - fr[12]).mean() > 0.5                            # sliding in
    assert np.abs(fr[12] - ref[12]).mean() < 2.0                          # lands in register
    three = _frames(_plan([dict(lay, **{"in": {"verb": "row_slide", "dur": 0.4, "rows": 3}})]))[12]
    assert _ink(three) > 1.3 * _ink(fr[12])                               # the 2-line block repeats as rows


def test_scramble_multiline():
    lay = text_layer("NEON HEART", 0.0, 1.0, "scramble", "cut", box=TALL, **{"in": {"verb": "scramble", "dur": 0.5}})
    fr = _frames(_plan([lay]))
    ref = _frames(_plan([text_layer("NEON HEART", 0.0, 1.0, "cut", "cut", box=TALL)]))
    assert _two_lines(fr[3])                                             # both lines scramble
    assert np.abs(fr[3] - ref[3]).mean() > 0.5
    assert np.array_equal(fr[12], ref[12])


def test_vary_multiline():
    lay = text_layer("WAVE FORM", 0.0, 2.0, "cut", "cut", box=TALL,
                     vary={"axis": "wght", "range": [100, 900], "period": 1.0, "phase": 0.1})
    p = _plan([lay], duration=2.0, display="Anybody")
    assert P.validate_plan(p, NOIMG) == []
    assert _lines(_layer_obj(lay, display="Anybody")) == ["WAVE", "FORM"]
    fr = _frames(p)
    inks = [_ink(f) for f in fr[:-7]]
    assert max(inks) > 1.2 * min(inks) > 0
    for f in fr[:-7]:
        assert _two_lines(f)
        lit = np.nonzero(_lit(f))[1]
        assert lit.min() > 0 and lit.max() < f.shape[1] - 1                # each line stays inside the frame


def test_outline_multiline():
    fill = _frames(_plan([text_layer("HOLLOW TYPE", 0.0, 1.0, "cut", "cut", box=TALL)]))[5]
    ol = _frames(_plan([text_layer("HOLLOW TYPE", 0.0, 1.0, "cut", "cut", box=TALL, outline={"width": 0.03})]))[5]
    assert _two_lines(fill) and _two_lines(ol)
    assert 0 < _ink(ol) < 0.6 * _ink(fill)


def _red_image():
    px = np.zeros((400, 300, 3), np.float32)
    px[..., 0] = 1.0
    return {"images": [{"id": "img0", "pixels": px, "width": 300, "height": 400, "focus": [0.5, 0.5],
                        "contours": [], "lines": []}]}


def test_window_fill_multiline():
    ana = _red_image()
    lay = text_layer("RED SEA", 0.0, 1.0, "cut", "cut", box=TALL, window={"mode": "fill", "src": "img0", "zoom": 1.0})
    p = _plan([lay])
    assert P.validate_plan(p, ana) == []
    f = _frames(p, ana)[5]
    red = (f[..., 0] > 200) & (f[..., 1] < 60) & (f[..., 2] < 60)
    assert len(_bands(red)) == 2 and red.mean() > 0.03


def test_window_knockout_multiline():
    bg = {"type": "color", "color": "#00ff00"}
    lay = text_layer("CUT OUT", 0.0, 1.0, "cut", "cut", box=TALL, window={"mode": "knockout", "plate": "palette:0"})
    f = _frames(_plan([lay], bg=bg))[5]
    green = (f[..., 1] > 200) & (f[..., 0] < 60)
    assert len(_bands(green)) == 2
    assert 0.03 < green.mean() < 0.5 and (f.mean(axis=2) < 40).mean() > 0.4
