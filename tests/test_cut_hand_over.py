"""Hand-over of a cut-owned word (2026-10-30): the promoted cuts (circle_grid word form, depth_lines) form B's word with the TYPE STAGE's layout of that word (rows, roll, cap height, face, axes),
so the last cut frame equals the first type-stage frame. Glyph mask of the cut (explore_cuts2.word_layout from pivot_cut.explore_plan's params) vs the stage's landed mask
(layout_preview.flat_frame, the stage's own placement of a word): IoU >= 0.95 for a line, a stack, a grid and a rolled layout. CPU, small frames."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import explore_cuts2 as E         # noqa: E402
from tools.fusion import layout_preview as LP       # noqa: E402
from tools.fusion import pivot_cut as PC            # noqa: E402
from tools.fusion import recipe_eval as RE          # noqa: E402
from tools.fusion import stable_type as ST          # noqa: E402

W, H = 288, 512
FONT = "Blackout_Midnight.ttf"


def piece(kind, word):
    return {"fps": 24, "audio": {"bpm": 88}, "shots": [{"id": "s1", "frames": 120}, {"id": "s2", "frames": 120}],
            "compose": {"crossfade": 0, "cuts": [{"after": "s1", "cut": kind, "params": {"word": "auto"} if kind == "circle_grid" else {}}]},
            "type": {"bpm": 88, "beats": [0.4, 1.08], "palette": {"fg": [238, 240, 242], "bg": [27, 31, 43]},
                     "words": [{"text": "ONE", "t0": 0.2, "t1": 3.0, "exit_s": 0.4, "render": "flat", "font": "Interlope-Regular.ttf", "form_s": 0.6}, word]}}


def drawn(text, layout):
    """the word as stable_type.fit_box draws it: lines + roll in placement."""
    box = ST.fit_box(text, FONT, None, layout, W / float(H))
    w = {"text": text, "t0": 5.2, "t1": 9.0, "exit_s": 0.4, "render": "flat", "font": FONT, "form_s": 0.6, "placement": {"roll": box["roll"]}}
    if layout in ("stack", "grid", "column"):
        w["lines"] = box["lines"]
    return w


def iou(a, b):
    a, b = a > 127, b > 127
    return (a & b).sum() / float((a | b).sum())


@pytest.mark.parametrize("kind", ["depth_lines", "circle_grid"])
@pytest.mark.parametrize("text,layout", [("LAMP", "line"), ("WARM LIGHT", "stack"), ("DREAMING", "grid"), ("HORSE", "rot90")])
def test_the_cut_lands_the_stage_mask(kind, text, layout):
    w = drawn(text, layout)
    assert LP.layout_of(w) == {"line": "line", "stack": "stack", "grid": "grid", "rot90": "rot +90"}[layout]
    p = piece(kind, w)
    lay, _ = RE.piece_layout(p)
    xp = PC.explore_plan(p, p["compose"]["cuts"] and PC.specs(p)[0], lay)
    assert xp["owned"] == 1 and not xp["problems"], xp["problems"]
    prm = xp["params"]
    assert prm["stage"]["text"] == text and (prm["stage"].get("lines") or None) == (w.get("lines") or None)
    union, lid, glyphs = E.word_layout(prm["word"], prm["font"], prm.get("axes"), W, H, prm.get("wf", 0.8), stage=prm["stage"])
    _, stage2 = LP.flat_frame(w, size=(2 * W, 2 * H))                     # the type stage's landed hold frame of the word, at the cut's own 2x mask resolution
    assert iou(union, stage2) >= 0.99, (kind, layout, iou(union, stage2))
    cut = cv2.resize(union, (W, H), interpolation=cv2.INTER_AREA)
    _, stage4 = LP.flat_frame(w, size=(4 * W, 4 * H))                     # ... and a finer stage frame shrunk to the frame size (flat_frame's own integer font size at 1x moves the mask by ~2 %)
    stage = cv2.resize(stage4, (W, H), interpolation=cv2.INTER_AREA)
    assert iou(cut, stage) >= 0.95, (kind, layout, iou(cut, stage))
    assert glyphs and lid.max() >= 0


def test_a_line_layout_differs_from_the_old_one_line_form():
    """the grid word is NOT the old centred one-liner: the stage's rows are what the cut forms now."""
    w = drawn("DREAMING", "grid")
    old = E.word_layout("DREAMING", FONT, None, W, H)[0]
    new = E.word_layout("DREAMING", FONT, None, W, H, stage=w)[0]
    assert iou(cv2.resize(old, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(new, (W, H), interpolation=cv2.INTER_AREA)) < 0.5
