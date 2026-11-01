"""SOLID LETTERS BY DEFAULT (tools/fusion/letter_fill.py, user 2026-11-01: "I'd like the text not to be filled with that textured pattern"). The textured modes stay and run
when the user's direction asks for them. Pure python. Run: python -m pytest tests/test_letter_fill.py -q"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import letter_fill as LF  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

KNOCK = {"op": "knockout", "out": "fx", "fx": True, "args": {"plate_rgb": [0.05, 0.05, 0.06], "type_a": "$type_matte", "footage": "$footage", "fg_matte": "$m", "min_contrast": 0.22}}
WOTHER = {"op": "window_cut", "out": "image", "args": {"type_a": "$type_matte", "a": "$footage", "b": "$next", "mode": "other", "fg_matte": "$m", "t": "$t"}}
WCUT = {"op": "window_cut", "out": "image", "args": {"type_a": "$type_matte", "a": "$footage", "b": "$next", "mode": "cut", "t": "$t"}}


def _piece(direction=None, render=None):
    p = {"plan": {"piece_system": {"palette": {"bg": [20, 20, 24], "fg": [235, 230, 220], "accents": [[200, 60, 40]]}}},
         "type": {"words": [{"text": "GOLDEN HOUR", "render": render or "flat", "material": "polished crimson marble" if render else None}]}}
    if direction:
        p["direction"] = direction
    return p


def test_default_letter_fill_is_solid():
    p = _piece()
    assert LF.solid(p)
    ops, n = LF.solid_ops(p, [KNOCK, WOTHER, WCUT])
    assert n == 2
    fill = next(o for o in ops if o["op"] == "fill" and o["out"].startswith("_solid"))
    kn = next(o for o in ops if o["op"] == "knockout")
    assert kn["args"]["footage"] == "$" + fill["out"] and kn["args"]["fg"] == "$footage"        # letters solid, the subject in front keeps its footage
    assert isinstance(fill["args"]["color"], list) and fill.get("fx") is True                       # an ink, gated with its effect
    assert sum(fill["args"]["color"]) / 3 > 0.5                                                     # light ink on the dark knockout ground
    sw = next(o for o in ops if o["op"] == "sandwich")
    assert sw["args"]["type_rgb"] == "$type_beauty" and sw["args"]["footage"] == "$footage"        # window_other -> the standard drawer (inked from the shot by the legibility pass)
    assert any(o["op"] == "window_cut" and o["args"]["mode"] == "cut" for o in ops)                # a shot cut THROUGH the letters is kept
    assert LF.solid_mode(p, "plate_in_type") not in LF.TEXTURED_MODES and LF.solid_mode(p, "window_subject_front") == "plate_in_subject"      # no plate / footage inside the letters
    assert LF.solid_mode(p, "plate_in_subject") == "plate_in_subject" and LF.solid_mode(p, None) is None
    p2 = _piece(render="handoff")
    notes = LF.solid_owners(p2)
    assert notes and p2["type"]["words"][0]["render"] == "flat" and p2["type"]["words"][0]["render_asked"] == "handoff"


def test_colour_words_and_on_screen_words_keep_solid():
    for d in ("palette: crimson, gold", "more legible text", "GOLDEN HOUR", "make it in red and gold", "BORN IN GOLD", "STEEL HEART"):
        assert LF.solid(_piece(d)), d


def test_a_user_direction_asks_for_texture():
    for d in ("polished marble letters", "put the footage inside the letters", "letters as windows onto the shot", "textured text please", "fill the letters with the image",
              "metallic type"):
        assert US.wants_textured_letters(d), d
        p = _piece(d, render="handoff")
        assert not LF.solid(p)
        ops, n = LF.solid_ops(p, [KNOCK, WOTHER])
        assert n == 0 and ops == [KNOCK, WOTHER]
        assert LF.solid_mode(p, "plate_in_type") == "plate_in_type"
        assert LF.solid_owners(p) == [] and p["type"]["words"][0]["render"] == "handoff"
    assert not LF.solid({"compose": {"letter_fill": "texture"}})
    assert not LF.solid({"user_style": {"textured": True}})
def test_plate_in_type_is_plain_type_not_a_flat_ground_mode():
    """render R17: plate_in_type used to be remapped to plate_in_subject = a flat ground over the whole shot with a ghost subject. Now: no plate mode (plain type over the footage)."""
    p = _piece()
    assert LF.solid_mode(p, "plate_in_type") is None
    assert LF.solid_mode(p, "window_subject_front") == "plate_in_subject" and LF.solid_mode(p, "text_as_mask") == "plate_in_subject"
    assert LF.solid_mode(_piece("put the footage inside the letters"), "plate_in_type") == "plate_in_type"        # asked for: kept
