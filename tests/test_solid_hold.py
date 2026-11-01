"""NO SPECKLE LETTERS (user 2026-11-01, R18 / R19: "i dont like the text being filled with texture like this ... not this bitsy texture fill", then "throughout any generation").
1. the `particles` word module holds SOLID: inside the glyph mask the hold frames are one ink (coverage >= 0.9, low variance); the old dot-cloud look (solid fill off) fails the same check.
2. a piece / reply that asks for the `particles` module or io style is swapped at plan time (owner.fix_speckle_forms via text_guard.fix_plan), and no choosable set offers it any more.
Run: python -m pytest tests/test_solid_hold.py -q"""
import copy
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from mograph.modules import Placement, Timing  # noqa: E402
from mograph.modules import sequence as sq  # noqa: E402
from mograph.modules.words.particles import Particles  # noqa: E402
from mograph.render import cam3d  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import openings as OP  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402
from tools.fusion import text_guard as TG  # noqa: E402

SIZE = (480, 270)
TIMING = Timing(1.2, 1.5, 1.0)
BG = (12, 12, 16)


def _module(**kw):
    return Particles(words=["RISING SWELL"], font="Anybody[wdth,wght].ttf", placement=Placement.centred(), timing=TIMING, **kw)


def _frame(layers):
    return sq._draw(layers, cam3d.Camera.home(), BG, SIZE, 2)


def _interior(m, t):
    """(luminance inside the glyph mask, ink luminance): the mask is the solid fill at the frame's own pulse, eroded past the fine outline."""
    p = m.pulse_at(t)
    fill = [L for L in m._solid_layers(1.0, p) if L.kind == "fill"]
    mask = sq._draw(fill, cam3d.Camera.home(), (0, 0, 0), SIZE, 2).max(axis=2) > 127
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    img = _frame(m.layers(t)).astype(float)
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    ink = np.array(m.palette.fg, float) @ np.array([0.2126, 0.7152, 0.0722])
    assert mask.sum() > 2000
    return lum[mask], ink


def test_particles_hold_is_solid_pulse_on_and_off():
    for kw in ({}, {"pulse": False}):
        m = _module(**kw)
        for t in (1.25, 1.6, 1.95, 2.4, 2.65):                 # the hit, the beats, the late hold
            v, ink = _interior(m, t)
            cover = float((v >= 0.8 * ink).mean())
            assert cover >= 0.9, (kw, t, cover)
            assert float(v.std()) <= 0.08 * ink, (kw, t, float(v.std()))


def test_dot_cloud_hold_fails_the_same_check():
    """the old look (solid fill switched off) is a speckle: the check is not vacuous."""
    m = _module()
    m.solid_at = lambda t: 0.0
    v, ink = _interior(m, 1.95)
    assert float((v >= 0.8 * ink).mean()) < 0.9 or float(v.std()) > 0.08 * ink


def test_solid_melts_in_by_the_hit_and_lets_go_at_the_exit():
    m = _module()
    tm = m.timing
    assert m.solid_at(0.0) == 0.0 and m.solid_at(tm.form) > 0.999
    assert all(m.solid_at(tm.form + x * tm.hold) > 0.999 for x in (0.0, 0.5, 1.0))
    assert m.solid_at(tm.total) < 0.001


def _piece(mods=("particles", "particles", "particles"), fin=("particles", "blobs", "particles"), fout=("blobs", "wipe", "blobs")):
    words = [dict(text=t, module=m, form_in=a, form_out=b, params=dict(n=3000, size=2, bpm=120)) for t, m, a, b in zip(("STILLNESS", "GLOWING", "ASCEND"), mods, fin, fout)]
    return {"type": {"words": words}, "plan": {"piece_system": {"module": "particles", "modules": ["particles"], "io": {"form_in": ["particles", "blobs"], "form_out": ["blobs", "wipe"]}}}}


def test_a_piece_that_asks_for_particles_is_swapped():
    p = _piece(fout=("particles", "wipe", "blobs"))
    fixes = TG.fix_plan(p)                                      # the design-step entry point
    ws = p["type"]["words"]
    assert not any(w["module"] in DD.SPECKLE_MODULES for w in ws)
    assert not any(w[k] in DD.SPECKLE_IO for w in ws for k in ("form_in", "form_out"))
    assert len({w["module"] for w in ws}) >= 2                  # not one module for every word again
    assert len({w["module"] for w in ws}) <= DD.MAX_MODULES
    assert all("n" not in w["params"] and "size" not in w["params"] for w in ws)          # the particles module's own parameters go with it
    ps = p["plan"]["piece_system"]
    assert "particles" not in ps["modules"] and ps["module"] != "particles" and "particles" not in sum(ps["io"].values(), [])
    assert any("speckle" in f for f in fixes)
    assert OW.fix_speckle_forms(p) == []                        # idempotent


def test_a_clean_piece_is_untouched():
    p = _piece(mods=("lanes", "slats", "lanes"), fin=("lines", "flicker", "wipe"), fout=("wipe", "lines", "flicker"))          # blobs is a speckle io since R20 (dots at the exit)
    p["plan"]["piece_system"] = {"module": "lanes", "modules": ["lanes", "slats"], "io": {"form_in": ["lines", "flicker"], "form_out": ["flicker", "wipe"]}}
    before = copy.deepcopy(p)
    assert OW.fix_speckle_forms(p) == [] and p == before


def test_no_choosable_set_offers_particles():
    assert "particles" in DD.AVOID_MODULES and "particles" not in PS.KEPT_MODULES and "particles" not in DD.EFFECT_MODULES
    assert not any("particles" in r[:4] for rows in PS.IO_ROWS.values() for r in rows)
    assert "particles" not in sum(PS.FAMILY.values(), ())
    assert "particles" not in QP.WORD_MODULES and "particles" not in QP.IO_STYLES
    assert "particles" not in OP.STYLE_ORDER and "particles" not in OP.SECOND_STYLE
    assert all(OP.default_style(i, [], s) != "particles" for i in range(6) for s in range(6))
    for tone in ("soft", "tense", "driven", "bright"):                                    # the draw itself, every tone, many seeds
        for seed in range(40):
            sy = PS.choose(seed, mood=tone)
            assert "particles" not in sy["modules"] and "particles" not in sy["io"]["form_in"] + sy["io"]["form_out"], (tone, seed)
