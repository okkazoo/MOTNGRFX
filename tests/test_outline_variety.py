"""2026-11-01 (user: "it doesnt always have to be complimentary ... mostly black though ... all the time no"): the default text-outline colour varies word to word
between families (complement / shot-palette hue / near-black / light), every pick keeps the 3:1 contrast, black stays a minority, a USER palette stays user hues only.
Pure maths (legible_color.support_pick / support_color), numpy, CPU."""
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import legible_color as LC  # noqa: E402

INKS = [[0.95, 0.75, 0.20], [0.55, 0.65, 0.95], [0.85, 0.45, 0.55], [0.35, 0.85, 0.60], [0.90, 0.60, 0.30], [0.60, 0.50, 0.90]]      # mid-to-light sRGB inks (6 words)
SHOT = [dict(h=30.0, c=0.1), dict(h=145.0, c=0.1), dict(h=265.0, c=0.1), dict(h=330.0, c=0.1)]


def _hue(rgb):
    lab = LC.lin_to_oklab(LC.srgb_to_lin(np.asarray(rgb, np.float32)))
    return math.degrees(math.atan2(lab[2], lab[1])) % 360.0


def _piece(base):
    out = []
    for i, rgb in enumerate(INKS):
        ink = LC.srgb_to_lin(np.array(rgb, np.float32))
        light = float(LC.rel_lum(ink)) <= 0.18
        out.append((ink, *LC.support_pick(ink, SHOT, light=light, pick=base + i)))
    return out


def _is_black(lin):
    return float(LC.rel_lum(lin)) < 0.01


def test_families_vary_and_pass_contrast_and_black_is_a_minority():
    for base in (0, 1, 5, 6, 37, 90):                                                   # any per-piece offset
        res = _piece(base)
        fams = [f for _i, _s, f in res]
        assert len(set(fams)) >= 3, (base, fams)
        for ink, sup, _f in res:
            assert LC.contrast_ratio(float(LC.rel_lum(ink)), float(LC.rel_lum(sup))) >= 3.0
        assert sum(_is_black(s) for _i, s, _f in res) <= len(res) / 3.0, (base, fams)
        assert all(a != b for a, b in zip(fams, fams[1:])), (base, fams)               # never the same family twice in a row


def test_pick_none_keeps_the_old_complement_and_support_color_matches_pick():
    ink = LC.srgb_to_lin(np.array([0.95, 0.75, 0.2], np.float32))
    old = LC.support_color(ink, SHOT, light=False)
    assert LC.hue_dist(_hue(LC.lin_to_srgb(old)), _hue(LC.lin_to_srgb(ink))) >= 90
    assert np.allclose(LC.support_color(ink, SHOT, light=False, pick=3), LC.support_pick(ink, SHOT, light=False, pick=3)[0])


def test_user_palette_gives_user_hues_only_whatever_the_pick():
    hues = LC.hues_of_colors([[0.8, 0.08, 0.16], [0.9, 0.75, 0.2]])
    pal_h = [h["h"] for h in hues]
    for pick in range(12):
        for rgb, light in (([1.0, 0.55, 0.55], False), ([1.0, 0.89, 0.66], False), ([0.45, 0.05, 0.1], True)):
            ink = LC.srgb_to_lin(np.array(rgb, np.float32))
            sup = LC.support_color(ink, hues, light=light, pick=pick)
            assert min(LC.hue_dist(_hue(LC.lin_to_srgb(sup)), ph) for ph in pal_h) < 30
            assert LC.contrast_ratio(float(LC.rel_lum(ink)), float(LC.rel_lum(sup))) >= 3.0


def test_dark_ink_still_gets_contrasting_outlines_when_a_family_cannot():
    ink = LC.srgb_to_lin(np.array([0.10, 0.12, 0.25], np.float32))                     # near-black ink: a dark family cannot reach 3:1, falls through
    for pick in range(24):
        sup, fam = LC.support_pick(ink, SHOT, light=True, pick=pick)
        assert LC.contrast_ratio(float(LC.rel_lum(ink)), float(LC.rel_lum(sup))) >= 3.0, (pick, fam)
