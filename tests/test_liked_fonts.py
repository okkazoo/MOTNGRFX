"""Hard rule: only the user's liked fonts (mograph/fonts/liked.json). CPU only.
Run: MOGRAPH_SKIP_SLOW=1 MOGRAPH_FAST=1 python -m pytest tests/test_liked_fonts.py -q"""
from __future__ import annotations

import json
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph import font_registry as fr  # noqa: E402
from mograph import fonts_liked as fl  # noqa: E402
from mograph.modules import core  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402

FONT_DIR = os.path.join(ROOT, "mograph", "fonts")
LIKED_KEYS = """Anybody BigShouldersDisplay Blackout-Midnight FacadeVF Interlope MajorMonoDisplay SpaceGrotesk-Bold Jacquard12
Jacquard24 JacquardaBastarda9 BungeeHairline BungeeInline BungeeOutline Codystar LondrinaSketch Monofett RubikBrokenFax
RubikGlitchPop RubikLines ZenTokyoZoo BadeenDisplay EricaOne Fascinate FolditThin Honk MoiraiOne Ultra PassionOne-Black
AlienBlock BBHBartle Barrio Bitcount BitcountSingle Bytesized Doto GeistPixel LibreBarcode39Text NixieOne PaletteMosaic
Rock3D Silkscreen Tourney""".split()
NOT_LIKED = ["Monoton", "Pilowlava", "Bodoni Moda", "Unbounded", "JetBrainsMono-Bold", "Space Mono", "Syne", "Bungee Shade",
             "Rubik", "Rubik[wght].ttf", "Fraunces", "Inter", "Le Murmure", "Anton", "Bungee", "Playfair Display"]
MOODS = ["calm", "hype", "warm", "nostalgic", "techno", "luxury", "playful", "dark"]


def test_liked_json_has_the_42():
    assert [e["key"] for e in fl.LIKED] == LIKED_KEYS
    assert len(fl.LIKED) == 42 and all(e["role"] in fl.ROLES for e in fl.LIKED)


@pytest.mark.parametrize("key", LIKED_KEYS)
def test_every_liked_font_resolves_to_an_existing_file(key):
    e = fl.require_liked(key)
    assert os.path.isfile(os.path.join(FONT_DIR, e["file"])), e["file"]
    assert e["family"] in fr.REGISTRY and fr.REGISTRY[e["family"]]["file"] == e["file"]
    for spelling in (e["family"], e["file"], key):               # every spelling a spec may use resolves
        assert fl.require_liked(spelling) is e
        k, _loc = core.resolve_font(spelling)
        assert os.path.isfile(core.ol.font_path(k))


def test_foldit_is_the_thin_weight():
    assert fl.require_liked("FolditThin")["axes"] == {"wght": 100}
    assert core.resolve_font("Foldit")[1] == {"wght": 100}


@pytest.mark.parametrize("name", NOT_LIKED)
def test_unliked_fonts_raise(name):
    with pytest.raises(fl.FontNotLiked):
        fl.require_liked(name)
    with pytest.raises(KeyError):
        core.resolve_font(name)


def test_piece_spec_with_unliked_font_is_rejected():
    from tools.fusion import piece_runner as prn
    prn.check_liked_fonts({"type": {"words": [{"font": "Anybody"}, {"font": "Big Shoulders Display"}]},
                           "signoff": {"font": "Space Grotesk"}})
    with pytest.raises(fl.FontNotLiked):
        prn.check_liked_fonts({"type": {"words": [{"font": "Anybody"}, {"font": "Monoton"}]}})
    with pytest.raises(fl.FontNotLiked):
        prn.check_liked_fonts({"type": {"words": []}, "signoff": {"font": "Space Mono", "off": False}})
    prn.check_liked_fonts({"type": {"words": []}, "signoff": {"font": "Space Mono", "off": True}})     # not drawn


def test_planner_picks_use_only_liked_fonts_and_lines_alternate_role():
    rng = random.Random(7)
    seen = set()
    for i in range(50):
        moods = rng.sample(MOODS, rng.randint(1, 3))
        energy, n = rng.random(), rng.randint(1, 8)
        files = pr.pick_fonts(moods, energy, rng.randint(0, 10 ** 6), n)
        assert 2 <= len(files) <= 3
        roles = []
        for f in files:
            e = fl.require_liked(f)
            assert e["file"] == f
            roles.append(e["role"])
        seen.update(files)
        assert len(set(roles)) == len(roles), roles                  # distinct roles
        seq = [roles[k % len(roles)] for k in range(n + 3)]          # the order lines use them in
        assert all(a != b for a, b in zip(seq, seq[1:])), seq
        # scale / weight / fill contrast between consecutive lines
        prev, prev_bleed = None, 1.0
        for k in range(n + 3):
            cur = files[k % len(files)]
            bleed = pr.line_scale(prev, cur, prev_bleed)
            if prev is not None:
                diff = fl.contrast_axes(fl.role_of(prev), fl.role_of(cur))
                assert diff or bleed != prev_bleed, (prev, cur)
            prev, prev_bleed = cur, bleed
    assert len(seen) >= 8                                            # the pick is not stuck on a few faces


def test_planned_pieces_use_only_liked_fonts():
    from test_fusion_plan import synth_music
    for seed in range(1, 6):
        piece = pr.plan_piece({"words": "AFTER HOURS / THE CITY KEEPS DREAMING / NOTHING SLEEPS / NEON RAIN / LAST TRAIN",
                               "scene": "", "music": "", "length": 12, "look": "auto", "seed": seed, "quality": "final"},
                              synth_music(120 + seed * 7, 12), dry=False)
        ws = piece["type"]["words"]
        for w in ws:
            fl.require_liked(w["font"])
        fl.require_liked(piece["signoff"]["font"])
        roles = [fl.role_of(w["font"]) for w in ws]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles
