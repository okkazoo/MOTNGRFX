"""User 2026-10-30 (v027 r02): "some of the fonts used look a little plain and not part of the favourites I want used". Every face a planned + normalised piece draws is in mograph/fonts/liked.json
(the hard allowlist) and the faces are mostly the characterful liked ones; the plain ones (SpaceGrotesk, MajorMono) are rare. Run: python -m pytest tests/test_fonts_liked_only.py -q"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from mograph import fonts_liked as fl  # noqa: E402
from tools.fusion import random_system as RS  # noqa: E402


def test_the_draw_is_mostly_characterful_liked_faces_and_plain_ones_are_rare():
    share, plain, total = [], 0, 0
    for seed in range(120):
        for tone in ("soft", "tense", "driven", "bright"):
            fs = RS.draw_fonts(seed, tone, length_s=16)
            assert fs and all(fl.is_liked(f) for f in fs), fs
            share.append(sum(1 for f in fs if fl.role_of(f) in RS.EXPRESSIVE_ROLES) / len(fs))
            plain += sum(1 for f in fs if f in RS.PLAIN_FACES)
            total += len(fs)
    assert sum(share) / len(share) >= 0.55
    assert plain / total < 0.04


@pytest.mark.parametrize("seed", [3, 11, 21, 34, 55])
def test_a_whole_planned_and_normalised_piece_uses_only_liked_faces(seed):
    from test_fusion_core import synth_music
    from tools.fusion import design_defaults as DD
    from tools.fusion import plan_rules as PR
    from tools.fusion import stable_type as ST
    req = {"words": "FIRST WORD\nDREAMING\nWILD\nSEARCHING\nSTEAM\nAMBER", "scene": "a wide frozen lake under pale winter sun", "music": "calm piano", "length": 16, "look": "auto",
           "seed": seed, "quality": "preview", "format": "9:16", "shots": "cut"}
    plan = PR.plan_piece(req, synth_music(100, 16))
    words = plan["type"]["words"]
    fixes: list = []
    DD.enforce_type(words, fixes)
    ST.enforce(plan["type"], (plan.get("plan") or {}).get("piece_system"), 100, fixes)
    for w in words:
        assert fl.is_liked(w["font"]), (w["text"], w["font"])
    ps = (plan.get("plan") or {}).get("piece_system") or {}
    assert all(fl.is_liked(f) for f in ps.get("fonts", []))
