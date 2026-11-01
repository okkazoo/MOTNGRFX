"""User 2026-10-30: "the font choices I made are important ... never fall back to generic fonts". A face that is NOT in mograph/fonts/liked.json (a Gemma reply, an old piece JSON, a module default)
is replaced by the closest liked face of the SAME ROLE and one warning is logged; it never crashes a render and never yields a generic / system face. Strict mode (MOGRAPH_FONT_STRICT=1, which
tests/conftest.py sets for the whole suite, or strict=True) raises FontNotLiked instead. Run: python -m pytest tests/test_font_substitution.py -q"""
from __future__ import annotations

import logging
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph import fonts_liked as fl  # noqa: E402

# (a not-liked name, the role its stand-in must have)
CASES = [("Anton", "tall"), ("Bebas Neue", "tall"), ("League Gothic", "tall"), ("Inter", "text"), ("Arial", "text"), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "text"),
         ("Playfair Display", "fine"), ("Bodoni Moda", "fine"), ("JetBrains Mono", "mono"), ("Space Mono", "mono"), ("Bungee", "heavy"), ("Climate Crisis", "heavy"),
         ("Monoton", "ornamental"), ("Bungee Shade", "decon"), ("Some Pixel Face", "pixel"), ("My Blackletter", "blackletter")]


@pytest.mark.parametrize("name,role", CASES)
def test_a_non_liked_name_gets_a_liked_face_of_the_same_role(name, role, caplog):
    assert fl.lookup(name) is None
    with caplog.at_level(logging.WARNING, logger="mograph.fonts_liked"):
        e = fl.require_liked(name, strict=False)
    assert e["role"] == role and fl.is_liked(e["file"])
    assert fl.require_liked(name, strict=False) is e or fl.require_liked(name, strict=False) == e          # deterministic: same request, same face
    assert os.path.isfile(fl.font_file(name, strict=False))


def test_it_logs_one_warning_line_per_name(caplog):
    fl._WARNED.discard("zzunlikedface")
    with caplog.at_level(logging.WARNING, logger="mograph.fonts_liked"):
        for _ in range(3):
            fl.require_liked("ZZ Unliked Face", strict=False)
    assert len([r for r in caplog.records if "ZZ Unliked Face" in r.getMessage()]) == 1


def test_no_code_path_returns_a_path_outside_the_liked_set(monkeypatch):
    names = [n for n, _ in CASES] + ["Comic Sans", "", "None", "C:\\Windows\\Fonts\\arial.ttf", "Impact", "Anton-Regular.ttf|wght=900", "x/y/Inter-Bold.ttf"]
    monkeypatch.delenv(fl.STRICT_ENV, raising=False)
    liked_files = {e["file"] for e in fl.LIKED}
    for n in names:
        assert os.path.basename(fl.font_file(n)) in liked_files, n
        assert os.path.basename(fl.assert_liked_path(n).split("|", 1)[0]) in liked_files, n
        assert os.path.basename(fl.truetype(n, 40).path) in liked_files, n
        from mograph.modules import core
        key, _loc = core.resolve_font(n)
        assert os.path.basename(core.ol.font_path(key)) in liked_files, (n, key)
    from mograph.render import text as T
    for n in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/tmp/Anton-Regular.ttf|wght=900", os.path.join(ROOT, "mograph", "fonts", "Anton-Regular.ttf")):
        assert os.path.basename(T.get_font(n, 40).path) in liked_files, n
    from mograph import font_registry as fr
    for seed in range(40):
        d, c = fr.pick_fonts(["hype"], 0.5, seed)
        assert fl.is_liked(d) and fl.is_liked(c), (d, c)
        d, c = fr.pick_fonts(["nonexistent_mood"], 0.5, seed)
        assert fl.is_liked(d) and fl.is_liked(c), (d, c)


def test_strict_mode_raises(monkeypatch):
    with pytest.raises(fl.FontNotLiked):
        fl.require_liked("Anton", strict=True)
    with pytest.raises(fl.FontNotLiked):
        fl.font_file("Comic Sans", strict=True)
    with pytest.raises(fl.FontNotLiked):
        fl.assert_liked_path("/usr/share/fonts/DejaVuSans.ttf", strict=True)
    monkeypatch.setenv(fl.STRICT_ENV, "1")                     # the env switch (what the suite runs under)
    with pytest.raises(fl.FontNotLiked):
        fl.require_liked("Inter")
    from mograph.modules import core
    with pytest.raises(fl.FontNotLiked):
        core.resolve_font("Anton")
    with pytest.raises(fl.FontNotLiked):
        from mograph.render import text as T
        T.get_font("/tmp/Arial.ttf", 40)
    assert fl.require_liked("Rock 3D")["key"] == "Rock3D"      # liked names still resolve in strict mode
    monkeypatch.setenv(fl.STRICT_ENV, "0")
    assert fl.require_liked("Anton")["role"] == "tall"


def test_the_models_are_only_offered_liked_faces():
    """Gemma's reply schema (grammar enums), the design-step font list and the legacy plan guide name only fonts from liked.json."""
    from tools.fusion import qwen_piece as QP
    from tools.fusion import reply_schema as RSCH
    from mograph import font_registry as fr
    liked_files = {e["file"] for e in fl.LIKED}
    assert {f["file"] for f in QP.liked_fonts()} == liked_files
    ws = RSCH.word_schema("WORD")
    assert set(ws["properties"]["font"]["enum"]) == liked_files
    guide = fr.prompt_guide()
    for name in fr.REGISTRY:
        if not fl.is_liked(name):
            assert name + " (" not in guide, name
    assert "Anton (" not in guide and "Inter (" not in guide
