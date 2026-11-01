"""User 2026-10-30: "always use our favourited fonts, not generic ones". (1) No source file of the pipeline falls back to PIL's default font, DejaVu, Arial, Liberation or a system font directory;
(2) a rendered type pass only opens font files that are in mograph/fonts/liked.json; (3) the hero word wears a characterful liked face. Run: python -m pytest tests/test_fonts_never_generic.py -q"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from mograph import fonts_liked as fl  # noqa: E402

BANNED = re.compile(r"load_default\(|DejaVu|dejavu|\bArial\b|\barial\b|Liberation|/usr/share/fonts|Windows[/\\]Fonts|C:\\Windows")
SCAN = ("mograph", "nodes_fusion", "tools/fusion", "tools/app", "tools/mathtype", "nodes.py", "app_routes.py")
ALLOWED_FILES = {"mograph/fonts_liked.py"}            # the helper itself names what it forbids in its docstring


def _py_files():
    for rel in SCAN:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            yield rel, p
            continue
        for d, _dirs, files in os.walk(p):
            if "__pycache__" in d:
                continue
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(d, f)
                    yield os.path.relpath(full, ROOT).replace(os.sep, "/"), full


def test_no_generic_font_fallback_anywhere_in_the_source():
    bad = []
    for rel, full in _py_files():
        if rel in ALLOWED_FILES or rel.endswith("/test_fonts_never_generic.py"):
            continue
        for i, ln in enumerate(open(full, encoding="utf-8", errors="replace").read().splitlines(), 1):
            if BANNED.search(ln) and not ln.lstrip().startswith("#") and "font-alias-ok" not in ln:      # the alias map names what a model may WRITE; every value is a liked face
                bad.append(f"{rel}:{i}: {ln.strip()[:100]}")
    assert not bad, "generic / system font reference outside the liked-font helper:\n" + "\n".join(bad)


def test_the_helper_raises_for_anything_not_liked_and_resolves_liked_names():
    assert os.path.isfile(fl.font_file("Rock 3D")) and os.path.basename(fl.font_file("Honk")) == "Honk[MORF,SHLN].ttf"
    with pytest.raises(fl.FontNotLiked):
        fl.font_file("Comic Sans")
    with pytest.raises(fl.FontNotLiked):
        fl.assert_liked_path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    with pytest.raises(fl.FontNotLiked):
        from mograph.render import text as T
        T.get_font("/tmp/Arial.ttf", 40)


def test_a_rendered_type_pass_only_opens_liked_font_files(tmp_path, monkeypatch):
    from PIL import ImageFont
    import test_layout_preview as TLP
    from tools.fusion import layout_preview as LP
    seen = []
    real = ImageFont.truetype

    def rec(font=None, *a, **k):
        seen.append(str(font))
        return real(font, *a, **k)
    monkeypatch.setattr(ImageFont, "truetype", rec)
    p, info = LP.redraw_piece(TLP.piece(), 21, TLP.stills())
    spec = LP.type_spec(p)
    LP.render_frames([dict(spec=spec, frame=40)], str(tmp_path), LP.TYPE_SIZE, procs=0)
    files = {os.path.basename(s.split("|", 1)[0]) for s in seen if s and s != "None"}
    assert files, "the render opened no font"
    assert all(fl.is_liked(f) for f in files), sorted(f for f in files if not fl.is_liked(f))


def test_the_hero_word_wears_a_characterful_liked_face():
    from tools.fusion import design_defaults as DD
    words = [{"text": "FIRST", "font": "Ultra-Regular.ttf", "hold_s": 3.0}, {"text": "SECOND", "font": "Rock3D-Regular.ttf", "hold_s": 1.0}, {"text": "THIRD", "font": "Honk[MORF,SHLN].ttf", "hold_s": 1.5}]
    fixes: list = []
    assert DD.hero_face(words, fixes) == 1
    assert fl.role_of(words[0]["font"]) in ("pixel", "ornamental", "decon", "blackletter") and words[1]["font"] == "Ultra-Regular.ttf"      # the shortest-holding expressive word took the reader face
    assert DD.hero_face(words, []) == 0                                                                                                     # idempotent
    win = [{"text": "A", "font": "Ultra-Regular.ttf", "hold_s": 3.0}, {"text": "B", "font": "Rock3D-Regular.ttf", "hold_s": 1.0}]
    assert DD.hero_face(win, [], protected={0}) == 0                                                                                        # a window word keeps its fat face
