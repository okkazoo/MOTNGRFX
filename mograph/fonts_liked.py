"""Hard allowlist of fonts: only the fonts the user liked (mograph/fonts/liked.json, 42 from the fonts review artifact).

``require_liked(name)`` accepts a liked key ("FolditThin"), a family ("Big Shoulders Display"), a font file name
("BigShouldersDisplay[wght].ttf", "Silkscreen-Bold.ttf") or a path to one, ignoring case, spaces, '-', '_' and the file
extension, and returns the liked.json entry. A face that is NOT liked (Gemma's reply, a module default, a recipe, an old piece JSON) is
SUBSTITUTED by the closest liked face of the same role (``substitute``: deterministic per name, one logged warning per name) so a render never
crashes and never draws a generic / system face. STRICT mode (env ``MOGRAPH_FONT_STRICT=1`` or ``strict=True``) RAISES FontNotLiked instead;
tests/conftest.py turns it on, so a stray non-liked name in code still fails CI. The module resolver
(mograph.modules.core.resolve_font) calls it, so the planner, hand specs (piece_runner), the type passes and every word
module go through it. To allow a new font, add it to liked.json (and fonts.json) explicitly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import zlib

LIKED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "liked.json")
with open(LIKED_PATH, encoding="utf-8") as _f:
    _DOC = json.load(_f)

LIKED: list[dict] = list(_DOC["fonts"])
ROLES = ("tall", "heavy", "fine", "mono", "ornamental", "pixel", "blackletter", "decon", "text")

# role -> (weight class, fill) used to judge contrast between consecutive lines; roles with the same pair also
# need a scale difference (see contrast_axes).
_ROLE_STYLE = {"tall": ("bold", "solid"), "heavy": ("heavy", "solid"), "fine": ("light", "outline"),
               "mono": ("light", "solid"), "ornamental": ("heavy", "outline"), "pixel": ("bold", "solid"),
               "blackletter": ("bold", "solid"), "decon": ("light", "outline"), "text": ("bold", "solid")}


class FontNotLiked(KeyError):
    """raised for any font that is not in mograph/fonts/liked.json."""

    def __str__(self):
        return str(self.args[0]) if self.args else "font is not liked"


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _forms(name) -> list[str]:
    """normalised spellings of a font name / file name, most specific first."""
    s = os.path.basename(str(name).replace("\\", "/"))
    if s.lower().endswith((".ttf", ".otf")):
        s = s[:-4]
    out = [_alnum(s), _alnum(re.split(r"\[", s)[0]), _alnum(re.split(r"[\[\-_]", s)[0])]
    return [o for i, o in enumerate(out) if o and o not in out[:i]]


_INDEX: dict[str, dict] = {}
for _e in LIKED:
    for _n in (_e["key"], _e["family"], _e["file"]):
        for _form in _forms(_n):
            _INDEX.setdefault(_form, _e)
BY_KEY = {e["key"]: e for e in LIKED}
BY_FAMILY = {e["family"]: e for e in LIKED}
FILES = {e["file"] for e in LIKED}


def lookup(name) -> dict | None:
    """liked.json entry for a key / family / file name, or None."""
    if isinstance(name, dict):
        name = name.get("key") or name.get("name") or name.get("file") or ""
    for f in _forms(name):
        if f in _INDEX:
            return _INDEX[f]
    return None


def is_liked(name) -> bool:
    return lookup(name) is not None


STRICT_ENV = "MOGRAPH_FONT_STRICT"
log = logging.getLogger("mograph.fonts_liked")
_WARNED: set[str] = set()


def is_strict(strict=None) -> bool:
    """True when a non-liked face must RAISE (``strict`` arg wins, else env MOGRAPH_FONT_STRICT=1)."""
    if strict is not None:
        return bool(strict)
    return os.environ.get(STRICT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# generic / old default faces (alnum names) -> the role of the liked face that replaces them
_GENERIC_ROLE = {
    "anton": "tall", "bebasneue": "tall", "leaguegothic": "tall", "oswald": "tall", "impact": "tall", "unbounded": "heavy",
    "inter": "text", "interbold": "text", "interregular": "text", "instrumentsans": "text", "rubik": "text", "bricolagegrotesque": "text",
    "syne": "text", "arial": "text", "helvetica": "text", "roboto": "text", "dejavusans": "text", "dejavu": "text", "liberation": "text",
    "playfairdisplay": "fine", "bodonimoda": "fine", "fraunces": "fine", "dmserifdisplay": "fine", "instrumentserif": "fine",
    "instrumentserifitalic": "fine", "georgia": "fine", "times": "fine", "timesnewroman": "fine", "cantique": "fine", "lemurmure": "fine",
    "jetbrainsmono": "mono", "spacemono": "mono", "couriernew": "mono", "courier": "mono", "consolas": "mono",
    "bungee": "heavy", "rubikmonoone": "heavy", "climatecrisis": "heavy", "shrikhand": "heavy", "spratextended": "heavy",
    "bungeeshade": "decon", "monoton": "ornamental", "pilowlava": "ornamental", "kaerukaeru": "ornamental", "ouroboros": "ornamental",
    "trickster": "ornamental", "flordruina": "decon", "cirruscumulus": "ornamental", "sinistre": "blackletter",
    "maziusdisplayitalic": "fine", "spratcondensedthin": "tall",
}
_KEYWORD_ROLE = (("blackletter", "blackletter"), ("fraktur", "blackletter"), ("pixel", "pixel"), ("bitmap", "pixel"), ("8bit", "pixel"),
                 ("mono", "mono"), ("courier", "mono"), ("code", "mono"), ("outline", "decon"), ("inline", "decon"), ("shade", "decon"),
                 ("stencil", "decon"), ("sketch", "decon"), ("hairline", "fine"), ("thin", "fine"), ("light", "fine"), ("serif", "fine"),
                 ("condensed", "tall"), ("narrow", "tall"), ("gothic", "tall"), ("sans", "text"), ("grotesk", "text"), ("dejavu", "text"),
                 ("liberation", "text"), ("arial", "text"), ("helvet", "text"), ("black", "heavy"), ("heavy", "heavy"), ("bold", "heavy"),
                 ("ultra", "heavy"), ("extra", "heavy"), ("deco", "ornamental"), ("neon", "ornamental"), ("orna", "ornamental"))


# legible stand-ins first (a barcode face or a lined face is a liked FACE but a poor stand-in for a serif / mono request)
_PREFER = {"fine": ("Interlope", "NixieOne"), "mono": ("MajorMonoDisplay",), "tall": ("BigShouldersDisplay", "FacadeVF"), "text": ("SpaceGrotesk-Bold",)}


def guess_role(name) -> str:
    """the role (one of ROLES) of a face that is NOT liked: known generic names first, then name keywords, then the font registry's width / role tags, else 'heavy'."""
    if isinstance(name, dict):
        name = name.get("key") or name.get("name") or name.get("file") or ""
    base = os.path.basename(str(name).replace("\\", "/"))
    if base.lower().endswith((".ttf", ".otf")):
        base = base[:-4]
    forms = [_alnum(base), _alnum(re.split(r"[\[\-_]", base)[0])]
    for f in forms:
        if f in _GENERIC_ROLE:
            return _GENERIC_ROLE[f]
    for f in forms:
        for k, r in _KEYWORD_ROLE:
            if k in f:
                return r
    try:
        from mograph import font_registry as fr
        reg = fr.REGISTRY.get(str(name)) or next((e for n, e in fr.REGISTRY.items() if _alnum(n) in forms), None)
    except Exception:
        reg = None
    if reg:
        if reg.get("width") in ("condensed", "narrow"):
            return "tall"
        if "text" in reg.get("role", ()) and "display" not in reg.get("role", ()):
            return "text"
    return "heavy"


def substitute(name) -> dict:
    """the liked.json entry that stands in for a NOT-liked face: same role (``guess_role``), picked deterministically from the name, so the same request always gets the same face.
    Logs one warning line per distinct name."""
    role = guess_role(name)
    pool = [e for e in LIKED if e["role"] == role] or [e for e in LIKED if e["role"] == "heavy"] or LIKED
    pool = sorted(pool, key=lambda e: e["key"])
    pool = [e for e in pool if e["key"] in _PREFER.get(role, ())] or pool
    sig = _alnum(os.path.basename(str(name).replace("\\", "/"))) or "none"
    e = pool[zlib.crc32(sig.encode()) % len(pool)]
    if sig not in _WARNED:
        _WARNED.add(sig)
        log.warning("font %r is not liked -> using liked %s face %r instead (mograph/fonts/liked.json)", str(name), role, e["family"])
    return e


def require_liked(name, strict=None) -> dict:
    """the liked.json entry for ``name``. A face that is not liked is substituted by the closest liked face of the same role (one warning logged);
    in strict mode (``strict=True`` or env MOGRAPH_FONT_STRICT=1) it raises FontNotLiked (a KeyError) instead."""
    e = lookup(name)
    if e is not None:
        return e
    if not is_strict(strict):
        return substitute(name)
    raise FontNotLiked(f"font {name!r} is not in the liked-font allowlist (mograph/fonts/liked.json, "
                       f"{len(LIKED)} fonts); add it there explicitly to allow it. liked: "
                       f"{', '.join(sorted(BY_FAMILY))}")


def role_of(name, strict=None) -> str:
    return require_liked(name, strict)["role"]


def families(role: str | None = None) -> list[str]:
    return [e["family"] for e in LIKED if role is None or e["role"] == role]


def contrast_axes(role_a: str, role_b: str) -> set[str]:
    """which of 'weight' / 'fill' differ between two roles (scale is decided by the planner)."""
    a, b = _ROLE_STYLE.get(role_a, ("bold", "solid")), _ROLE_STYLE.get(role_b, ("bold", "solid"))
    return {n for n, x, y in (("weight", a[0], b[0]), ("fill", a[1], b[1])) if x != y}


# ----------------------------------------------------------------------------------------------------------- THE ONE PLACE a font file is resolved (user 2026-10-30: "always use our favourited fonts, not generic ones")
def font_file(name, strict=None) -> str:
    """absolute path of the liked face ``name`` (key / family / file name, case and punctuation forgiving). A face that is not in liked.json gets the closest liked face of the same
    role (``substitute``); strict mode raises FontNotLiked. FileNotFoundError when the file is not in mograph/fonts: there is NO generic fallback (no PIL default, no DejaVu / Arial /
    system font) anywhere in the pipeline."""
    e = require_liked(name, strict)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", e["file"])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"liked font file missing: {p}")
    return p


def assert_liked_path(path, strict=None) -> str:
    """``path`` (a font file path, with or without a '|axes' suffix) -> a path of a LIKED file: unchanged when it already is one; otherwise the substitute liked file of the same role
    (axes dropped, they belong to the other face), or FontNotLiked in strict mode."""
    base = os.path.basename(str(path).split("|", 1)[0])
    if lookup(base) is not None:
        return str(path)
    return font_file(base, strict)


def truetype(name, size, strict=None, **kw):
    """PIL ImageFont of the liked face ``name`` (the only way the pipeline opens a font by name)."""
    from PIL import ImageFont
    return ImageFont.truetype(font_file(name, strict), size, **kw)
