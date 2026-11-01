"""`/mograph/fonts/*`: the liked fonts for the storyboard page (16_app/MEDIA_BOX.md "STORYBOARD v2": click through / shuffle the fonts, words shown over the images in the chosen font).

    GET /mograph/fonts/liked          -> [{file, key, name, family_css, role, weight, style, variable, axes, risky, url}] from mograph/fonts/liked.json (the HARD liked-fonts allowlist)
    GET /mograph/fonts/file/<file>    -> the font file (font/ttf | font/otf | font/woff2); whitelist = liked.json files only, anything else is 404

``risky`` = the face is in design_defaults.MAIN_HOLD_RISKY (dotted / thin / outline-only / very tall-thin: hard to read small); the page never gives such a face as a RANDOM default, because a
default goes into the brief as the user's own choice (font_pins) and the legibility rules then keep it. ``family_css`` is a safe CSS font-family name ("mg-<alnum key>") for a `@font-face` / FontFace the page builds from ``url``. ``weight`` is the fixed wght of a liked variable
instance (liked.json ``axes``, e.g. Foldit thin = 100; ``variation`` = all of those axes for font-variation-settings) else null (the face's own default); ``variable`` = the file has axes (its name carries ``[wght,...]``).

The helpers are plain Python (no aiohttp / ComfyUI import at module level): tests call them directly; ``register(routes)`` wires them into aiohttp.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "mograph" / "fonts"
CONTENT_TYPES = {".ttf": "font/ttf", ".otf": "font/otf", ".woff2": "font/woff2", ".woff": "font/woff"}
CACHE = {"Cache-Control": "public, max-age=86400"}


def _liked():
    from mograph import fonts_liked as fl          # the ONE liked.json helper
    return fl.LIKED


def css_family(key: str) -> str:
    return "mg-" + (re.sub(r"[^A-Za-z0-9]", "", str(key)) or "font")


# fallback when tools.fusion.design_defaults does not import (same names as its MAIN_HOLD_RISKY = HARD_READ_FONTS + the round-3 QC faces)
_HARD_READ_FALLBACK = ("Monofett", "RubikBrokenFax", "RubikGlitchPop", "RubikLines", "LondrinaSketch", "BungeeOutline", "BungeeHairline")
_RISKY_FALLBACK = _HARD_READ_FALLBACK + ("Doto", "Codystar", "FolditThin", "Interlope", "NixieOne", "FacadeVF")


def _risky_names() -> tuple:
    try:
        from tools.fusion import design_defaults as dd          # READ-ONLY
        names = tuple(dd.MAIN_HOLD_RISKY)
        if names:
            return names
    except Exception:
        pass
    return _RISKY_FALLBACK


def is_risky(entry: dict, names: tuple | None = None) -> bool:
    """design_defaults compares e["key"] to MAIN_HOLD_RISKY exactly; the family name (spaces / punctuation dropped) is matched the same way."""
    names = _risky_names() if names is None else names
    fam = re.sub(r"[^A-Za-z0-9]", "", str(entry.get("family") or ""))
    return str(entry.get("key")) in names or (bool(fam) and fam in names)


def liked_list() -> list[dict]:
    out = []
    risky = _risky_names()
    for e in _liked():
        f = e["file"]
        m = re.search(r"\[([^\]]+)\]", f)
        axes = (m.group(1).split(",") if m else [])
        w = (e.get("axes") or {}).get("wght")
        out.append({"file": f, "key": e["key"], "name": e["family"], "family_css": css_family(e["key"]), "role": e["role"],
                    "weight": int(w) if w is not None else None, "variation": dict(e.get("axes") or {}), "style": "normal", "variable": bool(axes), "axes": axes,
                    "risky": is_risky(e, risky), "url": "/mograph/fonts/file/" + re.sub(r"[^A-Za-z0-9._-]", lambda mm: "%%%02X" % ord(mm.group(0)), f)})
    return out


def font_path(name: str, base: Path | None = None) -> Path | None:
    """the file of a liked font by its exact file name, else None (not liked, missing, or a path trick)."""
    name = str(name or "")
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    if name not in {e["file"] for e in _liked()}:
        return None
    p = (base or FONT_DIR) / name
    return p if p.is_file() else None


def content_type(path) -> str:
    return CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def register(routes) -> None:
    from aiohttp import web

    @routes.get("/mograph/fonts/liked")
    async def mograph_fonts_liked(request):
        return web.json_response(liked_list(), headers={"Cache-Control": "no-cache, must-revalidate"})

    @routes.get("/mograph/fonts/file/{file}")
    async def mograph_font_file(request):
        p = font_path(request.match_info["file"])
        if p is None:
            return web.Response(status=404, text="not a liked font")
        return web.FileResponse(p, headers={**CACHE, "Content-Type": content_type(p), "Access-Control-Allow-Origin": "*"})
