"""Bundled font registry (mograph/fonts/fonts.json): per-font file, variation defaults, mood / role /
width / contrast tags and pairings, plus a deterministic display + companion picker for planners.

Plan font names are the registry keys. ``font_key`` encodes a file plus variation-axis values as one
string ("<path>|wght=800,wdth=100") so the renderer can cache one FreeType instance per variation.
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
with open(os.path.join(FONT_DIR, "fonts.json"), encoding="utf-8") as _f:
    _DATA = json.load(_f)

REGISTRY: dict[str, dict] = _DATA["fonts"]
FONT_NAMES: list[str] = list(REGISTRY)
MOOD_TAGS: list[str] = list(_DATA["moods"])

# OpenType axis tag -> the axis name FreeType / PIL reports (lower case)
AXIS_NAMES = {"wght": "weight", "wdth": "width", "opsz": "optical size", "year": "year", "soft": "softness",
              "wonk": "wonky", "slnt": "slant", "ital": "italic", "ctst": "contrast",
              # custom axes (verified: ELXP/ELSH/CRSV Bitcount, ROND Doto, MORF/SHLN Honk; the last six are
              # unverified names for faces not bundled yet)
              "elxp": "element expansion", "elsh": "element shape", "crsv": "cursive", "rond": "roundness",
              "morf": "morph", "shln": "shadow", "bled": "bleed", "scan": "scanlines", "xrot": "x rotation",
              "yrot": "y rotation", "edpt": "edge depth", "ehlt": "edge highlight"}


def font_path(name: str, bold: bool = False) -> str:
    e = REGISTRY[name]
    return os.path.join(FONT_DIR, e["bold_file"] if bold and e.get("bold_file") else e["file"])


def font_axes(name: str, bold: bool = False, extra: dict | None = None) -> dict:
    """Variation values for a use of ``name``: registry defaults, + bold_axes, + per-layer overrides."""
    e = REGISTRY.get(name) or {}
    ax = dict(e.get("axes") or {})
    if bold:
        ax.update(e.get("bold_axes") or {})
    if extra:
        ax.update({str(k): float(v) for k, v in extra.items()})
    return ax


def font_key(path: str, axes: dict | None = None) -> str:
    if not axes:
        return path
    return path + "|" + ",".join(f"{k}={float(v):g}" for k, v in sorted(axes.items()))


def split_key(key: str) -> tuple[str, dict]:
    path, _, ax = key.partition("|")
    axes = {}
    for kv in filter(None, ax.split(",")):
        k, _, v = kv.partition("=")
        axes[k] = float(v)
    return path, axes


def leading(name: str, default: float = 1.12) -> float:
    return float((REGISTRY.get(name) or {}).get("leading", default))


def preferred_case(name: str) -> str:
    return str((REGISTRY.get(name) or {}).get("case", "upper"))


def is_variable(name: str) -> bool:
    return bool((REGISTRY.get(name) or {}).get("vary"))


def is_rare(name: str) -> bool:
    return (REGISTRY.get(name) or {}).get("rarity") == "rare"


def foils(name: str) -> list[str]:
    """Contrasting display faces to set against ``name`` (puffy <-> fine / tall), best first."""
    return [f for f in (REGISTRY.get(name) or {}).get("foil", ()) if f in REGISTRY]


_REQ_W = (1.0, 0.45, 0.3)
_TAG_W = (1.0, 0.8, 0.6, 0.5)
RARE_BOOST = 20.0         # weight multiplier for rarity "rare" display faces when prefer_rare


def _is_liked(name: str) -> bool:
    from mograph import fonts_liked as _fl
    return _fl.is_liked(name)


def pick_fonts(moods: list[str] | tuple[str, ...], energy: float, seed: int, salt: str = "",
               prefer_rare: bool = True, allowed=None) -> tuple[str, str]:
    """Display + companion for registry moods (first = strongest), deterministic per (seed, salt).

    A display's weight is (request-mood rank weight) x (where that mood sits in the font's own tags); a face
    outside its energy range is down-weighted by distance. With ``prefer_rare`` (default) the rare faces
    (independent libre foundries) get RARE_BOOST x weight, so they lead the display role and the common
    Google Fonts faces remain as fallbacks / occasional picks. The companion is drawn from the display's
    text-role pairings (first favoured). ``allowed`` (registry names) restricts the display draw; with it the
    fallback is drawn from ``allowed`` too; with ``allowed=None`` the pool is the liked faces and the companion is a liked text face."""
    if allowed is None:                                    # favourite fonts only (user 2026-10-30): the default pool is liked.json, never the whole registry
        from mograph import fonts_liked as _fl
        allowed = {n for n in REGISTRY if _fl.is_liked(n)}
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(("fonts:" + salt).encode())])
    moods = [m for m in moods if m in MOOD_TAGS] or ["hype"]
    e = float(np.clip(energy, 0.0, 1.0))
    names, weights = [], []
    for name, f in REGISTRY.items():
        if "display" not in f.get("role", ()) or (allowed is not None and name not in allowed):
            continue
        hits = [k for k, m in enumerate(moods) if m in f.get("mood", ())]
        if not hits:
            continue
        # request side: first mood 1, later moods 0.45, 0.3; font side: its first tag 1, then 0.8, 0.6, 0.5
        w = max(_REQ_W[min(k, len(_REQ_W) - 1)] * _TAG_W[min(f["mood"].index(moods[k]), len(_TAG_W) - 1)]
                for k in hits)
        lo, hi = f.get("energy", [0.0, 1.0])
        d = max(0.0, lo - e, e - hi)
        w *= max(0.1, 1.0 - 3.0 * d)
        if prefer_rare and f.get("rarity") == "rare":
            w *= RARE_BOOST
        names.append(name)
        weights.append(w)
    if not names and allowed is not None:                  # no mood hit among the allowed faces: any allowed display face
        names = [n for n, f in REGISTRY.items() if n in allowed and "display" in f.get("role", ())]
        if not names:
            raise ValueError("pick_fonts: no allowed display face")
        weights = [1.0] * len(names)
    if not names:
        return "Big Shoulders Display", "Space Grotesk"
    p = np.asarray(weights, np.float64)
    display = names[int(rng.choice(len(names), p=p / p.sum()))]
    comp = [c for c in REGISTRY[display].get("pairs", ()) if c in REGISTRY and "text" in REGISTRY[c].get("role", ()) and _is_liked(c)]
    if not comp:
        return display, "Space Grotesk"
    cw = np.array([1.0 / (k + 1) ** 1.2 for k in range(len(comp))])
    return display, comp[int(rng.choice(len(comp), p=cw / cw.sum()))]


def prompt_guide() -> str:
    """One compact line per font for the LLM schema prompt: name (roles; moods; width, contrast), rare faces
    marked with their shape, plus a one-line hint to prefer them."""
    out = []
    for name, f in REGISTRY.items():
        if not _is_liked(name):                       # favourite fonts only: a model is never offered a face outside liked.json
            continue
        vary = f"; animates {'/'.join(f['vary'])}" if f.get("vary") else ""
        rare = f"; rare {f.get('shape', '')}".rstrip() if f.get("rarity") == "rare" else ""
        wd = "" if f["width"] == "normal" else f"; {f['width']}"      # "normal" is the default: omit (word budget)
        out.append(f"{name} ({'/'.join(f['role'])}; {'/'.join(f['mood'][:3])}{wd}{vary}{rare})")
    return ("; ".join(out) + ". Prefer a 'rare' face as display; set a puffy face against a fine or tall one "
            "(huge vs small) for contrast")
