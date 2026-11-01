"""Fusion brief: the ~200-token JSON a VLM (design_llm_8b on the box) writes for the app's piece planner.

Rules write the piece spec (``tools/fusion/plan_rules.py``); the model only writes the creative text: scene, still prompt, one
motion line per shot, music tags and tempo, a technique, a mood, palette hints and a signoff line. Every field is optional:
``parse_brief`` never raises, drops what it cannot use (logged) and the planner's rules fill the gap, so empty or broken model
output gives the same piece as no brief at all.

``brief_prompt`` follows ``mograph.brief.brief_prompt``: field descriptions and enumerations only, no example values to copy,
enumeration order shuffled per seed.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from .brief import MOODS, _hex, _moods, _norm_key, _num, _salt, auto_lines, display_text, order_palette

__all__ = ["TECHNIQUES", "TECH_TEXT", "MAX_SHOTS", "SHOT_FRAMES", "XFADE", "shot_count", "words_lines", "brief_prompt",
           "parse_brief", "empty_brief", "brief_is_empty", "DIRECTION", "PACE_TIER", "DIRECTION_PALETTES", "MAX_BLOCKS", "block_names"]

MAX_SHOTS = 6
SHOT_FRAMES = 121          # LTX frames per shot (8k+1, proven size)
XFADE = 12                 # crossfade frames between shots
FPS = 24

TECHNIQUES = ["built_in", "behind", "traced"]
TECH_TEXT = {
    "built_in": "the letters are built into the scene, solid objects in the same world",
    "behind": "a clear subject stands in front of the letters, the letters sit behind it in mist, glass or smoke",
    "traced": "the letters are drawn as flowing lines, ribbons or contours that the footage follows",
}
_TECH_SYN = {"builtin": "built_in", "built": "built_in", "in_scene": "built_in", "inscene": "built_in", "inside": "built_in",
             "embedded": "built_in", "solid": "built_in", "depth": "built_in", "behind_subject": "behind",
             "occluded": "behind", "sandwich": "behind", "layered": "behind", "trace": "traced", "tracing": "traced",
             "lines": "traced", "line": "traced", "contour": "traced", "outline": "traced", "drawn": "traced"}


# --------------------------------------------------------------------------
# lego blocks + direction (v005): the brief may NAME up to MAX_BLOCKS blocks (tools/fusion/blocks.py) and give a direction; rules fill the rest
# --------------------------------------------------------------------------
MAX_BLOCKS = 2
# direction fields and their enumerations (tests/test_plan_blocks.py checks they match tools/fusion/blocks.py and the camera registry)
DIRECTION = {
    "pace": ["slow", "mid", "driving", "building", "flowing"],
    "palette": ["near_black_one_light", "high_key_one_accent", "flat_primaries", "duotone_warm", "silver_bw", "dark_neon"],
    "type_role": ["massive", "solid_3d", "flat_strips", "from_footage", "small_precise", "window"],
    "camera": ["follow", "pullback_reveal", "hold_drift", "orbit_even", "spin_read_spin", "push_through", "tilt_to_plan", "graze", "ease_reversal"],
    "shots": ["single", "cut", "chain"],
}
PACE_TIER = {"slow": "calm", "flowing": "calm", "mid": "mid", "driving": "kinetic", "building": "kinetic"}
DIRECTION_PALETTES = {                      # [near-black, near-white, accent, accent, accent] (plan_rules.palette_for orders and pads them)
    "near_black_one_light": ["#05090d", "#d8f3f7", "#58d6e6", "#2a8da0", "#9fe9f2"],
    "high_key_one_accent": ["#26282b", "#f6f7f8", "#1f4fd8", "#5a6f9e", "#8fa4d8"],
    "flat_primaries": ["#14110f", "#efe7d8", "#d6281e", "#1e4fc8", "#f2c230"],
    "duotone_warm": ["#140806", "#ffe3b8", "#f08a24", "#a8321c", "#ffb85c"],
    "silver_bw": ["#050505", "#f2f2f2", "#c9ccd0", "#8e9298", "#e4e6e9"],
    "dark_neon": ["#07060d", "#f2f0ff", "#ff2bd6", "#27e0ff", "#9d5cff"],
}
_DIR_SYN = {"calm": "slow", "weightless": "slow", "steady": "mid", "medium": "mid", "rapid": "driving", "fast": "driving",
            "intense": "building", "open": "flowing", "flow": "flowing", "strips": "flat_strips", "flat": "flat_strips", "3d": "solid_3d",
            "small": "small_precise", "precise": "small_precise", "sparks": "from_footage", "footage": "from_footage", "huge": "massive",
            "giant": "massive", "knockout": "window", "cuts": "cut", "one": "single", "chained": "chain"}


def block_names() -> list[str]:
    """names of the registered lego blocks (tools/fusion/blocks.py); [] when that package is not importable (mograph alone)."""
    try:
        from tools.fusion import blocks
        return list(blocks.NAMES)
    except Exception:       # noqa: BLE001
        return []


def shot_count(nframes: int, frames: int = SHOT_FRAMES, xf: int = XFADE, cap: int = MAX_SHOTS) -> int:
    """N = ceil((nframes - xf) / (frames - xf)), at least 1, at most ``cap``."""
    n = int(math.ceil((int(nframes) - xf) / float(frames - xf) - 1e-9))
    return max(1, min(cap, n))


def words_lines(words: str) -> str:
    """The user's words with " / " (and "|") as line breaks, directive lines ("Style: ...") removed."""
    t = re.sub(r"\s+[/|]\s+|\s*\|\s*", "\n", str(words or ""))
    return display_text(t)


def lines_of(words: str, max_words: int = 5) -> list[str]:
    return auto_lines(words_lines(words), max_words)


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def brief_prompt(request: dict, seed: int | None = None) -> str:
    """Instruction for the VLM from a MOGRAPH_REQUEST (words, scene, music, look, length, seed, image)."""
    req = request or {}
    seed = int(req.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(_salt(seed, "fusion-prompt"))

    def shuf(items):
        items = list(items)
        return [items[int(i)] for i in rng.permutation(len(items))]

    n = shot_count(int(req.get("length", 12) or 12) * FPS)
    words = words_lines(req.get("words", ""))
    scene = " ".join(str(req.get("scene") or "").split())
    music = " ".join(str(req.get("music") or "").split())
    has_img = req.get("image") is not None
    techs = "; ".join(f"{t} ({TECH_TEXT[t]})" for t in shuf(TECHNIQUES))
    moods = ", ".join(shuf(MOODS))
    wide = str(req.get("format") or "9:16") == "16:9"
    rows = [
        "You art-direct a short " + ("horizontal" if wide else "vertical") + " video where the user's words appear as large 3D type inside moving footage. "
        "Code does the timing, fonts and animation; you write only the creative text. Reply with one JSON object only.",
        "Keys:",
        "scene: one sentence, 15 to 30 words, describing the place and light of the footage"
        + (", matching the attached image" if has_img else "") + "; no text, letters or logos in it.",
        "still: a photographic image prompt for the first frame of that scene: subject, setting, light, lens, "
        "palette; " + ("wide horizontal" if wide else "tall vertical") + " frame; no text or letters.",
        f"motion: a list of exactly {n} short phrases (5 to 14 words), one per shot, in order: what moves in the footage "
        "and how the camera moves; each shot continues the one before.",
        "music: instrumental music tags, 8 to 16 comma-separated words (genre, instruments, texture, mood); no vocals.",
        "bpm: tempo, 60 to 150.",
        f"technique: one of {techs}.",
        f"mood: one word from {moods}.",
        "palette: 3 to 5 colours as \"#rrggbb\": a near-black, a near-white, then 1 to 3 vivid accents.",
        "signoff: a tiny closing caption of 2 to 6 lowercase words.",
    ]
    names = block_names()
    if names:
        from tools.fusion import blocks as _bk
        docs = dict(_bk.names_for_brief())
        rows.append(f"blocks: a list of at most {MAX_BLOCKS} building blocks that give the piece its main idea, from: "
                    + "; ".join(f"{n} ({docs[n]})" for n in shuf(names)) + ". Pick the ones that fit the words and scene; the first is the main one.")
    dm = DIRECTION
    rows.append("direction: an object with the keys pace (one of " + ", ".join(shuf(dm["pace"])) + "), palette (one of "
                + ", ".join(shuf(dm["palette"])) + "), type_role (one of " + ", ".join(shuf(dm["type_role"])) + "), camera (one of "
                + ", ".join(shuf(dm["camera"])) + "), shots (one of " + ", ".join(shuf(dm["shots"])) + "), music_genre (2 to 4 words), "
                "music_lead (the lead instrument, 1 to 3 words), bpm (60 to 150).")
    if scene:
        rows.append(f'USER SCENE (follow it): """{scene}"""')
    if music:
        rows.append(f'USER MUSIC (follow it): """{music}"""')
    look = str(req.get("look") or "auto")
    if look != "auto":
        rows.append(f"The type look is {look}; choose the technique and mood that suit it.")
    rows.append(f'USER WORDS: """{words.strip()}"""')
    return "\n".join(rows)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def empty_brief() -> dict:
    return {"scene": "", "still": "", "motions": [], "music": "", "bpm": None, "technique": None, "mood": [],
            "palette": [], "signoff": "", "blocks": [], "direction": {}}


def brief_is_empty(b: dict | None) -> bool:
    b = b or {}
    return not any(b.get(k) for k in ("scene", "still", "motions", "music", "bpm", "technique", "mood", "palette", "signoff", "blocks", "direction"))


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _text(v, limit: int) -> str:
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v if isinstance(x, (str, int, float)))
    if not isinstance(v, str):
        return ""
    v = " ".join(_CTRL.sub(" ", v).split()).strip(" \"'")
    return v[:limit].rstrip(" ,;")


def _motions(v, log: list) -> list[str]:
    if isinstance(v, str):
        v = [s for s in re.split(r"\n+|;\s*|\s*\|\s*|\s\d\.\s", v) if s.strip()]
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for it in v if isinstance(v, (list, tuple)) else []:
        if isinstance(it, dict):
            it = it.get("motion") or it.get("text") or it.get("prompt") or next(iter(it.values()), "")
        t = _text(re.sub(r"^\s*(?:shot\s*)?\d+\s*[:.)-]\s*", "", it) if isinstance(it, str) else it, 220)
        if t:
            out.append(t)
    if v and not out:
        log.append("motion: no usable phrase -> rules")
    return out[:MAX_SHOTS]


def _technique(v, log: list) -> str | None:
    k = _norm_key(v)
    if not k:
        return None
    if k in TECHNIQUES:
        return k
    if k in _TECH_SYN:
        log.append(f"technique: {v!r} -> {_TECH_SYN[k]!r}")
        return _TECH_SYN[k]
    for t in TECHNIQUES:                      # "built in", "Built-In (the letters ...)"
        if t in k or t.replace("_", "") in k.replace("_", ""):
            log.append(f"technique: {v!r} -> {t!r}")
            return t
    log.append(f"technique: {v!r} unknown -> rules")
    return None


def _enum(v, options: list, log: list, what: str):
    k = _norm_key(v)
    if not k:
        return None
    if k in options:
        return k
    if k in _DIR_SYN and _DIR_SYN[k] in options:
        log.append(f"{what}: {v!r} -> {_DIR_SYN[k]!r}")
        return _DIR_SYN[k]
    for o in options:
        if o in k or o.replace("_", "") in k.replace("_", ""):
            log.append(f"{what}: {v!r} -> {o!r}")
            return o
    log.append(f"{what}: {v!r} unknown -> rules")
    return None


def _blocks(v, log: list) -> list[str]:
    if isinstance(v, str):
        v = [x for x in re.split(r"[,;\n|]+", v) if x.strip()]
    if isinstance(v, dict):
        v = list(v.keys())
    names = block_names()
    out = []
    for it in v if isinstance(v, (list, tuple)) else []:
        if isinstance(it, dict):
            it = it.get("name") or it.get("block") or ""
        k = _norm_key(it) if isinstance(it, str) else ""
        if k in names and k not in out:
            out.append(k)
        elif k:
            hit = next((n for n in names if n.replace("_", "") == k.replace("_", "")), None)
            if hit and hit not in out:
                out.append(hit)
            else:
                log.append(f"blocks: {it!r} unknown -> dropped")
    if len(out) > MAX_BLOCKS:
        log.append(f"blocks: {len(out)} named, first {MAX_BLOCKS} kept")
    return out[:MAX_BLOCKS]


def _direction(v, log: list) -> dict:
    """the brief's direction object -> {pace, palette, type_role, camera, shots, music_genre, music_lead, bpm} with only the usable fields."""
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, opts in DIRECTION.items():
        e = _enum(v.get(k, v.get(k.replace("_", " "))), opts, log, f"direction.{k}")
        if e:
            out[k] = e
    for k in ("music_genre", "music_lead"):
        t = _text(v.get(k, v.get(k.replace("music_", ""))), 60)
        if t:
            out[k] = t
    bpm = _num(v.get("bpm", v.get("tempo")))
    if bpm is not None and bpm > 0:
        out["bpm"] = int(min(150, max(60, round(bpm))))
    return out


def parse_brief(text, request: dict | None = None, log: list | None = None) -> dict:
    """Raw model text (or an already parsed dict) -> normalised brief. Never raises; unusable fields stay empty."""
    from .llm_plan import loads_tolerant
    log = log if log is not None else []
    raw = None
    if isinstance(text, dict):
        raw = text
    elif isinstance(text, str) and text.strip():
        try:
            raw = loads_tolerant(text, log)
        except Exception as e:      # noqa: BLE001  PlanExtractError, RecursionError, anything: fall back to rules
            log.append(f"brief: no JSON object ({type(e).__name__}) -> rules only")
    if not isinstance(raw, dict):
        return empty_brief()
    if isinstance(raw.get("brief"), dict):
        raw = raw["brief"]
        log.append("brief: unwrapped {'brief': ...}")
    out = empty_brief()
    try:
        out["scene"] = _text(raw.get("scene", raw.get("scene_prompt")), 360)
        out["still"] = _text(raw.get("still", raw.get("still_prompt", raw.get("image"))), 600)
        out["motions"] = _motions(raw.get("motion", raw.get("motions", raw.get("shots"))), log)
        out["music"] = _text(raw.get("music", raw.get("music_tags", raw.get("tags"))), 320)
        bpm = _num(raw.get("bpm", raw.get("tempo")))
        if bpm is not None:
            out["bpm"] = int(min(150, max(60, round(bpm)))) if bpm > 0 else None
            if out["bpm"] is not None and round(bpm) != out["bpm"]:
                log.append(f"bpm: {bpm:g} clipped to {out['bpm']}")
        out["technique"] = _technique(raw.get("technique", raw.get("style")), log)
        out["mood"] = _moods(raw.get("mood", raw.get("moods")), log)
        pal = raw.get("palette", raw.get("colors"))
        cols = [h for h in (_hex(c) for c in (pal if isinstance(pal, list) else [])) if h]
        if len(cols) >= 3:
            out["palette"] = order_palette(cols[:6])
        elif pal:
            log.append(f"palette: {len(cols)} usable colours -> rules")
        out["signoff"] = _text(raw.get("signoff", raw.get("caption")), 48).lower()
        out["blocks"] = _blocks(raw.get("blocks", raw.get("block")), log)
        out["direction"] = _direction(raw.get("direction", raw.get("dir")), log)
        d = out["direction"]
        if not out["music"] and (d.get("music_genre") or d.get("music_lead")):       # the direction's genre / lead become the music tags
            out["music"] = ", ".join(x for x in (d.get("music_genre"), d.get("music_lead"), "instrumental") if x)
        if out["bpm"] is None and d.get("bpm"):
            out["bpm"] = d["bpm"]
    except Exception as e:      # noqa: BLE001  a malformed field must never take the planner down
        log.append(f"brief: field error {type(e).__name__}: {e} -> rules for the rest")
    return out


def brief_json(brief: dict) -> str:
    return json.dumps(brief, ensure_ascii=False, sort_keys=True)
