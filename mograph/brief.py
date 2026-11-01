"""Creative brief: the small JSON a VLM (design_llm) fills instead of writing a whole plan.

Free planner path: Qwen -> brief (about 200 tokens) -> ``planner_rules.plan_from_brief``.
The model only chooses taste: style family, palette, energy, tempo, arc, how the user's own
words split into display lines, the hero image, where text sits, and mood words. Code owns
beats, scene timing, treatments, layout, contrast and fonts (see _info/decisions.md,
"Planner architecture").

``brief_prompt`` writes the instruction: field descriptions and enumerations only, no
example values to copy, enumeration order shuffled per seed. ``parse_brief`` turns raw model
text into a normalised brief (JSON repair from ``llm_plan``), logging every fix. ``lines``
must be verbatim runs of the user's words; anything invented falls back to ``auto_lines``.
``slot_machine`` is the seeded spin (family, palette, theme) used when there is no input.
"""
from __future__ import annotations

import colorsys
import json
import re
import zlib

import numpy as np

from .font_registry import MOOD_TAGS

__all__ = ["FAMILIES", "FAMILY_SPECS", "ARCS", "TEXT_REGIONS", "MOODS", "THEMES", "brief_prompt", "parse_brief",
           "normalize_brief", "auto_lines", "display_text", "slot_machine", "is_brief"]

# style family -> short description (prompt text)
FAMILIES: dict[str, str] = {
    "film_soft": "warm film, slow dissolves",
    "film_hectic": "film, flash cuts, strobe",
    "type_soft": "elegant type, words hold",
    "type_hectic": "giant kinetic type, sliding rows",
    "film_type_soft": "film under small still captions",
    "film_type_hectic": "type over film, punch zooms",
    "paper_assembly": "halftone paper cut-outs",
    "hype_print": "screenprint, strobe, huge type",
    "ambient_glow": "darkness, one light grows",
    "neon_loop": "neon night, echo trails",
    "soft_loop": "pastel shapes that breathe",
    "xerox_zine": "photocopy zine, one accent",
}

# family -> energy range, bpm range, default arc, pace (soft | mid | hectic), default moods
FAMILY_SPECS: dict[str, dict] = {
    "film_soft": {"energy": (0.15, 0.35), "bpm": (64, 84), "arc": "chapters", "pace": "soft",
                  "moods": ["warm", "nostalgic", "calm"]},
    "film_hectic": {"energy": (0.85, 1.0), "bpm": (128, 150), "arc": "hold_drop_bursts", "pace": "hectic",
                    "moods": ["hype", "dark"]},
    "type_soft": {"energy": (0.1, 0.3), "bpm": (60, 78), "arc": "chapters", "pace": "soft",
                  "moods": ["calm", "luxury"]},
    "type_hectic": {"energy": (0.85, 1.0), "bpm": (132, 150), "arc": "hold_drop_bursts", "pace": "hectic",
                    "moods": ["hype", "techno"]},
    "film_type_soft": {"energy": (0.15, 0.35), "bpm": (64, 84), "arc": "chapters", "pace": "soft",
                       "moods": ["calm", "warm"]},
    "film_type_hectic": {"energy": (0.85, 1.0), "bpm": (128, 150), "arc": "hold_drop_bursts", "pace": "hectic",
                         "moods": ["hype", "dark"]},
    "paper_assembly": {"energy": (0.3, 0.6), "bpm": (84, 110), "arc": "chapters", "pace": "mid",
                       "moods": ["nostalgic", "warm", "playful"]},
    "hype_print": {"energy": (0.8, 1.0), "bpm": (120, 140), "arc": "hold_drop_bursts", "pace": "hectic",
                   "moods": ["hype", "dark"]},
    "ambient_glow": {"energy": (0.15, 0.45), "bpm": (60, 80), "arc": "dark_grow", "pace": "soft",
                     "moods": ["dark", "calm", "techno"]},
    "neon_loop": {"energy": (0.55, 0.8), "bpm": (100, 124), "arc": "loop", "pace": "mid",
                  "moods": ["nostalgic", "techno", "playful"]},
    "soft_loop": {"energy": (0.1, 0.3), "bpm": (60, 76), "arc": "loop", "pace": "soft",
                  "moods": ["calm", "playful"]},
    "xerox_zine": {"energy": (0.6, 0.9), "bpm": (110, 140), "arc": "acceleration", "pace": "hectic",
                   "moods": ["dark", "nostalgic", "hype"]},
}

# energy arcs (motion-principles.md, idea engine step 4)
ARCS: dict[str, str] = {
    "dark_grow": "dark start, grows, still end",
    "hold_drop_bursts": "hold, drop, quick-cut bursts",
    "chapters": "chapters, new ground colour each",
    "acceleration": "cuts keep speeding up",
    "loop": "end matches start",
}
TEXT_REGIONS = ["top", "middle", "bottom", "left", "right", "auto"]
MOODS: list[str] = list(MOOD_TAGS)

# curated palettes per family: [dark, light, accents...]
PALETTES: dict[str, list[list[str]]] = {
    "film_soft": [["#1c1712", "#f4eadb", "#e7a977", "#c9874f", "#7d9c9a"],
                  ["#14181c", "#f2ece2", "#d98c5f", "#5f8fa0", "#e8c27a"],
                  ["#1d1a22", "#f6eee6", "#e3a6a1", "#b88a5a", "#8aa39b"]],
    "film_hectic": [["#0d1b22", "#f4ead8", "#e86a33", "#2f8f9d", "#d9b26f"],
                    ["#120c0a", "#fbeee0", "#ff5a1f", "#ffb000", "#3a86ff"],
                    ["#0b0f14", "#f1f1ea", "#e63946", "#f4a261", "#2a9d8f"]],
    "type_soft": [["#1d2629", "#f0e9dc", "#3d6670", "#c9b28f"],
                  ["#222018", "#f3eee4", "#9a5b3c", "#7f8f6a"],
                  ["#1b1f2b", "#eef0f2", "#6c7fa8", "#d9a5a0"]],
    "type_hectic": [["#0d0d0d", "#f2eee4", "#ff3d1f"],
                    ["#111111", "#f5f5f0", "#2e5bff", "#ffd60a"],
                    ["#0a0a0a", "#efeee9", "#00c46a", "#ff4fa3"]],
    "film_type_soft": [["#1d2a30", "#f6efe3", "#e7a977", "#5f9ea0", "#c9b79c"],
                       ["#201a17", "#f5ede2", "#c97b63", "#a8b5a2"],
                       ["#161b22", "#eef1f4", "#9db4c0", "#e0b1a4"]],
    "film_type_hectic": [["#0e0b0b", "#f7efe5", "#ff4d2e", "#ffc145"],
                         ["#0c1218", "#f3f3ee", "#ff2e63", "#08d9d6"],
                         ["#100d0c", "#f6ead7", "#e8572a", "#4ea8de"]],
    "paper_assembly": [["#1d1b19", "#e8dcc3", "#e2582a", "#dca232", "#3e8a86"],
                       ["#22201d", "#efe4cc", "#c4452f", "#3b6ea5", "#e3b23c"],
                       ["#1a1a1a", "#f0e6d0", "#d7263d", "#1b998b", "#f4c95d"]],
    "hype_print": [["#0b0b0b", "#f4f1ea", "#ff3b1f"],
                   ["#0d0d12", "#f5f5f0", "#ffd60a", "#ff3b30"],
                   ["#0a0a0a", "#f1efe8", "#1f4fff", "#ff5c00"]],
    "ambient_glow": [["#02060c", "#e9f7f7", "#36c2c9", "#0b2a3a"],
                     ["#05030a", "#f3eefc", "#b388ff", "#1e1033"],
                     ["#020805", "#eefaf0", "#5dffa0", "#0c2a1a"]],
    "neon_loop": [["#0d0221", "#d1f7ff", "#ff2a6d", "#05d9e8", "#f9c80e"],
                  ["#0a0118", "#fdf0ff", "#ff00a0", "#7b2cff", "#00f0ff"],
                  ["#05070f", "#e8fbff", "#00ffb2", "#ff3caa", "#ffe45e"]],
    "soft_loop": [["#2b3440", "#eef0f2", "#c9d8e8", "#f1c9c0", "#8fb8a8"],
                  ["#2f2a3a", "#f6f2f8", "#d7c6ee", "#f6d6c2", "#a9cbd8"],
                  ["#283236", "#f1f5f2", "#bfe0d0", "#f3dcae", "#c2c8ec"]],
    "xerox_zine": [["#141414", "#eeece4", "#ff3b1f"],
                   ["#161616", "#efede6", "#0047ff"],
                   ["#121212", "#f2efe6", "#ff2d8a"]],
}

# slot-machine themes (the spin's headline when no model is in the loop)
THEMES = ["Night swim", "Paper moon", "Static bloom", "Slow orbit", "Salt and smoke", "Glass garden",
          "Last ferry home", "Velvet hour", "Desert radio", "Cold open", "Small fires", "Hello, voltage",
          "Tidal hum", "Wild arcade", "Inkwell weather", "Lunar laundromat", "Sugar static", "After the rain",
          "Neon orchard", "Quiet machines", "Copper sunrise", "Pocket thunder", "Midnight bakery", "Velocity",
          "Signal lost", "Soft landing", "Dust and gold", "Ghost frequency", "Open water", "Hard candy"]

_MOOD_SYN = {"cozy": "warm", "cosy": "warm", "tender": "warm", "homely": "warm", "intimate": "warm",
             "friendly": "warm", "retro": "nostalgic", "vintage": "nostalgic", "memory": "nostalgic",
             "wistful": "nostalgic", "energetic": "hype", "aggressive": "hype", "intense": "hype", "fast": "hype",
             "bold": "hype", "loud": "hype", "hectic": "hype", "exciting": "hype", "serene": "calm",
             "peaceful": "calm", "soft": "calm", "airy": "calm", "gentle": "calm", "quiet": "calm", "slow": "calm",
             "dreamy": "calm", "relaxed": "calm", "mysterious": "dark", "moody": "dark", "noir": "dark",
             "eerie": "dark", "dread": "dark", "ominous": "dark", "nocturnal": "dark", "night": "dark",
             "tech": "techno", "digital": "techno", "futuristic": "techno", "cyber": "techno",
             "electronic": "techno", "elegant": "luxury", "premium": "luxury", "glamorous": "luxury",
             "romantic": "luxury", "refined": "luxury", "fun": "playful", "bright": "playful",
             "cheerful": "playful", "whimsical": "playful", "joyful": "playful", "quirky": "playful"}

_ARC_SYN = [("dark", "dark_grow"), ("grow", "dark_grow"), ("burst", "hold_drop_bursts"),
            ("drop", "hold_drop_bursts"), ("chapter", "chapters"), ("accel", "acceleration"),
            ("faster", "acceleration"), ("loop", "loop")]

_DIRECTIVE_RE = re.compile(r"^\s*(style|mood|vibe|music|look|feel|tone)\s*:", re.I)
_WORD_RE = re.compile(r"[\w']+(?:[-.:/][\w']+)*", re.U)
_SHORT = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "at", "by", "or", "with", "&"}
MAX_LINES = 6


def _salt(seed: int, tag: str) -> list[int]:
    return [int(seed) & 0xFFFFFFFF, zlib.crc32(tag.encode())]


# --------------------------------------------------------------------------
# user text -> display text and automatic lines
# --------------------------------------------------------------------------

def display_text(user_text: str) -> str:
    """The user's text without directive lines ("Style: ...", "Mood: ..."), which steer but are not shown."""
    return "\n".join(ln for ln in (user_text or "").splitlines() if ln.strip() and not _DIRECTIVE_RE.match(ln))


def directives(user_text: str) -> str:
    return " ".join(ln for ln in (user_text or "").splitlines() if _DIRECTIVE_RE.match(ln))


def _balanced_split(words: list[str], max_words: int = 5) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    n = int(np.ceil(len(words) / max_words))
    best = None
    # choose n-1 cut points minimising length spread; never end a line on a short word ("and", "the")
    import itertools
    for cuts in itertools.combinations(range(1, len(words)), n - 1):
        parts = [words[a:b] for a, b in zip((0,) + cuts, cuts + (len(words),))]
        if any(len(p) > max_words for p in parts):
            continue
        spread = max(len(p) for p in parts) - min(len(p) for p in parts)
        bad = sum(1 for p in parts[:-1] if p[-1].lower().strip(",.;:") in _SHORT)
        comma = -sum(1 for p in parts[:-1] if p[-1].endswith(","))
        key = (bad, spread, comma, cuts)
        if best is None or key < best[0]:
            best = (key, parts)
    return best[1] if best else [words[i:i + max_words] for i in range(0, len(words), max_words)]


def auto_lines(user_text: str, max_words: int = 5) -> list[str]:
    """Display lines from the user's text, words untouched: one per text line, split at sentence ends and
    dashes, at commas when both sides keep 2+ words, then balanced to at most ``max_words`` words."""
    out: list[str] = []
    for ln in display_text(user_text).splitlines():
        parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|\s+[—–-]+\s+", ln.strip()) if p.strip()]
        for p in parts:
            sub = [s.strip() for s in re.split(r"(?<=,)\s+", p) if s.strip()]
            if len(sub) > 1 and all(len(s.split()) >= 2 for s in sub):
                chunks = sub
            else:
                chunks = [p]
            for c in chunks:
                out += [" ".join(w) for w in _balanced_split(c.split(), max_words)]
    out = [o.strip(" -—–") for o in out if o.strip(" -—–")]
    while len(out) > MAX_LINES:            # too many: merge the shortest neighbouring pair
        j = min(range(len(out) - 1), key=lambda k: len(out[k].split()) + len(out[k + 1].split()))
        out[j:j + 2] = [out[j] + " " + out[j + 1]]
    return out


def _tok_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).casefold().replace("’", "'"), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _verbatim_lines(cand, user_text: str, log: list) -> list[str] | None:
    """Model lines -> the matching runs of the user's own text (original spelling and punctuation), or None
    when a line adds, changes or reorders words, or the lines drop too much of the text."""
    if not isinstance(cand, list) or not cand:
        return None
    src = display_text(user_text)
    toks = _tok_spans(src)
    words = [t[0] for t in toks]
    if not words:
        return None
    pos, out, covered = 0, [], 0
    for c in cand:
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        if not isinstance(c, str) or not c.strip():
            continue
        lw = [t[0] for t in _tok_spans(c)]
        if not lw:
            continue
        hit = next((i for i in range(pos, len(words) - len(lw) + 1) if words[i:i + len(lw)] == lw), None)
        if hit is None:
            log.append(f"lines: {c!r} is not a run of the user's words -> auto split")
            return None
        a, b = toks[hit][1], toks[hit + len(lw) - 1][2]
        while b < len(src) and src[b] in ".,!?;:'\"”’)…":      # keep the user's trailing punctuation
            b += 1
        out.append(src[a:b].strip())
        covered += len(lw)
        pos = hit + len(lw)
    if not out:
        return None
    if covered < 0.6 * len(words):
        log.append(f"lines: cover {covered}/{len(words)} of the user's words -> auto split")
        return None
    fixed = []
    for ln in out:                    # very long lines: balance them (still the user's words)
        n = len(ln.split())
        fixed += [" ".join(w) for w in _balanced_split(ln.split(), 6)] if n > 7 else [ln]
    return fixed[:MAX_LINES]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def _hex(c) -> str | None:
    if not isinstance(c, str):
        return None
    m = re.match(r"^\s*#?([0-9a-fA-F]{6})(?:[0-9a-fA-F]{2})?\s*$", c)
    if m:
        return "#" + m.group(1).lower()
    m = re.match(r"^\s*#?([0-9a-fA-F]{3})\s*$", c)
    if m:
        return "#" + "".join(ch * 2 for ch in m.group(1)).lower()
    from .llm_plan import _NAMED
    return _NAMED.get(c.strip().lower())


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


def _luma(h: str) -> float:
    c = _rgb(h)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _chroma(h: str) -> float:
    c = _rgb(h)
    return float(c.max() - c.min())


def _to_hex(c) -> str:
    return "#" + "".join(f"{int(round(min(1.0, max(0.0, float(v))) * 255)):02x}" for v in c)


def order_palette(cols: list[str]) -> list[str]:
    """[darkest, lightest, accents in the given order]; the dark end pushed to luma <= 0.12 and the light
    end to >= 0.85 so text on either always reads."""
    cols = list(dict.fromkeys(cols))
    dark = min(cols, key=_luma)
    light = max(cols, key=_luma)
    acc = [c for c in cols if c not in (dark, light)]
    d, l_ = _rgb(dark), _rgb(light)
    ld, ll = _luma(dark), _luma(light)
    if ld > 0.12:
        d = d * (0.12 / ld)
    if ll < 0.85:
        t = (0.85 - ll) / max(1e-6, 1.0 - ll)
        l_ = l_ + t * (1.0 - l_)
    return [_to_hex(d), _to_hex(l_)] + acc


def _norm_key(v) -> str:
    return re.sub(r"[\s\-/+&]+", "_", str(v or "").strip().lower()).strip("_")


def _family(v, log: list) -> str | None:
    k = _norm_key(v)
    if k in FAMILIES:
        return k
    if not k:
        return None
    k2 = k.replace("filmic", "film").replace("typographic", "type").replace("typography", "type")
    k2 = k2.replace("and_", "").replace("plus_", "").replace("calm", "soft").replace("slow", "soft")
    k2 = k2.replace("fast", "hectic").replace("print_hype", "hype_print")
    if k2 in FAMILIES:
        log.append(f"family: {v!r} -> {k2!r}")
        return k2
    for name in sorted(FAMILIES, key=lambda f: -f.count("_")):   # loose: all parts of a family name present
        if all(p in k2 for p in name.split("_")):
            log.append(f"family: {v!r} -> {name!r}")
            return name
    log.append(f"family: {v!r} unknown")
    return None


def _infer_family(moods: list[str], energy: float | None, text: str, rng) -> str:
    t = text.lower()
    hectic = (energy or 0.5) >= 0.75 or any(w in t for w in ("hectic", "fast", "hype", "aggressive"))
    soft = (energy if energy is not None else 0.5) <= 0.35 or any(w in t for w in ("soft", "slow", "calm"))
    if "film" in t and ("type" in t or "typo" in t):
        return "film_type_hectic" if hectic else "film_type_soft"
    if "film" in t:
        return "film_hectic" if hectic else "film_soft"
    if "typo" in t or "type" in t:
        return "type_hectic" if hectic else "type_soft"
    m = moods[0] if moods else None
    if hectic:
        return {"techno": "type_hectic", "dark": "film_hectic", "nostalgic": "xerox_zine"}.get(m, "hype_print")
    if soft:
        return {"dark": "ambient_glow", "warm": "film_soft", "luxury": "type_soft", "playful": "soft_loop",
                "nostalgic": "film_soft", "techno": "ambient_glow"}.get(m, "film_type_soft")
    return {"nostalgic": "paper_assembly", "warm": "paper_assembly", "techno": "neon_loop", "playful": "neon_loop",
            "dark": "ambient_glow", "luxury": "film_type_soft", "hype": "film_type_hectic"}.get(
        m, str(rng.choice(["paper_assembly", "neon_loop", "film_type_soft"])))


def _moods(v, log: list) -> list[str]:
    items = v if isinstance(v, list) else re.split(r"[,/;\s]+", v) if isinstance(v, str) else []
    out = []
    for it in items:
        w = str(it).strip().lower()
        m = w if w in MOODS else _MOOD_SYN.get(w)
        if m is None and w:
            log.append(f"mood: {it!r} not a registry mood, dropped")
        if m and m not in out:
            out.append(m)
    return out[:3]


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", v)
        if m:
            return float(m.group(1))
    return None


def slot_machine(seed: int) -> dict:
    """Seeded spin for empty input: family, palette, theme, arc, energy, bpm.

    Consecutive seeds walk a seeded permutation of the families (a new permutation every len(FAMILIES)
    seeds), so any run of 10 consecutive seeds shows at least 6 different families; within a family the
    palette, its accent hues, theme, arc, tempo and energy are drawn per seed."""
    seed = int(seed)
    names = list(FAMILIES)
    n = len(names)
    perm = np.random.default_rng(_salt(seed // n, "slot-perm")).permutation(n)
    family = names[int(perm[seed % n])]
    rng = np.random.default_rng(_salt(seed, "slot"))
    spec = FAMILY_SPECS[family]
    bank = PALETTES[family]
    pal = list(bank[int(rng.integers(len(bank)))])
    shift = float(rng.uniform(-0.06, 0.06))          # rotate accent hues a little (<= ~20 deg)
    acc = []
    for h in pal[2:]:
        hh, ll, ss = colorsys.rgb_to_hls(*_rgb(h))
        acc.append(_to_hex(colorsys.hls_to_rgb((hh + shift) % 1.0, ll, ss)))
    arc = spec["arc"] if rng.random() < 0.6 else str(rng.choice(list(ARCS)))
    lo, hi = spec["energy"]
    blo, bhi = spec["bpm"]
    return {"family": family, "palette": pal[:2] + acc, "theme": str(rng.choice(THEMES)), "arc": arc,
            "energy": round(float(rng.uniform(lo, hi)), 3), "bpm": float(int(rng.integers(blo, bhi + 1))),
            "mood": list(spec["moods"][:2])}


def is_brief(obj) -> bool:
    return isinstance(obj, dict) and "scenes" not in obj and any(k in obj for k in ("family", "lines", "palette"))


def normalize_brief(raw, user_text: str = "", n_images: int = 0, seed: int = 0, log: list | None = None) -> dict:
    """Any brief-like dict (or None) -> a complete brief:
    {family, palette (list | None = planner picks), energy, bpm, arc, lines, hero_image, text_region, mood,
     theme (slot machine only)}. Idempotent; every fix goes to ``log``."""
    log = log if log is not None else []
    raw = dict(raw) if isinstance(raw, dict) else {}
    if isinstance(raw.get("brief"), dict) and "family" not in raw:
        raw = dict(raw["brief"])
        log.append("brief: unwrapped {'brief': ...}")
    rng = np.random.default_rng(_salt(seed, "brief"))
    shown = display_text(user_text)
    empty = not shown.strip() and n_images == 0
    spin = slot_machine(seed) if empty else None

    moods = _moods(raw.get("mood", raw.get("moods")), log)
    energy = _num(raw.get("energy"))
    if energy is not None and energy > 1.0:
        energy = energy / 10.0 if energy <= 10 else 1.0
        log.append("energy: rescaled to 0..1")
    family = _family(raw.get("family"), log)
    if spin is not None:
        if family and family != spin["family"]:
            log.append(f"family: slot machine -> {spin['family']!r}")
        family = spin["family"]
    if family is None:
        family = _infer_family(moods, energy, (user_text or "") + " " + directives(user_text), rng)
        log.append(f"family: missing -> {family!r} (from mood / energy / words)")
    spec = FAMILY_SPECS[family]
    lo, hi = spec["energy"]
    if energy is None:
        energy = spin["energy"] if spin else (lo + hi) / 2.0
        log.append(f"energy: missing -> {energy:.3f}")
    e2 = float(np.clip(energy, lo, hi))
    if abs(e2 - energy) > 1e-9:
        log.append(f"energy: {energy:.3f} outside {family} range -> {e2:.3f}")
    energy = round(e2, 3)
    bpm = _num(raw.get("bpm", raw.get("tempo")))
    blo, bhi = spec["bpm"]
    if bpm is None:
        bpm = spin["bpm"] if spin else float(round(blo + (bhi - blo) * (energy - lo) / max(1e-6, hi - lo)))
        log.append(f"bpm: missing -> {bpm:g}")
    b2 = float(np.clip(round(bpm), blo, bhi))
    if b2 != bpm:
        log.append(f"bpm: {bpm:g} outside {family} range -> {b2:g}")
    bpm = b2

    pal = raw.get("palette", raw.get("colors"))
    cols = [h for h in (_hex(c) for c in (pal if isinstance(pal, list) else [])) if h]
    if spin is not None:
        palette = spin["palette"]
    elif len(cols) >= 3:
        palette = order_palette(cols[:6])
        if palette != cols[:6]:
            log.append("palette: reordered dark, light, accents / contrast stretched")
        vivid = [c for c in palette[2:] if _chroma(c) >= 0.2]
        if len(vivid) < len(palette) - 2:
            bank = PALETTES[family][int(rng.integers(len(PALETTES[family])))]
            palette = palette[:2] + (vivid or list(bank[2:]))
            log.append("palette: grey accents dropped" if vivid else "palette: no vivid accent -> family accents")
    else:
        if pal is not None:
            log.append(f"palette: {len(cols)} usable colours -> planner default")
        palette = None
    arc = _norm_key(raw.get("arc"))
    if arc not in ARCS:
        arc2 = next((a for k, a in _ARC_SYN if k in arc), None) if arc else None
        if arc2 is None:
            arc2 = spin["arc"] if spin else spec["arc"]
            if arc:
                log.append(f"arc: {raw.get('arc')!r} unknown -> {arc2!r}")
        else:
            log.append(f"arc: {raw.get('arc')!r} -> {arc2!r}")
        arc = arc2

    cand = raw.get("lines", raw.get("headline_split"))
    if isinstance(cand, str):
        cand = [s for s in re.split(r"\n|\s*/\s*|\|", cand) if s.strip()]
    if shown.strip():
        lines = _verbatim_lines(cand, user_text, log)
        if lines is None:
            if cand:
                log.append("lines: fell back to auto split")
            lines = auto_lines(user_text)
    else:
        lines = []
        for c in cand if isinstance(cand, list) else []:
            c = " ".join(str(x) for x in c) if isinstance(c, list) else c
            if isinstance(c, str) and c.strip() and len(c.split()) <= 5:
                lines.append(c.strip())
        lines = lines[:2]
        if spin is not None and not lines:
            lines = [spin["theme"]]
            log.append(f"lines: slot machine theme {spin['theme']!r}")

    hero = raw.get("hero_image")
    if isinstance(hero, str):
        m = re.search(r"(\d+)", hero)
        hero = int(m.group(1)) if m else None
    if isinstance(hero, (int, float)) and not isinstance(hero, bool) and n_images:
        h = int(hero)
        if h >= n_images and 1 <= h <= n_images:          # 1-based answer
            h -= 1
        hero = h if 0 <= h < n_images else 0
    else:
        hero = 0 if n_images else None
    region = _norm_key(raw.get("text_region"))
    if region not in TEXT_REGIONS:
        region = next((r for r in TEXT_REGIONS if r in region), "auto") if region else "auto"
    if not moods:
        moods = list(spin["mood"] if spin else spec["moods"][:2])
    out = {"family": family, "palette": palette, "energy": energy, "bpm": bpm, "arc": arc, "lines": lines,
           "hero_image": hero, "text_region": region, "mood": moods}
    theme = raw.get("theme") or (spin["theme"] if spin else None)
    if theme:
        out["theme"] = str(theme)
    return out


def parse_brief(text: str, user_text: str = "", n_images: int = 0, seed: int = 0, log: list | None = None) -> dict:
    """Raw model output -> normalised brief. Unparsable output gives the rule defaults (logged)."""
    from .llm_plan import PlanExtractError, loads_tolerant
    log = log if log is not None else []
    raw = None
    if isinstance(text, dict):
        raw = text
    else:
        try:
            raw = loads_tolerant(text or "", log)
        except PlanExtractError as e:
            log.append(f"brief: no JSON object ({e}) -> defaults")
    return normalize_brief(raw, user_text, n_images, seed, log)


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def brief_prompt(user_text: str, n_images: int = 0, seed: int = 0) -> str:
    """Instruction for the VLM. Field descriptions and enumerations only (no example values to copy);
    the order of every enumeration is shuffled per seed. Under 250 words for typical input."""
    rng = np.random.default_rng(_salt(seed, "prompt"))

    def shuf(items):
        items = list(items)
        return [items[int(i)] for i in rng.permutation(len(items))]

    arcs = "; ".join(f"{a} ({ARCS[a]})" for a in shuf(ARCS))
    moods = ", ".join(shuf(MOODS))
    regions = ", ".join(shuf(TEXT_REGIONS))
    shown = display_text(user_text)
    head = ("You art-direct a 10-second motion graphic; code animates it from your brief. "
            "Reply with one JSON object only.")
    if not shown.strip() and n_images == 0:
        spin = slot_machine(seed)
        return "\n".join([
            head,
            f'There is no user text. Theme: "{spin["theme"]}". Style: {FAMILIES[spin["family"]]}.',
            "Keys:",
            "lines: a headline of 2 to 4 fresh words for the theme, optionally a second line of up to 4 "
            "words; no slogans or cliches.",
            f"arc: one of {arcs}.",
            f"mood: a list of 1 to 3 words from {moods}.",
            "energy: 0 (still) to 1 (frantic).",
        ])
    fams = "; ".join(f"{f} ({FAMILIES[f]})" for f in shuf(FAMILIES))
    imgs = n_images > 0
    rows = [
        head,
        "Keys:",
        f"family: one of {fams}.",
        "palette: 3 to 6 colours as \"#rrggbb\": a near-black, a near-white, then 1 to 4 vivid accents"
        + (", sampled from the images." if imgs else ", chosen for the mood."),
        "energy: 0 (still) to 1 (frantic).",
        "bpm: music tempo, 60 to 150.",
        f"arc: one of {arcs}.",
        "lines: the user's text cut into 1 to 4 display lines, in order, words copied exactly; skip style "
        "instructions; never add or change words.",
    ]
    if imgs:
        rows.append(f"hero_image: index 0 to {n_images - 1} of the strongest image.")
        rows.append(f"text_region: one of {regions}: the calm, empty part of the hero image.")
    else:
        rows.append(f"text_region: one of {regions}.")
    rows += [
        f"mood: a list of 1 to 3 words from {moods}.",
        "Calm, soft or slow words mean low energy and tempo; fast, hype or hectic words mean high; a stated "
        "style picks the family.",
        f'USER TEXT: """{(user_text or "").strip()}"""',
        f"IMAGES: {n_images} attached." if imgs else "IMAGES: none.",
    ]
    return "\n".join(rows)


def brief_json(brief: dict) -> str:
    return json.dumps(brief, ensure_ascii=False)
