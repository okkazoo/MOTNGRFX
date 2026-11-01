"""Deterministic rule planner: analysis dict -> plan dict (SPEC Interface 6).

No LLM. All randomness comes from ``numpy.random.default_rng(seed)``.
"""
from __future__ import annotations

import colorsys
import re
import unicodedata

import numpy as np

from . import geometry
from .brief import FAMILY_SPECS, PALETTES, _balanced_split, normalize_brief, parse_brief
from .font_registry import REGISTRY, pick_fonts, preferred_case
from .plan import (FORMATS, PLAN_VERSION, TRANSITIONS, VERBS, validate_plan)

FPS = 30
MIN_DUR, MAX_DUR, DEFAULT_DUR = 5.0, 15.0, 10.0
PREFERRED_SCENE_LEN = 2.0
SAFE = {"9:16": (0.05, 0.06, 0.95, 0.94), "16:9": (0.05, 0.06, 0.95, 0.94)}   # x0, y0, x1, y1
LOWER_THIRD = {"9:16": [0.08, 0.66, 0.84, 0.16], "16:9": [0.06, 0.64, 0.6, 0.22]}
CENTRE_BOX = {"9:16": [0.08, 0.38, 0.84, 0.2], "16:9": [0.1, 0.34, 0.8, 0.24]}
MIN_TEXT_W = {"9:16": 0.5, "16:9": 0.3}
MAX_TEXT_H = {"9:16": 0.24, "16:9": 0.3}
SUB_H = {"9:16": 0.07, "16:9": 0.09}

HIGH_ENERGY_TRANSITIONS = ["zoom_blur", "glitch", "push", "flash"]
LOW_ENERGY_TRANSITIONS = ["crossfade", "luma_wipe", "shape_reveal"]

TEXT_IN_BY_MOTION = {
    "snappy": ["word_stagger", "slide_up", "scale_pop", "mask_wipe", "kinetic_split"],
    "smooth": ["fade", "blur_in", "slide_up", "mask_wipe", "word_stagger"],
    "bouncy": ["scale_pop", "word_stagger", "slide_up", "kinetic_split"],
    "glitch": ["type_on", "cut", "kinetic_split", "word_stagger"],
}
TEXT_OUT_BY_MOTION = {
    "snappy": ["slide_up", "scale_down", "mask_wipe", "cut"],
    "smooth": ["fade", "blur_out"],
    "bouncy": ["scale_down", "slide_up"],
    "glitch": ["cut", "blur_out"],
}
IN_DUR = {"snappy": 0.4, "smooth": 0.7, "bouncy": 0.5, "glitch": 0.3}

# keyword -> palette/style table (used for text-only and audio-only palettes, and for
# fonts / motion / music mood everywhere). palette = [dark, light, accent, accent, accent]
MOOD_TABLE = {
    "hype": {"keywords": ["hype", "energy", "energetic", "launch", "sale", "sport", "go", "now", "fast", "win",
                          "power", "drop", "new", "boost", "run", "driving", "move", "beat"],
             "palette": ["#0d0d12", "#f5f5f0", "#ff3b30", "#ffd60a", "#2e6bff"],
             "fonts": ("Big Shoulders Display", "Space Grotesk"), "motion": "snappy", "music": "driving", "scale": "minor",
             "keys": ["A", "E", "D", "F#"], "energy": 0.85},
    "calm": {"keywords": ["calm", "relax", "peace", "peaceful", "soft", "gentle", "mindful", "yoga", "breathe",
                          "slow", "quiet", "serene", "sleep", "spa", "still"],
             "palette": ["#1d2a2e", "#f3efe6", "#9cc5b4", "#e8b98a", "#6f8fa8"],
             "fonts": ("Interlope", "Space Grotesk"), "motion": "smooth", "music": "calm", "scale": "major",
             "keys": ["F", "C", "G", "D"], "energy": 0.25},
    "dark": {"keywords": ["dark", "night", "horror", "mystery", "noir", "shadow", "fear", "secret", "midnight",
                          "ghost", "black", "storm"],
             "palette": ["#07070a", "#d9d6e0", "#8b1e3f", "#5b3fa6", "#c9a227"],
             "fonts": ("Facade", "Space Grotesk"), "motion": "smooth", "music": "dark", "scale": "minor",
             "keys": ["D", "C#", "F", "A#"], "energy": 0.4},
    "luxury": {"keywords": ["luxury", "gold", "premium", "elegant", "fashion", "style", "exclusive", "classic",
                            "beauty", "perfume", "jewel", "couture"],
               "palette": ["#0f0e0c", "#f6f1e7", "#c9a45c", "#8a6f3e", "#e7d3a7"],
               "fonts": ("Interlope", "Space Grotesk"), "motion": "smooth", "music": "calm", "scale": "minor",
               "keys": ["C", "F", "G#", "D#"], "energy": 0.45},
    "tech": {"keywords": ["tech", "ai", "code", "future", "digital", "cyber", "data", "robot", "software",
                          "app", "cloud", "algorithm", "machine", "neural", "startup", "glitch"],
             "palette": ["#060b14", "#e6f1ff", "#00e5ff", "#7c4dff", "#39ff88"],
             "fonts": ("Space Grotesk", "JetBrains Mono"), "motion": "glitch", "music": "driving",
             "scale": "minor", "keys": ["B", "E", "F#", "C#"], "energy": 0.7},
    "nature": {"keywords": ["nature", "forest", "green", "earth", "eco", "plant", "organic", "garden", "leaf",
                            "mountain", "wild", "tree", "bloom", "spring", "hike", "hiking", "alps", "alpine", "trail",
                            "trek", "summit", "peak", "valley", "lake", "outdoor", "outdoors", "camping", "camp",
                            "grow", "growing", "seed", "seedling", "tomato", "vegetable", "veg", "allotment",
                            "farm", "flower", "meadow"],
               "palette": ["#13201a", "#f1f4ea", "#6fbf4a", "#e0b84f", "#3f8f7a"],
               "fonts": ("Facade", "Space Grotesk"), "motion": "smooth", "music": "bright", "scale": "major",
               "keys": ["G", "D", "C", "A"], "energy": 0.45},
    "playful": {"keywords": ["fun", "kids", "play", "party", "candy", "happy", "celebrate", "birthday", "joy",
                             "pop", "games", "smile", "colorful", "colourful", "playful"],
                "palette": ["#1b1035", "#fff8f0", "#ff4fa3", "#ffd23f", "#2ec4b6"],
                "fonts": ("Facade", "Space Grotesk"), "motion": "bouncy", "music": "playful", "scale": "major",
                "keys": ["C", "F", "G", "D"], "energy": 0.7},
    "romance": {"keywords": ["love", "romance", "romantic", "heart", "wedding", "valentine", "kiss", "together",
                             "forever", "dream"],
                "palette": ["#24101a", "#fff1f3", "#ff5d8f", "#ffb3c6", "#c77dff"],
                "fonts": ("Interlope", "Space Grotesk"), "motion": "smooth", "music": "calm", "scale": "major",
                "keys": ["F", "A#", "D#", "C"], "energy": 0.35},
    "ocean": {"keywords": ["ocean", "sea", "summer", "beach", "wave", "water", "travel", "surf", "island",
                           "sun", "sky", "blue", "holiday", "adventure", "bright"],
              "palette": ["#06202b", "#f2fbff", "#00b4d8", "#ffb703", "#90e0ef"],
              "fonts": ("Big Shoulders Display", "Space Grotesk"), "motion": "bouncy", "music": "bright", "scale": "major",
              "keys": ["E", "A", "D", "B"], "energy": 0.6},
    "retro": {"keywords": ["retro", "80s", "90s", "synth", "arcade", "neon", "disco", "cassette",
                           "vhs", "throwback"],
              "palette": ["#140b2d", "#fdf0d5", "#ff2a6d", "#05d9e8", "#f9c80e"],
              "fonts": ("Facade", "Space Grotesk"), "motion": "glitch", "music": "driving",
              "scale": "minor", "keys": ["A", "D", "E", "G"], "energy": 0.7},
    "epic": {"keywords": ["epic", "legend", "hero", "rise", "bold", "strong", "fight", "battle", "fire",
                          "victory", "champion", "unstoppable", "glory"],
             "palette": ["#0b0b0b", "#f2efe9", "#ff6b1a", "#c1121f", "#fcbf49"],
             "fonts": ("Big Shoulders Display", "Space Grotesk"), "motion": "snappy", "music": "epic", "scale": "minor",
             "keys": ["D", "C", "A#", "G"], "energy": 0.85},
    "warm": {"keywords": ["cafe", "coffee", "espresso", "latte", "autumn", "fall", "menu", "bakery", "bake",
                          "bread", "pastry", "croissant", "pie", "cake", "cosy", "cozy", "warm", "homemade",
                          "kitchen", "recipe", "food", "brunch", "breakfast", "tea", "harvest", "pumpkin",
                          "cinnamon", "vintage", "restoration", "restore", "restored", "craft", "handmade",
                          "workshop", "rustic", "heritage", "wood", "leather", "family", "local"],
             "palette": ["#1e130c", "#fbf3e6", "#d9822b", "#a4452c", "#e8c07d"],
             "fonts": ("Interlope", "Space Grotesk"), "motion": "smooth", "music": "playful", "scale": "major",
             "keys": ["F", "C", "G", "D"], "energy": 0.45},
    "lounge": {"keywords": ["jazz", "jazzy", "sax", "saxophone", "blues", "soul", "vinyl", "lounge", "cocktail",
                            "club", "groove", "groovy", "funk", "swing", "piano", "trumpet", "gig", "band",
                            "speakeasy", "bar", "live"],
               "palette": ["#0b0a12", "#f3ead8", "#c8553d", "#3d5a80", "#e0a458"],
               "fonts": ("Facade", "Space Grotesk"), "motion": "smooth", "music": "playful", "scale": "minor",
               "keys": ["D", "G", "C", "F"], "energy": 0.5},
}

# planner mood -> font-registry moods (strongest first); the display + body pair is drawn from the registry
# per seed (MOOD_TABLE "fonts" is the fallback pair)
FONT_MOODS = {"hype": ["hype"], "calm": ["calm"], "dark": ["dark"], "luxury": ["luxury"], "tech": ["techno"],
              "nature": ["calm", "warm"], "playful": ["playful"], "romance": ["warm", "luxury"],
              "ocean": ["playful", "calm"], "retro": ["nostalgic", "techno"], "epic": ["hype", "dark"],
              "warm": ["warm", "nostalgic"], "lounge": ["nostalgic", "luxury"]}


def _headline_case(display: str) -> str:
    c = preferred_case(display)
    return c if c in ("as_is", "lower") else "upper"


# Generic words (time of day, "new", "go" ...) count half: a subject word ("jazz", "cafe", "hike") beats
# them. The analyzer's own mood label (derived from image brightness) counts a quarter, so it only
# breaks ties and never outvotes a single word the user typed.
WEAK_KEYWORDS = {"new", "now", "go", "night", "win", "run", "move", "drop", "black", "green", "blue", "sky",
                 "sun", "still", "style", "classic", "pop", "play", "bold", "strong", "rise", "fire", "bright",
                 "fast", "power", "beat", "together", "forever", "dream", "fall", "live", "bar", "local",
                 "family", "food", "warm", "wood"}
DERIVED_WEIGHT = 0.25
# fixed tie-break order after "earliest word in the text" (deterministic, not dict order)
MOOD_PRIORITY = ["lounge", "warm", "nature", "calm", "romance", "luxury", "ocean", "playful", "retro", "tech",
                 "epic", "dark", "hype"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _num_or(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _r(x: float, n: int = 4) -> float:
    return float(round(float(x), n))


def _hex(rgb) -> str:
    r, g, b = (int(round(min(1.0, max(0.0, float(c))) * 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def _luma(c) -> float:
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _chroma(c) -> float:
    return float(max(c) - min(c))


def _clip_box(b, fmt: str | None = None) -> list[float]:
    x, y, w, h = (float(v) for v in b)
    x = min(max(x, 0.0), 0.98)
    y = min(max(y, 0.0), 0.98)
    w = min(max(w, 0.01), 1.0 - x)
    h = min(max(h, 0.01), 1.0 - y)
    return [_r(x), _r(y), _r(w), _r(h)]


def _square_box(fmt: str, cx: float, cy: float, size_short: float) -> list[float]:
    """Box that is square in pixels: size_short is a fraction of the frame short side."""
    fw, fh = FORMATS[fmt]
    s = size_short * min(fw, fh)
    w, h = s / fw, s / fh
    x = min(max(cx - w / 2, 0.0), 1.0 - w)
    y = min(max(cy - h / 2, 0.0), 1.0 - h)
    return _clip_box([x, y, w, h])


def _anim(verb: str, dur: float, **extra) -> dict:
    d = {"verb": verb, "dur": _r(dur, 3)}
    d.update(extra)
    return d


# --------------------------------------------------------------------------
# mood / palette
# --------------------------------------------------------------------------

def _fold(text: str) -> str:
    """Lower-case and strip accents ("Café" -> "cafe")."""
    t = unicodedata.normalize("NFKD", str(text or "")).lower()
    return "".join(ch for ch in t if not unicodedata.combining(ch)).replace("’", "'")


def _word_forms(tok: str) -> list[str]:
    tok = tok[:-2] if tok.endswith("'s") else tok.rstrip("'")
    forms = [tok]
    if len(tok) > 3 and tok.endswith("es"):
        forms.append(tok[:-2])
    if len(tok) > 3 and tok.endswith("s"):
        forms.append(tok[:-1])
    return forms


def _text_words(analysis: dict) -> list[list[str]]:
    """Per word of the user's text (in order): its candidate forms."""
    return [_word_forms(t) for t in re.findall(r"[a-z0-9']+", _fold(analysis.get("text") or ""))]


def _tokens(analysis: dict) -> list[str]:
    words = [f[0] for f in _text_words(analysis)]
    kws = [str(k).lower() for k in ((analysis.get("mood") or {}).get("keywords") or [])]
    return kws + words


def _mood_scores(analysis: dict) -> dict[str, tuple[float, int]]:
    """mood -> (score, position of its first text hit). User words weigh 1 (generic words 0.5);
    analyzer keywords that are not in the text (the brightness-derived label) weigh DERIVED_WEIGHT."""
    words = _text_words(analysis)
    text_forms = {f for fs in words for f in fs}
    derived = [_fold(k) for k in ((analysis.get("mood") or {}).get("keywords") or [])]
    derived = [k for k in derived if k not in text_forms]
    out = {}
    for name in MOOD_PRIORITY:
        kw = set(MOOD_TABLE[name]["keywords"])
        score, first = 0.0, 10 ** 6
        for pos, forms in enumerate(words):
            hit = next((f for f in forms if f in kw), None)
            if hit is not None:
                score += 0.5 if hit in WEAK_KEYWORDS else 1.0
                first = min(first, pos)
        score += DERIVED_WEIGHT * sum(1 for k in derived if k in kw)
        out[name] = (score, first)
    return out


def _pick_mood(analysis: dict, energy: float, valence: float) -> tuple[str, bool]:
    scores = _mood_scores(analysis)
    best = min(MOOD_PRIORITY, key=lambda m: (-scores[m][0], scores[m][1], MOOD_PRIORITY.index(m)))
    if scores[best][0] > 0:
        return best, True
    if energy >= 0.7:
        return ("hype" if valence >= 0.5 else "epic"), False
    if energy < 0.4:
        return ("calm" if valence >= 0.5 else "dark"), False
    if valence >= 0.6:
        return "playful", False
    if valence >= 0.45:
        return "ocean", False
    return "tech", False


def _image_palette(images: list[dict], fallback: list[str]) -> list[str]:
    entries = []   # (rgb, weight)
    n = max(1, len(images))
    for img in images:
        for p in img.get("palette") or []:
            try:
                entries.append((_rgb(p["hex"]), float(p.get("weight", 0.0)) / n))
            except (KeyError, ValueError, TypeError):
                continue
    if not entries:
        return list(fallback)

    dark = min(entries, key=lambda e: _luma(e[0]))[0].copy()
    lum = _luma(dark)
    if lum > 0.12:
        dark = dark * (0.12 / lum)
    light = max(entries, key=lambda e: _luma(e[0]))[0].copy()
    lum = _luma(light)
    if lum < 0.9:
        t = (0.9 - lum) / max(1e-6, 1.0 - lum)
        light = light + t * (1.0 - light)

    cands = []
    for img in images:
        if img.get("accent"):
            try:
                cands.append(_rgb(img["accent"]))
            except (ValueError, TypeError):
                pass
    for rgb, w in sorted(entries, key=lambda e: -_chroma(e[0]) * np.sqrt(max(e[1], 1e-6))):
        cands.append(rgb)
    accents: list[np.ndarray] = []
    for c in cands:
        if _chroma(c) < 0.2:
            continue
        c = c.copy()
        if c.max() < 0.85:
            c = c * (0.85 / max(c.max(), 1e-6))
        if any(np.linalg.norm(c - a) < 0.25 for a in accents):
            continue
        accents.append(c)
        if len(accents) == 3:
            break
    pal = [_hex(dark), _hex(light)] + [_hex(a) for a in accents]
    if not accents:
        pal += list(fallback[2:4])
    return pal[:6]


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def _beat_grid(analysis: dict, duration: float, energy: float, bpm_override: float | None = None):
    audio = analysis.get("audio")
    if audio:
        bpm = float(audio.get("bpm") or 0.0)
        beats = [float(b) for b in (audio.get("beats") or []) if 0.0 <= float(b) < duration]
        if not (20.0 <= bpm <= 300.0):
            if len(beats) >= 2:
                bpm = 60.0 / float(np.median(np.diff(beats)))
            else:
                bpm = 120.0
        bpm = float(min(300.0, max(20.0, bpm)))
        if len(beats) < 2:
            step = 60.0 / bpm
            beats = list(np.arange(0.0, duration - 1e-9, step))
        downbeats = [float(b) for b in (audio.get("downbeats") or []) if 0.0 <= float(b) < duration]
        if not downbeats:
            downbeats = beats[::4]
        return _r(bpm, 2), [_r(b) for b in beats], [_r(b) for b in downbeats]
    bpm = float(int(round(84 + float(np.clip(energy, 0, 1)) * (132 - 84))))
    if bpm_override:
        bpm = float(bpm_override)
    step = 60.0 / bpm
    beats = [_r(b) for b in np.arange(0.0, duration - 1e-9, step)]
    return bpm, beats, beats[::4]


_TOL = 1e-3   # beat times are rounded to 1e-4 s, so bar spacing can come out a hair short


def _greedy_bounds(n: int, duration: float, cands: list[float], min_len: float, weights=None,
                   prefer=None, beat: float = 0.5):
    """n scenes over [0, duration], each >= min_len, cuts on ``cands``. ``weights`` set each scene's share
    of the time (a text scene gets longer); cuts in ``prefer`` (downbeats) win near-ties."""
    w = list(weights) if weights else [1.0] * n
    cum = np.cumsum(w) / float(sum(w))
    prefer = prefer or set()
    bounds = [0.0]
    for k in range(1, n):
        target = duration * float(cum[k - 1])
        need_after = (n - k) * min_len
        ok = [c for c in cands if c >= bounds[-1] + min_len - _TOL and duration - c >= need_after - _TOL]
        if not ok:
            return None
        bounds.append(min(ok, key=lambda c: (abs(c - target) - (0.35 * beat if c in prefer else 0.0), c)))
    bounds.append(float(duration))
    if any(b - a < min_len - _TOL for a, b in zip(bounds, bounds[1:])):
        return None
    return bounds


def _scene_bounds(n_desired: int, duration: float, downbeats: list[float], bar: float,
                  beats: list[float] | None = None, weights_for=None, min_n: int = 1) -> list[float]:
    """Cut points for up to n_desired scenes. Tries downbeats (2 s+ preferred, then 1 bar), then half bars
    (>= 2 beats), then any beat (>= 1 beat) before giving up a scene, so every image gets its own scene
    whenever one beat each fits. ``weights_for(n)`` gives per-scene time weights."""
    cands = sorted({_r(d) for d in downbeats if 1e-3 < d < duration - 1e-3})
    beats = sorted({_r(b) for b in (beats or []) if 1e-3 < b < duration - 1e-3})
    beat = bar / 4.0
    half = []
    if beats:
        d0 = downbeats[0] if downbeats else 0.0
        j0 = int(np.argmin([abs(b - d0) for b in beats])) if beats else 0
        half = sorted(set(cands) | {b for j, b in enumerate(beats) if (j - j0) % 2 == 0})
    pref = max(bar, PREFERRED_SCENE_LEN)
    tiers = [(cands, pref), (cands, bar)]
    if beats:
        tiers += [(half, 2.0 * beat), (sorted(set(beats) | set(cands)), beat)]
    for n in range(max(1, n_desired), max(1, min_n) - 1, -1):
        if n == 1:
            break
        wts = weights_for(n) if weights_for else None
        for cs, min_len in tiers:
            b = _greedy_bounds(n, duration, cs, min_len, wts, set(cands), beat)
            if b is not None:
                return b
    return [0.0, float(duration)]


def _end_card_start(duration: float, downbeats: list[float], beats: list[float], beat: float) -> float | None:
    """Start of the end card: the downbeat where the synth plays its final hit (the last downbeat leaving
    >= 1 s, see synth.render_soundtrack), so the cut, the held title and the music's stop coincide."""
    dbs = [d for d in downbeats if 0.0 <= d < duration]
    cands = [d for d in dbs if d <= duration - 1.0]
    if len(dbs) >= 3 and len(cands) >= 2:
        return float(cands[-1])
    bs = [b for b in beats if 0.0 < b <= duration - 1.2]
    if bs and bs[-1] >= 0.4 * duration:
        return float(bs[-1])
    return None


_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "tonight", "today",
         "tomorrow", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat")
_MONTHS = ("january", "february", "march", "april", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
           "dec")
_TIME_RE = re.compile(r"^(\d{1,2}[:.]\d{2}(am|pm|h)?|\d{1,2}(am|pm)|\d{1,2}(st|nd|rd|th)|"
                      r"\d{1,2}/\d{1,2}(/\d{2,4})?|noon|midnight|(19|20)\d{2})$")


def _is_when(word: str) -> bool:
    w = _fold(word).strip(".,;:!?()[]'\"-")
    if not w:
        return False
    if w in _DAYS or w in _MONTHS or _TIME_RE.match(w):
        return True
    return w.endswith("s") and w[:-1] in _DAYS[:7]          # "fridays"


def _key_phrase(line: str) -> tuple[str, str]:
    """(key, rest): the day/time/date span of an event line ("Jazz night Friday 9pm" -> ("Friday 9pm",
    "Jazz night")), else (line, "")."""
    words = line.split()
    idx = [i for i, w in enumerate(words) if _is_when(w)]
    if not idx:
        return line, ""
    i0, i1 = idx[0], idx[-1]
    if i1 - i0 > 4:                                        # scattered tokens: not one phrase
        return line, ""
    key = " ".join(words[i0:i1 + 1]).strip(" ,;:-")
    rest = " ".join(words[:i0] + words[i1 + 1:]).strip(" ,;:-")
    if not rest or not key:
        return line, ""
    return key, rest


def _first_beat_after(beats: list[float], t: float, limit: float, fallback: float) -> float:
    for b in beats:
        if b >= t - 1e-6:
            return b if b <= limit else fallback
    return fallback


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def _region_boxes(img: dict, focus, zoom: float) -> dict | None:
    """Headline box per format from the image's largest usable empty region."""
    regions = img.get("empty_regions") or []
    if not regions:
        return None
    out = {}
    for fmt, (fw, fh) in FORMATS.items():
        crop = geometry.cover_crop(int(img.get("w") or 1), int(img.get("h") or 1), fw, fh,
                                   tuple(focus), zoom)
        sx0, sy0, sx1, sy1 = SAFE[fmt]
        chosen = None
        for r in regions:
            x, y, w, h = geometry.rect_image_to_frame(r, crop)
            x0, y0 = max(x + 0.02, sx0), max(y + 0.02, sy0)
            x1, y1 = min(x + w - 0.02, sx1), min(y + h - 0.02, sy1)
            if x1 - x0 >= MIN_TEXT_W[fmt] and y1 - y0 >= 0.07:
                chosen = (x0, y0, x1, y1)
                break
        if chosen is None:
            return None
        x0, y0, x1, y1 = chosen
        h = min(y1 - y0, MAX_TEXT_H[fmt])
        cy = (y0 + y1) / 2
        out[fmt] = _clip_box([x0, cy - h / 2, x1 - x0, h])
    return out


def _sub_box(head: list[float], fmt: str) -> tuple[list[float], list[float]]:
    x, y, w, h = head
    sh = SUB_H[fmt]
    _, sy0, _, sy1 = SAFE[fmt]
    if y + h + 0.01 + sh <= sy1:
        return head, _clip_box([x, y + h + 0.01, w, sh])
    if y - 0.01 - sh >= sy0:
        return head, _clip_box([x, y - 0.01 - sh, w, sh])
    hh = max(0.05, h - sh - 0.01)
    return _clip_box([x, y, w, hh]), _clip_box([x, y + hh + 0.01, w, sh])


# Candidate headline boxes when the photo has no empty region (frame space, per format).
AWAY_BOXES = {
    "9:16": [[0.08, 0.72, 0.84, 0.16], [0.08, 0.08, 0.84, 0.16], [0.08, 0.40, 0.84, 0.16]],
    "16:9": [[0.06, 0.66, 0.56, 0.22], [0.38, 0.66, 0.56, 0.22], [0.06, 0.10, 0.56, 0.22],
             [0.38, 0.10, 0.56, 0.22]],
}


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _away_boxes(img: dict, focus, zoom: float) -> dict:
    """Headline box per format on the side away from the subject: the candidate box that covers the least
    of the saliency box (mapped through the crop), then the one farthest from the saliency peak."""
    sal = img.get("saliency_box") or [0.25, 0.25, 0.5, 0.5]
    out = {}
    for fmt, (fw, fh) in FORMATS.items():
        crop = geometry.cover_crop(int(img.get("w") or 1), int(img.get("h") or 1), fw, fh, tuple(focus), zoom)
        sb = geometry.rect_image_to_frame(sal, crop)
        fx, fy = geometry.image_to_frame(tuple(focus), crop)

        def cost(j, b):
            cover = _overlap(b, sb) / max(b[2] * b[3], 1e-6)
            dist = float(np.hypot(b[0] + b[2] / 2 - fx, b[1] + b[3] / 2 - fy))
            return (round(cover, 2) - 0.25 * dist, j)

        boxes = AWAY_BOXES[fmt]
        j = min(range(len(boxes)), key=lambda k: cost(k, boxes[k]))
        out[fmt] = _clip_box(boxes[j])
    return out


def _contrast(a, b) -> float:
    la, lb = _luma(a) + 0.05, _luma(b) + 0.05
    return max(la, lb) / min(la, lb)


def _region_mean(img: dict, boxes: dict | None, focus, zoom: float) -> list[np.ndarray]:
    """Mean RGB of the photo under each format's text box (empty when pixels are unavailable)."""
    px = img.get("pixels")
    out = []
    if px is None or not boxes:
        return out
    px = np.asarray(px)
    if px.ndim != 3 or px.shape[0] < 2 or px.shape[1] < 2:
        return out
    H, W = px.shape[:2]
    for fmt, (fw, fh) in FORMATS.items():
        b = boxes.get(fmt)
        if not b:
            continue
        crop = geometry.cover_crop(int(img.get("w") or W), int(img.get("h") or H), fw, fh, tuple(focus), zoom)
        x0, y0 = geometry.frame_to_image((b[0], b[1]), crop)
        x1, y1 = geometry.frame_to_image((b[0] + b[2], b[1] + b[3]), crop)
        c0, c1 = int(np.clip(x0 * W, 0, W - 1)), int(np.clip(np.ceil(x1 * W), 1, W))
        r0, r1 = int(np.clip(y0 * H, 0, H - 1)), int(np.clip(np.ceil(y1 * H), 1, H))
        step = max(1, int(max(r1 - r0, c1 - c0) // 64))
        reg = px[r0:max(r1, r0 + 1):step, c0:max(c1, c0 + 1):step, :3].astype(np.float64)
        if reg.size:
            m = reg.reshape(-1, 3).mean(0)
            out.append(m / 255.0 if m.max() > 1.5 else m)
    return out


def _plate_colors(img: dict, palette: list[str], boxes: dict | None = None, focus=(0.5, 0.5),
                  zoom: float = 1.0) -> tuple[str, str]:
    """(plate, text) palette refs: the palette accent nearest this image's accent that still stands apart
    from the photo under the box (else the dark palette colour), and whichever of the dark / light palette
    colours reads best on it."""
    n = len(palette)
    if n <= 2:
        return "palette:0", "palette:1"
    target = None
    try:
        target = _rgb(img.get("accent") or "")
        if target.max() < 0.85:
            target = target * (0.85 / max(target.max(), 1e-6))
    except (ValueError, TypeError, IndexError):
        target = None
    if target is None:
        order = list(range(2, n))
    else:
        order = sorted(range(2, n), key=lambda i: (float(np.linalg.norm(_rgb(palette[i]) - target)), i))
    under = _region_mean(img, boxes, focus, zoom)
    j = next((i for i in order if all(np.linalg.norm(_rgb(palette[i]) - u) >= 0.35 for u in under)), 0)
    plate = _rgb(palette[j])
    text = 0 if _contrast(plate, _rgb(palette[0])) >= _contrast(plate, _rgb(palette[1])) else 1
    return f"palette:{j}", f"palette:{text}"


def _good_line(img: dict, ln: dict) -> bool:
    """A Hough line worth drawing: long, along the image's long direction, and clear of the subject."""
    iw, ih = float(img.get("w") or 1), float(img.get("h") or 1)
    try:
        (x0, y0), (x1, y1) = ln["p0"], ln["p1"]
    except (KeyError, TypeError, ValueError):
        return False
    dx, dy = (x1 - x0) * iw, (y1 - y0) * ih
    length = float(np.hypot(dx, dy)) / max(iw, ih)
    if length < 0.35:
        return False
    ang = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
    horiz = min(ang, 180.0 - ang) <= 8.0
    vert = abs(ang - 90.0) <= 8.0
    if not ((horiz and iw >= ih) or (vert and ih > iw)):
        return False
    sx, sy, sw, sh = img.get("saliency_box") or [0.0, 0.0, 0.0, 0.0]
    for t in np.linspace(0.0, 1.0, 21):              # any point inside the subject box -> it crosses it
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        if sx <= px <= sx + sw and sy <= py <= sy + sh:
            return False
    return True


def _contour_hits_boxes(img: dict, c: dict, focus, zoom: float, boxes: dict | None) -> bool:
    """True when the outline, mapped into either format's frame, runs through a text box."""
    pts = c.get("points") or []
    if not boxes or len(pts) < 2:
        return False
    for fmt, (fw, fh) in FORMATS.items():
        b = boxes.get(fmt)
        if not b:
            continue
        crop = geometry.cover_crop(int(img.get("w") or 1), int(img.get("h") or 1), fw, fh, tuple(focus), zoom)
        p = np.asarray(pts, np.float64)
        if c.get("closed", True):
            p = np.vstack([p, p[:1]])
        seg = np.linspace(0.0, 1.0, 6)[:, None, None]
        dense = (p[:-1][None] * (1 - seg) + p[1:][None] * seg).reshape(-1, 2)
        fx = (dense[:, 0] - crop[0]) / crop[2]
        fy = (dense[:, 1] - crop[1]) / crop[3]
        if np.any((fx >= b[0] - 0.01) & (fx <= b[0] + b[2] + 0.01) & (fy >= b[1] - 0.01) & (fy <= b[1] + b[3] + 0.01)):
            return True
    return False


def _pick_contours(contours: list[dict], energy: float, img: dict | None = None, focus=None, zoom: float = 1.0,
                   boxes: dict | None = None) -> list[dict]:
    """Only the largest salient subject outlines: salience >= 0.6 and edge support >= 0.5 (analysis), a
    sensible size, not through the headline, at most two."""
    ok = [c for c in contours
          if float(c.get("salience", 1.0) or 0.0) >= 0.6 and float(c.get("edge_support", 1.0) or 0.0) >= 0.5
          and 0.02 <= float(c.get("area", 0.1) or 0.0) <= 0.6
          and not (img is not None and _contour_hits_boxes(img, c, focus or [0.5, 0.5], zoom, boxes))]
    if not ok:
        return []
    first = ok[0]
    picked = [first]
    if energy >= 0.6:
        for c in ok[1:]:
            if (float(c.get("salience", 1.0) or 0.0) >= 0.8
                    and float(c.get("area", 0.0) or 0.0) >= 0.2 * float(first.get("area", 0.0) or 0.0)):
                picked.append(c)
                break
    return picked


def _assign_lines(lines: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        return []
    if len(lines) >= n:
        return [[lines[int(j)] for j in c] for c in np.array_split(np.arange(len(lines)), n)]
    out = [[] for _ in range(n)]
    for j, ln in enumerate(lines):
        out[int(j * n / len(lines))].append(ln)
    return out


def _end_card_text(lines: list[str]) -> tuple[str, str]:
    """(main, small): the event's day/time with its name, else the headline alone."""
    for ln in lines:
        if any(_is_when(w) for w in ln.split()):
            key, rest = _key_phrase(ln)
            if not rest:
                key = ln
            small = rest if ln == lines[0] else lines[0]
            return key, (small if small != key else "")
    return lines[0], ""


def _line_hits_boxes(img: dict, ln: dict, focus, zoom: float, boxes: dict | None) -> bool:
    """True when the line, mapped into either format's frame, runs through a text box."""
    if not boxes:
        return False
    try:
        (x0, y0), (x1, y1) = ln["p0"], ln["p1"]
    except (KeyError, TypeError, ValueError):
        return True
    for fmt, (fw, fh) in FORMATS.items():
        b = boxes.get(fmt)
        if not b:
            continue
        crop = geometry.cover_crop(int(img.get("w") or 1), int(img.get("h") or 1), fw, fh, tuple(focus), zoom)
        bx0, by0, bx1, by1 = b[0] - 0.02, b[1] - 0.02, b[0] + b[2] + 0.02, b[1] + b[3] + 0.02
        for t in np.linspace(0.0, 1.0, 41):
            px, py = geometry.image_to_frame((x0 + t * (x1 - x0), y0 + t * (y1 - y0)), crop)
            if bx0 <= px <= bx1 and by0 <= py <= by1:
                return True
    return False


INSET_BOXES = {"9:16": [[0.52, 0.08, 0.40, 0.26], [0.08, 0.08, 0.40, 0.26]],
               "16:9": [[0.66, 0.08, 0.28, 0.38], [0.06, 0.08, 0.28, 0.38]]}


def _inset_layer(img: dict, k: int, t0: float, t1: float, beat: float, head_boxes) -> dict:
    """An extra photo shown as a rounded inset in the second half of a scene (used only when there are
    more photos than one-beat scenes)."""
    box = {}
    for fmt in FORMATS:
        cands = INSET_BOXES[fmt]
        hb = (head_boxes or {}).get(fmt)
        order = sorted(range(len(cands)), key=lambda j: ((_overlap(cands[j], hb) if hb else 0.0), (j + k) % 2))
        box[fmt] = _clip_box(cands[order[0]])
    st0 = _r(min(t1 - 0.3, t0 + max(beat, 0.4 * (t1 - t0))))
    return {"type": "image", "src": img["id"], "mask": "rounded", "box": box, "t0": st0, "t1": _r(t1),
            "in": _anim("scale_pop", min(0.35, (t1 - st0) * 0.5)), "out": _anim("fade", 0.15),
            "emphasis": [], "border": {"color": "palette:1", "width": 0.004}, "treatment": None}


END_MAIN = {"9:16": [0.12, 0.42, 0.76, 0.10], "16:9": [0.22, 0.39, 0.56, 0.13]}
END_SMALL = {"9:16": [0.18, 0.535, 0.64, 0.045], "16:9": [0.3, 0.55, 0.4, 0.065]}
END_BAR = {"9:16": [0.42, 0.395, 0.16, 0.006], "16:9": [0.45, 0.36, 0.1, 0.008]}
END_IN = {"snappy": ("slide_up", 0.3), "smooth": ("fade", 0.4), "bouncy": ("scale_pop", 0.35),
          "glitch": ("cut", 0.0)}


def _end_card(idx: int, t0: float, t1: float, text: tuple[str, str], hero: dict, beat: float, beats,
              motion: str, entry: dict, accents: list[str]) -> dict:
    """The closing card: the hero photo, blurred and dark, held still; the key phrase lands on the final
    hit and stays put (no pulse, no exit) while the music rings out; a small accent rule and the name
    follow one beat later."""
    main, small = text
    display = entry["fonts"][0]
    acc = accents[0] if accents else "palette:1"
    verb, din = END_IN[motion]
    span = t1 - t0
    t_next = _first_beat_after(beats, t0 + 0.5 * beat, t1 - 0.4, _r(t0 + min(0.3, span / 3)))
    layers = [
        {"type": "shape", "source": "primitive", "ref": None, "primitive": "bar",
         "box": {f: _clip_box(END_BAR[f]) for f in FORMATS}, "stroke": None, "stroke_width": 0.006,
         "fill": acc, "fill_opacity": 1.0, "offset": [0.0, 0.0], "scale": 1.0, "t0": _r(t_next), "t1": _r(t1),
         "in": _anim("scale_pop", min(0.3, (t1 - t_next) * 0.5)), "out": _anim("cut", 0.0), "emphasis": [],
         "echo": None},
        {"type": "text", "text": main, "role": "headline", "font": "display",
         "case": _headline_case(display), "color": "palette:1", "highlight": None,
         "align": "center", "box": {f: _clip_box(END_MAIN[f]) for f in FORMATS}, "t0": _r(t0), "t1": _r(t1),
         "in": _anim(verb, min(din, span * 0.3)), "out": _anim("cut", 0.0), "emphasis": []},
    ]
    if small:
        layers.append({"type": "text", "text": small, "role": "sub", "font": "body", "case": "upper",
                       "color": acc, "highlight": None, "align": "center",
                       "box": {f: _clip_box(END_SMALL[f]) for f in FORMATS}, "t0": _r(t_next), "t1": _r(t1),
                       "in": _anim("fade", min(0.3, (t1 - t_next) * 0.4)), "out": _anim("cut", 0.0),
                       "emphasis": []})
    return {"id": f"s{idx}", "t0": _r(t0), "t1": _r(t1),
            "background": {"type": "image", "src": hero["id"], "color": None, "colors": None, "angle": 0,
                           "focus": [_r(v) for v in (hero.get("focus") or [0.5, 0.5])],
                           "motion": {"verb": "static"},
                           "treatment": {"duotone": None, "blur": 0.5, "darken": 0.6}},
            "layers": layers, "transition_out": {"type": "cut", "dur": 0.0}}


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def plan_from_analysis(analysis: dict, seed: int = 0, duration: float | None = None) -> dict:
    rng = np.random.default_rng(int(seed))
    analysis = analysis or {}
    images = list(analysis.get("images") or [])
    lines = [s.strip() for s in (analysis.get("text_lines") or []) if str(s).strip()]
    audio_in = analysis.get("audio")
    mood_in = analysis.get("mood") or {}
    energy_in = float(np.clip(float(_num_or(mood_in.get("energy"), 0.5)), 0.0, 1.0))
    valence = float(np.clip(float(_num_or(mood_in.get("valence"), 0.5)), 0.0, 1.0))

    # ---- duration
    dur = None
    for cand in (duration, analysis.get("duration_hint"), (audio_in or {}).get("duration")):
        if cand is not None and float(cand) > 0:
            dur = float(cand)
            break
    if dur is None:
        dur = DEFAULT_DUR
    dur = _r(float(np.clip(dur, MIN_DUR, MAX_DUR)), 3)

    # ---- mood, style
    mood_name, matched = _pick_mood(analysis, energy_in, valence)
    entry = MOOD_TABLE[mood_name]
    energy = 0.7 * energy_in + 0.3 * entry["energy"] if matched else energy_in
    energy = _r(float(np.clip(energy, 0.0, 1.0)), 3)
    entry = dict(entry, fonts=pick_fonts(FONT_MOODS.get(mood_name, ["hype"]), energy, seed, mood_name))
    motion = entry["motion"]
    palette = _image_palette(images, entry["palette"]) if images else list(entry["palette"])
    n_pal = len(palette)
    accents = [f"palette:{i}" for i in range(2, n_pal)]

    bpm, beats, downbeats = _beat_grid(analysis, dur, energy)
    if images and lines and not audio_in:
        # synth tempo: nudge (<= 8 %) so the final hit leaves a 1.4-2.2 s end card, not a 3 s still
        for cand in sorted(range(int(round(bpm * 0.92)), int(round(bpm * 1.08)) + 1),
                           key=lambda b: (abs(b - bpm), b)):
            _, bs, ds = _beat_grid(analysis, dur, energy, cand)
            te = _end_card_start(dur, ds, bs, 60.0 / cand)
            if te is not None and 1.4 - _TOL <= dur - te <= 2.2 + _TOL:
                bpm, beats, downbeats = _beat_grid(analysis, dur, energy, cand)
                break
    beat = 60.0 / bpm
    bar = 4.0 * beat

    # ---- scenes
    if images:
        n_desired = max(len(images), len(lines))
    elif lines:
        n_desired = len(lines)
    else:
        n_desired = int(min(3, max(1, round(dur / 4.0))))
    # end card (photos + text): the key phrase held still from the music's final hit to the end
    t_body = dur
    end_text = None
    if images and lines:
        t_end = _end_card_start(dur, downbeats, beats, beat)
        if t_end is not None and t_end >= min(len(images), n_desired) * beat - _TOL:
            t_body, end_text = t_end, _end_card_text(lines)

    def weights_for(k):        # text scenes get more time so the headline can be read
        return [1.6 if (i == 0 or tl) and lines else 1.0 for i, tl in enumerate(_assign_lines(lines, k))]

    bounds = _scene_bounds(n_desired, t_body, downbeats, bar, beats, weights_for if images else None)
    n = len(bounds) - 1
    scene_lines = _assign_lines(lines, n)
    if images and len(lines) == 1:
        key, rest = _key_phrase(lines[0])
        if rest:                   # event line: the name as the headline, the day/time on its own line
            scene_lines[0] = [rest, key]
    insets = [[] for _ in range(n)]
    if images and len(images) > n:  # more photos than one-beat scenes: the rest go in as insets
        for j, im in enumerate(images[n:]):
            insets[(j + 1) % n if n > 1 else 0].append(im)

    high = energy >= 0.5 and motion != "smooth"
    scenes = []
    prev_prim = None
    bg_kind_start = int(rng.integers(0, 2))
    for i in range(n):
        t0, t1 = bounds[i], bounds[i + 1]
        slen = t1 - t0
        sc = {"id": f"s{i}", "t0": _r(t0), "t1": _r(t1), "background": None, "layers": [],
              "transition_out": {"type": "cut", "dur": 0.0}}
        layers = sc["layers"]
        texts = scene_lines[i]
        text_t0 = _first_beat_after(beats, t0 + 0.2, t1 - 1.0, _r(t0 + min(0.1, slen / 4)))
        head_boxes = None
        highlight = None
        text_color = "palette:1"

        if images:
            img = images[i % len(images)]
            focus = [_r(v) for v in (img.get("focus") or [0.5, 0.5])]
            bg_motion = _bg_motion(rng, img, energy)
            zoom0 = float(bg_motion.get("zoom", [1.0])[0])
            darken = float(np.clip(0.15 + 0.35 * float(img.get("luma_mean", 0.5)), 0.1, 0.45)) if texts else 0.1
            duo = ["palette:0", accents[0]] if float(img.get("saturation", 0.5)) < 0.12 and accents else None
            sc["background"] = {"type": "image", "src": img["id"], "color": None, "colors": None, "angle": 0,
                                "focus": focus, "motion": bg_motion,
                                "treatment": {"duotone": duo, "blur": 0.0, "darken": _r(darken, 3)}}
            if texts:
                head_boxes = _region_boxes(img, focus, zoom0)
                if head_boxes is None:   # no empty area: an accent plate on the side away from the subject
                    head_boxes = _away_boxes(img, focus, zoom0)
                    highlight, text_color = _plate_colors(img, palette, head_boxes, focus, zoom0)
            # contour outlines: only the largest salient subject outline(s)
            contours = list(img.get("contours") or [])
            if contours:
                picked = _pick_contours(contours, energy, img, focus, zoom0, head_boxes)
                for ci, c in enumerate(picked):
                    st0 = _r(min(t0 + ci * 0.5 * beat, t0 + slen * 0.5))
                    draw = float(np.clip(2 * beat, 0.6, 1.4))
                    draw = min(draw, (t1 - st0) * 0.6)
                    layers.append({
                        "type": "shape", "source": "contour", "ref": c["id"], "primitive": None, "box": None,
                        "stroke": accents[ci % len(accents)] if accents else "palette:1",
                        "stroke_width": 0.006 if ci == 0 else 0.004,
                        "fill": None, "fill_opacity": 0.0, "offset": [0.0, 0.0], "scale": 1.0,
                        "t0": st0, "t1": _r(t1),
                        "in": _anim("stroke_draw", draw),
                        "out": _anim("cut" if motion == "glitch" else "fade", 0.0 if motion == "glitch" else 0.2),
                        "emphasis": ([{"verb": "pulse", "on": "beats", "amount": _r(0.04 + 0.06 * energy, 3)}]
                                     if ci == 0 and energy >= 0.45 else []),
                        "echo": ({"count": 2, "spacing": 0.015, "decay": 0.5} if ci == 0
                                 else {"count": 1, "spacing": 0.012, "decay": 0.5}),
                    })
            # one strong line as a slash / underline element
            img_lines = [ln for ln in (img.get("lines") or [])
                         if _good_line(img, ln) and not _line_hits_boxes(img, ln, focus, zoom0, head_boxes)]
            has_line = False
            if img_lines:
                li = max(range(len(img_lines)), key=lambda j: (float(img_lines[j].get("strength", 0.0) or 0.0), -j))
                ln = img_lines[li]
                li = (img.get("lines") or []).index(ln)
                if float(ln.get("strength", 1.0) or 0.0) >= 0.2 or "strength" not in ln:
                    has_line = True
                    lt0 = text_t0 if texts else _r(t0 + min(beat, slen / 3))
                    layers.append({
                        "type": "shape", "source": "line", "ref": ln.get("id") or f"{img['id']}:l{li}",
                        "primitive": None, "box": None,
                        "stroke": accents[1 % len(accents)] if accents else "palette:1", "stroke_width": 0.008,
                        "fill": None, "fill_opacity": 0.0, "offset": [0.0, 0.0], "scale": 1.0,
                        "t0": lt0, "t1": _r(t1),
                        "in": _anim("stroke_draw", min(0.5, (t1 - lt0) * 0.4)),
                        "out": _anim("fade", 0.2), "emphasis": [], "echo": None,
                    })
            for k, im in enumerate(insets[i]):
                layers.append(_inset_layer(im, k, t0, t1, beat, head_boxes))
        else:
            kind = "shapes" if (not lines or (i + bg_kind_start) % 2) else "gradient"
            if kind == "gradient":
                acc = _rgb(palette[2 + (i % max(1, n_pal - 2))]) if n_pal > 2 else _rgb(palette[1])
                mid = _hex(0.55 * _rgb(palette[0]) + 0.45 * acc)
                sc["background"] = {"type": "gradient", "src": None, "color": None,
                                    "colors": ["palette:0", mid], "angle": int(rng.integers(0, 360)),
                                    "focus": [0.5, 0.5], "motion": {"verb": "drift"},
                                    "treatment": {"duotone": None, "blur": 0.0, "darken": 0.0}}
            else:
                sc["background"] = {"type": "shapes", "src": None, "color": "palette:0",
                                    "colors": accents[:2] or ["palette:1"], "angle": 0,
                                    "focus": [0.5, 0.5], "motion": {"verb": "drift"},
                                    "treatment": {"duotone": None, "blur": 0.0, "darken": 0.0}}
            if texts:
                head_boxes = {f: list(CENTRE_BOX[f]) for f in FORMATS}
            if not lines:
                # audio-only / empty: a pulsing ring + audio-reactive wave
                layers.append(_primitive("ring", head_boxes, accents, t0, t1, energy, rng))
                layers.append(_primitive("wave", head_boxes, accents, t0, t1, energy, rng))
            else:
                pool = [p for p in ("ring", "bar", "dot_field", "grid", "cross", "circle") if p != prev_prim]
                prim = str(rng.choice(pool))
                prev_prim = prim
                dim = _hex(0.6 * _rgb(palette[0]) + 0.4 * _rgb(palette[2 if n_pal > 2 else 1]))
                layers.append(_primitive(prim, head_boxes, accents, t0, t1, energy, rng, dim=dim))
            has_line = False

        if texts:
            _text_layers(layers, texts, head_boxes, highlight, t0, t1, text_t0, beat, beats, energy, motion,
                         entry, rng, has_line, bool(images), text_color)
        scenes.append(sc)
    if end_text is not None:
        scenes.append(_end_card(len(scenes), t_body, dur, end_text, images[0], beat, beats, motion, entry, accents))

    # ---- transitions + sfx + risers
    sfx, risers = [], []
    prev_tr = None
    for i in range(len(scenes) - 1):
        b = scenes[i]["t1"]
        if end_text is not None and i == len(scenes) - 2:
            # hard cut onto the end card, on the final hit
            sfx.append({"t": _r(b), "type": "impact", "gain": _r(0.6 + 0.3 * energy, 3)})
            if energy >= 0.5:
                risers.append({"t0": _r(max(0.0, b - bar)), "t1": _r(b)})
            continue
        pool = HIGH_ENERGY_TRANSITIONS if high else LOW_ENERGY_TRANSITIONS
        pool = [p for p in pool if p != prev_tr] or pool
        typ = str(rng.choice(pool))
        prev_tr = typ
        dur_t = float(rng.uniform(0.25, 0.35)) if high else float(rng.uniform(0.5, 0.8))
        shortest = min(scenes[i]["t1"] - scenes[i]["t0"], scenes[i + 1]["t1"] - scenes[i + 1]["t0"])
        dur_t = _r(min(dur_t, 0.5 * shortest), 3)
        tr = {"type": typ, "dur": dur_t}
        if typ in ("push", "wipe"):
            tr["dir"] = str(rng.choice(["left", "right", "up", "down"]))
            sfx.append({"t": _r(max(0.0, b - dur_t / 2)), "type": "whoosh", "gain": 0.6})
        scenes[i]["transition_out"] = tr
        sfx.append({"t": _r(b), "type": "impact", "gain": _r(0.6 + 0.3 * energy, 3)})
        if energy >= 0.5:
            risers.append({"t0": _r(max(0.0, b - bar)), "t1": _r(b)})
    sfx.sort(key=lambda s: s["t"])

    # ---- audio block
    music = entry["music"]
    if not matched:
        music = _music_mood(energy, valence)
    scale = entry["scale"] if matched else ("major" if valence >= 0.5 else "minor")
    key = str(rng.choice(entry["keys"]))
    sections = _sections(audio_in, scenes, dur, downbeats)

    plan = {
        "version": PLAN_VERSION,
        "seed": int(seed),
        "duration": dur,
        "fps": FPS,
        "style": {
            "palette": palette,
            "background": "palette:0",
            "fonts": {"display": entry["fonts"][0], "body": entry["fonts"][1]},
            "motion": motion,
            "energy": energy,
            "grain": _r(0.06 + 0.12 * energy + (0.08 if motion == "glitch" else 0.0), 3),
            "vignette": 0.25,
            "beat_flash": _r(0.1 + 0.15 * (energy - 0.6) / 0.4, 3) if energy >= 0.6 else 0.0,
        },
        "audio": {
            "source": "user" if audio_in else "synth",
            "bpm": bpm, "key": key, "scale": scale, "mood": music,
            "beats": beats, "downbeats": downbeats,
            "sections": sections, "risers": risers, "sfx": sfx,
        },
        "voice": None,
        "scenes": scenes,
    }
    errors = validate_plan(plan, analysis)
    if errors:
        raise RuntimeError("rule planner produced an invalid plan (bug):\n" + "\n".join(errors))
    return plan


def _music_mood(energy: float, valence: float) -> str:
    if energy >= 0.7:
        return "epic" if valence >= 0.6 else "driving"
    if energy >= 0.4:
        if valence >= 0.6:
            return "playful"
        return "bright" if valence >= 0.45 else "dark"
    return "calm" if valence >= 0.4 else "dark"


def _sections(audio_in, scenes, dur, downbeats):
    if audio_in and audio_in.get("sections"):
        secs = sorted((float(s.get("t", 0.0)), float(s.get("energy", 0.0)))
                      for s in audio_in["sections"] if 0.0 <= float(s.get("t", 0.0)) < dur)
        if secs:
            if secs[0][0] > 1e-3:
                secs.insert(0, (0.0, 0.0))
            peak = max(range(len(secs)), key=lambda j: (secs[j][1], -j))
            out = []
            for j, (t, _) in enumerate(secs):
                label = "intro" if j == 0 else "drop" if j == peak else "build" if j < peak else "break"
                out.append({"t": _r(t), "label": label})
            return out
    out = [{"t": 0.0, "label": "intro"}]
    starts = [s["t0"] for s in scenes[1:]]
    if starts:
        drop = min(starts, key=lambda t: (abs(t - 0.4 * dur), t))
    else:
        inner = [d for d in downbeats if 0 < d < dur - 1.0]
        drop = min(inner, key=lambda t: (abs(t - 0.4 * dur), t)) if inner else None
    if drop is not None:
        out.append({"t": _r(drop), "label": "drop"})
    if len(scenes) >= 4 and scenes[-1]["t0"] > (drop or 0.0):
        out.append({"t": _r(scenes[-1]["t0"]), "label": "outro"})
    return out


def _bg_motion(rng, img: dict, energy: float) -> dict:
    sal = img.get("saliency_box") or [0.25, 0.25, 0.5, 0.5]
    sal_area = float(sal[2]) * float(sal[3])
    if energy >= 0.65:
        pool = ["beat_zoom", "ken_burns"] + (["parallax_split"] if 0.05 <= sal_area <= 0.6 else [])
    else:
        pool = ["ken_burns", "drift"]
    verb = str(rng.choice(pool))
    dx, dy = (float(v) for v in rng.uniform(-0.03, 0.03, size=2))
    if verb == "ken_burns":
        z1 = 1.06 + 0.08 * energy
        zoom = [1.0, _r(z1, 3)] if rng.random() < 0.6 else [_r(z1, 3), 1.0]
        return {"verb": verb, "zoom": zoom, "pan": [[0.0, 0.0], [_r(dx), _r(dy)]]}
    if verb == "drift":
        return {"verb": verb, "zoom": [1.05, 1.05], "pan": [[0.0, 0.0], [_r(dx), _r(dy)]]}
    if verb == "beat_zoom":
        return {"verb": verb, "zoom": [1.0, 1.05], "amount": _r(0.03 + 0.04 * energy, 3)}
    return {"verb": verb, "zoom": [1.0, 1.04], "amount": 0.03}


def _primitive(prim: str, head_boxes, accents, t0, t1, energy, rng, dim: str | None = None) -> dict:
    """One primitive shape layer. ``dim`` (a muted hex) is used for the full-frame grid / dot field."""
    a0 = accents[0] if accents else "palette:1"
    a1 = accents[1 % len(accents)] if accents else "palette:1"
    box, stroke, fill, fop, sw = {}, a0, None, 0.0, 0.006
    in_v, emph = _anim("fade", 0.4), []
    for fmt in FORMATS:
        hb = (head_boxes or {}).get(fmt) or CENTRE_BOX[fmt]
        cy = hb[1] + hb[3] / 2
        if prim in ("ring", "circle"):
            box[fmt] = _square_box(fmt, 0.5, cy, 0.62)
        elif prim == "bar":
            box[fmt] = _clip_box([0.3, min(0.93, hb[1] + hb[3] + 0.01), 0.4, 0.012])
        elif prim in ("dot_field", "grid"):
            box[fmt] = [0.0, 0.0, 1.0, 1.0] if prim == "grid" else [0.05, 0.05, 0.9, 0.9]
        elif prim == "cross":
            box[fmt] = _square_box(fmt, 0.86, 0.12, 0.1)
        elif prim == "wave":
            box[fmt] = _clip_box([0.05, cy - 0.1, 0.9, 0.2]) if fmt == "9:16" else _clip_box([0.05, cy - 0.15, 0.9, 0.3])
        else:
            box[fmt] = _clip_box(CENTRE_BOX[fmt])
    if prim == "ring":
        in_v, sw = _anim("stroke_draw", 0.8), 0.008
        emph = [{"verb": "pulse", "on": "beats", "amount": _r(0.04 + 0.06 * energy, 3)}]
    elif prim == "circle":
        stroke, fill, fop, in_v = None, a1, 0.25, _anim("scale_pop", 0.45)
        emph = [{"verb": "pulse", "on": "beats", "amount": _r(0.03 + 0.05 * energy, 3)}]
    elif prim == "bar":
        stroke, fill, fop, in_v = None, a0, 1.0, _anim("scale_pop", 0.35)
    elif prim == "dot_field":
        stroke, fill, fop = None, dim or a1, 0.35
    elif prim == "grid":
        stroke, sw = dim or a1, 0.002
    elif prim == "cross":
        in_v = _anim("scale_pop", 0.35)
        emph = [{"verb": "pulse", "on": "downbeats", "amount": 0.1}]
    elif prim == "wave":
        stroke, sw, in_v = a0, 0.006, _anim("fade", 0.3)
    return {"type": "shape", "source": "primitive", "ref": None, "primitive": prim, "box": box,
            "stroke": stroke, "stroke_width": sw, "fill": fill, "fill_opacity": fop,
            "offset": [0.0, 0.0], "scale": 1.0, "t0": _r(t0), "t1": _r(t1),
            "in": in_v, "out": _anim("fade", 0.2), "emphasis": emph, "echo": None}


# in-verbs where a highlight plate grows with its words (the plate is wiped / moved / scaled together with
# the text); word_stagger, kinetic_split and type_on would show a full-size plate before the words
PLATE_IN_VERBS = ("mask_wipe", "slide_up", "scale_pop", "fade", "blur_in")


def _text_layers(layers, texts, head_boxes, highlight, t0, t1, text_t0, beat, beats, energy, motion,
                 entry, rng, has_line, on_image, text_color="palette:1"):
    head = texts[0]
    sub = " ".join(texts[1:]).strip()
    boxes_h, boxes_s = {}, {}
    for fmt in FORMATS:
        if sub:
            boxes_h[fmt], boxes_s[fmt] = _sub_box(head_boxes[fmt], fmt)
        else:
            boxes_h[fmt] = head_boxes[fmt]

    in_verb = str(rng.choice(TEXT_IN_BY_MOTION[motion]))
    if highlight is not None and in_verb not in PLATE_IN_VERBS:
        grow = [v for v in TEXT_IN_BY_MOTION[motion] if v in PLATE_IN_VERBS] or ["mask_wipe"]
        in_verb = grow[0] if "mask_wipe" not in grow else "mask_wipe"
    out_verb = str(rng.choice(TEXT_OUT_BY_MOTION[motion]))
    span = t1 - text_t0
    in_dur = min(IN_DUR[motion], span * 0.35)
    out_dur = 0.0 if out_verb == "cut" else min(0.3 if motion == "smooth" else 0.25, span * 0.2)
    extra = {}
    if in_verb == "word_stagger":
        n_words = max(1, len(head.split()))
        sb = 0.5
        if (n_words - 1) * sb * beat > 0.4 * span:
            sb = 0.25
        extra["stagger_beats"] = sb

    color, hl = (text_color if highlight is not None else "palette:1"), highlight
    if not on_image and energy >= 0.6 and rng.random() < 0.3:
        color, hl = "palette:0", "palette:2"

    emph = []
    if energy >= 0.6:
        emph.append({"verb": "pulse", "on": "beats", "amount": _r(0.03 + 0.05 * energy, 3)})
    if motion == "glitch":
        emph.append({"verb": "color_flip", "on": "downbeats", "amount": 1.0, "color": "palette:2"})
    elif not has_line and rng.random() < 0.4:
        emph.append({"verb": "underline_draw", "on": "downbeats", "amount": 1.0, "color": "palette:2"})

    display = entry["fonts"][0]
    layers.append({
        "type": "text", "text": head, "role": "headline", "font": "display",
        "case": _headline_case(display),
        "color": color, "highlight": hl, "align": "center", "box": boxes_h,
        "t0": _r(text_t0), "t1": _r(t1),
        "in": _anim(in_verb, in_dur, **extra), "out": _anim(out_verb, out_dur), "emphasis": emph,
    })
    if sub:
        st0 = _first_beat_after(beats, text_t0 + beat, t1 - 0.8, text_t0)
        sspan = t1 - st0
        layers.append({
            "type": "text", "text": sub, "role": "sub", "font": "body", "case": "as_is",
            "color": color if highlight is not None else "palette:1", "highlight": highlight,
            "align": "center", "box": boxes_s,
            "t0": _r(st0), "t1": _r(t1),
            "in": _anim("fade" if motion == "smooth" else "slide_up", min(0.4, sspan * 0.3)),
            "out": _anim("fade", min(0.2, sspan * 0.2)), "emphasis": [],
        })


# ==========================================================================
# brief planner: a VLM's small creative brief -> plan (decisions.md, "Planner architecture")
# ==========================================================================
# The brief (mograph.brief) carries taste only: family, palette, energy, bpm, arc, the user's words split
# into lines, hero image, text region, mood. Everything below is code: the arc becomes a shot list on
# the beat grid (bursts, rests, holds), the family becomes treatment, grounds, transitions, text verbs,
# type scale and one primitive; boxes, contrast and fonts are computed.

# family -> look. text: giant | headline | medium | small (type scale); prim: the piece's one primitive
BRIEF_LOOK: dict[str, dict] = {
    "film_soft": {"motion": "smooth", "music": "calm", "text": "small", "vignette": 0.3,
                  "trans": ["crossfade", "luma_wipe"], "tmax": 1.6, "prim": None},
    "film_hectic": {"motion": "snappy", "music": "driving", "text": "giant", "vignette": 0.25,
                    "trans": ["flash", "zoom_blur", "cut", "glitch"], "tmax": 0.2, "prim": "strobe"},
    "type_soft": {"motion": "smooth", "music": "calm", "text": "medium", "vignette": 0.12,
                  "trans": ["crossfade"], "tmax": 1.0, "prim": "underline"},
    "type_hectic": {"motion": "snappy", "music": "driving", "text": "giant", "vignette": 0.0,
                    "trans": ["flash", "cut", "cut"], "tmax": 0.12, "prim": "strobe"},
    "film_type_soft": {"motion": "smooth", "music": "calm", "text": "small", "vignette": 0.3,
                       "trans": ["crossfade"], "tmax": 1.6, "prim": "underline"},
    "film_type_hectic": {"motion": "snappy", "music": "driving", "text": "giant", "vignette": 0.25,
                         "trans": ["flash", "zoom_blur", "cut"], "tmax": 0.2, "prim": "strobe"},
    "paper_assembly": {"motion": "bouncy", "music": "playful", "text": "headline", "vignette": 0.15,
                       "trans": ["push", "wipe", "cut"], "tmax": 0.34, "prim": "cutouts"},
    "hype_print": {"motion": "snappy", "music": "driving", "text": "giant", "vignette": 0.0,
                   "trans": ["push", "zoom_blur", "flash", "cut"], "tmax": 0.24, "prim": "lane"},
    "ambient_glow": {"motion": "smooth", "music": "dark", "text": "small", "vignette": 0.5,
                     "trans": ["luma_wipe", "crossfade"], "tmax": 1.2, "prim": "glow"},
    "neon_loop": {"motion": "glitch", "music": "driving", "text": "headline", "vignette": 0.3,
                  "trans": ["wipe", "glitch", "push"], "tmax": 0.3, "prim": "neon"},
    "soft_loop": {"motion": "smooth", "music": "calm", "text": "medium", "vignette": 0.12,
                  "trans": ["crossfade", "shape_reveal"], "tmax": 1.2, "prim": "breath"},
    "xerox_zine": {"motion": "glitch", "music": "driving", "text": "giant", "vignette": 0.2,
                   "trans": ["cut", "glitch", "flash"], "tmax": 0.16, "prim": "cross"},
}

# type scale: (w, h) per format. 9:16 boxes are tall (pixel h >= 0.5 w) so display auto-fit stacks lines.
BRIEF_TEXT_WH = {"giant": {"9:16": (0.94, 0.42), "16:9": (0.94, 0.62)},
                 "headline": {"9:16": (0.84, 0.30), "16:9": (0.66, 0.42)},
                 "medium": {"9:16": (0.70, 0.20), "16:9": (0.50, 0.30)},
                 "small": {"9:16": (0.56, 0.17), "16:9": (0.40, 0.22)}}
BRIEF_TITLE = {"9:16": [0.2, 0.47, 0.6, 0.05], "16:9": [0.32, 0.46, 0.36, 0.07]}
STROBE_T = 0.067                                    # exposure strobe: 2 frames


def _q(d: float, step: float = 0.02) -> float:
    """Transition durations on a 0.02 s grid, so scene.t1 + dur/2 stays exact at 4 decimals."""
    return float(round(np.floor(max(0.0, d) / step + 1e-9) * step, 2))


def _mix(a: str, b: str, t: float) -> str:
    return _hex((1.0 - t) * _rgb(a) + t * _rgb(b))


def _split_len(total: float, size: float) -> list[float]:
    """``total`` beats as shots of ``size`` beats; a remainder under half a shot joins the last one."""
    if total <= 1e-9:
        return []
    if total < size:
        return [float(total)]
    n = int(np.floor(total / size + 1e-9))
    out = [float(size)] * n
    rem = total - n * size
    if rem > 1e-9:
        if rem >= 0.5 * size:
            out.append(rem)
        else:
            out[-1] += rem
    return out


def _soft_len(beat: float) -> int:
    k = 4
    while k * beat < 2.0:
        k += 4
    return k


def _arc_shots(arc: str, pace: str, nb: float, beat: float) -> list[tuple[float, str]]:
    """Shot list (length in beats, kind) filling ``nb`` beats. Kinds: hold, drop, burst, rest, dark, grow,
    peak, chapter, accel, loop."""
    soft = _soft_len(beat)
    base = {"soft": soft, "mid": 4 if 4 * beat <= 2.8 else 2, "hectic": 2}[pace]
    burst = (0.5 if 0.5 * beat >= 0.19 else 1.0) if pace == "hectic" else (1.0 if pace == "mid" else 2.0)
    shots: list[tuple[float, str]] = []
    if arc == "hold_drop_bursts" and nb >= 8:
        rest = float(max(1, int(round(0.9 / beat)))) if pace != "soft" else 0.0
        b_tot = (4.0 if nb >= 16 else 2.0) if pace != "soft" else float(min(6, max(2, int(nb // 4))))
        d_tot = (8.0 if nb - rest - b_tot - 8 >= 4 else 4.0) if pace != "soft" else float(min(soft, 4))
        hold = nb - rest - b_tot - d_tot
        if hold < 2:
            d_tot = max(0.0, d_tot + hold - 2)
            hold = nb - rest - b_tot - d_tot
        shots += [(x, "hold") for x in _split_len(hold, 4 if pace != "soft" else soft)]
        shots += [(x, "drop") for x in _split_len(d_tot, base if pace != "soft" else 4)]
        shots += [(burst, "burst")] * int(round(b_tot / burst))
        if rest:
            shots.append((rest, "rest"))
    elif arc == "dark_grow" and nb >= 6:
        g = 4 if nb >= 24 else 2
        d = max(g, g * round(0.3 * nb / g))
        gr = max(g, g * round(0.35 * nb / g))
        p = nb - d - gr
        if p < g:
            gr = max(g, gr - (g - p))
            p = nb - d - gr
        shots.append((float(d), "dark"))
        shots += [(x, "grow") for x in _split_len(gr, gr if pace == "soft" else 4)]
        shots += [(x, "peak") for x in _split_len(p, p if pace == "soft" else base)]
    elif arc == "acceleration" and nb >= 6:
        L = {"soft": 8.0, "mid": 4.0, "hectic": 4.0}[pace]
        lo = {"soft": 2.0, "mid": 1.0, "hectic": burst}[pace]
        rem, k = float(nb), 0
        while rem > 1e-9:
            ln = min(L, rem)
            if 1e-9 < rem - ln < lo:
                ln = rem
            shots.append((ln, "accel"))
            rem -= ln
            k += 1
            if k % 2 == 0:
                L = max(lo, L / 2.0)
    else:
        kind = "loop" if arc == "loop" else "chapter"
        size = {"soft": soft, "mid": 4, "hectic": 4}[pace] if kind == "chapter" else base
        shots = [(x, kind) for x in _split_len(nb, size)]
    return shots or [(float(nb), "chapter")]


def _words(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        out += [w.strip(",.;:!?") or w for w in ln.split()]
    return [w for w in out if w]


def _assign_text(shots: list[dict], lines: list[str], pace: str, family: str) -> None:
    """Sets shot["text"] (str | None) and shot["cont"] (the same words held over from the previous shot)."""
    for s in shots:
        s["text"], s["cont"] = None, False
    if not lines:
        return
    skip = ("burst", "rest", "dark") if pace == "hectic" else ("rest", "dark")
    msg = [i for i, s in enumerate(shots) if s["kind"] not in skip]
    if not msg:
        msg = [i for i, s in enumerate(shots) if s["kind"] != "rest"] or [0]
    if pace == "hectic":
        words = _words(lines)
        wi = 0
        holds = [i for i in msg if shots[i]["kind"] in ("hold", "chapter", "grow", "peak", "loop")
                 or (shots[i]["kind"] == "accel" and shots[i]["beats"] >= 2)]
        for k, i in enumerate(holds):
            shots[i]["text"] = lines[k % len(lines)] if len(holds) > 1 else "\n".join(lines[:3])
        for s in shots:
            if s["text"] is None and s["kind"] not in ("rest", "dark"):
                s["text"] = words[wi % len(words)]
                wi += 1
        return
    if family == "neon_loop":
        for k, i in enumerate(msg):
            shots[i]["text"] = lines[k % len(lines)]
        return
    if len(lines) > len(msg):
        groups = [list(g) for g in np.array_split(np.arange(len(lines)), len(msg))]
        for i, g in zip(msg, groups):
            shots[i]["text"] = "\n".join(lines[int(j)] for j in g)
        return
    for j, g in enumerate(np.array_split(np.arange(len(msg)), len(lines))):
        for k, gi in enumerate(g):
            i = msg[int(gi)]
            shots[i]["text"] = lines[j]
            shots[i]["cont"] = k > 0 and msg[int(gi) - 1] == i - 1


def _region_for(img: dict | None, fmt: str, region: str, style: str) -> str:
    """Brief text region -> top | middle | bottom | left | right for this format ("auto" reads the hero
    image's empty regions, else the side away from the subject)."""
    if region != "auto":
        return region if (fmt == "16:9" or region in ("top", "middle", "bottom")) else "middle"
    if img is None:
        return "bottom" if style == "small" else "middle"
    fw, fh = FORMATS[fmt]
    focus = tuple(img.get("focus") or (0.5, 0.5))
    crop = geometry.cover_crop(int(img.get("w") or 1), int(img.get("h") or 1), fw, fh, focus, 1.06)
    for r in img.get("empty_regions") or []:
        x, y, w, h = geometry.rect_image_to_frame(r, crop)
        if w * h < 0.06:
            continue
        cx, cy = x + w / 2, y + h / 2
        if fmt == "16:9" and abs(cx - 0.5) > 0.12 and h > 0.4:
            return "left" if cx < 0.5 else "right"
        return "top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle"
    sx, sy, sw, sh = geometry.rect_image_to_frame(img.get("saliency_box") or [0.25, 0.25, 0.5, 0.5], crop)
    if fmt == "16:9" and sw < 0.5:
        return "right" if sx + sw / 2 < 0.5 else "left"
    return "bottom" if sy + sh / 2 < 0.5 else "top"


def _text_box(style: str, region: str, fmt: str) -> list[float]:
    w, h = BRIEF_TEXT_WH[style][fmt]
    if fmt == "16:9" and region in ("left", "right") and style != "giant":
        w = min(w, 0.46)
    x = 0.5 - w / 2
    if region == "left" and fmt == "16:9":
        x = 0.05
    elif region == "right" and fmt == "16:9":
        x = 0.95 - w
    y = {"top": 0.08, "bottom": 0.92 - h}.get(region, 0.5 - h / 2)
    if fmt == "9:16" and region == "bottom" and style in ("small", "medium"):
        y = 0.86 - h
    return _clip_box([x, y, w, h])


def _away_point(boxes: dict, fmt: str) -> tuple[float, float]:
    """Centre for the piece's primitive, on the side away from the text."""
    x, y, w, h = boxes[fmt]
    cx, cy = x + w / 2, y + h / 2
    if fmt == "16:9" and abs(cx - 0.5) > 0.1:
        return (0.74 if cx < 0.5 else 0.26), 0.5
    if h >= 0.55:
        return 0.5, 0.5
    return 0.5, (0.22 if cy > 0.5 else 0.78)


def _ground_rgbs(pal: list[str], bg: dict, img: dict | None, boxes: dict | None) -> list[np.ndarray]:
    def col(c):
        if isinstance(c, str) and c.startswith("palette:"):
            return _rgb(pal[int(c.split(":")[1])])
        return _rgb(c)
    t = bg["type"]
    if t == "color":
        return [col(bg["color"])]
    if t == "gradient":                       # the text sits mid-frame: sample the ramp between its ends
        c0, c1 = col(bg["colors"][0]), col(bg["colors"][-1])
        return [c0 + (c1 - c0) * f for f in (0.3, 0.5, 0.7)]
    if t == "shapes":
        return [col(bg.get("color") or "palette:0")]
    if img is None:
        return [np.array([0.3, 0.3, 0.3])]
    tr = bg.get("treatment") or {}
    dk = float(tr.get("darken", 0.0)) if isinstance(tr, dict) else 0.0
    under = _region_mean(img, boxes, bg.get("focus") or [0.5, 0.5], 1.06)
    if not under:
        under = [np.full(3, float(img.get("luma_mean", 0.5)))]
    return [u * (1.0 - dk) for u in under]


def _wcag(a, b) -> float:
    """WCAG contrast ratio of two sRGB colours (0..1), on linear relative luminance."""
    def lum(c):
        c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
    la, lb = lum(a) + 0.05, lum(b) + 0.05
    return max(la, lb) / min(la, lb)


def _text_color(pal: list[str], grounds: list[np.ndarray], accent_ok: bool = False) -> tuple[str, float]:
    """Palette ref that reads best on every ground (min WCAG contrast); accents only when allowed and
    they still read (>= 4.5)."""
    best, score = 1, -1.0
    for i in range(len(pal)):
        if i >= 2 and not accent_ok:
            continue
        s = min(_wcag(_rgb(pal[i]), g) for g in grounds)
        s_eff = (s * 1.3 if s >= 4.5 else 0.0) if i >= 2 else s
        if s_eff > score:
            best, score = i, s_eff
    return f"palette:{best}", float(min(_wcag(_rgb(pal[best]), g) for g in grounds))


def _family_treatment(family: str, pal: list[str]) -> list[dict] | None:
    acc = [_rgb(c) for c in pal[2:]] or [_rgb(pal[1])]
    warm = float(np.mean([c[0] - c[2] for c in acc])) >= 0.0
    preset = "warm" if warm else "teal_orange"
    if family in ("film_soft", "film_type_soft"):
        return [{"type": "film_grade", "preset": preset, "halation": 0.6, "crushed_blacks": 0.12, "lift": 0.04},
                {"type": "light_leak", "strength": 0.55, "speed": 0.5}, {"type": "grain", "amount": 0.25},
                {"type": "dust_scratches", "amount": 0.3}]
    if family in ("film_hectic", "film_type_hectic"):
        return [{"type": "film_grade", "preset": preset, "halation": 0.9, "crushed_blacks": 0.25},
                {"type": "light_leak", "strength": 0.8, "speed": 4.0}, {"type": "grain", "amount": 0.3},
                {"type": "dust_scratches", "amount": 0.5}]
    if family == "type_soft":
        return [{"type": "paper", "tone": _mix(pal[1], "#f4eee2", 0.5), "fibre": 0.6, "speckle": 0.25,
                 "amount": 0.85}, {"type": "grain", "amount": 0.3}]
    if family == "type_hectic":
        return [{"type": "posterize_flat", "smooth": 0.0}]
    if family == "paper_assembly":
        return [{"type": "paper", "tone": "#f1ebdd", "fibre": 0.6, "speckle": 0.35, "amount": 0.75},
                {"type": "grain", "amount": 0.18}]
    if family == "hype_print":
        return [{"type": "posterize_flat", "smooth": 0.2}, {"type": "paper", "tone": "#ffffff", "amount": 0.5},
                {"type": "grain", "amount": 0.12}]
    if family == "ambient_glow":
        return [{"type": "film_grade", "preset": "cool", "halation": 0.9, "crushed_blacks": 0.4, "lift": 0.0},
                {"type": "grain", "amount": 0.18}]
    if family == "neon_loop":
        return [{"type": "film_grade", "preset": "cool", "halation": 1.2, "crushed_blacks": 0.3},
                {"type": "grain", "amount": 0.1}]
    if family == "soft_loop":
        return [{"type": "grain", "amount": 0.1}]
    if family == "xerox_zine":
        return [{"type": "paper", "tone": "#efede6", "fibre": 0.4, "speckle": 0.5, "amount": 0.6},
                {"type": "grain", "amount": 0.3}, {"type": "dust_scratches", "amount": 0.4}]
    return None


def _flat_ground(family: str, k: int, kind: str, n_pal: int) -> dict | None:
    """Flat colour ground for shot k (no image), or None for the family's gradient look."""
    acc = list(range(2, n_pal)) or [1]
    a = acc[k % len(acc)]
    a2 = acc[(k + 1) % len(acc)]
    if kind == "rest":
        return {"type": "color", "color": "palette:0"}
    if family == "type_soft":
        return {"type": "color", "color": f"palette:{[1, a, 0][k % 3]}"}
    if family in ("type_hectic", "hype_print", "xerox_zine"):
        seq = [0, a, 1] if family != "xerox_zine" else [1, 0, a]
        return {"type": "color", "color": f"palette:{seq[k % 3]}"}
    if family == "paper_assembly":
        return {"type": "color", "color": "palette:1"}
    if family in ("film_hectic", "film_type_hectic") and kind in ("burst", "drop"):
        return {"type": "color", "color": f"palette:{[a, 1, 0, a2][k % 4]}"}
    return None


def plan_from_brief(analysis: dict, brief=None, seed: int = 0, duration: float | None = None,
                    log: list | None = None) -> dict:
    """Plan from a creative brief (dict, JSON text or raw model output; None = rules only).

    The brief picks family, palette, energy, bpm, arc, display lines, hero image, text region and mood;
    the shot list, treatments, transitions, verbs, type boxes, contrast, fonts and end card are rules.
    Empty input (no text, no images) spins the slot machine (seeded family / palette / theme)."""
    import copy
    log = log if log is not None else []
    analysis = analysis or {}
    images = list(analysis.get("images") or [])
    text = str(analysis.get("text") or "")
    if isinstance(brief, str):
        brief = parse_brief(brief, text, len(images), seed, log)
    else:
        brief = normalize_brief(brief, text, len(images), seed, log)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0xB12EF])
    family = brief["family"]
    spec = FAMILY_SPECS[family]
    look = BRIEF_LOOK[family]
    pace = spec["pace"]
    energy = float(brief["energy"])
    lines = list(brief["lines"])
    audio_in = analysis.get("audio")
    hero_i = brief["hero_image"] if brief["hero_image"] is not None else 0
    order = ([images[hero_i]] + [im for j, im in enumerate(images) if j != hero_i]) if images else []
    hero = order[0] if order else None

    # ---- duration, palette, fonts
    dur = None
    for cand in (duration, analysis.get("duration_hint"), (audio_in or {}).get("duration")):
        if cand is not None and float(cand) > 0:
            dur = float(cand)
            break
    dur = _r(float(np.clip(dur if dur is not None else DEFAULT_DUR, MIN_DUR, MAX_DUR)), 3)
    if brief["palette"]:
        pal = list(brief["palette"])[:6]
    else:
        bank = PALETTES[family]
        fallback = list(bank[int(rng.integers(len(bank)))])
        pal = _image_palette(images, fallback) if images else fallback
    n_pal = len(pal)
    moods = list(dict.fromkeys(list(brief["mood"]) + list(spec["moods"])))
    display, body = pick_fonts(moods, energy, seed, family)
    if look["text"] == "giant":            # giant cropped type wants a narrow face: re-draw away from wide ones
        for k in range(4):
            if (REGISTRY.get(display) or {}).get("width") not in ("wide", "extended"):
                break
            display, body = pick_fonts(moods, energy, seed, f"{family}:{k}")
    soft_case = pace == "soft" or family == "soft_loop"
    head_case = "as_is" if soft_case else _headline_case(display)

    # ---- tempo, beat grid, end card
    bpm, beats, downbeats = _beat_grid(analysis, dur, energy, float(brief["bpm"]))
    loop = brief["arc"] == "loop"
    lo_e, hi_e = (1.6, 3.4) if pace == "soft" else (1.4, 2.4)
    if not audio_in and not loop and lines:
        blo, bhi = spec["bpm"]
        for cand in sorted(range(int(round(bpm * 0.92)), int(round(bpm * 1.08)) + 1),
                           key=lambda b: (abs(b - bpm), b)):
            if not blo - 4 <= cand <= bhi + 4:
                continue
            _, bs, ds = _beat_grid(analysis, dur, energy, cand)
            te = _end_card_start(dur, ds, bs, 60.0 / cand)
            if te is not None and lo_e - _TOL <= dur - te <= hi_e + _TOL:
                bpm, beats, downbeats = _beat_grid(analysis, dur, energy, cand)
                break
    beat = 60.0 / bpm
    bar = 4.0 * beat
    t_body = dur
    if lines and not loop:
        te = _end_card_start(dur, downbeats, beats, beat)
        if te is not None and te >= 0.45 * dur:
            t_body = te
    raw_shots = _arc_shots(brief["arc"], pace, t_body / beat, beat)
    shots, acc_b = [], 0.0
    for k, (ln, kind) in enumerate(raw_shots):
        t0 = _r(acc_b * beat)
        acc_b += ln
        t1 = _r(acc_b * beat) if k < len(raw_shots) - 1 else _r(t_body)
        if shots and t1 - t0 < 0.05:
            shots[-1]["t1"] = t1
            continue
        shots.append({"t0": t0, "t1": t1, "kind": kind, "beats": ln})
    shots[0]["t0"] = 0.0
    shots[-1]["t1"] = _r(t_body)
    if loop and len(shots) > 2:                # seamless loop: the last shot repeats the first
        _assign_text(shots[:-1], lines, pace, family)
        shots[-1]["text"], shots[-1]["cont"] = shots[0]["text"], False
    else:
        _assign_text(shots, lines, pace, family)
    log.append(f"plan: {family} / {brief['arc']} / {len(shots)} shots / bpm {bpm:g} / fonts {display}+{body}")

    # ---- backgrounds
    style_t = look["text"]
    accents = [f"palette:{i}" for i in range(2, n_pal)] or ["palette:1"]
    image_led = bool(images) and family not in ("type_hectic", "paper_assembly")
    scenes = []
    img_k = 0
    flat_k = 0
    for i, sh in enumerate(shots):
        t0, t1, kind = sh["t0"], sh["t1"], sh["kind"]
        sc = {"id": f"s{i}", "t0": t0, "t1": t1, "background": None, "layers": [],
              "transition_out": {"type": "cut", "dur": 0.0}}
        img = None
        last_loop = loop and i == len(shots) - 1 and len(shots) > 2
        flat = _flat_ground(family, flat_k, kind, n_pal)
        if image_led and kind != "rest" and (flat is None or kind != "burst" or img_k % 2 == 0):
            img = order[0] if last_loop else order[img_k % len(order)]
            img_k += 0 if last_loop else 1
            focus = [_r(v) for v in (img.get("focus") or [0.5, 0.5])]
            if pace == "hectic":
                motion = {"verb": "beat_zoom", "zoom": [1.0, 1.06], "amount": _r(0.04 + 0.04 * energy, 3)}
            elif pace == "mid":
                motion = {"verb": "drift", "zoom": [1.06, 1.06],
                          "pan": [[0.0, 0.0], [_r(float(rng.uniform(-0.02, 0.02))), 0.0]]}
            else:
                motion = {"verb": "ken_burns", "zoom": [1.0, 1.06] if i % 2 == 0 else [1.06, 1.0],
                          "pan": [[0.0, 0.0], [0.0, -0.01 if i % 2 else 0.01]]}
            tr = {"duotone": None, "blur": 0.0, "darken": 0.2}
            if family == "hype_print":
                tr = [{"type": "screenprint", "inks": ["palette:0", accents[0]], "paper": "palette:1", "grit": 0.35}]
            elif family == "xerox_zine":
                tr = [{"type": "xerox", "paper": "palette:1", "ink": "palette:0", "boil": 2}]
            elif family == "neon_loop":
                tr = {"duotone": ["palette:0", accents[i % len(accents)]], "blur": 0.0, "darken": 0.15}
            elif family == "ambient_glow":
                tr = {"duotone": None, "blur": 0.0, "darken": {"dark": 0.75, "grow": 0.45}.get(kind, 0.3)}
            elif family == "type_soft":
                tr = {"duotone": None, "blur": 0.35, "darken": 0.45}
            sc["background"] = {"type": "image", "src": img["id"], "focus": focus, "motion": motion, "treatment": tr}
        else:
            if flat is None:
                a = pal[2 + (flat_k % max(1, n_pal - 2))] if n_pal > 2 else pal[1]
                b = pal[2 + ((flat_k + 1) % max(1, n_pal - 2))] if n_pal > 2 else pal[0]
                if kind == "dark":
                    cols = [pal[0], _mix(pal[0], a, 0.15)]
                elif family in ("film_soft", "film_type_soft", "film_hectic", "film_type_hectic"):
                    cols = [a, _mix(b, pal[0], 0.45)]
                elif family == "ambient_glow":
                    cols = [pal[0], _mix(pal[0], a, {"dark": 0.0, "grow": 0.18}.get(kind, 0.3))]
                elif family == "neon_loop":
                    cols = [pal[0], _mix(pal[0], a, 0.3)]
                elif family == "soft_loop":
                    cols = [_mix(pal[1], a, 0.35), _mix(pal[1], b, 0.7)]
                else:
                    cols = [pal[0], _mix(pal[0], a, 0.45)]
                if pace == "hectic":
                    motion = {"verb": "beat_zoom", "amount": _r(0.04 + 0.03 * energy, 3)}
                else:
                    motion = {"verb": "ken_burns", "zoom": [1.0, 1.06] if i % 2 == 0 else [1.06, 1.0],
                              "pan": [[0.0, 0.0], [0.0, -0.01]]}
                sc["background"] = {"type": "gradient", "colors": cols, "angle": int(90 + 20 * ((i % 3) - 1)),
                                    "motion": motion}
            else:
                flat = dict(flat)
                flat["motion"] = {"verb": "static"}
                sc["background"] = flat
            if not last_loop:
                flat_k += 1
        if last_loop:
            sc["background"] = copy.deepcopy(scenes[0]["background"])
            img = shots[0].get("img")
        sh["img"] = img
        scenes.append(sc)

    # ---- transitions (need both neighbours' lengths)
    for i in range(len(scenes) - 1):
        a, b = shots[i], shots[i + 1]
        shortest = min(a["t1"] - a["t0"], b["t1"] - b["t0"])
        if b["kind"] == "rest" or a["kind"] == "rest":
            tr = {"type": "cut", "dur": 0.0}
        elif a["kind"] == "burst" and b["kind"] == "burst":
            tr = ({"type": "flash", "dur": _q(min(0.08, 0.8 * shortest))}
                  if family in ("film_hectic", "film_type_hectic") and i % 2 == 0 else {"type": "cut", "dur": 0.0})
        else:
            prev = scenes[i - 1]["transition_out"]["type"] if i else None
            pool = [t for t in look["trans"] if t != prev] or look["trans"]
            typ = str(rng.choice(pool))
            if pace == "soft":
                d = min(look["tmax"], 0.55 * shortest)
            elif typ == "cut":
                d = 0.0
            else:
                d = min(look["tmax"], 0.6 * shortest)
            d = _q(d)
            tr = {"type": typ if d > 0 else "cut", "dur": d}
            if tr["type"] in ("push", "wipe"):
                tr["dir"] = str(rng.choice(["left", "right", "up", "down"]))
        scenes[i]["transition_out"] = tr

    # ---- layers: the one primitive, image cut-outs (paper), text with contrast-safe colour
    texts_seen = 0
    for i, (sc, sh) in enumerate(zip(scenes, shots)):
        t0, t1, kind = sc["t0"], sc["t1"], sh["kind"]
        d_out = sc["transition_out"]["dur"] if i < len(scenes) - 1 else 0.0
        hi = _r(min(dur, t1 + d_out / 2.0))
        span = t1 - t0
        img = sh["img"]
        regions = {f: _region_for(img or hero, f, brief["text_region"], style_t) for f in FORMATS}
        boxes = {f: _text_box(style_t, regions[f], f) for f in FORMATS}
        layers = sc["layers"]
        _brief_primitive(layers, look["prim"], family, kind, sh, boxes, t0, t1, hi, beat, accents, pal,
                         energy, i)
        if family == "paper_assembly" and images and kind != "rest":
            im = order[i % len(order)]
            _, py = _away_point(boxes, "9:16")
            qx, _ = _away_point(boxes, "16:9")
            layers.append({"type": "image", "src": im["id"], "mask": "rounded",
                           "box": {"9:16": _clip_box([0.1, py - 0.19, 0.8, 0.38]),
                                   "16:9": _clip_box([qx - 0.22, 0.12, 0.44, 0.76])},
                           "t0": t0, "t1": hi, "in": _anim("scale_pop", min(0.35, span * 0.3)),
                           "out": _anim("cut", 0.0), "emphasis": [],
                           "border": {"color": "palette:0", "width": 0.004},
                           "treatment": {"type": "halftone", "cell": 9, "ink": "palette:0", "paper": "palette:1",
                                         "texture": 0.3}})
        if not sh["text"]:
            continue
        grounds = _ground_rgbs(pal, sc["background"], img, boxes)
        color, con = _text_color(pal, grounds, accent_ok=family in ("type_hectic", "hype_print", "xerox_zine",
                                                                    "neon_loop"))
        highlight = None
        if sc["background"]["type"] == "image" and con < 4.0 and isinstance(sc["background"].get("treatment"), dict):
            sc["background"]["treatment"]["darken"] = 0.55
            grounds = _ground_rgbs(pal, sc["background"], img, boxes)
            color, con = _text_color(pal, grounds)
        if con < 3.0:
            highlight, color = "palette:0", "palette:1"
        if family == "paper_assembly" and texts_seen % 2 == 1:
            highlight = accents[texts_seen % len(accents)]
            color, _ = _text_color(pal, [_rgb(pal[int(highlight.split(":")[1])])])
        texts_seen += 1
        lay = {"type": "text", "text": sh["text"], "role": "headline", "font": "display", "case": head_case,
               "color": color, "highlight": highlight, "align": "center", "box": boxes, "t0": _r(t0), "t1": hi,
               "in": _anim("cut", 0.0), "out": _anim("cut", 0.0), "emphasis": []}
        nxt_cont = i + 1 < len(shots) and shots[i + 1]["cont"]
        _brief_text_motion(lay, family, pace, kind, sh, span, beat, energy, accents, rng, nxt_cont, i)
        if lay["in"]["verb"] == "row_slide" and style_t == "giant" and "\n" not in sh["text"]:
            # repeated rows are one line each: in portrait stack the words two to a line (the user's words)
            rows = _balanced_split(sh["text"].split(), 2)
            if len(rows) > 1:
                lay["text"] = {"9:16": "\n".join(" ".join(r) for r in rows), "16:9": sh["text"]}
        layers.append(lay)

    # ---- end card: the words small and still from the final hit
    if t_body < dur - 1e-6:
        title = " ".join(lines) if len(" ".join(lines).split()) <= 8 else lines[0]
        light_end = family in ("type_soft", "soft_loop", "paper_assembly")
        bg = {"type": "color", "color": "palette:1" if light_end else "palette:0", "motion": {"verb": "static"}}
        grounds = [_rgb(pal[1 if light_end else 0])]
        if hero is not None and family not in ("type_hectic", "type_soft", "paper_assembly", "soft_loop"):
            bg = {"type": "image", "src": hero["id"], "focus": [_r(v) for v in (hero.get("focus") or [0.5, 0.5])],
                  "motion": {"verb": "static"}, "treatment": {"duotone": None, "blur": 0.6, "darken": 0.7}}
            grounds = [np.full(3, 0.3 * float(hero.get("luma_mean", 0.5)))]
        color, _ = _text_color(pal, grounds)
        tin = ("fade", 0.6) if pace == "soft" else ("cut", 0.0)
        bar_box = {"9:16": [0.46, 0.535, 0.08, 0.004], "16:9": [0.48, 0.545, 0.04, 0.007]}
        layers = [_shape("bar", bar_box, t_body, dur, fill=accents[0], tin=("stroke_draw", 0.5)),
                  {"type": "text", "text": title, "role": "caption", "font": "body",      # small: the quiet face reads
                   "case": "as_is" if soft_case else "upper", "color": color, "highlight": None,
                   "align": "center", "box": {f: list(BRIEF_TITLE[f]) for f in FORMATS}, "t0": _r(t_body),
                   "t1": dur, "in": _anim(*tin), "out": _anim("cut", 0.0), "emphasis": []}]
        scenes.append({"id": f"s{len(scenes)}", "t0": _r(t_body), "t1": dur, "background": bg, "layers": layers,
                       "transition_out": {"type": "cut", "dur": 0.0}})
        shots.append({"t0": t_body, "t1": dur, "kind": "end", "beats": (dur - t_body) / beat})

    # ---- audio cues
    sfx, risers, sections = [], [], [{"t": 0.0, "label": "intro"}]
    first: dict[str, float] = {}
    for sh in shots:
        first.setdefault(sh["kind"], sh["t0"])
    t_drop = next((first[k] for k in ("drop", "peak", "burst") if k in first), None)
    if t_drop is None and len(shots) > 2:
        t_drop = shots[len(shots) // 2]["t0"]
    if t_drop:
        sections.append({"t": _r(t_drop), "label": "drop"})
        sfx.append({"t": _r(t_drop), "type": "impact", "gain": _r(0.6 + 0.4 * energy, 3)})
        if energy >= 0.5:
            risers.append({"t0": _r(max(0.0, t_drop - bar)), "t1": _r(t_drop)})
    if "rest" in first:
        sections.append({"t": _r(first["rest"]), "label": "break"})
    if "end" in first:
        sections.append({"t": _r(first["end"]), "label": "outro"})
        sfx.append({"t": _r(first["end"]), "type": "impact" if pace != "soft" else "swell",
                    "gain": _r(0.5 + 0.4 * energy, 3)})
    for i, sh in enumerate(shots):
        if sh["kind"] == "burst":
            sfx.append({"t": _r(sh["t0"]), "type": "hit", "gain": 0.5})
        if i < len(scenes) - 1:
            tr = scenes[i]["transition_out"]
            tb = scenes[i]["t1"]
            if tr["type"] in ("push", "wipe"):
                sfx.append({"t": _r(max(0.0, tb - tr["dur"] / 2)), "type": "whoosh", "gain": 0.6})
            elif tr["type"] in ("crossfade", "luma_wipe", "shape_reveal") and pace == "soft":
                sfx.append({"t": _r(max(0.0, tb - tr["dur"] / 2)), "type": "swell", "gain": 0.35})
            elif tr["type"] == "glitch":
                sfx.append({"t": _r(tb), "type": "glitch", "gain": 0.4})
    sfx = sorted([s for s in sfx if 0.0 <= s["t"] <= dur], key=lambda s: (s["t"], s["type"]))
    sections = sorted({s["t"]: s for s in sections}.values(), key=lambda s: s["t"])
    minor = any(m in ("dark", "hype", "techno") for m in moods[:2])
    keys = ["A", "E", "D", "F#", "C#"] if minor else ["C", "F", "G", "D", "A#"]
    tr_style = _family_treatment(family, pal)
    has_grain = any(t.get("type") == "grain" for t in (tr_style or []))
    beat_flash = 0.0
    if any(t.get("type") == "posterize_flat" for t in (tr_style or [])):
        beat_flash = 0.0                       # posterize snaps a lifted frame to the light ink: whiteouts
    elif pace == "hectic":
        beat_flash = _r(0.25 + 0.2 * (energy - 0.6) / 0.4, 3)
    elif family == "neon_loop":
        beat_flash = 0.12
    plan = {
        "version": PLAN_VERSION, "seed": int(seed), "duration": dur, "fps": FPS,
        "style": {"palette": pal, "background": "palette:0", "fonts": {"display": display, "body": body},
                  "motion": look["motion"], "energy": _r(energy, 3), "grain": 0.0 if has_grain else 0.06,
                  "vignette": look["vignette"], "beat_flash": float(np.clip(beat_flash, 0.0, 1.0)),
                  "treatment": tr_style},
        "audio": {"source": "user" if audio_in else "synth", "bpm": bpm, "key": str(rng.choice(keys)),
                  "scale": "minor" if minor else "major", "mood": look["music"], "beats": beats,
                  "downbeats": downbeats, "sections": sections, "risers": risers, "sfx": sfx},
        "voice": None,
        "scenes": scenes,
    }
    errors = validate_plan(plan, analysis)
    if errors:
        raise RuntimeError("brief planner produced an invalid plan (bug):\n" + "\n".join(errors))
    return plan


def _shape(prim: str, box: dict, t0: float, t1: float, stroke=None, fill=None, fo: float = 1.0, sw: float = 0.006,
           tin=("fade", 0.4), tout=("cut", 0.0), emph=None, echo=None) -> dict:
    return {"type": "shape", "source": "primitive", "ref": None, "primitive": prim,
            "box": {f: _clip_box(b) for f, b in box.items()}, "stroke": stroke,
            "stroke_width": sw, "fill": fill, "fill_opacity": fo if fill else 0.0, "offset": [0.0, 0.0],
            "scale": 1.0, "t0": _r(t0), "t1": _r(t1), "in": _anim(*tin), "out": _anim(*tout),
            "emphasis": emph or [], "echo": echo}


def _brief_primitive(layers, prim, family, kind, sh, boxes, t0, t1, hi, beat, accents, pal, energy, i):
    """The family's one primitive, kept off the text box (drawn before the text)."""
    full = {f: [0.0, 0.0, 1.0, 1.0] for f in FORMATS}
    span = t1 - t0
    if kind == "rest" or prim is None:
        return
    if prim == "strobe":
        if kind in ("hold", "chapter", "accel", "peak", "loop") and span >= 3 * beat:
            for k in (2, 3):
                ts = t0 + k * beat
                if ts + STROBE_T < t1:
                    layers.append(_shape("rect", full, ts, ts + STROBE_T, fill="palette:1", fo=0.9, tin=("cut", 0.0)))
        return
    if prim == "underline":
        if sh.get("text"):
            ub = {}
            for f in FORMATS:
                x, y, w, h = boxes[f]
                ub[f] = [x + 0.3 * w, min(0.95, y + h + 0.012), 0.4 * w, 0.004 if f == "9:16" else 0.006]
            layers.append(_shape("bar", ub, t0, hi, fill=accents[0],
                                 tin=("cut", 0.0) if sh.get("cont") else ("stroke_draw", min(1.4, 0.5 * span))))
        return
    if prim == "lane":
        lb = {}
        for f in FORMATS:
            x, y, w, h = boxes[f]
            lb[f] = [0.0, 0.86 if y + h / 2 < 0.6 else 0.1, 1.0, 0.02 if f == "9:16" else 0.03]
        layers.append(_shape("bar", lb, t0, hi, fill=accents[0],
                             tin=("stroke_draw", min(2 * beat, 0.5 * span)) if kind == "hold" else ("cut", 0.0)))
        return
    centre = {f: _away_point(boxes, f) for f in FORMATS}
    if prim == "glow":
        size = {"dark": 0.06, "grow": 0.24, "peak": 0.48}.get(kind, 0.3)
        cb, rb = {}, {}
        for f in FORMATS:                     # grow, but never into the text
            x, y, w, h = boxes[f]
            cx, cy = centre[f]
            fw, fh = FORMATS[f]
            gy = max(0.0, y - cy, cy - (y + h)) * fh / min(fw, fh)
            gx = max(0.0, x - cx, cx - (x + w)) * fw / min(fw, fh)
            room = 2.0 * max(gx, gy) * 0.92 if sh.get("text") else 2.0
            cb[f] = _square_box(f, cx, cy, max(0.04, min(size, room)))
            rb[f] = _square_box(f, cx, cy, max(0.06, min(0.9, size * 1.8, room * 1.4)))
        layers.append(_shape("circle", cb, t0, hi, fill=accents[0], fo=0.9,
                             tin=("fade", min(2.0, 0.6 * span)) if i == 0 else ("scale_pop", min(0.9, 0.4 * span)),
                             emph=[{"verb": "pulse", "on": "downbeats", "amount": 0.05}]))
        layers.append(_shape("ring", rb, t0 + min(beat, 0.3 * span), hi, stroke=accents[0], sw=0.003,
                             tin=("stroke_draw", min(1.4, 0.5 * span)),
                             echo={"count": 4, "spacing": 0.03, "decay": 0.6}))
        return
    if prim == "neon":
        dim = _mix(pal[0], pal[2 if len(pal) > 2 else 1], 0.35)
        layers.append(_shape("grid", full, t0, hi, stroke=dim, sw=0.002, tin=("cut", 0.0)))
        rb = {f: _square_box(f, centre[f][0], centre[f][1], 0.34) for f in FORMATS}
        layers.append(_shape("ring", rb, t0, hi, stroke=accents[i % len(accents)], sw=0.008,
                             tin=("stroke_draw", min(0.6, 0.4 * span)),
                             emph=[{"verb": "pulse", "on": "beats", "amount": _r(0.04 + 0.05 * energy, 3)}],
                             echo={"count": 3, "spacing": 0.025, "decay": 0.55}))
        return
    if prim == "breath":
        for k, (sz, fo) in enumerate(((0.75, 0.45), (0.5, 0.35))):
            cb = {f: _square_box(f, 0.5 + (0.08 if k else -0.06), 0.5 + (0.05 if k else -0.04), sz) for f in FORMATS}
            layers.append(_shape("circle", cb, t0, hi, fill=accents[(i + k) % len(accents)], fo=fo,
                                 tin=("fade", min(1.2, 0.4 * span)),
                                 emph=[{"verb": "pulse", "on": "downbeats", "amount": 0.08}]))
        return
    if prim == "cutouts":
        for k in range(2):
            cx = centre["9:16"][0] + (-0.22 if k == 0 else 0.24)
            cy = centre["9:16"][1] + (0.06 if k == 0 else -0.08)
            qx = centre["16:9"][0] + (-0.1 if k == 0 else 0.12)
            cb = {"9:16": _square_box("9:16", cx, cy, 0.34 if k == 0 else 0.22),
                  "16:9": _square_box("16:9", qx, 0.32 if k == 0 else 0.66, 0.34 if k == 0 else 0.22)}
            layers.append(_shape("circle" if (i + k) % 2 == 0 else "rect", cb, t0 + k * 0.5 * beat, hi,
                                 fill=accents[(i + k) % len(accents)], fo=1.0,
                                 tin=("scale_pop", min(0.35, 0.3 * span))))
        return
    if prim == "cross":
        cb = {f: _square_box(f, centre[f][0], centre[f][1], 0.16) for f in FORMATS}
        layers.append(_shape("cross", cb, t0, hi, stroke=accents[0], sw=0.012, tin=("cut", 0.0),
                             emph=[{"verb": "shake", "on": "beats", "amount": 0.5}]))


ROW_SOFT_IN = {"verb": "row_slide", "rows": 1, "travel": 0.025, "track": 0.15, "drift": 0.006, "fade": True,
               "stagger": 0.4}
ROW_SOFT_OUT = {"verb": "row_slide", "rows": 1, "travel": 0.0, "track": 0.25, "drift": 0.006, "fade": True}


def _brief_text_motion(lay, family, pace, kind, sh, span, beat, energy, accents, rng, nxt_cont, i):
    """In / out verbs and emphasis for one text layer, by family and shot kind."""
    cont = sh.get("cont")
    if pace == "soft" or family == "soft_loop":
        if cont:
            lay["in"] = _anim("cut", 0.0)
        elif family == "type_soft":
            lay["in"] = dict(ROW_SOFT_IN, dur=_r(min(1.1, 0.4 * span), 3))
        elif family == "ambient_glow":
            lay["in"] = _anim("blur_in", min(1.0, 0.4 * span))
        else:
            lay["in"] = _anim(str(rng.choice(["fade", "blur_in"])), min(1.0, 0.35 * span))
        if not nxt_cont:
            lay["out"] = (dict(ROW_SOFT_OUT, dur=_r(min(0.8, 0.25 * span), 3)) if family == "type_soft"
                          else _anim("fade", min(0.6, 0.2 * span)))
        if family == "soft_loop":
            lay["emphasis"] = [{"verb": "pulse", "on": "downbeats", "amount": 0.03}]
        return
    if family == "paper_assembly":
        v = str(rng.choice(["slide_up", "scale_pop", "mask_wipe"]))
        lay["in"] = _anim("cut", 0.0) if cont else _anim(v, min(0.4, 0.3 * span))
        lay["out"] = _anim("cut", 0.0) if nxt_cont else _anim("slide_up", min(0.25, 0.2 * span))
        return
    if family == "neon_loop":
        v = str(rng.choice(["word_stagger", "type_on", "scramble"]))
        lay["in"] = _anim(v, min(0.6, 0.4 * span), **({"stagger_beats": 0.5} if v == "word_stagger" else {}))
        lay["emphasis"] = [{"verb": "pulse", "on": "beats", "amount": _r(0.03 + 0.04 * energy, 3)}]
        return
    # hectic families: giant type, words on cuts, bursts, strobe
    punch = [{"verb": "pulse", "on": "beats", "amount": _r(0.05 + 0.04 * energy, 3)}]
    if kind == "burst":
        if family == "type_hectic":
            lay["in"] = {"verb": "row_slide", "dur": 0.0, "rows": int(rng.choice([3, 5])),
                         "scale": _r(float(rng.uniform(1.5, 3.5)), 2), "travel": 0.0, "alt_color": accents[0]}
        return
    if kind in ("hold", "chapter", "peak", "grow", "loop") or (kind == "accel" and sh["beats"] >= 2):
        if family in ("type_hectic", "hype_print", "xerox_zine"):
            lay["in"] = {"verb": "row_slide", "dur": _r(min(beat, 0.4 * span), 3), "rows": int(rng.choice([3, 5])),
                         "scale": _r(float(rng.uniform(1.0, 1.4)), 2), "travel": 0.9, "drift": 0.015,
                         "alt_color": accents[0]}
            lay["emphasis"] = punch + ([{"verb": "color_flip", "on": "beats", "amount": 1.0, "color": accents[0]}]
                                       if family == "type_hectic" else [])
        elif family == "film_type_hectic" and sh.get("img") is not None:
            lay["window"] = {"mode": "fill", "src": sh["img"]["id"], "zoom": [1.0, 1.08]}
            lay["in"] = _anim("scale_pop", min(0.2, 0.3 * span))
            lay["emphasis"] = punch
        else:
            lay["in"] = _anim("scale_pop", min(0.18, 0.3 * span))
            lay["emphasis"] = punch
        return
    # drop / short accel shots: one word, verbs alternate (scramble resolves on the next hit)
    v = (["scramble", "kinetic_split", "scale_pop", "slide_left"][i % 4] if family != "film_hectic"
         else ["scale_pop", "kinetic_split", "cut"][i % 3])
    if span < 0.5 or (v == "scramble" and span < 0.8):      # short shots: the word lands on the cut
        v = "cut" if span < 0.5 else "scale_pop"
    d = min(beat, 0.5 * span) if v == "scramble" else (0.0 if v == "cut" else min(0.2, 0.35 * span))
    lay["in"] = _anim(v, d, **({"rate": 20} if v == "scramble" else {}))
    lay["emphasis"] = [{"verb": "shake", "on": "beats", "amount": 0.4}] if family == "xerox_zine" else punch


__all__ = ["plan_from_analysis", "plan_from_brief", "MOOD_TABLE", "TRANSITIONS", "VERBS"]
