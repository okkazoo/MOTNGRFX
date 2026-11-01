"""Plan schema: constants, validation and helpers.

The plan is a plain JSON-serialisable dict (see SPEC.md, Interface 2). It is the
contract between planners (rule-based now, LLM later) and the renderer / synth.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

import numpy as np

from .font_registry import FONT_NAMES, prompt_guide

# --------------------------------------------------------------------------
# Schema constants
# --------------------------------------------------------------------------

PLAN_VERSION = 1

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080)}

# user 2026-10-30: favourite fonts only. The six legacy slots are now liked faces of the same role (Anton -> Big Shoulders Display, Bebas Neue -> Facade, Inter -> Space Grotesk,
# Playfair Display -> Interlope, JetBrains Mono -> Major Mono Display); BUNDLED_FONTS = those + every other liked face in mograph/fonts/fonts.json.
LEGACY_FONTS = ["Big Shoulders Display", "Facade", "Space Grotesk", "Interlope", "Major Mono Display", "Anybody"]
from . import fonts_liked as _fl  # noqa: E402  (relative: ComfyUI imports this pack as a package, top-level `mograph` is not on sys.path yet)
BUNDLED_FONTS = LEGACY_FONTS + [f for f in FONT_NAMES if f not in LEGACY_FONTS and _fl.is_liked(f)]
OUTLINE_SWAP = ["beats", "downbeats", "onsets"]       # text "outline.swap" (or a list of times)
WINDOW_MODES = ["fill", "knockout"]                   # text "window.mode"
TEXT_PATHS = ["arc", "circle", "wave", "line"]        # text "path.type"
BACKDROPS = ["dim", "blur", "band"]                   # text "backdrop.type" (or "none" / false)
LETTER_SCALES = ["list", "random", "wave"]            # text "letter_scale.type"
TEXT_BOX_RANGE = (-1.0, 2.0)                          # text boxes may run off-frame (giant type, cropped)

VERBS: dict[str, list[str]] = {
    "text_in": ["cut", "fade", "type_on", "word_stagger", "slide_up", "slide_left",
                "scale_pop", "mask_wipe", "blur_in", "kinetic_split", "scramble", "row_slide"],
    "text_out": ["cut", "fade", "slide_up", "slide_left", "scale_down", "mask_wipe", "blur_out", "row_slide"],
    "emphasis": ["pulse", "shake", "color_flip", "underline_draw"],
    "shape_in": ["stroke_draw", "fade", "scale_pop", "cut"],
    "shape_out": ["cut", "fade", "scale_down"],
    "image_in": ["cut", "fade", "scale_pop", "slide_up", "slide_left"],
    "image_out": ["cut", "fade", "scale_down", "slide_up", "slide_left"],
    "background_motion": ["static", "ken_burns", "drift", "beat_zoom", "parallax_split"],
    "primitives": ["circle", "ring", "rect", "bar", "cross", "grid", "dot_field", "wave"],
    "masks": ["rect", "rounded", "circle"],          # plus "contour:<contour id>"
    "sfx": ["whoosh", "impact", "hit", "click", "riser", "swell", "glitch", "pop"],
    "moods": ["driving", "calm", "dark", "bright", "playful", "epic"],
    "motion": ["snappy", "smooth", "bouncy", "glitch"],
    "treatments": ["halftone", "screenprint", "xerox", "duotone", "tritone", "film_grade", "posterize_flat",
                   "paper", "grain", "vignette", "light_leak", "dust_scratches"],
}

TRANSITIONS = ["cut", "crossfade", "push", "wipe", "zoom_blur", "shape_reveal", "glitch", "luma_wipe", "flash"]
DIRECTIONAL_TRANSITIONS = ["push", "wipe"]
DIRECTIONS = ["left", "right", "up", "down"]

LAYER_TYPES = ["text", "shape", "image"]
BACKGROUND_TYPES = ["image", "color", "gradient", "shapes"]
TEXT_ROLES = ["headline", "sub", "caption", "label"]
TEXT_CASES = ["upper", "lower", "as_is"]
TEXT_FIT = ["auto", "none"]                          # text "fit": balanced auto line breaks in tall boxes, or off
ALIGNS = ["left", "center", "right"]
FONT_SLOTS = ["display", "body"]
SHAPE_SOURCES = ["contour", "line", "primitive"]
EMPHASIS_ON = ["beats", "downbeats", "onsets"]
AUDIO_SOURCES = ["synth", "user"]
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = ["major", "minor"]
SECTION_LABELS = ["intro", "build", "drop", "break", "outro"]   # advisory; synth treats "drop" specially
FILM_PRESETS = ["mono", "teal_orange", "warm", "cool"]
HALFTONE_SHAPES = ["dot", "line"]
DITHERS = ["noise", "ordered", "diffusion", "none"]
# typed treatment params: numeric (lo, hi) ranges; None = unbounded
TREATMENT_NUMS = {"cell": (2.0, 200.0), "angle": (None, None), "contrast": (0.0, 3.0), "texture": (0.0, 1.0),
                  "misregister": (0.0, 60.0), "grit": (0.0, 1.0), "smooth": (0.0, 1.0), "dither_amount": (0.0, 1.0),
                  "dither_px": (1.0, 16.0), "boil": (0.0, 20.0), "threshold": (0.0, 1.0), "toner_grit": (0.0, 1.0),
                  "crushed_blacks": (0.0, 1.0), "halation": (0.0, 2.0), "lift": (0.0, 0.5),
                  "saturation": (0.0, 3.0), "fibre": (0.0, 1.0), "speckle": (0.0, 1.0), "amount": (0.0, 1.0),
                  "strength": (0.0, 2.0), "speed": (0.0, 10.0)}

_EPS = 1e-6
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_PAL_RE = re.compile(r"^palette:(\d+)$")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _is_num(x: Any) -> bool:
    if isinstance(x, bool) or isinstance(x, np.bool_):
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.isfinite(float(x))
    return False


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"bad hex colour {h!r}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def resolve_color(plan: dict, c: Any) -> tuple[float, float, float]:
    """Resolve "#rrggbb" / "#rgb" / "palette:N" / (r, g, b) to floats 0..1."""
    if isinstance(c, (tuple, list, np.ndarray)) and len(c) == 3:
        vals = [float(v) for v in c]
        if any(v > 1.0 for v in vals):
            vals = [v / 255.0 for v in vals]
        return (vals[0], vals[1], vals[2])
    if not isinstance(c, str):
        raise ValueError(f"bad colour {c!r}")
    s = c.strip()
    m = _PAL_RE.match(s)
    if m:
        pal = (plan or {}).get("style", {}).get("palette", [])
        n = int(m.group(1))
        if n >= len(pal):
            raise ValueError(f"{s} out of range (palette has {len(pal)} colours)")
        return resolve_color(plan, pal[n])
    if s.startswith("#"):
        try:
            return _hex_to_rgb(s)
        except ValueError:
            pass
    raise ValueError(f"bad colour {c!r}")


def beat_snap(t: float, beats: Iterable[float]) -> float:
    """Nearest beat time to t (ties go to the earlier beat). Empty beats -> t."""
    arr = np.asarray(list(beats) if not isinstance(beats, np.ndarray) else beats, dtype=np.float64)
    if arr.size == 0:
        return float(t)
    i = int(np.argmin(np.abs(arr - float(t))))
    return float(arr[i])


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def to_json(plan: dict, indent: int | None = 2) -> str:
    """Serialise a plan; numpy scalars/arrays become plain JSON numbers/lists."""
    return json.dumps(plan, default=_json_default, indent=indent)


_FENCE_RE = re.compile(r"^\s*```(?:json|jsonc)?\s*\n(.*?)\n\s*```\s*$", re.S)


def from_json(s: str | bytes) -> dict:
    """Parse a plan from JSON. Tolerates a surrounding ```json fence (LLM output)."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("plan JSON must be an object")
    return obj


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _analysis_ids(analysis: dict | None):
    images, contours, lines = set(), set(), set()
    for img in (analysis or {}).get("images") or []:
        iid = img.get("id")
        images.add(iid)
        for c in img.get("contours") or []:
            if c.get("id") is not None:
                contours.add(c["id"])
        for i, ln in enumerate(img.get("lines") or []):
            lines.add(ln.get("id") or f"{iid}:l{i}")
    return images, contours, lines


class _V:
    def __init__(self, plan: dict, analysis: dict | None):
        self.plan = plan
        self.errors: list[str] = []
        self.check_refs = analysis is not None
        self.images, self.contours, self.lines = _analysis_ids(analysis)
        style = plan.get("style") if isinstance(plan.get("style"), dict) else {}
        pal = style.get("palette")
        self.n_pal = len(pal) if isinstance(pal, list) else 0

    def err(self, path: str, msg: str):
        self.errors.append(f"{path}: {msg}")

    def num(self, d: dict, key: str, path: str, lo=None, hi=None, required=True) -> float | None:
        if key not in d or d[key] is None:
            if required:
                self.err(f"{path}.{key}", "missing")
            return None
        v = d[key]
        if not _is_num(v):
            self.err(f"{path}.{key}", f"not a number ({v!r})")
            return None
        v = float(v)
        if lo is not None and v < lo - _EPS:
            self.err(f"{path}.{key}", f"{v} < {lo}")
        if hi is not None and v > hi + _EPS:
            self.err(f"{path}.{key}", f"{v} > {hi}")
        return v

    def enum(self, value, allowed: list, path: str, what: str):
        if value not in allowed:
            self.err(path, f"{value!r} is not a valid {what} (allowed: {', '.join(map(str, allowed))})")

    def color(self, c, path: str, allow_none=False):
        if c is None:
            if not allow_none:
                self.err(path, "colour missing")
            return
        if not isinstance(c, str):
            self.err(path, f"colour must be a string, got {c!r}")
            return
        m = _PAL_RE.match(c)
        if m:
            n = int(m.group(1))
            if n >= self.n_pal:
                self.err(path, f"{c} out of range (palette has {self.n_pal} colours)")
            return
        if not _HEX_RE.match(c):
            self.err(path, f"{c!r} is not '#rrggbb' or 'palette:N'")

    def box(self, b, path: str, lo: float = 0.0, hi: float = 1.0):
        if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(_is_num(v) for v in b):
            self.err(path, f"box must be [x, y, w, h] numbers, got {b!r}")
            return
        x, y, w, h = (float(v) for v in b)
        if w <= 0 or h <= 0:
            self.err(path, f"box w/h must be > 0, got {b!r}")
        if x < lo - _EPS or y < lo - _EPS or x + w > hi + _EPS or y + h > hi + _EPS:
            self.err(path, f"box {b!r} not inside {lo:g}..{hi:g} frame space")

    def fmt_boxes(self, layer: dict, path: str, lo: float = 0.0, hi: float = 1.0):
        box = layer.get("box")
        if not isinstance(box, dict):
            self.err(f"{path}.box", "needs a per-format dict with '9:16' and '16:9'")
            return
        for fmt in FORMATS:
            if fmt not in box:
                self.err(f"{path}.box", f"missing format {fmt!r}")
            else:
                self.box(box[fmt], f"{path}.box[{fmt}]", lo, hi)

    def point(self, p, path: str):
        if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(_is_num(v) for v in p):
            self.err(path, f"point must be [x, y], got {p!r}")

    def anim(self, a, kind: str, path: str):
        if a is None:
            self.err(path, "missing (use {'verb': 'cut', 'dur': 0})")
            return
        if not isinstance(a, dict):
            self.err(path, f"must be an object, got {a!r}")
            return
        self.enum(a.get("verb"), VERBS[kind], f"{path}.verb", kind.replace("_", " ") + " verb")
        self.num(a, "dur", path, lo=0.0, required=False)

    def emphasis(self, emph, path: str):
        if emph is None:
            return
        if not isinstance(emph, list):
            self.err(path, "must be a list")
            return
        for i, e in enumerate(emph):
            p = f"{path}[{i}]"
            if not isinstance(e, dict):
                self.err(p, "must be an object")
                continue
            self.enum(e.get("verb"), VERBS["emphasis"], f"{p}.verb", "emphasis verb")
            on = e.get("on", "beats")
            if isinstance(on, list):
                if not all(_is_num(t) for t in on):
                    self.err(f"{p}.on", "time list must be numbers")
            else:
                self.enum(on, EMPHASIS_ON, f"{p}.on", "emphasis trigger")
            self.num(e, "amount", p, required=False)
            if e.get("color") is not None:
                self.color(e.get("color"), f"{p}.color")

    def font(self, f, path: str):
        if f in FONT_SLOTS or f in BUNDLED_FONTS:
            return
        self.err(path, f"font {f!r} is not 'display'/'body' or a bundled font ({', '.join(BUNDLED_FONTS)})")

    def ref_image(self, src, path: str):
        if self.check_refs and src not in self.images:
            self.err(path, f"image id {src!r} not in analysis ({sorted(self.images)})")

    def ref_contour(self, ref, path: str):
        if self.check_refs and ref not in self.contours:
            self.err(path, f"contour id {ref!r} not in analysis")

    def ref_line(self, ref, path: str):
        if self.check_refs and ref not in self.lines:
            self.err(path, f"line id {ref!r} not in analysis")


def _validate_treatment(v: _V, tr, path: str):
    """Typed print/film treatment: {"type": <VERBS["treatments"]>, ...params} or a list of them."""
    if isinstance(tr, list):
        for i, t in enumerate(tr):
            _validate_treatment(v, t, f"{path}[{i}]")
        return
    if not isinstance(tr, dict):
        v.err(path, "must be a treatment object, a list of them, or null")
        return
    typ = tr.get("type")
    v.enum(typ, VERBS["treatments"], f"{path}.type", "treatment")
    for k in ("ink", "paper", "tone"):
        if tr.get(k) is not None:
            v.color(tr.get(k), f"{path}.{k}")
    for k, (lo, hi) in (("inks", (1, 4)), ("colors", (2, 5)), ("palette", (2, 16))):
        cols = tr.get(k)
        if cols is None:
            continue
        if not isinstance(cols, list) or not (lo <= len(cols) <= hi):
            v.err(f"{path}.{k}", f"must be a list of {lo}..{hi} colours")
            continue
        for i, c in enumerate(cols):
            v.color(c, f"{path}.{k}[{i}]")
    for k, (lo, hi) in TREATMENT_NUMS.items():
        v.num(tr, k, path, lo, hi, required=False)
    if "preset" in tr:
        v.enum(tr.get("preset"), FILM_PRESETS, f"{path}.preset", "film preset")
    if "shape" in tr:
        v.enum(tr.get("shape"), HALFTONE_SHAPES, f"{path}.shape", "halftone shape")
    if "dither" in tr:
        v.enum(tr.get("dither"), DITHERS, f"{path}.dither", "dither")


def _validate_style(v: _V, plan: dict):
    style = plan.get("style")
    if not isinstance(style, dict):
        v.err("style", "missing or not an object")
        return
    pal = style.get("palette")
    if not isinstance(pal, list) or not (3 <= len(pal) <= 6):
        v.err("style.palette", "must be a list of 3..6 colours")
    else:
        for i, c in enumerate(pal):
            if not isinstance(c, str) or not _HEX_RE.match(c):
                v.err(f"style.palette[{i}]", f"{c!r} is not '#rrggbb'")
    v.color(style.get("background", "palette:0"), "style.background")
    fonts = style.get("fonts")
    if not isinstance(fonts, dict):
        v.err("style.fonts", "missing; needs {'display': font, 'body': font}")
    else:
        for slot in FONT_SLOTS:
            if fonts.get(slot) not in BUNDLED_FONTS:
                v.err(f"style.fonts.{slot}", f"{fonts.get(slot)!r} is not a bundled font ({', '.join(BUNDLED_FONTS)})")
    v.enum(style.get("motion"), VERBS["motion"], "style.motion", "motion style")
    for k in ("energy", "grain", "vignette", "beat_flash"):
        v.num(style, k, "style", 0.0, 1.0, required=(k == "energy"))
    if style.get("treatment") is not None:
        _validate_treatment(v, style.get("treatment"), "style.treatment")


def _validate_audio(v: _V, plan: dict, duration: float | None):
    audio = plan.get("audio")
    if not isinstance(audio, dict):
        v.err("audio", "missing or not an object")
        return
    hi = duration if duration is not None else None
    v.enum(audio.get("source"), AUDIO_SOURCES, "audio.source", "audio source")
    v.num(audio, "bpm", "audio", 20.0, 300.0)
    v.enum(audio.get("key"), KEYS, "audio.key", "key")
    v.enum(audio.get("scale"), SCALES, "audio.scale", "scale")
    v.enum(audio.get("mood"), VERBS["moods"], "audio.mood", "music mood")
    for k in ("beats", "downbeats"):
        arr = audio.get(k)
        if not isinstance(arr, list) or not all(_is_num(t) for t in arr):
            v.err(f"audio.{k}", "must be a list of seconds")
            continue
        if any(float(b) < float(a) - _EPS for a, b in zip(arr, arr[1:])):
            v.err(f"audio.{k}", "must be sorted ascending")
        if arr and (float(arr[0]) < -_EPS or (hi is not None and float(arr[-1]) > hi + _EPS)):
            v.err(f"audio.{k}", "times must lie in [0, duration]")
    for i, s in enumerate(audio.get("sections") or []):
        p = f"audio.sections[{i}]"
        if not isinstance(s, dict):
            v.err(p, "must be an object")
            continue
        v.num(s, "t", p, 0.0, hi)
        if not isinstance(s.get("label"), str):
            v.err(f"{p}.label", "must be a string")
    for i, r in enumerate(audio.get("risers") or []):
        p = f"audio.risers[{i}]"
        if not isinstance(r, dict):
            v.err(p, "must be an object")
            continue
        t0 = v.num(r, "t0", p, 0.0, hi)
        t1 = v.num(r, "t1", p, 0.0, hi)
        if t0 is not None and t1 is not None and t1 <= t0:
            v.err(p, f"t1 ({t1}) must be > t0 ({t0})")
    for i, s in enumerate(audio.get("sfx") or []):
        p = f"audio.sfx[{i}]"
        if not isinstance(s, dict):
            v.err(p, "must be an object")
            continue
        v.num(s, "t", p, 0.0, hi)
        v.enum(s.get("type"), VERBS["sfx"], f"{p}.type", "sfx type")
        v.num(s, "gain", p, 0.0, 2.0, required=False)


def _validate_background(v: _V, bg, path: str):
    if not isinstance(bg, dict):
        v.err(path, "missing or not an object")
        return
    t = bg.get("type")
    v.enum(t, BACKGROUND_TYPES, f"{path}.type", "background type")
    if t == "image":
        v.ref_image(bg.get("src"), f"{path}.src")
        if bg.get("focus") is not None:
            v.point(bg.get("focus"), f"{path}.focus")
    elif t == "color":
        v.color(bg.get("color"), f"{path}.color")
    elif t == "gradient":
        cols = bg.get("colors")
        if not isinstance(cols, list) or len(cols) < 2:
            v.err(f"{path}.colors", "gradient needs a list of >= 2 colours")
        else:
            for i, c in enumerate(cols):
                v.color(c, f"{path}.colors[{i}]")
        v.num(bg, "angle", path, required=False)
    elif t == "shapes":
        v.color(bg.get("color"), f"{path}.color", allow_none=True)
        for i, c in enumerate(bg.get("colors") or []):
            v.color(c, f"{path}.colors[{i}]")
    motion = bg.get("motion")
    if motion is not None:
        if not isinstance(motion, dict):
            v.err(f"{path}.motion", "must be an object")
        else:
            v.enum(motion.get("verb"), VERBS["background_motion"], f"{path}.motion.verb", "background motion verb")
    tr = bg.get("treatment")
    if isinstance(tr, list) or (isinstance(tr, dict) and "type" in tr):
        _validate_treatment(v, tr, f"{path}.treatment")
    if isinstance(tr, dict):
        duo = tr.get("duotone")
        if duo is not None:
            if not isinstance(duo, list) or len(duo) != 2:
                v.err(f"{path}.treatment.duotone", "must be [c_dark, c_light] or null")
            else:
                v.color(duo[0], f"{path}.treatment.duotone[0]")
                v.color(duo[1], f"{path}.treatment.duotone[1]")
        v.num(tr, "blur", f"{path}.treatment", 0.0, 1.0, required=False)
        v.num(tr, "darken", f"{path}.treatment", 0.0, 1.0, required=False)
    elif tr is not None and not isinstance(tr, list):
        v.err(f"{path}.treatment", "must be an object, a list of typed treatments, or null")


def _validate_shape_options(v: _V, L: dict, path: str):
    """Optional shapes-and-sizes options: rotate (+ in/out spin), overscale, path, backdrop, letter_scale."""
    rot = L.get("rotate")
    if rot is not None and not _is_num(rot):
        if not isinstance(rot, list) or not rot or not all(
                isinstance(k, (list, tuple)) and len(k) == 2 and all(_is_num(x) for x in k) for k in rot):
            v.err(f"{path}.rotate", "must be degrees or keyframes [[t, deg], ...]")
    for ph in ("in", "out"):
        if isinstance(L.get(ph), dict):
            v.num(L[ph], "spin", f"{path}.{ph}", -3600.0, 3600.0, required=False)
    v.num(L, "overscale", path, 0.1, 10.0, required=False)
    p = L.get("path")
    if p is not None:
        if not isinstance(p, dict):
            v.err(f"{path}.path", "must be an object or null")
        else:
            pp = f"{path}.path"
            v.enum(p.get("type"), TEXT_PATHS, f"{pp}.type", "text path")
            for k, lo, hi in (("radius", 0.01, 20.0), ("angle", -3600.0, 3600.0), ("span", 1.0, 360.0),
                              ("fill", 0.05, 1.5), ("amp", -2.0, 2.0), ("cycles", 0.0, 20.0), ("phase", None, None),
                              ("speed", -20.0, 20.0), ("step", -4.0, 4.0), ("travel", -10.0, 10.0),
                              ("offset", -5.0, 5.0), ("size", 0.005, 2.0)):
                v.num(p, k, pp, lo, hi, required=False)
            for k in ("center", "p0", "p1"):
                if p.get(k) is not None:
                    v.point(p.get(k), f"{pp}.{k}")
            on = p.get("on")
            if isinstance(on, list):
                if not all(_is_num(t) for t in on):
                    v.err(f"{pp}.on", "time list must be numbers")
            elif on is not None:
                v.enum(on, EMPHASIS_ON, f"{pp}.on", "path trigger")
    bd = L.get("backdrop")
    if bd is not None and bd is not False and bd != "none":
        if isinstance(bd, str):
            v.enum(bd, BACKDROPS, f"{path}.backdrop", "backdrop")
        elif not isinstance(bd, dict):
            v.err(f"{path}.backdrop", "must be an object, a backdrop type, 'none', false or null")
        else:
            v.enum(bd.get("type", "dim"), BACKDROPS + ["none"], f"{path}.backdrop.type", "backdrop")
            v.num(bd, "amount", f"{path}.backdrop", 0.0, 1.0, required=False)
            v.num(bd, "pad", f"{path}.backdrop", 0.0, 10.0, required=False)
            v.num(bd, "soft", f"{path}.backdrop", 0.0, 10.0, required=False)
            if bd.get("color") is not None:
                v.color(bd.get("color"), f"{path}.backdrop.color")
    ls = L.get("letter_scale")
    if ls is not None:
        lp = f"{path}.letter_scale"
        if isinstance(ls, list):
            if not ls or not all(_is_num(x) and float(x) > 0 for x in ls):
                v.err(lp, "list must be positive numbers")
        elif isinstance(ls, str):
            v.enum(ls, ["random", "wave"], lp, "letter scale")
        elif not isinstance(ls, dict):
            v.err(lp, "must be a list, 'random', 'wave' or an object")
        else:
            v.enum(ls.get("type", "list" if ls.get("values") else "random"), LETTER_SCALES, f"{lp}.type", "letter scale")
            vals = ls.get("values")
            if vals is not None and (not isinstance(vals, list) or not vals
                                     or not all(_is_num(x) and float(x) > 0 for x in vals)):
                v.err(f"{lp}.values", "must be a list of positive numbers")
            r = ls.get("range")
            if r is not None and (not isinstance(r, list) or len(r) != 2 or not all(_is_num(x) and float(x) > 0 for x in r)):
                v.err(f"{lp}.range", "must be [lo, hi] > 0")
            if ls.get("seed") is not None and not _is_num(ls.get("seed")):
                v.err(f"{lp}.seed", "must be an integer")
            v.num(ls, "phase", lp, -1.0, 1.0, required=False)
            v.num(ls, "speed", lp, -20.0, 20.0, required=False)
            v.enum(ls.get("valign", "baseline"), ["baseline", "center"], f"{lp}.valign", "letter valign")


def _validate_text_options(v: _V, L: dict, path: str):
    """Optional bold-type options: axes, vary (variable-font wave), outline (+ swap), window (type as mask)."""
    _validate_shape_options(v, L, path)
    axes = L.get("axes")
    if axes is not None and (not isinstance(axes, dict) or not all(_is_num(x) for x in axes.values())):
        v.err(f"{path}.axes", "must be an object of axis tag -> number, e.g. {'wght': 300}")
    vary = L.get("vary")
    if vary is not None:
        if not isinstance(vary, dict):
            v.err(f"{path}.vary", "must be an object or null")
        else:
            if vary.get("axis") is not None and not isinstance(vary.get("axis"), str):
                v.err(f"{path}.vary.axis", "must be an axis tag such as 'wght'")
            rng_ = vary.get("range")
            if rng_ is not None and (not isinstance(rng_, list) or len(rng_) != 2 or not all(_is_num(x) for x in rng_)):
                v.err(f"{path}.vary.range", "must be [lo, hi]")
            v.num(vary, "period", f"{path}.vary", 0.05, 60.0, required=False)
            v.num(vary, "period_beats", f"{path}.vary", 0.125, 64.0, required=False)
            v.num(vary, "phase", f"{path}.vary", -1.0, 1.0, required=False)
            v.num(vary, "scale", f"{path}.vary", 0.0, 2.0, required=False)
            if vary.get("on") is not None:
                v.enum(vary.get("on"), EMPHASIS_ON, f"{path}.vary.on", "vary trigger")
            v.enum(vary.get("fit", "both"), ["both", "lo", "hi"], f"{path}.vary.fit", "vary fit")
    ol = L.get("outline")
    if ol is not None:
        if not isinstance(ol, dict):
            v.err(f"{path}.outline", "must be an object or null")
        else:
            v.num(ol, "width", f"{path}.outline", 0.005, 0.3, required=False)
            sw = ol.get("swap")
            if isinstance(sw, list):
                if not all(_is_num(t) for t in sw):
                    v.err(f"{path}.outline.swap", "time list must be numbers")
            elif sw is not None:
                v.enum(sw, OUTLINE_SWAP, f"{path}.outline.swap", "outline swap trigger")
            v.enum(ol.get("start", "outline"), ["outline", "fill"], f"{path}.outline.start", "outline start")
            v.enum(ol.get("rows", "all"), ["all", "alt"], f"{path}.outline.rows", "outline rows")
    win = L.get("window")
    if win is not None:
        if not isinstance(win, dict):
            v.err(f"{path}.window", "must be an object or null")
            return
        mode = win.get("mode", "fill")
        v.enum(mode, WINDOW_MODES, f"{path}.window.mode", "window mode")
        if mode == "fill" or win.get("src") is not None:
            v.ref_image(win.get("src"), f"{path}.window.src")
        if win.get("plate") is not None:
            v.color(win.get("plate"), f"{path}.window.plate")
        v.num(win, "plate_opacity", f"{path}.window", 0.0, 1.0, required=False)
        v.num(win, "pad", f"{path}.window", 0.0, 10.0, required=False)
        z = win.get("zoom")
        if z is not None and not _is_num(z) and not (isinstance(z, list) and len(z) == 2 and all(_is_num(x) for x in z)):
            v.err(f"{path}.window.zoom", "must be a number or [z0, z1]")
        if win.get("focus") is not None:
            v.point(win.get("focus"), f"{path}.window.focus")


def _validate_layer(v: _V, L, path: str, lo: float, hi: float):
    if not isinstance(L, dict):
        v.err(path, "must be an object")
        return
    typ = L.get("type")
    v.enum(typ, LAYER_TYPES, f"{path}.type", "layer type")
    t0 = v.num(L, "t0", path)
    t1 = v.num(L, "t1", path)
    if t0 is not None and t1 is not None:
        if t1 <= t0:
            v.err(path, f"t1 ({t1}) must be > t0 ({t0})")
        if t0 < lo - _EPS or t1 > hi + _EPS:
            v.err(path, f"times [{t0}, {t1}] outside allowed [{lo}, {hi}] (scene.t0 .. scene.t1 + transition_out.dur/2)")
    v.emphasis(L.get("emphasis"), f"{path}.emphasis")

    if typ == "text":
        txt = L.get("text")
        if isinstance(txt, dict):                       # per-format text, like "box"
            for k in txt:
                if k not in FORMATS:
                    v.err(f"{path}.text", f"unknown format {k!r}; expected {sorted(FORMATS)}")
            for fmt in FORMATS:
                if not isinstance(txt.get(fmt), str) or not txt.get(fmt).strip():
                    v.err(f"{path}.text[{fmt}]", "must be a non-empty string")
        elif not isinstance(txt, str) or not txt.strip():
            v.err(f"{path}.text", "must be a non-empty string or a per-format dict {'9:16': ..., '16:9': ...}")
        v.enum(L.get("fit", "auto"), TEXT_FIT, f"{path}.fit", "text fit")
        v.enum(L.get("role", "headline"), TEXT_ROLES, f"{path}.role", "text role")
        v.font(L.get("font", "display"), f"{path}.font")
        v.enum(L.get("case", "as_is"), TEXT_CASES, f"{path}.case", "text case")
        v.enum(L.get("align", "center"), ALIGNS, f"{path}.align", "align")
        v.color(L.get("color"), f"{path}.color")
        v.color(L.get("highlight"), f"{path}.highlight", allow_none=True)
        v.fmt_boxes(L, path, *TEXT_BOX_RANGE)
        v.anim(L.get("in"), "text_in", f"{path}.in")
        v.anim(L.get("out"), "text_out", f"{path}.out")
        _validate_text_options(v, L, path)
    elif typ == "shape":
        src = L.get("source")
        v.enum(src, SHAPE_SOURCES, f"{path}.source", "shape source")
        if src == "contour":
            v.ref_contour(L.get("ref"), f"{path}.ref")
        elif src == "line":
            v.ref_line(L.get("ref"), f"{path}.ref")
        elif src == "primitive":
            v.enum(L.get("primitive"), VERBS["primitives"], f"{path}.primitive", "primitive")
            v.fmt_boxes(L, path)
        v.color(L.get("stroke"), f"{path}.stroke", allow_none=True)
        v.color(L.get("fill"), f"{path}.fill", allow_none=True)
        if L.get("stroke") is None and L.get("fill") is None:
            v.err(path, "shape needs a stroke or a fill colour")
        v.num(L, "stroke_width", path, 0.0, 0.2, required=False)
        v.num(L, "fill_opacity", path, 0.0, 1.0, required=False)
        v.num(L, "scale", path, 0.0, None, required=False)
        if L.get("offset") is not None:
            v.point(L.get("offset"), f"{path}.offset")
        v.anim(L.get("in"), "shape_in", f"{path}.in")
        v.anim(L.get("out"), "shape_out", f"{path}.out")
        echo = L.get("echo")
        if echo is not None:
            if not isinstance(echo, dict):
                v.err(f"{path}.echo", "must be an object or null")
            else:
                v.num(echo, "count", f"{path}.echo", 0, 8)
                v.num(echo, "spacing", f"{path}.echo", 0.0, 0.2, required=False)
                v.num(echo, "decay", f"{path}.echo", 0.0, 1.0, required=False)
    elif typ == "image":
        v.ref_image(L.get("src"), f"{path}.src")
        mask = L.get("mask", "rect")
        if isinstance(mask, str) and mask.startswith("contour:"):
            v.ref_contour(mask[len("contour:"):], f"{path}.mask")
        else:
            v.enum(mask, VERBS["masks"], f"{path}.mask", "mask")
        v.fmt_boxes(L, path)
        v.anim(L.get("in"), "image_in", f"{path}.in")
        v.anim(L.get("out"), "image_out", f"{path}.out")
        border = L.get("border")
        if isinstance(border, dict):
            v.color(border.get("color"), f"{path}.border.color")
            v.num(border, "width", f"{path}.border", 0.0, 0.2, required=False)
        if L.get("treatment") is not None:
            _validate_treatment(v, L.get("treatment"), f"{path}.treatment")


def validate_plan(plan: dict, analysis: dict | None) -> list[str]:
    """Return a list of human-readable problems; empty list means valid.

    ``analysis`` supplies the image / contour / line ids that ``src``/``ref``
    may point at. Pass ``None`` to skip reference checks.
    """
    if not isinstance(plan, dict):
        return ["plan: must be an object"]
    v = _V(plan, analysis)

    if plan.get("version") != PLAN_VERSION:
        v.err("version", f"must be {PLAN_VERSION}")
    if "seed" in plan and not _is_num(plan["seed"]):
        v.err("seed", "must be an integer")
    duration = v.num(plan, "duration", "plan", 0.0)
    if duration is not None and duration <= 0:
        v.err("duration", "must be > 0")
    v.num(plan, "fps", "plan", 1.0, 120.0)

    _validate_style(v, plan)
    _validate_audio(v, plan, duration)

    voice = plan.get("voice")
    if voice is not None and not isinstance(voice, dict):
        v.err("voice", "must be null or an object")

    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        v.err("scenes", "must be a non-empty list")
        return v.errors

    # Rule: scenes cover [0, duration] contiguously, sorted, no gaps.
    times = []
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            v.err(f"scenes[{i}]", "must be an object")
            times.append((None, None))
            continue
        t0 = v.num(sc, "t0", f"scenes[{i}]")
        t1 = v.num(sc, "t1", f"scenes[{i}]")
        times.append((t0, t1))
        if t0 is not None and t1 is not None and t1 <= t0 + _EPS:
            v.err(f"scenes[{i}]", f"t1 ({t1}) must be > t0 ({t0})")
    if times[0][0] is not None and abs(times[0][0]) > _EPS:
        v.err("scenes[0].t0", f"must be 0 (got {times[0][0]})")
    if duration is not None and times[-1][1] is not None and abs(times[-1][1] - duration) > _EPS:
        v.err(f"scenes[{len(scenes) - 1}].t1", f"must equal duration {duration} (got {times[-1][1]})")
    for i in range(len(scenes) - 1):
        a, b = times[i][1], times[i + 1][0]
        if a is not None and b is not None and abs(a - b) > _EPS:
            v.err(f"scenes[{i}]", f"gap/overlap: t1 {a} != scenes[{i + 1}].t0 {b} (scenes must be contiguous and sorted)")

    ids = set()
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue
        p = f"scenes[{i}]"
        sid = sc.get("id")
        if sid in ids:
            v.err(f"{p}.id", f"duplicate scene id {sid!r}")
        ids.add(sid)
        _validate_background(v, sc.get("background"), f"{p}.background")

        last = i == len(scenes) - 1
        tr = sc.get("transition_out")
        tr_dur = 0.0
        if tr is None:
            if not last:
                v.err(f"{p}.transition_out", "missing")
        elif not isinstance(tr, dict):
            v.err(f"{p}.transition_out", "must be an object")
        else:
            v.enum(tr.get("type"), TRANSITIONS, f"{p}.transition_out.type", "transition")
            d = v.num(tr, "dur", f"{p}.transition_out", 0.0, required=False)
            tr_dur = d or 0.0
            if tr.get("type") in DIRECTIONAL_TRANSITIONS:
                v.enum(tr.get("dir", "left"), DIRECTIONS, f"{p}.transition_out.dir", "direction")
            if not last and tr_dur > 0:
                t0, t1 = times[i]
                n0, n1 = times[i + 1]
                if None not in (t0, t1, n0, n1):
                    if tr_dur / 2 > (t1 - t0) + _EPS or tr_dur / 2 > (n1 - n0) + _EPS:
                        v.err(f"{p}.transition_out.dur", f"{tr_dur} too long: half must fit inside this and the next scene")
        if last:
            tr_dur = 0.0   # last transition is ignored by the renderer

        t0, t1 = times[i]
        if t0 is None or t1 is None:
            continue
        lo, hi = t0, t1 + tr_dur / 2
        if duration is not None:
            hi = min(hi, duration)
        layers = sc.get("layers")
        if layers is None:
            layers = []
        if not isinstance(layers, list):
            v.err(f"{p}.layers", "must be a list")
            continue
        for j, L in enumerate(layers):
            _validate_layer(v, L, f"{p}.layers[{j}]", lo, hi)

    return v.errors


# --------------------------------------------------------------------------
# LLM-facing schema description
# --------------------------------------------------------------------------

_EXAMPLE_PLAN = {
    "version": 1, "seed": 7, "duration": 6.0, "fps": 30,
    "style": {"palette": ["#101418", "#f4f1ea", "#ff5a36", "#3a7bff"], "background": "palette:0",
              "fonts": {"display": "Big Shoulders Display", "body": "Space Grotesk"}, "motion": "snappy", "energy": 0.75,
              "grain": 0.1, "vignette": 0.2, "beat_flash": 0.15},
    "audio": {"source": "synth", "bpm": 120, "key": "A", "scale": "minor", "mood": "driving",
              "beats": [0.0, 0.5, 1.0, "..."], "downbeats": [0.0, 2.0, 4.0],
              "sections": [{"t": 0.0, "label": "intro"}, {"t": 4.0, "label": "drop"}],
              "risers": [{"t0": 2.0, "t1": 4.0}], "sfx": [{"t": 4.0, "type": "impact", "gain": 0.8}]},
    "voice": None,
    "scenes": [
        {"id": "s0", "t0": 0.0, "t1": 4.0,
         "background": {"type": "image", "src": "img0", "focus": [0.5, 0.4],
                        "motion": {"verb": "ken_burns", "zoom": [1.0, 1.1], "pan": [[0, 0], [0.02, -0.01]]},
                        "treatment": {"duotone": None, "blur": 0.0, "darken": 0.3}},
         "layers": [
             {"type": "shape", "source": "contour", "ref": "img0:c0", "primitive": None, "box": None,
              "stroke": "palette:2", "stroke_width": 0.006, "fill": None, "fill_opacity": 0.0,
              "offset": [0, 0], "scale": 1.0, "t0": 0.0, "t1": 4.0,
              "in": {"verb": "stroke_draw", "dur": 1.0}, "out": {"verb": "fade", "dur": 0.2},
              "emphasis": [{"verb": "pulse", "on": "beats", "amount": 0.06}],
              "echo": {"count": 2, "spacing": 0.015, "decay": 0.5}},
             {"type": "text", "text": "Make it move", "role": "headline", "font": "display", "case": "upper",
              "color": "palette:1", "highlight": None, "align": "center",
              "box": {"9:16": [0.08, 0.64, 0.84, 0.16], "16:9": [0.06, 0.62, 0.55, 0.22]},
              "t0": 0.5, "t1": 4.0, "in": {"verb": "word_stagger", "dur": 0.5, "stagger_beats": 0.5},
              "out": {"verb": "slide_up", "dur": 0.25}, "emphasis": []}],
         "transition_out": {"type": "zoom_blur", "dur": 0.3}},
        {"id": "s1", "t0": 4.0, "t1": 6.0,
         "background": {"type": "gradient", "colors": ["palette:0", "palette:3"], "angle": 120,
                        "motion": {"verb": "drift"}},
         "layers": [], "transition_out": {"type": "cut", "dur": 0.0}}],
}


def plan_schema_prompt() -> str:
    """Compact system prompt describing the plan JSON for an LLM / VLM planner."""
    V = VERBS
    j = ", ".join
    example = json.dumps(_EXAMPLE_PLAN, separators=(",", ":")).replace('"..."', "...")
    return f"""You are the director of a 5-15 second motion graphic (one shot, rendered at 9:16 and 16:9).
You receive an ANALYSIS of the user's inputs (text lines, images with ids, palettes, empty regions,
contours, lines; optional audio beats) and must answer with ONE JSON object: the PLAN. Output JSON only,
no comments, no prose.

CONVENTIONS
- Times are seconds, absolute on the clip timeline. Boxes and points are normalised 0..1, [x, y, w, h],
  top-left origin. Frame-space boxes are given per format: {{"9:16": [...], "16:9": [...]}}.
- Colours are "#rrggbb" or "palette:N" (index into style.palette).
- Ids: images "img0", "img1"...; contours "img0:c0"...; lines "img0:l0"... Use only ids present in the analysis.

TOP LEVEL
version (=1), seed (int), duration (5..15), fps (30), style, audio, voice (null), scenes.

style: palette (3-6 hex colours: [0]=dark, [1]=light, [2..]=accents, prefer colours taken from the images),
background (colour), fonts {{"display": F, "body": F}} with F a bundled font. Fonts (roles; moods; width):
{prompt_guide()}.
Pick a characterful display face for the mood and a "text" face from its pairing as body; vary choices.
motion in [{j(V["motion"])}], energy 0..1 (amplitude and cut density), grain 0..1, vignette 0..1,
beat_flash 0..1.

audio: source ("synth", or "user" if the analysis has audio), bpm, key in [{j(KEYS)}], scale in
[{j(SCALES)}], mood in [{j(V["moods"])}], beats and downbeats (seconds, within [0, duration]; 4/4,
a downbeat every 4 beats; copy them from the analysis when user audio exists), sections [{{"t", "label"}}]
(labels: {j(SECTION_LABELS)}; the music gets busier in "drop"), risers [{{"t0", "t1"}}] (riser ending at t1,
usually the bar before a transition), sfx [{{"t", "type", "gain"}}] with type in [{j(V["sfx"])}]
(impact on downbeats that start a scene, whoosh on push/wipe transitions).

scenes: list of {{id, t0, t1, background, layers, transition_out}}.
- background.type in [{j(BACKGROUND_TYPES)}].
  image: src (image id), focus [x, y] image-space point kept in frame,
  motion {{"verb", ...}} with verb in [{j(V["background_motion"])}] (ken_burns: "zoom" [z0, z1], "pan" [[dx,dy],[dx,dy]];
  beat_zoom: "amount"), treatment {{"duotone": [c_dark, c_light] or null, "blur" 0..1, "darken" 0..1}}.
  color: "color". gradient: "colors" [c, c], "angle" degrees. shapes: "color", "colors" (abstract animated shapes).
- transition_out: {{"type", "dur"}} with type in [{j(TRANSITIONS)}]; push and wipe also take "dir" in
  [{j(DIRECTIONS)}]. It is centred on scene.t1 (spans t1-dur/2 .. t1+dur/2). Fast/energetic: zoom_blur, glitch,
  push, flash (0.2-0.35 s). Calm: crossfade, luma_wipe, shape_reveal (0.5-0.8 s). The last scene's
  transition_out is ignored (use {{"type": "cut", "dur": 0}}).

LAYERS (drawn in list order; later = on top). Every layer has t0, t1, in, out, emphasis.
- in/out: {{"verb", "dur"}}. emphasis: list of {{"verb", "on", "amount"}} with verb in [{j(V["emphasis"])}],
  on in [{j(EMPHASIS_ON)}] or a list of times.
- text: text, role in [{j(TEXT_ROLES)}], font ("display" | "body"), case in [{j(TEXT_CASES)}], color,
  highlight (colour of a box behind the text, or null), align in [{j(ALIGNS)}], box (per format; text auto-fits),
  in verbs [{j(V["text_in"])}] (word_stagger takes "stagger_beats"; scramble resolves at the end of "dur";
  row_slide takes "rows", "scale", "travel", "track", "drift", "fade"), out verbs [{j(V["text_out"])}].
  "font" may also name a bundled font directly (one word in a contrasting face = its own layer).
  Bold options: "axes" {{"wght": 300}} (static variation); "vary" {{"axis", "range", "period_beats", "phase",
  "scale"}} per-letter variable-font wave; "outline" {{"width" (em), "swap": beats|downbeats|onsets}} outline/fill
  swap; "window" {{"mode": "fill" (letters show image "src") | "knockout" (a "plate" colour with the letters cut
  out, showing the background), "zoom"}}. Shape options: "rotate" deg (or [[t, deg], ...]; in/out "spin" deg),
  "overscale" (> 1 = giant type cropped by the frame; text boxes may run -1..2), "path" {{"type": arc|circle|wave|line}}
  (glyphs on a curve; "travel"), "letter_scale" "random"|"wave", "backdrop" {{"type": dim|blur|band}} behind hairlines.
- shape: source in [{j(SHAPE_SOURCES)}]. contour/line: "ref" = analysis id, drawn where that shape sits in the
  scene's background image (mapped through its crop) - this is how graphics trace the imagery. primitive:
  "primitive" in [{j(V["primitives"])}] plus a per-format "box" ("wave" reacts to the audio). Also stroke,
  stroke_width (fraction of the frame short side, ~0.004-0.01), fill, fill_opacity, offset [dx, dy], scale,
  echo {{"count", "spacing", "decay"}} or null. in verbs [{j(V["shape_in"])}] (stroke_draw traces the path),
  out verbs [{j(V["shape_out"])}].
- image: src (image id), mask in [{j(V["masks"])}] or "contour:<contour id>", box per format,
  border {{"color", "width"}} or null, in verbs [{j(V["image_in"])}], out verbs [{j(V["image_out"])}].

HARD RULES (plans that break them are rejected)
1. Scenes are sorted and contiguous: scenes[0].t0 = 0, scenes[i].t1 = scenes[i+1].t0, last t1 = duration.
2. Each layer's [t0, t1] lies within [scene.t0, scene.t1 + transition_out.dur/2], with t1 > t0.
3. Every src/ref names an existing analysis id; every "palette:N" is in range; fonts are bundled; every verb,
   type and enum value comes from the lists above.
4. Every text, image and primitive layer has a box for both "9:16" and "16:9", inside 0..1 (text: -1..2).
5. Half of a transition's dur must fit inside both scenes it joins.

CRAFT
- Put scene boundaries on downbeats; scenes of at least one bar, ideally 2 s or more.
- One idea per scene: a short headline (<= 6 words), optionally a sub line. Place headlines in the image's
  empty regions (mapped to each format) or the lower third; keep 5% safe margins.
- Text enters on a beat; outline the 1-3 largest contours with stroke_draw in an accent colour; darken busy
  images (treatment.darken 0.2-0.4) so light text reads.
- Match motion, transitions, fonts and music mood to the content and the user's words.

EXAMPLE (beats list shortened)
{example}
"""
