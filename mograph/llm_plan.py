"""Tolerant plan extraction for LLM / VLM planner output.

``extract_plan(text)`` turns raw model output into a plan that passes
``mograph.plan.validate_plan``: it strips think blocks and code fences, cuts the
first balanced JSON object out of the prose, fixes common JSON slips (trailing
commas, comments, ``...``, Python literals, truncated output), then repairs the
plan itself (defaults, enum synonyms, colours, contiguous scenes, beat grid,
boxes, time windows) and validates. Every repair is logged so planner quality
can be measured (pass ``log=[]``).
"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

from .plan import (ALIGNS, AUDIO_SOURCES, BACKGROUND_TYPES, BUNDLED_FONTS, DIRECTIONAL_TRANSITIONS,
                   DIRECTIONS, DITHERS, EMPHASIS_ON, FILM_PRESETS, FONT_SLOTS, FORMATS, HALFTONE_SHAPES,
                   KEYS, LAYER_TYPES, PLAN_VERSION, SCALES, SECTION_LABELS, SHAPE_SOURCES, TEXT_CASES,
                   TEXT_ROLES, TRANSITIONS, TREATMENT_NUMS, VERBS, validate_plan)

__all__ = ["extract_json_text", "loads_tolerant", "repair_plan", "extract_plan", "PlanExtractError"]


class PlanExtractError(ValueError):
    """Raised when no valid plan can be recovered. ``.errors`` lists what is still wrong."""

    def __init__(self, msg: str, errors: list[str] | None = None, plan: dict | None = None):
        super().__init__(msg)
        self.errors = errors or []
        self.plan = plan


# --------------------------------------------------------------------------
# Text -> JSON
# --------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json|jsonc|JSON)?\s*\n?(.*?)(?:```|$)", re.S)


def extract_json_text(text: str, log: list | None = None) -> str:
    """Return the substring holding the plan object (best effort, may be truncated)."""
    log = log if log is not None else []
    s = text or ""
    if _THINK_RE.search(s):
        s = _THINK_RE.sub("", s)
        log.append("text: removed <think> block")
    if "```" in s:
        m = _FENCE_RE.search(s)
        if m and "{" in m.group(1):
            s = m.group(1)
            log.append("text: stripped code fence")
    i = s.find("{")
    if i < 0:
        return s.strip()
    if s[:i].strip():
        log.append("text: dropped prose before JSON")
    j = _balanced_end(s, i)
    if j is None:
        return s[i:].strip()
    if s[j + 1:].strip():
        log.append("text: dropped text after JSON")
    return s[i:j + 1]


def _balanced_end(s: str, i: int) -> int | None:
    depth, in_str, esc = 0, False, False
    for k in range(i, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return k
    return None


def _strip_outside_strings(s: str, fn) -> str:
    """Apply ``fn`` to every run of text outside JSON strings."""
    out, buf, in_str, esc = [], [], False, False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            out.append(fn("".join(buf)))
            buf = []
            out.append(ch)
            in_str = True
        else:
            buf.append(ch)
    out.append(fn("".join(buf)))
    return "".join(out)


def _cleanup(s: str, log: list) -> str:
    before = s

    def fix(seg: str) -> str:
        seg = re.sub(r"//[^\n]*", "", seg)
        seg = re.sub(r"/\*.*?\*/", "", seg, flags=re.S)
        seg = re.sub(r",\s*\.\.\.\s*(?=[\]}])", "", seg)        # [1, 2, ...]
        seg = re.sub(r"\.\.\.\s*,?", "", seg)
        seg = re.sub(r"\bNone\b", "null", seg)
        seg = re.sub(r"\bTrue\b", "true", seg)
        seg = re.sub(r"\bFalse\b", "false", seg)
        seg = re.sub(r",(\s*[\]}])", r"\1", seg)                 # trailing commas
        return seg

    s = _strip_outside_strings(s, fix)
    s = _strip_outside_strings(s, lambda seg: re.sub(r",(\s*[\]}])", r"\1", seg))
    if s != before:
        log.append("json: removed trailing commas / comments / ellipsis / python literals")
    return s


def _close_truncated(s: str) -> str | None:
    """Close a JSON text cut off mid-way: drop the incomplete tail, then close brackets."""
    cut = len(s)
    for _ in range(400):
        t = s[:cut].rstrip()
        stack, in_str, esc = [], False, False
        for ch in t:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if not in_str:
            cand = re.sub(r"[,:]\s*$", "", t)
            cand = re.sub(r'[{,]\s*"[^"]*"\s*$', lambda m: m.group(0)[0] if m.group(0)[0] == "{" else "", cand)
            cand = cand + "".join(reversed(stack))
            cand = re.sub(r",(\s*[\]}])", r"\1", cand)
            try:
                json.loads(cand)
                return cand
            except ValueError:
                pass
        # step back to the previous comma / bracket outside a string and retry
        k = max(t.rfind(","), t.rfind("{"), t.rfind("["))
        if k <= 0:
            return None
        cut = k if t[k] == "," else k + 1
    return None


def loads_tolerant(text: str, log: list | None = None) -> dict:
    """Parse model output into a dict, logging every fix. Raises PlanExtractError."""
    log = log if log is not None else []
    s = extract_json_text(text, log)
    try:
        obj = json.loads(s)
    except ValueError:
        s2 = _cleanup(s, log)
        try:
            obj = json.loads(s2)
        except ValueError as e:
            closed = _close_truncated(s2)
            if closed is None:
                raise PlanExtractError(f"no parsable JSON object ({e})") from e
            log.append("json: output truncated; closed open brackets")
            obj = json.loads(closed)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
        log.append("json: took first object of a list")
    if not isinstance(obj, dict):
        raise PlanExtractError("JSON is not an object")
    if "plan" in obj and isinstance(obj["plan"], dict) and "scenes" not in obj:
        obj = obj["plan"]
        log.append("json: unwrapped {'plan': ...}")
    return obj


# --------------------------------------------------------------------------
# Plan repair helpers
# --------------------------------------------------------------------------

_NAMED = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00a050", "blue": "#2050ff",
          "yellow": "#ffd400", "orange": "#ff8000", "pink": "#ff4fa0", "purple": "#7a3cff", "cyan": "#00d0e0",
          "teal": "#2a8a86", "gray": "#808080", "grey": "#808080", "gold": "#d4a020", "navy": "#0a1a3a"}
_HEX6 = re.compile(r"^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")
_HEX3 = re.compile(r"^#?([0-9a-fA-F]{3})$")
_PALN = re.compile(r"^\s*palette\s*[:\[]\s*(\d+)\s*\]?\s*$", re.I)
_DEFAULT_PAL = ["#101418", "#f4f1ea", "#ff5a36", "#3a7bff"]

_SYN = {
    "transition": {"fade": "crossfade", "dissolve": "crossfade", "cross_fade": "crossfade", "crossdissolve": "crossfade",
                   "light_leak": "flash", "lightleak": "flash", "leak": "flash", "whip": "push", "whip_pan": "push",
                   "slide": "push", "zoom": "zoom_blur", "blur": "zoom_blur", "whip_blur": "zoom_blur",
                   "strobe": "flash", "white_flash": "flash", "dip": "crossfade", "dip_to_black": "crossfade",
                   "mask": "shape_reveal", "iris": "shape_reveal", "luma": "luma_wipe", "grid_wipe": "wipe",
                   "hard_cut": "cut", "none": "cut", "match_cut": "cut", "morph": "crossfade"},
    "text_in": {"fade_in": "fade", "typewriter": "type_on", "type": "type_on", "stagger": "word_stagger",
                "slide": "slide_up", "slide_in": "slide_up", "pop": "scale_pop", "scale": "scale_pop",
                "scale_in": "scale_pop", "blur": "blur_in", "wipe": "mask_wipe", "scramble": "kinetic_split",
                "kinetic": "kinetic_split", "track_in": "fade", "letter_stagger": "word_stagger", "none": "cut"},
    "text_out": {"fade_out": "fade", "slide": "slide_up", "blur": "blur_out", "scale": "scale_down",
                 "wipe": "mask_wipe", "track_out": "fade", "none": "cut", "scale_up": "scale_down"},
    "shape_in": {"draw": "stroke_draw", "draw_on": "stroke_draw", "stroke": "stroke_draw", "pop": "scale_pop",
                 "scale": "scale_pop", "fade_in": "fade", "grow": "scale_pop", "none": "cut"},
    "shape_out": {"fade_out": "fade", "scale": "scale_down", "shrink": "scale_down", "none": "cut"},
    "image_in": {"fade_in": "fade", "pop": "scale_pop", "scale": "scale_pop", "slide": "slide_up", "none": "cut"},
    "image_out": {"fade_out": "fade", "scale": "scale_down", "slide": "slide_up", "none": "cut"},
    "emphasis": {"flash": "pulse", "strobe": "pulse", "scale": "pulse", "punch": "pulse", "jitter": "shake",
                 "glitch": "shake", "color": "color_flip", "colour_flip": "color_flip", "underline": "underline_draw"},
    "background_motion": {"push_in": "ken_burns", "zoom": "ken_burns", "slow_push": "ken_burns", "pan": "drift",
                          "parallax": "parallax_split", "none": "static", "still": "static", "scroll": "drift"},
    "primitives": {"line": "bar", "square": "rect", "rectangle": "rect", "dots": "dot_field", "dot": "dot_field",
                   "heart": "circle", "sine": "wave", "disc": "circle", "plus": "cross", "x": "cross",
                   "lines": "grid", "ellipse": "circle", "rounded_rect": "rect", "triangle": "rect"},
    "masks": {"rectangle": "rect", "round": "rounded", "rounded_rect": "rounded", "ellipse": "circle", "none": "rect"},
    "sfx": {"boom": "impact", "thud": "impact", "ping": "click", "sonar": "click", "sonar_ping": "click",
            "tap": "click", "tick": "click", "swoosh": "whoosh", "rise": "riser", "snap": "click",
            "crash": "impact", "tape_stop": "glitch", "scratch": "glitch", "pad": "swell"},
    "moods": {"chill": "calm", "ambient": "calm", "lofi": "calm", "lo-fi": "calm", "warm": "calm", "soft": "calm",
              "energetic": "driving", "hype": "driving", "aggressive": "driving", "intense": "driving",
              "mysterious": "dark", "moody": "dark", "nostalgic": "calm", "happy": "bright", "uplifting": "bright",
              "romantic": "bright", "retro": "driving", "cinematic": "epic", "dramatic": "epic", "fun": "playful"},
    "motion": {"slow": "smooth", "soft": "smooth", "calm": "smooth", "fast": "snappy", "punchy": "snappy",
               "elastic": "bouncy", "glitchy": "glitch", "hectic": "snappy"},
    "background": {"solid": "color", "colour": "color", "image_bg": "image", "abstract": "shapes", "linear": "gradient"},
    "treatments": {"film": "film_grade", "filmgrade": "film_grade", "grade": "film_grade", "lightleak": "light_leak",
                   "leak": "light_leak", "dust": "dust_scratches", "scratches": "dust_scratches",
                   "posterize": "posterize_flat", "poster": "posterize_flat", "screen_print": "screenprint",
                   "photocopy": "xerox", "noise": "grain", "film_grain": "grain"},
}
# 2026-10-30 (user: "always use our favourited fonts, not generic ones"): the names a model may write (serif, helvetica, ...) map to LIKED faces only (mograph/fonts/liked.json), characterful ones first.
_FONT_ALIASES = {"serif": "Fascinate", "playfair": "Fascinate", "bebas": "BigShouldersDisplay", "condensed": "BigShouldersDisplay",
                 "mono": "MajorMonoDisplay", "monospace": "MajorMonoDisplay", "courier": "MajorMonoDisplay", "garamond": "Fascinate", "georgia": "Fascinate", "times": "Fascinate",  # font-alias-ok
                 "sans": "Anybody", "sans-serif": "Anybody", "grotesk": "Anybody", "helvetica": "Anybody", "arial": "Anybody", "futura": "Anybody",  # font-alias-ok
                 "impact": "Ultra"}
_KEY_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _f(x, default: float) -> float:
    if _isnum(x):
        return float(x)
    if isinstance(x, str):
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", x)
        if m:
            return float(m.group(1))
    return float(default)


def _clamp(v: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


class _R:
    """Repair context: palette size, analysis ids, and the fix log."""

    def __init__(self, plan: dict, analysis: dict | None, log: list):
        self.plan = plan
        self.log = log
        self.check_refs = analysis is not None
        self.images, self.contours, self.lines = set(), set(), set()
        for img in (analysis or {}).get("images") or []:
            self.images.add(img.get("id"))
            for c in img.get("contours") or []:
                self.contours.add(c.get("id"))
            for i, ln in enumerate(img.get("lines") or []):
                self.lines.add(ln.get("id") or f"{img.get('id')}:l{i}")
        self.n_pal = 0

    def fix(self, msg: str):
        self.log.append(msg)

    def enum(self, value, allowed, kind: str, default, path: str):
        if value in allowed:
            return value
        if isinstance(value, str):
            v = value.strip().lower().replace("-", "_").replace(" ", "_")
            if v in allowed:
                self.fix(f"enum: {path} {value!r} -> {v!r}")
                return v
            syn = _SYN.get(kind, {}).get(v)
            if syn in allowed:
                self.fix(f"enum: {path} {value!r} -> {syn!r}")
                return syn
        self.fix(f"enum: {path} {value!r} -> default {default!r}")
        return default

    def hexcol(self, c) -> str | None:
        if isinstance(c, (list, tuple)) and len(c) >= 3 and all(_isnum(v) for v in c[:3]):
            vals = [float(v) for v in c[:3]]
            if all(v <= 1.0 for v in vals):
                vals = [v * 255 for v in vals]
            return "#" + "".join(f"{int(_clamp(round(v), 0, 255)):02x}" for v in vals)
        if not isinstance(c, str):
            return None
        s = c.strip()
        m = _HEX6.match(s)
        if m:
            return "#" + m.group(1).lower()
        m = _HEX3.match(s)
        if m:
            return "#" + "".join(ch * 2 for ch in m.group(1)).lower()
        return _NAMED.get(s.lower())

    def color(self, c, path: str, default: str | None = "palette:1", allow_none: bool = False):
        if c is None:
            if allow_none:
                return None
            if default is not None:
                self.fix(f"color: {path} missing -> {default}")
            return default
        if isinstance(c, str):
            m = _PALN.match(c)
            if m:
                n = int(m.group(1))
                if n < self.n_pal:
                    out = f"palette:{n}"
                    if out != c:
                        self.fix(f"color: {path} {c!r} -> {out!r}")
                    return out
                out = f"palette:{n % max(self.n_pal, 1)}"
                self.fix(f"color: {path} {c!r} out of range -> {out!r}")
                return out
        h = self.hexcol(c)
        if h is not None:
            if h != c:
                self.fix(f"color: {path} {c!r} -> {h!r}")
            return h
        if allow_none:
            self.fix(f"color: {path} {c!r} unreadable -> null")
            return None
        self.fix(f"color: {path} {c!r} unreadable -> {default}")
        return default

    def box(self, b, path: str, default):
        if isinstance(b, dict) and all(k in b for k in ("x", "y", "w", "h")):
            b = [b["x"], b["y"], b["w"], b["h"]]
            self.fix(f"box: {path} object -> list")
        if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(_isnum(v) for v in b):
            self.fix(f"box: {path} {b!r} -> default")
            return list(default)
        x, y, w, h = (float(v) for v in b)
        if max(x, y, w, h) > 1.5:            # pixels or percent
            sc = 100.0 if max(x + w, y + h) <= 100.5 else max(x + w, y + h)
            x, y, w, h = x / sc, y / sc, w / sc, h / sc
            self.fix(f"box: {path} rescaled to 0..1")
        nx, ny = _clamp(x, 0.0, 0.98), _clamp(y, 0.0, 0.98)
        nw, nh = _clamp(w, 0.02, 1.0 - nx), _clamp(h, 0.02, 1.0 - ny)
        out = [round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)]
        if any(abs(a - b_) > 1e-6 for a, b_ in zip(out, (x, y, w, h))):
            self.fix(f"box: {path} clamped into frame")
        return out

    def fmt_boxes(self, L: dict, path: str, default=(0.08, 0.4, 0.84, 0.2)):
        box = L.get("box")
        if isinstance(box, (list, tuple)) or (isinstance(box, dict) and not any(f in box for f in FORMATS)
                                              and all(k in box for k in ("x", "y", "w", "h"))):
            self.fix(f"box: {path}.box single box -> both formats")
            box = {f: box for f in FORMATS}
        if not isinstance(box, dict):
            self.fix(f"box: {path}.box missing -> default")
            box = {}
        out = {}
        for fmt in FORMATS:
            src = box.get(fmt)
            if src is None:
                alt = [v for k, v in box.items() if k != fmt]
                if alt:
                    self.fix(f"box: {path}.box[{fmt}] missing -> copied")
                    src = alt[0]
            out[fmt] = self.box(src, f"{path}.box[{fmt}]", default)
        L["box"] = out


# --------------------------------------------------------------------------
# Plan repair
# --------------------------------------------------------------------------

def _repair_treatment(r: _R, tr, path: str):
    if tr is None:
        return None
    items = tr if isinstance(tr, list) else [tr]
    out = []
    for i, t in enumerate(items):
        p = f"{path}[{i}]" if isinstance(tr, list) else path
        if isinstance(t, str):
            t = {"type": t}
            r.fix(f"treatment: {p} string -> object")
        if not isinstance(t, dict):
            r.fix(f"treatment: {p} dropped (not an object)")
            continue
        typ = t.get("type") or t.get("name")
        if typ not in VERBS["treatments"]:
            v = str(typ or "").strip().lower().replace("-", "_").replace(" ", "_")
            v = v if v in VERBS["treatments"] else _SYN["treatments"].get(v)
            if v is None:
                r.fix(f"treatment: {p} unknown type {typ!r} dropped")
                continue
            r.fix(f"enum: {p}.type {typ!r} -> {v!r}")
            typ = v
        t = dict(t)
        t.pop("name", None)
        t["type"] = typ
        for k in ("ink", "paper", "tone"):
            if k in t:
                t[k] = r.color(t[k], f"{p}.{k}", default=None, allow_none=True)
                if t[k] is None:
                    t.pop(k)
        for k, (lo, hi) in (("inks", (1, 4)), ("colors", (2, 5)), ("palette", (2, 16))):
            if k in t:
                cols = t[k] if isinstance(t[k], list) else []
                cols = [c for c in (r.color(c, f"{p}.{k}", default=None, allow_none=True) for c in cols) if c]
                if lo <= len(cols) <= hi:
                    t[k] = cols
                else:
                    r.fix(f"treatment: {p}.{k} invalid -> palette default")
                    t.pop(k)
        for k, (lo, hi) in TREATMENT_NUMS.items():
            if k in t:
                if t[k] is None or not (_isnum(t[k]) or isinstance(t[k], str)):
                    t.pop(k)
                    r.fix(f"treatment: {p}.{k} dropped")
                    continue
                v = _clamp(_f(t[k], lo if lo is not None else 0.0), lo, hi)
                if v != t[k]:
                    r.fix(f"treatment: {p}.{k} {t[k]!r} -> {v}")
                t[k] = v
        if "preset" in t:
            t["preset"] = r.enum(t["preset"], FILM_PRESETS, "film_preset", "warm", f"{p}.preset")
        if "shape" in t:
            t["shape"] = r.enum(t["shape"], HALFTONE_SHAPES, "halftone_shape", "dot", f"{p}.shape")
        if "dither" in t:
            t["dither"] = r.enum(t["dither"], DITHERS, "dither", "noise", f"{p}.dither")
        out.append(t)
    if not out:
        return None
    return out if isinstance(tr, list) or len(out) > 1 else out[0]


def _repair_style(r: _R, plan: dict):
    style = plan.get("style")
    if not isinstance(style, dict):
        r.fix("style: missing -> defaults")
        style = {}
    pal = style.get("palette")
    if not isinstance(pal, list):
        pal = []
    good = []
    for c in pal:
        h = r.hexcol(c)
        if h is None:
            r.fix(f"style.palette: dropped {c!r}")
            continue
        good.append(h)
    if len(good) > 6:
        r.fix("style.palette: trimmed to 6")
        good = good[:6]
    if len(good) < 3:
        r.fix(f"style.palette: {len(good)} colours -> padded to 3")
        for c in _DEFAULT_PAL:
            if len(good) >= 3:
                break
            if c not in good:
                good.append(c)
    if len(good) == len(pal) and good != pal:
        r.fix("style.palette: hex normalised")
    style["palette"] = good
    r.n_pal = len(good)
    style["background"] = r.color(style.get("background", "palette:0"), "style.background", "palette:0")
    fonts = style.get("fonts")
    if isinstance(fonts, str):
        fonts = {"display": fonts, "body": fonts}
        r.fix("style.fonts: string -> object")
    if not isinstance(fonts, dict):
        r.fix("style.fonts: missing -> Big Shoulders Display/Space Grotesk")
        fonts = {}
    for slot, dflt in (("display", "Big Shoulders Display"), ("body", "Space Grotesk")):
        f = fonts.get(slot)
        if f in BUNDLED_FONTS:
            continue
        g = None
        if isinstance(f, str):
            low = f.strip().lower()
            g = next((b for b in BUNDLED_FONTS if b.lower() == low or b.lower().replace(" ", "") == low.replace(" ", "")),
                     None) or next((v for k, v in _FONT_ALIASES.items() if k in low), None)
        fonts[slot] = g or dflt
        r.fix(f"style.fonts.{slot}: {f!r} -> {fonts[slot]!r}")
    style["fonts"] = {"display": fonts["display"], "body": fonts["body"]}
    energy = _clamp(_f(style.get("energy"), 0.5), 0.0, 1.0)
    if style.get("energy") != energy:
        r.fix(f"style.energy: {style.get('energy')!r} -> {energy}")
    style["energy"] = energy
    style["motion"] = r.enum(style.get("motion"), VERBS["motion"], "motion",
                             "snappy" if energy >= 0.6 else "smooth", "style.motion")
    for k in ("grain", "vignette", "beat_flash"):
        if k in style:
            v = style[k]
            nv = _clamp(_f(v, 0.0), 0.0, 1.0) if v is not None else 0.0
            if nv != v:
                r.fix(f"style.{k}: {v!r} -> {nv}")
            style[k] = nv
    if "treatment" in style:
        style["treatment"] = _repair_treatment(r, style.get("treatment"), "style.treatment")
    plan["style"] = style


def _grid(bpm: float, duration: float):
    beat = 60.0 / bpm
    n = int(math.floor(duration / beat + 1e-6)) + 1
    beats = [round(k * beat, 4) for k in range(n) if k * beat <= duration + 1e-6]
    return beats, beats[::4]


def _repair_audio(r: _R, plan: dict, duration: float, has_user_audio: bool):
    audio = plan.get("audio")
    if not isinstance(audio, dict):
        r.fix("audio: missing -> defaults")
        audio = {}
    audio["source"] = r.enum(audio.get("source"), AUDIO_SOURCES if has_user_audio else ["synth"], "source",
                             "user" if has_user_audio else "synth", "audio.source")
    bpm = _f(audio.get("bpm"), 100.0)
    nb = _clamp(bpm, 50.0, 190.0)
    if audio.get("bpm") != nb:
        r.fix(f"audio.bpm: {audio.get('bpm')!r} -> {nb}")
    audio["bpm"] = nb
    key = audio.get("key")
    if key not in KEYS:
        k = str(key or "").strip()
        scale_hint = None
        m = re.match(r"^([A-Ga-g])([#b]?)\s*(m|min|minor|maj|major)?$", k)
        nk = "A"
        if m:
            base = m.group(1).upper() + m.group(2)
            nk = _KEY_FLATS.get(base, base)
            if m.group(3):
                scale_hint = "minor" if m.group(3).startswith("m") and not m.group(3).startswith("maj") else "major"
        if nk not in KEYS:
            nk = "A"
        r.fix(f"audio.key: {key!r} -> {nk!r}")
        audio["key"] = nk
        if scale_hint and audio.get("scale") not in SCALES:
            audio["scale"] = scale_hint
    audio["scale"] = r.enum(audio.get("scale"), SCALES, "scale", "minor", "audio.scale")
    energy = plan["style"]["energy"]
    audio["mood"] = r.enum(audio.get("mood"), VERBS["moods"], "moods", "driving" if energy >= 0.6 else "calm",
                           "audio.mood")
    beats, downs = _grid(nb, duration)
    b_in, d_in = audio.get("beats"), audio.get("downbeats")

    def ok(arr):
        return (isinstance(arr, list) and arr and all(_isnum(t) for t in arr)
                and all(b >= a for a, b in zip(arr, arr[1:])) and arr[0] >= 0 and arr[-1] <= duration + 1e-6)

    if not (ok(b_in) and abs(len(b_in) - len(beats)) <= 1):
        r.fix("audio.beats: regenerated from bpm")
        audio["beats"] = beats
    if not (ok(d_in) and abs(len(d_in) - len(downs)) <= 1):
        r.fix("audio.downbeats: regenerated from bpm")
        audio["downbeats"] = downs
    secs = []
    for i, s in enumerate(audio.get("sections") or []):
        if not isinstance(s, dict) or not (_isnum(s.get("t")) or isinstance(s.get("t"), str)):
            r.fix(f"audio.sections[{i}]: dropped")
            continue
        t = _clamp(_f(s.get("t"), 0.0), 0.0, duration)
        lab = s.get("label")
        if not isinstance(lab, str):
            lab = "build"
            r.fix(f"audio.sections[{i}].label: missing -> 'build'")
        elif lab not in SECTION_LABELS:
            r.log.append(f"note: audio.sections[{i}].label {lab!r} not in advisory list")
        secs.append({"t": t, "label": lab})
    audio["sections"] = sorted(secs, key=lambda s: s["t"])
    risers = []
    for i, x in enumerate(audio.get("risers") or []):
        if not isinstance(x, dict):
            r.fix(f"audio.risers[{i}]: dropped")
            continue
        t0 = _clamp(_f(x.get("t0"), 0.0), 0.0, duration)
        t1 = _clamp(_f(x.get("t1"), t0 + 1.0), 0.0, duration)
        if t1 <= t0:
            r.fix(f"audio.risers[{i}]: t1 <= t0, dropped")
            continue
        if (t0, t1) != (x.get("t0"), x.get("t1")):
            r.fix(f"audio.risers[{i}]: times clamped")
        risers.append({"t0": t0, "t1": t1})
    audio["risers"] = risers
    sfx = []
    for i, x in enumerate(audio.get("sfx") or []):
        if not isinstance(x, dict):
            r.fix(f"audio.sfx[{i}]: dropped")
            continue
        t = _f(x.get("t"), -1.0)
        if t < 0 or t > duration:
            r.fix(f"audio.sfx[{i}]: t {x.get('t')!r} outside clip, dropped")
            continue
        typ = r.enum(x.get("type"), VERBS["sfx"], "sfx", "hit", f"audio.sfx[{i}].type")
        g = _clamp(_f(x.get("gain"), 0.7), 0.0, 2.0)
        sfx.append({"t": t, "type": typ, "gain": g})
    audio["sfx"] = sfx
    plan["audio"] = audio


def _repair_anim(r: _R, a, kind: str, path: str, default_verb: str, default_dur: float):
    if isinstance(a, str):
        a = {"verb": a}
        r.fix(f"anim: {path} string -> object")
    if not isinstance(a, dict):
        r.fix(f"anim: {path} missing -> {default_verb}")
        return {"verb": default_verb, "dur": default_dur}
    a = dict(a)
    if "verb" not in a and "type" in a:
        a["verb"] = a.pop("type")
        r.fix(f"anim: {path} 'type' -> 'verb'")
    a["verb"] = r.enum(a.get("verb"), VERBS[kind], kind, default_verb, f"{path}.verb")
    d = a.get("dur")
    if d is None:
        a["dur"] = default_dur
    else:
        nd = max(0.0, _f(d, default_dur))
        if nd != d:
            r.fix(f"anim: {path}.dur {d!r} -> {nd}")
        a["dur"] = nd
    return a


def _repair_emphasis(r: _R, emph, path: str):
    if emph is None:
        return []
    if isinstance(emph, dict):
        emph = [emph]
        r.fix(f"emphasis: {path} object -> list")
    if not isinstance(emph, list):
        r.fix(f"emphasis: {path} invalid -> []")
        return []
    out = []
    for i, e in enumerate(emph):
        if isinstance(e, str):
            e = {"verb": e}
        if not isinstance(e, dict):
            r.fix(f"emphasis: {path}[{i}] dropped")
            continue
        e = dict(e)
        v = e.get("verb")
        if v not in VERBS["emphasis"]:
            vv = _SYN["emphasis"].get(str(v or "").strip().lower())
            if vv is None:
                r.fix(f"emphasis: {path}[{i}] unknown verb {v!r} dropped")
                continue
            r.fix(f"enum: {path}[{i}].verb {v!r} -> {vv!r}")
            e["verb"] = vv
        on = e.get("on", "beats")
        if isinstance(on, list):
            on = [float(t) for t in on if _isnum(t)]
        elif on not in EMPHASIS_ON:
            on = r.enum(on, EMPHASIS_ON, "emphasis_on", "beats", f"{path}[{i}].on")
        e["on"] = on
        if "amount" in e:
            e["amount"] = _f(e["amount"], 0.08)
        if e.get("color") is not None:
            e["color"] = r.color(e["color"], f"{path}[{i}].color", "palette:2")
        out.append(e)
    return out


def _known_image(r: _R, src) -> bool:
    return (not r.check_refs) or src in r.images


def _repair_background(r: _R, bg, path: str, i: int):
    if isinstance(bg, str):
        bg = {"type": "color", "color": bg}
        r.fix(f"background: {path} string -> color")
    if not isinstance(bg, dict):
        r.fix(f"background: {path} missing -> gradient")
        bg = {"type": "gradient", "colors": ["palette:0", f"palette:{min(2, r.n_pal - 1)}"], "angle": 120}
    bg = dict(bg)
    bg["type"] = r.enum(bg.get("type"), BACKGROUND_TYPES, "background", "gradient", f"{path}.type")
    if bg["type"] == "image" and not _known_image(r, bg.get("src")):
        r.fix(f"ref: {path}.src {bg.get('src')!r} has no image -> gradient")
        bg["type"] = "gradient"
        bg.pop("src", None)
        bg.pop("focus", None)
        tr = bg.get("treatment")
        if isinstance(tr, dict) and "type" not in tr:
            bg.pop("treatment", None)
    t = bg["type"]
    if t == "image":
        f = bg.get("focus")
        if f is not None and not (isinstance(f, (list, tuple)) and len(f) == 2 and all(_isnum(v) for v in f)):
            r.fix(f"background: {path}.focus invalid -> centre")
            bg["focus"] = [0.5, 0.5]
    elif t == "color":
        bg["color"] = r.color(bg.get("color"), f"{path}.color", "palette:0")
    elif t == "gradient":
        cols = bg.get("colors")
        if not isinstance(cols, list) or len(cols) < 2:
            r.fix(f"background: {path}.colors invalid -> palette 0/2")
            cols = ["palette:0", f"palette:{min(2, r.n_pal - 1)}"]
        bg["colors"] = [r.color(c, f"{path}.colors[{k}]", "palette:0") for k, c in enumerate(cols)]
        if "angle" in bg:
            bg["angle"] = _f(bg["angle"], 90.0)
    elif t == "shapes":
        if bg.get("color") is not None:
            bg["color"] = r.color(bg.get("color"), f"{path}.color", "palette:0", allow_none=True)
        if bg.get("colors") is not None:
            cols = bg["colors"] if isinstance(bg["colors"], list) else []
            bg["colors"] = [c for c in (r.color(c, f"{path}.colors[{k}]", None, True) for k, c in enumerate(cols)) if c]
    m = bg.get("motion")
    if isinstance(m, str):
        m = {"verb": m}
        r.fix(f"background: {path}.motion string -> object")
    if m is not None:
        if not isinstance(m, dict):
            r.fix(f"background: {path}.motion invalid -> drift")
            m = {"verb": "drift"}
        m = dict(m)
        m["verb"] = r.enum(m.get("verb"), VERBS["background_motion"], "background_motion", "drift",
                           f"{path}.motion.verb")
        bg["motion"] = m
    tr = bg.get("treatment")
    if isinstance(tr, list) or (isinstance(tr, dict) and ("type" in tr or "name" in tr)) or isinstance(tr, str):
        bg["treatment"] = _repair_treatment(r, tr, f"{path}.treatment")
    elif isinstance(tr, dict):
        duo = tr.get("duotone")
        if duo is not None:
            if isinstance(duo, list) and len(duo) == 2:
                tr["duotone"] = [r.color(duo[0], f"{path}.treatment.duotone[0]", "palette:0"),
                                 r.color(duo[1], f"{path}.treatment.duotone[1]", "palette:1")]
            else:
                r.fix(f"background: {path}.treatment.duotone invalid -> null")
                tr["duotone"] = None
        for k in ("blur", "darken"):
            if k in tr:
                tr[k] = _clamp(_f(tr[k], 0.0), 0.0, 1.0)
    elif tr is not None:
        r.fix(f"background: {path}.treatment invalid -> dropped")
        bg.pop("treatment", None)
    return bg


def _repair_layer(r: _R, L, path: str, lo: float, hi: float):
    if not isinstance(L, dict):
        r.fix(f"layer: {path} not an object, dropped")
        return None
    L = dict(L)
    typ = L.get("type")
    if typ not in LAYER_TYPES:
        if "text" in L:
            typ = "text"
        elif "primitive" in L or "source" in L:
            typ = "shape"
        elif "src" in L:
            typ = "image"
        else:
            r.fix(f"layer: {path} unknown type {typ!r}, dropped")
            return None
        r.fix(f"enum: {path}.type {L.get('type')!r} -> {typ!r}")
        L["type"] = typ
    t0 = _f(L.get("t0"), lo)
    t1 = _f(L.get("t1"), hi)
    n0, n1 = _clamp(t0, lo, hi), _clamp(t1, lo, hi)
    if n1 - n0 < 0.05:
        r.fix(f"time: {path} [{L.get('t0')}, {L.get('t1')}] outside scene, dropped")
        return None
    if (n0, n1) != (L.get("t0"), L.get("t1")):
        r.fix(f"time: {path} [{L.get('t0')}, {L.get('t1')}] -> [{n0:.3f}, {n1:.3f}]")
    L["t0"], L["t1"] = round(n0, 4), round(n1, 4)
    L["emphasis"] = _repair_emphasis(r, L.get("emphasis"), f"{path}.emphasis")
    if typ == "text":
        txt = L.get("text")
        if isinstance(txt, list):
            txt = " ".join(str(x) for x in txt)
        if isinstance(txt, dict):          # per-format text {"9:16": ..., "16:9": ...}
            per = {}
            for k, v in txt.items():
                fk = str(k).strip().replace("x", ":").replace("/", ":").replace(" ", "")
                if fk in FORMATS and isinstance(v, (str, list)):
                    v = " ".join(str(x) for x in v) if isinstance(v, list) else v
                    if v.strip():
                        per[fk] = v
            if per and set(per) != set(FORMATS):
                other = next(iter(per.values()))
                for fmt in FORMATS:
                    if fmt not in per:
                        per[fmt] = other
                        r.fix(f"text: {path}.text[{fmt}] missing -> copied")
            if per and set(per) != set(txt):
                r.fix(f"text: {path}.text per-format dict cleaned")
            txt = {fmt: per[fmt] for fmt in FORMATS} if per else None
        if not isinstance(txt, dict) and (not isinstance(txt, str) or not txt.strip()):
            r.fix(f"layer: {path} empty text, dropped")
            return None
        L["text"] = txt
        L["role"] = r.enum(L.get("role", "headline"), TEXT_ROLES, "role", "headline", f"{path}.role")
        f = L.get("font", "display")
        if f not in FONT_SLOTS and f not in BUNDLED_FONTS:
            g = next((b for b in BUNDLED_FONTS if isinstance(f, str) and b.lower() == f.strip().lower()), None)
            nf = g or ("body" if L["role"] in ("caption", "label", "sub") else "display")
            r.fix(f"enum: {path}.font {f!r} -> {nf!r}")
            L["font"] = nf
        L["case"] = r.enum(L.get("case", "as_is"), TEXT_CASES, "case", "as_is", f"{path}.case")
        L["align"] = r.enum(L.get("align", "center"), ALIGNS, "align", "center", f"{path}.align")
        L["color"] = r.color(L.get("color"), f"{path}.color", "palette:1")
        L["highlight"] = r.color(L.get("highlight"), f"{path}.highlight", None, allow_none=True)
        r.fmt_boxes(L, path, (0.08, 0.42, 0.84, 0.16))
        L["in"] = _repair_anim(r, L.get("in"), "text_in", f"{path}.in", "fade", 0.4)
        L["out"] = _repair_anim(r, L.get("out"), "text_out", f"{path}.out", "fade", 0.3)
    elif typ == "shape":
        src = L.get("source")
        if src not in SHAPE_SOURCES:
            nsrc = "primitive"
            r.fix(f"enum: {path}.source {src!r} -> 'primitive'")
            if isinstance(src, str) and src.strip().lower() in VERBS["primitives"] and "primitive" not in L:
                L["primitive"] = src.strip().lower()
            L["source"] = src = nsrc
        if src in ("contour", "line"):
            known = r.contours if src == "contour" else r.lines
            if r.check_refs and L.get("ref") not in known:
                r.fix(f"ref: {path}.ref {L.get('ref')!r} not in analysis -> primitive ring")
                L["source"] = src = "primitive"
                L["primitive"] = "ring"
                L.pop("ref", None)
        if src == "primitive":
            L["primitive"] = r.enum(L.get("primitive"), VERBS["primitives"], "primitives", "circle",
                                    f"{path}.primitive")
            r.fmt_boxes(L, path, (0.3, 0.35, 0.4, 0.3))
        if L.get("stroke") is not None:
            L["stroke"] = r.color(L.get("stroke"), f"{path}.stroke", "palette:2", allow_none=True)
        if L.get("fill") is not None:
            L["fill"] = r.color(L.get("fill"), f"{path}.fill", "palette:2", allow_none=True)
        if L.get("stroke") is None and L.get("fill") is None:
            r.fix(f"color: {path} shape without stroke/fill -> stroke palette:2")
            L["stroke"] = f"palette:{min(2, r.n_pal - 1)}"
        if "stroke_width" in L and L["stroke_width"] is not None:
            sw = _f(L["stroke_width"], 0.006)
            if sw > 0.2:
                sw = sw / 1080.0 if sw > 1 else 0.2
            nsw = _clamp(sw, 0.0, 0.2)
            if nsw != L["stroke_width"]:
                r.fix(f"shape: {path}.stroke_width {L['stroke_width']!r} -> {nsw}")
            L["stroke_width"] = nsw
        if "fill_opacity" in L and L["fill_opacity"] is not None:
            L["fill_opacity"] = _clamp(_f(L["fill_opacity"], 1.0), 0.0, 1.0)
        if "scale" in L and L["scale"] is not None:
            L["scale"] = max(0.0, _f(L["scale"], 1.0))
        off = L.get("offset")
        if off is not None and not (isinstance(off, (list, tuple)) and len(off) == 2 and all(_isnum(v) for v in off)):
            r.fix(f"shape: {path}.offset invalid -> [0, 0]")
            L["offset"] = [0, 0]
        L["in"] = _repair_anim(r, L.get("in"), "shape_in", f"{path}.in", "fade", 0.4)
        L["out"] = _repair_anim(r, L.get("out"), "shape_out", f"{path}.out", "fade", 0.3)
        echo = L.get("echo")
        if echo is not None:
            if not isinstance(echo, dict):
                r.fix(f"shape: {path}.echo invalid -> null")
                L["echo"] = None
            else:
                echo = dict(echo)
                echo["count"] = int(_clamp(_f(echo.get("count"), 2), 0, 8))
                if "spacing" in echo:
                    echo["spacing"] = _clamp(_f(echo["spacing"], 0.015), 0.0, 0.2)
                if "decay" in echo:
                    echo["decay"] = _clamp(_f(echo["decay"], 0.5), 0.0, 1.0)
                L["echo"] = echo
    elif typ == "image":
        if not _known_image(r, L.get("src")):
            r.fix(f"ref: {path}.src {L.get('src')!r} has no image, layer dropped")
            return None
        mask = L.get("mask", "rect")
        if isinstance(mask, str) and mask.startswith("contour:"):
            if r.check_refs and mask[len("contour:"):] not in r.contours:
                r.fix(f"ref: {path}.mask {mask!r} not in analysis -> rounded")
                L["mask"] = "rounded"
        else:
            L["mask"] = r.enum(mask, VERBS["masks"], "masks", "rect", f"{path}.mask")
        r.fmt_boxes(L, path, (0.15, 0.25, 0.7, 0.4))
        L["in"] = _repair_anim(r, L.get("in"), "image_in", f"{path}.in", "fade", 0.4)
        L["out"] = _repair_anim(r, L.get("out"), "image_out", f"{path}.out", "fade", 0.3)
        b = L.get("border")
        if b is not None:
            if isinstance(b, dict):
                b = dict(b)
                b["color"] = r.color(b.get("color"), f"{path}.border.color", "palette:1")
                if "width" in b:
                    b["width"] = _clamp(_f(b["width"], 0.01), 0.0, 0.2)
                L["border"] = b
            else:
                r.fix(f"image: {path}.border invalid -> null")
                L["border"] = None
        if L.get("treatment") is not None:
            L["treatment"] = _repair_treatment(r, L.get("treatment"), f"{path}.treatment")
    return L


def _repair_scenes(r: _R, plan: dict, duration: float):
    scenes = plan.get("scenes")
    if isinstance(scenes, dict):
        scenes = list(scenes.values())
        r.fix("scenes: object -> list")
    if not isinstance(scenes, list):
        scenes = []
    scenes = [s for s in scenes if isinstance(s, dict)]
    if not scenes:
        r.fix("scenes: none -> one gradient scene")
        scenes = [{"id": "s0", "t0": 0.0, "t1": duration, "layers": []}]
    # timing: keep the model's relative scene lengths, but make them contiguous and cover [0, duration]
    lens = []
    for s in scenes:
        t0, t1 = _f(s.get("t0"), float("nan")), _f(s.get("t1"), float("nan"))
        lens.append(t1 - t0 if math.isfinite(t0) and math.isfinite(t1) and t1 > t0 else float("nan"))
    order = sorted(range(len(scenes)), key=lambda k: _f(scenes[k].get("t0"), k))
    if order != list(range(len(scenes))):
        r.fix("scenes: re-sorted by t0")
        scenes = [scenes[k] for k in order]
        lens = [lens[k] for k in order]
    known = [x for x in lens if math.isfinite(x)]
    fill = (sum(known) / len(known)) if known else duration / len(scenes)
    lens = [x if math.isfinite(x) else fill for x in lens]
    old = [(s.get("t0"), s.get("t1")) for s in scenes]
    bounds = [0.0]
    for x in lens:
        bounds.append(bounds[-1] + x)
    total = bounds[-1]
    if abs(total - duration) > 1e-6:
        bounds = [b * duration / total for b in bounds]
    bounds = [round(b, 4) for b in bounds]
    bounds[-1] = duration
    # shifts of each scene's start, used to move its layers with it
    shifts, scales = [], []
    for k, s in enumerate(scenes):
        o0, o1 = _f(old[k][0], bounds[k]), _f(old[k][1], bounds[k + 1])
        span = (o1 - o0) if o1 > o0 else (bounds[k + 1] - bounds[k])
        shifts.append((o0, span))
        s["t0"], s["t1"] = bounds[k], bounds[k + 1]
    if any(_f(o[0], -1) != b0 or _f(o[1], -1) != b1 for o, b0, b1 in zip(old, bounds, bounds[1:])):
        r.fix("scenes: timing made contiguous / fitted to duration")
    ids = set()
    out = []
    n = len(scenes)
    for i, s in enumerate(scenes):
        p = f"scenes[{i}]"
        sid = s.get("id")
        if not isinstance(sid, str) or sid in ids:
            s["id"] = sid = f"s{i}" if f"s{i}" not in ids else f"s{i}_{len(ids)}"
            r.fix(f"scenes: {p}.id missing/duplicate -> {sid!r}")
        ids.add(sid)
        s["background"] = _repair_background(r, s.get("background"), f"{p}.background", i)
        t0, t1 = s["t0"], s["t1"]
        last = i == n - 1
        tr = s.get("transition_out")
        if isinstance(tr, str):
            tr = {"type": tr}
            r.fix(f"transition: {p} string -> object")
        if last:
            tr = {"type": "cut", "dur": 0.0} if not isinstance(tr, dict) else tr
        if not isinstance(tr, dict):
            r.fix(f"transition: {p}.transition_out missing -> crossfade")
            tr = {"type": "crossfade", "dur": 0.4}
        tr = dict(tr)
        tr["type"] = r.enum(tr.get("type"), TRANSITIONS, "transition", "crossfade", f"{p}.transition_out.type")
        d = max(0.0, _f(tr.get("dur"), 0.0 if tr["type"] == "cut" else 0.4))
        if tr["type"] in DIRECTIONAL_TRANSITIONS:
            tr["dir"] = r.enum(tr.get("dir", "left"), DIRECTIONS, "direction", "left", f"{p}.transition_out.dir")
        if not last:
            n0, n1 = scenes[i + 1]["t0"], scenes[i + 1]["t1"]
            cap = 2.0 * min(t1 - t0, n1 - n0)
            if d > cap:
                r.fix(f"transition: {p}.dur {d} too long -> {cap * 0.9:.3f}")
                d = round(cap * 0.9, 4)
        tr["dur"] = d
        s["transition_out"] = tr
        tr_dur = 0.0 if last else d
        lo, hi = t0, min(t1 + tr_dur / 2.0, duration)
        o0, span = shifts[i]
        layers = s.get("layers") or []
        if not isinstance(layers, list):
            r.fix(f"layers: {p}.layers not a list -> []")
            layers = []
        new_layers = []
        for j, L in enumerate(layers):
            if isinstance(L, dict) and (abs(o0 - t0) > 1e-6 or abs(span - (t1 - t0)) > 1e-6):
                # carry layer times with the moved / stretched scene
                k = (t1 - t0) / span if span > 0 else 1.0
                for key in ("t0", "t1"):
                    if _isnum(L.get(key)):
                        L[key] = round(t0 + (float(L[key]) - o0) * k, 4)
            NL = _repair_layer(r, L, f"{p}.layers[{j}]", lo, hi)
            if NL is not None:
                new_layers.append(NL)
        s["layers"] = new_layers
        out.append(s)
    plan["scenes"] = out


def repair_plan(plan: dict, analysis: dict | None = None, log: list | None = None,
                duration_range: tuple[float, float] = (5.0, 15.0)) -> dict:
    """Return a repaired deep copy of ``plan``; fixes are appended to ``log``."""
    log = log if log is not None else []
    plan = copy.deepcopy(plan)
    r = _R(plan, analysis, log)
    if plan.get("version") != PLAN_VERSION:
        r.fix(f"top: version {plan.get('version')!r} -> {PLAN_VERSION}")
        plan["version"] = PLAN_VERSION
    seed = plan.get("seed")
    if not _isnum(seed):
        r.fix(f"top: seed {seed!r} -> 0")
        plan["seed"] = 0
    else:
        plan["seed"] = int(seed)
    dur = plan.get("duration")
    if not _isnum(dur):
        last = None
        sc = plan.get("scenes")
        if isinstance(sc, list) and sc and isinstance(sc[-1], dict):
            last = _f(sc[-1].get("t1"), float("nan"))
        dur = last if last is not None and math.isfinite(last) else 10.0
        r.fix(f"top: duration missing -> {dur}")
    nd = _clamp(float(dur), *duration_range)
    if nd != dur:
        r.fix(f"top: duration {dur} -> {nd}")
    plan["duration"] = nd
    fps = plan.get("fps")
    if not _isnum(fps) or not (1 <= fps <= 120):
        r.fix(f"top: fps {fps!r} -> 30")
        plan["fps"] = 30
    if plan.get("voice") is not None and not isinstance(plan.get("voice"), dict):
        r.fix("top: voice -> null")
        plan["voice"] = None
    plan.setdefault("voice", None)
    _repair_style(r, plan)
    has_audio = bool((analysis or {}).get("audio"))
    _repair_audio(r, plan, nd, has_audio)
    _repair_scenes(r, plan, nd)
    return plan


def extract_plan(text: str, analysis: dict | None = None, log: list | None = None) -> dict:
    """Raw LLM text -> valid plan dict. Raises PlanExtractError if it cannot be made valid.

    ``analysis`` (optional) lets the repair drop references to images / contours that do not
    exist; pass the same analysis the renderer will get. ``log`` collects every fix applied
    (an empty log means the model's plan was valid as written).
    """
    log = log if log is not None else []
    raw = loads_tolerant(text, log)
    if not validate_plan(raw, analysis):
        return raw
    plan = repair_plan(raw, analysis, log)
    errs = validate_plan(plan, analysis)
    if errs:
        raise PlanExtractError(f"plan still invalid after repair ({len(errs)} problems)", errs, plan)
    return plan
