"""Kinetic type. Each word is pre-rendered once (2x supersampled, PIL FreeType AA) into
an alpha sprite at its laid-out position; per frame the sprites are transformed and
composited, so animation costs one small warp per word."""
from __future__ import annotations

import math
import os
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..font_registry import AXIS_NAMES, REGISTRY, font_axes, font_key, font_path, split_key
from ..geometry import cover_crop
from .background import warp_crop
from .common import (Anim, Ctx, Timing, affine, apply_emphasis, apply_in, apply_out, composite,
                     glitch_jitter, translate)
from .easing import back_out, clamp01, cubic_in_out, decay_env, ease_in, ease_out
from .shapes import draw_ops, get_box

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

# plan font name -> bundled TTF (absolute path); the registry is mograph/fonts/fonts.json
FONTS = {name: font_path(name) for name in REGISTRY}
# heavier cut used for headline / label roles when the family ships one (variable faces use bold_axes)
BOLD_VARIANTS = {name: font_path(name, True) for name, e in REGISTRY.items() if e.get("bold_file")}
# line pitch as a multiple of font size
LEADING = {name: float(e.get("leading", 1.12)) for name, e in REGISTRY.items()}

SS = 2  # supersample factor for glyph rendering


def _set_axes(f: ImageFont.FreeTypeFont, axes: dict) -> None:
    """Apply OpenType-tag axis values ({'wght': 800}) to a variable font; static fonts are left alone."""
    try:
        info = f.get_variation_axes()
    except (OSError, AttributeError):
        return
    want = {AXIS_NAMES.get(k.lower(), k.lower()): float(v) for k, v in axes.items()}
    vals = []
    for a in info:
        nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
        v = want.get(nm.lower(), a["default"])
        vals.append(min(max(v, a["minimum"]), a["maximum"]))
    f.set_variation_by_axes(vals)


@lru_cache(maxsize=512)
def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """``path`` is a font key: a TTF path, optionally "|tag=value,..." variation values (font_key)."""
    file, axes = split_key(path)
    from mograph import fonts_liked as _fl
    liked_file = _fl.assert_liked_path(file)                     # HARD rule (user 2026-10-30): a face that is not in liked.json is never drawn (non-strict: the liked face of the same role; strict: raises)
    if liked_file != file:
        file, axes = liked_file, None
    f = ImageFont.truetype(file, max(1, int(size)), layout_engine=ImageFont.Layout.BASIC)
    if axes:
        _set_axes(f, axes)
    return f


def resolve_font(plan: dict, layer: dict) -> tuple[str, str]:
    """-> (family name, font key: ttf path + variation axes)."""
    fonts = ((plan or {}).get("style") or {}).get("fonts") or {}
    key = layer.get("font", "display")
    name = fonts.get(key, key)
    if name not in FONTS:
        name = fonts.get("display") if fonts.get("display") in FONTS else "Space Grotesk"
    path = FONTS[name]
    bold = layer.get("role", "headline") in ("headline", "label")
    if bold and name in BOLD_VARIANTS:
        path = BOLD_VARIANTS[name]
    axes = layer.get("axes") if isinstance(layer.get("axes"), dict) else None
    return name, font_key(path, font_axes(name, bold, axes))


def apply_case(text: str, case: str) -> str:
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    return text


def _layout(paras, path, size, bw, leading, wrap=True):
    """wrap=False: every paragraph is exactly one line (auto-fit has already chosen the breaks)."""
    f = get_font(path, size * SS)
    space = f.getlength(" ") / SS
    lines = []
    for words in paras:
        cur, cw = [], 0.0
        for w in words:
            wl = f.getlength(w) / SS
            if wrap and cur and cw + space + wl > bw:
                lines.append((cur, cw))
                cur, cw = [w], wl
            else:
                cw = wl if not cur else cw + space + wl
                cur.append(w)
        if cur:
            lines.append((cur, cw))
    if not lines:
        return None
    asc, desc = 0.0, 0.0
    for words, _ in lines:
        bb = f.getbbox(" ".join(words), anchor="ls")
        asc = max(asc, -bb[1] / SS)
        desc = max(desc, bb[3] / SS)
    pitch = size * leading
    height = asc + (len(lines) - 1) * pitch + desc
    width = max(lw for _, lw in lines)
    return dict(size=size, lines=lines, asc=asc, desc=desc, pitch=pitch, height=height,
                width=width, space=space)


def fit_text(text: str, path: str, bw: float, bh: float, leading: float, wrap: bool = True) -> dict | None:
    """Largest integer font size whose wrapped text fits the box (binary search). wrap=False keeps every
    "\n"-separated paragraph on one line."""
    paras = [p.split() for p in text.split("\n")]
    paras = [p for p in paras if p]
    if not paras:
        return None
    lo, hi, best = 4, max(6, int(bh * 1.3)), None
    while lo <= hi:
        mid = (lo + hi) // 2
        lay = _layout(paras, path, mid, bw, leading, wrap)
        if lay is not None and lay["width"] <= bw + 0.5 and lay["height"] <= bh + 0.5:
            best, lo = lay, mid + 1
        else:
            hi = mid - 1
    return best or _layout(paras, path, 4, bw, leading, wrap)


# ----------------------------------------------------------------------------- auto-fit (display text)
# Per-format text: "text" may be {"9:16": "NEON\nHEART", "16:9": "NEON HEART"} (like "box"); layer_text picks
#   the frame's format.
# "fit": auto | none. Auto (the default for headline-role or display-font layers): when the box is tall
#   relative to its width (pixel h >= AUTO_FIT_ASPECT * w) the text is re-broken into balanced lines -- 1..4
#   lines at word boundaries, short words ("the", "and", "of", ...) kept with the word after them -- using the
#   line count that gives the largest font size inside the box. Text containing an explicit "\n" keeps its
#   own breaks; long text that only fits clearly bigger in more than 4 lines keeps the plain greedy wrap.
AUTO_FIT_ASPECT = 0.5
AUTO_FIT_MAX_LINES = 4
SHORT_WORDS = frozenset("a an the and & or but nor of to in on at by for with from as into onto per via vs "
                        "de la le el y et".split())
_AUTO_GAIN = 1.02          # one more line must be at least this much bigger to be chosen
_AUTO_MAX_SPLITS = 6000    # candidate breaks examined per line count


def layer_text(layer: dict, fmt: str) -> str:
    """The layer's text for this format; "text" is a string or a per-format dict like "box"."""
    t = layer.get("text", "")
    if isinstance(t, dict):
        t = t.get(fmt) or next((v for v in t.values() if isinstance(v, str) and v.strip()), "")
    return str(t or "")


def auto_fit_on(layer: dict, text: str, bw: float, bh: float) -> bool:
    fit = layer.get("fit")
    if fit == "none" or "\n" in text or not text.strip():
        return False
    if fit != "auto" and layer.get("role", "headline") != "headline" and layer.get("font", "display") != "display":
        return False
    return bh >= AUTO_FIT_ASPECT * bw


def _bind_short(words: list[str]) -> list[str]:
    """Unbreakable units: a short word is glued to the word after it ("the sea.", "and the sea.")."""
    units, cur = [], []
    for w in words:
        cur.append(w)
        if w.strip(".,;:!?'\"()").lower() not in SHORT_WORDS:
            units.append(" ".join(cur))
            cur = []
    if cur:
        if units:
            units[-1] = units[-1] + " " + " ".join(cur)
        else:
            units.append(" ".join(cur))
    return units


def balanced_lines(text: str, path: str, bw: float, bh: float, leading: float,
                   max_lines: int = AUTO_FIT_MAX_LINES) -> str | None:
    """Re-break ``text`` into the balanced 1..max_lines lines that fit the box biggest -> "line\nline", or None
    when the text only fits clearly bigger in more than max_lines greedy-wrapped lines."""
    units = _bind_short(text.split())
    k = len(units)
    if k == 0:
        return None
    ref = 100
    f = get_font(path, ref * SS)
    uw = [f.getlength(u) / SS / ref for u in units]
    sp = f.getlength(" ") / SS / ref
    bb = f.getbbox(" ".join(units), anchor="ls")
    asc, desc = max(0.0, -bb[1] / SS / ref), max(0.0, bb[3] / SS / ref)
    pre = np.concatenate([[0.0], np.cumsum(uw)])

    def line_w(i, j):
        return pre[j] - pre[i] + (j - i - 1) * sp

    best_size, best_cut = None, None
    for n in range(1, min(max_lines, k) + 1):
        if math.comb(k - 1, n - 1) > _AUTO_MAX_SPLITS:
            break
        cand = None
        for cuts in combinations(range(1, k), n - 1):
            edges = (0,) + cuts + (k,)
            ws = [line_w(a, b) for a, b in zip(edges, edges[1:])]
            key = (max(ws), sum(w * w for w in ws))
            if cand is None or key < cand[0]:
                cand = (key, edges)
        size = min(bw / max(cand[0][0], 1e-6), bh / max(asc + (n - 1) * leading + desc, 1e-6))
        if best_size is None or size > best_size * _AUTO_GAIN:
            best_size, best_cut = size, cand[1]
    out = "\n".join(" ".join(units[a:b]) for a, b in zip(best_cut, best_cut[1:]))
    greedy = fit_text(text, path, bw, bh, leading)
    mine = fit_text(out, path, bw, bh, leading, wrap=False)
    if (greedy is not None and mine is not None and len(greedy["lines"]) > max_lines
            and mine["size"] < 0.9 * greedy["size"]):
        return None
    return out


def _render_word(word, path, size, x, base, stroke=0.0):
    """Alpha sprite of ``word`` with the pen at (x, base). stroke > 0 (px): outline only (hollow letters)."""
    f = get_font(path, size * SS)
    sw = int(round(stroke * SS))
    bb = f.getbbox(word, anchor="ls", stroke_width=sw) if sw else f.getbbox(word, anchor="ls")
    left, top = x + bb[0] / SS, base + bb[1] / SS
    right, bottom = x + bb[2] / SS, base + bb[3] / SS
    ox, oy = int(math.floor(left)) - 1, int(math.floor(top)) - 1
    w, h = int(math.ceil(right)) + 1 - ox, int(math.ceil(bottom)) + 1 - oy
    img = Image.new("L", (w * SS, h * SS), 0)
    if sw:
        ImageDraw.Draw(img).text(((x - ox) * SS, (base - oy) * SS), word, font=f, fill=0, anchor="ls",
                                 stroke_width=sw, stroke_fill=255)
    else:
        ImageDraw.Draw(img).text(((x - ox) * SS, (base - oy) * SS), word, font=f, fill=255, anchor="ls")
    a = np.asarray(img, np.float32) * (1.0 / 255.0)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    # char boundaries (relative to x) for type_on
    cx = [0.0] + [f.getlength(word[:k + 1]) / SS for k in range(len(word))]
    return a, ox, oy, cx


def _compose(A, B):
    """2x3 affine A after B."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def _isect(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


# ----------------------------------------------------------------------------- letter-level verbs
# "scramble" (in): every letter cycles through random glyphs and resolves to the real one; the last
#   letter resolves exactly at the end of the in-phase (land it on a hit). Params: "rate" glyph changes
#   per second (18), "color" of unresolved glyphs (layer colour).
# "row_slide" (in / out): the text repeated as "rows" stacked copies (3) sliding sideways at different
#   speeds and alternating directions, easing into alignment at the end of the in-phase (out: apart
#   again). Params: "scale" size over the box fit (1; > 1 overflows the box and is cropped by the frame),
#   "travel" start offset in frame widths (0.6), "track" extra letter spacing in em at the start of the
#   in / end of the out (0), "drift" constant slide in frame widths per second (0; keeps holds alive),
#   "fade" fade with the phase (false), "stagger" 0..1 letter-by-letter spread of the in-fade (0),
#   "gap" extra row spacing in line pitches (0), "alt_color" for the off-centre rows (layer colour).
LETTER_VERBS = ("scramble", "row_slide")
_SCRAMBLE_CHARS = {"upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*+=/<>?",
                   "lower": "abcdefghijklmnopqrstuvwxyz0123456789#%&*+=/<>?"}
_ROW_SPEED = (1.0, 0.55, 1.45, 0.8, 1.2, 0.65, 1.35)


@lru_cache(maxsize=4096)
def _glyph(path: str, size: int, ch: str, stroke: float = 0.0):
    """One character's alpha sprite with the pen at (0, 0) on the baseline -> (alpha, ox, oy)."""
    a, ox, oy, _ = _render_word(ch, path, size, 0.0, 0.0, stroke)
    return a, ox, oy


# ----------------------------------------------------------------------------- bold-type options
# "axes" {tag: value}: static variation for this layer (e.g. {"wght": 200} for a hairline cut of a variable
#   face), on top of the registry defaults.
# "vary": per-letter wave drawn letter by letter, re-flowing each line around its anchor. "axis" (tag; default
#   the face's first registry "vary" axis), "range" [lo, hi] (registry range), "period" s or "period_beats" (2),
#   "phase" cycle offset per letter (0.08), "scale" per-glyph size amplitude (0; 0.3 if the face has no such axis),
#   "on" (beats | downbeats | onsets): instead of the free-running wave, each event kicks the letters to "hi"
#   in a ripple ("phase" s per letter) that decays back to "lo". "fit": both (size so neither end overflows
#   the box) | lo | hi (fit that end; the other may run past the box / frame edge and is cropped).
# "outline": hollow letters. "width" stroke in em (0.035), "swap" (beats | downbeats | onsets | [times]) toggles
#   outline <-> fill on each event after t0, "start" outline | fill, "rows" all | alt (row_slide: only the
#   off-centre rows are hollow).
# "window": type as a mask. mode "fill": the letters show image "src" (cover-cropped to the frame, "focus",
#   "zoom" z or [z0, z1] over the layer's life). mode "knockout": a "plate" colour ("plate_opacity" 1) covers
#   the frame (or the text block padded by "pad" em) with the letters cut out, showing what is underneath
#   (or "src" when given).
VARY_STEPS = 32
OUTLINE_EM = 0.02                 # 2026-11-01 user "they should be finer strokes": hollow / outlined letters default to a fine stroke (was 0.035 em)


def _pf(d: dict, k: str, default: float) -> float:
    v = d.get(k)
    return float(default if v is None else v)


# ----------------------------------------------------------------------------- shapes-and-sizes options
# "rotate": degrees clockwise, a number or keyframes [[t, deg], ...] (absolute s, eased between, held outside).
#   "in" / "out" take "spin": extra degrees at the start of the in-phase that settle to the rotation with the
#   in easing (bouncy overshoots), or that the out-phase turns to (needs a non-cut verb). The pivot is the text
#   block centre (the box centre for path text). A rotated layer is drawn into an off-frame canvas covering the
#   frame's footprint, matted to premultiplied RGBA (black / white difference) and rotated into the frame, so
#   every verb and option works and off-frame type rotates in uncut.
# "overscale": the fitted text is scaled by this factor about the box anchor (> 1 runs past the box and the frame
#   edge and is cropped). Text boxes may lie anywhere in -1..2 frame space (giant type cropped on purpose).
# "path": glyphs set along a curve inside the layer box, each turned to the tangent; all lines run as one line.
#   Units are box-relative (0..1 across the box). {"type": "arc" | "circle", "center" [u, v] (0.5, 0.5),
#   "radius" (1 = half the box's smaller side), "angle" deg of the text centre (0 = top, clockwise), "span" deg the
#   text is fitted into (arc 180, circle 360), "flip" (reads along the bottom, tops towards the centre)}.
#   {"type": "wave", "amp" (box-height fraction, 0.25), "cycles" across the box (1.5), "phase" (cycles), "speed"
#   (cycles/s), "on": beats | downbeats | onsets | [times] (phase steps "step" cycles (0.25) on each event)}.
#   {"type": "line", "p0", "p1" ([0, 1] -> [1, 0], a rising diagonal)}. All: "fill" fraction of the path length
#   the text spans (0.92), "size" em as a fraction of the frame short side (default: fitted to the path),
#   "travel" path lengths per second (text runs along the path; a circle wraps), "offset" baseline shift in em
#   towards the glyph tops (arc / circle 0 = baseline on the curve; wave / line -0.35 = caps centred on it).
# "backdrop": {"type": "dim" | "blur" | "band", "amount" 0..1, "pad" em, "soft" em, "color"} darkens / blurs the
#   picture behind the text block (soft-edged rect, rotated with the layer; band runs the full frame width).
#   Automatic ("dim", 0.45) for registry shape "fine" faces over an image background; false / "none" turns it off.
# "letter_scale": per-letter sizes. A list (cycled), "random", "wave" or {"type": "list" | "random" | "wave",
#   "values", "range": [lo, hi] (0.45, 1), "seed", "phase" per letter (0.12), "speed" cycles/s (0.6), "valign":
#   baseline | center}. Sizes are normalised so the largest letter is 1 and static sets are refitted to the box.
#   Letters re-flow per line; word_stagger / kinetic_split (drop and bounce) / type_on act letter by letter.
PATH_TYPES = ("arc", "circle", "wave", "line")
BACKDROP_TYPES = ("dim", "blur", "band")
AUTO_BACKDROP = {"type": "dim", "amount": 0.45}
_BD_AMOUNT = {"dim": 0.5, "blur": 0.6, "band": 0.7}
_PATH_REF = 100            # reference size for path advances


def bounce_out(p: float) -> float:
    p = clamp01(p)
    n, d = 7.5625, 2.75
    if p < 1.0 / d:
        return n * p * p
    if p < 2.0 / d:
        p -= 1.5 / d
        return n * p * p + 0.75
    if p < 2.5 / d:
        p -= 2.25 / d
        return n * p * p + 0.9375
    p -= 2.625 / d
    return n * p * p + 0.984375


def rotation_at(rot, t: float) -> float:
    """Static degrees, or keyframes [[t, deg], ...] eased (cubic in-out) between keys, held outside."""
    if rot is None:
        return 0.0
    if not isinstance(rot, (list, tuple)):
        return float(rot)
    keys = sorted((float(k[0]), float(k[1])) for k in rot if isinstance(k, (list, tuple)) and len(k) >= 2)
    if not keys:
        return 0.0
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, a), (tb, b) in zip(keys, keys[1:]):
        if t < tb:
            return a + (b - a) * cubic_in_out((t - ta) / max(tb - ta, 1e-6))
    return keys[-1][1]


def _soft(d: np.ndarray, soft: float) -> np.ndarray:
    """d = signed distance inside the edge (px) -> 0..1 smoothstep over ``soft`` px outside it."""
    m = np.clip(d / max(soft, 1e-3) + 1.0, 0.0, 1.0)
    return m * m * (3.0 - 2.0 * m)


@lru_cache(maxsize=2048)
def _glyph_shadow(path: str, size: int, ch: str):
    a, ox, oy = _glyph(path, size, ch)
    sig = max(1.2, size * 0.07)
    p = int(math.ceil(sig * 3))
    return cv2.GaussianBlur(np.pad(a, p), (0, 0), sig), ox - p, oy - p


class TextLayer:
    kind = "text"

    def __init__(self, layer: dict, ctx: Ctx, uid: int, on_image: bool = False):
        self.layer = layer
        self.ctx = ctx
        self.uid = uid
        self.tm = Timing(layer)
        self.words = []
        self.letters = []
        self._shift = (0, 0)               # canvas offset while drawing off-frame (rotation)
        self._masking = False
        self.tpath = None
        self.backdrop = None
        W, H, S = ctx.W, ctx.H, ctx.S
        text = apply_case(layer_text(layer, ctx.fmt), layer.get("case", "as_is"))
        name, path = resolve_font(ctx.plan, layer)
        bx, by, bw, bh = get_box(layer, ctx, (0.08, 0.7, 0.84, 0.18))
        bx, by, bw, bh = bx * W, by * H, bw * W, bh * H
        wrap = True                       # False once auto-fit has fixed the line breaks
        if auto_fit_on(layer, text, bw, bh):
            broken = balanced_lines(text, path, bw, bh, LEADING.get(name, 1.12))
            if broken is not None:
                text, wrap = broken, False
        self.color = ctx.color(layer.get("color", "palette:1"), (1, 1, 1))
        hl = layer.get("highlight")
        self.highlight = ctx.color(hl) if hl else None
        self.align = layer.get("align", "center")
        self._setup_vary(name, path)
        self._setup_lscale(layer, len("".join(text.split())))
        lay = fit_text(text, path, bw, bh, LEADING.get(name, 1.12), wrap) if text.strip() else None
        if lay is not None and self.vary is not None and self.vary_axis:
            # "fit": "both" sizes to the widest end of the range (never overflows); "lo" / "hi" fit that end only
            # (e.g. a beat pulse fitted at rest, so every kick pushes the letters past the frame edge)
            paras = [p for p in (q.split() for q in text.split("\n")) if p]
            fit = self.vary.get("fit", "both")
            ends = {"lo": self.vary_range[:1], "hi": self.vary_range[1:]}.get(fit, self.vary_range)
            if fit in ("lo", "hi"):
                # line breaks and anchors from the fitted end (the letters are drawn per frame anyway)
                lay = fit_text(text, self._vary_key(ends[0]), bw, bh, LEADING.get(name, 1.12), wrap) or lay
            for v in ends:
                alt = fit_text(text, self._vary_key(v), bw, bh, LEADING.get(name, 1.12), wrap)
                if alt is not None and alt["size"] < lay["size"]:
                    lay = _layout(paras, path, alt["size"], bw, LEADING.get(name, 1.12), wrap) or lay
        if lay is not None and self._ls_static is not None:
            lay = self._fit_lscale(lay, path, bw, bh, LEADING.get(name, 1.12))
        self.overscale = max(0.05, _pf(layer, "overscale", 1.0))
        if lay is not None and abs(self.overscale - 1.0) > 1e-6 and layer.get("path") is None:
            lay = _layout([ws for ws, _ in lay["lines"]], path, max(4, int(round(lay["size"] * self.overscale))),
                          bw, LEADING.get(name, 1.12), False) or lay
        self.lay = lay
        if lay is None:
            return
        size = lay["size"]
        self.size = size
        ol = layer.get("outline")
        self.ol = ol if isinstance(ol, dict) else None
        self.ol_px = max(1.0, _pf(self.ol, "width", OUTLINE_EM) * size) if self.ol else 0.0
        self._setup_window(layer)
        top = by + (bh - lay["height"]) / 2.0
        self.line_rects = []
        self.boxes = []
        pad_x, pad_y = 0.22 * size, 0.14 * size
        k = 0
        for li, (words, lw) in enumerate(lay["lines"]):
            base = top + lay["asc"] + li * lay["pitch"]
            if self.align == "left":
                x = bx
            elif self.align == "right":
                x = bx + bw - lw
            else:
                x = bx + (bw - lw) / 2.0
            f = get_font(path, size * SS)
            self.line_rects.append((x, base - lay["asc"] - pad_y, x + lw, base + lay["desc"] + pad_y))
            for w in words:
                a, ox, oy, cx = _render_word(w, path, size, x, base)
                self.words.append(dict(text=w, a=a, ox=ox, oy=oy, x=x, base=base, line=li, idx=k, cx=cx,
                                       w=f.getlength(w) / SS))
                if self.ol:
                    a_ol, ox_ol, oy_ol, _ = _render_word(w, path, size, x, base, self.ol_px)
                    self.words[-1].update(a_ol=a_ol, ox_ol=ox_ol, oy_ol=oy_ol)
                x += f.getlength(w) / SS + lay["space"]
                k += 1
        xs0 = [r[0] for r in self.line_rects]
        xs1 = [r[2] for r in self.line_rects]
        self.block = (min(xs0), self.line_rects[0][1], max(xs1), self.line_rects[-1][3])
        self.pivot = ((self.block[0] + self.block[2]) / 2.0, (self.block[1] + self.block[3]) / 2.0)
        if self.highlight is not None:
            for (x0, y0, x1, y1) in self.line_rects:
                X0, Y0 = int(round(x0 - pad_x)), int(round(y0))
                X1, Y1 = int(round(x1 + pad_x)), int(round(y1))
                self.boxes.append((np.ones((max(1, Y1 - Y0), max(1, X1 - X0)), np.float32), X0, Y0))
        # soft shadow for legibility over imagery
        self.shadows = []
        if on_image and self.highlight is None:
            sig = max(1.2, size * 0.07)
            p = int(math.ceil(sig * 3))
            for wd in self.words:
                sa = cv2.GaussianBlur(np.pad(wd["a"], p), (0, 0), sig)
                self.shadows.append((sa, wd["ox"] - p, wd["oy"] + size * 0.04 - p))
        self.n_chars = sum(len(w["text"]) for w in self.words)
        self.ul_color = self.color if self.highlight is not None else ctx.accent(0)
        if np.allclose(self.ul_color, self.color, atol=0.05):
            self.ul_color = ctx.palette[1] if len(ctx.palette) > 1 else self.ul_color
        self.U = {"slide": lay["pitch"] * 1.05, "slide_x": 0.25 * W, "blur": max(2.0, size * 0.18)}
        # stagger
        vin = self.tm.vin
        n = len(self.words)
        if "stagger_beats" in vin:
            stg = float(vin.get("stagger_beats") or 0.0) * 60.0 / max(ctx.bpm, 1.0)
        else:
            stg = float(vin.get("stagger", 0.06 if self.tm.in_verb == "kinetic_split" else 0.09))
        avail = (self.tm.t1 - self.tm.dout) - self.tm.t0 - self.tm.din
        if n > 1 and (n - 1) * stg > max(0.0, avail):
            stg = max(0.0, avail) / (n - 1)
        self.stagger = stg
        self.in_total = self.tm.din + (n - 1) * stg if self.tm.in_verb in ("word_stagger", "kinetic_split") else self.tm.din
        p = layer.get("path")
        if (self.tm.in_verb in LETTER_VERBS or self.tm.out_verb in LETTER_VERBS or self.vary is not None
                or self.lscale is not None or isinstance(p, dict)):
            self._build_letters(path, text, bx + bw / 2.0, by + bh / 2.0)
        n_l = max(1, len(self.letters))
        if "stagger_beats" in vin:
            lst = float(vin.get("stagger_beats") or 0.0) * 60.0 / max(ctx.bpm, 1.0)
        else:
            lst = float(vin.get("stagger", 0.05))
        self.lstagger = max(0.0, min(lst, max(0.0, avail) / max(1, n_l - 1)))
        self._shadow_on = on_image and self.highlight is None
        if isinstance(p, dict) and p.get("type") in PATH_TYPES and self.letters:
            self._setup_path(p, bx, by, bw, bh)
        self._setup_backdrop(layer, name, on_image)

    # ------------------------------------------------------------------ bold-type option setup
    def _setup_vary(self, name: str, path: str) -> None:
        v = self.layer.get("vary")
        self.vary = v if isinstance(v, dict) else None
        self.vary_axis, self.vary_range = None, (0.0, 0.0)
        if self.vary is None:
            return
        self._file, self._axes = split_key(path)
        reg = (REGISTRY.get(name) or {}).get("vary") or {}
        axis = self.vary.get("axis") or (next(iter(reg)) if reg else None)
        if axis is not None and axis in reg or (axis is not None and self.vary.get("range")):
            rng_ = self.vary.get("range") or reg.get(axis)
            self.vary_axis, self.vary_range = str(axis), (float(rng_[0]), float(rng_[1]))
        tm, bpm = self.tm, max(self.ctx.bpm, 1.0)
        per = self.vary.get("period")
        self.vary_period = float(per) if per else _pf(self.vary, "period_beats", 2.0) * 60.0 / bpm
        self.vary_scale = _pf(self.vary, "scale", 0.0 if self.vary_axis else 0.3)
        self.vary_phase = _pf(self.vary, "phase", 0.08)
        self.vary_ev = self.ctx.events(self.vary["on"]) if self.vary.get("on") else None
        if self.vary_ev is not None:
            self.vary_ev = self.vary_ev[self.vary_ev >= tm.t0 - 1e-6]

    def _vary_key(self, v: float) -> str:
        return font_key(self._file, dict(self._axes, **{self.vary_axis: float(v)}))

    def _setup_window(self, layer: dict) -> None:
        w = layer.get("window")
        self.window = w if isinstance(w, dict) else None
        self.win_src = None
        if self.window is None:
            return
        ctx = self.ctx
        z = self.window.get("zoom", [1.0, 1.12])
        self.win_zoom = (float(z), float(z)) if isinstance(z, (int, float)) else (float(z[0]), float(z[1]))
        im = ctx.images.get(self.window.get("src")) if self.window.get("src") is not None else None
        px = (im or {}).get("pixels")
        if px is None:
            return
        ih, iw = px.shape[:2]
        f = self.window.get("focus") or (im or {}).get("focus") or [0.5, 0.5]
        self.win_focus = (float(f[0]), float(f[1]))
        cw = cover_crop(iw, ih, ctx.W, ctx.H, self.win_focus, 1.0)[2] * iw
        s = ctx.W * max(max(self.win_zoom), 1.0) / max(cw, 1.0)
        src = px
        if s < 1.0:
            src = cv2.resize(px, (max(2, int(round(iw * s))), max(2, int(round(ih * s)))), interpolation=cv2.INTER_AREA)
        self.win_src = np.ascontiguousarray(src, np.float32)
        self.win_wh = (iw, ih)
        self._win_cache = (None, None)

    # ------------------------------------------------------------------ shapes-and-sizes setup
    def _setup_lscale(self, layer: dict, n: int) -> None:
        ls = layer.get("letter_scale")
        if isinstance(ls, (list, tuple)):
            ls = {"type": "list", "values": list(ls)}
        elif isinstance(ls, str):
            ls = {"type": ls}
        self.lscale = ls if isinstance(ls, dict) and n > 0 else None
        self._ls_static = None
        if self.lscale is None:
            return
        typ = ls.get("type") or ("list" if ls.get("values") else "random")
        rng_ = ls.get("range") or [0.45, 1.0]
        lo, hi = sorted((max(0.05, float(rng_[0])), max(0.05, float(rng_[1]))))
        self.ls_type, self.ls_lo, self.ls_hi = typ, lo, hi
        self.ls_center = ls.get("valign", "baseline") == "center"
        if typ == "list" and ls.get("values"):
            vals = [max(0.05, abs(float(v))) for v in ls["values"]]
            v = np.array([vals[j % len(vals)] for j in range(n)], np.float64)
        elif typ == "wave":
            self.ls_phase, self.ls_speed = _pf(ls, "phase", 0.12), _pf(ls, "speed", 0.6)
            return
        else:
            r = (np.random.default_rng(int(ls["seed"]) & 0xFFFFFFFF) if ls.get("seed") is not None
                 else self.ctx.rng(self.uid, 733))
            v = r.uniform(lo, hi, n)
        self._ls_static = v / max(float(v.max()), 1e-6)

    def _lscales(self, t: float) -> np.ndarray | None:
        if self.lscale is None:
            return None
        if self._ls_static is not None:
            return self._ls_static
        n = len(self.letters)
        u = 0.5 + 0.5 * np.sin(2.0 * np.pi * (self.ls_phase * np.arange(n) - self.ls_speed * (t - self.tm.t0)))
        return (self.ls_lo + (self.ls_hi - self.ls_lo) * u) / self.ls_hi

    def _fit_lscale(self, lay: dict, path: str, bw: float, bh: float, leading: float) -> dict:
        """Static letter sizes (largest 1) leave the lines narrower: grow the size until the widest scaled line fills
        the box width or the block reaches the box height."""
        f = get_font(path, lay["size"] * SS)
        sc, j, widest = self._ls_static, 0, 0.0
        for words, _ in lay["lines"]:
            wl = 0.0
            for wi, w in enumerate(words):
                prev = 0.0
                for k in range(len(w)):
                    cx = f.getlength(w[:k + 1]) / SS
                    wl += (cx - prev) * sc[min(j, len(sc) - 1)]
                    prev, j = cx, j + 1
                wl += lay["space"] if wi else 0.0
            widest = max(widest, wl)
        paras = [ws for ws, _ in lay["lines"]]
        hfit = fit_text("\n".join(" ".join(ws) for ws in paras), path, 1e9, bh, leading, wrap=False)
        new = int(min(hfit["size"] if hfit else lay["size"], lay["size"] * bw / max(widest, 1e-6)))
        if new > lay["size"]:
            lay = _layout(paras, path, new, bw, leading, False) or lay
        return lay

    def _setup_path(self, p: dict, bx: float, by: float, bw: float, bh: float) -> None:
        ctx, typ = self.ctx, p["type"]
        # the run: every letter in order, a space between words and lines
        chars, gap_after = [], []
        for i, L in enumerate(self.letters):
            chars.append(L["ch"])
            nxt = self.letters[i + 1] if i + 1 < len(self.letters) else None
            gap_after.append(nxt is not None and (nxt["line"] != L["line"] or nxt["k"] - L["k"] > 1))
        seq = "".join(c + (" " if g else "") for c, g in zip(chars, gap_after))
        f = get_font(self._path, _PATH_REF * SS)
        cum = [0.0] + [f.getlength(seq[:k + 1]) / SS for k in range(len(seq))]
        adv, gap, k = [], [], 0
        for c, g in zip(chars, gap_after):
            adv.append(cum[k + 1] - cum[k])
            k += 1
            gap.append(cum[k + 1] - cum[k] if g else 0.0)
            k += 1 if g else 0
        self.p_adv, self.p_gap = np.array(adv), np.array(gap)
        S = ctx.S
        P = {"type": typ, "travel": _pf(p, "travel", 0.0), "fill": max(0.05, _pf(p, "fill", 0.92))}
        if typ in ("arc", "circle"):
            c = p.get("center") or [0.5, 0.5]
            P["c"] = (bx + float(c[0]) * bw, by + float(c[1]) * bh)
            P["r"] = max(2.0, _pf(p, "radius", 1.0) * min(bw, bh) / 2.0)
            P["a"] = math.radians(_pf(p, "angle", 180.0 if p.get("flip") else 0.0))
            P["flip"] = bool(p.get("flip"))
            P["L"] = P["r"] * math.radians(min(360.0, _pf(p, "span", 360.0 if typ == "circle" else 180.0)))
            P["off"] = _pf(p, "offset", 0.0)
            cap = 0.6 * P["r"]
        elif typ == "line":
            p0, p1 = p.get("p0") or [0.0, 1.0], p.get("p1") or [1.0, 0.0]
            P["p0"] = np.array([bx + float(p0[0]) * bw, by + float(p0[1]) * bh])
            d = np.array([bx + float(p1[0]) * bw, by + float(p1[1]) * bh]) - P["p0"]
            P["L"] = max(1.0, float(np.hypot(*d)))
            P["u"] = d / P["L"]
            P["off"] = _pf(p, "offset", -0.35)
            cap = 1e9
        else:   # wave
            P["x0"], P["bw"], P["cy"] = bx, bw, by + bh / 2.0
            P["A"] = _pf(p, "amp", 0.25) * bh
            P["k"] = 2.0 * math.pi * _pf(p, "cycles", 1.5) / max(bw, 1.0)
            P["ph0"], P["speed"], P["step"] = _pf(p, "phase", 0.0), _pf(p, "speed", 0.0), _pf(p, "step", 0.25)
            ev = ctx.events(p["on"]) if p.get("on") is not None else np.zeros(0)
            P["ev"] = ev[ev >= self.tm.t0 - 1e-6]
            P["off"] = _pf(p, "offset", -0.35)
            P["_cache"] = (None, None, None)
            self.tpath = P
            X, Sx = self._wave_table(P["ph0"])
            P["L"] = float(np.interp(bx + bw, X, Sx) - np.interp(bx, X, Sx))
            cap = max(0.2 * bh, bh - 2.0 * abs(P["A"]))
        sc = self._ls_static if self._ls_static is not None else np.ones(len(adv))
        total = float((self.p_adv * sc).sum() + self.p_gap.sum())
        if p.get("size") is not None:
            size = _pf(p, "size", 0.1) * S
        else:
            size = min(cap, P["fill"] * P["L"] / max(total, 1e-6) * _PATH_REF)
        self.psize = max(4, int(round(size * self.overscale)))
        self.tpath = P
        self.boxes = []                                         # no highlight plates on path text
        self.pivot = (P["c"] if typ in ("arc", "circle") else (bx + bw / 2.0, by + bh / 2.0))
        self.block = self._path_bbox(self._path_place(self.tm.t0))

    def _wave_table(self, ph: float):
        P = self.tpath
        if P["_cache"][0] == ph:
            return P["_cache"][1], P["_cache"][2]
        bw = P["bw"]
        X = np.linspace(P["x0"] - 3.0 * bw, P["x0"] + 4.0 * bw, int(7 * max(bw, 8) / 2) + 2)
        Y = P["A"] * np.sin(P["k"] * (X - P["x0"]) + 2.0 * math.pi * ph)
        Sx = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
        Sx -= np.interp(P["x0"], X, Sx)
        P["_cache"] = (ph, X, Sx)
        return X, Sx

    def _wave_phase(self, t: float) -> float:
        P, tm = self.tpath, self.tm
        ph = P["ph0"] + P["speed"] * (t - tm.t0)
        for e in P["ev"]:
            if e > t:
                break
            ph += P["step"] * ease_in(clamp01((t - e) / 0.3), self.ctx.motion)
        return round(ph, 4)

    def _path_eval(self, s: np.ndarray, t: float):
        """Arclength (from the path's text anchor) -> x, y, tangent angle (rad, screen sense)."""
        P = self.tpath
        typ = P["type"]
        if typ in ("arc", "circle"):
            r = P["r"]
            a = P["a"] - s / r if P["flip"] else P["a"] + s / r
            x, y = P["c"][0] + r * np.sin(a), P["c"][1] - r * np.cos(a)
            return x, y, (a + math.pi if P["flip"] else a)
        if typ == "line":
            s = s + P["L"] / 2.0
            return P["p0"][0] + s * P["u"][0], P["p0"][1] + s * P["u"][1], np.full_like(s, math.atan2(P["u"][1], P["u"][0]))
        ph = self._wave_phase(t)
        X, Sx = self._wave_table(ph)
        s = s + float(np.interp(P["x0"] + P["bw"] / 2.0, X, Sx))
        x = np.interp(s, Sx, X)
        arg = P["k"] * (x - P["x0"]) + 2.0 * math.pi * ph
        return x, P["cy"] + P["A"] * np.sin(arg), np.arctan(P["A"] * P["k"] * np.cos(arg))

    def _path_place(self, t: float):
        """-> (x, y, theta, scale, adv) per letter at time t (glyph baseline centres, layer space)."""
        P, size = self.tpath, self.psize
        sc = self._lscales(t)
        sc = np.ones(len(self.p_adv)) if sc is None else np.asarray(sc, np.float64)
        k = size / _PATH_REF
        adv = self.p_adv * k * sc
        run = adv + self.p_gap * k
        start = np.concatenate([[0.0], np.cumsum(run)[:-1]])
        s = start + adv / 2.0 - run.sum() / 2.0 + P["travel"] * P["L"] * (t - self.tm.t0)
        x, y, th = self._path_eval(s, t)
        th = np.broadcast_to(np.asarray(th, np.float64), s.shape)
        off = P["off"] * size
        return x + off * np.sin(th), y - off * np.cos(th), th, sc, adv

    def _path_bbox(self, place) -> tuple:
        x, y = place[0], place[1]
        e = 0.75 * self.psize
        return (float(x.min() - e), float(y.min() - e), float(x.max() + e), float(y.max() + e))

    def _setup_backdrop(self, layer: dict, name: str, on_image: bool) -> None:
        bd = layer.get("backdrop")
        if bd is False or bd == "none" or (isinstance(bd, dict) and bd.get("type") == "none"):
            return
        if isinstance(bd, str):
            bd = {"type": bd}
        if bd is None and on_image and self.highlight is None and self.window is None \
                and (REGISTRY.get(name) or {}).get("shape") == "fine":
            bd = dict(AUTO_BACKDROP)
        if isinstance(bd, dict) and bd.get("type", "dim") in BACKDROP_TYPES:
            self.backdrop = bd

    def _letter_in(self, j: int, t: float, p_in: float):
        """Per-letter entrance (path / letter_scale): -> (opacity, scale, dy in px, positive = down)."""
        tm, verb = self.tm, self.tm.in_verb
        if verb == "type_on":
            if p_in >= 1.0:
                return 1.0, 1.0, 0.0
            return (1.0 if j < math.floor(p_in * len(self.letters) + 1e-6) else 0.0), 1.0, 0.0
        if verb not in ("word_stagger", "kinetic_split") or tm.din <= 0:
            return 1.0, 1.0, 0.0
        pi = clamp01((t - tm.t0 - j * self.lstagger) / tm.din)
        if pi >= 1.0:
            return 1.0, 1.0, 0.0
        size = self.psize if self.tpath is not None else self.size
        if verb == "word_stagger":
            return clamp01(pi * 2.5), 1.0, (1.0 - ease_in(pi, self.ctx.motion)) * 0.55 * size
        return clamp01(pi * 5.0), 1.0, -(1.0 - bounce_out(pi)) * 1.4 * size     # drop and bounce

    def _window_image(self, t: float) -> np.ndarray:
        tm, ctx = self.tm, self.ctx
        p = clamp01((t - tm.t0) / max(1e-3, tm.t1 - tm.t0))
        z = self.win_zoom[0] + (self.win_zoom[1] - self.win_zoom[0]) * p
        key = round(z, 5)
        if self._win_cache[0] != key:
            crop = cover_crop(self.win_wh[0], self.win_wh[1], ctx.W, ctx.H, self.win_focus, z)
            self._win_cache = (key, warp_crop(self.win_src, crop, ctx.W, ctx.H))
        return self._win_cache[1]

    def _outlined(self, t: float) -> bool:
        if not self.ol:
            return False
        start = self.ol.get("start", "outline") == "outline"
        sw = self.ol.get("swap")
        if sw is None:
            return start
        ev = self.ctx.events(sw)
        n = int(np.searchsorted(ev, t + 1e-6, side="right") - np.searchsorted(ev, self.tm.t0 + 1e-6, side="right"))
        return start if n % 2 == 0 else not start

    def _build_letters(self, path: str, text: str, bcx: float, bcy: float) -> None:
        """Per-letter pen positions (from the word layout) for the letter-level verbs."""
        self._path = path
        self._case = "lower" if text == text.lower() and text != text.upper() else "upper"
        self.box_c = (bcx, bcy)
        for li in range(len(self.lay["lines"])):
            row, k = [], 0
            for wd in (w for w in self.words if w["line"] == li):
                for j, ch in enumerate(wd["text"]):
                    row.append(dict(ch=ch, x=wd["x"] + wd["cx"][j], base=wd["base"], k=k, line=li,
                                    cx=wd["x"] + (wd["cx"][j] + wd["cx"][j + 1]) / 2.0))
                    k += 1
                k += 1                                   # the space between words
            for L in row:
                L["kc"] = L["k"] - (k - 2) / 2.0         # signed index from the line centre (tracking)
            self.letters.extend(row)
        n = len(self.letters)
        order = self.ctx.rng(self.uid, 515).permutation(n)
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        for j, (L, r) in enumerate(zip(self.letters, rank)):
            L["res"] = 0.3 + 0.7 * (r + 1) / n               # scramble: in-progress at which it resolves
            L["j"] = j

    def _draw_letters(self, frame, t, st, col, Mb, p_in, p_out, wclip) -> bool:
        """Letter-level verbs; True when they drew the layer this frame (the word sprites are skipped)."""
        tm = self.tm
        if not self.letters:
            return False
        if self.tpath is not None:
            self._draw_path(frame, t, st, col, Mb, p_in, wclip)
            return True
        if tm.in_verb == "row_slide" or tm.out_verb == "row_slide":
            self._draw_rows(frame, t, st, col, Mb, p_in, p_out, wclip)
            return True
        if tm.in_verb == "scramble" and p_in < 1.0:
            self._draw_scramble(frame, t, st, col, Mb, p_in, wclip)
            return True
        if self.vary is not None or self.lscale is not None:
            self._draw_vary(frame, t, st, col, Mb, wclip, p_in)
            return True
        return False

    def _draw_vary(self, frame, t, st, col, Mb, wclip, p_in=1.0) -> None:
        """Per-letter variable-font / scale wave and letter_scale sizes; each line re-flows around its anchor."""
        tm = self.tm
        lo, hi = self.vary_range
        stroke = self.ol_px if self._outlined(t) else 0.0
        size = self.size
        ls = self._lscales(t)
        per = self.lscale is not None
        vc = 0.35 * size if per and self.ls_center else 0.0
        shadow = per and self._shadow_on and not self._masking and not stroke
        for li, (_, lw) in enumerate(self.lay["lines"]):
            row = [L for L in self.letters if L["line"] == li]
            if not row:
                continue
            us = []
            for j, L in enumerate(row):
                if self.vary is None:
                    u = 0.5
                elif self.vary_ev is not None:
                    u = decay_env(self.vary_ev, t - j * abs(self.vary_phase), max(0.05, 0.35 * self.vary_period))
                else:
                    u = 0.5 - 0.5 * math.cos(2.0 * math.pi * ((t - tm.t0) / self.vary_period - self.vary_phase * j))
                us.append(round(u * (VARY_STEPS - 1)) / (VARY_STEPS - 1))
            keys, advs, scales = [], [], []
            for L, u in zip(row, us):
                key = self._vary_key(lo + (hi - lo) * u) if self.vary_axis else self._path
                s = max(0.05, 1.0 + self.vary_scale * (2.0 * u - 1.0)) if self.vary is not None else 1.0
                if ls is not None:
                    s *= float(ls[L["j"]])
                keys.append(key)
                scales.append(s)
                advs.append(get_font(key, size * SS).getlength(L["ch"]) / SS * s)
            gaps = [self.lay["space"] if (b["k"] - a["k"]) > 1 else 0.0 for a, b in zip(row, row[1:])] + [0.0]
            width = sum(advs) + sum(gaps)
            x0 = row[0]["x"]
            if self.align == "left":
                pen = x0
            elif self.align == "right":
                pen = x0 + lw - width
            else:
                pen = x0 + lw / 2.0 - width / 2.0
            draws = []
            for L, key, s, adv, gap in zip(row, keys, scales, advs, gaps):
                op, s_in, dy = self._letter_in(L["j"], t, p_in) if per else (1.0, 1.0, 0.0)
                if op > 0.0:
                    draws.append((key, L["ch"], s * s_in, pen, L["base"], dy, op))
                pen += adv + gap
            if shadow:
                for key, ch, s, x, base, dy, op in draws:
                    sa, sx, sy = _glyph_shadow(key, size, ch)
                    M = affine(scale=s, pivot=(x, base - vc), ox=x + sx, oy=base + sy, dy=dy + 0.04 * size)
                    composite(frame, sa, (0.0, 0.0, 0.0), _compose(Mb, M), st.opacity * op * 0.55, wclip, st.blur)
            for key, ch, s, x, base, dy, op in draws:
                a, ox, oy = _glyph(key, size, ch, stroke)
                M = affine(scale=s, pivot=(x, base - vc), ox=x + ox, oy=base + oy, dy=dy)
                composite(frame, a, col, _compose(Mb, M), st.opacity * op, wclip, st.blur)

    def _draw_path(self, frame, t, st, col, Mb, p_in, wclip) -> None:
        """Glyphs along the layer's path, each turned to the tangent (scramble / per-letter entrances apply)."""
        tm, ctx = self.tm, self.ctx
        x, y, th, sc, adv = self._path_place(t)
        size = self.psize
        stroke = max(1.0, _pf(self.ol, "width", OUTLINE_EM) * size) if self._outlined(t) else 0.0
        vc = 0.35 * size if (self.lscale is not None and self.ls_center) else 0.0
        scr = tm.in_verb == "scramble" and p_in < 1.0
        if scr:
            rate = max(1.0, _pf(tm.vin, "rate", 18.0))
            tick = int(math.floor((t - tm.t0) * rate))
            scol = ctx.color(tm.vin["color"]) if tm.vin.get("color") else col
            chars = _SCRAMBLE_CHARS[self._case]
        draws = []
        for j, L in enumerate(self.letters):
            op, s_in, dy = self._letter_in(j, t, p_in)
            if op <= 0.0:
                continue
            ch, c = L["ch"], col
            hx = adv[j] / max(sc[j], 1e-6) / 2.0
            if scr and p_in < L["res"] - 1e-9:
                ch, c = chars[int(ctx.rng(self.uid, 517, j, tick).integers(len(chars)))], scol
            sn, cs = math.sin(th[j]), math.cos(th[j])
            T = (x[j] + (vc - dy) * sn, y[j] + (dy - vc) * cs)
            draws.append((ch, c, sc[j] * s_in, th[j], hx, T, op, ch != L["ch"]))
        if self._shadow_on and not self._masking and not stroke:
            for ch, c, s, a_, hx, T, op, rnd in draws:
                sa, sx, sy = _glyph_shadow(self._path, size, ch)
                if rnd:
                    sx = hx - sa.shape[1] / 2.0
                M = affine(scale=s, rot=a_, pivot=(hx, -vc), dx=T[0] - hx, dy=T[1] + vc + 0.04 * size, ox=sx, oy=sy)
                composite(frame, sa, (0.0, 0.0, 0.0), _compose(Mb, M), st.opacity * op * 0.55, wclip, st.blur)
        for ch, c, s, a_, hx, T, op, rnd in draws:
            a, ox, oy = _glyph(self._path, size, ch, stroke)
            if rnd:
                ox = hx - a.shape[1] / 2.0                     # random glyph centred on the letter's cell
            M = affine(scale=s, rot=a_, pivot=(hx, -vc), dx=T[0] - hx, dy=T[1] + vc, ox=ox, oy=oy)
            composite(frame, a, c, _compose(Mb, M), st.opacity * op, wclip, st.blur)

    def _draw_scramble(self, frame, t, st, col, Mb, p_in, wclip) -> None:
        vin, ctx = self.tm.vin, self.ctx
        rate = max(1.0, _pf(vin, "rate", 18.0))
        tick = int(math.floor((t - self.tm.t0) * rate))
        scol = ctx.color(vin["color"]) if vin.get("color") else col
        chars = _SCRAMBLE_CHARS[self._case]
        stroke = self.ol_px if self._outlined(t) else 0.0
        for j, L in enumerate(self.letters):
            if p_in >= L["res"] - 1e-9:
                a, ox, oy = _glyph(self._path, self.size, L["ch"], stroke)
                X, c = L["x"] + ox, col
            else:
                ch = chars[int(ctx.rng(self.uid, 517, j, tick).integers(len(chars)))]
                a, ox, oy = _glyph(self._path, self.size, ch, stroke)
                X, c = L["cx"] - a.shape[1] / 2.0, scol       # random glyph centred on the letter's cell
            composite(frame, a, c, _compose(Mb, translate(X, L["base"] + oy)), st.opacity, wclip, st.blur)

    def _draw_rows(self, frame, t, st, col, Mb, p_in, p_out, wclip) -> None:
        tm, ctx = self.tm, self.ctx
        vin = tm.vin if tm.in_verb == "row_slide" else {}
        vout = tm.vout if tm.out_verb == "row_slide" else {}
        spec = vin or vout
        rows = max(1, int(_pf(spec, "rows", 3)))
        sc = max(0.05, _pf(spec, "scale", 1.0))
        e_in = ease_in(p_in, ctx.motion) if vin else 1.0
        e_out = clamp01(ease_out(p_out, ctx.motion)) if (vout and p_out > 0.0) else 0.0
        off = (1.0 - e_in) * _pf(vin, "travel", 0.6) + e_out * _pf(vout, "travel", 0.6)
        track = ((1.0 - e_in) * _pf(vin, "track", 0.0) + e_out * _pf(vout, "track", 0.0)) * self.size * sc
        drift = _pf(spec, "drift", 0.0) * ctx.W * (t - (tm.t0 + tm.din))
        op = st.opacity * (1.0 - e_out if vout.get("fade") else 1.0)
        fade_in = bool(vin.get("fade")) and p_in < 1.0
        stagger = min(0.95, max(0.0, _pf(vin, "stagger", 0.0)))
        alt = ctx.color(spec["alt_color"]) if spec.get("alt_color") else col
        gs = max(4, int(round(self.size * sc)))
        bcx, bcy = self.box_c
        pitch = len(self.lay["lines"]) * self.lay["pitch"] * sc * (1.0 + _pf(spec, "gap", 0.0))
        n = len(self.letters)
        outlined = self._outlined(t)
        alt_rows = bool(self.ol) and self.ol.get("rows", "all") == "alt"
        for r in range(rows):
            rr = r - (rows - 1) / 2.0
            dx = (1.0 if r % 2 == 0 else -1.0) * _ROW_SPEED[r % len(_ROW_SPEED)] * off * ctx.W + drift
            c = col if abs(rr) <= 0.5 else alt
            hollow = (abs(rr) > 0.5) if alt_rows else outlined
            stroke = _pf(self.ol, "width", OUTLINE_EM) * gs if (self.ol and hollow) else 0.0
            for j, L in enumerate(self.letters):
                a_op = op
                if fade_in:
                    q = j / max(1, n - 1)
                    a_op *= clamp01((p_in - stagger * q) / (1.0 - stagger))
                if a_op <= 1.0 / 512.0:
                    continue
                a, ox, oy = _glyph(self._path, gs, L["ch"], stroke)
                X = bcx + (L["x"] - bcx) * sc + L["kc"] * track + ox + dx
                Y = bcy + (L["base"] - bcy) * sc + oy + rr * pitch
                composite(frame, a, c, _compose(Mb, translate(X, Y)), a_op, wclip, st.blur)

    # ------------------------------------------------------------------ per frame
    def rotation(self, t: float) -> float:
        """Layer rotation in degrees (clockwise) at t: "rotate" plus the in / out "spin"."""
        tm, ctx = self.tm, self.ctx
        deg = rotation_at(self.layer.get("rotate"), t)
        sp = _pf(tm.vin, "spin", 0.0)
        if sp and tm.din > 0:
            deg += sp * (1.0 - ease_in(tm.p_in(t), ctx.motion))
        sp = _pf(tm.vout, "spin", 0.0)
        if sp and tm.dout > 0:
            deg += sp * clamp01(ease_out(tm.p_out(t), ctx.motion))
        return deg

    def draw(self, frame: np.ndarray, t: float, fi: int) -> None:
        if not self.words or not self.tm.active(t):
            return
        rot = self.rotation(t)
        if abs(rot) < 1e-3 and self.backdrop is None:
            self._draw_on(frame, t, fi)
            return
        st = self._state(t, fi)
        if st is None:
            return
        pv = (self.pivot[0] + st.dx, self.pivot[1] + st.dy)
        if self.backdrop is not None:
            self._apply_backdrop(frame, t, st, rot, pv)
        if abs(rot) < 1e-3:
            self._draw_on(frame, t, fi)
        else:
            self._draw_rotated(frame, t, fi, rot, pv)

    def _draw_rotated(self, frame, t, fi, rot, pv) -> None:
        """Draw into a canvas covering the frame's footprint in layer space, matte it (black / white difference)
        to premultiplied RGBA and composite it rotated by ``rot`` degrees about ``pv``."""
        H, W = frame.shape[:2]
        R = affine(rot=math.radians(rot), pivot=pv)
        Ri = affine(rot=-math.radians(rot), pivot=pv)
        c = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], np.float64) @ Ri.T
        x0, y0 = int(math.floor(c[:, 0].min())) - 2, int(math.floor(c[:, 1].min())) - 2
        x1, y1 = int(math.ceil(c[:, 0].max())) + 2, int(math.ceil(c[:, 1].max())) + 2
        self._shift = (-x0, -y0)
        try:
            black = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            self._draw_on(black, t, fi)
            white = np.ones_like(black)
            self._draw_on(white, t, fi)
        finally:
            self._shift = (0, 0)
        alpha = 1.0 - cv2.cvtColor(white - black, cv2.COLOR_BGR2GRAY)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        bx, by, bw, bh = cv2.boundingRect((alpha > 1.0 / 512.0).astype(np.uint8))
        if bw == 0 or bh == 0:
            return
        rgba = np.dstack([black[by:by + bh, bx:bx + bw], alpha[by:by + bh, bx:bx + bw]])
        composite(frame, rgba, None, _compose(R, translate(x0 + bx, y0 + by)), 1.0)

    def _apply_backdrop(self, frame, t, st, rot, pv) -> None:
        """Dim / blur / band the picture behind the text block: a soft-edged rect, scaled, moved and rotated with
        the layer, faded with its opacity."""
        bd = self.backdrop
        typ = bd.get("type", "dim")
        amt = clamp01(_pf(bd, "amount", _BD_AMOUNT[typ]))
        k = st.opacity * (1.0 if typ == "blur" else amt)
        if k <= 1.0 / 512.0:
            return
        size = (self.psize if self.tpath is not None else self.size) * st.scale
        x0, y0, x1, y1 = self._path_bbox(self._path_place(t)) if self.tpath is not None else self.block
        px, py = self.pivot
        cx = px + ((x0 + x1) / 2.0 - px) * st.scale + st.dx
        cy = py + ((y0 + y1) / 2.0 - py) * st.scale + st.dy
        pad = _pf(bd, "pad", 0.3) * size
        soft = max(1.0, _pf(bd, "soft", 0.25 if typ == "band" else 0.9) * size)
        hw = (x1 - x0) / 2.0 * st.scale + pad
        hh = (y1 - y0) / 2.0 * st.scale + pad
        a = math.radians(rot)
        ca, sa = math.cos(a), math.sin(a)
        cx, cy = pv[0] + ca * (cx - pv[0]) - sa * (cy - pv[1]), pv[1] + sa * (cx - pv[0]) + ca * (cy - pv[1])
        H, W = frame.shape[:2]
        if typ == "band":
            X0, Y0, X1, Y1 = 0, 0, W, H
        else:
            ex = abs(ca) * (hw + soft) + abs(sa) * (hh + soft)
            ey = abs(sa) * (hw + soft) + abs(ca) * (hh + soft)
            X0, Y0 = max(0, int(cx - ex) - 1), max(0, int(cy - ey) - 1)
            X1, Y1 = min(W, int(math.ceil(cx + ex)) + 1), min(H, int(math.ceil(cy + ey)) + 1)
        if X1 <= X0 or Y1 <= Y0:
            return
        xs = np.arange(X0, X1, dtype=np.float32) + 0.5 - np.float32(cx)
        ys = np.arange(Y0, Y1, dtype=np.float32)[:, None] + 0.5 - np.float32(cy)
        v = -sa * xs[None, :] + ca * ys
        m = _soft(hh - np.abs(v), soft)
        if typ != "band":
            u = ca * xs[None, :] + sa * ys
            m = m * _soft(hw - np.abs(u), soft)
        if typ == "band" and abs(sa) < 1e-6:                    # horizontal band: only the rows it covers
            rows = np.nonzero(m[:, 0] > 1e-4)[0]
            if not len(rows):
                return
            Y0, Y1 = Y0 + int(rows[0]), Y0 + int(rows[-1]) + 1
            m = np.ascontiguousarray(np.broadcast_to(m[rows[0]:rows[-1] + 1, :1], (Y1 - Y0, X1 - X0)))
        m3 = cv2.cvtColor((m * np.float32(k)).astype(np.float32), cv2.COLOR_GRAY2BGR)
        reg = frame[Y0:Y1, X0:X1]
        if typ == "blur":
            sig = max(1.0, (0.08 + 0.35 * amt) * size)
            p = int(math.ceil(3 * sig))
            Z0, Q0, Z1, Q1 = max(0, X0 - p), max(0, Y0 - p), min(W, X1 + p), min(H, Y1 + p)
            bl = cv2.GaussianBlur(frame[Q0:Q1, Z0:Z1], (0, 0), sig)[Y0 - Q0:Y1 - Q0, X0 - Z0:X1 - Z0]
            bl *= np.float32(1.0 - 0.3 * amt)                    # and a little darker
            reg += (bl - reg) * m3
        else:
            col = self.ctx.color(bd.get("color"), (0.0, 0.0, 0.0)) if bd.get("color") else \
                (np.zeros(3, np.float32) if typ == "dim" else self.ctx.bg_color)
            reg += (np.asarray(col, np.float32) - reg) * m3

    def _win_img(self, t: float, shape) -> np.ndarray:
        """The window image in the current canvas (frame, or the off-frame rotation canvas)."""
        img = self._window_image(t)
        sx, sy = self._shift
        if (sx, sy) == (0, 0) and img.shape == tuple(shape):
            return img
        out = np.zeros(shape, np.float32)
        h, w = img.shape[:2]
        X0, Y0, X1, Y1 = max(0, sx), max(0, sy), min(shape[1], sx + w), min(shape[0], sy + h)
        if X1 > X0 and Y1 > Y0:
            out[Y0:Y1, X0:X1] = img[Y0 - sy:Y1 - sy, X0 - sx:X1 - sx]
        return out

    def _draw_on(self, frame: np.ndarray, t: float, fi: int) -> None:
        win = getattr(self, "window", None)
        if win is None:
            self._draw(frame, t, fi)
            return
        if not self.words or not self.tm.active(t):
            return
        # type as a mask: draw the letters as white into a matte, then fill / knock out through it
        m = np.zeros_like(frame)
        st = self._draw(m, t, fi, mask=True)
        if st is None:
            return
        a3 = m
        np.clip(a3, 0.0, 1.0, out=a3)
        op = np.float32(st.opacity)
        if win.get("mode", "fill") == "knockout":
            pa = np.float32(_pf(win, "plate_opacity", 1.0)) * op
            plate = np.empty_like(frame)
            plate[:] = np.asarray(self.ctx.color(win.get("plate", "palette:0"), (0, 0, 0)), np.float32)
            reg = np.zeros(frame.shape[:2], np.float32)
            if win.get("pad") is not None:
                pd = _pf(win, "pad", 0.3) * self.size
                x0, y0, x1, y1 = self.block
                X0, Y0 = int(max(0, x0 - pd + st.dx)), int(max(0, y0 - pd + st.dy))
                X1, Y1 = int(min(frame.shape[1], x1 + pd + st.dx)), int(min(frame.shape[0], y1 + pd + st.dy))
                reg[Y0:Y1, X0:X1] = 1.0
            else:
                reg[:] = 1.0
            under = self._win_img(t, frame.shape) if self.win_src is not None else None
            if under is not None:                              # holes show the window image
                frame += (under - frame) * (a3 * op)
            k = (1.0 - a3) * (pa * cv2.cvtColor(reg, cv2.COLOR_GRAY2BGR))
            frame += (plate - frame) * k
        elif self.win_src is not None:
            frame += (self._win_img(t, frame.shape) - frame) * (a3 * op)

    def _state(self, t: float, fi: int):
        """Block-level animation state at t (None when invisible)."""
        tm, ctx = self.tm, self.ctx
        st = Anim()
        p_in = tm.p_in(t)
        apply_in(st, "cut" if tm.in_verb in LETTER_VERBS else tm.in_verb, p_in, ctx.motion, self.U, self.kind)
        p_out = tm.p_out(t)
        apply_out(st, "cut" if tm.out_verb in LETTER_VERBS else tm.out_verb, p_out, ctx.motion, self.U, self.kind)
        apply_emphasis(st, self.layer.get("emphasis"), t, fi, ctx, tm, self.uid)
        glitch_jitter(st, p_in, ctx, self.uid, fi)
        if st.opacity <= 0.0 or st.scale <= 1e-3:
            return None
        return st

    def _draw(self, frame: np.ndarray, t: float, fi: int, mask: bool = False):
        """Draw the layer; mask=True draws a white matte at full opacity (no plates / shadows / rules) and
        returns the layer's animation state (its opacity applies to the matte)."""
        tm, ctx = self.tm, self.ctx
        if not self.words or not tm.active(t):
            return None
        st = self._state(t, fi)
        if st is None:
            return None
        p_in, p_out = tm.p_in(t), tm.p_out(t)
        st.dx += self._shift[0]                       # off-frame canvas (rotation): layer px -> canvas px
        st.dy += self._shift[1]
        self._masking = mask
        col, hcol = self.color, self.highlight
        if mask:
            real = st.opacity
            st.opacity, col, hcol = 1.0, (1.0, 1.0, 1.0), None
            st.flip = False
        if st.flip:
            if hcol is not None:
                col, hcol = hcol, col
            else:
                col = self.ul_color
        Mb = affine(scale=st.scale, pivot=self.pivot, dx=st.dx, dy=st.dy)
        wclip = None
        if st.wipe_lo > 0.0 or st.wipe_hi < 1.0:
            x0, x1 = self.block[0] - 2, self.block[2] + 2
            wclip = (x0 + (x1 - x0) * st.wipe_lo + st.dx, -1e6, x0 + (x1 - x0) * st.wipe_hi + st.dx, 1e6)

        verb = tm.in_verb
        motion = ctx.motion
        p_all = clamp01((t - tm.t0) / self.in_total) if self.in_total > 0 else 1.0

        # highlight boxes: wipe on with the in-phase, off with the out-phase
        if self.boxes and not mask:
            on = 1.0 if verb == "cut" else clamp01(ease_in(p_all, "snappy" if motion == "bouncy" else motion))
            off = 0.0 if tm.out_verb == "cut" else clamp01(p_out * 1.0)
            for (bs, X0, Y0) in self.boxes:
                bw_ = bs.shape[1]
                clip = (X0 + bw_ * off, -1e6, X0 + bw_ * on, 1e6)
                clip = (clip[0] + st.dx, clip[1], clip[2] + st.dx, clip[3])
                composite(frame, bs, hcol, _compose(Mb, translate(X0, Y0)), st.opacity, _isect(clip, wclip))

        letters_drawn = self._draw_letters(frame, t, st, col, Mb, p_in, p_out, wclip)

        # per-word state
        n_vis = None
        if verb == "type_on" and p_in < 1.0:
            n_vis = int(math.floor(p_in * self.n_chars + 1e-6))
        word_states = []
        seen = 0
        cursor = None
        for wd in self.words:
            i = wd["idx"]
            op, dx, dy, s = 1.0, 0.0, 0.0, 1.0
            clip = None
            if verb == "word_stagger":
                pi = clamp01((t - tm.t0 - i * self.stagger) / tm.din) if tm.din > 0 else 1.0
                if pi < 1.0:
                    e = ease_in(pi, motion)
                    op = clamp01(pi * 2.5)
                    dy = (1.0 - e) * 0.55 * self.lay["pitch"]
            elif verb == "kinetic_split":
                pi = clamp01((t - tm.t0 - i * self.stagger) / tm.din) if tm.din > 0 else 1.0
                if pi < 1.0:
                    e = ease_in(pi, motion)
                    side = -1.0 if i % 2 == 0 else 1.0
                    dx = side * (1.0 - e) * 0.35 * ctx.W
                    s = 1.0 + (1.0 - clamp01(e if motion == "glitch" else back_out(pi, 1.4))) * 0.5
                    op = clamp01(pi * 3.0)
            elif n_vis is not None:
                k = n_vis - seen
                L = len(wd["text"])
                seen += L
                if k <= 0:
                    op = 0.0
                    if cursor is None:
                        cursor = (wd["x"], wd)
                elif k < L:
                    clip = (-1e6, -1e6, wd["x"] + wd["cx"][k] + self._shift[0], 1e6)
                    cursor = (wd["x"] + wd["cx"][k], wd)
                elif seen == n_vis and cursor is None:
                    cursor = (wd["x"] + wd["w"] + 0.06 * self.size, wd)
            if op <= 0.0 or letters_drawn:
                word_states.append(None)
                continue
            cw = (wd["ox"] + wd["a"].shape[1] / 2.0, wd["oy"] + wd["a"].shape[0] / 2.0)
            Mw = affine(scale=s, pivot=cw, dx=dx, dy=dy, ox=wd["ox"], oy=wd["oy"])
            M = _compose(Mb, Mw)
            if st.line_clip:
                r = self.line_rects[wd["line"]]
                clip = _isect(clip, (-1e6, r[1] + self._shift[1], 1e6, r[3] + self._shift[1]))
            word_states.append((M, op * st.opacity, _isect(clip, wclip), dx, dy, s, cw))

        outlined = self._outlined(t)
        if self.shadows and not mask and not outlined:
            for (sa, sx, sy), ws in zip(self.shadows, word_states):
                if ws is None:
                    continue
                M, op, clip, dx, dy, s, cw = ws
                Ms = _compose(Mb, affine(scale=s, pivot=cw, dx=dx, dy=dy, ox=sx, oy=sy))
                composite(frame, sa, (0.0, 0.0, 0.0), Ms, op * 0.55, clip, st.blur)
        for wd, ws in zip(self.words, word_states):
            if ws is None:
                continue
            M, op, clip = ws[:3]
            if outlined:
                dx, dy, s, cw = ws[3:]
                M = _compose(Mb, affine(scale=s, pivot=cw, dx=dx, dy=dy, ox=wd["ox_ol"], oy=wd["oy_ol"]))
                composite(frame, wd["a_ol"], col, M, op, clip, st.blur)
            else:
                composite(frame, wd["a"], col, M, op, clip, st.blur)

        if mask:
            st.opacity = real
            return st

        if cursor is not None and p_in < 1.0 and self.tpath is None and self.lscale is None:
            x, wd = cursor
            r = self.line_rects[wd["line"]]
            cw_ = max(1.5, 0.07 * self.size)
            pts = np.array([[x + cw_ / 2, wd["base"] - self.lay["asc"]], [x + cw_ / 2, wd["base"] + 0.05 * self.size]])
            pts = pts @ Mb[:, :2].T + Mb[:, 2]
            draw_ops(frame, [("line", pts, False, cw_)], hcol if hcol is not None else col, st.opacity)

        if st.underline:
            r = self.line_rects[-1]
            y = r[3] + 0.16 * self.size
            pts = np.array([[self.block[0], y], [self.block[0] + (self.block[2] - self.block[0]) * st.underline, y]])
            pts = pts @ Mb[:, :2].T + Mb[:, 2]
            draw_ops(frame, [("line", pts, False, max(2.0, 0.07 * self.size))], self.ul_color, st.opacity)
        return st
