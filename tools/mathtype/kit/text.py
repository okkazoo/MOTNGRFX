"""Text: liked-font registry with roles, phrase splitting with readability timing, line layout and
justification, automatic mixed-font assignment, glyph rasterising, SDFs (cached), SDF morphs, outlines.

Setup-time code (CPU, once per phrase and size); the per-frame work samples the returned arrays on the
device (draw.sample, camera.warp).

    ph = text.split_phrases("There is geometry in the humming of the strings.", n=3)
    lines = text.auto_mix(text.split_lines(ph[0].text))          # tall face + puffy key word
    lay = text.layout(lines, W=720, H=900, text_w=600)            # design px, justified stack
    d = lay.sdf()                                                 # (H, W) float32, negative inside
    polys = lay.outlines(ds=2.0)                                  # closed glyph outlines (design px)
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mograph.font_registry import AXIS_NAMES, REGISTRY, font_axes, font_path

# ------------------------------------------------------------------------------------------- registry
# The user's liked faces by role (_info/decisions.md, 2026-10-23). First = preferred.
LIKED = {
    "puffy": ["Pilowlava", "Blackout", "Bungee", "Climate Crisis", "Shrikhand"],
    "tall": ["Le Murmure", "Facade", "Big Shoulders Display"],
    "fine": ["Major Mono Display", "Space Mono"],
    "ornamental": ["Monoton"],
    "text": ["Space Grotesk"],
}
ROLE = {f: r for r, fs in LIKED.items() for f in fs}
# notes: Pilowlava loses legibility as a 3D height field (use flat / particle fill).
CAVEATS = {"Pilowlava": "flat or particle fill only; not as an inflated height field"}

FUNCTION_WORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "it", "as",
                  "by", "with", "from", "that", "this", "be", "we", "us", "our", "its", "into", "was", "are"}


def fonts(role: str) -> list[str]:
    return [f for f in LIKED[role] if f in REGISTRY]


def role_of(name: str) -> str | None:
    return ROLE.get(name) or (REGISTRY.get(name) or {}).get("shape")


_TAG_OF = {v: k for k, v in AXIS_NAMES.items()}           # "weight" -> "wght", "contrast" -> "ctst"


@functools.lru_cache(maxsize=512)
def _font(name: str, size: int, axes_key: tuple) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(font_path(name), size)
    ax = {k.lower(): v for k, v in axes_key}
    if ax:
        try:
            vals = []
            for a in f.get_variation_axes():
                nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
                tag = _TAG_OF.get(nm.lower(), nm.lower())
                vals.append(ax.get(tag, a["default"]))
            f.set_variation_by_axes(vals)
        except OSError:                                    # static font: no axes
            pass
    return f


def load_font(name: str, size: int, axes: dict | None = None) -> ImageFont.FreeTypeFont:
    """Registry face at `size` px with its default variation axes (all tags mapped, incl. CTST), plus
    `axes` overrides. Cached: the same (name, size, axes) returns the same object."""
    ax = dict(font_axes(name))
    ax.update(axes or {})
    return _font(name, int(size), tuple(sorted(ax.items())))


def line_width(f, s: str, track: float = 0.0) -> float:
    """Ink width of s set glyph by glyph with tracking `track` (em)."""
    adv = sum(f.getlength(ch) for ch in s[:-1]) + track * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def cap_height(f) -> float:
    hb = f.getbbox("H")
    return hb[3] - hb[1]


# ------------------------------------------------------------------------------------------- phrases
def read_time(text: str) -> float:
    """Minimum read window (s): 0.5 s + 0.3 s per word, and never above 20 characters per second."""
    words = len(text.split())
    return max(0.5 + 0.3 * words, len(text.replace(" ", "")) / 20.0)


@dataclass
class Phrase:
    text: str
    words: list = field(default_factory=list)
    read: float = 0.0

    def __post_init__(self):
        self.words = self.text.split()
        self.read = read_time(self.text)


def _clean(s):
    return re.sub(r"\s+", " ", s.strip())


def _split(words, n, max_chars):
    """Break words into n groups: minimise the longest group (chars) + penalties for groups that end on a
    function word or start after a comma-less break. Exhaustive DP (sentences are short)."""
    W = len(words)
    n = max(1, min(n, W))

    def grp_cost(i, j):
        g = words[i:j]
        s = " ".join(g)
        c = len(s)
        pen = 0.0
        if g[-1].lower().strip(",.;:!?") in FUNCTION_WORDS and j < W:
            pen += 6.0
        if g[-1][-1] in ",;:." and j < W:
            pen -= 3.0
        if max_chars and c > max_chars:
            pen += 4.0 * (c - max_chars)
        return c, pen

    @functools.lru_cache(maxsize=None)
    def best(i, k):
        if k == 1:
            c, p = grp_cost(i, W)
            return (c + p, [W])
        out = None
        for j in range(i + 1, W - k + 2):
            c, p = grp_cost(i, j)
            sub, cuts = best(j, k - 1)
            sc = max(c, sub) + p + 0.1 * abs(c - sub)
            if out is None or sc < out[0]:
                out = (sc, [j] + cuts)
        return out

    _, cuts = best(0, n)
    groups, i = [], 0
    for j in cuts:
        groups.append(" ".join(words[i:j]))
        i = j
    return groups


def split_phrases(sentence: str, n: int | None = None, max_words: int = 4, upper: bool = True) -> list[Phrase]:
    """Split a sentence into n phrases (default: enough that no phrase exceeds max_words), balanced by
    length, avoiding breaks after function words. Each Phrase carries its minimum read window."""
    s = _clean(sentence)
    words = s.split()
    if n is None:
        n = max(1, -(-len(words) // max_words))
    out = []
    for g in _split(words, n, 0):
        g = g.rstrip(".")
        out.append(Phrase(g.upper() if upper else g))
    return out


def split_lines(text: str, max_lines: int = 3, max_chars: int = 11) -> list[str]:
    """Break a phrase into 1..max_lines lines, as few as keep each line <= max_chars (balanced)."""
    words = _clean(text).split()
    for n in range(1, max_lines + 1):
        lines = _split(words, n, max_chars)
        if max(len(x) for x in lines) <= max_chars or n == min(max_lines, len(words)):
            return lines
    return lines


def key_word(text: str) -> str:
    """The word that carries the meaning: longest non-function word (ties: last)."""
    ws = [w.strip(",.;:!?") for w in text.split()]
    cands = [w for w in ws if w.lower() not in FUNCTION_WORDS] or ws
    return max(reversed(cands), key=len)


def auto_mix(lines: list[str], key: str | None = None, tall: str = "Le Murmure", puffy: str = "Blackout",
             morph_to: str | None = None, cap_tall: float | None = None, cap_key: float | None = None):
    """Mixed-font assignment: the key word (given, or key_word()) gets its own line in the puffy face, the
    other lines the tall face. Returns line specs (text, face, morph_face or None, cap or None)."""
    joined = " ".join(lines)
    key = (key or key_word(joined)).upper() if joined.isupper() else (key or key_word(joined))
    words = joined.split()
    bare = [w.strip(",.;:!?") for w in words]
    out = list(lines)
    if key in bare and len(words) > 1:
        i = bare.index(key)
        before, after = " ".join(words[:i]), " ".join(words[i + 1:])
        out = ((split_lines(before, 2) if before else []) + [words[i]] + (split_lines(after, 2) if after else []))
    return [(ln, puffy, morph_to, cap_key) if ln.strip(",.;:!?") == key else (ln, tall, None, cap_tall)
            for ln in out]


# ------------------------------------------------------------------------------------------- layout
@dataclass
class Layout:
    """A justified stack of lines on a (Hpx, Wpx) canvas at k px per design px. fa / fb: fonts of the A and
    B (morph) faces per line; centres: per-glyph ink centre x (px) of the A layout (B glyphs sit on them, so
    a face morph keeps every letter in place); slots: per line (y0, y1) in design px."""
    lines: list
    Wpx: int
    Hpx: int
    k: float
    track: float
    fa: list
    fb: list
    caps: list
    gap: float
    top: float
    centres: list
    slots: list
    _cache: dict = field(default_factory=dict, repr=False)

    def draw(self, which: str = "A") -> np.ndarray:
        """uint8 L raster (Hpx, Wpx) of the A or B faces (glyph by glyph at the A centres)."""
        if ("img", which) in self._cache:
            return self._cache[("img", which)]
        im = Image.new("L", (self.Wpx, self.Hpx), 0)
        d = ImageDraw.Draw(im)
        y0 = self.top
        fonts = self.fa if which == "A" else self.fb
        for (s, *_), f, sl, cs in zip(self.lines, fonts, self.caps, self.centres):
            cap = cap_height(f)
            hb = f.getbbox("H")
            for ch, c in zip(s, cs):
                bb = f.getbbox(ch)
                d.text((c - (bb[0] + bb[2]) / 2, y0 + (sl - cap) - hb[1]), ch, font=f, fill=255)
            y0 += sl + self.gap
        a = np.asarray(im)
        self._cache[("img", which)] = a
        return a

    def mask(self, which="A") -> np.ndarray:
        return self.draw(which) >= 128

    def sdf(self, which="A", precise=False, units="design") -> np.ndarray:
        """Signed distance, negative inside (units: 'design' px or 'px'). Cached per layout."""
        key = ("sdf", which, precise, units)
        if key not in self._cache:
            d = sdf(self.mask(which), precise=precise)
            self._cache[key] = (d / self.k if units == "design" else d).astype(np.float32)
        return self._cache[key]

    def line_mask(self, i: int) -> np.ndarray:
        """Design-space y band of line i as a boolean row mask (Hpx,)."""
        y = (np.arange(self.Hpx) + 0.5) / self.k
        return (y >= self.slots[i][0]) & (y <= self.slots[i][1])

    def outlines(self, which="A", ds: float = 2.0, smooth: float = 1.2, min_pts: int = 12):
        """Closed glyph outlines as (line index, (M, 2) design-px polylines), smoothed and resampled every ds
        design px, starting at the leftmost point; holes included (RETR_CCOMP)."""
        m = self.mask(which).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cs:
            p = c[:, 0, :].astype(np.float64) / self.k
            if len(p) < min_pts:
                continue
            li = int(np.argmin([abs(np.median(p[:, 1]) - 0.5 * (a + b)) for a, b in self.slots]))
            out.append((li, resample_closed(p, ds, smooth * self.k)))
        return out


def layout(lines, W: float, H: float, scale: float = 1.0, text_w: float = 600.0, cap_max: float = 150.0,
           track: float = 0.03, gap: float = 0.32, align: str = "justify", centre=None, min_size: int = 4,
           axes: dict | None = None):
    """Lay out line specs (text, face[, morph_face[, cap]]) on a W x H design canvas rendered at `scale` px per
    design px. align='justify': every line's ink spans text_w (cap height <= its cap or cap_max), the E1/E4
    look; 'centre': one size for all lines (the smallest fit), centred. gap = line gap x the smallest cap.
    axes: per-face variation overrides, e.g. {"Facade": {"CTST": 800}}."""
    axes = axes or {}
    specs = []
    for ln in lines:
        ln = (ln,) if isinstance(ln, str) else tuple(ln)
        s, face = ln[0], (ln[1] if len(ln) > 1 else "Le Murmure")
        fb = ln[2] if len(ln) > 2 and ln[2] else face
        cap = ln[3] if len(ln) > 3 and ln[3] else cap_max
        specs.append((s, face, fb, cap))
    k = scale

    def fit(name, s, cap):
        probe = load_font(name, 200, axes.get(name))
        return min(200 * text_w / line_width(probe, s, track), 200 * cap / cap_height(probe))

    zA = [fit(fn, s, cap) for s, fn, _, cap in specs]
    zB = [fit(fb, s, cap) for s, _, fb, cap in specs]
    if align == "centre":
        zA = [min(zA)] * len(zA)
        zB = [min(zB)] * len(zB)
    fa = [load_font(fn, max(min_size, int(z * k)), axes.get(fn)) for z, (_, fn, _, _) in zip(zA, specs)]
    fbs = [load_font(fb, max(min_size, int(z * k)), axes.get(fb)) for z, (_, _, fb, _) in zip(zB, specs)]
    caps = [max(cap_height(a), cap_height(b)) for a, b in zip(fa, fbs)]
    g = gap * min(caps)
    Wpx, Hpx = int(round(W * k)), int(round(H * k))
    cx, cy = (centre[0] * k, centre[1] * k) if centre else (Wpx / 2, Hpx / 2)
    centres = []
    for (s, *_), f in zip(specs, fa):
        x, cs = cx - line_width(f, s, track) / 2 - f.getbbox(s[0])[0], []
        for ch in s:
            bb = f.getbbox(ch)
            cs.append(x + (bb[0] + bb[2]) / 2)
            x += f.getlength(ch) + track * f.size
        centres.append(cs)
    top = cy - (sum(caps) + g * (len(specs) - 1)) / 2
    slots, y0 = [], top
    for sl in caps:
        slots.append(((y0 - 0.12 * sl) / k, (y0 + 1.12 * sl) / k))
        y0 += sl + g
    return Layout(specs, Wpx, Hpx, k, track, fa, fbs, caps, g, top, centres, slots)


# ------------------------------------------------------------------------------------------- fields
def sdf(mask: np.ndarray, precise: bool = True) -> np.ndarray:
    """Signed distance in px, negative inside, |d| = 0.5 at the pixel next to the edge (E1 convention).
    precise=False uses cv2's 5x5 mask (E4 / E5)."""
    inside = (np.asarray(mask) > 0).astype(np.uint8)
    mk = cv2.DIST_MASK_PRECISE if precise else 5
    din = cv2.distanceTransform(inside, cv2.DIST_L2, mk)
    dout = cv2.distanceTransform(1 - inside, cv2.DIST_L2, mk)
    return np.where(inside > 0, -(din - 0.5), dout - 0.5).astype(np.float32)


def sdf_morph(a, b, u):
    """Lerp between two SDFs (a face morph); u scalar or per-pixel field."""
    return a + (b - a) * u


def smin(a, b, r):
    """Polynomial smooth minimum (union with a fillet of radius r); numpy or torch."""
    import xp
    h = xp.clip(0.5 + 0.5 * (b - a) / r, 0.0, 1.0) if xp.is_t(a) else np.clip(0.5 + 0.5 * (b - a) / r, 0.0, 1.0)
    return b + (a - b) * h - r * h * (1 - h)


def resample_closed(p: np.ndarray, ds: float, sigma: float = 2.4) -> np.ndarray:
    """Smooth a closed polyline (periodic Gaussian, sigma in points) and resample it every ds by arc length,
    starting at the leftmost (slightly top-biased) point."""
    n = len(p)
    k = int(3 * sigma)
    if k >= 1:
        g = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
        g /= g.sum()
        pp = np.concatenate([p[-k:], p, p[:k]])
        p = np.stack([np.convolve(pp[:, i], g, mode="valid") for i in range(2)], 1)[:n]
    q = np.vstack([p, p[:1]])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    m = max(int(round(cum[-1] / ds)), 8)
    s = np.arange(m) * cum[-1] / m
    pts = np.stack([np.interp(s, cum, q[:, 0]), np.interp(s, cum, q[:, 1])], 1)
    start = np.argmin(pts[:, 0] + 0.08 * pts[:, 1])
    return np.roll(pts, -start, axis=0)


# ------------------------------------------------------------------------------------------- disk cache
def cached(cache_dir: str, key: dict, make):
    """np.load(cache_dir/<sha1(key)>.npy) or make() and save. key must be JSON-serialisable; include the
    font files' mtimes yourself if they may change."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    p = os.path.join(cache_dir, f"kit_{h}.npy")
    if os.path.exists(p):
        return np.load(p)
    a = make()
    np.save(p, a)
    return a
