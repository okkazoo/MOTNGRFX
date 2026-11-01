#!/usr/bin/env python
"""Rough animatic of "Assembly - a collage that builds itself" (_info/concept-collage.md).

Self-contained and deterministic (seeded). Renders 10 s @ 24 fps with audio, 9:16 and 16:9.

    python tools/animatic_assembly.py                 # both formats, full res
    python tools/animatic_assembly.py --draft         # short side 540, into review/
    python tools/animatic_assembly.py --fmt 9x16 --stills 1.5,2.1,4.5

Beat grid: 96 bpm (beat 0.625 s) phased so the drop at 2.0 s is a downbeat
(beats at 2.0 + k*0.625; downbeats 2.0 / 4.5 / 7.0 / 9.5). Arc boundaries are exact;
callouts, landings and burst swaps sit on beats / half-beats of that grid.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mograph.encode import encode_mp4  # noqa: E402
from mograph.mix import mix  # noqa: E402
from mograph.render.easing import clamp01, cubic_in_out, cubic_out, expo_out  # noqa: E402
from mograph.render.text import FONTS  # noqa: E402
from mograph.synth import render_sfx, render_soundtrack  # noqa: E402

SEED = 1729
FPS = 24
DUR = 10.0
NF = int(round(DUR * FPS))
SR = 48000
BPM = 96.0
BEAT = 60.0 / BPM          # 0.625
HB = BEAT / 2.0            # 0.3125
PHASE = 2.0                # the drop is a downbeat

OUT_DIR = os.path.join(ROOT, "01_assembly", "v001", "out")
REVIEW_DIR = os.path.join(ROOT, "01_assembly", "v001", "review")
FIX = os.path.join(ROOT, "tests", "fixtures")


def _hex(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


CREAM, INK, ORANGE, MUSTARD, TEAL = (_hex(c) for c in ("#e8dcc3", "#1d1b19", "#e2582a", "#dca232", "#3e8a86"))
DISC_COLS = (ORANGE, MUSTARD, TEAL)

# ----------------------------------------------------------------------------- timeline
T_DROP = 2.0
CALLOUT_T = (0.75, 1.375, 1.6875)                  # beats, last on the half-beat pickup
SWAPS = (5.75, 6.0625, 6.375, 6.6875)              # half-beats inside 5.5-7.0
SWAP_CONTENT = (1, 2, 3, 4)                        # subject window variant after each swap
T_REST = 7.0
T_REVEAL = 7.5
PULL_DUR = 0.75                                    # 7.5 -> 8.25 (a beat-grid point)
T_TITLE = 9.0


def snap(t):
    """Event time -> first frame index that shows it (half-beats round down by half a frame)."""
    return int(math.floor(t * FPS + 1e-6))


def ft(t):
    return snap(t) / FPS


# ----------------------------------------------------------------------------- layouts
# positions are fractions of W/H; sizes are px at a 1080 short side (scaled by u)
LAYOUTS = {
    "9x16": dict(
        size=(1080, 1920),
        disc=dict(c=(0.5, 0.33), r=430),
        subject=dict(c=(0.5, 0.605), h=1000, content=(0.55, 0.5)),
        pieces=[
            dict(name="mustard", kind="mustard", c=(0.5, 0.885), size=(700, 250), rot=1.5, land=3.25),
            dict(name="scrapB", kind="photo", img="texture", focus=(0.45, 0.42), zoom=1.6,
                 c=(0.22, 0.135), size=(300, 230), rot=5.0, land=4.1875),
            dict(name="subject"),
            dict(name="scrapA", kind="photo", img="stripes", focus=(0.5, 0.35), zoom=1.3,
                 c=(0.76, 0.165), size=(300, 380), rot=-4.0, land=2.625),
            dict(name="flower", kind="flower", c=(0.165, 0.665), size=(190, 190), rot=0.0, land=4.8125),
            dict(name="bar", kind="bar", c=(0.905, 0.60), size=(24, 560), rot=0.0, land=5.125),
        ],
        callouts=[
            dict(num="01.", label="figure", pos=(0.045, 0.43), target="subject_nose", t=CALLOUT_T[0]),
            dict(num="02.", label="plate", pos=(0.43, 0.025), target="scrapA_tl", t=CALLOUT_T[1]),
            dict(num="03.", label="flora", pos=(0.045, 0.745), target="flower_c", t=CALLOUT_T[2]),
        ],
        dim_x=0.965,
        reveal=dict(S=0.6, C=(0.5, 0.42), rot=-2.0, neighbours=[
            dict(kind="disc_piece", col="teal", c=(0.86, 0.085), size=(300, 300), rot=0.0,
                 land=8.25, dir=(1, -1), z=-1),
            dict(kind="mustard", c=(0.075, 0.55), size=(240, 520), rot=3.0, land=8.25, dir=(-1, 0), z=-1),
            dict(kind="photo", img="stripes", focus=(0.3, 0.3), zoom=1.0, c=(0.80, 0.80),
                 size=(320, 380), rot=6.0, land=8.5625, dir=(1, 0.2), z=1),
            dict(kind="photo", img="blob", focus=(0.6, 0.55), zoom=1.4, c=(0.24, 0.825),
                 size=(300, 260), rot=-5.0, land=8.875, dir=(0, 1), z=1),
        ], title=dict(c=(0.56, 0.94), size=64)),
    ),
    "16x9": dict(
        size=(1920, 1080),
        disc=dict(c=(0.5, 0.40), r=400),
        subject=dict(c=(0.5, 0.60), h=960, content=(0.55, 0.5)),
        pieces=[
            dict(name="mustard", kind="mustard", c=(0.74, 0.83), size=(520, 220), rot=1.5, land=3.25),
            dict(name="scrapB", kind="photo", img="texture", focus=(0.45, 0.42), zoom=1.6,
                 c=(0.255, 0.25), size=(320, 240), rot=5.0, land=4.1875),
            dict(name="subject"),
            dict(name="scrapA", kind="photo", img="stripes", focus=(0.5, 0.35), zoom=1.3,
                 c=(0.755, 0.27), size=(300, 370), rot=-4.0, land=2.625),
            dict(name="flower", kind="flower", c=(0.30, 0.70), size=(200, 200), rot=0.0, land=4.8125),
            dict(name="bar", kind="bar", c=(0.20, 0.90), size=(360, 22), rot=-2.0, land=5.125),
        ],
        callouts=[
            dict(num="01.", label="figure", pos=(0.07, 0.48), target="subject_nose", t=CALLOUT_T[0]),
            dict(num="02.", label="plate", pos=(0.88, 0.06), target="scrapA_tr", t=CALLOUT_T[1]),
            dict(num="03.", label="flora", pos=(0.09, 0.66), target="flower_c", t=CALLOUT_T[2]),
        ],
        dim_x=0.965,
        reveal=dict(S=0.56, C=(0.45, 0.45), rot=-1.5, neighbours=[
            dict(kind="disc_piece", col="teal", c=(0.86, 0.20), size=(300, 300), rot=0.0,
                 land=8.25, dir=(1, -0.3), z=-1),
            dict(kind="mustard", c=(0.085, 0.60), size=(300, 440), rot=3.0, land=8.25, dir=(-1, 0), z=-1),
            dict(kind="photo", img="stripes", focus=(0.3, 0.3), zoom=1.0, c=(0.84, 0.66),
                 size=(300, 360), rot=6.0, land=8.5625, dir=(1, 0.1), z=1),
            dict(kind="photo", img="blob", focus=(0.6, 0.55), zoom=1.4, c=(0.20, 0.86),
                 size=(320, 240), rot=-4.0, land=8.875, dir=(0, 1), z=1),
        ], title=dict(c=(0.64, 0.905), size=60)),
    ),
}

# profile head-and-shoulders, facing left; (x, y) in 0..1 of its bbox, open curve from
# bottom-left up the front, over the crown, down the back to bottom-right
PROFILE = [
    (0.03, 1.00), (0.04, 0.92), (0.07, 0.83), (0.15, 0.755), (0.265, 0.69), (0.305, 0.625),
    (0.295, 0.575), (0.235, 0.545), (0.19, 0.53), (0.162, 0.505), (0.168, 0.47), (0.152, 0.447),
    (0.163, 0.425), (0.145, 0.405), (0.15, 0.385), (0.112, 0.37), (0.078, 0.348), (0.10, 0.315),
    (0.135, 0.275), (0.15, 0.24), (0.145, 0.205), (0.165, 0.15), (0.225, 0.085), (0.33, 0.035),
    (0.50, 0.012), (0.67, 0.04), (0.79, 0.115), (0.855, 0.24), (0.85, 0.36), (0.805, 0.455),
    (0.735, 0.525), (0.715, 0.59), (0.745, 0.655), (0.86, 0.715), (0.945, 0.79), (0.978, 0.90),
    (0.99, 1.00),
]
PROFILE_ASPECT = 0.72   # w / h
NOSE_IDX_FRAC = (0.078, 0.348)


# ----------------------------------------------------------------------------- small helpers
def rot_pts(pts, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = np.asarray(pts, np.float64)
    return np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], 1)


def affine_M(anchor, pos, scale=1.0, deg=0.0):
    a = math.radians(deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    ax, ay = anchor
    return np.array([[c, -s, pos[0] - (c * ax - s * ay)],
                     [s, c, pos[1] - (s * ax + c * ay)]], np.float64)


def rasterize(polys, w, h, ss=2):
    """Anti-aliased fill of one or more polygons -> float32 (h, w) 0..1."""
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    pl = [np.round(np.asarray(p) * ss * 16).astype(np.int32) for p in polys]
    cv2.fillPoly(canvas, pl, 255, lineType=cv2.LINE_AA, shift=4)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def jitter_closed(pts, amp, rng, sigma=2.0):
    """Push a closed polygon along its normals by smooth noise (torn / scissor edge)."""
    pts = np.asarray(pts, np.float64)
    nxt = np.roll(pts, -1, 0)
    prv = np.roll(pts, 1, 0)
    tang = nxt - prv
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    n = rng.normal(size=len(pts))
    n = gaussian_filter1d(n, sigma, mode="wrap")
    n /= n.std() + 1e-9
    hf = gaussian_filter1d(rng.normal(size=len(pts)), 0.6, mode="wrap")
    off = amp * (0.6 * n + 0.35 * hf)
    return pts + nrm * off[:, None]


def rect_perimeter(w, h, step):
    out = []
    for (x0, y0), (x1, y1) in (((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(n):
            f = k / n
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    return np.array(out)


def circle_pts(cx, cy, r, n=None):
    n = n or max(48, int(2 * math.pi * r / 3))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def halftone(dark, cell, angle=45.0):
    """Dot screen coverage (0..1) for a darkness map; analytic AA at 1 px."""
    h, w = dark.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    uu = (xx * c + yy * s) / cell
    vv = (-xx * s + yy * c) / cell
    fu = uu - np.floor(uu) - 0.5
    fv = vv - np.floor(vv) - 0.5
    d = np.sqrt(fu * fu + fv * fv) * cell
    r = cell * 0.74 * np.sqrt(np.clip(dark, 0.0, 1.0))
    return np.clip(r - d + 0.5, 0.0, 1.0).astype(np.float32)


def cover(img, w, h, focus=(0.5, 0.5), zoom=1.0):
    H0, W0 = img.shape[:2]
    sc = max(w / W0, h / H0) * zoom
    cw, ch = w / sc, h / sc
    x0 = float(np.clip(focus[0] * W0 - cw / 2, 0, W0 - cw))
    y0 = float(np.clip(focus[1] * H0 - ch / 2, 0, H0 - ch))
    crop = img[int(y0):int(math.ceil(y0 + ch)), int(x0):int(math.ceil(x0 + cw))]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)


def darkness(rgb):
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
    n = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
    return (0.06 + 0.86 * (1.0 - n)).astype(np.float32)


def paper_noise(h, w, rng, amp=0.035):
    n = rng.normal(size=(max(2, h // 3), max(2, w // 3))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    lo = cv2.resize(rng.normal(size=(max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    return 1.0 + amp * (0.45 * n + 0.32 * lo)


def text_alpha(text, font, size, tracking=0.0):
    ss = 3
    f = ImageFont.truetype(FONTS[font], max(4, int(round(size * ss))))
    l, t, r, b = f.getbbox(text)
    extra = int(abs(tracking) * size * ss * len(text))
    im = Image.new("L", (r - l + 8 * ss + extra, b - t + 8 * ss), 0)
    d = ImageDraw.Draw(im)
    if tracking:
        x = 4 * ss - l
        for ch in text:
            d.text((x, 4 * ss - t), ch, font=f, fill=255)
            x += f.getlength(ch) + tracking * size * ss
    else:
        d.text((4 * ss - l, 4 * ss - t), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255.0
    return cv2.resize(a, (max(1, a.shape[1] // ss), max(1, a.shape[0] // ss)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- sprites
class Sprite:
    """Premultiplied RGBA float32 + blurred shadow alpha. anchor = piece centre in sprite px."""

    def __init__(self, rgb, alpha, pad, u, shadow_sigma=2.2):
        self.alpha = alpha.astype(np.float32)
        self.variants = [self._pack(rgb)]
        self.pad = pad
        h, w = alpha.shape
        self.anchor = (w / 2.0, h / 2.0)
        self.shadow = cv2.GaussianBlur(self.alpha, (0, 0), max(0.8, shadow_sigma * u))

    def _pack(self, rgb):
        return np.dstack([rgb * self.alpha[..., None], self.alpha]).astype(np.float32)

    def add_variant(self, rgb):
        self.variants.append(self._pack(rgb))


def _warp_region(dst_shape, src, M):
    h, w = src.shape[:2]
    cs = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    H, W = dst_shape[:2]
    x0 = max(0, int(math.floor(cs[:, 0].min())) - 1)
    y0 = max(0, int(math.floor(cs[:, 1].min())) - 1)
    x1 = min(W, int(math.ceil(cs[:, 0].max())) + 1)
    y1 = min(H, int(math.ceil(cs[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    out = cv2.warpAffine(src, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, (x0, y0, x1, y1)


def blit(dst, rgba, M, opacity=1.0):
    r = _warp_region(dst.shape, rgba, M)
    if r is None:
        return
    w, (x0, y0, x1, y1) = r
    if w.ndim == 2:
        return
    if opacity < 1.0:
        w = w * opacity
    reg = dst[y0:y1, x0:x1]
    reg *= (1.0 - w[..., 3:4])
    reg += w[..., :3]


def blit_shadow(dst, sa, M, opacity):
    r = _warp_region(dst.shape, sa, M)
    if r is None:
        return
    w, (x0, y0, x1, y1) = r
    a = (w * opacity)[..., None]
    reg = dst[y0:y1, x0:x1]
    reg *= (1.0 - a)
    reg += INK * a


def blend_mask(dst, mask_u8, color):
    ys, xs = np.nonzero(mask_u8[::8, ::8])
    if len(ys) == 0:
        return
    y0, y1 = max(0, ys.min() * 8 - 8), min(dst.shape[0], ys.max() * 8 + 16)
    x0, x1 = max(0, xs.min() * 8 - 8), min(dst.shape[1], xs.max() * 8 + 16)
    a = mask_u8[y0:y1, x0:x1].astype(np.float32)[..., None] * (1.0 / 255.0)
    reg = dst[y0:y1, x0:x1]
    reg *= (1.0 - a)
    reg += color * a


# ----------------------------------------------------------------------------- line guides
class Guide:
    """A polyline that draws on over time. style: solid | dashed | dotted."""

    def __init__(self, pts, t0, dur, style="solid", thick=1, alpha=0.85, closed=False, hide_at=None,
                 dash=(14.0, 8.0), dot_gap=8.0, dot_r=1.4, ease=cubic_out):
        p = np.asarray(pts, np.float64)
        if closed:
            p = np.vstack([p, p[:1]])
        self.p = p
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(self.cum[-1])
        self.t0, self.dur, self.style = t0, dur, style
        self.thick, self.val, self.hide_at = thick, int(round(255 * alpha)), hide_at
        self.ease = ease
        self.dot_r = dot_r
        if style == "dashed":
            on, off = dash
            starts = np.arange(0.0, self.L, on + off)
            self.dashes = [(s, min(s + on, self.L)) for s in starts]
            self.dash_pts = [self._sub(a, b) for a, b in self.dashes]
        if style == "dotted":
            self.dots = np.arange(0.0, self.L + 1e-6, dot_gap)
            self.dot_xy = self._at(self.dots)

    def _at(self, s):
        s = np.clip(np.atleast_1d(s), 0, self.L)
        x = np.interp(s, self.cum, self.p[:, 0])
        y = np.interp(s, self.cum, self.p[:, 1])
        return np.stack([x, y], 1)

    def _sub(self, a, b):
        inner = self.p[(self.cum > a) & (self.cum < b)]
        return np.vstack([self._at(a), inner, self._at(b)])

    def progress(self, t):
        if t < self.t0 or (self.hide_at is not None and t >= self.hide_at - 1e-9):
            return 0.0
        return self.ease(clamp01((t - self.t0) / max(self.dur, 1e-6)))

    def draw(self, mask, t):
        p = self.progress(t)
        if p <= 0.0:
            return
        Lp = self.L * p
        S = 16.0
        if self.style == "solid":
            pts = self._sub(0.0, Lp)
            cv2.polylines(mask, [np.round(pts * S).astype(np.int32)], False, self.val, self.thick,
                          cv2.LINE_AA, 4)
        elif self.style == "dashed":
            pl = [np.round(d * S).astype(np.int32) for (a, _), d in zip(self.dashes, self.dash_pts) if a < Lp]
            if pl:
                cv2.polylines(mask, pl, False, self.val, self.thick, cv2.LINE_AA, 4)
        else:
            n = int(np.searchsorted(self.dots, Lp, side="right"))
            r = int(round(self.dot_r * S))
            for x, y in self.dot_xy[:n]:
                cv2.circle(mask, (int(round(x * S)), int(round(y * S))), r, self.val, -1, cv2.LINE_AA, 4)


# ----------------------------------------------------------------------------- renderer
class Animatic:
    def __init__(self, fmt, draft=False):
        L = LAYOUTS[fmt]
        bw, bh = L["size"]
        k = 0.5 if draft else 1.0
        self.W, self.H = int(round(bw * k)), int(round(bh * k))
        self.u = u = min(self.W, self.H) / 1080.0
        self.fmt, self.L = fmt, L
        W, H = self.W, self.H
        self.rng = np.random.default_rng(SEED)
        self.photos = {n: np.asarray(Image.open(os.path.join(FIX, f + ".png")).convert("RGB"), np.float32) / 255.0
                       for n, f in (("blob", "blob_gradient"), ("stripes", "stripes"), ("texture", "subject_texture"))}
        self.cell = max(4.0, 9.0 * u)
        self.pad = int(math.ceil(14 * u))
        self.sh_off = np.array([3.0 * u, 3.5 * u])

        self.ground = self._paper(W, H, np.random.default_rng([SEED, 1]), vign=0.20)
        self._build_disc()
        self._build_subject()
        self._build_pieces()
        self._build_guides()
        self._build_callouts()
        self._build_reveal()
        g = np.random.default_rng([SEED, 9])
        self.grain = [cv2.GaussianBlur(g.normal(size=(H, W)).astype(np.float32), (0, 0), 0.6 * max(u, 0.5))
                      [..., None] * np.float32(0.028) for _ in range(6)]
        self.grain_order = g.integers(0, 6, NF)

    # ---- paper
    def _paper(self, W, H, rng, vign=0.18):
        u = self.u
        img = np.empty((H, W, 3), np.float32)
        img[:] = CREAM
        img *= paper_noise(H, W, rng, 0.03)[..., None]
        dark = np.zeros((H, W), np.uint8)
        light = np.zeros((H, W), np.uint8)
        nf = int(W * H / 3500)
        for i in range(nf):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            a = rng.uniform(0, math.pi)
            ln = rng.uniform(6, 34) * u
            bend = rng.normal(0, 0.35)
            pts = []
            for k in range(5):
                f = k / 4
                aa = a + bend * f
                pts.append((x + math.cos(aa) * ln * f, y + math.sin(aa) * ln * f))
            target = dark if i % 3 else light
            cv2.polylines(target, [np.round(np.array(pts) * 16).astype(np.int32)], False,
                          int(rng.uniform(50, 140)), 1, cv2.LINE_AA, 4)
        img *= (1.0 - 0.07 * dark.astype(np.float32) / 255.0)[..., None]
        img += (0.05 * light.astype(np.float32) / 255.0)[..., None]
        speck = np.zeros((H, W), np.uint8)
        for _ in range(int(W * H / 7000)):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            r = rng.uniform(0.4, 1.5) * u
            cv2.circle(speck, (int(x * 16), int(y * 16)), max(4, int(r * 16)), int(rng.uniform(60, 170)), -1,
                       cv2.LINE_AA, 4)
        sa = speck.astype(np.float32)[..., None] / 255.0
        img = img * (1 - 0.6 * sa) + INK * 0.6 * sa
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        v = 1.0 - vign * np.clip(r, 0, 1) ** 2.2
        img *= v[..., None]
        return np.clip(img, 0, 1).astype(np.float32)

    # ---- disc anchor: colour plate + halftone ink shading plate misregistered by 2 px
    def _build_disc(self):
        u, W, H = self.u, self.W, self.H
        d = self.L["disc"]
        cx, cy, r = d["c"][0] * W, d["c"][1] * H, d["r"] * u
        self.disc_c, self.disc_r = (cx, cy), r
        rng = np.random.default_rng([SEED, 2])
        poly = jitter_closed(circle_pts(cx, cy, r), 0.8 * u, rng, sigma=4)
        m = rasterize([poly], W, H)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dd = np.sqrt((xx - (cx - 0.45 * r)) ** 2 + (yy - (cy - 0.45 * r)) ** 2) / (1.55 * r)
        shade = np.clip((dd - 0.62) / 0.4, 0, 1) * 0.42
        ink_plate = halftone(shade, self.cell * 0.9, 15.0) * shift(m, 2.0 * u, 2.0 * u)
        self.ground_disc = []
        tex = paper_noise(H, W, rng, 0.04)[..., None]
        for col in DISC_COLS:
            img = self.ground * (1 - m[..., None]) + (col * tex) * m[..., None]
            img = img * (1 - ink_plate[..., None] * 0.9) + INK * (ink_plate[..., None] * 0.9)
            self.ground_disc.append(np.clip(img, 0, 1).astype(np.float32))

    # ---- subject: profile silhouette window
    def _profile_poly(self, w, h, off=(0.0, 0.0)):
        p = np.array(PROFILE, np.float64)
        tck, _ = splprep([p[:, 0], p[:, 1]], s=0.0002, k=3)
        x, y = splev(np.linspace(0, 1, 900), tck)
        x, y = np.clip(x, 0, 1), np.clip(y, 0, 1)
        y[-1] = y[0] = 1.0
        pts = np.stack([x * w + off[0], y * h + off[1]], 1)
        return pts

    def _build_subject(self):
        u, W, H = self.u, self.W, self.H
        s = self.L["subject"]
        h = int(round(s["h"] * u))
        w = int(round(h * PROFILE_ASPECT))
        pad = self.pad
        rng = np.random.default_rng([SEED, 3])
        poly = self._profile_poly(w, h, (pad, pad))
        cut = jitter_closed(poly, 0.9 * u, rng, sigma=3)
        sw, sh = w + 2 * pad, h + 2 * pad
        m = rasterize([cut], sw, sh)
        pos = (s["c"][0] * W, s["c"][1] * H)
        self.subject_pos = pos
        self.subject_outline = poly - np.array([sw / 2, sh / 2]) + np.array(pos)
        nose = np.array([NOSE_IDX_FRAC[0] * w + pad, NOSE_IDX_FRAC[1] * h + pad]) - np.array([sw / 2, sh / 2])
        self.anchors = {"subject_nose": tuple(nose + np.array(pos) + np.array([-4 * u, 0]))}
        tex = paper_noise(sh, sw, rng, 0.035)[..., None]
        cell = max(4.0, 10.0 * u)

        def ht(name, focus, zoom=1.0, invert=False):
            dk = darkness(cover(self.photos[name], sw, sh, focus, zoom))
            dk = cv2.GaussianBlur(dk, (0, 0), cell * 0.35)
            if invert:  # tonal negative: cream dots on ink, dark areas print light
                c = halftone(dk, cell)
                return (INK * (1 - c[..., None]) + CREAM * c[..., None]) * tex
            c = halftone(dk, cell)
            return (CREAM * (1 - c[..., None]) + INK * c[..., None]) * tex

        v0 = ht("blob", s["content"])
        self.subject = Sprite(v0, m, pad, u)
        self.subject.add_variant(ht("stripes", (0.5, 0.4)))                          # 1
        # flat field misregistration: colour plate offset against the cut
        mm = shift(m, 2.0 * u, -2.0 * u)
        flat = CREAM * tex * (1 - mm[..., None]) + ORANGE * tex * mm[..., None]
        self.subject.add_variant(flat)                                              # 2
        self.subject.add_variant(ht("blob", s["content"], invert=True))             # 3
        self.subject.add_variant(ht("texture", (0.47, 0.56), 1.15))                # 4

    # ---- secondary pieces
    def _make_piece(self, spec, key):
        u = self.u
        rng = np.random.default_rng([SEED, 100 + key])
        pad = self.pad
        kind = spec["kind"]
        w, h = int(round(spec["size"][0] * u)), int(round(spec["size"][1] * u))
        sw, sh = w + 2 * pad, h + 2 * pad
        tex = paper_noise(sh, sw, rng, 0.04)[..., None]
        if kind in ("photo", "mustard", "bar"):
            amp = (1.6 if kind != "bar" else 0.7) * u
            per = rect_perimeter(w, h, 4.0 * u) + pad
            m = rasterize([jitter_closed(per, amp, rng, sigma=2.5)], sw, sh)
        if kind == "photo":
            dk = darkness(cover(self.photos[spec["img"]], sw, sh, spec.get("focus", (0.5, 0.5)), spec.get("zoom", 1.0)))
            dk = cv2.GaussianBlur(dk, (0, 0), self.cell * 0.35)
            c = halftone(dk, self.cell)[..., None]
            rgb = (CREAM * (1 - c) + INK * c) * tex
        elif kind in ("mustard", "bar"):
            col = MUSTARD if kind == "mustard" else ORANGE
            plate = np.zeros((sh, sw), np.float32)
            plate[pad:pad + h, pad:pad + w] = 1.0
            plate = cv2.GaussianBlur(plate, (0, 0), 0.6)
            plate = shift(plate, 2.0 * u, -1.5 * u) if kind == "mustard" else shift(plate, 1.5 * u, 2.0 * u)
            rgb = CREAM * (1 - plate[..., None]) + col * plate[..., None]
            if kind == "mustard":
                yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
                band = np.clip((xx - pad - 0.55 * w) / (0.45 * w), 0, 1) * 0.5
                ink = halftone(band, self.cell * 0.9, 45.0) * shift(plate, -2.0 * u, 1.0 * u)
                rgb = rgb * (1 - ink[..., None]) + INK * ink[..., None]
            rgb = rgb * tex
        elif kind == "flower":
            R = w / 2.0
            cx, cy = sw / 2.0, sh / 2.0
            petals = []
            for k in range(6):
                a = k * math.pi / 3 + 0.2
                petals.append(circle_pts(cx + 0.56 * R * math.cos(a), cy + 0.56 * R * math.sin(a), 0.42 * R))
            centre = circle_pts(cx, cy, 0.34 * R)
            m = rasterize([jitter_closed(p, 0.6 * u, rng) for p in petals + [centre]], sw, sh)
            pet_plate = shift(rasterize(petals, sw, sh), 2.0 * u, 1.5 * u)
            cen_plate = shift(rasterize([centre], sw, sh), -1.5 * u, 2.0 * u)
            dot = rasterize([circle_pts(cx + 1.5 * u, cy - 1.0 * u, 0.12 * R)], sw, sh)
            rgb = CREAM * np.ones((sh, sw, 3), np.float32)
            rgb = rgb * (1 - pet_plate[..., None]) + ORANGE * pet_plate[..., None]
            rgb = rgb * (1 - cen_plate[..., None]) + MUSTARD * cen_plate[..., None]
            rgb = rgb * (1 - dot[..., None]) + INK * dot[..., None]
            rgb = rgb * tex
        elif kind == "disc_piece":
            R = w / 2.0
            cx, cy = sw / 2.0, sh / 2.0
            m = rasterize([jitter_closed(circle_pts(cx, cy, R), 1.2 * u, rng)], sw, sh)
            plate = shift(rasterize([circle_pts(cx, cy, R - 1.5 * u)], sw, sh), 2.0 * u, -1.5 * u)
            col = {"teal": TEAL, "orange": ORANGE, "mustard": MUSTARD}[spec.get("col", "teal")]
            rgb = (CREAM * (1 - plate[..., None]) + col * plate[..., None]) * tex
        return Sprite(np.clip(rgb, 0, 1), m, pad, u), (w, h)

    def _build_pieces(self):
        W, H, u = self.W, self.H, self.u
        self.pieces = []
        drng = np.random.default_rng([SEED, 4])
        for i, spec in enumerate(self.L["pieces"]):
            d = dict(name=spec["name"],
                     drift=(drng.uniform(1.2, 2.2) * u, drng.uniform(1.2, 2.2) * u, drng.uniform(0.15, 0.32),
                            drng.uniform(0.12, 0.27), drng.uniform(0, 6.28), drng.uniform(0, 6.28),
                            drng.uniform(0.15, 0.3), drng.uniform(0.1, 0.2)))
            if spec["name"] == "subject":
                d.update(sprite=self.subject, pos=self.subject_pos, rot=0.0, land=T_DROP)
                d["drift"] = (0.8 * u, 1.0 * u) + d["drift"][2:6] + (0.12, 0.09)
            else:
                spr, (w, h) = self._make_piece(spec, i)
                pos = (spec["c"][0] * W, spec["c"][1] * H)
                d.update(sprite=spr, pos=pos, rot=spec["rot"], land=spec["land"], kind=spec["kind"], wh=(w, h))
                corners = rot_pts([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)], spec["rot"]) + pos
                d["corners"] = corners
                if spec["kind"] == "flower":
                    self.anchors["flower_c"] = (pos[0] - 0.5 * w - 6 * u, pos[1] + 0.05 * h)
                if spec["name"] == "scrapA":
                    self.anchors["scrapA_tl"] = tuple(corners[0] + np.array([-5 * u, -5 * u]))
                    self.anchors["scrapA_tr"] = tuple(corners[1] + np.array([5 * u, -5 * u]))
            self.pieces.append(d)

    def _build_guides(self):
        W, H, u = self.W, self.H, self.u
        cx, cy = self.disc_c
        r = self.disc_r
        G = []
        dash = (16 * u, 9 * u)
        # construction
        G.append(Guide([(cx, 0.015 * H), (cx, 0.985 * H)], 0.0, 1.7, "dashed", 1, 0.55, dash=(40 * u, 7 * u)))
        G.append(Guide([(0.03 * W, cy), (0.97 * W, cy)], 0.1, 1.5, "solid", 1, 0.5))
        top = self.subject_outline[:, 1].min()
        G.append(Guide([(0.97 * W, top), (0.03 * W, top)], 0.35, 1.3, "solid", 1, 0.35))
        G.append(Guide(circle_pts(cx, cy, r + 16 * u)[::-1], 0.0, 1.9, "dashed", 1, 0.6, closed=True, dash=dash))
        G.append(Guide(circle_pts(cx, cy, r * 0.5, 120)[60:110], 0.45, 1.2, "solid", 1, 0.45))
        c = 18 * u
        G.append(Guide([(cx - c, cy), (cx + c, cy)], 0.2, 0.4, "solid", 2, 0.85))
        G.append(Guide([(cx, cy - c), (cx, cy + c)], 0.25, 0.4, "solid", 2, 0.85))
        # dimension line with ticks
        dx = self.L["dim_x"] * W
        bot = self.subject_outline[:, 1].max()
        G.append(Guide([(dx, cy - r), (dx, bot)], 0.5, 1.3, "solid", 1, 0.7))
        for yy in (cy - r, cy, bot):
            G.append(Guide([(dx - 9 * u, yy), (dx + 9 * u, yy)], 0.5, 0.3, "solid", 1, 0.8))
        # landing outlines (dashed) - vanish on the hit
        G.append(Guide(self.subject_outline, 0.25, 1.6, "dashed", 1, 0.8, hide_at=ft(T_DROP), dash=dash))
        starts = {"scrapA": 0.45, "mustard": 0.6, "scrapB": 0.7, "flower": 0.8, "bar": 0.9}
        for p in self.pieces:
            if p["name"] == "subject":
                continue
            t0 = starts[p["name"]]
            if p["kind"] == "flower":
                pts = circle_pts(p["pos"][0], p["pos"][1], p["wh"][0] / 2 + 4 * u)
                G.append(Guide(pts, t0, 1.1, "dashed", 1, 0.8, closed=True, hide_at=ft(p["land"]), dash=dash))
            else:
                G.append(Guide(p["corners"], t0, 1.1, "dashed", 1, 0.8, closed=True, hide_at=ft(p["land"]), dash=dash))
        self.guides = G

    def _build_callouts(self):
        W, H, u = self.W, self.H, self.u
        self.callouts = []
        for c in self.L["callouts"]:
            na = text_alpha(c["num"], "Space Grotesk", 34 * u)
            la = text_alpha(c["label"], "Space Grotesk", 19 * u)
            w = max(na.shape[1], la.shape[1])
            img = np.zeros((na.shape[0] + la.shape[0], w), np.float32)
            img[:na.shape[0], :na.shape[1]] = na
            img[na.shape[0]:, :la.shape[1]] = la
            rgba = np.dstack([np.broadcast_to(INK, img.shape + (3,)) * img[..., None], img]).astype(np.float32)
            x, y = c["pos"][0] * W, c["pos"][1] * H
            tgt = self.anchors[c["target"]]
            ymid = y + na.shape[0] * 0.5
            if tgt[0] > x + w / 2:
                a = (x + na.shape[1] + 4 * u, ymid)
                elbow = (a[0] + 28 * u, ymid)
            else:
                a = (x - 4 * u, ymid)
                elbow = (a[0] - 28 * u, ymid)
            t0 = ft(c["t"])
            dur = min(1.0, T_DROP - t0) if c["t"] < T_DROP else 0.6
            lead = Guide([a, elbow, tgt], t0, max(dur, 0.25), "dotted", 1, 0.95, dot_gap=7 * u, dot_r=1.5 * u)
            self.callouts.append(dict(rgba=rgba, xy=(x, y), t=t0, lead=lead, a=a, tgt=tgt))

    def _build_reveal(self):
        W, H, u = self.W, self.H, self.u
        R = self.L["reveal"]
        self.outer = self._paper(W, H, np.random.default_rng([SEED, 7]), vign=0.24)
        # print-sheet furniture: crop marks, registration targets, colour bar
        m = np.zeros((H, W), np.uint8)
        S = 16
        inset, ln = 34 * u, 38 * u

        def line(p, q, th=1):
            cv2.line(m, (int(p[0] * S), int(p[1] * S)), (int(q[0] * S), int(q[1] * S)), 200, th, cv2.LINE_AA, 4)

        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x = inset if sx == 0 else W - inset
            y = inset if sy == 0 else H - inset
            dx = ln if sx == 0 else -ln
            dy = ln if sy == 0 else -ln
            line((x, y - (8 * u if sy == 0 else -8 * u)), (x, y + dy))
            line((x - (8 * u if sx == 0 else -8 * u), y), (x + dx, y))
        for (rx, ry) in ((W - inset - 60 * u, inset + 22 * u), (inset + 60 * u, H - inset - 22 * u)):
            cv2.circle(m, (int(rx * S), int(ry * S)), int(11 * u * S), 200, 1, cv2.LINE_AA, 4)
            line((rx - 18 * u, ry), (rx + 18 * u, ry))
            line((rx, ry - 18 * u), (rx, ry + 18 * u))
        self.outer = self.outer * (1 - m[..., None] / 255.0 * 0.8) + INK * (m[..., None] / 255.0 * 0.8)
        sq = int(round(16 * u))
        x0, y0 = int(inset + 110 * u), int(H - inset - sq)
        for k, col in enumerate((INK, ORANGE, MUSTARD, TEAL, CREAM)):
            xs = x0 + k * (sq + int(3 * u))
            self.outer[y0:y0 + sq, xs:xs + sq] = col * 0.97
        cv2.rectangle(m, (0, 0), (1, 1), 0)
        self.outer = np.clip(self.outer, 0, 1).astype(np.float32)
        # inner collage as a torn paper piece
        rng = np.random.default_rng([SEED, 8])
        per = rect_perimeter(W, H, 5.0 * u)
        per = per + (np.array([W / 2, H / 2]) - per) * (3.0 * u / np.array([W / 2, H / 2]))
        self.inner_mask = rasterize([jitter_closed(per, 3.0 * u, rng, sigma=2.5)], W, H)
        self.rv = R
        self.neigh = []
        for i, spec in enumerate(R["neighbours"]):
            spr, wh = self._make_piece(spec, 50 + i)
            self.neigh.append(dict(sprite=spr, pos=(spec["c"][0] * W, spec["c"][1] * H), rot=spec["rot"],
                                   land=ft(spec["land"]), dir=np.array(spec["dir"], float), z=spec["z"]))
        # end label: paper slip with the title + orange bar beside it
        ta = text_alpha("ASSEMBLY", "Big Shoulders Display", R["title"]["size"] * u, tracking=0.06)
        th, tw = ta.shape
        padx, pady = int(16 * u), int(10 * u)
        w, h = tw + 2 * padx, th + 2 * pady
        pad = self.pad
        sw, sh = w + 2 * pad, h + 2 * pad
        per = rect_perimeter(w, h, 4 * u) + pad
        sm = rasterize([jitter_closed(per, 1.0 * u, rng)], sw, sh)
        rgb = np.ones((sh, sw, 3), np.float32) * CREAM * paper_noise(sh, sw, rng, 0.03)[..., None]
        tt = np.zeros((sh, sw), np.float32)
        tt[pad + pady:pad + pady + th, pad + padx:pad + padx + tw] = ta
        rgb = rgb * (1 - tt[..., None]) + INK * tt[..., None]
        self.title = Sprite(rgb, sm, pad, u)
        self.title_pos = (R["title"]["c"][0] * W, R["title"]["c"][1] * H)
        bw, bh = int(round(110 * u)), int(round(max(6, 0.32 * th)))
        bs = self._make_piece(dict(kind="bar", size=(bw / u, bh / u)), 99)[0]
        self.title_bar = bs
        self.title_bar_pos = (self.title_pos[0] - w / 2 - bw / 2 - 14 * u, self.title_pos[1] + 0.12 * th)

    # ---- animation helpers
    def _drift(self, d, t):
        ax, ay, fx, fy, px, py, ra, fr = d
        return (ax * math.sin(2 * math.pi * fx * t + px), ay * math.sin(2 * math.pi * fy * t + py),
                ra * math.sin(2 * math.pi * fr * t + px + py))

    def _land(self, t, tl):
        """-> (scale, lift 0..1, opacity) or None. One anticipation frame, hit at 1.08, settle over a beat."""
        f, fl = int(round(t * FPS)), snap(tl)
        if f < fl - 1:
            return None
        if f == fl - 1:
            return 1.2, 1.0, 0.55
        k = expo_out((t - fl / FPS) / BEAT)
        return 1.08 - 0.08 * k, 1.0 - k, 1.0

    def _draw_sprite(self, img, spr, variant, pos, scale, deg, lift, opacity):
        u = self.u
        off = self.sh_off * (1.0 + 3.0 * lift)
        Ms = affine_M(spr.anchor, (pos[0] + off[0], pos[1] + off[1]), scale, deg)
        blit_shadow(img, spr.shadow, Ms, 0.34 * (1.0 - 0.35 * lift) * opacity)
        blit(img, spr.variants[variant], affine_M(spr.anchor, pos, scale, deg), opacity)

    def subject_variant(self, t):
        v = 0
        for ts, c in zip(SWAPS, SWAP_CONTENT):
            if t >= ft(ts) - 1e-9:
                v = c
        return v

    # ---- frame
    def inner(self, t):
        if t < T_DROP:
            f = cubic_out(clamp01(t / 0.9))
            img = self.ground + (self.ground_disc[0] - self.ground) * f
        else:
            img = self.ground_disc[1 if t < T_REVEAL else 2].copy()
        mask = np.zeros((self.H, self.W), np.uint8)
        for g in self.guides:
            g.draw(mask, t)
        blend_mask(img, mask, INK)
        for p in self.pieces:
            st = self._land(t, p["land"])
            if st is None:
                continue
            sc, lift, op = st
            dx, dy, dr = self._drift(p["drift"], t)
            var = self.subject_variant(t) if p["name"] == "subject" else 0
            self._draw_sprite(img, p["sprite"], var, (p["pos"][0] + dx, p["pos"][1] + dy), sc,
                              p["rot"] + dr, lift, op)
        mask[:] = 0
        for c in self.callouts:
            c["lead"].draw(mask, t)
            if t >= c["t"] - 1e-9 and c["lead"].progress(t) > 0.97:
                cv2.circle(mask, tuple(int(round(v * 16)) for v in c["tgt"]), int(6 * self.u * 16), 240, 1,
                           cv2.LINE_AA, 4)
        blend_mask(img, mask, INK)
        for c in self.callouts:
            if t >= c["t"] - 1e-9:
                k = clamp01((t - c["t"]) / 0.12)
                sc = 1.18 - 0.18 * cubic_out(k)
                h, w = c["rgba"].shape[:2]
                blit(img, c["rgba"], affine_M((0, h / 2), (c["xy"][0], c["xy"][1] + h / 2), sc, 0.0))
        return img

    def reveal(self, inner, t):
        W, H, u = self.W, self.H, self.u
        R = self.rv
        p = cubic_in_out(clamp01((t - T_REVEAL) / PULL_DUR))
        S = 1.0 + (R["S"] - 1.0) * p
        C = (W / 2 + (R["C"][0] * W - W / 2) * p, H / 2 + (R["C"][1] * H - H / 2) * p)
        deg = R["rot"] * p
        out = self.outer.copy()

        def neighbours(z):
            for n in self.neigh:
                if n["z"] != z:
                    continue
                d0 = n["land"] - 0.45
                if t < d0 - 1e-9:
                    continue
                k = expo_out((t - d0) / 0.45)
                dist = 0.75 * max(W, H) * (1.0 - k)
                dvec = n["dir"] / np.linalg.norm(n["dir"])
                pos = (n["pos"][0] + dvec[0] * dist, n["pos"][1] + dvec[1] * dist)
                self._draw_sprite(out, n["sprite"], 0, pos, 1.0, n["rot"] + 8.0 * (1 - k), 0.6 * (1 - k), 1.0)

        neighbours(-1)
        sw, sh = max(2, int(round(W * S))), max(2, int(round(H * S)))
        if S < 0.999:
            small = cv2.resize(inner, (sw, sh), interpolation=cv2.INTER_AREA)
            m = cv2.resize(self.inner_mask, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small, m, sw, sh = inner, self.inner_mask, W, H
        P = self.pad * 2
        rgba = np.zeros((sh + 2 * P, sw + 2 * P, 4), np.float32)
        rgba[P:P + sh, P:P + sw, :3] = small * m[..., None]
        rgba[P:P + sh, P:P + sw, 3] = m
        sa = cv2.GaussianBlur(rgba[..., 3], (0, 0), 3.0 * u)
        anchor = (rgba.shape[1] / 2, rgba.shape[0] / 2)
        off = np.array([5.0 * u, 6.0 * u])
        blit_shadow(out, sa, affine_M(anchor, (C[0] + off[0], C[1] + off[1]), 1.0, deg), 0.42)
        blit(out, rgba, affine_M(anchor, C, 1.0, deg))
        neighbours(1)
        if t >= ft(T_TITLE) - 1e-9:
            self._draw_sprite(out, self.title_bar, 0, self.title_bar_pos, 1.0, 0.0, 0.0, 1.0)
            self._draw_sprite(out, self.title, 0, self.title_pos, 1.0, -1.0, 0.0, 1.0)
        return out

    def frame(self, i):
        t = i / FPS
        img = self.inner(t)
        if t >= T_REVEAL - 1e-9:
            img = self.reveal(img, t)
        fd = snap(T_DROP)
        if fd <= i <= fd + 2:          # 2-frame flash + slam shake/settle
            k = i - fd
            sy = (6.0, -2.5, 1.0)[k] * self.u
            img = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, sy]]), (self.W, self.H),
                                 borderMode=cv2.BORDER_REFLECT)
            fl = (0.38, 0.14, 0.0)[k]
            if fl:
                img = img * (1 - fl) + CREAM * fl
        img = img + self.grain[self.grain_order[i]]
        return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- audio
def build_plan():
    beats = [round(PHASE + k * BEAT, 6) for k in range(-3, 15)]          # 0.125 ... 10.625
    downbeats = [2.0, 4.5, 7.0, 9.5]
    sfx = [dict(type="click", t=t, gain=0.55) for t in CALLOUT_T]
    sfx.append(dict(type="impact", t=T_DROP, gain=0.95))
    for k, t in enumerate((2.625, 3.25, 4.1875, 4.8125, 5.125)):
        sfx.append(dict(type="pop" if k % 2 == 0 else "hit", t=t, gain=0.6))
    for k, t in enumerate(SWAPS):
        sfx.append(dict(type="glitch" if k % 2 == 0 else "pop", t=t, gain=0.28))
    sfx.append(dict(type="swell", t=T_REVEAL, dur=0.5, gain=0.6))
    for t in (8.25, 8.5625, 8.875):
        sfx.append(dict(type="pop", t=t, gain=0.4))
    sfx.append(dict(type="hit", t=T_TITLE, gain=0.85))
    return {
        # synth duration runs 0.75 s past the cut so its final-bar treatment lands on the
        # 9.5 s downbeat (with 10.0 it would fall on 7.0 and the groove would stop there);
        # the mix is trimmed to exactly 10.0 s -> hard stop with the picture.
        "duration": 10.75,
        "seed": SEED,
        "style": {"energy": 0.55},
        "audio": {
            "bpm": BPM, "key": "D", "scale": "minor", "mood": "playful",
            "beats": beats, "downbeats": downbeats,
            "sections": [{"t": 0.0, "label": "intro"}, {"t": 2.0, "label": "drop"}, {"t": 7.0, "label": "break"}],
            "risers": [{"t0": 7.0, "t1": 7.5, "gain": 0.5}],
            "sfx": sfx,
        },
    }


def render_audio(plan):
    music = render_soundtrack(plan, SR)
    fx = render_sfx(plan, SR)
    a = mix(music, fx, None, SR)
    n = int(round(DUR * SR))
    a = a[:, :n].copy()
    dc = int(0.004 * SR)
    a[:, -dc:] *= np.linspace(1, 0, dc)[None, :]     # de-click only; the stop is hard
    return a


PROMPT = """Assembly - a collage that builds itself (rough CPU animatic, v001)
Concept: _info/concept-collage.md. Script: tools/animatic_assembly.py (seed {seed}).

Stand-ins (no _inspiration art used):
- main subject: procedurally drawn profile head-and-shoulders silhouette (smooth spline, facing left);
  its interior is a window onto a halftoned stand-in photo
- photos: tests/fixtures/blob_gradient.png, stripes.png, subject_texture.png, all as B/W halftone
  cut-outs (45 deg dot screen, ink on cream) with torn edges and 3 px paper shadows
- flat spot pieces: mustard rectangle, orange bar, six-disc flower, all with 2 px colour misregistration
Palette only: cream #e8dcc3, ink #1d1b19, orange #e2582a, mustard #dca232, teal #3e8a86.

Grid: 96 bpm, beat 0.625 s, phased so the 2.0 s drop is a downbeat (downbeats 2.0/4.5/7.0/9.5).
Arc:
0.0-2.0  01. Drawing - paper; disc (orange) fades up; dashed construction lines, landing outlines,
         dimension line draw on (ease-out); callouts 01./02./03. at 0.75 / 1.375 / 1.6875 with dotted leaders
2.0      Drop - silhouette slams into its outline (1 anticipation frame, 1.08 -> 1.0 over a beat),
         shadow lands, 2-frame paper flash + vertical shake; disc -> mustard
2.0-5.5  02. Assembly - stripes scrap 2.625, mustard rect 3.25, texture scrap 4.1875, flower 4.8125,
         orange bar 5.125, each with the same punch-and-settle; slow drift on every piece
5.5-7.0  Burst - silhouette window swaps on half-beats 5.75 / 6.0625 / 6.375 / 6.6875:
         stripes -> flat orange -> inverted blob halftone -> texture (star)
7.0-7.5  Rest - drift + grain only
7.5-9.0  03. Reveal - disc teal; collage pulls back (cubic in-out, 7.5-8.25) into a torn paper piece
         on a larger print sheet (crop marks, registration targets, colour bar); 4 neighbour pieces
         slide in landing 8.25 / 8.25 / 8.5625 / 8.875
9.0-10.0 End - "ASSEMBLY" label slip (Big Shoulders Display) with an orange bar, held; hard stop at 10.0
Audio: mograph.synth, D minor, mood playful, energy 0.55, sections intro 0 / drop 2.0 / break 7.0,
riser 7.0-7.5, sfx click (callouts), impact 2.0, pop/hit (landings), glitch/pop low (swaps),
swell into 7.5, pop (neighbours), hit 9.0; mixed with mograph.mix, cut hard at 10.0 s.
"""


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", action="store_true", help="short side 540; output to review/")
    ap.add_argument("--fmt", choices=("9x16", "16x9", "both"), default="both")
    ap.add_argument("--stills", default="", help="comma list of times: write PNGs to review/ instead of video")
    a = ap.parse_args()
    fmts = ("9x16", "16x9") if a.fmt == "both" else (a.fmt,)
    os.makedirs(REVIEW_DIR, exist_ok=True)

    if a.stills:
        for fmt in fmts:
            t0 = time.time()
            r = Animatic(fmt, a.draft)
            print(f"[{fmt}] build {time.time() - t0:.1f}s", flush=True)
            for s in a.stills.split(","):
                i = min(NF - 1, int(round(float(s) * FPS)))
                fr = r.frame(i)
                p = os.path.join(REVIEW_DIR, f"still_{fmt}_{float(s):05.2f}s{'_draft' if a.draft else ''}.png")
                Image.fromarray(fr).save(p)
                print(p, flush=True)
        return

    t0 = time.time()
    plan = build_plan()
    audio = render_audio(plan)
    print(f"audio {time.time() - t0:.1f}s", flush=True)
    with open(os.path.join(ROOT, "01_assembly", "v001", "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(PROMPT.format(seed=SEED))
    for fmt in fmts:
        t1 = time.time()
        r = Animatic(fmt, a.draft)
        tb = time.time() - t1
        if a.draft:
            out = os.path.join(REVIEW_DIR, f"motion-graphic_assembly_v001_{fmt}_draft.mp4")
        else:
            out = os.path.join(OUT_DIR, f"motion-graphic_assembly_v001_{fmt}.mp4")
        encode_mp4((r.frame(i) for i in range(NF)), FPS, r.W, r.H, audio, SR, out)
        print(f"[{fmt}] {r.W}x{r.H} build {tb:.1f}s total {time.time() - t1:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
