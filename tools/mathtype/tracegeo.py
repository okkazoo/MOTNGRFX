"""E3 tracegeo: trace natural shapes from the project's own imagery into geometry, then morph it into type.

Sentence (Galileo, paraphrased, no credit on screen):
  "The book of nature is written in the language of mathematics,
   and its letters are triangles, circles and lines."

Pipeline (precomputed once, cached; every frame is then a pure function of t):
  1. Trace: greyscale -> Gaussian blur -> Canny whose thresholds come from Otsu on the Sobel gradient
     magnitude (hi = Otsu split of |grad|, lo = hi/2) -> cv2.findContours -> approxPolyDP.
     Osculating circles at contour curvature maxima (kappa = (x'y''-y'x'')/|v|^3, r = 1/kappa).
  2. Feature points: contour vertices + Poisson-disk samples whose candidates are drawn with density
     proportional to edge strength; golden-angle (Vogel) spiral samples inside sunflower discs.
  3. Text points: Bungee glyph outlines (arc-length resampled) + Poisson interior; for the last phrase,
     TRIANGLES = facets, CIRCLES = greedy inscribed-circle packing, LINES = scanline hatch segments.
  4. One pool of N points visits every state (stones, P1, jelly, P2, sunflower, P3, P4). Consecutive
     states are matched 1:1 with the Hungarian algorithm (centred squared distance), so each point has
     one target per state. Per frame the pool is re-triangulated (Delaunay) / re-tessellated (Voronoi);
     each triangle's weight is its validity in the from-state and to-state, lerped by progress.

Variants: --variant a = Delaunay low-poly facet morph; --variant b = Voronoi + circle-packing lines.
--colour adds the E3 nature palette as a minority accent (flat triangles / solid discs on resolves).

Usage:
  python tools/mathtype/tracegeo.py --variant a --preview          # 360x450 -> E3a_facet_gNN_preview.mp4
  python tools/mathtype/tracegeo.py --variant a --final            # 720x900 + score/chords mux
  python tools/mathtype/tracegeo.py --trace                        # trace debug sheet -> review/
  python tools/mathtype/tracegeo.py --variant a --still 1.0 2.0    # ad-hoc stills
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, Voronoi, cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from mograph.font_registry import font_axes, font_path  # noqa: E402  (read-only use)
import gens  # noqa: E402

VER = os.path.join(REPO, "11_mathtype", "v009")
LOCAL = os.path.join(VER, "local")
REVIEW = os.path.join(VER, "review")
CACHE = os.path.join(LOCAL, ".cache")
T2I = os.path.join(REPO, "02_t2i", "v001", "out", "57178352")

FPS = 25
DUR = 10.0
NFR = int(round(FPS * DUR))
DW, DH = 720, 900                     # design space (= final px); world units are design px
N = 2200                              # point pool
SEED = 3
TRACK = 0.03
TEXT_W = 0.84 * DW
MAX_CAP = 0.13 * DH                  # heavy faces
MAX_CAP_TALL = 0.17 * DH             # tall faces

SOURCES = [
    dict(key="stones", file="motion-graphic_t2i_v001_B2_stones_s2_00001_.png", centre=(0.0, 0.0),
         focus=(0.5, 0.52), height=1000, polarity="dark", heads=False),
    dict(key="jelly", file="motion-graphic_t2i_v001_B5_jelly_s1_00001_.png", centre=(1080.0, 430.0),
         focus=(0.42, 0.40), height=1050, polarity="bright", heads=False),
    dict(key="sunflower", file="motion-graphic_t2i_v001_B4_sunflowers_s2_00001_.png", centre=(2160.0, 860.0),
         focus=(0.5, 0.235), height=1500, polarity="dark", heads=True),
]

PHRASES = [  # (text, font, style): heavy faces built from triangles, tall faces drawn in lines
    [("THE BOOK OF", "Facade", "wire"), ("NATURE", "Blackout", "facet"), ("IS WRITTEN", "Facade", "wire")],
    [("IN THE", "Facade", "wire"), ("LANGUAGE OF", "Facade", "wire"), ("MATHE-", "Climate Crisis", "facet"),
     ("MATICS", "Climate Crisis", "facet")],
    [("AND ITS", "Le Murmure", "wire"), ("LETTERS", "Shrikhand", "facet"), ("ARE", "Le Murmure", "wire")],
    [("TRIANGLES,", "Climate Crisis", "facet"), ("CIRCLES", "Blackout", "circle"), ("AND LINES", "Facade", "hatch")],
]
# state order: S1 T1 S2 T2 S3 T3 T4 ; (from, to, t0, t1)
SEGMENTS = [(0, 1, 0.70, 1.35), (1, 2, 2.60, 3.22), (2, 3, 3.40, 3.95), (3, 4, 5.20, 5.82),
            (4, 5, 6.00, 6.55), (5, 6, 7.80, 8.50)]
LANDINGS = [1.35, 3.95, 6.55, 8.50]
READ = [(1.35, 2.60), (3.95, 5.20), (6.55, 7.80), (8.50, 10.0)]
INVERT = [(6.28, 8.08)]               # black on white for P3 (snap flips)

# palette (E3 nature as geometry)
CHALK = np.array([0xF1, 0xED, 0xE4]) / 255.0
GRAPHITE = np.array([0x2B, 0x2B, 0x2B]) / 255.0
LEAF = np.array([0x5B, 0x8C, 0x51]) / 255.0
TERRA = np.array([0xC8, 0x55, 0x3D]) / 255.0
SKY = np.array([0x7A, 0xA6, 0xC2]) / 255.0

# accent per text state index: (colour, hash threshold out of 1000 = share of facets)
ACCENT = {1: (LEAF, 80), 3: (SKY, 80), 5: (TERRA, 80), 6: (LEAF, 90)}

VARIANT = "a"
COLOUR = False


# ---------------------------------------------------------------------------------------------- utils
def smoothstep(a, b, x):
    u = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def arrive(x):
    """Start from rest, peak early, long settle (skewed smootherstep)."""
    return smootherstep(np.clip(x, 0, 1) ** 0.72)


class WMask:
    """A raster in world space: value at world (x, y) = arr[(y - oy) * sc, (x - ox) * sc]."""

    def __init__(self, arr, off, sc):
        self.arr, self.off, self.sc = arr, np.asarray(off, float), float(sc)

    def sample(self, xy):
        q = (np.asarray(xy) - self.off) * self.sc
        xi = np.round(q[..., 0]).astype(int)
        yi = np.round(q[..., 1]).astype(int)
        h, w = self.arr.shape[:2]
        ok = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
        out = np.zeros(xi.shape, np.float32)
        out[ok] = self.arr[yi[ok], xi[ok]]
        return out


def poisson_greedy(cands, r, rr=None):
    """Greedy Poisson-disk thinning of candidates (priority order kept). Fixed radius r, or a
    per-candidate radius r * rr[k] (variable density): accept k if no accepted point lies within it."""
    rel = np.ones(len(cands)) if rr is None else np.asarray(rr, float)
    rmin = r * float(rel.min()) if len(rel) else r
    cell = rmin / np.sqrt(2.0)
    grid = {}
    acc = []
    for k in range(len(cands)):
        x, y = cands[k]
        rk = r * rel[k]
        r2 = rk * rk
        gx, gy = int(x // cell), int(y // cell)
        m = int(np.ceil(rk / cell))
        ok = True
        for dx in range(-m, m + 1):
            for dy in range(-m, m + 1):
                q = grid.get((gx + dx, gy + dy))
                if q is not None and (q[0] - x) ** 2 + (q[1] - y) ** 2 < r2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[(gx, gy)] = (x, y)
            acc.append(k)
    return np.array(acc, int)


def fit_count(fn, need, lo, hi, it=14):
    """Bisection on a spacing parameter s so fn(s) returns >= need items (largest such s)."""
    best = fn(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        res = fn(mid)
        if len(res[0]) >= need:
            lo, best = mid, res
        else:
            hi = mid
    return best


# ---------------------------------------------------------------------------------------------- trace
def trace_source(src, rng):
    im = cv2.imread(os.path.join(T2I, src["file"]), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sc = src["height"] / g.shape[0]
    g = cv2.resize(g, (int(round(g.shape[1] * sc)), src["height"]), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    off = np.array(src["centre"]) - np.array([src["focus"][0] * w, src["focus"][1] * h])
    gb = cv2.GaussianBlur(g, (0, 0), 1.6)
    # structured Canny threshold: Otsu split of the Sobel gradient magnitude (Canny's own units)
    sx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    mmax = float(mag.max()) + 1e-6
    m8 = np.clip(mag / mmax * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = otsu / 255.0 * mmax
    edges = cv2.Canny(gb, 0.5 * hi, hi, L2gradient=True)
    # subject mask (Otsu on intensity, polarity per source)
    t_i, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    subj = (gb < t_i) if src["polarity"] == "dark" else (gb > min(t_i, 40))
    subj = subj.astype(np.uint8) * 255
    subj = cv2.morphologyEx(subj, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    subj = cv2.morphologyEx(subj, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(subj)
    keep = np.zeros(nl, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] > 0.002 * h * w
    subj = (keep[lab]).astype(np.uint8) * 255
    heads = find_heads(gb, subj) if src["heads"] else []
    if heads:   # sunflowers: keep the flower heads (disc + petal ring) as the subject, drop the leaves
        hm = np.zeros_like(subj)
        for (cx, cy, rd) in heads:
            cv2.circle(hm, (int(cx), int(cy)), int(2.05 * rd), 255, -1)
        subj = np.minimum(subj, hm)
        roi = cv2.dilate(hm, np.ones((41, 41), np.uint8))
        edges = np.minimum(edges, roi)
    # contours of the Canny edges -> simplified polylines
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cs:
        L = cv2.arcLength(c, False)
        if L < 45:
            continue
        ap = cv2.approxPolyDP(c, 1.1, False)[:, 0, :].astype(np.float32)
        if len(ap) >= 2:
            polys.append((L, ap))
    polys.sort(key=lambda p: -p[0])
    polys = [p for _, p in polys[:260]]
    # silhouettes of the subject (external), simplified; plus osculating circles at curvature maxima
    sc_, _ = cv2.findContours(subj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sil = []
    circles = []
    for c in sc_:
        L = cv2.arcLength(c, True)
        if L < 120:
            continue
        sil.append(cv2.approxPolyDP(c, 1.4, True)[:, 0, :].astype(np.float32))
        circles += osculating(c[:, 0, :].astype(np.float64))
    circles.sort(key=lambda c: -c[3])
    circles = circles[:22]
    # feature points
    vtx = np.concatenate([p for p in polys] + sil, 0) if (polys or sil) else np.zeros((0, 2), np.float32)
    rng_v = rng.permutation(len(vtx))
    vtx = vtx[rng_v]
    insub = cv2.dilate(subj, np.ones((15, 15), np.uint8)) > 0
    vtx = vtx[insub[np.clip(vtx[:, 1].astype(int), 0, h - 1), np.clip(vtx[:, 0].astype(int), 0, w - 1)]]
    spiral = np.zeros((0, 2))
    if heads:
        sp = []
        for (cx, cy, rd) in heads:
            c = 6.0
            n = int((rd / c) ** 2)
            j = np.arange(1, n + 1)
            rr = c * np.sqrt(j)
            th = j * np.deg2rad(137.50776)
            sp.append(np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1))
        spiral = np.concatenate(sp, 0) if sp else spiral
    emag = cv2.GaussianBlur(mag, (0, 0), 3.0)
    emag = emag / (np.percentile(emag, 99.5) + 1e-6)
    dens = 0.015 + 0.22 * (subj > 0) + np.clip(emag, 0, 1.2)
    p = dens.ravel() / dens.sum()
    idx = rng.choice(h * w, 60000, p=p)
    rnd = np.stack([idx % w, idx // w], 1).astype(np.float64) + rng.random((60000, 2))
    cands = np.concatenate([spiral, vtx, rnd], 0)
    # variable radius: r ~ 1/sqrt(normalised density); spiral seeds get their own (dense) radius
    dn = dens / np.percentile(dens, 99.0)
    ci = np.clip(cands.astype(int), 0, [w - 1, h - 1])
    rel = 1.0 / np.sqrt(np.clip(dn[ci[:, 1], ci[:, 0]], 0.05, 1.0))
    rel = np.minimum(rel, 4.0)
    rel[: len(spiral)] = 0.0  # set below, relative to the run radius

    def run(r):
        rr = rel.copy()
        rr[: len(spiral)] = 5.0 / r
        acc = poisson_greedy(cands, r, np.maximum(rr, 0.3))
        return (acc,)

    acc = fit_count(run, N, 3.0, 30.0, it=11)[0][:N]
    pts = cands[acc]
    spacing = np.sqrt((subj > 0).sum() / max(1, ((subj > 0)[pts[:, 1].astype(int).clip(0, h - 1),
                                                           pts[:, 0].astype(int).clip(0, w - 1)]).sum()))
    gbl = cv2.GaussianBlur(g, (0, 0), 2.5).astype(np.float32) / 255.0
    subjd = cv2.dilate(subj, np.ones((7, 7), np.uint8)).astype(np.float32) / 255.0
    return dict(key=src["key"], off=off, grey=g.astype(np.float32) / 255.0, grey_bl=gbl,
                subj=subjd, polys=[p + off for p in polys], sil=[s + off for s in sil],
                circles=[(cx + off[0], cy + off[1], r, s) for cx, cy, r, s in circles],
                heads=[(cx + off[0], cy + off[1], r) for cx, cy, r in heads],
                pts=pts + off, spacing=float(spacing), edges=edges, size=(w, h))


def osculating(c):
    """Osculating circles at local curvature maxima of a closed contour (x, y) array."""
    L = np.sum(np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1))
    n = max(16, int(L / 3.0))
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1))]
    cc = np.vstack([c, c[:1]])
    s = np.linspace(0, d[-1], n, endpoint=False)
    x = np.interp(s, d, cc[:, 0])
    y = np.interp(s, d, cc[:, 1])
    k = cv2.getGaussianKernel(25, 4.0)[:, 0]
    xs = np.convolve(np.r_[x[-12:], x, x[:12]], k, "valid")
    ys = np.convolve(np.r_[y[-12:], y, y[:12]], k, "valid")
    x1, y1 = np.gradient(xs), np.gradient(ys)
    x2, y2 = np.gradient(x1), np.gradient(y1)
    kap = (x1 * y2 - y1 * x2) / np.maximum((x1 * x1 + y1 * y1) ** 1.5, 1e-9)
    out = []
    a = np.abs(kap)
    for i in range(n):
        win = a[[(i + j) % n for j in range(-7, 8)]]
        if a[i] < win.max() or not (1 / 70.0 < a[i] < 1 / 7.0):
            continue
        sp = np.hypot(x1[i], y1[i])
        nx, ny = -y1[i] / sp, x1[i] / sp
        r = 1.0 / kap[i]
        out.append((xs[i] + nx * r, ys[i] + ny * r, abs(r), a[i] * min(abs(r), 40)))
    return out


def find_heads(gb, subj):
    """Sunflower discs: dark compact blobs in the upper part -> (cx, cy, r)."""
    h, w = gb.shape
    dark = ((gb < 95) & (np.arange(h)[:, None] < 0.32 * h)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        A = cv2.contourArea(c)
        if A < 0.0015 * h * w:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if A / (np.pi * r * r) > 0.5:
            out.append((cx, cy, r))
    return out


# ---------------------------------------------------------------------------------------------- type
def load_font(name, size):
    f = ImageFont.truetype(font_path(name), int(size))
    ax = font_axes(name)
    if ax:
        tags = {"weight": "wght", "width": "wdth", "optical size": "opsz", "year": "YEAR", "contrast": "CTST"}
        vals = []
        for a in f.get_variation_axes():
            nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
            vals.append(ax.get(tags.get(nm.lower(), nm), a["default"]))
        f.set_variation_by_axes(vals)
    return f


def line_w(f, s):
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def text_masks(lines, ss=2):
    """Justified stack centred in a DWxDH frame, at ss x; one mask per line (text, font, style)."""
    sizes = []
    for txt, fn, sty in lines:
        probe = load_font(fn, 200)
        pb = probe.getbbox("H")
        cap = MAX_CAP_TALL if sty in ("wire", "hatch") else MAX_CAP
        sizes.append(min(200 * TEXT_W / line_w(probe, txt), 200 * cap / (pb[3] - pb[1])))
    fonts = [load_font(fn, z * ss) for (txt, fn, _), z in zip(lines, sizes)]
    caps = [f.getbbox("H")[3] - f.getbbox("H")[1] for f in fonts]
    gap = 0.22 * float(np.median(caps))
    top = DH * ss / 2 - (sum(caps) + gap * (len(lines) - 1)) / 2
    masks = []
    y0 = top
    for (txt, fn, _), f, cap in zip(lines, fonts, caps):
        im = Image.new("L", (DW * ss, DH * ss), 0)
        d = ImageDraw.Draw(im)
        hb = f.getbbox("H")
        x = DW * ss / 2 - line_w(f, txt) / 2 - f.getbbox(txt[0])[0]
        for ch in txt:
            d.text((x, y0 - hb[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        masks.append(np.array(im))
        y0 += cap + gap
    return masks


def outline_interior(mask, need, rng, k_in=1.7, lo=2.0, hi=40.0):
    """Outline (arc-resampled + corners) and Poisson interior points of a 0/255 mask (mask px)."""
    cs, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    conts = [c[:, 0, :].astype(np.float64) for c in cs if len(c) > 4]
    corners = np.concatenate([cv2.approxPolyDP(c.astype(np.float32)[:, None, :], 2.0, True)[:, 0, :]
                              for c in conts], 0).astype(np.float64)
    dt = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt > 0)
    order = rng.permutation(len(xs))[:40000]
    inner_c = np.stack([xs[order], ys[order]], 1).astype(np.float64) + rng.random((len(order), 2)) - 0.5

    def run(s):
        res = [corners]
        for c in conts:
            cc = np.vstack([c, c[:1]])
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))]
            m = max(3, int(d[-1] / s))
            u = np.linspace(0, d[-1], m, endpoint=False)
            res.append(np.stack([np.interp(u, d, cc[:, 0]), np.interp(u, d, cc[:, 1])], 1))
        oc = np.concatenate(res, 0)
        o = oc[poisson_greedy(oc, 0.72 * s)]
        si = k_in * s
        ic = inner_c[dt[inner_c[:, 1].astype(int), inner_c[:, 0].astype(int)] > 0.62 * si]
        i = ic[poisson_greedy(ic, si)] if len(ic) else np.zeros((0, 2))
        return (np.concatenate([o, i], 0), len(o))

    pts, no = fit_count(run, need, lo, hi, it=14)
    o, i = pts[:no], pts[no:]
    if len(o) + len(i) > need:
        drop = len(o) + len(i) - need
        if drop <= len(i):
            i = i[np.sort(rng.permutation(len(i))[: len(i) - drop])]
        else:
            o = o[np.sort(rng.permutation(len(o))[: need])]
            i = i[:0]
    return o, i


def circle_pack(mask, rmin, nmax, gap=1.2):
    """Greedy inscribed-circle packing: repeatedly take the distance-transform maximum."""
    dt = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    h, w = dt.shape
    out = []
    while len(out) < nmax:
        k = int(np.argmax(dt))
        cy, cx = divmod(k, w)
        r = float(dt[cy, cx])
        if r < rmin:
            break
        out.append((cx, cy, r - 0.5 * gap))
        R = int(2 * r + gap + 2)
        x0, x1, y0, y1 = max(0, cx - R), min(w, cx + R + 1), max(0, cy - R), min(h, cy + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r - gap
        dt[y0:y1, x0:x1] = np.minimum(dt[y0:y1, x0:x1], d)
    return np.array(out)


def hatch(mask, step, minlen):
    segs = []
    h, w = mask.shape
    for y in np.arange(step / 2, h, step):
        row = mask[int(y)] > 127
        if not row.any():
            continue
        dd = np.diff(np.r_[0, row.astype(np.int8), 0])
        st, en = np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]
        for a, b in zip(st, en):
            if b - a >= minlen:
                segs.append((a + 0.5, y, b - 0.5, y))
    return np.array(segs)


MODE = {"facet": 0, "circle": 1, "hatch": 2, "wire": 3}


def text_state(lines, centre, rng, literal=False):
    """Point targets for one phrase. Per line style: facet / wire -> outline + Poisson interior points
    (Delaunay facets or wireframe), circle -> inscribed-circle packing centres, hatch -> scanline segment
    end points (pairs)."""
    ss = 2
    masks = text_masks(lines, ss)
    full = np.maximum.reduce(masks)
    off = np.array(centre) - np.array([DW / 2, DH / 2])
    styles = [sty for _, _, sty in lines]
    parts, kinds, modes, rp = [], [], [], []
    for m, sty in zip(masks, styles):
        if sty == "circle":
            c = circle_pack(m, rmin=2.6 * ss, nmax=520)
            parts.append(c[:, :2])
            kinds.append(sty)
            modes.append(np.full(len(c), 1))
            rp.append(c[:, 2] / ss)
        elif sty == "hatch":
            sg = hatch(m, step=5.0 * ss, minlen=2 * ss)
            if len(sg) > 360:
                sg = sg[np.sort(rng.permutation(len(sg))[:360])]
            parts.append(np.concatenate([sg[:, :2], sg[:, 2:]], 0))
            kinds.append(sty)
            modes.append(np.full(2 * len(sg), 2))
            rp.append(np.zeros(2 * len(sg)))
    nfixed = sum(len(p) for p in parts)
    fw = np.maximum.reduce([m for m, sty in zip(masks, styles) if sty in ("facet", "wire")])
    o, i = outline_interior(fw, N - nfixed, rng)
    tri = np.concatenate([o, i], 0)
    lab = np.argmax(np.stack([cv2.dilate(m, np.ones((5, 5), np.uint8))[
        np.clip(tri[:, 1].astype(int), 0, DH * ss - 1), np.clip(tri[:, 0].astype(int), 0, DW * ss - 1)]
        for m in masks], 0), 0)
    tri_mode = np.array([MODE[styles[k]] for k in lab])
    pts = np.concatenate([tri] + parts, 0) / ss + off
    mode = np.concatenate([tri_mode] + modes)
    rpack = np.concatenate([np.zeros(len(tri))] + rp)
    partner = -np.ones(len(pts), int)
    base = len(tri)
    for sty, p in zip(kinds, parts):
        if sty == "hatch":
            ns = len(p) // 2
            partner[base:base + ns] = np.arange(base + ns, base + 2 * ns)
            partner[base + ns:base + 2 * ns] = np.arange(base, base + ns)
        base += len(p)
    assert len(pts) == N, (len(pts), N)
    dt = cv2.distanceTransform((full > 127).astype(np.uint8), cv2.DIST_L2, 5) / ss
    mk = cv2.dilate(fw, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    st = dict(kind="T", pts=pts, centre=np.array(centre, float), mask=WMask(mk, off, ss),
              full=WMask(cv2.dilate(full, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0, off, ss),
              dt=WMask(dt.astype(np.float32), off, ss), nout=len(o), mode=mode, partner=partner, rpack=rpack,
              literal=literal, lines=lines)
    st["spacing"] = float(np.median(cKDTree(tri / ss).query(tri / ss, k=2)[0][:, 1]))
    return st


# ---------------------------------------------------------------------------------------------- scene
def build():
    rng = np.random.default_rng(SEED)
    srcs = [trace_source(s, rng) for s in SOURCES]
    states = []
    for k in range(3):
        s = srcs[k]
        st = dict(kind="S", pts=s["pts"].copy(), centre=np.array(SOURCES[k]["centre"], float),
                  mask=WMask(s["subj"], s["off"], 1.0), grey=WMask(s["grey_bl"], s["off"], 1.0), src=k)
        st["spacing"] = float(np.median(cKDTree(st["pts"]).query(st["pts"], k=2)[0][:, 1]))
        states.append(st)
        states.append(text_state(PHRASES[k], SOURCES[k]["centre"], rng))
    states.append(text_state(PHRASES[3], SOURCES[2]["centre"], rng, literal=True))
    # chain Hungarian matching: reorder every state to the index order of the previous one
    for k in range(1, len(states)):
        A, B = states[k - 1], states[k]
        a = A["pts"] - A["centre"]
        b = B["pts"] - B["centre"]
        C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(C)
        perm = col                                    # B index matched to A index i
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        B["pts"] = B["pts"][perm]
        for key in ("mode", "rpack"):
            if B.get(key) is not None:
                B[key] = B[key][perm]
        if B.get("partner") is not None:
            p = B["partner"][perm]
            B["partner"] = np.where(p >= 0, inv[np.maximum(p, 0)], -1)
        if B["kind"] == "T":
            no = np.zeros(N, bool)
            no[: B["nout"]] = True
            B["outline"] = no[perm]
    # per-point attributes per state
    for st in states:
        P = st["pts"]
        nn = cKDTree(P).query(P, k=2)[0][:, 1]
        if st["kind"] == "S":
            st["pv"] = np.clip(st["mask"].sample(P), 0, 1)
            st["r"] = 0.42 * nn
            st["z"] = np.zeros(N)
        else:
            d = st["dt"].sample(P)
            st["pv"] = np.ones(N)
            st["r"] = np.minimum(0.5 * nn, np.maximum(d, 0.8))
            st["z"] = np.minimum(d, 7.0)
            m = st["mode"]
            st["r"] = np.where(m == 1, st["rpack"], np.where(m == 0, st["r"], 0.0))
    srcs_light = [{k: v for k, v in s.items() if k != "edges"} for s in srcs]
    hsh = np.random.default_rng(99).random(N)
    return dict(srcs=srcs_light, states=states, hash=hsh)


def cache_key():
    h = hashlib.sha1()
    with open(os.path.abspath(__file__), "rb") as f:
        src = f.read()
    # only the build code and its constants matter (timeline / render edits keep the cache)
    h.update(src[src.index(b"def poisson_greedy"):src.index(b"# ==== render")])
    h.update(repr((SOURCES, PHRASES, N, SEED, DW, DH, TEXT_W, MAX_CAP, MAX_CAP_TALL, TRACK)).encode())
    return h.hexdigest()[:12]


def load_scene():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"scene_{cache_key()}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    sc = build()
    with open(p + ".tmp", "wb") as f:
        pickle.dump(sc, f)
    os.replace(p + ".tmp", p)
    print(f"built scene in {time.time() - t0:.1f}s -> {p}")
    return sc


# ==== render
def style(st, variant):
    """Per-point style weights in a state: facet f, voronoi v, circle c, line l, wire wr."""
    f, v, c, ln, wr = (np.zeros(N) for _ in range(5))
    if st["kind"] == "S":
        if variant == "a":
            f[:] = 1
        else:
            v[:] = 1
            c[:] = 1
        return f, v, c, ln, wr
    m = st["mode"]
    if variant == "a" or st.get("literal"):
        f[m == 0] = 1
    else:
        v[m == 0] = 1
        c[m == 0] = 1
    c[m == 1] = 1
    ln[m == 2] = 1
    wr[m == 3] = 1
    return f, v, c, ln, wr


CAM_KEYS = [  # (t0, t1, from-centre idx, to-centre idx)
    (2.60, 3.22, 0, 1), (5.20, 5.82, 1, 2)]


def camera(t):
    cs = [np.array(s["centre"], float) for s in SOURCES]
    c, z = cs[0], 1.0
    z = 1.10 - 0.10 * float(smootherstep(t / 1.35))      # intro pull-back to the P1 framing
    for t0, t1, a, b in CAM_KEYS:
        if t >= t0:
            e = float(smootherstep((t - t0) / (t1 - t0)))
            c = cs[a] + (cs[b] - cs[a]) * e
            z = np.exp(-0.30 * np.sin(np.pi * e))
    return c, z


def state_at(t):
    for a, b, t0, t1 in reversed(SEGMENTS):
        if t >= t0:
            return a, b, t0, t1
    a, b, t0, t1 = SEGMENTS[0]
    return a, b, t0, t1


class Scene:
    def __init__(self, variant="a", colour=False, backend="numpy", device="cpu"):
        self.sc = load_scene()
        self.backend = backend
        self.raster = TorchRaster(device) if backend == "torch" else CvRaster()
        self.variant = variant
        self.colour = colour
        self.states = self.sc["states"]
        self.styles = [style(s, variant) for s in self.states]
        self.hash = self.sc["hash"]
        self.rank = {}
        for k, (a, b, t0, t1) in enumerate(SEGMENTS):
            A, B = self.states[a], self.states[b]
            ref = B if B["kind"] == "T" else A
            x = ref["pts"][:, 0] - ref["centre"][0]
            y = ref["pts"][:, 1] - ref["centre"][1]
            r = (x + 0.25 * y)
            r = (r - r.min()) / (np.ptp(r) + 1e-9)
            self.rank[k] = np.clip(0.88 * r + 0.12 * self.hash, 0, 1)
        g = np.random.default_rng(7)
        self.arc = g.uniform(-0.22, 0.22, N)
        self.jit = (g.random((N, 2)) - 0.5) * 1e-3

    # -- per-point state
    def points(self, t):
        a, b, t0, t1 = state_at(t)
        k = SEGMENTS.index((a, b, t0, t1))
        A, B = self.states[a], self.states[b]
        if t < t0:
            u = np.zeros(N)
        else:
            sf = 0.38
            D = t1 - t0
            x = (t - t0 - sf * D * self.rank[k]) / ((1 - sf) * D)
            u = arrive(x)
        pa, pb = A["pts"], B["pts"]
        rel = (pb - B["centre"]) - (pa - A["centre"])
        perp = np.stack([-rel[:, 1], rel[:, 0]], 1)
        # cross-location segments: the pool rides the camera (shared ease) and rearranges locally
        dc = B["centre"] - A["centre"]
        ec = float(smootherstep((t - t0) / (t1 - t0))) if np.any(dc) else 0.0
        P = (pa + dc * ec + rel * u[:, None] + perp * (self.arc * np.sin(np.pi * u))[:, None] + self.jit)
        return P, u, A, B, a, b

    # -- validity of triangles in a state (vertex positions taken from that state)
    def tri_valid(self, st, tri):
        """Validity of each triangle judged with the vertex positions of state st, and max edge length."""
        P = st["pts"]
        p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        L = np.maximum.reduce([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1),
                               np.linalg.norm(p0 - p2, axis=1)])
        cen = (p0 + p1 + p2) / 3
        if st["kind"] == "S":
            v = st["mask"].sample(cen) * (L < 3.4 * st["spacing"])
        else:
            m = st["mask"]
            v = (m.sample(cen) * m.sample((p0 + p1) / 2) * m.sample((p1 + p2) / 2) * m.sample((p2 + p0) / 2)
                 * (L < 6.0 * st["spacing"]))
        return v, L

    @staticmethod
    def tri_min(x, tri):
        return np.minimum.reduce([x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]])

    def tri_shade(self, st, tri):
        P = st["pts"]
        if st["kind"] == "S":
            cen = (P[tri[:, 0]] + P[tri[:, 1]] + P[tri[:, 2]]) / 3
            return 0.10 + 0.72 * st["grey"].sample(cen)
        z = st["z"] * 1.6
        v0 = np.c_[P[tri[:, 0]], z[tri[:, 0]]]
        v1 = np.c_[P[tri[:, 1]], z[tri[:, 1]]]
        v2 = np.c_[P[tri[:, 2]], z[tri[:, 2]]]
        n = np.cross(v1 - v0, v2 - v0)
        n = n * np.sign(n[:, 2:3] + 1e-9)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        Ld = np.array([-0.55, -0.62, 0.56])
        Ld /= np.linalg.norm(Ld)
        lam = np.clip(n @ Ld, 0, 1)
        e = v1[:, :2] - v0[:, :2]
        ang = np.arctan2(e[:, 1], e[:, 0])
        return np.clip(0.30 + 0.62 * lam + 0.07 * np.cos(2 * ang), 0, 1)

    # -- geometry: one frame as a draw list (numpy; Delaunay / Voronoi are per-frame CPU constructions)
    def geom(self, t, W, H, ss=2):
        CW, CH = W * ss, H * ss
        cam, zoom = camera(t)
        self._cam = (cam, zoom)
        s = zoom * CW / DW
        off = np.array([CW / 2, CH / 2])

        def to_scr(p):
            return (p - cam) * s + off

        k = CW / DW                      # resolution factor: stroke widths are preview-equivalent x k
        self._k = k
        g = dict(CW=CW, CH=CH, W=W, H=H, s=s, cam=cam, k=k)
        hair = max(1, int(round(1.0 * s / k)))
        srcs = self.sc["srcs"]
        # 1) photographic base (stones), fades out
        ap = 0.42 * (1 - float(smoothstep(0.15, 1.25, t)))
        g["photo"] = []
        if ap > 0.003:
            s0 = srcs[0]
            g["photo"].append((ap, 0, s, (s0["off"][0] - cam[0]) * s + CW / 2, (s0["off"][1] - cam[1]) * s + CH / 2))
        # 2) traced contours + osculating circles (world-static construction layer)
        windows = [(0.0, 0.75, 0.95, 1.45), (2.62, 3.22, 3.45, 3.95), (5.22, 5.82, 6.05, 6.50)]
        lsegs, arcs, discs = [], [], []
        for k, (d0, d1, f0, f1) in enumerate(windows):
            if t < d0 - 0.05:
                continue
            prog = float(smoothstep(d0, d1, t))
            fade = 1 - 0.84 * float(smoothstep(f0, f1, t))
            if prog <= 0 or fade <= 0:
                continue
            val = int(255 * fade)
            src = srcs[k]
            lsegs.append(self._draw_on([to_scr(p) for p in src["polys"]], prog, int(val * 0.55) / 255.0, 1, False))
            lsegs.append(self._draw_on([to_scr(p) for p in src["sil"]], prog, int(val * 0.9) / 255.0, hair, True))
            for j, (cx, cy, r, _) in enumerate(src["circles"]):
                pj = float(np.clip(prog * 1.6 - 0.6 * j / max(1, len(src["circles"])), 0, 1))
                if pj <= 0:
                    continue
                c = to_scr(np.array([cx, cy]))
                rr = r * s
                if -rr < c[0] < CW + rr and -rr < c[1] < CH + rr:
                    arcs.append((c[0], c[1], rr, -90.0, 360.0 * pj, int(val * 0.85) / 255.0, hair))
                    discs.append((c[0], c[1], 1.5 * s, int(val * 0.8) / 255.0))
        lsegs = [x for x in lsegs if x is not None]
        g["lay_segs"] = tuple(np.concatenate([x[i] for x in lsegs]) for i in range(4)) if lsegs else None
        g["lay_arcs"] = np.array(arcs, float).reshape(-1, 7)
        g["lay_discs"] = np.array(discs, float).reshape(-1, 4)
        # 3) the point pool
        P, u, A, B, ia, ib = self.points(t)
        Sa, Sb = self.styles[ia], self.styles[ib]
        self._sab = (Sa, Sb, ia, ib)
        sty = [Sa[j] + (Sb[j] - Sa[j]) * u for j in range(5)]
        Pscr = to_scr(P)
        tri = Delaunay(P).simplices
        s_tri = (u[tri[:, 0]] + u[tri[:, 1]] + u[tri[:, 2]]) / 3
        vA, LA = self.tri_valid(A, tri)
        vB, LB = self.tri_valid(B, tri)
        fA, fB = vA * self.tri_min(Sa[0], tri), vB * self.tri_min(Sb[0], tri)
        e_a = 0.85 if A["kind"] == "S" else 0.5
        e_b = 0.85 if B["kind"] == "S" else 0.5
        eA = vA * (e_a * self.tri_min(Sa[0], tri) + 0.95 * self.tri_min(Sa[4], tri))
        eB = vB * (e_b * self.tri_min(Sb[0], tri) + 0.95 * self.tri_min(Sb[4], tri))
        w = fA * (1 - s_tri) + fB * s_tri
        ew = eA * (1 - s_tri) + eB * s_tri
        Lc = np.maximum.reduce([np.linalg.norm(P[tri[:, i]] - P[tri[:, (i + 1) % 3]], axis=1) for i in range(3)])
        Lref = np.maximum(LA * (1 - s_tri) + LB * s_tri, 1.0)
        stretch = np.clip((1.8 - Lc / Lref) / 0.6, 0, 1)    # stretched mid-flight triangles fade out
        w *= stretch
        ew *= np.clip((2.6 - Lc / Lref) / 0.8, 0, 1)        # wire survives more stretch than fill
        # assembly draw-on for the opening (top-down sweep, short edges first)
        if t < 1.0:
            cy_ = (P[tri[:, 0], 1] + P[tri[:, 1], 1] + P[tri[:, 2], 1]) / 3
            rk = (0.65 * np.argsort(np.argsort(cy_)) + 0.35 * np.argsort(np.argsort(Lc))) / max(1, len(Lc) - 1)
            edge_on = np.clip((t - 0.05) / 0.6 - rk, 0, 1) * 4
            fill_on = np.clip(((t - 0.35) / 0.55 - rk) * 3, 0, 1)
        else:
            edge_on = 1.0
            fill_on = 1.0
        gA = self.tri_shade(A, tri)
        gB = self.tri_shade(B, tri)
        shade = gA * (1 - s_tri) + gB * s_tri
        alpha_f = np.clip(w * fill_on, 0, 1)
        keep = alpha_f > 0.02
        g["fill"] = (Pscr[tri[keep]], np.clip(shade[keep], 0, 1) * alpha_f[keep], alpha_f[keep])
        g["cfill"] = self._colour_facets(tri[keep], Pscr, alpha_f[keep], s_tri[keep], A, B, t) if self.colour else None
        ew = np.clip(ew * edge_on, 0, 1)
        ek = ew > 0.03
        g["edges"] = (Pscr[tri[ek]], ew[ek])
        # travelling feature points: visible as dots while in flight
        fl = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.6
        mv = (fl > 0.04) & (Pscr[:, 0] >= 0) & (Pscr[:, 0] < CW) & (Pscr[:, 1] >= 0) & (Pscr[:, 1] < CH)
        g["dots"] = np.c_[Pscr[mv], np.full(mv.sum(), max(1.0, 1.3 * s)), 235 * fl[mv] / 255.0]
        g["ridges"] = self._voronoi(P, u, A, B, sty, to_scr, s) if sty[1].max() > 0.01 else None
        g["circles"] = self._circles(Pscr, u, A, B, sty, s, CW, CH, t) if sty[2].max() > 0.01 else None
        g["lines"] = self._lines(Pscr, u, A, B, sty, s) if sty[3].max() > 0.01 else None
        inv = 0.0
        for i0, i1 in INVERT:
            inv = max(inv, float(smoothstep(i0, i0 + 0.08, t)) * (1 - float(smoothstep(i1, i1 + 0.08, t))))
        g["inv"] = inv
        return g

    def _draw_on(self, polys, prog, val, th, closed):
        """Prefix of each polyline up to arc-length fraction prog -> segments (p0, p1, val, thickness)."""
        if val <= 0:
            return None
        p0s, p1s = [], []
        for p in polys:
            if closed:
                p = np.vstack([p, p[:1]])
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            Lp = cum[-1] * prog
            n = int(np.searchsorted(cum, Lp))
            if n < 1:
                continue
            q = p[:n]
            if n < len(p):
                f = (Lp - cum[n - 1]) / max(seg[n - 1], 1e-6)
                q = np.vstack([q, p[n - 1] + (p[n] - p[n - 1]) * f])
            if len(q) >= 2:
                p0s.append(q[:-1])
                p1s.append(q[1:])
        if not p0s:
            return None
        a, b = np.concatenate(p0s), np.concatenate(p1s)
        return a, b, np.full(len(a), val), np.full(len(a), float(th))

    def _voronoi(self, P, u, A, B, sty, to_scr, s):
        far = np.array([[-1e5, -1e5], [1e5, -1e5], [-1e5, 1e5], [1e5, 1e5]]) + P.mean(0)
        vor = Voronoi(np.vstack([P, far]))
        rp = vor.ridge_points
        rv = np.array(vor.ridge_vertices)
        ok = (rp < N).all(1) & (rv >= 0).all(1)
        rp, rv = rp[ok], rv[ok]
        i, j = rp[:, 0], rp[:, 1]
        su = (u[i] + u[j]) / 2

        def pair_valid(st):
            Q = st["pts"]
            d = np.linalg.norm(Q[i] - Q[j], axis=1)
            mid = (Q[i] + Q[j]) / 2
            return st["mask"].sample(mid) * (d < 2.6 * st["spacing"])

        w = pair_valid(A) * (1 - su) + pair_valid(B) * su
        w *= np.minimum(sty[1][i], sty[1][j])
        V = to_scr(vor.vertices)
        seg = V[rv]
        Lr = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1) / s
        w *= np.clip(2 - Lr / (3.0 * (A["spacing"] * (1 - su) + B["spacing"] * su)), 0, 1)
        keep = w > 0.03
        sg = float(u.mean())
        # clip the ridges to the two states' masks, weighted by mean progress
        return seg[keep, 0], seg[keep, 1], np.clip(w[keep], 0, 1) * 0.8, [(1 - sg, A), (sg, B)]

    def _circles(self, Pscr, u, A, B, sty, s, CW, CH, t):
        c = sty[2]
        pv = A["pv"] * (1 - u) + B["pv"] * u
        r = (A["r"] * (1 - u) + B["r"] * u) * s
        a = c * pv
        keep = (a > 0.03) & (r > 0.6)
        keep &= (Pscr[:, 0] > -r) & (Pscr[:, 1] > -r) & (Pscr[:, 0] < CW + r) & (Pscr[:, 1] < CH + r)
        if not keep.any():
            return None
        idx = np.nonzero(keep)[0]
        rmax = max(1e-6, np.percentile(r[idx], 95))
        th = max(1, int(round(0.9 * s / self._k)))
        g = 0.28 + 0.45 * np.minimum(1.0, r[idx] / rmax)
        disc = self._colour_circles(idx, r, a, u, A, B, t) if self.colour else None
        return Pscr[idx], r[idx], g * a[idx], a[idx], th, disc

    def _lines(self, Pscr, u, A, B, sty, s):
        st = B if B.get("partner") is not None and (B["partner"] >= 0).any() else A
        if st.get("partner") is None:
            return None
        prt = st["partner"]
        ln = sty[3]
        i = np.nonzero((prt >= 0) & (np.arange(N) < np.maximum(prt, 0)))[0]
        j = prt[i]
        a = np.minimum(ln[i], ln[j])
        keep = a > 0.03
        th = max(1, int(round(1.6 * s / self._k)))
        return Pscr[i][keep], Pscr[j][keep], a[keep], th

    # -- colour accents (E3 nature palette, minority)
    def _colour_facets(self, tri, Pscr, alpha, s_tri, A, B, t):
        """Clustered flat palette triangles inside the heavy words (spatial-cell hash + vertex-triple hash)."""
        Sa, Sb, ia, ib = self._sab
        ts = np.sort(tri, 1)
        hv = ((ts[:, 0] * 73856093) ^ (ts[:, 1] * 19349663) ^ (ts[:, 2] * 83492791)) % 1000
        out = []
        for st, sty, wgt, idx in ((A, Sa, 1 - s_tri, ia), (B, Sb, s_tri, ib)):
            if st["kind"] != "T":
                continue
            pal = ACCENT.get(idx)
            if pal is None:
                continue
            f = self.tri_min(sty[0], tri)
            Q = st["pts"]
            cen = (Q[tri[:, 0]] + Q[tri[:, 1]] + Q[tri[:, 2]]) / 3 - st["centre"]
            cell = np.floor(cen / 34.0).astype(np.int64)
            hc = ((cell[:, 0] * 92837111) ^ (cell[:, 1] * 689287499) ^ idx * 283923481) % 1000
            a = alpha * wgt * f * (hc < 2.2 * pal[1]) * (hv < 520)
            a *= self._land(idx, t)
            sel = a > 0.05
            if sel.any():
                out.append((Pscr[tri[sel]], np.repeat(pal[0][None], sel.sum(), 0), a[sel]))
        if not out:
            return None
        return tuple(np.concatenate([o[i] for o in out]) for i in range(3))

    def _colour_circles(self, idx, r, a, u, A, B, t):
        """Solid terracotta discs: a few of the packed circles in CIRCLES. Returns a mask over idx + alpha."""
        _, _, ia, ib = self._sab
        if B.get("literal"):
            st, w = B, u * self._land(ib, t)
        elif A.get("literal"):
            st, w = A, (1 - u) * self._land(ia, t)
        else:
            return None
        hsel = (self.hash < 0.13) & (st["mode"] == 1)
        al = w[idx] * a[idx] * hsel[idx]
        return al

    # -- frame
    def frame(self, t, W, H, ss=2):
        g = self.geom(t, W, H, ss)
        return self.raster.render(g, self)

    @staticmethod
    def _land(idx, t):
        """Colour arrives on the resolve: snaps in around the landing of that state's phrase."""
        k = {1: 0, 3: 1, 5: 2, 6: 3}.get(idx)
        if k is None:
            return 1.0
        L = LANDINGS[k]
        return float(smoothstep(L - 0.12, L + 0.04, t))

    def render_frame(self, t, W, H):
        self._cam = camera(t)
        return self.frame(t, W, H)


class CvRaster:
    """Reference rasteriser (cv2 on the CPU; integer-bucketed values, LINE_AA strokes)."""

    def render(self, g, sc):
        CW, CH, s = g["CW"], g["CH"], g["s"]
        img = np.zeros((CH, CW), np.float32)
        col = np.zeros((3, CH, CW), np.float32)
        cal = np.zeros((CH, CW), np.float32)
        for ap, k, sp, tx, ty in g["photo"]:
            M = np.array([[sp, 0, tx], [0, sp, ty]], np.float32)
            img += ap * cv2.warpAffine(sc.sc["srcs"][k]["grey"], M, (CW, CH), flags=cv2.INTER_LINEAR,
                                       borderValue=0)
        lay = np.zeros((CH, CW), np.uint8)
        if g["lay_segs"] is not None:
            a, b, v, th = g["lay_segs"]
            for key in np.unique(np.c_[np.round(v * 255), th], axis=0):
                sel = (np.round(v * 255) == key[0]) & (th == key[1])
                segs = np.round(np.stack([a[sel], b[sel]], 1) * 8).astype(np.int32)
                cv2.polylines(lay, list(segs), False, int(key[0]), int(round(key[1] * g["k"])), cv2.LINE_AA, 3)
        for cx, cy, rr, a0, sw, v, th in g["lay_arcs"]:
            cv2.ellipse(lay, (int(cx * 16), int(cy * 16)), (int(rr * 16), int(rr * 16)), 0, a0, a0 + sw,
                        int(round(v * 255)), int(round(th * g["k"])), cv2.LINE_AA, 4)
        for cx, cy, rr, v in g["lay_discs"]:
            cv2.circle(lay, (int(cx * 16), int(cy * 16)), int(rr * 16), int(round(v * 255)), -1, cv2.LINE_AA, 4)
        img = np.maximum(img, lay.astype(np.float32) / 255.0)
        tsc, fv, fa = g["fill"]
        if len(tsc):
            F = np.zeros((CH, CW), np.uint8)
            Fa = np.zeros((CH, CW), np.uint8)
            qa = np.round(fa * 15).astype(int)
            qc = np.round(fv * 63).astype(int)
            key = qa * 64 + qc
            polys = np.round(tsc * 8).astype(np.int32)
            for kk in np.unique(key):
                sel = polys[key == kk]
                a_, c_ = divmod(int(kk), 64)
                cv2.fillPoly(Fa, list(sel), int(a_ * 17), cv2.LINE_8, 3)
                cv2.fillPoly(F, list(sel), int(c_ * 255 / 63), cv2.LINE_8, 3)
            img = img * (1 - Fa.astype(np.float32) / 255.0) + F.astype(np.float32) / 255.0
        if g["cfill"] is not None:
            tc, rgb, al = g["cfill"]
            polys = np.round(tc * 8).astype(np.int32)
            q = np.round(al * 8).astype(int)
            for lv in range(1, 9):
                sel = q == lv
                if sel.any():
                    pp = list(polys[sel])
                    for ch in range(3):
                        cv2.fillPoly(col[ch], pp, float(rgb[sel][0, ch] * lv / 8), cv2.LINE_8, 3)
                    cv2.fillPoly(cal, pp, float(lv / 8), cv2.LINE_8, 3)
        te, ev = g["edges"]
        if len(te):
            E = np.zeros((CH, CW), np.uint8)
            polys = np.round(te * 8).astype(np.int32)
            q = np.round(ev * 8).astype(int)
            for lv in range(1, 9):
                sel = polys[q == lv]
                if len(sel):
                    cv2.polylines(E, list(sel), True, int(lv * 255 / 8), int(round(g["k"])), cv2.LINE_AA, 3)
            img = np.maximum(img, E.astype(np.float32) / 255.0)
        if len(g["dots"]):
            D = np.zeros((CH, CW), np.uint8)
            for x, y, rr, v in g["dots"]:
                cv2.circle(D, (int(x * 16), int(y * 16)), int(rr * 16), int(255 * v), -1, cv2.LINE_AA, 4)
            img = np.maximum(img, D.astype(np.float32) / 255.0)
        if g["ridges"] is not None:
            a, b, v, clips = g["ridges"]
            R = np.zeros((CH, CW), np.uint8)
            q = np.round(v / 0.8 * 8).astype(int)
            segs = np.round(np.stack([a, b], 1) * 8).astype(np.int32)
            for lv in range(1, 9):
                sel = segs[q == lv]
                if len(sel):
                    cv2.polylines(R, list(sel), False, int(lv * 255 / 8 * 0.8), int(round(g["k"])), cv2.LINE_AA,
                                  3)
            clip = sum(wt * self._mask_scr(st, g) for wt, st in clips)
            img = np.maximum(img, R.astype(np.float32) / 255.0 * clip)
        if g["circles"] is not None:
            C, r, gv, av, th, disc = g["circles"]
            Fl = np.zeros((CH, CW), np.uint8)
            Ol = np.zeros((CH, CW), np.uint8)
            for k in np.argsort(-r):
                cc = (int(C[k, 0] * 16), int(C[k, 1] * 16))
                rr = int(r[k] * 16)
                cv2.circle(Fl, cc, rr, int(255 * gv[k]), -1, cv2.LINE_AA, 4)
                cv2.circle(Ol, cc, rr, int(255 * av[k]), int(round(th * g["k"])), cv2.LINE_AA, 4)
            img = np.maximum(img, Fl.astype(np.float32) / 255.0)
            img = np.maximum(img, Ol.astype(np.float32) / 255.0)
            if disc is not None:
                for k in np.nonzero(disc >= 0.05)[0]:
                    cc = (int(C[k, 0] * 16), int(C[k, 1] * 16))
                    rr = int(r[k] * 16)
                    for ch in range(3):
                        cv2.circle(col[ch], cc, rr, float(TERRA[ch] * disc[k]), -1, cv2.LINE_AA, 4)
                    cv2.circle(cal, cc, rr, float(disc[k]), -1, cv2.LINE_AA, 4)
        if g["lines"] is not None:
            a, b, v, th = g["lines"]
            Lr = np.zeros((CH, CW), np.uint8)
            q = np.round(v * 8).astype(int)
            segs = np.round(np.stack([a, b], 1) * 8).astype(np.int32)
            for lv in range(1, 9):
                sel = segs[q == lv]
                if len(sel):
                    cv2.polylines(Lr, list(sel), False, int(lv * 255 / 8), int(round(th * g["k"])), cv2.LINE_AA,
                                  3)
            img = np.maximum(img, Lr.astype(np.float32) / 255.0)
        img = np.clip(img, 0, 1)
        inv = g["inv"]
        if inv > 0:
            img = img * (1 - inv) + (1 - img) * inv
        rgb = np.repeat(img[..., None], 3, 2)
        if sc.colour:
            rgb = rgb * (1 - cal[..., None]) + np.moveaxis(col, 0, -1)
        out = cv2.resize(rgb, (g["W"], g["H"]), interpolation=cv2.INTER_AREA)
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    @staticmethod
    def _mask_scr(st, g):
        m = st["mask"]
        cam, s, CW, CH = g["cam"], g["s"], g["CW"], g["CH"]
        k = s / m.sc
        M = np.array([[k, 0, (m.off[0] - cam[0]) * s + CW / 2], [0, k, (m.off[1] - cam[1]) * s + CH / 2]],
                     np.float32)
        out = cv2.warpAffine(m.arr, M, (CW, CH), flags=cv2.INTER_LINEAR, borderValue=0)
        if st["kind"] == "T":
            out = cv2.dilate(out, np.ones((3, 3), np.uint8))
        return out


def cv_hw(th):
    """Half-width of an SDF stroke whose integral matches a cv2 LINE_AA stroke of int thickness th
    (measured: th 1 -> 1.35 px, 2 -> 3.45 px, 3-4 -> 5.5 px)."""
    th = np.asarray(th, float)
    return np.where(th <= 1, 0.68, np.where(th <= 2, 1.72, 2.75 + 0.5 * np.maximum(th - 4, 0)))


class TorchRaster:
    """GPU rasteriser (torch, float32; cpu or cuda). Everything is drawn implicitly per pixel:
    triangles by batched barycentric (edge-function) tests over bucketed bounding boxes, strokes by exact
    point-to-segment distance on pixels sampled densely along each segment, discs / rings / arcs as
    analytic SDFs over bucketed boxes, photo and masks by grid_sample, downsample by avg_pool. All layer
    writes are scatter_reduce(amax), which is order-independent (deterministic)."""

    def __init__(self, device="cpu"):
        import torch
        self.T = torch
        self.dev = torch.device(device)
        self.f32 = torch.float32
        self._tex = {}
        self._ar = torch.arange(4096, device=self.dev, dtype=self.f32)

    def t(self, x):
        return self.T.as_tensor(np.ascontiguousarray(x), dtype=self.f32, device=self.dev)

    def tex(self, key, arr):
        if key not in self._tex:
            self._tex[key] = self.t(arr)
        return self._tex[key]

    # -- primitives (buf: flat (CH*CW) float tensor)
    def _amax(self, buf, idx, val):
        buf.scatter_reduce_(0, idx, val, reduce="amax", include_self=True)

    def segs(self, bufs, vals, CH, CW, p0, p1, hw):
        """Anti-aliased strokes: coverage = clamp(hw + 0.5 - dist(pixel, segment)). Pixels are visited
        column by column along each segment's major axis (DDA band, no duplicates); bufs/vals: lists."""
        T = self.T
        if len(p0) == 0:
            return
        p0, p1, hw = self.t(p0), self.t(p1), self.t(hw)
        vals = [self.t(v) for v in vals]
        ext = hw + 0.5
        lo = T.minimum(p0, p1) - ext[:, None] - 1
        hi = T.maximum(p0, p1) + ext[:, None] + 1
        vis = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < CW) & (lo[:, 1] < CH)
        if not bool(vis.any()):
            return
        p0, p1, hw, ext = p0[vis], p1[vis], hw[vis], ext[vis]
        vals = [v[vis] for v in vals]
        d = p1 - p0
        L = d.norm(dim=1).clamp_min(1e-6)
        dv = d / L[:, None]
        swap = d[:, 1].abs() > d[:, 0].abs()                  # major axis y
        m0 = T.where(swap, p0[:, 1], p0[:, 0])
        n0 = T.where(swap, p0[:, 0], p0[:, 1])
        dm = T.where(swap, d[:, 1], d[:, 0])
        dn = T.where(swap, d[:, 0], d[:, 1])
        slope = dn / T.where(dm.abs() < 1e-6, T.ones_like(dm), dm)
        mlo = T.minimum(m0, m0 + dm)
        mhi = T.maximum(m0, m0 + dm)
        c0 = (mlo - ext).floor()
        ncol = ((mhi + ext).ceil() - c0 + 1).long()
        sid = T.repeat_interleave(T.arange(len(L), device=self.dev), ncol)
        start = T.cumsum(ncol, 0) - ncol
        col = c0[sid] + (T.arange(len(sid), device=self.dev) - start[sid]).to(self.f32)
        nc = n0[sid] + (T.minimum(T.maximum(col, mlo[sid]), mhi[sid]) - m0[sid]) * slope[sid]
        K = int(np.ceil((float(hw.max()) + 0.5) * 1.415)) + 1
        offs = T.arange(-K, K + 1, device=self.dev, dtype=self.f32)
        nn = nc.round()[:, None] + offs[None, :]
        mm = col[:, None].expand_as(nn)
        sw = swap[sid][:, None]
        px = T.where(sw, nn, mm)
        py = T.where(sw, mm, nn)
        qx = px - p0[sid, 0][:, None]
        qy = py - p0[sid, 1][:, None]
        tt = T.minimum(T.clamp(qx * dv[sid, 0][:, None] + qy * dv[sid, 1][:, None], min=0), L[sid][:, None])
        ex = qx - dv[sid, 0][:, None] * tt
        ey = qy - dv[sid, 1][:, None] * tt
        cov = (hw[sid][:, None] + 0.5 - T.sqrt(ex * ex + ey * ey)).clamp(0, 1)
        xi, yi = px.long(), py.long()
        ok = (xi >= 0) & (yi >= 0) & (xi < CW) & (yi < CH)
        cov = cov * ok
        idx = (yi.clamp(0, CH - 1) * CW + xi.clamp(0, CW - 1)).reshape(-1)
        for buf, v in zip(bufs, vals):
            self._amax(buf, idx, (v[sid][:, None] * cov).reshape(-1))

    _SIZES = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
                       3072, 4096])
    _SIZES2 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    def _boxes(self, lo, hi, CH, CW):
        """Bucket boxes by size -> list of (indices, bw, bh). CPU: (w, h) on a ~1.5x ladder (less waste);
        CUDA: power-of-two squares (fewer launches). One host transfer per call, no per-bucket syncs."""
        wh = ((hi - lo) + 2).cpu().numpy()
        l_, h_ = lo.cpu().numpy(), hi.cpu().numpy()
        vis = (h_[:, 0] >= 0) & (h_[:, 1] >= 0) & (l_[:, 0] < CW) & (l_[:, 1] < CH)
        if self.dev.type == "cuda":
            sz = self._SIZES2
            e = np.minimum(np.searchsorted(sz, wh.max(1)), len(sz) - 1)
            key = e * 64 + e
        else:
            sz = self._SIZES
            bx = np.minimum(np.searchsorted(sz, wh[:, 0]), len(sz) - 1)
            by = np.minimum(np.searchsorted(sz, wh[:, 1]), len(sz) - 1)
            key = bx * 64 + by
        key = np.where(vis, key, -1)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        out = []
        for kk in np.unique(ks[ks >= 0]):
            sel_np = order[np.searchsorted(ks, kk, "left"):np.searchsorted(ks, kk, "right")]
            bw, bh = int(sz[kk // 64]), int(sz[kk % 64])
            step = max(1, int(4_000_000 // (bw * bh)))
            for i in range(0, len(sel_np), step):
                out.append((self.T.as_tensor(sel_np[i:i + step], device=self.dev), bw, bh))
        return out

    def _grid(self, lo, sel, bw, bh):
        ar = self._ar
        x = lo[sel, 0][:, None, None] + ar[:bw][None, None, :]
        y = lo[sel, 1][:, None, None] + ar[:bh][None, :, None]
        return x, y

    def _scatter(self, bufs, vals, sel, x, y, cov, CH, CW):
        T = self.T
        xi, yi = x.expand_as(cov).long(), y.expand_as(cov).long()
        ok = (xi >= 0) & (yi >= 0) & (xi < CW) & (yi < CH)
        cov = cov * ok
        idx = (yi.clamp(0, CH - 1) * CW + xi.clamp(0, CW - 1)).reshape(-1)
        for buf, val in zip(bufs, vals):
            self._amax(buf, idx, (val[sel][:, None, None] * cov).reshape(-1))

    def tris(self, bufs, vals, CH, CW, tri):
        """Filled triangles (pixel centres at integer coords; inside = all edge functions >= 0)."""
        T = self.T
        if len(tri) == 0:
            return
        V = self.t(tri)
        vals = [self.t(v) for v in vals]
        lo = V.min(dim=1).values.floor()
        hi = V.max(dim=1).values.ceil()
        for sel, bw, bh in self._boxes(lo, hi, CH, CW):
            v = V[sel]
            x, y = self._grid(lo, sel, bw, bh)
            area = ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                    - (v[:, 1, 1] - v[:, 0, 1]) * (v[:, 2, 0] - v[:, 0, 0]))
            sg = T.where(area >= 0, 1.0, -1.0)[:, None, None]
            inside = None
            for i in range(3):
                a, b = v[:, i], v[:, (i + 1) % 3]
                e = ((b[:, 0] - a[:, 0])[:, None, None] * (y - a[:, 1][:, None, None])
                     - (b[:, 1] - a[:, 1])[:, None, None] * (x - a[:, 0][:, None, None]))
                ok = (e * sg) >= 0
                inside = ok if inside is None else inside & ok
            self._scatter(bufs, vals, sel, x, y, inside.to(self.f32), CH, CW)

    def circles(self, bufs, vals, CH, CW, c, r, hw=None, a0=None, sweep=None):
        """Discs (hw None): clamp(r + 0.5 - d); rings: clamp(hw + 0.5 - |d - r|); optional arc mask
        (start angle a0, sweep, degrees, image coords: clockwise on screen, as cv2.ellipse)."""
        T = self.T
        if len(c) == 0:
            return
        c, r = self.t(c), self.t(r)
        vals = [self.t(v) for v in vals]
        hwt = self.t(hw) if hw is not None else None
        ext = r + (hwt if hwt is not None else 0) + 2
        lo = (c - ext[:, None]).floor()
        hi = (c + ext[:, None]).ceil()
        a0t = self.t(a0) if a0 is not None else None
        swt = self.t(sweep) if sweep is not None else None
        for sel, bw, bh in self._boxes(lo, hi, CH, CW):
            x, y = self._grid(lo, sel, bw, bh)
            dx = x - c[sel, 0][:, None, None]
            dy = y - c[sel, 1][:, None, None]
            d = T.sqrt(dx * dx + dy * dy)
            rr = r[sel][:, None, None]
            if hwt is None:
                cov = (rr + 0.5 - d).clamp(0, 1)
            else:
                cov = (hwt[sel][:, None, None] + 0.5 - (d - rr).abs()).clamp(0, 1)
            if a0t is not None:
                ang = T.rad2deg(T.atan2(dy, dx))
                rel = T.remainder(ang - a0t[sel][:, None, None], 360.0)
                cov = cov * (rel <= swt[sel][:, None, None])
            self._scatter(bufs, vals, sel, x, y, cov, CH, CW)

    def warp(self, src, sp, tx, ty, CH, CW):
        """dst(x, y) = src((x - tx) / sp, (y - ty) / sp), bilinear, zero outside (= cv2.warpAffine)."""
        T = self.T
        h, w = src.shape
        xs = (T.arange(CW, device=self.dev, dtype=self.f32) - tx) / sp
        ys = (T.arange(CH, device=self.dev, dtype=self.f32) - ty) / sp
        gx = xs / (w - 1) * 2 - 1
        gy = ys / (h - 1) * 2 - 1
        grid = T.stack(T.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
        return T.nn.functional.grid_sample(src[None, None], grid, mode="bilinear", padding_mode="zeros",
                                           align_corners=True)[0, 0]

    def render(self, g, sc):
        T = self.T
        CW, CH = g["CW"], g["CH"]
        n = CH * CW
        z = lambda: T.zeros(n, device=self.dev, dtype=self.f32)
        img = T.zeros((CH, CW), device=self.dev, dtype=self.f32)
        for ap, k, sp, tx, ty in g["photo"]:
            img = img + ap * self.warp(self.tex(("photo", k), sc.sc["srcs"][k]["grey"]), sp, tx, ty, CH, CW)
        lay = z()
        if g["lay_segs"] is not None:
            a, b, v, th = g["lay_segs"]
            self.segs([lay], [v], CH, CW, a, b, cv_hw(th) * g["k"])
        if len(g["lay_arcs"]):
            A_ = g["lay_arcs"]
            self.circles([lay], [A_[:, 5]], CH, CW, A_[:, :2], A_[:, 2], hw=cv_hw(A_[:, 6]) * 0.94 * g["k"], a0=A_[:, 3],
                         sweep=A_[:, 4])
        if len(g["lay_discs"]):
            D_ = g["lay_discs"]
            self.circles([lay], [D_[:, 3]], CH, CW, D_[:, :2], D_[:, 2])
        img = T.maximum(img, lay.view(CH, CW))
        tsc, fv, fa = g["fill"]
        if len(tsc):
            F, Fa = z(), z()
            self.tris([F, Fa], [fv, fa], CH, CW, tsc)
            img = img * (1 - Fa.view(CH, CW)) + F.view(CH, CW)
        col = [z(), z(), z()]
        cal = z()
        if g["cfill"] is not None:
            tc, rgb, al = g["cfill"]
            self.tris(col + [cal], [rgb[:, 0] * al, rgb[:, 1] * al, rgb[:, 2] * al, al], CH, CW, tc)
        te, ev = g["edges"]
        if len(te):
            E = z()
            a = te.reshape(-1, 2)
            b = te[:, [1, 2, 0]].reshape(-1, 2)
            self.segs([E], [np.repeat(ev, 3)], CH, CW, a, b, np.full(len(a), float(cv_hw(1)) * g["k"]))
            img = T.maximum(img, E.view(CH, CW))
        if len(g["dots"]):
            Dl = z()
            d_ = g["dots"]
            self.circles([Dl], [d_[:, 3]], CH, CW, d_[:, :2], d_[:, 2])
            img = T.maximum(img, Dl.view(CH, CW))
        if g["ridges"] is not None:
            a, b, v, clips = g["ridges"]
            R = z()
            self.segs([R], [v], CH, CW, a, b, np.full(len(a), float(cv_hw(1)) * g["k"]))
            clip = sum(wt * self._mask_scr(st, g) for wt, st in clips)
            img = T.maximum(img, R.view(CH, CW) * clip)
        if g["circles"] is not None:
            C, r, gv, av, th, disc = g["circles"]
            Fl, Ol = z(), z()
            self.circles([Fl], [gv], CH, CW, C, r)
            self.circles([Ol], [av], CH, CW, C, r, hw=np.full(len(r), float(cv_hw(th)) * 0.94 * g["k"]))
            img = T.maximum(T.maximum(img, Fl.view(CH, CW)), Ol.view(CH, CW))
            if disc is not None and (disc >= 0.05).any():
                k = disc >= 0.05
                dd = disc[k]
                self.circles(col + [cal], [TERRA[0] * dd, TERRA[1] * dd, TERRA[2] * dd, dd], CH, CW, C[k], r[k])
        if g["lines"] is not None:
            a, b, v, th = g["lines"]
            Lr = z()
            self.segs([Lr], [v], CH, CW, a, b, np.full(len(a), float(cv_hw(th)) * g["k"]))
            img = T.maximum(img, Lr.view(CH, CW))
        img = img.clamp(0, 1)
        inv = g["inv"]
        if inv > 0:
            img = img * (1 - inv) + (1 - img) * inv
        rgb = img[None].expand(3, CH, CW)
        if sc.colour:
            ca = cal.view(CH, CW)
            rgb = rgb * (1 - ca)[None] + T.stack([c_.view(CH, CW) for c_ in col], 0)
        ss = CW // g["W"]
        out = T.nn.functional.avg_pool2d(rgb[None], ss)[0]
        out = (out.clamp(0, 1) * 255 + 0.5).to(T.uint8).permute(1, 2, 0)
        return out.cpu().numpy()

    def _mask_scr(self, st, g):
        m = st["mask"]
        cam, s, CW, CH = g["cam"], g["s"], g["CW"], g["CH"]
        out = self.warp(self.tex(("mask", id(m)), m.arr), s / m.sc, (m.off[0] - cam[0]) * s + CW / 2,
                        (m.off[1] - cam[1]) * s + CH / 2, CH, CW)
        if st["kind"] == "T":
            out = self.T.nn.functional.max_pool2d(out[None, None], 3, 1, 1)[0, 0]
        return out


# ---------------------------------------------------------------------------------------------- output
_SC = None


def _init(variant, colour, backend="numpy", device="cpu", threads=0):
    global _SC
    if threads:
        cv2.setNumThreads(threads)
        if backend == "torch":
            import torch
            torch.set_num_threads(threads)
    _SC = Scene(variant, colour, backend, device)


def _work(args):
    n, W, H = args
    return _SC.render_frame(n / FPS, W, H).tobytes()


def render(path, W, H, variant, colour, procs=4, backend="numpy", device="cpu", threads=0):
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")
    load_scene()
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with mp.Pool(procs, initializer=_init, initargs=(variant, colour, backend, device, threads)) as pool:
        for buf in pool.imap(_work, [(n, W, H) for n in range(NFR)], chunksize=2):
            proc.stdin.write(buf)
    t1 = time.perf_counter()
    proc.stdin.close()
    proc.wait()
    t2 = time.perf_counter()
    print(f"{path}  {W}x{H}  {NFR} frames  render {t1 - t0:.1f}s  encode-tail {t2 - t1:.1f}s")
    return t1 - t0, t2 - t1


def trace_sheet():
    sc = load_scene()
    tiles = []
    for s in sc["srcs"]:
        w, h = s["size"]
        im = np.zeros((h, w), np.float32) + 0.25 * s["grey"]
        lay = np.zeros((h, w), np.uint8)
        off = s["off"]
        cv2.polylines(lay, [np.round(p - off).astype(np.int32) for p in s["polys"]], False, 150, 1, cv2.LINE_AA)
        cv2.polylines(lay, [np.round(p - off).astype(np.int32) for p in s["sil"]], True, 255, 2, cv2.LINE_AA)
        for cx, cy, r, _ in s["circles"]:
            cv2.circle(lay, (int(cx - off[0]), int(cy - off[1])), int(r), 200, 1, cv2.LINE_AA)
        im = np.maximum(im, lay / 255.0)
        rgb = (np.repeat(im[..., None], 3, 2) * 255).astype(np.uint8)
        for x, y in s["pts"] - off:
            cv2.circle(rgb, (int(x), int(y)), 1, (60, 160, 255), -1)
        sc_ = 480 / h
        tiles.append(cv2.resize(rgb, (int(w * sc_), 480), interpolation=cv2.INTER_AREA))
    for st in sc["states"]:
        if st["kind"] == "T":
            m = (st["mask"].arr > 0.5).astype(np.uint8) * 60
            rgb = np.repeat(m[..., None], 3, 2)
            for x, y in (st["pts"] - st["mask"].off) * st["mask"].sc:
                cv2.circle(rgb, (int(x), int(y)), 2, (255, 255, 255), -1)
            tiles.append(cv2.resize(rgb, (384, 480), interpolation=cv2.INTER_AREA))
    hh = max(t.shape[0] for t in tiles)
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    wmax = max(row1.shape[1], row2.shape[1])
    pad = lambda r: np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)))
    os.makedirs(REVIEW, exist_ok=True)
    p = os.path.join(REVIEW, f"E3_trace_{cache_key()}.jpg")
    cv2.imwrite(p, np.vstack([pad(row1), pad(row2)]))
    print(p)


TIM_HDR = ("gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,"
           "encode_s,audio_s,notes\n")


def log_timing(nn, test, kind, device, frames, res, started, finished, make_min, render_s, encode_s, audio_s,
               notes):
    path = os.path.join(LOCAL, "TIMINGS.csv")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(TIM_HDR)
    fmt = lambda x, d=1: "" if x is None else f"{x:.{d}f}"
    pf = None if render_s is None or not frames else render_s * 1000 / frames
    row = [f"g{nn:02d}", test, kind, device, str(frames or ""), res, started, finished, fmt(make_min),
           fmt(render_s), fmt(pf, 0), fmt(encode_s), fmt(audio_s), notes.replace(",", ";")]
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")


def nz_now():
    return gens.now_nz()


SCORE_ARGS: list = []


def score_wav(test, nn):
    wav = os.path.join(LOCAL, f"score_{test}_g{nn:02d}.wav")
    if os.path.exists(wav):
        return wav, "score"
    ch = ",".join(f"{x:g}" for x in LANDINGS)
    sp = os.path.join(HERE, "score.py")
    if os.path.exists(sp):
        r = subprocess.run([sys.executable, sp, "--preset", "E3", "--duration", f"{DUR:g}", "--changes", ch,
                            "--out", wav] + SCORE_ARGS)
        if r.returncode == 0 and os.path.exists(wav):
            return wav, "score"
    wav = os.path.join(LOCAL, f"chords_{test}_g{nn:02d}.wav")
    if not os.path.exists(wav):
        subprocess.run([sys.executable, os.path.join(HERE, "chords.py"), "--duration", f"{DUR:g}",
                        "--changes", ch, "--out", wav], check=True)
    return wav, "chords"


def parity(a, test, W=360, H=450):
    """Reference (cv2) vs torch frames: PSNR, per-frame ms; side-by-side strip to review/."""
    ref = Scene(a.variant, a.colour, "numpy")
    tor = Scene(a.variant, a.colour, "torch", a.device)
    tor.render_frame(a.parity[0], W, H)                       # warm-up (kernels, textures)
    rows = []
    tr = tg = 0.0
    for t in a.parity:
        t0 = time.perf_counter()
        x = ref.render_frame(t, W, H)
        t1 = time.perf_counter()
        y = tor.render_frame(t, W, H)
        if tor.backend == "torch":
            tor.raster.T.cuda.synchronize() if tor.raster.dev.type == "cuda" else None
        t2 = time.perf_counter()
        tr += t1 - t0
        tg += t2 - t1
        mse = float(np.mean((x.astype(float) - y.astype(float)) ** 2))
        psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        print(f"t={t:.2f}  psnr {psnr:.1f} dB  numpy {1000 * (t1 - t0):.0f} ms  torch-{a.device} {1000 * (t2 - t1):.0f} ms")
        rows.append(np.hstack([x, y, np.clip(np.abs(x.astype(int) - y.astype(int)) * 4, 0, 255).astype(np.uint8)]))
    print(f"mean per frame: numpy {1000 * tr / len(a.parity):.0f} ms, torch-{a.device} {1000 * tg / len(a.parity):.0f} ms")
    fn = os.path.join(REVIEW, f"{test}_parity_{a.device}.jpg")
    Image.fromarray(np.vstack(rows)).save(fn, quality=88)
    print(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["a", "b"], default="a")
    ap.add_argument("--colour", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="per-process torch/cv2 threads (0 = default)")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                    help="numpy = cv2 reference rasteriser; torch = implicit GPU rasteriser")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parity", type=float, nargs="*", help="numpy vs torch frames at these t: PSNR + timing")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--work-start", type=float, default=None, help="epoch s when work on this gen began")
    ap.add_argument("--device-label", default="local-numpy")
    ap.add_argument("--score-args", default="", help="extra args for score.py, e.g. '--backend torch --device cuda'")
    a = ap.parse_args()
    SCORE_ARGS[:] = ["--timings", os.path.join(LOCAL, "TIMINGS.csv")] + a.score_args.split()
    test = {"a": "E3a_facet", "b": "E3b_voronoi"}[a.variant] + ("_col" if a.colour else "")
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    if a.trace:
        trace_sheet()
    h = gens.code_hash([os.path.abspath(__file__)])
    for kind, flag, (W, H) in (("preview", a.preview, (360, 450)), ("final", a.final, (720, 900))):
        if not flag:
            continue
        nn = gens.gen_for(LOCAL, test, test, kind, h, a.parent, a.trigger, a.change)
        name = gens.video_name(test, nn, kind)
        started = nz_now()
        dt, enc = render(os.path.join(LOCAL, name), W, H, a.variant, a.colour, a.procs, a.backend, a.device,
                         a.threads)
        sc = Scene(a.variant, a.colour, a.backend, a.device)
        stills = []
        for t in (3.30, 4.60):
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            Image.fromarray(sc.render_frame(t, 432, 540)).save(os.path.join(REVIEW, fn), quality=90)
            stills.append(fn)
        files = [name]
        audio_s = None
        if kind == "final":
            ta = time.perf_counter()
            wav, how = score_wav(test, nn)
            audio_s = time.perf_counter() - ta
            mux = f"{test}_g{nn:02d}_final_audio.mp4"
            if not os.path.exists(os.path.join(LOCAL, mux)):
                subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", os.path.join(LOCAL, name), "-i", wav,
                                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                                os.path.join(LOCAL, mux)], check=True)
            files += [os.path.basename(wav), mux]
            print("audio:", how)
        gens.add_files(LOCAL, test, nn, files, stills)
        mk = None if a.work_start is None else (time.time() - a.work_start) / 60
        log_timing(nn, test, kind, a.device_label, NFR, f"{W}x{H}", started, nz_now(), mk, dt, enc, audio_s,
                   f"procs={a.procs}; " + (a.change or ""))
        print(f"render {dt:.1f}s")
    if a.parity:
        parity(a, test)
    if a.still:
        sc = Scene(a.variant, a.colour, a.backend, a.device)
        for t in a.still:
            Image.fromarray(sc.render_frame(t, 360, 450)).save(os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))


if __name__ == "__main__":
    main()
