#!/usr/bin/env python
"""Math-type E5: growth mathematics resolving into type (Gray-Scott reaction-diffusion), 10 s, 4:5.

"Everything that grows keeps a quiet arithmetic, one small rule repeated until it becomes a living form."

Two chemicals U, V on a periodic canvas (600x750 canvas units, grid = canvas * --scale), Karl Sims'
3x3 Laplacian (centre -1, sides .2, diagonals .05) scaled by 1/h^2:
    dU/dt = Du lap(U) - U V^2 + F(x,t) (1 - U)
    dV/dt = Dv lap(V) + U V^2 - (F(x,t) + k(x,t)) V
F, k are a per-pixel parameter map: inside the letters a growth regime (coral F .0545 k .062, or
spots), outside a decay regime (F .02, k .08) where V dies. Letter masks are signed distance fields;
a word can morph between two faces by lerping their SDFs (tall face -> puffy face while it grows).
Growth starts from seeds deposited by a sweeping front; between phrases a vine (a narrow channel of
the growth regime, seeded at its head) grows from the old words into the new ones, the camera
follows it, and the old words' map is morphed to decay so their pattern breaks up and fades.
Variant a: coral letters, sweep-seeded, hard threshold. Variant b: spot letters, soft tone ramp.
Variant c (the hybrid): coral letters, hard threshold, and b's opener. Variants b and c open with a golden-angle phyllotaxis of seed spots (r = c sqrt(n), theta = n 137.508 deg).

GPU-first: every per-frame computation (parameter map, seeding, RD steps, camera resample, tone
curve, ground flip, accent discs) is torch on --device (cpu for local previews, cuda on the box).
CPU work is one-off setup only: glyph rasterising and the distance transforms for SDFs, seed-time
and vine-time maps.

Determinism exception (noted in the brief): RD is a time integration, so the simulation is stateful.
It is precomputed ONCE per sim-parameter set into a frame cache (local/cache/E5_sim_<hash>.npy: V and
1-U quantised to uint8 per frame). frame(t) is then a pure function of t: it reads cache[round(25 t)]
and applies the closed-form camera, tone curve and ground flip. Same seed and params -> same cache.

    python tools/mathtype/growth.py --variant a --preview                     # local, scale 1, cpu torch
    python tools/mathtype/growth.py --variant a --final --scale 2 --device cuda   # on the box
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mograph.font_registry import font_path  # noqa: E402  (read-only use)
import gens  # noqa: E402

VDIR = os.path.join(REPO, "11_mathtype", "v011")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")
CACHE = os.path.join(LOCAL, "cache")

SIM_VERSION = 5
FPS, DUR = 25, 10.0
NFR = int(round(FPS * DUR))
CW, CH = 600, 750                      # canvas (torus) in canvas units
VW, VH = 360.0, 450.0                  # camera window at zoom 1 (canvas units) = half of 720x900

# lines: (text, face) or (text, faceA, faceB) for an SDF morph A -> B while the phrase grows
FONTS_MIX = [
    [("EVERYTHING", "Big Shoulders Display"), ("THAT", "Le Murmure"), ("GROWS", "Big Shoulders Display", "Blackout")],
    [("KEEPS A", "Le Murmure"), ("QUIET", "Shrikhand"), ("ARITHMETIC", "Big Shoulders Display")],
    [("ONE SMALL", "Climate Crisis"), ("RULE", "Blackout"), ("REPEATED", "Le Murmure")],
    [("UNTIL IT", "Le Murmure"), ("BECOMES A", "Big Shoulders Display"), ("LIVING", "Blackout"),
     ("FORM", "Le Murmure", "Pilowlava")],
]
# SDF dilation per face (canvas units): fine faces get room for the RD stripes to hold their shape
BOLD = {"Le Murmure": 1.4, "Big Shoulders Display": 0.3}
LAND = [1.5, 3.9, 6.3, 8.6]            # fully grown, legible
READ = 1.2                             # read window after each landing
PCEN = [(300.0, 225.0), (440.0, 585.0), (270.0, 945.0), (430.0, 1305.0)]   # unwrapped canvas centres

OUT_FK = (0.020, 0.080)                # decay: V dies
REGIMES = {"coral": (0.0545, 0.062), "spots": (0.030, 0.0625), "mitosis": (0.0367, 0.0649)}

INK = (0x12, 0x12, 0x12)
BONE = (0xEE, 0xE8, 0xDC)
MOSS = (0x3F, 0x6B, 0x3A)
LICHEN = (0xB7, 0xC4, 0x8A)
SEED_OR = (0xE0, 0x7B, 0x39)


def ss(a, b, x):
    """smoothstep for floats, numpy arrays and torch tensors."""
    if torch.is_tensor(x):
        u = ((x - a) / (b - a)).clamp(0, 1)
    else:
        u = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (u * (6 * u - 15) + 10)


def zoom(t):
    return 1.0 + 0.022 * t                                  # slow push


def read_end(i):
    return LAND[i] + READ


def camera(t):
    c = np.array(PCEN[0], float) + np.array([0.0, -18.0]) * (1 - ease(t / LAND[0]))
    for i in range(3):
        t0, t1 = read_end(i) - 0.35, LAND[i + 1] - 0.05
        c = c + ease((t - t0) / (t1 - t0)) * (np.array(PCEN[i + 1]) - np.array(PCEN[i]))
    return c + np.array([6.0 * np.sin(0.7 * t), 4.0 * np.sin(0.45 * t + 1.0)])


# ---------------------------------------------------------------------------------------------- glyphs
def load_font(name, size):
    f = ImageFont.truetype(font_path(name), max(1, int(round(size))))
    if name == "Big Shoulders Display":
        f.set_variation_by_axes([900])
    elif name == "Facade":
        f.set_variation_by_axes([900, 0])
    elif name == "Climate Crisis":
        f.set_variation_by_axes([1979])
    return f


def line_metrics(text, face, width, capmax):
    """Font size so the ink spans `width` (cap height capped); returns size, ink w, cap h."""
    p = load_font(face, 200)
    bb = p.getbbox(text)
    hb = p.getbbox("H")
    size = 200 * width / (bb[2] - bb[0])
    size = min(size, 200 * capmax / (hb[3] - hb[1]))
    f = load_font(face, size)
    bb, hb = f.getbbox(text), f.getbbox("H")
    return size, bb[2] - bb[0], hb[3] - hb[1]


def raster_line(text, face, size, cx, ytop, s, S=3):
    """Glyph coverage on the full grid (CH*s, CW*s); line centred at cx, cap top at ytop (canvas units)."""
    f = load_font(face, size * s * S)
    bb, hb = f.getbbox(text), f.getbbox("H")
    w = (bb[2] - bb[0])
    im = Image.new("L", (CW * s * S, CH * s * S), 0)
    ImageDraw.Draw(im).text((cx * s * S - w / 2 - bb[0], ytop * s * S - hb[1]), text, font=f, fill=255)
    return np.asarray(im.resize((CW * s, CH * s), Image.BOX), np.float32) / 255


def sdf(cov, clampv):
    """Signed distance (grid px, negative inside) from a coverage image."""
    inside = (cov > 0.5).astype(np.uint8)
    di = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    do = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return np.clip(do - di, -clampv, clampv).astype(np.float32)


def phrase_sdfs(lines, width, s):
    """Block centred at the canvas centre: (static sdf, [(sdfA, sdfB)], boxes) on the grid."""
    capmax = 0.34 * width
    mets = []
    for ln in lines:
        mA = line_metrics(ln[0], ln[1], width, capmax)
        mB = line_metrics(ln[0], ln[2], width, capmax) if len(ln) > 2 else None
        mets.append((mA, mB))
    caps = [max(mA[2], mB[2] if mB else 0) for mA, mB in mets]
    gap = 0.17 * min(caps)
    y = CH / 2 - (sum(caps) + gap * (len(caps) - 1)) / 2
    cl = 12.0 * s
    static = np.full((CH * s, CW * s), cl, np.float32)
    morphs, boxes = [], []
    for ln, (mA, mB), cap in zip(lines, mets, caps):
        yA = y + (cap - mA[2]) / 2
        dA = sdf(raster_line(ln[0], ln[1], mA[0], CW / 2, yA, s), cl) - BOLD.get(ln[1], 0.0) * s
        if mB:
            yB = y + (cap - mB[2]) / 2
            dB = sdf(raster_line(ln[0], ln[2], mB[0], CW / 2, yB, s), cl) - BOLD.get(ln[2], 0.0) * s
            morphs.append((dA, dB))
        else:
            static = np.minimum(static, dA)
        w = max(mA[1], mB[1] if mB else 0)
        boxes.append((CW / 2 - w / 2, y, CW / 2 + w / 2, y + cap))
        y += cap + gap
    return static, morphs, boxes


def roll_to(img, dx, dy, s):
    return np.roll(img, (int(round(dy * s)) % (CH * s), int(round(dx * s)) % (CW * s)), axis=(0, 1))


# ---------------------------------------------------------------------------------------------- plan
class Plan:
    """One-off setup on the CPU (rasterising, distance transforms), then tensors on the device."""

    def __init__(self, a, dev):
        self.a, self.dev, s = a, dev, a.scale
        self.s = s
        rng = np.random.default_rng(a.seed)
        self.phr, self.boxes = [], []
        cov_for_seeds = []
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        for i, lines in enumerate(self.lines()):
            w = 0.88 * VW / zoom(LAND[i])
            st, mo, bx = phrase_sdfs(lines, w, s)
            dx, dy = PCEN[i][0] - CW / 2, PCEN[i][1] - CH / 2
            st = roll_to(st, dx, dy, s)
            mo = [(roll_to(A, dx, dy, s), roll_to(B, dx, dy, s)) for A, B in mo]
            self.boxes.append([(b0 + dx, b1 + dy, b2 + dx, b3 + dy) for b0, b1, b2, b3 in bx])
            self.phr.append((T(st), [(T(A), T(B)) for A, B in mo]))
            u = st.copy()                                   # seeds go where either face has ink
            for A, B in mo:
                u = np.minimum(u, np.minimum(A, B))
            cov_for_seeds.append(u)
        self.gin = REGIMES[a.regime]
        # ---- seed activation-time map (min over seeds); inf = never seeded
        G = np.full((CH * s, CW * s), np.inf, np.float32)
        seeds = []
        self.entry = []
        for i in range(4):
            if i == 0:
                ent = np.array([self.boxes[0][0][0] - 20.0, PCEN[0][1]]) if a.variant == "a" else np.array(PCEN[0])  # noqa: E501
            else:
                ent = self.vine_path(i - 1)[-1]
            self.entry.append(ent)
            pts = self.stratified(cov_for_seeds[i], a.seed_sp, rng, PCEN[i])
            dist = np.hypot(pts[:, 0] - ent[0], pts[:, 1] - ent[1])
            sn = (dist - dist.min()) / (np.ptp(dist) + 1e-6)
            sn = np.clip(sn + rng.normal(0, 0.04, len(sn)), 0, 1)
            g0 = {"a": 0.15, "b": 0.95, "c": 0.75}[a.variant] if i == 0 else read_end(i - 1) + 0.30
            ta = g0 + (LAND[i] - 0.55 - g0) * sn
            ta = np.maximum(ta, self.cover_time(i, pts))
            seeds.append(np.c_[pts, ta, np.full(len(pts), 1.0)])
        self.phyllo = None
        if a.variant in ("b", "c"):
            n = np.arange(1, a.phyllo_n + 1)
            r = a.phyllo_c * np.sqrt(n)
            th = np.deg2rad(137.508) * n
            P = np.c_[PCEN[0][0] + r * np.cos(th), PCEN[0][1] + r * np.sin(th)]
            self.phyllo = np.c_[P, 0.05 + 0.85 * (n / n[-1]) ** 0.8, np.full(len(n), 1.6)]
            seeds.insert(0, self.phyllo)
        seeds = np.concatenate(seeds, 0)
        for x, y, ta, r in seeds[np.argsort(-seeds[:, 2])]:        # latest first, earliest wins
            for ox in (-CW, 0, CW):
                for oy in (-CH, 0, CH):
                    cx, cy = (x % CW + ox) * s, (y % CH + oy) * s
                    if -8 * s < cx < (CW + 8) * s and -8 * s < cy < (CH + 8) * s:
                        cv2.circle(G, (int(round(cx)), int(round(cy))), int(round(r * s)), float(ta), -1)
        # ---- vines: distance-to-path and head-passage-time maps
        self.vines = []
        for i in range(3):
            dmap, tmap = self.vine_maps(i)
            hit = (dmap <= 1.0 * s) & (tmap < G)
            G = np.where(hit, tmap, G)                                  # the head deposits seeds
            self.vines.append((T(dmap), T(tmap)))
        self.seedtime = T(G)

    def cover_time(self, i, pts):
        """First time the (morphing) letter mask of phrase i covers each point; inf if never."""
        s = self.s
        st, mo = self.phr[i]
        iy = torch.from_numpy(np.mod(np.round(pts[:, 1] * s).astype(int), CH * s))
        ix = torch.from_numpy(np.mod(np.round(pts[:, 0] * s).astype(int), CW * s))
        out = np.full(len(pts), np.inf)
        for t in np.arange(0.0, DUR, 0.04)[::-1]:
            d = st[iy, ix]
            for A, B in mo:
                u = self.morph_u(i, t)
                d = torch.minimum(d, A[iy, ix] + (B[iy, ix] - A[iy, ix]) * u)
            out = np.where(d.cpu().numpy() <= -1.0 * s, t, out)
        return out

    def lines(self):
        if self.a.fonts == "mix":
            return FONTS_MIX
        return [[(ln[0], self.a.fonts) for ln in ph] for ph in FONTS_MIX]

    def stratified(self, d, sp, rng, cen):
        s = self.s
        gx, gy = np.meshgrid(np.arange(0, CW, sp), np.arange(0, CH, sp))
        x = (gx + rng.random(gx.shape) * sp).ravel()
        y = (gy + rng.random(gy.shape) * sp).ravel()
        keep = d[np.clip((y * s).astype(int), 0, CH * s - 1), np.clip((x * s).astype(int), 0, CW * s - 1)] <= -1.2 * s
        x, y = x[keep], y[keep]
        x = x + CW * np.round((cen[0] - x) / CW)
        y = y + CH * np.round((cen[1] - y) / CH)
        return np.c_[x, y]

    def exit_pt(self, i):
        b = self.boxes[i]
        nxt = np.array(PCEN[i + 1])
        cands = [np.array([b[-1][0] + 8, b[-1][3] - 4]), np.array([b[-1][2] - 8, b[-1][3] - 4])]
        return min(cands, key=lambda p: np.hypot(*(p - nxt)))

    def vine_path(self, i, n=600):
        """S-curve cubic Bezier from phrase i's last line down into phrase i+1's top line."""
        p0 = self.exit_pt(i)
        b = self.boxes[i + 1][0]
        p3 = np.array([b[0] + 10.0 if p0[0] < PCEN[i + 1][0] else b[2] - 10.0, (b[1] + b[3]) / 2])
        p1 = p0 + np.array([0.0, 0.55 * (p3[1] - p0[1])])
        p2 = p3 + np.array([0.0, -0.55 * (p3[1] - p0[1])])
        u = np.linspace(0, 1, n)[:, None]
        return ((1 - u) ** 3) * p0 + 3 * ((1 - u) ** 2) * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3

    def vine_window(self, i):
        return LAND[i] + 0.55, read_end(i) + 0.35       # head leaves, head arrives

    def vine_maps(self, i):
        """Distance (grid px) to the vine centreline and the time the head passed the nearest point."""
        s = self.s
        P = self.vine_path(i)
        t0, t1 = self.vine_window(i)
        tp = t0 + (t1 - t0) * np.linspace(0, 1, len(P))
        m = 12
        x0, y0 = np.floor(P.min(0)) - m
        w, h = (np.ceil(P.max(0)) + m - np.array([x0, y0])).astype(int)
        img = np.ones((h * s, w * s), np.uint8)
        tim = np.zeros((h * s, w * s), np.float32)
        for (x, y), tt in zip(P, tp):
            ix, iy = int(round((x - x0) * s)), int(round((y - y0) * s))
            if 0 <= ix < w * s and 0 <= iy < h * s and img[iy, ix]:
                img[iy, ix] = 0
                tim[iy, ix] = tt
        dist, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zy, zx = np.nonzero(img == 0)
        lut = np.zeros(lab.max() + 1, np.float32)
        lut[lab[zy, zx]] = tim[zy, zx]
        tl = lut[lab]
        D = np.full((CH * s, CW * s), 1e4, np.float32)
        Tm = np.full((CH * s, CW * s), 1e4, np.float32)
        yy = (np.arange(h * s) + int(y0 * s)) % (CH * s)
        xx = (np.arange(w * s) + int(x0 * s)) % (CW * s)
        D[np.ix_(yy, xx)] = dist
        Tm[np.ix_(yy, xx)] = tl
        return D, Tm

    def weights(self, t):
        w = []
        for i in range(4):
            on = ss(0.0, 0.2, t - (0.0 if i == 0 else read_end(i - 1) + 0.1))
            off = 1 - ss(read_end(i) + 0.15, read_end(i) + 0.85, t) if i < 3 else 1.0
            w.append(float(on * off))
        return w

    def morph_u(self, i, t):
        """Face morph progress for phrase i: tall -> puffy across its growth."""
        g0 = 0.3 if i == 0 else read_end(i - 1) + 0.3
        return float(ease((t - g0) / (LAND[i] - 0.45 - g0)))

    def mask(self, i, t):
        st, mo = self.phr[i]
        d = st
        for A, B in mo:
            u = self.morph_u(i, t)
            d = torch.minimum(d, A + (B - A) * u)
        return ss(0.8 * self.s, -0.8 * self.s, d)

    def param_map(self, t):
        w = self.weights(t)
        Wm = torch.zeros_like(self.seedtime)
        for i in range(4):
            if w[i] > 0:
                Wm = torch.maximum(Wm, w[i] * self.mask(i, t))
        s = self.s
        for dmap, tmap in self.vines:
            band = ss(7.0 * s, 5.5 * s, dmap) * (tmap <= t) * (1 - ss(1.0, 1.6, t - tmap))
            Wm = torch.maximum(Wm, band)
        fin, kin = self.gin
        Fm = OUT_FK[0] + (fin - OUT_FK[0]) * Wm
        Km = OUT_FK[1] + (kin - OUT_FK[1]) * Wm
        if self.phyllo is not None:
            hold = (1 - float(ss(1.0, 1.5, t))) * (1 - Wm)
            Fm = Fm + (REGIMES["spots"][0] - Fm) * hold
            Km = Km + (REGIMES["spots"][1] - Km) * hold
        return Fm, Km


# ---------------------------------------------------------------------------------------------- simulation
def sim_key(a):
    d = dict(v=SIM_VERSION, variant=a.variant, regime=a.regime, fonts=a.fonts, seed=a.seed, sp=a.seed_sp,
             steps=a.steps, ds=a.dscale, pn=a.phyllo_n, pc=a.phyllo_c, scale=a.scale)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def simulate(plan, a, dev):
    """Integrate once on the device; returns a uint8 cache (NFR, 2, H, W): V*400 and (1-U)*320."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"E5_sim_{sim_key(a)}.npy")
    if os.path.exists(path):
        print("sim cache", path)
        return np.load(path, mmap_mode="r"), 0.0
    s = a.scale
    H, W = CH * s, CW * s
    dt = 1.0 / s                                       # Du*dt stays <= .64 at scale 2
    nsteps = a.steps * s
    out = np.zeros((NFR, 2, H, W), np.uint8)
    X = torch.zeros(2, H, W, device=dev)
    X[0] = 1
    P = torch.empty(2, H + 2, W + 2, device=dev)
    D = torch.tensor([1.0, 0.5], device=dev).view(2, 1, 1) * (a.dscale * s * s * dt)
    T0 = time.perf_counter()
    with torch.inference_mode():
        for f in range(NFR):
            t, tp = f / FPS, (f - 1) / FPS
            Fm, Km = plan.param_map(t)
            Fd = Fm * dt
            FKd = (Fm + Km) * dt
            st = (plan.seedtime > tp) & (plan.seedtime <= t)
            X[1][st] = 0.9
            X[0][st] = 0.3
            for _ in range(nsteps):
                P[:, 1:-1, 1:-1] = X
                P[:, 0, 1:-1] = X[:, -1]
                P[:, -1, 1:-1] = X[:, 0]
                P[:, :, 0] = P[:, :, -2]
                P[:, :, -1] = P[:, :, 1]
                S = P[:, :-2, 1:-1] + P[:, 2:, 1:-1] + P[:, 1:-1, :-2] + P[:, 1:-1, 2:]
                Dg = P[:, :-2, :-2] + P[:, :-2, 2:] + P[:, 2:, :-2] + P[:, 2:, 2:]
                L = S.mul_(0.2).add_(Dg, alpha=0.05).sub_(X).mul_(D)
                u, v = X[0], X[1]
                uvv = u * v * v * dt
                L[0].sub_(uvv).add_(Fd - Fd * u)
                L[1].add_(uvv).sub_(FKd * v)
                X.add_(L)
            q = torch.stack([(X[1] * 400).clamp(0, 255), ((1 - X[0]) * 320).clamp(0, 255)]).to(torch.uint8)
            out[f] = q.cpu().numpy()
            if f % 25 == 0:
                print(f"sim frame {f}/{NFR}  {time.perf_counter() - T0:.0f}s", flush=True)
    sim_s = time.perf_counter() - T0
    np.save(path, out)
    print(f"sim done {sim_s:.0f}s -> {path}")
    return out, sim_s


# ---------------------------------------------------------------------------------------------- render
def gauss_kernel(sig, dev):
    r = max(1, int(3 * sig))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum(), r


class Renderer:
    def __init__(self, plan, cache, a, W, H, dev):
        self.plan, self.cache, self.a, self.W, self.H, self.dev = plan, cache, a, W, H, dev
        self.s = a.scale
        self.xs = ((torch.arange(W, device=dev) + 0.5) / W - 0.5).view(1, W)
        self.ys = ((torch.arange(H, device=dev) + 0.5) / H - 0.5).view(H, 1)
        self.k, self.r = gauss_kernel(a.plate_blur * self.s, dev)
        c = lambda v: torch.tensor(v, device=dev, dtype=torch.float32).view(1, 1, 3) / 255  # noqa: E731
        self.INK, self.BONE, self.MOSS, self.LICHEN, self.SEED = c(INK), c(BONE), c(MOSS), c(LICHEN), c(SEED_OR)

    def coords(self, t):
        cx, cy = camera(t)
        z = zoom(t)
        X = float(cx) + self.xs * (VW / z)
        Y = float(cy) + self.ys * (VH / z)
        return X.expand(self.H, self.W), Y.expand(self.H, self.W)

    def sample(self, img, X, Y, pad=3):
        """Bicubic resample of the periodic grid at canvas coords (X, Y)."""
        s = self.s
        Hg, Wg = img.shape[-2:]
        p = TF.pad(img[None], (pad, pad, pad, pad), mode="circular")
        gx = torch.remainder(X * s - 0.5, Wg) + pad + 0.5
        gy = torch.remainder(Y * s - 0.5, Hg) + pad + 0.5
        g = torch.stack([gx / (Wg + 2 * pad) * 2 - 1, gy / (Hg + 2 * pad) * 2 - 1], -1)[None]
        return TF.grid_sample(p, g, mode="bicubic", padding_mode="border", align_corners=False)[0]

    def blur(self, x):
        k, r = self.k, self.r
        x = TF.pad(x[None, None], (r, r, 0, 0), mode="circular")
        x = TF.conv2d(x, k.view(1, 1, 1, -1))
        x = TF.pad(x, (0, 0, r, r), mode="circular")
        return TF.conv2d(x, k.view(1, 1, -1, 1))[0, 0]

    def flip(self, t, X, Y, px):
        inv = torch.zeros_like(X)
        for i in self.a.flips:
            e = self.plan.entry[i]
            t0 = read_end(i - 1) + 0.30
            r = 900 * ease((t - t0) / 1.0) ** 1.4
            if r <= 0:
                continue
            dx, dy = X - float(e[0]), Y - float(e[1])
            d = torch.hypot(dx, dy)
            th = torch.atan2(dy, dx)
            wob = (0.10 * torch.sin(5 * th + 1.3 + 2.0 * t) + 0.06 * torch.sin(9 * th - 0.7 - 3.1 * t)
                   + 0.035 * torch.sin(17 * th + 2.2)) * min(r, 160.0)
            m = ss(r + wob + 1.5 * px, r + wob - 1.5 * px, d)
            inv = inv + m - 2 * inv * m          # soft XOR
        return inv

    def frame(self, t):
        f = int(min(NFR - 1, max(0, round(t * FPS))))
        fr = torch.from_numpy(np.ascontiguousarray(self.cache[f])).to(self.dev).float()
        V, Dp = fr[0] / 400, self.blur(fr[1] / 320)
        X, Y = self.coords(t)
        S = self.sample(torch.stack([V, Dp]), X, Y)
        V, Dp = S[0], S[1]
        px = VW / zoom(t) / self.W                                    # canvas units per output px
        a = self.a
        th = 0.20
        e = (0.012 if a.tone == "hard" else 0.07) + 0.02 * px
        ink = ss(th - e, th + e, V)[..., None]
        bw = 0.02 + 0.02 * px
        band = ss(a.plate_th - bw, a.plate_th + bw, Dp)[..., None]                # one flat posterised plate
        tone = (1 - band) + a.plate_grey * band                     # tonal value, 1 = ground
        tone = tone * (1 - ink)
        inv = self.flip(t, X, Y, px)[..., None]
        tone = tone * (1 - inv) + (1 - tone) * inv
        if not a.colour:
            img = tone.expand(self.H, self.W, 3)
        else:                                                        # tone -> ink..bone, accents after
            img = self.INK + (self.BONE - self.INK) * tone
            cm = float(ss(LAND[3] - 0.9, LAND[3] - 0.2, t))
            if cm > 0:                                     # "a living form": lichen plate under FORM only
                b = self.plan.boxes[3][-1]
                inbox = ss(b[1] - 8, b[1] - 4, Y) * ss(b[3] + 8, b[3] + 4, Y)
                pm = band * (1 - ink) * (1 - inv) * cm * inbox[..., None]
                img = img * (1 - pm) + self.LICHEN * pm
            if self.plan.phyllo is not None:
                img = self.seed_discs(img, t, X, Y, px)
        return (img.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()

    def seed_discs(self, img, t, X, Y, px):
        """Analytic SDF discs in seed orange on every 7th phyllotaxis seed, while the spiral is laid."""
        P = self.plan.phyllo[::7]
        vis = ss(0.0, 0.08, t - P[:, 2]) * (1 - ss(0.95, 1.35, t))
        P = P[vis > 0.5]
        if len(P) == 0:
            return img
        cx = torch.tensor(P[:, 0], device=self.dev, dtype=torch.float32).view(-1, 1, 1)
        cy = torch.tensor(P[:, 1], device=self.dev, dtype=torch.float32).view(-1, 1, 1)
        d = torch.hypot(X[None] - cx, Y[None] - cy).amin(0)
        cov = ss(2.8 + 0.7 * px, 2.8 - 0.7 * px, d)[..., None]
        return img * (1 - cov) + self.SEED * cov


# ---------------------------------------------------------------------------------------------- io
TIM_HDR = "gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,encode_s,audio_s,notes"


def nz_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("Pacific/Auckland"))
    except Exception:
        return datetime.datetime.now()


def log_timing(row):
    p = os.path.join(LOCAL, "TIMINGS.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        if new:
            f.write(TIM_HDR + "\n")
        csv.writer(f).writerow(row)


def code_and_args_hash(a):
    h = gens.code_hash([os.path.abspath(__file__)])
    skip = ("preview", "final", "stills", "threads", "change", "trigger", "parent", "audio", "device", "work_start")
    d = {k: v for k, v in vars(a).items() if k not in skip}
    return h + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["a", "b", "c"], default="c")
    ap.add_argument("--regime", choices=list(REGIMES), default=None)
    ap.add_argument("--fonts", default="mix", help="'mix' (per-line faces + morphs) or one face name")
    ap.add_argument("--tone", choices=["hard", "soft"], default=None)
    ap.add_argument("--colour", action="store_true")
    ap.add_argument("--flips", type=int, nargs="*", default=[2, 3], help="phrases whose arrival flips the ground")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--seed-sp", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=110, help="RD steps per frame at scale 1")
    ap.add_argument("--dscale", type=float, default=0.32)
    ap.add_argument("--plate-blur", type=float, default=2.6, help="sigma (canvas units) of the 1-U plate")
    ap.add_argument("--plate-th", type=float, default=0.16)
    ap.add_argument("--plate-grey", type=float, default=0.80)
    ap.add_argument("--scale", type=int, default=2, help="grid = canvas * scale (1 = 600x750, 2 = 1200x1500)")
    ap.add_argument("--phyllo-n", type=int, default=520)
    ap.add_argument("--phyllo-c", type=float, default=6.8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="torch device: cpu | cuda")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--stills", type=float, nargs="*", default=None)
    ap.add_argument("--audio", default=None, help="wav to mux into the final")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--work-start", default=None, help="NZ 'YYYY-mm-dd HH:MM' when work on this gen began")
    a = ap.parse_args()
    a.regime = a.regime or ("spots" if a.variant == "b" else "coral")
    a.tone = a.tone or ("soft" if a.variant == "b" else "hard")
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    started = nz_now()
    plan = Plan(a, dev)
    cache, sim_s = simulate(plan, a, dev)
    test = f"E5{a.variant}_growth{'_col' if a.colour else ''}"
    W, H = (720, 900) if a.final else (360, 450)
    kind = "final" if a.final else "draft"
    if a.stills is not None and not (a.preview or a.final):
        R = Renderer(plan, cache, a, 360, 450, dev)
        with torch.inference_mode():
            for t in a.stills:
                cv2.imwrite(os.path.join(REVIEW, f"_tmp_{test}_{t:.2f}.png"), R.frame(t)[..., ::-1])
        return
    nn = gens.gen_for(LOCAL, test, test, kind, code_and_args_hash(a), parent=a.parent, trigger=a.trigger,
                      change=a.change)
    name = gens.video_name(test, nn, kind)
    path = os.path.join(LOCAL, name)
    R = Renderer(plan, cache, a, W, H, dev)
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
           path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rend = enc = 0.0
    with torch.inference_mode():
        for f in range(NFR):
            t0 = time.perf_counter()
            fr = R.frame(f / FPS)
            t1 = time.perf_counter()
            p.stdin.write(fr.tobytes())
            enc += time.perf_counter() - t1
            rend += t1 - t0
        t1 = time.perf_counter()
        p.stdin.close()
        p.wait()
        enc += time.perf_counter() - t1
    print(f"rendered {path}: sim {sim_s:.0f}s, frames {rend:.1f}s, encode {enc:.1f}s")
    files, stills = [name], []
    R2 = Renderer(plan, cache, a, 360, 450, dev) if W > 540 else R
    with torch.inference_mode():
        for t in (read_end(1) + 0.45, LAND[2] + 0.3):
            sn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            cv2.imwrite(os.path.join(REVIEW, sn), R2.frame(t)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            stills.append(sn)
    if a.audio and a.final:
        an = f"{test}_g{nn:02d}_final_audio.mp4"
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", path, "-i", a.audio, "-c:v", "copy", "-c:a",
                        "aac", "-b:a", "192k", "-shortest", os.path.join(LOCAL, an)], check=True)
        files.append(an)
    gens.add_files(LOCAL, test, nn, files, stills)
    fin = nz_now()
    devname = ("cuda-" + torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace(" ", "")) \
        if dev.type == "cuda" else ("box-cpu" if os.path.isdir("/workspace") else "local-cpu-torch")
    make_min = ""
    if a.work_start:
        ws = datetime.datetime.strptime(a.work_start, "%Y-%m-%d %H:%M").replace(tzinfo=fin.tzinfo)
        make_min = f"{(fin - ws).total_seconds() / 60:.1f}"
    total = rend + sim_s
    log_timing([f"g{nn:02d}", test, "final" if a.final else "preview", devname, NFR, f"{W}x{H}",
                started.strftime("%Y-%m-%d %H:%M"), fin.strftime("%Y-%m-%d %H:%M"), make_min, f"{total:.1f}",
                f"{total * 1000 / NFR:.0f}", f"{enc:.1f}", "",
                f"sim {sim_s:.1f}s (0 = cached) + frames {rend:.1f}s; grid {CW * a.scale}x{CH * a.scale}"])
    print("gen", nn, files, stills)


if __name__ == "__main__":
    main()
