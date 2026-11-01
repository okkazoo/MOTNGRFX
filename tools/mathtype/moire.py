#!/usr/bin/env python
"""Math-type E4: interference / moire typography (10 s, 4:5, greyscale).

"Look closely: the pattern was always there, waiting in the noise for someone patient enough to let it
settle."  The words are invisible in either grating alone; they exist only as a phase offset in layer 2
and appear through interference with layer 1.

    phi1(p, t) = 2 pi f B(q)                                     + N1(q, t)
    phi2(p, t) = 2 pi f B(C + s(t) R(dth(t)) (q - C) + d(t))     + N2(q, t) + pi [pol(t) + sum_k a_k(t) M_k(p)]
    g_i        = square(phi_i)  (analytic AA: clip(1/2 + sin(phi) / |grad phi|), at 2x, then 2x2 box filter)
    frame      = g1 XOR g2 = g1 + g2 - 2 g1 g2       -> low-pass = triangle(phi1 - phi2)

q is world space (camera drift / zoom / roll over an endless grating field); M_k is the blurred phrase mask
in screen space; B(q) = q.n (variant "lin", linear grating) or |q - Cr| (variant "ring", concentric ring
grating; offset rings give hyperbolic fringes, scale moire gives flowing rings). N_i is the noise: per-cell
random grating angle and phase (a stochastic line screen) plus smooth warp, scaled by A(p, t), a damped
spring per event with a radial stagger, so the screen orders itself outward and the interference locks.
Between phrases dth / s / d bump (travelling moire bands), noise is partly re-injected and pol flips the
ground. Everything is closed form in t.

    python tools/mathtype/moire.py --variant lin --draft     # 360x450 -> v010/local/E4a_moire_gNN_draft.mp4
    python tools/mathtype/moire.py --variant ring --draft    # -> E4b_rings_gNN_draft.mp4
    python tools/mathtype/moire.py --variant lin --final     # 720x900 + audio -> _final.mp4, _final_audio.mp4
    python tools/mathtype/moire.py --variant lin --still 1.5 2.9 --size 360
    python tools/mathtype/moire.py --variant lin --qc        # contact sheet + transition strips (review/)
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from mograph.font_registry import font_axes, font_path  # noqa: E402  (read-only use)
import xp  # noqa: E402

VDIR = os.path.join(REPO, "11_mathtype", "v010")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

DW, DH = 720.0, 900.0                 # design space (= final size); every length below is in design px
FPS, DUR = 25, 10.0
SS = 2                                # supersample factor (box filter down)

TEXT_W = 600.0                        # ink width of each line
MAX_CAP = 150.0
TRACK = 0.03
LINE_GAP = 0.32                       # x cap height
EDGE = 2.4                            # design px; phase ramp width at the glyph edge (on the SDF)
MASK_GROW = 0.018                     # screen-space mask zoom per second (type drifts towards the viewer)
SHIMMER = 0.065                       # layer-2 bend (periods) during holds: a travelling wave in the transparency
SHIM_L, SHIM_V = 280.0, 120.0         # its wavelength and speed (design px, px/s)
COBALT = (0x2D, 0x4B, 0xD8)           # --colour: the one accent, flat blocks where the pattern locks
CBLOCK = 20.0                         # lock-block size (design px): the key word fills block by block

PERIOD = 9.0                          # grating period (design px)
CELL = 15.0                           # noise cell (stochastic screen) size
WARP_L = 90.0                         # smooth warp scale
ZETA, OMEGA = 0.5, 2 * math.pi * 1.2
GROUND = 0.12                         # ground phase offset (x pi) at rest: hairlines, not solid
LETTER = 0.76                         # glyph phase amplitude (x pi): letter stripes of duty GROUND + LETTER

# phrases: lines (text, font, morph-to font or None, cap ceiling), landing time, morph window, mask drift dir.
# "bungee": one face (g01-g02). "mix": heavy key words against tall lines, SDF morphs between faces.
FONT_SETS = {
    "bungee": [
        dict(lines=[("LOOK", "Bungee", None, MAX_CAP), ("CLOSELY", "Bungee", None, MAX_CAP)], land=1.35),
        dict(lines=[("THE PATTERN", "Bungee", None, MAX_CAP), ("WAS ALWAYS", "Bungee", None, MAX_CAP),
                    ("THERE", "Bungee", None, MAX_CAP)], land=3.35),
        dict(lines=[("WAITING", "Bungee", None, MAX_CAP), ("IN THE NOISE", "Bungee", None, MAX_CAP)], land=5.65),
        dict(lines=[("FOR SOMEONE", "Bungee", None, MAX_CAP), ("PATIENT ENOUGH", "Bungee", None, MAX_CAP),
                    ("TO LET IT", "Bungee", None, MAX_CAP), ("SETTLE", "Bungee", None, MAX_CAP)], land=7.85),
    ],
    "mix": [
        dict(lines=[("LOOK", "Blackout", None, 230), ("CLOSELY", "Le Murmure", None, 150)], land=1.35, key=0),
        dict(lines=[("THE", "Facade", None, 70), ("PATTERN", "Climate Crisis", None, 150),
                    ("WAS ALWAYS THERE", "Le Murmure", None, 95)], land=3.35, key=1),
        dict(lines=[("WAITING IN THE", "Le Murmure", None, 120), ("NOISE", "Shrikhand", None, 230)], land=5.65, key=1),
        dict(lines=[("FOR SOMEONE PATIENT", "Le Murmure", None, 88), ("ENOUGH TO LET IT", "Le Murmure", None, 88),
                    ("SETTLE", "Le Murmure", "Blackout", 230)], land=7.85, mt=(8.15, 0.45, 1.1), key=2),
    ],
}
PHRASES = FONT_SETS["bungee"]
# transitions k -> k+1: wash window [t0, t1]; peak relative rotation (rad), scale, shift (design px);
# noise re-injection height; radial stagger origin (design px); target ground polarity after the transition
TRANS = [
    dict(t0=2.65, t1=3.30, dth=0.14, ds=0.05, dsh=(0.0, 0.0), h=0.18, org=(110, 160), pol=1.0),
    dict(t0=4.95, t1=5.60, dth=-0.15, ds=-0.05, dsh=(0.0, 0.0), h=0.18, org=(620, 780), pol=0.0),
    dict(t0=7.10, t1=7.80, dth=0.13, ds=0.06, dsh=(0.0, 0.0), h=0.25, org=(360, 450), pol=1.0),
]
E0 = dict(tm=0.15, st=0.95, org=(360, 450))   # opening: full noise relaxes outward from the centre

# ring variant: screen path of the ring centre (the vortex of every wash sits on it), keyed (t, x, y)
RING_PATH = [(0.0, 360, 450), (2.97, 205, 300), (5.27, 545, 650), (7.45, 330, 360), (10.0, 360, 470)]

VARIANTS = {
    "lin": dict(test="E4a", stem="E4a_moire", theta=math.radians(32.0)),
    "ring": dict(test="E4b", stem="E4b_rings", theta=0.0),
}


# ---------------------------------------------------------------------------------------------- maths
def smoothstep(x):
    x = xp.clip(x, 0.0, 1.0) if xp.is_t(x) else np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def spring_decay(tau):
    """Damped spring released from 1 at tau=0 (closed form); 1 for tau<0."""
    tau = xp.maximum(tau, 0.0)
    wd = OMEGA * math.sqrt(1 - ZETA ** 2)
    return xp.exp(-ZETA * OMEGA * tau) * (xp.cos(wd * tau) + ZETA / math.sqrt(1 - ZETA ** 2) * xp.sin(wd * tau))


def bump(t, t0, t1):
    """0 outside [t0, t1], smooth 1 in the middle."""
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return np.sin(np.pi * u) ** 2


def to_int(x):
    return x.long() if xp.is_t(x) else x.astype(np.int64)


def hash01(ix, iy, seed):
    """Integer hash of integer cell indices -> [0, 1). int64 maths, no overflow (identical on both backends)."""
    M = 0xFFFFFFFF
    h = (ix * 374761393 + iy * 668265263 + (seed * 2246822519 & M)) & M
    h = ((h ^ (h >> 13)) * 1274126177) & M
    h = h ^ (h >> 16)
    return (h.float() if xp.is_t(h) else h.astype(np.float32)) * (1.0 / 4294967296.0)


def value_noise(x, y, L, seed):
    gx, gy = x / L, y / L
    fx0, fy0 = xp.floor(gx), xp.floor(gy)
    fx, fy = gx - fx0, gy - fy0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = to_int(fx0), to_int(fy0)
    a = hash01(ix, iy, seed)
    b = hash01(ix + 1, iy, seed)
    c = hash01(ix, iy + 1, seed)
    d = hash01(ix + 1, iy + 1, seed)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy - 0.5


def upsample(x, shape):
    """bilinear (half-pixel centres) resize of a 2D field to shape (H, W)."""
    if xp.is_t(x):
        F = xp.torch().nn.functional
        return F.interpolate(x[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def gradient(phi):
    return xp.torch().gradient(phi) if xp.is_t(phi) else np.gradient(phi)


# ---------------------------------------------------------------------------------------------- glyphs
def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(font_path(name), size)
    ax = font_axes(name)
    if ax:
        tags = {"weight": "wght", "width": "wdth", "optical size": "opsz", "year": "YEAR"}
        vals = []
        for a in f.get_variation_axes():
            nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
            vals.append(ax.get(tags.get(nm.lower(), nm), a["default"]))
        f.set_variation_by_axes(vals)
    return f


def line_w(f, s):
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def fit_font(name, s, cap_max, k):
    """font sized so the line's ink spans TEXT_W (cap height <= cap_max), at k px per design px."""
    probe = load_font(name, 200)
    pb = probe.getbbox("H")
    z = min(200 * TEXT_W / line_w(probe, s), 200 * cap_max / (pb[3] - pb[1])) * k
    f = load_font(name, max(4, int(z)))
    hb = f.getbbox("H")
    return f, hb[3] - hb[1]


def phrase_sdfs(lines, W, H):
    """Justified stack at SS output scale: SDF (design px, inside > 0) of the A faces and of the B (morph) faces.
    CPU, once per size (setup)."""
    k = SS * W / DW
    fa = [fit_font(fnt, s, cap, k) for s, fnt, _, cap in lines]
    fb = [fit_font(fb_ or fnt, s, cap, k) for s, fnt, fb_, cap in lines]
    slot = [max(a[1], b[1]) for a, b in zip(fa, fb)]
    gap = LINE_GAP * min(slot)
    cx, cy = SS * W / 2, SS * H / 2
    centres = []                                   # ink centre x of every glyph of the A layout
    for (s, *_), (f, cap) in zip(lines, fa):
        x, cs = cx - line_w(f, s) / 2 - f.getbbox(s[0])[0], []
        for ch in s:
            bb = f.getbbox(ch)
            cs.append(x + (bb[0] + bb[2]) / 2)
            x += f.getlength(ch) + TRACK * f.size
        centres.append(cs)
    out, slots = [], []
    y0 = cy - (sum(slot) + gap * (len(lines) - 1)) / 2
    for sl in slot:
        slots.append(((y0 - 0.12 * sl) / k, (y0 + 1.12 * sl) / k))
        y0 += sl + gap
    for fonts in (fa, fb):
        im = Image.new("L", (SS * W, SS * H), 0)
        d = ImageDraw.Draw(im)
        y0 = cy - (sum(slot) + gap * (len(lines) - 1)) / 2
        for (s, *_), (f, cap), sl, cs in zip(lines, fonts, slot, centres):
            hb = f.getbbox("H")
            for ch, c in zip(s, cs):
                bb = f.getbbox(ch)
                d.text((c - (bb[0] + bb[2]) / 2, y0 + (sl - cap) - hb[1]), ch, font=f, fill=255)
            y0 += sl + gap
        m = (np.asarray(im) >= 128).astype(np.uint8)
        din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
        out.append(((din - dout + 0.5 * (2 * m - 1.0)) / k).astype(np.float32))
    return out, slots


def sample2(img2, X, Y, k):
    """bilinear sample of a (2, Hs, Ws) field at design coords X, Y (k SS px per design px); edge clamp."""
    if xp.is_t(img2):
        T = xp.torch()
        Hs, Ws = img2.shape[1], img2.shape[2]
        g = T.stack([X * k / Ws * 2 - 1, Y * k / Hs * 2 - 1], -1)[None]
        return T.nn.functional.grid_sample(img2[None], g, mode="bilinear", padding_mode="border",
                                           align_corners=False)[0]
    mx, my = (X * k - 0.5).astype(np.float32), (Y * k - 0.5).astype(np.float32)
    return np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in img2])


# ---------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, variant="lin", be=None):
        self.be = be or xp.Backend("torch", "cpu")
        self.colour = False
        self.v = variant
        self.cfg = VARIANTS[variant]
        self.theta = self.cfg["theta"]
        self._cache = {}

    # --- timeline (scalars) ---
    def mask_weights(self, t):
        """a_k(t): mask k phases out / k+1 phases in across transition k (smooth, centred in the wash)."""
        w = np.zeros(len(PHRASES))
        w[0] = 1.0
        for k, tr in enumerate(TRANS):
            u = float(smoothstep((t - (tr["t0"] + 0.30 * (tr["t1"] - tr["t0"]))) / (0.4 * (tr["t1"] - tr["t0"]))))
            w[k + 1] = u
            w[k] *= 1 - u
        return w

    def morph(self, k, t, px):
        """face morph weight; (t0, per-glyph duration, left-to-right sweep time): sweeps across x."""
        mt = PHRASES[k].get("mt")
        if mt is None or t < mt[0]:
            return 0.0
        if t > mt[0] + mt[1] + mt[2]:
            return 1.0
        x0, x1 = (DW - TEXT_W) / 2, (DW + TEXT_W) / 2
        return smoothstep((t - mt[0] - mt[2] * xp.clip((px - x0) / (x1 - x0), 0.0, 1.0)) / mt[1])

    def mask(self, k, t, px, py, W, H, key=False):
        """M_k on screen: SDF lerp between the phrase's faces, slow zoom about the centre, soft edge.
        key=True: the key line only (for the colour lock)."""
        sc = 1.0 + MASK_GROW * (t - PHRASES[k]["land"])
        X, Y = DW / 2 + (px - DW / 2) / sc, DH / 2 + (py - DH / 2) / sc
        tex, slots = self.masks(W, H)[k]
        sd = sample2(tex, X, Y, SS * W / DW)
        m = self.morph(k, t, px)
        d = sd[0] if isinstance(m, float) and m == 0 else (sd[1] if isinstance(m, float) else sd[0] * (1 - m) + sd[1] * m)
        M = smoothstep(0.5 + d / EDGE)
        if key:
            y0, y1 = slots[PHRASES[k]["key"]]
            M = M * ((Y >= y0) * (Y <= y1))
        return M

    def lock_blocks(self, t, px, py, W, H):
        """C(p, t) in [0, 1]: cobalt where the key word has locked. Screen blocks of CBLOCK flip on one by one
        after the landing (radial stagger from the event origin + jitter) and drop out at the next wash."""
        C = None
        for k, ph in enumerate(PHRASES):
            if "key" not in ph:
                continue
            t_end = TRANS[k]["t0"] - 0.05 if k < len(TRANS) else DUR + 1
            if not (ph["land"] - 0.2 < t < t_end + 0.4):
                continue
            org = E0["org"] if k == 0 else TRANS[k - 1]["org"]
            bx, by = xp.floor(px / CBLOCK), xp.floor(py / CBLOCK)
            ibx, iby = to_int(bx), to_int(by)
            rho = xp.sqrt(((bx + 0.5) * CBLOCK - org[0]) ** 2 + ((by + 0.5) * CBLOCK - org[1]) ** 2) / math.hypot(DW, DH)
            t_on = ph["land"] - 0.1 + 0.5 * rho + 0.3 * hash01(ibx, iby, 97)
            t_off = t_end + 0.35 * hash01(ibx, iby, 98)
            on = (t >= t_on) * (t < t_off)
            c = on * self.mask(k, t, px, py, W, H, key=True)
            C = c if C is None else C + c
        return C

    def pol(self, t):
        p = 0.0
        for tr in TRANS:
            u = float(smoothstep((t - tr["t0"]) / (tr["t1"] - tr["t0"])))
            p = p + (tr["pol"] - p) * u
        return p

    def rel(self, t):
        """relative rotation, scale-1, shift of layer 2 (scalars)."""
        dth = sum(tr["dth"] * float(bump(t, tr["t0"] - 0.1, tr["t1"])) for tr in TRANS)
        ds = sum(tr["ds"] * float(bump(t, tr["t0"] - 0.1, tr["t1"])) for tr in TRANS)
        return dth, ds

    def camera(self, t):
        """screen -> world: centre, zoom, roll, translation. Velocity eases to zero at the end (settle)."""
        e = t - t * t / (2 * DUR)                       # integral of (1 - t/DUR): decelerates to rest
        zoom = 1.0 + 0.22 * (e / (DUR / 2))             # slow push-in over the field
        roll = 0.10 * math.sin(0.21 * e) + 0.04 * e
        tx, ty = 38.0 * e, -16.0 * e + 10 * math.sin(0.5 * e)
        return zoom, roll, tx, ty

    def noise_amp(self, t, px, py):
        """A(p, t): opening noise + re-injections, each a staggered rise then damped-spring relax."""
        def rho(org):
            return xp.sqrt((px - org[0]) ** 2 + (py - org[1]) ** 2) / math.hypot(DW, DH)
        A = spring_decay(t - E0["tm"] - E0["st"] * rho(E0["org"]))
        for tr in TRANS:
            tm = 0.5 * (tr["t0"] + tr["t1"]) - 0.15
            tau = t - tm - 0.55 * rho(tr["org"])
            rise = smoothstep((tau + 0.35) / 0.35)
            A = A + tr["h"] * xp.where(tau < 0, rise, spring_decay(tau))
        return A

    # --- geometry ---
    def masks(self, W, H):
        key = (W, H)
        if key not in self._cache:
            self._cache[key] = []
            for p in PHRASES:
                sd, slots = phrase_sdfs(p["lines"], W, H)
                self._cache[key].append((self.be.asarray(np.stack(sd), np.float32 if not self.be.torch else None),
                                         slots))
        return self._cache[key]

    def base(self, X, Y, layer, t):
        """B(q) in design px and its unit normal direction (nx, ny)."""
        if self.v == "lin":
            nx, ny = math.cos(self.theta), math.sin(self.theta)
            return X * nx + Y * ny, nx, ny
        # concentric rings about Cr (world): a screen point wandering along RING_PATH (cosine ease), so each
        # wash's fringe vortex opens in a new place; layer 2's centre offset by d(t) during washes
        cx, cy = self.ring_world(t)
        if layer == 2:
            off = sum(float(bump(t, tr["t0"] - 0.1, tr["t1"])) * s for tr, s in zip(TRANS, (1, -1, 1)))
            cx, cy = cx + 34.0 * off, cy + 22.0 * off
        dx, dy = X - cx, Y - cy
        r = xp.sqrt(dx * dx + dy * dy) + 1e-3
        return r, dx / r, dy / r

    def ring_screen(self, t):
        for (t0, x0, y0), (t1, x1, y1) in zip(RING_PATH, RING_PATH[1:]):
            if t <= t1:
                u = 0.5 - 0.5 * math.cos(math.pi * min(max((t - t0) / (t1 - t0), 0.0), 1.0))
                return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        return RING_PATH[-1][1:]

    def ring_world(self, t):
        sx, sy = self.ring_screen(t)
        zoom, roll, tx, ty = self.camera(t)
        u, v = (sx - DW / 2) / zoom, (sy - DH / 2) / zoom
        cr, sr = math.cos(roll), math.sin(roll)
        return cr * u - sr * v + DW / 2 + tx, sr * u + cr * v + DH / 2 + ty

    def layer_phase(self, layer, X, Y, A, t):
        """phi for one layer in world coords X, Y (design px), noise amplitude A."""
        f = 2 * np.pi / PERIOD
        B, nx, ny = self.base(X, Y, layer, t)
        phi = f * B
        if A is not None:
            seed = 11 if layer == 1 else 23
            off = 0.37 if layer == 1 else 0.71           # offset cell grids between layers
            cx = xp.floor(X / CELL + off)
            cy = xp.floor(Y / CELL + off * 0.6)
            icx, icy = to_int(cx), to_int(cy)
            ang = (hash01(icx, icy, seed) - 0.5) * np.pi * A          # random grating angle per cell
            ph = (hash01(icx, icy, seed + 1) - 0.5) * 2 * np.pi * A   # random phase per cell
            ccx, ccy = (cx + 0.5 - off) * CELL, (cy + 0.5 - off * 0.6) * CELL
            ca, sa = xp.cos(ang), xp.sin(ang)
            rnx, rny = ca * nx - sa * ny, sa * nx + ca * ny
            phi = phi + f * ((X - ccx) * (rnx - nx) + (Y - ccy) * (rny - ny)) + ph
            sub = 8                                       # smooth warp: evaluate coarse, upsample
            vn = upsample(value_noise(X[::sub, ::sub], Y[::sub, ::sub], WARP_L, seed + 2), tuple(X.shape))
            phi = phi + 5.0 * np.pi * A * abs(A) * vn      # warp fades faster than the cells
        return phi

    def frame(self, t, W, H):
        Ws, Hs = SS * W, SS * H
        k = DW / Ws                                          # design px per SS pixel
        be = self.be
        if be.torch:
            T = xp.torch()
            ys, xs = T.meshgrid(T.arange(Hs, device=be.dev, dtype=T.float32),
                                T.arange(Ws, device=be.dev, dtype=T.float32), indexing="ij")
        else:
            ys, xs = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
        px, py = (xs + 0.5) * k, (ys + 0.5) * k             # screen, design px
        zoom, roll, tx, ty = self.camera(t)
        cr, sr = math.cos(roll), math.sin(roll)
        u, v = (px - DW / 2) / zoom, (py - DH / 2) / zoom
        X = cr * u - sr * v + DW / 2 + tx
        Y = sr * u + cr * v + DH / 2 + ty
        A = self.noise_amp(t, px, py)
        A = A if float(abs(A).max()) > 4e-3 else None
        phi1 = self.layer_phase(1, X, Y, A, t)
        # layer 2: relative rotation / scale about the screen centre (in world)
        dth, ds = self.rel(t)
        wcx, wcy = (DW / 2 + tx, DH / 2 + ty) if self.v == "lin" else self.ring_world(t)   # pivot
        c2, s2 = math.cos(dth), math.sin(dth)
        sc = 1.0 + ds
        X2 = wcx + sc * (c2 * (X - wcx) - s2 * (Y - wcy))
        Y2 = wcy + sc * (s2 * (X - wcx) + c2 * (Y - wcy))
        phi2 = self.layer_phase(2, X2, Y2, A, t)
        w = self.mask_weights(t)
        lat = float(self.pol(t) + GROUND)
        for kk, wk in enumerate(w):
            if wk > 1e-4:
                lat = lat + float(wk * LETTER) * self.mask(kk, t, px, py, W, H)
        phi2 = phi2 + np.pi * lat
        sh = SHIMMER * (1 - float(smoothstep((t - 8.4) / 1.4)))   # the last phrase settles to stillness
        if sh > 1e-4:
            kx, ky = math.cos(-0.9), math.sin(-0.9)
            phi2 = phi2 + 2 * np.pi * sh * xp.sin(2 * np.pi * ((X * kx + Y * ky) - SHIM_V * t) / SHIM_L)
        g1 = square(phi1)
        g2 = square(phi2)
        img = g1 + g2 - 2 * g1 * g2
        img = img * 255.0
        C = self.lock_blocks(t, px, py, W, H) if self.colour else None
        if C is not None:                                       # flat cobalt over the locked key word
            C = xp.clip(C, 0.0, 1.0)
            img = [img * (1 - C) + c * C for c in COBALT]
            img = xp.torch().stack(img, -1) if be.torch else np.stack(img, -1)
        elif self.colour:
            img = xp.torch().stack([img] * 3, -1) if be.torch else np.stack([img] * 3, -1)
        img = img.reshape(H, SS, W, SS, *img.shape[2:]).mean((1, 3))   # SS x SS box filter
        img = xp.clip(img + 0.5, 0, 255)
        return img.to(xp.torch().uint8).cpu().numpy() if be.torch else img.astype(np.uint8)


def square(phi):
    """Anti-aliased square wave of phase phi (per-pixel slope from the numerical phase gradient)."""
    gy, gx = gradient(phi)
    g = xp.sqrt(gx * gx + gy * gy) + 1e-3
    return xp.clip(0.5 + xp.sin(phi) / g, 0.0, 1.0)     # ramp ~1 SS px wide; -> grey past Nyquist


# ---------------------------------------------------------------------------------------------- output
def render(sc, path, W, H):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nfr = int(round(FPS * DUR))
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24" if sc.colour else "gray", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    sc.frame(0.0, W, H)                                     # warm-up: masks, kernels (not timed)
    t0 = time.perf_counter()
    frames = [sc.frame(n / FPS, W, H) for n in range(nfr)]  # all frames first, so render and encode time apart
    render_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    encode_s = time.perf_counter() - t1
    print(f"{path}  {W}x{H}  {nfr} frames  render {render_s:.1f}s ({render_s * 1000 / nfr:.0f} ms/frame)  "
          f"encode {encode_s:.1f}s  [{sc.be}]")
    return dict(frames=nfr, render_s=render_s, encode_s=encode_s)


TIMING_HDR = ("gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,"
              "encode_s,audio_s,notes\n")


def nz_now():
    import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("Pacific/Auckland"))
    except Exception:
        return datetime.datetime.now()


def log_timing(nn, test, kind, device, W, H, started, tm, audio_s=None, work_start=None, notes=""):
    """one measured row per generation output in local/TIMINGS.csv (timing rule)."""
    import datetime
    path = os.path.join(LOCAL, "TIMINGS.csv")
    fin = nz_now()
    make = ""
    if work_start:
        ws = datetime.datetime.strptime(work_start, "%Y-%m-%d %H:%M").replace(tzinfo=fin.tzinfo)
        make = f"{(fin - ws).total_seconds() / 60:.1f}"
    row = [f"g{nn:02d}", test, kind, device, str(tm.get("frames", "")) if tm else "", f"{W}x{H}" if W else "",
           started.strftime("%Y-%m-%d %H:%M"), fin.strftime("%Y-%m-%d %H:%M"), make,
           f"{tm['render_s']:.1f}" if tm else "", f"{tm['render_s'] * 1000 / tm['frames']:.0f}" if tm else "",
           f"{tm['encode_s']:.1f}" if tm else "", f"{audio_s:.1f}" if audio_s is not None else "", notes]
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(TIMING_HDR)
        f.write(",".join(c.replace(",", ";") for c in row) + "\n")


def landings():
    return [p["land"] for p in PHRASES]


def qc(sc, test, nn, video):
    """Contact sheet of the video + one strip per transition (8 frames, 180 px wide)."""
    os.makedirs(REVIEW, exist_ok=True)
    tag = f"{test}_g{nn:02d}" if nn is not None else f"{test}_adhoc"
    out = []
    if video:
        sheet = os.path.join(REVIEW, f"{tag}_sheet.jpg")
        if not os.path.exists(sheet):
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", video, "-vf",
                            "fps=5,scale=144:-2,tile=10x5", "-frames:v", "1", sheet], check=True)
        out.append(sheet)
    spans = [(0.0, 1.6)] + [(tr["t0"] - 0.1, tr["t1"] + 0.25) for tr in TRANS]
    for i, (a, b) in enumerate(spans):
        fn = os.path.join(REVIEW, f"{tag}_strip{i}.jpg")
        if os.path.exists(fn):
            continue
        ims = [sc.frame(t, 180, 225) for t in np.linspace(a, b, 8)]
        Image.fromarray(np.concatenate(ims, axis=1)).save(fn, quality=90)
        out.append(fn)
    return out


def device_label(be):
    if not be.torch:
        return "local-numpy" if sys.platform == "win32" else "box-numpy"
    if be.dev.type == "cuda":
        name = xp.torch().cuda.get_device_name(be.dev)
        return "cuda-" + name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
    return "local-cpu-torch" if sys.platform == "win32" else "box-cpu"


def use_kit():
    """--kit: route the shared pieces through tools/mathtype/kit (eases, spring, noise, grating AA, sampling,
    glyph layout + SDF, box filter, encode loop, TIMINGS, QC). The E4 maths (Scene) is unchanged; the old
    module functions stay the default path. Same numbers by construction (parity: kit/parity.py)."""
    global smoothstep, spring_decay, bump, hash01, value_noise, upsample, gradient, sample2, square, phrase_sdfs
    global render, log_timing, qc
    from kit import camera as kc, draw as kd, io as kio, qc as kqc, text as kt, timeline as ktl
    smoothstep = lambda x: kc.smoothstep(0.0, 1.0, x)            # noqa: E731
    spring_decay = lambda tau: kc.spring_decay(tau, ZETA, OMEGA)  # noqa: E731
    bump, hash01, value_noise, upsample, gradient = kc.bump, kd.hash01, kd.value_noise, kd.upsample, kd.gradient
    sample2, square = kd.sample, kd.square_wave

    def phrase_sdfs(lines, W, H):                                 # E4's SDF convention: inside > 0, |d| >= 0.5 px
        lay = kt.layout([(s, f, fb, cap) for s, f, fb, cap in lines], DW, DH, scale=SS * W / DW, text_w=TEXT_W,
                        track=TRACK, gap=LINE_GAP, axes={"Facade": {"CTST": 800}})   # E4 left CTST at its default
        out = [((-lay.sdf(w, units="px") + (2.0 * lay.mask(w) - 1.0)) / lay.k).astype(np.float32) for w in "AB"]
        return out, lay.slots

    def render(sc, path, W, H):
        sc.frame(0.0, W, H)
        return kio.render(lambda t, w, h: sc.frame(t, w, h), path, W, H, int(round(FPS * DUR)), FPS, sync=sc.be.sync)

    def log_timing(nn, test, kind, device, W, H, started, tm, audio_s=None, work_start=None, notes=""):
        fin = kio.nz_now()
        kio.timing_row(LOCAL, nn, test, kind, device, tm["frames"] if tm else "", f"{W}x{H}" if W else "", started,
                       fin, kio.make_minutes(work_start, fin), tm["render_s"] if tm else None,
                       tm["encode_s"] if tm else None, audio_s, notes + " [kit]")

    def qc(sc, test, nn, video):
        tag = f"{test}_g{nn:02d}" if nn is not None else f"{test}_adhoc"
        out = []
        if video:
            out.append(kio.contact_sheet(video, os.path.join(REVIEW, f"{tag}_sheet.jpg"), dur=DUR))
            tl = ktl.Timeline.fixed([" ".join(s for s, *_ in p["lines"]) for p in PHRASES], landings(),
                                    [(tr["t0"], tr["t1"]) for tr in TRANS], DUR)
            kqc.review(video, tl, None, out_prefix=os.path.join(REVIEW, f"{tag}_qc"))
            out.append(os.path.join(REVIEW, f"{tag}_qc.json"))
        spans = [(0.0, 1.6)] + [(tr["t0"] - 0.1, tr["t1"] + 0.25) for tr in TRANS]
        for i, (a, b) in enumerate(spans):
            fn = os.path.join(REVIEW, f"{tag}_strip{i}.jpg")
            if not os.path.exists(fn):
                out.append(kio.strip(lambda t, w, h: sc.frame(t, w, h), np.linspace(a, b, 8), 180, 225, fn))
        return out


def main():
    global PHRASES
    import gens
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="lin")
    ap.add_argument("--fonts", choices=list(FONT_SETS), default="bungee")
    ap.add_argument("--colour", action="store_true", help="cobalt lock blocks on the key words")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="torch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--draft", action="store_true", help="360x450 preview generation")
    ap.add_argument("--final", action="store_true", help="720x900 final + score + mux")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--qc", action="store_true", help="sheet + strips for the latest draft")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--work-start", default=None, help="'YYYY-MM-DD HH:MM' NZ: start of work on this generation")
    ap.add_argument("--kit", action="store_true", help="shared pieces from tools/mathtype/kit (same output)")
    a = ap.parse_args()
    if a.kit:
        use_kit()
    PHRASES = FONT_SETS[a.fonts]
    cfg = VARIANTS[a.variant]
    test, stem = cfg["test"], cfg["stem"]
    if a.backend == "torch" and a.threads:
        xp.torch().set_num_threads(a.threads)
    be = xp.Backend(a.backend, a.device)
    sc = Scene(a.variant, be)
    sc.colour = a.colour
    dev = device_label(be)
    h = gens.code_hash([os.path.abspath(__file__)]) + "-" + a.variant + "-" + a.fonts + ("-cobalt" if a.colour else "")
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    if a.draft:
        started = nz_now()
        nn = gens.gen_for(LOCAL, test, stem, "draft", h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, "draft")
        tm = render(sc, os.path.join(LOCAL, name), 360, 450)
        log_timing(nn, test, "draft", dev, 360, 450, started, tm, work_start=a.work_start, notes=a.fonts + (" cobalt" if a.colour else ""))
        stills = []
        for t in (TRANS[0]["t0"] + 0.3, PHRASES[1]["land"] + 0.6):
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            Image.fromarray(sc.frame(t, 432, 540)).save(os.path.join(REVIEW, fn), quality=92)
            stills.append(fn)
        gens.add_files(LOCAL, test, nn, [name], stills)
        if a.qc:
            print("\n".join(qc(sc, test, nn, os.path.join(LOCAL, name))))
    if a.final:
        started = nz_now()
        nn = gens.gen_for(LOCAL, test, stem, "final", h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, "final")
        tm = render(sc, os.path.join(LOCAL, name), 720, 900)
        log_timing(nn, test, "final", dev, 720, 900, started, tm, work_start=a.work_start, notes=a.fonts + (" cobalt" if a.colour else ""))
        files = [name]
        changes = ",".join(f"{x:g}" for x in landings())
        score = os.path.join(HERE, "score.py")
        if os.path.exists(score):
            wav = f"score_{test}_g{nn:02d}.wav"
            cmd = [sys.executable, score, "--preset", "E4", "--duration", f"{DUR:g}", "--changes", changes,
                   "--out", os.path.join(LOCAL, wav), "--timings", os.path.join(LOCAL, "TIMINGS.csv"),
                   "--note", f"E4 g{nn:02d}"]
            if a.backend == "torch":
                cmd += ["--backend", "torch", "--device", a.device]
        else:
            wav = f"chords_{test}_g{nn:02d}.wav"
            cmd = [sys.executable, os.path.join(HERE, "chords.py"), "--duration", f"{DUR:g}", "--changes", changes,
                   "--out", os.path.join(LOCAL, wav)]
        if not os.path.exists(os.path.join(LOCAL, wav)):
            ta = nz_now()
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            if "--timings" not in cmd:                           # score.py logs its own music row
                log_timing(nn, test, "music", "box-cpu" if sys.platform != "win32" else "local-cpu", 0, 0, ta, None,
                           audio_s=time.perf_counter() - t0, notes=os.path.basename(cmd[1]) + " " + wav)
        mux = name.replace("_final.mp4", "_final_audio.mp4")
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", os.path.join(LOCAL, name),
                        "-i", os.path.join(LOCAL, wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", os.path.join(LOCAL, mux)], check=True)
        files += [wav, mux]
        gens.add_files(LOCAL, test, nn, files)
    if a.still:
        for t in a.still:
            Image.fromarray(sc.frame(t, a.size, int(a.size * 1.25))).save(
                os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))
    if a.qc and not a.draft:
        print("\n".join(qc(sc, test, None, None)))


if __name__ == "__main__":
    main()
