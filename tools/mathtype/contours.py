#!/usr/bin/env python
"""E1 concentric outline cascades in 3D (11_mathtype v007).

"Every line is the edge of something, and every edge is the beginning of a shape we have not learned
to see." Each phrase is a signed distance field f (design px, negative inside the glyphs). Its iso-
contours f = d_k echo the letters outward like topographic lines. Level k is a flat plate at its own
height h_k, so the stack is a terraced 3D height field seen through a moving perspective camera:

- terrace: h = -beta * d (a stepped pyramid, letters on top). Phrases sit at different places on the
  plane; the camera flies from one to the next. Phrases blend by a smooth-min union of eroded fields,
  smin(A + E*wa(s), B + E*(1 - wb(s)), r(s)), so A's echoes collapse into its letters while B grows from
  its stroke skeletons and ripples outward; level k runs k*tau late (an outward cascade).
- tunnel: h = +beta * d (a funnel, letters at the bottom of the well), seen from above while the camera
  dollies down it. Rings are emitted at the letter edge and flow outward at speed v(t) toward the camera;
  a ring keeps the shape the field had when it was emitted, so the tunnel holds the recent past and
  each phrase change travels out through it. Phrases blend by a plain SDF lerp plus a noise warp.

Rendering has no polygons. A plate at height h is a plane, so its projection is a homography; each
level warps its source fields to the screen (cv2.warpPerspective, or torch grid_sample on --device
cuda), blends them there, and gets an exactly anti-aliased edge and line from the screen-space distance
(f - d) / |grad f|. Plates are painted from the lowest to the highest, so the terraces occlude correctly.
Every frame is a pure function of t.

    python tools/mathtype/contours.py --variant terrace --preview         # 360x450 -> local/E1_terrace_gNN_draft.mp4 + QC
    python tools/mathtype/contours.py --variant tunnel --preview
    python tools/mathtype/contours.py --variant tunnel --colour --final --backend torch --device cuda
    python tools/mathtype/contours.py --variant terrace --still 1.5 2.8   # ad-hoc PNGs to review/
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

VDIR = os.path.join(REPO, "11_mathtype", "v007")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

DW, DH = 720, 900                      # design space (= final size); every length below is design px
FPS, DUR = 25, 10.0
NFR = int(round(FPS * DUR))

FONT = "Big Shoulders Display"
PHRASES = [("EVERY LINE IS", "THE EDGE OF", "SOMETHING"),
           ("AND EVERY", "EDGE IS THE", "BEGINNING"),
           ("OF A SHAPE", "WE HAVE NOT"),
           ("LEARNED", "TO SEE")]
LAND = [0.95, 3.10, 5.60, 7.90]        # letter outline fully formed
TT = [0.85, 1.00, 1.00, 1.00]          # morph length before each landing
# --fontmix: per line (text, face A, face B or None). The phrase lands in its A faces; mid-read the key
# line's SDF lerps to its B face, so the letters and all their echoes morph between typefaces.
FONTMIX = [
    [("EVERY LINE IS", "Le Murmure", None), ("THE EDGE OF", "Le Murmure", None),
     ("SOMETHING", "Le Murmure", "Pilowlava")],
    [("AND EVERY", "Big Shoulders Display", None), ("EDGE IS THE", "Big Shoulders Display", None),
     ("BEGINNING", "Big Shoulders Display", "Blackout")],
    [("OF A", "Le Murmure", None), ("SHAPE", "Climate Crisis", "Shrikhand"), ("WE HAVE NOT", "Le Murmure", None)],
    [("LEARNED", "Le Murmure", None), ("TO  SEE", "Blackout", "Monoton")],
]
MORPH0, MORPH_LEN = 0.45, 0.80         # face morph: starts this long after landing, lasts this long
TEXT_W = 520.0
MAX_CAP = 150.0
LINE_GAP = 0.30                        # x cap height
TRACK = 0.03                           # em
G = 2                                  # design px per SDF grid cell
MARGIN = 840                           # SDF canvas margin around a phrase (> farthest level drawn)

# palette (0..1 RGB)
def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


INK, BONE = _hex("#111111"), _hex("#ECE6DA")
OCHRE, TEAL, VERMILION = _hex("#D9A441"), _hex("#1F5E63"), _hex("#D2462A")
BLACK, WHITE = np.zeros(3, np.float32), np.ones(3, np.float32)

VARIANTS = {
    "terrace": dict(
        centers=[(0.0, 0.0), (230.0, -720.0), (-190.0, -1440.0), (40.0, -2160.0)],
        blend="union", delta=14.0, kout=26, beta=0.75, tau=0.022, E=470.0, rmax=60.0, bg=300.0,
        an=26.0, ao=7.0, R=1150.0, theta_r=64.0, theta_f=30.0, fl=1.05, fog=(2100.0, 3300.0)),
    "tunnel": dict(
        centers=[(0.0, 0.0)] * 4,
        blend="union", E=520.0, rmax=60.0, delta=18.0, dfar=430.0, beta=2.4, v0=40.0, v1=420.0, an=34.0,
        H0=1180.0, F=1150.0),
}

T_FLIP = LAND[3] + 0.05                # ground inversion starts at the letters ...
C_FLIP = 950.0                         # ... and runs outward at this speed (design px / s)


def smoothstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smootherstep(x):
    u = min(max(x, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def state_at(tl):
    """(a, b, s): blending phrase a -> b at s in [0, 1]; a = -1 is the opening noise terrain."""
    for j in range(len(LAND)):
        t0 = LAND[j] - TT[j]
        if tl < t0:
            return j - 1, j - 1, 0.0
        if tl < LAND[j]:
            return j - 1, j, smootherstep((tl - t0) / TT[j])
    n = len(LAND) - 1
    return n, n, 0.0


# ------------------------------------------------------------------------------------------- glyphs
def load_font(name, size):
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


def block_mask(lines):
    """Justified stack at 1 design px: each line's ink spans TEXT_W (cap height capped)."""
    probe = load_font(FONT, 200)
    hb = probe.getbbox("H")
    rows = []
    for s in lines:
        size = int(min(200 * TEXT_W / line_w(probe, s), 200 * MAX_CAP / (hb[3] - hb[1])))
        f = load_font(FONT, size)
        b = f.getbbox("H")
        cap = b[3] - b[1]
        im = Image.new("L", (int(TEXT_W) + 40, cap + 8), 0)
        d = ImageDraw.Draw(im)
        x = 20 + (TEXT_W - line_w(f, s)) / 2 - f.getbbox(s[0])[0]
        for ch in s:
            d.text((x, 4 - b[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        rows.append((np.array(im), cap))
    gap = LINE_GAP * min(c for _, c in rows)
    hh = int(math.ceil(sum(r.shape[0] for r, _ in rows) + gap * (len(rows) - 1))) + 2
    out = np.zeros((hh, int(TEXT_W) + 40), np.uint8)
    y = 0.0
    for r, cap in rows:
        y0 = int(round(y))
        out[y0:y0 + r.shape[0]] = np.maximum(out[y0:y0 + r.shape[0]], r)
        y += r.shape[0] + gap
    return out, min(c for _, c in rows)


def render_line(s, face, width=TEXT_W, max_h=MAX_CAP * 1.18):
    """Line s in `face`, ink fitted to `width` (ink height capped), cropped to its ink (+4 px)."""
    probe = load_font(face, 200)
    pb = probe.getbbox(s)
    size = int(min(200 * width / line_w(probe, s), 200 * max_h / max(pb[3] - pb[1], 1)))
    f = load_font(face, size)
    bb = f.getbbox(s)
    im = Image.new("L", (int(width) + 200, bb[3] - bb[1] + 40), 0)
    d = ImageDraw.Draw(im)
    x = 100 + (width - line_w(f, s)) / 2 - f.getbbox(s[0])[0]
    for ch in s:
        d.text((x, 20 - bb[1]), ch, font=f, fill=255)
        x += f.getlength(ch) + TRACK * f.size
    a = np.array(im)
    ys = np.where(a.max(1) > 0)[0]
    return a[max(ys[0] - 4, 0):ys[-1] + 5]


def block_mask_mix(spec, which):
    """Mixed-face stack; each row is as tall as the taller of its A/B renders, ink centred in the row."""
    rows = []
    for text, fa, fb in spec:
        ra = render_line(text, fa)
        rb = render_line(text, fb) if fb else ra
        rows.append((ra if which == "A" or not fb else rb, max(ra.shape[0], rb.shape[0])))
    gap = 0.16 * min(h for _, h in rows)
    W = rows[0][0].shape[1]
    hh = int(math.ceil(sum(h for _, h in rows) + gap * (len(rows) - 1))) + 2
    out = np.zeros((hh, W), np.uint8)
    y = 0.0
    for r, h in rows:
        y0 = int(round(y + (h - r.shape[0]) / 2))
        out[y0:y0 + r.shape[0]] = np.maximum(out[y0:y0 + r.shape[0]], r)
        y += h + gap
    return out


def sdf_from_mask(m):
    inside = (m > 127).astype(np.uint8)
    din = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(inside > 0, -(din - 0.5), dout - 0.5).astype(np.float32)


# accent shapes for the colour generation: (phrase, kind, centre offset from block centre in block
# units (fx of half-width, fy of half-height + px), size px, colour)
SHAPES = {
    0: ("bar", (-0.42, 1.0, 46.0), (150.0, 9.0), OCHRE),
    1: ("tri", (0.80, -1.0, -58.0), 34.0, TEAL),
    2: ("box", (0.78, 1.0, 54.0), 30.0, OCHRE),
    3: ("disc", (0.0, 1.0, 74.0), 30.0, VERMILION),
}


def shape_sdf(kind, X, Y, cx, cy, size):
    px, py = X - cx, Y - cy
    if kind == "disc":
        return np.sqrt(px * px + py * py) - size
    if kind in ("box", "bar"):
        bx, by = (size, size) if kind == "box" else size
        qx, qy = np.abs(px) - bx, np.abs(py) - by
        return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
    # equilateral triangle, pointing up (Inigo Quilez)
    k = math.sqrt(3.0)
    px = np.abs(px) - size
    py = -py + size / k
    sw = px + k * py > 0
    px2 = np.where(sw, (px - k * py) / 2, px)
    py2 = np.where(sw, (-k * px - py) / 2, py)
    px2 = px2 - np.clip(px2, -2 * size, 0)
    return -np.sqrt(px2 * px2 + py2 * py2) * np.sign(py2)


def periodic_noise(n, sigma_cells, seed):
    """Smooth periodic noise tile (n x n), unit std, via an FFT Gaussian filter."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    kk = k[:, None] ** 2 + k[None, :] ** 2
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * np.exp(-2 * (math.pi * sigma_cells) ** 2 * kk)))
    return ((f - f.mean()) / f.std()).astype(np.float32)


# ------------------------------------------------------------------------------------------- backend
class Warp:
    """warpPerspective of a source field onto the screen (numpy/cv2 or torch grid_sample)."""

    def __init__(self, backend, device, W, H):
        self.torch = backend == "torch"
        self.W, self.H = W, H
        if self.torch:
            import torch
            self.T = torch
            self.dev = torch.device(device)
            v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                                  indexing="ij")              # float32: FP64 is 1/64 rate on consumer GPUs
            self.uv1 = torch.stack([u, v, torch.ones_like(u)], 0).reshape(3, -1).to(self.dev)

    def put(self, a):
        return self.T.as_tensor(a, device=self.dev) if self.torch else a

    def __call__(self, src, Hm, wrap=False):
        if not self.torch:
            return cv2.warpPerspective(src, Hm, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE)
        T = self.T
        Hi = T.as_tensor(np.linalg.inv(Hm), dtype=T.float32, device=self.dev)
        p = Hi @ self.uv1
        x, y = p[0] / p[2], p[1] / p[2]
        h, w = src.shape
        if wrap:
            x, y = T.remainder(x, w), T.remainder(y, h)
            src = T.nn.functional.pad(src[None, None], (0, 1, 0, 1), mode="circular")[0, 0]
            h, w = h + 1, w + 1
        g = T.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], -1).reshape(1, self.H, self.W, 2).float()
        return T.nn.functional.grid_sample(src[None, None], g, mode="bilinear", padding_mode="border",
                                           align_corners=True)[0, 0]

    # elementwise helpers that work on both
    def zeros_rgb(self, c):
        if self.torch:
            return self.T.as_tensor(c, device=self.dev).view(1, 1, 3).expand(self.H, self.W, 3).clone()
        return np.broadcast_to(np.asarray(c, np.float32), (self.H, self.W, 3)).copy()

    def gradmag(self, f):
        """|grad f| per screen px (central differences, one-sided at the edges)."""
        if self.torch:
            pad = self.T.nn.functional.pad(f[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        else:
            pad = np.pad(f, 1, mode="edge")
        gx = pad[1:-1, 2:] - pad[1:-1, :-2]
        gy = pad[2:, 1:-1] - pad[:-2, 1:-1]
        g = (gx * gx + gy * gy) ** 0.5 * 0.5 + 1e-4
        self.last_gyn = gy * 0.5 / g             # unit-gradient y component (screen), for the parallax test
        return g

    def floor(self, x, v):
        return x.clamp(min=v) if self.torch else np.maximum(x, v)

    def screen_uv(self):
        """Pixel offsets from the frame centre (two H x W fields), cached."""
        if not hasattr(self, "_uv"):
            u = np.broadcast_to(np.arange(self.W, dtype=np.float32) - self.W / 2, (self.H, self.W))
            v = np.broadcast_to((np.arange(self.H, dtype=np.float32) - self.H / 2)[:, None], (self.H, self.W))
            self._uv = (self.put(np.ascontiguousarray(u)), self.put(np.ascontiguousarray(v)))
        return self._uv

    def clip01(self, x):
        return x.clamp(0, 1) if self.torch else np.clip(x, 0, 1)

    def minimum(self, a, b):
        return self.T.minimum(a, b) if self.torch else np.minimum(a, b)

    def maximum0(self, a):
        return a.clamp(min=0) if self.torch else np.maximum(a, 0)

    def paint(self, canvas, alpha, colour):
        key = tuple(np.round(np.asarray(colour, np.float32), 5))
        if not hasattr(self, "_cc"):
            self._cc = {}
        c = self._cc.get(key)
        if c is None:
            c = self._cc[key] = self.put(np.asarray(colour, np.float32))
        if self.torch:
            canvas += (c.view(1, 1, 3) - canvas) * alpha[..., None]
        else:
            canvas += (c[None, None, :] - canvas) * alpha[..., None]

    def to_u8(self, canvas):
        if self.torch:
            return (canvas.clamp(0, 1) * 255 + 0.5).to(self.T.uint8).cpu().numpy()
        return (np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------- scene
class Src:
    """A field on a grid: world x = ox + (col + 0.5) * g; border replicate (or wrap)."""

    def __init__(self, a, ox, oy, g, wrap=False):
        self.a, self.ox, self.oy, self.g, self.wrap = a, ox, oy, g, wrap
        self.Tg = np.array([[g, 0, ox + g / 2], [0, g, oy + g / 2], [0, 0, 1]], float)


class Scene:
    def __init__(self, variant="terrace", colour=False, backend="numpy", device="cpu", W=360, H=450,
                 fontmix=False):
        self.var, self.colour, self.fontmix = variant, colour, fontmix
        self.cfg = VARIANTS[variant]
        self.W, self.H = W, H
        self.wp = Warp(backend, device, W, H)
        self.ph, self.sh, self.phB = [], [], []
        for i, lines in enumerate(PHRASES):
            if fontmix:
                m = block_mask_mix(FONTMIX[i], "A")
                mB = block_mask_mix(FONTMIX[i], "B") if any(fb for _, _, fb in FONTMIX[i]) else None
            else:
                m, cap = block_mask(lines)
                mB = None
            hh, ww = m.shape
            cw = (ww + 2 * MARGIN) // G * G
            chh = (hh + 2 * MARGIN) // G * G
            can = np.zeros((chh, cw), np.uint8)
            y0, x0 = (chh - hh) // 2, (cw - ww) // 2
            can[y0:y0 + hh, x0:x0 + ww] = m
            sdf = sdf_from_mask(can)
            sdfB = None
            if mB is not None:
                canB = np.zeros_like(can)
                canB[y0:y0 + hh, x0:x0 + ww] = mB
                sdfB = sdf_from_mask(canB)
            cx, cy = self.cfg["centers"][i]
            ox, oy = cx - cw / 2, cy - chh / 2
            shp = None
            if colour and i in SHAPES:
                kind, (fx, fy, off), size, col = SHAPES[i]
                ys, xs = np.mgrid[0:chh, 0:cw].astype(np.float32)
                scx = cw / 2 + fx * TEXT_W / 2
                scy = chh / 2 + fy * hh / 2 + off
                shp = shape_sdf(kind, xs + 0.5, ys + 0.5, scx, scy, size).astype(np.float32)
                sdf = np.minimum(sdf, shp)
                if sdfB is not None:
                    sdfB = np.minimum(sdfB, shp)
                shp = Src(self.wp.put(cv2.resize(shp, (cw // G, chh // G), interpolation=cv2.INTER_AREA)),
                          ox, oy, G)
            grid = cv2.resize(sdf, (cw // G, chh // G), interpolation=cv2.INTER_AREA)
            self.ph.append(Src(self.wp.put(grid), ox, oy, G))
            self.phB.append(None if sdfB is None else Src(self.wp.put(
                cv2.resize(sdfB, (cw // G, chh // G), interpolation=cv2.INTER_AREA)), ox, oy, G))
            self.sh.append(shp)
        # opening terrain: periodic noise with |grad| ~ 1 (design px), offset so the levels cut it
        gn = 4
        P = periodic_noise(512, 100.0 / gn, 11).astype(np.float64)
        gy, gx = np.gradient(P)
        P = P / (np.median(np.hypot(gx, gy)) / gn)
        pre = (200.0 + 0.9 * P).astype(np.float32)
        self.pre = Src(self.wp.put(pre), -1024.0, -1024.0, gn, wrap=True)
        self.bg = None
        if "bg" in self.cfg:              # standing terrain between phrases (terrace)
            Q = periodic_noise(512, 120.0 / gn, 12).astype(np.float64)
            gy, gx = np.gradient(Q)
            Q = Q / (np.median(np.hypot(gx, gy)) / gn)
            self.bg = Src(self.wp.put((self.cfg["bg"] + 0.9 * Q).astype(np.float32)), -1300.0, -700.0, gn, wrap=True)
        self.nz = [Src(self.wp.put(periodic_noise(256, 60.0 / gn, s)), -512.0, -512.0, gn, wrap=True)
                   for s in (21, 22)]
        if variant == "tunnel":
            c = self.cfg
            self.tt = np.linspace(-40.0, DUR + 2.0, 60001)
            self.phi_tab = self.phi(self.tt)

    # ----- time functions
    def src(self, i):
        return self.pre if i < 0 else self.ph[i]

    def phi(self, t):
        """Ring flow distance (design px): integral of v(t) = v0 + v1 * sum of Gaussian bumps."""
        c = self.cfg
        t = np.asarray(t, float)
        out = c["v0"] * t
        w = 0.40
        for j in range(len(LAND)):
            m = LAND[j] + 0.22               # echo burst just after each landing
            out = out + c["v1"] * w * math.sqrt(math.pi) / 2 * (_erf((t - m) / w) - _erf(-m / w))
        return out

    def noise_coef(self, t):
        return math.cos(1.3 * t + 0.4), math.sin(1.3 * t + 0.4)

    # ----- camera
    def camera(self, t):
        """(P, r, dn, f, F) in world coords: x right, y down the page, z = height toward the viewer."""
        if self.var == "terrace":
            return self.cam_terrace(t)
        return self.cam_tunnel(t)

    def cam_terrace(self, t):
        c = self.cfg
        C = [np.array(p) for p in c["centers"]]
        drift = np.array([0.0, -22.0])
        keys = [(-1.0, C[0] + np.array([140.0, 1050.0]), np.array([-60.0, -820.0]))]
        for j in range(len(LAND)):
            p0 = C[j] + np.array([0.0, 18.0])
            keys.append((LAND[j] + 0.1, p0, drift))
            t1 = (LAND[j + 1] - c["fl"]) if j + 1 < len(LAND) else DUR + 1.0
            keys.append((t1, p0 + drift * (t1 - LAND[j] - 0.1), drift))
        T = hermite(keys, t)
        # flight bump: 0 at reads, 1 mid-flight
        b, sgn = 0.0, 1.0
        flights = [(-1.0, LAND[0] + 0.1)] + [(LAND[j] - c["fl"], LAND[j] + 0.1) for j in range(1, len(LAND))]
        for j, (a0, a1) in enumerate(flights):
            if a0 <= t <= a1:
                u = (t - a0) / (a1 - a0)
                b = math.sin(math.pi * u) ** 2
                sgn = (-1.0) ** j
        theta = math.radians(c["theta_r"] - (c["theta_r"] - c["theta_f"]) * b)
        psi = math.radians(14.0 * sgn * b + 2.5 * math.sin(0.45 * t + 0.7))
        R = c["R"] * (1.0 + 0.30 * b) * (1.0 - 0.012 * t)
        bdir = np.array([math.sin(psi), math.cos(psi), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        P = np.array([T[0], T[1], 0.0]) + R * (math.cos(theta) * bdir + math.sin(theta) * z)
        f = -(math.cos(theta) * bdir + math.sin(theta) * z)
        dn = math.sin(theta) * bdir - math.cos(theta) * z
        r = np.cross(f, dn)
        roll = math.radians(-9.0 * sgn * b)
        r, dn = math.cos(roll) * r + math.sin(roll) * dn, -math.sin(roll) * r + math.cos(roll) * dn
        return P, r, dn, f, c["R"]

    def cam_tunnel(self, t):
        c = self.cfg
        S = sum(0.5 * (1 + _erf((t - (L - 0.35)) / 0.40)) for L in LAND)
        Hc = c["H0"] - 14.0 * t - 34.0 * S
        P = np.array([95.0 * math.sin(0.55 * t + 0.3) + 40.0 * math.sin(1.25 * t + 2.0),
                      70.0 * math.sin(0.42 * t + 1.9) + 30.0 * math.sin(1.05 * t), Hc])
        T = np.array([0.0, 10.0, 0.0])
        f = T - P
        f /= np.linalg.norm(f)
        dn = np.array([0.0, 1.0, 0.0]) - f[1] * f
        dn /= np.linalg.norm(dn)
        r = np.cross(f, dn)
        roll = 0.07 * math.sin(0.33 * t + 0.5) + 0.05 * (S - 1.5) / 1.5
        r, dn = math.cos(roll) * r + math.sin(roll) * dn, -math.sin(roll) * r + math.cos(roll) * dn
        return P, r, dn, f, c["F"]

    def plane_H(self, cam, h, src):
        """Homography: src grid (col, row) -> screen px for the plane at height h."""
        P, r, dn, f, F = cam
        Fs = F * self.W / DW
        K = np.array([[Fs, 0, self.W / 2], [0, Fs, self.H / 2], [0, 0, 1]])
        O = np.array([0.0, 0.0, h]) - P
        M = np.array([[r[0], r[1], r @ O], [dn[0], dn[1], dn @ O], [f[0], f[1], f @ O]])
        return K @ M @ src.Tg

    def depth(self, cam, x, y, h):
        P, r, dn, f, F = cam
        return float((np.array([x, y, h]) - P) @ f)

    # ----- fields on the screen
    def field(self, cam, h, a, b, s, tl, amp, which="ph"):
        wp = self.wp
        srcs = self.ph if which == "ph" else self.sh

        def get(i):
            if which == "ph" and i >= 0 and self.phB[i] is not None:
                m = float(smoothstep(LAND[i] + MORPH0, LAND[i] + MORPH0 + MORPH_LEN, tl))
                fa = wp(self.ph[i].a, self.plane_H(cam, h, self.ph[i]), False) if m < 1 else None
                fb = wp(self.phB[i].a, self.plane_H(cam, h, self.phB[i]), False) if m > 0 else None
                return fa if fb is None else (fb if fa is None else fa + (fb - fa) * m)
            if which == "ph":
                sr = self.src(i)
            else:
                sr = srcs[i] if i >= 0 else None
            if sr is None:
                return None
            return wp(sr.a, self.plane_H(cam, h, sr), sr.wrap)

        if a == b or s <= 0.0:
            f = get(a)
        elif s >= 1.0:
            f = get(b)
        else:
            fa, fb = get(a), get(b)
            if fa is None and fb is None:
                return None
            big = 3000.0
            fa = fa if fa is not None else fb * 0 + big
            fb = fb if fb is not None else fa * 0 + big
            if self.cfg["blend"] == "lerp" and which == "ph":
                f = fa + (fb - fa) * s
            else:
                E = self.cfg.get("E", 900.0)
                wa = float(smoothstep(0.30, 0.80, s))   # A holds while the camera leaves, then erodes
                wb = float(smoothstep(0.08, 0.85, s))   # B grows from its stroke skeletons, in view
                A = fa + E * wa
                B = fb + E * (1.0 - wb)
                rr = self.cfg.get("rmax", 60.0) * math.sin(math.pi * s)
                f = wp.minimum(A, B)
                if rr > 1e-3:
                    hh = wp.maximum0(rr - abs(A - B)) / rr
                    f = f - hh * hh * (rr / 4.0)
        if f is None:
            return None
        if self.bg is not None and which == "ph":
            bw = wp(self.bg.a, self.plane_H(cam, h, self.bg), True)
            rr = 45.0
            hh = wp.maximum0(rr - abs(f - bw)) / rr
            f = wp.minimum(f, bw) - hh * hh * (rr / 4.0)
        if amp > 1e-3 and which == "ph":
            c1, c2 = self.noise_coef(tl)
            n1 = wp(self.nz[0].a, self.plane_H(cam, h, self.nz[0]), True)
            n2 = wp(self.nz[1].a, self.plane_H(cam, h, self.nz[1]), True)
            f = f + (n1 * c1 + n2 * c2) * amp
        return f

    def sdist(self, f, d, cam, h):
        """Screen-px signed distance to the iso-line f = d. |grad f| is floored at 0.6x its nominal
        value on this plate, so flat spots (smin joins, lerp ridges) cannot smear a line wide."""
        P, r, dn, fv, F = cam
        z = max(float((np.array([0.0, 0.0, h]) - P) @ fv), 1.0) if self.var == "tunnel" else             max(self.depth(cam, *hermite_target_cache(self, None, cam)[:2], h), 1.0)
        nominal = z / (F * self.W / DW)
        self._gs = self.wp.floor(self.wp.gradmag(f), 0.6 * nominal)
        return (f - d) / self._gs

    def parallax(self, cam, T, h, dh):
        """Screen-y shift (px) of the look-at point when its height changes from h to h + dh."""
        P, r, dn, f, F = cam
        Fs = F * self.W / DW

        def py(z):
            q = np.array([T[0], T[1], z]) - P
            return Fs * (q @ dn) / max(q @ f, 1.0)
        return py(h + dh) - py(h)

    def fill_amount(self, a, b, s):
        """Letter fill: full on a held phrase, off through the morph (none on the opening terrain)."""
        if a == b:
            return 0.0 if a < 0 else 1.0
        fa = 0.0 if a < 0 else 1.0
        return float(fa * (1 - smoothstep(0.30, 0.42, s)) + smoothstep(0.45, 0.62, s))

    # ----- tones
    def inverted(self, t, d):
        return t >= T_FLIP + max(d, 0.0) / C_FLIP

    def tones(self, inv):
        """(dark band A, dark band B, line base, letter fill) before the flip; inverted after."""
        if self.colour:
            dark, dark2, light = INK, INK * 0.82 + BONE * 0.18 * 0.55 + INK * 0.0, BONE
            dark2 = INK + (BONE - INK) * 0.09
        else:
            dark, dark2, light = BLACK, np.full(3, 0.085, np.float32), WHITE
        if not inv:
            return dark, dark2, light
        return light, light + (dark - light) * 0.10, dark

    # ----- frame
    def frame(self, t):
        if self.var == "terrace":
            return self.frame_terrace(t)
        return self.frame_tunnel(t)

    def line_alpha(self, dist, w, spacing=None):
        """AA line of width w px on the iso-line. With `spacing` (field units between levels), lines
        fade where foreshortening packs them closer than ~3.5 widths, instead of fusing into a smear."""
        a = self.wp.clip01((w * 0.5 + 0.5) - abs(dist))
        if spacing is not None:
            # screen gap to the next level out: in-plane spacing plus the parallax of its height step
            sp = spacing / self._gs + self._par * self.wp.last_gyn
            a = a * self.wp.clip01((sp - w) / (2.5 * w))
        return a

    def frame_terrace(self, t):
        c, wp = self.cfg, self.wp
        cam = self.camera(t)
        P, r, dn, f, F = cam
        sc = self.W / DW
        K = c["kout"]
        inv_g = self.inverted(t, (K + 1) * c["delta"])
        bandA, bandB, light = self.tones(inv_g)
        canvas = wp.zeros_rgb(bandA if (K + 1) % 2 == 0 else bandB)
        a0, b0, s0 = state_at(t)
        T = np.array(hermite_target_cache(self, t))
        for k in range(K, -1, -1):
            d = k * c["delta"]
            h = -c["beta"] * d
            tl = t - k * c["tau"]
            a, b, s = state_at(tl)
            amp = c["an"] * math.sin(math.pi * s) ** 2 * (0.6 + 0.4 * k / K) + c["ao"] * (k / K) ** 2
            fw = self.field(cam, h, a, b, s, tl, amp)
            dist = self.sdist(fw, d, cam, h)
            self._par = self.parallax(cam, T, h, -c["beta"] * c["delta"])
            inv = self.inverted(t, d)
            bA, bB, li = self.tones(inv)
            tone = bA if k % 2 == 0 else bB
            if k == 0:
                tone = tone + (li - tone) * self.fill_amount(a, b, s)
            elif self.colour:
                tone = self.band_colour_terrace(k, a, b, s, tl, tone)
            wp.paint(canvas, wp.clip01(0.5 - dist), tone)
            # accent shape plate (colour gen), on the letter plane
            if k == 0 and self.colour:
                fs = self.field(cam, h, a, b, s, tl, 0.0, which="sh")
                if fs is not None:
                    col = self.shape_colour(a, b, s, inv)
                    if col is not None:
                        wp.paint(canvas, wp.clip01(0.5 - fs / wp.gradmag(fs)), col)
            # contour line on the plate edge
            z = max(self.depth(cam, T[0], T[1], h), 1.0)
            persp = F / z
            if k == 0:
                w, lv = 2.3, 1.0
            else:
                w, lv = 1.55 - 0.7 * k / K, 0.16 + 0.70 * math.exp(-(k - 1) / 7.0)
            w = max(w * sc * persp, 0.85)
            lv = min(lv + self.flash(t, d), 1.0)
            lt = self.line_tone(lv, inv)
            wp.paint(canvas, self.line_alpha(dist, w, c["delta"] if k > 0 else None), lt)
        # fog toward the ground tone (far terrain)
        canvas = self.fog(canvas, cam, bandA)
        return wp.to_u8(canvas)

    def flash(self, t, d):
        """Brightness wave running out through the echoes after each landing."""
        out = 0.0
        for L in LAND:
            out += 0.35 * math.exp(-(((t - L - 0.04) - d / 650.0) / 0.08) ** 2)
        return out

    def line_tone(self, lv, inv):
        if self.colour:
            base = INK + (BONE - INK) * lv
            return BONE + (INK - BONE) * lv if inv else base
        return np.full(3, (1.0 - lv) if inv else lv, np.float32)

    def fog(self, canvas, cam, tone):
        c, wp = self.cfg, self.wp
        if "fog" not in c:
            return canvas
        P, r, dn, f, F = cam
        Fs = F * self.W / DW
        U, V = wp.screen_uv()
        dz = (U * (r[2] / Fs) + V * (dn[2] / Fs) + f[2])
        dz = dz.clamp(max=-1e-6) if wp.torch else np.minimum(dz, -1e-6)
        lam = (-60.0 - P[2]) / dz                      # ray length to the plane (huge near the horizon)
        u = wp.clip01((lam - c["fog"][0]) / (c["fog"][1] - c["fog"][0]))
        wp.paint(canvas, u * u * (3 - 2 * u), tone)
        return canvas

    def band_colour_terrace(self, k, a, b, s, tl, tone):
        """Colour gen: a few echo bands fill flat after a landing (cascade outward), never in a morph."""
        if a != b or a < 0:
            return tone
        spec = {0: {3: OCHRE}, 1: {3: TEAL}, 2: {3: OCHRE, 8: TEAL}, 3: {2: OCHRE, 6: TEAL}}[a]
        if k not in spec:
            return tone
        on = tl - (LAND[a] + 0.12 + 0.05 * k)
        if on < 0:
            return tone
        return spec[k]

    def shape_colour(self, a, b, s, inv):
        if a == b:
            return SHAPES[a][3] if a in SHAPES else None
        i = b if s > 0.5 else a
        return SHAPES[i][3] if i in SHAPES and i >= 0 else None

    # ----- tunnel
    def emit_time(self, phi_val):
        return float(np.interp(phi_val, self.phi_tab, self.tt))

    def frame_tunnel(self, t):
        c, wp = self.cfg, self.wp
        cam = self.camera(t)
        P, r, dn, f, F = cam
        sc = self.W / DW
        D = c["delta"]
        ph = float(self.phi(t))
        n_new = math.floor(ph / D)                 # newest emitted ring
        a, b, s = state_at(t)
        inv0 = self.inverted(t, 0.0)
        bA, bB, li = self.tones(inv0)
        # floor (shows inside the letters): letter fill
        tone_next = bA if (n_new + 1) % 2 == 0 else bB
        floor = tone_next + (li - tone_next) * self.fill_amount(a, b, s)
        canvas = wp.zeros_rgb(floor)
        amp0 = c["an"] * math.sin(math.pi * s) ** 2
        f0 = self.field(cam, 0.0, a, b, s, t, amp0)
        dist0 = self.sdist(f0, 0.0, cam, 0.0)
        wp.paint(canvas, wp.clip01(0.5 + dist0), self.ring_tone(n_new + 1, t, inv0))
        rings = []
        n = n_new
        while True:
            d = ph - n * D
            if d > c["dfar"]:
                break
            rings.append((n, d))
            n -= 1
        z0 = max(self.depth(cam, 0.0, 0.0, 0.0), 1.0)
        for n, d in rings:                          # lowest (newest, smallest d) first
            h = c["beta"] * d
            z = self.depth(cam, 0.0, 0.0, h)
            if z < 120.0:
                continue
            te = self.emit_time(n * D)
            ra, rb, rs = state_at(te)
            amp = c["an"] * math.sin(math.pi * rs) ** 2
            fw = self.field(cam, h, ra, rb, rs, te, amp)
            dist = self.sdist(fw, d, cam, h)
            self._par = 0.0      # funnel: outer rings magnify radially, so parallax only widens the gaps
            inv = self.inverted(t, d)
            wp.paint(canvas, wp.clip01(0.5 + dist), self.ring_tone(n, te, inv, ra, rb))
            persp = F / z
            near = float(smoothstep(160.0, 420.0, z))
            lv = (0.14 + 0.72 * math.exp(-d / 230.0)) * float(smoothstep(0.0, 0.5 * D, d)) * near
            lv = min(lv + self.flash(t, d) * near, 1.0)
            w = max((1.5 - 0.4 * min(d / 400.0, 1.0)) * sc * min(persp, 3.5), 0.85)
            wp.paint(canvas, self.line_alpha(dist, w, D), self.line_tone(lv, inv))
        # letters on top: older rings were cut from older shapes and may cross the current letters
        fill = self.fill_amount(a, b, s)
        if fill > 0:
            wp.paint(canvas, wp.clip01(0.5 - dist0), floor)
        if self.colour:                           # accent shape: flat colour on the letter plane
            fs = self.field(cam, 0.0, a, b, s, t, 0.0, which="sh")
            col = self.shape_colour(a, b, s, inv0)
            if fs is not None and col is not None:
                wp.paint(canvas, wp.clip01(0.5 - fs / wp.gradmag(fs)), col)
        # letter outline (it is the lowest plate's edge, but keep it crisp over the floor)
        lv = min(1.0 + self.flash(t, 0.0), 1.0)
        w = max(2.3 * sc * F / z0, 0.85)
        wp.paint(canvas, self.line_alpha(dist0, w), self.line_tone(lv, inv0))
        return wp.to_u8(canvas)

    def ring_tone(self, n, te, inv, ra=None, rb=None):
        bA, bB, li = self.tones(inv)
        tone = bA if n % 2 == 0 else bB
        if self.colour and ra is not None and ra == rb and ra >= 0 and te > LAND[ra] + 0.1:
            spec = {0: (OCHRE,), 1: (TEAL,), 2: (OCHRE, TEAL), 3: (OCHRE, TEAL)}[ra]
            if n % 9 == 4:
                tone = spec[(n // 9) % len(spec)]
        return tone


def _erf(x):
    if isinstance(x, np.ndarray):
        from scipy.special import erf
        return erf(x)
    return math.erf(x)


def hermite(keys, t):
    """Cubic Hermite through (time, pos, vel) keys; clamps to the first/last key."""
    keys = sorted(keys, key=lambda k: k[0])
    if t <= keys[0][0]:
        return keys[0][1] + keys[0][2] * (t - keys[0][0])
    for (t0, p0, v0), (t1, p1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            dt = t1 - t0
            u = (t - t0) / dt
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1
    return keys[-1][1] + keys[-1][2] * (t - keys[-1][0])


def hermite_target_cache(scene, t, cam=None):
    """Camera target on the plane (terrace): the point the camera looks at."""
    P, r, dn, f, F = cam if cam is not None else scene.camera(t)
    lam = -P[2] / f[2]
    return P + lam * f


# ------------------------------------------------------------------------------------------- render
_SC = None


def _init(kw):
    global _SC
    cv2.setNumThreads(1)
    _SC = Scene(**kw)


def _work(i):
    return _SC.frame(i / FPS).tobytes()


def render(path, kw, procs=3, nfr=NFR):
    """Returns (wall_s, encode_s): encode_s is the time spent blocked on ffmpeg (writes + final wait)."""
    t0 = time.perf_counter()
    W, H = kw["W"], kw["H"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    enc = 0.0

    def put(buf):
        nonlocal enc
        e0 = time.perf_counter()
        ff.stdin.write(buf)
        enc += time.perf_counter() - e0

    if kw.get("backend") == "torch" or procs <= 1:
        sc = Scene(**kw)
        for i in range(nfr):
            put(sc.frame(i / FPS).tobytes())
    else:
        import multiprocessing as mp
        with mp.Pool(procs, initializer=_init, initargs=(kw,)) as pool:
            for buf in pool.imap(_work, range(nfr), chunksize=2):
                put(buf)
    e0 = time.perf_counter()
    ff.stdin.close()
    ff.wait()
    enc += time.perf_counter() - e0
    dt = time.perf_counter() - t0
    print(f"{path}  {W}x{H}  {nfr} frames  {dt:.1f}s (encode-blocked {enc:.1f}s)", flush=True)
    return dt, enc


TIMINGS_HDR = "gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,encode_s,audio_s,notes" + chr(10)


def now_nz():
    import datetime
    from zoneinfo import ZoneInfo
    return datetime.datetime.now(ZoneInfo("Pacific/Auckland"))


def log_timing(row):
    p = os.path.join(LOCAL, "TIMINGS.csv")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(TIMINGS_HDR)
    with open(p, "a", encoding="utf-8") as f:
        f.write(",".join(str(row.get(k, "")) for k in TIMINGS_HDR.strip().split(",")) + chr(10))


def qc(video, stem_nn, W):
    """Contact sheet (fps 4, 8x5) + one strip per transition (every 0.1 s over the morph)."""
    os.makedirs(REVIEW, exist_ok=True)
    out = []
    sheet = os.path.join(REVIEW, f"{stem_nn}_sheet.jpg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", "fps=4,scale=180:-2,tile=8x5",
                    "-frames:v", "1", sheet], check=True)
    out.append(sheet)
    for j, L in enumerate(LAND):
        t0 = max(L - TT[j] - 0.3, 0.0)
        strip = os.path.join(REVIEW, f"{stem_nn}_strip_T{j}.jpg")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", video, "-vf",
                        "fps=10,scale=150:-2,tile=12x1", "-frames:v", "1", strip], check=True)
        out.append(strip)
    return out


def use_kit():
    """--kit: route the shared pieces through tools/mathtype/kit (fonts, line width, SDF, accent shapes,
    eases, Hermite keys, homography warp, paint / to_u8, encode loop, TIMINGS, QC). The E1 maths (Scene) is
    unchanged; the old module functions stay the default path."""
    global load_font, line_w, sdf_from_mask, shape_sdf, smoothstep, smootherstep, hermite
    global render, log_timing, qc, now_nz
    from kit import camera as kc, draw as kd, io as kio, palette as kpal, qc as kqc, text as kt, timeline as ktl
    load_font = kt.load_font
    line_w = lambda f, s: kt.line_width(f, s, TRACK)                              # noqa: E731
    sdf_from_mask = lambda m: kt.sdf(m > 127, precise=True)                         # noqa: E731
    smoothstep, smootherstep, hermite, now_nz = kc.smoothstep, kc.smootherstep, kc.hermite, kio.nz_now
    shapes = {"disc": lambda X, Y, cx, cy, s: kd.disc(X, Y, cx, cy, s),
              "box": lambda X, Y, cx, cy, s: kd.box(X, Y, cx, cy, s, s),
              "bar": lambda X, Y, cx, cy, s: kd.box(X, Y, cx, cy, s[0], s[1]),
              "tri": lambda X, Y, cx, cy, s: kd.tri(X, Y, cx, cy, s)}
    shape_sdf = lambda kind, X, Y, cx, cy, size: shapes[kind](X, Y, cx, cy, size)  # noqa: E731
    Warp.__call__ = lambda self, src, Hm, wrap=False: kc.warp(src, Hm, self.W, self.H, wrap=wrap)
    Warp.paint = lambda self, canvas, alpha, colour: kd.paint(canvas, alpha, colour)
    Warp.to_u8 = lambda self, canvas: kd.to_u8(canvas)

    def render(path, kw, procs=3, nfr=NFR):
        """Returns (wall_s, encode_s) like the old render; kit encoder (h264_nvenc on CUDA)."""
        t0 = time.perf_counter()
        W, H = kw["W"], kw["H"]
        enc = kio.Encoder(path, W, H, FPS, nvenc=kw.get("device", "cpu").startswith("cuda"))
        if kw.get("backend") == "torch" or procs <= 1:
            sc = Scene(**kw)
            for i in range(nfr):
                enc.write(sc.frame(i / FPS))
        else:
            import multiprocessing as mp
            with mp.Pool(procs, initializer=_init_kit, initargs=(kw,)) as pool:
                for buf in pool.imap(_work, range(nfr), chunksize=2):
                    enc.write(np.frombuffer(buf, np.uint8))
        e = enc.close()
        dt = time.perf_counter() - t0
        print(f"{path}  {W}x{H}  {nfr} frames  {dt:.1f}s (encode-blocked {e:.1f}s) [kit]", flush=True)
        return dt, e

    def log_timing(row):
        kio.timing_row(LOCAL, row.get("gen", ""), row.get("test", ""), row.get("kind", ""), row.get("device", ""),
                       row.get("frames", ""), row.get("resolution", ""), row.get("started_nz"), row.get("finished_nz"),
                       float(row["make_min"]) if row.get("make_min") else None, float(row["render_s"]),
                       float(row["encode_s"]), float(row["audio_s"]) if row.get("audio_s") else None,
                       row.get("notes", "") + " [kit]")

    def qc(video, stem_nn, W):
        out = [kio.contact_sheet(video, os.path.join(REVIEW, f"{stem_nn}_sheet.jpg"), fps=4, width=180, cols=8, rows=5)]
        trans = [(LAND[j] - TT[j], LAND[j]) for j in range(1, len(LAND))]
        tl = ktl.Timeline.fixed([" ".join(p) for p in PHRASES], LAND, trans, DUR, form0=LAND[0] - TT[0])
        kqc.review(video, tl, kpal.get("e1"), out_prefix=os.path.join(REVIEW, f"{stem_nn}_qc"))
        return out + [os.path.join(REVIEW, f"{stem_nn}_qc.json")]


def _init_kit(kw):
    use_kit()
    _init(kw)


def main():
    import gens
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="terrace", choices=list(VARIANTS))
    ap.add_argument("--colour", action="store_true", help="E1 palette accents (bands + solid shapes)")
    ap.add_argument("--fontmix", action="store_true", help="mixed faces per line + SDF face morph mid-read")
    ap.add_argument("--preview", action="store_true", help="360x450 draft")
    ap.add_argument("--final", action="store_true", help="720x900 final")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--audio", action="store_true", help="with --final: score.py E1 + mux")
    ap.add_argument("--gen", type=int, default=None, help="force generation NN (e.g. a box final of a local draft)")
    ap.add_argument("--started", default=None, help="NZ HH:MM when work on this generation began (make_min)")
    ap.add_argument("--device-label", default=None, help="TIMINGS device, e.g. cuda-RTX3060 / box-cpu")
    ap.add_argument("--kit", action="store_true", help="shared pieces from tools/mathtype/kit (same output)")
    a = ap.parse_args()
    if a.kit:
        use_kit()
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    test = f"E1_{a.variant}" + ("_mix" if a.fontmix else "") + ("_colour" if a.colour else "")
    stem = test
    kw = dict(variant=a.variant, colour=a.colour, backend=a.backend, device=a.device, fontmix=a.fontmix)
    h = gens.code_hash([os.path.abspath(__file__)])
    if a.still:
        sc = Scene(W=360, H=450, **kw)
        for t in a.still:
            Image.fromarray(sc.frame(t)).save(os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))
    for kind, flag, (W, H) in (("draft", a.preview, (360, 450)), ("final", a.final, (720, 900))):
        if not flag:
            continue
        if a.gen:
            nn = a.gen
            if str(nn) not in gens._load(LOCAL, test):
                gens.backfill(LOCAL, test, nn, gens.now_nz(), a.parent or "", a.trigger, a.change, [], [])
        else:
            nn = gens.gen_for(LOCAL, test, stem, kind, h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, kind)
        path = os.path.join(LOCAL, name)
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; never overwrite")
        st = now_nz()
        dt, enc = render(path, dict(W=W, H=H, **kw), procs=a.procs)
        fin = now_nz()
        make = ""
        if a.started:
            hh, mm = map(int, a.started.split(":"))
            s0 = st.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if s0 > st:
                s0 = s0.replace(day=s0.day - 1)
            make = f"{(fin - s0).total_seconds() / 60:.1f}"
        dev = a.device_label or ("local-numpy" if a.backend == "numpy" else
                                 ("local-cpu-torch" if a.device == "cpu" else f"cuda-{a.device}"))
        trow = dict(gen=f"g{nn:02d}", test=test, kind="preview" if kind == "draft" else kind, device=dev,
                    frames=NFR, resolution=f"{W}x{H}", started_nz=st.strftime("%H:%M:%S"),
                    finished_nz=fin.strftime("%H:%M:%S"), make_min=make, render_s=f"{dt - enc:.1f}",
                    per_frame_ms=f"{(dt - enc) * 1000 / NFR:.0f}", encode_s=f"{enc:.1f}",
                    notes=f"procs={a.procs if a.backend == 'numpy' else 1}")
        files = [name]
        stills = []
        sc = Scene(W=360, H=450, **kw)
        for t in (LAND[1] - 0.35, LAND[1] + 0.8, LAND[3] + 1.2):
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            Image.fromarray(sc.frame(t)).save(os.path.join(REVIEW, fn), quality=90)
            stills.append(fn)
        sheets = qc(path, f"{test}_g{nn:02d}_{kind}", W)
        if kind == "final" and a.audio:
            wav = f"score_{test}_g{nn:02d}.wav"
            a0 = time.perf_counter()
            subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--preset", "E1", "--duration",
                            f"{DUR:g}", "--changes", ",".join(f"{x:g}" for x in LAND),
                            "--out", os.path.join(LOCAL, wav)], check=True)
            trow["audio_s"] = f"{time.perf_counter() - a0:.1f}"
            mux = f"{stem}_g{nn:02d}_final_audio.mp4"
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", path, "-i", os.path.join(LOCAL, wav),
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                            os.path.join(LOCAL, mux)], check=True)
            files += [wav, mux]
        gens.add_files(LOCAL, test, nn, files, stills)
        log_timing(trow)
        print("render", f"{dt:.1f}s", "sheets", sheets)


if __name__ == "__main__":
    main()
