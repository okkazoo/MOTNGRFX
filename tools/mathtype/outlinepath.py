#!/usr/bin/env python
"""E2 (11_mathtype v008): the glyph outline as a travel path.

"We are the slow echo of the first light, still travelling outward, still searching for somewhere to land."

Each glyph outline is a closed polyline P(s) (cv2.findContours on the rasterised face, smoothed and
resampled by arc length s). Phrases sit on planes placed along an outward spiral in 3D. Per contour two
writer heads start at the leftmost point and run round both sides with progress p(t) (cubic ease-out),
meeting on the far side. A point at arc s is laid at t_pass(s) and carries K strands offset by
    r(a) (N cos th + B sin th)  (helix)   or   r(a) sin(th) N  (flat neon),   th = m s - w t + 2 pi j / K,
with r(a) = R0 exp(-a / tau) (1 - R(t)), a = t - t_pass, so the tube collapses onto the crisp outline
as the phrase resolves (R -> 1). Extra runner lights lap the contours with s(t) = s0 + v (1 - e^{-lam t}) / lam
(speed -> 0). A lead light leaves the last glyph of a phrase on a cubic Bezier through space and writes
the first glyph of the next; the camera follows on its own Bezier and comes to rest square-on to read.
Everything is a pure function of t (``Scene.frame(t, W, H)``).

    python tools/mathtype/outlinepath.py --variant b --preview      # 360x450 -> v008/local/E2b_helix_gNN_preview.mp4
    python tools/mathtype/outlinepath.py --variant b --final        # 720x900 + score/chords + mux
    python tools/mathtype/outlinepath.py --variant b --still 1.0 2.4
    python tools/mathtype/outlinepath.py --qc <video>               # contact sheet + transition strips
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
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

VDIR = os.path.join(REPO, "11_mathtype", "v008")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

FPS = 25
DUR = 10.0
NFR = int(round(FPS * DUR))
DW, DH = 1080, 1350                     # design space (plane units)
CX, CY = DW / 2, DH / 2

FONT = "Big Shoulders Display"
WGHT = 800
PHRASES = [
    ("WE ARE", "THE SLOW", "ECHO"),
    ("OF THE", "FIRST", "LIGHT"),
    ("STILL", "TRAVELLING", "OUTWARD"),
    ("STILL", "SEARCHING", "FOR SOMEWHERE", "TO LAND"),
]
# per-line face (font mix): fine/tall lines against one puffy or neon key word per phrase.
# None = FONT. Each entry: (font, axes override, max cap height in design px, width fraction of TEXT_W)
MIX = False
LINE_FONTS = [
    [("Le Murmure", {}, 240, 0.62), ("Le Murmure", {}, 240, 0.85), ("Pilowlava", {}, 400, 1.0)],
    [("Le Murmure", {}, 240, 0.62), ("Le Murmure", {}, 240, 0.62), ("Monoton", {}, 380, 1.0)],
    [("Le Murmure", {}, 240, 0.50), ("Le Murmure", {}, 240, 1.0), ("Shrikhand", {}, 330, 1.0)],
    [("Le Murmure", {}, 220, 0.50), ("Le Murmure", {}, 220, 0.9), ("Le Murmure", {}, 220, 1.0),
     ("Pilowlava", {}, 380, 0.84)],
]
# (arrive = lead light reaches first glyph and writing starts, land = resolved, depart = lead light leaves)
TIMES = [(-0.40, 0.50, 2.00), (2.40, 3.30, 4.55), (4.95, 5.85, 7.10), (7.50, 8.40, 99.0)]
INVERT = 1                              # phrase index whose ground floods white
TEXT_W = 940.0
MAX_CAP = 300.0
TRACK = 0.04
DS = 2.5                                # arc-length sample spacing (design px)
FOCAL = 1250.0                          # design px; rest distance == FOCAL frames the plane 1:1
STAG = 0.45                             # left-to-right writing wave across a phrase (s)
WRITE_T = 0.45                          # one contour's writing duration (s)

VARIANTS = {
    "a": dict(stem="E2a_neon", test="E2a", helix=False, K=3, R0=32.0, m=0.030, w=8.0, tau_r=0.34,
              runners=2, colour=False),
    "b": dict(stem="E2b_helix", test="E2b", helix=True, K=3, R0=46.0, m=0.020, w=7.0, tau_r=0.38,
              runners=2, colour=False),
    "c": dict(stem="E2c_helixmix", test="E2c", helix=True, K=3, R0=46.0, m=0.020, w=7.0, tau_r=0.38,
              runners=2, colour=False, mix=True),
    "d": dict(stem="E2d_dawn", test="E2d", helix=True, K=3, R0=46.0, m=0.020, w=7.0, tau_r=0.38,
              runners=2, colour=True, mix=True),
}
CFG = VARIANTS["b"]
PREVIEW = False

# palette for the colour generation (E2: first light travelling)
NIGHT = np.array([0x0A, 0x0B, 0x10], np.float32) / 255
AMBER = np.array([0xF2, 0xA6, 0x5A], np.float32) / 255
CORAL = np.array([0xE8, 0x6F, 0x68], np.float32) / 255
GOLD = np.array([0xF6, 0xE7, 0xB8], np.float32) / 255
KEY_COLS = [CORAL, NIGHT, GOLD, AMBER]   # block colour inside each phrase's key word (ECHO, LIGHT, OUTWARD, LAND)
DISC_COLS = [AMBER, CORAL, AMBER, CORAL]  # solid disc where the lead light lands
DUST_GAIN = 0.9
SUN_R = 230.0                             # design px
SUN_DROP = 260.0
STOP_R = 36.0
DISC_R = 70.0                             # design px on the plane


# ---------------------------------------------------------------------------------------------- utils
def smoothstep(a, b, x):
    u = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def ease_out3(x):
    x = np.clip(x, 0.0, 1.0)
    return 1 - (1 - x) ** 3


def ease_out3_inv(y):
    y = np.clip(y, 0.0, 1.0)
    return 1 - (1 - y) ** (1 / 3)


def bezier(p0, p1, p2, p3, u):
    u = np.asarray(u, np.float64)[..., None]
    v = 1 - u
    return v ** 3 * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u ** 3 * p3


def rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_font(name: str, size: int, axes=None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(font_path(name), size)
    ax = dict(font_axes(name))
    if axes is None and name == FONT:
        ax["wght"] = WGHT
    ax.update(axes or {})
    try:
        vals = []
        for a in f.get_variation_axes():
            nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
            tag = {"weight": "wght", "contrast": "CTST"}.get(nm.lower(), nm)
            vals.append(ax.get(tag, a["default"]))
        f.set_variation_by_axes(vals)
    except Exception:
        pass
    return f


def line_w(f, s):
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


# ---------------------------------------------------------------------------------------------- glyphs
def phrase_layout(lines, ss=2, faces=None):
    """Render the phrase at ss x design size; return (mask at design/2, list of contours (design px))."""
    faces = faces or [(FONT, None, MAX_CAP, 1.0)] * len(lines)
    sizes = []
    for s, (fn, ax, mcap, wf) in zip(lines, faces):
        probe = load_font(fn, 200, ax)
        pb = probe.getbbox("H")
        size = 200 * TEXT_W * wf / line_w(probe, s)
        sizes.append(min(size, 200 * mcap / (pb[3] - pb[1])))
    fonts = [load_font(fn, int(z * ss), ax) for z, (fn, ax, _, _) in zip(sizes, faces)]
    caps = [(f.getbbox("H")[3] - f.getbbox("H")[1]) / ss for f in fonts]
    gap = 0.20 * min(caps)
    top = CY - (sum(caps) + gap * (len(lines) - 1)) / 2
    contours = []
    full = np.zeros((DH * ss, DW * ss), np.uint8)
    y0 = top
    for li, (s, f, cap) in enumerate(zip(lines, fonts, caps)):
        im = Image.new("L", (DW * ss, DH * ss), 0)
        d = ImageDraw.Draw(im)
        hb = f.getbbox("H")
        x = CX * ss - line_w(f, s) / 2 - f.getbbox(s[0])[0]
        for ch in s:
            d.text((x, y0 * ss - hb[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        arr = np.array(im)
        full = np.maximum(full, arr)
        if li == len(lines) - 1:
            keyarr = arr
        cs, _ = cv2.findContours((arr > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            p = c[:, 0, :].astype(np.float64) / ss
            if len(p) < 12:
                continue
            contours.append((li, resample(p)))
        y0 += cap + gap
    mask = cv2.resize(full, (DW // 2, DH // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    key = cv2.resize(keyarr, (DW // 2, DH // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return mask, contours, key


def resample(p):
    """Smooth a closed polyline (periodic Gaussian) and resample it uniformly by arc length."""
    n = len(p)
    sig = 1.2 * 2
    k = int(3 * sig)
    g = np.exp(-0.5 * (np.arange(-k, k + 1) / sig) ** 2)
    g /= g.sum()
    pp = np.concatenate([p[-k:], p, p[:k]])
    sm = np.stack([np.convolve(pp[:, i], g, mode="valid") for i in range(2)], 1)[:n]
    seg = np.linalg.norm(np.diff(np.vstack([sm, sm[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    m = max(int(round(L / DS)), 8)
    s = np.arange(m) * L / m
    q = np.vstack([sm, sm[:1]])
    x = np.interp(s, cum, q[:, 0])
    y = np.interp(s, cum, q[:, 1])
    pts = np.stack([x, y], 1)
    start = np.argmin(pts[:, 0] * 1.0 + pts[:, 1] * 0.08)   # leftmost (slightly top-biased) point
    return np.roll(pts, -start, axis=0)


class Contour:
    def __init__(self, li, pts, rng):
        self.li = li
        self.p = pts                                    # (M,2) design px
        M = len(pts)
        self.M = M
        self.L = M * (np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).mean())
        t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
        t = unit(t)
        self.T = t
        self.N = np.stack([t[:, 1], -t[:, 0]], 1)       # in-plane normal (design coords)
        s = np.arange(M) * self.L / M
        self.s = s
        self.ss = np.where(s <= self.L / 2, s, s - self.L)   # signed arc from the start point
        self.xmin = pts[:, 0].min()
        self.xc = pts[:, 0].mean()
        self.psi = rng.uniform(0, 2 * np.pi)
        self.runner = [dict(s0=rng.uniform(0, self.L), dir=rng.choice([-1, 1]),
                            v=self.L * rng.uniform(1.1, 2.2), lam=rng.uniform(2.2, 3.4),
                            delay=rng.uniform(0.0, 0.25), ph=rng.uniform(0, 2 * np.pi)) for _ in range(4)]


# ---------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        rng = np.random.default_rng(11)
        self.cfg = CFG
        self.ph = []
        # outward spiral of planes
        ang = np.radians([0, -25, 65, 155])
        rolls = np.radians([0, 7, -6, 0])
        yaws = np.radians([0, -10, 9, -5])
        pitches = np.radians([0, 5, -6, 3])
        Q = np.zeros(3)
        for k, lines in enumerate(PHRASES):
            if k > 0:
                Q = Q + np.array([1500 * np.cos(ang[k]), 1100 * np.sin(ang[k]), -2600.0])
            R = rot(yaws[k], pitches[k], rolls[k])
            mask, cons, keymask = phrase_layout(lines, faces=LINE_FONTS[k] if CFG.get("mix") else None)
            cs = [Contour(li, p, rng) for li, p in cons]
            nl = len(lines)
            xs = np.array([c.xmin for c in cs])
            for c in cs:
                lx = [d.xmin for d in cs if d.li == c.li]
                xn = (c.xmin - min(lx)) / max(max(lx) - min(lx), 1)
                c.t0 = TIMES[k][0] + STAG * (c.li + xn) / nl
            first = min(cs, key=lambda c: (c.li, c.xmin))
            first.t0 = TIMES[k][0]
            last_line = max(c.li for c in cs)
            last = max([c for c in cs if c.li == last_line], key=lambda c: c.p[:, 0].max())
            a_pt = first.p[0]
            d_idx = np.argmax(last.p[:, 0] - 0.3 * last.p[:, 1])
            d_pt = last.p[d_idx]
            self.ph.append(dict(Q=Q.copy(), R=R, n=R @ np.array([0, 0, 1.0]), up=R @ np.array([0, 1.0, 0]),
                                mask=mask, key=keymask, cs=cs,
                                key_c=(float((keymask * np.arange(DW // 2)[None]).sum() / keymask.sum() * 2),
                                       float((keymask * np.arange(DH // 2)[:, None]).sum() / keymask.sum() * 2)), arrive=TIMES[k][0], land=TIMES[k][1], depart=TIMES[k][2],
                                A=self.to_world(k, a_pt[None], Q, R)[0], D=self.to_world(k, d_pt[None], Q, R)[0]))
        # dust field for parallax
        nd = 700
        lo = np.array([-3500, -3000, -9500.0])
        hi = np.array([5500, 4500, 2500.0])
        self.dust = lo + rng.random((nd, 3)) * (hi - lo)
        self.dust_w = rng.uniform(0.15, 0.6, nd)

    @staticmethod
    def to_world(k, p2, Q, R, off=None):
        loc = np.stack([p2[:, 0] - CX, -(p2[:, 1] - CY), np.zeros(len(p2))], 1)
        if off is not None:
            loc = loc + off
        return Q + loc @ R.T

    # ------------------------------------------------------------------ camera + lead light
    def rest_eye(self, k, t):
        P = self.ph[k]
        h = smoothstep(P["land"], P["depart"], t)
        return P["Q"] + P["n"] * FOCAL * (1.0 - 0.12 * h)

    def camera(self, t):
        """eye, target, up. Flight k: from depart_{k-1} to land_k (k=0 starts at -0.9)."""
        ph = self.ph
        for k in range(len(ph)):
            fs = ph[k - 1]["depart"] - 0.05 if k else -0.9
            fe = ph[k]["arrive"] + 0.45
            if t < fs:
                continue
            if t <= fe:
                u = (t - fs) / (fe - fs)
                e = smootherstep(u ** 0.55)
                P3 = self.rest_eye(k, fe)
                if k:
                    P0 = self.rest_eye(k - 1, fs)
                    n0, Q0, up0 = ph[k - 1]["n"], ph[k - 1]["Q"], ph[k - 1]["up"]
                else:
                    P0 = ph[0]["Q"] + np.array([-900.0, -700.0, FOCAL + 2600.0])
                    n0, Q0, up0 = unit(P0 - ph[0]["Q"]), ph[0]["Q"], np.array([0, 1.0, 0])
                dist = np.linalg.norm(P3 - P0)
                P1 = P0 - n0 * 0.30 * dist
                P2 = P3 + (ph[k]["n"] * 0.32 + ph[k]["R"] @ np.array([0.22, -0.12, 0.0])) * dist
                eye = bezier(P0, P1, P2, P3, e)
                et = smootherstep(np.clip(u / 0.55, 0, 1) ** 0.8)
                tgt = Q0 + (ph[k]["Q"] - Q0) * et
                if k:   # the lead light pulls the gaze while it flies
                    L, _ = self.lead(t)
                    if L is not None:
                        wl = 0.6 * np.sin(np.pi * np.clip(u / 0.75, 0, 1))
                        tgt = tgt * (1 - wl) + L * wl
                roll = (0.30 if k % 2 else -0.30) * np.sin(np.pi * e)
                up = unit(up0 * (1 - e) + ph[k]["up"] * e)
                fwd = unit(tgt - eye)
                rt = unit(np.cross(fwd, up))
                up2 = np.cross(rt, fwd)
                up = up2 * np.cos(roll) + rt * np.sin(roll)
                return eye, tgt, up
            nxt = ph[k + 1]["depart"] if False else (ph[k]["depart"])
            if t <= nxt or k == len(ph) - 1:
                return self.rest_eye(k, t), ph[k]["Q"], ph[k]["up"]
        k = len(ph) - 1
        return self.rest_eye(k, t), ph[k]["Q"], ph[k]["up"]

    def lead_path(self, k):
        """Bezier control points for the lead light flying into phrase k."""
        ph = self.ph
        A = ph[k]["A"]
        if k == 0:
            D = ph[0]["Q"] + np.array([-1800.0, 1400.0, -3500.0])
            fs = ph[0]["arrive"] - 0.9
            P1 = D + np.array([600, -200, 900.0])
        else:
            D = ph[k - 1]["D"]
            fs = ph[k - 1]["depart"]
            P1 = D + ph[k - 1]["n"] * 900 + unit(ph[k]["Q"] - ph[k - 1]["Q"]) * 700
        P2 = A + ph[k]["n"] * 1100 - ph[k]["R"] @ np.array([350, 0, 0.0])
        return fs, ph[k]["arrive"], (D, P1, P2, A)

    def lead_at(self, k, tt):
        fs, fe, (P0, P1, P2, P3) = self.lead_path(k)
        u = np.clip((np.asarray(tt) - fs) / (fe - fs), 0, 1)
        e = smootherstep(u ** 0.85)
        return bezier(P0, P1, P2, P3, e), u

    def lead(self, t):
        for k in range(len(self.ph)):
            fs, fe, _ = self.lead_path(k)
            if fs <= t <= fe:
                p, _ = self.lead_at(k, t)
                return p, k
        return None, None

    # ------------------------------------------------------------------ render
    def frame(self, t, W, H, colour=None):
        cfg = self.cfg
        colour = cfg["colour"] if colour is None else colour
        sc = W / 360.0
        f_px = FOCAL * W / DW
        eye, tgt, up = self.camera(t)
        fwd = unit(tgt - eye)
        rt = unit(np.cross(fwd, up))
        upv = np.cross(rt, fwd)
        Mv = np.stack([rt, upv, fwd])

        def proj(X):
            d = (X - eye) @ Mv.T
            z = d[..., 2]
            ok = z > 40
            zz = np.where(ok, z, 1.0)
            x = W / 2 + f_px * d[..., 0] / zz
            y = H / 2 - f_px * d[..., 1] / zz
            return np.stack([x, y], -1), z, ok

        ink = np.zeros((H, W), np.float32)     # crisp lines
        hot = np.zeros((H, W), np.float32)     # heads (bloom)

        def lines(X, I, valid, buf, depth_fade=True):
            """X (n,m,3) polylines, I (n,m) intensity, valid (n,m) per point."""
            P, z, ok = proj(X)
            if depth_fade:
                I = I * np.clip(FOCAL * 1.15 / np.maximum(z, 1), 0.15, 1.0) ** 0.8
            v = valid & ok
            a, b = P[:, :-1].reshape(-1, 2), P[:, 1:].reshape(-1, 2)
            ia, ib = I[:, :-1].ravel(), I[:, 1:].ravel()
            vs = (v[:, :-1] & v[:, 1:]).ravel()
            inb = ((np.minimum(a[:, 0], b[:, 0]) < W + 4) & (np.maximum(a[:, 0], b[:, 0]) > -4) &
                   (np.minimum(a[:, 1], b[:, 1]) < H + 4) & (np.maximum(a[:, 1], b[:, 1]) > -4))
            sel = vs & inb & ((ia > 1e-3) | (ib > 1e-3))
            if not sel.any():
                return
            a, b, ia, ib = a[sel], b[sel], ia[sel], ib[sel]
            L = np.hypot(*(b - a).T)
            n = np.clip(np.ceil(L / 0.6).astype(np.int64), 1, 600)
            seg = np.repeat(np.arange(len(n)), n)
            k = np.arange(seg.size) - np.repeat(np.cumsum(n) - n, n)
            u = (k + 0.5) / n[seg]
            p = a[seg] + (b - a)[seg] * u[:, None]
            w = (ia[seg] + (ib - ia)[seg] * u) * (L[seg] / n[seg])
            splat(buf, p, w)

        def splat(buf, p, w):
            x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
            x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
            w = w.astype(np.float32)
            for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                               (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi, yi = x0 + dx, y0 + dy
                m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                buf += np.bincount(yi[m] * W + xi[m], weights=(w * ww)[m], minlength=W * H).reshape(H, W).astype(np.float32)

        fills = []
        K = cfg["K"]
        for k, P in enumerate(self.ph):
            if t < P["arrive"] - 0.05:
                continue
            Res = float(smoothstep(P["land"] - 0.30, P["land"] + 0.10, t))
            lv = 0.40 + 0.60 * Res
            echo = float(smoothstep(P["depart"] + 0.1, P["depart"] + 1.3, t))
            lv = lv * (1 - 0.62 * echo)
            for c in P["cs"]:
                tc = c.t0
                if t < tc:
                    continue
                prog = ease_out3((t - tc) / WRITE_T)
                cov = np.abs(c.ss) <= prog * c.L / 2 + 1e-6
                tpass = tc + WRITE_T * ease_out3_inv(2 * np.abs(c.ss) / c.L)
                age = np.where(cov, t - tpass, 0.0)
                r = cfg["R0"] * np.exp(-age / cfg["tau_r"]) * (1 - Res)
                th = cfg["m"] * c.s - cfg["w"] * t + c.psi
                j = np.arange(K)[:, None]
                thj = th[None] + 2 * np.pi * j / K
                Nl = np.stack([c.N[:, 0], -c.N[:, 1], np.zeros(c.M)], 1)
                if cfg["helix"]:
                    off = r[None, :, None] * (np.cos(thj)[..., None] * Nl[None] +
                                               np.sin(thj)[..., None] * np.array([0, 0, 1.0]))
                    shade = 1 - (r / cfg["R0"])[None] * (0.5 - 0.5 * np.sin(thj))
                else:
                    off = (r[None] * np.sin(thj))[..., None] * Nl[None]
                    shade = np.ones_like(thj)
                pts = np.vstack([c.p, c.p[:1]])
                offc = np.concatenate([off, off[:, :1]], 1)
                X = np.stack([self.to_world(k, pts, P["Q"], P["R"], offc[i]) for i in range(K)])
                heat = np.exp(-age / 0.22)
                I0 = lv + (1.0 - lv) * heat
                Ij = np.where(j == 0, I0[None], 0.75 * np.exp(-age / 0.45)[None] * (1 - Res)) * shade
                Ij = np.concatenate([Ij, Ij[:, :1]], 1)
                cv_ = np.concatenate([cov, cov[:1]])
                lines(X, Ij, np.repeat(cv_[None], K, 0), ink)
                # writer heads
                if prog < 1:
                    spd = (1 - np.clip((t - tc) / WRITE_T, 0, 1)) ** 2
                    for sgn in (1, -1):
                        sh = sgn * prog * c.L / 2
                        i = int(round((sh % c.L) / c.L * c.M)) % c.M
                        X = self.to_world(k, c.p[i:i + 1], P["Q"], P["R"])
                        Ph, z, ok = proj(X)
                        if ok[0]:
                            splat(hot, Ph, np.array([6.0 * spd + 0.4]) * sc)
                # runners: lights lapping the tube, speed -> 0
                for ru in c.runner[:cfg["runners"]]:
                    tau = t - tc - ru["delay"]
                    if tau <= 0:
                        continue
                    fade = np.exp(-tau / 0.55) * (1 - Res)
                    if fade < 0.02:
                        continue
                    taus = tau - np.linspace(0.0, 0.22, 26)
                    taus = np.maximum(taus, 0)
                    sr = ru["s0"] + ru["dir"] * ru["v"] * (1 - np.exp(-ru["lam"] * taus)) / ru["lam"]
                    fi = (sr % c.L) / c.L * c.M
                    i0 = np.floor(fi).astype(int) % c.M
                    i1 = (i0 + 1) % c.M
                    fr = (fi - np.floor(fi))[:, None]
                    p2 = c.p[i0] * (1 - fr) + c.p[i1] * fr
                    N2 = c.N[i0]
                    A = 22.0 * np.exp(-taus / 0.4) * (1 - Res)
                    tt = t - (tau - taus)
                    ph_ = 0.06 * sr - 13.0 * tt + ru["ph"]
                    Nl = np.stack([N2[:, 0], -N2[:, 1], np.zeros(len(N2))], 1)
                    if cfg["helix"]:
                        off = A[:, None] * (np.cos(ph_)[:, None] * Nl + np.sin(ph_)[:, None] * np.array([0, 0, 1.0]))
                    else:
                        off = (A * np.sin(ph_))[:, None] * Nl
                    X = self.to_world(k, p2, P["Q"], P["R"], off)
                    I = fade * np.exp(-(tau - taus) / 0.07) * 0.9
                    lines(X[None], I[None], np.ones((1, len(I)), bool), ink)
                    Ph, z, ok = proj(X[:1])
                    if ok[0]:
                        splat(hot, Ph, np.array([1.6 * fade]) * sc)
            # fill inside the closed outlines, fades in on resolve
            ff = float(smoothstep(P["land"], P["land"] + 0.6, t)) * (1 - float(smoothstep(P["depart"], P["depart"] + 0.35, t)))
            if ff > 0.01:
                fills.append((k, ff))

        # lead light + two echo followers spiralling round its path
        Lp, lk = self.lead(t)
        if lk is not None:
            fs, fe, _ = self.lead_path(lk)
            for e_i, (dl, rad, gain) in enumerate(((0.0, 0.0, 1.0), (0.05, 90.0, 0.6), (0.10, 150.0, 0.4))):
                tt = t - dl - np.linspace(0, 0.6, 48)
                pts, u = self.lead_at(lk, tt)
                if rad:
                    tg = unit(np.gradient(pts, axis=0))
                    n1 = unit(np.cross(tg, np.array([0.3, 1.0, 0.2])))
                    b1 = np.cross(tg, n1)
                    a_ = 9.0 * tt + e_i * 2.1
                    pts = pts + rad * np.sin(np.pi * u)[:, None] * (n1 * np.cos(a_)[:, None] + b1 * np.sin(a_)[:, None])
                I = gain * 2.6 * np.exp(-np.linspace(0, 0.6, 48) / 0.2) * ((u > 0) & (u < 1) | (np.arange(48) == 0))
                lines(pts[None], I[None], np.ones((1, 48), bool), ink, depth_fade=False)
                lines(pts[None], 1.0 * I[None], np.ones((1, 48), bool), hot, depth_fade=False)
                Ph, z, ok = proj(pts[:1])
                if ok[0] and gain == 1.0:
                    splat(hot, Ph, np.array([22.0]) * sc)
        # pre-departure flare on the last glyph
        for k, P in enumerate(self.ph):
            fl = float(smoothstep(P["depart"] - 0.25, P["depart"], t)) * (1 - float(smoothstep(P["depart"], P["depart"] + 0.05, t)))
            if fl > 0:
                Ph, z, ok = proj(P["D"][None])
                if ok[0]:
                    splat(hot, Ph, np.array([5.0 * fl]) * sc)

        # dust streaks (parallax)
        e0, t0_, _ = self.camera(t - 0.05)
        Pd, zd, okd = proj(self.dust)
        sv_eye, sv_t, sv_up = eye, tgt, up
        # previous-frame projection of the dust for a short motion streak
        fwd0 = unit(t0_ - e0)
        rt0 = unit(np.cross(fwd0, sv_up))
        Mv0 = np.stack([rt0, np.cross(rt0, fwd0), fwd0])
        d0 = (self.dust - e0) @ Mv0.T
        z0 = np.where(d0[:, 2] > 40, d0[:, 2], 1.0)
        P0 = np.stack([W / 2 + f_px * d0[:, 0] / z0, H / 2 - f_px * d0[:, 1] / z0], -1)
        okk = okd & (d0[:, 2] > 40)
        dpix = np.hypot(*(Pd - P0).T)
        okk &= dpix < 0.25 * W
        if okk.any():
            Xs = np.stack([P0[okk], Pd[okk]], 1)
            wv = self.dust_w[okk] * np.clip(2500 / zd[okk], 0.1, 1.0) * DUST_GAIN
            a, b = Xs[:, 0], Xs[:, 1]
            L = np.maximum(np.hypot(*(b - a).T), 0.6)
            n = np.clip(np.ceil(L / 0.6).astype(np.int64), 1, 400)
            seg = np.repeat(np.arange(len(n)), n)
            kk = np.arange(seg.size) - np.repeat(np.cumsum(n) - n, n)
            u = (kk + 0.5) / n[seg]
            p = a[seg] + (b - a)[seg] * u[:, None]
            w = wv[seg] * (L[seg] / n[seg])
            splat(ink, p, w)

        # tone
        s_c = 0.70 * sc
        core = cv2.GaussianBlur(ink, (0, 0), s_c) * (2.0 * sc)
        g1 = cv2.GaussianBlur(ink * 0.35 + hot, (0, 0), 2.2 * sc) * 0.9
        g2 = cv2.GaussianBlur(ink * 0.12 + hot, (0, 0), 7.0 * sc) * 0.9
        hotc = cv2.GaussianBlur(hot, (0, 0), 0.9 * sc) * 1.5
        Y = 1 - np.exp(-(core * 1.5 + g1 + g2 + hotc))
        # fills
        fillY = np.zeros((H, W), np.float32)
        for k, ff in fills:
            Hm = self.plane_h(k, proj, W, H)
            if Hm is None:
                continue
            m = cv2.warpPerspective(self.ph[k]["mask"], Hm, (W, H), flags=cv2.INTER_LINEAR)
            fillY = np.maximum(fillY, m * ff * (0.22 if not colour else 1.0))
        if not colour:
            Y = 1 - (1 - Y) * (1 - fillY)
        # inversion flood for the INVERT phrase
        inv = self.invert_mask(t, proj, W, H)
        if not colour:
            Y = Y + inv * (1 - 2 * Y)
            Y = np.clip(Y, 0, 1)
            g = np.random.default_rng(int(t * FPS) + 3).random((H, W), dtype=np.float32)
            out = np.clip(Y * 255 + (g - 0.5) * 1.5, 0, 255).astype(np.uint8)
            return np.repeat(out[..., None], 3, 2)
        return self.colourise(t, Y, fillY, inv, proj, W, H)

    def colourise(self, t, Y, fillY, inv, proj, W, H):
        raise NotImplementedError("colour is rendered by the torch path only (frame_t)")

    # ================================================================== torch renderer (GPU-first)
    # Same maths as frame(), evaluated as tensors: per-point arc-length writing, helix offsets, runners,
    # lead-light Bezier, dust, projection, segment densify + bilinear scatter-add splat (silkwind's
    # algorithm: repeat_interleave + index_put_(accumulate=True)), xp.gaussian_blur, grid_sample fills,
    # analytic SDF discs. The CPU only computes the 3x3 camera basis and 4-corner homographies per frame.
    def setup_torch(self, be):
        T = xp.torch()
        self.be = be
        dev = be.dev
        tt = lambda a, dt=None: T.as_tensor(np.ascontiguousarray(a), device=dev,
                                            dtype=dt or T.float32)
        self.tp = []
        for k, P in enumerate(self.ph):
            cs = P["cs"]
            Mc = np.array([c.M for c in cs])
            offc = np.concatenate([[0], np.cumsum(Mc)[:-1]])
            cid = np.repeat(np.arange(len(cs)), Mc)
            nxt = np.concatenate([o + (np.arange(m) + 1) % m for o, m in zip(offc, Mc)])
            pts = np.concatenate([c.p for c in cs])
            N2 = np.concatenate([c.N for c in cs])
            base = self.to_world(k, pts, P["Q"], P["R"])
            Nw = np.stack([N2[:, 0], -N2[:, 1], np.zeros(len(N2))], 1) @ P["R"].T
            Lc = np.array([c.L for c in cs])
            t0c = np.array([c.t0 for c in cs])
            ru = [[c.runner[j] for j in range(CFG["runners"])] for c in cs]
            g = lambda key: np.array([[r[key] for r in row] for row in ru], float)
            self.tp.append(dict(
                base=tt(base), Nw=tt(Nw), B=tt(P["n"]), s=tt(np.concatenate([c.s for c in cs])),
                ss=tt(np.concatenate([c.ss for c in cs])), L=tt(Lc[cid]), t0=tt(t0c[cid]),
                psi=tt(np.array([c.psi for c in cs])[cid]), nxt=tt(nxt, T.long),
                Lc=tt(Lc), t0c=tt(t0c), Mc=tt(Mc, T.long), offc=tt(offc, T.long),
                r_s0=tt(g("s0")), r_dir=tt(g("dir")), r_v=tt(g("v")), r_lam=tt(g("lam")),
                r_delay=tt(g("delay")), r_ph=tt(g("ph")),
                mask=tt(P["mask"])[None, None], key=tt(P["key"])[None, None]))
        self.dust_t = tt(self.dust)
        self.dust_wt = tt(self.dust_w)
        self._grids = {}

    def _grid(self, W, H):
        if (W, H) not in self._grids:
            T = xp.torch()
            yy, xx = T.meshgrid(T.arange(H, device=self.be.dev, dtype=T.float32),
                                T.arange(W, device=self.be.dev, dtype=T.float32), indexing="ij")
            self._grids[(W, H)] = (xx, yy)
        return self._grids[(W, H)]

    def frame_t(self, t, W, H, colour=None):
        T = xp.torch()
        dev = self.be.dev
        cfg = self.cfg
        colour = cfg["colour"] if colour is None else colour
        sc = W / 360.0
        f_px = FOCAL * W / DW
        eye_np, tgt_np, up_np = self.camera(t)
        fwd = unit(tgt_np - eye_np)
        rt = unit(np.cross(fwd, up_np))
        Mv_np = np.stack([rt, np.cross(rt, fwd), fwd])
        eye = T.as_tensor(eye_np, dtype=T.float32, device=dev)
        Mv = T.as_tensor(Mv_np, dtype=T.float32, device=dev)

        def proj(X, eye=eye, Mv=Mv):
            d = (X - eye) @ Mv.T
            z = d[..., 2]
            ok = z > 40
            zz = T.where(ok, z, T.ones_like(z))
            return T.stack([W / 2 + f_px * d[..., 0] / zz, H / 2 - f_px * d[..., 1] / zz], -1), z, ok

        SA, SB, SIA, SIB, SL = [], [], [], [], []      # segment endpoints (px), intensities, layer
        PP, PW, PL = [], [], []                        # point splats

        def add_segs(X, I, ia, ib, valid, layer=0, depth_fade=True):
            """X (M,3) points, I (M,), segments ia -> ib (index tensors) where valid."""
            P2, z, ok = proj(X)
            if depth_fade:
                I = I * T.clamp(FOCAL * 1.15 / T.clamp(z, min=1.0), 0.15, 1.0) ** 0.8
            v = valid & ok[ia] & ok[ib]
            ia, ib = ia[v], ib[v]
            SA.append(P2[ia]); SB.append(P2[ib]); SIA.append(I[ia]); SIB.append(I[ib])
            SL.append(T.full((ia.numel(),), layer, dtype=T.long, device=dev))

        def add_pts(X, w, layer=1):
            P2, z, ok = proj(X)
            PP.append(P2[ok]); PW.append(w[ok]); PL.append(T.full((int(ok.sum()),), layer, dtype=T.long, device=dev))

        K = cfg["K"]
        fills = []
        for k, (P, tp) in enumerate(zip(self.ph, self.tp)):
            if t < P["arrive"] - 0.05:
                continue
            Res = float(smoothstep(P["land"] - 0.30, P["land"] + 0.10, t))
            echo = float(smoothstep(P["depart"] + 0.1, P["depart"] + 1.3, t))
            lv = (0.40 + 0.60 * Res) * (1 - 0.62 * echo)
            tc = tp["t0"]
            active = tc <= t
            x = T.clamp((t - tc) / WRITE_T, 0, 1)
            prog = 1 - (1 - x) ** 3
            aL = tp["ss"].abs()
            cov = active & (aL <= prog * tp["L"] / 2 + 1e-6)
            tpass = tc + WRITE_T * (1 - (1 - T.clamp(2 * aL / tp["L"], 0, 1)) ** (1 / 3))
            age = T.where(cov, t - tpass, T.zeros_like(tpass))
            r = cfg["R0"] * T.exp(-age / cfg["tau_r"]) * (1 - Res)
            th = cfg["m"] * tp["s"] - cfg["w"] * t + tp["psi"]
            heat = T.exp(-age / 0.22)
            I0 = lv + (1.0 - lv) * heat
            ia = T.arange(tc.numel(), device=dev)
            ib = tp["nxt"]
            valid = cov[ia] & cov[ib]
            for j in range(K):
                thj = th + 2 * np.pi * j / K
                if cfg["helix"]:
                    off = r[:, None] * (T.cos(thj)[:, None] * tp["Nw"] + T.sin(thj)[:, None] * tp["B"])
                    shade = 1 - (r / cfg["R0"]) * (0.5 - 0.5 * T.sin(thj))
                else:
                    off = (r * T.sin(thj))[:, None] * tp["Nw"]
                    shade = T.ones_like(thj)
                I = (I0 if j == 0 else 0.75 * T.exp(-age / 0.45) * (1 - Res)) * shade
                add_segs(tp["base"] + off, I, ia, ib, valid)
            # writer heads (two per contour, one each way round)
            xc = T.clamp((t - tp["t0c"]) / WRITE_T, 0, 1)
            pc = 1 - (1 - xc) ** 3
            live = (tp["t0c"] <= t) & (pc < 1)
            if bool(live.any()):
                spd = (1 - xc) ** 2
                Mf = tp["Mc"].float()
                for sgn in (1.0, -1.0):
                    sh = T.remainder(sgn * pc * tp["Lc"] / 2, tp["Lc"])
                    i = T.remainder(T.round(sh / tp["Lc"] * Mf).long(), tp["Mc"]) + tp["offc"]
                    add_pts(tp["base"][i[live]], ((6.0 * spd + 0.4) * sc)[live])
            # runners: lights lapping the tube, speed -> 0
            tau = t - tp["t0c"][:, None] - tp["r_delay"]                         # (C,J)
            fade = T.exp(-tau / 0.55) * (1 - Res)
            okr = (tau > 0) & (fade >= 0.02)
            if bool(okr.any()):
                lin = T.linspace(0.0, 0.22, 26, device=dev)
                taus = T.clamp(tau[..., None] - lin, min=0)                       # (C,J,26)
                lam = tp["r_lam"][..., None]
                sr = tp["r_s0"][..., None] + tp["r_dir"][..., None] * tp["r_v"][..., None] * (1 - T.exp(-lam * taus)) / lam
                Lc = tp["Lc"][:, None, None]
                Mf = tp["Mc"].float()[:, None, None]
                fi = T.remainder(sr, Lc) / Lc * Mf
                fl = T.floor(fi)
                fr = (fi - fl)[..., None]
                i0 = T.remainder(fl.long(), tp["Mc"][:, None, None]) + tp["offc"][:, None, None]
                i1 = T.remainder(fl.long() + 1, tp["Mc"][:, None, None]) + tp["offc"][:, None, None]
                p3 = tp["base"][i0] * (1 - fr) + tp["base"][i1] * fr
                N3 = tp["Nw"][i0]
                A = 22.0 * T.exp(-taus / 0.4) * (1 - Res)
                ttm = t - (tau[..., None] - taus)
                ph_ = 0.06 * sr - 13.0 * ttm + tp["r_ph"][..., None]
                if cfg["helix"]:
                    off = A[..., None] * (T.cos(ph_)[..., None] * N3 + T.sin(ph_)[..., None] * tp["B"])
                else:
                    off = (A * T.sin(ph_))[..., None] * N3
                X = (p3 + off).reshape(-1, 3)
                I = (fade[..., None] * T.exp(-(tau[..., None] - taus) / 0.07) * 0.9).reshape(-1)
                n = X.shape[0]
                ia = T.arange(n, device=dev).reshape(-1, 26)[:, :-1].reshape(-1)
                vr = okr[..., None].expand(-1, -1, 25).reshape(-1)
                add_segs(X, I, ia, ia + 1, vr)
                hp = (p3 + off)[..., 0, :][okr]
                add_pts(hp, (1.6 * fade[okr]) * sc)
            ff = float(smoothstep(P["land"], P["land"] + 0.6, t)) * (1 - float(smoothstep(P["depart"], P["depart"] + 0.35, t)))
            if ff > 0.01:
                fills.append((k, ff))

        # lead light + two echo followers spiralling round its path
        Lp, lk = self.lead(t)
        if lk is not None:
            fs, fe, ctrl = self.lead_path(lk)
            C4 = [T.as_tensor(c, dtype=T.float32, device=dev) for c in ctrl]
            ages = T.linspace(0, 0.6, 48, device=dev)
            for e_i, (dl, rad, gain) in enumerate(((0.0, 0.0, 1.0), (0.05, 90.0, 0.6), (0.10, 150.0, 0.4))):
                tt_ = t - dl - ages
                u = T.clamp((tt_ - fs) / (fe - fs), 0, 1)
                e = T.clamp(u ** 0.85, 0, 1)
                e = e * e * e * (e * (6 * e - 15) + 10)
                v = (1 - e)[:, None]
                ee = e[:, None]
                pts = v ** 3 * C4[0] + 3 * v * v * ee * C4[1] + 3 * v * ee * ee * C4[2] + ee ** 3 * C4[3]
                if rad:
                    tg = T.gradient(pts, dim=0)[0]
                    tg = tg / (T.linalg.vector_norm(tg, dim=1, keepdim=True) + 1e-12)
                    n1 = T.linalg.cross(tg, T.tensor([0.3, 1.0, 0.2], device=dev).expand_as(tg), dim=1)
                    n1 = n1 / (T.linalg.vector_norm(n1, dim=1, keepdim=True) + 1e-12)
                    b1 = T.linalg.cross(tg, n1, dim=1)
                    a_ = 9.0 * tt_ + e_i * 2.1
                    pts = pts + (rad * T.sin(np.pi * u))[:, None] * (n1 * T.cos(a_)[:, None] + b1 * T.sin(a_)[:, None])
                I = gain * 2.6 * T.exp(-ages / 0.2) * (((u > 0) & (u < 1)) | (T.arange(48, device=dev) == 0)).float()
                ia = T.arange(47, device=dev)
                vv = T.ones(47, dtype=T.bool, device=dev)
                add_segs(pts, I, ia, ia + 1, vv, 0, False)
                add_segs(pts, I, ia, ia + 1, vv, 1, False)
                if gain == 1.0:
                    add_pts(pts[:1], T.tensor([22.0 * sc], device=dev))
        # pre-departure flare on the last glyph
        for k, P in enumerate(self.ph):
            fl = float(smoothstep(P["depart"] - 0.25, P["depart"], t)) * (1 - float(smoothstep(P["depart"], P["depart"] + 0.05, t)))
            if fl > 0:
                add_pts(T.as_tensor(P["D"][None], dtype=T.float32, device=dev), T.tensor([5.0 * fl * sc], device=dev))

        # dust streaks (camera motion over the last 0.05 s)
        e0, t0_, _ = self.camera(t - 0.05)
        fwd0 = unit(t0_ - e0)
        rt0 = unit(np.cross(fwd0, up_np))
        Mv0 = T.as_tensor(np.stack([rt0, np.cross(rt0, fwd0), fwd0]), dtype=T.float32, device=dev)
        Pd, zd, okd = proj(self.dust_t)
        P0, z0, ok0 = proj(self.dust_t, T.as_tensor(e0, dtype=T.float32, device=dev), Mv0)
        okk = okd & ok0 & (T.linalg.vector_norm(Pd - P0, dim=1) < 0.25 * W)
        wv = self.dust_wt * T.clamp(2500 / T.clamp(zd, min=1.0), 0.1, 1.0) * DUST_GAIN
        SA.append(P0[okk]); SB.append(Pd[okk]); SIA.append(wv[okk]); SIB.append(wv[okk])
        SL.append(T.zeros(int(okk.sum()), dtype=T.long, device=dev))

        buf = self._splat(SA, SB, SIA, SIB, SL, PP, PW, PL, W, H)
        ink, hot = buf[0], buf[1]
        G = xp.gaussian_blur
        core = G(ink, 0.70 * sc) * (2.0 * sc)
        g1 = G(ink * 0.35 + hot, 2.2 * sc) * 0.9
        g2 = G(ink * 0.12 + hot, 7.0 * sc) * 0.9
        hotc = G(hot, 0.9 * sc) * 1.5
        Yl = 1 - T.exp(-(core * 1.5 + g1 + g2 + hotc))          # light layer (lines, heads, trails)

        # fills (grid_sample through the inverse plane homography)
        fillG = T.zeros((H, W), device=dev)
        keyF = T.zeros((H, W), device=dev)
        keyC = T.zeros((H, W, 3), device=dev)
        for k, ff in fills:
            Hm = self.plane_h(k, lambda X: self._proj_np(X, eye_np, Mv_np, f_px, W, H), W, H)
            if Hm is None:
                continue
            tp = self.tp[k]
            m_all = self._warp(tp["mask"], Hm, W, H)
            if colour:
                m_key = self._warp(tp["key"], Hm, W, H)
                fillG = T.maximum(fillG, T.clamp(m_all - m_key, 0, 1) * ff * 0.22)
                blk = T.clamp((m_key - 0.5) * 4 + 0.5, 0, 1) * (ff > 0.85)        # hard-edged block
                keyF = T.maximum(keyF, blk)
                keyC = T.where(blk[..., None] > 0, T.as_tensor(KEY_COLS[k], device=dev).expand(H, W, 3), keyC)
            else:
                fillG = T.maximum(fillG, m_all * ff * 0.22)
        inv = self._invert_t(t, eye_np, Mv_np, f_px, W, H)
        xx, yy = self._grid(W, H)
        # tonal ground: grey fill, then the inversion flip
        base = fillG + inv * (1 - 2 * fillG)
        if not colour:
            Y = base * (1 - Yl) + Yl * (1 - inv)
            Y = T.clamp(Y, 0, 1)
            g = self._hash(xx, yy, int(round(t * FPS)))
            out = T.clamp(Y * 255 + (g - 0.5) * 1.5, 0, 255).to(T.uint8)
            return out[..., None].expand(-1, -1, 3).contiguous().cpu().numpy()
        # colour: night ground, flat palette blocks inside the key word, solid landing discs, then lines
        night = T.as_tensor(NIGHT, device=dev)
        img = night + base[..., None] * (1 - night)
        disc, dcol = self._discs_t(t, eye_np, Mv_np, f_px, W, H)
        img = img * (1 - disc[..., None]) + dcol * disc[..., None]
        img = img * (1 - keyF[..., None]) + keyC * keyF[..., None]
        linec = (1 - inv)[..., None] * T.ones(3, device=dev) + inv[..., None] * night
        img = img * (1 - Yl[..., None]) + linec * Yl[..., None]
        g = self._hash(xx, yy, int(round(t * FPS)))[..., None]
        out = T.clamp(img * 255 + (g - 0.5) * 1.5, 0, 255).to(T.uint8)
        return out.contiguous().cpu().numpy()

    @staticmethod
    def _proj_np(X, eye, Mv, f_px, W, H):
        d = (X - eye) @ Mv.T
        z = d[..., 2]
        ok = z > 40
        zz = np.where(ok, z, 1.0)
        return np.stack([W / 2 + f_px * d[..., 0] / zz, H / 2 - f_px * d[..., 1] / zz], -1), z, ok

    def _splat(self, SA, SB, SIA, SIB, SL, PP, PW, PL, W, H, nlayer=2):
        """Densify segments (<= 0.6 px steps) and scatter-add bilinear weights into (nlayer, H, W)."""
        T = xp.torch()
        dev = self.be.dev
        A, B = T.cat(SA), T.cat(SB)
        IA, IB, LY = T.cat(SIA), T.cat(SIB), T.cat(SL)
        lo, hi = T.minimum(A, B), T.maximum(A, B)
        keep = ((IA > 1e-3) | (IB > 1e-3)) & (lo[:, 0] < W + 4) & (hi[:, 0] > -4) & (lo[:, 1] < H + 4) & (hi[:, 1] > -4)
        A, B, IA, IB, LY = A[keep], B[keep], IA[keep], IB[keep], LY[keep]
        D = B - A
        ln = T.linalg.vector_norm(D, dim=1)
        ns = T.clamp(T.ceil(ln / 0.6), 1, 600).long()
        rep = T.repeat_interleave(ns)
        start = (T.cumsum(ns, 0) - ns)[rep]
        f = ((T.arange(rep.numel(), device=dev) - start).float() + 0.5) / ns[rep].float()
        x = A[rep, 0] + D[rep, 0] * f
        y = A[rep, 1] + D[rep, 1] * f
        w = (IA[rep] + (IB - IA)[rep] * f) * (ln / ns.float())[rep]
        ly = LY[rep]
        if PP:
            pp = T.cat(PP)
            x = T.cat([x, pp[:, 0]]); y = T.cat([y, pp[:, 1]])
            w = T.cat([w, T.cat(PW)]); ly = T.cat([ly, T.cat(PL)])
        x0, y0 = T.floor(x - 0.5), T.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        x0, y0 = x0.long(), y0.long()
        buf = T.zeros(nlayer * (H + 2) * (W + 2), device=dev)
        ok = (x0 >= -1) & (y0 >= -1) & (x0 < W) & (y0 < H)
        x0, y0, fx, fy, w, ly = x0[ok] + 1, y0[ok] + 1, fx[ok], fy[ok], w[ok], ly[ok]
        gi = ly * ((H + 2) * (W + 2)) + y0 * (W + 2) + x0
        idx = T.cat([gi, gi + 1, gi + W + 2, gi + W + 3])
        val = T.cat([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
        buf.index_put_((idx,), val, accumulate=True)
        return buf.view(nlayer, H + 2, W + 2)[:, 1:-1, 1:-1].contiguous()

    def _warp(self, m, Hm, W, H):
        """cv2.warpPerspective(m, Hm) (mask px -> screen px) as grid_sample through Hm^-1."""
        T = xp.torch()
        Hi = T.as_tensor(np.linalg.inv(Hm), dtype=T.float32, device=self.be.dev)
        xx, yy = self._grid(W, H)
        den = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
        u = (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / den
        v = (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / den
        mh, mw = m.shape[-2:]
        g = T.stack([(2 * u + 1) / mw - 1, (2 * v + 1) / mh - 1], -1)[None]
        return T.nn.functional.grid_sample(m, g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]

    def _invert_t(self, t, eye, Mv, f_px, W, H):
        T = xp.torch()
        P = self.ph[INVERT]
        xx, yy = self._grid(W, H)
        if t < P["arrive"] or t > P["depart"] + 0.6:
            return T.zeros_like(xx)
        diag = float(np.hypot(W, H))
        soft = 1.5 * W / 360
        pr = lambda X: self._proj_np(X[None], eye, Mv, f_px, W, H)[0][0]
        r1 = float(smootherstep((t - P["arrive"]) / 0.5)) * 1.25 * diag
        c1 = pr(P["A"])
        m1 = T.clamp((r1 - T.hypot(xx - float(c1[0]), yy - float(c1[1]))) / soft + 0.5, 0, 1)
        if t > P["depart"]:
            r2 = float(smootherstep((t - P["depart"]) / 0.5)) * 1.25 * diag
            L, _ = self.lead(t)
            c2 = pr(L if L is not None else P["D"])
            m2 = T.clamp((r2 - T.hypot(xx - float(c2[0]), yy - float(c2[1]))) / soft + 0.5, 0, 1)
            m1 = m1 * (1 - m2)
        return m1

    def _discs_t(self, t, eye, Mv, f_px, W, H):
        """Solid dawn discs where the lead light lands: pops on arrival, holds, sinks on resolve (SDF)."""
        T = xp.torch()
        xx, yy = self._grid(W, H)
        dev = self.be.dev
        acc = T.zeros_like(xx)
        col = T.zeros((H, W, 3), device=dev)
        for k, P in enumerate(self.ph):
            a = t - P["arrive"]
            if a < 0 or a > (P["land"] - P["arrive"]) + 0.9:
                continue
            grow = float(1 - (1 - min(a / 0.18, 1.0)) ** 3)
            sink = float(smoothstep(P["land"] - P["arrive"] + 0.2, P["land"] - P["arrive"] + 0.9, a))
            Pp, z, ok = self._proj_np(P["A"][None], eye, Mv, f_px, W, H)
            if not ok[0]:
                continue
            R = DISC_R * f_px / z[0] * grow * (1 - sink)
            if R < 0.5:
                continue
            d = T.hypot(xx - float(Pp[0, 0]), yy - float(Pp[0, 1])) - R
            m = T.clamp(0.5 - d, 0, 1)
            acc = T.maximum(acc, m)
            col = T.where(m[..., None] > 0, T.as_tensor(DISC_COLS[k], device=dev).expand(H, W, 3), col)
        # the first light: a flat amber sun rises behind the key word of the inverted phrase
        P = self.ph[INVERT]
        a = t - P["land"]
        if -0.35 < a < (P["depart"] - P["land"]) + 0.3:
            rise = float(smootherstep((a + 0.35) / 0.6))
            sink = float(smoothstep(P["depart"] - P["land"] - 0.05, P["depart"] - P["land"] + 0.3, a))
            kx, ky = P["key_c"]
            c3 = self.to_world(INVERT, np.array([[kx, ky + SUN_DROP * (1 - rise)]]), P["Q"], P["R"])
            Pp, z, ok = self._proj_np(c3, eye, Mv, f_px, W, H)
            R = SUN_R * f_px / z[0] * (0.92 + 0.08 * rise) * (1 - sink)
            if ok[0] and R > 0.5 and rise > 0:
                d = T.hypot(xx - float(Pp[0, 0]), yy - float(Pp[0, 1])) - R
                m = T.clamp(0.5 - d, 0, 1)
                acc = T.maximum(acc, m)
                col = T.where(m[..., None] > 0, T.as_tensor(AMBER, device=dev).expand(H, W, 3), col)
        # somewhere to land: the light finally lands as a full stop after the last word and stays
        P = self.ph[-1]
        a = t - (P["land"] + 0.45)
        if a > 0:
            grow = float(1 - (1 - min(a / 0.22, 1.0)) ** 4)
            ys, xs = np.nonzero(P["key"] > 0.5)
            c3 = self.to_world(len(self.ph) - 1, np.array([[xs.max() * 2 + 30 + STOP_R, ys.max() * 2 - STOP_R]]), P["Q"], P["R"])
            Pp, z, ok = self._proj_np(c3, eye, Mv, f_px, W, H)
            R = STOP_R * f_px / z[0] * grow
            if ok[0] and R > 0.5:
                d = T.hypot(xx - float(Pp[0, 0]), yy - float(Pp[0, 1])) - R
                m = T.clamp(0.5 - d, 0, 1)
                acc = T.maximum(acc, m)
                col = T.where(m[..., None] > 0, T.as_tensor(CORAL, device=dev).expand(H, W, 3), col)
        return acc, col

    @staticmethod
    def _hash(xx, yy, n):
        T = xp.torch()
        v = T.sin(xx * 12.9898 + yy * 78.233 + n * 37.719) * 43758.5453
        return v - T.floor(v)

    def plane_h(self, k, proj, W, H):
        P = self.ph[k]
        src = np.array([[0, 0], [DW, 0], [DW, DH], [0, DH]], np.float64)
        X = self.to_world(k, src, P["Q"], P["R"])
        Pp, z, ok = proj(X)
        if not ok.all():
            return None
        return cv2.getPerspectiveTransform((src / 2).astype(np.float32), Pp.astype(np.float32))

    def invert_mask(self, t, proj, W, H):
        P = self.ph[INVERT]
        if t < P["arrive"] or t > P["depart"] + 0.6:
            return 0.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        diag = float(np.hypot(W, H))
        r1 = float(smootherstep((t - P["arrive"]) / 0.5)) * 1.25 * diag
        c1, _, ok1 = proj(P["A"][None])
        m1 = np.clip((r1 - np.hypot(xx - c1[0, 0], yy - c1[0, 1])) / (1.5 * W / 360) + 0.5, 0, 1)
        if t > P["depart"]:
            r2 = float(smootherstep((t - P["depart"]) / 0.5)) * 1.25 * diag
            L, _ = self.lead(t)
            c2 = proj((L if L is not None else P["D"])[None])[0]
            m2 = np.clip((r2 - np.hypot(xx - c2[0, 0], yy - c2[0, 1])) / (1.5 * W / 360) + 0.5, 0, 1)
            m1 = m1 * (1 - m2)
        return m1


# ---------------------------------------------------------------------------------------------- output
_SC = None


BACKEND = ("numpy", "cpu")


def _init(variant, preview, backend=("numpy", "cpu"), threads=1):
    global _SC, CFG, PREVIEW, BACKEND
    CFG = VARIANTS[variant]
    PREVIEW = preview
    BACKEND = backend
    _SC = Scene()
    if backend[0] == "torch":
        xp.torch().set_num_threads(threads)
        _SC.setup_torch(xp.Backend(*backend))


def _work(args):
    n, W, H = args
    f = _SC.frame_t if BACKEND[0] == "torch" else _SC.frame
    return f(n / FPS, W, H).tobytes()


def render(path, W, H, variant, procs):
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if BACKEND[1].startswith("cuda"):          # one process drives the GPU; ffmpeg encodes alongside
        _init(variant, PREVIEW, BACKEND)
        for n in range(NFR):
            proc.stdin.write(_work((n, W, H)))
    else:
        with mp.Pool(procs, initializer=_init, initargs=(variant, PREVIEW, BACKEND, 1)) as pool:
            for buf in pool.imap(_work, [(n, W, H) for n in range(NFR)], chunksize=2):
                proc.stdin.write(buf)
    proc.stdin.close()
    t1 = time.time()
    proc.wait()
    t2 = time.time()
    dt = t2 - t0
    print(f"{path}  {W}x{H}  {NFR} frames  {dt:.1f}s")
    return t1 - t0, t2 - t1


TIMINGS_HDR = "gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,encode_s,audio_s,notes\n"


def log_timing(nn, test, kind, device, frames, res, started, finished, render_s="", encode_s="", audio_s="",
               make_min="", notes=""):
    """Append one measured row to local/TIMINGS.csv (timing rule)."""
    path = os.path.join(LOCAL, "TIMINGS.csv")
    new = not os.path.exists(path)
    pf = f"{render_s * 1000 / frames:.1f}" if render_s != "" and frames else ""
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(TIMINGS_HDR)
        f.write(",".join([f"g{nn:02d}", test, kind, device, str(frames), res, started, finished, fmt(make_min),
                          fmt(render_s), pf, fmt(encode_s), fmt(audio_s), notes.replace(",", ";")]) + "\n")


def nz_now():
    import gens
    import datetime
    t = datetime.datetime.now(gens._NZ) if gens._NZ else datetime.datetime.now()
    return t.strftime("%Y-%m-%d %H:%M:%S")


def qc(video):
    """Contact sheet (5 fps) + a strip across each transition, into review/."""
    os.makedirs(REVIEW, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    sheet = os.path.join(REVIEW, f"{base}_sheet.jpg")
    if not os.path.exists(sheet):
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", video, "-vf",
                        "fps=5,scale=180:-2,tile=10x5", "-frames:v", "1", sheet], check=True)
    out = [sheet]
    for k in range(1, len(TIMES)):
        a, b = TIMES[k - 1][2] - 0.2, TIMES[k][1] + 0.2
        strip = os.path.join(REVIEW, f"{base}_strip{k}.jpg")
        if not os.path.exists(strip):
            step = (b - a) / 7
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{b - a + 0.05:.3f}",
                            "-i", video, "-vf", f"fps={1 / step:.4f},scale=200:-2,tile=8x1", "-frames:v", "1", strip],
                           check=True)
        out.append(strip)
    print("\n".join(out))
    return out


def audio_for(nn, test):
    lands = ",".join(f"{x[1]:g}" for x in TIMES)
    score = os.path.join(HERE, "score.py")
    if os.path.exists(score):
        wav = os.path.join(LOCAL, f"score_{test}_g{nn:02d}.wav")
        r = subprocess.run([sys.executable, score, "--preset", "E2", "--duration", f"{DUR:g}", "--changes", lands,
                            "--out", wav])
        if r.returncode == 0 and os.path.exists(wav):
            return wav
    wav = os.path.join(LOCAL, f"chords_{test}_g{nn:02d}.wav")
    if not os.path.exists(wav):
        subprocess.run([sys.executable, os.path.join(HERE, "chords.py"), "--duration", f"{DUR:g}",
                        "--changes", lands, "--out", wav], check=True)
    return wav


def main():
    global CFG, PREVIEW
    import gens
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="b", choices=sorted(VARIANTS))
    ap.add_argument("--preview", action="store_true", help="360x450 iteration render")
    ap.add_argument("--final", action="store_true", help="720x900 final + audio + mux")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--qc", default=None, help="video to build contact sheet + strips for")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--device-label", default=None)
    ap.add_argument("--backend", default="torch", choices=["numpy", "torch"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gen", type=int, default=None, help="force generation number (box runs, where local/ lacks the history)")
    ap.add_argument("--bench", type=float, nargs="*", help="time frame_t at these t (final size) and exit")
    ap.add_argument("--make-min", type=float, default=None, help="wall minutes since work on this gen began")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", default=None)
    a = ap.parse_args()
    CFG = VARIANTS[a.variant]
    PREVIEW = a.preview
    global BACKEND
    BACKEND = (a.backend, a.device)
    if a.device_label is None:
        a.device_label = ("cuda-RTX3060" if a.device.startswith("cuda") else "local-cpu-torch") if a.backend == "torch" else "local-numpy"
    if a.bench:
        _init(a.variant, False, BACKEND, 8)
        f = _SC.frame_t if a.backend == "torch" else _SC.frame
        f(a.bench[0], 720, 900)
        for t in a.bench:
            t0 = time.perf_counter()
            f(t, 720, 900)
            if a.backend == "torch":
                _SC.be.sync()
            print(f"bench t={t:.2f} 720x900 {a.backend}/{a.device}: {1000 * (time.perf_counter() - t0):.1f} ms")
        return
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    test, stem = CFG["test"], CFG["stem"]
    h = gens.code_hash([os.path.abspath(__file__)]) + "-" + a.variant
    if a.qc:
        qc(a.qc)
    stills_t = [TIMES[1][0] - 0.15, TIMES[1][1] + 0.6]
    for kind, (W, H), on in (("preview", (360, 450), a.preview), ("final", (720, 900), a.final)):
        if not on:
            continue
        nn = a.gen if a.gen is not None else gens.gen_for(LOCAL, test, stem, kind, h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, kind)
        started = nz_now()
        rs, es = render(os.path.join(LOCAL, name), W, H, a.variant, a.procs)
        log_timing(nn, test, kind, a.device_label, NFR, f"{W}x{H}", started, nz_now(), rs, es,
                   make_min=a.make_min if a.make_min is not None else "",
                   notes=f"procs={a.procs}; render_s = frame loop incl. piped encode; encode_s = encoder tail")
        _init(a.variant, PREVIEW, BACKEND, 8)
        stills = []
        for t in stills_t:
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            if not os.path.exists(os.path.join(REVIEW, fn)):
                Image.fromarray(np.frombuffer(_work((t * FPS, 540, 675)), np.uint8).reshape(675, 540, 3)).save(
                    os.path.join(REVIEW, fn), quality=90)
            stills.append(fn)
        files = [name]
        if kind == "final":
            st_a, t0a = nz_now(), time.perf_counter()
            wav = audio_for(nn, test)
            log_timing(nn, test, "music", "box-cpu" if a.device.startswith("cuda") else "local-cpu", 0, "", st_a,
                       nz_now(), audio_s=time.perf_counter() - t0a, notes=os.path.basename(wav))
            mux = f"{stem}_g{nn:02d}_final_audio.mp4"
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", os.path.join(LOCAL, name), "-i", wav,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", os.path.join(LOCAL, mux)],
                           check=True)
            files += [os.path.basename(wav), mux]
        if a.gen is None:
            gens.add_files(LOCAL, test, nn, files, stills)
        else:
            print("forced gen: GENERATIONS.md not updated here;", files, stills)
        qc(os.path.join(LOCAL, name))
    if a.still:
        _init(a.variant, PREVIEW, BACKEND, 8)
        for t in a.still:
            Image.fromarray(np.frombuffer(_work((t * FPS, 360, 450)), np.uint8).reshape(450, 360, 3)).save(
                os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))


if __name__ == "__main__":
    main()
