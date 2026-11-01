"""Test C, "Chladni strings": sand on a vibrating square plate settles on the nodal lines of a
Chladni mode, then the same grains flow into each phrase of the Pythagoras quote.

Field  f(x,y) = cos(n*pi*x)*cos(m*pi*y) - cos(m*pi*x)*cos(n*pi*y),  x,y in [-0.5,0.5] (plate centred).
Modes (2,3) -> P1, (3,5) -> P2, (4,7) -> P3 (rising with the harmonic of each pluck in hum.wav).

Every grain position is a closed-form sum of damped-spring steps (zeta = 0.6) between stage
targets (scatter -> C1 -> T1 -> C2 -> T2 -> C3 -> T3); stages are paired by optimal assignment
(scipy linear_sum_assignment), so each grain makes a short, non-crossing move. frame(t) is a pure
function of t; the targets are precomputed once from a fixed seed.

Usage:
  python tools/mathtype/chladni.py draft      -> local/C_chladni_draft.mp4 (360x450)
  python tools/mathtype/chladni.py final      -> local/C_chladni.mp4 (1080x1350) (+ _hum mux)
  python tools/mathtype/chladni.py still T..  -> review/C_still_<t>.png at draft size
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(REPO, "11_mathtype", "v001", "local")
REVIEW = os.path.join(REPO, "11_mathtype", "v001", "review")
FONT = os.path.join(REPO, "mograph", "fonts", "Interlope-Regular.ttf")   # registry: "Interlope" (liked; was Bodoni Moda)

FPS, NFR, DUR = 25, 125, 5.0
ASPECT = 1.25                     # H / W (4:5); all geometry in units of frame width
BG = np.array([0x05, 0x05, 0x05], np.float32) / 255
INK = np.array([0xf3, 0xea, 0xdb], np.float32) / 255
ACC = np.array([0xe2, 0x58, 0x2a], np.float32) / 255

NG = 6000                         # grains
SEED = 31415
PLATE = 0.88                      # plate side (frame widths)
PCX, PCY = 0.5, ASPECT / 2        # plate centre
ZETA = 0.6

PHRASES = [("THERE IS", "GEOMETRY"), ("IN THE", "HUMMING"), ("OF THE", "STRINGS")]
MODES = [(2, 3), (3, 5), (4, 7)]
HARM = [2, 3, 4]                  # pluck harmonic per phrase (hum.py)
PLUCK_T = [0.35, 1.95, 3.55]      # pluck times in hum.wav
PHRASE_T = [0.0, 1.60, 3.20]
CREDIT_T = 4.2

# stage schedule: (start time, 97%-settle time, ripple span from centre to edge)
SNAP = [(0.00, 0.17, 0.05), (1.60, 0.12, 0.03), (3.20, 0.12, 0.03)]   # -> Chladni figure k
FLOW = [(0.22, 0.18, 0.04), (1.84, 0.17, 0.03), (3.44, 0.17, 0.03)]   # figure k -> phrase k


# ----------------------------------------------------------------------------- field
def chladni(u, v, n, m):
    u, v = u - 0.5, v - 0.5                                    # centred plate: classic symmetric figures
    return np.cos(n * np.pi * u) * np.cos(m * np.pi * v) - np.cos(m * np.pi * u) * np.cos(n * np.pi * v)


def chladni_grad(u, v, n, m):
    u, v = u - 0.5, v - 0.5
    pi = np.pi
    fu = -n * pi * np.sin(n * pi * u) * np.cos(m * pi * v) + m * pi * np.sin(m * pi * u) * np.cos(n * pi * v)
    fv = -m * pi * np.cos(n * pi * u) * np.sin(m * pi * v) + n * pi * np.cos(m * pi * u) * np.sin(n * pi * v)
    return fu, fv


def to_frame(u, v):
    return PCX - PLATE / 2 + u * PLATE, PCY - PLATE / 2 + v * PLATE


def to_plate(x, y):
    return (x - (PCX - PLATE / 2)) / PLATE, (y - (PCY - PLATE / 2)) / PLATE


# ----------------------------------------------------------------------------- targets
def nodal_targets(n, m, rng, band=0.0042):
    """Stratified sample of the nodal set: jittered grid, keep points within `band` of a nodal line."""
    g = 700
    ii, jj = np.meshgrid(np.arange(g), np.arange(g))
    u = (ii.ravel() + rng.uniform(0.1, 0.9, g * g)) / g
    v = (jj.ravel() + rng.uniform(0.1, 0.9, g * g)) / g
    f = chladni(u, v, n, m)
    fu, fv = chladni_grad(u, v, n, m)
    d = np.abs(f) / np.maximum(np.hypot(fu, fv), 1e-6)       # first-order distance to the nodal line
    keep = np.flatnonzero(d < band)
    # keep a small margin off the plate edge
    keep = keep[(u[keep] > 0.01) & (u[keep] < 0.99) & (v[keep] > 0.01) & (v[keep] < 0.99)]
    pick = rng.choice(keep, NG, replace=keep.size < NG)
    x, y = to_frame(u[pick], v[pick])
    return np.stack([x, y], 1)


def text_mask(lines, ref_w=1080):
    """Two-line phrase mask at reference resolution; returns bool array (ref_w*1.25, ref_w) and font size."""
    W, H = ref_w, int(round(ref_w * ASPECT))
    target_w = 0.86 * W
    size = 200
    for _ in range(3):
        font = ImageFont.truetype(FONT, size)
        font.set_variation_by_axes([900, 11])                  # wght 900, opsz 11 (sturdier hairlines)
        wmax = max(font.getbbox(s)[2] - font.getbbox(s)[0] for s in lines)
        size = int(size * target_w / wmax)
    font = ImageFont.truetype(FONT, size)
    font.set_variation_by_axes([900, 11])
    sw = max(1, int(round(0.012 * size)))                      # slight dilation so hairlines catch grains
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    cap = font.getbbox("H")
    cap_h = cap[3] - cap[1]
    gap = 0.38 * cap_h
    total = 2 * cap_h + gap
    y0 = H / 2 - total / 2
    for k, s in enumerate(lines):
        bb = font.getbbox(s)
        x = W / 2 - (bb[2] - bb[0]) / 2 - bb[0]
        y = y0 + k * (cap_h + gap) - cap[1]
        dr.text((x, y), s, font=font, fill=255, stroke_width=sw, stroke_fill=255)
    return np.asarray(img) > 127, size, (y0 + total) / W


def mask_targets(mask, rng, ref_w=1080):
    """Stratified (jittered-grid) sample of NG points inside the mask, in frame-width units."""
    area = mask.sum()
    h = np.sqrt(area / NG)
    H, W = mask.shape
    xs = np.arange(0, W, h)
    ys = np.arange(0, H, h)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel() + rng.uniform(0, h, gx.size)
    gy = gy.ravel() + rng.uniform(0, h, gy.size)
    ok = (gx < W) & (gy < H)
    gx, gy = gx[ok], gy[ok]
    inside = mask[gy.astype(int), gx.astype(int)]
    pts = np.stack([gx[inside], gy[inside]], 1)
    if len(pts) > NG:
        pts = pts[rng.choice(len(pts), NG, replace=False)]
    elif len(pts) < NG:
        yy, xx = np.nonzero(mask)
        extra = rng.choice(len(xx), NG - len(pts), replace=False)
        pts = np.concatenate([pts, np.stack([xx[extra] + rng.random(extra.size),
                                             yy[extra] + rng.random(extra.size)], 1)])
    return pts / ref_w


def match(src, dst, sectors=12):
    """Reorder dst so grain i (at src[i]) goes to dst[i]: both sets are cut into equal-count angular
    sectors about the plate centre, then each sector pair is solved exactly (min total squared travel).
    Sector boundaries are shared rays, so paths never cross between sectors; full LSA is ~30x slower."""
    def order(p):
        return np.argsort(np.arctan2(p[:, 1] - PCY, p[:, 0] - PCX), kind="stable")
    so, do = order(src), order(dst)
    out = np.empty_like(dst)
    for a, b in zip(np.array_split(so, sectors), np.array_split(do, sectors)):
        _, col = linear_sum_assignment(cdist(src[a], dst[b], "sqeuclidean"))
        out[a] = dst[b][col]
    return out


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        # scatter: uniform over the plate plus a little bleed
        s = rng.uniform(-0.04, 1.04, (NG, 2))
        S = np.stack(to_frame(s[:, 0], s[:, 1]), 1)
        self.S = S
        self.C, self.T = [], []
        self.text_bottom = 0.0
        prev = S
        for k in range(3):
            C = match(prev, nodal_targets(*MODES[k], rng))
            mask, self.font_size, bottom = text_mask(PHRASES[k])
            T = match(C, mask_targets(mask, rng))
            self.C.append(C)
            self.T.append(T)
            if k == 2:
                self.text_bottom = bottom
            prev = T
        # flow control points: leave the nodal line along the field gradient, bow toward the letter
        self.ctrl = []
        for k in range(3):
            C, T = self.C[k], self.T[k]
            u, v = to_plate(C[:, 0], C[:, 1])
            fu, fv = chladni_grad(u, v, *MODES[k])
            g = np.stack([fu, fv], 1)
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            d = T - C
            L = np.linalg.norm(d, axis=1, keepdims=True)
            perp = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(L, 1e-9)
            nrm = np.where(gn > 1e-3, g / np.maximum(gn, 1e-9), perp)
            nrm *= np.sign(np.sum(nrm * d, 1, keepdims=True) + 1e-12)
            self.ctrl.append(C + 0.30 * d + nrm * 0.25 * L)
        # per-grain look and stagger
        self.bright = rng.uniform(0.6, 1.0, NG).astype(np.float32)
        self.phase = rng.uniform(0, 2 * np.pi, NG)
        ang = rng.uniform(0, 2 * np.pi, NG)
        self.jdir = np.stack([np.cos(ang), np.sin(ang)], 1)
        self.jit = rng.uniform(-1, 1, (6, NG)) * 0.012          # small random part of the stagger
        rmax = np.hypot(PLATE / 2, PLATE / 2)
        self.r_snap = [np.hypot(*(self.C[k] - [PCX, PCY]).T) / rmax for k in range(3)]
        self.r_flow = [np.hypot(*(self.T[k] - [PCX, PCY]).T) / rmax for k in range(3)]
        self.prep_s = time.time() - t0

    # ------------------------------------------------------------------- motion
    @staticmethod
    def spring(tau, settle):
        """Damped-spring step response (zeta=0.6), ~97% settled after `settle` seconds."""
        w = 3.73 / (ZETA * settle)
        wd = w * np.sqrt(1 - ZETA ** 2)
        tau = np.maximum(tau, 0.0)
        return 1 - np.exp(-ZETA * w * tau) * (np.cos(wd * tau) + ZETA / np.sqrt(1 - ZETA ** 2) * np.sin(wd * tau))

    def positions(self, t):
        p = self.S.copy()
        prev = self.S
        for k in range(3):
            ts, st, span = SNAP[k]
            s = self.spring(t - ts - span * self.r_snap[k] - self.jit[2 * k] - 0.012, st)[:, None]
            p += (self.C[k] - prev) * s
            tf, sf, spanf = FLOW[k]
            # ripple outward: grains landing near the centre go first
            s = self.spring(t - tf - spanf * self.r_flow[k] - self.jit[2 * k + 1] - 0.012, sf)[:, None]
            A, B, Cc = self.C[k], self.T[k], self.ctrl[k]
            bez = (1 - s) ** 2 * A + 2 * s * (1 - s) * Cc + s ** 2 * B
            p += bez - A
            prev = self.T[k]
        # shimmer on the nodal lines while each figure is showing
        for k in range(3):
            ts = SNAP[k][0]
            tf = FLOW[k][0]
            w = np.clip((t - ts) / 0.08, 0, 1) * np.clip((tf + 0.06 - t) / 0.08, 0, 1)
            if w <= 0:
                continue
            n, m = MODES[k]
            u, v = to_plate(p[:, 0], p[:, 1])
            ph = 2 * np.pi * 3.0 * t
            fa = np.abs(chladni(u + 0.03 * np.cos(ph), v + 0.03 * np.sin(ph), n, m)) / 2.0
            osc = np.sin(2 * np.pi * 11.0 * t + self.phase)
            p += self.jdir * (w * 0.006 * fa * osc)[:, None]
        # the settled text hums with the margin strings (same mode, in phase, smaller amplitude)
        u, _ = to_plate(p[:, 0], p[:, 1])
        p[:, 1] += 0.35 * self.wave(t, np.clip(u, 0, 1))
        return p

    # ------------------------------------------------------------------- drawing
    def frame(self, t, W):
        H = int(round(W * ASPECT))
        sc = W / 360.0
        acc = np.zeros(H * W, np.float32)
        p = self.positions(t) * W
        x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        b = self.bright
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            np.add.at(acc, yi[ok] * W + xi[ok], (b * wt)[ok])
        acc = acc.reshape(H, W)
        if sc > 1.01:
            # keep the grain the same size relative to the frame as in the draft
            acc = cv2.GaussianBlur(acc, (0, 0), 0.34 * sc) * (sc ** 2) * 0.48
        a = np.clip(acc * 1.9, 0, 1)[..., None]
        img = BG + (INK - BG) * a
        img = self.strings(img, t, W, H, sc)
        img = self.credit(img, t, W, H, sc)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    @staticmethod
    def wave(t, xs):
        """Standing wave y = A(t) sin(k pi x) cos(w t) on x in [0,1]: k = harmonic of the current phrase,
        A(t) kicked by the pluck in hum.wav and decaying. Drives the margin strings and the settled text."""
        disp = np.zeros_like(xs)
        for k in range(3):
            lo = PHRASE_T[k]
            hi = PHRASE_T[k + 1] if k < 2 else 99
            win = np.clip((t - lo + 0.05) / 0.1, 0, 1) * np.clip((hi + 0.05 - t) / 0.1, 0, 1)
            if win <= 0:
                continue
            tp = PLUCK_T[k]
            ex = 0.0 if t < tp else np.exp(-(t - tp) / 0.45) * min(1.0, (t - tp) / 0.03)
            amp = 0.0035 + 0.016 * ex
            kk = HARM[k]
            disp += win * amp * np.sin(kk * np.pi * xs) * np.cos(2 * np.pi * 1.7 * kk * (t - tp))
        return disp

    def strings(self, img, t, W, H, sc):
        """Hairline standing waves in the top/bottom margins, excited by the plucks."""
        lay = np.zeros((H, W), np.float32)
        xs = np.linspace(0, 1, 240)
        xl, xr = PCX - PLATE / 2, PCX + PLATE / 2
        disp = self.wave(t, xs)
        gap = (PCY - PLATE / 2) / 2
        for yc, sgn in ((PCY - PLATE / 2 - gap, 1), (PCY + PLATE / 2 + gap, -1)):
            pts = np.stack([(xl + xs * (xr - xl)) * W, (yc + sgn * disp) * W], 1)
            cv2.polylines(lay, [np.round(pts * 16).astype(np.int32)], False, 1.0,
                          max(1, int(round(0.9 * sc))), cv2.LINE_AA, shift=4)
        a = (lay * 0.38)[..., None]
        return img * (1 - a) + ACC * a

    def credit(self, img, t, W, H, sc):
        if t < CREDIT_T:
            return img
        u = np.clip((t - CREDIT_T) / 0.5, 0, 1)
        e = 1 - (1 - u) ** 3
        size = max(8, int(round(0.042 * W)))
        font = ImageFont.truetype(FONT, size)
        font.set_variation_by_axes([500, 24])
        text = "PYTHAGORAS"
        track = (0.20 + 0.25 * (1 - e)) * size              # letters close in as they appear
        widths = [font.getlength(c) for c in text]
        tw = sum(widths) + track * (len(text) - 1)
        dash = 0.07 * W
        total = dash + 0.35 * size + tw
        x0 = W / 2 - total / 2
        ybase = (self.text_bottom + 0.075) * W
        lay = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(lay)
        x = x0 + dash + 0.35 * size
        for i, c in enumerate(text):
            # letters arrive left to right
            ui = np.clip((t - CREDIT_T - 0.04 * i) / 0.35, 0, 1)
            dr.text((x, ybase - 0.08 * size * (1 - ui)), c, font=font, fill=int(255 * ui), anchor="ls")
            x += widths[i] + track
        a = (np.asarray(lay, np.float32) / 255 * 0.9)[..., None]
        img = img * (1 - a) + INK * a
        # the dash is a plucked string: drawn out in the accent colour
        acc = np.zeros((H, W), np.float32)
        yd = ybase - 0.33 * size
        cv2.line(acc, (int(round(x0 * 16)), int(round(yd * 16))),
                 (int(round((x0 + dash * e) * 16)), int(round(yd * 16))), 1.0,
                 max(1, int(round(0.9 * sc))), cv2.LINE_AA, shift=4)
        a = (acc * 0.85)[..., None]
        return img * (1 - a) + ACC * a


def render(scene, W, out):
    H = int(round(W * ASPECT))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "16", "-r", str(FPS), out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(NFR):
        pr.stdin.write(scene.frame(i / FPS, W).tobytes())
    pr.stdin.close()
    pr.wait()
    return time.time() - t0


def mux(video, out):
    hum = os.path.join(OUTDIR, "hum.wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", hum,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "draft"
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    sc = Scene()
    print(f"targets ready in {sc.prep_s:.1f} s (font size {sc.font_size}px @1080)")
    if mode == "still":
        for a in sys.argv[2:]:
            W = 1080 if a.endswith("f") else 360
            t = float(a.rstrip("f"))
            Image.fromarray(sc.frame(t, W)).save(os.path.join(REVIEW, f"C_still_{t:.2f}_{W}.png"))
        return
    if mode == "draft":
        out = os.path.join(OUTDIR, "C_chladni_draft.mp4")
        dt = render(sc, 360, out)
    else:
        out = os.path.join(OUTDIR, "C_chladni.mp4")
        dt = render(sc, 1080, out)
        mux(out, os.path.join(OUTDIR, "C_chladni_hum.mp4"))
    print(f"wrote {out} in {dt:.1f} s (+{sc.prep_s:.1f} s prep)")


if __name__ == "__main__":
    main()
