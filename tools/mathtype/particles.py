#!/usr/bin/env python
"""Math-type test A: particle glyphs (5 s, 4:5).

A fixed cloud of N white bubbles (thin dark outline, varied radii) forms each phrase of
"There is geometry in the humming of the strings." Between landings every particle follows a
closed-form path: linear interpolation start -> end driven by a damped spring u(t), plus a detour
whose amplitude is sin(pi*u): a differential vortex about the view axis, a 3D tumble about a tilted
axis (simple perspective, so the cloud appears to rotate in depth) and a per-particle Lissajous
offset. Starts are staggered up to 0.15 s by position so the change ripples across the word.
Hairline trails are earlier samples of the same closed-form path. Nothing is carried between
frames: ``Scene.frame(t)`` is a pure function of t.

    python tools/mathtype/particles.py --draft            # 360x450 -> local/A_particles_draft.mp4
    python tools/mathtype/particles.py --final            # 1080x1350 -> local/A_particles.mp4
    python tools/mathtype/particles.py --still 1.0 2.3    # PNG stills (draft size) to review/
    python tools/mathtype/particles.py --mux              # + hum.wav -> local/A_particles_hum.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from mograph.font_registry import font_axes, font_path  # noqa: E402  (read-only use)

VDIR = os.path.join(REPO, "11_mathtype", "v001")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")

DW, DH = 1080, 1350                     # design space; every length below is in design px
FPS, NFR = 25, 125
BG = np.array([5, 5, 5], float)
INK = np.array([0xF3, 0xEA, 0xDB], float)
EDGE = (16, 15, 14)                     # bubble outline
ACCENT = (0xE2, 0x58, 0x2A)

FONT = "Big Shoulders Display"
CREDIT_FONT = "Major Mono Display"
PHRASES = [("THERE IS", "GEOMETRY"), ("IN THE", "HUMMING"), ("OF THE", "STRINGS")]
N = 3000
SEED = 7
ZETA = 0.55
STAGGER = 0.15
FOCAL = 1500.0
CX, CY = 540.0, 640.0
TEXT_W = 980.0                          # ink width of each line
TRACK = 0.05                            # extra letter spacing (em)
MAX_CAP = 300.0                         # cap-height ceiling for short lines
TRAIL_DT = 0.012                        # spacing of the trail samples (s)
TRAIL_K = 4                             # previous-time samples per trail
CREDIT_T = 4.2

# transitions: start time, from-state, to-state, spring omega, tumble axis, tumble angle (rad),
# vortex angle (rad), Lissajous amplitude (px), Lissajous freqs, stagger direction (x, y)
TRANSITIONS = [
    dict(ts=-0.10, a=0, b=1, w=22.0, axis=(0.25, 1.0, 0.15), tumble=1.1, vortex=1.6, liss=70.0,
         freq=(3, 2, 5), sdir=(1.0, 0.25)),
    dict(ts=1.32, a=1, b=2, w=15.0, axis=(1.0, 0.35, 0.2), tumble=1.25, vortex=2.2, liss=110.0,
         freq=(3, 2, 4), sdir=(1.0, 0.0)),
    dict(ts=2.92, a=2, b=3, w=15.0, axis=(-0.3, 1.0, -0.25), tumble=-1.25, vortex=-2.2, liss=110.0,
         freq=(2, 3, 5), sdir=(0.7, 0.7)),
]


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


def line_w(f, s: str) -> float:
    """Ink width of s set with TRACK em of extra letter spacing."""
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def phrase_mask(lines) -> tuple[np.ndarray, dict]:
    """Two-line justified stack: each line sized so its ink spans TEXT_W (cap height capped)."""
    probe = load_font(FONT, 200)
    pb = probe.getbbox("H")
    sizes = []
    for s in lines:
        size = 200 * TEXT_W / line_w(probe, s)
        sizes.append(int(min(size, 200 * MAX_CAP / (pb[3] - pb[1]))))
    fonts = [load_font(FONT, z) for z in sizes]
    caps = [f.getbbox("H")[3] - f.getbbox("H")[1] for f in fonts]
    gap = 0.22 * min(caps)
    top = CY - (sum(caps) + gap) / 2
    im = Image.new("L", (DW, DH), 0)
    d = ImageDraw.Draw(im)
    rights, y0 = [], top
    for s, f, cap in zip(lines, fonts, caps):
        hb = f.getbbox("H")
        x = CX - line_w(f, s) / 2 - f.getbbox(s[0])[0]
        rights.append(CX + line_w(f, s) / 2)
        for ch in s:              # set letter by letter with open tracking so bubbles keep the gaps
            d.text((x, y0 - hb[1]), ch, font=f, fill=255)
            x += f.getlength(ch) + TRACK * f.size
        y0 += cap + gap
    return np.array(im), dict(cap=caps[-1], bottom=y0 - gap, right=max(rights))


def stratified(mask: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n points by jittered-grid sampling of the mask interior; also returns distance-to-edge."""
    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    inner = dist >= 2.5
    c = float(np.sqrt(inner.sum() / n))
    while True:
        gx, gy = np.meshgrid(np.arange(0, DW, c), np.arange(0, DH, c))
        x = gx + rng.random(gx.shape) * c
        y = gy + rng.random(gy.shape) * c
        xi = np.clip(x.astype(int), 0, DW - 1)
        yi = np.clip(y.astype(int), 0, DH - 1)
        keep = inner[yi, xi]
        if keep.sum() >= n:
            break
        c *= 0.97
    pts = np.stack([x[keep], y[keep]], 1)
    pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts, dist[pts[:, 1].astype(int), pts[:, 0].astype(int)]


def hilbert_order(xy: np.ndarray, order: int = 11) -> np.ndarray:
    """Indices that sort points along a Hilbert curve over the frame (locality-preserving)."""
    n = 1 << order
    q = np.clip((xy / max(DW, DH) * (n - 1)).astype(np.int64), 0, n - 1)
    x, y = q[:, 0].copy(), q[:, 1].copy()
    d = np.zeros(len(x), np.int64)
    s = n // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        m = ry == 0
        f = m & (rx == 1)
        x = np.where(f, n - 1 - x, x)
        y = np.where(f, n - 1 - y, y)
        x, y = np.where(m, y, x), np.where(m, x, y)
        s //= 2
    return np.argsort(d, kind="stable")


def rodrigues(v: np.ndarray, k: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N,3) about unit axis k (3,) by per-row angle th (N,)."""
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    return v * c + np.cross(k, v) * s + k * (v @ k)[:, None] * (1 - c)


def spring(tau: np.ndarray, w: float) -> np.ndarray:
    """Closed-form underdamped step response: 0 before the start, ~1 after settling."""
    tau = np.maximum(tau, 0.0)
    wd = w * np.sqrt(1 - ZETA * ZETA)
    return 1 - np.exp(-ZETA * w * tau) * (np.cos(wd * tau) + (ZETA * w / wd) * np.sin(wd * tau))


# ---------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.size_q = rng.random(N)
        self.phase = rng.random((N, 6)) * 2 * np.pi
        self.hash = rng.random(N)
        self.liss_amp = 0.35 + 0.65 * rng.random(N) ** 0.7
        base_r = None
        targets, radii = [], []
        self.metrics = []
        # state 0: Fibonacci sphere (spun as a function of t in state0)
        i = np.arange(N) + 0.5
        phi = np.arccos(1 - 2 * i / N)
        th = np.pi * (1 + 5 ** 0.5) * i
        sph = np.stack([np.sin(phi) * np.cos(th), np.cos(phi), np.sin(phi) * np.sin(th)], 1) * 430.0
        self.sphere = sph
        for lines in PHRASES:
            mask, met = phrase_mask(lines)
            pts, dist = stratified(mask, N, rng)
            if base_r is None:
                spacing = np.sqrt((mask > 127).sum() / N)
                base_r = 0.62 * spacing
            r_i = base_r * (0.55 + 0.9 * self.size_q ** 1.6)
            o = hilbert_order(pts)
            p3 = np.zeros((N, 3))
            p3[:, :2] = pts[o]
            targets.append(p3)
            radii.append(np.minimum(r_i, dist[o] + 2.0))
            self.metrics.append(met)
        # sphere particles matched to phrase 1 by Hilbert order of their projection
        so = hilbert_order(sph[:, :2] + [CX, CY])
        self.sphere = sph[so]
        self.base_r = base_r
        self.r_free = base_r * (0.55 + 0.9 * self.size_q ** 1.6)
        self.targets = targets
        self.radii = radii
        self.stagger = []
        for tr in TRANSITIONS:
            ref = targets[tr["b"] - 1][:, :2] if tr["a"] == 0 else targets[tr["a"] - 1][:, :2]
            dvec = np.array(tr["sdir"], float)
            dvec /= np.linalg.norm(dvec)
            p = (ref - [CX, CY]) @ dvec
            p = (p - p.min()) / (np.ptp(p) + 1e-9)
            self.stagger.append(STAGGER * (0.85 * p + 0.15 * self.hash))

    # closed-form state ------------------------------------------------------------------------
    def state0(self, t: float) -> np.ndarray:
        a = 0.9 * t + 0.4
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        p = self.sphere @ R.T
        return p + [CX, CY, 0.0]

    def pos(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """World positions (N,3) and radii (N,) at time t."""
        prev = self.state0(t)
        prev_r = self.r_free
        p = prev.copy()
        r = prev_r.copy()
        us = []
        for k, tr in enumerate(TRANSITIONS):
            u = spring(t - tr["ts"] - self.stagger[k], tr["w"])
            tgt, tr_r = self.targets[tr["b"] - 1], self.radii[tr["b"] - 1]
            src = prev if k == 0 else self.targets[tr["a"] - 1]
            src_r = prev_r if k == 0 else self.radii[tr["a"] - 1]
            p = p + (tgt - src) * u[:, None]
            r = r + (tr_r - src_r) * u
            us.append(u)
        home = p.copy()
        ctr = np.array([CX, CY, 0.0])
        for k, tr in enumerate(TRANSITIONS):
            u = us[k]
            env = np.sin(np.pi * np.clip(u, 0.0, 1.25))
            if not np.any(np.abs(env) > 1e-4):
                continue
            v = p - ctr
            # differential vortex about the view axis: inner particles turn further
            rho = np.hypot(v[:, 0], v[:, 1])
            alpha = tr["vortex"] * env / (1 + rho / 260.0)
            ca, sa = np.cos(alpha), np.sin(alpha)
            v[:, 0], v[:, 1] = ca * v[:, 0] - sa * v[:, 1], sa * v[:, 0] + ca * v[:, 1]
            # 3D tumble about the transition axis, each particle by its own angle
            ax = np.array(tr["axis"], float)
            ax /= np.linalg.norm(ax)
            th = tr["tumble"] * env * (0.85 + 0.3 * self.hash)
            v = rodrigues(v, ax, th)
            # Lissajous detour
            fx, fy, fz = tr["freq"]
            pu = np.pi * u
            amp = (tr["liss"] * self.liss_amp * env)[:, None]
            L = np.stack([np.sin(fx * pu + self.phase[:, 0]),
                          np.sin(fy * pu + self.phase[:, 1]),
                          np.sin(fz * pu + self.phase[:, 2])], 1)
            p = ctr + v + amp * L
        # breathing: tiny smooth 2D noise in t, coherent in space
        bx = np.sin(0.011 * home[:, 0] + 0.017 * home[:, 1] + 1.3 * t + self.phase[:, 3])
        by = np.cos(0.013 * home[:, 0] - 0.010 * home[:, 1] + 1.1 * t + self.phase[:, 4])
        p[:, 0] += 1.6 * bx
        p[:, 1] += 1.6 * by
        r = np.maximum(r, 0.5) * (1 + 0.07 * np.sin(2 * np.pi * 0.7 * t + self.phase[:, 5]))
        return p, r

    def project(self, t: float):
        p, r = self.pos(t)
        z = np.minimum(p[:, 2], 0.75 * FOCAL)
        s = FOCAL / (FOCAL - z)
        xy = np.stack([CX + (p[:, 0] - CX) * s, CY + (p[:, 1] - CY) * s], 1)
        return xy, r * s, z

    # drawing ------------------------------------------------------------------------------------
    def frame(self, t: float, W: int, H: int) -> np.ndarray:
        ss = max(1, int(round(2 * DW / W)))       # supersample: x3 for the draft, x2 for the final
        osc = W / DW
        sc = osc * ss
        img = np.empty((H * ss, W * ss, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        xy, r, z = self.project(t)
        S = 16.0  # cv2 shift=4 subpixel
        # hairline trails: previous-time samples of the same closed-form path
        hist = [xy] + [self.project(t - j * TRAIL_DT)[0] for j in range(1, TRAIL_K + 1)]
        speed = np.linalg.norm(xy - hist[-1], axis=1) / (TRAIL_K * TRAIL_DT)
        lev = np.clip((speed - 900.0) / 5000.0, 0, 1) * (self.hash < 0.45)
        trail = np.stack(hist, 1) * sc * S
        for b in range(1, 6):
            sel = (lev > (b - 1) / 5) & (lev <= b / 5) & (lev > 0)
            if not sel.any():
                continue
            col = tuple(float(c) for c in BG + (INK - BG) * min(0.6, (0.05 + 0.05 * b) * 0.9 * ss))
            cv2.polylines(img, list(np.round(trail[sel]).astype(np.int32)), False, col, 1, cv2.LINE_AA, 4)
        # bubbles, far to near
        bright = np.clip(1.0 + z / 900.0, 0.32, 1.0)
        fill = BG + (INK - BG) * bright[:, None]
        lw = max(1.3 * osc, 0.45) * ss
        cen = np.round(xy * sc * S).astype(np.int64)
        R = np.maximum(r * sc, 0.6 * ss)
        Ro = np.round(R * S).astype(np.int64)
        Ri = np.round(np.maximum(R - lw, 0.3) * S).astype(np.int64)
        for i in np.argsort(z, kind="stable"):
            c = (int(cen[i, 0]), int(cen[i, 1]))
            cv2.circle(img, c, int(Ro[i]), EDGE, -1, cv2.LINE_AA, 4)
            cv2.circle(img, c, int(Ri[i]), tuple(float(v) for v in fill[i]), -1, cv2.LINE_AA, 4)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if t >= CREDIT_T:
            img = self.credit(img, t, osc)
        return img

    def credit(self, img: np.ndarray, t: float, sc: float) -> np.ndarray:
        """'- PYTHAGORAS' sets in with spring-driven tracking that closes up as it appears."""
        u = float(spring(np.array([t - CREDIT_T]), 9.0)[0])
        a = float(np.clip((t - CREDIT_T) / 0.45, 0, 1))
        a = a * a * (3 - 2 * a)
        met = self.metrics[2]
        size = max(int(round(40 * sc)), 9)
        f = load_font(CREDIT_FONT, size)
        text = "— PYTHAGORAS"
        track = (0.16 + 0.9 * (1 - u)) * size
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + track * (len(text) - 1)
        x = met["right"] * sc - total
        y = (met["bottom"] + 0.55 * met["cap"]) * sc
        layer = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i, ch in enumerate(text):
            col = ACCENT if i == 0 else tuple(int(v) for v in INK)
            d.text((x, y), ch, font=f, fill=col + (int(255 * a),))
            x += widths[i] + track
        out = Image.fromarray(img).convert("RGBA")
        out.alpha_composite(layer)
        return np.array(out.convert("RGB"))


# ---------------------------------------------------------------------------------------------- output
def render(path: str, W: int, H: int) -> float:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc = Scene()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(NFR):
        proc.stdin.write(sc.frame(n / FPS, W, H).tobytes())
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"{path}  {W}x{H}  {NFR} frames  {dt:.1f}s")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--still", type=float, nargs="*")
    a = ap.parse_args()
    if a.draft:
        render(os.path.join(LOCAL, "A_particles_draft.mp4"), 360, 450)
    if a.final:
        render(os.path.join(LOCAL, "A_particles.mp4"), 1080, 1350)
    if a.still:
        os.makedirs(REVIEW, exist_ok=True)
        sc = Scene()
        for t in a.still:
            Image.fromarray(sc.frame(t, 360, 450)).save(os.path.join(REVIEW, f"A_still_{t:.2f}.png"))
    if a.mux:
        wav = os.path.join(LOCAL, "hum.wav")
        if not os.path.exists(wav):
            sys.exit("hum.wav not found")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(LOCAL, "A_particles.mp4"),
                        "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        os.path.join(LOCAL, "A_particles_hum.mp4")], check=True)


if __name__ == "__main__":
    main()
