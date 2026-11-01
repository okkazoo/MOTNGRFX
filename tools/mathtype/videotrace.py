#!/usr/bin/env python
"""E6 (11_mathtype v012): video-traced motion graphics; the sea's own motion writes the sentence.

"The sea keeps rewriting the same wave, and every shape it makes is a letter it has already forgotten."

Footage: LTX clips slowed to 10.4 s with motion-compensated interpolation (ffmpeg minterpolate, as v005),
scaled to a panorama the camera pans across (S1_i2v_0 shore -> 1576x900, pan right; B5_i2v_0 jellyfish ->
792x1386, pan down). Arc: graded greyscale footage -> the graphic layer traces it -> footage dissolves ->
the pattern resolves into each phrase -> releases back into the flow as the camera travels on.

(a) flow: dense optical flow u_n(x) (cv2 DIS, cached .npy) advects N particles, integrated once on the
    device (RK2 via grid_sample):  x_{n+1} = x_n + f(t) u_n(x_n + u_n/2) + sum_j k_j(t) (g_ij - x_n),
    k_j = k_max s^2 (growing attraction to glyph targets g_ij, s = formation progress), f = 1 - (1-rho) h.
    At hold the offset x - g = f u / k is the local current: the letters keep the sea's motion. Marks =
    trails (past positions), splatted (dense samples + bilinear scatter-add) and tone-mapped.
(b) topo: H_v = L * grade(G_sigma * I_t) (luminance levels); glyph field H_g = -sdf/Delta + w t (echo rings
    running outward); H = (1 - a) H_v + a H_g with a = a_j(t) exp(-max(sdf, 0) / R) (letters are islands on
    a sea chart). Lines where |H - round H| / |grad H| < w px (fwidth-style AA); letters filled.
Everything per frame runs in torch (cpu for previews, cuda on the box); frame = f(t) from cached data.

    python tools/mathtype/videotrace.py --prep                         # sides + flow cache (CPU, once)
    python tools/mathtype/videotrace.py --variant a --preview          # 360x450 -> v012/local/E6a_flow_gNN_preview.mp4
    python tools/mathtype/videotrace.py --variant a --final --device cuda --gen N
    python tools/mathtype/videotrace.py --variant a --still 2.0 3.2
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from mograph.font_registry import font_axes, font_path  # noqa: E402  (read-only use)
import xp  # noqa: E402

VDIR = os.path.join(REPO, "11_mathtype", "v012")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")
CACHE = os.path.join(LOCAL, "cache")
CLIPS = os.path.join(REPO, "04_routes", "v001", "local", "clips")

FPS = 25
DUR = 10.0
NFR = int(round(FPS * DUR))
DW, DH = 720, 900                                   # design space == final frame
SEED = 6
MCI = "minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
FLOW_DS = 3                                         # flow grid = panorama / 3

# panorama size (design px) and camera keys (top-left of the 720x900 window, world px); between keys a
# smootherstep move, equal keys = hold (the camera is still while a phrase reads).
SOURCES = {
    "wave": dict(clip="S1_i2v_0", size=(1576, 900), axis=0,
                 cam=[(0.0, 40.0), (2.75, 330.0), (4.15, 330.0), (5.55, 600.0), (6.95, 600.0), (8.25, 856.0),
                      (10.0, 856.0)]),
    "jelly": dict(clip="B5_i2v_0", size=(792, 1386), axis=1,
                  cam=[(0.0, 0.0), (2.75, 150.0), (4.15, 150.0), (5.55, 330.0), (6.95, 330.0), (8.25, 486.0),
                       (10.0, 486.0)]),
}

# phrases: lines (text, face, width fraction of TEXT_W, max cap in design px); F formation start, L landed,
# D release. Font mix: tall Le Murmure / Facade against one puffy or fat key word per phrase.
TEXT_W = 620.0
TRACK = 0.03
PHRASES = [
    dict(lines=[("THE SEA KEEPS", "Le Murmure", 0.64, 120), ("REWRITING", "Le Murmure", 1.0, 150),
                ("THE SAME", "Le Murmure", 0.40, 110), ("WAVE", "Pilowlava", 1.0, 230)], F=1.45, L=2.75, D=4.15),
    dict(lines=[("AND EVERY", "Facade", 0.62, 120), ("SHAPE IT MAKES", "Facade", 1.0, 150),
                ("IS A", "Facade", 0.36, 120), ("LETTER", "Blackout", 1.0, 230)], F=4.25, L=5.55, D=6.95),
    dict(lines=[("IT HAS", "Le Murmure", 0.36, 120), ("ALREADY", "Le Murmure", 0.72, 150),
                ("FORGOTTEN", "Shrikhand", 1.0, 200)], F=7.05, L=8.25, D=9.55),
]
STAG = 0.30                                         # left-to-right wave across a phrase (s)
FOOT_OUT = (0.9, 2.5)                               # footage dissolve
MARK_IN = (0.15, 1.2)                               # graphic layer emerges

VARIANTS = {
    "a": dict(stem="E6a_flow", test="E6a", tech="flow", src="wave", N=36000, gain=2.0, kmax=0.22, rho=0.35,
              K=16, vmax=16.0, free_read=0.18, expo=1.25, dot=1.0, dot_k=26.0, colour=False),
    "b": dict(stem="E6b_topo", test="E6b", tech="topo", src="jelly", levels=10, sigma=9.0, delta=9.0, omega=1.6,
              R=110.0, res=0.6, lw=0.55, colour=False),
    "c": dict(stem="E6c_topocol", test="E6c", tech="topo", src="jelly", levels=10, sigma=9.0, delta=9.0, omega=1.6,
              R=110.0, res=0.6, lw=0.55, colour=True, band_out=6.6),
}

# palette (E6 sea), colour generation only
NAVY = np.array([0x0B, 0x1F, 0x33], np.float32) / 255
FOAM = np.array([0xCF, 0xE3, 0xE0], np.float32) / 255
SAND = np.array([0xE8, 0xD3, 0xA9], np.float32) / 255
CORAL = np.array([0xFF, 0x7A, 0x59], np.float32) / 255


# ---------------------------------------------------------------------------------------------- utils
def smoothstep(a, b, x):
    u = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def cam_at(src, t):
    """Camera offset (world px, top-left) at time t: piecewise smootherstep through the keys."""
    keys = SOURCES[src]["cam"]
    v = keys[-1][1]
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            v = a if t <= t0 else a + (b - a) * float(smootherstep((t - t0) / (t1 - t0)))
            break
    return (v, 0.0) if SOURCES[src]["axis"] == 0 else (0.0, v)


def read_amount(t):
    """1 while some phrase reads (L..D), 0 between, smooth edges."""
    r = 0.0
    for p in PHRASES:
        r = max(r, float(smoothstep(p["L"] - 0.45, p["L"], t) * (1 - smoothstep(p["D"], p["D"] + 0.5, t))))
    return r


def load_font(name, size):
    f = ImageFont.truetype(font_path(name), size)
    ax = dict(font_axes(name))
    if ax:
        tags = {"weight": "wght", "width": "wdth", "optical size": "opsz", "year": "YEAR", "contrast": "CTST"}
        try:
            vals = []
            for a in f.get_variation_axes():
                nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
                vals.append(ax.get(tags.get(nm.lower(), nm), a["default"]))
            f.set_variation_by_axes(vals)
        except Exception:
            pass
    return f


def line_w(f, s):
    adv = sum(f.getlength(ch) for ch in s[:-1]) + TRACK * f.size * (len(s) - 1)
    return adv + f.getbbox(s[-1])[2] - f.getbbox(s[0])[0]


def phrase_mask(lines, ss=2):
    """Mixed-face stack centred in the design frame. Returns (mask 0..1 at design res, sdf design px,
    sdf of the key (last) line)."""
    sizes = []
    for s, fn, wf, mcap in lines:
        probe = load_font(fn, 200)
        pb = probe.getbbox("H")
        sizes.append(min(200 * TEXT_W * wf / line_w(probe, s), 200 * mcap / (pb[3] - pb[1])))
    fonts = [load_font(fn, int(z * ss)) for z, (_, fn, _, _) in zip(sizes, lines)]
    caps = [(f.getbbox("H")[3] - f.getbbox("H")[1]) / ss for f in fonts]
    gap = 0.24 * min(caps)
    y0 = DH / 2 - (sum(caps) + gap * (len(lines) - 1)) / 2
    full = Image.new("L", (DW * ss, DH * ss), 0)
    key = Image.new("L", (DW * ss, DH * ss), 0)
    for li, ((s, _, _, _), f, cap) in enumerate(zip(lines, fonts, caps)):
        hb = f.getbbox("H")
        x = DW / 2 * ss - line_w(f, s) / 2 - f.getbbox(s[0])[0]
        for im in ((full, key) if li == len(lines) - 1 else (full,)):
            d = ImageDraw.Draw(im)
            xx = x
            for ch in s:
                d.text((xx, y0 * ss - hb[1]), ch, font=f, fill=255)
                xx += f.getlength(ch) + TRACK * f.size
        y0 += cap + gap
    out = []
    for im in (full, key):
        a = np.array(im)
        m = (a > 127).astype(np.uint8)
        sdf = (cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) - cv2.distanceTransform(m, cv2.DIST_L2, 5)) / ss
        out.append((cv2.resize(a, (DW, DH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255,
                    cv2.resize(sdf.astype(np.float32), (DW, DH), interpolation=cv2.INTER_AREA)))
    return out[0][0], out[0][1], out[1][1]


def stratified(mask, n, rng, inset=0.8):
    """~n jittered-grid points inside the mask (design px), exactly n returned."""
    m = mask > 0.5
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    inner = dist >= inset
    c = float(np.sqrt(inner.sum() / n))
    while True:
        gx, gy = np.meshgrid(np.arange(0, DW, c), np.arange(0, DH, c))
        x = gx + rng.random(gx.shape) * c
        y = gy + rng.random(gy.shape) * c
        keep = inner[np.clip(y.astype(int), 0, DH - 1), np.clip(x.astype(int), 0, DW - 1)]
        if keep.sum() >= n:
            break
        c *= 0.97
    pts = np.stack([x[keep], y[keep]], 1)
    return pts[rng.choice(len(pts), n, replace=False)]


def hilbert_rank(xy, order=11, box=False):
    """Order of points along a Hilbert curve (locality-preserving rank); box=True normalises each set to its
    own bounding box first, otherwise a fixed world square (co-located sets match locally)."""
    if box:
        lo, hi = xy.min(0), xy.max(0)
        span = max(float((hi - lo).max()), 1e-6)
    else:
        lo, span = np.array([-200.0, -200.0]), 2000.0
    n = 1 << order
    q = np.clip(((xy - lo) / span * (n - 1)).astype(np.int64), 0, n - 1)
    x, y = q[:, 0].copy(), q[:, 1].copy()
    d = np.zeros(len(x), np.int64)
    s = n // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        mm = ry == 0
        f = mm & (rx == 1)
        x = np.where(f, n - 1 - x, x)
        y = np.where(f, n - 1 - y, y)
        x, y = np.where(mm, y, x), np.where(mm, x, y)
        s //= 2
    return np.argsort(d, kind="stable")


def lsa_match(P, G, strips=6):
    """Near-optimal assignment (squared distance) of particles P to glyph targets G: both sets cut into equal-
    count vertical strips by x (optimal in 1-D), then scipy's exact LSA (LAPJV) inside each strip. Full 9000^2 LSA
    measured median 37 / p99 67 px at ~70 s; strips=6 gives 39 / 101 px at ~8 s. Cached by a hash of both sets.
    Returns (rows into P, rows into G)."""
    import hashlib
    from scipy.optimize import linear_sum_assignment
    h = hashlib.sha1(np.ascontiguousarray(P, np.float32).tobytes() + np.ascontiguousarray(G, np.float32).tobytes())
    fp = os.path.join(CACHE, f"lsa{strips}_{h.hexdigest()[:16]}.npy")
    if os.path.exists(fp):
        r = np.load(fp)
        return r[0], r[1]
    t0 = time.perf_counter()
    ia, ib = np.argsort(P[:, 0], kind="stable"), np.argsort(G[:, 0], kind="stable")
    rr, cc = [], []
    for ca, cb in zip(np.array_split(ia, strips), np.array_split(ib, strips)):   # 1-D OT in x, then exact LSA
        C = ((P[ca][:, None, :] - G[cb][None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        rr.append(ca[r])
        cc.append(cb[c])
    r, c = np.concatenate(rr), np.concatenate(cc)
    np.save(fp, np.stack([r, c]))
    print(f"lsa {len(P)}x{len(G)}: {time.perf_counter() - t0:.1f}s (cached {os.path.basename(fp)})")
    return r, c


# ---------------------------------------------------------------------------------------------- prep (CPU, once)
def side_path(src):
    return os.path.join(CACHE, f"{src}_side.mp4")


def flow_path(src):
    return os.path.join(CACHE, f"{src}_flow_ds{FLOW_DS}.npy")


def prep(src):
    """Slowed panorama side (minterpolate) + DIS optical flow cache. One-off CPU setup."""
    os.makedirs(CACHE, exist_ok=True)
    cfg = SOURCES[src]
    W, H = cfg["size"]
    sp = side_path(src)
    if not os.path.exists(sp):
        clip = os.path.join(CLIPS, cfg["clip"] + ".mp4")
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    clip], capture_output=True, text=True).stdout)
        slow = (DUR + 0.4) / dur
        vf = f"setpts=PTS*{slow:.5f},scale={W}:{H}:flags=lanczos,{MCI}"
        t0 = time.perf_counter()
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", clip, "-vf", vf, "-t", f"{DUR + 0.2:g}",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", sp], check=True)
        print(f"side {src}: {cfg['clip']} slowed x{slow:.2f} -> {sp} ({time.perf_counter() - t0:.1f}s)")
    fp = flow_path(src)
    if not os.path.exists(fp):
        t0 = time.perf_counter()
        g = read_gray(sp, W // FLOW_DS, H // FLOW_DS)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        fl = np.zeros((NFR - 1, H // FLOW_DS, W // FLOW_DS, 2), np.float16)
        for n in range(NFR - 1):
            fl[n] = (dis.calc(g[n], g[n + 1], None) * FLOW_DS).astype(np.float16)   # design px / frame
        np.save(fp, fl)
        mag = np.linalg.norm(fl.astype(np.float32), axis=-1)
        print(f"flow {src}: {fl.shape} median |u| {np.median(mag):.2f} p95 {np.percentile(mag, 95):.2f} px/frame "
              f"({time.perf_counter() - t0:.1f}s)")


def read_gray(path, W, H, n=NFR):
    """n luma frames resized to (W, H): the decoded Y plane itself (H.264 decoding is bit-exact; ffmpeg's
    gray conversion and cv2's BGR->grey both differed between machines), so every machine renders from identical
    footage."""
    sw, sh = (int(v) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         path], capture_output=True, text=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-frames:v", str(n), "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout   # no conversion
    fsz = sw * sh * 3 // 2
    fr = np.frombuffer(raw, np.uint8).reshape(-1, fsz)[:, :sw * sh].reshape(-1, sh, sw)       # Y plane
    out = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        f = fr[min(i, len(fr) - 1)]
        out[i] = f if (sw, sh) == (W, H) else cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    return out


# ---------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, cfg, be, W, H):
        T = xp.torch()
        self.T, self.be, self.dev, self.cfg = T, be, be.dev, cfg
        self.W, self.H, self.s = W, H, W / DW
        src = cfg["src"]
        self.src = src
        PW, PH = SOURCES[src]["size"]
        self.PW, self.PH = PW, PH
        rw, rh = int(round(PW * self.s)), int(round(PH * self.s))
        g = read_gray(side_path(src), rw, rh)
        lo, hi = np.percentile(g[::10], [2.0, 99.7])
        self.lo, self.hi = float(lo), float(hi)
        self.foot = T.as_tensor(g, device=self.dev)                     # (NFR, rh, rw) uint8
        self.rw, self.rh = rw, rh
        ys, xs = T.meshgrid(T.arange(H, device=self.dev, dtype=T.float32) + 0.5,
                            T.arange(W, device=self.dev, dtype=T.float32) + 0.5, indexing="ij")
        self.px, self.py = xs, ys                                       # render pixel centres
        rng = np.random.default_rng(SEED)
        self.ph = []
        for p in PHRASES:
            mask, sdf, sdf_key = phrase_mask(p["lines"])
            self.ph.append(dict(mask=mask, sdf=sdf, sdf_key=sdf_key, cam=cam_at(src, p["L"]), **p))
        if cfg["tech"] == "flow":
            self.integrate(rng)
        else:
            for p in self.ph:
                r = cv2.resize(p["sdf"], (W, H), interpolation=cv2.INTER_LINEAR) if W != DW else p["sdf"]
                p["sdf_t"] = T.as_tensor(r, device=self.dev)
                rk = cv2.resize(p["sdf_key"], (W, H), interpolation=cv2.INTER_LINEAR) if W != DW else p["sdf_key"]
                p["sdfk_t"] = T.as_tensor(rk, device=self.dev)
                ys_, xs_ = np.nonzero(p["sdf_key"] < 0)
                p["kbox"] = (float(xs_.min()), float(ys_.min()), float(xs_.max()), float(ys_.max()))   # design px

    # ------------------------------------------------------------------ footage
    def footage(self, n, cam):
        """Graded greyscale window of footage frame n at camera cam (world px) -> (H, W) float 0..1."""
        T = self.T
        f = self.foot[min(n, NFR - 1)].float()[None, None]
        cx, cy = cam[0] * self.s, cam[1] * self.s
        gx = (self.px + cx) / self.rw * 2 - 1
        gy = (self.py + cy) / self.rh * 2 - 1
        grid = T.stack([gx, gy], -1)[None]
        v = T.nn.functional.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
        u = T.clamp((v - self.lo) / (self.hi - self.lo), 0, 1)
        return u * u * (3 - 2 * u)

    # ------------------------------------------------------------------ (a) flow particles
    def integrate(self, rng):
        T, dev, cfg = self.T, self.dev, self.cfg
        PW, PH = self.PW, self.PH
        fl = np.load(flow_path(self.src))                              # (NFR-1, h, w, 2) f16, design px
        flow = T.as_tensor(fl, device=dev).permute(0, 3, 1, 2).contiguous()   # (NFR-1, 2, h, w) f16
        fh, fw = fl.shape[1], fl.shape[2]
        N = cfg["N"]
        x = T.as_tensor(rng.random((N, 2)) * [PW, PH], dtype=T.float32, device=dev)
        bid = T.arange(N, device=dev, dtype=T.int32)
        next_id = N
        life = T.as_tensor(rng.uniform(3.0, 6.5, N) * FPS, dtype=T.float32, device=dev)
        age = T.as_tensor(rng.uniform(0, 1, N), dtype=T.float32, device=dev) * life
        traj = T.empty((NFR, N, 2), dtype=T.float32, device=dev)
        births = T.empty((NFR, N), dtype=T.int32, device=dev)
        J = len(self.ph)
        tgt = T.zeros((J, N, 2), dtype=T.float32, device=dev)
        fS = T.full((J, N), 1e9, dtype=T.float32, device=dev)           # per-particle formation start
        tL = T.full((J, N), 2e9, dtype=T.float32, device=dev)
        tR = T.full((J, N), 3e9, dtype=T.float32, device=dev)           # release
        member = T.zeros((J, N), dtype=T.bool, device=dev)
        bright = T.as_tensor(rng.uniform(0.45, 1.0, N), dtype=T.float32, device=dev)
        n_assign = [int(round((p["F"] - STAG) * FPS)) for p in self.ph]
        gain, kmax, rho = cfg["gain"], cfg["kmax"], cfg["rho"]

        def sample(fr, p):
            gx = p[:, 0] / PW * 2 - 1
            gy = p[:, 1] / PH * 2 - 1
            g = T.stack([gx, gy], -1)[None, None]
            v = T.nn.functional.grid_sample(fr[None], g, mode="bilinear", padding_mode="border",
                                            align_corners=False)[0, :, 0]
            return v.t()

        for n in range(NFR):
            t = n / FPS
            for j, na in enumerate(n_assign):
                if n == na:
                    self._assign(j, x, member, tgt, fS, tL, tR, rng)
            traj[n] = x
            births[n] = bid
            if n == NFR - 1:
                break
            fr = flow[n].float()
            v = sample(fr, x) * gain
            v = sample(fr, x + 0.5 * v) * gain
            k = T.zeros(N, device=dev)
            hold = T.zeros(N, device=dev)
            pull = T.zeros((N, 2), device=dev)
            for j in range(J):
                s = T.clamp((t - fS[j]) / (tL[j] - fS[j]), 0, 1)
                rel = T.clamp((t - tR[j]) / 0.35, 0, 1)
                rel = rel * rel * (3 - 2 * rel)
                kj = kmax * s * s * (1 - rel) * member[j]
                k = k + kj
                hold = hold + s * (1 - rel) * member[j]
                pull = pull + kj[:, None] * (tgt[j] - x)
            hold = T.clamp(hold, 0, 1)
            pn = T.linalg.vector_norm(pull, dim=1, keepdim=True)
            pull = pull * T.clamp(cfg["vmax"] / (pn + 1e-6), max=1.0)        # flight speed cap
            x = x + (1 - (1 - rho) * hold)[:, None] * v + pull
            age = age + 1
            free = hold < 1e-4
            oob = (x[:, 0] < -30) | (x[:, 0] > PW + 30) | (x[:, 1] < -30) | (x[:, 1] > PH + 30)
            dead = free & (oob | (age > life))
            nd = int(dead.sum())
            if nd:
                r2 = np.random.default_rng((SEED, n))
                x[dead] = T.as_tensor(r2.random((nd, 2)) * [PW, PH], dtype=T.float32, device=dev)
                bid[dead] = T.arange(next_id, next_id + nd, device=dev, dtype=T.int32)
                age[dead] = 0
                next_id += nd
        self.traj, self.births, self.bright = traj, births, bright
        self.member, self.fS, self.tL, self.tR = member, fS, tL, tR
        del flow

    def _assign(self, j, x, member, tgt, fS, tL, tR, rng):
        """Targets for phrase j: the previous phrase's particles first, then the free particles nearest the
        phrase window; matched by Hilbert rank (each set normalised to its own box). CPU, once per phrase."""
        T = self.T
        p = self.ph[j]
        cam = np.array(p["cam"])
        pts = stratified(p["mask"], self.cfg.get("M", 9000), rng) + cam
        M = len(pts)
        xn = x.cpu().numpy()
        cx, cy = cam + [DW / 2, DH / 2]
        d = np.hypot(np.maximum(np.abs(xn[:, 0] - cx) - DW / 2, 0), np.maximum(np.abs(xn[:, 1] - cy) - DH / 2, 0))
        sx = np.clip((xn[:, 0] - cam[0]).astype(int), 0, DW - 1)
        sy = np.clip((xn[:, 1] - cam[1]).astype(int), 0, DH - 1)
        d = d + np.maximum(p["sdf"][sy, sx], 0)                         # nearest the glyphs condense first
        if j > 0:
            d = np.where(member[j - 1].cpu().numpy(), d - self.cfg.get("carry", 120.0), d)
        sel = np.argsort(d, kind="stable")[:M]
        if self.cfg.get("match", "lsa") == "lsa":
            a, b = lsa_match(xn[sel], pts)                               # Hungarian, cached (CPU setup)
            a = sel[a]
        else:
            a = sel[hilbert_rank(xn[sel], box=True)]
            b = hilbert_rank(pts, box=True)
        gx = np.zeros((len(xn), 2), np.float32)
        gx[a] = pts[b]
        xr = (gx[:, 0] - cam[0]) / DW
        f0 = p["F"] + STAG * (np.clip(xr, 0, 1) - 1)
        if j > 0:
            prev = tR[j - 1].cpu().numpy() + 0.15
            f0 = np.where(member[j - 1].cpu().numpy(), np.maximum(f0, prev), f0)
        f0 = np.minimum(f0, p["L"] - 0.7)
        rel = p["D"] + STAG * np.clip(xr, 0, 1)
        m = np.zeros(len(xn), bool)
        m[a] = True
        member[j] = T.as_tensor(m, device=self.dev)
        tgt[j] = T.as_tensor(gx, device=self.dev)
        fS[j] = T.as_tensor(np.where(m, f0, 1e9), dtype=T.float32, device=self.dev)
        tL[j] = T.as_tensor(np.where(m, p["L"], 2e9), dtype=T.float32, device=self.dev)
        tR[j] = T.as_tensor(np.where(m, rel, 3e9), dtype=T.float32, device=self.dev)

    def splat(self, A, B, wa, wb, dots, wd, step=0.7):
        """Bilinear splat of segments A->B (render px, weights wa->wb per unit length) + dots; returns (H, W)."""
        T, dev = self.T, self.dev
        W, H = self.W, self.H
        D = B - A
        ln = T.linalg.vector_norm(D, dim=1)
        ns = T.clamp(T.ceil(ln / step), 1, 400).long()
        rep = T.repeat_interleave(ns)
        start = (T.cumsum(ns, 0) - ns)[rep]
        f = ((T.arange(rep.numel(), device=dev) - start).float() + 0.5) / ns[rep].float()
        x = T.cat([A[rep, 0] + D[rep, 0] * f, dots[:, 0]])
        y = T.cat([A[rep, 1] + D[rep, 1] * f, dots[:, 1]])
        blurw = T.clamp(ln / (3.0 * self.s), min=1.0) ** -1.6               # motion-blur energy: fast = fainter
        w = T.cat([(wa[rep] + (wb - wa)[rep] * f) * (ln / ns.float() * blurw)[rep], wd])
        x0, y0 = T.floor(x - 0.5), T.floor(y - 0.5)
        fx, fy = x - 0.5 - x0, y - 0.5 - y0
        x0, y0 = x0.long(), y0.long()
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1)
        x0, y0, w, fx, fy = x0[ok], y0[ok], w[ok], fx[ok], fy[ok]
        gi = y0 * W + x0
        idx = T.cat([gi, gi + 1, gi + W, gi + W + 1])
        val = T.cat([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
        buf = T.zeros(W * H, dtype=T.float32, device=dev)
        buf.index_put_((idx,), val, accumulate=True)
        return buf.view(H, W)

    def marks_flow(self, n, t, cam):
        T, cfg = self.T, self.cfg
        K = cfg["K"]
        ks = [max(n - k, 0) for k in range(K + 1)]
        P = (self.traj[ks] - T.as_tensor(cam, dtype=T.float32, device=self.dev)) * self.s   # (K+1, N, 2)
        Bi = self.births[ks]
        valid = (Bi[1:] == Bi[:1])                                       # same life as the head
        hold = T.zeros_like(self.bright)
        for j in range(len(self.ph)):
            s = T.clamp((t - self.fS[j]) / (self.tL[j] - self.fS[j]), 0, 1)
            rel = T.clamp((t - self.tR[j]) / 0.6, 0, 1)
            hold = hold + s * (1 - rel) * self.member[j]
        hold = T.clamp(hold, 0, 1)
        free_a = 1 - (1 - cfg["free_read"]) * read_amount(t)
        hb = hold * hold
        alpha = (self.bright * free_a * 0.8) * (1 - hb) + hb
        taper = T.linspace(1.0, 0.0, K + 1, device=self.dev)
        A = P[:-1].reshape(-1, 2)
        Bp = P[1:].reshape(-1, 2)
        wa = (taper[:-1, None] * alpha[None]).reshape(-1)
        wb = (taper[1:, None] * alpha[None]).reshape(-1)
        keep = valid.reshape(-1) & (wa > 0.01)
        lo = T.minimum(A, Bp)
        hi = T.maximum(A, Bp)
        keep &= (hi[:, 0] > -2) & (hi[:, 1] > -2) & (lo[:, 0] < self.W + 2) & (lo[:, 1] < self.H + 2)
        keep &= T.linalg.vector_norm(Bp - A, dim=1) < 120 * self.s
        lw = 0.55 * self.s ** 0.5
        e2 = A[:0]
        acc = self.splat(A[keep], Bp[keep], wa[keep] * lw, wb[keep] * lw, e2, wa[:0])
        acc = xp.gaussian_blur(acc, 0.45 * self.s ** 0.5 + 0.1)
        # heads: weight ~ s^2 keeps the stipple density (dots per px) the same at any resolution; the dot layer is
        # blurred ~1 px at the final so landed letters read as a fine grain, not loose points
        dw = alpha * (0.05 + cfg["dot"] * hb) * (cfg["dot_k"] * self.s * self.s)
        dots = self.splat(e2, e2, wa[:0], wa[:0], P[0], dw)
        acc = acc + xp.gaussian_blur(dots, 1.1 * self.s)
        return 1 - T.exp(-acc * cfg["expo"])

    # ------------------------------------------------------------------ (b) topographic
    def marks_topo(self, n, t, cam, foot):
        T, cfg = self.T, self.cfg
        s = self.s
        sig = cfg["sigma"] * s
        g = xp.gaussian_blur(foot, sig)
        L = cfg["levels"]
        Hv = L * g + 0.5                                                 # flat black is not a level
        big = xp.gaussian_blur(g, 6 * sig)
        Hn = Hv.clone()
        a_tot = T.zeros_like(g)
        fill = T.zeros_like(g)
        for p in self.ph:
            aj = float(smoothstep(p["F"], p["L"], t) * (1 - smoothstep(p["D"], p["D"] + 0.6, t)))
            if aj <= 0:
                continue
            dx = (cam[0] - p["cam"][0]) * s
            dy = (cam[1] - p["cam"][1]) * s
            sd = self._shift(p["sdf_t"], dx, dy)                         # design px
            ring = T.exp(-T.clamp(sd, min=0) / cfg["R"])
            # left-to-right arrival / release wave
            xr = self.px / self.W
            ax = T.clamp((t - (p["F"] + STAG * (xr - 1))) / (p["L"] - p["F"]), 0, 1)
            ax = ax * ax * (3 - 2 * ax)
            rx = T.clamp((t - (p["D"] + STAG * xr)) / 0.6, 0, 1)
            ax = ax * (1 - rx * rx * (3 - 2 * rx))
            a = ax * ring
            Hg = -sd / cfg["delta"] + cfg["omega"] * (t - p["L"])
            Hn = Hn + a * (Hg - Hn) + a * cfg["res"] * L * (g - big)
            a_tot = T.maximum(a_tot, a)
            # letters grow out of their skeleton on arrival and erode away on release (forgetting)
            grow = T.clamp((t - (p["F"] + STAG * (xr - 1))) / (p["L"] - p["F"]), 0, 1)
            ero = 18.0 * (1 - grow ** 1.5) + 18.0 * rx
            fill = T.maximum(fill, T.clamp(0.5 - (sd + ero) * s, 0, 1) * T.clamp((grow - 0.25) * 4, 0, 1))
        gy, gx = T.gradient(Hn)
        gm = T.sqrt(gx * gx + gy * gy) + 1e-6
        fr = Hn - T.round(Hn)
        d = T.abs(fr) / gm                                               # render px to the nearest level
        w = cfg["lw"] * s ** 0.5
        line = T.clamp(w + 0.6 - d, 0, 1)
        line = line * T.clamp((1 / gm - 1.5) / 2.0, 0, 1)                # fade where levels crowd (< ~3 px)
        return T.clamp(line, 0, 1), fill, a_tot

    def _shift(self, img, dx, dy):
        """img sampled at (pixel + (dx, dy)) render px, border-clamped (bilinear)."""
        T = self.T
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return img
        gx = (self.px + dx) / self.W * 2 - 1
        gy = (self.py + dy) / self.H * 2 - 1
        return T.nn.functional.grid_sample(img[None, None], T.stack([gx, gy], -1)[None], mode="bilinear",
                                           padding_mode="border", align_corners=False)[0, 0]

    # ------------------------------------------------------------------ frame
    def frame(self, n):
        T, cfg = self.T, self.cfg
        t = n / FPS
        cam = cam_at(self.src, t)
        foot = self.footage(n, cam)
        fo = 1 - float(smoothstep(*FOOT_OUT, t))
        mk = float(smoothstep(*MARK_IN, t))
        if cfg["tech"] == "flow":
            marks = self.marks_flow(n, t, cam)
            lum = 1 - (1 - foot * fo * 0.92) * (1 - marks * mk)
        else:
            line, fill, a_tot = self.marks_topo(n, t, cam, foot)
            lw = line * mk * (0.45 + 0.55 * a_tot)
            base = (foot * fo * 0.92)[..., None].expand(-1, -1, 3)
            if cfg.get("colour"):
                base = self.colour_layer(base, t, cam)
            img = 1 - (1 - base * (1 - 0.5 * lw[..., None])) * (1 - lw[..., None])
            img = T.maximum(img, fill[..., None])
            if cfg.get("colour"):
                img = self.accents(img, t, cam)
            return (T.clamp(img, 0, 1) * 255 + 0.5).to(T.uint8).cpu().numpy()
        img = T.clamp(lum, 0, 1)[..., None].expand(-1, -1, 3)
        return (img * 255 + 0.5).to(T.uint8).cpu().numpy()

    # ------------------------------------------------------------------ colour (palette E6), topo only
    BAND_COLS = (FOAM, CORAL, SAND)            # alternate echo bands round each phrase's key word
    ACCENT_COLS = (CORAL, SAND, CORAL)         # comma disc after WAVE, bar under LETTER, full stop after FORGOTTEN

    def _phase_amount(self, p, t, fall=0.6):
        return float(smoothstep(p["L"] - 0.5, p["L"], t) * (1 - smoothstep(p["D"], p["D"] + fall, t)))

    def colour_layer(self, base, t, cam):
        """Flat block colour in alternate echo bands (2..4.5 rings out) round the key word; bands travel outward
        with the rings. AA from the analytic level gradient (1 level = delta design px)."""
        T, cfg, s = self.T, self.cfg, self.s
        out = base
        for j, p in enumerate(self.ph):
            a = self._phase_amount(p, t)
            if a <= 0:
                continue
            sk = self._shift(p["sdfk_t"], (cam[0] - p["cam"][0]) * s, (cam[1] - p["cam"][1]) * s)
            sf = self._shift(p["sdf_t"], (cam[0] - p["cam"][0]) * s, (cam[1] - p["cam"][1]) * s)
            Hk = -sk / cfg["delta"] + cfg["omega"] * (t - p["L"])
            q = Hk / 2 - T.floor(Hk / 2)
            sd = T.where(q < 0.5, 0.25 - T.abs(q - 0.25), -(0.25 - T.abs(q - 0.75)))
            al = T.clamp(sd * 2 * cfg["delta"] * s + 0.5, 0, 1)
            win = T.clamp((sk - 1.5 * cfg["delta"]) * s + 0.5, 0, 1) * T.clamp((cfg["band_out"] * cfg["delta"] - sk) * s * 0.25, 0, 1)
            moat = T.clamp((sf - 1.2 * cfg["delta"]) * s + 0.5, 0, 1)        # keep a black moat round the other lines
            al = al * win * moat * a
            col = T.as_tensor(self.BAND_COLS[j], device=self.dev)
            out = out * (1 - al[..., None]) + col * al[..., None]
        return out

    def accents(self, img, t, cam):
        """Few solid shapes (analytic SDF, smoothstep edge): a disc rises through the sea and lands as the
        comma / full stop; a bar slides in under LETTER. They stay world-anchored, then leave on release."""
        T, s = self.T, self.s
        X, Y = self.px / s, self.py / s                                   # design px
        for j, p in enumerate(self.ph):
            a = self._phase_amount(p, t, fall=1.0)
            if a <= 0 or "kbox" not in p:
                continue
            x0, y0, x1, y1 = p["kbox"]
            h = y1 - y0
            ox, oy = cam[0] - p["cam"][0], cam[1] - p["cam"][1]          # world anchoring
            u = float(np.clip((t - (p["L"] - 1.1)) / 1.1, 0, 1))
            e = 1 - (1 - u) ** 3
            lv = float(np.clip((t - p["D"]) / (0.6 if j == 1 else 1.2), 0, 1)) ** 2
            col = T.as_tensor(self.ACCENT_COLS[j], device=self.dev)
            if j == 1:                                                    # bar under the key word
                bx0, bx1 = x0, x0 + (x1 - x0) * e
                bx0 = bx0 + (x1 - x0) * lv
                by0, by1 = y1 + 0.10 * h, y1 + 0.20 * h
                cx, cy, hx, hy = (bx0 + bx1) / 2 - ox, (by0 + by1) / 2 - oy, max((bx1 - bx0) / 2, 0), (by1 - by0) / 2
                dx, dy = T.abs(X - cx) - hx, T.abs(Y - cy) - hy
                d = T.sqrt(T.clamp(dx, min=0) ** 2 + T.clamp(dy, min=0) ** 2) + T.clamp(T.maximum(dx, dy), max=0)
            else:                                                         # disc = punctuation
                r = (0.17 if j == 0 else 0.24) * h
                cx = min(x1 + 0.22 * h + r, DW - r - 10) + 26 * (1 - e) * np.sin(7 * u) - ox
                cy = y1 - r + 560 * (1 - e) - 1100 * lv - oy
                d = T.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r
            al = T.clamp(0.5 - d * s, 0, 1)[..., None]
            img = img * (1 - al) + col * al
        return img


# ---------------------------------------------------------------------------------------------- driver
TIMINGS_HDR = "gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,encode_s,audio_s,notes\n"


def log_timing(nn, test, kind, device, frames, res, started, finished, render_s="", encode_s="", audio_s="",
               make_min="", notes=""):
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
    import datetime
    import gens
    t = datetime.datetime.now(gens._NZ) if gens._NZ else datetime.datetime.now()
    return t.strftime("%Y-%m-%d %H:%M:%S")


def render(scene, path, W, H):
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.perf_counter()
    for n in range(NFR):
        proc.stdin.write(scene.frame(n).tobytes())
    proc.stdin.close()
    t1 = time.perf_counter()
    proc.wait()
    t2 = time.perf_counter()
    print(f"{path}  {W}x{H}  {NFR} frames  render {t1 - t0:.1f}s  encode tail {t2 - t1:.1f}s")
    return t1 - t0, t2 - t1


def qc(video):
    os.makedirs(REVIEW, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    sheet = os.path.join(REVIEW, f"{base}_sheet.jpg")
    if not os.path.exists(sheet):
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", video, "-vf",
                        "fps=5,scale=180:-2,tile=10x5", "-frames:v", "1", sheet], check=True)
    out = [sheet]
    spans = [(0.0, PHRASES[0]["L"] + 0.2)] + [(PHRASES[k - 1]["D"] - 0.2, PHRASES[k]["L"] + 0.2)
                                             for k in range(1, len(PHRASES))]
    for k, (a, b) in enumerate(spans):
        strip = os.path.join(REVIEW, f"{base}_strip{k}.jpg")
        if not os.path.exists(strip):
            step = (b - a) / 7
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{b - a + 0.05:.3f}",
                            "-i", video, "-vf", f"fps={1 / step:.4f},scale=200:-2,tile=8x1", "-frames:v", "1",
                            strip], check=True)
        out.append(strip)
    print("\n".join(out))
    return out


def audio_for(nn, test, preset, device="cpu"):
    lands = ",".join(f"{p['L']:g}" for p in PHRASES)
    wav = os.path.join(LOCAL, f"score_{test}_g{nn:02d}.wav")
    while os.path.exists(wav):
        nn += 1
        wav = os.path.join(LOCAL, f"score_{test}_g{nn:02d}.wav")
    t0 = time.perf_counter()
    st = nz_now()
    dev = ["--backend", "torch", "--device", device] if device.startswith("cuda") else []
    subprocess.run([sys.executable, os.path.join(HERE, "score.py"), "--preset", preset, "--duration", f"{DUR:g}",
                    "--changes", lands, "--out", wav] + dev, check=True)
    return wav, st, time.perf_counter() - t0


def main():
    import gens
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true", help="build slowed sides + flow caches (CPU, once)")
    ap.add_argument("--variant", default="a", choices=sorted(VARIANTS))
    ap.add_argument("--preview", action="store_true", help="360x450")
    ap.add_argument("--final", action="store_true", help="720x900 + score + mux")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--qc", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device-label", default=None)
    ap.add_argument("--preset", default="E4", help="score.py preset for the final (E4 noise settling into pattern)")
    ap.add_argument("--gen", type=int, default=None, help="force generation number (box runs)")
    ap.add_argument("--bench", type=float, nargs="*", help="time frame() at these t (final size) and exit")
    ap.add_argument("--make-min", type=float, default=None)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--parent", default=None)
    a = ap.parse_args()
    cfg = VARIANTS[a.variant]
    T = xp.torch()
    T.set_num_threads(a.threads)
    be = xp.Backend("torch", a.device)
    label = a.device_label or ("cuda-RTX3060" if a.device.startswith("cuda") else "local-cpu-torch")
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    if a.prep:
        for src in SOURCES:
            prep(src)
        return
    if a.qc:
        qc(a.qc)
    if a.bench:
        t0 = time.perf_counter()
        sc = Scene(cfg, be, DW, DH)
        be.sync()
        print(f"setup {time.perf_counter() - t0:.2f}s")
        sc.frame(int(a.bench[0] * FPS))
        for t in a.bench:
            t0 = time.perf_counter()
            sc.frame(int(round(t * FPS)))
            be.sync()
            print(f"bench t={t:.2f} 720x900 torch/{a.device}: {1000 * (time.perf_counter() - t0):.1f} ms")
        return
    test, stem = cfg["test"], cfg["stem"]
    h = gens.code_hash([os.path.abspath(__file__)]) + "-" + a.variant
    for kind, (W, H), on in (("preview", (360, 450), a.preview), ("final", (720, 900), a.final)):
        if not on:
            continue
        nn = a.gen if a.gen is not None else gens.gen_for(LOCAL, test, stem, kind, h, a.parent, a.trigger, a.change)
        name = gens.video_name(stem, nn, kind)
        started = nz_now()
        t0 = time.perf_counter()
        sc = Scene(cfg, be, W, H)
        be.sync()
        setup_s = time.perf_counter() - t0
        rs, es = render(sc, os.path.join(LOCAL, name), W, H)
        log_timing(nn, test, kind, label, NFR, f"{W}x{H}", started, nz_now(), rs, es,
                   make_min=a.make_min if a.make_min is not None else "",
                   notes=f"setup_s={setup_s:.1f} (sides/flow load + glyphs + particle integration); render_s = frame loop incl. piped encode")
        stills = []
        st_sc = sc if W == 360 else None
        st_sc = st_sc or Scene(cfg, be, 360, 450)
        for t in (PHRASES[1]["F"] + 0.5, PHRASES[1]["L"] + 0.6):
            fn = f"{test}_g{nn:02d}_still_{t:.2f}.jpg"
            if not os.path.exists(os.path.join(REVIEW, fn)):
                Image.fromarray(st_sc.frame(int(round(t * FPS)))).save(os.path.join(REVIEW, fn), quality=90)
            stills.append(fn)
        files = [name]
        if kind == "final":
            wav, st_a, aud_s = audio_for(nn, test, a.preset, a.device)
            log_timing(nn, test, "music", label, 0, "", st_a,
                       nz_now(), audio_s=aud_s, notes=f"score.py --preset {a.preset}; {os.path.basename(wav)}")
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
        sc = Scene(cfg, be, 360, 450)
        for t in a.still:
            Image.fromarray(sc.frame(int(round(t * FPS)))).save(os.path.join(REVIEW, f"{test}_adhoc_{t:.2f}.png"))


if __name__ == "__main__":
    main()
