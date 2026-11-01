"""Test B -- inflated dot field.

Puffy glyphs (Pilowlava) drawn only as a fine regular dot grid. Each phrase is a signed-distance field
inflated to a balloon height h = sqrt(clip(d/R, 0, 1)); phrases melt into each other by lerping the SDFs
(smoothed, with a transient domain warp) under a closed-form damped-spring progress (zeta 0.7). Every grid
dot is lifted up the screen by k*h (pseudo-3D, viewed from just above), culled by a per-column horizon,
shaded Lambert x travelling contour bands, and splatted back to front.

Every frame is a pure function of t: ``frame(t, W, H)``.

    python tools/mathtype/dotfield.py --draft            # 360x450 -> local/B_dotfield_draft.mp4
    python tools/mathtype/dotfield.py --final            # 1080x1350 -> local/B_dotfield.mp4 (+ _hum mux)
    python tools/mathtype/dotfield.py --stills 0.3 1.0   # PNGs into review/ for inspection
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VDIR = os.path.join(REPO, "11_mathtype", "v001")
LOCAL = os.path.join(VDIR, "local")
REVIEW = os.path.join(VDIR, "review")
FONT_DIR = os.path.join(REPO, "mograph", "fonts")
with open(os.path.join(FONT_DIR, "fonts.json"), encoding="utf-8") as _f:
    _FONTS = json.load(_f)["fonts"]
FACE = os.environ.get("DOTFIELD_FACE", "Climate Crisis")
CREDIT_FACE = "Instrument Sans"

FPS, NFRAMES = 25, 125
BG = np.array([5, 5, 5], np.float32)
INK = np.array([0xF3, 0xEA, 0xDB], np.float32)
ACC = np.array([0xE2, 0x58, 0x2A], np.float32)

FW, FH = 540, 675                       # field resolution (independent of output size)
PHRASES = [("THERE IS", "GEOMETRY"), ("IN THE", "HUMMING"), ("OF THE", "STRINGS")]
# (start time, natural frequency) of each spring: lattice->P1, P1->P2, P2->P3
SPRINGS = [(0.0, 15.0), (1.27, 8.5), (2.87, 8.5)]
SWEEP = [0.07, 0.11, 0.11]   # max per-pixel delay: the melt travels left -> right across the lines
ZETA = 0.70
HOLDS = [(0.45, 1.27), (2.05, 2.87), (3.65, 5.0)]   # settled windows (accent band sweeps once in each)
CREDIT = "— PYTHAGORAS"
CREDIT_T0 = 4.2

STRETCH_Y = 1.5           # vertical stretch of the glyphs to fill the 4:5 frame
TRACK = 0.09              # extra letter spacing (em)
ROUND = 1.8               # corner rounding blur (field px)
PITCH_1080 = 4.0          # dot pitch at 1080 wide
DOT_R_1080 = 1.2          # base dot radius at 1080 wide
BANDS = 5.0               # contour bands across h in [0,1]
BAND_SPEED = 0.55         # bands per second (travel inward)
LIFT = 0.80               # screen lift k in units of R (output px)
PUFF = 2.4                # normal-shading height in units of R


# ----------------------------------------------------------------------------- math
def spring(t: float, t0: float, w: float, zeta: float = ZETA) -> float:
    """Closed-form under-damped step response, 0 at t0, -> 1."""
    tau = t - t0
    if tau <= 0:
        return 0.0
    wd = w * math.sqrt(1 - zeta * zeta)
    return 1.0 - math.exp(-zeta * w * tau) * (math.cos(wd * tau) + zeta * w / wd * math.sin(wd * tau))


def spring_v(tau: np.ndarray, w: float, zeta: float = ZETA) -> np.ndarray:
    """Vectorised step response for an array of local times (0 where tau <= 0)."""
    tau = np.maximum(tau, 0.0)
    wd = w * math.sqrt(1 - zeta * zeta)
    return 1.0 - np.exp(-zeta * w * tau) * (np.cos(wd * tau) + zeta * w / wd * np.sin(wd * tau))


def smoothstep(a: float, b: float, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ----------------------------------------------------------------------------- fields (cached per process)
_CACHE: dict = {}


def _font(name: str, size: float) -> ImageFont.FreeTypeFont:
    e = _FONTS[name]
    f = ImageFont.truetype(os.path.join(FONT_DIR, e["file"]), max(1, int(round(size))))
    if e.get("axes"):
        try:
            f.set_variation_by_axes([float(v) for v in e["axes"].values()])
        except Exception:
            pass
    return f


def _phrase_mask(lines) -> np.ndarray:
    """Two justified lines of FACE, each fitted to the same ink width, centred in the field."""
    SS = 3
    W, H = FW * SS, FH * SS
    target_w = 0.88 * W
    max_cap = 0.27 * H
    gap = 0.06 * H
    items = []
    for s in lines:
        li = _tracked_line(s, 400)
        size = 400 * target_w / li.width
        size = min(size, 400 * max_cap / (li.height * STRETCH_Y))
        li = _tracked_line(s, size)
        li = li.resize((li.width, int(round(li.height * STRETCH_Y))), Image.LANCZOS)   # tall balloons
        items.append(li)
    total = sum(li.height for li in items) + gap * (len(items) - 1)
    y = 0.5 * H - 0.5 * total + 0.035 * H         # sit a little low: the lift pushes the surface up
    im = Image.new("L", (W, H), 0)
    for li in items:
        im.paste(li, (int(0.5 * W - 0.5 * li.width), int(y)))
        y += li.height + gap
    m = np.asarray(im.resize((FW, FH), Image.LANCZOS), np.float32) / 255.0
    m = gaussian_filter(m, ROUND)           # round the corners -> balloon silhouettes
    return m > 0.5


def _tracked_line(s: str, size: float) -> Image.Image:
    """One line of FACE with extra tracking (inflation needs air between letters), cropped to ink."""
    f = _font(FACE, size)
    track = TRACK * size
    widths = [f.getlength(ch) for ch in s]
    W = int(sum(widths) + track * len(s) + size)
    H = int(size * 1.6)
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    x = 0.5 * size
    for ch, w in zip(s, widths):
        dr.text((x, 0.3 * size), ch, font=f, fill=255)
        x += w + (track if ch != " " else 0)
    return im.crop(im.getbbox())


def _sdf(mask: np.ndarray) -> np.ndarray:
    d = distance_transform_edt(mask) - distance_transform_edt(~mask)
    return gaussian_filter(d.astype(np.float32), 1.0)


def fields():
    if "F" not in _CACHE:
        F = [_sdf(_phrase_mask(p)) for p in PHRASES]
        Rs = [float(np.percentile(f[f > 0], 97)) for f in F]
        yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
        # text band envelope for the opening lattice
        m = np.zeros((FH, FW), np.float32)
        for f in F:
            m = np.maximum(m, (f > -12).astype(np.float32))
        env = gaussian_filter(m, 14)
        _CACHE.update(F=F, R=Rs, xx=xx, yy=yy, env=env / env.max())
    return _CACHE


def lattice_sdf(t: float) -> np.ndarray:
    """Opening state: an egg-crate of small breathing puffs, confined to the text band."""
    c = fields()
    xx, yy, R = c["xx"], c["yy"], c["R"][0]
    k = 2 * math.pi / 34.0
    v = np.sin(xx * k + 2.1 * t) * np.sin(yy * k - 1.4 * t + 0.6)
    return (R * 1.6 * (v - 0.25)) * c["env"] - (1 - c["env"]) * 2 * R


def sdf_at(t: float):
    """Signed distance field, puff radius R and transition intensity at time t."""
    c = fields()
    F, Rs = c["F"], c["R"]
    stage = max(i for i, (t0, _) in enumerate(SPRINGS) if t >= t0 or i == 0)
    t0, w = SPRINGS[stage]
    xx, yy = c["xx"], c["yy"]
    delay = SWEEP[stage] * (0.8 * xx / FW + 0.2 * (0.5 + 0.5 * np.sin(yy * 0.021 + xx * 0.008)))
    pf = spring_v(t - t0 - delay, w).astype(np.float32)   # per-pixel progress: the melt travels
    p = float(pf.mean())
    A = lattice_sdf(t) if stage == 0 else F[stage - 1]
    RA = Rs[0] if stage == 0 else Rs[stage - 1]
    B, RB = F[stage], Rs[stage]
    d = (1 - pf) * A + pf * B
    R = (1 - p) * RA + p * RB
    mf = np.clip(4 * pf * (1 - pf), 0, 1)                  # 0 at both ends, 1 mid-morph
    melt = float(mf.max())
    if melt > 1e-3:
        amp = 13.0 * mf
        dx = amp * np.sin(0.031 * yy + 0.017 * xx + 7.0 * t + stage)
        dy = amp * 0.8 * np.sin(0.027 * xx - 0.012 * yy - 5.0 * t + 2 * stage)
        d = cv2.remap(d, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = d * (1 - mf) + cv2.GaussianBlur(d, (0, 0), 3.5) * mf    # smoothed where melting
    return d, R, melt, stage, p


# ----------------------------------------------------------------------------- render
def frame(t: float, W: int = 1080, H: int = 1350) -> np.ndarray:
    """RGB uint8 frame at time t (pure function of t)."""
    c = fields()
    d, R, melt, stage, p = sdf_at(t)
    h = np.sqrt(np.clip(d / R, 0, 1)).astype(np.float32)
    Hs = cv2.GaussianBlur(h, (0, 0), 0.8) * (R * PUFF)
    gy, gx = np.gradient(Hs)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny = -gx * nz, -gy * nz

    # slowly orbiting light, ~35-50 deg elevation
    a = 2.3 + 2 * math.pi * t / 8.0
    el = math.radians(32 + 6 * math.sin(t * 0.9))
    L = (math.cos(a) * math.cos(el), math.sin(a) * math.cos(el) * 0.8 - 0.25, math.sin(el))
    Ln = math.sqrt(sum(v * v for v in L))
    lam = np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / Ln, 0, 1)
    shade = 0.16 + 0.84 * lam

    # dot grid (output px) -> field coords
    s = W / 1080.0
    pitch = PITCH_1080 * s
    nxg, nyg = int(W / pitch), int(H / pitch)
    gxs = (np.arange(nxg) + 0.5) * W / nxg
    gys = (np.arange(nyg) + 0.5) * H / nyg
    GX, GY = np.meshgrid(gxs, gys)
    mx = (GX * FW / W).astype(np.float32)
    my = (GY * FH / H).astype(np.float32)
    hg = cv2.remap(h, mx, my, cv2.INTER_LINEAR)
    shg = cv2.remap(shade.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    inside = hg > 0.0

    # travelling contour bands with an undulating warp
    xf = GX / W
    yf = GY / H
    warp = 0.10 * np.sin(xf * 11.0 + 1.3 * t) + 0.06 * np.sin(yf * 17.0 - 0.9 * t + xf * 4.0)
    ph = hg * BANDS - t * BAND_SPEED + warp
    band = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
    line = band ** 4
    b = shg * (0.45 + 0.55 * line)
    b = np.clip(1.6 * b * (0.93 + 0.07 * hg), 0, 1)

    # accent: one band sweeping crest -> rim once per settled hold
    acc = np.zeros_like(b)
    for i, (h0, h1) in enumerate(HOLDS):
        if h0 <= t <= h1 and i == stage:
            u = (t - h0) / (h1 - h0)
            cpos = 1.15 - 1.4 * u
            env = math.sin(math.pi * min(1.0, u * 1.15)) if u < 1 / 1.15 else 0.0
            acc = env * np.exp(-((hg - cpos) / 0.045) ** 2) * (0.6 + 0.4 * line)

    # pseudo-3D lift + horizon cull (a dot is hidden if any dot in front of it, same column, rises above it)
    kpx = LIFT * R * (W / FW)
    sy = GY - kpx * hg
    big = np.where(inside, sy, np.inf)
    front_min = np.minimum.accumulate(big[::-1], axis=0)[::-1]
    front_min = np.vstack([front_min[1:], np.full((1, nxg), np.inf)])
    vis = inside & (sy < front_min - 0.15 * pitch)

    # painter's order: rows top (far) -> bottom (near); grid is already row-major
    idx = np.flatnonzero(vis.ravel())
    px = GX.ravel()[idx]
    py = sy.ravel()[idx]
    bb = np.clip(b.ravel()[idx], 0, 1)
    aa = np.clip(acc.ravel()[idx], 0, 1)
    rad = DOT_R_1080 * s * (0.70 + 0.55 * bb)
    col = BG[None] + (INK - BG)[None] * (0.08 + 0.92 * bb[:, None])
    col = col * (1 - aa[:, None]) + (ACC[None] * (0.55 + 0.45 * bb[:, None])) * aa[:, None]

    SS = max(3, int(round(3240 / W)))
    CW, CH = W * SS, H * SS
    canvas = np.empty((CH * CW, 3), np.uint8)
    canvas[:] = BG.astype(np.uint8)
    rmax = float(rad.max()) * SS if len(rad) else 1.0
    ro = int(math.ceil(rmax))
    oy, ox = np.mgrid[-ro:ro + 1, -ro:ro + 1]
    oy, ox = oy.ravel(), ox.ravel()
    od2 = (ox * ox + oy * oy).astype(np.float32)
    cx = np.round(px * SS).astype(np.int64)
    cy = np.round(py * SS).astype(np.int64)
    keep = od2[None, :] <= ((rad * SS) ** 2)[:, None]
    di, oi = np.nonzero(keep)                         # di ascending -> painter's order preserved
    X = cx[di] + ox[oi]
    Y = cy[di] + oy[oi]
    ok = (X >= 0) & (X < CW) & (Y >= 0) & (Y < CH)
    canvas[Y[ok] * CW + X[ok]] = np.clip(col[di[ok]], 0, 255).astype(np.uint8)
    img = cv2.resize(canvas.reshape(CH, CW, 3), (W, H), interpolation=cv2.INTER_AREA)

    # credit: small, tracked in from wide as it fades up
    if t >= CREDIT_T0:
        u = float(smoothstep(CREDIT_T0, CREDIT_T0 + 0.6, t))
        img = _credit(img, u, W, H)
    return img


def _credit(img: np.ndarray, u: float, W: int, H: int) -> np.ndarray:
    SS = 3
    size = 0.022 * H * SS
    f = _font(CREDIT_FACE, size)
    track = (0.55 - 0.30 * u) * size
    widths = [f.getlength(ch) for ch in CREDIT]
    tw = sum(widths) + track * (len(CREDIT) - 1)
    lay = Image.new("L", (W * SS, int(size * 2)), 0)
    dr = ImageDraw.Draw(lay)
    x = 0.5 * W * SS - 0.5 * tw
    for ch, w in zip(CREDIT, widths):
        dr.text((x, 0.3 * size), ch, font=f, fill=255)
        x += w + track
    m = np.asarray(lay.resize((W, lay.height // SS), Image.LANCZOS), np.float32)[..., None] / 255.0 * u * 0.75
    y0 = int(0.885 * H)
    roi = img[y0:y0 + m.shape[0]].astype(np.float32)
    m = m[: roi.shape[0]]
    img[y0:y0 + m.shape[0]] = (roi * (1 - m) + INK[None, None] * m).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------- output
def _worker(args):
    i, W, H = args
    return frame(i / FPS, W, H).tobytes()


def render(path: str, W: int, H: int, procs: int = 6) -> float:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
           "-movflags", "+faststart", path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    jobs = [(i, W, H) for i in range(NFRAMES)]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            for buf in pool.imap(_worker, jobs, chunksize=2):
                ff.stdin.write(buf)
    else:
        for j in jobs:
            ff.stdin.write(_worker(j))
    ff.stdin.close()
    ff.wait()
    dt = time.time() - t0
    print(f"wrote {path} in {dt:.1f}s")
    return dt


def mux(video: str, out: str) -> bool:
    wav = os.path.join(LOCAL, "hum.wav")
    if not os.path.exists(wav):
        return False
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", out], check=True)
    print(f"wrote {out}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--size", default="1080x1350", help="size for --stills")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.stills:
        W, H = map(int, a.size.split("x"))
        os.makedirs(REVIEW, exist_ok=True)
        for t in a.stills:
            p = os.path.join(REVIEW, f"B_dotfield_still_{W}_{t:.2f}.png")
            Image.fromarray(frame(t, W, H)).save(p)
            print(p)
    if a.draft:
        render(os.path.join(LOCAL, "B_dotfield_draft.mp4"), 360, 450, a.procs)
    if a.final:
        v = os.path.join(LOCAL, "B_dotfield.mp4")
        render(v, 1080, 1350, a.procs)
        if not mux(v, os.path.join(LOCAL, "B_dotfield_hum.mp4")):
            print("hum.wav not present; silent only")


if __name__ == "__main__":
    main()
