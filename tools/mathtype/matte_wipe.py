#!/usr/bin/env python
"""Music-driven matte wipe between two filmic sea shots (v005, test D_wipe_<mode>).

A side is the background; the silk-wind strand matte (silkwind.py --matte) is the window through which
the B side shows. Per frame, all in float RGB:
    m   = clip(gain * matte, 0, 1) ** gamma
    out = A (1 - k(t) m) + B k(t) m
k(t) is the semi-transparency: k_pre (~0.35-0.55) while the strands are far threads, rising to 1.0 through
the wipe (default curve from the matte's sidecar json: wipe_at / wipe_dur; or --opacity-curve "t:k,...",
smoothstep between knots). Before the wipe the threads mostly screen B over A (a double exposure, so a dark
B never cuts dark slashes), becoming a true replace through the wipe. After the wipe, matte gaps still show
A faintly (--after-floor). Optional: a shared film grade
(B blacks screen-lifted to a deep-sea blue so dark-B threads still read, gentle S curve), seeded grain (frame-indexed,
deterministic) and a slight glow on matte edges where B shows.

    python tools/mathtype/matte_wipe.py --sides                                  # build A / B 12 s sides
    python tools/mathtype/matte_wipe.py --a A.mp4 --b B.mp4 --matte D_wipe_beaty_g01_matte.mp4 --audio x.wav \
        [--opacity-curve "0:0.45,9.4:0.5,10.2:1"] [--gain ..] [--gamma ..] [--trigger ..] [--change ..]
Output: v005/local/D_wipe_<mode>_gNN_preview.mp4 + _preview_audio.mp4 (gNN = the matte's, or the next one if
that is taken: never overwrite), stills + QC strip in v005/review, row in local/GENERATIONS.md.
GPU: --backend torch --device cuda blends on the device (same math, same seeded grain).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VDIR = os.path.join(REPO, "11_mathtype", "v005")
LOCAL, REVIEW = os.path.join(VDIR, "local"), os.path.join(VDIR, "review")
CLIPS = os.path.join(REPO, "04_routes", "v001", "local", "clips")
FPS, T = 25, 12.0
NFR = int(round(FPS * T))
SEED = 5

# sides: (clip, crop x, crop y, crop w, crop h) in source px; each clip slowed to fill 12 s with
# motion-compensated interpolation (one continuous shot reads more natural than the i2v/flf chain, which
# leaves the sea for a window / road / submarine).
SIDES = {"A": ("S1_i2v_0", 422, 0, 410, 512),       # dusk shore, sun on the right third, calm waves
         "B": ("B5_i2v_0", 0, 83, 512, 640)}        # bioluminescent jellyfish on deep-sea black
SIDE_W, SIDE_H = 720, 900
MCI = "minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"



def smoothstep(a, b, x):
    x = np.clip((x - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def free(path: str) -> str:
    """Never overwrite: bump a trailing _gNN until free."""
    while os.path.exists(path):
        m = re.search(r"_g(\d+)(\.\w+)$", path)
        path = path[:m.start()] + f"_g{int(m.group(1)) + 1:02d}" + m.group(2)
    return path


def build_sides() -> list[str]:
    out = []
    for side, (clip, x, y, w, h) in SIDES.items():
        src = os.path.join(CLIPS, clip + ".mp4")
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "csv=p=0", src], capture_output=True, text=True).stdout)
        slow = (T + 0.25) / dur
        dst = free(os.path.join(LOCAL, f"D_wipe_side{side}_g01.mp4"))
        vf = f"setpts=PTS*{slow:.5f},crop={w}:{h}:{x}:{y},scale={SIDE_W}:{SIDE_H}:flags=lanczos,{MCI}"
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", src, "-vf", vf, "-t", str(T), "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", dst], check=True)
        print(side, clip, f"x{1 / slow:.3f} speed", dst)
        out.append(dst)
    return out


def read_all(path: str, W: int, H: int, grey=False) -> np.ndarray:
    cap, fr = cv2.VideoCapture(path), []
    while len(fr) < NFR:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        fr.append(f[..., 0] if grey else f)
    while len(fr) < NFR:                  # hold the last frame if a side is a frame short
        fr.append(fr[-1])
    return np.stack(fr)


def opacity(t, curve, k_pre, wipe_at, wipe_dur):
    """k(t): from knots 't:k,...' (smoothstep between knots), else k_pre -> 1 on the wipe sweep."""
    t = np.asarray(t, float)
    if curve:
        kn = sorted((float(a), float(b)) for a, b in (p.split(":") for p in curve.split(",")))
        k = np.full_like(t, kn[0][1])
        for (t0, k0), (t1, k1) in zip(kn, kn[1:]):
            k = np.where(t >= t0, k0 + (k1 - k0) * smoothstep(t0, t1, t), k)
        return k
    return k_pre + (1 - k_pre) * smoothstep(wipe_at - 0.25 * wipe_dur, wipe_at + 0.45 * wipe_dur, t)


BE = None   # --backend torch: an xp.Backend; composite() then blends on the device (_composite_torch)


def _composite_torch(A, B, M, n, k, a) -> np.ndarray:
    """composite() in torch float32 on BE's device; same math, same seeded grain (drawn by numpy)."""
    import xp
    T, dev = xp.torch(), BE.dev
    fa = T.from_numpy(A[n]).to(dev).float()
    fb = T.from_numpy(B[n]).to(dev).float()
    L = T.tensor([int(a.lift_col[i:i + 2], 16) for i in (4, 2, 0)], dtype=T.float32, device=dev) * a.lift
    fb = 255 - (255 - fb) * (255 - L) / 255.0
    m = T.clamp(a.gain * T.from_numpy(M[n]).to(dev).float() / 255.0, 0, 1) ** a.gamma
    prog = float(np.clip((k - a.k_pre) / (1 - a.k_pre), 0, 1) if not a.opacity_curve else np.clip(k, 0, 1) ** 4)
    m = m + (1 - m) * a.after_floor * prog * prog
    km = (float(k) * m)[..., None]
    if a.ghost_screen > 0:
        scr = 255 - (255 - fa) * (255 - fb) / 255.0
        fb = fb + a.ghost_screen * (1 - prog) * (scr - fb)
    out = fa * (1 - km) + fb * km
    if a.glow > 0:
        halo = T.clamp(xp.gaussian_blur(km[..., 0], 2.5) - km[..., 0], 0, 1)[..., None]
        out = out + a.glow * halo * xp.gaussian_blur(fb.permute(2, 0, 1), 2.5).permute(1, 2, 0)
    x = T.clamp(out / 255.0, 0, 1)
    x = x + a.curve * x * (1 - x) * (2 * x - 1)
    g = x.mean(-1, keepdim=True)
    x = g + a.sat * (x - g)
    if a.grain > 0:
        rng = np.random.default_rng(SEED * 100003 + n)
        gr = xp.gaussian_blur(T.from_numpy(rng.normal(0, 1, x.shape[:2]).astype(np.float32)).to(dev), 0.6)
        x = x + a.grain * gr[..., None] * (0.4 + 0.6 * T.sqrt(T.clamp(g, 0, 1)))
    return T.clamp(x * 255 + 0.5, 0, 255).to(T.uint8).cpu().numpy()


def composite(A, B, M, n, k, a) -> np.ndarray:
    if BE is not None and BE.torch:
        return _composite_torch(A, B, M, n, k, a)
    fa, fb = A[n].astype(np.float32), B[n].astype(np.float32)
    L = np.array([int(a.lift_col[i:i + 2], 16) for i in (4, 2, 0)], np.float32) * a.lift   # hex RGB -> BGR
    fb = 255 - (255 - fb) * (255 - L) / 255.0                         # screen-lift B blacks to deep-sea blue
    m = np.clip(a.gain * M[n].astype(np.float32) / 255.0, 0, 1) ** a.gamma
    prog = np.clip((k - a.k_pre) / (1 - a.k_pre), 0, 1) if not a.opacity_curve else np.clip(k, 0, 1) ** 4
    m = m + (1 - m) * a.after_floor * prog * prog                      # after the wipe, gaps show A faintly
    km = (k * m)[..., None]
    if a.ghost_screen > 0:                 # before the wipe the threads double-expose B over A (screen), so a
        scr = 255 - (255 - fa) * (255 - fb) / 255.0     # dark B never cuts dark slashes; becomes a true replace
        fb = fb + a.ghost_screen * (1 - prog) * (scr - fb)      # as the wipe completes
    out = fa * (1 - km) + fb * km
    if a.glow > 0:                         # slight glow on matte edges where B shows: B's colour bleeding out
        halo = np.clip(cv2.GaussianBlur(km[..., 0], (0, 0), 2.5) - km[..., 0], 0, 1)[..., None]
        out = out + a.glow * halo * cv2.GaussianBlur(fb, (0, 0), 2.5)
    x = np.clip(out / 255.0, 0, 1)         # shared grade: gentle S curve, slight desaturation
    x = x + a.curve * x * (1 - x) * (2 * x - 1)
    g = x.mean(-1, keepdims=True)
    x = g + a.sat * (x - g)
    if a.grain > 0:
        rng = np.random.default_rng(SEED * 100003 + n)
        gr = cv2.GaussianBlur(rng.normal(0, 1, x.shape[:2]).astype(np.float32), (0, 0), 0.6)
        x = x + a.grain * gr[..., None] * (0.4 + 0.6 * np.sqrt(np.clip(g, 0, 1)))
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", action="store_true", help="build the A / B 12 s sides from the LTX clips")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--matte", help="D_wipe_<mode>_gNN_matte.mp4 (silkwind.py --matte)")
    ap.add_argument("--audio", help="wav to mux (the v004 one for that mode)")
    ap.add_argument("--opacity-curve", default="", help="k(t) knots 't:k,...'; default k_pre -> 1 on the wipe")
    ap.add_argument("--k-pre", type=float, default=0.45)
    ap.add_argument("--gain", type=float, default=2.5, help="matte gain before gamma")
    ap.add_argument("--gamma", type=float, default=0.6, help="matte gamma")
    ap.add_argument("--lift", type=float, default=1.0, help="B black screen-lift amount (0-1)")
    ap.add_argument("--lift-col", default="0e2a40", help="B black lift colour (hex RGB, deep-sea blue)")
    ap.add_argument("--ghost-screen", type=float, default=0.75, help="pre-wipe threads: share of screen blend")
    ap.add_argument("--after-floor", type=float, default=0.3, help="matte floor at full wipe (gaps show A at 1-x)")
    ap.add_argument("--glow", type=float, default=0.35)
    ap.add_argument("--curve", type=float, default=0.25, help="shared S-curve amount")
    ap.add_argument("--sat", type=float, default=0.92)
    ap.add_argument("--grain", type=float, default=0.018)
    ap.add_argument("--size", default="360x450")
    ap.add_argument("--parent", default="")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy", help="blend backend (xp.py)")
    ap.add_argument("--device", default="cpu", help="torch device: cpu | cuda")
    a = ap.parse_args()
    global BE
    if a.backend == "torch":
        sys.path.insert(0, HERE)
        import xp
        BE = xp.Backend("torch", a.device)
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(REVIEW, exist_ok=True)
    if a.sides:
        build_sides()
        return
    sys.path.insert(0, HERE)
    import gens as G
    W, H = (int(v) for v in a.size.split("x"))
    mb = os.path.basename(a.matte)
    mm = re.match(r"(D_wipe_(beaty|ethereal))_g(\d+)_matte\.mp4$", mb)
    stem, mode, mg = mm.group(1), mm.group(2), int(mm.group(3))
    with open(a.matte.replace(".mp4", ".json"), encoding="utf-8") as f:
        mp = json.load(f)
    g = mg
    if os.path.exists(os.path.join(LOCAL, f"{stem}_g{g:02d}_preview.mp4")):   # compositor-only change: new gen
        g = max([mg] + [int(m.group(1)) for f in os.listdir(LOCAL)
                        for m in [re.match(re.escape(stem) + r"_g(\d+)_", f)] if m]) + 1
        while os.path.exists(os.path.join(LOCAL, f"{stem}_g{g:02d}_preview.mp4")):
            g += 1
        G.backfill(LOCAL, stem, g, G.now_nz(), a.parent or f"{stem} g{mg:02d}", a.trigger,
                   (a.change + f" (matte g{mg:02d} reused)").strip(), [mb], [])
    path = os.path.join(LOCAL, f"{stem}_g{g:02d}_preview.mp4")
    t0 = time.time()
    A, B = read_all(a.a, W, H), read_all(a.b, W, H)
    M = read_all(a.matte, W, H, grey=True)
    ts = np.arange(NFR) / FPS
    K = opacity(ts, a.opacity_curve, a.k_pre, mp["wipe_at"], mp["wipe_dur"])
    cmd = ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cover = []
    for n in range(NFR):
        proc.stdin.write(composite(A, B, M, n, K[n], a).tobytes())
        m = np.clip(a.gain * M[n] / 255.0, 0, 1) ** a.gamma
        prog = np.clip((K[n] - a.k_pre) / (1 - a.k_pre), 0, 1) if not a.opacity_curve else np.clip(K[n], 0, 1) ** 4
        m = m + (1 - m) * a.after_floor * prog * prog
        cover.append(float((K[n] * m).mean()))
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    out = path.replace(".mp4", "_audio.mp4")
    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", path, "-i", a.audio, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", out], check=True)
    # stills (before / wipe moment / after) and a QC strip read back from the encoded video
    wa = mp["wipe_at"]
    still_t = (round(wa - 3.0, 2), round(wa, 2), round(min(wa + 1.6, T - 0.2), 2))
    rows_t = [("before", [wa - 5.0, wa - 4.0, wa - 3.0, wa - 2.0, wa - 1.0]),
              ("during", [wa - 0.4, wa - 0.2, wa, wa + 0.2, wa + 0.4, wa + 0.6]),
              ("after", [wa + 0.8, wa + 1.2, (wa + T) / 2 + 0.4, T - 0.6, T - 0.08])]
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fidx = lambda t: int(np.clip(round(t * FPS), 0, len(frames) - 1))
    stills = []
    for t in still_t:
        nm = f"{stem}_g{g:02d}_still_{t:.1f}.jpg"
        cv2.imwrite(os.path.join(REVIEW, nm), frames[fidx(t)], [cv2.IMWRITE_JPEG_QUALITY, 92])
        stills.append(nm)
    rows = []
    for label, tt in rows_t:
        tiles = []
        for t in tt:
            im = cv2.resize(frames[fidx(t)], (240, 300), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{t:.2f} k{K[fidx(t)]:.2f} c{cover[fidx(t)]:.2f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (42, 88, 226), 1, cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < 6:
            tiles.append(np.full_like(tiles[0], 5))
        bar = np.full((22, 240 * 6, 3), 20, np.uint8)
        cv2.putText(bar, f"{label}  (wipe at {wa:.2f} s)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (219, 234, 243), 1,
                    cv2.LINE_AA)
        rows.append(np.concatenate([bar, np.concatenate(tiles, 1)], 0))
    strip = f"{stem}_g{g:02d}_qc_strip.jpg"
    cv2.imwrite(os.path.join(REVIEW, strip), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    G.add_files(LOCAL, stem, g, [mb, os.path.basename(path), os.path.basename(out), os.path.basename(a.audio)],
                stills + [strip])
    params = {k: v for k, v in vars(a).items() if k not in ("parent", "trigger", "change")}
    with open(path.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(params, matte_params=mp, render_s=round(dt, 1)), f, indent=1)
    c = np.array(cover)
    print(f"{out}  {W}x{H}  composite {dt:.1f}s  coverage k*m: before {c[:int(FPS * (wa - 1))].mean():.3f}  "
          f"after {c[int(FPS * (wa + 0.8)):].mean():.3f} (min {c[int(FPS * (wa + 0.8)):].min():.3f})")
    print("stills", stills, "strip", strip)


if __name__ == "__main__":
    main()
