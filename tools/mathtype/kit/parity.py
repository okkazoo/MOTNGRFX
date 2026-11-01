#!/usr/bin/env python
"""Migration parity: render sampled frames through a piece's old code path and its --kit path, and compare
both with each other and with the delivered final (decoded). PSNR in dB (8-bit, peak 255).

    python tools/mathtype/kit/parity.py e4 --ref 11_mathtype/v010/local/E4b_rings_g04_final.mp4 --n 12
    python tools/mathtype/kit/parity.py e1 --ref 11_mathtype/v007/local/<E1 final>.mp4 --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import kit  # noqa: E402,F401  (sets sys.path for the repo)


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def decode(path, W, H, gray):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape((-1, H, W) if gray else (-1, H, W, 3))


def scene_e4(device, kit_path):
    import moire
    import xp
    moire.PHRASES = moire.FONT_SETS["mix"]
    if kit_path:
        moire.use_kit()
    sc = moire.Scene("ring", xp.Backend("torch", device))
    return (lambda n, W, H: sc.frame(n / moire.FPS, W, H)), int(round(moire.FPS * moire.DUR)), True


def scene_e1(device, kit_path, variant="terrace", colour=True, fontmix=True, W=720, H=900, backend="numpy"):
    import contours
    if kit_path:
        contours.use_kit()
    sc = contours.Scene(variant=variant, colour=colour, backend=backend, device=device, W=W, H=H, fontmix=fontmix)
    return (lambda n, W_, H_: sc.frame(n / contours.FPS)), contours.NFR, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=["e4", "e1"])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--e1", default="terrace,colour,mix,numpy", help="E1 variant,colour|-,mix|-,backend")
    ap.add_argument("--out", default=None, help="JSON report path")
    a = ap.parse_args()
    W, H = a.size, int(a.size * 1.25)
    res = {}
    for path_kind in ("old", "kit"):
        if a.piece == "e4":
            fn, nfr, gray = scene_e4(a.device, path_kind == "kit")
        else:
            v, c, m, b = a.e1.split(",")
            fn, nfr, gray = scene_e1(a.device, path_kind == "kit", v, c == "colour", m == "mix", W, H, b)
        idx = np.linspace(0, nfr - 1, a.n).round().astype(int)
        fn(0, W, H)                                                    # warm-up (setup)
        t0 = time.perf_counter()
        res[path_kind] = [fn(int(i), W, H) for i in idx]
        print(f"{path_kind}: {len(idx)} frames {W}x{H} in {time.perf_counter() - t0:.1f}s", flush=True)
    ref = decode(a.ref, W, H, gray)
    rows = []
    for j, i in enumerate(idx):
        o, k = res["old"][j], res["kit"][j]
        r = ref[i] if gray else ref[i]
        rows.append(dict(frame=int(i), t=round(i / 25, 2), old_vs_kit=psnr(o, k), kit_vs_final=psnr(k, r),
                         old_vs_final=psnr(o, r), max_abs_old_kit=int(np.abs(o.astype(int) - k.astype(int)).max())))
    for r in rows:
        print(f"  f{r['frame']:3d} t={r['t']:5.2f}  old~kit {r['old_vs_kit']:6.1f} dB (max |d| {r['max_abs_old_kit']})"
              f"  kit~final {r['kit_vs_final']:5.1f} dB  old~final {r['old_vs_final']:5.1f} dB")
    fin = [r for r in rows if np.isfinite(r["old_vs_kit"])]
    summ = dict(piece=a.piece, ref=os.path.basename(a.ref), size=f"{W}x{H}", frames=[int(i) for i in idx],
                old_vs_kit_min=min([r["old_vs_kit"] for r in rows]),
                identical_frames=sum(1 for r in rows if r["max_abs_old_kit"] == 0),
                kit_vs_final_min=min(r["kit_vs_final"] for r in rows),
                kit_vs_final_mean=float(np.mean([r["kit_vs_final"] for r in rows])),
                old_vs_final_mean=float(np.mean([r["old_vs_final"] for r in rows])), rows=rows)
    del fin
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=1)


if __name__ == "__main__":
    main()
