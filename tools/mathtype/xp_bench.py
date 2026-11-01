#!/usr/bin/env python
"""Verify and time the torch backend of silkwind.py against the numpy reference.

For each mode (plain 10 s loop / beaty audio / matte wipe / ethereal audio), setting (preview: 360x450, N/4
strands, 1x; full: 1080x1350, all strands, 2x supersampled = the --final settings) and time t, it renders the
frame with numpy and with torch, reports max / mean absolute pixel difference and PSNR, saves a side-by-side
(numpy | torch | |diff| x16) jpg and a full-res |diff| x16 png to 11_mathtype/v006/review/, and times numpy
per frame vs torch per frame (single frame, and batched over --frame-batch frames).

    python tools/mathtype/xp_bench.py                                   # torch-cpu vs numpy, all checks
    python tools/mathtype/xp_bench.py --device cuda --frame-batch 8     # same on a GPU box
    python tools/mathtype/xp_bench.py --device cuda --skip-numpy --throughput 50 --settings full
Review files get a run number xpv_rNN (never overwritten); metrics also go to xpv_rNN_metrics.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import silkwind as S  # noqa: E402
import xp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
V4 = os.path.join(REPO, "11_mathtype", "v004", "local")
REVIEW = os.path.join(REPO, "11_mathtype", "v006", "review")
AUDIO = {"beaty": ("beat_D_silkwind_beaty_g01.wav", "beat_D_silkwind_beaty_g01.json"),
         "ethereal": ("chords_D_silkwind_ethereal_g01.wav", "chords_D_silkwind_ethereal_g01.json")}
# the D_wipe_beaty g06 matte settings (v005/local/D_wipe_beaty_g06_matte.json)
WIPE = dict(WIPE_AT=9.6, WIPE_DUR=1.2, WIPE_GROW=30.0, WIPE_BASE=3.0, WIPE_ZN=2.6, WIPE_FAR=0.1, WIPE_SOFT=0.45,
            WIPE_LIGHT=0.8, WIPE_BREATH=0.85)
SETTINGS = {"preview": (360, 450, 1, True), "full": (1080, 1350, 2, False)}


def setup(mode: str) -> None:
    """Module state as silkwind.main() leaves it for that mode."""
    S.MOD, S.T, S.NFR, S.MATTE, S.WIPE_AT = None, 10.0, 250, False, None
    if mode == "plain":
        return
    wav, ev = AUDIO["ethereal" if mode == "ethereal" else "beaty"]
    S.configure(os.path.join(V4, wav), os.path.join(V4, ev), "ethereal" if mode == "ethereal" else "beaty",
                mode == "matte")
    if mode == "matte":
        for k, v in WIPE.items():
            setattr(S, k, v)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = float(np.mean((a.astype(np.float64) - b) ** 2))
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def run_tag() -> str:
    os.makedirs(REVIEW, exist_ok=True)
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(REVIEW, "xpv_r*"))
            for m in [re.match(r"xpv_r(\d+)_", os.path.basename(f))] if m]
    return f"xpv_r{(max(nums) + 1 if nums else 1):02d}"


def save_pair(tag, mode, setting, t, a, b):
    d = np.abs(a.astype(np.int16) - b).astype(np.uint8)
    amp = np.clip(d.astype(np.int32) * 16, 0, 255).astype(np.uint8)
    pw = 540 if a.shape[1] > 540 else a.shape[1]
    ph = int(round(a.shape[0] * pw / a.shape[1]))
    small = [cv2.resize(x, (pw, ph), interpolation=cv2.INTER_AREA) for x in (a, b)]
    small.append(cv2.resize(amp, (pw, ph), interpolation=cv2.INTER_AREA if pw < a.shape[1] else cv2.INTER_NEAREST))
    sbs = np.concatenate(small, 1)
    for i, lab in enumerate(("numpy", "torch", "|diff| x16")):
        cv2.putText(sbs, lab, (i * pw + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 88, 42), 1, cv2.LINE_AA)
    stem = f"{tag}_{mode}_{setting}_t{t:04.1f}"
    cv2.imwrite(os.path.join(REVIEW, stem + "_sbs.jpg"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(REVIEW, stem + "_diff.png"), cv2.cvtColor(amp, cv2.COLOR_RGB2BGR))
    return [stem + "_sbs.jpg", stem + "_diff.png"]


def read_frames(path, idx, W, H, grey=False):
    """{n: frame} for the frame indices idx (RGB order as matte_wipe reads them: cv2 BGR, used as is)."""
    cap, out, n = cv2.VideoCapture(path), {}, 0
    while n <= max(idx):
        ok, f = cap.read()
        if not ok:
            break
        if n in idx:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            out[n] = f[..., 0] if grey else f
        n += 1
    return out


def bench_wipe(a, be, tag, rows, timing):
    """matte_wipe.composite numpy vs torch on the D_wipe_beaty g06 inputs, at 360x450 and 1080x1350."""
    import types
    import matte_wipe as MW
    v5 = os.path.join(REPO, "11_mathtype", "v005", "local")
    with open(os.path.join(v5, "D_wipe_beaty_g06_preview.json"), encoding="utf-8") as f:
        p = json.load(f)
    mp = p.pop("matte_params")
    args = types.SimpleNamespace(**p)
    idx = [int(round(t * MW.FPS)) for t in a.times]
    K = MW.opacity(np.arange(MW.NFR) / MW.FPS, args.opacity_curve, args.k_pre, mp["wipe_at"], mp["wipe_dur"])
    for setting, (W, H) in (("preview", (360, 450)), ("full", (1080, 1350))):
        if setting not in a.settings:
            continue
        A = read_frames(os.path.join(v5, "D_wipe_sideA_g01.mp4"), idx, W, H)
        B = read_frames(os.path.join(v5, "D_wipe_sideB_g01.mp4"), idx, W, H)
        M = read_frames(os.path.join(v5, "D_wipe_beaty_g06_matte.mp4"), idx, W, H, grey=True)
        tn, tt = [], []
        for t, n in zip(a.times, idx):
            MW.BE = None
            t0 = time.perf_counter()
            fn = MW.composite(A, B, M, n, K[n], args)
            tn.append(time.perf_counter() - t0)
            MW.BE = be
            MW.composite(A, B, M, n, K[n], args)               # warm-up
            be.sync()
            t0 = time.perf_counter()
            ft = MW.composite(A, B, M, n, K[n], args)
            be.sync()
            tt.append(time.perf_counter() - t0)
            d = np.abs(fn.astype(np.int16) - ft)
            r = dict(mode="wipe", setting=setting, t=t, max=int(d.max()), mean=round(float(d.mean()), 5),
                     psnr=round(psnr(fn, ft), 2), px_gt8=int((d.max(-1) > 8).sum()))
            if tag:
                r["files"] = save_pair(tag, "wipe", setting, t, fn[..., ::-1].copy(), ft[..., ::-1].copy())
            rows.append(r)
            print(f"wipe     {setting:7s} t={t:4.1f}  max {r['max']:3d}  mean {r['mean']:.4f}  "
                  f"PSNR {r['psnr']:6.2f} dB  px>8 {r['px_gt8']}")
        tm = dict(mode="wipe", setting=setting, numpy_s=round(float(np.mean(tn)), 4),
                  torch_single_s=round(float(np.mean(tt)), 4))
        timing.append(tm)
        print("  timing", tm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["plain", "beaty", "matte", "wipe"],
                    choices=["plain", "beaty", "matte", "ethereal", "wipe"],
                    help="wipe = matte_wipe.py compositor (numpy vs torch) on the D_wipe_beaty g06 inputs")
    ap.add_argument("--settings", nargs="*", default=["preview", "full"], choices=list(SETTINGS))
    ap.add_argument("--times", type=float, nargs="*", default=[0.0, 2.5, 6.0, 9.6])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frame-batch", type=int, default=4)
    ap.add_argument("--skip-numpy", action="store_true", help="timing only (no numpy reference, no PSNR)")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--throughput", type=int, default=0,
                    help="also time N consecutive frames (25 fps grid from t=0) batched, per mode / setting")
    a = ap.parse_args()
    be = xp.Backend("torch", a.device)
    S.FRAME_BATCH = a.frame_batch
    tag = None if (a.skip_numpy or a.no_images) else run_tag()
    rows = []
    out = {"device": a.device, "frame_batch": a.frame_batch, "torch": xp.torch().__version__,
                     "threads": xp.torch().get_num_threads(), "rows": rows, "timing": []}
    if a.device.startswith("cuda"):
        out["gpu"] = xp.torch().cuda.get_device_name(0)
    for mode in a.modes:
        if mode == "wipe":
            bench_wipe(a, be, tag, rows, out["timing"])
            continue
        setup(mode)
        for setting in a.settings:
            W, H, ss, prev = SETTINGS[setting]
            sn, st = S.Silk(prev, xp.Backend()), S.Silk(prev, be)
            st.frame(a.times[0], W, H, ss)                     # warm-up (allocator, kernels)
            be.sync()
            tn, tt = [], []
            for t in a.times:
                if not a.skip_numpy:
                    t0 = time.perf_counter()
                    fn = sn.frame(t, W, H, ss)
                    tn.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                ft = st.frame(t, W, H, ss)
                be.sync()
                tt.append(time.perf_counter() - t0)
                if a.skip_numpy:
                    continue
                d = np.abs(fn.astype(np.int16) - ft)
                r = dict(mode=mode, setting=setting, t=t, max=int(d.max()), mean=round(float(d.mean()), 5),
                         psnr=round(psnr(fn, ft), 2), px_gt8=int((d.max(-1) > 8).sum()))
                if tag:
                    r["files"] = save_pair(tag, mode, setting, t, fn, ft)
                rows.append(r)
                print(f"{mode:8s} {setting:7s} t={t:4.1f}  max {r['max']:3d}  mean {r['mean']:.4f}  "
                      f"PSNR {r['psnr']:6.2f} dB  px>8 {r['px_gt8']}")
            t0 = time.perf_counter()
            st.frames(a.times, W, H, ss)
            be.sync()
            tb = (time.perf_counter() - t0) / len(a.times)
            tm = dict(mode=mode, setting=setting, numpy_s=round(float(np.mean(tn)), 3) if tn else None,
                      torch_single_s=round(float(np.mean(tt)), 3), torch_batched_s=round(tb, 3))
            if a.throughput:
                ts = [n / S.FPS for n in range(a.throughput)]
                t0 = time.perf_counter()
                st.frames(ts, W, H, ss)
                be.sync()
                tm["throughput_s_per_frame"] = round((time.perf_counter() - t0) / a.throughput, 3)
            out["timing"].append(tm)
            print("  timing", tm)
    if tag:
        with open(os.path.join(REVIEW, f"{tag}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("review", REVIEW, tag)
    if rows:
        print("worst PSNR", min(r["psnr"] for r in rows), "dB")


if __name__ == "__main__":
    main()
