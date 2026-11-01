"""Render the showcase plan (3 scenes, every feature) to tests/out/render_demo_{9x16,16x9}.mp4
at draft quality and print timings.  Run from the repo root:  python tests/render_demo.py"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mograph.encode import encode_mp4  # noqa: E402
from mograph.geometry import frame_size  # noqa: E402
from mograph.render import render_frames  # noqa: E402
from render_helpers import make_analysis, showcase_plan  # noqa: E402


def click_track(plan, sr=48000):
    n = int(round(plan["duration"] * sr))
    out = np.zeros((2, n), np.float32)
    t = np.arange(int(0.08 * sr)) / sr
    kick = (np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 40)) * t) * np.exp(-t * 25)).astype(np.float32)
    for b in plan["audio"]["beats"]:
        i = int(b * sr)
        seg = kick[: max(0, min(len(kick), n - i))]
        out[:, i:i + len(seg)] += seg * (0.8 if b in plan["audio"]["downbeats"] else 0.45)
    return out


def main():
    quality = sys.argv[1] if len(sys.argv) > 1 else "draft"
    plan = showcase_plan()
    analysis = make_analysis()
    audio = click_track(plan)
    out_dir = os.path.join(ROOT, "tests", "out")
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ("9:16", "16:9"):
        W, H = frame_size(fmt, quality)
        t = time.perf_counter()
        n = 0
        for _ in render_frames(plan, analysis, fmt, quality):
            n += 1
        t_render = time.perf_counter() - t
        path = os.path.join(out_dir, f"render_demo_{fmt.replace(':', 'x')}.mp4")
        t = time.perf_counter()
        encode_mp4(render_frames(plan, analysis, fmt, quality), plan["fps"], W, H, audio, 48000, path)
        t_total = time.perf_counter() - t
        print(f"{fmt} {quality} {W}x{H}: {n} frames, render only {t_render:.2f}s "
              f"({1000 * t_render / n:.1f} ms/frame), render+encode {t_total:.2f}s -> {path}")


if __name__ == "__main__":
    main()
