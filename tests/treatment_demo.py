"""Treatment contact sheet + timings.

    python tests/treatment_demo.py            # grid -> tests/out/treatments_grid.png (+ _full crops)
    python tests/treatment_demo.py --time     # also time every treatment at 360x640 and 1080x1920

Rows: every treatment / preset. Columns: subject_texture.png and blob_gradient.png.
treatments_grid.png processes each tile at its display size (draft-like scale);
treatments_grid_full.png shows 1:1 crops processed at full scale (scale 1.0)."""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mograph.render import treatments as T  # noqa: E402
from mograph.render.text import FONTS  # noqa: E402

OUT = os.path.join(HERE, "out")
FIX = os.path.join(HERE, "fixtures")
PALETTE = ["#111111", "#f2f2f2", "#ff4d2e", "#2e6bff", "#ffd23f"]

CASES = [
    ("original", None),
    ("halftone dot", {"type": "halftone", "cell": 18, "angle": 45}),
    ("halftone line", {"type": "halftone", "cell": 16, "angle": 30, "shape": "line", "ink": "palette:3"}),
    ("screenprint 2 inks", {"type": "screenprint", "inks": ["palette:0", "palette:2"], "misregister": 4, "grit": 0.4}),
    ("screenprint 3 inks", {"type": "screenprint", "inks": ["palette:0", "palette:2", "#e8dcc3"],
                            "misregister": 4, "grit": 0.4}),
    ("screenprint 4 ordered", {"type": "screenprint", "inks": ["#1b1d3a", "palette:3", "palette:2", "palette:4"],
                               "misregister": 3, "grit": 0.3, "dither": "ordered"}),
    ("screenprint diffusion", {"type": "screenprint", "inks": ["palette:0", "palette:2"], "dither": "diffusion",
                               "misregister": 3, "grit": 0.3}),
    ("xerox", {"type": "xerox"}),
    ("xerox heavy", {"type": "xerox", "threshold": 0.45, "contrast": 1.0, "toner_grit": 0.8, "crushed_blacks": 0.7}),
    ("duotone", {"type": "duotone", "colors": ["palette:0", "palette:3"]}),
    ("tritone", {"type": "tritone", "colors": ["#14112b", "palette:2", "#f6e7c8"]}),
    ("film teal_orange", {"type": "film_grade", "preset": "teal_orange", "halation": 0.6}),
    ("film warm", {"type": "film_grade", "preset": "warm"}),
    ("film cool", {"type": "film_grade", "preset": "cool"}),
    ("film mono", {"type": "film_grade", "preset": "mono", "crushed_blacks": 0.3}),
    ("posterize_flat", {"type": "posterize_flat", "smooth": 0.5}),
    ("posterize_flat 3", {"type": "posterize_flat", "palette": ["#1d2b3a", "#f4efe4", "palette:2"], "smooth": 0.7}),
    ("paper", {"type": "paper", "fibre": 0.8, "speckle": 0.6}),
    ("grain", {"type": "grain", "amount": 0.5}),
    ("vignette", {"type": "vignette", "amount": 0.6}),
    ("light_leak", {"type": "light_leak", "strength": 0.8}),
    ("dust_scratches", {"type": "dust_scratches", "amount": 1.0}),
    ("chain: grade+leak+dust", [{"type": "film_grade", "preset": "warm"}, {"type": "light_leak", "strength": 0.6},
                                {"type": "dust_scratches", "amount": 0.8}, {"type": "grain", "amount": 0.3}]),
]


def load(name):
    return np.asarray(Image.open(os.path.join(FIX, name)).convert("RGB"), np.float32) / 255.0


def _font(px):
    try:
        return ImageFont.truetype(FONTS["Space Grotesk"], px)
    except Exception:
        return ImageFont.load_default()


def grid(images, tile_hw, full=False, path="treatments_grid.png"):
    th, tw = tile_hw
    lab_w = 210
    rows = []
    font = _font(17)
    for label, spec in CASES:
        tiles = []
        for img in images:
            if full:     # 1:1 crop of a full-scale (scale 1.0) render
                y0, x0 = (img.shape[0] - th) // 2, (img.shape[1] - tw) // 2
                src = img[y0:y0 + th, x0:x0 + tw]
                out = T.apply_treatment(src, spec, PALETTE, frame=12, seed=7, scale=1.0)
            else:
                src = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                out = T.apply_treatment(src, spec, PALETTE, frame=12, seed=7)
            assert out.shape == src.shape and out.dtype == np.float32
            tiles.append(np.clip(out, 0, 1))
        row = np.concatenate([np.full((th, lab_w, 3), 0.12, np.float32)] + tiles, 1)
        pil = Image.fromarray((row * 255 + 0.5).astype(np.uint8))
        ImageDraw.Draw(pil).text((12, th // 2 - 10), label, fill=(235, 235, 235), font=font)
        rows.append(np.asarray(pil))
        rows.append(np.full((4, row.shape[1], 3), 20, np.uint8))
    sheet = np.concatenate(rows, 0)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, path)
    Image.fromarray(sheet).save(p)
    print("wrote", p, sheet.shape)
    return p


TIMED = [
    ("halftone", {"type": "halftone", "cell": 12}),
    ("screenprint (3 inks)", {"type": "screenprint", "inks": ["palette:0", "palette:2", "#e8dcc3"], "misregister": 3}),
    ("screenprint (4 inks)", {"type": "screenprint", "inks": ["palette:0", "palette:3", "palette:2", "palette:4"]}),
    ("screenprint diffusion", {"type": "screenprint", "inks": ["palette:0", "palette:2"], "dither": "diffusion"}),
    ("xerox", {"type": "xerox"}),
    ("duotone", {"type": "duotone", "colors": ["palette:0", "palette:3"]}),
    ("tritone", {"type": "tritone"}),
    ("film_grade teal_orange", {"type": "film_grade", "preset": "teal_orange"}),
    ("film_grade mono", {"type": "film_grade", "preset": "mono"}),
    ("posterize_flat", {"type": "posterize_flat"}),
    ("paper", {"type": "paper"}),
    ("grain", {"type": "grain", "amount": 0.2}),
    ("vignette", {"type": "vignette", "amount": 0.3}),
    ("light_leak", {"type": "light_leak", "strength": 0.6}),
    ("dust_scratches", {"type": "dust_scratches", "amount": 0.6}),
]


def time_one(spec, img, n):
    T.apply_treatment(img, spec, PALETTE, frame=0, seed=3)       # warm caches
    ts = []
    for k in range(n):
        t = time.perf_counter()
        T.apply_treatment(img, spec, PALETTE, frame=k, seed=3)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def timings():
    src = load("subject_texture.png")
    imgs = {sz: cv2.resize(src, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) for sz in [(640, 360), (1920, 1080)]}
    print(f"{'treatment':26s} {'360x640 ms':>11s} {'1080x1920 ms':>13s}")
    for name, spec in TIMED:
        a = time_one(spec, imgs[(640, 360)], 30)
        b = time_one(spec, imgs[(1920, 1080)], 10)
        print(f"{name:26s} {a:11.2f} {b:13.2f}")


if __name__ == "__main__":
    ims = [load("subject_texture.png"), load("blob_gradient.png")]
    grid(ims, (288, 384))
    grid(ims, (288, 384), full=True, path="treatments_grid_full.png")
    if "--time" in sys.argv:
        timings()
