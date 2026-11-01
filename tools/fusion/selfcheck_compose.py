"""Visual self-check for compose.py: 3 s 576x1024 synthetic demo (moving gradient footage, a soft radial 'mist'
matte, a big letter 'A' from MoiraiOne), every op run, contact sheet -> 15_fusion/v001/review/compose_selfcheck.png.
Also renders a recipe through compose_shot and checks the mp4. Run: python tools/fusion/selfcheck_compose.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H, N, FPS = 576, 1024, 72, 24
FONT = ROOT / "mograph" / "fonts" / "MoiraiOne-Regular.ttf"
OUT_SHEET = ROOT / "15_fusion" / "v001" / "review" / "compose_selfcheck.png"
WORK = ROOT / "15_fusion" / "v001" / "local" / "compose_selfcheck"


def letter_alpha(ch="A", size=W, h=H, font_px=None, stroke=34):
    font = ImageFont.truetype(str(FONT), font_px or int(W * 1.1))
    im = Image.new("L", (size, h), 0)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=font)
    # Moirai One's A is a hairline glyph (~2% coverage): thicken with a stroke so the demo letter is a solid shape
    d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), ch, fill=255, font=font,
           stroke_width=stroke, stroke_fill=255)
    return np.asarray(im, np.float32) / 255.0


def make_inputs(n=N):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    A = letter_alpha()
    foot, depth, matte, tya, tydepth, disc_m = [], [], [], [], [], []
    for i in range(n):
        t = i / FPS
        ph = t * 0.35
        r = 0.5 + 0.5 * np.sin(2 * np.pi * (ys / H * 0.7 + xs / W * 0.3 + ph))
        g = 0.5 + 0.5 * np.sin(2 * np.pi * (ys / H * 0.7 + xs / W * 0.3 + ph + 0.33))
        b = 0.5 + 0.5 * np.sin(2 * np.pi * (ys / H * 0.7 + xs / W * 0.3 + ph + 0.66))
        f = np.stack([0.08 + 0.4 * r, 0.06 + 0.36 * g, 0.12 + 0.45 * b], -1)
        band = np.clip((np.sin(2 * np.pi * ((xs * 0.6 + ys * 0.8) / 330.0 - t * 0.4)) - 0.75) * 25.0, 0, 1)
        f = f * (1 - 0.55 * band[..., None])  # dark contour bands so luma edges exist everywhere
        cx, cy, rad = W * (0.15 + 0.7 * (i / n)), H * 0.58, 80
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        disc = np.clip((rad - dist) / 2.5 + 0.5, 0, 1)
        shade = 0.8 + 0.2 * np.clip(1 - dist / rad, 0, 1)
        f = f * (1 - disc[..., None]) + disc[..., None] * (np.array([1.0, 0.78, 0.4]) * shade[..., None])
        foot.append(f.astype(np.float32))
        dp = 0.2 + 0.15 * (1 - ys / H)
        depth.append((dp * (1 - disc) + 0.9 * disc).astype(np.float32))
        disc_m.append(disc.astype(np.float32))
        mx, my = W * (0.7 - 0.4 * i / n), H * 0.45
        mist = 0.9 * np.exp(-(((xs - mx) / (W * 0.45)) ** 2 + ((ys - my) / (H * 0.2)) ** 2))
        matte.append(mist.astype(np.float32))
        s = 1.0 + 0.04 * np.sin(2 * np.pi * t / 2.0)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        a = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR)
        tya.append(a)
        tydepth.append(cv2.GaussianBlur(a, (0, 0), 4))
    return dict(footage=foot, depth=depth, matte=matte, type_a=tya, type_depth=tydepth, disc=disc_m)


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 13], fill=(0, 0, 0))
    d.text((4, 1), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    d = make_inputs()
    cols = [0, 24, 48, N - 1]
    CELL = (144, 256)
    cream = (0.96, 0.93, 0.85)
    rows = []

    def thumb(a):
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        return C._to_u8(cv2.resize(a, CELL, interpolation=cv2.INTER_AREA))

    def row(name, fn):
        cells = [thumb(fn(i)) for i in cols]
        cells[0] = label(cells[0], name)
        rows.append(np.concatenate(cells, 1))

    pal = C.palette_from_footage(d["footage"][::6], k=5)
    print("palette", pal.round(2).tolist())

    def sw(i, **k):
        return C.sandwich(d["footage"][i], cream, d["type_a"][i], d["matte"][i] * 1.0, **k)

    # edges are stateful: precompute a smoothed run
    tr = C.EdgeTracer()
    edges = [tr.step(f) for f in d["footage"]]

    row("footage", lambda i: d["footage"][i])
    row("matte (mist)", lambda i: d["matte"][i])
    row("sandwich", lambda i: sw(i))
    row("sandwich g.6", lambda i: sw(i, matte_gamma=0.6, matte_gain=1.2))
    row("sandwich invert", lambda i: sw(i, invert=True))
    row("depth_occlude .5", lambda i: C.depth_occlude(d["footage"][i], d["depth"][i], cream, d["type_a"][i], 0.5, 0.1))
    row("depth_occlude .95", lambda i: C.depth_occlude(d["footage"][i], d["depth"][i], cream, d["type_a"][i], 0.95, 0.1))
    row("knockout", lambda i: C.knockout((0.05, 0.05, 0.07), d["type_a"][i], d["footage"][i]))
    row("knockout+fg", lambda i: C.knockout((0.05, 0.05, 0.07), d["type_a"][i], d["footage"][i], d["disc"][i]))
    row("tint sample", lambda i: C.over(d["footage"][i], C.tint_type(d["type_a"][i], d["footage"][i], "sample"), d["type_a"][i]))
    row("tint complement", lambda i: C.over(d["footage"][i], C.tint_type(d["type_a"][i], d["footage"][i], "complement"), d["type_a"][i]))
    row("tint gradmap", lambda i: C.over(d["footage"][i], C.tint_type(d["type_a"][i], d["footage"][i], "gradient_map", palette=pal), d["type_a"][i]))
    row("edges", lambda i: edges[i])
    row("edges_in_type", lambda i: C.over(d["footage"][i] * 0.5, (1, 1, 1), C.edges_in_type(edges[i], d["type_a"][i])))
    row("type_from_edges", lambda i: C.over(d["footage"][i] * 0.35, cream, C.type_from_edges(edges[i], d["type_a"][i], grow=14, line_width=1.5, fill=0.12)))
    row("refract", lambda i: C.refract(d["footage"][i], d["type_depth"][i], strength=18, blur=7, chroma=0.2, rim=0.7))
    row("refract+body", lambda i: C.over(
        C.refract(d["footage"][i], d["type_depth"][i], strength=18, blur=7, chroma=0.2, rim=0.7),
        cream, d["type_a"][i] * 0.2))

    def finish(i):
        x = sw(i)
        x = C.add_glow(x, d["type_a"][i], radius=18, intensity=0.3, color=(1.0, 0.8, 0.5))
        x = C.halation(x, 0.6, 20, 0.5)
        x = C.grade(x, lift=0.03, gamma=1.1, gain=1.05, sat=1.15)
        x = C.grain(x, i, seed=3, amount=0.04)
        return C.vignette(x, 0.5)

    row("finish chain", finish)

    # upscale_matte: truth = disc (hard-ish); low-res = 1/4 area -> cubic vs guided
    def up(i):
        truth = d["disc"][i]
        small = cv2.resize(truth, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        cub = C.upscale_matte(small, (W, H))
        gui = C.upscale_matte(small, (W, H), guide=d["footage"][i])
        return np.concatenate([cub, gui, np.abs(gui - truth) * 4, np.abs(cub - truth) * 4], 1)

    cells = [thumb(cv2.resize(up(cols[1]), (W * 4, H)))]
    cells[0] = cv2.resize(cells[0], (CELL[0] * 4, CELL[1]))
    cells[0] = label(cells[0], "upscale_matte: cubic | guided | err guided x4 | err cubic x4")
    rows.append(cells[0])

    f1 = 24
    a = d["footage"][f1]
    b = C.tint_type(d["type_a"][f1], a, "complement")
    bl = [C.screen(a, b), C.multiply(a, b), C.overlay(a, b), C.soft_light(a, b)]
    cells = [thumb(x) for x in bl]
    cells[0] = label(cells[0], "screen|mul|overlay|softlight")
    rows.append(np.concatenate(cells, 1))
    lk = [C.luminosity_key(d["footage"][f1], 0.3, 0.7), C.luminosity_key(d["footage"][f1], 0.3, 0.7, invert=True)]
    xf = [C.crossfade(d["footage"][0], d["footage"][N - 1], 0.5), C.crossfade(d["footage"][0], sw(N - 1), 0.5)]
    cells = [thumb(x) for x in lk + xf]
    cells[0] = label(cells[0], "lumkey|inv|xfade|xfade")
    rows.append(np.concatenate(cells, 1))

    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    pad = np.zeros_like(rows[0])
    right = right + [pad] * (half - len(right))
    sheet = np.concatenate([np.concatenate(left, 0), np.concatenate(right, 0)], 1)
    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    C.imwrite(OUT_SHEET, sheet.astype(np.float32) / 255.0)
    print("sheet", OUT_SHEET, sheet.shape)

    # ---- recipe path: write inputs as dirs, render a shot through compose_shot
    for k in ("footage", "depth", "matte", "type_a", "type_depth"):
        C.write_png_sequence(d[k], WORK / "in" / k)
    C.write_png_sequence([np.broadcast_to(np.array(cream, np.float32), (H, W, 3))] * 1, WORK / "in" / "type_rgb")
    recipe = {
        "name": "demo", "size": [W, H], "fps": FPS, "palette": {"source": "footage", "k": 5},
        "inputs": {
            "footage": str(WORK / "in" / "footage"), "matte": str(WORK / "in" / "matte"),
            "type_a": str(WORK / "in" / "type_a"), "type_depth": str(WORK / "in" / "type_depth"),
            "type_rgb": str(WORK / "in" / "type_rgb"),
        },
        "ops": [
            {"op": "refract", "out": "plate", "args": {"footage": "$footage", "type_depth_or_matte": "$type_depth",
                                                      "strength": {"anim": [[0, 4], [1, 20]], "ease": "smooth"}}},
            {"op": "tint_type", "out": "tint", "args": {"type_a": "$type_a", "footage": "$footage", "mode": "gradient_map", "palette": "$palette"}},
            {"op": "sandwich", "out": "image", "args": {"footage": "$plate", "type_rgb": "$tint", "type_a": "$type_a", "matte": "$matte", "matte_gamma": 0.8}},
            {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 1, "amount": 0.03}},
        ],
    }
    (WORK / "recipe.json").parent.mkdir(parents=True, exist_ok=True)
    (WORK / "recipe.json").write_text(json.dumps(recipe, indent=1))
    r = C.compose_shot(recipe, WORK / "shot", base_dir=WORK)
    print("shot", json.dumps(r))


if __name__ == "__main__":
    main()
