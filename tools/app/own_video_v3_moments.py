#!/usr/bin/env python
"""(runs ON THE BOX, ComfyUI venv python, CPU numpy / cv2) splice CONTRAST MOMENTS into the composed base video and mux the ORIGINAL audio at the SOURCE frame rate.

    /venv/main/bin/python own_video_v3_moments.py --base base.mp4 --work W --alpha-dir D --type-dir T --global-clip G.mp4 --piece piece.json \
        --moments '[{"a":..,"b":..,"mode":"plate_in_type"}]' --src /workspace/own_video/your_clip.mp4 --fps 30 --out out.mp4 --report out.json [--plate-layers plate_layers.py]

base.mp4 = the redesign graph's assemble output: the 720 source frames on the stretched 24 fps clock (nothing resampled). Here the frames are only RELABELLED (encoded at --fps with the original
audio), and in every moment the picture is replaced by tools/fusion/plate_layers.bake(mode=plate_in_subject | plate_in_type): flat palette ground + the clean plate (Clean-Plate IC-LoRA) in the
subject's shape / in the letters, the subject's matte following its own clip frame for frame (same clip, same matte, same transform = identity). The letters are the type engine's own matte pass of
the base run (<type-dir>/matte/matte_NNNNN.png, index = frame). A moment is [a, b) in source frames: fully contrast on [a, b), an 8-frame iris opens before a and closes after b (the visible
transition); every frame of [a-8, b+8) must lie inside ONE real-cut piece with a matte and a plate (checked).
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--alpha-dir", required=True)
ap.add_argument("--stem", default="motion-graphic_own-video_v003")
ap.add_argument("--type-dir", required=True)
ap.add_argument("--global-clip", required=True)
ap.add_argument("--piece", required=True)
ap.add_argument("--moments", default="[]")
ap.add_argument("--src", required=True)
ap.add_argument("--fps", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", required=True)
ap.add_argument("--plate-layers", default="")
ap.add_argument("--iris", type=int, default=8)
a = ap.parse_args()

if a.plate_layers:
    spec = importlib.util.spec_from_file_location("plate_layers_overlay", a.plate_layers)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(a.plate_layers)))))     # <tree>/tools/fusion/plate_layers.py -> <tree>
    PLT = importlib.util.module_from_spec(spec)
    sys.modules["plate_layers_overlay"] = PLT
    spec.loader.exec_module(PLT)
else:
    sys.path.insert(0, "/workspace/ComfyUI/custom_nodes/motion-graphic")
    from tools.fusion import plate_layers as PLT

segs = json.load(open(os.path.join(a.work, "segments.json")))
piece = json.load(open(a.piece))
pal = piece["type"].get("palette") or {}
flat = (pal.get("accents") or [None])[0]
word = pal.get("fg")
moments = json.loads(a.moments)
OUT_W, OUT_H = 360, 640


def read_all(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def cover_fit(img):
    """384x704 layer -> the 360x640 compose frame: x0.9375 then a centred 10 px crop top and bottom (plan_rules.sizes_for('preview'): every layer is cover-fitted identically)."""
    r = cv2.resize(img, (OUT_W, int(round(img.shape[0] * OUT_W / img.shape[1]))), interpolation=cv2.INTER_AREA)
    y0 = (r.shape[0] - OUT_H) // 2
    return r[y0:y0 + OUT_H]


def piece_of(j):
    for s in segs:
        if s["a"] <= j < s["b"]:
            return s
    return None


def iris(p, shape):
    """soft circular reveal from the centre: p 0..1 -> float mask HxW (1 = the contrast picture)."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    dm = float(np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2))
    r = p * 1.15 * dm
    soft = 0.10 * dm
    return np.clip((r - d) / soft + 0.5, 0.0, 1.0)


base = read_all(a.base)
glob_clip = read_all(a.global_clip)
n = len(base)
assert n == len(glob_clip), (n, len(glob_clip))
H0, W0 = glob_clip[0].shape[:2]
final = list(base)
report = {"frames": n, "moments": []}
cache = {}


def read_png(path):
    if path not in cache:
        cache[path] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cache[path]


for m in moments:
    A, B, mode = int(m["a"]), int(m["b"]), m["mode"]
    F0, F1 = A - a.iris, B + a.iris
    idx = list(range(F0, F1))
    pcs = [piece_of(j) for j in idx]
    bad = [j for j, s in zip(idx, pcs) if s is None or not s["person"]]
    if bad:
        raise SystemExit(f"moment {m}: frames {bad[:5]}.. are not inside a piece with the man (matte + plate): move it")
    clip = np.stack([cv2.cvtColor(glob_clip[j], cv2.COLOR_BGR2RGB) for j in idx])
    matte = np.stack([read_png(f"{a.alpha_dir}/{a.stem}_alpha_h{s['job']}_image_{j - s['a'] + 1:05d}_.png") for j, s in zip(idx, pcs)])
    matte = matte[..., ::-1].copy() if matte.ndim == 4 else matte
    plate = np.stack([cv2.cvtColor(read_png(f"{a.alpha_dir}/{a.stem}_plate_p{s['job']}_image_{j - s['a'] + 1:05d}_.png"), cv2.COLOR_BGR2RGB) for j, s in zip(idx, pcs)])
    files = sorted(glob.glob(os.path.join(a.type_dir, "matte", "*.png")))
    soft = PLT.LETTER_SOFT

    def letters(k, F0=F0):
        im = cv2.imread(files[min(max(F0 + int(k), 0), len(files) - 1)], cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = im[..., 3] if im.shape[2] == 4 and im[..., 3].max() > 0 else im[..., :3].mean(axis=2)
        L = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
        if L.shape != (H0, W0):
            L = cv2.resize(L, (W0, H0), interpolation=cv2.INTER_AREA)
        return np.clip(cv2.GaussianBlur(L, (0, 0), soft * min(H0, W0)), 0.0, 1.0)

    ident = dict(move={"dx": 0.0, "dy": 0.0, "scale": 1.0}, pan={"dx": 0.0, "dy": 0.0, "scale": 1.0})           # nothing moves: matte, clip and plate stay registered
    cols = [c for c in [pal.get("bg"), pal.get("fg")] + list(pal.get("accents") or []) if c]
    wl = lambda c: sum(w_ * (((v / 255.0) / 12.92) if v / 255.0 <= 0.04045 else (((v / 255.0) + 0.055) / 1.055) ** 2.4) for w_, v in zip((0.2126, 0.7152, 0.0722), c))      # noqa: E731  WCAG luminance
    wc = lambda x, y: (max(wl(x), wl(y)) + 0.05) / (min(wl(x), wl(y)) + 0.05)                                                                                              # noqa: E731
    wcol = (piece["type"]["words"][m["k"]].get("palette") or {}).get("fg") or word        # the word's own ink (picked from the picture, see own_video_v2_retime_box --colors)
    if mode in ("window_subject_front", "text_as_mask"):  # same rule as plate_layers.bake_clip: the darkest (shot key flat = "light": the lightest) palette colour
        fl = (max if m.get("flat") == "light" else min)(cols, key=wl) if cols else flat
    else:                                               # flat ground: the palette colour that reads best against the word's ink
        fl = max(cols, key=lambda c: wc(c, wcol)) if cols else flat
    r = PLT.bake(plate, clip, matte, mode=mode, letters=letters, flat_rgb=fl, word_rgb=wcol, guard=bool(m.get("guard", True)), edge_line=bool(m.get("edge_line", True)),
                 outline=False, mask_fill=str(m.get("fill", "footage")), frame0=F0, fill_source=str(m.get("fill_source", "auto")), **ident)
    for k, j in enumerate(idx):
        pic = cv2.cvtColor(r["footage"][k], cv2.COLOR_RGB2BGR)
        pic = cover_fit(pic).astype(np.float32)
        if j < A:
            w = iris((j - F0 + 1) / float(a.iris), pic.shape)
        elif j >= B:
            w = iris(1.0 - (j - B + 1) / float(a.iris), pic.shape)
        else:
            w = np.ones(pic.shape[:2], np.float32)
        w = w[..., None]
        final[j] = np.clip(np.rint(final[j].astype(np.float32) * (1.0 - w) + pic * w), 0, 255).astype(np.uint8)
    report["moments"].append(dict(m, frames=len(idx), pieces=sorted({s["job"] for s in pcs}), ground=[int(v) for v in fl], word_ink=[int(v) for v in wcol], color=r["info"]["color"], occlusion=r["info"].get("occlusion"), fill_source=(r["info"].get("fill") or {}).get("fill_source"), fill_measured=r["info"].get("fill"), subject_strength=r["info"].get("subject_strength"), fill_tint=r["info"].get("fill_tint")))
    del clip, matte, plate, r

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", f"{a.fps:g}", "-i", "-", "-i", a.src, "-map", "0:v", "-map", "1:a",
       "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", a.out]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for f in final:
    p.stdin.write(np.ascontiguousarray(f).tobytes())
p.stdin.close()
p.wait()
json.dump(report, open(a.report, "w"), indent=1)
print("wrote", a.out, report)
