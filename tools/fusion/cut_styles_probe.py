#!/usr/bin/env python3
"""cut_styles_probe - render the 3D shot cuts side by side on two real shots of an ingredients folder and write a sheet + mp4s + a timing json.

Default: 22_batch-24gb v010 r01 s1 -> s2 (clean clips, Alpha-Gen mattes, depth clips), the four kinds pivot / push / pullback / fly at 360 x 640, each in its own
default style (pivot chunks, the rest smooth), plus `--old` (the same push / pullback / fly as chunks, the QC look) for a comparison row.

    python tools/fusion/cut_styles_probe.py                                   # 22_batch-24gb/v010 r01 s1 -> s2, short 360
    python tools/fusion/cut_styles_probe.py --ingredients <dir> --a s1 --b s2 --short 360 --kinds depth_pivot,orbit:push --old --slide

It calls exactly what the node MographPivotCut calls (pivot_cut.apply_cut: the clip slices, the matte put on the clip's own frame, the depth first frame), so the
frames and the matte through the transition are the ones a piece would get. Output (never overwritten: a new numbered folder under --out):
    <out>/cuts_sheet.png     one row per kind: 10 frames across the transition, a second thin row under each = the matte through the transition
    <out>/<name>.mp4         the transition frames (A's window then B's head) and <name>_matte.mp4 (gray)
    <out>/probe.json         kinds, styles, frames, i2, seconds, frames per second, host
CPU (numpy / cv2): run it on the GPU box (all frame rendering runs there); frames per second are the CPU speed of whatever machine runs it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import matte_align as MA  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402

DEFAULT_ING = os.path.join(ROOT, "22_batch-24gb", "v010", "ingredients", "motion-graphic_batch-24gb_v010_r01")
DEFAULT_KINDS = ("depth_pivot", "orbit:push", "orbit:pullback", "orbit:fly")


def read_clip(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr)


def find(ing, shot, what):
    hit = sorted(glob.glob(os.path.join(ing, f"*_{shot}_{what}")))
    if not hit:
        raise FileNotFoundError(f"{ing}: no *_{shot}_{what}")
    return hit[0]


def read_matte(folder, hw):
    """Alpha-Gen PNG stack (a cover-resized crop of the clip) -> float32 (T,H,W) on the clip's own frame (matte_align.ref_to_full)."""
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    out = []
    for p in pngs:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        out.append(MA.ref_to_full(m, hw))
    return np.stack(out).astype(np.float32)


def read_depth(path, hw):
    """first frame of a depth clip (bright = near) -> float32 (1,H,W) on the clip's frame."""
    cap = cv2.VideoCapture(path)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return MA.ref_to_full(g, hw)[None].astype(np.float32)


def unique_dir(base):
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    n = 2
    while os.path.exists(f"{base}_g{n:02d}"):
        n += 1
    os.makedirs(f"{base}_g{n:02d}")
    return f"{base}_g{n:02d}"


def write_mp4(path, frames, fps=24, gray=False):
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    pr.stdin.write(np.ascontiguousarray(frames).tobytes())
    pr.stdin.close()
    pr.wait()


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run(args):
    ing = os.path.abspath(args.ingredients)
    ca, cb = read_clip(find(ing, args.a, "clean.mp4")), read_clip(find(ing, args.b, "clean.mp4"))
    hw_a, hw_b = ca.shape[1:3], cb.shape[1:3]
    ma, mb = read_matte(find(ing, args.a, "matte"), hw_a), read_matte(find(ing, args.b, "matte"), hw_b)
    try:
        da, db = read_depth(find(ing, args.a, "depth.mp4"), hw_a), read_depth(find(ing, args.b, "depth.mp4"), hw_b)
    except FileNotFoundError:
        da = db = None                                              # the engine estimates depth on the CPU (Depth-Anything-V2-Small)
    piece = {"fps": 24, "shots": [{"id": args.a, "frames": len(ca)}, {"id": args.b, "frames": len(cb)}], "compose": {"crossfade": 0, "cuts": []}}
    kinds = [(k, None) for k in args.kinds]
    if args.old:
        kinds += [(k, "chunks") for k in args.kinds if k.startswith("orbit:")]
    if args.slide:
        kinds += [("orbit:push", "slide")]
    out = unique_dir(args.out or os.path.join(ing, "..", "..", "review", "cut_probe"))
    rows, timing = [], []
    for kind, style in kinds:
        par = {"short": args.short}
        if style:
            par["style"] = style
        piece["compose"]["cuts"] = [{"after": args.a, "cut": kind, "params": par}]
        spec = PC.specs(piece)[0]
        t0 = time.time()
        res = PC.apply_cut(piece, spec, {args.a: ca, args.b: cb}, mattes={args.a: ma, args.b: mb}, depths={args.a: da, args.b: db})
        dt = time.time() - t0
        pl = res.plan
        a0, i2, bl = pl["a0"], pl["i2"], pl["b_len"]
        frames = np.concatenate([res.a[a0:a0 + i2], res.b[:bl]])
        mat = np.concatenate([res.matte_a[1], res.matte_b[1]])
        name = f"{kind.replace(':', '-')}_{pl['params']['style']}"
        write_mp4(os.path.join(out, name + ".mp4"), frames)
        write_mp4(os.path.join(out, name + "_matte.mp4"), (np.clip(mat, 0, 1) * 255).astype(np.uint8), gray=True)
        idx = np.linspace(0, len(frames) - 1, args.cols).round().astype(int)
        th = args.tile
        tw = int(round(th * frames.shape[2] / frames.shape[1]))
        top = np.concatenate([cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA) for i in idx], 1)
        bot = np.concatenate([cv2.resize((np.clip(mat[i], 0, 1) * 255).astype(np.uint8), (tw, th // 3), interpolation=cv2.INTER_AREA) for i in idx], 1)
        rows.append(np.concatenate([label(top, f"{kind} / {pl['params']['style']}  ({len(frames)} frames, B from {i2})"), np.repeat(bot[..., None], 3, -1)], 0))
        timing.append(dict(kind=kind, style=pl["params"]["style"], frames=int(pl["n"]), i2=int(i2), seconds=round(dt, 1), frames_per_s=round(pl["n"] / dt, 3), short=args.short,
                           matte_pass=bool(pl["matte_pass"])))
        print(f"{kind:16s} {pl['params']['style']:7s} {pl['n']} frames in {dt:6.1f} s = {pl['n'] / dt:.2f} frames/s", flush=True)
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(os.path.join(out, "cuts_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    with open(os.path.join(out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ingredients=ing, a=args.a, b=args.b, host=platform.node(), cpu_count=os.cpu_count(), runs=timing), f, indent=1)
    print("wrote", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingredients", default=DEFAULT_ING)
    ap.add_argument("--a", default="s1")
    ap.add_argument("--b", default="s2")
    ap.add_argument("--short", type=int, default=360, help="short side of the render (360 -> 360 x 640)")
    ap.add_argument("--kinds", type=lambda s: [x for x in s.split(",") if x], default=list(DEFAULT_KINDS))
    ap.add_argument("--old", action="store_true", help="also render the orbit kinds as chunks (the style of the QC sheets) for comparison")
    ap.add_argument("--slide", action="store_true", help="also render a push as the slide style")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--tile", type=int, default=320, help="tile height in the sheet")
    ap.add_argument("--out", default="")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
