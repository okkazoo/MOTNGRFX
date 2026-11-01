"""CPU previews of the masked Motion-Track probe: footage + latent-mask cells + the track video the IC guide will see.

    python tools/fusion/mask_preview.py            # all styles -> 16_app/mask_ltx/previews/

Per style: preview_<style>.mp4 (540x960, 24 fps): original footage, tinted 32 px latent cells that get regenerated
(latent frame = frame_to_latent(pixel frame); soft cells fade), white matte contour, and the LTXVDrawTracks-style dots
(ltx_ic.render_tracks, the exact guide image) painted over. contact_sheet.png: one row per style, 6 frames.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import latent_mask as lm  # noqa: E402
import ltx_ic  # noqa: E402
import ltx_mask  # noqa: E402
import mask_tracks  # noqa: E402

PW, PH = 540, 960                       # preview size (cells are 30 px)
FP_KW = dict(dilate_cells=2, feather=0.75, keep_first_frame=True, main_subject_only=True)   # = build_graph defaults
FP_KW_CORRIDOR = dict(FP_KW, dilate_cells=1)                                                  # = the *_corridor graphs


def read_frames(path: str, size=(PW, PH)) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def write_mp4(frames: np.ndarray, path: str, fps: int = 24) -> None:
    h, w = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed")


def style_frames(foot: np.ndarray, matte: np.ndarray, fp: np.ndarray, guide: np.ndarray, style: str, f0: int) -> np.ndarray:
    import cv2
    F = len(foot)
    t_lat, h, w = fp.shape
    cell_w, cell_h = PW // w, PH // h
    out = np.empty_like(foot)
    for f in range(F):
        img = foot[f].astype(np.float32)
        k = int(lm.frame_to_latent(f))
        cells = fp[k]
        up = np.kron(cells, np.ones((cell_h, cell_w), np.float32))[:PH, :PW, None]
        tint = np.array([255.0, 190.0, 60.0], np.float32)          # BGR: cyan-ish
        img = img * (1 - 0.32 * up) + tint * 0.32 * up
        on = cells >= 0.999
        grid = np.zeros((PH, PW), np.float32)
        for y, x in zip(*np.nonzero(on)):
            grid[y * cell_h, x * cell_w:(x + 1) * cell_w] = 1
            grid[(y + 1) * cell_h - 1, x * cell_w:(x + 1) * cell_w] = 1
            grid[y * cell_h:(y + 1) * cell_h, x * cell_w] = 1
            grid[y * cell_h:(y + 1) * cell_h, (x + 1) * cell_w - 1] = 1
        img = img * (1 - 0.35 * grid[..., None]) + 255.0 * 0.35 * grid[..., None]
        m = cv2.resize((matte[f] > 0.5).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = np.ascontiguousarray(img)
        cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)
        g = guide[f].astype(np.float32)
        gm = (g.max(axis=2, keepdims=True) > 20).astype(np.float32)
        img = img * (1 - 0.92 * gm) + g * 0.92 * gm
        img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        state = "following subject" if f < f0 else "moving"
        cv2.putText(img, f"{style}  f{f:03d}  latent {k:02d}  {state}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, f"{style}  f{f:03d}  latent {k:02d}  {state}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        out[f] = img
    return out


def contact_sheet(rows: dict[str, np.ndarray], f0: int, path: str, tile=(270, 480), n_cols: int = 6) -> None:
    import cv2
    F = len(next(iter(rows.values())))
    picks = sorted({0, f0, *(int(round(f0 + (F - 1 - f0) * q)) for q in (0.25, 0.5, 0.75)), F - 1})[:n_cols]
    lab_h = 22
    sheet = np.zeros((len(rows) * (tile[1] + lab_h), len(picks) * tile[0], 3), np.uint8)
    for r, (style, fr) in enumerate(rows.items()):
        for c, f in enumerate(picks):
            t = cv2.resize(fr[f], tile, interpolation=cv2.INTER_AREA)
            y0 = r * (tile[1] + lab_h)
            sheet[y0:y0 + lab_h, c * tile[0]:(c + 1) * tile[0]] = 30
            cv2.putText(sheet, f"{style} f{f} ({f / 24:.2f}s)", (c * tile[0] + 6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            sheet[y0 + lab_h:y0 + lab_h + tile[1], c * tile[0]:(c + 1) * tile[0]] = t
    cv2.imwrite(path, sheet)


def main() -> None:
    import cv2
    pack = ltx_mask.PACK
    out_dir = os.path.join(pack, "previews")
    os.makedirs(out_dir, exist_ok=True)
    foot = read_frames(os.path.join(pack, "refs", ltx_mask.FOOTAGE_NAME))
    matte = mask_tracks.load_matte_seq(ltx_mask.MATTE_DIR, ltx_mask.STAGE1)
    t_lat = lm.latent_frames(len(matte))
    fp = lm.make_footprint(matte, t_lat, 32, 18, **FP_KW)
    rows, rows_c = {}, {}
    for style in ltx_mask.GRID_STYLES:
        tr = mask_tracks.from_json_tracks(json.load(open(os.path.join(pack, "tracks", f"tracks_{style}.json"))))
        meta = json.load(open(os.path.join(pack, "tracks", f"tracks_{style}.meta.json")))
        f0 = meta["start_frame"]
        guide = ltx_ic.render_tracks(mask_tracks.to_json_tracks(tr), *ltx_mask.STAGE1)
        guide = np.stack([cv2.resize(g[..., ::-1], (PW, PH), interpolation=cv2.INTER_AREA) for g in guide])  # RGB -> BGR
        fr = style_frames(foot, matte, fp, guide, style, f0)
        write_mp4(fr, os.path.join(out_dir, f"preview_{style}.mp4"))
        rows[style] = fr
        cor = mask_tracks.corridor_matte(tr, ltx_mask.STAGE1, ltx_mask.CORRIDOR_RADIUS, ltx_mask.CORRIDOR_TRAIL)
        fpc = lm.make_footprint(matte, t_lat, 32, 18, extra=cor, **FP_KW_CORRIDOR)
        frc = style_frames(foot, matte, fpc, guide, style + " +corridor", f0)
        write_mp4(frc, os.path.join(out_dir, f"preview_{style}_corridor.mp4"))
        rows_c[style + " +corridor"] = frc
        print(style, "done")
    contact_sheet(rows, f0, os.path.join(out_dir, "contact_sheet.png"))
    contact_sheet(rows_c, f0, os.path.join(out_dir, "contact_sheet_corridor.png"))
    with open(os.path.join(out_dir, "cells_frame_ascii.txt"), "w") as f:
        for k in (0, 1, 5, 10, 15):
            f.write(f"latent frame {k} (32 rows x 18 cols; # regenerate, + soft, . keep) - alpha only, dilate 2, feather 0.75\n"
                    f"{lm.cells_to_text(fp, k)}\n\n")
    print("previews ->", out_dir)


if __name__ == "__main__":
    main()
