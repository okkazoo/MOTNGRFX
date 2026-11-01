#!/usr/bin/env python
"""VARIANT 0 (2026-10-30): learned upscale of a WHOLE FINISHED VIDEO (type included) to 1080x1920, audio copied, same fps and frame count. RUNS ON THE BOX (ComfyUI venv python).

    python upscale_finished.py run   --src X.mp4 --model RealESRGAN_x2plus.pth --out OUT.mp4 --lanczos-out BASE.mp4 --report R.json [--min-free-mib 6144]
    python upscale_finished.py sheet --a BASE.mp4 --b OUT.mp4 --frame N --crop x,y,w,h [--crop ...] --label "..." --out SHEET.png
    python upscale_finished.py frames --src X.mp4 --frames 40,200 --out DIR           # overview PNGs to pick crops by eye

The model (spandrel, fp16, tiles of 256 with 16 px overlap: a few hundred MiB) runs on the model's scale; when the result is not 1080x1920 it is resized with lanczos to exactly
1080x1920 (x4plus on 360x640 gives 1440x2560 -> down). It waits until nvidia-smi shows >= --min-free-mib free before loading (another render may hold the card) and, every 24 frames,
yields while less than 1.5 GiB is free. The baseline is the same encoder fed by ffmpeg lanczos only. Report: seconds, frames/s, torch + nvidia-smi (this process) VRAM peak.
Not a project design tool: no design change, only pixels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time

import numpy as np

W, H = 1080, 1920
ENC = ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]


def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                                   "-of", "json", path], capture_output=True, text=True, check=True).stdout)["streams"][0]
    return int(j["width"]), int(j["height"]), j["r_frame_rate"], int(j["nb_read_frames"])


def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0])


class Sampler(threading.Thread):
    """peak nvidia-smi memory of THIS process (compute-apps list)."""
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = pid, 0, False

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                for ln in o.strip().splitlines():
                    p, m = [x.strip() for x in ln.split(",")]
                    if int(p) == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(0.5)


def run(a):
    import os
    import cv2
    import torch
    from spandrel import ModelLoader
    sw, sh, fps, n = probe(a.src)
    rep = {"src": a.src, "model": os.path.basename(a.model), "src_size": [sw, sh], "fps": fps, "frames": n, "target": [W, H]}
    t_wait = time.time()
    while free_mib() < a.min_free_mib:
        time.sleep(5)
    rep["waited_for_vram_s"] = round(time.time() - t_wait, 1)
    smp = Sampler(os.getpid())
    smp.start()
    m = ModelLoader().load_from_file(a.model).cuda().eval().half()
    s = int(m.scale)
    rep["model_scale"] = s
    torch.cuda.reset_peak_memory_stats()

    def up(bgr):
        x = torch.from_numpy(bgr[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].half().div(255.0)
        _, c, h, w = x.shape
        out = torch.zeros((1, c, h * s, w * s), dtype=x.dtype, device=x.device)
        T, P = 256, 16
        for y0 in range(0, h, T):
            for x0 in range(0, w, T):
                y1, x1 = min(y0 + T, h), min(x0 + T, w)
                ya, xa, yb, xb = max(y0 - P, 0), max(x0 - P, 0), min(y1 + P, h), min(x1 + P, w)
                with torch.no_grad():
                    o = m(x[:, :, ya:yb, xa:xb])
                oy, ox = (y0 - ya) * s, (x0 - xa) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        y = out.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]
        y = np.ascontiguousarray(y)
        if y.shape[1] != W or y.shape[0] != H:
            y = cv2.resize(y, (W, H), interpolation=cv2.INTER_LANCZOS4)
        return y

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", a.src, "-map", "0:v", "-map", "1:a:0?",
                            *ENC, "-c:a", "copy", "-r", fps, "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    t0 = time.time()
    i = 0
    fb = sw * sh * 3
    while True:
        buf = dec.stdout.read(fb)
        if len(buf) < fb:
            break
        if i % 24 == 0 and free_mib() < 1500:
            torch.cuda.empty_cache()
            while free_mib() < 4000:
                time.sleep(3)
        enc.stdin.write(up(np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)).tobytes())
        i += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    rep["upscale_s"] = round(time.time() - t0, 2)
    rep["upscale_fps"] = round(i / rep["upscale_s"], 2)
    rep["frames_out"] = i
    rep["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    rep["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    smp.stop = True
    rep["nvidia_smi_peak_mib_this_process"] = smp.peak
    t0 = time.time()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-vf", f"scale={W}:{H}:flags=lanczos", "-map", "0:v", "-map", "0:a:0?", *ENC, "-c:a", "copy", "-movflags", "+faststart", a.lanczos_out],
                   check=True)
    rep["lanczos_s"] = round(time.time() - t0, 2)
    for k, p in (("out_probe", a.out), ("lanczos_probe", a.lanczos_out)):
        ow, oh, ofps, on = probe(p)
        rep[k] = {"size": [ow, oh], "fps": ofps, "frames": on}
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def read_frame(path, idx):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"{path}: no frame {idx}")
    return f


def frames(a):
    import os
    import cv2
    os.makedirs(a.out, exist_ok=True)
    for i in [int(x) for x in a.frames.split(",")]:
        cv2.imwrite(os.path.join(a.out, f"f{i:04d}.png"), read_frame(a.src, i))


def sheet(a):
    import cv2
    rows = []
    for c in a.crop:                                              # "x,y,w,h" (frame --frame) or "FRAME:x,y,w,h"
        fr, _, box = c.rpartition(":")
        fr = int(fr) if fr else a.frame
        A, B = read_frame(a.a, fr), read_frame(a.b, fr)
        x, y, w, h = [int(v) for v in box.split(",")]
        ca, cb = A[y:y + h, x:x + w].copy(), B[y:y + h, x:x + w].copy()
        for im, t in ((ca, "lanczos"), (cb, a.label)):
            cv2.rectangle(im, (0, 0), (len(t) * 9 + 8, 18), (0, 0, 0), -1)
            cv2.putText(im, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([ca, np.full((h, 4, 3), 255, np.uint8), cb]))
    mw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, mw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows))


def _label(im, t):
    import cv2
    cv2.rectangle(im, (0, 0), (len(t) * 9 + 8, 18), (0, 0, 0), -1)
    cv2.putText(im, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def sheetn(a):
    """rows = crops "FRAME:x,y,w,h", columns = the given videos (label=path), all 1:1."""
    import cv2
    vids = [v.split("=", 1) for v in a.v]
    rows = []
    for c in a.crop:
        fr, _, box = c.rpartition(":")
        fr = int(fr) if fr else a.frame
        x, y, w, h = [int(t) for t in box.split(",")]
        cells = []
        for lab, path in vids:
            im = read_frame(path, fr)[y:y + h, x:x + w].copy()
            _label(im, lab)
            cells += [im, np.full((h, 4, 3), 255, np.uint8)]
        rows.append(np.hstack(cells[:-1]))
    mw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, mw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows))


def strip(a):
    """rows = videos (label=path), columns = consecutive frames f0..f0+n-1, one crop each (1:1); prints the mean |frame-to-frame| change inside the crop per video."""
    import cv2
    x, y, w, h = [int(t) for t in a.crop.split(",")]
    rows = []
    for lab, path in [v.split("=", 1) for v in a.v]:
        cells, prev, d = [], None, []
        for i in range(a.f0, a.f0 + a.n):
            im = read_frame(path, i)[y:y + h, x:x + w].copy()
            if prev is not None:
                d.append(float(np.abs(im.astype(np.int16) - prev.astype(np.int16)).mean()))
            prev = im.copy()
            _label(im, f"{lab} f{i}")
            cells += [im, np.full((h, 3, 3), 255, np.uint8)]
        print(f"{lab}: mean abs change between consecutive frames in the crop = {np.mean(d):.3f} (per step {[round(v, 2) for v in d]})")
        rows.append(np.hstack(cells[:-1]))
    cv2.imwrite(a.out, np.vstack(rows))


def thumbs(a):
    """rows = videos, columns = frames, whole frame scaled to --h px high."""
    import cv2
    rows = []
    for lab, path in [v.split("=", 1) for v in a.v]:
        cells = []
        for i in [int(t) for t in a.frames.split(",")]:
            im = read_frame(path, i)
            im = cv2.resize(im, (int(im.shape[1] * a.h / im.shape[0]), a.h), interpolation=cv2.INTER_AREA)
            _label(im, f"{lab} f{i}")
            cells += [im, np.full((a.h, 3, 3), 255, np.uint8)]
        rows.append(np.hstack(cells[:-1]))
    mw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, mw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--src", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lanczos-out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--min-free-mib", type=int, default=6144)
    p = sub.add_parser("frames")
    p.add_argument("--src", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--crop", action="append", required=True)
    p.add_argument("--label", default="learned")
    p.add_argument("--out", required=True)
    p = sub.add_parser("sheetn")
    p.add_argument("--v", action="append", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--crop", action="append", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("strip")
    p.add_argument("--v", action="append", required=True)
    p.add_argument("--f0", type=int, required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--crop", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("thumbs")
    p.add_argument("--v", action="append", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "frames": frames, "sheet": sheet, "sheetn": sheetn, "strip": strip, "thumbs": thumbs}[a.cmd](a)


if __name__ == "__main__":
    main()
