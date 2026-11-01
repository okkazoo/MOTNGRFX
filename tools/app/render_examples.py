#!/usr/bin/env python
"""Re-render the example posters in app/examples/ through a graph (default fusion) on a running ComfyUI.

    python tools/app/render_examples.py --port 8190 [--only after-hours] [--graph app/graphs/fusion.json]

For every <id>.json: submits the example's inputs (bench_ws.py, per-node timings land in 16_app/bench/example_<id>.*),
keeps the previous poster as <id>_v1.mp4 / <id>_v1.jpg (once), writes the new <id>.mp4 + <id>.jpg (frame at 55 %).
Files are replaced through a temp name + rename (retry, then in-place copy: the Drive sync may hold a hard link).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EX = os.path.join(ROOT, "app", "examples")


def put(src: str, dst: str) -> None:
    """Replace dst by rename (new inode); Windows refuses while the Drive sync / a browser holds it open -> retry, then
    overwrite in place."""
    import time
    for _ in range(4):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(1.5)
    shutil.copyfile(src, dst)
    os.remove(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--graph", default=os.path.join(ROOT, "app", "graphs", "fusion.json"))
    ap.add_argument("--only")
    ap.add_argument("--output-dir", default=os.environ.get("COMFYUI_OUTPUT") or os.path.join(os.path.expanduser("~"), "Documents", "Dev", "ComfyUI", "output"))
    a = ap.parse_args()
    rc = 0
    for fn in sorted(os.listdir(EX)):
        if not fn.endswith(".json"):
            continue
        ex = json.load(open(os.path.join(EX, fn), encoding="utf-8"))
        if a.only and ex["id"] != a.only:
            continue
        inp = ex["inputs"]
        out = os.path.join(ROOT, "16_app", "bench", f"example_{ex['id']}")
        cmd = [sys.executable, os.path.join(ROOT, "tools", "app", "bench_ws.py"), a.graph, "--port", str(a.port), "--out", out]
        for k in ("words", "length", "seed", "quality", "scene", "music", "look"):
            if k in inp:
                cmd += ["--set", f"{k}={str(inp[k]).replace(chr(10), chr(92) + 'n')}"]
        for k in ("image", "audio"):
            if inp.get(k):
                cmd += [f"--{k}", os.path.join(EX, inp[k])]
        print(" ".join(cmd[:6]), "...", flush=True)
        if subprocess.call(cmd) != 0:
            rc = 1
            continue
        res = json.load(open(out + ".json", encoding="utf-8"))
        src = os.path.join(a.output_dir, *res["outputs"]["OUT final"][0].split("/"))
        base = os.path.join(EX, ex["poster"][:-4])
        for ext in (".mp4", ".jpg"):
            if os.path.exists(base + ext) and not os.path.exists(base + "_v1" + ext):
                shutil.copyfile(base + ext, base + "_v1" + ext)
        shutil.copyfile(src, base + ".tmp.mp4")
        put(base + ".tmp.mp4", base + ".mp4")
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                             "default=nw=1:nk=1", base + ".mp4"], text=True))
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-ss", f"{dur * 0.55:.2f}", "-i", base + ".mp4", "-frames:v", "1",
                               "-q:v", "4", base + ".tmp.jpg"])
        put(base + ".tmp.jpg", base + ".jpg")
        print(f"{ex['id']}: {base}.mp4 ({dur:.1f} s) + .jpg", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
