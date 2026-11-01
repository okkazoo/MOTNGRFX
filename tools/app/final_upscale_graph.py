#!/usr/bin/env python
"""API graph of "Final (1080)" (v20, user 2026-10-31: "final this clip and it upscales the smaller version"): the side graph `_final_upscale` the page queues under a finished Preview.

    IN video (core LoadVideo, `file` = "<subfolder>/<name>.mp4 [output]" = the finished clip in the output folder)
      -> GetVideoComponents -> UpscaleModelLoader + ImageUpscaleWithModel (Real-ESRGAN x2plus for 540x960, x4plus for 360x640) -> ImageScale lanczos to the exact 1080x1920 (16:9: 1920x1080)
      -> CreateVideo (same fps, same audio) -> SaveVideo "OUT upscaled"

Nothing is re-rendered: the whole finished video (type included) is upscaled. The chain is `tools/app/final_upscale.fragment` (the one the app graphs' lazy Final save also uses); all nodes are
ComfyUI core (no custom node, no restart). The template on disk is the 9:16 540x960 form; the page patches model / target for another source size or format with the SAME rule as
`plan()` (web_app/bind.js `finalUpscalePlan`, checked against this file in the page check).

    python tools/app/final_upscale_graph.py --write-template        # app/graphs/_final_upscale.json (a leading `_` keeps it out of the template dropdown)
    python tools/app/final_upscale_graph.py plan --size 540x960 [--fmt 9:16]
    python tools/app/final_upscale_graph.py --video "motngrfx_app/video/x_00001_.mp4 [output]" --size 540x960 --prefix motngrfx_app/video/x_up1080 --out g.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TEMPLATE = os.path.join(ROOT, "app", "graphs", "_final_upscale.json")

from tools.app import final_upscale as FU  # noqa: E402

IN_TITLE = "IN video"
OUT_TITLE = "OUT upscaled"
SUFFIX = "_up1080"                       # the new file's name = <original stem> + SUFFIX (next to the original, never over it)
DEFAULT_SRC = (540, 960)                 # the draft tier of the app's Preview (compose size)


def plan(src: tuple[int, int], fmt: str = "9:16") -> dict:
    """what the upscale does for a source of `src` (w, h): target size, model, scale; model None = nothing to do (the clip already is that size or bigger)."""
    t = FU.final_size(fmt)
    m = FU.pick_model(tuple(src), t)
    return {"source": list(src), "target": list(t), "model": m, "scale": FU.SCALE.get(m)}


def build(video_file: str = "", src: tuple[int, int] = DEFAULT_SRC, fmt: str = "9:16", prefix: str = "video/motngrfx" + SUFFIX) -> dict:
    """-> API-format graph (dict). `video_file` = '<subfolder>/<name> [output]' (the page overwrites IN video.inputs.file)."""
    g = FU.standalone_graph(video_file, tuple(src), fmt, prefix)
    g["load"]["_meta"] = {"title": IN_TITLE}
    g["up_comp"]["_meta"]["title"] = "Split video (frames, audio, fps)"
    g["out_final"]["_meta"] = {"title": OUT_TITLE}
    return g


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["plan"])
    ap.add_argument("--video", default="")
    ap.add_argument("--size", default="540x960")
    ap.add_argument("--fmt", default="9:16")
    ap.add_argument("--prefix", default="video/motngrfx" + SUFFIX)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-template", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan(_size(a.size), a.fmt)))
        return 0
    out = TEMPLATE if a.write_template else a.out
    if not out:
        ap.error("give --out or --write-template")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(a.video, _size(a.size), a.fmt, a.prefix), f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
