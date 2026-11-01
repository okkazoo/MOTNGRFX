#!/usr/bin/env python
"""FINAL ROUTE (user decision 2026-10-30): the finished piece is upscaled WHOLE (type included) to 1080x1920 with Real-ESRGAN, fps and audio carried through.

    GetVideoComponents -> UpscaleModelLoader + ImageUpscaleWithModel -> ImageScale (lanczos, exact size) -> CreateVideo (fps, audio) -> SaveVideo

All five nodes are ComfyUI core (no custom node, no restart). Model by source size (`pick_model`): RealESRGAN_x2plus.pth when 2x reaches the target (540x960 draft ->
1080x1920 exactly), RealESRGAN_x4plus.pth otherwise (360x640 preview: 4x = 1440x2560, then lanczos down to 1080x1920). 16:9 swaps the sides. A piece already at the final size
(quality final) is left alone.

    from tools.app.final_upscale import add_final_upscale
    add_final_upscale(graph, quality="draft", fmt="9:16")        # re-points `out_final` (SaveVideo) at the upscaled video; `assemble` is the MographAssemble node id

CLI (an existing finished mp4 on the box, as a standalone API graph to submit with progress_standin.py):
    python tools/app/final_upscale.py plan  --size 540x960 [--fmt 9:16]
    python tools/app/final_upscale.py graph --video "projects/.../x_video_00001_.mp4 [output]" --size 540x960 --prefix projects/.../x_up1080 --out g.json
    streaming equivalent without ComfyUI (low RAM, measured in BUILT.md Speeds): tools/app/upscale_finished.py run ...

Memory: ImageUpscaleWithModel returns the whole clip as one float32 IMAGE on the CPU (frames x H x W x 3 x 4 bytes: 384 frames at 1440x2560 = 17 GB before the lanczos step), so a
long piece wants RAM headroom; the GPU holds one tile (a few hundred MiB, measured 0.95 GB / 1.9 GB whole process with the streaming runner).
Real-ESRGAN: BSD-3-Clause (xinntao), weights from the official GitHub releases (card rows `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth`, models/upscale_models).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_X2 = "RealESRGAN_x2plus.pth"
MODEL_X4 = "RealESRGAN_x4plus.pth"
SCALE = {MODEL_X2: 2, MODEL_X4: 4}
FINAL_LONG = (1080, 1920)          # 9:16; 16:9 swaps the sides (plan_rules.sizes_for("final")["compose"])


def wants_upscale(piece: dict | None) -> bool:
    """does the graph's lazy save node run the upscale? Final only: not the still Preview, the app Final tier (the 540x960 draft sizes the x2 model is picked for), vertical 9:16
    (the chain's lanczos target is 1080x1920). The internal 360x640 `preview` tier is upscaled by hand (x4plus), never by the graph."""
    from tools.fusion import plan_rules as PR
    from tools.fusion import still_preview as SP
    if not piece or SP.is_still(piece) or str(piece.get("quality", "")) != SP.APP_FINAL_TIER:
        return False
    if (piece.get("request") or {}).get(SP.NO_UPSCALE):               # v20 Preview (quality draft540): the full render at the draft size, the upscale is a separate step on the finished clip
        return False
    return PR.format_of(piece) == "9:16"


def final_size(fmt: str = "9:16") -> tuple[int, int]:
    from tools.fusion import plan_rules as PR
    return tuple(PR.sizes_for("final", fmt)["compose"])


def source_size(quality: str, fmt: str = "9:16") -> tuple[int, int]:
    """compose size of a tier = the size of its finished video (preview 360x640, draft 540x960, final 1080x1920)."""
    from tools.fusion import plan_rules as PR
    return tuple(PR.sizes_for(quality, fmt)["compose"])


def pick_model(src: tuple[int, int], target: tuple[int, int]) -> str | None:
    """None = the source already is the target (or larger). x2plus when 2x covers the target, else x4plus."""
    need = max(target[0] / src[0], target[1] / src[1])
    if need <= 1.0:
        return None
    return MODEL_X2 if need <= 2.0 + 1e-9 else MODEL_X4


def fragment(video: list, src: tuple[int, int], target: tuple[int, int], prefix: str = "up", model: str | None = None) -> dict:
    """the nodes of the route, ids `<prefix>_*`; `video` = [node id, slot] of a VIDEO output. -> {id: node}, the last one is `<prefix>_vid` (a VIDEO). No SaveVideo (the caller owns it)."""
    model = model or pick_model(src, target)
    if model is None:
        raise ValueError(f"source {src} already reaches {target}: no upscale")
    p = prefix
    return {
        f"{p}_comp": {"class_type": "GetVideoComponents", "inputs": {"video": list(video)}, "_meta": {"title": "UP split video (frames, audio, fps)"}},
        f"{p}_model": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}, "_meta": {"title": f"UP model {model}"}},
        f"{p}_run": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": [f"{p}_model", 0], "image": [f"{p}_comp", 0]},
                     "_meta": {"title": f"UP Real-ESRGAN x{SCALE[model]}"}},
        f"{p}_fit": {"class_type": "ImageScale", "inputs": {"image": [f"{p}_run", 0], "upscale_method": "lanczos", "width": int(target[0]), "height": int(target[1]), "crop": "disabled"},
                     "_meta": {"title": f"UP lanczos to {target[0]}x{target[1]}"}},
        f"{p}_vid": {"class_type": "CreateVideo", "inputs": {"images": [f"{p}_fit", 0], "fps": [f"{p}_comp", 2], "audio": [f"{p}_comp", 1]},
                     "_meta": {"title": "UP video (same fps, same audio)"}},
    }


def add_final_upscale(graph: dict, quality: str = "draft", fmt: str = "9:16", video: str = "assemble", save: str = "out_final", prefix: str = "up",
                      keep_draft: bool = False, model: str | None = None, src: tuple[int, int] | None = None) -> dict:
    """hook the route after the compose / encode output of an app graph, in place: the node `save` (SaveVideo) takes the upscaled video instead of `video` (a MographAssemble).
    keep_draft=True also keeps a second SaveVideo `<save>_draft` with the original draft video. quality final (already 1080 high) -> unchanged. -> graph."""
    if video not in graph or save not in graph:
        raise KeyError(f"final upscale: graph has no node {video!r} / {save!r}")
    s = tuple(src) if src else source_size(quality, fmt)
    t = final_size(fmt)
    m = model or pick_model(s, t)
    if m is None:
        return graph
    old = list(graph[save]["inputs"]["video"])
    graph.update(fragment(old, s, t, prefix, m))
    if keep_draft:
        graph[f"{save}_draft"] = json.loads(json.dumps(graph[save]))
        a = graph[f"{save}_draft"]
        a["inputs"]["filename_prefix"] = str(a["inputs"].get("filename_prefix", "video/draft")) + "_draft"
        a["_meta"] = {"title": "OUT draft (before the upscale)"}
    graph[save]["inputs"]["video"] = [f"{prefix}_vid", 0]
    return graph


def standalone_graph(video_file: str, src: tuple[int, int], fmt: str = "9:16", prefix: str = "projects/up1080", model: str | None = None) -> dict:
    """an API graph that upscales an existing finished mp4: core LoadVideo (`file` = name in the input dir, or "<relative path> [output]" for a file in the output dir)."""
    g = {"load": {"class_type": "LoadVideo", "inputs": {"file": video_file}, "_meta": {"title": "IN finished video"}}}
    g.update(fragment(["load", 0], tuple(src), final_size(fmt), "up", model))
    g["out_final"] = {"class_type": "SaveVideo", "inputs": {"video": ["up_vid", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}, "_meta": {"title": "OUT upscaled"}}
    return g


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--size", required=True)
    p.add_argument("--fmt", default="9:16")
    p = sub.add_parser("graph")
    p.add_argument("--video", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--fmt", default="9:16")
    p.add_argument("--prefix", default="projects/up1080")
    p.add_argument("--model")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        s, t = _size(a.size), final_size(a.fmt)
        m = pick_model(s, t)
        print(json.dumps({"source": s, "target": t, "model": m, "scale": SCALE.get(m), "model_output": [s[0] * SCALE[m], s[1] * SCALE[m]] if m else None}))
        return 0
    g = standalone_graph(a.video, _size(a.size), a.fmt, a.prefix, a.model)
    json.dump(g, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"{len(g)} nodes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
