#!/usr/bin/env python
"""1080x1920 FINAL route probe (2026-10-30): the SAME piece JSON of a finished draft, footage layers upscaled, type + compose rendered NATIVELY at the final size.

    python tools/app/final_route_probe.py piece  --src PIECE.json --out PIECE_final.json     # LOCAL: quality final, sizes = plan_rules.sizes_for("final"), literal compose sizes
    python tools/app/final_route_probe.py prep   --variant A|B --shots DIR --arch DIR --piece PIECE_final.json [--model PATH]     # ON THE BOX (ffmpeg / cv2 / spandrel)
    python tools/app/final_route_probe.py graph  --src r03_graph.json --arch DIR --source-video MP4 --piece PIECE_final.json --out GRAPH.json [--prefix P]      # LOCAL

Why not upscale the finished video: a generative upscale would redraw the type. Only the FOOTAGE layers (clean clip, text-rendered background clip, matte) are upscaled;
the type passes and compose run at 1088x1920 / 1080x1920 from the piece JSON (same seeds, words, timings, cuts).

What an archive has (a draft run): sK_clean.mp4 / sK_textbg.mp4 at the COMPOSE size 540x960 (MographDecodeClip resizes the 576x1024 LTX frames to the compose size and
only that is archived, h264), sK_matte/NNNNN.png at the Alpha-Gen size 576x1024, so the footage is upscaled x2 from 540x960 to the final compose size 1080x1920 and the
matte x2 from 576x1024 to 1152x2048 (the same 1.0667 ratio to the clip as in the draft; compose's `upscale_matte` stretches it to the compose size).

prep variants (clips and mattes):
    A   clips: ffmpeg lanczos; mattes: bilinear x2, then cv2.ximgproc guided filter against the upscaled clean clip (r 8 at 2048 high, eps 2e-4: the engine's own constants)
    B   clips: Real-ESRGAN x2plus (spandrel, fp16, whole frame, tiles on OOM); mattes as in A
graph: the v018 r03 app graph cut down to what runs after the footage: archive loader (request, music, clean clips, mattes, audio) -> piece from JSON ->
    text-layer clips (VHS_LoadVideoPath -> MographClipFromImages) -> layer gather -> depth (Video-Depth-Anything, same chains) -> footage flow -> pivot cuts -> type passes ->
    compose -> assemble -> archive. No ACE / MOSS / LTX / Alpha-Gen / LLM. Nodes are the live ones on the box (ComfyUI is not restarted).
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_MATTE = (1152, 2048)       # Alpha-Gen size x2 (draft matte 576x1024)
FINAL_CLIP = (1080, 1920)        # compose size of the final tier


# ---------------------------------------------------------------------------------------------- piece
def _fit_sizes(obj, size):
    """literal 2-number `size` lists (upscale_matte) -> the compose size (same rule as plan_rules._fit_sizes)."""
    if isinstance(obj, dict):
        return {k: (list(size) if k == "size" and isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else _fit_sizes(v, size)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fit_sizes(v, size) for v in obj]
    return obj


def final_piece(piece: dict) -> dict:
    """the draft piece with quality final + the final sizes. Everything else (words, timings, seeds, cuts, ops) is untouched."""
    sys.path.insert(0, ROOT)
    from tools.fusion import plan_rules
    p = copy.deepcopy(piece)
    fmt = plan_rules.format_of(p)
    p["quality"] = "final"
    p["sizes"] = plan_rules.sizes_for("final", fmt)
    size = list(p["sizes"]["compose"])
    for key in ("compose", "compose.ops"):
        if key in p:
            p[key] = _fit_sizes(p[key], size)
    if isinstance(p.get("compose"), dict):
        p["compose"]["size"] = size
    return p


# ---------------------------------------------------------------------------------------------- prep (box)
def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def _read_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


def _encoder(path, w, h, fps=24):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-crf", "6", "-preset", "veryfast", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)


def upscale_lanczos(src, dst):
    _run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale={FINAL_CLIP[0]}:{FINAL_CLIP[1]}:flags=lanczos", "-c:v", "libx264", "-crf", "6", "-preset", "veryfast",
          "-pix_fmt", "yuv420p", dst])


class Esrgan:
    def __init__(self, path):
        import torch
        from spandrel import ModelLoader
        self.torch = torch
        self.m = ModelLoader().load_from_file(path).cuda().eval().half()
        self.scale = int(self.m.scale)
        self.tiled = False

    def _whole(self, x):
        with self.torch.no_grad():
            return self.m(x)

    def _tiled(self, x, tile=256, pad=16):
        torch = self.torch
        _, c, h, w = x.shape
        s = self.scale
        out = torch.zeros((1, c, h * s, w * s), dtype=x.dtype, device=x.device)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                ya, xa, yb, xb = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, h), min(x1 + pad, w)
                with torch.no_grad():
                    o = self.m(x[:, :, ya:yb, xa:xb])
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, (y0 - ya) * s:(y0 - ya) * s + (y1 - y0) * s, (x0 - xa) * s:(x0 - xa) * s + (x1 - x0) * s]
        return out

    def __call__(self, bgr):
        torch = self.torch
        x = torch.from_numpy(bgr[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].half().div(255.0)
        if self.tiled:
            y = self._tiled(x)
        else:
            try:
                y = self._whole(x)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.tiled = True
                print("[esrgan] OOM on the whole frame: tiles 256 + 16 overlap from now on", flush=True)
                y = self._tiled(x)
        y = y.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y[..., ::-1].copy()


def upscale_esrgan(model: Esrgan, src, dst):
    fr = _read_frames(src)
    enc = _encoder(dst, FINAL_CLIP[0], FINAL_CLIP[1])
    for f in fr:
        out = model(f)
        if out.shape[1] != FINAL_CLIP[0] or out.shape[0] != FINAL_CLIP[1]:
            raise ValueError(f"esrgan output {out.shape} is not {FINAL_CLIP}")
        enc.stdin.write(out.tobytes())
    enc.stdin.close()
    enc.wait()
    return len(fr)


def upscale_matte(matte_dir, clip_path, out_dir):
    """matte PNGs (576x1024) -> bilinear 1152x2048 -> guided filter against the upscaled clean clip (resized to the matte size)."""
    import cv2
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(matte_dir, "*.png")))
    clip = _read_frames(clip_path)
    if len(clip) != len(files):
        print(f"[matte] {os.path.basename(matte_dir)}: {len(files)} mattes vs {len(clip)} clip frames (the shorter wins)", flush=True)
    W, H = FINAL_MATTE
    gf = cv2.ximgproc.guidedFilter
    for i, f in enumerate(files):
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        up = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        g = cv2.resize(clip[min(i, len(clip) - 1)], (W, H), interpolation=cv2.INTER_LINEAR)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        r = gf(g, up, 8, 2e-4)
        r8 = np.clip(r * 255.0 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), cv2.merge([r8, r8, r8]))
    return len(files)


def prep(variant, shots, arch, piece_path, model_path=None):
    import shutil
    os.makedirs(arch, exist_ok=True)
    timing = {"variant": variant, "stages": {}}
    shutil.copy(piece_path, os.path.join(arch, "piece.json"))
    for n in ("index.json", "still.png"):
        if os.path.isfile(os.path.join(shots, n)):
            shutil.copy(os.path.join(shots, n), os.path.join(arch, n))
    sids = sorted((os.path.basename(p)[:-len("_clean.mp4")] for p in glob.glob(os.path.join(shots, "s*_clean.mp4"))), key=lambda s: int(s[1:]))
    model = None
    if variant == "B":
        import torch
        model = Esrgan(model_path)
        torch.cuda.reset_peak_memory_stats()
    for layer in ("clean", "textbg"):
        t0 = time.time()
        n = 0
        for sid in sids:
            src, dst = os.path.join(shots, f"{sid}_{layer}.mp4"), os.path.join(arch, f"{sid}_{layer}.mp4")
            if not os.path.isfile(src):
                continue
            if variant == "A":
                upscale_lanczos(src, dst)
                n += 1
            else:
                n += upscale_esrgan(model, src, dst)
        timing["stages"][f"clips_{layer}"] = {"s": round(time.time() - t0, 2), "frames": n}
        print(f"[prep {variant}] {layer}: {n} {'files' if variant == 'A' else 'frames'} in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    n = 0
    for sid in sids:
        n += upscale_matte(os.path.join(shots, f"{sid}_matte"), os.path.join(arch, f"{sid}_clean.mp4"), os.path.join(arch, f"{sid}_matte"))
    timing["stages"]["mattes"] = {"s": round(time.time() - t0, 2), "frames": n}
    print(f"[prep {variant}] mattes: {n} frames in {time.time() - t0:.1f} s", flush=True)
    if model is not None:
        import torch
        timing["torch_peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        timing["esrgan_tiled"] = model.tiled
    json.dump(timing, open(os.path.join(arch, "prep_timing.json"), "w"), indent=1)
    return timing


# ---------------------------------------------------------------------------------------------- graph
def build_graph(src_graph, arch, source_video, piece_path, prefix, concept):
    g = json.load(open(src_graph, encoding="utf-8"))
    piece = json.load(open(piece_path, encoding="utf-8"))
    lay = piece.get("layout")
    if not lay:                                           # a raw piece file has no layout yet: finalize like MographPieceFromJSON
        sys.path.insert(0, ROOT)
        from tools.fusion import plan_rules
        lay = plan_rules.finalize_piece(piece, None)["layout"] if False else None
    sids = sorted(((s["id"] if "id" in s else None) for s in piece["shots"]), key=lambda s: int(str(s).lstrip("s") or 0))
    n = len(sids)
    new = {}

    def add(k, cls, title=None, **inputs):
        new[k] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title or k}}

    add("archive_in", "PrimitiveString", "IN archive", value=arch)
    add("source_in", "PrimitiveString", "IN source video", value=source_video)
    add("concept_in", "PrimitiveStringMultiline", "IN concept", value=json.dumps(concept))
    add("piece_in", "PrimitiveStringMultiline", "IN piece json (v018 r03, final sizes)", value=json.dumps(piece, ensure_ascii=False))
    add("load", "MographArchiveLoad", archive=["archive_in", 0], source_video=["source_in", 0], concept=["concept_in", 0])
    add("design_piece", "MographPieceFromJSON", piece_json=["piece_in", 0], music=["load", 1])
    for k in range(1, n + 1):
        add(f"tv{k}", "VHS_LoadVideoPath", f"text clip s{k}", video=f"{arch}/s{k}_textbg.mp4", force_rate=0, custom_width=0, custom_height=0, frame_load_cap=0,
            skip_first_frames=0, select_every_nth=1, format="AnimateDiff")
    # frames per shot come from the piece layout (finalize adds it); the loader prints them too
    sys.path.insert(0, ROOT)
    from tools.fusion import plan_rules
    full = plan_rules.finalize_piece(piece, None) if not piece.get("layout") else piece
    frames = {s: int(v["frames"]) for s, v in full["layout"].items()}
    for k in range(1, n + 1):
        add(f"tc{k}", "MographClipFromImages", f"text clip s{k} -> clip", images=[f"tv{k}", 0], piece=["design_piece", 0], index=k, frames=frames[f"s{k}"])
    add("gather", "MographLayerGather", piece=["design_piece", 0], request=["load", 0], clean=["load", 3], plates=False,
        **{f"text_{k}": [f"tc{k}", 0] for k in range(1, n + 1)})
    for key in ("vda",):
        new[key] = copy.deepcopy(g[key])
    for k in range(1, n + 1):
        new[f"d{k}.proc"] = copy.deepcopy(g[f"d{k}.proc"])
        new[f"d{k}.proc"]["inputs"]["images"] = ["gather", 6 + k]
        new[f"d{k}.out"] = copy.deepcopy(g[f"d{k}.out"])
    mat = {f"matte_{k}": ["load", 5 + k] for k in range(1, n + 1)}
    dep = {f"depth_{k}": [f"d{k}.out", 0] for k in range(1, n + 1)}
    add("flow", "MographFootageFlow", piece=["design_piece", 0], clips=["gather", 0], method="auto", model="raft_large")
    add("pivot", "MographPivotCut", piece=["design_piece", 0], clips=["flow", 0], **mat, **dep)
    add("type_final", "MographTypePasses", piece=["design_piece", 0], music=["load", 1], procs=10)
    add("compose", "MographCompose", piece=["design_piece", 0], type=["type_final", 0], clips=["pivot", 0], backend="auto", music=["load", 1], **mat, **dep)
    add("assemble", "MographAssemble", composed=["compose", 0], piece=["design_piece", 0], audio=["load", 5], crf=int(g["assemble"]["inputs"].get("crf", 20)), encoder="auto")
    add("out_final", "SaveVideo", "OUT final", video=["assemble", 0], filename_prefix=prefix + "_video", format="auto", codec="auto")
    add("archive", "MographArchive", filename_prefix=prefix + "_out", piece=["design_piece", 0], clips=["flow", 0], first=["load", 4], video=["assemble", 0], type=["type_final", 0],
        composed=["compose", 0], cuts=["pivot", 0], **mat, **dep)
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("piece")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--piece", required=True)
    p.add_argument("--model", default="/workspace/ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth")
    p = sub.add_parser("graph")
    p.add_argument("--src", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--source-video", required=True)
    p.add_argument("--piece", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="final1080")
    p.add_argument("--seed", type=int, default=653)
    p.add_argument("--length", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "piece":
        json.dump(final_piece(json.load(open(a.src, encoding="utf-8"))), open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    elif a.cmd == "prep":
        print(json.dumps(prep(a.variant, a.shots, a.arch, a.piece, a.model)))
    else:
        g = build_graph(a.src, a.arch, a.source_video, a.piece, a.prefix, {"seed": a.seed, "length": a.length})
        json.dump(g, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{len(g)} nodes -> {a.out}")


if __name__ == "__main__":
    main()
