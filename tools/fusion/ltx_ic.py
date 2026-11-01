#!/usr/bin/env python
"""Patch the LTX-2.5 IC-LoRA API graphs in workflows/api/ and write the result.

Kinds (graph, LoRA, what the reference video is):
  alpha        ltx25_ic_v2v.json      ltx-2.5-22b-ic-lora-alpha-gen-0.9          RGB clip -> grey alpha matte (empty prompt)
  control      ltx25_ic_control.json  ltx-2.3-22b-ic-lora-union-control-ref0.5   canny / depth / pose pass of a clip
  motiontrack  ltx25_ic_control.json  ltx-2.3-22b-ic-lora-motion-track-control-ref0.5   point-track video (see render_tracks)

    python tools/fusion/ltx_ic.py alpha --ref p01_clip.mp4 --size 576x1024 --out /tmp/alpha.json
    python tools/fusion/ltx_ic.py control --ref p01_canny.mp4 --first-frame p01_f0.png --prompt "..." \
        --size 1152x2048 --two-stage --out /tmp/ctl.json
    python tools/fusion/ltx_ic.py motiontrack --tracks tracks.json --first-frame p01_f0.png --prompt "..." --out g.json
    python tools/fusion/ltx_ic.py --render-tracks tracks.json --size 576x1024 --out-dir tracks_png   # track video frames
    python tools/fusion/ltx_ic.py --check                                                           # self-check all graphs

--ref / --first-frame are file names directly inside ComfyUI/input on the box (upload them first; core LoadVideo /
LoadImage only list top-level files, so no sub-folders). All reference clips are
read at 24 fps without temporal resampling: re-encode to 24 fps first.

Size rules (size = FINAL output size, 9:16 = (w, h)):
  alpha       multiple of 32, single stage only (stage-1 at native size is the LoRA's recipe). Draft/final 576x1024 / 1088x1920.
  control     the reference runs at half size (reference_downscale_factor 2), so stage-1 w/h must be multiples of 64: draft single
              stage 576x1024, final two-stage 1152x2048 (stage 1 = 576x1024). 544x960 -> 1088x1920 (stage 1 17x30 latents) FAILS on
              the box: LTXAddVideoICLoRAGuide rearrange "can't divide axis of length 17 in chunks of 2" (tested 2026-10-26).
  motiontrack single stage: size multiple of 64 (576x1024); two-stage: size multiple of 128 (1152x2048, stage 1 = 576x1024).
              Masked trails (style rise) use it. Same sizes as control, so the compose scales 0.9375 to 1080x1920 / 540x960.
  plan_size() picks (size, two_stage) for a planned shot: final two-stage, draft single stage, IC sizes snapped to the rules.
Frames: default = reference length snapped down to 8k+1, capped (alpha 145, others 193). `frames` lowers the cap.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = os.path.join(ROOT, "workflows", "api")

LORA_ALPHA = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
LORA_UNION = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
LORA_TRACK = "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"

KINDS = {
    "alpha": dict(graph="ltx25_ic_v2v", lora=LORA_ALPHA, ds=1.0, max_frames=145, align=32, first_strength=None),
    "control": dict(graph="ltx25_ic_control", lora=LORA_UNION, ds=2.0, max_frames=193, align=64, first_strength=1.0),
    "motiontrack": dict(graph="ltx25_ic_control", lora=LORA_TRACK, ds=2.0, max_frames=193, align=64,
                        first_strength=0.7),
}

# class_type -> (source pack, number of outputs)  (used by check_graph and the README)
CORE = "ComfyUI core"
LTXV = "ComfyUI-LTXVideo"
CLASSES = {
    "UNETLoader": (CORE, 1), "CLIPLoader": (CORE, 1), "VAELoader": (CORE, 1), "LatentUpscaleModelLoader": (CORE, 1),
    "CLIPTextEncode": (CORE, 1), "LTXVConditioning": (CORE, 2), "PrimitiveInt": (CORE, 1),
    "PrimitiveBoolean": (CORE, 1), "ComfyMathExpression": (CORE, 2), "LoadVideo": (CORE, 1),
    "GetVideoComponents": (CORE, 3), "ResizeImageMaskNode": (CORE, 1), "GetImageSize": (CORE, 3),
    "ImageFromBatch": (CORE, 1), "EmptyAudio": (CORE, 1), "VAEEncodeAudio": (CORE, 1), "LoadImage": (CORE, 2),
    "EmptyLTXVLatentVideo": (CORE, 1), "LTXVEmptyLatentAudio": (CORE, 1), "LTXVConcatAVLatent": (CORE, 1),
    "LTXVSeparateAVLatent": (CORE, 2), "LTXVCropGuides": (CORE, 3), "LTXVPreprocess": (CORE, 1),
    "LTXVImgToVideoInplace": (CORE, 1), "LTXVLatentUpsampler": (CORE, 1), "RandomNoise": (CORE, 1),
    "CFGGuider": (CORE, 1), "KSamplerSelect": (CORE, 1), "ManualSigmas": (CORE, 1),
    "SamplerCustomAdvanced": (CORE, 2), "VAEDecodeTiled": (CORE, 1), "CreateVideo": (CORE, 1),
    "SaveVideo": (CORE, 0), "SaveImage": (CORE, 0),
    "LTXICLoRALoaderModelOnly": (LTXV, 2), "LTXAddVideoICLoRAGuide": (LTXV, 3),
    "LTXVSetAudioRefTokens": (LTXV, 3), "LTXVDrawTracks": (LTXV, 1),
}


# --------------------------------------------------------------------------------------------- helpers
def snap_frames(n: int) -> int:
    """Largest 8k+1 <= n (min 9)."""
    return max(9, ((int(n) - 1) // 8) * 8 + 1)


def load(name: str) -> dict:
    with open(os.path.join(API, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _drop(g: dict, ids) -> None:
    for i in ids:
        g.pop(i, None)


def probe_video(path: str):
    """-> dict(frames, fps, width, height) via ffprobe, or None (file missing / no ffprobe)."""
    if not os.path.isfile(path):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
             "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", path],
            capture_output=True, text=True, check=True).stdout
        s = json.loads(out)["streams"][0]
        a, b = s["r_frame_rate"].split("/")
        return dict(frames=int(s["nb_read_frames"]), fps=float(a) / float(b), width=int(s["width"]),
                    height=int(s["height"]))
    except Exception:
        return None


def _norm_tracks(tracks):
    """JSON str / list -> list of tracks, each a list of (x, y) floats."""
    if isinstance(tracks, str):
        tracks = json.loads(tracks)
    out = []
    for t in tracks:
        if isinstance(t, str):
            t = json.loads(t)
        out.append([(float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1]))
                    for p in t])
    if not out or not all(out):
        raise ValueError("tracks: need at least one non-empty track")
    return out


# --------------------------------------------------------------------------------------------- build
def plan_size(kind: str, size, draft: bool = False, preview: bool = False) -> tuple[tuple[int, int], bool]:
    """(size, two_stage) to build the IC `kind` (control | motiontrack) for a planned shot `size` (plan_rules.sizes_for "shot").
    final: two-stage (sides multiples of 128), draft / preview: single stage, no stage 2 / x2 decode (sides multiples of 64). A size
    that breaks the rule (e.g. 1088x1920, 544x960) is replaced by 1152x2048 / 576x1024 (preview 384x704) in the same orientation."""
    w, h = int(size[0]), int(size[1])
    single = draft or preview
    a = 64 if single else 128
    if w % a == 0 and h % a == 0:
        return (w, h), not single
    land = w > h
    if preview:
        return ((704, 384) if land else (384, 704)), False
    if draft:
        return ((1024, 576) if land else (576, 1024)), False
    return ((2048, 1152) if land else (1152, 2048)), True


def build(kind: str, ref_video: str | None = None, first_frame: str | None = None, prompt: str = "",
          negative: str = "", lora_strength: float = 1.0, size: tuple[int, int] = (576, 1024),
          frames: int | None = None, seed: int = 1, two_stage: bool = False, out_prefix: str | None = None, *,
          tracks=None, lora: str | None = None, ref_downscale: float | None = None, max_frames: int | None = None,
          cfg: float = 1.0, png: bool | None = None, crop: str = "center",
          first_frame_strength: float | None = None) -> dict:
    """Return an API-format graph for `kind` (alpha | control | motiontrack).

    size          final output (w, h). two_stage (control/motiontrack only) renders stage 1 at size/2, then x2 latent upscale.
    ref_video     control video name in ComfyUI/input (alpha: the RGB clip). motiontrack may pass `tracks` instead
                  (JSON / list of tracks, coordinates in FINAL-size pixels, one point per frame) -> drawn in-graph
                  by LTXVDrawTracks.
    first_frame   optional image name (control/motiontrack). Not supported for alpha.
    frames        None = reference length snapped to 8k+1 (<= max_frames); an int lowers the cap.
    ref_downscale literal latent_downscale_factor for the guide (default: the kind's value, 1 for alpha, 2 for ref0.5 LoRAs).
    png           alpha: also save a lossless PNG sequence (default True for alpha).
    negative      only used when cfg > 1 (the distilled path runs cfg 1, which ignores the negative prompt).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    k = KINDS[kind]
    w, h = int(size[0]), int(size[1])
    ds = float(ref_downscale if ref_downscale is not None else k["ds"])
    if kind == "alpha":
        if two_stage:
            raise ValueError("alpha: single stage only (stage 1 at native size is the LoRA recipe)")
        if first_frame:
            raise ValueError("alpha: first_frame is not supported (empty prompt, RGB clip is the only guide)")
        if tracks is not None:
            raise ValueError("alpha: tracks only apply to motiontrack")
    if ref_video and tracks is not None:
        raise ValueError("give either ref_video or tracks, not both")
    if tracks is not None and kind != "motiontrack":
        raise ValueError("tracks only apply to motiontrack")
    if not ref_video and tracks is None:
        raise ValueError("ref_video is required (or tracks for motiontrack)")
    s = 2 if two_stage else 1
    align = k["align"] * s
    if w % align or h % align:
        raise ValueError(f"{kind}{' two-stage' if two_stage else ''}: size {w}x{h} must be a multiple of {align} "
                         f"(try {max(align, round(w / align) * align)}x{max(align, round(h / align) * align)})")
    w1, h1 = w // s, h // s
    cap = snap_frames(frames if frames else (max_frames or k["max_frames"]))
    if tracks is not None:
        tr = _norm_tracks(tracks)
        cap = min(cap, snap_frames(max(len(t) for t in tr)))
    prefix = out_prefix or f"fusion/{kind}"

    g = load(k["graph"])
    g["5"]["inputs"]["lora_name"] = lora or k["lora"]
    g["5"]["inputs"]["strength_model"] = float(lora_strength)
    g["6"]["inputs"]["text"] = prompt
    g["7"]["inputs"]["text"] = negative
    g["10"]["inputs"]["value"] = w1
    g["11"]["inputs"]["value"] = h1
    g["14"]["inputs"]["value"] = cap
    if ref_video:
        g["15"]["inputs"]["file"] = ref_video
        g["17"]["inputs"]["resize_type.crop"] = crop

    if kind == "alpha":
        g["31"]["inputs"]["latent_downscale_factor"] = ds
        g["33"]["inputs"]["noise_seed"] = int(seed)
        g["34"]["inputs"]["cfg"] = float(cfg)
        g["42"]["inputs"]["filename_prefix"] = prefix
        g["43"]["inputs"]["filename_prefix"] = prefix + "_png"
        if png is False:
            _drop(g, ["43"])
        return g

    # control / motiontrack
    if (w1 / ds) % 32 or (h1 / ds) % 32:
        raise ValueError(f"stage-1 size {w1}x{h1} / downscale {ds:g} must be a multiple of 32")
    g["21"]["inputs"]["value"] = int(w1 / ds)
    g["22"]["inputs"]["value"] = int(h1 / ds)
    g["27"]["inputs"]["value"] = w
    g["28"]["inputs"]["value"] = h
    g["32"]["inputs"]["latent_downscale_factor"] = ds
    for nid in ("35", "53"):
        g[nid]["inputs"]["noise_seed"] = int(seed)
    for nid in ("36", "54"):
        g[nid]["inputs"]["cfg"] = float(cfg)
    g["62"]["inputs"]["filename_prefix"] = prefix
    if first_frame:
        g["23"]["inputs"]["image"] = first_frame
        g["26"]["inputs"]["value"] = False
        st = first_frame_strength if first_frame_strength is not None else k["first_strength"]
        g["31"]["inputs"]["strength"] = float(st)
    if tracks is not None:  # replace LoadVideo -> GetVideoComponents -> Resize by an in-graph track drawer
        sc = w1 / w
        scaled = [[{"x": x * sc, "y": y * sc} for x, y in t] for t in tr]
        _drop(g, ["15", "16", "21", "22"])
        g["17"] = {"inputs": {"tracks": json.dumps(scaled), "width": ["10", 0], "height": ["11", 0]},
                   "class_type": "LTXVDrawTracks", "_meta": {"title": "Draw motion tracks (stage-1 canvas)"}}
    if not two_stage:
        _drop(g, ["9", "50", "51", "52", "53", "54", "55", "56", "57", "58"])
        g["60"]["inputs"]["samples"] = ["41", 2]
        g["60"]["inputs"]["temporal_size"] = 64
        g["60"]["inputs"]["temporal_overlap"] = 8
    problems = check_graph(g)
    if problems:
        raise AssertionError("; ".join(problems))
    return g


# --------------------------------------------------------------------------------------------- check
def check_graph(g: dict) -> list[str]:
    """Structural self-check: ids exist, slots in range, class known. -> list of problems."""
    bad = []
    for nid, n in g.items():
        ct = n.get("class_type")
        if ct not in CLASSES:
            bad.append(f"{nid}: class {ct} not in CLASSES")
        for name, v in n.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src = g.get(v[0])
                if src is None:
                    bad.append(f"{nid}.{name} -> missing node {v[0]}")
                elif src.get("class_type") in CLASSES and not (0 <= v[1] < CLASSES[src["class_type"]][1]):
                    bad.append(f"{nid}.{name} -> {v[0]}[{v[1]}] out of range for {src['class_type']}")
    return bad


def check_all() -> int:
    errs = 0
    files = sorted(f for f in os.listdir(API) if f.startswith("ltx25_") and f.endswith(".json"))
    used: dict[str, set] = {}
    for f in files:
        g = load(f[:-5])
        p = check_graph(g)
        print(f"{f}: {len(g)} nodes, {'OK' if not p else p}")
        errs += len(p)
        used[f] = {n["class_type"] for n in g.values()}
    variants = [
        ("alpha", dict(ref_video="r.mp4")), ("alpha png off", dict(kind="alpha", ref_video="r.mp4", png=False)),
        ("control draft", dict(kind="control", ref_video="r.mp4", first_frame="f.png")),
        ("control two-stage", dict(kind="control", ref_video="r.mp4", size=(1152, 2048), two_stage=True)),
        ("motiontrack draft", dict(kind="motiontrack", ref_video="r.mp4")),
        ("motiontrack tracks", dict(kind="motiontrack", tracks=[[(10, 10), (50, 80), (90, 120)]],
                                    first_frame="f.png")),
        ("motiontrack tracks two-stage", dict(kind="motiontrack", tracks=[[(10, 10), (50, 80)]],
                                              size=(1152, 2048), two_stage=True)),
    ]
    for label, kw in variants:
        kw = dict(kw)
        kind = kw.pop("kind", label.split()[0])
        g = build(kind, **kw)
        p = check_graph(g)
        print(f"build[{label}]: {len(g)} nodes, {'OK' if not p else p}")
        errs += len(p)
        json.dumps(g)
    print("\nclass_type -> source pack")
    for ct in sorted(set().union(*used.values()), key=lambda c: (CLASSES.get(c, ("?",))[0], c)):
        print(f"  {ct:28s} {CLASSES.get(ct, ('UNLISTED',))[0]}")
    return errs


# --------------------------------------------------------------------------------------------- track renderer
# Exact numpy port of ComfyUI-LTXVideo LTXVDrawTracks (sparse_tracks.py) - the format the motion-track IC-LoRA
# was trained on. See graphs_pending/README.md for the format description.
TRACK_MIN_R, TRACK_MAX_R, TRACK_TRAIL, TRACK_REF_SHORT = 2, 8, 50, 1080


def _track_colors(ratios):
    import numpy as np
    r = np.asarray(ratios, dtype=np.float32)
    c = np.zeros((r.shape[0], 3), np.float32)
    m1 = r <= 1 / 3
    c[m1, 1] = r[m1] * 3
    c[m1, 2] = 1 - r[m1] * 3
    m2 = (r > 1 / 3) & (r <= 2 / 3)
    c[m2, 0] = (r[m2] - 1 / 3) * 3
    c[m2, 1] = 1
    m3 = r > 2 / 3
    c[m3, 0] = 1
    c[m3, 1] = 1 - (r[m3] - 2 / 3) * 3
    return c[:, [2, 1, 0]]  # node output is BGR ("matches IC-LoRA training data"); saved as-is, so head = blue


def _bilinear(img, out_h, out_w):
    """torch interpolate(mode=bilinear, align_corners=False, no antialias) in numpy; img (H, W, 3) float32."""
    import numpy as np
    H, W = img.shape[:2]
    ys = np.clip((np.arange(out_h) + 0.5) * H / out_h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * W / out_w - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None, None].astype(np.float32)
    wx = (xs - x0)[None, :, None].astype(np.float32)
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def render_tracks(tracks, width: int, height: int):
    """tracks: list of tracks, each a list of (x, y) or {"x","y"} per frame, in `width` x `height` pixel coordinates.
    -> uint8 array (N, height, width, 3), N = longest track. Black background; feed as the reference video of
    `motiontrack` (RGB, exactly what LTXVDrawTracks hands the guide node)."""
    import numpy as np
    tr = _norm_tracks(tracks)
    n_frames = max(len(t) for t in tr)
    if height <= width:
        rw, rh = int(width * TRACK_REF_SHORT / height), TRACK_REF_SHORT
    else:
        rw, rh = TRACK_REF_SHORT, int(height * TRACK_REF_SHORT / width)
    sx, sy = rw / width, rh / height
    half = 2 * TRACK_MAX_R + 3
    half //= 2
    off = np.arange(2 * half + 1) - half
    d2 = (off[:, None] ** 2 + off[None, :] ** 2).astype(np.float32)
    frames = np.zeros((n_frames, height, width, 3), np.uint8)
    for t in range(n_frames):
        tau0 = max(0, t - TRACK_TRAIL)
        canvas = np.zeros((rh, rw, 3), np.float32)
        circles = []  # (age, track, tau)
        for i, trk in enumerate(tr):
            for tau in range(tau0, min(t, len(trk) - 1) + 1):
                circles.append((t - tau, i, tau))
        circles.sort(key=lambda c: -c[0])  # oldest first, newest painted last
        ages = np.array([c[0] for c in circles], np.float32)
        ratios = 1.0 - ages / TRACK_TRAIL
        radii = TRACK_MIN_R + (TRACK_MAX_R - TRACK_MIN_R) * ratios
        cols = _track_colors(ratios)
        for j, (_, i, tau) in enumerate(circles):
            cx = int(np.round(tr[i][tau][0] * sx))
            cy = int(np.round(tr[i][tau][1] * sy))
            y0, y1 = max(0, cy - half), min(rh, cy + half + 1)
            x0, x1 = max(0, cx - half), min(rw, cx + half + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            m = d2[y0 - (cy - half):y1 - (cy - half), x0 - (cx - half):x1 - (cx - half)] <= radii[j] ** 2
            canvas[y0:y1, x0:x1][m] = cols[j]
        small = _bilinear(canvas, height, width)
        frames[t] = np.clip(np.round(small.astype(np.float16).astype(np.float32) * 255), 0, 255).astype(np.uint8)
    return frames


def write_frames(frames, out_dir: str) -> None:
    """Write PNG frames 00000.png... (PIL); encode with: ffmpeg -framerate 24 -i %05d.png -c:v libx264 -crf 12 -pix_fmt yuv420p t.mp4"""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(out_dir, f"{i:05d}.png"))


# --------------------------------------------------------------------------------------------- CLI
def _size(s: str):
    a, b = s.lower().split("x")
    return int(a), int(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", nargs="?", choices=sorted(KINDS))
    ap.add_argument("--check", action="store_true", help="self-check all ltx25_* graphs and every kind's build")
    ap.add_argument("--render-tracks", metavar="TRACKS_JSON", help="render a track video's PNG frames (needs --out-dir)")
    ap.add_argument("--out-dir")
    ap.add_argument("--ref")
    ap.add_argument("--tracks", help="JSON file: list of tracks of {x,y} / [x,y] per frame, FINAL-size pixel coords")
    ap.add_argument("--first-frame")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--lora")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--size", type=_size, default=(576, 1024), help="final WxH, e.g. 576x1024")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--two-stage", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--crop", choices=["center", "disabled"], default="center")
    ap.add_argument("--out-prefix")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check_all() else 0)
    if a.render_tracks:
        with open(a.render_tracks, encoding="utf-8") as f:
            fr = render_tracks(json.load(f), *a.size)
        write_frames(fr, a.out_dir or "tracks_png")
        print(f"{len(fr)} frames {a.size[0]}x{a.size[1]} -> {a.out_dir or 'tracks_png'}")
        return
    if not a.kind or not a.out:
        ap.error("kind and --out are required")
    tracks = None
    if a.tracks:
        with open(a.tracks, encoding="utf-8") as f:
            tracks = json.load(f)
    if a.ref:
        info = probe_video(a.ref)
        if info:
            print(f"ref: {info['frames']} frames @ {info['fps']:.3f} fps {info['width']}x{info['height']}")
            if abs(info["fps"] - 24) > 0.01:
                print("WARNING: reference is not 24 fps; the graph does not resample in time - re-encode first")
    g = build(a.kind, a.ref, a.first_frame, a.prompt, a.negative, a.lora_strength, a.size, a.frames, a.seed,
              a.two_stage, a.out_prefix, tracks=tracks, lora=a.lora, max_frames=a.max_frames, cfg=a.cfg,
              png=False if a.no_png else None, crop=a.crop)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1, ensure_ascii=False)
    print(a.out)


if __name__ == "__main__":
    main()
