#!/usr/bin/env python
"""P01 masked Motion-Track probe: LTX-2.5 video-to-video where only the subject (Alpha-Gen matte area) is regenerated,
steered by the Motion-Track IC-LoRA; the rest is the original footage (pixel-exact after the final recomposite).

Graph = ltx_ic.build("motiontrack", tracks=...) with the empty latent replaced by the VAE-encoded footage:

    footage mp4 -> LoadVideo -> GetVideoComponents -> Resize (stage-1 size) -> VAEEncode ----+
    matte mp4   -> LoadVideo -> GetVideoComponents -> Resize (stage-1 size) ----------------+-> MographVideoLatentMask
        -> LTXAddVideoICLoRAGuide (track video drawn in-graph by LTXVDrawTracks) -> LTXVConcatAVLatent -> sampler
        (ManualSigmas starting at `denoise`) -> LTXVCropGuides -> [x2 upsample + 2nd MographVideoLatentMask + stage-2 sampler]
        -> VAEDecodeTiled -> resize to footage size -> ImageCompositeMasked(original footage, generated, region_mask)
        -> SaveVideo (+ the raw un-composited decode as a second SaveVideo)

    python tools/fusion/ltx_mask.py bundle          # tracks, graphs, previews, matte mp4 -> 16_app/mask_ltx/
    python tools/fusion/ltx_mask.py check           # structural + /object_info validation of every graph in the bundle

Mask semantics: 1 = regenerate, 0 = keep the input latent. The mask is set BEFORE the IC guide is added (the guide node
expands a spatial noise_mask and appends its own frames' mask). Stage-1 canvas / size 576x1024; footage 1088x1920.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ltx_ic  # noqa: E402  (read/import only)
import mask_tracks  # noqa: E402

PACK = os.path.join(ROOT, "16_app", "mask_ltx")
FOOTAGE_SRC = os.path.join(ROOT, "15_fusion", "v002", "out", "motion-graphic_fusion_v002_P01_s1_1_00001_.mp4")
MATTE_DIR = os.path.join(ROOT, "15_fusion", "v002", "local", "pieces", "P01", "seq", "P01_alpha_s1_1")
ANALYSIS = os.path.join(ROOT, "15_fusion", "v002", "local", "pieces", "P01", "analysis.json")
FOOTAGE_NAME = "p01_s1_footage.mp4"     # names inside ComfyUI/input on the box
MATTE_NAME = "p01_s1_matte.mp4"
STAGE1 = (576, 1024)
FOOTAGE_SIZE = (1088, 1920)
FRAMES = 121

NODE_MASK = "MographVideoLatentMask"
LOCAL_OBJECT_INFO = "http://127.0.0.1:8190/object_info"

# class_type -> (pack, number of outputs): ltx_ic.CLASSES + what this graph adds
CLASSES = dict(ltx_ic.CLASSES)
CLASSES.update({"VAEEncode": (ltx_ic.CORE, 1), "ImageToMask": (ltx_ic.CORE, 1), "ImageCompositeMasked": (ltx_ic.CORE, 1),
                NODE_MASK: ("ComfyUI-Mograph (nodes_fusion/latent_mask.py)", 2)})

DISTILLED = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE2_SIGMAS = "0.725, 0.421875, 0.0"

BASE_PROMPT = ("thick rolling sea mist drifts slowly across the dark water and curls around the black rocks, soft waves wash "
               "against the rocks, grey dawn on a cold sea, soft silver light, 35mm film, muted desaturated silver and slate "
               "palette")
STYLE_PROMPTS = {
    "pull_apart": ("the lone figure in the long dark coat splits down the middle and pulls apart into long fine silver "
                   "threads that stream away to the left and to the right in parallel lines and dissolve into the mist"),
    "unravel": ("the lone figure in the long dark coat unravels outward into fine silver threads that spiral away from the "
                "body and dissolve into the mist"),
    "rain": ("the lone figure in the long dark coat dissolves into fine silver threads that fall straight down like rain "
             "onto the black rocks"),
    "rise": ("the lone figure in the long dark coat dissolves into fine silver threads that float straight up and thin out "
             "as they rise into the mist above"),
}


# --------------------------------------------------------------------------------------------- helpers
def sigmas_for(denoise: float) -> str:
    """ManualSigmas string for a partial-denoise start: [denoise] + the distilled schedule values below it.
    1.0 -> the full 8-step schedule; 0.8 -> 0.8, 0.725, 0.421875, 0.0; 0.6 -> 0.6, 0.421875, 0.0."""
    d = float(denoise)
    if not 0.05 <= d <= 1.0:
        raise ValueError("denoise must be in 0.05..1")
    if d >= 0.999:
        vals = DISTILLED
    else:
        vals = [d] + [s for s in DISTILLED if s < d - 1e-6]
    return ", ".join("0.0" if v == 0 else f"{v:g}" for v in vals)


def _node(cls, title=None, **inputs):
    n = {"inputs": inputs, "class_type": cls}
    if title:
        n["_meta"] = {"title": title}
    return n


def _resize(src, w_ref, h_ref, crop="disabled", method="area"):
    return {"inputs": {"scale_method": method, "input": src, "resize_type": "scale dimensions",
                       "resize_type.crop": crop, "resize_type.width": w_ref, "resize_type.height": h_ref},
            "class_type": "ResizeImageMaskNode"}


def check_graph(g: dict) -> list[str]:
    """ltx_ic.check_graph with this module's CLASSES (adds VAEEncode, ImageCompositeMasked, MographVideoLatentMask)."""
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


# --------------------------------------------------------------------------------------------- the graph
def build_graph(tracks, *, denoise: float = 0.8, prompt: str | None = None, style: str = "pull_apart", seed: int = 1,
                guide_strength: float = 1.0, lora_strength: float = 1.0, dilate_cells: int = 2, feather: float = 0.75,
                mask_strength: float = 1.0, keep_first_frame: bool = True, region_feather_px: float = 12.0,
                two_stage: bool = False, guide_canvas: str = "full", frames: int = FRAMES, footage: str = FOOTAGE_NAME,
                matte: str = MATTE_NAME, corridor: str | None = None, main_subject_only: bool = True, out_prefix: str = "mask_ltx/probe", tracks_size=STAGE1,
                footage_size=FOOTAGE_SIZE, save_raw: bool = True, stage2_sigmas: str = STAGE2_SIGMAS) -> dict:
    """API graph for one probe run.

    tracks        list of tracks ({"x","y"} or (x,y) per frame) in `tracks_size` pixels (default the 576x1024 stage-1 canvas).
    denoise       first sigma of the (stage-1) schedule: 1.0 = full distilled schedule, 0.8 / 0.6 = start lower (keeps more of the footage).
    guide_canvas  'full' (as ltx_ic.build: track video drawn at the stage-1 size, like the earlier probes) or 'half'
                  (drawn at stage-1/2 = the control-video convention; single stage only).
    corridor      file name (in ComfyUI/input) of a 24 fps mp4 sweep matte (see mask_tracks.corridor_matte): its cells are unioned into
                  the regenerated area, so threads can exist where the tracks travel and not only on the alpha area.
    two_stage     stage 1 at 576x1024, x2 latent upsample, stage 2 at 1152x2048 with its own latent mask (decode is resized
                  back to the footage size for the recomposite).
    """
    if guide_canvas not in ("full", "half"):
        raise ValueError("guide_canvas: full | half")
    if guide_canvas == "half" and two_stage:
        raise ValueError("guide_canvas 'half' is single-stage only")
    w1, h1 = STAGE1
    final = (w1 * 2, h1 * 2) if two_stage else (w1, h1)
    tr = mask_tracks.from_json_tracks(tracks) if not hasattr(tracks, "shape") else tracks
    tr_final = mask_tracks.scale_tracks(tr, tracks_size, final)             # build() wants FINAL-size pixels
    p = prompt if prompt is not None else f"{BASE_PROMPT}, {STYLE_PROMPTS[style]}"
    g = ltx_ic.build("motiontrack", tracks=mask_tracks.to_json_tracks(tr_final), size=final, frames=frames, prompt=p,
                     two_stage=two_stage, seed=seed, lora_strength=lora_strength, out_prefix=out_prefix)

    # first-frame conditioning is not used here (the footage is the input latent)
    for nid in ("23", "24", "25", "26", "27", "28", "30", "31", "51"):
        g.pop(nid, None)
    g["32"]["inputs"]["strength"] = float(guide_strength)
    g["38"]["inputs"]["sigmas"] = sigmas_for(denoise)
    if guide_canvas == "half":
        ds = 2
        g["21"] = _node("PrimitiveInt", "guide canvas W", value=w1 // ds)
        g["22"] = _node("PrimitiveInt", "guide canvas H", value=h1 // ds)
        g["17"]["inputs"]["width"], g["17"]["inputs"]["height"] = ["21", 0], ["22", 0]
        half = mask_tracks.scale_tracks(tr_final, final, (w1 // ds, h1 // ds))
        g["17"]["inputs"]["tracks"] = json.dumps(mask_tracks.to_json_tracks(half))
    w_ref, h_ref = ["10", 0], ["11", 0]                                        # stage-1 w / h primitives

    # footage -> latent (v2v input)
    g["70"] = _node("LoadVideo", "IN footage", file=footage)
    g["71"] = _node("GetVideoComponents", video=["70", 0])
    g["72"] = _resize(["71", 0], w_ref, h_ref, "disabled", "area")
    g["73"] = _node("ImageFromBatch", image=["72", 0], batch_index=0, length=["19", 1])
    g["74"] = _node("VAEEncode", "footage -> latent", pixels=["73", 0], vae=["3", 0])
    # matte -> latent mask
    g["75"] = _node("LoadVideo", "IN matte", file=matte)
    g["76"] = _node("GetVideoComponents", video=["75", 0])
    g["77"] = _resize(["76", 0], w_ref, h_ref, "disabled", "bilinear")
    mask_kw = dict(dilate_cells=int(dilate_cells), feather=float(feather), strength=float(mask_strength),
                   keep_first_frame=bool(keep_first_frame), threshold=0.5, cell_threshold=0.0,
                   region_feather_px=float(region_feather_px), main_subject_only=bool(main_subject_only), min_blob_frac=0.25)
    if corridor:
        g["90"] = _node("LoadVideo", "IN corridor (track sweep)", file=corridor)
        g["91"] = _node("GetVideoComponents", video=["90", 0])
        g["92"] = _resize(["91", 0], w_ref, h_ref, "disabled", "bilinear")
        g["93"] = _node("ImageToMask", image=["92", 0], channel="red")
        mask_kw["extra_mask"] = ["93", 0]
    g["78"] = _node(NODE_MASK, "latent mask (stage 1)", samples=["74", 0], image=["77", 0], **mask_kw)
    g["32"]["inputs"]["latent"] = ["78", 0]
    region = ["78", 1]
    decoded = ["60", 0]
    if two_stage:
        g["79"] = _node(NODE_MASK, "latent mask (stage 2)", samples=["50", 0], image=["77", 0], **mask_kw)
        g["52"]["inputs"]["video_latent"] = ["79", 0]
        g["56"]["inputs"]["sigmas"] = stage2_sigmas
        region = ["79", 1]

    # recomposite: original footage where the mask is 0, generated where it is 1
    fw, fh = footage_size
    g["80"] = _node("PrimitiveInt", "footage W", value=int(fw))
    g["81"] = _node("PrimitiveInt", "footage H", value=int(fh))
    g["82"] = _resize(decoded, ["80", 0], ["81", 0], "disabled", "lanczos")
    g["83"] = _node("ImageFromBatch", image=["71", 0], batch_index=0, length=["19", 1])
    g["84"] = _resize(region, ["80", 0], ["81", 0], "disabled", "bilinear")
    g["85"] = _node("ImageCompositeMasked", "recomposite (original outside the mask)", destination=["83", 0],
                    source=["82", 0], x=0, y=0, resize_source=False, mask=["84", 0])
    g["61"]["inputs"]["images"] = ["85", 0]
    g["62"]["_meta"] = {"title": "OUT composite"}
    if save_raw:
        g["86"] = _node("CreateVideo", bit_depth=8, color_space="sRGB", codec="none", images=decoded, fps=["13", 0])
        g["87"] = _node("SaveVideo", "OUT raw decode (no recomposite)", filename_prefix=out_prefix + "_raw", format="auto",
                        codec="auto", video=["86", 0])
    bad = check_graph(g)
    if bad:
        raise AssertionError("; ".join(bad))
    return g


# --------------------------------------------------------------------------------------------- validation vs ComfyUI
def object_info(url: str = LOCAL_OBJECT_INFO, timeout: float = 20.0):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def node_info_from_pack() -> dict:
    """object_info-style entry for MographVideoLatentMask taken from the node's own schema (needs comfy_api)."""
    for path in (os.path.abspath(os.path.join(ROOT, "..", "..", "..", "ComfyUI")), os.environ.get("MOGRAPH_COMFYUI_PATH")):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from nodes_fusion import latent_mask as node          # noqa: PLC0415
    return {NODE_MASK: node.MographVideoLatentMask.GET_NODE_INFO_V1()}


# inputs the existing box graphs (ltx_i2v.json, known to load on the box) set but the older local ComfyUI (8190) lacks
BOX_ONLY_INPUTS = {("CreateVideo", "bit_depth"), ("CreateVideo", "color_space"), ("CreateVideo", "codec")}


def validate_vs_object_info(g: dict, oi: dict) -> tuple[list[str], list[str], list[str]]:
    """-> (problems, classes_checked, classes_not_available). Checks: class exists, every non-dotted input name is a known
    required/optional input, links point at an existing output slot of the source class."""
    problems, checked, missing = [], set(), set()
    for nid, n in g.items():
        ct = n["class_type"]
        info = oi.get(ct)
        if info is None:
            missing.add(ct)
            continue
        checked.add(ct)
        spec = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        for name in n.get("inputs", {}):
            if "." in name:
                continue                                 # dynamic combo / autogrow members (resize_type.width, values.a)
            if name not in spec and (ct, name) not in BOX_ONLY_INPUTS:
                problems.append(f"{nid} {ct}: input '{name}' unknown (has {sorted(spec)})")
        for name, v in n.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                src = g[v[0]]["class_type"]
                if src in oi and v[1] >= len(oi[src]["output"]):
                    problems.append(f"{nid}.{name} -> {v[0]}[{v[1]}] but {src} has {len(oi[src]['output'])} outputs")
        for name, s in spec.items():
            if name in n.get("inputs", {}) or name in info.get("input", {}).get("optional", {}):
                continue
            if isinstance(s, list) and s and isinstance(s[0], str) and s[0].startswith("COMFY_"):
                continue
            problems.append(f"{nid} {ct}: required input '{name}' not set")
    return problems, sorted(checked), sorted(missing)


# --------------------------------------------------------------------------------------------- bundle
GRID_STYLES = ("pull_apart", "unravel", "rain", "rise")
GRID_DENOISE = (0.8, 1.0)
TRACK_KW = dict(n_points=40, distance=300.0, stagger_s=0.6, seed=1)


CORRIDOR_RADIUS, CORRIDOR_TRAIL = 16.0, 12


def corridor_name(style: str) -> str:
    return f"p01_s1_corridor_{style}.mp4"


def dn_tag(d: float) -> str:
    return f"d{int(round(d * 100)):03d}"


def write_gray_mp4(frames, dst: str, fps: int = 24) -> None:
    """(F,H,W) float 0..1 -> mp4 (gray replicated, yuv420p, crf 6)."""
    import numpy as np
    f, h, w = frames.shape
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "6", "-preset", "slow", "-pix_fmt", "yuv420p", dst],
                         stdin=subprocess.PIPE)
    p.stdin.write((np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed")


def make_matte_video(dst: str) -> None:
    """Matte PNG sequence -> 24 fps mp4 (crf 6, yuv420p) for LoadVideo (core has no image-sequence loader)."""
    import glob
    n = len(glob.glob(os.path.join(MATTE_DIR, "*.png")))
    pat = os.path.join(MATTE_DIR, "motion-graphic_fusion_v002_P01_alpha_s1_1_image_%05d_.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "24", "-i", pat, "-frames:v", str(n), "-c:v", "libx264",
                    "-crf", "6", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", "24", dst], check=True)


def cmd_bundle(args) -> None:
    import shutil
    os.makedirs(os.path.join(PACK, "refs"), exist_ok=True)
    os.makedirs(os.path.join(PACK, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(PACK, "graphs", "extra"), exist_ok=True)
    os.makedirs(os.path.join(PACK, "tracks"), exist_ok=True)
    os.makedirs(os.path.join(PACK, "previews"), exist_ok=True)
    foot = os.path.join(PACK, "refs", FOOTAGE_NAME)
    mat = os.path.join(PACK, "refs", MATTE_NAME)
    if not os.path.isfile(foot):
        shutil.copyfile(FOOTAGE_SRC, foot)
    if not os.path.isfile(mat):
        make_matte_video(mat)
    matte = mask_tracks.load_matte_seq(MATTE_DIR, STAGE1)
    an = json.load(open(ANALYSIS, encoding="utf-8"))
    f0, t0, label = mask_tracks.snap_start(args.start_s, an, 24.0, "beat", len(matte))
    print(f"start: frame {f0} ({t0:.3f}s) {label}")
    tracks = {}
    for style in GRID_STYLES:
        tr, meta = mask_tracks.build_tracks(matte, style, start_frame=f0, **TRACK_KW)
        meta["start_snap"] = label
        tracks[style] = tr
        with open(os.path.join(PACK, "tracks", f"tracks_{style}.json"), "w", encoding="utf-8") as f:
            json.dump(mask_tracks.to_json_tracks(tr), f)
        with open(os.path.join(PACK, "tracks", f"tracks_{style}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print(style, meta["net_displacement_px"])
    written = []
    for style in GRID_STYLES:
        for d in GRID_DENOISE:
            name = f"mask_{style}_{dn_tag(d)}"
            g = build_graph(tracks[style], denoise=d, style=style, out_prefix=f"mask_ltx/{name}")
            _write(os.path.join(PACK, "graphs", name + ".json"), g)
            written.append(name)
    for style in GRID_STYLES:   # sweep (corridor) matte per style, 24 fps mp4
        write_gray_mp4(mask_tracks.corridor_matte(tracks[style], STAGE1, CORRIDOR_RADIUS, CORRIDOR_TRAIL),
                       os.path.join(PACK, "refs", corridor_name(style)))
    extras = [(f"mask_{st}_d080_corridor", dict(style=st, denoise=0.8, corridor=corridor_name(st), dilate_cells=1))
              for st in GRID_STYLES] + [("mask_pull_apart_d060", dict(style="pull_apart", denoise=0.6)),
              ("mask_pull_apart_d080_guidehalf", dict(style="pull_apart", denoise=0.8, guide_canvas="half")),
              ("mask_pull_apart_d080_2stage", dict(style="pull_apart", denoise=0.8, two_stage=True))]
    for name, kw in extras:
        g = build_graph(tracks[kw.get("style", "pull_apart")], out_prefix=f"mask_ltx/{name}", **kw)
        _write(os.path.join(PACK, "graphs", "extra", name + ".json"), g)
        written.append("extra/" + name)
    print("graphs:", ", ".join(written))


def _write(path: str, g: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1, ensure_ascii=False)


def cmd_check(args) -> int:
    import glob
    errs = 0
    try:
        oi = object_info()
        print(f"local object_info: {len(oi)} classes")
    except Exception as e:      # noqa: BLE001
        print(f"local ComfyUI not reachable ({e}); structural check only")
        oi = {}
    try:
        oi.update(node_info_from_pack())
        print(f"{NODE_MASK}: schema taken from nodes_fusion/latent_mask.py")
    except Exception as e:      # noqa: BLE001
        print(f"{NODE_MASK}: could not load schema ({type(e).__name__}: {e})")
    not_local: set[str] = set()
    for path in sorted(glob.glob(os.path.join(PACK, "graphs", "**", "*.json"), recursive=True)):
        g = json.load(open(path, encoding="utf-8"))
        bad = check_graph(g)
        msg = []
        if oi:
            pb, chk, miss = validate_vs_object_info(g, oi)
            bad += pb
            not_local |= set(miss)
        print(f"{os.path.relpath(path, PACK)}: {len(g)} nodes, {'OK' if not bad else bad}")
        errs += len(bad)
    if oi:
        print("not available locally (cannot be validated against a live /object_info):", sorted(not_local))
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--start-s", type=float, default=1.3, help="track start time, snapped to the nearest beat")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "bundle":
        cmd_bundle(a)
    else:
        sys.exit(1 if cmd_check(a) else 0)


if __name__ == "__main__":
    main()
