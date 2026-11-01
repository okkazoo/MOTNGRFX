"""MographArchive: output node that keeps the intermediates of a run next to the final video.

    MographArchive  filename_prefix, piece, plan_piece, clips, first, video, matte_1..6 (lazy)

Writes into the ComfyUI output dir, under ``<filename_prefix>_shots/`` (progress_standin re-points ``filename_prefix`` into the
PROJECTS version, so the dashboard sync lands the folder in ``out/``):

    still.png                      the MOSS still (the first frame of the piece)
    sK_clip.mp4                    the generated LTX shot K (the footage as it enters compose); single-clip shots only
    sK_clean.mp4                   three-layer shots (nodes_fusion/layers.py): the CLEAN clip (plain image-to-video; the pixels the matte was cut
                                   from and the subject on top in compose)
    sK_textbg.mp4                  three-layer shots with a text pass: the TEXT-RENDERED background (union-control V2V of the clean clip)
    sK_matte/NNNNN.png             the Alpha-Gen matte frames of shot K, one PNG per frame, as decoded (native matte size)
    index.json                     piece id, fps, per shot: frames, clip size, matte size

    (whole-composition set, nodes_fusion/archive_ingredients.py: <stem>_cut_s1-s2_<kind>.mp4 + _matte/ PNGs of the 3D shot cuts, <stem>_s1_depth.png (+ _depth/ PNG
    sequence), <stem>_s1_comp.mp4 composed shots, <stem>_s1_plate.mp4, <stem>_s1_sources.json, <stem>_type_matte/ PNGs, <stem>_design_sheet.png, <stem>_places.json:
    16_app/INGREDIENTS.md; the heavy ones only with full_archive on)

Since 2026-10-29 EVERY file and folder is also written with the run stem in front (``<stem>_still.png``, ``<stem>_s1_matte/<stem>_s1_matte_00001.png``,
``<stem>_s1_clip.mp4``, ``<stem>_piece.json`` ...; stem = the filename_prefix stem without ``_out``, tools/fusion/ingredient_names.py) and the extras of
nodes_fusion/archive_ingredients.py are added (music wav, one still per place, type passes as mp4, depth / flow / contour previews, LTX-read crops), so the whole
composition can be found and pulled by one name (tools/app/ingredients.py, 16_app/INGREDIENTS.md). The short names above stay as hardlinks of the prefixed files
(``LEGACY_NAMES``) for MographArchiveLoad and tools/fusion/generations.py.

It waits for ``video`` (the assembled piece) so it runs LAST and adds nothing to a stage's time. The mattes are requested lazily and
ONLY for the shots whose recipe (rules plan or designed piece) reads one, i.e. the ones that were generated anyway: no extra
Alpha-Gen work is ever triggered by this node.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from comfy_api.latest import io

try:
    import folder_paths
except Exception:   # pragma: no cover
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY, FPS, MAX_SHOTS, MographClips, MographPiece, MographType
from .compose import _wired
from . import archive_ingredients as AI

ensure_root_on_path()
from tools.fusion import ingredient_names as IN   # noqa: E402
from tools.fusion import compose as fc            # noqa: E402
from tools.fusion import recipe_eval as RE        # noqa: E402


def _needs_matte(piece, sid: str, layered: bool = False) -> bool:
    try:
        return bool(piece) and sid in (piece.get("layout") or {}) and bool(RE.shot_needs(piece, sid, layered=layered)["matte"])
    except Exception:   # noqa: BLE001
        return False


def _needs(piece, sid: str, what: str, layered: bool = False) -> bool:
    try:
        return bool(piece) and sid in (piece.get("layout") or {}) and bool(RE.shot_needs(piece, sid, layered=layered)[what])
    except Exception:   # noqa: BLE001
        return False


def design_summary(piece: dict) -> dict:
    """the designed piece in a few fields for the batch report: technique, per word font / module / io / window, transitions, window cuts,
    compose op names, where each section came from (model or rules)."""
    t = piece.get("type") or {}
    comp = piece.get("compose") or {}
    ops: list[str] = []
    chains = [comp.get("default") or {}] + [v for v in (comp.get("shots") or {}).values() if isinstance(v, dict)]
    for ch in chains:
        for o in ch.get("ops") or []:
            if isinstance(o, dict) and o.get("op") and o["op"] not in ops:
                ops.append(o["op"])
    d = piece.get("design") or {}
    return {
        "id": piece.get("id"), "technique": piece.get("technique") or d.get("technique"),
        "words": [{k: w.get(k) for k in ("text", "font", "module", "form_in", "form_out", "t0", "t1", "window_source") if w.get(k) is not None}
                  for w in t.get("words") or []],
        "transitions": t.get("transitions"), "camera": t.get("camera"), "mood": t.get("mood"),
        "window_cuts": comp.get("cuts") or [], "compose_ops": ops,
        "src": (piece.get("plan") or {}).get("design", {}).get("src"),
        "shots": len(piece.get("shots") or []), "stills": len(piece.get("stills") or []),
    }


class MographArchive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographArchive",
            display_name="MOTNGRFX Archive",
            category=CATEGORY,
            description="Output node: saves the MOSS still, each shot clip (mp4) and each Alpha-Gen matte (PNG per frame) under "
                        "<filename_prefix>_shots/. Runs after the assemble; requests no matte that was not already made.",
            inputs=[
                io.String.Input("filename_prefix", default="archive/motngrfx"),
                MographPiece.Input("piece"),
                MographPiece.Input("plan_piece", optional=True),
                MographClips.Input("clips"),
                io.Image.Input("first"),
                io.Video.Input("video", tooltip="The assembled piece; only orders this node after the assemble."),
                io.Boolean.Input("light", default=False, optional=True, tooltip="True: only piece.json, index.json and still.png; no shot clips, no matte PNGs "
                                 "(the redesign graph: the footage already exists)."),
                io.Image.Input("places", optional=True, tooltip="MographFirstFrames batch: one still per place -> <stem>_still_<k>.png."),
                MographType.Input("type", optional=True, tooltip="The type passes compose read -> <stem>_type_<pass>.mp4 (+ LTX-read crops)."),
                io.Boolean.Input("full_archive", default=True, optional=True, tooltip="ON (default): also keep the heavy extras of the composition - the rendered 3D shot cuts + their matte PNGs, "
                                 "the composited shots (<stem>_sK_comp.mp4), the per-frame depth PNGs and the type matte PNGs. OFF: only the light set (clips, mattes, stills, previews, json)."),
                MographClips.Input("composed", optional=True, tooltip="MographCompose output: each composited shot before assemble -> <stem>_sK_comp.mp4 (full_archive)."),
                MographClips.Input("cuts", optional=True, tooltip="MographPivotCut output: the rendered 3D shot cuts ride on its clips -> <stem>_cut_<a>-<b>_<kind>.mp4 + matte PNGs."),
                io.Image.Input("sheet", optional=True, tooltip="MographDesignPackage 'sheet': the image the vision model was shown -> <stem>_design_sheet.png."),
                *[io.Image.Input(f"matte_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
                *[io.Image.Input(f"depth_{k}", optional=True, lazy=True) for k in range(1, MAX_SHOTS + 1)],
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, filename_prefix=None, piece=None, plan_piece=None, clips=None, first=None, video=None, **imgs) -> list[str]:
        need = []
        if imgs.get("light"):
            return need
        for clip in clips or []:
            sid = clip["sid"]
            name = f"matte_{sid[1:]}"
            if imgs.get(name) is None and _wired(cls, name) and (_needs_matte(piece, sid, clip.get("subject") is not None) or _needs_matte(plan_piece, sid, clip.get("subject") is not None)):
                need.append(name)
            dname = f"depth_{sid[1:]}"
            if imgs.get(dname) is None and _wired(cls, dname) and (_needs(piece, sid, "depth", clip.get("subject") is not None) or _needs(plan_piece, sid, "depth", clip.get("subject") is not None)):
                need.append(dname)
        for c in imgs.get("cuts") or []:                                 # a rendered 3D cut read these shots' depth (and matte): already evaluated for the cut, so no extra work
            for rec in c.get("cut_renders") or []:
                for sid in (rec["a"], rec["b"]):
                    for kind in ("depth", "matte"):
                        name = f"{kind}_{sid[1:]}"
                        if imgs.get(name) is None and _wired(cls, name) and name not in need:
                            need.append(name)
        return need

    @classmethod
    def execute(cls, filename_prefix: str, piece: dict, clips: list, first, video=None, plan_piece=None, light=False, places=None, type=None,
                full_archive=True, composed=None, cuts=None, sheet=None, **imgs) -> io.NodeOutput:
        t0 = time.time()
        base = Path(folder_paths.get_output_directory() if folder_paths is not None else ".")
        pre = str(filename_prefix).replace("\\", "/").strip("/")
        root = base / (pre + "_shots")
        root.mkdir(parents=True, exist_ok=True)
        stem = IN.run_stem(pre)
        legacy = IN.LEGACY_NAMES
        extras = AI.enabled() and not light
        heavy = AI.full_enabled(full_archive) and AI.enabled()      # full_archive: cut renders, composed shots, depth + type matte PNGs (INGREDIENTS.md)

        def keep(new_name: str, old_name: str) -> None:
            """the short (pre-2026-10-29) name of a prefixed file, as a hardlink: MographArchiveLoad / generations.py read it."""
            if legacy:
                AI.link_or_copy(root / new_name, root / old_name)

        info = {"piece": piece.get("id") or piece.get("slug"), "fps": FPS, "stem": stem, "full_archive": bool(heavy), "shots": {}, "files": []}
        if first is not None:
            f0 = (first[0].cpu().float().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            cv2.imwrite(str(root / IN.still(stem)), cv2.cvtColor(f0, cv2.COLOR_RGB2BGR))
            keep(IN.still(stem), "still.png")
            info["still"] = list(f0.shape[:2])
            info["files"].append(IN.still(stem))
        if light:                                                   # nothing but the shot list; the footage is somebody else's archive
            info["shots"] = {c["sid"]: {"frames": int(c["u8"].shape[0]), "start": int(c.get("start", 0))} for c in clips}
        for clip in ([] if light else clips):
            sid = clip["sid"]
            u8 = clip["u8"].numpy()
            rec = {"frames": int(u8.shape[0]), "clip_hw": [int(u8.shape[1]), int(u8.shape[2])], "start": int(clip.get("start", 0))}
            sub = clip.get("subject")
            if sub is None:
                n_ = IN.shot_file(stem, sid, "clip")
                fc.encode_mp4((u8[i].astype(np.float32) / 255.0 for i in range(u8.shape[0])), root / n_, fps=FPS, crf=14)     # encode_mp4 takes floats 0..1
                keep(n_, f"{sid}_clip.mp4")
                info["files"].append(n_)
            else:                                                   # three-layer shot: clean clip always, text-rendered background when one was made
                cl = sub.numpy() if hasattr(sub, "numpy") else np.asarray(sub)
                n_ = IN.shot_file(stem, sid, "clean")
                fc.encode_mp4((cl[i].astype(np.float32) / 255.0 for i in range(cl.shape[0])), root / n_, fps=FPS, crf=14)
                keep(n_, f"{sid}_clean.mp4")
                info["files"].append(n_)
                rec["layers"] = {"clean": f"{sid}_clean.mp4", "textbg": None}
                if clip.get("textbg"):
                    n_ = IN.shot_file(stem, sid, "textbg")
                    fc.encode_mp4((u8[i].astype(np.float32) / 255.0 for i in range(u8.shape[0])), root / n_, fps=FPS, crf=14)
                    keep(n_, f"{sid}_textbg.mp4")
                    info["files"].append(n_)
                    rec["layers"]["textbg"] = f"{sid}_textbg.mp4"
                if extras and clip.get("plate"):                    # clean-plate route: u8 is the plate, not a text pass
                    pl_ = AI.write_plate(root, stem, sid, clip)
                    info["files"] += pl_
                    if pl_:
                        rec["layers"]["plate"] = pl_[0]
            m = imgs.get(f"matte_{sid[1:]}")
            if m is not None:
                d = root / IN.matte_dir(stem, sid)
                d.mkdir(exist_ok=True)
                dl = root / f"{sid}_matte"
                if legacy:
                    dl.mkdir(exist_ok=True)
                arr = (m.cpu().float().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                for i in range(arr.shape[0]):
                    fn = IN.matte_frame(stem, sid, i + 1)
                    cv2.imwrite(str(d / fn), cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
                    if legacy:
                        AI.link_or_copy(d / fn, dl / f"{i + 1:05d}.png")
                rec["matte_frames"] = int(arr.shape[0])
                rec["matte_hw"] = [int(arr.shape[1]), int(arr.shape[2])]
                info["files"].append(IN.matte_dir(stem, sid) + "/")
            if extras:
                dep = imgs.get(f"depth_{sid[1:]}")
                info["files"] += AI.write_shot_previews(root, stem, sid, clip, depth=dep)
                info["files"] += AI.write_depth_stills(root, stem, sid, dep, heavy=heavy)
                plate = sub if sub is not None else clip["u8"]
                info["files"] += AI.write_contours(root, stem, sid, piece, plate, matte=m, depth=dep, full=heavy)
                info["files"] += AI.write_ltx_reads(root, stem, piece, clip, type)
                info["files"] += AI.write_sources(root, stem, sid, piece)
            info["shots"][sid] = rec
        if extras:
            info["files"] += AI.write_music(root, stem, video)
            info["files"] += AI.write_places(root, stem, places)
            info["files"] += AI.write_type_passes(root, stem, type)
            info["files"] += AI.write_places_meta(root, stem, piece)
            info["files"] += AI.write_cuts(root, stem, cuts, heavy=heavy)
            if heavy:
                info["files"] += AI.write_type_matte_png(root, stem, type)
        if AI.enabled():                                            # also on the light (redesign) archive: the sheet is cheap, the composed shots are THIS run's composition
            info["files"] += AI.write_design_sheet(root, stem, sheet)
            if heavy:
                info["files"] += AI.write_composed(root, stem, composed)
        info["bytes"] = AI.sizes(root, info["files"])               # per-composition estimate; 'heavy' = what full_archive off would leave out
        (root / IN.nm(stem, "index.json")).write_text(json.dumps(info, indent=1), encoding="utf-8")
        keep(IN.nm(stem, "index.json"), "index.json")
        summary = design_summary(piece)
        try:
            (root / IN.nm(stem, "piece.json")).write_text(json.dumps(piece, indent=1, default=str), encoding="utf-8")
            keep(IN.nm(stem, "piece.json"), "piece.json")
        except Exception as e:   # noqa: BLE001
            print(f"[MographArchive] piece.json not written: {e}")
        info["design"] = summary
        print(f"[MographArchive] {root} stem {stem} ({len(info['shots'])} shots, {sum(1 for r in info['shots'].values() if 'matte_frames' in r)} mattes, "
              f"{len(info['files'])} ingredient entries, {info['bytes']['total'] / 1e6:.1f} MB of which {info['bytes']['heavy'] / 1e6:.1f} MB full_archive extras) in {time.time() - t0:.1f}s")
        return io.NodeOutput(ui={"text": [json.dumps({"dir": pre + "_shots", **info}, default=str)]})


NODES = [MographArchive]
