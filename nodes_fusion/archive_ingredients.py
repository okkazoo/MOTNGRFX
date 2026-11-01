"""The extra ingredients MographArchive writes next to the shot clips (16_app/INGREDIENTS.md). Every function is best effort: it returns the list of file
names it wrote (relative to the archive dir) and never raises (one failed preview must not lose the archive). All names carry `<stem>_`
(tools/fusion/ingredient_names.py). Set MOTNGRFX_ARCHIVE_EXTRAS=0 to skip the whole extras step.

    write_music(video)            -> <stem>_music.wav          the mix of the assembled piece (from the VIDEO's audio)
    write_places(places)          -> <stem>_still_<k>.png      one still per place (MographFirstFrames batch)
    write_type_passes(type_t)     -> <stem>_type_<pass>.mp4    beauty / matte / depth / line / clay / flow of the type render compose read (dir_layers)
    write_shot_previews(...)      -> <stem>_<sid>_depth.mp4, <stem>_<sid>_flow.mp4 (cheap encodes)
    write_contours(...)           -> <stem>_<sid>_contour_<kind>.mp4 + .npz   the footage contours the type words use as entry / exit / via ports
    write_ltx_reads(...)          -> <stem>_ltx_read_<sid>_<j>.png + <stem>_ltx_reads.json

Whole-composition set (2026-10-29, "keep everything of a finished video"). The HEAVY ones (marked *) are written only with full_archive on (MographArchive input,
default ON; MOTNGRFX_ARCHIVE_FULL=0 forces it off); the rest with every extras run:
    write_cuts(...)               -> <stem>_cut_<a>-<b>_<kind>.mp4 *, <stem>_cut_<a>-<b>_matte/<..._matte_00001.png> *, <stem>_cut_<a>-<b>_<kind>.json   the 3D shot cuts as rendered
    write_depth_stills(...)       -> <stem>_<sid>_depth.png (the first frame the cuts / parallax use), <stem>_<sid>_depth/<..._00001.png> *   (PNG, never lossy)
    write_plate(...)              -> <stem>_<sid>_plate.mp4     the clean plate of a clean-plate shot (clip['plate'])
    write_composed(...)           -> <stem>_<sid>_comp.mp4 *    the composited shot as compose made it, before assemble
    write_sources(...)            -> <stem>_<sid>_sources.json  what the shot's compose ops read from OTHER shots (window_other) + which archive files hold it
    write_type_matte_png(...)     -> <stem>_type_matte/<stem>_type_matte_00001.png *   the type matte pass as PNG (the mp4 of write_type_passes is lossy)
    write_design_sheet(...)       -> <stem>_design_sheet.png    the sheet image the vision model was shown
    write_places_meta(...)        -> <stem>_places.json         per place: id, fresh prompt, angle-still source + edit prompt
    sizes(...)                    -> bytes written, total and heavy (index.json 'bytes')
"""
from __future__ import annotations

import json
import os
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import ensure_root_on_path
from ._types import FPS

ensure_root_on_path()
from tools.fusion import ingredient_names as N        # noqa: E402


def enabled() -> bool:
    return os.environ.get("MOTNGRFX_ARCHIVE_EXTRAS", "1") != "0"


def _log(what: str, e: Exception) -> None:
    print(f"[MographArchive] {what} not written: {type(e).__name__}: {e}")
    if os.environ.get("MOTNGRFX_ARCHIVE_DEBUG"):
        traceback.print_exc()


def link_or_copy(src, dst) -> None:
    """hardlink src at dst (same bytes, no extra disk); copy when the filesystem refuses."""
    src, dst = str(src), str(dst)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _u8(x) -> np.ndarray:
    a = _np(x)
    return a if a.dtype == np.uint8 else (a.astype(np.float32).clip(0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------------------------- music / stills
def write_music(root: Path, stem: str, video) -> list[str]:
    try:
        from .compose import write_wav
        comp = video.get_components()
        audio = getattr(comp, "audio", None)
        if audio is None:
            return []
        name = N.nm(stem, "music.wav")
        write_wav(audio, root / name)
        return [name]
    except Exception as e:   # noqa: BLE001
        _log("music wav", e)
        return []


def write_places(root: Path, stem: str, places) -> list[str]:
    out = []
    try:
        if places is None:
            return out
        arr = _u8(places)
        for k in range(arr.shape[0]):
            name = N.still(stem, k + 1)
            cv2.imwrite(str(root / name), cv2.cvtColor(arr[k][..., :3], cv2.COLOR_RGB2BGR))
            out.append(name)
    except Exception as e:   # noqa: BLE001
        _log("place stills", e)
    return out


# ----------------------------------------------------------------------------------------------- type passes
def _ffmpeg_seq(pattern: str, out_mp4: Path, fps: int, crf: int = 20) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-start_number", "0", "-i", pattern, "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-400:])


def write_type_passes(root: Path, stem: str, type_t) -> list[str]:
    """each pass of the type render the compose read (dir_layers, final size) as an mp4: the graphic-design animation by itself."""
    out = []
    try:
        if not type_t:
            return out
        d = Path(type_t["dir_layers"])
        jobs = []
        for p in N.TYPE_PASSES:
            sub = "flow_vis" if p == "flow" else p
            pat = d / sub / f"{sub}_%05d.png"
            if (d / sub / f"{sub}_00000.png").is_file():
                jobs.append((p, str(pat)))
        def one(j):
            name = N.type_pass_file(stem, j[0])
            try:
                _ffmpeg_seq(j[1], root / name, FPS)
                return name
            except Exception as e:   # noqa: BLE001
                _log(f"type pass {j[0]}", e)
                return None
        with ThreadPoolExecutor(max_workers=3) as ex:
            out = [n for n in ex.map(one, jobs) if n]
        mp = d / "manifest.json"
        if mp.is_file():
            name = N.nm(stem, "type_manifest.json")
            link_or_copy(mp, root / name)
            out.append(name)
    except Exception as e:   # noqa: BLE001
        _log("type passes", e)
    return out


# ----------------------------------------------------------------------------------------------- per shot previews
def _enc(frames_rgb_u8, path: Path, fps: int = FPS, crf: int = 22) -> None:
    from tools.fusion import compose as fc
    fc.encode_mp4((f.astype(np.float32) / 255.0 for f in frames_rgb_u8), path, fps=fps, crf=crf, preset="veryfast")


def write_shot_previews(root: Path, stem: str, sid: str, clip: dict, depth=None) -> list[str]:
    out = []
    try:
        if depth is not None:
            a = _u8(depth)
            if a.ndim == 3:
                a = a[..., None]
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            name = N.shot_file(stem, sid, "depth")
            _enc((a[i][..., :3] for i in range(a.shape[0])), root / name)
            out.append(name)
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} depth preview", e)
    try:
        fl = clip.get("flow")
        if fl is not None:
            from tools.fusion import flow as FL
            f = _np(fl).astype(np.float32)
            ref = max(float(np.percentile(np.hypot(f[..., 0], f[..., 1]), 99)), 1e-3)
            name = N.shot_file(stem, sid, "flow")
            _enc((cv2.cvtColor(FL.colour_wheel(f[i], ref), cv2.COLOR_BGR2RGB) for i in range(f.shape[0])), root / name)
            out.append(name)
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} flow preview", e)
    return out


# ----------------------------------------------------------------------------------------------- contours
def contour_kinds_used(piece: dict, full: bool = False) -> list[str]:
    """the contour port kinds the designed piece uses: word entry / exit / via ports (contour_ports.split_port_spec) and edge_run_type sources. `full`: a piece with a
    `handoff` word (type carried onto the footage's contour) that names no port gets matte_outline + luma_edges traced anyway (the two hand-off traces)."""
    kinds: list[str] = []

    def add(k):
        if k in N.CONTOUR_KINDS and k not in kinds:
            kinds.append(k)
    try:
        from mograph.modules import contour_ports as CP
        for w in ((piece.get("type") or {}).get("words")) or []:
            _, _, ports = CP.split_port_spec(w)
            for key in ("entry", "exit", "via"):
                if ports.get(key):
                    add(ports[key].get("port"))
    except Exception as e:   # noqa: BLE001
        _log("contour kinds", e)
    comp = piece.get("compose") or {}
    chains = [comp.get("default") or {}] + [v for v in (comp.get("shots") or {}).values() if isinstance(v, dict)]
    for ch in chains:
        for o in ch.get("ops") or []:
            if isinstance(o, dict) and o.get("op") == "edge_run_type":
                add((o.get("args") or {}).get("source") or "luma_edges")
    if full and not kinds and any(isinstance(w, dict) and str(w.get("render")) == "handoff" for w in ((piece.get("type") or {}).get("words")) or []):
        for k in ("matte_outline", "luma_edges"):
            add(k)
    return kinds


def write_contours(root: Path, stem: str, sid: str, piece: dict, plate_u8, matte=None, depth=None, max_side: int = 576, full: bool = False) -> list[str]:
    """track the contour kinds the piece uses on this shot and save them: an overlay mp4 (lines over the plate) and the raw tracks (.npz: P [F,L,N,2] in
    plate px of `size`, valid, closed). Kinds that need a channel the shot lacks (no matte / depth) are skipped."""
    out = []
    kinds = contour_kinds_used(piece, full)
    if not kinds:
        return out
    try:
        from mograph.modules import contour_ports as CP
        plate = _np(plate_u8)
        h, w = plate.shape[1:3]
        sc = min(1.0, max_side / max(h, w))
        size = (max(2, int(round(w * sc / 2)) * 2), max(2, int(round(h * sc / 2)) * 2))

        def fit(a, gray):
            fr = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in _u8(a)]
            fr = np.stack(fr)
            return fr[..., 0] if (gray and fr.ndim == 4) else fr
        data = CP.PlateData(plate=fit(plate, False), matte=None if matte is None else fit(matte, True),
                            depth=None if depth is None else fit(depth, True), fps=FPS)
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} contour data", e)
        return out
    for kind in kinds:
        try:
            if not data.has(kind):
                continue
            tr = CP.build_track(data, kind)
            frames = []
            for f in range(tr.F):
                im = np.ascontiguousarray(data.plate[min(f, data.n - 1)]).copy()
                im = (im * 0.45).astype(np.uint8)
                for li in tr.valid_lines(f):
                    pts = np.round(tr.P[f, li]).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(im, [pts], bool(tr.closed[li]), (255, 220, 60), 2, cv2.LINE_AA)
                frames.append(im)
            mp4, npz = N.contour_file(stem, sid, kind), N.contour_file(stem, sid, kind, "npz")
            _enc(frames, root / mp4)
            np.savez_compressed(root / npz, P=tr.P, valid=tr.valid, closed=tr.closed, size=np.array(size), fps=np.array(FPS))
            out += [mp4, npz]
        except Exception as e:   # noqa: BLE001
            _log(f"{sid} contour {kind}", e)
    return out


# ----------------------------------------------------------------------------------------------- LTX read crops
def write_ltx_reads(root: Path, stem: str, piece: dict, clip: dict, type_t) -> list[str]:
    out = []
    reads = clip.get("ltx_reads")
    if not reads:
        return out
    sid = clip["sid"]
    try:
        if type_t and clip.get("textbg"):
            from tools.fusion import ltx_check as LC
            jobs = LC.crop_jobs(piece, sid, type_t["dir_layers"], clip["u8"])
            for j, job in enumerate(jobs, 1):
                name = N.ltx_read_crop(stem, sid, j)
                cv2.imwrite(str(root / name), cv2.cvtColor(np.ascontiguousarray(job["image"]), cv2.COLOR_RGB2BGR))
                out.append(name)
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} ltx read crops", e)
    try:
        p = root / N.nm(stem, "ltx_reads.json")
        old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        old[sid] = reads
        p.write_text(json.dumps(old, indent=1, default=str), encoding="utf-8")
        if p.name not in out:
            out.append(p.name)
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} ltx_reads.json", e)
    return out


# ----------------------------------------------------------------------------------------------- whole-composition set
def full_enabled(flag=True) -> bool:
    """full_archive input (default True) AND the environment switch MOTNGRFX_ARCHIVE_FULL (0 = off)."""
    return bool(flag) and os.environ.get("MOTNGRFX_ARCHIVE_FULL", "1") != "0"


def _png(path: Path, a: np.ndarray) -> None:
    """RGB / gray uint8 -> PNG (fast compression: these are archive frames, not deliverables)."""
    a = np.ascontiguousarray(a)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), a, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"cv2.imwrite failed: {path}")


def _gray_frames(x) -> np.ndarray:
    """depth / matte tensor [T,H,W] | [T,H,W,C] float 0..1 or uint8 -> uint8 [T,H,W]."""
    a = _u8(x)
    if a.ndim == 4:
        a = a[..., 0]
    return a[None] if a.ndim == 2 else a


def write_cuts(root: Path, stem: str, cut_clips, heavy: bool = True) -> list[str]:
    """the 3D shot cuts (MographPivotCut: clip['cut_renders'] = pivot_cut.cut_record): the transition frames as ONE mp4 per cut, the subject matte through
    the transition as a PNG folder (stack member), and the cut's plan + resolved params as json. Rides on clips from the pivot node (archive input `cuts`)."""
    out, seen = [], set()
    try:
        from tools.fusion import compose as fc
        for clip in cut_clips or []:
            for rec in clip.get("cut_renders") or []:
                key = (rec["a"], rec["b"], rec["kind"])
                if key in seen:
                    continue
                seen.add(key)
                a, b, kind = rec["a"], rec["b"], rec["kind"]
                try:
                    meta = {k: rec.get(k) for k in ("a", "b", "kind", "n", "i2", "params", "window_a", "window_b", "info")}
                    meta["matte_through"] = rec.get("matte") is not None
                    meta["depth"] = (rec.get("info") or {}).get("depth")
                    jn = N.cut_file(stem, a, b, kind, "json")
                    (root / jn).write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
                    out.append(jn)
                    if not heavy:
                        continue
                    fa, fb = (_np(x) for x in rec["frames"])
                    mp = N.cut_file(stem, a, b, kind)
                    fc.encode_mp4((f.astype(np.float32) / 255.0 for f in np.concatenate([fa, fb])), root / mp, fps=FPS, crf=18, preset="veryfast")
                    out.append(mp)
                    if rec.get("matte") is not None:
                        d = root / N.cut_matte_dir(stem, a, b)
                        d.mkdir(exist_ok=True)
                        i = 0
                        for part in rec["matte"]:
                            for m in _gray_frames(np.clip(_np(part), 0.0, 1.0)):
                                i += 1
                                _png(d / N.cut_matte_frame(stem, a, b, i), m)
                        out.append(N.cut_matte_dir(stem, a, b) + "/")
                except Exception as e:   # noqa: BLE001
                    _log(f"cut {a}-{b}", e)
    except Exception as e:   # noqa: BLE001
        _log("cuts", e)
    return out


def write_depth_stills(root: Path, stem: str, sid: str, depth, heavy: bool = True) -> list[str]:
    """depth of a shot as PNG: the FIRST frame (what the pivot / orbit / dolly cuts and the parallax read; `<stem>_s1_depth.png`) always, the whole per-frame
    sequence (`<stem>_s1_depth/<stem>_s1_depth_00001.png`) with heavy. 8-bit gray, bright = near as the depth model made it."""
    out = []
    if depth is None:
        return out
    try:
        g = _gray_frames(depth)
        name = N.shot_file(stem, sid, "depth", "png")
        _png(root / name, g[0])
        out.append(name)
        if heavy and g.shape[0] > 1:
            d = root / N.depth_dir(stem, sid)
            d.mkdir(exist_ok=True)
            for i in range(g.shape[0]):
                _png(d / N.depth_frame(stem, sid, i + 1), g[i])
            out.append(N.depth_dir(stem, sid) + "/")
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} depth png", e)
    return out


def write_plate(root: Path, stem: str, sid: str, clip: dict) -> list[str]:
    """the CLEAN PLATE of a clean-plate shot (MographLayerGather: clip['plate'] True, u8 = the plate, subject = the clean clip)."""
    try:
        if clip.get("plate") and clip.get("subject") is not None:
            u8 = _np(clip["u8"])[..., :3]
            name = N.shot_file(stem, sid, "plate")
            _enc((u8[i] for i in range(u8.shape[0])), root / name, crf=18)
            return [name]
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} clean plate", e)
    return []


def write_composed(root: Path, stem: str, composed) -> list[str]:
    """every shot as MographCompose made it (type woven in, graded), before assemble: `<stem>_sK_comp.mp4` (compose size, crf 20 veryfast)."""
    out = []
    try:
        for c in composed or []:
            try:
                u8 = _np(c["u8"])[..., :3]
                name = N.shot_file(stem, c["sid"], "comp")
                _enc((u8[i] for i in range(u8.shape[0])), root / name, crf=20)
                out.append(name)
            except Exception as e:   # noqa: BLE001
                _log(f"{c.get('sid')} composed shot", e)
    except Exception as e:   # noqa: BLE001
        _log("composed shots", e)
    return out


def write_sources(root: Path, stem: str, sid: str, piece: dict) -> list[str]:
    """what shot `sid`'s compose ops read from OTHER shots (window_other / $next / $prev / $shot:<id>), as json: [{ref, shot, kind, file}] with the archive file that holds it
    (footage -> that shot's clip, subject -> its clean clip, matte -> its matte folder). Nothing is written for a shot that reads none."""
    try:
        if not (piece or {}).get("shots"):
            return []
        from tools.fusion import window_cut
        refs = window_cut.other_refs(piece, sid)
        if not refs:
            return []
        rows = []
        for ref, tgt, kind in refs:
            f = (N.matte_dir(stem, tgt) + "/" if kind == "matte" else N.shot_file(stem, tgt, "clean" if kind == "subject" else "clip"))
            rows.append({"ref": ref, "shot": tgt, "kind": kind, "file": f})
        name = N.nm(stem, f"{sid}_sources.json")
        (root / name).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return [name]
    except Exception as e:   # noqa: BLE001
        _log(f"{sid} window sources", e)
        return []


def write_type_matte_png(root: Path, stem: str, type_t) -> list[str]:
    """the type MATTE pass as a PNG sequence (hardlinked from the type render's own PNGs when the volume allows, else copied; renumbered from 1). A matte is never kept
    lossy: the `_type_matte.mp4` of write_type_passes is only the preview."""
    try:
        if not type_t:
            return []
        src = sorted((Path(type_t["dir_layers"]) / "matte").glob("matte_*.png"))
        if not src:
            return []
        d = root / N.type_matte_dir(stem)
        d.mkdir(exist_ok=True)
        for i, f in enumerate(src, 1):
            link_or_copy(f, d / N.type_matte_frame(stem, i))
        return [N.type_matte_dir(stem) + "/"]
    except Exception as e:   # noqa: BLE001
        _log("type matte png", e)
        return []


def write_design_sheet(root: Path, stem: str, sheet) -> list[str]:
    """the ONE sheet image the vision model was shown for the design (MographDesignPackage 'sheet'; a ComfyUI temp preview otherwise)."""
    try:
        if sheet is None:
            return []
        a = _u8(sheet)
        a = a[0] if a.ndim == 4 else a
        name = N.nm(stem, "design_sheet.png")
        _png(root / name, a[..., :3])
        return [name]
    except Exception as e:   # noqa: BLE001
        _log("design sheet", e)
        return []


def write_places_meta(root: Path, stem: str, piece: dict) -> list[str]:
    """`<stem>_places.json`: per place (row k of the places batch = `<stem>_still_<k>.png`) its id, fresh prompt and, for an angle place, the source place + the anchored edit
    prompt (tools/fusion/places.py angle_*). Whether the edit really ran depends on the graph's `angles` switch: when it did, `_still_<k>.png` IS the edited still and the
    source's original is `_still_<j>.png`."""
    try:
        stills = (piece or {}).get("stills") or []
        if not stills:
            return []
        from tools.fusion import places as PL
        rows = []
        for i, st in enumerate(stills[:PL.MAX_STILLS]):
            j = PL.angle_source(piece, i)
            rows.append({"place": i + 1, "id": st.get("id"), "file": N.still(stem, i + 1), "prompt": st.get("prompt"), "angle_of": st.get("angle_of"),
                         "angle_source_place": None if j is None else j + 1, "angle_source_file": None if j is None else N.still(stem, j + 1),
                         "angle_edit_prompt": PL.angle_prompt(piece, i) if j is not None else None})
        name = N.nm(stem, "places.json")
        (root / name).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        return [name]
    except Exception as e:   # noqa: BLE001
        _log("places json", e)
        return []


def sizes(root: Path, names) -> dict:
    """bytes of the listed archive entries (folders recursive): {'total', 'heavy', 'files'}: the per-composition estimate in index.json; `heavy` = what full_archive=False would not write."""
    tot = hv = nf = 0
    for n in names:
        p = root / n.rstrip("/")
        files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else ([p] if p.is_file() else [])
        b = sum(f.stat().st_size for f in files)
        tot += b
        nf += len(files)
        if N.is_heavy(n):
            hv += b
    return {"total": tot, "heavy": hv, "files": nf}
