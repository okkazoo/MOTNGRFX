#!/usr/bin/env python
"""box_stages: the heavy stages of piece_runner `--exec box`, run ON the rented GPU box (cwd = /workspace/mograph) or, for
`piece_runner.py --box-sim DIR`, as plain local subprocesses in DIR. Only stdlib at import time; everything else is imported by the
command that needs it, so `probe` works before the pip step.

    python -m tools.fusion.box_stages probe
    python -m tools.fusion.box_stages type      --spec type_spec.json --out OUT/type --passes matte,line --procs 10 [--sheet S.png]
    python -m tools.fusion.box_stages compose   --job job.json        # all shots (torch/CUDA backend) + assemble + encode
    python -m tools.fusion.box_stages final     --mp4 F.mp4 --sheet S.png [--strip T.png]
    python -m tools.fusion.box_stages masktrack --job job.json        # tracks + matte/corridor mp4 + footage copy + raw-only graph + region npy
    python -m tools.fusion.box_stages flow      --job job.json        # footage optical flow (RAFT CUDA) -> <sid>_flow.npy
    python -m tools.fusion.box_stages recomposite --job job.json      # original + upscaled raw decode blended by the footprint, uint8, chunked

Every command takes `--marker FILE`: on exit it writes {"ok": bool, "result"|"error": ..., "wall_s": ...} there (tmp + rename), which
the runner polls while the command runs under nohup. Result dicts are also printed as one JSON line on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIP_NAMES = {"numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python-headless", "PIL": "Pillow", "fontTools": "fonttools",
             "contourpy": "contourpy"}


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def ncores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ================================================================================================== probe
def nvenc_ok():
    """h264_nvenc is listed by `ffmpeg -encoders` AND a 1-frame encode works (driver / GPU present)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
        if "h264_nvenc" not in r.stdout:
            return False
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=24:d=0.1", "-frames:v", "1",
                            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True, timeout=60)
        return p.returncode == 0
    except Exception:       # noqa: BLE001
        return False


def cmd_probe(a):
    import importlib.util
    missing = [PIP_NAMES[m] for m in PIP_NAMES if importlib.util.find_spec(m) is None]
    info = dict(cores=ncores(), nvenc=nvenc_ok(), python=sys.executable, missing=missing)
    try:
        import torch
        info.update(torch=torch.__version__, cuda=bool(torch.cuda.is_available()),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:      # noqa: BLE001
        info.update(torch=None, cuda=False, gpu=None, torch_error=str(e)[:200])
    return info


# ================================================================================================== type passes
def cmd_type(a):
    from tools.fusion import type_passes
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    size = tuple(spec["size"])
    passes = a.passes.split(",")
    t0 = time.time()
    man = type_passes.render_passes(spec, a.out, size=size, passes=passes, procs=a.procs, overwrite=True, mp4=False, verbose=False)
    if a.sheet:
        type_passes.make_sheet(a.out, a.sheet, n=6, passes=[p for p in passes if p != "flow"][:4])
    return dict(nf=man["nf"], passes=passes, size=list(size), wall_s=round(time.time() - t0, 1), render_wall_s=man["timing"]["wall_s"],
                procs=man["timing"]["procs"], info=man["info"]["words"], notes=man["info"].get("notes", []))


# ================================================================================================== frame io
def ffprobe_size(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                   "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    return int(j["streams"][0]["width"]), int(j["streams"][0]["height"])


def decode_u8(path):
    """mp4 -> uint8 [T,H,W,3] numpy (ffmpeg pipe, native size)."""
    import numpy as np
    w, h = ffprobe_size(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)


def read_gray_seq(files, as_float=False):
    """PNG files -> [T,H,W] (uint8, or float32 0..1). Colour PNGs are reduced to luma, 16-bit to 0..1 (as compose.imread 'gray')."""
    import cv2
    import numpy as np
    out = []
    for f in files:
        im = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"cannot read image {f}")
        if im.dtype == np.uint16:
            im = (im.astype(np.float32) / 257.0).round().astype(np.uint8)
        if im.ndim == 3:
            im = im[..., 0] if im.shape[2] == 1 else (im[..., 2] * 0.2126 + im[..., 1] * 0.7152 + im[..., 0] * 0.0722).round().astype(np.uint8)
        out.append(im)
    arr = np.stack(out)
    return arr.astype(np.float32) / 255.0 if as_float else arr


# ================================================================================================== compose + assemble
def encoder_args(encoder, crf, draft, nvenc):
    use_nvenc = encoder == "nvenc" or (encoder == "auto" and nvenc)
    if use_nvenc and not nvenc:
        print("h264_nvenc requested but not usable here; falling back to libx264", file=sys.stderr)
        use_nvenc = False
    if use_nvenc:
        cq = int(crf) + 2 + (2 if draft else 0)            # crf 16 -> cq 18 (16_app/DESIGN.md)
        return dict(name="h264_nvenc", vcodec="h264_nvenc", vopts=["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"])
    pre = "veryfast" if draft else "medium"
    return dict(name="libx264", vcodec="libx264", vopts=["-preset", pre, "-crf", str(max(int(crf), 20) if draft else int(crf)), "-threads", "0"])


def others_for(piece, sid, shots_by, decode=None, read_seq=None):
    """`others=` of run_compose for shot `sid`: the neighbour shots' frames / mattes its compose ops read ($next, $prev, $shot:<id>; window cuts,
    tools/fusion/window_cut.py), decoded from the job's files. None when the shot reads no other shot. shots_by = {sid: job shot dict}."""
    from tools.fusion import window_cut
    decode, read_seq = decode or decode_u8, read_seq or read_gray_seq

    def sh(s_):
        if s_ not in shots_by:
            raise KeyError(f"compose {sid} reads shot {s_}, which is not in the job's shots ({', '.join(shots_by)})")
        return shots_by[s_]

    return window_cut.collect_others(piece, sid, lambda s_: decode(sh(s_)["footage"]),
                                     lambda s_: read_seq(sh(s_)["matte"]) if sh(s_).get("matte") else None)


def cmd_compose(a):
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    os.environ["FUSION_BACKEND"] = job.get("backend", "torch")        # before compose.py is imported (module-level backend)
    import numpy as np
    from tools.fusion import compose as fc
    from tools.fusion import recipe_eval as RE
    piece = json.loads(Path(job["piece"]).read_text(encoding="utf-8"))
    ana = json.loads(Path(job["analysis"]).read_text(encoding="utf-8"))
    curves = {k: v for k, v in ana["curves"].items() if isinstance(v, list)}
    cut = float(ana["cut"])
    draft = piece.get("quality") in ("draft", "preview")        # preview encodes like draft (veryfast)
    size = RE.compose_size(piece)
    fps = int(piece.get("fps", 24))
    lay, _ = RE.piece_layout(piece)
    clips, report = {}, dict(shots={}, backend=None)
    t_all = time.time()
    for sh in job["shots"]:
        sid = sh["sid"]
        t0 = time.time()
        footage = decode_u8(sh["footage"])
        t_dec = time.time() - t0
        matte = read_gray_seq(sh["matte"]) if sh.get("matte") else None
        depth = read_gray_seq(sh["depth"]) if sh.get("depth") else None
        flow = np.load(sh["flow"]) if sh.get("flow") else None            # [T,H/2,W/2,2] float16 (cmd_flow); flow_wake keeps its state per piece
        t_io = time.time() - t0
        u8, st = RE.run_compose(piece, sid, type_dir=job["type_dir"], footage=footage, matte=matte, depth=depth, curves=curves,
                                cut=cut, backend=job.get("backend", "torch"), flow=flow, others=others_for(piece, sid, {x["sid"]: x for x in job["shots"]}))
        clips[sid] = u8
        report["backend"] = st["backend"]
        report["shots"][sid] = dict(wall_s=round(time.time() - t0, 2), io_s=round(t_io, 2), decode_s=round(t_dec, 2), frames=st["n"],
                                    s_per_frame=round(st["s_per_frame"], 4), footage=sh["footage"])
        if st.get("hold_net"):                                   # the hold-legibility net (compose.hold_net): score + whether it fired, per hold
            report["shots"][sid]["hold_net"] = st["hold_net"]
            for hh in st["hold_net"]:
                print(f"hold_net {sid} {hh['hit']:.2f}-{hh['release']:.2f}s: mid score {hh['mid_score']} min {hh['min_score']} "
                      f"fired {hh['fired']} (max k {hh['max_k']}, {hh['fired_frames']}/{hh['frames']} frames)", flush=True)
        if st.get("legible_color"):                              # the legible-colour stage (tools/fusion/legible_color.py): measured bg, chosen colour, ratio per word
            report["shots"][sid]["legible_color"] = st["legible_color"]
            from tools.fusion import legible_color as LCOL
            print(f"legible_color {sid}:", flush=True)
            print(LCOL.format_table(st["legible_color"]), flush=True)
        print(f"compose {sid}: {st['n']} frames in {time.time() - t0:.1f}s ({st['s_per_frame']:.3f} s/frame, {st['backend']})", flush=True)
        del footage, matte, depth, flow
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:       # noqa: BLE001
        pass
    report["compose_wall_s"] = round(time.time() - t_all, 2)
    t1 = time.time()
    parts, cfl = RE.assemble_parts(piece)
    parts = [p for p in parts if p["sid"] in clips]
    enc = encoder_args(job.get("encoder", "auto"), job.get("crf", 16), draft, bool(job.get("nvenc")) if "nvenc" in job else nvenc_ok())
    name = enc.pop("name")
    sources = [RE.ClipSource(clips[p["sid"]], size) for p in parts]
    Path(job["out_mp4"]).parent.mkdir(parents=True, exist_ok=True)
    r = fc.assemble([dict(cut=p["cut"]) for p in parts], job["audio"], job["out_mp4"], crossfades=list(cfl)[:len(parts) - 1], fps=fps,
                    size=size, crf=int(job.get("crf", 16)), sources=sources, **enc)
    report["assemble"] = dict(wall_s=round(time.time() - t1, 2), encoder=name, frames=r["frames"], seconds=r["seconds"],
                              out=job["out_mp4"], size=list(size))
    report["wall_s"] = round(time.time() - t_all, 2)
    if job.get("report"):
        Path(job["report"]).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# ================================================================================================== footage flow
def cmd_flow(a):
    """tools/fusion/footage_flow.flow_for of a clip: RAFT on CUDA (torchvision raft_large), DIS fallback if RAFT cannot run. job: video, out (npy), model, batch."""
    import numpy as np
    from tools.fusion import footage_flow as FF
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    t0 = time.time()
    frames = FF.read_u8(job["video"])
    fallback = None
    try:
        fl = FF.flow_for(frames, method="auto", model=job.get("model", "raft_large"), batch=int(job.get("batch", 4)), verbose=True)
        method = "raft" if FF.torch is not None and FF.torch.cuda.is_available() and FF._have_raft() else "dis"
    except Exception as e:      # noqa: BLE001  (weights download / CUDA OOM): the CPU path still gives a usable field
        fallback = f"{type(e).__name__}: {e}"[:200]
        fl = FF.flow_for(frames, method="dis", device="cpu", verbose=True)
        method = "dis"
    Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(job["out"], fl)
    dt = time.time() - t0
    return dict(out=job["out"], shape=list(fl.shape), method=method, fallback=fallback, wall_s=round(dt, 1), s_per_frame=round(dt / max(len(frames), 1), 3))


# ================================================================================================== final: contact sheets
def make_sheets(mp4, sheet, strip=None, fps=24):
    import cv2
    import numpy as np
    from tools.fusion import compose
    fs = compose.FrameSource(mp4)
    n = len(fs)

    def tile(i, w):
        f = fs[i]
        im = cv2.cvtColor(np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{i / fps:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 60), 1, cv2.LINE_AA)
        return np.pad(im, ((0, 3), (0, 3), (0, 0)), constant_values=40)

    idx = np.unique(np.linspace(0, n - 1, 12).round().astype(int))
    tiles = [tile(int(i), 270) for i in idx]
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(len(tiles) // 6)])
    Path(sheet).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet), img)
    if strip:
        step = max(1, int(round(fps / 6)))
        t6 = [tile(i, 144) for i in range(0, n, step)]
        cols = 12
        while len(t6) % cols:
            t6.append(np.zeros_like(t6[0]))
        cv2.imwrite(str(strip), np.vstack([np.hstack(t6[r:r + cols]) for r in range(0, len(t6), cols)]))
    fs.close()
    return n


def cmd_final(a):
    t0 = time.time()
    n = make_sheets(a.mp4, a.sheet, a.strip, a.fps)
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration", "-of", "csv=p=0",
                         a.mp4], capture_output=True, text=True, check=True).stdout.strip().replace("\n", " | ")
    return dict(frames=n, probe=pr, wall_s=round(time.time() - t0, 1), sheet=a.sheet, strip=a.strip)


# ================================================================================================== masktrack prep
def cmd_masktrack(a):
    """tracks from the shot's Alpha-Gen matte + the LTX masked-Motion-Track graph (tools/fusion/ltx_mask.py).
    job: sid, id, style, start_frame (shot-local), frames, matte (png list), footage (mp4), input_dir, names{footage,matte,corridor},
    graph_out, region_out, prompt, denoise, corridor, seed, n_points, distance, dilate, feather, out_prefix, copy_footage."""
    import shutil
    import cv2
    import numpy as np
    from tools.fusion import ltx_mask
    mt = ltx_mask.mask_tracks
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    F = int(job["frames"])
    matte = read_gray_seq(job["matte"], as_float=True)
    w1, h1 = ltx_mask.STAGE1
    if matte.shape[2] != w1 or matte.shape[1] != h1:
        matte = np.stack([cv2.resize(m, (w1, h1), interpolation=cv2.INTER_LINEAR) for m in matte])
    if len(matte) < F:                                  # hold the last matte frame
        matte = np.concatenate([matte, np.repeat(matte[-1:], F - len(matte), 0)])
    matte = matte[:F]
    f0 = int(max(0, min(job["start_frame"], F - 2)))
    kw = dict(ltx_mask.TRACK_KW)
    kw.update(n_points=int(job.get("n_points", kw["n_points"])), distance=float(job.get("distance", kw["distance"])),
              seed=int(job.get("seed", kw["seed"])))
    tracks, meta = mt.build_tracks(matte, job["style"], start_frame=f0, **kw)
    inp = Path(job["input_dir"])
    inp.mkdir(parents=True, exist_ok=True)
    nm = job["names"]
    ltx_mask.write_gray_mp4(matte, str(inp / nm["matte"]))
    corridor = None
    if job.get("corridor"):
        ltx_mask.write_gray_mp4(mt.corridor_matte(tracks, ltx_mask.STAGE1, ltx_mask.CORRIDOR_RADIUS, ltx_mask.CORRIDOR_TRAIL),
                                str(inp / nm["corridor"]))
        corridor = nm["corridor"]
    if job.get("copy_footage"):
        shutil.copyfile(job["footage"], inp / nm["footage"])
    fw, fh = ffprobe_size(job["footage"])
    g = ltx_mask.build_graph(tracks, denoise=float(job.get("denoise", 0.8)), prompt=job["prompt"], style=job["style"],
                             seed=int(job.get("seed", 1)), dilate_cells=int(job.get("dilate", 1 if corridor else 2)),
                             feather=float(job.get("feather", 0.75)), frames=F, footage=nm["footage"], matte=nm["matte"],
                             corridor=corridor, out_prefix=job["out_prefix"], footage_size=(fw, fh), save_raw=True)
    # raw decode only: dropping the in-graph composite (SaveVideo 62 + CreateVideo 61 + recomposite chain 80-85) means ComfyUI never builds the
    # 1088x1920 float32 recomposite (it OOM-killed the 62 GB box); `recomposite` below does it in uint8 chunks
    for nid in ("61", "62", "80", "81", "82", "83", "84", "85"):
        g.pop(nid, None)
    bad = ltx_mask.check_graph(g)
    if bad:
        raise AssertionError("; ".join(bad))
    # the footprint the node will use, per pixel frame at matte resolution (12 px feather) -> uint8 npy for `recomposite`
    from tools.fusion import latent_mask as LM
    extra = mt.corridor_matte(tracks, ltx_mask.STAGE1, ltx_mask.CORRIDOR_RADIUS, ltx_mask.CORRIDOR_TRAIL) if corridor else None
    fp = LM.make_footprint(matte, LM.latent_frames(F), h1 // 32, w1 // 32, dilate_cells=int(job.get("dilate", 1 if corridor else 2)),
                           feather=float(job.get("feather", 0.75)), keep_first_frame=True, threshold=0.5, cell_threshold=0.0,
                           main_subject_only=True, min_blob_frac=0.25, extra=extra)
    region = LM.region_pixels(fp, F, h1, w1, 12.0)
    Path(job["region_out"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(job["region_out"], (np.clip(region, 0, 1) * 255 + 0.5).astype(np.uint8))
    Path(job["graph_out"]).write_text(json.dumps(g, indent=1), encoding="utf-8")
    meta.update(region=job["region_out"], region_mean=round(float(region.mean()), 4),start_frame_local=f0, graph=job["graph_out"], footage_size=[fw, fh], corridor=bool(corridor), denoise=job.get("denoise", 0.8))
    return meta


def _read_frames(path, w, h):
    """mp4 -> generator of uint8 [h,w,3] frames (ffmpeg pipe, one frame at a time)."""
    import numpy as np
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                return
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def cmd_recomposite(a):
    """original shot where the footprint is 0, the (upscaled) raw masked-Motion-Track decode where it is 1 - uint8, one frame at a time
    (the in-graph version held ~10 GB of float32 frames and OOM-killed ComfyUI). job: footage, raw, region (npy uint8 F,H,W), out_mp4, crf."""
    import cv2
    import numpy as np
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    fw, fh = ffprobe_size(job["footage"])
    rw, rh = ffprobe_size(job["raw"])
    region = np.load(job["region"], mmap_mode="r")
    Path(job["out_mp4"]).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}", "-framerate", "24", "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "fast", "-crf", str(job.get("crf", 12)), "-threads", "0", "-pix_fmt", "yuv420p",
                            "-r", "24", "-movflags", "+faststart", job["out_mp4"]], stdin=subprocess.PIPE)
    raw_it, last, n = _read_frames(job["raw"], rw, rh), None, 0
    try:
        for i, f in enumerate(_read_frames(job["footage"], fw, fh)):
            r = next(raw_it, None)
            last = r if r is not None else last
            if last is None:
                raise RuntimeError("raw decode has no frames")
            up = cv2.resize(last, (fw, fh), interpolation=cv2.INTER_LANCZOS4) if (rw, rh) != (fw, fh) else last
            al = cv2.resize(np.asarray(region[min(i, len(region) - 1)]), (fw, fh), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
            out = f.astype(np.float32) * (1.0 - al) + up.astype(np.float32) * al
            enc.stdin.write(np.clip(out + 0.5, 0, 255).astype(np.uint8).tobytes())
            n += 1
    finally:
        enc.stdin.close()
        rc = enc.wait()
    if rc:
        raise RuntimeError("ffmpeg encode failed")
    return dict(frames=n, out=job["out_mp4"], size=[fw, fh], raw_size=[rw, rh], region_mean=round(float(np.asarray(region[::8]).mean()) / 255.0, 4))


# ================================================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    t = sub.add_parser("type")
    t.add_argument("--spec", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--passes", required=True)
    t.add_argument("--procs", type=int, required=True)
    t.add_argument("--sheet")
    c = sub.add_parser("compose")
    c.add_argument("--job", required=True)
    f = sub.add_parser("final")
    f.add_argument("--mp4", required=True)
    f.add_argument("--sheet", required=True)
    f.add_argument("--strip")
    f.add_argument("--fps", type=int, default=24)
    m = sub.add_parser("masktrack")
    m.add_argument("--job", required=True)
    r_ = sub.add_parser("recomposite")
    r_.add_argument("--job", required=True)
    fl_ = sub.add_parser("flow")
    fl_.add_argument("--job", required=True)
    for p in (t, c, f, m, r_, fl_, sub.choices["probe"]):
        p.add_argument("--marker")
    a = ap.parse_args(argv)
    t0 = time.time()
    rc, out = 0, None
    try:
        out = dict(ok=True, result={"probe": cmd_probe, "type": cmd_type, "compose": cmd_compose, "final": cmd_final,
                                    "masktrack": cmd_masktrack, "recomposite": cmd_recomposite,
                                    "flow": cmd_flow}[a.cmd](a))
    except BaseException as e:      # noqa: BLE001
        traceback.print_exc()
        out, rc = dict(ok=False, error=f"{type(e).__name__}: {e}"[:600]), 1
    out["wall_s"] = round(time.time() - t0, 1)
    txt = json.dumps(out, default=str)
    if a.marker:
        tmp = Path(a.marker + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(txt, encoding="utf-8")
        os.replace(tmp, a.marker)
    print(txt, flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
