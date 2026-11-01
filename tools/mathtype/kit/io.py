"""Output plumbing: frame-batch render loop, encoder (h264_nvenc on CUDA, else libx264), score.py + mux,
generation numbering (gens.py), TIMINGS.csv rows, stills, contact sheets and strips, and `main(piece)`,
the whole CLI for a piece.

A piece is any object with:
    test, stem          names for GENERATIONS / files (stem_gNN_draft.mp4)
    version_dir         11_mathtype/vNNN folder (local/ and review/ are created)
    timeline            kit.timeline.Timeline (dur, fps, landings, read windows)
    frame(t, W, H)      -> uint8 (H, W, 3) or (H, W)
optional: frames(ts, W, H) (batched), be (xp.Backend), palette (kit.palette.Palette), score_preset ("E4"),
    code_files (paths hashed for the generation id), text_mask(i, W, H) (bool glyph mask for QC),
    sizes = {"draft": (360, 450), "final": (1080, 1350)}, notes (str for TIMINGS).
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import gens

from .core import cuda, device_label

MATHTYPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING_HEADER = ("gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,per_frame_ms,"
                 "encode_s,audio_s,notes")


# ------------------------------------------------------------------------------------------- time / folders
def nz_now() -> datetime.datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("Pacific/Auckland"))
    except Exception:  # pragma: no cover
        return datetime.datetime.now()


def dirs(version_dir: str):
    local, review = os.path.join(version_dir, "local"), os.path.join(version_dir, "review")
    os.makedirs(local, exist_ok=True)
    os.makedirs(review, exist_ok=True)
    return local, review


def _ts(d):
    return "" if not d else (d if isinstance(d, str) else d.strftime("%Y-%m-%d %H:%M"))


def timing_row(local: str, gen, test, kind, device, frames="", res="", started=None, finished=None,
               make_min="", render_s=None, encode_s=None, audio_s=None, notes=""):
    """Append one measured row to local/TIMINGS.csv (header created if missing)."""
    p = os.path.join(local, "TIMINGS.csv")
    fmt = lambda v, f="{:.1f}": "" if v is None or v == "" else f.format(v)  # noqa: E731
    pf = (render_s * 1000 / frames) if (render_s is not None and frames) else None
    row = [f"g{gen:02d}" if isinstance(gen, int) else str(gen), test, kind, device, str(frames), res,
           _ts(started), _ts(finished),
           fmt(make_min), fmt(render_s), fmt(pf, "{:.0f}"), fmt(encode_s), fmt(audio_s), notes]
    new = not os.path.exists(p)
    with open(p, "a", encoding="utf-8") as f:
        if new:
            f.write(TIMING_HEADER + "\n")
        f.write(",".join(str(c).replace(",", ";") for c in row) + "\n")


def make_minutes(work_start: str | None, fin: datetime.datetime):
    """--work-start 'YYYY-MM-DD HH:MM' or 'HH:MM' (NZ, today / yesterday) -> minutes until fin."""
    if not work_start:
        return None
    if len(work_start) <= 5:
        hh, mm = map(int, work_start.split(":"))
        s0 = fin.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if s0 > fin:
            s0 -= datetime.timedelta(days=1)
    else:
        s0 = datetime.datetime.strptime(work_start, "%Y-%m-%d %H:%M").replace(tzinfo=fin.tzinfo)
    return (fin - s0).total_seconds() / 60


# ------------------------------------------------------------------------------------------- encode
def encoder_args(use_nvenc: bool, crf: int = 16):
    if use_nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


class Encoder:
    """ffmpeg rawvideo pipe; never overwrites (-n). Times the blocked writes as encode time."""

    def __init__(self, path, W, H, fps, rgb=True, nvenc=False, crf=16):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        cmd = (["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24" if rgb else "gray",
                "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"] + encoder_args(nvenc, crf) + [path])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.enc = 0.0

    def write(self, fr):
        t0 = time.perf_counter()
        self.p.stdin.write(np.ascontiguousarray(fr).tobytes())
        self.enc += time.perf_counter() - t0

    def close(self):
        t0 = time.perf_counter()
        self.p.stdin.close()
        rc = self.p.wait()
        self.enc += time.perf_counter() - t0
        if rc:
            raise RuntimeError(f"ffmpeg exited {rc} for {self.path}")
        return self.enc


def render(frame_fn, path, W, H, nframes, fps=25, batch=1, frames_fn=None, nvenc=False, sync=None, crf=16,
           log=True):
    """Render nframes (t = n / fps) and encode. frames_fn(ts, W, H) renders a batch; else frame_fn(t, W, H).
    Warm-up frame 0 first (setup, untimed). Returns dict(frames, render_s, encode_s)."""
    first = frame_fn(0.0, W, H) if frames_fn is None else frames_fn([0.0], W, H)[0]
    rgb = first.ndim == 3
    enc = Encoder(path, W, H, fps, rgb=rgb, nvenc=nvenc, crf=crf)
    rend = 0.0
    for b0 in range(0, nframes, batch):
        ts = [n / fps for n in range(b0, min(b0 + batch, nframes))]
        t0 = time.perf_counter()
        frs = frames_fn(ts, W, H) if frames_fn is not None else [frame_fn(t, W, H) for t in ts]
        if sync:
            sync()
        rend += time.perf_counter() - t0
        for fr in frs:
            enc.write(fr)
    es = enc.close()
    if log:
        print(f"{path}  {W}x{H}  {nframes} frames  render {rend:.1f}s ({rend * 1000 / nframes:.0f} ms/frame)  "
              f"encode {es:.1f}s", flush=True)
    return dict(frames=nframes, render_s=rend, encode_s=es)


def score(preset, duration, changes, out_wav, timings=None, note="", be=None, seed=None):
    """score.py preset -> wav (skips if it exists). Returns wall seconds (0 if skipped)."""
    if os.path.exists(out_wav):
        return 0.0
    cmd = [sys.executable, os.path.join(MATHTYPE, "score.py"), "--preset", preset, "--duration", f"{duration:g}",
           "--changes", ",".join(f"{x:.3f}" for x in changes), "--out", out_wav]
    if timings:
        cmd += ["--timings", timings, "--note", note]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if be is not None and be.torch:
        cmd += ["--backend", "torch", "--device", str(be.device)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def mux(video, wav, out):
    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", out], check=True)
    return out


# ------------------------------------------------------------------------------------------- review images
def contact_sheet(video, out, fps=5, width=144, cols=10, rows=None, dur=None):
    """ffmpeg tile sheet of the video (default 5 fps, 144 px tiles, 10 columns, rows to fit dur)."""
    if os.path.exists(out):
        return out
    if rows is None:
        rows = int(np.ceil((dur or 10.0) * fps / cols))
    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", video, "-vf",
                    f"fps={fps},scale={width}:-2,tile={cols}x{rows}", "-frames:v", "1", out], check=True)
    return out


def strip(frame_fn, times, W, H, out, quality=90):
    """Frames at `times` side by side (rendered at W x H)."""
    ims = []
    for t in times:
        fr = frame_fn(t, W, H)
        ims.append(fr if fr.ndim == 3 else np.repeat(fr[..., None], 3, -1))
    Image.fromarray(np.concatenate(ims, axis=1)).save(out, quality=quality)
    return out


def still(frame_fn, t, W, H, out, max_w=540, quality=92):
    fr = frame_fn(t, W, H)
    im = Image.fromarray(fr)
    if im.width > max_w:
        im = im.resize((max_w, int(round(im.height * max_w / im.width))), Image.LANCZOS)
    im.save(out, quality=quality)
    return out


# ------------------------------------------------------------------------------------------- piece CLI
def _next_free(local, stem, nn, kind):
    while os.path.exists(os.path.join(local, gens.video_name(stem, nn, kind))):
        nn += 1
    return nn


def run(piece, kind, a, be):
    """One draft / final generation of a piece: gens number, render, TIMINGS, stills, sheet, QC, audio."""
    from . import qc as _qc
    local, review = dirs(piece.version_dir)
    tl = piece.timeline
    sizes = getattr(piece, "sizes", {"draft": (360, 450), "final": (1080, 1350)})
    W, H = (a.size, int(round(a.size * sizes[kind][1] / sizes[kind][0]))) if a.size else sizes[kind]
    files = list(getattr(piece, "code_files", [])) or [sys.modules[type(piece).__module__].__file__]
    h = gens.code_hash(files + [os.path.join(os.path.dirname(__file__), f) for f in sorted(os.listdir(os.path.dirname(__file__))) if f.endswith(".py")])
    h += "-" + getattr(piece, "variant_key", "")
    started = nz_now()
    nn = gens.gen_for(local, piece.test, piece.stem, kind, h, a.parent, a.trigger, a.change)
    name = gens.video_name(piece.stem, nn, kind)
    path = os.path.join(local, name)
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; never overwrite")
    nfr = a.frames or tl.nframes
    tm = render(piece.frame, path, W, H, nfr, tl.fps, batch=a.batch,
                frames_fn=getattr(piece, "frames", None) if a.batch > 1 else None,
                nvenc=cuda(be) and not a.x264, sync=be.sync)
    fin = nz_now()
    mk = make_minutes(a.work_start, fin)
    dev = device_label(be)
    stills = []
    mids = [0.5 * (t0 + t1) for t0, t1 in tl.trans[:1]] + tl.read_moments()[1:2]
    for t in mids:
        fn = f"{piece.test}_g{nn:02d}_still_{t:.2f}.jpg"
        still(piece.frame, t, 432, int(432 * H / W), os.path.join(review, fn))
        stills.append(fn)
    out = [name]
    audio_s = None
    if kind == "final" and getattr(piece, "score_preset", None) and not a.no_audio:
        wav = f"score_{piece.test}_g{nn:02d}.wav"
        audio_s = score(piece.score_preset, tl.dur, tl.changes(), os.path.join(local, wav),
                        os.path.join(local, "TIMINGS.csv"), f"{piece.test} g{nn:02d}", be)
        out += [wav, mux(path, os.path.join(local, wav), path.replace("_final.mp4", "_final_audio.mp4")).split(os.sep)[-1]]
    timing_row(local, nn, piece.test, "preview" if kind == "draft" else kind, dev, nfr, f"{W}x{H}", started, fin,
               mk, tm["render_s"], tm["encode_s"], audio_s, getattr(piece, "notes", ""))
    gens.add_files(local, piece.test, nn, out, stills)
    sheet = contact_sheet(path, os.path.join(review, f"{piece.test}_g{nn:02d}_{kind}_sheet.jpg"), dur=nfr / tl.fps)
    rep = _qc.review(path, tl, getattr(piece, "palette", None),
                     mask_fn=getattr(piece, "text_mask", None),
                     out_prefix=os.path.join(review, f"{piece.test}_g{nn:02d}_{kind}_qc"))
    print("sheet", sheet)
    return dict(gen=nn, path=path, timing=tm, qc=rep, stills=stills, files=out)


def parser(ap=None):
    ap = ap or argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="draft size (default 360x450)")
    ap.add_argument("--final", action="store_true", help="final size (default 1080x1350) + score + mux")
    ap.add_argument("--still", type=float, nargs="*", help="ad-hoc PNG stills at these times (review/)")
    ap.add_argument("--size", type=int, default=0, help="override width (height keeps the aspect)")
    ap.add_argument("--frames", type=int, default=0, help="render only the first N frames (tests)")
    ap.add_argument("--qc", default=None, help="run the automatic QC on an existing video and exit")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | numpy")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="frames per batch (needs piece.frames)")
    ap.add_argument("--x264", action="store_true", help="libx264 even on CUDA")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--work-start", default=None, help="NZ 'HH:MM' or 'YYYY-MM-DD HH:MM' when work began")
    return ap


def main(make_piece, argv=None, ap=None):
    """CLI for a piece: make_piece(args, be) -> piece. Handles --draft / --final / --still / --qc."""
    from .core import backend
    from . import qc as _qc
    a = parser(ap).parse_args(argv)
    be = backend(a.device, a.threads)
    piece = make_piece(a, be)
    print(piece.timeline.summary())
    for f in piece.timeline.check():
        print("TIMING FLAG", f)
    if a.qc:
        _qc.review(a.qc, piece.timeline, getattr(piece, "palette", None), mask_fn=getattr(piece, "text_mask", None),
                   out_prefix=os.path.splitext(a.qc)[0].replace(os.sep + "local" + os.sep, os.sep + "review" + os.sep) + "_qc")
        return None
    res = {}
    if a.still:
        _, review = dirs(piece.version_dir)
        for t in a.still:
            W = a.size or 360
            fr = piece.frame(t, W, int(W * 1.25))
            Image.fromarray(fr).save(os.path.join(review, f"{piece.test}_adhoc_{t:.2f}.png"))
    for kind, flag in (("draft", a.draft), ("final", a.final)):
        if flag:
            res[kind] = run(piece, kind, a, be)
    return res
