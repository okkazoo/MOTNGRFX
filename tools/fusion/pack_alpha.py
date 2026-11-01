#!/usr/bin/env python3
"""Pack an LTX Alpha-Gen matte PNG sequence into ONE file per animated element (CPU, ffmpeg + OpenCV).

An Alpha-Gen render leaves one PNG per frame (73 per P06 shot, ~47 MB for 6 shots). This step turns each shot into
  <stem>_matte.mkv   FFV1 lossless, native matte size, 24 fps, planar RGB (gbrp): the PNG RGB values BIT-EXACT.
                     The matte PNGs are RGB gray (R/G/B differ by up to 8), so the RGB triplets are kept, not a gray copy.
  <stem>_rgba.mov    ProRes 4444 (yuva444p10le), the colour clip at its native size with the matte (luma, bilinear resize)
                     as the alpha channel, same frame count / fps, audio stream copied.
and `load_matte()` reads either back as the same uint8 arrays the code got from cv2.imread(png, IMREAD_GRAYSCALE).

Run it after EVERY Alpha-Gen render lands (PNGs are never deleted or touched):
  python tools/fusion/pack_alpha.py pack                      # P06 v003, all six shots -> 15_fusion/v003/ingredients/P06/
  python tools/fusion/pack_alpha.py pack --shots s1a,s2a      # some shots
  python tools/fusion/pack_alpha.py pack --glob "DIR/*_alpha_s1a_1_image_*_.png" --clip CLIP.mp4 --stem NAME --out DIR2
  python tools/fusion/pack_alpha.py gather                    # hardlink colour clips + depth caches next to the packed files
  python tools/fusion/pack_alpha.py verify --shots s1a        # re-check packed files against the PNGs
Existing outputs are never overwritten (error) unless --replace is given. Every output is verified before it is
renamed into place; a failed check deletes the partial file and exits non-zero.

load_matte(src) accepts a packed .mkv/.mov, a PNG glob / folder / %05d pattern, or the FIRST frame PNG of a numbered
sequence (`..._image_00001_.png`). In the last form the packed `ingredients/<piece>/<stem>_matte.mkv` of the same shot
is used when it exists (env MOTNGRFX_NO_PACKED=1 forces the PNGs), else the PNG sequence.
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V3 = os.path.join(ROOT, "15_fusion", "v003")
PIECE = "P06"
SHOTS = ("s1a", "s1b", "s2a", "s2b", "s3a", "s3b")
FPS = 24


# ----------------------------------------------------------------------------------------------- paths
def ingredients_dir(piece=PIECE, version_dir=V3):
    return os.path.join(version_dir, "ingredients", piece)


def stem_of(shot, piece=PIECE, prefix="motion-graphic_fusion_v003"):
    return "%s_%s_%s_1" % (prefix, piece, shot)


def default_pngs(shot, piece=PIECE, version_dir=V3):
    """Sorted matte PNG list of one shot (local/pieces/<piece>/seq first, then out/)."""
    name = "motion-graphic_fusion_v003_%s_alpha_%s_1_image_*_.png" % (piece, shot)
    for d in (os.path.join(version_dir, "local", "pieces", piece, "seq"), os.path.join(version_dir, "out")):
        files = sorted(_glob.glob(os.path.join(d, name)))
        if files:
            return files
    raise FileNotFoundError("no matte PNGs for %s %s under %s" % (piece, shot, version_dir))


def default_clip(shot, piece=PIECE, version_dir=V3):
    return os.path.join(version_dir, "out", "motion-graphic_fusion_v003_%s_%s_1_00001_.mp4" % (piece, shot))


_SEQ_RE = re.compile(r"^(?P<prefix>.+_(?P<piece>P\d+))_alpha_(?P<shot>s\w+?)_1_image_(?P<n>\d+)_\.png$")


def packed_for(first_png):
    """Packed *_matte.mkv belonging to a matte PNG sequence member (any frame), or None."""
    m = _SEQ_RE.match(os.path.basename(first_png))
    if not m:
        return None
    parts = os.path.abspath(first_png).replace("\\", "/").split("/")
    idx = [i for i, p in enumerate(parts) if re.fullmatch(r"v\d{3}", p)]
    if not idx:
        return None
    vdir = "/".join(parts[:idx[-1] + 1])                          # .../15_fusion/v003
    f = os.path.join(vdir, "ingredients", m.group("piece"), "%s_%s_1_matte.mkv" % (m.group("prefix"), m.group("shot")))
    return f if os.path.exists(f) else None


# ----------------------------------------------------------------------------------------------- ffmpeg io
def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode:
        raise RuntimeError("%s failed (%d): %s" % (cmd[0], r.returncode, r.stderr.decode("utf8", "replace")[-1500:]))
    return r


def _probe(path):
    """-> dict(w, h, pix_fmt, fps, frames, audio) via ffprobe (frames counted, not trusted from the header)."""
    r = _run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,pix_fmt,r_frame_rate,nb_read_frames", "-of", "default=nw=1", path])
    d = dict(l.split("=", 1) for l in r.stdout.decode().split() if "=" in l)
    num, den = d["r_frame_rate"].split("/")
    a = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path])
    return dict(w=int(d["width"]), h=int(d["height"]), pix_fmt=d["pix_fmt"], fps=float(num) / float(den),
                frames=int(d["nb_read_frames"]), audio=a.stdout.decode().split())


def _decode_bgr(path):
    """Every frame of a video as one (n,h,w,3) uint8 BGR array (lossless sources only: exact)."""
    p = _probe(path)
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    a = np.frombuffer(r.stdout, np.uint8)
    n = a.size // (p["w"] * p["h"] * 3)
    if n * p["w"] * p["h"] * 3 != a.size:
        raise RuntimeError("decode size mismatch for " + path)
    return a.reshape(n, p["h"], p["w"], 3)


def _read_png_bgr(f):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if im is None or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("matte PNG must be 8-bit 3-channel (RGB gray): %s -> %s" % (f, None if im is None else (im.dtype, im.shape)))
    return im


# ----------------------------------------------------------------------------------------------- matte
def pack_matte(files, out_mkv, fps=FPS):
    """PNG list -> FFV1 mkv (planar RGB, intra only). Not verified here: see `verify_matte`."""
    first = _read_png_bgr(files[0])
    h, w = first.shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
           "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1", "-pix_fmt", "gbrp",
           "-f", "matroska", out_mkv]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in files:
            im = first if f == files[0] else _read_png_bgr(f)
            if im.shape[:2] != (h, w):
                raise ValueError("matte size changes inside the sequence: %s" % f)
            pr.stdin.write(np.ascontiguousarray(im).tobytes())
        pr.stdin.close()
    except BrokenPipeError:
        pass
    err = pr.stderr.read().decode("utf8", "replace")
    if pr.wait():
        raise RuntimeError("ffmpeg (matte) failed: " + err[-1500:])


def verify_matte(files, mkv):
    """Decode every frame of the mkv and compare with the PNGs. Raises AssertionError on ANY differing pixel."""
    dec = _decode_bgr(mkv)
    if len(dec) != len(files):
        raise AssertionError("matte frame count %d != %d PNGs" % (len(dec), len(files)))
    bad = []
    for i, f in enumerate(files):
        if not np.array_equal(dec[i], _read_png_bgr(f)):
            bad.append(i)
    if bad:
        raise AssertionError("matte NOT bit-exact in %d frames (first: %s)" % (len(bad), bad[:8]))
    return len(dec)


# ----------------------------------------------------------------------------------------------- rgba
def pack_rgba(clip, matte_mkv, out_mov, n_frames):
    """Colour clip + matte (luma, bilinear to the clip size) as alpha -> ProRes 4444 yuva444p10le, audio copied."""
    p = _probe(clip)
    fc = ("[1:v]scale=%d:%d:flags=bilinear,format=gray[a];[0:v]format=yuv444p10le[c];"
          "[c][a]alphamerge,format=yuva444p10le[v]" % (p["w"], p["h"]))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", clip, "-i", matte_mkv, "-filter_complex", fc,
           "-map", "[v]", "-map", "0:a?", "-frames:v", str(n_frames), "-r", str(FPS),
           "-c:v", "prores_ks", "-profile:v", "4444", "-vendor", "apl0", "-pix_fmt", "yuva444p10le", "-alpha_bits", "16",
           "-c:a", "copy", "-f", "mov", out_mov]
    _run(cmd)


def verify_rgba(clip, mkv, mov, probe_frame=None):
    """ffprobe checks + the alpha of one frame (default the middle) must correlate with the matte. Returns a dict."""
    pc, pm, pr = _probe(clip), _probe(mkv), _probe(mov)
    if not pr["pix_fmt"].startswith("yuva"):
        raise AssertionError("rgba.mov has no alpha pix_fmt: " + pr["pix_fmt"])
    if pr["frames"] != pc["frames"] or pr["frames"] != pm["frames"]:
        raise AssertionError("frame counts differ: clip %d matte %d rgba %d" % (pc["frames"], pm["frames"], pr["frames"]))
    if (pr["w"], pr["h"]) != (pc["w"], pc["h"]):
        raise AssertionError("rgba size %s != clip %s" % ((pr["w"], pr["h"]), (pc["w"], pc["h"])))
    if abs(pr["fps"] - FPS) > 1e-3:
        raise AssertionError("rgba fps %s" % pr["fps"])
    if bool(pc["audio"]) != bool(pr["audio"]):
        raise AssertionError("audio streams: clip %s rgba %s" % (pc["audio"], pr["audio"]))
    n = pr["frames"] // 2 if probe_frame is None else probe_frame
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", mov, "-vf", "select=eq(n\\,%d),alphaextract" % n,
              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    al = np.frombuffer(r.stdout, np.uint8).reshape(pr["h"], pr["w"]).astype(np.float32)
    gray = cv2.cvtColor(_decode_bgr(mkv)[n], cv2.COLOR_BGR2GRAY)
    ref = cv2.resize(gray, (pr["w"], pr["h"]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if al.std() < 1e-3 or ref.std() < 1e-3:
        corr = 1.0 if np.allclose(al, ref, atol=2) else 0.0          # constant matte: correlation undefined
    else:
        corr = float(np.corrcoef(al.ravel(), ref.ravel())[0, 1])
    if corr < 0.995:
        raise AssertionError("alpha of frame %d does not match the matte: corr %.4f" % (n, corr))
    return dict(pix_fmt=pr["pix_fmt"], frames=pr["frames"], size=(pr["w"], pr["h"]), fps=pr["fps"], audio=pr["audio"],
                probe_frame=n, alpha_corr=corr, alpha_mean_abs_diff=float(np.abs(al - ref).mean()))


# ----------------------------------------------------------------------------------------------- loader
def _png_seq_files(src):
    """src -> sorted PNG list. src = glob, folder, %05d pattern, or one numbered member (`..._image_00001_.png`)."""
    if os.path.isdir(src):
        files = sorted(_glob.glob(os.path.join(src, "*.png")))
    elif "%" in os.path.basename(src):
        files, i = [], 1
        while os.path.exists(src % i):                            # consecutive from 1, like depth_orbit.load_video
            files.append(src % i)
            i += 1
    elif any(c in src for c in "*?["):
        files = sorted(_glob.glob(src))
    else:
        m = re.search(r"^(.*_image_)(\d+)(_\.png)$", src)
        if not m:
            return [src] if os.path.exists(src) else []
        width = len(m.group(2))
        files, i = [], int(m.group(2))
        while os.path.exists("%s%0*d%s" % (m.group(1), width, i, m.group(3))):
            files.append("%s%0*d%s" % (m.group(1), width, i, m.group(3)))
            i += 1
    return files


def _gray_like_imread(bgr):
    """BGR -> the uint8 gray cv2.imread(png, IMREAD_GRAYSCALE) returns for the same pixels. cv2.cvtColor differs by 1 level
    (libpng's RGB->gray rounding), so the pixels take the same PNG decoder path (stored, uncompressed, in memory)."""
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def load_matte(src, frames=None, gray=True, prefer_packed=True):
    """-> uint8 array (n,H,W) of the matte frames (gray=True: identical to cv2.imread(png, IMREAD_GRAYSCALE) per frame),
    or (n,H,W,3) BGR with gray=False. `frames` = list of frame indices to return (default all).
    src: packed .mkv/.mov | PNG glob / folder / %05d pattern | first-frame PNG of a numbered sequence (the packed
    ingredients/<piece>/*_matte.mkv of that shot is used when present and prefer_packed and not MOTNGRFX_NO_PACKED)."""
    src = str(src)
    if src.lower().endswith((".mkv", ".mov")):
        arr = _decode_bgr(src)
    else:
        pk = packed_for(src) if (prefer_packed and not os.environ.get("MOTNGRFX_NO_PACKED")
                                 and src.lower().endswith(".png") and "*" not in src and "%" not in src) else None
        if pk:
            arr = _decode_bgr(pk)
            m = re.search(r"_image_(\d+)_\.png$", src)
            if m and int(m.group(1)) > 1:                          # a later member of the sequence: start there
                arr = arr[int(m.group(1)) - 1:]
        else:
            files = _png_seq_files(src)
            if not files:
                raise FileNotFoundError("no matte frames for %s" % src)
            if frames is not None:
                files = [files[i] for i in frames]
                frames = None
            if gray:
                out = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
            else:
                out = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
            if any(o is None for o in out):
                raise IOError("unreadable matte PNG in " + src)
            return np.stack(out)
    if frames is not None:
        arr = arr[list(frames)]
    return np.stack([_gray_like_imread(a) for a in arr]) if gray else np.ascontiguousarray(arr)


# ----------------------------------------------------------------------------------------------- commands
def _target(path, replace):
    if os.path.exists(path) and not replace:
        raise FileExistsError("%s exists (never overwritten; pass --replace to redo it)" % path)
    return path


def pack_shot(files, clip, stem, out_dir, replace=False, log=print):
    """Pack + verify one shot. Returns the report dict. Partial files are removed when a check fails."""
    os.makedirs(out_dir, exist_ok=True)
    mkv = _target(os.path.join(out_dir, stem + "_matte.mkv"), replace)
    mov = _target(os.path.join(out_dir, stem + "_rgba.mov"), replace)
    tm, tv = mkv + ".part", mov + ".part"
    rep = dict(stem=stem, frames=len(files), matte=mkv, rgba=mov)
    try:
        pack_matte(files, tm)
        rep["matte_exact_frames"] = verify_matte(files, tm)
        log("%s: matte.mkv bit-exact over %d frames" % (stem, rep["matte_exact_frames"]))
        pack_rgba(clip, tm, tv, len(files))
        rep["rgba"] = verify_rgba(clip, tm, tv)
        log("%s: rgba.mov %s" % (stem, {k: v for k, v in rep["rgba"].items() if k != "audio"}))
        os.replace(tm, mkv)
        os.replace(tv, mov)
    finally:
        for f in (tm, tv, mkv + ".part", mov + ".part"):
            if os.path.exists(f):
                os.remove(f)
    rep["bytes"] = os.path.getsize(mkv) + os.path.getsize(mov)
    return rep


def link(src, dst):
    """Hardlink src at dst (NTFS: no extra space). Existing dst is left alone."""
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.link(src, dst)
    return True


def cmd_pack(a):
    out_dir = a.out or ingredients_dir()
    if a.glob:
        if not (a.clip and a.stem):
            sys.exit("--glob needs --clip and --stem")
        jobs = [(sorted(_glob.glob(a.glob)), a.clip, a.stem)]
    else:
        shots = a.shots.split(",") if a.shots else SHOTS
        jobs = [(default_pngs(s), default_clip(s), stem_of(s)) for s in shots]
    tot = 0
    for files, clip, stem in jobs:
        rep = pack_shot(files, clip, stem, out_dir, replace=a.replace)
        tot += rep["bytes"]
    print("packed %d shot(s), %.1f MB total in %s" % (len(jobs), tot / 1e6, out_dir))


def cmd_verify(a):
    out_dir = a.out or ingredients_dir()
    for s in (a.shots.split(",") if a.shots else SHOTS):
        stem = stem_of(s)
        mkv, mov = os.path.join(out_dir, stem + "_matte.mkv"), os.path.join(out_dir, stem + "_rgba.mov")
        n = verify_matte(default_pngs(s), mkv)
        r = verify_rgba(default_clip(s), mkv, mov)
        print("%s: matte exact %d frames; rgba alpha corr %.5f" % (stem, n, r["alpha_corr"]))


def cmd_gather(a):
    out_dir = a.out or ingredients_dir()
    n = 0
    for s in SHOTS:
        c = default_clip(s)
        n += link(c, os.path.join(out_dir, os.path.basename(c)))
    dd = os.path.join(ROOT, "17_depth-dolly", "v001", "local")          # frame-0 depth caches (depth_orbit --depth-from)
    for s in ("s1a", "s2a", "s3a"):
        for f in ("depth_%s_raw.npy" % s,):
            if os.path.exists(os.path.join(dd, f)):
                n += link(os.path.join(dd, f), os.path.join(out_dir, f))
        pv = os.path.join(ROOT, "17_depth-dolly", "v001", "review", "depth_%s.png" % s)
        if os.path.exists(pv):
            n += link(pv, os.path.join(out_dir, "depth_%s.png" % s))
    print("hardlinked %d new file(s) into %s" % (n, out_dir))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("pack", cmd_pack), ("verify", cmd_verify), ("gather", cmd_gather)):
        p = sub.add_parser(name)
        p.add_argument("--shots", default=None, help="comma list (default all six P06 shots)")
        p.add_argument("--out", default=None, help="ingredients folder (default 15_fusion/v003/ingredients/P06)")
        if name == "pack":
            p.add_argument("--glob", default=None)
            p.add_argument("--clip", default=None)
            p.add_argument("--stem", default=None)
            p.add_argument("--replace", action="store_true", help="overwrite existing packed files")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
