"""ingest.py - normalise any user-dropped video / still to the MOTNGRFX base format (16_app/INGEST.md).

    probe(path)                         ffprobe -> size after rotation, r/avg fps, VFR flag, duration, audio
    timeline_fps(src_fps, vfr, avg)     the fps policy: integer-ratio only (keep / exact halve / VFR -> constant), NEVER 30 -> 24
    plan_shot(...) / build_plan(...)    the aspect policy per shot: crop (cover, window follows the subject, away from captions) | fit
                                        (contain on a flat ground colour); never upscales (max_tier)
    render_commands(plan, tier, outdir) the ffmpeg command(s) that write the normalised clip / still (crop path: static -> one ffmpeg
                                        command, moving / mixed shots -> an ffmpeg -> cv2 -> ffmpeg frame pipe; audio is copied)
    CLI: ingest.py plan <video> [--track t.json] [--captions c.json] [--cuts c.json|detect] --tier preview -o ingest.json
         ingest.py render ingest.json --tier preview -o outdir/

The planning functions are pure stdlib (numpy / cv2 only inside the pixel helpers), so the module runs standalone on the box.
The subject box track (a low-res matte -> x0,y0,x1,y1 normalised per frame) and the caption boxes are INPUT DATA; the matte itself is
the GPU part and is not made here.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

TIERS = ["preview", "draft", "final"]
# == plan_rules.sizes_for(tier)["compose"] (9:16); tests/test_ingest.py checks the parity. Kept here so the module runs without the repo.
COMPOSE = {"preview": (360, 640), "draft": (540, 960), "final": (1080, 1920)}
AR = (9, 16)                           # the base ratio (w, h); 16:9 = (16, 9)
NEAR_TARGET = 0.10                     # |ratio / target - 1| <= this: a light crop
SUBJECT_PAD = 0.08                     # subject union padded by this fraction of its size on every side
SMOOTH_S = 0.5                         # moving-average window of the crop path
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", "mp2"}

STD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
             Fraction(48000, 1001), Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60)]


# ---------------------------------------------------------------------------------------------------------------- sizes
def target_size(tier: str, fmt: str = "9:16") -> tuple[int, int]:
    w, h = COMPOSE[tier]
    return (h, w) if fmt == "16:9" else (w, h)


def ar_of(fmt: str = "9:16") -> tuple[int, int]:
    return (16, 9) if fmt == "16:9" else (9, 16)


def tier_min(a: str, b: str) -> str:
    return a if TIERS.index(a) <= TIERS.index(b) else b


def largest_window(W: int, H: int, ar=AR) -> tuple[int, int]:
    """largest (a*k, b*k) window inside W x H: the exact base ratio with integer sides."""
    k = min(W // ar[0], H // ar[1])
    return (W, H) if k < 1 else (ar[0] * k, ar[1] * k)


def max_tier_for(mode: str, W: int, H: int, window, ar=AR, fmt: str = "9:16"):
    """highest tier this shot can feed without upscaling (scale factor source -> output <= 1); None if even preview needs an upscale."""
    best = None
    for t in TIERS:
        tw, th = target_size(t, fmt)
        s = tw / window[0] if mode == "crop" else min(tw / W, th / H)
        if s <= 1.0 + 1e-9:
            best = t
    return best


# ---------------------------------------------------------------------------------------------------------------- fps
def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        x = x.strip()
        if "/" in x:
            n, d = x.split("/")
            return Fraction(int(n), int(d)) if int(d) else Fraction(0)
        return Fraction(x)
    return Fraction(float(x)).limit_denominator(100000)


def snap_rate(x) -> Fraction:
    """a measured rate -> the exact standard rational (29.97 -> 30000/1001) when within 0.04 %, else a small rational."""
    f = _frac(x)
    if f <= 0:
        return f
    for s in STD_RATES:
        if abs(float(f) - float(s)) / float(s) < 4e-4:
            return s
    return f if f.denominator <= 1001 else f.limit_denominator(1001)


def fmt_rate(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def detect_vfr(r_fps, avg_fps) -> bool:
    r, a = float(_frac(r_fps)), float(_frac(avg_fps))
    return r > 0 and a > 0 and abs(r - a) / r > 0.01


def timeline_fps(src_fps, vfr: bool = False, avg_fps=None) -> dict:
    """The fps policy. Only integer frame ratios, never a fractional resample:
       23.976 / 24 / 25 / 29.97 / 30 and every other rate < 46 fps: kept (rational, exact)
       47.95 / 48 / 50 / 59.94 / 60 (>= 46 fps): exactly halved - select every 2nd frame, no blending (90 -> 30, 120 -> 30: every 3rd / 4th)
       VFR: constant at the nearest of 24 / 25 / 30 to the average rate (>= 46 avg: halved first), fps filter
    Returns dict(src, fps, fps_float, policy, step, vfr, filter, note); `filter` is the ffmpeg -vf fragment that does it."""
    f = snap_rate(src_fps)
    if vfr:
        a = float(snap_rate(avg_fps if avg_fps is not None else src_fps))
        if a >= 46:
            a /= 2
        out = Fraction(min((24, 25, 30), key=lambda s: abs(s - a)))
        return _fps_result(f, out, 1, "vfr_const", f"fps={fmt_rate(out)}", f"variable frame rate (avg {a:.3f}): constant {fmt_rate(out)}")
    k = 1
    if float(f) >= 46:
        k = 2
        while float(f) / k > 30.5:
            k += 1
    out = f / k
    if k == 1:
        return _fps_result(f, out, 1, "keep", f"fps={fmt_rate(out)}", "kept as is (identity on a constant-rate stream)")
    ex = f"N/(({out.numerator}/{out.denominator})*TB)"
    return _fps_result(f, out, k, "halve" if k == 2 else f"divide{k}",
                       f"select='not(mod(n,{k}))',setpts={ex},fps={fmt_rate(out)}", f"every {k}{'nd' if k == 2 else 'th'} frame kept, no blending")


def _fps_result(src, out, k, policy, flt, note):
    assert out > 0
    return {"src": fmt_rate(src), "fps": fmt_rate(out), "fps_float": round(float(out), 5), "policy": policy, "step": k,
            "vfr": policy == "vfr_const", "filter": flt, "note": note}


# ---------------------------------------------------------------------------------------------------------------- probe
def _rot_of(stream: dict) -> int:
    rot = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    if rot is None and "rotate" in (stream.get("tags") or {}):
        rot = stream["tags"]["rotate"]
    try:
        return int(round(float(rot))) % 360 if rot is not None else 0
    except (TypeError, ValueError):
        return 0


def _sar_of(stream: dict) -> Fraction:
    s = str(stream.get("sample_aspect_ratio") or "1:1")
    try:
        n, d = s.replace("/", ":").split(":")
        f = Fraction(int(n), int(d))
        return f if f > 0 else Fraction(1)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)


def parse_probe(j: dict, path: str | None = None) -> dict:
    """ffprobe -show_streams -show_format -of json  ->  the source description."""
    streams, fmt = j.get("streams") or [], j.get("format") or {}
    vids = [s for s in streams if s.get("codec_type") == "video"]
    v = next((s for s in vids if not (s.get("disposition") or {}).get("attached_pic")), vids[0] if vids else None)
    if v is None:
        raise ValueError("no video stream")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    w, h = int(v["width"]), int(v["height"])
    sar = _sar_of(v)
    rot = _rot_of(v)
    dw, dh = int(round(w * float(sar))), h
    if rot % 180 == 90:
        dw, dh = dh, dw
    name = str(fmt.get("format_name", ""))
    is_image = "image2" in name or name.endswith("_pipe")
    r_s, avg_s = v.get("r_frame_rate", "0/0"), v.get("avg_frame_rate", "0/0")
    r, avg = _frac(r_s), _frac(avg_s)
    vfr = False if is_image else detect_vfr(r, avg)
    dur = v.get("duration") or fmt.get("duration")
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        dur = None
    nbf = v.get("nb_frames")
    audio = None
    if a is not None:
        try:
            adur = float(a.get("duration") or fmt.get("duration"))
        except (TypeError, ValueError):
            adur = None
        audio = {"codec": a.get("codec_name"), "duration": adur, "sample_rate": a.get("sample_rate"), "channels": a.get("channels")}
    return {"path": path, "kind": "image" if is_image else "video", "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
            "width": w, "height": h, "sar": fmt_rate(sar), "rotation": rot, "display_w": dw, "display_h": dh,
            "r_frame_rate": r_s, "avg_frame_rate": avg_s, "r_fps": round(float(r), 4), "avg_fps": round(float(avg), 4), "vfr": vfr,
            "duration": dur, "nb_frames": int(nbf) if nbf and str(nbf).isdigit() else None, "audio": audio}


def probe(path: str, ffprobe: str = "ffprobe") -> dict:
    out = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return parse_probe(json.loads(out), str(path))


# ---------------------------------------------------------------------------------------------------------------- tracks
def _clean_box(b):
    if b is None:
        return None
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in b)
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return None
    return [x0, y0, x1, y1]


def track_boxes(track, times, fps=None) -> list:
    """per time -> normalised box | None. track: {"fps": f, "boxes": [box|None, ...]} (frame index = round(t * f); fps null = the
    output frame index) or {"samples": [{"t": s, "box": [..]}, ...]} (nearest sample)."""
    if not track:
        return [None] * len(times)
    if isinstance(track, list):
        track = {"fps": None, "boxes": track}
    if "samples" in track:
        s = sorted(track["samples"], key=lambda e: e["t"])
        res = []
        for t in times:
            e = min(s, key=lambda e: abs(e["t"] - t)) if s else None
            res.append(_clean_box(e["box"]) if e else None)
        return res
    boxes = track.get("boxes") or []
    f = track.get("fps")
    res = []
    for i, t in enumerate(times):
        k = i if not f else int(round(t * float(_frac(f))))
        res.append(_clean_box(boxes[min(max(k, 0), len(boxes) - 1)]) if boxes else None)
    return res


def _fill(boxes):
    """None -> the nearest known box (forward, then backward fill)."""
    out, last = [], None
    for b in boxes:
        last = b if b is not None else last
        out.append(last)
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def _pad_box(b, pad=SUBJECT_PAD):
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - pad * w), max(0.0, b[1] - pad * h), min(1.0, b[2] + pad * w), min(1.0, b[3] + pad * h)]


def _captions(captions) -> list:
    out = []
    for i, c in enumerate(captions or []):
        b = _clean_box(c.get("box"))
        if b is None:
            continue
        out.append({"index": i, "box": b, "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", math.inf))})
    return out


# ---------------------------------------------------------------------------------------------------------------- geometry
def smooth(vals, win: int) -> list:
    """centred moving average, edges replicated (no lag, no drift at the ends)."""
    n = len(vals)
    if win <= 1 or n <= 1:
        return [float(v) for v in vals]
    h = win // 2
    pad = [vals[0]] * h + list(vals) + [vals[-1]] * h
    cs = [0.0]
    for v in pad:
        cs.append(cs[-1] + v)
    w = 2 * h + 1
    return [(cs[i + w] - cs[i]) / w for i in range(n)]


def _overlap(x, y, ww, wh, c) -> float:
    ox = min(x + ww, c[2]) - max(x, c[0])
    oy = min(y + wh, c[3]) - max(y, c[1])
    return ox * oy if ox > 0 and oy > 0 else 0.0


def _push(x, y, ww, wh, rng, caps):
    """move the window (x, y) inside rng = (xlo, xhi, ylo, yhi) so it overlaps the caption boxes (px) as little as possible,
    staying as close to (x, y) as it can; no caption: just the clamp."""
    xlo, xhi, ylo, yhi = rng
    cands = [(x, y)]
    for c in caps:
        cands += [(c[0] - ww, y), (c[2], y), (x, c[1] - wh), (x, c[3])]
    best, bkey = (x, y), None
    for cx, cy in cands:
        cx, cy = min(max(cx, xlo), xhi), min(max(cy, ylo), yhi)
        ov = sum(_overlap(cx, cy, ww, wh, c) for c in caps)
        key = (round(ov, 1), abs(cx - x) + abs(cy - y))
        if bkey is None or key < bkey:
            best, bkey = (cx, cy), key
    return best


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb)


def plan_shot(W: int, H: int, times, track=None, captions=None, *, ar=AR, fmt: str = "9:16", ground=None, fps=None, smooth_s: float = SMOOTH_S,
              pad: float = SUBJECT_PAD, index: int = 0, start: float = 0.0, end: float = 0.0, f0: int = 0) -> dict:
    """The aspect policy of one shot. W x H = display size; times = the shot's output-frame times (s, from the clip start);
    track = subject boxes (see track_boxes); captions = [{box, t0, t1}] normalised. Returns the shot dict (mode crop | fit)."""
    n = len(times)
    R = ar[0] / ar[1]
    r = W / H
    ww, wh = largest_window(W, H, ar)
    boxes = track_boxes(track, times) if track else [None] * n
    have = any(b is not None for b in boxes)
    boxes = _fill(boxes) if have else boxes
    padded = [_pad_box(b, pad) for b in boxes] if have else []
    union = None
    if have:
        # union of the per-frame boxes over the shot, padded by `pad` of its own size, clamped to the frame
        ub = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        union = _pad_box(ub, pad)
    near = abs(r / R - 1.0) <= NEAR_TARGET
    if near:
        mode, reason = "crop", ("native" if (ww, wh) == (W, H) else "near_target")
    elif have:
        uw, uh = (union[2] - union[0]) * W, (union[3] - union[1]) * H
        if uw <= ww + 1e-6 and uh <= wh + 1e-6:
            mode, reason = "crop", "subject_fits"
        else:
            mode, reason = "fit", "subject_wider_than_window" if uw > ww + 1e-6 else "subject_taller_than_window"
    else:
        src_matches = (r < 1) if R < 1 else (r > 1)
        mode, reason = ("crop", "default_centre_crop") if src_matches else ("fit", "default_fit")
    shot = {"index": index, "start": round(start, 4), "end": round(end, 4), "f0": f0, "f1": f0 + n, "mode": mode, "reason": reason,
            "subject_union": [round(v, 4) for v in union] if union else None}
    if mode == "fit":
        shot.update({"window": None, "ground": ground, "static": True, "xy": None, "path": None, "captions": [],
                     "max_tier": max_tier_for("fit", W, H, None, ar, fmt)})
        return shot

    xmax, ymax = W - ww, H - wh
    caps = _captions(captions)
    sm = max(1, int(round(smooth_s * fps))) if (fps and n > 1) else 1
    if sm % 2 == 0:
        sm += 1
    dil = (sm // 2) / fps if fps else 0.0
    # desired window top-left: follows the subject centroid, else the centre of the free range
    des, rngs = [], []
    for i in range(n):
        if have:
            b = boxes[i]
            cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
            dx, dy = cx - ww / 2, cy - wh / 2
            p = padded[i]
            xl, xh = max(0.0, p[2] * W - ww), min(float(xmax), p[0] * W)
            yl, yh = max(0.0, p[3] * H - wh), min(float(ymax), p[1] * H)
            rng = (xl if xl <= xh else 0.0, xh if xl <= xh else float(xmax), yl if yl <= yh else 0.0, yh if yl <= yh else float(ymax))
        else:
            dx, dy, rng = xmax / 2, ymax / 2, (0.0, float(xmax), 0.0, float(ymax))
        des.append((min(max(dx, rng[0]), rng[1]), min(max(dy, rng[2]), rng[3])))
        rngs.append(rng)

    def active(i, extra):
        t = times[i]
        return [[c["box"][0] * W, c["box"][1] * H, c["box"][2] * W, c["box"][3] * H] for c in caps if c["t0"] - extra <= t < c["t1"] + extra]

    pre = [_push(des[i][0], des[i][1], ww, wh, rngs[i], active(i, dil)) for i in range(n)]
    sx, sy = smooth([p[0] for p in pre], sm), smooth([p[1] for p in pre], sm)
    fin = [_push(sx[i], sy[i], ww, wh, rngs[i], active(i, 0.0)) for i in range(n)]
    fin = [(min(max(x, 0.0), xmax), min(max(y, 0.0), ymax)) for x, y in fin]
    static = (max(p[0] for p in fin) - min(p[0] for p in fin) < 0.75) and (max(p[1] for p in fin) - min(p[1] for p in fin) < 0.75)
    if static:
        mx, my = sum(p[0] for p in fin) / n, sum(p[1] for p in fin) / n
        fin = [(mx, my)] * n
    avoided = []
    for c in caps:
        px = [c["box"][0] * W, c["box"][1] * H, c["box"][2] * W, c["box"][3] * H]
        area = (px[2] - px[0]) * (px[3] - px[1])
        idx = [i for i in range(n) if c["t0"] <= times[i] < c["t1"]]
        if not idx:
            continue
        res = max(_overlap(fin[i][0], fin[i][1], ww, wh, px) for i in idx) / area
        avoided.append({"index": c["index"], "box": [round(v, 4) for v in c["box"]], "t0": c["t0"], "t1": None if c["t1"] == math.inf else c["t1"],
                        "avoided": res < 1e-6, "residual_overlap": round(res, 4)})
    shot.update({"window": [ww, wh], "ground": None, "static": static, "captions": avoided,
                 "xy": [round(fin[0][0], 2), round(fin[0][1], 2)] if static else None,
                 "path": None if static else [[round(x, 2), round(y, 2)] for x, y in fin],
                 "max_tier": max_tier_for("crop", W, H, (ww, wh), ar, fmt)})
    return shot


# ---------------------------------------------------------------------------------------------------------------- plan
def _norm_cuts(cuts, duration):
    if isinstance(cuts, dict):
        cuts = cuts.get("cuts", [])
    cs = sorted({round(float(c), 4) for c in (cuts or []) if 0 < float(c) < (duration or 1e9)})
    return cs


def build_plan(pr: dict, tier: str = "preview", track=None, captions=None, cuts=None, ground_fn=None, fmt: str = "9:16",
               smooth_s: float = SMOOTH_S) -> dict:
    """ingest.json from a probe. ground_fn(t_seconds) -> '#rrggbb' | None supplies the fit-mode ground colour (needs a frame grab)."""
    ar = ar_of(fmt)
    W, H = pr["display_w"], pr["display_h"]
    if pr["kind"] == "image":
        sh = plan_shot(W, H, [0.0], track, captions, ar=ar, fmt=fmt, ground=None, fps=None, smooth_s=smooth_s)
        if sh["mode"] == "fit" and ground_fn:
            sh["ground"] = ground_fn(0.0)
        shots, fps, n_out, dur = [sh], None, 1, None
    else:
        src_fps = pr["avg_fps"] if pr["vfr"] else pr["r_fps"]
        fps = timeline_fps(pr["avg_frame_rate"] if pr["vfr"] else pr["r_frame_rate"], vfr=pr["vfr"], avg_fps=pr["avg_frame_rate"])
        fo = float(_frac(fps["fps"]))
        dur = pr["duration"] or ((pr["nb_frames"] or 1) / max(src_fps, 1e-6))
        n_out = max(1, int(round(dur * fo)))
        bounds = [0.0] + _norm_cuts(cuts, dur) + [dur]
        shots = []
        for k in range(len(bounds) - 1):
            f0 = 0 if k == 0 else int(round(bounds[k] * fo))
            f1 = n_out if k == len(bounds) - 2 else int(round(bounds[k + 1] * fo))
            if f1 <= f0:
                continue
            times = [i / fo for i in range(f0, f1)]
            sh = plan_shot(W, H, times, track, captions, ar=ar, fmt=fmt, fps=fo, smooth_s=smooth_s, index=len(shots),
                           start=f0 / fo, end=f1 / fo, f0=f0)
            if sh["mode"] == "fit" and ground_fn:
                sh["ground"] = ground_fn(f0 / fo)
            shots.append(sh)
    warnings = []
    maxes = [s["max_tier"] for s in shots]
    if any(m is None for m in maxes):
        mx = "preview"
        warnings.append("source too small: even the preview tier needs an upscale (flagged, not done)")
    else:
        mx = min(maxes, key=TIERS.index)
    eff = tier_min(tier, mx)
    if eff != tier:
        warnings.append(f"requested tier {tier} dropped to {eff}: the source cannot feed {tier} without upscaling")
    if pr["vfr"]:
        warnings.append("variable frame rate: constant-rate conversion at the nearest standard rate (frames dropped / duplicated by timestamp)")
    if pr.get("audio") is None and pr["kind"] == "video":
        warnings.append("no audio stream")
    audio = None
    if pr.get("audio"):
        cp = (pr["audio"].get("codec") or "") in MP4_AUDIO
        audio = {"codec": pr["audio"].get("codec"), "copy": cp, "note": "stream copy, untouched" if cp else "codec not allowed in mp4: re-encoded to AAC 256k"}
    return {"version": 1, "kind": pr["kind"], "format": fmt, "ratio": f"{ar[0]}:{ar[1]}", "source": pr, "tier": tier, "effective_tier": eff,
            "max_tier": mx, "targets": {t: list(target_size(t, fmt)) for t in TIERS}, "fps": fps, "frames": n_out, "duration": dur,
            "audio": audio, "shots": shots, "warnings": warnings}


# ---------------------------------------------------------------------------------------------------------------- ffmpeg
def _even(v) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def fit_geometry(W: int, H: int, tw: int, th: int) -> tuple[int, int, int, int]:
    """contain: scale so the whole frame fits, centred. -> (sw, sh, px, py); never larger than the source (no upscale is planned: max_tier)."""
    s = min(tw / W, th / H)
    sw, sh = min(tw, _even(W * s)), min(th, _even(H * s))
    return sw, sh, ((tw - sw) // 2) // 2 * 2, ((th - sh) // 2) // 2 * 2


def _pre_filters(pr: dict) -> list:
    """non-square pixels (SAR != 1, unrotated) are made square first, so every later number is in display pixels."""
    if abs(float(_frac(pr["sar"])) - 1) > 0.005 and pr["rotation"] % 180 == 0:
        return [f"scale={pr['display_w']}:{pr['display_h']},setsar=1"]
    return []


def shot_filter(plan: dict, shot: dict, tw: int, th: int) -> str:
    """the spatial ffmpeg filter of a static shot."""
    pr = plan["source"]
    W, H = pr["display_w"], pr["display_h"]
    if shot["mode"] == "crop":
        ww, wh = shot["window"]
        x, y = (int(round(v)) for v in shot["xy"])
        return f"crop={ww}:{wh}:{x}:{y}:exact=1,scale={tw}:{th}:flags=lanczos,setsar=1"
    sw, sh, px, py = fit_geometry(W, H, tw, th)
    col = (shot.get("ground") or "#000000").lstrip("#")
    return f"scale={sw}:{sh}:flags=lanczos,pad={tw}:{th}:{px}:{py}:color=0x{col},setsar=1"


def out_name(plan: dict, tier: str) -> str:
    stem = Path(plan["source"]["path"] or "clip").stem
    return f"{stem}_ingest_{tier}." + ("png" if plan["kind"] == "image" else "mp4")


def render_commands(plan: dict, tier: str, outdir, src: str | None = None, ffmpeg: str = "ffmpeg") -> list:
    """The command(s) that write the normalised clip at `tier` (clipped to plan.max_tier: nothing is upscaled).
    [{"kind": "ffmpeg", "argv": [...], "out": path}]                      static single shot / still
    [{"kind": "pipe", "reader": [...], "writer": [...], "size": [w, h], "out": path}]   moving crop or several shots (frames through cv2)"""
    pr = plan["source"]
    src = src or pr["path"]
    tier = tier_min(tier, plan["max_tier"])
    tw, th = target_size(tier, plan["format"])
    out = str(Path(outdir) / out_name(plan, tier))
    shots = plan["shots"]
    if plan["kind"] == "image":
        vf = ",".join(_pre_filters(pr) + [shot_filter(plan, shots[0], tw, th)])
        return [{"kind": "ffmpeg", "out": out, "tier": tier, "size": [tw, th],
                 "argv": [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", src, "-vf", vf, "-frames:v", "1", out]}]
    audio = ["-c:a", "copy"] if (plan["audio"] or {}).get("copy", True) else ["-c:a", "aac", "-b:a", "256k"]
    venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    fpsf = plan["fps"]["filter"]
    if len(shots) == 1 and shots[0]["static"]:
        vf = ",".join(_pre_filters(pr) + [fpsf, shot_filter(plan, shots[0], tw, th)])
        argv = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", src, "-vf", vf, "-map", "0:v:0", "-map", "0:a:0?", *venc, *audio, out]
        return [{"kind": "ffmpeg", "out": out, "tier": tier, "size": [tw, th], "argv": argv}]
    W, H = pr["display_w"], pr["display_h"]
    rd = [ffmpeg, "-hide_banner", "-v", "error", "-i", src, "-an", "-vf", ",".join(_pre_filters(pr) + [fpsf]), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    wr = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{tw}x{th}", "-r", plan["fps"]["fps"], "-thread_queue_size", "128", "-i", "-",
          "-i", src, "-map", "0:v:0", "-map", "1:a:0?", *venc, *audio, out]
    return [{"kind": "pipe", "out": out, "tier": tier, "size": [tw, th], "src_size": [W, H], "reader": rd, "writer": wr}]


def _bgr(hexcol):
    h = (hexcol or "#000000").lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def run_pipe(plan: dict, cmd: dict) -> int:
    """execute a 'pipe' command: decode (fps policy applied) -> per frame crop / fit via cv2 -> encode with the source audio. -> frames written."""
    import cv2
    import numpy as np
    W, H = cmd["src_size"]
    tw, th = cmd["size"]
    shots = plan["shots"]
    rd = subprocess.Popen(cmd["reader"], stdout=subprocess.PIPE)
    wr = subprocess.Popen(cmd["writer"], stdin=subprocess.PIPE)
    fsz = W * H * 3
    grounds = {s["index"]: np.full((th, tw, 3), _bgr(s.get("ground")), np.uint8) for s in shots if s["mode"] == "fit"}
    i = 0
    while True:
        buf = rd.stdout.read(fsz)
        if len(buf) < fsz:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        sh = next((s for s in shots if s["f0"] <= i < s["f1"]), shots[-1])
        if sh["mode"] == "crop":
            ww, wh = sh["window"]
            k = min(max(i - sh["f0"], 0), len(sh["path"]) - 1) if sh["path"] else 0
            x, y = sh["path"][k] if sh["path"] else sh["xy"]
            x0, y0 = min(max(int(round(x)), 0), W - ww), min(max(int(round(y)), 0), H - wh)
            out = cv2.resize(frame[y0:y0 + wh, x0:x0 + ww], (tw, th), interpolation=cv2.INTER_AREA)
        else:
            sw, shh, px, py = fit_geometry(W, H, tw, th)
            out = grounds[sh["index"]].copy()
            out[py:py + shh, px:px + sw] = cv2.resize(frame, (sw, shh), interpolation=cv2.INTER_AREA)
        wr.stdin.write(np.ascontiguousarray(out).tobytes())
        i += 1
    wr.stdin.close()
    rd.stdout.close()
    rd.wait()
    rc = wr.wait()
    if rc:
        raise RuntimeError(f"ffmpeg writer exited {rc}")
    return i


def execute(plan: dict, cmds: list) -> list:
    outs = []
    for c in cmds:
        Path(c["out"]).parent.mkdir(parents=True, exist_ok=True)
        if c["kind"] == "ffmpeg":
            subprocess.run(c["argv"], check=True)
        else:
            run_pipe(plan, c)
        outs.append(c["out"])
    return outs


# ---------------------------------------------------------------------------------------------------------------- frame helpers (box side)
def ground_colour(rgb, border: float = 0.06) -> str:
    """flat ground colour of a frame: per-channel median of its border strips (numpy HxWx3, RGB) -> '#rrggbb'."""
    import numpy as np
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    b = max(1, int(round(border * min(h, w))))
    px = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return _hex(np.median(px, axis=0))


def grab_frame(path: str, t: float, pr: dict, width: int = 96, ffmpeg: str = "ffmpeg"):
    import numpy as np
    h = max(2, int(round(width * pr["display_h"] / pr["display_w"])))
    raw = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    if len(raw) < width * h * 3:
        raise RuntimeError("frame grab failed")
    return np.frombuffer(raw[:width * h * 3], np.uint8).reshape(h, width, 3)


def detect_cuts(path: str, thr: float = 0.35, ffmpeg: str = "ffmpeg") -> list:
    """scene cuts (s) by ffmpeg scene score; slow on long clips, box side."""
    import re
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path), "-an", "-vf", f"select='gt(scene,{thr})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    return [round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]


# ---------------------------------------------------------------------------------------------------------------- CLI
def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if p else None


def cmd_plan(a) -> int:
    pr = probe(a.video)
    cuts = _load(a.cuts) if (a.cuts and a.cuts != "detect") else (detect_cuts(a.video) if a.cuts == "detect" else None)

    def gfn(t):
        try:
            return ground_colour(grab_frame(a.video, t, pr))
        except Exception as e:                   # no numpy / unreadable frame: black ground, flagged in warnings
            print("ground colour failed:", e, file=sys.stderr)
            return "#000000"
    plan = build_plan(pr, a.tier, _load(a.track), (_load(a.captions) or None) if a.captions else None, cuts, gfn, a.format)
    Path(a.o).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    sm = ", ".join(f"{s['mode']}({s['reason']})" for s in plan["shots"])
    print(f"{a.o}: {pr['kind']} {pr['display_w']}x{pr['display_h']} -> {plan['ratio']}; fps {plan['fps']['src'] if plan['fps'] else '-'} -> "
          f"{plan['fps']['fps'] if plan['fps'] else '-'}; shots: {sm}; max_tier {plan['max_tier']}; effective {plan['effective_tier']}")
    for w in plan["warnings"]:
        print("WARNING:", w)
    return 0


def cmd_render(a) -> int:
    plan = _load(a.plan)
    if a.src:
        plan["source"]["path"] = a.src
    cmds = render_commands(plan, a.tier, a.o, ffmpeg=a.ffmpeg)
    if a.dry_run:
        for c in cmds:
            print(json.dumps(c, indent=1))
        return 0
    for o in execute(plan, cmds):
        print(o)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("video")
    p.add_argument("--track")
    p.add_argument("--captions")
    p.add_argument("--cuts", help="json list of cut times (s), or 'detect'")
    p.add_argument("--tier", default="preview", choices=TIERS)
    p.add_argument("--format", default="9:16", choices=["9:16", "16:9"])
    p.add_argument("-o", default="ingest.json")
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("render")
    r.add_argument("plan")
    r.add_argument("--tier", default="preview", choices=TIERS)
    r.add_argument("-o", default="ingest_out")
    r.add_argument("--src", help="override the source path stored in the plan")
    r.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_render)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
