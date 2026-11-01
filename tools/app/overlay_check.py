#!/usr/bin/env python
"""overlay_check.py - did the OVERLAY route leave the user's picture alone?  (box side; needs ffprobe + ffmpeg + numpy + cv2)

    python tools/app/overlay_check.py SOURCE.mp4 FINAL.mp4 [--start S --frames N] [--report overlay_report.json] [--mask-video MASK.mp4] [--json out.json]

SOURCE.mp4   the user's clip. With --start (seconds) / --frames the window the job used is compared (the job's overlay_report.json has `window` and `frames`; --report reads both).
FINAL.mp4    the job's final.mp4 (out dir).

Checks (exit code 1 when any FAILS; 0 when all pass):
  1 size      width and height equal                                                                                       (the engine ran at the source size, no 9:16 crop, no tier)
  2 fps       r_frame_rate equal (exact rational)                                                                           (no resample, no halving)
  3 frames    frame count equal; duration equal within one frame                                                            (same cuts at the same frames)
  4 picture   per frame, outside the type mask: mean abs diff <= --mad (default 3.0 levels) and PSNR >= --psnr (default 35 dB). The mask is
                --mask-video   a black / white video (white = type), when you have one; else
                --report       the word windows (overlay_report.json word_frames, frames padded by --pad): frames OUTSIDE every window are compared whole;
                               frames INSIDE a window are compared outside the pixels that changed by more than --thr (the type), and the changed share must stay <= --max-changed (default 0.45)
                none of the above: the whole frame of every frame is compared (a clip with type fails unless --max-changed is given: then the changed share is all that is checked)
  5 cuts      the source's cuts are cut frames in the final too: frame-to-frame jumps (mean abs gray diff > 25 AND > 4x the clip's median jump) sit on the same frame numbers (+-0)

Codec noise: the final is an x264 crf 14 re-encode of near-lossless frames, so a flat region differs by 1-2 levels; the defaults allow that and nothing more.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from fractions import Fraction


def probe(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    s = (j.get("streams") or [{}])[0]
    n, d = (s.get("r_frame_rate") or "0/1").split("/")
    return dict(width=int(s.get("width") or 0), height=int(s.get("height") or 0), fps=Fraction(int(n), int(d or 1)) if int(d or 1) else Fraction(0), frames=int(s.get("nb_read_frames") or s.get("nb_frames") or 0),
                duration=float((j.get("format") or {}).get("duration") or 0.0))


def frames_of(path, w: int, h: int, start: float = 0.0, limit: int | None = None, gray: bool = False):
    """frames as uint8 arrays via an ffmpeg pipe, in order, no fps filter."""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.6f}"] if start else []) + ["-i", str(path), "-an"] + (["-frames:v", str(limit)] if limit else []) + \
          ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    c = 1 if gray else 3
    try:
        while True:
            buf = p.stdout.read(w * h * c)
            if len(buf) < w * h * c:
                break
            yield np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def psnr(mse: float) -> float:
    return 99.0 if mse <= 1e-9 else 10.0 * math.log10(255.0 * 255.0 / mse)


MERGE_GAP = 120            # frames: word windows closer than this are one window (the word-to-word strand transitions live in the gap between two words' spans, and the strands run across the WHOLE gap:
                          # a 30 fps source with sparse lyrics has 50-frame gaps (box run 20261101-160830-aec4: outline strands on screen in the 41 frames between two words, compared whole-frame at 36). The frames of a gap
                          # are still checked outside the changed pixels (mad / psnr / changed share); only the frames before the first and after the last word are compared whole)


def word_windows(report: dict, pad: int, merge_gap: int = MERGE_GAP) -> list[tuple[int, int]]:
    """[(first, last)] source frames (inclusive) where type can be on screen, from overlay_report.json word_frames (start_frame .. end_frame), padded; windows separated by <= `merge_gap` frames are
    merged (box run R16: frame 98 sat in a one-frame gap between two words' spans, where the braid transition's strands were drawn, and was compared whole-frame)."""
    raw = []
    for wf in report.get("word_frames") or []:
        a, b = wf.get("start_frame", wf.get("hit_frame")), wf.get("end_frame", wf.get("release_frame"))
        if a is None or b is None:
            continue
        raw.append([max(0, int(a) - pad), int(b) + pad])
    out: list[list[int]] = []
    for a, b in sorted(raw):
        if out and a - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def changed_pixels(d, thr: int, min_area: int = 8):
    """0/1 mask of pixels of `d` (uint8, max over channels) above `thr`, minus specks under `min_area` px (same rule as overlay_mode.changed_pixels; a median filter would erase 1-2 px lines)."""
    import cv2
    import numpy as np
    m = (d > thr).astype(np.uint8)
    if m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
        keep[0] = False
        m = keep[lab].astype(np.uint8)
    return m


def cut_frames(path, w: int, h: int, start: float, n: int, jump: float = 25.0, rel: float = 4.0) -> list[int]:
    """frames i where the mean abs gray diff to frame i-1 is a real cut: above `jump` levels AND above `rel` x the clip's median frame-to-frame diff. (A fixed 40 split two equal soft cuts: 39.0 in the source,
    40.7 in the final, box run R16.)"""
    import numpy as np
    prev, js = None, []
    for f in frames_of(path, w, h, start, n, gray=True):
        js.append(0.0 if prev is None else float(np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = f
    med = float(np.median(js)) if js else 0.0
    return [i for i, j in enumerate(js) if i > 0 and j > jump and j > rel * med]


def compare(src_path, fin_path, start: float = 0.0, nframes: int | None = None, report: dict | None = None, mask_video=None, mad_max: float = 3.0, psnr_min: float = 35.0, thr: int = 24,
            max_changed: float | None = None, pad: int = 4) -> dict:
    """the whole check -> {ok, checks: {name: {ok, ...}}, per_frame_worst}. Pure function of the two files (+ optional report / mask)."""
    import cv2
    import numpy as np
    ps, pf = probe(src_path), probe(fin_path)
    n_src = ps["frames"]
    fps = ps["fps"]
    s0f = int(round(start * float(fps)))
    want = nframes if nframes else max(0, n_src - s0f)
    checks: dict = {}
    checks["size"] = dict(ok=(ps["width"], ps["height"]) == (pf["width"], pf["height"]), source=[ps["width"], ps["height"]], final=[pf["width"], pf["height"]])
    checks["fps"] = dict(ok=ps["fps"] == pf["fps"], source=str(ps["fps"]), final=str(pf["fps"]))
    tol = 1.0 / float(fps) + 1e-3 if fps else 0.05
    checks["frames"] = dict(ok=pf["frames"] == want and abs(pf["duration"] - want / float(fps)) <= tol + 0.02, source_window=want, final=pf["frames"], final_duration=round(pf["duration"], 4),
                            expected_duration=round(want / float(fps), 4))
    if not (checks["size"]["ok"] and checks["frames"]["ok"]):
        return dict(ok=False, checks=checks, note="size / frame count differ: the pixel comparison was skipped")
    w, h = pf["width"], pf["height"]
    wins = word_windows(report, pad) if report else []
    in_win = np.zeros(want, bool)
    for a_, b_ in wins:
        in_win[max(0, a_):min(want, b_ + 1)] = True
    masks = frames_of(mask_video, w, h, 0.0, want, gray=True) if mask_video else None
    mads, psnrs, shares, worst = [], [], [], (0, 0.0)
    bad = []
    for i, (S, F) in enumerate(zip(frames_of(src_path, w, h, start, want), frames_of(fin_path, w, h, 0.0, want))):
        d = np.abs(S.astype(np.int16) - F.astype(np.int16)).max(axis=2)
        if masks is not None:
            m = next(masks) > 127
            keep = ~cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        elif wins and in_win[i]:
            m = cv2.dilate(changed_pixels(d.astype(np.uint8), thr), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
            keep = ~m
            shares.append(float(m.mean()))
        elif max_changed is not None and not wins:
            m = cv2.dilate(changed_pixels(d.astype(np.uint8), thr), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
            keep = ~m
            shares.append(float(m.mean()))
        else:
            keep = np.ones((h, w), bool)
        if not keep.any():
            continue
        diff = (S.astype(np.float32) - F.astype(np.float32))[keep]
        mad, mse = float(np.abs(diff).mean()), float((diff ** 2).mean())
        mads.append(mad)
        psnrs.append(psnr(mse))
        if mad > worst[1]:
            worst = (i, mad)
        if mad > mad_max or psnr(mse) < psnr_min:
            bad.append(i)
    checks["picture"] = dict(ok=not bad, frames_compared=len(mads), mad_mean=round(float(np.mean(mads)), 3) if mads else None, mad_max=round(max(mads), 3) if mads else None,
                             psnr_min=round(min(psnrs), 2) if psnrs else None, frames_failing=bad[:20], n_failing=len(bad), limits=dict(mad=mad_max, psnr=psnr_min), mask=("mask-video" if mask_video else "word windows" if wins else "whole frame"),
                             frames_inside_word_windows=int(in_win.sum()))
    if shares:
        mx = max_changed if max_changed is not None else 0.45
        checks["type_share"] = dict(ok=max(shares) <= mx, max_changed_share=round(max(shares), 4), mean_changed_share=round(float(np.mean(shares)), 4), limit=mx)
    cs = cut_frames(src_path, w, h, start, want)
    cf = cut_frames(fin_path, w, h, 0.0, want)
    checks["cuts"] = dict(ok=cs == cf, source=cs, final=cf)
    return dict(ok=all(c["ok"] for c in checks.values()), checks=checks, worst_frame=dict(index=worst[0], mad=round(worst[1], 3)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("final")
    ap.add_argument("--start", type=float, default=None, help="window start in the source (s); default from --report, else 0")
    ap.add_argument("--frames", type=int, default=None, help="window length in frames; default from --report, else the rest of the source")
    ap.add_argument("--report", default="", help="overlay_report.json of the job (window, frames, word_frames)")
    ap.add_argument("--mask-video", default="", help="black/white video, white = type, same size and frame count as the final")
    ap.add_argument("--mad", type=float, default=3.0, help="max mean abs diff (levels, 0-255) outside the type, per frame (the crf 14 re-encode alone measures ~2.2 on grainy footage; a +14 level grade measures 14)")
    ap.add_argument("--psnr", type=float, default=35.0, help="min PSNR (dB) outside the type, per frame (the re-encode alone: ~40 dB)")
    ap.add_argument("--thr", type=int, default=24, help="a pixel changed by more than this is 'type' (word-window frames without a mask video)")
    ap.add_argument("--max-changed", type=float, default=None, help="max share of a frame the type may cover (default 0.45 when word windows are known)")
    ap.add_argument("--pad", type=int, default=4, help="frames added before / after each word window")
    ap.add_argument("--json", default="", help="write the result here")
    a = ap.parse_args(argv)
    rep = json.load(open(a.report, encoding="utf-8")) if a.report else None
    start = a.start if a.start is not None else float(((rep or {}).get("window") or [0.0])[0])
    nfr = a.frames if a.frames is not None else ((rep or {}).get("frames"))
    res = compare(a.source, a.final, start, nfr, rep, a.mask_video or None, a.mad, a.psnr, a.thr, a.max_changed, a.pad)
    for name, c in res["checks"].items():
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {name:10s} " + json.dumps({k: v for k, v in c.items() if k != 'ok'}))
    if res.get("note"):
        print("NOTE:", res["note"])
    print("OVERLAY CHECK:", "PASS - the picture is the source's outside the type" if res["ok"] else "FAIL - the picture changed")
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
