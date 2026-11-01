#!/usr/bin/env python
"""Window rule check for the own-video route (user 2026-10-30): mattes / control nets only over the output window, per shot, trimmed first.

    python tools/app/own_video_v2_check.py --version-dir 26_own-video/v002 [--arch DIR]

Checks: sum(spans) == the window (576 frames); every Alpha-Gen PNG stack (`*_alpha_h<j>_image_*`) belongs to ONE piece (segments.json: a real-cut segment cut at the shot boundaries) and has
<= piece frames + the 8k+1 pad frames; every piece lies inside one shot's span and inside the window; no control-net stacks (depth / canny / union-control) exist in the version;
the archive's per-shot matte stacks (sK_matte, if --arch) have <= span + 8k+1 pad frames. Prints one row per stack and exits 1 on a violation.
"""
import argparse
import glob
import json
import os
import re
import sys


def pad_frames(n):
    return ((int(n) - 1 + 7) // 8) * 8 + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", required=True)
    ap.add_argument("--arch")
    a = ap.parse_args()
    work = os.path.join(a.version_dir, "work")
    sp = json.load(open(os.path.join(work, "spans.json")))
    segs = json.load(open(os.path.join(work, "segments.json")))
    spans, window = sp["spans"], 576
    bad = []
    print(f"spans {spans} sum {sum(spans)} window {window}")
    if sum(spans) != window:
        bad.append("sum of spans != window")
    starts = sp["starts"]
    cnt = {}
    for f in glob.glob(os.path.join(a.version_dir, "out", "*_alpha_h*_image_*.png")):
        m = re.search(r"_alpha_h(\d+)_image_", f)
        cnt[int(m.group(1))] = cnt.get(int(m.group(1)), 0) + 1
    print("matte stacks (Alpha-Gen PNG frames per piece)")
    for s in segs:
        if not s["person"]:
            continue
        k = s["shot"]
        lo, hi = starts[k - 1], starts[k - 1] + spans[k - 1]
        n = cnt.get(s["job"], 0)
        limit = pad_frames(max(s["n"], 9))
        ok = lo <= s["a"] and s["b"] <= hi and s["b"] <= window and n <= limit and n <= spans[k - 1] + 8
        print(f"  h{s['job']:<2d} shot s{k} span {spans[k - 1]:3d} piece {s['a']:3d}-{s['b']:3d} ({s['n']:2d} frames) stack {n:3d} <= {limit:3d} (pad) {'OK' if ok else 'VIOLATION'}")
        if not ok:
            bad.append(f"h{s['job']}")
    ctl = [f for f in glob.glob(os.path.join(a.version_dir, "out", "*")) if re.search(r"depth|canny|union|control", os.path.basename(f), re.I) and not re.search(r"redesign", os.path.basename(f))]
    print("control-net stacks:", len(ctl))
    if a.arch:
        print("archive stacks")
        for k, n in enumerate(spans, 1):
            c = len(glob.glob(os.path.join(a.arch, f"s{k}_matte", "*.png")))
            ok = c <= n + 8 and c == pad_frames(n)
            print(f"  s{k} span {n:3d} matte frames {c:3d} (8k+1 pad {pad_frames(n)}) {'OK' if ok else 'VIOLATION'}")
            if not ok:
                bad.append(f"s{k}")
    print("RESULT", "OK" if not bad else "VIOLATIONS " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
