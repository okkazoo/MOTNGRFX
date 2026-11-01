"""Matte preview: the Alpha-Gen matte PNG frames of a run (`<prefix>_shots/sK_matte/NNNNN.png`, written by MographArchive) stacked shot after shot
into ONE gray mp4 at 24 fps, at the matte resolution. The PNGs are only read.

    python tools/app/matte_preview.py one <..._shots dir> <out.mp4>
    python tools/app/matte_preview.py scan <version dir>          # every `*_shots` folder under <version>/out and <version>/local -> <version>/review/<name>_matte_preview.mp4
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

FPS = 24
FFMPEG_S = 540.0      # hard ceiling of the one ffmpeg call (run_batch.finalize bounds the whole step at 600 s)


def shot_dirs(shots_dir: str) -> list[str]:
    """`sK_matte` folders of one `_shots` dir in shot order."""
    found = []
    for n in os.listdir(shots_dir):
        m = re.fullmatch(r"s(\d+)_matte", n)
        if m and os.path.isdir(os.path.join(shots_dir, n)):
            found.append((int(m.group(1)), os.path.join(shots_dir, n)))
    return [p for _, p in sorted(found)]


def build(shots_dir: str, out_mp4: str, fps: int = FPS) -> dict:
    """-> {'path', 'shots', 'frames', 'size'}; raises if there is no matte or ffmpeg fails."""
    dirs = shot_dirs(shots_dir)
    if not dirs:
        raise FileNotFoundError(f"no sK_matte folder in {shots_dir}")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for d in dirs:
        cmd += ["-framerate", str(fps), "-i", os.path.join(d, "%05d.png")]
    n = len(dirs)
    # gray, same size for every shot (they share the matte size; scale only guards a mismatch), then concat
    parts = "".join(f"[{i}:v]format=gray,setsar=1[v{i}];" for i in range(n))
    cat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p,crop=trunc(iw/2)*2:trunc(ih/2)*2[o]"
    cmd += ["-filter_complex", parts + cat, "-map", "[o]", "-r", str(fps), "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-movflags", "+faststart", out_mp4]
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out after {FFMPEG_S:.0f}s")
    if r.returncode:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-600:])
    frames = sum(len([f for f in os.listdir(d) if f.endswith(".png")]) for d in dirs)
    return {"path": out_mp4, "shots": n, "frames": frames}


def scan(version_dir: str) -> list[dict]:
    res = []
    for sub in ("out", "local"):
        base = os.path.join(version_dir, sub)
        for root, dirs, _ in os.walk(base):
            for d in list(dirs):
                if d.endswith("_shots") and shot_dirs(os.path.join(root, d)):
                    name = d[: -len("_shots")]
                    out = os.path.join(version_dir, "review", f"{name}_matte_preview.mp4")
                    try:
                        res.append(dict(build(os.path.join(root, d), out), shots_dir=os.path.join(root, d)))
                    except Exception as e:      # noqa: BLE001
                        res.append({"shots_dir": os.path.join(root, d), "error": str(e)[:300]})
    return res


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "one":
        print(build(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "scan":
        for x in scan(sys.argv[2]):
            print(x)
    else:
        raise SystemExit(__doc__)
