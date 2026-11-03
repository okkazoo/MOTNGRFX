"""One Alpha-Gen matte for one frame of the user's video: the storyboard frame a shot's words sit on (user 2026-11-03: "single mattes for the frame that we have in the gui
that we want to put text on, only those; the full matte run can come at Make it time").

Alpha-Gen is a video model (8k+1 frames), so a "single" matte is the smallest clip it takes: 9 real frames starting at the shot's frame, never crossing the shot's end (the last
frame is held when the shot is shorter), run at the same size as the Make it run (`overlay_mode.matte_size`). Only the matte of the first frame is kept. The key is the video
file + the frame + the matte size: words, fonts and layout cannot enter it, so editing them never redoes a matte. One job at a time, in order; ComfyUI queues behind other work.

    fm = FrameMattes(output_dir, input_dir, port)
    fm.request(video, t, end) -> ("ready", path) | ("making", None) | ("error", message)    # raises ValueError on a bad request
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAMES = 9                 # the smallest Alpha-Gen clip above one frame (8k+1)
SUBDIR = "motngrfx_app/frame_mattes"
TIMEOUT_S = 900.0          # once it runs; waiting behind other queued work does not count


def clip_plan(t: float, end: float | None, fps: float, nframes: int | None = None, frames: int = FRAMES) -> tuple[int, int, int]:
    """-> (first source frame, real frames to take, frames to hold at the end). The clip stays inside the shot: [t, end) and the video's own length."""
    first = max(0, int(round(float(t) * float(fps) - 1e-6)))
    stop = first + frames
    if end is not None:
        stop = min(stop, int(math.ceil(float(end) * float(fps) - 1e-6)))
    if nframes:
        first = min(first, int(nframes) - 1)
        stop = min(stop, int(nframes))
    real = max(1, min(frames, stop - first))
    return first, real, frames - real


def matte_key(name: str, size: int, mtime_ns: int, first: int, matte: tuple[int, int]) -> str:
    """cache key: the file's identity + the frame + the matte size (nothing the page edits)."""
    return hashlib.sha256(json.dumps([name, int(size), int(mtime_ns), int(first), list(matte)]).encode("utf-8")).hexdigest()[:24]


def safe_video(input_dir, video) -> Path:
    """the video as a file inside the input folder, or ValueError."""
    if not isinstance(video, str) or not video.strip():
        raise ValueError("video: a file name in the ComfyUI input folder is required")
    base = Path(input_dir).resolve()
    p = (base / video).resolve()
    if base not in p.parents or not p.is_file():
        raise ValueError(f"video: {video} is not in the input folder")
    return p


def number(value, label: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: not a number") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{label}: not a time")
    return v


def probe(path: Path) -> tuple[int, int, float, int | None]:
    """-> (width, height, fps, frame count if the file states it)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60)
    streams = json.loads(out.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"{path.name}: ffprobe cannot read the clip")
    s = streams[0]
    a, _, b = str(s.get("r_frame_rate") or "0/1").partition("/")
    fps = float(a) / float(b or 1)
    if fps <= 0:
        raise RuntimeError(f"{path.name}: no frame rate")
    try:
        n = int(s.get("nb_frames"))
    except (TypeError, ValueError):
        n = None
    return int(s["width"]), int(s["height"]), fps, n


def cut_clip(src: Path, dst: Path, first: int, real: int, pad: int, fps: float, size: tuple[int, int]) -> None:
    """the Alpha-Gen input: the frames at the matte size, tagged with the source rate like the Make it run's clips."""
    vf = f"scale={size[0]}:{size[1]}:flags=area" + (f",tpad=stop_mode=clone:stop={pad}" if pad > 0 else "")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{first / fps:.6f}", "-i", str(src), "-an", "-frames:v", str(real), "-vf", vf, "-frames:v", str(real + pad),
                    "-r", f"{fps:.6f}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)], check=True, capture_output=True, timeout=180)


def alpha_graph(ref_video: str, size: tuple[int, int], prefix: str) -> dict:
    from tools.fusion import ltx_ic
    g = ltx_ic.build("alpha", ref_video=ref_video, prompt="", size=tuple(size), frames=FRAMES, seed=1, out_prefix=prefix, crop="disabled", png=True)
    for nid in [k for k, v in g.items() if v["class_type"] == "SaveVideo"]:
        del g[nid]                                                     # the PNGs are all this needs
    return g


class Comfy:
    def __init__(self, port: int):
        self.base, self.cid = f"http://127.0.0.1:{int(port)}", uuid.uuid4().hex

    def _get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def run(self, graph: dict) -> dict:
        """queue the graph, wait for it, -> its history entry; RuntimeError with ComfyUI's reason."""
        req = urllib.request.Request(self.base + "/prompt", data=json.dumps({"prompt": graph, "client_id": self.cid}).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                pid = json.loads(r.read().decode("utf-8"))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the matte graph: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None
        t_end = time.time() + TIMEOUT_S
        while time.time() < t_end:
            time.sleep(1.0)
            try:
                h = self._get(f"/history/{pid}").get(pid)
                if not h:
                    if pid in json.dumps(self._get("/queue").get("queue_pending") or []):
                        t_end = time.time() + TIMEOUT_S
                    continue
            except (urllib.error.URLError, OSError):
                continue
            st = h.get("status") or {}
            if st.get("status_str") == "error":
                msg = next((f"{m[1].get('node_type')}: {str(m[1].get('exception_message'))[:400]}" for m in st.get("messages") or [] if m and m[0] == "execution_error"), "")
                raise RuntimeError(f"Alpha-Gen failed: {msg or st.get('status_str')}")
            if st.get("completed") or st.get("status_str") == "success":
                return h
        raise RuntimeError(f"Alpha-Gen did not finish in {int(TIMEOUT_S)} s")


def first_png(history: dict, output_dir: Path) -> Path:
    """the first frame's PNG of a finished graph."""
    names = []
    for node in (history.get("outputs") or {}).values():
        for im in node.get("images") or []:
            if str(im.get("filename", "")).lower().endswith(".png"):
                names.append(output_dir / (im.get("subfolder") or "") / im["filename"])
    if not names:
        raise RuntimeError("Alpha-Gen finished but saved no PNG")
    return sorted(names, key=str)[0]


class FrameMattes:
    """the mattes on disk + one worker thread. `make` is replaceable in the tests."""

    def __init__(self, output_dir, input_dir, port: int, make=None):
        self.out, self.inp, self.port = Path(output_dir), Path(input_dir), int(port)
        self.dir = self.out / SUBDIR
        self.make = make or self._make
        self.lock = threading.Lock()
        self.pending: dict[str, bool] = {}
        self.errors: dict[str, str] = {}
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

    def request(self, video, t, end=None, peek: bool = False) -> tuple[str, Path | str | None]:
        src = safe_video(self.inp, video)
        tv = number(t, "t")
        ev = number(end, "end") if end not in (None, "") else None
        from tools.app import overlay_mode as OM
        w, h, fps, n = probe(src)
        size = OM.matte_size(OM.even(w), OM.even(h), "draft")
        first, real, pad = clip_plan(tv, ev, fps, n)
        st = src.stat()
        key = matte_key(src.name, st.st_size, st.st_mtime_ns, first, size)
        png = self.dir / f"{key}.png"
        if png.is_file():
            return "ready", png
        if peek:                                                       # the page only asks if one exists; it makes its own preview when not
            return "none", None
        with self.lock:
            if key in self.errors:
                return "error", self.errors.pop(key)                   # reported once: the next request tries again
            if key not in self.pending:
                self.pending[key] = True
                self.q.put((key, src, first, real, pad, fps, size, png))
                if self.worker is None or not self.worker.is_alive():
                    self.worker = threading.Thread(target=self._drain, name="frame-mattes", daemon=True)
                    self.worker.start()
        return "making", None

    def _drain(self) -> None:
        while True:
            try:
                job = self.q.get(timeout=30)
            except queue.Empty:
                return
            key = job[0]
            try:
                self.make(*job)
            except Exception as e:   # noqa: BLE001  (the page shows the reason)
                with self.lock:
                    self.errors[key] = str(e)[:600] or type(e).__name__
            finally:
                with self.lock:
                    self.pending.pop(key, None)

    def _make(self, key, src, first, real, pad, fps, size, png) -> None:
        name = f"frame_matte_{key}.mp4"
        clip = self.inp / name
        cut_clip(src, clip, first, real, pad, fps, size)
        try:
            hist = Comfy(self.port).run(alpha_graph(name, size, f"{SUBDIR}/{key}/m"))
            made = first_png(hist, self.out)
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = png.with_suffix(".tmp")
            shutil.copyfile(made, tmp)
            tmp.replace(png)
        finally:
            clip.unlink(missing_ok=True)
