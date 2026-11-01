"""MographTypePasses: the 3D type passes of a piece as two ``tools/fusion/type_passes.py`` subprocesses (DESIGN.md sections 0, 1, 4).

    piece (+ music?) -> MOGRAPH_TYPE {dir_control, dir_layers, manifest, job}, sheet IMAGE ("OUT type")

- control job (blocking): the passes the LTX control video is cut from, at ``sizes.control`` (576x1024). The sheet is built from it.
- layers job (async, final quality only): every pass compose reads, at ``sizes.type`` (1088x1920). Compose calls ``wait_type()``.
- draft: ``sizes.type`` is 576x1024, there is no control job (a draft shot has no type guide: all words flat, and a still preview has no LTX at all): one blocking layers job, dir_control = dir_layers.
- Subprocesses, not in-process multiprocessing: spawned workers would re-import ComfyUI's main.py. OMP/MKL/OPENBLAS threads = 1 so
  ``procs`` single-threaded workers do not fight over the cores.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from comfy_api.latest import io

try:
    import folder_paths
except Exception:   # pragma: no cover  (unit tests without ComfyUI's folder_paths)
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY, FPS, MographMusic, MographPiece, MographType

ROOT = Path(ensure_root_on_path())
TYPE_PASSES_PY = ROOT / "tools" / "fusion" / "type_passes.py"
DEFAULT_PROCS = 10
SIZES = {"preview": {"control": (384, 704), "type": (384, 704)}, "draft": {"control": (576, 1024), "type": (576, 1024)}, "final": {"control": (576, 1024), "type": (1088, 1920)}}


# --------------------------------------------------------------------------------------------- the MOGRAPH_TYPE value
class TypeHandle(dict):
    """MOGRAPH_TYPE: a plain dict {dir_control, dir_layers, manifest, job, log} with a ``.wait()`` for the async layers job."""

    def wait(self, timeout=None) -> "TypeHandle":
        return wait_type(self, timeout)


def wait_type(t: dict, timeout=None) -> dict:
    """block until the async layers job has finished (raises if it failed) and load its manifest into ``t['manifest_layers']``."""
    job = t.get("job")
    if job is not None:
        rc = job.wait(timeout)
        log = t.get("log")
        if log is not None and not getattr(log, "closed", True):
            log.close()
        if rc != 0:
            tail = ""
            try:
                tail = Path(t["log_path"]).read_text(errors="replace")[-1500:]
            except Exception:   # noqa: BLE001
                pass
            raise RuntimeError(f"type_passes (layers) failed with rc {rc}\n{tail}")
    mp = Path(t["dir_layers"]) / "manifest.json"
    if mp.exists() and "manifest_layers" not in t:
        t["manifest_layers"] = json.loads(mp.read_text(encoding="utf-8"))
    return t


# --------------------------------------------------------------------------------------------- helpers
def _work_dir() -> Path:
    base = Path(folder_paths.get_temp_directory()) if folder_paths is not None else Path(os.environ.get("TEMP", "/tmp"))
    d = base / "mograph" / f"type_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thread_env() -> dict:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    return env


def build_spec(piece: dict, music: dict | None = None) -> dict:
    """the type spec handed to type_passes: ``piece['type']`` with fps / duration / bpm / beats / downbeats filled from the music
    analysis when given (as Piece._type does). A planner that already filled them is left alone when ``music`` is None."""
    spec = copy.deepcopy(piece["type"])
    spec["fps"] = int(piece.get("fps", FPS))
    if music:
        spec["duration"] = float(music["cut"])
        spec["bpm"] = float(music["bpm"])
        spec["beats"] = music["beats"]
        spec["downbeats"] = music["downbeats"]
    elif "duration" not in spec:
        lay = piece.get("layout") or {}
        cut = (piece.get("audio") or {}).get("cut")
        spec["duration"] = float(cut) if cut else max((v["start"] + v["frames"] for v in lay.values()), default=0) / spec["fps"]
    spec.setdefault("bar", 4)
    return spec


def pass_sizes(piece: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    from tools.fusion import plan_rules
    q = piece.get("quality") if piece.get("quality") in ("preview", "draft") else "final"
    fmt = plan_rules.format_of(piece)                      # 9:16 | 16:9 swaps the sides of the portrait defaults
    sz = piece.get("sizes") or {}
    ctl = tuple(sz.get("control") or plan_rules.orient(SIZES[q]["control"], fmt))
    typ = tuple(sz.get("type") or plan_rules.orient(SIZES[q]["type"], fmt))
    return (int(ctl[0]), int(ctl[1])), (int(typ[0]), int(typ[1]))


def _launch(spec_path: Path, out: Path, size, passes, procs: int, frames=None) -> tuple[subprocess.Popen, object, Path]:
    cmd = [sys.executable, str(TYPE_PASSES_PY), str(spec_path), "--out", str(out), "--size", f"{size[0]}x{size[1]}",
           "--passes", ",".join(passes), "--procs", str(int(procs)), "--no-mp4", "--overwrite"]
    if frames is not None:
        cmd += ["--frames", f"{int(frames[0])}:{int(frames[1])}"]
    log_path = out.parent / f"{out.name}.log"
    log = open(log_path, "w", encoding="utf-8")
    job = subprocess.Popen(cmd, cwd=str(ROOT), env=thread_env(), stdout=log, stderr=subprocess.STDOUT)
    return job, log, log_path


def _finish(job: subprocess.Popen, log, log_path: Path, what: str) -> None:
    rc = job.wait()
    log.close()
    if rc != 0:
        raise RuntimeError(f"type_passes ({what}) failed with rc {rc}\n{log_path.read_text(errors='replace')[-1500:]}")


def start_type_passes(piece: dict, music: dict | None = None, procs: int = DEFAULT_PROCS, work: Path | None = None,
                      frames=None) -> TypeHandle:
    """run the control job (blocking) and start the layers job (async; blocking in draft). -> TypeHandle."""
    from tools.fusion import recipe_eval

    work = Path(work) if work else _work_dir()
    work.mkdir(parents=True, exist_ok=True)
    procs = max(1, min(int(procs), os.cpu_count() or 1))
    spec = build_spec(piece, music)
    spec_path = work / "type_spec.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (csize, tsize) = pass_sizes(piece)
    layer_passes = recipe_eval.need_passes(piece)
    draft = piece.get("quality") == "draft"
    t0 = time.time()
    if draft:                                   # no type-guided LTX shot (flat words only): no control video is cut, one blocking job at the layers size
        out = work / "layers"
        job, log, lp = _launch(spec_path, out, tsize, layer_passes, procs, frames)
        _finish(job, log, lp, "draft layers")
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        return TypeHandle(dir_control=str(out), dir_layers=str(out), manifest=man, job=None, log=None, log_path=str(lp),
                          passes=layer_passes, seconds_control=time.time() - t0)
    ctl_passes = recipe_eval.control_passes(piece)
    ctl_out = work / "control"
    job, log, lp = _launch(spec_path, ctl_out, csize, ctl_passes, procs, frames)
    _finish(job, log, lp, "control")
    man = json.loads((ctl_out / "manifest.json").read_text(encoding="utf-8"))
    sec = time.time() - t0
    lay_out = work / "layers"
    job, log, lp = _launch(spec_path, lay_out, tsize, layer_passes, procs, frames)       # async: compose waits on it
    return TypeHandle(dir_control=str(ctl_out), dir_layers=str(lay_out), manifest=man, job=job, log=log, log_path=str(lp),
                      passes=layer_passes, seconds_control=sec)


def make_sheet_image(t: dict, width: int = 216, n: int = 6) -> np.ndarray:
    """contact sheet (frames x passes) of the control render -> HxWx3 float32 RGB 0..1."""
    from tools.fusion import type_passes

    ctl = Path(t["dir_control"])
    passes = [p for p in t["manifest"]["passes"] if p != "flow"][:4]
    path = ctl.parent / f"{ctl.name}_sheet.png"
    type_passes.make_sheet(str(ctl), str(path), n=n, passes=passes, width=width)
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(im[..., ::-1].astype(np.float32) / 255.0)


# --------------------------------------------------------------------------------------------- node
class MographTypePasses(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographTypePasses",
            display_name="MOTNGRFX Type Passes",
            category=CATEGORY,
            description="Renders the 3D type passes with tools/fusion/type_passes.py in subprocesses. The control pass "
                        "(576x1024) blocks and feeds the sheet and the shot control videos; the compose layers "
                        "(1088x1920) render asynchronously and MOTNGRFX Compose waits for them. Draft: one 576x1024 job.",
            inputs=[
                MographPiece.Input("piece"),
                MographMusic.Input("music", optional=True,
                                   tooltip="Music analysis (cut, bpm, beats, downbeats); filled into the type spec when "
                                           "the piece's type block does not carry them."),
                io.Int.Input("procs", default=DEFAULT_PROCS, min=1, max=64,
                             tooltip="Worker processes per type_passes job (single-threaded BLAS in each)."),
            ],
            outputs=[MographType.Output(display_name="type"), io.Image.Output(display_name="sheet")],
        )

    @classmethod
    def execute(cls, piece: dict, procs: int = DEFAULT_PROCS, music: dict | None = None) -> io.NodeOutput:
        t = start_type_passes(piece, music, procs)
        sheet = torch.from_numpy(make_sheet_image(t))[None]
        return io.NodeOutput(t, sheet)


NODES = [MographTypePasses]
