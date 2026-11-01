"""Lazy-gating spike (DESIGN.md "Spike results"): stand-in nodes with the same wiring contract as the real
ShotSlot -> LTX chain -> DecodeClip -> Gather. Loaded only when env ``MOGRAPH_SPIKE=1`` (see ``__init__``).

    SpikeCount (n)                       -> piece dict {"layout": {"s1".."sn"}}  (stand-in for MographPiecePlan)
    SpikeSlot  (index, piece, prev lazy) -> slot dict            (stand-in for MographShotSlot)
    SpikeWork  (slot)                    -> clip dict            (stand-in for the sampler chain + DecodeClip; slow)
    SpikeGather(piece, clip_1..6 lazy)   -> list of clips        (stand-in for MographFootageGather)

Every execute() appends ``<node> <sid> <enter|exit>`` to ``MOGRAPH_SPIKE_LOG`` (default ComfyUI/temp/mograph_spike.log).
"""
from __future__ import annotations

import os
import time

from ._types import (MAX_SHOTS, MographClip, MographClips, MographPiece, MographMusic, MographRequest, MographType, active_sids,
                     sid_of)
from comfy_api.latest import io

_LOG = os.environ.get("MOGRAPH_SPIKE_LOG")


def _log(line: str) -> None:
    path = _LOG
    if not path:
        import folder_paths
        path = os.path.join(folder_paths.get_temp_directory(), "mograph_spike.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f} {line}\n")


class SpikeCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikeCount", display_name="Spike: piece with N shots", category="Mograph/Spike",
                         inputs=[io.Int.Input("n", default=1, min=1, max=MAX_SHOTS)],
                         outputs=[MographPiece.Output(display_name="piece")])

    @classmethod
    def execute(cls, n: int) -> io.NodeOutput:
        _log(f"count n={n}")
        return io.NodeOutput({"layout": {sid_of(k): {"start": 0, "frames": 9, "xf": 0} for k in range(1, int(n) + 1)}})


class SpikeSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographSpikeSlot", display_name="Spike: shot slot", category="Mograph/Spike",
            inputs=[io.Int.Input("index", default=1, min=1, max=MAX_SHOTS), MographPiece.Input("piece"),
                    MographClip.Input("prev", optional=True, lazy=True)],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def check_lazy_status(cls, index: int, piece=None, prev=None) -> list[str]:
        # slot 1 has nothing to wait for; later slots need the previous clip (its last frame is their first frame)
        return ["prev"] if index > 1 and prev is None else []

    @classmethod
    def execute(cls, index: int, piece: dict, prev=None) -> io.NodeOutput:
        _log(f"slot s{index} prev={'none' if prev is None else prev['sid']}")
        return io.NodeOutput(f"prompt {index}")


class SpikeWork(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographSpikeWork", display_name="Spike: shot work", category="Mograph/Spike",
            inputs=[io.String.Input("prompt"), io.Int.Input("index", default=1, min=1, max=MAX_SHOTS),
                    io.Float.Input("seconds", default=0.2, min=0.0, max=10.0)],
            outputs=[MographClip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, prompt: str, index: int, seconds: float) -> io.NodeOutput:
        _log(f"work s{index} enter")
        time.sleep(float(seconds))
        _log(f"work s{index} exit")
        return io.NodeOutput({"sid": sid_of(index), "u8": None, "start": 0, "fps": 24})


class SpikeGather(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographSpikeGather", display_name="Spike: footage gather", category="Mograph/Spike",
            inputs=[MographPiece.Input("piece")] + [MographClip.Input(f"clip_{k}", optional=True, lazy=True)
                                                    for k in range(1, MAX_SHOTS + 1)],
            outputs=[MographClips.Output(display_name="clips")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id], is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, piece=None, **clips) -> list[str]:
        wired = (cls.hidden.prompt or {}).get(str(cls.hidden.unique_id), {}).get("inputs", {})
        sids = set(active_sids(piece))
        return [k for k, v in clips.items() if v is None and k.replace("clip_", "s") in sids
                and isinstance(wired.get(k), list)]

    @classmethod
    def execute(cls, piece: dict, **clips) -> io.NodeOutput:
        got = [c for _, c in sorted(clips.items()) if c is not None]
        _log("gather got " + ",".join(c["sid"] for c in got))
        return io.NodeOutput(got)


# ----------------------------------------------------------------------------- test harness for footage / inputs
# (stand-ins for WP-A / WP-B nodes so footage.py can be exercised on its own; only loaded with MOGRAPH_SPIKE=1)
class SpikePiece(io.ComfyNode):
    """Loads a P*.json (15_fusion/pieces) and adds layout / slug / sizes / quality like the planner will."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikePiece", display_name="Spike: piece from P*.json", category="Mograph/Spike",
                         inputs=[io.String.Input("path", default="15_fusion/pieces/P02.json"),
                                 io.Int.Input("shots", default=3, min=1, max=MAX_SHOTS),
                                 io.String.Input("quality", default="draft"), io.String.Input("slug", default="")],
                         outputs=[MographPiece.Output(display_name="piece")])

    @classmethod
    def execute(cls, path: str, shots: int, quality: str, slug: str = "") -> io.NodeOutput:
        import json
        from . import ROOT
        p = path if os.path.isabs(path) else os.path.join(ROOT, path)
        j = json.load(open(p, encoding="utf-8"))
        j["shots"] = j["shots"][:int(shots)]
        cf = j.get("compose", {}).get("crossfade", 0.5)
        lay, start = {}, 0
        for k, s in enumerate(j["shots"]):
            c = int(round(cf * j.get("fps", 24))) if k else 0
            start -= c
            n = int(s.get("span") or s["frames"])
            lay[s["id"]] = {"start": start, "frames": n, "xf": c}
            start += n
        j["layout"], j["quality"] = lay, quality
        if slug:
            j["slug"] = slug
        j["sizes"] = {"control": [576, 1024], "type": [576, 1024], "compose": [540, 960]}
        return io.NodeOutput(j)


class SpikeRequest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikeRequest", display_name="Spike: request", category="Mograph/Spike",
                         inputs=[io.Int.Input("seed", default=1, min=0, max=0xFFFFFFFF),
                                 io.String.Input("quality", default="draft"),
                                 io.Image.Input("image", optional=True)],
                         outputs=[MographRequest.Output(display_name="request")])

    @classmethod
    def execute(cls, seed: int, quality: str, image=None) -> io.NodeOutput:
        return io.NodeOutput({"words": "x", "seed": int(seed), "quality": quality, "image": image, "audio": None})


class SpikeMusic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikeMusic", display_name="Spike: music analysis json", category="Mograph/Spike",
                         inputs=[io.String.Input("path", default="15_fusion/v002/local/pieces/P02/analysis.json")],
                         outputs=[MographMusic.Output(display_name="music")])

    @classmethod
    def execute(cls, path: str) -> io.NodeOutput:
        import json
        from . import ROOT
        p = path if os.path.isabs(path) else os.path.join(ROOT, path)
        return io.NodeOutput(json.load(open(p, encoding="utf-8")))


class SpikeType(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikeType", display_name="Spike: type dir", category="Mograph/Spike",
                         inputs=[io.String.Input("dir", default="")], outputs=[MographType.Output(display_name="type")])

    @classmethod
    def execute(cls, dir: str) -> io.NodeOutput:
        return io.NodeOutput({"dir_control": dir, "dir_layers": dir, "manifest": {}, "job": None})


class SpikePick(io.ComfyNode):
    """clips list -> one clip (by shot index)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikePick", display_name="Spike: pick clip", category="Mograph/Spike",
                         inputs=[MographClips.Input("clips"), io.Int.Input("index", default=1, min=1, max=MAX_SHOTS)],
                         outputs=[MographClip.Output(display_name="clip")])

    @classmethod
    def execute(cls, clips, index: int) -> io.NodeOutput:
        return io.NodeOutput(next(c for c in clips if c["sid"] == f"s{index}"))


class SpikeInspect(io.ComfyNode):
    """Output node: writes a one-line description of every input to the spike log."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MographSpikeInspect", display_name="Spike: inspect", category="Mograph/Spike",
                         inputs=[io.String.Input("tag", default="x"), io.AnyType.Input("a", optional=True),
                                 io.AnyType.Input("b", optional=True), io.AnyType.Input("c", optional=True)],
                         outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, tag: str, a=None, b=None, c=None) -> io.NodeOutput:
        import torch

        def d(x):
            if isinstance(x, torch.Tensor):
                xf = x.float()
                return f"tensor{tuple(x.shape)} {x.dtype} min={xf.min():.3f} max={xf.max():.3f} mean={xf.mean():.3f}"
            if isinstance(x, list):
                return "[" + "; ".join(d(i) for i in x) + "]"
            if isinstance(x, dict):
                return "{" + ", ".join(f"{k}={d(v)}" for k, v in x.items()
                                       if k in ("sid", "u8", "start", "fps")) + "}"
            return repr(x)[:80]
        _log(f"inspect {tag}: a={d(a)} | b={d(b)} | c={d(c)}")
        return io.NodeOutput()


SPIKE_NODES = [SpikeCount, SpikeSlot, SpikeWork, SpikeGather, SpikePiece, SpikeRequest, SpikeMusic, SpikeType, SpikePick,
               SpikeInspect]
