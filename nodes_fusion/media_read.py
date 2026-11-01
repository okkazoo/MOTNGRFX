"""MographMediaPack / MographMediaReadParse: the READ step of the "Add media" box (16_app/MEDIA_BOX.md). Logic: tools/fusion/media_read.py (pure); graph: tools/app/media_read_graph.py.

    media (the page's MEDIA JSON)  -> MographMediaPack      -> sheet IMAGE (one contact sheet), prompt STRING, info STRING json
    sheet + prompt                 -> MographGGUFVL (Gemma 4 31B QAT, json_mode)  -> reply
    reply + info + media           -> MographMediaReadParse -> read STRING (READ JSON v1)

Pack: files are looked up in ComfyUI's input dir; ffmpeg frames (~1 per second, <= 16) labelled V1 + second, images I1..I5, ONE sheet <= 1024 px; the DSP part of the music
reading (bpm / key / sections, CPU) of the video's own audio or of the audio file. No model call in either node.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from comfy_api.latest import io

try:
    import folder_paths
except Exception:   # pragma: no cover
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY

ensure_root_on_path()
from tools.fusion import media_read as MR   # noqa: E402


def _input_dir() -> str:
    return folder_paths.get_input_directory() if folder_paths is not None else "."


class MographMediaPack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMediaPack",
            display_name="MOTNGRFX Media Pack (contact sheet + prompt)",
            category=CATEGORY,
            description="MEDIA JSON ({video, images[<=5], audio, text, length, format}; file names in ComfyUI's input folder) -> ONE labelled contact sheet (video frames V1 + second, "
                        "images I1..I5), the prompt for the READ model and an info json (durations, pixel palettes, DSP music numbers). CPU: ffmpeg + PIL + the beat analysis.",
            inputs=[io.String.Input("media", multiline=True, default="{}", tooltip='{"video": "motngrfx_clip.mp4"|null, "images": [...], "audio": ...|null, "text": "...", "length": 12, "format": "9:16"}')],
            outputs=[io.Image.Output(display_name="sheet"), io.String.Output(display_name="prompt"), io.String.Output(display_name="info")],
        )

    @classmethod
    def fingerprint_inputs(cls, media="{}", **kw):
        return media

    @classmethod
    def execute(cls, media: str = "{}") -> io.NodeOutput:
        sheet, prompt, info = MR.pack(media, _input_dir())
        print(f"[MographMediaPack] {len(info['items'])} item(s), sheet {sheet.shape[1]}x{sheet.shape[0]}, music {info['music']}, {info['pack_s']} s")
        t = torch.from_numpy(np.ascontiguousarray(sheet).astype(np.float32) / 255.0)[None]
        return io.NodeOutput(t, prompt, json.dumps(info))


class MographMediaReadParse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographMediaReadParse",
            display_name="MOTNGRFX Media Read Parse (READ JSON v1)",
            category=CATEGORY,
            description="The READ model's reply + the Pack info + the MEDIA JSON -> READ JSON v1 with the code guarantees: valid JSON with every key, quoted text verbatim in on_screen, "
                        "no invented on_screen lines, video window clamped, palette from pixels when missing, music from the DSP numbers, a rule split when the reply is unusable.",
            inputs=[io.String.Input("reply", multiline=True, default=""), io.String.Input("info", multiline=True, default="{}"), io.String.Input("media", multiline=True, default="{}")],
            outputs=[io.String.Output(display_name="read")],
        )

    @classmethod
    def execute(cls, reply: str = "", info: str = "{}", media: str = "{}") -> io.NodeOutput:
        try:
            inf = json.loads(info) if info else {}
        except ValueError:
            inf = {}
        read = MR.parse(reply, inf, media)
        return io.NodeOutput(json.dumps(read, indent=1, ensure_ascii=False))


NODES = [MographMediaPack, MographMediaReadParse]
