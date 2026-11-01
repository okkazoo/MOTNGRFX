"""Optional file inputs (``IN image`` / ``IN audio`` of the fusion graph) and the first-frame node.

* MographOptionalImage / MographOptionalAudio: a filename in ComfyUI/input ("" = nothing) -> IMAGE / AUDIO or None.
  The page uploads through ``/upload/image`` and fills the string; an empty string keeps the node in the graph
  (links to it stay valid) and yields ``None``, so the optional inputs downstream are simply unset.
* MographUserImages: the user's images 2..5 (comma-separated names, ``IN images``) -> ONE IMAGE batch at the shot size; MographFirstFrames puts image k in place of the generated
  still of shot k (tools/fusion/places.user_image_rows).
* MographFirstFrame: user image -> cover-fit to the shot size (1152x2048, or 2048x1152 for 16:9); else the generated still (lazy: the still model chain
  never runs when a user image was given).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
import folder_paths
from comfy_api.latest import io

from . import ensure_root_on_path
from ._types import CATEGORY, MAX_SHOTS, MographPiece, MographRequest
from ._util import cover_resize, first_image

ensure_root_on_path()
from tools.fusion import places as PL      # noqa: E402

FIRST_SIZE = (1152, 2048)    # (w, h) first frame handed to the control graph (final output size of the shots)


def _input_path(name: str) -> str:
    path = folder_paths.get_annotated_filepath(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name!r} not found in the ComfyUI input folder")
    return path


def _file_hash(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    try:
        h = hashlib.sha256()
        with open(folder_paths.get_annotated_filepath(name), "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except OSError:
        return "missing"


class MographOptionalImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographOptionalImage",
            display_name="MOTNGRFX Optional Image",
            category=CATEGORY,
            description="Image file in the ComfyUI input folder -> IMAGE; empty name = no image (None).",
            inputs=[io.String.Input("image", default="", tooltip="File name in ComfyUI/input; empty = none.")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def fingerprint_inputs(cls, image: str):
        return _file_hash(image)

    @classmethod
    def execute(cls, image: str) -> io.NodeOutput:
        name = (image or "").strip()
        if not name:
            return io.NodeOutput(None)
        from PIL import Image, ImageOps
        with Image.open(_input_path(name)) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return io.NodeOutput(torch.from_numpy(np.ascontiguousarray(rgb))[None])


class MographOptionalAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographOptionalAudio",
            display_name="MOTNGRFX Optional Audio",
            category=CATEGORY,
            description="Audio file in the ComfyUI input folder -> AUDIO; empty name = no audio (None).",
            inputs=[io.String.Input("audio", default="", tooltip="File name in ComfyUI/input; empty = none.")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def fingerprint_inputs(cls, audio: str):
        return _file_hash(audio)

    @classmethod
    def execute(cls, audio: str) -> io.NodeOutput:
        name = (audio or "").strip()
        if not name:
            return io.NodeOutput(None)
        from comfy_extras.nodes_audio import load
        wave, sr = load(_input_path(name))
        return io.NodeOutput({"waveform": wave.unsqueeze(0), "sample_rate": int(sr)})


class MographUserImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographUserImages",
            display_name="MOTNGRFX User Images (2..5)",
            category=CATEGORY,
            description="Comma-separated image names in the ComfyUI input folder (the user's images 2..5) -> one IMAGE batch, cover-fitted to the shot size of the format; "
                        "empty = none (None). MographFirstFrames uses image k as the first frame of shot k instead of generating that still.",
            inputs=[io.String.Input("names", default="", tooltip="File names in ComfyUI/input, comma separated; empty = none."),
                    io.String.Input("format", default="9:16", optional=True, tooltip="9:16 | 16:9")],
            outputs=[io.Image.Output(display_name="images")],
        )

    @staticmethod
    def split(names: str) -> list[str]:
        out = [n.strip() for n in str(names or "").split(",") if n.strip()]
        return out[:PL.MAX_USER_IMAGES]

    @classmethod
    def fingerprint_inputs(cls, names: str, format: str = "9:16"):
        return "|".join(f"{n}:{_file_hash(n)}" for n in cls.split(names)) + f"|{format}"

    @classmethod
    def execute(cls, names: str, format: str = "9:16") -> io.NodeOutput:
        files = cls.split(names)
        if not files:
            return io.NodeOutput(None)
        from PIL import Image, ImageOps
        w, h = (FIRST_SIZE[1], FIRST_SIZE[0]) if format == "16:9" else FIRST_SIZE
        rows = []
        for n in files:
            with Image.open(_input_path(n)) as im:
                im = ImageOps.exif_transpose(im)
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(np.ascontiguousarray(cover_resize(rgb, w, h), dtype=np.float32))
        return io.NodeOutput(torch.from_numpy(np.stack(rows)))


def shot_wh(piece: dict, request: dict) -> tuple[int, int]:
    """(w, h) of the first frame handed to the shots: the piece's shot size, else FIRST_SIZE turned into the request's format."""
    sz = (piece.get("sizes") or {}).get("shot")
    return (int(sz[0]), int(sz[1])) if sz else ((FIRST_SIZE[1], FIRST_SIZE[0]) if request.get("format") == "16:9" else FIRST_SIZE)


class MographFirstFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographFirstFrame",
            display_name="MOTNGRFX First Frame",
            category=CATEGORY,
            description="First frame of shot 1 at the shot size (1152x2048 or 2048x1152): the user's image if there is one, else the generated "
                        "still (lazy).",
            inputs=[MographPiece.Input("piece"), MographRequest.Input("request"),
                    io.Image.Input("generated", optional=True, lazy=True,
                                   tooltip="Output of the still model; only evaluated when no user image is given.")],
            outputs=[io.Image.Output(display_name="image")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, piece=None, request=None, generated=None) -> list[str]:
        if generated is not None or (request or {}).get("image") is not None:
            return []
        wired = (cls.hidden.prompt or {}).get(str(cls.hidden.unique_id), {}).get("inputs", {})
        return ["generated"] if isinstance(wired.get("generated"), list) else []

    @classmethod
    def execute(cls, piece: dict, request: dict, generated=None) -> io.NodeOutput:
        w, h = shot_wh(piece, request)
        src = first_image(request.get("image"))
        if src is None:
            src = first_image(generated)
        if src is None:
            raise ValueError("MographFirstFrame: no still arrived (no user image and the still model gave nothing; there is no placeholder gradient any more)")
        out = cover_resize(src, w, h)
        return io.NodeOutput(torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))[None])


class MographStillPrompt(io.ComfyNode):
    """The prompt and seed of the still of place `index` (2..6) from the plan's `stills` list (still 1 keeps the planner node's outputs)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographStillPrompt",
            display_name="MOTNGRFX Still Prompt (place k)",
            category=CATEGORY,
            description="Prompt + seed of the k-th still of the plan (one still per place). Only runs when the still is needed (its sampler is behind a lazy input of "
                        "MographFirstFrames); a piece without that still gets an empty prompt.",
            inputs=[io.Int.Input("index", default=2, min=1, max=MAX_SHOTS, tooltip="Still number 1..6 (A..F)."), MographPiece.Input("piece"),
                    io.Boolean.Input("angle", default=False, optional=True,
                                     tooltip="true = the ANGLE-STILL chain (MOSS edit of another place's still, tools/fusion/places.angle_prompt): the anchored edit prompt when this place is an "
                                             "angle place (`angle_of`), else the place's own prompt.")],
            outputs=[io.String.Output(display_name="prompt"), io.Int.Output(display_name="seed")],
        )

    @classmethod
    def execute(cls, index: int, piece: dict, angle: bool = False) -> io.NodeOutput:
        stills = (piece or {}).get("stills") or []
        st = stills[int(index) - 1] if 0 < int(index) <= len(stills) else {}
        prompt = (PL.angle_prompt(piece or {}, int(index) - 1) if angle else "") or str(st.get("prompt", ""))
        return io.NodeOutput(prompt, int(st.get("seed", 0)))


def _angle_of(piece: dict, i: int, own: dict) -> int | None:
    """the source place of angle place i, None when it is no angle place OR its source is a user image (the edit would start from a picture the user replaced: the place gets a fresh still of its own)."""
    j = PL.angle_source(piece, i)
    return j if j is not None and j not in own else None


def _n_rows(t) -> int:
    return int(t.shape[0]) if t is not None and hasattr(t, "shape") and len(t.shape) == 4 else 0


class MographFirstFrames(io.ComfyNode):
    """The first frames of ALL places as one IMAGE batch (row i = piece.stills[i]); MographShotSlot picks its row by the shot's `first`."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographFirstFrames",
            display_name="MOTNGRFX First Frames (one per place)",
            category=CATEGORY,
            description="Batch of the places' first frames: row 0 = MographFirstFrame (the user's image or still A), rows 1.. = the stills B, C ... generated by "
                        "their own chains. A still is only evaluated (lazy) when a shot of the piece starts from it, so a one-place piece generates nothing extra "
                        "and gets a batch of one row.",
            inputs=[MographPiece.Input("piece"), MographRequest.Input("request"),
                    io.Image.Input("first", tooltip="First frame of the piece (MographFirstFrame): row 0."),
                    io.Image.Input("user_images", optional=True, tooltip="The user's images 2..5 (MographUserImages): image k is the first frame of shot k, in place of that shot's generated still."),
                    *[io.Image.Input(f"still_{k}", optional=True, lazy=True, tooltip=f"Still {chr(64 + k)} (place {k}); only evaluated when a shot starts from it.")
                      for k in range(2, PL.MAX_STILLS + 1)],
                    io.Boolean.Input("angles", default=False, optional=True,
                                     tooltip="true = a place with `angle_of` is the MOSS EDIT of its source place's still (`angle_k`, 24 GB tier, needs moss_image_edit_chinoll.safetensors) "
                                             "instead of a fresh text-to-image still (`still_k`). false = the angle chains never run."),
                    *[io.Image.Input(f"angle_{k}", optional=True, lazy=True,
                                     tooltip=f"Angle still of place {k} (an edit of its source place's still); evaluated only when `angles` is on, a shot starts from it and the place has `angle_of`.")
                      for k in range(2, PL.MAX_STILLS + 1)]],
            outputs=[io.Image.Output(display_name="firsts")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @staticmethod
    def wanted(piece: dict, angles: bool, n_user: int = 0) -> dict[int, str]:
        """{place index: name of the input that supplies its image}: `angle_k` for an angle place while `angles` is on, else `still_k` (places 1.. = B..; place 0 is `first`).
        A place the user's image replaces (places.user_image_rows) is not asked for: its chain never runs."""
        own = PL.user_image_rows(piece, n_user)
        return {i: (f"angle_{i + 1}" if angles and _angle_of(piece, i, own) is not None else f"still_{i + 1}") for i in PL.needed_stills(piece) if i >= 1 and i not in own}

    @classmethod
    def check_lazy_status(cls, piece=None, request=None, first=None, user_images=None, angles=False, **stills) -> list[str]:
        if piece is None:
            return []
        wired = (cls.hidden.prompt or {}).get(str(cls.hidden.unique_id), {}).get("inputs", {})
        need = set()
        for i, name in cls.wanted(piece, bool(angles), _n_rows(user_images)).items():
            if name.startswith("angle_") and not isinstance(wired.get(name), list):          # a graph without the angle chain (older file): the fresh still
                name = f"still_{i + 1}"
            need.add(name)
        return [name for name in sorted(need) if stills.get(name) is None and isinstance(wired.get(name), list)]

    @classmethod
    def execute(cls, piece: dict, request: dict, first, user_images=None, angles=False, **stills) -> io.NodeOutput:
        if PL.batch_size(piece) <= 1:
            return io.NodeOutput(first)                        # one place: exactly what MographFirstFrame made (same tensor)
        w, h = shot_wh(piece, request)
        own = PL.user_image_rows(piece, _n_rows(user_images))

        def image(i: int):                                     # the user's image for that shot; else the angle still when the place is an angle place and its chain ran, else the fresh still
            if i in own:
                return first_image(user_images[own[i]:own[i] + 1])
            if angles and _angle_of(piece, i, own) is not None and stills.get(f"angle_{i + 1}") is not None:
                return first_image(stills.get(f"angle_{i + 1}"))
            return first_image(stills.get(f"still_{i + 1}"))
        rows = PL.batch_rows(piece, first_image(first), {i: image(i) for i in range(1, PL.MAX_STILLS)})
        out = np.stack([np.ascontiguousarray(cover_resize(r, w, h), dtype=np.float32) for r in rows])
        return io.NodeOutput(torch.from_numpy(out))


class MographAngleRef(io.ComfyNode):
    """The REFERENCE of angle place `index` (2..6): the ORIGINAL still of the place its `angle_of` names (never an edit of an edit), at the still size, for the
    `TextEncodeMOSSImageEdit` pair of the angle chain. Place A's image is `first` (the user's own image or still A); another source place's image is its lazy `still_j`
    input (only that one is evaluated). A place that is not an angle place gets `first` (the chain behind it never runs)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographAngleRef",
            display_name="MOTNGRFX Angle Reference (original still of the source place)",
            category=CATEGORY,
            description="Reference image of an angle still: the original still of the source place (angle_of), cover-fitted to the still size. Place A = the first frame; "
                        "any other source place = its own still (lazy). Used as image_0 of both TextEncodeMOSSImageEdit nodes of the angle chain.",
            inputs=[io.Int.Input("index", default=2, min=1, max=MAX_SHOTS, tooltip="Place number 1..6 (A..F) of the ANGLE place."), MographPiece.Input("piece"),
                    io.Image.Input("first", tooltip="First frame of the piece (MographFirstFrame): the image of place A."),
                    io.Int.Input("width", default=1152, min=64, max=4096, tooltip="still width (the planner's still size)"),
                    io.Int.Input("height", default=2048, min=64, max=4096, tooltip="still height"),
                    *[io.Image.Input(f"still_{k}", optional=True, lazy=True, tooltip=f"Still of place {k} (fresh text-to-image); evaluated only when it is the source of this angle place.")
                      for k in range(2, PL.MAX_STILLS + 1)]],
            outputs=[io.Image.Output(display_name="reference")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def source_name(cls, piece: dict, index: int) -> str | None:
        """the lazy input that holds the source image (None = `first`)."""
        j = PL.angle_source(piece or {}, int(index) - 1)
        return f"still_{j + 1}" if j is not None and j >= 1 else None

    @classmethod
    def check_lazy_status(cls, index=2, piece=None, first=None, width=0, height=0, **stills) -> list[str]:
        name = cls.source_name(piece, index) if piece is not None else None
        wired = (cls.hidden.prompt or {}).get(str(cls.hidden.unique_id), {}).get("inputs", {})
        return [name] if name and stills.get(name) is None and isinstance(wired.get(name), list) else []

    @classmethod
    def execute(cls, index: int, piece: dict, first, width: int, height: int, **stills) -> io.NodeOutput:
        name = cls.source_name(piece, index)
        src = first_image(stills.get(name)) if name else None
        if src is None:
            src = first_image(first)
        out = cover_resize(src, int(width), int(height))
        return io.NodeOutput(torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))[None])


NODES = [MographOptionalImage, MographOptionalAudio, MographUserImages, MographFirstFrame, MographStillPrompt, MographFirstFrames, MographAngleRef]
