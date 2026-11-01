"""MographSaveFinalUpscaled: the SECOND output of the app's Final (the Real-ESRGAN 1080x1920 video, tools/app/final_upscale.py), saved only when the piece is a Final.

The upscale chain (core nodes: GetVideoComponents -> UpscaleModelLoader -> ImageUpscaleWithModel -> ImageScale -> CreateVideo) sits in every app graph but hangs on this node's LAZY `video`
input, so a Preview (quality `still`), the internal 360x640 `preview` tier or a 16:9 piece never runs it (no model load, no GPU time). The 540x960 render keeps its own SaveVideo
(`OUT final`); this node saves the upscale next to it under its own name (progress_standin gives it the `_video2` suffix)."""
from __future__ import annotations

import os

from comfy_api.latest import io, ui

from . import ensure_root_on_path
from ._types import CATEGORY, MographPiece

ensure_root_on_path()


from tools.app.final_upscale import wants_upscale  # noqa: E402


class MographSaveFinalUpscaled(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographSaveFinalUpscaled",
            display_name="MOTNGRFX Save Final (1080x1920 upscale)",
            category=CATEGORY,
            description="Saves the Real-ESRGAN upscaled Final next to the 540x960 render. The video input is lazy: a Preview / low-res / 16:9 piece never runs the upscale.",
            inputs=[MographPiece.Input("piece"), io.Video.Input("video", lazy=True), io.String.Input("filename_prefix", default="video/motngrfx_1080")],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, piece=None, video=None, filename_prefix=None) -> list[str]:
        return ["video"] if video is None and wants_upscale(piece) else []

    @classmethod
    def execute(cls, piece, video=None, filename_prefix="video/motngrfx_1080") -> io.NodeOutput:
        if video is None or not wants_upscale(piece):
            return io.NodeOutput()
        import folder_paths
        from comfy_api.latest import Types
        w, h = video.get_dimensions()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), w, h)
        fn = f"{name}_{counter:05}_.mp4"
        video.save_to(os.path.join(folder, fn), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(fn, subfolder, io.FolderType.output)]))


NODES = [MographSaveFinalUpscaled]
