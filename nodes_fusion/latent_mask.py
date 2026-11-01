"""MographVideoLatentMask: matte sequence -> spatial ``noise_mask`` on an LTX video latent (5D [B,C,T,H,W]).

Core ``SetLatentNoiseMask`` reshapes the mask to 4D (-1,1,H,W), which breaks 5D video latents, so this node writes a
[B,1,T_lat,H_lat,W_lat] mask itself. 1 = regenerate, 0 = keep the input latent. Put it BEFORE LTXAddVideoICLoRAGuide /
LTXVAddGuide: those expand a spatial noise_mask and append the guide frames' own mask. The maths is in
``tools/fusion/latent_mask.py`` (``make_footprint`` ...); this file only converts torch <-> numpy.

Works inside ComfyUI-Mograph (``nodes_fusion`` package) and as a stand-alone 2-file pack (``latent_mask.py`` +
``latent_mask_core.py`` = a copy of tools/fusion/latent_mask.py, see 16_app/mask_ltx/make_box_pack.py).
"""
from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

try:   # inside ComfyUI-Mograph
    from . import ensure_root_on_path
    ensure_root_on_path()
    from tools.fusion.latent_mask import (combine_noise_masks, frame_to_latent, make_footprint,  # noqa: F401
                                          noise_mask_from_footprint, pixel_frames, region_pixels)
except ImportError:   # stand-alone copy: latent_mask_core.py sits next to this file
    from .latent_mask_core import (combine_noise_masks, frame_to_latent, make_footprint,  # type: ignore # noqa: F401
                                   noise_mask_from_footprint, pixel_frames, region_pixels)

CATEGORY = "MOTNGRFX/Fusion"


class MographVideoLatentMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographVideoLatentMask",
            display_name="Mograph Video Latent Mask (matte -> LTX noise mask)",
            category=CATEGORY,
            description=("Sets a spatial denoise mask on an LTX video latent from a matte sequence: only the matte area "
                         "(plus dilate_cells) is re-noised/regenerated, everything else keeps the input latent. "
                         "Matte frames are max-pooled per latent frame (frame 0 alone, then groups of 8). "
                         "region_mask = the same footprint as a per-pixel-frame mask at the matte's resolution, for the "
                         "final pixel recomposite. Combined with an existing noise_mask by multiplication."),
            inputs=[
                io.Latent.Input("samples", tooltip="Video latent [B,C,T,H,W] (e.g. VAEEncode of the footage)."),
                io.Mask.Input("mask", optional=True, tooltip="Matte sequence [F,H,W], white = subject. Or use `image`."),
                io.Image.Input("image", optional=True,
                               tooltip="Matte as an IMAGE sequence [F,H,W,C] (red channel is used). Used when `mask` is empty."),
                io.Mask.Input("extra_mask", optional=True,
                              tooltip="Second matte sequence [F,H,W] unioned into the footprint (not blob-filtered), e.g. the area the motion tracks sweep."),
                io.Int.Input("dilate_cells", default=1, min=0, max=16,
                             tooltip="Grow the footprint by this many 32 px cells (3x3 square each), per latent frame."),
                io.Float.Input("feather", default=0.0, min=0.0, max=8.0, step=0.05,
                               tooltip="Soft edge in cells (gaussian sigma, edge-centred: inside stays 1). 0 = hard cells."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Mask value inside the footprint (1 = regenerate fully)."),
                io.Boolean.Input("keep_first_frame", default=False,
                                 tooltip="Latent frame 0 -> mask 0 (keeps the first latent frame of the input)."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Matte pixels above this count as subject."),
                io.Float.Input("cell_threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Fraction of a cell that must be subject (0 = any pixel)."),
                io.Boolean.Input("main_subject_only", default=True,
                                 tooltip="Per frame keep only the main blob(s) of the matte (drops foam / mist specks). Needs scipy or cv2."),
                io.Float.Input("min_blob_frac", default=0.25, min=0.0, max=1.0, step=0.01,
                               tooltip="With main_subject_only: blobs smaller than this fraction of the largest blob are dropped."),
                io.Float.Input("region_feather_px", default=12.0, min=0.0, max=64.0, step=0.5,
                               tooltip="Pixel feather of region_mask (gaussian sigma in matte pixels)."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.Mask.Output(display_name="region_mask")],
        )

    @classmethod
    def execute(cls, samples, mask=None, image=None, dilate_cells=1, feather=0.0, strength=1.0,
                keep_first_frame=False, threshold=0.5, cell_threshold=0.0, region_feather_px=12.0,
                extra_mask=None, main_subject_only=True, min_blob_frac=0.25) -> io.NodeOutput:
        x = samples["samples"]
        if x.ndim != 5:
            raise ValueError(f"MographVideoLatentMask: need a 5D video latent [B,C,T,H,W], got {tuple(x.shape)}")
        b, _, t_lat, h_lat, w_lat = x.shape
        if mask is not None:
            m = mask.detach().float().cpu().numpy()
            if m.ndim == 4:      # [B,1,H,W] / [F,1,H,W]
                m = m[:, 0]
        elif image is not None:
            m = image.detach().float().cpu().numpy()[..., 0]
        else:
            raise ValueError("MographVideoLatentMask: connect `mask` or `image` (the matte sequence)")
        if m.ndim == 2:
            m = m[None]
        extra = None
        if extra_mask is not None:
            extra = extra_mask.detach().float().cpu().numpy()
            extra = extra[:, 0] if extra.ndim == 4 else (extra[None] if extra.ndim == 2 else extra)
        fp = make_footprint(m, t_lat, h_lat, w_lat, dilate_cells=int(dilate_cells), feather=float(feather),
                            keep_first_frame=bool(keep_first_frame), threshold=float(threshold),
                            cell_threshold=float(cell_threshold), main_subject_only=bool(main_subject_only),
                            min_blob_frac=float(min_blob_frac), extra=extra)
        new = noise_mask_from_footprint(fp, float(strength))
        new = np.repeat(new, b, axis=0) if b > 1 else new
        existing = samples.get("noise_mask", None)
        ex = existing.detach().float().cpu().numpy() if existing is not None else None
        nm = combine_noise_masks(ex, new)
        out = dict(samples)
        out["noise_mask"] = torch.from_numpy(np.ascontiguousarray(nm)).to(dtype=torch.float32)
        region = region_pixels(fp, pixel_frames(t_lat), m.shape[1], m.shape[2], float(region_feather_px))
        return io.NodeOutput(out, torch.from_numpy(np.ascontiguousarray(region)))


NODES = [MographVideoLatentMask]
