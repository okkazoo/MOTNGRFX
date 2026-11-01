"""Small numpy/cv2 helpers shared by the inputs / footage nodes (no ComfyUI imports at module level except torch)."""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import torch


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale HxWx3 (uint8 or float) to cover (w, h) and centre-crop. Returns the same dtype."""
    ih, iw = img.shape[:2]
    if (ih, iw) == (h, w):
        return img
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    out = out[y0:y0 + h, x0:x0 + w]
    if out.dtype != np.uint8:
        out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out)


def first_image(t) -> np.ndarray | None:
    """IMAGE tensor [B,H,W,C] (or None) -> first image as HxWx3 float32 0..1."""
    if t is None:
        return None
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.ascontiguousarray(a[..., :3], dtype=np.float32)


def rng_for(seed: int, tag: str) -> np.random.Generator:
    """Deterministic sub-seed like ``brief._salt``: crc32 of ``seed:tag``."""
    return np.random.default_rng(zlib.crc32(f"{int(seed)}:{tag}".encode()))
