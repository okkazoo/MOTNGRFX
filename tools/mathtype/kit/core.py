"""Backend selection and device labels (thin layer over tools/mathtype/xp.py)."""
from __future__ import annotations

import sys

import xp


def torch_or_none():
    try:
        return xp.torch()
    except Exception:  # pragma: no cover - torch missing
        return None


def backend(device: str = "auto", threads: int = 0) -> xp.Backend:
    """xp.Backend for the kit. device: "auto" (cuda if torch has it, else torch cpu), "cuda", "cpu" (torch),
    or "numpy" (the numpy reference path; draw primitives that need scatter fall back to torch cpu)."""
    if device == "numpy":
        return xp.Backend("numpy", "cpu")
    T = torch_or_none()
    if T is None:
        return xp.Backend("numpy", "cpu")
    if device == "auto":
        device = "cuda" if T.cuda.is_available() else "cpu"
    if threads and device == "cpu":
        T.set_num_threads(threads)
    return xp.Backend("torch", device)


def cuda(be) -> bool:
    return bool(be.torch and be.dev.type == "cuda")


def device_label(be) -> str:
    """TIMINGS.csv device: cuda-RTX3060 | box-cpu | local-cpu-torch | local-numpy."""
    box = sys.platform != "win32"          # local dev is Windows; the GPU box is Linux
    if not be.torch:
        return "box-numpy" if box else "local-numpy"
    if be.dev.type == "cuda":
        name = xp.torch().cuda.get_device_name(be.dev)
        return "cuda-" + name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
    return "box-cpu" if box else "local-cpu-torch"


def grid(be, W: int, H: int, scale: float = 1.0):
    """Pixel-centre coordinates (px, py), each (H, W), in units of `scale` per pixel (design px if
    scale = DW / W). float32 tensors on the device (numpy: float32 arrays)."""
    if be.torch:
        T = xp.torch()
        ys, xs = T.meshgrid(T.arange(H, device=be.dev, dtype=T.float32),
                            T.arange(W, device=be.dev, dtype=T.float32), indexing="ij")
    else:
        import numpy as np
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return (xs + 0.5) * scale, (ys + 0.5) * scale
