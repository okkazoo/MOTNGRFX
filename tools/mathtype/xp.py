"""Array backend for the mathtype renderers: numpy (default, the reference) or torch (cpu / cuda).

Ops dispatch on the type of their first array argument, so one closed-form function (breath, transform,
the noise banks, the curl flow...) runs unchanged on numpy float64 arrays or torch float32 tensors. On
numpy each op calls exactly the numpy function the renderers used before, so numpy output is
bit-identical to the pre-port code. Torch is float32 throughout.

    be = xp.Backend("torch", "cuda")    # or xp.Backend() -> numpy
    x = be.asarray(np_array)            # float32 tensor on the device (numpy: float64 array)
    xp.sin(x), xp.clip(x, 0, 1), xp.norm(x, axis=1, keepdims=True), xp.interp(t, grid, table), ...

CUDA determinism: Backend("torch", "cuda") turns on torch.use_deterministic_algorithms (warn only) and
turns TF32 off for matmul and cuDNN, because the noise banks put |arg| ~ 20 rad through a matmul, where
TF32's 10-bit mantissa would cost ~0.01 rad.
"""
from __future__ import annotations

import os

import numpy as np

_torch = None


def torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def is_t(x) -> bool:
    """True for a torch tensor (without importing torch when it was never loaded)."""
    return _torch is not None and isinstance(x, _torch.Tensor)


class Backend:
    def __init__(self, name: str = "numpy", device: str = "cpu"):
        if name not in ("numpy", "torch"):
            raise ValueError(f"backend {name!r}: numpy | torch")
        self.name, self.device = name, device
        self.torch = name == "torch"
        if self.torch:
            if device.startswith("cuda"):
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            T = torch()
            if device.startswith("cuda"):
                if not T.cuda.is_available():
                    raise SystemExit("--device cuda: torch has no CUDA here (torch " + T.__version__ + ")")
                T.backends.cuda.matmul.allow_tf32 = False
                T.backends.cudnn.allow_tf32 = False
                T.backends.cudnn.benchmark = False
                T.use_deterministic_algorithms(True, warn_only=True)
            T.set_float32_matmul_precision("highest")
            self.dev = T.device(device)
            self.f32 = T.float32
        elif device != "cpu":
            raise SystemExit("--device cuda needs --backend torch")

    def __repr__(self):
        return f"Backend({self.name}, {self.device})" if self.torch else "Backend(numpy)"

    def asarray(self, x, dtype=None):
        """numpy: np.asarray (float64 unless dtype); torch: tensor on the device (float32 for floats)."""
        if not self.torch:
            return np.asarray(x, float if dtype is None else dtype)
        T = torch()
        if is_t(x):
            return x.to(self.dev)
        a = np.asarray(x)
        if dtype is None:
            dtype = T.bool if a.dtype == bool else (T.int64 if np.issubdtype(a.dtype, np.integer) else self.f32)
        return T.as_tensor(a, device=self.dev).to(dtype)

    def sync(self):
        if self.torch and self.dev.type == "cuda":
            torch().cuda.synchronize()


def numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if is_t(x) else np.asarray(x)


def const(like, v):
    """A constant (list / array) in the backend, dtype and device of `like` (numpy: float64 array)."""
    if is_t(like):
        return torch().as_tensor(np.asarray(v, float), dtype=like.dtype, device=like.device)
    return np.asarray(v, float)


def asfloat(x):
    """np.asarray(x, float) for numpy / scalars; tensors pass through."""
    return x if is_t(x) else np.asarray(x, float)


# ------------------------------------------------------------------------------------------- elementwise
def sin(x):
    return torch().sin(x) if is_t(x) else np.sin(x)


def cos(x):
    return torch().cos(x) if is_t(x) else np.cos(x)


def exp(x):
    return torch().exp(x) if is_t(x) else np.exp(x)


def sqrt(x):
    return torch().sqrt(x) if is_t(x) else np.sqrt(x)


def floor(x):
    return torch().floor(x) if is_t(x) else np.floor(x)


def ceil(x):
    return torch().ceil(x) if is_t(x) else np.ceil(x)


def log2(x):
    return torch().log2(x) if is_t(x) else np.log2(x)


def round_(x):
    return torch().round(x) if is_t(x) else np.round(x)


def clip(x, lo, hi):
    return torch().clamp(x, lo, hi) if is_t(x) else np.clip(x, lo, hi)


def maximum(a, b):
    if is_t(a) or is_t(b):
        T = torch()
        if not is_t(a):
            return T.clamp(b, min=a) if np.isscalar(a) else T.maximum(const(b, a), b)
        if not is_t(b):
            return T.clamp(a, min=b) if np.isscalar(b) else T.maximum(a, const(a, b))
        return T.maximum(a, b)
    return np.maximum(a, b)


def minimum(a, b):
    if is_t(a) or is_t(b):
        T = torch()
        if not is_t(a):
            return T.clamp(b, max=a) if np.isscalar(a) else T.minimum(const(b, a), b)
        if not is_t(b):
            return T.clamp(a, max=b) if np.isscalar(b) else T.minimum(a, const(a, b))
        return T.minimum(a, b)
    return np.minimum(a, b)


def where(c, a, b):
    if is_t(c):
        T = torch()
        ref = a if is_t(a) else (b if is_t(b) else None)
        a = a if is_t(a) else (T.as_tensor(a, dtype=ref.dtype if ref is not None else T.float32, device=c.device))
        b = b if is_t(b) else T.as_tensor(b, dtype=a.dtype, device=c.device)
        return T.where(c, a, b)
    return np.where(c, a, b)


def ones_like(x):
    return torch().ones_like(x) if is_t(x) else np.ones_like(x)


def zeros_like(x):
    return torch().zeros_like(x) if is_t(x) else np.zeros_like(x)


# ------------------------------------------------------------------------------------------- shape / reduce
def stack(xs, axis=0):
    return torch().stack(list(xs), axis) if is_t(xs[0]) else np.stack(xs, axis)


def concat(xs, axis=0):
    return torch().cat(list(xs), axis) if is_t(xs[0]) else np.concatenate(xs, axis)


def column_stack(xs):
    return torch().column_stack(list(xs)) if is_t(xs[0]) else np.column_stack(xs)


def norm(x, axis=None, keepdims=False):
    return torch().linalg.vector_norm(x, dim=axis, keepdim=keepdims) if is_t(x) else np.linalg.norm(x, axis=axis, keepdims=keepdims)


def sum_(x, axis=None, keepdims=False):
    return torch().sum(x, axis, keepdim=keepdims) if is_t(x) else np.sum(x, axis, keepdims=keepdims)


def einsum(eq, *ops):
    return torch().einsum(eq, *ops) if is_t(ops[0]) else np.einsum(eq, *ops)


def cross(a, b):
    """Cross product on the last axis; a or b may be a constant 3-vector."""
    if is_t(a) or is_t(b):
        ref = a if is_t(a) else b
        a, b = (a if is_t(a) else const(ref, a)), (b if is_t(b) else const(ref, b))
        a, b = torch().broadcast_tensors(a, b)
        return torch().linalg.cross(a, b, dim=-1)
    return np.cross(a, b)


def interp(x, grid, table, grid_t=None, table_t=None):
    """np.interp(x, grid, table) for any-shape x (clamped at both ends). Torch: pass the grid / table as
    device tensors (grid_t, table_t) to avoid a copy per call; searchsorted + lerp."""
    if not is_t(x):
        x = np.asarray(x, float)
        return np.interp(x.ravel(), grid, table).reshape(x.shape)
    T = torch()
    g = grid_t if grid_t is not None else const(x, grid)
    f = table_t if table_t is not None else const(x, table)
    xc = x.clamp(g[0], g[-1])
    i = T.searchsorted(g, xc.reshape(-1).contiguous(), right=True).clamp(1, g.numel() - 1)
    g0, g1, f0, f1 = g[i - 1], g[i], f[i - 1], f[i]
    w = ((xc.reshape(-1) - g0) / (g1 - g0)).clamp(0.0, 1.0)
    return (f0 + (f1 - f0) * w).reshape(x.shape)


# ------------------------------------------------------------------------------------------- image ops
def gauss_kernel(sigma: float) -> np.ndarray:
    """cv2.GaussianBlur's kernel for a float image with ksize (0, 0): size round(8 sigma + 1) | 1,
    computed in double and normalised (cv2.getGaussianKernel)."""
    ks = int(np.floor(sigma * 4 * 2 + 1 + 0.5)) | 1
    return cv2_kernel(ks, sigma)


def cv2_kernel(ks: int, sigma: float) -> np.ndarray:
    x = np.arange(ks) - (ks - 1) * 0.5
    k = np.exp(-x * x / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _reflect101(n: int, r: int, device):
    """Source indices of an n-long axis padded by r on each side, cv2 BORDER_REFLECT_101 (any r)."""
    T = torch()
    i = T.arange(-r, n + r, device=device)
    if n == 1:
        return T.zeros_like(i)
    p = 2 * (n - 1)
    i = T.remainder(i, p)
    return T.where(i >= n, p - i, i)


def gaussian_blur(x, sigma: float):
    """cv2.GaussianBlur(x, (0, 0), sigma) on float32 images (..., H, W): same kernel and border."""
    T = torch()
    k = gauss_kernel(sigma)
    r = (k.size - 1) // 2
    sh = x.shape
    H, W = sh[-2], sh[-1]
    y = x.reshape(-1, 1, H, W)
    w = T.as_tensor(k, device=x.device)
    pad = T.nn.functional.pad                     # torch "reflect" == BORDER_REFLECT_101, valid for r < n
    y = pad(y, (r, r, 0, 0), mode="reflect") if r < W else y.index_select(3, _reflect101(W, r, x.device))
    y = T.nn.functional.conv2d(y, w.view(1, 1, 1, -1))
    y = pad(y, (0, 0, r, r), mode="reflect") if r < H else y.index_select(2, _reflect101(H, r, x.device))
    y = T.nn.functional.conv2d(y, w.view(1, 1, -1, 1))
    return y.reshape(sh)
