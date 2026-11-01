"""Matte sequence -> LTX video-latent denoise mask (pure numpy; used by nodes_fusion/latent_mask.py).

LTX latent grid: 32 px spatial, 8 frames temporal, causal: latent frame 0 = pixel frame 0, latent frame k >= 1 =
pixel frames 8(k-1)+1 .. 8k.  So F pixel frames (F = 8n+1) give T_lat = (F-1)/8 + 1 latent frames (121 -> 16) and a
W x H pixel size gives (H/32, W/32) cells (576x1024 -> 32 rows x 18 columns).

Mask semantics are ComfyUI's ``noise_mask`` for LTXAV: 1 = regenerate (full denoise), 0 = keep the input latent
(token timestep 0), values in between = partial.  Pipeline (``make_footprint``):

    matte (F,H,W) -> binarise (> threshold), keep the main blob per frame -> area-average into (h,w) cells
    [-> union with the same pooling of an optional `extra` matte, e.g. the corridor the tracks sweep] -> max over each 8-frame group
    -> cell is on if coverage > cell_threshold -> dilate (3x3 square, ``dilate_cells`` times, per latent frame)
    -> feather (gaussian of sigma ``feather`` cells around the edge; inside stays 1, a soft ring falls off outward)
    -> keep_first_frame (latent frame 0 -> 0)                               = footprint, floats in 0..1
    noise mask = footprint * strength;  combined with an existing mask by MULTIPLICATION (0 = keep wins).

``region_pixels`` turns the footprint back into a per-pixel-frame mask (nearest cell upsample + pixel feather) for the
final recomposite, so the pixels that are blended back from the original are exactly the ones that were regenerated.
"""
from __future__ import annotations

import numpy as np

SPATIAL = 32     # pixels per latent cell
TEMPORAL = 8     # pixel frames per latent frame (after the first)


# --------------------------------------------------------------------------------------------- timing
def latent_frames(n_pixel_frames: int) -> int:
    """121 -> 16.  (F-1)//8 + 1 (F should be 8k+1; others round down)."""
    return (int(n_pixel_frames) - 1) // TEMPORAL + 1


def pixel_frames(t_lat: int) -> int:
    """16 -> 121."""
    return (int(t_lat) - 1) * TEMPORAL + 1


def frame_to_latent(f):
    """Pixel frame index (int or int array) -> latent frame index: 0->0, 1..8->1, 9..16->2 ..."""
    f = np.asarray(f)
    return (f - 1) // TEMPORAL + 1


def group_slice(k: int) -> tuple[int, int]:
    """Latent frame k -> [a, b) range of pixel frames it covers."""
    return (0, 1) if k == 0 else ((k - 1) * TEMPORAL + 1, k * TEMPORAL + 1)


def fit_frames(m: np.ndarray, n: int) -> np.ndarray:
    """Truncate to n frames or pad by repeating the last frame (axis 0)."""
    if m.shape[0] >= n:
        return m[:n]
    return np.concatenate([m, np.repeat(m[-1:], n - m.shape[0], axis=0)], 0)


# --------------------------------------------------------------------------------------------- spatial / temporal pooling
def _edges(n_pix: int, n_cells: int) -> np.ndarray:
    e = np.round(np.linspace(0, n_pix, n_cells + 1)).astype(int)
    if np.any(np.diff(e) <= 0):
        raise ValueError(f"cannot split {n_pix} px into {n_cells} cells")
    return e


def keep_main_blob(b: np.ndarray, min_frac: float = 0.25) -> np.ndarray:
    """(H,W) bool -> bool with only the 8-connected blobs whose area >= min_frac x the largest blob (drops foam / mist specks
    the matte model leaves around). Needs scipy or cv2; without either the mask is returned unchanged (BLOB_BACKEND == None)."""
    if not b.any():
        return b
    if BLOB_BACKEND == "scipy":
        from scipy import ndimage
        lab, n = ndimage.label(b, structure=np.ones((3, 3), bool))
        if n <= 1:
            return b
        sizes = np.bincount(lab.ravel())[1:]
    elif BLOB_BACKEND == "cv2":
        import cv2
        n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
        if n <= 2:
            return b
        sizes = st[1:, cv2.CC_STAT_AREA]
    else:
        return b
    keep = np.nonzero(sizes >= min_frac * sizes.max())[0] + 1
    return np.isin(lab, keep)


def _backend():
    try:
        import scipy.ndimage  # noqa: F401
        return "scipy"
    except Exception:   # noqa: BLE001
        pass
    try:
        import cv2  # noqa: F401
        return "cv2"
    except Exception:   # noqa: BLE001
        return None


BLOB_BACKEND = _backend()


def binarize(matte: np.ndarray, threshold: float = 0.5, main_only: bool = False, min_frac: float = 0.25) -> np.ndarray:
    """(F,H,W) matte -> (F,H,W) float32 0/1 (> threshold), optionally keeping only the main blob(s) of every frame."""
    b = np.asarray(matte, np.float32) > threshold
    if main_only:
        b = np.stack([keep_main_blob(f, min_frac) for f in b])
    return b.astype(np.float32)


def cell_coverage(matte: np.ndarray, h_lat: int, w_lat: int, threshold: float = 0.5) -> np.ndarray:
    """(F,H,W) matte -> (F,h,w) fraction of each cell's pixels with matte > threshold."""
    b = (np.asarray(matte, np.float32) > threshold).astype(np.float32)
    _, H, W = b.shape
    ey, ex = _edges(H, h_lat), _edges(W, w_lat)
    s = np.add.reduceat(np.add.reduceat(b, ey[:-1], axis=1), ex[:-1], axis=2)
    area = np.outer(np.diff(ey), np.diff(ex)).astype(np.float32)
    return s / area[None]


def pool_time(cov: np.ndarray, t_lat: int) -> np.ndarray:
    """(F,h,w) -> (T_lat,h,w): max over each latent frame's pixel group (frame 0 alone). Pads/truncates to 8(T-1)+1 frames."""
    cov = fit_frames(cov, pixel_frames(t_lat))
    out = np.empty((t_lat,) + cov.shape[1:], cov.dtype)
    for k in range(t_lat):
        a, b = group_slice(k)
        out[k] = cov[a:b].max(0)
    return out


def dilate(cells: np.ndarray, n: int) -> np.ndarray:
    """Binary (T,h,w) dilation by a 3x3 square, n times (spatial only). Returns bool."""
    c = np.asarray(cells, bool)
    for _ in range(int(n)):
        p = np.pad(c, ((0, 0), (1, 1), (1, 1)))
        o = np.zeros_like(c)
        for dy in range(3):
            for dx in range(3):
                o |= p[:, dy:dy + c.shape[1], dx:dx + c.shape[2]]
        c = o
    return c


def gauss_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian over the last two axes, zero padding. a: (...,H,W) float."""
    a = np.asarray(a, np.float32)
    if sigma <= 0:
        return a
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    out = a
    for axis in (-2, -1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad)
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += w * p[tuple(sl)]
        out = acc
    return out


def soften(binary: np.ndarray, sigma: float) -> np.ndarray:
    """Binary/float mask -> float mask: edge-centred gaussian x2 (clipped), never below the input. Inside stays 1."""
    b = np.asarray(binary, np.float32)
    if sigma <= 0:
        return b
    return np.maximum(b, np.clip(2.0 * gauss_blur(b, sigma), 0.0, 1.0))


# --------------------------------------------------------------------------------------------- main entry points
def make_footprint(matte: np.ndarray, t_lat: int, h_lat: int, w_lat: int, *, dilate_cells: int = 1,
                   feather: float = 0.0, keep_first_frame: bool = False, threshold: float = 0.5,
                   cell_threshold: float = 0.0, main_subject_only: bool = False, min_blob_frac: float = 0.25,
                   extra: np.ndarray | None = None) -> np.ndarray:
    """Matte (F,H,W) in 0..1 -> footprint (T_lat,h,w) float32 in 0..1 (1 = regenerate). See module doc.
    main_subject_only keeps only the main blob(s) of the matte per frame; `extra` (F,H,W) is a second matte (not
    blob-filtered) whose cells are added (max) before the cell threshold."""
    matte = np.asarray(matte, np.float32)
    if matte.ndim != 3:
        raise ValueError(f"matte must be (F,H,W), got {matte.shape}")
    n = pixel_frames(t_lat)
    b = binarize(fit_frames(matte, n), threshold, main_subject_only, min_blob_frac)
    cov = cell_coverage(b, h_lat, w_lat, 0.5)
    if extra is not None:
        extra = np.asarray(extra, np.float32)
        if extra.ndim != 3:
            raise ValueError(f"extra must be (F,H,W), got {extra.shape}")
        cov = np.maximum(cov, cell_coverage(fit_frames(extra, n), h_lat, w_lat, threshold))
    cov = pool_time(cov, t_lat)
    cells = cov > float(cell_threshold)
    cells = dilate(cells, dilate_cells)
    fp = soften(cells, feather).astype(np.float32)
    if keep_first_frame:
        fp[0] = 0.0
    return fp


def noise_mask_from_footprint(fp: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """(T,h,w) -> (1,1,T,h,w) float32 (batch of 1; the node expands it)."""
    return (np.asarray(fp, np.float32) * float(strength))[None, None]


def combine_noise_masks(existing, new: np.ndarray) -> np.ndarray:
    """Multiply `new` (B?,1,T,h,w) with an `existing` noise mask ((B,1,T,1,1) from LTXVImgToVideoInplace, or full
    (B,1,T,h,w)). Product = intersection of the two 'regenerate' areas: a frame/cell either side keeps wins.
    -> (B,1,T,h,w). Frame counts must match."""
    new = np.asarray(new, np.float32)
    if existing is None:
        return new
    ex = np.asarray(existing, np.float32)
    if ex.ndim != 5:
        raise ValueError(f"existing noise_mask must be 5D (B,1,T,h|1,w|1), got {ex.shape}")
    if ex.shape[2] != new.shape[2]:
        raise ValueError(f"existing noise_mask has {ex.shape[2]} latent frames, new one {new.shape[2]}")
    if ex.shape[3] not in (1, new.shape[3]) or ex.shape[4] not in (1, new.shape[4]):
        raise ValueError(f"existing noise_mask spatial {ex.shape[3:]} does not match {new.shape[3:]}")
    return (ex * new).astype(np.float32)


def region_pixels(fp: np.ndarray, n_pixel_frames: int, height: int, width: int, feather_px: float = 0.0) -> np.ndarray:
    """Footprint (T,h,w) -> (F,H,W) float32 per pixel frame: each cell nearest-upsampled to H x W, pixel frame f takes
    latent frame frame_to_latent(f), then the same edge-centred soften with sigma ``feather_px`` pixels."""
    t, h, w = fp.shape
    yi = np.minimum((np.arange(height) + 0.5) * h / height, h - 1).astype(int)
    xi = np.minimum((np.arange(width) + 0.5) * w / width, w - 1).astype(int)
    per_latent = np.empty((t, height, width), np.float32)
    for k in range(t):
        up = fp[k][yi][:, xi]
        per_latent[k] = np.maximum(up, np.clip(2.0 * gauss_blur(up, feather_px), 0.0, 1.0)) if feather_px > 0 else up
    idx = np.clip(frame_to_latent(np.arange(n_pixel_frames)), 0, t - 1)
    return per_latent[idx]


def cells_to_text(fp: np.ndarray, k: int) -> str:
    """ASCII picture of latent frame k (debug / README)."""
    return "\n".join("".join("#" if v >= 0.999 else ("+" if v > 0.01 else ".") for v in row) for row in fp[k])
