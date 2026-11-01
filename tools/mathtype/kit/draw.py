"""Drawing primitives, implicit and per pixel, on torch (cuda or cpu) with numpy fallbacks where cheap.

Coordinates: pixel centres at (i + 0.5) unless noted; fields are (H, W) float32; colours are 0..1 RGB.
Every write is order-independent (scatter-add / amax), so results are deterministic on CUDA under
xp.Backend's settings.

- canvas / paint / to_u8 / box_down / blur ............ compositing, supersample-and-box-down
- gradmag / fill / iso_line / iso_bands ............... SDF fills and AA iso-lines (E1 contour bands)
- disc / box / tri / coverage / discs .................. analytic SDF shapes (E1 accents, E5 seeds)
- splat / splat_polylines .............................. bilinear scatter-add of points and dense segments (E2, D)
- fill_tris ............................................ batched triangle fill, edge functions (E3)
- square_wave / hash01 / value_noise / sample / upsample  E4 helpers (AA gratings, integer-hash noise)
"""
from __future__ import annotations

import math

import numpy as np

import xp


def _T():
    return xp.torch()


# ------------------------------------------------------------------------------------------- compositing
def canvas(be, H, W, colour=(0.0, 0.0, 0.0), batch=None):
    """(H, W, 3) (or (B, H, W, 3)) float32 filled with colour."""
    c = np.asarray(colour, np.float32)
    shape = ((batch,) if batch else ()) + (H, W, 3)
    if be.torch:
        return _T().as_tensor(c, device=be.dev).expand(*shape).clone()
    return np.broadcast_to(c, shape).copy()


def paint(img, alpha, colour):
    """In place: img += (colour - img) * alpha. alpha (..., H, W); colour RGB tuple / array."""
    if xp.is_t(img):
        c = _T().as_tensor(np.asarray(colour, np.float32), device=img.device)
        img += (c - img) * alpha[..., None]
    else:
        img += (np.asarray(colour, np.float32) - img) * alpha[..., None]
    return img


def tone(acc, exposure=1.0):
    """Film-like tone map of accumulated ink: 1 - exp(-exposure * acc)."""
    return 1 - xp.exp(-exposure * acc)


def to_u8(img):
    """0..1 float image -> uint8 numpy (rounded)."""
    if xp.is_t(img):
        return (img.clamp(0, 1) * 255 + 0.5).to(_T().uint8).cpu().numpy()
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def box_down(img, ss: int):
    """Supersample -> output: ss x ss box mean of (H*ss, W*ss[, C]) or batched (B, H*ss, W*ss[, C])."""
    if ss == 1:
        return img
    sh = tuple(img.shape)
    if len(sh) >= 3 and sh[-1] in (1, 3, 4) and sh[-2] > 4:     # channels last
        lead, (Hs, Ws, C) = sh[:-3], sh[-3:]
        r = img.reshape(*lead, Hs // ss, ss, Ws // ss, ss, C)
        ax = (len(lead) + 1, len(lead) + 3)
    else:
        lead, (Hs, Ws) = sh[:-2], sh[-2:]
        r = img.reshape(*lead, Hs // ss, ss, Ws // ss, ss)
        ax = (len(lead) + 1, len(lead) + 3)
    return r.mean(ax)


def blur(img, sigma):
    """cv2.GaussianBlur(img, (0, 0), sigma) on (..., H, W) float32 (torch: identical kernel and border)."""
    if xp.is_t(img):
        return xp.gaussian_blur(img, sigma)
    import cv2
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


# ------------------------------------------------------------------------------------------- SDF fills and lines
def gradmag(f):
    """|grad f| per pixel (central differences, one-sided at the edges), + 1e-4."""
    if xp.is_t(f):
        pad = _T().nn.functional.pad(f[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    else:
        pad = np.pad(f, 1, mode="edge")
    gx = pad[1:-1, 2:] - pad[1:-1, :-2]
    gy = pad[2:, 1:-1] - pad[:-2, 1:-1]
    return (gx * gx + gy * gy) ** 0.5 * 0.5 + 1e-4


def fill(f, gm=None, aa=1.0):
    """AA coverage of the region f < 0 (f in any units; gm = |grad f| per pixel, computed if None)."""
    gm = gradmag(f) if gm is None else gm
    return xp.clip(0.5 - f / (gm * aa), 0.0, 1.0)


def iso_line(f, level, width, gm=None):
    """AA line of `width` px on the iso-contour f = level (screen distance (f - level) / |grad f|)."""
    gm = gradmag(f) if gm is None else gm
    d = (f - level) / gm
    return xp.clip(width * 0.5 + 0.5 - (d.abs() if xp.is_t(d) else np.abs(d)), 0.0, 1.0)


def iso_bands(f, spacing, width, gm=None, offset=0.0, lo=None, hi=None):
    """AA lines on every iso-level offset + n * spacing of f (topographic echoes), optionally only for
    levels in [lo, hi]. Returns (alpha, level index n) so callers can tone bands by n."""
    gm = gradmag(f) if gm is None else gm
    q = (f - offset) / spacing
    n = xp.floor(q + 0.5)
    d = (q - n) * spacing / gm
    a = xp.clip(width * 0.5 + 0.5 - (d.abs() if xp.is_t(d) else np.abs(d)), 0.0, 1.0)
    if lo is not None:
        a = a * (n >= lo)
    if hi is not None:
        a = a * (n <= hi)
    return a, n


def coverage(sd, aa=1.0):
    """Pixel-unit SDF -> AA coverage (1 inside)."""
    return xp.clip(0.5 - sd / aa, 0.0, 1.0)


def disc(px, py, cx, cy, r):
    return xp.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r


def box(px, py, cx, cy, hx, hy):
    qx = (px - cx).abs() - hx if xp.is_t(px) else np.abs(px - cx) - hx
    qy = (py - cy).abs() - hy if xp.is_t(py) else np.abs(py - cy) - hy
    return xp.sqrt(xp.maximum(qx, 0.0) ** 2 + xp.maximum(qy, 0.0) ** 2) + xp.minimum(xp.maximum(qx, qy), 0.0)


def tri(px, py, cx, cy, size):
    """Equilateral triangle pointing up, half-width `size` (Inigo Quilez's SDF; E1's accent)."""
    k = math.sqrt(3.0)
    x = ((px - cx).abs() if xp.is_t(px) else np.abs(px - cx)) - size
    y = -(py - cy) + size / k
    sw = x + k * y > 0
    x2 = xp.where(sw, (x - k * y) / 2, x)
    y2 = xp.where(sw, (-k * x - y) / 2, y)
    x2 = x2 - xp.clip(x2, -2 * size, 0.0)
    sgn = y2.sign() if xp.is_t(y2) else np.sign(y2)
    return -xp.sqrt(x2 * x2 + y2 * y2) * sgn


def discs(px, py, C, R, chunk=64):
    """Union SDF of N discs, centres C (N, 2), radii R (N,) or scalar; chunked min over N."""
    C = np.asarray(C, np.float32).reshape(-1, 2)
    R = np.broadcast_to(np.asarray(R, np.float32), (len(C),))
    out = None
    for i in range(0, len(C), chunk):
        c, r = C[i:i + chunk], R[i:i + chunk]
        if xp.is_t(px):
            T = _T()
            cx = T.as_tensor(c[:, 0], device=px.device).view(-1, 1, 1)
            cy = T.as_tensor(c[:, 1], device=px.device).view(-1, 1, 1)
            rr = T.as_tensor(r, device=px.device).view(-1, 1, 1)
            d = (T.sqrt((px[None] - cx) ** 2 + (py[None] - cy) ** 2) - rr).amin(0)
        else:
            d = (np.sqrt((px[None] - c[:, 0, None, None]) ** 2 + (py[None] - c[:, 1, None, None]) ** 2)
                 - r[:, None, None]).min(0)
        out = d if out is None else xp.minimum(out, d)
    return out


# ------------------------------------------------------------------------------------------- splats
def splat(xy, w, H, W, layer=None, nlayer=1, batch=None):
    """Bilinear scatter-add of points xy (N, 2) (pixel coords, centres at +0.5) with weights w (N,) into
    (nlayer, H, W) [(batch, nlayer, H, W) if batch: pass layer = batch_index * nlayer + layer]. torch."""
    T = _T()
    dev = xy.device
    x, y = xy[:, 0], xy[:, 1]
    x0, y0 = T.floor(x - 0.5), T.floor(y - 0.5)
    fx, fy = x - 0.5 - x0, y - 0.5 - y0
    x0, y0 = x0.long() + 1, y0.long() + 1                  # 1 px guard ring
    ok = (x0 >= 0) & (y0 >= 0) & (x0 <= W) & (y0 <= H)
    nl = nlayer * (batch or 1)
    ly = layer[ok] if layer is not None else T.zeros(int(ok.sum()), dtype=T.long, device=dev)
    x0, y0, fx, fy, w = x0[ok], y0[ok], fx[ok], fy[ok], w[ok]
    gi = ly * ((H + 2) * (W + 2)) + y0 * (W + 2) + x0
    idx = T.cat([gi, gi + 1, gi + W + 2, gi + W + 3])
    val = T.cat([w * (1 - fx) * (1 - fy), w * fx * (1 - fy), w * (1 - fx) * fy, w * fx * fy])
    buf = T.zeros(nl * (H + 2) * (W + 2), device=dev, dtype=val.dtype)
    buf.index_put_((idx,), val, accumulate=True)           # deterministic under use_deterministic_algorithms
    out = buf.view(nl, H + 2, W + 2)[:, 1:-1, 1:-1]
    return (out.reshape(batch, nlayer, H, W) if batch else out).contiguous()


def densify(A, B, IA, IB, step=0.6, max_n=600):
    """Points every <= step px along segments A->B (M, 2) with linearly interpolated intensity; weight =
    intensity x sub-segment length (so line density is independent of sampling). Returns xy, w, seg id."""
    T = _T()
    D = B - A
    ln = T.linalg.vector_norm(D, dim=1)
    ns = T.clamp(T.ceil(ln / step), 1, max_n).long()
    rep = T.repeat_interleave(ns)
    start = (T.cumsum(ns, 0) - ns)[rep]
    f = ((T.arange(rep.numel(), device=A.device) - start).to(A.dtype) + 0.5) / ns[rep].to(A.dtype)
    xy = A[rep] + D[rep] * f[:, None]
    w = (IA[rep] + (IB - IA)[rep] * f) * (ln / ns.to(A.dtype))[rep]
    return xy, w, rep


def splat_polylines(P, I, H, W, step=0.6, closed=False, layer=None, nlayer=1, margin=4.0):
    """Polylines P (K, M, 2) px with per-vertex intensity I (K, M) -> (nlayer, H, W) line density.
    layer: (K,) long layer per polyline. Segments fully off-screen or dark are culled first."""
    T = _T()
    if closed:
        P = T.cat([P, P[:, :1]], 1)
        I = T.cat([I, I[:, :1]], 1)
    K, M = P.shape[:2]
    A, B = P[:, :-1].reshape(-1, 2), P[:, 1:].reshape(-1, 2)
    IA, IB = I[:, :-1].reshape(-1), I[:, 1:].reshape(-1)
    LY = (layer[:, None].expand(K, M - 1).reshape(-1) if layer is not None
          else T.zeros(K * (M - 1), dtype=T.long, device=P.device))
    lo, hi = T.minimum(A, B), T.maximum(A, B)
    keep = (((IA > 1e-4) | (IB > 1e-4)) & (lo[:, 0] < W + margin) & (hi[:, 0] > -margin)
            & (lo[:, 1] < H + margin) & (hi[:, 1] > -margin))
    A, B, IA, IB, LY = A[keep], B[keep], IA[keep], IB[keep], LY[keep]
    xy, w, rep = densify(A, B, IA, IB, step)
    return splat(xy, w, H, W, LY[rep], nlayer)


# ------------------------------------------------------------------------------------------- triangles
_POW2 = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


def fill_tris(tri, vals, H, W, bufs=None, budget=4_000_000):
    """Filled triangles (pixel centres at integer coords; inside = all edge functions >= 0), batched by
    power-of-two bounding-box buckets. tri (N, 3, 2) px; vals: list of (N,) per-triangle values; each value
    is written to its own flat (H*W) buffer with amax (order-independent). Returns the buffers as (H, W)."""
    T = _T()
    V = tri if xp.is_t(tri) else T.as_tensor(np.asarray(tri, np.float32))
    dev = V.device
    vals = [v if xp.is_t(v) else T.as_tensor(np.asarray(v, np.float32), device=dev) for v in vals]
    bufs = bufs or [T.zeros(H * W, device=dev) for _ in vals]
    if len(V) == 0:
        return [b.view(H, W) for b in bufs]
    lo, hi = V.min(dim=1).values.floor(), V.max(dim=1).values.ceil()
    wh = ((hi - lo) + 2).cpu().numpy()
    l_, h_ = lo.cpu().numpy(), hi.cpu().numpy()
    vis = (h_[:, 0] >= 0) & (h_[:, 1] >= 0) & (l_[:, 0] < W) & (l_[:, 1] < H)
    e = np.minimum(np.searchsorted(_POW2, wh.max(1)), len(_POW2) - 1)
    e = np.where(vis, e, -1)
    ar = T.arange(int(_POW2[-1]), device=dev, dtype=T.float32)
    area = ((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0]))
    sg_all = T.where(area >= 0, 1.0, -1.0)
    for kk in np.unique(e[e >= 0]):
        sel_np = np.nonzero(e == kk)[0]
        bw = int(_POW2[kk])
        step = max(1, budget // (bw * bw))
        for i in range(0, len(sel_np), step):
            sel = T.as_tensor(sel_np[i:i + step], device=dev)
            v = V[sel]
            x = lo[sel, 0][:, None, None] + ar[:bw][None, None, :]
            y = lo[sel, 1][:, None, None] + ar[:bw][None, :, None]
            sg = sg_all[sel][:, None, None]
            inside = None
            for j in range(3):
                a, b = v[:, j], v[:, (j + 1) % 3]
                ef = ((b[:, 0] - a[:, 0])[:, None, None] * (y - a[:, 1][:, None, None])
                      - (b[:, 1] - a[:, 1])[:, None, None] * (x - a[:, 0][:, None, None]))
                ok = (ef * sg) >= 0
                inside = ok if inside is None else inside & ok
            xi, yi = x.expand_as(inside).long(), y.expand_as(inside).long()
            cov = (inside & (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)).float()
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).reshape(-1)
            for buf, val in zip(bufs, vals):
                buf.scatter_reduce_(0, idx, (val[sel][:, None, None] * cov).reshape(-1), reduce="amax", include_self=True)
    return [b.view(H, W) for b in bufs]


# ------------------------------------------------------------------------------------------- E4 helpers
def gradient(phi):
    return _T().gradient(phi) if xp.is_t(phi) else np.gradient(phi)


def square_wave(phi):
    """Anti-aliased square wave of phase phi: clip(1/2 + sin(phi) / |grad phi|) (per-pixel slope from the
    numerical gradient; ramp ~1 px wide, -> grey past Nyquist)."""
    gy, gx = gradient(phi)
    g = xp.sqrt(gx * gx + gy * gy) + 1e-3
    return xp.clip(0.5 + xp.sin(phi) / g, 0.0, 1.0)


def to_int(x):
    return x.long() if xp.is_t(x) else x.astype(np.int64)


def hash01(ix, iy, seed):
    """Integer hash of integer cell indices -> [0, 1). int64 maths, identical on numpy and torch."""
    M = 0xFFFFFFFF
    h = (ix * 374761393 + iy * 668265263 + (seed * 2246822519 & M)) & M
    h = ((h ^ (h >> 13)) * 1274126177) & M
    h = h ^ (h >> 16)
    return (h.float() if xp.is_t(h) else h.astype(np.float32)) * (1.0 / 4294967296.0)


def value_noise(x, y, L, seed):
    """Smooth value noise in [-0.5, 0.5) on cells of size L (C1 smoothstep interpolation of hash01)."""
    gx, gy = x / L, y / L
    fx0, fy0 = xp.floor(gx), xp.floor(gy)
    fx, fy = gx - fx0, gy - fy0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = to_int(fx0), to_int(fy0)
    a = hash01(ix, iy, seed)
    b = hash01(ix + 1, iy, seed)
    c = hash01(ix, iy + 1, seed)
    d = hash01(ix + 1, iy + 1, seed)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy - 0.5


def upsample(x, shape):
    """Bilinear (half-pixel centres) resize of a 2D field to shape (H, W)."""
    if xp.is_t(x):
        return _T().nn.functional.interpolate(x[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
    import cv2
    return cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def sample(img, X, Y, k=1.0, border="border"):
    """Bilinear sample of a (C, h, w) field at coords X, Y (k field px per coord unit; pixel centres at
    +0.5), edge clamp. Returns (C, *X.shape). numpy: cv2.remap per channel."""
    if xp.is_t(img):
        T = _T()
        Hs, Ws = img.shape[1], img.shape[2]
        g = T.stack([X * k / Ws * 2 - 1, Y * k / Hs * 2 - 1], -1)[None]
        return T.nn.functional.grid_sample(img[None], g, mode="bilinear", padding_mode=border, align_corners=False)[0]
    import cv2
    mx, my = (X * k - 0.5).astype(np.float32), (Y * k - 0.5).astype(np.float32)
    bm = cv2.BORDER_REPLICATE if border == "border" else cv2.BORDER_CONSTANT
    return np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=bm) for c in img])
