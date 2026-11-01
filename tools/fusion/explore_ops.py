"""explore_ops: EXPLORATION compose ops for masks / grounds / outlines (2026-10-30, `16_app/core_check/explore_masks/GALLERY.md`).

Single-shot interplay of text, matte and image: grounds (duotone / halftone / depth bands / matte bands), masks (counters, morph, mask maths), brush outlines (leading
edge of motion, colour sampled from the footage). Numpy ops (the engine's numpy detour), images HxWx3 float 0..1, masks HxW 0..1. Registered by
`compose._ops_registry` under their own names only: nothing in a default chain uses them, so no default changes."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _g(x, h=None, w=None):
    a = np.asarray(x, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[2] == 1 else a.mean(2)
    if h is not None and a.shape[:2] != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return a


def _rgb(c, like):
    if isinstance(c, (tuple, list)):
        return np.broadcast_to(np.asarray(c, np.float32).reshape(1, 1, 3), like.shape).copy()
    a = np.asarray(c, np.float32)
    if a.shape[:2] != like.shape[:2]:
        a = cv2.resize(a, (like.shape[1], like.shape[0]), interpolation=cv2.INTER_LINEAR)
    return a


def _lum(img):
    i = np.asarray(img, np.float32)
    return i[..., 0] * 0.2126 + i[..., 1] * 0.7152 + i[..., 2] * 0.0722


# ------------------------------------------------------------------ grounds
def duotone(img, dark=(0.05, 0.05, 0.08), light=(0.95, 0.9, 0.8), contrast=1.3, pivot=0.5):
    """luma of img mapped between two colours (a moving duotone of the footage)."""
    l = np.clip((_lum(img) - pivot) * float(contrast) + pivot, 0.0, 1.0)[..., None]
    d, li = np.asarray(dark, np.float32).reshape(1, 1, 3), np.asarray(light, np.float32).reshape(1, 1, 3)
    return d * (1.0 - l) + li * l


def halftone(img, cell=7.0, ink=(0.08, 0.08, 0.1), paper=(0.86, 0.85, 0.82), angle=22.0, gain=1.0, colour=False):
    """round-dot screen of img: dot radius from the darkness (or, colour=True, dots carry the footage colour). AA dots, fixed screen so the image moves under it."""
    im = np.asarray(img, np.float32)
    H, W = im.shape[:2]
    a = math.radians(angle)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx * math.cos(a) + yy * math.sin(a)) / cell
    v = (-xx * math.sin(a) + yy * math.cos(a)) / cell
    fu, fv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
    d = np.sqrt(fu * fu + fv * fv) * cell                                   # px from the cell centre
    small = cv2.blur(im, (int(cell), int(cell)))
    dark = np.clip((1.0 - _lum(small)) * float(gain), 0.0, 1.0)
    r = np.sqrt(dark) * cell * 0.62
    dot = np.clip(r - d + 0.5, 0.0, 1.0)[..., None]
    pa = np.asarray(paper, np.float32).reshape(1, 1, 3)
    ik = small if colour else np.asarray(ink, np.float32).reshape(1, 1, 3)
    return pa * (1.0 - dot) + ik * dot


def posterize(img, palette, soft=0.0, invert=False):
    """each pixel -> its nearest palette colour by luma order (k tones, moving with the footage); invert = the tone order reversed (lights go dark: contrast against the shot)."""
    pal = np.asarray(palette, np.float32).reshape(-1, 3)
    pal = pal[np.argsort(pal @ np.array([0.2126, 0.7152, 0.0722], np.float32))]
    if invert:
        pal = pal[::-1]
    l = _lum(img)
    k = len(pal)
    q = np.clip((l * k).astype(np.int32), 0, k - 1)
    return pal[q]


def depth_bands(depth, n=5.0, width=0.42, speed=0.35, t=0.0, soft=0.04, like=None):
    """HxW 0..1: topographic slices of the depth (frac(depth * n - t * speed) < width), drifting through depth over time."""
    d = _g(depth)
    if like is not None:
        h, w = np.asarray(like).shape[:2]
        d = _g(d, h, w)
    x = d * float(n) - float(t) * float(speed)
    f = x - np.floor(x)
    s = float(soft)
    return np.clip(np.minimum((f + s) / (2 * s), (width + s - f) / (2 * s)), 0.0, 1.0)


def matte_bands(matte, spacing=0.05, width=0.45, speed=0.6, t=0.0, thresh=0.5, reach=0.9):
    """HxW 0..1: rings around the subject (distance from its matte) travelling outward, `spacing` as share of the frame diagonal; 0 inside the subject."""
    m = _g(matte)
    H, W = m.shape
    b = (m <= thresh).astype(np.uint8)
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    diag = math.hypot(H, W)
    x = dist / (float(spacing) * diag) - float(t) * float(speed)
    f = x - np.floor(x)
    ring = np.clip(np.minimum(f / 0.08, (width - f) / 0.08), 0.0, 1.0)
    fade = np.clip(1.0 - dist / (reach * diag), 0.0, 1.0)
    return ring * (dist > 0.5) * (fade > 0)


# ------------------------------------------------------------------ mask maths
def mask_max(a, b):
    a, b = _g(a), _g(b)
    return np.maximum(a, _g(b, *a.shape))


def mask_sub(a, b):
    a, b = _g(a), _g(b)
    return np.clip(a - _g(b, *a.shape), 0.0, 1.0)


def mask_xor(a, b):
    a, b = _g(a), _g(b)
    b = _g(b, *a.shape)
    return np.clip(a + b - 2.0 * a * b, 0.0, 1.0)


def mask_inv(a):
    return 1.0 - np.clip(_g(a), 0.0, 1.0)


def counters(type_a, thresh=0.5, soften=0.6):
    """the enclosed counters of the letters (O, A, R, D ...) as a mask: holes filled minus the letters."""
    a = np.clip(_g(type_a), 0.0, 1.0)
    b = (a > thresh).astype(np.uint8)
    cnts, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(b)
    if hier is not None:
        for i, hrow in enumerate(hier[0]):
            if hrow[3] >= 0:                                                # an inner contour = a counter
                cv2.drawContours(out, cnts, i, 1, thickness=-1)
    f = out.astype(np.float32)
    if soften > 0:
        f = cv2.GaussianBlur(f, (0, 0), soften)
    return np.clip(f - a, 0.0, 1.0)


def sdf_morph(a, b, k=0.0, soft=1.0):
    """shape between mask a (k=0) and mask b (k=1) by blending signed distance fields (the subject's form turns into the letters)."""
    a, b = _g(a), _g(b)
    b = _g(b, *a.shape)

    def sdf(m):
        bi = (m > 0.5).astype(np.uint8)
        return cv2.distanceTransform(1 - bi, cv2.DIST_L2, 5) - cv2.distanceTransform(bi, cv2.DIST_L2, 5)     # >0 outside
    kk = float(np.clip(k, 0.0, 1.0))
    s = (1.0 - kk) * sdf(a) + kk * sdf(b)
    return np.clip(0.5 - s / max(2.0 * soft, 1e-3), 0.0, 1.0)


def mask_shift(a, dx=0.0, dy=0.0):
    a = _g(a)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------------ blends
def difference(a, b, alpha=None):
    a = np.asarray(a, np.float32)
    r = np.abs(a - _rgb(b, a))
    if alpha is None:
        return r
    al = _g(alpha, *a.shape[:2])[..., None]
    return a * (1.0 - al) + r * al


def invert_in(img, alpha):
    im = np.asarray(img, np.float32)
    al = _g(alpha, *im.shape[:2])[..., None]
    return im * (1.0 - al) + (1.0 - im) * al


# ------------------------------------------------------------------ outlines
def _sampled_colour(sample, sat=1.6, lum_lo=0.55, sigma=9.0, lum=None):
    """footage colour, blurred, saturation boosted and lifted to a readable brightness (lum = a fixed luminance target: the hue of the footage at that brightness)."""
    s = cv2.GaussianBlur(np.asarray(sample, np.float32), (0, 0), sigma)
    l = _lum(s)[..., None]
    c = l + (s - l) * float(sat)
    lt = np.maximum(l, lum_lo) if lum is None else np.full_like(l, float(lum))
    return np.clip(c * (lt / np.maximum(l, 1e-3)), 0.0, 1.0)


def brush_edge(img, mask, frame_idx=0, color=(1.0, 1.0, 1.0), sample=None, width_px=2.2, n_strokes=2, wobble=0.018, seed=0, lead=None, lead_gain=6.0,
               trail=0, trail_step=4.0, avoid=None, opacity=1.0, guard=True, sat=1.6, sample_lum=None, lead_sigma=0.03, var=1.0):
    """live brush-vector strokes along the contour of `mask` (contour_sweep.edge_alpha: variable width, tapered, 2-3 strokes, wobble, breaks; idle flow) painted on img.
    sample = an image to take the stroke colour from (footage colour under the stroke, boosted), else `color`. var = thick-thin variation along the stroke (1.0 = the contour_sweep band 0.3-1.6 x the width; 1.25 = the liked line scale of explore_cuts2 draw_brush). lead = the mask a few frames earlier: the strokes are kept only on
    the LEADING edge of the motion (where the form is newly arriving) and `trail` thinner echo strokes are laid behind it, offset against the motion."""
    from tools.fusion.contour_sweep import edge_alpha
    im = np.asarray(img, np.float32)
    H, W = im.shape[:2]
    m = _g(mask, H, W)
    pr = dict(n_strokes=int(n_strokes), wobble=float(wobble))
    if float(var) != 1.0:
        pr["width_range"] = (max(0.05, 0.95 - 0.65 * float(var)), 0.95 + 0.65 * float(var))
    la = edge_alpha(m, int(frame_idx), seed=int(seed), width_px=float(width_px) * W / 360.0, params=pr, guard=bool(guard))
    if lead is not None:
        p = _g(lead, H, W)
        arrive = np.clip(m - p, 0.0, 1.0)
        leave = np.clip(p - m, 0.0, 1.0)
        mov = cv2.GaussianBlur(arrive, (0, 0), float(lead_sigma) * W)
        w8 = np.clip(mov * float(lead_gain) / max(float(mov.max()), 1e-4), 0.0, 1.0)
        la = la * w8
        if trail:
            ys, xs = np.nonzero(arrive > 0.2)
            yl, xl = np.nonzero(leave > 0.2)
            if len(xs) and len(xl):
                v = np.array([xs.mean() - xl.mean(), ys.mean() - yl.mean()], np.float32)
                v = v / max(float(np.linalg.norm(v)), 1e-3)
                acc = la.copy()
                for k in range(1, int(trail) + 1):
                    sh = cv2.warpAffine(la, np.float32([[1, 0, -v[0] * trail_step * k], [0, 1, -v[1] * trail_step * k]]), (W, H))
                    acc = np.maximum(acc, sh * (0.75 ** k))
                la = acc
    if avoid is not None:
        la = la * (1.0 - np.clip(_g(avoid, H, W), 0.0, 1.0))
    col = _sampled_colour(sample, sat=sat, lum=sample_lum) if sample is not None else _rgb(color, im)
    a3 = np.clip(la * float(opacity), 0.0, 1.0)[..., None]
    return im * (1.0 - a3) + col * a3


def type_rows(like, text_mask_row, rows=9, speed=24.0, t=0.0, gap=0.25):
    """the subject filled with type: a strip of small word(s) (`text_mask_row`, HxW mask of ONE row, any width) tiled in rows that scroll alternately left / right. -> HxW mask."""
    lk = np.asarray(like)
    H, W = lk.shape[:2]
    r = _g(text_mask_row)
    ys, xs = np.nonzero(r > 0.3)
    if not len(ys):
        return np.zeros((H, W), np.float32)
    tile = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    th, tw = tile.shape
    pad = int(tw * gap / 4) + 2
    tile = np.pad(tile, ((0, 0), (0, pad)))
    th, tw = tile.shape
    step = H / float(rows)
    sc = (step * (1.0 - gap)) / th
    tile = cv2.resize(tile, (max(4, int(tw * sc)), max(2, int(th * sc))), interpolation=cv2.INTER_AREA)
    th, tw = tile.shape
    out = np.zeros((H, W), np.float32)
    reps = W // tw + 3
    strip = np.tile(tile, (1, reps))
    for i in range(int(rows) + 1):
        y0 = int(round(i * step))
        off = int(round((float(t) * speed * (1 if i % 2 == 0 else -1)) % tw))
        seg = strip[:, off:off + W]
        y1 = min(H, y0 + th)
        if y1 > y0:
            out[y0:y1, :seg.shape[1]] = np.maximum(out[y0:y1, :seg.shape[1]], seg[:y1 - y0])
    return out


OPS = {
    "duotone": duotone, "halftone": halftone, "posterize": posterize, "depth_bands": depth_bands, "matte_bands": matte_bands,
    "mask_max": mask_max, "mask_sub": mask_sub, "mask_xor": mask_xor, "mask_inv": mask_inv, "counters": counters, "sdf_morph": sdf_morph,
    "mask_shift": mask_shift, "difference": difference, "invert_in": invert_in, "brush_edge": brush_edge, "type_rows": type_rows,
}
