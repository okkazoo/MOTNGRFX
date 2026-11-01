"""Image + audio analysis (SPEC Interface 1).

Pure numpy / scipy / OpenCV; no ComfyUI imports. All boxes and points are
normalised 0..1 in image space, ``[x, y, w, h]`` with a top-left origin.

Line ids follow the convention ``f"{image_id}:l{i}"`` (contours: ``:c{i}``).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import cv2
import numpy as np
from scipy import ndimage, signal

__all__ = ["analyze_image", "analyze_audio", "audio_envelope", "analyze_inputs"]

_WORK_LONG_SIDE = 512
_PALETTE_MAX_SIDE = 200
_SAL_SIDE = 128

# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------


def _to_rgb_float(pixels) -> np.ndarray:
    """Any HxW, HxWx1/3/4 array (uint8 or float) -> HxWx3 float32 0..1."""
    a = np.asarray(pixels)
    if a.ndim == 4:  # a batch of one
        a = a[0]
    if a.ndim == 2:
        a = a[:, :, None]
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if a.size and float(np.nanmax(a)) > 1.5:  # 0..255 floats
            a = a / 255.0
    a = np.nan_to_num(a, nan=0.0)
    c = a.shape[2]
    if c == 1:
        a = np.repeat(a, 3, axis=2)
    elif c == 2:  # gray + alpha
        al = a[:, :, 1:2]
        a = np.repeat(a[:, :, :1] * al, 3, axis=2)
    elif c >= 4:  # RGBA: composite over black
        al = a[:, :, 3:4]
        a = a[:, :, :3] * al
    return np.ascontiguousarray(np.clip(a, 0.0, 1.0), dtype=np.float32)


def _resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if s >= 1.0:
        return img
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _hex(rgb) -> str:
    r, g, b = (int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _luma(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.float32)


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# ----------------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------------


def _palette(img: np.ndarray, k: int = 6):
    small = _resize_long(img, _PALETTE_MAX_SIDE)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k_eff = int(min(k, max(1, len(lab))))
    cv2.setRNGSeed(1234)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(lab, k_eff, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k_eff).astype(np.float64)
    # merge near-identical centres (delta E < 4) so flat images do not report duplicates
    order = np.argsort(-counts)
    merged: list[list] = []  # [lab_center, count]
    for i in order:
        if counts[i] <= 0:
            continue
        c = centers[i].astype(np.float64)
        for m in merged:
            if np.linalg.norm(m[0] - c) < 4.0:
                tot = m[1] + counts[i]
                m[0] = (m[0] * m[1] + c * counts[i]) / tot
                m[1] = tot
                break
        else:
            merged.append([c, counts[i]])
    total = sum(m[1] for m in merged)
    merged.sort(key=lambda m: -m[1])
    labs = np.array([m[0] for m in merged], np.float32).reshape(-1, 1, 3)
    rgbs = cv2.cvtColor(labs, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    pal = []
    for m, rgb in zip(merged, rgbs):
        pal.append({"hex": _hex(rgb), "weight": float(m[1] / total)})
    # fix rounding so weights sum to exactly 1
    if pal:
        s = sum(p["weight"] for p in pal)
        pal[0]["weight"] += 1.0 - s
    chroma = [float(np.hypot(m[0][1], m[0][2])) for m in merged]
    cand = [i for i, p in enumerate(pal) if p["weight"] >= 0.05] or [0]
    accent_i = max(cand, key=lambda i: chroma[i])
    return pal, pal[accent_i]["hex"]


# ----------------------------------------------------------------------------
# saliency (spectral residual + frequency-tuned colour contrast)
# ----------------------------------------------------------------------------


def _spectral_residual(gray: np.ndarray, out_hw) -> np.ndarray:
    g = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64)
    F = np.fft.fft2(g)
    log_amp = np.log(np.abs(F) + 1e-9)
    phase = np.angle(F)
    avg = cv2.blur(log_amp, (3, 3), borderType=cv2.BORDER_REPLICATE)
    sal = np.abs(np.fft.ifft2(np.exp((log_amp - avg) + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)
    sal = cv2.resize(sal, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return _norm01(sal)


def _saliency(img_small: np.ndarray):
    """Returns (map HxW 0..1 at <=128 px, mask uint8 of the main component, box, focus)."""
    s = _resize_long(img_small, _SAL_SIDE)
    h, w = s.shape[:2]
    gray = _luma(s)
    sr = _spectral_residual(gray, (h, w))
    lab = cv2.cvtColor(cv2.GaussianBlur(s, (5, 5), 0), cv2.COLOR_RGB2LAB)
    ft = np.linalg.norm(lab - lab.reshape(-1, 3).mean(0), axis=2)
    ft = _norm01(ft)
    sal = 0.5 * sr + 0.5 * ft
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    centre = np.exp(-(((xx / w - 0.5) / 0.45) ** 2 + ((yy / h - 0.5) / 0.45) ** 2))
    sal = _norm01(sal * (0.6 + 0.4 * centre))
    sal = cv2.GaussianBlur(sal, (0, 0), max(1.0, 0.02 * max(h, w)))
    sal = _norm01(sal)

    u8 = (sal * 255).astype(np.uint8)
    thr, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        box = [0.25, 0.25, 0.5, 0.5]
        main = np.zeros_like(mask)
    else:
        # largest component by saliency mass (area weighted by map)
        mass = ndimage.sum(sal, lbl, index=np.arange(1, n))
        j = int(np.argmax(mass)) + 1
        x, y, bw, bh = stats[j, :4]
        box = [x / w, y / h, bw / w, bh / h]
        main = (lbl == j).astype(np.uint8) * 255
    wts = np.where(sal >= thr / 255.0, sal, 0.0)
    if wts.sum() <= 1e-9:
        wts = sal + 1e-9
    fx = float((wts * (xx + 0.5)).sum() / wts.sum() / w)
    fy = float((wts * (yy + 0.5)).sum() / wts.sum() / h)
    focus = [float(np.clip(fx, 0, 1)), float(np.clip(fy, 0, 1))]
    return sal, main, [float(v) for v in box], focus


# ----------------------------------------------------------------------------
# empty regions
# ----------------------------------------------------------------------------


def _largest_rect(grid: np.ndarray):
    """Largest all-True axis-aligned rectangle in a bool grid -> (area, (c, r, w, h))."""
    gy, gx = grid.shape
    best_area, best = 0, None
    hgt = np.zeros(gx, np.int32)
    for r in range(gy):
        hgt = np.where(grid[r], hgt + 1, 0)
        hl = hgt.tolist() + [0]
        stack: list[tuple[int, int]] = []
        for c, cur in enumerate(hl):
            start = c
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (c - s)
                if area > best_area:
                    best_area, best = area, (s, r - hh + 1, c - s, hh)
                start = s
            stack.append((start, cur))
    return best_area, best


def _empty_regions(gray: np.ndarray, edges: np.ndarray, subject_mask: np.ndarray, grid: int = 32):
    h, w = gray.shape
    ys = np.linspace(0, h, grid + 1).round().astype(int)
    xs = np.linspace(0, w, grid + 1).round().astype(int)

    def cell_sums(a):
        S = cv2.integral(a.astype(np.float64))
        return S[ys[1:]][:, xs[1:]] - S[ys[:-1]][:, xs[1:]] - S[ys[1:]][:, xs[:-1]] + S[ys[:-1]][:, xs[:-1]]

    area = np.maximum(np.outer(np.diff(ys), np.diff(xs)), 1).astype(np.float64)
    ed = cell_sums(edges > 0) / area
    m = cell_sums(gray) / area
    var = np.maximum(cell_sums(gray.astype(np.float64) ** 2) / area - m ** 2, 0.0)
    sd = np.sqrt(var)
    subj = cv2.resize(subject_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sc = cell_sums(subj) / area

    out = []
    for e_thr, s_thr in ((0.03, 0.06), (0.08, 0.12)):
        free = (ed <= e_thr) & (sd <= s_thr) & (sc <= 0.3)
        rects = []
        for _ in range(16):
            a, r = _largest_rect(free)
            if r is None or a < 0.04 * grid * grid:
                break
            c0, r0, cw, ch = r
            free[r0:r0 + ch, c0:c0 + cw] = False
            if cw < 3 or ch < 3:
                continue
            rects.append((a / float(grid * grid), r))
            if len(rects) >= 8:
                break
        if rects and rects[0][0] >= 0.15 or e_thr > 0.05:
            if rects:
                big = [rc for rc in rects if rc[0] >= 0.15]
                use = big if big else rects
                for _, (c0, r0, cw, ch) in use[:4]:
                    x0, x1 = xs[c0] / w, xs[c0 + cw] / w
                    y0, y1 = ys[r0] / h, ys[r0 + ch] / h
                    out.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
            if out:
                break
    out.sort(key=lambda r: -r[2] * r[3])
    return out[:4]


# ----------------------------------------------------------------------------
# contours
# ----------------------------------------------------------------------------


def _border_fraction(cnt: np.ndarray, w: int, h: int, tol: float = 3.0) -> float:
    p = cnt.reshape(-1, 2).astype(np.float64)
    q = np.roll(p, -1, axis=0)
    seg = np.linalg.norm(q - p, axis=1)
    mid = 0.5 * (p + q)
    on = (mid[:, 0] <= tol) | (mid[:, 0] >= w - 1 - tol) | (mid[:, 1] <= tol) | (mid[:, 1] >= h - 1 - tol)
    tot = seg.sum()
    return float(seg[on].sum() / tot) if tot > 0 else 1.0


def _smooth_chain(cnt: np.ndarray, short_side: int) -> np.ndarray:
    """Gaussian-smooth a dense closed point chain (kills the pixel/texture sawtooth) -> float32 (N,1,2)."""
    p = cnt.reshape(-1, 2).astype(np.float64)
    n = len(p)
    sigma = min(max(2.0, 0.008 * short_side), n / 8.0)
    if n >= 8 and sigma >= 1.0:
        p = ndimage.gaussian_filter1d(p, sigma, axis=0, mode="wrap")
    return p.astype(np.float32).reshape(-1, 1, 2)


def _simplify(cnt: np.ndarray) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = 0.002 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    while len(approx) > 200:
        eps *= 1.6
        approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    while len(pts) < 8:  # subdivide edges, keeps corners
        nxt = np.roll(pts, -1, axis=0)
        mids = 0.5 * (pts + nxt)
        pts = np.stack([pts, mids], axis=1).reshape(-1, 2)
    return pts[:200]


def _subject_mask(img_small: np.ndarray, sal: np.ndarray) -> np.ndarray | None:
    """GrabCut seeded by the saliency map -> uint8 0/255 mask of the main subject at img_small size."""
    h, w = img_small.shape[:2]
    s = min(1.0, 200.0 / max(h, w))
    gw, gh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    im8 = (cv2.resize(img_small, (gw, gh), interpolation=cv2.INTER_AREA) * 255.0 + 0.5).astype(np.uint8)
    sr = cv2.resize(sal.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    thr, _ = cv2.threshold((sr * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(thr) / 255.0
    m = np.full((gh, gw), cv2.GC_PR_BGD, np.uint8)
    m[sr >= thr] = cv2.GC_PR_FGD
    m[sr < 0.5 * thr] = cv2.GC_BGD
    hi = sr >= thr + 0.6 * (1.0 - thr)
    if hi.sum() >= 4:
        m[hi] = cv2.GC_FGD
    n_fg = int(np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).sum())
    if n_fg < 16 or n_fg > 0.9 * m.size:
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(1234)
        cv2.grabCut(im8, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (lbl == j).astype(np.uint8) * 255
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)


def _edge_support(pts: np.ndarray, edge_d: np.ndarray, tol: float = 3.0) -> float:
    """Fraction of the outline (off the image border) that lies on a real image edge."""
    h, w = edge_d.shape
    p = pts.reshape(-1, 2)
    on = (p[:, 0] <= tol) | (p[:, 0] >= w - 1 - tol) | (p[:, 1] <= tol) | (p[:, 1] >= h - 1 - tol)
    p = p[~on]
    if len(p) == 0:
        return 0.0
    xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
    return float((edge_d[yi, xi] > 0).mean())


def _open_at_border(pts: np.ndarray, w: int, h: int, tol: float = 3.0):
    """Drop the longest run of points lying on the image border -> (pts, closed)."""
    on = (pts[:, 0] <= tol) | (pts[:, 0] >= w - 1 - tol) | (pts[:, 1] <= tol) | (pts[:, 1] >= h - 1 - tol)
    if not on.any() or on.all() or on.sum() < 2:
        return pts, True
    n = len(pts)
    start = int(np.flatnonzero(~on)[0])           # rotate so the chain starts off the border
    r = np.roll(on, -start)
    best, bi, i = 0, -1, 0
    while i < n:
        if r[i]:
            j = i
            while j < n and r[j]:
                j += 1
            if j - i > best:
                best, bi = j - i, i
            i = j
        else:
            i += 1
    if best < 2:
        return pts, True
    rp = np.roll(pts, -start, axis=0)
    # keep the two border points that anchor the run so the outline still reaches the edge
    keep = np.concatenate([rp[bi + best - 1:], rp[:bi + 1]])
    return keep, False


def _contours(img_small: np.ndarray, gray8: np.ndarray, edges: np.ndarray, image_id: str,
              sal: np.ndarray | None = None):
    h, w = gray8.shape
    img_area = float(w * h)
    blur = cv2.GaussianBlur(gray8, (5, 5), 0)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def polarity(mask):
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        return 255 - mask if (border > 0).mean() > 0.5 else mask

    # A: Otsu on luma
    _, ma = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ma = polarity(ma)
    ma = cv2.morphologyEx(cv2.morphologyEx(ma, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k5)
    # C: colour distance from the border (background) colour, Otsu
    lab = cv2.cvtColor(cv2.GaussianBlur(img_small, (5, 5), 0), cv2.COLOR_RGB2LAB)
    bpx = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(bpx, axis=0), axis=2)
    d8 = (np.clip(dist / max(float(dist.max()), 1e-6), 0, 1) * 255).astype(np.uint8)
    _, mc = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mc = cv2.morphologyEx(cv2.morphologyEx(mc, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k5)
    # B: Canny + morphological close, filled
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k7, iterations=2)

    sources = [(0, mc, cv2.RETR_EXTERNAL), (1, ma, cv2.RETR_CCOMP), (2, closed, cv2.RETR_EXTERNAL)]
    sal_w = None
    if sal is not None:
        sal_w = cv2.resize(sal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        subj = _subject_mask(img_small, sal)
        if subj is not None:
            _, subj = cv2.threshold(subj, 127, 255, cv2.THRESH_BINARY)
            sources.insert(0, (-1, subj, cv2.RETR_EXTERNAL))
    sal_mean = float(sal_w.mean()) if sal_w is not None else 0.0
    short = min(w, h)

    cands = []
    edge_d = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    for src, mask, mode in sources:
        cs, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) < 8:
                continue
            a = cv2.contourArea(c)
            if a < 0.003 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh > 0.9 * img_area:
                continue
            if _border_fraction(c, w, h) > (0.5 if src < 0 else 0.35):
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            solidity = a / max(hull, 1e-9)
            if src >= 0 and solidity < 0.45:          # spiky texture / background shapes, not objects
                continue
            sm = _smooth_chain(c, short)
            hull_p = cv2.arcLength(cv2.convexHull(sm), True)
            if cv2.arcLength(sm, True) > 1.35 * max(hull_p, 1e-6):
                continue                              # zig-zag outline (foliage, texture, skyline noise)
            salience = 1.0
            if sal_w is not None:
                fill = np.zeros((h, w), np.uint8)
                cv2.drawContours(fill, [c], -1, 255, -1)
                inside = fill > 0
                mi = mo = sal_mean
                if inside.any() and (~inside).any():
                    mi, mo = float(sal_w[inside].mean()), float(sal_w[~inside].mean())
                    salience = float(np.clip((mi - mo) / max(mi + mo, 1e-6) * 2.0, 0.0, 1.0))
                if src >= 0 and (salience < 0.25 or mi < sal_mean):
                    continue                          # background texture: low saliency inside
            support = _edge_support(sm, edge_d)
            cands.append((a, src, c, (x, y, bw, bh), salience, support))
    cands.sort(key=lambda t: (-t[0], t[1]))

    kept = []
    for a, src, c, bb, salience, support in cands:
        dup = False
        for a2, _, _, bb2, _, _ in kept:
            ix = max(0, min(bb[0] + bb[2], bb2[0] + bb2[2]) - max(bb[0], bb2[0]))
            iy = max(0, min(bb[1] + bb[3], bb2[1] + bb2[3]) - max(bb[1], bb2[1]))
            inter = ix * iy
            union = bb[2] * bb[3] + bb2[2] * bb2[3] - inter
            if union > 0 and inter / union > 0.8 and 0.7 < a / max(a2, 1e-9) < 1.43:
                dup = True
                break
        if not dup:
            kept.append((a, src, c, bb, salience, support))
        if len(kept) >= 12:
            break

    out = []
    for i, (a, _, c, (x, y, bw, bh), salience, support) in enumerate(kept):
        pts = _simplify(_smooth_chain(c, short))
        pts, is_closed = _open_at_border(pts, w, h)
        norm = np.stack([(pts[:, 0] + 0.5) / w, (pts[:, 1] + 0.5) / h], axis=1)
        norm = np.clip(norm, 0.0, 1.0)
        out.append({
            "id": f"{image_id}:c{i}",
            "points": [[round(float(px), 5), round(float(py), 5)] for px, py in norm],
            "closed": bool(is_closed),
            "salience": round(float(salience), 4),
            "edge_support": round(float(support), 4),
            "area": float(a / img_area),
            "bbox": [float(x / w), float(y / h), float(bw / w), float(bh / h)],
        })
    return out


# ----------------------------------------------------------------------------
# lines
# ----------------------------------------------------------------------------


def _lines(edges: np.ndarray, image_id: str):
    h, w = edges.shape
    L = max(h, w)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(30, int(L * 0.06)),
                           minLineLength=max(10, int(L * 0.08)), maxLineGap=int(L * 0.015) + 2)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lens)[:300]
    groups = []  # dict(p, d, tmin, tmax, support)
    tol_d = max(4.0, 0.012 * L)
    for i in order:
        x0, y0, x1, y1 = segs[i]
        ln = lens[i]
        if ln < 1:
            continue
        # frame-border lines (image frames / letterbox edges) are not content
        if (max(x0, x1) <= 3 or min(x0, x1) >= w - 4) and abs(x1 - x0) < 3:
            continue
        if (max(y0, y1) <= 3 or min(y0, y1) >= h - 4) and abs(y1 - y0) < 3:
            continue
        d = np.array([x1 - x0, y1 - y0]) / ln
        p0 = np.array([x0, y0])
        merged = False
        for g in groups:
            if abs(float(np.dot(d, g["d"]))) < math.cos(math.radians(4)):
                continue
            nrm = np.array([-g["d"][1], g["d"][0]])
            if abs(float(np.dot(p0 - g["p"], nrm))) > tol_d or abs(float(np.dot(np.array([x1, y1]) - g["p"], nrm))) > tol_d:
                continue
            t0 = float(np.dot(p0 - g["p"], g["d"]))
            t1 = float(np.dot(np.array([x1, y1]) - g["p"], g["d"]))
            lo, hi = min(t0, t1), max(t0, t1)
            if lo > g["tmax"] + 0.03 * L or hi < g["tmin"] - 0.03 * L:
                continue
            g["tmin"], g["tmax"] = min(g["tmin"], lo), max(g["tmax"], hi)
            g["support"] += ln
            merged = True
            break
        if not merged:
            groups.append({"p": p0, "d": d, "tmin": 0.0, "tmax": float(ln), "support": float(ln)})
    if not groups:
        return []
    for g in groups:
        g["score"] = min(g["support"], 1.5 * (g["tmax"] - g["tmin"])) + (g["tmax"] - g["tmin"])
    groups.sort(key=lambda g: -g["score"])
    top = groups[:8]
    smax = top[0]["score"]
    out = []
    for i, g in enumerate(top):
        a = g["p"] + g["d"] * g["tmin"]
        b = g["p"] + g["d"] * g["tmax"]
        out.append({
            "id": f"{image_id}:l{i}",
            "p0": [float(np.clip((a[0] + 0.5) / w, 0, 1)), float(np.clip((a[1] + 0.5) / h, 0, 1))],
            "p1": [float(np.clip((b[0] + 0.5) / w, 0, 1)), float(np.clip((b[1] + 0.5) / h, 0, 1))],
            "strength": float(g["score"] / smax),
        })
    return out


# ----------------------------------------------------------------------------
# analyze_image
# ----------------------------------------------------------------------------


def _auto_canny(gray8: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray8, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag, 90))
    hi = float(np.clip(hi, 40.0, 400.0))
    return cv2.Canny(blur, 0.4 * hi, hi, L2gradient=True)


def analyze_image(pixels: np.ndarray, image_id: str) -> dict:
    """Analyse one RGB image (HxWx3 float32 0..1; uint8 / RGBA also accepted)."""
    img = _to_rgb_float(pixels)
    H, W = img.shape[:2]
    small = _resize_long(img, _WORK_LONG_SIDE)
    gray = _luma(small)
    gray8 = (gray * 255.0 + 0.5).astype(np.uint8)

    palette, accent = _palette(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    luma_mean = float(gray.mean())
    contrast = float(np.clip(gray.std() * 2.0, 0.0, 1.0))
    saturation = float(np.clip(hsv[:, :, 1].mean(), 0.0, 1.0))

    sal, sal_mask, sal_box, focus = _saliency(small)
    edges = _auto_canny(gray8)
    empty = _empty_regions(gray, edges, sal_mask)
    contours = _contours(small, gray8, edges, image_id, sal)
    lines = _lines(edges, image_id)

    return {
        "id": image_id,
        "pixels": img,
        "w": int(W), "h": int(H),
        "palette": palette,
        "accent": accent,
        "luma_mean": luma_mean, "contrast": contrast, "saturation": saturation,
        "saliency_box": sal_box,
        "focus": focus,
        "empty_regions": empty,
        "contours": contours,
        "lines": lines,
        "meaning": None,
    }


# ----------------------------------------------------------------------------
# audio helpers
# ----------------------------------------------------------------------------

_AUDIO_SR = 22050
_HOP = 256
_NFFT = 1024
# STFT frames are centred; spectral flux peaks a little after a transient
# starts. Measured on synthetic click tracks (see tests): constant correction.
_ONSET_LAG = 0.006


def _mono(waveform) -> np.ndarray:
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 1:
        return np.nan_to_num(w)
    if w.shape[0] > 8 and w.shape[1] <= 8:  # (samples, channels)
        w = w.T
    return np.nan_to_num(w.mean(axis=0))


def _stereo(waveform) -> np.ndarray:
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 1:
        w = w[None, :]
    if w.shape[0] > 8 and w.shape[1] <= 8:
        w = w.T
    return w


def _resample(x: np.ndarray, sr: int, target: int = _AUDIO_SR):
    if sr <= target:
        return x.astype(np.float32), int(sr)
    fr = Fraction(int(target), int(sr)).limit_denominator(1000)
    y = signal.resample_poly(x, fr.numerator, fr.denominator)
    return y.astype(np.float32), int(round(sr * fr.numerator / fr.denominator))


def _onset_envelope(x: np.ndarray, sr: int):
    """Spectral-flux onset strength. Returns (env, frame_rate, hop, log_mag, mag_freqs)."""
    hop = _HOP if sr >= 16000 else max(64, int(round(sr * _HOP / _AUDIO_SR)))
    nfft = 4 * hop
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    f, _, Z = signal.stft(x, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop,
                          boundary="zeros", padded=True)
    mag = np.abs(Z).astype(np.float32)
    # log-spaced band reduction (~48 bands, 30 Hz .. nyquist)
    edges = np.unique(np.searchsorted(f, np.geomspace(30.0, sr / 2.0, 49)))
    edges = edges[edges < len(f)]
    if len(edges) < 3:
        bands = mag
    else:
        bands = np.add.reduceat(mag, edges, axis=0)
    S = np.log1p(100.0 * bands)
    flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)
    # frame 0 is compared against silence so a hit right at t = 0 still registers
    env = np.concatenate([[float(S[:, 0].sum())], flux]).astype(np.float64)
    # remove slow trend
    trend = ndimage.uniform_filter1d(env, size=max(3, int(round(sr / hop * 0.5))), mode="nearest")
    env = np.maximum(env - 0.5 * trend, 0.0)
    return env, sr / float(hop), hop, mag, f


def _estimate_period(env: np.ndarray, fr: float):
    """Tempo via autocorrelation, 70..180 bpm, octave preference 90..140."""
    e = env - env.mean()
    n = len(e)
    if n < 8 or not np.any(e):
        return 120.0, 60.0 * fr / 120.0
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    E = np.fft.rfft(e, nf)
    ac = np.fft.irfft(E * np.conj(E), nf)[:n]
    ac = ac / (ac[0] + 1e-12)
    lag_lo = max(1, int(np.floor(60.0 * fr / 180.0)))
    lag_hi = min(n - 2, int(np.ceil(60.0 * fr / 70.0)))
    if lag_hi <= lag_lo + 1:
        return 120.0, 60.0 * fr / 120.0

    def interp_peak(l):
        l = int(l)
        if 1 <= l < n - 1:
            a, b, c = ac[l - 1], ac[l], ac[l + 1]
            den = a - 2 * b + c
            if den < 0:
                return l + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)), float(b)
        return float(l), float(ac[l])

    def score_at(lag_f):
        l = int(round(lag_f))
        if l < 1 or l >= n - 1:
            return -1.0
        lo, hi = max(1, l - 1), min(n - 2, l + 1)
        return float(ac[lo:hi + 1].max())

    seg = ac[lag_lo:lag_hi + 1]
    best = lag_lo + int(np.argmax(seg))
    lag, sc = interp_peak(best)
    bpm = 60.0 * fr / lag
    if not (90.0 <= bpm <= 140.0):
        for mult in (2.0, 0.5):
            alt_bpm = bpm * mult
            if 87.0 <= alt_bpm <= 145.0:
                alt_lag = 60.0 * fr / alt_bpm
                l = int(round(alt_lag))
                lo, hi = max(1, l - 2), min(n - 2, l + 2)
                if hi > lo:
                    l = lo + int(np.argmax(ac[lo:hi + 1]))
                if score_at(l) >= 0.5 * sc:
                    lag, _ = interp_peak(l)
                    bpm = 60.0 * fr / lag
                    break
    return float(bpm), float(lag)


def _beat_track(env: np.ndarray, period: float, tightness: float = 100.0) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns fractional frame indices."""
    n = len(env)
    if n == 0 or period <= 1:
        return np.zeros(0)
    sd = env.std()
    oe = env / sd if sd > 0 else env.copy()
    half = int(round(period))
    tt = np.arange(-half, half + 1, dtype=np.float64)
    win = np.exp(-0.5 * (tt * 32.0 / period) ** 2)
    local = np.convolve(oe, win, mode="same")

    offs = -np.arange(int(round(period / 2.0)), int(round(2.0 * period)) + 1)
    txwt = -tightness * np.log(-offs / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, np.int64)
    first_ok = False
    for i in range(n):
        idx = i + offs
        valid = idx >= 0
        if valid.any():
            cand = cum[idx[valid]] + txwt[valid]
            j = int(np.argmax(cand))
            cum[i] = local[i] + cand[j]
            back[i] = idx[valid][j]
        else:
            cum[i] = local[i]
        first_ok = True
    if not first_ok:
        return np.zeros(0)
    # last beat: best cumulative score within the final beat period
    tail0 = max(0, n - int(round(period)) - 1)
    last = tail0 + int(np.argmax(cum[tail0:]))
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1], np.int64)
    # trim weak leading / trailing beats
    if len(beats) > 2:
        bl = local[beats]
        w = np.hanning(7)[1:-1]
        smooth = ndimage.convolve1d(bl, w / w.sum(), mode="nearest")
        thr = 0.5 * np.sqrt(np.mean(smooth ** 2))
        keep = np.flatnonzero(smooth > thr)
        if len(keep):
            beats = beats[keep[0]:keep[-1] + 1]
    # drop end beats that do not sit on an onset (DP boundary artefacts)
    if len(beats) > 2:
        med = float(np.median(local[beats]))
        while len(beats) > 2 and local[beats[-1]] < 0.3 * med:
            beats = beats[:-1]
        while len(beats) > 2 and local[beats[0]] < 0.3 * med:
            beats = beats[1:]
    # sub-frame refinement
    out = beats.astype(np.float64)
    for k, b in enumerate(beats):
        if 1 <= b < n - 1:
            a, c, d = local[b - 1], local[b], local[b + 1]
            den = a - 2 * c + d
            if den < 0:
                out[k] = b + float(np.clip(0.5 * (a - d) / den, -0.5, 0.5))
    return out


def _peak_pick(env: np.ndarray, fr: float) -> np.ndarray:
    if len(env) < 3 or env.max() <= 0:
        return np.zeros(0, np.int64)
    e = env / env.max()
    w_max = max(1, int(round(0.03 * fr)))
    w_avg = max(1, int(round(0.1 * fr)))
    is_max = e >= ndimage.maximum_filter1d(e, 2 * w_max + 1, mode="constant")
    avg = ndimage.uniform_filter1d(e, 2 * w_avg + 1, mode="constant")
    cand = np.flatnonzero(is_max & (e >= avg + 0.07) & (e > 0.05))
    out = []
    wait = max(1, int(round(0.05 * fr)))
    for c in cand:
        if not out or c - out[-1] >= wait:
            out.append(int(c))
    return np.array(out, np.int64)


def _sections(x: np.ndarray, sr: int, mag: np.ndarray, freqs: np.ndarray, fr: float,
              beat_t: np.ndarray, duration: float):
    """Novelty on a self-similarity matrix of log band energies; max 6 sections."""
    nfr = mag.shape[1]
    edges = np.unique(np.searchsorted(freqs, np.geomspace(40.0, sr / 2.0, 13)))
    edges = edges[edges < len(freqs)]
    bands = np.log1p(100.0 * np.add.reduceat(mag ** 2, edges, axis=0))  # (B, frames)
    rms = np.sqrt(np.mean(mag ** 2, axis=0) + 1e-12)
    # segment grid: beats if regular enough, else 0.5 s
    if len(beat_t) >= 8:
        bounds_t = np.concatenate([[0.0], beat_t[(beat_t > 0.05) & (beat_t < duration - 0.05)], [duration]])
    else:
        bounds_t = np.arange(0.0, duration + 0.5, 0.5)
        bounds_t[-1] = duration
    bounds_f = np.clip(np.round(bounds_t * fr).astype(int), 0, nfr)
    feats, segrms = [], []
    for a, b in zip(bounds_f[:-1], bounds_f[1:]):
        b = max(b, a + 1)
        feats.append(bands[:, a:b].mean(axis=1) if a < nfr else bands[:, -1])
        segrms.append(rms[a:b].mean() if a < nfr else 0.0)
    F = np.array(feats, np.float64)
    segrms = np.array(segrms)
    nseg = len(F)
    bounds = [0.0]
    if nseg >= 8:
        # absolute (not z-scored) similarity so a homogeneous track yields no boundaries
        lr = np.log(segrms + 1e-6)[:, None]
        Fa = np.hstack([F, 2.0 * lr])
        d = np.sqrt(np.mean((Fa[:, None, :] - Fa[None, :, :]) ** 2, axis=2))
        Ssm = np.exp(-(d / 0.75) ** 2)
        K = int(max(2, min(8, nseg // 4)))
        idx = np.arange(-K, K)
        sign = np.where(idx < 0, -1.0, 1.0)
        g = np.exp(-0.5 * ((idx + 0.5) / (0.5 * K)) ** 2)
        kern = np.outer(sign * g, sign * g)
        kern /= np.abs(kern).sum()
        nov = np.zeros(nseg)
        Sp = np.pad(Ssm, K, mode="constant")
        for i in range(nseg):
            nov[i] = (Sp[i:i + 2 * K, i:i + 2 * K] * kern).sum()
        nov[:K] = 0
        nov[nseg - K // 2:] = 0
        nov = np.maximum(nov, 0)
        if nov.max() > 0:
            thr = max(0.08, nov.mean() + 0.5 * nov.std())
            pk = [i for i in range(1, nseg - 1) if nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1] and nov[i] > thr]
            pk.sort(key=lambda i: -nov[i])
            min_gap = max(2.0, duration / 12.0)
            chosen = []
            for i in pk:
                t = float(bounds_t[i])
                if t < 1.0 or t > duration - 1.0:
                    continue
                if all(abs(t - c) >= min_gap for c in chosen):
                    chosen.append(t)
                if len(chosen) >= 5:
                    break
            bounds += sorted(chosen)
    # energies per section relative to the loudest
    t_all = np.concatenate([bounds, [duration]])
    energies = []
    for a, b in zip(t_all[:-1], t_all[1:]):
        fa, fb = int(a * fr), max(int(a * fr) + 1, int(b * fr))
        energies.append(float(np.sqrt(np.mean(rms[fa:fb] ** 2))) if fa < nfr else 0.0)
    emax = max(energies) if energies and max(energies) > 0 else 1.0
    return [{"t": float(round(t, 4)), "energy": float(e / emax)} for t, e in zip(bounds, energies)]


def analyze_audio(waveform: np.ndarray, sample_rate: int) -> dict:
    """Tempo, beats, downbeats, onsets and sections of a (channels, samples) waveform."""
    sample_rate = int(sample_rate)
    wav = _stereo(waveform)
    mono = _mono(wav)
    duration = len(mono) / float(sample_rate)
    x, sr = _resample(mono, sample_rate)
    env, fr, hop, mag, freqs = _onset_envelope(x, sr)
    bpm, period = _estimate_period(env, fr)
    bf = _beat_track(env, period)
    beat_t = np.clip(bf / fr + _ONSET_LAG, 0.0, max(duration, 0.0))
    if len(beat_t) and beat_t[0] < 0.02:
        beat_t[0] = 0.0  # a hit at the very start is a beat at t = 0
    if len(beat_t) >= 4:
        # refine bpm with a linear fit over the tracked beats
        k = np.arange(len(beat_t))
        slope = np.polyfit(k, beat_t, 1)[0]
        if slope > 0 and 70.0 <= 60.0 / slope <= 180.0 * 1.05:
            bpm = 60.0 / slope
    # downbeats: every 4th beat, phase by low-band (< 150 Hz) energy
    downbeats = []
    if len(beat_t):
        low = mag[freqs < 150.0].sum(axis=0) if np.any(freqs < 150.0) else mag[:2].sum(axis=0)
        bfi = np.clip(np.round(bf).astype(int), 0, len(low) - 1)
        w = max(1, int(round(0.05 * fr)))
        lowb = np.array([low[max(0, i - 1):i + w].max() for i in bfi])
        phase = int(np.argmax([lowb[p::4].mean() if len(lowb[p::4]) else -1 for p in range(4)]))
        downbeats = beat_t[phase::4]
    on = _peak_pick(env, fr)
    onsets = np.clip(on / fr + _ONSET_LAG, 0.0, duration)
    sections = _sections(x, sr, mag, freqs, fr, beat_t, duration)
    return {
        "waveform": wav,
        "sample_rate": sample_rate,
        "duration": float(duration),
        "bpm": float(round(bpm, 3)),
        "beats": [float(round(t, 4)) for t in beat_t],
        "downbeats": [float(round(t, 4)) for t in downbeats],
        "onsets": [float(round(t, 4)) for t in onsets],
        "sections": sections,
    }


def _robust_norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return a.astype(np.float32)
    p = float(np.percentile(a, 95))
    if p <= 1e-12:
        p = float(a.max())
    if p <= 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip(a / p, 0.0, 1.0).astype(np.float32)


def audio_envelope(waveform: np.ndarray, sample_rate: int, fps: float, n_frames: int) -> dict:
    """Per-video-frame rms/low/mid/high/onset, float32 length n_frames, normalised 0..1."""
    n_frames = int(n_frames)
    keys = ("rms", "low", "mid", "high", "onset")
    if n_frames <= 0:
        return {k: np.zeros(0, np.float32) for k in keys}
    sr = int(sample_rate)
    x = _mono(waveform).astype(np.float64)
    spf = sr / float(fps)
    L = int(round(spf))
    win_len = max(64, 2 * L)
    win_len += win_len % 2
    centres = np.round((np.arange(n_frames) + 0.5) * spf).astype(np.int64)
    pad = win_len
    xp = np.pad(x, (pad, pad + max(0, int(centres[-1]) + win_len - len(x))))
    idx = (centres[:, None] - win_len // 2 + pad) + np.arange(win_len)[None, :]
    seg = xp[idx]  # (n_frames, win_len)
    # rms over one frame's worth of samples (centre half of the window)
    core = seg[:, win_len // 2 - L // 2: win_len // 2 - L // 2 + max(1, L)]
    rms = np.sqrt(np.mean(core ** 2, axis=1))
    X = np.fft.rfft(seg * np.hanning(win_len)[None, :], axis=1)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(win_len, 1.0 / sr)
    low = np.sqrt(P[:, f < 200.0].sum(axis=1))
    mid = np.sqrt(P[:, (f >= 200.0) & (f < 2000.0)].sum(axis=1))
    high = np.sqrt(P[:, f >= 2000.0].sum(axis=1))
    edges = np.unique(np.searchsorted(f, np.geomspace(30.0, sr / 2.0, 25)))
    edges = edges[edges < len(f)]
    B = np.log1p(1000.0 * np.add.reduceat(np.abs(X), edges, axis=1))
    flux = np.maximum(0.0, np.diff(B, axis=0)).sum(axis=1)
    onset = np.concatenate([[0.0], flux])
    return {
        "rms": _robust_norm(rms),
        "low": _robust_norm(low),
        "mid": _robust_norm(mid),
        "high": _robust_norm(high),
        "onset": _robust_norm(onset),
    }


# ----------------------------------------------------------------------------
# text + mood
# ----------------------------------------------------------------------------

_MOODS = {
    #            energy valence
    "driving": (0.75, 0.55),
    "calm": (0.2, 0.6),
    "dark": (0.5, 0.15),
    "bright": (0.6, 0.85),
    "playful": (0.65, 0.8),
    "epic": (0.9, 0.5),
}

_LEXICON = {
    "driving": ["drive", "driving", "fast", "speed", "run", "race", "rush", "move", "power", "push",
                "go", "launch", "now", "urban", "city", "street", "beat", "pulse", "energy", "sport",
                "action", "motion", "boost", "turbo", "momentum"],
    "calm": ["calm", "relax", "peace", "peaceful", "quiet", "soft", "gentle", "slow", "breathe",
             "still", "serene", "zen", "rest", "sleep", "dream", "ocean", "nature", "mindful",
             "spa", "tranquil", "cozy", "warm", "simple"],
    "dark": ["dark", "night", "shadow", "fear", "horror", "mystery", "noir", "black", "death",
             "storm", "danger", "secret", "midnight", "haunted", "grim", "cold", "void", "doom",
             "villain", "crime"],
    "bright": ["bright", "sun", "sunny", "summer", "light", "fresh", "clean", "happy", "joy",
               "hope", "shine", "new", "morning", "spring", "smile", "glow", "clear", "pure",
               "golden", "vibrant"],
    "playful": ["fun", "play", "playful", "party", "kids", "game", "silly", "cute", "candy",
                "pop", "bounce", "toy", "colorful", "colourful", "crazy", "wow", "yay", "dance",
                "funky", "weird", "cartoon"],
    "epic": ["epic", "legend", "legendary", "hero", "glory", "rise", "battle", "war", "ultimate",
             "giant", "massive", "empire", "victory", "champion", "destiny", "power", "titan",
             "dragon", "kingdom", "unstoppable", "launch", "future"],
}

_WORD_TO_MOODS: dict[str, list[str]] = {}
for _m, _ws in _LEXICON.items():
    for _w in _ws:
        _WORD_TO_MOODS.setdefault(_w, []).append(_m)


def _split_phrases(line: str, max_words: int = 6) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.,;:!?—–])\s+|\s+[-—–]+\s+", line) if p.strip()]
    out = []
    for p in parts:
        words = p.split()
        if len(words) <= max_words:
            out.append(" ".join(words))
            continue
        n_chunks = int(math.ceil(len(words) / max_words))
        size = int(math.ceil(len(words) / n_chunks))
        for i in range(0, len(words), size):
            out.append(" ".join(words[i:i + size]))
    return out


def _text_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and len(lines[0].split()) > 6:
        return _split_phrases(lines[0])
    return lines


def _match_words(text: str):
    found, moods = [], []
    for tok in re.findall(r"[a-zA-Z']+", text.lower()):
        cands = [tok]
        if tok.endswith("s") and len(tok) > 3:
            cands.append(tok[:-1])
        if tok.endswith("ing") and len(tok) > 5:
            cands += [tok[:-3], tok[:-3] + "e"]
        if tok.endswith("ed") and len(tok) > 4:
            cands += [tok[:-2], tok[:-1]]
        for c in cands:
            if c in _WORD_TO_MOODS:
                if c not in found:
                    found.append(c)
                moods += _WORD_TO_MOODS[c]
                break
    return found, moods


def _mood(text: str, images: list[dict], audio: dict | None, audio_rms_db: float | None):
    words, moods = _match_words(text)
    e_parts, v_parts = [], []  # (value, weight)
    if moods:
        e = float(np.mean([_MOODS[m][0] for m in moods]))
        v = float(np.mean([_MOODS[m][1] for m in moods]))
        w = 2.0
        e_parts.append((e, w))
        v_parts.append((v, w))
    if images:
        sat = float(np.mean([im["saturation"] for im in images]))
        con = float(np.mean([im["contrast"] for im in images]))
        lum = float(np.mean([im["luma_mean"] for im in images]))
        e_parts.append((0.15 + 0.85 * (0.5 * sat + 0.5 * con), 1.0))
        v_parts.append((0.55 * lum + 0.45 * sat + 0.05, 1.0))
    if audio is not None:
        bpm_n = float(np.clip((audio["bpm"] - 70.0) / 110.0, 0, 1))
        rms_n = float(np.clip(((audio_rms_db if audio_rms_db is not None else -20.0) + 32.0) / 24.0, 0, 1))
        e_parts.append((0.6 * bpm_n + 0.4 * rms_n, 1.5))
    energy = sum(v * w for v, w in e_parts) / sum(w for _, w in e_parts) if e_parts else 0.5
    valence = sum(v * w for v, w in v_parts) / sum(w for _, w in v_parts) if v_parts else 0.5
    energy = float(np.clip(energy, 0, 1))
    valence = float(np.clip(valence, 0, 1))
    if moods:
        counts = {m: moods.count(m) for m in _MOODS}
        top = max(counts.values())
        tied = [m for m in _MOODS if counts[m] == top]
        label = min(tied, key=lambda m: (_MOODS[m][0] - energy) ** 2 + (_MOODS[m][1] - valence) ** 2)
    else:
        label = min(_MOODS, key=lambda m: (_MOODS[m][0] - energy) ** 2 + (_MOODS[m][1] - valence) ** 2)
    keywords = list(words)
    if label not in keywords:
        keywords.append(label)
    return {"energy": round(energy, 4), "valence": round(valence, 4), "keywords": keywords}


def analyze_inputs(text: str, images: list, audio: tuple | None, duration: float | None) -> dict:
    """Bundle text, image and audio analysis into the Interface 1 analysis dict."""
    text = (text or "").strip()
    flat = []
    for im in images or []:
        a = np.asarray(im)
        if a.ndim == 4:
            flat.extend(list(a))
        elif a is not None and a.size:
            flat.append(a)
    imgs = [analyze_image(a, f"img{i}") for i, a in enumerate(flat)]
    aud = None
    rms_db = None
    if audio is not None:
        wav, sr = audio
        if wav is not None and np.asarray(wav).size:
            aud = analyze_audio(wav, int(sr))
            m = _mono(wav)
            rms_db = float(20.0 * np.log10(np.sqrt(np.mean(m.astype(np.float64) ** 2)) + 1e-9))
    hint = float(duration) if duration is not None and float(duration) > 0 else None
    return {
        "text": text,
        "text_lines": _text_lines(text),
        "duration_hint": hint,
        "images": imgs,
        "audio": aud,
        "mood": _mood(text, imgs, aud, rms_db),
    }
