"""still_palette: the piece's LIMITED PALETTE discerned from its own imagery (user 2026-10-30: "... a limited palette discerned from the imagery"). Pure numpy, no network / ComfyUI.

    extract(img, k=5, seed=0)       -> dict(colors [3-5 rgb 0..255], dark, light, bg, fg, accents, pairs, source="still", ...) from one RGB image (the piece's first MOSS still: frame 0 of its first clip)
    contrast(a, b)                  -> WCAG contrast ratio of two rgb colours (1..21)
    legible_pairs(colors, min_ratio=4.5) -> [(ink, ground)] of the palette whose contrast is >= min_ratio (type is drawn ONLY in a legible pair)
    snap(rgb, colors)               -> the palette colour nearest to rgb (Lab distance)
    fold_palette(t_pal, pal, fixes) -> a RAW reply palette {bg, fg, accents} folded into the palette: every colour becomes one of the extracted ones, fg/bg a legible pair
    ground_for(pal, picture_mean)   -> the palette colour for a PLAIN ground behind letters that show footage (the best luminance contrast with the footage)

Method: k-means in CIE Lab on a downsampled copy (seeded k-means++ start, deterministic), clusters ordered by share; near-duplicates (Lab distance < MERGE_DE) are merged into the bigger one;
the darkest and the lightest cluster become `dark` and `light` and are pushed apart if the picture has no real dark / light (so type always has a legible pair, contrast >= MIN_RATIO); the
remaining colours (up to k - 2) are the accents, ranked by chroma x share. Fewer distinct colours than 3 are topped up from the dark / light by hue-preserving tints.
"""
from __future__ import annotations

import numpy as np

MIN_RATIO = 4.5               # WCAG AA for text: ink vs its ground
MERGE_DE = 14.0               # Lab distance under which two colours count as one
DARK_L, LIGHT_L = 14.0, 93.0  # where a too-mid darkest / lightest colour is pushed to
K_CLUSTERS = 9
MAX_SIDE = 96                 # the picture is shrunk to this long side before clustering


# ---------------------------------------------------------------------------------------------------------------------------------------------- colour maths
def _lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlin(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    """rgb 0..255 (..., 3) -> Lab (..., 3)."""
    xyz = (_lin(rgb) @ _M.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    return np.clip(np.rint(_unlin(xyz @ np.linalg.inv(_M).T)), 0, 255).astype(int)


def luminance(rgb) -> float:
    return float(_lin(np.asarray(rgb, float)) @ np.array([0.2126729, 0.7151522, 0.0721750]))


def contrast(a, b) -> float:
    """WCAG contrast ratio (1..21) of two rgb colours (0..255)."""
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def delta_e(a, b) -> float:
    return float(np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(b)))


def snap(rgb, colors):
    """the colour of `colors` nearest to rgb in Lab."""
    lab = rgb_to_lab(np.asarray(rgb, float))
    cl = rgb_to_lab(np.asarray(colors, float))
    return [int(v) for v in colors[int(np.argmin(np.linalg.norm(cl - lab, axis=-1)))]]


def legible_pairs(colors, min_ratio: float = MIN_RATIO) -> list:
    """[(ink, ground)] ordered pairs of the palette with a contrast ratio >= min_ratio (lightest-on-darkest first)."""
    cs = [list(map(int, c)) for c in colors]
    out = [(a, b) for a in cs for b in cs if a is not b and contrast(a, b) >= min_ratio]
    return sorted(out, key=lambda p: -contrast(*p))


# ---------------------------------------------------------------------------------------------------------------------------------------------- clustering
def _kmeans(X, w, k, rng, iters=14):
    """weighted k-means in Lab, k-means++ start from a seeded rng. -> (centres [k,3], share [k])."""
    n = len(X)
    k = min(k, n)
    idx = [int(rng.choice(n, p=w / w.sum()))]
    d2 = ((X - X[idx[0]]) ** 2).sum(-1)
    for _ in range(1, k):
        p = d2 * w
        if p.sum() <= 0:
            break
        idx.append(int(rng.choice(n, p=p / p.sum())))
        d2 = np.minimum(d2, ((X - X[idx[-1]]) ** 2).sum(-1))
    C = X[idx].copy()
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(len(C)):
            m = lab == j
            if m.any():
                C[j] = (X[m] * w[m, None]).sum(0) / w[m].sum()
    lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    share = np.array([w[lab == j].sum() for j in range(len(C))]) / w.sum()
    return C, share


def _chroma(lab):
    return float(np.hypot(lab[1], lab[2]))


def _push(lab, to_l, chroma_keep=0.7):
    """a Lab colour moved to lightness `to_l`, chroma scaled so the hue stays but the colour stays inside the gamut."""
    return np.array([to_l, lab[1] * chroma_keep, lab[2] * chroma_keep], float)


def extract(img, k: int = 5, seed: int = 0, min_ratio: float = MIN_RATIO) -> dict:
    """3-5 colours from `img` (uint8 / float 0..1 [H,W,3], or [N,H,W,3] = first frame). -> dict(colors, dark, light, bg, fg, accents, pairs, coverage, source, min_ratio).
    `bg` / `fg` = the legible type pair (dark ground + light ink when the picture is mostly dark, else the reverse); accents are the other colours."""
    a = np.asarray(img)
    if a.ndim == 4:
        a = a[0]
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a.astype(np.float32) * (255.0 if a.max() <= 1.5 else 1.0)), 0, 255).astype(np.uint8)
    a = a[..., :3]
    h, w = a.shape[:2]
    s = max(h, w) / float(MAX_SIDE)
    if s > 1:
        ys = np.linspace(0, h - 1, max(2, int(h / s))).astype(int)
        xs = np.linspace(0, w - 1, max(2, int(w / s))).astype(int)
        a = a[np.ix_(ys, xs)]
    px = a.reshape(-1, 3).astype(float)
    lab = rgb_to_lab(px)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    C, share = _kmeans(lab, np.ones(len(lab)), max(K_CLUSTERS, k + 2), rng)
    order = np.argsort(-share)
    C, share = C[order], share[order]
    keep: list = []                                            # merge near-duplicates into the bigger cluster
    for c, sh in zip(C, share):
        for j, (kc, ks) in enumerate(keep):
            if np.linalg.norm(c - kc) < MERGE_DE:
                keep[j] = ((kc * ks + c * sh) / (ks + sh), ks + sh)
                break
        else:
            keep.append((c, sh))
    cols = [np.array(c) for c, _ in keep]
    shares = [float(sh) for _, sh in keep]
    k_ = lambda: min(range(len(cols)), key=lambda j: cols[j][0])        # noqa: E731  darkest
    l_ = lambda: max(range(len(cols)), key=lambda j: cols[j][0])        # noqa: E731  lightest
    ids_d, ids_l = k_(), l_()
    dark, light = cols[ids_d].copy(), cols[ids_l].copy()
    if ids_d == ids_l or light[0] - dark[0] < 30:              # a flat picture: make a pair from the one colour
        base = cols[ids_d]
        dark, light = _push(base, DARK_L), _push(base, LIGHT_L, 0.45)
    else:
        if dark[0] > DARK_L + 8:
            dark = _push(dark, DARK_L)
        if light[0] < LIGHT_L - 10:
            light = _push(light, LIGHT_L, 0.55)
    rgb_d, rgb_l = lab_to_rgb(dark), lab_to_rgb(light)
    guard = 0
    while contrast(rgb_d, rgb_l) < min_ratio and guard < 12:   # push the pair apart until it reads
        dark, light = _push(dark, max(dark[0] - 4, 2.0)), _push(light, min(light[0] + 2, 98.0))
        rgb_d, rgb_l = lab_to_rgb(dark), lab_to_rgb(light)
        guard += 1
    rest = [(c, sh) for j, (c, sh) in enumerate(zip(cols, shares)) if j not in (ids_d, ids_l)]
    rest = sorted(rest, key=lambda cs: -(_chroma(cs[0]) + 12.0) * (cs[1] ** 0.5))
    acc: list = []
    for c, _sh in rest:
        rgb = lab_to_rgb(c)
        if all(delta_e(rgb, x) >= MERGE_DE for x in [rgb_d, rgb_l] + acc):
            acc.append(rgb)
        if len(acc) >= max(1, int(k) - 2):
            break
    base_lab = np.mean([dark, light], axis=0)
    tops = [(0.5, 1.0), (0.3, 0.85), (0.7, 0.9)]               # tints of the pair when the picture has fewer than 3 distinct colours
    for u, ch in tops:
        if len(acc) + 2 >= 3:
            break
        t = np.array([dark[0] * (1 - u) + light[0] * u, base_lab[1] * ch, base_lab[2] * ch])
        rgb = lab_to_rgb(t)
        if all(delta_e(rgb, x) >= MERGE_DE * 0.6 for x in [rgb_d, rgb_l] + acc):
            acc.append(rgb)
    colors = [[int(v) for v in rgb_d], [int(v) for v in rgb_l]] + [[int(v) for v in c] for c in acc]
    colors = colors[:max(3, min(5, int(k)))]
    mean_l = float(np.average(lab[:, 0]))
    dark_ground = mean_l < 55.0                               # mostly dark picture: pale ink on a dark ground (the house look), else dark ink on a pale ground
    bg, fg = (colors[0], colors[1]) if dark_ground else (colors[1], colors[0])
    accents = [c for c in colors[2:]]
    return dict(colors=colors, dark=colors[0], light=colors[1], bg=bg, fg=fg, accents=accents, pairs=[[list(a_), list(b_)] for a_, b_ in legible_pairs(colors, min_ratio)],
                mean_lightness=round(mean_l, 1), source="still", min_ratio=min_ratio, seed=int(seed))


def padded_accents(pal: dict, n: int = 3) -> list:
    """accents of the palette padded to n colours from the palette itself (the 3D modules read accents[2])."""
    acc = [list(a) for a in pal.get("accents") or []] or [list(pal["fg"])]
    pool = acc + [list(c) for c in pal["colors"] if list(c) not in (list(pal["bg"]),)]
    k = 0
    while len(acc) < n:
        acc.append(pool[k % len(pool)])
        k += 1
    return acc[:max(n, len(acc))]


def type_palette(pal: dict) -> dict:
    """the palette as the type engine's {bg, fg, accents} (>= 3 accents, all taken from the palette)."""
    return dict(bg=list(pal["bg"]), fg=list(pal["fg"]), accents=padded_accents(pal))


def fold_palette(t_pal: dict, pal: dict, fixes: list, where: str = "type.palette") -> int:
    """a RAW reply palette (in place) -> only the still's colours: fg / bg = the legible pair (the reply's own ground decides which of dark / light is the ground), accents = each reply
    accent snapped to a palette colour that is neither bg nor fg when one exists (duplicates dropped), else the palette's accents. -> 1 when changed."""
    colors = [list(c) for c in pal["colors"]]
    old = (t_pal.get("bg"), t_pal.get("fg"), t_pal.get("accents"))
    bg, fg = list(pal["bg"]), list(pal["fg"])
    rb = t_pal.get("bg")
    if isinstance(rb, list) and len(rb) == 3:                  # the reply wants a pale ground: keep that choice, it is the same pair swapped
        if luminance(rb) > 0.5 and luminance(bg) < 0.5:
            bg, fg = fg, bg
        elif luminance(rb) < 0.18 and luminance(bg) > 0.5:
            bg, fg = fg, bg
    if contrast(bg, fg) < pal.get("min_ratio", MIN_RATIO):
        bg, fg = list(pal["bg"]), list(pal["fg"])
    others = [c for c in colors if c not in (bg, fg)]
    acc: list = []
    for a in (t_pal.get("accents") or []):
        if isinstance(a, list) and len(a) == 3 and others:
            s_ = snap(a, others)
            if s_ not in acc:
                acc.append(s_)
    for a in others:
        if len(acc) >= 3:
            break
        if list(a) not in acc:
            acc.append(list(a))
    pool = [list(a) for a in acc] + [list(fg), list(bg)]                  # padding comes from the palette itself, never a blend: the type uses ONLY the extracted colours
    k = 0
    while len(acc) < 3:
        acc.append(pool[k % len(pool)])
        k += 1
    new = (bg, fg, acc)
    if new == old:
        return 0
    t_pal["bg"], t_pal["fg"], t_pal["accents"] = bg, fg, acc
    fixes.append(f"{where} -> the colours of the piece's own still ({len(colors)} colours, fg/bg contrast {contrast(bg, fg):.1f}:1); type, grounds and strands use only those")
    return 1


def ground_for(pal: dict, picture_mean) -> list:
    """the palette colour for a PLAIN ground behind letters that show a picture of mean colour `picture_mean` (0..255): the colour whose luminance differs most from it, so the picture
    inside the letters reads against the ground (flat-ground inverse type)."""
    cols = [list(c) for c in pal["colors"]]
    lp = luminance(picture_mean)
    return max(cols, key=lambda c: (abs(luminance(c) - lp), contrast(c, picture_mean)))
