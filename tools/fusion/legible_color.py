"""legible_color: choose the type colour from the MEASURED background (v005 round 3, 2026-10-27).

User review of the v005 previews: "a lot of the text was illegible. we want to measure the background colour to make sure the text's colour is
legible over the images - video". Before this the type stage picked word colours blind and compose only rescued holds afterwards (hold_net).

Compose op `legible_color(face, alpha, plate, ...)` (inserted by recipe_eval.legibility_pass in front of every op that draws the type, flat and
multiplane alike) does, per frame:
  1 MEASURE  the drawn word (alpha > 0.25, split into word clusters) and a ring around it on the PLATE, downsampled to LC_AN px on the backend (torch on the
             box, numpy locally); linear RGB -> WCAG relative luminance histogram (behind-letters pixels count double), mean linear colour (hue / chroma in OKLab),
             spread. Smoothed over time per tracked cluster (EMA, slower inside the word's hold) so a flickering plate does not flicker the type.
  2 CHOOSE   a fill that reaches the target WCAG contrast ratio (4.5 small type, 3.0 large type, by the word's cap height in px at a 1080 short side,
             aimed LC_AIM x higher) over >= LC_NEED of the measured pixels: the designed colour (the beauty pass colour) when it passes; else the same hue
             with the OKLab lightness shifted the least; else the complement of the background hue (lightness searched again); else near-white /
             near-black. When no single colour gets there (busy plate) the minimum SUPPORT is added in this order, picked by how much of the surround still
             fails: a thin outline, a soft shadow, a local scrim, all in the opposite luminance of the text. The colour is eased (OKLab, LC_TAU_COL) so a change
             of background or word never pops; support amounts ease the same way.
  3 APPLY    the beauty pass's shading is kept: the chosen colour is multiplied by the beauty luma relative to its mean over the letters (extrusion / bevel
             detail survives; highlights are compressed when the colour is already bright). Returns RGBA: rgb = recoloured face, a = the type alpha
             extended by the support (mode "layer", so the subject matte / depth occlusion treats the support like the type) or the plain alpha (mode
             "post": multiplane; the support is then painted under the type on the finished image by `legible_support`).
Opt out: `"color": "fixed"` on the piece (`compose.color`, `color`, `type.color`) or on a word (`type.words[i].color`): the face passes through unchanged.
Report: LC_LOG per frame -> `lc_report` per word (bg luminance, chosen colour, achieved ratio min / mean over the hold, cover, support) -> compose_report.json
(`shots.<sid>.legible_color`); `python -m tools.fusion.legible_color <compose_report.json>` prints the table.

All colour maths is on sRGB 0..1 frames as compose carries them (display referred); luminance is WCAG 2.x relative luminance of the linearised colour.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

from tools.fusion import compose as C

LC_AN = 192               # analysis long side (px): the whole measurement + choice runs on a frame this size
LC_NB = 48                # luminance histogram bins (uniform in sqrt(Y))
LC_TARGET_SMALL = 4.5     # WCAG AA, normal text
LC_TARGET_LARGE = 3.0     # WCAG AA, large text
LC_LARGE_CAP = 36.0       # cap height in px at a 1080 short side from which type counts as large (~ 50 px font)
LC_AIM = 1.10             # choose for target x this, so smoothing lag / shading do not drop it below the target
LC_NEED = 0.85            # share of the word + surround pixels that must reach the target
LC_OUTLINE_MAX = 0.30     # residual failing share up to which a thin outline is enough ...
LC_SHADOW_MAX = 0.55      # ... up to which a soft shadow is ... above it a local scrim
LC_FLOOR = 3.0            # 2026-10-30 outline rule: the thin outline is the LAST resort, only when even the lightness-extreme ink reaches less than this ratio ...
LC_OUTLINE_BELOW = 0.40   # ... over more than this share of the word's surround (and the piece's outline budget, ceil(n / 3) words, has room)
LC_TAU_STATS = 0.30       # s, background statistics EMA outside a hold
LC_TAU_STATS_HOLD = 0.65  # s, inside a hold (slower: the colour must sit still while the word reads)
LC_TAU_COL = 0.12         # s, easing of the chosen colour (OKLab)
LC_TAU_SUP = 0.15         # s, easing of the support amounts
LC_WARM = 12              # frames the compose loop runs before a chunk so the smoothing state matches a sequential run
LC_MIN_PX = 6             # a cluster needs this many letter pixels at analysis size
LC_JOIN = 0.045           # word clustering: letter cores dilated by this x long side are one word
LC_RING = 0.03            # ring around the word x long side
LC_BEHIND_W = 2.0         # weight of the pixels behind the letters against the ring
LC_RING_FLOOR = True      # 2026-11-01 contrast floor vs the ring around the letters (ring_floor); False = the old combined measure only (bench before/after)
LC_SUP_OPACITY = {"outline": 1.0, "shadow": 0.75, "scrim": 0.55}
LC_SUP_LIGHT, LC_SUP_DARK = 0.97, 0.03     # support colours (sRGB level)
LC_NEAR_WHITE, LC_NEAR_BLACK = 0.97, 0.04  # fallback text colours (sRGB level)

LC_LOG = {}               # state_key -> {piece frame: [cluster record, ...]}  (lc_report)
_STATE = {}
_SUPPORT = {}             # state_key -> dict(frame, sa, level): the support painted by legible_support (mode "post")
_OUTLINE_CLAIM = {}       # state_key -> {word index}: the words that took an ADAPTIVE outline (counts against the ceil(n / 3) outline cap)


# ================================================================================================== colour maths (numpy, any shape [...,3])
def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


def rel_lum(lin):
    """WCAG 2.x relative luminance of linear RGB [...,3]."""
    lin = np.asarray(lin, np.float32)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def contrast_ratio(y1, y2):
    """WCAG contrast ratio of two relative luminances (broadcasts), >= 1."""
    y1, y2 = np.asarray(y1, np.float32), np.asarray(y2, np.float32)
    return (np.maximum(y1, y2) + 0.05) / (np.minimum(y1, y2) + 0.05)


def lin_to_oklab(lin):
    lin = np.asarray(lin, np.float64)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)


def oklab_to_lin(lab):
    lab = np.asarray(lab, np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                     -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                     -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)


def oklab_gamut(L, a, b):
    """OKLab colours (broadcast arrays) -> linear RGB inside the sRGB gamut: chroma is reduced (hue and lightness kept) until it fits."""
    L = np.asarray(L, np.float64)
    a, b = np.broadcast_to(np.asarray(a, np.float64), L.shape), np.broadcast_to(np.asarray(b, np.float64), L.shape)
    lo, hi = np.zeros(L.shape), np.ones(L.shape)

    def fits(k):
        rgb = oklab_to_lin(np.stack([L, a * k, b * k], -1))
        return (rgb >= -1e-4).all(-1) & (rgb <= 1.0001).all(-1)
    ok = fits(hi)
    for _ in range(14):
        mid = (lo + hi) / 2
        f = fits(mid)
        lo = np.where(f, mid, lo)
        hi = np.where(f, hi, mid)
    k = np.where(ok, 1.0, lo)
    return np.clip(oklab_to_lin(np.stack([L, a * k, b * k], -1)), 0.0, 1.0).astype(np.float32)


def hex_of(enc):
    v = np.clip(np.asarray(enc, np.float32).reshape(-1)[:3] * 255.0 + 0.5, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(int(x) for x in v)


# ================================================================================================== choice (pure numpy, unit-testable)
def _centres():
    return ((np.arange(LC_NB, dtype=np.float32) + 0.5) / LC_NB) ** 2          # bin centres in Y (histogram is uniform in sqrt(Y))


def _hist(y, w):
    q = np.clip(np.sqrt(np.clip(y, 0.0, 1.0)) * LC_NB, 0, LC_NB - 1e-3).astype(np.int32)
    h = np.bincount(q.reshape(-1), weights=np.asarray(w, np.float64).reshape(-1), minlength=LC_NB).astype(np.float64)
    s = h.sum()
    return (h / s if s > 0 else h).astype(np.float32)


def target_for(cap_norm):
    """WCAG target ratio of a word from its cap height (px at a 1080 short side)."""
    return LC_TARGET_LARGE if float(cap_norm) >= LC_LARGE_CAP else LC_TARGET_SMALL


def cover(hist, y_text, target):
    """share of the measured background (luminance histogram) against which a text of luminance y_text reaches `target`."""
    return float(np.asarray(hist)[contrast_ratio(y_text, _centres()) >= target].sum())


def choose(hist, design_lin, bg_lin, target, need=LC_NEED, aim=LC_AIM, prev_lin=None, soft=False, palette=None):
    """-> dict(kind, lin, cover, support, resid). kind: design | lightness | hue | extreme | support. The text colour that reaches `target` x `aim`
    over >= `need` of the histogram, in the order: designed colour, a colour of `palette` (linear RGB list; the nearest passing one to the design), lightness shift (same hue), complement of the
    background hue, near-white / near-black; when none does, the lightness extreme of the design hue (kind 'extreme', NO support) unless even that reaches less than LC_FLOOR over more than
    LC_OUTLINE_BELOW of the surround: only then kind 'support', support = 'outline' (2026-10-30, user: "we dont need to outline them all": the outline is the last resort; the op also caps it at
    ceil(n / 3) words per piece, see _legible_color); 2026-10-30 (user: "no soft blur behind the type, just pure type"): the support is ALWAYS the thin outline unless `soft` (compose.soft_support, effect
    `soft_support`) allows the soft shadow / scrim by the residual failing share (<= LC_OUTLINE_MAX, <= LC_SHADOW_MAX, above). prev_lin: last choice of the track (hysteresis: kept while it still passes / is not clearly worse)."""
    ta = float(target) * float(aim)
    hist = np.asarray(hist, np.float32)
    design_lin = np.clip(np.asarray(design_lin, np.float32).reshape(3), 0.0, 1.0)
    Cn = _centres()
    ratio = lambda Y: (np.maximum(Y[:, None], Cn[None]) + 0.05) / (np.minimum(Y[:, None], Cn[None]) + 0.05)     # noqa: E731
    covs = lambda Ys: (ratio(np.atleast_1d(np.asarray(Ys, np.float32))) >= ta).astype(np.float32) @ hist           # noqa: E731
    yd = float(rel_lum(design_lin))
    cd = float(covs(yd)[0])
    prev_design = prev_lin is not None and float(np.abs(np.asarray(prev_lin, np.float32) - design_lin).max()) < 1e-3
    if cd >= need + (0.03 if (prev_lin is not None and not prev_design) else (-0.04 if prev_design else 0.0)):
        return dict(kind="design", lin=design_lin, cover=cd, support=None, resid=1.0 - cd)
    if prev_lin is not None and not prev_design:
        cp = float(covs(rel_lum(np.asarray(prev_lin, np.float32)))[0])
        if cp >= need - 0.04:
            return dict(kind="kept", lin=np.asarray(prev_lin, np.float32), cover=cp, support=None, resid=1.0 - cp)
    lab0 = lin_to_oklab(design_lin)
    if palette is not None and len(palette):
        pls = [np.clip(np.asarray(q, np.float32).reshape(3), 0.0, 1.0) for q in palette]
        cvp = covs([rel_lum(q) for q in pls])
        okp = [i for i in range(len(pls)) if cvp[i] >= need]
        if okp:
            i = min(okp, key=lambda k: float(np.abs(lin_to_oklab(pls[k]) - lab0).sum()))
            return dict(kind="palette", lin=pls[i], cover=float(cvp[i]), support=None, resid=1.0 - float(cvp[i]))
    Ls = np.linspace(0.0, 1.0, 101)
    fams = [("lightness", lab0[1], lab0[2])]
    lbg = lin_to_oklab(np.clip(np.asarray(bg_lin, np.float32).reshape(3), 0.0, 1.0))
    if math.hypot(lbg[1], lbg[2]) > 0.02:                                       # a hue to complement
        ch = min(max(math.hypot(lab0[1], lab0[2]), 0.06), 0.15)
        n = math.hypot(lbg[1], lbg[2])
        fams.append(("hue", -lbg[1] / n * ch, -lbg[2] / n * ch))
    best = (cd, "design", design_lin)
    for kind, a, b in fams:
        lin_c = oklab_gamut(Ls, a, b)
        cv = covs(rel_lum(lin_c))
        ok = cv >= need
        if ok.any():
            i = int(np.argmin(np.where(ok, np.abs(Ls - lab0[0]), 9.0)))
            return dict(kind=kind, lin=lin_c[i], cover=float(cv[i]), support=None, resid=1.0 - float(cv[i]))
        i = int(np.argmax(cv))
        if cv[i] > best[0]:
            best = (float(cv[i]), kind, lin_c[i])
    ext = [srgb_to_lin(np.full(3, LC_NEAR_WHITE, np.float32)), srgb_to_lin(np.full(3, LC_NEAR_BLACK, np.float32))]
    ce = covs([rel_lum(e) for e in ext])
    for e, c_ in zip(ext, ce):
        if c_ >= need:
            return dict(kind="extreme", lin=e, cover=float(c_), support=None, resid=1.0 - float(c_))
    j = int(np.argmax(ce))
    if ce[j] > best[0]:
        best = (float(ce[j]), "extreme", ext[j])
    bc = best[0]
    # nothing reaches the target over enough of the surround: the text goes to the lightness extreme that does best (design hue), support in the opposite
    light = oklab_gamut(np.array([0.93]), lab0[1], lab0[2])[0]
    dark = oklab_gamut(np.array([0.16]), lab0[1], lab0[2])[0]
    cl_, cd_ = float(covs(rel_lum(light))[0]), float(covs(rel_lum(dark))[0])
    if prev_lin is not None:                                                    # hysteresis between the two extremes
        pl = rel_lum(np.asarray(prev_lin, np.float32))
        if abs(cl_ - cd_) < 0.08:
            pick = light if pl > 0.18 else dark
        else:
            pick = light if cl_ > cd_ else dark
    else:
        pick = light if cl_ >= cd_ else dark
    resid = 1.0 - max(bc, cl_, cd_)
    pick_cov = cl_ if pick is light else cd_
    floor_cov = float(((ratio(np.atleast_1d(np.asarray(rel_lum(pick), np.float32))) >= LC_FLOOR).astype(np.float32) @ hist)[0])
    if (1.0 - floor_cov) <= LC_OUTLINE_BELOW and not (soft and resid > LC_OUTLINE_MAX):      # the extreme ink still reads (>= 3:1 over >= 60 % of the surround): no outline (the opt-in soft shadow / scrim keep their residual rule)
        return dict(kind="extreme", lin=pick, cover=pick_cov, support=None, resid=1.0 - pick_cov)
    sup = "outline" if (resid <= LC_OUTLINE_MAX or not soft) else ("shadow" if resid <= LC_SHADOW_MAX else "scrim")
    return dict(kind="support", lin=pick, cover=max(bc, cl_, cd_), support=sup, resid=resid)


# ================================================================================================== state
class _Track:
    def __init__(self, tid, cx, cy):
        self.tid, self.cx, self.cy = tid, cx, cy
        self.hist = None
        self.mean_lin = None
        self.lab = None            # eased text colour (OKLab)
        self.target_lin = None     # last choice (hysteresis)
        self.sup = {"outline": 0.0, "shadow": 0.0, "scrim": 0.0}
        self.lightsup = 0.0        # support colour: 1 = light (text is dark), 0 = dark
        self.age = 0
        self.ring = None           # luminance histogram of the ring AROUND the letters only (ring_floor)
        self.from_shot = False     # the last choice was a shot-palette ink (its hysteresis never keeps a non-shot colour)


class _St:
    def __init__(self):
        self.last = None
        self.tracks = []
        self.nid = 0


def _state(key, frame_idx, u):
    st = _STATE.get(key)
    cont = (st is not None and frame_idx is not None and st.last is not None and int(frame_idx) == st.last + 1 and not (u is not None and float(u) <= 0.0))
    if not cont:
        st = _STATE[key] = _St()
    st.last = int(frame_idx) if frame_idx is not None else None
    return st


def reset(key=None):
    """drop the smoothing state / log (tests; a new render of the same piece)."""
    for d in (_STATE, LC_LOG, _SUPPORT, _OUTLINE_CLAIM, _SHOT_PAL, _PIECE_INKS, _TONE):
        if key is None:
            d.clear()
        else:
            d.pop(key, None)


def _ema(old, new, k):
    return new if old is None else old + (new - old) * k


# ================================================================================================== measurement
def _clusters(a_s):
    """word clusters of the small alpha: [dict(core, cx, cy, n, cap)] (cx, cy normalised by the long side)."""
    hs, ws = a_s.shape
    core = (a_s > 0.25).astype(np.uint8)
    if int(core.sum()) < LC_MIN_PX:
        return [], core
    r = max(2, int(round(LC_JOIN * max(hs, ws))))
    joined = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab = cv2.connectedComponents(joined)
    out = []
    for k in range(1, n):
        m = (lab == k) & (core > 0)
        cnt = int(m.sum())
        if cnt < LC_MIN_PX:
            continue
        ys, xs = np.nonzero(m)
        nn, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        hts = [int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, nn) if stats[i, cv2.CC_STAT_AREA] >= 3]
        cap = float(np.median(hts)) if hts else float(ys.max() - ys.min() + 1)
        out.append(dict(core=m, cx=float(xs.mean()) / max(hs, ws), cy=float(ys.mean()) / max(hs, ws), n=cnt, cap=cap))
    out.sort(key=lambda c: -c["n"])
    return out, core


def _measure(cl, plate_lin, ylum):
    """luminance histogram + mean linear colour of the pixels behind the letters (double weight) and in the ring around them."""
    hs, ws = cl["core"].shape
    r = max(2, int(round(LC_RING * max(hs, ws))))
    near = cv2.dilate(cl["core"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    w = np.where(cl["core"], LC_BEHIND_W, 0.0) + np.where(near & ~cl["core"], 1.0, 0.0)
    m = w > 0
    wm = w[m]
    hist = _hist(ylum[m], wm)
    mean_lin = (plate_lin[m] * wm[:, None]).sum(0) / max(float(wm.sum()), 1e-6)
    q = np.sqrt(np.clip(ylum[m], 0.0, 1.0))
    spread = float(np.sqrt(max(float(np.average((q - np.average(q, weights=wm)) ** 2, weights=wm)), 0.0)))      # std of sqrt(Y): the plate's busyness
    return hist, mean_lin.astype(np.float32), spread


def _ring_hist(cl, ylum):
    """luminance histogram of the ring around the letters ONLY (the pixels the opaque ink sits against; the pixels behind the letters are hidden by it)."""
    hs, ws = cl["core"].shape
    r = max(2, int(round(LC_RING * max(hs, ws))))
    near = cv2.dilate(cl["core"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    m = near & ~cl["core"]
    return _hist(ylum[m], np.ones(int(m.sum()), np.float32)) if m.any() else None


def ring_floor(ch, ring, design_lin, bg_lin, target, soft=False, palette=None):
    """CONTRAST FLOOR vs the SURROUND (2026-11-01, QC R5 "ASC/END": near-black ink on a navy flat ground, because the measurement weighted the light
    window behind the letters double): when the chosen ink reaches LC_FLOOR over less than (1 - LC_OUTLINE_BELOW) of the ring around the letters, the
    ink is chosen again against the ring alone (same rules, outline last resort). -> the choice (marked ring_floor=True when replaced)."""
    if not LC_RING_FLOOR or ring is None or not np.asarray(ring).sum():
        return ch
    Cn = _centres()
    y = float(rel_lum(np.asarray(ch["lin"], np.float32)))
    share = float(np.asarray(ring, np.float32)[contrast_ratio(y, Cn) >= LC_FLOOR].sum())
    if share >= 1.0 - LC_OUTLINE_BELOW:
        return ch
    return dict(choose(ring, design_lin, bg_lin, target, soft=soft, palette=palette), ring_floor=True)


# ================================================================================================== shot palette inks (2026-11-01 colour review)
# User review of R1-R7: "more colours chosen from the shot colours themselves for the text fills ... the outlines not just being black but contrasting or
# complimentary colours to the fill colour"; "the same fill colors ... over the frames - colours picked from the imagery of each shot". The ink of a word is
# a HUE of ITS shot (dominant hues of the plate, weighted by chroma, so a small crimson crown counts), lightness searched so it still passes the contrast
# target over LC_NEED of the measured pixels AND the ring floor; consecutive shots avoid the previous shot's hue; the outline / support and the echo take a
# second, complementary hue of the shot. Legibility wins: no passing hue -> the old chain (design / palette / lightness / complement / near-white / black).
LC_SHOT_HUES = True       # False = the old design-first choice (bench before/after)
LC_SHOT_K = 5             # hues kept per shot
LC_SHOT_MIN_C = 0.03      # OKLab chroma from which a plate pixel counts as hued
LC_INK_C = (0.10, 0.20)   # chroma range of a shot ink (boosted from the plate's: characterful, not muddy)
LC_VARY_DEG = 40.0        # a hue within this of the previous shot's ink goes to the back of the order
LC_ECHO_ALPHA = 0.30      # the echo / wake behind the crisp front copy: this opacity, tinted with the support hue (0.4 still hazed the subject in R1)
LC_ECHO_AN = 384          # analysis long side of echo_face (echo copies drawn by the type pass)
ECHO_MODULES = ("fly_into", "depth_echo", "bold_thin_echo", "spectrum", "depth_tunnel", "outflow")      # word modules whose type pass draws a train of copies
_SHOT_PAL = {}            # state_key -> dict(hues=[..], avoid=deg | None, src='plate' | 'user')
_PIECE_INKS = {}          # piece key -> {shot start (s): ink hue (deg)}: the variety memory (shots compose in start order in one process)


def _hue_deg(lab):
    return float(math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360.0)


def hue_dist(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def shot_hues(plate_lin, k=LC_SHOT_K, valid=None):
    """dominant hues of a plate (linear RGB [...,3]) -> [dict(h deg, c, L, share, score)] best first. Hue histogram (10 deg bins) of the hued pixels weighted
    by chroma, circular smoothing, peaks >= 25 deg apart; score = share^0.3 x chroma^1.5 (a small saturated accent beats a large grey-blue wash). Deterministic.
    valid (2026-11-02): optional bool array, one per pixel (any shape that flattens to the plate's pixels): only those pixels count (the subject's own pixels are left out, so the
    hues are the SHOT's: its light, its set, not the person's skin)."""
    lab = lin_to_oklab(np.asarray(plate_lin, np.float32).reshape(-1, 3))
    c = np.hypot(lab[:, 1], lab[:, 2])
    m = c > LC_SHOT_MIN_C
    if valid is not None:
        vm = np.asarray(valid, bool).reshape(-1)
        m = m & vm
        lab, c = lab, c
        n_all = max(int(vm.sum()), 1)
    else:
        n_all = len(c)
    if int(m.sum()) < 8:
        return []
    h = (np.degrees(np.arctan2(lab[m, 2], lab[m, 1])) % 360.0)
    nb = 36
    hist = np.bincount((h // 10).astype(int) % nb, weights=c[m], minlength=nb)
    sm = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    out = []
    for b in np.argsort(-sm):
        if sm[b] <= 0 or len(out) >= k:
            break
        if not (sm[b] >= sm[(b - 1) % nb] and sm[b] >= sm[(b + 1) % nb]):
            continue
        hc = b * 10.0 + 5.0
        if any(hue_dist(hc, o["h"]) < 25.0 for o in out):
            continue
        sel = np.array([hue_dist(x, hc) <= 15.0 for x in h])
        if not sel.any():
            continue
        cs, ls = c[m][sel], lab[m, 0][sel]
        share = float(sel.sum()) / float(n_all)
        hh = float(np.degrees(np.arctan2(np.average(lab[m, 2][sel], weights=cs), np.average(lab[m, 1][sel], weights=cs))) % 360.0)
        out.append(dict(h=round(hh, 1), c=round(float(np.median(cs)), 4), L=round(float(np.median(ls)), 3), share=round(share, 4),
                        score=round(share ** 0.3 * float(np.median(cs)) ** 1.5, 6)))
    out.sort(key=lambda o: -o["score"])
    return out


def hues_of_colors(colors):
    """a USER palette ([[r, g, b] sRGB 0..1 or 0..255], in the user's order) -> the same hue records as shot_hues (greys keep a tiny chroma so their hue is the user's)."""
    out = []
    for q in colors or []:
        v = np.asarray(q, np.float32).reshape(-1)[:3]
        v = v / 255.0 if float(v.max()) > 1.0 else v
        lab = lin_to_oklab(srgb_to_lin(v))
        c = float(math.hypot(lab[1], lab[2]))
        out.append(dict(h=round(_hue_deg(lab), 1), c=round(c, 4), L=round(float(lab[0]), 3), share=1.0, score=1.0, user=True))
    return out


def order_hues(hues, avoid=None, vary=LC_VARY_DEG):
    """hues with the ones within `vary` deg of `avoid` (the previous shot's ink) moved to the back, order otherwise kept."""
    if avoid is None:
        return list(hues)
    return [h for h in hues if hue_dist(h["h"], avoid) >= vary] + [h for h in hues if hue_dist(h["h"], avoid) < vary]


def _hue_inks(hu, n=61):
    """candidate inks of one hue record over a lightness ramp: (Ls, lin [n,3], effective chroma [n])."""
    Ls = np.linspace(0.12, 0.97, n)
    if hu.get("user"):
        c = float(hu["c"])                                                      # the user's colour: its own chroma, never boosted
    else:
        c = float(np.clip(hu["c"] * 1.8, LC_INK_C[0], LC_INK_C[1]))
    a, b = c * math.cos(math.radians(hu["h"])), c * math.sin(math.radians(hu["h"]))
    lin = oklab_gamut(Ls, a, b)
    lab = lin_to_oklab(lin)
    return Ls, lin, np.hypot(lab[..., 1], lab[..., 2])


def ink_for_hue(hist, ring, hu, target, need=LC_NEED, aim=LC_AIM, min_c=0.0, reject=None):
    """the most saturated ink of hue record `hu` that reaches target x aim over >= need of `hist` and LC_FLOOR over >= (1 - LC_OUTLINE_BELOW) of `ring`
    -> (lin, cover) or None. Ties: the lightness nearest the hue's own in the plate (the colour still looks like the shot's).
    min_c (2026-11-02): inks whose effective chroma is under it do not count (near-white with a trace of the hue is not the shot's colour); reject(lin) -> True drops an ink (no peach)."""
    Ls, lin, ce = _hue_inks(hu)
    Y = rel_lum(lin)
    Cn = _centres()
    ta = float(target) * float(aim)
    cov = (contrast_ratio(Y[:, None], Cn[None]) >= ta).astype(np.float32) @ np.asarray(hist, np.float32)
    ok = cov >= need
    if min_c > 0.0:
        ok &= ce >= float(min_c)
    if reject is not None:
        ok &= np.asarray([not reject(l) for l in lin], bool)
    if ring is not None and np.asarray(ring).sum() > 0:
        rs = (contrast_ratio(Y[:, None], Cn[None]) >= LC_FLOOR).astype(np.float32) @ np.asarray(ring, np.float32)
        ok &= rs >= 1.0 - LC_OUTLINE_BELOW
    if not ok.any():
        return None
    score = np.where(ok, ce - 0.05 * np.abs(Ls - float(hu.get("L", 0.6))), -9.0)
    i = int(np.argmax(score))
    return lin[i], float(cov[i])


def choose_shot(hist, ring, hues, target, prev_lin=None, need=LC_NEED, aim=LC_AIM, min_c=0.0, reject=None):
    """the shot-palette ink: the previous choice while it still passes (hysteresis), else the first hue of `hues` (ordered) with a passing ink.
    -> choice dict (kind 'kept' | 'shot', hue) or None (no hue passes: the caller falls back to `choose`)."""
    Cn = _centres()
    ta = float(target) * float(aim)
    if prev_lin is not None:
        yp = float(rel_lum(np.asarray(prev_lin, np.float32)))
        cp = float(np.asarray(hist, np.float32)[contrast_ratio(yp, Cn) >= ta].sum())
        rp = 1.0 if ring is None or not np.asarray(ring).sum() else float(np.asarray(ring, np.float32)[contrast_ratio(yp, Cn) >= LC_FLOOR].sum())
        if cp >= need - 0.04 and rp >= 1.0 - LC_OUTLINE_BELOW:
            return dict(kind="kept", lin=np.asarray(prev_lin, np.float32), cover=cp, support=None, resid=1.0 - cp,
                        hue=_hue_deg(lin_to_oklab(np.asarray(prev_lin, np.float32))))
    for hu in hues or []:
        r = ink_for_hue(hist, ring, hu, target, need, aim, min_c=min_c, reject=reject)
        if r is not None:
            return dict(kind="shot", lin=r[0].astype(np.float32), cover=r[1], support=None, resid=1.0 - r[1], hue=float(hu["h"]))
    return None


# ------------------------------------------------------------------------------------------------ 2026-11-02 COLOUR FROM THE SHOT, NOT THE SKIN
# user review of an own-video overlay job (frame 406): "why do we keep getting that light peach fill colour? what happened to pulling colors from the shot?". Causes found in the job:
# (1) the hues were taken from the WHOLE frame, the person's skin / jacket included, so the first hue of 3 of 5 words was the skin / wood orange (OKLab 31 deg); (2) the contrast test was a
# luminosity histogram of a wide band across the frame (dark jacket + dark ceiling + white tee) that no single ink reaches 85 % of, so 4 of 5 words failed the shot-hue pick and fell back to
# (3) the video-wide k-means palette (a brown (98, 56, 24) = skin + leather + corner wood) with its lightness raised until it read: a light peach (220, 165, 128), (244, 187, 149).
# Now: hues from the shot's pixels WITHOUT the subject (`shot_hues(valid=)`), the subject's own hue(s) (or, without a matte, the generic skin band) are demoted to a last resort (`prefer_hues`),
# a recently used ink hue goes behind fresh ones (variety), and the contrast test is a ladder (`pick_ink`: AA 4.5 then 3.0 over 85 % then 70 % of the pixels the word sits on) so a mixed
# background still gets a shot-hue ink instead of the palette fallback.
SKIN_HUE = (20.0, 72.0)         # OKLab hue band of skin (dark brown to light peach) used when there is no subject matte to learn the subject's own hue from ...
SKIN_C = (0.025, 0.14)          # ... and this chroma band (a grey / a saturated accent is not skin)
LC_SUBJ_NEAR = 18.0             # a plate hue within this of the subject's own hue is the subject's colour (skin, leather), not the shot's
LC_USED_NEAR = 40.0             # a hue within this of one of the last inks goes behind the fresh ones
LC_INK_LADDER = ((4.5, 0.85), (3.0, 0.85), (4.5, 0.70), (3.0, 0.70), (3.0, 0.55))      # (WCAG target, share of the word's pixels that must reach it); the ink is always saturated, the outline backs the rest
LC_INK_MIN_C = 0.075            # a shot ink keeps at least this OKLab chroma (a near-white with a trace of the hue is the "light peach" of the review, not a colour of the shot)
PEACH = dict(L=0.70, hue=(20.0, 75.0), c=0.14)      # an ink lighter than L, in the skin hue band, with chroma under c: peach / cream / skin tone, never a pick unless nothing else is left


def is_peach_ink(lin) -> bool:
    lab = lin_to_oklab(np.clip(np.asarray(lin, np.float32).reshape(3), 0.0, 1.0))
    c = float(math.hypot(lab[1], lab[2]))
    return float(lab[0]) >= PEACH["L"] and PEACH["hue"][0] <= _hue_deg(lab) <= PEACH["hue"][1] and c < PEACH["c"]


def is_skin_hue(h, c=None):
    """True for a hue record (or a bare hue deg) in the generic skin band."""
    hh = float(h["h"]) if isinstance(h, dict) else float(h)
    cc = float(h.get("c", 0.08)) if isinstance(h, dict) else (0.08 if c is None else float(c))
    return SKIN_HUE[0] <= hh <= SKIN_HUE[1] and SKIN_C[0] <= cc <= SKIN_C[1]


def prefer_hues(hues, subject=None, used=(), near=LC_SUBJ_NEAR, vary=LC_USED_NEAR):
    """order the shot's hue records: (fresh, not the subject's) first in score order, then (used lately, not the subject's), the subject's own hue(s) last.
    subject = the subject's hue records (shot_hues of its pixels) or None (then the generic skin band stands for it). -> (ordered, subject_like, n_fresh)."""
    def subj_like(h):
        if subject:
            return any(hue_dist(h["h"], s["h"]) < near for s in subject)
        return is_skin_hue(h)
    keep = [h for h in hues or [] if not subj_like(h)]
    skin = [h for h in hues or [] if subj_like(h)]
    fresh = [h for h in keep if not any(hue_dist(h["h"], u) < vary for u in used or ())]
    stale = [h for h in keep if h not in fresh]
    return fresh + stale, skin, len(fresh)


def pick_ink(hist, hues, skin_hues=(), ring=None, boost=1.0, ladder=LC_INK_LADDER, n_fresh=0):
    """the shot-palette ink of one word over the luminosity histogram `hist` of the pixels the word sits on: the ladder rung by rung (strictest first), over `hues` (preferred order), the
    subject's own hues (`skin_hues`) only after every rung failed for the others. -> (choice dict | None, info dict(rung, target, need, skin))."""
    passes = ((("shot-fresh", list(hues or [])[:int(n_fresh)], LC_INK_MIN_C, is_peach_ink, 4, False),) if n_fresh else ()) + (       # a hue not used by the last words first (variety), rungs 0-3
              ("shot", list(hues or []), LC_INK_MIN_C, is_peach_ink, 4, False),       # a coloured, non-peach ink of the shot, rungs 0-3
              ("shot-pale", list(hues or []), 0.0, is_peach_ink, 5, False),          # a paler ink of the shot (never peach), every rung
              ("subject", list(skin_hues or []), LC_INK_MIN_C, is_peach_ink, 5, True),     # the subject's own hue only when nothing else is left
              ("last", list(hues or []) + list(skin_hues or []), 0.0, None, 5, True))
    for name, pool, min_c, rej, nr, skin in passes:
        if not pool:
            continue
        for r, (tgt, need) in enumerate(ladder[:nr]):
            ch = choose_shot(hist, ring, pool, tgt * boost, need=need, min_c=min_c, reject=rej)
            if ch is not None:
                return ch, dict(rung=r, target=tgt, need=need, skin=skin, pass_=name)
    return None, dict(rung=None, target=None, need=None, skin=False, pass_=None)


SUP_FAMILIES = ("complement", "palette", "dark", "light")
_SUP_ROUND = ("complement", "complement", "palette", "palette", "dark", "light")     # one round of 6 picks: exactly ONE near-black per round -> any 6 consecutive words hold <= 2 (1/3)
_SUP_ROUNDS = {}


def _sup_round(r):
    """the family order of pick round `r` (6 picks): a seeded shuffle of _SUP_ROUND with no family twice in a row, also across the round boundary. Pure + cached."""
    r = max(int(r), 0)
    if r in _SUP_ROUNDS:
        return _SUP_ROUNDS[r]
    import random
    prev = _sup_round(r - 1)[-1] if r > 0 else None
    rng = random.Random(0xC0107 + 7919 * r)
    out = None
    for _ in range(200):
        p = list(_SUP_ROUND)
        rng.shuffle(p)
        if p[0] != prev and all(p[i] != p[i + 1] for i in range(len(p) - 1)):
            out = tuple(p)
            break
    if out is None:
        out = ("palette", "complement", "dark", "complement", "palette", "light") if prev not in ("palette",) else ("complement", "palette", "dark", "complement", "palette", "light")
    _SUP_ROUNDS[r] = out
    return out


def support_family_at(pick):
    """the family the pick asks for first (see support_pick)."""
    p = max(int(pick), 0)
    return _sup_round(p // 6)[p % 6]


def support_pick(ink_lin, hues=(), light=False, min_ratio=3.0, pick=0):
    """2026-11-01 (user: "it doesnt always have to be complimentary ... mostly black though ... all the time no"): the outline colour of one word, its FAMILY chosen by
    the seeded `pick` (word index + a per-piece offset): complement of the ink / a contrasting hue of the shot's own palette (`hues`) / near-black / light (near-white or
    a pale tint of a shot hue). Every family must reach `min_ratio` against the ink; a family that cannot is skipped for the next (cyclic order, near-black last), and
    near-black / near-white (LC_SUP_DARK / LIGHT) only when none can. consecutive picks never ask for the same family. -> (linear RGB (3,), family name)."""
    ink_lin = np.clip(np.asarray(ink_lin, np.float32).reshape(3), 0.0, 1.0)
    yi = float(rel_lum(ink_lin))
    hi = _hue_deg(lin_to_oklab(ink_lin))
    own = [dict(h=h["h"], c=h["c"]) for h in hues or []]
    comp = dict(h=(hi + 180.0) % 360.0, c=0.12)
    far = sorted([o for o in own if hue_dist(o["h"], hi) >= 30.0], key=lambda o: -hue_dist(o["h"], hi))
    tint = far[0] if far else (own[0] if own else comp)

    def ok(lin):
        return float(contrast_ratio(yi, float(rel_lum(lin)))) >= min_ratio

    def ladder(cd):
        for L in ((0.95, 0.90, 0.85) if light else (0.42, 0.34, 0.26, 0.18)):
            c = float(np.clip(cd["c"] * 1.5, 0.06, 0.16 if not light else 0.08))
            lin = oklab_gamut(np.array([L]), c * math.cos(math.radians(cd["h"])), c * math.sin(math.radians(cd["h"])))[0]
            if ok(lin):
                return lin
        return None

    def tinted(L, c, cd):
        return oklab_gamut(np.array([L]), c * math.cos(math.radians(cd["h"])), c * math.sin(math.radians(cd["h"])))[0]

    def make(fam):
        if fam == "complement":
            return ladder(comp)
        if fam == "palette":
            return ladder(far[int(pick) % len(far)]) if far else None
        if fam == "dark":
            lin = tinted(0.14, 0.025, tint)
            return lin if ok(lin) else None
        if fam == "deep":                                                          # fall-through only: a deep saturated tint of the shot (not black), for a light ink whose "light" slot cannot reach the contrast
            lin = tinted(0.26, 0.09, tint)
            return lin if ok(lin) else None
        lin = tinted(0.965, 0.03, far[int(pick) % len(far)] if far else comp)
        return lin if ok(lin) else None

    first = support_family_at(pick)
    i0 = SUP_FAMILIES.index(first)
    order = [SUP_FAMILIES[(i0 + k) % 4] for k in range(4)]
    nb = ({support_family_at(int(pick) - 1)} if int(pick) > 0 else set()) | {support_family_at(int(pick) + 1)}      # a fall-through avoids the families the neighbouring words ask for
    order = [f for f in order if f != "dark" and f not in nb] + ["deep"] + [f for f in order if f != "dark" and f in nb] + [f for f in order if f == "dark"]      # near-black is the last resort of a fall-through
    if first == "dark":
        order = ["dark"] + [f for f in order if f != "dark"]
    for fam in order:
        lin = make(fam)
        if lin is not None:
            return lin, fam
    return srgb_to_lin(np.full(3, LC_SUP_LIGHT if light else LC_SUP_DARK, np.float32)), ("light" if light else "dark")


def support_color(ink_lin, hues=(), light=False, min_ratio=3.0, pick=None):
    """the outline / support colour for an ink: a hue of the shot (or the ink's complement) as FAR round the wheel from the ink as possible, at the
    opposite lightness (light=True: a light support behind dark ink), reaching min_ratio (then 4.5 preferred) against the ink. Near-white / near-black
    only when no hue does. -> linear RGB (3,).
    pick (int, 2026-11-01): the seeded family pick of support_pick (complement / shot-palette hue / near-black / light), so the outlines of a piece vary word to word;
    None = the old always-complement behaviour; a USER palette ignores pick (its own hues only)."""
    if pick is not None and not any(h.get("user") for h in hues or []):
        return support_pick(ink_lin, hues, light=light, min_ratio=min_ratio, pick=pick)[0]
    ink_lin = np.clip(np.asarray(ink_lin, np.float32).reshape(3), 0.0, 1.0)
    lab_i = lin_to_oklab(ink_lin)
    yi = float(rel_lum(ink_lin))
    hi = _hue_deg(lab_i)
    own = [dict(h=h["h"], c=h["c"]) for h in hues or []]
    user = any(h.get("user") for h in hues or [])
    cands = own if (user and own) else [dict(h=(hi + 180.0) % 360.0, c=0.12)] + own      # 2026-11-01 (R12b): the USER's palette never gets a complement outline (crimson + gold came out teal / navy): its own hues only
    best, bs = None, -1.0
    for L in ((0.95, 0.90, 0.85) if light else (0.42, 0.34, 0.26, 0.18)):
        for cd in cands:
            c = float(np.clip(cd["c"] * 1.5, 0.06, 0.16 if not light else 0.08))
            lin = oklab_gamut(np.array([L]), c * math.cos(math.radians(cd["h"])), c * math.sin(math.radians(cd["h"])))[0]
            r = float(contrast_ratio(yi, float(rel_lum(lin))))
            if r < min_ratio:
                continue
            s = hue_dist(cd["h"], hi) / 180.0 + (0.5 if r >= 4.5 else 0.0)
            if s > bs:
                best, bs = lin, s
        if best is not None:
            return best
    return srgb_to_lin(np.full(3, LC_SUP_LIGHT if light else LC_SUP_DARK, np.float32))


def echo_color(ink_lin, sup_lin, bg_lin):
    """the subdued echo tint: the support hue at a lightness a third of the way from the background to the ink (it reads as a tinted shadow of the word, never a
    pale haze over the picture), chroma 0.06-0.12 -> linear RGB (3,)."""
    li, ls, lb = (lin_to_oklab(np.clip(np.asarray(x, np.float32).reshape(3), 0.0, 1.0)) for x in (ink_lin, sup_lin, bg_lin))
    hs = math.atan2(ls[2], ls[1])
    c = min(max(math.hypot(ls[1], ls[2]), 0.06), 0.12)
    return oklab_gamut(np.array([lb[0] + 0.35 * (li[0] - lb[0])]), c * math.cos(hs), c * math.sin(hs))[0]


def _shot_pal(state_key, p_lin, piece_key=None, shot_t0=None, user=None):
    """the frozen hue list of a shot (first frame the op sees; a user palette wins) + the hue the previous shot of the piece used."""
    sp = _SHOT_PAL.get(state_key)
    if sp is None:
        hues = hues_of_colors(user) if user else shot_hues(p_lin)
        avoid = None
        mem = _PIECE_INKS.get(piece_key) if piece_key is not None else None
        if mem and shot_t0 is not None:
            prev = [t for t in mem if float(t) < float(shot_t0) - 1e-3]
            avoid = mem[max(prev)] if prev else None
        sp = _SHOT_PAL[state_key] = dict(hues=hues, avoid=avoid, src="user" if user else "plate")
    return sp


def _match(st, cls):
    """assign each cluster (largest first) the nearest unclaimed track within 0.35 of the long side, else a new one."""
    free = list(st.tracks)
    res = []
    for c in cls:
        best, bd = None, 0.35
        for tr in free:
            d = math.hypot(tr.cx - c["cx"], tr.cy - c["cy"])
            if d < bd:
                best, bd = tr, d
        if best is None:
            st.nid += 1
            best = _Track(st.nid, c["cx"], c["cy"])
            st.tracks.append(best)
        else:
            free.remove(best)
        best.cx, best.cy, best.age = c["cx"], c["cy"], 0
        res.append(best)
    for tr in free:
        tr.age += 1
    st.tracks = [tr for tr in st.tracks if tr.age <= 8]
    return res


def _words_state(words, t):
    """(hold weight 0..1, fixed bool) at piece time t from words = [[start, hit, release, end, fixed], ..]."""
    wt, fixed = 0.0, False
    for w in words or []:
        s, a, b, e = (float(x) for x in w[:4])
        wt = max(wt, C._leg_weight(float(t), a, b, 0.25))
        if len(w) > 4 and w[4] and s - 1e-3 <= t <= e + 1e-3:
            fixed = True
    return wt, fixed


# ================================================================================================== the op
def _in_windows(windows, t) -> bool:
    return any(float(a) - 1e-3 <= float(t) <= float(b) + 1e-3 for a, b in (windows or []))


def _word_at(words, t):
    """index of the word the type on screen at piece time t belongs to: the hold (hit .. release) that contains t, else the word whose life (start .. end) is nearest to t. None without words."""
    best, bd = None, 1e9
    for i, w in enumerate(words or []):
        s, a, b, e = (float(x) for x in w[:4])
        if a - 1e-3 <= t <= b + 1e-3:
            return i
        d = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
        d = d + 1e-3 * abs(t - a)
        if d < bd:
            best, bd = i, d
    return best


def outline_budget(words, windows):
    """(cap, drawn): cap = ceil(n / 3) words of the piece may carry an outline (n = len(words)); drawn = the word indices the random system's outline windows already cover (they count first)."""
    n = len(words or [])
    drawn = set()
    for i, w in enumerate(words or []):
        a, b = float(w[1]), float(w[2])
        if any(float(x0) < b + 1e-3 and float(x1) > a - 1e-3 for x0, x1 in (windows or [])):
            drawn.add(i)
    return -(-n // 3), drawn


def _legible_color(face, alpha, plate, t=0.0, frame_idx=None, u=None, fps=24.0, words=None, state_key="lc", mode="layer",
                   target=None, smooth=1.0, enable=True, outline=None, soft_support=False, ink_palette=None, shot_palette=False, user_palette=None,
                   piece_key=None, shot_t0=None, front=None, front_face=None, echo_alpha=LC_ECHO_ALPHA, boost=False, outline_w=1.0,
                   life=None, echo_face=False):
    """see the module doc. face HxWx3 (type colour, beauty pass), alpha HxW (the type alpha as it is drawn), plate HxWx3 (what the type is drawn over).
    mode 'layer': returns RGBA, a = alpha extended by the support; 'post': a = alpha, support kept for `legible_support`. target: force the WCAG ratio
    (default by cap height). smooth: multiplies the time constants (0 = no smoothing). words: [[start, hit, release, end, fixed], ..] piece seconds.
    outline: [[t0, t1], ..] piece seconds where the thin outline support is forced on (the random system's stand-out option, random_system.outline_windows): the same outline, eased in and out,
    that the op adds by itself where even the lightness-extreme ink cannot reach the contrast floor; not on a `fixed` word. 2026-10-30 (user: "we dont need to outline them all"): an outline appears ONLY on the
    words the `outline` windows draw, plus at most (ceil(n / 3) - drawn) words where the ink alone fails (last resort, see choose); n = len(words); a word past the cap keeps its extreme-colour ink without outline.
    ink_palette: [[r, g, b], ..] sRGB 0..1, the piece palette colours tried (nearest to the design first) before the lightness shift.
    2026-11-01 colour review: shot_palette=True: the ink is a hue of THIS shot's plate (shot_hues, frozen on the first frame; `user_palette` [[r, g, b], ..] = the user's
    colours instead), the first hue (previous shot's hue of `piece_key` at start < `shot_t0` sent to the back) whose ink passes the target and the ring floor; the
    support (outline / shadow / scrim) is the complementary hue of the shot at the opposite lightness (support_color), not black / white. front (+ front_face): the
    crisp, un-displaced type alpha (and its face) when `alpha` / `face` come from a flow_wake: the front copy is drawn opaque in the ink, the rest of the wake
    (displaced letters + echo trail) behind it at `echo_alpha`, tinted with echo_color. boost (user asked for legible text): target >= 4.5 x 1.2, outline always on.
    outline_w: multiplies the outline width, a number or one per word (weight variety, recipe_eval.outline_weights).
    life: {beats: [s], amp} the HOLD GLINT (user 2026-11-01 "a very subtle effect ... when it is at the point of being read"): inside a word's hold a hairline
    ring of the support colour breathes with the beat (hold_life.pulse_env), opacity amp x (0.35 .. 1); off when None.
    echo_face: the TYPE PASS itself drew echo copies (R1: module fly_into, the matte holds every copy at full alpha, only the beauty shows them dimmer): letter
    letter pixels at or below 0.6 x the face's 95th-percentile luma lying beyond 1.5 % of the short side from the bright copy are the echo copies; when they are > 25 %
    of the letters, the bright copy is the crisp front and the rest is drawn like a wake echo (echo_alpha, echo tint). Extrusion / bevel shading next to the face stays."""
    B = C.B
    a2 = C._gray(alpha)
    h, w = a2.shape[:2]
    face = C._fit(face, h, w)
    fps = float(fps) if fps else 24.0
    wt, fixed = _words_state(words, float(t))

    def pack(rgb, a_out):
        return B.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2], a_out], axis=-1)

    if not enable or fixed:
        _SUPPORT.pop(state_key, None)
        return pack(face, a2)
    plate = C._fit(plate, h, w)
    f = min(1.0, LC_AN / float(max(h, w)))
    hs, ws = max(2, int(round(h * f))), max(2, int(round(w * f)))
    a_s = np.asarray(B.tonumpy(B.resize(a2, hs, ws, "area")), np.float32)
    cls, core = _clusters(a_s)
    st = _state(state_key, frame_idx, u)
    if not cls:
        for tr in st.tracks:
            tr.age += 1
        st.tracks = [tr for tr in st.tracks if tr.age <= 8]
        _SUPPORT.pop(state_key, None)
        return pack(face, a2)
    p_s = np.asarray(B.tonumpy(B.resize(plate, hs, ws, "area")), np.float32)[..., :3]
    f_s = np.asarray(B.tonumpy(B.resize(face, hs, ws, "area")), np.float32)[..., :3]
    p_lin = srgb_to_lin(p_s)
    ylum = rel_lum(p_lin)
    dt = 1.0 / fps
    sm = max(float(smooth), 0.0)
    tau_s = (LC_TAU_STATS + (LC_TAU_STATS_HOLD - LC_TAU_STATS) * wt) * sm
    k_s = 1.0 if tau_s < 1e-6 else 1.0 - math.exp(-dt / tau_s)
    k_c = 1.0 if sm < 1e-6 else 1.0 - math.exp(-dt / (LC_TAU_COL * sm))
    k_u = 1.0 if sm < 1e-6 else 1.0 - math.exp(-dt / (LC_TAU_SUP * sm))
    tracks = _match(st, cls)
    pal_lin = [srgb_to_lin(np.asarray(q, np.float32).reshape(3)) for q in ink_palette] if ink_palette else None
    sp = _shot_pal(state_key, p_lin, piece_key, shot_t0, user_palette) if ((shot_palette and LC_SHOT_HUES) or user_palette) else None
    hues = order_hues(sp["hues"], sp["avoid"]) if sp else []
    cap, drawn = outline_budget(words, outline)
    claim = _OUTLINE_CLAIM.setdefault(state_key, set())
    widx = _word_at(words, float(t))
    sup_pick = None if widx is None else zlib.crc32(str(piece_key if piece_key is not None else str(state_key).split(":")[0]).encode()) % 97 + int(widx)      # outline family varies word to word (support_pick)
    cap_full = float(np.median([c["cap"] for c in cls])) / f
    cap_norm = cap_full * 1080.0 / float(min(h, w))
    tgt = float(target) if target else target_for(cap_norm)
    if boost:
        tgt = max(tgt, LC_TARGET_SMALL) * 1.2
    recs, per = [], []
    for c, tr in zip(cls, tracks):
        hist, mean_lin, spread = _measure(c, p_lin, ylum)
        fresh = tr.hist is None
        tr.hist = hist if fresh else _ema(tr.hist, hist, k_s)
        tr.mean_lin = mean_lin if fresh else _ema(tr.mean_lin, mean_lin, k_s)
        wcore = a_s[c["core"]]
        design = (srgb_to_lin(f_s[c["core"]]) * wcore[:, None]).sum(0) / max(float(wcore.sum()), 1e-6)
        rh = _ring_hist(c, ylum)
        if rh is not None:
            tr.ring = rh if (fresh or tr.ring is None) else _ema(tr.ring, rh, k_s)
        sh_hist = tr.ring if (tr.ring is not None and np.asarray(tr.ring).sum() > 0) else tr.hist     # an opaque shot ink is read against the RING around the letters (ring_floor)
        ch = choose_shot(sh_hist, tr.ring, hues, tgt, prev_lin=tr.target_lin if tr.from_shot else None) if (sp and hues) else None      # hysteresis only on a shot ink
        if ch is None:
            ch = choose(tr.hist, design, tr.mean_lin, tgt, prev_lin=tr.target_lin, soft=bool(soft_support), palette=pal_lin)
            if sp and hues and ch["kind"] in ("extreme", "support"):          # busy plate: nothing reaches the full share; a shot hue as legible as the extreme ink wins
                alt = choose_shot(sh_hist, tr.ring, hues, tgt, need=max(float(ch["cover"]) - 0.02, 0.0))
                if alt is not None and alt["cover"] >= float(ch["cover"]) - 0.02:
                    ch = dict(alt, support=ch["support"], resid=ch["resid"], near=True)
        tr.from_shot = ch.get("hue") is not None
        if ch.get("hue") is not None and piece_key is not None and shot_t0 is not None:
            _PIECE_INKS.setdefault(piece_key, {})[round(float(shot_t0), 3)] = float(ch["hue"])     # variety memory for the next shot
        ch = ring_floor(ch, tr.ring, design, tr.mean_lin, tgt, soft=bool(soft_support), palette=pal_lin)
        if ch["support"] == "outline" and widx is not None and (widx in drawn or widx in claim):
            pass                                                                # the word is already one of the outlined ones
        elif ch["support"] == "outline" and widx is not None and len(drawn | claim) < cap:
            claim.add(widx)                                                     # takes one of the ceil(n / 3) outline slots
        elif ch["support"] == "outline":
            ch = dict(ch, kind="extreme", support=None)                         # cap full / no word: the extreme ink stands alone, no outline
        tr.target_lin = ch["lin"]
        tlab = lin_to_oklab(ch["lin"])
        tr.lab = tlab if (fresh or tr.lab is None) else tr.lab + (tlab - tr.lab) * k_c
        lin_now = oklab_gamut(tr.lab[0], tr.lab[1], tr.lab[2])
        text_light = float(rel_lum(lin_now)) > 0.18
        want = {k: (LC_SUP_OPACITY[k] if ch["support"] == k else 0.0) for k in LC_SUP_OPACITY}
        forced = _in_windows(outline, t) or bool(boost)
        if forced:
            want["outline"] = LC_SUP_OPACITY["outline"]
        for k in want:
            tr.sup[k] = want[k] if fresh else tr.sup[k] + (want[k] - tr.sup[k]) * k_u
        tr.lightsup = ((0.0 if text_light else 1.0) if (fresh or ch["support"] or forced) else tr.lightsup)
        enc = lin_to_srgb(lin_now)
        sup_lin = (support_color(lin_now, hues, light=not text_light, pick=sup_pick) if sp else
                   srgb_to_lin(np.full(3, LC_SUP_DARK if text_light else LC_SUP_LIGHT, np.float32)))
        lm = float((C._luma(f_s)[c["core"]] * wcore).sum() / max(float(wcore.sum()), 1e-6))
        keep = 1.0 - float(np.clip((np.abs(lin_to_oklab(design) - tr.lab).sum() - 0.01) / 0.05, 0.0, 1.0))      # = 1: the colour IS the design: pass the face through
        per.append(dict(c=c, tr=tr, enc=enc, lm=lm, keep=keep, hist=hist, ch=ch, spread=spread, bg_mean_lin=mean_lin, design=design,
                        sup=lin_to_srgb(sup_lin), echo=lin_to_srgb(echo_color(lin_now, sup_lin, mean_lin))))
    amt = [float(max(p["tr"].sup[k] for p in per)) for k in ("outline", "shadow", "scrim")]
    life_amt = 0.0
    if life and wt > 0.0:
        from tools.fusion import hold_life as _HL                              # pure envelope maths (read only)
        env = float(_HL.pulse_env(float(t), [float(b) for b in (life.get("beats") or [])]))
        life_amt = float(wt) * float(life.get("amp", 0.4)) * (0.35 + 0.65 * env)
    ow = outline_w
    if isinstance(ow, (list, tuple)):
        ow = ow[widx] if (widx is not None and widx < len(ow)) else 1.0
    ow = float(ow)
    a_out, o_s = a2, None
    if echo_face:
        fe = min(1.0, LC_ECHO_AN / float(max(h, w)))                         # finer than LC_AN: thin echo rings must not merge with the word
        he, we = max(2, int(round(h * fe))), max(2, int(round(w * fe)))
        src_a = a2 if front is None else C._clip(C._fit(C._gray(front), h, w))
        src_f = face if front_face is None else C._fit(front_face, h, w)
        a_e = np.asarray(B.tonumpy(B.resize(src_a, he, we, "area")), np.float32)
        lum_e = C._luma(np.asarray(B.tonumpy(B.resize(src_f, he, we, "area")), np.float32)[..., :3])
        let = a_e > 0.5
        if int(let.sum()) >= LC_MIN_PX:
            q = float(np.percentile(lum_e[let], 95))
            bright = (let & (lum_e > 0.6 * q)).astype(np.uint8)
            rr = max(1, int(round(0.015 * min(he, we))))
            near = (cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) > 0) | ~let
            if float((let & ~near).sum()) > 0.25 * float(let.sum()):
                keep = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 0.7)
                front = src_a * C._clip(B.resize(B.asarray(keep), h, w, "linear"))
                front_face = src_f
    if front is not None:
        fa = C._clip(C._fit(C._gray(front), h, w))
        ff_ = C._fit(front_face, h, w) if front_face is not None else face
        wake = C._clip(a2 - fa)                                                 # displaced letters + echo trail, outside the crisp copy
        face = C._lerp(face, ff_, C._c3(fa))                                    # shading of the front copy from the un-displaced face
        a2 = fa + (1.0 - fa) * wake * float(echo_alpha)
        a_out = a2
    if front is None and all(p["keep"] > 0.999 for p in per) and max(amt) <= 0.004 and life_amt <= 0.004:        # the designed colour passes and no support: the face goes through untouched (no full-res work)
        rgb, o_s = face, f_s
        _SUPPORT.pop(state_key, None)
    else:
        # ---- spatial fields (small): nearest cluster's colour / shading reference / support amounts
        n = len(per)
        if n == 1:
            idx = np.zeros((hs, ws), np.int32)
        else:
            seeds = np.full((hs, ws), 255, np.uint8)
            for c in cls:
                seeds[c["core"]] = 0
            _, lbl = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
            look = np.zeros(int(lbl.max()) + 1, np.int32)
            for i, c in enumerate(cls):
                look[np.unique(lbl[c["core"]])] = i
            idx = look[lbl]
        colf = np.stack([p["enc"] for p in per])[idx]
        lmf = np.array([p["lm"] for p in per], np.float32)[idx]
        keepf = np.array([p["keep"] for p in per], np.float32)[idx]
        supf = np.stack([[p["tr"].sup["outline"], p["tr"].sup["shadow"], p["tr"].sup["scrim"]] for p in per]).astype(np.float32)[idx]
        lvf = np.stack([p["sup"] for p in per]).astype(np.float32)[idx]
        ecf = np.stack([p["echo"] for p in per]).astype(np.float32)[idx]
        field = np.concatenate([colf, lmf[..., None], keepf[..., None], supf, lvf, ecf], -1).astype(np.float32)
        if n > 1:
            field = cv2.GaussianBlur(field, (0, 0), 1.5)
        ff = B.resize(B.asarray(field), h, w, "linear")
        col, lmean, keep, sup, lev, ecol = ff[..., 0:3], ff[..., 3], ff[..., 4], ff[..., 5:8], ff[..., 8:11], ff[..., 11:14]
        # ---- recolour keeping the beauty's shading
        lum = C._luma(face)
        r = (lum + 0.08) / (lmean + 0.08)
        cmax = B.maximum(B.maximum(col[..., 0], col[..., 1]), col[..., 2])
        hk = C._clip((1.0 / B.maximum(cmax, 1e-3) - 1.0) / 0.6)
        r = B.where(r > 1.0, 1.0 + (r - 1.0) * hk, r)
        r = C._clip(r, 0.0, 2.5)
        rgb = C._clip(col * C._c3(r))
        rgb = C._lerp(rgb, face, C._c3(keep))
        if front is not None:                                                   # the wake behind the front copy: the subdued echo tint, not the ink
            rgb = C._lerp(ecol, rgb, C._c3(C._clip(fa / B.maximum(a2, 1e-4))))
        # ---- support
        if max(amt) > 0.004 or life_amt > 0.004:
            ab = C._clip(a2)
            foot = []
            if life_amt > 0.004:
                foot.append(B.blur(C._dilate(ab, int(np.clip(round(0.016 * cap_full * ow), 1, 4))), 0.6) * life_amt)      # 2026-11-01 finer strokes: glint ring 0.016 cap (was 0.025, max 6 px)
            if amt[0] > 0.004:
                foot.append(sup[..., 0] * B.blur(C._dilate(ab, int(np.clip(round(0.028 * cap_full * ow), 1, 9))), 0.8))      # 2026-11-01 finer strokes: support outline 0.028 cap (was 0.045, 2-14 px)
            if amt[1] > 0.004:
                dx, dy = 0.02 * cap_full, 0.05 * cap_full
                sh = B.warp(ab, B.asarray(np.full((h, w), -dx, np.float32)), B.asarray(np.full((h, w), -dy, np.float32)))
                foot.append(sup[..., 1] * C._clip(B.blur(sh, max(1.0, 0.12 * cap_full)) * 1.6))
            if amt[2] > 0.004:
                foot.append(sup[..., 2] * C._clip(B.blur(C._dilate(ab, int(np.clip(round(0.2 * cap_full), 2, 12))), max(1.0, 0.3 * cap_full)) * 1.3))
            sa = foot[0]
            for x in foot[1:]:
                sa = B.maximum(sa, x)
            if mode == "layer":
                a_tot = ab + (1.0 - ab) * sa
                mixc = (rgb * C._c3(ab) + lev * C._c3((1.0 - ab) * sa)) / C._c3(B.maximum(a_tot, 1e-4))
                rgb = C._lerp(rgb, mixc, C._c3(C._clip(sa * 4.0)))
                a_out = a_tot
                _SUPPORT.pop(state_key, None)
            else:
                _SUPPORT[state_key] = dict(frame=None if frame_idx is None else int(frame_idx), sa=sa, lev=lev)
        else:
            _SUPPORT.pop(state_key, None)
    # ---- log (achieved contrast measured on the output, small)
    if o_s is None:
        o_s = np.asarray(B.tonumpy(B.resize(rgb, hs, ws, "area")), np.float32)
    lg = []
    for p in per:
        c, tr, ch = p["c"], p["tr"], p["ch"]
        wc = a_s[c["core"]]
        y_col = float(rel_lum(srgb_to_lin(o_s[c["core"]])).dot(wc) / max(float(wc.sum()), 1e-6))
        y_bg = float(rel_lum(p["bg_mean_lin"]))
        lg.append(dict(y_bg=round(y_bg, 4), y_col=round(y_col, 4), ratio=round(float(contrast_ratio(y_col, y_bg)), 3), target=tgt,
                       cover=round(float((contrast_ratio(y_col, _centres()) >= tgt).astype(np.float32) @ p["hist"]), 3),
                       kind=ch["kind"], support=[k for k in ("outline", "shadow", "scrim") if tr.sup[k] > 0.05], hex=hex_of(p["enc"]),
                       design_hex=hex_of(lin_to_srgb(p["design"])), bg_hex=hex_of(lin_to_srgb(p["bg_mean_lin"])),
                       bg_spread=round(p["spread"], 3), cap_px=round(cap_norm, 1), n=int(c["n"]), wt=round(wt, 3),
                       sup_hex=hex_of(p["sup"]), ink_hue=round(_hue_deg(lin_to_oklab(srgb_to_lin(p["enc"]))), 1),
                       shot_pal=[round(x["h"], 1) for x in hues] if sp else None, echo_hex=hex_of(p["echo"]) if front is not None else None,
                       outline_w=round(ow, 2), life=round(life_amt, 3)))
    if frame_idx is not None:
        LC_LOG.setdefault(state_key, {})[int(frame_idx)] = lg
    return pack(rgb, a_out)


legible_color = C.api(_legible_color)
legible_color.__doc__ = _legible_color.__doc__


def _legible_support(image, alpha, state_key="lc", frame_idx=None):
    """paint the support `legible_color` (mode 'post') decided for this frame UNDER the type on the finished image (multiplane): image unchanged
    where the type alpha is 1, darkened / lightened toward the support level elsewhere. No decision for this frame -> image unchanged."""
    s = _SUPPORT.get(state_key)
    if s is None or (frame_idx is not None and s["frame"] != int(frame_idx)):
        return image
    B = C.B
    h, w = image.shape[:2]
    a = C._fit(C._gray(alpha), h, w) if alpha is not None else 0.0
    sa = C._fit(s["sa"], h, w)
    lev = C._fit(s["lev"], h, w)
    k = C._c3(C._clip(sa * (1.0 - a)))
    return image * (1.0 - k) + (lev if lev.ndim == 3 else C._c3(lev)) * k


legible_support = C.api(_legible_support)
legible_support.__doc__ = _legible_support.__doc__

# ================================================================================================== shot tones (flat grounds / inverse type / plates)
_TONE = {}                # state_key -> the shot's hue list (frozen on the first frame)


def tone_pair(hues, ground, ink=None, min_ratio=4.5):
    """a design's LITERAL flat-ground colour (+ the literal ink drawn on it), sRGB 0..1 -> (ground, ink) re-coloured from the shot's hues: the ground keeps its
    lightness and takes the shot's top hue at a low chroma (a black / white plate becomes a deep / pale tint of the shot), the ink takes the shot hue farthest
    round the wheel from the ground's (else the complement) at a characterful chroma, its lightness moved away from the ground until it reaches min_ratio (then 3.0)
    against the new ground. No hues: both returned unchanged."""
    g = np.clip(np.asarray(ground, np.float32).reshape(-1)[:3], 0.0, 1.0)
    i_ = None if ink is None else np.clip(np.asarray(ink, np.float32).reshape(-1)[:3], 0.0, 1.0)
    if not hues:
        return g, i_
    gl = lin_to_oklab(srgb_to_lin(g))
    hu = hues[0]
    cg = float(np.clip(hu["c"], 0.03, 0.08)) * (0.6 if (gl[0] > 0.9 or gl[0] < 0.2) else 1.0)
    g_new = lin_to_srgb(oklab_gamut(np.array([gl[0]]), cg * math.cos(math.radians(hu["h"])), cg * math.sin(math.radians(hu["h"])))[0])
    if i_ is None:
        return g_new, None
    far = max(hues[1:], key=lambda x: hue_dist(x["h"], hu["h"]), default=None)
    hi = far["h"] if (far is not None and hue_dist(far["h"], hu["h"]) >= 60.0) else (hu["h"] + 180.0) % 360.0
    ci = float(np.clip((far["c"] if far is not None else hu["c"]) * 1.8, LC_INK_C[0], LC_INK_C[1]))
    yg = float(rel_lum(srgb_to_lin(g_new)))
    up = yg < 0.18
    for floor in (min_ratio, 3.0):
        for L in (np.linspace(0.55, 0.97, 43) if up else np.linspace(0.55, 0.10, 46)):          # from the mid lightness out: the first passing ink keeps the most chroma
            lin = oklab_gamut(np.array([L]), ci * math.cos(math.radians(hi)), ci * math.sin(math.radians(hi)))[0]
            if float(contrast_ratio(float(rel_lum(lin)), yg)) >= floor:
                return g_new, lin_to_srgb(lin)
    return g_new, i_


def _shot_tone(like, plate, ground, ink=None, part="ground", state_key="tone", user_palette=None):
    """flat colour image like `like`: the shot-toned ground (part 'ground') or ink (part 'ink') of tone_pair, hues from the plate's first frame (or the user's palette)."""
    B = C.B
    hues = _TONE.get(state_key)
    if hues is None:
        pl = np.asarray(B.tonumpy(B.resize(plate, *(lambda hh, ww: (max(2, int(round(hh * LC_AN / max(hh, ww)))), max(2, int(round(ww * LC_AN / max(hh, ww))))))(*plate.shape[:2]), "area")), np.float32)[..., :3]
        hues = _TONE[state_key] = hues_of_colors(user_palette) if user_palette else shot_hues(srgb_to_lin(pl))
    g = [float(v) for v in np.asarray(ground, np.float32).reshape(-1)[:3]]
    i_ = None if ink is None else [float(v) for v in np.asarray(ink, np.float32).reshape(-1)[:3]]
    gn, inn = tone_pair(hues, g, i_)
    col = gn if (part == "ground" or inn is None) else inn
    return C._color(tuple(float(v) for v in col), like)


shot_tone = C.api(_shot_tone)
shot_tone.__doc__ = _shot_tone.__doc__

OPS = {"legible_color": legible_color, "legible_support": legible_support, "shot_tone": shot_tone}


# ================================================================================================== report
def lc_report(state_key, words, fps=24, texts=None):
    """per word: what the type colour stage measured and chose. words = [[start, hit, release, end, fixed], ..] (piece seconds). Per word hold (hit ..
    release): bg luminance (mean), bg / chosen colour, the achieved WCAG ratio (min / mean over the hold frames, worst cluster of each frame), the share of
    the surround that reaches the target (cover), the target, how the colour was found (design kept / lightness / hue / extreme / support), whether support fired
    (outline / shadow / scrim) and how many frames. `life_ratio_min` = the same minimum over the word's whole life (start .. end)."""
    log = LC_LOG.get(state_key) or {}
    out = []
    for i, w in enumerate(words or []):
        s, a, b, e = (float(x) for x in w[:4])
        fixed = bool(len(w) > 4 and w[4])
        row = dict(word=(texts[i] if texts and i < len(texts) else i), start=round(s, 3), hit=round(a, 3), release=round(b, 3), fixed=fixed)
        f0, f1 = int(round(a * fps)), int(round(b * fps))
        fr = [f for f in sorted(log) if f0 <= f < f1 and log[f]]
        life = [f for f in sorted(log) if int(round(s * fps)) <= f < int(round(e * fps)) and log[f]]
        row["frames"] = len(fr)
        if fixed or not fr:
            row.update(measured=False)
            out.append(row)
            continue
        worst = [min(log[f], key=lambda r: r["ratio"]) for f in fr]
        main = [log[f][0] for f in fr]
        kinds = [m["kind"] for m in main]
        sup = {k: sum(1 for f in fr for r in log[f] if k in r["support"]) for k in ("outline", "shadow", "scrim")}
        row.update(measured=True, bg_lum=round(float(np.mean([m["y_bg"] for m in main])), 4), bg_hex=main[len(main) // 2]["bg_hex"],
                   bg_spread=round(float(np.mean([m["bg_spread"] for m in main])), 3), color=main[len(main) // 2]["hex"],
                   color_lum=round(float(np.mean([m["y_col"] for m in main])), 4), target=worst[0]["target"], cap_px=main[0]["cap_px"],
                   ratio_min=round(min(r["ratio"] for r in worst), 3), ratio_mean=round(float(np.mean([r["ratio"] for r in worst])), 3),
                   cover_min=round(min(r["cover"] for r in worst), 3), cover_mean=round(float(np.mean([r["cover"] for r in worst])), 3),
                   sup_hex=main[len(main) // 2].get("sup_hex"), echo_hex=main[len(main) // 2].get("echo_hex"),      # the outline (support) colour the word was drawn with
                   how=max(set(kinds), key=kinds.count), design_kept=round(sum(1 for k in kinds if k == "design") / len(kinds), 2),
                   support=sup, support_fired=any(v > 0 for v in sup.values()),
                   life_ratio_min=round(min(min(r["ratio"] for r in log[f]) for f in life), 3) if life else None)
        row["pass"] = bool(row["ratio_min"] >= row["target"] or row["cover_min"] >= LC_NEED or row["support_fired"])
        out.append(row)
    return out


def format_table(rows):
    """text table of lc_report rows (one shot)."""
    hdr = f"{'word':<14}{'hold s':<13}{'bg Y':>6} {'bg':<8}{'colour':<8}{'tgt':>4}{'min':>7}{'mean':>7}{'cover':>7}  {'how':<10}{'support':<16}ok"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        nm = str(r.get("word"))[:13]
        hold = f"{r['hit']:.2f}-{r['release']:.2f}"
        if not r.get("measured"):
            lines.append(f"{nm:<14}{hold:<13}" + ("fixed (opt-out)" if r.get("fixed") else "not measured (no frames)"))
            continue
        sup = ",".join(k for k, v in r["support"].items() if v) or "-"
        lines.append(f"{nm:<14}{hold:<13}{r['bg_lum']:>6.3f} {r['bg_hex']:<8}{r['color']:<8}{r['target']:>4.1f}{r['ratio_min']:>7.2f}{r['ratio_mean']:>7.2f}"
                     f"{r['cover_min']:>7.2f}  {r['how']:<10}{sup:<16}{'ok' if r['pass'] else 'LOW'}")
    return "\n".join(lines)


def main(argv=None):
    """python -m tools.fusion.legible_color <compose_report.json> [shot id]: print the per-word table of every shot (or one)."""
    av = list(sys.argv[1:] if argv is None else argv)
    if not av:
        print(main.__doc__)
        return 2
    rep = json.loads(Path(av[0]).read_text(encoding="utf-8"))
    n = 0
    for sid, d in (rep.get("shots") or {}).items():
        if len(av) > 1 and sid != av[1]:
            continue
        if d.get("legible_color"):
            print(f"== shot {sid}")
            print(format_table(d["legible_color"]))
            n += 1
    if not n:
        print("no legible_color section in this report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
