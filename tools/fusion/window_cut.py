"""window_cut - compose op + recipe helpers for the three "text as window" design choices (user 2026-10-28: "use the generated words as masks
themselves to contain the background images, or use as a cut to another shot, where we swap to it being the background and instead have it
contained within the text, and then the new shot appears where the previous bg was"). Built on P10's look (knockout of the footage through the
letters, the subject stepping out over everything, flat ground that FLIPS on the hard moments): the ground may now be a SECOND SHOT.

    window_cut(type_a, a, b, t, mode, ...)      compose op (registered through OPS in compose._ops_registry): letters are windows between
                                                two images; WHICH image is inside / outside depends on the time `t` (see window_cut_phase)
    window_cut_phase(t, mode, ...)              pure phase function (what is inside, what is outside, grow / exit progress) for tests + planners
    other_shot_ref(name, sid, ids)              parse the input refs `$next` `$prev` `$shot:<id>` (+ `_matte` / `:matte`) of a two-shot op
    shot_ops(spec, role, ...)                   the compose ops of the window choice for the shot on one side of a cut
    apply_to_piece(piece, type_dir)             piece["compose"]["cuts"] + words' `window_source` -> compose.shots ops (recipe side of design_options.py)
    owned_boundaries(piece)                     boundaries a window cut owns = HARD cuts in the assembler (no crossfade); hard_cut_parts()

Modes (inside = what shows in the letters, outside = the plate; A = the shot before the cut, B = the shot after it; `matte_*` = subject cut-outs):
    other   window_other: from t_open to t_end the letters show B, A stays outside (the same shot of the piece, `$next` or `$shot:<id>`). With A as a
            flat colour and B the footage this IS P10's knockout (bit-identical frames, tests/test_window_cut.py).
    cut     window_cut: over A the word is a window onto B; the window then GROWS (a distance-field dilation of the letters, `grow_s` long) until B
            fills the frame at t_cut = the hard cut. Nothing crossfades. `exit: "io"` keeps the letters as they are and cuts straight to B.
    swap    window_swap: B inside / A outside until t_cut; at t_cut they swap (A inside, B the full background); from t_end the letters leave
            (`exit`: io = the type pass's own exit, fade, shrink = the letters erode away) leaving B.
    zoom    window_zoom (2026-10-30): like `cut`, but the camera SCALES INTO ONE LETTER of the holding word: letters, ground (A) and A's subject matte scale together about the letter's
            thickest point, which glides to the frame centre, while B stays still inside the letters; the letter's stroke grows until it covers the frame and B fills it at t_cut.
            `zoom_s` (>= 0.5 s) is the length of the move, `letter` picks the letter (index left to right; default: the thickest, most central one).
Flat ground (inverse type): `a` may be a plain palette COLOUR instead of a shot; with `b` = the word's own footage (or the next shot) the picture shows ONLY inside the letters over the flat ground
(word key `window_ground`, see ground_colour / apply_to_piece).
The subject (matte) of the OUTSIDE shot stands over both (P10); while a window grows the subject fades out with it so the last frame is exactly B.
Numbers are the frame being composed: t = piece time in seconds, times t_open / t_cut / t_end in seconds of piece time.
"""
from __future__ import annotations

import copy
import re

import cv2
import numpy as np

from tools.fusion import compose as C

KINDS = ("window_other", "window_cut", "window_swap", "window_zoom")
MODE_OF = {"window_other": "other", "window_cut": "cut", "window_swap": "swap", "window_zoom": "zoom"}
EXITS = {"window_other": ("io", "fade", "shrink"), "window_cut": ("grow", "io"), "window_swap": ("io", "fade", "shrink"), "window_zoom": ("zoom",)}
DEFAULT_EXIT = {"window_other": "fade", "window_cut": "grow", "window_swap": "fade", "window_zoom": "zoom"}
ZOOM_MIN_S = 0.5                 # the zoom is a move you can SEE: at least 0.5 s (12 frames at 24 fps)
ZOOM_S = 0.9                     # default length of the zoom (s)
ZOOM_MAX_SCALE = 400.0           # the most the scale may reach (a hair-thin letter)
TAIL_OPS = ("add_glow", "grade", "grain", "vignette", "halation")      # closing look ops of a shot's own chain that are kept around the window op


# ================================================================================================== phases
def _smooth(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def window_cut_phase(t, mode="swap", t_open=0.0, t_cut=None, t_end=None, grow_s=0.7, exit="io", exit_s=0.5, zoom_s=ZOOM_S):
    """what is where at piece time `t`. -> dict(name, bg, inside, grow, exit_k, zoom):
    bg = 'a' | 'b' (the plate outside the letters), inside = 'a' | 'b' | None (None = no window: bg only), grow 0..1 (cut mode: how far the window
    has grown, 1 = fills the frame), exit_k 0..1 (swap / other: how far the letters have faded / eroded away, 1 = gone), zoom 0..1 (zoom mode: progress of the camera move into the letter)."""
    ph = _phase(t, mode, t_open, t_cut, t_end, grow_s, exit, exit_s, zoom_s)
    ph.setdefault("zoom", 0.0)
    return ph


def _phase(t, mode, t_open, t_cut, t_end, grow_s, exit, exit_s, zoom_s):
    t, t_open = float(t) + 1e-3, float(t_open)            # +1 ms: a frame exactly on a (rounded) key time belongs to the new phase
    inf = float("inf")
    te = inf if t_end is None else float(t_end)
    tc = None if t_cut is None else float(t_cut)
    P = dict(name="pre", bg="a", inside=None, grow=0.0, exit_k=0.0)
    if t < t_open:
        return P
    if mode == "zoom":
        if tc is None:
            raise ValueError("window_cut mode 'zoom' needs t_cut")
        if t >= tc:
            return dict(name="post", bg="b", inside=None, grow=1.0, exit_k=0.0, zoom=1.0)
        zs = max(float(zoom_s), ZOOM_MIN_S)
        z0 = tc - zs
        if t >= z0:
            return dict(name="zoom", bg="a", inside="b", grow=0.0, exit_k=0.0, zoom=_smooth((t - z0) / zs))
        return dict(name="window", bg="a", inside="b", grow=0.0, exit_k=0.0, zoom=0.0)
    if mode == "cut":
        if tc is None:
            raise ValueError("window_cut mode 'cut' needs t_cut")
        if t >= tc:
            return dict(name="post", bg="b", inside=None, grow=1.0, exit_k=0.0)
        g0 = tc - max(float(grow_s), 1e-6)
        if exit == "grow" and t >= g0:
            return dict(name="grow", bg="a", inside="b", grow=_smooth((t - g0) / (tc - g0)), exit_k=0.0)
        return dict(name="window", bg="a", inside="b", grow=0.0, exit_k=0.0)
    if mode == "swap" and tc is not None and t >= tc:
        bg, inside = "b", "a"
    else:
        bg, inside = "a", "b"
    name = "w2" if (mode == "swap" and bg == "b") else "w1"
    if t < te:
        return dict(name=name, bg=bg, inside=inside, grow=0.0, exit_k=0.0)
    if exit in ("fade", "shrink"):
        k = (t - te) / max(float(exit_s), 1e-6)
        if k < 1.0:
            return dict(name="exit", bg=bg, inside=inside, grow=0.0, exit_k=_smooth(k))
        return dict(name="post", bg=bg, inside=None, grow=0.0, exit_k=1.0)
    return dict(name=name, bg=bg, inside=inside, grow=0.0, exit_k=0.0)        # exit "io": the type pass takes the letters away


GROUND_RAMP_S = 0.25             # a flat ground fades in over this long as its window opens (and out with the letters' exit)


def ground_presence(ph, t, t_open) -> float:
    """how much of the flat ground colour is on screen at piece time t (0 = only the shot's own picture, 1 = the full ground + window): 0 before the window opens and once it is gone,
    a GROUND_RAMP_S fade-in from t_open, 1 while the letters hold, 1 - exit_k while they leave."""
    name = ph.get("name")
    if name in ("pre", "post") or ph.get("inside") is None:
        return 0.0
    if name == "exit":
        return max(0.0, 1.0 - float(ph.get("exit_k", 0.0)))
    return _smooth((float(t) + 1e-3 - float(t_open)) / GROUND_RAMP_S)


# ================================================================================================== the op
class _Shape:
    def __init__(self, h, w):
        self.shape = (h, w, 3)


def _plane(x, h, w):
    """image (backend array / numpy) or colour (3 numbers) -> HxWx3 backend array."""
    if C._is_arr(x):
        x = C._in(x)
        if x.ndim == 2:
            x = C.B.stack([x, x, x])
        return C._fit(x[..., :3], h, w)
    return C._color(x, _Shape(h, w))


def _sdf(type_a, h, w):
    """signed distance of the letters as a fraction of the long side (+ outside, - inside), full-res backend array, + (max outside, max inside)."""
    a = C._small_np(type_a, 256)
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        m[m.shape[0] // 2, m.shape[1] // 2] = 1
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    s = (dout - din) / float(max(m.shape))
    sf = cv2.resize(s.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return C.B.asarray(np.ascontiguousarray(sf)), float(s.max()), float(-s.min())


# ================================================================================================== zoom into one letter
def zoom_plan(type_a, letter=None, small=512) -> dict:
    """where the camera dives: the thickest point of one letter. type_a = letters alpha (HxW, backend or numpy). -> dict(fx, fy = the focus as fractions of the frame width / height,
    r = the letter's inscribed radius as a fraction of the frame WIDTH, k = index of the letter chosen (left to right), n = letters found, bbox = [x0, y0, x1, y1] fractions of the letter).
    letter: index among the letters ordered left to right (clamped) or None = the one with the thickest stroke, weighted toward the middle of the frame."""
    a = np.asarray(C._small_np(type_a, small), np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    hs, ws = a.shape[:2]
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        m[hs // 2, ws // 2] = 1
    n, lab = cv2.connectedComponents(m, connectivity=8)
    cands = []
    for k in range(1, n):
        mk = (lab == k).astype(np.uint8)
        if int(mk.sum()) < 3:
            continue
        d = cv2.distanceTransform(mk, cv2.DIST_L2, 3)
        yy, xx = np.nonzero(mk)
        iy, ix = np.unravel_index(int(d.argmax()), d.shape)
        cands.append(dict(r=float(d.max()), fx=(ix + 0.5) / ws, fy=(iy + 0.5) / hs, cx=float(xx.mean()) / ws, bbox=[xx.min() / ws, yy.min() / hs, (xx.max() + 1) / ws, (yy.max() + 1) / hs]))
    if not cands:
        cands = [dict(r=1.0, fx=0.5, fy=0.5, cx=0.5, bbox=[0.4, 0.4, 0.6, 0.6])]
    ordered = sorted(cands, key=lambda c: c["cx"])
    if letter is None:
        pick = max(ordered, key=lambda c: c["r"] * (1.0 - 0.6 * abs(c["fx"] - 0.5)))
    else:
        pick = ordered[min(max(int(letter), 0), len(ordered) - 1)]
    return dict(fx=float(pick["fx"]), fy=float(pick["fy"]), r=float(pick["r"] / ws), k=ordered.index(pick), n=len(ordered), bbox=[float(v) for v in pick["bbox"]])


def zoom_scale_end(plan, w, h) -> float:
    """the scale at which the letter's inscribed circle (centred on the frame centre) covers the frame's far corner: s_end * r >= half the frame diagonal (+ 10 %: the radius is measured on a 512 px map)."""
    r_px = max(plan["r"] * w, 1.0)
    return float(min(1.10 * 0.5 * float(np.hypot(w, h)) / r_px, ZOOM_MAX_SCALE))


def zoom_state(plan, z, w, h) -> dict:
    """the camera at zoom progress z (0..1): s = scale (geometric from 1 to zoom_scale_end), (cx, cy) = where the focus sits on screen in pixels (glides from its place in the frame to the
    centre). Output pixel p shows source pixel f + (p - c) / s."""
    z = min(max(float(z), 0.0), 1.0)
    s_end = zoom_scale_end(plan, w, h)
    s = float(np.exp(z * np.log(s_end)))
    fx, fy = plan["fx"] * w, plan["fy"] * h
    g = z * z * (3.0 - 2.0 * z)
    return dict(s=s, fx=fx, fy=fy, cx=fx + g * (w / 2.0 - fx), cy=fy + g * (h / 2.0 - fy), s_end=s_end)


def zoom_maps(h, w, st):
    """cv2.remap maps (src x, src y) for the output frame: p_src = f + (p - c) / s."""
    xs = ((np.arange(w, dtype=np.float64) - st["cx"]) / st["s"] + st["fx"]).astype(np.float32)
    ys = ((np.arange(h, dtype=np.float64) - st["cy"]) / st["s"] + st["fy"]).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(xs[None, :], (h, w))), np.ascontiguousarray(np.broadcast_to(ys[:, None], (h, w)))


def _remap(img, mx, my, interp=cv2.INTER_LINEAR):
    return cv2.remap(np.ascontiguousarray(img, dtype=np.float32), mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


def zoom_alpha(type_a, st, h, w, edge_px=1.5, long=960):
    """alpha (numpy HxW) of the letters at zoom state `st`: the signed distance field of the letters (taken at <= `long` px, so edges stay crisp at any scale) sampled through the zoom map,
    scaled by s, thresholded with a constant `edge_px` anti-aliasing. At s = 1 and no glide it is the original mask."""
    a = np.asarray(C._small_np(type_a, long), np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    hs, ws = a.shape[:2]
    m = (a > 0.5).astype(np.uint8)
    if not m.any():
        return np.zeros((h, w), np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = (dout - din).astype(np.float32)
    sdf = np.where(np.abs(sdf) <= 1.5, 0.5 - a, sdf).astype(np.float32)   # sub-pixel edge from the anti-aliased alpha (the binary distance has stair steps that a 20x zoom would show)
    sdf = cv2.GaussianBlur(sdf, (0, 0), 0.9) * np.float32(w / float(ws))   # + outside, - inside, in OUTPUT pixels at s = 1; a distance field is smooth, a light blur only rounds the steps
    mx, my = zoom_maps(h, w, st)
    d = _remap(sdf, (mx * np.float32(ws / float(w))).astype(np.float32), (my * np.float32(hs / float(h))).astype(np.float32), cv2.INTER_CUBIC)
    return np.clip(0.5 - d * st["s"] / max(edge_px, 1e-3), 0.0, 1.0).astype(np.float32)


def zoom_layers(plate, ta, top_m, line, zk, h, w, letter=None, plan=None):
    """the zoom applied to the layers that travel with the camera: the plate (A), A's subject matte, the outline. -> (plate_z, alpha_z, matte_z | None, line_z | None, plan, state) as backend arrays."""
    plan = plan or zoom_plan(ta, letter)
    st = zoom_state(plan, zk, w, h)
    mx, my = zoom_maps(h, w, st)
    pl = C.B.asarray(np.ascontiguousarray(_remap(np.asarray(C.B.tonumpy(plate), np.float32), mx, my, cv2.INTER_CUBIC if st["s"] < 6 else cv2.INTER_LINEAR)))
    al = C.B.asarray(np.ascontiguousarray(zoom_alpha(ta, st, h, w)))
    mt = None if top_m is None else C.B.asarray(np.ascontiguousarray(_remap(np.asarray(C.B.tonumpy(top_m), np.float32), mx, my)))
    ln = None if line is None else C.B.asarray(np.ascontiguousarray(_remap(np.asarray(C.B.tonumpy(line), np.float32), mx, my)))
    return pl, al, mt, ln, plan, st


def _window_alpha(ta, ph, soft, exit):
    """letters alpha for this phase: grown (distance field) / faded / eroded. Backend HxW."""
    h, w = ta.shape[:2]
    soft = max(float(soft), 1e-4)
    if ph["grow"] > 0.0:
        s, dmax, _ = _sdf(ta, h, w)
        r = ph["grow"] * (dmax + soft)
        return C._clip(_maxi(ta, C._clip((r - s) / soft + 0.5)))
    k = ph["exit_k"]
    if k > 0.0 and ph["inside"] is not None:
        if exit == "shrink":
            s, _, din = _sdf(ta, h, w)
            e = k * (din + soft)
            f = C._clip((-e - s) / soft + 0.5)
            return _mini(ta, f)
        return ta * (1.0 - k)
    return ta


def _maxi(x, y):
    return np.maximum(x, y) if isinstance(x, np.ndarray) else C.torch.maximum(x, y)


def _mini(x, y):
    return np.minimum(x, y) if isinstance(x, np.ndarray) else C.torch.minimum(x, y)


def _reads(plate, ta, ins, gap, cgap=0.18, core=0.5, ring=0.035):
    """does the window already read? luma of the picture under the letters vs the ring around the word differs by >= gap (either way), or the
    mean colours are >= cgap apart (RGB distance, 0..1). False when there are no letter pixels to measure."""
    a = (C._small_np(ta) > core).astype(np.uint8)
    if int(a.sum()) < 12:
        return False
    f, p = C._small_np(ins), C._small_np(plate)
    r = max(2, int(round(ring * max(a.shape))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = (cv2.dilate(a, k) > 0) & (a == 0)
    if not band.any():
        return False
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    ci, co = f[a > 0].mean(0), p[band].mean(0)
    return bool(abs(float(ci @ w) - float(co @ w)) >= gap or float(np.linalg.norm(ci - co)) >= cgap)


def _window_cut(type_a, a, b, t=0.0, mode="swap", t_open=0.0, t_cut=None, t_end=None, grow_s=0.7, exit="io", exit_s=0.5, soft=0.012,
                matte_a=None, matte_b=None, subject="outside", fg_matte=None, fg=None, min_contrast=None, face=None, line=None,
                line_color=None, reads_gap=0.12, a_subject=None, b_subject=None, zoom_s=ZOOM_S, letter=None, under=None):
    """see the module doc. a / b: images (HxWx3) or colours; type_a: letters alpha HxW; matte_a / matte_b: subject cut-outs of A / B;
    subject 'outside' (default) = the matte of the shot that is the plate right now steps out over the window, 'inside' = the matte of the
    shot in the letters, 'none'; fg_matte / fg override that (P10: fg_matte = the shot's matte, fg None = the window's footage);
    min_contrast / face: knockout's legibility guard (skipped while the window already differs from its surround by `reads_gap` in luma, so the
    plate is not darkened when the inside is the DARKER picture); line / line_color: optional outline (alpha HxW, colour; default near-white on dark / near-black on light)."""
    arrs = [x for x in (a, b) if C._is_arr(x)]
    if arrs:
        h, w = C._in(arrs[0]).shape[:2]
    else:
        h, w = C._in(type_a).shape[:2]
    ph = window_cut_phase(float(t), mode, t_open, t_cut, t_end, grow_s, exit, exit_s, zoom_s)
    if under is not None and mode == "other" and not C._is_arr(a):       # flat GROUND (inverse type on a plain colour): the ground exists only while the window does, the shot's own picture (`under`) is the frame before and after
        p = ground_presence(ph, float(t), t_open)                        # (QC v026 r02: before t_open and after t_end + exit_s the frame was the bare ground colour = blank grey frames, 9.5-10.4 s and 12.0-12.75 s)
        U = _plane(under, h, w)
        if p <= 0.0:
            return U
        if p < 1.0:
            inner = _window_cut(type_a, a, b, t=t, mode=mode, t_open=t_open, t_cut=t_cut, t_end=t_end, grow_s=grow_s, exit=exit, exit_s=exit_s, soft=soft, matte_a=matte_a,
                                matte_b=matte_b, subject=subject, fg_matte=fg_matte, fg=fg, min_contrast=min_contrast, face=face, line=line, line_color=line_color,
                                reads_gap=reads_gap, a_subject=a_subject, b_subject=b_subject, zoom_s=zoom_s, letter=letter, under=None)
            return C._lerp(U, inner, p)
    A, Bm = _plane(a, h, w), _plane(b, h, w)
    plate = A if ph["bg"] == "a" else Bm
    if ph["inside"] is None:
        return plate
    ins = A if ph["inside"] == "a" else Bm
    sa = None if a_subject is None else _plane(a_subject, h, w)          # three-layer shots (16_app/LAYERS.md): the clean-clip pixels of A / B,
    sb = None if b_subject is None else _plane(b_subject, h, w)          # taken instead of the plate / window pixels when the subject steps out
    subj_bg = sa if ph["bg"] == "a" else sb
    subj_in = sa if ph["inside"] == "a" else sb
    ta = _window_alpha(C._clip(C._mask(C._in(type_a), h, w)), ph, soft, exit)
    gate = 1.0 - ph["grow"] - (ph["zoom"] if mode == "zoom" else 0.0)
    gate = max(gate, 0.0)
    top_m, top = None, None
    if fg_matte is not None:
        top_m, top = fg_matte, (fg if fg is not None else subj_in)
    elif subject == "outside":
        top_m = matte_a if ph["bg"] == "a" else matte_b
        top = (plate if subj_bg is None else subj_bg) if top_m is not None else None
    elif subject == "inside":
        top_m = matte_a if ph["inside"] == "a" else matte_b
        top = (ins if subj_in is None else subj_in) if top_m is not None else None
    if top_m is not None and gate < 1.0:
        top_m = C._clip(C._mask(C._in(top_m), h, w)) * gate
    mc = None if min_contrast is None else float(min_contrast) * gate
    if mc is not None and mc < 1e-3:
        mc = None
    if mc is not None and reads_gap is not None:                 # a window that already differs from its surround by `reads_gap` in luma (either way: bright
        if _reads(plate, ta, ins, float(reads_gap)):            # on dark or dark on bright, or by colour) reads: the plate is left alone, the guard only
            mc = None                                           # rescues similar ones
    line_z = None
    if mode == "zoom" and ph["zoom"] > 0.0:                    # the camera scales into one letter: ground (A), letters and A's subject matte move together (the matte follows its own picture); B stays still inside
        line_m = None if line is None else C._clip(C._mask(C._in(line), h, w))
        plate, ta, top_m, line_z, _plan, _st = zoom_layers(plate, ta, top_m, line_m, ph["zoom"], h, w, letter)
        if top is not None:
            top = plate
    out = C.knockout.raw(plate, ta, ins, fg_matte=top_m, fg=top, min_contrast=mc, face=face)
    if line is not None and mode == "zoom" and ph["zoom"] > 0.0:
        la = (line_z if line_z is not None else C._clip(C._mask(C._in(line), h, w))) * (1.0 - ph["zoom"])
        if line_color is None:
            lum = float(np.mean(C._small_np(plate, 48)))
            line_color = (0.03, 0.03, 0.028) if lum > 0.5 else (0.9, 0.9, 0.88)
        return C._lerp(out, C._color(line_color, out), C._c3(la))
    if line is not None:
        la = C._clip(C._mask(C._in(line), h, w)) * (1.0 - ph["grow"]) * (1.0 - ph["exit_k"])
        if line_color is None:
            lum = float(np.mean(C._small_np(plate, 48)))
            line_color = (0.03, 0.03, 0.028) if lum > 0.5 else (0.9, 0.9, 0.88)
        out = C._lerp(out, C._color(line_color, out), C._c3(la))
    return out


window_cut = C.api(_window_cut)
OPS = {"window_cut": window_cut}


# ================================================================================================== input refs
def other_shot_ref(name, sid, ids):
    """compose input ref (without the $) of a two-shot op -> (target shot id, 'footage' | 'matte') or None when it is not such a ref.
    next / prev (+ _matte) = the neighbour of shot `sid`, shot:<id> (+ :matte) = any shot. ValueError for a missing neighbour / unknown id."""
    ids = list(ids)
    m = re.fullmatch(r"(next|prev)(_matte|_subject)?", name)
    if m:
        k = ids.index(sid) + (1 if m.group(1) == "next" else -1)
        if k < 0 or k >= len(ids):
            raise ValueError(f"compose {sid}: ${name} but shot {sid} has no {m.group(1)} shot")
        return ids[k], ({"_matte": "matte", "_subject": "subject"}.get(m.group(2), "footage"))
    m = re.fullmatch(r"shot:([^:]+)(:matte|:subject)?", name)
    if m:
        if m.group(1) not in ids:
            raise ValueError(f"compose {sid}: ${name}: no shot {m.group(1)!r} (shots: {', '.join(ids)})")
        return m.group(1), ({":matte": "matte", ":subject": "subject"}.get(m.group(2), "footage"))
    return None


def other_refs(piece, sid):
    """[(ref name, target sid, 'footage'|'matte')] the compose ops of shot `sid` read from OTHER shots (the caller supplies those frames)."""
    from tools.fusion import recipe_eval as RE
    _, spec = RE._compose_spec(piece, sid)
    rs = set()
    RE.refs([o.get("args", {}) for o in spec["ops"]], rs)
    ids = [s["id"] for s in piece["shots"]]
    out = []
    for r in sorted(rs):
        o = other_shot_ref(r, sid, ids)
        if o:
            out.append((r, o[0], o[1]))
    return out


def collect_others(piece, sid, footage_of, matte_of=None, subject_of=None):
    """`others=` argument of recipe_eval.run_compose for shot `sid`: {target sid: {"footage": frames, "matte": frames}} for every other shot its compose ops
    read ($next / $prev / $shot:<id>). footage_of(sid) / matte_of(sid) return the frames of that shot (matte_of may return None). None when it reads none."""
    out = {}
    for _r, tgt, kd in other_refs(piece, sid):
        e = out.setdefault(tgt, {})
        e[kd] = footage_of(tgt) if kd == "footage" else ((subject_of(tgt) if subject_of else None) if kd == "subject" else (matte_of(tgt) if matte_of else None))
    if subject_of is not None:                       # layered shots: the layerized ops also read the neighbours' clean-clip pixels ($next_subject ...)
        for tgt, e in out.items():
            sub = subject_of(tgt)
            if sub is not None:
                e["subject"] = sub
    return out or None


# ================================================================================================== recipe side
def _is_3d_cut(kind):
    """depth_pivot / orbit:<mode> / depth_dolly:<dir> (tools/fusion/pivot_cut.py): the transition frames already hold the whole change of picture."""
    if not isinstance(kind, str) or not any(kind.lower().startswith(x) for x in ("depth_", "orbit", "pivot", "dolly", "push", "pull", "fly", "circle", "dot_grid", "lines", "iso_")):
        return False
    from tools.fusion import pivot_cut
    return pivot_cut.normalize(kind) is not None


def owned_boundaries(piece):
    """indices k (boundary between shot k and k+1) that a window cut owns (or that the design made a plain `hard` cut): HARD cuts, no
    crossfade. [] for pieces without compose.cuts."""
    ids = [s["id"] for s in piece.get("shots", [])]
    out = []
    for c in ((piece.get("compose") or {}).get("cuts") or []):
        if (c.get("cut") in ("window_cut", "window_swap", "window_zoom", "hard") or _is_3d_cut(c.get("cut"))) and c.get("after") in ids:
            k = ids.index(c["after"])
            if k + 1 < len(ids) and k not in out:
                out.append(k)
    return sorted(out)


def hard_cut_parts(parts, crossfades, owned, layout, fps):
    """assemble() inputs with the owned boundaries made HARD: `parts` = [{sid, cut}] in shot order, `crossfades` = per-boundary seconds,
    layout = {sid: {start, frames, xf}}. The owned crossfade becomes 0 and the incoming shot starts exactly at its piece start (the shots keep
    their layout, so the music timing and the locked crossfade of the piece stay as designed). -> (parts, crossfades) copies."""
    parts = [dict(p) for p in parts]
    cfl = list(crossfades)
    for k in owned:
        if k < len(cfl) and k + 1 < len(parts):
            cfl[k] = 0.0
            parts[k + 1]["cut"] = layout[parts[k + 1]["sid"]]["start"] / float(fps)
    return parts, cfl


def _ref(spec, role, key):
    """the op input for the 'other' side. role from: next shot is `to`; role to: previous shot is `from`."""
    if key == "a":
        return "$footage" if role == "from" else "$prev"
    return "$next" if role == "from" else "$footage"


def shot_ops(spec, role, tail=None, type_ref="$type_matte", line_ref="$type_line", extra=None):
    """compose ops of one window choice for the shot on one side of the cut. spec: kind, from, to, t_open, t_cut, t_end, exit, exit_s, grow_s,
    min_contrast, line (bool), soft. role 'from' (the shot before the cut, A) | 'to' (the shot after it, B; only cut / swap). tail = closing ops
    (grade / grain / vignette) appended on `$image`. window_other reads its second shot as `$shot:<to>` (or `$next`)."""
    kind = spec["kind"]
    mode = MODE_OF[kind]
    if kind == "window_other" and spec.get("ground") is not None:        # inverse type on a PLAIN ground: a palette colour outside, the picture only inside the letters
        own = spec.get("to") in ("self", "own")
        b_ref = "$footage" if own else ("$next" if spec.get("to") in (None, "next") else f"$shot:{spec['to']}")
        a_ref, ma, mb = [round(float(v), 4) for v in spec["ground"]], None, ("$matte" if own else (b_ref + ("_matte" if b_ref == "$next" else ":matte")))
    elif kind == "window_other":
        nxt = "$next" if spec.get("to") in (None, "next") else f"$shot:{spec['to']}"
        a_ref, b_ref, ma, mb = "$footage", nxt, "$matte", nxt + ("_matte" if nxt == "$next" else ":matte")
    else:
        a_ref, b_ref = _ref(spec, role, "a"), _ref(spec, role, "b")
        ma = "$matte" if role == "from" else "$prev_matte"
        mb = "$next_matte" if role == "from" else "$matte"
    args = dict(type_a=type_ref, a=a_ref, b=b_ref, t="$t", mode=mode, t_open=round(float(spec["t_open"]), 4),
                exit=spec.get("exit", DEFAULT_EXIT[kind]), exit_s=round(float(spec.get("exit_s", 0.5)), 3),
                matte_a=ma, matte_b=mb)
    if ma is None:
        args.pop("matte_a")
    if kind == "window_other" and spec.get("ground") is not None:
        args["subject"] = "inside"                                       # the subject of the picture in the letters steps out over the flat ground (P10)
        args["under"] = "$footage"                                       # the shot's own picture shows before the window opens and after it has gone (the ground is not a frame of its own)
    if spec.get("t_cut") is not None:
        args["t_cut"] = round(float(spec["t_cut"]), 4)
    if spec.get("t_end") is not None:
        args["t_end"] = round(float(spec["t_end"]), 4)
    if kind == "window_cut":
        args["grow_s"] = round(float(spec.get("grow_s", 0.7)), 3)
    if kind == "window_zoom":
        args["zoom_s"] = round(max(float(spec.get("zoom_s", ZOOM_S)), ZOOM_MIN_S), 3)
        if spec.get("letter") is not None:
            args["letter"] = int(spec["letter"])
    if spec.get("min_contrast") is not None:
        args["min_contrast"] = float(spec["min_contrast"])
    if spec.get("soft") is not None:
        args["soft"] = float(spec["soft"])
    if spec.get("line"):
        args["line"] = line_ref
    args.update(extra or {})
    ops = [{"op": "window_cut", "out": "image", "args": args}]
    for o in (tail or []):
        ops.append(copy.deepcopy(o))
    return ops


def tail_ops(ops):
    """the closing look ops of a shot's own chain (TAIL_OPS) as ops on `$image`; everything that draws the type or prepares a plate is dropped
    (the window op replaces it). add_glow keeps its mask; frame / img arguments are re-pointed to $image."""
    out = []
    for o in ops:
        if o["op"] not in TAIL_OPS:
            continue
        o = copy.deepcopy(o)
        a = o["args"]
        for k in ("frame", "img"):
            if k in a:
                a[k] = "$image"
        o["out"] = "image"
        out.append(o)
    return out


def apply_to_piece(piece, type_dir=None):
    """piece["compose"]["cuts"] (+ words' `window_source`) -> compose.shots[<sid>] ops. In place; returns [notes]. Idempotent (rebuilds the shots it owns
    from the pieces's default chain's closing ops). Each cut dict: after (shot id), cut (window_cut | window_swap | crossfade | hard), word (index),
    exit, exit_s, grow_s, freeze_s, and optional explicit t_open / t_cut / t_end (seconds) that win over the computed ones.
    Times: t_cut = the incoming shot's piece start; word start / hit / release / end from the type pass manifest (type_dir) or the piece's plan."""
    from tools.fusion import recipe_eval as RE
    comp = piece.setdefault("compose", {})
    cuts = [c for c in (comp.get("cuts") or []) if c.get("cut") in ("window_cut", "window_swap", "window_zoom")]
    others = [(i, w) for i, w in enumerate((piece.get("type") or {}).get("words") or []) if w.get("window_source") not in (None, "self") or w.get("window_ground")]
    if not cuts and not others:
        return []
    fps = int(piece.get("fps", 24))
    lay, _ = RE.piece_layout(piece)
    ids = [s["id"] for s in piece["shots"]]
    words = _word_times(piece, type_dir)
    base_tail = tail_ops(((comp.get("default") or {}).get("ops")) or [])
    shots = comp.setdefault("shots", {})
    notes, freezes = [], []
    for c in cuts:
        sa = c["after"]
        if sa not in ids or ids.index(sa) + 1 >= len(ids):          # a single-shot piece, or `after` is the last / an unknown shot: no boundary to cut at
            notes.append(f"{c['cut']} after {sa!r} skipped: {'no such shot' if sa not in ids else 'the last shot has no boundary after it'} (shots: {', '.join(ids)})")
            continue
        sb = ids[ids.index(sa) + 1]
        wi = int(c.get("word", 0))
        if not 0 <= wi < len(words):
            notes.append(f"{c['cut']} {sa}->{sb} skipped: word {wi} does not exist ({len(words)} words)")
            continue
        wt = words[wi]
        tcut = lay[sb]["start"] / float(fps)
        spec = dict(kind=c["cut"], t_open=c.get("t_open", wt["start"]), t_cut=c.get("t_cut", tcut),
                    t_end=c.get("t_end", wt["release"] if c["cut"] == "window_swap" else None),
                    exit=c.get("exit", DEFAULT_EXIT[c["cut"]]), exit_s=c.get("exit_s", 0.5), grow_s=c.get("grow_s", 0.7),
                    min_contrast=c.get("min_contrast", RE.LP_MIN_CONTRAST), line=c.get("line", True), soft=c.get("soft"),
                    zoom_s=c.get("zoom_s", ZOOM_S), letter=c.get("letter"))
        shots[sa] = {"ops": shot_ops(spec, "from", base_tail), "output": "image"}
        shots[sb] = {"ops": shot_ops(spec, "to", base_tail), "output": "image"}
        if c.get("freeze_s"):
            freezes.append([round(spec["t_cut"], 4), round(float(c["freeze_s"]), 3)])
        notes.append(f"{c['cut']} {sa}->{sb}: word {wi} {wt['text']!r}, open {spec['t_open']:.2f} cut {spec['t_cut']:.2f}"
                     + (f" end {spec['t_end']:.2f}" if spec["t_end"] is not None else ""))
    for i, w in others:
        if len(ids) < 2 or i >= len(words):                         # nothing to show through the letters in a single-shot piece
            notes.append(f"window_other word {i} skipped: {'the piece has a single shot' if len(ids) < 2 else 'no such word'}")
            continue
        wt = words[i]
        t_o, t_e = w.get("window_t") or (wt["start"], wt["release"])        # optional explicit [t_open, t_end] seconds on the word
        sid = _shot_at(lay, fps, (t_o + t_e) / 2.0)
        tgt = w.get("window_source") or "self"
        if tgt == "next":
            tgt = ids[min(ids.index(sid) + 1, len(ids) - 1)]
        if tgt == sid:
            tgt = "self"
        spec = dict(kind="window_other", to=tgt, t_open=t_o, t_end=t_e, exit=w.get("window_exit", "fade"), exit_s=0.5,
                    min_contrast=RE.LP_MIN_CONTRAST, line=True)
        if w.get("window_ground"):                                       # inverse type on a plain ground: a colour of the piece's palette (ground_colour)
            spec["ground"] = ground_colour(piece, w["window_ground"])
            spec["min_contrast"] = None                                  # the flat ground is chosen against the picture; the plate guard has nothing to protect
        if sid not in shots:
            shots[sid] = {"ops": shot_ops(spec, "from", base_tail), "output": "image"}
        notes.append(f"window_other {sid}: word {i} {wt['text']!r} shows {tgt}")
    if freezes:
        comp["holds"] = sorted(list(comp.get("holds") or []) + [f for f in freezes if f not in (comp.get("holds") or [])])
        comp.setdefault("holds_type", True)
    return notes


def ground_colour(piece, ground="plain"):
    """the flat ground (r, g, b in 0..1) of an inverse-type word. A colour given on the word wins; `plain` = the palette colour that stands OFF the piece's imagery: the extracted
    still palette's light colour when the first still is mostly dark, its dark colour when it is light (the picture inside the letters then reads against it); without a still palette the
    type palette's fg / bg."""
    if isinstance(ground, (list, tuple)) and len(ground) == 3:
        v = [float(x) for x in ground]
        return [x / 255.0 for x in v] if max(v) > 1.5 else v
    plan = piece.get("plan") or {}
    sp = plan.get("still_palette")
    if isinstance(sp, dict) and sp.get("colors"):
        rgb = sp["light"] if float(sp.get("mean_lightness", 40.0)) < 55.0 else sp["dark"]
    else:
        pal = (piece.get("type") or {}).get("palette") or {}
        rgb = pal.get("fg") or [235, 235, 235]
    return [round(float(x) / 255.0, 4) for x in rgb]


def _shot_at(lay, fps, t):
    f = t * fps
    best = None
    for sid, L in lay.items():
        if L["start"] <= f < L["start"] + L["frames"]:
            best = sid
    return best or list(lay)[-1]


def _word_times(piece, type_dir=None):
    """[{text, start, hit, release, end}] piece seconds per word: the type pass manifest first, else the piece's plan / word fields."""
    from tools.fusion import recipe_eval as RE
    try:
        if type_dir is not None:
            import json
            from pathlib import Path
            ws = ((json.loads((Path(type_dir) / "manifest.json").read_text(encoding="utf-8")).get("info") or {}).get("words")) or []
            if ws:
                return [dict(text=w["text"], start=float(w.get("start", w["hit"])), hit=float(w["hit"]), release=float(w["release"]),
                             end=float(w.get("end", w["release"]))) for w in ws]
    except (OSError, ValueError, KeyError):
        pass
    pw = (piece.get("type") or {}).get("words") or []
    hr = RE.piece_word_holds(piece, None)
    out = []
    for i, w in enumerate(pw):
        hit, rel = hr[i] if i < len(hr) else (float(w.get("t0", 0.0)) + float(w.get("form_s", 0.75)), float(w.get("t1", 0.0)))
        out.append(dict(text=w.get("text", str(i)), start=float(w.get("t0", hit - float(w.get("form_s", 0.75)))), hit=float(hit),
                        release=float(rel), end=float(w.get("t1", rel + float(w.get("exit_s", 0.5))))))
    return out
