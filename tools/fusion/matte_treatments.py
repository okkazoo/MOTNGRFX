"""matte_treatments: interesting uses of the subject matte for the type, composable per line (v005, 2026-10-26).

A word spec carries `"matte": "front" | "window" | "fill" | "wipe" | "glow"` (or {"mode": .., options}); `ops_for(treatments)` turns the
list of (line window, treatment) into compose recipe ops that run after the plain image is built (graded plate in `image`):

  front   type half in FRONT of the subject, half behind: `sandwich(front=region)`; region = a half-frame split (axis x|y|diag, position
          a timeline so the split can sweep across) or any buffer (e.g. the type's depth key `$near`, as P14 does)
  window  type as a WINDOW into a brighter plate: the plate dims while the line is on, the letters are `knockout` windows onto a brighter
          grade of the same footage, the subject steps out over both. Contrast is GUARANTEED (v005 fix): compose.knockout(min_contrast) darkens the
          surround until the window is clearly brighter, and falls back to solid beauty-coloured type when the footage itself is too dark
  fill    type FILLS the subject silhouette: letters show at full strength inside the matte, as a ghost outside it, over a darkened subject
  wipe    matte-edge REVEAL wipe (tools/mathtype/matte_wipe: out = A (1 - km) + B km, glow on the edge): a front grows out from the subject
          outline; the brighter plate and the letters exist only inside it, a thin ring glows on the front
  glow    OUTLINE GLOW: contour_lines(outline | scribble) of the matte, bloomed, pulsing on the beat (`intensity` may be {"curve": "onset",..})

Legibility guard: `front` and `wipe` end in a real `sandwich` op, so recipe_eval.guard_ops inserts the `legible` guard before it (a matte never
hides a word's letters for the whole hold). `window` carries an explicit `legible` op for its subject-steps-out matte. `fill` and `glow` put
the type / line in front of the subject, nothing hides.

New numpy ops (registered into compose._ops_registry through OPS): mask_gate, mask_mul, split_region, matte_reach, matte_ring.
Numpy / cv2 only, run per frame on the box with the rest of compose."""
from __future__ import annotations

import cv2
import numpy as np

MODES = ("front", "window", "fill", "wipe", "glow")
NEEDS_MATTE = {"front": True, "window": False, "fill": True, "wipe": True, "glow": True}


# ======================================================================================================== ops (numpy, HxW float32)
def _g(x, h=None, w=None):
    a = np.asarray(x, np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] == 1 else (a[..., 3] if a.shape[-1] == 4 else a[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    if h is not None and a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(a)


def _ss(x, e0, e1):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mask_gate(a, gate=1.0, scale=1.0):
    """mask * gate * scale (gate: a timeline value 0..1 that switches a treatment on for its line)."""
    return np.clip(_g(a) * float(gate) * float(scale), 0.0, 1.0)


def mask_mul(a, b, floor=0.0, gain=1.0):
    """a * (floor + (1 - floor) * clip(b * gain)): a seen through b, with `floor` of it left where b is 0 (a ghost)."""
    a = _g(a)
    b = np.clip(_g(b, *a.shape) * float(gain), 0.0, 1.0)
    return np.clip(a * (float(floor) + (1.0 - float(floor)) * b), 0.0, 1.0)


def split_region(like, axis="x", pos=0.5, soft=0.06, side="left", angle=35.0):
    """HxW 0..1 region of half the frame: axis x (side left | right of `pos`), y (top | bottom), diag (along `angle` degrees); `pos` 0..1 may
    be a timeline so the split sweeps; `soft` = ramp width as a fraction of the frame."""
    h, w = like.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / max(w - 1, 1), yy / max(h - 1, 1)
    if axis == "y":
        u = y
    elif axis == "diag":
        a = np.radians(float(angle))
        u = (x * np.cos(a) * w + y * np.sin(a) * h) / (abs(np.cos(a)) * w + abs(np.sin(a)) * h)
        u = (u - u.min()) / max(float(u.max() - u.min()), 1e-6)
    else:
        u = x
    p, s = float(pos), max(float(soft), 1e-4)
    m = 1.0 - _ss(u, p - 0.5 * s, p + 0.5 * s)                       # 1 before the split (left / top)
    return (1.0 - m if side in ("right", "bottom") else m).astype(np.float32)


def _dist_from_subject(matte, thresh):
    """distance in px of every pixel from the subject (0 inside), computed at <= 320 px and scaled back."""
    m = _g(matte)
    h, w = m.shape
    f = min(1.0, 320.0 / max(h, w))
    small = cv2.resize(m, (max(8, int(round(w * f))), max(8, int(round(h * f)))), interpolation=cv2.INTER_AREA) if f < 1 else m
    d = cv2.distanceTransform((small < thresh).astype(np.uint8), cv2.DIST_L2, 3)
    d = d / f
    return (cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if f < 1 else d), np.hypot(h, w)


def matte_reach(matte, progress=0.0, reach=0.7, soft=0.05, thresh=0.5):
    """HxW 0..1: the region within progress * reach * (frame diagonal) px of the subject (inside the subject = 1): a front that grows from
    the matte outline. soft = ramp width as a fraction of the diagonal."""
    d, diag = _dist_from_subject(matte, thresh)
    r = float(np.clip(progress, 0.0, 1.0)) * float(reach) * diag
    s = max(float(soft) * diag, 1.0)
    if progress <= 0.0:
        return np.clip(_g(matte) * 0.0, 0, 1)
    return (1.0 - _ss(d, r - 0.5 * s, r + 0.5 * s)).astype(np.float32)


def matte_ring(matte, progress=0.0, reach=0.7, width=0.012, thresh=0.5):
    """HxW 0..1: a thin ring on the front of `matte_reach` (the glowing edge of the wipe); 0 at progress 0 and 1."""
    d, diag = _dist_from_subject(matte, thresh)
    p = float(np.clip(progress, 0.0, 1.0))
    r, wd = p * float(reach) * diag, max(float(width) * diag, 1.0)
    ring = np.exp(-0.5 * ((d - r) / wd) ** 2) * float(np.sin(np.pi * p) ** 0.5)
    return np.clip(ring, 0.0, 1.0).astype(np.float32)


OPS = {"mask_gate": mask_gate, "mask_mul": mask_mul, "split_region": split_region, "matte_reach": matte_reach, "matte_ring": matte_ring}


# ======================================================================================================== recipe builder
def _tl(points, ease="smooth"):
    return {"timeline": [[round(float(t), 4), float(v)] for t, v in points], "ease": ease}


def gate(t0, t1, fade=0.15):
    """timeline 0 -> 1 over `fade` s before t0, 1 until t1, 1 -> 0 over `fade` s after."""
    return _tl([(t0 - fade, 0.0), (t0, 1.0), (t1, 1.0), (t1 + fade, 0.0)])


def _timeline_like(v, default):
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        return _tl(v)
    return default if v is None else v


EDGE_GLOW = False       # 2026-10-30 (user: no glow behind the type unless it is hold-life): the wipe ring / glow-mode outline carry NO add_glow by default (v030 r01 / r02 chains had `$_wi2g` / `$_gl0c` / `$_wi4g`);
                        # a treatment opts in with `"edge_glow": true` (a request, never a default)


def _edge_glow(tr) -> bool:
    return bool(tr.get("edge_glow", EDGE_GLOW))


def ops_for(treatments, *, base="image", plate="$footage", matte="$m", type_a="$type_matte", type_rgb=(1.0, 0.92, 0.78)):
    """compose ops for [{"matte": mode, "t0": s, "t1": s, ..options}, ...]; each line's treatment is gated to its window. `base` = the image
    buffer the ops paint on (read and written), `plate` = the (graded) footage, `matte` = the subject matte buffer, `type_a` = the type alpha
    buffer, `type_rgb` = colour or buffer. Per-mode options:
      front   split 'x'|'y'|'diag'|'depth' ('depth' needs front='$near'), side 'left'|'right'|'top'|'bottom', pos 0.5 | [[t, v]..], soft 0.06,
              front '$buf', opacity 0.96
      window  dim 0.28 (plate gain while on), gain 1.3 (the window plate), sat 1.1, subject_out true, min_contrast 0.22 (window luma over the
              surround's, enforced by compose.knockout: surround darkened, then solid type), face '$buf'
      fill    outside 0.12 (ghost strength outside the silhouette), dim 0.45 (subject darkening under the letters), gain 1.6
      wipe    reveal 0.7 (s the front takes to grow), reach 0.8, soft 0.05, brighten 0.55, ring 0.9 (glow), opacity 0.96
      glow    mode 'outline'|'scribble', width 2, radius 18, intensity 0.8 | {"curve": "onset", "lo": .., "hi": ..}, colour [r, g, b], line 0.9
    """
    out = []
    B = "$" + base
    for k, tr in enumerate(treatments):
        mode = tr.get("matte") or tr.get("mode")
        if mode not in MODES:
            raise ValueError(f"unknown matte treatment {mode!r}; one of {MODES}")
        t0, t1 = float(tr["t0"]), float(tr["t1"])
        g = gate(t0, t1, float(tr.get("fade", 0.15)))
        col = tr.get("colour") or tr.get("color")
        trgb = list(col) if col is not None else (list(type_rgb) if not isinstance(type_rgb, str) else type_rgb)
        n = f"_{mode[:2]}{k}"
        if mode == "front":
            split = tr.get("split", "x")
            if tr.get("front") or split == "depth":
                fr = tr.get("front", "$near")
            else:
                fr = "$" + n + "f"
                out.append({"op": "split_region", "out": n + "f", "args": {
                    "like": plate, "axis": split, "pos": _timeline_like(tr.get("pos"), 0.5), "soft": float(tr.get("soft", 0.06)),
                    "side": tr.get("side", "left"), "angle": float(tr.get("angle", 35.0))}})
            op_ = tr.get("opacity", 0.96)
            out.append({"op": "sandwich", "out": base, "args": {
                "footage": B, "fg": plate, "type_rgb": trgb, "type_a": type_a, "matte": matte, "matte_threshold": 0.5, "matte_soft": 0.08,
                "front": fr, "type_opacity": _scale_gate(g, op_)}})
        elif mode == "window":
            out += [
                {"op": "grade", "out": n + "d", "args": {"frame": plate, "gain": float(tr.get("dim", 0.28)), "sat": 0.55}},
                {"op": "crossfade", "out": n + "p", "args": {"a": plate, "b": "$" + n + "d", "t": g}},
                {"op": "grade", "out": n + "b", "args": {"frame": plate, "gain": float(tr.get("gain", 1.3)), "gamma": 1.06,
                                                         "sat": float(tr.get("sat", 1.1))}},
                {"op": "mask_gate", "out": n + "a", "args": {"a": type_a, "gate": g}}]
            kn = {"plate_rgb": "$" + n + "p", "type_a": "$" + n + "a", "footage": "$" + n + "b",
                  "min_contrast": float(tr.get("min_contrast", 0.22))}        # the window must read: darken the surround / fall back to solid type
            if tr.get("face"):
                kn["face"] = tr["face"]                                       # type colour of the solid fallback (recipe_eval.legibility_pass fills it)
            if tr.get("subject_out", True):
                out.append({"op": "legible", "out": n + "m", "args": {"type_a": type_a, "matte": matte}})
                kn.update(fg_matte="$" + n + "m", fg=plate)
            out += [{"op": "knockout", "out": n + "k", "args": kn},
                    {"op": "crossfade", "out": base, "args": {"a": B, "b": "$" + n + "k", "t": g}}]
        elif mode == "fill":
            out += [
                {"op": "mask_mul", "out": n + "a", "args": {"a": type_a, "b": matte, "floor": float(tr.get("outside", 0.12)),
                                                            "gain": float(tr.get("gain", 1.6))}},
                {"op": "mask_gate", "out": n + "a", "args": {"a": "$" + n + "a", "gate": g, "scale": float(tr.get("opacity", 0.96))}},
                {"op": "mask_mul", "out": n + "u", "args": {"a": type_a, "b": matte, "floor": 0.0, "gain": float(tr.get("gain", 1.6))}},
                {"op": "mask_gate", "out": n + "u", "args": {"a": "$" + n + "u", "gate": g, "scale": float(tr.get("dim", 0.45))}},
                {"op": "over", "out": base, "args": {"base": B, "fg": [0.0, 0.0, 0.0], "alpha": "$" + n + "u"}},
                {"op": "over", "out": base, "args": {"base": B, "fg": trgb, "alpha": "$" + n + "a"}}]
        elif mode == "wipe":
            rv = float(tr.get("reveal", 0.7))
            prog = _timeline_like(tr.get("progress"), _tl([(t0 - 0.2, 0.0), (t0, 0.0), (t0 + rv, 1.0), (t1, 1.0), (t1 + 0.2, 1.0)], "smooth"))
            reach, soft = float(tr.get("reach", 0.8)), float(tr.get("soft", 0.05))
            out += [
                {"op": "matte_reach", "out": n + "r", "args": {"matte": matte, "progress": prog, "reach": reach, "soft": soft}},
                {"op": "mask_mul", "out": n + "a", "args": {"a": type_a, "b": "$" + n + "r"}},
                {"op": "grade", "out": n + "b", "args": {"frame": plate, "gain": 1.22, "gamma": 1.05, "sat": 1.15}},
                {"op": "mask_gate", "out": n + "t", "args": {"a": "$" + n + "r", "gate": g, "scale": float(tr.get("brighten", 0.55))}},
                {"op": "crossfade", "out": base, "args": {"a": B, "b": "$" + n + "b", "t": "$" + n + "t"}},
                {"op": "sandwich", "out": base, "args": {
                    "footage": B, "fg": plate, "type_rgb": trgb, "type_a": "$" + n + "a", "matte": matte, "matte_threshold": 0.5,
                    "matte_soft": 0.08, "type_opacity": _scale_gate(g, tr.get("opacity", 0.96))}},
                {"op": "matte_ring", "out": n + "g", "args": {"matte": matte, "progress": prog, "reach": reach,
                                                              "width": float(tr.get("ring_w", 0.012))}},
                {"op": "mask_gate", "out": n + "g", "args": {"a": "$" + n + "g", "gate": g}},
                *([{"op": "add_glow", "out": base, "args": {"img": B, "mask": "$" + n + "g", "radius": float(tr.get("radius", 14)),
                                                            "intensity": float(tr.get("ring", 0.9)), "color": trgb}}] if _edge_glow(tr) else [])]
        else:                                                                                  # glow
            line = {"op": "contour_lines", "out": n + "c", "args": {
                "src": matte, "mode": tr.get("mode_lines", "outline"), "width": int(tr.get("width", 2)), "smooth": 3.0, "min_len": 60.0,
                "max_contours": int(tr.get("max_contours", 2)), "strokes": 2, "jitter": 4.0, "frame_idx": "$frame_idx", "seed": 1 + k}}
            out += [line,
                    {"op": "mask_gate", "out": n + "c", "args": {"a": "$" + n + "c", "gate": g}},
                    *([{"op": "add_glow", "out": base, "args": {"img": B, "mask": "$" + n + "c", "radius": float(tr.get("radius", 18)),
                                                                "intensity": tr.get("intensity", 0.8), "color": trgb}}] if _edge_glow(tr) else []),
                    {"op": "over", "out": base, "args": {"base": B, "fg": trgb, "alpha": "$" + n + "c"}}]
    return out


def _scale_gate(g, k):
    """gate timeline scaled by a constant (type_opacity of sandwich)."""
    k = float(k)
    if k == 1.0:
        return g
    return {"timeline": [[t, v * k] for t, v in g["timeline"]], "ease": g.get("ease", "smooth")}


def from_words(words, windows=None, **defaults):
    """treatments from word specs: a word with `"matte": "front"` (or {"mode": .., ..options}) gives one treatment over its window. The window
    is the word's t0 / t1 unless `windows[i] = (t0, t1)` (e.g. hit .. release from the timeline) is given. Words without `matte` give nothing."""
    out = []
    for i, w in enumerate(words):
        m = w.get("matte")
        if not m:
            continue
        d = {"matte": m} if isinstance(m, str) else dict(m)
        if "matte" not in d:
            d["matte"] = d.pop("mode")
        t0, t1 = windows[i] if windows else (w["t0"], w["t1"])
        out.append(dict(defaults, **d, t0=float(t0), t1=float(t1)))
    return out
