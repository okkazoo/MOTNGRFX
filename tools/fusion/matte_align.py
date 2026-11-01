"""matte_align: a matte (or depth) is only ever laid over the exact picture it was cut from, frame for frame (user rule 2026-10-29: "if we are morphing anything that has an alpha we need to
do a separate morph on the alpha that matches that morph, so we dont get double ups where we see two people"). Pure functions, numpy + cv2, no ComfyUI, no GPU.

Three things can pull a matte off its picture inside compose, and each has a function here:

  1. GEOMETRY. The Alpha-Gen / depth references are cut from the clip with `footage._u8_to_ref` = `cover_resize` (scale to cover the reference size, centre-crop). The matte that comes
     back covers only that CROPPED part of the clip. Compose used to stretch it over the whole frame (`upscale_matte` resizes to the compose size) while the footage was only
     cover-fitted, so the subject drifted by up to half the crop at the frame edges (batch-24gb v003 r02 s1: 384x704 matte of a 360x640 clip = 5.5 px of 360 at the edges; measured
     on 4 frames the edge-agreement of the matte with the picture is 61 / 57 / 58 / 54 with the inverse cover and 56 / 53 / 54 / 52 with the stretch). `ref_to_frame` cuts the window of
     the matte that is the compose frame (replicating the matte's own border where the crop left nothing), so `upscale_matte` stretches the right thing.
  2. TIME. A matte clip with another frame count than its footage was clamped at its last frame (`hold`), which shifts every frame after the difference. `frame_index` maps a footage
     frame to the matte frame at the same fraction of the clip.
  3. THE TWO PICTURES OF A LAYERED SHOT. The matte of a three-layer shot is cut from the CLEAN clip, but the background under the type is the text-rendered clip (union-control V2V of the
     clean one), which re-draws the scene and can shift the subject: the clean subject pasted over a shifted background shows two people. `subject_drift` measures the shift of the
     subject between the two clips; `ltx_check.choose_background` falls back to the clean clip (bg == subject == the picture the matte came from) where it is over the limit or no
     LTX letters survived.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DRIFT_MAX = 0.012            # fraction of the short side: more subject shift than this between the clean and the text-rendered clip = the matte no longer fits the background
DRIFT_FRAMES = 6             # frames measured
DRIFT_MIN_COVER = 0.0008     # a frame counts when the matte covers at least this share of the picture (a skater in a wide shot is 0.4 %)
DRIFT_SEARCH = 0.15          # search window round the subject, fraction of the short side
DRIFT_MIN_RESPONSE = 0.25    # template-match score below this = no reliable measurement (ok None: the text clip is then NOT trusted as the background, ltx_check.choose_background)


# --------------------------------------------------------------------------------------------------------------- geometry
def ref_scale(ref_hw, footage_hw, size_wh) -> float:
    """reference pixels per compose pixel when the reference is the footage cover-resized to its own size and the compose frame is the footage cover-fitted to `size_wh`."""
    mh, mw = ref_hw
    fh, fw = footage_hw
    cw, ch = size_wh
    s = max(mw / float(fw), mh / float(fh))       # footage -> reference (cover)
    big = max(cw / float(fw), ch / float(fh))     # footage -> compose (cover)
    return s / big


def canvas_hw(ref_hw, footage_hw, size_wh) -> tuple[int, int]:
    """(h, w) in reference pixels of the window of the reference that is the compose frame. Equal to ref_hw (within a pixel) when the reference covers the frame exactly."""
    k = ref_scale(ref_hw, footage_hw, size_wh)
    ch_, cw_ = int(round(size_wh[1] * k)), int(round(size_wh[0] * k))
    if abs(ch_ - ref_hw[0]) <= 1 and abs(cw_ - ref_hw[1]) <= 1:
        return int(ref_hw[0]), int(ref_hw[1])
    return max(ch_, 1), max(cw_, 1)


def ref_to_frame(ref, footage_hw, size_wh):
    """one reference frame (matte / depth, HxW or HxWxC, numpy) -> the window of it that is the compose frame, at reference resolution (pad = replicate the border, crop = centred).
    The caller stretches the result to the compose size (upscale_matte / fit_cover)."""
    a = np.asarray(ref)
    h, w = a.shape[:2]
    th, tw = canvas_hw((h, w), footage_hw, size_wh)
    if (th, tw) == (h, w):
        return a
    pads, sl = [], []
    for have, want in ((h, th), (w, tw)):
        if want > have:
            lo = (want - have) // 2
            pads.append((lo, want - have - lo))
            sl.append(slice(0, want))
        else:
            pads.append((0, 0))
            o = (have - want) // 2
            sl.append(slice(o, o + want))
    if any(p for pp in pads for p in pp):
        a = np.pad(a, tuple(pads) + ((0, 0),) * (a.ndim - 2), mode="edge")
    return a[sl[0], sl[1]]


def frame_index(j: int, n_ref: int, n_foot: int) -> int:
    """matte frame for footage frame j: the same index when the clips have the same length, else the frame at the same fraction of the clip."""
    if n_ref == n_foot or n_foot <= 1:
        return min(max(int(j), 0), max(n_ref - 1, 0))
    return min(max(int(round(int(j) * (n_ref - 1) / float(n_foot - 1))), 0), n_ref - 1)


# --------------------------------------------------------------------------------------------------------------- two pictures of a layered shot
def _gray_u8(f):
    f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
    if f.dtype != np.uint8:
        f = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(f[..., :3]), cv2.COLOR_RGB2GRAY)


def _matte_u8(m, hw):
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = np.clip(np.rint(m * 255.0), 0, 255).astype(np.uint8)
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) if m.shape[:2] != tuple(hw) else m


def subject_drift(clean, bg, matte, n: int = DRIFT_FRAMES) -> dict:
    """how far the subject of the background clip `bg` (the text-rendered clip) sits from the subject of `clean` (the clip the matte was cut from), in pixels of the frame.
    Template matching: the clean clip's subject (the matte's box plus a margin, grey) is searched for in `bg` within +-DRIFT_SEARCH of the short side, on up to `n` evenly spaced frames; the
    median offset of the best match is the drift. (Phase correlation inside the matte region missed small subjects: v005 r01 s5, a skater under 1 % of the frame that the text pass moved by
    ~24 px of 360 = two skaters side by side.) clean / bg: [T,H,W,3] (numpy or torch, uint8 or 0..1 float), matte: [T',H',W'(,C)] of the clean clip (any size / length: resized to the frames,
    indexed by the same fraction of the clip). -> dict(px (median shift length), dx, dy, rel (px / short side), response (median match score), n, ok (bool | None: None = no reliable
    measurement: no subject, a flat patch, or a weak match))."""
    T = min(int(clean.shape[0]), int(bg.shape[0]))
    H, W = int(clean.shape[1]), int(clean.shape[2])
    sel = sorted({int(round(k * (T - 1) / max(n - 1, 1))) for k in range(n)}) if T > 0 else []
    rows = []
    for j in sel:
        mj = _matte_u8(matte[frame_index(j, int(matte.shape[0]), T)], (H, W)) > 32
        if float(mj.mean()) < DRIFT_MIN_COVER or mj.sum() < 30:
            continue
        ys, xs = np.where(mj)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pad = max(4, int(round(0.25 * max(y1 - y0, x1 - x0))))
        y0, y1, x0, x1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        a = _gray_u8(clean[j])
        tpl = a[y0:y1, x0:x1].astype(np.float32)
        if float(tpl.std()) < 3.0:
            continue
        r = max(4, int(round(DRIFT_SEARCH * min(H, W))))
        sy0, sy1, sx0, sx1 = max(0, y0 - r), min(H, y1 + r), max(0, x0 - r), min(W, x1 + r)
        win = _gray_u8(bg[j])[sy0:sy1, sx0:sx1].astype(np.float32)
        if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        rows.append((float(sx0 + loc[0] - x0), float(sy0 + loc[1] - y0), float(score)))
    if not rows:
        return dict(px=None, dx=None, dy=None, rel=None, response=None, n=0, ok=None)
    dx, dy = float(np.median([r[0] for r in rows])), float(np.median([r[1] for r in rows]))
    resp = float(np.median([r[2] for r in rows]))
    px = math.hypot(dx, dy)
    rel = px / float(min(H, W))
    ok = None if resp < DRIFT_MIN_RESPONSE else bool(rel <= DRIFT_MAX)
    return dict(px=round(px, 2), dx=round(dx, 2), dy=round(dy, 2), rel=round(rel, 4), response=round(resp, 3), n=len(rows), ok=ok)


# --------------------------------------------------------------------------------------------------------------- reference <-> full footage frame (3D cuts)
def ref_to_full(ref, footage_hw):
    """one reference frame (matte / depth cut from the footage by a cover-resize) -> the matte of the WHOLE footage frame, at the footage's size (HxW or HxWxC)."""
    a = np.asarray(ref)
    fh, fw = int(footage_hw[0]), int(footage_hw[1])
    w = ref_to_frame(a, (fh, fw), (fw, fh))
    if w.shape[:2] == (fh, fw):
        return w
    return cv2.resize(w, (fw, fh), interpolation=cv2.INTER_LINEAR if w.shape[0] < fh else cv2.INTER_AREA)


def frame_to_ref(patch, ref_hw, footage_hw):
    """the inverse for a patch made in footage geometry (the 3D cut's matte through the transition): resized to the window of the reference that is the whole footage frame and cut / padded
    to the reference's own size, so `RefSource` (ref_to_frame) puts it back on the footage exactly. patch HxW float / uint8 -> (ref_hw) same dtype."""
    a = np.asarray(patch)
    th, tw = canvas_hw(tuple(ref_hw), tuple(footage_hw), (int(footage_hw[1]), int(footage_hw[0])))
    r = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA if th < a.shape[0] else cv2.INTER_LINEAR)
    rh, rw = int(ref_hw[0]), int(ref_hw[1])
    pads = ((max((rh - th) // 2, 0), max(rh - th - (rh - th) // 2, 0)), (max((rw - tw) // 2, 0), max(rw - tw - (rw - tw) // 2, 0)))
    if any(p for pp in pads for p in pp):
        r = np.pad(r, pads + ((0, 0),) * (r.ndim - 2), mode="constant")
    y0, x0 = max((th - rh) // 2, 0), max((tw - rw) // 2, 0)
    return r[y0:y0 + rh, x0:x0 + rw]
