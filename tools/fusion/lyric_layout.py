"""lyric_layout.py - static typographic layouts over a still + subject matte (18_lyric-layouts, CPU, draft size).

A design is data (JSON): a plate (video frame or image), its subject matte, and a list of elements drawn in order:
  text   {text, font, axes?, cap | fit_w, x, y, ax?, ay?, rot?, track?, ink, layer?, support?, name?}
         cap = cap height as a fraction of the frame height (HOLD_CAP_FRAC 0.07 for the lead, MIN_CAP_FRAC 0.04 for small lines)
         fit_w = fit the ink width to this fraction of the frame width (cap then follows); x, y = anchor point (fractions),
         ax, ay = which point of the (rotated) ink box sits there (0..1); rot = degrees counter-clockwise; track = letter spacing in em.
  ring   {text, font, cap, cx, cy, r, a0, ink, ...} text set on a circle (r = fraction of the width, a0 = start angle deg, clockwise from top)
  rect   {cx, cy, w, h, angle, color} / poly {pts, color}: flat backing shapes (hold_layouts shape conventions)
  counter {of, letter, color}: fill the counter (hole) of letter index `letter` of text element `of` with a flat colour
  text + quad [[x,y]*4]: the line laid on a plane of the photo (perspective; raster corners TL, TR, BR, BL -> quad)
  path   {text, font, cap, pts | contour {offset, from, to}, align, shift}: text along a polyline or along the subject's offset silhouette
  reflection {of, axis, fade, blur, ripple, opacity}: wet-floor mirror of an element; cast_shadow {of, base, length, shear, opacity, tint}
layer: "behind" (default; behind the subject via compose.sandwich, solid matte) | "front" (over the subject).
ink: the DESIGN colour; the drawn colour is chosen against the MEASURED background under + around the letters by
legible_color.choose (WCAG target by cap height). If no single colour passes, the support it asks for (outline / shadow / scrim) is drawn.
Fonts: only mograph/fonts/liked.json (fonts_liked.require_liked). Text rendering is PIL with the liked font files (static stills; the
type engine's 3D passes are overkill for flat layout studies).

CLI:  python -m tools.fusion.lyric_layout render <designs.json> [--only ID] [--w 540]
      python -m tools.fusion.lyric_layout sheet <designs.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mograph import fonts_liked as FL
from tools.fusion import compose as C
from tools.fusion import legible_color as LC
from tools.fusion.hold_layouts import HOLD_CAP_FRAC, MIN_CAP_FRAC

REPO = Path(__file__).resolve().parents[2]
FONT_DIR = REPO / "mograph" / "fonts"
SS = 2                      # supersampling of the type raster


# ------------------------------------------------------------------------------------------------ inputs
def load_plate(src, frame, W, H):
    p = str(REPO / src) if not os.path.isabs(src) else src
    if p.lower().endswith((".mp4", ".mov", ".webm")):
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {frame} of {p}")
    else:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img[..., ::-1].astype(np.float32) / 255.0


def load_matte(src, W, H, thr=0.5, soft=0.08):
    p = str(REPO / src) if not os.path.isabs(src) else src
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else m[..., 0]
    m = cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    return C._smoothstep(m, thr - soft / 2, thr + soft / 2)


# ------------------------------------------------------------------------------------------------ fonts / type raster
def font(key, px, axes=None):
    e = FL.require_liked(key)                       # HARD rule: liked fonts only
    f = ImageFont.truetype(str(FONT_DIR / e["file"]), max(4, int(round(px))))
    ax = dict(e.get("axes") or {})
    ax.update(axes or {})
    if ax:
        try:
            names = [a["name"].decode() if isinstance(a["name"], bytes) else a["name"] for a in f.get_variation_axes()]
            tag = {"wght": "Weight", "wdth": "Width", "ROND": "Roundness", "ELSH": "Element Shape", "MORF": "Morph", "SHLN": "Shadow Length",
                   "slnt": "Slant", "CRSV": "Cursive", "ELXP": "Element Expansion"}
            vals = []
            for a, n in zip(f.get_variation_axes(), names):
                v = a.get("default", a["minimum"])
                for k, x in ax.items():
                    if tag.get(k, k).lower() == n.lower() or k.lower() == n.lower():
                        v = x
                vals.append(min(max(v, a["minimum"]), a["maximum"]))
            f.set_variation_by_axes(vals)
        except OSError:
            pass
    return f, e


def cap_ratio(key, axes=None):
    f, _ = font(key, 200, axes)
    b = f.getbbox("H")
    return (b[3] - b[1]) / 200.0


def _line_raster(text, f, track_px):
    """L image of one line with tracking + per-glyph x boxes (canvas coords)."""
    asc, desc = f.getmetrics()
    widths = [f.getlength(ch) for ch in text]
    tw = int(sum(widths) + track_px * max(0, len(text) - 1)) + 4 * f.size
    img = Image.new("L", (tw, asc + desc + 2 * f.size), 0)
    d = ImageDraw.Draw(img)
    x, y0, boxes = f.size, f.size, []
    for ch, w in zip(text, widths):
        d.text((x, y0), ch, font=f, fill=255)
        boxes.append((x, x + w))
        x += w + track_px
    return img, boxes


def text_layer(el, W, H):
    """-> (alpha HxW float, meta). Supersampled, rotated, anchored."""
    key, axes = el["font"], el.get("axes")
    track = float(el.get("track", 0.0))
    if "fit_w" in el:
        f0, _ = font(key, 200 * SS, axes)
        im0, _ = _line_raster(el["text"], f0, track * 200 * SS)
        bb = im0.getbbox()
        px = 200 * SS * (el["fit_w"] * W * SS) / max(1, bb[2] - bb[0])
    else:
        px = el["cap"] * H * SS / cap_ratio(key, axes)
    f, entry = font(key, px, axes)
    im, gboxes = _line_raster(el["text"], f, track * px)
    bb = im.getbbox()
    im = im.crop(bb)
    gboxes = [(a - bb[0], b - bb[0]) for a, b in gboxes]
    rot = float(el.get("rot", 0.0))
    w0, h0 = im.size
    imr = im.rotate(rot, resample=Image.BICUBIC, expand=True) if rot else im
    wr, hr = imr.size
    ax, ay = float(el.get("ax", 0.5)), float(el.get("ay", 0.5))
    X = int(round(el["x"] * W * SS - ax * wr))
    Y = int(round(el["y"] * H * SS - ay * hr))
    if "glyph_at" in el and not rot:                   # put the centre of glyph i at (gx, gy): e.g. an O's counter over the sun
        gi, gx, gy = el["glyph_at"]
        a_, b_ = gboxes[int(gi)]
        X = int(round(gx * W * SS - (a_ + b_) / 2))
        Y = int(round(gy * H * SS - h0 / 2))
    big = Image.new("L", (W * SS, H * SS), 0)
    cap_px = cap_ratio(key, axes) * px / SS
    if "quad" in el:                                   # lay the line on a plane of the photo: raster corners TL,TR,BR,BL -> quad (frame fractions)
        q = np.float32([[x * W * SS, y * H * SS] for x, y in el["quad"]])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]), q)
        warped = cv2.warpPerspective(np.asarray(im, np.uint8), M, (W * SS, H * SS), flags=cv2.INTER_CUBIC)
        big = Image.fromarray(warped)
        hl, hr_ = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
        cap_px = cap_px * max(hl, hr_) / (h0 * 1.0)  # cap at the near end of the plane
        X, Y, wr, hr = int(q[:, 0].min()), int(q[:, 1].min()), int(np.ptp(q[:, 0])), int(np.ptp(q[:, 1]))
    else:
        big.paste(imr, (X, Y))
    a = np.asarray(big.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    meta = dict(font=entry["key"], role=entry["role"], px=px / SS, cap_px=cap_px, cap_frac=cap_px / H, rot=rot,
                ink_box=(X / SS, Y / SS, wr / SS, hr / SS), size=(w0, h0), glyphs=gboxes, origin=(X, Y), rot_img=(wr, hr), raster=im)
    return np.clip(a, 0, 1), meta


def ring_layer(el, W, H):
    key, axes = el["font"], el.get("axes")
    px = el["cap"] * H * SS / cap_ratio(key, axes)
    f, entry = font(key, px, axes)
    cx, cy, r = el["cx"] * W * SS, el["cy"] * H * SS, el["r"] * W * SS
    big = Image.new("L", (W * SS, H * SS), 0)
    track = float(el.get("track", 0.0)) * px
    ang = math.radians(float(el.get("a0", 0.0)))
    for ch in el["text"]:
        w = f.getlength(ch)
        ang += (w / 2) / r
        g = Image.new("L", (int(w + px), int(px * 1.6)), 0)
        ImageDraw.Draw(g).text((px / 2, px * 0.8), ch, font=f, fill=255, anchor="ms")
        g = g.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
        x, y = cx + r * math.sin(ang), cy - r * math.cos(ang)
        big.paste(g, (int(x - g.size[0] / 2), int(y - g.size[1] / 2)), g)
        ang += (w / 2 + track) / r
    a = np.asarray(big.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    cap_px = cap_ratio(key, axes) * px / SS
    return np.clip(a, 0, 1), dict(font=entry["key"], role=entry["role"], cap_px=cap_px, cap_frac=cap_px / H)


def matte_path(hide, offset, p_from, p_to, W, H, smooth=9):
    """outer contour of the subject matte pushed out by `offset` (fraction of W): the arc from the point nearest p_from to the point
    nearest p_to, in the direction whose text up-normal points AWAY from the subject. -> Nx2 float px."""
    m = (hide > 0.5).astype(np.uint8)
    k = max(1, int(offset * W))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = cv2.morphologyEx(cv2.dilate(m, se), cv2.MORPH_CLOSE, se)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    ker = np.ones(smooth, np.float32) / smooth
    c = np.stack([np.convolve(np.r_[c[-smooth:, i], c[:, i], c[:smooth, i]], ker, "same")[smooth:-smooth] for i in (0, 1)], 1)
    fa = np.float32([p_from[0] * W, p_from[1] * H])
    fb = np.float32([p_to[0] * W, p_to[1] * H])
    i0, i1 = int(np.argmin(((c - fa) ** 2).sum(1))), int(np.argmin(((c - fb) ** 2).sum(1)))
    n = len(c)
    fwd = c[np.arange(i0, i0 + ((i1 - i0) % n) + 1) % n]
    bwd = c[np.arange(i0, i0 - ((i0 - i1) % n) - 1, -1) % n]

    def outward(pp):                                    # share of points whose text-up side is off the subject
        t = np.gradient(pp, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
        up = np.stack([t[:, 1], -t[:, 0]], 1)           # text up = left of the direction of travel in a y-down frame
        q = (pp + up * (3 * k + 4)).astype(int)
        q[:, 0] = q[:, 0].clip(0, W - 1)
        q[:, 1] = q[:, 1].clip(0, H - 1)
        return 1.0 - m[q[:, 1], q[:, 0]].mean()
    return fwd if outward(fwd) >= outward(bwd) else bwd


def path_layer(el, W, H, hide):
    """text set glyph by glyph along a polyline (el['pts'], frame fractions) or along the subject's offset silhouette (el['contour'])."""
    key, axes = el["font"], el.get("axes")
    px = el["cap"] * H * SS / cap_ratio(key, axes)
    f, entry = font(key, px, axes)
    if "contour" in el:
        cc = el["contour"]
        P = matte_path(hide, cc["offset"], cc["from"], cc["to"], W, H, int(cc.get("smooth", 9))) * SS
    else:
        P = np.float32([[x * W * SS, y * H * SS] for x, y in el["pts"]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.r_[0, np.cumsum(seg)]
    track = float(el.get("track", 0.0)) * px
    widths = [f.getlength(ch) for ch in el["text"]]
    total = sum(widths) + track * (len(widths) - 1)
    s0 = {"start": 0.0, "center": (L[-1] - total) / 2, "end": L[-1] - total}[el.get("align", "start")] + float(el.get("shift", 0.0)) * L[-1]
    big = Image.new("L", (W * SS, H * SS), 0)
    pos = s0
    for ch, w in zip(el["text"], widths):
        sc = pos + w / 2
        pos += w + track
        if sc < 0 or sc > L[-1] or ch == " ":
            continue
        j = int(np.clip(np.searchsorted(L, sc) - 1, 0, len(seg) - 1))
        u = (sc - L[j]) / max(seg[j], 1e-6)
        x, y = P[j] + (P[j + 1] - P[j]) * u
        d = P[min(len(P) - 1, j + 7)] - P[max(0, j - 6)]
        ang = math.degrees(math.atan2(d[1], d[0]))
        g = Image.new("L", (int(w + px), int(px * 1.6)), 0)
        ImageDraw.Draw(g).text((g.size[0] / 2, px * 0.8), ch, font=f, fill=255, anchor="ms")
        g = g.rotate(-ang, resample=Image.BICUBIC, expand=True)
        big.paste(g, (int(x - g.size[0] / 2), int(y - g.size[1] / 2)), g)
    a = np.asarray(big.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    cap_px = cap_ratio(key, axes) * px / SS
    return np.clip(a, 0, 1), dict(font=entry["key"], role=entry["role"], cap_px=cap_px, cap_frac=cap_px / H, fits=bool(total <= L[-1]))


def reflect_alpha(a, axis_y, fade, blur, ripple, H, W, seed=3):
    """mirror an element's alpha about y = axis_y (fraction), fade it with distance, smear it vertically (wet floor), ripple it."""
    ay = int(axis_y * H)
    r = np.zeros_like(a)
    top = a[:ay][::-1]
    n = min(len(top), H - ay)
    r[ay:ay + n] = top[:n]
    if ripple:
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3) * float(ripple) * W * 3.0
        r = cv2.remap(r, xx + dx.astype(np.float32), yy, cv2.INTER_LINEAR)
    if blur:
        r = cv2.GaussianBlur(r, (0, 0), sigmaX=blur * W * 0.25, sigmaY=blur * W)
    d = np.clip((np.arange(H, dtype=np.float32) - ay) / max(1.0, fade * H), 0, 1)
    return r * (1.0 - d)[:, None] * (np.arange(H) >= ay)[:, None]


def cast_alpha(a, base_y, length, shear, blur, H, W):
    """a long cast shadow on the ground: an element flipped about its base line y = base_y, stretched by `length`, sheared sideways by
    `shear` (x shift per px below the base), softened."""
    by = base_y * H
    # dst (x', y') = (x + shear*(y'-by), by + length*(by - y))  -> inverse map for warpAffine (WARP_INVERSE_MAP): src from dst
    Minv = np.float32([[1, -shear, shear * by], [0, -1.0 / length, by + by / length]])
    r = cv2.warpAffine(a, Minv, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    r *= (np.arange(H) >= by)[:, None]
    if blur:
        r = cv2.GaussianBlur(r, (0, 0), blur * W)
    return r


def shape_layer(el, W, H):
    big = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(big)
    if el["kind"] == "rect":
        cx, cy, w, h = el["cx"] * W, el["cy"] * H, el["w"] * W, el["h"] * H
        a = math.radians(el.get("angle", 0.0))
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = sx * w / 2, sy * h / 2
            pts.append(((cx + x * math.cos(a) - y * math.sin(a)) * SS, (cy + x * math.sin(a) + y * math.cos(a)) * SS))
    else:
        pts = [(x * W * SS, y * H * SS) for x, y in el["pts"]]
    d.polygon(pts, fill=255)
    return np.asarray(big.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def counter_mask(meta, letter, W, H):
    """hole(s) of glyph `letter` of a rendered (unrotated) text element, frame coords."""
    im = np.asarray(meta["raster"], np.uint8)
    a, b = meta["glyphs"][letter]
    a, b = int(max(0, a)), int(min(im.shape[1], b))
    g = (im[:, a:b] > 110).astype(np.uint8)
    ff = np.pad(1 - g, 1, constant_values=1).copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    hole = (ff[1:-1, 1:-1] == 1).astype(np.uint8) * 255
    full = np.zeros(im.shape, np.uint8)
    full[:, a:b] = hole
    full = cv2.dilate(full, np.ones((3, 3), np.uint8))           # tuck under the glyph's antialiased rim
    X, Y = meta["origin"]
    big = np.zeros((H * SS, W * SS), np.uint8)
    h, w = full.shape
    x0, y0, x1, y1 = max(0, X), max(0, Y), min(W * SS, X + w), min(H * SS, Y + h)
    big[y0:y1, x0:x1] = full[y0 - Y:y1 - Y, x0 - X:x1 - X]
    return cv2.resize(big.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------ colour
def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def pick_ink(bg, alpha, hide, design_hex, cap_px, W, H):
    """legible_color.choose on the measured background (letters weight 2, ring weight 1, pixels the subject hides excluded)."""
    lin = LC.srgb_to_lin(bg)
    y = LC.rel_lum(lin)
    core = (alpha > 0.25).astype(np.uint8)
    rad = max(2, int(LC.LC_RING * max(W, H)))
    ring = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))) - core
    vis = (hide < 0.5).astype(np.float32)
    w = (core * LC.LC_BEHIND_W + ring) * vis
    if w.sum() < 10:
        w = core * LC.LC_BEHIND_W + ring
    hist = LC._hist(y, w)
    bg_lin = (lin * w[..., None]).sum((0, 1)) / max(1e-6, w.sum())
    cap_norm = cap_px * 1080.0 / min(W, H)
    target = LC.target_for(cap_norm)
    ch = LC.choose(hist, LC.srgb_to_lin(hex_rgb(design_hex)), bg_lin, target)
    rgb = LC.lin_to_srgb(ch["lin"])
    y_text = float(LC.rel_lum(ch["lin"]))
    return dict(rgb=rgb, hex=LC.hex_of(rgb), kind=ch["kind"], cover=round(ch["cover"], 3), support=ch["support"],
                target=target, bg_hex=LC.hex_of(LC.lin_to_srgb(bg_lin)), dark_text=y_text < 0.18)


def support_layer(alpha, kind, W):
    a8 = (alpha * 255).astype(np.uint8)
    if kind == "outline":
        k = max(2, W // 270)
        return cv2.dilate(a8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(np.float32) / 255.0, 1.0
    if kind == "shadow":
        s = cv2.GaussianBlur(cv2.dilate(a8, np.ones((3, 3), np.uint8)), (0, 0), W / 180.0).astype(np.float32) / 255.0
        return np.clip(s * 1.6, 0, 1), 0.75
    s = cv2.GaussianBlur(cv2.dilate(a8, np.ones((W // 30, W // 30), np.uint8)), (0, 0), W / 60.0).astype(np.float32) / 255.0
    return np.clip(s * 1.3, 0, 1), 0.55


def over(img, rgb, a):
    return C._lerp(img, C._color(rgb, img), C._c3(np.clip(a, 0, 1)))


# ------------------------------------------------------------------------------------------------ render one design
def render(design, plate_cfg, W=540):
    H = int(round(W * 16 / 9))
    plate = load_plate(plate_cfg["plate"], plate_cfg.get("frame", 0), W, H)
    hide = load_matte(plate_cfg["matte"], W, H, plate_cfg.get("matte_threshold", 0.5))
    under = plate.copy()          # everything behind the subject
    fronts = []
    metas, report = {}, []
    for i, el in enumerate(design["elements"]):
        kind = el.get("kind", "text")
        name = el.get("name", f"e{i}")
        layer = el.get("layer", "behind")
        if kind in ("rect", "poly"):
            a = shape_layer(el, W, H)
            col = hex_rgb(el["color"]) * float(el.get("opacity_rgb", 1.0))
            if layer == "front":
                fronts.append((col, a * el.get("opacity", 1.0), None))
            else:
                under = over(under, col, a * el.get("opacity", 1.0))
            report.append(dict(name=name, kind=kind, color=el["color"]))
            continue
        if kind == "counter":
            a = counter_mask(metas[el["of"]], int(el["letter"]), W, H)
            under = over(under, hex_rgb(el["color"]), a)
            report.append(dict(name=name, kind=kind, color=el["color"]))
            continue
        if "below" in el or "above" in el:             # stack under / over an earlier element: gap = fraction of the frame height
            ref = metas[el.get("below") or el.get("above")]["ink_box"]
            el = dict(el)
            if "below" in el:
                el["y"], el["ay"] = (ref[1] + ref[3]) / H + float(el.get("gap", 0.012)), 0.0
            else:
                el["y"], el["ay"] = ref[1] / H - float(el.get("gap", 0.012)), 1.0
        if kind in ("reflection", "cast_shadow"):     # the photo's own light acting on an earlier element (wet floor / low sun)
            src = metas[el["of"]]
            if kind == "reflection":
                ra = reflect_alpha(src["alpha"], el["axis"], el.get("fade", 0.25), el.get("blur", 0.01), el.get("ripple", 0.004), H, W)
                under = over(under, src["ink_rgb"], ra * el.get("opacity", 0.45))
            else:
                ra = cast_alpha(src["alpha"], el["base"], el.get("length", 1.6), el.get("shear", 0.4), el.get("blur", 0.004), H, W)
                under = under * (1.0 - C._c3(ra * el.get("opacity", 0.55)) * (1.0 - hex_rgb(el.get("tint", "#000000"))))
            report.append(dict(name=name, kind=kind, of=el["of"]))
            continue
        a, meta = (ring_layer(el, W, H) if kind == "ring" else path_layer(el, W, H, hide) if kind == "path" else text_layer(el, W, H))
        metas[name] = meta
        bg = under if layer == "behind" else C.sandwich(plate, under, np.ones((H, W), np.float32), hide)
        ink = pick_ink(bg, a, hide if layer == "behind" else np.zeros_like(hide), el["ink"], meta["cap_px"], W, H)
        if el.get("color") == "fixed":                 # legible_color opt-out (same key as compose): keep the design ink, still report + support
            rgb = hex_rgb(el["ink"])
            ink.update(rgb=rgb, hex=el["ink"], kind="fixed", dark_text=float(LC.rel_lum(LC.srgb_to_lin(rgb))) < 0.18)
        sup = el.get("support", "auto")
        sup = ink["support"] if sup == "auto" else sup
        target = under if layer == "behind" else None
        sup_rgb = np.full(3, LC.LC_SUP_LIGHT if ink["dark_text"] else LC.LC_SUP_DARK, np.float32)
        ops = []
        if sup:
            sa, op = support_layer(a, sup, W)
            ops.append((sup_rgb, sa * op))
        if el.get("shade"):                            # paint on a surface: the ink takes the plate's light (wall falloff, shadows on the road)
            lum = C._luma(plate)
            ref = float((lum * a).sum() / max(1e-6, a.sum()))
            k = np.clip((lum / max(ref, 1e-3)) ** float(el["shade"]), 0.35, 1.6)[..., None]
            ops.append((np.clip(ink["rgb"][None, None, :] * k, 0, 1).astype(np.float32), a))
        else:
            ops.append((ink["rgb"], a))
        meta["alpha"], meta["ink_rgb"] = a, ink["rgb"]
        for rgb, aa in ops:
            if layer == "behind":
                under = over(under, rgb, aa)
            else:
                fronts.append((rgb, aa, None))
        hidden = float((a * hide).sum() / max(1e-6, a.sum()))
        report.append(dict(name=name, kind=kind, text=el["text"], font=meta["font"], role=meta["role"], cap_frac=round(meta["cap_frac"], 4),
                           layer=layer, design_ink=el["ink"], ink=ink["hex"], ink_kind=ink["kind"], cover=ink["cover"], target=ink["target"],
                           bg=ink["bg_hex"], support=sup, hidden_by_runner=round(hidden, 3),
                           cap_ok=bool(meta["cap_frac"] >= (HOLD_CAP_FRAC if el.get("lead") else MIN_CAP_FRAC) - 1e-4)))
    # behind the runner: compose.sandwich, solid matte (already thresholded)
    out = C.sandwich(plate, under, np.ones((H, W), np.float32), hide)
    for rgb, aa, _ in fronts:
        out = over(out, rgb, aa)
    return np.clip(out, 0, 1), report


# ------------------------------------------------------------------------------------------------ sheet
def _label_font(px):
    from mograph import fonts_liked as _fl
    return _fl.truetype("SpaceGrotesk-Bold", px)


def sheet(doc, root):
    rows = []
    tw = 270
    for img_id in ("img1", "img2"):
        tiles = []
        for d in [d for d in doc["designs"] if d["image"] == img_id]:
            p = root / "out" / d["file"]
            im = Image.open(p).convert("RGB").resize((tw, int(tw * 16 / 9)), Image.LANCZOS)
            cap = Image.new("RGB", (tw, 120), (16, 16, 18))
            dr = ImageDraw.Draw(cap)
            dr.text((6, 4), f"{d['id']}  {d['title']}", font=_label_font(15), fill=(255, 150, 110))
            y = 26
            for chunk in _wrap(f"layout: {d['layout']}", 46)[:3]:
                dr.text((6, y), chunk, font=_label_font(11), fill=(200, 200, 200)); y += 14
            fonts = " + ".join(dict.fromkeys(e["font"] for e in d["elements"] if e.get("kind", "text") in ("text", "ring", "path")))
            for chunk in _wrap(f"fonts: {fonts}", 44):
                dr.text((6, y), chunk, font=_label_font(11), fill=(170, 170, 170)); y += 14
            dr.text((6, y), f"pairs with: {d.get('pairs', '-')}", font=_label_font(11), fill=(140, 200, 255))
            t = Image.new("RGB", (tw, im.size[1] + 120), (16, 16, 18))
            t.paste(im, (0, 0)); t.paste(cap, (0, im.size[1]))
            tiles.append(t)
        row = Image.new("RGB", (sum(t.size[0] for t in tiles) + 8 * (len(tiles) + 1), tiles[0].size[1] + 16), (8, 8, 10))
        x = 8
        for t in tiles:
            row.paste(t, (x, 8)); x += t.size[0] + 8
        rows.append(row)
    W = max(r.size[0] for r in rows)
    head = Image.new("RGB", (W, 40), (8, 8, 10))
    ImageDraw.Draw(head).text((10, 8), doc.get("sheet_title", "lyric layouts"), font=_label_font(18), fill=(235, 235, 235))
    out = Image.new("RGB", (W, 40 + sum(r.size[1] for r in rows)), (8, 8, 10))
    out.paste(head, (0, 0))
    y = 40
    for r in rows:
        out.paste(r, (0, y)); y += r.size[1]
    p = root / "review" / "layouts_sheet.png"
    out.save(p)
    return p


def _wrap(s, n):
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def _free(p: Path) -> Path:
    """never overwrite: name.png -> name_g02.png, _g03 ..."""
    if not p.exists():
        return p
    k = 2
    while (p.with_name(f"{p.stem}_g{k:02d}{p.suffix}")).exists():
        k += 1
    return p.with_name(f"{p.stem}_g{k:02d}{p.suffix}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "sheet"])
    ap.add_argument("designs")
    ap.add_argument("--only", default=None)
    ap.add_argument("--w", type=int, default=540)
    ap.add_argument("--draft", default=None, help="write to this folder (overwrite ok) instead of out/ (iteration)")
    args = ap.parse_args(argv)
    dp = Path(args.designs).resolve()
    doc = json.loads(dp.read_text(encoding="utf-8"))
    root = dp.parent
    if args.cmd == "sheet":
        print(sheet(doc, root))
        return
    reports = {}
    for d in doc["designs"]:
        if args.only and d["id"] not in args.only.split(","):
            continue
        t0 = time.time()
        img, rep = render(d, doc["images"][d["image"]], args.w)
        if args.draft:
            p = Path(args.draft) / d["file"]
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p = _free(root / "out" / d["file"])
            d["file"] = p.name
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(p)
        reports[d["id"]] = dict(file=str(p), seconds=round(time.time() - t0, 2), elements=rep)
        print(f"{d['id']}: {p.name} {time.time() - t0:.2f}s")
        for r in rep:
            if "ink" in r:
                print(f"   {r['name']:>8} {r['text'][:28]:28} {r['font']:18} cap {r['cap_frac']:.3f} ok={r['cap_ok']} ink {r['design_ink']}->{r['ink']} "
                      f"({r['ink_kind']}, cover {r['cover']}, sup {r['support']}) bg {r['bg']} hidden {r['hidden_by_runner']}")
    if not args.draft:
        rp = root / "review" / "render_report.json"
        old = json.loads(rp.read_text()) if rp.exists() else {}
        old.update(reports)
        rp.write_text(json.dumps(old, indent=1, default=float))
        dp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")     # file names now point at the newest gNN (for the sheet)
    else:
        (Path(args.draft) / "render_report.json").write_text(json.dumps(reports, indent=1, default=float))


if __name__ == "__main__":
    main()
