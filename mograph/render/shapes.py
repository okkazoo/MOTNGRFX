"""Shape layers: contour / line outlines borrowed from the imagery, and primitives.

Vector content is rasterised on a 2x supersampled mask (cv2 anti-aliased, sub-pixel
``shift``) cropped to its bounding box, then downsampled and composited.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import (Anim, Ctx, Timing, apply_emphasis, apply_in, apply_out, composite,
                     glitch_jitter, translate)
from .easing import clamp01, decay_env, ease_in

PRIMITIVES = ("circle", "ring", "rect", "bar", "cross", "grid", "dot_field", "wave")

_SHIFT = 4
_ONE = 1 << _SHIFT


# ----------------------------------------------------------------------------- geometry helpers

def trim_polyline(pts: np.ndarray, closed: bool, a: float = 0.0, b: float = 1.0):
    """Sub-path between fractions a..b of the cumulative length. Returns (pts, closed) or None."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return None
    a, b = clamp01(a), clamp01(b)
    if b - a <= 1e-4:
        return None
    if a <= 0.0 and b >= 1.0:
        return pts, closed
    p = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    if L <= 1e-9:
        return None
    s0, s1 = a * L, b * L

    def at(s):
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        f = 0.0 if seg[i] <= 0 else (s - cum[i]) / seg[i]
        return p[i] + (p[i + 1] - p[i]) * f

    inner = p[(cum > s0) & (cum < s1)]
    out = np.vstack([at(s0)[None], inner, at(s1)[None]])
    return out, False


def raster(ops, W: int, H: int, ss: int = 2):
    """Rasterise ops into an anti-aliased float32 alpha mask cropped to the visible bbox.

    ops: ("line", pts, closed, width_px[, value 0..255]) | ("fill", pts) | ("dot", (x, y), r).
    Later ops paint over earlier ones, so echo copies can share one mask with the main stroke.
    Returns (mask, x0, y0) or None.
    """
    xs0, ys0, xs1, ys1 = [], [], [], []
    for op in ops:
        if op[0] == "dot":
            (x, y), r = op[1], op[2]
            xs0.append(x - r); xs1.append(x + r); ys0.append(y - r); ys1.append(y + r)
        else:
            pts = op[1]
            m = op[3] / 2.0 if op[0] == "line" else 0.0
            xs0.append(pts[:, 0].min() - m); xs1.append(pts[:, 0].max() + m)
            ys0.append(pts[:, 1].min() - m); ys1.append(pts[:, 1].max() + m)
    if not xs0:
        return None
    x0 = max(0, int(math.floor(min(xs0))) - 2)
    y0 = max(0, int(math.floor(min(ys0))) - 2)
    x1 = min(W, int(math.ceil(max(xs1))) + 2)
    y1 = min(H, int(math.ceil(max(ys1))) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0
    m = np.zeros((h * ss, w * ss), np.uint8)
    off = np.array([x0, y0], np.float64)
    k = ss * _ONE
    # keep coordinates inside int32 / cv2 limits
    lim = float(1 << 20)
    for op in ops:
        if op[0] == "dot":
            c = np.round((np.asarray(op[1]) - off) * k).astype(np.int64)
            r = int(round(op[2] * k))
            if r > 0:
                cv2.circle(m, (int(c[0]), int(c[1])), r, 255, -1, cv2.LINE_AA, _SHIFT)
            continue
        p = np.clip((op[1] - off) * k, -lim, lim).round().astype(np.int32).reshape(-1, 1, 2)
        if op[0] == "line":
            th = max(1, int(round(op[3] * ss)))
            val = int(op[4]) if len(op) > 4 else 255
            cv2.polylines(m, [p], bool(op[2]), val, th, cv2.LINE_AA, _SHIFT)
        else:
            cv2.fillPoly(m, [p], 255, cv2.LINE_AA, _SHIFT)
    if ss > 1:
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)
    else:
        mask = m.astype(np.float32) * (1.0 / 255.0)
    return mask, x0, y0


def draw_ops(frame, ops, color, opacity=1.0, clip=None, blur=0.0, ss=2):
    if not ops or color is None or opacity <= 0.0:
        return
    H, W = frame.shape[:2]
    r = raster(ops, W, H, ss)
    if r is None:
        return
    mask, x0, y0 = r
    composite(frame, mask, color, translate(x0, y0), opacity, clip=clip, blur=blur)


def circle_pts(cx, cy, r, n=None):
    n = n or int(np.clip(r * 0.8, 32, 180))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def rect_pts(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)


def scale_about(pts, c, s):
    return (pts - c) * s + c


# ----------------------------------------------------------------------------- layer

def get_box(layer: dict, ctx: Ctx, default=(0.1, 0.1, 0.8, 0.8)):
    box = layer.get("box")
    if isinstance(box, dict):
        b = box.get(ctx.fmt) or next((v for v in box.values() if v), None)
    else:
        b = box
    if not b or len(b) < 4:
        b = default
    return tuple(float(v) for v in b[:4])


class ShapeLayer:
    kind = "shape"

    def __init__(self, layer: dict, ctx: Ctx, uid: int, scene):
        self.layer = layer
        self.ctx = ctx
        self.uid = uid
        self.scene = scene
        self.tm = Timing(layer)
        self.source = layer.get("source", "primitive")
        self.primitive = layer.get("primitive") or "rect"
        if self.primitive not in PRIMITIVES:
            self.primitive = "rect"
        self.ref = None
        if self.source == "contour":
            self.ref = ctx.contours.get(layer.get("ref"))
        elif self.source == "line":
            self.ref = ctx.lines.get(layer.get("ref"))
            if self.ref is None:
                self.ref = ctx.contours.get(layer.get("ref"))
        if self.source in ("contour", "line") and self.ref is None:
            self.source = "missing"
        stroke = layer.get("stroke", "palette:2")
        self.stroke = ctx.color(stroke) if stroke is not None else None
        self.fill = ctx.color(layer.get("fill")) if layer.get("fill") is not None else None
        self.fill_op = float(layer.get("fill_opacity", 0.0) or 0.0)
        self.width = max(1.0, float(layer.get("stroke_width", 0.006) or 0.006) * ctx.S)
        off = layer.get("offset") or [0.0, 0.0]
        self.offset = np.array([float(off[0]) * ctx.W, float(off[1]) * ctx.H])
        self.base_scale = float(layer.get("scale", 1.0) or 1.0)
        echo = layer.get("echo") or None
        self.echo_n = int(echo.get("count", 0) or 0) if isinstance(echo, dict) else 0
        self.echo_sp = float(echo.get("spacing", 0.015) or 0.015) if self.echo_n else 0.0
        self.echo_decay = float(echo.get("decay", 0.5) or 0.5) if self.echo_n else 0.0
        bx, by, bw, bh = get_box(layer, ctx, (0.25, 0.4, 0.5, 0.2))
        self.box = (bx * ctx.W, by * ctx.H, bw * ctx.W, bh * ctx.H)
        flip_alt = ctx.palette[1] if len(ctx.palette) > 1 else np.ones(3, np.float32)
        if self.stroke is not None and np.allclose(flip_alt, self.stroke, atol=0.05):
            flip_alt = ctx.accent(1)
        self.flip_color = flip_alt
        self.U = {"slide": 0.12 * ctx.H, "slide_x": 0.25 * ctx.W, "blur": 0.03 * ctx.S}

    # -- geometry at time t: list of (pts, closed) in frame px, fills, dots, centre
    def _geometry(self, t: float, fi: int, trim_hi: float):
        ctx = self.ctx
        if self.source in ("contour", "line"):
            iid, pts, closed, _ = self.ref
            px = self.scene.map_image_points(iid, pts, t)
            c = (px.min(0) + px.max(0)) / 2.0
            fills = [px] if (closed and self.fill is not None and self.fill_op > 0) else []
            return [(px, closed)], fills, [], c, None
        bx, by, bw, bh = self.box
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        c = np.array([cx, cy])
        prim = self.primitive
        strokes, fills, dots = [], [], []
        default_fill = None
        if prim in ("circle", "ring"):
            p = circle_pts(cx, cy, min(bw, bh) / 2.0)
            strokes.append((p, True))
            if prim == "circle":
                default_fill = p
        elif prim == "rect":
            strokes.append((rect_pts(bx, by, bw, bh), True))
        elif prim == "bar":
            p = rect_pts(bx, by, bw * max(trim_hi, 0.0), bh)
            if trim_hi > 0:
                default_fill = p
        elif prim == "cross":
            strokes.append((np.array([[bx, cy], [bx + bw, cy]]), False))
            strokes.append((np.array([[cx, by], [cx, by + bh]]), False))
        elif prim == "grid":
            cols = 4 if bw >= bh else 3
            rows = max(2, int(round(cols * bh / max(bw, 1.0))))
            for i in range(cols + 1):
                x = bx + bw * i / cols
                strokes.append((np.array([[x, by], [x, by + bh]]), False))
            for j in range(rows + 1):
                y = by + bh * j / rows
                strokes.append((np.array([[bx, y], [bx + bw, y]]), False))
        elif prim == "dot_field":
            sp = max(4.0, min(bw, bh) / 5.0)
            nx, ny = max(1, int(bw // sp)), max(1, int(bh // sp))
            ox, oy = bx + (bw - (nx - 1) * sp) / 2.0, by + (bh - (ny - 1) * sp) / 2.0
            beat = decay_env(ctx.beats, t, 0.2)
            total = nx * ny
            k = 0
            for j in range(ny):
                for i in range(nx):
                    x, y = ox + i * sp, oy + j * sp
                    if k / total < trim_hi:
                        wv = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 0.8 - (i + j) / 6.0))
                        r = sp * (0.08 + 0.1 * wv + 0.08 * beat)
                        dots.append(((x, y), r))
                    k += 1
        elif prim == "wave":
            lo, mid, hi = self._bands(t, fi)
            u = np.linspace(0.0, 1.0, 140)
            taper = np.sin(np.pi * u) ** 0.6
            y = (lo * 0.6 * np.sin(2 * np.pi * 1.5 * u + t * 2.0)
                 + mid * 0.3 * np.sin(2 * np.pi * 4.0 * u - t * 5.0)
                 + hi * 0.15 * np.sin(2 * np.pi * 11.0 * u + t * 9.0)) * taper
            strokes.append((np.stack([bx + u * bw, cy + y * bh * 0.5], 1), False))
        if default_fill is not None or (self.fill is not None and self.fill_op > 0 and strokes and strokes[0][1]):
            fills.append(default_fill if default_fill is not None else strokes[0][0])
        return strokes, fills, dots, c, default_fill

    def _bands(self, t, fi):
        ctx = self.ctx
        lo, mid, hi = ctx.env_value("low", fi), ctx.env_value("mid", fi), ctx.env_value("high", fi)
        if lo is None:
            b = decay_env(ctx.beats, t, 0.22)
            lo = 0.3 + 0.7 * b
            mid = 0.3 + 0.4 * decay_env(ctx.onsets, t, 0.12)
            hi = 0.25 + 0.25 * (0.5 + 0.5 * math.sin(t * 7.0))
        return lo, mid, hi

    def draw(self, frame: np.ndarray, t: float, fi: int) -> None:
        tm, ctx = self.tm, self.ctx
        if not tm.active(t) or self.source == "missing":
            return
        st = Anim()
        p_in = tm.p_in(t)
        apply_in(st, tm.in_verb, p_in, ctx.motion, self.U, self.kind)
        apply_out(st, tm.out_verb, tm.p_out(t), ctx.motion, self.U, self.kind)
        apply_emphasis(st, self.layer.get("emphasis"), t, fi, ctx, tm, self.uid)
        glitch_jitter(st, p_in, ctx, self.uid, fi)
        if st.opacity <= 0.0 or st.scale <= 1e-3:
            return
        stroke_col = self.flip_color if (st.flip and self.stroke is not None) else self.stroke
        strokes, fills, dots, c, default_fill = self._geometry(t, fi, st.trim_hi)
        if not strokes and not fills and not dots:
            return
        clip = None
        if st.wipe_lo > 0.0 or st.wipe_hi < 1.0:
            allp = np.vstack([s[0] for s in strokes] + fills + [np.array([d[0]]) for d in dots])
            xa, xb = allp[:, 0].min() - self.width, allp[:, 0].max() + self.width
            clip = (xa + (xb - xa) * st.wipe_lo + st.dx, -1e6, xa + (xb - xa) * st.wipe_hi + st.dx, 1e6)
        scale = self.base_scale * st.scale
        shift = self.offset + np.array([st.dx, st.dy])

        def xf(p, s_extra=1.0):
            return scale_about(p, c, scale * s_extra) + shift

        r_mean = 1.0
        if strokes:
            allp = np.vstack([s[0] for s in strokes])
            r_mean = max(4.0, float(np.mean(np.hypot(*(allp - c).T))))
        fill_col = self.fill if self.fill is not None else stroke_col
        fill_op = self.fill_op if (self.fill is not None and self.fill_op > 0) else (1.0 if default_fill is not None else 0.0)
        if self.primitive != "bar" or self.source != "primitive":
            fill_op *= clamp01((st.trim_hi - 0.5) * 2.0) if st.trim_hi < 1.0 else 1.0

        is_line = self.source == "line"
        trim_all = self.primitive != "bar" or self.source != "primitive"
        sops = []
        for k in range(self.echo_n, -1, -1):
            val = 255.0 * (self.echo_decay ** k)
            if val < 3.0:
                continue
            if k and tm.in_verb == "stroke_draw" and tm.din > 0:
                th = min(st.trim_hi, ease_in(clamp01((t - tm.t0 - 0.09 * k) / tm.din), ctx.motion))
            else:
                th = st.trim_hi
            d = k * self.echo_sp * ctx.S
            for pts, closed in strokes:
                if k and is_line and len(pts) == 2:
                    v = pts[1] - pts[0]
                    nrm = np.array([-v[1], v[0]]) / max(1e-9, float(np.hypot(*v)))
                    q = xf(pts) + nrm * d
                else:
                    q = xf(pts, 1.0 + d / r_mean if k else 1.0)
                tr = trim_polyline(q, closed, st.trim_lo, th) if trim_all else (q, closed)
                if tr is not None:
                    sops.append(("line", tr[0], tr[1], self.width * (1.0 if k == 0 else 0.75), val))
        if fills and fill_op > 0 and fill_col is not None:
            draw_ops(frame, [("fill", xf(f)) for f in fills], fill_col, st.opacity * fill_op, clip, st.blur)
        if dots:
            dops = [("dot", tuple(xf(np.array([d_[0]]))[0]), d_[1] * scale) for d_ in dots]
            draw_ops(frame, dops, stroke_col if stroke_col is not None else fill_col, st.opacity, clip, st.blur)
        if stroke_col is not None and sops:
            draw_ops(frame, sops, stroke_col, st.opacity, clip, st.blur)
