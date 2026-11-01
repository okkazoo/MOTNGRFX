"""Scene backgrounds (image | color | gradient | shapes) with motion verbs and
treatments, plus image layers. Image sources are pre-scaled and treated once per
scene; each frame is a single cv2.warpAffine of the cached source."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..geometry import cover_crop
from .common import (Anim, Ctx, Timing, affine, apply_emphasis, apply_in, apply_out,
                     composite, glitch_jitter)
from .easing import decay_env
from .shapes import circle_pts, draw_ops, get_box
from .treatments import apply_treatment

BG_MOTIONS = ("static", "ken_burns", "drift", "beat_zoom", "parallax_split")


def warp_crop(src: np.ndarray, crop, W: int, H: int) -> np.ndarray:
    """Resample the normalised image-space ``crop`` of ``src`` to a WxH frame."""
    sh, sw = src.shape[:2]
    cx, cy, cw, ch = crop
    ax = cw * sw / W
    ay = ch * sh / H
    Minv = np.array([[ax, 0.0, cx * sw + 0.5 * ax - 0.5],
                     [0.0, ay, cy * sh + 0.5 * ay - 0.5]])
    return cv2.warpAffine(src, Minv, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _lerp(a, b, p):
    return a + (b - a) * p


def _pair(v, default):
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return float(v[0]), float(v[0])
    return default


def gradient_image(W, H, c0, c1, angle_deg) -> np.ndarray:
    a = math.radians(float(angle_deg))
    dx, dy = math.cos(a), math.sin(a)
    xs = (np.arange(W, dtype=np.float32) + 0.5) / W - 0.5
    ys = (np.arange(H, dtype=np.float32) + 0.5) / H - 0.5
    ext = 0.5 * (abs(dx) + abs(dy))
    u = (xs[None, :] * dx + ys[:, None] * dy) / (2 * ext) + 0.5
    u = np.clip(u, 0, 1)
    u = u * u * (3 - 2 * u)
    return (c0[None, None, :] * (1 - u[..., None]) + c1[None, None, :] * u[..., None]).astype(np.float32)


class Background:
    def __init__(self, scene: dict, ctx: Ctx, scene_idx: int):
        self.ctx = ctx
        self.idx = scene_idx
        self.t0 = float(scene.get("t0", 0.0))
        self.t1 = float(scene.get("t1", self.t0 + 1.0))
        bg = scene.get("background") or {}
        self.type = bg.get("type", "color")
        tr = bg.get("treatment")     # typed print/film treatment: {"type": ...} (with or without legacy keys) or a list
        self.treat = tr if isinstance(tr, list) else (tr if isinstance(tr, dict) and tr.get("type") else None)
        mo = bg.get("motion") or {}
        self.motion = mo
        self.verb = mo.get("verb", "static")
        if self.verb not in BG_MOTIONS:
            self.verb = "static"
        self.image_id = None
        self.src = None
        self.fg_mask = None
        self.sal_box = None
        self._cache_key = None
        self._cache = None
        W, H = ctx.W, ctx.H

        self.color = ctx.color(bg.get("color") or ctx.style.get("background") or "palette:0", (0.07, 0.07, 0.07))
        z0, z1 = _pair(mo.get("zoom"), (1.0, 1.0))
        self.z = (z0, z1)
        pan = mo.get("pan") or [[0.0, 0.0], [0.0, 0.0]]
        try:
            self.pan = (np.array(pan[0], np.float64), np.array(pan[1], np.float64))
        except Exception:
            self.pan = (np.zeros(2), np.zeros(2))
        self.amount = float(mo.get("amount", 0.06) or 0.06)
        ang = float(ctx.rng(scene_idx, 11).uniform(0, 2 * np.pi))
        self.drift_dir = np.array([math.cos(ang), math.sin(ang)])

        if self.type == "image":
            im = ctx.images.get(bg.get("src"))
            if im is None or im.get("pixels") is None:
                self.type = "color"
            else:
                self.image_id = bg.get("src")
                self._prepare_image(im, bg)
        if self.type == "gradient":
            cols = bg.get("colors") or [ctx.style.get("background", "palette:0"), "palette:2"]
            c0 = ctx.color(cols[0], (0.05, 0.05, 0.08))
            c1 = ctx.color(cols[1] if len(cols) > 1 else cols[0], (0.2, 0.2, 0.3))
            # 1.3x oversized source so drift / zoom have room to move
            self.src = gradient_image(int(W * 1.3), int(H * 1.3), c0, c1, bg.get("angle", 135.0))
            self.src_wh = (self.src.shape[1], self.src.shape[0])
            if self.treat:
                self.src = self._treated(self.src, *self.src_wh)
            self.focus = np.array([0.5, 0.5])
            if self.verb == "static":
                self.z = (1.0, 1.0)
        elif self.type == "shapes":
            self._prepare_shapes(bg)
        elif self.type not in ("image",):
            self.type = "color"
            self.flat = np.empty((H, W, 3), np.float32)
            self.flat[:] = self.color
            if self.treat:
                self.flat = self._treated(self.flat, W, H)

    def _treated(self, src: np.ndarray, iw: int, ih: int, fi: int = 0) -> np.ndarray:
        """Apply the typed treatment to a source whose crop at zoom 1 covers the frame; px sizes
        in the spec are frame px at 1080, so they are scaled by source px per frame px."""
        ctx = self.ctx
        f = src.shape[1] * cover_crop(iw, ih, ctx.W, ctx.H)[2] / ctx.W
        if f < 1.0:      # print structure (dots, grit) needs at least frame resolution
            src = np.clip(cv2.resize(src, (max(2, int(round(src.shape[1] / f))), max(2, int(round(src.shape[0] / f)))),
                                     interpolation=cv2.INTER_LINEAR), 0, 1)
            f = src.shape[1] * cover_crop(iw, ih, ctx.W, ctx.H)[2] / ctx.W
        seed = int(ctx.rng(self.idx, 97).integers(1 << 30))
        return apply_treatment(src, self.treat, ctx.style.get("palette"), fi, seed, ctx.fps, f * ctx.S / 1080.0)

    # ------------------------------------------------------------------ image source
    def _zoom_bounds(self):
        z0, z1 = self.z
        v = self.verb
        if v == "static":
            return max(1.0, z0)
        if v == "drift":
            return max(1.08, z0, z1)
        if v == "beat_zoom":
            return max(z0, z1, 1.03) + self.amount + 0.02
        if v == "parallax_split":
            return max(1.15, z1)
        return max(z0, z1, 1.0)

    def _prepare_image(self, im: dict, bg: dict):
        ctx = self.ctx
        W, H = ctx.W, ctx.H
        px = im["pixels"]
        ih, iw = px.shape[:2]
        self.img_wh = (iw, ih)
        focus = bg.get("focus") or im.get("focus") or [0.5, 0.5]
        self.focus = np.array([float(focus[0]), float(focus[1])])
        zmax = self._zoom_bounds()
        cw = cover_crop(iw, ih, W, H, tuple(self.focus), 1.0)[2] * iw
        s = W * zmax / max(cw, 1.0)
        src = px
        if s < 1.0:
            nw, nh = max(2, int(round(iw * s))), max(2, int(round(ih * s)))
            src = cv2.resize(px, (nw, nh), interpolation=cv2.INTER_AREA)
        src = np.ascontiguousarray(src, np.float32)
        tr = bg.get("treatment")
        tr = tr if isinstance(tr, dict) else {}
        blur = float(tr.get("blur", 0.0) or 0.0)
        if blur > 0:
            sigma = blur * 0.025 * ctx.S * (src.shape[1] * cover_crop(iw, ih, W, H)[2] / W)
            if sigma > 0.3:
                src = cv2.GaussianBlur(src, (0, 0), sigma)
        if self.treat:   # typed treatment: after blur, before the legacy duotone / darken
            src = self._treated(src, iw, ih)
        duo = tr.get("duotone")
        if isinstance(duo, (list, tuple)) and len(duo) >= 2:
            cd, cl = ctx.color(duo[0], (0, 0, 0)), ctx.color(duo[1], (1, 1, 1))
            L = src @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            lo, hi = np.percentile(L, [2, 98])
            L = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)[..., None]
            src = (cd[None, None] * (1 - L) + cl[None, None] * L).astype(np.float32)
        darken = float(tr.get("darken", 0.0) or 0.0)
        if darken > 0:
            src = src * np.float32(1.0 - min(darken, 1.0))
        self.src = np.ascontiguousarray(src, np.float32)
        self.src_wh = (iw, ih)        # crop maths in original image aspect
        if self.verb == "parallax_split":
            sb = im.get("saliency_box") or [0.3, 0.3, 0.4, 0.4]
            self.sal_box = [float(v) for v in sb]
            sh_, sw_ = self.src.shape[:2]
            mask = np.zeros((sh_, sw_), np.float32)
            x, y, w, h = self.sal_box
            # foreground = largest contour centred inside the saliency box (a clean subject
            # cut-out); otherwise a feathered ellipse over the saliency box
            subj, best = None, 0.0
            for cid, (iid, pts, closed, area) in ctx.contours.items():
                if iid != self.image_id or not closed or len(pts) < 3:
                    continue
                c = (pts.min(0) + pts.max(0)) / 2.0
                a = float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))
                if x <= c[0] <= x + w and y <= c[1] <= y + h and a > best:
                    subj, best = pts, a
            if subj is not None:
                q = np.round(subj * [sw_ * 16, sh_ * 16]).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [q], 1.0, cv2.LINE_AA, 4)
                self.fg_mask = cv2.GaussianBlur(mask, (0, 0), max(0.6, 0.004 * min(sw_, sh_)))
            else:
                cxp, cyp = (x + w / 2) * sw_, (y + h / 2) * sh_
                axes = (max(2, int(w * sw_ * 0.62)), max(2, int(h * sh_ * 0.62)))
                cv2.ellipse(mask, (int(cxp), int(cyp)), axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
                self.fg_mask = cv2.GaussianBlur(mask, (0, 0), 0.025 * min(sw_, sh_))
            self.fg_shadow = cv2.GaussianBlur(mask, (0, 0), 0.06 * min(sw_, sh_)) * np.float32(0.35)

    # ------------------------------------------------------------------ crops
    def _p(self, t):
        span = max(1e-3, self.t1 - self.t0)
        return float(np.clip((t - self.t0) / span, -0.5, 1.5))

    def crop_at(self, t: float, layer: str = "bg"):
        """Normalised image-space crop at time t (for image/gradient backgrounds)."""
        iw, ih = self.src_wh
        W, H = self.ctx.W, self.ctx.H
        p = self._p(t)
        v = self.verb
        focus = self.focus
        z0, z1 = self.z
        if v == "ken_burns":
            zoom = _lerp(z0, z1, p)
            focus = focus + _lerp(self.pan[0], self.pan[1], p)
        elif v == "drift":
            zoom = max(1.08, z0)
            focus = focus + self.drift_dir * self.amount * (p - 0.5)
        elif v == "beat_zoom":
            zoom = max(1.0, z0) + 0.02 * p + self.amount * decay_env(self.ctx.beats, t, 0.14)
        elif v == "parallax_split":
            if layer == "fg":
                zoom = _lerp(1.0, max(1.15, z1), p)
            else:
                zoom = _lerp(1.0, 1.04, p)
                focus = focus + _lerp(self.pan[0], self.pan[1], p)
        else:
            zoom = max(1.0, z0)
        return cover_crop(iw, ih, W, H, (float(focus[0]), float(focus[1])), max(1.0, zoom))

    def map_points(self, pts: np.ndarray, t: float) -> np.ndarray:
        """Image-space normalised points -> frame px through the current crop."""
        crop = self.crop_at(t)
        if self.verb == "parallax_split" and self.sal_box is not None and len(pts):
            c = (pts.min(0) + pts.max(0)) / 2.0
            x, y, w, h = self.sal_box
            if x <= c[0] <= x + w and y <= c[1] <= y + h:
                crop = self.crop_at(t, "fg")
        cx, cy, cw, ch = crop
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = (pts[:, 0] - cx) / cw * self.ctx.W
        out[:, 1] = (pts[:, 1] - cy) / ch * self.ctx.H
        return out

    # ------------------------------------------------------------------ shapes bg
    def _prepare_shapes(self, bg):
        ctx = self.ctx
        W, H, S = ctx.W, ctx.H, ctx.S
        r = ctx.rng(self.idx, 23)
        self.flat = np.empty((H, W, 3), np.float32)
        self.flat[:] = self.color
        self.blobs = []
        self.lo_div = 4
        lw, lh = max(2, W // self.lo_div), max(2, H // self.lo_div)
        self.flat_lo = np.empty((lh, lw, 3), np.float32)
        self.flat_lo[:] = self.color
        for k in range(3):
            rad = S * float(r.uniform(0.35, 0.6)) / self.lo_div
            R = int(rad * 1.6)
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
            spr = np.exp(-(xx * xx + yy * yy) / (2 * (rad * 0.55) ** 2)).astype(np.float32)
            self.blobs.append(dict(
                spr=spr, R=R, col=ctx.accent(k), a=float(r.uniform(0.35, 0.55)),
                p=np.array([r.uniform(0.1, 0.9) * W, r.uniform(0.1, 0.9) * H]),
                v=np.array([r.uniform(-1, 1), r.uniform(-1, 1)]) * S * 0.03,
                ph=float(r.uniform(0, 6.28))))
        light = ctx.palette[1] if len(ctx.palette) > 1 else np.ones(3, np.float32)
        self.rings = [dict(c=np.array([r.uniform(0.2, 0.8) * W, r.uniform(0.2, 0.8) * H]),
                           rad=S * float(r.uniform(0.15, 0.4)), col=light, ph=float(r.uniform(0, 6.28)))
                      for _ in range(2)]
        self.bars = [dict(y=float(r.uniform(0.15, 0.85)) * H, ang=float(r.uniform(-0.5, 0.5)),
                          spd=float(r.uniform(0.05, 0.12)) * W * (1 if r.random() < 0.5 else -1),
                          col=ctx.accent(k + 1), len=float(r.uniform(0.3, 0.6)) * W,
                          th=float(r.uniform(0.008, 0.02)) * S)
                     for k in range(2)]

    def _render_shapes(self, t: float) -> np.ndarray:
        ctx = self.ctx
        W, H, S = ctx.W, ctx.H, ctx.S
        lo = self.flat_lo.copy()
        beat = decay_env(ctx.beats, t, 0.2)
        dv = float(self.lo_div)
        for b in self.blobs:
            pos = b["p"] + b["v"] * t + np.array([math.sin(t * 0.7 + b["ph"]), math.cos(t * 0.5 + b["ph"])]) * S * 0.05
            pos = pos / dv
            s = 1.0 + 0.05 * beat
            M = affine(scale=s, pivot=(b["R"], b["R"]), dx=pos[0] - b["R"], dy=pos[1] - b["R"])
            composite(lo, b["spr"], b["col"], M, b["a"])
        out = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)
        ops = []
        for rg in self.rings:
            rad = rg["rad"] * (1.0 + 0.04 * math.sin(t * 0.9 + rg["ph"]) + 0.03 * beat)
            ops.append(("line", circle_pts(rg["c"][0], rg["c"][1], rad), True, max(1.0, 0.004 * S)))
        draw_ops(out, ops, self.rings[0]["col"], 0.22)
        for b in self.bars:
            x = ((b["spd"] * t) % (W + b["len"])) - b["len"] / 2 if b["spd"] > 0 else W - ((-b["spd"] * t) % (W + b["len"])) + b["len"] / 2
            d = np.array([math.cos(b["ang"]), math.sin(b["ang"])]) * b["len"] / 2
            c = np.array([x, b["y"]])
            draw_ops(out, [("line", np.array([c - d, c + d]), False, b["th"])], b["col"], 0.7)
        return out

    # ------------------------------------------------------------------ render
    def render(self, t: float) -> np.ndarray:
        ctx = self.ctx
        if self.type == "color":
            return self.flat.copy()
        if self.type == "shapes":
            out = self._render_shapes(t)
            if self.treat:
                out = self._treated(out, ctx.W, ctx.H, int(round(t * ctx.fps)))
            return out
        crop = self.crop_at(t)
        key = tuple(round(v, 7) for v in crop)
        if key == self._cache_key and self.verb != "parallax_split":
            return self._cache.copy()
        out = warp_crop(self.src, crop, ctx.W, ctx.H)
        if self.verb == "parallax_split" and self.fg_mask is not None:
            fcrop = self.crop_at(t, "fg")
            fg = warp_crop(self.src, fcrop, ctx.W, ctx.H)
            m = cv2.cvtColor(warp_crop(self.fg_mask, fcrop, ctx.W, ctx.H), cv2.COLOR_GRAY2BGR)
            # soft drop shadow under the lifted foreground
            sh = cv2.cvtColor(warp_crop(self.fg_shadow, fcrop, ctx.W, ctx.H), cv2.COLOR_GRAY2BGR)
            np.subtract(np.float32(1.0), sh, out=sh)
            out *= sh
            fg -= out
            fg *= m
            out += fg
        self._cache_key, self._cache = key, out
        return out.copy()


# ----------------------------------------------------------------------------- image layer

MASKS = ("rect", "rounded", "circle")


class ImageLayer:
    kind = "image"

    def __init__(self, layer: dict, ctx: Ctx, uid: int):
        self.layer = layer
        self.ctx = ctx
        self.uid = uid
        self.tm = Timing(layer)
        self.sprite = None
        W, H, S = ctx.W, ctx.H, ctx.S
        bx, by, bw, bh = get_box(layer, ctx, (0.1, 0.1, 0.8, 0.45))
        X0, Y0 = int(round(bx * W)), int(round(by * H))
        BW, BH = max(2, int(round(bw * W))), max(2, int(round(bh * H)))
        self.rect = (X0, Y0, BW, BH)
        self.pivot = (X0 + BW / 2.0, Y0 + BH / 2.0)
        self.U = {"slide": 0.12 * H, "slide_x": 0.25 * W, "blur": 0.03 * S}
        self.crop = None
        im = ctx.images.get(layer.get("src"))
        if im is None or im.get("pixels") is None:
            return
        px = im["pixels"]
        ih, iw = px.shape[:2]
        focus = im.get("focus") or [0.5, 0.5]
        crop = cover_crop(iw, ih, BW, BH, (float(focus[0]), float(focus[1])), 1.0)
        self.crop = crop
        self.img_id = layer.get("src")
        x0, y0 = int(crop[0] * iw), int(crop[1] * ih)
        x1, y1 = max(x0 + 1, int(round((crop[0] + crop[2]) * iw))), max(y0 + 1, int(round((crop[1] + crop[3]) * ih)))
        sub = px[y0:y1, x0:x1]
        interp = cv2.INTER_AREA if sub.shape[1] > BW else cv2.INTER_LINEAR
        rgb = cv2.resize(sub, (BW, BH), interpolation=interp).astype(np.float32)
        if layer.get("treatment"):
            rgb = apply_treatment(rgb, layer["treatment"], ctx.style.get("palette"), 0,
                                  int(ctx.rng(uid, 98).integers(1 << 30)), ctx.fps, S / 1080.0)

        mask_name = str(layer.get("mask") or "rect")
        ss = 4
        border = layer.get("border") or None
        bcol = ctx.color(border.get("color", "palette:1")) if isinstance(border, dict) else None
        bwid = float(border.get("width", 0.004) or 0.004) * S if isinstance(border, dict) else 0.0
        mimg = Image.new("L", (BW * ss, BH * ss), 0)
        bimg = Image.new("L", (BW * ss, BH * ss), 0)
        dm, db = ImageDraw.Draw(mimg), ImageDraw.Draw(bimg)
        bw_ss = max(1, int(round(bwid * ss)))
        if mask_name.startswith("contour:"):
            cid = mask_name.split(":", 1)[1]
            ent = ctx.contours.get(cid)
            if ent is not None and len(ent[1]) >= 3:
                c_img = ctx.images.get(ent[0])
                ccrop = crop
                if ent[0] != self.img_id and c_img is not None and c_img.get("pixels") is not None:
                    chh, cww = c_img["pixels"].shape[:2]
                    ccrop = cover_crop(cww, chh, BW, BH)
                pts = ent[1]
                q = np.stack([(pts[:, 0] - ccrop[0]) / ccrop[2] * BW * ss,
                              (pts[:, 1] - ccrop[1]) / ccrop[3] * BH * ss], 1)
                poly = [tuple(map(float, p)) for p in q]
                dm.polygon(poly, fill=255)
                if bcol is not None:
                    db.line(poly + [poly[0]], fill=255, width=bw_ss, joint="curve")
            else:
                mask_name = "rect"
        if mask_name == "circle":
            d = min(BW, BH) * ss
            ox, oy = (BW * ss - d) / 2, (BH * ss - d) / 2
            dm.ellipse([ox, oy, ox + d - 1, oy + d - 1], fill=255)
            if bcol is not None:
                db.ellipse([ox, oy, ox + d - 1, oy + d - 1], outline=255, width=bw_ss)
        elif mask_name == "rounded":
            rad = int(0.08 * min(BW, BH) * ss)
            dm.rounded_rectangle([0, 0, BW * ss - 1, BH * ss - 1], radius=rad, fill=255)
            if bcol is not None:
                db.rounded_rectangle([0, 0, BW * ss - 1, BH * ss - 1], radius=rad, outline=255, width=bw_ss)
        elif not mask_name.startswith("contour:"):
            dm.rectangle([0, 0, BW * ss - 1, BH * ss - 1], fill=255)
            if bcol is not None:
                db.rectangle([0, 0, BW * ss - 1, BH * ss - 1], outline=255, width=bw_ss)
        a = cv2.resize(np.asarray(mimg, np.float32) / 255.0, (BW, BH), interpolation=cv2.INTER_AREA)
        b = cv2.resize(np.asarray(bimg, np.float32) / 255.0, (BW, BH), interpolation=cv2.INTER_AREA)
        spr = np.empty((BH, BW, 4), np.float32)
        spr[..., :3] = rgb * a[..., None]
        spr[..., 3] = a
        if bcol is not None:
            spr[..., :3] = spr[..., :3] * (1 - b[..., None]) + bcol[None, None] * b[..., None]
            spr[..., 3] = a + b * (1 - a)
        self.sprite = spr
        accent = ctx.color(layer.get("highlight") or "palette:2")
        self.ul_color = accent

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        X0, Y0, BW, BH = self.rect
        cx, cy, cw, ch = self.crop
        return np.stack([X0 + (pts[:, 0] - cx) / cw * BW, Y0 + (pts[:, 1] - cy) / ch * BH], 1)

    def draw(self, frame: np.ndarray, t: float, fi: int) -> None:
        tm, ctx = self.tm, self.ctx
        if self.sprite is None or not tm.active(t):
            return
        st = Anim()
        p_in = tm.p_in(t)
        apply_in(st, tm.in_verb, p_in, ctx.motion, self.U, self.kind)
        apply_out(st, tm.out_verb, tm.p_out(t), ctx.motion, self.U, self.kind)
        apply_emphasis(st, self.layer.get("emphasis"), t, fi, ctx, tm, self.uid)
        glitch_jitter(st, p_in, ctx, self.uid, fi)
        if st.opacity <= 0.0 or st.scale <= 1e-3:
            return
        X0, Y0, BW, BH = self.rect
        M = affine(scale=st.scale, pivot=self.pivot, dx=st.dx, dy=st.dy, ox=X0, oy=Y0)
        clip = None
        if st.wipe_lo > 0.0 or st.wipe_hi < 1.0:
            clip = (X0 + st.dx + BW * st.wipe_lo, -1e6, X0 + st.dx + BW * st.wipe_hi, 1e6)
        composite(frame, self.sprite, None, M, st.opacity, clip=clip, blur=st.blur)
        if st.underline:
            y = Y0 + BH + 0.02 * ctx.S + st.dy
            x = X0 + st.dx
            draw_ops(frame, [("line", np.array([[x, y], [x + BW * st.underline, y]]), False,
                              max(2.0, 0.008 * ctx.S))], self.ul_color, st.opacity)
