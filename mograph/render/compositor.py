"""Frame loop: scenes -> layer stack -> transitions -> globals (beat flash, vignette,
grain) -> final fade. Frames are produced one at a time."""
from __future__ import annotations

import bisect
from typing import Iterator

import cv2
import numpy as np

from ..geometry import cover_crop, frame_size
from .background import Background, ImageLayer
from .common import Ctx
from .easing import clamp01, decay_env
from .shapes import ShapeLayer
from .text import TextLayer
from .transitions import blend
from .treatments import apply_treatment

FINAL_FADE = 0.3


class Scene:
    def __init__(self, scene: dict, ctx: Ctx, idx: int):
        self.ctx = ctx
        self.idx = idx
        self.t0 = float(scene.get("t0", 0.0))
        self.t1 = float(scene.get("t1", ctx.duration))
        self.bg = Background(scene, ctx, idx)
        tr = scene.get("transition_out") or {"type": "cut", "dur": 0.0}
        self.tr = tr if isinstance(tr, dict) else {"type": str(tr)}
        self.tr_type = str(self.tr.get("type", "cut"))
        self.tr_dur = float(self.tr.get("dur", 0.3) or 0.0) if self.tr_type != "cut" else 0.0
        on_image = self.bg.type == "image"
        self.layers = []
        for li, layer in enumerate(scene.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            uid = idx * 1000 + li
            kind = layer.get("type")
            if kind == "text":
                self.layers.append(TextLayer(layer, ctx, uid, on_image))
            elif kind == "shape":
                self.layers.append(ShapeLayer(layer, ctx, uid, self))
            elif kind == "image":
                self.layers.append(ImageLayer(layer, ctx, uid))
        self._largest = None
        if self.bg.type == "image":
            # shape_reveal mask: the largest closed subject outline; contours the analysis marks as
            # background texture (low salience / not on real edges) fall back to the circle reveal
            meta = {c.get("id"): c for c in ((ctx.images.get(self.bg.image_id) or {}).get("contours") or [])
                    if isinstance(c, dict)}
            best, best_a = None, -1.0
            for cid, (iid, pts, closed, area) in ctx.contours.items():
                if iid != self.bg.image_id or len(pts) < 3 or not closed:
                    continue
                m = meta.get(cid) or {}
                if float(m.get("salience", 1.0) or 0.0) < 0.6 or float(m.get("edge_support", 1.0) or 0.0) < 0.5:
                    continue
                a = area if area > 0 else 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                                                    - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
                if a > best_a:
                    best, best_a = pts, a
            self._largest = best

    def map_image_points(self, iid: str, pts: np.ndarray, t: float) -> np.ndarray:
        ctx = self.ctx
        if self.bg.image_id == iid:
            return self.bg.map_points(pts, t)
        for L in self.layers:
            if isinstance(L, ImageLayer) and L.crop is not None and L.img_id == iid:
                return L.map_points(pts)
        im = ctx.images.get(iid)
        if im is not None and im.get("pixels") is not None:
            h, w = im["pixels"].shape[:2]
            f = im.get("focus") or [0.5, 0.5]
            cx, cy, cw, ch = cover_crop(w, h, ctx.W, ctx.H, (float(f[0]), float(f[1])))
            return np.stack([(pts[:, 0] - cx) / cw * ctx.W, (pts[:, 1] - cy) / ch * ctx.H], 1)
        return pts * np.array([ctx.W, ctx.H])

    def largest_contour_px(self, t: float):
        if self._largest is None:
            return None
        return self.bg.map_points(self._largest, t)

    def render(self, t: float, fi: int) -> np.ndarray:
        frame = self.bg.render(t)
        for L in self.layers:
            L.draw(frame, t, fi)
        return frame


def _vignette(W, H, amount):
    if amount <= 0:
        return None
    x = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1
    y = (np.arange(H, dtype=np.float32) + 0.5) / H * 2 - 1
    r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2) / np.sqrt(2.0)
    v = np.clip((r - 0.35) / 0.65, 0, 1) ** 1.6
    return cv2.cvtColor((1.0 - 0.85 * min(amount, 1.0) * v).astype(np.float32), cv2.COLOR_GRAY2BGR)


class Renderer:
    def __init__(self, plan: dict, analysis: dict, fmt: str, quality: str = "draft", audio_env=None):
        W, H = frame_size(fmt, quality)
        self.ctx = ctx = Ctx(plan, analysis, fmt, W, H, audio_env)
        scenes = sorted((s for s in (plan.get("scenes") or []) if isinstance(s, dict)),
                        key=lambda s: float(s.get("t0", 0.0)))
        if not scenes:
            scenes = [{"id": "s0", "t0": 0.0, "t1": ctx.duration,
                       "background": {"type": "color", "color": ctx.style.get("background", "palette:0")},
                       "layers": []}]
        self.scenes = [Scene(s, ctx, i) for i, s in enumerate(scenes)]
        self.starts = [s.t0 for s in self.scenes]
        style = ctx.style
        self.vig = _vignette(W, H, float(style.get("vignette", 0.0) or 0.0))
        self.grain = float(style.get("grain", 0.0) or 0.0)
        self.flash = float(style.get("beat_flash", 0.0) or 0.0)
        self.treat = style.get("treatment") or None     # whole-frame print/film treatment
        self.fade_col = ctx.bg_color.astype(np.float32)
        # grain: a seeded bank of noise fields; each frame picks one (and a flip) with its own seed
        self.grain_bank = []
        if self.grain > 0:
            gh, gw = (H // 2, W // 2) if ctx.S > 540 else (H, W)
            amp = np.float32(self.grain * 0.07 * (1.4 if ctx.S > 540 else 1.0))
            for k in range(12):
                self.grain_bank.append(ctx.rng(4242, k).standard_normal((gh, gw), dtype=np.float32) * amp)

    def _scene_frame(self, t: float, fi: int) -> np.ndarray:
        sc = self.scenes
        for j in range(len(sc) - 1):
            a = sc[j]
            d = a.tr_dur
            if d <= 0:
                continue
            lo, hi = a.t1 - d / 2.0, a.t1 + d / 2.0
            if lo <= t < hi:
                A = a.render(t, fi)
                b = sc[j + 1]
                B = b.render(t, fi)
                poly = b.largest_contour_px(t) if a.tr_type == "shape_reveal" else None
                return blend(a.tr_type, A, B, (t - lo) / d, self.ctx, a.tr, fi, poly)
        i = max(0, bisect.bisect_right(self.starts, t + 1e-9) - 1)
        return sc[i].render(t, fi)

    def frame(self, fi: int) -> np.ndarray:
        ctx = self.ctx
        t = fi / ctx.fps
        f = self._scene_frame(t, fi)
        if f.dtype != np.float32:
            f = f.astype(np.float32)
        if self.flash > 0:
            env = decay_env(ctx.downbeats, t, 0.09)
            if env > 0.01:
                lift = np.float32(self.flash * 0.45 * env)
                f *= np.float32(1.0) - lift
                f += lift
        if self.treat:
            f = apply_treatment(f, self.treat, ctx.style.get("palette"), fi, ctx.seed, ctx.fps, ctx.S / 1080.0)
        if self.vig is not None:
            f *= self.vig
        if self.grain_bank:
            H, W = f.shape[:2]
            r = ctx.rng(fi, 4242)
            n = self.grain_bank[int(r.integers(len(self.grain_bank)))]
            fl = int(r.integers(4))
            if fl < 3:
                n = cv2.flip(n, fl - 1)
            if n.shape[0] != H:
                n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
            f += cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)
        fade = clamp01((t + 1.0 / ctx.fps - (ctx.duration - FINAL_FADE)) / FINAL_FADE)
        if fade > 0:
            f *= np.float32(1.0 - fade)
            f += self.fade_col * np.float32(fade)
        np.clip(f, 0.0, 1.0, out=f)
        return cv2.convertScaleAbs(f, alpha=255.0)

    def __iter__(self) -> Iterator[np.ndarray]:
        for fi in range(self.ctx.n_frames):
            yield self.frame(fi)


def render_frames(plan: dict, analysis: dict, fmt: str, quality: str = "draft",
                  audio_env: dict | None = None) -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 RGB frames, n = round(duration * fps). Setup runs eagerly."""
    return iter(Renderer(plan, analysis, fmt, quality, audio_env))
