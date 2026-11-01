"""Shared render helpers: render context, colour resolution, sprite compositing and
the block-level animation state machine used by text, shape and image layers."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .easing import back_out, clamp01, decay_env, ease_in, ease_out

IN_VERBS = ("cut", "fade", "type_on", "word_stagger", "slide_up", "slide_left", "scale_pop",
            "mask_wipe", "blur_in", "kinetic_split", "stroke_draw")
OUT_VERBS = ("cut", "fade", "slide_up", "slide_left", "scale_down", "mask_wipe", "blur_out",
             "stroke_draw")


# ----------------------------------------------------------------------------- colours

def _parse_hex(s: str):
    h = s.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    except ValueError:
        return None


def resolve_color(plan: dict, c, default=(1.0, 1.0, 1.0)):
    """"#rrggbb" | "palette:N" | (r, g, b) -> float32 array (3,) in 0..1. None -> None."""
    if c is None:
        return None
    if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 3:
        v = np.array([float(x) for x in c[:3]], np.float32)
        return np.clip(v / 255.0 if v.max() > 1.0 else v, 0, 1)
    if isinstance(c, str):
        s = c.strip()
        if s.startswith("palette:"):
            pal = ((plan or {}).get("style") or {}).get("palette") or []
            try:
                idx = int(s.split(":", 1)[1])
            except ValueError:
                idx = 0
            if pal:
                v = pal[idx % len(pal)]
                if isinstance(v, str) and not v.startswith("palette:"):
                    r = _parse_hex(v)
                    if r is not None:
                        return r
        else:
            r = _parse_hex(s)
            if r is not None:
                return r
    return np.array(default, np.float32)


def luminance(rgb) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


# ----------------------------------------------------------------------------- context

def _peak_times(x: np.ndarray, fps: float, thr: float = 0.5) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size < 3:
        return np.zeros(0, np.float32)
    m = (x[1:-1] >= thr) & (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    return (np.nonzero(m)[0] + 1).astype(np.float32) / float(fps)


class Ctx:
    """Everything a layer needs that is constant for one render call."""

    def __init__(self, plan: dict, analysis: dict, fmt: str, W: int, H: int, audio_env=None):
        self.plan = plan
        self.analysis = analysis or {}
        self.fmt = fmt
        self.W, self.H = int(W), int(H)
        self.S = min(self.W, self.H)
        self.fps = float(plan.get("fps", 30) or 30)
        self.duration = float(plan.get("duration", 10.0))
        self.n_frames = int(round(self.duration * self.fps))
        self.seed = int(plan.get("seed", 0) or 0)
        style = plan.get("style") or {}
        self.style = style
        self.motion = style.get("motion", "snappy")
        self.energy = float(style.get("energy", 0.6) or 0.0)
        self.audio_env = audio_env

        # images / contours / lines by id
        self.images = {}
        self.contours = {}
        self.lines = {}
        for k, im in enumerate(self.analysis.get("images") or []):
            iid = im.get("id", f"img{k}")
            px = im.get("pixels")
            if px is not None:
                px = np.asarray(px)
                if px.dtype == np.uint8:
                    px = px.astype(np.float32) / 255.0
                px = np.ascontiguousarray(px[..., :3], dtype=np.float32)
            self.images[iid] = dict(im, pixels=px)
            for ci, c in enumerate(im.get("contours") or []):
                cid = c.get("id", f"{iid}:c{ci}")
                pts = np.asarray(c.get("points") or [], np.float64).reshape(-1, 2)
                if len(pts) >= 2:
                    self.contours[cid] = (iid, pts, bool(c.get("closed", True)), float(c.get("area", 0.0)))
            for li, ln in enumerate(im.get("lines") or []):
                lid = ln.get("id", f"{iid}:l{li}")
                pts = np.array([ln.get("p0", [0, 0]), ln.get("p1", [1, 1])], np.float64)
                self.lines[lid] = (iid, pts, False, 0.0)
                self.lines.setdefault(f"{iid}:l{li}", self.lines[lid])

        # timing
        audio = plan.get("audio") or {}
        self.bpm = float(audio.get("bpm") or 120.0)
        beats = audio.get("beats")
        if not beats:
            step = 60.0 / max(self.bpm, 1.0)
            beats = list(np.arange(0.0, self.duration + 1e-6, step))
        self.beats = np.sort(np.asarray(beats, np.float64))
        downbeats = audio.get("downbeats")
        if not downbeats:
            downbeats = list(self.beats[::4])
        self.downbeats = np.sort(np.asarray(downbeats, np.float64))
        onsets = None
        if audio_env is not None and audio_env.get("onset") is not None:
            onsets = _peak_times(audio_env["onset"], self.fps)
        if (onsets is None or len(onsets) == 0) and (self.analysis.get("audio") or {}).get("onsets"):
            onsets = np.asarray(self.analysis["audio"]["onsets"], np.float64)
        if onsets is None or len(onsets) == 0:
            onsets = self.beats
        self.onsets = np.sort(np.asarray(onsets, np.float64))

        self.palette = [resolve_color(plan, c) for c in (style.get("palette") or ["#111111", "#f2f2f2", "#ff4d2e"])]
        self.bg_color = resolve_color(plan, style.get("background", "palette:0"), (0.07, 0.07, 0.07))

    # -- helpers
    def color(self, c, default=(1.0, 1.0, 1.0)):
        return resolve_color(self.plan, c, default)

    def accent(self, k: int = 0):
        if len(self.palette) > 2:
            return self.palette[2 + (k % (len(self.palette) - 2))]
        return self.palette[-1]

    def events(self, on) -> np.ndarray:
        if isinstance(on, (list, tuple, np.ndarray)):
            return np.sort(np.asarray(on, np.float64))
        if on == "downbeats":
            return self.downbeats
        if on == "onsets":
            return self.onsets
        return self.beats

    def env_value(self, key: str, fi: int):
        if self.audio_env is None:
            return None
        arr = self.audio_env.get(key)
        if arr is None or len(arr) == 0:
            return None
        return float(arr[min(max(fi, 0), len(arr) - 1)])

    def rng(self, *keys) -> np.random.Generator:
        return np.random.default_rng([self.seed & 0xFFFFFFFF] + [int(k) & 0xFFFFFFFF for k in keys])


# ----------------------------------------------------------------------------- compositing

_EYE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def affine(scale=1.0, pivot=(0.0, 0.0), dx=0.0, dy=0.0, ox=0.0, oy=0.0, rot=0.0) -> np.ndarray:
    """Sprite px -> frame px: place sprite at (ox, oy), scale/rotate about ``pivot``, then translate."""
    c, s = math.cos(rot) * scale, math.sin(rot) * scale
    A = np.array([[c, -s], [s, c]])
    px, py = pivot
    b = A @ np.array([ox - px, oy - py]) + np.array([px + dx, py + dy])
    return np.array([[A[0, 0], A[0, 1], b[0]], [A[1, 0], A[1, 1], b[1]]])


def composite(frame: np.ndarray, sprite: np.ndarray, color=None, M=None, opacity: float = 1.0,
              clip=None, blur: float = 0.0) -> None:
    """Draw a sprite into ``frame`` (HxWx3 float32, modified in place).

    sprite: (h, w) float32 alpha (needs ``color``) or (h, w, 4) float32 premultiplied RGBA.
    M: 2x3 affine, sprite px -> frame px (continuous coordinates, pixel edges at integers).
    clip: (x0, y0, x1, y1) frame px rectangle. blur: Gaussian sigma in px.
    """
    if opacity <= 1.0 / 512.0 or sprite is None or sprite.size == 0:
        return
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    M = _EYE if M is None else np.asarray(M, np.float64)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    pad = int(math.ceil(3.0 * blur)) + 1 if blur > 0.3 else 0
    x0 = int(math.floor(corners[:, 0].min())) - pad
    x1 = int(math.ceil(corners[:, 0].max())) + pad
    y0 = int(math.floor(corners[:, 1].min())) - pad
    y1 = int(math.ceil(corners[:, 1].max())) + pad
    # region to compute (frame-limited, plus blur pad)
    rx0, ry0 = max(x0, -pad), max(y0, -pad)
    rx1, ry1 = min(x1, W + pad), min(y1, H + pad)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    A = M[:, :2]
    is_translate = (pad == 0 and abs(A[0, 0] - 1) < 1e-9 and abs(A[1, 1] - 1) < 1e-9
                    and abs(A[0, 1]) < 1e-9 and abs(A[1, 0]) < 1e-9
                    and abs(M[0, 2] - round(M[0, 2])) < 1e-6 and abs(M[1, 2] - round(M[1, 2])) < 1e-6)
    if is_translate:
        tx, ty = int(round(M[0, 2])), int(round(M[1, 2]))
        sx0, sy0 = max(0, rx0 - tx), max(0, ry0 - ty)
        sx1, sy1 = min(w, rx1 - tx), min(h, ry1 - ty)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        warped = sprite[sy0:sy1, sx0:sx1]
        rx0, ry0 = sx0 + tx, sy0 + ty
        rx1, ry1 = sx1 + tx, sy1 + ty
    else:
        Mt = np.empty((2, 3), np.float64)
        Mt[:, :2] = A
        Mt[:, 2] = A @ np.array([0.5, 0.5]) + M[:, 2] - 0.5 - np.array([rx0, ry0])
        warped = cv2.warpAffine(sprite, Mt, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if pad:
            warped = cv2.GaussianBlur(warped, (0, 0), blur)
    # final clip to frame and clip rect
    cx0, cy0, cx1, cy1 = 0, 0, W, H
    if clip is not None:
        cx0, cy0 = max(cx0, int(math.floor(clip[0]))), max(cy0, int(math.floor(clip[1])))
        cx1, cy1 = min(cx1, int(math.ceil(clip[2]))), min(cy1, int(math.ceil(clip[3])))
    fx0, fy0, fx1, fy1 = max(rx0, cx0), max(ry0, cy0), min(rx1, cx1), min(ry1, cy1)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    warped = warped[fy0 - ry0:fy1 - ry0, fx0 - rx0:fx1 - rx0]
    region = frame[fy0:fy1, fx0:fx1]
    # same-shape ops only (broadcasting a (h,w,1) alpha over 3 channels is ~2x slower)
    if warped.ndim == 2:
        a = np.ascontiguousarray(warped * np.float32(opacity) if opacity < 1.0 else warped)
        a3 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        d = np.empty_like(a3)
        d[:] = np.asarray(color, np.float32)
        d -= region
        d *= a3
        region += d
    else:
        op = np.float32(opacity)
        a = np.ascontiguousarray(warped[..., 3] * op if opacity < 1.0 else warped[..., 3])
        a3 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        np.subtract(np.float32(1.0), a3, out=a3)
        region *= a3
        region += warped[..., :3] * op if opacity < 1.0 else warped[..., :3]


def translate(x, y) -> np.ndarray:
    return np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)]])


# ----------------------------------------------------------------------------- animation state

class Anim:
    __slots__ = ("opacity", "dx", "dy", "scale", "blur", "wipe_lo", "wipe_hi", "line_clip",
                 "trim_lo", "trim_hi", "flip", "underline", "visible")

    def __init__(self):
        self.opacity = 1.0
        self.dx = 0.0
        self.dy = 0.0
        self.scale = 1.0
        self.blur = 0.0
        self.wipe_lo = 0.0
        self.wipe_hi = 1.0
        self.line_clip = False
        self.trim_lo = 0.0
        self.trim_hi = 1.0
        self.flip = False
        self.underline = None
        self.visible = True


class Timing:
    """Resolved in/out windows of a layer."""

    def __init__(self, layer: dict, scene_t1: float | None = None):
        self.t0 = float(layer.get("t0", 0.0))
        self.t1 = float(layer.get("t1", self.t0 + 1.0))
        if self.t1 <= self.t0:
            self.t1 = self.t0 + 1e-3
        self.vin = layer.get("in") or {"verb": "cut"}
        self.vout = layer.get("out") or {"verb": "cut"}
        self.in_verb = str(self.vin.get("verb", "cut"))
        self.out_verb = str(self.vout.get("verb", "cut"))
        span = self.t1 - self.t0
        din = float(self.vin.get("dur", 0.4) or 0.0) if self.in_verb != "cut" else 0.0
        dout = float(self.vout.get("dur", 0.3) or 0.0) if self.out_verb != "cut" else 0.0
        self.din = max(0.0, min(din, span * 0.7))
        self.dout = max(0.0, min(dout, span - self.din))

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t1

    def p_in(self, t: float) -> float:
        return 1.0 if self.din <= 0 else clamp01((t - self.t0) / self.din)

    def p_out(self, t: float) -> float:
        return 0.0 if self.dout <= 0 else clamp01((t - (self.t1 - self.dout)) / self.dout)


def apply_in(st: Anim, verb: str, p: float, motion: str, U: dict, kind: str) -> None:
    """Block-level entrance. ``kind`` is text | shape | image (text handles per-word verbs itself)."""
    if p >= 1.0 or verb == "cut":
        return
    e = ease_in(p, motion)
    if kind == "text" and verb in ("word_stagger", "kinetic_split", "type_on"):
        return
    if verb == "stroke_draw" and kind != "shape":
        verb = "mask_wipe"
    if verb == "type_on":
        verb = "mask_wipe"
        e = math.floor(p * 12) / 12.0
    if verb == "word_stagger":
        verb = "slide_up" if kind == "image" else "fade"
    if verb == "kinetic_split":
        verb = "scale_pop"
    if verb == "fade":
        st.opacity *= clamp01(e)
    elif verb == "slide_up":
        st.dy += (1.0 - e) * U["slide"]
        st.opacity *= clamp01(p * 3.0)
        st.line_clip = kind == "text"
    elif verb == "slide_left":
        st.dx += (1.0 - e) * U["slide_x"]
        st.opacity *= clamp01(p * 2.5)
    elif verb == "scale_pop":
        st.scale *= max(0.0, e if motion == "glitch" else back_out(p, 2.2))
        st.opacity *= clamp01(p * 4.0)
    elif verb == "mask_wipe":
        st.wipe_hi = min(st.wipe_hi, clamp01(e))
    elif verb == "blur_in":
        st.blur += (1.0 - clamp01(e)) * U["blur"]
        st.opacity *= clamp01(e * 1.4)
    elif verb == "stroke_draw":
        st.trim_hi = min(st.trim_hi, clamp01(e))
    else:  # unknown verb -> fade
        st.opacity *= clamp01(e)


def apply_out(st: Anim, verb: str, p: float, motion: str, U: dict, kind: str) -> None:
    if p <= 0.0 or verb == "cut":
        return
    e = clamp01(ease_out(p, motion))
    if verb == "stroke_draw" and kind != "shape":
        verb = "mask_wipe"
    if verb == "fade":
        st.opacity *= 1.0 - e
    elif verb == "slide_up":
        st.dy -= e * U["slide"]
        st.opacity *= clamp01((1.0 - p) * 3.0)
        st.line_clip = kind == "text"
    elif verb == "slide_left":
        st.dx -= e * U["slide_x"]
        st.opacity *= 1.0 - e
    elif verb == "scale_down":
        st.scale *= 1.0 - 0.85 * e
        st.opacity *= 1.0 - e
    elif verb == "mask_wipe":
        st.wipe_lo = max(st.wipe_lo, e)
    elif verb == "blur_out":
        st.blur += e * U["blur"]
        st.opacity *= 1.0 - e
    elif verb == "stroke_draw":
        st.trim_lo = max(st.trim_lo, e)
    else:
        st.opacity *= 1.0 - e


def apply_emphasis(st: Anim, emphasis, t: float, fi: int, ctx: Ctx, tm: Timing, uid: int) -> None:
    for k, em in enumerate(emphasis or []):
        if not isinstance(em, dict):
            continue
        verb = em.get("verb")
        ev = ctx.events(em.get("on", "beats"))
        ev = ev[(ev >= tm.t0 - 1e-6) & (ev < tm.t1)]
        if verb == "pulse":
            env = decay_env(ev, t, tau=0.16)
            if env <= 0.0:
                continue
            amt = float(em.get("amount", 0.06) or 0.06)
            rms = ctx.env_value("rms", fi)
            if rms is not None:
                amt *= 0.4 + 1.2 * rms
            st.scale *= 1.0 + amt * env
        elif verb == "shake":
            env = decay_env(ev, t, tau=0.18)
            if env <= 0.02:
                continue
            amp = float(em.get("amount", 0.012) or 0.012)
            if amp > 0.2:      # tolerate pixel-ish amounts
                amp = 0.012
            r = ctx.rng(uid, 31 + k, fi)
            st.dx += float(r.uniform(-1, 1)) * amp * ctx.S * env
            st.dy += float(r.uniform(-1, 1)) * amp * ctx.S * env
        elif verb == "color_flip":
            n = int(np.searchsorted(ev, t + 1e-6, side="right"))
            if n % 2 == 1:
                st.flip = not st.flip
        elif verb == "underline_draw":
            start = tm.t0 + tm.din
            later = ev[ev >= start - 1e-6]
            te = float(later[0]) if len(later) else start
            if t >= te:
                st.underline = max(st.underline or 0.0, ease_in(clamp01((t - te) / 0.35), ctx.motion))
        # unknown emphasis verbs are ignored


def glitch_jitter(st: Anim, p_in: float, ctx: Ctx, uid: int, fi: int) -> None:
    if ctx.motion != "glitch" or p_in >= 1.0:
        return
    r = ctx.rng(uid, 77, fi)
    k = 1.0 - p_in
    st.dx += float(r.uniform(-1, 1)) * 0.02 * ctx.S * k
    if r.random() < 0.25 * k:
        st.opacity *= 0.35
