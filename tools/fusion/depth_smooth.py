#!/usr/bin/env python3
"""depth_smooth - the SMOOTH and SLIDE styles of the 3D shot cuts (orbit push / pullback / fly): a real camera move through two depth pictures, no card mosaic.

Why (QC of 22_batch-24gb v009 / v010, `review/qc_rNN_cuts_all.png`): every push / pullback / fly looked like the pivot's chunk mosaic plus blur. Here the picture
is NOT broken up. Each image is two LAYERS, both warped by the one camera through the depth map (inverse depth warp, no splat, no holes to hide):
    BACKGROUND layer   the frame with the subject filled out of it (`depth_orbit.subject_split`: push-pull of THIS frame, not the frame-0 LDI colour), per-pixel
                       depth = the clip's depth map with the subject removed and smoothed -> continuous parallax (near things move more than far things)
    SUBJECT layer      ONE flat card cut by the frame's own SOFT matte at the subject's median depth (the colour has the background mix taken out at its edge):
                       crisp, never chunked, never doubled, one position
and a per-pixel hole fill beyond a layer's footprint (blurred extension of its own edge). The matte of the cut is the SUBJECT cards' coverage in the very same
composite (colour and matte come out of one pass, so the type sandwich sees exactly what is drawn: the matte pass costs nothing extra).

    push       the camera dives toward A's subject (aim 1 = straight into its centre), A's subject card swells through the lens and fades by its zoom, A's
               background dissolves from the subject's screen position outward (soft iris, no tiles) while B, placed dz behind A, grows to its home and the camera
               settles on it. Frame 0 is A, the last frame is B.
    pullback   the push film played backwards between the swapped pictures: the camera backs away from A, B's picture arrives from the lens and settles.
    fly        the push with the volume_pose S-curve: lateral sweep, rise / dip, heading change and a bank; the hole swings sideways with the move.
    slide      the camera travels `dir` by one frame-span at the far depth: A leaves with depth parallax (the subject faster than the background), B enters from the
               other side, every pixel sorted by its own camera depth; no iris, nothing fades.

Interface = the scene methods the adapter (mograph/modules/transitions/depth_pivot.py) uses on the chunk scenes: `q(t)`, `set_frames(i1, i2)`, `image2_start()`,
`render_both(t)` -> (uint8 BGR frame, float32 matte), `nsub_at`. Homes are exact: q = 0 gives A's own frame, q = 1 B's own frame, mattes the source mattes.
CPU, numpy + cv2, about 0.1-0.2 s per sub-frame at 432 x 768.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402

STYLES = ("chunks", "smooth", "slide")
SMOOTH_MODES = ("push", "pullback", "fly")
ITER = 3                    # fixed-point steps of the inverse depth warp (smooth depth: converges in 2-3)
SLIDE_DEPTH = 0.22          # slide: fraction of the depth range kept (parallax between the subject and the background, not a collage)
SMOOTH_DZ = 3.6             # world distance of the back picture behind the front one: B starts at ~45 % of its size, not 30 %, so its subject is readable while it grows
SMOOTH_ZOOM_FADE = (3.1, 5.2)    # the front subject card fades by its zoom over this range (the chunk scenes use 2.5 - 4): it stays on screen a little longer through the dive
FLY_ROLL, FLY_AMP = 44.0, (1.15, 0.5)    # fly: peak bank (~37 deg on screen after the envelope) and sweep, stronger than the chunk fly so the bank reads while B is revealed
SUBJ_FEATHER = 0.24         # the same for the subject card where the frame cropped the subject (legs / arms): it ends in the frame's edge, so it dissolves over a longer run
FEATHER = 0.10              # a picture that has shrunk inside the frame fades into the diffused surround over this fraction of the frame width (no hard-edged rectangle)
MAX_SUB = 6                 # sub-frames of the motion blur at the fastest frames
I2_Q = {"push": 0.36, "fly": 0.36, "pullback": 0.30, "slide": 0.20}    # q at which image B is on screen enough to start playing its clip


def _sm01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def uses(op) -> bool:
    """does this OrbitParams ask for the smooth engine? (pivot has no smooth form: it stays on the cards)"""
    return getattr(op, "style", "chunks") in ("smooth", "slide") and op.mode in SMOOTH_MODES


def image2_start(op) -> int:
    """first output frame at which image B plays (the plan arithmetic of pivot_cut needs it without a scene)."""
    thr = I2_Q["slide" if op.style == "slide" else op.mode]
    fl = do.frame_list(op)
    for i, (t, _k) in enumerate(fl):
        if do.q_at(t, op, 1.0, 1.0) >= thr:
            return i
    return len(fl)


def _extend(rgb, alpha):
    """colour beyond a layer's footprint: the footprint's own colour diffused outward (normalised blur at 1/4 size), so nothing is black and nothing is a mirror."""
    if float(alpha.min()) > 0.999:
        return rgb
    h, w = alpha.shape
    s = 4
    hs, ws = max(h // s, 2), max(w // s, 2)
    a_s = cv2.resize(alpha, (ws, hs), interpolation=cv2.INTER_AREA)
    c_s = cv2.resize(rgb * alpha[..., None], (ws, hs), interpolation=cv2.INTER_AREA)
    ext = np.zeros_like(c_s)
    have = np.zeros((hs, ws), bool)
    for sig in (4.0, 12.0, 40.0):
        num = cv2.GaussianBlur(c_s, (0, 0), sig)
        den = cv2.GaussianBlur(a_s, (0, 0), sig)
        ok = (den > 0.02) & ~have
        ext[ok] = num[ok] / den[ok][:, None]
        have |= ok
    if not have.all():
        tot = float(a_s.sum())
        ext[~have] = (c_s.reshape(-1, 3).sum(0) / max(tot, 1e-6)) if tot > 1e-6 else 0.0
    ext = cv2.GaussianBlur(cv2.resize(ext, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 6.0)
    return rgb * alpha[..., None] + ext * (1.0 - alpha[..., None])


class SVol:
    """one image as a depth volume: static geometry from frame 0 (the clip's ONE depth map, like the card engine), live textures per clip frame."""

    def __init__(self, layer, video):
        self.H, self.W = layer.H, layer.W
        self.f, self.cx, self.cy = float(layer.f), float(layer.cx), float(layer.cy)
        self.frames, self.mattes = video
        subj = np.asarray(layer.subj, bool) if layer.subj is not None else np.zeros((self.H, self.W), bool)
        disp = layer.disp.astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        self.has_sub = bool(subj.any())
        if self.has_sub:
            core = cv2.erode(subj.astype(np.uint8), k3, iterations=2) > 0
            if not core.any():
                core = subj
            sdil = cv2.dilate(subj.astype(np.uint8), k3, iterations=3) > 0
            bgd = dd.push_pull(disp, ~sdil) if (~sdil).any() else disp
            self.Zs = float(np.median(dd.disp_to_z(disp[core], layer.params)))
            ys, xs = np.nonzero(subj)
            self.sx, self.sy = float(xs.mean()), float(ys.mean())
        else:
            bgd, self.Zs, self.sx, self.sy = disp, None, self.cx, self.cy
        z = dd.disp_to_z(bgd, layer.params).astype(np.float32)
        self.Zbg = np.maximum(cv2.GaussianBlur(z, (0, 0), 2.0), 0.2)
        self.Zmed = float(np.median(self.Zbg))
        self.Zl = float(np.median(self.Zbg[subj])) if self.has_sub else self.Zmed       # the background depth right behind the subject
        self.Zfar = float(np.percentile(self.Zbg, 90.0))
        self._cache: dict = {}

    def n(self) -> int:
        return len(self.frames)

    def idx(self, f) -> int:
        return int(min(max(int(f), 0), self.n() - 1))

    def tex(self, f):
        """{bg (H,W,3) f32 BGR, sub (H,W,4) premultiplied BGR + alpha, raw (H,W,3) u8, m (H,W) f32} of clip frame f (cached, last few)."""
        f = self.idx(f)
        hit = self._cache.get(f)
        if hit is not None:
            return hit
        img = self.frames[f].astype(np.float32)
        a = np.where(self.mattes[f] > 0.02, self.mattes[f], 0.0).astype(np.float32)
        fg, bg = do.subject_split(img, a)
        sub = np.concatenate([fg * a[..., None], a[..., None]], axis=2).astype(np.float32)
        hole = cv2.GaussianBlur(cv2.dilate((a > 0.03).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=4).astype(np.float32), (0, 0), 2.0)
        out = dict(bg=np.concatenate([bg, hole[..., None]], axis=2).astype(np.float32), sub=sub, raw=self.frames[f], m=np.asarray(self.mattes[f], np.float32), has=bool(a.max() > 0.02))
        if len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        self._cache[f] = out
        return out


class SmoothScene:
    def __init__(self, layers, op, video):
        import dataclasses
        kw = dict(vol_dz=min(float(op.vol_dz), SMOOTH_DZ), zoom_fade=SMOOTH_ZOOM_FADE)
        if op.mode == "fly":
            kw.update(roll_deg=max(float(op.roll_deg), FLY_ROLL), fly_amp=tuple(max(a, b) for a, b in zip(op.fly_amp, FLY_AMP)))
        op = dataclasses.replace(op, **kw)
        self.op = op
        self.W, self.H = layers[0].W, layers[0].H
        self.f, self.cx, self.cy = float(layers[0].f), float(layers[0].cx), float(layers[0].cy)
        self.vols = [SVol(layers[0], video[0]), SVol(layers[1], video[1])]
        self.slide = op.style == "slide"
        self.kind = op.mode
        self.rev = op.mode == "pullback" and not self.slide
        self.front, self.back = (self.vols[1], self.vols[0]) if self.rev else (self.vols[0], self.vols[1])
        self.uu, self.vv = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        self.aspect = self.H / float(self.W)
        self.fi = [0, 0]
        self._hole = None
        self.t_cur = 0.0
        d = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}.get(op.dir, (1.0, 0.0))
        if self.slide:                                                  # a lateral slide shows how much depth RANGE there is: squeeze it (0.22) so the near and far parts of a picture leave together enough that nothing is left uncovered
            for v in self.vols:
                v.Zbg = np.maximum(v.Zmed + SLIDE_DEPTH * (cv2.GaussianBlur(v.Zbg, (0, 0), 7.0) - v.Zmed), 0.2)      # and soften its edges: no stretched wedges at a depth step
                if v.has_sub:                                   # the subject card sits just in front of what is behind it: it leads the slide a little, the patch filled in under it stays hidden
                    v.Zs = 0.88 * (v.Zmed + SLIDE_DEPTH * (v.Zl - v.Zmed))
                v.Zfar = v.Zmed + SLIDE_DEPTH * (v.Zfar - v.Zmed)
        zfar = 0.5 * (self.vols[0].Zfar + self.vols[1].Zfar)
        span = (self.W if d[1] == 0.0 else self.H) / self.f * zfar
        self.slide_vec = np.array([d[0] * span, d[1] * span, 0.0])
        self.back_t = self.slide_vec if self.slide else np.array([0.0, 0.0, float(op.vol_dz)])
        s = self.front
        self._aim = np.zeros(2)
        if s.has_sub and not self.slide:
            self._aim = np.array([(s.sx - s.cx) / s.f * s.Zs, (s.sy - s.cy) / s.f * s.Zs])
        # the iris centre: the front subject's screen position (the move heads into it), halfway to the frame centre for a subject-less frame
        sx, sy = (s.sx, s.sy) if s.has_sub else (self.cx, self.cy)
        self.iris0 = np.array([0.75 * sx + 0.25 * self.cx, 0.75 * sy + 0.25 * self.cy])

    # --- timing + pose
    def q(self, t):
        return do.q_at(t, self.op, 1.0, 1.0)

    def image2_start(self):
        return image2_start(self.op)

    def set_frames(self, f1, f2):
        self.fi = [int(f1), int(f2)]

    def nsub_at(self, t):
        return 1

    def pose(self, q):
        """CamG of the film progress q (0 = image A home, 1 = image B home)."""
        op = self.op
        if self.slide:
            return do.CamG(self.slide_vec * float(q), 0.0, 0.0, 0.0, 1.0)
        qq = 1.0 - q if self.rev else q
        mode = "fly" if op.mode == "fly" else "push"
        p = do.volume_pose(mode, qq, op, self.aspect)
        pos = np.asarray(p["pos"], np.float64).copy()
        if self.front.has_sub and op.aim > 0.0:
            w = 1.0 - float(_sm01((qq - 0.5) / 0.45))
            k = float(op.aim) * (0.5 if mode == "fly" else 1.0)
            pos[:2] += self._aim * (pos[2] / max(self.front.Zs, 1e-3)) * w * k
        return do.CamG(pos, p["yaw"], p["pitch"], p["roll"], p["fk"])

    def _iris(self, q):
        """(centre px, radius norm) of the soft hole in the front volume's background."""
        op = self.op
        qq = 1.0 - q if self.rev else q
        c = self.iris0.copy()
        if op.mode == "fly":
            c[0] += 0.28 * self.W * math.sin(2.0 * math.pi * qq)
        R = -op.hole_soft + (op.hole_max + op.hole_soft) * float(_sm01((qq - op.hole0) / max(op.hole1 - op.hole0, 1e-6)))
        return c, R

    # --- warps
    def _dirs(self, cam):
        rx = (self.uu - self.cx) / (self.f * cam.fk)
        ry = (self.vv - self.cy) / (self.f * cam.fk)
        R = cam.R
        return (R[0, 0] * rx + R[0, 1] * ry + R[0, 2], R[1, 0] * rx + R[1, 1] * ry + R[1, 2], np.maximum(R[2, 0] * rx + R[2, 1] * ry + R[2, 2], 1e-4))

    def _bg(self, vol, tex, cam, d, toff):
        """background layer through the camera: (rgb f32, footprint alpha, camera depth)."""
        dx, dy, dz = d
        o = cam.pos - toff
        Zh = vol.Zbg
        for _ in range(ITER):
            lam = (Zh - o[2]) / dz
            xs = (vol.f * (o[0] + lam * dx) / Zh + vol.cx).astype(np.float32)
            ys = (vol.f * (o[1] + lam * dy) / Zh + vol.cy).astype(np.float32)
            Zh = np.maximum(cv2.remap(vol.Zbg, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0.2)
        lam = (Zh - o[2]) / dz
        xs = (vol.f * (o[0] + lam * dx) / Zh + vol.cx).astype(np.float32)
        ys = (vol.f * (o[1] + lam * dy) / Zh + vol.cy).astype(np.float32)
        got = cv2.remap(tex["bg"], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        rgb = got[..., :3]
        self._hole = got[..., 3]                                      # where the subject stood (the filled patch), warped with the layer
        edge = np.minimum(np.minimum(xs, vol.W - 1 - xs), np.minimum(ys, vol.H - 1 - ys))
        ramp = self._feather(Zh[self.H // 2, self.W // 2] / max(float(lam[self.H // 2, self.W // 2]), 1e-3), vol)
        fa = np.clip((edge + 1.5) / (1.5 + ramp), 0.0, 1.0) * (lam > self.op.near_clip)
        return rgb, fa.astype(np.float32), lam.astype(np.float32)

    def _feather(self, zoom, vol, frac=None):
        """source px over which a picture's edge fades: 0 at home scale (exact), FEATHER x the width once it has shrunk (zoom < 0.85) - the shrunk picture dissolves into the surround instead of showing its rectangle."""
        return float((FEATHER if frac is None else frac) * vol.W * _sm01((1.0 - zoom) / 0.15))

    def _sub(self, vol, tex, cam, d, toff):
        """the subject card (one flat plane at its median depth, the frame's soft matte as alpha): (rgb f32, alpha, camera depth) or None."""
        if not (vol.has_sub and tex["has"]):
            return None
        dx, dy, dz = d
        o = cam.pos - toff
        lam = (vol.Zs - o[2]) / dz
        xs = (vol.f * (o[0] + lam * dx) / vol.Zs + vol.cx).astype(np.float32)
        ys = (vol.f * (o[1] + lam * dy) / vol.Zs + vol.cy).astype(np.float32)
        zoom = vol.Zs / max(float(lam[self.H // 2, self.W // 2]), 1e-3)
        pm = cv2.remap(tex["sub"], xs, ys, cv2.INTER_CUBIC if zoom > 1.15 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = np.clip(pm[..., 3], 0.0, 1.0) * (lam > self.op.near_clip)
        ramp = self._feather(zoom, vol, SUBJ_FEATHER)
        if ramp > 0.5:        # a subject the frame cropped (legs, arms) ends in the frame's edge: once the card has shrunk that edge would be a hard line, so it fades out instead
            edge = np.minimum(np.minimum(xs, vol.W - 1 - xs), np.minimum(ys, vol.H - 1 - ys))
            a = a * _sm01(edge / ramp).astype(np.float32)
        rgb = np.clip(pm[..., :3] / np.maximum(pm[..., 3:4], 1e-3), 0.0, 255.0)
        return rgb.astype(np.float32), a.astype(np.float32), lam.astype(np.float32)

    # --- one pose
    def _compose(self, q):
        """-> (BGR float32 0..255, matte float32 0..1) of the film progress q (homes are NOT shortcut here: render_both does that)."""
        op = self.op
        cam = self.pose(q)
        d = self._dirs(cam)
        fv, bv = self.front, self.back
        fi_f = self.fi[1] if self.rev else self.fi[0]
        fi_b = self.fi[0] if self.rev else self.fi[1]
        tf, tb = fv.tex(fi_f), bv.tex(fi_b)
        zero = np.zeros((self.H, self.W), np.float32)
        if self.slide:
            return self._compose_slide(q, cam, d, tf, tb)
        b_rgb, b_fa, _ = self._bg(bv, tb, cam, d, self.back_t)
        base = _extend(b_rgb, b_fa)
        bs = self._sub(bv, tb, cam, d, self.back_t)
        bs_a = zero
        if bs is not None:
            bs_a = bs[1]
            base = bs[1][..., None] * bs[0] + (1.0 - bs[1][..., None]) * base
        f_rgb, f_fa, f_zc = self._bg(fv, tf, cam, d, np.zeros(3))
        hole_w = self._hole
        f_rgb = _extend(f_rgb, f_fa)                       # a front layer that does not reach the frame edge (the camera has shifted) is extended, never shown as a hard-edged picture
        f_fa = (f_zc > self.op.near_clip).astype(np.float32)
        c, R = self._iris(q)
        rmax = max(math.hypot(c[0], c[1]), math.hypot(self.W - c[0], c[1]), math.hypot(c[0], self.H - c[1]), math.hypot(self.W - c[0], self.H - c[1]))
        r = np.hypot(self.uu - c[0], self.vv - c[1]) / rmax
        iris = 1.0 - _sm01((R - r) / op.hole_soft + 0.5)
        lens = _sm01((f_zc - op.near_clip) / max(op.near_fade - op.near_clip, 1e-6))
        a_fb = f_fa * lens * iris
        if fv.has_sub and tf["has"]:       # the subject card leaves by its zoom: the patch filled in where it stood leaves WITH it (a window onto B in its shape), not as a smear
            zcs = max(float(fv.Zs - cam.pos[2]), 1e-3)
            gone = 1.0 - float(_sm01((op.zoom_fade[1] - fv.Zs / zcs) / (op.zoom_fade[1] - op.zoom_fade[0])))
            a_fb = a_fb * (1.0 - np.clip(hole_w, 0.0, 1.0) * gone)
        mid = a_fb[..., None] * f_rgb + (1.0 - a_fb[..., None]) * base
        fs = self._sub(fv, tf, cam, d, np.zeros(3))
        a_fs = zero
        out = mid
        if fs is not None:
            zoom = fv.Zs / np.maximum(fs[2], 1e-3)
            g = _sm01((op.zoom_fade[1] - zoom) / (op.zoom_fade[1] - op.zoom_fade[0])) * _sm01((fs[2] - op.near_clip) / max(op.near_fade - op.near_clip, 1e-6))
            a_fs = fs[1] * g.astype(np.float32)
            out = a_fs[..., None] * fs[0] + (1.0 - a_fs[..., None]) * mid
        matte = a_fs + (1.0 - a_fs) * ((1.0 - a_fb) * bs_a)
        return self._roll(out, cam), self._roll(np.clip(matte, 0.0, 1.0), cam)

    def _compose_slide(self, q, cam, d, tf, tb):
        """A and B side by side in the world; every pixel takes the nearest of the four layers (bg / subject of each image) by camera depth."""
        fa_, fb_ = self.vols
        ta, tb_ = (tf, tb)
        gate = float(_sm01(q / 0.12))                                  # B is not allowed to peek in front of A's far content in the first frames
        layers = []
        for vol, tex, toff, bias, gain in ((fa_, ta, np.zeros(3), 0.0, 1.0), (fb_, tb_, self.back_t, -0.25 * float(_sm01(q)), gate)):
            rgb, fa, zc = self._bg(vol, tex, cam, d, toff)
            layers.append((rgb, fa * gain, zc + bias, False))
            s = self._sub(vol, tex, cam, d, toff)
            if s is not None:
                layers.append((s[0], s[1] * gain, s[2] + bias - 0.15, True))
        zs = np.stack([np.where(l[1] > 0.02, l[2], 1e9) for l in layers])
        order = np.argsort(zs, axis=0)
        rgbs = np.stack([l[0] for l in layers])
        als = np.stack([l[1] for l in layers])
        subj = np.array([1.0 if l[3] else 0.0 for l in layers], np.float32)
        out = np.zeros((self.H, self.W, 3), np.float32)
        matte = np.zeros((self.H, self.W), np.float32)
        T = np.ones((self.H, self.W), np.float32)
        for k in range(len(layers)):
            idx = order[k]
            a = np.take_along_axis(als, idx[None], 0)[0]
            c = np.take_along_axis(rgbs, idx[None, ..., None].repeat(3, -1), 0)[0]
            out += (T * a)[..., None] * c
            matte += T * a * subj[idx]
            T = T * (1.0 - a)
        cover = np.clip(1.0 - T, 0.0, 1.0)
        norm = out / np.maximum(cover, 1e-4)[..., None]                 # out is premultiplied by the coverage
        res = _extend(norm, cover) if float(cover.min()) < 0.999 else norm
        return res, np.clip(matte, 0.0, 1.0)

    def _roll(self, img, cam):
        if abs(cam.roll) < 1e-6 and abs(cam.fk - 1.0) < 1e-9:
            return img
        M = cv2.getRotationMatrix2D((self.cx, self.cy), cam.roll, 1.0 / cam.fk)
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # --- frame
    def _speed_px(self, t):
        """largest on-screen travel (px) of a probe point over one shutter, for the sub-frame count."""
        op = self.op
        ta, tb = t - 0.5 * op.shutter, t + 0.5 * op.shutter
        ca, cb = self.pose(self.q(ta)), self.pose(self.q(tb))
        zs = self.front.Zs if self.front.has_sub else self.front.Zmed
        best = 0.0
        for X in (-0.5 * self.W / self.f * zs, 0.5 * self.W / self.f * zs):
            for Y in (-0.5 * self.H / self.f * zs, 0.0, 0.5 * self.H / self.f * zs):
                P = np.array([X, Y, zs])
                us = []
                for cam in (ca, cb):
                    pc = cam.R.T @ (P - cam.pos)
                    z = max(pc[2], op.near_fade)
                    us.append((self.f * pc[0] / z, self.f * pc[1] / z))
                best = max(best, math.hypot(us[1][0] - us[0][0], us[1][1] - us[0][1]))
        return best

    def render_both(self, t):
        """-> (uint8 BGR frame, float32 matte (H,W) 0..1) at output time t (frames). Homes are the sources exactly."""
        op = self.op
        self.t_cur = float(t)
        q = self.q(t)
        if q <= 0.0 or q >= 1.0:
            v, f = (self.vols[0], self.fi[0]) if q <= 0.0 else (self.vols[1], self.fi[1])
            tx = v.tex(f)
            return np.ascontiguousarray(tx["raw"]), tx["m"].copy()
        px = self._speed_px(t) * op.shutter
        n = int(np.clip(math.ceil(px / max(op.sub_px, 1.0)), 1, MAX_SUB))
        acc = np.zeros((self.H, self.W, 3), np.float32)
        macc = np.zeros((self.H, self.W), np.float32)
        for i in range(n):
            tt = t + op.shutter * ((i + 0.5) / n - 0.5) if n > 1 else t
            qi = float(np.clip(self.q(tt), 0.0, 1.0))
            fr, m = self._compose(qi)
            acc += fr
            macc += m
        return np.clip(acc / n + 0.5, 0, 255).astype(np.uint8), macc / n

    def render_t(self, t, nsub=1):
        return self.render_both(t)[0]

    def render_matte_t(self, t, nsub=1):
        return self.render_both(t)[1]
