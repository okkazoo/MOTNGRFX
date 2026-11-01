"""depth_type: a piece's words as FLAT TYPE PLANES inside the depth-orbit pivot scene (CPU, numpy + OpenCV).

17_depth-dolly v014. User: "creating a graphic on top ... the camera move over the text transition with the depth video
transformation so they look in sync". The depth scene's camera supplies ALL the motion: the type engine only renders a
STATIC matte (camera=None, placement yaw 0, white on black), and the word is a flat textured quad in the image's own volume:

  * SEARCHING is attached to image 1's volume, A LIGHT to image 2's (same RigidLayer transform as its image), at depth
    `depth` x the subject's median Z (default 0.8 = between the camera and the runner card), so it floats in front of the runner;
  * the quad is NOT camera-facing: its four corners are projected with the very same pose / pinhole as the cards
    (u = f x / z + cx) and the texture is warped by that homography (2x supersampled, area downscaled), so it turns in
    perspective with its image. At the image's home pose with yaw 0 the quad maps 1:1 onto the frame (a flat 2D overlay);
    the piece's placement yaw (+-16 deg) becomes the plane's yaw about its vertical centre axis (+ = right side nearer);
  * motion blur: the plane is warped at n poses spread over the shutter (n from the corner travel, ~2 px steps) and averaged
    = the same box-filtered shutter as the cards' continuous blur, with the same camera poses (scene.q at blur_t +- shutter/2);
  * the form-in / form-out of the word is the engine's own (lanes, 'lines'), retimed from the module's clock to the camera
    progress q: the word on image 1 is formed (hit pose) until q0, then plays its form-out up to the moment image 2 starts to
    enter (q_split); the word on image 2 plays its form-in from q_split to q1 and is formed before the landing;
  * ink colour from the MEASURED image region behind the word at home (tools/fusion/legible_color.choose: the piece's palette
    first, then the lightness / hue shifts, extremes, outline / shadow / scrim support when nothing reaches the WCAG target).

Hooked in by `depth_orbit.py render --type P06:SEARCHING,A LIGHT` (PivotScene.typ; layer(which, q, blur_t) -> premultiplied BGR + alpha).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import legible_color as LC  # noqa: E402

MATTE_SIZE = (1152, 2048)         # type render size (multiples of 64, 9:16): world 540 x 960, scaled to the depth frame
SS = 2                            # texture / warp supersampling
Q_OUT0 = 0.02                     # word on image 1: form-out starts at this camera progress (formed before)
Q_IN1 = 0.93                      # v014 style: word on image 2 fully formed at this camera progress (before the landing at 1.0)
Q_OUT1 = 0.58                     # organic style: word on image 1 gone at this camera progress (image 2 is opaque from ~0.63)
FORM_FRAMES = 18.0                # organic style: word on image 2 forms over this many frames after the landing (0.75 s)
STEP_PX = 2.0                     # motion-blur sub-pose spacing (px of the fastest corner)
MAX_SUB = 40
NEAR = 0.02


# ----------------------------------------------------------------------------------------------- spec / flag
def parse_flag(s):
    """'P06:SEARCHING,A LIGHT' -> ('P06', ['SEARCHING', 'A LIGHT']): word on image 1, word on image 2."""
    m = re.match(r"^\s*(P\d+)\s*:\s*(.+?)\s*$", s or "")
    words = [w.strip() for w in m.group(2).split(",") if w.strip()] if m else []
    if not m or len(words) != 2:
        raise ValueError("--type wants 'P06:WORD ON IMAGE 1,WORD ON IMAGE 2', got %r" % (s,))
    return m.group(1), words


PIECE_PIN = {"P06": "P06_v003.json"}            # the hand-designed piece the approved depth looks (g01-g04) were made from; P06_v004-v011 are open-model re-designs with other modules (tests broke on them 2026-10-28)


def piece_file(pid):
    pin = os.path.join(ROOT, "15_fusion", "pieces", PIECE_PIN.get(pid, ""))
    if PIECE_PIN.get(pid) and os.path.isfile(pin):
        return pin
    fs = sorted(glob.glob(os.path.join(ROOT, "15_fusion", "pieces", "%s_v*.json" % pid)))
    if not fs:
        raise FileNotFoundError("no piece json for %s in 15_fusion/pieces" % pid)
    return fs[-1]


def find_word(typ, text):
    for w in typ["words"]:
        if w["text"].strip().upper() == text.strip().upper():
            return w
    raise KeyError("word %r not in the piece's type.words" % text)


# ----------------------------------------------------------------------------------------------- the type matte
class WordType:
    """One word of a piece rendered by the type engine as a STATIC white-on-black matte (single-word scene, camera None,
    placement yaw 0 - the yaw goes onto the plane). `info` = start / hit / release / end of the module clock (s).
    entry / exit: the module sides the 'lines' strands arrive from / leave to ('left' | 'right' | 'up' | 'down'; None = the piece's
    own). light: the `wght` axis is held at the low end of the word's own range (a lighter, less heavy face)."""

    def __init__(self, typ, word, size=MATTE_SIZE, entry=None, exit=None, light=False, home=False):
        from tools.fusion import spec as S
        w = dict(word)
        pl = dict(w.get("placement") or {})
        self.yaw = float(pl.get("yaw", 0.0))
        pl["yaw"] = 0.0
        w["placement"] = pl
        if entry:
            w["entry"] = entry
        if exit:
            w["exit"] = exit
        if light and isinstance(w.get("axes"), dict) and "wght" in w["axes"]:
            v = w["axes"]["wght"]
            w["axes"] = dict(w["axes"], wght=(v[0] if isinstance(v, (list, tuple)) else v))
        pal = typ.get("palette") or {}
        self.palette = pal
        white = dict(bg=[0, 0, 0], fg=[255, 255, 255], accents=[[255, 255, 255]] * 3)
        dur = float(w.get("t1", 2.0)) - float(w.get("t0", 0.0)) + 1.5
        spec = dict(size=list(size), fps=24, duration=dur, camera=None, palette=white, words=[w], transitions="auto",
                    **{k: typ[k] for k in ("seed", "energy", "mood", "line", "bpm") if k in typ})
        self.size = tuple(size)
        self.sc = S.build_scene(spec, self.size)
        self.info = self.sc.info["words"][0]
        self.text = w["text"]
        self._cache = {}
        self._bbox = None
        self.home = bool(home)          # True: the cam3d HOME camera (placement dx / dy / ribbons land where the world says); False: the engine's demo camera, which re-centres a lone word

    def matte(self, tau):
        """float32 (h, w) 0..1 at module time tau, or None when the word is not on screen."""
        if tau <= self.info["start"] - 1e-6 or tau >= self.info["end"] - 1e-6:
            return None
        k = int(round(tau * 1000))
        if k not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            cam = self.sc.chain
            if self.home:
                from mograph.render import cam3d
                cam3d.set_world(*self.sc.world)
                cam = cam3d.Camera.home()
            f = self.sc.seq.frame(tau, cam, size=self.size, S=1)
            self._cache[k] = f.max(-1).astype(np.float32) / 255.0
        return self._cache[k]

    def bbox(self):
        """(x0, x1, y0, y1) of the formed word (hold pose) in matte pixels."""
        if self._bbox is None:
            ys, xs = np.nonzero(self.matte(self.info["hit"]) > 0.1)
            self._bbox = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        return self._bbox


def place_matte(matte, W, H, ss=SS, scale=1.0, center=None, bbox=None):
    """matte (world aspect) -> alpha float32 (ss*H, ss*W). scale 1 + no centre: scaled to the frame width, centred (a flat 2D overlay
    of the piece's layout). scale s: the word is drawn s times as large (home px = matte px * W / mw * s); center (cx, cy) home px:
    the centre of `bbox` (matte px; default the matte's centre) goes there."""
    mh, mw = matte.shape
    kk = scale * W / mw * ss
    r = cv2.resize(matte, (max(1, int(round(mw * kk))), max(1, int(round(mh * kk)))), interpolation=cv2.INTER_AREA if kk < 1 else cv2.INTER_LINEAR)
    rh, rw = r.shape
    if bbox is None:
        bx, by = rw / 2.0, rh / 2.0
    else:
        bx, by = 0.5 * (bbox[0] + bbox[1]) * rw / mw, 0.5 * (bbox[2] + bbox[3]) * rh / mh
    cx, cy = (W / 2.0, H / 2.0) if center is None else center
    ox, oy = int(round(ss * cx - bx)), int(round(ss * cy - by))
    out = np.zeros((ss * H, ss * W), np.float32)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + rw, ss * W), min(oy + rh, ss * H)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = r[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out


def to_tex(matte, W, H, ss=SS):
    """v014 placement: the matte scaled to the frame width and centred (kept for the flat-overlay checks)."""
    return place_matte(matte, W, H, ss)


# ----------------------------------------------------------------------------------------------- ink colour
def measure_bg(col_bgr, alpha_home, hide=None):
    """(hist, bg_lin, y, weight) of the image region behind the word (letters count LC_BEHIND_W, the ring around them 1). hide
    (0..1, e.g. the runner matte of the frame where the word is read): pixels the runner covers do not count (the word is behind him)."""
    h, w = alpha_home.shape
    lin = LC.srgb_to_lin(col_bgr[..., ::-1].astype(np.float32) / 255.0)
    y = LC.rel_lum(lin)
    behind = alpha_home > 0.25
    r = max(3, int(round(LC.LC_RING * max(h, w))))
    ring = cv2.dilate(behind.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool) & ~behind
    wgt = (behind * LC.LC_BEHIND_W + ring * 1.0).astype(np.float32)
    if hide is not None:
        keep = 1.0 - np.clip(np.asarray(hide, np.float32), 0.0, 1.0)
        if (wgt * keep).sum() > 0:
            wgt = wgt * keep
    if wgt.sum() <= 0:
        wgt = np.ones_like(y)
    hist = LC._hist(y, wgt)
    bg_lin = (lin * wgt[..., None]).sum((0, 1)) / wgt.sum()
    return hist, bg_lin, y, wgt


def choose_ink(col_bgr, alpha_home, palette, cap_norm=1000.0, target=None, hide=None):
    """-> dict(bgr 0..255, kind, support, cover, ratio_mean, share_ok, target, name). The piece's fg first, then its accents (a palette
    colour is used as is when it passes legible_color's test); else legible_color.choose's lightness / hue / extreme / support."""
    hist, bg_lin, y, wgt = measure_bg(col_bgr, alpha_home, hide)
    target = LC.target_for(cap_norm) if target is None else float(target)
    cands = [("fg", palette.get("fg", [235, 240, 255]))] + [("accent%d" % i, a) for i, a in enumerate(palette.get("accents") or [])]
    res = []
    for name, c in cands:
        lin = LC.srgb_to_lin(np.asarray(c, np.float32) / 255.0)
        ch = LC.choose(hist, lin, bg_lin, target)
        res.append((name, c, ch))
    ok = [r for r in res if r[2]["kind"] == "design"]
    if ok:
        name, c, ch = ok[0] if ok[0][0] == "fg" else max(ok, key=lambda r: r[2]["cover"])
    else:
        name, c, ch = res[0]                                     # fg's own fallback (lightness / hue / extreme / support)
    srgb = LC.lin_to_srgb(ch["lin"])
    y_ink = float(LC.rel_lum(ch["lin"]))
    ratio = LC.contrast_ratio(y_ink, y)
    m = wgt > 0
    return dict(bgr=(srgb[::-1] * 255.0).astype(np.float32), kind=ch["kind"], support=ch["support"], cover=float(ch["cover"]), name=name,
                ratio_mean=float((ratio[m] * wgt[m]).sum() / wgt[m].sum()), share_ok=float(wgt[m & (ratio >= target)].sum() / wgt[m].sum()),
                target=target, hex=LC.hex_of(srgb), y_ink=y_ink)


def support_alpha(a, kind, ss=SS):
    """alpha float32 of the support painted UNDER the letters (opposite luminance of the ink): outline / shadow / scrim."""
    if kind == "outline":
        r = max(1, int(round(1.5 * ss)))
        d = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return np.clip(cv2.GaussianBlur(d, (0, 0), 0.7 * ss), 0, 1) * LC.LC_SUP_OPACITY["outline"]
    if kind == "shadow":
        M = np.float32([[1, 0, 2 * ss], [0, 1, 2 * ss]])
        sh = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
        return np.clip(cv2.GaussianBlur(sh, (0, 0), 2.0 * ss), 0, 1) * LC.LC_SUP_OPACITY["shadow"]
    if kind == "scrim":
        r = 4 * ss
        d = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return np.clip(cv2.GaussianBlur(d, (0, 0), 8.0 * ss) * 1.6, 0, 1) * LC.LC_SUP_OPACITY["scrim"]
    return None


def premult_tex(a, ink):
    """alpha (h,w) + ink dict -> premultiplied BGRA float32 (h,w,4): colour in 0..255 x alpha, alpha 0..1 (support underneath)."""
    rgb = a[..., None] * ink["bgr"][None, None]
    al = a.copy()
    if ink["support"]:
        sa = support_alpha(a, ink["support"])
        lev = LC.LC_SUP_DARK if ink["y_ink"] > 0.18 else LC.LC_SUP_LIGHT
        rgb = rgb + (sa * (1.0 - a))[..., None] * np.float32(lev * 255.0)
        al = a + sa * (1.0 - a)
    return np.concatenate([rgb, al[..., None]], -1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- plane geometry
def _psi_R(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)


class Plane:
    """A flat quad in an image's home camera space: the frame rectangle at depth zt (so at home it covers the frame 1:1), turned by
    `yaw_deg` about the vertical axis through (pivot_x, frame centre row) at depth zt (default the frame centre), then carried by the
    image's rigid transform (world = t + Rk (s X)). points(dx, dyaw): the same plane with an extra local translation along x (world units
    of the image's home space) and extra yaw (deg): the secondary motion of the word on top of the image's own."""

    def __init__(self, layer, zt, yaw_deg, pivot_x=None):
        self.W, self.H, self.f, self.cx, self.cy = layer.W, layer.H, layer.f, layer.cx, layer.cy
        self.zt, self.yaw = float(zt), float(yaw_deg)
        hx, hy = (-0.5, self.W - 0.5), (-0.5, self.H - 0.5)
        P = np.array([[(x - self.cx) / self.f * zt, (y - self.cy) / self.f * zt, zt] for y in hy for x in hx], np.float64)  # TL TR BL BR
        self.P0 = P[[0, 1, 3, 2]]                                                                                              # TL TR BR BL
        C = self.P0.mean(0)
        if pivot_x is not None:
            C = np.array([(pivot_x - self.cx) / self.f * zt, C[1], zt])
        self.C = C
        self.P = self.points()

    @property
    def width(self):
        """frame width in the plane's world units."""
        return self.W * self.zt / self.f

    def points(self, dx=0.0, dyaw=0.0):
        P = self.C + (self.P0 - self.C) @ _psi_R(math.radians(self.yaw + dyaw)).T
        P[:, 0] += dx
        return P

    def project(self, Rcam, pos, Rk, t, s=1.0, dx=0.0, dyaw=0.0):
        """-> (4,2) corner pixel coordinates (frame index coords, as the cards use) or None when a corner is behind the near plane."""
        world = np.asarray(t, np.float64) + (self.points(dx, dyaw) * s) @ Rk.T
        cam = (world - np.asarray(pos, np.float64)) @ Rcam
        zc = cam[:, 2]
        if (zc <= NEAR).any():
            return None
        return np.stack([self.f * cam[:, 0] / zc + self.cx, self.f * cam[:, 1] / zc + self.cy], -1)

    def warp(self, tex, dst):
        """premultiplied BGRA tex (ss*H, ss*W, 4) warped onto the frame by the homography of the 4 projected corners; -> (H, W, 4)."""
        th, tw = tex.shape[:2]
        ss = tw // self.W
        src = np.float32([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
        d2 = np.float32(dst * ss + (ss - 1) / 2.0)                     # frame index coords -> the ss-times grid
        M = cv2.getPerspectiveTransform(src, d2)
        out = cv2.warpPerspective(tex, M, (self.W * ss, self.H * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.resize(out, (self.W, self.H), interpolation=cv2.INTER_AREA) if ss > 1 else out


# ----------------------------------------------------------------------------------------------- placement
def place_beside(runner, ww, W, hidden=None, m_frac=0.04, hid_max=0.30):
    """-> (cx, right): the horizontal centre (home px) of a word of width `ww` on the side of the frame the runner leaves free (`right` = the
    word goes to the right of him), the whole word kept inside the frame (margin m_frac x W). hidden(cx) -> share of the letters behind the
    runner at that centre: when given, the word slides toward the free edge until at most `hid_max` of it is behind him."""
    ys, xs = np.nonzero(np.asarray(runner) > 0.5)
    if not len(xs):
        return W / 2.0, True
    r0, r1 = float(xs.min()), float(xs.max())
    right = 0.5 * (r0 + r1) < W / 2.0
    lo, hi = (r1, float(W)) if right else (0.0, r0)
    m = m_frac * W
    cx = float(np.clip(0.5 * (lo + hi), ww / 2 + m, W - ww / 2 - m))
    if hidden is None:
        return cx, right
    cx_lim = (W - ww / 2 - m) if right else (ww / 2 + m)
    for _ in range(12):
        if hidden(cx) <= hid_max or abs(cx - cx_lim) < 0.5:
            break
        cx = min(cx + 0.03 * W, cx_lim) if right else max(cx - 0.03 * W, cx_lim)
    return cx, right


# ----------------------------------------------------------------------------------------------- springs
def spring_response(target, ts, omega, zeta, x0=0.0, v0=0.0, sub=4):
    """damped spring x'' = w^2 (target - x) - 2 zeta w x' sampled on the grid ts (frames; omega in rad / frame), semi-implicit Euler with
    `sub` steps per grid step. zeta < 1 overshoots a little and settles (the organic follow-through), zeta = 1 never overshoots."""
    x, v = float(x0), float(v0)
    out = np.empty(len(ts))
    for i in range(len(ts)):
        if i:
            dt = (ts[i] - ts[i - 1]) / sub
            for _ in range(sub):
                v += (omega * omega * (target[i] - x) - 2.0 * zeta * omega * v) * dt
                x += v * dt
        out[i] = x
    return out


def _smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


# ----------------------------------------------------------------------------------------------- the layer
class TypeLayer:
    """The two words in a PivotScene. `layer(which, q, blur_t, t)` -> (premultiplied BGR float32 0..255 (H,W,3), alpha (H,W)) of the
    word on image `which` (0 / 1) at camera progress q / output time t (frames; the move starts at t=0 and lands at t=travel), or None.

    mode  'behind_subject' (default; compose's `behind` recipe: type over the footage, the subject over the type by its matte): the plane
          sits between the runner card and the background (depth = zs + depth * (zb - zs), depth default 0.5) and PivotScene composites
          background cards -> type -> runner card (his per-frame matte), the weights of the image apply to the whole sandwich;
          'front' (v014): the plane is in front of the runner (depth x zs, default 0.8) and drawn over the finished image.
    style 'organic' (default): word 1 formed on image 1's hold, then CARRIED in the direction of the turn (extra local translation +
          yaw through a damped spring, strands streaming to that side) and gone by q_out1; word 2 invisible during the swing, forms
          form_frames after the landing (strands arriving from the turn direction, a spring settles it); words scaled `scale`, placed
          beside the runner, wght at the low end of its range. 'v014': q-retimed form-out / form-in, centred, full size, no extras."""

    def __init__(self, scene, flag, depth=None, matte_size=MATTE_SIZE, q_out0=Q_OUT0, q_in1=Q_IN1, mode="behind_subject",
                 style="organic", scale=0.75, q_out1=Q_OUT1, form_frames=FORM_FRAMES, light=True):
        pid, texts = parse_flag(flag)
        self.piece_path = piece_file(pid)
        typ = json.load(open(self.piece_path, encoding="utf-8"))["type"]
        self.sc = scene
        self.mode, self.style = mode, style
        self.v14 = style == "v014"
        self.W, self.H = scene.W, scene.H
        self.q_split = scene.out_angle / math.degrees(scene.Phi)
        self.q_out0, self.q_in1, self.q_out1 = q_out0, q_in1, q_out1
        self.form_frames = float(form_frames)
        self.scale = 1.0 if self.v14 else float(scale)
        self.t_land = float(scene.op.travel)
        self.rls = (scene.rl1, scene.rl2)
        behind = mode == "behind_subject"
        self.depth = depth if depth is not None else (0.5 if behind else 0.8)
        zts = []
        for which in (0, 1):
            lay = self.rls[which].L
            Z = lay.grids[0][2]
            has = lay.subj is not None and bool(lay.subj.any())
            zs = float(np.median(Z[lay.subj])) if has else float(np.median(Z))
            if behind:                                            # between the runner card and the background behind him
                from tools.fusion import depth_dolly as _dd
                zbk = _dd.disp_to_z(lay.bdisp, lay.params)
                zb = float(np.median(zbk[lay.subj])) if has else float(np.percentile(Z, 90))
                zb = max(zb, 1.25 * zs)
                zts.append(zs + self.depth * (zb - zs))
            else:
                zts.append(self.depth * zs)
        # direction of the turn on screen: the way image 1's plane travels while the camera orbits (-1 = to the left)
        probe = Plane(self.rls[0].L, zts[0], 0.0)

        def cen(q):
            Rc, pos = self._cam(q)
            return probe.project(Rc, pos, self.rls[0].Rk, self.rls[0].t, self.rls[0].s).mean(0)
        v = cen(q_out1) - cen(0.0)
        self.turn = -1.0 if v[0] < 0 else 1.0
        self.turn_vec = v / max(float(np.hypot(*v)), 1e-9)
        sides = {-1.0: ("left", "right"), 1.0: ("right", "left")}[self.turn]       # (exit side of word 1, entry side of word 2)
        self.words, self.planes, self.inks, self.report, self.centers = [], [], [], [], [None, None]
        for which, text in enumerate(texts):
            wd = find_word(typ, text)
            kw = {} if self.v14 else (dict(exit=sides[0]) if which == 0 else dict(entry=sides[1]))
            self.words.append(WordType(typ, wd, matte_size, light=(light and not self.v14), **kw))
        self.planes = [None, None]
        self.inks = [None, None]
        self._tex = {}
        self._reads = [self._read_frame(0), self._read_frame(1)]          # (colour BGR, runner matte 0..1) where each word is read
        for which in (0, 1):
            wt = self.words[which]
            lay = self.rls[which].L
            col, runner = self._reads[which]
            cap = float(wt.info.get("cap_px", 200.0)) * 1080.0 / 540.0
            cx, cy = self._place(which, wt, runner)
            self.centers[which] = (cx, cy)
            self.planes[which] = Plane(lay, zts[which], wt.yaw, pivot_x=None if self.v14 else cx)
            tex1 = self._tex_of(which, wt.matte(wt.info["hit"]))[..., None].repeat(4, -1)
            a_home = self._warp_at(which, 0.0 if which == 0 else 1.0, tex1, extras=False)[..., 3]
            self.inks[which] = choose_ink(col, a_home, wt.palette, cap, hide=runner if behind else None)
            ik = self.inks[which]
            hid = float(((runner > 0.5) & (a_home > 0.25)).sum() / max((a_home > 0.25).sum(), 1))
            self.report.append(dict(word=wt.text, image=which + 1, ink=ik["hex"], palette_pick=ik["name"], kind=ik["kind"], support=ik["support"],
                                    contrast_mean=round(ik["ratio_mean"], 2), share_over_target=round(ik["share_ok"], 3), target=ik["target"],
                                    plane_depth=round(zts[which], 3), yaw=wt.yaw, mode=mode, style=style, scale=self.scale,
                                    centre_px=[round(cx, 1), round(cy, 1)], hidden_by_runner=round(hid, 3),
                                    side=None if self.v14 else sides[which]))
        self._tex = {}
        self._springs()

    # --- read frame / placement
    def _read_frame(self, which):
        """(colour BGR, runner matte float 0..1) of the frame where the word is read: image 1 = its hold (clip frame 12), image 2 = after the
        word has formed (landing + form_frames + 12). No video: the first-frame layer."""
        lay = self.rls[which].L
        vid = getattr(self.sc, "video", None)
        if vid is None:
            return lay.col, (lay.subj.astype(np.float32) if lay.subj is not None else np.zeros((self.H, self.W), np.float32))
        frames, mattes = vid[which]
        if which == 0:
            f = 0 if self.v14 else 12
        else:
            from tools.fusion.depth_orbit import frame_list
            fl = frame_list(self.sc.op)
            tt = self.t_land + (0.0 if self.v14 else self.form_frames + 12.0)
            i = next((j for j, (t, _) in enumerate(fl) if t >= tt), len(fl) - 1)
            f = i - self.sc.image2_start()
        f = int(min(max(f, 0), len(frames) - 1))
        return frames[f], mattes[f]

    def _tex_of(self, which, matte, ss=SS):
        """matte -> placed alpha (ss*H, ss*W) for word `which`."""
        if self.v14 or self.centers[which] is None:
            return place_matte(matte, self.W, self.H, ss, self.scale)
        return place_matte(matte, self.W, self.H, ss, self.scale, self.centers[which], self.words[which].bbox())

    def _place(self, which, wt, runner):
        """word centre (home px): v014 = the frame centre; organic = on the side of the frame the runner leaves free, the whole word kept
        inside the frame, nudged away from him until at most ~30 % of its letters are behind him (behind_subject)."""
        W, H = self.W, self.H
        if self.v14:
            return W / 2.0, H / 2.0
        x0m, x1m, _, _ = wt.bbox()
        ww = (x1m - x0m) * W / wt.size[0] * self.scale
        cy = 0.46 * H
        if self.mode != "behind_subject":
            return place_beside(runner, ww, W)[0], cy

        def hidden(cx):                                                           # share of the letters behind the runner at this centre
            self.centers[which] = (cx, cy)
            a = cv2.resize(self._tex_of(which, wt.matte(wt.info["hit"])), (W, H), interpolation=cv2.INTER_AREA)
            return float(((runner > 0.5) & (a > 0.25)).sum() / max((a > 0.25).sum(), 1))
        cx = place_beside(runner, ww, W, hidden)[0]
        self.centers[which] = None
        return cx, cy

    # --- springs (secondary motion), on a grid of 1/8 frame over the whole move + holds
    def _springs(self):
        op = self.sc.op
        self._tg = np.arange(0.0, float(op.travel + op.hold2 + 4), 0.125)
        if self.v14:
            self._e = [np.zeros_like(self._tg), np.zeros_like(self._tg)]
            return
        qs = np.array([self.sc.q(t) for t in self._tg])
        c1 = _smoother((qs - self.q_out0) / max(self.q_out1 - self.q_out0, 1e-6))              # word 1: carried while image 1 turns away
        e1 = spring_response(c1, self._tg, omega=0.30, zeta=0.70)
        e2 = np.ones_like(self._tg)                                                            # word 2: arrives displaced, settles to 0
        m = self._tg >= self.t_land
        e2[m] = spring_response(np.zeros(int(m.sum())), self._tg[m], omega=0.26, zeta=0.72, x0=1.0, v0=-0.05)
        self._e = [e1, e2]

    def extras(self, which, t):
        """(dx in the image's home world units, dyaw deg) of the word's secondary motion at output time t."""
        if self.v14:
            return 0.0, 0.0
        x = float(np.interp(t, self._tg, self._e[which]))
        w = self.planes[which].width
        if which == 0:
            return self.turn * 0.9 * w * x, 20.0 * x                                           # on with the turn
        return -self.turn * 0.4 * w * x, -20.0 * x                                             # arrives from the turn side

    # --- poses
    def _cam(self, q):
        x, z, psi = self.sc.pose_phi(q * self.sc.Phi)
        return _psi_R(psi), np.array([x, 0.0, z])

    def _t(self, t):
        return float(getattr(self.sc, "t_cur", 0.0)) if t is None else float(t)

    def corners(self, which, q, t=None, extras=True):
        Rc, pos = self._cam(q)
        rl = self.rls[which]
        dx, dy = self.extras(which, self._t(t)) if extras else (0.0, 0.0)
        return self.planes[which].project(Rc, pos, rl.Rk, rl.t, rl.s, dx, dy)

    def _warp_at(self, which, q, tex, extras=True, t=None):
        d = self.corners(which, q, t, extras)
        return np.zeros((self.H, self.W, 4), np.float32) if d is None else self.planes[which].warp(tex, d)

    # --- timing
    def tau(self, which, q, t=None):
        """module-clock time of the word on image `which` (None = not on screen)."""
        i = self.words[which].info
        if which == 0:
            end_q = self.q_split if self.v14 else self.q_out1
            u = float(np.clip((q - self.q_out0) / max(end_q - self.q_out0, 1e-6), 0.0, 1.0))
            return i["hit"] if u <= 0 else i["release"] + u * (i["end"] - i["release"])
        if self.v14:
            u = float(np.clip((q - self.q_split) / max(self.q_in1 - self.q_split, 1e-6), 0.0, 1.0))
            return None if q < self.q_split else i["start"] + u * (i["hit"] - i["start"])
        tt = self._t(t)
        if tt < self.t_land - 1e-6:
            return None                                                                        # invisible during the swing
        u = float(np.clip((tt - self.t_land) / self.form_frames, 0.0, 1.0))
        return i["start"] + (1.0 - (1.0 - u) ** 3) * (i["hit"] - i["start"])                   # ease-out: no hard stop

    def texture(self, which, tau):
        k = (which, int(round(tau * 1000)))
        if k not in self._tex:
            if len(self._tex) > 8:
                self._tex.clear()
            m = self.words[which].matte(tau)
            self._tex[k] = None if m is None else premult_tex(self._tex_of(which, m), self.inks[which])
        return self._tex[k]

    def visibility(self, which, q):
        return float(self.sc.weights(q)[which])

    # --- one frame
    def sub_poses(self, which, q, t, blur_t):
        """[(q, t)] of the shutter sub-poses (just [(q, t)] when sharp)."""
        if blur_t is None:
            return [(q, t)]
        sh = self.sc.op.shutter
        ta, tb = blur_t - 0.5 * sh, blur_t + 0.5 * sh
        ca, cb = self.corners(which, self.sc.q(ta), ta), self.corners(which, self.sc.q(tb), tb)
        if ca is None or cb is None:
            return [(q, t)]
        n = int(np.clip(math.ceil(float(np.abs(cb - ca).max()) / STEP_PX), 1, MAX_SUB))
        return [(self.sc.q(blur_t + sh * ((i + 0.5) / n - 0.5)), blur_t + sh * ((i + 0.5) / n - 0.5)) for i in range(n)]

    def sub_qs(self, which, q, blur_t):
        """camera progress of the shutter sub-poses (v014 name, kept)."""
        return [qq for qq, _ in self.sub_poses(which, q, self._t(blur_t), blur_t)]

    def layer(self, which, q, blur_t=None, t=None, weight=None):
        """weight: multiply by the image's visibility (default: yes in 'front'; no in 'behind_subject', where the sandwich is weighted)."""
        if weight is None:
            weight = self.mode != "behind_subject"
        t = float(blur_t) if blur_t is not None else self._t(t)
        if weight:
            vis = self.visibility(which, q) if 0.0 < q < 1.0 else (1.0 if (q <= 0.0) == (which == 0) else 0.0)
        else:
            vis = 1.0
        if vis <= 0.0:
            return None
        acc = None
        subs = self.sub_poses(which, q, t, blur_t)
        for qq, tt in subs:
            tau = self.tau(which, qq if blur_t is not None else q, tt)
            if tau is None:
                continue
            tex = self.texture(which, tau)
            d = self.corners(which, qq, tt)
            if tex is None or d is None:
                continue
            w = self.planes[which].warp(tex, d)
            acc = w if acc is None else acc + w
        if acc is None:
            return None
        acc = acc / len(subs)                                       # sub-poses where the word is not (yet) there dim the frame
        return acc[..., :3] * vis, np.clip(acc[..., 3] * vis, 0.0, 1.0)

    def over(self, img, which, q, blur_t=None, t=None):
        """img float32 BGR 0..255 -> img with the word of image `which` over it (front mode)."""
        r = self.layer(which, q, blur_t, t)
        if r is None:
            return img
        pm, a = r
        return img * (1.0 - a[..., None]) + pm
