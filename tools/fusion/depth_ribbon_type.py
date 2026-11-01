"""depth_ribbon_type: v017 - the word unravels into long ribbons that stream ACROSS the depth video (counter to the video's own motion) through the
whole turn and re-form into the next word after the landing; the type sits BEHIND the runner by the transition's own matte pass.

User (v016 review, 4 sketches `17_depth-dolly/v017/review/user_sketch_1..4.png`): "I want this sort of movement - the text is countering the video
direction - the text is still behind the matte": (1) a big two-line word left of centre behind the runner, (2) as the tunnel breaks up its strokes
become flowing ribbons streaming left -> right across the frame, (3) the same ribbons keep flowing across the new video (one thread, no dead air),
(4) after image 2 comes to rest the ribbons gather into the next two-line word, behind the runner.

Existing pieces used (nothing rebuilt):
  * the `lanes` word module (mograph/modules/words/lanes.py) + the `lines` io style through tools/fusion/spec.build_scene (depth_type.WordType):
    lanes' own exit - "every arc detaches in a sweep and runs out along its own lane toward the exit side" - IS the ribbon unravel, and its entry -
    "lanes stream in from the entry side as a ribbon, peel off one by one and draw the outline" - IS the ribbons gathering into the word. The
    stream direction is set through the module's exit / entry SIDES: flow right = A exits 'right', B enters from 'left' (so B's ribbons also travel
    left -> right, one continuous thread across the cut). The P06 words (SEARCHING -> A LIGHT) in two lines each (SEARCH/ING, A/LIGHT), the piece's
    fonts / axes / line px; layout box 75 % of the frame width via the engine's own placement (`wf`, `dx`), so the lanes run to the world edges;
  * the video's dominant screen motion is MEASURED (`measure_flow`, Farneback on the depth video) and `--type-flow-dir counter` flows against it;
  * ink: depth_type.choose_ink (tools/fusion/legible_color.choose) measured per frame on the video behind the ribbons / word (runner excluded),
    smoothed in time; the 2D composite is compose.sandwich with the depth pass's runner matte (v016 `--matte-pass`), type between background and runner.

  python tools/fusion/depth_ribbon_type.py --depth-mp4 17_depth-dolly/v016/out/..._v016_depth.mp4 --matte-dir 17_depth-dolly/v016/local/matte_depth \
      --out 17_depth-dolly/v017 --type "P06:SEARCHING,A LIGHT" --type-flow-dir counter
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_flat_type as DF  # noqa: E402
from tools.fusion import depth_type as DT  # noqa: E402

OPPOSITE = {"left": "right", "right": "left"}


# ----------------------------------------------------------------------------------------------- direction
def measure_flow(frames, i0, i1):
    """dominant horizontal screen motion of the video over frames i0..i1 (px / frame, + = right): Farneback optical flow between consecutive
    frames, mean over the pixels that move (> 0.5 px) - what the viewer sees travel. -> dict(mean_dx, median_dx, moving_dx, per_frame)."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    per, mov = [], []
    for i in range(max(i0 - 1, 0), min(i1, len(g) - 1)):
        fl = cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, 0.5, 4, 25, 5, 7, 1.5, 0)
        dx = fl[..., 0]
        mag = np.hypot(fl[..., 0], fl[..., 1])
        per.append(float(dx.mean()))
        mov.append(float(dx[mag > 0.5].mean()) if (mag > 0.5).any() else 0.0)
    return dict(mean_dx=float(np.mean(per)), median_dx=float(np.median(per)), moving_dx=float(np.mean(mov)), per_frame=per)


def flow_side(mode, measured_dx=None):
    """--type-flow-dir -> the side the ribbons flow TOWARD ('left' | 'right'): right / left as given; counter = opposite of the measured video motion."""
    if mode in ("left", "right"):
        return mode
    if mode == "counter":
        if measured_dx is None:
            raise ValueError("counter needs the measured video motion")
        return "left" if measured_dx > 0 else "right"
    raise ValueError("--type-flow-dir: right | left | counter")


# ----------------------------------------------------------------------------------------------- the two words
def letter_visibility(alpha, runner, thr=0.25, join_px=2):
    """per-letter visible share of a formed word: the letters are the connected blobs of the (slightly dilated) word alpha; share = 1 - covered by the
    runner matte (the word is BEHIND him). -> list of (x0, y0, x1, y1, visible_fraction)."""
    m = (alpha > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join_px + 1, 2 * join_px + 1))
    n, lab = cv2.connectedComponents(cv2.dilate(m, k))
    out = []
    for j in range(1, n):
        px = (lab == j) & (m > 0)
        if px.sum() < 12:
            continue
        ys, xs = np.nonzero(px)
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), float(1.0 - (np.asarray(runner)[px] > 0.5).mean())))
    return out


class RibbonType:
    """Word A (formed on image 1's hold, unravels into ribbons from i_start) and word B. Two single-word type scenes (lanes, sides = the flow
    direction), one clock; A's exit and B's arrival overlap so the stream never stops.
    b_mode 'reverse' (v017 g03): B is lanes' EXIT toward the opposite side played BACKWARDS - long horizontal wavy lines arrive from the
    upstream edge, flowing the same way as A's, and converge into the letters, the fill solidifying last (ease-out, no snap); formed at b_end
    (landing + 18) and frozen there (A is frozen on its hold the same way). 'entry' (v017 g01/g02): lanes' own entry ribbon.
    dx_pref / auto-fit: the word's placement dx (fraction of the frame width, + = right) is searched around dx_pref so that every letter stays
    >= min_vis visible behind the runner (runners = the runner matte at the frame each word is read)."""

    def __init__(self, flag, n_frames, fps=24, i_start=24, i_land=59, flow="right", exit_frames=36, form_frames=21, b_lead=3, size=DT.MATTE_SIZE,
                 box=1.0, dx=-0.06, lines=(("SEARCH", "ING"), ("A", "LIGHT")), shutter=0.55, sub=3, b_mode="reverse", b_frames=33,
                 b_end=None, a_cut=0.78, b_skip=0.28, fade=8, runners=None, min_vis=0.6, dx_range=(-0.16, 0.10), left_of_centre=-0.02, dy_range=(-0.20, 0.16)):
        pid, texts = DT.parse_flag(flag)
        self.typ = json.load(open(DT.piece_file(pid), encoding="utf-8"))["type"]
        self.fps, self.n_frames, self.size, self.shutter, self.sub = int(fps), int(n_frames), tuple(size), float(shutter), int(sub)
        self.i_start, self.i_land, self.flow, self.b_mode = int(i_start), int(i_land), flow, b_mode
        self.a_cut, self.b_skip, self.fade = float(a_cut), float(b_skip), float(fade)      # g03: the late (wall / clump) end of the lanes' exit is never shown
        self.exit_s = exit_frames / float(fps)
        if b_mode == "reverse":
            self.b_end = int(b_end if b_end is not None else self.i_land + 18)
            self.b_frames = int(b_frames)
            self.i_b0 = self.b_end - self.b_frames                 # first ribbons of B at the upstream edge
            self.form_s = self.b_frames / float(fps)
        else:
            self.b_lead = int(b_lead)
            self.i_b0 = self.i_land - self.b_lead                  # B's entry starts here; it is formed at i_b0 + form_frames (= landing + 18)
            self.form_s = form_frames / float(fps)
            self.b_end = self.i_b0 + int(form_frames)
            self.b_frames = int(form_frames)
        self.b_lead = self.i_land - self.i_b0
        sgn = 1.0 if flow == "right" else -1.0
        ex, en = flow, OPPOSITE[flow]
        self.sgn, self.box, self.lines = sgn, box, lines
        self.fit, self.words, self.wt = [], [], []
        for k, text in enumerate(texts):
            dx_k = dx if k == 0 else dx - 0.04
            wt = self._word(k, text, dx_k, ex, en)
            if runners is not None:
                (bdx, bdy), rep = self._fit_dx(k, wt, dx_k, runners[k], dx_range, min_vis, left_of_centre, dy_range)
                if abs(bdx - dx_k) > 0.004 or abs(bdy) > 0.004:
                    wt = self._word(k, text, bdx, ex, en, bdy)
                rep["dx"], rep["dy"] = round(bdx, 3), round(bdy, 3)
                self.fit.append(rep)
            self.wt.append(wt)
            self.words.append(wt.info)
        self.a, self.b = self.wt

    def _word(self, k, text, dx_k, ex, en, dy_k=0.0):
        w = dict(DT.find_word(self.typ, text))
        w.pop("transition", None)
        pl = dict(w.get("placement") or {})
        pl.update(wf=self.box, dx=self.sgn * dx_k, dy=dy_k)
        w.update(lines=list(self.lines[k]), placement=pl)
        if k == 0:
            w.update(t0=0.0, form_s=0.5, exit_s=self.exit_s, t1=0.5 + 1.2 + self.exit_s, exit=ex)
        elif self.b_mode == "reverse":
            w.update(t0=0.0, form_s=0.5, exit_s=self.form_s, t1=0.5 + 4.0 + self.form_s, exit=en)     # exit toward the UPSTREAM side, played backwards
        else:
            w.update(t0=0.0, form_s=self.form_s, exit_s=0.4, t1=self.form_s + 4.0 + 0.4, entry=en)
        return DT.WordType(self.typ, w, self.size, light=False, home=True)      # home camera: dx / dy are real

    def _fit_dx(self, k, wt, dx0, runner, dx_range, min_vis, left_of_centre, dy_range):
        """pick the placement (dx, dy) - closest to (dx0, 0), left of centre - with every letter >= min_vis visible behind the runner and the word inside
        the frame; if nothing is left of centre, the closest anywhere; if nothing reaches min_vis, the best one (reported)."""
        H, W = np.asarray(runner).shape
        a0 = DT.place_matte(wt.matte(wt.info["release"]), W, H, 1)
        res = []
        for d in np.arange(dx_range[0], dx_range[1] + 1e-9, 0.02):
            for e in np.arange(dy_range[0], dy_range[1] + 1e-9, 0.02):
                M = np.float32([[1, 0, self.sgn * (d - dx0) * W], [0, 1, e * H]])
                a = cv2.warpAffine(a0, M, (W, H))
                ys, xs = np.nonzero(a > 0.25)
                if not len(xs) or xs.min() < 4 or xs.max() > W - 4 or ys.min() < 4 or ys.max() > H - 4:
                    continue
                lv = letter_visibility(a, runner)
                res.append((float(d), float(e), min(v for *_, v in lv) if lv else 1.0, lv))
        cost = lambda r: abs(r[0] - dx0) + abs(r[1])                                            # noqa: E731
        ok = [r for r in res if r[2] >= min_vis and self.sgn * r[0] <= left_of_centre + 1e-9]
        if ok:
            r, how = min(ok, key=cost), "left of centre, every letter >= %.0f%% visible" % (100 * min_vis)
        else:
            ok2 = [r for r in res if r[2] >= min_vis]
            if ok2:
                r, how = min(ok2, key=cost), "RIGHT of the left-of-centre limit (nothing left of centre keeps every letter >= %.0f%%)" % (100 * min_vis)
            else:
                r, how = max(res, key=lambda r: r[2]), "no placement reaches %.0f%%: the best" % (100 * min_vis)
        return (r[0], r[1]), dict(word=wt.text, min_letter_visible=round(r[2], 3), letters=[round(v, 2) for *_, v in r[3]], how=how)

    # --- clock
    def tau_a(self, i):
        ia = self.a.info
        if i <= self.i_start:
            return ia["release"] if self.b_mode == "reverse" else ia["hit"]                 # g03: frozen on the same state the exit starts from
        return ia["release"] + (float(i) - self.i_start) / self.fps

    def b_progress(self, i):
        """0..1 convergence of B's ribbons into the word: smootherstep (the strands drift in, gather in the middle, the word settles - no snap)."""
        u = float(np.clip((i - self.i_b0) / float(self.b_frames), 0.0, 1.0))
        return float(DT._smoother(u))

    def tau_b(self, i):
        ib = self.b.info
        if self.b_mode == "reverse":
            if i < self.i_b0:
                return None
            e = self.b_progress(i)
            t0 = ib["release"] + (1.0 - self.b_skip) * (ib["end"] - ib["release"])      # start where the strands are already long horizontal lines
            return t0 - e * (t0 - ib["release"])                           # backwards through the exit; frozen (release) once formed
        return ib["start"] + (float(i) - self.i_b0) / self.fps if i >= self.i_b0 - 1e-6 else None

    def gain(self, which, i):
        """g03 cross-dissolve of the two streams: A fades out over the last `fade` frames of its shown exit (a_cut of the window), B fades in over its
        first `fade` frames, so the thread has no hard start / end (no wall, no clump) and both overlap."""
        if self.b_mode != "reverse":
            return 1.0
        if which == 0:
            i_end = self.i_start + self.a_cut * self.exit_s * self.fps
            return float(np.clip((i_end - i) / self.fade, 0.0, 1.0)) if i > self.i_start else 1.0
        return float(np.clip((i - self.i_b0 + 1) / self.fade, 0.0, 1.0))

    def matte_of(self, which, i):
        """float32 (h, w) 0..1 of word `which` (0 / 1) at frame i, averaged over the shutter (`sub` sub-frames while it moves)."""
        return self._matte_raw(which, i) * self.gain(which, i)

    def _matte_raw(self, which, i):
        wt = self.wt[which]
        tau = self.tau_a if which == 0 else self.tau_b
        t0 = tau(i)
        if t0 is None:
            return np.zeros(wt.size[::-1], np.float32)
        moving = (which == 0 and i > self.i_start) or (which == 1 and i < self.b_end + 1)
        n = self.sub if moving else 1
        acc, cnt = None, 0
        for k in range(n):
            dt = self.shutter / self.fps * ((k + 0.5) / n - 0.5) if n > 1 else 0.0
            if which == 1 and self.b_mode == "reverse":
                dt = -dt                                                    # the clock runs backwards
            t = (t0 + dt) if which else (t0 if i <= self.i_start else t0 + dt)
            m = wt.matte(t)
            if m is None:
                m = np.zeros(wt.size[::-1], np.float32)
            acc = m if acc is None else acc + m
            cnt += 1
        return acc / cnt

    def matte(self, i, wmax=432):
        return np.maximum(self.matte_of(0, i), self.matte_of(1, i))


# ----------------------------------------------------------------------------------------------- wave + per-frame ink
def squash_alpha(a2, cy, s):
    """vertical squeeze of the supersampled alpha about the row `cy` (factor s <= 1): keeps the ribbons a flat horizontal band while they flow."""
    if s >= 0.999:
        return a2
    h, w = a2.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    return cv2.remap(a2, np.repeat(xs, h, 0), np.repeat(((ys - cy) / s + cy).astype(np.float32), w, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def wave_alpha(a2, amp_px, wavelength_px, phase, ss=2):
    """flowing waves on the ribbons: y' = y + amp sin(2 pi x / wavelength + phase) on the supersampled alpha (a flat 2D displacement)."""
    if amp_px <= 1e-3:
        return a2
    h, w = a2.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    dy = np.float32(amp_px * ss) * np.sin(2.0 * math.pi * xs / (wavelength_px * ss) + phase)
    return cv2.remap(a2, np.repeat(xs, h, 0), (ys + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class InkTrack:
    """legible_color's choice per frame on the video behind the type (runner excluded), smoothed: the colour eases (EMA), the support (none <
    outline < shadow < scrim) is held for `hold` frames so it never pops."""
    RANK = {None: 0, "outline": 1, "shadow": 2, "scrim": 3}

    def __init__(self, palette, cap_norm=1000.0, k=0.35, hold=10):
        self.palette, self.cap, self.k, self.hold = palette, cap_norm, k, hold
        self.ink, self.sup_hist = None, []
        self.log = []

    def update(self, col, alpha, runner):
        if (alpha > 0.25).sum() >= 24:
            ch = DT.choose_ink(col, alpha, self.palette, self.cap, hide=runner)
            if self.ink is None:
                self.ink = dict(ch)
            else:
                self.ink["bgr"] = (1.0 - self.k) * self.ink["bgr"] + self.k * ch["bgr"]
                self.ink["ratio_mean"], self.ink["share_ok"], self.ink["name"] = ch["ratio_mean"], ch["share_ok"], ch["name"]
            self.sup_hist.append(ch["support"])
        elif self.ink is None:
            self.ink = dict(DT.choose_ink(col, np.ones(alpha.shape, np.float32) * 0.0 + 0.5, self.palette, self.cap, hide=runner))
            self.sup_hist.append(None)
        else:
            self.sup_hist.append(self.sup_hist[-1] if self.sup_hist else None)
        recent = self.sup_hist[-self.hold:]
        sup = max(recent, key=lambda s: self.RANK[s])
        self.ink["support"] = sup
        y = float(DT.LC.rel_lum(DT.LC.srgb_to_lin(np.clip(self.ink["bgr"][::-1] / 255.0, 0, 1))))
        self.ink["y_ink"] = y
        self.ink["hex"] = DT.LC.hex_of(self.ink["bgr"][::-1] / 255.0)
        self.log.append((self.ink["hex"], sup, round(float(self.ink["ratio_mean"]), 2)))
        return self.ink


class RibbonComposer:
    """ribbons / words (placed by the engine, left of centre) + per-frame ink -> compose.sandwich with the runner matte."""

    def __init__(self, rt: RibbonType, frames, mattes, wave_px=14.0, wavelength=210.0, wave_speed=0.045, squash=0.4):
        self.rt, self.frames, self.mattes = rt, frames, mattes
        H, W = frames[0].shape[:2]
        self.W, self.H = W, H
        self.wave_px, self.wavelength, self.wave_speed, self.squash = wave_px, wavelength, wave_speed, squash
        self.cy2 = []
        for wt in rt.wt:                                              # row (2x grid) of each formed word's centre: the squeeze pivots there
            f = DT.place_matte(wt.matte(wt.info["release"]), W, H, 2)
            ys = np.nonzero(f.max(1) > 0.25)[0]
            self.cy2.append(0.5 * (ys.min() + ys.max()))
        cap = float(rt.words[0].get("cap_px", 100.0)) * 1080.0 / 540.0
        self.track = InkTrack(rt.typ.get("palette") or {}, cap)
        self.k = W / rt.size[0]
        self.cy = (H - rt.size[1] * self.k) / 2.0
        self._ink = {}

    def alpha(self, i, ss=2):
        """placed (centred scale to the frame width) flat alpha of the ribbons / words at frame i on the ss-times grid; each word's ribbons are squeezed
        to a flatter band and waved while it flows (amplitude 0 on the formed word)."""
        rt = self.rt
        sgn = 1.0 if rt.flow == "right" else -1.0
        out = None
        for w in (0, 1):
            m = rt.matte_of(w, i)
            if not m.any():
                continue
            a2 = DT.place_matte(m, self.W, self.H, ss)
            env = self.wave_env(i, w)
            if env > 0:
                a2 = squash_alpha(a2, self.cy2[w], 1.0 - self.squash * env)
                a2 = wave_alpha(a2, self.wave_px * env, self.wavelength, -sgn * 2.0 * math.pi * self.wave_speed * i, ss)
            out = a2 if out is None else np.maximum(out, a2)
        return out if out is not None else np.zeros((self.H * ss, self.W * ss), np.float32)

    def wave_env(self, i, which=None):
        """0..1: how much of word `which` is ribbon at frame i (A while it unravels, B while it gathers): waves / squeeze live on the ribbons, not on
        the formed word. which None = the larger of the two."""
        rt = self.rt
        n_a = rt.a_cut * rt.exit_s * rt.fps if rt.b_mode == "reverse" else rt.exit_s * rt.fps
        e_a = float(np.clip((i - rt.i_start) / 8.0, 0, 1)) * float(np.clip((rt.i_start + n_a + 6 - i) / 10.0, 0, 1))
        if rt.b_mode == "reverse":                                        # follows the convergence: ribbon-like while the strands are far, none once they have gathered
            u = rt.b_progress(i)
            e_b = float(np.clip((i - rt.i_b0 + 2) / 6.0, 0, 1)) * (1.0 - float(DT._smoother((u - 0.1) / 0.7)))
        else:
            e_b = float(np.clip((i - rt.i_b0 + 2) / 6.0, 0, 1)) * float(np.clip((rt.i_b0 + rt.form_s * rt.fps - i) / 8.0, 0, 1))
        return max(e_a, e_b) if which is None else (e_a if which == 0 else e_b)

    def type_layer(self, i):
        a2 = self.alpha(i)
        a_home = cv2.resize(a2, (self.W, self.H), interpolation=cv2.INTER_AREA)
        ink = self.track.update(self.frames[i], a_home, self.mattes[i])
        pm = cv2.resize(DT.premult_tex(a2, ink), (self.W, self.H), interpolation=cv2.INTER_AREA)
        return pm[..., :3], np.clip(pm[..., 3], 0.0, 1.0)

    def frame(self, i):
        pm, a = self.type_layer(i)
        foot = self.frames[i][..., ::-1].astype(np.float32) / 255.0
        rgb = (pm[..., ::-1] / 255.0) / np.maximum(a, 1e-6)[..., None]
        out = C.sandwich(foot, rgb.astype(np.float32), a, self.mattes[i].astype(np.float32), matte_threshold=None)
        return np.clip(np.asarray(out)[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-mp4", required=True)
    ap.add_argument("--matte-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="P06:SEARCHING,A LIGHT")
    ap.add_argument("--type-flow-dir", default="right", choices=["right", "left", "counter"],
                    help="side the ribbons flow toward; counter = opposite of the video's measured dominant motion during the turn")
    ap.add_argument("--i-start", type=int, default=24, help="video frame the turn starts (word A starts to unravel)")
    ap.add_argument("--i-land", type=int, default=59, help="video frame image 2 has landed (word B starts to gather)")
    ap.add_argument("--exit-frames", type=int, default=36)
    ap.add_argument("--form-frames", type=int, default=21, help="B entry length; B is formed at (landing - b-lead) + form-frames")
    ap.add_argument("--b-lead", type=int, default=3, help="B's first ribbons arrive this many frames before the landing (keeps the thread unbroken)")
    ap.add_argument("--box", type=float, default=1.0, help="word box width as a share of the frame (engine placement wf; 1.0 = words ~75-80 %% wide)")
    ap.add_argument("--b-mode", default="reverse", choices=["reverse", "entry"], help="reverse = word B is lanes' exit toward the upstream side played backwards (g03); entry = lanes' own entry (g01/g02)")
    ap.add_argument("--b-frames", type=int, default=33, help="reverse: frames the ribbons take to converge into word B")
    ap.add_argument("--b-end", type=int, default=None, help="frame word B is formed (default landing + 18)")
    ap.add_argument("--a-cut", type=float, default=0.78, help="reverse mode: share of A's exit window shown (the late end is a hard wall / clump)")
    ap.add_argument("--b-skip", type=float, default=0.28, help="reverse mode: share of B's reversed exit skipped at its far end")
    ap.add_argument("--fade", type=float, default=8.0, help="reverse mode: frames A fades out / B fades in (the two streams overlap)")
    ap.add_argument("--min-vis", type=float, default=0.6, help="every letter must keep this visible share behind the runner (dx is searched)")
    ap.add_argument("--no-fit", action="store_true", help="no dx search (keep the placement dx)")
    ap.add_argument("--squash", type=float, default=0.4, help="how much the flowing ribbons are squeezed vertically (0 = not at all)")
    ap.add_argument("--wave", type=float, default=14.0, help="ribbon wave amplitude in video px (0 = straight lanes)")
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args(argv)
    outd, review, local = dd.version_dirs(a.out)
    frames = DF.read_frames(a.depth_mp4)
    mattes = DF.read_mattes(a.matte_dir, len(frames))
    t0 = time.time()
    meas = measure_flow(frames, a.i_start, a.i_land)
    flow = flow_side(a.type_flow_dir, meas["mean_dx"])
    print("video motion over frames %d-%d: mean dx %.2f px/frame (median %.2f, moving pixels %.2f) -> %s; ribbons flow %s%s" % (
        a.i_start, a.i_land, meas["mean_dx"], meas["median_dx"], meas["moving_dx"], "LEFT" if meas["mean_dx"] < 0 else "RIGHT", flow.upper(),
        " (counters the video)" if (flow == "right") == (meas["mean_dx"] < 0) else " (WITH the video)"), flush=True)
    b_end = a.b_end if a.b_end is not None else a.i_land + 18
    runners = None if a.no_fit else (mattes[min(12, len(mattes) - 1)], mattes[min(b_end + 12, len(mattes) - 1)])
    rt = RibbonType(a.type, len(frames), a.fps, a.i_start, a.i_land, flow, a.exit_frames, a.form_frames, a.b_lead, box=a.box, b_mode=a.b_mode,
                    b_frames=a.b_frames, b_end=a.b_end, a_cut=a.a_cut, b_skip=a.b_skip, fade=a.fade, runners=runners, min_vis=a.min_vis)
    for r in rt.fit:
        print('placement:', r, flush=True)
    comp = RibbonComposer(rt, frames, mattes, wave_px=a.wave, squash=a.squash)
    print("type scenes %.1fs | A %s hit %.2f exit %.2f-%.2f | B %s start %.2f hit %.2f" % (
        time.time() - t0, rt.words[0]["text"], rt.words[0]["hit"], rt.words[0]["release"], rt.words[0]["end"], rt.words[1]["text"],
        rt.words[1]["start"], rt.words[1]["hit"]), flush=True)
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), a.tag)))
    proc = dd._ffmpeg_writer(mp4, comp.W, comp.H, a.fps)
    keep, mass = {}, []
    sheet_i = sorted({0, a.i_start, a.i_start + 6, a.i_start + 12, a.i_start + 18, 48, 52, 56, 60, 64, 68, 72, rt.b_end, 90, len(frames) - 1})
    t1 = time.time()
    for i in range(len(frames)):
        fr = comp.frame(i)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        mass.append(float(rt.matte(i).mean()))
        if i in sheet_i:
            keep[i] = fr.copy()
        if i % 6 == 0:
            print("  frame %d/%d  %.0fs" % (i, len(frames), time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    import re
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    gen = ("_" + gm.group(1)) if gm else ""
    tiles = [(keep[i], "f%d" % i) for i in sorted(keep)]
    sheet = dd.unique_path(os.path.join(review, "ribbon_type_sheet%s.png" % gen))
    dd._sheet({"ribbons behind the runner matte": tiles}, sheet, tw=140, cols=len(tiles))
    json.dump(dict(flag=a.type, flow=flow, flow_dir_arg=a.type_flow_dir, video_motion=dict({k: v for k, v in meas.items() if k != "per_frame"}, per_frame=[round(x, 2) for x in meas["per_frame"]]),
                   i_start=a.i_start, i_land=a.i_land, exit_frames=a.exit_frames, form_frames=a.form_frames, b_lead=a.b_lead, b_mode=a.b_mode, b_frames=a.b_frames,
                   b_end=rt.b_end, i_b0=rt.i_b0, placement_fit=rt.fit, box=a.box, wave_px=a.wave,
                   words=[{k: w[k] for k in ("text", "start", "hit", "release", "end")} for w in rt.words], type_mass=[round(m, 5) for m in mass],
                   ink_log=comp.track.log, depth_mp4=a.depth_mp4, matte_dir=a.matte_dir),
              open(os.path.join(local, "ribbon_type%s.json" % gen), "w"), indent=1)
    print("ribbon type -> %s (%.0fs) sheet %s" % (mp4, time.time() - t0, sheet), flush=True)


if __name__ == "__main__":
    main()
