"""depth_slide_type: v020 - ONE eased move: SEARCHING slides off LEFT, horizontal lines travel LEFT -> RIGHT across the screen, A LIGHT slides in from the RIGHT and
forms from those lines; all on the depth camera's own ease q(t), flat 2D, behind the runner by the through-the-breakup matte.

User (v019 review): "can we just do a single transition that does the same ease in and ease out as the camera motion but we see the type moving off screen left and
then the transition lines horizontally travelling but we can see the ends of them as they seem to move across screen and then the next words slide in from the right and
form from the lines that we watched travelling from left to right".

Pieces (all existing; the pan geometry is applied directly because the engine's pan clock cannot be driven by q):
  * the P06 words through the type engine (tools/fusion/spec.build_scene via depth_type.WordType, cam3d HOME camera): SEARCHING (Doto, SEA/RCH/ING) and A LIGHT
    (GeistPixel, A/LIG/HT) in v018's layout (box 0.76, A right of centre, B left of centre, nudged so no letter is > 40 % behind the runner);
  * PAN geometry (mograph/modules/camera/pan_dir.py: pure translation, "right" = the camera travels right = the PICTURE SLIDES LEFT: A exits left, B enters from the
    right; sign checked against v016, whose `left` moved the words right): the words are world-locked on one line, the camera progress is the depth pivot's own
    quintic q(t) (depth_orbit.q_at over the 36-frame travel), so a word's screen position is exactly linear in q: A at x0 - q XA, B at x1 + (1 - q) XB;
  * the LINES are word B's own `lanes` ENTRY strands (io style `lines`, entry side LEFT: the strands travel left -> right and peel into the glyph outline), straight
    (fan 0, no wave), n_strands moderate, tail_len = a fraction of the frame so each strand is a segment with both ends visible; rendered with the static home camera
    (screen space, they do not slide with the words) and with the fill layer switched off (the solid letters are the sliding word B, which becomes opaque as q -> 1 and
    meets the outline the lines have drawn at the same place);
  * composite as v016: legible ink per frame (depth_ribbon_type.InkTrack) + compose.sandwich with the v016 matte pass (type between the video and the runner).

  python tools/fusion/depth_slide_type.py --depth-mp4 17_depth-dolly/v016/out/..._v016_depth.mp4 --matte-dir 17_depth-dolly/v016/local/matte_depth --out 17_depth-dolly/v020
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

from tools.fusion import depth_dolly as dd  # noqa: E402
from tools.fusion import depth_flat_type as DF  # noqa: E402
from tools.fusion import depth_orbit as do  # noqa: E402
from tools.fusion import depth_ribbon_type as RB  # noqa: E402
from tools.fusion import depth_type as DT  # noqa: E402


def _smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


class SlideType:
    """Words A and B (formed, world-locked) + the strand lines. `parts(i, ss)` -> the three alpha layers (supersampled video grid) at frame i."""

    def __init__(self, flag, n_frames, fps=24, i_start=24, travel=36, size=DT.MATTE_SIZE, wf=0.76, dx_a=0.07, dx_b=-0.11, dy_b=-0.18,
                 xa=830.0, xb=850.0, lines_start=28, lines_form=1.4, n_lines=30, tail=0.34, speed=520.0, width=3.2, gate=(0.5, 1.0), shutter=0.55, sub=4, video_w=432,
                 lines="circuit", b_font="Tourney[wdth,wght].ttf", b_axes=None, b_hf=0.72, centre=True, k_strands=24, b_k=0.85,
                 a_font=None, a_axes=None, a_wf=None, b_wf=None, a_pos=None, b_pos=None, a_lines=None, b_lines=None, a_exit="left", b_entry="right"):
        # model_design flags (20_open-model-design): all default to None / g04 behaviour, so a run without them is g04.
        # DEFAULTS = the user's pick, v020 g04 (circuit_board strands, centred words, A LIGHT in Tourney). g01 (lanes strands) = lines='lanes', centre=False, b_font=None, b_hf=None,
        # b_k=1, dy_b=0, gate=(0.6, 1.0); g02 (scanlines) = the same with lines='scan'.
        if b_font and b_axes is None:
            b_axes = {"wght": 400.0, "wdth": 100.0}
        pid, texts = DT.parse_flag(flag)
        self.typ = json.load(open(DT.piece_file(pid), encoding="utf-8"))["type"]
        self.fps, self.n_frames, self.size, self.i_start, self.shutter, self.sub = int(fps), int(n_frames), tuple(size), int(i_start), float(shutter), int(sub)
        self.xa, self.xb, self.gate = float(xa), float(xb), tuple(gate)
        self.a_exit, self.b_entry = a_exit, b_entry
        self.lines_start, self.lines_form = int(lines_start), float(lines_form)
        self.lines_mode = lines
        self.wl = None
        self.op = do.OrbitParams(mode="pivot", travel=int(travel), hold1=int(i_start), hold2=max(self.n_frames - i_start - travel, 1))
        self.fl = do.frame_list(self.op)
        self.i_land = i_start + int(travel) - 1                          # the frame whose t == travel (q = 1)
        self.vw = float(video_w)
        words = []
        for k, text in enumerate(texts):
            w = dict(DT.find_word(self.typ, text))
            w.pop("transition", None)
            pl = dict(w.get("placement") or {})
            pl.update(yaw=0.0, wf=wf * (b_k if k == 1 else 1.0), dx=(0.0 if centre else (dx_a if k == 0 else dx_b)), dy=0.0 if k == 0 else dy_b)
            if k == 1 and b_hf:
                pl.update(hf=float(b_hf) * b_k)
            if k == 1 and b_font:
                w.update(font=b_font, axes=dict(b_axes or {}))              # g03: word B in another liked font (Tourney, its double-line outline)
            if k == 0 and a_font:
                w.update(font=a_font, axes=dict(a_axes or {}))
            if (a_wf if k == 0 else b_wf):
                pl.update(wf=float(a_wf if k == 0 else b_wf))
            if (a_pos if k == 0 else b_pos) is not None:
                pl.update(dx=float((a_pos if k == 0 else b_pos)[0]), dy=float((a_pos if k == 0 else b_pos)[1]))
            if a_lines if k == 0 else b_lines:
                w.update(lines=list(a_lines if k == 0 else b_lines))
            w.update(placement=pl, t0=0.0, form_s=0.5, exit_s=0.3, t1=0.5 + 6.0 + 0.3)
            words.append(w)
        self.wa = DT.WordType(self.typ, words[0], self.size, home=True)
        self.wb = DT.WordType(self.typ, words[1], self.size, home=True)
        if lines == "lanes":                                              # g01: the lanes module's own entry strands (thinned)
            lw = dict(words[1])
            lw.update(form_s=self.lines_form, t1=self.lines_form + 6.0 + 0.3, entry="left",
                      params=dict(n_strands=int(n_lines), tail_len=float(tail) * 540.0, fan=0.0, speed=float(speed), width=float(width)))     # straight, parallel rows; segments shorter than the frame
            self.wl = DT.WordType(self.typ, lw, self.size, home=True)
            for m in (it for it, _a, _b in self.wl.sc.seq.timeline if hasattr(it, "placement")):
                m.fill_layers = lambda t: []                                  # the lines only: the solid letters are the sliding word B
                # a dotted face has one strand per dot contour (hundreds of strands, a solid-looking sheet): keep every `step`-th strand by its lateral rank, so the lines
                # are a moderate number of distinct, evenly spaced segments (the module's own strands, just fewer of them)
                step = max(1, int(round(m.N / float(n_lines))))
                keep = (np.asarray(m._in["rank"]) % step) == 0
                self.n_strands_all, self.n_strands_shown = int(m.N), int(keep.sum())
                m.windows = (lambda orig, kp: (lambda t: [w if kp[j] else None for j, w in enumerate(orig(t))]))(m.windows, keep)
        self.cs = None
        if lines == "circuit":
            self.cs = CircuitScene(self.typ, words, self.size, self.fps, k_strands, exit_side=a_exit, entry_side=b_entry)
            self.cs.fit_clock()
        self.words = [self.wa.info, self.wb.info] + ([self.wl.info] if self.wl is not None else [])
        self.k = self.vw / self.size[0]
        self._cache = {}

    # --- the camera's ease
    def t_of(self, i):
        return self.fl[min(max(int(i), 0), len(self.fl) - 1)][0]

    def q_t(self, t):
        return float(do.q_at(float(t), self.op, 1.0, 1.0))

    def q(self, i):
        return self.q_t(self.t_of(i))

    def gate_of(self, q):
        return _smooth((q - self.gate[0]) / max(self.gate[1] - self.gate[0], 1e-6))

    # --- the layers
    def _hold(self, wt):
        k = ("hold", id(wt))
        if k not in self._cache:
            self._cache[k] = wt.matte(wt.info["release"])
        return self._cache[k]

    def lines_matte(self, tau):
        """strand-only matte of word B's lanes entry at module time tau (None before it starts)."""
        if tau is None or tau <= 0.0:
            return None
        return self.wl.matte(min(tau, self.wl.info["release"]))

    def tau_lines(self, t_frame):
        """module time of the lines at (fractional) frame time t_frame."""
        return (float(t_frame) - self.lines_start) / self.fps

    def shifts(self, q):
        """video-px x shifts of word A (leaves left) and word B (arrives from the right) at camera progress q."""
        return -q * self.xa, (1.0 - q) * self.xb

    _DIR = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}

    def shifts_xy(self, q):
        """((x, y) of word A, (x, y) of word B) in video px: A leaves toward a_exit, B arrives from the b_entry side. The vertical travel is scaled by the frame aspect."""
        r = self.size[1] / float(self.size[0])
        ua, ub = self._DIR[self.a_exit], self._DIR[self.b_entry]
        return (q * self.xa * ua[0], q * self.xa * r * ua[1]), ((1.0 - q) * self.xb * ub[0], (1.0 - q) * self.xb * r * ub[1])

    def sub_times(self, i):
        """[(frame time, q)] of the shutter sub-samples of frame i (frame i spans the clock of fl[i]); one when nothing moves."""
        t = self.t_of(i)
        moving = (self.i_start - 1 <= i <= self.i_land + 8) or (self.lines_start <= i <= self.lines_start + 40)
        n = self.sub if moving else 1
        out = []
        for k in range(n):
            d = self.shutter * ((k + 0.5) / n - 0.5) if n > 1 else 0.0
            out.append((i + d, self.q_t(t + d)))
        return out


class SlideComposer(RB.RibbonComposer):
    """RibbonComposer's per-frame ink + sandwich; the alpha is the union of the sliding words and the lines."""

    def __init__(self, st: SlideType, frames, mattes):
        self.rt, self.frames, self.mattes = st, frames, mattes
        H, W = frames[0].shape[:2]
        self.W, self.H = W, H
        cap = float(st.words[0].get("cap_px", 200.0)) * 1080.0 / 540.0
        self.track = RB.InkTrack(st.typ.get("palette") or {}, cap)
        self._base = {}

    def _placed(self, key, m, ss):
        if (key, ss) not in self._base:
            self._base[(key, ss)] = DT.place_matte(m, self.W, self.H, ss)
        return self._base[(key, ss)]

    @staticmethod
    def _shift(a, dx, dy=0.0):
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return a
        return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)

    def parts(self, i, ss=2):
        """(A, B, lines) float alpha layers on the ss-times video grid, shutter-averaged."""
        st = self.rt
        a0 = self._placed("A", st._hold(st.wa), ss)
        b0 = self._placed("B", st._hold(st.wb), ss)
        A = np.zeros_like(a0)
        B = np.zeros_like(b0)
        L = np.zeros_like(a0)
        subs = st.sub_times(i)
        for tf, q in subs:
            sa, sb = st.shifts(q)
            if i >= st.i_start - 1 and q > 0.0:
                A += self._shift(a0, sa * ss)
            else:
                A += a0
            g = st.gate_of(q)
            if g > 0.0:
                B += g * self._shift(b0, sb * ss)
            m = st.lines_matte(st.tau_lines(tf))
            if m is not None:
                L += DT.place_matte(m, self.W, self.H, ss)
        n = float(len(subs))
        return A / n, B / n, L / n

    def alpha(self, i, ss=2):
        A, B, L = self.parts(i, ss)
        return np.clip(np.maximum(np.maximum(A, B), L), 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- g02: scanline lines
def _runs(row):
    """(x0, x1) of every run of True in a 1-D bool array."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


class RowSet:
    """The scanline representation of a word mask on the video grid: N evenly spaced rows across `y_range`; for each row the overall x extent [xl, xr] of the
    (horizontally closed, so dotted strokes merge) mask in a thin band round it, plus its runs (the row spans of the strokes). The lines of v020 g02 are these rows:
    a segment per row, and a word is formed by thickening each row's band over the mask."""

    def __init__(self, mask, ys, close_px=11, band=3.0):
        m = (mask > 0.3).astype(np.uint8)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, close_px), np.uint8)) > 0
        self.ys = np.asarray(ys, float)
        self.ext, self.runs = [], []
        for y in self.ys:
            lo, hi = int(np.floor(y - 0.5 * band)), int(np.ceil(y + 0.5 * band))
            bnd = closed[max(lo, 0):hi + 1].any(0)          # the whole band the row owns (a dotted face has gaps between its dot rows)
            xs = np.nonzero(bnd)[0]
            self.ext.append((float(xs.min()), float(xs.max())) if len(xs) else None)
            self.runs.append(_runs(bnd) if len(xs) else [])


def _seg(canvas, y, x0, x1, thick, ss, gain=1.0):
    """anti-aliased horizontal bar y +- thick/2, x0..x1 (video px) onto the ss-times canvas (max-composited)."""
    h, w = canvas.shape
    if x1 <= x0 or gain <= 0:
        return
    Y0, Y1 = (y - 0.5 * thick) * ss, (y + 0.5 * thick) * ss
    X0, X1 = x0 * ss, x1 * ss
    iy0, iy1 = max(int(np.floor(Y0)), 0), min(int(np.ceil(Y1)), h)
    ix0, ix1 = max(int(np.floor(X0)), 0), min(int(np.ceil(X1)), w)
    if iy1 <= iy0 or ix1 <= ix0:
        return
    ys = np.arange(iy0, iy1)
    xs = np.arange(ix0, ix1)
    cy = np.clip(np.minimum(Y1, ys + 1.0) - np.maximum(Y0, ys.astype(float)), 0, 1)
    cx = np.clip(np.minimum(X1, xs + 1.0) - np.maximum(X0, xs.astype(float)), 0, 1)
    cv = (cy[:, None] * cx[None, :] * gain).astype(np.float32)
    np.maximum(canvas[iy0:iy1, ix0:ix1], cv, out=canvas[iy0:iy1, ix0:ix1])


class ScanComposer(SlideComposer):
    """v020 g02: the words' own scanline rows are the lines. A: solid word slides left on q while its rows detach and travel RIGHT as straight segments (stretching
    to `seg` px, staggered per row, head x = row end + q * travel). B: segments enter from the LEFT, head chases B's run end - which itself comes in from the RIGHT on q
    (end x = row end + (1 - q) xb) - and the tail contracts onto the row extent, so each segment lands exactly on its row's x-extent; then the row's band thickens over
    B's real dotted mask until the glyph is complete: B shows only where its rows have landed. The A and B rows are the same grid, so the stream is one thread."""

    def __init__(self, st: SlideType, frames, mattes, rows=36, seg=0.34, thick=5.0, b_start=28, b_stag=8.0, b_dur=20.0, rev_dur=8.0, a_stag=0.06):
        super().__init__(st, frames, mattes)
        W, H = self.W, self.H
        self.rows, self.seg, self.thick = int(rows), float(seg) * W, float(thick)
        self.b_start, self.b_stag, self.b_dur, self.rev_dur, self.a_stag = float(b_start), float(b_stag), float(b_dur), float(rev_dur), float(a_stag)
        self.a_mask = DT.place_matte(st._hold(st.wa), W, H, 1)
        self.b_mask = DT.place_matte(st._hold(st.wb), W, H, 1)
        ya = np.nonzero(self.a_mask.max(1) > 0.3)[0]
        yb = np.nonzero(self.b_mask.max(1) > 0.3)[0]
        y0, y1 = float(min(ya.min(), yb.min())), float(max(ya.max(), yb.max()))
        self.pitch = (y1 - y0) / self.rows
        self.ys = y0 + (np.arange(self.rows) + 0.5) * self.pitch
        self.A, self.B = RowSet(self.a_mask, self.ys, band=self.pitch), RowSet(self.b_mask, self.ys, band=self.pitch)
        self.a_ss, self.b_ss = DT.place_matte(st._hold(st.wa), W, H, 2), DT.place_matte(st._hold(st.wb), W, H, 2)

    # --- per-row clocks
    def b_row(self, r):
        """(start frame, landing frame) of row r's incoming segment: rows land staggered over `b_stag` frames, top to bottom."""
        s0 = self.b_start + self.b_stag * r / max(self.rows - 1, 1)
        return s0, s0 + self.b_dur

    @staticmethod
    def _ease(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * x * (x * (x * 6 - 15) + 10)

    def a_head_tail(self, r, q):
        """head / tail x (video px) of row r's A segment at camera progress q: starts as the row's extent, stretches to `seg`, travels right with q."""
        xl, xr = self.A.ext[r]
        d = r / max(self.rows - 1, 1) * self.a_stag
        u = float(np.clip((q - d) / max(1.0 - d, 1e-6), 0.0, 1.0))
        dr = self.W + self.seg + 24.0 - xr
        head = xr + u * dr
        ln = (xr - xl) + (self.seg - (xr - xl)) * self._ease(u / 0.3)
        return head, head - ln

    def b_head_tail(self, r, tf, q):
        """head / tail x of row r's B segment at frame time tf (camera progress q): enters from the left, lands on the (sliding) row extent."""
        xl, xr = self.B.ext[r]
        s0, s1 = self.b_row(r)
        e = self._ease((tf - s0) / (s1 - s0))
        shift = (1.0 - q) * self.rt.xb
        tgt = xr + shift
        x_in = -self.seg - 8.0
        head = x_in + (tgt - x_in) * e
        ln = self.seg + ((xr - xl) - self.seg) * e
        return head, head - ln, e, s1

    def parts(self, i, ss=2):
        st = self.rt
        Hs, Ws = self.H * ss, self.W * ss
        A = np.zeros((Hs, Ws), np.float32)
        B = np.zeros((Hs, Ws), np.float32)
        L = np.zeros((Hs, Ws), np.float32)
        a0 = self.a_ss if ss == 2 else DT.place_matte(st._hold(st.wa), self.W, self.H, ss)
        b0 = self.b_ss if ss == 2 else DT.place_matte(st._hold(st.wb), self.W, self.H, ss)
        subs = st.sub_times(i)
        for tf, q in subs:
            Ls = np.zeros((Hs, Ws), np.float32)
            Bs = np.zeros((Hs, Ws), np.float32)
            sa, sb = st.shifts(q)
            A += self._shift(a0, sa * ss) if (i >= st.i_start - 1 and q > 0.0) else a0
            fade = float(np.clip((tf - st.i_start) / 3.0, 0.0, 1.0))
            if tf >= st.i_start - 1 and q > 0.0:
                for r in range(self.rows):
                    if self.A.ext[r] is None:
                        continue
                    h, t = self.a_head_tail(r, q)
                    _seg(Ls, self.ys[r], t, h, self.thick, ss, fade)
            bsh = self._shift(b0, sb * ss)
            for r in range(self.rows):
                if self.B.ext[r] is None or tf < self.b_start - 1:
                    continue
                h, t, e, land = self.b_head_tail(r, tf, q)
                w = float(np.clip((tf - land) / 3.0, 0.0, 1.0))          # the bar hands over to the row's letter pixels
                _seg(Ls, self.ys[r], t, h, self.thick, ss, 1.0 - w)
                rev = self._ease((tf - land) / self.rev_dur)
                if tf >= land:
                    th = self.thick + (self.pitch + 3.0 - self.thick) * rev
                    y0, y1 = (self.ys[r] - 0.5 * th) * ss, (self.ys[r] + 0.5 * th) * ss
                    iy0, iy1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), Hs)
                    if iy1 > iy0:
                        yy = np.arange(iy0, iy1)
                        cov = np.clip(np.minimum(y1, yy + 1.0) - np.maximum(y0, yy.astype(float)), 0, 1).astype(np.float32)
                        Bs[iy0:iy1] = np.maximum(Bs[iy0:iy1], bsh[iy0:iy1] * cov[:, None])
            L += Ls
            B += Bs
        n = float(len(subs))
        return A / n, B / n, L / n


# ----------------------------------------------------------------------------------------------- g03: the circuit_board library transition, screen-space
class CircuitScene:
    """The `circuit_board` transition (mograph/modules/transitions/circuit_board.py) between the two words, at its defaults (orthogonal runs with 45-degree corners,
    lane-split bus, jogs, vias), in SCREEN SPACE: the static cam3d home camera (no type camera, no 3D cable near a lens: the words and the route are in one plane).
    A leaves toward the LEFT (exit side left: its strokes leave with the sliding word), the bus carries the strands left -> right across the frame, B's strands arrive
    from the right into its glyph outlines. Restrained: a dotted face has one strand per dot contour, so the strands the module sees (and the words draw) are thinned to
    `k` evenly spread ones; the solid fill layers are off (the solid letters are the sliding words). Module time is driven by the camera ease: tau(q)."""

    def __init__(self, typ, words, size, fps, k=24, dur=1.5, exit_s=0.5, entry_s=1.0, exit_side="left", entry_side="right"):
        from tools.fusion import spec as S
        from mograph.render import cam3d
        import mograph.modules.words._strand as ST
        white = DF.WHITE
        A, B = dict(words[0]), dict(words[1])
        A.update(t0=0.0, form_s=0.5, exit_s=exit_s, t1=0.5 + 1.0 + exit_s, exit=exit_side)
        B.update(t0=0.0, form_s=entry_s, exit_s=0.3, t1=entry_s + 6.0 + 0.3, entry=entry_side)
        for w in (A, B):
            w.pop("transition", None)
        spec = dict(size=list(size), fps=int(fps), duration=14.0, bpm=None, bar=4, camera=None, words=[A, B],
                    transitions=[dict(module="circuit_board", dur=float(dur))], palette=white,
                    **{kk: typ[kk] for kk in ("seed", "energy", "mood", "line") if kk in typ})

        def thin(b):
            if b.N <= k:
                return b
            order = np.argsort(b.P[:, 0, 1])                                  # by the head's y: an even spread over the rows
            return b.subset(order[np.linspace(0, b.N - 1, k).round().astype(int)])
        oe, on = ST.StrandWord.exit_bundle, ST.StrandWord.entry_bundle
        ST.StrandWord.exit_bundle = lambda self: thin(oe(self))
        ST.StrandWord.entry_bundle = lambda self: thin(on(self))
        try:
            self.sc = S.build_scene(spec, tuple(size))
        finally:
            ST.StrandWord.exit_bundle, ST.StrandWord.entry_bundle = oe, on
        cam3d.set_world(*self.sc.world)
        self.cam = cam3d.Camera.home()
        self.size, self.fps = tuple(size), int(fps)
        self.words = self.sc.info["words"]
        self.tr = self.sc.info["transitions"][0]
        for m in (it for it, _a, _b in self.sc.seq.timeline if hasattr(it, "placement")):
            m.fill_layers = lambda t: []
            step = max(1, int(round(m.N / float(k))))
            keep_out = (np.asarray(m._out["rank"]) % step) == 0
            keep_in = (np.asarray(m._in["rank"]) % step) == 0
            tm = m.timing

            def mk(orig, mod, ko, ki):
                def win(t):
                    ws = orig(t)
                    kp = ko if t >= mod.timing.form + mod.timing.hold else ki
                    return [w if kp[j] else None for j, w in enumerate(ws)]
                return win
            m.windows = mk(m.windows, m, keep_out, keep_in)
        self.t0, self.t1 = float(self.words[0]["release"]), float(self.words[1]["hit"])
        self._clock = None
        self._cache = {}

    def matte(self, tau, size=None):
        if tau is None:
            return None
        from mograph.render import cam3d
        cam3d.set_world(*self.sc.world)
        k = round(tau * 1000)
        if size is None and k in self._cache:
            return self._cache[k]
        sz = size or self.size
        f = self.sc.seq.frame(min(max(tau, 0.0), 13.9), self.cam, size=sz, S=1).max(-1).astype(np.float32) / 255.0
        if size is None:
            if len(self._cache) > 12:
                self._cache.clear()
            self._cache[k] = f
        return f

    def fit_clock(self, thr=6.0e-3, step=1.0 / 48.0, pass_through=0.03):
        """module time <-> camera progress q: q in 0..1 spreads the VISIBLE module time (A's strokes leaving, the bus, B's strands) evenly, invisible stretches
        are crossed at 3 % of that rate: the circuit scene has strands on screen all along the move and it eases exactly like the camera."""
        ts = np.arange(self.t0, self.t1 + 1e-6, step)
        mass = np.array([float(self.matte(t, (288, 512)).mean()) for t in ts])
        vis = mass > thr
        w = np.clip(mass / 0.03, pass_through, 1.0)                      # thin moments (little type on screen) are crossed fast, busy ones linger
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
        k0 = next((j for j in range(len(ts) - 3) if vis[j:j + 4].all()), 0)
        self._clock = (ts, cum, k0)
        self.probe_mass = (ts, mass)
        return float(ts[k0])

    def tau_of_q(self, q):
        ts, cum, k0 = self._clock
        return float(np.interp(cum[k0] + float(np.clip(q, 0.0, 1.0)) * (cum[-1] - cum[k0]), cum, ts))


class ModuleComposer(SlideComposer):
    """g03: sliding solid words (as g01) + the circuit_board strands in screen space, module time = f(q)."""

    def parts(self, i, ss=2):
        st = self.rt
        a0 = self._placed("A", st._hold(st.wa), ss)
        b0 = self._placed("B", st._hold(st.wb), ss)
        A = np.zeros_like(a0)
        B = np.zeros_like(b0)
        L = np.zeros_like(a0)
        subs = st.sub_times(i)
        for tf, q in subs:
            (sax, say), (sbx, sby) = st.shifts_xy(q)
            A += self._shift(a0, sax * ss, say * ss) if (i >= st.i_start - 1 and q > 0.0) else a0
            g = st.gate_of(q)
            if g > 0.0:
                B += g * self._shift(b0, sbx * ss, sby * ss)
            if i >= st.i_start - 1 and q > 0.0:
                m = st.cs.matte(st.cs.tau_of_q(q))
                L += DT.place_matte(m, self.W, self.H, ss)
        n = float(len(subs))
        return A / n, B / n, L / n


# ----------------------------------------------------------------------------------------------- CLI
SHEET_I = (0, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 70, 80, 100)


def _pair(v):
    return None if v is None else tuple(float(x) for x in str(v).split(","))


def _brk(v):
    return None if not v else [p for p in str(v).split("/") if p]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-mp4", required=True)
    ap.add_argument("--matte-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="P06:SEARCHING,A LIGHT")
    ap.add_argument("--i-start", type=int, default=24)
    ap.add_argument("--xa", type=float, default=830.0, help="word A's total slide to the left (video px at q = 1): A is off screen at q = (its right edge) / xa")
    ap.add_argument("--xb", type=float, default=850.0, help="word B's offset to the right at q = 0 (video px): it slides in as q grows")
    ap.add_argument("--lines-start", type=int, default=28, help="frame the first strands enter from the left edge")
    ap.add_argument("--lines-form", type=float, default=1.4, help="seconds the lanes entry takes (strands arrive, peel, draw the outline)")
    ap.add_argument("--lines", default="circuit", choices=["scan", "lanes", "circuit"], help="scan = the words' own scanline rows are the lines (g02); lanes = the lanes entry strands (g01); circuit = the circuit_board library transition, screen space (g03)")
    ap.add_argument("--centre", action=argparse.BooleanOptionalAction, default=True, help="both words centred horizontally (g03/g04; --no-centre = g01 / g02 placement: A right, B left of centre)")
    ap.add_argument("--b-font", default="Tourney[wdth,wght].ttf", help="word B font (liked.json file name); 'none' = the piece's GeistPixel (g01 / g02)")
    ap.add_argument("--b-wght", type=float, default=400.0, help="word B font weight axis")
    ap.add_argument("--b-hf", type=float, default=0.72, help="word B box height as a share of the frame height (a tall face needs it)")
    ap.add_argument("--gate-lo", type=float, default=0.5, help="word B's alpha rises from this camera progress q to 1 at the landing")
    ap.add_argument("--b-k", type=float, default=0.85, help="word B scale (box width / height factor): a tall outline face needs a little less to keep every letter >= 60 %% visible")
    ap.add_argument("--b-dy", type=float, default=-0.18, help="word B vertical nudge (fraction of the frame height, - = up)")
    ap.add_argument("--k-strands", type=int, default=24, help="circuit: strands the module carries (thinned from the dotted faces' hundreds)")
    ap.add_argument("--rows", type=int, default=36, help="scan: rows across the word height (one straight segment per row and word)")
    ap.add_argument("--seg", type=float, default=0.34, help="scan: segment length as a share of the frame width")
    ap.add_argument("--thick", type=float, default=5.0, help="scan: segment thickness (video px)")
    ap.add_argument("--n-lines", type=int, default=30)
    ap.add_argument("--width", type=float, default=3.2, help="strand width in world px (the lanes default is 1.5: hairlines too thin to read as lines)")
    ap.add_argument("--speed", type=float, default=520.0, help="strand speed in world px / s (the module's `speed` param; the frame is 540 world px wide)")
    ap.add_argument("--tail", type=float, default=0.34, help="strand length as a share of the frame width")
    ap.add_argument("--wf", type=float, default=0.76)
    ap.add_argument("--b-dx", type=float, default=-0.11)
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--fps", type=int, default=24)
    # model_design (20_open-model-design) flags: every default = g04
    ap.add_argument("--a-font", default=None, help="word A font (liked.json file name); default = the piece's Doto")
    ap.add_argument("--a-wf", type=float, default=None, help="word A box width share (default --wf)")
    ap.add_argument("--b-wf", type=float, default=None, help="word B box width share (default --wf x --b-k)")
    ap.add_argument("--a-pos", default=None, help="word A placement 'dx,dy' (fractions of the frame; default 0,0 when centred)")
    ap.add_argument("--b-pos", default=None, help="word B placement 'dx,dy' (default 0,--b-dy when centred)")
    ap.add_argument("--a-lines", default=None, help="word A line breaks 'SEA/RCH/ING' (default: the engine's)")
    ap.add_argument("--b-lines", default=None, help="word B line breaks (default: the piece's)")
    ap.add_argument("--a-exit", default="left", choices=["left", "right", "up", "down"], help="side word A slides off toward (circuit only for non left/right)")
    ap.add_argument("--b-entry", default="right", choices=["left", "right", "up", "down"], help="side word B slides in from")
    ap.add_argument("--build-only", action="store_true", help="build the scenes and compose frames 30 and 44, then stop (preflight)")
    ap.add_argument("--front", action="store_true", help="type in FRONT of the runner (no matte occlusion); default = behind")
    a = ap.parse_args(argv)
    outd, review, local = dd.version_dirs(a.out)
    frames = DF.read_frames(a.depth_mp4)
    mattes = DF.read_mattes(a.matte_dir, len(frames))
    t0 = time.time()
    st = SlideType(a.type, len(frames), a.fps, a.i_start, wf=a.wf, dx_b=a.b_dx, xa=a.xa, xb=a.xb, lines_start=a.lines_start, lines_form=a.lines_form,
                   n_lines=a.n_lines, tail=a.tail, speed=a.speed, width=a.width, lines=a.lines, b_font=(None if str(a.b_font).lower() in ("none", "") else a.b_font),
                   b_axes=({"wght": a.b_wght, "wdth": 100.0} if a.b_wght > 0 else {}), b_hf=a.b_hf, centre=a.centre, k_strands=a.k_strands, b_k=a.b_k, dy_b=a.b_dy, gate=(a.gate_lo, 1.0),
                   a_font=a.a_font, a_axes={}, a_wf=a.a_wf, b_wf=a.b_wf, a_pos=_pair(a.a_pos), b_pos=_pair(a.b_pos), a_lines=_brk(a.a_lines), b_lines=_brk(a.b_lines),
                   a_exit=a.a_exit, b_entry=a.b_entry)
    if a.front:
        mattes = [np.zeros_like(m) for m in mattes]                      # no occlusion: the type sits over the runner
    comp = (SlideComposer(st, frames, mattes) if a.lines == "lanes" else ModuleComposer(st, frames, mattes) if a.lines == "circuit" else ScanComposer(st, frames, mattes, a.rows, a.seg, a.thick))
    print("scenes %.1fs | A %s | B %s | lines (%s): %d, %d px segments, entry %.2fs from frame %d | landing frame %d" % (
        time.time() - t0, st.words[0]["lines"], st.words[1]["lines"], a.lines, a.rows if a.lines == "scan" else a.n_lines, int(a.tail * 432), a.lines_form, a.lines_start, st.i_land), flush=True)
    if a.build_only:                                                    # model_design preflight: scenes built + two composed frames, no video
        for i in (30, 44):
            comp.frame(i)
        print("build-only ok")
        return
    idea = os.path.basename(os.path.dirname(os.path.normpath(os.path.abspath(a.out)))).split("_", 1)[-1]
    mp4 = dd.unique_path(os.path.join(outd, "motion-graphic_%s_%s_%s.mp4" % (idea, os.path.basename(os.path.normpath(a.out)), a.tag)))
    proc = dd._ffmpeg_writer(mp4, comp.W, comp.H, a.fps)
    keep, mass, pos = {}, [], []
    t1 = time.time()
    for i in range(len(frames)):
        fr = comp.frame(i)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        A, B, L = comp.parts(i, 1)
        mass.append([round(float(x.mean()), 5) for x in (A, B, L)])
        pos.append(round(st.q(i), 4))
        if i in SHEET_I:
            keep[i] = fr.copy()
        if i % 12 == 0:
            print("  frame %d/%d  %.0fs" % (i, len(frames), time.time() - t1), flush=True)
    proc.stdin.close()
    proc.wait()
    import re
    gm = re.search(r"_(g\d\d)\.mp4$", mp4)
    gen = ("_" + gm.group(1)) if gm else ""
    sheet = dd.unique_path(os.path.join(review, "slide_sheet%s.png" % gen))
    dd._sheet({"slide + lines + form": [(keep[i], "f%d" % i) for i in sorted(keep)]}, sheet, tw=120, cols=len(keep))
    json.dump(dict(flag=a.type, args=vars(a), q=pos, mass_A_B_lines=mass, ink_log=comp.track.log, words=[{k: w[k] for k in ("text", "lines", "release", "cap_px")} for w in st.words]),
              open(os.path.join(local, "slide%s.json" % gen), "w"), indent=1)
    print("slide type -> %s (%.0fs) sheet %s" % (mp4, time.time() - t0, sheet), flush=True)


if __name__ == "__main__":
    main()
