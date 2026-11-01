#!/usr/bin/env python
"""Math-type E7 (kit demo): the sentence drawn by harmonograph pens that resolve into the letter outlines.

One machine per phrase: every glyph outline is a pen, and all pens share the pendulums (two per axis,
frequencies near 2:3 with a small detune, damped along the trace), each pen phase-offset. Pen point s of
outline c at time t:
    F_c(s, t) = O + sum_i A_i (.35 + .65 e^{-d_i tau}) (sin(f_i tau + p_i + phi_c + w_i t), sin(g_i tau + q_i + phi_c + v_i t))
    P_c(s, t) = (1 - R_c(t)) C_c(s) + R_c(t) F_c(s, t),      tau = 2 pi L s
R_c is 1 while the figure swings and resolves to 0 at the landing (arrive ease, staggered left to right),
then rises again on the exit (exit ease) while the lines fade. A faint copy of the figure keeps swinging
behind the read. Mono palette; the key word's pen heads are the one accent.

    python tools/mathtype/harmonograph.py --draft          # 360x450 -> 11_mathtype/v014/local/E7a_harmonograph_gNN_draft.mp4
    python tools/mathtype/harmonograph.py --final          # 1080x1350 + score + mux
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit  # noqa: E402
from kit import camera, draw, io, palette, text, timeline  # noqa: E402

SENTENCE = "There is geometry in the humming of the strings."
DW, DH = 720.0, 900.0
VDIR = os.path.join(kit.REPO, "11_mathtype", "v014")
AMP, LOOPS = 250.0, 7.0                     # figure amplitude (design px), pendulum loops along a pen
FREQ = [(2.0, 3.01), (3.0, 2.0)]            # (x, y) frequency per pendulum pair: near 2:3, slightly detuned


class Harmonograph:
    test, stem, score_preset, notes = "E7a", "E7a_harmonograph", "E3", "kit demo"
    sizes = {"draft": (360, 450), "final": (1080, 1350)}

    def __init__(self, a, be):
        self.be, self.version_dir, self.palette = be, VDIR, palette.get("mono")
        self.code_files = [os.path.abspath(__file__)]
        phrases = text.split_phrases(SENTENCE, n=3)
        self.timeline = timeline.Timeline.build(phrases, bpm=96, start=0.0, form=1.1, trans=0.95, tail=0.5)
        self.lays, self.pens = [], []
        T = kit.torch_or_none()
        for i, p in enumerate(phrases):
            lay = text.layout(text.auto_mix(text.split_lines(p.text)), DW, DH, scale=2.0, text_w=560, cap_max=230,
                              gap=0.22, track=0.05)
            outs = lay.outlines(ds=2.5)
            key = max(range(len(lay.lines)), key=lambda j: lay.lines[j][1] == "Blackout")
            M = max(len(o) for _, o in outs) + 1
            C = np.zeros((len(outs), M, 2), np.float32)
            V = np.zeros((len(outs), M), np.float32)
            for c, (li, o) in enumerate(outs):            # close each outline, pad by repeating its end
                o = np.vstack([o, o[:1]])
                C[c, :len(o)], C[c, len(o):], V[c, :len(o)] = o, o[-1], 1.0
            s = np.pad(np.cumsum(np.linalg.norm(np.diff(C, axis=1), axis=2), 1), ((0, 0), (1, 0)))
            s = s / np.maximum(s[:, -1:], 1e-6)
            xc = np.array([o[:, 0].mean() for _, o in outs])
            rng = np.random.default_rng(7 + i)
            pend = dict(A=[AMP, 0.62 * AMP], d=[0.10, 0.16], p=rng.uniform(0, 2 * np.pi, 4).tolist(),
                        w=(rng.uniform(0.5, 0.9, 2) * (1 if i % 2 == 0 else -1)).tolist())
            self.pens.append(dict(C=T.as_tensor(C, device=be.dev), V=T.as_tensor(V, device=be.dev),
                                  s=T.as_tensor(s.astype(np.float32), device=be.dev),
                                  phi=T.as_tensor(np.linspace(0, 2 * np.pi, len(outs), endpoint=False), device=be.dev),
                                  order=(xc - xc.min()) / max(np.ptp(xc), 1.0),
                                  key=np.array([li == key for li, _ in outs]), pend=pend))
            self.lays.append(lay)

    def figure(self, k, t):
        """F_c(s, t): the shared harmonograph figure traced by every pen of phrase k (K, M, 2)."""
        P, pd = self.pens[k], self.pens[k]["pend"]
        tau = 2 * math.pi * LOOPS * P["s"]
        ph = P["phi"][:, None]
        x = sum(pd["A"][j] * (0.35 + 0.65 * draw.xp.exp(-pd["d"][j] * tau)) * draw.xp.sin(FREQ[j][0] * tau + pd["p"][j] + ph + pd["w"][j] * t)
                for j in range(2))
        y = sum(pd["A"][j] * (0.35 + 0.65 * draw.xp.exp(-pd["d"][j] * tau)) * draw.xp.sin(FREQ[j][1] * tau + pd["p"][2 + j] + ph + 0.8 * pd["w"][j] * t)
                for j in range(2))
        return draw.xp.stack([x + DW / 2, y + DH / 2], -1)

    def state(self, k, t):
        """(R per pen, line intensity, ghost intensity, fill) for phrase k at t."""
        tl, P = self.timeline, self.pens[k]
        t_in = tl.form0 - 0.4 if k == 0 else tl.trans[k - 1][0] + 0.15
        land, r1 = tl.land[k], tl.read[k][1]
        u = (t - t_in - 0.30 * P["order"] * (land - t_in) * 0.5) / ((land - t_in) * 0.85)
        R = 1 - camera.ease("arrive", u)                                  # resolve onto the outline
        alive = float(camera.smoothstep(t_in, t_in + 0.25, t))
        if k < tl.n - 1:
            e = (t - r1) / (tl.trans[k][1] - r1)
            R = R + camera.ease("exit", e)                                 # unwind into the next figure
            alive *= 1 - float(camera.smoothstep(0.25, 0.85, e))
        fill = 0.16 * float(camera.smoothstep(land - 0.15, land + 0.1, t)) * (1 - float(camera.smoothstep(r1, r1 + 0.2, t)))
        return np.clip(R, 0, None), alive, 0.12 * alive, fill

    def frame(self, t, W, H, ss=2):
        be, T = self.be, kit.torch_or_none()
        Ws, Hs, k, lw = W * ss, H * ss, W * ss / DW, W / 360.0       # lw: line weight scales with size
        acc, heads = T.zeros(Hs, Ws, device=be.dev), []
        fillm = T.zeros(H, W, device=be.dev)
        for i in range(self.timeline.n):
            R, alive, ghost, fill = self.state(i, t)
            if alive <= 1e-3:
                continue
            P, F = self.pens[i], self.figure(i, t)
            Rt = T.as_tensor(R, dtype=T.float32, device=be.dev)[:, None, None]
            X = ((1 - Rt) * P["C"] + Rt * F) * k
            reveal = 1.3 * float(camera.smoothstep(self.timeline.form0 if i == 0 else self.timeline.trans[i - 1][0],
                                                   self.timeline.land[i] - 0.25, t))
            I = P["V"] * draw.xp.clip((reveal - P["s"]) / 0.12, 0.0, 1.0) * alive
            acc += draw.splat_polylines(X, I * 0.9 * ss * lw, Hs, Ws)[0]
            acc += draw.splat_polylines(F[::4] * k, P["V"][::4] * ghost * ss * lw, Hs, Ws)[0]   # ghost: every 4th pen
            if fill > 0:
                sd = T.as_tensor(self.lays[i].sdf(), device=be.dev)[None]
                px, py = kit.core.grid(be, W, H, DW / W)
                fillm = T.maximum(fillm, fill * draw.coverage(draw.sample(sd, px, py, 2.0)[0] * W / DW))
            if 0.05 < reveal < 1.0:                                    # accent: the key word's pen heads
                j = draw.xp.clip((reveal * (P["V"].sum(1) - 1)).long(), 0, P["C"].shape[1] - 1)
                hp = X[T.arange(len(j)), j][T.as_tensor(P["key"], device=be.dev)] / ss
                heads += hp.cpu().numpy().tolist()
        ink = draw.box_down(draw.tone(draw.blur(acc, 0.55 * ss * lw), 2.2), ss)
        pal = self.palette
        img = draw.canvas(be, H, W, pal.ground)
        draw.paint(img, T.clamp(ink + fillm, 0, 1), pal.ink)
        if heads:
            px, py = kit.core.grid(be, W, H)
            draw.paint(img, draw.coverage(draw.discs(px, py, heads, 2.2 * W / 360)), pal.accent)
        return draw.to_u8(img)

    def text_mask(self, i, W, H):
        import cv2
        return cv2.resize(self.lays[i].draw(), (W, H), interpolation=cv2.INTER_AREA) > 127


if __name__ == "__main__":
    io.main(lambda a, be: Harmonograph(a, be))
