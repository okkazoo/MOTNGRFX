"""beat_bars: every letter is a bar. The letters shoot in along the entry direction as streaks (a letter face with a trailing
edge line per corner), land one after another, and then their solid graduated bars (grey at the face to ground at the back)
swing out to the rest axis and jump to new lengths on the beat, staccato like a bar chart, ending on a clean ramp before
they settle flat into the letters. To leave, the faces release, the bars jump long again as the letters accelerate, and the
edge lines run out along the exit direction. (mechanism: 11_mathtype b3R 'beat bars' / 13_camera strips Bars, rewritten for
the Bundle interface: the strands of the hand-off bundle are the bar corner edges)

Params: axis (rest axis of the bars: 'down' default oblique, any Port name, angle in degrees or a 3-vector), beats (jump
times as fractions of the form phase), depth (how far the bars lean away from the viewer, default 0.3)."""
import math
import numpy as np
from mograph.modules.core import register, Bundle, Layer, Port, unit, mixc, clamp
from mograph.modules.words._slab import (SlabWord, ssm, eo, ei, eio, simplify, contour_sign, wall_quads, wall_layers,
                                         corner_points)


@register("beat_bars")
class BeatBars(SlabWord):
    description = ("Letters shoot in as streaks, land one after another, and their solid graduated bars swing out and jump "
                   "to new lengths on the beat, staccato, before settling flat into the letters. To leave, the bars jump "
                   "long again as the letters accelerate away and shrink to edge lines along the exit direction.")
    DEMO_WORDS = "BEATS"
    DEFAULT_FONT = ("bigsh", {"wght": 900})
    P_IN, P_OUT = 2.4, 2.2
    BEATS = (0.40, 0.56, 0.72)

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 7)))
        ext = max(pl.w, pl.h)
        self.Lb = float(P.get("tail_len") or 0.22 * ext)          # bar length (streak) at the hand-offs
        self.Dscale = 0.55 * ext
        self.Ls = max(6.0, self.Dscale / 12.0)
        self.beats = tuple(P.get("beats", self.BEATS))
        self.tilt_z = float(P.get("depth", 0.3))
        ax = P.get("axis")
        if ax is None:
            a = 0.34 * pl.u + 0.94 * pl.v
        else:
            a = Port.coerce(ax, "down").side(pl)
            if isinstance(ax, str) and ax == "down":
                a = 0.34 * pl.u + 0.94 * pl.v
        self.ax_rest = unit(unit(a) + self.tilt_z * pl.n)
        letters = self.letters
        n = len(letters)
        self.n = n
        self.cons_s = [[simplify(c, 0.55) for c in l["cons"]] for l in letters]
        self.sg = [contour_sign(l["cons"]) for l in letters]
        self.edge_pts = [corner_points(cs) for cs in self.cons_s]
        o = pl.origin
        rin, rout, Di, Dx = [], [], [], []
        for l in letters:
            Pw = np.concatenate([pl.lift(c) for c in l["cons"]]) - o
            rin.append(float((Pw @ self.d_in).mean()))
            rout.append(float((Pw @ self.d_out).mean()))
            Di.append(float((Pw @ self.d_in).max()) + self.reach_in)
            Dx.append(self.reach_out - float((Pw @ self.d_out).min()) + self.Lb)
        rin, rout = np.array(rin), np.array(rout)
        fin = 1.0 - (rin - rin.min()) / max(np.ptp(rin), 1e-6)
        fout = 1.0 - (rout - rout.min()) / max(np.ptp(rout), 1e-6)
        self.fin = fin
        tm = self.timing
        F, E = tm.form, tm.exit
        self.Ti = (0.30 + 0.20 * fin) * F + 0.04 * rng.rand(n)
        self.Di = np.array(Di) + 0.08 * ext * rng.rand(n)
        self.dx = 0.38 * E * (0.9 * fout + 0.1 * rng.rand(n))
        self.Tx = E - self.dx
        self.Dx = np.array(Dx) + 0.06 * ext * rng.rand(n)
        sc = self.Dscale
        self.tg_in = [list(rng.uniform(0.08, 0.62, len(self.beats) - 1) * sc) + [(0.10 + 0.52 * (1 - fin[j])) * sc] for j in range(n)]
        self.tg_out = [list(rng.uniform(0.10, 0.62, 2) * sc) + [self.Lb] for j in range(n)]
        self.N = int(sum(len(e) for e in self.edge_pts))
        self.near = mixc(self.palette.fg, self.palette.bg, 0.12)
        self.far = mixc(self.palette.fg, self.palette.bg, 0.80)

    # ------------------------------------------------------------ state
    def bar_len_in(self, j, t):
        q = t / self.timing.form
        v = prev = self.Lb
        for b, tg in zip(self.beats, self.tg_in[j]):
            u = eio(clamp((q - b - 0.04 * self.fin[j]) / 0.13), 3.0)
            v += (tg - prev) * u
            prev = tg
        return v * (1.0 - eio(clamp((q - 0.84 - 0.04 * self.fin[j]) / 0.12), 2.0))

    def bar_len_out(self, j, tau):
        v = 0.0
        prev = 0.0
        for (a, b), tg in zip(((0.02, 0.18), (0.40, 0.56), (0.78, 0.98)), self.tg_out[j]):
            u = eio(clamp((tau - a) / (b - a)), 3.0)
            v += (tg - prev) * u
            prev = tg
        return v

    def state(self, j, t):
        tm = self.timing
        if t < tm.form + tm.hold:
            tau = t / self.Ti[j]
            d = self.Di[j] * (1 - eo(tau, self.P_IN)) if tau < 1 else 0.0
            off = -self.d_in * d
            D = self.bar_len_in(j, min(t, tm.form)) if t < tm.form else 0.0
            w = ssm(t, 0.75 * self.Ti[j], self.Ti[j] + 0.25)
            ax = unit((1 - w) * (-self.d_in) + w * self.ax_rest)
            s = ssm(tau, 0.08, 0.40)
            v = self.P_IN * self.Di[j] / self.Ti[j]
        else:
            te = t - tm.form - tm.hold
            tau = (te - self.dx[j]) / self.Tx[j]
            if tau <= 0:
                d = 0.0
            elif tau >= 1:
                d = self.Dx[j] * (1 + self.P_OUT * (tau - 1))
            else:
                d = self.Dx[j] * ei(tau, self.P_OUT)
            off = self.d_out * d
            D = self.bar_len_out(j, max(tau, 0.0)) if tau > 0 else 0.0
            w = 1.0 - ssm(tau, 0.0, 0.30)
            ax = unit((1 - w) * (-self.d_out) + w * self.ax_rest)
            s = 1.0 - ssm(tau, 0.65, 0.97)
            v = self.P_OUT * self.Dx[j] / self.Tx[j]
        return dict(off=off, D=D, ax=ax, s=s, v=v)

    # ------------------------------------------------------------ bundle / frames
    def bundle_at(self, t):
        pl = self.placement
        K = 8
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        v = np.zeros(self.N)
        f = np.linspace(0, 1, K)[:, None]
        entry = t < self.timing.form + self.timing.hold
        i = 0
        for j in range(self.n):
            st = self.state(j, t)
            for p in self.edge_pts[j]:
                head = pl.lift(p[None])[0] + st["off"]
                Pm[i] = head + st["ax"] * st["D"] * f
                Dv[i] = self.d_in if entry else self.d_out
                v[i] = st["v"]
                i += 1
        col = np.tile(np.array(self.near, float), (self.N, 1))
        return Bundle(Pm, Dv, col, 1.2, t, v, 0.0)

    def layers(self, t):
        pl = self.placement
        solid, part, faces, lines = [], [], [], []
        L = []
        for j in range(self.n):
            st = self.state(j, t)
            if st["D"] > 0.5:
                q = wall_quads(pl, self.cons_s[j], self.sg[j], st["off"], st["ax"], st["D"], self.Ls)
                (solid if st["s"] >= 0.998 else part).append((q, st["s"], st["D"]))
                for p in self.edge_pts[j]:
                    h = pl.lift(p[None])[0] + st["off"]
                    lines.append(np.array([h, h + st["ax"] * st["D"]]))
            polys = [pl.lift(c) + st["off"] for c in self.letters[j]["cons"]]
            faces.append((polys, st["s"]))
        for q, s, D in solid + part:
            if s > 0.003:
                L += wall_layers([q], self.Ls, max(D, 0.45 * self.Dscale), self.near, self.far, alpha=s if s < 0.998 else 1.0, bias0=3.0, gamma=0.85)
        if lines:
            L.append(Layer(lines, self.near, "line", 1.1, closed=False, bias=5.0))
        for polys, s in faces:
            if s > 0.003:
                L.append(Layer(polys, self.palette.fg, "fill", alpha=s, bias=6.0))
        return L
