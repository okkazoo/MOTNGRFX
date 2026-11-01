"""gradient_fade: a huge word with a one-point-perspective extrusion toward a vanishing point; the walls run a smooth gradient
from warm grey at the face down to the ground colour at the back, so the depth fades into the dark, and the vanishing point
travels across the top of the word on an eased arc, so the long gradient walls sweep under the type like a slow camera move.
Forming: the letters arrive along the entry direction as comets, a face with a hairline streak behind it; as each lands the
streak opens into the gradient walls, which then grow toward the vanishing point. Leaving: the walls pull back into a streak
and the letters run out along the exit direction. The vanishing point starts at the entry side and ends at the exit side.
(mechanism: 11_mathtype q3 gradfade, rewritten for the Bundle interface)

Params: depth (back scale of the extrusion, default 0.22), vp_height (how far above the word the vanishing point arcs, in word
heights, default 1.7), slabs (default 30)."""
import math
import numpy as np
from mograph.modules.core import register, Bundle, Layer, unit, mixc, clamp
from mograph.modules.words._slab import (SlabWord, LetterFlow, PrismItem, prism_point, grad_prisms, ssm, eo, ei, eio,
                                         simplify, contour_sign, corner_points)


@register("gradient_fade")
class GradientFade(SlabWord):
    description = ("Letters arrive as comets, then the streak behind each one opens into a smooth gradient extrusion toward a "
                   "vanishing point that fades from warm grey to the ground colour, and the vanishing point sweeps across "
                   "the top so the walls glide under the type. It leaves as it came: the walls pull back into streaks.")
    DEMO_WORDS = "FADE"
    DEFAULT_FONT = ("bigsh", {"wght": 900})
    P_IN, P_OUT = 2.4, 2.2

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 9)))
        ext = max(pl.w, pl.h)
        self.Lb = float(P.get("tail_len") or 0.22 * ext)
        self.sb_end = float(P.get("depth", 0.22))
        self.nsl = int(P.get("slabs", 30))
        n = len(self.letters)
        self.n = n
        self.cons_s = [[simplify(c, 0.55) for c in l["cons"]] for l in self.letters]
        self.sg = [contour_sign(l["cons"]) for l in self.letters]
        self.edge_pts = [corner_points(cs, kmax=6) for cs in self.cons_s]
        self.N = int(sum(len(e) for e in self.edge_pts))
        pts = [np.concatenate([pl.lift(c) for c in l["cons"]]) for l in self.letters]
        self.flow = LetterFlow(self, pts, self.Lb, rng, p_in=self.P_IN, p_out=self.P_OUT)
        self.near = mixc(self.palette.fg, self.palette.bg, 0.30)
        self.far = self.palette.bg
        # vanishing point path (local 2D): from the entry side to the exit side, arcing above the word
        d_in2 = np.array([self.d_in @ pl.u, self.d_in @ pl.v])
        d_out2 = np.array([self.d_out @ pl.u, self.d_out @ pl.v])
        e_in = d_in2 / np.linalg.norm(d_in2) if np.linalg.norm(d_in2) > 0.35 else np.array([1.0, 0.0])
        e_out = d_out2 / np.linalg.norm(d_out2) if np.linalg.norm(d_out2) > 0.35 else np.array([1.0, 0.0])
        lat = np.array([-e_in[1], e_in[0]])
        if lat[1] > 0:
            lat = -lat
        wh = 0.5 * np.array([pl.w, pl.h])
        R = 1.15 * float(np.abs(e_in) @ wh)
        Rb = 1.15 * float(np.abs(e_out) @ wh)
        lat_ext = float(np.abs(lat) @ wh)
        hgt = float(P.get("vp_height", 1.7))
        self.vp_a = -e_in * R
        self.vp_b = e_out * Rb
        self.vp_lat = lat
        self.vp_A0 = 0.15 * lat_ext
        self.vp_A1 = hgt * 0.5 * self.cap

    def vp(self, t):
        u = eio(clamp((t - 0.05 * self.duration) / (0.9 * self.duration)), 2.2)
        base = self.vp_a + (self.vp_b - self.vp_a) * u
        return base + self.vp_lat * (self.vp_A0 + self.vp_A1 * math.sin(math.pi * u))

    # ------------------------------------------------------------ state
    def state(self, j, t):
        tm, fl = self.timing, self.flow
        off, v, tau, entering = fl.at(j, t)
        if entering:
            Ti = fl.Ti[j]
            w = ssm(t, 0.65 * Ti, Ti + 0.25)
            g = ssm(t, 0.35 * tm.form, 0.95 * tm.form) if t < tm.form else 1.0
            s = ssm(t, 0.0, 0.55 * Ti)
            d3 = -self.d_in * self.Lb * (1 - w)
        else:
            tau_ = max(tau, 0.0)
            w = 1.0 - ssm(tau_, 0.40, 0.85)
            g = 1.0 - ssm(tau_, 0.0, 0.55)
            s = 1.0 - ssm(tau_, 0.70, 1.0)
            d3 = -self.d_out * self.Lb * (1 - w)
        sb = 1.0 - (1.0 - self.sb_end) * g
        return dict(off=off, v=v, w=w, d3=d3, sb=sb, s=s, entering=entering)

    # ------------------------------------------------------------ bundle / frames
    def bundle_at(self, t):
        pl = self.placement
        K = 8
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        v = np.zeros(self.N)
        us = np.linspace(0, 1, K)
        i = 0
        vp = self.vp(t)
        for j in range(self.n):
            st = self.state(j, t)
            it = PrismItem([], 1.0, st["off"], st["d3"], vp, st["sb"])
            for p in self.edge_pts[j]:
                Pm[i] = np.array([prism_point(pl, it, p[None], u)[0] for u in us])
                Dv[i] = self.d_in if st["entering"] else self.d_out
                v[i] = st["v"]
                i += 1
        col = np.tile(np.array(self.palette.fg, float), (self.N, 1))
        return Bundle(Pm, Dv, col, 1.2, t, v, 0.0)

    def layers(self, t):
        pl, pal = self.placement, self.palette
        vp = self.vp(t)
        items, faces, lines = [], [], []
        for j in range(self.n):
            st = self.state(j, t)
            items.append(PrismItem(self.cons_s[j], self.sg[j], st["off"], st["d3"], vp, st["sb"], st["s"]))
            faces.append(([pl.lift(c) + st["off"] for c in self.letters[j]["cons"]], st["s"]))
            if st["w"] < 0.98:
                it = items[-1]
                for p in self.edge_pts[j]:
                    lines.append((np.array([prism_point(pl, it, p[None], u)[0] for u in np.linspace(0, 1, 6)]), 1.0 - st["w"]))
        L = grad_prisms(pl, items, self.nsl, self.near, self.far, pal.bg, gamma=0.8, bias0=3.0)
        done = {}
        for pol, k in lines:
            done.setdefault(round(k, 2), []).append(pol)
        for k, pols in done.items():
            L.append(Layer(pols, pal.fg, "line", 1.2 * k, closed=False, bias=5.0))
        for polys, s in faces:
            if s > 0.004:
                L.append(Layer(polys, mixc(pal.bg, pal.fg, s), "fill", bias=6.0))
        return L
