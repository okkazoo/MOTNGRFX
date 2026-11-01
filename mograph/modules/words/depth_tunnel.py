"""depth_tunnel: the frame is a slab and the word is cut into it. Strands draw the outline on the surface, then every letter
opens in turn as a shaft going straight down into the page (the counters stay as pillars): walls shaded amber to vermilion to
deep with depth, contour bands every few slabs, and bright pulses running down the shafts like the strands streaming away
down them. On exit the shafts fill back to the surface and the strands leave. Exit/entry 'in' sends the strands straight
down the shaft axis. (mechanism: 11_mathtype q4 deep tunnel; 13_camera relay / book_counters tunnels, rewritten for the Bundle
interface: true 3D walls, the slab is a ground-coloured mask with the word cut out of it)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc, unit, AMBER, VERM
from mograph.modules.words._depth import MechWord, ss, eio, eo, seg, letter_edges

PAGE = 1.0e5
DEEP = (40, 14, 10)
DEP = [0.7, 1.0, 0.5, 0.85, 0.6, 0.95]


@register("depth_tunnel")
class DepthTunnel(MechWord):
    description = ("Strands draw the outline on a slab, then each letter opens in turn as a shaft straight down into the page, "
                   "walls shaded amber to vermilion to deep with contour bands and pulses streaming down the shafts. On exit "
                   "the shafts fill back to the surface and the strands leave.")
    DEMO_WORDS = "DEEP"
    DEFAULT_FONT = "ericaone"
    DEFAULT_PALETTE = "bone"
    DEFAULT_TIMING = (2.4, 1.0, 2.0)
    MS = 0.30
    STRAND_BIAS = 2.0e5
    LIB_CAMERA = dict(yaw=(24.0, -10.0, -24.0), pitch=(16.0, 12.0, 14.0), dist=(1150.0, 1120.0, 1150.0), cz=130.0)
    KS = 14

    def mech_build(self):
        pl = self.placement
        self.nl = len(self.lay["letters"])
        self.E = letter_edges(self.lay["letters"], 0.8)
        self.idx = [np.where(self.E["L"] == i)[0] for i in range(self.nl)]
        self.cap = self.lay["cap"]
        self.Dmax = float(self.params.get("depth", 2.2)) * self.cap
        self.LT = unit(0.5 * pl.u - 0.86 * pl.v - 0.2 * pl.n)
        self.page = [self.lift(np.array([[-pl.w, -pl.h], [pl.w, -pl.h], [pl.w, pl.h], [-pl.w, pl.h]]))] + \
                    [self.lift(c) for c in self.cons2()]

    def wall(self, u):
        stops = [(0.0, AMBER), (0.35, VERM), (0.75, (96, 32, 22)), (1.0, DEEP)]
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if u <= b:
                return mixc(ca, cb, (u - a) / (b - a))
        return DEEP

    def depth_of(self, i, p):
        a = 0.5 * i / max(self.nl - 1, 1)
        return DEP[i % len(DEP)] * self.Dmax * eo(seg(p, a, a + 0.5), 2.6)

    def pulse(self, t, z):
        """bright bands running down the shafts (one every 0.55 s after the shafts open)."""
        pm = 0.0
        t0 = self.t_ms + 0.35 * (self.timing.form - self.t_ms)
        n = 0
        while True:
            tn = t0 + 0.55 * n
            if tn > t or n > 12:
                break
            zp = (t - tn) * 0.9 * self.Dmax - 0.1 * self.Dmax
            pm = max(pm, math.exp(-((z - zp) / (0.07 * self.Dmax)) ** 2))
            n += 1
        return pm

    def mech_layers(self, t):
        pal, pl, E = self.palette, self.placement, self.E
        p = self.progress(t)
        out = []
        fa = self.face_alpha(t)
        if p <= 1e-4 and fa <= 0.01:
            return out
        eye = self.eye_at(t)
        K = self.KS
        groups = {}
        floors = []
        pulse_on = (self.timing.form * 0.7 < t < self.t_c - 0.2 * (self.t_c - self.timing.form - self.timing.hold))
        for i in range(self.nl):
            D = self.depth_of(i, p)
            cons = self.lay["letters"][i]["cons"]
            fcol = mixc(pal.fg, DEEP, ss(D / (0.25 * self.cap)))
            floors.append(([self.lift(c, D) for c in cons], fcol))
            if D < 1.0:
                continue
            ix = self.idx[i]
            A, B, N2 = E["A"][ix], E["B"][ix], E["N"][ix]
            n3 = N2[:, 0:1] * pl.u + N2[:, 1:2] * pl.v
            sh = np.clip(-(n3 @ self.LT) * 0.5 + 0.5, 0, 1)
            lv = np.round(sh * 2).astype(int)
            zs = np.linspace(0.0, D, K + 1)
            for k in range(K):
                z0, z1 = zs[k], zs[k + 1]
                P0a, P0b = self.lift(A, z0), self.lift(B, z0)
                P1a, P1b = self.lift(A, z1), self.lift(B, z1)
                mid = 0.25 * (P0a + P0b + P1a + P1b)
                vis = ((eye - mid) * n3).sum(1) < 0
                if not vis.any():
                    continue
                zm = 0.5 * (z0 + z1)
                u = zm / self.Dmax
                base = (150, 52, 30) if (k % 5 == 4 and k < K - 1) else self.wall(u)
                if pulse_on:
                    base = mixc(base, (255, 244, 226), 0.92 * self.pulse(t, zm))
                for m in np.where(vis)[0]:
                    f = 0.66 + 0.17 * lv[m]
                    col = tuple(int(min(255, c * f)) for c in base)
                    groups.setdefault(col, []).append(np.array([P0a[m], P0b[m], P1b[m], P1a[m]]))
        for polys, fcol in floors:
            out.append(Layer(polys, fcol, "fill", alpha=max(fa, ss(p / 0.12)), bias=0.0))
        for col, polys in groups.items():
            out.append(Layer(polys, col, "fill", split=True, bias=0.0))
        out.append(Layer(self.page, pal.bg, "fill", bias=PAGE))
        face = [self.lift(c) for c in self.cons2()]
        out.append(Layer(face, pal.fg, "line", self.w * 1.2, closed=True, bias=PAGE + 1.0))
        return out
