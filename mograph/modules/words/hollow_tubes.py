"""hollow_tubes: outside and inside at once. Strands draw the outline; then every letter in turn rises toward the viewer as
an open-topped hollow tube (staggered skyline): outer walls graduated amber to vermilion, and through the bone rim the shaded
inside of the tube falling away to black. On exit the tubes sink back into the page, then the strands leave.
(mechanism: 11_mathtype b4 hollow tubes, rewritten for the Bundle interface: true 3D walls, drawn only where they face the eye)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc, unit
from mograph.modules.words._depth import MechWord, ss, eio, eo, seg, letter_edges

DEPTH = [0.42, 0.66, 0.30, 0.80, 0.52, 0.95]


@register("hollow_tubes")
class HollowTubes(MechWord):
    description = ("Strands draw the outline, then each letter rises toward the viewer in turn as an open-topped hollow tube: "
                   "outer walls graduated amber to vermilion, and through the bone rim the shaded inside falling away to "
                   "black. On exit the tubes sink back into the page and the strands run out.")
    DEMO_WORDS = "HOLLOW"
    DEFAULT_FONT = "passionone"
    DEFAULT_PALETTE = "ink"
    DEFAULT_TIMING = (2.4, 0.9, 2.0)
    MS = 0.28
    LIB_CAMERA = dict(yaw=(26.0, -14.0, -24.0), pitch=(16.0, 20.0, 16.0), dist=(1120.0, 1090.0, 1120.0), cz=-90.0)
    KS = 9

    def mech_build(self):
        pal, pl = self.palette, self.placement
        self.nl = len(self.lay["letters"])
        self.E = letter_edges(self.lay["letters"], 0.8)
        self.idx = [np.where(self.E["L"] == i)[0] for i in range(self.nl)]
        self.cap = self.lay["cap"]
        self.Hmax = float(self.params.get("height", 0.95)) * self.cap
        self.LT = unit(-0.7 * pl.u - 0.7 * pl.v - 0.5 * pl.n)
        self.out_n, self.out_f = pal.accents[0], mixc(pal.accents[2], pal.bg, 0.55)
        self.stops = [(0.0, mixc(pal.fg, pal.accents[0], 0.12)), (0.06, mixc(pal.accents[2], pal.fg, 0.1)),
                      (0.18, mixc(pal.accents[2], pal.bg, 0.55)), (0.45, mixc(pal.accents[2], pal.bg, 0.9)),
                      (1.0, mixc(pal.bg, (0, 0, 0), 0.5))]

    def inside(self, u):
        u = min(max(u, 0.0), 1.0)
        for (a, ca), (b, cb) in zip(self.stops, self.stops[1:]):
            if u <= b:
                return mixc(ca, cb, (u - a) / (b - a))
        return self.stops[-1][1]

    def height(self, i, p):
        a = 0.55 * i / max(self.nl - 1, 1)
        return DEPTH[i % len(DEPTH)] * self.Hmax * eo(seg(p, a, a + 0.45), 2.8)

    def mech_layers(self, t):
        pal, pl, E = self.palette, self.placement, self.E
        p = self.progress(t)
        eye = self.eye_at(t)
        K = self.KS
        groups = {}
        floors, lips = [], []
        shade_levels = 4
        for i in range(self.nl):
            h = self.height(i, p)
            cons = self.lay["letters"][i]["cons"]
            fcol = self.inside(min(1.0, h / (0.8 * self.cap))) if h > 0.5 else pal.fg
            floors.append((cons, fcol))
            if h < 1.0:
                continue
            ix = self.idx[i]
            A, B, N2 = E["A"][ix], E["B"][ix], E["N"][ix]
            n3 = N2[:, 0:1] * pl.u + N2[:, 1:2] * pl.v
            sh = np.clip((n3 @ self.LT) * 0.5 + 0.5, 0, 1)
            lv = np.round(sh * (shade_levels - 1)).astype(int)
            zs = np.linspace(-h, 0.0, K + 1)
            for k in range(K):
                z0, z1 = zs[k], zs[k + 1]
                P0a, P0b = self.lift(A, z0), self.lift(B, z0)
                P1a, P1b = self.lift(A, z1), self.lift(B, z1)
                mid = 0.25 * (P0a + P0b + P1a + P1b)
                ext = ((eye - mid) * n3).sum(1) > 0
                um = (k + 0.5) / K
                dabs = um * h
                for m in range(len(A)):
                    if ext[m]:
                        col = mixc(self.out_n, self.out_f, um ** 0.9)
                        f = 0.7 + 0.3 * lv[m] / (shade_levels - 1)
                    else:
                        col = self.inside(dabs / (0.8 * self.cap))
                        f = 0.72 + 0.28 * lv[m] / (shade_levels - 1)
                    col = tuple(int(min(255, c * f)) for c in col)
                    groups.setdefault(col, []).append(np.array([P0a[m], P0b[m], P1b[m], P1a[m]]))
            lips.append([self.lift(c, -h) for c in cons])
        out = []
        fa = self.face_alpha(t)
        for cons, fcol in floors:
            if len(cons) and (p > 1e-4 or fa > 0.01):
                out.append(Layer([self.lift(c) for c in cons], fcol, "fill", alpha=max(fa, ss(p / 0.12)), bias=0.0))
        for col, polys in groups.items():
            out.append(Layer(polys, col, "fill", split=True, bias=0.0))
        for lp in lips:
            out.append(Layer(lp, pal.fg, "line", 2.0, closed=True, bias=3.0))
        return out
