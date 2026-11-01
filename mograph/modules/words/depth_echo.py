"""depth_echo: the type's outline is copied backward through true depth. Strands draw the outline, then the copies peel off the
face one after another and fan out behind it as a stack of cut-out outlines (each paper-filled, fading into the ground with
depth), breathing along the stack and swelling in perspective. On exit the stack folds back into the face, then the strands
leave. (mechanism: breath_echo / echo_traverse / q5_loud, rewritten for the Bundle interface)"""
import math
import numpy as np
from mograph.modules.core import register, Layer, mixc, clamp
from mograph.modules.words._depth import MechWord, ss, eio, eo, seg


@register("depth_echo")
class DepthEcho(MechWord):
    description = ("The outline is drawn by arriving strands, then copies of it peel off the face one by one and fan out behind "
                   "it at true depth, a stack of paper-filled outlines fading into the ground that breathes and swells in "
                   "perspective. On exit the stack folds back into the face and the strands run out.")
    DEMO_WORDS = "ECHO"
    DEFAULT_FONT = ("rubik", {"wght": 900})
    DEFAULT_PALETTE = "ink"
    DEFAULT_TIMING = (2.0, 0.9, 1.8)
    MS = 0.22
    LIB_CAMERA = dict(yaw=(26.0, -15.0, -26.0), pitch=(12.0, 10.0, 8.0), dist=(1200.0, 1180.0, 1220.0), cz=200.0)

    def mech_build(self):
        P, pl = self.params, self.placement
        self.NC = int(P.get("copies", 14))
        self.DZ = float(P.get("dz") or 0.042 * max(pl.w, pl.h))
        self.swell = float(P.get("swell", 0.30))            # log-scale of the deepest copy swings +-swell (q5: perspective growth)
        self.drift = float(P.get("drift", 0.05)) * pl.w      # lateral throw of the deepest copy (q5: the echo axis turns)
        self.cons3 = self.cons2()
        self.cen = np.zeros(2)

    def copy_params(self, t, p):
        """per copy k=1..NC: release f_k (0 at the face -> 1 in its slot), z, scale, shift."""
        NC = self.NC
        k = np.arange(1, NC + 1)
        a = 0.52 * (NC - k) / max(NC - 1, 1)                 # the deepest copy leaves first, the next one nests in front of it
        f = np.array([eio(seg(p, a_, a_ + 0.48), 2.2) for a_ in a])
        u = k / NC
        breath = 1.0 + 0.16 * np.sin(2 * math.pi * (t / 2.1 - 0.8 * u))
        z = k * self.DZ * f * breath
        g = self.swell * math.sin(2 * math.pi * (t / 3.4) + 0.6)
        sc = np.exp(g * u * f)
        ang = 2.3 + 0.7 * math.sin(2 * math.pi * t / 5.0)
        sh = np.stack([math.cos(ang) * self.drift * u * f, math.sin(ang) * self.drift * 0.6 * u * f], 1)
        return f, z, sc, sh

    def mech_layers(self, t):
        p = self.progress(t)
        pal = self.palette
        out = []
        if p > 1e-4:
            f, z, sc, sh = self.copy_params(t, p)
            for j in range(self.NC - 1, -1, -1):             # far first
                if f[j] < 1e-3:
                    continue
                u = (j + 1) / self.NC
                polys = [self.lift(c, z[j], sc[j], sh[j]) for c in self.cons3]
                tone = mixc(pal.fg, pal.bg, 0.04 + 0.78 * u ** 0.9)
                tone = mixc(pal.fg, tone, ss(f[j] * 3.0))
                out.append(Layer(polys, pal.bg, "fill", bias=0.0))
                out.append(Layer(polys, tone, "line", 1.3, closed=True))
        # the face: solid word with its accent outline
        fa = self.face_alpha(t)
        face = [self.lift(c) for c in self.cons3]
        if fa > 0.004:
            out.append(Layer(face, pal.fg, "fill", alpha=fa, bias=1.0))
        acc = pal.accents[0]
        ed = ss(p / 0.15)
        out.append(Layer(face, mixc(pal.fg, acc, ed), "line", self.w * (1 + 0.6 * ed), closed=True, bias=2.0)) if p > 1e-4 else None
        return out
