"""wipe: a directional mask sweeps across the formed word along the entry direction (reveal) or the exit direction (conceal).
Nothing is carried, so the material is `none` (a transition next to a wipe becomes a cut). Options:
  soft   width of the soft edge as a fraction of the word's extent along the sweep (0 = hard edge), default 0.16
  edge   draw a hairline scan line on the edge (default True)
  steps  alpha steps in the soft edge (default 7)"""
import numpy as np
from mograph.modules.core import Layer
from mograph.modules.material import Material
from mograph.modules.io.base import IOStyle, register_style, side_frame, clip_layer, fade, ss


def _keep_le(layers, d, F):
    return [q for q in (clip_layer(L, -d, d * F) for L in layers) if q is not None]


def _keep_ge(layers, d, F):
    return [q for q in (clip_layer(L, d, d * F) for L in layers) if q is not None]


@register_style("wipe")
class Wipe(IOStyle):
    description = ("A directional mask sweeps over the finished word along the entry direction to reveal it, and along the "
                   "exit direction to wipe it away, with a soft or hard edge and an optional scan line.")
    material_kind = "none"

    def attach(self, body, side):
        super().attach(body, side)
        self.d, self.e1, self.e2, self.gate = side_frame(body, side)
        pl = body.placement
        self.E = pl.extent_along(self.d) * 1.08 + 6.0
        self.s0 = float(pl.origin @ self.d)

    def base(self, t):
        b = self.body
        return b.reveal_layers(0.0, t) if self.side == "in" else b.conceal_layers(1.0, t)

    def layers(self, t, p):
        d, E, s0 = self.d, self.E, self.s0
        soft = float(self.opt("soft", 0.16)) * 2 * E
        steps = int(self.opt("steps", 7))
        eps = 1e-6
        if self.side == "in":
            if p <= eps:
                return []
            if p >= 1.0 - eps:
                return self.base(t)
        else:
            if p >= 1.0 - eps:
                return []
            if p <= eps:
                return self.base(t)
        q = 0.5 * (p + ss(p))
        reveal = self.side == "in"
        f = s0 - E - (0.0 if reveal else soft) + (2 * E + soft) * q       # leading edge of the soft band along d (absolute)
        base = self.base(t)
        bw = soft / steps
        out = []
        if reveal:                                  # visible: behind the edge (smaller s); band [f - soft, f] fades out
            out += _keep_le(base, d, f - soft)
            for k in range(steps if soft > 0 else 0):
                lo, hi = f - soft + k * bw, f - soft + (k + 1) * bw
                out += fade(_keep_ge(_keep_le(base, d, hi), d, lo), 1.0 - (k + 0.5) / steps)
            ex = f - 0.5 * soft
        else:                                       # visible: ahead of the edge (larger s); band [f, f + soft] fades in
            out += _keep_ge(base, d, f + soft)
            for k in range(steps if soft > 0 else 0):
                lo, hi = f + k * bw, f + (k + 1) * bw
                out += fade(_keep_ge(_keep_le(base, d, hi), d, lo), (k + 0.5) / steps)
            ex = f + 0.5 * soft
        if self.opt("edge", True):
            pl = self.body.placement
            half = pl.extent_along(self.e1) * 1.02
            c = pl.origin + d * (ex - s0)
            out.append(Layer([np.array([c - self.e1 * half, c + self.e1 * half])], self.body.palette.fg, "line", 1.6,
                             closed=False, bias=0.5, alpha=min(1.0, 4 * min(p, 1 - p) + 0.25)))
        return out

    def material(self):
        pl = self.body.placement
        sgn = -1.0 if self.side == "in" else 1.0
        return Material.none(pl.origin + self.d * sgn * (self.gate), self.d)
