"""fly_into: the letters are tunnels cut into a slab and the stripes stream through them. A train of outline rings flows in
along the entry axis (the shaft), the rings that are inside the shaft are filled as cobalt walls darkening with depth with a
thin light stripe at every ring; the stripes keep streaming through the opening and out toward the exit side, growing by
perspective, until the train has passed and the tunnels close. The rings are the strands: the entry bundle is the first rings
at the entry gate, the exit bundle the last rings at the exit gate. Entry 'in' / exit 'out' give the straight tunnel of the
original piece; camera_hint(t) is the camera that flies down the shaft. (mechanism: 11_mathtype b5R INTO, stripes streaming
past, as the forming mechanic)"""
import numpy as np
from mograph.modules.core import register, Palette, Layer, mixc
from mograph.modules.words._rings import RingStream
from mograph.modules.words._cb import ss

BONE, INK, COBALT = (236, 230, 218), (12, 12, 11), (70, 110, 240)
RING = (200, 208, 240)
WALL = [(0.0, (150, 172, 246)), (0.25, COBALT), (0.65, (18, 26, 92)), (1.0, (8, 9, 22))]


def wall(u):
    u = min(1.0, max(0.0, u))
    for (a, ca), (b, cb) in zip(WALL[:-1], WALL[1:]):
        if u <= b:
            return mixc(ca, cb, (u - a) / (b - a))
    return WALL[-1][1]


@register("fly_into")
class FlyInto(RingStream):
    description = ("The letters are tunnels in a bone slab. Rings flow in along the entry axis; the ones inside the shaft are "
                   "filled as cobalt walls darkening with depth with a light stripe at every ring, and the stripes keep "
                   "streaming out through the opening until the tunnels close.")
    DEMO_WORDS = "INTO"
    DEFAULT_FONT = "bungee"
    NR = 20
    TILT = 0.9
    DEFAULT_PALETTE = Palette(bg=BONE, fg=INK, accents=(COBALT, (45, 75, 216), (200, 130, 40)))

    def __init__(self, words=None, font=None, placement=None, entry="left", exit="right", timing=None, palette=None, **params):
        super().__init__(words, font, placement, entry, exit, timing, palette if palette is not None else self.DEFAULT_PALETTE, **params)

    def ring_colours(self, sig):
        pal = self.palette
        cols = []
        for s in sig:
            if s > 0:
                cols.append(mixc(COBALT, pal.bg, min(1.0, s / self.Lx) * 0.75))
            else:
                u = min(1.0, -s / self.Lt)
                cols.append(mixc(RING, wall(u), min(1.0, 0.25 + 0.75 * u)))
        return np.array(cols, float)

    def lip_colour(self):
        return self.palette.fg

    def fill_layers(self, t, sig, off):
        pl = self.placement
        out = []
        for r in range(self.NR):
            s = sig[r]
            if s >= 0.1 * self.Lt or s <= -self.Lt:
                continue
            u = min(1.0, max(0.0, -s / self.Lt))
            a = ss((s + self.Lt) / (0.2 * self.Lt)) * (1.0 - ss(s / (0.1 * self.Lt)))
            if a <= 0.004:
                continue
            polys = [c3 + pl.origin + off[r] for c3 in self.cons3]
            out.append(Layer(polys, wall(u), "fill", alpha=a, bias=0.0))
        return out
