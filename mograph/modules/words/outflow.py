"""outflow: the word is a shaft that rings flow out of. A train of outline rings streams in along the entry axis, small and
dim at the entry gate (graduated: bone fading to dark with depth), passes through the word's place and out toward the exit
side as a growing echo; while it passes, the word stands as a bone lip outline with the rings flowing through it and the flow
never stops. When the last ring has passed the lip fades and the train leaves down the exit axis. The rings are the strands:
the entry bundle is the first rings at the entry gate, the exit bundle the last rings at the exit gate. Real 3D: the shaft is
the entry side tilted into the depth (tilt), so a camera sees the rings recede / grow; entry 'in' and exit 'out' give the
straight tunnel of the original piece. (mechanism: 11_mathtype q8 OUTFLOW as vector streams)"""
import numpy as np
from mograph.modules.core import register, mixc
from mograph.modules.words._rings import RingStream
from mograph.modules.words._cb import ss

DARK = (40, 39, 36)


@register("outflow")
class Outflow(RingStream):
    description = ("A train of outline rings streams in along the entry lanes, graduated from dim at the gate to bone at the "
                   "word, flows through the letters and out as a growing echo; the word stands as the lip the rings flow "
                   "through, and the train leaves down the exit lanes.")
    DEMO_WORDS = "OUT"
    DEFAULT_FONT = "ericaone"
    NR = 12
    TILT = 1.4

    def ring_colours(self, sig):
        pal = self.palette
        u_in = np.clip(-sig / self.Lt, 0.0, 1.0)
        u_out = np.clip(sig / self.Lx, 0.0, 1.0)
        dark = mixc(pal.bg, pal.fg, 0.22)
        cols = []
        for a, b in zip(u_in, u_out):
            if b > 0:
                cols.append(mixc(pal.fg, pal.bg, 0.8 * b))
            else:
                cols.append(mixc(pal.fg, dark, a ** 0.9))
        return np.array(cols, float)
