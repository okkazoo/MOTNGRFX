"""spectrum: the word is a stack of ten outline copies receding up and to the right (iso), every depth its own colour
(bone front, amber, vermilion, plum, cobalt at the back). Strands arrive down the entry lanes and run in along the outline's
own iso-snapped tangents, back copies first, so the arrival is colour-banded; the front copy fills solid. To leave, the fill
melts and every outline strand runs off along its own edge vector into the exit lanes, back band first.
(mechanism: 11_mathtype a1 SPECTRUM, outline-vector driven colour lines, rewritten for the Bundle interface)"""
import math
from mograph.modules.core import register, mixc
from mograph.modules.words._stack import StackWord, ramp


@register("spectrum")
class Spectrum(StackWord):
    description = ("The word is a stack of outline copies, each depth its own colour from bone to cobalt. Colour-banded "
                   "strands arrive along the entry lanes, back copies first, and run in along the outline's own tangents to "
                   "draw it; to leave, every line runs off along its own edge vector, back band first.")
    DEMO_WORDS = "SHIFT"
    DEFAULT_FONT = "anton"
    NL = 10
    SP = 9.0
    DIR2 = (math.cos(math.radians(30)), -0.5)
    DZ = 6.0
    SPRAY = 0.55
    SNAP = True

    def ramp_cols(self):
        pal = self.palette
        return [pal.fg, pal.accents[0], pal.accents[2], (150, 60, 120), pal.accents[1]]

    def layer_col(self, l):
        return ramp(self.ramp_cols(), l / max(self.NLd - 1, 1))
