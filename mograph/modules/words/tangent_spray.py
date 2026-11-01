"""tangent_spray: a high-contrast serif word flat to the camera on a vermilion field, with a stack of copies of its outline
receding on a shallow oblique. The contours run in (and out) along their EXACT edge tangents, never snapped to a grid, so
every curve sprays lines in its own direction: a radial burst instead of parallel hatching. Strands arrive down the entry
lanes, swing onto their tangent line and draw their arc, back copies first; the front copy fills solid. To leave, the arcs run
off along their own tangents, back copies first, then bend into the exit lanes.
(mechanism: 11_mathtype a3 TANGENT SPRAY, rewritten for the Bundle interface)"""
from mograph.modules.core import register, Palette, mixc
from mograph.modules.words._stack import StackWord

VERM_GROUND, SHADOW, INK = (214, 72, 44), (120, 34, 22), (12, 12, 11)


@register("tangent_spray")
class TangentSpray(StackWord):
    description = ("A serif word on vermilion made of stacked outline copies. Strands arrive along the entry lanes and run "
                   "in along the outline's exact tangents, so every curve sprays its own lines, then draw the arcs; to "
                   "leave they run off along those tangents and bend into the exit lanes.")
    DEMO_WORDS = "BURST"
    DEFAULT_FONT = "dmserifdisplay"
    NL = 8
    SP = 7.0
    DIR2 = (0.94, 0.34)
    DZ = 5.0
    SPRAY = 0.9
    SNAP = False
    DEFAULT_PALETTE = Palette(bg=VERM_GROUND, fg=INK, accents=((242, 166, 90), (70, 110, 240), (210, 70, 42)))

    def layer_col(self, l):
        return self.palette.fg if l == 0 else mixc(self.palette.fg, SHADOW, l / max(self.NLd - 1, 1))
