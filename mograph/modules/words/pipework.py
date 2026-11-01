"""pipework: shaded tubes (rim, coloured body, highlight) arrive packed side by side like a cable harness, jog with
chamfered right-angle bends and plumb themselves into the outline of the type; the word stays as tube letters over a
faint panel. On exit the tubes unplug and bend away down the exit lanes. (mechanism: 11_mathtype a5R, rewritten)"""
from mograph.modules.core import register
from mograph.modules.words._strand import StrandWord, manhattan


@register("pipework")
class Pipework(StrandWord):
    description = ("Shaded tubes arrive packed side by side like a cable harness, jog through right-angle bends and plumb "
                   "themselves into the outline of the type, which stays as tube letters. On exit the tubes unplug and "
                   "bend away down their lanes.")
    DEMO_WORDS = "PIPEWORK"
    DEFAULT_FONT = "anton"
    TUBE = 1.0
    FILL = "panel"
    ARC_PER_CAP = 0.95

    def connect(self, p0, t0, p1, t1, ctx):
        side, i, N = ctx["side"], ctx["i"], ctx["N"]
        d, e1, e2 = side["d"], side["e1"], side["e2"]
        shift = float((p1 - p0) @ e1)
        f = self._jog(side["rank"][i], N, shift)
        ch = float(self.params.get("bend") or max(6.0, 2.2 * self.w))
        return manhattan(p0, p1, d, e1, e2, f, ch)
