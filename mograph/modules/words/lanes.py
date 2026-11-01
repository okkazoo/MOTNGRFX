"""lanes: parallel hairline lanes stream in from the entry side as a ribbon, peel off one by one and draw the outline of
the type; the word fills solid. To leave, the fill melts back to outline, every arc detaches in a sweep and runs out
along its own lane toward the exit side. (mechanism: 11_mathtype q1R_lanes, rewritten for the Bundle interface)"""
from mograph.modules.core import register
from mograph.modules.words._strand import StrandWord


@register("lanes")
class Lanes(StrandWord):
    description = ("Hairline lanes stream in as a tight ribbon, peel off one by one to draw the outline of the type, and the "
                   "word fills solid. It leaves the same way in reverse: the fill melts to outline and every line runs "
                   "out along its own lane.")
    DEMO_WORDS = "LANES"
    DEFAULT_FONT = "anybody"
    TUBE = 0.0
    FILL = "solid"
    ARC_PER_CAP = 0.42
    LINE_W = 1.5
