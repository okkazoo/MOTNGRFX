"""pan: the camera travel as a transition. Nothing in this module moves: it draws word A (frozen on its hold) and word B (already
formed) in the world, where B sits one frame-span further along the travel direction (spec.build_scene lays the words out; see
mograph/modules/camera/pan_dir.py), and the PanDir camera does ALL the motion. Both words are world-locked, so A leaves the frame on the
side opposite to the travel and B arrives as the camera settles. It composes with ANY word module (the module's own geometry is drawn
at its hold time) and with the io styles at the frame edge:

    Pan(dir='down', beats=2, ease='io', exit_io=False, entry_io=False, gap=0.0)

dir       camera travel direction: down | up | right | left (the picture slides the other way)
beats     travel length in beats (the planner turns it into seconds: dur = beats x beat interval; `dur=` overrides)
exit_io   A's form-out style (blobs / particles / lines ...) plays while A leaves, mapped on the travel (default: A stays whole)
entry_io  B's form-in style plays while B arrives
gap       extra air between the two frames, as a share of one frame-span (placement only; spec.build_scene reads it)
exit_win / entry_win  (a, b) share of the travel the exit / entry io style plays on (defaults (0.35, 1.0) / (0.0, 0.65); set as attributes
          after spec.build_scene, e.g. a long travel with a dead gap whose B form-in should only play as B comes into view)

The words must be placed by spec.build_scene (or by hand: B.origin = A.origin + travel direction x frame_span). The neighbouring words
are flagged `carry`, so their own form / exit windows stay frozen on the hold: the sequence timing is hold -> travel -> hold."""
import numpy as np
from mograph.modules.core import TransitionModule, register
from mograph.modules.camera.pan_dir import DIRS, DIR_VEC, FILL, profile


def _smooth(x, a, b):
    u = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@register("pan")
class Pan(TransitionModule):
    description = ("The camera travel IS the transition: A holds, the camera travels down / up / right / left, A leaves the frame on the "
                   "opposite side as a world-locked object would and B (placed one frame-span further along) arrives as the camera "
                   "settles on it. Any word module, any io style at the frame edge.")
    hidden = True            # not in list_modules('transition'): the engine / batch coverage pool is unchanged; the planner places it explicitly
    DEMO_DUR = 1.6
    DEMO_PARAMS = dict(dir="down")

    def __init__(self, a_exit=None, b_entry=None, dur=None, **params):
        d = params.get("dir", "down")
        if d not in DIRS:
            raise ValueError(f"pan dir {d!r}: one of {DIRS}")
        self.dir = d
        self.ease = params.get("ease", "io")
        self.exit_io, self.entry_io = bool(params.get("exit_io", False)), bool(params.get("entry_io", False))
        self.fill = float(params.get("fill", FILL))
        self.exit_win = tuple(params.get("exit_win", (0.35, 1.0)))       # share of the travel A's form-out / B's form-in play on (exit_io / entry_io)
        self.entry_win = tuple(params.get("entry_win", (0.0, 0.65)))
        super().__init__(a_exit, b_entry, dur, **params)

    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        return 1.6

    def build(self):
        a, b = self.a_word, self.b_word
        if a is None or b is None:
            raise ValueError("pan needs the neighbouring word modules (use it inside a Sequence)")
        a.carry = dict(a.carry or {}, out=True)
        b.carry = dict(b.carry or {}, **{"in": True})
        ta, tb = a.timing, b.timing
        self.t_a = ta.form + ta.hold                 # module time of A's release
        self.t_b = tb.form                           # module time of B's hit
        self.va = np.asarray(DIR_VEC[self.dir], float)

    def _p(self, t):
        return float(profile(t / max(self.dur, 1e-9), self.ease))

    def layers(self, t):
        u = min(max(t / max(self.dur, 1e-9), 0.0), 1.0)
        a, b = self.a_word, self.b_word
        # A: frozen on its hold, or its form-out style mapped on the second half of the travel (edge io)
        qa = float(_smooth(u, *self.exit_win)) if self.exit_io else 0.0
        ta = self.t_a + qa * a.timing.exit
        # B: formed, or its form-in style mapped on the first part of the travel
        qb = float(_smooth(u, *self.entry_win)) if self.entry_io else 1.0
        tb = qb * b.timing.form
        la = a.layers_uncarried(ta) if self.exit_io else a.layers_uncarried(self.t_a)
        lb = b.layers_uncarried(tb) if self.entry_io else b.layers_uncarried(self.t_b)
        return list(la) + list(lb)

    def focus(self, t):
        s = self._p(t)
        return self.a_word.placement.origin * (1.0 - s) + self.b_word.placement.origin * s

    def bounds_points(self):
        return np.concatenate([self.a_word.placement.corners(), self.b_word.placement.corners()])
