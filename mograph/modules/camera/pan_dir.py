"""pan (PanDir): the directional camera travel that IS the transition. Words sit one frame-span apart along the travel direction in the
same world plane (spec.build_scene places them with `offsets`); the camera rests face-on on word A (hold), travels in direction D with a
quintic ease (settle-hold-exit, no zoom, no roll, pure translation) and comes to rest face-on on word B exactly when the transition ends.
Nothing else moves: A leaves the frame and B arrives because the camera moved, exactly as world-locked objects do. The footage plate is
panned by the same on-screen displacement (tools/fusion/plate_pan.py, track from the real camera), so type and plate stay one picture.

    PanDir(fill=0.7, dist=None, ease='io')   window = the run of words joined by `pan` transitions (spec.build_camera carves it)

Geometry shared with the planner (no camera needed): `fit_dist` (the eye distance at which a word box fills `fill` of the frame, the
same rule as WordInfo.fit), `frame_span` (world units for one frame along a direction at that distance), `DIR_VEC` / `OPPOSITE`.
Directions are the direction the CAMERA travels (image axes, y down): down = the camera moves down the world, so the picture slides UP
on screen (A exits at the top, B enters from the bottom); up / right / left likewise."""
import numpy as np

from mograph.modules.camera.base import CameraModule, register_camera, smoother, lerp
from mograph.render import cam3d

DIRS = ("down", "up", "right", "left")
DIR_VEC = {"down": (0.0, 1.0), "up": (0.0, -1.0), "right": (1.0, 0.0), "left": (-1.0, 0.0)}      # camera travel in image axes (x right, y down)
OPPOSITE = {"down": "up", "up": "down", "right": "left", "left": "right"}
FILL = 0.7                                   # the word fills this much of the frame at its hold (same default as hold_drift / follow)


def fit_dist(w, h, F=1000.0, fill=FILL):
    """eye distance (world units) at which a w x h word box fills `fill` of the frame: WordInfo.fit written without a WordInfo."""
    fill = fill * cam3d.FIT_FILLK
    return F * max(w / (fill * cam3d.W), h / (fill * cam3d.FIT_HK * cam3d.H))


def frame_span(direction, dist, F=1000.0, world=None):
    """world units one full frame covers along `direction` at eye distance `dist` (the offset of the next word for a 'frame-span further')."""
    W, H = world if world is not None else (cam3d.W, cam3d.H)
    vx, vy = DIR_VEC[direction]
    return abs(vx) * W * dist / F + abs(vy) * H * dist / F


def profile(u, ease="io"):
    """travel progress 0..1 over the transition: 'io' quintic smootherstep (C2, rest at both ends), 'soft' a longer settle (the arrival is
    slower than the departure)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    if ease == "soft":
        return smoother(u ** 0.85)
    return smoother(u)


@register_camera("pan")
class PanDir(CameraModule):
    description = ("The camera travels in one direction (down / up / right / left) from word A to word B, which sits one frame-span "
                   "further along in the same world plane: pure translation, quintic ease, rest on both words. A leaves the frame on the "
                   "opposite side and B arrives as the camera settles, exactly as world-locked objects do.")
    DEFAULTS = dict(fill=FILL, dist=None, ease="io")

    def _words(self, hints, t0, t1):
        ws = [w for w in hints.words if w.a >= t0 - 1e-6 and w.b <= t1 + 1e-6]
        return ws or [hints.word_at(0.5 * (t0 + t1))]

    def dist_of(self, words):
        if self.p("dist") is not None:
            return float(self.p("dist"))
        return max(w.fit(self.p("fill")) for w in words)

    def _poses(self, words, D):
        return [(w.origin - w.vn * D, w.origin.copy()) for w in words]

    def curves(self, hints, ts, t0, t1):
        words = self._words(hints, t0, t1)
        D = self.dist_of(words)
        poses = self._poses(words, D)
        eye = np.tile(poses[0][0], (len(ts), 1))
        look = np.tile(poses[0][1], (len(ts), 1))
        for tr in hints.trans:
            if tr.wa not in words or tr.wb not in words:
                continue
            k = words.index(tr.wa)
            (ea, la), (eb, lb) = poses[k], poses[k + 1]
            s = profile((ts - tr.a) / max(tr.b - tr.a, 1e-9), self.p("ease"))
            m = (ts >= tr.a) & (ts < tr.b)
            done = ts >= tr.b
            eye[done], look[done] = eb, lb
            if m.any():
                eye[m], look[m] = lerp(ea, eb, s[m]), lerp(la, lb, s[m])
        return dict(eye=eye, look=look)

    def declare(self, hints, t0, t1):
        words = self._words(hints, t0, t1)
        rests, start = [], t0
        for tr in hints.trans:
            if tr.wa in words and tr.wb in words:
                rests.append((start, tr.a))
                start = tr.b
        rests.append((start, t1))
        return [r for r in rests if r[1] > r[0] + 0.05], []
