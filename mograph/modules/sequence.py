"""Sequence: [word module, transition, word module, ...] -> one timeline of cam3d Layers -> frames -> mp4.

Transitions are built from their neighbours' bundles automatically (bind): `Sequence([A, Twist(turns=2), B])`; two word
modules side by side get the default transition ('twist'). Items are played back to back; absolute times in `.timeline`.
Camera: None / 'demo' (fixed front camera framing every placement), a cam3d Camera, a FlowPath / CamPath (anything with
.at(t)), `seq.follow_camera()` (a FlowPath that follows the strands and rests on each formed word), or (phase 3b) 'auto' /
a mograph.modules.camera.CameraChain (camera modules joined with C2 continuity)."""
import os, math, pickle, subprocess, copy
import numpy as np
import cv2
from mograph.modules.core import (WordModule, TransitionModule, Palette, get_module, unit)
from mograph.render import cam3d
from mograph.render.cam3d import Camera, FlowPath

# 2026-11-01 (user: "the library transitions ... should be deforming from the previous words ... then into the next text"): in a gap whose transition is
# one of the approved LIBRARY modules, word A's exit and word B's entry become the glyph-outline hand-off (mograph/modules/io/outline.py): the
# module's own strands peel off A's outlines, run its unchanged route / pattern and land on B's outlines. The old hand-off stays reachable:
# env MOTNGRFX_HANDOFF=gate (or HANDOFF = "gate"), or a transition param handoff="gate".
OUTLINE_KINDS = ("braid", "circuit_board", "fan", "spiral_around", "twist", "ripple")
OUTLINE_EXIT_S = 0.35      # A's exit window with the outline hand-off (fill -> outline); the rest of its old exit goes to the transition
OUTLINE_FORM_S = 0.55      # B's form window (the strands ease onto the outline, the fill comes in); the rest of its old form goes to the transition
                           # (R16: the engine had left the library module 0.4 s between a 0.75 s exit and a 2.6 s form-in; hits / releases do not move)
HANDOFF = os.environ.get("MOTNGRFX_HANDOFF", "outline").strip().lower()


def outline_handoff(tr) -> bool:
    """does this transition take the glyph-outline hand-off? (a library kind, the outline hand-off not switched off globally or on the transition)."""
    mode = str((getattr(tr, "params", None) or {}).get("handoff") or HANDOFF).lower()
    return mode == "outline" and (getattr(tr, "name", "") or "") in OUTLINE_KINDS
ol = cam3d.ol


class FixedCamera:
    def __init__(self, cam):
        self.cam = cam

    def at(self, t):
        return self.cam


class FocusLook:
    """picklable look-at target for a FlowPath: the sequence's focus point (strand heads / formed word), tabulated at 60 Hz."""

    def __init__(self, seq):
        self.seq = seq
        self.ts = np.arange(0.0, seq.duration + seq.tail + 0.5, 1 / 60.0)
        self.xs = np.array([seq.focus(float(t)) for t in self.ts])

    def __call__(self, t):
        t = min(max(t, self.ts[0]), self.ts[-1])
        return np.array([np.interp(t, self.ts, self.xs[:, k]) for k in range(3)])


class Sequence:
    def __init__(self, items, bg=None, default_transition="twist", tail=0.0, bare=False):
        """bare=True: a single, already bound TransitionModule on its own (library clip of a transition)."""
        self.tail = float(tail)           # seconds the last module keeps coasting after its exit (render / camera horizon)
        self.items = []
        self.bare = bare
        if bare:
            self.items = list(items)
            self.timeline, t = [], 0.0
            for it in self.items:
                self.timeline.append((it, t, t + it.duration))
                t += it.duration
            self._dur, self.words = t, []
            self.bg = Palette.get(bg).bg
            return
        prev_word = False
        for it in items:
            if isinstance(it, type):
                it = it()
            if isinstance(it, WordModule):
                if prev_word:
                    self.items.append(get_module(default_transition)())
                self.items.append(it)
                prev_word = True
            elif isinstance(it, TransitionModule):
                self.items.append(it)
                prev_word = False
            else:
                raise TypeError(f"not a module: {it!r}")
        assert isinstance(self.items[0], WordModule) and isinstance(self.items[-1], WordModule), \
            "a sequence starts and ends with a word module"
        self.notes = []                   # phase 3a: transitions replaced by a cut because a style carries no material
        for it in self.items:             # a word an earlier Sequence handed off by its outline (spec.py: Sequence.auto first) starts clean
            if isinstance(it, WordModule) and hasattr(it, "handoff_reset"):
                it.handoff_reset()
        for i, it in enumerate(self.items):
            if isinstance(it, TransitionModule):
                a_w, b_w = self.items[i - 1], self.items[i + 1]
                if outline_handoff(it):                                              # 2026-11-01: the library transition peels off A's glyph outlines and lands on B's
                    a_w.set_style("out", "outline")
                    b_w.set_style("in", "outline")
                    gain = a_w.retime(exit=OUTLINE_EXIT_S) + b_w.retime(form=OUTLINE_FORM_S)     # the library module gets the time: A's release and B's hit stay where they were
                    if gain > 0 and it.dur is not None:
                        it.dur = float(it.dur) + gain
                    elif gain > 0:
                        it._gain = gain                                              # an auto-sized transition: added after bind
                    it.params.setdefault("unravel", True)                            # the modules' unravel look: separate fine flat strands (braid plaits, ripple flat rings)
                    for k_ in ("ghost", "pads", "web"):                              # no proxy geometry (spiral drum wireframe, circuit pads, fan web): only the strands
                        it.params.setdefault(k_, False)
                if (a_w.exit_material().kind == "none" or b_w.entry_material().kind == "none") and it.name != "cut":
                    self.notes.append(f"{it.name or type(it).__name__} between {a_w.name} and {b_w.name} -> cut "
                                      f"(exit {a_w.exit_material().kind} / entry {b_w.entry_material().kind})")
                    it = self.items[i] = get_module("cut")(replaced=it.name)
                it.a_word, it.b_word = a_w, b_w                                      # neighbours (transitions may use their placement)
                it.bind(a_w.exit_bundle(), b_w.entry_bundle())
                if getattr(it, "_gain", 0.0):
                    it.dur, it._gain = it.dur + it._gain, 0.0
                    it.build()
        self.timeline, t = [], 0.0
        for it in self.items:
            self.timeline.append((it, t, t + it.duration))
            t += it.duration
        self._dur = t
        self.words = [it for it in self.items if isinstance(it, WordModule)]
        pal = Palette.get(bg) if bg is not None else self.words[0].palette
        self.bg = pal.bg

    @classmethod
    def auto(cls, items, seed=0, energy=0.5, bpm=None, mood=None, duration=None, **kw):
        """phase 4: a chain whose transitions the engine picks (mograph.modules.engine): `Sequence.auto([w0, w1, w2], seed=3,
        energy=0.4, bpm=96)`. Explicit transitions between the words are kept. .engine / .gaps hold the choices / gap readouts."""
        from mograph.modules.engine import build_sequence
        return build_sequence(items, seed=seed, energy=energy, bpm=bpm, mood=mood, duration=duration, **kw)

    @property
    def duration(self):
        return self._dur

    def item_at(self, t):
        for it, a, b in self.timeline:
            if t < b:
                return it, t - a
        it, a, b = self.timeline[-1]
        return it, t - a

    def layers(self, t):
        it, lt = self.item_at(t)
        return it.layers(max(lt, 0.0))

    def focus(self, t):
        it, lt = self.item_at(t)
        return it.focus(max(lt, 0.0))

    # ------------------------------------------------------------ cameras
    def demo_camera(self, margin=1.12, F=1000.0):
        """fixed front camera (looking along +z) that frames every word placement and the transition axes; never closer than
        the home camera, so a single centred placement is framed exactly like the 960x540 frame."""
        if self.bare:
            return Camera.home(F)
        pts = [p for w in self.words for p in w.placement.corners()]
        for it in self.items:
            if isinstance(it, TransitionModule):
                pts += list(it.bounds_points()[::3])
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        c = 0.5 * (lo + hi)
        zc = float(pts[:, 2].mean())
        dist = max(F, F * 0.5 * (hi - lo)[0] * margin / (cam3d.W / 2.0), F * 0.5 * (hi - lo)[1] * margin / (cam3d.H / 2.0))
        for _ in range(8):                      # perspective: nearer parts of tilted planes grow, so refit by projection
            cam = Camera((c[0], c[1], zc - dist), (c[0], c[1], zc), 0.0, F)
            xy, z = cam.project(pts)
            r = max(np.abs(xy[:, 0] - cam3d.W / 2.0).max() / (cam3d.W / 2.0), np.abs(xy[:, 1] - cam3d.H / 2.0).max() / (cam3d.H / 2.0)) * margin
            if r <= 1.0 + 1e-3 and dist <= F + 1e-6:
                break
            dist = max(F, dist * r) if r > 1.0 + 1e-3 else max(F, dist * r)
            if abs(r - 1.0) < 2e-3:
                break
        return Camera((c[0], c[1], zc - dist), (c[0], c[1], zc), 0.0, F)

    def follow_camera(self, F=1000.0, fill=0.7, pull=1.3, ease_in=0.5, ease_out=0.7, hold_ease=0.45):
        """FlowPath that follows the strands: rests face-on in front of every formed word (holds), travels between them
        through the transition (eye behind the cable, looking at the strand heads), starts and ends pulled back. The look-at
        target is `focus(t)` (strand heads while moving, the word while it stands)."""
        T = self.duration + self.tail
        pts, holds, back = [], [], []
        words = self.words
        poses = []
        for w in words:
            pl = w.placement
            fl = fill * cam3d.FIT_FILLK
            dist = F * max(pl.w / (fl * cam3d.W), pl.h / (fl * cam3d.FIT_HK * cam3d.H))
            poses.append(w.hold_pose(dist))
        e0, t0_ = poses[0]
        pts.append(e0 + (e0 - t0_) * (pull - 1.0) * 0.8)
        for k, w in enumerate(words):
            tw = [a for it, a, b in self.timeline if it is w][0]
            tm = w.timing
            tc = tw + tm.form + tm.hold / 2.0
            dur = max(0.0, tm.hold * 0.8)
            pts.append(poses[k][0])
            if dur > 0.25:
                holds.append((len(pts) - 1, tc, dur))
            if k < len(words) - 1:
                tr = [it for it, a, b in self.timeline if isinstance(it, TransitionModule) and a >= tw + w.duration - 1e-6][0]
                f = tr.focus(tr.duration * 0.5)
                n_avg = unit(words[k].placement.n + words[k + 1].placement.n)
                dist = 0.5 * (np.linalg.norm(poses[k][0] - poses[k][1]) + np.linalg.norm(poses[k + 1][0] - poses[k + 1][1]))
                pts.append(f - n_avg * dist * pull)
        el, tl_ = poses[-1]
        pts.append(el + (el - tl_) * (pull - 1.0) * 0.8)
        look = FocusLook(self)
        path = FlowPath(np.array(pts), 0.0, T, ease_in=ease_in, ease_out=ease_out, look=look, F=F, holds=holds,
                        hold_ease=hold_ease, look_lag=0.2, drift_px=0.0)
        path.at(0.0)             # build the look tracker table now (pickled with the path)
        return path

    def camera_hints(self, F=1000.0):
        """what the active modules tell a camera module (word placements / hold poses / counters, transition bounds, strand
        focus): mograph.modules.camera.Hints. Additive (phase 3b)."""
        from mograph.modules.camera.base import Hints
        return Hints(self, F)

    def _camera(self, camera):
        if isinstance(camera, str) and camera == "auto":              # phase 3b: CameraChain.auto (follow + hold / orbit rules)
            from mograph.modules.camera import CameraChain
            return CameraChain.auto(self)
        if camera is None or (isinstance(camera, str) and camera == "demo"):
            return FixedCamera(self.demo_camera())
        if isinstance(camera, Camera):
            return FixedCamera(camera)
        if isinstance(camera, str) and camera == "follow":
            return self.follow_camera()
        return camera

    # ------------------------------------------------------------ frames
    def frame(self, t, camera=None, size=(960, 540), S=2):
        """one RGB uint8 frame at time t."""
        cam = self._camera(camera).at(t)
        return _draw(self.layers(t), cam, self.bg, size, S)

    def render(self, path, camera=None, fps=30, size=(960, 540), S=2, audio=None, procs=8, sub=3, shutter=0.5, tail=None,
               world=None):
        """render an mp4 (never overwrites). camera: see module doc. sub / shutter: motion-blur sub-frames and shutter
        (fraction of a frame). tail: seconds after the end (default self.tail) so the last strands finish leaving.
        audio: optional audio file muxed in. world: (w, h) frame plane, e.g. (540, 960) for a 9:16 render (default: whatever
        cam3d.W / H are, 960x540); the worker processes set it too. -> (seconds, frame count)."""
        if world is not None:
            cam3d.set_world(*world)
        cam = self._camera(camera)
        tail = self.tail if tail is None else float(tail)
        nf = int(round((self.duration + tail) * fps))
        job = dict(seq=self, cam=cam, fps=fps, size=tuple(size), S=S, sub=sub, shutter=shutter, nf=nf, world=(cam3d.W, cam3d.H))
        out = path if not audio else path[:-4] + "_silent.mp4"
        try:
            blob = pickle.dumps(job)
        except Exception as e:                                   # e.g. a camera with lambdas: render serially
            print("render: job not picklable (%s); serial render" % e)
            blob, procs = job, 1
        if procs <= 1:
            _init(blob)
        secs = ol.render(_frame, out, nf, size[0], size[1], fps, procs, init=_init, initargs=(blob,))
        if audio:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", out, "-i", audio, "-c:v", "copy", "-c:a", "aac",
                            "-shortest", path], check=True)
            os.remove(out)
        return secs, nf


# ------------------------------------------------------------------ frame workers (module level so a process pool can pickle them)
_JOB = {}


def _init(blob):
    _JOB.clear()
    _JOB.update(pickle.loads(blob) if isinstance(blob, (bytes, bytearray)) else blob)
    if _JOB.get("world"):
        cam3d.set_world(*_JOB["world"])


def _draw(layers, cam, bg, size, S):
    k = size[0] / float(cam3d.W)
    if abs(k - 1.0) > 1e-6:
        cam = copy.copy(cam)
        cam.F = cam.F * k
    im = ol.canvas(size[0], size[1], S, bg)
    cam3d.render_layers(im, layers, cam, S)
    return ol.down(im, S)


def _frame(i):
    J = _JOB
    seq, fps, sub, nf = J["seq"], J["fps"], J["sub"], J["nf"]

    def at(f):
        t = f / fps
        return _draw(seq.layers(t), J["cam"].at(t), seq.bg, J["size"], J["S"]).astype(np.float32)
    if sub <= 1:
        return np.ascontiguousarray(at(float(i)).astype(np.uint8))
    acc = 0
    for k in range(sub):
        f = min(max(i + J["shutter"] * ((k + 0.5) / sub - 0.5), 0.0), nf - 1.0)
        acc = acc + at(f)
    return (acc / sub + 0.5).astype(np.uint8)
