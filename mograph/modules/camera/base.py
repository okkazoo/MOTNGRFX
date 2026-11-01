"""mograph.modules.camera.base: camera modules and the chain that joins them.

A CameraModule is one camera MOVE over a time window of a Sequence. Given the sequence, the window [t0, t1] and the camera hints
of the modules that are active there (Hints: word placements + hold poses + counters, transition bounds, the strand focus
point), it returns a Segment: tabulated eye / look / roll / focal-length / right-hint curves (240 Hz) that also run a little
BEFORE t0 and AFTER t1 (the pad), so neighbours can be joined.

    chain = CameraChain(seq, [Follow().at(0, 3.2), OrbitEven(sweep=70).at(3.2, 7.4), ...])     # or CameraChain.auto(seq)
    seq.render("out.mp4", camera=chain)          # chain.at(t) -> cam3d.Camera
    chain.report()                               # speed_report style numbers + the v003 gates

Joining (C2): consecutive segments are cross-faded over a join window (2 x `join` s, centred on the boundary) with the quintic
smootherstep weight, so position, velocity and acceleration stay continuous when the module curves are C2 (they are: closed
forms or Gaussian-smoothed tables). The chain's start / end ease from / to rest by blending with the frozen first / last pose
(same weight, same continuity). A module declares where it is meant to be at rest (`rests`: holds, reversals) and where it
spins fast (`fast`); the report measures everything else against the gates (no stops, heading jump <= 3 deg/frame,
look turn <= 3 deg/frame, roll <= 1.5 deg/frame).
"""
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from mograph.render import cam3d
from mograph.render.cam3d import Camera, smoother, view_dir

GRID = 240.0                         # tabulation rate (Hz); every table lives on multiples of 1/GRID
PAD = 1.0                            # seconds a segment is tabulated beyond its window on each side
CAMERA_REG = {}


def unit(v):
    v = np.asarray(v, float)
    if v.ndim > 1:
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def lerp(a, b, w):
    """a + (b - a) * w. Vectors (3,) or (n,3) with a weight array w (n,) broadcast over the coordinates."""
    w = np.asarray(w, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if w.ndim >= 1 and max(a.ndim, b.ndim) >= 1 and (a.ndim > 1 or b.ndim > 1 or (a.shape[-1] == 3 and w.shape[0] != 3)):
        w = w.reshape(w.shape + (1,) * (max(a.ndim, b.ndim, 1) - (w.ndim - 1) - 1 + (1 if max(a.ndim, b.ndim) == 1 else 0)))
    return a + (b - a) * w


def yaw_pitch(n):
    """(yaw, pitch) in radians of the unit view direction n (cam3d convention: yaw 0 -> +z, pitch > 0 looks down +y)."""
    n = unit(n)
    return math.atan2(n[0], n[2]), math.asin(max(-1.0, min(1.0, n[1])))


def vdir(yaw, pitch):
    """vectorised cam3d.view_dir: (n,) yaw / pitch (rad) -> (n,3) unit view directions."""
    yaw, pitch = np.asarray(yaw, float), np.asarray(pitch, float)
    return np.stack([np.sin(yaw) * np.cos(pitch), np.sin(pitch) * np.ones_like(yaw), np.cos(yaw) * np.cos(pitch)], -1)


def slerp_dir(a, b, w):
    """unit directions a -> b by weight w (array), spherical interpolation. -> (len(w), 3)."""
    a, b = unit(a), unit(b)
    w = np.atleast_1d(np.asarray(w, float))
    c = float(np.clip(a @ b, -1, 1))
    om = math.acos(c)
    if om < 1e-4:
        return np.tile(a, (len(w), 1))
    s = math.sin(om)
    return (np.sin((1 - w) * om)[:, None] * a + np.sin(w * om)[:, None] * b) / s


def smooth_series(fn, ts, sigma):
    """fn(ts_array) -> (n,...) evaluated on an extended grid and Gaussian-smoothed (sigma seconds, zero phase) -> on ts.
    C-infinity, no lag; the extension (3 sigma each side) keeps the ends exact."""
    ext = int(math.ceil(3.0 * sigma * GRID)) + 2
    tt = np.r_[ts[0] - np.arange(ext, 0, -1) / GRID, ts, ts[-1] + np.arange(1, ext + 1) / GRID]
    X = np.asarray(fn(tt), float)
    if sigma <= 0:
        return X[ext:ext + len(ts)]
    Y = gaussian_filter1d(X, sigma * GRID, axis=0, mode="nearest")
    return Y[ext:ext + len(ts)]


def cumtrapz(y, dt):
    return np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)]


def transport_roll(Fv, Rv):
    """frame-to-frame roll of the image about the view axis (degrees per step, unsigned): the image-right vector Rv[i] carried
    to the next view direction by the minimal rotation Fv[i] -> Fv[i+1] (parallel transport), compared with Rv[i+1]. Zero for a
    camera that only tilts / pans without rolling the picture; honest near the vertical, where the horizon-level roll is
    undefined."""
    f0, f1 = Fv[:-1], Fv[1:]
    r0, r1 = Rv[:-1], Rv[1:]
    ax = np.cross(f0, f1)
    sn = np.linalg.norm(ax, axis=1)
    cs = np.clip((f0 * f1).sum(1), -1, 1)
    k = ax / np.maximum(sn, 1e-12)[:, None]
    # Rodrigues: rotate r0 about k by the angle between f0 and f1
    rt = (r0 * cs[:, None] + np.cross(k, r0) * sn[:, None] + k * (k * r0).sum(1)[:, None] * (1 - cs)[:, None])
    rt = np.where((sn > 1e-9)[:, None], rt, r0)
    c = np.clip((rt * r1).sum(1), -1, 1)
    sgn = (np.cross(rt, r1) * f1).sum(1)
    return np.degrees(np.abs(np.arctan2(sgn, c)))


def register_camera(name):
    def deco(cls):
        cls.name = name
        CAMERA_REG[name] = cls
        return cls
    return deco


def list_cameras():
    return sorted(CAMERA_REG)


def get_camera(name):
    if name not in CAMERA_REG:
        raise KeyError(f"unknown camera module {name!r}; registered: {list_cameras()}")
    return CAMERA_REG[name]


# ------------------------------------------------------------------ hints (what the active modules tell the camera)
class WordInfo:
    """camera hints of one word module: times, plane, hold pose, counters."""

    def __init__(self, idx, mod, a, b, F):
        self.idx, self.mod, self.a, self.b, self.F = idx, mod, a, b, F
        tm = mod.timing
        self.t_form, self.t_hold0, self.t_hold1 = a, a + tm.form, a + tm.form + tm.hold
        self.form, self.hold, self.exit = tm.form, tm.hold, tm.exit
        pl = mod.placement
        self.pl, self.origin, self.n, self.u, self.v = pl, pl.origin.copy(), pl.n, pl.u, pl.v
        self.w, self.h = pl.w, pl.h
        self._counters = None
        self.vn = self._view_normal(pl.n, pl.v)

    @staticmethod
    def _view_normal(n, v, max_elev=58.0):
        """the direction a camera looks at the word to read it: the plane normal, but never steeper than `max_elev` degrees
        above / below the horizon (a floor or ceiling word is read at a raked angle, not top-down, where a horizon-level camera
        has no defined 'up'). The azimuth of a (near) horizontal word is that of its up direction (-v), so it reads upright."""
        n = unit(n)
        el = math.degrees(math.asin(max(-1.0, min(1.0, n[1]))))
        if abs(el) <= max_elev:
            return n
        up = -np.asarray(v, float)
        az = math.atan2(up[0], up[2]) if math.hypot(up[0], up[2]) > 1e-6 else 0.0
        if abs(n[1]) < 0.999:
            az = math.atan2(n[0], n[2])
        e = math.radians(math.copysign(max_elev, el))
        return np.array([math.sin(az) * math.cos(e), math.sin(e), math.cos(az) * math.cos(e)])

    @property
    def yaw_deg(self):
        return math.degrees(yaw_pitch(self.vn)[0])

    def fit(self, fill=0.7):
        """eye distance at which the word fills `fill` of the frame (same rule as Sequence.follow_camera)."""
        fill = fill * cam3d.FIT_FILLK                       # portrait world planes frame tall words fuller (cam3d.set_world)
        w, h = (self.h, self.w) if abs(float(self.u[1])) > abs(float(self.u[0])) else (self.w, self.h)       # a word rolled +-90 reads vertically: its long side runs along the frame's height
        return self.F * max(w / (fill * cam3d.W), h / (fill * cam3d.FIT_HK * cam3d.H))

    def pose(self, dist=None, fill=0.7):
        d = self.fit(fill) if dist is None else dist
        return self.origin - self.vn * d, self.origin.copy()

    def counters(self):
        """holes of the letters (contours inside a letter): [(world centre, radius px, letter char)], biggest first."""
        if self._counters is None:
            out = []
            lay = getattr(self.mod, "lay", None)
            for l in (lay["letters"] if lay else []):
                cons = l["cons"]
                if len(cons) < 2:
                    continue
                area = [0.5 * abs(float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))) for c in cons]
                order = np.argsort(area)[::-1]
                for k in order[1:]:
                    c = cons[k]
                    ctr = 0.5 * (c.min(0) + c.max(0))
                    out.append((self.pl.lift(ctr), math.sqrt(area[k] / math.pi), l["ch"], area[k]))
            out.sort(key=lambda e: -e[3])
            self._counters = [e[:3] for e in out]
        return self._counters


class TransInfo:
    def __init__(self, idx, mod, a, b, wa, wb):
        self.idx, self.mod, self.a, self.b, self.wa, self.wb = idx, mod, a, b, wa, wb

    def bounds(self):
        return self.mod.bounds_points()


class Hints:
    """everything a camera module may ask of the sequence. Built by Sequence.camera_hints(F)."""

    def __init__(self, seq, F=1000.0):
        from mograph.modules.core import WordModule, TransitionModule
        from mograph.modules.sequence import FocusLook
        assert not seq.bare, "camera hints need a sequence of words (not a bare transition)"
        self.seq, self.F = seq, float(F)
        self.T = seq.duration + seq.tail
        self.words, self.trans = [], []
        for it, a, b in seq.timeline:
            if isinstance(it, WordModule):
                self.words.append(WordInfo(len(self.words), it, a, b, self.F))
            else:
                self.trans.append(TransInfo(len(self.trans), it, a, b, None, None))
        for k, tr in enumerate(self.trans):
            tr.wa, tr.wb = self.words[k], self.words[k + 1]
        self._focus = FocusLook(seq)
        self.nth = {}                                   # module name -> number of earlier segments (set by the chain)
        self.enter = None                               # dict(eye, look, vel, acc, lvel, t): where the chain is when this segment starts
        self.next_module, self.next_window, self.next_nth = None, None, 0

    # ---- sequence queries
    def focus(self, t):
        """the strand heads / formed word (what Sequence.focus gives), tabulated; accepts a scalar or an array of times."""
        t = np.asarray(t, float)
        if t.ndim == 0:
            return self._focus(float(t))
        ts, xs = self._focus.ts, self._focus.xs
        tc = np.clip(t, ts[0], ts[-1])
        return np.stack([np.interp(tc, ts, xs[:, k]) for k in range(3)], 1)

    def word_at(self, t):
        """the word module whose window [a, b] contains t; between words (a transition) the nearer word."""
        best = min(self.words, key=lambda w: 0.0 if w.a <= t <= w.b else min(abs(t - w.a), abs(t - w.b)))
        return best

    def item_at(self, t):
        """('word', WordInfo) or ('trans', TransInfo) active at t."""
        for w in self.words:
            if w.a <= t < w.b:
                return "word", w
        for tr in self.trans:
            if tr.a <= t < tr.b:
                return "trans", tr
        return ("word", self.words[-1]) if t >= self.T - 1e-9 or t >= self.words[-1].b else ("word", self.words[0])

    def scene_points(self):
        pts = [p for w in self.words for p in w.pl.corners()]
        for tr in self.trans:
            pts += list(tr.bounds()[::3])
        return np.array(pts)

    def window(self, name):
        """time window by name: 'word:i' (the whole item), 'hold:i', 'form:i', 'exit:i', 'trans:i', 'all'."""
        if name == "all":
            return 0.0, self.T
        kind, _, i = name.partition(":")
        i = int(i)
        if kind == "word":
            w = self.words[i]
            return w.a, w.b
        if kind == "hold":
            w = self.words[i]
            return w.t_hold0, w.t_hold1
        if kind == "form":
            w = self.words[i]
            return w.a, w.t_hold0
        if kind == "exit":
            w = self.words[i]
            return w.t_hold1, w.b
        if kind == "trans":
            tr = self.trans[i]
            return tr.a, tr.b
        raise KeyError(name)

    def default_windows(self, lead_out=0.45, lead_in=0.55):
        """the default tiling of the timeline: a FOLLOW window per transition, from `lead_out` of the previous word's exit to
        `lead_in` of the next word's form; a WORD window in between (the formed word, hold included). -> [(kind, i, t0, t1)]."""
        out, t = [], 0.0
        for k, w in enumerate(self.words):
            if k < len(self.trans):
                tr = self.trans[k]
                f0 = w.b - lead_out * w.exit
                f1 = tr.wb.a + lead_in * tr.wb.form
                out.append(("word", k, t, f0))
                out.append(("trans", k, f0, f1))
                t = f1
            else:
                out.append(("word", k, t, self.T))
        return out


# ------------------------------------------------------------------ segment
class Segment:
    """one module's curves on the 240 Hz grid over [t0 - PAD, t1 + PAD] (clamped outside)."""

    def __init__(self, name, t0, t1, ts, eye, look, roll, logF, right_w, rests=(), fast=(), note="", right_v=None):
        self.name, self.t0, self.t1, self.ts = name, float(t0), float(t1), ts
        self.eye, self.look = np.asarray(eye, float), np.asarray(look, float)
        n = len(ts)
        self.roll = np.broadcast_to(np.asarray(roll, float), (n,)).copy()
        self.logF = np.broadcast_to(np.asarray(logF, float), (n,)).copy()
        self.right_w = np.broadcast_to(np.asarray(right_w, float), (n,)).copy()
        if right_v is None:                          # default hint: the horizon-level right of the view's own azimuth
            f = self.look - self.eye
            hz = np.stack([f[:, 2], np.zeros(n), -f[:, 0]], 1)
            nz = np.linalg.norm(hz, axis=1)
            good = nz > 1e-3 * np.maximum(np.linalg.norm(f, axis=1), 1e-9)
            rv = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
            rv[good] = hz[good] / nz[good, None]
            for i in range(1, n):                    # near-vertical views: keep the last good direction
                if not good[i]:
                    rv[i] = rv[i - 1]
            right_v = rv
        self.right_v = np.asarray(right_v, float)
        self.rests, self.fast, self.note = [tuple(map(float, r)) for r in rests], [tuple(map(float, r)) for r in fast], note

    def sample(self, tg):
        tc = np.clip(tg, self.ts[0], self.ts[-1])
        f = lambda X: np.stack([np.interp(tc, self.ts, X[:, k]) for k in range(X.shape[1])], 1)
        return dict(eye=f(self.eye), look=f(self.look), roll=np.interp(tc, self.ts, self.roll),
                    logF=np.interp(tc, self.ts, self.logF), right_w=np.interp(tc, self.ts, self.right_w), right_v=f(self.right_v))


# ------------------------------------------------------------------ module base
class CameraModule:
    """one camera move. Subclass: set `description`, DEFAULTS, implement curves(); optionally declare()."""
    kind = "camera"
    name = ""
    description = ""
    DEFAULTS = {}
    max_relax = None          # longest join relax (s) the chain may use when entering this move (sustained moves keep their own arc)

    def __init__(self, **params):
        bad = set(params) - set(self.DEFAULTS) - {"F"}
        assert not bad, f"{self.name}: unknown params {sorted(bad)}; known: {sorted(self.DEFAULTS)}"
        self.params = dict(self.DEFAULTS)
        self.params.update(params)
        self.window = None

    def at(self, t0, t1=None):
        """fluent window: Follow().at(0, 3.2) or Follow().at('word:1')."""
        self.window = (t0, t1)
        return self

    def p(self, k):
        return self.params[k]

    # ---- to implement
    def curves(self, hints, ts, t0, t1):
        """-> dict(eye (n,3), look (n,3), roll (n,) | 0, F (n,) | float, right_w (n,) | 0, right_v (n,3) | None) on the time array ts."""
        raise NotImplementedError

    def declare(self, hints, t0, t1):
        """-> (rests, fast): [(a, b)] windows where the eye is meant to be at rest / the view spins fast."""
        return [], []

    def noclear(self, hints, t0, t1):
        """[(a, b)] windows where the eye may go close to strand geometry (e.g. diving through a counter)."""
        return []

    # ---- build
    def segment(self, seq, t0, t1, hints=None):
        hints = hints or seq.camera_hints()
        n0 = int(round((t0 - PAD) * GRID))
        n1 = int(round((t1 + PAD) * GRID))
        ts = np.arange(n0, n1 + 1) / GRID
        c = self.curves(hints, ts, t0, t1)
        F = np.broadcast_to(np.asarray(c.get("F", hints.F), float), (len(ts),))
        rests, fast = self.declare(hints, t0, t1)
        if c.get("fast") is not None:                     # a module may return a boolean array on ts instead
            fast = list(fast) + _runs(ts, c["fast"])
        sg = Segment(self.name, t0, t1, ts, c["eye"], c["look"], c.get("roll", 0.0), np.log(F), c.get("right_w", 0.0),
                     rests, fast, right_v=c.get("right_v"))
        sg.noclear = [tuple(map(float, r)) for r in self.noclear(hints, t0, t1)]
        sg.max_relax = self.max_relax
        return sg

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.params.items() if self.DEFAULTS.get(k) != v)})"


def _runs(ts, mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((ts[i], ts[j]))
            i = j + 1
        else:
            i += 1
    return out


# ------------------------------------------------------------------ chain
class CameraChain:
    """a camera made of consecutive CameraModules, joined with C2 continuity.

    plan: list of CameraModule with a window set via .at(t0, t1 | name), or (module, t0, t1) tuples. Windows must follow each
    other (gaps are closed by extending the earlier one; the first starts at 0 and the last ends at the sequence end).
    join: half-width (s) of the cross-fade at every boundary. ease_in / ease_out: the chain starts from / ends at rest.
    Gates (phase 3b fix): a join whose route doubles back / turns hard (heading change > rest_head deg per frame) becomes an
    explicit ease-to-rest turn (the camera brakes along its own path to rest, the next module leaves from rest); whatever
    still turns faster than turn_deg deg/frame is slowed by the governor (see _govern), so the heading / view gates hold.
    Cameras are horizon-level (image-right = down x view) unless a module supplies right_w / right_v (tilt_to_plan)."""
    DEFAULT_GOVERN = True

    def __init__(self, seq, plan, F=1000.0, join=0.7, ease_in=0.6, ease_out=0.8, join_smooth=0.0, clear=140.0, hints=None,
                 govern=None, turn_deg=2.4, min_rate=0.25, max_rate=1.35, rest_join=True, rest_head=3.5, look_relax=1.0):
        self.F, self.join, self.ease_in, self.ease_out = float(F), float(join), float(ease_in), float(ease_out)
        govern = self.DEFAULT_GOVERN if govern is None else govern
        self.govern, self.turn_deg, self.min_rate, self.max_rate = bool(govern), float(turn_deg), float(min_rate), float(max_rate)
        self.rest_join, self.rest_head = bool(rest_join), float(rest_head)
        self.look_relax = float(look_relax)
        self.gov = dict(max_lag=0.0, slowed_s=0.0, applied=False)
        self.join_smooth, self.clear = float(join_smooth), float(clear)
        hints = hints or Hints(seq, F)
        self._hints = hints
        self.T = hints.T
        items = []
        for e in plan:
            if isinstance(e, CameraModule):
                assert e.window is not None, f"{e!r} has no window: use .at(t0, t1) or (module, t0, t1)"
                m, a, b = e, e.window[0], e.window[1]
            else:
                m, a, b = e
            if isinstance(a, str):
                a, b = hints.window(a)
            items.append((m, float(a), None if b is None else float(b)))
        items.sort(key=lambda e: e[1])
        win = []
        for k, (m, a, b) in enumerate(items):
            nxt = items[k + 1][1] if k + 1 < len(items) else self.T
            win.append((m, 0.0 if k == 0 else a, nxt))
        win[-1] = (win[-1][0], win[-1][1], self.T)
        self.plan = [(m.name, dict(m.params), a, b) for m, a, b in win]
        n = int(round(self.T * GRID))
        tg = np.arange(n + 1) / GRID
        self.tg = tg
        self.segments, self.cur, counts = [], None, {}
        for k, (m, a, b) in enumerate(win):
            hints.nth[m.name] = counts.get(m.name, 0)
            nm = win[k + 1][0] if k + 1 < len(win) else None
            hints.next_module = nm
            hints.next_window = (win[k + 1][1], win[k + 1][2]) if nm is not None else None
            hints.next_nth = counts.get(nm.name, 0) + (1 if (nm is not None and nm.name == m.name) else 0) if nm else 0
            hints.enter = self._state(a) if self.cur is not None else None
            sg = m.segment(seq, a, b, hints)
            counts[m.name] = counts.get(m.name, 0) + 1
            self.segments.append(sg)
            self._append(sg, rest_start=(k == 0))
        if self.clear > 0:
            self._clear(seq)
        self._finish()

    # ---- compose: sequential, every segment starts exactly on the pose / velocity / acceleration the chain has reached
    def _back(self, Y, i):
        """value, velocity, acceleration of table Y at index i from BACKWARD differences (the table beyond i belongs to a
        segment that is about to be replaced)."""
        G = GRID
        return (Y[i], (3 * Y[i] - 4 * Y[i - 1] + Y[i - 2]) * G / 2.0,
                (2 * Y[i] - 5 * Y[i - 1] + 4 * Y[i - 2] - Y[i - 3]) * G * G)

    def _fwd(self, X, i):
        G = GRID
        return (X[i], (-3 * X[i] + 4 * X[i + 1] - X[i + 2]) * G / 2.0,
                (2 * X[i] - 5 * X[i + 1] + 4 * X[i + 2] - X[i + 3]) * G * G)

    def _state(self, t):
        """(eye, look, vel, acc) of the composite so far at time t (from the 240 Hz table, backward differences)."""
        i = min(max(int(round(t * GRID)), 3), len(self.tg) - 4)
        e, l = self._back(self.cur["eye"], i), self._back(self.cur["look"], i)
        return dict(eye=e[0].copy(), look=l[0].copy(), vel=e[1], acc=e[2], lvel=l[1], t=t)

    @staticmethod
    def _h5(u):
        return (1 - 10 * u ** 3 + 15 * u ** 4 - 6 * u ** 5, u - 6 * u ** 3 + 8 * u ** 4 - 3 * u ** 5,
                0.5 * u ** 2 - 1.5 * u ** 3 + 1.5 * u ** 4 - 0.5 * u ** 5)

    def _corrected(self, X, start, i0, relax):
        """module table X plus the quintic Hermite correction that starts it on `start` = (p, v, a) at index i0."""
        tg = self.tg
        _, v, a = self._fwd(X, i0)
        p0, v0, a0 = start
        u = np.clip((tg - tg[i0]) / relax, 0.0, 1.0)
        H0, H1, H2 = self._h5(u)
        sh = (-1,) + (1,) * (X.ndim - 1)
        return X + ((p0 - X[i0]) * H0.reshape(sh) + (v0 - v) * relax * H1.reshape(sh) + (a0 - a) * relax ** 2 * H2.reshape(sh))

    def _hermite(self, X, start, i0, relax):
        """the module table X, but over [i0, i0 + relax] replaced by the quintic Hermite curve from `start` = (p, v, a) to the
        state (p, v, a) X has at the end of the window (C2 on both sides). Leaves rest along the chord, never along X's wiggles."""
        tg = self.tg
        i1 = min(i0 + int(round(relax * GRID)), len(tg) - 5)
        T = (i1 - i0) / GRID
        p0, v0, a0 = start
        p1, v1, a1 = self._fwd(X, i1)
        out = X.copy()
        u = (tg[i0:i1 + 1] - tg[i0]) / T
        sh = (-1,) + (1,) * (X.ndim - 1)
        H0, H1, H2 = self._h5(u)
        G0, G1, G2 = self._h5(1.0 - u)                     # mirrored basis for the end state
        P = (H0.reshape(sh) * p0 + H1.reshape(sh) * T * v0 + H2.reshape(sh) * T * T * a0
             + G0.reshape(sh) * p1 - G1.reshape(sh) * T * v1 + G2.reshape(sh) * T * T * a1)
        out[i0:i1 + 1] = P
        return out

    @staticmethod
    def _join_cost(E, i0, relax):
        """how bad the join looks: worst heading change per frame (30 fps) and speed dips over the relax window."""
        step = int(round(GRID / 30.0))
        seg = E[i0:i0 + int((relax + 0.5) * GRID):step]
        if len(seg) < 5:
            return 0.0
        d = np.diff(seg, axis=0)
        sp = np.linalg.norm(d, axis=1) * 30.0
        ref = max(float(np.median(sp)), 30.0)
        h = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip((h[1:] * h[:-1]).sum(1), -1, 1)))
        ok = (sp[1:] > 0.05 * ref) & (sp[:-1] > 0.05 * ref)
        head = float(ang[ok].max()) if ok.any() else 0.0
        dip = max(0.0, 0.4 - float(sp.min()) / ref)
        return head + 30.0 * dip + 0.4 * relax + 0.004 * float(sp.max() / 10.0)

    @staticmethod
    def _join_head(E, i0, relax):
        """worst heading change per frame (deg, 30 fps) of the eye path over the relax window, moving frames only."""
        step = int(round(GRID / 30.0))
        seg = E[i0:i0 + int((relax + 0.6) * GRID):step]
        if len(seg) < 5:
            return 0.0
        d = np.diff(seg, axis=0)
        sp = np.linalg.norm(d, axis=1) * 30.0
        ref = max(float(np.median(sp)), 30.0)
        h = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip((h[1:] * h[:-1]).sum(1), -1, 1)))
        ok = (sp[1:] > 0.07 * ref) & (sp[:-1] > 0.07 * ref)
        return float(ang[ok].max()) if ok.any() else 0.0

    def _brake(self, ib, i_s):
        """the composite so far, brought to REST at index i_s: over [ib, i_s] every channel replays its own recent past with a
        quintic-eased time rate (1 -> 0), so the path is unchanged and the camera just decelerates along it (C2). -> new tables."""
        n = i_s - ib
        u = np.arange(n + 1) / n
        g = u - (u ** 6 - 3 * u ** 5 + 2.5 * u ** 4)                       # integral of 1 - smootherstep
        pos = ib + n * g                                                   # fractional source index
        out = {}
        for k, X in self.cur.items():
            Y = X.copy()
            if X.ndim == 2:
                Y[ib:i_s + 1] = np.stack([np.interp(pos, np.arange(len(X)), X[:, j]) for j in range(X.shape[1])], 1)
            else:
                Y[ib:i_s + 1] = np.interp(pos, np.arange(len(X)), X)
            Y[i_s:] = Y[i_s]
            out[k] = Y
        return out

    def _focus_pen(self, E, L, i0, relax, half_fov_deg=26.0):
        """penalty for join candidates that look away from the creation lines (strand focus outside the frame)."""
        step = int(round(GRID / 15.0))
        idx = np.arange(i0, min(i0 + int((relax + 0.3) * GRID), len(self.tg) - 1), step)
        if len(idx) < 2:
            return 0.0
        f = self._hints.focus(self.tg[idx])
        a = L[idx] - E[idx]
        b = f - E[idx]
        c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
        off = np.degrees(np.arccos(np.clip(c, -1, 1))) > half_fov_deg
        return 25.0 * float(off.mean())

    def _append(self, sg, rest_start):
        tg = self.tg
        S = sg.sample(tg)
        i0 = min(max(int(round(sg.t0 * GRID)), 3), len(tg) - 5)
        if self.cur is None:                                      # the chain starts at rest on the module's first pose
            start = {k: (X[i0], 0.0 * X[i0], 0.0 * X[i0]) for k, X in S.items()}
            relax = self.ease_in
        else:
            start = {k: self._back(self.cur[k], i0) for k in S}
            best = None
            mr = getattr(sg, "max_relax", None) or 9.0
            for r in [r_ for r_ in (0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.6) if r_ <= mr + 1e-9] or [0.5]:
                ze = self._corrected(S["eye"], start["eye"], i0, r)
                zl = self._corrected(S["look"], start["look"], i0, min(r, self.look_relax))
                c = self._join_cost(ze, i0, r) + self._focus_pen(ze, zl, i0, r)
                if best is None or c < best[0] - 1e-9:
                    best = (c, r, self._join_head(ze, i0, r))
            relax = best[1]
            sg.rest_join = None
            sg.join_info = dict(cont_head=round(best[2], 2), relax=relax)
            if best[2] > self.rest_head and self.rest_join:
                # the route doubles back / turns hard at this join: ease the camera to REST along its own path, then out on the
                # new module's route (an explicit ease-to-rest reversal; the new curve starts from rest)
                cand = None
                for tb in (0.55, 0.8):
                    i_s = min(i0 + int(0.1 * GRID), len(tg) - 5)
                    ib = i_s - int(tb * GRID)
                    if ib < 4:
                        continue
                    br = self._brake(ib, i_s)
                    st2 = {k: (br[k][i_s], 0.0 * br[k][i_s], 0.0 * br[k][i_s]) for k in S}
                    for r in [r_ for r_ in (0.6, 0.9, 1.2, 1.6, 2.0, 2.6, 3.2) if r_ <= min((sg.t1 - sg.t0) + PAD - 0.2, mr)] or [0.6]:
                        for how in ("hermite", "offset"):         # chord from rest, or the module's own route plus a decaying offset
                            fn = self._hermite if how == "hermite" else self._corrected
                            ze = fn(S["eye"], st2["eye"], i_s, r)
                            zl = fn(S["look"], st2["look"], i_s, min(r, self.look_relax))
                            hh = self._join_head(ze, i_s, r)
                            c = hh + self._focus_pen(ze, zl, i_s, r) + 0.4 * r + 3.0 * tb
                            if cand is None or c < cand[0]:
                                cand = (c, r, tb, i_s, br, st2, hh, how)
                sg.join_info["rest_head"] = None if cand is None else round(cand[6], 2)
                if cand is not None and cand[6] < best[2] - 0.5:
                    _, relax, tb, i0, br, start, hh, how = cand
                    self.cur = br
                    sg.rest_join = (i0 / GRID, tb)
                    sg.rest_how = how
        sg.relax = relax
        sg.join_rests = []
        out = {}
        for key, X in S.items():
            rk = min(relax, self.look_relax) if key == "look" else relax       # the view settles on a new target quicker than the eye
            if getattr(sg, "rest_join", None) and sg.rest_how == "hermite":
                Z = self._hermite(np.asarray(X, float), start[key], i0, rk)
            else:
                Z = self._corrected(np.asarray(X, float), start[key], i0, rk)
            if self.cur is not None:
                Y = self.cur[key].copy()
                Y[i0:] = Z[i0:]
                Z = Y
            else:
                Z = Z.copy()
                Z[:i0] = Z[i0]
            out[key] = Z
        if getattr(sg, "rest_join", None):
            ts_, tb = sg.rest_join
            sg.join_rests.append((ts_ - 0.25, ts_ + 0.25))
        elif self.cur is not None and sg.join_rests == []:
            # a join may have to turn the camera round (a V in the route): where the eye eases through rest inside the relax
            # window it is declared as an ease-to-rest reversal (like ease_reversal), not left to read as an accidental stop
            step = int(round(GRID / 30.0))
            E = out["eye"][i0:i0 + int((relax + 0.5) * GRID):step]
            if len(E) > 4:
                sp = np.linalg.norm(np.diff(E, axis=0), axis=1) * 30.0
                ref = max(float(np.median(sp)), 30.0)
                k = int(np.argmin(sp))
                if sp[k] < 0.3 * ref and 0 < k < len(sp) - 1:
                    tm = (i0 + (k + 0.5) * step) / GRID
                    sg.join_rests.append((tm - 0.3, tm + 0.3))
        self.cur = out

    def _clear(self, seq, fps=20.0, iters=5):
        """keep the eye out of the strands: wherever it comes within `clear` px of a strand line it is pushed away from the
        nearest point (push = shortfall, smoothed over 0.2 s so it stays C-infinity). Windows a module declares `noclear` (diving
        through a counter) are exempt. Run twice so the corrected path is checked again."""
        from scipy.spatial import cKDTree
        tg = self.tg
        ts = np.arange(0.0, self.T + 1e-9, 1.0 / fps)
        trees = []
        for t in ts:
            Q = []
            for L in seq.layers(float(min(t, self.T))):
                if L.kind != "line":
                    continue
                for poly in L.polys:
                    if len(poly) < 2:
                        continue
                    a, b = poly[:-1], poly[1:]
                    Q.append(poly)
                    Q.append(a * 0.75 + b * 0.25)
                    Q.append(a * 0.5 + b * 0.5)
                    Q.append(a * 0.25 + b * 0.75)
            trees.append(cKDTree(np.concatenate(Q)) if Q else None)
        ex = np.zeros_like(ts)
        for sg in self.segments:
            for a, b in getattr(sg, "noclear", []):
                ex = np.maximum(ex, smoother((ts - (a - 0.2)) / 0.2) * (1.0 - smoother((ts - b) / 0.2)))
        step = int(round(GRID / fps))
        for _ in range(iters):
            E = self.cur["eye"][::step][:len(ts)]
            push = np.zeros((len(ts), 3))
            for i, tr in enumerate(trees):
                if tr is None:
                    continue
                d, j = tr.query(E[i])
                if d < self.clear:
                    v = E[i] - tr.data[j]
                    push[i] = v / max(np.linalg.norm(v), 1e-6) * (self.clear - d) * 1.25 * (1.0 - ex[i])
            if not push.any():
                break
            ps = gaussian_filter1d(push, 0.2 * fps, axis=0, mode="nearest")
            full = np.stack([np.interp(tg, ts, ps[:, k]) for k in range(3)], 1)
            self.cur["eye"] = self.cur["eye"] + full

    def _round_joins(self):
        """round the corners of the eye path inside every join zone (Gaussian smoothing of sigma `join_smooth` s, masked by a
        quintic window that is 1 over the relax interval and fades over 0.3 s each side): turns at joins get a bigger radius."""
        tg = self.tg
        M = np.zeros_like(tg)
        for sg in self.segments[1:]:
            a, b = sg.t0, sg.t0 + sg.relax
            m = smoother((tg - (a - 0.3)) / 0.3) * (1.0 - smoother((tg - b) / 0.3))
            M = np.maximum(M, m)
        E = self.cur["eye"]
        Es = gaussian_filter1d(E, self.join_smooth * GRID, axis=0, mode="nearest")
        self.cur["eye"] = E + (Es - E) * M[:, None]

    def _finish(self):
        if self.join_smooth > 0:
            self._round_joins()
        tg, cur = self.tg, self.cur
        # the end eases to rest: blend with the frozen last pose (quintic weight, C2)
        wo = smoother((self.T - tg) / self.ease_out) if self.ease_out > 0 else np.ones_like(tg)
        for k in cur:
            X = cur[k]
            sh = (-1,) + (1,) * (X.ndim - 1)
            X1 = X[-1]
            cur[k] = X1 + wo.reshape(sh) * (X - X1)
        self.rests = [(a - 0.25, b + 0.25) for s in self.segments for a, b in s.rests]
        self.join_rests = [r for s in self.segments for r in getattr(s, 'join_rests', [])]
        self.rests += self.join_rests
        self.fast = [(a - 0.1, b + 0.1) for s in self.segments for a, b in s.fast]
        if self.govern:
            self._govern()
        sp = lambda X: CubicSpline(tg, X, axis=0)
        self._eye, self._look = sp(cur["eye"]), sp(cur["look"])
        self._roll, self._logF, self._rw = sp(cur["roll"]), sp(cur["logF"]), sp(cur["right_w"])
        self._rv = sp(cur["right_v"])
        if self.govern and self.gov.get("applied"):
            self._declare_slow()

    # ---- turn governor
    def _govern(self):
        """the join / turn governor. Heading change of the eye path and turn of the view are rates (deg per s = curvature x
        speed), so a tight turn at speed is a heading jump. Wherever the composed path would turn faster than `turn_deg` deg per
        frame (outside declared rests and declared fast spins) the WHOLE camera (eye, look, roll, F) is slowed through that
        stretch of its own path, anticipating it (the camera eases into the turn, down to a near stop for a true reversal), and
        catches the lost time up again wherever the path is straight (rate up to `max_rate`). Geometry is untouched; only the
        time along it is re-mapped (monotone, C2), so every module's route and framing stay as designed. Declared windows are
        mapped through the same re-mapping; stretches that end up (almost) at rest are declared ease-to-rest turns."""
        tg, cur = self.tg, self.cur
        n = len(tg)
        G = GRID
        om_lim = math.radians(self.turn_deg) * 30.0                          # rad / s
        sg = 2.0 / G * G                                                      # derivative smoothing: 2 grid steps
        d1 = lambda X: gaussian_filter1d(X, 3.0, axis=0, order=1, mode="nearest") * G
        v = d1(cur["eye"])
        a = d1(v)
        sp = np.linalg.norm(v, axis=1)
        om_h = np.linalg.norm(np.cross(v, a), axis=1) / np.maximum(sp, 1e-6) ** 2
        look_dir = unit(cur["look"] - cur["eye"])
        om_l = np.linalg.norm(d1(look_dir), axis=1)
        rr = self._roll_rate(cur) / 36.0 * om_lim                            # deg/s of image roll, scaled to the same limit
        free_h = np.zeros(n, bool)                                            # heading is not limited inside declared rests / fast spins
        for lo, hi in list(self.rests) + list(self.fast):
            free_h |= (tg > lo) & (tg < hi)
        free_l = np.zeros(n, bool)                                            # view turn / roll: only inside declared fast spins
        for lo, hi in self.fast:
            free_l |= (tg > lo) & (tg < hi)
        ease = (tg < self.ease_in) | (tg > self.T - self.ease_out)
        free_h |= ease
        free_l |= ease
        need = np.maximum(np.where(free_h, 0.0, om_h), np.where(free_l, 0.0, np.maximum(om_l, rr)))
        cap = np.minimum(9.0, om_lim / np.maximum(need, 1e-9))
        # the slow-down begins before the turn: erode (min filter) by `lead` s, then smooth
        lead = int(0.1 * G)
        from scipy.ndimage import minimum_filter1d
        cap = minimum_filter1d(cap, 2 * lead + 1, mode="nearest")
        cap = gaussian_filter1d(cap, 0.04 * G, mode="nearest")
        cap = np.maximum(cap, self.min_rate)
        if cap.min() > 0.999:
            return                                                       # nothing turns too fast: no re-timing
        # integrate source time tau(t): tau' = min(cap(tau), catch-up rate)
        tau = np.zeros(n)
        r_arr = np.zeros(n)
        for k in range(n - 1):
            lag = tg[k] - tau[k]
            c = float(np.interp(tau[k], tg, cap))
            want = 1.0 + min(self.max_rate - 1.0, 1.5 * lag)
            r = min(c, want) if lag > 1e-6 else min(c, 1.0)
            r_arr[k] = r
            tau[k + 1] = min(tau[k] + r / G, tg[-1])
        r_arr[-1] = r_arr[-2]
        tau = np.minimum(tau, tg[-1])
        # smooth the warp a little (keeps C2, tiny effect) and keep it monotone
        ts_ = gaussian_filter1d(r_arr, 0.04 * G, mode="nearest")
        tau = np.r_[0.0, np.cumsum(ts_[:-1] / G)]
        tau = np.minimum(tau, tg[-1])
        for k in cur:
            X = cur[k]
            cur[k] = np.stack([np.interp(tau, tg, X[:, j]) for j in range(X.shape[1])], 1) if X.ndim == 2 else np.interp(tau, tg, X)
        mp = lambda t0: float(np.interp(t0, tau, tg))                       # source time -> output time
        self.rests = [(mp(a_), mp(b_)) for a_, b_ in self.rests]
        self.fast = [(mp(a_), mp(b_)) for a_, b_ in self.fast]
        self.join_rests = [(mp(a_), mp(b_)) for a_, b_ in self.join_rests]
        slowed = ts_ < 0.9
        self.gov = dict(max_lag=float((tg - tau).max()), slowed_s=float(slowed.sum() / G), applied=True,
                        min_rate=float(ts_.min()), end_lag=float(tg[-1] - tau[-1]))

    @staticmethod
    def _roll_rate(cur):
        """image roll about the view axis (deg per second, parallel transport) of the composed tables, 240 Hz."""
        eye, look = cur["eye"], cur["look"]
        f = unit(look - eye)
        r_h = np.cross(np.array([0.0, 1.0, 0.0]), f)
        rv = cur["right_v"]
        rh = rv - (rv * f).sum(1, keepdims=True) * f
        w = np.clip(cur["right_w"], 0.0, 1.0)[:, None]
        nh, nr = np.linalg.norm(r_h, axis=1, keepdims=True), np.linalg.norm(rh, axis=1, keepdims=True)
        r = np.where((w >= 0.9999) | (nh < 1e-6), rh, (1 - w) * r_h / np.maximum(nh, 1e-9) + w * rh / np.maximum(nr, 1e-9))
        r = np.where(np.linalg.norm(r, axis=1, keepdims=True) < 1e-6, np.array([1.0, 0.0, 0.0]), r)
        r = unit(r)
        d = np.cross(f, r)
        c, sn = np.cos(cur["roll"])[:, None], np.sin(cur["roll"])[:, None]
        r2 = r * c - d * sn
        tr = transport_roll(f, r2) * GRID
        tr = np.r_[tr, tr[-1:]]
        return gaussian_filter1d(tr, 3.0, mode="nearest")

    def _declare_slow(self):
        """stretches where the governed camera is (almost) at rest outside the declared rests become declared ease-to-rest turns."""
        fps = 30.0
        ts = np.arange(int(self.T * fps) + 1) / fps
        E = np.array([self._eye(t) for t in ts])
        sp = np.linalg.norm(np.diff(E, axis=0), axis=1) * fps
        tm = 0.5 * (ts[1:] + ts[:-1])
        stop = np.zeros(len(tm), bool)
        for lo, hi in self.rests:
            stop |= (tm > lo) & (tm < hi)
        ease = (tm < self.ease_in) | (tm > self.T - self.ease_out)
        mask = ~stop & ~ease
        if not mask.any():
            return
        med = float(np.median(sp[mask]))
        low = mask & (sp < 0.09 * med)
        i = 0
        while i < len(low):
            if low[i]:
                j = i
                while j + 1 < len(low) and low[j + 1]:
                    j += 1
                self.join_rests.append((tm[i] - 0.12, tm[j] + 0.12))
                self.rests.append((tm[i] - 0.12, tm[j] + 0.12))
                i = j + 1
            else:
                i += 1

    def at(self, t):
        t = min(max(float(t), 0.0), self.T)
        eye, look = self._eye(t), self._look(t)
        if np.linalg.norm(look - eye) < 1.0:
            look = eye + np.array([0.0, 0.0, 1.0])
        rw = min(max(float(self._rw(t)), 0.0), 1.0)
        return Camera(eye, look, float(self._roll(t)), math.exp(float(self._logF(t))), right=self._rv(t), right_w=rw)

    # ---- text
    def plan_text(self):
        return " | ".join(f"{n} [{a:.2f}-{b:.2f}]" for n, p, a, b in self.plan)

    # ---- validation (speed_report style, on the cameras the renderer will actually get)
    def report(self, fps=30.0, verbose=False):
        N = int(round(self.T * fps)) + 1
        ts = np.arange(N) / fps
        cams = [self.at(t) for t in ts]
        E = np.array([c.eye for c in cams])
        Fv = np.array([c.f for c in cams])
        dE = np.diff(E, axis=0)
        sp = np.linalg.norm(dE, axis=1) * fps
        tm = 0.5 * (ts[1:] + ts[:-1])
        hd = dE / np.maximum(np.linalg.norm(dE, axis=1, keepdims=True), 1e-12)
        hj = np.degrees(np.arccos(np.clip((hd[1:] * hd[:-1]).sum(1), -1, 1)))                    # per frame, steps 1..
        look_deg = np.degrees(np.arccos(np.clip((Fv[1:] * Fv[:-1]).sum(1), -1, 1)))
        stop = np.zeros(len(tm), bool)
        for lo, hi in self.rests:
            stop |= (tm > lo) & (tm < hi)
        fast = np.zeros(len(tm), bool)
        for lo, hi in self.fast:
            fast |= (tm > lo) & (tm < hi)
        ease = (tm < self.ease_in) | (tm > self.T - self.ease_out)
        mask = ~stop & ~ease
        if not mask.any():
            mask = ~ease
        med = float(np.median(sp[mask]))
        droll = transport_roll(Fv, np.array([c.r for c in cams]))
        jerk = np.abs(np.diff(sp, 2)) / max(med, 1e-9)
        mov = sp > 0.05 * med
        ok = mov[1:] & mov[:-1] & ~stop[1:] & ~stop[:-1] & ~ease[1:] & ~ease[:-1]
        okf, okn = ok & fast[1:], ok & ~fast[1:]
        mv = (sp > 0.35 * med)
        okm = ok & ~fast[1:] & mv[1:] & mv[:-1]
        lk_n, lk_f = look_deg[~fast & ~ease], look_deg[fast & ~ease]
        s = dict(median=med, min_mid=float(sp[mask].min()), max_mid=float(sp[mask].max()),
                 min_ratio=float(sp[mask].min() / max(med, 1e-9)),
                 near_zero_outside=int((sp[mask] < 0.05 * med).sum()),
                 stop_frames=int(stop.sum()), fast_frames=int(fast.sum()),
                 max_speed_in_rests=float(sp[stop].max()) if stop.any() else 0.0,
                 max_heading_jump_deg=float(hj[okn].max()) if okn.any() else 0.0,
                 max_heading_jump_moving_deg=float(hj[okm].max()) if okm.any() else 0.0,
                 max_heading_jump_fast_deg=float(hj[okf].max()) if okf.any() else 0.0,
                 max_look_turn_deg=float(lk_n.max()) if len(lk_n) else 0.0,
                 max_look_turn_fast_deg=float(lk_f.max()) if len(lk_f) else 0.0,
                 max_roll_change_deg=float(droll[~ease[:len(droll)] & ~fast[:len(droll)]].max()) if (~ease[:len(droll)] & ~fast[:len(droll)]).any() else 0.0,
                 max_jerk_frac=float(jerk[(mask[2:] if len(mask) > 2 else mask)].max()) if len(jerk) else 0.0,
                 start_speed=float(sp[0]), end_speed=float(sp[-1]), join_reversals=[round(0.5 * (a + b), 2) for a, b in self.join_rests])
        s["gates"] = dict(no_stop=s["near_zero_outside"] == 0, heading=s["max_heading_jump_deg"] <= 3.0,
                          look=s["max_look_turn_deg"] <= 3.0, roll=s["max_roll_change_deg"] <= 1.5)
        s["pass"] = all(s["gates"].values())
        out = dict(t=tm, eye_speed=sp, heading_jump_deg=hj, look_turn_deg=look_deg, stop_mask=stop, fast_mask=fast,
                   roll_step_deg=droll, summary=s)
        if verbose:
            print(self.speed_line(s))
        return out

    @staticmethod
    def speed_line(s):
        g = s["gates"]
        ok = "gates pass" if s["pass"] else "gates FAIL: " + ", ".join(k for k, v in g.items() if not v)
        fast = (f", fast spins up to {max(s['max_heading_jump_fast_deg'], s['max_look_turn_fast_deg']):.1f} deg/frame"
                if s["fast_frames"] else "")
        mv = (f" ({s['max_heading_jump_moving_deg']:.1f} while above 35% of median speed)"
              if s["max_heading_jump_deg"] > 3.0 else "")
        return (f"Eye speed median {s['median']:.0f} px/s (min {s['min_mid']:.0f}, max {s['max_mid']:.0f} outside declared "
                f"rests); heading jump max {s['max_heading_jump_deg']:.1f} deg/frame{mv}, view turn max "
                f"{s['max_look_turn_deg']:.1f} deg/frame{fast}, roll max {s['max_roll_change_deg']:.2f} deg/frame; "
                f"{s['stop_frames']} declared rest frames"
                + (f" (ease-to-rest turn at {', '.join(str(x) + ' s' for x in s['join_reversals'])})" if s["join_reversals"] else "")
                + f"; {ok}.")

    # ---- auto
    @classmethod
    def auto(cls, seq, override=None, words="rules", F=1000.0, **kw):
        """default camera: FOLLOW through every transition (strands lead the eye), and per word window by a simple rule
        (words='rules'): a word on a tilted plane (|yaw| >= 12 deg) with a hold >= 0.6 s gets ORBIT_EVEN (the eye swings from its
        oblique view toward face-on, direction alternating word to word); any other word gets HOLD_DRIFT (composed off-centre
        still with a slow push and sub-pixel drift). words='hold_drift' forces HOLD_DRIFT on every word.
        override: {window name or (t0, t1): module} replaces / carves into that window, e.g. {'word:1': OrbitEven()} or
        {(3.4, 7.4): OrbitEven()}; the neighbouring default windows shrink to make room."""
        from mograph.modules.camera import Follow, HoldDrift, OrbitEven
        hints = Hints(seq, F)
        plan = []
        for kind, i, a, b in hints.default_windows():
            if kind == "trans":
                plan.append((Follow(), a, b))
                continue
            w = hints.words[i]
            if words == "rules" and abs(w.yaw_deg) >= 12.0 and w.hold >= 0.6:
                plan.append((OrbitEven(), a, b))
            else:
                plan.append((HoldDrift(), a, b))
        for key, m in (override or {}).items():
            a, b = hints.window(key) if isinstance(key, str) else key
            plan = carve(plan, m, float(a), float(b))
        return cls(seq, plan, F=F, hints=hints, **kw)


def carve(plan, m, a, b, min_len=0.3):
    """insert module m on [a, b] into a tiled plan [(module, t0, t1)]; overlapped windows shrink / vanish."""
    out = []
    for mod, t0, t1 in plan:
        if t1 <= a + 1e-6 or t0 >= b - 1e-6:
            out.append((mod, t0, t1))
            continue
        if t0 < a - min_len:
            out.append((mod, t0, a))
        if t1 > b + min_len:
            out.append((mod, b, t1))
    out.append((m, a, b))
    out.sort(key=lambda e: e[1])
    # close small gaps: each window starts where the previous ends
    fixed = []
    for k, (mod, t0, t1) in enumerate(out):
        fixed.append((mod, t0, out[k + 1][1] if k + 1 < len(out) else t1))
    return fixed
