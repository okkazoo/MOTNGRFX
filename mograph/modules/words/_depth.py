"""Shared engine for the depth / volume word modules (batch B: depth_echo, candy_orbit, xray_unfold, hinge_unfold,
topo_tower, hollow_tubes, depth_tunnel).

Every module here forms in two hand-offs and one mechanic:
  1. the incoming strands arrive along `entry`, peel off and draw the OUTLINE of the type in the placement plane
     (exactly the `lanes` hand-off, built from `_strand.StrandWord`, but compressed into the first part of `form`);
  2. the module's own mechanic then takes the outline over and builds the volume (echo copies, banded rod, hinged slices,
     tower, tubes, tunnel...) -- progress p in [0, 1], `MechWord.progress(t)`;
  3. on exit the mechanic folds back into the flat outline, then the strands detach and leave along `exit`.
`HIDE = True` modules (the outline itself moves, e.g. hinges) draw the strands only before and after the mechanic.

Layers are camera independent, but a solid wall is only drawn where it faces the eye, so modules ask `eye_at(t)`: the
library driver sets `eye_hint` to the real camera; without it a nominal eye in front of the placement is used.
"""
import math
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator

from mograph.modules.core import Timing, Layer, clamp, mixc, unit, strand_layers
from mograph.modules.words._strand import StrandWord
from mograph.render import cam3d


def ss(x):
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


def eio(x, p=2.4):
    x = min(1.0, max(0.0, float(x)))
    if x <= 0 or x >= 1:
        return x
    a, b = x ** p, (1 - x) ** p
    return a / (a + b)


def eo(x, p=2.6):
    x = min(1.0, max(0.0, float(x)))
    return 1.0 - (1.0 - x) ** p


def seg(t, a, b):
    return clamp((t - a) / max(b - a, 1e-9))


def lerp(a, b, u):
    return a + (b - a) * u


# ------------------------------------------------------------------ cameras for the library clips
class ArcCamera:
    """slow oblique arc around `center`; keys = [(t, yaw_deg, pitch_deg, dist)], pchip-smooth. Picklable. yaw > 0 turns the
    view toward +x (the eye swings to the -x side), pitch > 0 looks down."""

    def __init__(self, center, keys, F=1000.0):
        k = np.asarray(keys, float)
        self.c, self.F = np.asarray(center, float), float(F)
        self.t0, self.t1 = float(k[0, 0]), float(k[-1, 0])
        self.f = [PchipInterpolator(k[:, 0], k[:, i]) for i in (1, 2, 3)]

    def at(self, t):
        t = min(max(float(t), self.t0), self.t1)
        yaw, pit, dist = (float(f(t)) for f in self.f)
        return cam3d.Camera.orbit(self.c, dist, math.radians(yaw), math.radians(pit), 0.0, self.F)


class EyeOf:
    """picklable eye(t) of a camera object (anything with .at(t))."""

    def __init__(self, cam):
        self.cam = cam

    def __call__(self, t):
        return self.cam.at(t).eye


# ------------------------------------------------------------------ contour helpers
def simplify(c, eps=0.7):
    c = np.asarray(c, np.float32)
    if len(c) < 8:
        return c.astype(float)
    a = cv2.approxPolyDP(c.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    return a.astype(float) if len(a) >= 3 else c.astype(float)


def area2(c):
    return float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))


def letter_edges(letters, eps=0.7):
    """wall edges of all letter contours: dict(A, B (M,2) edge ends, N (M,2) unit normal pointing OUT of the letter solid,
    L (M,) letter index, C (M,) contour index). Winding is made consistent per letter from its largest contour."""
    As, Bs, Ns, Ls, Cs = [], [], [], [], []
    cid = 0
    for li, l in enumerate(letters):
        cons = [simplify(c, eps) for c in l["cons"]]
        if not cons:
            continue
        s = np.sign(area2(max(cons, key=lambda c: abs(area2(c))))) or 1.0
        for c in cons:
            a, b = c, np.roll(c, -1, 0)
            t = b - a
            n = s * np.stack([t[:, 1], -t[:, 0]], 1)
            nn = np.linalg.norm(n, axis=1)
            ok = nn > 1e-6
            As.append(a[ok]); Bs.append(b[ok]); Ns.append(n[ok] / nn[ok][:, None])
            Ls.append(np.full(ok.sum(), li)); Cs.append(np.full(ok.sum(), cid))
            cid += 1
    return dict(A=np.concatenate(As), B=np.concatenate(Bs), N=np.concatenate(Ns), L=np.concatenate(Ls), C=np.concatenate(Cs))


def quant_layers(polys, vals, ramp, levels=12, bias=0.0, lines=False):
    """group polygons (list of (k,3)) by quantised value in [0, 1] -> one split Layer per level; ramp(v) -> RGB."""
    out = []
    if not len(polys):
        return out
    q = np.clip(np.round(np.asarray(vals) * levels).astype(int), 0, levels)
    for lv in np.unique(q):
        idx = np.where(q == lv)[0]
        col = ramp(lv / levels)
        out.append(Layer([polys[i] for i in idx], col, "fill", split=True, bias=bias))
        if lines:
            out.append(Layer([polys[i] for i in idx], col, "line", 0.5, closed=True, split=True, bias=bias + 0.05))
    return out


# ------------------------------------------------------------------ the module base
class MechWord(StrandWord):
    """strands (hand-off) + mechanic. Subclass: implement mech_build() and mech_layers(t); use self.progress(t) (0..1)."""
    FA = 0.52                 # share of `form` taken by the arriving strands
    FB = 0.42                 # share of `exit` taken by the leaving strands
    MS = 0.30                 # mechanic starts at MS * form (HIDE modules: when the strands have finished)
    HIDE = False              # True: the mechanic moves the outline itself, strands only before / after it
    TUBE = 0.0
    LINE_W = 1.4
    ARC_PER_CAP = 0.5
    FILL = "none"
    DEFAULT_PALETTE = "ink"
    DEFAULT_TIMING = (2.2, 0.9, 1.9)
    LIB_CAMERA = dict(yaw=(26.0, -12.0, -28.0), pitch=(10.0, 9.0, 8.0), dist=(1060.0, 1040.0, 1080.0))   # None: demo camera
    STRAND_BIAS = 0.0

    def __init__(self, words=None, font=None, placement=None, entry="left", exit="right", timing=None, palette=None,
                 **params):
        super().__init__(words, font=font, placement=placement, entry=entry, exit=exit,
                         timing=timing if timing is not None else Timing(*self.DEFAULT_TIMING),
                         palette=palette if palette is not None else self.DEFAULT_PALETTE, **params)

    # ------------------------------------------------------------ build
    def _routes(self, tm):
        super()._routes(Timing(tm.form * self.FA, tm.hold, tm.exit * self.FB))

    def build(self):
        super().build()
        tm = self.timing
        self.t_swap = float(self.Tin.max()) + 0.01            # every strand has drawn its arc
        self.t_out0 = self.duration - tm.exit * self.FB      # the strands start to leave
        self.t_c = self.t_out0
        self.t_ms = self.t_swap if self.HIDE else self.MS * tm.form
        self.eye_hint = None
        self.mech_build()

    def mech_build(self):
        pass

    # ------------------------------------------------------------ timeline
    def progress(self, t):
        tm = self.timing
        if t < tm.form:
            return clamp((t - self.t_ms) / max(tm.form - self.t_ms, 1e-6))
        if t < tm.form + tm.hold:
            return 1.0
        return clamp(1.0 - (t - tm.form - tm.hold) / max(self.t_c - tm.form - tm.hold, 1e-6))

    def face_alpha(self, t):
        """solid face: fades in as the strands finish, out as the mechanic has folded back (before the strands leave)."""
        tm = self.timing
        if t < tm.form + tm.hold:
            return ss((t - 0.7 * self.t_swap) / (0.3 * self.t_swap + 1e-6))
        coll = max(self.t_c - tm.form - tm.hold, 1e-6)
        return 1.0 - ss((t - self.t_c + 0.34 * coll) / (0.34 * coll))

    def mech_on(self, t):
        return (not self.HIDE) or (self.t_swap <= t < self.t_out0)

    def windows(self, t):
        tm = self.timing
        te = t - (self.duration - tm.exit * self.FB)
        if te < 0:
            return [sl.window(t)[0] for sl in self.in_slide]
        return [sl.window(te)[0] for sl in self.out_slide]

    def strands_on(self, t):
        return (not self.HIDE) or t < self.t_swap or t >= self.t_out0

    # ------------------------------------------------------------ frame
    def draw_strands(self, wins, t):
        return strand_layers(wins, self.cols, [self.w] * self.N, 0.0, None, bias0=self.STRAND_BIAS)

    def mech_layers(self, t):
        return []

    def layers(self, t):
        t = min(max(t, 0.0), self.duration)
        L = []
        if self.strands_on(t):
            L += self.draw_strands(self.windows(t), t)
        if self.mech_on(t):
            L += self.mech_layers(t)
        return L

    # ------------------------------------------------------------ I/O styles (phase 3a)
    STYLE_MECH = 0.45         # with a non-lines form-in / form-out style: share of the window kept for the mechanic

    def reveal_layers(self, q, t_real=None):
        """flat formed word (strands finished, face solid) at q = 0, the full mechanic (the hold) at q = 1."""
        return self.raw_layers(self.t_swap + q * (self.timing.form - self.t_swap), t_real)

    def conceal_layers(self, q, t_real=None):
        """the hold at q = 0, the mechanic folded back (face still solid, before the strands leave) at q = 1."""
        tm = self.timing
        coll = max(self.t_c - tm.form - tm.hold, 1e-6)
        return self.raw_layers(tm.form + tm.hold + q * (self.t_c - 0.34 * coll - tm.form - tm.hold), t_real)

    # ------------------------------------------------------------ helpers
    def eye_at(self, t):
        if self.eye_hint is not None:
            return np.asarray(self.eye_hint(t), float)
        return self.placement.origin - self.placement.n * 1100.0

    def cons2(self):
        """all contours (local 2D) of the laid-out word."""
        return [c for l in self.lay["letters"] for c in l["cons"]]

    def lift(self, P2, z=0.0, scale=1.0, shift=(0.0, 0.0)):
        pl = self.placement
        P2 = np.asarray(P2, float)
        return pl.origin + (P2[..., 0:1] * scale + shift[0]) * pl.u + (P2[..., 1:2] * scale + shift[1]) * pl.v + z * pl.n

    def pose(self, P2, yb, th, d=0.0, shift=0.0):
        """local 2D points hinged about the horizontal axis at local y = yb (th = 0: in the placement plane; th > 0 tips the
        top of the type away from the viewer, th = pi/2 lying on the floor with the type's top pointing away). d = depth
        behind the face (th = 0: along +n; th = pi/2: down, i.e. along +v = into the floor). Returns world points."""
        pl = self.placement
        P2 = np.asarray(P2, float)
        up = yb - P2[..., 1:2]                                   # height above the hinge (positive up)
        c, s = math.cos(th), math.sin(th)
        return pl.origin + P2[..., 0:1] * pl.u + (yb - up * c + d * s) * pl.v + (up * s + d * c + shift) * pl.n

    def row_baselines(self):
        """baseline y (local, down) of the text row each letter sits on."""
        cap, n = self.lay["cap"], len(self.lines)
        blockH = self.lay["size"][1]
        base = [-0.5 * blockH + cap + i * cap * 1.22 for i in range(n)]
        out = []
        for l in self.lay["letters"]:
            ymax = max(float(c[:, 1].max()) for c in l["cons"])
            out.append(base[int(np.argmin([abs(ymax - b) for b in base]))])
        return out

    def hold_pose(self, dist=900.0):
        pl = self.placement
        return pl.origin - pl.n * dist, pl.origin.copy()

    def library_camera(self, seq):
        c = self.LIB_CAMERA
        if c is None:
            return None
        T = seq.duration + seq.tail
        tm = self.timing
        tk = [0.0, tm.form + 0.5 * tm.hold, T]
        keys = [(tk[i], c["yaw"][i], c["pitch"][i], c["dist"][i]) for i in range(3)]
        pl = self.placement
        return ArcCamera(pl.origin + pl.n * c.get("cz", 0.0) + pl.v * c.get("cy", 0.0), keys)
