"""(app copy of 13_camera/v003/local/cam3d.py; only ROOT is derived from this file's location) cam3d: shared 2.5D/3D perspective camera over the 2D type modules (13_camera v003: v002 FlowPath + holds / reversals / horizon lock).

World: the 960x540 frame plane sits at z = 0, x right, y down, +z AWAY from the viewer. Layers live on planes in
that world; one pinhole Camera(eye, target, roll, F) projects them. The HOME camera (eye (480,270,-F), target
(480,270,0), roll 0) maps every z = 0 point to exactly its 2D frame position, so a module that emits frame-px polygons
runs unchanged at home and every other camera is a real perspective move over it.

Screen convention: x right, y down, as in the frame. Roll > 0 turns the image clockwise on screen.
Yaw/pitch (view direction): yaw 0 looks along +z, yaw > 0 turns toward +x; pitch > 0 looks DOWN (+y).
Plane yaw/pitch (plane_rot): pitch > 0 tips the top edge away (a card falling back to the floor); yaw > 0 brings
the right edge toward the viewer.

Drawing is ol.fill / ol.lines at supersample S (vector, sharp at any punch-in).
"""
import os, sys, math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")).replace("\\", "/") + "/"
ENGINE = ROOT + "11_mathtype/engine/dimtype"
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)
import ol  # noqa: E402

W, H = 960, 540
CEN = np.array([W / 2.0, H / 2.0])
F0 = 1000.0           # default focal length (px)
FIT_HK, FIT_FILLK = 0.8, 1.0      # camera-module framing constants (WordInfo.fit / follow_camera): height share of a word box, fill multiplier


def set_world(w=960, h=540):
    """change the world frame plane (default 960x540) for this process: W, H, CEN (in place) and the framing constants.
    A portrait plane (h > w, e.g. 540x960) frames words by their full height and fills more of the frame. Call it again in
    every worker process (the state is module-level). Returns the previous (w, h)."""
    global W, H, FIT_HK, FIT_FILLK
    prev = (W, H)
    W, H = int(w), int(h)
    CEN[0], CEN[1] = W / 2.0, H / 2.0
    FIT_HK, FIT_FILLK = (0.8, 1.0) if W >= H else (0.95, 1.35)
    return prev


class world:
    """context manager: `with cam3d.world(540, 960): ...` runs with a portrait frame plane, then restores the old one."""

    def __init__(self, w, h):
        self.size, self.prev = (w, h), None

    def __enter__(self):
        self.prev = set_world(*self.size)
        return self

    def __exit__(self, *a):
        set_world(*self.prev)


# ------------------------------------------------------------------ small math
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def view_dir(yaw, pitch=0.0):
    """unit view direction: yaw 0 -> +z, yaw > 0 toward +x; pitch > 0 looks down (+y)."""
    return np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])


def orbit_eye(center, dist, yaw, pitch=0.0):
    """eye position `dist` from `center`, looking at it along view_dir(yaw, pitch)."""
    return np.asarray(center, float) - dist * view_dir(yaw, pitch)


# ------------------------------------------------------------------ camera
class Camera:
    """pinhole camera. eye/target in world units, roll in radians, F = focal length in px."""

    def __init__(self, eye, target, roll=0.0, F=F0, right=None, right_w=1.0):
        """right (optional, v002): world vector that fixes the image's right direction (projected perpendicular to the view
        axis). Default None = horizon-level (right = down x forward), which spins the image when the view is near vertical;
        pass (1, 0, 0) for a steep dive/plan view so the image stays put.
        right_w (v003): 0..1 blend between horizon-level (0) and the hint (1), so the hint can be faded in / out with no
        jump (a hard None <-> vector switch is a roll jump). `image_roll` = signed angle (rad) of the final image-right
        from the horizon-level image-right (explicit roll included): 0 for a plain horizon-level camera."""
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.roll = float(roll)
        self.F = float(F)
        f = _unit(self.target - self.eye)
        if np.linalg.norm(f) < 0.5:
            f = np.array([0.0, 0.0, 1.0])
        # r = down x f: for the home camera r = +x, d = +y. Continuous until f is exactly vertical,
        # where we fall back to r = +x (so keep a tiny horizontal component when looking straight down).
        r_h = np.cross([0.0, 1.0, 0.0], f)
        r = r_h
        if right is not None:
            rh = np.asarray(right, float)
            rh = rh - (rh @ f) * f
            if np.linalg.norm(rh) > 1e-6:
                w = min(max(float(right_w), 0.0), 1.0)
                if w >= 0.9999 or np.linalg.norm(r_h) < 1e-6:
                    r = rh
                elif w > 1e-4:
                    r = (1.0 - w) * _unit(r_h) + w * _unit(rh)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        r = _unit(r)
        d = np.cross(f, r)
        c, s = math.cos(self.roll), math.sin(self.roll)
        r2, d2 = r * c - d * s, r * s + d * c
        if np.linalg.norm(r_h) > 1e-6:                  # signed angle (about the view axis) horizon-level right -> actual right
            rhu = _unit(r_h)
            self.image_roll = math.atan2(float(f @ np.cross(rhu, r2)), float(rhu @ r2))
        else:
            self.image_roll = -self.roll
        self.f, self.r, self.d = f, r2, d2
        self.R = np.stack([r2, d2, f])          # world -> camera (u right, v down, z forward)

    @staticmethod
    def home(F=F0):
        return Camera((CEN[0], CEN[1], -F), (CEN[0], CEN[1], 0.0), 0.0, F)

    @staticmethod
    def orbit(center, dist, yaw, pitch=0.0, roll=0.0, F=F0):
        return Camera(orbit_eye(center, dist, yaw, pitch), center, roll, F)

    def to_cam(self, P):
        return (np.asarray(P, float) - self.eye) @ self.R.T

    def project(self, P):
        """world (n,3) -> (xy (n,2) frame px, depth (n,)). Points with depth <= 0 give inf/garbage; clip first."""
        Q = self.to_cam(np.atleast_2d(P))
        z = Q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = CEN + self.F * Q[:, :2] / z[:, None]
        return xy, z

    def scale_at(self, P):
        """px per world unit (perpendicular to the view) at world point P."""
        return self.F / max(float(self.to_cam(np.atleast_2d(P))[0, 2]), 1e-9)


# ------------------------------------------------------------------ planes
class Plane:
    """frame point (x, y) -> origin + (x - px) * u + (y - py) * v. pivot (px, py) is the frame point sitting at origin."""

    def __init__(self, origin, u=(1, 0, 0), v=(0, 1, 0), pivot=(0.0, 0.0)):
        self.origin = np.asarray(origin, float)
        self.u = np.asarray(u, float)
        self.v = np.asarray(v, float)
        self.pivot = np.asarray(pivot, float)

    def lift(self, P2):
        P2 = np.asarray(P2, float) - self.pivot
        return self.origin + P2[:, 0:1] * self.u + P2[:, 1:2] * self.v

    @property
    def normal(self):
        return _unit(np.cross(self.u, self.v))


def plane_z(z):
    """the frame plane pushed to depth z (wall facing the home camera)."""
    return Plane((0.0, 0.0, z), (1, 0, 0), (0, 1, 0))


def plane_floor(y, z0=0.0, h=None):
    """frame lying on a floor at world height y: frame bottom edge at depth z0, frame top edge at z0 + h (farther).
    A top-down camera with roll 0 sees it upright."""
    h = H if h is None else h
    return Plane((0.0, y, z0 + h), (1, 0, 0), (0, 0, -1))


def plane_rot(origin, yaw=0.0, pitch=0.0, pivot=None):
    """frame plane rotated about its pivot point (default frame centre), which is placed at `origin`.
    pitch tips the top edge away from the viewer (pi/2 = lying flat), then yaw about world y (> 0: right edge toward viewer)."""
    pivot = (W / 2.0, H / 2.0) if pivot is None else pivot
    Ry = _roty(yaw)
    Rx = _rotx(-pitch)
    u = Ry @ (Rx @ np.array([1.0, 0, 0]))
    v = Ry @ (Rx @ np.array([0, 1.0, 0]))
    return Plane(origin, u, v, pivot)


# ------------------------------------------------------------------ layers
class Layer:
    """polys: list of (n,2) frame-px polygons (lifted by `plane`, default plane_z(z)) or (n,3) world polygons.
    kind 'fill' (even-odd, so a letter's counters stay open when its contours are in ONE layer) or 'line'.
    width = line width in px at home depth (scaled by F/depth at render, clamped). closed: line contours closed.
    split: every polygon is its own depth-sorted item (use for prism faces); default one item per layer.
    bias: world units toward the viewer added to the sort depth (lines default 0.25 so an outline sits on its fill).
    union: fill every polygon on its own (overlapping contours merge instead of cancelling, as with non-zero winding). Set
    automatically when `polys` is a core.ConList with .union (a letter whose contours all wind the same way: pixel fonts)."""

    def __init__(self, polys, color, kind="fill", width=1.0, plane=None, z=0.0, alpha=1.0, closed=True, split=False,
                 bias=None, name="", union=None):
        self.union = bool(getattr(polys, "union", False)) if union is None else bool(union)
        if isinstance(polys, np.ndarray) and polys.ndim == 2:
            polys = [polys]
        polys = [np.asarray(p, float) for p in polys if p is not None and len(p) > (2 if kind == "fill" else 1)]
        if polys and polys[0].shape[1] == 2:
            pl = plane if plane is not None else plane_z(z)
            polys = [pl.lift(p) for p in polys]
        self.polys = polys
        self.pts = np.concatenate(polys) if polys else np.zeros((0, 3))
        self.offs = np.r_[0, np.cumsum([len(p) for p in polys])].astype(int)
        self.color = tuple(int(c) for c in color)
        self.kind, self.width, self.alpha = kind, float(width), float(alpha)
        self.closed, self.split, self.name = closed, split, name
        self.bias = (0.25 if kind == "line" else 0.0) if bias is None else float(bias)


# ------------------------------------------------------------------ clipping (vectorised Sutherland-Hodgman)
def clip_poly(P, d):
    """clip closed polygon P (n,k) to the half-space d >= 0 (d = signed distance per vertex, linear in P).
    Intersections interpolate every column of P. Returns (m,k) or None."""
    ins = d >= 0
    if ins.all():
        return P
    if not ins.any():
        return None
    Pp, dp, ip = np.roll(P, 1, 0), np.roll(d, 1), np.roll(ins, 1)
    cross = ins != ip
    den = dp - d
    den[~cross] = 1.0
    t = dp / den
    I = Pp + (P - Pp) * t[:, None]
    out = np.stack([I, P], 1)[np.stack([cross, ins], 1)]
    return out if len(out) >= 3 else None


def clip_line(P, d, closed):
    """clip a polyline to d >= 0. Returns a list of open polylines (closed contours come back as open pieces)."""
    ins = d >= 0
    if ins.all():
        return [P] if not closed else [np.vstack([P, P[:1]])]
    if not ins.any():
        return []
    if closed:                        # start at an outside vertex so the contour opens cleanly
        k = int(np.argmin(ins))
        P, d, ins = np.roll(P, -k, 0), np.roll(d, -k), np.roll(ins, -k)
        P, d, ins = np.vstack([P, P[:1]]), np.r_[d, d[0]], np.r_[ins, ins[0]]
    out, cur = [], []
    for i in range(len(P) - 1):
        a, b, ia, ib = P[i], P[i + 1], ins[i], ins[i + 1]
        if ia and ib:
            if not cur:
                cur = [a]
            cur.append(b)
        elif ia and not ib:
            t = d[i] / (d[i] - d[i + 1])
            if not cur:
                cur = [a]
            cur.append(a + (b - a) * t)
            out.append(np.array(cur)); cur = []
        elif ib and not ia:
            t = d[i] / (d[i] - d[i + 1])
            cur = [a + (b - a) * t, b]
    if cur:
        out.append(np.array(cur))
    return [o for o in out if len(o) > 1]


# ------------------------------------------------------------------ renderer
def render_layers(im, layers, cam, S, near=6.0, lw_clamp=(0.35, 4.0), guard=3000.0, stats=None):
    """draw `layers` as seen by `cam` onto canvas `im` (ol.canvas at scale S). Every layer is transformed to camera
    space, clipped against the near plane (z = near), projected, depth sorted far -> near (stable: author order holds at
    equal depth) and drawn with ol.fill / ol.lines. Sort depth = unclipped mean camera depth, so clipping never reorders
    parallel copies. Line width = layer.width * clamp(F / depth, lw_clamp). Polygons are also clipped to a +-guard px
    box around the frame so huge punch-ins stay numerically safe."""
    Hs, Ws = im.shape[:2]
    Wf, Hf = Ws / S, Hs / S
    cen = np.array([Wf / 2.0, Hf / 2.0])
    R, eye, F = cam.R, cam.eye, cam.F
    items = []
    nclip = 0
    for L in layers:
        if L.pts.shape[0] == 0:
            continue
        Q = (L.pts - eye) @ R.T
        zc = Q[:, 2]
        if zc.max() < near:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xy_all = cen + F * Q[:, :2] / zc[:, None]
        npoly = len(L.polys)
        groups = [(i, i + 1) for i in range(npoly)] if L.split else [(0, npoly)]
        for g0, g1 in groups:
            a, b = L.offs[g0], L.offs[g1]
            key = float(zc[a:b].mean()) - L.bias
            polys = []                     # (screen xy, mean depth, closed)
            for pi in range(g0, g1):
                o0, o1 = L.offs[pi], L.offs[pi + 1]
                z = zc[o0:o1]
                if z.min() >= near:
                    xy, zz = xy_all[o0:o1], z
                    pieces = [(xy, float(zz.mean()), L.closed or L.kind == "fill")]
                else:
                    nclip += 1
                    Qp = Q[o0:o1]
                    if L.kind == "fill":
                        c = clip_poly(Qp, Qp[:, 2] - near)
                        pieces = []
                        if c is not None:
                            pieces = [(cen + F * c[:, :2] / c[:, 2:3], float(c[:, 2].mean()), True)]
                    else:
                        pieces = []
                        for c in clip_line(Qp, Qp[:, 2] - near, L.closed):
                            pieces.append((cen + F * c[:, :2] / c[:, 2:3], float(c[:, 2].mean()), False))
                for xy, zm, cl in pieces:
                    lo, hi = xy.min(0), xy.max(0)
                    if hi[0] < 0 or hi[1] < 0 or lo[0] > Wf or lo[1] > Hf:
                        continue
                    if L.kind == "fill" and (lo.min() < -guard or hi[0] > Wf + guard or hi[1] > Hf + guard):
                        for ax, sg, lim in ((0, 1, -guard), (0, -1, Wf + guard), (1, 1, -guard), (1, -1, Hf + guard)):
                            if xy is None:
                                break
                            xy = clip_poly(xy, sg * (xy[:, ax] - lim))
                        if xy is None:
                            continue
                    polys.append((xy, zm, cl))
            if polys:
                items.append((key, L, polys))
    order = sorted(range(len(items)), key=lambda i: -round(items[i][0], 2))      # stable
    for i in order:
        _, L, polys = items[i]
        _draw_item(im, L, polys, S, F, lw_clamp)
    if stats is not None:
        stats.update(items=len(items), clipped=nclip)
    return im


def _draw_item(im, L, polys, S, F, lw_clamp):
    target, shift = im, None
    if L.alpha < 0.998:
        if L.alpha <= 0.002:
            return
        allp = np.concatenate([p[0] for p in polys])
        pad = (L.width * lw_clamp[1] + 2)
        x0 = int(max(0, math.floor(allp[:, 0].min() - pad))); y0 = int(max(0, math.floor(allp[:, 1].min() - pad)))
        x1 = int(min(im.shape[1] / S, math.ceil(allp[:, 0].max() + pad))); y1 = int(min(im.shape[0] / S, math.ceil(allp[:, 1].max() + pad)))
        if x1 <= x0 or y1 <= y0:
            return
        base = im[y0 * S:y1 * S, x0 * S:x1 * S]
        target, shift = base.copy(), np.array([x0, y0], float)
    if L.kind == "fill":
        pl = [p[0] - (shift if shift is not None else 0) for p in polys]
        if L.union and len(pl) > 1:
            for p in pl:
                ol.fill(target, [p], L.color, S)
        else:
            ol.fill(target, pl, L.color, S)
    else:
        groups = {}
        for xy, zm, cl in polys:
            sc = min(max(F / max(zm, 1e-6), lw_clamp[0]), lw_clamp[1])
            th = max(1, int(round(L.width * sc * S)))
            groups.setdefault((th, cl), []).append(xy - (shift if shift is not None else 0))
        for (th, cl), pl in groups.items():
            ol.lines(target, pl, L.color, S, th / S, closed=cl)
    if shift is not None:
        import cv2
        cv2.addWeighted(target, L.alpha, base, 1 - L.alpha, 0, dst=base)


def frame_from_layers(layers, cam, ground, S=2, **kw):
    """convenience: ground-coloured canvas -> render_layers -> downsampled RGB (H, W, 3) uint8."""
    im = ol.canvas(W, H, S, ground)
    render_layers(im, layers, cam, S, **kw)
    return ol.down(im, S)


# ------------------------------------------------------------------ easing
def _ease(name, u):
    u = min(1.0, max(0.0, u))
    if name == "linear":
        return u
    if name == "io":
        return ol.ease_io(u, 3.0)
    if name == "in":
        return u ** 3
    if name == "out":
        return 1 - (1 - u) ** 3
    if name == "smooth":
        return u * u * (3 - 2 * u)
    raise ValueError(name)


def spring_resp(tau, zeta=0.7, omega=18.0):
    """damped spring step response (tau in seconds): overshoots 1 then settles."""
    if tau <= 0:
        return 0.0
    s = math.sqrt(1 - zeta * zeta)
    return 1 - math.exp(-zeta * omega * tau) * (math.cos(omega * s * tau) + zeta / s * math.sin(omega * s * tau))


# ------------------------------------------------------------------ paths
class Key:
    """camera keyframe at time t (s). `ease` shapes the segment that STARTS at this key (towards the next key):
    'linear' | 'io' | 'in' | 'out' | 'smooth' | float p (ease-in-out with power p) | callable u->u | 'cr'
    ('cr' = C1 Catmull-Rom through neighbouring keys, no stop at this key: use for orbits / long arcs).
    spring=(zeta, omega) replaces the ease on that segment with a damped-spring overshoot that lands on the next key."""

    def __init__(self, t, eye, target, roll=0.0, F=F0, ease="io", spring=None):
        self.t, self.eye, self.target = float(t), np.asarray(eye, float), np.asarray(target, float)
        self.roll, self.F, self.ease, self.spring = float(roll), float(F), ease, spring

    @staticmethod
    def orbit(t, center, dist, yaw, pitch=0.0, roll=0.0, F=F0, ease="io", spring=None, target=None):
        """key from look-at orbit parameters (eye = center - dist * view_dir). target defaults to center."""
        return Key(t, orbit_eye(center, dist, yaw, pitch), center if target is None else target, roll, F, ease, spring)


def track(target_fn, lag=0.0, n=14):
    """-> fn(t) giving a (3,) world point that follows target_fn(t) with a stateless exponential lag (seconds):
    the mean of target_fn over the last ~4*lag s, weights exp(-s/lag). lag = 0 returns target_fn unchanged."""
    if lag <= 1e-6:
        return lambda t: np.asarray(target_fn(t), float)
    ss = np.linspace(0.0, 4.0 * lag, n)
    w = np.exp(-ss / lag); w /= w.sum()

    def fn(t):
        return sum(wi * np.asarray(target_fn(max(0.0, t - s)), float) for wi, s in zip(w, ss))
    return fn


def _hermite(p0, p1, m0, m1, u):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


class CamPath:
    """keyframed camera. at(t) -> Camera.
    eye/target lerp (or Hermite for 'cr' segments), roll lerp, F interpolated in log space.
    tracker: fn(t) -> world point (see track()). track_w (float or fn(t)) blends the target from the keyed target to
    the tracked point; follow (0..1) also moves the eye by the same delta (0 = camera pans to follow, 1 = camera
    trucks and keeps its view direction).
    drift_px: subtle handheld drift (low-frequency sine noise, amplitude in px at the target depth; 0 = off);
    drift_hz sets its speed, drift_roll (rad) adds a little roll wobble."""

    def __init__(self, keys, tracker=None, track_w=1.0, follow=0.0, drift_px=0.0, drift_hz=0.45, drift_roll=0.0, seed=0):
        self.keys = sorted(keys, key=lambda k: k.t)
        self.tracker, self.track_w, self.follow = tracker, track_w, follow
        self.drift_px, self.drift_hz, self.drift_roll = drift_px, drift_hz, drift_roll
        rng = np.random.RandomState(seed)
        self._ph = rng.uniform(0, 2 * np.pi, (4, 3))
        self._fr = rng.uniform(0.6, 1.5, (4, 3))
        self._t = np.array([k.t for k in self.keys])

    # --- channel stack: eye(3) target(3) roll logF
    @staticmethod
    def _vec(k):
        return np.r_[k.eye, k.target, k.roll, math.log(k.F)]

    def _tangent(self, i, V):
        n = len(self.keys)
        if i <= 0:
            return np.zeros_like(V[0])
        if i >= n - 1:
            return np.zeros_like(V[0])
        return (V[i + 1] - V[i - 1]) / max(self._t[i + 1] - self._t[i - 1], 1e-9)

    def _state(self, t):
        ks = self.keys
        if t <= ks[0].t:
            return self._vec(ks[0])
        if t >= ks[-1].t:
            return self._vec(ks[-1])
        i = int(np.searchsorted(self._t, t, "right") - 1)
        a, b = ks[i], ks[i + 1]
        dt = b.t - a.t
        u = (t - a.t) / dt
        va, vb = self._vec(a), self._vec(b)
        if a.spring is not None:
            z, om = a.spring
            s = spring_resp(u * dt, z, om) / max(spring_resp(dt, z, om), 1e-6)
            return va + (vb - va) * s
        e = a.ease
        if e == "cr":
            V = [self._vec(k) for k in ks]
            m0 = self._tangent(i, V) * dt
            m1 = self._tangent(i + 1, V) * dt
            return _hermite(va, vb, m0, m1, u)
        if callable(e):
            s = e(u)
        elif isinstance(e, (int, float)):
            s = ol.ease_io(u, float(e))
        else:
            s = _ease(e, u)
        return va + (vb - va) * s

    def _drift(self, t, cam_eye, cam_target, F):
        dist = max(np.linalg.norm(cam_target - cam_eye), 1.0)
        k = dist / F                                          # world units per px at the target
        w = 2 * np.pi * self.drift_hz * t
        n = (np.sin(w * self._fr + self._ph)).sum(0) / 4.0     # each of 3 channels, ~[-1,1]
        n2 = (np.sin(w * self._fr[::-1] * 0.8 + self._ph[::-1] + 1.3)).sum(0) / 4.0
        return n * self.drift_px * k, n2[0] * self.drift_roll

    def at(self, t):
        s = self._state(t)
        eye, tgt, roll, F = s[0:3].copy(), s[3:6].copy(), s[6], math.exp(s[7])
        if self.tracker is not None:
            w = self.track_w(t) if callable(self.track_w) else self.track_w
            if w > 0:
                tp = np.asarray(self.tracker(t), float)
                delta = (tp - tgt) * w
                tgt = tgt + delta
                eye = eye + self.follow * delta
        if self.drift_px > 0 or self.drift_roll > 0:
            cam0 = Camera(eye, tgt, roll, F)
            dv, dr = self._drift(t, eye, tgt, F)
            off = cam0.r * dv[0] + cam0.d * dv[1] + cam0.f * dv[2] * 0.5
            eye, tgt, roll = eye + off, tgt + off, roll + dr
        return Camera(eye, tgt, roll, F)


def blend_cams(a, b, u):
    """lerp two cameras (eye, target, roll, log F)."""
    return Camera(a.eye + (b.eye - a.eye) * u, a.target + (b.target - a.target) * u, a.roll + (b.roll - a.roll) * u,
                  math.exp(math.log(a.F) + (math.log(b.F) - math.log(a.F)) * u))


# ------------------------------------------------------------------ motion blur helper
def motion_px(cam_a, cam_b, probes):
    """max screen displacement (px) of world probe points (m,3) between two cameras (points behind either: ignored)."""
    xa, za = cam_a.project(probes)
    xb, zb = cam_b.project(probes)
    ok = (za > 1.0) & (zb > 1.0)
    if not ok.any():
        return 400.0
    d = np.linalg.norm(np.clip(xb[ok], -3000, 3000) - np.clip(xa[ok], -3000, 3000), axis=1)
    return float(d.max())


def auto_sub(cam_fn, f, fps, probes, px_per_sub=5.0, max_sub=10, min_sub=1):
    """motion-blur sub-frame count for frame f from how far probe points travel across one frame."""
    t = f / fps
    d = motion_px(cam_fn(t - 0.5 / fps), cam_fn(t + 0.5 / fps), probes)
    return int(min(max_sub, max(min_sub, math.ceil(d / px_per_sub))))


# ================================================================== FlowPath: ONE continuous camera move (v002)
from scipy.interpolate import CubicSpline  # noqa: E402


def smoother(u):
    """quintic smootherstep 0..1 (C2 at both ends)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _ism(a):
    """integral of smoother(clip(x, 0, 1)) from 0 to a (a may exceed 1)."""
    a = np.asarray(a, float)
    ac = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0, 0.0, np.where(a <= 1, ac ** 6 - 3 * ac ** 5 + 2.5 * ac ** 4, 0.5 + (a - 1.0)))


class _Chan:
    """scalar channel over the path phase u in [0, 1]: constant | callable(u) | knots [(u, value), ...] (C2 spline).
    log=True interpolates log(value) (use for focal length)."""

    def __init__(self, spec, default, log=False):
        spec = default if spec is None else spec
        self.log = log
        self.fn, self.c, self.cs, self.k2 = None, None, None, None
        if callable(spec):
            self.fn = spec
        elif np.isscalar(spec):
            self.c = float(spec)
        else:
            k = np.array([(float(u), float(v)) for u, v in spec], float)
            k = k[np.argsort(k[:, 0])]
            v = np.log(k[:, 1]) if log else k[:, 1]
            self.u0, self.u1 = k[0, 0], k[-1, 0]
            if len(k) > 2:
                self.cs = CubicSpline(k[:, 0], v, bc_type="natural")
            elif len(k) == 1:
                self.c = float(k[0, 1])
            else:
                self.k2 = (k, v)

    def __call__(self, u):
        if self.fn is not None:
            return float(self.fn(u))
        if self.c is not None:
            return self.c
        if self.cs is None:                      # two knots: smoothstep blend between them (C1)
            k, v = self.k2
            w = float(smoother((u - k[0, 0]) / max(k[1, 0] - k[0, 0], 1e-9)))
            x = v[0] + (v[1] - v[0]) * w
        else:
            x = float(self.cs(min(max(u, self.u0), self.u1)))
        return math.exp(x) if self.log else x


class FlowPath:
    """One continuous camera move. at(t) -> Camera for t in [t0, t1] (clamped outside).

    FlowPath(points, t0, t1, speed=None, ease_in=0.6, ease_out=0.8, look=None, up_roll=None, F=F0,
             times=None, blend=0.8, look_lag=0.25, look_u=None, lead=300.0, v_start=0.0, v_end=0.0, right=None)

    points   (n, 3) world control points the EYE passes through, in order. The eye curve is a natural cubic spline
             (C2: position, velocity and acceleration all continuous) on chord-length parameter, then re-parameterised
             by arc length, so speed is exactly what the profile says (no speed wobble from point spacing).
    t0, t1   start / end time (s).
    speed    None | float | list of n-1 positive multipliers, one per segment (segment i = points[i] -> points[i+1]).
             The multiplier is blended smoothly (quintic) across each point where it changes, over `blend` x the shorter of
             the two adjacent runs of equal multiplier, so speed changes smoothly, never steps and never reaches zero.
    times    optional list of n times for the control points (increasing; entries may be None = unconstrained; first and
             last default to t0 / t1). Overrides `speed`: all segments between two constrained points share one speed
             multiplier (blended smoothly between groups), fitted so the eye passes each constrained point at its time
             (within a few frames at worst). Use a handful of landmark times, not one per point.
    ease_in / ease_out   seconds. Speed ramps (smootherstep) from v_start*cruise to cruise over the first ease_in s and
             from cruise to v_end*cruise over the last ease_out s. Nothing else slows the camera. v_start = 0 starts from
             rest; v_start = 0.5 starts already moving (use when the piece should begin mid-move).
    look     None / 'ahead'  -> look at the eye `lead` world units further along the path (heading follows travel);
             (m, 3) array of points -> another smooth C2 curve through them, spread over the path phase u = s / length
             at `look_u` (default evenly);
             callable f(t) -> (3,) world point (a tracker, e.g. growth front of a module) -> smoothed by a critically
             damped filter (steady-state lag = `look_lag` s behind a tracker moving at constant speed), so it can never
             jerk; keep the tracked point well away from the eye (the look direction is its offset from the eye). The callable may close over the FlowPath (path.ahead, path.eye_at).
    up_roll  roll (rad) as a function of phase u: None (0) | float | callable(u) | [(u, rad), ...] knots (C2 spline).
    F        focal length (px), same forms as up_roll (knots interpolate log F).
    right    None | (3,) vector | callable(t) -> (3,) or None: world vector fixing the image's right direction (see
             Camera). Use (1, 0, 0) for a stretch where the view is near vertical (dive down a shaft, plan view) so the
             image cannot spin; blend it back to None (horizon-level) for oblique views.
    --- v003 additions -------------------------------------------------------------------------------------------
    holds      list of (i, t, dur) | (i, t, dur, ease) | dict(i=, t=, dur=, ease=): the eye eases to REST at control point
               index i (negative = from the end), stays for dur s centred on time t, then eases out. i = 0: rest at the start
               (t ignored, rests t0 .. t0+dur); i = n-1: rest at the end (t ignored, rests t1-dur .. t1: the path ends in a
               hold). ease = seconds of deceleration/acceleration each side (float or (in, out)); default `hold_ease`. The
               speed profile between rests is smootherstep (velocity AND acceleration continuous), so holds never jolt. Look-at
               and roll also settle (their targets are filtered / functions of the phase, constant while the eye rests).
               The eye curve stays smooth through a hold (no cusp). Timing is exact: the eye is at point i from t-dur/2 to
               t+dur/2. Each stretch between rests runs at its own cruise speed (the per-segment `speed` multipliers still
               shape speed inside a stretch). `times` is not allowed together with holds / reversals (the hold times ARE the
               time anchors).
    reversals  list of (i, t) | (i, t, dur) | dict: travel REVERSES at control point i. Same as a hold (dur default 0:
               zero speed for an instant, no pause) but the eye curve has a CUSP there: the path splits into independent
               spline legs at i, so the incoming leg ends at i (arriving at rest along its own tangent) and the outgoing leg
               leaves i (from rest along its own tangent), in any direction. Speed is zero at the cusp so velocity stays
               continuous; heading never swings at speed. With reversals `look` must be a callable or a curve (the
               default 'ahead' look would flip 180 deg at the cusp).
    hold_ease  default ease seconds for holds / reversals (0.4); clipped so both ramps of a stretch fit inside it.
    drift_px   tiny low-frequency camera-plane drift (px at the look distance, ~0.8) that fades in only while the eye rests
               in a hold with dur > 0 (so a hold is not dead); drift_hz its frequency (0.35).
    roll       up_roll (above) now defaults to 0 = horizon lock. A non-zero roll curve is rate-limited to `roll_max_dps`
               (default 45 deg/s = 1.5 deg/frame at 30 fps) and then smoothed (critically damped, lag `roll_lag` s) so there is
               never a tilt-then-correct kink.
    right_w    None | float | callable(t) -> 0..1: blend of the `right` hint against horizon-level. Use it to fade a
               hint in / out smoothly (right=(1,0,0), right_w=lambda t: float(smoother((t-2.0)/0.8))). When right_w is
               None the hint applies fully whenever `right` returns a vector (v002 behaviour).
    Helpers: eye_at(t), tangent(t), ahead(t, dist), phase(t), speed_at(t), knot_times(), length, speed_report(fps)
             (v003: stop windows, heading jump, image roll, jerk), stops (the normalised hold / reversal list).
    """

    def __init__(self, points, t0, t1, speed=None, ease_in=0.6, ease_out=0.8, look=None, up_roll=None, F=F0,
                 times=None, blend=0.8, look_lag=0.25, look_u=None, lead=300.0, v_start=0.0, v_end=0.0, right=None,
                 right_w=None, holds=None, reversals=None, hold_ease=0.4, drift_px=0.0, drift_hz=0.35,
                 roll_lag=0.15, roll_max_dps=45.0):
        self._right, self._right_w = right, right_w
        self._drift_px, self._drift_hz = float(drift_px), float(drift_hz)
        self._roll_lag, self._roll_max = float(roll_lag), math.radians(float(roll_max_dps))
        P = np.asarray(points, float)
        assert P.ndim == 2 and P.shape[1] == 3 and len(P) >= 2, "points must be (n>=2, 3)"
        self.pts, self.t0, self.t1 = P, float(t0), float(t1)
        self.ease_in, self.ease_out = float(ease_in), float(ease_out)
        self.v_start, self.v_end = float(v_start), float(v_end)
        assert self.ease_in + self.ease_out <= self.t1 - self.t0 + 1e-9, "ease_in + ease_out longer than the move"
        self.blend, self.lead, self.look_lag = float(blend), float(lead), float(look_lag)
        n = len(P)
        self._n = n
        self._normalise_stops(holds, reversals, hold_ease, times)
        # ---- eye curve: natural cubic spline on chord length, then arc-length table. One chunk per cusp-free run of
        # points (chunks meet at the reversal points, where the eye is at rest); no reversals = ONE chunk (v002 exactly).
        cuts = [0] + sorted(st["i"] for st in self.stops if st["cusp"]) + [n - 1]
        self._chunks, knot_s, off = [], [0.0], 0.0
        for ia, ib in zip(cuts[:-1], cuts[1:]):
            Pc = P[ia:ib + 1]
            chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
            spl = CubicSpline(chord, Pc, bc_type="natural")
            pd = np.linspace(0.0, chord[-1], 500 * (len(Pc) - 1) + 1)
            Pd = spl(pd)
            sd = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))]
            self._chunks.append((off, float(sd[-1]), CubicSpline(sd, Pd, bc_type="natural")))
            knot_s += list(off + np.interp(chord[1:], pd, sd))
            off += float(sd[-1])
        self.length = float(off)
        self._cS = np.array([c[0] for c in self._chunks])
        self.knot_s = np.array(knot_s)
        if any(st["cusp"] for st in self.stops):
            assert callable(look) or (look is not None and not isinstance(look, str)),                 "reversals need an explicit look (callable or curve): 'ahead' would flip 180 deg at the cusp"
        # ---- speed multipliers
        nseg = n - 1
        seg = np.diff(self.knot_s)
        self._groups = None
        if times is not None and not self.stops:
            tl = list(times)
            assert len(tl) == n, "times: one entry per control point (None = free)"
            if tl[0] is None:
                tl[0] = self.t0
            if tl[-1] is None:
                tl[-1] = self.t1
            idx = [i for i, v in enumerate(tl) if v is not None]
            tv = [float(tl[i]) for i in idx]
            assert all(b_ > a_ for a_, b_ in zip(tv[:-1], tv[1:])), "times must increase"
            self._groups = [(idx[j], idx[j + 1], tv[j + 1] - tv[j]) for j in range(len(idx) - 1)]
            self._times_idx = (idx, tv)
            m = np.ones(nseg)
            for (i0, i1, dtg) in self._groups:
                m[i0:i1] = (self.knot_s[i1] - self.knot_s[i0]) / dtg
            m = m / m.mean()
        elif speed is None:
            m = np.ones(nseg)
        elif np.isscalar(speed):
            m = np.full(nseg, float(speed))
        else:
            m = np.asarray(speed, float)
            assert len(m) == nseg, f"speed needs {nseg} multipliers (one per segment)"
        assert np.all(m > 0), "speed multipliers must be > 0 (the camera never stops inside the move)"
        self._seg = seg
        self.mult = np.asarray(m, float).copy()
        self._build_profile()
        if self._groups is not None and not self.stops:
            idx, tv = self._times_idx
            for _ in range(30):                                   # fit the group multipliers so knot times land on `times`
                kt = self.knot_times()
                err = max(abs(kt[i] - v) for i, v in zip(idx, tv))
                if err < 0.004:
                    break
                for (i0, i1, dtg) in self._groups:
                    self.mult[i0:i1] = self.mult[i0:i1] * ((kt[i1] - kt[i0]) / dtg) ** 0.8
                self.mult /= self.mult.mean()
                self._build_profile()
        # ---- channels
        self._roll = _Chan(up_roll, 0.0)
        self._roll_zero = up_roll is None or (np.isscalar(up_roll) and float(up_roll) == 0.0)
        self._F = _Chan(F, F0, log=True)
        # ---- look-at
        self._look_mode = "ahead"
        if callable(look):
            self._look_mode = "track"
            self._look_fn, self._tk_t = look, None          # table built on first use, so look may close over this path
        elif look is not None and not isinstance(look, str):
            L = np.asarray(look, float)
            assert L.ndim == 2 and L.shape[1] == 3
            lu = np.linspace(0.0, 1.0, len(L)) if look_u is None else np.asarray(look_u, float)
            self._look_cs = CubicSpline(lu, L, bc_type="natural") if len(L) > 2 else None
            self._look_L, self._look_u = L, lu
            self._look_mode = "curve"

    # ------------------------------------------------------------ holds / reversals (v003)
    def _normalise_stops(self, holds, reversals, hold_ease, times):
        n, got = self._n, {}

        def add(spec, cusp):
            if isinstance(spec, dict):
                d = dict(spec)
            else:
                sp = list(spec) + [None] * 4
                d = dict(i=sp[0], t=sp[1], dur=sp[2], ease=sp[3])
            i = int(d["i"]) % n
            dur = d.get("dur")
            dur = 0.0 if (dur is None and cusp) else dur
            assert dur is not None, "a hold needs (i, t, dur)"
            dur = float(dur)
            e = d.get("ease")
            e = hold_ease if e is None else e
            ein, eout = (float(e[0]), float(e[1])) if isinstance(e, (tuple, list)) else (float(e), float(e))
            if i == 0:
                if dur <= 1e-9:
                    return
                a, b, cusp = self.t0, self.t0 + dur, False
            elif i == n - 1:
                if dur <= 1e-9:
                    return
                a, b, cusp = self.t1 - dur, self.t1, False
            else:
                assert d.get("t") is not None, f"stop at point {i} needs a time t"
                a, b = float(d["t"]) - 0.5 * dur, float(d["t"]) + 0.5 * dur
            assert i not in got, f"point {i} declared twice (hold + reversal): use one"
            got[i] = dict(i=i, a=a, b=b, ein=ein, eout=eout, cusp=bool(cusp), dur=dur)

        for h in (holds or []):
            add(h, False)
        for r in (reversals or []):
            add(r, True)
        self.stops = [got[i] for i in sorted(got)]
        assert not (self.stops and times is not None), "times cannot be combined with holds / reversals"
        for x, y in zip(self.stops[:-1], self.stops[1:]):
            assert y["a"] > x["b"] + 0.05, f"stops at points {x['i']} and {y['i']} overlap in time (need > 0.05 s of travel)"
        self._stop_win = []

    def _build_profile_stops(self):
        sg = np.linspace(0.0, self.length, 4001)
        wg = self._w(sg)
        th = np.r_[0.0, np.cumsum(0.5 * (1 / wg[1:] + 1 / wg[:-1]) * np.diff(sg))]
        self._sg, self._wg, self._th = sg, wg, th
        self._s_of_th = CubicSpline(th, sg, bc_type="natural")
        n, legs = self._n, []
        ts, i0, v0, up = self.t0, 0, self.v_start, self.ease_in
        for st in self.stops:
            if st["i"] == 0:
                ts, v0, up = st["b"], 0.0, st["eout"]
            elif st["i"] != n - 1:
                legs.append(dict(ts=ts, te=st["a"], i0=i0, i1=st["i"], v0=v0, v1=0.0, up=up, dn=st["ein"]))
                ts, i0, v0, up = st["b"], st["i"], 0.0, st["eout"]
        end = [st for st in self.stops if st["i"] == n - 1]
        if end:
            legs.append(dict(ts=ts, te=end[0]["a"], i0=i0, i1=n - 1, v0=v0, v1=0.0, up=up, dn=end[0]["ein"]))
        else:
            legs.append(dict(ts=ts, te=self.t1, i0=i0, i1=n - 1, v0=v0, v1=self.v_end, up=up, dn=self.ease_out))
        for L in legs:
            T = L["te"] - L["ts"]
            assert T > 0.05, "a stretch between rests is too short"
            if L["up"] + L["dn"] > 0.96 * T:                      # both ramps must fit: scale them down together
                k = 0.96 * T / (L["up"] + L["dn"])
                L["up"], L["dn"] = L["up"] * k, L["dn"] * k
        # declared windows: each rest +- half its ease (inside them the eye may be slower than 50% of cruise, even stopped)
        wins = []
        if self.stops and self.stops[0]["i"] == 0:
            wins.append((self.t0, legs[0]["ts"] + 0.5 * legs[0]["up"]))
        for a, b in zip(legs[:-1], legs[1:]):
            wins.append((a["te"] - 0.5 * a["dn"], b["ts"] + 0.5 * b["up"]))
        if end:
            wins.append((legs[-1]["te"] - 0.5 * legs[-1]["dn"], self.t1))
        self._stop_win, self._legs = wins, legs
        # speed shape g(t) per stretch (smootherstep ramps, 0 inside rests); theta(t) = reparametrised distance
        N = int(round((self.t1 - self.t0) * 1000.0)) + 1
        tg = np.linspace(self.t0, self.t1, N)
        g = np.zeros(N)
        for L in legs:
            m = (tg >= L["ts"]) & (tg <= L["te"])
            x = tg[m]
            a_ = (L["v0"] + (1.0 - L["v0"]) * smoother((x - L["ts"]) / L["up"])) if L["up"] > 1e-9 else np.ones_like(x)
            b_ = (1.0 - (1.0 - L["v1"]) * smoother((x - (L["te"] - L["dn"])) / L["dn"])) if L["dn"] > 1e-9 else np.ones_like(x)
            g[m] = a_ * b_
        G = np.r_[0.0, np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(tg))]
        th_t = np.zeros(N)
        theta_at = lambda i: float(np.interp(self.knot_s[i], sg, th))
        for j, L in enumerate(legs):
            ths, the = theta_at(L["i0"]), theta_at(L["i1"])
            Gs, Ge = float(np.interp(L["ts"], tg, G)), float(np.interp(L["te"], tg, G))
            L["c"] = (the - ths) / max(Ge - Gs, 1e-12)
            m = (tg >= L["ts"]) & (tg <= L["te"])
            th_t[m] = ths + L["c"] * (G[m] - Gs)
            nxt = legs[j + 1]["ts"] if j + 1 < len(legs) else self.t1 + 1.0
            th_t[(tg > L["te"]) & (tg < nxt)] = the
        th_t[tg < legs[0]["ts"]] = theta_at(0)
        self._tgs, self._th_t = tg, th_t
        self._thS = CubicSpline(tg, th_t, bc_type="natural")

    # ------------------------------------------------------------ speed profile
    def _g(self, t):
        a = (t - self.t0) / self.ease_in if self.ease_in > 1e-9 else 1e9
        b = (t - (self.t1 - self.ease_out)) / self.ease_out if self.ease_out > 1e-9 else -1e9
        return 1.0 - (1.0 - self.v_start) * (1.0 - float(smoother(a))) - (1.0 - self.v_end) * float(smoother(b))

    def _G(self, t):
        t = np.asarray(t, float)
        out = t - self.t0
        if self.ease_in > 1e-9:
            a = (t - self.t0) / self.ease_in
            out = out - (1.0 - self.v_start) * self.ease_in * (a - _ism(a))
        if self.ease_out > 1e-9:
            b = (t - (self.t1 - self.ease_out)) / self.ease_out
            out = out - (1.0 - self.v_end) * self.ease_out * _ism(b)
        return out

    def _w(self, s):
        """smoothly blended per-segment speed multiplier at arc length s (scalar or array)."""
        m, ks = self.mult, self.knot_s
        s = np.asarray(s, float)
        # runs of equal multiplier -> boundaries (knot index, run lengths either side)
        runs, i0 = [], 0
        for i in range(1, len(m) + 1):
            if i == len(m) or abs(m[i] - m[i0]) > 1e-9 * max(m[i0], 1.0):
                runs.append((i0, i))
                i0 = i
        w = np.full_like(s, m[0])
        for j in range(1, len(runs)):
            kb = runs[j][0]
            la = ks[runs[j - 1][1]] - ks[runs[j - 1][0]]
            lb = ks[runs[j][1]] - ks[runs[j][0]]
            h = 0.5 * self.blend * min(la, lb)
            w = w + (m[kb] - m[kb - 1]) * smoother((s - (ks[kb] - h)) / (2 * h))
        return w

    def _build_profile(self):
        if self.stops:
            return self._build_profile_stops()
        sg = np.linspace(0.0, self.length, 4001)
        wg = self._w(sg)
        th = np.r_[0.0, np.cumsum(0.5 * (1 / wg[1:] + 1 / wg[:-1]) * np.diff(sg))]       # theta(s) = int ds / w
        self._sg, self._wg, self._th = sg, wg, th
        self._s_of_th = CubicSpline(th, sg, bc_type="natural")
        self._c = th[-1] / float(self._G(self.t1))
        tg = np.linspace(self.t0, self.t1, 6001)
        self._tg, self._Gg = tg, self._G(tg)

    def s_of_t(self, t):
        t = min(max(float(t), self.t0), self.t1)
        if self.stops:
            return float(np.clip(self._s_of_th(min(max(float(self._thS(t)), 0.0), self._th[-1])), 0.0, self.length))
        th = min(max(self._c * float(self._G(t)), 0.0), self._th[-1])
        return float(np.clip(self._s_of_th(th), 0.0, self.length))

    def speed_at(self, t):
        """analytic eye speed (world units / s)."""
        t = min(max(float(t), self.t0), self.t1)
        if self.stops:
            return max(0.0, float(self._thS(t, 1))) * float(np.interp(self.s_of_t(t), self._sg, self._wg))
        return float(self._c * self._g(t) * np.interp(self.s_of_t(t), self._sg, self._wg))

    def t_of_s(self, s):
        th = float(np.interp(s, self._sg, self._th))
        if self.stops:                                   # inside a rest this returns the START of the rest
            return float(np.interp(th, self._th_t + 1e-9 * (self._tgs - self.t0), self._tgs))
        return float(np.interp(th / self._c, self._Gg, self._tg))

    def knot_times(self):
        """time at which the eye passes each control point."""
        return np.array([self.t_of_s(s) for s in self.knot_s])

    def phase(self, t):
        return self.s_of_t(t) / self.length

    # ------------------------------------------------------------ eye helpers
    def _E(self, s, d=0, side=0):
        """eye curve (d = 0) or unit-speed tangent (d = 1) at arc length s. At a cusp (reversal) `side` < 0 picks the
        incoming leg, >= 0 the outgoing one (position is identical either way)."""
        k = int(np.clip(np.searchsorted(self._cS, s, "right") - 1, 0, len(self._chunks) - 1))
        if side < 0 and k > 0 and abs(s - self._cS[k]) < 1e-9:
            k -= 1
        S0, L, spl = self._chunks[k]
        x = s - S0
        if x > L:                                          # past the end: continue straight along the end tangent
            return spl(L) + (x - L) * spl(L, 1) if d == 0 else spl(L, 1)
        return spl(max(x, 0.0), d)

    def _eye_extended(self, s):
        return self._E(s)

    def eye_at(self, t):
        return np.asarray(self._E(self.s_of_t(t)), float)

    def tangent(self, t):
        s = self.s_of_t(t)
        side = 0
        for st in self.stops:
            if st["cusp"] and abs(s - self.knot_s[st["i"]]) < 1e-6 * max(self.length, 1.0):
                side = -1 if t < 0.5 * (st["a"] + st["b"]) else 1
        return _unit(self._E(s, 1, side))

    def ahead(self, t, dist):
        """eye position `dist` world units further along the path (extrapolated past the end along the tangent)."""
        return np.asarray(self._eye_extended(self.s_of_t(t) + dist), float)

    # ------------------------------------------------------------ look-at
    def _build_tracker(self, fn):
        dt = 1.0 / 480.0
        n = int(math.ceil((self.t1 - self.t0) / dt)) + 2
        om = 2.0 / max(self.look_lag, 1e-3)        # critically damped: steady-state lag behind a moving target = look_lag s
        ts = self.t0 + dt * np.arange(n)
        x = np.asarray(fn(self.t0), float)
        v = (np.asarray(fn(self.t0 + 0.02), float) - x) / 0.02
        X, V = [x.copy()], [v.copy()]
        e = math.exp(-om * dt)
        for i in range(n - 1):
            T = np.asarray(fn(ts[i] + 0.5 * dt), float)
            d = x - T
            k = v + om * d
            x, v = T + (d + k * dt) * e, (v - om * k * dt) * e
            X.append(x.copy()); V.append(v.copy())
        self._tk_t, self._tk_x, self._tk_v = ts, np.array(X), np.array(V)

    def _tracker_at(self, t):
        if self._tk_t is None:
            self._build_tracker(self._look_fn)
        ts = self._tk_t
        dt = ts[1] - ts[0]
        i = int(min(max((t - ts[0]) / dt, 0), len(ts) - 2))
        u = (t - ts[i]) / dt
        return _hermite(self._tk_x[i], self._tk_x[i + 1], self._tk_v[i] * dt, self._tk_v[i + 1] * dt, u)

    def look_at(self, t):
        t = min(max(float(t), self.t0), self.t1)
        if self._look_mode == "track":
            return self._tracker_at(t)
        if self._look_mode == "curve":
            u = self.phase(t)
            if self._look_cs is None:                              # 1-2 points: smooth blend
                a, b = self._look_L[0], self._look_L[-1]
                return a + (b - a) * float(smoother(u))
            return np.asarray(self._look_cs(min(max(u, self._look_u[0]), self._look_u[-1])), float)
        return self.ahead(t, self.lead)

    # ------------------------------------------------------------ camera
    def _roll_target(self, t):
        return float(self._roll(self.phase(t)))

    def _build_roll(self):
        """roll table at 240 Hz: target roll(u(t)) -> slew limit (roll_max_dps) -> critically damped smoothing (roll_lag s).
        The smoothing is linear with a non-negative kernel, so the output rate never exceeds the slew limit."""
        dt = 1.0 / 240.0
        n = int(math.ceil((self.t1 - self.t0) / dt)) + 2
        ts = self.t0 + dt * np.arange(n)
        T = np.array([self._roll_target(min(t, self.t1)) for t in ts])
        lim = 0.95 * self._roll_max * dt                  # 5% margin so sampling never touches the limit
        y = np.empty(n)
        y[0] = T[0]
        for i in range(1, n):
            y[i] = y[i - 1] + min(max(T[i] - y[i - 1], -lim), lim)
        om = 2.0 / max(self._roll_lag, 1e-3)
        x, v = y[0], 0.0
        X = np.empty(n)
        X[0] = x
        e = math.exp(-om * dt)
        for i in range(n - 1):
            d = x - y[i]
            k = v + om * d
            x, v = y[i] + (d + k * dt) * e, (v - om * k * dt) * e
            X[i + 1] = x
        self._rl_t, self._rl_x = ts, X

    def roll_at(self, t):
        if self._roll_zero:
            return 0.0
        if getattr(self, "_rl_t", None) is None:
            self._build_roll()
        return float(np.interp(t, self._rl_t, self._rl_x))

    def drift_env(self, t):
        """0..1 envelope: 1 while the eye rests in a hold (dur > 0), smooth 0.15 s ramps each side."""
        e = 0.0
        for st in self.stops:
            if st["dur"] > 1e-9:
                pad = 0.15
                up = float(smoother((t - (st["a"] - pad)) / pad)) * (1.0 - float(smoother((t - st["b"]) / pad)))
                e = max(e, up)
        return e

    def at(self, t):
        t = min(max(float(t), self.t0), self.t1)
        s = self.s_of_t(t)
        u = s / self.length
        rh = self._right(t) if callable(self._right) else self._right
        rw = self._right_w(t) if callable(self._right_w) else self._right_w
        eye, tgt, F = np.asarray(self._E(s), float), self.look_at(t), self._F(u)
        roll = self.roll_at(t)
        if self._drift_px > 0 and self.stops:
            env = self.drift_env(t)
            if env > 1e-4:
                c0 = Camera(eye, tgt, roll, F, right=rh, right_w=1.0 if rw is None else rw)
                if not hasattr(self, "_dph"):
                    rng = np.random.RandomState(5)
                    self._dph, self._dfr = rng.uniform(0, 2 * np.pi, (4, 2)), rng.uniform(0.6, 1.5, (4, 2))
                w = 2 * np.pi * self._drift_hz * t
                nz = np.sin(w * self._dfr + self._dph).sum(0) / 4.0
                k = max(np.linalg.norm(tgt - eye), 1.0) / F
                off = (c0.r * nz[0] + c0.d * nz[1]) * self._drift_px * k * env
                eye, tgt = eye + off, tgt + off
        return Camera(eye, tgt, roll, F, right=rh, right_w=1.0 if rw is None else rw)

    # ------------------------------------------------------------ test hook
    def speed_report(self, fps=30.0):
        """per-frame diagnostics of the ACTUAL cameras returned by at() (not the analytic profile).
        Frames i = 0..N-1 at t0 + i / fps; per-step quantities use steps i -> i+1 (N-1 steps, time = step midpoint).
        Returns dict: t (step midpoints), eye_speed (units/s), look_speed (rad/s, change of the view direction),
        heading_turn (rad/s, change of the eye velocity direction), accel_frac (|d speed| per frame / median speed),
        mask (steps where the camera must be moving: between the global ease-in end and ease-out start, and OUTSIDE every
        declared stop window), stop_mask (steps inside a declared hold / reversal window = rest +- half its ease; zero or low
        speed is allowed only there), heading_jump_deg (per-step angle between consecutive eye-velocity directions),
        roll_step_deg (per-frame change of the image roll), and a 'summary' dict:
        median (of eye speed over mask), min_mid, max_mid, min_ratio (min_mid / median over mask), max_accel_frac,
        max_look_speed, max_look_jump, max_turn_per_frame_deg (heading jump over mask steps), start_speed, end_speed,
        path_length, travelled,
        near_zero_outside (steps in mask with speed < 5% of median: must be 0), stop_frames (steps inside stop windows),
        min_speed_in_stops, max_speed_in_rests (max eye speed while exactly inside a rest, u/s),
        max_heading_jump_deg (steps with speed > 5% of median, outside stop windows: 3 deg/frame is the limit),
        max_look_turn_deg (largest per-frame change of the view direction, all steps: 3 deg/frame is the limit),
        max_roll_change_deg (largest per-frame change of the image roll incl. `right`-hint blending: 1.5 deg/frame limit),
        max_jerk_frac (largest |second difference of speed| per frame^2 as a fraction of median),
        no_stop (min_ratio > 0.25 and near_zero_outside == 0)."""
        N = int(round((self.t1 - self.t0) * fps)) + 1
        ts = self.t0 + np.arange(N) / fps
        cams = [self.at(t) for t in ts]
        E = np.array([c.eye for c in cams])
        Fv = np.array([c.f for c in cams])
        dE = np.diff(E, axis=0)
        sp = np.linalg.norm(dE, axis=1) * fps
        ang = np.arccos(np.clip((Fv[1:] * Fv[:-1]).sum(1), -1, 1)) * fps
        hd = dE / np.maximum(np.linalg.norm(dE, axis=1, keepdims=True), 1e-12)
        turn = np.arccos(np.clip((hd[1:] * hd[:-1]).sum(1), -1, 1)) * fps
        tm = 0.5 * (ts[1:] + ts[:-1])
        stop = np.zeros(len(tm), bool)
        for lo, hi in self._stop_win:
            stop |= (tm > lo) & (tm < hi)
        e_in = self.t0 + self.ease_in if not (self.stops and self.stops[0]["i"] == 0) else self.t0
        e_out = self.t1 - self.ease_out if not (self.stops and self.stops[-1]["i"] == self._n - 1) else self.t1
        mask = (tm > e_in) & (tm < e_out) & ~stop
        if not mask.any():
            mask = np.ones_like(tm, bool)
        med = float(np.median(sp[mask]))
        dsp = np.abs(np.diff(sp))
        m2 = mask[1:] & mask[:-1]
        mov = sp > 0.05 * med
        hj_ok = mov[1:] & mov[:-1] & ~stop[1:] & ~stop[:-1]
        hj = np.degrees(turn / fps)                               # steps 1.. : angle between step i-1 and i (per frame)
        rl = np.unwrap(np.array([c.image_roll for c in cams]))
        droll = np.degrees(np.abs(np.diff(rl)))
        look_deg = np.degrees(ang / fps)
        jerk = np.abs(np.diff(sp, 2)) / med if len(sp) > 2 else np.zeros(1)
        rest = np.zeros(len(tm), bool)
        for st in self.stops:
            rest |= (tm >= st["a"] + 1e-9) & (tm <= st["b"] - 1e-9)
        summ = dict(median=med, min_mid=float(sp[mask].min()), max_mid=float(sp[mask].max()),
                    min_ratio=float(sp[mask].min() / med),
                    max_accel_frac=float(dsp[m2].max() / med) if m2.any() else 0.0,
                    max_look_speed=float(ang.max()), max_look_jump=float(np.abs(np.diff(ang)).max()) if len(ang) > 1 else 0.0,
                    max_turn_per_frame_deg=float(math.degrees(turn[m2[:len(turn)]].max() / fps)) if (len(turn) and m2[:len(turn)].any()) else 0.0,
                    start_speed=float(sp[0]), end_speed=float(sp[-1]), path_length=self.length,
                    travelled=float(sp.sum() / fps),
                    near_zero_outside=int((sp[mask] < 0.05 * med).sum()),
                    stop_frames=int(stop.sum()),
                    min_speed_in_stops=float(sp[stop].min()) if stop.any() else float("nan"),
                    max_speed_in_rests=float(sp[rest].max()) if rest.any() else 0.0,
                    max_heading_jump_deg=float(hj[hj_ok].max()) if (len(hj) and hj_ok.any()) else 0.0,
                    max_look_turn_deg=float(look_deg.max()),
                    max_roll_change_deg=float(droll.max()) if len(droll) else 0.0,
                    max_jerk_frac=float(jerk.max()))
        summ["no_stop"] = bool(summ["min_ratio"] > 0.25 and summ["near_zero_outside"] == 0)
        return dict(t=tm, eye_speed=sp, look_speed=ang, heading_turn=turn, mask=mask, stop_mask=stop,
                    heading_jump_deg=hj, roll_step_deg=droll, look_turn_deg=look_deg,
                    accel_frac=np.r_[0.0, dsp] / med, summary=summ)
