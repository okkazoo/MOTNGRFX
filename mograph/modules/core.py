"""mograph.modules.core: the lego interface for vector type modules.

A *word module* forms any words from nothing out of an incoming strand Bundle, holds, and deconstructs into an
outgoing Bundle. A *transition module* carries the exit Bundle of one module to the entry Bundle of the next, anywhere in
the world. Everything is cam3d Layers in a shared 3D world (x right, y down, +z away from the viewer; the 960x540 frame
plane sits at z = 0); the camera is separate and never owned by a module.

Contract that makes modules chain:
  A.layers(A.duration) draws exactly A.exit_bundle();  T.layers(0) draws exactly its a_exit;  T.layers(T.duration)
  draws exactly its b_entry;  B.layers(0) draws exactly B.entry_bundle().
"""
import os, sys, math, glob
import numpy as np
from scipy.optimize import linear_sum_assignment

from mograph.render import cam3d
from mograph.render.cam3d import Layer, Plane
ol = cam3d.ol
from mograph.fonts_liked import require_liked, lookup as lookup_liked  # noqa: E402  (hard allowlist: mograph/fonts/liked.json)

INK, BONE = (12, 12, 11), (236, 230, 218)
GREY = (120, 116, 108)
AMBER, VERM, COBALT, CORAL = (242, 166, 90), (210, 70, 42), (70, 110, 240), (232, 111, 104)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True) if v.ndim > 1 else np.linalg.norm(v)
    return v / np.maximum(n, 1e-12)


def clamp(x, a=0.0, b=1.0):
    return min(b, max(a, x))


def mixc(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


# ------------------------------------------------------------------ palette
class Palette:
    """bg = ground colour, fg = main ink, accents = tube / pulse colours, panel = faint fill used behind outline type."""

    def __init__(self, bg=INK, fg=BONE, accents=(AMBER, COBALT, VERM), panel=None):
        self.bg, self.fg, self.accents = tuple(bg), tuple(fg), tuple(tuple(a) for a in accents)
        self.panel = tuple(panel) if panel is not None else mixc(self.bg, self.fg, 0.12)

    @staticmethod
    def get(p):
        if isinstance(p, Palette):
            return p
        return PALETTES[p or "ink"]


PALETTES = {"ink": Palette(INK, BONE, (AMBER, COBALT, VERM)),
            "bone": Palette(BONE, INK, ((210, 70, 42), (45, 75, 216), (200, 130, 40)))}


# ------------------------------------------------------------------ fonts (resolve_font enforces mograph/fonts/liked.json)
FONT_DIR = ol.FD


def _key_of(fn):
    stem = os.path.splitext(fn)[0]
    for ch in "[-_":
        stem = stem.split(ch)[0]
    return stem.lower()


def font_files():
    out = {}
    for p in sorted(glob.glob(FONT_DIR + "*.[to]tf")):
        fn = os.path.basename(p)
        out.setdefault(_key_of(fn), fn)
    return out


LIKED = font_files()


def _norm_font(name):
    """'Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf' / 'Rock 3D' / 'Rock3D-Regular.ttf' -> 'bitcount' / 'rock3d' / 'rock3d'."""
    s = os.path.basename(str(name))
    if s.lower().endswith((".ttf", ".otf")):
        s = s[:-4]
    return _key_of(s.replace(" ", "")).replace(" ", "")


def resolve_font(font):
    """font: ol key ('anybody'), liked-font key ('bungeeshade'), file name, registry name ('Rock 3D'), (key, loc) or
    {'key'|'name'|'file': .., 'loc'|'axes': {..}, 'ar': ..} -> (ol key, loc). Variable fonts without an explicit loc get their
    heaviest weight. ('ar' = horizontal squeeze for tall faces, read by layout_word.)"""
    loc = None
    if isinstance(font, (tuple, list)):
        font, loc = font
    elif isinstance(font, dict):
        font, loc = (font.get("key") or font.get("name") or font.get("file")), (font.get("loc") or font.get("axes"))
    font = str(font)
    if font in ol.FONTS:
        key = font
        liked = require_liked(ol.FONTS[key])              # HARD rule: only the fonts in mograph/fonts/liked.json
        if lookup_liked(ol.FONTS[key]) is None:           # not liked (non-strict): the substitute liked face of the same role, never the generic file
            key = _key_of(liked["file"])
            ol.FONTS.setdefault(key, liked["file"])
    else:
        fn = font if font in LIKED.values() else LIKED.get(font.lower().replace(" ", "").replace("_", ""))
        if fn is None:
            fn = LIKED.get(_norm_font(font))
        if fn is None and lookup_liked(font):                 # liked.json key / family ('Facade' -> FacadeVF.ttf)
            fn = lookup_liked(font)["file"]
        if fn is None:                                    # unknown / not liked: strict mode raises FontNotLiked, otherwise the closest liked face of the same role (one warning)
            fn = require_liked(font)["file"]
        liked = require_liked(fn)                         # HARD rule: only the fonts in mograph/fonts/liked.json
        if lookup_liked(fn) is None:                      # a bundled but not-liked file (Anton-Regular.ttf): the substitute liked file, never the generic one
            fn = liked["file"]
        key = _key_of(fn)
        if key not in ol.FONTS:
            ol.FONTS[key] = fn
    if loc is None and key in DEFAULT_LOC:
        loc = DEFAULT_LOC[key]
    if loc is None and liked.get("axes"):                 # liked variant of a variable face (FolditThin = wght 100)
        loc = dict(liked["axes"])
    if loc is None:
        loc = _heavy_loc(key)
    return key, loc


DEFAULT_LOC = {"anybody": {"wght": 900, "wdth": 100}}
_HLOC = {}


def _heavy_loc(key):
    if key in _HLOC:
        return _HLOC[key]
    loc = None
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(ol.font_path(key))
        if "fvar" in f:
            for a in f["fvar"].axes:
                if a.axisTag == "wght":
                    loc = {"wght": a.maxValue}
    except Exception:
        loc = None
    _HLOC[key] = loc
    return loc


def norm_lines(words):
    if isinstance(words, str):
        return [s for s in words.split("\n")]
    return [str(s) for s in words]


_LAY = {}


class ConList(list):
    """a letter's contour list; .union = True when all contours wind the same way (pixel fonts built from overlapping squares /
    dots): cam3d.Layer then fills every contour on its own, so overlaps merge instead of cancelling (even-odd)."""
    union = False


def _signed_area(c):
    return 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))


def layout_word(font, lines, w, h, track=0.02, leading=0.22, ar=None):
    """words fitted to a w x h box, centred on its middle, local coords x right / y DOWN. -> dict(cons=[contours per
    letter], letters, cap, scale). Letters keep font proportions (uniform scale) unless ar < 1: a squeezed (tall) face, x scaled by
    ar relative to y (`ar` argument, or font = {'key':.., 'ar': 0.6}); the box is then filled by height."""
    key, loc = resolve_font(font)
    if ar is None:
        ar = float(font.get("ar", 1.0)) if isinstance(font, dict) else 1.0
    ck = (key, repr(sorted(loc.items())) if loc else "", tuple(lines), round(w), round(h), track, leading, round(ar, 4))
    if ck in _LAY:
        return _LAY[ck]
    H, _ = ol.word(key, "H", loc=loc, step=5)
    capu = float(np.concatenate(H[0]["contours"])[:, 1].max())
    rows, widths = [], []
    for ln in lines:
        L, upm = ol.word(key, ln, loc=loc, track=track, step=5)
        rows.append(L)
        if L:
            P = np.concatenate([c for l in L for c in l["contours"]])
            widths.append((P[:, 0].min(), P[:, 0].max()))
        else:
            widths.append((0.0, 0.0))
    n = len(lines)
    blockH = capu * (n + leading * (n - 1))
    maxW = max(b - a for a, b in widths) or 1.0
    s = min(w / (maxW * ar), h / blockH)
    sx = s * ar
    letters = []
    for i, (L, (x0, x1)) in enumerate(zip(rows, widths)):
        base = -0.5 * blockH * s + capu * s + i * capu * (1 + leading) * s       # baseline y (down)
        cx = 0.5 * (x0 + x1)
        for l in L:
            cons = ConList(np.stack([(c[:, 0] - cx) * sx, base - c[:, 1] * s], 1) for c in l["contours"])
            if len(cons) > 1:
                sg = np.sign([_signed_area(c) for c in cons])
                cons.union = bool(np.all(sg == sg[0]) and sg[0] != 0)
            letters.append(dict(ch=l["ch"], cons=cons))
    out = dict(letters=letters, cap=capu * s, scale=s, size=(maxW * sx, blockH * s), ar=ar)
    _LAY[ck] = out
    return out


# ------------------------------------------------------------------ placement / port / timing
class Placement:
    """where a word module lives in the world: box w x h centred on `origin`, spanned by unit vectors u (box x, to the right
    of the type) and v (box y, DOWN the type); normal n = u x v points away from the viewer for a front-facing box."""

    def __init__(self, origin=(480.0, 270.0, 0.0), u=(1, 0, 0), v=(0, 1, 0), w=800.0, h=340.0):
        self.origin, self.u, self.v = np.asarray(origin, float), unit(u), unit(v)
        self.w, self.h = float(w), float(h)

    @classmethod
    def at(cls, origin, yaw=0.0, pitch=0.0, roll=0.0, w=800.0, h=340.0, degrees=True):
        """box centred at `origin`, rotated like cam3d.plane_rot (pitch > 0 tips the top edge away; yaw > 0 brings the right
        edge toward the viewer), then rolled in-plane (roll > 0 clockwise on screen when front-facing)."""
        k = math.pi / 180.0 if degrees else 1.0
        pl = cam3d.plane_rot(origin, yaw * k, pitch * k)
        u, v = pl.u, pl.v
        r = roll * k
        if r:
            u, v = math.cos(r) * pl.u + math.sin(r) * pl.v, -math.sin(r) * pl.u + math.cos(r) * pl.v
        return cls(origin, u, v, w, h)

    @classmethod
    def centred(cls, w=800.0, h=340.0):
        """library default: front-facing, centred in the 960x540 frame."""
        return cls((cam3d.W / 2.0, cam3d.H / 2.0, 0.0), (1, 0, 0), (0, 1, 0), w, h)

    @property
    def n(self):
        return unit(np.cross(self.u, self.v))

    @property
    def plane(self):
        return Plane(self.origin, self.u, self.v, pivot=(0.0, 0.0))

    def lift(self, P2):
        P2 = np.asarray(P2, float)
        return self.origin + P2[..., 0:1] * self.u + P2[..., 1:2] * self.v

    def extent_along(self, d):
        """half-extent of the box along unit vector d."""
        d = np.asarray(d, float)
        return 0.5 * (abs(d @ self.u) * self.w + abs(d @ self.v) * self.h)

    def corners(self):
        return self.lift(np.array([[-self.w / 2, -self.h / 2], [self.w / 2, -self.h / 2], [self.w / 2, self.h / 2],
                                   [-self.w / 2, self.h / 2]]))

    def __repr__(self):
        return f"Placement(origin={np.round(self.origin, 1).tolist()}, n={np.round(self.n, 2).tolist()}, {self.w:.0f}x{self.h:.0f})"


class Port:
    """a side of a placement. `direction`: 'left'|'right'|'up'|'down' (in the placement plane), 'in' (away from the viewer,
    along +n) | 'out' (toward the viewer), a number = angle in degrees in the placement plane (0 = right, 90 = up, counter-
    clockwise as seen from the viewer), or a 3-vector = world direction. `side(placement)` is the unit vector from the word
    toward that side: an ENTRY port 'left' means strands come from the left (travelling along +u); an EXIT port 'right' means
    strands leave toward the right (travelling along +u). spread >= 0 fans the lanes out toward the far end (0 = parallel)."""
    NAMES = ("left", "right", "up", "down", "in", "out")

    def __init__(self, direction="left", spread=0.0):
        self.direction, self.spread = direction, float(spread)

    @staticmethod
    def coerce(x, default="left"):
        if x is None:
            return Port(default)
        if isinstance(x, Port):
            return x
        if isinstance(x, str):
            s = x.strip().lower()
            if s in Port.NAMES:
                return Port(s)
            parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) == 3:
                return Port([float(p) for p in parts])
            return Port(float(s))
        if np.ndim(x) == 0:
            return Port(float(x))
        return Port(list(np.asarray(x, float)))

    def side(self, pl):
        d = self.direction
        if isinstance(d, str):
            return {"left": -pl.u, "right": pl.u, "up": -pl.v, "down": pl.v, "in": pl.n, "out": -pl.n}[d]
        if np.ndim(d) == 0:
            a = math.radians(float(d))
            return unit(math.cos(a) * pl.u - math.sin(a) * pl.v)
        return unit(np.asarray(d, float))

    def __repr__(self):
        return f"Port({self.direction!r}, spread={self.spread})"


class Timing:
    """seconds: form (strands arrive and build the type), hold (type stands), exit (it deconstructs and leaves)."""

    def __init__(self, form=1.6, hold=1.0, exit=1.4):
        self.form, self.hold, self.exit = float(form), float(hold), float(exit)

    @property
    def total(self):
        return self.form + self.hold + self.exit

    def snap(self, beat):
        """round every phase to a whole number of beats (>= 1 beat; the hold may be 0)."""
        r = lambda x, lo: max(lo, round(x / beat) * beat)
        return Timing(r(self.form, beat), r(self.hold, 0.0), r(self.exit, beat))

    def __repr__(self):
        return f"Timing(form={self.form}, hold={self.hold}, exit={self.exit})"


# ------------------------------------------------------------------ bundle
class Bundle:
    """the hand-off currency. N strands as 3D polylines P (N,K,3), HEAD FIRST (P[:,0] is the leading end, P[:,-1] the tail),
    D (N,3) unit direction of travel at the head, col (N,3) colours, w (N,) widths (px at home depth), v (N,) head speed
    (world units / s) at this state, t = time stamp, tube 0|1 (hairline | shaded tube), rim (N,3) tube rim colours."""

    def __init__(self, P, D, col, w, t=0.0, v=None, tube=0.0, rim=None):
        self.P = np.asarray(P, float)
        n = len(self.P)
        self.D = unit(np.asarray(D, float).reshape(n, 3))
        self.col = np.broadcast_to(np.asarray(col, float), (n, 3)).copy()
        self.w = np.broadcast_to(np.asarray(w, float), (n,)).copy()
        self.v = np.zeros(n) if v is None else np.broadcast_to(np.asarray(v, float), (n,)).copy()
        self.t, self.tube = float(t), float(tube)
        self.rim = self.col * 0.3 if rim is None else np.broadcast_to(np.asarray(rim, float), (n, 3)).copy()

    @property
    def N(self):
        return len(self.P)

    @property
    def K(self):
        return self.P.shape[1]

    def heads(self):
        return self.P[:, 0]

    def tails(self):
        return self.P[:, -1]

    def centroid(self):
        return self.P.mean((0, 1))

    def copy(self):
        return Bundle(self.P.copy(), self.D.copy(), self.col.copy(), self.w.copy(), self.t, self.v.copy(), self.tube,
                      self.rim.copy())

    def subset(self, idx):
        """strands idx (array of indices; repeats allowed -> duplicated strands)."""
        idx = np.asarray(idx, int)
        return Bundle(self.P[idx], self.D[idx], self.col[idx], self.w[idx], self.t, self.v[idx], self.tube, self.rim[idx])

    def resample_points(self, K):
        """K points per strand, uniform in arc length (head stays first)."""
        out = np.zeros((self.N, K, 3))
        for i, p in enumerate(self.P):
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            s = np.linspace(0, cum[-1], K)
            out[i] = np.stack([np.interp(s, cum, p[:, k]) for k in range(3)], 1) if cum[-1] > 1e-9 else p[:1]
        return Bundle(out, self.D, self.col, self.w, self.t, self.v, self.tube, self.rim)

    def order(self):
        """strand indices ordered along the principal axis of the head positions (neighbours in the cross-section stay neighbours)."""
        H = self.heads() - self.heads().mean(0)
        if self.N < 3:
            return np.arange(self.N)
        ax = np.linalg.svd(H, full_matrices=False)[2][0]
        return np.argsort(H @ ax)

    def resample_strands(self, M):
        """M strands by linear interpolation between neighbours along `order()` (positions, colours, widths, speed)."""
        o = self.order()
        f = np.linspace(0, self.N - 1, M)
        i0 = np.minimum(f.astype(int), self.N - 2) if self.N > 1 else np.zeros(M, int)
        i1 = np.minimum(i0 + 1, self.N - 1)
        a = (f - i0)[:, None]
        g = lambda X: X[o[i0]] * (1 - a.reshape((-1,) + (1,) * (X.ndim - 1))) + X[o[i1]] * a.reshape((-1,) + (1,) * (X.ndim - 1))
        return Bundle(g(self.P), g(self.D), g(self.col), g(self.w), self.t, g(self.v), self.tube, g(self.rim))

    def __repr__(self):
        return f"Bundle(N={self.N}, K={self.K}, tube={self.tube}, centroid={np.round(self.centroid(), 0).tolist()})"


def _rot_between(a, b):
    """rotation matrix taking unit vector a to unit vector b (minimal rotation)."""
    a, b = unit(a), unit(b)
    c = float(a @ b)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        p = unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Pairing:
    """result of adapt(): K = max(N, M) strand pairs. ia[k] / ib[k] index the strands of bundle a / b (a strand of the
    smaller bundle is used more than once: it splits / merges). `a_cut`, `b_cut` are the matching pre-paired cross-sections."""

    def __init__(self, ia, ib, cost):
        self.ia, self.ib, self.cost = ia, ib, cost

    @property
    def K(self):
        return len(self.ia)

    def bundles(self, a, b):
        return a.subset(self.ia), b.subset(self.ib)


def adapt(a, b):
    """pair the strands of a (exit bundle) with the strands of b (entry bundle) so the streams do not cross.
    Cross-sections = a's head positions vs b's tail positions (where the carried strand joins b's body), both relative to
    their centroid, b rotated by the minimal rotation that turns b's travel direction into a's. Optimal assignment
    (scipy linear_sum_assignment) on squared distance. N != M: the unmatched strands of the larger side pair with their
    nearest strand on the smaller side (resample N->M by splitting / merging). -> Pairing with equal-length ia / ib."""
    if not (isinstance(a, Bundle) and isinstance(b, Bundle)):            # any Material kind: convert to strands first (phase 3a)
        from mograph.modules.material import ensure_bundle
        a, b = ensure_bundle(a, b), ensure_bundle(b, a)
    Ha, Tb = a.heads(), b.tails()
    R = _rot_between(unit(b.D.mean(0)), unit(a.D.mean(0)))
    oa = Ha - Ha.mean(0)
    ob = (Tb - Tb.mean(0)) @ R.T
    cost = ((oa[:, None, :] - ob[None, :, :]) ** 2).sum(-1)
    ri, ci = linear_sum_assignment(cost)
    ia, ib = list(ri), list(ci)
    N, M = a.N, b.N
    if N > M:
        for i in sorted(set(range(N)) - set(ri)):
            ia.append(i); ib.append(int(np.argmin(cost[i])))
    elif M > N:
        for j in sorted(set(range(M)) - set(ci)):
            ib.append(j); ia.append(int(np.argmin(cost[:, j])))
    ia, ib = np.array(ia, int), np.array(ib, int)
    o = np.lexsort((ib, ia))
    return Pairing(ia[o], ib[o], float(cost[ia, ib].sum()))


# ------------------------------------------------------------------ route + kinematics helpers
class Route3:
    """3D polyline with arc length. window(a, b) = the piece between arc lengths a < b (None when shorter than 0.5)."""

    def __init__(self, P):
        P = np.asarray(P, float)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        self.P = P[keep]
        self.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))]
        self.L = float(self.cum[-1])

    def at(self, s):
        s = min(max(float(s), 0.0), self.L)
        i = int(min(np.searchsorted(self.cum, s, "right") - 1, len(self.P) - 2))
        d = self.cum[i + 1] - self.cum[i]
        u = (s - self.cum[i]) / d if d > 1e-9 else 0.0
        return self.P[i] + (self.P[i + 1] - self.P[i]) * u

    def window(self, a, b):
        a, b = max(0.0, float(a)), min(self.L, float(b))
        if b - a < 0.5:
            return None
        i0 = int(np.searchsorted(self.cum, a, "right"))
        i1 = int(np.searchsorted(self.cum, b, "left"))
        return np.vstack([self.at(a), self.P[i0:i1], self.at(b)])

    def tangent_at(self, s):
        e = 2.0
        d = self.at(min(s + e, self.L)) - self.at(max(s - e, 0.0))
        return unit(d)


def herm(p0, p1, m0, m1, u):
    """cubic Hermite scalar/vector: position p0 -> p1 over u in [0,1] with slopes m0, m1 (per unit u)."""
    u = min(1.0, max(0.0, u))
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


def clamp_slope(m, D, k=2.9):
    """keep a Hermite end slope m*T below k*D so the profile stays monotone (D = distance travelled)."""
    return min(m, k * max(D, 1e-9))


class Slide:
    """window [tail, head] sliding along a Route3 between two states over duration T: head p_h0 -> p_h1, tail p_t0 -> p_t1
    (arc lengths), both with Hermite speed profiles v0 (start) and v1 (end) in world units / s. Because head and tail share
    the same time profile shape, head - tail never goes negative while both ends move forward."""

    def __init__(self, route, h0, h1, t0, t1, T, v0, v1, delay=0.0, k=2.9):
        self.R, self.h0, self.h1, self.t0, self.t1 = route, h0, h1, t0, t1
        self.T, self.delay = max(T, 1e-6), delay
        self.v0, self.v1, self.k = v0, v1, k
        self.tail_lag = 0.0         # 0..~0.4: extra mid-way body stretch (fraction of the route)
        self.coast = False          # True: after the end, keep sliding at the end speed (the route must be long enough)

    def pos(self, a0, a1, tt, lag=0.0):
        u = (tt - self.delay) / self.T
        if u <= 0:
            return a0
        if lag and u < 1.0:                      # the tail falls behind mid-way (body stretches) and catches up at the end;
            u = u - lag * math.sin(math.pi * u) ** 2        # speed stays continuous at both ends
        D = a1 - a0
        m0 = clamp_slope(self.v0 * self.T, D, self.k)
        m1 = clamp_slope(self.v1 * self.T, D, self.k)
        if u >= 1:
            return a1 + (m1 / self.T) * (tt - self.delay - self.T) if self.coast else a1
        return float(herm(a0, a1, m0, m1, u))

    def window(self, tt):
        h = self.pos(self.h0, self.h1, tt)
        t = self.pos(self.t0, self.t1, tt, self.tail_lag)
        return self.R.window(t, h), h, t


def flock(K, T, seed=0, mid=None, lead=0.2, amp=0.0):
    """FLOCK timing + paths for an unravelling transition (2026-11-01, user: "all the lines dont need to move at the same time they can flock ... around
    a movement"): per strand a start delay in [0, lead x T] and an end in [(1 - lead) x T, T] (leaders by rank + seeded jitter, so some lead and some lag
    and the spacing loosens and tightens; every strand has arrived by T: B lands complete on its beat), and, when `mid` (K arrays (n, 3) of the
    shared route) is given, each route bent off the shared movement by a smooth seeded offset of up to `amp` px (zero at both ends). Deterministic.
    -> (delays (K,), durations (K,), mids)."""
    rng = np.random.default_rng(int(seed) + 7919)
    r = np.arange(K) / max(K - 1, 1)
    j = rng.uniform(-0.35, 0.35, K)
    d = lead * T * np.clip(0.6 * r + 0.4 * rng.uniform(0, 1, K), 0, 1)
    e = T - lead * T * np.clip(0.6 * (1 - r) + 0.4 * rng.uniform(0, 1, K) + 0.0 * j, 0, 1)
    out = None
    if mid is not None and amp > 0:
        out = []
        for k, m in enumerate(mid):
            m = np.asarray(m, float)
            n = len(m)
            u = np.linspace(0, 1, n)[:, None]
            env = np.sin(math.pi * u) ** 2
            a = rng.uniform(0, 2 * math.pi)
            ph = rng.uniform(0, 2 * math.pi)
            wave = 0.6 + 0.4 * np.sin(2 * math.pi * 1.3 * u + ph)
            out.append(m + env * wave * float(rng.uniform(0.35, 1.0)) * amp * np.array([[math.cos(a), math.sin(a), 0.0]]))
    return d, np.maximum(e - d, 0.25 * T), out


# ------------------------------------------------------------------ strand drawing shared by modules / transitions
TUBE_BIAS = (0.0, 3000.0, 6000.0)


def strand_layers(polys, cols, widths, tube=0.0, rims=None, hi_off=None, bias0=0.0):
    """Layers for a list of strand polylines (3D, tail->head or any order). tube 0..1: 0 = hairline (strands of equal
    colour / width share one Layer), 1 = shaded tube (rim, body, highlight; all rims then all bodies then all highlights
    by depth, so pieces of one strand never overdraw each other's joints)."""
    items = [(p, c, w, i) for i, (p, c, w) in enumerate(zip(polys, cols, widths)) if p is not None and len(p) > 1 and w > 0.02]
    if not items:
        return []
    if tube <= 0.01:
        groups = {}
        for p, c, w, i in items:
            groups.setdefault((tuple(int(round(x)) for x in c), round(float(w), 2)), []).append(p)
        return [Layer(ps, c, "line", w, closed=False, bias=0.25 + bias0) for (c, w), ps in groups.items()]
    out = []
    ho = np.zeros(3) if hi_off is None else np.asarray(hi_off, float)
    for p, c, w, i in items:
        rim = tuple(int(x) for x in (rims[i] if rims is not None else np.asarray(c) * 0.3))
        hi = mixc(tuple(int(x) for x in c), (255, 252, 246), 0.55)
        c = tuple(int(round(x)) for x in c)
        out.append(Layer([p], rim, "line", w * (1.0 + 0.6 * tube), closed=False, bias=bias0 + TUBE_BIAS[0]))
        out.append(Layer([p], c, "line", w, closed=False, bias=bias0 + TUBE_BIAS[1]))
        out.append(Layer([p + ho * (w * 0.28)], hi, "line", w * 0.28 * tube, closed=False, bias=bias0 + TUBE_BIAS[2]))
    return out


def chamfer(P, ch):
    """cut every interior corner of polyline P (n,k) by ch."""
    P = [np.asarray(p, float) for p in P]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        u1, u2 = P[i] - P[i - 1], P[i + 1] - P[i]
        n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        c = min(ch, 0.45 * n1, 0.45 * n2)
        out += [P[i] - u1 / n1 * c, P[i] + u2 / n2 * c]
    out.append(P[-1])
    return np.array(out)


def bezier(p0, p1, p2, p3, step=6.0):
    L = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    n = max(8, int(L / step))
    t = np.linspace(0, 1, n + 1)[:, None]
    m = 1 - t
    return m ** 3 * p0 + 3 * m * m * t * p1 + 3 * m * t * t * p2 + t ** 3 * p3


# ------------------------------------------------------------------ registry
_REG = {}


def register(name):
    """class decorator: `@register('lanes') class Lanes(WordModule)`. The name is the library / CLI id."""
    def deco(cls):
        cls.name = name
        _REG[name] = cls
        return cls
    return deco


def list_modules(kind=None):
    """registered module ids, optionally filtered by kind 'word' | 'transition'."""
    return sorted(n for n, c in _REG.items() if (kind is None or c.kind == kind) and not getattr(c, "hidden", False))


def get_module(name):
    if name not in _REG:
        raise KeyError(f"unknown module {name!r}; registered: {list_modules()}")
    return _REG[name]


# ------------------------------------------------------------------ base classes
class _ConstEye:
    """picklable constant eye(t) (the real camera's eye at the real time, while a style freezes the body's own clock)."""

    def __init__(self, p):
        self.p = np.asarray(p, float)

    def __call__(self, t):
        return self.p


def sw_focus(self, orig, t, ph):
    if ph is not None and ph[0] == "style":
        return ph[1].focus(t, ph[2])
    return orig(self, t)


def _io_wrap(nm, orig):
    """phase 3a: the public layers / entry_bundle / exit_bundle / focus of every word module route through its form-in /
    form-out style. With the default 'lines' style (or while a style asks for the module's own look: `_raw`) they are the
    module's own methods, untouched."""
    if nm == "layers":
        def w(self, t):
            if self.carry is not None and not self._carry_off:
                t = self._carry_time(t)
            if self._raw or getattr(self, "_plain", True):
                return orig(self, t)
            return self._styled_layers(t)
    elif nm == "focus":
        def w(self, t):
            if self._raw or getattr(self, "_plain", True):
                return orig(self, t)
            ph = self._phase(t)
            return sw_focus(self, orig, t, ph)
    else:
        side = "in" if nm == "entry_bundle" else "out"

        def w(self):
            st = getattr(self, "form_in" if side == "in" else "form_out", None)
            if self._raw or st is None or st.is_lines:
                return orig(self)
            m = self.entry_material() if side == "in" else self.exit_material()
            return self._rawcall(nm) if m.kind == "none" else m.as_bundle()
    w.__name__, w.__doc__, w._io = nm, orig.__doc__, True
    return w


class WordModule:
    """forms `words` from nothing, holds, deconstructs. Subclass: implement build() (precompute geometry, entry / exit
    bundles) and layers(t). Times t in [0, duration]; t < form: building, then hold, then exit.

    entry_bundle(): the strands at t = 0 (waiting just outside the placement on the entry side, heads at the gate).
    exit_bundle():  the strands at t = duration (just past the placement on the exit side).
    focus(t): world point a follow-camera may track.  hold_pose(dist): (eye, target) facing the formed word.

    Phase 3a (body / form-in / form-out): `form_in` / `form_out` name an I/O style (mograph.modules.io: 'lines' (default = the
    module's own strand form / exit, unchanged), 'blobs', 'flicker', 'particles', 'wipe'; a style instance or ('name',
    {options}) also works). A non-lines style owns the form window (reveal) and the exit window (conceal); the module's own look
    and mechanic run in the hold. The module itself stays a plain body: layers(t) / entry_bundle() / exit_bundle() written
    for the 'lines' look are all it implements; the base class does the rest (see body_layers, hold_shape, entry_material)."""
    kind = "word"
    name = ""
    description = ""
    DEMO_WORDS = "WORDS"
    DEFAULT_FONT = "anybody"
    _raw = False
    carry = None              # {"in": bool, "out": bool}: a transition (pan) carries this word world-locked, so its own form / exit windows stay frozen on the hold
    _carry_off = False

    def _carry_time(self, t):
        """module time with the carried windows frozen on the formed word (form window of a carried-in word, exit window of a carried-out one)."""
        tm, c = self.timing, self.carry
        if c.get("in") and t < tm.form:
            return tm.form
        if c.get("out") and t > tm.form + tm.hold:
            return tm.form + tm.hold
        return t

    def layers_uncarried(self, t):
        """layers(t) at the module's own time t, ignoring the carry freeze (the pan transition draws the styled exit / entry through it)."""
        prev, self._carry_off = self._carry_off, True
        try:
            return self.layers(t)
        finally:
            self._carry_off = prev

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for nm in ("layers", "entry_bundle", "exit_bundle", "focus"):
            f = cls.__dict__.get(nm)
            if f is not None and not getattr(f, "_io", False):
                setattr(cls, nm, _io_wrap(nm, f))

    def __init__(self, words=None, font=None, placement=None, entry="left", exit="right", timing=None, palette=None,
                 form_in="lines", form_out="lines", **params):
        self.words = words if words is not None else self.DEMO_WORDS
        self.lines = norm_lines(self.words)
        self.font = font if font is not None else self.DEFAULT_FONT
        self.placement = placement if placement is not None else Placement.centred()
        self.entry, self.exit = Port.coerce(entry, "left"), Port.coerce(exit, "right")
        self.timing = timing if timing is not None else Timing()
        self.palette = Palette.get(palette)
        self.params = params
        from mograph.modules.io import get_style
        self.form_in, self.form_out = get_style(form_in), get_style(form_out)
        self._plain = self.form_in.is_lines and self.form_out.is_lines
        self._mats, self._shape = {}, None
        self.build()
        self.form_in.attach(self, "in")
        self.form_out.attach(self, "out")

    def build(self):
        raise NotImplementedError

    def set_style(self, side, spec):
        """swap the form-in ('in') or form-out ('out') style after construction (mograph.modules.sequence: the glyph-outline hand-off of a library
        transition gap). Clears the cached material of that side."""
        from mograph.modules.io import get_style
        self._ho_save()
        st = get_style(spec)
        if side == "in":
            self.form_in = st
        else:
            self.form_out = st
        st.attach(self, side)
        self._plain = self.form_in.is_lines and self.form_out.is_lines
        self._mats.pop(side, None)

    def retime(self, form=None, exit=None):
        """shorten the form and / or exit window (seconds) and rebuild the body for the new timing (the hold is untouched). -> seconds saved."""
        self._ho_save()
        t0, tm = self._ho_orig[0], self.timing                 # saved time is counted against the ORIGINAL timing (a word may be re-sequenced)
        f = tm.form if form is None else min(t0.form, float(form))
        e = tm.exit if exit is None else min(t0.exit, float(exit))
        saved = ((t0.form - f) if form is not None else 0.0) + ((t0.exit - e) if exit is not None else 0.0)
        if abs(f - tm.form) > 1e-9 or abs(e - tm.exit) > 1e-9:
            self.timing = Timing(f, tm.hold, e)
            self.build()
            self.form_in.attach(self, "in")
            self.form_out.attach(self, "out")
            self._mats = {}
        return saved

    def _ho_save(self):
        if getattr(self, "_ho_orig", None) is None:
            self._ho_orig = (self.timing, self.form_in, self.form_out)

    def handoff_reset(self):
        """undo what a sequence's glyph-outline hand-off did to this word (styles + retime), so another Sequence of the same words starts clean
        (tools/fusion/spec.py builds Sequence.auto on the words first, then the real sequence)."""
        o = getattr(self, "_ho_orig", None)
        if o is None:
            return
        self._ho_orig = None
        tm, self.form_in, self.form_out = o
        if tm is not self.timing:
            self.timing = tm
            self.build()
        self.form_in.attach(self, "in")
        self.form_out.attach(self, "out")
        self._plain = self.form_in.is_lines and self.form_out.is_lines
        self._mats = {}

    @property
    def duration(self):
        return self.timing.total

    def entry_bundle(self):
        raise NotImplementedError

    def exit_bundle(self):
        raise NotImplementedError

    def layers(self, t):
        raise NotImplementedError

    def focus(self, t):
        return self.placement.origin.copy()

    def hold_pose(self, dist=900.0):
        pl = self.placement
        return pl.origin - pl.n * dist, pl.origin.copy()

    def bounds_points(self):
        """world points the demo camera should frame (default: placement corners)."""
        return self.placement.corners()

    # ------------------------------------------------------------ body / style plumbing (phase 3a)
    def _rawcall(self, nm, *a):
        """call the module's own `nm` (layers / entry_bundle / exit_bundle / focus), bypassing the styles."""
        prev, self._raw = self._raw, True
        try:
            return getattr(self, nm)(*a)
        finally:
            self._raw = prev

    def raw_layers(self, t, t_real=None):
        """the module's own layers at its own time t. t_real: the real time, so a module that looks at the camera (eye_hint)
        sees the camera of that moment although a style froze the body's clock."""
        eh = getattr(self, "eye_hint", None)
        swap = eh is not None and t_real is not None
        if swap:
            self.eye_hint = _ConstEye(eh(t_real))
        try:
            return self._rawcall("layers", t)
        finally:
            if swap:
                self.eye_hint = eh

    def body_time_in(self):
        """own time of the formed word at the start of the hold (what a reveal builds toward)."""
        tm = self.timing
        return tm.form + min(1e-3, 0.5 * tm.hold)

    def body_time_out(self):
        tm = self.timing
        return tm.form + max(tm.hold - 1e-3, 0.5 * tm.hold)

    def body_layers(self, t):
        """the module's own layers during its hold (its mechanic still animating); t is clamped into the hold."""
        return self.raw_layers(min(max(t, self.body_time_in()), self.body_time_out()), t)

    # A style reveals / conceals the body in its flat, formed state; a body whose own mechanic builds a volume during the form
    # window (STYLE_MECH > 0: depth_echo, hollow_tubes ... ) gets that share of the window after the reveal (and before the
    # conceal) to run it: q = 0..1 over that share.
    STYLE_MECH = 0.0

    def reveal_layers(self, q, t_real=None):
        """the body as a style hands over to it. q: progress of the body's own mechanic (0 = flat formed word, 1 = hold)."""
        return self.raw_layers(self.body_time_in(), t_real)

    def conceal_layers(self, q, t_real=None):
        """the body as it hands over to a conceal style (q = 0 at the hold, 1 = flat formed word)."""
        return self.raw_layers(self.body_time_out(), t_real)

    def hold_shape(self):
        """the formed word as plane geometry (outlines and fills in placement coordinates): io.shape.Shape."""
        if self._shape is None:
            from mograph.modules.io.shape import Shape
            self._shape = Shape.from_body(self)
        return self._shape

    def style_window(self, side):
        """(t0, T) of the style's window in module time: the form / exit window minus the share the body keeps for its mechanic."""
        tm, s = self.timing, self.STYLE_MECH
        if side == "in":
            return 0.0, tm.form * (1.0 - s)
        return tm.form + tm.hold + tm.exit * s, tm.exit * (1.0 - s)

    def _phase(self, t):
        """None (the module's own look) | ('style', style, p) | ('mech_in', q) | ('mech_out', q); p, q in 0..1 (p beyond 1 on
        exit: the carried material is still coasting away)."""
        tm, s = self.timing, self.STYLE_MECH
        if t < tm.form:
            if self.form_in.is_lines:
                return None
            p = max(t, 0.0) / max(tm.form, 1e-9)
            return ("style", self.form_in, p / (1.0 - s)) if (p < 1.0 - s or s <= 0.0) else ("mech_in", (p - (1.0 - s)) / s)
        if t >= tm.form + tm.hold:
            if self.form_out.is_lines:
                return None
            p = (t - tm.form - tm.hold) / max(tm.exit, 1e-9)
            return ("mech_out", p / s) if (s > 0.0 and p < s) else ("style", self.form_out, (p - s) / (1.0 - s))
        return None

    def _styled_layers(self, t):
        ph = self._phase(t)
        if ph is None:
            return self.raw_layers(t)
        if ph[0] == "style":
            return ph[1].layers(t, ph[2])
        return self.reveal_layers(ph[1], t) if ph[0] == "mech_in" else self.conceal_layers(ph[1], t)

    def entry_material(self):
        """what arrives at t = 0 as a Material (strands for the 'lines' style; the style's own kind otherwise)."""
        if "in" not in self._mats:
            from mograph.modules.material import Material
            self._mats["in"] = (Material.from_bundle(self._rawcall("entry_bundle")) if self.form_in.is_lines
                                else self.form_in.material())
        return self._mats["in"]

    def exit_material(self):
        if "out" not in self._mats:
            from mograph.modules.material import Material
            self._mats["out"] = (Material.from_bundle(self._rawcall("exit_bundle")) if self.form_out.is_lines
                                 else self.form_out.material())
        return self._mats["out"]


WordModule.focus = _io_wrap("focus", WordModule.focus)


class TransitionModule:
    """carries strands from a_exit (a Bundle) to b_entry (a Bundle) over `dur` seconds. Construct with both bundles, or
    construct bare (params only) and let Sequence call bind(a_exit, b_entry, dur). Subclass: implement build() and layers(t)."""
    kind = "transition"
    name = ""
    description = ""
    DEMO_PARAMS = {}         # library clip params (the straight-ribbon demo)
    DEMO_DUR = 3.0           # library clip duration (s)
    a_word = None            # the neighbouring word modules, set by Sequence before bind() (None when built bare)
    b_word = None

    def __init__(self, a_exit=None, b_entry=None, dur=None, **params):
        a_exit, b_entry = self._coerce(a_exit, b_entry)
        self.a_exit, self.b_entry, self.params = a_exit, b_entry, params
        self.dur = None if dur is None else float(dur)       # None: sized from the bundles at bind time (auto_dur)
        if a_exit is not None and b_entry is not None:
            if self.dur is None:
                self.dur = self.auto_dur(a_exit, b_entry)
            self.build()

    @staticmethod
    def demo_bundles(N=18, tube=1.0, v=520.0, palette=None):
        """a straight ribbon of N strands entering from the left (heads at x=150, bodies trailing off-frame) and a ribbon
        leaving to the right (tails at x=980): the library clip of a transition runs between them."""
        pal = Palette.get(palette)
        y = np.linspace(165, 375, N)
        s = np.linspace(0, 1, 8)[None, :, None]
        z = np.zeros(N)
        a = np.stack([150.0 - 400.0 * s[..., 0] + 0 * y[:, None], np.repeat(y[:, None], 8, 1), np.zeros((N, 8))], 2)
        b = np.stack([1380.0 - 400.0 * s[..., 0] + 0 * y[:, None], np.repeat(y[:, None], 8, 1), np.zeros((N, 8))], 2)
        col = np.array([pal.accents[i % len(pal.accents)] if tube else pal.fg for i in range(N)], float)
        w = 4.0 if tube else 1.5
        d = np.tile([1.0, 0, 0], (N, 1))
        return (Bundle(a, d, col, w, 0.0, v, tube), Bundle(b, d, col, w, 0.0, v, tube))

    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        """duration that keeps strand speed near the neighbours' hand-off speeds (no slow dip mid-way)."""
        d = float(np.linalg.norm(b_entry.centroid() - a_exit.centroid())) * 1.2
        d += float(np.mean(np.linalg.norm(b_entry.P[:, 0] - b_entry.P[:, -1], axis=1)))
        v = max(0.5 * (float(a_exit.v.mean()) + float(b_entry.v.mean())), 200.0)
        return float(min(max(1.7 * d / v, 1.0), 3.2))

    @staticmethod
    def _coerce(a, b):
        """a transition consumes strands: any other Material kind (points, blobs) is converted (phase 3a)."""
        if (a is not None and not isinstance(a, Bundle)) or (b is not None and not isinstance(b, Bundle)):
            from mograph.modules.material import ensure_bundle
            a, b = (None if a is None else ensure_bundle(a, b)), (None if b is None else ensure_bundle(b, a))
        return a, b

    def bind(self, a_exit, b_entry, dur=None):
        a_exit, b_entry = self._coerce(a_exit, b_entry)
        self.a_exit, self.b_entry = a_exit, b_entry
        if dur is not None:
            self.dur = float(dur)
        elif self.dur is None:
            self.dur = self.auto_dur(a_exit, b_entry)
        self.build()
        return self

    @property
    def duration(self):
        return self.dur

    def build(self):
        raise NotImplementedError

    def layers(self, t):
        raise NotImplementedError

    def focus(self, t):
        return 0.5 * (self.a_exit.centroid() + self.b_entry.centroid())

    def bounds_points(self):
        return np.concatenate([self.a_exit.P.reshape(-1, 3), self.b_entry.P.reshape(-1, 3)])
