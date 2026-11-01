"""dimtype vector engine: glyph contours, iso affines, stacked/solid extrusion, travelling strands, video out.
All drawing in screen space at supersample SS with cv2 AA + subpixel shift."""
import os, math, subprocess, time, json
import numpy as np
import cv2
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")).replace("\\", "/") + "/"
FD = REPO + "mograph/fonts/"
# optional dev-only font folders (not shipped): set MOGRAPH_FONT_TMP to a folder holding fonts_new/, fonts_fat/, ...
TMP = os.environ.get("MOGRAPH_FONT_TMP", REPO + "_fonts_tmp").replace("\\", "/").rstrip("/") + "/"
FALLBACK = [TMP + d + "/" for d in ("fonts_new", "fonts_fat", "fonts_decon", "fonts_bl", "fonts_heavy")]
OUTDIR = REPO + "11_mathtype/v020/local/dimtype/"      # callers (library.py) overwrite OUTDIR / GEN before writing
GEN = REPO + "11_mathtype/v020/local/GENERATIONS.md"

FONTS = {
    "bungee": "Bungee-Regular.ttf", "bungee_hair": "BungeeHairline-Regular.ttf", "ultra": "Ultra-Regular.ttf",
    "blackout": "Blackout_Midnight.ttf", "alien": "AlienBlock-Regular.ttf", "bartle": "BBHBartle-Regular.ttf",
    "badeen": "BadeenDisplay-Regular.ttf", "rubik": "Rubik[wght].ttf", "silk": "Silkscreen-Bold.ttf",
    "silk_r": "Silkscreen-Regular.ttf", "bytes": "Bytesized-Regular.ttf", "spacemono": "SpaceMono-Bold.ttf",
    "jet": "JetBrainsMono-Bold.ttf", "major": "MajorMonoDisplay-Regular.ttf", "bigsh": "BigShouldersDisplay[wght].ttf",
    "monofett": "Monofett-Regular.ttf", "syne": "Syne[wght].ttf", "anybody": "Anybody[wdth,wght].ttf",
    "nixie": "NixieOne-Regular.ttf", "tourney": "Tourney[wdth,wght].ttf", "interlope": "Interlope-Regular.ttf",
    "spacegro": "SpaceGrotesk-Bold.ttf", "pilow": "Pilowlava-Regular.otf", "doto": "Doto[ROND,wght].ttf",
    "geistpx": "GeistPixel[ELSH].ttf", "monoton": "Monoton-Regular.ttf", "codystar": "Codystar-Regular.ttf",
    "rubiklines": "RubikLines-Regular.ttf", "facade": "FacadeVF.ttf", "passion": "PassionOne-Black.ttf",
    "erica": "EricaOne-Regular.ttf", "jacq12": "Jacquard12-Regular.ttf",
}
TMPFONTS = {"erica": TMP + "fonts_fat/ericaone/EricaOne-Regular.ttf",
            "passion": TMP + "fonts_heavy/passionone/PassionOne-Black.ttf",
            "fascinate": TMP + "fonts_fat/fascinate/Fascinate-Regular.ttf",
            "moirai": TMP + "fonts_fat/moiraione/MoiraiOne-Regular.ttf",
            "jacq24": TMP + "fonts_bl/jacquard24/Jacquard24-Regular.ttf"}

INK, BONE, GREY, DIM = (12, 12, 11), (236, 230, 218), (120, 116, 108), (52, 50, 46)
AMBER, VERM, COBALT, CORAL = (242, 166, 90), (210, 70, 42), (45, 75, 216), (232, 111, 104)


def font_path(key):
    if key in TMPFONTS and os.path.exists(TMPFONTS[key]):
        return TMPFONTS[key]
    fn = FONTS[key]
    if os.path.exists(FD + fn):
        return FD + fn
    for d in FALLBACK:
        for root, _, files in os.walk(d):
            if fn in files:
                return os.path.join(root, fn)
    raise FileNotFoundError(fn)


# ---------------------------------------------------------------- glyph outlines
class FlatPen(BasePen):
    def __init__(self, gs, step=6.0):
        super().__init__(gs)
        self.contours, self.cur, self.step = [], [], step

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = np.array(self.cur[-1]); P = np.array([p0, p1, p2, p3], float)
        L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        n = max(2, int(L / self.step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            m = 1 - t
            self.cur.append(tuple(m**3 * P[0] + 3 * m * m * t * P[1] + 3 * m * t * t * P[2] + t**3 * P[3]))

    def _qCurveToOne(self, p1, p2):
        p0 = np.array(self.cur[-1]); P = np.array([p0, p1, p2], float)
        L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
        n = max(2, int(L / self.step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            m = 1 - t
            self.cur.append(tuple(m * m * P[0] + 2 * m * t * P[1] + t * t * P[2]))

    def _closePath(self):
        c = np.array(self.cur, float)
        if len(c) > 2:
            if np.allclose(c[0], c[-1]):
                c = c[:-1]
            # drop consecutive duplicates
            keep = np.r_[True, np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-6]
            c = c[keep]
            if len(c) > 2:
                self.contours.append(c)
        self.cur = []

    _endPath = _closePath


_FCACHE = {}
_TTCACHE = {}


def _font(key, loc):
    k = (key, json.dumps(loc, sort_keys=True) if loc else "")
    if k not in _FCACHE:
        if key not in _TTCACHE:                  # one TTFont per file; a glyph set per variation location (fusion: animated axes)
            _TTCACHE[key] = TTFont(font_path(key))
        f = _TTCACHE[key]
        gs = f.getGlyphSet(location=loc) if loc else f.getGlyphSet()
        _FCACHE[k] = (f, gs)
    return _FCACHE[k]


def word(key, text, loc=None, track=0.0, step=6.0):
    """-> list of letters; each letter = list of contours (font units, y up), plus total advance.
    Letters are positioned along x. Missing glyphs raise."""
    f, gs = _font(key, loc)
    cmap = f.getBestCmap()
    upm = f["head"].unitsPerEm
    x, letters = 0.0, []
    for ch in text:
        if ch == " ":
            x += gs[cmap[32]].width if 32 in cmap else upm * 0.3
            continue
        if ord(ch) not in cmap:
            raise KeyError(f"{key} lacks {ch!r}")
        g = gs[cmap[ord(ch)]]
        pen = FlatPen(gs, step)
        g.draw(pen)
        cs = [c + [x, 0] for c in pen.contours]
        if cs:
            letters.append({"ch": ch, "contours": cs, "x0": x, "adv": g.width})
        x += g.width + track * upm
    return letters, upm


def bbox(letters):
    P = np.concatenate([c for L in letters for c in L["contours"]])
    return P.min(0), P.max(0)


def place(letters, x, y, w, h, mode="stretch"):
    """map letters (font units, y up) into rect (px, y down). mode stretch|fit|fitw|fith."""
    lo, hi = bbox(letters)
    sx, sy = w / (hi[0] - lo[0]), h / (hi[1] - lo[1])
    if mode == "fit":
        sx = sy = min(sx, sy)
    elif mode == "fitw":
        sy = sx
    elif mode == "fith":
        sx = sy
    out = []
    for L in letters:
        cs = [np.stack([x + (c[:, 0] - lo[0]) * sx, y + h - (c[:, 1] - lo[1]) * sy], 1) for c in L["contours"]]
        out.append({"ch": L["ch"], "contours": cs})
    return out


def xform(letters, A, origin):
    """apply 2x2 affine A about origin to all contours."""
    o = np.asarray(origin, float)
    return [{"ch": L["ch"], "contours": [(c - o) @ np.asarray(A).T + o for c in L["contours"]]} for L in letters]


def iso_A(face="right", amt=1.0):
    """2x2 affine taking flat (x right, y down) into an iso face plane; amt in [0,1] blends from identity."""
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    if face == "right":      # x -> up-right, y -> down
        T = np.array([[c, 0], [-s, 1]])
    elif face == "left":     # x -> down-right
        T = np.array([[c, 0], [s, 1]])
    elif face == "top":      # x -> down-right, y -> down-left  (lying on the ground)
        T = np.array([[c, -c], [s, s]])
    else:
        T = np.eye(2)
    return (1 - amt) * np.eye(2) + amt * T


ISO = {"ur": np.array([math.cos(math.radians(30)), -0.5]), "dl": np.array([-math.cos(math.radians(30)), 0.5]),
       "ul": np.array([-math.cos(math.radians(30)), -0.5]), "dr": np.array([math.cos(math.radians(30)), 0.5]),
       "u": np.array([0.0, -1.0]), "d": np.array([0.0, 1.0])}


# ---------------------------------------------------------------- polylines
def arclen(P, closed=False):
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    return np.r_[0, np.cumsum(seg)]


def resample(P, step, closed=True):
    Q = np.vstack([P, P[:1]]) if closed else P
    cum = arclen(Q)
    n = max(3, int(round(cum[-1] / step)))
    s = np.linspace(0, cum[-1], n + (0 if closed else 1), endpoint=not closed)
    return np.stack([np.interp(s, cum, Q[:, 0]), np.interp(s, cum, Q[:, 1])], 1)


def split_strands(c, corner_deg=38.0, max_len=1e9, min_len=0.0):
    """split a closed contour at corners into open polylines (with shared end points)."""
    n = len(c)
    d_in = c - np.roll(c, 1, 0)
    d_out = np.roll(c, -1, 0) - c
    a = np.arctan2(d_in[:, 1], d_in[:, 0]); b = np.arctan2(d_out[:, 1], d_out[:, 0])
    turn = np.abs((b - a + np.pi) % (2 * np.pi) - np.pi)
    corners = list(np.where(turn > math.radians(corner_deg))[0])
    if not corners:
        corners = [0]
    strands = []
    for i, ci in enumerate(corners):
        cj = corners[(i + 1) % len(corners)]
        idx = list(range(ci, cj + 1)) if cj > ci else list(range(ci, n)) + list(range(0, cj + 1))
        P = c[idx]
        L = arclen(P)[-1]
        if L > max_len:
            k = int(math.ceil(L / max_len))
            cum = arclen(P)
            cuts = np.linspace(0, L, k + 1)
            for j in range(k):
                strands.append(sub(P, cum, cuts[j], cuts[j + 1]))
        else:
            strands.append(P)
    return [s for s in strands if arclen(s)[-1] >= min_len]


def sub(P, cum, a, b):
    """sub-polyline of P between arc lengths a<b."""
    a = max(0.0, min(a, cum[-1])); b = max(0.0, min(b, cum[-1]))
    if b <= a + 1e-6:
        return None
    i0 = np.searchsorted(cum, a, "right"); i1 = np.searchsorted(cum, b, "left")
    pa = np.array([np.interp(a, cum, P[:, 0]), np.interp(a, cum, P[:, 1])])
    pb = np.array([np.interp(b, cum, P[:, 0]), np.interp(b, cum, P[:, 1])])
    return np.vstack([pa, P[i0:i1], pb])


def end_tangent(P, start=False, look=6.0):
    cum = arclen(P)
    if start:
        q = np.array([np.interp(min(look, cum[-1]), cum, P[:, 0]), np.interp(min(look, cum[-1]), cum, P[:, 1])])
        d = P[0] - q
    else:
        L = cum[-1]
        q = np.array([np.interp(max(0, L - look), cum, P[:, 0]), np.interp(max(0, L - look), cum, P[:, 1])])
        d = P[-1] - q
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


class Route:
    """polyline route with cumulative length; window(a,b) returns the visible piece."""
    def __init__(self, P):
        self.P = np.asarray(P, float)
        self.cum = arclen(self.P)
        self.L = self.cum[-1]

    def window(self, a, b):
        return sub(self.P, self.cum, a, b)


# ---------------------------------------------------------------- easing
def clamp01(x):
    return min(1.0, max(0.0, x))


def ss(a, b, x):
    u = clamp01((x - a) / (b - a)) if b != a else float(x >= b)
    return u * u * (3 - 2 * u)


def ease_io(u, p=3.0):
    u = clamp01(u)
    return 0.5 * (2 * u) ** p if u < 0.5 else 1 - 0.5 * (2 * (1 - u)) ** p


def ease_out(u, p=3.0):
    u = clamp01(u)
    return 1 - (1 - u) ** p


def ease_in(u, p=3.0):
    u = clamp01(u)
    return u ** p


def seg(t, a, b):
    return clamp01((t - a) / (b - a))


# ---------------------------------------------------------------- drawing
SHIFT = 4
_SF = float(1 << SHIFT)


def ipts(P, S):
    return np.round(np.asarray(P) * S * _SF).astype(np.int32)


def canvas(W, H, S, col):
    im = np.empty((H * S, W * S, 3), np.uint8)
    im[:] = col[::-1]  # BGR
    return im


def fill(im, polys, col, S):
    polys = [ipts(p, S) for p in polys if p is not None and len(p) > 2]
    if polys:
        cv2.fillPoly(im, polys, col[::-1], cv2.LINE_AA, SHIFT)


def lines(im, polys, col, S, w=1.0, closed=False):
    polys = [ipts(p, S) for p in polys if p is not None and len(p) > 1]
    if polys:
        cv2.polylines(im, polys, closed, col[::-1], max(1, int(round(w * S))), cv2.LINE_AA, SHIFT)


def blend(im, layer_fn, alpha):
    """draw layer_fn on a copy and blend by alpha (for fades)."""
    if alpha <= 0.002:
        return
    if alpha >= 0.998:
        layer_fn(im); return
    tmp = im.copy(); layer_fn(tmp)
    cv2.addWeighted(tmp, alpha, im, 1 - alpha, 0, dst=im)


def down(im, S):
    if S == 1:
        return im[:, :, ::-1]
    h, w = im.shape[:2]
    return cv2.resize(im, (w // S, h // S), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def mix(c1, c2, t):
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


# ---------------------------------------------------------------- extrusion (solid, painter's)
def solid_extrude(im, letters, d, S, cap_col, wall_cols, light=(-0.6, -0.8), edge=None, edge_w=1.0, levels=3,
                  ground=None, wall_alpha=1.0, Ainv=None):
    """oblique extrusion of planar letters by screen vector d (cap at the front, back = cap + d).
    walls sorted by projection on d (far first). wall colour from 2D edge normal vs light, posterised.
    wall_cols: (lit, mid, dark)."""
    d = np.asarray(d, float)
    dn = np.linalg.norm(d)
    if dn < 0.3:
        fill(im, [c for L in letters for c in L["contours"]], cap_col, S)
        if edge is not None:
            lines(im, [c for L in letters for c in L["contours"]], edge, S, edge_w, closed=True)
        return
    e = d / dn
    quads, keys, cols = [], [], []
    lt = np.array(light, float); lt /= np.linalg.norm(lt)
    for L in letters:
        for c in L["contours"]:
            p0 = c; p1 = np.roll(c, -1, 0)
            t = p1 - p0
            if Ainv is not None:
                t = t @ np.asarray(Ainv).T     # shade by the edge direction in the flat glyph plane
            n = np.stack([t[:, 1], -t[:, 0]], 1)   # outward for one orientation; fix via facing test below
            nl = np.linalg.norm(n, axis=1) + 1e-9
            n = n / nl[:, None]
            # wall visible if its normal has component opposite to e (faces viewer side)... both shown, sorted
            mid = 0.5 * (p0 + p1)
            k = -(mid @ e)
            sh = np.abs(n @ lt)
            for i in range(len(c)):
                quads.append(np.array([p0[i], p1[i], p1[i] + d, p0[i] + d]))
                keys.append(k[i])
                cols.append(sh[i])
    order = np.argsort(keys)  # most negative first? key=-(mid.e): bigger mid.e (= nearer) -> smaller key -> drawn first
    # nearer walls must be drawn LAST: nearer = larger mid.e projection?  see notes in caller tests
    order = order[::-1]
    # posterise shade
    sh = np.array(cols)
    if levels:
        q = np.clip((sh * levels).astype(int), 0, levels - 1)
    for i in order:
        if levels:
            col = wall_cols[::-1][q[i]] if levels == len(wall_cols) else wall_cols[min(q[i], len(wall_cols) - 1)]
        else:
            col = mix(wall_cols[2], wall_cols[0], sh[i])
        fill(im, [quads[i]], col, S)
        if edge is not None:
            lines(im, [quads[i]], edge, S, edge_w, closed=True)
    fill(im, [c for L in letters for c in L["contours"]], cap_col, S)
    if edge is not None:
        lines(im, [c for L in letters for c in L["contours"]], edge, S, edge_w, closed=True)


# ---------------------------------------------------------------- render loop
def _work(args):
    fn, i = args
    return fn(i)


LAST_STATS = None


def render(frame_fn, path, n, W, H, fps=30, procs=8, crf=16, init=None, initargs=()):
    """frame_fn(i)->HxWx3 uint8 RGB. Uses a process pool; frame_fn must be picklable (module-level)."""
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    t0 = time.time()
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    global LAST_STATS
    diffs, prev = [], [None]

    def _st(fr):
        sm = cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev[0] is not None:
            diffs.append(float(np.abs(sm - prev[0]).mean()))
        prev[0] = sm
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs, initializer=init, initargs=initargs) as pool:
            for fr in pool.imap(frame_fn, range(n), chunksize=2):
                _st(fr)
                p.stdin.write(np.ascontiguousarray(fr).tobytes())
    else:
        for i in range(n):
            fr = frame_fn(i); _st(fr)
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
    LAST_STATS = np.array(diffs)
    p.stdin.close(); p.wait()
    return time.time() - t0


def next_gen(stem, kind):
    g = 1
    while any(os.path.exists(f"{OUTDIR}{stem}_g{g:02d}_{kind}.{ext}") for ext in ("mp4", "png")):
        g += 1
    return g


def log_gen(name, parent, change, why, secs):
    new = not os.path.exists(GEN)
    with open(GEN, "a", encoding="utf-8") as f:
        if new:
            f.write("# v020 local GENERATIONS (dimtype round 4, notes applied)\n\n")
        f.write(f"- {name} | parent: {parent} | {change} | why: {why} | render {secs:.0f}s\n")


# ---------------------------------------------------------------- layout helpers
def base(key, text, loc=None, track=0.0, aspect=1.0, h=100.0, step=6.0):
    """letters in px at height h, y down, origin top-left of bbox; aspect stretches x."""
    L, _ = word(key, text, loc, track, step)
    lo, hi = bbox(L)
    s = h / (hi[1] - lo[1])
    return place(L, 0, 0, (hi[0] - lo[0]) * s * aspect, h, "stretch")


def bbox_px(letters):
    P = np.concatenate([c for L in letters for c in L["contours"]])
    return P.min(0), P.max(0)


def fit_into(letters, A, x, y, w, h, anchor=(0.5, 0.5), mode="fit"):
    """apply 2x2 A, then uniformly scale (fit/fitw/fith) and translate into rect at anchor."""
    T = [{"ch": L["ch"], "contours": [c @ np.asarray(A).T for c in L["contours"]]} for L in letters]
    lo, hi = bbox_px(T)
    sx, sy = w / (hi[0] - lo[0]), h / (hi[1] - lo[1])
    s = min(sx, sy) if mode == "fit" else (sx if mode == "fitw" else sy)
    bw, bh = (hi - lo) * s
    ox = x + (w - bw) * anchor[0] - lo[0] * s
    oy = y + (h - bh) * anchor[1] - lo[1] * s
    return [{"ch": L["ch"], "contours": [c * s + [ox, oy] for c in L["contours"]]} for L in T]


def lerp_letters(A, B, u):
    return [{"ch": a["ch"], "contours": [ca + (cb - ca) * u for ca, cb in zip(a["contours"], b["contours"])]}
            for a, b in zip(A, B)]


def allc(letters):
    return [c for L in letters for c in L["contours"]]


def exit_len(p, d, W, H, pad=40.0):
    """distance along unit d from p until outside the padded frame."""
    ts = []
    for k, lo, hi in ((0, -pad, W + pad), (1, -pad, H + pad)):
        if abs(d[k]) > 1e-9:
            ts += [(lo - p[k]) / d[k], (hi - p[k]) / d[k]]
    ts = [t for t in ts if t > 0]
    return max(0.0, min(ts)) if ts else 0.0


ISO6 = np.array([ISO[k] for k in ("u", "d", "ur", "dl", "ul", "dr")])


def snap_dir(T, dirs=None):
    D = ISO6 if dirs is None else dirs
    return D[int(np.argmax(D @ T))]


def bez2(a, b, c, n=8):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c


def run_route(P, mode, W, H, r=14.0, snap=True, extra=20.0):
    """exit: strand P then continue along (snapped) end tangent off frame.
    enter: come from off frame along (snapped) start tangent into P. returns (Route, L, E)."""
    L = arclen(P)[-1]
    if mode == "exit":
        T = end_tangent(P); Ts = snap_dir(T) if snap else T
        tail = [P[-1]]
        if snap and T @ Ts < 0.995:
            tail = list(bez2(P[-1], P[-1] + T * r, P[-1] + T * r + Ts * r)[1:])
        q = tail[-1]
        E0 = arclen(np.array([P[-1]] + tail))[-1] if len(tail) > 1 else 0.0
        E = E0 + exit_len(q, Ts, W, H) + L + extra
        mid = np.array(tail if len(tail) > 1 else []).reshape(-1, 2)
        R = np.vstack([P, mid, q + Ts * (E - E0)])
    else:
        T = end_tangent(P, start=True); Ts = snap_dir(T) if snap else T
        head = [P[0]]
        if snap and T @ Ts < 0.995:
            head = list(bez2(P[0], P[0] + T * r, P[0] + T * r + Ts * r)[1:])
        q = head[-1]
        E0 = arclen(np.array([P[0]] + head))[-1] if len(head) > 1 else 0.0
        E = E0 + exit_len(q, Ts, W, H) + L + extra
        pre = np.array(head[::-1] if len(head) > 1 else []).reshape(-1, 2)
        R = np.vstack([q + Ts * (E - E0), pre, P])
    return Route(R), L, E


def caption(key, text, x, y, h, anchor=(0, 0), A=None, track=0.05, loc=None):
    """small text as filled contours; (x,y) anchored corner of bbox; h = cap height px."""
    b = base(key, text, loc=loc, track=track, h=h, step=2.0)
    A = np.eye(2) if A is None else A
    T = [{"ch": L["ch"], "contours": [c @ np.asarray(A).T for c in L["contours"]]} for L in b]
    lo, hi = bbox_px(T)
    off = np.array([x, y]) - lo - (hi - lo) * np.asarray(anchor)
    return [{"ch": L["ch"], "contours": [c + off for c in L["contours"]]} for L in T]


def sheet(ims, cols=3, scale=0.5):
    rows = []
    for j in range(0, len(ims), cols):
        r = ims[j:j + cols]
        r = r + [np.zeros_like(ims[0])] * (cols - len(r))
        rows.append(np.hstack(r))
    sh = np.vstack(rows)
    return cv2.resize(sh, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def motion_stats(frames):
    """mean abs diff between consecutive frames (0-255) per step: reveals pops."""
    d = [float(np.abs(frames[i + 1].astype(np.int16) - frames[i].astype(np.int16)).mean()) for i in range(len(frames) - 1)]
    return np.array(d)


# ---------------------------------------------------------------- v017 additions: wireframe extrusion, split palettes
def feature_idx(c, e, corner_deg=35.0):
    """vertex indices of closed contour c that carry a visible depth edge for extrusion along unit e:
    corners (turn > corner_deg) and silhouette tangencies (edge direction crosses e)."""
    d_in = c - np.roll(c, 1, 0); d_out = np.roll(c, -1, 0) - c
    a = np.arctan2(d_in[:, 1], d_in[:, 0]); b = np.arctan2(d_out[:, 1], d_out[:, 0])
    turn = np.abs((b - a + np.pi) % (2 * np.pi) - np.pi)
    cr_in = d_in[:, 0] * e[1] - d_in[:, 1] * e[0]
    cr_out = d_out[:, 0] * e[1] - d_out[:, 1] * e[0]
    sil = np.sign(cr_in) != np.sign(cr_out)
    return np.where((turn > math.radians(corner_deg)) | sil)[0]


def wire_extrude(im, letters, e, z0, z1, S, fg, bg, lw=1.0, cap=None, wall=None, cap_edge=True, back_edge=True,
                 zfn=None):
    """oblique extrusion between depths z0 (front) and z1 (back) along unit e, drawn painter's style.
    walls filled with `wall` (default bg), depth lines only at feature vertices, back contour edges.
    zfn(points)->per-vertex back depth (overrides z1) for staggered heads."""
    e = np.asarray(e, float)
    quads, keys, edges = [], [], []
    caps = []
    for L in letters:
        for c in L["contours"]:
            f = c + e * z0
            caps.append(f)
            zb = np.full(len(c), float(z1)) if zfn is None else np.maximum(z0, zfn(c))
            bk = c + e[None, :] * zb[:, None]
            feat = np.zeros(len(c), bool); feat[feature_idx(c, e)] = True
            n = len(c)
            for i in range(n):
                j = (i + 1) % n
                q = np.array([f[i], f[j], bk[j], bk[i]])
                quads.append(q)
                keys.append(-(0.5 * (c[i] + c[j]) @ e))
                ed = [[bk[i], bk[j]]] if back_edge else []
                if feat[i]:
                    ed.append([f[i], bk[i]])
                if feat[j]:
                    ed.append([f[j], bk[j]])
                edges.append(ed)
    wcol = bg if wall is None else wall
    if isinstance(wcol, str):   # "none": x-ray wireframe, every depth edge visible
        if abs(z1 - z0) > 0.3 or zfn is not None:
            lines(im, [np.array(p) for ed in edges for p in ed], fg, S, lw)
        fill(im, caps, fg if cap is None else cap, S)
        return
    shaded = isinstance(wcol[0], (tuple, list))
    if abs(z1 - z0) > 0.3 or zfn is not None:
        for i in np.argsort(keys)[::-1]:
            if shaded:   # (lit, dark) by edge direction: walls whose base edge runs near-horizontal are lit
                q = quads[i]; tv = q[1] - q[0]; nt = np.linalg.norm(tv) + 1e-9
                fill(im, [q], wcol[0] if abs(tv[1]) / nt < 0.6 else wcol[1], S)
            else:
                fill(im, [quads[i]], wcol, S)
            lines(im, [np.array(p) for p in edges[i]], fg, S, lw)
    fill(im, caps, fg if cap is None else cap, S)
    if cap_edge:
        lines(im, caps, fg, S, lw, closed=True)


def split_compose(W, H, S, draw, field, dark=(INK, BONE), light=(BONE, INK)):
    """draw(im, fg, bg) twice; inside `field` polys use dark palette (bg, fg), outside light. returns RGB."""
    A = canvas(W, H, S, dark[0]); draw(A, dark[1], dark[0])
    B = canvas(W, H, S, light[0]); draw(B, light[1], light[0])
    m = np.zeros((H * S, W * S), np.uint8)
    pl = [ipts(p, S) for p in field if p is not None and len(p) > 2]
    if pl:
        cv2.fillPoly(m, pl, 255, cv2.LINE_AA, SHIFT)
    a = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, None] / 255.0
    ra, rb = down(A, S).astype(np.float32), down(B, S).astype(np.float32)
    return (ra * a + rb * (1 - a) + 0.5).astype(np.uint8)


def jag_field(W, H, xs, side="left"):
    """stepped vertical split: xs = list of (y0, y1, x). returns polygon covering the `side` part."""
    pts = []
    for y0, y1, x in xs:
        pts += [(x, y0), (x, y1)]
    P = np.array(pts, float)
    if side == "left":
        return np.vstack([[[-10, -10]], [[P[0, 0], -10]], P, [[P[-1, 0], H + 10]], [[-10, H + 10]]])
    return np.vstack([[[W + 10, -10]], [[P[0, 0], -10]], P, [[P[-1, 0], H + 10]], [[W + 10, H + 10]]])


def lazy(G, init):
    """for spawn-based pools: init from env DT_W, DT_H, DT_S once per process."""
    if not G:
        init(int(os.environ.get("DT_W", 960)), int(os.environ.get("DT_H", 540)), int(os.environ.get("DT_S", 2)))


# ---------------------------------------------------------------- v018 additions: motion blur, graduated extrude,
# perspective (vanishing-point) scaling, raster tunnels
def mb(frame_t, i, fps, sub=None, shutter=None):
    """subframe motion blur: average frame_t(t) over `sub` samples spread over `shutter` of a frame, centred on i/fps.
    sub/shutter default to env DT_MB (1 = off) / DT_SHUT (0.5)."""
    sub = int(os.environ.get("DT_MB", 1)) if sub is None else sub
    shutter = float(os.environ.get("DT_SHUT", 0.5)) if shutter is None else shutter
    if sub <= 1:
        return frame_t(i / fps)
    acc = None
    for k in range(sub):
        t = (i + shutter * ((k + 0.5) / sub - 0.5)) / fps
        f = frame_t(t).astype(np.float32)
        acc = f if acc is None else acc + f
    return (acc / sub + 0.5).astype(np.uint8)


def scale_about(letters, o, s):
    o = np.asarray(o, float)
    return [{"ch": L["ch"], "contours": [o + (c - o) * s for c in L["contours"]]} for L in letters]


def grad_extrude(im, letters, d, S, cap_col, near_col, far_col, n=None, shade=0.35, light=(-0.6, -0.8), Ainv=None,
                 gamma=1.0, edge=None, edge_w=1.0, vp=None, s_back=1.0):
    """solid extrusion with a smooth tone gradient along depth: walls cut into n slabs, back slab first,
    each slab's walls painter-sorted; colour = mix(near, far, depth^gamma), darkened by shade*(1-|n.light|).
    vp/s_back: optional perspective, slab at depth u is scaled about vp by 1+(s_back-1)*u (after adding d*u)."""
    d = np.asarray(d, float)
    dn = np.linalg.norm(d)
    cs = allc(letters)
    if dn < 0.3 and (vp is None or abs(s_back - 1) < 1e-3):
        fill(im, cs, cap_col, S)
        return
    if n is None:
        n = int(np.clip(dn / 2.0, 6, 90))
    lt = np.array(light, float); lt /= np.linalg.norm(lt)
    e = d / (dn + 1e-9)
    walls = []

    def sarea(c):
        return 0.5 * float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
    csg = []
    for L in letters:          # outward-from-solid normal sign per letter, from its largest contour
        ar = [sarea(c) for c in L["contours"]]
        sg = np.sign(ar[int(np.argmax(np.abs(ar)))]) if ar else 1.0
        csg += [sg] * len(L["contours"])
    for c, sg in zip(cs, csg):
        p0 = c; p1 = np.roll(c, -1, 0); t = p1 - p0
        no = sg * np.stack([t[:, 1], -t[:, 0]], 1)
        mid0 = 0.5 * (p0 + p1)
        tow = (vp - mid0) if (vp is not None and dn < 1.0) else d[None, :]
        vis = np.sum(no * tow, axis=1) > 0      # back-face cull: only walls facing the depth direction show
        if not vis.any():
            continue
        p0, p1, t = p0[vis], p1[vis], t[vis]
        if Ainv is not None:
            t = t @ np.asarray(Ainv).T
        nn = np.stack([t[:, 1], -t[:, 0]], 1); nn /= (np.linalg.norm(nn, axis=1)[:, None] + 1e-9)
        mid = 0.5 * (p0 + p1)
        key = -np.linalg.norm(mid - vp, axis=1) if (vp is not None and dn < 1.0) else -(mid @ e)
        walls.append((p0, p1, np.abs(nn @ lt), key))

    def at(P, u):
        Q = P + d * u
        if vp is not None:
            Q = vp + (Q - vp) * (1 + (s_back - 1) * u)
        return Q
    us = np.linspace(0, 1, n + 1)
    for k in range(n - 1, -1, -1):
        ua, ub = us[k], us[k + 1]
        tone = ((ua + ub) * 0.5) ** gamma
        base_c = np.array(mix(near_col, far_col, tone), float)
        quads, keys, cols = [], [], []
        for p0, p1, sh, key in walls:
            A0, A1, B0, B1 = at(p0, ua), at(p1, ua), at(p0, ub), at(p1, ub)
            quads.append(np.stack([A0, A1, B1, B0], 1)); keys.append(key)
            cols.append(1 - shade * (1 - sh))
        q = np.concatenate(quads); kk = np.concatenate(keys); cc = np.concatenate(cols)
        order = np.argsort(kk)[::-1]
        lv = np.round(cc * 16).astype(int)
        run, rc = [], None
        for i in order:     # batch consecutive same-tone quads (keeps painter order)
            if rc is not None and lv[i] != rc:
                cc_ = tuple(int(x) for x in np.clip(base_c * rc / 16, 0, 255))
                fill(im, run, cc_, S); lines(im, run, cc_, S, 0.5, closed=True); run = []
            rc = lv[i]; run.append(q[i])
        if run:
            cc_ = tuple(int(x) for x in np.clip(base_c * rc / 16, 0, 255))
            fill(im, run, cc_, S); lines(im, run, cc_, S, 0.5, closed=True)
    fill(im, cs, cap_col, S)
    if edge is not None:
        lines(im, cs, edge, S, edge_w, closed=True)


def tunnel(im, letters, vp, depth_s, S, cols_fn, floor_col, n=40, rings=0, ring_col=None, lip=None,
           light=None, shade=0.0, rlines=None):
    """look INTO the letters: each letter is an opening whose straight walls recede toward vanishing point vp.
    depth_s: scale of the deepest cross-section (1 = flat, 0.4 = deep); a list gives per-letter depth.
    cols_fn(u)->RGB wall colour at depth u in [0,1]; every `rings`th slab uses ring_col (depth contour bands).
    Each pixel takes the colour of the depth at which its ray first leaves the cross-section (raster at S)."""
    Hs, Ws = im.shape[:2]
    vp = np.asarray(vp, float)
    for li, L in enumerate(letters):
        ds = depth_s[li] if isinstance(depth_s, (list, tuple, np.ndarray)) else depth_s
        cs = L["contours"]
        P = np.concatenate(cs)
        lo = np.floor(np.maximum(P.min(0) - 2, 0) * S).astype(int)
        hi = np.ceil(np.minimum(P.max(0) + 2, [Ws / S, Hs / S]) * S).astype(int)
        if (hi - lo).min() <= 2:
            continue
        w, h = hi - lo
        off = lo / S

        def m_of(s):
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [ipts(vp + (c - vp) * s - off, S) for c in cs], 255, cv2.LINE_AA, SHIFT)
            return m.astype(np.float32) / 255.0
        open_m = m_of(1.0)
        run = open_m.copy()
        out = np.zeros((h, w, 3), np.float32)
        ov_a = np.zeros((h, w), np.float32); ov_c = np.zeros((h, w, 3), np.float32)
        rl = sorted(rlines or [], key=lambda r: r[0])
        ri = 0
        if ds < 0.999:
            for k in range(1, n + 1):
                u = k / n
                while ri < len(rl) and rl[ri][0] <= u:     # ring lines at depth in ((k-1)/n, k/n]: visible where the ray is still inside
                    ru, rcol, rw = rl[ri]; ri += 1
                    if ru <= 0:
                        continue
                    mm = np.zeros((h, w), np.uint8)
                    cv2.polylines(mm, [ipts(vp + (c - vp) * (1 + (ds - 1) * ru) - off, S) for c in cs], True, 255,
                                  max(1, int(round(rw * S))), cv2.LINE_AA, SHIFT)
                    a = mm.astype(np.float32) / 255.0 * run * (1 - ov_a)
                    ov_c += a[:, :, None] * np.array(rcol, np.float32)[::-1]; ov_a += a
                nr = np.minimum(run, m_of(1 + (ds - 1) * u))
                hit = run - nr
                col = ring_col if (rings and k % rings == 0 and k < n) else cols_fn(u - 0.5 / n)
                out += hit[:, :, None] * np.array(col, np.float32)[::-1]
                run = nr
        if light is not None and shade > 0:   # wall seen at x faces roughly toward vp: shade by (vp - x).light
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dx = vp[0] - (xx + 0.5) / S - off[0]; dy = vp[1] - (yy + 0.5) / S - off[1]
            nr_ = np.sqrt(dx * dx + dy * dy) + 1e-3
            lt = np.array(light, float); lt /= np.linalg.norm(lt)
            f = 1 - shade * (0.5 - 0.5 * (dx * lt[0] + dy * lt[1]) / nr_)
            out *= f[:, :, None]
        out += run[:, :, None] * np.array(floor_col, np.float32)[::-1]
        out = out * (1 - ov_a[:, :, None]) + ov_c * open_m[:, :, None]
        sub = im[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32)
        res = sub * (1 - open_m[:, :, None]) + out
        im[lo[1]:hi[1], lo[0]:hi[0]] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
        if lip is not None:
            lines(im, cs, lip[0], S, lip[1], closed=True)
