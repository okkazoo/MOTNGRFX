"""ripple: A's exit strands are split into "letters" (contiguous groups along the word). Each letter expands from its own centre
(staggered left to right); its fill fades in slow, held steps (a slow glitch, ~8 Hz, not a flicker) while the stroke turns into an
outline; 2-3 echo outlines spawn behind it at the centre and expand after it like the rings of a drop in a puddle, thinning as they
go, until everything has left the frame. A few frame-held horizontal slice offsets (seeded, low frequency) shear the rings. B's
strands slide in over the tail of the rings and settle at b_entry (T.layers(0) draws a_exit, T.layers(dur) draws b_entry)."""
import math
import numpy as np
from mograph.modules.core import TransitionModule, register, strand_layers, unit
from mograph.render.cam3d import Layer

FPS_GLITCH = 8.0          # the held-step rate of the fill fade
PTS = 14                  # dense points per strand


def _ss(x):
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


def _dense(P, n=PTS):
    """resample a polyline (k,3) to n points by arc length."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return np.repeat(P[:1], n, 0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, c]) for c in range(3)], 1)


def _slice(poly, bands):
    """shear a polyline: parts inside a horizontal band (y0, y1, dx) move by dx in x; returns a list of polylines. Segments are
    cut exactly at the band edges."""
    pieces = [np.asarray(poly, float)]
    for y0, y1, dx in bands:
        nxt = []
        for P in pieces:
            for lo, hi, inside in ((-1e9, y0, False), (y0, y1, True), (y1, 1e9, False)):
                for seg in _clip_y(P, lo, hi):
                    nxt.append(seg + np.array([dx, 0.0, 0.0]) if inside else seg)
        pieces = nxt
    return [p for p in pieces if len(p) > 1]


def _clip_y(P, lo, hi):
    """parts of polyline P whose y lies in [lo, hi] (linear cut at the edges)."""
    out, cur = [], []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ya, yb = a[1], b[1]
        t0, t1 = 0.0, 1.0
        if abs(yb - ya) < 1e-9:
            if not (lo <= ya <= hi):
                if cur:
                    out.append(np.array(cur)); cur = []
                continue
        else:
            ta, tb = (lo - ya) / (yb - ya), (hi - ya) / (yb - ya)
            t0, t1 = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
            if t0 >= t1:
                if cur:
                    out.append(np.array(cur)); cur = []
                continue
        p0, p1 = a + (b - a) * t0, a + (b - a) * t1
        if cur and t0 <= 1e-9:
            cur.append(p1)
        else:
            if cur:
                out.append(np.array(cur))
            cur = [p0, p1]
        if t1 < 1.0 - 1e-9:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return [o for o in out if len(o) > 1]


@register("ripple")
class Ripple(TransitionModule):
    """params: rings (echo outlines per letter, 3), seed (glitch schedule, default from the strand count), glitch (0..1 how
    many slice events, 1.0), span (fraction of the duration one letter takes to leave, 0.5), letters (override the letter
    count; default = letters of the outgoing word, else from the strand count)."""
    DEMO_PARAMS = dict(rings=3)
    DEMO_DUR = 2.8
    SUITS = dict(
        brief="calm to mid energy, any distance: the word spreads out of frame in puddle rings before the next one arrives",
        base=(0.5, 2.0), energy=((-1, 0.0, 0.5, 0.85), 1.0), dur=(1.4, 3.2, 1.15),
        moods={"calm": 0.9, "melancholy": 0.9, "mysterious": 0.9, "nostalgic": 0.7, "joyful": 0.4, "playful": 0.3, "tense": 0.3,
               "hype": 0.2})
    description = ("The outgoing letters expand from their own centres, their fill fades out in slow held steps while the "
                   "strokes become outlines, and echo outlines follow like the rings of a drop in a puddle until all have "
                   "left the frame; the next word's strands slide in over the tail of the rings.")

    @classmethod
    def auto_dur(cls, a_exit, b_entry):
        return 2.4

    # ------------------------------------------------------------ build
    def _letter_count(self, K):
        n = self.params.get("letters")
        if n is None and self.a_word is not None:
            n = sum(ch.isalnum() for ln in getattr(self.a_word, "lines", []) for ch in ln)
        if not n:
            n = int(round(1.2 * math.sqrt(K)))
        return int(max(1, min(int(n), K, 9)))

    def build(self):
        P = self.params
        A, B = self.a_exit, self.b_entry
        K = A.N
        self.rings = int(max(0, min(int(P.get("rings", 3)), 4)))
        self.span = float(P.get("span", 0.5))
        rng = np.random.default_rng(int(P.get("seed", 1000 + K)))
        # letters: contiguous groups along the principal axis of the strand centroids
        cen = A.P.mean(1)
        c0 = cen - cen.mean(0)
        if np.abs(c0).max() < 1e-6:
            order = np.arange(K)
        else:
            ax = np.linalg.svd(c0, full_matrices=False)[2][0]
            order = np.argsort(c0 @ ax, kind="stable")
        G = self._letter_count(K)
        self.groups = [g for g in np.array_split(order, G) if len(g)]
        G = len(self.groups)
        self.dense = [_dense(A.P[k]) for k in range(K)]
        self.cg, self.S, self.delay, self.r0 = [], [], [], []
        for gi, g in enumerate(self.groups):
            pts = np.concatenate([A.P[k] for k in g])
            c = pts.mean(0)
            r0 = float(np.linalg.norm(pts[:, :2] - c[:2], axis=1).max())
            self.cg.append(c)
            self.r0.append(min(r0, 120.0))
            self.S.append(float(min(max(1.7 * 700.0 / max(r0, 40.0), 4.0), 40.0)))
            self.delay.append(0.22 * (gi / max(G - 1, 1)) + float(rng.uniform(0.0, 0.03)))
        # slow glitch events (seeded, frame-held): [(t0, t1, [(y0, y1, dx), ...])]
        T = self.dur
        n_ev = int(round(float(P.get("glitch", 1.0)) * max(1.0, T / 1.1)))
        self.events = []
        if n_ev > 0:
            starts = np.sort(rng.uniform(0.14 * T, 0.78 * T, n_ev))
            last = -1.0
            for t0 in starts:
                t0 = max(float(t0), last + 0.3)
                hold = float(rng.uniform(0.17, 0.3))
                bands = []
                for _ in range(int(rng.integers(1, 4))):
                    y0 = float(rng.uniform(-20, 520))
                    bands.append((y0, y0 + float(rng.uniform(16, 64)), float(rng.choice([-1, 1]) * rng.uniform(22, 70))))
                self.events.append((t0, t0 + hold, bands))
                last = t0 + hold
        self.lb = max(320.0, 1.2 * float(np.linalg.norm(B.P[:, 0] - B.P[:, -1], axis=1).mean()))

    # ------------------------------------------------------------ geometry
    def _scaled(self, k, gi, s):
        c = self.cg[gi]
        Q = self.dense[k].copy()
        Q[:, :2] = c[:2] + (Q[:, :2] - c[:2]) * s
        return Q

    def _ring(self, Q, hw):
        """closed outline (capsule, round caps) of a stroke of half width hw along polyline Q (xy normals)."""
        d = np.gradient(Q[:, :2], axis=0)
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(dn > 1e-9, d / np.maximum(dn, 1e-9), np.array([1.0, 0.0]))
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        L, R = Q.copy(), Q.copy()
        L[:, :2] += n * hw
        R[:, :2] -= n * hw
        ph = np.linspace(0.0, math.pi, 9)[1:-1]

        def cap(i, a0):
            f = a0 + ph
            C = np.repeat(Q[i][None, :], len(f), 0)
            C[:, :2] += hw * (np.cos(f)[:, None] * n[i] + np.sin(f)[:, None] * d[i])
            return C
        return np.vstack([L, cap(-1, 0.0), R[::-1], cap(0, math.pi), L[:1]])

    def _hw(self, k, gi, s):
        """half width of the stroke outline at scale s: the stroke thickens with the letter so the outlines round out to blobs."""
        A = self.a_exit
        return max(0.5 * A.w[k] * (1.0 + 0.6 * A.tube), 2.0) * s ** 0.9 + 0.22 * self.r0[gi] * (s - 1.0)

    def _bands(self, t):
        for t0, t1, bands in self.events:
            if t0 <= t < t1:
                return bands
        return []

    def _qs(self, u, gi, lag=0.0):
        return min(1.0, max(0.0, (u - self.delay[gi] - lag) / self.span))

    @staticmethod
    def _scale(q, S):
        return math.exp(math.log(S) * q ** 1.4)

    # ------------------------------------------------------------ frame
    def layers(self, t):
        A, B, T = self.a_exit, self.b_entry, self.dur
        u = min(1.0, max(0.0, t / max(T, 1e-9)))
        if u >= 1.0 - 1e-9:
            return self._b_layers(1.0)
        if all(self._qs(u, gi) <= 0.0 for gi in range(len(self.groups))) and u <= 1e-9:
            return strand_layers([A.P[k] for k in range(A.N)], list(A.col), list(A.w), A.tube,
                                 A.rim if A.tube > 0.01 else None, hi_off=np.array([-0.8, -0.8, 0.0]))
        bands = self._bands(t)
        uq = math.floor(t * FPS_GLITCH + 1e-9) / FPS_GLITCH / max(T, 1e-9)       # held time for the fill fade
        out = []
        out_lines = {}            # (colour, width, alpha) -> polylines (outlines + echoes), batched
        for gi, g in enumerate(self.groups):
            S = self.S[gi]
            q = self._qs(u, gi)
            if q >= 1.0:
                continue
            s = self._scale(q, S)
            fq = _ss(self._qs(uq, gi) / 0.45)
            leave = 1.0 - _ss((q - 0.72) / 0.28)
            polys, cols, ws, rims = [], [], [], []
            for k in g:
                Q = self._scaled(k, gi, s)
                polys.append(Q)
                cols.append(A.col[k]); ws.append(A.w[k] * (1.0 - 0.5 * fq)); rims.append(A.rim[k])
                if fq > 0.0 or q > 0.0:                 # the stroke outline emerges as the fill fades
                    hw = self._hw(k, gi, s)
                    ring = Q if self.params.get("unravel") else self._ring(Q, hw)      # unravel (glyph-outline hand-off): the flat outline strand itself, never a capsule / tube round it
                    key = (tuple(int(round(x)) for x in A.col[k]), round(1.3 * (1.0 - 0.55 * q), 2), round(fq * leave, 2))
                    out_lines.setdefault(key, []).append(ring)
                for j in range(1, self.rings + 1):       # puddle rings spawn at the centre, trail the letter
                    qj = self._qs(u, gi, lag=0.06 * j)
                    if qj <= 0.0 or qj >= 1.0:
                        continue
                    sj = self._scale(qj, S)
                    hwj = self._hw(k, gi, sj)
                    Qj = self._scaled(k, gi, sj)
                    fade = (0.78 ** j) * _ss(qj / 0.12) * (1.0 - _ss((qj - 0.72) / 0.28))
                    key = (tuple(int(round(x)) for x in A.col[k]), round(max(0.4, 1.3 * (1.0 - 0.18 * j) * (1.0 - 0.6 * qj)), 2),
                           round(fade, 2))
                    out_lines.setdefault(key, []).append(Qj if self.params.get("unravel") else self._ring(Qj, hwj))
            if fq < 1.0:
                ls = strand_layers(polys, cols, ws, A.tube, np.array(rims) if A.tube > 0.01 else None,
                                   hi_off=np.array([-0.8, -0.8, 0.0]))
                for L in ls:
                    L.alpha = max(0.0, (1.0 - fq) * leave)
                out += ls
        for (col, w, al), rings in out_lines.items():
            if al <= 0.004:
                continue
            pl = []
            for r in rings:
                pl += _slice(r, bands) if bands else [r]
            if pl:
                out.append(Layer(pl, col, "line", w, closed=False, alpha=min(1.0, al), bias=0.1))
        # B slides in over the tail of the rings
        if u > 0.45:
            out += self._b_layers(u)
        return out

    def _b_layers(self, u):
        B = self.b_entry
        if u >= 1.0 - 1e-9:
            return strand_layers([B.P[k] for k in range(B.N)], list(B.col), list(B.w), B.tube,
                                 B.rim if B.tube > 0.01 else None, hi_off=np.array([-0.8, -0.8, 0.0]))
        x = _ss((u - 0.45) / 0.55)
        e = 1.0 - (1.0 - x) ** 2
        al = _ss((u - 0.45) / 0.3)
        polys = [B.P[k] - B.D[k][None, :] * self.lb * (1.0 - e) for k in range(B.N)]
        ls = strand_layers(polys, list(B.col), list(B.w), B.tube, B.rim if B.tube > 0.01 else None,
                           hi_off=np.array([-0.8, -0.8, 0.0]))
        for L in ls:
            L.alpha = al
        return ls

    def focus(self, t):
        u = min(max(t / max(self.dur, 1e-9), 0.0), 1.0)
        return (1 - u) * self.a_exit.centroid() + u * self.b_entry.centroid()

    def bounds_points(self):
        return np.concatenate([self.a_exit.P.reshape(-1, 3), self.b_entry.P.reshape(-1, 3)])
