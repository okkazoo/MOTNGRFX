"""mograph.modules.material: the typed hand-off between modules (phase 3a). A `Material` generalises the strand `Bundle`:

  kind 'strands'  a Bundle (the phase-1/2 currency; what every transition draws)
  kind 'points'   N marks: P (N,3) positions, D (N,3) travel direction, col, size (mark / line width px), v (N,) speed,
                  mark (N,) 0 dot | 1 dash '-' | 2 bar '|' | 3 plus '+', sid / idx = strand id and position along it when the
                  points were sampled from strands (so points -> strands can trace them back exactly), length (N,) streak
                  length of a travelling mark
  kind 'blobs'    N soft discs: P centres, R radii, D, col, v
  kind 'none'     nothing is carried (flicker / wipe / cut styles): only `anchor` (the gate point on that side) and `dir`

Converters (`Material.to(kind, like=None)`):
  strands -> points   sample every strand (arc length), keeps sid / idx          points -> strands  trace by sid / idx (exact
                                                                                    round trip), else one short dash strand per point
  points  -> blobs    cluster (k-means), radius from the spread                    blobs -> points    scatter inside every disc
  blobs   -> strands  one capsule strand per blob (head at the centre, width 2R)
  any     -> none     keeps the gate point and direction                          none -> any        spawn on that side: `like`
                                                                                    (a Bundle / Material) is moved to the anchor
`as_bundle(like)` is what a transition consumes (at most MAX_STRANDS strands). `none` has no strands: with `like` the like-bundle
is spawned at the anchor; without it ValueError (Sequence replaces such a hand-off by a `cut` transition instead).
"""
import numpy as np
from mograph.modules.core import Bundle, unit, _rot_between

KINDS = ("strands", "points", "blobs", "none")
MARK_DOT, MARK_DASH, MARK_BAR, MARK_PLUS = 0, 1, 2, 3
MAX_STRANDS = 140
CAPSULE_K = 4


def _stride_idx(n, m):
    if n <= m:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, m)).astype(int))


def _f(a, n, d=0.0):
    return np.broadcast_to(np.asarray(d if a is None else a, float), (n,)).copy()


class Material:
    def __init__(self, kind, bundle=None, P=None, D=None, col=None, size=None, R=None, v=None, mark=None, sid=None, idx=None,
                 length=None, anchor=None, dir=None, t=0.0, src=None):
        assert kind in KINDS, kind
        self.kind, self.t = kind, float(t)
        self.bundle, self._src = bundle, src
        self.P = self.D = self.col = self.size = self.R = self.v = self.mark = self.sid = self.idx = self.length = None
        if kind in ("points", "blobs"):
            self.P = np.asarray(P, float).reshape(-1, 3)
            n = len(self.P)
            self.D = unit(np.broadcast_to(np.asarray(D if D is not None else (1.0, 0, 0), float), (n, 3)).copy())
            self.col = np.broadcast_to(np.asarray(col if col is not None else (236, 230, 218), float), (n, 3)).copy()
            self.v = _f(v, n)
            if kind == "points":
                self.size = _f(size, n, 1.5)
                self.mark = np.broadcast_to(np.asarray(0 if mark is None else mark, int), (n,)).copy()
                self.sid = np.full(n, -1, int) if sid is None else np.asarray(sid, int)
                self.idx = np.zeros(n, int) if idx is None else np.asarray(idx, int)
                self.length = _f(length, n, 0.0)
                self.length = np.where(self.length > 0, self.length, np.clip(self.size * 6.0, 12.0, 30.0))
            else:
                self.R = _f(R, n, 4.0)
                self.length = _f(length, n, 0.0)
                self.length = np.where(self.length > 0, self.length, 2.6 * self.R + 6.0)
        if kind == "strands":
            self.anchor = bundle.heads().mean(0)
            self.dir = unit(bundle.D.mean(0))
        elif kind == "none":
            self.anchor = np.asarray(anchor, float)
            self.dir = unit(np.asarray(dir if dir is not None else (1.0, 0, 0), float))
        else:
            self.anchor = np.asarray(anchor, float) if anchor is not None else self.P.mean(0)
            self.dir = unit(np.asarray(dir, float)) if dir is not None else unit(self.D.mean(0))

    # ------------------------------------------------------------ constructors
    @staticmethod
    def from_bundle(b):
        return Material("strands", bundle=b, t=b.t)

    @staticmethod
    def points(P, D, col=(236, 230, 218), size=1.5, v=0.0, mark=None, sid=None, idx=None, length=None, t=0.0, anchor=None, dir=None):
        return Material("points", P=P, D=D, col=col, size=size, v=v, mark=mark, sid=sid, idx=idx, length=length, t=t,
                        anchor=anchor, dir=dir)

    @staticmethod
    def blobs(P, R, D, col=(236, 230, 218), v=0.0, length=None, t=0.0, anchor=None, dir=None):
        return Material("blobs", P=P, R=R, D=D, col=col, v=v, length=length, t=t, anchor=anchor, dir=dir)

    @staticmethod
    def none(anchor, dir=(1.0, 0, 0), t=0.0):
        return Material("none", anchor=anchor, dir=dir, t=t)

    # ------------------------------------------------------------ queries
    @property
    def N(self):
        if self.kind == "strands":
            return self.bundle.N
        return 0 if self.kind == "none" else len(self.P)

    def heads(self):
        return self.bundle.heads() if self.kind == "strands" else (self.P if self.kind != "none" else self.anchor[None])

    def centroid(self):
        return self.heads().mean(0)

    def mean_v(self):
        if self.kind == "strands":
            return float(self.bundle.v.mean())
        return float(self.v.mean()) if self.kind != "none" else 0.0

    def __repr__(self):
        return f"Material({self.kind}, N={self.N}, anchor={np.round(self.anchor, 0).tolist()})"

    # ------------------------------------------------------------ converters
    def to(self, kind, like=None):
        if kind == self.kind:
            return self
        if kind == "none":
            return Material.none(self.anchor, self.dir, self.t)
        if self.kind == "none":
            return self._spawn(like).to(kind)
        if self.kind == "strands":
            return self._strands_to_points() if kind == "points" else self._strands_to_points()._points_to_blobs()
        if self.kind == "points":
            return self._points_to_strands() if kind == "strands" else self._points_to_blobs()
        return self._blobs_to_strands() if kind == "strands" else self._blobs_to_points()

    def as_bundle(self, like=None, max_n=MAX_STRANDS):
        """strands for a transition (<= max_n of them)."""
        m = self.to("strands", like)
        b = m.bundle
        if b.N > max_n:
            b = b.subset(b.order()[_stride_idx(b.N, max_n)])
        return b

    # -- strands <-> points
    def _strands_to_points(self, per=6):
        b = self.bundle
        S = b.resample_points(per)
        P = S.P.reshape(-1, 3)
        T = np.zeros_like(S.P)
        T[:, :-1] = S.P[:, :-1] - S.P[:, 1:]                 # head first: travel direction = toward the head
        T[:, -1] = T[:, -2]
        T = unit(T.reshape(-1, 3))
        n = b.N
        m = Material.points(P, T, np.repeat(b.col, per, 0), np.repeat(b.w, per), np.repeat(b.v, per), MARK_DOT,
                            sid=np.repeat(np.arange(n), per), idx=np.tile(np.arange(per), n), t=b.t)
        m._src = (b, P.copy())
        return m

    def _points_to_strands(self):
        if self._src is not None and self._src[1].shape == self.P.shape and np.allclose(self._src[1], self.P):
            return Material.from_bundle(self._src[0])
        ok = self.sid >= 0
        if ok.sum() >= 4 and len(np.unique(self.sid[ok])) >= 2:
            ids = np.unique(self.sid[ok])
            K = 6
            Ps, Ds, cols, ws, vs = [], [], [], [], []
            for s in ids:
                sel = np.where(self.sid == s)[0]
                sel = sel[np.argsort(self.idx[sel])]
                q = self.P[sel]
                if len(q) < 2:
                    continue
                cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
                u = np.linspace(0, cum[-1], K)
                Ps.append(np.stack([np.interp(u, cum, q[:, k]) for k in range(3)], 1))
                Ds.append(self.D[sel[0]]); cols.append(self.col[sel[0]]); ws.append(self.size[sel].mean()); vs.append(self.v[sel].mean())
            if len(Ps) >= 2:
                return Material.from_bundle(Bundle(np.array(Ps), np.array(Ds), np.array(cols), np.array(ws), self.t, np.array(vs), 0.0))
        return self._dash_strands()

    def _dash_strands(self):
        n = self.N
        L = self.length[:, None, None]
        s = np.linspace(0, 1, CAPSULE_K)[None, :, None]
        P = self.P[:, None, :] - self.D[:, None, :] * (L * s)
        w = np.maximum(self.size, 1.2) if self.kind == "points" else 2.0 * self.R
        return Material.from_bundle(Bundle(P, self.D, self.col, w, self.t, self.v, 0.0))

    # -- points <-> blobs
    def _points_to_blobs(self, k=None):
        from scipy.cluster.vq import kmeans2
        n = self.N
        k = int(k or np.clip(n // 4, 1, 40))
        if k >= n:
            return Material.blobs(self.P, np.maximum(self.size * 1.5, 2.0), self.D, self.col, self.v, t=self.t)
        _, lab = kmeans2(self.P, k, minit="++", seed=3)
        C, R, D, col, v = [], [], [], [], []
        for j in range(k):
            sel = np.where(lab == j)[0]
            if len(sel) == 0:
                continue
            c = self.P[sel].mean(0)
            C.append(c)
            R.append(max(1.3 * float(np.sqrt(((self.P[sel] - c) ** 2).sum(1).mean())) + float(self.size[sel].mean()), 2.0))
            D.append(self.D[sel].mean(0)); col.append(self.col[sel].mean(0)); v.append(self.v[sel].mean())
        return Material.blobs(np.array(C), np.array(R), np.array(D), np.array(col), np.array(v), t=self.t)

    def _blobs_to_points(self, plane=((1.0, 0, 0), (0, 1.0, 0))):
        e1, e2 = np.asarray(plane[0], float), np.asarray(plane[1], float)
        P, D, col, size, v, sid = [], [], [], [], [], []
        for i in range(self.N):
            r = self.R[i]
            n = int(np.clip(round(np.pi * r * r / max(r * 0.7, 4.0) ** 2), 1, 14))
            k = np.arange(n) + 0.5
            rad = r * np.sqrt(k / n) * (0.0 if n == 1 else 1.0)
            ang = k * 2.399963
            P.append(self.P[i] + rad[:, None] * np.cos(ang)[:, None] * e1 + rad[:, None] * np.sin(ang)[:, None] * e2)
            D.append(np.tile(self.D[i], (n, 1))); col.append(np.tile(self.col[i], (n, 1)))
            size.append(np.full(n, max(0.35 * r, 1.5))); v.append(np.full(n, self.v[i])); sid.append(np.full(n, -1))
        return Material.points(np.concatenate(P), np.concatenate(D), np.concatenate(col), np.concatenate(size),
                               np.concatenate(v), MARK_DOT, t=self.t)

    def _blobs_to_strands(self):
        return self._dash_strands()

    # -- none -> any
    def _spawn(self, like):
        """`like` (Bundle or Material) moved to this gate: heads centred on the anchor, travel direction turned to `dir`."""
        if like is None:
            raise ValueError("a 'none' material has nothing to carry; pass like=<Bundle> to spawn strands at its anchor")
        b = like.to("strands").bundle if isinstance(like, Material) else like
        R = _rot_between(unit(b.D.mean(0)), self.dir)
        c = b.heads().mean(0)
        P = (b.P - c) @ R.T + self.anchor
        return Material.from_bundle(Bundle(P, b.D @ R.T, b.col, b.w, self.t, b.v, b.tube, b.rim))


def ensure_bundle(x, like=None):
    """Bundle | Material -> Bundle (used by core.adapt); `like` = the other side, used to spawn strands for a 'none' material."""
    if isinstance(x, Bundle):
        return x
    if like is not None and not isinstance(like, Bundle):
        like = like.as_bundle() if like.kind != "none" else None
    return x.as_bundle(like)
