"""Helpers shared by the batch-C word modules (voxel_roll, voxel_orbit, particles, circuit, spectrum, tangent_spray,
outflow, fly_into). Kept out of _strand.py / core.py (owned by other batches); nothing here changes their behaviour."""
import math
import numpy as np
import cv2
from mograph.modules.core import Layer, unit, mixc, ol, Route3


def ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ssa(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a))) if b != a else float(t >= b)


def hash01(i, k=0):
    """deterministic hash in [0, 1)"""
    return ((int(i) * 2654435761 + int(k) * 40503 + 977) % 4294967296) / 4294967296.0


# ------------------------------------------------------------------ profile: arc length vs time, piecewise monotone Hermite
class Chain:
    """Per-element arc-length profile, vectorised over N elements. T, S, V: (N, K) knot times, arc positions, speeds (>= 0).
    Between knots: cubic Hermite with the end slopes clamped to 3x the secant (monotone, no overshoot). Before the first
    knot the element coasts backwards at V[:, 0]; after the last knot it keeps moving at V[:, -1] when coast, else stays."""

    def __init__(self, T, S, V, coast=True):
        self.T, self.S, self.V = (np.asarray(a, float) for a in (T, S, V))
        self.coast = coast
        self.N, self.K = self.T.shape

    def pos(self, t):
        T, S, V, K = self.T, self.S, self.V, self.K
        t = np.broadcast_to(np.asarray(t, float), (self.N,))
        i = np.clip((t[:, None] >= T[:, 1:]).sum(1), 0, K - 2)
        r = np.arange(self.N)
        t0, t1 = T[r, i], T[r, i + 1]
        s0, s1 = S[r, i], S[r, i + 1]
        v0, v1 = V[r, i], V[r, i + 1]
        dt = np.maximum(t1 - t0, 1e-9)
        ds = s1 - s0
        cap = 3.0 * np.abs(ds) / dt
        m0 = np.where(ds >= 0, np.minimum(v0, cap), -np.minimum(np.abs(v0), cap)) * dt
        m1 = np.where(ds >= 0, np.minimum(v1, cap), -np.minimum(np.abs(v1), cap)) * dt
        u = np.clip((t - t0) / dt, 0.0, 1.0)
        u2, u3 = u * u, u * u * u
        p = (2 * u3 - 3 * u2 + 1) * s0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * s1 + (u3 - u2) * m1
        before = t < T[:, 0]
        after = t > T[:, -1]
        p = np.where(before, S[:, 0] + V[:, 0] * (t - T[:, 0]), p)
        p = np.where(after, S[:, -1] + (V[:, -1] * (t - T[:, -1]) if self.coast else 0.0), p)
        return p

    def vel(self, t, e=2e-3):
        return (self.pos(t + e) - self.pos(t - e)) / (2 * e)


# ------------------------------------------------------------------ rotations
def rot(axis, a):
    axis = unit(axis)
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


# ------------------------------------------------------------------ lattice (voxel words)
def lattice_cells(lay, rows, thr=0.4, ppc=8):
    """voxel cells of a laid-out word. lay = core.layout_word dict (local coords, x right, y DOWN, centred). rows = cap-height
    in cells. -> (cells (M,2) int (ix, iy; iy grows downward), c = cell size (local px), ox, oy = local coords of cell (0,0)'s
    top-left corner, nx, ny)."""
    cap = lay["cap"]
    c = cap / float(rows)
    wd, ht = lay["size"]
    nx, ny = int(math.ceil(wd / c)) + 1, int(math.ceil(ht / c)) + 1
    ox, oy = -0.5 * wd - 0.5 * (nx * c - wd), -0.5 * ht - 0.5 * (ny * c - ht)
    mask = np.zeros((ny * ppc, nx * ppc), np.uint8)
    for l in lay["letters"]:
        m = np.zeros_like(mask)
        polys = [np.round(((q - [ox, oy]) / c * ppc) * 16).astype(np.int32) for q in l["cons"] if len(q) > 2]
        if polys:
            cv2.fillPoly(m, polys, 255, cv2.LINE_8, 4)
            mask |= m
    cov = mask.reshape(ny, ppc, nx, ppc).mean((1, 3)) / 255.0
    iy, ix = np.nonzero(cov >= thr)
    return np.stack([ix, iy], 1).astype(int), c, ox, oy, nx, ny


# ------------------------------------------------------------------ cubes
def shade(base, lam, ink=(12, 12, 11)):
    """3-stop ramp [base darkened 62%, base darkened 25%, base] at 0.35 + 0.65 * lam (the c2 / c3 cube shading)."""
    u = 0.35 + 0.65 * lam
    d1, d2 = mixc(base, ink, 0.62), mixc(base, ink, 0.25)
    if u < 0.5:
        return mixc(d1, d2, u / 0.5)
    return mixc(d2, tuple(base), (u - 0.5) / 0.5)


CUBE_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * 0.5
# faces: (vertex ids, local axis index, sign)
CUBE_F = [((0, 1, 2, 3), 2, -1), ((4, 5, 6, 7), 2, 1), ((0, 1, 5, 4), 1, -1), ((3, 2, 6, 7), 1, 1),
          ((0, 3, 7, 4), 0, -1), ((1, 2, 6, 5), 0, 1)]


def cube_layers(centers, Rs, sizes, face_cols, light, cull_dir=None, bias=0.0, edge_col=None, ink=(12, 12, 11)):
    """Layers (one per visible face, own depth-sorted item) for cubes. centers (M,3), Rs (M,3,3) local->world, sizes (M,),
    face_cols (M,3,3) base colour per LOCAL axis (index 0..2; opposite faces share it), light = unit vector toward the light.
    cull_dir: unit vector toward the nominal viewer; faces turned away from it are skipped (None: draw all six)."""
    out = []
    for c, R, sz, fc in zip(centers, Rs, sizes, face_cols):
        if sz < 0.03:
            continue
        V = c + (CUBE_V * sz) @ R.T
        for ids, ax, sg in CUBE_F:
            n = R[:, ax] * sg
            if cull_dir is not None and n @ cull_dir < -0.02:
                continue
            col = shade(fc[ax], max(0.0, float(n @ light)), ink)
            out.append(Layer([V[list(ids)]], col, "fill", bias=bias))
            if edge_col is not None:
                out.append(Layer([V[list(ids)]], col, "line", 0.5, closed=True, bias=bias + 0.05))
    return out
