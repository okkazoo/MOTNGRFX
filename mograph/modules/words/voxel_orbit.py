"""voxel_orbit: the word is a cloud of cubes with real depth that, seen from the front, reads the word exactly. Hairline
strands stream in along the entry lanes and each condenses into a cube on its place in the cloud while the whole cloud
swings round its vertical axis (an orbit that settles face-on) and every cube turns one eased full turn about its own
axis, so the cloud resolves into the word. To leave, the cloud swings away the other way, each cube spins, shrinks to a
point and its strand runs out down the exit lanes. (mechanism: 11_mathtype c3R voxel orbit: orbiting cube cloud + local
spin of each block, rewritten for the Bundle interface; the camera is separate, so the orbit is done by the cloud)

Cube depths are anamorphic: every cube sits on a ray through an eye `eye` px in front of the placement, so a camera at
that distance sees the word exactly whatever the depths. Params: rows (cap height in cells), depth (half-thickness of the
cloud, px), swing (degrees of orbit), eye, tail_len, speed."""
import math
import numpy as np
from mograph.modules.core import (WordModule, Bundle, Route3, Layer, register, layout_word, unit, strand_layers, mixc)
from mograph.modules.words._cb import Chain, rot, lattice_cells, cube_layers, ss, hash01


def _eo(u):
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 2


def _ei(u):
    u = min(1.0, max(0.0, u))
    return u * u


@register("voxel_orbit")
class VoxelOrbit(WordModule):
    description = ("Hairline strands stream in and each condenses into a cube while the whole cube cloud swings round and "
                   "every cube turns on its own axis; the cloud settles face-on into the word. To leave, the cloud swings "
                   "away, the cubes spin and shrink to points, and their strands run out down the exit lanes.")
    DEMO_WORDS = "ORBIT"
    DEFAULT_FONT = "silkscreen"
    ROWS = 7

    def build(self):
        P, pl, pal = self.params, self.placement, self.palette
        rows = int(P.get("rows", self.ROWS))
        lead = max(1, round(0.22 * rows)) / rows
        self.lay = layout_word(self.font, self.lines, pl.w * 0.94, pl.h * 0.9, track=0.0, leading=lead)
        cells, c, ox, oy, nx, ny = lattice_cells(self.lay, rows)
        self.cells, self.c = cells, c
        self.N = M = len(cells)
        rng = np.random.default_rng(int(P.get("seed", 21)))
        loc = np.stack([ox + (cells[:, 0] + 0.5) * c, oy + (cells[:, 1] + 0.5) * c], 1)
        D = float(P.get("eye", 1000.0))
        dep = float(P.get("depth", 0.14 * max(pl.w, pl.h)))
        # coherent depth: slow waves over the word + a share of random scatter (clamped to a safe range)
        sx = loc[:, 0] / max(pl.w, 1.0)
        sy = loc[:, 1] / max(pl.h, 1.0)
        dz = dep * (0.55 * np.sin(5.2 * sx + 1.3) * np.cos(3.1 * sy + 0.4) + 0.45 * rng.uniform(-1, 1, M))
        dz = np.clip(dz, -0.8 * D, 0.8 * D)
        o = pl.origin
        E0 = o - pl.n * D
        plane = o + loc[:, 0:1] * pl.u + loc[:, 1:2] * pl.v
        k = (D + dz) / D
        self.W = E0 + (plane - E0) * k[:, None]                       # rest positions (cloud frame, word facing the eye)
        self.sz = 0.92 * c * k                                         # cube edge so the word still tiles from the eye
        self.axis_u, self.axis_v = pl.u, pl.v
        ax = rng.normal(size=(M, 3))
        ax[np.abs(ax).max(1) < 0.2] = 1.0
        self.spin_ax = ax / np.linalg.norm(ax, axis=1)[:, None]
        self.spin_dir = rng.choice([-1.0, 1.0], M)
        self.jit = rng.random(M)
        self.swing = math.radians(float(P.get("swing", 80.0)))
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.Rb = np.stack([pl.u, pl.v, pl.n], 1)                       # cube local frame at rest (front face = -z)
        self.light = unit(-0.55 * pl.u - 0.55 * pl.v - 0.62 * pl.n)
        pal = self.palette
        fc = np.array([pal.accents[0], pal.accents[2], pal.fg], float)
        self.fc = np.tile(fc[None], (M, 1, 1))
        self.rmax = float(np.linalg.norm(self.W - o, axis=1).max()) + 0.5 * c
        self.gap = 0.06 * max(pl.w, pl.h) + 12.0
        self.coast_len = 2.5 * max(pl.w, pl.h)
        self.maxLb = float(P.get("tail_len") or 0.2 * max(pl.w, pl.h))
        self._build_entry()
        self._build_exit()
        self._bundles()

    # cloud pose
    def yaw(self, t):
        tm = self.timing
        if t < tm.form:
            return -self.swing * (1.0 - _eo(t / tm.form))
        if t < tm.form + tm.hold:
            return 0.0
        return self.swing * _ei((t - tm.form - tm.hold) / tm.exit)

    def pos_at(self, t, idx=None):
        R = rot(self.axis_v, self.yaw(t))
        W = self.W if idx is None else self.W[idx]
        return self.placement.origin + (W - self.placement.origin) @ R.T

    def _build_entry(self):
        tm, M, pl = self.timing, self.N, self.placement
        d = self.d_in
        q0 = self.pos_at(0.0)
        # spawn order: cubes on the entry side first; spawn position = where the orbiting cloud has them then
        al = (self.W - pl.origin) @ d
        fr = (al - al.min()) / max(al.max() - al.min(), 1.0)
        ts = tm.form * (0.14 + 0.50 * fr + 0.10 * self.jit)
        self.ts = ts
        q = np.array([self.pos_at(ts[i])[i] for i in range(M)])
        self.q_in = q
        Lf = np.maximum((q - pl.origin) @ d + self.rmax + self.gap, 40.0)
        J = q - d * Lf[:, None]
        V0 = 2.0 * Lf / np.maximum(ts, 0.12)
        sp = self.params.get("speed")
        Lb = np.clip(0.1 * V0, 14.0, self.maxLb)
        sA = Lb + Lf
        self.Lb_in, self.V_in, self.J_in, self.sA_in = Lb, V0, J, sA
        T = np.stack([np.zeros(M), ts], 1)
        S = np.stack([Lb, sA], 1)
        V = np.stack([V0, np.zeros(M)], 1)
        self.ch_in = Chain(T, S, V, coast=False)
        self.R_in = [Route3(np.vstack([J[i] - d * Lb[i], J[i], q[i]])) for i in range(M)]

    def _build_exit(self):
        tm, M, pl = self.timing, self.N, self.placement
        d = self.d_out
        ex = tm.exit
        al = (self.W - pl.origin) @ d
        fr = (al - al.min()) / max(al.max() - al.min(), 1.0)
        te = ex * (0.12 + 0.42 * (1.0 - fr) + 0.08 * self.jit)           # cubes nearest the exit side leave first
        self.te = te
        tg = tm.form + tm.hold
        q = np.array([self.pos_at(tg + te[i])[i] for i in range(M)])
        self.q_out = q
        Lg = np.maximum((pl.origin - q) @ d + self.rmax + self.gap, 40.0)
        J2 = q + d * Lg[:, None]
        Tb = np.maximum(ex - te - 0.1, 0.2)
        Vo = np.clip(2.0 * Lg / Tb, 200.0, 3200.0)
        Lb = np.clip(0.1 * Vo, 14.0, self.maxLb)
        self.Lb_out, self.V_out, self.J_out = Lb, Vo, J2
        T = np.stack([te, np.full(M, ex)], 1)
        S = np.stack([np.zeros(M), Lg], 1)
        V = np.stack([np.zeros(M), Vo], 1)
        self.ch_out = Chain(T, S, V, coast=True)
        self.R_out = [Route3(np.vstack([q[i], J2[i], J2[i] + d * self.coast_len])) for i in range(M)]

    def _bundles(self):
        s = np.linspace(0, 1, 8)[None, :, None]
        M = self.N
        fg = np.tile(np.array(self.palette.fg, float), (M, 1))
        Pin = self.J_in[:, None, :] - self.d_in[None, None, :] * (self.Lb_in[:, None, None] * s)
        self._entry = Bundle(Pin, np.tile(self.d_in, (M, 1)), fg, 1.4, 0.0, self.V_in, 0.0)
        Pout = self.J_out[:, None, :] - self.d_out[None, None, :] * (self.Lb_out[:, None, None] * s)
        self._exit = Bundle(Pout, np.tile(self.d_out, (M, 1)), fg, 1.4, self.duration, self.V_out, 0.0)

    def entry_bundle(self):
        return self._entry

    def exit_bundle(self):
        return self._exit

    # ------------------------------------------------------------------ frame
    def spin_sp(self, t):
        """eased spin progress per cube (0..1 over its own window); 0 / 1 are both the identity orientation."""
        tm = self.timing
        if t < tm.form:
            u = (t - self.ts) / np.maximum(tm.form - self.ts, 1e-3)
        elif t < tm.form + tm.hold:
            return np.zeros(self.N)
        else:
            u = (t - tm.form - tm.hold) / max(0.9 * tm.exit, 1e-3) * np.ones(self.N)
        u = np.clip(u, 0.0, 1.0)
        return u * u * (3 - 2 * u)

    def layers(self, t):
        t = max(t, 0.0)
        tm = self.timing
        M = self.N
        cen, Rs, sz, idx = [], [], [], []
        R_orb = rot(self.axis_v, self.yaw(t))
        o = self.placement.origin
        sp = self.spin_sp(t)
        if t < tm.form + tm.hold:
            h = self.ch_in.pos(t)
            alive = np.nonzero(t >= self.ts)[0]
            scl = np.array([ss((t - self.ts[i]) / 0.3) for i in alive])
            wins = []
            for i in range(M):
                if t >= self.ts[i] + 0.26 or t < 0:
                    wins.append(None)
                    continue
                tailk = 1.0 - ss((t - self.ts[i]) / 0.25)
                wins.append(self.R_in[i].window(max(h[i] - self.Lb_in[i] * tailk, 0.0), h[i]))
        else:
            te = t - tm.form - tm.hold
            h = self.ch_out.pos(te)
            alive = np.nonzero(te < self.te + 0.3)[0]
            scl = np.array([1.0 - ss((te - self.te[i]) / 0.3) for i in alive])
            wins = []
            for i in range(M):
                if te <= self.te[i] or h[i] < 0.5:
                    wins.append(None)
                    continue
                wins.append(self.R_out[i].window(max(h[i] - self.Lb_out[i], 0.0), h[i]))
        for j, i in enumerate(alive):
            cen.append(o + R_orb @ (self.W[i] - o))
            a = self.spin_dir[i] * 2 * math.pi * sp[i]
            R = R_orb @ (rot(self.spin_ax[i], a) if a != 0.0 else np.eye(3)) @ self.Rb
            Rs.append(R)
            sz.append(self.sz[i] * scl[j])
            idx.append(i)
        fg = np.array(self.palette.fg, float)
        out = strand_layers(wins, [fg] * M, [1.4] * M, 0.0)
        out += cube_layers(np.array(cen).reshape(-1, 3), np.array(Rs).reshape(-1, 3, 3), np.array(sz),
                           self.fc[idx] if idx else np.zeros((0, 3, 3)), self.light, ink=self.palette.bg)
        return out

    def focus(self, t):
        return self.placement.origin.copy()

    def hold_pose(self, dist=1000.0):
        pl = self.placement
        return pl.origin - pl.n * dist, pl.origin.copy()
