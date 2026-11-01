"""voxel_roll: the word is a field of cubes lying on a grid floor. Each cube is carried in by a hairline strand that arrives
along the entry lanes and condenses into a cube a few cells short of its slot; the cubes then roll edge over edge (a quarter
turn per cell, each roll flips the colour seen from above) into the letters. To leave, every cube rolls out along the exit
axis, shrinks away and its strand runs off down the exit lanes. (mechanism: 11_mathtype c2R voxel roll +
13_camera/v001 tilt_to_plan cube rolls, rewritten for the Bundle interface)

The cubes stand on a floor that coincides with the placement plane, leaned back by `lean` degrees (0 = flat wall) so the
cubes read as cubes from a front camera. Rolling is along the lattice axis closest to the entry / exit direction; any 3D
direction works (the strands fly straight along it; a direction along the plane normal drops the cubes in)."""
import math
import numpy as np
from mograph.modules.core import (WordModule, Bundle, Route3, Layer, register, layout_word, unit, strand_layers, mixc)
from mograph.modules.words._cb import Chain, rot, lattice_cells, cube_layers, ss, ssa, hash01


@register("voxel_roll")
class VoxelRoll(WordModule):
    description = ("Hairline strands arrive along the entry lanes and each condenses into a cube just short of its slot; the "
                   "cubes then roll edge over edge across a grid floor into the letters. To leave, every cube rolls out, "
                   "shrinks to a point and its strand runs away down the exit lanes.")
    DEMO_WORDS = "ROLL"
    DEFAULT_FONT = "silkscreen"
    ROWS = 6                    # cap height in cells
    LEAN = 32.0                 # degrees the cube floor leans back from the placement plane

    # ------------------------------------------------------------------ build
    def build(self):
        P, pl, pal = self.params, self.placement, self.palette
        lean = math.radians(float(P.get("lean", self.LEAN)))
        self.U = pl.u
        self.V = unit(pl.v * math.cos(lean) - pl.n * math.sin(lean))
        self.UP = -unit(np.cross(self.U, self.V))                 # toward the viewer: cubes stand up this way
        rows = int(P.get("rows", self.ROWS))
        lead = max(1, round(0.22 * rows)) / rows
        fitw, fith = pl.w * 0.94, pl.h * 0.9 / max(math.cos(lean), 0.55)
        self.lay = layout_word(self.font, self.lines, fitw, fith, track=0.0, leading=lead)
        cells, c, ox, oy, nx, ny = lattice_cells(self.lay, rows)
        self.cells, self.c, self.ox, self.oy = cells, c, ox, oy
        self.N = M = len(cells)
        loc = np.stack([ox + (cells[:, 0] + 0.5) * c, oy + (cells[:, 1] + 0.5) * c], 1)
        self.floor = pl.origin + loc[:, 0:1] * self.U + loc[:, 1:2] * self.V                # (M,3) slot centres on the floor
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.a_in = self._axis(self.d_in)
        self.a_out = self._axis(self.d_out)
        self.vin = self.a_in[0] * self.U + self.a_in[1] * self.V
        self.vout = self.a_out[0] * self.U + self.a_out[1] * self.V
        self.Rb = np.stack([self.U, self.V, self.UP], 1)                                       # cube local frame at rest
        self.light = unit(0.85 * self.UP - 0.35 * self.U - 0.3 * self.V)
        self.ext_in = pl.extent_along(self.d_in)
        self.ext_out = pl.extent_along(self.d_out)
        self.gap = 0.06 * max(pl.w, pl.h) + 12.0
        self.coast_len = 2.5 * max(pl.w, pl.h)
        self.maxLb = float(P.get("tail_len") or 0.2 * max(pl.w, pl.h))
        self._build_entry()
        self._build_exit()
        self._colours()
        self._bundles()

    def _axis(self, d):
        a = np.array([d @ self.U, d @ self.V])
        if np.hypot(*a) < 0.3:
            return np.array([1, 0])
        if abs(a[0]) >= abs(a[1]):
            return np.array([int(np.sign(a[0])), 0])
        return np.array([0, int(np.sign(a[1]))])

    def _rowinfo(self, a):
        """row id (perpendicular lattice coordinate), rank-fraction of the row, and along-axis fraction per cell."""
        b = np.array([-a[1], a[0]])
        rid = self.cells @ b
        u, inv = np.unique(rid, return_inverse=True)
        rf = inv / max(len(u) - 1.0, 1.0)
        al = self.cells @ a
        af = (al - al.min()) / max(al.max() - al.min(), 1)
        return rid, rf, af

    def _build_entry(self):
        tm, c, M = self.timing, self.c, self.N
        pl = self.placement
        k0 = int(self.params.get("run", 5))
        rid, rf, af = self._rowinfo(self.a_in)
        kk = k0 + (np.array([hash01(r, 7) for r in rid]) * 3).astype(int)              # cells rolled, equal inside a row
        self.k_in = kk
        T_row = tm.form * (0.26 + 0.30 * rf) + 0.05 * np.array([hash01(r, 3) for r in rid])
        self.T_in = T_row
        p0 = self.floor - kk[:, None] * c * self.vin                                       # stage centres (floor)
        Lf = (p0 - pl.origin) @ self.d_in + self.ext_in + self.gap
        Lf = np.maximum(Lf, 40.0)
        J = p0 - self.d_in * Lf[:, None]
        roll_t = np.maximum(tm.form - T_row, 0.25)
        vr = 2.0 * kk * c / roll_t
        V0 = np.clip(2.0 * Lf / np.maximum(T_row, 0.15) - vr, 0.3 * vr, 3200.0)
        Lb = np.clip(0.11 * V0, 14.0, self.maxLb)
        self.Lb_in, self.V_in, self.J_in = Lb, V0, J
        sA = Lb + Lf
        L = sA + kk * c
        T = np.stack([np.zeros(M), T_row, np.full(M, tm.form)], 1)
        S = np.stack([Lb, sA, L], 1)
        V = np.stack([V0, vr, np.zeros(M)], 1)
        self.ch_in = Chain(T, S, V, coast=False)
        self.sA_in, self.L_in = sA, L
        self.R_in = [Route3(np.vstack([J[i] - self.d_in * Lb[i], J[i], p0[i], self.floor[i]])) for i in range(M)]
        # the orientation at rest: k quarter turns about the roll axis
        ax = np.cross(self.UP, self.vin)
        self.R_hold = [rot(ax, kk[i] * math.pi / 2) @ self.Rb for i in range(M)]

    def _build_exit(self):
        tm, c, M = self.timing, self.c, self.N
        pl = self.placement
        k = int(self.params.get("run", 5)) + 1
        rid, rf, af = self._rowinfo(self.a_out)
        kk = np.full(M, k)
        self.k_out = kk
        ex = tm.exit
        # cubes nearest the exit side leave first (same row = lock step)
        t0 = ex * (0.04 + 0.30 * (1.0 - af)) + 0.04 * np.array([hash01(r, 5) for r in rid])
        self.T_out = t0
        p1 = self.floor + kk[:, None] * c * self.vout
        Lg = np.maximum((pl.origin - p1) @ self.d_out + self.ext_out + self.gap, 40.0)
        J2 = p1 + self.d_out * Lg[:, None]
        Troll = np.maximum(0.42 * (ex - t0), 0.2)
        vx = 2.0 * kk * c / Troll
        TB = np.maximum(ex - t0 - Troll, 0.15)
        Vo = np.clip(2.0 * Lg / TB - vx, 0.3 * vx, 3200.0)
        Lb = np.clip(0.11 * Vo, 14.0, self.maxLb)
        self.Lb_out, self.V_out, self.J_out = Lb, Vo, J2
        L2 = kk * c + Lg
        T = np.stack([t0, t0 + Troll, np.full(M, ex)], 1)
        S = np.stack([np.zeros(M), kk * c, L2], 1)
        V = np.stack([np.zeros(M), vx, Vo], 1)
        self.ch_out = Chain(T, S, V, coast=True)
        self.L2 = L2
        self.R_out = [Route3(np.vstack([self.floor[i], p1[i], J2[i], J2[i] + self.d_out * self.coast_len])) for i in range(M)]
        ax = np.cross(self.UP, self.vout)
        self.axo = ax

    def _colours(self):
        pal = self.palette
        bone, a1, a2 = pal.fg, pal.accents[2], pal.accents[0]
        self.fc = []
        for R in self.R_hold:
            top = int(np.argmax(np.abs(R.T @ self.UP)))
            cols = [None] * 3
            others = [i for i in range(3) if i != top]
            cols[top] = bone
            cols[others[0]], cols[others[1]] = a1, a2
            self.fc.append(cols)
        self.fc = np.array(self.fc, float)

    def _bundles(self):
        s = np.linspace(0, 1, 8)[None, :, None]
        M, tm = self.N, self.timing
        fg = np.tile(np.array(self.palette.fg, float), (M, 1))
        Pin = self.J_in[:, None, :] - self.d_in[None, None, :] * (self.Lb_in[:, None, None] * s)
        self._entry = Bundle(Pin, np.tile(self.d_in, (M, 1)), fg, 1.4, 0.0, self.V_in, 0.0)
        Pout = self.J_out[:, None, :] - self.d_out[None, None, :] * (self.Lb_out[:, None, None] * s)
        self._exit = Bundle(Pout, np.tile(self.d_out, (M, 1)), fg, 1.4, self.duration, self.V_out, 0.0)

    def entry_bundle(self):
        return self._entry

    def exit_bundle(self):
        return self._exit

    # ------------------------------------------------------------------ pose
    def _roll(self, c0, R0, d, u):
        """cube centre / rotation after u cells of rolling from rest centre c0 along unit d."""
        c, UP, cs = self.c, self.UP, self.c
        ax = np.cross(UP, d)
        full = math.floor(u)
        Rf = rot(ax, full * math.pi / 2)
        cc = c0 + d * (full * cs)
        f = ss(u - full)
        Rs = rot(ax, 0.5 * math.pi * f)
        piv = cc + 0.5 * d * cs - 0.5 * UP * cs
        return piv + Rs @ (cc - piv), Rs @ Rf @ R0

    def windows(self, t):
        """per strand polyline (3D, tail -> head) at time t (None when not drawn)."""
        tm = self.timing
        wins = []
        if t < tm.form + tm.hold:
            h = self.ch_in.pos(t)
            for i in range(self.N):
                R = self.R_in[i]
                lb = self.Lb_in[i]
                tail_k = 1.0 - ss((t - (tm.form - 0.3)) / 0.3)
                if tail_k <= 0.002 or t >= tm.form:
                    wins.append(None)
                    continue
                wins.append(R.window(max(h[i] - lb * tail_k, 0.0), h[i]))
        else:
            te = t - tm.form - tm.hold
            h = self.ch_out.pos(te)
            for i in range(self.N):
                R = self.R_out[i]
                if h[i] < 0.5:
                    wins.append(None)
                    continue
                wins.append(R.window(max(h[i] - self.Lb_out[i], 0.0), h[i]))
        return wins

    def cubes(self, t):
        tm, c = self.timing, self.c
        cen, Rs, sz = [], [], []
        up_c = self.UP * (0.5 * c)
        if t < tm.form + tm.hold:
            h = self.ch_in.pos(t)
            for i in range(self.N):
                u = (h[i] - self.sA_in[i]) / c
                if u < 0:
                    continue
                p0 = self.floor[i] - self.k_in[i] * c * self.vin + up_c
                if u >= self.k_in[i]:
                    cc, R = self.floor[i] + up_c, self.R_hold[i]
                else:
                    cc, R = self._roll(p0, self.Rb, self.vin, u)
                cen.append(cc); Rs.append(R); sz.append(0.94 * c * ss(u / 0.7))
                self._idx.append(i)
        else:
            te = t - tm.form - tm.hold
            h = self.ch_out.pos(te)
            for i in range(self.N):
                u = h[i] / c
                k = self.k_out[i]
                if u >= k + 0.8:
                    continue
                if u <= 0:
                    cc, R = self.floor[i] + up_c, self.R_hold[i]
                else:
                    cc, R = self._roll(self.floor[i] + up_c, self.R_hold[i], self.vout, min(u, k))
                cen.append(cc); Rs.append(R); sz.append(0.94 * c * (1.0 - ss((u - k) / 0.8)))
                self._idx.append(i)
        return np.array(cen).reshape(-1, 3), np.array(Rs).reshape(-1, 3, 3), np.array(sz)

    def grid_alpha(self, t):
        tm = self.timing
        a = ss(t / max(0.28 * tm.form, 1e-6))
        a *= 1.0 - ss((t - tm.form - tm.hold - 0.62 * tm.exit) / max(0.38 * tm.exit, 1e-6))
        return a

    def grid_layers(self, t):
        a = self.grid_alpha(t)
        if a <= 0.004 or not self.params.get("grid", True):
            return []
        pl, c = self.placement, self.c
        hl = 0.5 * pl.h / max(abs(self.V @ pl.v), 0.55) + c
        wl = 0.5 * pl.w + c
        xs = self.ox + c * np.arange(math.floor((-wl - self.ox) / c), math.ceil((wl - self.ox) / c) + 1)
        ys = self.oy + c * np.arange(math.floor((-hl - self.oy) / c), math.ceil((hl - self.oy) / c) + 1)
        o, U, V = pl.origin, self.U, self.V
        lines = [np.array([o + x * U + ys[0] * V, o + x * U + ys[-1] * V]) for x in xs]
        lines += [np.array([o + xs[0] * U + y * V, o + xs[-1] * U + y * V]) for y in ys]
        col = mixc(self.palette.bg, self.palette.fg, 0.2)
        return [Layer(lines, col, "line", 0.8, closed=False, alpha=a, bias=-12000.0)]

    def layers(self, t):
        t = max(t, 0.0)
        self._idx = []
        cen, Rs, sz = self.cubes(t)
        fcs = self.fc[self._idx] if len(self._idx) else np.zeros((0, 3, 3))
        out = self.grid_layers(t)
        wins = self.windows(t)
        fg = np.array(self.palette.fg, float)
        out += strand_layers(wins, [fg] * len(wins), [1.4] * len(wins), 0.0, bias0=-6000.0)
        cull = None if not self.params.get("cull", False) else -self.UP * -1
        out += cube_layers(cen, Rs, sz, fcs, self.light, ink=self.palette.bg)
        return out

    def focus(self, t):
        t = min(max(t, 0.0), self.duration)
        tm = self.timing
        if t < tm.form:
            h = self.ch_in.pos(t)
            pts = [self.R_in[i].at(h[i]) for i in range(self.N) if h[i] > self.sA_in[i] - 40.0]
        elif t < tm.form + tm.hold:
            return self.placement.origin.copy()
        else:
            h = self.ch_out.pos(t - tm.form - tm.hold)
            pts = [self.R_out[i].at(h[i]) for i in range(self.N)]
        return np.mean(pts, 0) if pts else self.placement.origin.copy()

    def hold_pose(self, dist=900.0):
        return self.placement.origin + self.UP * dist, self.placement.origin.copy()


# ---------------------------------------------------------------------------------------------------------------- legibility (render R17)
# R17: FINAL / ECHO stacked in Bungee Inline at wf 0.9 came out as ONE merged black block: at ROWS = 6 cells per cap height the 1-2 % gaps between a fat face's letters
# (and the gap between two stacked lines) fall inside a single cell, so neighbouring letters share cube edges. `cells_legible` measures that on the cell grid (no placement, no render),
# so a planner can swap the module for a plain readable one (tools/fusion/owner.fix_illegible_modules) before anything is drawn.
PIECES_MIN = 0.75         # the lattice must fall into at least this many x (number of letters) separate pieces (8-connected cubes): merged letters / lines make fewer
MIN_LETTER_CELLS = 4      # every letter needs at least this many cubes, and
MIN_MEAN_CELLS = 8.0      # the word this many per letter on average (a hairline face voxelises to nothing: MoiraiOne -> 0-3 cubes for a whole word)


def cells_legible(font, lines, wf=0.9, hf=0.3, aspect=0.5625, rows=None, lean=None, roll=0.0, thr=0.4, ppc=8):
    """does the cube lattice of this word keep its letters apart? Same fit as VoxelRoll.build (box wf x hf of the frame, `aspect` = frame width / height, lean, rows): the cubes are
    counted per letter (the letter covering most of the cell owns it) and as 8-connected pieces. -> {"ok", "reason", "cells", "letters", "pieces", "min_cells"}. ok = every letter has
    >= MIN_LETTER_CELLS cubes, MIN_MEAN_CELLS on average, and the pieces number >= PIECES_MIN x the letters. Pure (no Placement, no ComfyUI)."""
    import cv2
    from mograph.modules.core import layout_word
    from mograph.modules.words._cb import lattice_cells
    rows = int(rows or VoxelRoll.ROWS)
    lean = math.radians(float(VoxelRoll.LEAN if lean is None else lean))
    lines = [str(x) for x in (lines if isinstance(lines, (list, tuple)) else [lines]) if str(x).strip()]
    rolled = abs(abs(float(roll or 0.0)) - 90.0) < 1.0
    H = 1000.0                                             # world units: pixels of a 1000 px tall frame (layout_word caches on round(w), round(h): small units would collide)
    W = H * max(float(aspect), 1e-3)
    pw, ph = float(wf) * (H if rolled else W), float(hf) * (W if rolled else H)
    lead = max(1, round(0.22 * rows)) / rows
    lay = layout_word(font, lines, pw * 0.94, ph * 0.9 / max(math.cos(lean), 0.55), track=0.0, leading=lead)
    cells, c, ox, oy, nx, ny = lattice_cells(lay, rows, thr=thr, ppc=ppc)
    covs = []
    for l in lay["letters"]:
        if not str(l.get("ch", "x")).isalnum():
            continue
        m = np.zeros((ny * ppc, nx * ppc), np.uint8)
        polys = [np.round(((q - [ox, oy]) / c * ppc) * 16).astype(np.int32) for q in l["cons"] if len(q) > 2]
        if polys:
            cv2.fillPoly(m, polys, 255, cv2.LINE_8, 4)
        covs.append(m.reshape(ny, ppc, nx, ppc).mean((1, 3)) / 255.0)
    n = len(covs)
    if n == 0:
        return {"ok": False, "reason": "no letters", "cells": int(len(cells)), "letters": 0, "pieces": 0, "min_cells": 0}
    G = np.zeros((ny, nx), np.uint8)                       # the real lattice (all letters together, as VoxelRoll.build gets it)
    G[cells[:, 1], cells[:, 0]] = 1
    owner = np.argmax(np.stack(covs), 0)                   # each cube belongs to the letter that covers most of its cell
    counts = [int(((owner == i) & (G > 0)).sum()) for i in range(n)]
    pieces = int(cv2.connectedComponents(G, connectivity=8)[0]) - 1
    out = {"ok": True, "reason": "", "cells": int(len(cells)), "letters": n, "pieces": pieces, "min_cells": min(counts)}
    if min(counts) < MIN_LETTER_CELLS or sum(counts) / n < MIN_MEAN_CELLS:
        return dict(out, ok=False, reason=f"the face voxelises to too few cubes ({sum(counts)} for {n} letters, thinnest letter {min(counts)})")
    if pieces < PIECES_MIN * n:
        return dict(out, ok=False, reason=f"the cubes fall into {pieces} piece(s) for {n} letters: neighbouring letters / lines merge into blocks at {rows} cells per cap height")
    return out
