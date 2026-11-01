#!/usr/bin/env python
"""Math-type test A3c: A3b "as a volume" (18.2 s, 4:5). Sibling of follow_plus.py (A3b).

Same grid of marks (50% "+", 25% "|", 25% "-"), same patches, matching, ghosts, schedules, hover,
trail -> mark identity morph and timings as A3b (it subclasses its Scene). What changes:
- Every phrase is a plane in 3D world space with its own position and orientation: P2 sits to the
  right and 1200 deeper, turned 40 deg (yaw); P3 sits below and 1300 deeper again, tipped 30 deg
  (pitch). Local grid point (x, y) -> C_k + (x - 540) u_k + (y - 675) v_k.
- A perspective camera (focal = distance = 1600 design px, so a frontal phrase at the read distance
  looks exactly like A3b) dollies and turns between the three frontal poses: position lerp and
  yaw / pitch lerp, both power2 in-out, same timing as A3b's pan.
- Flights are cubic Hermite arcs in 3D: start tangent from a broad, static, divergence-free 3D flow
  mixed with the travel direction; end tangent a gentle approach mixed with the target plane's normal,
  so particles travel through depth. Trails are projected past world positions (parallax).
- Depth cues: brightness by (D/z)^0.6, fog beyond the read distance, near-plane fade; marks are the
  projected arm endpoints, so they scale with depth and foreshorten with the plane's orientation.
No credit. Numbered generations and lineage via gens.py (see follow_plus.cli).

    python tools/mathtype/follow_vol.py --draft --trigger "user: ..." --change "..."
    python tools/mathtype/follow_vol.py --final
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import follow_plus as FP  # noqa: E402
from follow_plus import (BG, INK, DW, DH, PEEL, CAM_LAG, CAM_DUR, TRAIL_M, TRAIL_DT, TRAIL_FADE,  # noqa: E402
                         LINE_PX, PLUS, PITCH, K, TAN0, TAN1, READABLE, GATHERED, DUR, flight_soft,
                         smooth, ease2)

FOCAL = DIST = 1600.0                     # design px; frontal plane at DIST maps 1:1
FOG = 3500.0                              # brightness falls by e every FOG px beyond DIST
NEAR = 250.0                              # near-plane fade starts here (camera-space z)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


YAW = [0.0, np.radians(40.0), np.radians(40.0)]
PITCH_ = [0.0, 0.0, np.radians(-30.0)]
ROT = [Ry(y) @ Rx(p) for y, p in zip(YAW, PITCH_)]
_C2 = np.zeros(3) + 1300.0 * ROT[0][:, 0] + 1200.0 * ROT[0][:, 2]
_C3 = _C2 + 1100.0 * ROT[1][:, 1] + 1300.0 * ROT[1][:, 2]
CENTRE = [np.zeros(3), _C2, _C3]


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class VolScene(FP.Scene):
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(FP.SEED + 7)
        N = self.N
        self.U = [R[:, 0] for R in ROT]
        self.V = [R[:, 1] for R in ROT]
        self.Nn = [R[:, 2] for R in ROT]
        self.w3 = [self.to_world(k, self.loc[k]) for k in range(3)]
        # P1 build: from the left and from deep behind the first plane
        q = self.loc[0] - [DW / 2, DH / 2]
        q[:, 1] *= 1.1
        self.src3 = (CENTRE[0] + np.outer(q[:, 0] - 1250 - 260 * rng.random(N), self.U[0])
                     + np.outer(q[:, 1] + 90 * (rng.random(N) - 0.5), self.V[0])
                     + np.outer(600 + 1200 * rng.random(N), self.Nn[0]))
        # broad divergence-free 3D flow: sum of two plane waves, amplitude vector perpendicular to k
        kd = _unit(rng.normal(size=(2, 3)))
        lam = np.array([2600.0, 3400.0])
        amp = _unit(np.cross(kd, _unit(rng.normal(size=(2, 3)))))
        self.flow3 = dict(k=kd * (2 * np.pi / lam)[:, None], a=amp, ph=rng.random(2) * 2 * np.pi)
        labp = self.labp
        jit = rng.random((3, K, 3)) * 2 - 1
        starts = [self.src3, self.w3[0], self.w3[1]]
        self.tan3 = []
        self.phi_in, self.phi_out = [], []
        for k in range(3):
            A_, B_ = starts[k], self.w3[k]
            L = np.linalg.norm(B_ - A_, axis=1, keepdims=True)
            travel = _unit(B_ - A_)
            jb = jit[k][labp[max(k - 1, 0)]]
            t0 = _unit(_unit(0.55 * travel + 0.45 * self.flow_dir3(A_)) + 0.25 * jb)
            t1 = _unit(0.5 * travel + 0.3 * self.Nn[k] + 0.2 * self.flow_dir3(B_) + 0.12 * jb[:, ::-1])
            T0, T1 = t0 * L * TAN0, t1 * L * TAN1
            self.tan3.append((T0, T1))
            # in-plane angles: approach seen on the target plane, departure seen on the source plane
            self.phi_in.append(np.arctan2(T1 @ self.V[k], T1 @ self.U[k]))
            ks = max(k - 1, 0)
            self.phi_out.append(np.arctan2(T0 @ self.V[ks], T0 @ self.U[ks]))

    # ---------------------------------------------------------------------------------- geometry
    def to_world(self, k, q):
        d = q - [DW / 2, DH / 2]
        return CENTRE[k] + np.outer(d[:, 0], self.U[k]) + np.outer(d[:, 1], self.V[k])

    def flow_dir3(self, q):
        w = self.flow3
        return _unit(np.cos(q @ w["k"].T + w["ph"]) @ w["a"])

    def cam_pose(self, t):
        e = [float(ease2((t - PEEL[m] - CAM_LAG) / CAM_DUR)) for m in range(2)]
        poses = [CENTRE[k] - DIST * self.Nn[k] for k in range(3)]
        P = poses[0] + (poses[1] - poses[0]) * e[0] + (poses[2] - poses[1]) * e[1]
        yaw = YAW[0] + (YAW[1] - YAW[0]) * e[0] + (YAW[2] - YAW[1]) * e[1]
        pit = PITCH_[0] + (PITCH_[1] - PITCH_[0]) * e[0] + (PITCH_[2] - PITCH_[1]) * e[1]
        return P, Ry(yaw) @ Rx(pit)

    def project(self, X, pose):
        P, R = pose
        Xc = (X - P) @ R                                   # camera coords (x right, y down, z ahead)
        z = Xc[..., 2]
        zs = np.maximum(z, 1.0)
        xy = np.stack([FOCAL * Xc[..., 0] / zs + DW / 2, FOCAL * Xc[..., 1] / zs + DH / 2], -1)
        return xy, z

    @staticmethod
    def depth_light(z):
        b = np.clip((DIST / np.maximum(z, 1.0)) ** 0.6, 0.0, 1.25)
        fog = np.exp(-np.maximum(z - DIST, 0.0) / FOG)
        return b * fog * smooth((z - NEAR) / 350.0)

    # ---------------------------------------------------------------------------------- state
    def pos3(self, t):
        homes = []
        for k in range(3):
            h2 = self.hover(k, t)
            homes.append(self.w3[k] + np.outer(h2[:, 0], self.U[k]) + np.outer(h2[:, 1], self.V[k]))
        p = self.src3.copy()
        vis = self.vis[0].copy()
        prev = self.src3
        for k, mv in enumerate(self.moves):
            x = (t - mv["d"]) / mv["f"]
            g = flight_soft(x, mv["f"])
            if k > 0:
                vis = vis + (self.vis[k] - self.vis[k - 1]) * smooth((np.clip(g, 0, 1) - 0.3) / 0.4)
            T0, T1 = self.tan3[k]
            g2, g3 = g * g, g * g * g
            hp = ((2 * g3 - 3 * g2 + 1)[:, None] * prev + (g3 - 2 * g2 + g)[:, None] * T0
                  + (-2 * g3 + 3 * g2)[:, None] * homes[k] + (g3 - g2)[:, None] * T1)
            p = p + hp - prev
            prev = homes[k]
        return p, np.clip(vis, 0, 1)

    def plane_index(self, t):
        idx = np.zeros(self.N, int)
        for k, mv in enumerate(self.moves):
            idx = np.where(t >= mv["open"], k, idx)
        return idx

    # ---------------------------------------------------------------------------------- drawing
    def trails(self, t, W, H):
        ss = FP.ss_for(W)
        sc = W / DW * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        TM, TDT = FP.trail_params()
        pose = self.cam_pose(t)
        p, vis = self.pos3(t)
        img = np.empty((Hc, Wc, 3), np.uint8)
        img[:] = BG.astype(np.uint8)
        idx = np.nonzero(vis > 0.01)[0]
        if not idx.size:
            return img
        samp = np.stack([p[idx]] + [self.pos3(t - j * TDT)[0][idx] for j in range(1, TM + 1)], 0)
        xy, z = self.project(samp, pose)
        lit = self.depth_light(z)                          # (M+1, n)
        q = np.round(xy * sc * Sh).astype(np.int32)
        layer = np.zeros((Hc, Wc), np.uint8)
        thick = max(1, int(round(LINE_PX * ss)))
        nb = 12
        for j in range(TM - 1, -1, -1):
            fade = (1 - j / TM) ** TRAIL_FADE
            a = vis[idx] * fade * np.minimum(0.5 * (lit[j] + lit[j + 1]), 1.0)
            a = a * (z[j] > NEAR) * (z[j + 1] > NEAR)
            lev = np.ceil(a * nb).astype(int)
            seg = np.stack([q[j], q[j + 1]], 1)
            for L in range(1, nb + 1):
                sel = lev == L
                if sel.any():
                    cv2.polylines(layer, list(seg[sel]), False, int(255 * L / nb), thick, cv2.LINE_AA, 4)
        lf = layer.astype(np.float32)[..., None] / 255.0
        return (BG + (INK - BG) * lf).astype(np.uint8)

    def draw(self, t, W, H, base):
        ss = FP.ss_for(W)
        sc = W / DW * ss
        Wc, Hc = W * ss, H * ss
        Sh = 16.0
        pose = self.cam_pose(t)
        p, vis = self.pos3(t)
        th1, a1, a2 = self.arms(t)
        pi = self.plane_index(t)
        U = np.stack(self.U)[pi]
        V = np.stack(self.V)[pi]
        e1 = np.cos(th1)[:, None] * U + np.sin(th1)[:, None] * V
        e2 = -np.sin(th1)[:, None] * U + np.cos(th1)[:, None] * V
        h = 0.5 * PLUS * PITCH
        ends = np.stack([p - e1 * (h * a1)[:, None], p + e1 * (h * a1)[:, None],
                         p - e2 * (h * a2)[:, None], p + e2 * (h * a2)[:, None]], 0)
        xy, z = self.project(ends, pose)
        zc = self.project(p, pose)[1]
        lit = np.minimum(self.depth_light(zc), 1.0) * vis
        q = np.round(xy * sc * Sh).astype(np.int64)
        m = 40 * sc * Sh
        ok = ((zc > NEAR) & (lit > 0.01) & (q[0, :, 0] > -m) & (q[0, :, 0] < Wc * Sh + m)
              & (q[0, :, 1] > -m) & (q[0, :, 1] < Hc * Sh + m))
        bar = max(1, int(round(LINE_PX * ss)))
        col = BG + (INK - BG) * lit[:, None]
        img = base.copy()
        for i in self.order:
            if not ok[i]:
                continue
            cl = tuple(float(v) for v in col[i])
            if a1[i] > 0.02:
                cv2.line(img, (int(q[0, i, 0]), int(q[0, i, 1])), (int(q[1, i, 0]), int(q[1, i, 1])),
                         cl, bar, cv2.LINE_AA, 4)
            if a2[i] > 0.02:
                cv2.line(img, (int(q[2, i, 0]), int(q[2, i, 1])), (int(q[3, i, 0]), int(q[3, i, 1])),
                         cl, bar, cv2.LINE_AA, 4)
        if ss > 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img


def main():
    FP.cli("A3c", "A3c_follow", VolScene, [PEEL[0] + 2.0, GATHERED[1] + 0.6], READABLE, DUR,
           [os.path.abspath(__file__), os.path.join(HERE, "follow_plus.py"), os.path.join(HERE, "follow.py"),
            os.path.join(HERE, "particles.py")])


if __name__ == "__main__":
    main()
