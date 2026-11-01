"""staircase: scale clash by routing. The strands arrive along the entry direction and first draw a TINY copy of the word (a small
hairline outline wire at the entry-side corner); then every edge of the tiny word leaves along its own stub and runs the SAME
staircase, one parallel bundle of lanes that steps down and across in two alternating directions, spreading at the foot of the
stairs into the wire outline of the vast word, which fills solid over a shallow stepped slab (linear fill). Leaving is the
mirror: the fill melts, the strands run back down a staircase to a tiny copy at the exit-side corner, which hairlines out along
the exit direction. (mechanism: 11_mathtype a4 'staircase', rewritten for the Bundle interface)

2026-11-01 (user: "lose the part that makes it small then large"): the ENTRY no longer draws a tiny copy first. The strands run the lanes
straight onto the staircase and spread into the outline of the word at its final size (so the word is never smaller than its hold scale
from its first drawn frame); the stair / stagger motion is unchanged. Later the same day ("never build or show a small copy of a word mid-transition")
the EXIT unravels from the full-size word too (big arc -> stairs -> lane out). Param tiny_copy=True brings the old entry AND exit back.

Params: from_words (the tiny word, default = the words), tiny_copy (old tiny-copy entry, default False), n_strands, width, steps (staircase steps, default 2), lanes (lane slots
across the bundle, default 36), slab (depth of the filled slab in placement widths, default 0.05)."""
import math
import numpy as np
from mograph.modules.core import (register, Bundle, Route3, Slide, Layer, layout_word, norm_lines, unit, ol, strand_layers,
                                  clamp, mixc, bezier)
from mograph.modules.words._slab import (SlabWord, ssm, simplify, contour_sign, wall_quads, wall_layers)
from mograph.modules.words._strand import make_arcs

WALLS = ((150, 145, 136), (92, 89, 83), (52, 50, 46))


def plen(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def tangent(a, start):
    k = min(5, len(a) - 1)
    return unit(a[k] - a[0]) if start else unit(a[-1] - a[-1 - k])


@register("staircase")
class Staircase(SlabWord):
    description = ("The strands first draw a tiny copy of the word, then every edge of it runs one shared staircase of parallel "
                   "lanes, stepping down and across in two alternating directions, and spreads into the outline of the vast "
                   "word, which fills solid. It leaves the same way in reverse, down a staircase to a tiny copy that hairlines "
                   "out along the exit direction.")
    DEMO_WORDS = "VAST"
    DEFAULT_FONT = "silkscreen"
    P_IN = 2.4
    LINE_W = 1.5

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _frame(d3, pl):
        d2 = np.array([d3 @ pl.u, d3 @ pl.v])
        e = d2 / np.linalg.norm(d2) if np.linalg.norm(d2) > 0.35 else np.array([1.0, 0.0])
        perp = np.array([-e[1], e[0]])
        if perp[1] < -1e-6 or (abs(perp[1]) <= 1e-6 and perp[0] < 0):
            perp = -perp
        return e, perp

    @staticmethod
    def _basis(e, perp, delta):
        X, Y = float(delta @ e), float(delta @ perp)
        if X >= 0 and Y >= 0.58 * X:
            return perp, unit(0.866 * e + 0.5 * perp)
        if X >= 0:
            return e, unit(0.5 * e + 0.866 * perp)
        return perp, unit(-0.866 * e + 0.5 * perp)

    @staticmethod
    def _legs(p, q, B0, B1):
        a, b = np.linalg.solve(np.stack([B0, B1], 1), q - p)
        return [p + B0 * a, q]

    def _stair(self, C0, C1, B0, B1, steps):
        a, b = np.linalg.solve(np.stack([B0, B1], 1), C1 - C0)
        st = [C0]
        for _ in range(steps):
            st.append(st[-1] + B0 * a / steps)
            st.append(st[-1] + B1 * b / steps)
        return np.array(st)

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 3)))
        k = pl.w / 960.0
        self.k = k
        cap = self.cap
        self.w = float(P.get("width") or self.LINE_W * max(0.8, cap / 200.0))
        self.Lb = float(P.get("tail_len") or 0.2 * max(pl.w, pl.h))
        steps = int(P.get("steps", 2))
        NLN = int(P.get("lanes", 36))
        tiny_copy = bool(P.get("tiny_copy", False))
        w2, h2 = 0.5 * pl.w, 0.5 * pl.h
        tiny_lines = norm_lines(P["from_words"]) if P.get("from_words") else self.lines
        tlay = layout_word(self.font, tiny_lines, 0.15 * pl.w, 0.09 * pl.h)
        tcons = [c for l in tlay["letters"] for c in l["cons"]]
        e_in, p_in = self._frame(self.d_in, pl)
        e_out, p_out = self._frame(self.d_out, pl)
        Ee = lambda e: float(np.abs(e) @ np.array([w2, h2]))
        # tiny word centres and staircase corners (local 2D, relative to the placement centre)
        Ct_in = -e_in * 0.40 * Ee(e_in) - p_in * 0.36 * Ee(p_in)
        Ct_out = e_out * 0.40 * Ee(e_out) - p_out * 0.36 * Ee(p_out)
        C0_in = Ct_in + e_in * 0.10 * Ee(e_in) + p_in * 0.22 * Ee(p_in)
        C1_in = p_in * 0.95 * Ee(p_in)
        C0_out = Ct_out - e_out * 0.10 * Ee(e_out) + p_out * 0.22 * Ee(p_out)
        C1_out = p_out * 0.95 * Ee(p_out)
        bi = self._basis(e_in, p_in, C1_in - C0_in)
        bo = self._basis(-e_out, p_out, C1_out - C0_out)         # exit stairs run from the foot back up to the tiny word
        stair_in = self._stair(C0_in, C1_in, bi[0], bi[1], steps)
        stair_out = self._stair(C0_out, C1_out, bo[0], bo[1], steps)
        od_in = np.array([-bi[1][1], bi[1][0]])
        od_out = np.array([-bo[1][1], bo[1][0]])
        # arcs
        perim = sum(float(ol.arclen(np.vstack([c, c[:1]]))[-1]) for c in self.cons)
        N = int(P.get("n_strands") or np.clip(round(perim / (0.55 * cap)), 12, 90))
        N = max(N, len(self.cons), len(tcons))
        big = make_arcs(self.cons, N)
        for _ in range(3):
            t_in = make_arcs([c + Ct_in for c in tcons], N, step=1.5)
            t_out = make_arcs([c + Ct_out for c in tcons], N, step=1.5)
            if len(big) == len(t_in) == len(t_out):
                break
            N = max(len(big), len(t_in), len(t_out))
            big = make_arcs(self.cons, N)
        N = min(len(big), len(t_in), len(t_out))
        self.N = N
        big, t_in, t_out = big[:N], t_in[:N], t_out[:N]
        key = lambda od: (lambda a: float(a.mean(0) @ od))
        big_i = sorted(big, key=key(od_in))
        big_o = sorted(big, key=key(od_out))
        t_in = sorted(t_in, key=key(od_in))
        t_out = sorted(t_out, key=key(od_out))
        gap = 6.5 * k * float(P.get("lane_gap", 1.0))
        gate_in, gate_out = self.reach_in, self.reach_out
        o_ = pl.origin
        self.R_in, self.R_out = [], []
        self.keys_in, self.keys_out = [], []
        L3 = lambda A: pl.lift(np.asarray(A, float))
        for i in range(N):
            off_i = od_in * gap * (i * NLN // N - NLN / 2)
            off_o = od_out * gap * (i * NLN // N - NLN / 2)
            # ---- entry route: lane -> peel -> tiny arc -> stub -> legs -> stairs -> legs -> big arc
            ta, ba = t_in[i], big_i[i]
            ts, te = tangent(ta, True), tangent(ta, False)
            tb = tangent(ba, True)
            if not tiny_copy:                                  # no tiny copy: the lane ends at the foot of the stairs, no scale ramp
                A0 = L3(np.array([stair_in[0] + off_i]))[0]
                peel = A0 - self.d_in * 20.0
                gd = gate_in + float((A0 - o_) @ self.d_in) + 30.0
                J = peel - self.d_in * max(gd - 20.0, 10.0)
                S = J - self.d_in * self.Lb
                q1 = ba[0] - tb * rng.uniform(10, 30) * k
                mid2 = [p + off_i for p in stair_in] + self._legs(stair_in[-1] + off_i, q1, bi[0], bi[1])[:]
                mid3 = L3(np.array(mid2))
                pts_big = L3(ba)
                lane = np.vstack([S, J, peel])
                full = np.vstack([lane, mid3, pts_big])
                s_t1 = plen(np.vstack([lane, mid3[:1]]))
                s_t0 = max(s_t1 - self.Lb, 0.0)
                s_b0 = s_t1 + plen(np.vstack([mid3, pts_big[:1]]))
                R = Route3(full)
                self.R_in.append(R)
                self.keys_in.append((s_t0, s_t1, s_b0, R.L))
            else:
                p_arc0 = L3(ta[:1])[0]
                peel = p_arc0 - self.d_in * 20.0
                Dl = unit(p_arc0 - peel)
                con = bezier(peel, peel + self.d_in * 8.0, p_arc0 - (L3(ta[1:2])[0] - p_arc0) / max(np.linalg.norm(L3(ta[1:2])[0] - p_arc0), 1e-6) * 8.0, p_arc0)
                gd = gate_in + float((p_arc0 - o_) @ self.d_in) + 30.0
                J = peel - self.d_in * max(gd - 20.0, 10.0)
                S = J - self.d_in * self.Lb
                lane = np.vstack([S, J, peel, con[1:]])
                q0 = ta[-1] + te * rng.uniform(6, 16) * k
                q1 = ba[0] - tb * rng.uniform(10, 30) * k
                mid2 = [q0] + self._legs(q0, stair_in[0] + off_i, bi[0], bi[1]) + [p + off_i for p in stair_in[1:]] + \
                    self._legs(stair_in[-1] + off_i, q1, bi[0], bi[1])
                mid3 = L3(np.array(mid2))
                pts_tiny = L3(ta)
                pts_big = L3(ba)
                full = np.vstack([lane, pts_tiny[1:], mid3, pts_big])
                s_t0 = plen(lane)
                s_t1 = s_t0 + plen(pts_tiny)
                s_b0 = s_t1 + plen(np.vstack([pts_tiny[-1:], mid3, pts_big[:1]]))
                R = Route3(full)
                self.R_in.append(R)
                self.keys_in.append((s_t0, s_t1, s_b0, R.L))
            # ---- exit route: big arc -> stub -> legs -> stairs (back up) -> legs -> tiny arc -> stub -> lane out
            ba2, ta2 = big_o[i], t_out[i]
            be = tangent(ba2, False)
            ts2, te2 = tangent(ta2, True), tangent(ta2, False)
            q0 = ba2[-1] + be * rng.uniform(6, 16) * k
            q1 = ta2[0] - ts2 * rng.uniform(10, 30) * k
            so = stair_out[::-1]
            mid2 = [q0] + self._legs(q0, so[0] + off_o, bo[0], bo[1]) + [p + off_o for p in so[1:]] + \
                self._legs(so[-1] + off_o, q1, bo[0], bo[1])
            if not tiny_copy:                                  # 2026-11-01 (user: "never a smaller word in between"): the exit unravels from the FULL-SIZE word: big arc -> stairs -> lane out, no tiny copy
                mid2 = [q0] + self._legs(q0, so[0] + off_o, bo[0], bo[1]) + [p + off_o for p in so[1:]]
                mid3 = L3(np.array(mid2))
                pts_big2 = L3(ba2)
                A_end = mid3[-1]
                peel2 = A_end + self.d_out * 20.0
                gd2 = gate_out - float((A_end - o_) @ self.d_out) + 30.0
                J2 = peel2 + self.d_out * max(gd2 - 20.0, 10.0)
                lane_end = J2 + self.d_out * (2.5 * max(pl.w, pl.h))
                full2 = np.vstack([pts_big2, mid3, peel2[None], J2[None], lane_end[None]])
                Lbig = plen(pts_big2)
                s2_t1 = plen(np.vstack([pts_big2, mid3]))
                s2_t0 = max(s2_t1 - self.Lb, 0.0)
                Lend = s2_t1 + plen(np.vstack([mid3[-1:], peel2[None], J2[None]]))
                self.R_out.append(Route3(full2))
                self.keys_out.append((Lbig, s2_t0, s2_t1, Lend))
                continue
            mid3 = L3(np.array(mid2))
            pts_big2, pts_tiny2 = L3(ba2), L3(ta2)
            p_end = pts_tiny2[-1]
            peel2 = p_end + self.d_out * 20.0
            con2 = bezier(p_end, p_end + L3(ta2[-1:])[0] * 0 + (p_end - pts_tiny2[-2]) / max(np.linalg.norm(p_end - pts_tiny2[-2]), 1e-6) * 8.0,
                          peel2 - self.d_out * 8.0, peel2)
            gd2 = gate_out - float((p_end - o_) @ self.d_out) + 30.0
            J2 = peel2 + self.d_out * max(gd2 - 20.0, 10.0)
            coast = 2.5 * max(pl.w, pl.h)
            lane_end = J2 + self.d_out * coast
            full2 = np.vstack([pts_big2, mid3, pts_tiny2[1:], con2[1:], J2, lane_end])
            Lbig = plen(pts_big2)
            s2_t0 = Lbig + plen(np.vstack([pts_big2[-1:], mid3, pts_tiny2[:1]]))
            s2_t1 = s2_t0 + plen(pts_tiny2)
            Lend = s2_t1 + plen(np.vstack([pts_tiny2[-1:], con2[1:], J2[None]]))
            R2 = Route3(full2)
            self.R_out.append(R2)
            self.keys_out.append((Lbig, s2_t0, s2_t1, Lend))
        # kinematics
        tm = self.timing
        F, E = tm.form, tm.exit
        f = np.arange(N) / max(N - 1, 1)
        self.A, self.B, self.C, self.D = [], [], [], []
        self.vin = np.zeros(N)
        self.vout = np.zeros(N)
        Lb = self.Lb
        for i in range(N):
            R, (s_t0, s_t1, s_b0, L) = self.R_in[i], self.keys_in[i]
            TA = (0.30 + 0.10 * f[i]) * F
            V = float(P.get("speed") or 1.35 * (s_t1 - Lb) / TA)
            self.vin[i] = min(V * TA, 2.9 * (s_t1 - Lb)) / TA
            self.A.append(Slide(R, Lb, s_t1, 0.0, s_t0, TA, V, 0.0))
            tB = (0.55 + 0.04 * f[i]) * F
            TB = 0.40 * F
            self.B.append((tB, Slide(R, s_t1, L, s_t0, s_b0, TB, 0.0, 0.0, delay=tB)))
            R2, (Lbig, s2_t0, s2_t1, Lend) = self.R_out[i], self.keys_out[i]
            dC = (0.14 + 0.08 * f[i]) * E
            TC = 0.34 * E
            dD = (0.56 + 0.06 * f[i]) * E
            TD = E - dD
            self.C.append((dD, Slide(R2, Lbig, s2_t1, 0.0, s2_t0, TC, 0.0, 0.0, delay=dC)))
            Vo = float(P.get("speed") or 1.35 * (Lend - s2_t1) / TD)
            self.vout[i] = min(Vo * TD, 2.9 * (Lend - s2_t1)) / TD
            sd = Slide(R2, s2_t1, Lend, s2_t0, Lend - Lb, TD, 0.0, Vo, delay=dD)
            sd.coast = True
            self.D.append(sd)
        self.cols = np.tile(np.array(self.palette.fg, float), (N, 1))
        # filled slab
        tz = 0.25
        a = (0.87 * pl.u - 0.5 * pl.v)
        self.slab_ax = unit(unit(a) + tz * pl.n)
        self.slab_D = float(P.get("slab", 0.05)) * pl.w
        self.cons_s = [[simplify(c, 0.5) for c in l["cons"]] for l in self.letters]
        self.sg = [contour_sign(l["cons"]) for l in self.letters]

    # ------------------------------------------------------------ windows
    def windows(self, t):
        tm = self.timing
        out = []
        if t < tm.form + tm.hold:
            for i in range(self.N):
                tB, sB = self.B[i]
                out.append((self.A[i] if t < tB else sB).window(t)[0])
        else:
            te = t - tm.form - tm.hold
            for i in range(self.N):
                dD, sC = self.C[i]
                out.append((sC if te < dD else self.D[i]).window(te)[0])
        return out

    def bundle_at(self, t):
        K = 8
        wins = self.windows(t)
        entry = t < self.timing.form + self.timing.hold
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        for i, wn in enumerate(wins):
            seg = wn[::-1]                                     # head first
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
            s = np.linspace(0, cum[-1], K)
            Pm[i] = np.stack([np.interp(s, cum, seg[:, c]) for c in range(3)], 1)
            Dv[i] = unit(seg[0] - seg[min(3, len(seg) - 1)])
        Dv = np.tile(self.d_in if entry else self.d_out, (self.N, 1))
        return Bundle(Pm, Dv, self.cols, self.w, t, self.vin if entry else self.vout, 0.0)

    def fill_alpha(self, t):
        tm = self.timing
        ph, lt = self.phase(t)
        if ph == "form":
            return ssm(t, 0.88 * tm.form, 1.0 * tm.form)
        if ph == "hold":
            return 1.0
        return 1.0 - ssm(lt, 0.0, 0.16 * tm.exit)

    def layers(self, t):
        pl, pal = self.placement, self.palette
        L = []
        a = self.fill_alpha(t)
        if a > 0.003:
            D = self.slab_D * a
            slabs = [wall_quads(pl, self.cons_s[j], self.sg[j], np.zeros(3), self.slab_ax, D, D + 1.0, light=(-0.5, -0.86))
                     for j in range(len(self.letters))]
            L += wall_layers(slabs, 1e9, 1.0, WALLS[0], WALLS[2], bias0=3.0, level_cols=(WALLS[2], WALLS[1], WALLS[0]))
            L += self.fill_layer(mixc(pal.bg, pal.fg, a), 1.0, 5.0)
        wins = self.windows(t)
        L += strand_layers(wins, self.cols, [self.w] * self.N, 0.0)
        return L
