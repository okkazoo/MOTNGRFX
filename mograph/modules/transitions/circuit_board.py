"""circuit_board: routes A's departing strands to B's entry like traces on a PCB. Every trace runs on an orthogonal grid with
45 degree corners: the departing ribbon steps inward lane by lane (inner traces first) into a tight bus split in lanes
(groups of traces with a wider gap between them), the bus runs a chicane (a jog out and back, like routing round an
obstacle) and any 90 degree dog-leg the two gates need, then the traces fan out lane by lane (outer first) into B's entry
arrangement. Vias (pad rings) light up at the corners while a trace passes through them."""
import math
import numpy as np
from mograph.modules.core import register, unit, chamfer, _rot_between, Layer, mixc
from mograph.modules.transitions._path import PathTransition, ribbon_axis, hermite_curve  # noqa: F401


def _interp(bp, s):
    """piecewise linear (n,3) breakpoints [(s, a, b)] -> (a, b) at s."""
    S = bp[:, 0]
    return float(np.interp(s, S, bp[:, 1])), float(np.interp(s, S, bp[:, 2]))


@register("circuit_board")
class CircuitBoard(PathTransition):
    """params: lanes (bus lane groups, 3), jog (px, chicane amplitude along the ribbon axis; None = auto, 0 = none),
    pitch (bus trace spacing, None = auto), stagger (px of start offset between neighbouring traces, 0.8 x pitch),
    chamfer (px, corner cut of the grid, None = auto), pads (draw vias, True), stretch (0.3), piece (70)."""
    SUITS = dict(
        brief="same or folded plane, orthogonal / 45 degree moves, techy",
        dist=((80, 220, 750, 1300), 1.0), plane=({"same": 1.0, "fold": 0.85, "opposite": 0.2}, 1.2),
        ortho=((0.25, 0.75, 1.0, 1.01), 2.2), detour=((-1, 0, 100, 160), 0.6), energy=((0.15, 0.4, 1.0, 1.0), 0.6),
        must=("plane", "detour"), dur=(1.6, 3.6, 1.05),
        moods={"tense": 0.9, "hype": 0.7, "mysterious": 0.7, "playful": 0.3, "calm": 0.1, "joyful": 0.3, "nostalgic": 0.1,
               "melancholy": 0.2})
    description = ("Routes the strands like traces on a circuit board: they step in on 45 degree turns into a lane-split bus, "
                   "run a chicane and dog-legs on an orthogonal grid, and step back out to the next word's entry, with via "
                   "rings lighting up at every corner.")

    # ------------------------------------------------------------ geometry
    def paths(self, A, B):
        P = self.params
        K = self.K
        Ga, Tb = A.P[:, 0], B.P[:, -1]
        ca, cb = Ga.mean(0), Tb.mean(0)
        da, db = unit(A.D.mean(0)), unit(B.D.mean(0))
        dist = float(np.linalg.norm(cb - ca))
        W, _ = self.cross_extent(A, B)
        pmin = self.min_pitch(A, B)
        pitch = float(P.get("pitch") or max(pmin, 0.6 * W / max(K - 1, 1)))
        G = int(max(1, min(P.get("lanes", 3), K)))
        gg = 1.6 * pitch
        stag = float(P.get("stagger", 0.8 * pitch))
        # --- lateral axis g (ribbon axis of A), perpendicular to da
        g = ribbon_axis(Ga)
        if g is None:
            g = np.cross(da, [0.0, 0.0, 1.0])
            if np.linalg.norm(g) < 0.2:
                g = np.cross(da, [0.0, 1.0, 0.0])
        g = unit(g - (g @ da) * da)
        if g[1] > 0.02 or (abs(g[1]) <= 0.02 and g[0] < 0):
            g = -g
        e1s = [g]
        # --- A / B cross-section offsets (B in the frame of the last segment, resolved after the centreline is built)
        relA = Ga - ca
        alA, beA = relA @ g, relA @ np.cross(da, g)
        # lane assignment: rank by lateral position of A's heads
        order = np.argsort(alA, kind="stable")
        rank = np.empty(K, int); rank[order] = np.arange(K)
        grp = np.minimum(rank * G // K, G - 1)
        pos = rank * pitch + grp * gg
        bl = pos - pos.mean()
        centre_rank = np.argsort(np.argsort(np.abs(bl), kind="stable"), kind="stable")     # 0 = innermost
        jog = P.get("jog")
        jog = (float(np.clip(0.5 * W, 3.0 * pitch, 0.9 * dist / 4.0)) if jog is None else float(jog)) if G > 0 else 0.0
        # slope of the departing / arriving strands (lane fan) in the lateral frame
        sl_a = np.array([((A.D[k] @ g) / max(A.D[k] @ da, 0.3), (A.D[k] @ np.cross(da, g)) / max(A.D[k] @ da, 0.3)) for k in range(K)])
        # --- try ramp scales until the regions fit into the centreline
        ch = float(P.get("chamfer") or max(0.5 * (bl.max() - bl.min()) * 0.5 + 2 * pitch, 18.0))
        for sc in (1.0, 0.7, 0.45, 0.2, 0.0):
            stg, jg = stag * sc, jog * sc
            lead = 0.04 * dist + 1.5 * pitch
            dm = np.hypot(bl - alA, 0 - beA)                                      # merge ramp lengths (Chebyshev ~ 45 deg)
            dm = np.maximum(np.abs(bl - alA), np.abs(beA))
            merge_len = float((stg * centre_rank + dm).max()) if K else 0.0
            l0 = lead + merge_len + 0.5 * pitch
            self_l1 = None
            # B-side numbers need the last segment frame: approximate with the lateral frame of db (rotation of g)
            R = _rot_between(da, db)
            gB = R @ g
            relB = Tb - cb
            alB0, beB0 = relB @ gB, relB @ np.cross(db, gB)
            df = np.maximum(np.abs(alB0 - bl), np.abs(beB0))
            fan_len = float((stg * (K - 1 - centre_rank) + df).max())
            l1 = lead + fan_len + 0.5 * pitch
            p1, p3 = ca + da * l0, cb - db * l1
            D = p3 - p1
            Df = float(D @ da)
            if Df >= 0.25 * dist * sc or sc == 0.0:
                break
        self.pitch, self.jog_amp = pitch, jg
        # --- centreline: seg0 along da, middle legs (da / q1 / da), last segment along db
        Dl = D - Df * da
        legs = []
        if np.linalg.norm(Dl) > 2.0:
            q1 = unit(Dl)
            legs = [da * Df * 0.5, Dl, da * Df * 0.5]
        else:
            legs = [da * Df]
        pts = [ca, p1]
        for lg in legs:
            pts.append(pts[-1] + lg)
        pts.append(cb)
        pts = [np.asarray(x, float) for x in pts]
        # drop zero-length legs
        keep = [pts[0]]
        for x in pts[1:]:
            if np.linalg.norm(x - keep[-1]) > 1e-3:
                keep.append(x)
        i = 1
        while i < len(keep) - 1:                           # collinear points are not corners
            u1, u2 = unit(keep[i] - keep[i - 1]), unit(keep[i + 1] - keep[i])
            if float(u1 @ u2) > 0.9999:
                keep.pop(i)
            else:
                i += 1
        CL = chamfer(keep, ch)
        d_ = np.diff(CL, axis=0)
        seglen = np.linalg.norm(d_, axis=1)
        okseg = seglen > 1e-6
        CL = np.vstack([CL[:1], CL[1:][okseg]])
        d_ = unit(np.diff(CL, axis=0))
        seglen = np.linalg.norm(np.diff(CL, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seglen)]
        Lc = float(cum[-1])
        nseg = len(d_)
        e1 = [g - (g @ d_[0]) * d_[0]]
        e1[0] = unit(e1[0])
        for i in range(1, nseg):
            e1.append(unit(_rot_between(d_[i - 1], d_[i]) @ e1[-1]))
        e2 = [np.cross(d_[i], e1[i]) for i in range(nseg)]
        # B cross-section in the last segment's frame
        relB = Tb - cb
        alB, beB = relB @ e1[-1], relB @ e2[-1]
        alA, beA = relA @ e1[0], relA @ e2[0]
        # --- stage layout on the arc coordinate
        free = []                                           # straight stretches [(s0, s1)] where ramps may live (corner cuts excluded)
        for i in range(nseg):
            a, b = cum[i], cum[i + 1]
            nxt = i + 1 < nseg and float(d_[i] @ d_[i + 1]) < 0.999
            prv = i > 0 and float(d_[i - 1] @ d_[i]) < 0.999
            if prv and seglen[i] < 4 * ch:                    # a chamfer diagonal / short piece
                continue
            free.append((a + (0.0), b))
        S_a = lead                                      # first ramp start
        S_b = Lc - lead                                 # last ramp end
        merge_s = S_a + stg * centre_rank
        merge_e = merge_s + dm
        fan_e = S_b - stg * centre_rank                 # outer strands finish... mirrored: start order outer first
        fan_s = fan_e - df
        # chicane: in the first / last straight stretch of the bus run
        jog_ok = abs(jg) > 1e-3 and G >= 1
        j1s = j2s = None
        if jog_ok:
            bus0, bus1 = float(merge_e.max()) + 1.0 * pitch, float(fan_s.min()) - 1.0 * pitch
            need = abs(jg) + stg * (G - 1) + pitch
            seg_free = [(max(a, bus0), min(b, bus1)) for a, b in free]
            seg_free = [(a, b) for a, b in seg_free if b - a > 0.0]
            if seg_free:
                a0, b0 = seg_free[0]
                a1, b1 = seg_free[-1]
                if len(seg_free) == 1:
                    if b0 - a0 >= 2 * need + 2 * pitch:
                        j1s, j2s = a0 + 0.5 * pitch, b0 - need - 0.5 * pitch
                else:
                    if b0 - a0 >= need + pitch and b1 - a1 >= need + pitch:
                        j1s, j2s = a0 + 0.5 * pitch, b1 - need - 0.5 * pitch
        if j1s is None:
            jog_ok = False
        self.stage = dict(CL=CL, cum=cum, d=d_, e1=e1, e2=e2, merge=(merge_s, merge_e), fan=(fan_s, fan_e),
                          j1=j1s, j2=j2s, jog=jg, S_a=S_a, S_b=S_b)
        # --- per strand breakpoints (s, al, be)
        paths, pads = [], []
        # group order for the chicane stagger: the group leading the jog direction (g > 0 side) goes first
        g_lead = (G - 1 - grp)                          # grp 0 = most negative lateral = trails; jog is toward +e1 side
        for k in range(K):
            bp = [(0.0, alA[k], beA[k])]
            a0, b0 = alA[k] + sl_a[k, 0] * S_a * 0.0, beA[k]
            bp.append((merge_s[k], alA[k] + sl_a[k, 0] * merge_s[k], beA[k] + sl_a[k, 1] * merge_s[k]))
            bp.append((merge_e[k], bl[k], 0.0))
            padk = [merge_e[k]]
            cur = bl[k]
            if jog_ok:
                s1 = j1s + stg * g_lead[k] * 1.0
                bp.append((s1, cur, 0.0)); bp.append((s1 + abs(jg), cur + jg, 0.0))
                s2 = j2s + stg * g_lead[k] * 1.0
                bp.append((s2, cur + jg, 0.0)); bp.append((s2 + abs(jg), cur, 0.0))
                padk += [s1 + abs(jg), s2]
            bp.append((fan_s[k], bl[k], 0.0))
            padk.append(fan_s[k])
            bp.append((fan_e[k], alB[k] - 0.0 * 0, beB[k]))
            bp.append((Lc, alB[k], beB[k]))
            bp = np.array(sorted(bp, key=lambda r: r[0]), float)
            # make sure s is non-decreasing and strictly separated
            for i in range(1, len(bp)):
                if bp[i, 0] <= bp[i - 1, 0] + 1e-6:
                    bp[i, 0] = bp[i - 1, 0] + 1e-6
            # vertices: breakpoints + centreline corner vertices
            ss = sorted(set([round(float(x), 6) for x in bp[:, 0]] + [round(float(x), 6) for x in cum]))
            ss = [x for x in ss if 0.0 <= x <= Lc]
            poly, pad_pts = [], []
            for s in ss:
                i = int(min(np.searchsorted(cum, s, "right") - 1, nseg - 1))
                al, be = _interp(bp, s)
                at_vertex = i > 0 and abs(s - cum[i]) < 1e-5
                Cs = CL[i] + d_[i] * (s - cum[i])
                if at_vertex:                                  # miter between segment i-1 and i
                    o0 = al * e1[i - 1] + be * e2[i - 1]
                    o1 = al * e1[i] + be * e2[i]
                    m = d_[i - 1] - (d_[i - 1] @ d_[i]) * d_[i]
                    den = float(d_[i - 1] @ m)
                    sft = -float((o0 - o1) @ m) / den if abs(den) > 1e-6 else 0.0
                    poly.append(Cs + o0 + sft * d_[i - 1])
                else:
                    poly.append(Cs + al * e1[i] + be * e2[i])
            poly = np.array(poly)
            poly[0] = Ga[k]
            poly[-1] = Tb[k]
            # pads: vertices at ramp ends / starts
            for s in padk:
                j = int(np.argmin(np.abs(np.array(ss) - s)))
                i = int(min(np.searchsorted(cum, s, "right") - 1, nseg - 1))
                pad_pts.append((poly[j].copy(), d_[i].copy(), e1[i].copy()))
            paths.append(poly)
            pads.append(pad_pts)
        self.pads = pads
        self.merge_e = merge_e
        return paths

    def build(self):
        super().build()
        # arc position of every pad on its route
        self.pad_s = []
        for k, pk in enumerate(self.pads):
            R = self.routes[k]
            row = []
            for (c, dv, ev) in pk:
                j = int(np.argmin(np.linalg.norm(R.P - c, axis=1)))
                row.append(float(R.cum[j]))
            self.pad_s.append(row)

    # ------------------------------------------------------------ pads
    def extra_layers(self, t):
        if not self.params.get("pads", True):
            return []
        col, w, rim, tube = self._mix(t)
        out = []
        th = np.linspace(0, 2 * math.pi, 14)
        wins = self.windows(t)
        for k, (win, h, tl) in enumerate(wins):
            if win is None:
                continue
            for (c, dv, ev), s in zip(self.pads[k], self.pad_s[k]):
                if tl - 2.0 <= s <= h + 2.0:
                    r_o = 0.95 * w[k] * (1 + 0.6 * tube) + 3.0
                    r_i = 0.45 * w[k]
                    ring = [c + r_o * (np.cos(th)[:, None] * dv + np.sin(th)[:, None] * ev),
                            c + r_i * (np.cos(th)[:, None] * dv + np.sin(th)[:, None] * ev)]
                    out.append(Layer(ring, mixc(tuple(int(x) for x in col[k]), (255, 250, 240), 0.55), "fill", bias=9000.0))
        return out

    def bounds_points(self):
        return np.concatenate([self.stage["CL"], self.A.P.reshape(-1, 3), self.B.P.reshape(-1, 3)])
