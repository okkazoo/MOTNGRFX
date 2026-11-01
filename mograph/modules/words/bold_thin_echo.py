"""bold_thin_echo: one cohesive word, bold to thin, with a stack of echoes running off it. The whole word is a single unit: one
weight value for every letter (variable font), letters locked on fixed slots (the width axis is solved per letter so the ink
width stays constant and nothing reflows), one shared stack of ~22 echo outlines (each copy scaled about the word centre and
stepped along one axis, filled with the ground so it hides the copies behind) so bold letters throw heavy echoes and thin
letters hairline ones. Forming: the word slides in along the entry direction trailing its echo like a motion streak (the
outlines fade in as it lands, the strands handed over are the corner trails through the stack); then the echo axis swings round
as one while the weight sweeps bold to thin. Leaving: the echo swings back into a trailing streak and the word runs out along
the exit direction. (mechanism: 11_mathtype b7R2 / 13_camera strips Echo, rewritten for the Bundle interface)

Needs a variable-weight font (a wght axis): default tourney (liked); anybody also works. A font without a wght axis falls back
to tourney. Params: wght_from (900), wght_to (250), echoes (22), swing (degrees of axis swing, 205), growth (echo scale
exponent, -0.4), step (px per echo, default 0.0078 x placement width)."""
import os, math, pickle, tempfile
import numpy as np
from mograph.modules.core import register, Bundle, Layer, unit, mixc, clamp, resolve_font, ol
from mograph.modules.words._slab import (SlabWord, LetterFlow, ssm, eo, ei, eio, simplify, nz_groups, corner_points)
from fontTools.ttLib import TTFont


def _axes(key):
    f = TTFont(ol.font_path(key))
    if "fvar" not in f:
        return f, {}
    return f, {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes}


class VarRow:
    """letters of one variable-font line on fixed slots. Per call the weight is set and the width axis (if any) is solved so
    the ink width of every letter stays constant. Units: font units, y up. Tables cached on disk (temp dir)."""

    def __init__(self, key, text, wlo, whi, f=0.55, nwg=5, nwd=6):
        self.key = key
        self.font, ax = _axes(key)
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.chars = [c for c in text]
        self.wlo, self.whi = max(wlo, ax["wght"][0]), min(whi, ax["wght"][2])
        self.has_wd = "wdth" in ax
        self.f = f
        self.default = {k: v[1] for k, v in ax.items()}
        self._c = {}
        letters = [c for c in text if c != " "]
        self.wg = np.linspace(self.wlo, self.whi, nwg)
        self.wd = np.linspace(ax["wdth"][0], ax["wdth"][2], nwd) if self.has_wd else np.array([0.0])
        cp = os.path.join(tempfile.gettempdir(), f"mograph_vw_{key}_{''.join(sorted(set(letters)))}_{int(self.wlo)}_{int(self.whi)}.pkl")
        if os.path.exists(cp):
            self.tab = pickle.load(open(cp, "rb"))
        else:
            self.tab = {}
            for c in sorted(set(letters)):
                T = np.array([[self._ink(c, a, b) for b in self.wd] for a in self.wg])
                self.tab[c] = np.maximum.accumulate(T, axis=1)
            pickle.dump(self.tab, open(cp, "wb"))
        self.lb, self.ub, self.slot = {}, {}, {}
        for c, T in self.tab.items():
            lb, ub = float(T[:, 0].max()), float(T[:, -1].min())
            self.slot[c] = max(lb, ub)
            self.lb[c], self.ub[c] = min(lb, 0.7 * ub), ub
        self.cap = self._cap()

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("font", None)
        d.pop("cmap", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.font, _ = _axes(self.key)
        self.cmap = self.font.getBestCmap()

    def _loc(self, wg, wd):
        loc = dict(self.default)
        loc["wght"] = float(wg)
        if self.has_wd:
            loc["wdth"] = float(wd)
        return loc

    def _raw(self, ch, wg, wd):
        gs = self.font.getGlyphSet(location=self._loc(wg, wd))
        pen = ol.FlatPen(gs, 3.0)
        gs[self.cmap[ord(ch)]].draw(pen)
        return pen.contours

    def _ink(self, ch, wg, wd):
        cs = self._raw(ch, wg, wd)
        return float(np.ptp(np.concatenate(cs)[:, 0])) if cs else 0.0

    def _cap(self):
        cs = self._raw("H", 0.5 * (self.wlo + self.whi), 100.0 if not self.has_wd else self.default["wdth"])
        P = np.concatenate(cs)
        return float(P[:, 1].max() - P[:, 1].min())

    def solve(self, ch, wght):
        if not self.has_wd:
            return 0.0
        T = self.tab[ch]
        k = float(np.interp(wght, self.wg, np.arange(len(self.wg))))
        k0 = int(min(k, len(self.wg) - 2))
        a = k - k0
        row = (1 - a) * T[k0] + a * T[k0 + 1]
        target = self.lb[ch] + (self.ub[ch] - self.lb[ch]) * self.f
        return float(np.interp(target, row + np.arange(len(row)) * 1e-6, self.wd))

    def glyph(self, ch, wght):
        w = float(round(wght / 20.0) * 20.0)
        w = min(max(w, self.wlo), self.whi)
        wd = round(self.solve(ch, w), 0)
        k = (ch, w, wd)
        if k not in self._c:
            self._c[k] = self._raw(ch, w, wd)
        return self._c[k]


@register("bold_thin_echo")
class BoldThinEcho(SlabWord):
    description = ("One cohesive word goes from very bold to hairline thin with a stack of outline echoes running off it, "
                   "heavy echoes behind the bold letters and hairline ones behind the thin. It slides in trailing its echo "
                   "like a streak, the echo swings round as one while the weight sweeps, and it leaves trailing the echo "
                   "again.")
    DEMO_WORDS = "THE SAME"
    DEFAULT_FONT = "tourney"
    P_IN, P_OUT = 2.4, 2.2

    # ------------------------------------------------------------ layout
    def build(self):
        pl, P = self.placement, self.params
        key, _ = resolve_font(self.font)
        f, ax = _axes(key)
        if "wght" not in ax:
            key = "tourney" if self.font != "tourney" else key
            self.font = key
        self.fkey = key
        self.w_from = float(P.get("wght_from", 900))
        self.w_to = float(P.get("wght_to", 250))
        self.rows = [VarRow(self.fkey, ln, min(self.w_from, self.w_to), max(self.w_from, self.w_to)) for ln in self.lines]
        n = len(self.rows)
        gap, wgap = 0.07, 0.55
        # per row: slot centres in units (relative to the row centre), total width in units
        self.layout = []
        wmax, hmax = pl.w * 0.94, pl.h * 0.9
        for r in self.rows:
            xs, x = [], 0.0
            for i, ch in enumerate(r.chars):
                if ch == " ":
                    x += wgap * r.cap
                    continue
                xs.append((ch, x + 0.5 * r.slot[ch]))
                x += r.slot[ch] + gap * r.cap
            width = x - gap * r.cap
            self.layout.append((xs, width))
        cap_u = self.rows[0].cap
        blockH = cap_u * (n + 0.22 * (n - 1))
        s = min(wmax / max(w for _, w in self.layout), hmax / blockH)
        self.s = s
        self.cap = cap_u * s
        self.ycs = []
        for i in range(n):
            base = -0.5 * blockH * s + cap_u * s + i * cap_u * 1.22 * s
            self.ycs.append(base)
        self.d_in = -self.entry.side(pl)
        self.d_out = self.exit.side(pl)
        self.n3 = pl.n
        self.reach_in = self._reach(self.d_in)
        self.reach_out = self._reach(self.d_out)
        self.setup()

    def row_contours(self, ri, wght):
        """local 2D contours (y down) of row ri at weight wght, per letter: list of (letter index, [contours])."""
        r = self.rows[ri]
        xs, width = self.layout[ri]
        out = []
        for j, (ch, cx) in enumerate(xs):
            cs = r.glyph(ch, wght)
            if not cs:
                continue
            P = np.concatenate(cs)
            mx = 0.5 * (P[:, 0].min() + P[:, 0].max())
            X0 = (cx - 0.5 * width)
            out.append([np.stack([(c[:, 0] - mx) * self.s + X0 * self.s, self.ycs[ri] - c[:, 1] * self.s], 1) for c in cs])
        return out

    def setup(self):
        pl, P = self.placement, self.params
        rng = np.random.RandomState(int(P.get("seed", 2)))
        self.NE = int(P.get("echoes", 22))
        self.GR = float(P.get("growth", -0.40))
        self.step = float(P.get("step") or 0.0078 * pl.w * 1.0)
        self.swing = float(P.get("swing", 205.0))
        self.Lb = self.NE * self.step
        # one unit per row (the word is one unit)
        pts = []
        for ri in range(len(self.rows)):
            cs = [c for l in self.row_contours(ri, self.w_from) for c in l]
            pts.append(np.concatenate([pl.lift(c) for c in cs]))
        self.flow = LetterFlow(self, pts, self.Lb, rng, land=(0.45, 0.10), p_in=self.P_IN, p_out=self.P_OUT)
        self.centres = [np.array([0.0, 0.5 * (self.ycs[ri] - 0.5 * self.cap)]) for ri in range(len(self.rows))]
        for ri in range(len(self.rows)):
            self.centres[ri] = np.array([0.0, self.ycs[ri] - 0.5 * self.cap])
        # trailing angle (degrees, 0 = +u, counter-clockwise on screen) of the entry / exit streaks
        self.th_in = self._ang(-self.d_in)
        self.th_out = self._ang(-self.d_out)
        self.th_a = float(P.get("axis_from", -150.0))
        self.N = 0
        self._strand_cache = {}
        self.NS = [min(4 * len([1 for l in self.row_contours(ri, self.w_from)]), 28) for ri in range(len(self.rows))]
        self.N = int(sum(self.NS))

    def _ang(self, v):
        pl = self.placement
        a = math.degrees(math.atan2(-(v @ pl.v), v @ pl.u))
        return a

    def wght(self, t):
        tm = self.timing
        u = clamp((t - 0.40 * tm.form) / max(tm.form + tm.hold + 0.5 * tm.exit - 0.40 * tm.form, 1e-6))
        return self.w_from + (self.w_to - self.w_from) * eio(u, 2.0)

    @staticmethod
    def _lerp_ang(a, b, w):
        d = (b - a + 180.0) % 360.0 - 180.0
        return a + d * w

    # ------------------------------------------------------------ state
    def state(self, ri, t):
        tm, fl = self.timing, self.flow
        off, v, tau, entering = fl.at(ri, t)
        if entering:
            Ti = fl.Ti[ri]
            w = ssm(t, 0.60 * Ti, Ti + 0.30)
            th_trail = self.th_in
            s = ssm(t, 0.0, 0.60 * Ti)
        else:
            tau_ = max(tau, 0.0)
            w = 1.0 - ssm(tau_, 0.35, 0.85)
            th_trail = self.th_out
            s = 1.0 - ssm(tau_, 0.70, 1.0)
        sw = self.th_a + self.swing * eio(clamp(t / self.duration), 2.2)
        th = self._lerp_ang(th_trail, sw, w)
        return dict(off=off, v=v, w=w, th=th, s=s, gr=self.GR * w, entering=entering)

    def _axis(self, th):
        pl = self.placement
        a = math.radians(th)
        return math.cos(a) * pl.u - math.sin(a) * pl.v

    def copy_pts(self, ri, P, q, st):
        """local points P (n,2) of echo copy q of row ri -> world."""
        pl = self.placement
        c = self.centres[ri]
        sc = math.exp(st["gr"] * q / self.NE)
        a = self._axis(st["th"])
        W = pl.lift(c + (P - c) * sc) + a * (self.step * q) + st["off"]
        return W + pl.n * (0.12 * self.step * q)

    # ------------------------------------------------------------ bundle / frames
    def _strand_pts(self, ri, wg):
        k = (ri, round(wg, 0))
        if k not in self._strand_cache:
            letters = self.row_contours(ri, wg)
            pts = []
            for cs in letters:
                pts += list(corner_points([simplify(c, 0.6) for c in cs], kmax=4))
            idx = np.linspace(0, len(pts) - 1, self.NS[ri]).astype(int) if len(pts) else []
            self._strand_cache[k] = np.array([pts[i] for i in idx])
        return self._strand_cache[k]

    def bundle_at(self, t):
        K = 8
        Pm = np.zeros((self.N, K, 3))
        Dv = np.zeros((self.N, 3))
        v = np.zeros(self.N)
        wg = self.wght(t)
        entry = t < self.timing.form + self.timing.hold
        i = 0
        qs = np.linspace(0, self.NE, K)
        for ri in range(len(self.rows)):
            st = self.state(ri, t)
            for p in self._strand_pts(ri, wg):
                for k, q in enumerate(qs):
                    Pm[i, k] = self.copy_pts(ri, p[None], q, st)[0]
                Dv[i] = self.d_in if entry else self.d_out
                v[i] = st["v"]
                i += 1
        col = np.tile(np.array(self.palette.fg, float), (self.N, 1))
        return Bundle(Pm, Dv, col, 1.0, t, v, 0.0)

    def layers(self, t):
        pal, pl = self.palette, self.placement
        wg = self.wght(t)
        L = []
        for ri in range(len(self.rows)):
            st = self.state(ri, t)
            letters = self.row_contours(ri, wg)
            cs0 = [simplify(c, 0.3) for l in letters for c in l]
            if not cs0:
                continue
            col_line = mixc(pal.bg, pal.fg, st["s"])
            groups_all = None
            if st["s"] > 0.004:
                for q in range(self.NE, 0, -1):
                    polys = [self.copy_pts(ri, c, q, st) for c in cs0]
                    L.append(Layer(polys, pal.bg, "fill", bias=0.0, name="echo"))
                    L.append(Layer(polys, col_line, "line", 0.9, closed=True, bias=0.1))
            # front word, overlap-safe
            front = [l for l in letters]
            if st["s"] > 0.004:
                colf = mixc(pal.bg, pal.fg, st["s"])
                for cs in front:
                    for g in nz_groups(cs):
                        L.append(Layer([self.copy_pts(ri, cs[i], 0, st) for i in g], colf, "fill", bias=0.5))
            # strands (corner trails through the stack), fading as the word lands
            if st["w"] < 0.98:
                pols = []
                qs = np.linspace(0, self.NE, 8)
                for p in self._strand_pts(ri, wg):
                    pols.append(np.array([self.copy_pts(ri, p[None], q, st)[0] for q in qs]))
                L.append(Layer(pols, pal.fg, "line", 1.0 * (1.0 - st["w"]), closed=False, bias=1.0))
        return L
