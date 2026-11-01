"""P07 Desert Signal: piece-local anamorph mark volume (+ - |) rendered as type passes (matte / depth / beauty / line) for piece_runner.

A fixed 3D volume (flattened ellipsoid) of ~10k marks. Three words are anamorphic: from word k's reading eye the marks of that word
lie on rays through a screen grid inside the glyph mask (each at its own depth inside the volume), so they read as the word only from
that angle; from every other angle they are scattered through the cloud. The camera orbits the pivot (yaw sweep with slow creep through
each reading eye, locked to the song's chord changes). Marks are born over time (accumulating volume), pulse on the kick (beat grid),
and the Bitcount words morph their mark shape (ELSH-like: + -> dot) and length (ELXP-like) with the bars.

  python 15_fusion/pieces/P07_anamorph.py preview [t t t ...]      -> review/P07_ana_preview.png (stills of the passes)
  python 15_fusion/pieces/P07_anamorph.py render [--procs 3]       -> v002/local/pieces/P07/type/{matte,depth,line,beauty}/
  python 15_fusion/pieces/P07_anamorph.py register                 -> puts the finished type dir into state.json (run when no runner is active)
Depth pass: grey = plane value for compose.depth_occlude (near marks high, far marks low), 0 where there is no mark.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PIECE = ROOT / "15_fusion" / "v002" / "local" / "pieces" / "P07"
FONTS = ROOT / "mograph" / "fonts"
W, H = 1088, 1920
CEN = np.array([W / 2.0, 0.40 * H])
F, D = 1500.0, 1700.0                       # focal length (px), orbit radius
RXZ, RY = 720.0, 620.0                      # volume (ellipsoid semi axes)
FPS = 24
DUR = 14.6
NF = int(math.ceil(DUR * FPS))
PL_NEAR, PL_FAR = 0.92, 0.34                # depth-pass range (compose plane): nearest marks .. farthest marks
WORDS = [
    dict(lines=["THE", "SIGNAL"], font="Bytesized-Regular.ttf", loc=None, yaw=0.0, pitch=7.0, hit=4.31, pitch_px=12.0, cy=0.40, wfrac=0.84, morph=False),
    dict(lines=["IS"], font="Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf", loc=dict(wght=700, ELSH=0, ELXP=0), yaw=105.0, pitch=-6.0, hit=8.12, pitch_px=14.0,
         cy=0.40, wfrac=0.80, morph=True),
    dict(lines=["EVERY", "WHERE"], font="Bitcount[CRSV,ELSH,ELXP,slnt,wght].ttf", loc=dict(wght=780, ELSH=0, ELXP=0), yaw=210.0, pitch=9.0, hit=11.93,
         pitch_px=11.0, cy=0.40, wfrac=0.90, morph=True),
]
N_FILL = 12000
FILL_FLOOR = 0.30
HALO_PX = 34
CACHE = PIECE / "ana_cache.npz"
G = {}


# ------------------------------------------------------------------ camera
def vdir(yaw, pitch):
    """unit vector from the pivot to the eye (yaw about +y, 0 = +z; pitch up positive)."""
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp])


def basis(yaw, pitch):
    e = vdir(yaw, pitch)
    f = -e
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return e * D, r, u, f


def project(pos, eye, r, u, f):
    d = pos - eye
    z = d @ f
    sc = F / np.maximum(z, 1e-6)
    return np.stack([CEN[0] + (d @ r) * sc, CEN[1] - (d @ u) * sc], 1), z


def yaw_pitch_at(t):
    from scipy.interpolate import PchipInterpolator
    ky, kp, kt = [], [], []
    y0 = WORDS[0]["yaw"] - 78.0
    kt.append(0.0); ky.append(y0); kp.append(12.0)
    for i, w in enumerate(WORDS):
        a, b = (0.55, 0.55) if i < 2 else (0.7, 1.6)
        for dt, dy in ((-a, -3.6), (0.0, 0.0), (b, 4.0 if i < 2 else 6.5)):
            kt.append(w["hit"] + dt); ky.append(w["yaw"] + dy); kp.append(w["pitch"] + (0.0 if dt == 0 else 0.0))
    kt.append(DUR + 0.2); ky.append(WORDS[2]["yaw"] + 6.5 + 26.0); kp.append(4.0)
    yaw = float(PchipInterpolator(kt, ky)(min(max(t, 0.0), DUR)))
    pit = float(PchipInterpolator(kt, kp)(min(max(t, 0.0), DUR)))
    pit += 1.3 * math.sin(2 * math.pi * 0.11 * t)
    return math.radians(yaw), math.radians(pit)


def cam_at(t):
    return basis(*yaw_pitch_at(t))


# ------------------------------------------------------------------ glyph masks (screen space at the reading eye)
def font_for(w, size):
    ft = ImageFont.truetype(str(FONTS / w["font"]), size)
    if w["loc"]:
        axes = ft.get_variation_axes()
        names = [a["name"] if isinstance(a["name"], str) else a["name"].decode() for a in axes]
        vals = []
        for a, nm in zip(axes, names):
            key = {"Weight": "wght", "Element Shape": "ELSH", "Element Expansion": "ELXP", "Cursive": "CRSV", "Slant": "slnt"}.get(nm, nm)
            vals.append(float(w["loc"].get(key, a["default"])))
        ft.set_variation_by_axes(vals)
    return ft


def glyph_mask(w):
    """HxW bool mask of the word (lines stacked, centred at (W/2, cy*H)); size so the widest line is wfrac of the width."""
    lo, hi = 40, 1400
    for _ in range(14):
        mid = (lo + hi) // 2
        ft = font_for(w, mid)
        mw = max(ft.getlength(l) for l in w["lines"])
        if mw > w["wfrac"] * W:
            hi = mid
        else:
            lo = mid
    ft = font_for(w, lo)
    asc, desc = ft.getmetrics()
    lh = int((asc + desc) * 0.82)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    n = len(w["lines"])
    total = lh * (n - 1) + (asc + desc)
    y0 = w["cy"] * H - total / 2.0
    for i, l in enumerate(w["lines"]):
        lw = ft.getlength(l)
        d.text(((W - lw) / 2.0, y0 + i * lh), l, fill=255, font=ft)
    m = np.asarray(img) > 128
    if w["loc"]:
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return m, lo


# ------------------------------------------------------------------ geometry
def ray_interval(E, dirs, shrink=0.97):
    sc = 1.0 / (np.array([RXZ, RY, RXZ]) * shrink)
    a = ((dirs * sc) ** 2).sum(1)
    b = 2 * ((E * sc) * (dirs * sc)).sum(1)
    c = ((E * sc) ** 2).sum() - 1.0
    disc = b * b - 4 * a * c
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0))
    return (-b - sq) / (2 * a), (-b + sq) / (2 * a), ok


def noise_density(pos):
    from scipy.ndimage import gaussian_filter, map_coordinates
    rng = np.random.RandomState(7)
    n = 44
    g = gaussian_filter(rng.randn(n, n, n), 3.4, mode="wrap")
    g /= g.std()
    u = pos / np.array([RXZ, RY, RXZ])
    r2 = (u ** 2).sum(1)
    idx = (np.clip(u, -1, 1) + 1) * 0.5 * (n - 1)
    nz = map_coordinates(g, idx.T, order=1, mode="nearest")
    return np.clip(0.62 + 0.55 * nz, 0.05, 1.25) * np.clip(1.0 - r2, 0.0, 1.0) ** 0.30


def build():
    if G.get("ready"):
        return G
    rng = np.random.RandomState(11)
    eyes = [basis(math.radians(w["yaw"]), math.radians(w["pitch"])) for w in WORDS]
    masks, halos, sizes = [], [], []
    for w in WORDS:
        m, sz = glyph_mask(w)
        masks.append(m)
        sizes.append(sz)
        halos.append(cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HALO_PX + 1, 2 * HALO_PX + 1))).astype(bool))
    pos, kind, wid, Lw, bri, acc, birth, morph_r = [], [], [], [], [], [], [], []

    def inside(xy, mk):
        ix, iy = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        out = np.zeros(len(xy), bool)
        out[ok] = mk[iy[ok], ix[ok]]
        return out

    for k, w in enumerate(WORDS):
        E, r, u, f = eyes[k]
        p = w["pitch_px"]
        ys, xs = np.mgrid[int(p / 2):H:int(round(p)), int(p / 2):W:int(round(p))]
        sel = masks[k][ys, xs]
        sx = xs[sel].astype(float) + rng.uniform(-0.07, 0.07, sel.sum()) * p
        sy = ys[sel].astype(float) + rng.uniform(-0.07, 0.07, sel.sum()) * p
        n = len(sx)
        dirs = r[None, :] * ((sx - CEN[0]) / F)[:, None] - u[None, :] * ((sy - CEN[1]) / F)[:, None] + f[None, :]
        Z1, Z2, ok = ray_interval(E, dirs)
        # K candidate depths per mark, picked to keep the other words' halos clear
        K = 28
        cand = Z1[:, None] + (Z2 - Z1)[:, None] * rng.uniform(0.08, 0.92, (n, K))
        pen = np.zeros((n, K))
        for j in range(3):
            if j == k:
                continue
            Ej, rj, uj, fj = eyes[j]
            for c in range(K):
                P = E[None, :] + cand[:, c:c + 1] * dirs
                xy, _ = project(P, Ej, rj, uj, fj)
                pen[:, c] += inside(xy, halos[j]) * 1.0 + inside(xy, masks[j]) * 0.4
        best = np.argmin(pen + rng.uniform(0, 0.05, pen.shape), axis=1)
        z = cand[np.arange(n), best]
        P = E[None, :] + z[:, None] * dirs
        P = P[ok]
        z = z[ok]
        n = len(P)
        pos.append(P)
        kind.append(rng.choice([1, 2, 3], n, p=[0.5, 0.25, 0.25]))
        wid.append(np.full(n, k))
        Lw.append(0.80 * p * z / F)
        bri.append(np.full(n, 0.72))
        acc.append(rng.rand(n) < 0.0)
        birth.append(rng.uniform(0.2, max(w["hit"] - 1.7, 1.0), n))
        morph_r.append(rng.rand(n))
    # filler marks: lumpy density in the ellipsoid, kept out of the reading halos
    cnt = 0
    fp = []
    while cnt < N_FILL:
        c = rng.uniform(-1, 1, (N_FILL * 3, 3)) * np.array([RXZ, RY, RXZ])
        c = c[((c / np.array([RXZ, RY, RXZ])) ** 2).sum(1) < 1.0]
        c = c[rng.rand(len(c)) < noise_density(c) / 1.25]
        keep = np.ones(len(c), bool)
        for j in range(3):
            Ej, rj, uj, fj = eyes[j]
            xy, _ = project(c, Ej, rj, uj, fj)
            inh = inside(xy, halos[j])
            keep &= ~inh | (rng.rand(len(c)) < FILL_FLOOR)
        fp.append(c[keep])
        cnt += keep.sum()
    fp = np.concatenate(fp)[:N_FILL]
    n = len(fp)
    pos.append(fp)
    kind.append(rng.choice([0, 1, 2, 3], n, p=[0.04, 0.48, 0.24, 0.24]))
    wid.append(np.full(n, -1))
    Lw.append(rng.uniform(0.55, 1.15, n) * 12.5 * D / F)
    bri.append(rng.uniform(0.25, 0.6, n))
    acc.append(rng.rand(n) < 0.07)
    birth.append(rng.uniform(0.0, 11.5, n) ** 1.0)
    morph_r.append(rng.rand(n))
    G.update(pos=np.concatenate(pos), kind=np.concatenate(kind), wid=np.concatenate(wid), Lw=np.concatenate(Lw), bri=np.concatenate(bri),
             acc=np.concatenate(acc), birth=np.concatenate(birth), mr=np.concatenate(morph_r), eyes=eyes, masks=masks, halos=halos, sizes=sizes,
             ready=True)
    return G


# ------------------------------------------------------------------ per-frame state
ANA = None


def ana():
    global ANA
    if ANA is None:
        ANA = json.loads((PIECE / "analysis.json").read_text(encoding="utf-8"))
    return ANA


def kick(t):
    b = np.array(ana()["beats"])
    prev = b[b <= t + 1e-6]
    if not len(prev):
        return 0.0
    return float(math.exp(-(t - prev[-1]) * 9.0))


def focus_of(eye_f, k):
    ef = G["eyes"][k][3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(eye_f @ ef)))))
    return math.exp(-(ang / 6.5) ** 2)


def draw_frame(i):
    build()
    t = i / FPS
    E, r, u, f = cam_at(t)
    pos, kind, wid, Lw, bri, acc, birth, mr = (G[k] for k in ("pos", "kind", "wid", "Lw", "bri", "acc", "birth", "mr"))
    xy, z = project(pos, E, r, u, f)
    foc = np.array([focus_of(f, k) for k in range(3)])
    fm = float(foc.max())
    pulse = kick(t)
    # bar phases for the Bitcount-style morphs
    bar = float(ana()["bar_len"])
    ph = (t / bar) % 1.0
    elsh = 0.5 - 0.5 * math.cos(2 * math.pi * ph)                                  # 0 -> 1 -> 0 each bar
    elxp = 0.5 + 0.5 * math.sin(2 * math.pi * (t / (2 * bar)))
    age = np.clip((t - birth) / 0.7, 0.0, 1.0)
    age = age * age * (3 - 2 * age)
    fz = np.clip((z - (D - RXZ)) / (2 * RXZ), 0.0, 1.0)
    fog = 1.0 - 0.55 * fz
    wi = np.maximum(wid, 0)
    oth = np.where(wid >= 0, np.max(np.where(np.arange(3)[None, :] == wid[:, None], 0.0, foc[None, :]), axis=1), fm)
    wb = np.where(wid >= 0, 1.0 + 0.55 * foc[wi], 1.0) * (1.0 - 0.68 * oth)
    b = np.clip(bri * fog * wb * age, 0.0, 1.2)
    b = b * (0.85 + 0.55 * pulse)
    L = Lw * F / np.maximum(z, 1e-6) * (1.0 + 0.40 * pulse) * (1.0 + 0.0 * elxp)
    is_bc = np.isin(wid, [1, 2])
    L = np.where(is_bc, L * (0.8 + 0.5 * elxp), L)
    L = np.where(acc, L * 1.25, L)
    L = np.minimum(L, 24.0)                                           # near marks never balloon into flakes
    b = b * np.clip((z - 650.0) / 450.0, 0.0, 1.0)                    # and fade out as they pass the lens
    m = 30.0
    vis = (z > 650) & (xy[:, 0] > -m) & (xy[:, 0] < W + m) & (xy[:, 1] > -m) & (xy[:, 1] < H + m) & (b > 0.02) & (age > 0.0)
    idx = np.nonzero(vis)[0]
    idx = idx[np.argsort(-z[idx])]
    SH = 3
    K = float(1 << SH)
    beauty = np.zeros((H, W, 3), np.uint8)
    matte = np.zeros((H, W), np.uint8)
    depth = np.zeros((H, W), np.uint8)
    u_d = np.clip((z - (D - RXZ)) / (2 * RXZ), 0.0, 1.0)
    pl = PL_NEAR + (PL_FAR - PL_NEAR) * u_d
    bone = np.array([255.0, 238.0, 210.0])
    ac = np.array([255.0, 120.0, 60.0])
    col = np.where(acc[:, None], ac[None, :], bone[None, :]) * np.clip(b, 0, 1.0)[:, None]
    col = np.clip(col, 0, 255)[:, ::-1].astype(int)
    lw = np.clip(0.14 * L + 1.2, 1.4, 3.2)
    AA = cv2.LINE_AA
    dot_thr = elsh
    for j in idx:
        x, y, h = xy[j, 0] * K, xy[j, 1] * K, 0.5 * L[j] * K
        xi, yi, hi = int(x), int(y), int(h)
        tw = max(1, int(round(lw[j])))
        c = (int(col[j, 0]), int(col[j, 1]), int(col[j, 2]))
        a = int(min(255, 255 * min(1.0, b[j] * 1.1)))
        d = int(255 * pl[j])
        kd = kind[j]
        if kd == 0 or (is_bc[j] and mr[j] < dot_thr * 0.8):
            rad = max(2, int(h * 0.5))
            cv2.circle(beauty, (xi, yi), rad, c, -1, AA, SH)
            cv2.circle(matte, (xi, yi), rad, a, -1, AA, SH)
            cv2.circle(depth, (xi, yi), rad, d, -1, AA, SH)
            continue
        if kd != 2:
            p0, p1 = (xi - hi, yi), (xi + hi, yi)
            cv2.line(beauty, p0, p1, c, tw, AA, SH)
            cv2.line(matte, p0, p1, a, tw, AA, SH)
            cv2.line(depth, p0, p1, d, tw, AA, SH)
        if kd != 3:
            p0, p1 = (xi, yi - hi), (xi, yi + hi)
            cv2.line(beauty, p0, p1, c, tw, AA, SH)
            cv2.line(matte, p0, p1, a, tw, AA, SH)
            cv2.line(depth, p0, p1, d, tw, AA, SH)
    depth = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    return beauty, matte, depth


def _job(args):
    i, out = args
    b, m, d = draw_frame(i)
    out = Path(out)
    cv2.imwrite(str(out / "beauty" / f"beauty_{i:05d}.png"), b, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(out / "matte" / f"matte_{i:05d}.png"), m, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(out / "line" / f"line_{i:05d}.png"), m, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(out / "depth" / f"depth_{i:05d}.png"), d, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return i


def legibility(t):
    build()
    E, r, u, f = cam_at(t)
    xy, z = project(G["pos"], E, r, u, f)
    ix, iy = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
    ins = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    out = []
    for k in range(3):
        sel = G["wid"] == k
        ok = np.zeros(sel.sum(), bool)
        ii = ins[sel]
        ok[ii] = G["halos"][k][iy[sel][ii], ix[sel][ii]]
        out.append(round(float(ok.mean()), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preview", "render", "register", "report"])
    ap.add_argument("times", nargs="*", type=float)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "preview":
        ts = a.times or [1.5, 3.0, 4.31, 6.2, 8.12, 10.0, 11.93, 13.5]
        tiles = []
        for t in ts:
            b, m, d = draw_frame(int(round(t * FPS)))
            t1 = cv2.resize(b, (272, 480), interpolation=cv2.INTER_AREA)
            t2 = cv2.resize(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR), (272, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(t1, f"{t:.2f}s {legibility(t)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 230, 60), 1, cv2.LINE_AA)
            tiles.append(np.vstack([t1, t2]))
        sheet = np.hstack(tiles[:8])
        p = ROOT / "15_fusion" / "v002" / "review" / "P07_ana_preview.png"
        cv2.imwrite(str(p), sheet)
        print("wrote", p, "marks", len(G["pos"]), "words", [int((G["wid"] == k).sum()) for k in range(3)], "font px", G["sizes"])
    elif a.cmd == "report":
        build()
        for t in np.arange(0, DUR, 0.5):
            print(f"{t:5.2f}s yaw {math.degrees(yaw_pitch_at(t)[0]):7.1f} legibility {legibility(t)}")
    elif a.cmd == "render":
        import multiprocessing as mp
        out = PIECE / "type"
        for p in ("beauty", "matte", "line", "depth"):
            (out / p).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        build()
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.procs) as pool:
            for n, _ in enumerate(pool.imap_unordered(_job, [(i, str(out)) for i in range(NF)], chunksize=4)):
                if n % 50 == 0:
                    print(f"  {n}/{NF} {time.time() - t0:.0f}s", flush=True)
        man = dict(frames=[0, NF], passes=["matte", "depth", "line", "beauty"], nf=NF, info=dict(words=[], notes=["P07 anamorph volume"]),
                   timing=dict(wall_s=time.time() - t0, procs=a.procs))
        (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        print("rendered", NF, "frames in", round(time.time() - t0), "s")
    elif a.cmd == "register":
        sp = PIECE / "state.json"
        st = json.loads(sp.read_text(encoding="utf-8"))
        out = PIECE / "type"
        st["items"]["type"] = dict(out=dict(paths=[str(out / "manifest.json")], dir=str(out), passes=["matte", "depth", "line", "beauty"], nf=NF,
                                            sheet="", info=[]), wall_s=float(json.loads((out / "manifest.json").read_text())["timing"]["wall_s"]),
                                   gpu_s=0.0, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        sp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print("registered type item")


if __name__ == "__main__":
    main()
