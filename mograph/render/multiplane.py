"""mograph.render.multiplane: the plate as depth cards in the type world (v005, 2026-10-27).

The footage frame is cut into 3-5 cards from its depth map and subject matte: background bands (far to near), the SUBJECT on its own card
and a FOREGROUND card (what is nearer than the subject: the floor at her feet). Every card is a fronto-parallel plane in the same world as the
type (cam3d: x right, y down, +z away), scaled so that from the reference camera (eye (W/2, H/2, -ref), focal F) it reproduces the original
frame exactly. A chain camera that pushes through, orbits or grazes then moves through the type AND the footage together: the cards parallax
against each other, the type sits between them (or crosses them: the type is composited per pixel by depth, not by slot).

    mp = MultiPlane(data, world=(540, 960), size=(360, 640), cards=4)          # data: contour_ports.PlateData (plate + depth + matte)
    cfg = plan({"cards": 4})                                                     # card depths, slot -> z for the words
    img = mp.render(frame, cam, type_rgb, type_a, type_depth)                    # float HxWx3 (torch on the box GPU)

Decomposition (per frame, numpy / cv2 inpaint at 1/4 size): class of a pixel = subject (matte), foreground (non-subject and nearer than the
subject's median depth + margin), else background band by depth quantiles fixed from the first frames (so a band keeps its identity). A card has
alpha 1 inside its class (soft edges) and is extended behind the nearer classes by `reveal` px with inpainted colour, so the parallax reveals
plausible picture, not holes; the back card is opaque everywhere (mirrored beyond the frame: grid_sample 'reflection'). Warp: per card the
camera ray of every output pixel meets the plane z = z_k (torch, one homography-equivalent grid per card), the card is sampled bilinearly there.

Spec hook (planner): "camera": {"chain": <camera module id | list of {module, word|window, ..} | "auto">, "plate": "multiplane", "cards": 4,
"params": {..camera params}, "z": [per-card depth, far to near; default plan()], "ref": 1430}; a word with "slot": k sits between card k and k+1
(k = -1: behind all, cards-1: in front of all); `apply_slots(words, cfg)` writes the placement z. Only numpy / cv2 / torch."""
from __future__ import annotations

import math

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as TF
except Exception:                                            # pragma: no cover
    torch = None


# ============================================================================================ plan
def plan(spec=None, F=1000.0):
    """card depths (far -> near) and the z of every slot: {"z": [..], "slots": [..], "ref": reference camera distance, "cards": n}.
    Defaults: 4 cards at +260 (back wall), +130 (mid), 0 (subject), -120 (foreground), all shifted by spec["subject_z"] (how far the subject card is
    from the type's home plane z = 0: the gallery camera moves are sized by the words, ~1400 px, so ~1000 lets a push_through end in front of her); the reference camera sits F / 0.7 away (a word
    framed at fill 0.7 and the plate then fill the frame together)."""
    spec = spec or {}
    n = max(int(spec.get("cards", 4)), 3)
    z = spec.get("z")
    if z is None:
        far, near = 260.0, -120.0
        z = [far + (near - far) * i / max(n - 1, 1) for i in range(n)]
        # keep the subject card at z = 0 whatever the count
        if n >= 3:
            z[n - 2] = 0.0
            for i in range(n - 2):
                z[i] = far + (0.0 - far) * i / max(n - 2, 1)
    z = [float(v) + float(spec.get("subject_z", 0.0)) for v in z] if spec.get("z") is None else [float(v) for v in z]
    slots = [z[0] + (z[0] - z[1]) * 0.6] + [0.5 * (z[i] + z[i + 1]) for i in range(len(z) - 1)] + [z[-1] - (z[-2] - z[-1]) * 0.9]
    return {"cards": len(z), "z": z, "slots": slots, "ref": float(spec.get("ref") or F / 0.7), "subject": n - 2 if n >= 3 else 0}


def apply_slots(words, cfg):
    """word specs with "slot": k get placement.z = slot depth (slot -1 = behind the back card ... cards - 1 = in front of the foreground)."""
    out = []
    for w in words:
        w = dict(w)
        if w.get("slot") is not None:
            k = int(w["slot"]) + 1
            pl = dict(w.get("placement") or {})
            pl["z"] = float(cfg["slots"][min(max(k, 0), len(cfg["slots"]) - 1)])
            w["placement"] = pl
        out.append(w)
    return out


def camera_spec(cam):
    """the planner's "camera": {"chain", "plate": "multiplane", "cards", ..} -> the plain spec camera keys (camera, camera_params) that
    tools.fusion.spec.build_camera understands, or None when the camera is not a multiplane one."""
    if isinstance(cam, dict) and cam.get("plate") == "multiplane":
        return {"camera": cam.get("chain", "auto"), "camera_params": cam.get("params") or {}}
    return None


# ============================================================================================ decomposition (numpy / cv2)
def _ss(x, a, b):
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _otsu(vals):
    """threshold (0..1) splitting 8-bit-quantised values into two classes with maximal between-class variance."""
    v = np.clip(np.round(vals * 255.0), 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def band_edges(data, n_bands, frames=(0,), margin=0.04, blur=3.0):
    """depth thresholds (0..1, near = 1) that split the non-subject background into `n_bands` bands: recursive Otsu on the blurred depth of
    the pixels behind the subject (the wall / the far floor / ...), from a few frames, so a band keeps its identity over the shot."""
    fr = []
    for f in frames:
        f = min(f, data.n - 1)
        fr.append((data.depth[f], data.matte[f] if data.matte is not None else None))
    return band_edges_frames(fr, n_bands, margin, blur)


def band_edges_frames(frames, n_bands, margin=0.04, blur=3.0):
    """band_edges on [(depth_u8, matte_u8 | None)] frames."""
    ds = []
    for dep, mat in frames:
        d = cv2.GaussianBlur(dep.astype(np.float32) / 255.0, (0, 0), blur)
        m = mat > 128 if mat is not None else np.zeros(d.shape, bool)
        ds_med = np.median(d[m]) if m.any() else float(np.percentile(d, 60))
        ds.append(d[(~m) & (d < ds_med + margin)])
    d = np.concatenate([x.reshape(-1) for x in ds])
    if n_bands <= 1 or d.size < 16:
        return []
    groups, edges = [d], []
    while len(edges) < n_bands - 1:
        k = max(range(len(groups)), key=lambda i: groups[i].var() * len(groups[i]) if len(groups[i]) > 16 else -1)
        g = groups.pop(k)
        t = _otsu(g)
        if not (g.min() < t < g.max()):
            t = float(np.median(g))
        edges.append(t)
        groups += [g[g < t], g[g >= t]]
    return sorted(edges)


def _fill(rgb, hole, support=None, ring=10, sigmas=(5.0, 14.0, 36.0, 90.0)):
    """colour for the pixels in `hole` (bool HxW): normalised convolution (push-pull) of the picture OUTSIDE the hole and a `ring` of px around it
    (the ring holds the subject's fringe, hair and shadow: copying it gives a dark halo when the cards slide apart), small to large sigma;
    the rest keeps its colour. support (bool HxW, optional): only these pixels are copied from (the card's own class: the floor behind a dancer
    is filled from floor, not from the dark wall above her)."""
    if not hole.any():
        return rgb
    src = ~(cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(ring) + 1,) * 2)) > 0)
    if support is not None and (src & support).sum() > 64:
        src = src & support
    img = rgb.astype(np.float32)
    if not src.any():
        src = ~hole
    if not src.any():
        out = rgb.copy()
        out[hole] = rgb.reshape(-1, 3).mean(0).astype(rgb.dtype)
        return out
    sm = src.astype(np.float32)
    est = np.zeros_like(img)
    got = np.zeros(hole.shape, bool)
    for sg in sigmas:
        den = cv2.GaussianBlur(sm, (0, 0), sg)
        num = cv2.GaussianBlur(img * sm[..., None], (0, 0), sg)
        ok = (den > 0.04) & ~got
        est[ok] = num[ok] / den[ok][:, None]
        got |= ok
        if got[hole].all():
            break
    if not got[hole].all():
        est[~got] = img[src].mean(0)
    out = rgb.copy()
    out[hole] = np.clip(est[hole], 0, 255).astype(rgb.dtype)
    return out


def decompose(plate, depth, matte, edges, n_cards, margin=0.04, reveal=5, soft=0.035):
    """plate (HxWx3 u8 RGB), depth (HxW u8, near = white), matte (HxW u8 or None) -> float32 (N, H, W, 4) straight-alpha cards, far to near:
    N-2 background bands, the subject, the foreground (N = 2: subject + foreground; N = 3: back, subject, foreground ...)."""
    h, w = depth.shape
    d = cv2.GaussianBlur(depth.astype(np.float32) / 255.0, (0, 0), 3.0)         # no depth noise rings at the band edges
    m = (matte.astype(np.float32) / 255.0) if matte is not None else np.zeros((h, w), np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    sub = m > 0.5
    d_s = float(np.median(cv2.GaussianBlur(depth.astype(np.float32) / 255.0, (0, 0), 1.0)[sub])) if sub.any() else 1.0
    nb = max(n_cards - 2, 0)
    # soft class masks, far -> near (backgrounds, subject, foreground), non-subject classes carry (1 - m)
    thr = list(edges[:max(nb - 1, 0)]) + [d_s + margin]               # band k is d < thr[k] (and >= thr[k-1])
    cls = []
    lo = -1.0
    for k in range(nb):
        t = thr[k] if k < len(thr) else d_s + margin
        a = (1.0 - _ss(d, t - soft, t + soft)) * (_ss(d, lo - soft, lo + soft) if lo > 0 else 1.0)
        cls.append(a * (1.0 - m))
        lo = t
    cls.append(_ss(m, 0.4, 0.85))                                      # the subject card is cut a little inside the matte: no wall-coloured fringe
    fg = _ss(d, d_s + margin - soft, d_s + margin + soft) * (1.0 - m)
    cls.append(fg)
    if nb == 0:                                                       # subject + foreground only: the rest is the back card
        cls = [np.clip(1.0 - m - fg, 0, 1), m, fg]
    cls = [cv2.GaussianBlur(np.clip(c, 0, 1).astype(np.float32), (0, 0), 1.0) for c in cls]
    N = len(cls)
    cards = np.zeros((N, h, w, 4), np.float32)
    rgb = plate
    cover = [None] * N
    acc = np.zeros((h, w), np.float32)
    for i in range(N - 1, -1, -1):                                    # near -> far: what is in front of class i
        cover[i] = acc.copy()
        acc = np.maximum(acc, cls[i])
    for i in range(N):
        a = cls[i]
        if i == 0:
            alpha = np.ones((h, w), np.float32)                      # back card: opaque everywhere
        else:
            hid = (cover[i] > 0.5).astype(np.uint8)                   # what the nearer classes hide: this card continues behind them
            if reveal > 0:
                hid = cv2.dilate(hid, np.ones((2 * int(reveal) + 1,) * 2, np.uint8))
            alpha = np.clip(np.maximum(a, cv2.GaussianBlur(hid.astype(np.float32), (0, 0), 1.0)), 0, 1)
        hole = (a < 0.5) & (alpha > 0.05)
        col = _fill(rgb, hole & ~(cls[N - 1] > 0.5), a > 0.5) if i < N - 1 else _fill(rgb, hole, a > 0.5)
        if i < N - 1:                                                  # what the foreground hides is more of the same floor: keep the picture
            keep = hole & (cls[N - 1] > 0.5)
            col[keep] = rgb[keep]
        cards[i, ..., :3] = col.astype(np.float32) / 255.0
        cards[i, ..., 3] = alpha
    return cards


# ============================================================================================ render (torch)
class _CardCam:
    """the camera one card is seen by: the chain's eye with a (possibly turned back) rotation R (rows right, down, forward)."""

    def __init__(self, eye, R, F):
        self.eye, self.R, self.F = np.asarray(eye, float), np.asarray(R, float), float(F)
        self.f = self.R[2]


class MultiPlane:
    """per-frame cards + the camera warp / composite. `world` = (w, h) of the frame plane, `size` = output (w, h) px (the plate is resampled
    to it), `device` cuda when there is one with room.
    max_mag: no card is ever shown more than this much larger than its source pixels (1.35). The camera dives through the TYPE; for every card
    the eye is held back along the view axis so that the card's magnification stays within the budget, so the cards only parallax (sideways /
    orbit motion is untouched; the nearest card is held hardest). pad: the cards are built on a canvas `pad` larger on every side (reflected
    picture), so a sideways / orbit move does not run out of card; `edge_report()` says how much of the frame still sampled beyond it."""

    def __init__(self, data, world, size, cards=4, z=None, ref=None, F=1000.0, reveal=5, device=None, edge_frames=(0, 30, 60),
                 max_mag=1.35, pad=0.15, min_mag=None):
        if torch is None:
            raise RuntimeError("multiplane needs torch")
        self.data, self.world, self.size, self.F = data, tuple(world), tuple(size), float(F)        # data None: frames come in per call (compose op)
        self.cfg = plan({"cards": cards, "z": z, "ref": ref}, F)
        self.n, self.z, self.ref = self.cfg["cards"], self.cfg["z"], self.cfg["ref"]
        self.reveal, self.max_mag, self.pad = reveal, float(max_mag), float(pad)
        self.min_mag = float(min_mag) if min_mag else 1.05 / (1.0 + 2.0 * self.pad)       # never smaller than the overscanned canvas covers
        if device is None:                                   # a shared box: use the GPU only when it has room for the cards
            ok = torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3 * 2 ** 30
            device = "cuda" if ok else "cpu"
        self.dev = torch.device(device)
        self.edges = (band_edges(data, max(self.n - 2, 1), edge_frames) if self.n > 2 else []) if data is not None else None
        w, h = self.size
        self.px, self.py = int(round(w * self.pad)), int(round(h * self.pad))
        self.cw, self.ch = w + 2 * self.px, h + 2 * self.py                            # card canvas size
        k = w / float(self.world[0])
        yy, xx = torch.meshgrid(torch.arange(h, device=self.dev, dtype=torch.float32), torch.arange(w, device=self.dev, dtype=torch.float32), indexing="ij")
        self.dx = (xx + 0.5 - w / 2.0) / (self.F * k)                  # camera-space ray (x / z, y / z) of every output pixel
        self.dy = (yy + 0.5 - h / 2.0) / (self.F * k)
        self._cache = {}
        self.stats = dict(frames=0, over_max=0.0, mag_max=0.0, held=0)

    # ---- cards
    def cards_at(self, f):
        f = int(min(max(f, 0), self.data.n - 1))
        if f in self._cache:
            return self._cache[f]
        t = self.cards_from(self.data.plate[f], self.data.depth[f], self.data.matte[f] if self.data.matte is not None else None)
        if len(self._cache) > 40:
            self._cache.pop(next(iter(self._cache)))
        self._cache[f] = t
        return t

    def cards_from(self, plate, depth, matte):
        """the cards of one frame from arrays (plate HxWx3 u8, depth HxW u8 near = white, matte HxW u8 | None) -> (N, 4, Hp, Wp) tensor."""
        w, h = self.size
        pl, dp, mt = plate, depth, matte
        if (pl.shape[1], pl.shape[0]) != (w, h):
            pl = cv2.resize(pl, (w, h), interpolation=cv2.INTER_AREA)
            dp = cv2.resize(dp, (w, h), interpolation=cv2.INTER_AREA)
            mt = cv2.resize(mt, (w, h), interpolation=cv2.INTER_AREA) if mt is not None else None
        if self.edges is None:                                  # no clip to look at: the bands come from the first frame seen
            self.edges = band_edges_frames([(dp, mt)], max(self.n - 2, 1)) if self.n > 2 else []
        px, py = self.px, self.py
        pad = lambda a: cv2.copyMakeBorder(a, py, py, px, px, cv2.BORDER_REFLECT_101)
        c = decompose(pad(pl), pad(dp), pad(mt) if mt is not None else None, self.edges, self.n, reveal=self.reveal)
        return torch.from_numpy(c).to(self.dev).permute(0, 3, 1, 2).contiguous()          # (N, 4, Hp, Wp)

    # ---- warp
    def _rays(self, cam):
        R = torch.as_tensor(cam.R, dtype=torch.float32, device=self.dev)                # world -> camera rows (r, d, f)
        eye = torch.as_tensor(cam.eye, dtype=torch.float32, device=self.dev)
        kF = cam.F / self.F                                                              # a zoomed camera (F differs from the nominal)
        dc = torch.stack([self.dx / kF, self.dy / kF, torch.ones_like(self.dx)], -1)    # (H, W, 3)
        return eye, dc @ R                                                               # world direction = R^T d  (row-vector form)

    def depth_axis(self, cam, zk):
        """depth of the card plane z = zk along the view axis (negative: the eye is past it)."""
        fz = float(cam.f[2])
        return (zk - float(cam.eye[2])) / fz if abs(fz) > 1e-3 else 1e9

    def hold_back(self, cam, zk):
        """how far (world units) the eye is moved back along the view axis for the card at z = zk so that its magnification stays within
        [min_mag, max_mag]: > 0 pulls the eye back (the camera dives close), < 0 pushes it in (the camera is far, the overscan would run out)."""
        if abs(float(cam.f[2])) < 0.2:
            return 0.0
        base = (self.ref + zk) * (cam.F / self.F)                                        # depth with magnification 1
        t0 = self.depth_axis(cam, zk)
        if t0 < base / self.max_mag:
            return base / self.max_mag - t0
        if t0 > base / self.min_mag:
            return base / self.min_mag - t0
        return 0.0

    def common_hold(self, cam):
        """the one hold-back (world units along the view axis) that every card is seen with: the largest any card needs to stay within max_mag
        (the nearest card sits at the budget, the rest a little below), else the largest push-in the overscan needs."""
        needs = [self.hold_back(cam, zk) for zk in self.z]
        return max(needs) if max(needs) > 0 else (min(needs) if min(needs) < 0 else 0.0)

    def magnification(self, cam, zk):
        """how many output px one source px of the card covers (1 at the reference camera) after the common hold-back."""
        t0 = self.depth_axis(cam, zk) + self.common_hold(cam)
        return (self.ref + zk) * (cam.F / self.F) / max(t0, 1e-3)

    def _card_rays(self, R, kF=1.0):
        """world ray directions (H, W, 3) of every output pixel for a camera with world->camera rows R (numpy 3x3), focal ratio kF."""
        Rt = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        dc = torch.stack([self.dx / kF, self.dy / kF, torch.ones_like(self.dx)], -1)
        return dc @ Rt

    @staticmethod
    def _turned(Rt, lam):
        """the camera rotation Rt turned back toward the card normal (looking straight along +z) by the share lam (geodesic)."""
        from scipy.spatial.transform import Rotation as Rot
        return Rot.from_rotvec(Rot.from_matrix(Rt).as_rotvec() * (1.0 - lam)).as_matrix() if lam > 0 else Rt

    def _card_view(self, cam, zk, lam, hb):
        """the card at z = zk seen by the camera turned back by lam and held back by hb along its view axis: -> (camera, hb, eye, ray directions)."""
        R = self._turned(np.asarray(cam.R, float), lam)
        cc = _CardCam(cam.eye, R, cam.F)
        eye = torch.as_tensor(cam.eye, dtype=torch.float32, device=self.dev)
        f = torch.as_tensor(cc.f, dtype=torch.float32, device=self.dev)
        e = eye - f * hb if hb != 0 else eye
        dw = self._card_rays(R, cam.F / self.F) if lam > 0 else None
        return cc, hb, e, dw

    def warp(self, cam):
        """-> grids (N, H, W, 2) into the padded card canvases, depth (N, H, W) = true camera depth of each card plane (for ordering against the
        type), valid (N, H, W): where the (held-back) camera rays meet every card plane.
        Each card is seen by its own camera: the chain's eye held back along the view axis (magnification budget) and, when the look direction
        turns further than the card's overscanned canvas can follow, the view direction turned back toward the card normal by the smallest share
        that keeps the frustum on the canvas (a smooth function of the camera: the turn is eased by the camera's own motion). A card is never
        rescaled by its own: its scale is a function of its depth and the held-back distance only."""
        eye, dw = self._rays(cam)
        cx, cy = self.world[0] / 2.0, self.world[1] / 2.0
        Wp, Hp = self.cw / self.size[0], self.ch / self.size[1]                         # canvas size in frame units (1 + 2 pad)
        grids, ts, valid = [], [], []
        dz = dw[..., 2]
        dzs = torch.where(dz.abs() < 1e-6, torch.full_like(dz, 1e-6), dz)
        info = dict(turn=[], hold=[], mag=[])
        # ONE camera for all the cards (a consistent scene seen from a held-back, eased-turn eye): the hold-back is the largest any card needs
        # (the nearest card sits at the magnification budget, the others a little below it); when only the overscan needs help, the largest push-in.
        hold_cache = {}

        def hold_for(lam):
            """the common hold-back measured along the view axis turned back by lam."""
            if lam not in hold_cache:
                cc0 = _CardCam(cam.eye, self._turned(np.asarray(cam.R, float), lam), cam.F)
                needs = [self.hold_back(cc0, zk) for zk in self.z]
                hold_cache[lam] = max(needs) if max(needs) > 0 else (min(needs) if min(needs) < 0 else 0.0)
            return hold_cache[lam]

        def uv(zk, lam):
            cc, hb_, e, dwk = self._card_view(cam, zk, lam, hold_for(lam))
            dwk = dw if dwk is None else dwk
            dzk = dwk[..., 2]
            dzks = torch.where(dzk.abs() < 1e-6, torch.full_like(dzk, 1e-6), dzk)
            t = (zk - e[2]) / dzks
            px = e[0] + t * dwk[..., 0]
            py = e[1] + t * dwk[..., 1]
            s = self.F / (self.ref + zk)
            u = (cx + (px - cx) * s) / self.world[0]
            v = (cy + (py - cy) * s) / self.world[1]
            return (u + self.pad) / Wp, (v + self.pad) / Hp, (t > 6.0) & (dzk > 1e-6), hb_, cc, t

        def width(un_, vn_):
            return max(float(un_.max() - un_.min()), float(vn_.max() - vn_.min()))

        lam = 0.0                                                                        # the smallest common turn-back that keeps every card's frustum on its canvas
        if self.keep_inside:
            for k_ in range(1, self.n):
                zk = self.z[k_]
                u0, v0, *_ = uv(zk, 0.0)
                if width(u0, v0) > 1.0:
                    lo_, hi_ = 0.0, 1.0
                    for _ in range(9):
                        mid = 0.5 * (lo_ + hi_)
                        u2, v2, *_r = uv(zk, mid)
                        lo_, hi_ = (lo_, mid) if width(u2, v2) <= 1.0 else (mid, hi_)
                    lam = max(lam, hi_)
        for k_, zk in enumerate(self.z):
            t_true = (zk - eye[2]) / dzs
            un, vn, ok, hb_, cc, t = uv(zk, lam)
            if k_ > 0 and self.keep_inside:
                un, vn = un + self._inside(un), vn + self._inside(vn)                    # a sideways overrun: slide the window (translation only)
            grids.append(torch.stack([un * 2 - 1, vn * 2 - 1], -1))
            ts.append(torch.where(t_true > 6.0, t_true, torch.full_like(t_true, 1e8)))   # a card the eye is past is a backdrop (behind all)
            valid.append(torch.ones_like(ok) if k_ == 0 else ok)                       # the back card is the backdrop whatever way the camera looks
            info["turn"].append(lam)
            info["hold"].append(hb_)
            info["mag"].append((self.ref + zk) * (cam.F / self.F) / max(self.depth_axis(cc, zk) + hb_, 1e-3))
            self.stats["held"] += int(hb_ != 0)
        self.last = info
        return torch.stack(grids), torch.stack(ts), torch.stack(valid)

    keep_inside = True

    @staticmethod
    def _inside(x):
        """shift (scalar) that moves the sampled range of x back into 0..1 (centres it when it is wider than the canvas)."""
        lo, hi = float(x.min()), float(x.max())
        if hi - lo >= 1.0:
            return 0.5 - 0.5 * (lo + hi)
        return -lo if lo < 0.0 else (1.0 - hi if hi > 1.0 else 0.0)

    def _note(self, f, grids, valid, cam):
        """diagnostics: share of the frame sampling beyond a card canvas (worst non-back card, with its frame), the largest card magnification."""
        out = ((grids.abs() > 1.0).any(-1) & valid)
        self.stats["frames"] += 1
        if self.n > 1:
            per = out[1:].float().mean((1, 2))
            v = float(per.max())
            if v > self.stats["over_max"]:
                self.stats.update(over_max=v, over_frame=int(f), over_card=int(per.argmax()) + 1)
            if v > 0.02:
                self.stats.setdefault("over_frames", []).append(int(f))
        self.stats["mag_max"] = max(self.stats["mag_max"], max(self.last["mag"]))
        self.stats["turn_max"] = max(self.stats.get("turn_max", 0.0), max(self.last["turn"]))

    def scales(self, cam):
        """per card: the median size (card canvas px per output px) of what one output pixel samples: the card's scale on screen. Continuous in the
        camera; the test asserts no jump between frames."""
        grids, _, valid = self.warp(cam)
        out = []
        for k in range(self.n):
            g = grids[k]
            gx, gy = g[..., 0] * (self.cw / 2.0), g[..., 1] * (self.ch / 2.0)
            dxx, dxy = gx[:, 1:, None] - gx[:, :-1, None], gy[:, 1:, None] - gy[:, :-1, None]
            dyx, dyy = gx[1:, :, None] - gx[:-1, :, None], gy[1:, :, None] - gy[:-1, :, None]
            det = (dxx[:-1, :, 0] * dyy[:, :-1, 0] - dxy[:-1, :, 0] * dyx[:, :-1, 0]).abs()
            m = valid[k][:-1, :-1]
            out.append(float(torch.sqrt(det[m]).median()) if bool(m.any()) else 0.0)
        return out

    def edge_report(self):
        return dict(self.stats)

    def occluder(self, f, cam, type_depth, cards=None):
        """HxW 0..1 numpy: how much of the cards in front of the type (per pixel, by depth) covers it: the `matte` of the legibility guard."""
        cards = self.cards_at(f) if cards is None else cards
        grids, ts, valid = self.warp(cam)
        D = torch.as_tensor(type_depth, dtype=torch.float32, device=self.dev)
        occ = torch.zeros_like(D)
        for k in range(self.n):
            smp = TF.grid_sample(cards[k:k + 1], grids[k:k + 1], mode="bilinear", padding_mode="reflection" if k == 0 else "zeros", align_corners=False)[0]
            occ = torch.maximum(occ, smp[3] * valid[k] * (ts[k] < D))
        return occ.cpu().numpy()

    def render(self, f, cam, type_rgb=None, type_a=None, type_depth=None, clear=None, cards=None):
        """frame f seen by `cam` -> float32 (H, W, 3) tensor on the device. type_*: premultiplied RGB (H,W,3), coverage (H,W) and camera depth
        (H,W) of the type layer (numpy or tensors); the type is composited per pixel by depth against the cards. clear: optional HxW 0..1
        (the legibility guard, `plan_legible`): where it is 1 the type is drawn in front of every card. cards: this frame's cards (`cards_from`)."""
        cards = self.cards_at(f) if cards is None else cards
        grids, ts, valid = self.warp(cam)
        self._note(f, grids, valid, cam)
        H, W = grids.shape[1:3]
        acc = torch.zeros(H, W, 3, device=self.dev)
        T = None
        if type_rgb is not None:
            T = (torch.as_tensor(type_rgb, dtype=torch.float32, device=self.dev), torch.as_tensor(type_a, dtype=torch.float32, device=self.dev),
                 torch.as_tensor(type_depth, dtype=torch.float32, device=self.dev))
            done = torch.zeros(H, W, dtype=torch.bool, device=self.dev)
        order = sorted(range(self.n), key=lambda k: -self.z[k])                         # far -> near
        bare = torch.zeros(H, W, 3, device=self.dev) if (T is not None and clear is not None) else None
        for k in order:
            if T is not None:
                before = (T[2] > ts[k]) & (T[1] > 0) & (~done)
                if before.any():
                    a = (T[1] * before)[..., None]
                    acc = T[0] * before[..., None] + acc * (1 - a)
                    done |= before
            smp = TF.grid_sample(cards[k:k + 1], grids[k:k + 1], mode="bilinear", padding_mode="reflection" if k == 0 else "zeros", align_corners=False)[0]
            a = (smp[3] * valid[k])[..., None]
            acc = smp[:3].permute(1, 2, 0) * a + acc * (1 - a)
            if bare is not None:
                bare = smp[:3].permute(1, 2, 0) * a + bare * (1 - a)
        if T is not None:
            rest = (T[1] > 0) & (~done)
            a = (T[1] * rest)[..., None]
            acc = T[0] * rest[..., None] + acc * (1 - a)
            if bare is not None:                                                         # legible: the flagged letters in front of every card
                c = torch.as_tensor(clear, dtype=torch.float32, device=self.dev)[..., None]
                front = T[0] + bare * (1 - T[1][..., None])
                acc = acc * (1 - c) + front * c
        return acc.clamp(0, 1)


def plan_legible(mp, chain, fps, type_at, depth_at, holds, **kw):
    """the legibility guard for card occlusion (user rule: a matte never hides a word's letters for the whole hold): compose.legible_plan with the
    cards in front of the type as the occluder. type_at(f) / depth_at(f) -> HxW coverage / camera depth of the type at frame f; holds = [(hit_s,
    release_s)]. -> the plan; `compose.legible_clear(type_a, plan, t)` then gives the `clear` map of `MultiPlane.render`."""
    from tools.fusion import compose
    occ = lambda f: mp.occluder(f, chain.at(f / fps), depth_at(f))
    return compose.legible_plan(type_at, occ, [tuple(h) for h in holds], fps=fps, **kw)


# ============================================================================================ camera hook
def camera_for(seq, cam, F=1000.0, extra=None):
    """CameraChain for the spec's "camera": {"chain": .., "plate": "multiplane", ..}. chain: "auto" (follow + hold / orbit rules), a gallery camera
    module id (that move on every word, hold_drift nowhere), or a list of {"module": id, "word": i | "window": "trans:0" | [t0, t1], ..params}
    carved into the default tiling (follow through the transitions, hold_drift on the other words). Any gallery chain move works: follow,
    pullback_reveal, hold_drift, orbit_even, spin_read_spin, push_through, tilt_to_plan, graze, ease_reversal, beat_orbit."""
    from mograph.modules.camera import CameraChain, get_camera
    chain = (cam or {}).get("chain", "auto")
    params = dict((cam or {}).get("params") or {})
    if chain == "auto":
        return CameraChain.auto(seq, override=dict(extra) if extra else None, F=F)
    if isinstance(chain, str):
        n = len([w for w in seq.words])
        entries = [{"module": chain, "word": i} for i in range(n)]
    else:
        entries = chain
    override = {}
    for e in entries:
        e = dict(e)
        name = e.pop("module")
        win = e.pop("window", None)
        if "word" in e:
            win = f"word:{e.pop('word')}"
        key = win if isinstance(win, str) else tuple(win)
        override[key] = get_camera(name)(**{**params, **e})
    override.update(extra or {})                      # v006: pan runs (the camera travel is the transition) replace the chain moves on their words
    return CameraChain.auto(seq, override=override, words="hold_drift", F=F)


# ============================================================================================ compose op (piece_runner / recipe_eval)
_STATE = {}
_SCENES = {}


def _state(key, setup):
    """per process: the type scene's camera chain + a MultiPlane without a clip, built once from `setup` (JSON: the piece's type spec, size, world, depth
    range of the type pass, palette bg, word holds)."""
    st = _STATE.get(key)
    if st is None:
        from tools.fusion import spec as S_
        sp = dict(setup["type_spec"])
        sk = setup.get("scene_key")                              # the shots of one piece share ONE type scene (slow to build: ports, plates)
        scene = _SCENES.get(sk) if sk else None
        if scene is None:
            scene = S_.build_scene(sp, size=tuple(setup["size"]))
            if sk:
                _SCENES[sk] = scene
        cam = sp.get("camera") if isinstance(sp.get("camera"), dict) else {}
        mp = MultiPlane(None, scene.world, tuple(setup["size"]), cards=int(cam.get("cards", 4)), z=plan(cam)["z"], ref=plan(cam)["ref"],
                        device=setup.get("device", "cpu"))
        if str(setup.get("device", "cpu")) == "cpu":
            torch.set_num_threads(1)                              # cpu compose runs one op per core
        _STATE[key] = st = dict(scene=scene, chain=scene.chain, mp=mp, setup=setup)
    return st


def compose_op(footage, depth, matte, type_rgb, type_a, type_depth, t=0.0, state_key="mp", setup=None, setup_path=None, legible=True, min_vis=0.8,
               ramp_s=0.25):
    """compose op `multiplane`: the plate as depth cards seen by the piece's type camera, the type composited between / through them per pixel by
    depth, the legibility guard on card occlusion (a letter the cards hide is drawn in front during its word's hold, ramped).
    footage HxWx3 float, depth / matte HxW float (near = 1 / subject = 1), type_rgb HxWx3 (the type pass beauty on the palette bg), type_a HxW (the type
    matte pass), type_depth HxW float (the type depth pass grey: 32 + 223 (1 - (z - near) / (far - near)), 0 = no type). -> HxWx3 float, footage size."""
    if setup is None and state_key not in _STATE:
        import json
        with open(setup_path, encoding="utf-8") as fh:
            setup = json.load(fh)
    st = _state(state_key, setup)
    mp, chain, S = st["mp"], st["chain"], st["setup"]
    w, h = mp.size
    fh, fw = footage.shape[:2]
    rs = lambda x, interp=cv2.INTER_AREA: x if x.shape[:2] == (h, w) else cv2.resize(np.ascontiguousarray(x), (w, h), interpolation=interp)
    plate = np.clip(rs(footage) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    dep = np.clip(rs(np.asarray(depth, np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mat = np.clip(rs(np.asarray(matte, np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8) if matte is not None else None
    A = np.clip(rs(np.asarray(type_a, np.float32), cv2.INTER_LINEAR), 0.0, 1.0)
    g = rs(np.asarray(type_depth, np.float32), cv2.INTER_LINEAR) * 255.0
    near, far = float(S["near"]), float(S["far"])
    if S.get("mode", "linear") == "inverse":
        u = 1.0 - (g - 32.0) / 223.0
        z = 1.0 / np.maximum(1.0 / far + u * (1.0 / near - 1.0 / far), 1e-9)
    else:
        z = near + (1.0 - (g - 32.0) / 223.0) * (far - near)
    D = np.where(A > 1e-3, z, 1e9).astype(np.float32)
    bg = np.asarray(S.get("bg", (0.0, 0.0, 0.0)), np.float32) / 255.0
    rgb = np.clip(rs(np.asarray(type_rgb, np.float32), cv2.INTER_LINEAR) - bg * (1.0 - A[..., None]), 0.0, 1.0)      # premultiplied by the coverage
    cam = chain.at(float(t))
    cards = mp.cards_from(plate, dep, mat)
    clear = None
    if legible and S.get("holds") and A.max() > 0.01:
        from tools.fusion import compose
        from tools.fusion.compose import _leg_weight
        wt = max((_leg_weight(float(t), float(a), float(b), ramp_s) for a, b in S["holds"]), default=0.0)
        if wt > 1e-3:
            clear = _clear_now(mp, cam, cards, A, D, wt, min_vis)
    img = mp.render(0, cam, rgb, A, D, clear=clear, cards=cards).cpu().numpy()
    return img if (fh, fw) == (h, w) else cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)


def _clear_now(mp, cam, cards, A, D, weight, min_vis=0.8):
    """stateless legibility guard for one frame of a hold: every letter (connected component of the type coverage) the cards in front of it leave
    less than `min_vis` visible is drawn in front of all cards (HxW 0..1, feathered), times the hold's ramp weight."""
    from tools.fusion import compose
    occ = mp.occluder(0, cam, D, cards=cards)
    s = compose._leg_scale(*A.shape[:2])
    lab, comps = compose._leg_components(compose._leg_small(A, s))
    if not comps:
        return None
    o = cv2.resize(occ, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR) if occ.shape != lab.shape else occ
    ta = compose._leg_small(A, s)
    flag = np.zeros(lab.shape, np.float32)
    for (i, x0, y0, x1, y1, _a) in comps:
        t = ta[y0:y1, x0:x1]
        tot = float(t.sum())
        if tot > 1e-3 and float((t * (1.0 - o[y0:y1, x0:x1])).sum()) / tot < min_vis:
            flag = np.maximum(flag, (lab == i).astype(np.float32))
    if not flag.any():
        return None
    d = max(1, int(round(6 * s)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    c = np.clip(cv2.GaussianBlur(cv2.dilate(flag, k), (0, 0), max(0.8, 0.5 * d)) * 1.5, 0.0, 1.0) * float(weight)
    return cv2.resize(c, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_LINEAR) if c.shape != A.shape else c
