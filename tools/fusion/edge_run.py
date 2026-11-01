"""edge_run.py - lego block `edge_run_small_type` (v005 P20 Delta, tools/fusion/blocks.py): small precise type RUNNING ALONG a footage edge - the
glyph baseline follows a traced polyline that is re-traced and tracked on every frame (river channel edges, coastline, depth lines).

    edge_run_type(img, text, font, t, t0, t1, ...)   compose op (numpy in / alpha HxW out, registered in compose._ops_registry): traces the
                                                     edge, picks the stretch the line fits, tracks it, sets the glyphs on it on the GPU.
    weave(piece, music, chk)                         the compose side of the block: for every line with ``"edge_run"`` (or a small_line line the
                                                     block picked) adds the op + a dark halo + the ink in the type colour, and hands the
                                                     module's own letters over to it for the hold (mask_gate on wm / wl).

Reused, not rewritten
  * contour_ports.iso_lines / _resample / _arc  (the luma_edges / depth_iso extraction of the contour ports = videotrace "topo" iso-lines)
  * mathtype kit: ``kit.draw.sample`` (bilinear glyph-atlas lookup), ``kit.draw.box_down`` (2x2 supersample -> 1x), ``kit.core.backend``
    (torch cuda / cpu), fonts from mograph/fonts (the liked files, PIL rasterising as ``kit.text``)
Tracking (per frame, state per ``state_key`` like flow_wake): iso-lines of the blurred luminance (levels fixed from the first frame) -> open
polylines -> heavy smoothing along the arc (letters need a gentle curve) -> the window of the text's length with low curvature, mild tilt,
strong gradient, inside the frame, near the line's layout box -> next frames: the candidate nearest the previous window (so the type stays on
the same channel), EMA 0.5 on the points, a lost edge holds the last window for a few frames and then re-picks.
Legibility: cap height = max(cap_frac * H, min_px_at_640 * H / 640) (26 px at 640 tall, the floor of hold_layouts.MIN_CAP_FRAC), baseline on the
edge, text always reads left to right, the halo is a dark blurred copy under the ink."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "mograph" / "fonts"
SS = 2                              # glyph atlas / render supersample (box-down 2x2 at the end)
DS = 2.0                            # polyline resample step, px
SMOOTH_PX = 18.0                    # gaussian sigma along the arc: gentle curves only
KMAX = 1.0 / 50.0                   # max curvature (1/px) of a window the type may follow
MAX_TILT = 55.0                     # deg: steeper stretches are not followed (the text would read sideways)
STRICT = dict(tilt=38.0, local=18.0)    # a window the type is set on: every tangent within +-tilt of horizontal, turning <= local deg per cap height
RELAXED = dict(tilt=52.0, local=30.0)   # used when no strict stretch exists
M_PTS = 64                          # points of a tracked window
_STATE: dict = {}
_ATLAS: dict = {}


# ======================================================================================================== glyph atlas
def font_file(font):
    """font name / file name / path -> the path of a LIKED face (mograph/fonts_liked.font_file: raises FontNotLiked for anything else; no 'path as given' fallback since 2026-10-30)."""
    from mograph import fonts_liked as fl
    return Path(fl.font_file(Path(str(font)).name if Path(str(font)).is_file() else str(font)))


class Atlas:
    """rasterised glyphs of one text at one cap height: per glyph the ink sprite (SS x), its ink box relative to the baseline origin and the
    advance (px at 1x, kerning from the font, + tracking)."""

    def __init__(self, text, font, cap_px, track=0.1):
        path = font_file(font)
        probe = ImageFont.truetype(str(path), 200)
        ratio = float(max(probe.getbbox("H", anchor="ls")[3] - probe.getbbox("H", anchor="ls")[1], 1)) / 200.0       # cap height per px of size
        size = max(8, int(round(cap_px * SS / ratio)))
        f = ImageFont.truetype(str(path), size)
        self.text, self.cap_px, self.size = text, float(cap_px), size
        self.pad = 3
        sprites, self.l, self.t, self.adv, self.ink = [], [], [], [], []
        for i, ch in enumerate(text):
            adv = (f.getlength(text[:i + 1]) - f.getlength(text[:i])) / SS + float(track) * size / SS
            ink, l, t = False, 0, 0
            if ch.strip() != "":                                        # measure the real ink (getbbox misses some variable-axis shapes)
                m = size
                scratch = Image.new("L", (3 * m, 3 * m), 0)
                ImageDraw.Draw(scratch).text((m, 2 * m), ch, font=f, fill=255, anchor="ls")
                a = np.asarray(scratch)
                ys, xs = np.nonzero(a > 0)
                if len(xs):
                    ink, l, t = True, int(xs.min()) - m, int(ys.min()) - 2 * m
                    spr = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    sprites.append(np.pad(spr, self.pad).astype(np.float32) / 255.0)
            if not ink:
                sprites.append(None)
            self.l.append(l)
            self.t.append(t)
            self.adv.append(adv)
            self.ink.append(ink)
        self.sprites = sprites
        self.length = float(sum(self.adv[:-1]) + (self.adv[-1] if self.adv else 0.0))
        self._dev = {}

    def tensors(self, device):
        k = str(device)
        if k not in self._dev:
            import torch
            self._dev[k] = [None if s is None else torch.from_numpy(s)[None].to(device) for s in self.sprites]
        return self._dev[k]


def atlas_for(text, font, cap_px, track=0.1):
    key = (text, str(font), round(float(cap_px), 2), round(float(track), 3))
    if key not in _ATLAS:
        if len(_ATLAS) > 12:
            _ATLAS.clear()
        _ATLAS[key] = Atlas(text, font, cap_px, track)
    return _ATLAS[key]


# ======================================================================================================== tracing (cv2 / numpy)
def _cp():
    from mograph.modules import contour_ports as CP
    return CP


def _smooth_line(P, sigma_px=SMOOTH_PX, ds=DS):
    from scipy.ndimage import gaussian_filter1d
    CP = _cp()
    if len(P) < 4:
        return None
    Q = CP._resample(P, max(int(CP._arc(P, False)[-1] / ds), 4), False)
    if len(Q) < 8:
        return None
    return np.stack([gaussian_filter1d(Q[:, i].astype(np.float64), sigma_px / ds, mode="nearest") for i in range(2)], 1).astype(np.float32)


def candidates(gray, levels, min_len=60.0, keep=4):
    """smooth open polylines (left -> right, resampled every DS px) of the luminance iso-lines: [(n, 2) float32 px]."""
    CP = _cp()
    out = []
    for lvl in CP.iso_lines(gray, levels, min_len=min_len, smooth=2.0, keep=keep):
        for pts, closed in lvl:
            if closed:                                                  # an island: cut it at its leftmost point, follow the upper arc
                k = int(np.argmin(pts[:, 0]))
                pts = np.roll(pts, -k, 0)
                half = np.argmax(pts[:, 0])
                pts = pts[:half + 1]
            P = _smooth_line(pts)
            if P is None:
                continue
            if P[-1, 0] < P[0, 0]:
                P = P[::-1].copy()
            out.append(P)
    return out


def _arcs(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _kappa(P):
    d = np.gradient(P, axis=0)
    phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    s = _arcs(P)
    return np.abs(np.gradient(phi) / np.maximum(np.gradient(s), 1e-3))


def window_ok(seg, cap, lim):
    """can type of cap height `cap` px be set along seg (M, 2)? all tangents within lim['tilt'] deg of horizontal and the direction turns at
    most lim['local'] deg over one cap height (a tight bend makes letters overlap / fan out)."""
    d = np.gradient(seg, axis=0)
    ang = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    if float(np.abs(ang).max()) > lim["tilt"]:
        return False
    s = _arcs(seg)
    step = max(float(s[-1]) / (len(seg) - 1), 1e-3)
    n = max(1, int(round(float(cap) / step)))
    return len(ang) <= n or float(np.abs(ang[n:] - ang[:-n]).max()) <= lim["local"]


def _window(P, s, i0, L):
    """the polyline stretch of arc length L from index i0 (None when the line is too short), resampled to M_PTS points."""
    j = int(np.searchsorted(s, s[i0] + L))
    if j >= len(P):
        return None
    seg = P[i0:j + 1]
    if len(seg) < 3:
        return None
    return _cp()._resample(seg, M_PTS, False)


def pick_window(lines, gradmag, L, near, size, margin, cap=30.0):
    """best stretch of length L over all candidate lines: lowest cost of curvature + tilt + distance to `near` (px) - edge strength; only
    windows type can be set on (window_ok strict, else relaxed). None when there is none."""
    W, H = size
    diag = math.hypot(W, H)
    best = {"strict": None, "relaxed": None}
    for P in lines:
        s = _arcs(P)
        if s[-1] < L * 1.02:
            continue
        kap = _kappa(P)
        step = max(1, int(round(6 / DS)))
        for i0 in range(0, len(P) - 2, step):
            if s[-1] - s[i0] < L * 1.02:
                break
            seg = _window(P, s, i0, L * 1.04)
            if seg is None:
                continue
            if seg[:, 0].min() < margin or seg[:, 0].max() > W - margin or seg[:, 1].min() < margin or seg[:, 1].max() > H - margin:
                continue
            j = int(np.searchsorted(s, s[i0] + L * 1.04))
            if float(kap[i0:j + 1].max()) > KMAX:
                continue
            ch = seg[-1] - seg[0]
            tilt = abs(math.degrees(math.atan2(ch[1], ch[0])))
            if tilt > MAX_TILT:
                continue
            cen = seg.mean(0)
            g = float(gradmag[np.clip(seg[:, 1].astype(int), 0, H - 1), np.clip(seg[:, 0].astype(int), 0, W - 1)].mean())
            cost = 2.0 * float(kap[i0:j + 1].mean()) / (1.0 / 120.0) + 2.0 * (tilt / MAX_TILT) ** 2 + 2.5 * float(np.linalg.norm(cen - near)) / diag - 1.5 * g
            for name, lim in (("strict", STRICT), ("relaxed", RELAXED)):
                if window_ok(seg, cap, lim):
                    if best[name] is None or cost < best[name][0]:
                        best[name] = (cost, seg)
                    break
    pick = best["strict"] or best["relaxed"]
    return None if pick is None else pick[1]


def straight_window(L, near, size):
    """fallback when no edge carries the text: a level baseline through `near`, kept inside the frame."""
    W, H = size
    cx = min(max(near[0], L * 0.52), W - L * 0.52)
    return np.stack([np.linspace(cx - L * 0.52, cx + L * 0.52, M_PTS), np.full(M_PTS, min(max(near[1], 0.1 * H), 0.9 * H))], 1).astype(np.float32)


def track_window(lines, prev, L, size, margin):
    """the candidate stretch that continues `prev` (M_PTS, 2): start at the point of each line nearest prev's start, length L; smallest mean
    distance wins. -> (window, mean distance) or (None, inf)."""
    W, H = size
    best, bd = None, 1e30
    for P in lines:
        s = _arcs(P)
        if s[-1] < L * 1.0:
            continue
        i0 = int(np.argmin(np.linalg.norm(P - prev[0], axis=1)))
        if s[-1] - s[i0] < L * 1.0:                                     # slide back so the window fits the line
            i0 = int(np.searchsorted(s, s[-1] - L * 1.04))
        seg = _window(P, s, max(i0, 0), L * 1.04)
        if seg is None:
            continue
        d = float(np.linalg.norm(seg - prev, axis=1).mean())
        if d < bd:
            best, bd = seg, d
    return best, bd


def edge_levels(gray, n=6):
    a, b = np.percentile(gray, 14), np.percentile(gray, 92)
    return list(np.linspace(a, b, n + 2)[1:-1])


# ======================================================================================================== GPU glyph setting (kit)
def _kit():
    p = str(ROOT / "tools" / "mathtype")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kit
    from kit import draw
    return kit, draw


def _device():
    import torch
    try:
        from tools.fusion import compose
        if getattr(compose.B, "name", "") == "torch":
            return compose.B.device
    except Exception:                                                   # noqa: BLE001
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def place_glyphs(atlas, Q, lift=0.0):
    """glyph poses along the tracked polyline Q (M, 2) px: [(cx, cy, theta_rad)] of every glyph's advance-box centre (baseline point lifted
    along the normal) and its tangent from the chord across its advance box."""
    s = _arcs(Q)
    out, o = [], 0.0
    for adv in atlas.adv:
        a, m, b = (min(max(x, 0.0), s[-1]) for x in (o, o + adv / 2.0, o + adv))
        P = lambda u: np.array([np.interp(u, s, Q[:, 0]), np.interp(u, s, Q[:, 1])])      # noqa: E731
        c, p0, p1 = P(m), P(a), P(b)
        th = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        out.append((c[0] + math.sin(th) * lift, c[1] - math.cos(th) * lift, th))
        o += adv
    return out


def render_glyphs(atlas, poses, H, W, alphas=None, device=None):
    """the glyphs of ``atlas`` at ``poses`` -> coverage (H, W) float32 numpy 0..1. GPU: per glyph a crop window sampled from the atlas with
    kit.draw.sample at 2x and boxed down with kit.draw.box_down; letters overlap by max."""
    import torch
    kit, draw = _kit()
    dev = device or _device()
    sp = atlas.tensors(dev)
    canvas = torch.zeros((H, W), dtype=torch.float32, device=dev)
    for i, (cx, cy, th) in enumerate(poses):
        spr = sp[i]
        if spr is None:
            continue
        a_i = 1.0 if alphas is None else float(alphas[i])
        if a_i <= 1e-3:
            continue
        l, t = atlas.l[i], atlas.t[i]
        hh, ww = spr.shape[1], spr.shape[2]
        R = int(math.ceil(math.hypot(ww / SS + atlas.adv[i], hh / SS + abs(t) / SS))) + 4      # the crop is centred on the BASELINE point: reach the ink top
        x0, y0 = int(math.floor(cx)) - R, int(math.floor(cy)) - R
        n = 2 * R
        if x0 >= W or y0 >= H or x0 + n <= 0 or y0 + n <= 0:
            continue
        ys, xs = torch.meshgrid(torch.arange(n * SS, device=dev, dtype=torch.float32), torch.arange(n * SS, device=dev, dtype=torch.float32), indexing="ij")
        px = x0 + (xs + 0.5) / SS - cx                                  # offset from the glyph centre, px at 1x
        py = y0 + (ys + 0.5) / SS - cy
        c, s_ = math.cos(th), math.sin(th)
        along = px * c + py * s_
        perp = px * s_ - py * c                                         # + = up (normal (sin, -cos))
        xo = along + atlas.adv[i] / 2.0                                 # from the baseline origin of the glyph
        u = xo * SS - l + atlas.pad                                     # sprite coordinates, SS px
        v = -perp * SS - t + atlas.pad
        cov = draw.sample(spr, u, v, k=1.0, border="zeros")[0]          # (n*SS, n*SS)
        cov = draw.box_down(cov, SS) * a_i
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + n, W), min(y0 + n, H)
        tile = cov[ya - y0:yb - y0, xa - x0:xb - x0]
        canvas[ya:yb, xa:xb] = torch.maximum(canvas[ya:yb, xa:xb], tile)
    return canvas.clamp(0, 1).cpu().numpy()


# ======================================================================================================== the op
def cap_px_for(H, cap_frac=0.05, min_px_at_640=26):
    return max(float(cap_frac) * H, float(min_px_at_640) * H / 640.0)


def _ss(x, a, b):
    u = min(max((x - a) / max(b - a, 1e-9), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def edge_run_type(img, text, font, t=0.0, t0=0.0, t1=1e9, fade=0.2, reveal_s=0.35, cap_frac=0.05, min_px_at_640=26, track=0.1, near=(0.5, 0.5),
                  source="luma_edges", depth=None, levels=6, lift=3.0, ema=0.5, frame_idx=None, state_key="edge_run", return_info=False):
    """Small type along a tracked footage edge -> alpha HxW (float32 0..1; the ink, no halo). ``img`` the plate frame (HxWx3, 0..1); ``source``
    luma_edges (iso-lines of its luminance) | depth_iso (of ``depth`` HxW). The text appears letter by letter from t0 (``reveal_s``), is
    there until t1 and fades over ``fade`` s at both ends; outside [t0 - fade, t1 + fade] nothing is computed and the track is dropped.
    State per ``state_key`` continues while frame_idx advances by 1 (else a cold start picks the stretch again)."""
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    t = float(t)
    if t < float(t0) - 1e-6 or t > float(t1) + float(fade):
        _STATE.pop(state_key, None)
        return np.zeros((H, W), np.float32)
    txt = str(text).upper()
    cap = cap_px_for(H, cap_frac, min_px_at_640)
    at = atlas_for(txt, font, cap, track)
    L = at.length
    if source == "depth_iso" and depth is not None:
        d = np.asarray(depth, np.float32)
        d = d[..., 0] if d.ndim == 3 else d
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) if d.shape != (H, W) else d
        gray = d
    else:
        gray = cv2.cvtColor(np.clip(a[..., :3], 0, 1), cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
    st = _STATE.get(state_key)
    cont = st is not None and (frame_idx is None or int(frame_idx) in (st["last"], st["last"] + 1))
    if not cont:
        st = _STATE[state_key] = dict(win=None, lost=0, levels=None, last=None)
    st["last"] = None if frame_idx is None else int(frame_idx)
    if st["levels"] is None:
        st["levels"] = edge_levels(gray, int(levels))
    lines = candidates(gray, st["levels"], min_len=max(L * 0.8, 40.0))
    margin = 0.04 * min(W, H)
    gm = cv2.GaussianBlur(np.hypot(*(cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, d, 1 - d, ksize=3) for d in (1, 0))), (0, 0), 2.0)
    gm = gm / (float(np.percentile(gm, 99)) + 1e-6)
    win = None
    if st["win"] is not None:
        win, dist = track_window(lines, st["win"], L, (W, H), margin)
        if win is not None and dist > 0.15 * min(W, H):
            win = None
        if win is not None and not window_ok(win, cap, RELAXED):
            win = None
        if win is None:
            st["lost"] += 1
            if st["lost"] <= 6:
                win = st["win"]
        else:
            st["lost"] = 0
            win = ema * st["win"] + (1.0 - ema) * win
    if win is None:
        where = st["win"].mean(0) if st["win"] is not None else np.array([near[0] * W, near[1] * H])       # re-pick close to where it was
        win = pick_window(lines, gm, L, where, (W, H), margin, cap)
        st["lost"] = 0
        if win is not None and st["win"] is not None and float(np.linalg.norm(win.mean(0) - st["win"].mean(0))) > 0.12 * math.hypot(W, H):
            st["born"] = t                                              # a new stretch (first pick / the edge was lost): the letters land again
    if win is None:
        win = straight_window(L, np.array([near[0] * W, near[1] * H]), (W, H))
    st["win"] = win
    poses = place_glyphs(at, win, lift)
    k = max(len(txt.replace(" ", "")), 1)
    al = []
    ink_i = 0
    for ch, ink in zip(txt, at.ink):
        if ink:
            al.append(_ss(t - max(float(t0), st.get("born", float(t0))) - ink_i * float(reveal_s) / k, 0.0, 0.12))
            ink_i += 1
        else:
            al.append(1.0)
    env = (1.0 - _ss(t, float(t1), float(t1) + float(fade))) if fade > 0 else 1.0
    cov = render_glyphs(at, poses, H, W, [x * max(env, 0.0) for x in al])
    return (cov, dict(window=win, cap_px=cap, length=L)) if return_info else cov


OPS = {"edge_run_type": edge_run_type}


# ======================================================================================================== weave
_TAIL_OPS = ("grade", "grain", "vignette")


def _lines_for(piece, blk):
    """words that get an edge run: an explicit ``"edge_run"`` key, else the block's lines whose layout is small_line."""
    words = piece["type"]["words"]
    out = []
    for i, w in enumerate(words):
        if w.get("edge_run"):
            out.append(i)
        elif blk is not None and i in blk.get("lines", []) and (w.get("hold_layout") or {}).get("name") == "small_line":
            out.append(i)
    return out


def weave(piece, music=None, chk=None, root=None):
    """block edge_run_small_type, compose side (idempotent, after weave_matte / weave_flow and plan.timing exist): per line one edge_run_type op
    + halo + ink in the type colour before the closing grade, the module's letters (wm / wl) gated off while the edge run holds the word."""
    plan = piece.setdefault("plan", {})
    blk = next((b for b in plan.get("blocks", []) if b.get("name") == "edge_run_small_type"), None)
    words = piece["type"]["words"]
    idx = _lines_for(piece, blk)
    if not idx:
        return piece
    if (plan.get("edge_run") or {}).get("lines"):
        return piece
    params = (blk or {}).get("params", {})
    timing = plan.get("timing") or []
    comp = piece.get("compose") or {}
    chains = ([comp["default"]] if comp.get("default") else []) + list((comp.get("shots") or {}).values())
    H = int((piece.get("compose") or {}).get("size", [360, 640])[1])
    info = []
    made = []
    for n, i in enumerate(idx):
        w = words[i]
        cfg = dict(w.get("edge_run") or {})
        tm = timing[i] if i < len(timing) else None
        hit = float(tm["hit"]) if tm else float(w["t0"]) + float(w.get("form_s", 0.5))
        rel = float(tm["release"]) if tm else float(w["t1"]) - float(w.get("exit_s", 0.5))
        box = (w.get("hold_layout") or {}).get("box") or {}
        a = dict(text=w["text"], font=w["font"], t="$t", t0=round(hit - 0.05, 3), t1=round(rel - 0.05, 3), fade=0.2, reveal_s=0.35,
                 cap_frac=float(cfg.get("cap_frac", params.get("cap_frac", 0.05))), min_px_at_640=int(cfg.get("min_px_at_640", params.get("min_px_at_640", 26))),
                 track=float(cfg.get("track", 0.1)), near=[float(box.get("cx", 0.5)), float(box.get("cy", 0.5))],
                 source=cfg.get("source", "luma_edges"), frame_idx="$frame_idx", state_key=f"{piece.get('id', 'piece')}:edge_run:{i}")
        made.append((n, i, a, hit, rel))
        info.append(dict(line=i, text=w["text"], window=[a["t0"], a["t1"]], cap_px_at_640=round(cap_px_for(640, a["cap_frac"], a["min_px_at_640"]), 1),
                         source=a["source"]))
    for ch in chains:
        ops = ch["ops"]
        if any(o.get("op") == "edge_run_type" for o in ops):
            continue
        outs = {o.get("out") for o in ops}
        k = len(ops)
        while k > 0 and ops[k - 1].get("op") in _TAIL_OPS:
            k -= 1
        new, gates = [], []
        for n, i, a, hit, rel in made:
            args = dict(a)
            args["img"] = "$footage"
            if a["source"] == "depth_iso":
                args["depth"] = "$depth"
            nm = f"_er{n}"
            new += [{"op": "edge_run_type", "out": nm, "args": args},
                    {"op": "blur", "out": nm + "h", "args": {"img": "$" + nm, "sigma": 2.2}},
                    {"op": "mask_gate", "out": nm + "h", "args": {"a": "$" + nm + "h", "gate": 1.0, "scale": 2.4}},
                    {"op": "over", "out": "image", "args": {"base": "$image", "alpha": "$" + nm + "h", "fg": [0.02, 0.02, 0.03]}},
                    {"op": "over", "out": "image", "args": {"base": "$image", "alpha": "$" + nm,
                                                            "fg": "$tcol" if "tcol" in outs else [0.95, 0.95, 0.97]}}]
            gates.append([[round(hit - 0.15, 3), 1.0], [round(hit + 0.1, 3), 0.0], [round(rel - 0.1, 3), 0.0], [round(rel + 0.15, 3), 1.0]])
        ch["ops"] = ops[:k] + new + ops[k:]
        # the module's own letters step aside while the edge run holds the word (right after the flow_wake outputs, before anything reads them)
        last = max((j for j, o in enumerate(ch["ops"]) if o.get("op") == "flow_wake" and o.get("out") in ("wm", "wl")), default=None)
        if last is not None:
            gate_ops = []
            for g in gates:
                for nm in ("wm", "wl"):
                    gate_ops.append({"op": "mask_gate", "out": nm, "args": {"a": "$" + nm, "gate": {"timeline": g, "ease": "smooth"}, "scale": 1.0}})
            ch["ops"][last + 1:last + 1] = gate_ops
    plan["edge_run"] = {"lines": info, "halo": "dark blurred copy of the ink (sigma 2.2 px)", "module_letters_gated": True}
    return piece
