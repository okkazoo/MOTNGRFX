"""strip_resnap.py - lego block `strips_persist_across_cuts` (v005 P18 Bowl, tools/fusion/blocks.py): flat primary-colour strips stay on screen
across the hard cuts and, on every cut downbeat, re-angle / re-position to the NEW shot's contours, easing in 0.25 s; the type sitting on the
strips moves with them.

It extends the existing compose op ``layout_shapes`` (no second strip op): a shape may carry a ``pose`` = {"pivot": [x, y], "keys": [[t, dx, dy,
dang], ...], "ease": "smooth"} - keyframes in piece seconds of a rigid move of the strip (dx, dy in frame fractions, dang in degrees, clockwise,
about the pivot, the centre of the strip). ``compose.shape_mask(..., t=)`` evaluates it with ``pose_at`` / ``move_xy`` below. The type on the strip
follows through ``rigid_pose`` (a numpy op registered in ``compose._ops_registry``), driven by the same keyframes as timelines.

    weave(piece, music, chk)      after plan_rules.weave_layout (+ weave_flow): reads the cut times of the piece layout and each shot's first
                                  frames (plate + matte, local files) -> per-cut targets {angle, cy}, writes the poses into the layout_shapes
                                  op of every chain and inserts the `rigid_pose` ops after the flow_wake outputs (wm, wl) of the type.
    cut_targets(piece, root)      {sid: {angle, cy, strength, edge_angle, edge_y}} from the first frames of every shot.
    dominant_angle / best_edge_row  the analysis on arrays (tests run on synthetic stripes).

Target of a strip on shot k: ``angle`` = the dominant EDGE direction of the shot's first frames (structure of the luminance gradient plus the
subject matte outline, orientation histogram within +-30 deg of horizontal) clamped to +-angle_deg (block param); ``cy`` = the strip's lower
boundary laid on the strongest edge line along that direction, kept inside the frame. Only full-bleed rect strips move; polygon / circle
layouts (split_sides, edge_L, type_circle ...) keep their shape, and the type follows only what has a strip."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import cv2
import numpy as np

EASE_S = 0.25                      # the strip eases to the new pose over this long, starting on the cut
ANGLE_WINDOW = 30.0                # deg either side of horizontal searched for the dominant edge direction
MIN_STRENGTH = 0.12                # share of edge mass inside the window below which the direction is not trusted (strip goes flat)
CY_BOUNDS = (0.2, 0.8)             # the strip centre stays here (the type sits on it and must stay on screen)
FULL_BLEED_W = 1.2                 # rects at least this wide (frame fractions) are strips; the rest of the shapes stay put
FOLLOW_BUFFERS = ("wm", "wl")      # the type layers after flow_wake (plan_rules.weave_flow); else $type_matte / $type_line are wrapped


# ======================================================================================================== pose maths (no compose import)
def pose_at(keys, t, ease="smooth"):
    """(dx, dy, dang) at time t from keyframes [[t, dx, dy, dang], ...] (sorted in time; clamped outside; smoothstep between keys)."""
    ks = sorted(keys, key=lambda k: k[0])
    if t <= ks[0][0]:
        return tuple(float(v) for v in ks[0][1:4])
    for a, b in zip(ks[:-1], ks[1:]):
        if t <= b[0]:
            u = (t - a[0]) / max(b[0] - a[0], 1e-9)
            u = u * u * (3.0 - 2.0 * u) if ease == "smooth" else u
            return tuple(float(a[i] + (b[i] - a[i]) * u) for i in (1, 2, 3))
    return tuple(float(v) for v in ks[-1][1:4])


def move_xy(pts, dx_px, dy_px, dang, px, py):
    """rigid move of points (n, 2) px: rotate dang degrees clockwise (y down) about (px, py), then translate."""
    a = math.radians(float(dang))
    ca, sa = math.cos(a), math.sin(a)
    p = np.asarray(pts, np.float64)
    x, y = p[..., 0] - px, p[..., 1] - py
    return np.stack([px + x * ca - y * sa + dx_px, py + x * sa + y * ca + dy_px], -1)


def rigid_pose(img, dx=0.0, dy=0.0, angle=0.0, cx=0.5, cy=0.5):
    """compose op (numpy): the picture / mask moves rigidly like a posed strip: rotated `angle` degrees clockwise about (cx, cy) (frame
    fractions), then shifted by (dx, dy) (fractions of the width / height). Zeros come in from outside (masks)."""
    a = np.asarray(img, np.float32)
    if abs(float(dx)) < 1e-6 and abs(float(dy)) < 1e-6 and abs(float(angle)) < 1e-4:
        return a
    h, w = a.shape[:2]
    m = cv2.getRotationMatrix2D((float(cx) * w, float(cy) * h), -float(angle), 1.0)
    m[0, 2] += float(dx) * w
    m[1, 2] += float(dy) * h
    out = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out if out.ndim == a.ndim else out[..., None]


OPS = {"rigid_pose": rigid_pose}


# ======================================================================================================== analysis (arrays)
def _grad(gray):
    g = cv2.GaussianBlur(np.asarray(gray, np.float32), (0, 0), 1.2)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def edge_field(grays, mattes=None, matte_w=1.0):
    """summed gradient field (gx, gy) of the luminance frames (+ the matte outline, `matte_w` x as strong after normalising both)."""
    gx = gy = None
    for g in grays:
        a, b = _grad(g)
        gx = a if gx is None else gx + a
        gy = b if gy is None else gy + b
    s = float(np.percentile(np.hypot(gx, gy), 99)) + 1e-6
    gx, gy = gx / s, gy / s
    if mattes is not None and len(mattes):
        mx = my = None
        for m in mattes:
            a, b = _grad(cv2.resize(np.asarray(m, np.float32), (gx.shape[1], gx.shape[0]), interpolation=cv2.INTER_LINEAR))
            mx = a if mx is None else mx + a
            my = b if my is None else my + b
        sm = float(np.percentile(np.hypot(mx, my), 99)) + 1e-6
        if sm > 1e-4 and float(np.abs(mx).max() + np.abs(my).max()) > 1e-4:
            gx, gy = gx + matte_w * mx / sm, gy + matte_w * my / sm
    return gx, gy


def dominant_angle(gx, gy, window=ANGLE_WINDOW, top_pct=80.0):
    """dominant edge direction within +-window degrees of horizontal, clockwise-positive on screen (y down) -> (angle_deg, strength).
    Edge direction = gradient direction + 90; orientation histogram (1 deg bins) weighted by the gradient magnitude of the strongest
    pixels, smoothed, peak, then a weighted mean around the peak. strength = share of that mass inside the window."""
    mag = np.hypot(gx, gy)
    sel = mag >= np.percentile(mag, top_pct)
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 90.0 + 90.0) % 180.0 - 90.0           # (-90, 90]
    wgt = mag[sel].astype(np.float64)
    tot = float(wgt.sum()) + 1e-9
    inside = np.abs(ang) <= window
    mass = float(wgt[inside].sum())
    if mass <= 0.0:
        return 0.0, 0.0
    bins = np.arange(-window, window + 1.0)
    hist = np.array([wgt[(ang >= b - 0.5) & (ang < b + 0.5)].sum() for b in bins])
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(hist, k / k.sum(), mode="same")
    pk = float(bins[int(np.argmax(sm))])
    near = np.abs(ang - pk) <= 5.0
    est = float((ang[near] * wgt[near]).sum() / max(float(wgt[near].sum()), 1e-9))
    return est, mass / tot


def best_edge_row(gx, gy, angle, y_lo, y_hi, step=2):
    """row (fraction of the height) of the strongest edge line along `angle` degrees through the middle of the frame, searched between y_lo
    and y_hi (fractions) -> (y_fraction, score). Line y = yc + (x - W/2) tan(angle); score = mean gradient magnitude along it."""
    mag = np.hypot(gx, gy).astype(np.float32)
    mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
    h, w = mag.shape
    xs = np.arange(w, dtype=np.float32)
    t = math.tan(math.radians(float(angle)))
    cands = np.arange(max(0.0, y_lo * h), min(h - 1.0, y_hi * h) + 1e-6, float(step), dtype=np.float32)
    if not len(cands):
        return 0.5, 0.0
    prof = []
    for yc in cands:
        ys = (yc + (xs - w / 2.0) * t).astype(np.float32)
        v = cv2.remap(mag, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        prof.append(float(v.mean()))
    prof = np.convolve(np.asarray(prof), np.ones(5) / 5.0, mode="same")
    i = int(np.argmax(prof))
    return float(cands[i] / h), float(prof[i])


def shot_target(grays, mattes=None, angle_max=6.0, strip_h=0.30, cy_bounds=CY_BOUNDS):
    """target pose of a strip on a shot: {angle, cy, strength, edge_angle, edge_y} from its first frames (luminance 0..1 HxW list, optional
    matte list)."""
    gx, gy = edge_field(grays, mattes)
    ea, strength = dominant_angle(gx, gy)
    angle = float(np.clip(ea, -angle_max, angle_max)) if strength >= MIN_STRENGTH else 0.0
    lo, hi = cy_bounds[0] + strip_h / 2.0, cy_bounds[1] + strip_h / 2.0              # the strip's lower boundary sits on the edge line
    ey, _ = best_edge_row(gx, gy, ea if strength >= MIN_STRENGTH else 0.0, lo, hi)
    cy = float(np.clip(ey - strip_h / 2.0, cy_bounds[0], cy_bounds[1]))
    return dict(angle=round(angle, 2), cy=round(cy, 4), strength=round(float(strength), 3), edge_angle=round(float(ea), 2), edge_y=round(ey, 4))


# ======================================================================================================== footage
def read_frames(path, n=8, gray=True, size=None):
    """first n frames of a video / image directory -> list of float32 0..1 HxW (gray) or HxWx3 RGB; [] when it cannot be read."""
    path = str(path)
    out = []
    try:
        if Path(path).is_dir():
            names = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in (".png", ".jpg"))
            frames = (cv2.imread(str(p), cv2.IMREAD_COLOR) for p in names[:n])
        else:
            cap = cv2.VideoCapture(path)

            def gen():
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        return
                    yield fr
            frames = gen()
        for fr in frames:
            if fr is None:
                break
            if size is not None and (fr.shape[1], fr.shape[0]) != tuple(size):
                fr = cv2.resize(fr, tuple(size), interpolation=cv2.INTER_AREA)
            out.append((cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if gray else fr[..., ::-1]).astype(np.float32) / 255.0)
            if len(out) >= n:
                break
    except Exception:                                     # noqa: BLE001
        return []
    return out


def shot_starts(piece):
    """[(sid, start_s)] of the piece's shots in order (compose layout; the crossfade of the piece is 0 for cut pieces)."""
    from tools.fusion import recipe_eval as RE
    lay, _ = RE.piece_layout(piece)
    fps = int(piece.get("fps", 24))
    return [(sid, L["start"] / fps) for sid, L in lay.items()]


def cut_targets(piece, root=None, n_frames=8, angle_max=6.0, strip_h=0.30):
    """{sid: target} from the first frames of every shot's plate (+ matte) in piece["type"]["plates"] (shot order). Shots whose files cannot
    be read are left out."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    ids = [sid for sid, _ in shot_starts(piece)]
    out = {}
    for sid, pl in zip(ids, (piece.get("type") or {}).get("plates") or []):
        def p(k):
            v = pl.get(k)
            return None if not v else (Path(v) if Path(v).is_absolute() else root / v)
        pp, mp = p("plate"), p("matte")
        grays = read_frames(pp, n_frames, True) if pp is not None and pp.exists() else []
        if not grays:
            continue
        mattes = read_frames(mp, n_frames, True) if mp is not None and mp.exists() else None
        out[sid] = shot_target(grays, mattes or None, angle_max, strip_h)
    return out


# ======================================================================================================== keyframes
def pose_keys(base, start, end, cuts, targets, ease_s=EASE_S, first=None):
    """keyframes [[t, dx, dy, dang]] of one strip group living from `start` to `end`. base = {cy, angle} of the layout's own strip;
    cuts = [(t_cut, sid)] (every cut of the piece); targets = {sid: {cy, angle}}; first = sid of the shot on screen at t = 0. A strip stands
    on the pose of the shot it appears in (it fades in, no ease); every later cut inside its life holds the old pose until the cut, then
    eases to the new one over `ease_s`."""
    def delta(tg):
        return [0.0, round(float(tg["cy"]) - float(base["cy"]), 4), round(float(tg["angle"]) - float(base["angle"]), 3)]
    cur = [0.0, 0.0, 0.0]
    sid0 = first
    for tc, sid in cuts:
        if tc - 0.06 <= start:                                            # the shot on screen when the strip appears
            sid0 = sid
    if sid0 in targets:
        cur = delta(targets[sid0])
    keys = [[round(start, 4)] + cur]
    for tc, sid in cuts:
        if start + 0.1 <= tc < end - 0.02 and sid in targets:
            new = delta(targets[sid])
            keys.append([round(tc, 4)] + cur)
            keys.append([round(min(tc + ease_s, end), 4)] + new)
            cur = new
    if round(end, 4) > keys[-1][0] + 1e-6:
        keys.append([round(end, 4)] + cur)
    return keys


def _band(group):
    """the strip of a shape group: first full-bleed rect -> (cx, cy, w, h, angle); None when the group has none."""
    for sh in group:
        if sh.get("kind") == "rect" and float(sh.get("w", 0.0)) >= FULL_BLEED_W:
            return sh
    return None


def _timeline(keys, col, ease="smooth"):
    ks = [[k[0], k[col]] for k in keys]
    return {"timeline": ks, "ease": ease}


def follow_ops(follow, ops, buffers=FOLLOW_BUFFERS):
    """`rigid_pose` ops that make the type follow the strips, one per group and type buffer (each is the identity outside its group's life):
    -> (new_ops, index to insert them at, wrap). They go right after the last flow_wake output among `buffers` (wm / wl), in place; without
    flow_wake the $type_matte / $type_line inputs are wrapped into _sr_type_* (wrap = {input: copy}, later ops must read the copies)."""
    if not follow:
        return [], 0, None
    last = max((j for j, o in enumerate(ops) if o.get("op") == "flow_wake" and o.get("out") in buffers), default=None)
    if last is not None:
        pairs, at, wrap = [(o["out"], o["out"]) for o in ops if o.get("op") == "flow_wake" and o.get("out") in buffers], last + 1, None
    else:
        pairs, at = [("type_matte", "_sr_type_matte"), ("type_line", "_sr_type_line")], 0
        wrap = dict(pairs)
    cur = {src: src for src, _ in pairs}
    new = []
    for g in follow:
        for src, out in pairs:
            new.append({"op": "rigid_pose", "out": out, "args": {"img": "$" + cur[src], "dx": _timeline(g["type_keys"], 1),
                                                               "dy": _timeline(g["type_keys"], 2), "angle": _timeline(g["type_keys"], 3),
                                                               "cx": g["pivot"][0], "cy": g["pivot"][1]}})
            cur[src] = out
    return new, at, wrap


# ======================================================================================================== weave
def _sub(obj, m):
    if isinstance(obj, str):
        return m.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _sub(v, m) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub(v, m) for v in obj]
    return obj


def weave(piece, music=None, chk=None, root=None, targets=None):
    """block strips_persist_across_cuts, compose side (idempotent; a piece without the block, without a strip, with fewer than 2 shots or
    without readable first frames is left as it is, the reason is in plan["strip_resnap"]). Call after weave_layout and weave_flow."""
    plan = piece.setdefault("plan", {})
    blk = next((b for b in plan.get("blocks", []) if b.get("name") == "strips_persist_across_cuts"), None)
    if blk is None:
        return piece
    if (plan.get("strip_resnap") or {}).get("groups"):
        return piece
    comp = piece.get("compose") or {}
    chains = ([comp["default"]] if comp.get("default") else []) + list((comp.get("shots") or {}).values())
    lsop = [o for ch in chains for o in ch.get("ops", []) if o.get("op") == "layout_shapes"]
    if not lsop:
        plan["strip_resnap"] = {"skipped": "no layout_shapes op (no layout with shapes)"}
        return piece
    starts = shot_starts(piece)
    first = starts[0][0] if starts else None
    cuts = [(round(t, 4), sid) for sid, t in starts[1:]]
    if not cuts:
        plan["strip_resnap"] = {"skipped": "one shot: no cuts"}
        return piece
    amax = float(blk.get("params", {}).get("angle_deg", 6.0))
    npers = int(blk.get("params", {}).get("persist_cuts", 1))
    extended = {}
    shapes0 = lsop[0]["args"]["shapes"]
    band0 = _band(shapes0)
    tg = targets if targets is not None else cut_targets(piece, root, angle_max=amax, strip_h=float(band0["h"]) if band0 else 0.30)
    if not tg:
        plan["strip_resnap"] = {"skipped": "no readable first frames of the shots (plates)"}
        return piece
    groups, order = [], []
    for sh in shapes0:                                                  # a strip group = the shapes of one word (same life)
        key = (sh.get("t0"), sh.get("t1"))
        if key not in order:
            order.append(key)
        groups.append(order.index(key))
    words = piece["type"]["words"]
    wi = [i for i, w in enumerate(words) if (w.get("hold_layout") or {}).get("shapes")]
    timing = plan.get("timing") or []
    info = []
    follow = []
    posed = {}
    for gi, (t0, t1) in enumerate(order):
        grp = [sh for sh, g in zip(shapes0, groups) if g == gi]
        band = _band(grp)
        if band is None:
            info.append({"t0": t0, "t1": t1, "moved": False, "why": "no full-bleed rect strip in this layout"})
            continue
        base = {"cy": float(band["cy"]), "angle": float(band.get("angle", 0.0))}
        word = words[wi[gi]] if gi < len(wi) else None
        tend = float(timing[wi[gi]]["end"]) if gi < len(wi) and wi[gi] < len(timing) and "end" in timing[wi[gi]] else float(t1)
        life_end = float(t1)
        after = [c for c in cuts if c[0] >= tend - 0.02]                 # cuts the strip survives after its line has gone
        if npers > 0 and len(after) >= 1:
            life_end = max(life_end, after[min(npers, len(after)) - 1][0] + EASE_S + 0.15)
            nxt = [float(o[0]) for o in order if o[0] is not None and float(o[0]) > float(t0)]
            life_end = min(life_end, (min(nxt) if nxt else 1e9), float(piece["audio"].get("cut", 1e9)) - 0.2)
        if life_end > float(t1) + 1e-6:
            extended[(t0, t1)] = round(life_end, 4)
        keys = pose_keys(base, float(t0), life_end, cuts, tg, first=first)
        pivot = [float(band["cx"]), float(band["cy"])]
        posed[(t0, t1)] = {"pivot": pivot, "keys": keys, "ease": "smooth"}
        tkeys = pose_keys(base, float(t0), min(tend, float(t1)), cuts, tg, first=first)
        tkeys.append([round(min(tend, float(t1)) + 0.04, 4), 0.0, 0.0, 0.0])
        if tkeys[0][0] > 0.0:
            tkeys.insert(0, [round(float(t0) - 0.04, 4), 0.0, 0.0, 0.0])
        follow.append({"pivot": pivot, "type_keys": tkeys})
        info.append({"t0": t0, "t1": t1, "moved": True, "word": word["text"] if word else None, "base": base,
                     "keys": keys, "snaps": sum(1 for c in cuts if float(t0) + 0.1 <= c[0] < life_end - 0.02), "life_end": round(life_end, 4)})
    if not posed:
        plan["strip_resnap"] = {"skipped": "no full-bleed rect strip", "groups": []}
        return piece
    for op in lsop:
        for sh in op["args"]["shapes"]:
            key = (sh.get("t0"), sh.get("t1"))
            p = posed.get(key)
            if p is not None and sh.get("kind") == "rect" and float(sh.get("w", 0.0)) >= FULL_BLEED_W:
                sh["pose"] = copy.deepcopy(p)
        for sh in op["args"]["shapes"]:                                   # the strip group (rects + their accent bar) lives on to the cut it survives
            if (sh.get("t0"), sh.get("t1")) in extended:
                sh["t1"] = extended[(sh["t0"], sh["t1"])]
    for ch in chains:
        ops = ch["ops"]
        if not any(o.get("op") == "layout_shapes" for o in ops):
            continue
        new, at, wrap = follow_ops(follow, ops)
        if not new:
            continue
        if wrap:                                                         # no flow_wake: later ops read the moved copies of the type inputs
            ops[at:] = [_sub(o, {"$" + k: "$" + v for k, v in wrap.items()}) for o in ops[at:]]
        ops[at:at] = new
    plan["strip_resnap"] = {"cuts": [list(c) for c in cuts], "targets": tg, "ease_s": EASE_S, "angle_deg": amax, "groups": info}
    return piece
