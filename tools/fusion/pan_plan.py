"""pan_plan.py - the planner / scene side of the directional pan transition (v006, user: "the text transitions through the camera travel were
unrelated to the camera movement ... start with simple top -> bottom / bottom -> top / left -> right / right -> left").

The camera travel IS the transition: word A holds, the camera travels in direction D, A leaves the frame on the opposite side exactly as a
world-locked object would, word B (placed one frame-span further along D in the same world) arrives as the camera settles, B's hit on a
downbeat. The footage plate is panned by the SAME on-screen displacement (compose op `plate_pan`, tools/fusion/plate_pan.py), taken from the
real camera, with directional motion blur proportional to the speed.

Spec keys (piece["type"], read by tools/fusion/spec.py):
    word    "transition": {"kind": "pan", "dir": "down", "beats": 2, ...}     the transition AFTER this word (to the next word)
    type    "transitions": [ ..., {"module": "pan" | "kind": "pan", "dir": "up", "beats": 3}, ... ]   (or the string "pan")
    type    "camera": {"move": "pan", "dir": "down" | ["down", "right"], "beats": 2}     every gap without its own transition becomes a pan
  keys of a pan: dir (down|up|right|left = the way the CAMERA travels; the picture slides the other way), beats (travel length, 2), dur (s,
  overrides beats), ease ('io' | 'soft'), gap (extra air between the frames, share of one frame-span, 0), fill (0.7, the word's frame share
  at its hold), exit_io / entry_io (the io style of A's exit / B's entry plays at the frame edge), land ('downbeat' | 'beat': where B lands),
  blur (motion-blur gain, 1.0), parallax (0..1, plate depth parallax, needs a depth pass; default 0 = one plane at the type's depth).

Timing (spec.plan_timing, which calls `boundaries`): A's hold is kept (>= its readable minimum), A's exit and B's form shrink to EDGE_S and
are frozen on the hold (WordModule.carry), the travel starts after A's hold and ENDS on B's hit, which lands on the first downbeat the
readable hold allows (grid.snap ceil). Words of a run (A -> B -> C joined by pans) are placed on one line, one frame-span apart.

Lego pieces: camera `mograph.modules.camera.pan_dir.PanDir`, transition `mograph.modules.transitions.pan.Pan` (hidden from the engine pool,
composes with any word module), plate op `plate_pan`, block `camera_pan_transition` (tools/fusion/blocks.py), `weave` (the plate ops +
the LTX camera phrase of the footage), `dir_from_flow` / `retarget` (pick D from the footage's own camera motion).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from mograph.modules.camera.pan_dir import DIRS, DIR_VEC, OPPOSITE, FILL, fit_dist, frame_span

EDGE_S = 0.05                 # A's exit / B's form window of a carried word (s): frozen on the hold, only here so the module timing stays valid
BEATS = 2                     # default travel length
DEFAULTS = dict(dir="down", beats=BEATS, dur=None, ease="io", gap=0.0, fill=FILL, exit_io=False, entry_io=False, land="downbeat", blur=1.0,
                parallax=0.0)
_NO_MATERIAL = ("flicker", "wipe")


# ================================================================================================== spec keys
def _norm(cfg, k):
    c = dict(DEFAULTS)
    c.update({kk: v for kk, v in dict(cfg or {}).items() if kk not in ("kind", "module")})
    d = c["dir"]
    if isinstance(d, (list, tuple)):
        d = d[k % len(d)]
    if d not in DIRS:
        raise ValueError(f"pan dir {d!r}: one of {DIRS}")
    c["dir"] = d
    c["beats"] = float(c["beats"])
    c["gap"] = float(c["gap"])
    c["fill"] = float(c["fill"])
    return c


def _is_pan(v):
    return v == "pan" or (isinstance(v, dict) and (v.get("kind") == "pan" or v.get("module") == "pan"))


def boundaries(spec, words=None):
    """{gap index i: pan config} of a type spec (the transition between word i and i + 1). Sources, in order of priority: the word's own
    `"transition": {"kind": "pan", ..}`, the `transitions` list entry, the piece camera `{"move": "pan", ..}` (every gap that has no explicit
    transition of its own)."""
    spec = spec or {}
    words = words if words is not None else (spec.get("words") or [])
    cam = spec.get("camera")
    default = cam if isinstance(cam, dict) and cam.get("move") == "pan" else None
    tspec = spec.get("transitions")
    out, n_pan = {}, 0
    for i in range(len(words) - 1):
        wt = words[i].get("transition")
        lt = tspec[i] if isinstance(tspec, list) and i < len(tspec) else None
        cfg = None
        if _is_pan(wt):
            cfg = dict(wt) if isinstance(wt, dict) else {}
        if _is_pan(lt):
            cfg = {**(dict(lt) if isinstance(lt, dict) else {}), **(cfg or {})}
        if cfg is None and default is not None and wt is None and not lt:
            cfg = {kk: v for kk, v in default.items() if kk != "move"}
        if cfg is not None:
            out[i] = _norm(cfg, n_pan)
            n_pan += 1
    return out


def runs(pans, n_words):
    """[(first word, last word)] of the maximal runs of words joined by pan gaps."""
    out, cur = [], None
    for i in sorted(pans):
        if cur and cur[1] == i:
            cur[1] = i + 1
        else:
            cur = [i, i + 1]
            out.append(cur)
    return [tuple(r) for r in out]


def travel_s(cfg, grid):
    if cfg.get("dur"):
        return float(cfg["dur"])
    return float(cfg["beats"]) * (grid.iv if grid else 0.5)


def word_box(w, W, H):
    """(pw, ph): the placement box of a word spec in world units (the rule spec.build_scene uses)."""
    p = dict(bleed=1.0)
    p.update(w.get("placement") or {})
    pw = float(p.get("w") or (float(p["wf"]) * W if p.get("wf") else W * p["bleed"]))
    ph = float(p.get("h") or (float(p["hf"]) * H if p.get("hf") else 1.08 * H * p["bleed"]))
    return pw, ph


def layout_offsets(words, pans, world, F=1000.0):
    """{word index: (dx, dy, run dist)} world offsets of every word of a pan run relative to the run's FIRST word: each next word sits one
    frame-span (at the run's hold distance, plus the pan's gap) further along the travel direction. The first word of a run is not listed."""
    W, H = world
    out = {}
    for (a, b) in runs(pans, len(words)):
        boxes = [word_box(words[i], W, H) for i in range(a, b + 1)]
        fill = pans[a]["fill"]
        D = max(fit_dist(pw, ph, F, fill) for pw, ph in boxes)
        cx = cy = 0.0
        for i in range(a, b):
            c = pans[i]
            span = frame_span(c["dir"], D, F, world) * (1.0 + c["gap"])
            vx, vy = DIR_VEC[c["dir"]]
            cx, cy = cx + vx * span, cy + vy * span
            out[i + 1] = (cx, cy, D)
    return out


def prepare_words(words, pans):
    """copies of the word specs with what a pan run needs: face-on placement (a pure translation), no screen offset (a layout offset would
    glide the picture on top of the camera), no animated axes (the axes warp only exists inside a word's own window), no contour ports on the
    pan sides, and io styles that carry material (flicker / wipe on a pan side become lines). -> (words, notes)."""
    words = [copy.deepcopy(w) for w in words]
    notes = []
    inrun = {i for (a, b) in runs(pans, len(words)) for i in range(a, b + 1)}
    for i in sorted(inrun):
        w = words[i]
        pl = dict(w.get("placement") or {})
        if any(abs(float(pl.get(k, 0.0))) > 1e-9 for k in ("yaw", "pitch", "roll")):
            notes.append(f"pan: word {i} placed face-on (yaw / pitch / roll 0)")
        pl.update(yaw=0.0, pitch=0.0, roll=0.0)
        w["placement"] = pl
        if w.get("screen_offset") and any(abs(float(x)) > 1e-9 for x in w["screen_offset"]):
            notes.append(f"pan: word {i} screen_offset dropped")
        w.pop("screen_offset", None)
        if w.get("axes"):
            notes.append(f"pan: word {i} axes animation off")
            w.pop("axes", None)
        keep_out = i in pans and (pans[i].get("exit_io"))
        keep_in = (i - 1) in pans and (pans[i - 1].get("entry_io"))
        for side, key, carried, keep in (("out", "form_out", i in pans, keep_out), ("in", "form_in", (i - 1) in pans, keep_in)):
            if not carried:
                continue
            st = w.get(key)
            nm = (st[0] if isinstance(st, (list, tuple)) else st) or "lines"
            if not keep or nm in _NO_MATERIAL:
                if nm != "lines":
                    notes.append(f"pan: word {i} {key} {nm} -> lines")
                w[key] = "lines"
        for key, port in (("exit", i in pans), ("entry", (i - 1) in pans)):
            if port and isinstance(w.get(key), dict):
                notes.append(f"pan: word {i} contour {key} dropped")
                w[key] = "down" if key == "exit" else "up"
        if i in pans:
            w.pop("via", None)
    return words, notes


# ================================================================================================== scene side
def scene_runs(seq):
    """[(first word idx, last word idx, t0, t1, [pan transitions])] of the built sequence."""
    from mograph.modules.core import WordModule
    ws = [(it, a, b) for it, a, b in seq.timeline if isinstance(it, WordModule)]
    trs = [(it, a, b) for it, a, b in seq.timeline if not isinstance(it, WordModule)]
    out, cur = [], None
    for k, (tr, a, b) in enumerate(trs):
        if getattr(tr, "name", "") == "pan":
            if cur and cur[1] == k:
                cur[1] = k + 1
                cur[4].append(tr)
            else:
                cur = [k, k + 1, 0.0, 0.0, [tr]]
                out.append(cur)
    return [(r[0], r[1], ws[r[0]][1], ws[r[1]][2], r[4]) for r in out]


def window_overrides(seq):
    """{(t0, t1): PanDir} for CameraChain.auto(override=..): one PanDir over every run of pan-joined words."""
    from mograph.modules.camera.pan_dir import PanDir
    return {(t0, t1): PanDir(fill=getattr(trs[0], "fill", FILL), ease=trs[0].ease) for (_, _, t0, t1, trs) in scene_runs(seq)}


def pan_windows(seq):
    """[(t0, t1, dir, word a, word b)] of every pan transition on the built timeline."""
    from mograph.modules.core import WordModule
    wi = -1
    out = []
    for it, a, b in seq.timeline:
        if isinstance(it, WordModule):
            wi += 1
        elif getattr(it, "name", "") == "pan":
            out.append((float(a), float(b), it.dir, wi, wi + 1))
    return out


def plate_track(scene):
    """The on-screen displacement of the plate that rides with the type, from the REAL camera of a built scene: per pan window, the screen
    motion of the point of the type's plane that sits at the screen centre when the pan starts (a plate locked to the type's depth). World-frame
    px (x right, y down), cumulative over the pans, constant between them. -> dict(fps, nf, world, dx, dy (nf,), vx, vy (px / frame), amt (0..1 bell
    over each travel), windows [(t0, t1, dir)]) or None without a camera."""
    chain = scene.chain
    if chain is None:
        return None
    seq, fps, nf = scene.seq, float(scene.fps), int(scene.nf)
    wins = pan_windows(seq)
    words = [it for it, _, _ in seq.timeline if hasattr(it, "placement")]
    t = np.arange(nf) / fps
    dx, dy = np.zeros(nf), np.zeros(nf)
    vx, vy = np.zeros(nf), np.zeros(nf)
    amt = np.zeros(nf)
    base = np.zeros(2)
    for (t0, t1, d, ia, ib) in wins:
        cam0 = chain.at(t0)
        z0 = float(words[ia].placement.origin[2])
        lam = (z0 - cam0.eye[2]) / cam0.f[2]
        P = cam0.eye + lam * cam0.f
        s0 = cam0.project(P)[0][0]

        def disp(tt):
            return chain.at(min(max(tt, t0), t1)).project(P)[0][0] - s0
        for i in range(nf):
            if t[i] >= t0 - 1e-9:
                d_ = disp(t[i])
                if t[i] > t1:
                    d_ = disp(t1)
                dx[i], dy[i] = base[0] + d_[0], base[1] + d_[1]
                if t[i] <= t1:
                    v = (disp(t[i] + 0.5 / fps) - disp(t[i] - 0.5 / fps))
                    vx[i], vy[i] = v
                    amt[i] = math.sin(math.pi * min(max((t[i] - t0) / max(t1 - t0, 1e-9), 0.0), 1.0))
        base = base + disp(t1)
    return dict(fps=fps, nf=nf, world=list(scene.world), dx=dx, dy=dy, vx=vx, vy=vy, amt=amt, windows=[(w[0], w[1], w[2]) for w in wins])


def type_motion(scene, word_idx, frames=None):
    """px (world frame) the centre of word `word_idx`'s box moves on screen from frame 0, per frame (project of the placement origin through the
    chain): the type side of the 'type and plate move together' check. -> (nf, 2); NaN where the point is behind the camera."""
    seq, chain, fps, nf = scene.seq, scene.chain, float(scene.fps), int(scene.nf)
    words = [it for it, _, _ in seq.timeline if hasattr(it, "placement")]
    P = words[word_idx].placement.origin[None, :]
    out = np.zeros((nf, 2))
    for i in range(nf):
        cam = chain.at(i / fps)
        xy, z = cam.project(P)
        out[i] = xy[0] if z[0] > 1.0 else np.nan
    return out - out[0]


# ================================================================================================== compose weave
def _timeline(t, v, fps, windows, pad=1):
    """sparse [[t, v], ..] keys: every frame inside (and one around) each window, the two ends; linear in between is exact (constant there)."""
    n = len(t)
    keep = {0, n - 1}
    for (t0, t1, *_r) in windows:
        a, b = int(math.floor(t0 * fps)) - pad, int(math.ceil(t1 * fps)) + pad
        keep.update(range(max(a, 0), min(b, n - 1) + 1))
    return [[round(float(t[i]), 5), round(float(v[i]), 6)] for i in sorted(keep)]


def _swap(obj, mp):
    if isinstance(obj, str):
        return mp.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _swap(v, mp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap(v, mp) for v in obj]
    return obj


PAN_OPS = ("plate_pan",)


def settle_wake(ops, windows, base):
    """flow_wake must not drag the type while the camera carries it: its `settle` timeline (1 = settled) goes to 1 over every pan travel (and a
    0.12 s ramp each side). `base` {op out: the original settle timeline} makes it repeatable. -> ops (copied)."""
    ops = copy.deepcopy(ops)
    for o in ops:
        st = o.get("args", {}).get("settle")
        if o.get("op") != "flow_wake" or not isinstance(st, dict) or "timeline" not in st:
            continue
        orig = base.setdefault(o.get("out", "w"), copy.deepcopy(st["timeline"]))
        xs, vs = [k[0] for k in orig], [k[1] for k in orig]
        new = [list(k) for k in orig if not any(a - 0.12 <= k[0] <= b + 0.12 for a, b in windows)]
        for a, b in windows:
            new += [[round(a - 0.12, 4), float(np.interp(a - 0.12, xs, vs))], [round(a, 4), 1.0], [round(b, 4), 1.0],
                    [round(b + 0.12, 4), float(np.interp(b + 0.12, xs, vs))]]
        o["args"]["settle"] = dict(st, timeline=sorted(new, key=lambda k: k[0]))
    return ops


def unsettle_wake(ops, base):
    for o in ops:
        if o.get("op") == "flow_wake" and o.get("out", "w") in base and isinstance(o["args"].get("settle"), dict):
            o["args"]["settle"] = dict(o["args"]["settle"], timeline=copy.deepcopy(base[o.get("out", "w")]))
    return ops


def unweave_ops(ops):
    """the chain without the plate_pan ops, the readers pointed back at the original plate buffers."""
    back = {}
    for o in ops:
        if o.get("op") in PAN_OPS and o.get("args", {}).get("img"):
            back["$" + o["out"]] = o["args"]["img"]
    keep = [o for o in ops if o.get("op") not in PAN_OPS]
    return [dict(o, args=_swap(o.get("args", {}), back)) for o in keep]


def track_ops(track, cfg=None, depth=False, matte=False, buffers=("footage",)):
    """compose ops that pan the plate buffers (`$footage`, optionally `$matte` / `$depth`) by the track. Displacements are frame fractions.
    cfg['plate'] = gain of the plate's travel (1 = the plate moves exactly like the type's plane; 0 = no plate pan, for footage that the LTX
    control video already makes travel with the type)."""
    W, H = track["world"]
    fps = track["fps"]
    t = np.arange(track["nf"]) / fps
    w = track["windows"]
    cfg = cfg or {}
    g = float(cfg.get("plate", 1.0))
    args = dict(dx={"timeline": _timeline(t, g * track["dx"] / W, fps, w)}, dy={"timeline": _timeline(t, g * track["dy"] / H, fps, w)},
                vx={"timeline": _timeline(t, g * track["vx"] / W, fps, w)}, vy={"timeline": _timeline(t, g * track["vy"] / H, fps, w)},
                seam_amt={"timeline": _timeline(t, track["amt"], fps, w)}, mode="wrap", shutter=0.5 * float(cfg.get("blur", 1.0)),
                seam=0.16, parallax=float(cfg.get("parallax", 0.0)))
    ops = []
    for b in buffers:
        a = copy.deepcopy(args)
        a["img"] = "$" + b
        if b != "footage":                              # matte / depth buffers: same displacement, no blur taps wasted on the seam blend
            a["seam_amt"] = 0.0
        if b == "footage" and depth and float(cfg.get("parallax", 0.0)) > 0:
            a["depth"] = "$depth"
        ops.append({"op": "plate_pan", "out": b + "_p", "args": a})
    return ops


def weave(piece, music=None, chk=None, size=None, scene=None, track=None):
    """add the plate pan to every compose chain of the piece (in place; returns it). Needs a built scene WITH its camera (built here from
    piece['type'] unless `scene` / `track` is given: ~30 s for a full piece, only for pieces that have a pan). Idempotent: an existing plate_pan
    is replaced. The shots' LTX prompts get the matching camera phrase."""
    t = piece.get("type") or {}
    plan = piece.setdefault("plan", {})
    pans = boundaries(t)
    if not pans:
        return piece
    if track is None:
        if scene is None:
            scene = build_camera_scene(piece, size)
        track = plate_track(scene)
    comp = piece.get("compose") or {}
    chains = [("default", comp["default"])] if comp.get("default") else []
    chains += list((comp.get("shots") or {}).items())
    n = 0
    sbase = plan.setdefault("pan", {}).setdefault("settle_base", {})
    wins = [(w[0], w[1]) for w in track["windows"]]
    for name, ch in chains:
        ops = unsettle_wake(unweave_ops(ch.get("ops", [])), sbase.get(name, {}))
        used = _refs(ops)
        bufs = ["footage"] + [b for b in ("matte", "depth") if b in used]
        cfg0 = pans[min(pans)]
        new = track_ops(track, cfg0, depth="depth" in used, buffers=bufs)
        mp = {"$" + b: "$" + b + "_p" for b in bufs}
        ops = settle_wake(ops, wins, sbase.setdefault(name, {}))
        ch["ops"] = new + [dict(o, args=_swap(o.get("args", {}), mp)) for o in ops]
        n += 1
    fps = float(track["fps"])
    downs = np.asarray(t.get("downbeats") or [], float)
    rows = []
    for (t0, t1, d) in track["windows"]:
        hit = t1 + EDGE_S                                                       # B's hit = the end of the travel + B's frozen form window
        row = dict(dir=d, t0=round(t0, 3), t1=round(t1, 3), hit=round(hit, 3), travel_s=round(t1 - t0, 3))
        if len(downs):
            row["d_down_f"] = round(float((hit - downs[np.argmin(np.abs(downs - hit))]) * fps), 2)
        rows.append(row)
    plan["pan"].update(boundaries=rows, chains=n, peak_px=round(float(np.hypot(track["vx"], track["vy"]).max()), 2),
                       total_dx=round(float(track["dx"][-1]), 1), total_dy=round(float(track["dy"][-1]), 1))
    match_shots(piece, [(w[0], w[1], w[2]) for w in track["windows"]])
    return piece


def _refs(ops):
    acc = set()

    def walk(o):
        if isinstance(o, str) and o.startswith("$"):
            acc.add(o[1:])
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk([o.get("args", {}) for o in ops])
    return acc


def build_camera_scene(piece, size=None):
    """spec.build_scene of the piece's type spec WITH its camera (the plate track needs the real chain)."""
    from tools.fusion import plan_rules as pr
    from mograph.render import cam3d
    sp = pr._spec()
    size = tuple(size) if size else tuple(pr.orient(pr.DRY_SIZE, pr.format_of(piece)))
    ts = copy.deepcopy(piece["type"])
    for w in ts.get("words", []):
        for k in ("entry", "exit"):
            if isinstance(w.get(k), dict):
                del w[k]
        w.pop("via", None)
        w.pop("matte", None)
    world = (cam3d.W, cam3d.H)
    try:
        return sp.build_scene(ts, size=size)
    finally:
        cam3d.set_world(*world)


# ================================================================================================== LTX phrase / footage match
_ADJ = {"calm": "slow", "mid": "steady", "kinetic": "fast"}
_PHRASE = {"down": "camera tilt down", "up": "camera tilt up", "right": "camera pan right", "left": "camera pan left"}


def ltx_phrase(direction, tier="mid"):
    """the LTX prompt clause for a camera that travels in `direction` (down = tilt down, the picture slides up; right = pan right ...)."""
    if direction not in _PHRASE:
        raise ValueError(f"direction {direction!r}: one of {DIRS}")
    return f"one {_ADJ.get(tier, 'steady')} continuous {_PHRASE[direction]}"


def _motion_texts():
    from tools.fusion import plan_rules as pr
    out = set()
    for pool in (pr._MOTION, pr._SWEEP):
        for lst in pool.values():
            out.update(lst)
    return out


def match_shots(piece, windows):
    """the shots whose time span holds a pan travel get a prompt that names the matching camera move (the planner's own motion clause is
    replaced); a shot with two pans in different directions keeps the first. -> [(shot id, dir)]."""
    from tools.fusion import plan_rules as pr
    tier = (piece.get("plan") or {}).get("tier", "mid")
    try:
        lay = pr.layout(piece)
    except Exception:                                               # noqa: BLE001
        return []
    fps = float(piece.get("fps", 24))
    motions = _motion_texts()
    done = {}
    for (t0, t1, d) in windows:
        mid = 0.5 * (t0 + t1)
        for s in piece.get("shots", []):
            L = lay.get(s["id"])
            if not L or s["id"] in done:
                continue
            if L["start"] / fps - 1e-6 <= mid < (L["start"] + L["frames"]) / fps:
                parts = [p.strip() for p in str(s.get("prompt", "")).split(", ")]
                parts = [p for p in parts if p and p not in motions and "camera" not in p.lower()]
                ins = max(len(parts) - 1, 0) if parts and parts[-1] == "cinematic" else len(parts)
                parts.insert(ins, ltx_phrase(d, tier))
                s["prompt"] = ", ".join(parts)
                done[s["id"]] = d
    if done:
        piece.setdefault("plan", {}).setdefault("pan", {})["ltx"] = done
    return list(done.items())


def dir_from_flow(flow, fps, t0, t1, start=0, min_px=0.15):
    """The direction the CAMERA travelled in the footage over [t0, t1] s (piece time; `start` = the piece frame the clip starts at) from its
    optical flow ([T, h, w, 2], px / frame, x right / y down; footage_flow.flow_for): the median flow per frame is the scene's motion, the
    camera moved the other way (content slides up = camera moved down). -> (dir | None, px per frame along it, confidence 0..1) where the
    confidence is the share of the dominant axis in the whole motion; dir None when the clip hardly moves (< min_px)."""
    a = max(int(round(t0 * fps)) - start, 0)
    b = min(int(round(t1 * fps)) - start, int(flow.shape[0]))
    if b <= a:
        return None, 0.0, 0.0
    fl = flow[a:b]
    fl = fl.cpu().numpy() if hasattr(fl, "cpu") else np.asarray(fl)
    m = np.median(fl[:, ::4, ::4].reshape(fl.shape[0], -1, 2).astype(np.float32), axis=1).mean(0)       # scene motion (px / frame)
    cam = -m
    ax, ay = abs(float(cam[0])), abs(float(cam[1]))
    tot = ax + ay
    if max(ax, ay) < min_px:
        return None, max(ax, ay), 0.0
    if ay >= ax:
        return ("down" if cam[1] > 0 else "up"), ay, ay / tot
    return ("right" if cam[0] > 0 else "left"), ax, ax / tot


def retarget(piece, flows, music=None, chk=None, size=None, min_conf=0.55):
    """pick every pan's direction from the footage's own camera motion in its transition window (flows: {shot id: flow array [T,h,w,2]}), write
    it into the words' `transition.dir`, re-weave the plate (the track is rebuilt from the new camera). A pan whose window has no clear motion
    keeps its planned direction. -> [(gap, old, new, confidence)]."""
    from tools.fusion import plan_rules as pr
    t = piece["type"]
    fps = float(piece.get("fps", 24))
    pans = boundaries(t)
    if not pans:
        return []
    tm = pr._spec().plan_timing(copy.deepcopy(t["words"]), t, pr.make_grid({"bpm": t["bpm"], "beats": t["beats"], "downbeats": t["downbeats"]},
                                                                         float(t["duration"])))
    lay = pr.layout(piece)
    changed = []
    for i, cfg in pans.items():
        t0, t1 = tm[i]["end"], tm[i + 1]["hit"]
        mid = 0.5 * (t0 + t1)
        sid = next((s["id"] for s in piece["shots"] if lay[s["id"]]["start"] / fps <= mid < (lay[s["id"]]["start"] + lay[s["id"]]["frames"]) / fps), None)
        if sid is None or sid not in flows:
            continue
        d, px, conf = dir_from_flow(flows[sid], fps, t0, t1, start=lay[sid]["start"])
        if d is None or conf < min_conf:
            continue
        changed.append((i, cfg["dir"], d, round(conf, 2)))
        w = t["words"][i]
        tr = dict(w["transition"]) if isinstance(w.get("transition"), dict) else dict(kind="pan")
        tr.update(kind="pan", dir=d)
        w["transition"] = tr
        lt = t.get("transitions")
        if isinstance(lt, list) and i < len(lt) and _is_pan(lt[i]):
            lt[i] = dict(lt[i], dir=d) if isinstance(lt[i], dict) else dict(module="pan", dir=d)
    if changed:
        weave(piece, music, chk, size)
        piece.setdefault("plan", {}).setdefault("pan", {})["retargeted"] = [list(c) for c in changed]
    return changed
