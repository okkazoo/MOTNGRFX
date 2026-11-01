"""Does every word stay inside the frame over its hold? (regression 2026-10-29, draft v014 r02 'Concrete Command': BREAK lost its K, CLAIM its C, STRIKE its E, OWN its O.)

Cause: the piece system's type camera `follow` / `hold_drift` frames the ink at ~0.89 of the frame width, then hold_curves composed the word 9 % of the frame off centre, alternating the
side per word, and pushed in 6 % over the hold; the design's `placement.wf` does not matter (the camera normalises the word box to the frame), so the placement caps could not catch it.

`word_extents(type_spec, size)` builds the type scene WITH its camera and projects every polygon of every word over [hit, release] (geometry only, no rasterisation: ~5 s for 4 words on the CPU);
`overflow(...)` -> the words whose box leaves the safe area; `guard(piece)` is the planner's remedy ladder (dry run of the camera, then the reader-safe `beat_orbit`).
"""
from __future__ import annotations

import copy
import os

import numpy as np

SAFE_MARGIN = 0.03      # a word's full ink box keeps this share of the frame (per side) free over its whole hold (the design rules ask 4 % for the box, the ink is ~6 % narrower)
STEP_S = 0.2            # sampling step over the hold (s); the hit and the release frame are always sampled


def _clean_spec(type_spec: dict) -> dict:
    ts = copy.deepcopy(type_spec)
    for w in ts.get("words", []):
        for k in ("entry", "exit"):
            if isinstance(w.get(k), dict):                       # contour-port keys do not move the word; the scene builds without plates
                del w[k]
        w.pop("via", None)
        w.pop("matte", None)
    return ts


def word_extents(type_spec: dict, size, step: float = STEP_S, scene=None) -> list[dict]:
    """per word (timeline order): {text, hit, release, t: [s..], x0, x1, y0, y1: arrays of frame FRACTIONS (0..1; < 0 / > 1 = off the frame)} over [hit, release]."""
    from mograph.render import cam3d
    from tools.fusion import raster
    from tools.fusion import spec as S
    size = tuple(int(v) for v in size)
    world = (cam3d.W, cam3d.H)
    try:
        sc = scene or S.build_scene(_clean_spec(type_spec), size=size)
        cam = sc.chain if sc.chain is not None else sc.seq.demo_camera()
        out = []
        for w in sc.info["words"]:
            t0, t1 = float(w["hit"]), float(w["release"])
            ts = sorted(set([t0, t1] + list(np.arange(t0, t1, step))))
            rows = []
            for t in ts:
                c = cam.at(t) if hasattr(cam, "at") else cam
                xs, ys = [], []
                for it in raster.collect(sc.seq.layers(t), c, size):
                    for xy, _Qc, _cl in it.polys:
                        xs += [float(xy[:, 0].min()), float(xy[:, 0].max())]
                        ys += [float(xy[:, 1].min()), float(xy[:, 1].max())]
                rows.append((min(xs), max(xs), min(ys), max(ys)) if xs else (np.nan,) * 4)
            a = np.array(rows, float)
            out.append(dict(text=w["text"], hit=t0, release=t1, t=ts, x0=a[:, 0] / size[0], x1=a[:, 1] / size[0], y0=a[:, 2] / size[1], y1=a[:, 3] / size[1]))
        return out
    finally:
        cam3d.set_world(*world)                                   # build_scene sets the global world plane; give it back


def overflow(ext: list[dict], margin: float = SAFE_MARGIN) -> list[dict]:
    """the words whose hold boxes leave [margin, 1 - margin] on either axis: [{text, worst: share of the frame outside the safe area, t}]"""
    bad = []
    for e in ext:
        over = np.nanmax(np.stack([margin - e["x0"], e["x1"] - (1 - margin), margin - e["y0"], e["y1"] - (1 - margin)]), axis=0)
        k = int(np.nanargmax(over))
        if over[k] > 1e-6:
            bad.append(dict(text=e["text"], worst=round(float(over[k]), 4), t=round(float(e["t"][k]), 2)))
    return bad


RISKY_CAMERAS = ("follow", "hold_drift")      # composed-hold cameras: the word sits ~9 % off centre on an alternating side, at ~0.89 of the frame width


def guard(piece: dict, fixes: list | None = None, size=None, only_risky: bool = False) -> list[str]:
    """in place. Measure the words' hold boxes with the piece's own type camera; a word outside the safe area with a composed-hold camera (RISKY_CAMERAS) -> the type camera becomes `beat_orbit` (the pre-diversity default, framed
    inside the safe area on every piece measured) with the piece system's params; measured again. -> notes (what was seen / changed). Pieces whose type scene cannot be built are skipped."""
    notes: list[str] = []
    ts = piece.get("type") or {}
    if not ts.get("words") or ts.get("camera") in (None, "demo"):
        return notes
    if only_risky and ts.get("camera") not in RISKY_CAMERAS:
        return notes
    size = tuple(size or (piece.get("sizes") or {}).get("type") or (576, 1024))
    try:
        bad = overflow(word_extents(ts, size))
    except Exception as e:                                         # noqa: BLE001  the type engine's own dry run reports build errors
        return [f"type frame check skipped: {type(e).__name__}: {e}"]
    if not bad:
        return notes
    cam0 = ts.get("camera")
    msg = ", ".join(f"{b['text']} {b['worst']:.0%} outside the safe area at {b['t']:g} s" for b in bad)
    if cam0 in RISKY_CAMERAS:
        from tools.fusion import piece_system as PS
        ts["camera"] = "beat_orbit"
        mo = PS.motion(ts.get("bpm"), int(ts.get("seed") or 0))
        ts["camera_params"] = {k: mo["suggest"][k] for k in ("read_dps", "intro_deg", "fast_dps", "elev", "bob")}
        ts.pop("camera_override", None)
        bad2 = overflow(word_extents(ts, size))
        notes.append(f"type camera {cam0!r} -> 'beat_orbit': with {cam0!r} the words left the frame ({msg})" + (f"; beat_orbit still has {len(bad2)} word(s) near the edge" if bad2 else ""))
    else:
        notes.append(f"type frame check: {msg} (camera {cam0!r} kept: a designed move)")
    if fixes is not None:
        fixes.extend(notes)
    return notes
