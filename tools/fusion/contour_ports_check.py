"""contour_ports_check: render the contour-port check strip on the box (CPU type engine, 360x640 over the real P14 plate).

    python -m tools.fusion.contour_ports_check --plates DIR --out DIR [--set ports|via|matte] [--size 360x640] [--procs 10]

DIR holds s1.mp4 / s1_matte.mkv / s1_depth.mkv (+ s2.* : the two shots are joined, 12 s). Sets:
  ports  four gallery modules (lanes, pipework, slats, hollow_tubes) enter from the subject outline and leave along a depth iso-line /
         around the outline; the strands between two words travel between the contours (twist / fan)
  via    A -> contour ride -> B on the matte outline / depth iso / luma edges (dir up | around | along)
  matte  the five matte treatments (front, window, fill, wipe, glow) on the type of the ports set, plate + matte only
Writes <out>/<set>/frames/*.png, <set>_strip.png (a contact sheet), <set>.mp4, <set>.json. Frame rendering is on the box only (CLAUDE.md rule 4)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mograph.modules import Sequence, contour_ports as cp  # noqa: E402
from mograph.modules.sequence import _draw  # noqa: E402
from mograph.render import cam3d  # noqa: E402
from tools.fusion import spec as S_  # noqa: E402
from tools.fusion.sequence import FusionSequence  # noqa: E402

GLOB = {}

PAL = {"bg": [0, 0, 0], "fg": [255, 240, 215], "accents": [[255, 205, 140], [140, 190, 255], [255, 255, 255]]}


def base_spec(words, duration=12.0, fps=24, camera="beat_orbit"):
    return {"fps": fps, "duration": duration, "bpm": 240.0, "seed": 14, "energy": 0.9, "mood": "driving", "tier": "kinetic",
            "size": [576, 1024], "palette": PAL, "camera": camera,
            "camera_params": {"read_dps": 6.0, "intro_deg": 40.0, "fast_dps": 60.0, "elev": 5.0, "bob": 1.2},
            "line": {"px": 4.0}, "words": words, "transitions": ["twist", "fan", "twist"],
            "beats": [round(0.25 * i, 3) for i in range(100)], "downbeats": [round(1.0 * i, 3) for i in range(25)], "bar": 4}


def _w(module, text, t0, t1, form, exit_, yaw, font, lines=None, **ports):
    d = {"module": module, "text": text, "font": font, "t0": t0, "t1": t1, "form_s": form, "exit_s": exit_, "placement": {"yaw": yaw}}
    if lines:
        d["lines"] = lines
    d.update(ports)
    return d


def _p(port, dir_, lead=0.55, **kw):
    return dict(port=port, dir=dir_, lead=lead, glow=0.9, pulse=120, **kw)


SETS = {
    "ports": [
        _w("lanes", "RISE", 0.0, 2.6, 1.0, 0.9, -22, "FacadeVF.ttf", entry=_p("matte_outline", "up"), exit=_p("depth_iso", "up")),
        _w("pipework", "UP", 3.1, 5.7, 1.0, 0.9, 22, "Blackout", entry=_p("matte_outline", "along"), exit=_p("matte_outline", "around")),
        _w("slats", "INTO", 6.2, 8.8, 1.0, 0.9, -22, "FacadeVF.ttf", entry=_p("matte_outline", "up"), exit=_p("depth_iso", "around")),
        _w("hollow_tubes", "THE LIGHT", 9.3, 11.9, 1.0, 0.9, 20, "Blackout", lines=["THE", "LIGHT"], entry=_p("matte_outline", "around"),
           exit=_p("matte_outline", "up", persist=0.0)),
    ],
    "via": [
        _w("lanes", "RISE", 0.0, 2.1, 0.6, 0.5, -22, "FacadeVF.ttf", via=_p("matte_outline", "around", lead=1.0)),
        _w("slats", "UP", 3.4, 5.5, 0.6, 0.5, 22, "Blackout", via=_p("depth_iso", "up", lead=1.0)),
        _w("pipework", "INTO", 6.8, 8.9, 0.6, 0.5, -22, "FacadeVF.ttf", via=_p("luma_edges", "along", lead=1.0)),
        _w("hollow_tubes", "THE LIGHT", 10.2, 12.3, 0.6, 0.5, 20, "Blackout", lines=["THE", "LIGHT"]),
    ],
}
SETS["matte"] = [dict(w) for w in SETS["ports"]]
for _w_, _m in zip(SETS["matte"], ("front", "window", "fill", "wipe")):
    _w_["entry"], _w_["exit"], _w_["matte"] = "up", "down", _m
    _w_["module"], _w_["text"], _w_["font"] = "lanes", _m.upper(), "Blackout"        # solid words, named by their treatment
    _w_.pop("lines", None)


# ================================================================================================== scene
def load_plate(d, fps=24.0):
    parts = []
    for s in ("s1", "s2"):
        if os.path.exists(os.path.join(d, s + ".mp4")):
            parts.append(cp.PlateData.from_files(plate=os.path.join(d, s + ".mp4"), matte=os.path.join(d, s + "_matte.mkv"),
                                                 depth=os.path.join(d, s + "_depth.mkv"), fps=fps))
    return cp.PlateData.concat(parts) if len(parts) > 1 else parts[0]


def build(set_name, plates, size, fps=24):
    words = copy.deepcopy(SETS[set_name])
    data = load_plate(plates, fps)
    plain = []
    for w in words:                                                # module-side strings for build_scene; ports are applied after
        e, x, ports = cp.split_port_spec(w)
        v = dict(w, entry=e, exit=x)
        v.pop("via", None), v.pop("matte", None)
        plain.append(v)
    spec = base_spec(plain, duration=data.n / fps, fps=fps)
    scene = S_.build_scene(spec, size=(576, 1024))
    ctx = cp.ContourContext(data, world=scene.world, n=192, beats=spec["beats"])
    ctx.bind_camera(scene.chain)
    for w in words:                                                # build every track before the workers fork
        for key in ("entry", "exit", "via"):
            p = cp.split_port_spec(w)[2][key]
            if p:
                cp.ContourPort.parse(ctx, p).track
    items = cp.apply_ports(list(scene.seq.items), words, ctx)
    seq = FusionSequence(items, scene.palette, 0.0)
    seq.tail = max(0.0, spec["duration"] - seq.duration)
    seq.warps = [None] * len(seq.timeline)
    cp.attach(seq, ctx)
    chain = S_.build_camera(seq, spec)
    ctx.bind_camera(chain)
    return dict(scene=scene, seq=seq, chain=chain, ctx=ctx, words=words, spec=spec, data=data)


def treat_ops(words, seq, spec):
    """matte treatments of the `matte` set as compose ops over the plate (hit .. release of every word)."""
    from tools.fusion import matte_treatments as mt
    wins = []
    for it, a, b in seq.timeline:
        if hasattr(it, "timing") and hasattr(it, "placement"):
            wins.append((a + it.timing.form, a + it.timing.form + it.timing.hold))
    tr = mt.from_words(words, windows=wins)
    for t in tr:
        t["reveal"] = 0.6
    return mt.ops_for(tr, plate="$foot", matte="$m", type_a="$type_a", type_rgb=[1.0, 0.92, 0.78])


# ================================================================================================== frames
def _render_type(t):
    G = GLOB
    seq, chain, size = G["seq"], G["chain"], G["size"]
    return _draw(seq.layers(t), chain.at(t), seq.bg, size, 2)


def _frame(i):
    G = GLOB
    t = i / G["fps"]
    data, size = G["data"], G["size"]
    plate = data.plate[min(i, data.n - 1)].astype(np.float32) / 255.0
    if (plate.shape[1], plate.shape[0]) != tuple(size):
        plate = cv2.resize(plate, tuple(size), interpolation=cv2.INTER_AREA)
    ty = _render_type(t).astype(np.float32) / 255.0
    if G["set"] == "matte":
        from tools.fusion import compose
        m = data.matte[min(i, data.n - 1)].astype(np.float32) / 255.0
        m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_LINEAR) if m.shape[::-1] != tuple(size) else m
        ta = ty.max(-1)
        buf = dict(footage=plate, foot=plate, image=plate * 0.85, m=m, type_a=np.clip(ta * 1.6, 0, 1), t=t, frame_idx=i, u=0.0,
                   near=np.zeros_like(ta))
        reg = compose._ops_registry()
        from tools.fusion import recipe_eval as RE

        def res(v):
            if isinstance(v, str) and v.startswith("$"):
                return buf[v[1:]]
            if isinstance(v, dict) and "timeline" in v:
                return compose._timeline(v, t)
            if isinstance(v, dict) and "curve" in v:
                return 0.5 + 0.5 * np.exp(-5.0 * ((t * 2.0) % 1.0))
            if isinstance(v, list):
                return [res(x) for x in v]
            return v
        for op in G["ops"]:
            buf[op.get("out", "image")] = reg[op["op"]](**{k: res(v) for k, v in op.get("args", {}).items()})
        out = buf["image"]
    else:
        out = 1.0 - (1.0 - plate * 0.62) * (1.0 - ty)                    # type added over a dimmed plate (screen)
    u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(u8[..., ::-1])
    cv2.putText(bgr, f"{t:5.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(G["frames"], f"{i:05d}.png"), bgr)
    return i


def render(set_name, plates, out, size=(360, 640), procs=10, fps=24, max_frames=None):
    from tools.fusion import compose
    compose.set_backend("numpy")                     # forked workers: no CUDA contexts
    S = build(set_name, plates, size, fps)
    nf = min(int(round(S["spec"]["duration"] * fps)), S["data"].n)
    if max_frames:
        nf = min(nf, max_frames)
    d = os.path.join(out, set_name)
    frames = os.path.join(d, "frames")
    os.makedirs(frames, exist_ok=True)
    GLOB.update(seq=S["seq"], chain=S["chain"], data=S["data"], size=tuple(size), fps=fps, set=set_name, frames=frames,
                ops=treat_ops(S["words"], S["seq"], S["spec"]) if set_name == "matte" else None)
    cam3d.set_world(*S["scene"].world)
    t0 = time.perf_counter()
    if procs <= 1:
        for i in range(nf):
            _frame(i)
    else:
        with get_context("fork").Pool(procs) as pool:
            for _ in pool.imap_unordered(_frame, range(nf), chunksize=2):
                pass
    wall = time.perf_counter() - t0
    mp4 = os.path.join(out, f"{set_name}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(frames, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", mp4], check=True)
    strip = make_strip(frames, nf, S, os.path.join(out, f"{set_name}_strip.png"), fps)
    info = dict(set=set_name, frames=nf, size=list(size), wall_s=round(wall, 1), timeline=[
        dict(item=getattr(it, "name", type(it).__name__), start=round(a, 2), end=round(b, 2)) for it, a, b in S["seq"].timeline],
        ports=cp.available(S["data"])["ports"], strip=strip, mp4=mp4)
    with open(os.path.join(out, f"{set_name}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))
    return info


def make_strip(frames, nf, S, path, fps, cols=8):
    """contact sheet: 2 rows x `cols` frames spread over the clip, with the active item named under each."""
    idx = np.linspace(0, nf - 1, 2 * cols).round().astype(int)
    tiles = []
    for i in idx:
        im = cv2.imread(os.path.join(frames, f"{i:05d}.png"))
        t = i / fps
        it, a, b = next(((x, y, z) for x, y, z in S["seq"].timeline if t < z), S["seq"].timeline[-1])
        nm = getattr(it, "name", "")
        if hasattr(it, "inner"):
            nm = f"{it.inner.name}+port"
        lab = np.zeros((22, im.shape[1], 3), np.uint8)
        cv2.putText(lab, nm[:30], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, lab]))
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(2)]
    cv2.imwrite(path, np.vstack(rows))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="ports", choices=list(SETS))
    ap.add_argument("--size", default="360x640")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.lower().split("x"))
    render(a.set, a.plates, a.out, (w, h), a.procs, 24, a.max_frames)


if __name__ == "__main__":
    main()
