"""multiplane_check: the plate as depth cards under gallery camera chains, type between the cards (box only: torch GPU warp + CPU type).

    python -m tools.fusion.multiplane_check --plates DIR --out DIR [--size 360x640] [--procs 10] [--cards 4] [--max-frames N]

DIR = s1.mp4 / s1_matte.mkv / s1_depth.mkv (+ s2.*). Four words sit between the cards (spec "slot"); camera = push_through on word 0 and orbit_even
on word 2 (gallery CameraChain moves, joined by follow / hold_drift). Stage A (CPU pool): the type of every frame as premultiplied colour,
coverage and camera depth. Stage B (GPU): cards warped by the chain camera, type composited per pixel by depth. Writes <out>/multiplane_strip.png,
multiplane.mp4, multiplane.json (+ frames/)."""
from __future__ import annotations

import argparse
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

from mograph.modules import contour_ports as cp  # noqa: E402
from mograph.modules.sequence import _draw  # noqa: E402
from mograph.render import cam3d, multiplane as MP  # noqa: E402
from tools.fusion import spec as S_, raster  # noqa: E402
from tools.fusion.contour_ports_check import PAL, load_plate  # noqa: E402
from tools.fusion.sequence import FusionSequence  # noqa: E402

GLOB = {}
SUBJECT_Z = 1000.0


def words(cards_cfg):
    def w(module, text, t0, t1, slot, yaw, font, **kw):
        d = {"module": module, "text": text, "font": font, "t0": t0, "t1": t1, "form_s": 1.0, "exit_s": 0.9, "slot": slot,
             "placement": {"yaw": yaw}}
        d.update(kw)
        return d
    return [
        w("lanes", "RISE", 0.0, 3.4, None, -12, "FacadeVF.ttf", entry={"port": "matte_outline", "dir": "up", "lead": 0.6, "glow": 0.9}),
        w("pipework", "UP", 3.8, 6.4, 1, 12, "Blackout"),
        w("slats", "INTO", 6.8, 9.6, 2, -12, "FacadeVF.ttf"),
        w("hollow_tubes", "THE LIGHT", 10.0, 12.3, 0, 10, "Blackout", lines=["THE", "LIGHT"]),
    ]


CAMERA = {"chain": [{"module": "push_through", "word": 0, "to_next": False, "beyond": 0.04, "approach": 0.5},
                    {"module": "orbit_even", "word": 2, "sweep": 50}],
          "plate": "multiplane", "cards": 4}


def build(plates, size, fps=24, cards=4, camera=None):
    cfg = MP.plan({"cards": cards, "subject_z": SUBJECT_Z})
    W = MP.apply_slots(words(cfg), cfg)
    data = load_plate(plates, fps)
    plain = []
    for w in W:
        e, x, _ = cp.split_port_spec(w)
        v = dict(w, entry=e, exit=x)
        v.pop("slot", None)
        plain.append(v)
    spec = {"fps": fps, "duration": data.n / fps, "bpm": 240.0, "seed": 14, "energy": 0.9, "mood": "driving", "tier": "kinetic",
            "size": [576, 1024], "palette": PAL, "camera": "demo", "words": plain, "transitions": ["twist", "fan", "twist"],
            "beats": [round(0.25 * i, 3) for i in range(100)], "downbeats": [round(1.0 * i, 3) for i in range(25)], "bar": 4}
    scene = S_.build_scene(spec, size=(576, 1024))
    ctx = cp.ContourContext(data, world=scene.world, n=192)
    ctx.set_plane(cfg["z"][cfg["subject"]], cfg["ref"])
    items = cp.apply_ports(list(scene.seq.items), W, ctx)
    seq = FusionSequence(items, scene.palette, 0.0)
    seq.tail = max(0.0, spec["duration"] - seq.duration)
    seq.warps = [None] * len(seq.timeline)
    cp.attach(seq, ctx)
    chain = MP.camera_for(seq, camera or CAMERA)
    return dict(scene=scene, seq=seq, chain=chain, data=data, cfg=cfg, spec=spec, ctx=ctx)


def _type_frame(i):
    G = GLOB
    t = i / G["fps"]
    seq, chain, size = G["seq"], G["chain"], G["size"]
    layers, cam = seq.layers(t), chain.at(t)
    rgb = _draw(layers, cam, (0, 0, 0), size, 2)
    g = raster.rasterize(layers, cam, size, S=2)
    A = np.clip(g.A, 0, 1)
    D = np.where(A > 1e-3, g.depth(), 1e9).astype(np.float32)
    np.savez(os.path.join(G["tmp"], f"{i:05d}.npz"), rgb=rgb, A=(A * 255 + 0.5).astype(np.uint8), D=D)
    return i


def free_comfy():
    """a shared box: ask ComfyUI (port 18188) to release its GPU memory before the warp (best effort)."""
    try:
        import urllib.request
        rq = urllib.request.Request("http://localhost:18188/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                    headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(rq, timeout=5).read()
        time.sleep(3)
    except Exception as e:
        print("free_comfy:", e)


def render(plates, out, size=(360, 640), procs=10, fps=24, cards=4, max_frames=None, tag="g02", only=None):
    import torch
    from tools.fusion import compose
    free_comfy()
    cam = dict(CAMERA)
    if only:
        cam["chain"] = [e for e in CAMERA["chain"] if e["module"] == only]
    S = build(plates, size, fps, cards, cam)
    nf = min(int(round(S["spec"]["duration"] * fps)), S["data"].n)
    if max_frames:
        nf = min(nf, max_frames)
    frames = os.path.join(out, "multiplane_" + tag, "frames")
    tmp = os.path.join(out, "multiplane_" + tag, "type")
    os.makedirs(frames, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    GLOB.update(seq=S["seq"], chain=S["chain"], size=tuple(size), fps=fps, tmp=tmp)
    cam3d.set_world(*S["scene"].world)
    t0 = time.perf_counter()
    with get_context("fork").Pool(procs) as pool:
        for _ in pool.imap_unordered(_type_frame, range(nf), chunksize=2):
            pass
    tA = time.perf_counter() - t0
    mp = MP.MultiPlane(S["data"], S["scene"].world, size, cards=cards, z=S["cfg"]["z"], ref=S["cfg"]["ref"])
    t1 = time.perf_counter()
    cache = {}

    def load(i):
        if i not in cache:
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            z = np.load(os.path.join(tmp, f"{min(i, nf - 1):05d}.npz"))
            cache[i] = (z["rgb"].astype(np.float32) / 255.0, z["A"].astype(np.float32) / 255.0, z["D"])
        return cache[i]
    holds = [(a + it.timing.form, a + it.timing.form + it.timing.hold) for it, a, b in S["seq"].timeline
             if hasattr(it, "timing") and hasattr(it, "placement")]
    plan_leg = MP.plan_legible(mp, S["chain"], fps, lambda f: load(f)[1], lambda f: load(f)[2], holds, frame1=nf)
    flagged = [dict(a=round(h["a"], 2), b=round(h["b"], 2), flagged=len(h["boxes"]), letters=h["n"], hidden=h["hidden"]) for h in plan_leg["holds"]]
    scl = []
    for i in range(nf):
        t = i / fps
        scl.append(mp.scales(S["chain"].at(t))[S["cfg"]["subject"]])
        rgb, A, D = load(i)
        clear = compose.legible_clear(A, plan_leg, t, ramp_s=0.25)
        img = mp.render(i, S["chain"].at(t), rgb, A, D, clear=clear)
        u8 = (img.cpu().numpy() * 255 + 0.5).astype(np.uint8)
        bgr = np.ascontiguousarray(u8[..., ::-1])
        cv2.putText(bgr, f"{t:5.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(frames, f"{i:05d}.png"), bgr)
    tB = time.perf_counter() - t1
    mp4 = os.path.join(out, f"multiplane_{tag}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", os.path.join(frames, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "17", mp4], check=True)
    strip = os.path.join(out, f"multiplane_strip_{tag}.png")
    make_strip(frames, nf, S, strip, fps)
    scl = np.asarray(scl)
    jump = np.abs(scl[1:] / np.maximum(scl[:-1], 1e-9) - 1.0)
    zoom = os.path.join(out, f"multiplane_zoom_{tag}.png")
    zi = [int(round(x * 0.1 * fps)) for x in range(28, 41) if int(round(x * 0.1 * fps)) < nf]
    tiles = [cv2.imread(os.path.join(frames, f"{i:05d}.png")) for i in zi]
    if tiles:
        cv2.imwrite(zoom, np.vstack([np.hstack(tiles[:7] + [np.zeros_like(tiles[0])] * (7 - len(tiles[:7]))),
                                     np.hstack(tiles[7:] + [np.zeros_like(tiles[0])] * (7 - len(tiles[7:])))]))
    info = dict(tag=tag, subject_scale_max_jump=round(float(jump.max()), 4) if len(jump) else 0.0,
                subject_scale_jump_2p8_4p0=round(float(jump[int(2.8 * fps):int(4.0 * fps)].max()), 4) if len(jump) > int(4.0 * fps) else None, device=str(mp.dev), edge=mp.edge_report(), legible=flagged, frames=nf, size=list(size), type_s=round(tA, 1), warp_s=round(tB, 1), cards=cards, z=S["cfg"]["z"], slots=S["cfg"]["slots"],
                ref=S["cfg"]["ref"], camera=S["chain"].plan_text(), timeline=[
                    dict(item=getattr(it, "name", type(it).__name__), start=round(a, 2), end=round(b, 2)) for it, a, b in S["seq"].timeline])
    with open(os.path.join(out, f"multiplane_{tag}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


def make_strip(frames, nf, S, path, fps, cols=8):
    idx = np.linspace(0, nf - 1, 2 * cols).round().astype(int)
    tiles = []
    for i in idx:
        im = cv2.imread(os.path.join(frames, f"{i:05d}.png"))
        t = i / fps
        nm = next((n for n, p, a, b in S["chain"].plan if a <= t < b), "")
        lab = np.zeros((22, im.shape[1], 3), np.uint8)
        cv2.putText(lab, str(nm)[:30], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, lab]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="360x640")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--cards", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tag", default="g02")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.lower().split("x"))
    render(a.plates, a.out, (w, h), a.procs, 24, a.cards, a.max_frames, a.tag, a.only)


if __name__ == "__main__":
    main()
