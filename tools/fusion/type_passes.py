"""type_passes: render a JSON piece spec of type modules (mograph.modules) as synchronised 9:16 passes for compositing with
generated video: beauty, matte, depth, line, clay, flow. CPU only.

    python tools/fusion/type_passes.py <spec.json> --out <dir> [--size 576x1024] [--passes beauty,matte,...] [--frames a:b]
                                       [--procs N] [--sheet sheet.png] [--overwrite]

    import: from tools.fusion.type_passes import render_passes;  render_passes(spec_dict_or_path, out_dir, size=(576, 1024))

Every pass uses the same Sequence timeline, the same camera (CameraChain) and the same size, frame for frame. Output (out/):
    beauty/beauty_00000.png  matte/matte_00000.png (L)  depth/depth_00000.png (L)  line/line_00000.png (L)  clay/clay_00000.png
    flow/flow_00000.npy (float16, H/2 x W/2 x 2, dx dy in half-res px)  flow_vis/flow_vis_00000.png
    <pass>.mp4 previews, flow.mp4 (colour wheel), manifest.json (timeline, camera plan, depth range, timings).
Sizes are multiples of 64: 576x1024 (draft), 1088x1920 (final). See spec.py for the spec schema and README notes in the report."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from mograph.render import cam3d  # noqa: E402
from mograph.modules.sequence import _draw  # noqa: E402
from tools.fusion import spec as S_  # noqa: E402
from tools.fusion import raster, clay, flow as flowmod  # noqa: E402

PASSES = S_.PASSES
GREY_PASSES = ("matte", "depth", "line")
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 3]
_J = {}


# ------------------------------------------------------------------ depth normalisation
def scan_depth(scene, step=4, pct=(1.0, 99.0)):
    """auto near / far (camera-space z, world units) from the vertices of the type that are on screen over the whole clip."""
    w, h = scene.size
    seq, ch = scene.seq, scene.camera
    zs = []
    for i in range(0, scene.nf, step):
        t = i / scene.fps
        cam = ch.at(t)
        for it in raster.collect(seq.layers(t), cam, scene.size):
            for xy, Qc, cl in it.polys:
                ok = (xy[:, 0] > -0.15 * w) & (xy[:, 0] < 1.15 * w) & (xy[:, 1] > -0.15 * h) & (xy[:, 1] < 1.15 * h)
                if ok.any():
                    zs.append(Qc[ok, 2])
    if not zs:
        return 500.0, 1500.0
    z = np.concatenate(zs)
    near, far = float(np.percentile(z, pct[0])), float(np.percentile(z, pct[1]))
    far = max(far, near * 1.05 + 10.0)
    return near, far


def depth_to_grey(G, near, far, mode="linear"):
    """near = white, far = dark (32), background 0. Anti-aliased by the coverage."""
    d = G.depth()
    if mode == "inverse":
        u = (1.0 / np.maximum(d, 1e-3) - 1.0 / far) / (1.0 / near - 1.0 / far)
        v = 32.0 + 223.0 * np.clip(u, 0, 1)
    else:
        u = (d - near) / max(far - near, 1e-6)
        v = 32.0 + 223.0 * (1.0 - np.clip(u, 0, 1))
    return np.clip(v * G.A + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ frame worker
def _init(blob):
    _J.clear()
    _J.update(pickle.loads(blob))
    cam3d.set_world(*_J["scene"].world)


def _save(path, im):
    cv2.imwrite(path, im, PNG)


def _frame(i):
    J = _J
    sc, out, passes, size, Ssup = J["scene"], J["out"], J["passes"], sc_size(J), J["S"]
    seq, camsrc, fps = sc.seq, sc.camera, sc.fps
    t = i / fps
    tm = {}
    T = time.perf_counter
    t0 = T()
    layers = seq.layers(t)
    cam = camsrc.at(t)
    tm["layers+camera"] = T() - t0
    if "beauty" in passes:
        t0 = T()
        im = _draw(layers, cam, seq.bg, size, Ssup)
        _save(f"{out}/beauty/beauty_{i:05d}.png", np.ascontiguousarray(im[:, :, ::-1]))
        tm["beauty"] = T() - t0
    need_g = any(p in passes for p in ("matte", "depth", "line", "clay", "flow"))
    G = None
    if need_g:
        t0 = T()
        G = raster.rasterize(layers, cam, size, S=Ssup, want_lines="line" in passes, line_px=J["line_px"])
        tm["gbuffer"] = T() - t0
    k = size[0] / float(cam3d.W)
    if "matte" in passes:
        t0 = T()
        a = G.A
        if J["matte_soft"] > 0:
            a = cv2.GaussianBlur(a, (0, 0), J["matte_soft"])
        _save(f"{out}/matte/matte_{i:05d}.png", np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8))
        tm["matte"] = T() - t0
    if "depth" in passes:
        t0 = T()
        _save(f"{out}/depth/depth_{i:05d}.png", depth_to_grey(G, J["near"], J["far"], J["depth_mode"]))
        tm["depth"] = T() - t0
    if "line" in passes:
        t0 = T()
        _save(f"{out}/line/line_{i:05d}.png", np.clip(G.E * 255.0 + 0.5, 0, 255).astype(np.uint8))
        tm["line"] = T() - t0
    if "clay" in passes:
        t0 = T()
        g = clay.render(G, cam, size, k, J["near"], J["far"], sc.ground_y, sc.pivot, **J["clay"])
        _save(f"{out}/clay/clay_{i:05d}.png", np.dstack([g, g, g]))
        tm["clay"] = T() - t0
    if "flow" in passes:
        t0 = T()
        t1 = (i + 1) / fps
        cam1 = camsrc.at(t1)
        fl = flowmod.camera_flow(G, cam, cam1, size, k, sc.ground_y, J["flow_bg"])
        if J["flow_mode"] == "hybrid":
            G1 = raster.rasterize(seq.layers(t1), cam1, size, S=Ssup)
            fl = fl + flowmod.residual(G.A, G1.A, fl, size)
        np.save(f"{out}/flow/flow_{i:05d}.npy", flowmod.half_res(fl))
        tm["flow"] = T() - t0
    return i, tm


def sc_size(J):
    return tuple(J["scene"].size)


# ------------------------------------------------------------------ encode / vis / sheet
def encode(out, name, pattern, fps, start, ext="mp4"):
    dst = f"{out}/{name}.{ext}"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", str(start), "-i", pattern,
           "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)
    return dst


def flow_vis(out, frames, size, fps, mp4=True):
    """colour-wheel PNGs + mp4 from the flow .npy files; magnitude reference = 99th percentile over the clip (half-res px)."""
    os.makedirs(f"{out}/flow_vis", exist_ok=True)
    mags = []
    for i in frames:
        f = np.load(f"{out}/flow/flow_{i:05d}.npy").astype(np.float32)
        m = np.hypot(f[..., 0], f[..., 1])
        mags.append(m[::2, ::2].ravel())
    ref = float(np.percentile(np.concatenate(mags), 99.0)) if mags else 1.0
    ref = max(ref, 0.5)
    for i in frames:
        f = np.load(f"{out}/flow/flow_{i:05d}.npy").astype(np.float32)
        v = flowmod.colour_wheel(f, ref)
        v = cv2.resize(v, tuple(size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{out}/flow_vis/flow_vis_{i:05d}.png", v, PNG)
    if mp4 and frames:
        encode(out, "flow", f"{out}/flow_vis/flow_vis_%05d.png", fps, frames[0])
    return ref


def make_sheet(out, path, n=6, passes=None, width=216, idx=None):
    """contact sheet: n frames (columns, evenly spread over the rendered range, nudged off empty frames) x passes (rows)."""
    man = json.load(open(f"{out}/manifest.json", encoding="utf-8"))
    passes = passes or man["passes"]
    a, b = man["frames"]
    if idx is None:
        idx = [int(round(x)) for x in np.linspace(a + 0.04 * (b - a), b - 1 - 0.03 * (b - a), n)]
        if "matte" in man["passes"]:
            def busy(i):
                m = cv2.imread(f"{out}/matte/matte_{i:05d}.png", 0)
                return m is not None and m.mean() > 1.0
            for j, i in enumerate(idx):
                for d in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
                    if a <= i + d < b and busy(i + d):
                        idx[j] = i + d
                        break
    rows = []
    for p in passes:
        tiles = []
        for i in idx:
            if p == "flow":
                im = cv2.imread(f"{out}/flow_vis/flow_vis_{i:05d}.png")
            else:
                im = cv2.imread(f"{out}/{p}/{p}_{i:05d}.png", cv2.IMREAD_COLOR)
            sc = width / im.shape[1]
            im = cv2.resize(im, (width, int(round(im.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{p} f{i} {i / man['fps']:.2f}s", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 230, 60), 1, cv2.LINE_AA)
            tiles.append(im)
        rows.append(np.hstack([np.pad(t, ((0, 0), (0, 3), (0, 0)), constant_values=40) for t in tiles]))
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, sheet)
    return path


# ------------------------------------------------------------------ driver
def render_passes(spec, out, size=None, passes=None, frames=None, procs=None, overwrite=False, mp4=True, S=2, verbose=True):
    """render the passes of `spec` (dict / json path) into `out`. frames = (a, b) half-open frame range. -> manifest dict."""
    t_start = time.time()
    scene = S_.build_scene(spec, size)
    sp = scene.spec
    passes = [p for p in (passes or sp.get("passes") or PASSES)]
    for p in passes:
        assert p in PASSES, f"unknown pass {p!r}; known: {PASSES}"
    a, b = (0, scene.nf) if frames is None else (max(0, int(frames[0])), min(scene.nf, int(frames[1])))
    assert b > a, "empty frame range"
    os.makedirs(out, exist_ok=True)
    for p in passes:
        d = f"{out}/{p}"
        if os.path.isdir(d) and os.listdir(d) and not overwrite:
            raise FileExistsError(f"{d} is not empty (never overwrites renders; pass --overwrite or use a new --out)")
        os.makedirs(d, exist_ok=True)
    scene.camera = scene.chain if scene.chain is not None else _fixed(scene)
    ww, wh = scene.world
    # ground plane under the type, and the pivot the backdrop is centred on
    scene.ground_y = max(float(m.placement.origin[1] + m.placement.extent_along(np.array([0.0, 1.0, 0.0])))
                         for m in scene.seq.words) * 1.0
    scene.pivot = (ww / 2.0, wh / 2.0, 0.0)
    dsp = sp.get("depth") or {}
    if any(p in passes for p in ("depth", "clay")):
        near, far = (float(dsp["near"]), float(dsp["far"])) if "near" in dsp and "far" in dsp else scan_depth(scene)
    else:
        near, far = 500.0, 1500.0
    csp = dict(sp.get("clay") or {})
    if "ground_y" in csp:
        scene.ground_y = float(csp.pop("ground_y"))
    fsp = sp.get("flow") or {}
    job = dict(scene=scene, out=out, passes=passes, S=S, near=near, far=far, depth_mode=dsp.get("mode", "linear"),
               line_px=float((sp.get("line") or {}).get("px", 1.5)), matte_soft=float((sp.get("matte") or {}).get("soft", 0.6)),
               clay=dict(bevel=csp.get("bevel", 0.0075), fog=csp.get("fog", 0.30), shadow=csp.get("shadow", 0.30),
                         grid=csp.get("grid", True)),
               flow_mode=fsp.get("mode", "hybrid"), flow_bg=fsp.get("bg", "camera"))
    blob = pickle.dumps(job)
    procs = procs or max(1, min(8, (os.cpu_count() or 2) - 1))
    todo = list(range(a, b))
    results = {}
    t0 = time.time()
    if procs <= 1 or len(todo) < 3:
        _init(blob)
        for i in todo:
            j, tm = _frame(i)
            results[j] = tm
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(procs, initializer=_init, initargs=(blob,)) as pool:
            for n, (j, tm) in enumerate(pool.imap_unordered(_frame, todo, chunksize=1)):
                results[j] = tm
                if verbose and (n + 1) % 24 == 0:
                    print(f"  {n + 1}/{len(todo)} frames", flush=True)
    wall = time.time() - t0
    keys = sorted({k for tm in results.values() for k in tm})
    mean = {k: float(np.mean([tm[k] for tm in results.values() if k in tm])) for k in keys}
    cost = {"beauty": mean.get("beauty", 0.0)}
    g = mean.get("gbuffer", 0.0)
    for p in ("matte", "depth", "line", "clay", "flow"):
        if p in passes:
            cost[p] = g + mean.get(p, 0.0)          # standalone: the shared G-buffer + the pass itself
    manifest = dict(spec=sp, size=list(scene.size), world=list(scene.world), fps=scene.fps, nf=scene.nf, frames=[a, b], passes=passes,
                    depth=dict(near=near, far=far, mode=job["depth_mode"]), ground_y=scene.ground_y, info=scene.info,
                    timing=dict(wall_s=wall, procs=procs, mean_stage_s_per_frame=mean, standalone_s_per_frame_per_pass=cost,
                                build_s=t0 - t_start),
                    encode=mp4)
    if "flow" in passes:
        manifest["flow_vis_ref_px"] = flow_vis(out, list(range(a, b)), scene.size, scene.fps, mp4)
    if mp4:
        for p in passes:
            if p != "flow":
                encode(out, p, f"{out}/{p}/{p}_%05d.png", scene.fps, a)
    json.dump(manifest, open(f"{out}/manifest.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if verbose:
        print(f"rendered frames {a}..{b - 1} of {scene.nf} ({scene.size[0]}x{scene.size[1]}, {scene.fps} fps) in {wall:.1f}s on {procs} procs")
        print("  mean s/frame by stage:", {k: round(v, 3) for k, v in mean.items()})
    return manifest


class _Fixed:
    def __init__(self, cam):
        self.cam = cam

    def at(self, t):
        return self.cam


def _fixed(scene):
    return _Fixed(scene.seq.demo_camera())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default=None, help="WxH, multiples of 64 (default: the spec's size, 576x1024)")
    ap.add_argument("--passes", default=None, help="comma list of: " + ",".join(PASSES))
    ap.add_argument("--frames", default=None, help="a:b (half-open) frame range")
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--S", type=int, default=2, help="supersample factor (anti-aliasing)")
    ap.add_argument("--sheet", default=None, help="write a contact sheet (6 frames x passes) PNG here afterwards")
    ap.add_argument("--sheet-frames", default=None, help="comma list of frame numbers for the sheet (default: 6 spread out)")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    size = S_.parse_size(a.size) if a.size else None
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    man = render_passes(a.spec, a.out, size=size, passes=a.passes.split(",") if a.passes else None, frames=fr, procs=a.procs,
                        overwrite=a.overwrite, mp4=not a.no_mp4, S=a.S)
    if a.sheet:
        idx = [int(x) for x in a.sheet_frames.split(",")] if a.sheet_frames else None
        print("sheet:", make_sheet(a.out, a.sheet, idx=idx))
    return man


if __name__ == "__main__":
    main()
