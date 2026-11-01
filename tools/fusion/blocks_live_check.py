"""blocks_live_check: ON THE BOX check render of the two live lego blocks over their real P18 / P20 footage (360x640 compose size).

    python -m tools.fusion.blocks_live_check strips 15_fusion/pieces/P18.json --a 3.2 --b 7.3 --out 16_app/core_check/v005_strips_resnap
    python -m tools.fusion.blocks_live_check edge   15_fusion/pieces/P20.json --a 4.2 --b 6.4 --out 16_app/core_check/v005_edge_run

Runs the type passes of the piece (the type follows the strips / the module letters hand over to the edge run), RAFT flow per shot, then the
piece's own compose recipe (tools/fusion/recipe_eval.run_compose, torch / CUDA) over the shots that overlap [a, b] s, cuts the window out
in piece time and writes <out>.png (contact sheet, time + shot on every tile), <out>.mp4 (24 fps) and <out>.json (timings). Footage:
<plates dir of the piece JSON> relative to the repo root (the plate / matte / depth mp4 of every shot). Frame rendering is on the box
only (CLAUDE.md rule 4); the unit tests of the blocks are tests/test_blocks_live.py."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _decode(path, gray=False):
    from tools.fusion import box_stages as BS
    a = BS.decode_u8(path)
    return a[..., 0] if gray else a


def _patch_screen_offset_pickle():
    """spec.ScreenOffsetCamera.__getattr__ recurses forever while the type job is UNPICKLED (self.inner is not set yet), which stops every type
    pass of a piece with a screen_offset layout (P18, P20: RecursionError in type_passes._init). spec.py belongs to another worker: patched
    in this process only (procs=1 keeps the unpickle in-process). The fix there: raise AttributeError for inner / keys / world / dunder names."""
    from tools.fusion import spec as S
    cls = S.ScreenOffsetCamera

    def ga(self, name):
        if name in ("inner", "keys", "world") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)
    cls.__getattr__ = ga


def _glow_colour_workaround(piece):
    """matte_treatments (wipe / glow) write add_glow(color="$tcol") with the tinted type IMAGE as the colour; compose.add_glow wants an RGB
    tuple (_vec3 cannot take an HxWx3 buffer: TypeError on CUDA). Not this block's code: replaced here by a fixed light tint so the
    check can run; the count is reported."""
    n = 0
    comp = piece.get("compose") or {}
    for ch in ([comp.get("default")] if comp.get("default") else []) + list((comp.get("shots") or {}).values()):
        for o in ch["ops"]:
            if o["op"] == "add_glow" and isinstance(o["args"].get("color"), str):
                o["args"]["color"] = [0.95, 0.95, 0.97]
                n += 1
    return n


def render_window(piece_path, a, b, out, procs=1, sheet_n=12, tiles=None, backend="torch", reuse_type=False):
    from tools.fusion import footage_flow as FF
    from tools.fusion import recipe_eval as RE
    from tools.fusion import type_passes
    piece = json.loads(Path(piece_path).read_text(encoding="utf-8"))
    fps = int(piece.get("fps", 24))
    n_glow = _glow_colour_workaround(piece)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.name + "_work")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _patch_screen_offset_pickle()
    spec = json.loads(json.dumps(piece["type"]))
    spec["size"] = list(piece["sizes"]["type"])
    passes = RE.need_passes(piece)
    if reuse_type and (work / "type" / "manifest.json").is_file():
        man = json.loads((work / "type" / "manifest.json").read_text(encoding="utf-8"))
    else:
        man = type_passes.render_passes(spec, str(work / "type"), size=tuple(spec["size"]), passes=passes, procs=procs, overwrite=True, mp4=False, verbose=False)
    t_type = time.time() - t0
    ana = piece["audio"].get("analysis")
    curves = {}
    if ana and (ROOT / ana).is_file():
        curves = {k: v for k, v in json.loads((ROOT / ana).read_text(encoding="utf-8")).get("curves", {}).items() if isinstance(v, list)}
    elif piece["type"].get("curves"):
        curves = piece["type"]["curves"]
    cut = float(piece["audio"]["cut"])
    lay, _ = RE.piece_layout(piece)
    fa, fb = int(round(a * fps)), int(round(b * fps))
    frames, timing = {}, {}
    for sh, pl in zip(piece["shots"], piece["type"]["plates"]):
        sid = sh["id"]
        L = lay[sid]
        if L["start"] + L["frames"] <= fa or L["start"] >= fb:
            continue
        t1 = time.time()
        footage = _decode(str(ROOT / pl["plate"]))[:L["frames"]]
        matte = _decode(str(ROOT / pl["matte"]), gray=True)[:L["frames"]] if pl.get("matte") else None
        depth = _decode(str(ROOT / pl["depth"]), gray=True)[:L["frames"]] if pl.get("depth") else None
        flow = FF.flow_for(footage, method="auto", verbose=False)
        t2 = time.time()
        u8, st = RE.run_compose(piece, sid, type_dir=str(work / "type"), footage=footage, matte=matte, depth=depth, curves=curves, cut=cut,
                                backend=backend, flow=flow)
        u8 = u8.numpy() if hasattr(u8, "numpy") else np.asarray(u8)
        frames[sid] = u8
        timing[sid] = dict(frames=int(u8.shape[0]), flow_s=round(t2 - t1, 1), compose_s=round(time.time() - t2, 1), s_per_frame=round(st["s_per_frame"], 4),
                           backend=st["backend"])
    seq, labels = [], []
    for f in range(fa, fb):
        for sid, u8 in frames.items():
            s0 = lay[sid]["start"]
            if s0 <= f < s0 + lay[sid]["frames"]:
                seq.append(u8[f - s0])
                labels.append((f / fps, sid))
                break
    assert seq, "no frames in the window"
    H, W = seq[0].shape[:2]
    # mp4 (24 fps, no audio)
    mp4 = str(out) + ".mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an",
                          "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", mp4], stdin=subprocess.PIPE)
    for fr in seq:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    # contact sheet: chosen times (tiles) or evenly spaced; every tile labelled with piece time + shot; cut frames get a red mark
    idx = [int(np.argmin([abs(t - x) for t, _ in labels])) for x in tiles] if tiles else list(np.unique(np.linspace(0, len(seq) - 1, sheet_n).round().astype(int)))
    cuts = {lay[sid]["start"] / fps for sid in lay if lay[sid]["start"] > 0}
    tl = []
    for i in idx:
        im = cv2.cvtColor(np.ascontiguousarray(seq[i]), cv2.COLOR_RGB2BGR)
        t, sid = labels[i]
        near = any(abs(t - c) < 0.5 / fps for c in cuts)
        cv2.rectangle(im, (0, 0), (W - 1, 18), (20, 20, 20), -1)
        cv2.putText(im, f"{t:.2f}s {sid}" + (" CUT" if near else ""), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 80, 255) if near else (80, 230, 80), 1, cv2.LINE_AA)
        tl.append(np.pad(im, ((0, 3), (0, 3), (0, 0)), constant_values=40))
    per = 6 if len(tl) > 6 else len(tl)
    while len(tl) % per:
        tl.append(np.zeros_like(tl[0]))
    sheet = np.vstack([np.hstack(tl[r:r + per]) for r in range(0, len(tl), per)])
    cv2.imwrite(str(out) + ".png", sheet)
    rep = dict(piece=str(piece_path), window=[a, b], frames=len(seq), size=[W, H], type_s=round(t_type, 1), type_frames=man["nf"], shots=timing,
               tiles=[round(labels[i][0], 3) for i in idx], wall_s=round(time.time() - t0, 1), mp4=mp4, png=str(out) + ".png",
               workaround=f"{n_glow} add_glow colour buffers replaced by a tuple (matte_treatments bug, not this block)")
    Path(str(out) + ".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["strips", "edge"])
    ap.add_argument("piece")
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1, help="type workers (1 = in-process; >1 needs the ScreenOffsetCamera pickle fix in spec.py)")
    ap.add_argument("--tiles", default="", help="comma separated piece seconds for the contact sheet (default: 12 evenly spaced)")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--reuse-type", action="store_true", help="keep the type passes of the last run in <out>_work (only the compose changed)")
    a = ap.parse_args(argv)
    os.environ.setdefault("FUSION_BACKEND", a.backend)
    tiles = [float(x) for x in a.tiles.split(",") if x] or None
    rep = render_window(a.piece, a.a, a.b, a.out, procs=a.procs, tiles=tiles, backend=a.backend, reuse_type=a.reuse_type)
    print(json.dumps(rep))
    return rep


if __name__ == "__main__":
    main()
