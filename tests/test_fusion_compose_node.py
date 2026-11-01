"""WP-B: compose runner (tools/fusion/recipe_eval.py), MographCompose / MographAssemble / MographTypePasses helpers.

Needs the P02 v002 assets (15_fusion/v002/out shot mp4s, local/pieces/P02 depth PNGs, type passes, composed frames).
Node-level tests also need a ComfyUI checkout (comfy_api); they skip without one.

    python -m pytest tests/test_fusion_compose_node.py -s -q      (-s shows the timings)
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402
from tools.fusion import compose as fc  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

PIECE_JSON = ROOT / "15_fusion" / "pieces" / "P02.json"
LOCAL = ROOT / "15_fusion" / "v002" / "local" / "pieces" / "P02"
STATE = LOCAL / "state.json"

pytestmark = pytest.mark.skipif(not (PIECE_JSON.exists() and STATE.exists() and (LOCAL / "type" / "matte").is_dir()),
                                reason="P02 v002 assets not present")


def say(msg):
    print(msg, flush=True)


def _comfy():
    cands = [os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI")]
    cands += [str(p / "ComfyUI") for p in ROOT.parents]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            try:
                import comfy_api.latest  # noqa: F401
                import folder_paths  # noqa: F401
            except Exception as e:   # noqa: BLE001
                pytest.skip(f"comfy_api import failed: {e}")
            return c
    pytest.skip("no ComfyUI checkout found (set MOGRAPH_COMFYUI_PATH)")


# --------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def items():
    return json.loads(STATE.read_text(encoding="utf-8"))["items"]


@pytest.fixture(scope="module")
def piece():
    return json.loads(PIECE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ana():
    return json.loads((LOCAL / "analysis.json").read_text(encoding="utf-8"))


def decode_frames(mp4, n):
    """first n frames of an mp4 -> uint8 torch [n,H,W,3]."""
    w, h, total, _ = fc._probe_video(mp4)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(mp4), "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "pipe:1"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy())


def read_pngs(files):
    out = []
    for f in files:
        im = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_UNCHANGED)
        out.append(im[..., ::-1] if im.ndim == 3 else im)
    return torch.from_numpy(np.ascontiguousarray(np.stack(out)))


@pytest.fixture(scope="module")
def s1(items):
    return dict(footage=decode_frames(items["shot:s1"]["out"]["local"], 11), depth=read_pngs(items["depth:s1"]["out"]["files"]))


def psnr(a, b):
    m = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def ref_png(sid, i):
    return cv2.imdecode(np.fromfile(str(LOCAL / "compose" / sid / "frames" / f"{i:05d}.png"), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


# --------------------------------------------------------------------------------------------- pure logic
def test_layout_and_needs(piece):
    lay, cfl = RE.layout(piece)
    assert lay == {"s1": dict(start=0, frames=121, xf=0), "s2": dict(start=109, frames=121, xf=12),
                   "s3": dict(start=218, frames=105, xf=12)}
    assert RE.need_passes(piece) == ["beauty", "depth", "line", "matte"]
    assert RE.control_passes(piece) == ["depth", "matte"]
    assert RE.shot_needs(piece, "s1") == {"matte": False, "depth": True}
    parts, _ = RE.assemble_parts(piece)
    assert [round(p["cut"], 4) for p in parts] == [0.0, round((109 + 6) / 24, 4), round((218 + 6) / 24, 4)]


# --------------------------------------------------------------------------------------------- compose: numpy vs v002 PNGs
def test_numpy_matches_v002_png_s1_frames_0_10(piece, ana, s1):
    t0 = time.time()
    u8, st = RE.run_compose(piece, "s1", type_dir=LOCAL / "type", footage=s1["footage"], depth=s1["depth"], curves=ana["curves"],
                            cut=ana["cut"], backend="numpy", frames=(0, 11))
    say(f"[numpy 1080x1920] s1 frames 0-10: {st['s_per_frame']:.3f} s/frame (single process, {st['backend']}), wall {time.time() - t0:.1f}s")
    assert tuple(u8.shape) == (11, 1920, 1080, 3) and u8.dtype == torch.uint8
    worst = 0
    for i in range(11):
        d = np.abs(u8[i].numpy().astype(int) - ref_png("s1", i).astype(int)).max()
        worst = max(worst, int(d))
    say(f"  max abs diff vs v002 PNGs over 11 frames: {worst}/255")
    assert worst <= 2


def test_numpy_matches_v002_png_s3_signoff(items, piece, ana):
    """s3 local frames 40-41 are inside the signoff fade (piece t = 10.75 s): exercises the signoff source + over op."""
    foot = decode_frames(items["shot:s3"]["out"]["local"], 42)
    depth = read_pngs(items["depth:s3"]["out"]["files"])
    u8, st = RE.run_compose(piece, "s3", type_dir=LOCAL / "type", footage=foot, depth=depth, curves=ana["curves"], cut=ana["cut"],
                            backend="numpy", frames=(40, 42))
    worst = max(int(np.abs(u8[k].numpy().astype(int) - ref_png("s3", 40 + k).astype(int)).max()) for k in range(2))
    say(f"[numpy 1080x1920] s3 frames 40-41 (signoff): {st['s_per_frame']:.3f} s/frame, max abs diff {worst}/255")
    assert worst <= 2


# --------------------------------------------------------------------------------------------- compose: torch vs numpy
def test_torch_psnr_vs_numpy_draft(piece, ana, s1):
    cuda = torch.cuda.is_available()
    dev = "CUDA" if cuda else "CPU (this torch wheel is %s; no CUDA device usable)" % torch.__version__
    pd = dict(copy.deepcopy(piece), sizes={"compose": [540, 960]}, quality="draft")
    res, spf = {}, {}
    for be in ("numpy", "torch"):
        t0 = time.time()
        u8, st = RE.run_compose(pd, "s1", type_dir=LOCAL / "type", footage=s1["footage"], depth=s1["depth"], curves=ana["curves"],
                                cut=ana["cut"], backend=be, frames=(0, 8))
        res[be], spf[be] = u8.numpy(), st["s_per_frame"]
        say(f"[{be:5} 540x960] 8 frames: {st['s_per_frame']:.3f} s/frame ({st['backend']}), wall {time.time() - t0:.1f}s")
    p = [psnr(res["numpy"][i], res["torch"][i]) for i in range(8)]
    say(f"  torch on {dev}: PSNR vs numpy per frame: {[round(float(x), 1) for x in p]}")
    assert min(p) > 38.0


def test_torch_psnr_vs_numpy_full_two_frames(piece, ana, s1):
    """final-size match check, 2 frames (torch on whatever device torch has)."""
    out = {}
    for be in ("numpy", "torch"):
        u8, st = RE.run_compose(piece, "s1", type_dir=LOCAL / "type", footage=s1["footage"], depth=s1["depth"], curves=ana["curves"],
                                cut=ana["cut"], backend=be, frames=(5, 7))
        out[be] = u8.numpy()
        say(f"[{be:5} 1080x1920] 2 frames: {st['s_per_frame']:.3f} s/frame ({st['backend']})")
    p = [psnr(out["numpy"][i], out["torch"][i]) for i in range(2)]
    say(f"  PSNR {[round(float(x), 1) for x in p]}")
    assert min(p) > 38.0


# --------------------------------------------------------------------------------------------- assemble
def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration:"
                        "format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next(s for s in j["streams"] if s["codec_type"] == "audio")
    return v, a, float(j["format"]["duration"])


def short_piece(piece, frames, cf, quality, size):
    p = copy.deepcopy(piece)
    p["shots"] = [dict(s, frames=frames) for s in p["shots"][:2]]
    p["compose"]["crossfade"] = cf
    p["quality"] = quality
    p["sizes"] = {"compose": list(size)}
    p.pop("signoff", None)
    return p


def trimmed_wav(seconds, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(LOCAL / "audio.wav"), "-t", f"{seconds:.6f}", "-ar", "48000", "-ac", "2",
                    str(dst)], check=True)
    return dst


def test_encoder_args_and_nvenc_probe():
    _comfy()
    from nodes_fusion import compose as nc
    avail = nc.nvenc_available()
    say(f"[encoder] h264_nvenc usable here: {avail}")
    d = nc.encoder_args("x264", 16, True)
    assert d["vcodec"] == "libx264" and d["vopts"] == ["-preset", "veryfast", "-crf", "20"]
    d = nc.encoder_args("x264", 16, False)
    assert d["vopts"] == ["-preset", "medium", "-crf", "16"]
    d = nc.encoder_args("auto", 16, False)
    assert d["vcodec"] == ("h264_nvenc" if avail else "libx264")
    if not avail:
        assert nc.encoder_args("nvenc", 16, False)["vcodec"] == "libx264"      # falls back


def test_assemble_final_1080x1920(tmp_path, piece):
    """two composed shots (v002 PNG frames) -> MP4: 1080x1920, 24 fps, AAC, duration = cut."""
    _comfy()
    from nodes_fusion import compose as nc
    p = short_piece(piece, 25, 0.5, "final", (1080, 1920))
    lay, _ = RE.layout(p)
    total = max(v["start"] + v["frames"] for v in lay.values())
    cut = total / 24.0
    composed = [{"u8": read_pngs([LOCAL / "compose" / sid / "frames" / f"{i:05d}.png" for i in range(25)]), "sid": sid,
                 "start": lay[sid]["start"], "fps": 24} for sid in ("s1", "s2")]
    wav = trimmed_wav(cut, tmp_path / "a.wav")
    out = tmp_path / "final.mp4"
    r = nc.assemble_video(p, composed, str(wav), str(out), crf=16, encoder="auto")
    v, a, dur = ffprobe(out)
    say(f"[assemble final] {r['frames']} frames, {r['encoder']}, {r['wall_s']:.2f}s; {v['width']}x{v['height']} {v['r_frame_rate']} "
        f"{v['codec_name']} + {a['codec_name']}, duration {dur:.4f}s (cut {cut:.4f}s)")
    assert (v["width"], v["height"]) == (1080, 1920) and v["r_frame_rate"] == "24/1"
    assert a["codec_name"] == "aac"
    assert abs(dur - cut) < 0.03 and r["frames"] == total


def test_compose_and_assemble_draft_540x960(tmp_path, items, piece, ana):
    """draft: compose 540x960 (auto backend) -> assemble with x264 veryfast crf 20."""
    _comfy()
    from nodes_fusion import compose as nc
    p = short_piece(piece, 20, 0.25, "draft", (540, 960))
    lay, _ = RE.layout(p)
    total = max(v["start"] + v["frames"] for v in lay.values())
    cut = total / 24.0
    clips = [{"u8": decode_frames(items[f"shot:{sid}"]["out"]["local"], 20), "sid": sid, "start": lay[sid]["start"], "fps": 24}
             for sid in ("s1", "s2")]
    depths = {sid: read_pngs(items[f"depth:{sid}"]["out"]["files"]) for sid in ("s1", "s2")}
    t0 = time.time()
    composed, stats = nc.compose_clips(p, {"dir_layers": str(LOCAL / "type")}, clips, music=dict(ana), depths=depths, backend="auto")
    say(f"[compose draft 540x960] {sum(s['n'] for s in stats)} frames in {time.time() - t0:.1f}s, "
        f"{[round(s['s_per_frame'], 3) for s in stats]} s/frame ({stats[0]['backend']})")
    assert all(tuple(c["u8"].shape[1:]) == (960, 540, 3) for c in composed)
    hero = nc.pick_hero(p, str(LOCAL / "type"), composed)
    assert hero.shape == (960, 540, 3) and 0 <= hero.min() and hero.max() <= 1
    wav = trimmed_wav(cut, tmp_path / "a.wav")
    out = tmp_path / "draft.mp4"
    r = nc.assemble_video(p, composed, str(wav), str(out), crf=16, encoder="x264")
    v, a, dur = ffprobe(out)
    say(f"[assemble draft] {r['frames']} frames, {r['encoder']}, {r['wall_s']:.2f}s; {v['width']}x{v['height']} {v['r_frame_rate']} "
        f"{v['codec_name']} + {a['codec_name']}, duration {dur:.4f}s (cut {cut:.4f}s)")
    assert (v["width"], v["height"]) == (540, 960) and v["r_frame_rate"] == "24/1" and a["codec_name"] == "aac"
    assert abs(dur - cut) < 0.03


def test_lazy_status_and_schema(piece):
    _comfy()
    from nodes_fusion import compose as nc
    p = copy.deepcopy(piece)
    clips = [{"sid": "s1"}, {"sid": "s2"}]
    assert nc.MographCompose.check_lazy_status(piece=p, clips=clips) == ["depth_1", "depth_2"]
    assert nc.MographCompose.check_lazy_status(piece=p, clips=clips, depth_1=object()) == ["depth_2"]
    for n in nc.NODES:
        assert n.define_schema().node_id in ("MographCompose", "MographAssemble")


# --------------------------------------------------------------------------------------------- type passes node
@pytest.mark.skipif(os.environ.get("MOGRAPH_SKIP_SLOW") == "1", reason="MOGRAPH_SKIP_SLOW=1 (type passes take minutes)")
def test_type_passes_draft_and_final(tmp_path, piece, ana):
    _comfy()
    from nodes_fusion import type as nt
    p = copy.deepcopy(piece)
    p["quality"] = "draft"
    t0 = time.time()
    t = nt.start_type_passes(p, dict(ana), procs=4, work=tmp_path / "d", frames=(0, 8))
    say(f"[type draft 576x1024] 8 frames, 4 procs: {time.time() - t0:.1f}s; passes {t['passes']}")
    assert t["job"] is None and t["dir_control"] == t["dir_layers"]
    assert len(list((Path(t["dir_layers"]) / "matte").glob("*.png"))) == 8
    assert cv2.imread(str(Path(t["dir_layers"]) / "matte" / "matte_00000.png"), 0).shape == (1024, 576)
    sheet = nt.make_sheet_image(t)
    assert sheet.ndim == 3 and sheet.shape[2] == 3 and sheet.dtype == np.float32 and 0.0 <= sheet.min() and sheet.max() <= 1.0

    p["quality"] = "final"
    t0 = time.time()
    t = nt.start_type_passes(p, dict(ana), procs=4, work=tmp_path / "f", frames=(0, 4))
    ctl = time.time() - t0
    assert t["job"] is not None and t["dir_control"] != t["dir_layers"]
    assert len(list((Path(t["dir_control"]) / "depth").glob("*.png"))) == 4        # control pass (blocking) is done
    t.wait()
    say(f"[type final] control 576x1024 blocking {ctl:.1f}s, layers 1088x1920 finished after {time.time() - t0:.1f}s")
    assert cv2.imread(str(Path(t["dir_layers"]) / "line" / "line_00000.png"), 0).shape == (1920, 1088)
    assert t["manifest_layers"]["size"] == [1088, 1920]
