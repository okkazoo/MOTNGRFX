"""Own-video Format: Original | Half scale (2026-11-03). Route validation, the half size (even rounding), and that the Alpha-Gen matte side does not follow the scale
(so a half job and a full job share their mattes in the stage cache). Pure python, no ComfyUI, no GPU."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import overlay_check as OC     # noqa: E402
from tools.app import overlay_mode as OM      # noqa: E402
from tools.app import own_video_job as J      # noqa: E402
from tools.app import own_video_route as R    # noqa: E402
from tools.app import stage_cache as SC       # noqa: E402


# ------------------------------------------------------------------------------------------------------------ route
class _Proc:
    def poll(self):
        return None


@pytest.fixture
def mgr(tmp_path):
    out, inp = tmp_path / "output", tmp_path / "input"
    out.mkdir()
    inp.mkdir()
    (inp / "clip.mp4").write_bytes(b"x")
    calls = []
    m = R.JobManager(out, inp, tmp_path, port=18188, popen=lambda argv, **kw: (calls.append(argv), _Proc())[1], python="py", script="runner.py")
    m.calls = calls
    return m


BODY = {"video": "clip.mp4", "brief": {"on_screen": ["A"]}, "format": "9:16", "quality": "final"}


def test_scale_defaults_to_full_and_is_passed_to_the_runner(mgr):
    assert R.parse_params(BODY, mgr.input_dir)["scale"] == "full"
    job = mgr.start(BODY)
    argv = mgr.calls[0]
    assert argv[argv.index("--scale") + 1] == "full"
    assert json.loads((mgr.job_dir(job) / "request.json").read_text())["scale"] == "full"


def test_half_reaches_the_runner_and_request_json(mgr):
    job = mgr.start(dict(BODY, scale="half", format="16:9"))          # a landscape clip: the page sends its orientation as the format
    argv = mgr.calls[0]
    assert argv[argv.index("--scale") + 1] == "half" and argv[argv.index("--format") + 1] == "16:9"
    assert json.loads((mgr.job_dir(job) / "request.json").read_text())["scale"] == "half"
    assert R.parse_params(dict(BODY, scale=" Half "), mgr.input_dir)["scale"] == "half"
    assert R.parse_params(dict(BODY, scale=""), mgr.input_dir)["scale"] == "full"
    assert R.parse_params(dict(BODY, scale=None), mgr.input_dir)["scale"] == "full"


@pytest.mark.parametrize("bad", ["quarter", "0.5", 0.5, 2, 0, "full,half", [], True])
def test_bad_scale_is_a_400(mgr, bad):
    with pytest.raises(R.BadRequest) as e:
        mgr.start(dict(BODY, scale=bad))
    assert "scale" in str(e.value) and not mgr.calls


def test_format_is_the_clip_orientation_only(mgr):
    assert R.parse_params(dict(BODY, format="16:9"), mgr.input_dir)["format"] == "16:9"
    with pytest.raises(R.BadRequest):
        R.parse_params(dict(BODY, format="4:3"), mgr.input_dir)


# ------------------------------------------------------------------------------------------------------------ sizes
@pytest.mark.parametrize("W,H,half", [
    (1080, 1920, (540, 960)), (1920, 1080, (960, 540)), (540, 960, (270, 480)), (720, 1280, (360, 640)), (3840, 2160, (1920, 1080)),
    (1081, 1921, (540, 960)),          # odd source: floored to even first (1080x1920), then halved
    (1090, 1922, (544, 960)),          # 1090 / 2 = 545 -> floored to even = 544; 1922 / 2 = 961 -> 960
    (270, 480, (134, 240)),            # half of an already half-size clip: 135 -> 134
    (3, 5, (2, 2)),                    # degenerate: never below 2
])
def test_half_size_is_even_and_floors(W, H, half):
    assert OM.scaled_size(W, H, "half") == half
    assert all(v % 2 == 0 for v in half)
    assert OM.scaled_size(W, H, "full") == (OM.even(W), OM.even(H))
    assert OC.scaled_size(W, H, "half") == half and OC.scaled_size(W, H, "full") == OM.scaled_size(W, H, "full")      # the box check script uses the same rule


def test_norm_scale():
    assert [OM.norm_scale(v) for v in ("half", "HALF", " half ", "0.5", "half_scale", "full", "", None, "x", 1)] == ["half", "half", "half", "half", "half", "full", "full", "full", "full", "full"]


@pytest.mark.parametrize("W,H", [(1080, 1920), (1920, 1080), (540, 960), (1081, 1921), (720, 1280), (1280, 720)])
@pytest.mark.parametrize("q", ["preview", "draft"])
def test_native_sizes_half_keeps_the_matte_and_the_source(W, H, q):
    full, half = OM.native_sizes(W, H, q, "full"), OM.native_sizes(W, H, q, "half")
    assert full["matte"] == half["matte"] and full["control"] == half["control"] and full["piece"]["control"] == half["piece"]["control"]          # the Alpha-Gen size follows the SOURCE
    assert full["source"] == half["source"] == (OM.even(W), OM.even(H))
    assert full["compose"] == full["source"] and half["compose"] == OM.scaled_size(W, H, "half")
    assert half["clip"] == half["shot"] == half["compose"] and half["piece"]["compose"] == list(half["compose"])
    assert full["format"] == half["format"]
    # type canvas: both sides multiples of 64, at or above the output size (compose cover-fits it back), for both scales
    for n in (full, half):
        tw, th = n["type"]
        assert tw % 64 == 0 and th % 64 == 0 and tw >= n["compose"][0] and th >= n["compose"][1]
        assert n["piece"]["type"] == [tw, th]
    assert OM.native_sizes(W, H, q)["compose"] == full["compose"]            # the default is the full size (old callers)


def test_half_type_canvas_keeps_the_aspect_of_the_half_frame():
    n = OM.native_sizes(1920, 1080, "draft", "half")          # 960x540 -> 1024x576
    assert n["compose"] == (960, 540) and n["type"] == (1024, 576) and abs(n["type_aspect"] - 960 / 540.0) < 1e-3
    n = OM.native_sizes(540, 960, "draft", "half")            # 270x480 -> the smallest multiple-of-64 canvas at 9:16 above it
    assert n["compose"] == (270, 480) and n["type"][0] % 64 == 0 and abs(n["type"][0] / n["type"][1] - 270 / 480.0) < 1e-3


# ------------------------------------------------------------------------------------------------------------ matte reuse
def _alpha_graph(stem, rel, size, job_no=1, frames=97):
    return {"ref": {"class_type": "LoadVideo", "inputs": {"file": f"{stem}_h{job_no}.mp4"}},
            "gen": {"class_type": "LTXAlpha", "inputs": {"frames": frames, "size": list(size), "seed": 1, "model": "ltx-2.3"}},
            "out": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{rel}/{stem}_alpha_h{job_no}_image"}}}


def _matte_key(tmp_path, tag, job, scale, W=1080, H=1920):
    out_root = tmp_path / tag / "output"
    d = out_root / "motngrfx_app" / "own" / job
    d.mkdir(parents=True)
    inp = tmp_path / tag / "input"
    inp.mkdir()
    stem, rel = f"motngrfx_own_{job.replace('-', '')}", f"motngrfx_app/own/{job}"
    jc = SC.JobCache(tmp_path / "cache", d, inp, job, stem, rel)
    clip = inp / f"{stem}_h1.mp4"
    clip.write_bytes(b"the matte clip: cut from the FULL window clip at the matte size" * 50)       # same bytes: the cut command for it does not mention the scale
    size = OM.native_sizes(W, H, "draft", scale)["matte"]                                              # what _run hands to V3.SIZE, hence to the Alpha-Gen graph
    return J.matte_key(jc, clip, _alpha_graph(stem, rel, size))


@pytest.mark.parametrize("W,H", [(1080, 1920), (1920, 1080), (1081, 1921)])
def test_matte_key_is_the_same_for_full_and_half(tmp_path, W, H):
    k_full = _matte_key(tmp_path, "a", "20261103-100000-aaaa", "full", W, H)
    k_half = _matte_key(tmp_path, "b", "20261103-110000-bbbb", "half", W, H)
    assert k_full and k_full == k_half                                  # a half job restores the mattes a full job stored, and the other way round


def test_matte_key_still_follows_the_source_size(tmp_path):
    assert _matte_key(tmp_path, "a", "20261103-100000-aaaa", "half", 1080, 1920) != _matte_key(tmp_path, "b", "20261103-110000-bbbb", "half", 1920, 1080)


def test_cut_script_matte_clips_do_not_depend_on_the_scale_but_shot_clips_do():
    """the ffmpeg lines that write the Alpha-Gen input clips (`..._hN.mp4`) read the window clip and the matte size only; the shot clips (`sK_clip.mp4`) carry the output size."""
    from tools.app import own_video as V1
    from tools.app import own_video_v2 as V2
    from tools.app import own_video_v3 as V3
    saved = {k: getattr(V3, k) for k in ("SIZE", "CLIP_SIZE", "GLOBAL_SIZE", "MATTE_FPS", "SRC_BOX", "BOX", "BOX_DIR", "H_NAME")}
    saved_v1 = (V1.BOX_IN, V1.SRC)
    saved_labels = V2.SEG_LABELS
    try:
        W, H = 1080, 1920
        spans, cuts, n = [48, 48], [48], 96
        V2.SEG_LABELS = [("subject", True), ("subject", True)]
        V3.SRC_BOX, V3.BOX, V3.BOX_DIR, V3.H_NAME = "window.mp4", "/box", "/box/w", "own_h"
        V1.BOX_IN, V1.SRC = "/box/input", "/box/w/music.wav"
        scripts = {}
        for scale in ("full", "half"):
            nat = OM.native_sizes(W, H, "draft", scale)
            V3.SIZE, V3.CLIP_SIZE, V3.MATTE_FPS = nat["matte"], tuple(nat["compose"]), "30"
            V3.GLOBAL_SIZE = tuple(nat["source"])
            segs = V2.segments(cuts, n, spans)
            scripts[scale] = V3.box_script(spans, segs, 30.0).splitlines()
        h_lines = {s: [l for l in L if "own_h" in l and ".mp4\"" not in l.split("-vf")[0] and "/box/input/own_h" in l] for s, L in scripts.items()}
        assert h_lines["full"] and h_lines["full"] == h_lines["half"]
        g_lines = {s: [l for l in L if "global_clip.mp4" in l and "-vf" in l] for s, L in scripts.items()}
        assert g_lines["full"] == g_lines["half"]
        s_lines = {s: [l for l in L if "/seg/s1_clip.mp4" in l and "-vf" in l] for s, L in scripts.items()}
        assert s_lines["full"] != s_lines["half"] and "scale=540:960" in s_lines["half"][0] and "scale=1080:1920" in s_lines["full"][0]
    finally:
        for k, v in saved.items():
            setattr(V3, k, v)
        V1.BOX_IN, V1.SRC = saved_v1
        V2.SEG_LABELS = saved_labels
