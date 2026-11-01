"""Unit tests for tools/app/ingest.py (fps policy, aspect policy, probe, emitted ffmpeg). Synthetic inputs, no ffmpeg / GPU needed."""
import os
import sys
from fractions import Fraction

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "app"))

import ingest as I  # noqa: E402


# ------------------------------------------------------------------------------------------------------------ fps
@pytest.mark.parametrize("src,out,policy,step", [
    ("30", "30", "keep", 1), ("30000/1001", "30000/1001", "keep", 1), (29.97, "30000/1001", "keep", 1),
    ("24", "24", "keep", 1), (23.976, "24000/1001", "keep", 1), ("25", "25", "keep", 1),
    ("60", "30", "halve", 2), ("60000/1001", "30000/1001", "halve", 2), (59.94, "30000/1001", "halve", 2),
    ("50", "25", "halve", 2), ("48", "24", "halve", 2), (47.95, "24000/1001", "halve", 2),
    ("15", "15", "keep", 1), ("20", "20", "keep", 1), ("120", "30", "divide4", 4), ("90", "30", "divide3", 3),
])
def test_fps_table(src, out, policy, step):
    r = I.timeline_fps(src)
    assert r["fps"] == out and r["policy"] == policy and r["step"] == step


@pytest.mark.parametrize("src", ["30", "29.97", "25", "24", "60", "59.94", "50", "48", "15", "20", "120", "23.976"])
def test_fps_never_fractional_resample(src):
    r = I.timeline_fps(src)
    ratio = I.snap_rate(src) / Fraction(r["fps"])
    assert ratio.denominator == 1 and ratio >= 1          # an integer frame ratio: no 30 -> 24
    assert r["fps"] != "24" or src in ("24", "48")


def test_30_never_becomes_24():
    assert I.timeline_fps("30")["fps"] == "30"
    assert I.timeline_fps("30000/1001")["fps"] == "30000/1001"


def test_halve_filter_selects_every_other_frame_without_blend():
    f = I.timeline_fps("60000/1001")["filter"]
    assert "select='not(mod(n,2))'" in f and "setpts=N/((30000/1001)*TB)" in f and f.endswith("fps=30000/1001")
    assert "tblend" not in f and "minterpolate" not in f and "framerate" not in f


def test_vfr_goes_to_nearest_standard_by_avg():
    assert I.detect_vfr("30/1", "27/1") and not I.detect_vfr("30/1", "299/10")
    r = I.timeline_fps("30/1", vfr=True, avg_fps="27/1")
    assert r["fps"] == "25" and r["policy"] == "vfr_const" and r["filter"] == "fps=25"
    assert I.timeline_fps("1000/1", vfr=True, avg_fps="59/2")["fps"] == "30"
    assert I.timeline_fps("1000/1", vfr=True, avg_fps="23.5")["fps"] == "24"
    assert I.timeline_fps("90000/1", vfr=True, avg_fps="58/1")["fps"] == "30"      # high-rate VFR: halved first


# ------------------------------------------------------------------------------------------------------------ probe
def _ff(w=1080, h=1920, rot=None, r="30/1", avg="30/1", sar="1:1", fmt="mov,mp4,m4a,3gp,3g2,mj2", tag_rot=None, audio=True):
    v = {"codec_type": "video", "codec_name": "h264", "width": w, "height": h, "r_frame_rate": r, "avg_frame_rate": avg,
         "sample_aspect_ratio": sar, "duration": "10.0", "nb_frames": "300", "pix_fmt": "yuv420p"}
    if rot is not None:
        v["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rot}]
    if tag_rot is not None:
        v["tags"] = {"rotate": str(tag_rot)}
    st = [v] + ([{"codec_type": "audio", "codec_name": "aac", "duration": "10.0", "sample_rate": "48000", "channels": 2}] if audio else [])
    return {"streams": st, "format": {"format_name": fmt, "duration": "10.0"}}


def test_probe_basic_and_audio():
    p = I.parse_probe(_ff())
    assert (p["display_w"], p["display_h"], p["kind"], p["vfr"]) == (1080, 1920, "video", False)
    assert p["audio"]["codec"] == "aac" and I.parse_probe(_ff(audio=False))["audio"] is None


@pytest.mark.parametrize("rot", [90, -90, 270])
def test_rotation_swaps_wh(rot):
    p = I.parse_probe(_ff(1920, 1080, rot=rot))
    assert (p["display_w"], p["display_h"]) == (1080, 1920)


def test_rotation_tag_and_180():
    assert (I.parse_probe(_ff(1920, 1080, tag_rot=90))["display_w"]) == 1080
    p = I.parse_probe(_ff(1920, 1080, rot=180))
    assert (p["display_w"], p["display_h"]) == (1920, 1080)


def test_probe_vfr_and_image():
    assert I.parse_probe(_ff(r="30/1", avg="26/1"))["vfr"]
    assert I.parse_probe(_ff(fmt="png_pipe"))["kind"] == "image"


# ------------------------------------------------------------------------------------------------------------ aspect
def _times(n=48, fps=24.0):
    return [i / fps for i in range(n)]


def _track(boxes, fps=24):
    return {"fps": fps, "boxes": boxes}


def test_window_exact_ratio_integer():
    assert I.largest_window(1920, 1080) == (603, 1072)
    assert I.largest_window(1080, 1920) == (1080, 1920)
    assert I.largest_window(1080, 2340) == (1080, 1920)


def test_portrait_no_track_centre_crop_and_landscape_fit():
    s = I.plan_shot(720, 1280, _times(), None, None, fps=24)          # exact 9:16
    assert s["mode"] == "crop" and s["reason"] == "native" and s["static"]
    s = I.plan_shot(1080, 2340, _times(), None, None, fps=24)         # tall phone: centre crop
    assert s["mode"] == "crop" and s["reason"] == "default_centre_crop" and s["xy"] == [0.0, 210.0]
    s = I.plan_shot(1920, 1080, _times(), None, None, fps=24)
    assert s["mode"] == "fit" and s["reason"] == "default_fit"


def test_near_target_portrait_light_crop():
    s = I.plan_shot(1080, 1800, _times(), None, None, fps=24)         # 0.6: within 10 % of 0.5625
    assert s["mode"] == "crop" and s["reason"] == "near_target"
    assert s["window"] == [1080 // 9 * 9, 1080 // 9 * 16][:2] or s["window"][0] / s["window"][1] == 9 / 16


def test_landscape_subject_fits_window_is_crop_and_follows_subject():
    n = 48
    boxes = [[0.20 + 0.01 * i / n * 10, 0.2, 0.30 + 0.01 * i / n * 10, 0.9] for i in range(n)]     # 10 % wide, moves right
    s = I.plan_shot(1920, 1080, _times(n), _track(boxes), None, fps=24)
    assert s["mode"] == "crop" and s["reason"] == "subject_fits" and s["window"] == [603, 1072]
    xs = [p[0] for p in s["path"]]
    assert xs[-1] > xs[0]                                              # follows the subject to the right
    # window contains the subject centroid
    for i, (x, _) in enumerate(s["path"]):
        cx = (boxes[i][0] + boxes[i][2]) / 2 * 1920
        assert x <= cx <= x + 603


def test_wide_subject_goes_fit_with_ground():
    boxes = [[0.1, 0.2, 0.9, 0.9]] * 24                                # a group across the frame
    s = I.plan_shot(1920, 1080, _times(24), _track(boxes), None, fps=24, ground="#102030")
    assert s["mode"] == "fit" and s["reason"] == "subject_wider_than_window" and s["ground"] == "#102030" and s["window"] is None


def test_subject_padding_decides_borderline():
    # window is 603 px of 1920 = 31.4 %: a box 29.5 % wide fits alone, but padded 8 % per side (x1.16 = 34.2 %) it does not
    boxes = [[0.35, 0.2, 0.645, 0.9]] * 24
    assert I.plan_shot(1920, 1080, _times(24), _track(boxes), None, fps=24)["mode"] == "fit"
    boxes = [[0.35, 0.2, 0.60, 0.9]] * 24                              # 25 % -> 29 % padded
    assert I.plan_shot(1920, 1080, _times(24), _track(boxes), None, fps=24)["mode"] == "crop"


def test_smoothing_clamps_inside_frame_and_edge_subject():
    n = 60
    # subject hugs the left edge, then jumps right (union still fits one window): path stays inside [0, W - ww] and is smoothed
    boxes = [[0.0, 0.2, 0.06, 0.9]] * 30 + [[0.17, 0.2, 0.23, 0.9]] * 30
    s = I.plan_shot(1920, 1080, _times(n), _track(boxes), None, fps=24)
    assert s["mode"] == "crop"
    for x, y in s["path"]:
        assert 0 <= x <= 1920 - 603 and 0 <= y <= 1080 - 1072
    steps = [abs(b[0] - a[0]) for a, b in zip(s["path"], s["path"][1:])]
    assert max(steps) < 0.17 * 1920 / 6 and steps and min(p[0] for p in s['path']) >= 0   # an unsmoothed jump is ~326 px in one frame
    assert I.smooth([0, 0, 10, 0, 0], 3)[2] == pytest.approx(10 / 3)


def test_smooth_edges_replicated():
    out = I.smooth([5.0] * 10, 5)
    assert all(abs(v - 5.0) < 1e-9 for v in out)


def test_caption_avoidance_pushes_window_off_caption():
    n = 48
    # subject centred (x 0.45-0.55); caption on the right of the window position, active the whole clip
    boxes = [[0.45, 0.2, 0.55, 0.9]] * n
    cap = [{"box": [0.60, 0.8, 0.75, 0.95], "t0": 0.0, "t1": 10.0}]
    free = I.plan_shot(1920, 1080, _times(n), _track(boxes), None, fps=24)
    assert free["static"]
    # window x in free case = 960 - 301.5 = ~658 -> spans 658..1261 ; caption starts at 1152 -> overlap
    assert free["xy"][0] + 603 > 0.60 * 1920
    s = I.plan_shot(1920, 1080, _times(n), _track(boxes), cap, fps=24)
    assert s["captions"][0]["avoided"] and s["captions"][0]["residual_overlap"] == 0.0
    x = s["xy"][0] if s["static"] else s["path"][0][0]
    assert x + 603 <= 0.60 * 1920 + 1e-6
    # the subject (padded) is still inside the window
    assert x <= 0.45 * 1920 and x + 603 >= 0.55 * 1920


def test_caption_only_in_its_time_span():
    n = 96
    boxes = [[0.45, 0.2, 0.55, 0.9]] * n
    cap = [{"box": [0.60, 0.8, 0.75, 0.95], "t0": 2.0, "t1": 3.0}]
    s = I.plan_shot(1920, 1080, _times(n), _track(boxes), cap, fps=24)
    assert not s["static"]
    xs = [p[0] for p in s["path"]]
    assert xs[0] > xs[60] or xs[0] != xs[60]
    assert xs[60] + 603 <= 0.60 * 1920 + 1e-6                          # inside the span (t = 2.5 s)
    assert s["captions"][0]["avoided"]


def test_caption_unavoidable_is_reported():
    n = 24
    boxes = [[0.2, 0.2, 0.8, 0.9]] * n                                 # subject nearly fills the window range... fit mode instead
    s = I.plan_shot(720, 1280, _times(n), None, [{"box": [0.1, 0.9, 0.9, 0.98], "t0": 0, "t1": 5}], fps=24)
    assert s["mode"] == "crop" and s["captions"][0]["avoided"] is False and s["captions"][0]["residual_overlap"] > 0


def test_never_upscale_tier_drop():
    # 540x960 portrait: window 540x960 feeds draft, not final
    p = I.parse_probe(_ff(540, 960))
    plan = I.build_plan(p, "final")
    assert plan["max_tier"] == "draft" and plan["effective_tier"] == "draft" and plan["warnings"]
    assert plan["shots"][0]["max_tier"] == "draft"
    # 1920x1080 landscape cropped: window 603x1072 feeds draft (540 wide) not final
    plan = I.build_plan(I.parse_probe(_ff(1920, 1080)), "final", track=_track([[0.4, 0.2, 0.5, 0.9]] * 300, 30))
    assert plan["shots"][0]["mode"] == "crop" and plan["max_tier"] == "draft"
    # landscape fit: width 1920 >= 1080: feeds final
    plan = I.build_plan(I.parse_probe(_ff(1920, 1080)), "final")
    assert plan["shots"][0]["mode"] == "fit" and plan["max_tier"] == "final"
    # tiny source: even preview needs an upscale, flagged
    plan = I.build_plan(I.parse_probe(_ff(320, 568)), "preview")
    assert plan["shots"][0]["max_tier"] is None and plan["effective_tier"] == "preview" and any("upscale" in w for w in plan["warnings"])


def test_max_tier_in_render_commands_clips_request():
    plan = I.build_plan(I.parse_probe(_ff(540, 960)), "preview")
    c = I.render_commands(plan, "final", "out")[0]
    assert c["size"] == [540, 960] and c["tier"] == "draft"


# ------------------------------------------------------------------------------------------------------------ plan + commands
def test_build_plan_frames_and_fps():
    plan = I.build_plan(I.parse_probe(_ff(1080, 1920, r="60000/1001", avg="60000/1001")), "preview")
    assert plan["fps"]["fps"] == "30000/1001" and plan["frames"] == round(10.0 * 30000 / 1001)
    assert plan["shots"][0]["f0"] == 0 and plan["shots"][-1]["f1"] == plan["frames"]


def test_cuts_make_shots_with_independent_modes():
    boxes_a = [[0.45, 0.2, 0.55, 0.9]] * 150
    boxes_b = [[0.05, 0.2, 0.95, 0.9]] * 150
    p = I.parse_probe(_ff(1920, 1080, r="30/1", avg="30/1"))
    plan = I.build_plan(p, "preview", track=_track(boxes_a + boxes_b, 30), cuts=[5.0], ground_fn=lambda t: "#202020")
    assert [s["mode"] for s in plan["shots"]] == ["crop", "fit"] and plan["shots"][1]["ground"] == "#202020"
    assert plan["shots"][0]["f1"] == plan["shots"][1]["f0"] == 150
    cmds = I.render_commands(plan, "preview", "out")
    assert cmds[0]["kind"] == "pipe"


def test_static_crop_ffmpeg_command():
    p = I.parse_probe(_ff(1920, 1080, r="30000/1001", avg="30000/1001"))
    p["path"] = "in.mp4"
    plan = I.build_plan(p, "preview", track=_track([[0.45, 0.2, 0.55, 0.9]] * 300, 30))
    c = I.render_commands(plan, "preview", "out")[0]
    assert c["kind"] == "ffmpeg" and c["size"] == [360, 640]
    vf = c["argv"][c["argv"].index("-vf") + 1]
    assert vf.startswith("fps=30000/1001,crop=603:1072:") and "exact=1" in vf and "scale=360:640:flags=lanczos" in vf
    assert "-c:a" in c["argv"] and c["argv"][c["argv"].index("-c:a") + 1] == "copy"
    assert c["out"].replace("\\", "/").endswith("out/in_ingest_preview.mp4")


def test_fit_ffmpeg_command_pads_with_ground_and_halves_60():
    p = I.parse_probe(_ff(1920, 1080, r="60/1", avg="60/1"))
    p["path"] = "in.mp4"
    plan = I.build_plan(p, "preview", ground_fn=lambda t: "#1a2b3c")
    c = I.render_commands(plan, "preview", "out")[0]
    vf = c["argv"][c["argv"].index("-vf") + 1]
    assert vf.startswith("select='not(mod(n,2))',setpts=N/((30/1)*TB),fps=30,")
    assert "scale=360:202:flags=lanczos,pad=360:640:0:218:color=0x1a2b3c" in vf


def test_fit_geometry_never_larger_than_target():
    sw, sh, px, py = I.fit_geometry(1920, 1080, 360, 640)
    assert (sw, sh) == (360, 202) and px == 0 and py % 2 == 0 and sh + 2 * py <= 640


def test_audio_not_mp4_is_reencoded():
    j = _ff()
    j["streams"][1]["codec_name"] = "opus"
    p = I.parse_probe(j)
    p["path"] = "in.webm"
    plan = I.build_plan(p, "preview")
    argv = I.render_commands(plan, "preview", "o")[0]["argv"]
    assert argv[argv.index("-c:a") + 1] == "aac"


def test_still_plan_no_fps():
    j = _ff(1080, 1350, fmt="png_pipe", audio=False)
    j["streams"][0].pop("duration"); j["format"].pop("duration")
    p = I.parse_probe(j)
    p["path"] = "pic.png"
    plan = I.build_plan(p, "preview", ground_fn=lambda t: "#000000")
    assert plan["kind"] == "image" and plan["fps"] is None and plan["frames"] == 1
    c = I.render_commands(plan, "preview", "o")[0]
    assert c["argv"][-1].replace("\\", "/").endswith("o/pic_ingest_preview.png") and "-frames:v" in c["argv"]


def test_ground_colour_border_median():
    np = pytest.importorskip("numpy")
    a = np.zeros((40, 40, 3), np.uint8)
    a[...] = (200, 100, 50)
    a[10:30, 10:30] = (0, 255, 0)                                      # subject in the middle is ignored
    assert I.ground_colour(a) == "#c86432"


def test_compose_sizes_match_plan_rules():
    try:
        sys.path.insert(0, ROOT)
        from tools.fusion import plan_rules as pr
    except Exception as e:                                             # heavy imports may be missing on a bare box
        pytest.skip(f"plan_rules not importable: {e}")
    for t in I.TIERS:
        assert tuple(pr.sizes_for(t)["compose"]) == I.COMPOSE[t]
