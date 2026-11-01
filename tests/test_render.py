"""Renderer, geometry and encoder tests (W1). Self-contained: hand-built plans and a
synthetic analysis dict from tests/render_helpers.py; no analyze / planner code."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mograph import encode  # noqa: E402
from mograph import geometry as G  # noqa: E402
from mograph.render import FONTS, render_frames  # noqa: E402
from mograph.render.shapes import PRIMITIVES, trim_polyline  # noqa: E402
from mograph.render.transitions import TRANSITIONS  # noqa: E402
from render_helpers import base_plan, box, make_analysis, showcase_plan, text_layer  # noqa: E402

OUT = os.environ.get("MOGRAPH_TEST_OUT") or os.path.join(__import__("tempfile").gettempdir(), "mograph_tests_out")  # outside PROJECTS: the dashboard indexes that tree
TEXT_IN = ["cut", "fade", "type_on", "word_stagger", "slide_up", "slide_left", "scale_pop",
           "mask_wipe", "blur_in", "kinetic_split"]
TEXT_OUT = ["cut", "fade", "slide_up", "slide_left", "scale_down", "mask_wipe", "blur_out"]
BG_MOTIONS = ["static", "ken_burns", "drift", "beat_zoom", "parallax_split"]


@pytest.fixture(scope="module")
def analysis():
    return make_analysis(scale=0.5)


def _scene(sid, t0, t1, bg, layers=(), tr=None):
    return {"id": sid, "t0": t0, "t1": t1, "background": bg, "layers": list(layers),
            "transition_out": tr or {"type": "cut", "dur": 0.0}}


def _img_bg(src="img0", verb="ken_burns", **kw):
    d = {"type": "image", "src": src, "focus": [0.45, 0.5],
         "motion": {"verb": verb, "zoom": [1.0, 1.15], "pan": [[0, 0], [0.02, -0.01]]},
         "treatment": {"duotone": None, "blur": 0.0, "darken": 0.2}}
    d.update(kw)
    return d


def _small_plan(duration=1.0, fps=30, layers=None, bg=None):
    p = base_plan(duration, fps)
    p["scenes"] = [_scene("s0", 0.0, duration, bg or _img_bg(), layers if layers is not None else [
        text_layer("Hello world", 0.0, duration, "fade", "fade"),
        {"type": "shape", "source": "contour", "ref": "img0:c0", "stroke": "palette:2", "stroke_width": 0.006,
         "fill": None, "fill_opacity": 0.0, "offset": [0, 0], "scale": 1.0, "t0": 0.0, "t1": duration,
         "in": {"verb": "stroke_draw", "dur": 0.5}, "out": {"verb": "fade", "dur": 0.2}, "emphasis": [],
         "echo": {"count": 2, "spacing": 0.015, "decay": 0.5}},
    ])]
    return p


def _render_all(plan, analysis, fmt="9:16", quality="draft", audio_env=None):
    W, H = G.frame_size(fmt, quality)
    frames = list(render_frames(plan, analysis, fmt, quality, audio_env))
    for f in frames:
        assert f.shape == (H, W, 3) and f.dtype == np.uint8
    return frames


# ----------------------------------------------------------------------------- geometry

def test_frame_size():
    assert G.frame_size("9:16", "full") == (1080, 1920)
    assert G.frame_size("16:9", "full") == (1920, 1080)
    assert G.frame_size("9:16", "draft") == (360, 640)
    assert G.frame_size("16:9", "draft") == (640, 360)
    with pytest.raises(ValueError):
        G.frame_size("4:3", "draft")


@pytest.mark.parametrize("img", [(1024, 768), (768, 1024), (500, 500), (4000, 1000)])
@pytest.mark.parametrize("frame", [(1080, 1920), (1920, 1080)])
def test_cover_crop_covers_and_clamps(img, frame):
    iw, ih = img
    fw, fh = frame
    for focus in [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.2, 0.9)]:
        for zoom in [1.0, 1.3, 0.5]:
            x, y, w, h = G.cover_crop(iw, ih, fw, fh, focus, zoom)
            assert -1e-9 <= x and -1e-9 <= y and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
            assert (w * iw) / (h * ih) == pytest.approx(fw / fh, rel=1e-6)   # frame aspect
            assert max(w, h) == pytest.approx(1.0 / max(1.0, zoom), rel=1e-6)  # covers at zoom
    # focus is centred when there is room
    x, y, w, h = G.cover_crop(4000, 1000, 1080, 1920, (0.5, 0.5))
    assert x + w / 2 == pytest.approx(0.5)


def test_point_round_trip_and_rect():
    rng = np.random.default_rng(0)
    crop = G.cover_crop(1024, 768, 360, 640, (0.3, 0.6), 1.2)
    for p in rng.random((50, 2)):
        q = G.image_to_frame(tuple(p), crop)
        r = G.frame_to_image(q, crop)
        assert r == pytest.approx(tuple(p), abs=1e-12)
    # crop corners map to frame corners
    assert G.image_to_frame((crop[0], crop[1]), crop) == pytest.approx((0.0, 0.0))
    assert G.image_to_frame((crop[0] + crop[2], crop[1] + crop[3]), crop) == pytest.approx((1.0, 1.0))
    # the whole image rect clips to the whole frame; a rect outside the crop collapses
    assert G.rect_image_to_frame((0, 0, 1, 1), crop) == pytest.approx((0, 0, 1, 1))
    x, y, w, h = G.rect_image_to_frame((0.0, 0.0, crop[0] * 0.5, 0.1), crop)
    assert w == 0.0
    inner = (crop[0] + 0.25 * crop[2], crop[1] + 0.25 * crop[3], 0.5 * crop[2], 0.5 * crop[3])
    assert G.rect_image_to_frame(inner, crop) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_trim_polyline():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    full = trim_polyline(sq, True, 0, 1)
    assert full[1] is True and len(full[0]) == 4
    half, closed = trim_polyline(sq, True, 0, 0.5)
    assert not closed and half[-1] == pytest.approx([10, 10])
    q, _ = trim_polyline(sq, True, 0.125, 0.375)
    assert q[0] == pytest.approx([5, 0]) and q[-1] == pytest.approx([10, 5])
    assert trim_polyline(sq, True, 0.4, 0.4) is None


def test_fonts_bundled():
    from PIL import ImageFont
    from mograph.font_registry import REGISTRY
    assert {"Big Shoulders Display", "Facade", "Space Grotesk", "Interlope", "Major Mono Display", "Anybody"} <= set(FONTS)
    assert set(FONTS) == set(REGISTRY) and len(FONTS) >= 20
    for name, path in FONTS.items():
        assert os.path.isfile(path), path
        ImageFont.truetype(path, 20)
        assert os.path.isfile(os.path.join(os.path.dirname(path), REGISTRY[name]["licence"])), name
    fdir = os.path.dirname(FONTS["Big Shoulders Display"])
    for fam in ["BigShouldersDisplay", "SpaceGrotesk", "Interlope", "MajorMonoDisplay", "Anybody"]:
        assert os.path.isfile(os.path.join(fdir, f"OFL-{fam}.txt")), fam
    assert os.path.isfile(os.path.join(fdir, "SpaceGrotesk-Bold.ttf"))


# ----------------------------------------------------------------------------- render basics

@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
@pytest.mark.parametrize("quality,duration,fps", [("draft", 2.0, 30), ("draft", 1.3, 24), ("full", 1.0, 30)])
def test_frame_count_and_size(analysis, fmt, quality, duration, fps):
    plan = _small_plan(duration, fps)
    frames = _render_all(plan, analysis, fmt, quality)
    assert len(frames) == round(duration * fps)
    assert frames[len(frames) // 2].std() > 5      # something was drawn


def test_render_is_lazy_generator(analysis):
    it = render_frames(_small_plan(1.0), analysis, "9:16")
    first = next(it)
    assert first.shape == (640, 360, 3)


def test_deterministic(analysis):
    plan = showcase_plan(2.0, 15)
    plan["scenes"] = plan["scenes"][:1]
    plan["scenes"][0]["t1"] = 2.0
    for ly in plan["scenes"][0]["layers"]:
        ly["t1"] = min(ly["t1"], 2.0)
    h = [hashlib.sha1(b"".join(f.tobytes() for f in render_frames(plan, analysis, "9:16"))).hexdigest()
         for _ in range(2)]
    assert h[0] == h[1]


def _two_scene_plan(tr, dur=0.5, fps=20, next_bg=None):
    p = base_plan(2.0, fps)
    p["scenes"] = [
        _scene("s0", 0.0, 1.0, _img_bg("img0"), [text_layer("First", 0.0, 1.0 + dur / 2, "fade", "fade")],
               {"type": tr, "dur": dur, "dir": "left"}),
        _scene("s1", 1.0, 2.0, next_bg or _img_bg("img1", "static"),
               [text_layer("Second", 1.0, 2.0, "fade", "fade")]),
    ]
    return p


@pytest.mark.parametrize("tr", list(TRANSITIONS) + ["not_a_transition"])
def test_every_transition(analysis, tr):
    plan = _two_scene_plan(tr)
    frames = _render_all(plan, analysis)
    assert len(frames) == 40
    a, mid, b = frames[14].astype(int), frames[20].astype(int), frames[26].astype(int)
    assert np.abs(a - b).mean() > 5           # the scenes differ
    if tr != "cut":
        assert np.abs(mid - a).mean() > 1 or np.abs(mid - b).mean() > 1


@pytest.mark.parametrize("d", ["left", "right", "up", "down"])
@pytest.mark.parametrize("tr", ["push", "wipe"])
def test_directional_transitions(analysis, tr, d):
    plan = _two_scene_plan(tr)
    plan["scenes"][0]["transition_out"]["dir"] = d
    assert len(_render_all(plan, analysis, "16:9")) == 40


def test_shape_reveal_falls_back_to_circle(analysis):
    plan = _two_scene_plan("shape_reveal", next_bg={"type": "gradient", "colors": ["palette:3", "palette:4"],
                                                    "angle": 45, "motion": {"verb": "drift"}})
    assert len(_render_all(plan, analysis)) == 40


@pytest.mark.parametrize("verb_in", TEXT_IN)
def test_text_in_verbs(analysis, verb_in):
    plan = _small_plan(1.0, 20, layers=[
        text_layer("Kinetic type test line", 0.0, 1.0, verb_in, "fade", highlight="palette:2",
                   **{"in": {"verb": verb_in, "dur": 0.5, "stagger_beats": 0.25}}),
        text_layer("caption here", 0.1, 1.0, verb_in, "cut", font="body", role="caption", case="as_is",
                   box=box([0.1, 0.85, 0.8, 0.06], [0.1, 0.85, 0.5, 0.08])),
    ])
    frames = _render_all(plan, analysis)
    assert len(frames) == 20
    if verb_in != "cut":
        assert np.abs(frames[2].astype(int) - frames[15].astype(int)).mean() > 0.5


@pytest.mark.parametrize("verb_out", TEXT_OUT)
def test_text_out_verbs(analysis, verb_out):
    plan = _small_plan(1.0, 20, layers=[
        text_layer("Out we go", 0.0, 0.9, "cut", verb_out, **{"out": {"verb": verb_out, "dur": 0.4}}),
    ])
    assert len(_render_all(plan, analysis)) == 20


@pytest.mark.parametrize("motion", ["snappy", "smooth", "bouncy", "glitch"])
def test_motion_styles(analysis, motion):
    plan = _small_plan(1.0, 15)
    plan["style"]["motion"] = motion
    assert len(_render_all(plan, analysis)) == 15


@pytest.mark.parametrize("verb", BG_MOTIONS)
def test_background_motions_and_treatments(analysis, verb):
    bg = _img_bg("img1", verb, treatment={"duotone": ["palette:0", "palette:4"], "blur": 0.3, "darken": 0.1})
    frames = _render_all(_small_plan(1.0, 15, bg=bg), analysis)
    assert len(frames) == 15
    if verb in ("ken_burns", "drift", "parallax_split"):
        assert np.abs(frames[0].astype(int) - frames[-3].astype(int)).mean() > 0.5


@pytest.mark.parametrize("bgtype", ["color", "gradient", "shapes"])
def test_non_image_backgrounds(analysis, bgtype):
    bg = {"type": bgtype, "color": "palette:0", "colors": ["palette:0", "palette:3"], "angle": 120,
          "motion": {"verb": "drift"}}
    assert len(_render_all(_small_plan(1.0, 15, bg=bg), analysis, "16:9")) == 15


def test_primitives_masks_emphasis_echo(analysis):
    layers = []
    for k, prim in enumerate(PRIMITIVES):
        layers.append({"type": "shape", "source": "primitive", "primitive": prim,
                       "box": box([0.1, 0.05 + 0.1 * k, 0.8, 0.08], [0.1, 0.05 + 0.1 * k, 0.4, 0.08]),
                       "stroke": "palette:2", "stroke_width": 0.004, "fill": "palette:3" if k % 2 else None,
                       "fill_opacity": 0.5, "offset": [0.01, 0], "scale": 1.0, "t0": 0.0, "t1": 1.0,
                       "in": {"verb": ["stroke_draw", "fade", "scale_pop", "cut"][k % 4], "dur": 0.4},
                       "out": {"verb": "fade", "dur": 0.2},
                       "emphasis": [{"verb": v, "on": on, "amount": 0.05}
                                    for v, on in [("pulse", "beats"), ("shake", "onsets"),
                                                  ("color_flip", [0.3, 0.6]), ("underline_draw", "downbeats")]],
                       "echo": {"count": 2, "spacing": 0.01, "decay": 0.5}})
    for k, mask in enumerate(["rect", "rounded", "circle", "contour:img2:c0", "contour:img0:c0"]):
        layers.append({"type": "image", "src": "img2", "mask": mask,
                       "box": box([0.05 + 0.18 * k, 0.8, 0.16, 0.1], [0.55, 0.05 + 0.18 * k, 0.2, 0.15]),
                       "t0": 0.1, "t1": 1.0, "in": {"verb": TEXT_IN[k], "dur": 0.3},
                       "out": {"verb": TEXT_OUT[k], "dur": 0.2},
                       "emphasis": [{"verb": "pulse", "on": "beats", "amount": 0.05},
                                    {"verb": "underline_draw", "on": "beats"}],
                       "border": {"color": "palette:1", "width": 0.004}})
    layers.append({"type": "shape", "source": "line", "ref": "img0:l0", "stroke": "palette:4",
                   "stroke_width": 0.008, "t0": 0.0, "t1": 1.0, "in": {"verb": "stroke_draw", "dur": 0.5},
                   "out": {"verb": "stroke_draw", "dur": 0.3}, "emphasis": [],
                   "echo": {"count": 2, "spacing": 0.02, "decay": 0.5}})
    n = 20
    env = {k: np.abs(np.sin(np.linspace(0, 6, n) + i)).astype(np.float32)
           for i, k in enumerate(["rms", "low", "mid", "high", "onset"])}
    plan = _small_plan(1.0, 20, layers=layers)
    assert len(_render_all(plan, analysis, "9:16", audio_env=env)) == n
    assert len(_render_all(plan, analysis, "16:9", audio_env=None)) == n


def test_wave_reads_audio_env(analysis):
    layer = {"type": "shape", "source": "primitive", "primitive": "wave",
             "box": box([0.05, 0.4, 0.9, 0.2], [0.05, 0.4, 0.9, 0.2]), "stroke": "#ffffff",
             "stroke_width": 0.006, "t0": 0.0, "t1": 1.0, "in": {"verb": "cut"}, "out": {"verb": "cut"},
             "emphasis": [], "echo": None}
    bg = {"type": "color", "color": "#000000"}
    plan = _small_plan(1.0, 10, layers=[layer], bg=bg)
    plan["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
    quiet = {k: np.zeros(10, np.float32) for k in ["rms", "low", "mid", "high", "onset"]}
    loud = {k: np.ones(10, np.float32) for k in ["rms", "low", "mid", "high", "onset"]}
    fq = _render_all(plan, analysis, audio_env=quiet)[3]
    fl = _render_all(plan, analysis, audio_env=loud)[3]
    rows_q = np.nonzero(fq.max(axis=(1, 2)) > 60)[0]
    rows_l = np.nonzero(fl.max(axis=(1, 2)) > 60)[0]
    assert (rows_l.max() - rows_l.min()) > (rows_q.max() - rows_q.min()) + 10


def test_unknown_values_do_not_crash(analysis):
    plan = _small_plan(1.0, 10, layers=[
        text_layer("Odd", 0.0, 1.0, "teleport", "evaporate", font="NoSuchFont"),
        {"type": "shape", "source": "primitive", "primitive": "hexagon", "box": None, "stroke": "palette:9",
         "t0": 0.0, "t1": 1.0, "in": {"verb": "spin"}, "out": {"verb": "melt"}, "emphasis": [{"verb": "wobble"}]},
        {"type": "shape", "source": "contour", "ref": "img9:c0", "t0": 0.0, "t1": 1.0},
        {"type": "image", "src": "img9", "mask": "star", "t0": 0, "t1": 1},
        {"type": "image", "src": "img1", "mask": "contour:img1:c99", "t0": 0, "t1": 1},
        {"type": "mystery"},
    ], bg={"type": "image", "src": "missing", "motion": {"verb": "orbit"}})
    plan["scenes"][0]["transition_out"] = {"type": "warp_drive", "dur": 0.4}
    assert len(_render_all(plan, analysis)) == 10


# ----------------------------------------------------------------------------- outline alignment

def test_contour_outline_sits_on_image_feature(analysis):
    """The mapped contour must trace the blob's edge in the rendered background while ken_burns moves."""
    def plan_with(layers):
        p = _small_plan(2.0, 10, layers=layers,
                        bg=_img_bg("img0", "ken_burns", treatment={"duotone": None, "blur": 0, "darken": 0},
                                   motion={"verb": "ken_burns", "zoom": [1.0, 1.35], "pan": [[0, 0], [0.05, 0.02]]}))
        p["style"].update(grain=0.0, vignette=0.0, beat_flash=0.0)
        return p

    outline = {"type": "shape", "source": "contour", "ref": "img0:c0", "stroke": "#00ff00",
               "stroke_width": 0.004, "t0": 0.0, "t1": 2.0, "in": {"verb": "cut"}, "out": {"verb": "cut"},
               "emphasis": [], "echo": None}
    for fmt in ("9:16", "16:9"):
        bare = _render_all(plan_with([]), analysis, fmt)
        drawn = _render_all(plan_with([outline]), analysis, fmt)
        for fi in (1, 8, 15):   # before the final 0.3 s fade
            bg = bare[fi]
            orange = (bg[..., 0].astype(int) - bg[..., 2].astype(int)) > 80   # blob + inner blob are warm
            edges = cv2.morphologyEx(orange.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            dist = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
            green = (drawn[fi][..., 1].astype(int) - bg[..., 1].astype(int)) > 60
            assert green.sum() > 50
            assert float(np.median(dist[green])) <= 1.5, (fmt, fi, float(np.median(dist[green])))


# ----------------------------------------------------------------------------- encode

def _probe(path):
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                            "stream=codec_type,nb_read_frames:format=duration", "-of", "json", path],
                           capture_output=True, text=True, check=True)
        d = json.loads(r.stdout)
        v = [s for s in d["streams"] if s["codec_type"] == "video"][0]
        a = [s for s in d["streams"] if s["codec_type"] == "audio"]
        return float(d["format"]["duration"]), int(v["nb_read_frames"]), bool(a)
    av = pytest.importorskip("av")
    with av.open(path) as c:
        n = sum(1 for _ in c.decode(video=0))
        return float(c.duration) / 1e6, n, any(s.type == "audio" for s in c.streams)


def _backends():
    out = []
    try:
        import av  # noqa: F401
        out.append("pyav")
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        out.append("ffmpeg")
    return out


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("with_audio", [True, False])
@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
def test_encode_mp4(analysis, monkeypatch, backend, with_audio, fmt):
    monkeypatch.setattr(encode, "BACKEND", backend)
    plan = _small_plan(1.5, 30)
    W, H = G.frame_size(fmt, "draft")
    sr = 48000
    audio = None
    if with_audio:
        t = np.arange(int(1.2 * sr)) / sr      # shorter than the video: must be padded
        audio = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)]).astype(np.float32) * 0.3
    path = os.path.join(OUT, f"test_encode_{backend}_{'a' if with_audio else 'na'}_{fmt.replace(':', 'x')}.mp4")
    out = encode.encode_mp4(render_frames(plan, analysis, fmt), 30, W, H, audio, sr, path)
    assert os.path.isfile(out)
    dur, n, has_audio = _probe(out)
    assert n == 45
    assert abs(dur - 1.5) <= 0.05
    assert has_audio == with_audio


def test_encode_long_audio_is_cut(monkeypatch):
    for backend in _backends():
        monkeypatch.setattr(encode, "BACKEND", backend)
        frames = (np.full((64, 96, 3), k * 8, np.uint8) for k in range(30))
        audio = np.zeros((2, 48000 * 3), np.float32)
        p = encode.encode_mp4(frames, 30, 96, 64, audio, 48000, os.path.join(OUT, f"test_encode_cut_{backend}.mp4"))
        dur, n, has_audio = _probe(p)
        assert n == 30 and abs(dur - 1.0) <= 0.05 and has_audio


def test_encoder_falls_back_to_ffmpeg_without_pyav(monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg binary")
    monkeypatch.setattr(encode, "BACKEND", None)
    monkeypatch.delenv("MOGRAPH_ENCODER", raising=False)
    monkeypatch.setitem(sys.modules, "av", None)      # makes `import av` raise ImportError
    assert encode._choose_backend() == "ffmpeg"
    frames = (np.full((64, 96, 3), k * 8, np.uint8) for k in range(15))
    p = encode.encode_mp4(frames, 30, 96, 64, None, 48000, os.path.join(OUT, "test_encode_fallback.mp4"))
    dur, n, _ = _probe(p)
    assert n == 15


def test_encode_rejects_bad_frames(monkeypatch):
    for backend in _backends():
        monkeypatch.setattr(encode, "BACKEND", backend)
        with pytest.raises(ValueError):
            encode.encode_mp4(iter([np.zeros((10, 10, 3), np.uint8)]), 30, 96, 64, None, 48000,
                              os.path.join(OUT, f"test_encode_bad_{backend}.mp4"))
