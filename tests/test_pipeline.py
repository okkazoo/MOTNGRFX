"""End-to-end: analyze -> plan -> validate -> soundtrack -> render (draft, both formats) -> encode.

Writes MP4s to tests/out/ (gitignored). Skips cleanly while the analysis / synth / render
modules are not available yet.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

analyze = pytest.importorskip("mograph.analyze")
synth = pytest.importorskip("mograph.synth")
mix = pytest.importorskip("mograph.mix")
encode = pytest.importorskip("mograph.encode")
geometry = pytest.importorskip("mograph.geometry")
render = pytest.importorskip("mograph.render")
if not hasattr(render, "render_frames"):
    pytest.skip("mograph.render.render_frames not available yet", allow_module_level=True)
if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg/ffprobe not on PATH", allow_module_level=True)

from PIL import Image  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from mograph import plan as P  # noqa: E402
from mograph.planner_rules import plan_from_analysis  # noqa: E402
from mograph.soundtrack import build_soundtrack  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures")
OUT = os.environ.get("MOGRAPH_TEST_OUT") or os.path.join(__import__("tempfile").gettempdir(), "mograph_tests_out")  # outside PROJECTS: the dashboard indexes that tree
TEXT = "Make it move\nShapes from your pictures\nCut on the beat"


def _img(name: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(FIX, name)).convert("RGB"), np.float32) / 255.0


def _wav(name: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(os.path.join(FIX, name))
    a = data.astype(np.float32)
    if data.dtype == np.int16:
        a /= 32768.0
    a = a[None, :] if a.ndim == 1 else a.T
    return a, int(sr)


CASES = {
    "text_only": lambda: (TEXT, [], None),
    "images_only": lambda: ("", [_img("blob_gradient.png"), _img("stripes.png")], None),
    "text_2images": lambda: (TEXT, [_img("blob_gradient.png"), _img("subject_texture.png")], None),
    "text_images_wav": lambda: (TEXT, [_img("blob_gradient.png"), _img("stripes.png")], _wav("beat_120.wav")),
}

_cache: dict = {}


def _prepare(case: str):
    if case not in _cache:
        text, images, audio = CASES[case]()
        analysis = analyze.analyze_inputs(text, images, audio, None)
        plan = plan_from_analysis(analysis, seed=0)
        errors = P.validate_plan(plan, analysis)
        assert errors == [], errors
        wave, sr = build_soundtrack(plan, analysis)
        assert wave.ndim == 2 and wave.shape[1] == int(round(plan["duration"] * sr))
        _cache[case] = (analysis, plan, wave, sr)
    return _cache[case]


def _ffprobe(path: str, *args: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", *args, path], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.mark.parametrize("fmt", ["9:16", "16:9"])
@pytest.mark.parametrize("case", list(CASES))
def test_pipeline(case, fmt):
    analysis, plan, wave, sr = _prepare(case)
    fps = plan["fps"]
    n_frames = int(round(plan["duration"] * fps))
    width, height = geometry.frame_size(fmt, "draft")
    env = analyze.audio_envelope(wave, sr, fps, n_frames)

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{case}_{fmt.replace(':', 'x')}.mp4")
    with open(os.path.join(OUT, f"{case}_plan.json"), "w", encoding="utf-8") as f:
        f.write(P.to_json(plan))
    t = time.perf_counter()
    frames = render.render_frames(plan, analysis, fmt, "draft", audio_env=env)
    path = encode.encode_mp4(frames, fps, width, height, wave, sr, out)
    elapsed = time.perf_counter() - t
    print(f"{case} {fmt}: {n_frames} frames {width}x{height} in {elapsed:.1f}s")

    assert os.path.isfile(path) and os.path.getsize(path) > 0
    info = json.loads(_ffprobe(path, "-count_frames", "-show_entries",
                               "format=duration:stream=codec_type,nb_read_frames,width,height", "-of", "json"))
    dur = float(info["format"]["duration"])
    assert abs(dur - plan["duration"]) <= 0.1, (dur, plan["duration"])
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(video) == 1 and int(video[0]["nb_read_frames"]) == n_frames
    assert (int(video[0]["width"]), int(video[0]["height"])) == (width, height)
    assert len(audio) == 1
