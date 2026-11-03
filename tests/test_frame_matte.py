"""tools/app/frame_matte.py: which frames a single-frame matte is cut from, its cache key, the request states. No GPU, no ComfyUI; ffprobe is replaced."""
import time

import pytest

from tools.app import frame_matte as FM


def test_clip_is_nine_real_frames_from_the_shots_frame():
    assert FM.clip_plan(1.72, 8.0, 25.0) == (43, 9, 0)


def test_clip_never_crosses_the_shots_end():
    # a 10-frame shot from frame 338: 4 frames left after frame 344, the rest held
    assert FM.clip_plan(344 / 25, 348 / 25, 25.0) == (344, 4, 5)


def test_clip_stops_at_the_videos_last_frame():
    assert FM.clip_plan(26.96, None, 25.0, nframes=675) == (674, 1, 8)


def test_clip_always_has_a_frame_and_the_full_length():
    first, real, pad = FM.clip_plan(5.0, 5.0, 25.0)
    assert (first, real, real + pad) == (125, 1, FM.FRAMES)


def test_key_follows_file_frame_and_size_only():
    k = FM.matte_key("a.mp4", 10, 20, 43, (576, 1024))
    assert k == FM.matte_key("a.mp4", 10, 20, 43, (576, 1024))
    assert k != FM.matte_key("a.mp4", 10, 20, 44, (576, 1024))
    assert k != FM.matte_key("a.mp4", 11, 20, 43, (576, 1024))
    assert k != FM.matte_key("a.mp4", 10, 20, 43, (704, 1280))


@pytest.mark.parametrize("name", ["", None, "../outside.mp4", "missing.mp4"])
def test_only_files_in_the_input_folder(tmp_path, name):
    with pytest.raises(ValueError):
        FM.safe_video(tmp_path, name)


@pytest.mark.parametrize("value", ["x", None, -1, float("inf")])
def test_bad_times_are_refused(value):
    with pytest.raises(ValueError):
        FM.number(value, "t")


@pytest.fixture
def box(tmp_path, monkeypatch):
    inp, out = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    out.mkdir()
    (inp / "clip.mp4").write_bytes(b"video")
    monkeypatch.setattr(FM, "probe", lambda p: (1080, 1920, 25.0, 675))
    return inp, out


def wait_until(state, fm, *req):
    for _ in range(200):
        got = fm.request(*req)
        if got[0] == state:
            return got
        time.sleep(0.01)
    raise AssertionError(f"never {state}: {got}")


def test_one_job_per_frame_then_ready_without_another_run(box):
    inp, out = box
    calls = []

    def make(key, src, first, real, pad, fps, size, png):
        calls.append((first, real, pad, size))
        time.sleep(0.05)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")

    fm = FM.FrameMattes(out, inp, 8188, make=make)
    assert fm.request("clip.mp4", 1.72, 8.0)[0] == "making"
    assert fm.request("clip.mp4", 1.72, 8.0)[0] == "making"
    state, path = wait_until("ready", fm, "clip.mp4", 1.72, 8.0)
    assert path.read_bytes() == b"png"
    fm.request("clip.mp4", 1.72, 8.0)
    assert calls == [(43, 9, 0, (576, 1024))]


def test_a_failure_is_reported_once_then_retried(box):
    inp, out = box
    calls = []

    def make(key, src, first, real, pad, fps, size, png):
        calls.append(first)
        if len(calls) == 1:
            raise RuntimeError("no model")
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")

    fm = FM.FrameMattes(out, inp, 8188, make=make)
    fm.request("clip.mp4", 1.72, 8.0)
    assert wait_until("error", fm, "clip.mp4", 1.72, 8.0)[1] == "no model"
    wait_until("ready", fm, "clip.mp4", 1.72, 8.0)
    assert len(calls) == 2


def test_the_graph_saves_pngs_only():
    g = FM.alpha_graph("frame_matte_x.mp4", (576, 1024), "motngrfx_app/frame_mattes/x/m")
    kinds = [v["class_type"] for v in g.values()]
    assert "SaveVideo" not in kinds and "SaveImage" in kinds
