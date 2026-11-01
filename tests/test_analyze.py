"""Unit tests for mograph.analyze (W2). Self-contained synthetic inputs."""

import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mograph import analyze as A  # noqa: E402

H, W = 768, 1024


def _hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255.0


@pytest.fixture(scope="module")
def circle_img():
    """Dark blue gradient, a large orange disc, a white horizontal bar."""
    img = np.zeros((H, W, 3), np.float32)
    g = np.linspace(0.12, 0.3, W, dtype=np.float32)
    img[:] = np.stack([g * 0.5, g * 0.7, g], -1)[None]
    cv2.circle(img, (700, 380), 200, (0.95, 0.45, 0.1), -1)
    cv2.rectangle(img, (80, 620), (600, 650), (0.95, 0.95, 0.95), -1)
    return img


@pytest.fixture(scope="module")
def circle_res(circle_img):
    return A.analyze_image(circle_img, "img0")


def test_image_shape(circle_res):
    r = circle_res
    for k in ("id", "pixels", "w", "h", "palette", "accent", "luma_mean", "contrast", "saturation",
              "saliency_box", "focus", "empty_regions", "contours", "lines", "meaning"):
        assert k in r
    assert r["id"] == "img0" and r["w"] == W and r["h"] == H
    assert r["pixels"].dtype == np.float32 and r["pixels"].shape == (H, W, 3)
    assert r["meaning"] is None
    for k in ("luma_mean", "contrast", "saturation"):
        assert 0.0 <= r[k] <= 1.0
    x, y, w, h = r["saliency_box"]
    assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
    assert all(0 <= v <= 1 for v in r["focus"])


def test_palette(circle_res):
    pal = circle_res["palette"]
    assert 1 <= len(pal) <= 6
    assert abs(sum(p["weight"] for p in pal) - 1.0) < 1e-6
    ws = [p["weight"] for p in pal]
    assert ws == sorted(ws, reverse=True)
    rgbs = [_hex_rgb(p["hex"]) for p in pal]
    # orange disc and white bar are recovered
    for target in ((0.95, 0.45, 0.1), (0.95, 0.95, 0.95)):
        d = min(np.linalg.norm(c - np.array(target)) for c in rgbs)
        assert d < 0.12, (target, [p["hex"] for p in pal])
    # dominant colour is the dark-blue background
    top = rgbs[0]
    assert top[2] > top[0] and top.mean() < 0.35
    # accent = the orange (most saturated with weight >= 0.05)
    acc = _hex_rgb(circle_res["accent"])
    assert np.linalg.norm(acc - np.array([0.95, 0.45, 0.1])) < 0.12


def test_saliency_on_disc(circle_res):
    fx, fy = circle_res["focus"]
    # disc centre is (700/1024, 380/768) = (0.684, 0.495)
    assert abs(fx - 0.684) < 0.15 and abs(fy - 0.495) < 0.15
    x, y, w, h = circle_res["saliency_box"]
    assert x <= 0.684 <= x + w and y <= 0.495 <= y + h


def test_contour_around_circle(circle_res):
    cs = circle_res["contours"]
    assert 1 <= len(cs) <= 12
    true_bb = np.array([500 / W, 180 / H, 400 / W, 400 / H])
    best = min(cs, key=lambda c: np.abs(np.array(c["bbox"]) - true_bb).sum())
    assert np.abs(np.array(best["bbox"]) - true_bb).max() < 0.03, best["bbox"]
    assert 8 <= len(best["points"]) <= 200
    pts = np.array(best["points"])
    # points lie on the circle (radius in normalised x/y units)
    r = np.hypot((pts[:, 0] - 700 / W) * W, (pts[:, 1] - 380 / H) * H)
    assert np.abs(r - 200).max() < 12
    areas = [c["area"] for c in cs]
    assert areas == sorted(areas, reverse=True)
    for i, c in enumerate(cs):
        assert c["id"] == f"img0:c{i}"
        assert isinstance(c["closed"], bool)
        assert all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in c["points"])
        assert 8 <= len(c["points"]) <= 200


def test_lines_on_bar(circle_res):
    ls = circle_res["lines"]
    assert 1 <= len(ls) <= 8
    for i, ln in enumerate(ls):
        assert ln["id"] == f"img0:l{i}"
        assert 0 <= ln["strength"] <= 1
    assert ls[0]["strength"] == pytest.approx(1.0)
    st = [ln["strength"] for ln in ls]
    assert st == sorted(st, reverse=True)
    # a horizontal line along the bar edge (y 620..650 px => 0.807..0.846), spanning ~x 0.08..0.59
    ok = False
    for ln in ls:
        (x0, y0), (x1, y1) = ln["p0"], ln["p1"]
        if abs(y0 - y1) < 0.01 and 0.79 < (y0 + y1) / 2 < 0.86 and abs(x1 - x0) > 0.4:
            ok = True
    assert ok, ls


def test_empty_regions_avoid_texture():
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.float32) + np.array([0.2, 0.5, 0.8], np.float32)
    img[:, : W // 2] = rng.random((H, W // 2, 3)).astype(np.float32)
    r = A.analyze_image(img, "img1")
    er = r["empty_regions"]
    assert 1 <= len(er) <= 4
    areas = [w * h for _, _, w, h in er]
    assert areas == sorted(areas, reverse=True)
    assert areas[0] >= 0.15
    for x, y, w, h in er:
        assert x >= 0.49, er  # entirely in the flat right half


def test_uint8_and_rgba_input(circle_img):
    u = (circle_img * 255).astype(np.uint8)
    rgba = np.dstack([u, np.full(u.shape[:2], 255, np.uint8)])
    r = A.analyze_image(rgba, "imgX")
    assert r["pixels"].dtype == np.float32 and r["pixels"].shape == (H, W, 3)
    assert r["pixels"].max() <= 1.0
    assert r["contours"] and r["contours"][0]["id"] == "imgX:c0"


def test_image_speed(circle_img):
    A.analyze_image(circle_img, "warm")
    t = time.perf_counter()
    A.analyze_image(circle_img, "img0")
    assert time.perf_counter() - t < 1.5


# ---------------------------------------------------------------- audio

SR = 48000


def _click_track(bpm=120.0, dur=10.0, start=0.25, sr=SR):
    x = np.zeros(int(sr * dur), np.float32)
    tt = np.arange(int(0.03 * sr)) / sr
    click = np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt / 0.004)
    kick = np.sin(2 * np.pi * 55 * tt) * np.exp(-tt / 0.03)
    truth = np.arange(start, dur - 0.05, 60.0 / bpm)
    for i, b in enumerate(truth):
        s = int(round(b * sr))
        seg = 0.5 * click + (0.6 * kick if i % 4 == 0 else 0.0)
        x[s:s + len(seg)] += seg[: len(x) - s]
    return np.stack([x, x]), truth


def test_audio_click_track():
    wav, truth = _click_track()
    t = time.perf_counter()
    a = A.analyze_audio(wav, SR)
    el = time.perf_counter() - t
    assert el < 2.0
    assert abs(a["bpm"] - 120.0) <= 2.0
    beats = np.array(a["beats"])
    assert len(beats) >= len(truth) - 2
    err = np.array([np.min(np.abs(truth - b)) for b in beats])
    assert err.max() < 0.030, err.max()
    assert a["duration"] == pytest.approx(10.0, abs=1e-3)
    assert a["sample_rate"] == SR and a["waveform"].shape == wav.shape
    # downbeats are every 4th beat, on the accented (kick) clicks
    db = np.array(a["downbeats"])
    assert len(db) >= 4
    acc = truth[::4]
    assert np.array([np.min(np.abs(acc - d)) for d in db]).max() < 0.03
    # onsets on clicks
    on = np.array(a["onsets"])
    assert len(on) >= len(truth) - 2
    assert np.array([np.min(np.abs(truth - o)) for o in on]).max() < 0.03
    secs = a["sections"]
    assert 1 <= len(secs) <= 6 and secs[0]["t"] == 0.0
    assert all(0 <= s["energy"] <= 1 for s in secs)


def test_audio_sections_on_change():
    rng = np.random.default_rng(1)
    wav, _ = _click_track(dur=16.0)
    x = wav[0].copy()
    x[: 8 * SR] *= 0.25
    x[8 * SR:] += 0.08 * rng.standard_normal(8 * SR).astype(np.float32)
    a = A.analyze_audio(x[None], SR)
    ts = [s["t"] for s in a["sections"]]
    assert len(ts) >= 2
    assert any(abs(t - 8.0) < 0.6 for t in ts[1:]), a["sections"]
    loud = [s for s in a["sections"] if s["t"] > 7.0][0]
    assert loud["energy"] > a["sections"][0]["energy"]


def test_audio_envelope():
    wav, truth = _click_track()
    fps, n = 30.0, 300
    env = A.audio_envelope(wav, SR, fps, n)
    assert set(env) == {"rms", "low", "mid", "high", "onset"}
    for k, v in env.items():
        assert v.dtype == np.float32 and v.shape == (n,), k
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # onset peaks near the click frames
    fr = (truth * fps).astype(int)
    off = ((truth + 0.25) * fps).astype(int)
    off = off[off < n]
    assert env["onset"][fr].mean() > 3 * env["onset"][off].mean()
    # low band (kick) is present on the accented beats, absent between beats
    assert env["low"][fr[::4]].mean() > 3 * env["low"][off].mean() + 1e-6
    assert env["rms"][fr].mean() > 3 * env["rms"][off].mean()
    # different lengths / silence
    short = A.audio_envelope(wav[:, : SR], SR, 24.0, 100)
    assert all(v.shape == (100,) for v in short.values())
    assert short["rms"][40:].max() == 0.0
    z = A.audio_envelope(np.zeros((2, SR)), SR, 30.0, 30)
    assert all(float(v.max()) == 0.0 for v in z.values())


# ---------------------------------------------------------------- inputs


def test_text_lines():
    r = A.analyze_inputs("  Hello world\n\n  second line  \n", [], None, None)
    assert r["text"] == "Hello world\n\n  second line"
    assert r["text_lines"] == ["Hello world", "second line"]
    r = A.analyze_inputs("Make it move, make it loud. The future of motion graphics starts right here today", [], None, 0)
    assert len(r["text_lines"]) >= 3
    assert all(len(p.split()) <= 6 for p in r["text_lines"])
    assert " ".join(r["text_lines"]).split() == "Make it move, make it loud. The future of motion graphics starts right here today".split()
    assert r["duration_hint"] is None
    assert A.analyze_inputs("short one", [], None, 8.0)["duration_hint"] == 8.0
    assert A.analyze_inputs("", [], None, None)["text_lines"] == []


def test_analyze_inputs_shape_and_mood(circle_img):
    wav, _ = _click_track(bpm=128.0, dur=6.0)
    img2 = (circle_img[::2, ::2] * 255).astype(np.uint8)
    r = A.analyze_inputs("Epic battle of legends\nRise to glory", [circle_img, img2], (wav, SR), None)
    assert [im["id"] for im in r["images"]] == ["img0", "img1"]
    assert all("pixels" in im and im["pixels"].dtype == np.float32 for im in r["images"])
    assert r["images"][1]["w"] == W // 2
    au = r["audio"]
    for k in ("waveform", "sample_rate", "duration", "bpm", "beats", "downbeats", "onsets", "sections"):
        assert k in au
    m = r["mood"]
    assert 0 <= m["energy"] <= 1 and 0 <= m["valence"] <= 1
    assert "epic" in m["keywords"]
    assert "battle" in m["keywords"] and "glory" in m["keywords"]

    calm = A.analyze_inputs("A calm and peaceful morning by the quiet ocean", [], None, None)["mood"]
    assert "calm" in calm["keywords"]
    hype = A.analyze_inputs("Fast. Loud. Unstoppable power. Race to victory", [], None, None)["mood"]
    assert hype["energy"] > calm["energy"] + 0.3
    dark = A.analyze_inputs("Midnight shadows and fear in the dark", [], None, None)["mood"]
    bright = A.analyze_inputs("Bright sunny summer smile", [], None, None)["mood"]
    assert bright["valence"] > dark["valence"] + 0.3
    assert "dark" in dark["keywords"] and "bright" in bright["keywords"]
    neutral = A.analyze_inputs("", [], None, None)
    assert neutral["images"] == [] and neutral["audio"] is None
    assert len(neutral["mood"]["keywords"]) == 1


def test_lexicon_size():
    assert sum(len(v) for v in A._LEXICON.values()) >= 60
    assert set(A._LEXICON) == {"driving", "calm", "dark", "bright", "playful", "epic"}
