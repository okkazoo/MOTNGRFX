"""User track -> the finished piece is EXACTLY the chosen Length (CPU, no ComfyUI).
Bug (app image-mode renders, 2026-11-01): MographMusicCut snapped a user track to the downbeat NEAREST the Length, so Length 8 with a
106 bpm track gave a 9.07 s cut (-> 9.083 s video) and Length 12 at 75 bpm gave 12.85 s. The cut is now the Length (pad=True).
Run: python -m pytest tests/test_music_cut_length.py -q"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import audio_cut as ac  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402

SR = 48000
FPS = 24


def _track(bpm: float, seconds: float, first: float = 0.3):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wave = np.stack([x, x])
    iv = 60.0 / bpm
    beats = [first + i * iv for i in range(int(seconds / iv))]
    nf = int(seconds * FPS)
    ana = dict(bpm=bpm, bar_len=4 * iv, downbeats=beats[::4], beats=beats, key={"estimate": "C major"},
               curves={"onset": [0.2] * (nf + 1)}, chord_changes=[])
    return wave, ana


@pytest.mark.parametrize("bpm,length,old_cut", [(106, 8, 9.0708), (75, 12, 12.8534)])
def test_user_track_is_cut_to_the_length_not_the_nearest_downbeat(bpm, length, old_cut):
    wave, ana = _track(bpm, 30.0, first=0.0)
    _, m_old = ac.cut_to_downbeat(wave, SR, ana, target_len=length)          # the old route: overshoots by 0.8-1.1 s
    assert m_old["cut"] > length + 0.5
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=float(length), fps=FPS, fade=0.5, pad=True)
    assert m["cut"] == pytest.approx(length) and m["nframes"] == length * FPS
    assert y.shape == (2, length * SR)
    assert abs(float(y[0, -1])) < 1e-3                                       # faded out at the very end
    assert float(np.abs(y[:, -int(0.1 * SR):]).max()) < 0.3 * 0.25           # inside the fade
    assert float(np.abs(y[:, int(1.0 * SR):int(2.0 * SR)]).max()) > 0.25     # music before it untouched


def test_short_track_is_looped_to_the_length_not_padded_with_silence():
    wave, ana = _track(100, 6.0)
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=8.0, fps=FPS, pad=True)
    assert y.shape == (2, 8 * SR) and m["cut"] == pytest.approx(8.0) and m["nframes"] == 192
    assert _longest_quiet_s(y) < 0.2                                         # music to the end (only the final fade is quiet)
    assert abs(float(y[0, -1])) < 1e-3                                       # the fade sits at the end of the cut
    assert len(m["curves"]["onset"]) == 193                                  # curves cover every frame
    wave, ana = _track(100, 6.0)
    y2, m2 = ac.cut_to_downbeat(wave, SR, ana, cut=8.0, fps=FPS)            # default (no pad) still clamps to the file
    assert y2.shape[1] == 6 * SR


def _longest_quiet_s(y, floor=1e-3, win_s=0.05):
    """longest run (s) of 50 ms windows whose RMS is below ``floor``."""
    w = int(win_s * SR)
    k = y.shape[1] // w
    rms = np.sqrt((y[:, :k * w].reshape(2, k, w) ** 2).mean(axis=(0, 2)))
    best = run = 0
    for q in rms < floor:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * win_s


def _click_track(bpm: float, seconds: float, first: float = 0.0):
    """tone bed + a loud 30 ms 880 Hz click on every beat (a louder one on each downbeat) - the beat is findable in the audio."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    iv = 60.0 / bpm
    beats = [first + i * iv for i in range(int((seconds - first) / iv))]
    k = int(0.03 * SR)
    click = (np.sin(2 * np.pi * 880 * np.arange(k) / SR) * np.hanning(k)).astype(np.float32)
    for i, b in enumerate(beats):
        a = int(round(b * SR))
        x[a:a + k] += click[:max(0, min(k, n - a))] * (0.8 if i % 4 == 0 else 0.4)
    wave = np.stack([x, x])
    nf = int(seconds * FPS)
    ana = dict(bpm=bpm, bar_len=4 * iv, downbeats=beats[::4], beats=beats, key={"estimate": "C major"},
               curves={"onset": [float(i % 7) for i in range(nf + 1)]}, chord_changes=[4.0, 8.0])
    return wave, ana


@pytest.mark.parametrize("first", [0.0, 0.77])
def test_16s_track_at_24s_loops_with_no_silence_and_the_seam_on_a_beat(first, capsys):
    """the box bug: a 16 s track at Length 24 was silent from its last downbeat to the end."""
    wave, ana = _click_track(120, 16.0, first=first)
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=24.0, fps=FPS, fade=0.5, pad=True)
    assert y.shape == (2, 24 * SR)                                            # sample exact
    assert m["cut"] == pytest.approx(24.0) and m["nframes"] == 24 * FPS
    assert _longest_quiet_s(y) <= 0.5                                         # no silent stretch (only the last 50 ms+ of the fade)
    assert float(np.abs(y[:, int(14.0 * SR):int(22.0 * SR)]).max()) > 0.3     # live right through the old silent stretch
    assert abs(float(y[0, -1])) < 1e-3                                        # end fade kept
    lp = m["loop"]
    assert lp["grid"] and lp["passes"] >= 2 and lp["seams"]
    iv = 0.5
    for p in lp["seams"]:                                                     # each repeat's downbeat lands on a beat of the grid
        assert abs(((p - first) / iv) - round((p - first) / iv)) < 0.01, p
        assert abs(((p - lp["start"]) / (4 * iv)) - round((p - lp["start"]) / (4 * iv))) < 0.01   # whole bars apart from the loop start
    p0 = lp["seams"][0]                                                       # and the audio has the click there
    a = int(round(p0 * SR))
    assert float(np.abs(y[:, a:a + int(0.03 * SR)]).max()) > 0.5
    assert min(m["downbeats"]) >= 0 and p0 in [round(d, 4) for d in m["downbeats"]]
    assert len(m["curves"]["onset"]) == 24 * FPS + 1
    assert "looped" in capsys.readouterr().out


def test_looped_track_without_a_beat_grid_loops_the_whole_track():
    wave, ana = _track(100, 6.0)
    ana = dict(ana, downbeats=[], beats=[])
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=20.0, fps=FPS, fade=0.5, pad=True)
    assert y.shape == (2, 20 * SR) and not m["loop"]["grid"] and m["loop"]["crossfade"] == pytest.approx(0.3)
    assert _longest_quiet_s(y) <= 0.5
    assert float(np.abs(y[:, int(10 * SR):int(19 * SR)]).max()) > 0.25


def test_track_with_a_silent_tail_loops_the_music_not_the_silence():
    wave, ana = _click_track(120, 16.0)
    wave[:, int(13.6 * SR):] = 0.0                                           # music ends at 13.6 s, file is 16 s
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=24.0, fps=FPS, pad=True)
    assert y.shape == (2, 24 * SR) and _longest_quiet_s(y) <= 0.5
    assert m["loop"]["end"] <= 13.6 + 1e-6


@pytest.mark.parametrize("bpm,length", [(106, 8), (75, 12)])
def test_long_track_is_cut_as_before_without_looping(bpm, length):
    wave, ana = _track(bpm, 30.0, first=0.0)
    y, m = ac.cut_to_downbeat(wave, SR, ana, cut=float(length), fps=FPS, fade=0.5, pad=True)
    assert "loop" not in m and y.shape == (2, length * SR)
    n = length * SR
    f = int(0.5 * SR)
    assert np.array_equal(y[:, :n - f], wave[:, :n - f])                     # untouched up to the fade
    assert np.allclose(y[:, n - f:], wave[:, n - f:n] * np.linspace(1, 0, f, dtype=np.float32)[None])


@pytest.mark.parametrize("bpm,length", [(106, 8), (75, 12), (120, 16), (90, 24)])
def test_plan_spans_sum_to_the_length(bpm, length):
    """the planner's shot spans (what the video lasts) add up to the Length, never the LTX 8k+1 clip sizes."""
    wave, ana = _track(bpm, length + 8.0)
    _, music = ac.cut_to_downbeat(wave, SR, ana, cut=float(length), fps=FPS, pad=True)
    assert music["nframes"] == length * FPS
    req = {"words": "FLOODED RHYTHM / THE WATER RISES", "scene": "a drummer in a flooded cathedral", "music": "", "length": length,
           "look": "auto", "seed": 5, "quality": "final", "format": "16:9"}
    piece = pr.plan_piece(req, music)
    assert piece["audio"]["cut"] == pytest.approx(length)
    spans = [s.get("span") for s in piece["shots"]]
    if len(piece["shots"]) > 1:
        assert all(spans[:-1]) is True
    got = sum(int(s["span"]) for s in piece["shots"] if s.get("span")) + (piece["shots"][-1]["frames"] if not piece["shots"][-1].get("span") else 0)
    assert abs(got - length * FPS) <= 1, (spans, [s["frames"] for s in piece["shots"]])
    assert piece["type"]["duration"] == pytest.approx(length, abs=0.01)
