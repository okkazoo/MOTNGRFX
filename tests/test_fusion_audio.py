"""Fusion audio: cut on a downbeat below the end of the music, mastering (-14 LUFS, <= -1.0 dBTP, width), the audio nodes (CPU).
Run: python -m pytest tests/test_fusion_audio.py -q"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mograph import mastering as M  # noqa: E402
from mograph.mix import loudness_lufs  # noqa: E402
from tools.fusion import audio_cut as ac  # noqa: E402

FUSION = os.path.join(ROOT, "15_fusion")


def _need(path):
    if not os.path.isfile(path):
        pytest.skip(f"{os.path.relpath(path, ROOT)} not present")
    return path


def _piece_audio(pid):
    return os.path.join(ROOT, "15_fusion", "v002", "local", "pieces", pid, "audio.wav")


def _read(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    return x.T, sr


# ----------------------------------------------------------------------------------------------- widen
def test_widen_is_mid_side_gain():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4800))
    assert np.allclose(M.widen(x, 1.0), x)
    mono = M.widen(x, 0.0)
    assert np.allclose(mono[0], mono[1]) and np.allclose(mono[0], 0.5 * (x[0] + x[1]))
    y = M.widen(x, 1.25)
    assert np.allclose(y[0] + y[1], x[0] + x[1])                                      # mid untouched
    assert np.allclose(y[0] - y[1], 1.25 * (x[0] - x[1]))                              # side x 1.25
    assert M.widen(x[:1], 1.5).shape == (2, 4800)


# ----------------------------------------------------------------------------------------------- cut
def test_cut_p07_on_a_downbeat_below_music_end():
    """P07's ACE render is ~25 s long but the music stops near 15.5 s; the rest is digital silence."""
    spec = json.load(open(_need(os.path.join(FUSION, "pieces", "P07.json")), encoding="utf-8"))
    src = _need(os.path.join(ROOT, spec["audio"]["src"]))
    wave, sr, music = ac.cut_file(src, None, target_len=24, bpm_hint=spec["audio"].get("bpm"))
    assert music["music_end"] < 16.5                                       # the tail really is silent
    assert music["cut"] <= music["music_end"] + 1e-9
    assert min(abs(music["cut"] - d) for d in music["downbeats"]) <= 1.0 / 24
    assert wave.shape == (2, int(round(music["cut"] * sr))) and sr == 48000
    assert music["nframes"] == math.ceil(music["cut"] * 24 - 1e-6)
    seg = wave[:, int((music["cut"] - 1.5) * sr):int((music["cut"] - 0.6) * sr)]      # just before the fade
    assert float(np.sqrt((seg ** 2).mean())) > 0.005                                  # music, not the silent tail
    assert np.abs(wave[:, -int(0.02 * sr):]).max() < 0.05                              # faded out
    for k in ("bpm", "bar_len", "key", "beats", "downbeats", "chord_changes", "curves", "music_end", "cut", "nframes"):
        assert k in music
    json.dumps(music)                                                                   # plain data
    assert music["beats"][-1] >= music["cut"] + 4


def test_target_length_picks_the_nearest_downbeat_but_never_past_the_music():
    sr = 48000
    t = np.arange(int(20 * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    x[int(13.0 * sr):] = 0.0                                               # ACE-like silent tail from 13 s
    wave = np.stack([x, x]).astype(np.float32)
    bar = 4 * 60 / 120.0
    ana = dict(bpm=120.0, bar_len=bar, downbeats=[0.0, bar], beats=[i * 0.5 for i in range(30)], key={"estimate": "C major"},
               curves={"onset": [0.1] * 600}, chord_changes=[])
    _, m12 = ac.cut_to_downbeat(wave, sr, ana, target_len=12)
    assert m12["cut"] == pytest.approx(12.0) and m12["music_end"] == pytest.approx(13.0, abs=0.06)
    _, m24 = ac.cut_to_downbeat(wave, sr, ana, target_len=24)
    assert m24["cut"] == pytest.approx(12.0)                               # 12.0 is the last downbeat below 13.0
    _, m5 = ac.cut_to_downbeat(wave, sr, ana, target_len=5.2)
    assert m5["cut"] == pytest.approx(6.0)                                 # nearest bar line (4.0 / 6.0 -> 6.0 is nearer)
    y, m = ac.cut_to_downbeat(wave, sr, ana, target_len=12, fade=0.5, ring=0.0)
    assert y.shape[1] == int(round(12.0 * sr)) and abs(float(y[0, -1])) < 1e-3


def test_cut_resamples_and_duplicates_mono():
    sr = 44100
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(6 * sr) / sr)).astype(np.float32)[None]
    ana = dict(bpm=120.0, bar_len=2.0, downbeats=[0.0], beats=[i * 0.5 for i in range(14)], key="A major", curves={}, chord_changes=[])
    y, m = ac.cut_to_downbeat(x, sr, ana, target_len=4)
    assert y.shape[0] == 2 and np.allclose(y[0], y[1]) and abs(y.shape[1] / 48000 - m["cut"]) < 1e-3
    assert m["key"] == "A major" and m["cut"] == pytest.approx(4.0)


# ----------------------------------------------------------------------------------------------- master
def _ebur128(path):
    if not shutil.which("ffmpeg"):
        return None
    r = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr.split("Summary:")[-1]
    import re
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", txt)
    return float(i.group(1)) if i else None


@pytest.mark.parametrize("pid", ["P02", "P05"])
def test_master_hits_loudness_and_true_peak(pid, tmp_path):
    wave, sr = _read(_need(_piece_audio(pid)))
    y, rep = ac.master_audio(wave, sr, preset="club", target_lufs=-14.0, width=1.25, bpm=120)
    assert y.shape == (2, wave.shape[1]) and y.dtype == np.float32 and np.isfinite(y).all()
    after = ac.measure(y, sr)
    assert abs(after["lufs"] + 14.0) <= 0.5, after
    assert after["true_peak_dbtp"] <= -1.0, after
    assert rep["true_peak_dbtp"] <= -1.0 and abs(rep["lufs"] + 14.0) <= 0.5
    wav = tmp_path / "m.wav"
    import soundfile as sf
    sf.write(str(wav), y.T, sr, subtype="FLOAT")
    ext = _ebur128(str(wav))                                                          # independent loudness meter
    if ext is not None:
        assert abs(ext + 14.0) <= 0.5, ext
    before = ac.measure(wave, sr)
    assert after["correlation"] < before["correlation"]                               # wider
    assert after["high_pct"] >= before["high_pct"]                                    # more top end


@pytest.mark.parametrize("preset", sorted(ac.MASTER_PRESETS))
def test_every_preset_and_target(preset):
    wave, sr = _read(_need(_piece_audio("P05")))
    wave, sr = wave[:, : 6 * sr], sr
    for target in (-14.0, -11.0):
        y, rep = ac.master_audio(wave, sr, preset=preset, target_lufs=target, width=1.25)
        m = ac.measure(y, sr)
        assert abs(m["lufs"] - target) <= 0.5 and m["true_peak_dbtp"] <= -1.0, (preset, target, m)


def test_warm_preset(monkeypatch):
    """warm: more body (low shelf), top rolled off (vs club), -14 LUFS / -1 dBTP, mono below 150 Hz (the others keep 120), picked for calm / epic."""
    assert "warm" in ac.MASTER_PRESETS and ac.preset_for_mood("calm") == "warm" and ac.preset_for_mood("epic") == "warm"
    assert ac.preset_for_mood("driving") == "club" and ac.preset_for_mood(None) == "club" and ac.preset_for_mood("dark") == "club"
    assert ("lowshelf", 120, 0.7, 2.5) in ac.MASTER_PRESETS["warm"]["eq"] and ("highshelf", 9000, 0.7, -1.0) in ac.MASTER_PRESETS["warm"]["eq"]
    sr = 48000
    rng = np.random.default_rng(5)
    n = 8 * sr
    from scipy import signal
    common = signal.lfilter([1.0], [1.0, -0.98], 0.05 * rng.standard_normal(n))             # low-heavy mono noise
    wave = (np.stack([common, common]) + 0.05 * rng.standard_normal((2, n))).astype(np.float32)
    from mograph import mastering as MM
    seen = []
    real = MM.mono_below
    monkeypatch.setattr(MM, "mono_below", lambda x, s, hz: (seen.append(hz), real(x, s, hz))[1])
    yw, _ = ac.master_audio(wave, sr, preset="warm", width=1.25)
    yc, _ = ac.master_audio(wave, sr, preset="club", width=1.25)
    ye, _ = ac.master_audio(wave, sr, preset="warm", width=1.25, mono_hz=90.0)             # an explicit argument still wins
    assert seen == [150.0, 120.0, 90.0]
    mw, mc = ac.measure(yw, sr), ac.measure(yc, sr)
    assert abs(mw["lufs"] + 14.0) <= 0.5 and mw["true_peak_dbtp"] <= -1.0, mw
    assert mw["high_pct"] <= mc["high_pct"] and mw["low_pct"] >= mc["low_pct"] - 0.5          # club lifts the top +3 dB; warm trims it


def test_bass_is_mono_and_width_is_applied():
    sr = 48000
    rng = np.random.default_rng(3)
    n = 6 * sr
    t = np.arange(n) / sr
    bass = np.stack([np.sin(2 * np.pi * 40 * t + p) for p in (0.0, 1.0)])               # out of phase bass: wide (side/mid 0.55)
    hiss = 0.1 * rng.standard_normal((2, n))
    wave = (0.3 * bass + hiss).astype(np.float32)
    y, _ = ac.master_audio(wave, sr, preset="neutral", width=1.25)
    from scipy import signal
    b, a = signal.butter(4, 100 / (sr / 2))
    low = signal.filtfilt(b, a, y.astype(np.float64), axis=-1)
    side_low = np.sqrt(((low[0] - low[1]) ** 2).mean())
    mid_low = np.sqrt(((low[0] + low[1]) ** 2).mean())
    assert side_low / mid_low < 0.35                    # mono_below(120) is a transparent split: most of the side is gone (0.55 x 1.25 in)
    hi = signal.filtfilt(*signal.butter(4, 2000 / (sr / 2), "high"), y.astype(np.float64), axis=-1)
    assert np.sqrt(((hi[0] - hi[1]) ** 2).mean()) > 0                                    # highs keep their side


def test_master_input_checks():
    with pytest.raises(ValueError):
        ac.master_audio(np.zeros((2, 4800), np.float32), 48000, preset="nope")
    y, rep = ac.master_audio(np.zeros((1, 48000), np.float32), 48000, preset="film")      # silence must not crash or blow up
    assert np.isfinite(y).all() and y.shape[0] == 2


# ----------------------------------------------------------------------------------------------- nodes
def _import_comfy():
    cands = [os.environ.get("MOGRAPH_COMFYUI_PATH"), os.path.abspath(os.path.join(ROOT, "..", "ComfyUI")),
             os.path.abspath(os.path.join(ROOT, "..", "..", "..", "ComfyUI")), os.path.abspath(os.path.join(ROOT, "..", ".."))]
    for c in filter(None, cands):
        if os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            try:
                import comfy_api.latest  # noqa: F401
                return c
            except Exception as e:      # noqa: BLE001
                pytest.skip(f"comfy_api import failed: {e}")
    pytest.skip("no ComfyUI checkout found (set MOGRAPH_COMFYUI_PATH)")


def test_audio_nodes():
    _import_comfy()
    import torch
    from nodes_fusion import audio as nodes
    for cls in nodes.NODES:
        cls.define_schema()
        cls.GET_SCHEMA()
    assert [c.define_schema().node_id for c in nodes.NODES] == ["MographMusicEncode", "MographMusicCut", "MographMaster"]

    wave, sr = _read(_need(_piece_audio("P02")))
    user = {"waveform": torch.from_numpy(wave)[None], "sample_rate": sr}
    req = {"words": "AFTER HOURS", "length": 12, "seed": 1, "look": "auto", "quality": "final", "scene": "", "music": "deep house, 122 bpm"}
    # lazy gating: user audio present -> the ACE input is never requested; absent and connected -> requested
    chk = nodes.MographMusicCut.check_lazy_status
    assert chk(req, 100, None, user) == []
    assert chk(req, 100, None, None) == ["ace_audio"]
    assert chk(req, 100, {"waveform": user["waveform"], "sample_rate": sr}, None) == []
    music, cut = nodes.MographMusicCut.execute(req, 122, None, user).args
    assert cut["sample_rate"] == 48000 and cut["waveform"].shape[:2] == (1, 2)
    assert abs(cut["waveform"].shape[2] / 48000 - music["cut"]) < 1e-3
    assert music["cut"] == pytest.approx(12.0) and cut["waveform"].shape[2] == 12 * 48000      # EXACTLY the chosen Length (was: nearest downbeat)
    assert music["src"] == "user" and music["bpm"] == pytest.approx(122, abs=1.5)       # tempo hint from the music prompt
    with pytest.raises(ValueError):
        nodes.MographMusicCut.execute(req, 122, None, None)
    (mastered,) = nodes.MographMaster.execute(cut, music, "club", -14.0, 1.25).args
    y = mastered["waveform"][0].numpy()
    m = ac.measure(y, mastered["sample_rate"])
    assert abs(m["lufs"] + 14.0) <= 0.5 and m["true_peak_dbtp"] <= -1.0, m


def test_music_encode_calls_ace_tokenizer_like_the_core_node():
    _import_comfy()
    from nodes_fusion import audio as nodes

    class FakeClip:
        def tokenize(self, tags, **kw):
            self.call = (tags, kw)
            return "tokens"

        def encode_from_tokens_scheduled(self, tokens):
            assert tokens == "tokens"
            return [["cond", {}]]

    clip = FakeClip()
    req = {"words": "AFTER HOURS", "length": 12, "seed": 7, "look": "auto", "quality": "final", "scene": "", "music": "deep house, 122 bpm, F minor"}
    cond, seconds, bpm = nodes.MographMusicEncode.execute(clip, req, "").args
    tags, kw = clip.call
    assert "deep house" in tags and "no vocals" in tags
    assert kw["bpm"] == 122 == bpm and kw["keyscale"] == "F minor" and kw["lyrics"] == "[Instrumental]"
    assert kw["duration"] == pytest.approx(seconds) == pytest.approx(22.4) and kw["timesignature"] == 4
    assert kw["seed"] == 0 or kw["seed"] > 0
    assert cond == [["cond", {}]]
