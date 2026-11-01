"""The approved music route in the app graph (BUILT.md "Music"): ACE-Step XL base text-to-music -> exact cut + measured grid -> tame highs
-> mono-safe master -> plan. CPU only; the ACE model itself is replaced by a synthetic song."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fusion import t2m_music as T  # noqa: E402

SR = 48000
MUSIC_KEYS = {"cut", "nframes", "bpm", "bar_len", "key", "beats", "downbeats", "chord_changes", "curves", "fps", "music_end", "fade"}


def synth_song(bpm: float, seconds: float, sr: int = SR, tone_hz: float | None = None, seed: int = 0) -> np.ndarray:
    """(N, 2): a 600 Hz burst on every beat, a 70 Hz kick on every 4th beat (the downbeat), a quiet 220 Hz pad; optional sustained tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = 0.03 * np.sin(2 * np.pi * 220 * t) + 0.002 * rng.standard_normal(n)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        i = int(k * beat * sr)
        m = min(int(0.12 * sr), n - i)
        tt = np.arange(m) / sr
        x[i:i + m] += 0.35 * np.sin(2 * np.pi * 600 * tt) * np.exp(-tt * 40)
        if k % 4 == 0:
            x[i:i + m] += 0.5 * np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 18)
    if tone_hz:
        x += 0.05 * np.sin(2 * np.pi * tone_hz * t)
    return np.stack([x, 0.9 * x], axis=1)


# ----------------------------------------------------------------------------------------------- pure functions
def test_fit_bpm_lands_on_whole_bars():
    assert T.fit_bpm(90, 16) == 90 and T.fit_bpm(75, 16) == 75 and T.fit_bpm(120, 12) == 120
    assert T.fit_bpm(90, 12) in (80, 100)                       # 4.5 bars: the nearest whole count
    for length in (8, 12, 16, 24):
        for want in (50, 72, 100, 124, 180, 220):
            b = T.fit_bpm(want, length)
            assert 50 <= b <= 180 and abs(b * length / 240.0 - round(b * length / 240.0)) < 0.02 * b * length / 240.0 + 1e-9, (length, want, b)


def test_caption_is_warm_and_keeps_the_grid_tempo():
    c = T.caption("bright melodic house, bouncy plucked synth hook, glass bells, round bass, 128 BPM, instrumental, no vocals", 100, "F minor")
    assert "100 BPM" in c and "128" not in c and "F minor" in c and "4/4" in c
    assert "no high-pitched lead" in c and "no vocals" in c
    for bad in ("bells", "bright", "bouncy", "plucked"):
        assert bad not in c.lower(), bad
    keep = T.caption("tiny music box bells, 90 bpm", 90, "C major", soften=False)         # the user's own prompt is used as written
    assert "music box bells" in keep and "90 BPM" in keep
    assert T.preset_for("calm") == "warm" and T.preset_for("dark") == "warm" and T.preset_for("techno") == "club"


def test_cut_and_fade_is_exact_and_ends_silent():
    x = synth_song(100, 16)
    y = T.cut_and_fade(x, SR, 12, 0.6)
    assert y.shape == (12 * SR, 2) and abs(y[-1]).max() < 1e-6 and abs(y[0]).max() < 1e-3
    assert np.abs(y[: 10 * SR]).max() == pytest.approx(np.abs(x[: 10 * SR] * np.r_[np.linspace(0, 1, 480), np.ones(10 * SR - 480)][:, None]).max(), rel=0.01)
    short = T.cut_and_fade(x[: 5 * SR], SR, 12, 0.6)                                       # ACE ended early: padded with silence
    assert short.shape == (12 * SR, 2) and np.abs(short[7 * SR:]).max() == 0.0


def test_track_beats_measures_the_song_not_the_ask():
    x = synth_song(100, 16)
    fit = T.track_beats(x.mean(axis=1), SR, 96.0)                                          # asked 96, the song is 100
    assert fit["bpm"] == pytest.approx(100.0, rel=0.02)
    assert fit["t0"] < 0.15 or fit["t0"] > fit["beat_s"] - 0.15
    g = T.t2m_grid(12.0, fit, T.cut_and_fade(x, SR, 12, fit["beat_s"]).T, SR, fit["beat_s"], key="A minor", mood="calm", bpm_hint=96)
    assert (MUSIC_KEYS - {"curves"}) <= set(g) and g["src"] == "t2m" and g["cut"] == 12.0 and g["nframes"] == 288 and g["n_samples"] == 12 * SR
    assert g["n_bars"] in (4, 5) and g["grid"]["length_s"] == 12.0 and g["beats"][-1] >= 12.0 + 3.0     # beats run 4 s past the end
    assert g["bar_len"] == pytest.approx(4 * fit["beat_s"])


def test_bar_count_tolerates_a_tempo_measured_a_hair_slow():
    """100 bpm measured as 99.98 (bar 2.4005 s): 5 bars in 12 s still count as 5, and the last bar ends at the piece length."""
    fit = dict(bpm=99.98, beat_s=60 / 99.98, t0=0.02, downbeat_phase=0)
    g = T.t2m_grid(12.0, fit, np.full((2, 12 * SR), 1e-3), SR, 0.6, key="F minor", mood="dark")
    assert g["n_bars"] == 5 and g["grid"]["bars"][-1]["end"] == 12.0
    fit2 = dict(bpm=100.0, beat_s=0.6, t0=0.3, downbeat_phase=0)                             # first bar starts late: only 4 whole bars fit
    assert T.t2m_grid(12.0, fit2, np.full((2, 12 * SR), 1e-3), SR, 0.6, key="F minor", mood="dark")["n_bars"] == 4


def test_tame_highs_removes_a_sustained_whistle_and_keeps_the_length():
    from scipy import signal
    x = synth_song(100, 12, tone_hz=3141.0)
    y, peaks = T.tame_highs(x, SR)
    assert any(abs(p["hz"] - 3141) < 40 for p in peaks), peaks

    def level(z):
        f, p = signal.welch(z[:, 0], SR, nperseg=32768)
        return 10 * np.log10(p[(f > 3100) & (f < 3180)].max())
    assert level(y) < level(x) - 6.0
    assert y.shape == x.shape
    clean = synth_song(100, 12)                                                            # no tone: nothing notched
    assert T.tame_highs(clean, SR)[1] == []


def test_safe_master_is_mono_below_200hz_and_exact():
    x = synth_song(100, 12)
    wide = x.copy()
    wide[:, 1] = -0.6 * x[:, 0]                                                            # out-of-phase right channel: bad mono fold-down
    m, rep = T.safe_master(wide, SR, "warm", 100.0, 12.0)
    assert m.shape == (2, 12 * SR) and rep["mono_after"]["folddown_db"] >= -1.6
    f = np.fft.rfftfreq(m.shape[1], 1.0 / SR)
    lo = f < 150
    L, R = np.fft.rfft(m[0])[lo], np.fft.rfft(m[1])[lo]
    assert np.abs(L - R).sum() < 0.05 * (np.abs(L).sum() + np.abs(R).sum())


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


class FakeClip:
    def __init__(self):
        self.args = None

    def tokenize(self, text, **kw):
        self.args = dict(kw, text=text)
        return "tokens"

    def encode_from_tokens_scheduled(self, tokens):
        return [["cond", {}]]


REQ = {"words": "AFTER HOURS\nTHE CITY KEEPS DREAMING", "scene": "", "music": "", "length": 12, "look": "auto", "seed": 7, "quality": "draft"}


def test_node_schemas_and_registration():
    _import_comfy()
    from nodes_fusion import music_t2m as nodes
    import nodes_fusion
    assert [c.define_schema().node_id for c in nodes.NODES] == ["MographMusicT2MEncode", "MographMusicT2MCut", "MographTameHighs"]
    for cls in nodes.NODES:
        cls.GET_SCHEMA()
    assert "music_t2m" in nodes_fusion.SUBMODULES
    assert {c.__name__ for c in nodes.NODES} <= {c.__name__ for c in nodes_fusion.NODES}, nodes_fusion.FAILED
    assert "MographScoreSkeleton" in {c.__name__ for c in nodes_fusion.NODES}               # old route stays registered, unused


def test_encode_asks_ace_for_no_codes_and_length_plus_four():
    _import_comfy()
    from nodes_fusion import music_t2m as nodes
    clip = FakeClip()
    cond, seconds, plan_json = nodes.MographMusicT2MEncode.execute(clip, REQ, "").args
    a = clip.args
    assert a["generate_audio_codes"] is False and a["duration"] == 16.0 and seconds == 16.0
    assert a["lyrics"] == "[Instrumental]" and a["timesignature"] == 4 and a["bpm"] * 12 % 240 == 0           # whole bars in 12 s
    plan = json.loads(plan_json)
    assert plan["bpm"] == a["bpm"] and plan["length"] == 12.0 and plan["preset"] in ("warm", "club") and "no high-pitched lead" in a["text"]
    user = dict(REQ, music="tiny bells and glass, 100 bpm, A minor")
    clip2 = FakeClip()
    nodes.MographMusicT2MEncode.execute(clip2, user, "")
    assert "bells" in clip2.args["text"] and clip2.args["keyscale"] == "A minor"           # the user's words are not softened


def test_cut_tame_master_and_plan_from_the_measured_grid():
    _import_comfy()
    import torch
    from nodes_fusion import audio as audio_nodes
    from nodes_fusion import music_t2m as nodes
    from nodes_fusion import plan as plan_nodes
    clip = FakeClip()
    _, _, plan_json = nodes.MographMusicT2MEncode.execute(clip, REQ, "").args
    asked = json.loads(plan_json)["bpm"]
    song = synth_song(100, 16, tone_hz=3141.0)                                             # "ACE" made 16 s at 100 bpm with a whistle
    raw = {"waveform": torch.from_numpy(song.T.astype(np.float32)).unsqueeze(0), "sample_rate": SR}
    audio, music = nodes.MographMusicT2MCut.execute(raw, REQ, plan_json).args
    assert audio["waveform"].shape == (1, 2, 12 * SR) and audio["sample_rate"] == SR       # EXACT piece length
    assert MUSIC_KEYS <= set(music) and music["src"] == "t2m" and music["cut"] == 12.0 and music["n_samples"] == 12 * SR
    assert music["bpm"] == pytest.approx(100.0, rel=0.03)                                  # the song's tempo, measured (not the asked)
    assert music["grid"]["bpm_hint"] == asked and len(music["curves"]["onset"]) == music["nframes"] + 1
    assert music["master_preset"] in ("warm", "club") and "t2m_end_gate" in music
    (tamed,) = nodes.MographTameHighs.execute(audio).args
    assert tamed["waveform"].shape == audio["waveform"].shape
    (mastered,) = audio_nodes.MographMaster.execute(tamed, music, "auto", -14.0, 1.25, True).args
    assert mastered["waveform"].shape[-1] == 12 * SR                                       # mono-safe master keeps the exact length
    (plain,) = audio_nodes.MographMaster.execute(tamed, music, "warm", -14.0, 1.25).args   # old signature still works
    assert plain["waveform"].shape[-1] == 12 * SR
    piece, piece_json, *_ = plan_nodes.MographPiecePlan.execute(REQ, music, "").args       # the plan is built from the measured grid
    assert piece["layout"] and piece["audio"]["src"] == "t2m" and piece["audio"]["bpm"] == int(round(music["bpm"]))
