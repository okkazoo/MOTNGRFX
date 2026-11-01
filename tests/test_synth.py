"""Unit tests for mograph.synth + mograph.mix (W2). Plans are minimal hand-built dicts."""

import os
import sys
import time

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mograph import mix as M  # noqa: E402
from mograph import synth as S  # noqa: E402

SR = 48000


def make_plan(seed=1, duration=10.0, bpm=120.0, mood="driving", energy=0.7, sections=None,
              risers=None, sfx=None, beats=None, downbeats=None, key="A", scale="minor"):
    if beats is None:
        beats = [round(float(t), 6) for t in np.arange(0.0, duration - 1e-9, 60.0 / bpm)]
    if downbeats is None:
        downbeats = beats[::4]
    return {
        "version": 1, "seed": seed, "duration": duration, "fps": 30,
        "style": {"palette": ["#111111", "#f2f2f2", "#ff4d2e"], "energy": energy, "motion": "snappy"},
        "audio": {
            "source": "synth", "bpm": bpm, "key": key, "scale": scale, "mood": mood,
            "beats": beats, "downbeats": downbeats,
            "sections": sections if sections is not None else [{"t": 0.0, "label": "intro"}, {"t": 4.0, "label": "drop"}],
            "risers": risers if risers is not None else [{"t0": 3.0, "t1": 4.0}],
            "sfx": sfx if sfx is not None else [{"t": 4.0, "type": "impact", "gain": 0.8}],
        },
        "voice": None,
        "scenes": [],
    }


def _low(x, sr=SR, fc=150.0):
    b, a = signal.butter(4, fc / (sr / 2), "low")
    return signal.lfilter(b, a, np.asarray(x, np.float64).mean(axis=0))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _tone_level(x, f, t0, t1, sr=SR):
    seg = np.asarray(x, np.float64).mean(axis=0)[int(t0 * sr):int(t1 * sr)]
    t = np.arange(len(seg)) / sr
    c = 2 * np.mean(seg * np.cos(2 * np.pi * f * t))
    s = 2 * np.mean(seg * np.sin(2 * np.pi * f * t))
    return float(np.hypot(c, s))


@pytest.fixture(scope="module")
def plan():
    return make_plan()


@pytest.fixture(scope="module")
def music(plan):
    return S.render_soundtrack(plan, SR)


# ---------------------------------------------------------------- soundtrack


@pytest.mark.parametrize("dur", [10.0, 7.33, 5.0])
def test_length_exact(dur):
    p = make_plan(duration=dur, risers=[], sfx=[])
    m = S.render_soundtrack(p, SR)
    assert m.shape == (2, int(round(dur * SR))) and m.dtype == np.float32
    s = S.render_sfx(p, SR)
    assert s.shape == m.shape and s.dtype == np.float32


def test_finite_and_peak_after_mix(plan, music):
    assert np.isfinite(music).all()
    assert np.abs(music).max() < 1.0
    sfx = S.render_sfx(plan, SR)
    assert np.isfinite(sfx).all()
    out = M.mix(music, sfx, None, SR)
    assert out.shape == music.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 10 ** (-1.0 / 20.0) + 1e-6  # -1 dBFS ceiling
    assert abs(out[:, 0]).max() < 1e-3 and abs(out[:, -1]).max() < 1e-3  # fades
    assert abs(float(out.mean())) < 1e-3  # no DC


def test_deterministic(plan, music):
    again = S.render_soundtrack(plan, SR)
    assert np.array_equal(music, again)
    other = S.render_soundtrack(dict(plan, seed=2), SR)
    assert not np.allclose(music, other)
    s1, s2 = S.render_sfx(plan, SR), S.render_sfx(plan, SR)
    assert np.array_equal(s1, s2)
    assert not np.allclose(s1, S.render_sfx(dict(plan, seed=99), SR))


@pytest.mark.parametrize("mood", ["driving", "bright"])
@pytest.mark.parametrize("energy", [0.3, 0.8])
def test_kick_on_beats(mood, energy):
    """Four-on-the-floor moods: low-band energy right after each beat beats the off-beat."""
    p = make_plan(duration=10.0, mood=mood, energy=energy, sections=[{"t": 0.0, "label": "drop"}],
                  risers=[], sfx=[])
    m = S.render_soundtrack(p, SR)
    low = _low(m)
    beats = np.array(p["audio"]["beats"])
    beats = beats[beats < 7.5]  # skip the final-bar ending
    half = 0.25
    on = np.mean([_rms(low[int(b * SR):int((b + 0.06) * SR)]) for b in beats])
    off = np.mean([_rms(low[int((b + half) * SR):int((b + half + 0.06) * SR)]) for b in beats])
    assert on > 1.6 * off, (on, off)


@pytest.mark.parametrize("mood", list(S.MOODS))
def test_kick_on_every_downbeat(mood):
    p = make_plan(duration=10.0, mood=mood, energy=0.8, sections=[{"t": 0.0, "label": "drop"}],
                  risers=[], sfx=[])
    low = _low(S.render_soundtrack(p, SR))
    for d in p["audio"]["downbeats"][1:-1]:
        pre = _rms(low[int((d - 0.03) * SR):int(d * SR)])
        post = _rms(low[int(d * SR):int((d + 0.03) * SR)])
        assert post > 1.4 * pre, (d, pre, post)


def test_follows_user_beat_grid():
    """Irregular (human) beat grid: kicks land on the given downbeats."""
    rng = np.random.default_rng(5)
    ivs = 0.52 + 0.04 * rng.random(24)
    beats = np.concatenate([[0.1], 0.1 + np.cumsum(ivs)])
    beats = [float(b) for b in beats if b < 11.5]
    dbs = beats[::4]
    p = make_plan(duration=12.0, beats=beats, downbeats=dbs, sections=[{"t": 0.0, "label": "drop"}],
                  risers=[], sfx=[], mood="calm", energy=0.2)
    m = S.render_soundtrack(p, SR)
    low = _low(m)
    for d in dbs[:-1]:
        pre = _rms(low[int((d - 0.03) * SR):int(d * SR)])
        post = _rms(low[int(d * SR):int((d + 0.03) * SR)])
        assert post > 1.5 * pre, (d, pre, post)


@pytest.mark.parametrize("mood", list(S.MOODS))
@pytest.mark.parametrize("energy", [0.2, 0.8])
def test_all_moods_render(mood, energy):
    p = make_plan(mood=mood, energy=energy, duration=6.0, key="F#", scale="major")
    m = S.render_soundtrack(p, SR)
    assert np.isfinite(m).all() and np.abs(m).max() < 1.0
    assert _rms(m[0]) > 0.02
    # no silence gaps longer than 0.25 s before the tail
    blk = int(0.25 * SR)
    lv = [_rms(m[0, i:i + blk]) for i in range(0, int(4.0 * SR), blk)]
    assert min(lv) > 1e-3


def test_soundtrack_speed():
    p = make_plan(duration=15.0, sections=[{"t": 0, "label": "intro"}, {"t": 4, "label": "drop"},
                                          {"t": 10, "label": "chorus"}])
    S.render_soundtrack(make_plan(duration=2.0), SR)  # warm caches
    t = time.perf_counter()
    S.render_soundtrack(p, SR)
    assert time.perf_counter() - t < 3.0


# ---------------------------------------------------------------- sfx


def test_riser_ends_at_t1():
    t0, t1 = 2.0, 3.5
    p = make_plan(duration=6.0, risers=[{"t0": t0, "t1": t1}], sfx=[])
    s = S.render_sfx(p, SR)
    e = int(round(t1 * SR))
    assert np.abs(s[:, : int(t0 * SR) - 1]).max() == 0.0
    assert np.abs(s[:, e:]).max() == 0.0  # nothing after t1
    assert _rms(s[0, e - int(0.1 * SR):e - int(0.005 * SR)]) > 0.05
    assert _rms(s[0, e - int(0.2 * SR):e]) > 4 * _rms(s[0, int(t0 * SR):int((t0 + 0.2) * SR)])
    # a riser given as an sfx event ends at its t
    p2 = make_plan(duration=6.0, risers=[], sfx=[{"t": 4.0, "type": "riser", "gain": 0.7}])
    s2 = S.render_sfx(p2, SR)
    assert np.abs(s2[:, int(round(4.0 * SR)):]).max() == 0.0
    assert _rms(s2[0, int(3.85 * SR):int(3.99 * SR)]) > 0.03


@pytest.mark.parametrize("typ", ["whoosh", "impact", "hit", "click", "pop", "glitch", "swell", "riser"])
def test_sfx_types(typ):
    p = make_plan(duration=5.0, risers=[], sfx=[{"t": 2.5, "type": typ, "gain": 1.0}])
    s = S.render_sfx(p, SR)
    assert np.isfinite(s).all() and np.abs(s).max() > 0.05 and np.abs(s).max() < 1.0
    near = s[:, int(1.4 * SR):int(3.6 * SR)]
    assert np.abs(near).max() == pytest.approx(np.abs(s).max())
    if typ in ("impact", "hit", "click", "pop", "glitch"):
        assert np.abs(s[:, : int(2.5 * SR) - 1]).max() == 0.0  # starts at t
    if typ in ("swell", "riser"):
        assert np.abs(s[:, int(2.5 * SR):]).max() == 0.0  # ends at t
    if typ == "whoosh":
        env = np.abs(s).max(axis=0)
        pk = np.argmax(signal.convolve(env, np.ones(960) / 960, "same")) / SR
        assert abs(pk - 2.5) < 0.12
        # pans from left to right
        l1, r1 = _rms(s[0, int(2.1 * SR):int(2.4 * SR)]), _rms(s[1, int(2.1 * SR):int(2.4 * SR)])
        l2, r2 = _rms(s[0, int(2.55 * SR):int(2.75 * SR)]), _rms(s[1, int(2.55 * SR):int(2.75 * SR)])
        assert l1 > r1 and r2 > l2
    gain_half = S.render_sfx(make_plan(duration=5.0, risers=[], sfx=[{"t": 2.5, "type": typ, "gain": 0.5}]), SR)
    assert np.abs(gain_half).max() == pytest.approx(0.5 * np.abs(s).max(), rel=1e-4)


# ---------------------------------------------------------------- mix


def _tones(dur=10.0):
    t = np.arange(int(dur * SR)) / SR
    music = np.stack([0.1 * np.sin(2 * np.pi * 100 * t)] * 2).astype(np.float32)
    return t, music


@pytest.mark.parametrize("duck_db", [-9.0, -6.0])
def test_mix_ducks_under_voice(duck_db):
    t, music = _tones()
    v = np.where((t >= 3.0) & (t < 6.0), 0.1 * np.sin(2 * np.pi * 1000 * t), 0.0)
    voice = np.stack([v, v]).astype(np.float32)
    out = M.mix(music, np.zeros_like(music), voice, SR, duck_db=duck_db)
    base = _tone_level(out, 100, 1.0, 2.5)
    ducked = _tone_level(out, 100, 4.0, 5.5)
    db = 20 * np.log10(ducked / base)
    assert abs(db - duck_db) < 1.0, db
    # recovers after the voice (release 250 ms)
    assert abs(20 * np.log10(_tone_level(out, 100, 7.0, 8.5) / base)) < 0.5
    # attack is fast: already mostly ducked 60 ms after the voice starts
    early = _tone_level(out, 100, 3.06, 3.16)
    assert 20 * np.log10(early / base) < 0.6 * duck_db


def test_mix_ducks_under_impacts():
    t, music = _tones()
    hit = np.where((t >= 5.0) & (t < 5.6), 0.3 * np.sin(2 * np.pi * 2000 * t), 0.0)
    sfx = np.stack([hit, hit]).astype(np.float32)
    out = M.mix(music, sfx, None, SR)
    db = 20 * np.log10(_tone_level(out, 100, 5.1, 5.5) / _tone_level(out, 100, 2.0, 3.0))
    assert -4.5 < db < -1.5, db


@pytest.mark.parametrize("target", [-14.0, -18.0])
def test_mix_loudness(plan, music, target):
    sfx = S.render_sfx(plan, SR)
    out = M.mix(music, sfx, None, SR, target_lufs=target)
    assert abs(M.loudness_lufs(out, SR) - target) <= 1.5
    assert np.abs(out).max() < 10 ** (-1.0 / 20.0) + 1e-6


def test_mix_edge_cases():
    m = np.zeros((2, SR), np.float32)
    out = M.mix(m, m, None, SR)
    assert out.shape == (2, SR) and not np.any(out)
    mono = (0.2 * np.sin(np.linspace(0, 400 * np.pi, SR))).astype(np.float32)
    out = M.mix(mono[None], mono[: SR // 2][None], mono[None, : SR // 3], SR)
    assert out.shape == (2, SR) and np.isfinite(out).all()


def test_loudness_meter_reference():
    """A 997 Hz sine at -20 dBFS peak in both channels reads about -20 + 3 - 0.691 + 0.69 ~ -17 LUFS (stereo sum)."""
    t = np.arange(5 * SR) / SR
    x = 0.1 * np.sin(2 * np.pi * 997 * t)
    L = M.loudness_lufs(np.stack([x, x]), SR)
    # mono 997 Hz sine at amplitude A: L = -0.691 + 10 log10(A^2 / 2) + ~0.0x (K-weight ~0 dB at 1 kHz); stereo adds 3.01
    expected = -0.691 + 10 * np.log10(0.01 / 2) + 3.01
    assert abs(L - expected) < 0.7, (L, expected)
