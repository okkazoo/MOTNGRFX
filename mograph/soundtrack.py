"""Soundtrack assembly shared by the ComfyUI node and the end-to-end tests.

User audio (plan.audio.source == "user") passes through, trimmed/padded to the plan
duration with short fades. Otherwise: synth music bed + sfx, mixed (W2's synth/mix).
"""
from __future__ import annotations

import numpy as np

FADE_IN = 0.02
FADE_OUT = 0.3


def fit_user_audio(waveform: np.ndarray, sample_rate: int, duration: float,
                   fade_in: float = FADE_IN, fade_out: float = FADE_OUT) -> np.ndarray:
    """(channels, samples) float32 trimmed or zero-padded to ``duration`` seconds, with fades."""
    a = np.asarray(waveform, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    n = int(round(float(duration) * int(sample_rate)))
    out = np.zeros((a.shape[0], n), np.float32)
    m = min(n, a.shape[1])
    out[:, :m] = a[:, :m]
    end = m                                     # fade the real audio's tail, not the padding
    fi = min(end, int(round(fade_in * sample_rate)))
    if fi > 0:
        out[:, :fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    fo = min(end, int(round(fade_out * sample_rate)))
    if fo > 0:
        out[:, end - fo:end] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return np.clip(np.nan_to_num(out), -1.0, 1.0)


def build_soundtrack(plan: dict, analysis: dict, sample_rate: int = 48000) -> tuple[np.ndarray, int]:
    """-> ((channels, samples) float32, sample_rate) covering exactly plan["duration"]."""
    audio_in = (analysis or {}).get("audio")
    if plan.get("audio", {}).get("source") == "user" and audio_in and audio_in.get("waveform") is not None:
        sr = int(audio_in["sample_rate"])
        return fit_user_audio(audio_in["waveform"], sr, plan["duration"]), sr
    from . import mix, synth
    music = synth.render_soundtrack(plan, sample_rate)
    sfx = synth.render_sfx(plan, sample_rate)
    out = mix.mix(music, sfx, None, sample_rate)
    return np.asarray(out, dtype=np.float32), int(sample_rate)
