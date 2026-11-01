"""Music mix chain for stems: per-part EQ, sidechain, plate / hall / tempo-delay sends, glue, true-peak limiter.

Builds on mograph.synth (biquad, reverb) and mograph.mix (loudness_lufs, block followers). All arrays are
(2, n) float64 unless noted. Design of the send/glue/limit order follows op55 (github.com/ynginkim/op55, MIT);
the code is our own.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

from .mix import _block_db, _blocks_to_samples, _follow, loudness_lufs
from .synth import biquad, reverb

__all__ = ["eq", "mono_below", "duck_curve", "send_reverb", "send_delay", "glue", "true_peak_db", "limit_tp", "master",
           "widen", "wav24_bytes", "write_wav24", "sha256_bytes"]


def eq(x: np.ndarray, sr: int, bands) -> np.ndarray:
    """Series biquads. bands: (kind, f0[, q[, gain_db]]) e.g. ("highpass", 120), ("peak", 300, 0.9, -3)."""
    for b in bands:
        kind, f0, *rest = b
        q = rest[0] if rest else 0.7071
        g = rest[1] if len(rest) > 1 else 0.0
        x =signal.lfilter(*biquad(kind, f0, sr, q, g), x, axis=-1)
    return x


def mono_below(x: np.ndarray, sr: int, f0: float = 120.0) -> np.ndarray:
    """Sum everything below f0 to mono (low = lowpass, high = x - low, so the split is transparent)."""
    low = signal.lfilter(*biquad("lowpass", f0, sr, 0.7071), x, axis=-1)
    return (x - low) + low.mean(axis=0, keepdims=True)


def duck_curve(times, depth_db: float, n: int, sr: int, tau: float = 0.18, attack: float = 0.004) -> np.ndarray:
    """Linear sidechain gain (n,): each trigger pulls -depth_db, recovering with time constant tau."""
    env = np.zeros(n)
    for tk in times:
        s = int(round(tk * sr))
        m = min(n - s, int(tau * 5 * sr))
        if s < 0 or m <= 0:
            continue
        tt = np.arange(m) / sr
        env[s:s + m] = np.maximum(env[s:s + m], depth_db * np.exp(-tt / tau) * np.minimum(1.0, tt / attack))
    return 10.0 ** (-env / 20.0)


def send_reverb(x: np.ndarray, sr: int, decay: float, predelay: float) -> np.ndarray:
    """Wet-only reverb return (the dry path is removed). Tail is cut at len(x): leave silence at the end of the bus."""
    x = np.atleast_2d(x)
    return (reverb(x, sr, decay=decay, wet=1.0, predelay=predelay) - x) / 0.35


def send_delay(x: np.ndarray, sr: int, bpm: float, beats: float = 0.75, feedback: float = 0.38, lp: float = 3600.0,
               hp: float = 180.0, repeats: int = 7) -> np.ndarray:
    """Tempo-synced ping-pong delay, wet only. beats=0.75 is a dotted eighth. Each repeat is low-passed again."""
    n = x.shape[1]
    D = int(round(beats * 60.0 / bpm * sr))
    cur = x.mean(axis=0)
    b, a = biquad("lowpass", lp, sr, 0.7071)
    wet = np.zeros((2, n))
    for k in range(1, repeats + 1):
        cur = signal.lfilter(b, a, cur) * feedback
        if k * D >= n:
            break
        main, other = k % 2, 1 - k % 2  # alternate L / R, with some crosstalk
        wet[main, k * D:] += cur[:n - k * D]
        wet[other, k * D:] += 0.3 * cur[:n - k * D]
    return signal.lfilter(*biquad("highpass", hp, sr, 0.7071), wet, axis=-1)


def glue(x: np.ndarray, sr: int, thr_db: float = -22.0, ratio: float = 2.0, knee_db: float = 6.0,
         attack: float = 0.03, release: float = 0.30) -> tuple[np.ndarray, float]:
    """Stereo-linked feed-forward compressor (block rate 5 ms). Returns (y, max gain reduction in dB)."""
    lvl, b = _block_db(np.sqrt((x ** 2).mean(axis=0)), sr)  # 10*log10(mean(rms^2)) -> dB of block power
    over = lvl - thr_db
    if knee_db > 0:
        soft = np.where(over > knee_db / 2, over, np.where(over > -knee_db / 2, (over + knee_db / 2) ** 2 / (2 * knee_db), 0.0))
    else:
        soft = np.maximum(over, 0.0)
    gr = _follow(soft * (1.0 - 1.0 / ratio), attack, release)
    g = 10.0 ** (-_blocks_to_samples(gr, b, x.shape[1]) / 20.0)
    return x * g[None, :], float(gr.max())


def true_peak_db(x: np.ndarray, os: int = 4) -> float:
    """Inter-sample peak estimate in dBTP: os-times polyphase oversampling, max |.|."""
    up = signal.resample_poly(x, os, 1, axis=-1)
    return 20.0 * math.log10(float(np.abs(up).max()) + 1e-12)


def limit_tp(x: np.ndarray, sr: int, ceiling_db: float = -1.5, release: float = 0.12, block: int = 48,
             look: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Look-ahead true-peak limiter: gain from the 4x-oversampled peak of every `block` samples, min-filtered over
    +/- `look` blocks (so the gain is already down when the peak arrives, linear ramp between block centres) and
    released exponentially. No clipper. Returns (y, gain-reduction dB per block)."""
    c = 10.0 ** (ceiling_db / 20.0)
    n = x.shape[1]
    nb = int(math.ceil(n / block))
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(axis=0)
    up = np.pad(up, (0, nb * block * 4 - len(up)))
    pk = up.reshape(nb, block * 4).max(axis=1)
    need = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    need = minimum_filter1d(need, size=2 * look + 1, mode="nearest")
    a = 1.0 - math.exp(-block / (sr * release))
    g = np.empty(nb)
    y = 1.0
    for i, v in enumerate(need.tolist()):
        y = v if v < y else min(v, y + (1.0 - y) * a)
        g[i] = y
    gs = np.interp(np.arange(n), (np.arange(nb) + 0.5) * block, g)
    return x * gs[None, :], -20.0 * np.log10(g)


def master(x: np.ndarray, sr: int, target_lufs: float = -14.0, tp_max: float = -1.5, tol: float = 0.1,
           max_iter: int = 12) -> tuple[np.ndarray, dict]:
    """Gain + true-peak limiter iterated until integrated loudness is within tol of target and dBTP <= tp_max.
    Returns (y, report); report["gain"] is the linear gain applied before the limiter."""
    g = 10.0 ** ((target_lufs - loudness_lufs(x, sr)) / 20.0)
    ceil = tp_max - 0.1
    best = None
    for it in range(max_iter):
        y, gr = limit_tp(x * g, sr, ceil)
        lu, tp = loudness_lufs(y, sr), true_peak_db(y)
        best = (y, {"lufs": lu, "true_peak_dbtp": tp, "sample_peak_dbfs": 20 * math.log10(float(np.abs(y).max()) + 1e-12),
                    "limiter_max_gr_db": float(gr.max()), "limiter_ceiling_db": ceil, "gain": g, "iterations": it + 1})
        ok_l, ok_t = abs(lu - target_lufs) <= tol, tp <= tp_max
        if ok_l and ok_t:
            break
        if not ok_t:
            ceil -= (tp - tp_max) + 0.02
        if not ok_l:
            g *= 10.0 ** ((target_lufs - lu) / 20.0)
    return best


def wav24_bytes(x: np.ndarray, sr: int) -> bytes:
    """(2, n) float in [-1, 1] -> 24-bit PCM WAV bytes (deterministic)."""
    x = np.atleast_2d(x)
    ch, n = x.shape
    i = np.ascontiguousarray(np.round(np.clip(x.T, -1.0, 1.0 - 2 ** -23) * 8388607.0).astype("<i4"))
    pcm = i.view(np.uint8).reshape(n, ch, 4)[:, :, :3].tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 3, ch * 3, 24)
    return hdr + b"data" + struct.pack("<I", len(pcm)) + pcm


def write_wav24(path, x: np.ndarray, sr: int) -> str:
    data = wav24_bytes(x, sr)
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def widen(x: np.ndarray, width: float = 1.25) -> np.ndarray:
    """Mid/side width: side * width (1.0 = unchanged, 0 = mono). (2, n) in, (2, n) out; mid is untouched, so the
    mono sum (and any bass that was already summed to mono) stays as it was. Mono input comes back as stereo copies."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1]) * float(width)
    return np.stack([mid + side, mid - side])
