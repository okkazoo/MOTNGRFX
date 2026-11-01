"""Stem mixing: sidechain ducking, impact ducking, loudness, limiting, fades (SPEC Interface 4)."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .synth import biquad

__all__ = ["mix", "loudness_lufs", "k_weight", "soft_limit"]

_BLOCK = 0.005  # envelope block, seconds


def _as_stereo(x, n: int | None = None) -> np.ndarray | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] > 8 and a.shape[1] <= 8:
        a = a.T
    if a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    elif a.shape[0] > 2:
        a = a[:2]
    a = np.nan_to_num(a)
    if n is not None:
        if a.shape[1] < n:
            a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
        else:
            a = a[:, :n]
    return a


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K-weighting approximation: +4 dB high shelf @ 1.68 kHz, then 2nd-order HP @ 38 Hz."""
    b1, a1 = biquad("highshelf", 1681.97, sr, 0.7071, 4.0)
    b2, a2 = biquad("highpass", 38.13, sr, 0.5)
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness (BS.1770-style, 400 ms blocks, 75 % overlap). -inf for silence."""
    y = k_weight(_as_stereo(x), sr)
    n = y.shape[1]
    blk, hop = int(0.4 * sr), int(0.1 * sr)
    if n == 0:
        return float("-inf")
    if n < blk:
        z = np.array([float((y ** 2).mean(axis=1).sum())])
    else:
        cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y ** 2, axis=1)], axis=1)
        st = np.arange(0, n - blk + 1, hop)
        z = ((cs[:, st + blk] - cs[:, st]) / blk).sum(axis=0)
    with np.errstate(divide="ignore"):
        lz = -0.691 + 10.0 * np.log10(z)
    z = z[lz > -70.0]
    if len(z) == 0:
        return float("-inf")
    rel = -0.691 + 10.0 * math.log10(float(z.mean())) - 10.0
    with np.errstate(divide="ignore"):
        z2 = z[-0.691 + 10.0 * np.log10(z) > rel]
    if len(z2) == 0:
        z2 = z
    return float(-0.691 + 10.0 * math.log10(float(z2.mean())))


def _block_db(x_mono: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    b = max(1, int(_BLOCK * sr))
    nb = int(math.ceil(len(x_mono) / b))
    pad = np.pad(x_mono, (0, nb * b - len(x_mono)))
    ms = (pad.reshape(nb, b) ** 2).mean(axis=1)
    return 10.0 * np.log10(ms + 1e-12), b


def _follow(target: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Attack/release one-pole follower over block-rate values (rising = attack)."""
    ca = math.exp(-_BLOCK / max(attack, 1e-6))
    cr = math.exp(-_BLOCK / max(release, 1e-6))
    out = np.empty_like(target)
    y = 0.0
    for i, v in enumerate(target.tolist()):
        c = ca if v > y else cr
        y = v + c * (y - v)
        out[i] = y
    return out


def _blocks_to_samples(v: np.ndarray, b: int, n: int) -> np.ndarray:
    centres = (np.arange(len(v)) + 0.5) * b
    return np.interp(np.arange(n), centres, v)


def soft_limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """Look-ahead gain-riding limiter (2.5 ms blocks, 80 ms release) + tanh soft clip to the ceiling."""
    c = 10.0 ** (ceiling_db / 20.0)
    n = x.shape[1]
    b = max(1, int(0.0025 * sr))
    nb = int(math.ceil(n / b))
    pk = np.pad(np.abs(x).max(axis=0), (0, nb * b - n)).reshape(nb, b).max(axis=1)
    need = np.minimum(1.0, (0.9 * c) / np.maximum(pk, 1e-9))
    need = np.minimum(need, np.concatenate([need[1:], [1.0]]))  # one block look-ahead
    need = np.minimum(need, np.concatenate([[1.0], need[:-1]]))
    cr = math.exp(-0.0025 / 0.08)
    g = np.empty(nb)
    y = 1.0
    for i, v in enumerate(need.tolist()):
        y = v if v < y else v + cr * (y - v)
        g[i] = y
    gs = np.interp(np.arange(n), (np.arange(nb) + 0.5) * b, g)
    y = x * gs[None, :]
    k = 0.8 * c
    a = np.abs(y)
    over = a > k
    y = np.where(over, np.sign(y) * (k + (c - k) * np.tanh((a - k) / (c - k))), y)
    return y


def mix(music: np.ndarray, sfx: np.ndarray, voice: np.ndarray | None, sample_rate: int,
        duck_db: float = -9.0, target_lufs: float = -14.0) -> np.ndarray:
    """Sum stems with ducking, normalise to target_lufs, limit to -1 dBFS, fade. Returns (2, N) float32."""
    sr = int(sample_rate)
    lens = [np.asarray(s).shape[-1] for s in (music, sfx, voice) if s is not None]
    N = int(np.asarray(music).shape[-1]) if music is not None else (max(lens) if lens else 0)
    m = _as_stereo(music, N) if music is not None else np.zeros((2, N))
    s = _as_stereo(sfx, N) if sfx is not None else np.zeros((2, N))
    v = _as_stereo(voice, N)
    if N == 0:
        return np.zeros((2, 0), np.float32)

    gain_db = np.zeros(N)
    # sidechain duck under the voice: 20 ms attack / 250 ms release
    if v is not None and np.any(v):
        vdb, b = _block_db(v.mean(axis=0), sr)
        target = np.clip((vdb + 50.0) / 10.0, 0.0, 1.0)
        r = _follow(target, 0.020, 0.250)
        gain_db += duck_db * _blocks_to_samples(r, b, N)
    # music -3 dB around loud sfx (impacts): 5 ms attack / 300 ms release
    if np.any(s):
        sdb, b = _block_db(s.mean(axis=0), sr)
        target = np.clip((sdb + 30.0) / 12.0, 0.0, 1.0)
        r = _follow(target, 0.005, 0.300)
        gain_db += -3.0 * _blocks_to_samples(r, b, N)
    out = m * (10.0 ** (gain_db / 20.0))[None, :] + s
    if v is not None:
        out = out + v

    # loudness normalisation (iterate around the limiter)
    pre = out
    total = 1.0
    lim = soft_limit(pre, sr)
    for _ in range(3):
        L = loudness_lufs(lim, sr)
        if not np.isfinite(L):
            break
        d = target_lufs - L
        if abs(d) < 0.3:
            break
        total *= 10.0 ** (d / 20.0)
        lim = soft_limit(pre * total, sr)
    out = lim

    # fades
    fi, fo = min(N, int(0.02 * sr)), min(N, int(0.3 * sr))
    if fi > 0:
        out[:, :fi] *= np.linspace(0.0, 1.0, fi)[None, :]
    if fo > 0:
        out[:, N - fo:] *= np.linspace(1.0, 0.0, fo)[None, :]
    return np.ascontiguousarray(out, dtype=np.float32)
