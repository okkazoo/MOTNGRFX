"""Deterministic audio features for the audio-reactive mathtype pieces.

Features on a fixed 200 Hz grid (frame centres at k * 5 ms), from a wav:
  rms     full-band RMS (30 ms centred window)
  sub     band envelope  < 120 Hz        (zero-phase Butterworth bands, centred RMS: 30 ms sub,
                                          15 ms low-mid, 10 ms mid/high)
  lowmid  band envelope  120 - 600 Hz
  mid     band envelope  600 - 2000 Hz
  high    band envelope  > 2000 Hz
  onset   spectral-flux onset strength (mograph.analyze._onset_envelope, lag-corrected), resampled
Each raw feature is normalised to 0-1 (divided by its 99.5th percentile, clipped).

`Features.curve(name, attack, release)` smooths a feature with a one-pole follower that uses a
separate attack and release time constant, run once over the whole array (no state between video
frames), then renormalises it to 0-1 (by max, or min-max with norm="minmax"). `Features.feat(name,
t, attack, release)` interpolates that curve at any time(s) t, so trail samples at past times read
the features at those times: the visuals stay a pure function of t. `Features.integral(...)` is the
running integral of a curve (for rates, e.g. wind speed), also interpolated at any t.

CLI: python tools/mathtype/audiofeat.py x.wav [--events x.json] [--png plot.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOP = 0.005
BANDS = {"sub": (None, 120.0), "lowmid": (120.0, 600.0), "mid": (600.0, 2000.0), "high": (2000.0, None)}
WIN = {"rms": 0.030, "sub": 0.030, "lowmid": 0.015, "mid": 0.010, "high": 0.010}   # >= 1.5 periods of the band floor
NAMES = ("rms", "sub", "lowmid", "mid", "high", "onset")


def _env(y: np.ndarray, sr: int, win: float, centres: np.ndarray) -> np.ndarray:
    ms = uniform_filter1d(y * y, max(1, int(round(win * sr))), mode="constant")
    idx = np.clip(np.round(centres * sr).astype(np.int64), 0, y.size - 1)
    return np.sqrt(np.maximum(ms[idx], 0.0))


def _norm(a: np.ndarray) -> np.ndarray:
    ref = np.percentile(a, 99.5)
    return np.clip(a / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(a)


def follow(x: np.ndarray, attack: float, release: float, hop: float = HOP) -> np.ndarray:
    """One-pole envelope follower with separate attack / release time constants (s)."""
    ka = 1.0 - np.exp(-hop / max(attack, 1e-6))
    kr = 1.0 - np.exp(-hop / max(release, 1e-6))
    y = np.empty_like(x)
    v = float(x[0])
    for i, xi in enumerate(x.tolist()):
        v += (ka if xi > v else kr) * (xi - v)
        y[i] = v
    return y


class Features:
    def __init__(self, wav: str, hop: float = HOP):
        sr, x0 = wavfile.read(wav)
        x = x0.astype(np.float64)
        if np.issubdtype(x0.dtype, np.integer):
            x /= float(np.iinfo(x0.dtype).max) + 1.0
        x = x.mean(axis=1) if x.ndim == 2 else x
        self.sr, self.hop, self.wav = sr, hop, wav
        self.duration = x.size / sr
        self.t = np.arange(int(np.ceil(self.duration / hop)) + 1) * hop
        raw = {"rms": _env(x, sr, WIN["rms"], self.t)}
        for name, (lo, hi) in BANDS.items():
            if lo is None:
                sos = signal.butter(4, hi, "low", fs=sr, output="sos")
            elif hi is None:
                sos = signal.butter(4, lo, "high", fs=sr, output="sos")
            else:
                sos = signal.butter(4, [lo, hi], "band", fs=sr, output="sos")
            raw[name] = _env(signal.sosfiltfilt(sos, x), sr, WIN[name], self.t)
        raw["onset"] = self._onset(x, sr)
        self.raw = {k: _norm(v) for k, v in raw.items()}
        self._cache: dict = {}

    def _onset(self, x: np.ndarray, sr: int) -> np.ndarray:
        sys.path.insert(0, REPO)
        from mograph.analyze import _ONSET_LAG, _onset_envelope, _resample
        y, sr2 = _resample(x.astype(np.float32), sr)
        env, fr, _, _, _ = _onset_envelope(y, sr2)
        te = np.arange(env.size) / fr - _ONSET_LAG
        return np.interp(self.t, te, env, left=0.0, right=0.0)

    def curve(self, name: str, attack: float = 0.0, release: float = 0.0, norm: str = "max") -> np.ndarray:
        key = (name, attack, release, norm)
        if key not in self._cache:
            a = self.raw[name]
            if attack > 0 or release > 0:
                a = follow(a, attack, release, self.hop)
            if norm == "minmax":
                lo, hi = a.min(), a.max()
                a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
            else:
                a = a / a.max() if a.max() > 0 else a
            self._cache[key] = a
        return self._cache[key]

    def feat(self, name: str, t, attack: float = 0.0, release: float = 0.0, norm: str = "max"):
        """Feature value(s) at time(s) t (any shape); clamped to the ends outside the file."""
        a = self.curve(name, attack, release, norm)
        t = np.asarray(t, float)
        return np.interp(t.ravel(), self.t, a).reshape(t.shape)

    def integral(self, name: str, t, attack: float = 0.0, release: float = 0.0, norm: str = "max"):
        """Running integral  int_0^t feat(tau) dtau  (feature-seconds) at time(s) t."""
        key = ("int", name, attack, release, norm)
        if key not in self._cache:
            a = self.curve(name, attack, release, norm)
            self._cache[key] = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * self.hop)])
        c = self._cache[key]
        t = np.asarray(t, float)
        a = self.curve(name, attack, release, norm)
        tr = t.ravel()
        out = np.interp(tr, self.t, c)                                    # clamped inside the file
        out += np.minimum(tr, 0.0) * a[0] + np.maximum(tr - self.t[-1], 0.0) * a[-1]   # linear beyond
        return out.reshape(t.shape)


def plot(F: Features, png: str, events: dict | None = None, smooth: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(NAMES), 1, figsize=(12, 1.4 * len(NAMES)), sharex=True)
    for a, n in zip(ax, NAMES):
        a.plot(F.t, F.raw[n], lw=0.6, color="0.6")
        if smooth and n in smooth:
            a.plot(F.t, F.curve(n, *smooth[n]), lw=1.2, color="C1")
        a.set_ylabel(n)
        a.set_ylim(0, 1.05)
        if events:
            for e in events.get("kick", []):
                a.axvline(e["t"], color="C0", lw=0.6, alpha=0.6)
            for e in events.get("snare", []):
                a.axvline(e["t"], color="C3", lw=0.6, alpha=0.6, ls="--")
            for c in events.get("chords", []):
                a.axvline(c["t"], color="C2", lw=1.2, alpha=0.8)
    ax[-1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--events")
    ap.add_argument("--png")
    a = ap.parse_args()
    F = Features(a.wav)
    ev = json.load(open(a.events, encoding="utf-8")) if a.events else None
    print(f"{a.wav}: {F.duration:.3f} s, {F.t.size} frames @ {1 / F.hop:.0f} Hz")
    for n in NAMES:
        r = F.raw[n]
        print(f"  {n:7s} mean {r.mean():.3f}  p10 {np.percentile(r, 10):.3f}  p90 {np.percentile(r, 90):.3f}")
    if ev:   # how well each band tracks the ground-truth hits (mean value 0-40 ms after vs 150 ms before)
        for inst in ("kick", "snare", "hat"):
            ts = np.array([e["t"] for e in ev.get(inst, [])])
            if ts.size:
                after = np.mean([F.feat(n, ts + 0.02) for n in NAMES], axis=1)
                before = np.mean([F.feat(n, ts - 0.1) for n in NAMES], axis=1)
                print(f"  {inst:5s} at hit / 100 ms before: " +
                      "  ".join(f"{n} {x:.2f}/{y:.2f}" for n, x, y in zip(NAMES, after, before)))
    if a.png:
        plot(F, a.png, ev)
        print("plot", a.png)


if __name__ == "__main__":
    main()
