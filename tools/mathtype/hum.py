"""Pythagorean string hum for the 11_mathtype tests (5.0 s, 48 kHz stereo, -14 LUFS).

- Drone on A2 (110 Hz): harmonics n = 1..8 at amplitude 1/n, plus a chorus voice
  detuned by a 0.3 Hz LFO so each partial beats slowly.
- Karplus-Strong plucks at the phrase landings on ratios 2, 3, 4 (220/330/440 Hz):
  t = 0.35, 1.95, 3.55 s. With the drone (ratio 1) this spells 1:2:3:4.
- A soft, glassy high-partial swell at 4.2 s for the credit.
- Fade in 0.15 s, fade out over the last 0.4 s, exact silence at 5.0 s.

Loudness: pyloudnorm is used if installed; otherwise RMS normalisation followed by a
measurement with ffmpeg's ebur128 filter and one corrective gain (reported on stdout).

Usage: python tools/mathtype/hum.py [out.wav]
       python tools/mathtype/hum.py --duration 10 --plucks 1.9 5.0 7.5 [--credit 8.4] --out x.wav
With no flags the output is identical to the 5 s default above. --plucks takes three times,
voiced on ratios 2, 3, 4 in order. --credit defaults to 4.2 s at 5 s, else 0.84 * duration.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

SR = 48000
DUR = 5.0
N = int(SR * DUR)
F0 = 110.0
PLUCKS = [(0.35, 2), (1.95, 3), (3.55, 4)]   # (time s, harmonic ratio)
CREDIT_T = 4.2
TARGET_LUFS = -14.0

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "11_mathtype", "v001", "local", "hum.wav")


def drone(t: np.ndarray) -> np.ndarray:
    """Stereo drone, shape (2, N)."""
    out = np.zeros((2, t.size))
    lfo_ph = 2 * np.pi * 0.3 * t
    for n in range(1, 9):
        a = 1.0 / n
        f = F0 * n
        base = np.sin(2 * np.pi * f * t + 0.7 * n)
        # chorus voice: frequency f * (1 + d*sin(0.3 Hz)); phase = integral of 2*pi*f(t)
        d = 0.0025
        ph = 2 * np.pi * f * t - (2 * np.pi * f * d / (2 * np.pi * 0.3)) * np.cos(lfo_ph)
        chL = np.sin(ph + 1.3 * n)
        chR = np.sin(ph + 2.1 * n + np.pi / 3)
        out[0] += a * (0.6 * base + 0.4 * chL)
        out[1] += a * (0.6 * base + 0.4 * chR)
    # gentle breathing on the drone
    out *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.2 * t - np.pi / 2) ** 2
    return out


def karplus_strong(freq: float, length: int, rng: np.random.Generator,
                   decay: float = 0.9965, bright: float = 0.55) -> np.ndarray:
    """Plucked string: delay line with a two-tap averaging loop filter (vectorised per period)."""
    P = max(2, int(round(SR / freq - 0.5)))          # averaging filter adds half a sample of delay
    exc = rng.uniform(-1, 1, P)
    # soften the excitation (one-pole low-pass) so the pluck is round, not a click
    for i in range(1, P):
        exc[i] = bright * exc[i] + (1 - bright) * exc[i - 1]
    exc -= exc.mean()
    y = np.zeros(length + P + 1)
    y[:P] = exc
    i = P
    while i < length:
        j = min(i + P, length)
        # y[n] = decay * 0.5 * (y[n-P] + y[n-P-1]); block of up to P samples depends only on the past
        prev = y[i - P:j - P]
        prev1 = y[i - P - 1:j - P - 1] if i - P - 1 >= 0 else np.concatenate(([0.0], y[i - P:j - P - 1]))
        y[i:j] = decay * 0.5 * (prev + prev1)
        i = j
    return y[:length]


def plucks(t: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1729)
    out = np.zeros((2, t.size))
    pans = {2: -0.25, 3: 0.25, 4: 0.0}
    for (t0, r) in PLUCKS:
        s0 = int(round(t0 * SR))
        s = karplus_strong(F0 * r, N - s0, rng)
        # attack ramp of 2 ms to keep it soft
        ramp = np.minimum(1.0, np.arange(s.size) / (0.002 * SR))
        s = s * ramp
        p = pans[r]
        out[0, s0:] += s * np.sqrt(0.5 * (1 - p))
        out[1, s0:] += s * np.sqrt(0.5 * (1 + p))
    return out


def glass(t: np.ndarray) -> np.ndarray:
    """Soft high glassy partials swelling in at the credit (ratios 12, 16, 20, 24 of A2)."""
    out = np.zeros((2, t.size))
    u = np.clip((t - CREDIT_T) / 0.45, 0, 1)
    env = 0.5 - 0.5 * np.cos(np.pi * u)                     # raised-cosine swell
    env *= np.exp(-np.clip(t - CREDIT_T - 0.45, 0, None) * 1.2)
    for k, (r, a) in enumerate([(12, 0.35), (16, 0.5), (20, 0.25), (24, 0.3)]):
        f = F0 * r
        vib = 0.002 * np.sin(2 * np.pi * (4.5 + k) * t)
        ph = 2 * np.pi * f * t * (1 + vib)
        out[0] += a * np.sin(ph + k)
        out[1] += a * np.sin(ph + k + 0.9)
    return out * env


def envelope(t: np.ndarray) -> np.ndarray:
    fin = np.clip(t / 0.15, 0, 1)
    fin = 0.5 - 0.5 * np.cos(np.pi * fin)
    fout = np.clip((DUR - t) / 0.4, 0, 1)
    fout = 0.5 - 0.5 * np.cos(np.pi * fout)
    return fin * fout


def measure_lufs_ffmpeg(x: np.ndarray) -> float:
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        wavfile.write(tmp, SR, x.T.astype(np.float32))
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", tmp, "-af", "ebur128",
                            "-f", "null", "-"], capture_output=True, text=True)
        m = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
        if not m:
            raise RuntimeError("ebur128 measurement failed:\n" + r.stderr[-2000:])
        return float(m[-1])
    finally:
        os.remove(tmp)


def normalise(x: np.ndarray) -> tuple[np.ndarray, str, float]:
    try:
        import pyloudnorm as pyln  # type: ignore
        meter = pyln.Meter(SR)
        L = meter.integrated_loudness(x.T)
        x = x * 10 ** ((TARGET_LUFS - L) / 20)
        return x, "pyloudnorm", meter.integrated_loudness(x.T)
    except ImportError:
        pass
    # RMS normalisation to the target as dBFS, then one ebur128-measured correction
    rms = np.sqrt(np.mean(x ** 2))
    x = x * (10 ** (TARGET_LUFS / 20) / rms)
    L = measure_lufs_ffmpeg(x)
    x = x * 10 ** ((TARGET_LUFS - L) / 20)
    return x, "rms+ffmpeg-ebur128-correction", measure_lufs_ffmpeg(x)


def build() -> np.ndarray:
    t = np.arange(N) / SR
    x = 0.24 * drone(t) + 1.0 * plucks(t) + 0.35 * glass(t)
    x *= envelope(t)
    return x


def main() -> None:
    global DUR, N, PLUCKS, CREDIT_T
    ap = argparse.ArgumentParser(description="Pythagorean string hum")
    ap.add_argument("out_pos", nargs="?", default=None, help="output wav (legacy positional)")
    ap.add_argument("--out", default=None, help="output wav")
    ap.add_argument("--duration", type=float, default=DUR, help="length in s (default 5.0)")
    ap.add_argument("--plucks", type=float, nargs=3, default=None, metavar="T",
                    help="three pluck times in s (ratios 2, 3, 4)")
    ap.add_argument("--credit", type=float, default=None, help="credit swell time in s")
    a = ap.parse_args()
    out = a.out or a.out_pos or DEFAULT_OUT
    if a.duration != DUR:
        DUR = a.duration
        N = int(SR * DUR)
        CREDIT_T = 0.84 * DUR
    if a.plucks is not None:
        PLUCKS = [(t, r) for t, r in zip(a.plucks, (2, 3, 4))]
    if a.credit is not None:
        CREDIT_T = a.credit
    os.makedirs(os.path.dirname(out), exist_ok=True)
    x = build()
    x, method, lufs = normalise(x)
    peak = float(np.abs(x).max())
    if peak > 0.98:
        raise SystemExit(f"peak {peak:.3f} would clip at -14 LUFS; rebalance the mix")
    x[:, -1] = 0.0
    wavfile.write(out, SR, (np.clip(x, -1, 1).T * 32767).astype(np.int16))
    print(f"wrote {out}: {x.shape[1]} samples ({x.shape[1] / SR:.3f} s), 2 ch, {SR} Hz; "
          f"loudness {lufs:.2f} LUFS via {method}; peak {20 * np.log10(peak):.2f} dBFS")


if __name__ == "__main__":
    main()
