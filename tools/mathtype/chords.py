"""Moody low chord soundtrack for the 11_mathtype pieces (48 kHz stereo, -14 LUFS, <= -1 dBTP).

A minor / D dorian, ~96 bpm feel, no drums. Three layers:
- Bass: sustained sine root (55-110 Hz) with a little 2nd harmonic, moving with the chords.
- Pad: detuned saw/triangle stacks voiced ~120-330 Hz, 2-pole low-pass at 1.5 kHz,
  ~0.45 s attack, long release, slight chorus. Each pitch has its own gate, so common
  tones between chords are held, not re-attacked.
- Top line: soft FM electric piano (ratio-1 FM, fast-decaying index), 3-note broken
  chord at each change (eighth notes at 96 bpm), fundamentals <= 659 Hz, low-passed 3.5 kHz.
Room: convolution with a synthetic decaying-noise IR (RT60 2.0 s, damped above 3 kHz).

Progression cycles per change: Am9 -> Fmaj7 -> Cmaj7/E -> Dm9 -> Am9 ...
Before the first change the pad/bass sit on Dm9 (a pickup that resolves to Am9 at the
first change); with --loop the opening plays the last segment's chord so the seam is seamless.

Loudness: RMS normalisation, then gain correction from an ffmpeg ebur128 measurement.
True peak is estimated with 4x oversampling; a linked look-ahead limiter engages only if
the -14 LUFS mix would exceed -1 dBTP. Deterministic for a given --seed.

Usage: python tools/mathtype/chords.py --duration 10 --changes 1.09375,4.53125,7.96875 \
           --out x.wav [--seed 7] [--loop]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
BPM = 96.0
BEAT = 60.0 / BPM
TARGET_LUFS = -14.0
TP_LIMIT = -1.0          # dBTP ceiling
TP_MARGIN = 0.3          # aim this far under the ceiling

# name, bass MIDI (55-110 Hz), pad voicing MIDI (voice-led, common tones held), top line MIDI
PROG = [
    ("Am9",     33, (48, 52, 55, 59, 64), (64, 67, 71)),   # A1 | C3 E3 G3 B3 E4 | E4 G4 B4
    ("Fmaj7",   41, (48, 53, 57, 60, 64), (69, 72, 76)),   # F2 | C3 F3 A3 C4 E4 | A4 C5 E5
    ("Cmaj7/E", 40, (47, 52, 55, 60, 64), (67, 71, 76)),   # E2 | B2 E3 G3 C4 E4 | G4 B4 E5
    ("Dm9",     38, (48, 53, 57, 62, 64), (65, 69, 76)),   # D2 | C3 F3 A3 D4 E4 | F4 A4 E5
]


def mtof(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ---------------------------------------------------------------- envelopes

def gate_env(t: np.ndarray, events: list[tuple[float, float]], tau_up: float,
             tau_down: float, v0: float) -> np.ndarray:
    """One-pole smoothed gate, evaluated analytically. events: sorted (time, target)."""
    env = np.full(t.size, v0)
    v = v0
    for i, (te, target) in enumerate(events):
        t_end = events[i + 1][0] if i + 1 < len(events) else np.inf
        tau = tau_up if target > v else tau_down
        m = (t >= te) & (t < t_end)
        env[m] = target + (v - target) * np.exp(-(t[m] - te) / tau)
        if np.isfinite(t_end):
            v = target + (v - target) * np.exp(-(t_end - te) / tau)
    return env


def segments(changes: list[float], dur_total: float, intro_idx: int):
    """[(start, end, chord_idx)] covering [0, dur_total)."""
    segs = []
    bounds = [0.0] + list(changes) + [dur_total]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        idx = intro_idx if k == 0 else (k - 1) % len(PROG)
        if b > a:
            segs.append((a, b, idx))
    return segs


def pitch_events(segs, pitch_of) -> dict[int, tuple[float, list[tuple[float, float]]]]:
    """Per pitch: (initial value, [(time, target)]) from the segment list."""
    pitches = sorted({p for _, _, c in segs for p in pitch_of(c)})
    out = {}
    for p in pitches:
        ev = []
        for a, _, c in segs:
            ev.append((a, 1.0 if p in pitch_of(c) else 0.0))
        v0 = ev[0][1]
        out[p] = (v0, ev)
    return out


# ---------------------------------------------------------------- voices

def bass(t: np.ndarray, segs) -> np.ndarray:
    y = np.zeros(t.size)
    for p, (v0, ev) in pitch_events(segs, lambda c: (PROG[c][1],)).items():
        f = mtof(p)
        ph = 2 * np.pi * f * t
        osc = np.sin(ph) + 0.18 * np.sin(2 * ph + 0.4) + 0.03 * np.sin(3 * ph + 1.1)
        ev = [(max(0.0, te - 0.02), g) for te, g in ev]
        y += osc * gate_env(t, ev, 0.07, 0.25, v0)
    # slow breathing so the sustain is not static
    y *= 0.92 + 0.08 * np.sin(2 * np.pi * 0.11 * t)
    return np.stack([y, y])


def pad(t: np.ndarray, segs, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((2, t.size))
    detune = [(-7.0, 0.75, 0.35, 0.23), (0.0, 0.6, 0.6, 0.0), (7.0, 0.35, 0.75, 0.29)]
    for p, (v0, ev) in pitch_events(segs, lambda c: PROG[c][2]).items():
        f = mtof(p)
        nh = max(1, int(3000.0 / f))
        n = np.arange(1, nh + 1)
        amp = 0.6 / n + np.where(n % 2 == 1, 0.4 * (8 / np.pi ** 2) / n ** 2, 0.0)
        hph = rng.uniform(0, 2 * np.pi, (3, nh))
        voice = np.zeros((3, t.size))
        for v, (cents, _, _, lfo_r) in enumerate(detune):
            ratio = 2 ** (cents / 1200)
            if lfo_r > 0:   # chorus: +-3 cent vibrato on the detuned voices
                d = 2 ** (3 / 1200) - 1
                base = 2 * np.pi * f * ratio * t - (f * ratio * d / lfo_r) * np.cos(
                    2 * np.pi * lfo_r * t + hph[v, 0])
            else:
                base = 2 * np.pi * f * t
            acc = np.zeros(t.size)
            for k in range(nh):
                acc += amp[k] * np.sin((k + 1) * base + hph[v, k])
            voice[v] = acc
        g = gate_env(t, ev, 0.15, 0.6, v0)
        for v, (_, gl, gr, _) in enumerate(detune):
            out[0] += gl * g * voice[v]
            out[1] += gr * g * voice[v]
    sos = signal.butter(2, 1500, "low", fs=SR, output="sos")
    return signal.sosfilt(sos, out, axis=1)


def epiano_note(f: float, length: int, vel: float) -> np.ndarray:
    tt = np.arange(length) / SR
    index = 1.1 * np.exp(-tt / 0.12) + 0.12
    y = np.sin(2 * np.pi * f * tt + index * np.sin(2 * np.pi * f * tt))
    y += 0.25 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.25)   # soft tine bloom
    env = np.minimum(tt / 0.006, 1.0) * np.exp(-tt / 1.3)
    return vel * y * env


def top_line(t: np.ndarray, segs, rng: np.random.Generator, skip_intro: bool) -> np.ndarray:
    out = np.zeros((2, t.size))
    pans = (-0.35, 0.0, 0.35)
    for si, (a, b, c) in enumerate(segs):
        if si == 0 and skip_intro:   # the pickup before the first change has no top line
            continue
        notes = PROG[c][3]
        hits = [(a + i * BEAT / 2, notes[i], (0.85, 0.6, 0.55)[i], pans[i]) for i in range(3)]
        if b - a > 2.4:   # a softer echo of the top two notes later in a long segment
            hits += [(a + 3 * BEAT, notes[1], 0.32, 0.2), (a + 3.5 * BEAT, notes[2], 0.28, -0.2)]
        for th, m, vel, pan in hits:
            th += rng.uniform(-0.006, 0.006) if th > a else 0.0
            if th >= t[-1]:
                continue
            i0 = int(round(th * SR))
            ring = min(t.size - i0, int((b - th + 0.5) * SR) if b < t[-1] else t.size - i0)
            if ring <= 0:
                continue
            y = epiano_note(mtof(m), ring, vel * rng.uniform(0.93, 1.0))
            rel = min(ring, int(0.4 * SR))
            y[-rel:] *= np.linspace(1, 0, rel) ** 2
            out[0, i0:i0 + ring] += y * np.sqrt(0.5 * (1 - pan))
            out[1, i0:i0 + ring] += y * np.sqrt(0.5 * (1 + pan))
    # gentle stereo tremolo (classic EP auto-pan), very shallow
    trem = 0.06 * np.sin(2 * np.pi * 4.2 * t)
    out[0] *= 1 + trem
    out[1] *= 1 - trem
    sos = signal.butter(2, 3500, "low", fs=SR, output="sos")
    return signal.sosfilt(sos, out, axis=1)


# ---------------------------------------------------------------- room

def room_ir(rng: np.random.Generator, rt60: float = 2.0, length: float = 2.6) -> np.ndarray:
    n = int(length * SR)
    tt = np.arange(n) / SR
    noise = rng.standard_normal((2, n))
    sos = signal.butter(2, 3000, "low", fs=SR, output="sos")
    noise = signal.sosfilt(sos, noise, axis=1)
    env = np.exp(-6.91 * tt / rt60) * np.minimum(tt / 0.02, 1.0)
    ir = noise * env
    pre = int(0.015 * SR)
    ir = np.concatenate([np.zeros((2, pre)), ir], axis=1)
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


def reverb(x: np.ndarray, ir: np.ndarray, circular: bool) -> np.ndarray:
    n = x.shape[1]
    wet = np.stack([signal.fftconvolve(x[ch], ir[ch]) for ch in range(2)])
    if circular:   # fold the tail back onto the start: exact circular convolution
        folded = np.zeros((2, n))
        for s in range(0, wet.shape[1], n):
            seg = wet[:, s:s + n]
            folded[:, :seg.shape[1]] += seg
        return folded
    return wet[:, :n]


# ---------------------------------------------------------------- loudness

def ebur128(x: np.ndarray) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) via ffmpeg."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        wavfile.write(tmp, SR, x.T.astype(np.float32))
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", tmp, "-af",
                            "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
        i = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
        p = re.findall(r"Peak:\s+(-?[\d.inf]+)\s+dBFS", r.stderr)
        if not i or not p:
            raise RuntimeError("ebur128 measurement failed:\n" + r.stderr[-2000:])
        return float(i[-1]), float(p[-1])
    finally:
        os.remove(tmp)


def true_peak_db(x: np.ndarray) -> float:
    up = signal.resample_poly(x, 4, 1, axis=1)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def limit(x: np.ndarray, ceiling_db: float, circular: bool) -> np.ndarray:
    thr = 10 ** (ceiling_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=1)
    pk = np.abs(up).max(axis=0).reshape(-1, 4).max(axis=1)[:x.shape[1]]
    g = np.minimum(1.0, thr / np.maximum(pk, 1e-12))
    w = int(0.008 * SR)
    mode = "wrap" if circular else "nearest"
    g = minimum_filter1d(g, 2 * w + 1, mode=mode)
    g = uniform_filter1d(g, 2 * w + 1, mode=mode)
    return x * g


def loudness(x: np.ndarray, circular: bool) -> tuple[np.ndarray, float, float, int]:
    rms = np.sqrt(np.mean(x ** 2))
    x = x * (10 ** (TARGET_LUFS / 20) / rms)
    passes = 0
    for passes in range(1, 6):
        L, _ = ebur128(x)
        x = x * 10 ** ((TARGET_LUFS - L) / 20)
        if true_peak_db(x) <= TP_LIMIT - TP_MARGIN:
            break
        x = limit(x, TP_LIMIT - TP_MARGIN - 0.2, circular)
    L, tp = ebur128(x)
    return x, L, tp, passes


# ---------------------------------------------------------------- main

def build(duration: float, changes: list[float], seed: int, loop: bool):
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    xf = int(round(min(1.0, duration / 4) * SR)) if loop else 0
    tail = xf
    t = np.arange(n + tail) / SR
    intro_idx = ((len(changes) - 1) % len(PROG)) if (loop and changes) else len(PROG) - 1
    segs = segments(changes, (n + tail) / SR, intro_idx)
    skip_intro = bool(changes) and changes[0] > 0

    stems = {"bass": bass(t, segs), "pad": pad(t, segs, rng),
             "ep": top_line(t, segs, rng, skip_intro)}
    for k in stems:   # equalise stems, then weight
        stems[k] = stems[k] / (np.sqrt(np.mean(stems[k] ** 2)) + 1e-12)
    w = {"bass": 0.75, "pad": 1.0, "ep": 0.55}
    sends = {"bass": 0.08, "pad": 0.35, "ep": 0.45}
    dry = sum(w[k] * stems[k] for k in stems)
    send = sum(w[k] * sends[k] * stems[k] for k in stems)

    if loop:   # crossfade the natural continuation past the end into the start
        fade = np.sin(0.5 * np.pi * np.arange(xf) / xf)
        def wrap(y):
            y2 = y[:, :n].copy()
            y2[:, :xf] = y[:, :xf] * fade + y[:, n:n + xf] * np.sqrt(1 - fade ** 2)
            return y2
        dry, send = wrap(dry), wrap(send)

    ir = room_ir(rng)
    x = dry + 0.9 * reverb(send, ir, circular=loop)
    sos = signal.butter(2, 30, "high", fs=SR, output="sos")
    if loop:   # filter a tiled copy so the high-pass is seamless too
        x = signal.sosfilt(sos, np.concatenate([x, x], axis=1), axis=1)[:, n:]
    else:
        x = signal.sosfilt(sos, x, axis=1)
        tt = np.arange(n) / SR
        fin = 0.5 - 0.5 * np.cos(np.pi * np.clip(tt / 0.3, 0, 1))
        fout = 0.5 - 0.5 * np.cos(np.pi * np.clip((duration - tt) / 0.8, 0, 1))
        x = x * fin * fout
    return x, segs


def main() -> None:
    ap = argparse.ArgumentParser(description="Moody low chord soundtrack")
    ap.add_argument("--duration", type=float, required=True, help="length in s")
    ap.add_argument("--changes", default="", help="comma-separated chord-change times in s")
    ap.add_argument("--out", required=True, help="output wav")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loop", action="store_true", help="crossfade the end into the start")
    a = ap.parse_args()

    if a.changes.strip():
        changes = sorted(float(s) for s in a.changes.split(",") if s.strip())
    else:   # one change per 4-beat bar at 96 bpm
        changes = list(np.arange(0.0, a.duration, 4 * BEAT))
    changes = [c for c in changes if 0.0 <= c < a.duration]

    x, segs = build(a.duration, changes, a.seed, a.loop)
    x, L, tp, passes = loudness(x, a.loop)
    if not a.loop:
        x[:, -1] = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wavfile.write(a.out, SR, (np.clip(x, -1, 1).T * 32767).round().astype(np.int16))

    print(f"wrote {a.out}: {x.shape[1]} samples ({x.shape[1] / SR:.3f} s), 2 ch, {SR} Hz, "
          f"loop={a.loop}, seed={a.seed}")
    print(f"loudness {L:.2f} LUFS (ffmpeg ebur128), true peak {tp:.2f} dBTP, "
          f"sample peak {20 * np.log10(np.abs(x).max()):.2f} dBFS, gain passes {passes}")
    for s, e, c in segs:
        if s < a.duration:
            print(f"  {s:8.4f} - {min(e, a.duration):8.4f} s  {PROG[c][0]}")


if __name__ == "__main__":
    main()
