"""Low, moody beat for the audio-reactive silk wind (48 kHz stereo, -14 LUFS, <= -1 dBTP).

Same world as chords.py (A minor / D dorian, Am9 -> Fmaj7 -> Cmaj7/E -> Dm9, one chord per bar),
with drums and a sub line on top of chords.py's pad (imported, chords.py is not edited):
- kick:  pitch-swept sine (160 -> 52 Hz body), soft 2 ms attack, ~0.3 s decay, faint low click;
- snare: dark clap (3 short noise bursts + band-passed noise tail) over a 185 Hz body, low-passed 5 kHz;
- hats:  muted closed hats, high-passed noise, ~22 ms decay, quiet: the only high content;
- bass:  saturated sine following the chord roots (55-110 Hz), locked to the kick, low-passed 300 Hz;
- pad:   chords.pad() voicings, side-chained by the kick.
Pattern: 16 steps per bar with a variation bar; in the second half of the bar before the last the
kick, hats and bass drop out, the pad's filter closes, a snare roll and a low noise swell build, and
everything returns on the last bar's downbeat (the "burst").

--events-json writes the exact hit times (ground truth for the visuals): kick / snare / hat
{t, v}, bass notes, chord changes, downbeats, the drop window and the burst time.

Usage: python tools/mathtype/beat.py --duration 12 --bpm 100 --out x.wav --events-json x.json [--seed 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chords as C  # noqa: E402  (pad, room, loudness; not edited)

SR = C.SR

# 16-step patterns {step: velocity}
KICK_A = {0: 1.0, 6: 0.75, 10: 0.9}
KICK_B = {0: 1.0, 3: 0.55, 10: 0.9}
SNARE_A = {4: 1.0, 12: 1.0}
SNARE_B = {4: 1.0, 12: 1.0, 15: 0.3}
HATS = {s: (0.55 if s % 4 == 0 else 0.9) for s in range(0, 16, 2)}
HATS_B = {**HATS, 7: 0.35, 15: 0.4}
FILL_SNARE = {10: 0.3, 12: 0.42, 13: 0.55, 14: 0.72, 15: 0.9}   # snare roll into the last bar


def fifth(root: int) -> int:
    return root + 7 if root + 7 <= 45 else root - 5


# ---------------------------------------------------------------- one-shots

def kick_hit(v: float) -> np.ndarray:
    tt = np.arange(int(0.7 * SR)) / SR
    f = 52.0 + (160.0 - 52.0) * np.exp(-tt / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(tt / 0.002, 1.0) * np.exp(-tt / 0.30)
    y = np.sin(ph) * env
    y = np.tanh(1.6 * y) / np.tanh(1.6)
    rng = np.random.default_rng(1)
    click = signal.sosfilt(signal.butter(2, 1800, "low", fs=SR, output="sos"), rng.standard_normal(tt.size))
    y += 0.10 * click * np.exp(-tt / 0.004)
    return v * y


def snare_hit(v: float, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(int(0.45 * SR)) / SR
    f = 180.0 + 40.0 * np.exp(-tt / 0.02)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.07) * np.minimum(tt / 0.001, 1)
    nz = signal.sosfilt(signal.butter(2, [400, 5000], "band", fs=SR, output="sos"), rng.standard_normal(tt.size))
    clap = sum(np.exp(-np.maximum(tt - d, 0) / 0.006) * (tt >= d) for d in (0.0, 0.011, 0.022))
    tail = np.exp(-np.maximum(tt - 0.022, 0) / 0.15) * (tt >= 0.022)
    y = 0.55 * body + 0.8 * nz * (0.6 * clap + tail) / 1.6
    y = signal.sosfilt(signal.butter(4, 5000, "low", fs=SR, output="sos"), y)
    return v * y


def hat_hit(v: float, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(int(0.09 * SR)) / SR
    nz = rng.standard_normal(tt.size)
    nz = signal.sosfilt(signal.butter(4, 7000, "high", fs=SR, output="sos"), nz)
    nz = signal.sosfilt(signal.butter(2, 11000, "low", fs=SR, output="sos"), nz)
    return v * nz * np.exp(-tt / 0.022) * np.minimum(tt / 0.0005, 1)


def bass_note(midi: int, length: float, v: float) -> np.ndarray:
    n = int((length + 0.08) * SR)
    tt = np.arange(n) / SR
    f = C.mtof(midi)
    y = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt + 0.3)
    y = np.tanh(1.8 * y) / np.tanh(1.8)
    env = np.minimum(tt / 0.006, 1.0) * np.where(tt < length, 1.0, np.exp(-(tt - length) / 0.03))
    return v * y * env


def place(buf: np.ndarray, x: np.ndarray, t: float) -> None:
    i0 = int(round(t * SR))
    if i0 >= buf.size:
        return
    m = min(x.size, buf.size - i0)
    buf[i0:i0 + m] += x[:m]


# ---------------------------------------------------------------- arrangement

def arrange(duration: float, bpm: float):
    bar = 240.0 / bpm
    step = bar / 16
    nbars = int(np.ceil(duration / bar - 1e-9))
    last = int(np.floor(duration / bar - 1e-9))
    if duration - last * bar < 0.5 * bar:
        last -= 1
    ev = dict(kick=[], snare=[], hat=[], bass=[], chords=[], downbeats=[])
    for b in range(nbars):
        t0 = b * bar
        c = b % len(C.PROG)
        ev["chords"].append(dict(t=round(t0, 6), name=C.PROG[c][0], idx=c))
        ev["downbeats"].append(round(t0, 6))
        pre = b == last - 1
        kick = KICK_B if b % 2 == 1 else KICK_A
        snare = SNARE_B if b % 2 == 1 else SNARE_A
        hats = HATS_B if b % 2 == 1 else HATS
        if b == last:
            kick = {**kick, 0: 1.15}
        if pre:   # drop: second half of the bar loses kick, hats and bass; snare roll fill
            kick = {s: v for s, v in kick.items() if s < 8}
            hats = {s: v for s, v in hats.items() if s < 8}
            snare = {**{s: v for s, v in snare.items() if s < 8}, **FILL_SNARE}
        root = C.PROG[c][1]
        bassline = {0: (root, 5, 1.0), 6: (root, 3, 0.8), 10: (fifth(root), 4, 0.85), 14: (root, 1.5, 0.6)}
        if b % 2 == 1:
            bassline = {0: (root, 3, 1.0), 3: (root, 2, 0.7), 10: (fifth(root), 4, 0.85), 14: (root + 12, 1.5, 0.5)}
        if pre:
            bassline = {s: x for s, x in bassline.items() if s < 8}
        for name, pat in (("kick", kick), ("snare", snare), ("hat", hats)):
            for s, v in sorted(pat.items()):
                t = t0 + s * step
                if t < duration - 1e-9:
                    ev[name].append(dict(t=round(t, 6), v=v))
        for s, (m, ln, v) in sorted(bassline.items()):
            t = t0 + s * step
            if t < duration - 1e-9:
                ev["bass"].append(dict(t=round(t, 6), midi=m, len=round(ln * step, 6), v=v))
    burst = last * bar
    ev.update(bpm=bpm, duration=duration, bar=bar, step=step, drop=[round(burst - bar / 2, 6), round(burst, 6)],
              burst=round(burst, 6))
    return ev


def build(duration: float, bpm: float, seed: int):
    rng = np.random.default_rng(seed)
    ev = arrange(duration, bpm)
    n = int(round(duration * SR))
    t = np.arange(n) / SR
    stems = {k: np.zeros(n) for k in ("kick", "snare", "hat", "bass", "riser")}
    for e in ev["kick"]:
        place(stems["kick"], kick_hit(e["v"]), e["t"])
    for e in ev["snare"]:
        place(stems["snare"], snare_hit(e["v"], rng), e["t"])
    for e in ev["hat"]:
        place(stems["hat"], hat_hit(e["v"] * rng.uniform(0.85, 1.0), rng), e["t"])
    for e in ev["bass"]:
        place(stems["bass"], bass_note(e["midi"], e["len"], e["v"]), e["t"])
    stems["bass"] = signal.sosfilt(signal.butter(2, 300, "low", fs=SR, output="sos"), stems["bass"])

    # side-chain: pad and bass duck under each kick (10 ms attack, 180 ms recovery)
    duck = np.zeros(n)
    for e in ev["kick"]:
        d = t - e["t"]
        m = d >= 0
        duck[m] += e["v"] * np.exp(-d[m] / 0.18) * np.minimum(d[m] / 0.01, 1)
    duck = 1 - 0.45 * np.clip(duck, 0, 1)

    # pad: chords.py voicings, one chord per bar
    changes = [c["t"] for c in ev["chords"]]
    segs = C.segments(changes, duration, len(C.PROG) - 1)
    pad = C.pad(t, segs, rng)
    d0, d1 = ev["drop"]
    w = np.clip((t - d0) / (d1 - d0 - 0.15), 0, 1) ** 1.5 * (t < d1)          # filter closes over the drop
    w = w * (1 - np.clip((t - d1) / 0.01, 0, 1))
    pad_lp = signal.sosfilt(signal.butter(2, 350, "low", fs=SR, output="sos"), pad, axis=1)
    pad = pad * (1 - w) + pad_lp * w * 1.4
    pad = pad * duck

    # low noise swell into the burst
    nz = signal.sosfilt(signal.butter(2, 1200, "low", fs=SR, output="sos"), rng.standard_normal(n))
    stems["riser"] = nz * np.clip((t - d0) / (d1 - d0), 0, 1) ** 3 * (t < d1)

    for k in stems:
        stems[k] = stems[k] / (np.abs(stems[k]).max() + 1e-12)
    pad = pad / (np.abs(pad).max() + 1e-12)
    stems["bass"] *= duck
    wts = dict(kick=0.85, snare=0.9, hat=0.2, bass=0.45, riser=0.12)
    PADW = 0.6
    sends = dict(kick=0.03, snare=0.3, hat=0.15, bass=0.0, riser=0.4)
    mono = sum(wts[k] * stems[k] for k in stems)
    pan_hat = 0.25
    dry = np.stack([mono, mono]) + PADW * pad
    dry[0] += wts["hat"] * stems["hat"] * (np.sqrt(0.5 * (1 - pan_hat)) - np.sqrt(0.5)) * np.sqrt(2)
    dry[1] += wts["hat"] * stems["hat"] * (np.sqrt(0.5 * (1 + pan_hat)) - np.sqrt(0.5)) * np.sqrt(2)
    send = np.stack([sum(wts[k] * sends[k] * stems[k] for k in stems)] * 2) + PADW * 0.35 * pad
    ir = C.room_ir(rng, rt60=1.6)
    x = dry + 0.8 * C.reverb(send, ir, circular=False)
    x = signal.sosfilt(signal.butter(2, 30, "high", fs=SR, output="sos"), x, axis=1)
    fin = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / 0.005, 0, 1))
    fout = 0.5 - 0.5 * np.cos(np.pi * np.clip((duration - t) / 0.8, 0, 1))
    x = x * fin * fout
    levels = {k: 20 * np.log10(np.sqrt(np.mean((wts[k] * stems[k]) ** 2)) + 1e-12) for k in stems}
    levels["pad"] = 20 * np.log10(np.sqrt(np.mean((PADW * pad) ** 2)) + 1e-12)
    return x, ev, levels


def main() -> None:
    ap = argparse.ArgumentParser(description="Low moody beat + events")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events-json", help="write exact hit times per instrument here")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    for p in [a.out] + ([a.events_json] if a.events_json else []):
        if os.path.exists(p):
            sys.exit(f"{p} exists; not overwriting")
    x, ev, levels = build(a.duration, a.bpm, a.seed)
    x, L, tp, passes = C.loudness(x, False)
    x[:, -1] = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wavfile.write(a.out, SR, (np.clip(x, -1, 1).T * 32767).round().astype(np.int16))
    if a.events_json:
        ev["wav"] = os.path.basename(a.out)
        with open(a.events_json, "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=1)
    print(f"wrote {a.out}: {x.shape[1] / SR:.3f} s, {a.bpm} bpm, seed {a.seed}")
    print(f"loudness {L:.2f} LUFS (ffmpeg ebur128), true peak {tp:.2f} dBTP, gain passes {passes}")
    print("stem RMS (pre-master, dBFS): " + ", ".join(f"{k} {v:.1f}" for k, v in levels.items()))
    print(f"kicks {len(ev['kick'])}, snares {len(ev['snare'])}, hats {len(ev['hat'])}, "
          f"chords {[c['name'] for c in ev['chords']]}, drop {ev['drop']}, burst {ev['burst']}")


if __name__ == "__main__":
    main()
