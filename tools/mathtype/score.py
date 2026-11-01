"""Short scores for the 11_mathtype pieces (8-10 s): deterministic CPU synth, no bleeps.

48 kHz stereo wav, -14 LUFS integrated, true peak <= -1 dBTP, last 0.6 s faded out.
Every --changes time is a musical arrival (a chord change or a struck note) that lands on a visual
resolve. Loudness, true-peak limiting and the room reverb are reused from chords.py (not edited).

Instruments (physical-ish, all low-passed; the only content above 2 kHz is soft harmonics):
- felt piano: additive partials with string inharmonicity (f_k = k f sqrt(1 + B k^2)), two-stage decay
  (fast drop into a sustain tail), soft 3.5 ms felt attack, hammer thump, velocity-tracking low-pass;
  a muted variant (short tail, darker) for pulses;
- bowed string / cello: band-limited sawtooth (additive) with slow bow attack, vibrato fading in after
  0.5 s, bow noise, then a body filter (resonators at ~205/290/440/600/800 Hz);
- low drone: sine + soft 2nd and 3rd harmonic, optionally as detuned beating pairs (E4);
- breath / air: band-passed noise with slow swells;
- wood mallet: modes at 1 : 3.93 : 9.2, short, low register (E3 only);
- sub swell: 35-55 Hz sine + 2nd harmonic, rising into a change and decaying after it.

Presets (one per sentence):
- E1 contemplative discovery: D dorian, felt-piano rising arpeggio figures over a cello D pedal,
  crescendo into each change; Dm9 -> G6/D -> C/D -> Dm6/9.
- E2 travel and longing: bowed-string pad of rising suspensions (sus2, then the 3rd enters: add9),
  soft muted-piano eighths, a rising top line; Gsus2 -> Em(add9) -> Csus2(add9) -> lands on Gmaj7.
- E3 structure and wonder: low wood-mallet ostinato in 7 (2+2+3), cello line, piano chords stacked in
  fifths, one more fifth at each change: C-G -> C-G-D -> C-G-D-A -> C-G-D-A-E.
- E4 noise settling into pattern: filtered air + beating detuned drones whose detune and pitch error
  shrink at each change, noise narrowing onto the chord's partials; ends on pure D major, piano only
  at the last change.
- E5 additive growth: a felt-piano cell (F pentatonic, 8 slots) that gains one note per cycle, a
  slightly faster phasing copy, cello roots, breath texture; blooms (string Fmaj9, sub) at the last change.

Usage:
  python tools/mathtype/score.py --preset E1 --duration 9 --changes 2.5,5.0,7.5 --out x.wav [--seed 11]
      [--backend torch --device cuda] [--timings 11_mathtype/v007/local/TIMINGS.csv]
Refuses to overwrite an existing --out unless --force is given. --backend torch runs all synthesis in
float64 on the device (numpy stays the reference; the two agree to better than -80 dB).
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chords as C  # noqa: E402  (room_ir, reverb, loudness, gate_env; not edited)

SR = C.SR
mtof = C.mtof
TAU = 2 * np.pi
FADE_OUT = 0.6


# ---------------------------------------------------------------- backend
# numpy (default, the reference) or torch float64 on cpu/cuda. All synthesis maths goes through K.
# float64 on the GPU too: additive phases reach ~3e5 rad in 9 s, which float32 cannot hold to -80 dB.
# On torch every IIR filter becomes an FFT convolution with its impulse response, which is synthesised
# on the device in closed form from the filter's poles and residues (h[n] = sum_i r_i p_i^n) and cut
# where it has decayed below 1e-14; zero initial state, so it is the same causal sum sosfilt computes.
# Still on the CPU: the seeded numpy RNG draws (so both backends get identical noise), filter design
# (butter / iirpeak / residuez), chords.gate_env for E4's order curve, and the loudness pass (ffmpeg).

class _NumpyNS:
    name = "numpy"
    torch = False

    def __getattr__(self, k):            # sin, cos, exp, sqrt, where, clip, maximum, stack, zeros ...
        return getattr(np, k)

    def asarray(self, x):
        return np.asarray(x, dtype=float)

    def arange(self, n):
        return np.arange(n, dtype=float)

    def cumsum(self, x):
        return np.cumsum(x, axis=-1)

    def tonp(self, x):
        return x

    def sync(self):
        pass


class _TorchNS:
    torch = True

    def __init__(self, device):
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("--device cuda: torch has no CUDA here (torch " + torch.__version__ + ")")
        self.T, self.dev, self.dt = torch, torch.device(device), torch.float64
        self.name = f"torch-{device}"
        # small tensors (single notes): torch's intra-op thread pool only adds contention on a shared
        # machine (measured 36 ms vs 0.46 ms per exp*sin on 67k samples, 8 threads vs 1)
        torch.set_num_threads(max(1, int(os.environ.get("SCORE_TORCH_THREADS", "1"))))

    def _t(self, x):
        return x if isinstance(x, self.T.Tensor) else self.T.tensor(float(x), dtype=self.dt, device=self.dev)

    def asarray(self, x):
        if isinstance(x, self.T.Tensor):
            return x.to(self.dev, self.dt)
        return self.T.as_tensor(np.asarray(x, dtype=float), device=self.dev)

    def arange(self, n):
        return self.T.arange(n, dtype=self.dt, device=self.dev)

    def zeros(self, shape):
        return self.T.zeros(shape, dtype=self.dt, device=self.dev)

    def sin(self, x):
        return self.T.sin(x)

    def cos(self, x):
        return self.T.cos(x)

    def exp(self, x):
        return self.T.exp(x)

    def sqrt(self, x):
        return self.T.sqrt(x)

    def where(self, c, a, b):
        return self.T.where(c, self._t(a), self._t(b))

    def clip(self, x, lo, hi):
        return self.T.clamp(x, lo, hi)

    def maximum(self, a, b):
        return self.T.maximum(self._t(a), self._t(b))

    def stack(self, xs):
        return self.T.stack(list(xs))

    def cumsum(self, x):
        return self.T.cumsum(x, dim=-1)

    def mean(self, x):
        return self.T.mean(x)

    def tonp(self, x):
        return x.detach().cpu().numpy()

    def sync(self):
        if self.dev.type == "cuda":
            self.T.cuda.synchronize()


K = _NumpyNS()


def set_backend(name="numpy", device="cpu"):
    global K
    if name == "numpy":
        if device != "cpu":
            raise SystemExit("--device cuda needs --backend torch")
        K = _NumpyNS()
    elif name == "torch":
        K = _TorchNS(device)
    else:
        raise SystemExit(f"backend {name!r}: numpy | torch")
    _IR.clear()
    return K


_IR: dict = {}


def _ir_t(b, a):
    """Impulse response of b/a on the device, closed form from poles and residues, cut at 1e-14."""
    key = (tuple(np.round(b, 15)), tuple(np.round(a, 15)))
    h = _IR.get(key)
    if h is None:
        T = K.T
        r, p, k = signal.residuez(b, a)
        rho = float(np.abs(p).max())
        m = int(np.ceil(np.log(1e-14) / np.log(rho))) + 1 if rho < 1 else 8 * SR
        m = max(m, len(k), 1)
        nn = T.arange(m, dtype=T.float64, device=K.dev)
        pt = T.as_tensor(np.ascontiguousarray(p, dtype=complex), device=K.dev)
        rt = T.as_tensor(np.ascontiguousarray(r, dtype=complex), device=K.dev)
        h = (rt[:, None] * T.exp(nn[None, :] * T.log(pt)[:, None])).sum(0).real.contiguous()
        if len(k):
            h[:len(k)] += T.tensor([float(v) for v in np.ravel(k)], dtype=T.float64, device=K.dev)
        if len(_IR) > 64:
            _IR.clear()
        _IR[key] = h
    return h


def fftconv(x, h):
    """Causal linear convolution along the last axis, truncated to x's length (torch tensors)."""
    T = K.T
    n, m = x.shape[-1], h.shape[-1]
    L = sfft.next_fast_len(n + min(m, n) - 1, real=True)
    y = T.fft.irfft(T.fft.rfft(x, n=L) * T.fft.rfft(h[:n], n=L), n=L)
    return y[..., :n].contiguous()


def filt_ba(x, b, a):
    if not K.torch:
        return signal.lfilter(b, a, x, axis=-1)
    return fftconv(x, _ir_t(np.asarray(b, float), np.asarray(a, float)))


def filt_sos(x, sos):
    if not K.torch:
        return signal.sosfilt(sos, x, axis=-1)
    b, a = signal.sos2tf(sos)
    return filt_ba(x, b, a)


def lp(x, fc, order=2):
    return filt_sos(x, signal.butter(order, min(fc, 0.45 * SR), "low", fs=SR, output="sos"))


def hp(x, fc, order=2):
    return filt_sos(x, signal.butter(order, fc, "high", fs=SR, output="sos"))


def bp(x, lo, hi, order=2):
    return filt_sos(x, signal.butter(order, [lo, hi], "band", fs=SR, output="sos"))


def noise(rng, shape):
    """Seeded numpy draw (identical on both backends), moved to the backend."""
    return K.asarray(rng.standard_normal(shape))


# ---------------------------------------------------------------- dsp helpers

def place(bus, y, t0, pan=0.0, gain=1.0):
    """Add mono (equal-power panned) or stereo y into bus (2, n) at time t0."""
    i0 = int(round(t0 * SR))
    if y.ndim == 1:
        y = K.stack([y * math.sqrt(0.5 * (1 - pan)), y * math.sqrt(0.5 * (1 + pan))])
    if i0 < 0:
        y, i0 = y[:, -i0:], 0
    if i0 >= bus.shape[1]:
        return
    m = min(y.shape[1], bus.shape[1] - i0)
    bus[:, i0:i0 + m] += gain * y[:, :m]


def ramp_env(t, t0, rise):
    """0 before t0, raised-cosine rise over `rise` s."""
    return 0.5 - 0.5 * K.cos(np.pi * K.clip((t - t0) / rise, 0, 1))


def active_rms(x):
    """RMS over the 50 ms frames within 30 dB of the loudest frame (so sparse stems compare fairly)."""
    x = K.tonp(x)
    m = x.mean(axis=0) if x.ndim == 2 else x
    hop = int(0.05 * SR)
    k = m.size // hop
    if k == 0:
        return np.sqrt(np.mean(m ** 2)) + 1e-12
    fr = np.sqrt(np.mean(m[:k * hop].reshape(k, hop) ** 2, axis=1))
    if fr.max() <= 0:
        return 1e-12
    act = fr[fr > fr.max() * 10 ** (-30 / 20)]
    return np.sqrt(np.mean(act ** 2)) + 1e-12


# ---------------------------------------------------------------- instruments

def piano(f, hold, vel, rng, muted=False):
    """Felt piano note, mono. hold = time until the damper falls."""
    rel = 0.07 if muted else 0.32
    n = int((hold + 5 * rel + 0.02) * SR)
    tt = K.arange(n) / SR
    B = 0.00035
    kmax = 6 if muted else 14
    tilt = 1.45 - 0.55 * vel            # softer notes have steeper spectral tilt
    ph = rng.uniform(0, TAU, kmax)
    beat = rng.uniform(0.25, 0.9, kmax)
    ks = np.arange(1, kmax + 1, dtype=float)
    fk = ks * f * np.sqrt(1 + B * ks * ks)                           # stiff-string inharmonicity
    keep = fk <= 5000
    ks, fk, ph, beat = ks[keep], fk[keep], ph[keep], beat[keep]
    tf = 0.05 + 0.22 / ks ** 0.7                                     # hammer transient stage
    ts = (0.2 if muted else 3.2) * (196.0 / f) ** 0.35 / (1 + 0.3 * (ks - 1))   # sustain tail
    col = lambda v: K.asarray(v)[:, None]   # noqa: E731  (partials x time, all at once)
    T_ = tt[None, :]
    env = 0.45 * K.exp(-T_ / col(tf)) + 0.55 * K.exp(-T_ / col(ts))
    # two/three-string unison: slow amplitude beating, stronger on upper partials
    env = env * (1 + col(np.minimum(0.04 * ks, 0.25)) * K.sin(TAU * col(beat) * T_ + col(ph)))
    y = (col(ks ** (-tilt)) * env * K.sin(TAU * col(fk) * T_ + col(ph))).sum(0)
    y = y * (1 - K.exp(-tt / 0.0035))                                # felt: no click
    nh = int(0.05 * SR)
    th = K.arange(nh) / SR
    thump = lp(noise(rng, nh), 500 + 900 * vel) * K.exp(-th / 0.009)
    y[:nh] += (0.06 + 0.06 * vel) * thump
    y = y * K.where(tt < hold, 1.0, K.exp(-(tt - hold) / rel))
    fc = (380 + 700 * vel) if muted else (900 + 3600 * vel ** 1.4)
    y = lp(y, fc)
    return vel ** 1.25 * y


def bowed(f, length, rng, attack=0.6, release=0.8, vib_cents=7.0, nh_max=30, swell=None):
    """Raw bowed-string source (sawtooth + bow noise), mono; run the bus through body()."""
    n = int((length + 3 * release + 0.05) * SR)
    tt = K.arange(n) / SR
    vd = K.clip((tt - 0.5) / 0.9, 0, 1)                              # vibrato fades in after 0.5 s
    vr = 4.9 + rng.uniform(-0.3, 0.3)
    p0, p1 = rng.uniform(0, TAU, 2)
    cents = vib_cents * vd * K.sin(TAU * vr * tt + p0) + 2.0 * K.sin(TAU * 0.21 * tt + p1)
    phase = TAU * K.cumsum(f * 2 ** (cents / 1200)) / SR + rng.uniform(0, TAU)
    nh = max(1, min(nh_max, int(3200 / f)))
    y = K.zeros(n)
    for k in range(1, nh + 1):
        y = y + K.sin(k * phase) / k
    y = y + 0.03 * bp(noise(rng, n), 700, 2200)                      # rosin / bow hair
    env = ramp_env(tt, 0, attack) * (1 + 0.07 * K.sin(TAU * 0.27 * tt + p1))
    env = env * K.where(tt < length, 1.0, K.exp(-(tt - length) * 2.3 / release))
    if swell is not None:
        env = env * swell(tt)
    return y * env


def body(x):
    """Cello/viola body: low dry path plus resonant bandpasses 205-800 Hz, then a 2.6 kHz low-pass."""
    out = 0.35 * lp(x, 380)
    for fc, q, g in ((205, 3.0, 1.0), (290, 4.0, 0.8), (440, 4.0, 0.7), (600, 5.0, 0.5), (800, 5.0, 0.4)):
        b, a = signal.iirpeak(fc, q, fs=SR)
        out = out + g * filt_ba(x, b, a)
    return lp(out, 2600)


def drone_osc(finst, h2=0.3, h3=0.12, ph0=0.0):
    """Sine + soft 2nd/3rd harmonic following an instantaneous-frequency array."""
    ph = TAU * K.cumsum(finst) / SR + ph0
    return K.sin(ph) + h2 * K.sin(2 * ph + 0.5) + h3 * K.sin(3 * ph + 1.3)


def breath(n, rng, lo=250, hi=1500):
    """Stereo air: decorrelated band-passed noise (apply your own swell envelope)."""
    x = bp(noise(rng, (2, n)), lo, hi)
    return lp(x, 1800)


def mallet(f, vel, rng):
    """Soft wood bar: modes 1 : 3.93 : 9.2, short."""
    n = int(1.4 * SR)
    tt = K.arange(n) / SR
    modes = ((1.0, 1.0, 0.45 * (196.0 / f) ** 0.4), (3.93, 0.25 * vel, 0.09), (9.2, 0.06 * vel, 0.025))
    y = sum(a * K.exp(-tt / tau) * K.sin(TAU * f * r * tt) for r, a, tau in modes)
    y = y * (1 - K.exp(-tt / 0.0018))
    nh = int(0.03 * SR)
    y[:nh] += 0.1 * vel * lp(noise(rng, nh), 700) * K.exp(-K.arange(nh) / SR / 0.004)
    return vel * lp(y, 1000 + 1400 * vel)


def sub_swell(t, tc, f, pre=1.3, post=1.6):
    env = K.where(t < tc, K.clip((t - (tc - pre)) / pre, 0, 1) ** 2, K.exp(-(t - tc) / post))
    y = (K.sin(TAU * f * t) + 0.25 * K.sin(2 * TAU * f * t)) * env
    return K.stack([y, y])


# ---------------------------------------------------------------- form helpers

def segments(changes, dur):
    b = [0.0] + list(changes) + [dur]
    return [(b[k], b[k + 1]) for k in range(len(b) - 1)]


def pick(prog, k, nseg):
    """Segment k's entry: the last entry always closes; the others cycle if there are many changes."""
    if k == nseg - 1:
        return prog[-1]
    return prog[k % (len(prog) - 1)]


def buses(n, *names):
    return {k: K.zeros((2, n)) for k in names}


# ---------------------------------------------------------------- presets

def preset_E1(t, changes, dur, rng):
    """Contemplative discovery: D dorian, rising felt-piano figures over a cello pedal."""
    n = t.shape[0]
    B = buses(n, "piano", "cello", "sub")
    prog = [("Dm9", (50, 53, 57, 60, 64)), ("G6/D", (50, 55, 59, 62, 64)),
            ("C/D", (50, 55, 60, 64, 67)), ("Dm6/9", (50, 53, 57, 59, 64))]
    segs = segments(changes, dur)
    ns = len(segs)
    log = []
    step, period = 0.15, 0.85
    for k, (a, b) in enumerate(segs):
        name, ch = pick(prog, k, ns)
        last = k == ns - 1
        notes = list(ch) + [ch[1] + 12, ch[2] + 12]
        if k > 0:   # arrival: low D + chord, gently rolled
            v = 0.5 if not last else 0.68          # short damped chord: the figures carry the crescendo
            for j, m in enumerate((38,) + ch[:3] + ((ch[4],) if last else ())):
                place(B["piano"], piano(mtof(m), 1.0 if not last else b - a + 0.3, v * (1 - 0.05 * j), rng),
                      a + 0.018 * j, pan=-0.3 + 0.15 * j)
            log.append(f"{a:.2f}s {name}: piano chord (D2 + {name}) rolled, sub swell, cello swells to peak")
        s0 = a + (0.25 if k == 0 else 0.5)
        fi = 0
        while True:
            fig = notes[fi % 3: fi % 3 + 5]
            if (s0 > b - 0.4) if not last else (s0 + 4 * step > dur - 1.6):
                break
            for j, m in enumerate(fig):
                tj = s0 + j * step + rng.uniform(-0.006, 0.006)
                if not last and tj > b - 0.08:   # leave the change itself to the arrival chord
                    break
                prog_ = np.clip((tj - a) / max(b - a, 1e-3), 0, 1)
                vel = (0.14 + 0.5 * prog_ ** 1.3) if not last else (0.42 - 0.22 * prog_)
                vel *= 1 - 0.06 * j
                place(B["piano"], piano(mtof(m), 1.1, vel, rng), tj, pan=-0.35 + 0.17 * j)
            s0 += period
            fi += 1
    # cello pedal on D2, swelling into every change; the fifth (A2) joins at the last change
    bounds = [0.0] + list(changes)

    def cello_swell(tt):   # rises into each change, peaks on it, relaxes after (no step)
        g = K.zeros(tt.shape[0])
        for k, (a, b) in enumerate(segs):
            m = (tt >= a) & (tt < b)
            rise = ((tt[m] - a) / max(b - a, 1e-3)) ** 2 if k < len(segs) - 1 else 0.0 * tt[m]
            fall = K.exp(-(tt[m] - a) / 0.8) if k > 0 else 0.0 * tt[m]
            g[m] = K.maximum(rise, fall)
        return 0.6 + 0.4 * g
    place(B["cello"], bowed(mtof(38), dur, rng, attack=1.2, swell=cello_swell), 0.0, pan=0.05)
    if changes:
        cl = changes[-1]
        place(B["cello"], 0.6 * bowed(mtof(45), dur - cl, rng, attack=0.7), cl - 0.15, pan=0.25)
    for i, c in enumerate(changes):
        B["sub"] += (0.5 if i < len(changes) - 1 else 1.0) * sub_swell(t, c, mtof(26))
    _ = bounds
    B["cello"] = body(B["cello"])
    mix = {"piano": (0.0, 0.35), "cello": (-5.0, 0.3), "sub": (-11.0, 0.0)}
    return B, mix, log


def preset_E2(t, changes, dur, rng):
    """Travel and longing: bowed suspensions, muted-piano eighths, lands on Gmaj7."""
    n = t.shape[0]
    B = buses(n, "strings", "cello", "pulse", "piano", "sub")
    # name, cello, sus2 voices, the entering 3rd, top line
    prog = [("Gsus2->Gadd9", 43, (50, 57), 59, 62), ("Em(add9)", 40, (47, 54), 55, 64),
            ("Csus2->Cadd9", 36, (43, 50), 52, 67), ("Gmaj7", 43, (50, 54), 59, 66)]
    segs = segments(changes, dur)
    ns = len(segs)
    log = []
    eighth = 0.3
    for k, (a, b) in enumerate(segs):
        name, root, sus, third, top = pick(prog, k, ns)
        last = k == ns - 1
        pre = 0.0 if k == 0 else 0.2
        ln = (b - a) + pre + 0.35
        atk = 1.4 if k == 0 else 0.55
        place(B["cello"], bowed(mtof(root), ln, rng, attack=0.35 if k else 1.0), a - pre * 0.6)
        for j, m in enumerate(sus):
            place(B["strings"], bowed(mtof(m), ln, rng, attack=atk), a - pre, pan=(-0.4, 0.15)[j])
        place(B["strings"], 0.8 * bowed(mtof(top), ln, rng, attack=atk + 0.2, vib_cents=9), a - pre, pan=0.35)
        if not last:   # the suspension rises: the 3rd enters halfway (sus2 -> add9)
            tm = a + 0.5 * (b - a)
            place(B["strings"], 0.85 * bowed(mtof(third), b - tm + 0.35, rng, attack=0.9), tm, pan=-0.1)
        else:
            place(B["strings"], 0.85 * bowed(mtof(third), b - a + 0.3, rng, attack=0.6), a - pre, pan=-0.1)
        # muted piano eighths: root+12 / fifth alternating, a downbeat lands on the change
        pul = (root + 12, root + 19) if root + 19 <= 62 else (root + 12, root + 7 + 12 - 12)
        nb = int(np.floor((b - a) / eighth + 1e-6))
        if last:
            nb = min(nb, 6)
        for i in range(nb):
            ti = a + i * eighth + (rng.uniform(-0.005, 0.005) if i else 0.0)
            v = (0.36 if i % 4 == 0 else 0.26) * (1 - i / 7 if last else 1.0)
            if v > 0.02:
                place(B["pulse"], piano(mtof(pul[i % 2]), 0.12, v, rng, muted=True), ti, pan=0.2 * (-1) ** i)
        if k > 0 and not last:
            log.append(f"{a:.2f}s {name}: cello {root}, strings sus2 (3rd rises in at {a + 0.5 * (b - a):.2f}s), "
                       f"top line MIDI {top}, muted-pulse downbeat")
        if last and k > 0:
            for j, m in enumerate((43, 50, 54, 59)):
                place(B["piano"], piano(mtof(m), b - a, 0.55 - 0.04 * j, rng), a + 0.02 * j, pan=-0.25 + 0.17 * j)
            B["sub"] += sub_swell(t, a, mtof(31))
            log.append(f"{a:.2f}s {name}: landing - piano Gmaj7 (G2 D3 F#3 B3), strings + top F#4 (maj7), "
                       f"sub swell, pulse thins out")
    for c in changes[:-1]:
        B["sub"] += 0.4 * sub_swell(t, c, mtof(31), pre=1.0, post=1.0)
    B["strings"] = body(B["strings"])
    B["cello"] = body(B["cello"])
    mix = {"strings": (0.0, 0.45), "cello": (-3.0, 0.3), "pulse": (-7.0, 0.25),
           "piano": (-4.0, 0.4), "sub": (-12.0, 0.0)}
    return B, mix, log


def preset_E3(t, changes, dur, rng):
    """Structure and wonder: wood ostinato in 7, cello line, piano chords stacked in fifths."""
    n = t.shape[0]
    B = buses(n, "mallet", "cello", "piano", "sub")
    stack = (36, 43, 50, 57, 64, 71)                                   # C G D A E B
    cello = [(36, "C2"), (43, "G2"), (45, "A2"), (36, "C2")]
    ost = ((48, 1.0), (55, 0.5), (48, 0.8), (50, 0.5), (43, 0.9), (48, 0.5), (50, 0.45))   # 2+2+3
    eighth = 0.19
    segs = segments(changes, dur)
    ns = len(segs)
    log = []
    for k, (a, b) in enumerate(segs):
        last = k == ns - 1
        nnotes = min(2 + k, 5) if not last else 5
        chord = stack[:nnotes]
        t0 = a if k > 0 else 0.15
        vel = 0.5 + 0.05 * k if not last else 0.62
        for j, m in enumerate(chord):
            v = vel * (1 - 0.07 * j)
            place(B["piano"], piano(mtof(m), b - t0 + 0.2, v, rng), t0 + 0.008 * j, pan=-0.3 + 0.15 * j)
        if b - a > 2.0 and not last:   # a softer re-strike holds the architecture up mid-segment
            for j, m in enumerate(chord[1:]):
                place(B["piano"], piano(mtof(m), b - a, 0.28, rng), a + 0.5 * (b - a) + 0.008 * j,
                      pan=-0.15 + 0.15 * j)
        if last:   # crown: the B (7th fifth) as soft high harmony
            place(B["piano"], piano(mtof(stack[5]), b - a, 0.24, rng), a + 0.6, pan=0.35)
        cm, cn = pick(cello, k, ns)
        place(B["cello"], bowed(mtof(cm), b - a + 0.25, rng, attack=0.5 if k else 1.0), a - (0.12 if k else 0))
        # ostinato: phase resets on the change so its downbeat is the arrival
        i = 0
        stop = b if not last else dur - 1.4
        while a + i * eighth < stop - 0.02:
            m, acc = ost[i % 7]
            v = (0.42 + 0.3 * acc) * (0.85 if not last else 0.75) * rng.uniform(0.94, 1.0)
            place(B["mallet"], mallet(mtof(m), v, rng), a + i * eighth + (rng.uniform(-0.004, 0.004) if i else 0),
                  pan=(-0.25, 0.1, 0.3)[i % 3])
            i += 1
        if k > 0:
            names = "-".join("CGDAEB"[:nnotes])
            log.append(f"{a:.2f}s piano fifths stack {names}, cello {cn}, ostinato downbeat (7/8 restarts)"
                       + (", sub swell, soft B on top" if last else ""))
    if changes:
        B["sub"] += sub_swell(t, changes[-1], mtof(31))
    B["cello"] = body(B["cello"])
    mix = {"mallet": (-6.0, 0.3), "cello": (-4.0, 0.3), "piano": (0.0, 0.35), "sub": (-14.0, 0.0)}
    return B, mix, log


def preset_E4(t, changes, dur, rng):
    """Noise settling into pattern: air + beating drones resolve to pure D major; piano at the end."""
    n = t.shape[0]
    B = buses(n, "air", "drone", "piano", "sub")
    segs = segments(changes, dur)
    ns = len(segs)
    lv = [k / (ns - 1) if ns > 1 else 1.0 for k in range(ns)]
    lv = [x ** 0.8 for x in lv]
    step = K.asarray(C.gate_env(K.tonp(t), [(a, lv[k]) for k, (a, _) in enumerate(segs)], 0.45, 0.45, 0.0))
    ramp = K.clip(t / (changes[-1] if changes else dur), 0, 1)
    o = K.clip(0.7 * step + 0.3 * ramp, 0, 1)                        # order: 0 = noise, 1 = pattern
    dis = (1 - o) ** 1.4

    # air: broad noise darkening and thinning as order grows; resonant noise peaks mid-way
    raw = noise(rng, (2, n))
    unit = lambda y: y / (float(K.sqrt(K.mean(y ** 2))) + 1e-12)   # noqa: E731
    broad = unit(lp(bp(raw, 150, 1600), 2400, order=4))
    dark = unit(lp(raw, 420))
    mid = 4 * o * (1 - o)
    air = broad * dis + 0.5 * dark * mid
    res = K.zeros((2, n))
    for m in (38, 45, 50, 54, 57, 62):
        b_, a_ = signal.iirpeak(mtof(m), 60.0, fs=SR)
        res = res + filt_ba(raw, b_, a_)
    air += 0.9 * unit(res) * K.clip(mid, 0, 1) ** 1.2
    swell = 0.85 + 0.15 * K.sin(TAU * 0.17 * t + 1.0)
    B["air"] = air * swell

    # drones: detuned pairs whose beating and pitch error shrink with order; voices enter per change
    voices = [(38, 0.0, 0.0, 1.0), (45, 0.0, -0.5, 0.8), (50, 1, -0.7, 0.6), (54, 2, 0.9, 0.5), (57, 3, 0.35, 0.45)]
    ent = [0.0] + list(changes)
    for m, ei, off, g in voices:
        ei = int(ei)
        if ei >= len(ent):
            ei = len(ent) - 1
        te = ent[ei]
        gate = ramp_env(t, te - (0.6 if ei else 0.0), 1.2 if ei else 2.0)
        semis = off * dis
        cents = 38 * dis + 0.4
        for sgn, pan in ((-1, -0.5), (1, 0.5), (0, 0.0)):
            if sgn == 0:
                fi = mtof(m) * 2 ** (semis * 1.15 / 12 + 0.6 * cents * K.sin(TAU * 0.09 * t) / 1200)
                gg = 0.6 * dis
            else:
                fi = mtof(m) * 2 ** (semis / 12 + sgn * cents / 2 / 1200)
                gg = 1.0
            y = drone_osc(fi, h2=0.3, h3=0.12, ph0=rng.uniform(0, TAU)) * gate * g * gg
            place(B["drone"], y, 0.0, pan=pan * (0.6 + 0.4 * float(K.mean(dis))))
    log = []
    names = {1: "D3", 2: "F#3", 3: "A3"}
    for i, c in enumerate(changes):
        lastc = i == len(changes) - 1
        if not lastc:
            B["sub"] += 0.45 * sub_swell(t, c, mtof(26), pre=1.0, post=1.2)
            log.append(f"{c:.2f}s order step to {lv[i + 1]:.2f}: drone {names.get(i + 1, '')} glides in, "
                       f"beating narrows, noise darkens toward the chord's partials, soft sub")
        else:
            for j, m in enumerate((50, 57, 62, 66)):
                place(B["piano"], piano(mtof(m), dur - c, 0.5 - 0.05 * j, rng), c + 0.03 * j, pan=-0.3 + 0.2 * j)
            if dur - c > 1.8:
                place(B["piano"], piano(mtof(69), dur - c - 1.0, 0.22, rng), c + 1.0, pan=0.3)
            B["sub"] += sub_swell(t, c, mtof(26))
            log.append(f"{c:.2f}s resolve: pure D major (D2 A2 D3 F#3 A3) in tune, noise gone, "
                       f"piano D3 A3 D4 F#4 (first piano), sub swell")
    mix = {"air": (-4.0, 0.4), "drone": (0.0, 0.25), "piano": (-3.0, 0.4), "sub": (-12.0, 0.0)}
    return B, mix, log


def preset_E5(t, changes, dur, rng):
    """Additive growth: a piano cell that gains a note per cycle, phasing copy, cello roots, bloom."""
    n = t.shape[0]
    B = buses(n, "piano", "phase", "cello", "strings", "air", "sub")
    step = 0.15
    slots = {0: 53, 1: 60, 2: 57, 3: 55, 4: 62, 5: 60, 6: 65, 7: 57}   # F pentatonic cell
    order = [0, 3, 6, 2, 5, 1, 4, 7]
    roots = [(41, "F2"), (38, "D2"), (36, "C2"), (41, "F2")]
    segs = segments(changes, dur)
    ns = len(segs)
    log = []
    cyc = 0
    for k, (a, b) in enumerate(segs):
        last = k == ns - 1
        stop = b if not last else dur - 1.3
        c0 = a
        while c0 < stop - 0.05:
            nn = min(8, cyc + 1)
            active = set(order[:nn])
            for s in range(8):
                if s not in active:
                    continue
                ts = c0 + s * step
                if ts >= stop - 0.02:
                    continue
                v = (0.5 if s == 0 else 0.36) * rng.uniform(0.92, 1.0) * (1.08 if last else 1.0)
                place(B["piano"], piano(mtof(slots[s]), 0.45, v, rng), ts + (rng.uniform(-0.004, 0.004) if s else 0),
                      pan=-0.3 + 0.08 * s)
                if last and s in (0, 6):   # bloom: soft octave doubling on top
                    place(B["piano"], piano(mtof(slots[s] + 12), 0.45, 0.2, rng), ts + 0.01, pan=0.3)
                if cyc >= 4:              # phasing copy, 4% faster, realigns at every change
                    tp = a + (c0 - a + s * step) * 0.96
                    if tp < stop - 0.02:
                        place(B["phase"], piano(mtof(slots[s]), 0.4, 0.26, rng), tp, pan=0.45)
            c0 += 8 * step
            cyc += 1
        rm, rn = pick(roots, k, ns)
        place(B["cello"], bowed(mtof(rm), b - a + 0.25, rng, attack=0.45 if k else 1.0), a - (0.1 if k else 0))
        if last and k > 0:
            for j, m in enumerate((48, 57, 60, 64, 67)):
                place(B["strings"], bowed(mtof(m), b - a, rng, attack=0.9, vib_cents=6), a - 0.15,
                      pan=-0.4 + 0.2 * j)
            B["sub"] += sub_swell(t, a, mtof(29))
            log.append(f"{a:.2f}s bloom: cello {rn}, string Fmaj9 swells in, cell full + octave doubling, "
                       f"breath swell, sub swell")
        elif k > 0:
            log.append(f"{a:.2f}s cello root {rn}, cell restarts on the change "
                       f"(now {min(8, cyc)} notes/cycle), phasing copy realigns")
    # breath texture: slow swells, a bigger one into the bloom
    env = 0.5 + 0.25 * K.sin(TAU * 0.13 * t + 0.4)
    if changes:
        c = changes[-1]
        env = env + 0.9 * K.where(t < c, K.clip((t - (c - 1.5)) / 1.5, 0, 1) ** 2, K.exp(-(t - c) / 1.2))
    B["air"] = breath(n, rng, 300, 1400) * env
    B["cello"] = body(B["cello"])
    B["strings"] = body(B["strings"])
    mix = {"piano": (0.0, 0.4), "phase": (-6.0, 0.45), "cello": (-5.0, 0.3), "strings": (-2.0, 0.5),
           "air": (-17.0, 0.5), "sub": (-12.0, 0.0)}
    return B, mix, log


PRESETS = {"E1": preset_E1, "E2": preset_E2, "E3": preset_E3, "E4": preset_E4, "E5": preset_E5}


# ---------------------------------------------------------------- render

def render(preset, duration, changes, seed):
    """Mix on the current backend K; returns a numpy (2, n) array (pre-loudness), log, stem levels."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    t = K.arange(n) / SR
    B, mix, log = PRESETS[preset](t, changes, duration, rng)
    dry = K.zeros((2, n))
    send = K.zeros((2, n))
    levels = {}
    for k, x in B.items():
        r = active_rms(x)
        if r <= 1e-11:
            continue
        g = 10 ** (mix[k][0] / 20) / r
        dry = dry + g * x
        send = send + g * mix[k][1] * x
        levels[k] = mix[k][0]
    ir = C.room_ir(rng, rt60=2.4, length=3.0)
    if K.torch:
        wet = K.stack([fftconv(send[ch], K.asarray(ir[ch])) for ch in range(2)])
    else:
        wet = C.reverb(send, ir, circular=False)
    x = dry + 0.9 * wet
    x = hp(x, 30)
    x = lp(x, 7000)
    fin = ramp_env(t, 0, 0.01)
    fout = 0.5 - 0.5 * K.cos(np.pi * K.clip((duration - t) / FADE_OUT, 0, 1))
    x = x * fin * fout
    K.sync()
    return K.tonp(x), log, levels


def hf_share(x, fc=2000.0):
    m = x.mean(axis=0)
    X = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.size, 1 / SR)
    return float(X[f > fc].sum() / (X.sum() + 1e-20))


def main() -> None:
    ap = argparse.ArgumentParser(description="Short no-bleep scores for the mathtype pieces (E1..E5)")
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--duration", type=float, required=True, help="length in s (8-10 typical)")
    ap.add_argument("--changes", default="", help="comma-separated arrival times in s (visual resolves)")
    ap.add_argument("--out", required=True, help="output wav (48 kHz stereo 16-bit)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force", action="store_true", help="allow overwriting --out")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "torch"), help="numpy = reference")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (torch only)")
    ap.add_argument("--timings", help="append a kind=music row to this TIMINGS.csv")
    ap.add_argument("--device-label", help="device column for --timings (default: inferred)")
    ap.add_argument("--note", default="", help="extra text for the timings notes column")
    ap.add_argument("--gen", help="gen column for --timings (default: gNN from the --out name)")
    a = ap.parse_args()
    t_start = time.perf_counter()
    started = _nz_now()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists; not overwriting (use a new gNN name, or --force)")
    if a.changes.strip():
        changes = sorted(float(s) for s in a.changes.split(",") if s.strip())
    else:
        changes = [round(a.duration * r, 3) for r in (0.28, 0.55, 0.83)]
    changes = [c for c in changes if 0.0 < c < a.duration - 0.2]

    set_backend(a.backend, a.device)
    t_setup = time.perf_counter()
    x, log, levels = render(a.preset, a.duration, changes, a.seed)
    t_synth = time.perf_counter()
    x, L, tp, passes = C.loudness(x, False)
    x[:, -1] = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    wavfile.write(a.out, SR, (np.clip(x, -1, 1).T * 32767).round().astype(np.int16))
    print(f"wrote {a.out}: preset {a.preset}, {x.shape[1] / SR:.3f} s, 2 ch, {SR} Hz, seed {a.seed}")
    print(f"loudness {L:.2f} LUFS (ffmpeg ebur128), true peak {tp:.2f} dBTP, "
          f"sample peak {20 * np.log10(np.abs(x).max() + 1e-12):.2f} dBFS, gain passes {passes}")
    print(f"energy above 2 kHz: {100 * hf_share(x):.2f} %   stems (dB rel.): "
          + ", ".join(f"{k} {v:+.0f}" for k, v in levels.items()))
    t_end = time.perf_counter()
    print(f"time: backend setup {t_setup - t_start:.2f} s, synth {t_synth - t_setup:.2f} s ({K.name}), "
          f"loudness+write {t_end - t_synth:.2f} s, total {t_end - t_start:.2f} s")
    print("arrivals:")
    for line in log:
        print("  " + line)
    if a.timings:
        append_timing(a, started, t_synth - t_setup, t_end - t_start,
                      f"preset {a.preset} seed {a.seed} {K.name}; setup {t_setup - t_start:.2f}s synth "
                      f"{t_synth - t_setup:.2f}s loudness+write {t_end - t_synth:.2f}s; {L:.2f} LUFS {tp:.2f} dBTP"
                      + (f"; {a.note}" if a.note else ""))


TIMING_HEADER = ("gen,test,kind,device,frames,resolution,started_nz,finished_nz,make_min,render_s,"
                 "per_frame_ms,encode_s,audio_s,notes")


def _nz_now() -> str:
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("Pacific/Auckland"))
    except Exception:   # no tzdata: NZDT
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=13)))
    return now.strftime("%Y-%m-%d %H:%M:%S")


def append_timing(a, started, synth_s, total_s, notes):
    import re
    m = re.search(r"_g(\d+)", os.path.basename(a.out))
    gen = a.gen or (f"g{m.group(1)}" if m else os.path.splitext(os.path.basename(a.out))[0].rsplit("_", 1)[-1])
    dev = a.device_label
    if not dev:
        if K.torch and K.dev.type == "cuda":
            dev = "cuda-" + K.T.cuda.get_device_name().replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
        elif K.torch:
            dev = "local-cpu-torch" if os.name == "nt" else "box-cpu-torch"
        else:
            dev = "local-numpy" if os.name == "nt" else "box-cpu"
    row = [gen, f"score_{a.preset}", "music", dev, "", "", started, _nz_now(), "", f"{synth_s:.2f}", "", "",
           f"{total_s:.2f}", '"' + notes.replace('"', "'") + '"']
    new = not os.path.exists(a.timings)
    with open(a.timings, "a", encoding="utf-8", newline="") as fh:   # append only (shared file)
        fh.write((TIMING_HEADER + "\n" if new else "") + ",".join(row) + "\n")


if __name__ == "__main__":
    main()
