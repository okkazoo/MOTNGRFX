"""CPU music bed + SFX synthesis (SPEC Interface 4).

All numpy / scipy, 48 kHz stereo by default, deterministic from ``plan["seed"]``.
No per-sample Python loops: oscillators are wavetable lookups, envelopes are
vector maths, filters are ``scipy.signal.lfilter`` (block-wise for sweeps) and
reverb is an FFT convolution with a Schroeder-network impulse response.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import signal

__all__ = ["render_soundtrack", "render_sfx", "reverb", "biquad", "MOODS"]

MOODS = ("driving", "calm", "dark", "bright", "playful", "epic")

# ----------------------------------------------------------------------------
# small DSP toolkit
# ----------------------------------------------------------------------------


def _rng(seed, *tags) -> np.random.Generator:
    try:
        s = int(seed)
    except (TypeError, ValueError):
        s = 0
    return np.random.default_rng([abs(s) % (2 ** 63)] + [int(t) for t in tags])


def biquad(kind: str, f0: float, sr: int, q: float = 0.7071, gain_db: float = 0.0):
    """RBJ cookbook biquad -> (b, a). kind: lowpass|highpass|bandpass|highshelf|lowshelf|peak."""
    f0 = float(np.clip(f0, 5.0, 0.49 * sr))
    w0 = 2.0 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("highshelf", "lowshelf"):
        sq = 2.0 * math.sqrt(A) * alpha
        if kind == "highshelf":
            b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
            a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
        else:
            b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
            a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    a = np.asarray(a, np.float64)
    return np.asarray(b, np.float64) / a[0], a / a[0]


def _filt(x: np.ndarray, kind: str, f0: float, sr: int, q: float = 0.7071, gain_db: float = 0.0) -> np.ndarray:
    b, a = biquad(kind, f0, sr, q, gain_db)
    return signal.lfilter(b, a, x, axis=-1)


def _sweep(x: np.ndarray, sr: int, f_start: float, f_end: float, kind: str = "lowpass",
           q: float = 0.9, block: int = 256, curve: float = 1.0) -> np.ndarray:
    """Time-varying biquad (exponential cutoff sweep), processed block-wise with carried state."""
    x2 = np.atleast_2d(x).astype(np.float64)
    n = x2.shape[1]
    if n == 0:
        return x2.reshape(np.shape(x))
    nb = int(math.ceil(n / block))
    u = (np.arange(nb) + 0.5) / nb
    fc = f_start * (f_end / f_start) ** (u ** curve)
    y = np.empty_like(x2)
    zi = np.zeros((x2.shape[0], 2))
    for i in range(nb):
        s, e = i * block, min(n, (i + 1) * block)
        b, a = biquad(kind, fc[i], sr, q)
        y[:, s:e], zi = signal.lfilter(b, a, x2[:, s:e], axis=-1, zi=zi)
    return y.reshape(np.shape(x)) if np.ndim(x) == 1 else y


def _onepole_lp(x: np.ndarray, sr: int, tau: float) -> np.ndarray:
    a = math.exp(-1.0 / max(1e-6, tau * sr))
    return signal.lfilter([1 - a], [1, -a], x, axis=-1)


_TABLE_N = 4096


@lru_cache(maxsize=256)
def _table(kind: str, K: int) -> np.ndarray:
    x = np.arange(_TABLE_N) / _TABLE_N
    k = np.arange(1, K + 1)
    if kind == "saw":
        amp = 1.0 / k
    elif kind == "square":
        amp = np.where(k % 2 == 1, 1.0 / k, 0.0)
    elif kind == "triangle":
        amp = np.where(k % 2 == 1, ((-1.0) ** ((k - 1) // 2)) / k ** 2, 0.0)
    else:  # sine
        amp = (k == 1).astype(np.float64)
    tab = (amp[:, None] * np.sin(2 * np.pi * k[:, None] * x[None, :])).sum(0)
    return (tab / (np.abs(tab).max() + 1e-12)).astype(np.float64)


def _osc(kind: str, freq: float, n: int, sr: int, phase0: float = 0.0, max_hz: float = 16000.0) -> np.ndarray:
    """Band-limited wavetable oscillator at a fixed frequency."""
    K = int(max(1, min(64, math.floor(min(max_hz, 0.45 * sr) / max(freq, 1.0)))))
    tab = _table(kind, K)
    ph = np.arange(n, dtype=np.float64)
    ph *= freq * _TABLE_N / sr
    ph += (phase0 % 1.0) * _TABLE_N
    return tab[ph.astype(np.int64) & (_TABLE_N - 1)]


def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float, gate: int | None = None) -> np.ndarray:
    """Linear attack, exponential decay to sustain, linear release after gate samples."""
    t = np.arange(n) / sr
    gate = n if gate is None else int(min(gate, n))
    env = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    if gate < n:
        tg = gate / sr
        lev = float(env[gate - 1]) if gate > 0 else 0.0
        rel = lev * np.clip(1.0 - (t - tg) / max(r, 1e-6), 0.0, 1.0)
        env = np.where(t >= tg, rel, env)
    return env


def _fade_edges(x: np.ndarray, sr: int, fin: float = 0.002, fout: float = 0.003) -> np.ndarray:
    n = x.shape[-1]
    a, b = min(n, int(fin * sr)), min(n, int(fout * sr))
    if a > 0:
        x[..., :a] *= np.linspace(0, 1, a)
    if b > 0:
        x[..., n - b:] *= np.linspace(1, 0, b)
    return x


def _pan_gains(p: float):
    p = float(np.clip(p, -1, 1))
    return min(1.0, 1.0 - p), min(1.0, 1.0 + p)


def _add(buf: np.ndarray, x: np.ndarray, start: int, gain: float = 1.0, pan: float = 0.0) -> None:
    """Add mono (n,) or stereo (2, n) x into buf (2, N) at sample start (clipped)."""
    N = buf.shape[1]
    n = x.shape[-1]
    s0, s1 = max(0, start), min(N, start + n)
    if s1 <= s0:
        return
    seg = x[..., s0 - start:s1 - start]
    if x.ndim == 1:
        gl, gr = _pan_gains(pan)
        buf[0, s0:s1] += gain * gl * seg
        buf[1, s0:s1] += gain * gr * seg
    else:
        buf[:, s0:s1] += gain * seg


# ----------------------------------------------------------------------------
# reverb (Schroeder comb/allpass network rendered to an IR, applied by FFT)
# ----------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _reverb_ir(sr: int, decay: float):
    L = int(sr * min(4.0, decay * 1.05)) + 1
    rng = np.random.default_rng(9001)
    burst_n = max(8, int(0.006 * sr))
    burst = rng.standard_normal(burst_n) * np.exp(-np.arange(burst_n) / (0.0015 * sr))
    irs = []
    for ch, spread in enumerate((0.0, 0.0023)):
        ir = np.zeros(L)
        for ms in (29.7, 37.1, 41.1, 43.7, 47.9, 53.3):
            D = int(round((ms / 1000.0 + spread * (1 + ch)) * sr))
            g = 10.0 ** (-3.0 * D / (sr * decay))
            k = np.arange(1, L // D + 1)
            ir[k * D - 1] += g ** k
        for ms, g in ((5.0, 0.7), (1.7, 0.7), (0.71, 0.6)):
            D = max(1, int(round(ms / 1000.0 * sr * (1 + 0.07 * ch))))
            K = int(math.log(1e-4) / math.log(g)) + 1
            ap = np.zeros(K * D + 1)
            ap[0] = -g
            kk = np.arange(1, K + 1)
            ap[kk * D] = (1 - g * g) * g ** (kk - 1)
            ir = signal.fftconvolve(ir, ap)[:L]
        ir = signal.fftconvolve(ir, burst)[:L]
        ir = _filt(ir, "lowpass", 6500.0, sr)
        ir = _filt(ir, "highpass", 120.0, sr)
        ir *= np.exp(-np.arange(L) / (decay * sr / 6.9))  # enforce RT60
        ir /= math.sqrt(float((ir ** 2).sum()) + 1e-12)
        irs.append(ir)
    return np.stack(irs)


def reverb(x: np.ndarray, sr: int, decay: float = 1.5, wet: float = 0.25, predelay: float = 0.012) -> np.ndarray:
    """Stereo reverb. x: (n,) or (2, n). Returns (2, n): dry + wet (tail truncated to n)."""
    x2 = np.atleast_2d(np.asarray(x, np.float64))
    if x2.shape[0] == 1:
        x2 = np.repeat(x2, 2, axis=0)
    n = x2.shape[1]
    if n == 0 or wet <= 0:
        return x2
    ir = _reverb_ir(int(sr), float(round(decay, 2)))
    pd = int(predelay * sr)
    out = x2.copy()
    for c in range(2):
        w = signal.oaconvolve(x2[c], ir[c])[: max(0, n - pd)]
        out[c, pd:pd + len(w)] += wet * 0.35 * w
    return out


# ----------------------------------------------------------------------------
# drum kit
# ----------------------------------------------------------------------------


def _kit(sr: int, mood: str, rng: np.random.Generator) -> dict:
    def t_(dur):
        n = int(dur * sr)
        return np.arange(n) / sr, n

    kit = {}
    # kick: pitch-swept sine + click
    t, n = t_(0.5)
    f_hi, f_lo = {"dark": (140, 42), "epic": (130, 40), "calm": (120, 50), "playful": (190, 58)}.get(mood, (160, 47))
    tail = {"dark": 0.32, "epic": 0.34, "calm": 0.2}.get(mood, 0.19)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.032)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / tail)
    click = _filt(rng.standard_normal(n) * np.exp(-t / 0.0025), "highpass", 1800.0, sr) * 0.35
    k = np.tanh(1.6 * (body + click)) / math.tanh(1.6)
    kit["kick"] = _fade_edges(k, sr, 0.0005, 0.02)
    # snare: tone + bandpassed noise
    t, n = t_(0.4)
    tone = (np.sin(2 * np.pi * 185 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.055) * 0.55
    nz = _filt(_filt(rng.standard_normal(n), "highpass", 1200.0, sr), "lowpass", 9000.0, sr) * np.exp(-t / 0.11)
    s = tone + 0.8 * nz
    kit["snare"] = _fade_edges(s / np.abs(s).max(), sr, 0.0005, 0.02)
    # closed / open hats: highpassed noise + metallic squares
    metal = sum(np.sign(np.sin(2 * np.pi * fm * np.arange(int(0.6 * sr)) / sr))
                for fm in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0))
    hn = _filt(_filt(rng.standard_normal(int(0.6 * sr)) + 0.25 * metal, "highpass", 7000.0, sr), "highpass", 6000.0, sr)
    t, n = t_(0.12)
    ch = hn[:n] * np.exp(-t / 0.03)
    kit["ch"] = _fade_edges(ch / np.abs(ch).max(), sr, 0.0003, 0.01)
    t, n = t_(0.6)
    oh = hn[:n] * np.exp(-t / 0.22)
    kit["oh"] = _fade_edges(oh / np.abs(oh).max(), sr, 0.0003, 0.05)
    # clap: three bursts + tail, bandpassed
    t, n = t_(0.45)
    env = np.zeros(n)
    for i, ti in enumerate((0.0, 0.011, 0.023)):
        env += np.where(t >= ti, np.exp(-(t - ti) / 0.004), 0.0) * (0.8 + 0.1 * i)
    env += np.where(t >= 0.03, 0.6 * np.exp(-(t - 0.03) / 0.13), 0.0)
    cl = _filt(rng.standard_normal(n) * env, "bandpass", 1500.0, sr, q=1.1)
    kit["clap"] = _fade_edges(cl / np.abs(cl).max(), sr, 0.0003, 0.02)
    # crash
    t, n = t_(2.5)
    metal = sum(np.sign(np.sin(2 * np.pi * fm * t)) for fm in (283.0, 417.0, 523.0, 678.0, 811.0, 1043.0))
    cr = _filt(rng.standard_normal(n) + 0.3 * metal, "highpass", 4000.0, sr) * np.exp(-t / 0.9)
    kit["crash"] = _fade_edges(cr / np.abs(cr).max(), sr, 0.0005, 0.1)
    return kit


# 16-step patterns. Digit = minimum density tier (0 thin .. 2 full); '.' = never.
_DRUMS = {
    "driving": {"kick": "0...0...0...0.2.", "snare": "....1.......1...", "clap": "....2.......2...",
                "ch": "1202120212021202", "oh": "..2...2...2...2."},
    "calm": {"kick": "0.........1.....", "snare": "........1.......", "clap": "................",
             "ch": "2.1.2.1.2.1.2.1.", "oh": "..............2."},
    "dark": {"kick": "0......1..1...2.", "snare": "........1.......", "clap": "........2.......",
             "ch": "1202120212021222", "oh": "......2.........", },
    "bright": {"kick": "0...0...0...0...", "snare": "....1.......1...", "clap": "....1.......1...",
               "ch": "1202120212021202", "oh": "..2...2...2...2."},
    "playful": {"kick": "0..1..1...0..1..", "snare": "....1.......1...", "clap": "............2...",
                "ch": "1202120212021202", "oh": "..............2."},
    "epic": {"kick": "0...2...1.1...2.", "snare": "....2...1.......", "clap": "........2.......",
             "ch": "2...2...2...2...", "oh": "..............2."},
}
_DRUM_GAIN = {"kick": 0.95, "snare": 0.6, "clap": 0.45, "ch": 0.2, "oh": 0.15}
_BASS = {"driving": "0.1.0.1.0.1.0.1.", "calm": "0...............", "dark": "0......0..0.....",
         "bright": "0.1.0.1.0.1.0.1.", "playful": "0..1..0.1..0.1..", "epic": "0.......0......."}
_PROG = {"driving": [0, 0, 5, 6], "calm": [0, 4, 5, 3], "dark": [0, 5, 3, 4],
         "bright": [0, 3, 5, 4], "playful": [0, 5, 1, 4], "epic": [0, 5, 2, 6]}
_ARP = {"driving": [0, 1, 2, 3, 2, 1], "calm": [0, 1, 2, 1], "dark": [0, 2, 1, 3],
        "bright": [0, 1, 2, 3], "playful": [0, 2, 1, 3, 1, 2], "epic": [0, 1, 2, 3, 2, 1]}
_SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11], "ionian": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10], "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11], "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11], "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SECTION_LEVEL = {"intro": -1, "break": -1, "breakdown": -1, "bridge": -1, "outro": -1,
                  "drop": 1, "chorus": 1, "hook": 1, "climax": 1}
_LEAD_SECTIONS = ("drop", "chorus", "hook", "climax")


def _key_pc(key) -> int:
    s = str(key or "A").strip()
    if not s:
        return 9
    pc = _PC.get(s[0].upper(), 9)
    if len(s) > 1 and s[1] in "#♯":
        pc += 1
    elif len(s) > 1 and s[1] in "b♭":
        pc -= 1
    return pc % 12


def _midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


# ----------------------------------------------------------------------------
# timeline helpers
# ----------------------------------------------------------------------------


class _Grid:
    """Beats / bars / 16th steps derived from plan.audio (follows user beat grids)."""

    def __init__(self, audio: dict, duration: float):
        bpm = float(audio.get("bpm") or 120.0)
        bpm = float(np.clip(bpm, 30.0, 300.0))
        beats = np.array(sorted(float(b) for b in (audio.get("beats") or []) if -1e-6 <= float(b) < duration), np.float64)
        if len(beats) < 2:
            beats = np.arange(0.0, duration, 60.0 / bpm)
        self.beats = beats
        lens = np.diff(beats)
        last = lens[-1] if len(lens) else 60.0 / bpm
        self.beat_len = np.concatenate([lens, [last]])
        db = np.array(sorted(float(d) for d in (audio.get("downbeats") or []) if -1e-6 <= float(d) < duration))
        # map each downbeat to its nearest beat index
        if len(db):
            idx = np.unique([int(np.argmin(np.abs(beats - d))) for d in db])
            idx = np.array([i for i in idx if abs(beats[i] - db[np.argmin(np.abs(db - beats[i]))]) < 0.08], int)
        else:
            idx = np.array([], int)
        if len(idx) == 0:
            idx = np.arange(0, len(beats), 4)
        self.db_idx = idx
        pos = np.zeros(len(beats), int)
        bar = np.full(len(beats), -1, int)
        for j, i0 in enumerate(idx):
            i1 = idx[j + 1] if j + 1 < len(idx) else len(beats)
            pos[i0:i1] = np.arange(i1 - i0)
            bar[i0:i1] = j
        if idx[0] > 0:  # pickup beats before the first downbeat
            k = idx[0]
            pos[:k] = (4 - k + np.arange(k)) % 4
        self.pos = pos
        self.bar = bar
        self.downbeats = beats[idx]
        self.bar_t0 = np.concatenate([[0.0] if idx[0] > 0 else [], self.downbeats])
        self.has_pickup = idx[0] > 0
        self.duration = duration

    def bar_bounds(self):
        """[(bar_number, t0, t1)] with bar_number -1 for a pickup."""
        out = []
        starts = list(self.bar_t0)
        for i, t0 in enumerate(starts):
            t1 = starts[i + 1] if i + 1 < len(starts) else self.duration
            num = i - 1 if self.has_pickup else i
            out.append((num, float(t0), float(t1)))
        return out

    def steps(self, swing: float = 0.0):
        """All 16th steps: arrays (time, step_in_bar 0..15, step_len, bar_number)."""
        k = np.arange(4)
        t = self.beats[:, None] + self.beat_len[:, None] * (k[None, :] / 4.0)
        if swing:
            t = t + (k[None, :] % 2 == 1) * swing * self.beat_len[:, None] / 4.0
        step = (self.pos[:, None] * 4 + k[None, :]) % 16
        sl = np.repeat(self.beat_len[:, None] / 4.0, 4, axis=1)
        bar = np.repeat(self.bar[:, None], 4, axis=1)
        return t.ravel(), step.ravel(), sl.ravel(), bar.ravel()


def _sections(audio: dict, duration: float):
    secs = sorted(({"t": float(s.get("t", 0.0)), "label": str(s.get("label", "main")).lower()}
                   for s in (audio.get("sections") or []) if isinstance(s, dict)), key=lambda s: s["t"])
    if not secs or secs[0]["t"] > 1e-3:
        secs.insert(0, {"t": 0.0, "label": "main"})
    out = []
    for i, s in enumerate(secs):
        t1 = secs[i + 1]["t"] if i + 1 < len(secs) else duration
        if t1 > s["t"]:
            out.append((s["t"], min(t1, duration), s["label"]))
    return out


def _label_at(secs, t: float) -> str:
    lab = secs[0][2] if secs else "main"
    for t0, _, l in secs:
        if t0 <= t + 1e-6:
            lab = l
    return lab


def _chord(scale: list[int], root_pc: int, degree: int, n_notes: int = 3) -> list[int]:
    """Diatonic chord as pitch offsets (semitones) from the key root in octave 0."""
    out = []
    for j in range(n_notes):
        d = degree + 2 * j
        out.append(root_pc + scale[d % 7] + 12 * (d // 7))
    return out


# ----------------------------------------------------------------------------
# render_soundtrack
# ----------------------------------------------------------------------------


def render_soundtrack(plan: dict, sample_rate: int = 48000) -> np.ndarray:
    """Synthesised music bed (2, round(duration * sr)) float32 following plan.audio."""
    sr = int(sample_rate)
    duration = float(plan.get("duration") or 10.0)
    N = int(round(duration * sr))
    audio = plan.get("audio") or {}
    style = plan.get("style") or {}
    seed = plan.get("seed", 0)
    mood = str(audio.get("mood") or "driving").lower()
    if mood not in MOODS:
        mood = "driving"
    energy = float(np.clip(style.get("energy", 0.6) if style.get("energy") is not None else 0.6, 0.0, 1.0))
    if N <= 0:
        return np.zeros((2, 0), np.float32)

    grid = _Grid(audio, duration)
    secs = _sections(audio, duration)
    scale = _SCALES.get(str(audio.get("scale") or "minor").lower(), _SCALES["minor"])
    root = _key_pc(audio.get("key"))
    prog = _PROG[mood]
    base_tier = 0 if energy < 0.35 else (1 if energy < 0.7 else 2)
    swing = 0.3 if mood == "playful" else 0.0

    # final-bar treatment: last downbeat leaving >= 1 s, if there are >= 3 bars
    final_t = None
    dbs = grid.downbeats
    cands = dbs[dbs <= duration - 1.0]
    if len(dbs) >= 3 and len(cands) >= 2:
        final_t = float(cands[-1])

    drums = np.zeros((2, N))
    bass = np.zeros((2, N))
    pad = np.zeros((2, N))
    lead = np.zeros((2, N))
    fx = np.zeros((2, N))
    send = np.zeros((2, N))  # reverb send

    kit = _kit(sr, mood, _rng(seed, 1))
    rng_v = _rng(seed, 2)

    # ---------------- drums
    st_t, st_step, st_len, st_bar = grid.steps(swing)
    kick_times = []
    for inst in ("kick", "snare", "clap", "oh", "ch"):
        pat = _DRUMS[mood][inst]
        for t, stp, sl in zip(st_t, st_step, st_len):
            if t >= duration or (final_t is not None and t >= final_t - 1e-6):
                continue
            ch_ = pat[stp]
            lab = _label_at(secs, t)
            tier = int(np.clip(base_tier + _SECTION_LEVEL.get(lab, 0), 0, 2))
            downbeat = stp == 0
            if ch_ == "." or int(ch_) > tier:
                if not (inst == "kick" and downbeat):
                    continue
            if inst == "ch" and _DRUMS[mood]["oh"][stp] != "." and int(_DRUMS[mood]["oh"][stp]) <= tier:
                continue  # open hat chokes the closed one
            vel = 1.0 if stp % 4 == 0 else (0.82 if stp % 2 == 0 else 0.62)
            vel *= 1.0 + 0.08 * (rng_v.random() - 0.5)
            g = _DRUM_GAIN[inst] * vel * (0.7 if mood == "calm" else 1.0)
            pan = {"ch": 0.25, "oh": -0.2, "clap": 0.08}.get(inst, 0.0)
            _add(drums, kit[inst], int(round(t * sr)), g, pan)
            if inst == "kick":
                kick_times.append(t)
            if inst in ("snare", "clap"):
                _add(send, kit[inst], int(round(t * sr)), g * 0.6, pan)

    # crash on drop/chorus starts (snapped to the nearest beat)
    for t0, _, lab in secs:
        if lab in _LEAD_SECTIONS and t0 > 0.05:
            tb = float(grid.beats[np.argmin(np.abs(grid.beats - t0))])
            _add(drums, kit["crash"], int(round(tb * sr)), 0.22, -0.3)

    # fill + reverse-cymbal swell into the final downbeat, then the final hit
    if final_t is not None:
        bl = float(grid.beat_len[np.argmin(np.abs(grid.beats - final_t))])
        f0 = final_t - 2 * bl
        nsteps = 8
        for i in range(nsteps):
            t = f0 + i * bl / 4.0
            if t < 0:
                continue
            g = 0.25 + 0.4 * (i / (nsteps - 1))
            _add(drums, kit["snare"], int(round(t * sr)), g * 0.7, 0.1 * ((i % 2) * 2 - 1))
            _add(send, kit["snare"], int(round(t * sr)), g * 0.3)
        sw_n = int(2 * bl * sr)
        cr = kit["crash"] if sw_n <= len(kit["crash"]) else np.pad(kit["crash"], (0, sw_n - len(kit["crash"])))
        sw = cr[:sw_n][::-1].copy()
        sw = sw * np.linspace(0, 1, len(sw)) ** 2
        _fade_edges(sw, sr, 0.01, 0.004)
        _add(drums, sw, int(round(final_t * sr)) - len(sw), 0.3, 0.2)
        _add(drums, kit["kick"], int(round(final_t * sr)), 1.0)
        _add(drums, kit["crash"], int(round(final_t * sr)), 0.32, -0.25)
        _add(send, kit["crash"], int(round(final_t * sr)), 0.25)
        kick_times.append(final_t)

    # ---------------- harmony: bass, pad, lead per bar
    rng_p = _rng(seed, 3)
    bass_kind = "square" if mood == "playful" else "saw"
    lead_kind = {"playful": "square", "bright": "square", "calm": "triangle"}.get(mood, "saw")
    pad_attack = 0.8 if mood == "calm" else (0.5 if mood in ("epic", "dark") else 0.3)
    pad_gain = 0.5 if mood == "calm" else 0.32
    bpat = _BASS[mood]
    arp_order = _ARP[mood]
    arp_div = 2 if mood in ("calm", "epic") else 4  # notes per beat

    for bar_num, t0, t1 in grid.bar_bounds():
        if t1 - t0 < 1e-3:
            continue
        is_final = final_t is not None and abs(t0 - final_t) < 1e-6
        if final_t is not None and t0 > final_t + 1e-6:
            continue
        deg = 0 if (is_final or bar_num < 0) else prog[bar_num % len(prog)]
        n_tones = 4 if mood in ("calm", "bright", "epic") else 3
        tones = _chord(scale, root, deg, n_tones)
        lab = _label_at(secs, t0)
        bar_end = duration if is_final else t1

        # pad: detuned saw stack, slow attack, released into the next bar
        rel = 1.2 if is_final else 0.5
        s0 = int(round(t0 * sr))
        n = int(round((bar_end - t0 + rel) * sr))
        gate = int(round((bar_end - t0) * sr))
        env = _adsr(n, sr, pad_attack if not is_final else 0.01, 1.5 if not is_final else 1.2,
                    0.8 if not is_final else 0.0, rel, gate)
        st = np.zeros((2, n))
        for m in tones:
            mid = 57 + ((m - 57) % 12)  # close voicing around A3..G#4
            f = _midi_hz(mid)
            for c, cents in ((0, -9.0), (1, 9.0)):
                ph = rng_p.random(2)
                v = _osc("saw", f * 2 ** (cents / 1200), n, sr, ph[0], 4500.0) + 0.6 * _osc("saw", f, n, sr, ph[1], 4500.0)
                st[c] += v
        st *= env[None, :] / (len(tones) * 1.6)
        _add(pad, st, s0, pad_gain)
        _add(send, st, s0, pad_gain * 0.8)

        # bass
        bass_mid = 28 + ((tones[0] - 28) % 12)
        steps_in_bar = [(t, s, l) for t, s, l, b in zip(st_t, st_step, st_len, st_bar)
                        if t0 - 1e-6 <= t < t1 - 1e-6]
        tier = int(np.clip(base_tier + _SECTION_LEVEL.get(lab, 0), 0, 2))
        ons = [(t, s) for t, s, _ in steps_in_bar if bpat[s] != "." and int(bpat[s]) <= max(tier, 1)]
        if is_final or not ons:
            ons = [(t0, 0)]
        for j, (t, s) in enumerate(ons):
            tn = ons[j + 1][0] if j + 1 < len(ons) else bar_end
            dur = max(0.06, (tn - t) * (0.92 if not is_final else 1.0))
            nn = int((dur + 0.05) * sr)
            octv = 12 if (mood in ("bright", "playful") and s % 4 == 2) else 0
            f = _midi_hz(bass_mid + octv)
            v = 0.8 * _osc(bass_kind, f, nn, sr, 0.0, 3000.0) + 0.35 * _osc("sine", f, nn, sr)
            e = _adsr(nn, sr, 0.004, 0.18 if mood != "calm" else 0.8, 0.65, 0.04, int(dur * sr))
            if is_final:
                e *= np.exp(-np.arange(nn) / (0.9 * sr))
            _add(bass, v * e, int(round(t * sr)), 0.34 * (1.0 if s % 4 == 0 else 0.7))

        # lead / arp
        lead_on = (lab in _LEAD_SECTIONS) or (energy >= 0.6 and lab not in ("intro", "outro", "break", "breakdown"))
        if lead_on and not is_final:
            chord_lead = [72 + ((m - 72) % 12) for m in tones[:3]]
            chord_lead = sorted(chord_lead) + [sorted(chord_lead)[0] + 12]
            k = 0
            for t, s, l in steps_in_bar:
                if arp_div == 2 and s % 2 == 1:
                    continue
                m = chord_lead[arp_order[k % len(arp_order)] % len(chord_lead)]
                k += 1
                step_len = l * (4 // arp_div)
                nn = int(step_len * 0.95 * sr)
                v = _osc(lead_kind, _midi_hz(m), nn, sr, 0.0, 9000.0)
                e = _adsr(nn, sr, 0.003, 0.11 if mood != "calm" else 0.3, 0.15, 0.02, int(nn * 0.8))
                pan = 0.3 * math.sin(k * 0.9)
                _add(lead, v * e, int(round(t * sr)), 0.2 if mood != "calm" else 0.13, pan)

    if np.any(lead):
        # ping-pong delay of 3 sixteenths
        d = int(round(0.75 * float(np.median(grid.beat_len)) * sr))
        echo = np.zeros_like(lead)
        if 0 < d < N:
            echo[0, d:] += 0.35 * lead[1, :-d]
        if 0 < 2 * d < N:
            echo[1, 2 * d:] += 0.25 * lead[0, :-2 * d]
        lead += echo
        lead = _filt(lead, "lowpass", 5000.0 if mood != "dark" else 3000.0, sr)
        send += 0.5 * lead

    bass = _filt(bass, "lowpass", 220.0 + 900.0 * energy, sr, q=0.9)
    pad = _filt(pad, "lowpass", (3200.0 if mood == "calm" else 1600.0 + 1400.0 * energy), sr)
    send = _filt(send, "lowpass", 5000.0, sr)

    # sidechain pump of bass + pad from the kick (four-on-the-floor styles)
    if mood in ("driving", "bright", "epic", "playful") and kick_times:
        imp = np.zeros(N)
        ks = np.clip(np.round(np.array(kick_times) * sr).astype(int), 0, N - 1)
        imp[ks] = 1.0
        env = signal.lfilter([1.0], [1.0, -math.exp(-1.0 / (0.11 * sr))], imp)
        env = np.minimum(env, 1.0)
        env = _onepole_lp(env, sr, 0.004)
        depth = 0.25 + 0.35 * energy
        pad *= 1.0 - depth * env
        bass *= 1.0 - 0.5 * depth * env

    music = drums + bass + pad + lead + fx
    decay = {"calm": 2.6, "epic": 2.8, "dark": 2.0}.get(mood, 1.4)
    music += reverb(send, sr, decay, wet=1.0)[:, :N] - send  # wet only from the send bus

    # intro sections: low-pass sweep up across the section
    for t0, t1, lab in secs:
        if lab == "intro" and t1 - t0 > 0.25:
            a, b = int(t0 * sr), min(N, int(t1 * sr))
            music[:, a:b] = _sweep(music[:, a:b], sr, 350.0 + 400.0 * energy, 16000.0, "lowpass", 0.8, curve=1.4)

    music = _filt(music, "highpass", 28.0, sr)
    music = _filt(music, "lowshelf", 90.0, sr, 0.7071, -5.0)  # tame the sub so the mids carry
    rms = float(np.sqrt(np.mean(music ** 2)) + 1e-12)
    peak = float(np.abs(music).max() + 1e-12)
    music *= min(10 ** (-16.0 / 20.0) / rms, 0.9 / peak)
    return np.ascontiguousarray(music, dtype=np.float32)


# ----------------------------------------------------------------------------
# SFX
# ----------------------------------------------------------------------------


def _whoosh(sr, rng, dur=0.7):
    n = int(dur * sr)
    t = np.arange(n) / sr
    u = t / dur
    peak = 0.6
    env = np.where(u < peak, (u / peak) ** 2.2, np.exp(-(u - peak) / 0.12))
    nz = rng.standard_normal((2, n))
    sw = _sweep(nz, sr, 350.0, 5000.0, "bandpass", 1.4, curve=0.7)
    sw = sw * env[None]
    pan = np.linspace(-0.85, 0.85, n)
    gl, gr = np.minimum(1, 1 - pan), np.minimum(1, 1 + pan)
    out = np.stack([sw[0] * gl, sw[1] * gr])
    return _fade_edges(out / (np.abs(out).max() + 1e-9) * 0.7, sr, 0.005, 0.01), int(peak * n)


def _impact(sr, rng):
    n = int(2.6 * sr)
    t = np.arange(n) / sr
    f = 38.0 + 90.0 * np.exp(-t / 0.05)
    thump = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.55)
    nz = _filt(rng.standard_normal(n), "lowpass", 3500.0, sr) * np.exp(-t / 0.07) * 0.6
    x = np.tanh(1.8 * (thump + nz)) / math.tanh(1.8)
    x = reverb(_fade_edges(x, sr, 0.0005, 0.05), sr, 2.2, wet=0.9, predelay=0.005)
    return x / (np.abs(x).max() + 1e-9) * 0.85


def _hit(sr, rng):
    n = int(0.9 * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * (95.0 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.12)
    nz = _filt(rng.standard_normal(n), "bandpass", 2200.0, sr, q=0.8) * np.exp(-t / 0.05)
    x = reverb(_fade_edges(tone + 0.9 * nz, sr, 0.0005, 0.03), sr, 1.0, wet=0.5)
    return x / (np.abs(x).max() + 1e-9) * 0.75


def _click(sr, rng):
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    x = (np.sin(2 * np.pi * 3200 * t) + 0.3 * rng.standard_normal(n)) * np.exp(-t / 0.0015)
    return _fade_edges(x / (np.abs(x).max() + 1e-9) * 0.5, sr, 0.0, 0.003)


def _pop(sr, rng):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    f = 330.0 + 900.0 * np.exp(-t / 0.012)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.035)
    return _fade_edges(x * 0.6, sr, 0.0005, 0.01)


def _glitch(sr, rng):
    reps = int(rng.integers(5, 9))
    grain_n = int(rng.uniform(0.022, 0.04) * sr)
    gap = int(rng.uniform(0.008, 0.02) * sr)
    f = float(rng.uniform(180, 900))
    t = np.arange(grain_n) / sr
    grain = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + rng.standard_normal(grain_n) * 0.5
    out = np.zeros((2, reps * (grain_n + gap)))
    for i in range(reps):
        hold = int(rng.integers(3, 16))
        g = np.repeat(grain[::hold], hold)[:grain_n]
        bits = int(rng.integers(3, 6))
        q = 2 ** (bits - 1)
        g = np.round(g * q) / q
        g = _fade_edges(g.copy(), sr, 0.001, 0.002) * (1.0 - 0.06 * i)
        p = float(rng.uniform(-0.8, 0.8))
        gl, gr = _pan_gains(p)
        s = i * (grain_n + gap)
        out[0, s:s + grain_n] += gl * g
        out[1, s:s + grain_n] += gr * g
    return out / (np.abs(out).max() + 1e-9) * 0.55


def _swell(sr, rng, dur):
    n = max(16, int(dur * sr))
    t = np.arange(n) / sr
    nz = _filt(rng.standard_normal((2, n)), "highpass", 2500.0, sr)
    env = np.exp((t - dur) / (dur / 4.0))
    x = nz * env[None]
    x = _fade_edges(x, sr, 0.005, 0.004)
    return x / (np.abs(x).max() + 1e-9) * 0.5


def _riser(sr, rng, dur):
    n = max(16, int(round(dur * sr)))
    t = np.arange(n) / sr
    u = t / max(dur, 1e-6)
    f = 140.0 * (2200.0 / 140.0) ** u
    ph = 2 * np.pi * np.cumsum(f) / sr
    saw = np.zeros(n)
    for k in range(1, 9):
        saw += np.where(k * f < 18000.0, np.sin(k * ph) / k, 0.0)
    nz = rng.standard_normal((2, n))
    x = 0.55 * saw[None, :] + 0.5 * nz
    x = _sweep(x, sr, 250.0, 12000.0, "lowpass", 1.6, curve=1.0)
    env = u ** 1.8
    x = x * env[None]
    x = _fade_edges(x, sr, 0.003, 0.002)
    return x / (np.abs(x).max() + 1e-9) * 0.6


def render_sfx(plan: dict, sample_rate: int = 48000) -> np.ndarray:
    """All plan.audio.sfx events + plan.audio.risers as a (2, N) float32 stem."""
    sr = int(sample_rate)
    duration = float(plan.get("duration") or 10.0)
    N = int(round(duration * sr))
    out = np.zeros((2, max(N, 0)))
    if N <= 0:
        return out.astype(np.float32)
    audio = plan.get("audio") or {}
    seed = plan.get("seed", 0)
    bpm = float(audio.get("bpm") or 120.0)
    bar = 4 * 60.0 / max(bpm, 1.0)

    def place_ending(x, t1):
        e = int(round(t1 * sr))
        _add(out, x, e - x.shape[-1])

    risers = [(float(r.get("t0", 0.0)), float(r.get("t1", 0.0)), float(r.get("gain", 0.7)))
              for r in (audio.get("risers") or []) if isinstance(r, dict)]
    for i, ev in enumerate(audio.get("sfx") or []):
        if not isinstance(ev, dict):
            continue
        typ = str(ev.get("type", "")).lower()
        t = float(ev.get("t", ev.get("t1", 0.0)))
        gain = float(ev.get("gain", 0.8))
        rng = _rng(seed, 1000 + i)
        s = int(round(t * sr))
        if typ == "riser":
            t1 = float(ev.get("t1", t))
            t0 = float(ev.get("t0", t1 - float(ev.get("dur", bar))))
            risers.append((t0, t1, gain))
        elif typ == "whoosh":
            x, pk = _whoosh(sr, rng, float(ev.get("dur", 0.7)))
            _add(out, x, s - pk, gain)
        elif typ == "impact":
            _add(out, _impact(sr, rng), s, gain)
        elif typ == "hit":
            _add(out, _hit(sr, rng), s, gain)
        elif typ == "click":
            _add(out, _click(sr, rng), s, gain)
        elif typ == "pop":
            _add(out, _pop(sr, rng), s, gain)
        elif typ == "glitch":
            _add(out, _glitch(sr, rng), s, gain)
        elif typ == "swell":
            x = _swell(sr, rng, float(ev.get("dur", 0.5 * bar)))
            place_ending(x * gain, t)
    for j, (t0, t1, g) in enumerate(risers):
        t0 = max(0.0, t0)
        if t1 - t0 < 0.05:
            continue
        x = _riser(sr, _rng(seed, 5000 + j), t1 - t0)
        place_ending(x * g, t1)
    return np.ascontiguousarray(out, dtype=np.float32)
