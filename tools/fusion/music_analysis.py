"""music_analysis.py - one audio file -> <name>_analysis.json (15_fusion music side).

    python tools/fusion/music_analysis.py song.flac [--out DIR] [--bpm-hint 72] [--beats-per-bar 4] [--fps 24]

Output keys: bpm (+bpm_candidates), beats, downbeats, bars (start/end/rms_db/bass/chord/conf + two half-bar chords),
chord_changes (bar level), harmonic_rhythm_fast, sections (energy), onsets, key (estimate, top scores, tuning),
curves (24 fps: rms, sub, low, mid, high, onset; each 0-1; value of frame i is sampled at t = i / fps).

Pure numpy / scipy (+ soundfile). librosa is NOT used. Reused code:
  - mograph.analyze (imported): _onset_envelope, _estimate_period, _beat_track, _peak_pick, _sections, _resample
  - tools/mathtype/audiofeat.py Features (imported by path): band envelopes on a 200 Hz grid, folded to fps here
  - chroma / key / chord / bass functions below are COPIED from 03_sound/v004/local/cut_and_score.py
    (stft_mag, spectral_peaks, estimate_tuning, chroma_from_mag, key_scores, chord_from_chroma,
    bass_pitch_per_bar), unchanged except for docstring trimming.
Tempo: mograph's tracker prefers 90-140 bpm, so slow pieces (60-84) can come back doubled. Pass --bpm-hint
(the planned tempo) to track at that level; without a hint, bpm_candidates lists the half / double.
"""
import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # .../motion-graphic
sys.path.insert(0, str(ROOT))
from mograph.analyze import (_beat_track, _estimate_period, _onset_envelope, _ONSET_LAG,  # noqa: E402
                             _peak_pick, _resample, _sections)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------- copied from cut_and_score.py
def stft_mag(x, sr, W, hop):
    f, t, Z = signal.stft(x, sr, nperseg=W, noverlap=W - hop, window="hann", boundary="zeros", padded=True)
    return f, t, np.abs(Z)


KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def spectral_peaks(f, A, fmin=100.0, fmax=2000.0, thr=0.3):
    """whitened-log spectral peaks with parabolic frequency interpolation. Returns (frame idx, freq Hz, weight)"""
    L = np.log1p(A * 60.0)
    Wh = np.clip(L - median_filter(L, size=(33, 1)), 0, None)
    mid = Wh[1:-1]
    m = (mid > Wh[:-2]) & (mid >= Wh[2:]) & (mid > thr)
    bi, fi = np.nonzero(m)
    bi = bi + 1
    a_, b_, c_ = Wh[bi - 1, fi], Wh[bi, fi], Wh[bi + 1, fi]
    den = a_ - 2 * b_ + c_
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (a_ - c_) / den, 0.0)
    fr = (bi + np.clip(delta, -0.5, 0.5)) * (f[1] - f[0])
    ok = (fr >= fmin) & (fr <= fmax)
    return fi[ok], fr[ok], b_[ok]


def estimate_tuning(f, A):
    fi, fr, w = spectral_peaks(f, A)
    if len(fr) == 0:
        return 0.0, 0.0
    cents = (1200 * np.log2(fr / 440.0)) % 100
    z = np.sum(w * np.exp(2j * np.pi * cents / 100.0))
    c = (np.angle(z) / (2 * np.pi)) % 1 * 100
    return float(c - 100 if c >= 50 else c), float(abs(z) / max(w.sum(), 1e-9))


def chroma_from_mag(f, A, tuning_cents=0.0):
    """A: (nf, nt) linear magnitude. Returns chroma (12, nt) from interpolated spectral peaks, soft-assigned to the two
    nearest semitone classes"""
    fi, fr, w = spectral_peaks(f, A)
    midi = 69 + 12 * np.log2(fr / 440.0) - tuning_cents / 100.0
    lo = np.floor(midi).astype(int)
    fracs = midi - lo
    out = np.zeros((12, A.shape[1]))
    np.add.at(out, (lo % 12, fi), w * (1 - fracs))
    np.add.at(out, ((lo + 1) % 12, fi), w * fracs)
    return out


def key_scores(chroma_vec):
    out = []
    for tonic in range(12):
        for mode, prof in (("major", KS_MAJ), ("minor", KS_MIN)):
            c = np.corrcoef(chroma_vec, np.roll(prof, tonic))[0, 1]
            out.append((float(c), f"{NOTE_NAMES[tonic]} {mode}"))
    out.sort(reverse=True)
    return out


CHORD_T = {"": [0, 4, 7], "m": [0, 3, 7], "sus4": [0, 5, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11]}


def chord_from_chroma(ch, bass_pc=None):
    best = []
    n = ch / max(ch.sum(), 1e-9)
    for root in range(12):
        for name, iv in CHORD_T.items():
            tmpl = np.zeros(12)
            for i in iv:
                tmpl[(root + i) % 12] = 1.0
            sc = float(np.dot(n, tmpl) / math.sqrt(tmpl.sum()))
            if bass_pc is not None and root == bass_pc:
                sc *= 1.15
            best.append((sc, f"{NOTE_NAMES[root]}{name}", root))
    best.sort(reverse=True)
    margin = (best[0][0] - best[1][0]) / max(best[0][0], 1e-9)
    return best[0][1], best[0][2], float(np.clip(best[0][0] * 1.4, 0, 1) * np.clip(0.5 + margin * 5, 0, 1)), best[:3]


def bass_pitch_per_bar(mono, sr, bar_t0, bar_len, nbars):
    """bass fundamental per bar: harmonic-sum salience over 38-175 Hz on long-window spectra, bar median.
    A bar gets a bass note only if its 35-170 Hz power is within 22 dB of the file's 90th percentile of that power."""
    W = 16384
    hopb = 2400
    f, t, A = stft_mag(mono, sr, W, hopb)
    df = f[1] - f[0]
    f0 = 38.0 * 2.0 ** (np.arange(0, int(1200 * math.log2(175 / 38.0) // 10) + 1) * 10 / 1200)
    sal = np.zeros((len(f0), A.shape[1]))
    for h in range(1, 5):
        idx = np.clip(np.round(h * f0 / df).astype(int), 0, A.shape[0] - 1)
        sal += np.log1p(A[idx] * 60.0) / h ** 0.5
    best = f0[np.argmax(sal, axis=0)]
    band = (f >= 35) & (f <= 170)
    pw = 10 * np.log10((A[band] ** 2).sum(0) + 1e-12)
    ref = np.percentile(pw, 90)
    out = []
    for b in range(nbars):
        a, e = bar_t0 + b * bar_len + 0.25 * bar_len, bar_t0 + (b + 1) * bar_len - 0.1 * bar_len
        sel = (t >= a) & (t < e)
        if sel.sum() == 0 or np.median(pw[sel]) < ref - 22:
            out.append(None)
            continue
        out.append(float(69 + 12 * np.log2(np.median(best[sel]) / 440.0)))
    return out


# ---------------------------------------------------------------- audiofeat (by path)
def _audiofeat():
    spec = importlib.util.spec_from_file_location("fusion_audiofeat", ROOT / "tools" / "mathtype" / "audiofeat.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def band_curves(stereo, sr, fps, n_frames):
    """sub (<120 Hz) / low (120-600) / mid (600-2000) / high (>2000) / rms / onset, 0-1, one value per video frame
    (frame i covers t in [(i-0.5)/fps, (i+0.5)/fps]; envelopes = mean, onset = max over that span)."""
    af = _audiofeat()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mono.wav"
        sf.write(str(p), stereo.mean(1).astype(np.float32), sr, subtype="FLOAT")
        F = af.Features(str(p))
    out = {}
    names = {"rms": "rms", "sub": "sub", "low": "lowmid", "mid": "mid", "high": "high", "onset": "onset"}
    for k, src in names.items():
        raw = F.raw[src]
        vals = np.zeros(n_frames)
        for i in range(n_frames):
            a = int(round(max(i - 0.5, 0) / fps / F.hop))
            b = max(int(round((i + 0.5) / fps / F.hop)), a + 1)
            seg = raw[a:b] if a < len(raw) else raw[-1:]
            vals[i] = seg.max() if k == "onset" else seg.mean()
        out[k] = [round(float(v), 4) for v in vals]
    return out


# ---------------------------------------------------------------- tempo / beats
def track_beats(mono, sr, bpm_hint=None, beats_per_bar=4):
    x, sr2 = _resample(mono.astype(np.float32), sr)
    env, fr, hop, mag, freqs = _onset_envelope(x, sr2)
    dur = len(mono) / sr
    bpm0, period = _estimate_period(env, fr)
    if bpm_hint:
        period = 60.0 * fr / float(bpm_hint)
    bf = _beat_track(env, period)
    beat_t = np.clip(bf / fr + _ONSET_LAG, 0.0, dur)
    bpm = 60.0 / period * fr if not bpm_hint else float(bpm_hint)
    if len(beat_t) >= 4:
        slope = np.polyfit(np.arange(len(beat_t)), beat_t, 1)[0]
        if slope > 0 and 0.85 < (60.0 / slope) / bpm < 1.15:
            bpm = 60.0 / slope
    downbeats = np.zeros(0)
    if len(beat_t):
        low = mag[freqs < 150.0].sum(axis=0) if np.any(freqs < 150.0) else mag[:2].sum(axis=0)
        bfi = np.clip(np.round(bf).astype(int), 0, len(low) - 1)
        w = max(1, int(round(0.05 * fr)))
        lowb = np.array([low[max(0, i - 1):i + w].max() for i in bfi])
        phase = int(np.argmax([lowb[p::beats_per_bar].mean() if len(lowb[p::beats_per_bar]) else -1
                               for p in range(beats_per_bar)]))
        downbeats = beat_t[phase::beats_per_bar]
    on = _peak_pick(env, fr)
    onsets = np.clip(on / fr + _ONSET_LAG, 0.0, dur)
    sections = _sections(x, sr2, mag, freqs, fr, beat_t, dur)
    return dict(bpm=bpm, bpm_raw=bpm0, beats=beat_t, downbeats=downbeats, onsets=onsets, sections=sections)


def bar_grid(downbeats, bpm, beats_per_bar, dur):
    """regular bar grid fitted to the tracked downbeats; returns (bar_len, start0, [bar start times])"""
    bar_len = beats_per_bar * 60.0 / bpm
    if len(downbeats) >= 3:
        k = np.round((downbeats - downbeats[0]) / bar_len)
        a, b = np.polyfit(k, downbeats, 1)
        if 0.9 < a / bar_len < 1.1:
            bar_len, t0 = float(a), float(b)
        else:
            t0 = float(downbeats[0])
    elif len(downbeats):
        t0 = float(downbeats[0])
    else:
        t0 = 0.0
    t0 = t0 % bar_len
    if t0 > bar_len - 0.03:
        t0 = 0.0
    n = int((dur - t0 + 0.05) // bar_len)
    return bar_len, t0, [t0 + i * bar_len for i in range(n)]


# ---------------------------------------------------------------- main analysis
def analyse(path, bpm_hint=None, beats_per_bar=4, fps=24, grid=None):
    """grid=(bar_len_s, first_bar_start_s) overrides the fitted bar grid (use for bar-aligned cuts)."""
    x, sr = sf.read(str(path), always_2d=True)
    mono = x.mean(1)
    dur = len(mono) / sr
    bt = track_beats(mono, sr, bpm_hint, beats_per_bar)
    bpm = bt["bpm"]
    bar_len, t0, starts = bar_grid(bt["downbeats"], bpm, beats_per_bar, dur)
    if grid:
        bar_len, t0 = float(grid[0]), float(grid[1])
        starts = [t0 + i * bar_len for i in range(int((dur - t0 + 0.05) // bar_len))]
    nb = len(starts)

    f, t, A = stft_mag(mono, sr, 8192, 960)
    tuning, _ = estimate_tuning(f, A)
    chroma = chroma_from_mag(f, A, tuning_cents=tuning)
    ks = key_scores(chroma.mean(1))
    bass_bar = bass_pitch_per_bar(mono, sr, t0, bar_len, nb) if nb else []
    half = bar_len / 2
    bass_half = bass_pitch_per_bar(mono, sr, t0, half, nb * 2) if nb else []

    def pc(m):
        return None if m is None else int(round(m - tuning / 100.0)) % 12

    bars, changes = [], []
    n_diff = 0
    for b in range(nb):
        s, e = starts[b], starts[b] + bar_len
        sel = (t >= s) & (t < e)
        chord, _, conf, _top = chord_from_chroma(chroma[:, sel].sum(1), pc(bass_bar[b]))
        a, c = int(round(s * sr)), min(int(round(e * sr)), len(mono))
        rms_db = 20 * math.log10(float(np.sqrt((mono[a:c] ** 2).mean())) + 1e-9) if c > a else -90.0
        hb = []
        for h in range(2):
            hs = s + h * half
            hsel = (t >= hs) & (t < hs + half)
            hc, _, hconf, _ = chord_from_chroma(chroma[:, hsel].sum(1), pc(bass_half[2 * b + h]))
            hb.append({"t": round(hs, 4), "chord": hc, "conf": round(hconf, 2)})
        n_diff += hb[0]["chord"] != hb[1]["chord"]
        bars.append({"n": b + 1, "start": round(s, 4), "end": round(e, 4), "rms_db": round(rms_db, 2),
                     "bass": None if bass_bar[b] is None else NOTE_NAMES[pc(bass_bar[b])],
                     "chord": chord, "conf": round(conf, 2), "half_bars": hb})
        if b and chord != bars[b - 1]["chord"]:
            changes.append({"t": round(s, 4), "bar": b + 1, "from": bars[b - 1]["chord"], "to": chord})
    n_frames = int(math.ceil(dur * fps))
    curves = {"fps": fps, "n": n_frames,
              "note": "value of frame i is the mean (onset: max) over [(i-0.5)/fps, (i+0.5)/fps]; each curve / its 99.5th percentile, 0-1"}
    curves.update(band_curves(x, sr, fps, n_frames))
    best_score, best_key = ks[0]
    return {
        "file": str(path), "duration": round(dur, 4), "sample_rate": int(sr),
        "bpm": round(float(bpm), 3), "bpm_untracked_estimate": round(float(bt["bpm_raw"]), 2),
        "bpm_candidates": [round(bpm / 2, 2), round(bpm, 2), round(bpm * 2, 2)],
        "bpm_hint": bpm_hint, "time_signature": f"{beats_per_bar}/4",
        "bar_len": round(bar_len, 4), "first_bar_start": round(t0, 4),
        "beats": [round(float(v), 4) for v in bt["beats"]],
        "downbeats": [round(float(v), 4) for v in bt["downbeats"]],
        "key": {"estimate": best_key, "corr": round(best_score, 3),
                "scores_top5": [{"key": k, "corr": round(s, 3)} for s, k in ks[:5]],
                "tuning_cents": round(tuning, 1)},
        "bars": bars, "chord_changes": changes,
        "harmonic_rhythm_fast": bool(nb and n_diff / nb >= 0.3),
        "harmonic_rhythm_note": "half_bars lists a chord per half bar; fast = half-bar chords differ in >=30 % of bars",
        "sections": bt["sections"],
        "onsets": [round(float(v), 4) for v in bt["onsets"]],
        "curves": curves,
        "method": "mograph.analyze tracker + cut_and_score chroma/chord + audiofeat bands; numpy/scipy only",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", help="output dir (default: next to the audio)")
    ap.add_argument("--bpm-hint", type=float)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--grid", type=float, nargs=2, metavar=("BAR_S", "T0"), help="force bar length and first bar start")
    a = ap.parse_args()
    p = Path(a.audio)
    res = analyse(p, a.bpm_hint, a.beats_per_bar, a.fps, a.grid)
    out = Path(a.out) if a.out else p.parent
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{p.stem}_analysis.json"
    dst.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{dst}\n  bpm {res['bpm']} (untracked {res['bpm_untracked_estimate']})  key {res['key']['estimate']}  "
          f"bars {len(res['bars'])} x {res['bar_len']} s  first bar {res['first_bar_start']} s  "
          f"fast harmonic rhythm {res['harmonic_rhythm_fast']}")
    print("  chords:", " ".join(b["chord"] for b in res["bars"]))


if __name__ == "__main__":
    main()
