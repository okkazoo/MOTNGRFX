#!/usr/bin/env python3
"""music_verify.py - measure a rendered music WAV against its own score. Numbers and pictures only; a human listens last.

    python tools/music_verify.py <wav> --score <score.json> --out <dir> [--determinism det.json] [--same-as other.wav] [--stems-dir DIR]

Writes <dir>/verify.json, <dir>/spectrogram.png, <dir>/waveform.png and prints a summary.

Score json keys used (all optional except bpm/downbeats for the plots):
  bpm, downbeats [s], bars [{n, chord}], sections [{name, start, end, energy}], drop_downbeat [s],
  events [{t, instrument, type, anchor}]   anchor events are checked against the grid (kick and accents, 5 ms),
  render_meta {limiter_max_gr_db, ...}      written by the renderer.
Checks follow _info/music-prompt.md step 5. The onset detector and the section / drop levels are our own
(op55's verify.py, MIT, was the model for what to measure; no code copied).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mograph.mix import k_weight, loudness_lufs  # noqa: E402

LIMITS = dict(lufs=(-14.5, -13.5), true_peak=-1.5, limiter_gr=4.0, low=35.0, mid=55.0, high=3.0, centroid=(900.0, 2000.0),
              onset_ms=5.0, rank=0.9, drop_db=3.0)


# --- io ---------------------------------------------------------------------------------------------------------
def read_wav(path):
    """PCM 16/24/32 or float32 WAV -> (float64 (ch, n), sr). Own parser: scipy reads 24-bit only in recent versions."""
    d = Path(path).read_bytes()
    if d[:4] != b"RIFF" or d[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    i, fmt, data = 12, None, None
    while i + 8 <= len(d):
        cid, sz = d[i:i + 4], struct.unpack("<I", d[i + 4:i + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", d[i + 8:i + 24])
        elif cid == b"data":
            data = d[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)
    tag, ch, sr, _, _, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(data, "<f4").astype(np.float64)
    elif bits == 16:
        x = np.frombuffer(data, "<i2").astype(np.float64) / 32768.0
    elif bits == 24:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / 8388608.0
    elif bits == 32:
        x = np.frombuffer(data, "<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav format tag {tag} bits {bits}")
    return x.reshape(-1, ch).T, sr


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def determinism_check(render_fn, n: int = 2) -> dict:
    """Call render_fn() n times; it returns bytes, a numpy array or a path. Equal sha256 = deterministic."""
    hs = []
    for _ in range(n):
        r = render_fn()
        if isinstance(r, (str, Path)):
            hs.append(sha256_file(r))
        elif isinstance(r, (bytes, bytearray)):
            hs.append(hashlib.sha256(bytes(r)).hexdigest())
        else:
            hs.append(hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest())
    return {"sha256": hs, "equal": len(set(hs)) == 1}


# --- measurements -----------------------------------------------------------------------------------------------
def true_peak_db(x, os=4):
    return 20 * math.log10(float(np.abs(signal.resample_poly(x, os, 1, axis=-1)).max()) + 1e-12)


def band_percent(x, sr):
    f, p = signal.welch(x, sr, nperseg=8192, axis=-1)
    p = p.sum(axis=0)
    tot = p.sum()
    return (100 * p[f < 150].sum() / tot, 100 * p[(f >= 150) & (f <= 6000)].sum() / tot, 100 * p[f > 6000].sum() / tot)


def centroid_mean(x, sr):
    """Mean magnitude-weighted spectral centroid over frames within 40 dB of the loudest frame (tail silence excluded)."""
    f, t, Z = signal.stft(x.mean(axis=0), sr, window="hann", nperseg=4096, noverlap=4096 - 1024)
    A = np.abs(Z)
    p = (A ** 2).sum(axis=0)
    keep = p > p.max() * 1e-4
    c = (A * f[:, None]).sum(axis=0) / np.maximum(A.sum(axis=0), 1e-20)
    return float(c[keep].mean())


def kpow_db(kx, a, b, sr):
    """K-weighted mean-square level (LUFS-style, ungated) of kx[:, a s : b s]."""
    seg = kx[:, int(round(a * sr)):int(round(b * sr))]
    return -0.691 + 10 * math.log10(float((seg ** 2).mean(axis=1).sum()) + 1e-12) if seg.size else float("nan")


def _envelopes(m, sr):
    """Causal band-passed analytic envelopes, 1 ms smoothed (zero-phase filtering would ring before the hit)."""
    env = {}
    for key, (lo, hi) in {"low": (45, 180), "kick": (60, 1500), "mid": (200, 8000)}.items():
        sos = signal.butter(2, [lo, hi], "bandpass", fs=sr, output="sos")
        e = np.abs(signal.hilbert(signal.sosfilt(sos, m)))
        env[key] = np.convolve(e, np.ones(int(0.001 * sr)) / int(0.001 * sr), mode="same")
    return env


def onset_errors(x, sr, events, stems=None, win=(0.030, 0.060), frac=0.25):
    """Detect the audible onset of each anchor event and compare with its grid time (ms, + = late).

    Onset = first time after the quietest point of the 30 ms before the hit where the band envelope has risen `frac`
    of the way to its peak in the following 60 ms. The sample's own attack time is inside this number.
    Source: the instrument's own stem when `stems` ({instrument: (ch, n) array}) has it ("stem"), else the full mix
    ("mix"; kicks use 45-180 Hz, other hits 200-8000 Hz, so a louder part in the same band can mask a hit).
    """
    envs = {"mix": _envelopes(x.mean(axis=0), sr)}
    for k, v in (stems or {}).items():
        envs[k] = _envelopes(v.mean(axis=0), sr)
    rows = []
    for ev in events:
        if not ev.get("anchor"):
            continue
        t = ev.get("grid_t", ev["t"])
        src = ev["instrument"] if ev["instrument"] in envs else "mix"
        band = ("kick" if src != "mix" else "low") if ev.get("type") == "kick" else "mid"
        e = envs[src][band]
        a, b = int((t - win[0]) * sr), int((t + win[1]) * sr)
        pre = int(win[0] * sr)
        row = {"t": round(t, 5), "instrument": ev["instrument"], "type": ev.get("type"), "source": "stem" if src != "mix" else "mix", "error_ms": None}
        if a >= 0 and b <= len(e):
            seg = e[a:b]
            i0 = int(np.argmin(seg[:pre]))
            base, peak = seg[i0], seg[pre:].max()
            if peak > base * 1.5:
                idx = i0 + int(np.argmax(seg[i0:] >= base + frac * (peak - base)))
                row["error_ms"] = round(1000 * ((a + idx) / sr - t), 2)
        rows.append(row)
    return rows


# --- plots ------------------------------------------------------------------------------------------------------
def plots(x, sr, score, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = x.mean(axis=0)
    dur = len(m) / sr
    downs = score.get("downbeats", [])
    secs = score.get("sections", [])
    bars = score.get("bars", [])
    bar_len = (downs[1] - downs[0]) if len(downs) > 1 else dur
    edges = list(downs) + [downs[-1] + bar_len] if downs else []

    def decorate(ax, ytxt, dark, chords=True):
        line, sec_c, txt = ("w", "yellow", "w") if dark else ("k", "tab:red", "k")
        for t in edges:
            ax.axvline(t, color=line, lw=0.7, alpha=0.6)
        for s in secs:
            ax.axvline(s["start"], color=sec_c, lw=1.5, alpha=0.9)
            ax.text((s["start"] + s["end"]) / 2, ytxt, s["name"].upper(), ha="center", va="center", fontsize=10, weight="bold", color=sec_c)
        if chords:
            for i, b in enumerate(bars):
                ax.text(downs[i] + bar_len / 2, ytxt * 0.62, f"{b['n']}\n{b.get('chord', '')}", ha="center", va="center", fontsize=8, color=txt)
        if "drop_downbeat" in score:
            ax.axvline(score["drop_downbeat"], color="c", lw=2.0, ls="--")

    f, t, Z = signal.stft(m, sr, window="hann", nperseg=4096, noverlap=4096 - 512)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
    k = (f >= 30) & (f <= 12000)
    ax.pcolormesh(t, f[k], S[k], vmin=S.max() - 85, vmax=S.max(), cmap="magma", shading="auto")
    ax.set_yscale("log")
    ax.set_ylim(30, 12000)
    ax.set_yticks([50, 100, 150, 300, 600, 1000, 2000, 6000, 10000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for hf in (150, 6000):
        ax.axhline(hf, color="c", lw=0.5, ls="--", alpha=0.6)
    decorate(ax, 8000, True)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz (log)")
    ax.set_title(f"{score.get('title', 'music')} - spectrogram (85 dB range); cyan lines: 150 Hz / 6 kHz, bar lines, dashed = drop")
    fig.tight_layout()
    fig.savefig(out / "spectrogram.png")
    plt.close(fig)

    tt = np.arange(len(m)) / sr
    blk = int(0.05 * sr)
    nb = len(m) // blk
    rms = np.sqrt((m[:nb * blk].reshape(nb, blk) ** 2).mean(axis=1))
    fig, ax = plt.subplots(figsize=(16, 4.2), dpi=100)
    ax.plot(tt[::8], m[::8], color="0.55", lw=0.4)
    ax.plot((np.arange(nb) + 0.5) * blk / sr, rms, color="tab:blue", lw=1.4, label="RMS 50 ms")
    ax.plot((np.arange(nb) + 0.5) * blk / sr, -rms, color="tab:blue", lw=1.4)
    ax.set_xlim(0, dur)
    decorate(ax, 0.92 * max(0.05, np.abs(m).max()), False, chords=False)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{score.get('title', 'music')} - waveform (mono mix), bar lines, sections")
    fig.tight_layout()
    fig.savefig(out / "waveform.png")
    plt.close(fig)


# --- main -------------------------------------------------------------------------------------------------------
def verify(wav, score, out_dir, determinism=None, same_as=None, stems_dir=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, sr = read_wav(wav)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    dur = x.shape[1] / sr
    meta = score.get("render_meta", {})
    r: dict = {"file": str(wav), "sample_rate": sr, "channels": int(x.shape[0]), "duration_s": round(dur, 4),
               "sha256": sha256_file(wav)}
    ck: dict = {}

    lufs = loudness_lufs(x, sr)
    tp = true_peak_db(x)
    sp = 20 * math.log10(float(np.abs(x).max()) + 1e-12)
    ck["lufs"] = {"value": round(lufs, 2), "pass": LIMITS["lufs"][0] <= lufs <= LIMITS["lufs"][1], "limit": "-14 +/- 0.5"}
    ck["true_peak_dbtp"] = {"value": round(tp, 2), "pass": tp <= LIMITS["true_peak"], "limit": "<= -1.5 (4x oversampled estimate)"}
    ck["sample_peak_dbfs"] = {"value": round(sp, 2)}
    if "limiter_max_gr_db" in meta:
        ck["limiter_gr_db"] = {"value": round(float(meta["limiter_max_gr_db"]), 2), "pass": meta["limiter_max_gr_db"] <= LIMITS["limiter_gr"], "limit": "<= 4"}
    if "glue_max_gr_db" in meta:
        ck["glue_gr_db"] = {"value": round(float(meta["glue_max_gr_db"]), 2)}
    lo, mid, hi = band_percent(x, sr)
    ck["band_power_pct"] = {"below_150": round(lo, 2), "150_6000": round(mid, 2), "above_6000": round(hi, 3),
                            "pass": lo <= LIMITS["low"] and mid >= LIMITS["mid"] and hi <= LIMITS["high"],
                            "limit": "<150 Hz <= 35, 150-6k >= 55, >6k <= 3"}
    c = centroid_mean(x, sr)
    ck["spectral_centroid_hz"] = {"value": round(c, 1), "pass": LIMITS["centroid"][0] <= c <= LIMITS["centroid"][1], "limit": "900-2000"}

    kx = k_weight(x, sr)
    downs = score.get("downbeats", [])
    bar_len = (downs[1] - downs[0]) if len(downs) > 1 else None
    if downs and bar_len:
        r["bar_levels_db"] = [round(kpow_db(kx, d, d + bar_len, sr), 2) for d in downs]

    evs = [e for e in score.get("events", []) if e.get("anchor")]
    if evs:
        stems = {}
        if stems_dir:
            for inst in {e["instrument"] for e in evs}:
                p_ = Path(stems_dir) / f"{inst}.wav"
                if p_.exists():
                    stems[inst] = read_wav(p_)[0]
        rows = onset_errors(x, sr, evs, stems)
        errs = np.array([w["error_ms"] for w in rows if w["error_ms"] is not None], dtype=float)
        if len(errs):
            by = {}
            for w in rows:
                if w["error_ms"] is not None:
                    by.setdefault(f"{w['instrument']}:{w['type']}:{w['source']}", []).append(w["error_ms"])
            ck["onsets_vs_grid"] = {
                "n_checked": len(errs), "n_undetected": len(rows) - len(errs),
                "max_abs_ms": round(float(np.abs(errs).max()), 2), "median_ms": round(float(np.median(errs)), 2),
                "by_type_median_max_abs_ms": {k: [round(float(np.median(v)), 2), round(float(np.abs(v).max()), 2), len(v)] for k, v in by.items()},
                "pass": bool(np.abs(errs).max() <= LIMITS["onset_ms"] and len(rows) == len(errs)),
                "limit": "anchors (kick, snare, clap) within 5 ms of the score grid; includes each sample's own attack time"}
            r["onset_rows"] = rows
            if stems:  # same detector on the full mix, kicks only: shows how much the rest of the mix masks them
                mrows = [w for w in onset_errors(x, sr, [e for e in evs if e.get("type") == "kick"]) if w["error_ms"] is not None]
                if mrows:
                    me = np.array([w["error_ms"] for w in mrows])
                    ck["kick_onsets_in_mix_info"] = {"n": len(me), "median_ms": round(float(np.median(me)), 2), "max_abs_ms": round(float(np.abs(me).max()), 2),
                                                     "note": "informational: 45-180 Hz envelope of the full mix, kick layers + sub + strings share the band"}

    secs = score.get("sections", [])
    if len(secs) >= 3 and all("energy" in s for s in secs):
        meas = [kpow_db(kx, s["start"], s["end"], sr) for s in secs]
        rho = float(spearmanr([s["energy"] for s in secs], meas).statistic)
        ck["section_energy_rank_corr"] = {"value": round(rho, 3), "pass": rho >= LIMITS["rank"], "limit": ">= 0.9",
                                          "sections": [{"name": s["name"], "planned": s["energy"], "measured_db": round(m_, 2)} for s, m_ in zip(secs, meas)]}
    if "drop_downbeat" in score and bar_len:
        d = score["drop_downbeat"]
        before, after = kpow_db(kx, d - bar_len, d, sr), kpow_db(kx, d, d + bar_len, sr)
        ck["drop_delta_db"] = {"value": round(after - before, 2), "bar_before_db": round(before, 2), "drop_bar_db": round(after, 2),
                               "pass": after - before >= LIMITS["drop_db"], "limit": ">= 3", "note": score.get("drop_note")}

    if determinism:
        ck["determinism"] = {**determinism, "pass": bool(determinism.get("equal"))}
    if same_as:
        ck["same_as"] = {"other": str(same_as), "equal": sha256_file(same_as) == r["sha256"], "pass": sha256_file(same_as) == r["sha256"]}

    r["checks"] = ck
    r["all_pass"] = all(v["pass"] for v in ck.values() if isinstance(v, dict) and "pass" in v)
    r["failed"] = [k for k, v in ck.items() if isinstance(v, dict) and v.get("pass") is False]
    plots(x, sr, score, out)
    r["images"] = ["spectrogram.png", "waveform.png"]
    (out / "verify.json").write_text(json.dumps(r, indent=1, default=_jsonable))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav")
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--determinism", help="json with {equal, sha256...} from a render-twice check")
    ap.add_argument("--same-as", help="another wav that must be byte-identical")
    ap.add_argument("--stems-dir", help="folder of <instrument>.wav stems: onsets are then measured on the instrument's own stem")
    a = ap.parse_args()
    score = json.loads(Path(a.score).read_text())
    det = json.loads(Path(a.determinism).read_text()) if a.determinism else None
    r = verify(a.wav, score, a.out, det, a.same_as, a.stems_dir)
    for k, v in r["checks"].items():
        flag = "" if "pass" not in v else ("  PASS" if v["pass"] else "  FAIL")
        print(f"{k:26s} {json.dumps({kk: vv for kk, vv in v.items() if kk not in ('pass', 'sections', 'limit', 'note', 'by_type_median_max_abs_ms')})}{flag}")
    print("bar levels (K-weighted dB):", r.get("bar_levels_db"))
    print("failed:", r["failed"] or "none", "->", Path(a.out) / "verify.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
