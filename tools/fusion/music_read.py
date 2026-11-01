"""music_read.py - the "Gemma music reading" step: DSP beat analysis + a timeline image -> a structured musical reading (open weights, Gemma only).

    video_cuts(mp4)                         shot-change times of the music video (ffmpeg scene score, CPU, read-only)
    timeline_png(analysis, cuts, out)       ONE wide PNG (1600x900): log-mel spectrogram, rms / low / high curves, onset strength, beat ticks,
                                            downbeats + bar numbers, chord per bar, the video's cuts, a seconds axis
    prompt(analysis, cuts)                  the text for the model: every number (grid, bars, chords, energy, onsets, cuts) + the STRICT JSON reply spec
    parse_reply(text)                       reply text -> dict (fences / think blocks / trailing commas / truncation repaired) or None
    validate(obj, analysis, cuts)           -> (reading, report): every time SNAPPED to the beat grid, enums fixed, word slots held >= 1 beat; rejects
                                            (raises ReadingRejected) a reply with nothing usable

Reply (the model writes it, validate() makes it exact):
    sections        [{start,end,label intro|verse|hook|build|drop|break|outro,energy 0-1,why}]   contiguous, bar-snapped, cover 0..duration
    arc             one sentence
    hit_moments     [{t,beat_index,why}]        the strongest beats to LAND words on (t == beats[beat_index])
    word_slots      [{start,land,hold_until,suggest_len_words}]   a word lands ON a beat and holds >= 1 beat (user rule), one slot at a time
    prominent_words [{text,sung_at,land,hold_until,why}]   4-8 words / phrases picked from the timed lyrics (optional input), each on its own word slot
    cut_suggestions [{t,kind_hint,why}]         on downbeats; video_cut_within_s says an existing cut of the source video is there
    dropouts        [t]                          beats where the music drops out / thins
    notes           str

CLI:  python tools/fusion/music_read.py prepare --video V.mp4 --analysis A.json --out DIR [--audio A.wav]
      python tools/fusion/music_read.py finish  --analysis A.json --cuts DIR/cuts.json --reply DIR/reply_raw.txt --out DIR
Graph that runs the model on the box: tools/app/music_read_graph.py.  No model is called from this file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

LABELS = ("intro", "verse", "hook", "build", "drop", "break", "outro")
_LABEL_ALIAS = {"chorus": "hook", "refrain": "hook", "pre-chorus": "build", "prechorus": "build", "pre chorus": "build", "buildup": "build", "build-up": "build",
                "riser": "build", "bridge": "break", "breakdown": "break", "interlude": "break", "instrumental": "break", "bar": "verse", "rap": "verse",
                "verse 1": "verse", "verse 2": "verse", "outro/end": "outro", "end": "outro", "ending": "outro", "beginning": "intro", "start": "intro",
                "climax": "drop", "peak": "drop", "main": "hook"}
KIND_HINTS = ("hard_cut", "match_cut", "whip", "zoom", "flash", "dissolve", "pivot", "depth", "window", "glitch", "hold")


class ReadingRejected(ValueError):
    """the reply has nothing usable (not JSON, or no section and no word slot after repair)"""


# ------------------------------------------------------------------------------------------------- video cuts
def video_cuts(mp4: str | Path, threshold: float = 0.3, width: int = 320, ffmpeg: str = "ffmpeg") -> list[float]:
    """Shot-change times (seconds, ascending) of a video: ffmpeg `scale -> select='gt(scene,T)' -> showinfo`, parsing `pts_time`. Read-only, CPU, one pass.
    The frames are scaled to `width` first (the scene score of a 1080x1920 clip is the same shape at 320 wide, ~10x faster)."""
    vf = f"scale={int(width)}:-2,select='gt(scene,{float(threshold)})',showinfo"
    p = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", str(mp4), "-an", "-vf", vf, "-f", "null", "-"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {mp4}: {p.stderr[-400:]}")
    out = [float(m.group(1)) for m in re.finditer(r"pts_time:\s*([0-9.]+)", p.stderr)]
    return sorted({round(t, 3) for t in out})


# ------------------------------------------------------------------------------------------------- numbers from the analysis
def _extend(times, step: float, dur: float, margin: float) -> np.ndarray:
    t = [float(x) for x in times]
    while t and t[-1] + step < dur - margin:
        t.append(round(t[-1] + step, 4))
    return np.asarray(t, dtype=float)


def _beats(a: dict) -> np.ndarray:
    """the DSP beat grid, continued at the median beat period to the end of the track (the tracker can stop 1-2 s early: 21.98 of 24 s on the test clip clip)"""
    b = np.asarray(a["beats"], dtype=float)
    if len(b) < 2:
        return b
    return _extend(b, float(np.median(np.diff(b))), float(a["duration"]), 0.2)


def _downbeats(a: dict) -> np.ndarray:
    """the DSP downbeats, continued every bar_len to the end of the track"""
    d = np.asarray(a["downbeats"], dtype=float)
    if len(d) < 1:
        return d
    step = float(np.median(np.diff(d))) if len(d) > 1 else float(a.get("bar_len") or 4 * 60.0 / float(a["bpm"]))
    return _extend(d, step, float(a["duration"]), 0.2)


def beat_period(a: dict) -> float:
    b = _beats(a)
    return float(np.median(np.diff(b))) if len(b) > 1 else 60.0 / float(a.get("bpm") or 90.0)


def _curve(a: dict, name: str) -> np.ndarray:
    return np.asarray(a["curves"][name], dtype=float)


def bar_stats(a: dict) -> list[dict]:
    """per bar: n, start, end, chord, and the mean rms / sub+low / mid / high (0-1, the curves are normalised) + onset max + number of onsets"""
    fps = float(a["curves"]["fps"])
    ons = np.asarray(a.get("onsets") or [], dtype=float)
    cv = {k: _curve(a, k) for k in ("rms", "sub", "low", "mid", "high", "onset")}
    out = []
    for b in a["bars"]:
        i0, i1 = int(round(b["start"] * fps)), max(int(round(b["end"] * fps)), int(round(b["start"] * fps)) + 1)
        sl = lambda k: cv[k][i0:i1]               # noqa: E731
        m = lambda k: float(np.mean(sl(k))) if len(sl(k)) else 0.0      # noqa: E731
        out.append(dict(n=int(b["n"]), start=float(b["start"]), end=float(b["end"]), chord=b.get("chord") or "?", rms_db=b.get("rms_db"),
                        rms=round(m("rms"), 2), low=round((m("sub") + m("low")) / 2, 2), mid=round(m("mid"), 2), high=round(m("high"), 2),
                        onset_max=round(float(np.max(sl("onset"))) if len(sl("onset")) else 0.0, 2),
                        n_onsets=int(((ons >= b["start"]) & (ons < b["end"])).sum())))
    return out


def top_onsets(a: dict, k: int = 24) -> list[dict]:
    """the k strongest onsets: time, strength (onset curve 0-1 at that time), nearest beat index and the offset to it (s)"""
    fps = float(a["curves"]["fps"])
    on = _curve(a, "onset")
    beats = _beats(a)
    rows = []
    for t in a.get("onsets") or []:
        i = min(max(int(round(t * fps)), 0), len(on) - 1)
        lo, hi = max(i - 1, 0), min(i + 2, len(on))
        j = int(np.argmin(np.abs(beats - t)))
        rows.append(dict(t=round(float(t), 3), strength=round(float(on[lo:hi].max()), 2), beat_index=j, off_beat_s=round(float(t - beats[j]), 3)))
    rows.sort(key=lambda r: -r["strength"])
    return sorted(rows[:k], key=lambda r: r["t"])


# ------------------------------------------------------------------------------------------------- the timeline image
def _font_path() -> str | None:
    """SpaceGrotesk-Bold (liked.json, role text) - the plain readable sans of the liked set"""
    try:
        liked = json.loads((ROOT / "mograph" / "fonts" / "liked.json").read_text(encoding="utf-8"))
        for f in liked["fonts"]:
            if f["key"] == "SpaceGrotesk-Bold":
                p = ROOT / "mograph" / "fonts" / f["file"]
                if p.is_file():
                    return str(p)
    except Exception:       # noqa: BLE001
        pass
    return None


def _mel_fb(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)       # noqa: E731
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)       # noqa: E731
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - l) / max(c - l, 1e-9)
        dn = (r - freqs) / max(r - c, 1e-9)
        fb[i] = np.clip(np.minimum(up, dn), 0, None)
    return fb / np.maximum(fb.sum(1, keepdims=True), 1e-12) * 2.0


def log_mel(audio: str | Path, n_mels: int = 96, n_fft: int = 2048, hop: int = 512, fmin: float = 30.0, fmax: float = 16000.0):
    """-> (db [n_mels, frames] 0 dB = loudest bin, hop_s, mel centre frequencies Hz). numpy / scipy only."""
    import soundfile as sf
    from scipy import signal
    x, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    x = x.mean(1)
    _, _, Z = signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary="zeros", padded=True)
    S = np.abs(Z) ** 2
    mel = _mel_fb(sr, n_fft, n_mels, fmin, min(fmax, sr / 2 - 1)) @ S
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= db.max()
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)       # noqa: E731
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)       # noqa: E731
    cf = mel2hz(np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr / 2 - 1)), n_mels + 2))[1:-1]
    return db, hop / sr, cf


def timeline_png(analysis: dict, cuts: list[float], out: str | Path, audio: str | Path | None = None, size: tuple[int, int] = (1600, 900)) -> Path:
    """One wide PNG for the model. Rows (top to bottom), all on the same seconds axis: bar numbers + chords | log-mel spectrogram | rms (white), low (orange), high
    (cyan) | onset strength + beat ticks | video cuts (red). Gold vertical lines = downbeats, faint white = beats, red = cuts of the source video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    fp = _font_path()                                          # a liked face or an exception: matplotlib's own default font is never used (user 2026-10-30)
    if not fp:
        raise FileNotFoundError("music_read: the liked label face (SpaceGrotesk-Bold) is missing from mograph/fonts")
    fm.fontManager.addfont(fp)
    fam = fm.FontProperties(fname=fp).get_name()
    plt.rcParams.update({"font.family": fam, "axes.facecolor": "#0b0b10", "figure.facecolor": "#0b0b10", "text.color": "#f0f0f0",
                         "axes.edgecolor": "#555", "xtick.color": "#ddd", "ytick.color": "#ddd", "axes.labelcolor": "#ddd"})
    dur = float(analysis["duration"])
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.95, 3.3, 1.7, 1.2, 0.7], hspace=0.07, left=0.062, right=0.992, top=0.955, bottom=0.075)
    axs = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axs[1:]:
        ax.sharex(axs[0])
    a_bar, a_spec, a_cur, a_ons, a_cut = axs
    a_bar.set_xlim(0, dur)
    beats, downs = _beats(analysis), _downbeats(analysis)
    fps = float(analysis["curves"]["fps"])
    tc = (np.arange(int(analysis["curves"]["n"])) + 0.5) / fps - 0.5 / fps

    # spectrogram (real log-mel when the audio is given, else the four band curves as four rows)
    if audio is None:
        audio = analysis.get("file") if analysis.get("file") and Path(str(analysis["file"])).is_file() else None
    if audio:
        db, hop_s, cf = log_mel(audio)
        a_spec.imshow(np.clip(db, -70, 0), origin="lower", aspect="auto", extent=[0, db.shape[1] * hop_s, 0, db.shape[0]], cmap="magma", vmin=-70, vmax=0, interpolation="bilinear")
        tick_hz = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
        a_spec.set_yticks([int(np.argmin(np.abs(cf - h))) for h in tick_hz])
        a_spec.set_yticklabels([f"{h // 1000}k" if h >= 1000 else str(h) for h in tick_hz], fontsize=10)
        a_spec.set_ylabel("log-mel Hz", fontsize=10)
    else:
        rows = np.vstack([_curve(analysis, k) for k in ("high", "mid", "low", "sub")])
        a_spec.imshow(rows, origin="lower", aspect="auto", extent=[0, dur, 0, 4], cmap="magma", vmin=0, vmax=1)
        a_spec.set_yticks([0.5, 1.5, 2.5, 3.5])
        a_spec.set_yticklabels(["sub", "low", "mid", "high"], fontsize=10)
        a_spec.set_ylabel("bands (no audio)", fontsize=10)

    # curves
    a_cur.plot(tc, _curve(analysis, "rms"), color="#ffffff", lw=2.2, label="rms")
    a_cur.plot(tc, (_curve(analysis, "sub") + _curve(analysis, "low")) / 2, color="#ff9a2e", lw=1.4, label="low")
    a_cur.plot(tc, _curve(analysis, "high"), color="#38d6ff", lw=1.4, label="high")
    a_cur.set_ylim(0, 1.05)
    a_cur.set_yticks([0, 0.5, 1])
    a_cur.set_yticklabels(["0", ".5", "1"], fontsize=10)
    a_cur.legend(loc="upper right", ncol=3, fontsize=10, frameon=False)
    a_cur.set_ylabel("energy", fontsize=10)

    # onsets + beats
    on = _curve(analysis, "onset")
    a_ons.fill_between(tc, 0, on, color="#7cf29a", alpha=0.85, lw=0)
    a_ons.vlines(beats, 0, 0.35, color="#ffffff", lw=1.0, alpha=0.9)
    a_ons.vlines(downs, 0, 0.7, color="#ffd23c", lw=2.0)
    a_ons.set_ylim(0, 1.05)
    a_ons.set_yticks([])
    a_ons.set_ylabel("onsets\nbeats", fontsize=10)

    # cuts row
    a_cut.set_ylim(0, 1)
    a_cut.set_yticks([])
    a_cut.set_ylabel("video\ncuts", fontsize=10)
    for i, t in enumerate(cuts):
        a_cut.vlines(t, 0, 1, color="#ff4b4b", lw=1.8)
        a_cut.text(t + 0.08, 0.08 + 0.42 * (i % 2), f"{t:.1f}", color="#ff9a9a", fontsize=8, va="bottom", ha="left")
    a_cut.set_xlabel("seconds", fontsize=11)
    a_cut.set_xticks(np.arange(0, dur + 0.1, 2))
    a_cut.tick_params(axis="x", labelsize=10)

    # grid lines across all rows
    for ax in (a_bar, a_spec, a_cur, a_ons, a_cut):
        ax.vlines(downs, *ax.get_ylim(), color="#ffd23c", lw=1.4, alpha=0.9, zorder=3) if ax is not a_ons else None
        if ax in (a_spec, a_cur):
            ax.vlines(beats, *ax.get_ylim(), color="#ffffff", lw=0.5, alpha=0.18, zorder=2)
            for t in cuts:
                ax.axvline(t, color="#ff4b4b", lw=1.1, alpha=0.75, ls=(0, (3, 2)), zorder=4)
        if ax is not a_cut:
            plt.setp(ax.get_xticklabels(), visible=False)
    # bar strip
    a_bar.set_ylim(0, 1)
    a_bar.set_yticks([])
    a_bar.set_ylabel("bar\nchord", fontsize=10)
    for b in analysis["bars"]:
        mid = (b["start"] + b["end"]) / 2
        a_bar.text(mid, 0.93, str(b["n"]), ha="center", va="top", fontsize=14, color="#ffd23c")
        a_bar.text(mid, 0.12, str(b.get("chord") or "?"), ha="center", va="bottom", fontsize=11.5, color="#ffffff")
    fig.suptitle(f"{analysis['bpm']:.1f} bpm   {analysis['key']['estimate']}   {analysis.get('time_signature', '4/4')}   {dur:.1f} s   "
                 f"gold = downbeats / bar numbers   white ticks = beats   red = video cuts", fontsize=12, y=0.992, va="top")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------- the prompt
SCHEMA_TEXT = """{
  "sections": [{"start": 0.0, "end": 5.3, "label": "intro|verse|hook|build|drop|break|outro", "energy": 0.0-1.0, "why": "one short clause"}],
  "arc": "one sentence: how the energy moves through the whole track",
  "hit_moments": [{"t": <seconds, a beat time from the grid>, "beat_index": <index into the beat grid>, "why": "short"}],
  "word_slots": [{"start": <beat time>, "land": <beat time>, "hold_until": <beat time>, "suggest_len_words": 1-4}],
  "prominent_words": [{"text": "<word or short phrase copied from the lyrics>", "sung_at": <seconds, when it is sung>, "land": <beat time>, "hold_until": <beat time>, "why": "short"}],
  "cut_suggestions": [{"t": <downbeat time>, "kind_hint": "hard_cut|match_cut|whip|zoom|flash|dissolve|pivot|depth|window|glitch|hold", "why": "short"}],
  "dropouts": [<beat time>],
  "notes": "anything the editor should know"
}"""


def lyric_words(lyrics) -> list[dict]:
    """flatten a Whisper-style lyrics json ({segments:[{words:[{w,s,e,p}]}]}, or a plain list of such words / segments) -> [{w, s, e, p}] sorted by s"""
    if not lyrics:
        return []
    segs = lyrics.get("segments") if isinstance(lyrics, dict) else lyrics
    out = []
    for sg in segs or []:
        ws = sg.get("words") if isinstance(sg, dict) and "words" in sg else [sg]
        for w in ws or []:
            if not isinstance(w, dict):
                continue
            t = str(w.get("w", w.get("word", ""))).strip()
            s_, e_ = _num(w.get("s", w.get("start"))), _num(w.get("e", w.get("end")))
            if t and s_ is not None:
                out.append(dict(w=t, s=float(s_), e=float(e_ if e_ is not None else s_ + 0.2), p=_num(w.get("p", w.get("probability")))))
    return sorted(out, key=lambda x: x["s"])


def lyric_table(analysis: dict, words: list[dict]) -> list[str]:
    """one line per lyric word: text, sung start-end, nearest beat index (+ offset), rms / onset of the music while it is sung (0-1), how often the word occurs"""
    beats = _beats(analysis)
    fps = float(analysis["curves"]["fps"])
    rms, on = _curve(analysis, "rms"), _curve(analysis, "onset")
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())       # noqa: E731
    count: dict[str, int] = {}
    for w in words:
        count[norm(w["w"])] = count.get(norm(w["w"]), 0) + 1
    rows = []
    for w in words:
        j = int(np.argmin(np.abs(beats - w["s"])))
        i0, i1 = int(w["s"] * fps), max(int(w["e"] * fps) + 1, int(w["s"] * fps) + 1)
        rows.append(f"{w['w']} {w['s']:.2f}-{w['e']:.2f} b{j}{w['s'] - beats[j]:+.2f} rms {float(rms[i0:i1].mean()) if rms[i0:i1].size else 0:.2f} "
                    f"onset {float(on[i0:i1].max()) if on[i0:i1].size else 0:.2f}" + (f" x{count[norm(w['w'])]}" if count[norm(w['w'])] > 1 and len(norm(w['w'])) > 2 else ""))
    return rows


def prompt(analysis: dict, cuts: list[float], lyrics=None) -> str:
    """lyrics = the Whisper json (optional): the timed words go into the prompt and Gemma picks `prominent_words`"""
    words = lyric_words(lyrics)
    dur = float(analysis["duration"])
    n_sec = (max(3, round(dur / 8)), max(4, round(dur / 4)))
    n_hit = (max(5, round(dur / 4.5)), max(7, round(dur / 2.5)))
    n_slot = (max(6, round(dur / 3)), max(9, round(dur / 1.7)))
    n_cut = (max(5, round(dur / 4)), max(8, round(dur / 2.2)))
    beats = _beats(analysis)
    per = beat_period(analysis)
    bars = bar_stats(analysis)
    key = analysis["key"]["estimate"]
    L = []
    L.append("You are the music reader for a motion-graphics editor. The attached image is the timeline of one music track (the soundtrack of a short music video): "
             "top strip = bar numbers (gold) and the chord per bar; then the log-mel spectrogram; then rms (white), low band (orange) and high band (cyan) energy; "
             "then onset strength (green) with beat ticks (white) and downbeats (gold); bottom row = the shot cuts of the source video (red, seconds). "
             "Gold vertical lines are downbeats (bar starts). Use the image for the SHAPE of the energy and the numbers below for EXACT times.")
    L.append("")
    L.append("GRID (DSP, exact)")
    L.append(f"duration {analysis['duration']:.2f} s; tempo {analysis['bpm']:.2f} bpm (beat = {per:.3f} s; half/double candidates {analysis.get('bpm_candidates')}); "
             f"{analysis.get('time_signature', '4/4')}; key {key}; bar = {analysis['bar_len']:.3f} s; first downbeat {analysis['first_bar_start']:.3f} s; "
             f"{len(bars)} bars, {len(beats)} beats (indices 0..{len(beats) - 1}), {len(analysis['onsets'])} onsets.")
    n_dsp = len(analysis["beats"])
    L.append("Beat grid, index:time(s); a downbeat (bar start) is marked with *" + (f"; beats {n_dsp}..{len(beats) - 1} continue the tracked grid at the same tempo to the end of the clip:" if len(beats) > n_dsp else ":"))
    dset = [float(d) for d in _downbeats(analysis)]
    L.append(" ".join(f"{i}{'*' if any(abs(float(t) - d) < 0.02 for d in dset) else ''}:{t:.3f}" for i, t in enumerate(beats)))
    L.append("")
    L.append("BARS (n, start-end s, chord, mean energy 0-1 of rms / low / mid / high, number of onsets)")
    for b in bars:
        L.append(f"bar {b['n']:>2}: {b['start']:.2f}-{b['end']:.2f}  {b['chord']:<7} rms {b['rms']:.2f} low {b['low']:.2f} mid {b['mid']:.2f} high {b['high']:.2f}  onsets {b['n_onsets']}")
    L.append(f"(before bar 1: 0.00-{bars[0]['start']:.2f} s lead-in; after bar {bars[-1]['n']}: {bars[-1]['end']:.2f}-{analysis['duration']:.2f} s tail)")
    L.append("")
    L.append("STRONGEST ONSETS (time s, strength 0-1, nearest beat index, offset from that beat s)")
    L.append("; ".join(f"{r['t']:.3f} {r['strength']:.2f} b{r['beat_index']} {r['off_beat_s']:+.3f}" for r in top_onsets(analysis)))
    L.append("")
    if cuts:
        L.append(f"EXISTING CUTS OF THE SOURCE VIDEO ({len(cuts)}; seconds): " + ", ".join(f"{t:.2f}" for t in cuts))
        L.append("They show where the original editor cut; a cut that already sits on a downbeat confirms that downbeat is a structural moment.")
    else:
        L.append("The source video has no detected cuts.")
    L.append("")
    if words:
        L.append(f"LYRICS, timed (Whisper, {len(words)} words): word, sung start-end s, nearest beat index with offset in s, rms / onset strength (0-1) of the music while it is sung, xN = how many times the word is sung")
        L.append("; ".join(lyric_table(analysis, words)))
        L.append("")
    L.append("TASK: read this track as a musician and editor. Say where its sections are, where the energy rises and drops, which beats are the strongest to LAND a word on, "
             "where words can land and be held, where cuts belong, and where the music drops out.")
    L.append("RULES (the answer is machine-checked and every time is snapped to the grid anyway):")
    L.append(f"- Every time is in seconds and MUST be one of the beat-grid times above (t = beats[beat_index]). Section starts/ends and cut_suggestions use downbeats (*) only. "
             f"Section starts at 0.0 for the first and ends at {analysis['duration']:.2f} for the last; sections are contiguous, no gaps.")
    L.append("- label is one of: intro, verse, hook, build, drop, break, outro. energy is 0-1 from the rms/low/high numbers and the image. Do not invent sections that the energy does not support; "
             f"{n_sec[0]}-{n_sec[1]} sections for a track this long. 'why' cites a number or something visible (a chord change, the low band entering, onset density).")
    L.append(f"- hit_moments: {n_hit[0]}-{n_hit[1]} beats, the strongest of the track (downbeats of arrivals, strong low-band hits, section starts, chord changes). Most should be downbeats; a few may be off-bar beats.")
    L.append(f"- word_slots: {n_slot[0]}-{n_slot[1]} slots, in time order, spread over the whole track following the sections. A word LANDS on a beat (land) and holds at least one full beat "
             f"(hold_until >= land + {per:.3f} s); holds are usually 2-4 beats or to the next slot's land. start is the beat where its entrance begins (start <= land, at most one beat earlier). "
             "Slots never overlap: the next land is at or after the previous hold_until. suggest_len_words 1-4: short words on fast dense bars, longer phrases where it breathes.")
    L.append(f"- cut_suggestions: {n_cut[0]}-{n_cut[1]}, on downbeats, kind_hint from the list in the schema (hard_cut on arrivals; whip/zoom/flash on builds; dissolve/hold on breaks...). "
             "When a downbeat has an existing video cut within 0.15 s say so in why.")
    L.append("- dropouts: beats where the music thins out or stops (low band gone, onsets sparse); [] if none.")
    if words:
        L.append("- prominent_words: 4-8 PROMINENT words or short phrases (1-3 words) of the lyrics above, copied exactly as sung: hook words, words repeated (xN), stressed or loud words "
                 "(high rms / onset), image words that can be pictured. Not the full text. sung_at = when the word is sung (its start s). land = the beat nearest to sung_at (a word "
                 "lands ON a beat; if it is sung just before a beat, land on that beat; never more than one beat away from sung_at). hold_until >= land + 1 beat. Each prominent word "
                 "also gets its own entry in word_slots with the same land and hold_until, and prominent words never overlap each other.")
    else:
        L.append("- prominent_words: [] (no lyrics were given).")
    L.append("")
    L.append("Reply with ONE JSON object only, no prose, no markdown fence, exactly these keys:")
    L.append(SCHEMA_TEXT)
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- reply parsing
def _strip_wrappers(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.S)
    text = text.replace("```json", "```")
    if "```" in text:
        parts = text.split("```")
        # the longest fenced block that has a brace
        cand = [p for p in parts[1::2] if "{" in p]
        if cand:
            text = max(cand, key=len)
    return text.strip()


def _close_truncated(s: str) -> str | None:
    """a reply cut off by max_tokens: drop the unfinished tail element and close the brackets (string-aware)"""
    # cut back, comma by comma from the end, to the first head that parses once its brackets are closed
    for pos in [m.start() for m in re.finditer(",", s)][::-1]:
        head = s[:pos]
        stk, ins, es = [], False, False
        for ch in head:
            if ins:
                if es:
                    es = False
                elif ch == "\\":
                    es = True
                elif ch == '"':
                    ins = False
                continue
            if ch == '"':
                ins = True
            elif ch in "{[":
                stk.append(ch)
            elif ch in "}]" and stk:
                stk.pop()
        if ins:
            continue
        closed = head + "".join("}" if c == "{" else "]" for c in reversed(stk))
        try:
            json.loads(closed)
            return closed
        except ValueError:
            continue
    return None


def parse_reply(text: str) -> tuple[dict | None, list[str]]:
    """-> (object or None, notes). Tries: the text as is, the outermost {...}, trailing commas removed, single quotes, a truncated tail closed."""
    notes: list[str] = []
    s = _strip_wrappers(text)
    i = s.find("{")
    if i < 0:
        return None, ["no JSON object in the reply"]
    j = s.rfind("}")
    cands = [("as is", s[i:j + 1] if j > i else s[i:])]
    body = cands[0][1]
    nocomma = re.sub(r",\s*([}\]])", r"\1", body)
    if nocomma != body:
        cands.append(("trailing commas removed", nocomma))
    if "'" in body and '"' not in body[:20]:
        cands.append(("single quotes", re.sub(r",\s*([}\]])", r"\1", body.replace("'", '"'))))
    for how, c in cands:
        try:
            o = json.loads(c)
            if isinstance(o, dict):
                if how != "as is":
                    notes.append(f"parsed after: {how}")
                return o, notes
        except ValueError:
            continue
    t = _close_truncated(re.sub(r",\s*([}\]])", r"\1", s[i:]))
    if t:
        try:
            o = json.loads(t)
            if isinstance(o, dict):
                notes.append("reply was cut off: the unfinished tail was dropped and the brackets closed")
                return o, notes
        except ValueError:
            pass
    return None, ["not parseable as JSON"]


# ------------------------------------------------------------------------------------------------- snapping + validation
def snap(t, grid: np.ndarray) -> tuple[int, float]:
    """nearest grid time to t -> (index, time)"""
    g = np.asarray(grid, dtype=float)
    k = int(np.argmin(np.abs(g - float(t))))
    return k, float(g[k])


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _grids(a: dict):
    beats = _beats(a)
    downs = _downbeats(a)
    dur = float(a["duration"])
    return beats, downs, dur


def validate(obj, analysis: dict, cuts: list[float] | None = None, lyrics=None) -> tuple[dict, dict]:
    """Exact reading from the model's object. -> (reading, report). report = {repairs: [str], dropped: [str], counts: {...}, snapped_max_s}.
    Raises ReadingRejected when obj is not a dict or neither a section nor a word slot survives."""
    if not isinstance(obj, dict):
        raise ReadingRejected("reply is not a JSON object")
    cuts = list(cuts or [])
    beats, downs, dur = _grids(analysis)
    per = beat_period(analysis)
    rep: list[str] = []
    dropped: list[str] = []
    shifts: list[float] = []

    def sn(t, grid, what):
        k, g = snap(t, grid)
        d = abs(g - float(t))
        shifts.append(d)
        if d > 0.01:
            rep.append(f"{what}: {float(t):.3f} -> {g:.3f}")
        return k, g

    # sections: bar-snapped, contiguous, 0..duration
    secs = []
    for i, s in enumerate(obj.get("sections") or []):
        if not isinstance(s, dict) or _num(s.get("start")) is None or "label" not in s:
            dropped.append(f"sections[{i}] has no numeric start or no label (a reply cut off mid-item?)")
            continue
        lab = str(s.get("label", "")).strip().lower()
        if lab not in LABELS:
            fixed = _LABEL_ALIAS.get(lab) or next((v for k, v in _LABEL_ALIAS.items() if k in lab), None) or next((l for l in LABELS if l in lab), None)
            if fixed:
                rep.append(f"sections[{i}].label {lab!r} -> {fixed}")
                lab = fixed
            else:
                rep.append(f"sections[{i}].label {lab!r} unknown -> verse")
                lab = "verse"
        e = _num(s.get("energy"))
        if e is None:
            e = 0.5
            rep.append(f"sections[{i}].energy missing -> 0.5")
        elif not 0 <= e <= 1:
            rep.append(f"sections[{i}].energy {e} clamped to 0-1")
        secs.append(dict(start=float(s["start"]), label=lab, energy=round(min(max(e, 0.0), 1.0), 2), why=str(s.get("why", ""))[:240]))
    secs.sort(key=lambda s: s["start"])
    sections: list[dict] = []
    for s in secs:
        if s["start"] <= 0.3:
            st = 0.0
        else:
            _, st = sn(s["start"], downs, "section start")
        if sections and abs(sections[-1]["start"] - st) < 1e-6:
            dropped.append(f"section {s['label']} at {st:.2f} duplicates the previous start")
            continue
        sections.append(dict(start=round(st, 4), end=None, label=s["label"], energy=s["energy"], why=s["why"]))
    if sections:
        if sections[0]["start"] != 0.0:
            rep.append(f"first section start {sections[0]['start']:.3f} -> 0.0")
            sections[0]["start"] = 0.0
        for a_, b_ in zip(sections, sections[1:]):
            a_["end"] = b_["start"]
        sections[-1]["end"] = round(dur, 4)
        if sections[-1]["end"] <= sections[-1]["start"]:
            dropped.append("last section starts at/after the end of the track")
            sections.pop()
            if sections:
                sections[-1]["end"] = round(dur, 4)

    # hit moments: t and beat_index agree; always a beat time
    hits, seen = [], set()
    for i, h in enumerate(obj.get("hit_moments") or []):
        if not isinstance(h, dict):
            continue
        t, bi = _num(h.get("t")), _num(h.get("beat_index"))
        bi = int(bi) if bi is not None and 0 <= int(bi) < len(beats) else None
        if t is None and bi is None:
            dropped.append(f"hit_moments[{i}] has neither t nor beat_index")
            continue
        if t is not None:
            k, g = snap(t, beats)
            if bi is not None and bi != k and abs(g - float(t)) > 0.5 * per:
                rep.append(f"hit_moments[{i}] t {t:.3f} and beat_index {bi} disagree -> used beat_index")
                k = bi
            elif bi is not None and bi != k:
                rep.append(f"hit_moments[{i}] beat_index {bi} -> {k} (t wins)")
            shifts.append(abs(float(beats[k]) - float(t)))
            if abs(float(beats[k]) - float(t)) > 0.01:
                rep.append(f"hit_moments[{i}]: {t:.3f} -> {beats[k]:.3f}")
        else:
            k = bi
        if k in seen:
            dropped.append(f"hit_moments[{i}] repeats beat {k}")
            continue
        seen.add(k)
        hits.append(dict(t=round(float(beats[k]), 4), beat_index=int(k), downbeat=bool(np.min(np.abs(downs - beats[k])) < 0.03), why=str(h.get("why", ""))[:240]))
    hits.sort(key=lambda h: h["t"])

    # prominent words (lyrics): the text is checked against the lyrics when given; each becomes a word slot (below)
    lw = lyric_words(lyrics)
    norm = lambda w: re.sub(r"[^a-z0-9' ]", "", str(w).lower()).strip()       # noqa: E731
    prom: list[dict] = []
    for i, w in enumerate(obj.get("prominent_words") or []):
        if not isinstance(w, dict) or not str(w.get("text", "")).strip():
            dropped.append(f"prominent_words[{i}] has no text")
            continue
        text = str(w["text"]).strip()
        sung = _num(w.get("sung_at"))
        if lw:
            toks = norm(text).split()
            hits_ = [x for x in lw if toks and norm(x["w"]) == toks[0]]
            if not hits_:
                dropped.append(f"prominent_words[{i}] {text!r} is not in the lyrics")
                continue
            ref = min(hits_, key=lambda x: abs(x["s"] - (sung if sung is not None else x["s"])))
            if sung is None or abs(ref["s"] - sung) > 0.25:
                rep.append(f"prominent_words[{i}] {text!r} sung_at {sung} -> {ref['s']:.2f} (the lyric word)")
                sung = ref["s"]
        if sung is None:
            dropped.append(f"prominent_words[{i}] {text!r} has no sung_at")
            continue
        land = _num(w.get("land"))
        if land is None or abs(land - sung) > 1.5 * per:
            k0, g0 = snap(sung, beats)
            rep.append(f"prominent_words[{i}] {text!r} land {land} is not within a beat of sung_at {sung:.2f} -> {g0:.3f}")
            land = g0
        prom.append(dict(i=i, text=text[:60], sung_at=float(sung), land=float(land), hold_until=_num(w.get("hold_until")), why=str(w.get("why", ""))[:240]))

    # word slots: land on a beat, hold >= 1 beat, one word at a time; a prominent word always owns a slot
    raw = []
    for i, w in enumerate(obj.get("word_slots") or []):
        if not isinstance(w, dict):
            continue
        land = _num(w.get("land"))
        if land is None:
            land = _num(w.get("start"))
        if land is None:
            dropped.append(f"word_slots[{i}] has no land/start")
            continue
        raw.append(dict(tag=f"word_slots[{i}]", land=land, start=_num(w.get("start")), hold=_num(w.get("hold_until")), n=_num(w.get("suggest_len_words")), text=None))
    for pw in prom:
        raw.append(dict(tag=f"prominent_words[{pw['i']}]", land=pw["land"], start=None, hold=pw["hold_until"], n=min(len(pw["text"].split()), 4), text=pw))
    raw.sort(key=lambda r: (snap(r["land"], beats)[0], r["text"] is None))
    slots: list[dict] = []
    for r in raw:
        tag, land = r["tag"], r["land"]
        kl, gl = sn(land, beats, f"{tag}.land")
        if slots and kl - slots[-1]["land_index"] < 1:
            last = slots[-1]
            if last["pw"] is not None and r["text"] is None and kl == last["land_index"]:
                continue                    # the model's own slot for the same beat as a prominent word: the prominent word is that slot
            if r["text"] is not None and last["pw"] is None:
                dropped.append(f"{last['tag']} (beat {last['land_index']}) removed: a prominent word lands on beat {kl}")
                slots.pop()
            else:
                dropped.append(f"{tag} lands on beat {kl}, < 1 beat after the previous slot (beat {last['land_index']})")
                continue
        if kl >= len(beats) - 1 and gl + per > dur + 0.05:
            dropped.append(f"{tag} lands at {gl:.2f}: no beat left to hold")
            continue
        ks = kl
        if r["start"] is not None:
            ks, _ = sn(r["start"], beats, f"{tag}.start")
            if ks > kl:
                rep.append(f"{tag}.start after land -> start = land")
                ks = kl
            elif kl - ks > 1:
                rep.append(f"{tag}.start {kl - ks} beats before land -> 1 beat")
                ks = kl - 1
        hu = r["hold"]
        if hu is None:
            kh = kl + 2
            rep.append(f"{tag}.hold_until missing -> land + 2 beats")
        else:
            kh = snap(hu, beats)[0]
            if abs(float(beats[kh]) - hu) > 0.01:
                shifts.append(abs(float(beats[kh]) - hu))
                rep.append(f"{tag}.hold_until: {hu:.3f} -> {beats[kh]:.3f}")
        if kh < kl + 1:
            rep.append(f"{tag} held < 1 beat -> hold_until = land + 1 beat")
            kh = kl + 1
        kh = min(kh, len(beats) - 1)
        if kh < kl + 1:
            dropped.append(f"{tag} cannot hold 1 beat before the grid ends")
            continue
        n = 1 if r["n"] is None else int(min(max(round(r["n"]), 1), 4))
        if slots and slots[-1]["hold_index"] > kl:
            rep.append(f"word_slots: hold_until of beat {slots[-1]['land_index']} cut to the next land {gl:.3f}")
            slots[-1]["hold_index"] = kl             # lands are >= 1 beat apart, so the held beat of the previous slot stays
        slots.append(dict(tag=tag, start_index=ks, land_index=kl, hold_index=kh, n=n, pw=r["text"]))
    word_slots = [dict(start=round(float(beats[s["start_index"]]), 4), land=round(float(beats[s["land_index"]]), 4), hold_until=round(float(beats[s["hold_index"]]), 4),
                       land_beat_index=s["land_index"], hold_beats=s["hold_index"] - s["land_index"], suggest_len_words=s["n"],
                       **({"word": s["pw"]["text"]} if s["pw"] else {})) for s in slots]
    prominent = []
    for s in slots:
        if s["pw"]:
            pw = s["pw"]
            prominent.append(dict(text=pw["text"], sung_at=round(pw["sung_at"], 3), land=round(float(beats[s["land_index"]]), 4), hold_until=round(float(beats[s["hold_index"]]), 4),
                                  land_beat_index=s["land_index"], why=pw["why"]))
    if lw and len(prominent) < 4:
        rep.append(f"only {len(prominent)} prominent words survived (4-8 asked)")

    # cut suggestions: downbeats
    cs, seen_d = [], set()
    for i, c in enumerate(obj.get("cut_suggestions") or []):
        if not isinstance(c, dict) or _num(c.get("t")) is None:
            dropped.append(f"cut_suggestions[{i}] has no numeric t")
            continue
        k, g = sn(c["t"], downs, f"cut_suggestions[{i}].t")
        if k in seen_d:
            dropped.append(f"cut_suggestions[{i}] repeats downbeat {k}")
            continue
        seen_d.add(k)
        kh = str(c.get("kind_hint", "")).strip().lower().replace(" ", "_").replace("-", "_")
        if kh not in KIND_HINTS:
            fixed = next((x for x in KIND_HINTS if x in kh or kh in x), None) if kh else None
            rep.append(f"cut_suggestions[{i}].kind_hint {kh!r} -> {fixed or 'hard_cut'}")
            kh = fixed or "hard_cut"
        near = min((abs(g - float(x)) for x in cuts), default=None)
        cs.append(dict(t=round(g, 4), bar=int(k + 1), kind_hint=kh, video_cut_within_s=(round(near, 3) if near is not None and near <= 0.15 else None), why=str(c.get("why", ""))[:240]))
    cs.sort(key=lambda c: c["t"])

    # dropouts: beats
    drops = []
    for i, d in enumerate(obj.get("dropouts") or []):
        v = _num(d.get("t") if isinstance(d, dict) else d)
        if v is None:
            dropped.append(f"dropouts[{i}] is not a time")
            continue
        _, g = sn(v, beats, f"dropouts[{i}]")
        if round(g, 4) not in drops:
            drops.append(round(g, 4))
    drops.sort()

    if not sections and not word_slots:
        raise ReadingRejected("no usable section and no usable word slot in the reply: " + "; ".join(dropped[:6]))
    reading = dict(duration=dur, bpm=float(analysis["bpm"]), key=analysis["key"]["estimate"], beat_s=round(per, 4), n_beats=len(beats),
                   sections=sections, arc=str(obj.get("arc", ""))[:600], hit_moments=hits, word_slots=word_slots, prominent_words=prominent, cut_suggestions=cs, dropouts=drops,
                   notes=str(obj.get("notes", ""))[:1200], video_cuts=[round(float(c), 3) for c in cuts])
    report = dict(repairs=rep, dropped=dropped, snapped_max_s=round(max(shifts), 4) if shifts else 0.0,
                  counts=dict(sections=len(sections), hit_moments=len(hits), word_slots=len(word_slots), prominent_words=len(prominent), cut_suggestions=len(cs), dropouts=len(drops)))
    return reading, report


def check_reading(reading: dict, analysis: dict) -> list[str]:
    """independent post-conditions of a validated reading (used by the tests and the CLI): [] = every rule holds"""
    beats, downs, dur = _grids(analysis)
    per = beat_period(analysis)
    bad = []
    on_beat = lambda t, g: bool(np.min(np.abs(g - t)) < 0.002)       # noqa: E731
    for s in reading["sections"]:
        if s["label"] not in LABELS:
            bad.append(f"label {s['label']}")
        if not (s["start"] == 0.0 or on_beat(s["start"], downs)):
            bad.append(f"section start {s['start']} not a downbeat")
    for a_, b_ in zip(reading["sections"], reading["sections"][1:]):
        if abs(a_["end"] - b_["start"]) > 1e-6:
            bad.append("sections not contiguous")
    for h in reading["hit_moments"]:
        if not on_beat(h["t"], beats) or abs(beats[h["beat_index"]] - h["t"]) > 0.002:
            bad.append(f"hit {h['t']} off the grid")
    last_hold = -1.0
    for w in reading["word_slots"]:
        if not (on_beat(w["land"], beats) and on_beat(w["start"], beats) and on_beat(w["hold_until"], beats)):
            bad.append(f"slot {w['land']} off the grid")
        if w["hold_until"] - w["land"] < per * 0.98:
            bad.append(f"slot {w['land']} held < 1 beat")
        if w["start"] > w["land"] + 1e-6:
            bad.append(f"slot {w['land']} start after land")
        if w["land"] < last_hold - 1e-6:
            bad.append(f"slot {w['land']} overlaps the previous hold")
        last_hold = w["hold_until"]
    slot_lands = {round(w["land"], 3) for w in reading["word_slots"]}
    for pw in reading.get("prominent_words", []):
        if not on_beat(pw["land"], beats) or pw["hold_until"] - pw["land"] < per * 0.98:
            bad.append(f"prominent word {pw['text']!r} off the grid or held < 1 beat")
        if round(pw["land"], 3) not in slot_lands:
            bad.append(f"prominent word {pw['text']!r} has no word slot")
        if abs(pw["land"] - pw["sung_at"]) > 1.5 * per:
            bad.append(f"prominent word {pw['text']!r} lands > 1.5 beats from when it is sung")
    for c in reading["cut_suggestions"]:
        if not on_beat(c["t"], downs):
            bad.append(f"cut {c['t']} not a downbeat")
    for d in reading["dropouts"]:
        if not on_beat(d, beats):
            bad.append(f"dropout {d} off the grid")
    return bad


# ------------------------------------------------------------------------------------------------- the comparison note
def section_energy_table(reading: dict, analysis: dict) -> list[dict]:
    """per section: mean rms / low / high of the DSP curves over [start, end), so Gemma's 'energy' can be set beside the measurement"""
    fps = float(analysis["curves"]["fps"])
    cv = {k: _curve(analysis, k) for k in ("rms", "sub", "low", "high", "onset")}
    rows = []
    for s in reading["sections"]:
        i0, i1 = int(round(s["start"] * fps)), max(int(round(s["end"] * fps)), int(round(s["start"] * fps)) + 1)
        f = lambda k: float(np.mean(cv[k][i0:i1])) if cv[k][i0:i1].size else 0.0       # noqa: E731
        rows.append(dict(start=s["start"], end=s["end"], label=s["label"], gemma_energy=s["energy"], rms=round(f("rms"), 2),
                         low=round((f("sub") + f("low")) / 2, 2), high=round(f("high"), 2), onset=round(f("onset"), 2)))
    return rows


def reading_md(reading: dict, analysis: dict, report: dict, stats: dict | None = None, dsp_s: float | None = None, title: str = "Music reading") -> str:
    """MUSIC_READ.md body: Gemma's sections beside the measured DSP energy, then hits / slots / prominent words / cuts, then the repairs the snapping made"""
    f = lambda t: f"{t:.2f}"       # noqa: E731
    L = [f"# {title}", "", f"{reading['duration']:.1f} s, {reading['bpm']:.1f} bpm, {reading['key']}, beat {reading['beat_s']} s. Arc: {reading['arc']}", ""]
    if stats or dsp_s is not None:
        bits = ([f"DSP {dsp_s:.1f} s"] if dsp_s is not None else []) + ([f"Gemma {stats['total_s']} s (load {stats.get('load_s')} s, prompt {stats.get('prompt_tokens')} tok at {stats.get('prompt_tok_s')} tok/s, "
                                                                        f"reply {stats.get('gen_tokens')} tok at {stats.get('gen_tok_s')} tok/s)"] if stats and "total_s" in stats else [])
        L += ["Timings: " + "; ".join(bits), ""]
    L += ["## Sections: Gemma vs the DSP curves", "", "| start | end | label | Gemma energy | DSP rms | DSP low | DSP high | DSP onset | why |", "|---|---|---|---|---|---|---|---|---|"]
    why = {(s["start"], s["end"]): s["why"] for s in reading["sections"]}
    for r in section_energy_table(reading, analysis):
        L.append(f"| {f(r['start'])} | {f(r['end'])} | {r['label']} | {r['gemma_energy']} | {r['rms']} | {r['low']} | {r['high']} | {r['onset']} | {why[(r['start'], r['end'])]} |")
    rows = section_energy_table(reading, analysis)
    if len(rows) > 2:
        g, d = np.array([r["gemma_energy"] for r in rows]), np.array([r["rms"] for r in rows])
        if g.std() > 0 and d.std() > 0:
            L += ["", f"Correlation of Gemma's section energy with the DSP rms of the same section: {float(np.corrcoef(g, d)[0, 1]):.2f} over {len(rows)} sections."]
    L += ["", "## Bars (DSP)", "", "| bar | start | chord | rms | low | high | onsets |", "|---|---|---|---|---|---|---|"]
    L += [f"| {b['n']} | {f(b['start'])} | {b['chord']} | {b['rms']} | {b['low']} | {b['high']} | {b['n_onsets']} |" for b in bar_stats(analysis)]
    L += ["", "## Hit moments", ""] + [f"- {f(h['t'])} s (beat {h['beat_index']}{', downbeat' if h['downbeat'] else ''}): {h['why']}" for h in reading["hit_moments"]]
    L += ["", "## Prominent words (lyrics)", ""] + [f"- \"{w['text']}\" sung {f(w['sung_at'])} s, lands {f(w['land'])} s (beat {w['land_beat_index']}), holds to {f(w['hold_until'])} s: {w['why']}"
                                                  for w in reading.get("prominent_words", [])]
    L += ["", f"## Word slots ({len(reading['word_slots'])})", "", "| land | start | hold until | beats held | words | word |", "|---|---|---|---|---|---|"]
    L += [f"| {f(w['land'])} | {f(w['start'])} | {f(w['hold_until'])} | {w['hold_beats']} | {w['suggest_len_words']} | {w.get('word', '')} |" for w in reading["word_slots"]]
    L += ["", "## Cut suggestions", ""] + [f"- {f(c['t'])} s (bar {c['bar']}) {c['kind_hint']}{' [video cut within ' + str(c['video_cut_within_s']) + ' s]' if c['video_cut_within_s'] is not None else ''}: {c['why']}"
                                         for c in reading["cut_suggestions"]]
    L += ["", f"Dropouts: {reading['dropouts']}", f"Video cuts: {reading['video_cuts']}", "", f"Notes: {reading['notes']}", "", "## What the snapping changed", "",
          f"max shift {report['snapped_max_s']} s; counts {report['counts']}", ""] + [f"- {x}" for x in report["repairs"]] + ["", "Dropped:", ""] + [f"- {x}" for x in report["dropped"]]
    L += ["", "Post-conditions (check_reading): " + (", ".join(report.get("check_reading", [])) or "all hold"), ""]
    return chr(10).join(L)


# ------------------------------------------------------------------------------------------------- CLI
def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare", help="video cuts + timeline.png + prompt.txt into --out")
    p1.add_argument("--video", required=True)
    p1.add_argument("--analysis", required=True)
    p1.add_argument("--audio", default=None)
    p1.add_argument("--out", required=True)
    p1.add_argument("--threshold", type=float, default=0.3)
    p1.add_argument("--lyrics", default=None, help="Whisper json {segments:[{words:[{w,s,e,p}]}]}, optional")
    p2 = sub.add_parser("finish", help="raw reply -> validated, snapped reading.json")
    p2.add_argument("--analysis", required=True)
    p2.add_argument("--cuts", required=True)
    p2.add_argument("--reply", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--lyrics", default=None)
    p2.add_argument("--stats", default=None, help="the MographGGUFVL stats json (optional, for MUSIC_READ.md)")
    p2.add_argument("--dsp-s", type=float, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    an = _load(a.analysis)
    if a.cmd == "prepare":
        t0 = time.time()
        cuts = video_cuts(a.video, a.threshold)
        t1 = time.time()
        (out / "cuts.json").write_text(json.dumps(cuts), encoding="utf-8")
        png = timeline_png(an, cuts, out / "timeline.png", audio=a.audio)
        t2 = time.time()
        txt = prompt(an, cuts, _load(a.lyrics) if a.lyrics and Path(a.lyrics).is_file() else None)
        (out / "prompt.txt").write_text(txt, encoding="utf-8")
        print(json.dumps(dict(cuts=len(cuts), video_cuts_s=round(t1 - t0, 2), timeline_s=round(t2 - t1, 2), png=str(png), prompt_chars=len(txt))))
        return 0
    cuts = _load(a.cuts)
    obj, notes = parse_reply(Path(a.reply).read_text(encoding="utf-8"))
    try:
        reading, report = validate(obj, an, cuts, _load(a.lyrics) if a.lyrics and Path(a.lyrics).is_file() else None)
    except ReadingRejected as e:
        (out / "reading_rejected.txt").write_text(f"{e}\nparse notes: {notes}\n", encoding="utf-8")
        print("REJECTED:", e)
        return 2
    report["parse_notes"] = notes
    report["check_reading"] = check_reading(reading, an)
    (out / "reading.json").write_text(json.dumps(reading, indent=1), encoding="utf-8")
    (out / "reading_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    st = _load(a.stats) if a.stats and Path(a.stats).is_file() else None
    (out / "MUSIC_READ.md").write_text(reading_md(reading, an, report, st, a.dsp_s), encoding="utf-8")
    print(json.dumps(report["counts"]), "repairs", len(report["repairs"]), "dropped", len(report["dropped"]), "check", report["check_reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
