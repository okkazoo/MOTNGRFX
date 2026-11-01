"""audio_cut.py - cut a song to a downbeat, fade it out, and build the MOGRAPH_MUSIC dict (pure numpy, no ComfyUI).

Lifted from ``piece_runner.Piece._audio`` (which stays as it is), with three changes for the app:
  * the cut is ON a downbeat (``ring`` defaults to 0; piece_runner adds a 0.5 s ring-out after the last downbeat),
  * ``music_end`` (last 50 ms window with a peak above 0.01) is a hard ceiling: ACE-Step 1.5 ends its music 1-10 s before the
    requested duration and pads the rest with silence (15_fusion/FINDINGS.md), so the cut never lands in the silent tail,
  * an optional ``target_len`` picks the downbeat nearest to the requested length (still <= music_end).

Everything works on arrays: wave (C, N) float, any sample rate; the result is stereo at the same rate (or resampled to
``out_sr``). The analysis dict is either ``tools/fusion/music_analysis.analyse`` output or a piece_runner ``analysis.json``.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np

FPS = 24
LIVE_THRESHOLD = 0.01       # peak above this = music (not padding)
LIVE_WINDOW_S = 0.05
MIN_CUT_S = 2.0


def as_stereo(wave, sr: int, out_sr: int | None = None) -> tuple[np.ndarray, int]:
    """(C, N) / (N,) / (B, C, N) float -> (2, N') float32 at out_sr (default: unchanged). Mono is duplicated, >2 ch dropped."""
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 1:
        w = w[None]
    if w.shape[0] > 8 and w.shape[1] <= 8:      # (N, C) given
        w = w.T
    if w.shape[0] == 1:
        w = np.repeat(w, 2, axis=0)
    w = np.nan_to_num(w[:2])
    if out_sr and int(out_sr) != int(sr):
        from math import gcd
        from scipy import signal
        g = gcd(int(out_sr), int(sr))
        w = signal.resample_poly(w, int(out_sr) // g, int(sr) // g, axis=-1).astype(np.float32)
        sr = int(out_sr)
    return np.ascontiguousarray(w), int(sr)


def music_end_of(stereo: np.ndarray, sr: int) -> float:
    """seconds up to the last 50 ms window whose peak exceeds 0.01 (the tail of an ACE render is digital silence)."""
    peak = np.abs(stereo).max(axis=0)
    win = max(1, int(LIVE_WINDOW_S * sr))
    n = len(peak) // win
    if n == 0:
        return len(peak) / sr
    env = peak[: n * win].reshape(n, win).max(axis=1)
    live = np.nonzero(env > LIVE_THRESHOLD)[0]
    return float((live[-1] + 1) * win / sr) if len(live) else len(peak) / sr


def extend_downbeats(downbeats, bar_len: float, music_end: float) -> list[float]:
    """the tracker stops early on tails: continue the bar grid until it passes the end of the music."""
    db = [float(d) for d in downbeats]
    bar = float(bar_len)
    if not db:
        db = [0.0]
    while db[-1] + bar < music_end + 0.1:
        db.append(db[-1] + bar)
    return db


def analyse_wave(wave, sr: int, bpm_hint: float | None = None, fps: int = FPS) -> dict:
    """tools/fusion/music_analysis.analyse on an in-memory wave (written to a temp wav first)."""
    import soundfile as sf
    from tools.fusion import music_analysis
    st, sr = as_stereo(wave, sr)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="mograph_ana_")
    os.close(fd)
    try:
        sf.write(path, st.T, sr, subtype="FLOAT")
        return music_analysis.analyse(path, bpm_hint, 4, fps)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _key_name(ana: dict) -> str:
    k = ana.get("key")
    if isinstance(k, dict):
        return str(k.get("estimate", ""))
    return str(k or "")


def pick_cut(downbeats, music_end: float, total: float, target_len: float | None = None) -> float:
    """the downbeat (<= music_end and <= the file length) nearest to ``target_len`` (None = the last one)."""
    ceiling = min(music_end, total) + 1e-6
    cand = [d for d in downbeats if MIN_CUT_S <= d <= ceiling]
    if not cand:
        cand = [d for d in downbeats if 0.0 < d <= ceiling]
    if not cand:
        return float(min(music_end, total))
    if target_len is None:
        return float(max(cand))
    return float(min(cand, key=lambda d: (abs(d - target_len), -d)))


LOOP_XFADE_MIN_S = 0.1      # musical loop join: ~1 beat, clamped
LOOP_XFADE_MAX_S = 0.5
LOOP_XFADE_NOGRID_S = 0.3   # no beat grid: the whole track loops with this crossfade


def _first_live_s(stereo: np.ndarray, sr: int) -> float:
    """seconds to the first 50 ms window whose peak exceeds 0.01 (end of any intro silence)."""
    peak = np.abs(stereo).max(axis=0)
    win = max(1, int(LIVE_WINDOW_S * sr))
    n = len(peak) // win
    if n == 0:
        return 0.0
    env = peak[: n * win].reshape(n, win).max(axis=1)
    live = np.nonzero(env > LIVE_THRESHOLD)[0]
    return float(live[0] * win / sr) if len(live) else 0.0


def _loop_span(stereo: np.ndarray, sr: int, ana: dict, db: list, bar: float, content_end: float):
    """-> (start, end) in s on the bar grid: the first downbeat after any intro silence .. the last downbeat <= the music end
    (whole bars, at least one); None when there is no usable grid."""
    if not ana.get("downbeats") or not math.isfinite(bar) or bar <= 0.2:
        return None
    t0 = _first_live_s(stereo, sr)
    starts = [d for d in db if d >= t0 - 0.05 and d < content_end]
    if not starts:
        return None
    s = float(starts[0])
    ends = [d for d in db if d <= content_end + 1e-6 and d - s >= max(0.9 * bar, 1.0)]
    return (s, float(max(ends))) if ends else None


def loop_to_length(st: np.ndarray, sr: int, n: int, s: float, e: float, xf: float, overlap: bool = False):
    """repeat st[:, s:e] after st[:, :e] until n samples; every join is an equal-power crossfade of ``xf`` s.
    grid join (overlap False): the repeat's first sample (the downbeat ``s``) plays exactly where the previous pass hit
    the downbeat ``e`` - the crossfade sits just BEFORE it (outgoing tail vs the repeat's own pre-roll), so the beat grid
    continues; with no pre-roll (s ~ 0) it is a 5 ms de-click instead.
    overlap join (no grid): the repeat starts ``xf`` early and overlaps the end of the previous pass.
    -> ((2, n) float32, [output sample where each repeat's start plays])."""
    S, E = int(round(s * sr)), min(int(round(e * sr)), st.shape[1])
    L = E - S
    xn = int(max(0, min(round(xf * sr), L // 2)))
    out = np.zeros((2, n), dtype=np.float32)
    m = min(E, n)
    out[:, :m] = st[:, :m]
    seams: list[int] = []
    pos = E - xn if overlap else E
    declick = max(1, int(0.005 * sr))
    while pos < n:
        seams.append(pos)
        if overlap:
            k = min(xn, n - pos)
            if k > 0:
                th = np.linspace(0.0, math.pi / 2, xn, dtype=np.float32)[:k]
                out[:, pos:pos + k] = out[:, pos:pos + k] * np.cos(th)[None] + st[:, S:S + k] * np.sin(th)[None]
            a = pos + xn
            b = min(pos + L, n)
            if b > a:
                out[:, a:b] = st[:, S + xn:S + xn + (b - a)]
            pos += L - xn
        else:
            pre = min(xn, S)
            if pre >= int(0.01 * sr):
                th = np.linspace(0.0, math.pi / 2, pre, dtype=np.float32)
                out[:, pos - pre:pos] = out[:, pos - pre:pos] * np.cos(th)[None] + st[:, S - pre:S] * np.sin(th)[None]
            else:
                d = min(declick, pos)
                out[:, pos - d:pos] *= np.linspace(1.0, 0.0, d, dtype=np.float32)[None]
            b = min(pos + L, n)
            out[:, pos:b] = st[:, S:S + (b - pos)]
            if pre < int(0.01 * sr):
                d = min(declick, b - pos)
                if d > 0:
                    out[:, pos:pos + d] *= np.linspace(0.0, 1.0, d, dtype=np.float32)[None]
            pos += L
    return out, seams


def _loop_times(times, s: float, e: float, seams_s: list, limit: float) -> list[float]:
    """a time list (beats / downbeats / chord changes - floats or {"t": ..} dicts) of the source -> the same events along the looped output."""
    def tm(x):
        return float(x["t"] if isinstance(x, dict) else x)

    def at(x, t):
        return dict(x, t=round(t, 4)) if isinstance(x, dict) else round(t, 6)

    out = [at(x, tm(x)) for x in times if tm(x) < e]
    for p in seams_s:
        out += [at(x, p + (tm(x) - s)) for x in times if s <= tm(x) < e and p + (tm(x) - s) <= limit]
    out.sort(key=tm)
    return [x for i, x in enumerate(out) if i == 0 or tm(x) != tm(out[i - 1])]


def _loop_src_time(t: float, s: float, seams_s: list) -> float:
    """output time -> the source time that plays there."""
    k = None
    for p in seams_s:
        if p <= t + 1e-9:
            k = p
    return t if k is None else s + (t - k)


def _try_loop(st: np.ndarray, sr: int, ana: dict, db: list, bar: float, m_end: float, c: float, n: int):
    """a track shorter than the Length: loop it musically (see ``loop_to_length``). -> dict or None (nothing to loop)."""
    total = st.shape[1] / sr
    content_end = min(m_end, total)
    if content_end < 0.2 or float(np.abs(st).max()) <= 1e-6:
        return None
    span = _loop_span(st, sr, ana, db, bar, content_end)
    bpm = float(ana.get("bpm") or 0.0)
    if span is not None:
        s, e = span
        xf = min(LOOP_XFADE_MAX_S, max(LOOP_XFADE_MIN_S, 60.0 / bpm if bpm > 0 else LOOP_XFADE_MAX_S))
        overlap = False
    else:
        s, e, xf, overlap = 0.0, content_end, LOOP_XFADE_NOGRID_S, True
    y, seams = loop_to_length(st, sr, n, s, e, xf, overlap)
    seams_s = [p / sr for p in seams]
    passes = len(seams) + 1
    print(f"[audio_cut] track {total:.1f} s < length {c:.1f} s: looped {passes}x at "
          + (f"downbeat {s:.2f} s" if span is not None else "no beat grid (whole track, 0.3 s crossfade)"))
    return dict(st=y, s=s, e=e, xf=xf, seams=seams_s, passes=passes, grid=span is not None)


def cut_to_downbeat(wave, sr: int, ana: dict, target_len: float | None = None, cut: float | None = None, fps: int = FPS,
                    fade: float = 0.5, ring: float = 0.0, out_sr: int | None = 48000, pad: bool = False):
    """-> (cut stereo float32 (2, n) at out_sr, music dict = MOGRAPH_MUSIC).

    cut       explicit length in s (overrides everything, still clamped to the file).
    target_len  nearest downbeat to this length, never beyond ``music_end``.
    neither   the last downbeat before the music ends.
    ring      seconds kept after the chosen downbeat (piece_runner used 0.5; 0 = the cut is on the downbeat).
    pad       with ``cut``: the result is EXACTLY ``cut`` seconds even when the file is shorter. A shorter file is LOOPED, not
              padded with silence: it repeats from its first downbeat after any intro silence to its last whole bar, every join
              an equal-power crossfade at a downbeat (~1 beat, 0.1-0.5 s; no beat grid = the whole track, 0.3 s), then the
              end fade. The app's user-track route uses this (cut = the chosen Length): snapping to the nearest downbeat made
              an 8 s piece 9.07 s long (106 bpm, bar 2.26 s) and a 12 s piece 12.85 s (75 bpm). A file at least as long as
              ``cut`` is cut with the fade at the end, as before.
    """
    st, sr = as_stereo(wave, sr)
    total = st.shape[1] / sr
    m_end = music_end_of(st, sr)
    bar = float(ana["bar_len"])
    db = extend_downbeats(ana["downbeats"], bar, m_end)
    beats_src = [float(b) for b in ana["beats"]]
    curves_src = ana.get("curves") or {}
    chords_src = list(ana.get("chord_changes", []))
    if cut is not None:
        c = float(cut) if pad else min(float(cut), total)
    else:
        c = min(pick_cut(db, m_end, total, target_len) + float(ring), total)
    c = c if (pad and cut is not None) else max(c, min(MIN_CUT_S, total))
    n = int(round(c * sr))
    loop = _try_loop(st, sr, ana, db, bar, m_end, c, n) if (pad and cut is not None and st.shape[1] < n) else None
    if loop is not None:                     # the looped audio is live to the end: grids / curves follow the repeats
        st, total, m_end = loop["st"], c, c
        s_l, e_l, seams_s = loop["s"], loop["e"], loop["seams"]
        db = _loop_times(db, s_l, e_l, seams_s, c + 4.0)
        beats_src = _loop_times(beats_src, s_l, e_l, seams_s, c + 4.0)
        chords_src = _loop_times(chords_src, s_l, e_l, seams_s, c + 4.0)
    n_real = min(n, st.shape[1])
    y = st[:, :n_real].copy()
    nf_fade = min(int(round(fade * sr)), n_real)
    if nf_fade > 0:
        y[:, n_real - nf_fade:] *= np.linspace(1.0, 0.0, nf_fade, dtype=np.float32)[None, :]
    if n > n_real:
        y = np.pad(y, ((0, 0), (0, n - n_real)))
    if out_sr and int(out_sr) != sr:
        y, _ = as_stereo(y, sr, out_sr)
    nf = int(math.ceil(c * fps - 1e-6))
    tail = db[-1] + bar
    db_ext = db + [tail + bar * i for i in range(int(4 / bar) + 2)]
    beats = list(beats_src)
    iv = float(np.median(np.diff(beats))) if len(beats) > 1 else 60.0 / float(ana["bpm"])
    if not beats:
        beats = [0.0]
    while beats[-1] < c + 4:
        beats.append(beats[-1] + iv)
    if loop is not None:                     # the curves of the source frame that plays at each output frame
        curves = {k: [v[min(len(v) - 1, max(0, int(round(_loop_src_time(f / fps, s_l, seams_s) * fps))))] for f in range(nf + 1)]
                  for k, v in curves_src.items() if isinstance(v, list) and v}
    else:
        curves = {k: v[:nf + 1] + ([0.0] * (nf + 1 - len(v)) if pad and len(v) < nf + 1 else [])      # pad: a short track's curves run on at 0 (silence)
                  for k, v in curves_src.items() if isinstance(v, list)}
    music = dict(music_end=round(m_end, 3), cut=round(c, 4), nframes=nf, fade=float(fade), bpm=float(ana["bpm"]),
                 bar_len=bar, key=_key_name(ana), beats=beats, downbeats=db_ext,
                 chord_changes=chords_src, curves=curves, fps=int(fps))
    if loop is not None:
        music["loop"] = dict(start=round(s_l, 4), end=round(e_l, 4), crossfade=round(loop["xf"], 4), passes=loop["passes"],
                             seams=[round(p, 4) for p in seams_s], grid=bool(loop["grid"]))
    return y, music


def cut_file(path, ana: dict | None = None, target_len: float | None = None, bpm_hint: float | None = None, **kw):
    """convenience for scripts / tests: read a file, analyse it when ``ana`` is None, cut. -> (wave, sr, music)."""
    import soundfile as sf
    x, sr = sf.read(str(Path(path)), always_2d=True, dtype="float32")
    wave = x.T
    if ana is None:
        ana = analyse_wave(wave, sr, bpm_hint)
    y, music = cut_to_downbeat(wave, sr, ana, target_len=target_len, **kw)
    return y, kw.get("out_sr", 48000) or sr, music


# =================================================================================================== mastering
# The v002 songs sounded "a bit tiny": nearly mono (L/R correlation 0.95), dull top end (1-3 % of the energy above 4 kHz),
# peaks over 0 dBTP and no common loudness. The chain: EQ (sub weight, less mud, presence, air) -> glue compressor ->
# mid/side width -> mono below 120 Hz -> gain + true-peak limiter to -14 LUFS / -1.0 dBTP.
MASTER_PRESETS = {
    "club": dict(eq=[("highpass", 28), ("peak", 55, 0.9, 1.0), ("peak", 320, 0.9, -2.0), ("peak", 3200, 0.8, 2.0),
                     ("highshelf", 9000, 0.7, 3.0)], glue=dict(thr_db=-16.0, ratio=2.0, knee_db=6.0)),
    "film": dict(eq=[("highpass", 25), ("lowshelf", 90, 0.7, 1.0), ("peak", 350, 0.9, -1.0), ("peak", 3000, 0.8, 1.5),
                     ("highshelf", 10000, 0.7, 2.0)], glue=dict(thr_db=-18.0, ratio=1.8, knee_db=6.0)),
    "neutral": dict(eq=[("highpass", 25), ("lowshelf", 100, 0.7, 1.0), ("peak", 3500, 0.8, 1.0),
                        ("highshelf", 10000, 0.7, 1.5)], glue=dict(thr_db=-18.0, ratio=1.5, knee_db=6.0)),
    # warm: calm / epic (soft piano + strings, orchestral). Body first: +2.5 dB low shelf at 120 Hz, the top rolled off a touch
    # (-1 dB at 9 kHz, no air boost: club / film lift it and the ACE output is already bright), bass mono below 150 Hz.
    "warm": dict(eq=[("highpass", 25), ("lowshelf", 120, 0.7, 2.5), ("highshelf", 9000, 0.7, -1.0)],
                 glue=dict(thr_db=-18.0, ratio=1.6, knee_db=6.0), mono_hz=150.0),
}
# mood -> preset for callers that know the mood (MographMaster's default stays "club")
MOOD_PRESET = {"calm": "warm", "epic": "warm"}


def preset_for_mood(mood, default: str = "club") -> str:
    """'warm' for calm / epic, `default` ('club') for every other mood."""
    return MOOD_PRESET.get(str(mood or "").lower(), default)


def master_audio(wave, sr: int, preset: str = "club", target_lufs: float = -14.0, width: float = 1.25,
                 bpm: float | None = None, tp_max: float = -1.0, mono_hz: float | None = None, cut: float | None = None):
    """(C, N) float -> (2, n) float32 mastered at the same rate, plus a report dict (lufs / true peak before and after, gain)."""
    from mograph import mastering as M
    from mograph.mix import loudness_lufs
    if preset not in MASTER_PRESETS:
        raise ValueError(f"preset {preset!r}; options: {sorted(MASTER_PRESETS)}")
    st, sr = as_stereo(wave, sr)
    if cut is not None:
        st = st[:, : int(round(float(cut) * sr))]
    x = st.astype(np.float64)
    before = dict(lufs=float(loudness_lufs(x, sr)), true_peak_dbtp=float(M.true_peak_db(x)))
    if not math.isfinite(before["lufs"]) or float(np.abs(x).max()) < 1e-6:          # silence: nothing to master
        return np.ascontiguousarray(st), dict(lufs=before["lufs"], true_peak_dbtp=before["true_peak_dbtp"], gain=1.0,
                                              preset=preset, width=float(width), before=before, iterations=0)
    p = MASTER_PRESETS[preset]
    y = M.eq(x, sr, p["eq"])
    release = float(np.clip(0.5 * 60.0 / float(bpm), 0.15, 0.40)) if bpm else 0.30
    y, gr = M.glue(y, sr, release=release, **p["glue"])
    y = M.widen(y, float(width))
    y = M.mono_below(y, sr, float(mono_hz if mono_hz is not None else p.get("mono_hz", 120.0)))
    y, rep = M.master(y, sr, target_lufs=float(target_lufs), tp_max=float(tp_max), tol=0.1, max_iter=16)
    rep = dict(rep, glue_max_gr_db=float(gr), preset=preset, width=float(width), before=before)
    return np.ascontiguousarray(y.astype(np.float32)), rep


def measure(wave, sr: int) -> dict:
    """loudness / true peak / stereo correlation / spectral balance (share of the 20 Hz-20 kHz power: low < 250 Hz,
    mid 250-4000 Hz, high > 4 kHz) of a stereo wave - the numbers in 16_app/AUDIO.md."""
    from scipy import signal
    from mograph import mastering as M
    from mograph.mix import loudness_lufs
    st, sr = as_stereo(wave, sr)
    x = st.astype(np.float64)
    f, pxx = signal.welch(x, sr, nperseg=8192, axis=-1)
    pxx = pxx.sum(0)
    tot = float(pxx[(f >= 20) & (f < 20000)].sum()) + 1e-30

    def band(lo, hi):
        return 100.0 * float(pxx[(f >= lo) & (f < hi)].sum()) / tot
    mid, side = x[0] + x[1], x[0] - x[1]
    corr = float(np.corrcoef(x[0], x[1])[0, 1]) if x.std() > 0 else 1.0
    return dict(lufs=round(float(loudness_lufs(x, sr)), 2), true_peak_dbtp=round(float(M.true_peak_db(x)), 2),
                correlation=round(corr, 3), low_pct=round(band(20, 250), 1), mid_pct=round(band(250, 4000), 1),
                high_pct=round(band(4000, 20000), 1), sub60_pct=round(band(20, 60), 1),
                side_to_mid_db=round(10 * math.log10(float((side ** 2).mean()) / (float((mid ** 2).mean()) + 1e-30) + 1e-30), 1),
                seconds=round(x.shape[1] / sr, 2))
