"""t2m_music.py - the approved ACE-Step text-to-music route as pure numpy functions (no ComfyUI, no model).

Moved here from ``15_fusion/v004/make_music.py`` (which now imports them; that folder is gitignored, the app pack is not) so the
``MographMusicT2M*`` / ``MographTameHighs`` / ``MographMaster`` nodes and the runner share ONE implementation:

    ACE-Step 1.5 XL BASE text-to-music (no LM), duration = piece length + EXTRA_S
      -> cut_and_fade   cut to exactly the piece length, one-beat cosine fade (the nominal tempo is nudged by ``fit_bpm`` so the piece
                        is a whole number of bars: the cut lands on a bar when ACE keeps the tempo)
      -> track_beats    measure the real tempo / phase / downbeats on the raw song -> ``t2m_grid`` = the grid that drives the plan
      -> tame_highs     notch sustained 2-8 kHz tones ACE leaves (whistle / bell tones) + a gentle high shelf
      -> safe_master    true mono below 200 Hz, width capped until the mono fold-down loses <= 1.5 dB, then audio_cut.master_audio

Why: BUILT.md "Music" (user rejected skeleton + ace audio-to-audio as "circus / tinker toy"; never turbo).
"""
from __future__ import annotations

import re

import numpy as np

EXTRA_S = 4.0                 # seconds requested beyond the piece length (ACE ends its song early)
FADE_BEATS = 1.0              # cosine fade at the end of the cut
MIN_BPM, MAX_BPM = 50, 180

STYLE = "Warm, mellow, rounded, low-mid focused, smooth, no high-pitched lead."      # the g2 prompts of v004 (user: no tweets / bleeps)
STUDIO = "High fidelity studio recording, wide stereo, rich natural reverb, slow build."
# generated (not user-written) tags must not ask for toy / glassy timbres; the user's own music prompt is used as written
_SOFTEN = {"bright": "warm", "airy": "soft", "bouncy": "gentle", "sparkling": "soft", "glassy": "warm", "shimmering": "soft"}
_DROP = re.compile(r"\b(?:bells?|chimes?|glockenspiel|music box|harp|marimba|vibraphone|xylophone|whistl\w*|plucked synth hook|plucked|"
                   r"arpeggiated|staccato|chiptune|8-bit)\b[,]?\s*", re.I)
_MOOD_CLUB = ("techno", "hype", "playful")        # master preset "club" for these moods, "warm" for everything else


def _flux(x: np.ndarray, sr: int, lo: float = 60.0, hi: float = 16000.0, hop: int = 240):
    """band-limited spectral flux (log-magnitude, positive differences) on a 5 ms grid, 1024-sample window; local mean removed."""
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    f, t, Z = signal.stft(x, sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
    M = np.log1p(np.abs(Z[(f >= lo) & (f < hi)]) * 50.0)
    d = np.concatenate([[0.0], np.maximum(M[:, 1:] - M[:, :-1], 0.0).sum(axis=0)])
    d = np.maximum(d - uniform_filter1d(d, 100), 0.0)             # 0.5 s local mean out
    return t, d / (np.percentile(d, 99) + 1e-12)


def _chroma_change(x: np.ndarray, sr: int, beats, T: float) -> np.ndarray:
    """per beat: 1 - cosine(chroma of the beat before, chroma of the beat after) from a 100-2000 Hz pitch-class profile."""
    n = 8192
    w = np.hanning(n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f >= 100) & (f <= 2000)
    pc = np.round(12 * np.log2(f[m] / 440.0) + 69).astype(int) % 12

    def chroma(t0, t1):
        i0, i1 = int(max(t0, 0) * sr), int(min(t1, len(x) / sr) * sr)
        if i1 - i0 < n:
            return None
        segs = [x[i:i + n] * w for i in range(i0, i1 - n + 1, n // 2)]
        S = np.mean([np.abs(np.fft.rfft(g))[m] ** 2 for g in segs], axis=0)
        c = np.bincount(pc, weights=S, minlength=12)
        return c / (np.linalg.norm(c) + 1e-12)

    out = []
    for b in beats:
        a_, c_ = chroma(b - T, b - 0.02), chroma(b + 0.02, b + T)
        out.append(0.0 if a_ is None or c_ is None else 1.0 - float(a_ @ c_))
    return np.array(out)


def track_beats(x: np.ndarray, sr: int, bpm: float, meter: int = 4, span: float = 0.08) -> dict:
    """Beat-track audio whose tempo is only approximately known (ACE text-to-music): onset flux -> Ellis dynamic-programming beat
    track started at ``bpm`` (librosa.beat_track's algorithm) -> constant-tempo grid refined by a comb search over the flux
    (tempo within +-3 % of the DP tempo, any phase) -> downbeats = the meter-phase with most low-frequency (60-200 Hz) onset energy.
    x mono float. Returns bpm / t0 (first beat, s, in [0, T)) / residuals of flux peaks vs the fitted grid."""
    from scipy.ndimage import gaussian_filter1d
    t, env = _flux(x, sr, lo=250.0, hi=4000.0)         # mid band: the full band's hats / air pull the phase half a beat off (checked on the skeleton)
    dt = float(t[1] - t[0])
    envs = gaussian_filter1d(env, 2.0)                           # ~10 ms
    # --- Ellis DP (tightness 100 like librosa)
    P = 60.0 / bpm / dt
    n = len(envs)
    score = envs.copy()
    back = -np.ones(n, int)
    win = np.arange(-int(round(2 * P)), -int(round(P / 2)) + 1)
    pen = -100.0 * (np.log(-win / P) ** 2)
    for i in range(n):
        j = i + win
        ok = j >= 0
        if ok.any():
            c = score[j[ok]] + pen[ok]
            k = int(np.argmax(c))
            if c[k] > 0 or True:
                score[i] = envs[i] + c[k]
                back[i] = j[ok][k]
    i = int(np.argmax(score[-int(2 * P):]) + n - int(2 * P))
    bt = [i]
    while back[bt[-1]] >= 0:
        bt.append(int(back[bt[-1]]))
    dp = t[np.array(bt[::-1])]
    dp_iv = np.diff(dp)
    T_dp = float(np.median(dp_iv))
    # --- comb refinement
    L = len(x) / sr

    def comb(T, t0s):
        ks = np.arange(0, int((L - 0.2) / T) + 1)
        pos = t0s[:, None] + ks[None, :] * T
        okp = pos < L - 0.1
        v = np.interp(pos, t, envs)
        return (v * okp).sum(axis=1) / okp.sum(axis=1)

    best = (-1, None, None)
    for T in T_dp * np.linspace(0.97, 1.03, 241):
        t0s = np.arange(0, T, 0.004)
        sc = comb(T, t0s)
        j = int(np.argmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), float(T), float(t0s[j]))
    _, T, t0 = best
    # polish phase to 1 ms
    t0s = np.arange(t0 - 0.006, t0 + 0.006, 0.001)
    t0 = float(t0s[int(np.argmax(comb(T, t0s)))]) % T
    if t0 > T - 0.04:                                            # a beat a few ms before 0: it is beat 1 at t = 0
        t0 -= T
    beats = np.maximum(t0 + np.arange(0, int((L - t0) / T) + 1) * T, 0.0)
    beats = beats[beats <= L]
    # --- residuals: strongest flux peak within +-T/4 of each grid beat (only beats that have a real onset)
    thr = 0.25 * np.percentile(envs, 95)
    res = []
    for b in beats:
        m = (t > b - 0.04) & (t < b + 0.04)
        if m.any():
            jj = np.argmax(envs[m])
            if envs[m][jj] > thr:
                res.append(t[m][jj] - b)
    res = np.array(res)
    # --- downbeats: per meter phase, share of (low-band onset, mid-band onset, chroma change across the beat) - the phase that is
    # strongest on the most cues wins; ``downbeat_conf`` = winner / runner-up of the summed share (1.0 = no information)
    tl, lo = _flux(x, sr, lo=40.0, hi=200.0)

    def at(tt, v, bs):
        return np.array([v[(tt > b - 0.05) & (tt < b + 0.05)].max() if ((tt > b - 0.05) & (tt < b + 0.05)).any() else 0.0 for b in bs])

    cues = [at(tl, lo, beats), at(t, env, beats), _chroma_change(x, sr, beats, T)]
    share = np.zeros(meter)
    for v in cues:
        ph_ = np.array([v[k::meter].mean() if len(v[k::meter]) else 0.0 for k in range(meter)])
        share += ph_ / (ph_.sum() + 1e-12)
    order = np.sort(share)[::-1]
    d0 = int(np.argmax(share))
    ph = share
    # autocorrelation tempo, free (45-200 bpm), as a cross-check
    ev = envs - envs.mean()
    ac = np.correlate(ev, ev, "full")[len(ev) - 1:]
    lag = np.arange(len(ac)) * dt
    sel = (lag > 0.3) & (lag < 1.3)
    ac_bpm = float(60.0 / lag[sel][int(np.argmax(ac[sel]))])
    return dict(bpm=float(60.0 / T), beat_s=float(T), t0=float(t0), dp_bpm=float(60.0 / T_dp), dp_n=len(dp),
                dp_interval_std_ms=float(dp_iv.std() * 1000), autocorr_bpm=ac_bpm,
                n_grid_beats=int(len(beats)), n_with_onset=int(len(res)), resid_std_ms=float(res.std() * 1000),
                resid_mean_ms=float(res.mean() * 1000), resid_mean_abs_ms=float(np.abs(res).mean() * 1000),
                resid_max_abs_ms=float(np.abs(res).max() * 1000), downbeat_phase=d0,
                downbeat_conf=float(order[0] / (order[1] + 1e-9)), first_downbeat_s=float(t0 + d0 * T),
                grid_offset_vs_requested_ms=float(((t0 + 0.5 * 60 / bpm) % (60 / bpm) - 0.5 * 60 / bpm) * 1000))

def _long_spec(mono: np.ndarray, sr: int, n: int = 8192):
    """long-window magnitude (dB) frames, rows = frames of music only (frame rms > -50 dBFS)."""
    from scipy import signal
    f, t, Z = signal.stft(mono, sr, nperseg=n, noverlap=n // 2, boundary=None, padded=False)
    hop = n // 2
    rms = np.array([np.sqrt((mono[i * hop:i * hop + n] ** 2).mean()) for i in range(Z.shape[1])])
    Z = Z[:, rms > 10 ** (-50 / 20)]
    return f, 20 * np.log10(np.abs(Z) + 1e-9).T


def _local_median(db: np.ndarray, f: np.ndarray, width_hz: float = 1200.0) -> np.ndarray:
    from scipy.ndimage import median_filter
    w = max(3, int(width_hz / (f[1] - f[0])) | 1)
    return median_filter(db, size=(1, w), mode="nearest")


def find_tonal_peaks(x: np.ndarray, sr: int, lo: float = 2000.0, hi: float = 8000.0, thresh_db: float = 9.0, persist: float = 0.3,
                     min_mean_db: float = 6.0) -> list[dict]:
    """sustained narrowband peaks in lo..hi: a bin counts in a frame when it is > thresh_db above the local median envelope (+-600 Hz);
    a peak = a run of bins flagged in > persist of the music frames, kept when its mean excess over all music frames is >= min_mean_db
    (harmonics of a real instrument sit at 3-5 dB mean; the constant C-major tones ACE leaves at 2093 / 3141 / 4192 / 5291 Hz sit at 10-18).
    -> [{hz, excess_db (mean over the frames), persist}]"""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    f, db = _long_spec(mono, sr)
    if len(db) < 3:
        return []
    ex = db - _local_median(db, f)
    band = (f >= lo) & (f <= hi)
    flag = (ex > thresh_db)
    fl = flag.copy()
    fl[:, 1:] |= flag[:, :-1]
    fl[:, :-1] |= flag[:, 1:]                                   # +-1 bin of frequency jitter
    pers = fl.mean(axis=0) * band
    hit = pers > persist
    out, i = [], 0
    while i < len(f):
        if hit[i]:
            j = i
            while j + 1 < len(f) and hit[j + 1]:
                j += 1
            seg = slice(i, j + 1)
            k = i + int(np.argmax(ex[:, seg].mean(axis=0)))
            out.append(dict(hz=float(f[k]), excess_db=float(ex[:, k].mean()), persist=float(pers[k])))
            i = j + 1
        else:
            i += 1
    return [o for o in out if o["excess_db"] >= min_mean_db]


def _peaking(f0: float, q: float, gain_db: float, sr: int):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return b / a[0], a / a[0]


def _high_shelf(f0: float, gain_db: float, sr: int, S: float = 1.0):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    c, sq = np.cos(w), 2 * np.sqrt(A) * al
    b = np.array([A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)])
    a = np.array([(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq])
    return b / a[0], a / a[0]


def tame_highs(x: np.ndarray, sr: int, q: float = 10.0, max_cut_db: float = 18.0, shelf_db: float = -2.5, shelf_hz: float = 5000.0, passes: int = 2):
    """x (N, C). Notch every sustained narrowband peak between 2 and 8 kHz (Q ~ 10, cut = the peak's excess, at most 18 dB; 2 passes, since a
    9 dB cap left 14-19 dB whistles in v005 P16/P20/P21), then a gentle high shelf (-2.5 dB at 5 kHz). Returns (y, peaks_found_first_pass)."""
    from scipy import signal
    peaks = find_tonal_peaks(x, sr)
    y = x.copy()
    for i in range(int(passes)):
        for pk in (peaks if i == 0 else find_tonal_peaks(y, sr)):
            b, a = _peaking(pk["hz"], q, -min(max_cut_db, pk["excess_db"]), sr)
            y = signal.lfilter(b, a, y, axis=0)
    b, a = _high_shelf(shelf_hz, shelf_db, sr)
    return signal.lfilter(b, a, y, axis=0), peaks

def end_gate(x: np.ndarray, sr: int, L: float, win: float = 0.5) -> dict:
    """END gate: 0.5 s RMS windows from 1.5 s to the piece length L must all stay within 20 dB of the median music level."""
    mono = x.mean(axis=1)
    w = int(win * sr)
    lv = np.array([20 * np.log10(np.sqrt((mono[i:i + w] ** 2).mean()) + 1e-9) for i in range(0, len(mono) - w + 1, w // 2)])
    tt = np.arange(len(lv)) * win / 2
    med = float(np.median(lv[lv > -50]))
    live = tt[lv > med - 20]
    chk = (tt >= 1.5) & (tt + win <= L + 1e-6)
    return dict(median_db=med, min_db_in_piece=float(lv[chk].min()), content_end_s=float(live.max() + win), passed=bool((lv[chk] > med - 20).all()))

def _bar_changes(mono: np.ndarray, sr: int, starts: list[float], bar: float) -> list[float]:
    """bar starts where the bar's pitch-class profile differs from the previous bar's (cosine < 0.9)."""
    n = 8192
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f >= 100) & (f <= 2000)
    pc = np.round(12 * np.log2(f[m] / 440.0) + 69).astype(int) % 12
    prof = []
    for b0 in starts:
        seg = mono[int(b0 * sr): int((b0 + bar) * sr)]
        segs = [seg[j:j + n] * np.hanning(n) for j in range(0, max(len(seg) - n, 1), n // 2) if len(seg[j:j + n]) == n]
        if not segs:
            prof.append(None)
            continue
        c = np.bincount(pc, weights=np.mean([np.abs(np.fft.rfft(g))[m] ** 2 for g in segs], axis=0), minlength=12)
        prof.append(c / (np.linalg.norm(c) + 1e-12))
    return [round(starts[i], 6) for i in range(1, len(starts))
            if prof[i] is not None and prof[i - 1] is not None and float(prof[i] @ prof[i - 1]) < 0.9]


def t2m_grid(length_s: float, fit: dict, wave: np.ndarray, sr: int, fade_s: float, *, key: str, mood: str, energy: float = 0.5,
             preset: str = "warm", music_prompt: str = "", bpm_hint: float | None = None, seed: int = 1, bar_tol: float = 0.04) -> dict:
    """tracked constant grid (``fit`` = track_beats output, ``wave`` (C, N) = the cut take) -> MOGRAPH_MUSIC with the keys
    P1x_s1_grid.json has (beats / downbeats run 4 s past the end like MusicCut's). ``curves`` are NOT filled here."""
    L = float(length_s)
    T, t0, meter = fit["beat_s"], fit["t0"], 4
    n_beats = int(np.ceil((L + 4.0 - t0) / T)) + 1
    allb = [round(max(t0 + k * T, 0.0), 6) for k in range(n_beats)]
    downs = allb[fit["downbeat_phase"]::meter]
    bar = T * meter
    first = downs[0]
    nb = int((L - first) / bar + bar_tol)                       # whole bars from the first downbeat; a bar short by <= bar_tol (100 bpm
                                                                # measured as 99.98) still counts: the cut is faded, 0.04 bar = ~100 ms
    bars = [dict(i=i, start=round(first + i * bar, 6), end=round(min(first + (i + 1) * bar, L), 6)) for i in range(nb)]
    chords = _bar_changes(wave.mean(axis=0), sr, [b["start"] for b in bars], bar)
    if nb >= 3:
        e0, e1 = round(first + bar, 6), round(first + (nb - 1) * bar, 6)
        secs = [dict(start=0.0, end=e0, role="intro", bar0=0, bar1=1), dict(start=e0, end=e1, role="drop", bar0=1, bar1=nb - 1),
                dict(start=e1, end=L, role="outro", bar0=nb - 1, bar1=nb)]
    else:
        secs = [dict(start=0.0, end=L, role="drop", bar0=0, bar1=max(nb, 1))]
    nfr = int(round(L * 24))
    ns = int(round(L * sr))
    sub = dict(n_bars=nb, bpm=fit["bpm"], meter=meter, n_frames=nfr, fps=24, length_s=L, sr=sr, n_samples=ns, bpm_hint=bpm_hint,
               in_window=True, beat_s=T, bar_s=bar, bar_len=bar, beats=allb, downbeats=downs, bars=bars, sections=secs, hits=[], hits_requested=[],
               key=key, key_ace=key, mood=mood, energy=energy, palette="t2m", seed=seed, chords=[], chord_changes=chords,
               backend="t2m", tail_fade_s=fade_s)
    music = dict(music_end=L, cut=L, nframes=nfr, fade=float(fade_s), bpm=float(fit["bpm"]), bar_len=bar, key=key, beats=allb, downbeats=downs,
                 chord_changes=chords, fps=24, src="t2m", meter=meter, n_bars=nb, sections=secs, hits=[], n_samples=ns, sr=sr, grid=sub,
                 mood=mood, palette="t2m", music_prompt=music_prompt, master_preset=preset)
    music["t2m_fit"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fit.items()}
    return music


# ---------------------------------------------------------------------------------------------------------------- request -> ACE text
def fit_bpm(bpm: float, length_s: float, meter: int = 4) -> int:
    """the whole-bar tempo nearest to ``bpm`` for a piece of ``length_s`` seconds (bars x meter x 60 / bpm = length), 50..180.
    16 s at 90 bpm = 6 bars: kept; 12 s at 90 bpm = 4.5 bars -> 100 bpm (5 bars)."""
    per_bar = 60.0 * meter / float(length_s)                      # bpm of ONE bar in the piece
    best = None
    for bars in range(1, int(MAX_BPM / per_bar) + 2):
        v = bars * per_bar
        if MIN_BPM <= v <= MAX_BPM:
            d = abs(v - float(bpm))
            if best is None or d < best[0]:
                best = (d, v)
    return int(round(best[1] if best else min(MAX_BPM, max(MIN_BPM, float(bpm)))))


def soften_tags(tags: str) -> str:
    """planner-written tags (mood table / Qwen brief) -> no toy / glassy words (BUILT.md Music; user: no high-pitched tweets)."""
    out = _DROP.sub("", tags or "")
    for k, v in _SOFTEN.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return re.sub(r"\s*,\s*(,\s*)+", ", ", re.sub(r"\s+", " ", out)).strip(" ,")


def caption(tags: str, bpm: int, keyscale: str, meter: int = 4, soften: bool = True) -> str:
    """prose caption in the style of the approved v004 prompts: the tags, tempo / time / key, the warm-mellow clause, studio clause.
    Any 'NNN bpm' in the tags is dropped (the grid's tempo wins); the no-vocals words stay."""
    t = re.sub(r"[,\s]*\b\d{2,3}(\.\d+)?\s*bpm\b", "", tags or "", flags=re.I).strip(" ,.")
    t = soften_tags(t) if soften else t
    return f"{t}. {int(bpm)} BPM in {meter}/4 time, {keyscale}. {STYLE} {STUDIO}"


def preset_for(mood) -> str:
    """master preset of a planner mood: warm (no presence / air boost) except the dance moods."""
    return "club" if str(mood or "").lower() in _MOOD_CLUB else "warm"


# ---------------------------------------------------------------------------------------------------------------- cut
def cut_and_fade(x: np.ndarray, sr: int, length_s: float, beat_s: float, fade_beats: float = FADE_BEATS) -> np.ndarray:
    """x (N, C) raw take -> (n, C) with n = round(length_s x sr) EXACTLY: shorter takes are padded with silence, a one-beat cosine
    fade ends at the last sample, 10 ms fade-in (click guard)."""
    n = int(round(float(length_s) * sr))
    y = (x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))).astype(np.float64).copy()
    nf = min(n, int(round(fade_beats * float(beat_s) * sr)))
    if nf > 0:
        y[n - nf:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(nf) / nf))[:, None]
    k = min(n, int(0.01 * sr))
    if k > 0:
        y[:k] *= np.linspace(0, 1, k)[:, None]
    return y


# ---------------------------------------------------------------------------------------------------------------- mono safety
def mono_stats(x: np.ndarray, sr: int) -> dict:
    """x (N, 2): broadband L/R correlation, correlation below 300 Hz, mono fold-down loss in dB (RMS of (L+R)/2 vs the stereo RMS)."""
    from scipy import signal
    sos = signal.butter(4, 300.0, "lowpass", fs=sr, output="sos")
    lo = signal.sosfiltfilt(sos, x, axis=0)
    mid = x.mean(axis=1)
    st = np.sqrt((x ** 2).mean())
    return dict(corr=float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), corr_lt300=float(np.corrcoef(lo[:, 0], lo[:, 1])[0, 1]),
                folddown_db=float(20 * np.log10(np.sqrt((mid ** 2).mean()) / (st + 1e-12) + 1e-12)))


def mono_below_lin(x: np.ndarray, sr: int, f0: float = 200.0, taps: int = 2401) -> np.ndarray:
    """x (N, 2): the part below f0 becomes true mono (L+R)/2; the rest is untouched. Linear-phase FIR lowpass, zero delay ('same'),
    high = x - low, so the split reconstructs exactly and nothing above f0 is smeared."""
    from scipy import signal
    h = signal.firwin(taps, f0, fs=sr, window=("kaiser", 8.0))
    low = np.stack([signal.fftconvolve(x[:, c], h, mode="same") for c in range(x.shape[1])], axis=1)
    return (x - low) + low.mean(axis=1, keepdims=True)


def side_scale(x: np.ndarray, k: float) -> np.ndarray:
    mid = x.mean(axis=1, keepdims=True)
    return mid + (x - mid) * k


def safe_master(y: np.ndarray, sr: int, preset: str, bpm: float, cut: float, width: float = 1.25, max_loss_db: float = 1.5, min_corr: float = 0.3):
    """y (N, 2) = cut + tamed take. Mono-safety, then audio_cut.master_audio:
    1. true mono below 200 Hz (linear-phase) before the master, and the master's own crossover at 200 Hz;
    2. if the broadband correlation is < min_corr after that, or the mastered file's fold-down loss is > max_loss_db: width 1.0 and the side channel
       is scaled until the mono fold-down loss of the MASTERED file is <= max_loss_db.
    Returns (master (2, n) float32, report dict with before / after mono_stats, width used, side factor k)."""
    from tools.fusion import audio_cut
    before = mono_stats(y, sr)
    y1 = mono_below_lin(y, sr)
    after_stage1 = mono_stats(y1, sr)
    w = float(width) if after_stage1["corr"] >= min_corr else 1.0
    k, it = 1.0, 0
    while True:
        m, rep = audio_cut.master_audio(side_scale(y1, k).T, sr, preset=preset, target_lufs=-14.0, width=w, bpm=bpm, cut=cut, mono_hz=200.0)
        st = mono_stats(m.T, sr)
        it += 1
        if w != 1.0 and (st["corr"] < min_corr or st["folddown_db"] < -max_loss_db):
            w = 1.0
            continue
        if w == 1.0 and st["folddown_db"] < -max_loss_db and it < 8:
            mid = m.mean(axis=0)
            pm, ps = float((mid ** 2).mean()), float((((m[0] - m[1]) / 2) ** 2).mean())
            # loss = 10 log10(pm / (pm + ps)) >= -max_loss  <=>  ps <= pm (10^(max/10) - 1)
            k *= float(np.sqrt(pm * (10 ** (max_loss_db / 10) - 1) / max(ps, 1e-18))) * 0.97
            continue
        break
    rep.update(width=w, side_k=k, iterations=it, mono_before=before, mono_after_stage1=after_stage1, mono_after=st)
    return m, rep
