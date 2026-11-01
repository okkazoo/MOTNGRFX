"""score_skeleton.py - a composed timing skeleton of EXACTLY the requested length with a KNOWN beat grid (CPU, no ACE).

Why: ACE-Step text-to-music ends 1-10 s early with tail silence and the beat tracker misreads tempo on ACE / user audio
(P02: 79 bpm read for a 122 bpm song; 15_fusion/FINDINGS.md). Here the piece is composed on a grid chosen so that
``bars * meter * 60 / bpm == length`` exactly (the tempo is nudged inside a window, the audio is never trimmed to fit), so
the grid is ground truth. The wav is then restyled by ACE-Step 1.5 audio-to-audio at denoise 0.3-0.7 (nodes_fusion/music_score.py,
16_app/MUSIC.md), and the grid is kept as the analysis.

    from mograph.score_skeleton import compose
    wave, grid = compose(length_s=16, bpm=122, mood="driving", seed=3, hits=[8.0])
    wave.shape == (2, 16 * 48000)          # exactly; grid["bpm"] == 120.0, grid["n_bars"] == 8

Parts (as in 03_sound v006): drums, bass, chords, pad, arp. SoundFont via mograph.sampled (GeneralUser GS, FreePats upright) when
symusic and the bank are available; otherwise mograph.synth.render_soundtrack on the same grid. Hits (video cut times) are
snapped to the nearest beat: riser into the beat, crash + impact + chord / bass accent on it, drums drop out for half a beat before.

Deterministic: the same arguments give the same bytes (sampled path: fresh synthesizer per note; numpy parts: seeded rng).
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy import signal

from . import synth

__all__ = ["compose", "plan_grid", "curves_of", "music_dict", "fit_exact", "match_loudness", "MOODS", "SECTION_ROLES",
           "parse_key", "ace_keyscale", "parse_sections", "parse_hits"]

SR = 48000
FPS = 24
MOODS = synth.MOODS                                  # driving calm dark bright playful epic
MOOD_ALIAS = {"hype": "driving", "techno": "driving", "warm": "calm", "nostalgic": "calm", "luxury": "dark",
              "energetic": "driving", "ambient": "calm", "cinematic": "epic", "filmic": "epic", "happy": "bright"}
SECTION_ROLES = ("intro", "build", "drop", "break", "outro")
_ROLE_ALIAS = {"main": "drop", "chorus": "drop", "hook": "drop", "climax": "drop", "verse": "drop", "groove": "drop",
               "bridge": "break", "breakdown": "break", "interlude": "break", "ending": "outro", "end": "outro",
               "start": "intro", "rise": "build", "buildup": "build"}

# tempo the mood sits at when no hint is given; typical energy; keys (tonic, mode) picked by seed
_MOOD_BPM = {"driving": 124, "calm": 80, "dark": 100, "bright": 116, "playful": 104, "epic": 76}
_MOOD_ENERGY = {"driving": 0.85, "calm": 0.3, "dark": 0.6, "bright": 0.7, "playful": 0.55, "epic": 0.65}
_MOOD_KEYS = {
    "driving": [("A", "minor"), ("F", "minor"), ("D", "minor"), ("G", "minor")],
    "calm": [("D", "dorian"), ("F", "major"), ("C", "major"), ("A", "minor")],
    "dark": [("C", "minor"), ("F#", "minor"), ("B", "minor"), ("E", "phrygian")],
    "bright": [("C", "major"), ("G", "major"), ("D", "major"), ("A", "major")],
    "playful": [("F", "major"), ("Bb", "major"), ("G", "mixolydian"), ("C", "major")],
    "epic": [("D", "minor"), ("C", "minor"), ("E", "minor"), ("A", "minor")],
}
_MINORISH = ("minor", "aeolian", "dorian", "phrygian", "harmonic minor", "harmonic_minor")
_PALETTE = {"driving": "electronic", "dark": "electronic", "bright": "electronic", "playful": "electronic",
            "calm": "soft", "epic": "filmic"}
# chord-stab steps (16ths) of the electronic palette
_STAB = {"driving": "0.....0...0.....", "bright": "0..0..0.0..0..0.", "dark": "0.........0.....",
         "playful": "0..0....0..0....", "calm": "0...............", "epic": "0..............."}

# part levels per section role (0 = silent). The drum / bass / arp levels are scaled by the energy.
_ROLE_LV = {
    "intro": dict(kick=0.0, snare=0.0, hat=0.5, bass=0.0, chord=0.7, pad=1.0, arp=0.0),
    "build": dict(kick=0.8, snare=0.7, hat=0.8, bass=0.8, chord=0.9, pad=0.9, arp=0.6),
    "drop": dict(kick=1.0, snare=1.0, hat=1.0, bass=1.0, chord=1.0, pad=0.7, arp=1.0),
    "break": dict(kick=0.0, snare=0.0, hat=0.4, bass=0.0, chord=0.8, pad=1.0, arp=0.5),
    "outro": dict(kick=0.0, snare=0.0, hat=0.3, bass=0.5, chord=0.7, pad=1.0, arp=0.0),
}

# stem loudness (K-weighted LUFS over the active blocks) before the bus
_TARGET = dict(kick=-15.0, snare=-18.0, hat=-22.0, crash=-24.0, bass=-16.0, sub=-24.0, chord=-20.0, pad=-24.0, arp=-23.0,
               fx=-27.0)
# soft / filmic (the warm palettes) carry a fuller bed: pad -19 (was -24: it read tiny), plus string / low / timpani stems
_TARGET_WARM = dict(_TARGET, pad=-19.0, strings=-21.0, low=-21.0, timp=-22.0)

_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# =================================================================================================== key / grid
def parse_key(key, mood: str = "driving", seed: int = 0) -> tuple[int, str, str]:
    """'A minor' / 'F# dorian' / 'Bb' / None -> (tonic pitch class, mode name, tonic name). None = picked by mood + seed."""
    s = str(key or "").strip()
    if not s:
        tonic, mode = _MOOD_KEYS[mood][int(seed) % len(_MOOD_KEYS[mood])]
        s = f"{tonic} {mode}"
    m = re.match(r"^([A-Ga-g])([#b♯♭]?)\s*(.*)$", s)
    if not m:
        raise ValueError(f"key {key!r}: expected e.g. 'A minor', 'F# dorian', 'Bb major'")
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()]
    acc = m.group(2)
    pc += 1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0
    mode = (m.group(3) or "major").strip().lower().replace("_", " ")
    if mode not in synth._SCALES:
        raise ValueError(f"mode {mode!r}: options {sorted(set(synth._SCALES))}")
    return pc % 12, mode, _NOTE_NAMES[pc % 12]


def ace_keyscale(tonic: str, mode: str) -> str:
    """ACE-Step keyscale string ('D minor'); dorian / phrygian / aeolian / harmonic minor count as minor."""
    return f"{tonic} {'minor' if mode in _MINORISH else 'major'}"


def plan_grid(length_s: float | None = None, n_frames: int | None = None, fps: int = FPS, bpm: float | None = None,
              meter: int = 4, mood: str = "driving", window: float = 0.15, sr: int = SR) -> dict:
    """Pick bars + tempo so that ``bars * meter * 60 / bpm`` is exactly the piece length.

    The piece length is a whole number of frames (``n_frames / fps``) and ``n_samples = round(length * sr)``. Candidates are
    every bar count B with bpm_B = B * meter * 60 / length in 40..200 bpm; the choice is the one nearest to ``bpm`` (default:
    the mood's tempo) in log distance, with a small preference for even / phrase-sized bar counts. ``window`` (fraction) only
    decides the ``in_window`` flag: the nearest candidate is used either way (adjust the tempo, never trim).
    """
    if n_frames is None:
        if length_s is None:
            raise ValueError("give length_s or n_frames")
        n_frames = int(round(float(length_s) * fps))
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("length must be at least one frame")
    length = n_frames / float(fps)
    meter = int(meter)
    if not 2 <= meter <= 7:
        raise ValueError("meter must be 2..7 beats per bar")
    hint = float(bpm) if bpm else float(_MOOD_BPM.get(mood, 110))
    best = None
    for b in range(1, int(length * 200.0 / (60.0 * meter)) + 2):
        v = b * meter * 60.0 / length
        if not 40.0 <= v <= 200.0:
            continue
        cost = abs(math.log(v / hint)) + (0.04 if (b > 1 and b % 2) else 0.0) - (0.02 if b in (2, 4, 8, 12, 16, 24, 32) else 0.0)
        if best is None or cost < best[0]:
            best = (cost, b, v)
    if best is None:
        raise ValueError(f"no bar count gives 40..200 bpm in {length:.2f} s at {meter} beats per bar")
    _, bars, v = best
    return dict(n_bars=bars, bpm=v, meter=meter, n_frames=n_frames, fps=int(fps), length_s=length, sr=int(sr),
                n_samples=int(round(length * sr)), bpm_hint=hint, in_window=abs(v / hint - 1.0) <= window,
                beat_s=60.0 / v, bar_s=meter * 60.0 / v)


def _norm_sections(sections, bars: int, bar_s: float, mood: str, energy: float) -> list[dict]:
    """[(start, end, role)] in seconds -> contiguous sections on bar lines covering all bars. None = default shape."""
    marks = []                                                  # (bar0, role)
    if sections:
        for s in sections:
            if isinstance(s, dict):
                t0, t1, role = s.get("start", s.get("t", 0.0)), s.get("end"), s.get("role", s.get("label", "drop"))
            else:
                t0, t1, role = s[0], s[1], s[2]
            role = str(role).lower()
            role = _ROLE_ALIAS.get(role, role)
            if role not in SECTION_ROLES:
                raise ValueError(f"section role {role!r}: options {SECTION_ROLES}")
            marks.append((int(round(float(t0) / bar_s)), role))
        marks.sort()
        first = []
        for b0, role in marks:                                  # later duplicates of a bar win
            if first and first[-1][0] == b0:
                first[-1] = (b0, role)
            else:
                first.append((b0, role))
        marks = [(max(0, min(bars - 1, b0)), r) for b0, r in first]
        if marks[0][0] != 0:
            marks.insert(0, (0, "intro" if marks[0][1] != "intro" else marks[0][1]))
    else:
        n_out = 0 if bars < 3 else (2 if bars >= 12 else 1)
        n_in = 0 if bars < 2 else max(1, round(0.2 * bars)) if bars >= 3 else 0
        n_bu = 0 if bars < 4 else max(1, round(0.2 * bars))
        n_dr = bars - n_out - n_in - n_bu
        while n_dr < 1 and (n_in > 1 or n_bu > 1 or n_out > 1):
            if n_in > 1:
                n_in -= 1
            elif n_bu > 1:
                n_bu -= 1
            else:
                n_out -= 1
            n_dr = bars - n_out - n_in - n_bu
        if n_dr < 1:
            n_in = n_bu = 0
            n_dr = bars - n_out
        b = 0
        for role, k in (("intro", n_in), ("build", n_bu), ("drop", n_dr), ("outro", n_out)):
            if k > 0:
                marks.append((b, role))
                b += k
    out = []
    for i, (b0, role) in enumerate(marks):
        b1 = marks[i + 1][0] if i + 1 < len(marks) else bars
        if b1 > b0:
            out.append(dict(start=b0 * bar_s, end=b1 * bar_s, role=role, bar0=b0, bar1=b1))
    return out


# =================================================================================================== harmony
def _voice(pcs: list[int], prev: list[int] | None, lo: int = 55, hi: int = 80) -> list[int]:
    """One MIDI note per pitch class, each the placement in [lo, hi] nearest to the same voice of the previous chord."""
    target = prev or [lo + 5 + 4 * i for i in range(len(pcs))]
    out = []
    for pc, tg in zip(pcs, target):
        cand = [m for m in range(lo, hi + 1) if m % 12 == pc % 12]
        out.append(min(cand, key=lambda m: abs(m - tg)))
    return out


def _bass_root(pc: int) -> int:
    return 33 + (pc - 33) % 12                                  # A1 (33) .. G#2 (44)


def _chords(bars: int, mood: str, tonic: int, mode: str, seed: int) -> list[dict]:
    scale = synth._SCALES[mode]
    prog = synth._PROG[mood]
    hold = 2 if (mood in ("calm", "epic") and bars >= 4) else 1
    off = int(seed) % len(prog) if hold == 1 and bars > len(prog) else 0
    out, prev = [], None
    for b in range(bars):
        deg = prog[((b // hold) + off) % len(prog)]
        if b == bars - 1:
            deg = 0
        elif b == bars - 2 and bars >= 4 and hold == 1:
            deg = 4
        pcs = [(tonic + p) % 12 for p in synth._chord(scale, 0, deg, 4)]
        voiced = _voice(pcs, prev)
        prev = voiced
        out.append(dict(bar=b, degree=int(deg), root_pc=pcs[0], pcs=pcs, notes=voiced, bass=_bass_root(pcs[0]),
                        name=_NOTE_NAMES[pcs[0]]))
    return out


# =================================================================================================== curves / music dict
def curves_of(wave, sr: int, fps: int, nframes: int) -> dict:
    """Per-frame 0..1 curves (rms, sub, low, mid, high, onset), nframes + 1 values; value i is measured around t = i / fps."""
    x = np.asarray(wave, np.float32)
    mono = x.mean(axis=0) if x.ndim == 2 else x
    n_fft = 2048 if sr <= 24000 else 4096
    pad = n_fft // 2
    xp = np.pad(mono, (pad, pad + n_fft))
    idx = np.round(np.arange(nframes + 1) / fps * sr).astype(int)
    fr = np.stack([xp[i:i + n_fft] for i in idx]) * np.hanning(n_fft)[None, :]
    S = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)

    def band(lo, hi):
        return np.sqrt((S[:, (f >= lo) & (f < hi)] ** 2).sum(axis=1))

    def unit(v):
        top = float(np.percentile(v, 99)) if len(v) else 0.0
        return np.clip(v / top, 0.0, 1.0) if top > 1e-12 else np.zeros_like(v)

    L = np.log1p(30.0 * S)
    flux = np.maximum(L[1:] - L[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[flux[0] if len(flux) else 0.0], flux])
    c = dict(rms=unit(np.sqrt((fr ** 2).mean(axis=1))), sub=unit(band(20, 60)), low=unit(band(60, 250)),
             mid=unit(band(250, 4000)), high=unit(band(4000, 16000)), onset=unit(flux))
    return {k: [round(float(a), 4) for a in v] for k, v in c.items()}


def music_dict(grid: dict, wave=None, sr: int | None = None, fps: int | None = None, src: str = "score",
               fade: float = 0.0) -> dict:
    """grid (compose) -> MOGRAPH_MUSIC with the keys MusicCut produces (cut, nframes, bpm, bar_len, key, beats, downbeats,
    chord_changes, curves, music_end, fade, fps, src); beats / downbeats run 4 s past the end like MusicCut's. Extras: meter,
    n_bars, sections, hits, n_samples, sr, grid. ``wave`` fills ``curves`` (else empty lists)."""
    fps = int(fps or grid["fps"])
    n = int(grid["n_frames"])
    L = float(grid["length_s"])
    beat, bar = float(grid["beat_s"]), float(grid["bar_len"])
    beats = [round(k * beat, 6) for k in range(int(math.ceil((L + 4.0) / beat)) + 1)]
    downbeats = [round(k * bar, 6) for k in range(int(math.ceil((L + 4.0) / bar)) + 1)]
    curves = curves_of(wave, sr or grid["sr"], fps, n) if wave is not None else {}
    return dict(music_end=round(L, 3), cut=round(L, 4), nframes=n, fade=float(fade), bpm=float(grid["bpm"]), bar_len=bar,
                key=grid["key"], beats=beats, downbeats=downbeats, chord_changes=list(grid["chord_changes"]), curves=curves,
                fps=fps, src=src, meter=int(grid["meter"]), n_bars=int(grid["n_bars"]), sections=list(grid["sections"]),
                hits=list(grid["hits"]), n_samples=int(grid["n_samples"]), sr=int(grid["sr"]), grid=grid)


# =================================================================================================== arrangement
def _tier(level: float) -> int:
    return 0 if level < 0.55 else (1 if level < 0.9 else 2)


def _rng(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)) % (2 ** 63), int(tag)])


def _arrange(g: dict, mood: str, palette: str, energy: float, tonic: int, mode: str, chords: list[dict], seed: int):
    """Events per part (sampled.Ev lists) + fx list. All times come from the grid indices (bar_s, beat_s), never accumulated."""
    from .sampled import Ev
    beat, bar, meter, bars = g["beat_s"], g["bar_s"], g["meter"], g["n_bars"]
    step_s = beat / 4.0
    n_steps = 4 * meter
    L = g["length_s"]
    secs = g["sections"]
    hits = g["hits"]
    dr = synth._DRUMS[mood]
    bpat = synth._BASS[mood]
    arp_idx = synth._ARP[mood]
    stab = _STAB[mood]
    swing = {"playful": 0.3, "calm": 0.12}.get(mood, 0.0)
    dscale = float(np.interp(energy, [0.0, 1.0], [0.6, 1.1]))
    rng = _rng(seed, 7)
    P = {k: [] for k in ("kick", "snare", "clap", "hat", "oh", "crash", "bass", "chord", "pad", "arp", "strings", "low", "timp")}
    fx = []                                                     # (kind, t0, t1_or_t, gain)
    kick_times = []

    def sec_of(b):
        for s in secs:
            if s["bar0"] <= b < s["bar1"]:
                return s
        return secs[-1]

    def muted(t):                                               # half a beat of drum silence before a hit
        return any(h - beat / 2.0 - 1e-9 <= t < h - 1e-9 for h in hits)

    for b in range(bars):
        s = sec_of(b)
        role = s["role"]
        lv = dict(_ROLE_LV[role])
        if role == "intro" and energy >= 0.75:
            lv.update(kick=0.6, hat=0.7)
        if role == "intro" and palette == "electronic" and energy >= 0.5:
            lv.update(bass=0.4)
        if role == "intro" and palette != "electronic":          # soft / filmic intros keep a quiet bass + arp so they are not bare
            lv.update(bass=0.5, arp=0.5)
        if role == "build":
            ramp = 0.65 + 0.35 * (b - s["bar0"] + 1) / max(1, s["bar1"] - s["bar0"])
            for k in ("kick", "snare", "hat", "arp", "chord"):
                lv[k] *= ramp
        for k in ("kick", "snare", "hat"):
            lv[k] = min(1.0, lv[k] * dscale)
        lv["bass"] = min(1.0, lv["bass"] * (0.7 + 0.5 * energy))
        lv["arp"] = min(1.0, lv["arp"] * (0.5 + 0.8 * energy))
        final = b == bars - 1 and bars >= 3
        if final:
            lv.update(kick=0.0, snare=0.0, hat=0.0, arp=0.0, bass=max(lv["bass"], 0.7))
        t0 = b * bar
        ch = chords[b]
        tiers = {k: _tier(v) for k, v in lv.items()}
        # ---- drums
        for st in range(n_steps):
            t = t0 + st * step_s + (swing * step_s if st % 2 else 0.0)
            pos = st % 16
            if muted(t):
                continue
            acc = 1.0 if st % 4 == 0 else 0.0
            if lv["kick"] > 0.05 and dr["kick"][pos] != "." and tiers["kick"] >= int(dr["kick"][pos]):
                P["kick"].append(Ev(t, 0, 114 if st == 0 else 104, 0.2, anchor=True))
                kick_times.append(t)
            if lv["snare"] > 0.05:
                if dr["snare"][pos] != "." and tiers["snare"] >= int(dr["snare"][pos]):
                    P["snare"].append(Ev(t, 0, 98, 0.2, anchor=True))
                if dr["clap"][pos] != "." and tiers["snare"] >= int(dr["clap"][pos]):
                    P["clap"].append(Ev(t, 0, 88, 0.2, anchor=True))
            if lv["hat"] > 0.05:
                d = dr["ch"][pos]
                if d != "." and tiers["hat"] >= int(d):
                    P["hat"].append(Ev(t, 0, {0: 74, 1: 62, 2: 50}[int(d)] + (6 if st % 2 == 0 else 0), 0.1))
                d = dr["oh"][pos]
                if d != "." and tiers["hat"] >= int(d) and lv["hat"] >= 0.8:
                    P["oh"].append(Ev(t, 0, 76, 0.2))
        # snare roll in the last half bar of a build
        if role == "build" and b == s["bar1"] - 1 and energy >= 0.4:
            half = meter * 2
            for j in range(half):
                t = t0 + (n_steps - half + j) * step_s
                if not muted(t):
                    P["snare"].append(Ev(t, 0, 52 + 58 * (j + 1) / half, 0.1, anchor=True))
        # ---- bass
        if lv["bass"] > 0.05:
            if palette == "electronic":
                evs = []
                for st in range(n_steps):
                    d = bpat[st % 16]
                    if d != "." and tiers["bass"] >= int(d):
                        evs.append(st)
                for i, st in enumerate(evs):
                    nxt = evs[i + 1] if i + 1 < len(evs) else n_steps
                    midi = ch["bass"] + (12 if (st % 4 == 2 and tiers["bass"] >= 2 and st > 0) else 0)
                    P["bass"].append(Ev(t0 + st * step_s, midi, 100 if st == 0 else 84,
                                        min(nxt - st, 6) * step_s * 0.92, anchor=(st == 0)))
            else:
                hold_bars = 2 if (mood in ("calm", "epic") and chords[min(b + 1, bars - 1)]["root_pc"] == ch["root_pc"]
                                  and b % 2 == 0 and b + 1 < bars) else 1
                P["bass"].append(Ev(t0, ch["bass"], 92, bar * hold_bars * 0.96, anchor=True))
                if tiers["bass"] >= 2 and hold_bars == 1 and meter >= 4:
                    P["bass"].append(Ev(t0 + 2.5 * beat, ch["bass"] + 7, 70, beat * 1.2))
        # ---- chords
        if lv["chord"] > 0.05:
            cv = int(52 + 26 * lv["chord"])
            if palette == "electronic":
                for st in range(n_steps):
                    d = stab[st % 16]
                    if d != "." and tiers["chord"] >= int(d):
                        for k, m in enumerate(ch["notes"]):
                            P["chord"].append(Ev(t0 + st * step_s, m, cv, step_s * 2.2))
            elif palette == "soft":
                hits_at = [0.0] + ([2.0] if (tiers["chord"] >= 2 and meter >= 4) else [])
                for hb in hits_at:
                    for k, m in enumerate(ch["notes"]):                       # slow strum, low to high
                        P["chord"].append(Ev(t0 + hb * beat + k * 0.022, m, cv - 4 + 3 * k, (2.0 if hb == 0 else 1.8) * beat))
            else:                                                             # filmic: sustained strings, whole bar (or two)
                P["chord"].extend(Ev(t0, m, cv, bar * 0.98) for m in ch["notes"])
        # ---- pad (long chord, high voicing)
        warm = palette != "electronic"
        if lv["pad"] > 0.05 and (warm or b % (2 if mood in ("calm", "epic") else 1) == 0 or final):
            hb = 2 if (not warm and mood in ("calm", "epic") and not final and b + 1 < bars) else 1
            top = [m + 12 for m in ch["notes"][1:]]
            # warm palettes: a pad in EVERY bar, each overlapping the next a little so the bed never thins out
            P["pad"].extend(Ev(t0, m, int(40 + 36 * lv["pad"]), bar * hb * (1.03 if warm and not final else 0.99)) for m in top)
        if warm and lv["pad"] > 0.05:
            sv = int(46 + 30 * lv["pad"])
            if palette == "soft":                  # sustained string pad under the piano (low voicing), legato across bars
                P["strings"].extend(Ev(t0, m - 12, sv, bar * (1.03 if not final else 0.99)) for m in ch["notes"])
            if lv["bass"] > 0.05 or role in ("intro", "break", "outro"):   # cello (soft) / horns (filmic) on the root, an octave over the bass
                lo = ch["bass"] + 12
                for m in ([lo] if palette == "soft" else [lo, lo + 7]):
                    P["low"].append(Ev(t0, m, sv - 4, bar * (1.03 if not final else 0.99)))
            if palette == "filmic":                # soft timpani on the downbeat: every bar in build / drop, every second bar elsewhere
                if role in ("build", "drop") or b % 2 == 0 or final:
                    root = ch["bass"] + (12 if ch["bass"] < 38 else 0)
                    P["timp"].append(Ev(t0, root, 96 if final else (74 if role in ("build", "drop") else 58), 1.5 * beat, anchor=True))
        # ---- arp
        if lv["arp"] > 0.05:
            st_step = 4 if mood in ("calm", "epic") else 2
            tones = _voice(ch["pcs"], None, 67, 91)
            tones = sorted(tones)
            for j, st in enumerate(range(0, n_steps, st_step)):
                if tiers["arp"] == 0 and (j % 2):
                    continue
                m = tones[arp_idx[j % len(arp_idx)] % len(tones)]
                tt = t0 + st * step_s + (swing * step_s if (st % 4 == 2) else 0.0)
                if not muted(tt):
                    P["arp"].append(Ev(tt, m, int(54 + 14 * lv["arp"] + (8 if st % 4 == 0 else 0)), st_step * step_s * 1.4))
        # ---- end of a build -> riser; start of a drop -> crash + soft impact
        if role == "build" and b == s["bar1"] - 1:
            span = min((s["bar1"] - s["bar0"]) * bar, 2 * bar)
            fx.append(("riser", s["bar1"] * bar - span, s["bar1"] * bar, 0.55))
        if role == "drop" and b == s["bar0"] and b > 0 and not any(abs(h - t0) < beat for h in hits):
            P["crash"].append(Ev(t0, 0, 100, 1.0, anchor=True))
            fx.append(("impact", t0, t0, 0.45))
        if final:
            P["crash"].append(Ev(t0, 0, 96, 1.0, anchor=True))
            fx.append(("impact", t0, t0, 0.4))
            for m in ch["notes"]:
                P["chord"].append(Ev(t0, m, 92, bar * 0.98))
    # ---- hits: accent on the beat, riser into it
    for h in hits:
        b = min(bars - 1, int(h // bar + 1e-9))
        ch = chords[b]
        P["crash"].append(Ev(h, 0, 112, 1.0, anchor=True))
        P["kick"].append(Ev(h, 0, 122, 0.2, anchor=True))
        kick_times.append(h)
        P["bass"].append(Ev(h, ch["bass"], 110, beat * 1.5, anchor=True))
        P["chord"].extend(Ev(h, m, 100, beat * 1.4) for m in ch["notes"])
        fx.append(("impact", h, h, 0.8))
        rise = min(bar, h)
        if rise >= beat * 0.99:
            fx.append(("riser", h - rise, h, 0.6))
    return P, fx, kick_times


# =================================================================================================== rendering
def _to_n(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.shape[1] >= n:
        return x[:, :n]
    return np.concatenate([x, np.zeros((x.shape[0], n - x.shape[1]))], axis=1)


def _level(x: np.ndarray, sr: int, target: float) -> np.ndarray:
    from .mix import loudness_lufs
    if float(np.abs(x).max()) < 1e-7:
        return x
    lu = loudness_lufs(x, sr)
    if not math.isfinite(lu):
        return x
    return x * 10.0 ** (float(np.clip(target - lu, -30.0, 30.0)) / 20.0)


def _sub_kick(kick_times, n: int, sr: int) -> np.ndarray:
    out = np.zeros((2, n))
    m = int(0.32 * sr)
    t = np.arange(m) / sr
    ph = 2 * np.pi * np.cumsum(46.0 + 70.0 * np.exp(-t / 0.03)) / sr
    body = np.sin(ph) * np.exp(-t / 0.13)
    for tk in kick_times:
        s = int(round(tk * sr))
        e = min(n, s + m)
        if e > s:
            out[:, s:e] += body[: e - s][None, :]
    return out


def _sub_bass(evs, n: int, sr: int) -> np.ndarray:
    out = np.zeros((2, n))
    for e in evs:
        m = int(round((e.dur + 0.06) * sr))
        s = int(round(e.t * sr))
        if m <= 0 or s >= n:
            continue
        t = np.arange(m) / sr
        f = 440.0 * 2.0 ** ((e.midi - 69) / 12.0)
        env = np.minimum(1.0, t / 0.008) * np.where(t < e.dur, 1.0, np.exp(-(t - e.dur) / 0.02))
        w = np.tanh(1.6 * np.sin(2 * np.pi * f * t)) * env
        end = min(n, s + m)
        out[:, s:end] += w[: end - s][None, :] * (e.vel / 127.0)
    return out


def _render_sampled(P, g, palette: str, sr: int, n: int, seed: int) -> dict:
    from .sampled import humanise, render_part, resolve_bank
    resolve_bank("gm")
    L = n / sr
    spec = {   # per palette: kit, drum notes, programs (bank, program, release)
        "electronic": dict(kit=25, notes=dict(kick=35, snare=38, clap=39, hat=42, oh=46, crash=49),
                           bass=("gm", 38, None), chord=("gm", 4, 0.3), pad=("gm", 89, 0.6), arp=("gm", 11, 0.4)),
        "soft": dict(kit=8, notes=dict(kick=36, snare=37, clap=39, hat=42, oh=46, crash=49),
                     bass=("gm", 33, 0.2), chord=("upright", 0, 0.4), pad=("gm", 89, 0.8), arp=("gm", 46, 0.5),
                     strings=("gm", 49, 0.8), low=("gm", 42, 0.6)),
        "filmic": dict(kit=48, notes=dict(kick=36, snare=38, clap=39, hat=42, oh=46, crash=49),
                       bass=("gm", 43, 0.3), chord=("gm", 49, 0.5), pad=("gm", 52, 0.8), arp=("gm", 46, 0.6),
                       low=("gm", 60, 0.7), timp=("gm", 47, 0.5)),
    }[palette]
    from dataclasses import replace
    stems = {}
    for part in ("kick", "snare", "clap", "hat", "oh", "crash"):
        evs = [replace(e, midi=spec["notes"][part]) for e in P[part]]
        if not evs:
            continue
        evs = humanise(evs, ms=3, vel=5, seed=seed + len(part))
        stems[part] = render_part(evs, "gm", kit=spec["kit"], sr=sr, length=L)
    for part in ("bass", "chord", "pad", "arp", "strings", "low", "timp"):
        if not P[part] or part not in spec:
            continue
        bank, prog, rel = spec[part]
        evs = humanise(P[part], ms=4, vel=5, seed=seed + 11 + len(part))
        stems[part] = render_part(evs, bank, program=prog, sr=sr, release=rel, length=L,
                                  width=0.7 if bank == "upright" else 1.0)
    return {k: _to_n(v, n) for k, v in stems.items()}


def _pack_stems(stems: dict, P: dict, kick_times, palette: str, g: dict, sr: int, n: int) -> dict:
    """merge snare+clap / hat+oh, add the numpy sub parts, level every stem."""
    out = {}
    if "kick" in stems:
        out["kick"] = stems["kick"] + (_sub_kick(kick_times, n, sr) * 0.5 if palette == "electronic" else 0.0)
    for name, parts in (("snare", ("snare", "clap")), ("hat", ("hat", "oh")), ("crash", ("crash",)), ("bass", ("bass",)),
                        ("chord", ("chord",)), ("pad", ("pad",)), ("arp", ("arp",)), ("strings", ("strings",)),
                        ("low", ("low",)), ("timp", ("timp",))):
        have = [stems[p] for p in parts if p in stems]
        if have:
            out[name] = sum(have)
    if palette == "electronic" and P["bass"]:
        out["sub"] = _sub_bass(P["bass"], n, sr)
    return out


def _render_synth_backend(g: dict, mood: str, energy: float, tonic: int, mode: str, seed: int, sr: int) -> dict:
    """fallback: mograph.synth.render_soundtrack on our exact grid (drums, bass, pad, lead in one bed)."""
    L = g["length_s"]
    beats = [k * g["beat_s"] for k in range(g["n_bars"] * g["meter"])]
    plan = dict(duration=L, seed=int(seed), style=dict(energy=energy),
                audio=dict(bpm=g["bpm"], beats=beats, downbeats=[b * g["bar_s"] for b in range(g["n_bars"])], mood=mood,
                           key=_NOTE_NAMES[tonic], scale=mode,
                           sections=[dict(t=s["start"], label={"build": "intro", "drop": "drop"}.get(s["role"], s["role"]))
                                     for s in g["sections"]]))
    return {"bed": synth.render_soundtrack(plan, sr)}


def _fx_stem(fx, g: dict, sr: int, n: int, seed: int) -> np.ndarray:
    if not fx:
        return np.zeros((2, n))
    plan = dict(duration=n / sr, seed=int(seed),
                audio=dict(bpm=g["bpm"], sfx=[dict(type=k, t0=a, t1=b, t=a, gain=gn) for k, a, b, gn in fx]))
    return _to_n(synth.render_sfx(plan, sr), n)


def _section_gain(g: dict, n: int, sr: int) -> np.ndarray:
    """bus gain per section role (intro / break quieter, build rising into the drop, outro settling), 0.25 beat ramps."""
    depth = {"intro": -3.5, "build": -3.0, "drop": 0.0, "break": -4.0, "outro": -2.5}
    pts_t, pts_g = [], []
    for s in g["sections"]:
        d0 = depth[s["role"]]
        d1 = 0.0 if s["role"] == "build" else d0
        pts_t += [s["start"], s["end"]]
        pts_g += [d0, d1]
    t = np.arange(n) / sr
    # piecewise-linear in dB; the jump between sections is smoothed by a short moving average
    db = np.interp(t, pts_t, pts_g)
    k = max(1, int(0.25 * g["beat_s"] * sr))
    db = np.convolve(np.pad(db, (k // 2, k - k // 2 - 1), mode="edge"), np.ones(k) / k, mode="valid")
    return 10.0 ** (db / 20.0)


def _mix(stems: dict, fxs: np.ndarray, kick_times, palette: str, g: dict, sr: int, n: int) -> np.ndarray:
    from . import mastering as M
    from .mix import loudness_lufs
    tg = _TARGET if palette == "electronic" else _TARGET_WARM
    lv = {k: _level(v, sr, tg.get(k, -22.0)) for k, v in stems.items()}
    if "bed" in lv:                                             # fallback bed: one stem
        bus = lv["bed"] * 1.0
    else:
        eqs = {"kick": [("highpass", 30)], "bass": [("highpass", 35), ("peak", 250, 0.9, -2.0)],
               "chord": [("highpass", 160), ("peak", 5333, 8.0, -12.0), ("peak", 6261, 8.0, -12.0)] if palette == "electronic"
               else [("highpass", 140)],
               "pad": [("highpass", 200), ("lowpass", 9000)], "strings": [("highpass", 90), ("lowpass", 8000)],
               "low": [("highpass", 45)], "timp": [("highpass", 35)], "arp": [("highpass", 300)], "hat": [("highpass", 500)],
               "snare": [("highpass", 120)], "sub": [("highpass", 28), ("lowpass", 140)]}
        duck = None
        if palette == "electronic" and kick_times:
            duck = M.duck_curve(kick_times, 4.5, n, sr, tau=0.12)
        bus = np.zeros((2, n))
        wet = np.zeros((2, n))
        for k, v in lv.items():
            if k in eqs:
                v = M.eq(v, sr, eqs[k])
            if duck is not None and k in ("bass", "sub", "chord", "pad", "arp"):
                v = v * duck[None, :]
            bus += v
            if k in ("chord", "pad", "arp", "snare", "strings", "low"):
                wet += v
        decay, gain_db = {"electronic": (1.3, -11.0), "soft": (1.8, -8.0), "filmic": (3.0, -6.0)}[palette]
        bus += M.send_reverb(wet, sr, decay, 0.02) * 10.0 ** (gain_db / 20.0)
    bus = bus + _level(fxs, sr, _TARGET["fx"]) if float(np.abs(fxs).max()) > 1e-7 else bus
    bus = bus * _section_gain(g, n, sr)[None, :]
    bus = M.mono_below(bus, sr, 100.0)
    bus, _ = M.glue(bus, sr, thr_db=-20.0, ratio=1.8, knee_db=6.0, release=float(np.clip(0.5 * g["beat_s"], 0.12, 0.4)))
    bus, rep = M.master(bus, sr, target_lufs=-16.0, tp_max=-1.5, tol=0.2, max_iter=10)
    return bus


def compose(length_s: float | None = None, bpm: float | None = None, meter: int = 4, key: str | None = None,
            mood: str = "driving", energy: float | None = None, sections=None, hits=(), seed: int = 0,
            n_frames: int | None = None, fps: int = FPS, sr: int = SR, backend: str = "auto",
            tail_fade_s: float | None = None) -> tuple[np.ndarray, dict]:
    """-> (wave float32 (2, n_samples), grid).  ``n_samples == round(n_frames / fps * sr)`` always.

    length_s / n_frames   piece length; frames win (length_s is rounded to whole frames).
    bpm                   hint: the tempo actually used is the nearest ``bars * meter * 60 / length`` (see plan_grid).
    mood                  driving calm dark bright playful epic (app mood words such as hype / warm are mapped).
    energy                0..1; None = the mood's. Scales drum / bass / arp density and the section levels.
    sections              [(start_s, end_s, role)] with role intro build drop break outro; snapped to bar lines. None = default arc.
    hits                  video cut times in s; each is snapped to the nearest beat (``grid['hits']``, requested in ``hits_requested``).
    backend               'auto' (SoundFont when symusic + bank load, else synth) | 'sampled' | 'synth'.
    tail_fade_s           fade-out at the very end (default min(0.5 bar, 0.8 s)); the last bar rings out on a final chord.
    """
    mood = MOOD_ALIAS.get(str(mood).lower(), str(mood).lower())
    if mood not in MOODS:
        raise ValueError(f"mood {mood!r}: options {MOODS} (+ {sorted(MOOD_ALIAS)})")
    seed = int(seed) & 0x7FFFFFFF
    p = plan_grid(length_s, n_frames, fps, bpm, meter, mood, sr=sr)
    n, L, bars, beat, bar = p["n_samples"], p["length_s"], p["n_bars"], p["beat_s"], p["bar_s"]
    energy = float(np.clip(_MOOD_ENERGY[mood] if energy is None else energy, 0.0, 1.0))
    tonic, mode, tname = parse_key(key, mood, seed)
    palette = _PALETTE[mood]
    secs = _norm_sections(sections, bars, bar, mood, energy)
    nbeats = bars * p["meter"]
    beats = [k * beat for k in range(nbeats)]
    req, ks = [float(h) for h in (hits or ())], set()
    for h in req:
        if 0 <= h < L:
            ks.add(min(nbeats - 1, int(round(h / beat))))
    snapped = [k * beat for k in sorted(ks)]                    # the same expression as grid['beats'], so they compare equal
    chords = _chords(bars, mood, tonic, mode, seed)
    changes = [round(chords[b]["bar"] * bar, 6) for b in range(1, bars) if chords[b]["root_pc"] != chords[b - 1]["root_pc"]]
    grid = dict(p, bar_len=bar, n_bars=bars, beats=beats, downbeats=[b * bar for b in range(bars)],
                bars=[dict(i=b, start=b * bar, end=(b + 1) * bar) for b in range(bars)],
                sections=secs, hits=snapped, hits_requested=req, key=f"{tname} {mode}", key_ace=ace_keyscale(tname, mode),
                tonic=tname, mode=mode, mood=mood, energy=energy, palette=palette, seed=seed,
                chords=[dict(bar=c["bar"], degree=c["degree"], root=c["name"], pcs=c["pcs"]) for c in chords],
                chord_changes=changes, length_s=L)
    P, fx, kick_times = _arrange(grid, mood, palette, energy, tonic, mode, chords, seed)
    used = None
    if backend in ("auto", "sampled"):
        try:
            stems = _render_sampled(P, grid, palette, sr, n, seed)
            used = "sampled"
        except Exception as e:                                  # noqa: BLE001  (symusic / bank missing)
            if backend == "sampled":
                raise
            print(f"[score_skeleton] SoundFont path unavailable ({type(e).__name__}: {e}); using mograph.synth")
    if used is None:
        stems = _render_synth_backend(grid, mood, energy, tonic, mode, seed, sr)
        used = "synth"
    packed = stems if used == "synth" else _pack_stems(stems, P, kick_times, palette, grid, sr, n)
    packed = {k: _to_n(v, n) for k, v in packed.items()}
    y = _mix(packed, _fx_stem(fx, grid, sr, n, seed), kick_times, palette, grid, sr, n)
    # end: short fade-in (click guard) and fade-out; length is exactly n
    fo = int(round((min(0.5 * bar, 0.8) if tail_fade_s is None else float(tail_fade_s)) * sr))
    fo = max(0, min(fo, n))
    y = _to_n(y, n)
    if fo:
        y[:, n - fo:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(fo) / fo))[None, :]
    fi = min(n, int(0.002 * sr))
    y[:, :fi] *= np.linspace(0.0, 1.0, fi)[None, :]
    wave = np.ascontiguousarray(np.nan_to_num(y).astype(np.float32))
    grid["backend"] = used
    grid["tail_fade_s"] = fo / sr
    assert wave.shape == (2, n)
    return wave, grid


# =================================================================================================== fit the restyled audio to the grid
def _stereo(wave, sr: int, out_sr: int) -> np.ndarray:
    x = np.nan_to_num(np.asarray(wave, np.float32))
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x[None]
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = x[:2]
    if int(sr) != int(out_sr):
        from math import gcd
        g = gcd(int(out_sr), int(sr))
        x = signal.resample_poly(x, int(out_sr) // g, int(sr) // g, axis=-1).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _live_end(x: np.ndarray, sr: int, thr: float = 0.01) -> int:
    """samples up to the last 50 ms window whose peak is above ``thr`` (ACE pads a short render with digital silence)."""
    peak = np.abs(x).max(axis=0)
    win = max(1, int(0.05 * sr))
    k = len(peak) // win
    if k == 0:
        return len(peak)
    env = peak[: k * win].reshape(k, win).max(axis=1)
    live = np.nonzero(env > thr)[0]
    return int((live[-1] + 1) * win) if len(live) else len(peak)


def match_loudness(x: np.ndarray, ref: np.ndarray, sr: int, max_db: float = 12.0, ceiling_db: float = -1.0) -> tuple[np.ndarray, float]:
    """gain x to the integrated loudness of ref (clamped to +-max_db, sample peak kept under ceiling_db). -> (y, gain_db)."""
    from .mix import loudness_lufs
    a, b = loudness_lufs(x.astype(np.float64), sr), loudness_lufs(ref.astype(np.float64), sr)
    if not (math.isfinite(a) and math.isfinite(b)):
        return x, 0.0
    gdb = float(np.clip(b - a, -max_db, max_db))
    y = x * np.float32(10.0 ** (gdb / 20.0))
    pk = float(np.abs(y).max())
    cap = 10.0 ** (ceiling_db / 20.0)
    if pk > cap:
        y = y * np.float32(cap / pk)
        gdb += 20.0 * math.log10(cap / pk)
    return np.ascontiguousarray(y, dtype=np.float32), gdb


def fit_exact(wave, sr: int, n: int, out_sr: int | None = None, skeleton=None, xfade_s: float = 0.08,
              tail_fade_s: float = 0.03, gap_s: float = 0.25, loudness: bool = False) -> tuple[np.ndarray, dict]:
    """Force ``wave`` to exactly ``n`` samples at ``out_sr`` (default: sr) -> ((2, n) float32, report).

    Longer than n: cut at n, ``tail_fade_s`` fade-out (no click). Shorter, or silent before n (the live end of the material is
    more than ``gap_s`` short of n): with ``skeleton`` (the composed stem at out_sr) the skeleton's tail takes over through an
    equal-power crossfade of ``xfade_s`` at the join, gain-matched to the restyled audio just before it; without it the
    material fades out and the rest is silence. ``loudness`` matches the integrated loudness to the skeleton's.
    """
    out_sr = int(out_sr or sr)
    n = int(n)
    x = _stereo(wave, sr, out_sr)
    if x.shape[1] == 0:
        raise ValueError("fit_exact: empty audio")
    live = min(_live_end(x, out_sr), x.shape[1])
    sk = _stereo(skeleton, out_sr, out_sr) if skeleton is not None else None
    rep = dict(in_samples=int(x.shape[1]), live_samples=int(live), n=n, mode="cut", join=None, gain_db=0.0)
    gap = int(round(gap_s * out_sr))
    tf = min(n, int(round(tail_fade_s * out_sr)))
    if live >= n - gap and x.shape[1] >= n:
        y = x[:, :n].copy()
        if tf:
            y[:, n - tf:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(tf) / tf)).astype(np.float32)[None, :]
    else:
        join = min(live, n)
        y = np.zeros((2, n), np.float32)
        y[:, :join] = x[:, :join]
        xf = min(int(round(xfade_s * out_sr)), join)
        if sk is not None and sk.shape[1] >= n and join < n:
            ref = max(1, min(join, int(0.5 * out_sr)))
            ra = float(np.sqrt((x[:, join - ref:join] ** 2).mean()) + 1e-9)
            rb = float(np.sqrt((sk[:, join - ref:join] ** 2).mean()) + 1e-9)
            gain = float(np.clip(ra / rb, 0.25, 4.0))
            tail = sk[:, :n] * np.float32(gain)
            w = np.arange(xf) / max(1, xf)
            fin, fout = np.sin(0.5 * np.pi * w).astype(np.float32), np.cos(0.5 * np.pi * w).astype(np.float32)
            if xf:
                y[:, join - xf:join] = x[:, join - xf:join] * fout[None, :] + tail[:, join - xf:join] * fin[None, :]
            y[:, join:] = tail[:, join:]
            rep.update(mode="crossfade", join=int(join), tail_gain=gain)
        else:
            if xf:
                y[:, join - xf:join] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(xf) / xf)).astype(np.float32)[None, :]
            rep.update(mode="pad", join=int(join))
    if loudness and sk is not None:
        y, rep["gain_db"] = match_loudness(y, sk[:, :n], out_sr)
    return np.ascontiguousarray(np.nan_to_num(y).astype(np.float32)), rep


# =================================================================================================== text -> sections / hits
def parse_hits(text) -> list[float]:
    """'3.5, 8, 12.25' (commas / spaces / semicolons) or a list -> seconds."""
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v) for v in re.split(r"[,\s;]+", str(text or "").strip()) if v]


def parse_sections(text):
    """'intro:0-3, build:3-6, drop:6-12' (or JSON [[0,3,'intro'],...]) -> [(start, end, role)]. Empty -> None (default arc)."""
    t = str(text or "").strip()
    if not t:
        return None
    if t[0] in "[{":
        import json
        out = []
        for e in json.loads(t):
            out.append((e["start"], e.get("end", 0), e["role"]) if isinstance(e, dict) else tuple(e))
        return out
    out = []
    for part in re.split(r"[,;]+", t):
        m = re.match(r"^\s*([A-Za-z]+)\s*[:@=]\s*([\d.]+)\s*[-~]\s*([\d.]+)\s*$", part)
        if not m:
            raise ValueError(f"section {part.strip()!r}: expected role:start-end, e.g. 'build:3-6'")
        out.append((float(m.group(2)), float(m.group(3)), m.group(1).lower()))
    return out


# =================================================================================================== CLI
def main(argv=None) -> int:
    """python -m mograph.score_skeleton --length 16 --bpm 122 --mood driving --seed 3 --hits 8 --out DIR/name [--master club]
    writes <out>.wav (24-bit, the skeleton) [+ <out>_master.wav with --master] + <out>_grid.json. Never overwrites: a taken name gets _r02, _r03..."""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--length", type=float, default=12.0)
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--mood", default="driving")
    ap.add_argument("--energy", type=float)
    ap.add_argument("--key")
    ap.add_argument("--meter", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hits", default="")
    ap.add_argument("--sections", default="")
    ap.add_argument("--backend", default="auto", choices=("auto", "sampled", "synth"))
    ap.add_argument("--master", choices=("club", "film", "neutral"))
    ap.add_argument("--out", required=True, help="path without extension")
    a = ap.parse_args(argv)
    wave, grid = compose(length_s=a.length, bpm=a.bpm, meter=a.meter, key=a.key, mood=a.mood, energy=a.energy,
                         sections=parse_sections(a.sections), hits=parse_hits(a.hits), seed=a.seed, backend=a.backend)
    from . import mastering
    base = Path(a.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    k = 1
    stem = base
    while any(Path(f"{stem}{suf}").exists() for suf in (".wav", "_master.wav", "_grid.json")):
        k += 1
        stem = base.with_name(f"{base.name}_r{k:02d}")
    mastering.write_wav24(f"{stem}.wav", wave, grid["sr"])
    out = [f"{stem}.wav"]
    if a.master:
        from tools.fusion import audio_cut
        y, rep = audio_cut.master_audio(wave, grid["sr"], preset=a.master)
        mastering.write_wav24(f"{stem}_master.wav", y, grid["sr"])
        out.append(f"{stem}_master.wav")
        grid["master"] = dict(preset=a.master, lufs=round(rep["lufs"], 2), true_peak_dbtp=round(rep["true_peak_dbtp"], 2))
    Path(f"{stem}_grid.json").write_text(json.dumps({k2: v for k2, v in grid.items()}, indent=1), encoding="utf-8")
    out.append(f"{stem}_grid.json")
    print(f"{grid['length_s']:.3f} s = {grid['n_bars']} bars x {grid['meter']} @ {grid['bpm']:.3f} bpm (hint {grid['bpm_hint']:.0f}), "
          f"{grid['key']}, {grid['mood']}/{grid['palette']} via {grid['backend']}, hits {grid['hits']}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
