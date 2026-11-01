"""Sampled instruments: SoundFont parts rendered offline, one stem per part (symusic / Prestosynth).

Idea after op55 (github.com/ynginkim/op55, MIT): the score is code on an exact bar grid, sampled parts come from
SoundFont banks, synth parts stay numpy. Everything here is deterministic (same input -> same bytes).

What the symusic synth does NOT do (checked on symusic 0.5.9 with 03_sound/v005/local/symusic_probe.py):
  * velocity only picks the sample layer; it does not scale the level (layers are normalised), so each note is
    rendered on its own and scaled here by the SF2 default velocity curve ``(vel/127) ** vel_curve``;
  * CC64 sustain pedal, CC7 volume and CC11 expression are ignored, so pedal extends note ends and CC7/CC11 become
    a gain curve over the whole stem;
  * a long-lived Synthesizer keeps a per-key gain set by the first note played on that key, so notes are rendered
    on a fresh Synthesizer each (history-free, deterministic);
  * only bank 0 (melodic) and bank 128 (drum kits) are reachable, no bank-select.

    from mograph.sampled import Grid, Ev, humanise, render_part
    g = Grid(112, 4)
    keys = humanise([g.ev(0, 0, 64, 70, 1.5), g.ev(0, 2, 69, 64, 1.0)], ms=8, vel=5, seed=1)
    stem = render_part(keys, "gm", program=4, sr=48000, release=0.4)      # (2, n) float32
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .banks import BANKS, ensure_banks

__all__ = ["Ev", "Grid", "humanise", "render_part", "list_presets", "preset_name", "resolve_bank"]


# ----------------------------------------------------------------------------
# events + grid
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Ev:
    """One note. t, dur in seconds; anchor=True means humanise() never moves it (kick, accents)."""
    t: float
    midi: int
    vel: float = 90
    dur: float = 0.5
    anchor: bool = False
    tag: str = ""


@dataclass(frozen=True)
class Grid:
    """Bar/beat grid. Times come from the index (``t0 + n * 60/bpm``), never from adding steps, so no drift."""
    bpm: float
    beats_per_bar: int = 4
    t0: float = 0.0

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_s(self) -> float:
        return self.beats_per_bar * self.beat_s

    def beat(self, n: float) -> float:
        return self.t0 + n * self.beat_s

    def bar(self, k: float, beat: float = 0.0) -> float:
        """Seconds at beat `beat` (0-based, may be fractional or run past the bar) of 0-based bar `k`."""
        return self.t0 + (k * self.beats_per_bar + beat) * self.beat_s

    def bar_of(self, t: float) -> float:
        """Fractional 0-based bar position of time t."""
        return (t - self.t0) / self.bar_s

    def ev(self, bar: float, beat: float, midi: int, vel: float = 90, beats: float = 1.0, **kw) -> Ev:
        """Note at bar/beat lasting `beats` beats."""
        return Ev(self.bar(bar, beat), int(midi), vel, beats * self.beat_s, **kw)


def humanise(events, ms: float = 8.0, vel: int = 6, seed: int = 0) -> list[Ev]:
    """Seeded timing (+/- ms, clipped normal) and velocity (+/- vel) jitter. Anchors keep time and velocity.

    Random draws are made for every event in (t, midi) order whether anchored or not, so marking one more event as
    an anchor does not change the others.
    """
    rng = np.random.default_rng(int(seed))
    out = []
    for e in sorted(events, key=lambda e: (e.t, e.midi)):
        dt = float(np.clip(rng.normal(0.0, ms / 2000.0), -ms / 1000.0, ms / 1000.0))
        dv = float(rng.integers(-int(vel), int(vel) + 1)) if vel else 0.0
        if e.anchor:
            out.append(e)
        else:
            out.append(replace(e, t=max(0.0, e.t + dt), vel=float(np.clip(e.vel + dv, 1, 127))))
    return out


# ----------------------------------------------------------------------------
# bank helpers
# ----------------------------------------------------------------------------


def resolve_bank(bank, cache_dir=None) -> Path:
    """Registry name ("gm", "upright") or a path to an .sf2 -> local path (downloads registry banks once)."""
    if isinstance(bank, Path) or (isinstance(bank, str) and bank not in BANKS):
        p = Path(bank)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    p = ensure_banks([bank], cache_dir)[bank]
    if p is None:
        raise RuntimeError(f"bank {bank!r} is optional and not installed")
    return p


@lru_cache(maxsize=8)
def _presets(path: str) -> tuple:
    """(bank, program, name) for every preset, read from the SF2 'phdr' chunk (symusic cannot list them)."""
    d = Path(path).read_bytes()

    def walk(i, end):
        res = []
        while i + 8 <= end:
            cid, sz = d[i:i + 4], struct.unpack("<I", d[i + 4:i + 8])[0]
            if cid == b"LIST":
                res += walk(i + 12, i + 8 + sz)
            elif cid == b"phdr":
                for k in range(0, sz - 38, 38):  # last record is the EOP terminator
                    name = d[i + 8 + k:i + 28 + k].split(b"\0")[0].decode("latin1")
                    prog, bnk = struct.unpack("<HH", d[i + 28 + k:i + 32 + k])
                    res.append((bnk, prog, name))
            i += 8 + sz + (sz & 1)
        return res

    return tuple(walk(12, len(d)))


def list_presets(bank, cache_dir=None) -> list[tuple[int, int, str]]:
    return list(_presets(str(resolve_bank(bank, cache_dir))))


def preset_name(bank, program: int | None = None, kit: int | None = None, cache_dir=None) -> str | None:
    want_b, want_p = (128, kit) if kit is not None else (0, program)
    for b, p, n in list_presets(bank, cache_dir):
        if (b, p) == (want_b, want_p):
            return n
    return None


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _note(path: str, sr: int, program: int, drum: bool, midi: int, vel: int, dur_n: int) -> np.ndarray:
    """One note at t=0 straight from the synth (includes the SF2 release tail): (2, n) float32, read-only.

    A fresh Synthesizer per note (~15 ms): a long-lived instance scales each key by a gain fixed by the first note
    played on that key (symusic 0.5.9), so a note would sound different depending on what was rendered before it.
    """
    from symusic import Synthesizer, core
    sc = core.ScoreSecond(480)
    tr = core.TrackSecond("p", int(program), bool(drum))
    tr.notes.append(core.NoteSecond(0.0, dur_n / sr, int(midi), int(vel)))
    sc.tracks.append(tr)
    x = np.array(Synthesizer(path, sr, 0).render(sc, True), dtype=np.float32)
    x.setflags(write=False)
    return x


def _pedal_spans(cc, pedal) -> list[tuple[float, float]]:
    spans = [(float(a), float(b)) for a, b in pedal]
    down = None
    for t, num, val in sorted(c for c in cc if c[1] == 64):
        if val >= 64 and down is None:
            down = t
        elif val < 64 and down is not None:
            spans.append((down, t))
            down = None
    if down is not None:
        spans.append((down, float("inf")))
    return sorted(spans)


def _cc_gain(cc, n: int, sr: int) -> np.ndarray | None:
    """CC7 (volume) and CC11 (expression) -> linear gain (v/127)^2 per sample, held between events, 1.0 before the first."""
    g = None
    for num in (7, 11):
        pts = sorted((t, v) for t, k, v in cc if k == num)
        if not pts:
            continue
        ts = np.array([p[0] for p in pts]) * sr
        vs = (np.array([p[1] for p in pts], dtype=np.float64) / 127.0) ** 2
        c = np.interp(np.arange(n), ts, vs, left=1.0)
        g = c if g is None else g * c
    return g


def render_part(events, bank="gm", program: int | None = None, kit: int | None = None, sr: int = 48000, *,
                cc=(), pedal=(), legato: float = 0.0, release: float | None = None, vel_curve: float = 2.0,
                width: float = 1.0, length: float | None = None, cache_dir=None) -> np.ndarray:
    """Render one part to a (2, n) float32 stem.

    events   Ev list. For drums `kit` is the GM drum-kit number (0 Standard 1, 8 Room, ...) and midi is the GM drum note.
    cc       (t, number, value) tuples: 64 = sustain pedal (extends note ends to pedal-up), 7/11 = stem gain curve.
    pedal    extra (down, up) spans in seconds.
    legato   seconds added to every note so it overlaps the next one.
    release  None keeps the SF2 release; a number (s) replaces it: the note sustains `release` s past its end and
             fades out (use for presets whose release is ~0, e.g. the GM Tine Electric Piano).
    vel_curve  gain = (vel/127) ** vel_curve (2.0 = SF2 default modulator, which this synth omits).
    width    mid/side width (1 = as sampled; GM piano is panned by key, 0.6 narrows it).
    length   total seconds (pad/trim) so every stem has the same length; default = end of the last tail.
    """
    if (program is None) == (kit is None):
        raise ValueError("give exactly one of program= (melodic) or kit= (drums)")
    drum = kit is not None
    prog = int(kit if drum else program)
    path = str(resolve_bank(bank, cache_dir))
    spans = _pedal_spans(cc, pedal)
    evs = sorted(events, key=lambda e: (e.t, e.midi, e.dur))

    placed = []  # (start sample, buffer)
    for e in evs:
        end = e.t + e.dur + legato
        if not drum:
            for down, up in spans:
                if down <= end < up:
                    end = up
        dur = max(0.02, end - e.t)
        dur_n = int(round((dur + (release or 0.0)) * sr))
        buf = _note(path, sr, prog, drum, int(e.midi), int(round(np.clip(e.vel, 1, 127))), dur_n)
        g = (float(np.clip(e.vel, 1, 127)) / 127.0) ** vel_curve
        if release:
            rn = int(round(release * sr))
            keep = min(buf.shape[1], dur_n)
            env = np.ones(keep, np.float32)
            env[max(0, keep - rn):] = 0.5 + 0.5 * np.cos(np.pi * np.arange(keep - max(0, keep - rn)) / rn)
            buf = buf[:, :keep] * env[None, :]
        placed.append((int(round(e.t * sr)), buf * np.float32(g)))

    n = max((s + b.shape[1] for s, b in placed), default=0)
    if length is not None:
        n = int(round(length * sr))
    out = np.zeros((2, n), np.float32)
    for s, b in placed:  # fixed order -> identical float32 sums every run
        e1 = min(n, s + b.shape[1])
        if e1 > s:
            out[:, s:e1] += b[:, :e1 - s]
    if width != 1.0:
        m, sd = 0.5 * (out[0] + out[1]), 0.5 * (out[0] - out[1]) * np.float32(width)
        out = np.stack([m + sd, m - sd]).astype(np.float32)
    cg = _cc_gain(cc, n, sr)
    if cg is not None:
        out = (out * cg[None, :]).astype(np.float32)
    return out
