"""Timeline: phrase landings on a bpm grid, read windows, transition windows, and the timing QC.

    tl = timeline.Timeline.build(phrases, bpm=96, start=0.2, form=0.5, trans=0.7)
    tl.land[i], tl.read[i] = (t0, t1), tl.trans[i] = (t0, t1)   # i -> i + 1
    a, b, s = tl.state(t)          # blending phrase a -> b at s (a == b during form / read)
    kind, i, u = tl.phase(t)       # "form" | "read" | "trans", phrase, progress in the window
    tl.check()                     # list of flags (read window shorter than 0.5 s + 0.3 s/word, > 20 cps, ...)

A phrase is "formed" (fully legible) at land[i] and must hold still from land[i] to read[i][1] >=
land[i] + read_time(text) (last phrase >= 2 s). Transitions run between read windows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import text as _text


@dataclass
class Timeline:
    phrases: list                      # text.Phrase or str
    land: list                         # fully formed
    read: list                         # (t0, t1) read windows
    trans: list                        # (t0, t1) transition i -> i + 1
    dur: float
    fps: int = 25
    form0: float = 0.0                 # the first phrase starts forming here
    bpm: float | None = None
    notes: list = field(default_factory=list)

    @staticmethod
    def build(phrases, bpm=None, start=0.2, form=0.5, trans=0.7, tail=0.6, min_last=2.0, read_pad=0.0,
              dur=None, fps=25):
        """Lay phrases end to end: form (first only), read >= read_time + read_pad, transition `trans`, ...
        With bpm, every landing snaps up to the next beat (beat = 60 / bpm) and dur rounds up to a beat."""
        ps = [p if isinstance(p, _text.Phrase) else _text.Phrase(p) for p in phrases]
        beat = 60.0 / bpm if bpm else None

        def snap(t):
            return math.ceil(t / beat - 1e-6) * beat if beat else t
        land, read, tr = [], [], []
        t = snap(start + form)
        for i, p in enumerate(ps):
            land.append(t)
            hold = p.read + read_pad
            if i == len(ps) - 1:
                hold = max(hold, min_last)
            r1 = t + hold
            read.append((t, r1))
            if i < len(ps) - 1:
                nxt = snap(r1 + trans)
                tr.append((r1, nxt))
                t = nxt
        end = read[-1][1] + tail
        if dur is None:
            dur = snap(end) if beat else end
        fps_dur = math.ceil(dur * fps - 1e-6) / fps
        return Timeline(ps, land, read, tr, fps_dur, fps, start, bpm)

    @staticmethod
    def fixed(phrases, land, trans, dur, read_end=None, fps=25, form0=0.0):
        """Wrap an existing piece's hand-set landings and transition windows (read window = landing to the
        next transition start, or to read_end / dur for the last)."""
        ps = [p if isinstance(p, _text.Phrase) else _text.Phrase(p) for p in phrases]
        read = []
        for i, L in enumerate(land):
            r1 = trans[i][0] if i < len(trans) else (read_end if read_end is not None else dur)
            read.append((L, r1))
        return Timeline(ps, list(land), read, list(trans), dur, fps, form0)

    # ----- queries
    @property
    def n(self):
        return len(self.land)

    @property
    def nframes(self):
        return int(round(self.dur * self.fps))

    def changes(self):
        """Musical arrivals for score.py --changes: the landings."""
        return list(self.land)

    def phase(self, t):
        if t < self.land[0]:
            return "form", 0, (t - self.form0) / max(self.land[0] - self.form0, 1e-6)
        for i in range(self.n):
            r0, r1 = self.read[i]
            if t < r1 or i == self.n - 1:
                return "read", i, min((t - r0) / max(r1 - r0, 1e-6), 1.0)
            t0, t1 = self.trans[i]
            if t < t1:
                return "trans", i, (t - t0) / (t1 - t0)
        return "read", self.n - 1, 1.0

    def state(self, t):
        """(a, b, s): a -> b at s in [0, 1]; a = -1 before the first phrase starts forming."""
        k, i, u = self.phase(t)
        if k == "form":
            return (-1, 0, max(0.0, min(u, 1.0)))
        if k == "trans":
            return (i, i + 1, u)
        return (i, i, 0.0)

    def reading(self, t):
        """Index of the phrase in its read window at t, else None."""
        for i, (r0, r1) in enumerate(self.read):
            if r0 <= t <= r1:
                return i
        return None

    def read_moments(self):
        """One sample time per read window (its middle), for the legibility QC."""
        return [0.5 * (r0 + r1) for r0, r1 in self.read]

    def read_mask(self, times):
        return [self.reading(t) is not None for t in times]

    # ----- QC
    def check(self):
        """Timing flags: read windows shorter than the formula, > 20 cps, last < 2 s, overlaps, overrun."""
        flags = []
        for i, (p, (r0, r1)) in enumerate(zip(self.phrases, self.read)):
            need = p.read if i < self.n - 1 else max(p.read, 2.0)
            if r1 - r0 + 1e-6 < need:
                flags.append(dict(kind="read_short", phrase=i, have=round(r1 - r0, 2), need=round(need, 2),
                                  text=p.text))
            cps = len(p.text.replace(" ", "")) / max(r1 - r0, 1e-6)
            if cps > 20:
                flags.append(dict(kind="cps", phrase=i, cps=round(cps, 1)))
        for i, (t0, t1) in enumerate(self.trans):
            if t0 < self.read[i][1] - 1e-6 or t1 > self.land[i + 1] + 1e-6:
                flags.append(dict(kind="overlap", trans=i))
        if self.read[-1][1] > self.dur + 1e-6:
            flags.append(dict(kind="overrun", end=self.read[-1][1], dur=self.dur))
        return flags

    def summary(self):
        rows = []
        for i, p in enumerate(self.phrases):
            rows.append(f"P{i + 1} '{p.text}': land {self.land[i]:.2f}s, read {self.read[i][0]:.2f}-"
                        f"{self.read[i][1]:.2f}s (need {p.read:.2f}s)")
        return "\n".join(rows) + f"\ndur {self.dur:.2f}s ({self.nframes} frames @ {self.fps})"
