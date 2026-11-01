"""cut_p06.py - bar-aligned cut of the user's Song audio for P06 (song-chords).

    python tools/fusion/cut_p06.py [--near 60] [--bars 11] [--src FILE] [--out DIR]

1. analyse the whole source (tools/fusion/music_analysis.py) -> bar grid
2. start = the bar start nearest --near seconds; length = --bars whole bars
   (default 11 bars = 16.5 s at 160 bpm: nearest whole-bar length to the PLAN's ~16 s; 8 bars would be 12 s)
3. 20 ms fade-in, raised-cosine fade-out over the 2nd half of the last bar (2 beats), written as 48 kHz 24-bit WAV
4. re-analyse the cut (bpm-hint = source bpm) -> P06_song_cut_analysis.json, write P06_notes.md
The source file is never modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import music_analysis as ma  # noqa: E402

SRC = (Path.home() / "Documents/Dev/vastai-api-manager/Instance_Data/machine_55158/54610898/media/audio/Song2"
       / "J03.2_t13_rhythmic_lead_s10211_00001.flac")
OUT = ROOT / "15_fusion" / "v001" / "local" / "audio"


def mmss(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--near", type=float, default=60.0)
    ap.add_argument("--bars", type=int, default=11)
    ap.add_argument("--name", default="P06_song_cut")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    whole = ma.analyse(a.src)
    bar = whole["bar_len"]
    starts = np.array([b["start"] for b in whole["bars"]])
    i0 = int(np.argmin(np.abs(starts - a.near)))
    t0 = float(starts[i0])
    n = a.bars
    assert i0 + n <= len(starts), "not enough bars after the start"
    x, sr = sf.read(a.src, always_2d=True)
    s0 = int(round(t0 * sr))
    s1 = int(round((t0 + n * bar) * sr))
    seg = x[s0:s1].copy()
    nfi = int(round(0.020 * sr))
    seg[:nfi] *= np.linspace(0, 1, nfi)[:, None]
    nfo = int(round(bar / 2 * sr))
    seg[-nfo:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nfo)))[:, None]
    peak = float(np.abs(seg).max())
    if peak > 0.98:
        seg *= 0.98 / peak
    wav = out / f"{a.name}.wav"
    sf.write(str(wav), seg, sr, subtype="PCM_24")

    res = ma.analyse(wav, bpm_hint=whole["bpm"], grid=(bar, 0.0))   # cut starts on a bar line by construction
    (out / f"{a.name}_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    seq = [b["chord"] for b in res["bars"]]
    src_seq = [whole["bars"][i0 + k]["chord"] for k in range(n)]
    runs = []
    for k, c in enumerate(src_seq):
        if runs and runs[-1][0] == c:
            runs[-1][2] = k + 1
        else:
            runs.append([c, k + 1, k + 1])
    lines = [
        "# P06 song-chords - cut notes",
        "",
        f"- source: `{a.src}` ({whole['duration']:.1f} s, {sr} Hz stereo); never modified",
        f"- cut: `{wav.name}`, {sr} Hz stereo 24-bit, {len(seg) / sr:.3f} s",
        f"- start in source: {t0:.4f} s ({mmss(t0)}) = source bar {i0 + 1} (bar start nearest {a.near:.0f} s; "
        f"bar 1 of an 8-bar phrase: bars 1, 9, 17 ... of the source)",
        f"- length: {n} bars x {bar:.4f} s = {n * bar:.3f} s (4/4); ends at source {t0 + n * bar:.4f} s",
        f"- bpm: {whole['bpm']:.2f} (source whole-file fit; cut re-analysis {res['bpm']:.2f}); cut bar 1 starts at 0.000 s (grid forced)",
        f"- key: {res['key']['estimate']} (cut; relative major of C minor), {whole['key']['estimate']} (whole source; 03_sound/v004 agrees)",
        f"- fades: 20 ms linear fade-in; raised-cosine fade-out over the last 2 beats ({bar / 2:.3f} s) of the last bar",
        f"- chord per bar (cut re-analysis): {' | '.join(seq)}",
        f"- chord per bar (from the whole-file analysis, same bars): {' | '.join(src_seq)}",
        "- chord blocks (cut bar numbers, 1-based): " + ", ".join(
            f"{c} bars {s}-{e} ({(s - 1) * bar:.2f}-{e * bar:.2f} s)" for c, s, e in runs),
        "- chord changes (cut time): " + (", ".join(f"{c['t']:.2f} s {c['from']}->{c['to']}" for c in res["chord_changes"]) or "none"),
        "",
        "Chords are chroma-template estimates (Cm / G# = Ab / Gm): treat G#maj7 as G# (Ab), and the one-bar Fsus4 (bar 3) as a suspension/passing "
        "chord inside the Cm block (the whole-file chord blocks run Cm 4 bars, Ab 4 bars, Gm 4 bars). Key/palette: C minor, so G# = Ab major, Gm = the v chord.",
        "",
        "Chord-change times are bar lines, so scene changes land on the downbeat grid in "
        f"`{a.name}_analysis.json` (`bars[].start`, `chord_changes`).",
    ]
    (out / "P06_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
