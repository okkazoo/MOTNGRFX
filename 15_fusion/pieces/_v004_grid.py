"""Shared by the v004 generators (_gen_P12.py, _gen_P14.py): which song + beat grid to plan on, the grid-locked audio analysis,
a beat-grid word plan (hits / holds / transition gaps) and a state reset so a re-planned piece re-runs type + compose only.

Song resolution (first hit wins): --song <wav> --grid <json> on the command line; 15_fusion/v004/audio/t2m/<id>.wav + <id>_grid.json
(pure ACE text-to-music, if present); else the current pick 15_fusion/v004/audio/<id>_<seed>.wav + _grid.json.

Re-plan on a new song without new footage:
    python 15_fusion/pieces/_gen_P12.py --reset-tc        # rewrite P12.json on the new grid + drop audio/type/compose/final from the
                                                          # runner state (shots / flow / alpha / masktrack records stay -> reused)
    python tools/fusion/piece_runner.py 15_fusion/pieces/P12.json --exec box ... --draft --out-version v004
Footage is only reusable if every shot's span still fits its rendered frames; plan prints a WARNING when it does not.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
AUD = ROOT / "15_fusion" / "v004" / "audio"
STATE = ROOT / "15_fusion" / "v004" / "local" / "pieces"
SETTLE_HOLD, RAMP = 0.75, 0.12


def args(pid, seed):
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", help="wav (repo-relative or absolute)")
    ap.add_argument("--grid", help="grid json with bpm / beats / downbeats / cut")
    ap.add_argument("--no-audio", action="store_true", help="do not rebuild the analysis json")
    ap.add_argument("--reset-tc", action="store_true", help="drop audio/type/compose/final from the runner state")
    ap.add_argument("--reset-compose", action="store_true", help="drop compose/final only (ops changed, timing not)")
    ap.add_argument("--wide", action="store_true", help="write the 16:9 version <id>_w.json (own still + shots)")
    ap.add_argument("--design-window", action="store_true", help="P12: the paused window redesign instead of the g03 line")
    a = ap.parse_args()
    if not a.song:
        t2m = AUD / "t2m" / f"{pid}.wav"
        if t2m.exists() and (AUD / "t2m" / f"{pid}_grid.json").exists():
            a.song, a.grid = str(t2m), str(AUD / "t2m" / f"{pid}_grid.json")
        else:
            a.song, a.grid = str(AUD / f"{pid}_{seed}.wav"), str(AUD / f"{pid}_{seed}_grid.json")
    if not a.grid:
        a.grid = str(Path(a.song).with_name(Path(a.song).stem + "_grid.json"))
    return a


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def load_grid(path):
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    cut = float(g.get("cut") or g["grid"]["length_s"])
    return dict(bpm=float(g["bpm"]), beats=[float(b) for b in g["beats"]], downbeats=[float(d) for d in g["downbeats"]],
                bar_len=float(g["bar_len"]), cut=cut, chord_changes=g.get("chord_changes", []))


def analysis(pid, song, grid, write=True):
    """music_analysis curves (onset / low bands for the pulses) + the score's exact beat grid -> repo-relative path."""
    stem = Path(song).stem
    tag = "t2m_" if Path(song).parent.name == "t2m" else ""
    p = ROOT / "15_fusion" / "v004" / "local" / "audio" / f"{tag}{stem}_analysis.json"
    if write:
        sys.path.insert(0, str(ROOT))
        from tools.fusion import music_analysis
        a = music_analysis.analyse(str(song), grid["bpm"], 4, 24)
        a.update(bpm=grid["bpm"], bar_len=grid["bar_len"], beats=grid["beats"], downbeats=grid["downbeats"],
                 chord_changes=grid["chord_changes"] or a.get("chord_changes", []),
                 first_bar_start=grid["downbeats"][0] if grid["downbeats"] else 0.0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(a, indent=1), encoding="utf-8")
    return rel(p)


def plan(grid, n=4, hold_beats=2, first_min=1.5, max_space=6):
    """4 words on the beat grid. First hit = first downbeat >= first_min s; hits `s` beats apart (s <= max_space = 1.5 bars, as
    wide as the clip allows); each word: 1-beat form, `hold_beats` hold (last word: to the end), 1-beat exit, the rest of the
    cycle is the transition gap (>= 1 beat). -> list of dicts t0 / t1 / form_s / exit_s / hit / hold=(a, b) / gap=(a, b)."""
    cut = grid["cut"]
    bt = [b for b in grid["beats"] if b <= cut + 1e-6]
    iv = float(np.median(np.diff(bt)))
    while bt[-1] + iv <= cut + 1e-6:
        bt.append(bt[-1] + iv)
    db = [d for d in grid["downbeats"] if d >= first_min - 1e-6]
    last_ok = max(i for i, b in enumerate(bt) if b <= cut - hold_beats * iv + 1e-6)
    def spaces(i0):                                     # per-gap beats: max_space each, shaved from the LAST gaps first
        sp = [max_space] * (n - 1)
        k = n - 2
        while sum(sp) > last_ok - i0 and min(sp) > hold_beats + 2:
            sp[k] -= 1
            k = k - 1 if k > 0 else n - 2
        return sp
    i0 = int(np.argmin([abs(b - db[0]) for b in bt]))
    sp = spaces(i0)
    if sum(sp) > last_ok - i0:                          # slow song: start on the first beat >= 1 s instead of a downbeat
        i0 = min(i for i, b in enumerate(bt) if b >= 1.0 - 1e-6)
        sp = spaces(i0)
    assert sum(sp) <= last_ok - i0, f"grid too tight: {sp} beats per word"
    idx = [i0 + sum(sp[:k]) for k in range(n)]
    out = []
    for k in range(n):
        i = idx[k]
        s = sp[k] if k < n - 1 else 0
        hit = bt[i]
        rel_ = bt[i + hold_beats] if k < n - 1 else cut
        start = 0.0 if k == 0 else bt[i - 1]
        end = (bt[i + hold_beats + 1] if k < n - 1 else cut + iv)
        gap = (end, bt[i + s - 1]) if k < n - 1 else None
        out.append(dict(t0=round(start, 4), t1=round(end, 4), form_s=round(hit - start, 4), exit_s=round(end - rel_, 4),
                        hit=round(hit, 4), hold=(round(hit, 4), round(rel_, 4)), gap=gap, hit_on="beat"))
    return out, iv


def snap8_up(n):
    """smallest 8k+1 >= n (LTX frame counts)."""
    return ((max(int(n), 1) - 1 + 7) // 8) * 8 + 1


def settle(holds):
    keys = [[0.0, 0.0]]
    for a, b in holds:
        keys += [[a - RAMP, 0.0], [a + RAMP, SETTLE_HOLD], [b - RAMP, SETTLE_HOLD], [b + RAMP, 0.0]]
    return {"timeline": [[round(t, 3), v] for t, v in keys], "ease": "smooth"}


def reset(pid, keep_type=False):
    """drop audio / type / compose / assemble / final (keep_type: compose / assemble / final only) from the runner state so the
    next run re-plans on the current piece json and reuses the footage (stills / shots / flow / alpha / depth / masktrack)."""
    sp = STATE / pid / "state.json"
    if not sp.exists():
        print("no runner state yet:", sp)
        return
    shutil.copy2(sp, sp.with_suffix(".json.bak"))
    s = json.loads(sp.read_text(encoding="utf-8"))
    drop_st = ["compose", "final"] + ([] if keep_type else ["audio", "type"])
    for k in list(s["items"]):
        base = k.split(":")[0]
        if base in drop_st or base == "assemble":
            del s["items"][k]
    for st in drop_st:
        s.get("stages", {}).pop(st, None)
    sp.write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print("reset", sp, "->", sorted(s["items"]))
