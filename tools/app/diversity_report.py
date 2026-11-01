#!/usr/bin/env python
"""diversity_report: the top-value share of every field the open models kept landing on (16_app/DIVERSITY_AUDIT.md), measured OFFLINE (no API call).

  python tools/app/diversity_report.py par r8                  the saved piece.json of a par round (what the code made of the reply back then)
  python tools/app/diversity_report.py renorm r8 [--system]    the saved replies of a round re-normalized with TODAY's code (--system: the brief's piece system folded in, as round r9 does)
  python tools/app/diversity_report.py synth [--n 8]           N synthetic seeds of the concept step (cues + piece system), a batch: each concept sees the earlier ones

Fields: module, form_in, form_in of word 1, form_out, font (per word and per piece), |yaw|, form_s, exit_s, camera, beat_orbit fast_dps / elev. Share = the commonest value / all values.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL = ROOT / "20_open-model-design" / "v003" / "local"


def top(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ("-", 0.0, 0)
    c = collections.Counter(vals)
    v, n = c.most_common(1)[0]
    return (v, round(n / len(vals), 3), len(vals))


def measure(pieces: list[dict]) -> dict:
    W = [[w for w in (p.get("type") or {}).get("words", [])] for p in pieces]
    allw = [w for ws in W for w in ws]
    out = {}
    out["module (words)"] = top([w.get("module") for w in allw])
    out["module (single-module pieces)"] = ("one", round(sum(1 for ws in W if len({w.get("module") for w in ws}) == 1) / max(len(W), 1), 3), len(W))
    out["form_in (words)"] = top([w.get("form_in") for w in allw])
    out["form_in word 1"] = top([ws[0].get("form_in") for ws in W if ws])
    out["lines share of word 1"] = ("lines", round(sum(1 for ws in W if ws and ws[0].get("form_in") == "lines") / max(len(W), 1), 3), len(W))
    out["form_out (words)"] = top([w.get("form_out") for w in allw])
    out["font (words)"] = top([w.get("font") for w in allw])
    out["font (pieces using the top face)"] = None
    fonts_p = collections.Counter(f for ws in W for f in {w.get("font") for w in ws})
    if fonts_p:
        f, n = fonts_p.most_common(1)[0]
        out["font (pieces using the top face)"] = (f, round(n / max(len(W), 1), 3), len(W))
    out["|yaw| (words)"] = top([abs(round(float((w.get("placement") or {}).get("yaw")), 1)) for w in allw if isinstance((w.get("placement") or {}).get("yaw"), (int, float))])

    def beats(w, k):
        return round(float(w.get(k, 0)), 2)
    out["form_s (words)"] = top([beats(w, "form_s") for w in allw if "form_s" in w])
    out["exit_s (words)"] = top([round(float(w["exit_s"]), 2) for w in allw if "exit_s" in w])
    out["camera (pieces)"] = top([(p.get("type") or {}).get("camera") for p in pieces])
    out["fast_dps (beat_orbit pieces)"] = top([(p.get("type") or {}).get("camera_params", {}).get("fast_dps") for p in pieces if (p.get("type") or {}).get("camera") == "beat_orbit"])
    out["elev (beat_orbit pieces)"] = top([(p.get("type") or {}).get("camera_params", {}).get("elev") for p in pieces if (p.get("type") or {}).get("camera") == "beat_orbit"])
    pal = [((p.get("type") or {}).get("palette") or {}) for p in pieces]
    out["palette fg neutral-white"] = ("neutral", round(sum(1 for q in pal if q.get("fg") and max(q["fg"]) - min(q["fg"]) <= 14) / max(len(pal), 1), 3), len(pal))
    return out


def load_round(rnd: str) -> list[tuple[str, Path]]:
    return [(d.name, d) for d in sorted((LOCAL / rnd).glob("*_s*")) if (d / "reply.txt").is_file()]


def par_pieces(rnd: str) -> list[dict]:
    out = []
    for _n, d in load_round(rnd):
        f = d / "piece.json"
        if f.is_file():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def renorm_pieces(rnd: str, system: bool) -> list[dict]:
    from tools.fusion import piece_brief as PB
    from tools.fusion import qwen_piece as QP
    out = []
    for name, d in load_round(rnd):
        pid = name.split("_")[0]
        if pid not in PB.BRIEFS:
            continue
        obj = QP.extract_json((d / "reply.txt").read_text(encoding="utf-8"))
        if obj is None:
            continue
        b = PB.load_brief(pid)
        if system:
            b["system"] = PB.system_for(b)
        piece, _fx, _errs = PB.reply_to_piece(obj, b)
        if piece:
            out.append(piece)
    return out


def synth(n: int) -> dict:
    """N concept steps with no API: the cues and the piece system of seeds 1..N as one batch (each sees the earlier ones). The mood is the cue mood, the tempo the tone's own."""
    from tools.fusion import concept as CO
    from tools.fusion import piece_system as PS
    prev, rows = [], []
    for seed in range(1, n + 1):
        cu = CO.cues(seed, prev)
        sysm = PS.choose(seed, cu["mood"], None, earlier=[p["system"] for p in prev], palette_mood=cu["mood"])
        prev.append({"cues": {k: cu[k] for k in ("place", "light", "weather", "colour", "genre", "register")}, "mood": cu["mood"], "system": sysm})
        rows.append(dict(cu, system=sysm))
    f = {
        "look": [r["look"] for r in rows], "module": [r["system"]["module"] for r in rows], "form_in word 1": [r["system"]["io"]["form_in"][0] for r in rows],
        "form_out": [r["system"]["io"]["form_out"][0] for r in rows], "type camera": [r["system"]["camera"]["type"] for r in rows], "palette family": [r["system"]["palette"]["family"] for r in rows],
        "photo style": [r["system"]["photo_key"] for r in rows], "yaw band hi": [r["system"]["motion"]["yaw"][1] for r in rows], "place": [r["place"] for r in rows], "light": [r["light"] for r in rows],
        "weather": [r["weather"] for r in rows], "register": [r["register"] for r in rows], "font (all shortlist faces)": [x for r in rows for x in r["system"]["fonts"]],
        "time-of-day light (blue hour / golden / night / moon)": ["t" if any(k in r["light"] for k in ("blue hour", "golden", "night", "moonlight")) else "-" for r in rows],
        "haze / mist / fog weather": ["h" if "haze" in r["weather"] else "-" for r in rows]}
    return {k: top(v) for k, v in f.items()}


def show(title: str, m: dict) -> None:
    print(f"== {title}")
    for k, v in m.items():
        if v:
            print(f"  {k:42s} top={v[0]!s:34.34} share={v[1]:.0%}  n={v[2]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["par", "renorm", "synth"])
    ap.add_argument("round", nargs="?", default="r8")
    ap.add_argument("--system", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "par":
        m = measure(par_pieces(a.round))
    elif a.cmd == "renorm":
        m = measure(renorm_pieces(a.round, a.system))
    else:
        m = synth(a.n)
    if a.json:
        print(json.dumps(m, indent=1, default=str))
    else:
        show(f"{a.cmd} {a.round if a.cmd != 'synth' else a.n}" + (" +system" if a.system else ""), m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
