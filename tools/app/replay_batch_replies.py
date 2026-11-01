#!/usr/bin/env python
"""Replay the RAW design replies of the 3060 batch v001 (16_app/bench/batch_3060/state.json `design_reply`) through the FIELD-LEVEL design step.

    python tools/app/replay_batch_replies.py [--out 21_batch-3060/v002/review/replay_v001]

Same replies, same footage layout (re-planned from the archive's piece.json + concept like the redesign graph does), the NEW parse: what share of each reply's fields
survives as the model wrote it. The old step (whole-section fallback) is quoted from the batch state (`archive.design.src`, the parse / repair check lines). CPU only
(plan + dry runs, ~15 s per run). No model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.fusion import design_step as DS      # noqa: E402
from tools.fusion import plan_rules as PR       # noqa: E402


def music_of(piece: dict) -> dict:
    ana = DS.ana_from_piece(piece)
    ana["fade"], ana["src"] = float((piece.get("audio") or {}).get("fade", 0.5)), (piece.get("audio") or {}).get("src", "")
    return ana


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="", help="comma list of batch run parts to replay (default all; the heavy-font ones are slow on a PC CPU)")
    ap.add_argument("--out", default=os.path.join(ROOT, "21_batch-3060", "v002", "review", "replay_v001"))
    a = ap.parse_args()
    st = json.load(open(os.path.join(ROOT, "16_app", "bench", "batch_3060", "state.json"), encoding="utf-8"))
    rows = []
    for r in st["runs"]:
        part = r.get("part")
        if not part or not r.get("design_reply") or not r.get("concept") or (a.runs and part not in a.runs.split(",")):
            continue
        d = glob.glob(os.path.join(ROOT, "21_batch-3060", "v001", "out", f"*_{part}_out_shots", "piece.json"))
        if not d:
            continue
        piece = json.load(open(d[0], encoding="utf-8"))
        c = r["concept"]
        ana = music_of(piece)
        req = dict(words="\n".join(c["words"]), scene=c["scene"], music=c["music"], length=c["length"], look=c["look"], seed=c["seed"], quality="preview", format="9:16",
                   shots=c["shots"], owners=c.get("owners"))
        plan = PR.plan_piece(req, ana)
        out = DS.parse(r["design_reply"], json.dumps(DS.make_locked(plan)), plan, ana, "")
        f = out["fields"] or {}
        old = (r.get("archive") or {}).get("design", {}).get("src")
        rows.append(dict(run=r["i"], part=part, old_src=old, old_check=(r.get("design_check") or "").splitlines()[:2],
                         new_sections=f.get("sections"), n_fields=f.get("n"), from_model=len(f.get("fields_from_model") or []), fixed=len(f.get("fields_fixed") or []),
                         from_rules=len(f.get("fields_from_rules") or []), pct_model=f.get("pct_model"), pct_model_or_fixed=f.get("pct_model_or_fixed"),
                         pending=len(f.get("pending") or []), ok=out["ok"], report=out["report"][:2500]))
        print(part, rows[-1]["pct_model"], rows[-1]["from_rules"], rows[-1]["new_sections"], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["# Replay of the batch v001 design replies through the field-level step", "",
         "Same raw model replies, same footage structure. `old` = what v001 actually used (section sources + the parse line); `new` = the field-level parse.", "",
         "| run | old (v001) | new type / compose | model fields % | model+nearest % | model / fixed / rules | pending for repair |", "|---|---|---|---:|---:|---|---:|"]
    for x in rows:
        old = "; ".join(x["old_check"])[:110] if x["old_check"] else json.dumps(x["old_src"])
        L.append(f"| {x['part']} | {old} | {(x['new_sections'] or {}).get('type')} / {(x['new_sections'] or {}).get('compose')} | {x['pct_model']} | {x['pct_model_or_fixed']} | "
                 f"{x['from_model']} / {x['fixed']} / {x['from_rules']} | {x['pending']} |")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
