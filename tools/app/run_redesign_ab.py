#!/usr/bin/env python
"""Model A/B by REDESIGNING finished generations (graphs app/graphs/fusion_redesign{,_q8,_gemma}.json). A plain driver: no LLM of its own.

    python tools/app/run_redesign_ab.py run     --iid 57999748 --port 9000 [--runs r01b,r03,r05,r07] [--arms q4,q8,gemma] [--target P/I/vNNN] [--resume]
    python tools/app/run_redesign_ab.py collect --target COMPETITIONS/motion-graphic/batch-3060/v002 --iid 57999748 --port 9000   # pull piece.json, build AB.json + AB.md

For every (run, arm), sequentially, it submits the arm's graph through tools/app/run_timing.py (= progress_standin.py submit, per-node / per-stage seconds, peak VRAM)
with the archive of the batch run (`redesign/src/<run>_out_shots` + `<run>_final.mp4` under the box's ComfyUI output dir), the run's concept (state.json of the batch), the
SAME design seed for every arm of a run (the batch run's seed), and joins the PROJECTS version `--target` with the part `<run>_<arm>` (q4 / q8 / g4).
State: <bench>/state.json after every job (--resume skips the done ones). Per job the record keeps the final mp4, the raw model reply, the parse / repair reports and the
stage seconds; `collect` adds the piece.json and the salvage numbers (plan.design.fields) and writes <version>/review/AB.json + AB.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "app"))
import run_batch as RB      # noqa: E402
import run_timing as RT     # noqa: E402

GRAPHS = {"q4": "fusion_redesign.json", "q8": "fusion_redesign_q8.json", "gemma": "fusion_redesign_gemma.json"}
PART = {"q4": "q4", "q8": "q8", "gemma": "g4"}
LABEL = {"q4": "design_llm_8b 4-bit", "q8": "design_llm_8b 8-bit", "gemma": "Gemma 4 12B int8"}
BATCH_STATE = os.path.join(ROOT, "16_app", "bench", "batch_3060", "state.json")
BENCH = os.path.join(ROOT, "16_app", "bench", "redesign_ab")
TARGET = "COMPETITIONS/motion-graphic/batch-3060/v002"
VERSION_DIR = os.path.join(ROOT, "21_batch-3060", "v002")
STAGE_KEYS = {"model": "Qwen design", "repair": "Qwen repair (only when the first reply failed)", "type": "Type passes (designed piece)", "compose": "Compose",
              "assemble": "Assemble + encode + save", "flow": "Footage flow (RAFT)", "package": "Design package", "parse": "Design parse / repair / piece",
              "load": "Archive load (files)", "plan": "Plan (rules)", "archive": "Archive (still, shot clips, matte PNGs)", "gemma_file": "Model file load (Gemma CLIPLoader)"}


def batch_runs(parts: list[str]) -> list[dict]:
    st = json.load(open(BATCH_STATE, encoding="utf-8"))
    by = {r["part"]: r for r in st["runs"] if r.get("part")}
    out = []
    for p in parts:
        r = by[p]
        c = r["concept"]
        out.append(dict(part=p, i=r["i"], seed=r["seed"], concept=dict(words=c["words"], scene=c["scene"], music=c["music"], look=c["look"], shots=c["shots"],
                                                                      length=c["length"], seed=r["seed"], title=c.get("title"), subject=c.get("subject"), mood=c.get("mood"))))
    return out


def jobs(parts: list[str], arms: list[str], suffix: str = "") -> list[dict]:
    return [dict(run=r["part"], arm=a, part=f"{r['part']}_{PART[a]}{suffix}", seed=r["seed"], concept=r["concept"], status="pending")
            for r in batch_runs(parts) for a in arms]


def save(path: str, st: dict) -> None:
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def record(j: dict, summ: dict, texts: dict) -> None:
    stages = {s["stage"]: s["seconds"] for s in summ.get("stages", [])}
    j.update(status="done" if summ.get("status") == "done" else "error", error=summ.get("error"), prompt_id=summ.get("prompt_id"), wall_s=summ.get("wall_s"),
             comfy_s=summ.get("total_seconds"), peak_vram_mib=summ.get("peak_vram_mib"), peak_ram_gib=summ.get("peak_ram_gib"), oom=summ.get("oom"),
             stages=stages, seconds={k: round(stages.get(v, 0.0), 1) for k, v in STAGE_KEYS.items()},
             final=((summ.get("outputs") or {}).get("OUT final") or [None])[0],
             repair_ran=any(n["id"] == "design_qwen2" and n["seconds"] > 0 and "Primitive" not in str(n.get("class_type")) for n in summ.get("nodes", [])),
             model_node_s=next((round(n["seconds"], 1) for n in summ.get("nodes", []) if n["id"] == "design_qwen"), None),
             repair_node_s=next((round(n["seconds"], 1) for n in summ.get("nodes", []) if n["id"] == "design_qwen2"), None))
    j["reply"] = (texts.get("OUT design reply") or [""])[0]
    j["parse_report"] = (texts.get("OUT design check") or [""])[0]
    j["final_report"] = (texts.get("OUT design final check") or [""])[0]


def run(a) -> int:
    os.makedirs(a.bench, exist_ok=True)
    sp = os.path.join(a.bench, "state.json")
    st = json.load(open(sp, encoding="utf-8")) if (a.resume and os.path.isfile(sp)) else None
    if st is None:
        st = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), iid=a.iid, port=a.port, target=a.target, jobs=jobs(a.runs.split(","), a.arms.split(","), a.suffix))
        save(sp, st)
    skip = set(filter(None, (a.skip_arms or "").split(",")))
    for j in st["jobs"]:
        if j["status"] == "done" or j["arm"] in skip or (a.only and j["part"] not in a.only.split(",")):
            continue
        gpath = os.path.join(ROOT, "app", "graphs", GRAPHS[j["arm"]])
        graph = json.load(open(gpath, encoding="utf-8"))
        sets = {"archive": f"redesign/src/{j['run']}_out_shots", "source": f"redesign/src/{j['run']}_final.mp4", "concept": json.dumps(j["concept"]), "seed": str(j["seed"])}
        j["status"], j["started"] = "running", time.strftime("%H:%M:%S")
        save(sp, st)
        out = os.path.join(a.bench, j["part"])
        proj = ["--idea", a.target, "--part", j["part"]]
        print(f"=== {j['part']}: {LABEL[j['arm']]} seed {j['seed']}", flush=True)
        try:
            summ = RT.run_graph(gpath, a.iid, a.port, sets, out, project=proj, timeout=a.timeout)
            texts = RB.history_texts(a.port, summ.get("prompt_id", ""), graph) if summ.get("prompt_id") else {}
            record(j, summ, texts)
            j["timing"] = out + ".json"
        except Exception as e:       # noqa: BLE001
            j.update(status="error", error=f"{type(e).__name__}: {e}")
        save(sp, st)
        print(f"    {j['status']}: wall {j.get('wall_s')} s, model {j.get('model_node_s')} s, peak VRAM {j.get('peak_vram_mib')} MiB, oom {bool(j.get('oom'))}", flush=True)
        if j["status"] != "done" and a.stop_on_error:
            break
    return 0


# --------------------------------------------------------------------------------------------- replay a saved reply
def replay(a) -> int:
    """run the graph fusion_redesign_replay.json with a SAVED model reply (no model call): the salvage / render half of an arm whose model call was done earlier
    (e.g. Gemma 4 12B: 1950 s for one reply on the 3060). The job record of `--part` in state.json is updated; the model seconds are the original call's."""
    sp = os.path.join(a.bench, "state.json")
    st = json.load(open(sp, encoding="utf-8"))
    j = next(x for x in st["jobs"] if x["part"] == a.part)
    gpath = os.path.join(ROOT, "app", "graphs", "fusion_redesign_replay.json")
    graph = json.load(open(gpath, encoding="utf-8"))
    for nid in ("design_parse", "design_fix"):
        graph[nid]["inputs"]["model"] = a.label
    reply = open(a.reply_file, encoding="utf-8").read()
    sets = {"archive": f"redesign/src/{j['run']}_out_shots", "source": f"redesign/src/{j['run']}_final.mp4", "concept": json.dumps(j["concept"]), "seed": str(j["seed"]),
            "reply": reply}
    out = os.path.join(a.bench, j["part"] + "_replay")
    summ = RT.run_graph(gpath, a.iid, a.port, sets, out, project=["--idea", a.target, "--part", j["part"]], graph_obj=graph, timeout=a.timeout)
    texts = RB.history_texts(a.port, summ.get("prompt_id", ""), graph) if summ.get("prompt_id") else {}
    keep = {k: j.get(k) for k in ("run", "arm", "part", "seed", "concept")}
    record(j, summ, texts)
    j.update(keep)
    j["model_node_s"], j["seconds"]["model"] = a.model_seconds, a.model_seconds
    j["wall_s"] = round((j.get("wall_s") or 0) + a.model_seconds, 1)
    j["note"] = (f"the model call ({a.model_seconds:.0f} s, {len(reply)} chars) ran in an earlier job; this job replays that reply through the field-level parse and renders; "
                 f"wall = model call + replay")
    j["reply"] = reply
    save(sp, st)
    print(f"    {j['status']}: replay wall {summ.get('wall_s')} s, peak VRAM {j.get('peak_vram_mib')} MiB")
    return 0


# --------------------------------------------------------------------------------------------- collect
NOTES_MD = """
## Notes

- Same footage (the batch v001 archives r01b / r03 / r05 / r07), same design seed per run for every arm, same prompt and contact sheet; wall = submit to last event,
  model s = the design call (load + prefill + generation), type / compose = the render of the designed piece.
- Jobs are named `<run>_<arm>_p2` (second pass, final code); the first pass (`<run>_<arm>`, parse code before the bracket-closing / minimal-timing / empty-lines fixes)
  is kept in `out/` and `16_app/bench/redesign_ab/`.
- 8-bit (`8-bit (Balanced)` of the design-LLM node pack): out of memory on the 12 GB card during the prefill of the ~7k-token prompt + sheet (weights ~10 GB); the node has no CPU offload.
- Gemma 4 12B: the Comfy-Org repo has NO 12B fp8 file (fp8 exists for E4B only): `gemma4_12b_int8_convrot.safetensors` (11.2 GiB) was used through native `CLIPLoader` +
  `TextGenerate` (thinking off, max_length 4096). Dynamic VRAM streams its weights over PCIe for every token: ~2 s per generated token on the 3060 (the one design reply
  took 1950 s), so it ran ONCE (r01b); its job here = that reply replayed through the field-level parse (graph `fusion_redesign_replay`) + the render.
- `model fields %` counts every design field the model wrote and the code kept as written; `model+nearest %` adds the fields mended to the nearest allowed value (incl. the
  piece-level rules: font mix, timing fit, window validator). Timing fields are the model's weak point: it writes hits that the beat grid cannot hold.
- `review/r01b_v001_vs_q4_vs_gemma.png`: the same 6 frames of r01b: v001 original / Qwen 4-bit redesign / Gemma 12B redesign.
- `review/replay_v001.md`: the RAW replies of the batch v001 through the new parse (0 fields from the rules in every run).
"""


def pull_outputs(a) -> None:
    """tar | ssh the version's files the dashboard sync does not carry (piece.json, index.json, still.png) from the box output dir into <version>/out/."""
    import io
    import tarfile
    key = os.path.expanduser("~/.ssh/vastai_comfyui_rsa")
    remote = (f"cd /workspace/ComfyUI/output/projects/{a.target}/ && find . \( -name '*.json' -o -name 'still.png' -o -name '*_video_*.mp4' \) -print0 | tar czf - --null -T -")
    p = subprocess.run(["ssh", "-p", str(a.ssh_port), "-i", key, f"root@{a.ssh_host}", remote], capture_output=True)
    dest = os.path.join(VERSION_DIR, "out")
    os.makedirs(dest, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as t:
            t.extractall(dest)
            print(f"pulled {len(t.getnames())} files into {dest}")
    except Exception as e:       # noqa: BLE001
        print(f"pull failed: {type(e).__name__}: {e}; ssh said: {p.stderr.decode(errors='replace')[:300]}")


def find_piece(part: str) -> dict | None:
    for p in glob.glob(os.path.join(VERSION_DIR, "out", f"*_{part}_out_shots", "piece.json")):
        return json.load(open(p, encoding="utf-8"))
    return None


def collect(a) -> int:
    st = json.load(open(os.path.join(a.bench, "state.json"), encoding="utf-8"))
    if a.pull:
        pull_outputs(a)
    if a.merge:                              # carry the first pass's results for the arms the second pass did not repeat (the 8-bit out-of-memory runs)
        p1 = {(j["run"], j["arm"]): j for j in json.load(open(a.merge, encoding="utf-8"))["jobs"]}
        for j in st["jobs"]:
            old = p1.get((j["run"], j["arm"]))
            if j["status"] == "pending" and old and old["status"] == "error":
                j.update({k: v for k, v in old.items() if k not in ("run", "arm", "seed", "concept")})
                j["note"] = (old.get("note") or "") + " (first pass; not repeated)"
    rows = []
    for j in st["jobs"]:
        piece = find_piece(j["part"])
        d = ((piece or {}).get("plan") or {}).get("design") or {}
        f = d.get("fields") or {}
        sec = j.get("seconds") or {}
        status, note = j["status"], j.get("note")
        if status == "pending":
            status = "not run"
            note = {"q8": "8-bit went out of memory on r01b and r03 (same prompt, same sheet): not repeated",
                    "gemma": "Gemma 4 12B int8 generates ~2 s per token on the 3060: one run only (r01b), see its row"}.get(j["arm"], note)
        rows.append(dict(
            run=j["run"], arm=j["arm"], label=LABEL[j["arm"]], part=j["part"], status=status, note=note, seed=j["seed"],
            pct_model=f.get("pct_model"), pct_model_or_fixed=f.get("pct_model_or_fixed"), n_fields=f.get("n"),
            fields_from_model=len(f.get("fields_from_model") or []), fields_fixed=len(f.get("fields_fixed") or []), fields_from_rules=len(f.get("fields_from_rules") or []),
            sections=f.get("sections"), src=d.get("src"), repair_ran=j.get("repair_ran"), pending_after_parse=None,
            wall_s=j.get("wall_s"), comfy_s=j.get("comfy_s"), model_s=j.get("model_node_s"), repair_s=j.get("repair_node_s"), type_s=sec.get("type"),
            compose_s=sec.get("compose"), flow_s=sec.get("flow"), assemble_s=sec.get("assemble"), peak_vram_mib=j.get("peak_vram_mib"), oom=bool(j.get("oom")),
            error=(j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error"), final=j.get("final"),
            reply_chars=len(j.get("reply") or ""), fields=f if piece else None, concept_words=(j.get("concept") or {}).get("words")))
    review = os.path.join(VERSION_DIR, "review")
    os.makedirs(review, exist_ok=True)
    json.dump(dict(rows=rows, state=os.path.relpath(os.path.join(a.bench, "state.json"), ROOT)), open(os.path.join(review, "AB.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(review, "AB.md"), "w", encoding="utf-8").write(markdown(rows) + NOTES_MD)
    print(f"wrote {review}/AB.json + AB.md ({len(rows)} jobs)")
    return 0


def markdown(rows: list[dict]) -> str:
    L = ["# Redesign A/B: design_llm_8b 4-bit vs 8-bit vs Gemma 4 12B, same footage, same seed", "",
         "Share of the design FIELDS (type + compose, locked text excluded) that are the model's as written (`model`), mended to the nearest allowed value (`fixed`: "
         "includes the piece-level rules) or taken from the rules planner (`rules`). `pct` = model / all fields.", "",
         "| run | arm | status | model fields % | model+nearest % | model / fixed / rules | wall s | model s | repair s | type s | compose s | peak VRAM MiB | notes |",
         "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        note = []
        if r.get("oom"):
            note.append("OOM")
        if r.get("error"):
            note.append(str(r["error"]).splitlines()[0][:90])
        if r.get("note"):
            note.append(r["note"])
        if r.get("repair_ran"):
            note.append("repair ran")
        if r.get("sections"):
            note.append(f"type {r['sections'].get('type')}, compose {r['sections'].get('compose')}")
        L.append(f"| {r['run']} | {r['label']} | {r['status']} | {r['pct_model'] if r['pct_model'] is not None else ''} | "
                 f"{r['pct_model_or_fixed'] if r['pct_model_or_fixed'] is not None else ''} | {r['fields_from_model']} / {r['fields_fixed']} / {r['fields_from_rules']} | "
                 f"{r['wall_s'] or ''} | {r['model_s'] if r['model_s'] is not None else ''} | {r['repair_s'] if r['repair_s'] is not None else ''} | "
                 f"{r['type_s'] if r['type_s'] is not None else ''} | {r['compose_s'] if r['compose_s'] is not None else ''} | {r['peak_vram_mib'] or ''} | {'; '.join(note)} |")
    by: dict = {}
    for r in rows:
        if r["status"] == "done" and r["pct_model"] is not None:
            by.setdefault(r["label"], []).append(r)
    if by:
        L += ["", "## Mean per arm", "", "| arm | jobs | model fields % | model+nearest % | wall s | model s | repair s |", "|---|---:|---:|---:|---:|---:|---:|"]
        for lab, rs in by.items():
            mean = lambda k: round(sum((x[k] or 0) for x in rs) / len(rs), 1)       # noqa: E731
            L.append(f"| {lab} | {len(rs)} | {mean('pct_model')} | {mean('pct_model_or_fixed')} | {mean('wall_s')} | {mean('model_s')} | {mean('repair_s')} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--iid", default="57999748")
        p.add_argument("--port", type=int, default=9000)
        p.add_argument("--target", default=TARGET)
        p.add_argument("--bench", default=BENCH)
        p.add_argument("--runs", default="r01b,r03,r05,r07")
        p.add_argument("--arms", default="q4,q8,gemma")
        p.add_argument("--suffix", default="", help="appended to every part name (a second pass over the same runs, e.g. _p2)")
        p.add_argument("--ssh-host", default="<box-ip>")
        p.add_argument("--ssh-port", type=int, default=60181)
    sub.choices["run"].add_argument("--resume", action="store_true")
    sub.choices["run"].add_argument("--timeout", type=float, default=3600)
    sub.choices["run"].add_argument("--stop-on-error", action="store_true")
    sub.choices["run"].add_argument("--skip-arms", default="", help="arms to leave out of this pass (e.g. gemma: ~2 s per generated token on the 3060)")
    sub.choices["run"].add_argument("--only", default="", help="comma list of job parts to run (e.g. r01b_g4)")
    sub.choices["collect"].add_argument("--pull", action="store_true")
    sub.choices["collect"].add_argument("--merge", default="", help="state.json of an earlier pass whose error jobs fill this pass's pending ones")
    rp = sub.add_parser("replay")
    for flag, default in (("--iid", "57999748"), ("--target", TARGET), ("--bench", BENCH)):
        rp.add_argument(flag, default=default)
    rp.add_argument("--port", type=int, default=9000)
    rp.add_argument("--part", required=True)
    rp.add_argument("--reply-file", required=True)
    rp.add_argument("--label", default="replayed-reply")
    rp.add_argument("--model-seconds", type=float, default=0.0)
    rp.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else (replay(a) if a.cmd == "replay" else collect(a))


if __name__ == "__main__":
    raise SystemExit(main())
