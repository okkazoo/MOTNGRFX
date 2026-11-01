#!/usr/bin/env python
"""Autonomous batch: N pieces, each invented by design_llm on the box (graph fusion_design_qwen_auto), timed. A plain driver: no LLM of its own.

    python tools/app/run_batch.py --iid 12345678 --port 8190                      # 10 runs, length 16, outputs into a NEW PROJECTS version
    python tools/app/run_batch.py --iid 12345678 --port 8190 --resume             # after a crash / ^C: continues where state.json stops
    python tools/app/run_batch.py --iid I --port P --n 3 --no-project --out 16_app/bench/batch_try

For i in 1..N it submits the auto graph of the run's footage route (`--video-plan`, all LTX; the mattes are LTX Alpha-Gen) through progress_standin.py (via tools/app/run_timing.py: per-node / per-stage seconds, peak VRAM, wall) with
`seed = i`, `length = 16` (fixed, so the timings compare; the music cut then lands on the downbeat nearest 16 s) and `previous` = the concepts of the
runs before it (read back from each finished run's ComfyUI history: the `OUT concept` PreviewAny text). The model decides the words, scene, music,
look and shots mode; the code decides the number of shots.

State: <out>/state.json (written after every step, atomically). --resume skips the runs that are `done`, re-submits one that was `running` when the
driver died (a half-finished ComfyUI prompt is not adopted), and reuses the PROJECTS version recorded in the state. Report: <out>/BATCH.json + BATCH.md
(per run: concept, mode, shots, stills, real cut length, wall, stage seconds, peak VRAM, OOM, whether a repair ran, the final mp4).

PROJECTS: `proj.py version-start COMPETITIONS/motion-graphic/<idea> --why "3060 autonomous batch: ..."` once at the start (idea default `app`; the idea must
exist: `proj.py new-idea COMPETITIONS/motion-graphic/app --kind video ...`); every run then joins that version (`progress_standin --idea <P>/<idea>/vNNN
--part rNN`), so all outputs land in its out/ folder. --no-project skips this (outputs stay where ComfyUI puts them).
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "app"))
import run_timing as RT      # noqa: E402
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.fusion import still_preview as SP      # noqa: E402

GRAPH = os.path.join(ROOT, "app", "graphs", "fusion_design_qwen_auto.json")           # footage route ltx: LTX 2.5 IC-LoRA shots
ROUTES = ("ltx",)
ASPECTS = ("9:16", "16:9")        # the page's Format select (app.json field `format`, node `IN format`); MographPieceSizes turns it into the sizes inside the graph
PROJ = os.path.expanduser("~/.claude/skills/project/proj.py")
PROJECT = "COMPETITIONS/motion-graphic"
WHY = "3060 autonomous batch: {n} Qwen-made pieces, open models only, timed"
SUBPROCESS_S = 120.0              # proj.py calls
STEP_S = {"matte_preview": 600.0, "model_replies": 300.0, "ingredients": 1800.0}      # finalize steps: hard ceilings (seconds)
MIN_CEILING_S, CEILING_X, DEFAULT_EXPECTED_WALL_S = 1800.0, 3.0, 1200.0               # per-run ceiling = max(30 min, 3 x expected wall)
HANG_LOG = [None]                 # open file of <out>/hang_stacks.log (faulthandler needs a real file object that stays open)
SUMMARY_RE = re.compile(r"piece: mode=(\S+) shots=(\d+) stills=(\d+) cut=([\d.]+)s frames=(\d+) words=(\d+) technique=(\S+) quality=(\S+)")


# --------------------------------------------------------------------------------------------- state
def default_plan(n: int) -> list[str]:
    """all LTX."""
    return ["ltx"] * n


def parse_plan(text: str | None, n: int) -> list[str]:
    plan = [t.strip().lower() for t in text.split(",") if t.strip()] if text else default_plan(n)
    bad = [t for t in plan if t not in ROUTES]
    if bad or len(plan) != n:
        raise SystemExit(f"--video-plan needs {n} entries from {', '.join(ROUTES)}, got {plan!r}")
    return plan


def graph_aspect(graph: dict | None) -> str:
    """the aspect a graph carries by itself: the value of its `IN format` primitive (9:16 in every app graph)."""
    for n in (graph or {}).values():
        if isinstance(n, dict) and (n.get("_meta") or {}).get("title") == "IN format":
            return str((n.get("inputs") or {}).get("value") or "9:16")
    return "9:16"


def new_state(a: dict) -> dict:
    plan = a.get("video_plan") or default_plan(a["n"])
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "graph": a["graph"], "graphs": {"ltx": a["graph"]},
            "length": a["length"], "aspect": a.get("aspect"), "n": a["n"], "iid": a["iid"], "port": a["port"], "version": None, "target": None,
            "runs": [{"i": i, "seed": i + int(a.get("seed_base") or 0), "video": plan[i - 1], "status": "pending"} for i in range(1, a["n"] + 1)]}


def save_state(path: str, st: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_state(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def previous_of(st: dict, i: int) -> list[dict]:
    """concepts of the runs before run i that have one (a failed run whose concept node had finished still counts: the next must differ from it)."""
    return [r["concept"] for r in st["runs"] if r["i"] < i and r.get("concept")]


def earlier_concepts(paths) -> list:
    """concepts of the finished runs of earlier batches (state.json files), in order, no duplicates."""
    out = []
    for p in paths or []:
        st = load_state(p) or {"runs": []}
        out += [r["concept"] for r in st["runs"] if r.get("concept") and r.get("status") == "done" and r["concept"] not in out]
    return out


def todo(st: dict) -> list[dict]:
    """runs to (re)submit: everything not done. `running` = the driver died mid-run: submitted again."""
    return [r for r in st["runs"] if r["status"] != "done"]


def reset_interrupted(st: dict) -> int:
    n = 0
    for r in st["runs"]:
        if r["status"] == "running":
            r["status"] = "pending"
            r["restarted"] = r.get("restarted", 0) + 1
            n += 1
    return n


# --------------------------------------------------------------------------------------------- reading a finished run back
def history_texts(port: int, prompt_id: str, graph: dict, budget: float = 150.0) -> dict:
    """{node title: [text, ...]} of the PreviewAny outputs of /history/<id>."""
    h = {}
    t_end = time.time() + budget                # total, not per request (a stalled tunnel used to cost 40 x 30 s)
    for _ in range(40):
        try:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=min(30, max(1.0, t_end - time.time())))).get(prompt_id, {})
        except Exception:       # noqa: BLE001
            h = {}
        if h.get("outputs") or time.time() > t_end:
            break
        time.sleep(0.5)
    out = {}
    for nid, o in (h.get("outputs") or {}).items():
        if o.get("text"):
            out[(graph.get(nid, {}).get("_meta") or {}).get("title", nid)] = [t if isinstance(t, str) else json.dumps(t) for t in o["text"]]
    return out


def parse_summary(texts: dict) -> dict:
    """mode / shots / stills / real cut length from the `piece:` line of the design reports."""
    for t in texts.get("OUT design final check", []) + texts.get("OUT design check", []):
        m = SUMMARY_RE.search(t)
        if m:
            return dict(mode=m.group(1), shots=int(m.group(2)), stills=int(m.group(3)), cut_s=float(m.group(4)), frames=int(m.group(5)),
                        words=int(m.group(6)), technique=m.group(7), quality=m.group(8))
    return {}


def concept_of(texts: dict) -> dict | None:
    for t in texts.get("OUT concept", []):
        try:
            c = json.loads(t)
            if isinstance(c, dict) and c.get("words"):
                return c
        except ValueError:
            continue
    return None


def record_run(r: dict, summ: dict, texts: dict) -> dict:
    """fold one run_timing summary + the history texts into the run record."""
    stages = {s["stage"]: s for s in summ.get("stages", [])}
    r.update(status="done" if summ.get("status") == "done" else "error", error=summ.get("error"), prompt_id=summ.get("prompt_id"),
             wall_s=summ.get("wall_s"), comfy_s=summ.get("total_seconds"), peak_vram_mib=summ.get("peak_vram_mib"), peak_ram_gib=summ.get("peak_ram_gib"),
             oom=summ.get("oom"), stages={k: v["seconds"] for k, v in stages.items()}, timing=r.get("timing"),
             repair_design_ran=any(k.startswith("Qwen repair") and v["seconds"] > 0 for k, v in stages.items()),
             repair_concept_ran=any(n["id"] == "concept_qwen2" for n in summ.get("nodes", [])),
             final_draft=((summ.get("outputs") or {}).get("OUT final") or [None])[0])
    f1080 = ((summ.get("outputs") or {}).get("OUT final 1080") or [None])[0]          # 9:16 final: the in-graph Real-ESRGAN x2 file (`_video2`)
    if f1080:
        r["final_1080"] = f1080
    r["final"] = f1080 or r["final_draft"]                    # the biggest file the run made (16:9 finals: run_batch's own upscale step below replaces it)
    r.update(parse_summary(texts))
    c = concept_of(texts)
    if c:
        r["concept"] = c
    from tools.fusion import openings as OP                       # how this piece OPENED (word 1: form_in / entry / module, first cut): the next runs are told to open differently (openings.py)
    op = next((o for o in (OP.parse_summary(t) for t in texts.get("OUT design final check", []) + texts.get("OUT design check", [])) if o), None)
    if op and isinstance(r.get("concept"), dict):
        r["concept"]["opening"] = op
        r["opening"] = op
    r["design_reply"] = (texts.get("OUT design reply") or [""])[0][:6000]
    r["design_check"] = (texts.get("OUT design check") or [""])[0][:3000]
    r["concept_check"] = (texts.get("OUT concept check") or [""])[0][:2000]
    r["fixes"] = {"concept": fix_lines((texts.get("OUT concept check") or [""])[0]),
                  "design": fix_lines((texts.get("OUT design check") or [""])[0]),
                  "design_final": fix_lines((texts.get("OUT design final check") or [""])[0])}
    for t in texts.get("Archive (still, shots, mattes)", []):
        try:
            r["archive"] = json.loads(t)
        except ValueError:
            pass
    return r


def fix_lines(text: str) -> list[str]:
    """the `fix:` / `error:` / header lines of a parse or repair report (the code fixes and the model's mistakes)."""
    return [l.strip() for l in (text or "").splitlines() if "fix:" in l or " error:" in l or "design valid" in l or l.lstrip().startswith(("concept", "clash", "rules"))]


# --------------------------------------------------------------------------------------------- PROJECTS
def version_start(idea: str, why: str, runner=subprocess.run) -> tuple[str | None, str]:
    """-> (target `P/idea/vNNN`, message). Does not create the idea."""
    try:
        p = runner([sys.executable, PROJ, "version-start", f"{PROJECT}/{idea}", "--why", why], capture_output=True, text=True, timeout=SUBPROCESS_S)
    except subprocess.TimeoutExpired:
        return None, f"proj.py version-start timed out after {SUBPROCESS_S:.0f}s"
    try:
        d = json.loads(p.stdout)
    except ValueError:
        return None, (p.stdout + p.stderr).strip()[:600]
    if not d.get("ok", True) or not d.get("version"):
        return None, json.dumps(d)[:600]
    return f"{PROJECT}/{idea}/{d['version']}", f"version {d['version']}"


# --------------------------------------------------------------------------------------------- report
def design_model_label(st: dict) -> str:
    """the design model named in the report: read from the graph file names of the batch (final_draft_gemma.json -> Gemma), never a fixed 'design_llm_8b' (the v014 report said Qwen while Gemma ran)."""
    names = " ".join(os.path.basename(g).lower() for g in ((st.get("graphs") or {}).values() or [st.get("graph") or ""]))
    for key, label in (("gemma", "Gemma 4"),):
        if key in names:
            return label
    return "the design model"


def report_md(st: dict) -> str:
    dm = design_model_label(st)
    L = [f"# Batch: {st['n']} pieces invented by {dm} ({', '.join(os.path.basename(g) for g in (st.get('graphs') or {'ltx': st['graph']}).values())}), "
         f"length {st['length']} s" + (f", aspect {st['aspect']}" if st.get("aspect") else ""),
         "", f"started {st['started']}; box {st['iid']}:{st['port']}; PROJECTS version: {st.get('target') or 'none (--no-project)'}", "",
         *([""] + [f"- {n}" for n in st.get("notes", [])] + [""] if st.get("notes") else []),
         "| # | video | aspect | seed | title / subject | mood | mode | shots | stills | cut s | words | wall s | design model s | peak VRAM MiB | OOM | repair c/d | status |",
         "|---:|---|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|"]
    for r in st["runs"]:
        c = r.get("concept") or {}
        q = (r.get("stages") or {}).get("Qwen design")
        L.append(f"| {r['i']} | {r.get('video', 'ltx')} | {r.get('aspect') or st.get('aspect') or ''} | {r['seed']} | {c.get('title', '')} / {c.get('subject', '')} | {c.get('mood', '')} | {r.get('mode', c.get('shots', ''))} | "
                 f"{r.get('shots', '')} | {r.get('stills', '')} | {r.get('cut_s', '')} | {' / '.join(c.get('words', []))} | {r.get('wall_s', '')} | "
                 f"{q if q is not None else ''} | {r.get('peak_vram_mib', '')} | {'yes' if r.get('oom') else 'no'} | "
                 f"{'Y' if r.get('repair_concept_ran') else '-'}/{'Y' if r.get('repair_design_ran') else '-'} | {r['status']} |")
    done = [r for r in st["runs"] if r["status"] == "done"]
    if done:
        names = []
        for r in done:
            for k in (r.get("stages") or {}):
                if k not in names:
                    names.append(k)
        L += ["", "## Seconds per stage (done runs)", "", "| stage | runs | total s | mean s | min s | max s |", "|---|---:|---:|---:|---:|---:|"]
        tot = {}
        for k in names:
            v = [r["stages"][k] for r in done if k in (r.get("stages") or {})]
            tot[k] = sum(v)
            L.append(f"| {k} | {len(v)} | {sum(v):.1f} | {sum(v) / len(v):.1f} | {min(v):.1f} | {max(v):.1f} |")
        if tot:
            top = max(tot, key=tot.get)
            L.append(f"\nbottleneck: **{top}** = {tot[top]:.0f} s of {sum(tot.values()):.0f} s node time ({100 * tot[top] / max(sum(tot.values()), 1e-9):.0f} %).")
        w = [r["wall_s"] for r in done if r.get("wall_s")]
        if w:
            L.append(f"\nwall per piece: mean {sum(w) / len(w):.0f} s, min {min(w):.0f}, max {max(w):.0f}; batch total {sum(w) / 60:.1f} min.")
    for route in ROUTES:
        rr = [r for r in done if r.get("video", "ltx") == route]
        if not rr:
            continue
        names = []
        for r in rr:
            for k in (r.get("stages") or {}):
                if k not in names:
                    names.append(k)
        L += ["", f"## Route {route.upper()}: {len(rr)} done run(s), runs {', '.join(str(r['i']) for r in rr)}", "", "| stage | mean s | min s | max s |", "|---|---:|---:|---:|"]
        for k in names:
            v = [r["stages"][k] for r in rr if k in (r.get("stages") or {})]
            L.append(f"| {k} | {sum(v) / len(v):.1f} | {min(v):.1f} | {max(v):.1f} |")
        w = [r["wall_s"] for r in rr if r.get("wall_s")]
        vr = [r["peak_vram_mib"] for r in rr if r.get("peak_vram_mib")]
        if w:
            L.append(f"\nwall: mean {sum(w) / len(w):.0f} s (min {min(w):.0f}, max {max(w):.0f}); peak VRAM max {max(vr) if vr else 'n/a'} MiB.")
    L += ["", "## Concepts and outputs"]
    for r in st["runs"]:
        c = r.get("concept") or {}
        L += ["", f"### {r['i']}. {c.get('title', '(no concept)')} ({r['status']}, footage {r.get('video', 'ltx')})", f"- words: {' / '.join(c.get('words', []))}", f"- scene: {c.get('scene', '')}",
              f"- music: {c.get('music', '')}; look {c.get('look', '')}; shots {c.get('shots', '')}; concept source: {c.get('source', '')}",
              f"- final mp4 (ComfyUI path): {r.get('final')}; 1080 file: {r.get('final_1080')}; draft render: {r.get('final_draft')}; error: {r.get('error')}"
              + (f"; 16:9 upscale error: {r['upscale_error']['message']}" if r.get("upscale_error") else "")]
        L += run_detail(r)
    return "\n".join(L) + "\n"


def run_detail(r: dict) -> list[str]:
    """design (fonts / modules / window cuts / technique), code fixes, timings, files of one run."""
    L = []
    a = r.get("archive") or {}
    d = a.get("design") or {}
    if d:
        L.append(f"- design: technique {d.get('technique')}; sections from {d.get('src')}; camera {d.get('camera')}; transitions {d.get('transitions')}")
        for w in d.get("words", []):
            L.append(f"  - {w.get('text')!r}: font {w.get('font')}, module {w.get('module')}, in {w.get('form_in')} / out {w.get('form_out')}, "
                     f"t {w.get('t0')}-{w.get('t1')}" + (f", window {w.get('window_source')}" if w.get("window_source") else ""))
        L.append(f"  - window cuts: {json.dumps(d.get('window_cuts'), ensure_ascii=False) if d.get('window_cuts') else 'none'}; compose ops: {', '.join(d.get('compose_ops') or [])}")
    fx = r.get("fixes") or {}
    for k in ("concept", "design", "design_final"):
        if fx.get(k):
            L.append(f"- code fixes / reports ({k}):")
            L += [f"  - {x}" for x in fx[k]]
    if r.get("stages"):
        L.append("- stage seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in r["stages"].items()))
    L.append(f"- wall {r.get('wall_s')} s, ComfyUI {r.get('comfy_s')} s, peak VRAM {r.get('peak_vram_mib')} MiB, peak RAM {r.get('peak_ram_gib')} GiB, OOM {r.get('oom') or 'no'}, "
             f"restarts {r.get('restarted', 0)}, concept repair {r.get('repair_concept_ran')}, design repair {r.get('repair_design_ran')}")
    f = r.get("files") or {}
    if f:
        L.append(f"- files: final {f.get('final')} (1080 {f.get('final_1080')}, draft {f.get('final_draft')}); still {f.get('still')}; shots {', '.join(f.get('clips') or [])}")
        L.append(f"  - mattes (PNG per frame): {', '.join(f.get('mattes') or [])}; matte preview {f.get('matte_preview')}; piece json {f.get('piece')}")
    return L


def version_dir(target: str, runner=subprocess.run) -> str | None:
    try:
        p = runner([sys.executable, PROJ, "resolve", "--target", target], capture_output=True, text=True, timeout=SUBPROCESS_S)
        return (json.loads(p.stdout).get("target") or {}).get("dir")
    except (ValueError, subprocess.TimeoutExpired):
        return None


def collect_files(vdir: str, part: str) -> dict:
    """what sync brought into <version>/out for a run's part label (`_rNN_video*.mp4`, `_rNN_out_shots/`)."""
    tag = f"_{part}_"
    f = {"final": None, "final_1080": None, "final_draft": None, "still": None, "clips": [], "mattes": [], "piece": None, "shots_dir": None}
    mp4 = []
    for root, dirs, files in os.walk(os.path.join(vdir, "out")):
        dirs.sort()
        for n in sorted(files):
            if tag in n and n.lower().endswith(".mp4") and "_shots" not in root:
                mp4.append(os.path.join(root, n))
        for d in dirs:
            if tag in d and d.endswith("_shots"):
                f["shots_dir"] = os.path.join(root, d)
    big = [p for p in mp4 if re.search(r"_up1080|_video2", os.path.basename(p).lower())]           # the 1080 files: 9:16 in-graph upscale (`_video2`), 16:9 run_batch upscale (`_up1080`)
    small = [p for p in mp4 if p not in big]
    f["final_1080"], f["final_draft"] = (sorted(big)[-1] if big else None), (sorted(small)[-1] if small else None)
    f["final"] = f["final_1080"] or f["final_draft"]            # deterministic: prefer the 1080 output, else the draft render
    sd = f["shots_dir"]
    if sd:
        for n in sorted(os.listdir(sd)):
            p = os.path.join(sd, n)
            if n == "still.png":
                f["still"] = p
            elif n.endswith("_clip.mp4"):
                f["clips"].append(p)
            elif n.endswith("_matte") and os.path.isdir(p):
                f["mattes"].append(f"{p} ({len([x for x in os.listdir(p) if x.endswith('.png')])} png)")
            elif n == "piece.json":
                f["piece"] = p
    return f


def reconcile(st: dict, out: str, log=print) -> int:
    """a run whose relay log ends in `RESULT OK` but was recorded as `error` (progress_standin exited on DONE before ComfyUI's last websocket event,
    run_timing before 2026-10-28 11:00) is re-summarised from its relay log and re-read from the ComfyUI history. -> number repaired."""
    n = 0
    for r in st["runs"]:
        lp = os.path.join(out, f"run_{r['i']:02d}.relay.log")
        if r["status"] != "error" or r.get("error") or not os.path.isfile(lp) or "RESULT OK" not in open(lp, encoding="utf-8").read():
            continue
        graph_path = st["graphs"][r.get("video", "ltx")]
        graph = json.load(open(graph_path, encoding="utf-8"))
        summ = RT.from_log(lp, graph_path)
        old = json.load(open(os.path.join(out, f"run_{r['i']:02d}.json"), encoding="utf-8"))
        for k in ("wall_s", "prompt_id", "outputs", "set", "port", "iid"):
            summ.setdefault(k, old.get(k))
        summ["wall_s"] = old.get("wall_s")
        texts = history_texts(st["port"], summ.get("prompt_id", ""), graph) if summ.get("prompt_id") else {}
        record_run(r, summ, texts)
        r["reconciled"] = True
        n += 1
        log(f"reconcile run {r['i']}: {r['status']}")
    return n


def finalize(st: dict, out: str, log=print) -> dict:
    """after the runs (and after the dashboard sync landed the files): paths of the finals / stills / shots / mattes into each run, one matte preview
    mp4 per run in <version>/review/, then the reports again. Safe to repeat (`--finalize`)."""
    import matte_preview as MP
    reconcile(st, out, log)
    vdir = version_dir(st["target"]) if st.get("target") else None
    if not vdir:
        log("finalize: no PROJECTS version; nothing to collect")
        pass      # model-reply archiving is a private dev tool (not in the export)
        return st
    for r in st["runs"]:
        f = collect_files(vdir, r.get("part") or f"r{r['i']:02d}")
        if f["shots_dir"] and f["mattes"]:
            name = os.path.basename(f["shots_dir"])[: -len("_shots")]
            res = bounded(lambda: MP.build(f["shots_dir"], os.path.join(vdir, "review", f"{name}_matte_preview.mp4")), STEP_S["matte_preview"],
                          f"matte preview {name}", log, out=out)           # ffmpeg over the matte PNGs: bounded (and by its own timeout)
            f["matte_preview"] = res["path"] if res else None
            if not res:
                f["matte_preview_error"] = "failed or timed out (see the log)"
        r["files"] = f
        log(f"finalize run {r['i']}: final {bool(f['final'])}, clips {len(f['clips'])}, mattes {len(f['mattes'])}, preview {f.get('matte_preview')}")
        if f["final"]:                                    # user 2026-10-30: every finished piece joins the dashboard Outputs selection set (best effort; skipped without the dashboard)
            try:
                api, grp = os.environ.get("MOGRAPH_DASHBOARD", "http://127.0.0.1:8000"), os.environ.get("MOGRAPH_OUTPUT_GROUP", "motngrfx-tests")
                rel = os.path.relpath(f["final"], os.environ.get("PROJECTS_ROOT") or os.path.join(os.path.expanduser("~"), "Documents", "Dev", "PROJECTS")).replace("\\", "/")
                gs = json.load(urllib.request.urlopen(f"{api}/api/outputs/groups", timeout=5))
                gid = next((g["id"] for g in (gs.get("groups", gs) if isinstance(gs, dict) else gs) if g.get("name") == grp), None)
                url, body = (f"{api}/api/outputs/groups/{gid}/members", {"paths": [f"@projects/{rel}"]}) if gid else (f"{api}/api/outputs/groups", {"name": grp, "paths": [f"@projects/{rel}"]})
                urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
                log(f"output group {grp!r}: + {rel}")
            except Exception as e:                        # noqa: BLE001 - a missing dashboard never fails a batch
                log(f"output group: skipped ({type(e).__name__})")
    st["version_dir"] = vdir
    write_reports(out, st)
    pass      # model-reply archiving is a private dev tool (not in the export)
    bounded(lambda: pull_ingredients(st, vdir, log), STEP_S["ingredients"], "ingredients pull", log, out=out)       # ssh/tar from the box: the 2026-10-29 hang sat here; after the replies, which are linked in too
    return st


def pull_ingredients(st: dict, vdir: str, log=print) -> list:
    """everything of each done run together under <version>/ingredients/<stem>/ (tools/app/ingredients.py; 16_app/INGREDIENTS.md). Never stops finalize."""
    try:
        import ingredients as ING
        box = ING.box_info(st["iid"]) if st.get("iid") else None
        recs = {r.get("part") or f"r{r['i']:02d}": r for r in st["runs"]}
        return [ING.pull_run(vdir, run, box, recs.get(run.get("part")), log) for run in ING.find_runs(vdir)]
    except Exception as e:       # noqa: BLE001
        log(f"finalize: ingredients not gathered: {type(e).__name__}: {str(e)[:300]}")
        return []


def write_reports(out: str, st: dict) -> None:
    save_state(os.path.join(out, "state.json"), st)
    json.dump(st, open(os.path.join(out, "BATCH.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(out, "BATCH.md"), "w", encoding="utf-8").write(report_md(st))


# --------------------------------------------------------------------------------------------- hang guards
def hang_log(out: str):
    path = os.path.join(out, "hang_stacks.log")
    if HANG_LOG[0] is None or HANG_LOG[0].closed or HANG_LOG[0].name != path:
        if HANG_LOG[0] is not None and not HANG_LOG[0].closed:
            HANG_LOG[0].close()
        HANG_LOG[0] = open(path, "a", encoding="utf-8")
    return HANG_LOG[0]


def arm_hang_dump(out: str, seconds: float, label: str = ""):
    """faulthandler.dump_traceback_later: if this process is still inside the guarded section after `seconds`, every thread's stack goes to
    <out>/hang_stacks.log (works on Windows, needs no signal, runs from a C thread so it fires even when Python is blocked). -> cancel()."""
    try:
        f = hang_log(out)
        f.write(f"--- armed {time.strftime('%H:%M:%S')} for {seconds:.0f}s: {label}\n")
        f.flush()
        faulthandler.dump_traceback_later(max(1.0, seconds), repeat=False, file=f)
    except Exception:       # noqa: BLE001  a stream without fileno, etc.: the guard is best effort
        return lambda: None

    def cancel():
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:       # noqa: BLE001
            pass
    return cancel


def bounded(fn, seconds: float, label: str, log=print, default=None, out: str | None = None):
    """fn() in a daemon thread, waited for at most `seconds`. Done -> its result. Not done -> all stacks to the log (and <out>/hang_stacks.log), one line saying
    what was abandoned, `default` returned and the batch goes on (the thread is left behind; it cannot block interpreter exit). fn raising -> logged, `default`."""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except BaseException as e:       # noqa: BLE001
            box["e"] = e
    t = threading.Thread(target=work, daemon=True, name=f"bounded-{label}")
    t.start()
    t.join(seconds)
    if t.is_alive():
        log(f"WATCHDOG {label}: still running after {seconds:.0f}s - abandoned, going on (stacks in hang_stacks.log)")
        for f in ([hang_log(out)] if out else []):
            try:
                f.write(f"--- {time.strftime('%H:%M:%S')} {label} abandoned after {seconds:.0f}s\n")
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
                f.flush()
            except Exception:       # noqa: BLE001
                pass
        return default
    if "e" in box:
        log(f"{label}: failed ({type(box['e']).__name__}: {str(box['e'])[:300]})")
        return default
    return box.get("v", default)


def run_ceiling(st: dict, a: dict) -> float:
    """hard ceiling of one run in seconds: `timeout` when given (> 0), else max(30 min, 3 x the mean wall of the runs done so far; 20 min assumed for the first)."""
    if a.get("timeout"):
        return float(a["timeout"])
    walls = [r["wall_s"] for r in st["runs"] if r.get("status") == "done" and r.get("wall_s")]
    expected = sum(walls) / len(walls) if walls else float(a.get("expected_wall_s") or DEFAULT_EXPECTED_WALL_S)
    return max(MIN_CEILING_S, CEILING_X * expected)


# --------------------------------------------------------------------------------------------- the 16:9 final upscale
UP_PART = "_up1080"
UP_TIMEOUT_S = 3600.0             # the page's side-graph ceiling (app.js finalThisClip)
DRAFT_SIZE = {"9:16": (540, 960), "16:9": (960, 540)}      # compose size of the draft tier = what APP_FINAL_TIER renders


def wants_batch_upscale(r: dict, quality: str | None) -> bool:
    """16:9 final: the graph's own `OUT final 1080` only runs for 9:16 (final_upscale.wants_upscale), so run_batch does the page's Final (1080) step itself on the finished clip."""
    return quality == "final" and r.get("aspect") == "16:9" and r.get("status") == "done" and bool(r.get("final_draft")) and not r.get("final_1080")


def build_upscale_graph(clip: str, fmt: str = "16:9") -> tuple[dict, str]:
    """-> (API graph, save prefix): final_upscale.standalone_graph on the finished clip `<subfolder>/<name>.mp4` (as /history lists it) of the draft tier; the new file sits next to it as <stem>_up1080 (the page's rule)."""
    from tools.app import final_upscale as FU
    clip = clip.replace("\\", "/")
    sub, _, name = clip.rpartition("/")
    stem = re.sub(r"_$", "", re.sub(r"_?\d{5}_?$", "", re.sub(r"\.[^.]+$", "", name)))
    prefix = (sub + "/" if sub else "") + stem + UP_PART
    return FU.standalone_graph(clip + " [output]", DRAFT_SIZE[fmt], fmt, prefix), prefix


def upscale_final(r: dict, st: dict, a: dict, out: str, run_graph, log=print) -> None:
    """submit the 16:9 upscale graph of run r through the same run_graph (progress_standin submit) as the run, wait, record r['final_1080'] (and r['final'] = it). Never fails the run."""
    i = r["i"]
    try:
        g, prefix = build_upscale_graph(r["final_draft"], "16:9")
        gpath = os.path.join(out, f"run_{i:02d}{UP_PART}_in.json")
        json.dump(g, open(gpath, "w", encoding="utf-8"), indent=1)
        proj = ["--idea", st["target"], "--part", r["part"] + UP_PART] if st.get("target") else []
        log(f"[{i}/{st['n']}] 16:9 final: upscaling {r['final_draft']} -> 1920x1080 ({prefix})")
        s2 = run_graph(gpath, st["iid"], st["port"], {}, os.path.join(out, f"run_{i:02d}{UP_PART}"), project=proj, timeout=UP_TIMEOUT_S, graph_obj=g)
        up = ((s2.get("outputs") or {}).get("OUT upscaled") or [None])[0]
        if s2.get("status") != "done" or not up:
            raise RuntimeError(f"upscale run {s2.get('status')}: {s2.get('error')}")
        r["final_1080"], r["final"], r["upscale_wall_s"] = up, up, s2.get("wall_s")
    except Exception as e:       # noqa: BLE001  the 960x540 render stays the final; the batch goes on
        r["upscale_error"] = {"type": type(e).__name__, "message": str(e)[:400]}
        log(f"[{i}/{st['n']}] 16:9 upscale failed: {r['upscale_error']['message']}")


# --------------------------------------------------------------------------------------------- the loop
def run_batch(a: dict, *, run_graph=RT.run_graph, texts_of=history_texts, start_version=version_start, log=print) -> dict:
    """a: iid, port, n, length, graph, out, idea, project (bool), resume (bool), timeout. The three callables are injectable (tests)."""
    out = a["out"]
    os.makedirs(out, exist_ok=True)
    sp = os.path.join(out, "state.json")
    st = load_state(sp) if a.get("resume") else None
    if st is None:
        if os.path.isfile(sp) and not a.get("resume"):
            raise SystemExit(f"{sp} exists: --resume to continue it, or choose another --out (a batch is never overwritten)")
        st = new_state(a)
    elif reset_interrupted(st):
        log("resume: a run was interrupted mid-way; it is submitted again")
    if a.get("project") and not st.get("target"):
        target, msg = start_version(a.get("idea", "app"), WHY.format(n=st["n"]))
        if not target:
            raise SystemExit("proj.py version-start failed: " + msg + f"\n  (the idea must exist: python {PROJ} new-idea {PROJECT}/{a.get('idea', 'app')} --kind video --title ... --question ...; "
                             "or run with --no-project)")
        st["target"] = target
        log(f"PROJECTS {msg}: {target}")
    write_reports(out, st)
    graphs = st.get("graphs") or {"ltx": st["graph"]}
    need = {r.get("video", "ltx") for r in todo(st)}
    missing = [f"{k}: {graphs.get(k)}" for k in sorted(need) if not (graphs.get(k) and os.path.isfile(graphs[k]))]
    if missing:
        raise SystemExit("graph file missing for the video plan: " + "; ".join(missing) + "")
    loaded = {k: json.load(open(graphs[k], encoding="utf-8")) for k in need}
    for r in todo(st):
        graph, route = loaded[r.get("video", "ltx")], r.get("video", "ltx")
        i = r["i"]
        r["status"], r["started"] = "running", time.strftime("%H:%M:%S")
        write_reports(out, st)
        prev = list(a.get("earlier") or []) + previous_of(st, i)          # `earlier` = concepts of an EARLIER batch (--previous-from): the new pieces must differ from those too
        sets = {"seed": r["seed"], "length": st["length"], "previous": json.dumps({"concepts": prev, "effects": a["effects"]} if a.get("effects") else prev, ensure_ascii=False)}      # --effects rides in `previous` (concept.parse_effects): no new graph input
        aspect = a.get("aspect") or st.get("aspect")             # --aspect: the page's Format field `format`, set on every run; None = the graph's own (IN format)
        if aspect:
            sets["format"] = aspect
        r["aspect"] = aspect or graph_aspect(graph)
        if a.get("quality"):
            sets["quality"] = a["quality"]                      # e.g. still = the Still preview (16_app/STILL_PREVIEW.md); empty = the graph's own default
        gname = os.path.basename(str(graphs[route])).lower()          # user 2026-10-29: label every output with its design + video model
        tag = ("gemma" if "gemma" in gname else "rules") + "_" + route
        part = r.setdefault("part", f"r{i:02d}_{tag}" + (SP.SUFFIX if a.get("quality") == SP.QUALITY else ""))      # a redo gets its own part (r01b) so the old files stay apart; a still preview ends in _still
        proj = ["--idea", st["target"], "--part", part] if st.get("target") else []
        log(f"[{i}/{st['n']}] footage {route}, seed {r['seed']}, {len(prev)} earlier concepts ...")
        ceiling = run_ceiling(st, a)
        r["ceiling_s"] = round(ceiling)
        cancel_dump = arm_hang_dump(out, ceiling + 300, f"run {i} (ceiling {ceiling:.0f}s + 300s)")    # last resort: stacks of every thread if even the ceiling failed
        try:
            summ = run_graph(graphs[route], st["iid"], st["port"], sets, os.path.join(out, f"run_{i:02d}"), project=proj, timeout=ceiling,
                             graph_obj=graph)
            if summ.get("ended_by"):
                log(f"[{i}/{st['n']}] progress_standin did not finish by itself ({summ['ended_by']}, /history: {summ.get('history_state')}): "
                    f"{'finalized from /history' if summ.get('status') == 'done' else 'run recorded as error; the batch goes on'}")
            texts = texts_of(st["port"], summ.get("prompt_id", ""), graph) if summ.get("prompt_id") else {}
            r["timing"] = os.path.join(out, f"run_{i:02d}.json")
            record_run(r, summ, texts)
            if wants_batch_upscale(r, a.get("quality")):
                upscale_final(r, st, a, out, run_graph, log)
        except Exception as e:       # noqa: BLE001  one failed run must not stop the batch
            r.update(status="error", error={"type": type(e).__name__, "message": str(e)[:400]})
        finally:
            cancel_dump()
        log(f"[{i}/{st['n']}] {r['status']}: wall {r.get('wall_s')} s, mode {r.get('mode')}, shots {r.get('shots')}, "
            f"concept {(r.get('concept') or {}).get('title')!r}, peak VRAM {r.get('peak_vram_mib')}")
        write_reports(out, st)
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iid", required=True)
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--graph", default=GRAPH, help="the LTX-footage auto graph")
    ap.add_argument("--video-plan", help="footage route per run, comma separated (only ltx; default: all ltx)")
    ap.add_argument("--out", default=os.path.join(ROOT, "16_app", "bench", "batch_3060"))
    ap.add_argument("--idea", default="app", help="PROJECTS idea under COMPETITIONS/motion-graphic (default app)")
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--finalize", action="store_true", help="only collect the file paths + matte previews of the recorded runs, rewrite the reports")
    ap.add_argument("--timeout", type=float, default=None, help="hard ceiling per run in seconds (default: max(30 min, 3 x the mean wall of the runs done so far; 20 min assumed for the first)). "
                    "At the ceiling /history decides: succeeded -> finalized from it, else recorded as an error; either way the next run starts")
    ap.add_argument("--seed-base", type=int, default=0, help="seed = i + seed-base (new seeds for a new batch)")
    ap.add_argument("--effects", default="", help="comma separated named effects FORCED into every piece (tools/fusion/effects_menu.py: window_subject_front, plate_in_subject, plate_in_type, flat_ground_window, window_zoom, outline, any cut kind); unknown names are reported, empty = random")
    ap.add_argument("--previous-from", action="append", default=[], metavar="STATE.json", help="state.json of an earlier batch: its finished runs' concepts go into every run's `previous` (the new pieces differ from them)")
    ap.add_argument("--aspect", choices=ASPECTS, default=None, help="output aspect for every run = the page's Format select (9:16 vertical / 16:9 horizontal), set through the graph's `format` field (IN format; "
                    "MographPieceSizes derives every size from it). Default: the graph's own (9:16). Recorded per run in state.json / BATCH.md")
    ap.add_argument("--quality", choices=("still", "final", "preview", "draft"), default=None,
                    help="the graph's Quality field for every run (default: the graph's own); `still` = the Still preview: draft sizes, each shot's own still instead of LTX video, parts end in _still")
    ap.add_argument("--upgrade", metavar="VERSION/PART", help="instead of a batch: re-render the still preview <PROJECTS version>/<part> at draft with the SAME stills and the SAME design (no model call), "
                    "e.g. COMPETITIONS/motion-graphic/app/v012/r01_gemma_ltx_still; tools/app/upgrade_still.py")
    ap.add_argument("--dry-run", action="store_true", help="with --upgrade: build the upgrade graph, upload and submit nothing")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):                  # redirected to a file Python block-buffers: the log then lags minutes behind and hides what is happening
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:       # noqa: BLE001
            pass
    if a.upgrade:
        from tools.app import upgrade_still as US
        target, part = a.upgrade.rstrip("/").rsplit("/", 1)
        US.run(dict(target=target, part=part, iid=a.iid, port=a.port, graph=None if os.path.abspath(a.graph) == os.path.abspath(GRAPH) else os.path.abspath(a.graph), quality=a.quality or "final", out=None, timeout=a.timeout, dry_run=a.dry_run))
        return 0
    if a.finalize:
        st = finalize(load_state(os.path.join(os.path.abspath(a.out), "state.json")), os.path.abspath(a.out))
        print(report_md(st).split("## Concepts")[0])
        return 0
    st = run_batch(dict(iid=a.iid, port=a.port, n=a.n, length=a.length, graph=os.path.abspath(a.graph),
                        video_plan=parse_plan(a.video_plan, a.n), out=os.path.abspath(a.out), idea=a.idea,
                        project=not a.no_project, resume=a.resume, timeout=a.timeout, seed_base=a.seed_base, quality=a.quality, aspect=a.aspect, earlier=earlier_concepts(a.previous_from), effects=[x.strip() for x in a.effects.split(",") if x.strip()]))
    time.sleep(30)                 # the dashboard sync lands the last outputs
    cancel_dump = arm_hang_dump(os.path.abspath(a.out), sum(STEP_S.values()) + 600, "finalize")
    st = finalize(st, os.path.abspath(a.out))
    cancel_dump()
    print(report_md(st).split("## Concepts")[0])
    return 0 if all(r["status"] == "done" for r in st["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
