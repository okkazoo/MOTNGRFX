#!/usr/bin/env python
"""Timed run of an app graph on a box (or any ComfyUI) through progress_standin.py: per-node and per-stage seconds, peak VRAM, total wall.

    python tools/app/run_timing.py run app/graphs/fusion_design_qwen.json --iid 12345678 --port 8190 --out 16_app/bench/design_qwen_3060
    python tools/app/run_timing.py run <graph> --iid I --port P --set seed=3 --idea COMPETITIONS/motion-graphic/app --why "..." --part r01
    python tools/app/run_timing.py history <prompt_id> --port 8190 [--graph app/graphs/fusion_design_qwen.json]
    python tools/app/run_timing.py summarize 16_app/bench/design_qwen_3060.relay.log --graph app/graphs/fusion_design_qwen.json

Where the numbers come from (checked against progress_standin.py 2026-10-28):
  per-node start / end   the websocket `executing` events of the SUBMITTING client only (ComfyUI sends them to that client alone), which is
                         why this script submits through progress_standin.py and reads its `--relay-event` lines (STANDIN_RELAY {json}), time-
                         stamped on arrival (a few ms after the server). ComfyUI's /history/<id> does NOT carry per-node times: only
                         execution_start / execution_success timestamps (ms), the cached node list and the outputs (`history` below).
  peak VRAM (whole box)  progress_standin.py samples nvidia-smi / cgroup over ssh every 5 s and prints `RESULT OK 217s peakVRAM 9876 MiB ...`
                         (we do NOT pass --no-mem). Per node we add our own 1 s sampler of /system_stats (vram_total - vram_free; ComfyUI's
                         free figure counts torch's cached-but-reusable memory as free, so this reads LOW; the RESULT peak is the number to quote).
  wall                   `wall_s` = from launching progress_standin.py to the end of the last event; `comfy_s` = execution_start .. execution_success.
  lazy nodes             a node entered several times (the lazy repair, Package) adds its intervals (`seconds`); `start_s` is the first entry.
Writes <out>.json, <out>.md and <out>.relay.log (every line of progress_standin's output, `T<seconds> ` prefixed: `summarize` re-reads it).
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANDIN = os.environ.get("PROGRESS_STANDIN") or os.path.join(os.path.expanduser("~"), "Documents/Dev/comfyui-projects/tools/progress_standin.py")


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_ws_t", os.path.join(ROOT, "tools", "app", "bench_ws.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()
RELAY_PREFIX, RELAY_TYPES = bench.RELAY_PREFIX, bench.RELAY_TYPES

# node id (regex, first match wins) -> stage; ids are those of tools/app/build_graphs.py
STAGES = [
    (r"^(words|scene|music|look|quality|length|seed|image|audio|format|request)$", "Inputs"),
    (r"^(archive_in|source_in|concept_in)$", "Inputs"),
    (r"^load$", "Archive load (files)"),
    (r"^design_clip$", "Model file load (Gemma CLIPLoader)"),
    (r"^concept", "Concept (Qwen invents the request)"),
    (r"^(t2m_|cut$|route$|master$)", "Music (ACE XL base)"),
    (r"^(plan|sizes)$", "Plan (rules)"),
    (r"^(still_|first$|out_still)", "Still (MOSS)"),
    (r"^type$|^out_type$", "Type passes (LTX control)"),
    (r"^(l\d+\.|lay_prompt\d|lay_clean$)", "Clean shots (LTX image-to-video, tools/app/layered_shots.py)"),
    (r"^(t\d+\.|lay_gate\d|lay_sig$)", "Text shots (LTX union-control video-to-video)"),
    (r"^(slot\d|c\d+\.|\d+$)", "LTX shots"),
    (r"^(a\.|a\d+\.)", "Alpha-Gen mattes"),
    (r"^(vda$|d\d+\.)", "Depth (Video Depth Anything)"),
    (r"^gather$", "Footage gather"),
    (r"^flow$", "Footage flow (RAFT)"),
    (r"^design_pkg$", "Design package"),
    (r"^design_qwen2$", "Qwen repair (only when the first reply failed)"),
    (r"^design_qwen$", "Qwen design"),
    (r"^design_(parse|fix|piece)$|^out_design", "Design parse / repair / piece"),
    (r"^type_final$", "Type passes (designed piece)"),
    (r"^compose$", "Compose"),
    (r"^(assemble|out_final)$", "Assemble + encode + save"),
    (r"^archive$", "Archive (still, shot clips, matte PNGs)"),
]
RESULT_RE = re.compile(r"RESULT\s+(\w+)\s+(\d+)s\s+peakVRAM\s+(\S+)\s+MiB(?:.*?peakRAM\s+(\S+)\s+GiB anon)?")


def stage_of(node_id: str, title: str = "") -> str:
    for rx, name in STAGES:
        if re.search(rx, node_id):
            return name
    return "Other"


def summarize(run, graph: dict, extra: dict | None = None, vram: list | None = None) -> dict:
    """bench_ws.Run -> summary with per-stage rows and (when `vram` samples [(t, MiB)] are given) a peak per node / stage."""
    s = run.summary(extra or {})
    for n in s["nodes"]:
        n["stage"] = stage_of(n["id"], n["title"])
        if vram and run.t0 is not None:
            a, b = run.t0 + n["start_s"], run.t0 + n["end_s"]
            v = [m for t, m in vram if a <= t <= b]
            n["peak_vram_mib"] = max(v) if v else None
    stages: dict = {}
    for n in s["nodes"]:
        st = stages.setdefault(n["stage"], {"stage": n["stage"], "nodes": 0, "seconds": 0.0, "first_start_s": n["start_s"], "last_end_s": n["end_s"],
                                            "peak_vram_mib": None})
        st["nodes"] += 1
        st["seconds"] = round(st["seconds"] + n["seconds"], 3)
        st["first_start_s"] = min(st["first_start_s"], n["start_s"])
        st["last_end_s"] = max(st["last_end_s"], n["end_s"])
        if n.get("peak_vram_mib") is not None:
            st["peak_vram_mib"] = max(st["peak_vram_mib"] or 0, n["peak_vram_mib"])
    s["stages"] = sorted(stages.values(), key=lambda x: x["first_start_s"])
    return s


def markdown(name: str, s: dict) -> str:
    tot = s.get("total_seconds") or 0.0001
    L = [f"# Timed run {name}", ""]
    L.append(f"status **{s['status']}**; wall (submit to last event) **{s.get('wall_s')} s**; ComfyUI execution **{s['total_seconds']} s**; "
             f"{s['n_executed']} nodes executed, {s['n_cached']} cached.")
    L.append(f"peak VRAM (nvidia-smi via progress_standin): **{s.get('peak_vram_mib', 'n/a')} MiB**; peak RAM (anon): {s.get('peak_ram_gib', 'n/a')} GiB; "
             f"OOM: **{'yes: ' + str(s['oom']) if s.get('oom') else 'no'}**.")
    if s.get("error"):
        L.append(f"error at node {s['error']['node']}: {s['error']['type']}: {s['error']['message']}")
    if s.get("outputs"):
        L += ["", "outputs: " + "; ".join(f"{t}: {', '.join(f)}" for t, f in s["outputs"].items())]
    L += ["", "## Per stage (in order of first start)", "", "| stage | nodes | seconds | % of run | first start s | last end s | peak VRAM MiB (own sampler) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for st in s["stages"]:
        L.append(f"| {st['stage']} | {st['nodes']} | {st['seconds']:.1f} | {100 * st['seconds'] / tot:.1f} | {st['first_start_s']:.1f} | "
                 f"{st['last_end_s']:.1f} | {st['peak_vram_mib'] if st['peak_vram_mib'] is not None else 'n/a'} |")
    L += ["", "## Nodes in execution order", "", "| # | id | title | class | stage | start s | seconds | peak VRAM MiB |", "|---:|---|---|---|---|---:|---:|---:|"]
    for i, n in enumerate(s["nodes"], 1):
        L.append(f"| {i} | {n['id']} | {n['title']} | {n['class_type']} | {n['stage']} | {n['start_s']:.2f} | {n['seconds']:.2f} | "
                 f"{n.get('peak_vram_mib') if n.get('peak_vram_mib') is not None else 'n/a'} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- parsing progress_standin's output
def parse_lines(lines, graph: dict):
    """iterable of (t, line) -> (bench Run, extra dict). `line` is one line of progress_standin's stdout; relay lines drive the Run, the RESULT
    line gives the peaks, an `out of memory` / `OutOfMemory` text or an execution_error with that exception type sets `oom`."""
    run, extra, pid = bench.Run(graph), {}, ""
    for t, line in lines:
        line = line.rstrip("\n")
        if line.startswith(RELAY_PREFIX):
            try:
                msg = json.loads(line[len(RELAY_PREFIX):])
            except ValueError:
                continue
            pid = pid or (msg.get("data") or {}).get("prompt_id", "")
            run.feed(msg, t)
            continue
        m = RESULT_RE.search(line)
        if m:
            extra["peak_vram_mib"] = None if m.group(3) in ("n/a", "None") else int(float(m.group(3)))
            extra["peak_ram_gib"] = m.group(4)
            extra["standin_result"] = line.strip()
            if m.group(1) == "OK" and run.status == "running":      # progress_standin can exit on DONE before the last websocket event (execution_success) arrives
                run._close(t)
                run.t1, run.status = t, "done"
                extra["status_from_result"] = True
        if re.search(r"out of memory|OutOfMemoryError|CUDA error: out of memory", line, re.I):
            extra["oom"] = line.strip()[:300]
    err = run.error or {}
    if re.search(r"OutOfMemory|out of memory", str(err.get("type", "")) + str(err.get("message", "")), re.I):
        extra["oom"] = extra.get("oom") or f"{err.get('type')}: {err.get('message')}"[:300]
    extra["prompt_id"] = pid
    return run, extra


class VramSampler(threading.Thread):
    """1 s samples of /system_stats on the ComfyUI port -> [(perf_counter, used MiB)]."""

    def __init__(self, port: int, every: float = 1.0):
        super().__init__(daemon=True)
        self.url, self.every, self.samples, self._stop_ev = f"http://127.0.0.1:{port}/system_stats", every, [], threading.Event()

    def run(self):
        while not self._stop_ev.is_set():
            try:
                d = json.load(urllib.request.urlopen(self.url, timeout=5))["devices"][0]
                self.samples.append((time.perf_counter(), round((d["vram_total"] - d["vram_free"]) / 2 ** 20)))
            except Exception:       # noqa: BLE001
                pass
            self._stop_ev.wait(self.every)

    def stop(self):
        self._stop_ev.set()


IDLE_WATCHDOG_S = 120.0          # no line from progress_standin for this long = suspect: stacks dumped, /history asked (follow())
HISTORY_CHECK_EVERY_S = 30.0     # while suspect, /history is asked at most this often
PID_RE = re.compile(r"submitted ([0-9a-f]{8}-[0-9a-f-]{27})|\"prompt_id\":\"([0-9a-f]{8}-[0-9a-f-]{27})\"")


def history_state(port: int, prompt_id: str, timeout: float = 10.0) -> str | None:
    """/history/<id> -> 'success' | 'error' | 'interrupted' | 'running' (entry exists, not finished) | None (no entry / box unreachable). One short request, never raises."""
    try:
        h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=timeout)).get(prompt_id)
    except Exception:       # noqa: BLE001
        return None
    if not h:
        return None
    st = h.get("status") or {}
    msgs = [m[0] for m in st.get("messages", []) if isinstance(m, (list, tuple)) and m]
    if "execution_interrupted" in msgs:
        return "interrupted"
    if "execution_error" in msgs or st.get("status_str") == "error":
        return "error"
    if st.get("status_str") == "success" or "execution_success" in msgs or st.get("completed"):
        return "success"
    return "running"


def follow(p, emit, *, deadline: float, port: int, idle_s: float = IDLE_WATCHDOG_S, check_every: float = HISTORY_CHECK_EVERY_S,
           history_fn=history_state, now=time.time, dump=None) -> dict:
    """Read the child's stdout line by line through a reader thread + queue, so a child that goes silent (or a pipe that stops) cannot block this loop.
    emit(line) gets every line. -> {"ended": 'eof' | 'history' | 'deadline', "history": state | None, "prompt_id": id}; the caller kills the child unless 'eof'.
      silent for idle_s  -> dump(reason) once (all thread stacks into the log), then /history every check_every s; success / error / interrupted ends the wait
                            (the child hung after the prompt finished), 'running' keeps waiting.
      deadline           -> one last /history look, then the wait ends whatever it says."""
    q: queue.Queue = queue.Queue()

    def reader():
        try:
            for line in p.stdout:
                q.put(line)
        except Exception:       # noqa: BLE001  pipe closed under us (child killed)
            pass
        q.put(None)
    threading.Thread(target=reader, daemon=True, name="standin-reader").start()
    pid, last_line, last_check, dumped = "", now(), 0.0, False
    tick = max(0.05, min(5.0, idle_s / 4))
    while True:
        try:
            line = q.get(timeout=tick)
        except queue.Empty:
            line = ""
        t = now()
        if line is None:
            return {"ended": "eof", "history": None, "prompt_id": pid}
        if line:
            last_line, dumped = t, False
            emit(line)
            if not pid:
                m = PID_RE.search(line)
                pid = (m.group(1) or m.group(2)) if m else ""
            if t <= deadline:
                continue
        if t > deadline:
            return {"ended": "deadline", "history": history_fn(port, pid) if pid else None, "prompt_id": pid}
        if t - last_line > idle_s:
            if not dumped:
                dumped = True
                if dump:
                    dump(f"no output from progress_standin for {t - last_line:.0f}s (prompt {pid or 'unknown'})")
            if pid and t - last_check >= check_every:
                last_check = t
                state = history_fn(port, pid)
                if state in ("success", "error", "interrupted"):
                    return {"ended": "history", "history": state, "prompt_id": pid}


def _dump_stacks(reason: str, logfile=None) -> None:
    """all thread stacks of THIS process to stderr and the relay log: the evidence the 2026-10-29 incident lacked."""
    for f in (sys.stderr, logfile):
        try:
            if f is None:
                continue
            f.write(f"WATCHDOG {time.strftime('%H:%M:%S')} {reason}: thread stacks follow\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.flush()
        except Exception:       # noqa: BLE001  a stream without fileno() (tests) or a closed one
            pass


def run_graph(graph_path: str, iid: str, port: int, sets: dict | None = None, out: str | None = None, *, standin: str = STANDIN, project: list | None = None,
              mem: bool = True, timeout: float = 7200, sample_vram: bool = True, graph_obj: dict | None = None, idle_s: float = IDLE_WATCHDOG_S,
              history_fn=history_state, popen=subprocess.Popen) -> dict:
    """fill the graph (field ids of <graph>.app.json), submit it through progress_standin.py, follow it, write <out>.json / .md / .relay.log. -> summary.
    `timeout` = hard ceiling of this run in seconds. The wait ends when the child closes stdout (normal), when /history says the prompt finished while the
    child has been silent for idle_s (child hung after the render: killed, run finished from /history), or at the ceiling (/history looked at once more)."""
    gpath = os.path.abspath(graph_path)
    graph = graph_obj or json.load(open(gpath, encoding="utf-8"))
    side_path = gpath[:-5] + ".app.json"
    sidecar = json.load(open(side_path, encoding="utf-8")) if os.path.isfile(side_path) else {"fields": []}
    graph = bench.fill(graph, sidecar, sets or {}, {})
    out = out or os.path.join(ROOT, "16_app", "bench", f"{os.path.basename(gpath)[:-5]}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    filled = out + "_graph.json"
    json.dump(graph, open(filled, "w", encoding="utf-8"), indent=1)
    cmd = [sys.executable, standin, "submit", filled, "--iid", str(iid), "--port", str(port)]
    if not mem:
        cmd.append("--no-mem")
    if not project:
        cmd.append("--no-project")
    for t in RELAY_TYPES:
        cmd += ["--relay-event", t]
    cmd += list(project or [])
    sampler = VramSampler(port) if sample_vram else None
    t_launch = time.perf_counter()
    if sampler:
        sampler.start()
    lines = []
    p = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    forced = None
    with open(out + ".relay.log", "w", encoding="utf-8") as log:
        def emit(line):
            t = time.perf_counter()
            lines.append((t, line))
            log.write(f"T{t:.3f} {line}")
            log.flush()
            if not line.startswith(RELAY_PREFIX):
                print("  standin:", line.rstrip(), flush=True)
        forced = follow(p, emit, deadline=time.time() + timeout, port=port, idle_s=idle_s, history_fn=history_fn, dump=lambda why: _dump_stacks(why, log))
        if forced["ended"] != "eof":
            note = (f"WATCHDOG run ended by {forced['ended']}: /history says {forced['history']} for {forced['prompt_id'] or '(no prompt id seen)'}; "
                    "progress_standin killed")
            lines.append((time.perf_counter(), note + "\n"))
            log.write(f"T{time.perf_counter():.3f} {note}\n")
            print("  standin:", note, flush=True)
            p.kill()
    try:
        rc = p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = -9
    if sampler:
        sampler.stop()
    run, extra = parse_lines(lines, graph)
    if forced["ended"] != "eof":                    # the child never got to say DONE: /history is the verdict
        extra["ended_by"], extra["history_state"] = forced["ended"], forced["history"]
        extra["prompt_id"] = extra.get("prompt_id") or forced["prompt_id"]
        if run.status in ("running", "idle"):
            t_end = time.perf_counter()
            if forced["history"] == "success":
                run._close(t_end)
                run.t1, run.status = t_end, "done"
            else:
                run._close(t_end)
                run.t1, run.status = t_end, "error"
                run.error = run.error or {"node": None, "type": "WatchdogTimeout" if forced["ended"] == "deadline" else str(forced["history"]),
                                          "message": f"run ended by {forced['ended']} (/history: {forced['history']}) after progress_standin went silent"}
    base = f"http://127.0.0.1:{port}"
    outputs = bench.history_outputs(base, extra["prompt_id"], graph) if extra.get("prompt_id") else {}
    t_last = lines[-1][0] if lines else t_launch
    extra.update(graph=os.path.basename(gpath), set=sets or {}, port=port, iid=str(iid), outputs=outputs, standin_rc=rc,
                 wall_s=round(max(t_last, run.t1 or 0) - t_launch, 1), via="progress_standin")
    summ = summarize(run, graph, extra, sampler.samples if sampler else None)
    json.dump(summ, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".md", "w", encoding="utf-8").write(markdown(os.path.basename(out), summ))
    return summ


def from_log(log_path: str, graph_path: str) -> dict:
    """re-summarise a saved <out>.relay.log."""
    graph = json.load(open(graph_path, encoding="utf-8"))
    lines = []
    for raw in open(log_path, encoding="utf-8"):
        m = re.match(r"T([\d.]+) (.*)", raw)
        if m:
            lines.append((float(m.group(1)), m.group(2)))
    run, extra = parse_lines(lines, graph)
    return summarize(run, graph, extra)


def history(port: int, prompt_id: str, graph: dict | None = None) -> dict:
    """/history/<id>: total ComfyUI time from the status messages, cached nodes, outputs. No per-node times exist there."""
    h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=30)).get(prompt_id, {})
    msgs = {m[0]: m[1] for m in (h.get("status") or {}).get("messages", [])}
    t0 = (msgs.get("execution_start") or {}).get("timestamp")
    t1 = next((msgs[k].get("timestamp") for k in ("execution_success", "execution_error", "execution_interrupted") if k in msgs), None)
    return {"prompt_id": prompt_id, "status": (h.get("status") or {}).get("status_str"),
            "comfy_s": round((t1 - t0) / 1000.0, 1) if t0 and t1 else None, "cached_nodes": (msgs.get("execution_cached") or {}).get("nodes", []),
            "outputs": bench.history_outputs(f"http://127.0.0.1:{port}", prompt_id, graph or {}),
            "note": "ComfyUI history has no per-node times; use `run` (websocket events through progress_standin.py)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("graph")
    r.add_argument("--iid", required=True)
    r.add_argument("--port", type=int, default=8190)
    r.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    r.add_argument("--out")
    r.add_argument("--standin", default=STANDIN)
    r.add_argument("--no-mem", action="store_true", help="skip the ssh peak VRAM/RAM sampling of progress_standin")
    r.add_argument("--idea", help="PROJECTS target P/I[/vNNN]: outputs land in that version (progress_standin --idea)")
    r.add_argument("--why")
    r.add_argument("--part")
    r.add_argument("--timeout", type=float, default=7200)
    h = sub.add_parser("history")
    h.add_argument("prompt_id")
    h.add_argument("--port", type=int, default=8190)
    h.add_argument("--graph")
    s = sub.add_parser("summarize")
    s.add_argument("log")
    s.add_argument("--graph", required=True)
    a = ap.parse_args()
    if a.cmd == "history":
        g = json.load(open(a.graph, encoding="utf-8")) if a.graph else None
        print(json.dumps(history(a.port, a.prompt_id, g), indent=1))
        return 0
    if a.cmd == "summarize":
        s_ = from_log(a.log, a.graph)
        print(markdown(os.path.basename(a.log), s_))
        return 0
    proj = []
    if a.idea:
        proj = ["--idea", a.idea] + (["--why", a.why] if a.why else [])
    if a.part:
        proj += ["--part", a.part]
    summ = run_graph(a.graph, a.iid, a.port, dict(x.split("=", 1) for x in a.set), a.out, standin=a.standin, project=proj, mem=not a.no_mem, timeout=a.timeout)
    print(f"{summ['status']}: wall {summ.get('wall_s')} s, comfy {summ['total_seconds']} s, peak VRAM {summ.get('peak_vram_mib', 'n/a')} MiB, "
          f"oom {bool(summ.get('oom'))}")
    for st in summ["stages"]:
        print(f"  {st['seconds']:8.1f} s  {st['stage']}")
    return 0 if summ["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
