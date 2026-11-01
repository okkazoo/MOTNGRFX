#!/usr/bin/env python
"""Submit one filled app graph to a ComfyUI server and record per-node ``executing`` timestamps from /ws.

Writes <out>.json (nodes in execution order with start / end / seconds / cached, totals, prompt ids) and <out>.md
(table + time per node class + the slowest nodes) - the raw material of 16_app/BENCH.md.

    python tools/app/bench_ws.py app/graphs/fusion.json --port 8190 --out 16_app/bench/draft_p02 \\
        --set words="AFTER HOURS\\nTHE CITY KEEPS DREAMING" --set length=12 --set seed=5 --set quality=draft \\
        --audio 15_fusion/v002/out/motion-graphic_fusion_v002_P02_s201_00001.flac

--set FIELD=VALUE uses the field ids and bindings of <graph>.app.json (same as the web page: node title + input name).
--image / --audio upload a file through the server's /upload/image and bind it to the image / audio field.

Two ways to watch the run:
  default        own websocket (client id) + POST /prompt - works on any server, including this PC.
  --standin PATH submit through scripts/progress_standin.py (the user rule for boxes: only the submitting
                 client gets progress events, and that script feeds the live tile on the dashboard Outputs page);
                 we read its ``--relay-event`` lines (STANDIN_RELAY {json}) and timestamp them on arrival
                 (a few ms later than the direct websocket). Needs --iid of a real box: on this PC it queues the
                 prompt, then dies with "no media dir under Instance_Data for instance <iid>" (it only knows boxes
                 from the dashboard's Instance_Data tree), so local benches use the default direct mode.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RELAY_PREFIX = "STANDIN_RELAY "
RELAY_TYPES = ("execution_start", "execution_cached", "executing", "executed", "execution_success", "execution_error",
               "execution_interrupted")


# --------------------------------------------------------------------------------------------- graph filling
def title_map(g: dict) -> dict:
    return {n["_meta"]["title"]: nid for nid, n in g.items() if n.get("_meta", {}).get("title")}


def coerce(field: dict, value):
    if field["kind"] in ("seed",):
        return int(value)
    if field["kind"] in ("number", "select"):
        for o in field.get("options", []):
            ov = o["value"] if isinstance(o, dict) else o
            if str(ov) == str(value):
                return ov
    return str(value).replace("\\n", "\n")


def fid_missing(by_id: dict, uploads: dict) -> bool:
    return any(k not in by_id for k in uploads)


def fill(g: dict, sidecar: dict, sets: dict, uploads: dict) -> dict:
    g = json.loads(json.dumps(g))
    tm = title_map(g)
    by_id = {f["id"]: f for f in sidecar.get("fields", [])}
    media = by_id.get("media")                       # the "Add media" box (16_app/MEDIA_BOX.md) replaced the image / audio fields: --image / --audio fill its IN image / IN audio nodes
    if media and fid_missing(by_id, uploads):
        for fid, key in (("image", "image_node"), ("audio", "audio_node")):
            if fid in uploads and media.get(key):
                by_id[fid] = {"id": fid, "kind": fid, "node_title": media[key], "input_name": fid}
    for fid, val in {**sets, **uploads}.items():
        f = by_id.get(fid)
        if f is None:
            raise SystemExit(f"unknown field {fid!r}; fields: {', '.join(by_id)}")
        if f["node_title"] not in tm:
            raise SystemExit(f"field {fid!r}: node {f['node_title']!r} is not in the graph")
        g[tm[f["node_title"]]]["inputs"][f["input_name"]] = val if fid in uploads else coerce(f, val)
    return g


def upload(base: str, path: str) -> str:
    name = os.path.basename(path)
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: {mimetypes.guess_type(name)[0] or 'application/octet-stream'}\r\n\r\n".encode(),
        open(path, "rb").read(), f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
        f"--{boundary}--\r\n".encode()])
    req = urllib.request.Request(base + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return (r["subfolder"] + "/" if r.get("subfolder") else "") + r["name"]


# --------------------------------------------------------------------------------------------- tracking
class Run:
    """Per-node timeline from websocket messages. Times are seconds on any monotonic clock."""

    def __init__(self, graph: dict):
        self.graph = graph
        self.t0 = self.t1 = None
        self.status = "idle"
        self.error = None
        self.order: list[str] = []
        self.start: dict[str, float] = {}
        self.end: dict[str, float] = {}
        self.busy: dict[str, float] = {}        # summed executing intervals (lazy nodes are entered several times)
        self.cached: set[str] = set()
        self.current = None
        self._open = 0.0

    def _close(self, t):
        if self.current is not None:
            self.end[self.current] = t
            self.busy[self.current] = self.busy.get(self.current, 0.0) + (t - self._open)
        self.current = None

    def feed(self, msg: dict, t: float) -> bool:
        """-> True when the run is over."""
        ty, d = msg.get("type"), msg.get("data") or {}
        if ty == "execution_start":
            self.t0, self.status = t, "running"
        elif ty == "execution_cached":
            self.cached.update(str(n) for n in d.get("nodes", []))
        elif ty == "executing":
            nid = d.get("node")
            self._close(t)
            if nid is None:
                self.t1, self.status = t, self.status if self.status == "error" else "done"
                return True
            nid = str(nid)
            if nid not in self.start:
                self.start[nid] = t
                self.order.append(nid)
            self.current, self._open = nid, t
        elif ty == "execution_success":
            self._close(t)
            self.t1, self.status = t, "done"
            return True
        elif ty in ("execution_error", "execution_interrupted"):
            self._close(t)
            self.t1, self.status = t, "error" if ty == "execution_error" else "interrupted"
            self.error = {"node": str(d.get("node_id")), "type": d.get("exception_type"),
                          "message": d.get("exception_message")}
            return True
        return False

    def summary(self, extra: dict) -> dict:
        nodes = []
        for nid in self.order:
            n = self.graph.get(nid, {})
            s, e = self.start[nid], self.end.get(nid, self.t1 or self.start[nid])
            nodes.append({"id": nid, "title": (n.get("_meta") or {}).get("title", nid), "class_type": n.get("class_type"),
                          "start_s": round(s - (self.t0 or s), 3), "end_s": round(e - (self.t0 or s), 3),
                          "seconds": round(self.busy.get(nid, e - s), 3)})
        total = round((self.t1 - self.t0), 3) if self.t0 is not None and self.t1 is not None else None
        return {"status": self.status, "error": self.error, "total_seconds": total, "n_executed": len(nodes),
                "n_cached": len(self.cached), "nodes": nodes, **extra}


def run_direct(base: str, port: int, graph: dict, timeout: float) -> tuple[Run, str]:
    import websocket
    cid = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"ws://127.0.0.1:{port}/ws?clientId={cid}", timeout=20)
    req = urllib.request.Request(base + "/prompt", json.dumps({"prompt": graph, "client_id": cid}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"POST /prompt -> {e.code}: {e.read().decode()[:3000]}")
    run = Run(graph)
    ws.settimeout(timeout)
    while True:
        raw = ws.recv()
        t = time.perf_counter()
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        d = msg.get("data") or {}
        if d.get("prompt_id") not in (None, pid):
            continue
        if run.feed(msg, t):
            break
    ws.close()
    return run, pid


def run_standin(script: str, iid: str, port: int, graph_path: str, graph: dict, timeout: float, extra: list[str]) -> tuple[Run, str]:
    cmd = [sys.executable, script, "submit", graph_path, "--iid", str(iid), "--port", str(port), "--no-mem", "--no-project"]
    for t in RELAY_TYPES:
        cmd += ["--relay-event", t]
    cmd += extra
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    run, pid = Run(graph), ""
    t_end = time.time() + timeout
    for line in p.stdout:
        t = time.perf_counter()
        if line.startswith(RELAY_PREFIX):
            msg = json.loads(line[len(RELAY_PREFIX):])
            pid = pid or (msg.get("data") or {}).get("prompt_id", "")
            run.feed(msg, t)
        else:
            print("  standin:", line.rstrip())
        if run.status in ("done", "error", "interrupted") or time.time() > t_end:
            break
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    return run, pid


# --------------------------------------------------------------------------------------------- report
def history_outputs(base: str, pid: str, graph: dict, budget: float = 150.0) -> dict:
    h = {}
    t_end = time.time() + budget                      # total, not per request: a stalled tunnel used to cost 40 x 30 s
    for _ in range(40):                               # the history entry is written a moment after the last event
        try:
            h = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=min(30, max(1.0, t_end - time.time())))).get(pid, {})
        except Exception:
            h = {}
        if h.get("outputs") or (h.get("status") or {}).get("status_str") == "error" or time.time() > t_end:
            break
        time.sleep(0.5)
    out = {}
    for nid, o in (h.get("outputs") or {}).items():
        files = [f for k in ("images", "gifs", "video", "videos", "audio") for f in o.get(k, [])]
        if files:
            out[(graph.get(nid, {}).get("_meta") or {}).get("title", nid)] = [
                (f.get("subfolder", "") + "/" if f.get("subfolder") else "") + f["filename"] for f in files]
    return out


def markdown(name: str, s: dict) -> str:
    total = s["total_seconds"] or 0.0001
    L = [f"# Bench {name}", "",
         f"status **{s['status']}**, wall **{s['total_seconds']} s**, {s['n_executed']} nodes executed, {s['n_cached']} cached.", ""]
    if s.get("error"):
        L += [f"error at node {s['error']['node']}: {s['error']['type']}: {s['error']['message']}", ""]
    if s.get("outputs"):
        L += ["outputs: " + "; ".join(f"{t}: {', '.join(f)}" for t, f in s["outputs"].items()), ""]
    by_cls: dict[str, list[float]] = {}
    for n in s["nodes"]:
        by_cls.setdefault(n["class_type"] or "?", []).append(n["seconds"])
    L += ["## Time per node class", "", "| class | nodes | seconds | % |", "|---|---:|---:|---:|"]
    for c, v in sorted(by_cls.items(), key=lambda kv: -sum(kv[1])):
        L.append(f"| {c} | {len(v)} | {sum(v):.2f} | {100 * sum(v) / total:.1f} |")
    L += ["", "## Nodes in execution order", "", "| # | id | title | class | start s | seconds | % |", "|---:|---|---|---|---:|---:|---:|"]
    for i, n in enumerate(s["nodes"], 1):
        L.append(f"| {i} | {n['id']} | {n['title']} | {n['class_type']} | {n['start_s']:.2f} | {n['seconds']:.2f} | "
                 f"{100 * n['seconds'] / total:.1f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--out", help="output prefix (default 16_app/bench/<graph>_<time>)")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--standin", metavar="PATH", help="submit through progress_standin.py (needs --iid)")
    ap.add_argument("--iid")
    a = ap.parse_args()

    gpath = os.path.abspath(a.graph)
    graph = json.load(open(gpath, encoding="utf-8"))
    side_path = gpath[:-5] + ".app.json"
    sidecar = json.load(open(side_path, encoding="utf-8")) if os.path.isfile(side_path) else {"fields": []}
    base = f"http://127.0.0.1:{a.port}"
    sets = dict(s.split("=", 1) for s in a.set)
    uploads = {}
    if a.image:
        uploads["image"] = upload(base, a.image)
    if a.audio:
        uploads["audio"] = upload(base, a.audio)
    graph = fill(graph, sidecar, sets, uploads)

    out = a.out or os.path.join(ROOT, "16_app", "bench", f"{os.path.basename(gpath)[:-5]}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if a.standin:
        if not a.iid:
            raise SystemExit("--standin needs --iid")
        filled = out + "_graph.json"
        json.dump(graph, open(filled, "w", encoding="utf-8"), indent=1)
        run, pid = run_standin(a.standin, a.iid, a.port, filled, graph, a.timeout, [])
    else:
        run, pid = run_direct(base, a.port, graph, a.timeout)
    summ = run.summary({"graph": os.path.basename(gpath), "prompt_id": pid, "port": a.port, "set": sets,
                        "via": "progress_standin" if a.standin else "direct websocket",
                        "outputs": history_outputs(base, pid, graph)})
    json.dump(summ, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".md", "w", encoding="utf-8").write(markdown(os.path.basename(out), summ))
    print(f"{summ['status']} in {summ['total_seconds']} s, {summ['n_executed']} nodes -> {out}.json / .md")
    for n in sorted(summ["nodes"], key=lambda n: -n["seconds"])[:8]:
        print(f"  {n['seconds']:8.2f} s  {n['title']} ({n['class_type']})")
    return 0 if summ["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
