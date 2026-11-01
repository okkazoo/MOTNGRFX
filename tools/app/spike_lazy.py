#!/usr/bin/env python
"""Lazy-gating spike driver: submit the 3-slot stand-in graph for N = 1, 2, 3 and print which nodes executed.

Start ComfyUI with the spike nodes first (own port, never 8188):
    MOGRAPH_SPIKE=1 MOGRAPH_SPIKE_LOG=/tmp/spike.log python main.py --cpu --port 8191
    python tools/app/spike_lazy.py --port 8191 --log /tmp/spike.log
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def graph(n: int, work_s: float = 0.2) -> dict:
    g = {"1": {"class_type": "MographSpikeCount", "inputs": {"n": n}}}
    gather = {"piece": ["1", 0]}
    for k in (1, 2, 3):
        slot = {"index": k, "piece": ["1", 0]}
        if k > 1:
            slot["prev"] = [f"w{k - 1}", 0]
        g[f"s{k}"] = {"class_type": "MographSpikeSlot", "inputs": slot}
        g[f"w{k}"] = {"class_type": "MographSpikeWork", "inputs": {"prompt": [f"s{k}", 0], "index": k, "seconds": work_s}}
        gather[f"clip_{k}"] = [f"w{k}", 0]
    g["g"] = {"class_type": "MographSpikeGather", "inputs": gather}
    return g


def post(port: int, g: dict) -> str:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", json.dumps({"prompt": g}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"POST /prompt -> {e.code}: {e.read().decode()[:1500]}")


def wait(port: int, pid: str, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=10))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            return h[pid]
        time.sleep(0.3)
    raise TimeoutError(pid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    bad = 0
    for n in (1, 2, 3):
        open(a.log, "w").close()
        h = wait(a.port, post(a.port, graph(n)))
        lines = [ln.split(" ", 1)[1].strip() for ln in open(a.log, encoding="utf-8") if ln.strip()]
        ran = [ln for ln in lines if ln.startswith(("slot", "work", "gather")) and not ln.endswith("enter")]
        want = [x for k in range(1, n + 1) for x in (f"slot s{k} prev=" + ("none" if k == 1 else f"s{k - 1}"),
                                                     f"work s{k} exit")]
        want.append("gather got " + ",".join(f"s{k}" for k in range(1, n + 1)))
        ok = ran == want and h["status"]["status_str"] == "success"
        bad += not ok
        print(f"N={n}: {'PASS' if ok else 'FAIL'}  status={h['status']['status_str']}")
        print("   executed:", ran)
        if not ok:
            print("   expected:", want)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
