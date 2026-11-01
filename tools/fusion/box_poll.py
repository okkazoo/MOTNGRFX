#!/usr/bin/env python
"""box_poll: runs ON the box (stdlib only), called by tools/fusion/box_run.py through its persistent ssh session.

Blocks up to --timeout seconds until something happens, then prints ONE json line:
    {"reqs": [{"name": ..., "req": {...}}], "log": "<new text of the runner log>", "offset": <new log offset>, "done": <rc or null>}
something = a submit request in --relay (`*.req.json`, renamed to `.req.taken` so it is returned once), new log text, or the done file.
A request is a GPU graph the on-box piece_runner wants queued; the PC answers it by writing `<name>.resp.json` next to it."""
import argparse
import glob
import json
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=20.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    while True:
        reqs = sorted(glob.glob(os.path.join(a.relay, "*.req.json")))
        size = os.path.getsize(a.log) if os.path.exists(a.log) else 0
        fin = os.path.exists(a.done)
        if reqs or size > a.offset or fin or time.time() - t0 >= a.timeout:
            break
        time.sleep(0.25)
    out = dict(reqs=[], log="", offset=a.offset, done=None)
    for r in reqs:
        try:
            req = json.load(open(r, encoding="utf-8"))
        except ValueError:          # half-written (the writer renames a tmp file, so this should not happen)
            continue
        os.replace(r, r[:-len(".req.json")] + ".req.taken")
        out["reqs"].append(dict(name=os.path.basename(r)[:-len(".req.json")], req=req))
    if os.path.exists(a.log):
        with open(a.log, "rb") as f:
            f.seek(a.offset)
            b = f.read(400000)
        out["log"], out["offset"] = b.decode("utf-8", "replace"), a.offset + len(b)
    if os.path.exists(a.done):
        try:
            out["done"] = int(open(a.done).read().strip() or 0)
        except ValueError:
            out["done"] = 1
    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
