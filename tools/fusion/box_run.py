#!/usr/bin/env python
"""box_run: run whole pieces ON the GPU box with no per-stage ssh round trips (16_app/BOX_OVERHEAD.md).

    python tools/fusion/box_run.py 15_fusion/pieces/P14.json [P15.json ... | @batch.txt] [--preview | --draft] [--out-version v004]
                                   [--iid ID --host H --ssh-port N --tunnel-port 9010] [--stage all] [--force stills] [--procs N]
                                   [--no-pull] [--detach]

What it does (this PC side only starts things, relays GPU submits and tails a log):
  1. one persistent ssh session; DELTA code sync (sha1 manifest, only changed files); the pieces' audio files + the local state.json
     (resume) go up in ONE tar stream.
  2. starts `piece_runner.py <pieces> --on-box` on the box under nohup (ComfyUI at localhost:18188, type / compose / flow / depth / assemble /
     final all local to the box). Several pieces run pipelined (piece_runner.run_batch: footage of N+1 renders while N composes).
  3. SUBMIT RELAY. progress_standin.py cannot run on the box: it writes the live Outputs-page tile to the dashboard backend on THIS PC
     (127.0.0.1:8000), reads PROJECTS / proj.py and the vast Instance_Data here, and samples box RAM over ssh. So the on-box runner writes
     each GPU graph to <root>/work/relay/<part>.req.json, this process (one poll call per <=20 s, or as soon as something happens) runs the
     SAME `progress_standin.py submit ... --port <tunnel>` command piece_runner always ran, and answers with <part>.resp.json. Everything else
     (file finds, copies, type, flow, compose, encode) is on the box.
  4. at the end ONE tar stream brings back the final mp4(s), sheet(s), state.json, timings.json, run.log and the BENCH md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import box_ssh  # noqa: E402
from tools.fusion import piece_runner as PR  # noqa: E402


def collect_inputs(piece_json: Path):
    """files a piece json names that are not in the synced code dirs (audio / analysis / grid) -> [ROOT-relative posix path]."""
    j = json.loads(piece_json.read_text(encoding="utf-8"))
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and len(o) < 260 and not o.startswith(("http", "/")):
            p = ROOT / o
            if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".json", ".png", ".jpg", ".mp4"):
                out.append(p.relative_to(ROOT).as_posix())
    walk(j)
    a = j.get("audio") or {}
    if a.get("src"):                                                       # the runner also looks for <ver>/local/audio/<stem>_analysis.json
        cand = PR.VER / "local" / "audio" / f"{Path(a['src']).stem}_analysis.json"
        if cand.is_file():
            out.append(cand.relative_to(ROOT).as_posix())
    return sorted(set(out))


def plate_files_of(piece_json: Path):
    """(pairs [(local file, box-root-relative path)], missing) of a piece's type.plates. FAILS LOUDLY (box_ssh.PlatesMissing) before anything
    is sent when the piece declares contour ports and a plate / matte / depth file is missing: the ports would silently fall back to module sides."""
    j = json.loads(piece_json.read_text(encoding="utf-8"))
    t = j.get("type") or {}
    pairs, missing = box_ssh.plate_pairs(t.get("plates"), ROOT)
    if missing and box_ssh.declares_ports(t):
        raise box_ssh.PlatesMissing(f"{j.get('id')}: contour ports declared but plate files missing: {', '.join(missing)}")
    return pairs, missing


class Relay:
    """PC side of the submit relay: runs progress_standin for every request the on-box runner writes."""

    def __init__(self, rx, relay_dir, log):
        self.rx, self.dir, self.log = rx, relay_dir, log
        self.threads = []

    def handle(self, name, req):
        t = threading.Thread(target=self._run, args=(name, req), daemon=True, name=f"relay-{name}")
        t.start()
        self.threads.append(t)

    def _run(self, name, req):
        piece, part, why = req["piece"], req["part"], req["why"]
        gdir = PR.PIECES_DIR / piece / "graphs"
        gdir.mkdir(parents=True, exist_ok=True)
        gpath, pidf = gdir / f"{part}.json", gdir / f"{part}.prompt_id"
        gpath.write_text(json.dumps(req["graph"], indent=1), encoding="utf-8")
        cmd = [sys.executable, str(PR.STANDIN), "submit", str(gpath), "--iid", PR.BOX["iid"], "--port", str(PR.BOX["tunnel_port"]),
               "--idea", PR.BOX["idea"], "--part", part, "--why", why, "--linger", "5", "--prompt-id-file", str(pidf)]
        self.log(f"[relay] submit {part}: {why}")
        ok, tail, keep = None, [], []
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-12:]
                if re.search(r"\b(DONE|FAILED|CANCELLED)\b", line) and re.match(r"\[\s*\d+s\]", line):
                    ok = " DONE " in line or line.split("]")[1].strip().startswith("DONE")
                    keep.append(line[:200])
                elif line.startswith("RESULT"):
                    keep.append(line[:200])
            p.wait()
        except Exception as e:      # noqa: BLE001
            tail = [f"relay could not run progress_standin: {type(e).__name__}: {e}"]
        pid = pidf.read_text().strip() if pidf.exists() else None
        resp = dict(ok=bool(ok), tail=tail, log=keep, prompt_id=pid)
        self.log(f"[relay] {part}: {'ok' if ok else 'FAILED'}")
        self.rx.put_many([(json.dumps(resp).encode("utf-8"), f"{self.dir}/{name}.resp.json")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pieces", nargs="+")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="append", default=[], choices=PR.STAGES)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--out-version", default="v004")
    ap.add_argument("--iid")
    ap.add_argument("--host")
    ap.add_argument("--ssh-port", type=int)
    ap.add_argument("--tunnel-port", type=int, help="local port of the ssh tunnel to the box's ComfyUI (9010 -> box 18188)")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--no-pull", action="store_true", help="leave the results on the box")
    ap.add_argument("--no-state", action="store_true", help="do not push the local state.json (fresh run; part names restart at _1)")
    a = ap.parse_args(argv)
    if a.draft and a.preview:
        ap.error("--draft and --preview are mutually exclusive")
    PR.set_version(a.out_version)
    PR.resolve_box(a.iid, a.host, a.ssh_port, a.tunnel_port, need_box=True)
    PR.ensure_version(a.out_version)
    t_all = time.time()
    stamp = {}

    def log(m):
        print(f"[box_run {PR.ts()}] {m}", flush=True)

    names = []
    for x in a.pieces:                                    # @batch.txt = one piece json per line (# comments allowed)
        if x.startswith("@"):
            names += [l.strip() for l in Path(x[1:]).read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]
        else:
            names.append(x)
    pieces = [Path(p).resolve() for p in names]
    ids = [json.loads(p.read_text(encoding="utf-8"))["id"] for p in pieces]
    for p in pieces:
        plate_files_of(p)                                 # fail before any ssh when a port piece lacks its plates
    rx = PR.Remote("batch", None, False, log, a.backend)
    root, work = rx.root, rx.work
    relay_dir, runlog, donef = f"{work}/relay", f"{work}/onbox.log", f"{work}/onbox.done"

    # ---- 1. sync + inputs
    t0 = time.time()
    rx.prepare()
    rx.wait_ready()
    stamp["prepare_s"] = round(time.time() - t0, 1)
    inputs = {}                                           # remote path -> local file; unchanged files (sha1 vs the box's manifest) are not re-sent
    for p in pieces:
        for rel in collect_inputs(p):
            inputs[f"{root}/{rel}"] = ROOT / rel
        pairs, _ = plate_files_of(p)                      # shot plates / depth / mattes the contour ports read in the TYPE stage
        for loc, rel in pairs:
            inputs[f"{root}/{rel}"] = loc
    push = []
    if not a.no_state:
        for pid in ids:
            sp = PR.PIECES_DIR / pid / "state.json"
            if sp.exists():
                push.append((sp, f"{root}/{sp.relative_to(ROOT).as_posix()}"))
    t0 = time.time()
    d = box_ssh.delta_put(rx, [(loc, rem) for rem, loc in inputs.items()])
    rx.put_many(push)
    stamp["inputs_s"] = round(time.time() - t0, 1)
    log(f"box ready in {stamp['prepare_s'] + stamp['inputs_s']:.1f}s ({len(inputs) + len(push)} input files: {d['sent']} sent "
        f"({d['bytes'] // 1024} KB), {d['skipped']} already on the box)")

    # ---- 2. launch on the box
    rel_pieces = " ".join(rx.q(p.relative_to(ROOT).as_posix()) for p in pieces)
    flags = ["--on-box", "--out-version", a.out_version, "--stage", a.stage, "--backend", a.backend, "--relay", relay_dir]
    flags += ["--draft"] if a.draft else (["--preview"] if a.preview else [])
    flags += [x for f in a.force for x in ("--force", f)]
    flags += ["--procs", str(a.procs)] if a.procs else []
    iid_env = f"MOGRAPH_BOX_IID={rx.q(str(PR.BOX['iid']))} "
    inner = (f"cd {root} && PYTHONUNBUFFERED=1 {iid_env}OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 "
             f"{rx.py} -m tools.fusion.piece_runner {rel_pieces} {' '.join(map(rx.q, flags))} > {runlog} 2>&1; echo $? > {donef}.tmp; mv {donef}.tmp {donef}")
    rx.sh(f"mkdir -p {relay_dir} && rm -f {donef} {relay_dir}/* {runlog} && (nohup bash -c {rx.q(inner)} > /dev/null 2>&1 < /dev/null &) ; echo started")
    t_run = time.time()
    log(f"running on the box: {', '.join(ids)} ({'preview' if a.preview else 'draft' if a.draft else 'final'} sizes); relay via tunnel :{PR.BOX['tunnel_port']}")

    # ---- 3. relay + tail
    relay = Relay(rx, relay_dir, log)
    offset, rc = 0, None
    poll = f"{rx.py} {root}/tools/fusion/box_poll.py --relay {relay_dir} --log {runlog} --done {donef}"
    while True:
        r = rx.sh(f"{poll} --offset {offset} --timeout 20", timeout=120, check=False)
        try:
            d = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            log(f"poll failed: {(r.stderr or r.stdout)[-200:]}")
            time.sleep(3)
            continue
        offset = d["offset"]
        for line in d["log"].splitlines():
            if line.strip():
                print("  | " + line[:220], flush=True)
        for q in d["reqs"]:
            relay.handle(q["name"], q["req"])
        if d["done"] is not None and not d["log"] and not d["reqs"]:
            rc = d["done"]
            break
    for t in relay.threads:
        t.join(timeout=600)
    stamp["run_s"] = round(time.time() - t_run, 1)
    log(f"on-box run finished rc={rc} in {stamp['run_s']}s")

    # ---- 4. results
    if not a.no_pull:
        t0 = time.time()
        rels = []
        for pid in ids:
            base = PR.PIECES_DIR.relative_to(ROOT).as_posix() + f"/{pid}"
            rels += [f"{base}/state.json", f"{base}/timings.json", f"{base}/run.log", f"{base}/{pid}_final.mp4",
                     PR.REVIEW.relative_to(ROOT).as_posix() + f"/{pid}_sheet.png", f"16_app/BENCH_{pid}.md"]
        for pid in ids:                                                     # keep the pre-run state next to the new one
            sp = PR.PIECES_DIR / pid / "state.json"
            if sp.exists():
                sp.with_suffix(".json.pre_onbox").write_bytes(sp.read_bytes())
        lst = " ".join(rx.q(x) for x in rels)
        got = rx._sess_call(f"cd {root} && for f in {lst}; do [ -e \"$f\" ] && echo \"$f\"; done | tar cf - -T -", 900)
        if got is not None and got[0] == 0:
            import io
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(got[1])) as t:
                names = t.getnames()
                for m in t.getmembers():
                    dst = ROOT / m.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if m.isfile():
                        dst.write_bytes(t.extractfile(m).read())
            for pid in ids:                                                 # its audio record names box paths; a PC-driven run must redo it
                sp = PR.PIECES_DIR / pid / "state.json"
                if sp.exists():
                    st = json.loads(sp.read_text(encoding="utf-8"))
                    if st.get("items", {}).pop("audio", None) is not None:
                        sp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
            log(f"pulled {len(names)} files in {time.time() - t0:.1f}s: " + ", ".join(Path(n).name for n in names))
        else:
            for x in rels:                                                  # plain-ssh fallback: one scp per file
                try:
                    rx.get(f"{root}/{x}", ROOT / x)
                except Exception as e:      # noqa: BLE001
                    log(f"  (no {x}: {str(e)[:80]})")
    stamp["total_s"] = round(time.time() - t_all, 1)
    log("timing " + json.dumps(stamp))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
