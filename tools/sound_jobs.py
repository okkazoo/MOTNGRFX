#!/usr/bin/env python3
"""Queue ACE-Step 1.5 jobs for the CPU synth drafts (from tools/sound_drafts.py) via progress_standin.

  python tools/sound_jobs.py --local 03_sound/v001/local --idea COMPETITIONS/motion-graphic/sound/v001
  python tools/sound_jobs.py ... --only B3 --kinds a2a --denoise 0.3 0.7 --dry
  python tools/sound_jobs.py ... --times      # print generation time per part from the box /history

Per draft <id>: <id>_draft (LoadAudio -> SaveAudio passthrough), <id>_a2a_dNNN (a2a at each
--denoise) and <id>_t2m_sN (pure text-to-music, one per --seeds). Tags/bpm/key/duration come from
<id>_draft.json. Patched graphs + prompt ids go to <local>/jobs/. At most --parallel submits
run at once (one GPU shared with other workers).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows" / "api"))
from acestep_patch import patch  # noqa: E402

STANDIN = os.environ.get("PROGRESS_STANDIN") or str(Path.home() / "Documents/Dev/comfyui-projects/tools/progress_standin.py")
GRAPHS = REPO / "workflows" / "api"


def build_jobs(local: Path, only, kinds, denoises, seeds):
    jobs = []
    for meta_p in sorted(local.glob("*_draft.json")):
        m = json.loads(meta_p.read_text())
        did = m["id"]
        if only and did not in only:
            continue
        audio = f"mograph/{m['wav']}"
        base = dict(tags=m["tags"], bpm=m["bpm"], key=m["keyscale"], seconds=m["duration"])
        if "draft" in kinds:
            jobs.append((f"{did}_draft", patch(json.loads((GRAPHS / "audio_passthrough.json").read_text()), audio=audio)))
        if "a2a" in kinds:
            g0 = json.loads((GRAPHS / "acestep_a2a.json").read_text())
            for d in denoises:
                jobs.append((f"{did}_a2a_d{round(d * 100):03d}", patch(g0, audio=audio, denoise=d, seed=1, **base)))
        if "t2m" in kinds:
            g0 = json.loads((GRAPHS / "acestep_t2m.json").read_text())
            for s in seeds:
                jobs.append((f"{did}_t2m_s{s}", patch(g0, seed=s, **base)))
    return jobs


def submit(job, jobs_dir: Path, idea: str, port: int, iid: str):
    part, g = job
    gp = jobs_dir / f"{part}.json"
    gp.write_text(json.dumps(g, indent=1))
    cmd = [sys.executable, STANDIN, "submit", str(gp), "--iid", iid, "--port", str(port), "--idea", idea,
           "--part", part, "--prompt-id-file", str(jobs_dir / f"{part}.pid")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if "RESULT" in ln or "DONE " in ln or "rror" in ln]
    print(f"{part:<20} rc={r.returncode}  " + " | ".join(tail[-3:]), flush=True)
    (jobs_dir / f"{part}.log").write_text(r.stdout + r.stderr)
    return part, r.returncode


def times(jobs_dir: Path, port: int) -> dict:
    """Execution time per part (s) from ComfyUI /history status messages."""
    out = {}
    for pid_p in sorted(jobs_dir.glob("*.pid")):
        pid = pid_p.read_text().strip()
        try:
            h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=20).read())
        except Exception as e:  # noqa: BLE001
            out[pid_p.stem] = f"n/a ({e})"
            continue
        msgs = {m[0]: m[1] for m in h.get(pid, {}).get("status", {}).get("messages", [])}
        t0 = msgs.get("execution_start", {}).get("timestamp")
        t1 = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
        out[pid_p.stem] = round((t1 - t0) / 1000.0, 2) if t0 and t1 else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default=str(REPO / "03_sound" / "v001" / "local"))
    ap.add_argument("--idea", default="COMPETITIONS/motion-graphic/sound/v001")
    ap.add_argument("--iid", default="57178352")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--kinds", nargs="*", default=["draft", "a2a", "t2m"])
    ap.add_argument("--denoise", nargs="*", type=float, default=[0.3, 0.5, 0.7, 0.85])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip (already done)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--times", action="store_true")
    a = ap.parse_args(argv)
    jobs_dir = Path(a.local) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    if a.times:
        t = times(jobs_dir, a.port)
        (jobs_dir / "times.json").write_text(json.dumps(t, indent=1))
        for k, v in t.items():
            print(f"{k:<20} {v}")
        return
    jobs = [j for j in build_jobs(Path(a.local), a.only, a.kinds, a.denoise, a.seeds) if j[0] not in a.skip]
    print(f"{len(jobs)} jobs: " + ", ".join(p for p, _ in jobs))
    if a.dry:
        return
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(lambda j: submit(j, jobs_dir, a.idea, a.port, a.iid), jobs))
    bad = [p for p, rc in res if rc]
    print(f"done: {len(res) - len(bad)} ok, failed: {bad}")


if __name__ == "__main__":
    main()
