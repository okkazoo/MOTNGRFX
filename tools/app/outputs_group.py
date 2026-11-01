"""Add a finished file to a dashboard Outputs group (the same calls as tools/app/run_batch.py finalize): PC side, best effort.

    python tools/app/outputs_group.py PATH_UNDER_PROJECTS [--group motngrfx-app]

The box cannot reach the PC's dashboard (127.0.0.1:8000 is this PC): the box writes `output/motngrfx_app/own/<job>/final.mp4`; once the file is pulled into the PROJECTS tree
(a version's `out/`), run this from the PC. `add()` returns True when the dashboard took it."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request


def add(path: str, group: str = "motngrfx-app", api: str | None = None, projects_root: str | None = None) -> bool:
    api = api or os.environ.get("MOGRAPH_DASHBOARD", "http://127.0.0.1:8000")
    root = projects_root or os.environ.get("PROJECTS_ROOT") or os.path.join(os.path.expanduser("~"), "Documents", "Dev", "PROJECTS")
    rel = os.path.relpath(os.path.abspath(path), root).replace("\\", "/")
    try:
        gs = json.load(urllib.request.urlopen(f"{api}/api/outputs/groups", timeout=5))
        gid = next((g["id"] for g in (gs.get("groups", gs) if isinstance(gs, dict) else gs) if g.get("name") == group), None)
        url, body = (f"{api}/api/outputs/groups/{gid}/members", {"paths": [f"@projects/{rel}"]}) if gid else (f"{api}/api/outputs/groups", {"name": group, "paths": [f"@projects/{rel}"]})
        urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"), timeout=5)
        return True
    except Exception as e:   # noqa: BLE001 - a missing dashboard never fails anything
        print(f"output group: skipped ({type(e).__name__}: {e})")
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--group", default="motngrfx-app")
    a = ap.parse_args()
    raise SystemExit(0 if add(a.path, a.group) else 1)
