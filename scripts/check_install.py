#!/usr/bin/env python3
"""Check a running ComfyUI against what the MOTNGRFX app graphs need, and say how to fix what is missing.

    python scripts/check_install.py --url http://127.0.0.1:8188
    python scripts/check_install.py --url http://127.0.0.1:8188 --graph fusion --graph local_v1

For every graph in app/graphs (default: all) it reads ``/object_info`` and checks
  * every node class is registered (else: which custom node pack provides it, or "update ComfyUI"),
  * every model filename a loader node references is offered by that loader (else: scripts/download_models.py),
  * the /mograph page is served by the pack.
Nodes a sidecar lists in ``optional_nodes`` (the page drops them when their class is missing) are reported as warnings.
Exit code 0 = everything required is present, 1 = something required is missing, 2 = server not reachable. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (GRAPHS_DIR, MODELS, graph_classes, graph_files, graph_model_refs, human_size,  # noqa: E402
                     load_graph, load_packs, pack_for_class)


def fetch(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "motngrfx-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def combo_options(info: dict, class_type: str, input_name: str):
    """List of allowed values of a combo input, or None when the input is not a combo (or unknown)."""
    spec = info.get(class_type, {}).get("input", {})
    for section in ("required", "optional"):
        v = spec.get(section, {}).get(input_name)
        if v:
            first = v[0]
            if isinstance(first, list):
                return first
            if first == "COMBO" and len(v) > 1 and isinstance(v[1], dict) and "options" in v[1]:
                return v[1]["options"]
    return None


def optional_node_ids(graph: dict, sidecar: dict) -> set[str]:
    """Node ids a sidecar marks optional (``optional_nodes``). Entries: {class_type, titles: [...]} as written by
    tools/app/optional_nodes.py, or a node title / node id / class type string."""
    out: set[str] = set()
    for e in sidecar.get("optional_nodes", []) or []:
        cls, keys = None, []
        if isinstance(e, dict):
            cls = e.get("class_type")
            keys = list(e.get("titles") or []) + [e.get("title"), e.get("node_title"), e.get("id")]
        else:
            keys = [e]
        for nid, n in graph.items():
            title = (n.get("_meta") or {}).get("title")
            if (cls and n.get("class_type") == cls) or any(k is not None and (k in (nid, title) or (not cls and k == n.get("class_type"))) for k in keys):
                out.add(nid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI base URL (default %(default)s)")
    ap.add_argument("--graph", action="append", help="graph name or file (default: every graph in app/graphs); repeatable")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    try:
        stats = json.loads(fetch(base + "/system_stats"))
        info = json.loads(fetch(base + "/object_info", timeout=120))
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"Cannot reach ComfyUI at {base}: {e}\nStart it first, e.g. `python main.py --listen 127.0.0.1 --port 8188`.")
        return 2

    sysinfo = stats.get("system", {})
    print(f"ComfyUI {sysinfo.get('comfyui_version', '?')}  python {str(sysinfo.get('python_version', '?')).split()[0]}  "
          f"RAM {human_size(sysinfo.get('ram_total', 0))}")
    for d in stats.get("devices", []):
        print(f"  device: {d.get('name')}  VRAM {human_size(d.get('vram_total', 0))}")
    if not stats.get("devices"):
        print("  no GPU device reported")
    print(f"  {len(info)} node classes registered\n")

    problems = 0
    try:
        page = fetch(base + "/mograph")
        print(f"[ok] /mograph page served ({len(page)} bytes)" if b"MOTNGRFX" in page or b"Mograph" in page
              else "[!!] /mograph answered but is not the app page")
    except urllib.error.URLError as e:
        problems += 1
        print(f"[!!] /mograph not served ({e}): the pack is not loaded. Clone it into ComfyUI/custom_nodes, "
              "install requirements.txt, restart ComfyUI and read the console for the import error.")
    print()

    packs = load_packs()
    graphs = graph_files(args.graph or ["*.json"])
    if not graphs:
        print(f"No graph matches {args.graph} in app/graphs.")
        return 2
    all_missing_models: dict[str, set[str]] = {}
    for g in graphs:
        graph = load_graph(g)
        side_path = GRAPHS_DIR / f"{g.stem}.app.json"
        side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.is_file() else {}
        opt_ids = optional_node_ids(graph, side)
        print(f"== {g.stem}  ({side.get('title', '')})")
        classes = graph_classes(graph)
        miss_req, miss_opt = [], []
        for cls, titles in sorted(classes.items()):
            if cls in info:
                continue
            ids = [nid for nid, n in graph.items() if n.get("class_type") == cls]
            pack = pack_for_class(cls, packs)
            optional = (all(i in opt_ids for i in ids) or bool(pack and not pack.get("required", True)))
            (miss_opt if optional else miss_req).append((cls, titles, pack))
        for label, rows, tag in (("required", miss_req, "!!"), ("optional (the page drops these nodes)", miss_opt, "--")):
            for cls, titles, pack in rows:
                fix = (f"install {pack['name']}: {pack['install']}" if pack
                       else "update ComfyUI to the latest release (core node) or install the custom node pack that provides it")
                print(f"  [{tag}] missing {label} node {cls}  (used by: {', '.join(titles[:3])}{'...' if len(titles) > 3 else ''})\n"
                      f"        fix: {fix}")
        problems += len(miss_req)
        if not miss_req and not miss_opt:
            print(f"  [ok] all {len(classes)} node classes are registered")

        for cls, inp, fname, title in graph_model_refs(graph):
            if cls not in info:
                continue                       # already reported as a missing node
            opts = combo_options(info, cls, inp)
            if opts is None:
                continue
            if fname in opts:
                continue
            m = MODELS.get(fname)
            nid_opt = all(i in opt_ids for i, n in graph.items() if n.get("class_type") == cls)
            where = f"models/{m['folder']}/" if m else "ComfyUI/models/"
            fix = (f"python scripts/download_models.py (puts it in {where}; {human_size(m['size'])}"
                   f"{', gated Hugging Face repo: needs HF_TOKEN' if m.get('gated') else ''}) then restart ComfyUI or press R"
                   if m else f"place the file in {where} and restart ComfyUI")
            print(f"  [{'--' if nid_opt else '!!'}] model not offered by {cls}.{inp}: {fname}  (node '{title}')\n        fix: {fix}")
            all_missing_models.setdefault(g.stem, set()).add(fname)
            if not nid_opt:
                problems += 1
        if g.stem not in all_missing_models:
            print("  [ok] every model file is offered by its loader")
        print()

    if problems:
        print(f"{problems} problem(s) found.")
        return 1
    print("Install looks complete for the checked graphs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
