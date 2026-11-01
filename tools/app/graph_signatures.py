#!/usr/bin/env python
"""CPU-free STATIC validation of the nodes a graph uses against graphs a ComfyUI server has already RUN (no server, no model, no ComfyUI install).

The box's ComfyUI (0.43.x) has nodes this PC's checkout does not (TextEncodeMOSSImageEdit, the LTX IC-LoRA nodes), so `build_graphs.py --server` is the full check. This is the part that
needs no server: every node class that appears in a graph the box RAN (the MOSS angle-edit probes 23_moss-angles/v003, the clean-plate probes 24_clean-plate/v001, the Alpha-Gen
graph) has a PROVEN input signature (the input names, incl. dotted ones such as `images.image_0`, and the fixed widget values). A node of the same class in the new app graphs must
use exactly that input-name set (or, for the classes the app graph legitimately wires differently, a subset of it) and the proven widget values.

    python tools/app/graph_signatures.py app/graphs/<graph>.json [--prefix ang --prefix pl --prefix p1.]      # problems, exit 1 when any

    proven_signatures(paths) -> {class: {"keys": [sorted input names seen], "all_keys": [union], "widgets": {name: sorted values seen}}}
    check(graph, sigs, prefixes=None, exact=EXACT) -> [problem, ...]   (only nodes whose id starts with a prefix; all nodes when prefixes is None)
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_GLOBS = ("23_moss-angles/v003/graphs/*.json", "24_clean-plate/v001/graphs/*.json")
EXACT = {"TextEncodeMOSSImageEdit", "LoraLoaderModelOnly", "EmptyFlux2LatentImage", "VAEDecode"}      # same input-name set as the proven node
FIXED_WIDGETS = {"TextEncodeMOSSImageEdit": ("edit_format", "image_strength"), "LoraLoaderModelOnly": ("strength_model",),
                 "EmptyFlux2LatentImage": ("batch_size",)}


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def proven_signatures(paths=None) -> dict:
    sigs: dict = {}
    for pth in paths or [p for g in PROBE_GLOBS for p in sorted(glob.glob(os.path.join(ROOT, g)))]:
        try:
            graph = json.load(open(pth, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for node in graph.values():
            if not isinstance(node, dict) or "class_type" not in node:
                continue
            s = sigs.setdefault(node["class_type"], {"keys": None, "all_keys": set(), "widgets": {}})
            ks = set(node.get("inputs", {}))
            s["keys"] = ks if s["keys"] is None else s["keys"] & ks              # inputs present in EVERY proven use
            s["all_keys"] |= ks
            for k, v in node.get("inputs", {}).items():
                if not _is_link(v):
                    s["widgets"].setdefault(k, set()).add(json.dumps(v))
    return {c: {"keys": sorted(s["keys"] or []), "all_keys": sorted(s["all_keys"]), "widgets": {k: sorted(v) for k, v in s["widgets"].items()}} for c, s in sigs.items()}


def check(graph: dict, sigs: dict, prefixes=None, exact=EXACT) -> list[str]:
    bad = []
    for nid, n in graph.items():
        if prefixes is not None and not any(nid.startswith(p) for p in prefixes):
            continue
        cls = n.get("class_type")
        sig = sigs.get(cls)
        if sig is None:
            continue
        have = set(n.get("inputs", {}))
        unknown = have - set(sig["all_keys"])
        if unknown:
            bad.append(f"{nid} ({cls}): input(s) {sorted(unknown)} never appear in a graph the server ran (proven: {sig['all_keys']})")
        if cls in exact and set(sig["keys"]) - have:
            bad.append(f"{nid} ({cls}): proven input(s) {sorted(set(sig['keys']) - have)} missing")
        for w in FIXED_WIDGETS.get(cls, ()):
            v = n.get("inputs", {}).get(w)
            if w in n.get("inputs", {}) and not _is_link(v) and sig["widgets"].get(w) and json.dumps(v) not in sig["widgets"][w] and cls in exact:
                bad.append(f"{nid} ({cls}): {w}={v!r}, the proven runs used {sig['widgets'][w]}")
    return bad


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--prefix", action="append", help="only nodes whose id starts with this (repeatable); default all")
    a = ap.parse_args(argv)
    sigs = proven_signatures()
    print(f"proven signatures: {len(sigs)} node classes from {len([p for g in PROBE_GLOBS for p in glob.glob(os.path.join(ROOT, g))])} probe graphs")
    total = 0
    for path in a.graphs:
        problems = check(json.load(open(path, encoding="utf-8")), sigs, a.prefix)
        print(f"{os.path.basename(path)}: {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        total += len(problems)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
