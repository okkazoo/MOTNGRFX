#!/usr/bin/env python
"""API graph of the RANDOMISE LAYOUT PREVIEW (node `MographLayoutPreview`, tools/fusion/layout_preview.py): archive folder of a Preview run -> MographLayoutPreview -> SaveImage (the contact sheet)
+ PreviewAny (the drawn system). Tiny: no model, no GPU; the app page queues it with a new seed for every press of Randomise.

    python tools/app/layout_preview_graph.py --write-template            # app/graphs/_layout_preview.json: the template the page fills in (archive, seed, effects, save prefix)
    python tools/app/layout_preview_graph.py --archive <dir> --seed 7 --out g.json [--effects grid,outline] [--via-load]
    python <progress_standin.py> submit g.json --iid <id> --port <port>     # every ComfyUI submission goes through progress_standin

`--via-load` uses MographArchiveLoad as the source (the redesign loader: it decodes every shot clip and matte PNG of the archive, so it is slower than the node's own `archive` input, which reads only
piece.json and the stills). Nodes: `layout` = MographLayoutPreview (outputs: 0 sheet, 1 system, 2 piece, 3 piece_json), `save` = SaveImage "OUT layout sheet", `out_system` = PreviewAny "OUT layout system".
The page (web_app/app.js, Layout panel) sets `layout.inputs.archive` / `seed` / `effects` and `save.inputs.filename_prefix`, then POSTs the graph to /prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "app", "graphs", "_layout_preview.json")        # a leading `_` keeps it out of the template dropdown (app_routes.list_graphs)
SAVE_PREFIX = "layout_preview/layout"


def build(archive: str, seed: int = 1, effects: str = "", stem: str | None = None, via_load: bool = False) -> dict:
    """-> API-format graph (dict). `archive` = a MographArchive folder (under the ComfyUI output dir, or absolute)."""
    prefix = f"layout_preview/{stem}_layout" if stem else SAVE_PREFIX
    if via_load:
        g = {
            "load": {"class_type": "MographArchiveLoad", "inputs": {"archive": archive, "source_video": "", "concept": ""}, "_meta": {"title": "IN archive"}},
            "layout": {"class_type": "MographLayoutPreview",
                       "inputs": {"archive": "", "seed": int(seed), "effects": effects, "piece": ["load", 2], "stills": ["load", 4], "save": False},
                       "_meta": {"title": "Layout preview"}},
        }
    else:
        g = {"layout": {"class_type": "MographLayoutPreview", "inputs": {"archive": archive, "seed": int(seed), "effects": effects, "save": True}, "_meta": {"title": "Layout preview"}}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["layout", 0], "filename_prefix": prefix}, "_meta": {"title": "OUT layout sheet"}}
    g["out_system"] = {"class_type": "PreviewAny", "inputs": {"source": ["layout", 1]}, "_meta": {"title": "OUT layout system"}}
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--effects", default="")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--via-load", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-template", action="store_true", help="write app/graphs/_layout_preview.json (archive empty: the page fills it in)")
    a = ap.parse_args(argv)
    g = build(a.archive, a.seed, a.effects, a.stem, a.via_load)
    out = TEMPLATE if a.write_template else a.out
    if not out:
        ap.error("give --out or --write-template")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
