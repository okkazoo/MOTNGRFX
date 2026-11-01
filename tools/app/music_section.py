#!/usr/bin/env python
"""Cut the music section out of the app graph (everything upstream of the Master node) and add a saver, to test the approved
music route on the box without the LTX / still / type chains:

    python tools/app/music_section.py out.json [--length 12] [--music "slow ambient piano, 72 bpm"] [--seed 1] [--prefix projects/X/music_test]
    python path/to/progress_standin.py submit out.json --iid <id> --port 9010

The graph is the real ``fusion.json`` music path (ACE XL base text-to-music -> T2M cut -> tame highs -> route -> mono-safe master), so
this tests the same nodes the app runs. Qwen is left out (the rules write the brief)."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def music_section(length: int = 12, music: str = "", seed: int = 1, prefix: str = "audio/motngrfx_music_test", words: str | None = None) -> dict:
    graph = json.load(open(os.path.join(ROOT, "app", "graphs", "fusion.json"), encoding="utf-8"))
    keep, stack = set(), ["master"]
    while stack:
        n = stack.pop()
        if n in keep:
            continue
        keep.add(n)
        stack += [v[0] for v in graph[n]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in graph]
    g = {k: graph[k] for k in graph if k in keep}
    for n in g.values():                                              # no Qwen in this slice
        n["inputs"].pop("brief", None)
    g["words"]["inputs"]["value"] = words or g["words"]["inputs"]["value"]
    g["music"]["inputs"]["value"] = music
    g["length"]["inputs"]["value"] = int(length)
    g["seed"]["inputs"]["value"] = int(seed)
    g["out_music"] = {"class_type": "SaveAudioAdvanced", "_meta": {"title": "OUT music"},
                      "inputs": {"audio": ["master", 0], "filename_prefix": prefix, "format": "flac"}}
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--music", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--words")
    ap.add_argument("--prefix", default="audio/motngrfx_music_test")
    a = ap.parse_args()
    graph = music_section(a.length, a.music, a.seed, a.prefix, a.words)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(graph, f, indent=1)
    print(f"{a.out}: {len(graph)} nodes ({sorted({v['class_type'] for v in graph.values()})})")
