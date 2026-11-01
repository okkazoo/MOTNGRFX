"""MographLayoutPreview: the RANDOMISE LAYOUT PREVIEW node (user 2026-10-30: "an option on the preview to create a random seed so we can see a preview of the text layout on the single frames and hit a
randomise button to quickly create a new one"). Doc: 16_app/STILL_PREVIEW.md, code: tools/fusion/layout_preview.py.

    archive (a MographArchive folder) or piece + stills, seed, effects  ->  sheet IMAGE, system STRING, piece, piece_json STRING

The per-piece random system is DRAWN AGAIN with `seed` on the same words / timings / shots (fonts by length, layouts incl. grid / column, the still palette, the outline choice, the hold-life choice, the cut
kinds), ONE frame per word at its landed hold is rendered (the type pass for those frames only, CPU subprocesses, then composed over that word's own shot still through legible_color) and the tiles are
returned as one contact sheet (word, layout and font under each tile, the seed in the header). No Gemma call, no LTX, no GPU. The sheet and the drawn piece are also written next to the archive as
`<stem>_layout_s<seed>.png` / `.json` (never overwritten: `_b`, `_c` ...). The seed has ComfyUI's randomize control; the app's Randomise button queues the graph with a new seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

try:
    import folder_paths
except Exception:   # pragma: no cover
    folder_paths = None

from . import ensure_root_on_path
from ._types import CATEGORY, MographPiece

ensure_root_on_path()
from tools.fusion import layout_preview as LPV   # noqa: E402


def _resolve(p: str) -> Path:
    q = Path(str(p).strip())
    if q.is_absolute():
        return q
    base = Path(folder_paths.get_output_directory() if folder_paths is not None else ".")
    return base / q


class MographLayoutPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLayoutPreview",
            display_name="MOTNGRFX Layout Preview (randomise)",
            category=CATEGORY,
            description="Draws the piece's random system again with a seed (fonts, layouts, palette, outline, hold life, cut kinds) on the same words and timings and shows ONE frame per word at its landed "
                        "hold over its still: a contact sheet. No Gemma, no LTX, CPU only (about 5-10 s).",
            inputs=[
                io.String.Input("archive", default="", tooltip="A MographArchive folder (under the ComfyUI output dir, or absolute) with <stem>_piece.json and the <stem>_still*.png of a Preview run. "
                                "Empty = use `piece` + `stills`."),
                io.Int.Input("seed", default=1, min=1, max=0xFFFFFFFF, control_after_generate=True, tooltip="The layout seed: the same seed gives the same draw."),
                io.String.Input("effects", default="", optional=True, tooltip="Named effects forced into the draw, comma separated (grid, column, outline, window_zoom, pulse ...)."),
                MographPiece.Input("piece", optional=True, tooltip="A planned / designed piece (the design step output) when no archive folder is given."),
                io.Image.Input("stills", optional=True, tooltip="The piece's stills, row k = place k (MographFirstFrames order); used with `piece`."),
                io.Boolean.Input("save", default=True, optional=True, tooltip="Write <stem>_layout_s<seed>.png / .json next to the archive (a new name every time)."),
            ],
            outputs=[io.Image.Output(display_name="sheet"), io.String.Output(display_name="system"), MographPiece.Output(display_name="piece"), io.String.Output(display_name="piece_json")],
        )

    @classmethod
    def fingerprint_inputs(cls, archive="", seed=1, effects="", save=True, **kw):
        p = _resolve(archive) if str(archive or "").strip() else None
        mt = 0.0
        if p is not None:
            for n in sorted(p.glob("*_piece.json")) + [p / "piece.json"]:
                if n.is_file():
                    mt = max(mt, n.stat().st_mtime)
        return f"{archive}|{seed}|{effects}|{save}|{mt}"

    @classmethod
    def execute(cls, archive: str = "", seed: int = 1, effects: str = "", piece=None, stills=None, save: bool = True) -> io.NodeOutput:
        out_dir, stem, stl = None, "piece", {}
        if str(archive or "").strip():
            out_dir = _resolve(archive)
            arc = LPV.load_archive(out_dir)
            piece, stl, stem = arc["piece"], arc["stills"], arc["stem"]
        elif piece is None:
            raise ValueError("MographLayoutPreview: give an `archive` folder or a `piece`")
        else:
            stem = str(piece.get("id") or piece.get("slug") or "piece")
        if stills is not None:
            stl = LPV.stills_from_batch(stills)
        if not stl:
            raise ValueError("MographLayoutPreview: no stills (the archive has no <stem>_still*.png and no `stills` input)")
        r = LPV.preview(piece, stl, int(seed), effects, out_dir=out_dir if (save and out_dir is not None) else None, stem=stem)
        t = r["timing"]
        print(f"[MographLayoutPreview] seed {int(seed)}: {len(r['tiles'])} tiles in {t['total_s']:.1f}s (redraw {t['redraw_s']}s, type frames {t['render']}, compose {t['compose_s']}s)"
              + (f" -> {r['files'].get('png')}" if r["files"] else ""))
        sheet = torch.from_numpy(np.ascontiguousarray(r["sheet"]).astype(np.float32) / 255.0)[None]
        system = json.dumps(dict(r["summary"], seed=int(seed), timing=t), indent=1, default=str)
        return io.NodeOutput(sheet, system, r["piece"], json.dumps(r["piece"], indent=1, default=str))


NODES = [MographLayoutPreview]
