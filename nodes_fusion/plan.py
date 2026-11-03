"""Planner nodes of the fusion pipeline (16_app/DESIGN.md section 1): Request, FusionBriefPrompt, PiecePlan, PieceFromJSON.

Thin wrappers: the logic is in ``mograph/fusion_brief.py`` (Qwen prompt + tolerant parser) and ``tools/fusion/plan_rules.py``
(rules -> piece spec). Rules write every field; the brief text only supplies creative text, so an empty or broken Qwen answer
gives a valid piece. The planned piece is a plain dict (MOGRAPH_PIECE) and its JSON is an output too: paste it into
MographPieceFromJSON to replay a run exactly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from comfy_api.latest import io

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted: ComfyUI's own top-level ``nodes`` must not be shadowed by nodes.py
    sys.path.append(_ROOT)

from tools.fusion import plan_rules  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402
from tools.fusion import user_layout as UL  # noqa: E402
from mograph import fusion_brief  # noqa: E402

from ._types import CATEGORY  # noqa: E402
from ._types import MographMusic as MusicT, MographPiece as PieceT, MographRequest as RequestT  # noqa: E402


class MographRequest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographRequest",
            display_name="MOTNGRFX Request",
            category=CATEGORY,
            description="The app inputs as one validated request: words, scene, music prompt, length (8/12/16/24 s), look, "
                        "seed, quality, format (9:16 or 16:9), optional user image and audio.",
            inputs=[
                io.String.Input("words", multiline=True, default="", tooltip="The words to show; ' / ' or a new line starts a new line."),
                io.String.Input("scene", multiline=True, default="", tooltip="Scene prompt; empty = written from the words."),
                io.String.Input("music", multiline=True, default="", tooltip="Music prompt (genre, instruments, 'F minor', '120 bpm'); empty = by mood."),
                io.Int.Input("length", default=12, min=8, max=24, step=4, tooltip="Seconds: 8, 12, 16 or 24 (snapped)."),
                io.String.Input("look", default="auto", tooltip="auto, a family (strand, mech, slab, stack, ring, voxel, particles) or a module name."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.String.Input("quality", default="final", tooltip="preview | draft | final | still (Still preview: draft sizes, each shot's own still held instead of LTX video) | draft540 (the app's Preview: the full draft-size render, no 1080 upscale)"),
                io.Image.Input("image", optional=True, tooltip="Your own first frame (skips the still model)."),
                io.Audio.Input("audio", optional=True, tooltip="Your own music (skips the music model)."),
                io.String.Input("format", default="9:16", optional=True, tooltip="9:16 (vertical) or 16:9 (horizontal)."),
                io.String.Input("shots", default="", optional=True, tooltip="cut (a still per shot, hard cuts on downbeats) | chain (one still, each shot "
                                "continues from the previous last frame) | single (one long shot); empty = the brief decides, else cut."),
                io.String.Input("owners", default="", multiline=True, optional=True, tooltip="JSON list, one entry per line of `words`: {render: flat | ltx | handoff, "
                                "ltx_at: 0.2..0.8, material} - who draws each lockup (the concept step chooses; empty = the rules' owners). Also accepted: the object "
                                "{owners: [..], shot_plan: [{prompt, camera}, ..], idea} - the concept step's own shot prompts + camera move per shot and its one-line visual idea "
                                "(the planner's templates only fill a shot without one)."),
                io.Int.Input("layout_seed", default=0, min=0, max=0xFFFFFFFF, optional=True, tooltip="0 = the planner's own draw. A number re-draws the LAYOUT side of the piece system (fonts, cut kinds, outline, "
                                "layouts, hold life, palette pick) with that seed, the stills and footage side stay: the app's 'Use this layout' (Randomise layout preview) sets it."),
            ],
            outputs=[RequestT.Output(display_name="request")],
        )

    @classmethod
    def execute(cls, words, scene, music, length, look, seed, quality, image=None, audio=None, format="9:16", shots="", owners="", layout_seed=0) -> io.NodeOutput:
        q, still = SP.split_quality(quality)             # `still` = a draft-size run whose footage is the shots' own stills (tools/fusion/still_preview.py)
        direction = US.direction_of(owners)              # the user's typed "Words or direction" text, carried in the concept's request extras (concept.request_extras)
        req = plan_rules.normalize_request({"words": words, "scene": scene, "music": music, "length": length, "look": look,
                                            "seed": seed, "quality": q, "format": format, "image": image, "audio": audio,
                                            "shots": shots, "owners": owners, "layout_seed": layout_seed})
        if still:
            req[SP.FLAG] = True
        if direction:
            req["direction"] = direction
        shows = US.shows_in(owners)                      # what each user image shows (subject_guard: the shot prompts keep that subject)
        if shows:
            req["shows"] = shows
        req["raw"] = SP.request_record({"words": words, "scene": scene, "music": music, "length": length, "look": look, "seed": seed, "format": format,
                                        "shots": shots, "owners": owners, "layout_seed": layout_seed or None, "direction": direction or None,
                                        SP.NO_UPSCALE: True if SP.is_preview_full(quality) else None, SP.PLATES_ONLY: True if SP.is_plates_only(quality) else None})      # the app inputs as given: the piece keeps them (piece.request) so a later upgrade re-runs the same piece
        return io.NodeOutput(req)


class MographFusionBriefPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographFusionBriefPrompt",
            display_name="MOTNGRFX Fusion Brief Prompt",
            category=CATEGORY,
            description="Instruction for design_llm (wire into DesignLLM custom_prompt): it answers with a small JSON brief "
                        "(scene, still prompt, motion line per shot, music tags, technique, mood, palette, signoff).",
            inputs=[RequestT.Input("request")],
            outputs=[io.String.Output(display_name="prompt"), io.Int.Output(display_name="seed")],
        )

    @classmethod
    def execute(cls, request) -> io.NodeOutput:
        req = plan_rules.normalize_request(request)
        return io.NodeOutput(fusion_brief.brief_prompt(req), plan_rules.sub_seed(req["seed"], "brief"))


class MographPiecePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPiecePlan",
            display_name="MOTNGRFX Piece Plan",
            category=CATEGORY,
            description="Rule-based planner: request + music analysis (+ Qwen brief) -> piece spec (words, timing on the beat "
                        "grid, modules, fonts, shots, compose recipe). Dry-runs the type scene at 576x1024 and shrinks the "
                        "windows if the timeline would be longer than the music.",
            inputs=[
                RequestT.Input("request"),
                MusicT.Input("music"),
                io.String.Input("brief", multiline=True, default="", optional=True,
                                tooltip="Raw Qwen output (fences and slips are repaired). Empty or unreadable = rules only."),
            ],
            outputs=[PieceT.Output(display_name="piece"), io.String.Output(display_name="piece_json"),
                     io.String.Output(display_name="still_prompt"), io.String.Output(display_name="still_negative")],
        )

    @classmethod
    def execute(cls, request, music, brief="") -> io.NodeOutput:
        piece = plan_rules.plan_piece(request, music, brief or None)
        SP.prepare_piece(piece, request.get("raw"), bool(request.get(SP.FLAG)))      # still preview flag + the app inputs on the piece
        if request.get("shows") and isinstance(piece.get("request"), dict):
            piece["request"]["shows"] = list(request["shows"])      # piece.request survives the design step (LOCKED_KEYS): MographShotSlot reads it for the prompt scrub (layered_shots.scrub_prompt)
        US.stamp(piece, request.get("direction"))         # IMAGE MODE: the user's typed text lands on the piece here (piece.direction + piece.user_style; piece.request.direction rides the design step)
        UL.apply(piece)                                    # every piece: "more legible text" re-fits the layout bigger / upright; else at most ONE rare mid-word split (user_layout; a design graph re-does this on the designed piece)
        still = piece["stills"][0]
        return io.NodeOutput(piece, plan_rules.piece_json(piece), still["prompt"], still.get("negative", ""))


class MographPieceSizes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPieceSizes",
            display_name="MOTNGRFX Piece Sizes",
            category=CATEGORY,
            description="The sizes of the piece's format (9:16 or 16:9) as numbers for the graph's size inputs: LTX stage 1 "
                        "(and the alpha pass), the control reference (stage 1 / 2), LTX stage 2 (x2 upscale) and the still.",
            inputs=[PieceT.Input("piece")],
            outputs=[io.Int.Output(display_name="stage1_w"), io.Int.Output(display_name="stage1_h"),
                     io.Int.Output(display_name="ref_w"), io.Int.Output(display_name="ref_h"),
                     io.Int.Output(display_name="stage2_w"), io.Int.Output(display_name="stage2_h"),
                     io.Int.Output(display_name="still_w"), io.Int.Output(display_name="still_h")],
        )

    @classmethod
    def execute(cls, piece) -> io.NodeOutput:
        sz = piece.get("sizes") or plan_rules.sizes_for(piece.get("quality", "final"), plan_rules.format_of(piece))
        c, st = sz["control"], sz["type"]
        return io.NodeOutput(int(c[0]), int(c[1]), int(c[0]) // 2, int(c[1]) // 2, int(c[0]) * 2, int(c[1]) * 2,
                             int(st[0]), int(st[1]))


def _music_for(piece: dict) -> dict | None:
    """analysis of a hand-written piece: P*.json 'audio.analysis' if it exists, else the piece_runner output dir."""
    root = Path(_ROOT)
    cands = []
    a = (piece.get("audio") or {}).get("analysis")
    if a:
        cands.append(Path(a) if os.path.isabs(a) else root / a)
    cands.append(root / "15_fusion" / "v002" / "local" / "pieces" / str(piece.get("id", "")) / "analysis.json")
    for c in cands:
        if c.is_file():
            try:
                d = json.loads(c.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if "downbeats" in d and "beats" in d and "cut" in d:
                return d
    return None


class MographPieceFromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPieceFromJSON",
            display_name="MOTNGRFX Piece From JSON",
            category=CATEGORY,
            description="Load a hand-written 15_fusion/pieces/P*.json (benchmark / regression) or the piece_json of an earlier "
                        "plan: adds layout, sizes and technique, fills the beat grid from the music when given.",
            inputs=[
                io.String.Input("piece_json", multiline=True, default=""),
                MusicT.Input("music", optional=True, tooltip="Without it the piece's analysis.json is looked up."),
            ],
            outputs=[PieceT.Output(display_name="piece")],
        )

    @classmethod
    def execute(cls, piece_json, music=None) -> io.NodeOutput:
        try:
            d = json.loads(piece_json or "")
        except ValueError as e:
            raise ValueError(f"piece_json is not valid JSON: {e}") from None
        if not isinstance(d, dict):
            raise ValueError("piece_json must be a JSON object (a P*.json piece)")
        piece = plan_rules.finalize_piece(d, music or _music_for(d))
        return io.NodeOutput(piece)


NODES = [MographRequest, MographFusionBriefPrompt, MographPiecePlan, MographPieceFromJSON, MographPieceSizes]
