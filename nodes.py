"""ComfyUI nodes (v3 schema API). Thin wrappers around the pure-Python ``mograph`` package.

torch <-> numpy conversion happens here and nowhere else.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import numpy as np
import torch
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, InputImpl

if __package__:
    from .mograph import analyze, geometry, planner_rules, soundtrack
    from .mograph import brief as mbrief
    from .mograph import plan as mplan
    from .mograph.encode import encode_mp4
    from .mograph.render import render_frames
else:   # loaded as a top-level module (tests / scripts)
    from mograph import analyze, geometry, planner_rules, soundtrack
    from mograph import brief as mbrief
    from mograph import plan as mplan
    from mograph.encode import encode_mp4
    from mograph.render import render_frames

CATEGORY = "MOTNGRFX"
MographAnalysis = io.Custom("MOGRAPH_ANALYSIS")
MographPlan = io.Custom("MOGRAPH_PLAN")


# --------------------------------------------------------------------------
# boundary conversions
# --------------------------------------------------------------------------

def _images_to_numpy(t) -> list[np.ndarray]:
    """IMAGE tensor [B, H, W, C] 0..1 -> list of HxWx3 float32 arrays (a batch is several images)."""
    if t is None:
        return []
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    if a.ndim == 3:
        a = a[None]
    out = []
    for im in a:
        if im.shape[-1] == 1:
            im = np.repeat(im, 3, axis=-1)
        out.append(np.ascontiguousarray(im[..., :3], dtype=np.float32))
    return out


def _audio_to_numpy(audio) -> tuple[np.ndarray, int] | None:
    """AUDIO {"waveform": [B, C, N], "sample_rate"} -> ((C, N) float32, sr). First batch item only."""
    if audio is None or audio.get("waveform") is None:
        return None
    w = audio["waveform"]
    w = w.detach().cpu().float().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, np.float32)
    if w.ndim == 3:
        w = w[0]
    elif w.ndim == 1:
        w = w[None]
    return np.ascontiguousarray(w, dtype=np.float32), int(audio["sample_rate"])


def _numpy_to_audio(wave: np.ndarray, sr: int) -> dict:
    w = np.asarray(wave, np.float32)
    if w.ndim == 1:
        w = w[None]
    return {"waveform": torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0), "sample_rate": int(sr)}


def _check(plan: dict, analysis: dict) -> None:
    errors = mplan.validate_plan(plan, analysis)
    if errors:
        raise ValueError("Mograph plan is invalid:\n- " + "\n- ".join(errors))


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

class MographAnalyze(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographAnalyze",
            display_name="Mograph Analyze",
            category=CATEGORY,
            description="Analyse text, images and audio (palette, empty regions, contours, lines, beats, mood).",
            inputs=[
                io.String.Input("text", multiline=True, default="",
                                tooltip="One line per beat of the story; long lines are split into short phrases."),
                io.Image.Input("image_1", optional=True, tooltip="A batch counts as several images."),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
                io.Image.Input("image_4", optional=True),
                io.Audio.Input("audio", optional=True, tooltip="Optional music; the cut follows its beats."),
                io.Float.Input("duration", default=0.0, min=0.0, max=15.0, step=0.5,
                               tooltip="Seconds; 0 = auto (audio length or 10 s, clamped to 5-15)."),
            ],
            outputs=[MographAnalysis.Output(display_name="analysis")],
        )

    @classmethod
    def execute(cls, text: str, duration: float, image_1=None, image_2=None, image_3=None, image_4=None,
                audio=None) -> io.NodeOutput:
        images = []
        for t in (image_1, image_2, image_3, image_4):
            images.extend(_images_to_numpy(t))
        analysis = analyze.analyze_inputs(text or "", images, _audio_to_numpy(audio),
                                          float(duration) if duration and duration > 0 else None)
        return io.NodeOutput(analysis)


class MographPlanRules(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPlanRules",
            display_name="Mograph Plan (Rules)",
            category=CATEGORY,
            description="Deterministic rule-based director: analysis -> plan (scenes, layers, transitions, audio cues).",
            inputs=[
                MographAnalysis.Input("analysis"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True),
            ],
            outputs=[MographPlan.Output(display_name="plan"), io.String.Output(display_name="plan_json")],
        )

    @classmethod
    def execute(cls, analysis: dict, seed: int) -> io.NodeOutput:
        plan = planner_rules.plan_from_analysis(analysis, seed=int(seed))
        return io.NodeOutput(plan, mplan.to_json(plan))


class MographBriefPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographBriefPrompt",
            display_name="Mograph Brief Prompt",
            category=CATEGORY,
            description="Instruction for a VLM (wire into DesignLLM custom_prompt, with the same images): it "
                        "answers with a small creative brief JSON for Mograph Plan (Brief).",
            inputs=[
                MographAnalysis.Input("analysis"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True,
                             tooltip="Use the same seed as Mograph Plan (Brief): with empty input it picks the "
                                     "slot-machine theme and family named in the prompt."),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, analysis: dict, seed: int) -> io.NodeOutput:
        n = len(analysis.get("images") or [])
        return io.NodeOutput(mbrief.brief_prompt(str(analysis.get("text") or ""), n, int(seed)))


class MographPlanBrief(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPlanBrief",
            display_name="Mograph Plan (Brief)",
            category=CATEGORY,
            description="Free hybrid director: a VLM's creative brief (style family, palette, energy, tempo, arc, "
                        "line split, hero image) -> rule-built plan. Unreadable or empty brief = rules only; empty "
                        "input = seeded slot machine.",
            inputs=[
                MographAnalysis.Input("analysis"),
                io.String.Input("brief", multiline=True, default="",
                                tooltip="Brief JSON (raw VLM output is fine: fences and slips are repaired)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.Float.Input("duration", default=0.0, min=0.0, max=15.0, step=0.5,
                               tooltip="Seconds; 0 = the analysis duration (audio length or 10 s)."),
            ],
            outputs=[MographPlan.Output(display_name="plan"), io.String.Output(display_name="plan_json")],
        )

    @classmethod
    def execute(cls, analysis: dict, brief: str, seed: int, duration: float) -> io.NodeOutput:
        plan = planner_rules.plan_from_brief(analysis, brief or None, seed=int(seed),
                                             duration=float(duration) if duration and duration > 0 else None)
        return io.NodeOutput(plan, mplan.to_json(plan))


class MographPlanFromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographPlanFromJSON",
            display_name="Mograph Plan From JSON",
            category=CATEGORY,
            description="Parse and validate a plan JSON (hand-edited or from an LLM planner).",
            inputs=[
                MographAnalysis.Input("analysis"),
                io.String.Input("plan_json", multiline=True, default=""),
            ],
            outputs=[MographPlan.Output(display_name="plan")],
        )

    @classmethod
    def execute(cls, analysis: dict, plan_json: str) -> io.NodeOutput:
        plan = mplan.from_json(plan_json)
        _check(plan, analysis)
        return io.NodeOutput(plan)


class MographSoundtrack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographSoundtrack",
            display_name="Mograph Soundtrack",
            category=CATEGORY,
            description="User audio passes through (trimmed, faded); otherwise a CPU-synth music bed + sfx, mixed.",
            inputs=[MographPlan.Input("plan"), MographAnalysis.Input("analysis")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, plan: dict, analysis: dict) -> io.NodeOutput:
        wave, sr = soundtrack.build_soundtrack(plan, analysis)
        return io.NodeOutput(_numpy_to_audio(wave, sr))


class MographRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographRender",
            display_name="Mograph Render",
            category=CATEGORY,
            description="Render the plan on CPU and encode an MP4 (H.264 + AAC).",
            inputs=[
                MographPlan.Input("plan"),
                MographAnalysis.Input("analysis"),
                io.Audio.Input("audio", optional=True),
                io.Combo.Input("format", options=list(mplan.FORMATS), default="9:16"),
                io.Combo.Input("quality", options=["draft", "full"], default="draft",
                               tooltip="draft: short side 360 px; full: 1080x1920 / 1920x1080."),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, plan: dict, analysis: dict, format: str, quality: str,
                audio: Optional[dict] = None) -> io.NodeOutput:
        _check(plan, analysis)
        fps = float(plan.get("fps", 30))
        n_frames = int(round(float(plan["duration"]) * fps))
        width, height = geometry.frame_size(format, quality)
        wave, sr, env = None, 48000, None
        a = _audio_to_numpy(audio)
        if a is not None:
            wave, sr = a
            env = analyze.audio_envelope(wave, sr, fps, n_frames)
        frames = render_frames(plan, analysis, format, quality, audio_env=env)
        out_dir = os.path.join(folder_paths.get_temp_directory(), "mograph")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"mograph_{format.replace(':', 'x')}_{quality}_{uuid.uuid4().hex[:12]}.mp4")
        path = encode_mp4(frames, fps, width, height, wave, sr, path)
        return io.NodeOutput(InputImpl.VideoFromFile(path))


NODES = [MographAnalyze, MographPlanRules, MographBriefPrompt, MographPlanBrief, MographPlanFromJSON,
         MographSoundtrack, MographRender]

try:   # fusion pipeline nodes (nodes_fusion/, each sub-module is optional; see 16_app/DESIGN.md)
    if __package__:
        from .nodes_fusion import NODES as _FUSION_NODES
    else:
        from nodes_fusion import NODES as _FUSION_NODES
    NODES += list(_FUSION_NODES)
except Exception as _fusion_err:   # noqa: BLE001  never block the v1 nodes
    print(f"[ComfyUI-Mograph] fusion nodes not loaded: {_fusion_err}")


class MographExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> MographExtension:
    return MographExtension()
