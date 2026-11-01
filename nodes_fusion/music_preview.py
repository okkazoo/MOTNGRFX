"""MographLiveWaveformPreview: the music sampler's live waveform for the web page (the owner's ACE15LiveWaveformPreview, renamed).

A MODEL patch between ModelSamplingAuraFlow and the ACE KSampler of the text-to-music chain (``tools/fusion/music_graph.t2m_chain``). At every sampler
step the x0 prediction is VAE-decoded and sent as a peak envelope (websocket event ``motngrfx.waveform``) to the client that submitted the prompt.
The sampled latent is not changed (bit-identical render); ``enabled`` off passes the model through. The work is ``tools/fusion/waveform_preview.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted (ComfyUI's own ``nodes`` module must stay first)
    sys.path.append(_ROOT)

from tools.fusion import waveform_preview as WP  # noqa: E402

from ._types import CATEGORY  # noqa: E402


class MographLiveWaveformPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MographLiveWaveformPreview",
            display_name="MOTNGRFX Live Waveform Preview",
            category=CATEGORY,
            description="ACE-Step: a live waveform while the song renders. At every sampler step the x0 prediction is VAE-decoded and sent as a peak "
                        "envelope (websocket event motngrfx.waveform, to the submitting client only). Steps are skipped while the last decode costs more "
                        "than the sampling since it. The sampled latent is not changed; enabled off = the model passes through untouched.",
            inputs=[io.Model.Input("model"), io.Vae.Input("vae"),
                    io.Boolean.Input("enabled", default=True),
                    io.Int.Input("bins", default=WP.BINS, min=64, max=8192, step=1)],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, vae, enabled=True, bins=WP.BINS) -> io.NodeOutput:
        m = WP.patch_model(model, vae, bool(enabled), int(bins))
        return io.NodeOutput(m, ui={"live_preview": [bool(enabled)], "event": [WP.EVENT]})


NODES = [MographLiveWaveformPreview]
