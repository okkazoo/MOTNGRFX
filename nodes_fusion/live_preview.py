"""live_preview: installs the MOTNGRFX frame-strip sampler preview into ComfyUI at import (no nodes).

The work is ``tools/fusion/strip_preview.py``: `latent_preview.get_previewer` is wrapped once, and ONLY for prompts that carry ``extra_data.motngrfx_strip_preview``
(the web page and tools/app/own_video_job.py set it) a 5D video latent is decoded as a whole frame strip and sent to the submitting client as the websocket event
``motngrfx.strip``. Everything else keeps ComfyUI's own previewer. A failure here only logs: the pack still loads and the page falls back to the single still preview.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:      # appended, never inserted (ComfyUI's own ``nodes`` module must stay first)
    sys.path.append(_ROOT)

NODES: list = []

try:
    from tools.fusion import strip_preview as _sp
    INSTALLED = _sp.install()
except Exception as _e:   # noqa: BLE001
    INSTALLED = False
    print(f"[ComfyUI-Mograph] strip preview not installed: {type(_e).__name__}: {_e}")
