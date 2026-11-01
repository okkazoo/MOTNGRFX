"""nodes_fusion: ComfyUI nodes for the fusion pipeline (DESIGN.md in 16_app/).

``NODES`` is the concatenation of the ``NODES`` lists of the sub-modules below. Each sub-module is imported in its own
try/except: a missing or broken module (written by another work package) is logged and skipped so the rest still loads.

Imports of the pure-Python code go through the repo ROOT on ``sys.path`` as top-level ``tools.fusion`` / ``mograph``
(never the relative ``.mograph`` package that ``nodes.py`` uses: that would create a second module registry).
ROOT is appended, not inserted, so ComfyUI's own top-level ``nodes`` module is never shadowed by the repo's ``nodes.py``.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_root_on_path() -> str:
    """Make ``tools.fusion`` and ``mograph`` importable as top-level packages. Idempotent."""
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    return ROOT


ensure_root_on_path()

SUBMODULES = ("plan", "audio", "type", "compose", "inputs", "footage", "latent_mask", "music_score", "music_t2m", "music_preview", "live_preview", "depth_transition", "design", "concept", "archive", "redesign", "layers", "llm_gguf", "final_save", "layout_preview", "media_read", "story")

NODES: list = []
LOADED: dict[str, int] = {}     # submodule -> number of nodes
FAILED: dict[str, str] = {}     # submodule -> error text

for _name in SUBMODULES:
    try:
        _mod = importlib.import_module(f"{__name__}.{_name}")
        _nodes = list(getattr(_mod, "NODES", []))
        NODES.extend(_nodes)
        LOADED[_name] = len(_nodes)
    except ModuleNotFoundError as _e:
        if _e.name == f"{__name__}.{_name}":
            FAILED[_name] = "module not present"
            print(f"[ComfyUI-Mograph] fusion module '{_name}' not present yet; skipped")
        else:
            FAILED[_name] = f"{type(_e).__name__}: {_e}"
            print(f"[ComfyUI-Mograph] fusion module '{_name}' failed to import: {_e}\n{traceback.format_exc()}")
    except Exception as _e:   # noqa: BLE001
        FAILED[_name] = f"{type(_e).__name__}: {_e}"
        print(f"[ComfyUI-Mograph] fusion module '{_name}' failed to import: {_e}\n{traceback.format_exc()}")

if os.environ.get("MOGRAPH_SPIKE") == "1":   # lazy-gating spike nodes (16_app/DESIGN.md "Spike results")
    try:
        from . import _spike
        NODES.extend(_spike.SPIKE_NODES)
    except Exception as _e:   # noqa: BLE001
        print(f"[ComfyUI-Mograph] spike nodes not loaded: {_e}")

try:   # HTTP routes (GET /mograph/api/looks); needs a running PromptServer
    from . import routes as _routes
    _routes.register()
except Exception as _e:   # noqa: BLE001
    print(f"[ComfyUI-Mograph] fusion routes not registered: {_e}")
