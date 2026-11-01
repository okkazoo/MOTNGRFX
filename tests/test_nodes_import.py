"""Node pack import + schema smoke test.

Needs a ComfyUI checkout importable (set MOGRAPH_COMFYUI_PATH, or the pack installed under
ComfyUI/custom_nodes, or a sibling ``ComfyUI`` folder). Skips cleanly when comfy_api cannot be
imported (e.g. its ``av`` dependency is missing) or when mograph modules are not built yet.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_IDS = ["MographAnalyze", "MographPlanRules", "MographBriefPrompt", "MographPlanBrief", "MographPlanFromJSON",
            "MographSoundtrack", "MographRender"]


def _comfy_candidates():
    env = os.environ.get("MOGRAPH_COMFYUI_PATH")
    if env:
        yield env
    yield os.path.abspath(os.path.join(ROOT, "..", ".."))          # installed in ComfyUI/custom_nodes
    yield os.path.abspath(os.path.join(ROOT, "..", "ComfyUI"))     # sibling checkout


def _import_comfy():
    for path in _comfy_candidates():
        if os.path.isfile(os.path.join(path, "comfy_api", "latest", "__init__.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import comfy_api.latest  # noqa: F401
                import folder_paths  # noqa: F401
            except Exception as e:  # missing deps (av, ...) in this python
                pytest.skip(f"comfy_api import failed from {path}: {type(e).__name__}: {e}")
            return path
    pytest.skip("no ComfyUI checkout found (set MOGRAPH_COMFYUI_PATH)")


def _load_pack():
    """Load the repo root as a package exactly like ComfyUI's custom node loader does."""
    name = "comfyui_mograph_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "__init__.py"),
                                                  submodule_search_locations=[ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        del sys.modules[name]
        if "mograph" in str(e) or "render" in str(e) or "analyze" in str(e):
            pytest.skip(f"mograph modules not complete yet: {e}")
        raise
    return mod


def test_node_schemas_build():
    _import_comfy()
    pack = _load_pack()
    ext = asyncio.run(pack.comfy_entrypoint())
    nodes = asyncio.run(ext.get_node_list())
    ids = []
    for cls in nodes:
        schema = cls.define_schema()
        assert schema.display_name.startswith(("Mograph", "MOTNGRFX")), schema.display_name
        assert schema.category in ("MOTNGRFX", "MOTNGRFX/Fusion", "Mograph/Spike")
        if schema.category == "MOTNGRFX":          # v1 nodes; the fusion pack (nodes_fusion/) is appended after them
            ids.append(schema.node_id)
        cls.GET_SCHEMA()          # full v3 finalisation path used by ComfyUI's loader
    assert ids == NODE_IDS


def test_workflow_uses_registered_node_ids():
    """Runs without ComfyUI: the example workflow only references our ids and core nodes."""
    with open(os.path.join(ROOT, "nodes.py"), encoding="utf-8") as f:
        declared = re.findall(r'node_id="([^"]+)"', f.read())
    assert declared == NODE_IDS
    with open(os.path.join(ROOT, "example_workflows", "mograph_basic.json"), encoding="utf-8") as f:
        wf = json.load(f)
    types = {n["type"] for n in wf["nodes"]}
    assert {"MographAnalyze", "MographPlanRules", "MographSoundtrack", "MographRender"} <= types
    assert types <= set(NODE_IDS) | {"LoadImage", "SaveVideo", "MarkdownNote", "Note"}
    node_ids = {n["id"] for n in wf["nodes"]}
    for link in wf["links"]:
        lid, src, _, dst, _, _ = link
        assert src in node_ids and dst in node_ids
    # every link is referenced by both ends
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") is not None:
                assert any(l[0] == inp["link"] and l[3] == n["id"] for l in wf["links"])
