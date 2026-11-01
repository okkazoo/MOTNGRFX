"""Web front-end routes for MOTNGRFX (the ComfyUI-Mograph node pack).

Serves the static page in ``web_app/`` at ``/mograph`` (``/motngrfx`` is an alias of the same page) plus the workflow templates in ``app/graphs/`` and the example
inputs in ``app/examples/``. The page itself talks to the server's own ``/upload/image``, ``/prompt``, ``/ws``,
``/view`` and ``/history``. Registered from ``__init__.py`` (only when loaded as a ComfyUI package).
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web_app"
GRAPHS_DIR = ROOT / "app" / "graphs"
EXAMPLES_DIR = ROOT / "app" / "examples"

_NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("video/mp4", ".mp4")


def _safe_child(base: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``base``; None if it escapes ``base`` or is not a file."""
    try:
        p = (base / rel).resolve()
        p.relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None


def _file_response(path: Path) -> web.StreamResponse:
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return web.FileResponse(path, headers={**_NO_CACHE, "Content-Type": ctype})


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def list_graphs() -> list[dict]:
    out = []
    for p in sorted(GRAPHS_DIR.glob("*.json")):
        if p.name.endswith(".app.json") or p.name.startswith("_"):      # `_name` = a helper graph the page fills in itself (_layout_preview), not a template
            continue
        side = p.with_name(p.stem + ".app.json")
        meta = _read_json(side) if side.exists() else {}
        out.append({"name": p.stem, "title": meta.get("title", p.stem), "has_sidecar": side.exists(),
                    "order": meta.get("order", 100)})
    out.sort(key=lambda t: (t["order"], t["name"]))     # the page opens the first one: sidecar "order" 0 = default
    return out


def _installed_classes():
    """class names the running server knows (None outside a server: nothing is pruned)."""
    try:
        import nodes as comfy_nodes     # ComfyUI core, already in sys.modules inside the server
        return set(comfy_nodes.NODE_CLASS_MAPPINGS)
    except Exception:   # noqa: BLE001
        return None


def _prune_optional(graph: dict, app: dict, available) -> tuple[dict, list]:
    """sidecar ``optional_nodes``: drop the nodes of optional packs that are not installed (one graph, with or without Qwen)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mograph_optional_nodes", ROOT / "tools" / "app" / "optional_nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prune_optional(graph, app, available)


def load_graph(name: str, available=None) -> dict | None:
    """template + sidecar. ``available`` = installed class names: nodes of missing optional packs (sidecar ``optional_nodes``) are
    removed and listed in ``pruned``."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    p = GRAPHS_DIR / f"{name}.json"
    if not p.is_file():
        return None
    side = GRAPHS_DIR / f"{name}.app.json"
    graph, app = _read_json(p), (_read_json(side) if side.exists() else {})
    out = {"name": name, "graph": graph, "app": app}
    if available is not None and app.get("optional_nodes"):
        out["graph"], out["pruned"] = _prune_optional(graph, app, available)
    return out


def list_examples() -> list[dict]:
    out = []
    for p in sorted(EXAMPLES_DIR.glob("*.json")):
        d = _read_json(p)
        d.setdefault("id", p.stem)
        out.append(d)
    return out


def save_layout_lock(archive: str, stem: str, seed, effects=None, output_dir=None) -> dict:
    """"Use this layout" of the Randomise-layout panel: writes `<archive>/<stem>_layout_lock.json` (tools/fusion/layout_preview.py) next to the Preview run's files; tools/app/upgrade_still.py applies it
    to the upgrade, and the page sends the seed as the request's `layout_seed` for the next run. `archive` is relative to the ComfyUI output dir and must stay inside it."""
    import re
    import sys
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from tools.fusion import layout_preview as LPV
    if output_dir is None:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    base = Path(output_dir).resolve()
    d = (base / str(archive or "")).resolve()
    try:
        d.relative_to(base)
    except ValueError:
        return {"error": "archive is outside the output folder"}
    stem = str(stem or "")
    n = LPV.layout_seed_of(seed)
    if not d.is_dir() or not re.fullmatch(r"[\w.\- ]{1,200}", stem) or n is None:
        return {"error": "need an archive folder, a stem and a seed >= 1"}
    p = LPV.write_lock(d, stem, n, LPV.effects_list(effects))
    return {"ok": True, "file": p.name, "layout_seed": n}


def register() -> None:
    import server   # ComfyUI's server.py; only importable inside a running ComfyUI

    routes = server.PromptServer.instance.routes

    async def index(request):
        return _file_response(WEB_DIR / "index.html")

    for _path in ("/mograph", "/mograph/", "/motngrfx", "/motngrfx/"):      # /motngrfx = the public name, /mograph kept for graphs
        routes.get(_path)(index)

    @routes.get("/mograph/static/{path:.*}")
    async def mograph_static(request):
        p = _safe_child(WEB_DIR, request.match_info["path"])
        if p is None:
            return web.Response(status=404, text="not found")
        return _file_response(p)

    @routes.get("/mograph/graphs")
    async def mograph_graphs(request):
        return web.json_response(list_graphs(), headers=_NO_CACHE)

    @routes.get("/mograph/graphs/{name}")
    async def mograph_graph(request):
        g = load_graph(request.match_info["name"], _installed_classes())
        if g is None:
            return web.json_response({"error": "unknown template"}, status=404)
        return web.json_response(g, headers=_NO_CACHE)

    @routes.post("/mograph/layout_lock")
    async def mograph_layout_lock(request):
        try:
            b = await request.json()
        except Exception:   # noqa: BLE001
            return web.json_response({"error": "bad json"}, status=400)
        r = save_layout_lock(b.get("archive"), b.get("stem"), b.get("seed"), b.get("effects"))
        return web.json_response(r, status=400 if "error" in r else 200, headers=_NO_CACHE)

    @routes.get("/mograph/examples")
    async def mograph_examples(request):
        return web.json_response(list_examples(), headers=_NO_CACHE)

    @routes.get("/mograph/examples/{file:.*}")
    async def mograph_example_file(request):
        p = _safe_child(EXAMPLES_DIR, request.match_info["file"])
        if p is None:
            return web.Response(status=404, text="not found")
        return _file_response(p)

    try:                                    # GET /mograph/fonts/liked + /mograph/fonts/file/<file> (tools/app/fonts_route.py; whitelist = mograph/fonts/liked.json)
        import sys as _sys0
        if str(ROOT) not in _sys0.path:
            _sys0.path.append(str(ROOT))
        from tools.app import fonts_route
        fonts_route.register(routes)
    except Exception as e:   # noqa: BLE001 - the page keeps working without the route
        print(f"[MOTNGRFX] fonts route not registered: {e}")

    try:                                    # POST /mograph/own_video + GET /mograph/own_video/<job> (tools/app/own_video_route.py; the runner is a subprocess)
        import sys as _sys
        if str(ROOT) not in _sys.path:
            _sys.path.append(str(ROOT))
        from tools.app import own_video_route
        own_video_route.register(routes)
    except Exception as e:   # noqa: BLE001 - the page keeps working without the route
        print(f"[MOTNGRFX] own-video route not registered: {e}")
