"""HTTP routes of the fusion pack: ``GET /mograph/api/looks``.

``register()`` is called from ``nodes_fusion/__init__.py`` (the root ``__init__.py`` belongs to the front-end worker);
like ``app_routes.register`` it needs a running PromptServer and is idempotent. The family list is static (DESIGN.md
section 2, modules grouped by base class) so the route costs no heavy import.
"""
from __future__ import annotations

# family -> module names (mograph.modules registry; asserted by tests/test_fusion_footage.py when the registry loads)
LOOK_FAMILIES: dict[str, list[str]] = {
    "strand": ["lanes", "pipework", "circuit"],
    "mech": ["depth_echo", "hollow_tubes", "topo_tower", "hinge_unfold", "xray_unfold", "candy_orbit", "depth_tunnel"],
    "slab": ["beat_bars", "bold_thin_echo", "gradient_fade", "scale_clash", "slats", "solid_hollow", "staircase"],
    "stack": ["spectrum", "tangent_spray"],
    "ring": ["fly_into"],           # `outflow` retired 2026-11-02 (design_defaults.RETIRED_MODULES)
    "voxel": ["voxel_orbit", "voxel_roll"],
    "particles": ["particles"],
}


def looks_payload() -> dict:
    """{"looks": [{"value","label","modules"}...], "families": {...}}; first entry is ``auto``.

    The Look input picks a FAMILY (DESIGN.md: "Look picks the family; seed picks modules within it"), so ``value`` is
    ``auto`` or a family name; ``modules`` lists what the family contains.
    """
    looks = [{"value": "auto", "label": "Auto", "modules": []}]
    for fam, mods in LOOK_FAMILIES.items():
        looks.append({"value": fam, "label": fam.capitalize(), "modules": list(mods)})
    return {"looks": looks, "families": LOOK_FAMILIES}


_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    import server   # ComfyUI's server.py; only importable inside a running ComfyUI
    from aiohttp import web

    routes = server.PromptServer.instance.routes

    @routes.get("/mograph/api/looks")
    async def mograph_api_looks(request):
        return web.json_response(looks_payload(), headers={"Cache-Control": "no-cache, must-revalidate"})

    _registered = True
