"""ComfyUI-Mograph: text / images / audio -> 5-15 s motion graphic (CPU, v1 rough cut)."""
# ComfyUI imports this file as a package (so __package__ is set). pytest also imports the repo
# root's __init__.py as a plain top-level module when collecting tests; skip the ComfyUI wiring
# there so the standalone tests never need comfy_api / torch.
if __package__:
    from .nodes import MographExtension, comfy_entrypoint

    __all__ = ["comfy_entrypoint", "MographExtension"]

    try:   # web front-end at /mograph (needs a running PromptServer; never block the nodes if it is missing)
        from . import app_routes
        app_routes.register()
    except Exception as _e:   # noqa: BLE001
        print(f"[ComfyUI-Mograph] web app routes not registered: {_e}")
