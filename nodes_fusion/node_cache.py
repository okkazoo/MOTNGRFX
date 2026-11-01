"""Node-level disk cache for the image-mode graph (2026-11-01, "cancel + restart with reuse"): a node whose outputs are TEXT and whose cost is a model call is wrapped so the same inputs
return the earlier reply from <ComfyUI root>/motngrfx_cache/<stage>/<key>/manifest.json instead of running the model again (also after a restart of ComfyUI, which ComfyUI's own
in-memory node cache does not survive).

    NC.cached_text_outputs(fn, stage, ignore=(), extra=None, make=None, on_hit=None)  -> a replacement for a node's `execute` function (first argument `cls`)

key = tools/app/stage_cache.make_key(stage, params = every input of the call except `ignore` (tensors by content hash), extra(params) = e.g. the model file's size + mtime).
MOTNGRFX_NO_CACHE=1 turns it off; a cache error never fails the node (it runs normally).

Wired in: MographGGUFVL (the Gemma 4 / Qwen GGUF call: concept, design, repair, story, media-read replies). NOT wired (and why, see BUILT.md): the LTX shot clips and the Alpha-Gen mattes of the
image graph are chains of core ComfyUI nodes (sampler -> VAE decode) whose result a custom node cannot reach before the sampler has run; skipping them would need a lazy cache node in the
app graphs (a graph change for every tier) and would put the LTX/IC-LoRA/sampler settings into a key the node cannot see. The own-video route caches its mattes (tools/app/own_video_job.py).
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import os
from pathlib import Path

from . import ensure_root_on_path

ensure_root_on_path()
from tools.app import stage_cache as SCH      # noqa: E402


def cache_root() -> Path | None:
    """env MOTNGRFX_CACHE_DIR, else <ComfyUI root>/motngrfx_cache (the parent of the output folder, same filesystem; never inside output/), else None (outside ComfyUI with no env: no cache)."""
    env = os.environ.get("MOTNGRFX_CACHE_DIR", "").strip()
    if env:
        return Path(env)
    try:
        import folder_paths
        return SCH.default_cache_root(Path(folder_paths.get_output_directory()).parent)
    except Exception:   # noqa: BLE001
        return None


def _plain(v):
    """a value as JSON-able key material: tensors / arrays by content hash, paths as text."""
    if hasattr(v, "detach") and hasattr(v, "cpu"):                       # torch tensor
        a = v.detach().cpu().contiguous().numpy()
        return dict(tensor=hashlib.sha256(a.tobytes()).hexdigest(), shape=list(a.shape), dtype=str(a.dtype))
    if hasattr(v, "tobytes") and hasattr(v, "shape"):                    # numpy array
        return dict(array=hashlib.sha256(v.tobytes()).hexdigest(), shape=list(v.shape), dtype=str(v.dtype))
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def file_stamp(path) -> dict:
    """size + mtime of a (large) model file: a replaced model changes the key without hashing gigabytes."""
    try:
        st = os.stat(str(path))
        return dict(path=os.path.basename(str(path)), size=st.st_size, mtime_ns=st.st_mtime_ns)
    except OSError:
        return dict(path=str(path), missing=True)


def cached_text_outputs(fn, stage: str, ignore=(), extra=None, make=None, on_hit=None):
    """wrap a node's classmethod function `fn(cls, **inputs)` whose result `.args` are JSON-able (strings): hit -> make(values), miss -> run + store.
    make(values) builds the node's output (default: the type of the result with the values as positional args); on_hit(values) may annotate them (e.g. a stats JSON)."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(cls, *args, **kwargs):
        if not SCH.enabled():
            return fn(cls, *args, **kwargs)
        store = key = None
        try:
            root = cache_root()
            if root is not None:
                bound = sig.bind(cls, *args, **kwargs)
                bound.apply_defaults()
                params = {k: _plain(v) for k, v in bound.arguments.items() if k != "cls" and k not in ignore}
                if extra is not None:
                    params["_extra"] = extra(dict(bound.arguments))
                store, key = SCH.StageCache(root), SCH.make_key(stage, params=params)
                hit = store.get_meta(stage, key)
                if hit is not None and isinstance(hit.get("values"), list) and make is not None:
                    vals = list(hit["values"])
                    vals = on_hit(vals) if on_hit else vals
                    print(f"[node_cache] hit {stage} {key[:10]}: reply reused, the model is not run")
                    return make(vals)
        except Exception as e:   # noqa: BLE001 - a cache fault must never fail the node
            print(f"[node_cache] {stage}: cache skipped ({type(e).__name__}: {e})")
            store = key = None
        res = fn(cls, *args, **kwargs)
        if store is not None and key:
            try:
                vals = list(getattr(res, "args", None) or [])
                if vals and all(isinstance(x, str) for x in vals):
                    store.put_meta(stage, key, dict(values=vals))
            except Exception as e:   # noqa: BLE001
                print(f"[node_cache] {stage}: store skipped ({type(e).__name__}: {e})")
        return res
    return wrapper
