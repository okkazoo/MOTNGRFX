"""optional_nodes: drop graph nodes whose custom-node class is not installed, so ONE graph runs with or without optional packs
(e.g. the design_llm brief: the box lacked DesignLLM_Advanced and needed a separate fusion_noqwen graph).

Sidecar (``app/graphs/<name>.app.json``) key::

    "optional_nodes": [{"class_type": "DesignLLM_Advanced",
                        "titles": ["design_llm creative brief", "Brief prompt"],      # nodes removed together (by _meta.title)
                        "note": "no Qwen: the rules write the brief"}]

When ``class_type`` is not in the server's ``NODE_CLASS_MAPPINGS`` (/object_info), the titled nodes (and every node of that
class) are removed and every input that linked to a removed node is deleted; the consumers must declare those inputs optional
(``MographMusicEncode.brief``, ``MographPiecePlan.brief``). No pure Python dependency: used by the app route
(``app_routes.load_graph``) and by ``tools/app/build_graphs.py --check``.
"""
from __future__ import annotations

import copy


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def prune_optional(graph: dict, app: dict, available) -> tuple[dict, list[dict]]:
    """-> (graph copy without the nodes of missing optional classes, [{class_type, removed: [titles], note}]).
    ``available`` = collection of installed class names; None = unknown, nothing is removed."""
    g = copy.deepcopy(graph)
    report: list[dict] = []
    if available is None:
        return g, report
    have = set(available)
    for spec in (app or {}).get("optional_nodes", []):
        cls = spec.get("class_type")
        if not cls or cls in have:
            continue
        titles = set(spec.get("titles") or [])
        ids = {nid for nid, n in g.items()
               if n.get("class_type") == cls or (n.get("_meta") or {}).get("title") in titles}
        if not ids:
            continue
        removed = sorted({(g[i].get("_meta") or {}).get("title", g[i].get("class_type", i)) for i in ids})
        for i in ids:
            del g[i]
        for n in g.values():
            for k in [k for k, v in n.get("inputs", {}).items() if _is_link(v) and v[0] in ids]:
                del n["inputs"][k]
        report.append({"class_type": cls, "removed": removed, "note": spec.get("note", "")})
    return g, report
