"""The shipped default graphs (Gemma 4 31B QAT) need no MOSS edit LoRA (licence undeclared), name their real design model,
and the LoRA is only an explicit download extra."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _common as C  # noqa: E402

GEMMA = "fusion_design_qwen_auto_layers_pivot_gemma31b_qat"
LORA = "moss_image_edit_chinoll"


def _load(name: str):
    g = json.loads((C.GRAPHS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    s = json.loads((C.GRAPHS_DIR / f"{name}.app.json").read_text(encoding="utf-8"))
    return g, s


def _names(*names):
    """the Qwen graph is legacy and left out of the public export (user 2026-10-30: Gemma only): its checks run only where the file exists."""
    return [n for n in names if (C.GRAPHS_DIR / f"{n}.json").is_file()]


def test_default_graphs_have_no_angle_lora():
    for name in _names(GEMMA):
        g, _ = _load(name)
        assert LORA not in json.dumps(g), name
        assert "ang_lora" not in g and not [k for k in g if k.startswith("ang")], name
        assert "angles" not in g["first_set"]["inputs"], name


def test_no_shipped_graph_loads_the_angle_lora_unless_built_with_angles():
    for p in C.graph_files(["*.json"]):
        if p.stem.endswith("_angles"):
            continue
        assert LORA not in p.read_text(encoding="utf-8"), p.name


def test_default_graph_uses_the_gguf_and_keeps_its_quality_field():
    g, _ = _load(GEMMA)
    models = {n["inputs"]["model"] for n in g.values() if n["class_type"] == "MographGGUFVL"}
    assert models == {"gemma-4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"}
    assert g["quality"]["class_type"] == "PrimitiveString"


def test_labels_name_the_real_design_model():
    for name, model, other in [x for x in ((GEMMA, "Gemma 4 31B", "Qwen3.8"),) if x[0] in _names(GEMMA)]:
        _, s = _load(name)
        text = json.dumps({"title": s["title"], "note": s["note"], "labels": s["stage_labels"]})
        assert model in s["title"] and model in s["note"], name
        assert "design_llm_8b invents" not in text and "Qwen designs the type" not in text, name
        assert model + " designs the type" in s["stage_labels"].values(), name
        assert other not in s["title"], name


def test_the_angle_lora_is_an_explicit_extra_never_a_default_download():
    assert LORA + ".safetensors" not in C.MODELS
    assert C.EXTRAS["angles"]["dest"] == LORA + ".safetensors"
    assert C.model_files(C.graph_files(["fusion*.json"], downloadable=True)).get(LORA + ".safetensors") is None


# ---- the shipped names (2026-10-30): motngrfx_default / _16gb / _3060 are the Gemma tier defaults under neutral names; the long fusion_design_qwen_* names stay as aliases
SHIPPED = {"motngrfx_default": (GEMMA, "gemma31b_qat", 0), "motngrfx_16gb": ("fusion_design_qwen_auto_layers_gemma12b_q8", "gemma12b_q8", 1),
           "motngrfx_3060": ("fusion_design_qwen_auto_layers_gemma12b_q6", "gemma12b_q6", 2)}


def test_shipped_names_exist_and_equal_the_arm_graphs():
    for alias, (old, _arm, order) in SHIPPED.items():
        assert (C.GRAPHS_DIR / f"{alias}.json").is_file(), alias
        g, s = _load(alias)
        g_old, s_old = _load(old)
        assert g == g_old, f"{alias} differs from {old}"
        assert s["alias_of"] == old and s["order"] == order, alias
        assert s["title"].startswith("MOTNGRFX") and "Qwen" not in s["title"], alias
        assert {k: v for k, v in s.items() if k not in ("title", "order", "alias_of")} == {k: v for k, v in s_old.items() if k not in ("title", "order")}, alias


_BASES = ["fusion_design_qwen_auto_layers_pivot.json", "fusion_design_qwen_auto_layers.json"]          # the builder's base graphs (LA.DEFAULT_ALIASES); the Qwen ones are not shipped
_GONE = [b for b in _BASES if not (C.GRAPHS_DIR / b).exists()]


@pytest.mark.skipif(bool(_GONE), reason=f"dev fixture not shipped in the public export: app/graphs/{_GONE[0] if _GONE else ''}")
def test_motngrfx_default_is_generated_by_the_builder_and_opens_first():
    sys.path.insert(0, str(ROOT))
    from tools.app import llm_arms as LA
    assert sorted(LA.DEFAULT_ALIASES) == sorted(SHIPPED)
    for alias, (_old, arm, _o) in SHIPPED.items():
        g, s = LA.alias_files(alias, str(C.GRAPHS_DIR))
        g_disk, s_disk = _load(alias)
        assert g == g_disk and s == s_disk, f"{alias} is stale (run tools/app/llm_arms.py)"
        assert LA.ARMS[arm]["family"] == "gemma"
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_routes_dg", ROOT / "app_routes.py")
    routes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routes)
    names = [t["name"] for t in routes.list_graphs()]
    assert names[0] == "motngrfx_default", names[:3]
