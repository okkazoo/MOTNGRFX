"""LLM arms (tools/app/llm_arms.py, tools/app/llm_gguf.py, tools/app/llm_probe.py, nodes_fusion/llm_gguf.py, tools/app/llm_arms.py): the graph conversion wiring,
the checker, the runner's pure helpers and the files on disk. No GPU, no model, no llama.cpp. Run: python -m pytest tests/test_llm_arms.py -q"""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.app import llm_arms as LA  # noqa: E402
from tools.app import llm_gguf as LG  # noqa: E402

GRAPHS = os.path.join(ROOT, "app", "graphs")
BASES = [b for b in LA.BASES if os.path.isfile(os.path.join(GRAPHS, b + ".json"))]


def _read(name):
    with open(os.path.join(GRAPHS, name), encoding="utf-8") as f:
        return json.load(f)


def _qwen(prompt, image=None, max_tokens=4096, temperature=0.7, seed=("seed", 0)):
    ins = {"model_name": "design_llm_8b", "quantization": "4-bit (VRAM-friendly)", "attention_mode": "auto", "use_torch_compile": False, "device": "auto",
           "preset_prompt": "x", "custom_prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9, "num_beams": 1,
           "repetition_penalty": 1.05, "frame_count": 16, "video_frame_size": "auto", "keep_model_loaded": False, "seed": list(seed)}
    if image:
        ins["image"] = image
    return {"inputs": ins, "class_type": LA.OLD, "_meta": {"title": "design_llm design" if image else "design_llm concept"}}


def _mini():
    """concept -> parse; design pkg -> design (image = sheet, slot 3) -> parse -> repair (lazy), consumers read slot 0."""
    return {
        "seed": {"inputs": {"value": 3}, "class_type": "PrimitiveInt", "_meta": {"title": "IN seed"}},
        "concept_prompt": {"inputs": {}, "class_type": "MographConceptPrompt", "_meta": {"title": "Concept prompt"}},
        "concept_qwen": _qwen(["concept_prompt", 0], max_tokens=700, temperature=0.9),
        "concept_parse": {"inputs": {"reply": ["concept_qwen", 0]}, "class_type": "MographConceptParse", "_meta": {"title": "Concept parse"}},
        "design_pkg": {"inputs": {}, "class_type": "MographDesignPackage", "_meta": {"title": "Design package"}},
        "design_qwen": _qwen(["design_pkg", 0], image=["design_pkg", 3]),
        "design_parse": {"inputs": {"reply": ["design_qwen", 0], "prompt": ["design_pkg", 0]}, "class_type": "MographDesignParse", "_meta": {"title": "Design parse"}},
        "design_qwen2": _qwen(["design_parse", 2]),
        "design_fix": {"inputs": {"piece_json": ["design_parse", 0], "ok": ["design_parse", 1], "repair_reply": ["design_qwen2", 0]},
                       "class_type": "MographDesignRepair", "_meta": {"title": "Design repair"}},
        "out_reply": {"inputs": {"source": ["design_qwen", 0]}, "class_type": "PreviewAny", "_meta": {"title": "OUT design reply"}},
    }


MINI_SIDE = {"title": "Fusion", "order": 21, "note": "n", "stage_labels": {"design_llm design": "Qwen designs the type"}}


@pytest.fixture(params=["mini"] + BASES)
def base(request):
    if request.param == "mini":
        return request.param, _mini(), copy.deepcopy(MINI_SIDE)
    return request.param, _read(request.param + ".json"), _read(request.param + ".app.json")


# --------------------------------------------------------------------------------------------- arms table
def test_arm_table_is_complete_and_open():
    assert len(LA.ARMS) >= 3
    fams = {a["family"] for a in LA.ARMS.values()}
    assert fams == {"gemma"}
    for k, a in LA.ARMS.items():
        assert a["licence"] == "Apache-2.0", k
        assert a["file"].endswith(".gguf") and a["mmproj_file"].endswith(".gguf") and "mmproj" in a["mmproj_file"]
        assert a["repo"].count("/") == 1 and a["kv_cache"] in LG.KV_TYPES and a["handler"] in LG.HANDLERS
        assert a["n_ctx"] >= 12288 + 4096 // 2            # the full design prompt (~9-12k tokens with the sheet) + the reply
        assert a["n_gpu_layers"] >= -1
    assert sum(1 for a in LA.ARMS.values() if a["family"] == "gemma") >= 2


def test_arm_files_and_fetch_lines():
    a = LA.ARMS["gemma12b_q6"]
    assert LA.arm_files(a) == [(a["file"], f"{a['dir']}/{a['file']}"), (a["mmproj_file"], f"{a['dir']}/{a['mmproj_file']}")]
    ls = LA.fetch_lines("gemma12b_q6")
    assert len(ls) == 2 and all("aria2c -x4" in l and "huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/" in l for l in ls)


# --------------------------------------------------------------------------------------------- conversion wiring
@pytest.mark.parametrize("arm", sorted(LA.ARMS))
def test_conversion_wiring(base, arm):
    name, g0, s0 = base
    n_old = len(LA.llm_nodes(g0))
    assert n_old >= 2
    g, s = LA.convert(g0, s0, arm)
    a = LA.ARMS[arm]
    assert not LA.llm_nodes(g) and len([n for n in g.values() if n["class_type"] == LA.NEW]) == n_old
    assert set(g) == set(g0)                                                # ids kept: run_timing stages / run_batch / consumers read them unchanged
    for nid in LA.llm_nodes(g0):
        o, n = g0[nid]["inputs"], g[nid]["inputs"]
        assert n["prompt"] == o["custom_prompt"]
        assert n["seed"] == o["seed"] and (n["max_tokens"] == o["max_tokens"] or (n.get("json_mode") and n["max_tokens"] == max(o["max_tokens"], LA.JSON_MODE_TOKENS))) and n["temperature"] == o["temperature"]
        assert n["model"] == f"{a['dir']}/{a['file']}" and n["keep_model_loaded"] is False and n["flash_attn"] is True
        assert n["n_gpu_layers"] == a["n_gpu_layers"] and n["kv_cache"] == a["kv_cache"]
        assert (n["top_p"], n["top_k"], n["repeat_penalty"]) == (a["top_p"], a["top_k"], a["repeat_penalty"])      # the sampling the probes were measured with
        assert n["mmproj_on_gpu"] == a["mmproj_on_gpu"] and n["n_cpu_moe"] == a["n_cpu_moe"]
        assert n["n_ctx"] >= n["max_tokens"] + 1024
        if "image" in o:
            assert n["image"] == o["image"] == ["design_pkg", 3] and n["mmproj"].endswith(a["mmproj_file"])   # the combined sheet, slot 3 of the Design Package
            assert n["n_ctx"] == a["n_ctx"]
        else:
            assert "image" not in n and n["n_ctx"] == LA.ctx_for(a, False) and n["mmproj"] == "none"
        assert g[nid]["_meta"]["title"].startswith(a["label"])
    assert LA.check(g, arm, g0) == []
    assert s["llm"]["file"] == a["file"] and arm_label_in(s["title"], a) and "LLM arm" in s["note"]
    assert g0 == (_mini() if name == "mini" else _read(name + ".json"))      # pure: the base is not modified


def arm_label_in(title, a):
    return a["label"] in title


def test_design_nodes_that_name_the_model_get_the_arms_real_model_name():
    """plan.design.model (the piece, the archive) must name the arm's model, not the node default 'design_llm_8b' (the generated graphs carry no `model` input on Parse / Repair)."""
    g0 = _mini()
    g0["design_parse"]["inputs"]["model"] = "design_llm_8b-4bit"
    g, _ = LA.convert(g0, None, "gemma12b_q6")
    assert g["design_parse"]["inputs"]["model"] == LA.ARMS["gemma12b_q6"]["label"]
    g1, _ = LA.convert(_mini(), None, "gemma31b_qat")
    assert "model" not in _mini()["design_parse"]["inputs"]
    assert g1["design_parse"]["inputs"]["model"] == "Gemma 4 31B QAT Q4_K_XL" and g1["design_fix"]["inputs"]["model"] == "Gemma 4 31B QAT Q4_K_XL"


def test_non_llm_nodes_untouched(base):
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q8")
    for nid, n in g0.items():
        if n["class_type"] != LA.OLD:
            assert LA._strip_label(g[nid]) == LA._strip_label(n)


def test_consumers_read_slot_zero_and_check_rejects_slot_one(base):
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q6")
    nid = LA.llm_nodes(g0)[0]
    bad = copy.deepcopy(g)
    for n in bad.values():
        for k, v in n["inputs"].items():
            if v == [nid, 0] and n["class_type"] != "PreviewAny":
                n["inputs"][k] = [nid, 1]
                break
        else:
            continue
        break
    assert any("only the response (slot 0)" in p for p in LA.check(bad))


def test_checker_catches_breakage(base):
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q6")
    dsg = next(n for n in LA.llm_nodes(g0) if "image" in g0[n]["inputs"])
    b = copy.deepcopy(g)
    b[dsg]["inputs"]["image"] = ["design_pkg", 1]                            # the padded batch, not the sheet
    assert any("must be the Design Package" in p for p in LA.check(b))
    b = copy.deepcopy(g)
    b[dsg]["inputs"]["keep_model_loaded"] = True
    assert any("keep_model_loaded" in p for p in LA.check(b))
    b = copy.deepcopy(g)
    b[dsg]["inputs"]["n_ctx"] = 2048
    assert any("n_ctx" in p for p in LA.check(b))
    b = copy.deepcopy(g)
    b[dsg]["inputs"]["model"] = "somewhere/else.gguf"
    assert any("is not the arm's" in p for p in LA.check(b, "gemma12b_q6"))
    b = copy.deepcopy(g)
    b[dsg]["class_type"] = LA.OLD
    assert any(LA.OLD in p for p in LA.check(b))
    b = copy.deepcopy(g)
    b["extra"] = {"inputs": {}, "class_type": "PrimitiveInt", "_meta": {"title": "x"}}
    assert any("node ids differ" in p for p in LA.check(b, None, g0))
    b = copy.deepcopy(g)
    nid = next(n for n in g0 if g0[n]["class_type"] != LA.OLD and g0[n]["inputs"])
    b[nid]["inputs"] = {}
    assert any("was changed" in p for p in LA.check(b, None, g0))


def test_convert_twice_and_unknown_arm_are_errors(base):
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q6")
    with pytest.raises(ValueError):
        LA.convert(g, None, "gemma12b_q6")
    with pytest.raises(ValueError):
        LA.convert(g0, s0, "nope")


def test_sidecar_stage_labels_follow_the_new_titles():
    g, s = LA.convert(_mini(), copy.deepcopy(MINI_SIDE), "gemma12b_q6")
    t = g["design_qwen"]["_meta"]["title"]
    assert t == "Gemma 4 12B Q6_K design" and t in s["stage_labels"]


def test_probe_graph_is_one_model_with_sheet_and_prompt():
    g = LA.probe_graph("gemma12b_q6", "design this", n_gpu_layers=40, n_ctx=12000)
    assert g["llm"]["class_type"] == LA.NEW and g["llm"]["inputs"]["n_gpu_layers"] == 40 and g["llm"]["inputs"]["n_ctx"] == 12000
    assert g["llm"]["inputs"]["prompt"] == ["prompt", 0] and g["llm"]["inputs"]["image"] == ["sheet", 0]
    assert g["prompt"]["inputs"]["value"] == "design this"
    assert [p for p in LA.check(g) if "Design Package" not in p] == []      # not an app graph: its only 'problem' is that the sheet is a LoadImage
    assert {n["class_type"] for n in g.values()} == {"PrimitiveStringMultiline", "LoadImage", LA.NEW, "PreviewAny"}


def test_baseline_probe_graph_matches_the_app_design_node():
    g = LA.baseline_probe_graph("p")
    n = g["llm"]["inputs"]
    assert g["llm"]["class_type"] == LA.OLD and n["quantization"] == "4-bit (VRAM-friendly)" and n["model_name"] == "design_llm_8b"
    assert n["max_tokens"] == 4096 and n["temperature"] == 0.7 and n["top_p"] == 0.9 and n["repetition_penalty"] == 1.05
    if BASES:                                                   # same settings as the real design node of the app graph
        real = _read(BASES[0] + ".json")["design_qwen"]["inputs"]
        for k in ("model_name", "quantization", "max_tokens", "temperature", "top_p", "repetition_penalty", "num_beams", "attention_mode"):
            assert n[k] == real[k], k


# --------------------------------------------------------------------------------------------- graph files on disk (written by `python tools/app/llm_arms.py`)
@pytest.mark.parametrize("arm", sorted(LA.ARMS))
@pytest.mark.parametrize("b", [b for b in LA.FILE_BASES if os.path.isfile(os.path.join(GRAPHS, b + ".json"))])
def test_graph_files_on_disk_match_the_builder(b, arm):
    path = os.path.join(GRAPHS, LA.graph_name(b, arm) + ".json")
    if not os.path.isfile(path):
        pytest.skip(f"{os.path.basename(path)} not written yet")
    g0, s0 = _read(b + ".json"), _read(b + ".app.json")
    g, s = LA.convert(g0, s0, arm)
    assert _read(os.path.basename(path)) == g, "stale: run python tools/app/llm_arms.py"
    assert _read(LA.graph_name(b, arm) + ".app.json") == s
    assert LA.check(g, arm, g0) == []


# --------------------------------------------------------------------------------------------- runner helpers
def test_list_ggufs_and_mmproj(tmp_path):
    for rel in ("m/a-Q4.gguf", "m/mmproj-F16.gguf", "m/mmproj-BF16.gguf", "m/mmproj-Q8_0.gguf", "m/MTP/mtp-a.gguf", "n/b.gguf", "n/imatrix_x.gguf", "n/mtp-b.gguf"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert LG.list_ggufs(tmp_path) == ["m/a-Q4.gguf", "n/b.gguf"]
    assert os.path.basename(LG.mmproj_for(tmp_path / "m" / "a-Q4.gguf")) == "mmproj-F16.gguf"
    assert LG.mmproj_for(tmp_path / "n" / "b.gguf") is None
    assert LG.list_ggufs(tmp_path / "nope") == []


def test_load_kwargs_kv_flash_and_projector():
    kw = LG.load_kwargs("/m/x.gguf", "/m/mmproj-F16.gguf", n_gpu_layers=41, n_ctx=16384, kv="q8_0", n_cpu_moe=2, mmproj_gpu=False, image_max_tokens=1024)
    assert kw["n_gpu_layers"] == 41 and kw["type_k"] == kw["type_v"] == 8 and kw["flash_attn_type"] == 1 and kw["n_cpu_moe"] == 2
    assert kw["mmproj_path"] == "/m/mmproj-F16.gguf" and kw["chat_handler_kwargs"]["use_gpu"] is False and kw["chat_handler_kwargs"]["image_max_tokens"] == 1024
    assert kw["chat_handler_kwargs"]["extra_template_arguments"] == {"enable_thinking": False}     # Qwen3.5+/Gemma 4 templates think by default
    f16 = LG.load_kwargs("/m/x.gguf", None)
    assert "type_k" not in f16 and "mmproj_path" not in f16 and "n_cpu_moe" not in f16
    with pytest.raises(ValueError):
        LG.load_kwargs("/m/x.gguf", None, kv="q8_0", flash_attn=False)        # quantised V cache needs flash attention
    with pytest.raises(ValueError):
        LG.kv_type("q3")
    with pytest.raises(ValueError):
        LG.load_kwargs("/m/x.gguf", None, handler="bogus")


def test_messages_image_first_and_system():
    m = LG.messages_for("design it", "QUJD", "be brief")
    assert m[0] == {"role": "system", "content": "be brief"}
    c = m[1]["content"]
    assert c[0]["type"] == "image_url" and c[0]["image_url"]["url"] == "data:image/png;base64,QUJD" and c[1] == {"type": "text", "text": "design it"}
    assert LG.messages_for("hi", None) == [{"role": "user", "content": "hi"}]


def test_parse_offload_and_strip_think():
    assert LG.parse_offload("load_tensors: offloaded 41/49 layers to GPU") == (41, 49)
    assert LG.parse_offload("nothing") is None
    assert LG.strip_think("<think>x\ny</think>\n{\"a\": 1}") == '{"a": 1}'
    assert LG.strip_think('<|channel>thought<channel|>{"a": 1}') == '{"a": 1}'


def test_node_module_and_registration():
    src = open(os.path.join(ROOT, "nodes_fusion", "llm_gguf.py"), encoding="utf-8").read()
    assert "node_id=\"MographGGUFVL\"" in src and "NODES = [MographGGUFVL]" in src
    init = open(os.path.join(ROOT, "nodes_fusion", "__init__.py"), encoding="utf-8").read()
    assert '"llm_gguf"' in init
    # outputs: response (slot 0) then stats (slot 1), as registered in the graph checker
    assert "n_ctx=n_ctx, kv_cache" not in src          # st (the chat stats) already carries n_ctx: a duplicate keyword crashed the first GPU run
    assert LA.ltx_ic.CLASSES[LA.NEW][1] == 2
    for name in ("model", "mmproj", "n_gpu_layers", "n_ctx", "max_tokens", "temperature", "seed", "kv_cache", "mmproj_on_gpu", "n_cpu_moe", "keep_model_loaded", "verbose", "image"):
        assert f'"{name}"' in src, name


def test_design_graph_still_open_models_only(base):
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q6")
    classes = {n["class_type"] for n in g.values()}
    assert not any("API" in c or "Claude" in c for c in classes)


# --------------------------------------------------------------------------------------------- tier arms (16_app/TIERS.md; NOT measured)
TIER_ARMS = sorted(k for k, a in LA.ARMS.items() if a.get("tier"))


def test_tier_arms_have_estimates_and_headroom():
    assert {LA.ARMS[k]["tier"] for k in TIER_ARMS} == set(LA.TIERS) == {16, 24}
    for k in TIER_ARMS:
        a = LA.ARMS[k]
        assert a["measured"] is False and a["vram_est_gib"] > a["gib"], k            # estimates are labelled, and include KV / compute buffer
        assert LA.TIERS[a["tier"]]["vram_gb"] - a["vram_est_gib"] >= 1.0, k          # >= 1 GiB spare on the smallest card of the tier
        assert a["n_ctx"] == LA.TIERS[a["tier"]]["n_ctx"] and a["kv_cache"] == "q8_0", k                              # per tier: the app's design call outgrew 16384
    assert {LA.TIERS[16]["n_ctx"], LA.TIERS[24]["n_ctx"]} == {24576, 32768} and LA.TIERS[16]["n_ctx"] < LA.TIERS[24]["n_ctx"]
    assert LA.ARMS["gemma31b_qat"]["mmproj_on_gpu"] is True and LA.ARMS["gemma31b_qat"]["mmproj_on_gpu"] is True      # fits at 32768: est. 21.8 / 20.4 of 23.99 GiB


@pytest.mark.parametrize("arm", TIER_ARMS)
def test_tier_graph_changes_only_llm_nodes_stills_loaders_and_sidecar(arm):
    b = "fusion_design_qwen_auto_layers"
    if not os.path.isfile(os.path.join(GRAPHS, b + ".json")):
        pytest.skip("layers base graph not built")
    g0, s0 = _read(b + ".json"), _read(b + ".app.json")
    g, s = LA.convert(g0, s0, arm)
    t = LA.TIERS[LA.ARMS[arm]["tier"]]
    assert g["still_unet"]["inputs"]["unet_name"] == "moss_image_sft_bf16.safetensors"
    assert g["still_clip"]["inputs"]["clip_name"] == t["loaders"]["still_clip"]["clip_name"] != g0["still_clip"]["inputs"]["clip_name"]
    for nid in ("1", "2", "3", "4", "5", "a.5", "t2m_unet", "vda"):                    # video, music, depth loaders are the same files on every tier
        assert g[nid] == g0[nid], nid
    assert LA.check(g, arm, g0) == []
    q = [f for f in s["fields"] if f["id"] == "quality"][0]
    assert q["default"] == "final" and [o["value"] for o in q["options"]] == ["still", "final"]       # the app shows two choices: Preview (held stills) and Final (= still_preview.APP_FINAL_TIER, single stage)
    assert s["tier"]["measured"] is False and s["server_flags"]["comfyui"] == t["flags"]
    assert any("0.975" in str(n["inputs"].get("sigmas", "")) for n in g.values())      # the V2V start sigma of the text pass stays 0.975
    assert all(n["inputs"]["n_ctx"] == t["n_ctx"] for n in g.values() if n["class_type"] in (LA.NEW, LA.READ))      # every llama.cpp node of the tier, concept + repairs + the legibility read
    assert any(n["class_type"] == LA.NEW for n in g.values()) and LA.ctx_for(LA.ARMS[arm], False) == t["n_ctx"]


@pytest.mark.skipif(not os.path.exists(os.path.join(GRAPHS, "fusion_design_qwen_auto_layers.json")),
                    reason="dev fixture not shipped in the public export: app/graphs/fusion_design_qwen_auto_layers.json")
@pytest.mark.parametrize("arm", TIER_ARMS)
def test_tier_graph_files_on_disk_match_the_builder(arm):
    b = "fusion_design_qwen_auto_layers"
    path = os.path.join(GRAPHS, LA.graph_name(b, arm) + ".json")
    if not os.path.isfile(path):
        pytest.skip("not written yet")
    g, s = LA.convert(_read(b + ".json"), _read(b + ".app.json"), arm)
    assert _read(os.path.basename(path)) == g and _read(LA.graph_name(b, arm) + ".app.json") == s, "stale: python tools/app/llm_arms.py build --arm %s --base %s" % (arm, b)


# --------------------------------------------------------------------------------------------- context per tier (par round: the app's design call outgrew 16384)
def test_non_tier_arms_keep_the_measured_context():
    for k, a in LA.ARMS.items():
        if not a.get("tier"):
            assert a["n_ctx"] == 16384 and LA.ctx_for(a, True) == 16384 and LA.ctx_for(a, False) == 16384, k          # the 3060 arms: measured at 16384


@pytest.mark.parametrize("arm", [k for k in TIER_ARMS if LA.ARMS[k]["tier"] == 24])
def test_24gb_graphs_have_32768_everywhere_the_sigmas_and_the_place_stills(arm):
    path = os.path.join(GRAPHS, LA.graph_name("fusion_design_qwen_auto_layers", arm) + ".json")
    if not os.path.isfile(path):
        pytest.skip("not written yet")
    g = _read(os.path.basename(path))
    llm = [n for n in g.values() if n["class_type"] in (LA.NEW, LA.READ)]
    assert len(llm) == 5 and all(n["inputs"]["n_ctx"] == 32768 and n["inputs"]["kv_cache"] == "q8_0" and n["inputs"]["mmproj_on_gpu"] is True for n in llm)
    assert g["lay_sig"]["inputs"]["sigmas"].startswith("0.975,")
    assert g["first_set"]["class_type"] == "MographFirstFrames" and all(f"still{k}_ks" in g for k in range(2, 7))
    assert all(g[f"slot{k}"]["inputs"]["first"] == ["first_set", 0] for k in range(1, 7))


@pytest.mark.parametrize("arm", [k for k in TIER_ARMS if LA.ARMS[k]["tier"] == 16])
def test_16gb_graphs_have_24576(arm):
    path = os.path.join(GRAPHS, LA.graph_name("fusion_design_qwen_auto_layers", arm) + ".json")
    if not os.path.isfile(path):
        pytest.skip("not written yet")
    g = _read(os.path.basename(path))
    assert all(n["inputs"]["n_ctx"] == 24576 for n in g.values() if n["class_type"] in (LA.NEW, LA.READ))


def test_check_allows_keep_loaded_only_on_the_two_side_graph_nodes(base):
    """GEMMA RESIDENT: keep_model_loaded true is legal for read_llm / story_llm with the shared config, nowhere else (the app / batch graphs stay false)."""
    from tools.app import media_read_graph as MRG
    from tools.app import story_graph as SG
    _, g0, s0 = base
    g, _ = LA.convert(g0, s0, "gemma12b_q6")
    dsg = next(n for n in LA.llm_nodes(g0) if "image" in g0[n]["inputs"])
    b = copy.deepcopy(g)
    b[dsg]["inputs"]["keep_model_loaded"] = True
    assert any("keep_model_loaded" in p for p in LA.check(b))                                    # an app node: still refused
    for gr, nid in ((MRG.build(), "read_llm"), (SG.build(), "story_llm")):
        ins = gr[nid]["inputs"]
        assert ins["keep_model_loaded"] is True and ins["n_ctx"] == LA.RESIDENT_N_CTX and ins["mmproj"] != "none"
        assert not LA._resident_ok("llm_design", ins)                                              # right config, wrong node id
        assert LA._resident_ok(nid, ins)
        assert not LA._resident_ok(nid, dict(ins, mmproj="none")) and not LA._resident_ok(nid, dict(ins, n_ctx=8192))
    r, s = MRG.build()["read_llm"]["inputs"], SG.build()["story_llm"]["inputs"]
    shared = ("model", "mmproj", "n_gpu_layers", "n_ctx", "kv_cache", "flash_attn", "mmproj_on_gpu", "n_cpu_moe", "handler", "image_max_tokens", "keep_model_loaded", "verbose")
    assert {k: r[k] for k in shared} == {k: s[k] for k in shared}                                  # ONE load config: a story after a read reuses the runner
    assert LA.RESIDENT_N_CTX == max(MRG.READ_N_CTX, SG.STORY_N_CTX) == 12288
