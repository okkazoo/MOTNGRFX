"""SAME-SUBJECT NEW-ANGLE STILLS in the app graph (tools/app/build_graphs.add_angle_chain, tools/fusion/places.angle_*, nodes_fusion/inputs.py MographFirstFrames /
MographStillPrompt / MographAngleRef, tools/fusion/angle_still.py; 16_app/ANGLE_STILLS.md) and the CPU-free static validation of the new nodes against graphs the box ran
(tools/app/graph_signatures.py). No model, no GPU. Run: python -m pytest tests/test_angle_graph.py -q"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.app import build_graphs as BG  # noqa: E402
from tools.app import graph_signatures as GS  # noqa: E402
from tools.fusion import angle_still as AS  # noqa: E402
from tools.fusion import places as PL  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402

STILL_A = "A woman in a red vest paddles a kayak across a calm lake, warm golden light, mist on the water, cinematic 35mm"


def piece(**kw):
    p = {"stills": [{"id": "A", "prompt": STILL_A, "seed": 1}, {"id": "B", "angle_of": "A", "framing": "close up", "seed": 2, "prompt": "fallback b"},
                    {"id": "C", "prompt": "a red lighthouse on a cliff", "seed": 3}, {"id": "D", "angle_of": "C", "framing": "low", "seed": 4, "prompt": "fallback d"}],
         "shots": [{"id": f"s{i + 1}", "first": f} for i, f in enumerate("ABCD")]}
    p.update(kw)
    return p


@pytest.fixture(scope="module")
def graph():
    return BG.build_graph(False, "moss", True, True)


# ============================================================================================ pure: which places are angle places
def test_angle_source_rules():
    p = piece()
    assert [PL.angle_source(p, i) for i in range(4)] == [None, 0, None, 2]
    assert PL.angle_sources(p) == {1: 0, 3: 2}
    p2 = piece()
    p2["stills"].append({"id": "E", "angle_of": "B", "framing": "profile"})            # an edit of an edit is refused
    p2["shots"].append({"id": "s5", "first": "E"})
    assert PL.angle_source(p2, 4) is None
    p3 = piece()
    p3["stills"][1]["angle_of"] = "Z"                                                   # an unknown source
    p3["stills"][3]["angle_of"] = "D"                                                   # itself
    assert PL.angle_source(p3, 1) is None and PL.angle_source(p3, 3) is None
    p4 = piece()
    p4["stills"][0]["angle_of"] = "C"                                                   # place A is the first frame (the user's image or still A): never an edit
    assert PL.angle_source(p4, 0) is None
    assert PL.angle_source({"stills": [{"id": "A"}, {"id": "B"}]}, 1) is None


def test_angle_prompt_is_the_anchored_edit_prompt_of_the_source_still():
    p = piece()
    ap = PL.angle_prompt(p, 1)
    assert ap.startswith("Change only the camera: " + AS.FRAMING_CLAUSES["close-up"])
    assert "red vest" in ap and "golden light" in ap and "the colour grade" in ap
    assert AS.FACE_HIDDEN in ap                                                          # close-up on a person: face hidden
    assert "fallback b" not in ap and "35mm" not in ap                                   # the fallback prompt and the style words are not used
    lh = PL.angle_prompt(p, 3)
    assert AS.FACE_HIDDEN not in lh and "lighthouse" in lh                               # not a person: no face line
    assert PL.angle_prompt(p, 0) == "" and PL.angle_prompt(p, 2) == ""                   # not angle places
    q = piece()
    q["stills"][1].update(subject="dark long-sleeve top, red life vest", hide_face=False, keep="the red kayak")
    qa = PL.angle_prompt(q, 1)
    assert "dark long-sleeve top, red life vest" in qa and "the red kayak" in qa and AS.FACE_HIDDEN not in qa


# ============================================================================================ the graph
def test_every_place_has_a_lazy_angle_chain_next_to_its_fresh_chain(graph):
    g = graph
    assert g["ang_lora"]["class_type"] == "LoraLoaderModelOnly" and g["ang_lora"]["inputs"] == {"model": ["still_unet", 0], "lora_name": "moss_image_edit_chinoll.safetensors", "strength_model": 1.0}
    fs = g["first_set"]["inputs"]
    assert fs["angles"] is False                                                         # off in the base graph: only the 24 GB tier switches it on
    for k in range(2, 7):
        assert fs[f"angle_{k}"] == [f"ang{k}_dec", 0] and fs[f"still_{k}"] == [f"still{k}_dec", 0]       # both chains are lazy inputs: the node asks for the right one
        assert g[f"ang{k}_prompt"]["class_type"] == "MographStillPrompt" and g[f"ang{k}_prompt"]["inputs"] == {"index": k, "piece": ["plan", 0], "angle": True}
        ref = g[f"ang{k}_ref"]
        assert ref["class_type"] == "MographAngleRef" and ref["inputs"]["first"] == ["first", 0] and ref["inputs"]["index"] == k
        assert ref["inputs"]["width"] == ["sizes", 6] and ref["inputs"]["height"] == ["sizes", 7]
        assert {f"still_{j}": [f"still{j}_dec", 0] for j in range(2, 7) if j != k}.items() <= ref["inputs"].items() and f"still_{k}" not in ref["inputs"]      # the source can be any OTHER place
        for role in ("pos", "neg"):
            n = g[f"ang{k}_{role}"]
            assert n["class_type"] == "TextEncodeMOSSImageEdit" and n["inputs"]["images.image_0"] == [f"ang{k}_ref", 0]       # the ORIGINAL still, the same node for the negative
            assert n["inputs"]["vae"] == ["still_vae", 0] and n["inputs"]["clip"] == ["still_clip", 0] and n["inputs"]["edit_format"] == "Moss-Image-Edit"
        assert g[f"ang{k}_pos"]["inputs"]["prompt"] == [f"ang{k}_prompt", 0] and g[f"ang{k}_neg"]["inputs"]["prompt"] == ["plan", 3]
        ks = g[f"ang{k}_ks"]["inputs"]
        assert ks["model"] == ["ang_lora", 0] and ks["seed"] == [f"ang{k}_prompt", 1] and ks["positive"] == [f"ang{k}_pos", 0] and ks["negative"] == [f"ang{k}_neg", 0]
        assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (30, 4.0, "er_sde", "simple")       # the edit settings, NOT the t2i 25 / 2.5
        assert g[f"ang{k}_dec"]["inputs"]["vae"] == ["still_vae", 0]
        assert g[f"still{k}_ks"]["inputs"]["steps"] == 25                                 # the fresh chain is untouched
    assert not [n for n in g if n.startswith("ang") and g[n]["class_type"].endswith("Loader")]       # no extra loader: the angle chain shares the still model


def test_the_angle_chain_has_the_shape_of_the_proven_probe_graph(graph):
    """angle_still.angle_graph is the box-proven shape (23_moss-angles/v003): the app chain has the same node classes, input names and sampler settings."""
    probe = AS.angle_graph(["ref", 0], "p", [1], prefix="x", loaders={"unet": ["u", 0], "clip": ["c", 0], "vae": ["v", 0]})
    for role, pid in (("pos", "pos"), ("neg", "neg")):
        assert set(graph[f"ang2_{role}"]["inputs"]) == set(probe[pid]["inputs"])
        assert graph[f"ang2_{role}"]["class_type"] == probe[pid]["class_type"]
    for mine, theirs in (("ang_lora", "lora"), ("ang2_lat", "latent"), ("ang2_ks", "ks_00"), ("ang2_dec", "dec_00")):
        assert graph[mine]["class_type"] == probe[theirs]["class_type"] and set(graph[mine]["inputs"]) == set(probe[theirs]["inputs"]), mine
    for key in ("steps", "cfg", "sampler_name", "scheduler", "denoise"):
        assert graph["ang2_ks"]["inputs"][key] == probe["ks_00"]["inputs"][key]
    assert graph["ang_lora"]["inputs"]["lora_name"] == probe["lora"]["inputs"]["lora_name"] == AS.LORA


def test_static_validation_against_graphs_the_server_ran(graph):
    sigs = GS.proven_signatures()
    if "TextEncodeMOSSImageEdit" not in sigs:
        pytest.skip("23_moss-angles/v003/graphs not present")
    assert {"TextEncodeMOSSImageEdit", "LoraLoaderModelOnly", "EmptyFlux2LatentImage", "KSampler", "VAEDecode"} <= set(sigs)
    assert GS.check(graph, sigs, prefixes=("ang",)) == []
    bad = json.loads(json.dumps(graph))
    bad["ang2_pos"]["inputs"]["images"] = ["ang2_ref", 0]                                # a wrong input name is caught
    del bad["ang2_neg"]["inputs"]["vae"]                                                 # a missing proven input too
    bad["ang3_pos"]["inputs"]["edit_format"] = "Moss-Edit"
    msgs = GS.check(bad, sigs, prefixes=("ang",))
    assert len(msgs) == 3 and any("images" in m for m in msgs) and any("vae" in m for m in msgs) and any("edit_format" in m for m in msgs)


def test_sidecar_labels_the_angle_samplers():
    s = BG.build_sidecar(False, "moss", True, True)
    g = BG.build_graph(False, "moss", True, True)
    assert all(g[f"ang{k}_ks"]["_meta"]["title"] in s["stage_labels"] for k in range(2, 7))


# ============================================================================================ last:sK becomes a real angle still
REQ = {"words": "STILL\nSEARCHING\nA LIGHT\nAHEAD", "scene": "a lone runner sprinting through a tunnel, cold blue light, cinematic", "music": "kinetic, 160 bpm",
       "length": 16, "look": "auto", "seed": 5, "quality": "draft", "format": "9:16", "shots": "chain"}


def test_a_chained_plan_gets_angle_places_that_the_graph_can_edit(graph):
    p = PR.plan_piece(dict(REQ), synth_music(160, 16))
    assert not [s for s in p["shots"] if str(s["first"]).startswith("last:")]            # no last-frame chaining
    srcs = PL.angle_sources(p)
    assert srcs and all(j == 0 for j in srcs.values()) and all(i >= 1 for i in srcs)    # the later shots of the place are edits of place A's ORIGINAL still
    for i, j in srcs.items():
        assert f"ang{i + 1}_dec" in graph and f"ang{i + 1}_ref" in graph                  # each has a real angle-still chain in the graph
        pr = PL.angle_prompt(p, i)
        assert pr.startswith("Change only the camera:") and AS.framing_clause(p["stills"][i]["framing"]) in pr
        assert p["stills"][i]["prompt"] not in pr                                         # the edit prompt is the anchored one, not the fresh-still fallback
    assert not PL.angle_source(p, 0)


# ============================================================================================ the nodes (need a ComfyUI checkout)
def _nodes():
    try:
        from test_layered_shots import _comfy
        _comfy()
        from nodes_fusion import inputs as NI
    except pytest.skip.Exception:
        raise
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"nodes_fusion.inputs does not import here: {type(e).__name__}: {e}")
    return NI


def test_first_frames_asks_for_the_angle_input_only_when_the_switch_is_on():
    NI = _nodes()
    p = piece()
    assert NI.MographFirstFrames.wanted(p, True) == {1: "angle_2", 2: "still_3", 3: "angle_4"}
    assert NI.MographFirstFrames.wanted(p, False) == {1: "still_2", 2: "still_3", 3: "still_4"}
    assert NI.MographAngleRef.source_name(p, 2) is None and NI.MographAngleRef.source_name(p, 4) == "still_3"       # place B edits A (= `first`), D edits C
    names = [i.id if hasattr(i, "id") else i.name for i in NI.MographFirstFrames.define_schema().inputs]
    assert {f"angle_{k}" for k in range(2, 7)} <= set(names) and "angles" in names
