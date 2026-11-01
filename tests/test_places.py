"""One still per PLACE in the app graph (tools/fusion/places.py, nodes_fusion/inputs.py MographFirstFrames / MographStillPrompt, nodes_fusion/footage.py MographShotSlot,
tools/app/build_graphs.py add_place_stills, 20_open-model-design/v003/review/PAR.md "Engine limit found"). No model, no GPU, no ComfyUI: the graph is a static JSON, so the
routing is checked by walking it with a synthetic P06-like plan (3 places x 2 shots) and by the pure helpers the nodes call.
Run: python -m pytest tests/test_places.py -q"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.app import build_graphs as BG  # noqa: E402
from tools.app import layered_shots as LS  # noqa: E402
from tools.fusion import places as PL  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402

P06_SHOTS = [
    {"prompt": "A runner sprints toward the camera through a wet cyan tunnel, LED strips streaking past overhead", "camera": "slow dolly in"},
    {"prompt": "The runner keeps sprinting deeper into the tunnel where the light is darker and the floor more reflective", "continues": True},
    {"prompt": "The runner bursts out into an amber dawn street and sprints toward the camera, long shadows on the wet asphalt", "camera": "tracking dolly", "continues": False},
    {"prompt": "The runner keeps sprinting down the amber street, the low sun behind her turning her into a near silhouette", "continues": True},
    {"prompt": "The runner leaps onto a rooftop ledge and sprints toward the camera against an open violet dusk sky", "continues": False},
    {"prompt": "The runner keeps running along the rooftop, the violet sky deeper and the city lights brighter below", "continues": True},
]
REQ = {"words": "STILL\nSEARCHING\nA LIGHT\nAHEAD\nKEEP\nGOING", "scene": "a lone runner sprinting, cinematic", "music": "kinetic, 160 bpm",
       "length": 24, "look": "auto", "seed": 5, "quality": "draft", "format": "9:16", "shots": "cut"}


@pytest.fixture(scope="module")
def p06():
    req = dict(REQ, owners={"owners": [], "shot_plan": P06_SHOTS, "idea": "one runner through three places"})
    return PR.plan_piece(req, synth_music(160, 24))


def _plan(**kw):
    return PR.plan_piece(dict(REQ, **kw), synth_music(160, 24))


# ------------------------------------------------------------------------------------------------ the plan: one still per place
def test_p06_like_plan_is_three_places_of_two_shots(p06):
    """no last-frame chaining: the 2nd shot of a place gets an ANGLE still of its place (angle_of + framing), so every shot has a still of its own"""
    assert [s["first"] for s in p06["shots"]] == ["A", "B", "C", "D", "E", "F"]
    assert [s["id"] for s in p06["stills"]] == list("ABCDEF") and len(p06["shots"]) == 6
    assert [s.get("angle_of") for s in p06["stills"]] == [None, "A", None, "C", None, "E"] and all(s.get("framing") for s in p06["stills"] if s.get("angle_of"))
    assert PL.needed_stills(p06) == [0, 1, 2, 3, 4, 5] and PL.batch_size(p06) == 6 and not any(str(s["first"]).startswith("last:") for s in p06["shots"])


def test_each_new_place_has_a_still_of_its_own_place(p06):
    b, c = p06["stills"][2]["prompt"], p06["stills"][4]["prompt"]
    assert "amber dawn street" in b and "rooftop" in c and "tunnel" not in b + c         # the still shows THE PLACE of its shot, not another view of the first place
    assert "camera:" not in b + c and b.endswith("tall vertical frame") and any(ph in b for _k, ph, _t in __import__("tools.fusion.piece_system", fromlist=["PHOTO"]).PHOTO)
    assert len({s["prompt"] for s in p06["stills"]}) == 6 and len({s["seed"] for s in p06["stills"]}) == 6


def test_place_still_falls_back_to_the_angle_variation_without_a_usable_shot_prompt():
    p = _plan()                                                                          # cut, no concept shot plan: the old behaviour
    assert [s["first"] for s in p["shots"]] == [chr(65 + k) for k in range(len(p["shots"]))]
    assert all(", a " in s["prompt"] and "of the same place" in s["prompt"] for s in p["stills"][1:])
    assert PR.place_still([], 1) == "" and PR.place_still([{}, {"prompt": "too short"}], 1) == ""
    assert PR.place_still([{}, {"prompt": "A runner sprints along a flooded tunnel lit by cold blue strip lights", "camera": "tracking"}], 1, "16:9").endswith("wide horizontal frame")


# ------------------------------------------------------------------------------------------------ routing: which first frame does each shot get
def test_shot_sources_of_the_p06_like_plan(p06):
    src = PL.shot_sources(p06)
    assert [(s["sid"], s["kind"], s["still"], s["prev"]) for s in src] == [
        ("s1", "still", 0, None), ("s2", "still", 1, None), ("s3", "still", 2, None),
        ("s4", "still", 3, None), ("s5", "still", 4, None), ("s6", "still", 5, None)]


@pytest.mark.parametrize("mode", ["single", "chain"])
def test_one_place_plans_need_only_the_first_still(mode):
    p = _plan(shots=mode, legacy_chain=True)
    assert len(p["stills"]) == 1 and PL.needed_stills(p) == [0] and PL.batch_size(p) == 1
    first = np.zeros((4, 3, 3), np.float32)
    assert PL.batch_rows(p, first, {1: np.ones_like(first)}) == [first]                  # a batch of one row: the slot reads exactly the first frame as before
    assert all(PL.pick_row(p, s.get("first"), 1) == 0 for s in p["shots"])


def test_batch_rows_and_pick_row_give_every_place_its_own_still(p06):
    vals = (0.1, 0.5, 0.9, 0.3, 0.7, 0.2)
    first, *others = (np.full((4, 3, 3), v, np.float32) for v in vals)
    rows = PL.batch_rows(p06, first, {k + 1: o for k, o in enumerate(others)})
    assert len(rows) == 6 and rows[0] is first and all(rows[k + 1] is o for k, o in enumerate(others))
    got = [rows[PL.pick_row(p06, s["first"], len(rows))] for s in p06["shots"]]
    assert [g[0, 0, 0] for g in got] == pytest.approx(list(vals))
    assert PL.pick_row(p06, "last:s1", 3) == 0 and PL.pick_row(p06, "Z", 3) == 0 and PL.pick_row(p06, "C", 2) == 0       # nothing else crashes a run: row 0


def test_a_place_no_shot_uses_is_not_generated_and_keeps_the_batch_dense():
    p = {"stills": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "C"}]}
    assert PL.needed_stills(p) == [0, 2] and PL.batch_size(p) == 3
    first, c = np.zeros((2, 2, 3), np.float32), np.ones((2, 2, 3), np.float32)
    rows = PL.batch_rows(p, first, {1: c * 0.5, 2: c})                                   # the unused B is replaced by row 0 even if something delivered it
    assert rows[1] is first and rows[2] is c
    assert PL.needed_stills(dict(p, layout={"s1": {}})) == [0]                           # a shot outside the layout does not count


def test_prev_is_adjacent_flags_a_continuation_of_an_earlier_shot():
    p = {"shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "B"}, {"id": "s3", "first": "last:s1"}]}
    assert PL.prev_is_adjacent(p) is False        # the graph wires slot k's `prev` to shot k-1 only; the planner never writes this


# ------------------------------------------------------------------------------------------------ the graph
@pytest.fixture(scope="module")
def graph():
    return BG.build_graph(False, "moss", True, True)


NEW = {f"still{k}_{x}" for k in range(2, 7) for x in ("prompt", "txt", "lat", "ks", "dec")} | {"first_set", "out_stills"}
ANGLE = {f"ang{k}_{x}" for k in range(2, 7) for x in ("prompt", "ref", "pos", "neg", "lat", "ks", "dec")} | {"ang_lora"}      # the lazy angle-still chains (tests/test_angle_graph.py)
NEW = NEW | ANGLE


def test_graph_has_a_lazy_still_chain_per_place_and_one_set_of_loaders(graph):
    g = graph
    assert NEW <= set(g) and len(NEW) == 27 + 36
    fs = g["first_set"]
    assert fs["class_type"] == "MographFirstFrames" and fs["inputs"]["first"] == ["first", 0] and fs["inputs"]["piece"] == ["plan", 0]
    for k in range(2, 7):
        assert fs["inputs"][f"still_{k}"] == [f"still{k}_dec", 0]                        # each still is a lazy input of the set: only a needed one runs
        assert g[f"still{k}_prompt"]["inputs"] == {"index": k, "piece": ["plan", 0]}
        ks = g[f"still{k}_ks"]["inputs"]
        assert ks["model"] == ["still_unet", 0] and ks["negative"] == ["still_neg", 0] and ks["seed"] == [f"still{k}_prompt", 1]
        assert g[f"still{k}_txt"]["inputs"]["clip"] == ["still_clip", 0] and g[f"still{k}_txt"]["inputs"]["text"] == [f"still{k}_prompt", 0]
        assert g[f"still{k}_dec"]["inputs"]["vae"] == ["still_vae", 0]
        for key in ("steps", "cfg", "sampler_name", "scheduler", "denoise"):                # the same sampler settings as the first still
            assert ks[key] == g["still_ks"]["inputs"][key]
    assert not [n for n in NEW if g[n]["class_type"].endswith("Loader")]                 # no loader of their own: no extra model load for the places
    assert sum(1 for v in g.values() if v["inputs"].get("unet_name") == BG.MOSS_UNET) == 1
    assert sum(1 for v in g.values() if v["inputs"].get("clip_name") == BG.MOSS_CLIP) == 1


def test_graph_slots_read_the_set_and_the_first_still_chain_is_unchanged(graph):
    g = graph
    for k in range(1, 7):
        assert g[f"slot{k}"]["inputs"]["first"] == ["first_set", 0]
    assert g["first"]["class_type"] == "MographFirstFrame" and g["first"]["inputs"]["generated"] == ["still_dec", 0]      # row 0 = the user's image / still A, as before
    assert g["still_ks"]["inputs"]["seed"] == ["seed", 0] and g["out_still"]["inputs"]["images"] == ["first", 0]
    assert g["gather"]["inputs"]["first"] == ["first_set", 0]                            # the gather holds each shot's own still in the Still preview (one row per place)
    others = [n for n in g if n not in NEW]
    old = copy.deepcopy({n: g[n] for n in others})
    for k in range(1, 7):
        old[f"slot{k}"]["inputs"]["first"] = ["first", 0]
    old["gather"]["inputs"]["first"] = ["first", 0]                                      # the gather's `first` is the set too (the Still preview holds each shot's own still)
    assert g["archive"]["inputs"]["places"] == ["first_set", 0]                          # MographArchive writes <stem>_still_<k>.png per place (INGREDIENTS)
    assert not [n for n in old if n != "archive" and any(isinstance(v, list) and len(v) == 2 and v[0] in NEW for v in old[n]["inputs"].values())]   # nothing outside the new nodes depends on them but the slots (via first_set) and the archive


def test_only_first_set_pulls_the_extra_stills(graph):
    users = {}
    for nid, n in graph.items():
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and v[0] in NEW and not (nid in NEW):
                users.setdefault(v[0], set()).add(nid)
    assert users == {"first_set": {f"slot{k}" for k in range(1, 7)} | {"archive", "gather"}}


def test_the_cpu_draft_graph_is_gone():
    with pytest.raises(ValueError):
        BG.build_graph(True)
    with pytest.raises(ValueError):
        BG.build_sidecar(True)


def test_sidecar_labels_the_place_samplers():
    s = BG.build_sidecar(False, "moss", True, True)
    assert s["stage_labels"]["Still sampler [place 3]"] and s["stage_labels"]["First frames (one per place)"]
    g = BG.build_graph(False, "moss", True, True)
    assert all(g[f"still{k}_ks"]["_meta"]["title"] in s["stage_labels"] for k in range(2, 7))


# ------------------------------------------------------------------------------------------------ the layered route: the right first frame
def _first_source(g, piece, shot):
    """walk the graph: where does the first frame of this shot's CLEAN chain come from? -> ('still', node id of the decode) | ('clean_clip', node id of the previous clean clip)."""
    k = int(shot["id"][1:])
    slot = g[f"slot{k}"]["inputs"]
    assert g[f"l{k}.24"]["inputs"]["input"] == [f"slot{k}", 4] and g[f"t{k}.24"]["inputs"]["input"] == [f"slot{k}", 4]     # clean AND text chain start from the slot's first frame
    if shot["first"].startswith("last:"):
        return "clean_clip", slot["prev"][0]
    assert slot["first"] == ["first_set", 0]
    n = len([k for k in g["first_set"]["inputs"] if k.startswith("still_")])            # the fresh-still inputs (the angle_k inputs and the `angles` switch are tested in test_angle_graph.py)
    row = PL.pick_row(piece, shot["first"], PL.batch_size(piece))
    node = g["first_set"]["inputs"]["first" if row == 0 else f"still_{row + 1}"][0]
    assert n == 5 and row < 6
    return "still", node


def test_layered_graph_starts_every_shot_from_its_own_place_or_the_clean_last_frame(graph, p06):
    side = BG.build_sidecar(False, "moss", True, True)
    g, _ = LS.convert(graph, side)
    assert LS.check(g) == []
    got = {s["id"]: _first_source(g, p06, s) for s in p06["shots"]}
    assert got == {"s1": ("still", "first"), "s2": ("still", "still2_dec"), "s3": ("still", "still3_dec"),
                   "s4": ("still", "still4_dec"), "s5": ("still", "still5_dec"), "s6": ("still", "still6_dec")}            # no last-frame chaining: every shot has a still of its own
    assert all(not v[1].startswith("t") for v in got.values())                           # never the text-pass clip (letters in the picture): the continuation reads the CLEAN clip
    assert g["first_set"]["inputs"]["still_2"] == ["still2_dec", 0] and "first_set" in g and LS.APP_CLASSES_KNOWN


def test_single_place_plan_reads_row_zero_in_the_layered_graph(graph):
    p = _plan(shots="chain", legacy_chain=True)
    g, _ = LS.convert(graph, BG.build_sidecar(False, "moss", True, True))
    got = [_first_source(g, p, s) for s in p["shots"]]
    assert got[0] == ("still", "first") and all(v[0] == "clean_clip" for v in got[1:])
    assert [v[1] for v in got[1:]] == [f"l{k}.dec" for k in range(1, len(got))]


# ------------------------------------------------------------------------------------------------ the user's images 2..5 replace the stills of shots 2..5 (16_app/MEDIA_BOX.md)
def test_user_images_replace_the_stills_of_shots_2_to_n(p06):
    assert PL.user_image_rows(p06, 0) == {}
    assert PL.user_image_rows(p06, 1) == {1: 0}                              # image 2 = shot 2 = still B (place index 1)
    assert PL.user_image_rows(p06, 4) == {1: 0, 2: 1, 3: 2, 4: 3}            # images 2..5 = shots 2..5 = stills B..E; still F stays generated
    assert PL.user_image_rows(p06, 9) == PL.user_image_rows(p06, 4)          # at most 4 user images (images 2..5)


def test_user_images_skip_shots_without_a_still_of_their_own():
    p = _plan(shots="chain", legacy_chain=True)                               # legacy chain: every shot continues the previous one's last frame
    assert PL.user_image_rows(p, 4) == {}
    p1 = _plan(shots="cut")
    assert PL.user_image_rows({"stills": [{"id": "A"}], "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "A"}]}, 3) == {}      # shot 2 starts again from place A: no still of its own
    assert set(PL.user_image_rows(p1, 5)) <= set(range(1, 6))


def _comfy_inputs():
    from test_concept import _comfy
    _comfy()
    from nodes_fusion import inputs as I
    return I


def test_first_frames_node_uses_the_user_images_and_does_not_ask_for_those_chains(p06):
    I = _comfy_inputs()
    import torch
    inp = {i.id: i for i in I.MographFirstFrames.define_schema().inputs}
    assert "user_images" in inp and getattr(inp["user_images"], "optional", False) and not getattr(inp["user_images"], "lazy", None)
    assert I.MographFirstFrames.wanted(p06, False, 0) == {i: f"still_{i + 1}" for i in range(1, 6)}
    assert I.MographFirstFrames.wanted(p06, False, 2) == {i: f"still_{i + 1}" for i in range(3, 6)}      # stills B and C are the user's: their chains never run
    assert I.MographFirstFrames.wanted(p06, True, 4) == {5: "still_6"}      # place F is an angle edit of E, which is the user's image now: a fresh still instead (angle_fallback)
    assert I.MographFirstFrames.wanted(p06, True, 0)[5] == "angle_6"
    w, h = I.shot_wh(p06, {})
    first = torch.full((1, h, w, 3), 0.5)
    users = torch.stack([torch.full((h, w, 3), v) for v in (0.1, 0.2)])
    stills = {f"still_{k}": torch.full((1, h, w, 3), 0.9) for k in range(2, 7)}
    out = I.MographFirstFrames.execute(p06, {}, first, users, False, **stills).args[0]
    assert out.shape[0] == 6
    assert [round(float(out[i].mean()), 2) for i in range(6)] == [0.5, 0.1, 0.2, 0.9, 0.9, 0.9]          # rows: first, user 1, user 2, generated ...
    none = I.MographFirstFrames.execute(p06, {}, first, None, False, **stills).args[0]
    assert [round(float(none[i].mean()), 2) for i in range(6)] == [0.5, 0.9, 0.9, 0.9, 0.9, 0.9]          # no user images = exactly as before


def test_user_images_node_loads_a_batch_at_the_shot_size(tmp_path, monkeypatch):
    I = _comfy_inputs()
    import folder_paths
    from PIL import Image
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda n: str(tmp_path / n))
    for n, c in (("a.png", (255, 0, 0)), ("b.jpg", (0, 255, 0))):
        Image.new("RGB", (300, 200), c).save(tmp_path / n)
    U = I.MographUserImages
    assert U.execute("").args[0] is None and U.execute(" , ").args[0] is None
    out = U.execute("a.png, b.jpg", "9:16").args[0]
    assert tuple(out.shape) == (2, 2048, 1152, 3) and float(out[0, 0, 0, 0]) > 0.9 and float(out[1, 0, 0, 1]) > 0.9
    assert tuple(U.execute("a.png", "16:9").args[0].shape) == (1, 1152, 2048, 3)
    assert U.execute("a.png,b.jpg,a.png,b.jpg,a.png,b.jpg").args[0].shape[0] == 4                          # images 2..5 only
    with pytest.raises(FileNotFoundError):
        U.execute("nope.png")


def test_graph_first_set_gets_the_user_images(graph):
    assert graph["first_set"]["inputs"]["user_images"] == ["user_images", 0]
    assert graph["user_images"]["class_type"] == "MographUserImages"


# ------------------------------------------------------------------------------------------------ STORYBOARD v12: 24 s + 5 pictures = 5 shots, shot k = picture k, no generated still
def test_story_five_pictures_make_five_shots_each_with_its_own_picture():
    p = _plan(length=24, words="A\nB\nC\nD\nE", owners={"owners": [], "story_shots": 5})
    assert len(p["shots"]) == 5 and [s["first"] for s in p["shots"]] == ["A", "B", "C", "D", "E"]
    assert PL.user_image_rows(p, 4) == {1: 0, 2: 1, 3: 2, 4: 3}                  # pictures 2..5 = shots 2..5; picture 1 is IN image = the first frame
    I = _comfy_inputs()
    assert I.MographFirstFrames.wanted(p, False, 4) == {}                        # no generated still chain runs: every shot has a picture
    assert I.MographFirstFrames.wanted(p, True, 4) == {}
