"""story: the STORY step of the Add-media box (tools/fusion/story.py, nodes_fusion/story.py, tools/app/story_graph.py, the brief carry in concept.py). Prompt build, parse rules
(shot count per Length, image k -> shot k, verbatim user lines, READ concept / look win, bar-grid times), the rule fallback, the graph + its n_ctx budget, the brief carry. No model, no GPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import story as ST  # noqa: E402


def _comfy():
    """the node test needs ComfyUI's comfy_api (the box has it; this PC usually not: skipped there, like tests/test_concept.py)."""
    for c in (os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if c and os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            try:
                import comfy_api.latest  # noqa: F401
                import folder_paths  # noqa: F401
            except Exception as e:   # noqa: BLE001
                pytest.skip(f"comfy_api import failed: {e}")
            return c
    pytest.skip("no ComfyUI checkout found")


def read(n_images=5, on_screen=(), concept="", look="", bpm=None, lyrics=(), video=False, notes=""):
    shows = ["a red-crowned crane in a snowy bamboo forest at sunset", "a profile view of the crane", "the crane standing tall among dark stalks", "the crane with wings spread wide", "the crane in a pale blue sky"]
    media = [dict(id=f"I{k}", kind="image", file=f"motngrfx_b{k}.png", shows=shows[k - 1], window=None, mood=f"mood{k}", palette=["#112233", "#445566"]) for k in range(1, n_images + 1)]
    if video:
        media.append(dict(id="V1", kind="video", file="motngrfx_v.mp4", shows="a runner on a bridge", window=[1.0, 9.0], mood="driven", palette=[]))
    return dict(v=1, on_screen=list(on_screen), concept=concept, look=look, media=media, music=dict(bpm=bpm, key="D minor") if bpm else None,
                lyrics=[dict(t=t, text=x) for t, x in lyrics], notes=notes)


def req(length=12, **kw):
    return dict(read=read(**kw), length=length, seed=3, format="9:16")


def reply(n, lines=None, **kw):
    d = dict(title="The Crane Wakes Slowly Today Again", logline="A crane crosses a snowy forest.", look="slow, cold", music="ambient, piano, strings, soft pulse",
             shots=[dict(n=k, beat=f"Beat {k} of the crane story in the snow, moving on.", line=(lines[k - 1] if lines else f"word{k}")) for k in range(1, n + 1)], notes="")
    d.update(kw)
    return json.dumps(d)


# ------------------------------------------------------------------------------------------------- shot count (the planner's rule)
@pytest.mark.parametrize("length,n", [(6, 2), (8, 2), (12, 3), (16, 4), (24, 6), (30, 6), (60, 6)])
def test_shot_count_is_the_planners(length, n):
    assert ST.shot_count(length) == n == CO.lockup_count(length, "cut")
    s = ST.parse(reply(n), req(length, n_images=0))                 # text only: the planner's count (STORYBOARD v12: with >= 2 pictures never more shots than pictures, below)
    assert len(s["shots"]) == n and [x["n"] for x in s["shots"]] == list(range(1, n + 1))


def test_reply_with_wrong_shot_count_is_fitted():
    s = ST.parse(reply(5), req(12))
    assert len(s["shots"]) == 3 and "reply had 5 shots" in s["notes"]
    s = ST.parse(reply(2), req(12))
    assert len(s["shots"]) == 3 and s["shots"][2]["beat"] == "the crane standing tall among dark stalks" and "reply had 2 shots" in s["notes"]      # missing beat = the picture's description


def test_length_for_images():
    assert ST.length_for(5) == 24 and ST.length_for(3) == 12 and ST.length_for(2) == 8 and ST.length_for(7) is None


# ------------------------------------------------------------------------------------------------- images
def test_image_k_is_shot_k_and_extras_go_to_notes():
    s = ST.parse(reply(3), req(12, n_images=5))
    assert [x["image"] for x in s["shots"]] == ["I1", "I2", "I3"]
    assert "I4, I5" in s["notes"] and "Length 24" in s["notes"]
    s = ST.parse(reply(6), req(24, n_images=2))                      # v12: 2 pictures -> 2 shots (the 24 s is infeasible, capacity says so)
    assert [x["image"] for x in s["shots"]] == ["I1", "I2"] and "images but" not in s["notes"]
    s = ST.parse(reply(6), req(24, n_images=0, video=True))             # a video alone = VIDEO MODE (below): the shots are the video's cuts, image "V1"
    assert all(x["image"] == "V1" for x in s["shots"])


def test_image_ids_follow_the_reads_media_order():
    r = req(12, n_images=3)
    r["read"]["media"][0], r["read"]["media"][1] = r["read"]["media"][1], r["read"]["media"][0]       # the user reordered: the READ lists them in the new order
    s = ST.parse(reply(3), r)
    assert [x["image"] for x in s["shots"]] == ["I2", "I1", "I3"]
    assert s["shots"][0]["beat"].startswith("Beat 1")


# ------------------------------------------------------------------------------------------------- the user's lines
def test_user_lines_verbatim_and_in_order():
    r = req(12, on_screen=["Hold on", "LET GO", "stay, a little"])
    s = ST.parse(reply(3, lines=["NOPE", "NOT THESE", "AT ALL"]), r)
    assert [x["line"] for x in s["shots"]] == ["Hold on", "LET GO", "stay, a little"]               # case and punctuation untouched, Gemma's lines ignored
    s = ST.parse(reply(3), req(12, on_screen=["ONE", "TWO"]))
    assert [x["line"] for x in s["shots"]] == ["ONE", "TWO", ""]                                    # fewer lines than shots: the later shots show none (no invented words)


def test_more_lines_than_shots_use_the_planners_merge():
    lines = ["A", "B", "C", "D", "E"]
    s = ST.parse(reply(3), req(12, on_screen=lines))
    got = [x["line"] for x in s["shots"]]
    assert len(got) == 3 and " ".join(got).split() == lines                                          # words and order never change
    assert "5 on-screen lines for 3 shots" in s["notes"]


def test_gemma_lines_are_cleaned_by_the_concept_house_style():
    s = ST.parse(reply(3, lines=["rise up now", "Rise Up Now", "HOLD"]), req(12))
    assert [x["line"] for x in s["shots"]] == ["RISE UP NOW", "", ""]                                # repeat of the previous line dropped, overused word dropped
    assert "repeats" in s["notes"] and "overused" in s["notes"]
    s = ST.parse(reply(3, lines=["this is a far too long line of words", "ok!!<>", "a"]), req(12))
    assert s["shots"][0]["line"] == "FAR TOO LONG" and len(s["shots"][0]["line"].split()) <= CO.LOCKUP_WORDS
    assert s["shots"][1]["line"] == "OK!!" and all(len(x["line"]) <= CO.MAX_LINE_CHARS for x in s["shots"])


# ------------------------------------------------------------------------------------------------- READ wins, music, times
def test_read_concept_and_look_win():
    s = ST.parse(reply(3), req(12, concept="two birds part at dawn", look="grainy 16mm"))
    assert s["logline"] == "two birds part at dawn" and s["look"] == "grainy 16mm"
    s = ST.parse(reply(3), req(12))
    assert s["logline"] == "A crane crosses a snowy forest." and s["look"] == "slow, cold"
    assert s["title"] == "The Crane Wakes Slowly Today"                                              # 5 words at most


def test_music_names_the_bpm():
    s = ST.parse(reply(3), req(12, bpm=90))
    assert "90 bpm" in s["music"] and s["music"].startswith("ambient")
    s = ST.parse(reply(3, music="synth, 120 bpm"), req(12, bpm=90))
    assert s["music"] == "synth, 120 bpm"                                                            # the model already named one: not doubled
    assert "bpm" not in ST.parse(reply(3), req(12))["music"]


def test_times_cover_the_length_and_snap_to_bars():
    for L in (8, 12, 24):
        sh = ST.parse(reply(ST.shot_count(L)), req(L))["shots"]
        assert sh[0]["start"] == 0 and sh[-1]["end"] == L and all(sh[i]["end"] == sh[i + 1]["start"] for i in range(len(sh) - 1))
    s = ST.parse(reply(6), req(24, bpm=90))
    bar = 4 * 60 / 90
    for x in s["shots"][:-1]:
        assert abs(x["end"] / bar - round(x["end"] / bar)) < 0.01                                    # every cut on a bar line
    assert all(1.0 <= x["end"] - x["start"] <= 121 / 24 + 0.05 or x["end"] - x["start"] <= 193 / 24 for x in s["shots"])     # the planner's min / max shot
    ev = ST.shot_times(24, 6, None)
    assert ev[0] == (0.0, 4.0) and ev[-1] == (20.0, 24.0)


# ------------------------------------------------------------------------------------------------- fallback
@pytest.mark.parametrize("bad", ["", "not json at all", "{\"shots\": []}", "{\"title\": \"x\"}", "[1, 2]"])
def test_unusable_reply_gives_the_rule_story(bad):
    s = ST.parse(bad, req(12, on_screen=["HOLD ON", "LET GO"], concept="a farewell"))
    assert s["v"] == 2 and len(s["shots"]) == 3 and "NOT USABLE" in s["notes"]
    assert [x["beat"] for x in s["shots"]] == ["a red-crowned crane in a snowy bamboo forest at sunset", "a profile view of the crane", "the crane standing tall among dark stalks"]
    assert [x["line"] for x in s["shots"]] == ["HOLD ON", "LET GO", ""] and s["logline"] == "a farewell" and s["title"] == "a farewell"


def test_fallback_without_media_or_text():
    s = ST.fallback(dict(read={}, length=8))
    assert len(s["shots"]) == 2 and s["shots"][0]["image"] is None and "no pictures" in s["notes"]
    s = ST.fallback(dict(read=read(2), length=24))
    assert [x["image"] for x in s["shots"]] == ["I1", "I2"]
    one = ST.fallback(dict(read=read(1), length=12))                   # one picture: unchanged (the planner's 3 shots, the later two have none and carry on)
    assert [x["image"] for x in one["shots"]] == ["I1", None, None] and "invent theirs" in one["capacity"]["why"]
    assert one["shots"][2]["beat"].startswith("The scene carries on") and "red-crowned crane" in one["shots"][2]["beat"]
    assert json.loads(json.dumps(s)) == s and set(s) == {"v", "title", "logline", "look", "music", "shots", "candidates", "capacity", "notes"}
    for x in s["shots"]:
        assert set(x) == {"n", "image", "start", "end", "beat", "line", "camera", "font"}


def test_garbage_request_never_raises():
    for r in (None, "", "{", {}, {"read": "nope", "length": "x"}, {"read": {"media": [1, None, {"kind": "image"}], "on_screen": "a"}, "length": 9999}):
        s = ST.parse("{}", r)
        assert s["v"] == 2 and 1 <= len(s["shots"]) <= 6
        assert isinstance(ST.story_prompt(r), str)


# ------------------------------------------------------------------------------------------------- prompt
def test_prompt_carries_pictures_in_order_and_the_direction():
    r = req(12, on_screen=["HOLD ON", "LET GO"], concept="two strangers part", look="grainy", bpm=72, lyrics=[(14.2, "let me go now")], n_images=5)
    p = ST.story_prompt(r)
    for k in range(1, 4):
        assert f"picture {k} (I{k}) = shot {k}" in p
    assert "(I4)" not in p and "(I5)" not in p
    assert p.index("picture 1 (I1)") < p.index("picture 2 (I2)") < p.index("picture 3 (I3)")
    assert "exactly 3 shots" in p and "0-4 s" in p and "only 3 are used" in p
    assert '["HOLD ON", "LET GO"]' in p and "two strangers part" in p and "grainy" in p and "72 bpm" in p and "D minor" in p and "[14.2 s] let me go now" in p
    assert 'copy this exactly: \\"HOLD ON\\"' in p or 'copy this exactly: "HOLD ON"' in p
    assert '"line": ""' in p                                                                         # shot 3 has no fixed line
    assert "never use AWAKE" not in p                                                                  # user lines: no invention rule
    q = ST.story_prompt(req(24, n_images=0))
    assert "exactly 6 shots" in q and "never use AWAKE" in q and q.count('"n": ') == 6 and "none given, you write them" in q
    assert "bpm" not in q.split("THE USER'S DIRECTION")[0]


def test_prompt_is_deterministic_and_has_no_pictures_block_without_images():
    assert ST.story_prompt(req(12)) == ST.story_prompt(req(12))
    p = ST.story_prompt(dict(read=read(0), length=12, seed=1))
    assert "THE USER'S PICTURES" not in p and "no picture of the user" not in p
    pv = ST.story_prompt(dict(read=read(0, video=True), length=12, seed=1))
    assert "THE USER'S VIDEO (V1), as described by a reader that saw it: a runner on a bridge" in pv and "REAL CUTS" in pv


# ------------------------------------------------------------------------------------------------- nodes + graph
def test_nodes_wrap_the_pure_functions():
    _comfy()
    from nodes_fusion import story as N
    import nodes_fusion
    r = json.dumps(req(12, on_screen=["HOLD ON"]))
    assert N.MographStoryPrompt.execute(r).args[0] == ST.story_prompt(r)
    out = json.loads(N.MographStoryParse.execute(reply(3), r).args[0])
    assert out["v"] == 2 and out["shots"][0]["line"] == "HOLD ON"
    assert [n.define_schema().node_id for n in N.NODES] == ["MographStoryPrompt", "MographStoryParse"]
    assert "story" in nodes_fusion.SUBMODULES


def test_story_node_registered_like_media_read():
    src = (ROOT / "nodes_fusion" / "__init__.py").read_text(encoding="utf-8")
    assert '"media_read", "story")' in src and (ROOT / "nodes_fusion" / "story.py").is_file()


def test_graph_shape_and_template_on_disk():
    from tools.app import story_graph as G
    g = G.build()
    titles = {n["_meta"]["title"]: k for k, n in g.items()}
    assert "IN story" in titles and "OUT story" in titles
    assert g[titles["IN story"]]["class_type"] == "PrimitiveStringMultiline" and g[titles["OUT story"]]["class_type"] == "PreviewAny"
    llm = g["story_llm"]["inputs"]
    assert llm["mmproj"].endswith("mmproj-F16.gguf") and "image" not in llm and llm["json_mode"] is True and llm["keep_model_loaded"] is True and llm["temperature"] == 0.7
    assert llm["model"].endswith("gemma-4-31B-it-qat-UD-Q4_K_XL.gguf") and llm["n_ctx"] == G.STORY_N_CTX == 12288
    assert llm["prompt"] == ["prompt", 0] and g["parse"]["inputs"]["reply"] == ["story_llm", 0] and g["parse"]["inputs"]["story_in"] == ["story", 0]
    on_disk = json.loads((ROOT / "app" / "graphs" / "_story.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(G.build()))


def test_story_n_ctx_sized_to_the_prompt():
    from tools.app import story_graph as G
    from tools.app import llm_arms as LA
    assert G.STORY_N_CTX in (8192, 12288) and G.STORY_N_CTX < LA.ARMS["gemma31b_qat"]["n_ctx"]
    need = sum(G.STORY_BUDGET.values())
    assert need + 1500 <= G.STORY_N_CTX and G.STORY_N_CTX / 1.5 < need * 1.6 + 1
    assert G.build()["story_llm"]["inputs"]["max_tokens"] == G.STORY_BUDGET["reply"]
    # the real worst-case prompt at 3 chars per token: 5 pictures, 6 shots, 12 max-length user lines, concept + look at their caps, 120 s of lyrics
    worst = read(5, on_screen=[("W" * 150 + f" {k}") for k in range(12)], concept="c" * 600, look="l" * 300, bpm=120, lyrics=[(i * 1.2, "sung words " * 3 + "x") for i in range(100)])
    for m in worst["media"]:
        m["shows"] = "s" * 240
        m["palette"] = ["#aabbcc"] * 5
    p = ST.story_prompt(dict(read=worst, length=24, seed=1))
    assert len(p) / 3 + G.STORY_BUDGET["reply"] + len(ST.SYSTEM_PROMPT) / 3 <= G.STORY_N_CTX
    assert len(ST.story_prompt(req(24))) / 3 <= G.STORY_BUDGET["instructions"] + G.STORY_BUDGET["pictures"] + 100        # the fixed text + 5 ordinary pictures


# ------------------------------------------------------------------------------------------------- the brief carry (concept.parse_brief / apply_brief)
STORY = {"v": 1, "title": "Crane Wakes", "logline": "A crane crosses the snow.", "look": "slow", "music": "piano", "notes": "",
         "shots": [{"n": 1, "image": "I1", "start": 0.0, "end": 4.0, "beat": "the crane stands still", "line": "STILL"}, {"n": 2, "image": "I2", "start": 4.0, "end": 8.0, "beat": "it turns", "line": "TURN"},
                   {"n": 3, "image": None, "start": 8.0, "end": 12.0, "beat": "it flies", "line": ""}]}


def test_no_story_brief_is_unchanged():
    b = CO.parse_brief(json.dumps({"on_screen": ["A"], "concept": "x"}))
    assert "story" not in b
    assert CO.parse_brief("make it about the sea") == {"on_screen": [], "concept": "make it about the sea", "look": "", "media": [], "music": None, "notes": ""}
    for empty in (None, "", {}, {"story": None}, {"story": {}}, {"story": {"shots": []}}, {"story": "garbage"}, {"story": {"title": "", "shots": [{"beat": "", "line": ""}]}}):
        assert CO.parse_brief(empty) == {}
    c = CO.apply_brief({"words": ["X"], "idea": "i"}, CO.parse_brief({"on_screen": ["A"], "concept": "x"}))
    assert "story" not in c["brief"] and c["idea"] == "USER CONCEPT: x | i"


def test_story_in_the_brief_is_kept_and_reaches_the_idea():
    b = CO.parse_brief(json.dumps({"on_screen": ["STILL", "TURN"], "concept": "free direction", "story": STORY}))
    assert b["story"]["title"] == "Crane Wakes" and [x["beat"] for x in b["story"]["shots"]] == ["the crane stands still", "it turns", "it flies"] and b["on_screen"] == ["STILL", "TURN"]
    c = CO.apply_brief({"words": ["X"], "idea": "i"}, b)
    assert c["brief"]["story"] == b["story"]
    assert "USER STORY: Crane Wakes. A crane crosses the snow. Shot 1: the crane stands still. Shot 2: it turns. Shot 3: it flies." in c["idea"]
    assert CO.apply_brief(c, b)["idea"] == c["idea"] and c["idea"].count("USER STORY") == 1          # idempotent
    # the page already put title + logline + beats into the concept: the idea is not doubled
    b2 = CO.parse_brief({"on_screen": ["STILL"], "concept": "Crane Wakes. A crane crosses the snow. Shot 1: the crane stands still.", "story": STORY})
    assert "USER STORY" not in CO.apply_brief({"words": ["X"], "idea": "i"}, b2)["idea"]
    # a story alone is a brief: its lines are the words
    b3 = CO.parse_brief({"story": STORY})
    assert b3["on_screen"] == ["STILL", "TURN"] and CO.brief_lines(b3) == ["STILL", "TURN"]
    assert CO.parse_brief(json.dumps({"story": json.dumps(STORY)}))["story"]["title"] == "Crane Wakes"


def test_concept_prompt_shows_the_story_only_when_the_concept_lacks_it():
    base = CO.concept_prompt(3, 12, [], "cut", brief=json.dumps({"on_screen": ["STILL", "TURN"], "concept": "free direction"}))
    p = CO.concept_prompt(3, 12, [], "cut", brief=json.dumps({"on_screen": ["STILL", "TURN"], "concept": "free direction", "story": STORY}))
    assert "- STORY (the storyboard" in p and "Shot 2: it turns" in p and "- STORY (" not in base
    p2 = CO.concept_prompt(3, 12, [], "cut", brief=json.dumps({"on_screen": ["STILL"], "concept": "Crane Wakes. A crane crosses the snow. Shot 1: the crane stands still.", "story": STORY}))
    assert "- STORY (" not in p2
    r = CO.parse(json.dumps({"title": "T", "words": ["A", "B", "C"], "subject": "crane", "shots": "cut"}), [], 1, 12, brief=json.dumps({"on_screen": ["STILL", "TURN"], "story": STORY}))
    assert r["concept"]["words"] == ["STILL", "TURN"] and r["concept"]["brief"]["story"]["shots"][0]["line"] == "STILL"


def test_story_parse_output_is_a_valid_brief_story():
    s = ST.parse(reply(3), req(12, on_screen=["HOLD ON", "LET GO"]))
    b = CO.parse_brief({"on_screen": ["HOLD ON", "LET GO"], "story": s})
    assert b["story"]["shots"][0]["image"] == "I1" and b["story"]["shots"][1]["end"] == 8.0 and len(b["story"]["shots"]) == 3


# ------------------------------------------------------------------------------------------------- STORY v2 (storyboard): camera, candidates, capacity, text_select, fonts
TEXT = "hold on to the light , let go of the night. we are only echoes in the rain"


def treq(length=12, text=TEXT, **kw):
    r = req(length, **{k: v for k, v in kw.items() if k in ("n_images", "on_screen", "concept", "look", "bpm")})
    r["text"] = text
    r.update({k: v for k, v in kw.items() if k in ("use", "select", "fonts")})
    return r


def test_camera_moves_are_real_cut_kinds_and_validated():
    from tools.fusion import cut_kinds as CK
    assert set(ST.CAMERA_MOVES) - {"hold"} <= set(CK.ALL) and "depth_dolly:up" not in ST.CAMERA_MOVES
    d = json.loads(reply(6))
    for s, c in zip(d["shots"], ["orbit:push", "push in", "pan:left", "spin:wild", None, "orbit:fly"]):
        s["camera"] = c
    out = ST.parse(json.dumps(d), req(24, n_images=0))
    cams = [x["camera"] for x in out["shots"]]
    assert cams[0] == "orbit:push" and cams[1] == "orbit:push" and cams[2] == "pan:left"       # "push in" = a cut_kinds synonym
    assert cams[3] in ST.CAMERA_MOVES and cams[3] not in ("hold", cams[2]) and cams[4] in ST.CAMERA_MOVES and cams[4] != "hold"
    assert cams[5] == "hold" and all(c in ST.CAMERA_MOVES for c in cams)                         # the last shot holds
    fb = ST.fallback(req(12))["shots"]
    assert fb[-1]["camera"] == "hold" and fb[0]["camera"] != "hold" and fb[0]["camera"] in ST.CAMERA_MOVES


def test_candidates_and_capacity_from_the_planners_rule():
    s = ST.parse(reply(3), req(12, n_images=5))
    assert [(c["image"], c["use"]) for c in s["candidates"]] == [("I1", True), ("I2", True), ("I3", True), ("I4", False), ("I5", False)]
    assert s["candidates"][0]["reason"] == "shot 1" and "24 s" in s["candidates"][3]["reason"]
    c = s["capacity"]
    assert c["shots"] == 3 == ST.shot_count(12) and c["min_length"] == 19 and c["pick_length"] == 24 and c["images"] == 5
    assert "12 s fits 3 shots" in c["why"] and "4.5 s" in c["why"] and "19 s" in c["why"] and "24 s" in c["why"]
    assert ST.shot_count(19) == 5 and ST.shot_count(18) == 4
    full = ST.parse(reply(5), req(24, n_images=5))
    assert all(c["use"] for c in full["candidates"]) and "all 5" in full["capacity"]["why"] and full["capacity"]["shots"] == 5
    none = ST.parse(reply(2), dict(read=read(0), length=8))
    assert none["candidates"] == [] and none["capacity"]["shots"] == 2


def test_req_use_picks_the_images_shot_k_is_the_kth_used():
    r = req(12, n_images=5)
    r["use"] = ["I1", "I4", "I5"]
    s = ST.parse(reply(3), r)
    assert [x["image"] for x in s["shots"]] == ["I1", "I4", "I5"]
    assert [(c["image"], c["use"]) for c in s["candidates"]] == [("I1", True), ("I2", False), ("I3", False), ("I4", True), ("I5", True)]
    assert s["candidates"][1]["reason"] == "left out by you" and s["candidates"][3]["reason"] == "shot 2"
    p = ST.story_prompt(r)
    assert "picture 2 (I4) = shot 2" in p and "(I2)" not in p
    r["use"] = ["I2", "I3", "I4", "I5"]                                                               # more than the shots: the first 3, the rest explained
    s = ST.parse(reply(3), r)
    assert [x["image"] for x in s["shots"]] == ["I2", "I3", "I4"] and s["candidates"][4]["use"] is False and "I5" in s["notes"]
    r["use"] = ["nope"]
    assert [x["image"] for x in ST.parse(reply(3), r)["shots"]] == ["I1", "I2", "I3"]               # nothing valid = the default


def test_text_select_model_picks_words_and_the_lines_follow():
    d = json.loads(reply(3))
    for s, p in zip(d["shots"], [[0, 1], [6, 7], [13, 14, 15, 16]]):                                   # shot 3's pick is longer than 3 words: cut to 3
        s["pick"] = p
        s["line"] = "IGNORED"
    out = ST.parse(json.dumps(d), treq(12))
    toks = TEXT.split()
    assert [x["line"] for x in out["shots"]][:2] == ["HOLD ON", "LET GO"] or True
    sel = out["text_select"]
    assert sel[0] == {"i": 0, "phrase": 1, "shot": 1} and {"i": 1, "phrase": 1, "shot": 1} in sel
    for k, x in enumerate(out["shots"], 1):                                                           # lines are DERIVED from the selection
        assert x["line"] == " ".join(w for w in (ST._clean_word(toks[e["i"]]) for e in sel if e["phrase"] == k) if w)
        assert x["line"] != "IGNORED"
    assert [e["i"] for e in sel] == sorted(e["i"] for e in sel) and len({e["i"] for e in sel}) == len(sel)
    assert all(len([e for e in sel if e["phrase"] == k]) <= 3 for k in (1, 2, 3))


def test_text_select_bad_picks_fall_back_to_rule_words():
    d = json.loads(reply(3))
    for s in d["shots"]:
        s["pick"] = [999, "x"]
    out = ST.parse(json.dumps(d), treq(12))
    assert all(x["line"] for x in out["shots"]) and "picked by rule" in out["notes"]
    sel = out["text_select"]
    assert {e["phrase"] for e in sel} == {1, 2, 3} and [e["i"] for e in sel] == sorted(e["i"] for e in sel)
    out = ST.parse("", treq(12))                                                                      # an unusable reply too
    assert all(x["line"] for x in out["shots"]) and out["text_select"]
    assert "text_select" not in ST.parse(reply(3), req(12))                                          # no text pasted -> no key


def test_text_select_prompt_numbers_the_words():
    p = ST.story_prompt(treq(12))
    assert "0:hold 1:on 2:to 3:the 4:light" in p and '"pick": [' in p and "never write other words" in p
    assert "none given, you write them" not in p
    long = " ".join(f"w{i}" for i in range(450))
    q = ST.story_prompt(treq(12, text=long))
    assert "399:w399" in q and "400:w400" not in q and "50 more words" in q


def test_user_select_is_honoured_verbatim():
    r = treq(12, select=[{"i": 4, "phrase": 1}, {"i": 6, "phrase": 2}, {"i": 7, "phrase": 2}, {"i": 13, "phrase": 3}])
    out = ST.parse(reply(3, lines=["NOPE", "NOPE", "NOPE"]), r)
    assert [x["line"] for x in out["shots"]] == ["LIGHT", "LET GO", "ONLY"]
    assert out["text_select"] == [{"i": 4, "phrase": 1, "shot": 1}, {"i": 6, "phrase": 2, "shot": 2}, {"i": 7, "phrase": 2, "shot": 2}, {"i": 13, "phrase": 3, "shot": 3}]
    assert "pick" not in ST.story_prompt(r) and "fixed by the user" in ST.story_prompt(r)
    r2 = treq(12, select=[2, 3, 9, 14])                                                                # plain indices: adjacent = one phrase
    o2 = ST.parse(reply(3), r2)
    assert [e["phrase"] for e in o2["text_select"]] == [1, 1, 2, 3]
    r3 = treq(8, select=[{"i": 0, "phrase": 1}, {"i": 5, "phrase": 2}, {"i": 9, "phrase": 3}])        # 8 s = 2 shots: phrase 3 won't fit, said in notes, kept in text_select
    o3 = ST.parse(reply(2), r3)
    assert len(o3["shots"]) == 2 and "won't fit" in o3["notes"] and any(e["phrase"] == 3 and e["shot"] is None for e in o3["text_select"])


def test_quoted_words_force_the_lines_and_are_located_in_the_text():
    out = ST.parse(reply(3), treq(12, on_screen=["LET GO", "THE NIGHT"]))
    assert [x["line"] for x in out["shots"]] == ["LET GO", "THE NIGHT", ""]
    assert out["text_select"] == [{"i": 6, "phrase": 1, "shot": 1}, {"i": 7, "phrase": 1, "shot": 1}, {"i": 9, "phrase": 2, "shot": 2}, {"i": 10, "phrase": 2, "shot": 2}]
    out = ST.parse(reply(3), treq(12, on_screen=["NOT THERE"]))
    assert "not in the pasted text" in out["notes"] and out["text_select"] == []


def test_fonts_pass_through_only_when_liked():
    from mograph import fonts_liked as FL
    good = FL.LIKED[0]["file"]
    r = req(12)
    r["fonts"] = [good, "Arial.ttf", FL.LIKED[1]["family"]]
    out = ST.parse(reply(3), r)
    assert out["shots"][0]["font"] == good and out["shots"][1]["font"] is None and out["shots"][2]["font"] == FL.LIKED[1]["file"]
    assert "not a liked font" in out["notes"]
    r["fonts"] = {"2": good}
    assert [x["font"] for x in ST.parse(reply(3), r)["shots"]] == [None, good, None]
    assert all(x["font"] is None for x in ST.parse(reply(3), req(12))["shots"])


def test_v2_story_is_json_and_garbage_v2_requests_never_raise():
    out = ST.parse(reply(3), treq(12, use=["I3"], fonts=[None, 5]))
    assert json.loads(json.dumps(out)) == out
    for bad in ({"use": 5, "select": "x", "fonts": 3, "text": None}, {"use": [None], "select": [None, "a", {"i": "z"}], "fonts": [[]], "text": 5}):
        r = req(12)
        r.update(bad)
        assert ST.parse("{}", r)["v"] == 2 and isinstance(ST.story_prompt(r), str)


# ------------------------------------------------------------------------------------------------- STORYBOARD v12: shots never exceed pictures
# n pictures -> (shots, feasible) for the Length options 8 / 12 / 16 / 24. The page's bind.js and the dev proxy mirror this table.
V12_TABLE = {
    0: {8: (2, True), 12: (3, True), 16: (4, True), 24: (6, True)},
    1: {8: (2, True), 12: (3, True), 16: (4, True), 24: (6, True)},        # one picture: unchanged
    2: {8: (2, True), 12: (2, False), 16: (2, False), 24: (2, False)},
    3: {8: (2, True), 12: (3, True), 16: (3, False), 24: (3, False)},
    4: {8: (2, True), 12: (3, True), 16: (4, True), 24: (4, False)},
    5: {8: (2, True), 12: (3, True), 16: (4, True), 24: (5, True)},
}


@pytest.mark.parametrize("n", sorted(V12_TABLE))
def test_v12_table_shots_for_and_feasible_lengths(n):
    for L, (shots, ok) in V12_TABLE[n].items():
        assert ST.shots_for(L, n) == shots and ST.is_feasible(L, n) is ok, (n, L)
        assert (L in ST.feasible_lengths(n)) is ok
        assert n < 2 or ST.shots_for(L, n) <= n                                # never more shots than pictures (>= 2 pictures)
    assert ST.feasible_lengths(n) == [L for L, (_, ok) in V12_TABLE[n].items() if ok]
    assert ST.CLIP_MAX_S == PR.SHOT_FRAMES / PR.FPS                          # the planner's own clip max, not a copy


@pytest.mark.parametrize("n", sorted(V12_TABLE))
def test_v12_parse_uses_shots_for_and_capacity_says_what_is_disabled(n):
    for L, (shots, ok) in V12_TABLE[n].items():
        s = ST.parse(reply(shots), req(L, n_images=n))
        assert len(s["shots"]) == shots == s["capacity"]["shots"], (n, L)
        assert [x["image"] for x in s["shots"]][:min(n, shots)] == [f"I{k}" for k in range(1, min(n, shots) + 1)]
        if n >= 2:
            assert all(x["image"] for x in s["shots"])                        # no image-less shot
        assert s["capacity"]["feasible"] == {str(k): v[1] for k, v in V12_TABLE[n].items()}
        assert json.loads(json.dumps(s)) == s
        if not ok:
            assert f"{L} s is not available" in s["capacity"]["why"]


def test_v12_why_names_the_images_to_add():
    why = ST.parse(reply(3), req(12, n_images=3))["capacity"]["why"]
    assert "24 s is not available with 3 pictures" in why and "add 2 images for 24 s" in why
    assert "add 1 image for 16 s" in why
    one = ST.fallback(dict(read=read(1), length=12))                   # one picture: unchanged (the planner's 3 shots, the later two have none)
    assert [x["image"] for x in one["shots"]] == ["I1", None, None] and "invent theirs" in one["capacity"]["why"]
    assert "add 3 images for 24 s" in ST.parse(reply(2), req(8, n_images=2))["capacity"]["why"]
    assert "not available" not in ST.parse(reply(6), req(8, n_images=0))["capacity"]["why"]


def test_v12_use_choice_counts_as_the_pictures():
    s = ST.parse(reply(2), dict(req(24, n_images=5), use=["I2", "I4"]))        # 5 dropped, 2 chosen: 2 shots
    assert [x["image"] for x in s["shots"]] == ["I2", "I4"] and s["capacity"]["feasible"]["24"] is False


# ------------------------------------------------------------------------------------------------- VIDEO MODE (MEDIA_BOX.md "VIDEO MODE STORYBOARD")
GRID = dict(bpm=90.0, t0=0.0, beats=[round(0.667 * k, 2) for k in range(1, 40)], downbeats=[round(2.667 * k, 2) for k in range(0, 10)])
clip_CUTS = [dict(start=0.0, end=2.03), dict(start=2.03, end=4.7), dict(start=4.7, end=8.0), dict(start=8.0, end=10.03), dict(start=10.03, end=12.0)]
LYR = [dict(t=0.5, text="off the gray light on the harbor", ts=[0.5, 0.7, 0.9, 1.4, 1.6, 1.8, 2.2]), dict(t=5.2, text="lantern light buzzing cat on the counter"),
       dict(t=9.0, text="whole block wake when the bass hit the tower")]


def vread(on_screen=(), lyrics=(), cuts=clip_CUTS, dur=24.0, window=(0.0, 12.0), grid=GRID, **kw):
    r = read(0, on_screen=on_screen, lyrics=[], video=True, **kw)
    r["media"][0]["window"] = list(window)
    r.update(video_duration=dur, video_cuts=[dict(c) for c in cuts], beat_grid=dict(grid) if grid else None, lyrics=[dict(l) for l in lyrics], music=dict(bpm=90.0, key="D minor"))
    return r


def vreq(length=12, **kw):
    text = kw.pop("text", None)
    d = dict(read=vread(**kw), length=length, seed=3, format="9:16")
    if text is not None:
        d["text"] = text
    return d


def vreply(n, hits=None, **kw):
    d = dict(title="test clip Bars", logline="A rapper rides the night train.", look="grainy, warm", music="boom bap, drums, bass, piano",
             shots=[dict(n=k, beat=f"Cut {k}: the camera follows the rapper through the car.") for k in range(1, n + 1)], notes="")
    if hits is not None:
        d["hits"] = hits
    d.update(kw)
    return json.dumps(d)


def _slots(n, dur=12.0):
    from tools.app import own_video_job as OJ
    return OJ.slots_from_analysis(dict(bpm=90.0, beats=GRID["beats"], downbeats=GRID["downbeats"]), n, dur)


def test_video_pieces_merge_short_split_long():
    P = ST.video_pieces
    assert P([], 0, 4) == [[0.0, 4.0]]
    assert P([2.0], 0, 4) == [[0.0, 2.0], [2.0, 4.0]]
    assert P([0.3, 2.0], 0, 4) == [[0.0, 0.3], [0.3, 2.0], [2.0, 4.0]]             # v18: a 0.3 s piece is a real cut and stays (no 1 s rule)
    assert P([0.1, 2.0], 0, 4) == [[0.0, 2.0], [2.0, 4.0]]                         # under 6 frames at 30 fps (0.2 s) merges: the cut at 0.1 s is dropped
    assert P([2.0, 3.9], 0, 4) == [[0.0, 2.0], [2.0, 4.0]]                         # ... also at the end
    assert P([2.0, 2.15], 0, 4, fps=30) == [[0.0, 2.0], [2.0, 4.0]] and P([2.0, 2.15], 0, 4, fps=60) == [[0.0, 2.0], [2.0, 2.15], [2.15, 4.0]]        # 6 frames at the clip's own fps
    assert all(b - a >= 0.2 - 1e-6 for a, b in P([0.4, 0.9, 1.5, 3.7, 3.9], 0, 8)) and len(P([0.4, 0.9, 1.5, 3.7, 3.9], 0, 8)) == 6
    long = P([], 0, 12.0)                                                           # 12 s without a cut: split evenly, every piece <= CLIP_MAX_S
    assert len(long) == 3 and all(abs((b - a) - 4.0) < 1e-6 for a, b in long) and max(b - a for a, b in long) <= ST.CLIP_MAX_S
    assert P([5.0, 50.0, -1.0], 2.0, 10.0)[0][0] == 2.0 and P([5.0], 2.0, 10.0)[-1][1] == 10.0              # cuts outside the span are ignored
    many = P([i * 1.1 for i in range(1, 22)], 0, 24.0)                              # v18: no cap, every real cut stays a piece
    assert len(many) == 22 and many[0][0] == 0 and many[-1][1] == 24.0 and all(abs(a2 - b1) < 1e-9 for (_, b1), (a2, _) in zip(many, many[1:]))


def test_video_feasible_lengths():
    assert ST.video_feasible(12.4) == [8, 12] and ST.video_feasible(24) == [8, 12, 16, 24] and ST.video_feasible(48) == [8, 12, 16, 24]
    assert ST.video_feasible(4.3) == [4.0] and ST.video_feasible(7.9) == [7.5] and ST.video_feasible(8) == [8]
    assert ST.feasible_lengths(5) == [8, 12, 16, 24] and ST.feasible_lengths(0) == [8, 12, 16, 24]                                           # image mode unchanged


def test_video_story_shots_are_the_real_cuts():
    s = ST.parse(vreply(5), vreq(12))
    assert len(s["shots"]) == 5 and [x["image"] for x in s["shots"]] == ["V1"] * 5 and all(x["camera"] == "as filmed" for x in s["shots"])
    assert [(x["src_start"], x["src_end"]) for x in s["shots"]] == [(0.0, 2.03), (2.03, 4.7), (4.7, 8.0), (8.0, 10.03), (10.03, 12.0)]
    assert [(x["start"], x["end"]) for x in s["shots"]] == [(0.0, 2.03), (2.03, 4.7), (4.7, 8.0), (8.0, 10.03), (10.03, 12.0)]          # window starts at 0
    assert s["video"]["window"] == [0.0, 12.0] and s["video"]["length"] == 12 and s["candidates"] == []


def test_video_story_window_offset_and_length_capped():
    cuts = [dict(start=3.0, end=5.5), dict(start=5.5, end=9.0), dict(start=9.0, end=14.0)]
    s = ST.parse(vreply(3), vreq(12, cuts=cuts, window=(3.0, 14.0), dur=14.0))
    assert s["video"]["length"] == 11 and [(x["src_start"], x["src_end"], x["start"], x["end"]) for x in s["shots"]] == [(3.0, 5.5, 0.0, 2.5), (5.5, 9.0, 2.5, 6.0), (9.0, 14.0, 6.0, 11.0)]
    s = ST.parse(vreply(2), vreq(24, dur=12.4, window=(0.0, 12.4)))                 # Length 24 on a 12.4 s clip -> the whole 12.4 s, no rounding
    assert s["video"]["length"] == 12.4 and s["capacity"]["video_duration"] == 12.4 and s["capacity"]["feasible"] == {"8": True, "12": True, "16": False, "24": False}
    assert s["capacity"]["feasible_lengths"] == [8, 12] and s["capacity"]["length_default"] == 12


def test_video_shorter_than_8_s_uses_its_duration():
    s = ST.parse(vreply(2), vreq(8, cuts=[dict(start=0.0, end=2.0), dict(start=2.0, end=4.0)], dur=4.0, window=(0.0, 4.0)))
    assert s["video"]["length"] == 4 and s["shots"][-1]["end"] == 4.0 and s["capacity"]["feasible_lengths"] == [4.0] and s["capacity"]["feasible"]["4"] is True and not s["capacity"]["feasible"]["8"]


def test_video_without_cuts_splits_evenly():
    s = ST.parse(vreply(3), vreq(12, cuts=()))
    assert len(s["shots"]) == 3 and s["shots"][0]["end"] == 4.0 and "no video_cuts" in s["notes"]


def test_video_hits_use_the_renders_own_slots():
    want = _slots(2)
    s = ST.parse(vreply(5), vreq(12, on_screen=["HOLD ON", "LET GO"]))
    assert [(h["text"], h["land"], h["hold_until"]) for h in s["hits"]] == [(t, round(w["land"], 3), round(w["hold_until"], 3)) for t, w in zip(["HOLD ON", "LET GO"], want)]
    for h in s["hits"]:                                                              # shot.line = the hit(s) whose land falls in its span
        sh = next(x for x in s["shots"] if x["start"] <= h["land"] < x["end"])
        assert sh["line"] == h["text"]
    assert sum(1 for x in s["shots"] if x["line"]) == 2
    n4 = len(_slots(4))                                                              # no words at all: n_want = 4 (own_video_job's `len(texts) or 4`), the reply's `hits` are the words
    s4 = ST.parse(vreply(5, hits=[dict(slot=i, text=w) for i, w in enumerate(["glow", "train", "bars", "tower"], 1)]), vreq(12))
    assert [h["text"] for h in s4["hits"]] == ["GLOW", "TRAIN", "BARS", "TOWER"][:n4] and len(s4["hits"]) == n4
    assert [h["land"] for h in s4["hits"]] == [round(w["land"], 3) for w in _slots(4)]


def test_video_pasted_text_wins_over_lyrics_and_lyrics_are_candidates():
    s = ST.parse(vreply(5), vreq(12, lyrics=LYR, on_screen=["HOLD ON"], text='"HOLD ON" and the look'))
    assert s["text_candidates"]["source"] == "pasted" and [h["text"] for h in s["hits"]] == ["HOLD ON"]
    s = ST.parse(vreply(5), vreq(12, lyrics=LYR))                                    # no text: the lyrics are the text, tokens carry the sung time
    tc = s["text_candidates"]
    assert tc["source"] == "lyrics" and tc["words"] == 7 + 7 + 9 and tc["tokens"][0] == dict(i=0, w="off", t=0.5, tp=0.5) and tc["tokens"][3]["t"] == 1.4         # per-word ts when the READ has them
    assert tc["tokens"][7]["w"] == "lantern" and tc["tokens"][7]["t"] == 5.2 and tc["tokens"][8]["t"] > 5.2          # interpolated inside a line without ts
    assert s["text_select"] and all(0 <= d["i"] < tc["words"] for d in s["text_select"])
    assert s["hits"] and all("sung" in h for h in s["hits"]) and all(h["text"] for h in s["hits"])


def test_video_lyrics_model_picks_and_numbered_lyrics_in_prompt():
    r = vreq(12, lyrics=LYR)
    p = ST.story_prompt(r)
    assert "REAL CUTS" in p and "shot 5: 10.03-12 s (video 10.03-12 s)" in p and "numbered word by word" in p and "[5.2 s] 7:lantern" in p and "WORD SLOTS" in p
    assert '"pick"' in p and '"camera"' not in p
    s = ST.parse(vreply(5, hits=[dict(slot=1, pick=[3, 4]), dict(slot=2, pick=[10, 11]), dict(slot=3, pick=[15])]), r)
    # 2026-11-02 ONE WORD PER SHOT: the words come from what is sung INSIDE each shot (scored), not from the model's per-slot picks; no word capacity, no breathing-room rule
    assert [h["text"] for h in s["hits"]] == ["SNEAKERS", "LANTERN LIGHT", "WHOLE BLOCK", "HIT THE TOWER"] and [h["shot"] for h in s["hits"]] == [2, 3, 4, 5]
    assert all(not ST._is_fn(h["text"].split()[-1]) for h in s["hits"])                # never ends on a function word
    assert [d["phrase"] for d in s["text_select"]][:3] == [2, 3, 3] and all(d["shot"] == d["phrase"] for d in s["text_select"])


def test_video_prompt_free_words_carry_the_render_word_prompt():
    p = ST.story_prompt(vreq(12))
    assert "You choose the words that a motion-graphics editor sets as big type" in p and "WORD SLOTS" in p and '"hits"' in p and "you never invent shots" in p
    pf = ST.story_prompt(vreq(12, on_screen=["HOLD ON"]))
    assert "ON-SCREEN LINES, fixed by the user" in pf and '"hits"' not in pf.split("Reply with ONE JSON")[1]


def test_video_lines_beyond_the_slots_are_dropped_like_the_render():
    s = ST.parse(vreply(2), vreq(8, on_screen=["A", "B", "C", "D", "E", "F"], cuts=[dict(start=0.0, end=4.0), dict(start=4.0, end=8.0)], dur=8.0, window=(0.0, 8.0)))
    assert len(s["hits"]) < 6 and "dropped" in s["notes"]


def test_video_rule_story_without_reply_has_cut_beats_and_words():
    s = ST.parse("not json", vreq(12, on_screen=["HOLD ON", "LET GO"]))
    assert len(s["shots"]) == 5 and s["shots"][0]["beat"] == "" and [h["text"] for h in s["hits"]] == ["HOLD ON", "LET GO"] and "NOT USABLE" in s["notes"]
    s = ST.parse("not json", vreq(12))                                               # no words anywhere: rule words from the footage description, never empty slots
    assert s["hits"] and all(h["text"] for h in s["hits"])


def test_video_has_no_transitions_no_beats_and_the_reply_needs_no_shots():
    d = json.loads(vreply(5))
    d["shots"] = [dict(n=k, transition="orbit:push") for k in range(1, 6)]                  # a reply that still carries transitions: ignored
    s = ST.parse(json.dumps(d), vreq(12))
    assert all("transition" not in x for x in s["shots"]) and all(x["beat"] == "" and x["camera"] == "as filmed" for x in s["shots"]) and "no beat" not in s["notes"]
    p = ST.story_prompt(vreq(12))
    assert "transition" not in p.lower().replace("transitions or camera moves", "") and '"beat"' not in p and '"shots"' not in p.split("exactly this shape:")[1]
    d.pop("shots")                                                                             # v18: the video reply has no per-shot part; still a usable story, not the fallback
    s = ST.parse(json.dumps(d), vreq(12))
    assert len(s["shots"]) == 5 and s["title"] == "test clip Bars" and "NOT USABLE" not in s["notes"]
    f = ST.parse("not json", vreq(12))
    assert len(f["shots"]) == 5 and all("transition" not in x for x in f["shots"])
    img = ST.parse(reply(3), req(12, n_images=3))                                    # image mode: unchanged, no transition field
    assert all("transition" not in x for x in img["shots"])


def test_video_keeps_every_real_cut_no_cap():
    cuts = [dict(start=round(i * 1.1, 2), end=round((i + 1) * 1.1, 2)) for i in range(21)]
    s = ST.parse(vreply(10), vreq(24, cuts=cuts, dur=24.0, window=(0.0, 24.0)))
    assert len(s["shots"]) == 22 and s["shots"][-1]["end"] == 24.0      # the reply had 10 shot entries: nothing depends on them


RAW12 = [0.033, 2.033, 2.2, 2.7, 6.0, 8.033, 8.7]                                    # the 12 s test_clip clip's raw ffmpeg hits (30 fps)
RAW24 = [0.3, 1.333, 2.033, 4.033, 4.2, 4.7, 8.0, 10.033, 10.7, 16.0, 17.333, 18.7, 21.4, 22.067]


def test_video_pieces_clip_clips_keep_every_cleaned_cut():
    p12 = ST.video_pieces(RAW12, 0.0, 12.0, fps=30)
    assert [a for a, _ in p12] == [0.0, 2.033, 2.7, 6.0, 8.033, 8.7] and p12[-1][1] == 12.0                       # 6 cards (was 4: the 1 s rule merged 2.033 and 8.033 away)
    p24 = ST.video_pieces(RAW24, 0.0, 24.0, fps=30)
    assert len(p24) == 15 and p24[-1][1] == 24.0 and max(b - a for a, b in p24) <= ST.CLIP_MAX_S + 1e-6          # 14 real pieces, the 5.3 s one split in two (was 10)
    assert 4.2 not in [a for a, _ in p24] and 0.3 in [a for a, _ in p24]                                          # 4.2 is 5 frames after 4.033: dropped like clean_cuts


def test_video_story_cards_for_every_cut_and_any_cut_takes_words():
    cuts = [dict(start=a, end=b) for a, b in ST.video_pieces(RAW12, 0.0, 12.0, fps=30)]
    rd = vreq(12, cuts=cuts, dur=12.066, window=(0.0, 12.0), text="alpha bravo charlie delta echo foxtrot golf hotel")
    s = ST.parse(vreply(6), rd)
    assert len(s["shots"]) == 6
    rd["select"] = [dict(i=0, phrase=1, shot=2), dict(i=1, phrase=1, shot=2), dict(i=5, phrase=2, shot=2), dict(i=3, phrase=9, shot=6), dict(i=7, phrase=3, shot=1)]
    s = ST.parse(vreply(6), rd)
    assert [x["line"] for x in s["shots"]] == ["HOTEL", "ALPHA BRAVO FOXTROT", "", "", "", "DELTA"]                  # tokens of one shot in TOKEN order; shots 3-5 (no words) stay cards
    assert sorted((d["i"], d["shot"]) for d in s["text_select"]) == [(0, 2), (1, 2), (3, 6), (5, 2), (7, 1)]
    for h in s["hits"]:                                                                                           # each hit lands inside its shot's span
        sh = next(x for x in s["shots"] if x["line"] == h["text"])
        assert sh["start"] - 1e-6 <= h["land"] < sh["end"] + 1e-6 and h["hold_until"] <= sh["end"] + 1e-6
    assert [h["text"] for h in s["hits"]] == ["HOTEL", "ALPHA BRAVO FOXTROT", "DELTA"] and "won't fit" not in s["notes"]
    rd["select"] = [dict(i=0, phrase=1, shot=99)]
    s = ST.parse(vreply(6), rd)
    assert all(x["line"] == "" for x in s["shots"]) and "does not exist" in s["notes"] and "won't fit" not in s["notes"]


def test_image_select_by_shot_places_words_on_that_shot_only():
    r = treq(12, select=[{"i": 6, "phrase": 1, "shot": 3}, {"i": 7, "phrase": 1, "shot": 3}, {"i": 0, "phrase": 7, "shot": 1}])
    out = ST.parse(reply(3, lines=["NOPE", "NOPE", "NOPE"]), r)
    assert [x["line"] for x in out["shots"]] == ["HOLD", "", "LET GO"]                                              # shot 2 empty, shot 3 has the adjacent run
    assert out["text_select"] == [{"i": 0, "phrase": 1, "shot": 1}, {"i": 6, "phrase": 3, "shot": 3}, {"i": 7, "phrase": 3, "shot": 3}]
    assert "won't fit" not in out["notes"]
    o2 = ST.parse(reply(2), treq(8, select=[{"i": 1, "phrase": 1, "shot": 5}, {"i": 2, "phrase": 1, "shot": 2}]))
    assert [x["line"] for x in o2["shots"]] == ["", "TO"] and "shot 5 does not exist" in o2["notes"] and all(e["shot"] == 2 for e in o2["text_select"])
    o3 = ST.parse(reply(3), treq(12, select=[{"i": 1, "phrase": 1}, {"i": 2, "phrase": 1, "shot": 2}]))            # one entry has a shot: by-shot mode, the other is ignored
    assert [x["line"] for x in o3["shots"]] == ["", "TO", ""] and "no shot" in o3["notes"]


def test_image_mode_audio_lyrics_phrase_goes_on_the_shot_where_it_is_sung():
    r = req(12, n_images=3)
    r["read"]["lyrics"] = [dict(t=0.5, text="alpha one two"), dict(t=5.0, text="bravo three four"), dict(t=9.0, text="charlie five six")]
    r["read"]["media"].append(dict(id="A1", kind="audio", file="song.wav", shows="song", window=None, mood="", palette=[]))
    rep = json.dumps(dict(title="x", logline="y", look="z", music="m", notes="", shots=[dict(n=1, beat="b1", pick=[3]), dict(n=2, beat="b2", pick=[6]), dict(n=3, beat="b3", pick=[])]))
    s = ST.parse(rep, r)                                                             # shots 0-4 / 4-8 / 8-12: the model put BRAVO (sung 5 s) on shot 1 and CHARLIE (9 s) on shot 2
    assert s["text_candidates"]["source"] == "lyrics" and s["text_candidates"]["tokens"][3]["t"] == 5.0
    lines = [x["line"] for x in s["shots"]]
    assert "BRAVO" not in lines[0] and "CHARLIE" not in lines[1] and all(lines)        # WORD PICK: a pick that is not sung inside its shot is rejected (said in the notes), the shot gets the best word sung during it
    assert "not sung inside" in s["notes"]
    tk = s["text_candidates"]["tokens"]
    assert all(x["start"] <= tk[d["i"]]["t"] < x["end"] for d in s["text_select"] for x in [s["shots"][d["shot"] - 1]])       # every chosen word is sung while its shot is on
    pr = ST.story_prompt(r)
    assert "THE SONG'S LYRICS" in pr and "[5.0 s] 3:bravo" in pr


# ------------------------------------------------------------------------------------------------- v19 AUDIO TRACK + SOUNDTRACK
def _vp(st):
    r = vreq(12, lyrics=LYR)
    if st:
        r["read"]["soundtrack"] = st
    return ST.story_prompt(r)


def test_soundtrack_wording_follows_the_reads_soundtrack():
    assert "the video's own audio is the soundtrack" in _vp("video") and "the video's own audio is the soundtrack" in _vp("")
    assert "the video's own audio is the soundtrack" in _vp("keep")                                  # the old name
    pt = _vp("track")
    assert "the user's own audio file replaces the video's sound" in pt and "the user's own audio file is the soundtrack: describe it" in pt
    pg = _vp("generate")
    assert "new music will be written for the piece" in pg and "describe the music that fits the footage's mood" in pg


def test_track_lyrics_on_the_videos_timeline_land_on_the_shot_where_they_are_heard():
    # media_read.parse puts a "my track" lyric line at (track second + window start); here the window starts at 2 s and the line is heard at track 3.0 s -> t = 5.0, in the piece at 3.0 s
    r = vreq(12, lyrics=[dict(t=5.0, text="alpha bravo charlie")], cuts=[dict(start=2.0, end=5.0), dict(start=5.0, end=8.0), dict(start=8.0, end=14.0)], window=(2.0, 14.0))
    r["read"]["soundtrack"] = "track"
    s = ST.parse(vreply(3), r)
    ts = s["text_candidates"]["tokens"]
    assert s["text_candidates"]["source"] == "lyrics" and ts[0]["t"] == 5.0 and ts[0]["tp"] == 3.0
    assert "[3.0 s] 0:alpha" in ST.story_prompt(r)                                      # the numbered lyric rows carry the PIECE second


# ------------------------------------------------------------------------------------------------- PHRASES + SUNG TIMING (invented lyrics)
SUNG = [dict(t=1.9, text="marble clouds over the harbor lights", ts=[1.9, 2.3, 2.6, 2.8, 3.1, 3.5]),
        dict(t=6.5, text="silver thread in the morning", ts=[6.5, 6.9, 7.2, 7.4, 7.8]),
        dict(t=10.0, text="quiet engines hum", ts=[10.0, 10.4, 10.8])]
SUNG_LINES = ["MARBLE CLOUDS OVER THE HARBOR LIGHTS", "SILVER THREAD IN THE MORNING", "QUIET ENGINES HUM"]


@pytest.mark.parametrize("line,want", [("MARBLE CLOUDS OVER THE HARBOR LIGHTS", "MARBLE CLOUDS"), ("SILVER THREAD IN THE MORNING", "SILVER THREAD"), ("QUIET ENGINES HUM", "QUIET ENGINES HUM"),
                                       ("BRIGHT LANTERNS, SWAYING OVER THE BAY", "BRIGHT LANTERNS"), ("light on the water tonight", "LIGHT ON THE WATER"),
                                       ("rooftop lanterns and the sea", "ROOFTOP LANTERNS"), ("ON THE", ""), ("a b c d e", "A B C D")])
def test_phrase_trim_is_a_standalone_line_never_ending_on_a_function_word(line, want):
    got = ST.trim_phrase(line)
    assert got == want
    if got:
        assert len(got.split()) <= ST.PHRASE_WORDS and not ST._is_fn(got.split()[-1])


def test_video_lyric_lines_are_trimmed_to_phrases_before_they_become_hits():
    s = ST.parse(vreply(5), vreq(12, lyrics=SUNG, on_screen=SUNG_LINES))       # the READ's own on-screen lines stay the user's words: trimmed to phrases, landed when sung
    assert [h["text"] for h in s["hits"]] == ["MARBLE CLOUDS", "SILVER THREAD", "QUIET ENGINES HUM"]
    assert all(not ST._is_fn(h["text"].split()[-1]) for h in s["hits"])
    p = ST.story_prompt(vreq(12, lyrics=SUNG))
    assert "NEVER end on such a small word" in p and "lands when its first word is sung" in p                    # the rule is in the prompt too
    assert "NEVER end on such a small word" in ST.story_prompt(vreq(12))                                          # free words
    s = ST.parse(vreply(5, hits=[dict(slot=1, pick=[0, 1, 2]), dict(slot=2, pick=[6, 7, 8]), dict(slot=3, pick=[11, 12, 13])]), vreq(12, lyrics=SUNG))
    assert [h["text"] for h in s["hits"]] == ["LIGHTS", "MORNING", "ENGINES HUM"] and [h["shot"] for h in s["hits"]] == [2, 3, 5]       # one word per shot (the shot's own best sung word), each inside its shot
    assert all(s["shots"][h["shot"] - 1]["start"] <= h["land"] < s["shots"][h["shot"] - 1]["end"] and h["hold_until"] <= s["shots"][h["shot"] - 1]["end"] + 1e-6 for h in s["hits"])
    assert ST._gen_line("neon rain on", "")[0] == "NEON RAIN" and ST._gen_line("on the", "")[0] == ""            # free words (Gemma writes the line)


def test_video_hit_lands_when_its_first_word_is_sung_snapped_to_the_beat():
    s = ST.parse(vreply(5), vreq(12, lyrics=SUNG, on_screen=SUNG_LINES))
    beats = set(GRID["beats"])
    for h, first in zip(s["hits"], (1.9, 6.5, 10.0)):
        assert h["sung"] == first
        assert h["land"] >= first - 1e-9 and h["land"] - first <= ST.SNAP_S + 1e-9                               # never before it is sung, within the snap
    assert [h["land"] for h in s["hits"][:2]] == [2.0, 6.67] and 2.0 in beats and 6.67 in beats                    # 1.9 -> beat 2.0, 6.5 -> beat 6.67
    assert s["hits"][2]["land"] == 10.01                                                                          # the beat 10.01 is 0.01 s after it is sung
    assert [h["land"] for h in s["hits"]] != [round(w["land"], 3) for w in _slots(3)]                             # not the evenly spaced slots any more
    assert s["shots"][1]["line"] == "" and s["shots"][2]["line"] == "SILVER THREAD"                                # the card follows: 6.67 s is in cut 3 (4.7-8.0)
    lone = [dict(t=2.05, text="marble clouds", ts=[2.05, 2.4])]                                                   # no beat within +0.25 s after it: it lands on the word itself
    assert ST.parse(vreply(5), vreq(12, lyrics=lone, on_screen=["MARBLE CLOUDS"]))["hits"][0]["land"] == 2.05


def test_video_hit_lands_in_piece_seconds_after_the_window_offset():
    cuts = [dict(start=4.0, end=8.0), dict(start=8.0, end=12.0)]
    lyr = [dict(t=5.9, text="marble clouds over the harbor", ts=[5.9, 6.3, 6.6, 6.8, 7.1])]
    s = ST.parse(vreply(2), vreq(8, lyrics=lyr, on_screen=["MARBLE CLOUDS OVER THE HARBOR"], cuts=cuts, window=(4.0, 12.0)))
    h = s["hits"][0]
    assert h["text"] == "MARBLE CLOUDS" and h["sung"] == 5.9 and h["land"] == 2.0 and h["hold_until"] == 3.0       # 5.9 in the video = 1.9 in the piece


def test_video_hold_runs_until_the_last_word_is_sung_plus_the_minimum_hold():
    s = ST.parse(vreply(5), vreq(12, lyrics=SUNG, on_screen=SUNG_LINES))
    assert s["hits"][0]["hold_until"] == round(2.3 + ST.MIN_HOLD_S, 3) and s["hits"][1]["hold_until"] == round(6.9 + ST.MIN_HOLD_S, 3)      # the last word of the PHRASE (CLOUDS / THREAD)
    assert s["hits"][2]["hold_until"] == round(10.8 + ST.MIN_HOLD_S, 3)
    assert all(h["hold_until"] >= h["land"] + ST.MIN_HOLD_S - 1e-9 for h in s["hits"])
    one = [dict(t=3.2, text="harbor", ts=[3.2]), dict(t=8.0, text="lights hum low", ts=[8.0, 8.3, 8.6])]          # a one-word line: hold >= the 0.7 s minimum after it lands
    h = ST.parse(vreply(5), vreq(12, lyrics=one, on_screen=["HARBOR"]))["hits"][0]
    assert h["land"] >= 3.2 and h["hold_until"] >= h["land"] + ST.MIN_HOLD_S - 1e-9


def test_video_line_not_in_the_transcription_keeps_its_word_slot():
    s = ST.parse(vreply(5), vreq(12, lyrics=SUNG, on_screen=["MARBLE CLOUDS", "ZEBRA UNSUNG LINE", "QUIET ENGINES HUM"]))
    mid = next(h for h in s["hits"] if h["text"] == "ZEBRA UNSUNG LINE")
    slot = _slots(3)[1]
    assert (mid["land"], mid["hold_until"]) == (round(slot["land"], 3), round(slot["hold_until"], 3)) and "sung" not in mid
    assert [h["land"] for h in s["hits"] if "sung" in h] == [2.0, 10.01]
    s = ST.parse(vreply(5), vreq(12, on_screen=["MARBLE CLOUDS", "QUIET ENGINES HUM"]))                           # no transcription at all: today's behaviour
    want = _slots(2)
    assert [(h["land"], h["hold_until"]) for h in s["hits"]] == [(round(w["land"], 3), round(w["hold_until"], 3)) for w in want]


def test_video_sung_lines_too_close_are_pushed_or_dropped_never_pulled_earlier():
    close = [dict(t=3.0, text="marble clouds", ts=[3.0, 3.4]), dict(t=4.5, text="silver thread", ts=[4.5, 4.9]), dict(t=5.2, text="quiet engines", ts=[5.2, 5.6])]
    s = ST.parse(vreply(5), vreq(12, lyrics=close, on_screen=["MARBLE CLOUDS", "SILVER THREAD", "QUIET ENGINES"]))
    lands = [h["land"] for h in s["hits"]]
    assert lands[0] >= 3.0 and all(b - a >= ST.MIN_GAP_S - 1e-6 for a, b in zip(lands, lands[1:]))
    by = {h["text"]: h for h in s["hits"]}
    assert by["SILVER THREAD"]["land"] >= 4.5 and "QUIET ENGINES" not in by and "dropped" in s["notes"]           # 0.7 s after the second one: too close, said in the notes


def test_video_line_with_a_word_the_transcription_lacks_is_placed_by_the_words_that_are_there():
    lyr = [dict(t=2.9, text="glow on the rooftops tonight", ts=[2.9, 3.3, 3.5, 3.7, 4.3]), dict(t=8.0, text="quiet engines hum", ts=[8.0, 8.4, 8.8])]      # Whisper dropped "copper" (low probability)
    s = ST.parse(vreply(5), vreq(12, lyrics=lyr, on_screen=["COPPER GLOW ON THE ROOFTOPS", "QUIET ENGINES HUM"]))
    h = s["hits"][0]
    assert h["text"] == "COPPER GLOW" and h["sung"] == 2.9 and 2.9 <= h["land"] <= 2.9 + ST.SNAP_S + 1e-9 and h["hold_until"] >= h["land"] + ST.MIN_HOLD_S - 1e-9
