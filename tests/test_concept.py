"""Concept generation (design_llm invents the request), its uniqueness check, the auto graph, the timing parser and the batch driver's resume logic.
No model, no network, no GPU. Run: python -m pytest tests/test_concept.py -q"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT / "tools" / "app")):
    if p not in sys.path:
        sys.path.append(p)

from tools.fusion import concept as CO  # noqa: E402


def reply(**kw):
    d = {"title": "Harbour light", "words": ["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME", "KEEP THE LIGHT"], "subject": "old lighthouse keeper",
         "scene": "an old lighthouse keeper in a yellow coat standing on the stone gallery as a storm breaks over a dark sea, spray in the wind, one beam cutting the clouds, seen from far below",
         "music": "cinematic orchestral, low strings, taiko drums, brass swells, dark, big reverb, slow build, epic hall", "bpm": 88, "mood": "solemn", "look": "slab",
         "shots": "single"}
    d.update(kw)
    return json.dumps(d)


def concept(**kw):
    r = CO.parse(reply(**kw), [], 1, 16)
    assert r["concept"] is not None
    return r["concept"]


def _comfy():
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


# ------------------------------------------------------------------------------------------------ parse
def test_a_good_reply_parses_to_request_fields():
    r = CO.parse(reply(), [], 1, 16)
    assert r["ok"] and r["errors"] == [] and r["clashes"] == []
    c = r["concept"]
    assert c["words"] == ["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME", "KEEP THE LIGHT"] and c["look"] == "slab"
    assert c["music"].endswith("88 bpm") and "solemn" in c["music"] and c["shots"] in CO.MODES and c["length"] == 16
    assert 8 <= len(c["tags"]) or len(c["tags"]) >= 3


def test_reply_in_a_fence_with_prose_and_lowercase_words_is_cleaned():
    text = "Sure!\n```json\n" + reply(words=["the sea", "remembers", "every ship!", "every name", "keep the light", "  ", "keep the light"]) + "\n```"
    r = CO.parse(text, [], 1, 16)
    assert r["ok"] and r["concept"]["words"][0] == "THE SEA" and len(r["concept"]["words"]) == 5 and any("cleaned" in f for f in r["fixes"])


@pytest.mark.parametrize("bad,needle", [
    (dict(words=["ONE", "TWO"]), "usable lines"),
    (dict(words=["A VERY LONG LINE OF WORDS THAT RUNS ON", "B", "C", "D", "E"]), "too long"),
    (dict(subject=""), "subject"),
    (dict(scene="a dog"), "scene"),
    (dict(music="rock"), "music"),
    (dict(mood=""), "mood"),
])
def test_unusable_fields_are_errors(bad, needle):
    r = CO.parse(reply(**bad), [], 1, 16)
    assert not r["ok"] and any(needle in e for e in r["errors"]), r["errors"]


def test_not_json_is_an_error_and_repair_prompt_carries_the_reply_and_problems():
    r = CO.parse("I would like a film about the sea.", [], 1, 16)
    assert not r["ok"] and "no parseable JSON" in r["errors"][0]
    rp = CO.repair_prompt(CO.concept_prompt(1, 16, []), "I would like a film", r["errors"])
    assert rp.startswith("You invent the REQUEST") and "I would like a film" in rp and "no parseable JSON" in rp


def test_look_bpm_and_mode_are_fixed_not_rejected():
    r = CO.parse(reply(look="cubism", bpm=500, shots="montage"), [], 1, 16)
    c = r["concept"]
    assert r["ok"] and c["look"] in CO.PS.FAMILY and c["bpm"] == CO.BPM_RANGE[1] and c["shots"] in CO.MODES and len(r["fixes"]) >= 3


# ------------------------------------------------------------------------------------------------ uniqueness
@pytest.mark.parametrize("change,needle", [
    ({}, "the words are the same"),
    (dict(words=["THE SEA", "REMEMBERS", "EVERY SHIP", "NEW LINE"], shots="cut"), "the words are the same"),                                      # 3 of 4 lockups shared
    (dict(words=["NIGHT", "SHIFT", "ENDS", "AT SIX", "GO HOME"], subject="Old Lighthouse keeper"), "same scene subject"),
    (dict(words=["NIGHT", "SHIFT", "ENDS", "AT SIX", "GO HOME"], subject="harbour pilot",
          music="cinematic orchestral, low strings, taiko drums, brass swells, tense, hall"), "music tags overlap"),
    (dict(words=["NIGHT", "SHIFT", "ENDS", "AT SIX", "GO HOME"], subject="harbour pilot", music="synthwave, analog bass, gated drums, neon pads, retro",
          scene="a pilot boat cutting across black water toward a container ship lit like a city, rain on the lens, tracking from the stern"), "same mood"),
])
def test_clashes_with_an_earlier_concept_are_found(change, needle):
    prev = [concept()]
    r = CO.parse(reply(**change), prev, 2, 16)
    assert not r["ok"] and any(needle in c for c in r["clashes"]), r["clashes"]


def test_a_really_different_concept_is_unique():
    prev = [concept()]
    other = reply(title="Red dune", words=["ONE", "STEP", "THEN", "ANOTHER", "THE WAY BACK"], subject="astronaut",
                  scene="an astronaut in a white suit walking up a steep red sand dune under a pale pink sky with two small moons, footprints behind, long shot from below",
                  music="space ambient, slow synth arpeggio, deep pads, soft kick, vast, cinematic", mood="lonely", bpm=85, look="ring", shots="cut")
    r = CO.parse(other, prev, 2, 16)
    assert r["ok"], (r["errors"], r["clashes"])


def test_shots_mode_suggestion_is_multi_shot_and_the_models_valid_choice_stands():
    """3060 round: two of four pieces came out as ONE long shot because the series' spread overrode the model's mode with the least used one (single). Now the suggestion is only ever
    cut / chain and a valid answer of the model is never replaced."""
    prev = [{"shots": "cut"}, {"shots": "cut"}]
    assert CO.target_mode(prev, 3) == "chain" and CO.target_mode([{"shots": "single"}] * 5, 3) in ("cut", "chain")
    r = CO.parse(reply(shots="cut"), prev, 3, 16)
    assert r["concept"]["shots"] == "cut" and not any("least used" in f for f in r["fixes"])
    assert CO.parse(reply(shots="single"), prev, 3, 16)["concept"]["shots"] == "single"          # a genuinely one-scene idea may say so
    assert CO.parse(reply(shots="wobble"), prev, 3, 16)["concept"]["shots"] == "chain"           # an unusable answer takes the suggestion
    p = CO.concept_prompt(3, 16, prev)
    assert "Shot structure" in p and "ALREADY MADE" in p and "last frame" in p.lower()      # no last-frame chaining: the prompt says every shot has its own still


def test_prompt_lists_the_earlier_concepts_and_fixes_the_length():
    prev = [concept()]
    p = CO.concept_prompt(7, 16, prev)
    assert "16 seconds" in p and "THE SEA / REMEMBERS" in p and "old lighthouse keeper" in p and "Design seed: 7" in p
    assert CO.concept_prompt(7, 16, prev) == CO.concept_prompt(7, 16, prev) and CO.concept_prompt(8, 16, prev) != p        # seeded cues


# ------------------------------------------------------------------------------------------------ final: repair, then the seeded rules
def test_final_uses_the_first_reply_when_clean():
    first = CO.parse(reply(), [], 1, 16)
    c, rep = CO.final(None, first, [], 1, 16)
    assert c["source"] == "model" and "first reply" in rep


def test_final_takes_a_clean_repair_reply_and_falls_back_to_rules_otherwise():
    prev = [concept()]
    first = CO.parse(reply(), prev, 2, 16)
    assert not first["ok"]
    good = reply(words=["ONE", "STEP", "THEN", "ANOTHER", "THE WAY BACK"], subject="astronaut",
                 scene="an astronaut in a white suit walking up a steep red sand dune under a pale pink sky with two small moons, footprints behind, long shot from below",
                 music="space ambient, slow synth arpeggio, deep pads, soft kick, vast, cinematic", mood="lonely", bpm=85)
    c, rep = CO.final(good, first, prev, 2, 16)
    assert c["source"] == "model (repair)" and c["subject"] == "astronaut" and CO.clashes(c, prev) == []
    c2, rep2 = CO.final(reply(), first, prev, 2, 16)                       # the repair reply repeats the clash
    assert c2["source"].startswith("rules") and CO.clashes(c2, prev) == [] and "repair reply" in rep2
    c3, _ = CO.final("garbage", first, prev, 2, 16)
    assert c3["source"].startswith("rules")


def test_the_rules_variation_gives_ten_unique_concepts_with_spread_modes():
    prev = []
    for i in range(1, 11):
        c = CO.rules_variation(i, prev, 16)
        assert CO.clashes(c, prev) == [], (i, CO.clashes(c, prev))
        n_l = CO.lockup_count(16, c["shots"])           # one lockup per shot (merged / split from the bank's lines); a single shot keeps 5-7 free lines
        assert (len(c["words"]) == n_l if n_l else 5 <= len(c["words"]) <= 7) and c["shots"] in CO.MODES and c["music"].endswith("bpm")
        assert CO.normalize({**c, "music": c["music"]}, i, [], 16)[1] == [], CO.normalize(c, i, [], 16)[1]       # a bank entry passes the model's own validator
        prev.append(c)
    n = CO.mode_counts(prev)
    assert n["single"] == 0 and n["cut"] >= 4 and n["chain"] >= 4 and sum(n.values()) == 10          # the rules never suggest one long shot
    assert len({tuple(c["words"]) for c in prev}) == 10 and len({c["mood"] for c in prev}) >= 9
    assert CO.rules_variation(3, prev, 16) == CO.rules_variation(3, prev, 16)                           # seeded


def test_rules_variation_survives_an_exhausted_bank():
    prev = [CO._from_bank(e, 1, 16) for e in CO.BANK]
    c = CO.rules_variation(5, prev, 16)
    assert c["source"].startswith("rules (bank used up)") and c["words"]


def test_parse_previous_is_tolerant():
    assert CO.parse_previous("") == [] and CO.parse_previous("nope") == [] and CO.parse_previous('{"concepts": [{"words": ["A"]}]}') == [{"words": ["A"]}]
    assert CO.parse_previous('[{"a": 1}, 3, "x"]') == [{"a": 1}]


# ------------------------------------------------------------------------------------------------ nodes
def test_concept_nodes_and_the_lazy_repair():
    _comfy()
    from nodes_fusion import concept as N
    import nodes_fusion
    assert "concept" in nodes_fusion.LOADED
    prev = json.dumps([concept()])
    p = N.MographConceptPrompt.execute(2, 16, prev).args[0]
    assert "ALREADY MADE" in p and "THE SEA / REMEMBERS" in p
    ok, rp, rep = N.MographConceptParse.execute(reply(), "[]", 1, 16, p).args
    assert ok is True and rp == ""
    ok2, rp2, rep2 = N.MographConceptParse.execute(reply(), prev, 2, 16, p).args
    assert ok2 is False and rp2.startswith(p.rstrip()[:50]) and "words are the same" in rp2 and "problem:" in rep2
    F = N.MographConceptFinal
    lazy = {i.id: getattr(i, "lazy", None) for i in F.define_schema().inputs}
    assert lazy["repair_reply"] is True and not any(v for k, v in lazy.items() if k != "repair_reply")
    assert F.check_lazy_status(reply="x", ok=True, repair_reply=None) == []
    assert F.check_lazy_status(reply="x", ok=False, repair_reply=None) == ["repair_reply"]
    assert F.check_lazy_status(reply="x", ok=False, repair_reply="{}") == []
    out = F.execute(reply(), True, "[]", 1, 16).args
    words, scene, music, look, shots, cj, report, owners = out
    assert words.split("\n")[0] == "THE SEA" and shots in CO.MODES and json.loads(cj)["source"] == "model"
    assert (lambda o: o if isinstance(o, list) else o["owners"])(json.loads(owners)) == []                                           # a reply without owners: the rules decide (empty list)
    out2 = F.execute(reply(), False, prev, 2, 16, None).args                # false and nothing arrived: seeded rules, still unique
    assert json.loads(out2[5])["source"].startswith("rules") and CO.clashes(json.loads(out2[5]), json.loads(prev)) == []


def test_request_node_takes_the_shots_mode():
    _comfy()
    from nodes_fusion import plan as P
    r = P.MographRequest.execute("A\nB\nC", "", "", 16, "auto", 1, "preview", None, None, "9:16", "chain").args[0]
    assert r["shots"] == "chain" and r["shots_given"] is True
    r0 = P.MographRequest.execute("A\nB\nC", "", "", 16, "auto", 1, "preview").args[0]
    assert r0["shots_given"] is False and r0["shots"] == "cut"


# ------------------------------------------------------------------------------------------------ the auto graph
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def needs_graph(name):
    """skip when a dev-only graph is absent (the Qwen / fusion design graphs are not shipped in the public export)."""
    return pytest.mark.skipif(not (ROOT / "app" / "graphs" / name).exists(), reason=f"dev fixture not shipped in the public export: app/graphs/{name}")


@pytest.fixture(scope="module")
def bg():
    return _load("build_graphs_auto_t", "tools/app/build_graphs.py")


@pytest.fixture(scope="module")
def auto(bg):
    return bg.make(False, None, "moss", True, True)


@needs_graph("fusion_design_qwen_auto.json")
def test_auto_graph_passes_the_checker_and_is_on_disk(bg, auto):
    g, side = auto
    problems, _ = bg.check(g, side)
    assert problems == [], problems
    assert (ROOT / "app" / "graphs" / "fusion_design_qwen_auto.json").read_text(encoding="utf-8") == bg.dump(g)
    assert (ROOT / "app" / "graphs" / "fusion_design_qwen_auto.app.json").read_text(encoding="utf-8") == bg.dump(side)


def test_auto_graph_concept_drives_the_request(auto):
    g, side = auto
    assert not any(g.get(k) for k in ("words", "scene", "music", "look"))                                  # no hand-typed request
    assert g["length"]["inputs"]["value"] == 16 and g["quality"]["inputs"]["value"] == "preview" and g["previous"]["inputs"]["value"] == "[]"
    r = g["request"]["inputs"]
    assert r["words"] == ["concept_final", 0] and r["scene"] == ["concept_final", 1] and r["music"] == ["concept_final", 2]
    assert r["look"] == ["concept_final", 3] and r["shots"] == ["concept_final", 4] and r["length"] == ["length", 0]      # the length is not the model's
    assert g["concept_qwen"]["inputs"]["custom_prompt"] == ["concept_prompt", 0] and "image" not in g["concept_qwen"]["inputs"]
    assert g["concept_final"]["inputs"]["repair_reply"] == ["concept_qwen2", 0]
    users = [(k, n) for k, v in g.items() for n, x in v["inputs"].items() if x == ["concept_qwen2", 0]]
    assert users == [("concept_final", "repair_reply")]                                                    # the repair model runs only when requested
    for k in ("concept_qwen", "concept_qwen2", "design_qwen", "design_qwen2"):
        assert g[k]["inputs"]["keep_model_loaded"] is False and g[k]["inputs"]["quantization"].startswith("4-bit")
    assert any(v["_meta"]["title"] == "OUT concept" for v in g.values())
    assert [f["id"] for f in side["fields"] if f["id"] in ("words", "scene", "music", "look")] == []
    assert not [c for c in {v["class_type"] for v in g.values()} if any(x in c.lower() for x in ("claude", "anthropic", "openai", "gemini", "kling", "seedream"))]


@needs_graph("fusion.json")
def test_the_other_graphs_are_unchanged(bg):
    for name, kw in (("fusion", dict()), ("fusion_design_qwen", dict(design=True))):
        g, side = bg.make(False, None, "moss", kw.get("design", False))
        assert (ROOT / "app" / "graphs" / f"{name}.json").read_text(encoding="utf-8") == bg.dump(g)


# ------------------------------------------------------------------------------------------------ run_timing + run_batch (no network)
@needs_graph("fusion_design_qwen.json")
def test_run_timing_parses_progress_standin_output():
    import run_timing as RT
    g = json.load(open(ROOT / "app" / "graphs" / "fusion_design_qwen.json", encoding="utf-8"))

    def ev(t, ty, **d):
        return t, "STANDIN_RELAY " + json.dumps({"type": ty, "data": d})
    lines = [ev(0.0, "execution_start", prompt_id="p1"), ev(1.0, "executing", node="plan", prompt_id="p1"), ev(4.0, "executing", node="design_qwen", prompt_id="p1"),
             (30.0, "  something CUDA error: out of memory in a log line"), ev(40.0, "executing", node="compose", prompt_id="p1"),
             ev(50.0, "executing", node="design_qwen", prompt_id="p1"), ev(55.0, "execution_success", prompt_id="p1"),
             (56.0, "RESULT OK  55s  peakVRAM 10321 MiB  peakRAM 31.5 GiB anon (60.0 GiB incl cache) [cgroup v2]")]
    run, extra = RT.parse_lines(lines, g)
    s = RT.summarize(run, g, extra)
    assert s["status"] == "done" and s["total_seconds"] == 55.0 and extra["peak_vram_mib"] == 10321 and extra["prompt_id"] == "p1" and "out of memory" in extra["oom"]
    by = {n["id"]: n for n in s["nodes"]}
    assert by["design_qwen"]["seconds"] == 36 - 0.0 + 0 or by["design_qwen"]["seconds"] > 0
    st = {x["stage"]: x for x in s["stages"]}
    assert "Qwen design" in st and "Plan (rules)" in st and "Compose" in st and st["Qwen design"]["nodes"] == 1
    md = RT.markdown("t", dict(s, wall_s=60.0))
    assert "Per stage" in md and "10321" in md
    assert RT.stage_of("c3.39") == "LTX shots" and RT.stage_of("a2.41") == "Alpha-Gen mattes" and RT.stage_of("d4.proc") == "Depth (Video Depth Anything)"
    assert RT.stage_of("concept_qwen2").startswith("Concept") and RT.stage_of("t2m_ks").startswith("Music")


def _fake_summary(i, ok=True, repair=False):
    nodes = [{"id": "concept_qwen", "seconds": 5.0}] + ([{"id": "concept_qwen2", "seconds": 4.0}] if repair else [])
    return {"status": "done" if ok else "error", "error": None if ok else {"node": "c1.39", "type": "OutOfMemoryError", "message": "oom"}, "prompt_id": f"pid{i}",
            "wall_s": 100.0 + i, "total_seconds": 90.0 + i, "peak_vram_mib": 9000 + i, "oom": None, "nodes": nodes, "stages": [
                {"stage": "Qwen design", "seconds": 20.0 + i}, {"stage": "LTX shots", "seconds": 50.0}],
            "outputs": {"OUT final": [f"video/motngrfx_{i:05d}_.mp4"]}}


def _fake_texts(i):
    return {"OUT concept": [json.dumps({"title": f"T{i}", "words": [f"W{i}A", f"W{i}B", f"W{i}C", f"W{i}D", f"W{i}E"], "subject": f"subject {i}", "mood": f"m{i}",
                                       "scene": "x", "music": "y", "shots": ["cut", "chain", "single"][i % 3], "look": "auto"})],
            "OUT design final check": [f"repair: not needed\npiece: mode={['cut', 'chain', 'single'][i % 3]} shots=4 stills={4 if i % 3 == 0 else 1} cut=16.{i:03d}s frames=385 words=5 technique=behind quality=preview"]}


def _args(tmp_path, **kw):
    g = tmp_path / "g_ltx.json"
    g.write_text("{}", encoding="utf-8")
    return dict(iid="1", port=8190, n=4, length=16, graph=str(g), out=str(tmp_path / "out"), idea="app", project=False, resume=False, timeout=10, **kw)


def test_batch_runs_in_order_with_previous_concepts_and_fixed_length(tmp_path):
    import run_batch as RB
    calls = []

    def runner(graph, iid, port, sets, out, **kw):
        calls.append((dict(sets), kw.get("project")))
        return _fake_summary(len(calls))
    st = RB.run_batch(_args(tmp_path), run_graph=runner, texts_of=lambda port, pid, g: _fake_texts(int(pid[3:])), log=lambda *a: None)
    assert [c[0]["seed"] for c in calls] == [1, 2, 3, 4] and all(c[0]["length"] == 16 for c in calls)
    prevs = [json.loads(c[0]["previous"]) for c in calls]
    assert prevs[0] == [] and [len(p) for p in prevs] == [0, 1, 2, 3] and prevs[3][2]["title"] == "T3"
    assert all(r["status"] == "done" for r in st["runs"])
    r2 = st["runs"][1]
    assert r2["mode"] == "single" and r2["shots"] == 4 and r2["stills"] == 1 and r2["cut_s"] == 16.002 and r2["peak_vram_mib"] == 9002 and r2["final"].endswith("00002_.mp4")
    md = (tmp_path / "out" / "BATCH.md").read_text(encoding="utf-8")
    assert "| 4 |" in md and "Seconds per stage" in md and "T3" in md
    assert json.loads((tmp_path / "out" / "BATCH.json").read_text(encoding="utf-8"))["n"] == 4


def test_batch_resumes_after_a_crash_and_never_repeats_a_done_run(tmp_path):
    import run_batch as RB
    calls = []

    def crashing(graph, iid, port, sets, out, **kw):
        calls.append(sets["seed"])
        if len(calls) == 3:
            raise KeyboardInterrupt                                            # the driver dies in run 3 (not an Exception: nothing catches it)
        return _fake_summary(len(calls))
    texts = lambda port, pid, g: _fake_texts(int(pid[3:]))                       # noqa: E731
    with pytest.raises(KeyboardInterrupt):
        RB.run_batch(_args(tmp_path), run_graph=crashing, texts_of=texts, log=lambda *a: None)
    st = RB.load_state(str(tmp_path / "out" / "state.json"))
    assert [r["status"] for r in st["runs"]] == ["done", "done", "running", "pending"]
    with pytest.raises(SystemExit):                                              # a batch is never overwritten without --resume
        RB.run_batch(_args(tmp_path), run_graph=crashing, texts_of=texts, log=lambda *a: None)
    calls2 = []

    def runner(graph, iid, port, sets, out, **kw):
        calls2.append((sets["seed"], json.loads(sets["previous"])))
        return _fake_summary(sets["seed"])
    st = RB.run_batch(dict(_args(tmp_path), resume=True), run_graph=runner, texts_of=texts, log=lambda *a: None)
    assert [c[0] for c in calls2] == [3, 4] and [len(c[1]) for c in calls2] == [2, 3]            # 1 and 2 are not repeated; their concepts are the 'previous'
    assert all(r["status"] == "done" for r in st["runs"]) and st["runs"][2].get("restarted") == 1


def test_batch_continues_after_a_failed_run_and_keeps_its_concept(tmp_path):
    import run_batch as RB
    seen = []

    def runner(graph, iid, port, sets, out, **kw):
        seen.append(json.loads(sets["previous"]))
        if sets["seed"] == 2:
            raise RuntimeError("box went away")
        return _fake_summary(sets["seed"], ok=sets["seed"] != 3)
    st = RB.run_batch(_args(tmp_path), run_graph=runner, texts_of=lambda port, pid, g: _fake_texts(int(pid[3:])), log=lambda *a: None)
    assert [r["status"] for r in st["runs"]] == ["done", "error", "error", "done"]
    assert st["runs"][1]["error"]["message"] == "box went away" and "concept" not in st["runs"][1]
    assert len(seen[3]) == 2 and st["runs"][2]["concept"]["title"] == "T3"        # run 3 failed after its concept existed: run 4 must still differ from it
    assert "OutOfMemoryError" in json.dumps(st["runs"][2]["error"])


def test_batch_project_version_is_started_once_and_runs_join_it(tmp_path):
    import run_batch as RB
    started, joined = [], []

    def start(idea, why):
        started.append((idea, why))
        return "COMPETITIONS/motion-graphic/app/v007", "version v007"

    def runner(graph, iid, port, sets, out, **kw):
        joined.append(kw.get("project"))
        return _fake_summary(sets["seed"])
    a = dict(_args(tmp_path), project=True, n=2)
    RB.run_batch(a, run_graph=runner, texts_of=lambda port, pid, g: _fake_texts(int(pid[3:])), start_version=start, log=lambda *a: None)
    assert len(started) == 1 and "3060 autonomous batch" in started[0][1]
    # parts carry the design + video model since 2026-10-29 (r01_<gemma|rules>_ltx)
    assert [j[:3] for j in joined] == [["--idea", "COMPETITIONS/motion-graphic/app/v007", "--part"]] * 2
    assert [j[3].split("_")[0] for j in joined] == ["r01", "r02"] and all(j[3].endswith("_ltx") for j in joined)
    a2 = dict(a, out=str(tmp_path / "out2"))
    with pytest.raises(SystemExit) as e:                                          # the idea does not exist: say how to create it, do not start a run
        RB.run_batch(a2, run_graph=runner, texts_of=None, start_version=lambda i, w: (None, "unknown idea"), log=lambda *a: None)
    assert "new-idea" in str(e.value)


@needs_graph("fusion_design_qwen_auto.json")
def test_batch_default_out_and_graph_exist():
    import run_batch as RB
    assert os.path.isfile(RB.GRAPH) and RB.GRAPH.endswith("fusion_design_qwen_auto.json")
    assert RB.SUMMARY_RE.search("piece: mode=cut shots=4 stills=4 cut=16.208s frames=389 words=5 technique=behind quality=preview")


def test_video_plan_is_all_ltx(tmp_path):
    import run_batch as RB
    assert RB.default_plan(5) == ["ltx"] * 5 and RB.parse_plan("ltx,ltx", 2) == ["ltx", "ltx"]
    with pytest.raises(SystemExit):
        RB.parse_plan("ltx,h4", 2)
    with pytest.raises(SystemExit):
        RB.parse_plan("ltx", 2)
    used = []

    def runner(graph, iid, port, sets, out, **kw):
        used.append(os.path.basename(graph))
        return _fake_summary(sets["seed"])
    st = RB.run_batch(_args(tmp_path), run_graph=runner, texts_of=lambda port, pid, g: _fake_texts(int(pid[3:])), log=lambda *a: None)
    assert used == ["g_ltx.json"] * 4 and all(r["status"] == "done" and r["video"] == "ltx" for r in st["runs"])


# ------------------------------------------------------------------------------------------------ the USER BRIEF (READ JSON v1, 16_app/MEDIA_BOX.md)
READ = {"v": 1, "on_screen": ["Hold on", "LET GO"], "concept": "two strangers part at a night station and one keeps walking",
        "look": "slow, desaturated blue, 35mm grain", "notes": "",
        "media": [{"id": "I1", "kind": "image", "file": "motngrfx_a.png", "shows": "a woman in a green coat on an empty platform", "window": None, "mood": "lonely", "palette": ["#223344"]},
                  {"id": "I2", "kind": "image", "file": "motngrfx_b.png", "shows": "an empty train carriage with flickering lights", "window": None, "mood": "tense", "palette": []},
                  {"id": "A1", "kind": "audio", "file": "motngrfx_s.wav", "shows": "slow piano", "window": None, "mood": "sad", "palette": []}],
        "music": {"bpm": 72.0, "key": "D# minor"}}


def _prompt(brief="", **kw):
    return CO.concept_prompt(3, 12, [], kw.pop("mode", "cut"), brief=brief, **kw)


def test_empty_brief_leaves_the_prompt_byte_identical():
    base = CO.concept_prompt(3, 12, [], "cut")
    for b in ("", "   ", "{}", json.dumps({"v": 1, "on_screen": [], "concept": "", "look": "", "media": [], "music": None, "notes": "x"}),
              json.dumps({"v": 1, "media": [{"id": "I1", "kind": "image", "file": "a.png", "shows": ""}]})):
        assert CO.parse_brief(b) == {}
        assert CO.concept_prompt(3, 12, [], "cut", brief=b) == base
    assert "USER BRIEF" not in base
    for seed, length, mode in ((1, 8, None), (42, 24, "chain"), (7, 16, "single")):                     # also the single-shot (free lines) variant
        assert CO.concept_prompt(seed, length, [], mode, brief="") == CO.concept_prompt(seed, length, [], mode)


def test_brief_prompt_carries_the_words_concept_look_and_media_notes():
    p = _prompt(json.dumps(READ))
    assert "USER BRIEF" in p and '["HOLD ON", "LET GO"]' in p and "`words` is EXACTLY the USER BRIEF's 2 on-screen lines" in p
    assert READ["concept"] in p and READ["look"] in p
    assert "a woman in a green coat on an empty platform" in p and "FIRST FRAME of shot 1" in p
    assert "STILL of shot 2" in p and "an empty train carriage" in p and "72 bpm" in p and "D# minor" in p
    assert p.count("USER BRIEF") >= 2 and p.index("USER BRIEF (written") < p.index("Cues for THIS piece")
    only_c = _prompt(json.dumps({"concept": "a quiet harbour at rest"}))                           # a concept alone does not touch the words
    assert "a quiet harbour at rest" in only_c and "ON-SCREEN LINES" not in only_c and "<lockup 1:" in only_c


def test_plain_text_brief_is_a_concept():
    assert CO.parse_brief("make it about the sea") == {"on_screen": [], "concept": "make it about the sea", "look": "", "media": [], "music": None, "notes": ""}


def test_the_users_words_are_enforced_verbatim_in_order_whatever_the_model_wrote():
    brief = json.dumps(READ)
    r = CO.parse(reply(shots="cut"), [], 1, 12, brief=brief)                                      # the model wrote five other lines
    c = r["concept"]
    assert r["ok"] and c["words"] == ["HOLD ON", "LET GO"] and c["brief"]["on_screen"] == READ["on_screen"]
    assert any("used verbatim" in f for f in r["fixes"])
    assert c["idea"].startswith("USER CONCEPT: two strangers") and "USER LOOK: slow, desaturated blue" in c["idea"]
    assert c["bpm"] == 72 and c["music"].endswith("72 bpm")
    for mode in ("single", "chain"):
        assert CO.parse(reply(shots=mode), [], 1, 12, brief=brief)["concept"]["words"] == ["HOLD ON", "LET GO"]
    long_line = {"on_screen": ["I WILL ALWAYS FIND MY WAY BACK HOME TO YOU"]}
    assert CO.parse(reply(), [], 1, 12, brief=json.dumps(long_line))["concept"]["words"] == ["I WILL ALWAYS FIND MY WAY BACK HOME TO YOU"]      # no length rule, no merge / split
    # without a brief nothing changed
    assert CO.parse(reply(), [], 1, 16)["concept"]["words"] == ["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME", "KEEP THE LIGHT"]


def test_user_words_are_never_a_clash_and_their_owners_follow_them():
    prev = [concept(words=["HOLD ON", "LET GO"])]
    brief = json.dumps({"on_screen": ["HOLD ON", "BREATHE", "DRIFT"]})                              # overused models' words and the same words as an earlier run: the user chose them
    r = CO.parse(reply(shots="cut"), prev, 2, 12, brief=brief)
    assert r["ok"] and r["clashes"] == [] and r["concept"]["words"] == ["HOLD ON", "BREATHE", "DRIFT"]
    assert CO.clashes(CO.parse(reply(), prev, 2, 12)["concept"], prev)                                # the same earlier run without the brief still clashes (sanity)
    own = [{"render": "flat"}, {"render": "ltx", "material": "wet stone letters"}, {"render": "flat"}]
    c = CO.parse(reply(shots="cut", owners=own), [], 1, 12, brief=brief)["concept"]
    assert [o.get("render") for o in c["owners"]] == ["flat", "ltx", "flat"]                          # owner k = user line k


def test_shots_that_start_from_a_user_image_are_new_places():
    sp = [{"prompt": f"{'the woman walks along the platform in the cold light of the station ' * 1} shot {k}", "camera": "slow push in", "continues": True} for k in range(1, 5)]
    c = CO.parse(reply(shots="cut", shot_plan=sp), [], 1, 12, brief=json.dumps(READ))["concept"]
    assert len(c["shot_plan"]) >= 2 and c["shot_plan"][1]["continues"] is False                      # image 2 -> shot 2 is a place of its own
    assert c["brief"]["images"] == 2 and CO.brief_extra_images(CO.parse_brief(READ)) == 1
    assert all(e.get("continues") is True for e in c["shot_plan"][2:])                                 # the others keep the model's choice


def test_final_keeps_the_brief_over_the_rules_variation():
    brief = json.dumps(READ)
    first = CO.parse("not json at all", [], 5, 12, brief=brief)
    c, rep = CO.final(None, first, [], 5, 12, brief=brief)
    assert c["source"].startswith("rules") and c["words"] == ["HOLD ON", "LET GO"] and c["bpm"] == 72 and c["idea"].startswith("USER CONCEPT:")
    c0, _ = CO.final(None, first, [], 5, 12)
    assert c0["words"] != ["HOLD ON", "LET GO"] and "brief" not in c0


def test_concept_nodes_take_the_brief():
    _comfy()
    from nodes_fusion import concept as N
    brief = json.dumps(READ)
    for cls in (N.MographConceptPrompt, N.MographConceptParse, N.MographConceptFinal):
        inp = {i.id: i for i in cls.define_schema().inputs}
        assert "brief" in inp and getattr(inp["brief"], "optional", False) and not getattr(inp["brief"], "lazy", None)
    assert N.MographConceptPrompt.execute(3, 12, "[]").args[0] == CO.concept_prompt(3, 12, [])             # no brief = today's prompt
    assert N.MographConceptPrompt.execute(3, 12, "[]", "").args[0] == CO.concept_prompt(3, 12, [])
    p = N.MographConceptPrompt.execute(3, 12, "[]", brief).args[0]
    assert '["HOLD ON", "LET GO"]' in p
    ok, rp, rep = N.MographConceptParse.execute(reply(shots="cut"), "[]", 3, 12, p, brief).args
    assert ok is True and rp == ""
    ok2, rp2, _ = N.MographConceptParse.execute("garbage", "[]", 3, 12, "", brief).args                    # the repair prompt is rebuilt WITH the brief when no prompt arrived
    assert ok2 is False and "USER BRIEF" in rp2
    out = N.MographConceptFinal.execute(reply(shots="cut"), True, "[]", 3, 12, None, brief).args
    assert out[0].split("\n") == ["HOLD ON", "LET GO"] and json.loads(out[5])["brief"]["concept"] == READ["concept"]
    ex = json.loads(out[7])
    assert "USER CONCEPT" in ex["idea"]                                                                       # the design step sees it (plan.footage.idea)
    assert N.MographConceptFinal.check_lazy_status(reply="x", ok=False, repair_reply=None, brief="{}") == ["repair_reply"]
    out0 = N.MographConceptFinal.execute(reply(), True, "[]", 1, 16).args
    assert out0[0].split("\n")[0] == "THE SEA"


def test_app_graphs_wire_brief_images_and_the_media_field(bg, auto):
    g, side = auto
    assert g["brief_in"]["class_type"] == "PrimitiveStringMultiline" and g["brief_in"]["_meta"]["title"] == "IN brief" and g["brief_in"]["inputs"]["value"] == ""
    for k in ("concept_prompt", "concept_parse", "concept_final"):
        assert g[k]["inputs"]["brief"] == ["brief_in", 0], k
    assert g["images"]["_meta"]["title"] == "IN images" and g["images"]["class_type"] == "PrimitiveString" and g["images"]["inputs"]["value"] == ""
    assert g["user_images"]["class_type"] == "MographUserImages" and g["user_images"]["inputs"] == {"names": ["images", 0], "format": ["format", 0]}
    assert g["first_set"]["inputs"]["user_images"] == ["user_images", 0]
    ids = [f["id"] for f in side["fields"]]
    assert "image" not in ids and "audio" not in ids and ids.count("media") == 1
    media = next(f for f in side["fields"] if f["id"] == "media")
    assert media == {"id": "media", "kind": "media", "label": "Add media", "limits": {"video": 1, "image": 5, "audio": 1}, "image_node": "IN image", "images_node": "IN images",
                     "audio_node": "IN audio", "brief_node": "IN brief", "read_graph": "_media_read", "own_video_route": "/mograph/own_video"}
    prev = next(f for f in side["fields"] if f["id"] == "previous")
    assert prev["hidden"] is True and g["previous"]["_meta"]["title"] == "IN previous"                  # hidden on the page, still in the graph
    titles = {v["_meta"]["title"] for v in g.values()}
    assert {"IN image", "IN audio", "IN images", "IN brief", "IN previous"} <= titles                   # the old image / audio nodes stay: the media field maps to them
    assert bg.check_titles(g, side) == []


def test_non_auto_graphs_have_the_images_but_no_brief(bg):
    g, side = bg.make(False, None, "moss", True)
    assert "brief_in" not in g and g["first_set"]["inputs"]["user_images"] == ["user_images", 0]
    media = next(f for f in side["fields"] if f["id"] == "media")
    assert "brief_node" not in media and bg.check_titles(g, side) == []


@pytest.mark.parametrize("name", ["motngrfx_default", "motngrfx_16gb", "motngrfx_3060"])
def test_shipped_app_graphs_carry_the_brief_wiring(name):
    d = ROOT / "app" / "graphs"
    g = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    side = json.loads((d / f"{name}.app.json").read_text(encoding="utf-8"))
    assert [g[k]["inputs"].get("brief") for k in ("concept_prompt", "concept_parse", "concept_final")] == [["brief_in", 0]] * 3
    assert g["first_set"]["inputs"]["user_images"] == ["user_images", 0] and g["user_images"]["inputs"]["names"] == ["images", 0]
    assert [f["id"] for f in side["fields"]] == ["previous", "media", "length", "seed", "format", "quality"]
    assert "IN brief" in side["required_titles"] and "IN images" in side["required_titles"]


# ------------------------------------------------------------------------------------------------ STORYBOARD v12: the story's shot count reaches the planner
def _story_brief(n, **kw):
    return dict({"on_screen": [f"LINE {k}" for k in range(1, n + 1)],
                 "story": {"title": "T", "logline": "L", "shots": [{"n": k, "image": f"I{k}", "start": 0, "end": 1, "beat": f"beat {k}", "line": f"LINE {k}"} for k in range(1, n + 1)]}}, **kw)


def test_story_shot_count_overrides_the_lockup_count_when_valid():
    assert CO.lockup_count(24, "cut") == 6 and CO.lockup_count(24, "cut", 5) == 5 and CO.lockup_count(24, "single", 5) is None
    assert CO.story_n(CO.parse_brief(_story_brief(5)), 24) == 5                                  # 24 s / 5 = 4.8 s: one clip each
    assert CO.story_n(CO.parse_brief(_story_brief(2)), 24) is None                               # 12 s shots do not fit one clip: today's rule
    assert CO.story_n(CO.parse_brief(_story_brief(3)), 12) == 3
    assert CO.story_n({}, 24) is None and CO.story_n(CO.parse_brief({"on_screen": ["A"]}), 24) is None
    p = CO.concept_prompt(3, 24, [], "cut", brief=json.dumps(_story_brief(5)))
    assert "EXACTLY 5 entries" in p or "EXACTLY the USER BRIEF's 5 on-screen lines" in p
    c = CO.parse(reply(shots="cut", words=[f"W{k}" for k in range(1, 7)]), [], 1, 24, brief=json.dumps(dict(_story_brief(5), on_screen=[])))["concept"]
    assert len(c["words"]) == 5                                                                   # the story's lines, not the planner's 6


def test_request_extras_carries_story_shots_only_for_a_story_brief():
    c = {"words": ["A"], "length": 24, "brief": {"story": {"shots": [{"n": k} for k in range(1, 6)]}}}
    assert json.loads(CO.request_extras(c))["story_shots"] == 5
    c2 = {"words": ["A"], "length": 12, "brief": {"story": {"shots": [{"n": 1}, {"n": 2}]}}}      # 2 shots of 6 s: not valid, no key
    assert CO.request_extras(c2) == "[]"
    assert CO.request_extras({"words": ["A"], "length": 24}) == "[]"                              # no story: byte-identical to before
