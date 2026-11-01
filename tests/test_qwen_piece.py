"""Tests for tools/fusion/qwen_piece.py. No network: the model is a fake chat function."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fusion import qwen_piece as Q  # noqa: E402

ROOT = Q.ROOT


def needs(*rels):
    """skip when a dev fixture is absent (15_fusion/ is not shipped in the public export)."""
    missing = [r for r in rels if not (ROOT / r).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"dev fixture not shipped in the public export: {missing[0] if missing else ''}")


P06_ANA = "15_fusion/v003/local/pieces/P06/analysis.json"
P10_ANA = "15_fusion/v002/local/pieces/P10/analysis.json"


HITS = {"P06": ([0.75, 3.75, 6.75, 9.75, 12.75, 15.75], 1.3), "P10": ([1.76, 7.07, 12.38], 2.4)}


def good_reply(name="P06"):
    """a reply that follows the schema: hits on grid beats, holds of 1.6 s / 2.4 s, a compose block."""
    orig = Q.load_original(name)
    fonts = ["Anybody[wdth,wght].ttf", "BigShouldersDisplay[wght].ttf"]
    hits, hold = HITS[name]
    words = []
    for i, w in enumerate(orig["type"]["words"]):
        words.append({"text": w["text"], "font": fonts[i % 2], "module": "lanes", "hit": hits[i], "hold_s": hold, "form_in": "lines", "form_out": "lines",
                      "form_s": 0.5, "exit_s": 0.5, "placement": {"yaw": 5, "wf": 0.9, "hf": 0.5}})
    return {"concept": "one idea", "treatment_notes": "behind the subject",
            "type": {"mood": "driving", "energy": 0.8, "palette": {"bg": [5, 5, 9], "fg": [240, 240, 245], "accents": [[90, 170, 255], [255, 170, 60], [170, 120, 255]]},
                     "camera": "beat_orbit", "camera_params": {"read_dps": 5}, "transitions": "auto", "line": {"px": 4}, "words": words},
            "compose": {"technique": "behind", "plate_gain": 0.7, "glow_lo": 0.05, "glow_hi": 0.3, "grain": 0.02, "holds": [[5.0, 0.4]]}}


def test_extract_json_fence_and_prose():
    assert Q.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert Q.extract_json('Here you go: {"a": {"b": [1, 2]}} done') == {"a": {"b": [1, 2]}}
    assert Q.extract_json("no json here") is None


@needs(P06_ANA)
def test_normalize_clamps_and_text_lock():
    r = good_reply()
    r["type"]["energy"] = 3
    r["type"]["words"][0]["placement"]["wf"] = 2.0
    r["compose"]["grain"] = 0.5
    d, fixes, errors = Q.normalize_design(r, "P06", diversify=False)           # the clamps; the font fixes have their own tests (test_design_nodes.py)
    assert errors == []
    assert d["type"]["energy"] == 1.0 and d["type"]["words"][0]["placement"]["wf"] == 1.0 and d["compose"]["grain"] == 0.06
    assert len(fixes) == 3
    r2 = good_reply()
    r2["type"]["words"][1]["text"] = "SOMETHING ELSE"
    _, _, errors = Q.normalize_design(r2, "P06")
    assert any("must be 'SEARCHING'" in e for e in errors)


@needs(P06_ANA)
def test_font_not_liked_and_module_checked():
    r = good_reply()
    r["type"]["words"][0]["font"] = "Unbounded-Regular.ttf"
    r["type"]["words"][1]["module"] = "sparkle"
    _, _, errors = Q.normalize_design(r, "P06")
    assert any("not in the liked font list" in e for e in errors)
    assert any("sparkle" in e for e in errors)


@needs(P06_ANA)
def test_schema_word_count_transitions_and_overlap():
    r = good_reply()
    r["type"]["words"].pop()
    r["type"]["transitions"] = ["twist", "twist", "cut", "cut", "cut"]
    r["type"]["words"][1]["hold_s"] = 0.9
    _, _, errors = Q.normalize_design(r, "P06")
    assert any("exactly 6 entries" in e for e in errors)
    assert any("used twice" in e for e in errors)
    assert any("must hold at least" in e for e in errors)


@needs(P06_ANA)
def test_lines_must_spell_the_word():
    r = good_reply()
    r["type"]["words"][2]["lines"] = ["A", "LIG", "XX"]
    _, _, errors = Q.normalize_design(r, "P06")
    assert any("must spell" in e for e in errors)
    r = good_reply()
    r["type"]["words"][2]["lines"] = ["A", "LIG", "HT"]
    _, _, errors = Q.normalize_design(r, "P06")
    assert errors == []


@needs(P06_ANA)
def test_axes_limited_to_the_fonts_axes():
    r = good_reply()
    r["type"]["words"][0]["axes"] = {"wght": [50, 2000, "beat"], "ELSH": 40}
    d, fixes, errors = Q.normalize_design(r, "P06", diversify=False)
    assert errors == []
    assert d["type"]["words"][0]["axes"] == {"wght": [100, 900, "beat"]}
    assert any("ELSH" in f for f in fixes)


@pytest.mark.parametrize("name,version", [pytest.param("P06", "v004", marks=needs(P06_ANA)), pytest.param("P10", "v003", marks=needs(P10_ANA))])
def test_merge_locked_fields_and_media(name, version):
    orig = Q.load_original(name)
    d, _, errors = Q.normalize_design(good_reply(name), name)
    assert errors == []
    d["type"]["words"][0]["matte"] = "front"
    p = Q.merge_locked(name, d, version, "Fake Model", 1)
    assert Q.locked_diff(name, p) == []
    assert p["id"] == f"{name}_{version}" and p["version"] == version and "Fake Model" in p["title"] and p["slug"] == orig["slug"]
    assert p["parent"]["piece"] == name and p["parent"]["version"] == Q.PIECES[name]["parent_version"]
    assert "audio" in p["parent"]["reuse"] and "shots" in p["parent"]["reuse"]
    assert p["signoff"]["off"] is True
    assert all("media" in s and "shot" in s["media"] and "matte" in s["media"] for s in p["shots"])
    assert p["compose"]["size"] == orig["compose"]["size"] and p["compose"]["crossfade"] == orig["compose"]["crossfade"]
    ops = [o["op"] for o in p["compose"]["default"]["ops"]]
    assert "sandwich" in ops and ops.count("grade") >= 2 and "flow_wake" not in ops
    assert p["compose"]["holds"] == [[5.0, 0.4]]
    assert p["type"]["words"][0]["matte"] == "front" and p["plan"]["matte_treatments"]["words"] == 1
    # the model cannot move the locked shot fields: shots / stills identical apart from media
    strip = lambda L: [{k: v for k, v in x.items() if k != "media"} for x in L]       # noqa: E731
    assert strip(p["shots"]) == strip(orig["shots"])


@needs(P06_ANA)
def test_compose_values_are_patched_into_the_recipe():
    d, _, _ = Q.normalize_design(good_reply(), "P06")
    d["compose"].update(plate_gain=0.55, plate_sat=0.2, vignette=0.4, type_opacity=0.8)
    comp = Q.build_compose(Q.load_original("P06"), d, Q.load_analysis("P06"))
    ops = {(o["op"], o.get("out")): o["args"] for o in comp["default"]["ops"]}
    assert ops[("grade", "foot")]["gain"] == 0.55 and ops[("grade", "foot")]["sat"] == 0.2
    assert ops[("vignette", "image")]["strength"] == 0.4
    assert ops[("sandwich", "image")]["type_opacity"] == 0.8


@needs(P06_ANA, P10_ANA)
def test_prompt_has_the_inputs_and_does_not_quote_the_originals():
    for name in ("P06", "P10"):
        prompt = Q.build_prompt(name, 1)
        orig = Q.load_original(name)
        for w in Q.words_of(orig):
            assert w in prompt
        for must in ("DESIGN GUIDE", "MUSIC ANALYSIS", "SHOTS", "FONTS", "OUTPUT SCHEMA", "downbeats", "dropouts", "chord"):
            assert must in prompt
        assert "Design seed: 1" in prompt
        assert "BadeenDisplay-Regular.ttf" in prompt                      # the liked list is complete
        assert "sk-or" not in prompt
    guide = Q.GUIDE.read_text(encoding="utf-8")
    assert len(guide.splitlines()) < 330          # 246 lines before the r8 defaults (sections 20-21, 2026-10-29); sections 22-23 (piece system, plates) came with the diversity round
    for quoted in ("Rock 3D", "Rock3D", "ELSH", "Song", "Noir Step", "knockout windows"):
        assert quoted not in guide


@needs(P10_ANA)
def test_package_images(tmp_path):
    pkg = Q.build_package("P10", 4, tmp_path)
    assert len(pkg["images"]) == 4 and all(os.path.getsize(p) > 5000 for p in pkg["images"])
    assert (tmp_path / "package_P10_seed4.txt").read_text(encoding="utf-8") == pkg["prompt"]


@needs(P10_ANA)
def test_dropouts_and_chords():
    ana = Q.load_analysis("P10")
    d = Q.dropouts(ana)
    assert any(abs(a - 4.83) < 0.15 for a, _ in d) and any(abs(a - 10.12) < 0.15 for a, _ in d)
    cb = Q.chord_blocks(ana)
    assert cb[0][2] == "B" and cb[-1][1] == 15.25


@needs(P06_ANA)
def test_repair_loop_same_model_and_cost(tmp_path):
    """turn 0: bad font; turn 1: valid. The repair turn goes to the same chat function with the errors in it; costs add up; no network."""
    calls = []
    bad = good_reply()
    bad["type"]["words"][0]["font"] = "Comic Sans"
    replies = [json.dumps(bad), json.dumps(good_reply())]

    def fake(msgs):
        calls.append(copy.deepcopy(msgs))
        return replies[len(calls) - 1], dict(tin=100, tout=50, cost=0.001, secs=0.1, model="fake")

    pkg = {"prompt": "P", "images": []}
    res = Q.design_once("P06", "fake/model", 1, 1, tmp_path, "t", package=pkg, chat_fn=fake)
    assert res["valid"] and res["repairs"] == 1 and len(calls) == 2
    assert calls[1][-1]["role"] == "user" and "liked font list" in calls[1][-1]["content"]
    assert calls[1][-2]["role"] == "assistant"
    assert res["cost"] == pytest.approx(0.002)
    assert (tmp_path / "t_raw_turn0.txt").exists() and (tmp_path / "t_raw_turn1.txt").exists()


@needs(P06_ANA)
def test_repair_gives_up_after_two_turns(tmp_path):
    def fake(msgs):
        return "not json at all", dict(tin=1, tout=1, cost=0.0001, secs=0.0, model="fake")

    res = Q.design_once("P06", "fake/model", 1, 1, tmp_path, "u", package={"prompt": "P", "images": []}, chat_fn=fake)
    assert not res["valid"] and res["repairs"] == 2 and res["design"] is None


@needs(P10_ANA)
def test_seed_state_covers_every_adopted_kind(tmp_path, monkeypatch):
    orig = Q.load_original("P10")
    d, _, _ = Q.normalize_design(good_reply("P10"), "P10")
    p = Q.merge_locked("P10", d, "v003", "Fake", 1)
    pj = tmp_path / "P10_v003.json"
    pj.write_text(json.dumps(p), encoding="utf-8")
    assert Q.media_check(pj) == []
    monkeypatch.setattr(Q, "ROOT", ROOT)
    sp = Q.seed_state(pj, "v006_test_tmp", log=lambda *_: None)
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
        keys = set(st["items"])
        assert {"audio", "still:A", "shot:s1", "shot:s2", "shot:s3", "alpha:s1", "alpha:s2", "alpha:s3"} <= keys
        assert not any(k.startswith(("type", "compose", "assemble", "final")) for k in keys)
        with pytest.raises(FileExistsError):
            Q.seed_state(pj, "v006_test_tmp", log=lambda *_: None)
    finally:
        import shutil
        shutil.rmtree(ROOT / "15_fusion" / "v006_test_tmp", ignore_errors=True)
    assert orig["shots"][0]["id"] == "s1"
