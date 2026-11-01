"""ONE OWNER PER WORD (tools/fusion/owner.py, 2026-10-28): a word is drawn once. Fake replies and tiny synthetic frames only: no model, no network, no GPU.
Covers: the owner fields and their salvage (invalid -> nearest allowed, missing -> the rules' default), the control pass (only what LTX owns, guide 0 for an
all-flat shot), the flat alpha timeline of a handoff in compose, words == shots, an old piece without owner fields composing exactly as before, the shot
prompt clause and clean_prompt.
Run: python -m pytest tests/test_owner.py -q"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket",
       "music": "calm piano", "length": 16, "look": "auto", "seed": 5, "quality": "preview", "format": "9:16"}


@pytest.fixture(scope="module")
def music():
    return synth_music(120, 16)


REQ_OWNED = dict(REQ, owners=json.dumps({"owners": [{"render": "handoff"}] * 6}))        # the concept asked for LTX on every word (the default is flat since 2026-10-29)


@pytest.fixture(scope="module")
def plan(music):
    from unittest import mock
    from tools.fusion import letter_fill as LF
    with mock.patch.object(LF, "solid", lambda piece: False):                         # 2026-11-01 letter_fill: LTX material letters only when the user asks for textured letters
        return pr.plan_piece(REQ_OWNED, music)


def _strip_owner(piece):
    """an OLD plan: no word carries an owner field, the shots keep the plan's prompts without the clause and no control guide."""
    p = copy.deepcopy(piece)
    for w in p["type"]["words"]:
        for k in ("render", "ltx_at", "ltx_t", "material"):
            w.pop(k, None)
    for s in p["shots"]:
        s["prompt"] = OW.strip_clause(s["prompt"])
        s["control"].pop("guide", None)
    return p


# ------------------------------------------------------------------------------------------------ the fields and their salvage
def test_norm_render_nearest_allowed():
    assert OW.norm_render("handoff") == ("handoff", "ok")
    assert OW.norm_render("Hand-off") == ("handoff", "fix")
    assert OW.norm_render("ltxx") == ("ltx", "fix")
    assert OW.norm_render("static") == ("flat", "fix")
    assert OW.norm_render("banana", "handoff") == ("handoff", None)
    assert OW.norm_render(None, "handoff") == ("handoff", None)


def test_ltx_at_is_clamped_and_material_is_trimmed():
    assert OW.clamp_ltx_at(0.5) == (0.5, "ok")
    assert OW.clamp_ltx_at(0.95) == (0.8, "fix") and OW.clamp_ltx_at(0.05) == (0.2, "fix")
    assert OW.clamp_ltx_at("50%")[0] == 0.5 and OW.clamp_ltx_at("abc") == (None, None)
    m, how = OW.clean_material("thick carved ice extruded with frosted edges and a glow on top letters")
    assert len(m.split()) == OW.MATERIAL_MAX_WORDS and how == "fix" and not m.endswith("letters")


def test_default_material_follows_the_scene_then_the_mood():
    assert "ice" in OW.default_material("calm", {"fg": [235, 235, 235]}, "a wide frozen lake under pale winter sun")
    assert "neon" in OW.default_material("techno", None, "")
    for scene in ("a harbour at night", "a desert salt flat", "a foggy pine forest", ""):
        assert 2 <= len(OW.default_material("dark", {"accents": [[200, 40, 40]]}, scene).split()) <= OW.MATERIAL_MAX_WORDS


def test_default_owner_is_flat(music):
    """2026-10-29 (clear text first): a plan whose concept chose no owner draws every word as the flat graphic; an LTX word only where the design asks for it."""
    p = pr.plan_piece(REQ, music)
    assert OW.DEFAULT_RENDER == "flat" and all(w["render"] == "flat" and "material" not in w for w in p["type"]["words"])
    assert all(s["control"].get("guide", 0.0) == OW.OFF_GUIDE for s in p["shots"])                      # all-flat shots: plain image-to-video, no type control pass
    assert OW.norm_render(None, OW.DEFAULT_RENDER) == ("flat", None) and OW.word_fields({})["render"] == "flat"
    assert [e["render"] for e in OW.effective([{"text": "A", "render": "ltx"}, {"text": "B"}])] == ["ltx", "flat"]


def test_plan_words_carry_owner_fields(plan):
    for w in plan["type"]["words"]:
        assert w["render"] == "handoff" and w["ltx_at"] == 0.5 and 2 <= len(w["material"].split()) <= OW.MATERIAL_MAX_WORDS


def test_normalize_design_mends_owner_fields_alone(plan, music):
    """invalid render -> the nearest allowed name (else the default), missing material -> the rules' default, ltx_at clamped; every other field is kept."""
    old = _strip_owner(plan)
    reply = DS.reply_from_piece(old, music)
    ws = reply["type"]["words"]
    ws[0].update(render="Hand-off", ltx_at=0.95, material="thick carved ice, extruded, frosted edges")
    ws[1].update(render="banana")
    ws[2].update(render="ltx")
    ws[2].pop("material", None)                                       # no material
    ws[3].update(render="flat", material="ignored on a flat word")
    for w in ws[:4]:
        w.pop("ltx_t", None)
    locked = json.dumps(DS.make_locked(old))
    out = DS.parse(json.dumps(reply), locked, old, music, "P" * 500)
    got = json.loads(out["piece_json"])["type"]["words"]
    assert got[0]["render"] == "handoff" and got[0]["ltx_at"] == 0.8 and got[0]["material"] == "thick carved ice, extruded, frosted edges"
    assert got[1]["render"] == "flat" and "ltx_at" not in got[1] and "material" not in got[1]                    # the rules' default (flat since 2026-10-29), never an error
    assert got[2]["render"] == "ltx" and "ltx_at" not in got[2] and 2 <= len(got[2]["material"].split()) <= OW.MATERIAL_MAX_WORDS
    assert got[3]["render"] == "flat" and "material" not in got[3]
    f = out["fields"]
    by = {x["path"]: x["why"] for x in f["fields_fixed"]}
    assert "type.words[0].render" in by and "type.words[0].ltx_at" in by
    assert "type.words[1].render" in {x["path"] for x in f["fields_from_rules"]}
    assert "type.words[2].material" not in {x["path"] for x in f["fields_from_rules"]}, "a field the reply does not mention is a default, not counted against the model"
    assert "type.words[0].material" in f["fields_from_model"]
    # valid fields of the same words are untouched (the font rules may swap a face to reach three faces; the module and the io styles stay)
    for i in range(4):
        assert got[i]["module"] == ws[i]["module"]                           # (the io styles may be re-assigned so that no two words enter the same way, openings.enforce; logged)


def test_owner_is_locked_to_the_generated_footage(plan, music):
    """the shots exist before the design step: where the PLAN words carry an owner, the designed word keeps it (and the absolute handoff time)."""
    reply = DS.reply_from_piece(plan, music)
    reply["type"]["words"][0].update(render="flat")
    reply["type"]["words"][1].update(render="ltx", material="chrome")
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "P" * 500)
    got = json.loads(out["piece_json"])["type"]["words"]
    assert got[0]["render"] == "handoff" and got[0]["material"] == plan["type"]["words"][0]["material"] and got[0]["ltx_t"] > 0
    assert got[1]["render"] == "handoff"
    assert any("the owner cannot change after the footage exists" in x for x in out["report"].splitlines())


def test_qwen_schema_documents_the_owner_fields():
    txt = QP.schema_text(["A", "B"], 10.0)
    for k in ('"render"', '"ltx_at"', '"material"'):
        assert k in txt
    for k in ("render", "ltx_at", "material"):
        assert k in QP.WORD_KEYS
    guide = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    assert "handoff" in guide and "ltx_at" in guide and "material" in guide


# ------------------------------------------------------------------------------------------------ windows and gains
def _piece_words(*specs):
    ws = []
    for t0, t1, r, la in specs:
        w = {"text": f"W{len(ws)}", "t0": t0, "t1": t1}
        if r:
            w["render"] = r
        if la is not None:
            w["ltx_at"] = la
        ws.append(w)
    return {"fps": 24, "type": {"words": ws}}


def test_old_piece_has_no_windows_and_everything_stays():
    p = _piece_words((0.0, 2.0, None, None), (2.5, 5.0, None, None))
    assert OW.windows(p) is None and OW.flat_gain(None, 1.0) == 1.0 and OW.control_gain(None, 1.0) == 1.0
    assert OW.control_gains(p, 0, 10) is None
    assert OW.slot_plan(p, None, 0, 48, 48, "a lake, cinematic", 0.4) == (None, 0.4, "a lake, cinematic")


def test_flat_alpha_timeline_of_a_handoff():
    wins = OW.windows(_piece_words((0.0, 4.0, "handoff", 0.5)))
    assert OW.flat_gain(wins, 0.5) == 1.0 and OW.flat_gain(wins, 1.99) == 1.0           # 1 before ltx_at (2.0 s)
    assert 0.0 < OW.flat_gain(wins, 2.1) < 1.0                                         # crossfade
    assert OW.flat_gain(wins, 2.0 + OW.FLAT_FADE_S) == pytest.approx(0.0, abs=1e-6)    # 0 after the ~0.25 s ramp
    assert OW.flat_gain(wins, 3.5) == 0.0
    flat = OW.windows(_piece_words((0.0, 4.0, "flat", None)))
    ltx = OW.windows(_piece_words((0.0, 4.0, "ltx", None)))
    assert [OW.flat_gain(flat, t) for t in (0.0, 2.0, 3.9)] == [1.0] * 3
    assert [OW.flat_gain(ltx, t) for t in (0.0, 2.0, 3.9)] == [0.0] * 3


def test_control_pass_membership_per_owner():
    """flat = never in the control pass, ltx = the whole span, handoff = from ltx_at (short ramp in); the transition after a word follows that word."""
    p = _piece_words((0.0, 2.0, "flat", None), (2.4, 4.4, "ltx", None), (4.8, 6.8, "handoff", 0.5))
    wins = OW.windows(p)
    g = lambda t: OW.control_gain(wins, t)          # noqa: E731
    assert g(0.5) == 0.0 and g(1.9) == 0.0 and g(2.2) == 0.0       # flat word and its transition
    assert g(2.4) == 1.0 and g(3.5) == 1.0 and g(4.6) == 1.0       # ltx word and its transition
    assert g(5.0) == 0.0 and g(5.8 - 1e-3) == pytest.approx(0.0, abs=1e-3)       # handoff not yet taken over (ltx_at = 5.8 s)
    assert 0.0 < g(5.9) < 1.0 and g(5.8 + OW.CONTROL_RAMP_S) == pytest.approx(1.0) and g(6.5) == 1.0
    gains = OW.control_gains(p, 0, 7 * 24)
    assert gains.shape == (168,) and gains[0] == 0.0 and gains[3 * 24] == 1.0 and gains[-1] == 1.0


def test_guide_is_zero_for_an_all_flat_shot_and_raised_for_owned_letters():
    p = _piece_words((0.0, 4.0, "flat", None), (4.5, 8.5, "handoff", 0.5))
    wins = OW.windows(p)
    assert OW.shot_guide(wins, 0.0, 4.2, 0.4) == OW.OFF_GUIDE == 0.0               # plain i2v, no IC guide
    assert OW.shot_guide(wins, 4.5, 8.5, 0.4) == OW.OWNED_GUIDE == 1.0
    assert OW.shot_guide(wins, 4.5, 8.5, 1.0) == 1.0                                # never lowered
    assert OW.shot_guide(None, 0.0, 4.0, 0.4) == 0.4


def test_slot_plan_gates_the_control_video_and_adds_the_clause():
    p = _piece_words((0.0, 4.0, "flat", None), (4.0, 8.0, "handoff", 0.5))
    p["type"]["words"][1].update(text="WIND WHISPERS", material="thick carved ice, extruded, frosted edges")
    g0, guide0, pr0 = OW.slot_plan(p, None, 0, 96, 96, "a lake, cinematic", 0.4)
    assert guide0 == 0.0 and not g0.any() and pr0 == "a lake, cinematic"
    g1, guide1, pr1 = OW.slot_plan(p, None, 96, 96, 96, "a lake, cinematic", 0.4)
    assert guide1 == 1.0 and g1[0] == 0.0 and g1[-1] == 1.0
    assert 'the words "WIND WHISPERS" formed as thick carved ice, extruded, frosted edges letters' in pr1 and pr1.endswith("cinematic")
    assert OW.slot_plan(p, None, 96, 96, 96, pr1, 0.4)[2] == pr1                    # idempotent


# ------------------------------------------------------------------------------------------------ words == shots
def test_words_equal_shots_and_start_with_their_shots(plan, music):
    lay = pr.layout(plan)
    words = plan["type"]["words"]
    assert len(words) == len(plan["shots"]) >= 3
    for w, s in zip(words, plan["shots"]):
        assert w["t0"] == pytest.approx(lay[s["id"]]["start"] / 24.0, abs=0.01)           # t0 = the shot start
    chk = pr.dry_run(plan)
    for w, s, c in zip(words, plan["shots"], chk["words"]):
        a = lay[s["id"]]["start"] / 24.0
        b = (lay[s["id"]]["start"] + int(s.get("span") or lay[s["id"]]["frames"])) / 24.0
        assert a - 0.05 <= c["start"] <= a + 0.6, "the word starts with its shot, never before it"
        assert c["hit"] < b and c["release"] <= b + 0.05, "its hit and hold are inside the shot"
    assert any("lockups" in n and "merged" in n for n in plan["plan"]["notes"])


def test_fit_lockups_merges_and_splits_deterministically():
    lines, note = pr.fit_lockups(["A B", "C", "D E F", "G", "H I", "J"], 4)
    assert len(lines) == 4 and " ".join(lines) == "A B C D E F G H I J" and "merged" in note
    lines, note = pr.fit_lockups(["AFTER HOURS", "THE CITY KEEPS DREAMING"], 3)
    assert len(lines) == 3 and " ".join(lines) == "AFTER HOURS THE CITY KEEPS DREAMING" and "split" in note
    lines, note = pr.fit_lockups(["A", "B"], 4)                         # single words cannot be split further: stays short, said so
    assert lines == ["A", "B"] and "only 2 lockups" in note
    assert pr.fit_lockups(["A", "B"], 2) == (["A", "B"], None)


def test_model_timing_is_fitted_inside_the_shot(plan, music):
    """the model's hit / hold / exit that do not fit shot k are moved to the nearest values that do, and each move is a fix."""
    old = _strip_owner(plan)
    reply = DS.reply_from_piece(old, music)
    for w in reply["type"]["words"]:
        w["hit"], w["hold_s"], w["form_s"], w["exit_s"] = 1.0, 5.5, 1.0, 1.4          # the same hit for every word, holds longer than any shot
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(old)), old, music, "P" * 500)
    p = json.loads(out["piece_json"])
    lay = pr.layout(p)
    chk = pr.dry_run(p)
    for s, c in zip(p["shots"], chk["words"]):
        a = lay[s["id"]]["start"] / 24.0
        b = a + int(s.get("span") or lay[s["id"]]["frames"]) / 24.0
        assert a - 0.3 <= c["start"] and a < c["hit"] < b and c["release"] <= b + 0.05, (s["id"], c["start"], c["hit"], c["release"])
    assert "lives in shot" in out["report"] or "must end before the cut" in out["report"]


# ------------------------------------------------------------------------------------------------ the prompt clause
def test_plan_prompts_name_the_owned_letters_and_clean_prompt_strips_them(plan):
    from tools.app import layered_shots as LS
    for w, s in zip(plan["type"]["words"], plan["shots"]):
        noun = "word" if len(w["text"].split()) == 1 else "words"
        look = OW.font_look(w["font"])                                       # 2026-10-29: the clause names the graphic's own look (the face) so the video model follows the control pass
        want = f'the {noun} "{w["text"].upper()}" in {look} capitals formed as {w["material"]} letters, extruded 3D lettering'
        assert want in s["prompt"]
        assert s["control"]["guide"] == 1.0 and s["guide"] == 1.0
        clean = LS.clean_prompt(s["prompt"])
        assert "the words" not in clean and "letters" not in clean and not clean.endswith(",")             # (no trailing "cinematic" tag any more, diversity round)
        assert "the subject stands perfectly still while the air moves around it" in clean      # only the letter clause goes
    assert LS.clean_prompt("a lake, long flowing strands draw out and trace the lines, calm") == "a lake, calm"


def test_all_flat_plan_gets_guide_zero_and_no_clause(plan):
    p = copy.deepcopy(plan)
    for w in p["type"]["words"]:
        w["render"] = "flat"
        w.pop("material", None)
    for s in p["shots"]:
        s["prompt"] = OW.strip_clause(s["prompt"])
    OW.apply_to_plan(p)
    assert all(s["control"]["guide"] == 0.0 and "the words" not in s["prompt"] for s in p["shots"])


# ------------------------------------------------------------------------------------------------ compose (tiny synthetic frames)
H, W, N = 96, 64, 48


def _type_dir(tmp, n=N):
    m = np.zeros((H, W), np.float32)
    cv2.rectangle(m, (8, 30), (56, 66), 1.0, -1)
    for k, im in (("matte", m), ("line", m)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), (im * 255).astype(np.uint8))
    return str(tmp)


def _ops():
    return [
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "matte": "$matte", "type_a": "$type_matte", "type_rgb": "$tcol",
                                                    "matte_threshold": 0.5, "type_opacity": 1.0, "legible": False}},
    ]


def _compose(tmp, words):
    piece = {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]}, "shots": [{"id": "s1", "frames": N, "matte": True}],
             "type": {"words": words}, "compose": {"crossfade": 0.0, "default": {"ops": _ops()}, "legible": False}}
    plate = np.full((N, H, W, 3), 51, np.uint8)
    u8, _ = RE.run_compose(piece, "s1", type_dir=_type_dir(tmp), footage=plate, matte=np.zeros((N, H, W), np.float32), backend="numpy", threads=0)
    return np.asarray(u8, np.uint8), piece


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_old_piece_without_owner_fields_composes_exactly_as_before(tmp_path):
    old = [{"text": "A", "t0": 0.0, "t1": 2.0}]
    flat = [{"text": "A", "t0": 0.0, "t1": 2.0, "render": "flat"}]
    a, piece = _compose(tmp_path / "a", old)
    b, piece_b = _compose(tmp_path / "b", flat)
    rec_old = RE.build_recipe(piece, "s1", type_dir=str(tmp_path / "a"))
    assert all("gain" not in sp for sp in rec_old["inputs"].values())                       # the recipe is the pre-owner recipe
    assert _hash(a) == _hash(b), "an all-flat owner set draws the same frames as an old piece"
    assert a[5, 48, 32, 0] > 200 and a[40, 48, 32, 0] > 200                                  # the type is on screen for the whole shot


def test_compose_draws_the_flat_type_until_ltx_at_then_fades_it(tmp_path):
    words = [{"text": "A", "t0": 0.0, "t1": 2.0, "render": "handoff", "ltx_at": 0.5}]       # handoff at 1.0 s = frame 24
    out, piece = _compose(tmp_path / "h", words)
    rec = RE.build_recipe(piece, "s1", type_dir=str(tmp_path / "h"))
    assert "gain" in rec["inputs"]["type_matte"]
    px = lambda i: float(out[i, 48, 32, 0])          # noqa: E731  the red channel inside the word box (plate = 51)
    assert px(0) > 200 and px(23) > 200                                                  # flat type until ltx_at
    assert 51 < px(26) < 200                                                              # mid-fade
    assert px(24 + 7) == pytest.approx(51, abs=1) and px(40) == pytest.approx(51, abs=1)  # gone after the ~0.25 s ramp
    ltx, _ = _compose(tmp_path / "l", [{"text": "A", "t0": 0.0, "t1": 2.0, "render": "ltx"}])
    assert np.allclose(ltx[:, 48, 32, 0], 51)                                             # an ltx word is never drawn flat


def test_hold_windows_only_cover_the_flat_part_of_a_word(tmp_path):
    p = _piece_words((0.0, 4.0, "flat", None), (4.5, 8.5, "ltx", None), (9.0, 13.0, "handoff", 0.5))
    p["plan"] = {"flow_wake": {"holds": [[1.0, 3.0], [5.5, 7.5], [10.0, 12.0]]}}
    assert RE.piece_word_holds(p) == [(1.0, 3.0), (10.0, 11.0)]                           # ltx dropped, handoff (11.0 s) clipped
    old = _piece_words((0.0, 4.0, None, None))
    old["plan"] = {"flow_wake": {"holds": [[1.0, 3.0]]}}
    assert RE.piece_word_holds(old) == [(1.0, 3.0)]


# ------------------------------------------------------------------------------------------------ concept: one lockup per shot
def test_concept_asks_for_one_lockup_per_shot_and_merges_or_splits():
    from tools.fusion import concept as CO
    assert CO.lockup_count(16, "cut") == 4 and CO.lockup_count(16, "chain") == 4 and CO.lockup_count(16, "single") is None and CO.lockup_count(24, "cut") == 6
    prompt = CO.concept_prompt(3, 16, [])
    want = CO.target_mode([], 3)
    if CO.lockup_count(16, want):
        assert f"EXACTLY {CO.lockup_count(16, want)} entries" in prompt and "1-3 words" in prompt
    obj = dict(title="x", subject="old fisherman", scene="an old fisherman on a quiet harbour at dawn with a small boat and a lantern in the mist",
               music="folk, guitar, soft drums, warm, airy", bpm=90, mood="tender", look="strand", shots="cut")
    c, errors, fixes = CO.normalize(dict(obj, words=["A B", "C", "D E", "F G H", "I", "J K"]), 3, [], 16)           # 6 lockups for 4 shots: merged
    assert not errors and len(c["words"]) == 4 and any("lockups: 6 lines merged to 4" in f for f in fixes)
    c, errors, fixes = CO.normalize(dict(obj, words=["A B C D E", "FGH", "I J"]), 3, [], 16)                         # a 5-word lockup is the model's error
    assert any("too long" in e for e in errors)
    c, errors, fixes = CO.normalize(dict(obj, words=["AFTER HOURS", "THE CITY KEEPS", "DREAMING"]), 3, [], 16)       # 3 lockups for 4 shots: split
    assert not errors and len(c["words"]) == 4 and any("split to 4" in f for f in fixes)
    c, errors, fixes = CO.normalize(dict(obj, shots="single", words=["A B", "C", "D E", "F G H", "I", "J K"]), 3, [], 16)  # a single shot keeps 5-7 free lines
    assert not errors and len(c["words"]) == 6 and c["shots"] in CO.MODES
    for i in range(1, 8):                                                                                              # the rules bank is fitted too
        r = CO.rules_variation(i, [], 16)
        n = CO.lockup_count(16, r["shots"])
        assert n is None or len(r["words"]) == n


# ------------------------------------------------------------------------------------------------ the three-layer hook of build_graphs
@pytest.mark.skipif(not (ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers.json").exists(),
                    reason="dev fixture not shipped in the public export: app/graphs/fusion_design_qwen_auto_layers.json")
def test_build_graphs_layers_hook_matches_the_graphs_on_disk_and_owner_gating_reaches_the_text_pass():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "app" / "build_graphs.py"), "--check", "--layers", "3"], capture_output=True, text=True, cwd=str(ROOT), timeout=900)
    assert r.returncode == 0, r.stdout[-800:]
    assert "fusion_design_qwen_auto_layers" in r.stdout and "problem" not in r.stdout.lower().replace("0 problem", "")
    g = json.loads((ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers.json").read_text(encoding="utf-8"))
    # the text pass (LTX letters live here) is guided by the SLOT's control slice and guide = what MographShotSlot gated by owner; the clean chain never sees the control
    for k in (1, 2, 3, 4):
        if f"t{k}.32" not in g:
            continue
        assert g[f"t{k}.32"]["inputs"]["strength"] == [f"slot{k}", 6]
        assert g[f"slot{k}"]["class_type"] == "MographShotSlot"
        assert not any(n["class_type"] == "LTXAddVideoICLoRAGuide" for nid, n in g.items() if nid.startswith(f"l{k}."))
    from tools.app import layered_shots as LS
    p = {"shots": [{"id": "s1", "control": {"pass": "depth", "guide": 0.0}}, {"id": "s2", "control": {"pass": "depth", "guide": 0.65}}]}
    assert LS.wants_text(p, "s1") is False and LS.wants_text(p, "s2") is True          # an all-flat shot (guide 0) never runs the text pass: background = the clean clip


# ---------------------------------------------------------------------------------------------------------------- v026 r01 GLASS: a handoff word stays flat for its first beat
def _glass_piece():
    """the shape of v026 r01: GLASS planned at 3.21-5.87 s (ltx_t 4.442 written for that), the real type timeline puts it at 6.37-8.63 s (hit 6.42)."""
    words = [dict(text="LAMP", t0=0.0, t1=2.81, form_s=0.8, render="flat"),
             dict(text="GLASS", t0=3.21, t1=5.87, form_s=0.45, render="handoff", ltx_at=0.4, ltx_t=4.442, material="blown peach glass")]
    return {"type": {"words": words, "bpm": 75.0}, "quality": "final", "fps": 24}


def test_handoff_stale_ltx_t_cannot_hide_the_flat_word():
    p = _glass_piece()
    rows = [dict(start=0.0, hit=0.82, release=4.72, end=4.77), dict(start=6.37, hit=6.42, release=7.82, end=8.63)]
    wins = OW.windows(p, rows)
    g = wins[1]
    assert OW.handoff_t(g) >= g["hit"] + 60.0 / 75.0 - 1e-6                      # >= 1 beat flat after the landing
    for t in (g["hit"], g["hit"] + 0.4, g["hit"] + 0.79):
        assert OW.flat_gain(wins, t) == pytest.approx(1.0, abs=1e-6), t           # fully drawn: it did fade to 0 at its first frame before the fix
    assert OW.flat_gain(wins, g["end"]) < 0.5                                     # LTX still takes over later
    assert OW.control_gain(wins, g["hit"] + 0.3) == pytest.approx(0.0, abs=1e-6)


def test_handoff_floor_survives_the_recipe_gain_spec():
    p = _glass_piece()
    rows = [dict(start=0.0, hit=0.82, release=4.72, end=4.77), dict(start=6.37, hit=6.42, release=7.82, end=8.63)]
    spec = OW.gain_spec(OW.windows(p, rows))
    assert OW.gain_at(spec, 6.42 + 0.7) == pytest.approx(1.0, abs=1e-6)
    assert OW.gain_at(spec, 8.6) < 0.5
    # the planner timeline (no manifest rows): the same floor from form_s + the tempo
    w = OW.windows(p)[1]
    assert OW.handoff_t(w) >= 3.21 + 0.45 + 0.8 - 1e-6


# ------------------------------------------------------------------------------------------------ STORYBOARD v12: the planner cuts exactly the story's shots
def test_story_shot_count_is_validated_and_cut_spans_are_exact():
    assert pr.story_shot_count(5, 24 * 24) == (5, None) and pr.story_shot_count(None, 576) == (None, None)
    n, note = pr.story_shot_count(2, 24 * 24)                                             # 12 s per shot > one clip
    assert n is None and "planner's own shot count" in note
    assert pr.story_shot_count(7, 576)[0] is None and pr.story_shot_count(0, 576)[0] is None and pr.story_shot_count("x", 576)[0] is None
    assert pr.story_shot_count(6, 5 * 24)[0] is None                                        # < MIN_SPAN frames per shot
    assert pr.story_shots_of({"story_shots": 5}) == 5 and pr.story_shots_of(None, json.dumps({"owners": [], "story_shots": 4})) == 4 and pr.story_shots_of(None, "[]") is None
    m = synth_music(120, 24)
    grid = pr.make_grid(m, m["cut"])
    free = pr.cut_spans(m["nframes"], grid, 24, n=5)
    assert len(pr.cut_spans(m["nframes"], grid, 24, n=5, exact=True)) == 5
    ex = pr.cut_spans(m["nframes"], grid, 24, n=5, exact=True)
    assert sum(ex) == m["nframes"] and max(ex) <= pr.SHOT_FRAMES and min(ex) >= pr.MIN_SPAN
    assert len(free) >= 5                                                                  # without `exact` the rule may add shots (more shots before longer ones)


def test_plan_piece_honours_the_story_shots():
    req = dict(REQ, length=24, words="A\nB\nC\nD\nE", shots="cut")
    m = synth_music(120, 24)
    base = pr.plan_piece(req, m)
    p = pr.plan_piece(dict(req, owners={"owners": [], "story_shots": 5}), m)
    assert len(p["shots"]) == 5 and len(p["type"]["words"]) == 5 and len(p["stills"]) == 5
    assert all(s["frames"] <= pr.SHOT_FRAMES and s["span"] <= pr.SHOT_FRAMES for s in p["shots"])
    assert not any("story has" in n for n in p["plan"]["notes"])
    bad = pr.plan_piece(dict(req, words="A\nB", owners={"owners": [], "story_shots": 2}), m)         # 2 shots of 12 s: invalid -> today's rule + a note
    assert len(bad["shots"]) == len(base["shots"]) and any("story has 2 shots" in n for n in bad["plan"]["notes"])
    one = pr.plan_piece(dict(req, shots="single", owners={"owners": [], "story_shots": 5}), m)         # single = one shot: the count is ignored
    assert len(one["shots"]) == 1
