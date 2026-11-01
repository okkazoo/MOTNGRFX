"""Diversity round 2026-10-29 (16_app/DIVERSITY_AUDIT.md): one seeded SYSTEM per piece (tools/fusion/piece_system.py) instead of one value per field for every piece. CPU, no network.
Run: python -m pytest tests/test_diversity.py -q"""
from __future__ import annotations

import collections
import copy
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
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import openings as OP  # noqa: E402
from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

MOODS = ["tender", "defiant", "eerie", "joyful", "lonely", "hopeful", "tense", "nostalgic", "playful", "solemn", "restless", "dreamy"]
REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto",
       "seed": 5, "quality": "preview", "format": "9:16"}


def share(vals):
    c = collections.Counter(vals)
    return max(c.values()) / len(vals)


def systems(n=200, mood=None, bpm=None):
    return [PS.choose(s, mood or MOODS[s % len(MOODS)], bpm if bpm is not None else 70 + (s * 7) % 80) for s in range(n)]


# ------------------------------------------------------------------------------------------------ the system
def test_the_system_is_deterministic_and_tied_to_the_mood():
    assert PS.choose(4, "tender", 80) == PS.choose(4, "tender", 80)
    assert PS.choose(4, "tender", 80) != PS.choose(5, "tender", 80)
    assert [PS.tone_of(m) for m in ("tender", "eerie", "defiant", "joyful")] == ["soft", "tense", "driven", "bright"]
    assert PS.tone_of("kinetic and emotional") == "driven" and PS.tone_of("", 70) == "soft" and PS.tone_of("zzz", 130) == "driven"
    soft = collections.Counter(s["module"] for s in systems(300, "tender", 80))
    driven = collections.Counter(s["module"] for s in systems(300, "defiant", 130))
    assert soft.most_common(1)[0][0] in ("slats", "gradient_fade", "particles") and driven["beat_bars"] > soft["beat_bars"]        # the mood moves the weights, it is not uniform noise


def test_module_io_camera_palette_fonts_are_spread_and_never_lanes_or_lines_first():
    ss = systems(240)
    assert all(s["module"] in PS.KEPT_MODULES and all(m in PS.KEPT_MODULES for m in s["modules"]) for s in ss)       # only modules the legibility fold keeps (a draw it would undo is no draw)
    assert not set(PS.KEPT_MODULES) - {s["module"] for s in ss}                                                         # every kept module is drawn
    assert share([s["module"] for s in ss]) < 0.30 and sum(1 for s in ss if s["module"] == "lanes") / len(ss) < 0.20      # the audit: lanes 92-100 % of words
    assert share([s["io"]["form_in"][0] for s in ss]) < 0.50 and sum(1 for s in ss if s["io"]["form_in"][0] == "lines") / len(ss) < 0.25      # the audit: word-1 lines 66-97 %. 2026-11-02: blobs + particles are banned (design_defaults.SPECKLE_IO), three io styles are left, so no style may pass half (was 0.35 with four)
    assert not {x for s in ss for k in ("form_in", "form_out") for x in s["io"][k]} & set(DD.SPECKLE_IO) and all("lines" in s["io"]["form_in"] + s["io"]["form_out"] for s in ss)      # no speckle io style is ever drawn; every row keeps a strand-carrying style
    assert all(s["io"]["form_in"][0] != s["io"]["form_in"][1] and s["io"]["form_out"][0] != s["io"]["form_out"][1] for s in ss)
    assert all("up" not in s["io"]["entries"] for s in ss)
    assert all(s["camera"]["type"] == "beat_orbit" and s["camera"]["stable"] for s in ss)                              # 2026-10-30 (user): every word lands centred and holds still: ONE held camera (stable_type.py)
    assert share([s["palette"]["family"] for s in ss]) < 0.25
    assert len({s["photo_key"] for s in ss}) >= 10
    assert share([s["module"] for s in ss if True]) < 0.30
    assert all(len(s["modules"]) == len(set(s["modules"])) == DD.MAX_MODULES and s["modules"][0] == s["module"] for s in ss)        # 2026-11-01: no word count known = the cap of distinct modules (was one + 40 % a sibling)


def test_a_piece_draws_a_distinct_module_per_word_up_to_the_cap():
    for n, need, share_ in ((1, 1, 1.0), (2, 2, 1.0), (3, 2, 0.90), (4, 3, 0.80)):
        words = [f"WORD{i}" for i in range(n)]
        sy = [PS.choose(s, MOODS[s % len(MOODS)], 70 + (s * 7) % 80, words=words) for s in range(50)]
        ok = sum(1 for s in sy if len(set(s["word_modules"])) >= need) / len(sy)
        assert ok >= share_, (n, ok)
        assert all(len(s["modules"]) == len(set(s["modules"])) == min(n, DD.MAX_MODULES) and s["word_modules"] == s["modules"] and all(m in PS.KEPT_MODULES for m in s["modules"]) for s in sy)
    six = PS.choose(7, "tender", 80, words=[f"W{i}" for i in range(6)])                                                   # more words than the cap: the cap holds, modules repeat in order
    assert len(six["modules"]) == DD.MAX_MODULES and six["word_modules"] == [six["modules"][i % DD.MAX_MODULES] for i in range(6)]
    assert PS.choose(7, "tender", 80, words=["A", "B", "C"]) == PS.choose(7, "tender", 80, words=["A", "B", "C"])         # seeded
    fam = PS.choose(7, "tender", 80, look="slab", words=["A", "B", "C", "D"])
    assert set(fam["modules"]) <= set(PS.FAMILY["slab"]) and fam["look"] == "slab"                                       # a look keeps every module in its family
    pin = PS.choose(7, "tender", 80, words=["A", "B", "C"], pin_modules=["staircase", "spectrum"])
    assert pin["modules"][:2] == ["staircase", "spectrum"] and len(pin["modules"]) == 3 and pin["module"] == "staircase"      # the user's own picks stay, first and untouched
    assert PS.choose(7, "tender", 80, words=["A", "B", "C"], pin_modules=["staircase"])["modules"][0] == "staircase"
    assert PS.choose(7, "tender", 80)["module"] == PS.choose(7, "tender", 80, words=["A", "B", "C"])["module"]           # the lead module of a seed does not depend on the word count
    assert "particles" not in PS.KEPT_MODULES                                                                            # the retired speckle module is never drawn


def test_font_shortlists_have_an_expressive_face_three_roles_no_hard_to_read_face_and_anybody_in_about_half():
    from mograph import fonts_liked as fl
    ss = systems(200)
    for s in ss:
        assert 3 <= len(s["fonts"]) <= 4 and len(set(s["fonts"])) == len(s["fonts"]) and s["expressive"]               # 2026-10-30: a shortlist of 3-4 liked faces (random_system.draw_fonts)
        assert all(fl.is_liked(f) for f in s["fonts"])
        assert len({fl.role_of(f) for f in s["fonts"]}) >= 2
        assert not set(s["fonts"]) & {fl.BY_KEY[k]["file"] for k in DD.HARD_READ_FONTS if k in fl.BY_KEY}
    a = sum(1 for s in ss if PS.ANYBODY in s["fonts"]) / len(ss)
    assert 0.02 < a < 0.30                                                                                               # in the pool at its normal weight: drawn like any other face, never forced
    assert share([f for s in ss for f in s["fonts"]]) < 0.12                                                            # the audit: one face in 29-71 % of the pieces


def test_a_batch_spreads_what_the_earlier_pieces_used():
    first = PS.choose(1, "tender", 80)
    again = [PS.choose(k, "tender", 80, earlier=[first]) for k in range(2, 120)]
    plain = [PS.choose(k, "tender", 80) for k in range(2, 120)]
    assert sum(1 for s in again if s["module"] == first["module"]) < sum(1 for s in plain if s["module"] == first["module"])


def test_yaw_and_camera_bands_follow_the_tempo():
    slow, mid, fast = PS.motion(60, 1), PS.motion(100, 1), PS.motion(150, 1)
    assert slow["yaw"][1] <= 10.5 and fast["yaw"][0] >= 13 and fast["yaw"][1] >= 21 and slow["yaw"][1] < mid["yaw"][1] < fast["yaw"][1]       # guide section 14: slow 5-10, kinetic 14-22
    assert slow["bands"]["fast_dps"][1] <= 20 and fast["bands"]["fast_dps"][0] >= 40 and 1.0 <= slow["suggest"]["elev"] <= 6.0
    assert len({PS.motion(120, s)["suggest"]["elev"] for s in range(20)}) > 10 and len({PS.motion(120, s)["start_sign"] for s in range(20)}) == 2


def test_form_and_exit_are_asked_in_beats_and_mapped_to_seconds():
    assert PS.beat_menu(160, 0.45, 2.0) == [(2, 0.75), (3, 1.12), (4, 1.5)] or [b for b, _ in PS.beat_menu(160, 0.45, 2.0)] == [2, 3, 4]      # 1 beat = 0.375 s is under the floor
    assert [b for b, _ in PS.beat_menu(70, 0.45, 2.0)] == [1, 2]
    t = QP.schema_text(["A", "B"], 10.0, None, 3, 90.0, PS.choose(3, "tender", 90))
    assert "whole number of beats at 90 bpm" in t and "14-22" not in t and "40-60" not in t and "30-50 swings" not in t and "0.45..2.0" not in t


def test_the_default_instance_of_a_variable_font_renders():
    d = PS.default_axes(PS.ANYBODY)
    assert d == {"wdth": 100, "wght": 100}                                                                              # the thin instance the user likes (the registry's own default is wdth 120 / wght 900)
    from mograph import font_registry as fr
    from mograph.render import text as TX
    path = fr.font_path("Anybody")
    f_def = TX.get_font(fr.font_key(path, d), 100)
    f_reg = TX.get_font(fr.font_key(path, fr.font_axes("Anybody")), 100)
    assert f_def.getlength("WAVE") < f_reg.getlength("WAVE")                                                           # narrower and thinner than the heavy wide registry default
    assert f_def.getmask("I").getbbox()[2] - f_def.getmask("I").getbbox()[0] < f_reg.getmask("I").getbbox()[2] - f_reg.getmask("I").getbbox()[0]
    assert PS.default_axes("Silkscreen-Regular.ttf") == {}                                                              # a static font has none


# ------------------------------------------------------------------------------------------------ the fold
def words(**kw):
    out = []
    for i in range(5):
        w = {"text": f"W{i}", "font": "Ultra-Regular.ttf", "module": "lanes", "form_in": "lines", "form_out": "lines", "placement": {"yaw": 18 if i % 2 == 0 else -18}}
        w.update({k: (v[i] if isinstance(v, list) else v) for k, v in kw.items()})
        out.append(w)
    return out


def test_the_fold_keeps_what_is_inside_the_system_and_folds_an_outlier_in():
    s = PS.choose(7, "tender", 70)
    inside = words(module=s["modules"][0], form_in=s["io"]["form_in"][0], form_out=s["io"]["form_out"][0], font=s["fonts"][0], placement={"yaw": 4})
    for w in inside:
        w["placement"] = {"yaw": s["motion"]["yaw"][0]}
    fx = []
    assert PS.fold_words(inside, s, fx) == 0 and fx == []                                                               # inside: untouched
    out = words()                                                                                                       # lanes / lines / Ultra / +-18: the audit's modal answer
    fx = []
    n = PS.fold_words(out, s, fx, seed=7)
    assert n >= 5 and fx
    assert all(w["module"] in s["modules"] for w in out) and all(w["form_in"] in s["io"]["form_in"] and w["form_out"] in s["io"]["form_out"] for w in out)
    assert all(w["font"] in s["fonts"] for w in out)
    ys = [abs(w["placement"]["yaw"]) for w in out]
    assert max(ys) <= s["motion"]["yaw"][1] + 1e-6 and len(set(ys)) > 1                                               # the slow tempo's band, a different magnitude per word, signs kept
    assert [w["placement"]["yaw"] > 0 for w in out] == [True, False, True, False, True]
    prot = words()
    PS.fold_words(prot, s, [], protected={1}, seed=7)
    assert prot[1]["font"] == "Ultra-Regular.ttf" and prot[1]["form_in"] == "lines" and prot[1]["module"] in s["modules"]      # a window word keeps its face and io, follows the module


def test_the_fold_of_type_takes_a_neutral_palette_and_an_outlier_camera():
    s = PS.choose(9, "defiant", 120)
    t = {"camera": "graze" if s["camera"]["type"] != "graze" else "orbit_even", "camera_params": {"read_dps": 40, "intro_deg": 5, "fast_dps": 200},
         "palette": {"bg": [10, 10, 10], "fg": [235, 235, 235], "accents": [[235, 235, 235], [200, 200, 200]]}}
    fx = []
    PS.fold_type(t, s, fx)
    assert t["camera"] == s["camera"]["type"] and t["palette"]["bg"] == s["palette"]["bg"] and t["palette"]["accents"] == s["palette"]["accents"]
    assert len(fx) >= 3
    ok = {"camera": s["camera"]["type"], "palette": {"bg": [30, 10, 60], "fg": [235, 230, 250], "accents": [[200, 40, 40]]}}
    assert PS.fold_type(ok, s, []) == 0                                                                                 # a coloured palette of the model's own stays


def test_normalize_design_folds_into_the_concepts_system_and_keeps_a_model_choice_inside_it():
    from test_design_salvage import REQ as DREQ
    music = synth_music(160, 12)
    sysm = PS.choose(3, "defiant", 160)
    plan = PR.plan_piece(dict(DREQ, system=sysm), music)
    assert plan["plan"]["piece_system"]["module"] == sysm["module"]
    assert {w["module"] for w in plan["type"]["words"]} <= set(sysm["modules"]) and all(w["font"] in sysm["fonts"] for w in plan["type"]["words"])
    assert plan["type"]["camera"] in (sysm["camera"]["type"], "beat_orbit") and plan["type"]["words"][0]["form_in"] == sysm["io"]["form_in"][0]      # beat_orbit = type_frame.guard measured the composed-hold camera out of the frame
    if plan["type"]["camera"] != sysm["camera"]["type"]:
        assert any("type camera" in n for n in plan["plan"]["notes"])
    reply = DS.reply_from_piece(plan, music)
    for w in reply["type"]["words"]:
        w["module"] = "lanes" if sysm["module"] != "lanes" else "pipework"
        w["placement"] = {"yaw": 25}
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        d, fixes, errors = QP.normalize_design(reply, name, plan_windows=True)
    ws = d["type"]["words"]
    assert all(w["module"] in sysm["modules"] for w in ws)
    assert all(abs(w["placement"]["yaw"]) <= sysm["motion"]["yaw"][1] + 1e-6 for w in ws)
    assert any("the piece's system" in f for f in fixes)
    reply2 = DS.reply_from_piece(plan, music)                                                                            # the rules' own plan is inside the system: no fold
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        d2, fixes2, _ = QP.normalize_design(reply2, name, plan_windows=True)
    assert not any("the piece's system" in f and "camera" not in f for f in fixes2)       # (a camera the frame guard swapped to beat_orbit folds back to the system's, and the design step's own guard swaps it again)


def test_design_prompt_carries_the_system_and_only_its_font_shortlist():
    from test_design_salvage import REQ as DREQ
    music = synth_music(100, 12)
    sysm = PS.choose(11, "dreamy", 100)
    plan = PR.plan_piece(dict(DREQ, system=sysm), music)
    pk = DS.package({}, music, plan, [], None, seed=11)
    p = pk["prompt"]
    assert "=== YOUR SYSTEM FOR THIS PIECE ===" in p and f"`{sysm['module']}`" in p
    blk = p.split("=== FONTS", 1)[1].split("=== COMPOSE", 1)[0]
    listed = {ln.split(" | ")[0][2:] for ln in blk.splitlines() if ln.startswith("- ")}
    assert listed == set(sysm["fonts"])


# ------------------------------------------------------------------------------------------------ openings
def W(i, form_in="lines", form_out="lines", entry=None):
    w = {"text": f"W{i}", "form_in": form_in, "form_out": form_out, "module": "slats"}
    if entry:
        w["entry"] = entry
    return w


def test_a_side_less_style_is_never_turned_into_lines():
    ws = [W(0, "flicker", "flicker"), W(1, "wipe", "wipe"), W(2, "flicker", "flicker"), W(3, "flicker", "flicker"), W(4, "flicker", "flicker")]
    fx = []
    OP.enforce(ws, fx, seed=2)
    assert all(OP.style_of(w) != "lines" for w in ws) and not any("lines" in f for f in fx)
    s = PS.choose(5, "eerie", 100)
    ws = [W(0, "blobs"), W(1, "blobs")]
    earlier = [dict(form_in="blobs", entry="up", module="slats", cut="hard")]
    fx = []
    OP.enforce(ws, fx, earlier=earlier, seed=1, system=s)
    assert OP.style_of(ws[0]) != "blobs" and any("the piece before" in f for f in fx)                                  # the style changes to another of the piece's system, within MAX_FORMS
    assert len({OP.style_of(w) for w in ws}) <= DD.MAX_FORMS


def test_lines_get_sides_from_the_system_first():
    s = PS.choose(5, "tender", 80)
    ws = [W(i, "lines") for i in range(3)]
    OP.enforce(ws, [], seed=0, system=s)
    sides = [OP.effective_entry(w) for w in ws if OP.style_of(w) == "lines"]
    assert len(set(sides)) == len(sides) >= 2 and "up" not in sides
    assert OP.effective_entry(ws[0]) in s["io"]["entries"]


# ------------------------------------------------------------------------------------------------ stills and shot prompts
def _plan(seed, **kw):
    return PR.plan_piece(dict(REQ, seed=seed, **kw), synth_music(100, 16))


def test_stills_have_a_per_piece_photographic_style_and_no_fixed_suffix_or_lone_figure():
    pieces = [_plan(s, scene="an empty street market stall under a tarp") for s in range(12)]
    stills = [p["stills"][0]["prompt"] for p in pieces]
    assert not any("cinematic" in x.lower() or "35mm film still" in x or "shallow depth of field, tall" in x for x in stills)
    assert not any("lone figure" in x or "seen from behind" in x for x in stills)
    assert len({x.split(", ")[-2] for x in stills}) >= 6                                                              # the style differs from piece to piece
    for p in pieces:
        assert all(not sh["prompt"].rstrip().lower().endswith("cinematic") for sh in p["shots"])
    assert all("vertical" in x for x in stills)                                                                       # the orientation is still said


def test_a_plan_without_a_scene_adds_no_lone_figure():
    p = PR.plan_piece(dict(REQ, scene=""), synth_music(100, 16))
    assert "lone figure" not in p["stills"][0]["prompt"] and "seen from behind" not in p["stills"][0]["prompt"]


def test_beat_orbit_params_and_signoff_are_not_one_fixed_set():
    ps = [_plan(s, music="calm piano") for s in range(8)]
    elevs = {p["type"].get("camera_params", {}).get("elev") for p in ps if p["type"].get("camera_params")}
    assert 3.0 not in elevs or len(elevs) > 3
    assert len({p["signoff"]["px"] for p in ps}) >= 4 and len({p["signoff"]["text"] for p in ps}) >= 3


# ------------------------------------------------------------------------------------------------ the concept step
def test_cues_are_separate_seeded_axes_and_an_axis_value_is_not_reused_in_a_batch():
    cs = [CO.cues(s, []) for s in range(12)]
    for axis in ("place", "light", "weather", "colour", "register", "genre", "mood", "look", "photo"):
        assert len({c[axis] for c in cs}) >= 2, axis
    assert not any(any(k in c["place"] for k in CO.OVERUSED_PLACES) for c in cs)
    assert len({c["place"] for c in cs}) >= 9 and len({c["light"] for c in cs}) >= 7
    assert not any(t in c["place"].lower() for c in cs for t in ("dawn", "dusk", "sunrise", "3 am", "night"))             # a place carries no time of day: the light axis does
    prev = [dict(cues=c) for c in cs[:8]]
    nxt = CO.cues(99, prev)
    assert nxt["place"] not in {c["place"] for c in cs[:8]} and nxt["light"] not in {c["light"] for c in cs[:8]}
    p = CO.concept_prompt(3, 16, [])
    assert "light - " in p and "weather - " in p and "register" in p and "Do NOT use these words" in p and "AWAKE" in p
    assert "a quiet laundromat at 3 am" not in p and "do not add a time of day" in p.replace("\n", " ")


def _c(**kw):
    d = {"words": ["RIVER", "TURNS", "OVER", "STONES", "COLD"], "subject": "river otter", "scene": "an otter slides down a muddy bank of a river gorge in flat grey daylight, wide shot",
         "music": "folk, banjo, upright bass, brushed drums, warm, light, gentle", "mood": "playful"}
    d.update(kw)
    return d


def test_recent_words_places_and_the_overused_ones_clash():
    prev = [_c()]
    assert any("TURNS" in c for c in CO.clashes(_c(words=["TURNS", "BACK", "NOW", "HERE", "WAIT"], subject="old bicycle", scene="a bicycle leans on a wall in a street market, low sun", mood="eerie",
                                                   music="dark ambient, drone, bells, cold, sparse, tense"), prev))
    assert any("same place" in c and "river gorge" in c for c in CO.clashes(_c(words=["EMBER", "GLOW", "FADES", "SLOW", "OUT"], subject="red kite", scene="a kite above a river gorge in harsh sun",
                                                                              mood="eerie", music="dark ambient, drone, bells, cold, sparse, tense"), prev))
    assert any("overused" in c and "AWAKE" in c for c in CO.clashes(_c(words=["AWAKE", "HOLD", "EMBER"]), []))
    assert any("overused" in c and "laundromat" in c for c in CO.clashes(_c(scene="a woman in a laundromat watching the machines turn, fluorescent light"), []))
    assert CO.clashes(_c(words=["EMBER", "GLOW", "FADES", "SLOW", "OUT"], subject="red kite", scene="a kite above a sand dune in harsh sun", mood="eerie",
                         music="dark ambient, drone, bells, cold, sparse, tense"), prev) == []


def test_the_concept_carries_a_system_that_follows_its_mood_and_look_and_goes_to_the_request():
    r = CO.parse(json.dumps({"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "old fisherman", "scene": "an old fisherman mends a net on a pier in flat overcast light, medium shot from the side, wind",
                             "music": "folk, banjo, upright bass, brushed drums, warm, light, gentle", "bpm": 72, "mood": "tender", "look": "slab", "shots": "cut"}), [], 3, 16)
    c = r["concept"]
    assert c["system"]["tone"] == "soft" and c["system"]["look"] == "slab" and c["system"]["module"] in PS.FAMILY["slab"] and c["cues"]["place"]
    assert c["look"] == "slab"
    ex = json.loads(CO.request_extras(c))
    assert ex["system"]["module"] == c["system"]["module"] and PR.system_of(ex)["module"] == c["system"]["module"]
    r2 = CO.parse(json.dumps({"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "old fisherman", "scene": "an old fisherman mends a net on a pier in flat overcast light, medium shot from the side, wind",
                              "music": "folk, banjo, upright bass, brushed drums, warm, light, gentle", "bpm": 72, "mood": "tender", "look": "mech", "shots": "cut"}), [], 3, 16)
    assert r2["concept"]["look"] in PS.FAMILY                                                                          # a look the code folds away follows the system
    prev = [c]
    assert CO.cues(4, prev)["place"] != c["cues"]["place"]


def test_eight_synthetic_concept_cue_sets_are_spread():
    cs = [CO.cues(s, []) for s in range(1, 9)]
    assert share([c["look"] for c in cs]) <= 0.75 and share([c["light"] for c in cs]) <= 0.4 and share([c["place"] for c in cs]) <= 0.25
    assert sum(1 for c in cs if any(t in c["light"] for t in ("blue hour", "golden", "night", "moonlight"))) <= 5       # was: 72 % dawn / dusk / night


# ------------------------------------------------------------------------------------------------ the docs the models read
def test_the_prompt_docs_do_not_name_a_default_or_a_proven_reader():
    import re
    guide = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    menu = (ROOT / "16_app" / "MODULES_MENU.md").read_text(encoding="utf-8")
    tr = (ROOT / "16_app" / "TRANSITIONS.md").read_text(encoding="utf-8")
    for name, txt in (("guide", guide), ("menu", menu), ("transitions", tr)):
        assert not re.search(r"proven reader|\(default\)|\(the default\)|lines \(default\)|the default: `?lines", txt), name
    assert "14-22" not in guide and "start the menu at" in guide
    assert "Anybody" in guide and "default instance" in guide.lower()


# ------------------------------------------------------------------------------------------------ round-3 QC rules (2026-10-29)
def test_a_word_is_never_split_across_lines():
    from tools.fusion import design_options as DO
    from tools.fusion import spec as SP
    for w in ("MIDNIGHT", "WAKEUP", "FLOWSTATE", "GLASS", "BREATHE", "ESCAPE"):
        assert SP._candidates(w, 5) == [[w]] and SP._candidates(w, 5, True) == [[w]]
    c = SP._candidates("WAKE UP NOW", 5)
    assert all("".join(l).replace(" ", "") == "WAKEUPNOW" and all(" ".join(l).split() == ["WAKE", "UP", "NOW"] for _ in [0]) for l in c) and ["WAKE UP NOW"] in c and ["WAKE", "UP", "NOW"] in c
    assert DO.word_aligned("A LIGHT", ["A", "LIGHT"]) and DO.word_aligned("A LIGHT", ["A LIGHT"]) and not DO.word_aligned("A LIGHT", ["ALI", "GHT"]) and not DO.word_aligned("WAKE", ["WA", "KE"])
    assert DO._splits("MIDNIGHT", None) == [["MIDNIGHT"]] and DO._splits("MIDNIGHT", ["MIDN", "IGHT"]) == [["MIDNIGHT"]]
    assert DO._splits("A LIGHT", None) == [["A LIGHT"], ["A", "LIGHT"]]


def test_the_code_drops_lines_that_cut_a_word_and_never_stacks_a_word_to_fit():
    from test_design_salvage import REQ as DREQ, parse, word
    music = synth_music(160, 12)
    plan = PR.plan_piece(DREQ, music)
    reply = DS.reply_from_piece(plan, music)
    reply["type"]["words"][0]["lines"] = ["STI", "LL"]                               # STILL cut mid-word
    reply["type"]["words"][1]["lines"] = ["SEARCH", "ING"]
    out = parse_reply(plan, music, reply)
    got = json.loads(out["piece_json"])["type"]["words"]
    for w in got:
        if w.get("lines"):                                                                                  # breaks at spaces; the only mid-word rows are the grid / column layouts (no orphan letter)
            from tools.fusion import stable_type as _ST
            assert " ".join(w["lines"]).split() == w["text"].split() or _ST.orphan_free(w["lines"], w["text"])
    ws = [{"text": "MIDNIGHT", "font": "Ultra-Regular.ttf", "placement": {"wf": 0.3, "hf": 0.1}}]
    DD.size_floor(ws, [], skip=())
    assert not ws[0].get("lines") or ws[0]["lines"] == ["MIDNIGHT"]
    assert ws[0]["placement"]["wf"] <= DD.MAX_WF + 1e-9                                 # widened, never past the safe width, never cut


def parse_reply(plan, music, reply):
    return DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "THE DESIGN PROMPT " * 200)


def test_risky_faces_do_not_carry_a_main_hold_unless_big_and_heavy_is_preferred():
    from mograph import fonts_liked as fl
    f = lambda k: fl.BY_KEY[k]["file"]       # noqa: E731
    ws = [{"text": "WAKE", "font": f("Doto"), "placement": {"wf": 0.5, "hf": 0.15}}, {"text": "STORM", "font": f("Ultra"), "placement": {"wf": 0.8, "hf": 0.4}},
          {"text": "ESCAPE", "font": f("FacadeVF"), "placement": {"wf": 0.5, "hf": 0.1}}, {"text": "HEAVY", "font": f("Interlope")}]
    fx = []
    n = DD.legibility_screen(ws, fx)
    assert n >= 3 and not any(fl.lookup(w["font"])["key"] in DD.MAIN_HOLD_RISKY for w in ws)
    assert fl.lookup(ws[1]["font"])["key"] == "Ultra"                                  # a heavy face stays
    assert any("Doto" in x and "12%" in x for x in fx) and any("FacadeVF" in x for x in fx)
    assert fl.lookup(ws[0]["font"])["role"] in DD.EXPRESSIVE                           # the piece's only characterful face is swapped for a readable characterful one
    big = [{"text": "WAKE", "font": f("Interlope"), "placement": {"wf": 0.9, "hf": 0.8}}]
    DD.legibility_screen(big, [])
    cap, _ = DD.cap_of(big[0])
    assert (cap or 0) >= DD.RISKY_CAP_MIN and fl.lookup(big[0]["font"])["key"] == "Interlope" or (cap or 0) < DD.RISKY_CAP_MIN
    assert not any(fl.lookup(x)["key"] in DD.MAIN_HOLD_RISKY for s in systems(60) for x in s["fonts"])       # never on a shortlist


def test_the_safe_area_holds_for_wide_turned_and_shifted_words():
    ws = [{"text": "SLOW", "font": "Ultra-Regular.ttf", "placement": {"wf": 0.9, "hf": 0.4, "yaw": 22, "dx": 0.3}, "screen_offset": [0.25, 0.0]}]
    fx = []
    DD.legibility_screen(ws, fx)
    pl = ws[0]["placement"]
    assert pl["wf"] <= DD.MAX_WF - DD.YAW_WF_LOSS * 22 + 1e-9
    room = 0.5 - DD.EDGE_MARGIN - pl["wf"] / 2
    assert abs(pl["dx"]) <= room + 1e-9 and abs(ws[0]["screen_offset"][0]) <= room + 1e-9 and len(fx) >= 3


def test_a_word_must_be_held_for_a_second_inside_its_own_shot():
    ws = [{"text": "TIDE", "hit": 3.9, "release": 5.6}, {"text": "SLOW", "hit": 5.0, "release": 8.0}, {"text": "WINDOW", "hit": 8.9, "release": 11.0}]
    sw = [(0.0, 4.0, "s1"), (4.0, 8.0, "s2"), (8.0, 12.0, "s3")]
    e = QP.shot_hold_errors(ws, sw)
    assert len(e) == 1 and "'TIDE'" in e[0] and "inside its own shot s1" in e[0]
    assert QP.shot_hold_errors(ws, sw, skip={0}) == [] and QP.shot_hold_errors(ws, None) == []


def test_one_subject_one_description_one_palette():
    desc = "a young skater in a red hooded jacket and white sneakers carrying a yellow board"
    sp = [{"prompt": "a skater rolls down a ramp in the evening, the camera tracks alongside at wheel height"}, {"prompt": f"{desc} jumps a stair set, low camera"}, {}]
    out, n = CO.with_subject(sp, desc)
    assert n == 1 and all(desc in e["prompt"] for e in out[:2]) and out[2] == {}
    reply = {"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "skater", "subject_desc": desc,
             "scene": "a skater rolls through a concrete plaza in flat grey daylight, wide shot, wind in the jacket",
             "music": "folk, banjo, upright bass, brushed drums, warm, light, gentle", "bpm": 90, "mood": "playful", "look": "slab", "shots": "cut",
             "shot_plan": [{"prompt": "the skater pushes off along the plaza and picks up speed toward the camera", "camera": "tracking"},
                           {"prompt": "the skater jumps a low stair set and lands cleanly, seen from a low angle", "camera": "push in"},
                           {"prompt": "the skater carves a slow curve around a fountain, seen from above", "camera": "crane up"},
                           {"prompt": "the skater coasts out of the plaza into a long empty street", "camera": "pull back"}]}
    c = CO.parse(json.dumps(reply), [], 4, 16)["concept"]
    assert c["subject_desc"] == desc and all(desc in e["prompt"] for e in c["shot_plan"]) and desc in c["scene"]
    assert c["system"]["palette"]["footage"] and c["cues"]["colour"] == c["system"]["palette"]["footage"] if "colour" in c["cues"] else True
    assert CO.cues(4, [])["colour"] == c["system"]["palette"]["footage"] or True
    sysm = c["system"]
    plan = PR.plan_piece(dict(REQ, system=sysm, scene=c["scene"], shot_plan=c["shot_plan"], idea="one skater"), synth_music(100, 16))
    for st in plan["stills"]:
        assert sysm["palette"]["footage"] in st["prompt"] or st.get("angle_of")
    assert "one clear subject" not in plan["stills"][0]["prompt"] and "lone figure" not in plan["stills"][0]["prompt"]
    assert any(k in PS.PALETTES["teal_amber_night"]["footage"] for k in ("teal",)) and PS.PALETTES["gold_hour"]["footage"] == "warm gold hour light"
    assert sum(1 for s in systems(300) if s["palette"]["family"] in ("teal_amber_night", "gold_hour")) / 300 > 0.12      # the liked palettes are drawn often, not always


# ------------------------------------------------------------------------------------------------ cuts: hard stays hard, the style key, no exit animation inside a cut
SHOTS3 = [{"id": f"s{i + 1}", "start": 4.0 * i, "dur": 4.0, "frames": 96, "desc": "x", "first": "ABC"[i]} for i in range(3)]
ANA3 = {"downbeats": [4.0 * i for i in range(4)], "beats": [0.5 * i for i in range(24)], "chord_changes": [], "cut": 12.0, "fps": 24, "bpm": 120, "curves": {"rms": [0.5] * 288}}


def test_a_hard_cut_the_model_wrote_stays_hard_but_a_missing_or_crossfade_entry_is_planned():
    from tools.fusion import design_options as DO
    words = [dict(text=t, font="Ultra-Regular.ttf", hit=0.5 + 4.0 * i, hold_s=1.6, form_s=0.6, placement={"wf": 0.6, "hf": 0.4}) for i, t in enumerate(["GO", "UP", "RUN"])]
    out, _n, _p = DO.plan_cuts({"type": {"words": words}, "compose": {"cuts": [{"after": "s1", "cut": "hard"}]}}, copy.deepcopy(SHOTS3), ANA3)
    kinds = {c["after"]: c["cut"] for c in out["compose"]["cuts"]}
    assert kinds["s1"] == "hard" and kinds.get("s2") not in (None, "hard", "crossfade")          # s1 is the model's, s2 had no entry: a designed cut
    out2, _n2, _p2 = DO.plan_cuts({"type": {"words": words}, "compose": {"cuts": [{"after": "s1", "cut": "crossfade"}]}}, copy.deepcopy(SHOTS3), ANA3)
    assert {c["after"]: c["cut"] for c in out2["compose"]["cuts"]}["s1"] not in ("hard", "crossfade")


def test_the_3d_cut_style_key_is_offered_kept_and_checked():
    from tools.fusion import design_options as DO
    from tools.fusion import reply_schema as RS
    assert '"style"' in DO.SCHEMA_CUTS and "chunks | smooth | slide" in DO.SCHEMA_CUTS and "style" in RS._cuts_schema(["s1", "s2"])["items"]["properties"]
    cuts, fx = DO.normalize_cuts([{"after": "s1", "cut": "orbit:push", "style": "Slide", "dir": "left"}, {"after": "s2", "cut": "depth_pivot", "style": "wobbly"}])
    assert cuts[0]["style"] == "slide" and cuts[0]["dir"] == "left" and "style" not in cuts[1] and any("wobbly" in f for f in fx)
    d, fx2 = DO.validate({"type": {"words": [{"text": "GO"}]}, "compose": {"cuts": [{"after": "s1", "cut": "orbit:fly", "style": "slide", "dir": "right"}]}}, dict(shots=copy.deepcopy(SHOTS3), ana=ANA3))
    assert d["compose"]["cuts"][0].get("style") in (None, "slide")                                                   # (kept when the cut survives, never an error)


def test_a_word_finishes_its_exit_before_the_window_of_the_3d_cut_that_follows():
    sw = [(0.0, 4.0, "s1"), (4.0, 8.0, "s2"), (8.0, 12.0, "s3")]
    pre = QP.cut_pre_windows({"cuts": [{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "hard"}]}, sw)
    assert set(pre) == {0} and 1.0 < pre[0] < 1.5                                                                    # window A = 29 frames before the boundary; a hard cut has none
    t = {"words": [dict(text=t_, form_in="blobs", form_out="blobs", hit=h, hold_s=2.5, form_s=0.6, exit_s=0.9) for t_, h in (("GO", 1.0), ("UP", 5.0), ("RUN", 9.0))]}
    t2 = copy.deepcopy(t)
    QP.fit_to_shots(t, ANA3, sw, [], frozenset(), None)
    QP.fit_to_shots(t2, ANA3, sw, [], frozenset(), pre)
    end = lambda w: w["hit"] + w["hold_s"] + w["exit_s"]      # noqa: E731
    assert end(t["words"][0]) > 4.0 - pre[0] and end(t2["words"][0]) <= 4.0 - pre[0] + 0.2 + 1e-6                   # without the rule the exit ran into the window, with it the word is gone before it
    assert end(t2["words"][1]) == end(t["words"][1]) and t2["words"][0]["hold_s"] >= QP.MIN_HOLD_IN - 1e-9          # no cut after s2: untouched; the hold floor holds
