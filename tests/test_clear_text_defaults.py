"""The draft strips' qualities as DEFAULTS (2026-10-29, user: "can we get more of this stuff happening - clear text and transitions?"): clean holds, one type system, the size floor, a designed
cut at every place change (window_other for up to six letters), at least two places, a different camera in every shot, shot_plan padding, the arm's real model name.
Real data where useful: the v005 batch pieces (tests/fixtures/v005_r0[12]_piece.json) and the saved r7 replies. CPU, no network.
Run: python -m pytest tests/test_clear_text_defaults.py -q"""
from __future__ import annotations

import copy
import json
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

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import design_options as DO  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import shot_camera as SC  # noqa: E402

FX = HERE / "fixtures"


def v005(n):
    return json.loads((FX / f"v005_r0{n}_piece.json").read_text(encoding="utf-8"))


def W(text, font, module="lanes", **k):
    return dict(text=text, font=font, module=module, form_in="lines", form_out="lines", hold_s=2.0, placement={"wf": 0.85, "hf": 0.35}, **k)


# ------------------------------------------------------------------------------------------------ 1a. clean holds
def test_flow_wake_settles_fully_inside_a_hold_and_the_default_is_a_full_settle():
    assert PR.SETTLE_HOLD == 1.0
    tl = PR.settle_timeline([(2.0, 4.0)])
    assert C._timeline(tl, 3.0) == pytest.approx(1.0) and C._timeline(tl, 0.5) == pytest.approx(0.0)
    h, w = 48, 64
    ta = np.zeros((h, w), np.float32)
    ta[18:30, 16:48] = 1.0
    flow = np.zeros((h // 2, w // 2, 2), np.float32)
    flow[..., 0] = 3.0
    matte = np.ones((h, w), np.float32)

    def run(settle_in_hold):
        st = C.FlowWake()
        outs = []
        for f in range(70):
            settle = 0.0 if f < 30 else settle_in_hold                 # frames 0-29: the word forms and is pushed; from 30 on it holds
            outs.append(np.asarray(st._step(ta, flow, matte, strength=4.0, settle=settle, trail=10.0)))
        return outs
    full, partial = run(1.0), run(0.75)
    assert np.abs(full[29] - ta).max() > 0.05                         # it was pushed while forming
    assert np.abs(full[-1] - ta).max() < 0.02                         # a full settle: back on the exact glyph, no echo left
    assert np.abs(partial[-1] - ta).max() > np.abs(full[-1] - ta).max()   # the old 0.75 left a residual push


def test_enforce_compose_replaces_a_models_flow_wake_settle_and_keeps_the_guard_on():
    piece = {"compose": {"legible": False, "default": {"ops": [{"op": "flow_wake", "out": "w", "args": {"type_a": "$type_matte", "settle": 0.0, "strength": 4.0}},
                                                                {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "legible": False}}]}}}
    notes = DD.enforce_compose(piece, [(1.0, 3.0), (5.0, 7.0)])
    st = piece["compose"]["default"]["ops"][0]["args"]["settle"]
    assert C._timeline(st, 2.0) == pytest.approx(1.0) and C._timeline(st, 6.0) == pytest.approx(1.0) and C._timeline(st, 4.0) < 0.05
    assert piece["compose"]["default"]["ops"][1]["args"]["legible"] is True and "legible" not in piece["compose"]
    assert len(notes) == 3 and piece["plan"]["clear_text"] == notes
    assert DD.enforce_compose(piece, [(1.0, 3.0), (5.0, 7.0)]) == []          # idempotent


# ------------------------------------------------------------------------------------------------ 2b. one type system
def test_modules_are_capped_at_max_modules_and_echo_modules_never_survive():
    ws = [W("A", "Anybody[wdth,wght].ttf", "depth_echo"), W("B", "Ultra-Regular.ttf", "pipework"), W("C", "Doto[ROND,wght].ttf", "circuit"), W("D", "Honk-Regular.ttf", "lanes"), W("E", "Tourney-Regular.ttf", "slats")]
    fx = []
    DD.one_module_system(ws, fx)
    mods = [w["module"] for w in ws]
    assert DD.MAX_MODULES == 4 and len(set(mods)) <= DD.MAX_MODULES and not set(mods) & set(DD.AVOID_MODULES) and set(mods) == {"pipework", "circuit", "lanes", "slats"}        # 2026-11-01 user: cap 2 -> 4; the model's own four kept, only the echo word is replaced
    assert len(fx) == 1 and "type.words[0].module" in fx[0]
    ws = ws + [W("F", "Ultra-Regular.ttf", "gradient_fade"), W("G", "Ultra-Regular.ttf", "staircase")]            # (the echo word is already replaced) six clean modules: the cap bites, the two extra slab words take the slab module kept
    fx = []
    DD.one_module_system(ws, fx)
    mods = [w["module"] for w in ws]
    assert len(set(mods)) <= DD.MAX_MODULES and not set(mods) & set(DD.AVOID_MODULES) and len(fx) == 2 and all("type.words[" in f and ".module" in f for f in fx)
    only_noisy = [W("A", "Ultra-Regular.ttf", "depth_echo"), W("B", "Ultra-Regular.ttf", "bold_thin_echo")]
    DD.one_module_system(only_noisy, [])
    assert only_noisy[0]["module"] == only_noisy[1]["module"] and only_noisy[0]["module"] in ("lanes", "pipework", "circuit", "slats", "gradient_fade", "staircase", "beat_bars")


def test_fonts_are_capped_at_three_with_the_expressive_one_kept():
    names = ["Anybody[wdth,wght].ttf", "Ultra-Regular.ttf", "Doto[ROND,wght].ttf", "Honk-Regular.ttf", "Interlope-Regular.ttf", "Tourney-Regular.ttf"]
    ws = [W(f"W{i}", f) for i, f in enumerate(names)]
    ws[2]["hold_s"] = 3.0                                          # the key word: pixel (expressive)
    ws[1]["axes"] = {"wght": 800.0}
    fx = []
    DD.one_font_system(ws, fx)
    faces = [w["font"] for w in ws]
    from mograph import fonts_liked as fl
    assert len(set(faces)) == 3 and "Doto[ROND,wght].ttf" in faces and any(fl.role_of(f) in DD.EXPRESSIVE for f in faces)
    assert all(faces[i] != faces[i + 1] for i in range(len(faces) - 1))         # neighbours differ
    assert any("one type system" in f for f in fx) and all(w.get("axes") is None or w["font"] == "Ultra-Regular.ttf" for w in ws)
    window = [W("A", "Anybody[wdth,wght].ttf"), W("B", "Ultra-Regular.ttf"), W("C", "Doto[ROND,wght].ttf"), W("D", "Honk-Regular.ttf")]
    DD.one_font_system(window, [], protected={0, 1})
    assert window[0]["font"] == "Anybody[wdth,wght].ttf" and window[1]["font"] == "Ultra-Regular.ttf" and len({w["font"] for w in window}) == 3


def test_one_treatment_family_and_two_io_styles():
    ws = [W("A", "Ultra-Regular.ttf", matte="window"), W("B", "Ultra-Regular.ttf", matte="fill"), W("C", "Ultra-Regular.ttf", matte="window")]
    for w, (fi, fo) in zip(ws, (("lines", "blobs"), ("wipe", "particles"), ("flicker", "lines"))):
        w["form_in"], w["form_out"] = fi, fo
    fx = []
    DD.one_treatment(ws, fx)
    assert [w.get("matte") for w in ws] == ["window", None, "window"]
    assert len({w["form_in"] for w in ws}) <= 2 and len({w["form_out"] for w in ws}) <= 2 and any("io styles" in f for f in fx)      # MAX_FORMS 2: a third style (wipe / flicker) lengthens the forms and breaks the window_other fit
    kept = [W("A", "Ultra-Regular.ttf", matte="fill"), W("B", "Ultra-Regular.ttf", matte="window")]
    DD.one_treatment(kept, [], protected={1})
    assert kept[1]["matte"] == "window"                                           # a window word keeps its treatment


def test_enforce_type_on_a_whole_noisy_design_leaves_one_system():
    names = ["Anybody[wdth,wght].ttf", "Ultra-Regular.ttf", "Doto[ROND,wght].ttf", "Honk-Regular.ttf", "Interlope-Regular.ttf"]
    mods = ["depth_echo", "pipework", "particles", "circuit", "lanes"]
    ws = [W(f"W{i}", f, m) for i, (f, m) in enumerate(zip(names, mods))]
    fx = []
    cnt = DD.enforce_type(ws, fx)
    assert len({w["module"] for w in ws}) <= DD.MAX_MODULES and len({w["font"] for w in ws}) <= 3 and cnt["modules"] >= 2 and cnt["fonts"] >= 2        # cap 4 (was 2): only the echo and the speckle word are replaced
    rep = DD.count_repairs(fx)
    assert rep["modules"] == cnt["modules"] and rep["fonts"] >= 2


# ------------------------------------------------------------------------------------------------ 1d. size floor
def test_size_floor_stacks_or_widens_a_word_that_cannot_be_read_on_a_phone():
    small = W("MOMENTUM", "BadeenDisplay-Regular.ttf")
    small["placement"] = {"wf": 0.5, "hf": 0.12}
    cap0 = DD.cap_of(small)[0]
    assert cap0 < DD.CAP_FLOOR
    fx = []
    assert DD.size_floor([small], fx) == 1
    assert not small.get("lines") and small["placement"]["wf"] == DD.MAX_WF and DD.cap_of(small)[0] > cap0 and "floor" in fx[0]          # a single word is NEVER cut across lines (round-3 QC): it is widened to the safe width instead
    phrase = W("MOMENTUM FOREVER", "BadeenDisplay-Regular.ttf")
    phrase["placement"] = {"wf": 0.5, "hf": 0.12}
    DD.size_floor([phrase], [])
    assert phrase.get("lines") in (None, ["MOMENTUM FOREVER"], ["MOMENTUM", "FOREVER"])                                  # a phrase may stack, at its spaces
    ok = W("HI", "Anybody[wdth,wght].ttf")
    ok["placement"] = {"wf": 0.8, "hf": 0.4}
    before = copy.deepcopy(ok)
    assert DD.size_floor([ok], []) == 0 and ok == before                           # a word above the floor is untouched
    measured = W("HI", "Anybody[wdth,wght].ttf")
    assert DD.size_floor([measured], [], caps={0: 0.05}) == 1                      # a measured cap height wins over the estimate


# ------------------------------------------------------------------------------------------------ 2a / 2b. a designed cut at every place change
SHOTS6 = [{"id": f"s{i + 1}", "start": 2.5 * i, "dur": 2.5, "frames": 60, "desc": "the runner sprints to the right", "first": ["A", "last:s1", "B", "last:s3", "C", "last:s5"][i]} for i in range(6)]
ANA = {"downbeats": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0], "chord_changes": [], "cut": 16.0, "fps": 24, "curves": {"rms": [0.5] * 120 + [1.0] * 264}}


def reply(cuts, words=None):
    words = words or [dict(text=t, font="BadeenDisplay-Regular.ttf", hit=1.0 + 2.5 * i, hold_s=1.6, form_s=0.6, placement={"hf": 0.4, "wf": 0.6}) for i, t in enumerate(["ONE", "TWO", "SPIN", "RISE", "KEEPS", "GO"])]
    return {"type": {"words": words}, "compose": {"cuts": cuts}}


def test_every_place_change_gets_a_designed_cut_and_no_kind_repeats():
    out, notes, pins = DO.plan_cuts(reply([{"after": "s1", "cut": "crossfade"}, {"after": "s3", "cut": "hard"}]), copy.deepcopy(SHOTS6), ANA)
    cuts = {c["after"]: c["cut"] for c in out["compose"]["cuts"]}
    place_changes = [a for a, b, _ in DO.boundaries(SHOTS6) if not DO._continuation(DO._row(SHOTS6, b))]
    assert place_changes == ["s2", "s4"]                                       # the boundaries into B and into C
    for a, b, _ in DO.boundaries(SHOTS6):
        row = DO._row(SHOTS6, b)
        if DO._continuation(row):
            assert cuts.get(a, "hard") in ("hard", "crossfade")                    # continuations stay plain
        else:
            assert cuts[a] in DO.SPECIAL or cuts[a] == "window_other", (a, cuts)   # a place change is never a crossfade / hard cut
    heavy = [c for c in cuts.values() if c in DO.SPECIAL or c == "window_other"]
    assert len(heavy) == len(set(heavy)) and sum(1 for c in cuts.values() if c == "window_other") <= 2
    assert not any(c.startswith("depth_dolly") for c in cuts.values())              # the slow dolly is never planned
    assert any("planned" in n for n in notes)


def test_a_model_written_crossfade_at_a_place_change_is_upgraded_and_logged_v005_r02():
    p = v005("2")
    lay = p["layout"]
    rows = [dict(id=s["id"], start=round(lay[s["id"]]["start"] / 24, 2), dur=round(lay[s["id"]]["frames"] / 24, 2), frames=lay[s["id"]]["frames"], first=s["first"], desc=s["prompt"]) for s in p["shots"]]
    assert [r["first"] for r in rows] == ["A", "last:s1", "B", "last:s3", "C"] and {c["cut"] for c in p["compose"]["cuts"]} == {"crossfade"}      # the box piece: every cut a crossfade
    words = [dict(text=w["text"], font=w["font"], hit=w["t0"] + w["form_s"], hold_s=1.6, placement=dict(w["placement"])) for w in p["type"]["words"]]
    rep = {"type": {"words": words}, "compose": {"cuts": [{"after": c["after"], "cut": "crossfade", "why": "soft"} for c in p["compose"]["cuts"]]}}
    ana = dict(ANA, downbeats=[i * 2.0 for i in range(9)])
    out, notes, pins = DO.plan_cuts(rep, rows, ana)
    cuts = {c["after"]: c for c in out["compose"]["cuts"]}
    assert (cuts["s2"]["cut"] == "window_other" or cuts["s2"]["cut"] in DO.SPECIAL) and cuts["s4"]["cut"] in DO.SPECIAL      # both place changes (s2->s3, s4->s5) are designed (a 5-letter word is never cut across lines, so it cannot be the window: a 3D cut)
    assert cuts["s1"]["cut"] == "crossfade" and cuts["s3"]["cut"] == "crossfade"                  # inside a place the model's soft cut stays
    assert any("replaces the design's 'crossfade'" in n for n in notes)
    if cuts["s2"]["cut"] == "window_other":
        w = out["type"]["words"][cuts["s2"]["word"]]
        assert len(w["text"]) <= DO.WO_MAX_LETTERS and DO.cap_estimates(w)[0] >= DO.WINDOW_MIN_CAP * DO.CAP_TOL * 0.99
        assert pins and list(pins)[0] == cuts["s2"]["word"]


def test_window_other_takes_up_to_six_letters_and_two_per_piece_never_neighbouring():
    for t in ("GO", "ONE", "RISE"):
        f = DO.window_fit2({"text": t, "placement": {"wf": 0.6}})
        assert f["ok"] and f["cap"] >= DO.WINDOW_MIN_CAP * DO.CAP_TOL and (f["lines"] is None or "".join(f["lines"]) == t)
    for t in ("KEEPS", "DRIFT", "STORMS"):                                             # 2026-10-30: up to six letters are windows (a word is never cut across lines: the long ones are read vertically, rolled 90)
        f = DO.window_fit2({"text": t})
        assert f["ok"] and f["cap"] >= DO.WINDOW_MIN_CAP * DO.CAP_TOL and (f["lines"] is None or "".join(f["lines"]) == t)
    for t in ("SEVENTH", "SEARCHING"):
        assert not DO.window_fit2({"text": t})["ok"]
    shots = [{"id": f"s{i + 1}", "start": 3.0 * i, "dur": 3.0, "frames": 72, "desc": "x", "first": "ABCDE"[i]} for i in range(5)]
    words = [dict(text=t, font="BadeenDisplay-Regular.ttf", hit=0.5 + 3.0 * i, hold_s=1.6, form_s=0.6, placement={"hf": 0.4, "wf": 0.6}) for i, t in enumerate(["ONE", "TWO", "SPIN", "RISE", "GO"])]
    ana = dict(ANA, downbeats=[3.0 * i for i in range(6)], cut=15.0)
    out, notes, pins = DO.plan_cuts({"type": {"words": words}, "compose": {}}, shots, ana)
    kinds = [(c["after"], c["cut"]) for c in out["compose"]["cuts"]]
    wo = [a for a, k in kinds if k == "window_other"]
    assert len(wo) == 2 and abs(int(wo[0][1:]) - int(wo[1][1:])) > 1                  # two, not neighbouring
    assert all(k in DO.SPECIAL or k == "window_other" for _, k in kinds) and len(kinds) == 4         # every boundary designed, the 3D ones alternate with them
    short = [{"id": "s1", "start": 0.0, "dur": 1.0, "frames": 24, "desc": "x", "first": "A"}, {"id": "s2", "start": 1.0, "dur": 1.0, "frames": 24, "desc": "x", "first": "B"}]
    o2, n2, _ = DO.plan_cuts({"type": {"words": [dict(text="A", hit=0.0, hold_s=1.6)]}, "compose": {}}, short, ana)
    assert not o2["compose"].get("cuts") or all(c["cut"] in ("hard", "window_other") for c in o2["compose"]["cuts"])      # shots too short for a 3D move stay plain


# ------------------------------------------------------------------------------------------------ minimum two places (plan)
def test_no_shot_starts_from_another_shots_last_frame_every_shot_has_its_own_still():
    """user 2026-10-29: "why do all these videos use the last frame to start the next video's first frame? we should either be doing different shots completely or a single long prompted video"."""
    from test_fusion_core import synth_music
    music = synth_music(120, 16)
    req = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto",
           "seed": 5, "quality": "preview", "format": "9:16"}
    for mode in ("chain", "cut"):
        p = PR.plan_piece(dict(req, shots=mode), music)
        firsts = [s["first"] for s in p["shots"]]
        assert len(firsts) >= 3 and not any(f.startswith("last:") for f in firsts) and firsts == [chr(65 + k) for k in range(len(firsts))] and len(p["stills"]) == len(firsts)
        assert p["plan"]["structure"]["last_frame_chaining"] is False
    chain = PR.plan_piece(dict(req, shots="chain"), music)
    assert all(st.get("angle_of") == "A" and st["framing"] and "the same subject and scene as above" in st["prompt"] for st in chain["stills"][1:])
    assert any("angle still" in n for n in chain["plan"]["notes"])
    one = PR.plan_piece(dict(req, shots="single", words="ONE\nTWO"), music)
    assert len(one["shots"]) == 1 and len(one["stills"]) == 1 and one["shots"][0]["first"] == "A"        # a single shot = one long generation from one prompt, no chaining


# ------------------------------------------------------------------------------------------------ the harness (r8): one design through the same functions
def test_saved_r7_reply_through_the_harness_comes_out_as_one_system():
    from tools.fusion import piece_brief as PB
    from tools.fusion import qwen_piece as QP
    f = ROOT / "20_open-model-design" / "v003" / "local" / "r7" / "P14_qwen_s1" / "reply.txt"
    if not f.is_file() or not (ROOT / "15_fusion" / "v004" / "BRIEFS.md").is_file():
        pytest.skip("the r7 replies are not present")
    obj = QP.extract_json(f.read_text(encoding="utf-8"))
    piece, fixes, errors = PB.reply_to_piece(copy.deepcopy(obj), PB.load_brief("P14"))
    assert not errors
    words = piece["type"]["words"]
    assert len({w["module"] for w in words}) <= 2 and len({w["font"] for w in words}) <= 3
    assert all(DD.cap_of(w)[0] is None or DD.cap_of(w)[0] >= DD.CAP_FLOOR - 1e-9 for w in words)
    flow = [o for ch in [piece["compose"].get("default") or {}] + list((piece["compose"].get("shots") or {}).values()) for o in ch.get("ops", []) if o["op"] == "flow_wake"]
    for o in flow:
        assert C._timeline(o["args"]["settle"], (words[0]["t0"] + words[0]["t1"]) / 2) <= 1.0
    r8, _ = PB.build_prompt("P14", "r8")
    r7, _ = PB.build_prompt("P14", "r7")
    assert "## 20. Clear text and one type system" in r8 and "## 21. A different camera in every shot" in r8 and "## 20." not in r7 and len(r8) > len(r7)


def test_a_reply_that_chains_shots_gets_an_angle_still_per_shot_instead():
    from tools.fusion import piece_brief as PB
    obj = {"stills": [{"id": "A", "prompt": "a paddler on a lake at dawn, warm golden light"}],
           "shots": [{"id": "s1", "first": "A", "until": 4, "prompt": "the paddler glides, slow tracking alongside"}, {"id": "s2", "first": "last:s1", "until": 8, "prompt": "extreme close-up on the paddle, static"},
                     {"id": "s3", "first": "last:s2", "until": 12, "prompt": "low angle from the water, pull back"}, {"id": "s4", "first": "last:s3", "until": 16, "prompt": "wide from behind"}]}
    fx = []
    out = PB._no_chain(copy.deepcopy(obj), fx)
    assert [s_["first"] for s_ in out["shots"]] == ["A", "B", "C", "D"] and [s_["id"] for s_ in out["stills"]] == ["A", "B", "C", "D"] and len(fx) == 3
    assert all(st["angle_of"] == "A" and st["angle_fallback"] and "the same subject and scene as above" in st["prompt"] and st["framing"] for st in out["stills"][1:])
    assert out["stills"][1]["framing"] == "extreme close-up" and out["stills"][2]["framing"] == "low"                  # read from the shot prompt
    assert not any(str(s_["first"]).startswith("last:") for s_ in out["shots"])
    # a still the model wrote with angle_of: the code writes its prompt; an edit of an edit is refused
    obj2 = {"stills": [{"id": "A", "prompt": "a runner on a pier"}, {"id": "B", "angle_of": "A", "framing": "close up", "prompt": "whatever"}, {"id": "C", "angle_of": "B", "framing": "low"}],
            "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "B"}, {"id": "s3", "first": "C"}]}
    fx2 = []
    o2 = PB._no_chain(copy.deepcopy(obj2), fx2)
    assert o2["stills"][1]["framing"] == "close-up" and "whatever" not in o2["stills"][1]["prompt"] and "angle_of" not in o2["stills"][2] and any("edit of an edit" in f for f in fx2)
    two = {"stills": [{"id": "A", "prompt": "a"}, {"id": "B", "prompt": "b"}], "shots": [{"id": "s1", "first": "A"}, {"id": "s2", "first": "B"}]}
    assert PB._no_chain(copy.deepcopy(two), []) == two


# ------------------------------------------------------------------------------------------------ 9. a different camera in every shot
def test_camera_parse_and_enforce():
    assert SC.parse("a low-angle crane up over the roofs") == ("low angle", "crane up") and SC.parse("backs away keeping him in frame")[1] == "pull back"
    ents = [{"prompt": "a", "camera": "slow push in"}, {"prompt": "b", "camera": "slow push in"}, {"prompt": "c", "camera": "close-up, push in"}, {"prompt": "d"},
            {"prompt": "e", "camera": "wide shot, pull back", "continues": True}]
    out, fx = SC.enforce(ents, 5)
    pairs = [SC.parse(e["camera"]) for e in out]
    assert all(p[0] and p[1] for p in pairs) and len(set(pairs)) == 5                                  # no pair repeats
    assert all(pairs[i][0] != pairs[i + 1][0] and pairs[i][1] != pairs[i + 1][1] for i in range(4))     # nothing repeats next to itself
    assert len(fx) >= 3 and all("shot_plan[" in f for f in fx) and out[4].get("continues") is True        # continuation keeps its flag, changes the camera
    assert SC.enforce([], 3) == ([], [])
    good = [{"prompt": "a", "camera": "wide, tracking alongside"}, {"prompt": "b", "camera": "close-up, static"}, {"prompt": "c", "camera": "low angle, pull back"}]
    o2, f2 = SC.enforce(copy.deepcopy(good), 3)
    assert f2 == [] and o2 == good                                                                      # a varied plan is left alone


def test_concept_normalize_varies_the_cameras_and_the_still_carries_the_framing():
    from tools.fusion import concept as CO
    good = {"title": "T", "words": ["ONE", "TWO", "THREE", "FOUR"], "subject": "old fisherman", "scene": "an old fisherman mending a net on a grey harbour wall at dawn, soft light, wide shot, calm water behind him",
            "music": "folk, guitar, fiddle, warm, gentle, light percussion, tender, acoustic", "bpm": 90, "mood": "tender", "look": "auto", "shots": "cut",
            "shot_plan": [{"prompt": f"The fisherman mends a net on the harbour wall number {k}, grey light, slow", "camera": "slow push in"} for k in range(4)]}
    c, errors, fixes = CO.normalize(good, 3, [], 16)
    pairs = [SC.parse(e["camera"]) for e in c["shot_plan"]]
    assert len(set(pairs)) == 4 and all(pairs[i][1] != pairs[i + 1][1] for i in range(3)) and sum("camera" in f for f in fixes) >= 3
    sp = [{"prompt": "The fisherman mends a net on the harbour wall, grey light, calm water", "camera": "close-up, static"}]
    assert "close-up shot" in PR.place_still(sp, 0)


# ------------------------------------------------------------------------------------------------ the arm's real model name
def test_gguf_arms_write_their_real_model_name_into_the_graph_labels():
    from tools.app import llm_arms as LA
    p = ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers.json"
    if not p.is_file():
        pytest.skip("graph not present")
    g0 = json.loads(p.read_text(encoding="utf-8"))
    g, _ = LA.convert(g0, None, "gemma31b_qat")
    labels = {n["inputs"].get("model") for n in g.values() if isinstance(n, dict) and n.get("class_type") in LA.LABELLED}
    assert labels == {"Qwen3.8-27B Q5_K_M"}


def test_a_repaired_move_removes_the_old_move_from_the_prompt_so_they_do_not_contradict():
    ents = [{"prompt": "a man walks along the pier, the camera pushes in on him", "camera": "wide, push in"},
            {"prompt": "a woman runs across the dune, the camera slowly pushes in on her, warm light", "camera": "close-up, slow push in"}]
    out, fx = SC.enforce(ents, 2)
    assert SC.parse(out[1]["camera"])[1] != "push in" and "pushes in" not in out[1]["prompt"] and "warm light" in out[1]["prompt"] and out[0]["prompt"] == ents[0]["prompt"]
    assert fx and "repeats the shot before" in fx[0]


def test_saved_r8_replies_a_downgraded_window_choice_at_a_place_change_becomes_a_3d_cut():
    """r8 P14 Qwen wrote a window_cut with no word holding across the cut (the validator downgrades it to hard); P06 Qwen a window_swap likewise. A place change is never left plain."""
    from tools.fusion import piece_brief as PB
    from tools.fusion import qwen_piece as QP
    for cell, pid in (("P14_qwen_s1", "P14"), ("P06_qwen_s1", "P06")):
        f = ROOT / "20_open-model-design" / "v003" / "local" / "r8" / cell / "reply.txt"
        if not f.is_file() or not (ROOT / "15_fusion" / "v004" / "BRIEFS.md").is_file():
            pytest.skip("the r8 replies are not present")
        piece, fixes, errors = PB.reply_to_piece(QP.extract_json(f.read_text(encoding="utf-8")), PB.load_brief(pid))
        assert not errors
        lay = piece["layout"]
        ids = [s["id"] for s in piece["shots"]]
        cuts = {c["after"]: c for c in (piece["compose"].get("cuts") or [])}
        for k, (a, b) in enumerate(zip(ids, ids[1:])):
            first = piece["shots"][k + 1]["first"]
            if str(first).startswith("last:"):
                continue
            kind = cuts[a].get("was") or cuts[a]["cut"]
            assert kind in DO.SPECIAL or kind == "window_other", (cell, a, kind, [f for f in fixes if "cuts" in f])
        assert any("cuts:" in f and ("planned" in f or "downgraded" in f) for f in fixes)


# ------------------------------------------------------------------------------------------------ legibility screen (24 GB QC 2026-10-29)
def test_hard_to_read_faces_only_on_long_words_and_no_box_wider_than_the_safe_area():
    ws = [W("TURN", "Monofett-Regular.ttf"), W("BREATHING", "Monofett-Regular.ttf"), W("AWAKE", "Anybody[wdth,wght].ttf"), W("GO", "BungeeOutline-Regular.ttf")]
    ws[2]["placement"]["wf"] = 1.0
    ws[0]["axes"] = {"wght": 400.0}
    fx = []
    DD.legibility_screen(ws, fx, protected={3})
    assert ws[0]["font"] != "Monofett-Regular.ttf" and "axes" not in ws[0] and (ws[1]["font"] != "Monofett-Regular.ttf")      # TURN (read "TURO") swapped; BREATHING (9 letters) too: a 9-letter word on one line cannot reach the 12 % cap a risky face needs (round-3 QC)
    assert ws[2]["placement"]["wf"] == DD.MAX_WF and ws[3]["font"] == "BungeeOutline-Regular.ttf"                         # the box stays in the safe area; a protected word is left alone
    assert ws[0]["font"] == "Anybody[wdth,wght].ttf" and any("hard to read" in f for f in fx) and any("safe area" in f for f in fx)      # a face the piece already uses
    used = [W("A", "Anybody[wdth,wght].ttf"), W("B", "Silkscreen-Regular.ttf"), W("C", "Monofett-Regular.ttf")]
    DD.legibility_screen(used, [])
    assert used[2]["font"] in ("Anybody[wdth,wght].ttf", "Silkscreen-Regular.ttf") + DD.HEAVY_READER_FILES                                          # a face the piece already uses: the face count does not grow


def test_two_style_and_extruded_modules_are_folded_into_the_reading_module():
    ws = [W("A", "Ultra-Regular.ttf", "scale_clash"), W("B", "Anybody[wdth,wght].ttf", "hollow_tubes"), W("C", "Honk-Regular.ttf", "solid_hollow"), W("D", "Tourney-Regular.ttf", "lanes"), W("E", "Doto[ROND,wght].ttf", "topo_tower")]
    fx = []
    DD.one_module_system(ws, fx)
    assert {w["module"] for w in ws} == {"lanes"} and sum("two styles" in f for f in fx) == 2 and sum("smears on footage" in f for f in fx) == 2
    assert not set(DD.READING_FIRST) & set(DD.AVOID_MODULES) and set(DD.AVOID_MODULES) >= set(DD.HOLD_NOISY)


# ------------------------------------------------------------------------------------------------ QC v026: pixel / dot faces never with ray / extrusion modules, only short big words
JAC, GEIST, ANY, PASSION, BARRIO = "Jacquard12-Regular.ttf", "GeistPixel[ELSH].ttf", "Anybody[wdth,wght].ttf", "PassionOne-Black.ttf", "Barrio-Regular.ttf"


def test_pixel_face_with_a_ray_or_extrusion_module_is_swapped_for_a_readable_face():
    """r02 'WHAT' (Jacquard 12 + gradient_fade) and r02 'FORGETS TURN' were unreadable: the pixel face is replaced, the module of the piece stays."""
    for mod in ("gradient_fade", "tangent_spray", "spectrum", "voxel_orbit", "voxel_roll", "particles", "outflow", "fly_into", "staircase"):
        words = [W("WHAT", JAC, mod), W("HANDS", PASSION, mod)]
        fixes: list = []
        n = DD.pixel_guard(words, fixes, caps={0: 0.2})
        assert n == 1 and words[0]["font"] != JAC, mod
        assert DD.pixel_problem(words[0], 0.2) is None
        assert words[0]["module"] == mod                                         # only the face moves
        assert any("never drawn by the extrusion / ray / spray module" in f for f in fixes)


def test_pixel_face_keeps_a_short_big_word_on_a_plain_module_and_drops_a_long_or_small_one():
    ok = W("GLASS", JAC, "slats")
    assert DD.pixel_problem(ok, 0.2) is None and DD.pixel_problem(W("GLOW", GEIST, "beat_bars"), 0.15) is None
    assert "at most 6 letters" in DD.pixel_problem(W("EVERY NAME", JAC, "beat_bars"), 0.3)
    assert "cap height" in DD.pixel_problem(W("GLOW", JAC, "beat_bars"), 0.08)             # fitted far below 12 %
    assert DD.pixel_problem(W("EVERYTHING", PASSION, "gradient_fade"), 0.05) is None        # a plain face is not this rule's business
    assert DD.pixel_problem(dict(W("GLASS", JAC, "gradient_fade"), render="handoff"), 0.2) is None   # an LTX-owned word keeps the graphic its footage was made from


def test_enforce_type_runs_the_pixel_guard_and_the_piece_keeps_a_characterful_face():
    words = [W("WHAT", JAC, "gradient_fade"), W("REMEMBER", ANY, "gradient_fade")]
    fixes: list = []
    DD.enforce_type(words, fixes)
    from mograph import fonts_liked as fl
    assert (fl.lookup(words[0]["font"]) or {})["key"] not in DD.PIXEL_DOT_FACES
    assert fl.lookup(words[0]["font"])["role"] in DD.EXPRESSIVE or fl.lookup(words[0]["font"])["role"] in ("heavy", "tall", "text")


def test_stable_type_enforce_swaps_a_pixel_face_that_the_final_layout_cannot_carry():
    """the grid layout of a long pixel word is the r03 'EVERY NAME' case: ST.enforce measures the cap of the FITTED layout and swaps the face."""
    from tools.fusion import stable_type as ST
    t = dict(words=[dict(W("EVERY NAME", "JacquardaBastarda9-Regular.ttf", "beat_bars"), hold_s=2.0), dict(W("KEEP", PASSION, "beat_bars"), hold_s=2.0)], camera="beat_orbit")
    fixes: list = []
    ST.enforce(t, dict(seed=3, layouts=["grid", "line"]), 90.0, fixes)
    assert t["words"][0]["font"] != "JacquardaBastarda9-Regular.ttf"
    assert any("pixel / dot face" in f for f in fixes)
    short = dict(words=[W("GLOW", JAC, "slats")], camera="beat_orbit")
    ST.enforce(short, dict(seed=3, layouts=["line"]), 90.0, [])
    assert short["words"][0]["font"] == JAC                                      # a 4-letter word on slats at a big cap keeps its pixel face
