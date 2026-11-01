"""The user's per-line font (storyboard STORY v2: brief.story.shots[].font) wins over every font rule (tools/fusion/font_pins.py). CPU, no network.
Run: python -m pytest tests/test_font_pins.py -q"""
from __future__ import annotations

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

from test_design_salvage import REQ  # noqa: E402
from test_fusion_core import synth_music  # noqa: E402
from mograph import fonts_liked as fl  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import font_pins as FP  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402

LINES = ["STILL", "SEARCHING", "A LIGHT", "AHEAD", "KEEP", "GOING"]
CHOSEN = {1: "Silkscreen-Regular.ttf", 3: "BigShouldersDisplay[wght].ttf", 4: "Honk[MORF,SHLN].ttf"}      # the user's picks for lines 2, 4, 5 (the rest: the system decides)


def fonts_list():
    return [CHOSEN.get(i) for i in range(len(LINES))]


def request(**kw):
    ex = {"owners": [], "fonts": fonts_list()}
    return dict(REQ, owners=json.dumps(ex), **kw)


# ----------------------------------------------------------------------------------------------------------- matching + validation
def test_pins_of_accepts_files_keys_families_and_nulls():
    pins = FP.pins_of(["Silkscreen-Regular.ttf", None, "BigShouldersDisplay", "Big Shoulders Display"], lines=["A", "B", "C", "D"])
    assert [p["font"] for p in pins] == ["Silkscreen-Regular.ttf", None, "BigShouldersDisplay[wght].ttf", "BigShouldersDisplay[wght].ttf"]
    assert FP.pins_of([None, None], lines=["A", "B"]) == [] and FP.pins_of(None) == [] and FP.pins_of([{"render": "flat"}, {"render": "ltx"}]) == []     # owners lists are not font lists
    assert FP.pins_of(json.dumps({"owners": [], "fonts": ["Doto[ROND,wght].ttf"]}), lines=["X"])[0]["font"] == "Doto[ROND,wght].ttf"


def test_a_pin_that_is_not_liked_raises_in_strict_mode_and_is_substituted_in_production():
    with pytest.raises(fl.FontNotLiked):
        FP.pins_of(["Arial.ttf"], lines=["A"])
    os.environ["MOGRAPH_FONT_STRICT"] = "0"
    try:
        p = FP.pins_of(["Arial.ttf"], lines=["A"])
    finally:
        os.environ["MOGRAPH_FONT_STRICT"] = "1"
    assert fl.is_liked(p[0]["font"])


def test_pin_for_matches_by_text_then_containment_then_position():
    pins = [{"text": "STILL", "font": "a"}, {"text": "SEARCHING", "font": None}, {"text": "A LIGHT", "font": "c"}]
    assert FP.pin_for("still", 0, 3, pins) == "a" and FP.pin_for("SEARCHING", 1, 3, pins) is None
    assert FP.pin_for("A LIGHT AHEAD", 2, 2, pins) == "c"                    # a merged lockup
    assert FP.pin_for("XYZ", 2, 3, pins) == "c"                              # same count -> same position
    assert FP.pin_for("XYZ", 2, 4, pins) is None


def test_apply_changes_only_pinned_words_and_drops_their_axes():
    ws = [{"text": "A", "font": "Anybody[wdth,wght].ttf", "axes": {"wght": 100}}, {"text": "B", "font": "Doto[ROND,wght].ttf", "axes": {"wght": 700}}]
    fx = []
    n = FP.apply(ws, [{"text": "A", "font": "Silkscreen-Regular.ttf"}, {"text": "B", "font": None}], fx)
    assert n == 1 and ws[0]["font"] == "Silkscreen-Regular.ttf" and "axes" not in ws[0] and ws[1]["axes"] == {"wght": 700} and ws[1]["font"] == "Doto[ROND,wght].ttf" and len(fx) == 1
    assert FP.apply(ws, [{"text": "A", "font": "Silkscreen-Regular.ttf"}], fx) == 0                    # idempotent


# ----------------------------------------------------------------------------------------------------------- concept -> request
def test_story_fonts_travel_in_the_concept_extras_and_the_request():
    story = {"title": "T", "logline": "L", "shots": [{"n": i + 1, "line": ln, "beat": "b", "font": CHOSEN.get(i), **({"font_user": True} if i in CHOSEN else {})} for i, ln in enumerate(LINES)]}      # font_user = the user changed it in the page (R17b)
    story["shots"][0]["font"] = "Arial.ttf"                                  # not liked: dropped, never a user choice
    b = CO.parse_brief({"on_screen": LINES, "story": story})
    want = [None, "Silkscreen-Regular.ttf", None, "BigShouldersDisplay[wght].ttf", "Honk[MORF,SHLN].ttf", None]
    assert [x.get("font") for x in b["story"]["shots"]] == want
    c = {"words": [ln.upper() for ln in LINES], "brief": b}
    assert CO.story_fonts(c) == want
    ex = json.loads(CO.request_extras(dict(c, owners=[], shot_plan=[], idea="")))
    assert ex["fonts"] == want
    req = PR.normalize_request(dict(REQ, words="\n".join(c["words"]), owners=json.dumps(ex)))
    assert [p["font"] for p in req["font_pins"]] == want
    assert PR.normalize_request(req)["font_pins"] == req["font_pins"]        # a normalised request keeps them
    plain = CO.request_extras({"words": ["A"], "owners": [{"render": "flat"}], "brief": CO.parse_brief({"on_screen": ["A"]})})
    assert "fonts" not in plain                                              # no user font: the extras are what they were


# ----------------------------------------------------------------------------------------------------------- plan -> design -> type passes
def test_plan_puts_the_chosen_font_on_its_line_and_the_rest_stays_liked():
    music = synth_music(160, 24)
    plan = PR.plan_piece(request(length=24), music)
    ws = plan["type"]["words"]
    assert len(ws) == len(LINES)
    for i, f in CHOSEN.items():
        assert ws[i]["font"] == f and ws[i]["font_pin"] == f, (i, ws[i]["text"])
    assert all(fl.is_liked(w["font"]) for w in ws) and plan["type"]["font_pins"]
    base = PR.plan_piece(dict(REQ), music)                                   # no user font: no pins key, the plan is what it was
    assert "font_pins" not in base["type"] and not any("font_pin" in w for w in base["type"]["words"])


def test_lines_the_planner_merged_into_one_lockup_take_the_first_chosen_font_among_them():
    plan = PR.plan_piece(request(), synth_music(160, 12))                    # 12 s: six lines fit three lockups
    ws = plan["type"]["words"]
    assert len(ws) < len(LINES)
    assert ws[0]["text"] == "STILL SEARCHING" and ws[0]["font"] == CHOSEN[1]


def test_the_design_step_cannot_undo_a_pin():
    music = synth_music(160, 24)
    plan = PR.plan_piece(request(length=24), music)
    reply = DS.reply_from_piece(plan, music)
    for w in reply["type"]["words"]:                                         # the model picks one plain face for every line (the v005 complaint) ...
        w["font"] = "SpaceGrotesk-Bold.ttf"
        w.pop("axes", None)
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        d, fixes, errors = QP.normalize_design(reply, name, plan_windows=True)
        QP.finish_design(d, name, fixes)
    ws = d["type"]["words"]
    for i, f in CHOSEN.items():                                              # ... the user's own lines still win,
        assert ws[i]["font"] == f
    assert ws[1]["module"] in ("beat_bars", "slats", "lanes")               # (a pixel face makes its MODULE give way, the face stays)
    free = [w["font"] for i, w in enumerate(ws) if i not in CHOSEN]
    assert all(fl.is_liked(f) for f in free)
    assert len({w["font"] for w in ws}) >= 3                                 # ... and the lines nobody chose still get the piece's variety rules
    out = DS.parse(json.dumps(reply), DS.make_locked(plan), plan, music)     # the whole parse route (salvage / repair / lock-back)
    fw = out["piece"]["type"]["words"]
    for i, f in CHOSEN.items():
        assert fw[i]["font"] == f
    assert out["piece"]["type"]["font_pins"]


def test_build_scene_is_the_last_guard():
    from tools.fusion import spec as SPEC
    music = synth_music(160, 24)
    plan = PR.plan_piece(request(length=24), music)
    t = copy.deepcopy(plan["type"])
    for w in t["words"]:
        w["font"] = "SpaceGrotesk-Bold.ttf"                                  # a later step wrote the plain face over every line
        w.pop("axes", None)
    used = []
    from mograph.modules import core
    real = core.resolve_font
    core.resolve_font = lambda name, *a, **k: (used.append(name), real(name, *a, **k))[1]
    try:
        SPEC.build_scene(dict(t, size=[128, 256]), size=(128, 256))
    finally:
        core.resolve_font = real
    assert set(CHOSEN.values()) <= {u for u in used if isinstance(u, str)}                                # every chosen face was opened by the type engine


# ----------------------------------------------------------------------------------------------------------- story camera (the cut after shot k)
def test_story_camera_passes_through_validated_and_becomes_cut_pins():
    cams = ["orbit:push", "hold", "pan:left", "nonsense", "depth_pivot", "orbit:fly"]
    story = {"title": "T", "logline": "L", "shots": [{"n": i + 1, "line": ln, "beat": "b", "camera": cams[i]} for i, ln in enumerate(LINES)]}
    b = CO.parse_brief({"on_screen": LINES, "story": story})
    assert [x.get("camera") for x in b["story"]["shots"]] == ["orbit:push", "hold", "pan:left", None, "depth_pivot", "orbit:fly"]
    c = {"words": LINES, "brief": b}
    assert CO.story_cuts(c) == ["orbit:push", "hard", None, None, "depth_pivot", None]          # pan stays in the brief only; the last shot has no cut
    ex = json.loads(CO.request_extras(dict(c, owners=[], shot_plan=[], idea="")))
    assert ex["cuts"] == CO.story_cuts(c) and "fonts" not in ex
    assert PR.normalize_request(dict(REQ, owners=json.dumps(ex)))["cut_pins"] == ["orbit:push", "hard", None, None, "depth_pivot", None]
    assert CO.parse_brief({"on_screen": LINES, "story": {"title": "T", "shots": [{"line": "A", "beat": "b"}]}})["story"]["shots"] == [{"n": 1, "image": None, "start": None, "end": None, "beat": "b", "line": "A"}]


def test_the_design_step_pins_the_users_cut_after_a_shot():
    music = synth_music(160, 24)
    cuts = ["orbit:pullback", "hard", None, None, None, None]
    plan = PR.plan_piece(dict(request(length=24), owners=json.dumps({"owners": [], "fonts": fonts_list(), "cuts": cuts})), music)
    assert plan["plan"]["cut_pins"] == cuts
    reply = DS.reply_from_piece(plan, music)
    with DS.app_context(plan, music, DS.make_locked(plan)) as name:
        d, fixes, errors = QP.normalize_design(reply, name, plan_windows=True)
        QP.finish_design(d, name, fixes)
    got = {c["after"]: c["cut"] for c in (d.get("compose") or {}).get("cuts") or []}
    assert got.get("s1") == "orbit:pullback" and got.get("s2") == "hard", got


# ----------------------------------------------------------------------------------------------------------- R17b: the page's random DEFAULT font is not a pin
def test_a_default_font_without_font_user_is_never_a_pin():
    """R17b: nobody picked a font, yet both words carried font_pin: the page's fillFonts default rode in brief.story.shots[].font and concept.story_fonts read it as the user's choice."""
    story = {"title": "T", "logline": "L", "shots": [{"n": i + 1, "line": ln, "beat": "b", "font": "Silkscreen-Regular.ttf"} for i, ln in enumerate(LINES)]}
    b = CO.parse_brief({"on_screen": LINES, "story": story})
    assert all(x.get("font") and "font_user" not in x for x in b["story"]["shots"]), "the default stays on the shot for display"
    c = {"words": [ln.upper() for ln in LINES], "brief": b}
    assert CO.story_fonts(c) == []
    assert "fonts" not in json.loads(CO.request_extras(dict(c, owners=[], shot_plan=[], idea="")))
    story["shots"][2]["font_user"] = "yes"                                   # only a real True counts
    assert CO.story_fonts({"words": c["words"], "brief": CO.parse_brief({"on_screen": LINES, "story": story})}) == []
    story["shots"][2]["font_user"] = True
    got = CO.story_fonts({"words": c["words"], "brief": CO.parse_brief({"on_screen": LINES, "story": story})})
    assert got[2] == "Silkscreen-Regular.ttf" and [x for i, x in enumerate(got) if i != 2] == [None] * (len(LINES) - 1)


def test_the_page_marks_only_the_users_own_font_picks():
    """web_app/bind.js briefFromStory(opts.userFonts) sets font_user on exactly those shots; app.js currentBrief passes the shots whose `n.font` the user edited (stepper / randomise)."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = ("const B=require(process.argv[1]);const read=JSON.stringify({on_screen:['A'],concept:'c',look:'l',music:'m',media:[]});"
          "const st={title:'T',logline:'L',shots:[{n:1,line:'A',beat:'b',font:'x.ttf'},{n:2,line:'B',beat:'b',font:'y.ttf'},{n:3,line:'C',beat:'b',font:''}]};"
          "const o=(u)=>JSON.parse(B.briefFromStory(read,st,u?{userFonts:u}:{})).story.shots.map(s=>[s.font,!!s.font_user]);"
          "console.log(JSON.stringify({none:o(),two:o([2]),all:o([1,2,3])}));")
    r = subprocess.run([node, "-e", js, str(ROOT / "web_app" / "bind.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["none"] == [["x.ttf", False], ["y.ttf", False], ["", False]]
    assert out["two"] == [["x.ttf", False], ["y.ttf", True], ["", False]]
    assert out["all"] == [["x.ttf", True], ["y.ttf", True], ["", False]], "no font = nothing to pin"
    app = (ROOT / "web_app" / "app.js").read_text(encoding="utf-8")
    assert "S.story.edited[`${x.n}.font`] && x.font" in app and "userFonts })" in app
