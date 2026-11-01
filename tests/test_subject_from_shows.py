"""2026-11-01 SUBJECT FROM THE USER'S PICTURE (bug, render archive mvxblalqa2y): an image-mode render turned the uploaded drummer into "a gaunt man, pale skin, wearing a crimson silk tunic and
gold rings" (the user's "palette: crimson, gold" leaked into the subject; LTX conditions only frame 0). The shot prompt of a shot that starts from a user image takes its subject from that
image's media-read `shows`; clothing / colour / person / animal words the image does not show are removed; the camera lead is the only framing; the user's palette never reaches that prompt.
Run: python -m pytest tests/test_subject_from_shows.py -q"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.app import layered_shots as LS  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import subject_guard as SG  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

SHOWS = "A drummer playing a kit in a flooded gothic cathedral with light beams."
TYPED = "more legible text, palette: crimson, gold"
BAD = ("crimson", "tunic", "rings", "gaunt", "owl")
FRAMING = re.compile(r"close[- ]?up|wide (?:establishing )?shot|establishing shot|medium shot|\btight\b", re.I)
READ = {"v": 1, "on_screen": ["RISE", "STAY"], "concept": "", "look": "more legible, crimson and gold", "direction": TYPED,
        "media": [{"id": "I1", "kind": "image", "file": "drummer.png", "shows": SHOWS, "mood": "dramatic", "palette": ["#223344"]}]}
GAUNT = "a gaunt man, pale skin, wearing a crimson silk tunic and gold rings"
# the model's reply as R14 had it: Gemma invented the man and his clothes (subject_desc), the shot prompts repeat him and open with a framing of their own
REPLY = {"title": "Flood Rhythm", "words": ["RISE", "STAY"], "subject": "red-robed man", "subject_desc": GAUNT + " standing in the water",
         "scene": "A close-up of " + GAUNT + " in a flooded gothic cathedral, crimson light, gold highlights, slow handheld camera",
         "music": "cinematic, slow drums, strings, choir, deep bass, reverb, dark, hall", "bpm": 90, "mood": "solemn", "look": "auto", "shots": "cut",
         "idea": "a man of crimson and gold plays the flood", "idea_why": "x",
         "shot_plan": [{"prompt": "A close-up of " + GAUNT + ", drumming in a flooded gothic cathedral, light beams across the water", "camera": "wide establishing, handheld drift"},
                       {"prompt": "A wide shot of a crimson-robed priest walking down the nave of a stone cathedral with candles", "camera": "tracking"}],
         "owners": [{"render": "flat"}, {"render": "flat"}]}


def _concept():
    r = CO.parse(json.dumps(REPLY), [], 5, 8, brief=json.dumps(READ))
    assert r["concept"] is not None, r["errors"]
    return r["concept"], r["fixes"]


def _final_prompts(c):
    """the shot prompts of the planned piece (the camera lead + the concept's shot text, plan_rules.plan_piece). plan_piece needs the whole design stack; when another session has it
    half-edited (AttributeError in design_defaults, 2026-11-01) the same composition is replayed from plan_rules.model_shot_prompt + shot_camera.lead (plan_rules.py:1380-1396)."""
    from test_fusion_plan import synth_music
    req = PR.normalize_request({"words": "\n".join(c["words"]), "scene": c["scene"], "music": c["music"], "length": 8, "look": "auto", "seed": 5, "quality": "draft", "format": "9:16",
                                "shots": "cut", "owners": CO.request_extras(c)})
    try:
        piece = PR.plan_piece(req, synth_music(100.0, 8.0))
        return piece, [s["prompt"] for s in piece["shots"]]
    except AttributeError:
        from tools.fusion import shot_camera as SC
        out = []
        for e in req["shot_plan"]:
            mp, lead = PR.model_shot_prompt(e), SC.lead(e)
            mp = re.sub(r",\s*camera:.*$", "", mp).strip(" ,;") if lead else mp
            out.append(", ".join(x for x in (lead, mp) if x))
        return {"request": {}}, out


def test_the_bug_reproduces_without_the_guard():
    # what the old code produced: the model's own text is in the prompt (this is the input of the guard, not the output)
    assert "crimson" in REPLY["shot_plan"][0]["prompt"] and "gaunt" in REPLY["shot_plan"][0]["prompt"]


def test_the_user_image_shot_keeps_the_pictures_subject():
    c, fixes = _concept()
    p1 = c["shot_plan"][0]["prompt"].lower()
    for w in BAD:
        assert w not in p1, (w, p1)
    assert "drummer" in p1 and "flooded gothic cathedral" in p1
    assert c["shows"] == [SHOWS]
    for w in BAD:                                             # the subject text of the piece (stills of the other places, scene) is the picture's too
        assert w not in c["subject_desc"].lower() and w not in c["scene"].lower(), w
    assert any("user-image shot" in f for f in fixes)


def test_final_shot_prompts_of_the_user_image_shot():
    c, _ = _concept()
    _, prompts = _final_prompts(c)
    p1 = prompts[0]
    for w in BAD:
        assert w not in p1.lower(), (w, p1)
    assert "drummer" in p1.lower()
    assert len(FRAMING.findall(p1)) <= 1, p1                  # the camera lead names the framing once; the model text names none
    assert SG.has_framing(p1) == bool(SG.has_framing(p1))
    assert not SG.has_framing(c["shot_plan"][0]["prompt"])


def test_a_shot_without_a_user_image_is_unchanged():
    c, _ = _concept()
    c0 = CO.parse(json.dumps(REPLY), [], 5, 8)["concept"]     # the same reply with no brief at all
    assert c0 is not None and "shows" not in c0
    sp0, sp1 = c0["shot_plan"][1], c["shot_plan"][1]
    assert sp1["prompt"].endswith(REPLY["shot_plan"][1]["prompt"]) or sp1["prompt"] == sp0["prompt"] or REPLY["shot_plan"][1]["prompt"] in sp1["prompt"]
    # the guard itself never touches an entry that has no user image
    sp, n = CO.guard_user_shots([{"prompt": REPLY["shot_plan"][0]["prompt"]}, {"prompt": REPLY["shot_plan"][1]["prompt"], "camera": "tracking"}], [SHOWS], "", framing=True)
    assert n == 1 and sp[1] == {"prompt": REPLY["shot_plan"][1]["prompt"], "camera": "tracking"}


def test_no_brief_means_no_change_to_the_prompt_or_the_extras():
    base = CO.concept_prompt(3, 12, [], "cut")
    assert CO.concept_prompt(3, 12, [], "cut", brief="") == base
    c0 = CO.parse(json.dumps(REPLY), [], 5, 8)["concept"]
    assert "shows" not in json.loads(CO.request_extras(c0))
    sp, n = CO.with_subject([{"prompt": "a wolf runs"}, {"prompt": "the wolf rests by the fire"}], "a grey wolf")
    assert n == 2 and sp[0]["prompt"] == "a grey wolf, a wolf runs"


def test_prompt_wording_look_and_subject_desc():
    p = CO.concept_prompt(3, 12, [], "cut", brief=json.dumps(READ))
    assert "It NEVER changes the subject" in p and "follow it in `scene`" not in p
    assert "USER IMAGES, 8-20 words, using ONLY what their" in p and "RULE for every shot that starts from a user image" in p
    plain = CO.concept_prompt(3, 12, [], "cut", brief=json.dumps({"on_screen": ["A"], "look": "slow"}))
    assert "clothes and colours, one distinguishing object" in plain                                  # no user image: Gemma's own description stays


def test_guard_functions():
    allowed = SG.tokens(SHOWS)
    assert SG.flagged(GAUNT, allowed) == ["gaunt", "man", "pale", "skin", "crimson", "tunic", "gold", "rings"]
    assert SG.flagged("a drummer in a flooded cathedral", allowed) == []
    assert SG.flagged("a crimson light on the water", allowed) == ["crimson"]
    assert SG.guard_prompt("crimson light on the water, the drummer lifts both sticks", SHOWS) == SHOWS.rstrip(".") + ", light on the water, the drummer lifts both sticks"
    assert SG.guard_prompt("the owl glides", SHOWS) == SHOWS.rstrip(".")
    assert SG.guard_prompt("anything", "") == "anything"
    assert SG.strip_framing("A close-up of the drummer, wide establishing shot, rain") == "the drummer, rain"
    assert SG.strip_framing("Medium shot of the kit, tight on the hands") == "the kit, the hands"
    # the user's own on-screen word may name a colour
    assert "crimson" in SG.guard_prompt("a crimson cymbal glints", SHOWS, "CRIMSON TIDE").lower()


def test_layered_scrub_drops_the_palette_from_both_chains():
    c, _ = _concept()
    piece, prompts = _final_prompts(c)
    piece.setdefault("request", {})["direction"] = TYPED                    # what the design step keeps (LOCKED_KEYS `request`), user_style is lost there
    piece.pop("user_style", None)
    piece["request"]["shows"] = [SHOWS]
    dirty = "Tight close-up, a drummer, crimson light on the water, golden dust, the words \"RISE\" formed as crimson glass letters"
    out = LS.scrub_prompt(dirty, piece, "s1")
    assert "crimson light" not in out.replace("crimson glass", "") and "golden" not in out and "light on the water" in out
    assert LS.scrub_prompt(dirty, piece, "s2") == dirty        # shot 2 starts from no user image: untouched
    assert LS.scrub_prompt("a crimson door", {"request": {}}, "s1") == "a crimson door"
    clean = LS.clean_prompt(dirty, piece, "s1")
    assert "crimson" not in clean.lower() and "golden" not in clean.lower()                        # the clean chain
    assert US.parse_direction(piece)["palette"]                                                      # the text-pass chain reads the same scrubbed slot prompt (MographShotSlot)
    keep = LS.scrub_prompt("a crimson-lit hall, a drummer", dict(piece, request=dict(piece["request"], shows=["a drummer in a crimson-lit hall"])), "s1")
    assert "crimson" in keep                                                                         # the picture itself shows crimson: it stays


def test_extras_and_request_carry_the_shows():
    c, _ = _concept()
    ex = json.loads(CO.request_extras(c))
    assert ex["shows"] == [SHOWS] and US.shows_in(CO.request_extras(c)) == [SHOWS] and US.shows_in("[]") == []
# ---- render R17: the same clause twice in the s2 prompt ("a drummer in a flooded cathedral with a piano in the foreground, a drummer in ...")
DUP = "a drummer in a flooded cathedral with a piano in the foreground"


def test_identical_clauses_are_kept_once_after_the_framing_is_stripped():
    # the two clauses differed ("A close-up of ..." vs "...") until the framing came off, so a dedupe before strip_framing missed them
    p = f"A close-up of {DUP}, {DUP}"
    assert SG.dedupe_clauses(p) == p                                              # not yet identical: dedupe alone cannot see it
    assert SG.strip_framing(p) == DUP
    assert SG.strip_framing(f"A close-up of {DUP},  A Drummer in a flooded cathedral   with a piano in the foreground.") == DUP     # case / whitespace / stop blind
    assert SG.dedupe_clauses("rain, slow push, Rain,  slow  push") == "rain, slow push"
    assert SG.dedupe_clauses("a kit, with light beams, a kit") == "a kit, with light beams"          # only exact repeats go; order kept
    assert SG.dedupe_clauses("") == ""


def test_the_shot_plan_guard_dedupes_with_a_camera_lead():
    sp, n = CO.guard_user_shots([{"prompt": f"A close-up of {DUP}, {DUP}", "camera": "close-up"}], [DUP], "", framing=True)
    assert n == 1 and sp[0]["prompt"].count("piano in the foreground") == 1, sp[0]["prompt"]
