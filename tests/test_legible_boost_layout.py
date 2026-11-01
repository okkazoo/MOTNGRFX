"""2026-11-01, real GPU render R17b (direction "more legible text, palette: crimson, gold", user_style legible_boost True): SACRED NOISE came out as a beat_bars GRID split mid-word
(SAC / RED / NOI / SE) in BitcountSingle (pixel), TOTAL LOSS in JacquardaBastarda9 (pixel blackletter). tools/fusion/user_layout.py now, WITH the legibility wish only,
(1) re-lays a mid-word split as whole words (line / stack, the bigger cap that passes the size floor) and (2) swaps a risky face (design_defaults.MAIN_HOLD_RISKY, pixel / dot-matrix /
blackletter) for a readable LIKED one and records both in piece.user_style_log. Without the wish nothing changes.
Run: python -m pytest tests/test_legible_boost_layout.py -q"""
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from mograph import fonts_liked as fl  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import stable_type as ST  # noqa: E402
from tools.fusion import user_layout as UL  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

TYPED = "more legible text, palette: crimson, gold"
BITCOUNT = "BitcountSingle[CRSV,ELSH,ELXP,slnt,wght].ttf"
JACQ = "JacquardaBastarda9-Regular.ttf"
LIKED_FILES = {e["file"] for e in json.loads((ROOT / "mograph" / "fonts" / "liked.json").read_text(encoding="utf-8"))["fonts"]}


def _piece(direction=TYPED, pinned=False):
    words = [{"text": "SACRED NOISE", "font": BITCOUNT, "font_pin": BITCOUNT, "axes": None, "lines": ["SAC", "RED", "NOI", "SE"], "module": "beat_bars",
              "placement": {"dx": 0.0, "dy": 0.0, "hf": 0.88, "pitch": 0.0, "roll": 0.0, "wf": 0.9, "yaw": 0.0}, "t0": 0.0, "t1": 3.9},
             {"text": "TOTAL LOSS", "font": JACQ, "font_pin": JACQ, "axes": None, "lines": ["TOTAL", "LOSS"], "module": "beat_bars",
              "placement": {"dx": 0.0, "dy": 0.0, "hf": 0.347, "pitch": 0.0, "roll": 0.0, "wf": 0.9, "yaw": 0.0}, "t0": 4.5, "t1": 8.0}]
    p = {"id": "R17b", "fps": 24, "format": "9:16", "quality": "draft",
         "type": {"duration": 8.0, "words": words, "font_pins": [{"font": BITCOUNT, "text": "SACRED NOISE"}, {"font": JACQ, "text": "TOTAL LOSS"}]},
         "request": {"format": "9:16", "length": 8, "direction": direction}}
    if not pinned:                                                   # a face the random / design system chose: no font_pin, no type.font_pins
        for w in words:
            w.pop("font_pin")
        del p["type"]["font_pins"]
    US.stamp(p, direction)
    return p


def _whole_words(w):
    toks = w["text"].split()
    return " ".join(w["lines"]).split() == toks


def test_risky_set_is_the_design_defaults_list_plus_pixel_faces():
    for k in DD.MAIN_HOLD_RISKY:
        assert UL.is_risky(k), k
    for f in (BITCOUNT, JACQ, "Silkscreen-Regular.ttf", "Jacquard12-Regular.ttf", "Jacquard24-Regular.ttf", "Bitcount[wght].ttf", "PixelifySans.ttf"):
        assert UL.is_risky(f), f
    for f in ("PassionOne-Black.ttf", "Anybody[wdth,wght].ttf", "Blackout_Midnight.ttf", "Fascinate-Regular.ttf"):
        assert not UL.is_risky(f), f


def test_boost_never_splits_a_word_mid_word():
    p, fixes = _piece(), []
    assert UL.apply(p, fixes) == 2
    for w in p["type"]["words"]:
        assert _whole_words(w), w["lines"]
        assert w["placement"]["roll"] == 0.0
    sac = p["type"]["words"][0]
    assert sac["lines"] in (["SACRED", "NOISE"], ["SACRED NOISE"])
    best = max((b for b in (ST.fit_box("SACRED NOISE", sac["font"], None, c) for c in ("line", "stack")) if b["ok"]), key=lambda b: b["cap"])
    assert sac["lines"] == best["lines"] and abs(sac["placement"]["hf"] - best["hf"]) < 1e-6, "the upright whole-word layout with the bigger cap that passes the size floor"
    assert best["cap"] >= UL.MIN_CAP, "the face was chosen so the whole-word layout is boost-sized"
    assert any("mid-word" in x for x in fixes)


def test_boost_swaps_a_risky_face_for_a_liked_readable_one():
    p, fixes = _piece(), []
    UL.apply(p, fixes)
    ws = p["type"]["words"]
    for w, old in zip(ws, (BITCOUNT, JACQ)):
        assert w["font"] != old and w["font"] in LIKED_FILES and not UL.is_risky(w["font"]), w["font"]
        assert (fl.lookup(w["font"]) or {}).get("role") in ("heavy", "tall", "ornamental")
        assert w["font"] not in DD.MAIN_HOLD_RISKY and fl.lookup(w["font"])["key"] not in DD.MAIN_HOLD_RISKY + DD.PIXEL_DOT_FACES
        assert w.get("axes") is None
    assert ws[0]["font"] != ws[1]["font"], "two risky words do not get the same face"
    assert p["user_style_log"] == fixes and sum("font" in x and "risky" in x for x in fixes) == 2


def test_a_font_the_user_picked_is_never_swapped():
    """the app sends only the user's own picks (web_app/app.js storyExtras `fonts`) -> type.font_pins + word.font_pin: those faces stay, the words are still whole."""
    for variant in ("both", "word_only", "pins_only"):
        p = _piece(pinned=True)
        if variant == "word_only":
            del p["type"]["font_pins"]
        if variant == "pins_only":
            for w in p["type"]["words"]:
                w.pop("font_pin")
        before = [w["font"] for w in p["type"]["words"]]
        pins = copy.deepcopy(p["type"].get("font_pins"))
        fixes = []
        UL.apply(p, fixes)
        ws = p["type"]["words"]
        assert [w["font"] for w in ws] == before == [BITCOUNT, JACQ], variant
        assert p["type"].get("font_pins") == pins and not any(".font " in x for x in fixes), variant
        assert all(_whole_words(w) for w in ws), "the layout is still re-laid as whole words"
    one = _piece(pinned=True)
    del one["type"]["words"][1]["font_pin"]
    one["type"]["font_pins"] = one["type"]["font_pins"][:1]            # only word 0 is the user's: word 1 (the system's) is swapped
    UL.apply(one)
    assert one["type"]["words"][0]["font"] == BITCOUNT and one["type"]["words"][1]["font"] != JACQ


def test_the_new_layout_fits_the_face():
    p = _piece(pinned=True)
    UL.apply(p)
    for w in p["type"]["words"]:
        b = ST.fit_box(w["text"], w["font"], None, "stack" if len(w["lines"]) > 1 else "line")
        assert b["lines"] == w["lines"] and abs(b["hf"] - w["placement"]["hf"]) < 1e-6 and abs(b["wf"] - w["placement"]["wf"]) < 1e-6
    q = _piece()
    UL.apply(q)
    for w in q["type"]["words"]:
        assert ST.fit_box(w["text"], w["font"], None, "stack")["ok"], "a swapped face passes the size floor"


def test_without_boost_the_words_are_unchanged():
    for direction in ("", "palette: crimson, gold", "keep it plain"):
        p = _piece(direction)
        before = copy.deepcopy(p)
        fixes = []
        UL.apply(p, fixes)
        # 2026-11-01 (render R19): without the wish only the RARE mid-word-split pass runs (tests/test_midword_rare.py): the split word may be re-laid as whole words, nothing else changes
        assert all(x.startswith("type.words[0]") for x in fixes), direction                 # only the split word (SACRED NOISE; a face swap only if whole words cannot reach the floor in it)
        p0, b0 = p["type"]["words"][0], before["type"]["words"][0]
        assert p0 == b0 or _whole_words(p0), direction
        assert p["type"]["words"][1:] == before["type"]["words"][1:], direction            # the risky Jacquard face of word 1 stays (no wish = no face swap)


def test_apply_json_and_a_clean_piece():
    out, fixes = UL.apply_json(json.dumps(_piece()))
    ws = json.loads(out)["type"]["words"]
    assert fixes and all(_whole_words(w) and not UL.is_risky(w["font"]) for w in ws)
    again, none = UL.apply_json(out)             # idempotent: the second pass finds nothing to do
    assert none == [] and again == out


def test_the_pages_default_font_is_not_a_pin_so_boost_swaps_it():
    """R17b chain: the page's random default font (brief.story.shots[].font without font_user) makes no font_pins, so a risky default is swapped; the same font WITH font_user stays."""
    from tools.fusion import concept as CO
    lines = ["SACRED NOISE", "TOTAL LOSS"]

    def pins(user):
        story = {"title": "T", "logline": "L", "shots": [{"n": i + 1, "line": ln, "beat": "b", "font": f, **({"font_user": True} if user else {})} for i, (ln, f) in enumerate(zip(lines, (BITCOUNT, JACQ)))]}
        c = {"words": lines, "brief": CO.parse_brief({"on_screen": lines, "story": story})}
        return [x for x in CO.story_fonts(c) if x]
    assert pins(False) == [] and len(pins(True)) == 2
    p = _piece(pinned=pins(False) != [])                                 # no pins -> the piece has no font_pin / font_pins
    UL.apply(p)
    assert all(not UL.is_risky(w["font"]) for w in p["type"]["words"])
    q = _piece(pinned=pins(True) != [])
    UL.apply(q)
    assert [w["font"] for w in q["type"]["words"]] == [BITCOUNT, JACQ]
