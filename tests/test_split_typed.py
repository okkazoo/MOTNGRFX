"""2026-11-01 user_style.split_typed: the ONE splitter of the typed "Words or direction" text. Real run: the box held only "more legible text, palette: crimson, gold" and the READ / story
used it as the ON-SCREEN WORDS (brief.on_screen = ["MORE LEGIBLE TEXT,", "CRIMSON, GOLD"]). Directive clauses / lines are direction only, never words.
Covers: the splitter (pure direction, mixed, plain words, colour-ish words that are NOT directions), the node port web_app/bind.js splitTyped (same cases, same answers), the READ prompt /
parse (media_read), the story request (story.py), the concept brief (concept.parse_brief) and that brief.direction still carries the typed text.
Run: python -m pytest tests/test_split_typed.py -q"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import media_read as MR  # noqa: E402
from tools.fusion import story as ST  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

NODE = shutil.which("node")
TYPED = "more legible text, palette: crimson, gold"

PURE_DIRECTION = [
    TYPED, "palette: crimson, gold", "colours: teal and orange", "Palette of red and gold", "in red and gold", "a crimson and gold palette", "make it in red and gold",
    "more legible", "make the text bigger", "clearer text please", "keep it plain", "no echo", "keep it simple and minimal", "make the text gold", "use gold", "don't use echo",
    "keep it plain, no echo, bigger text",
]
WORDS = ["GOLDEN HOUR", "RED SKY", "BORN IN GOLD", "TEAL AND ORANGE", "ROSE", "MAKE IT COUNT", "KEEP GOING", "NO FEAR", "USE YOUR VOICE", "SET FIRE TO THE RAIN", "CLEAR SKY",
         "golden hour in the city", "the sun rose in gold", "HELLO, WORLD", "RISE UP\nSTAY\nHOLD", "PLAIN SIGHT", "IN GOLD WE TRUST", "NEVER GIVE UP", "RED\nBLUE\nGOLD"]
MIXED = [
    ("RISE UP\npalette: teal, orange", "RISE UP", "palette: teal, orange"),
    ("RISE UP, make the text gold", "RISE UP", "make the text gold"),
    ("GOLDEN HOUR\nbigger text\nSTAY", "GOLDEN HOUR\nSTAY", "bigger text"),
    ("RISE UP, in red and gold", "RISE UP", "in red and gold"),
    ('"MAKE THE TEXT BIG" palette: red', '"MAKE THE TEXT BIG"', "palette: red"),
    ("BORN IN GOLD\nkeep it plain\nRED SKY", "BORN IN GOLD\nRED SKY", "keep it plain"),
    ("palette: crimson, gold\nRISE UP", "RISE UP", "palette: crimson, gold"),
]


@pytest.mark.parametrize("text", PURE_DIRECTION)
def test_pure_direction_has_no_words(text):
    words, direction = US.split_typed(text)
    assert words == "" and direction.strip() == text.strip()
    assert US.parse_direction(direction) == US.parse_direction(text)                      # the direction part carries every wish of the whole text


@pytest.mark.parametrize("text", WORDS)
def test_plain_and_colourish_words_stay_words(text):
    assert US.split_typed(text) == (text, "")
    assert US.words_of(text) == text


@pytest.mark.parametrize("text,words,direction", MIXED)
def test_mixed_input_splits(text, words, direction):
    assert US.split_typed(text) == (words, direction)
    assert US.parse_direction(direction) == US.parse_direction(text)


def test_text_without_direction_is_returned_exactly_as_typed():
    t = "HELLO  world and\n\n\nsecond   line in"
    assert US.split_typed(t) == (t, "")
    long = ("la la la\n" * 400).strip()                                                       # pasted lyrics are long: no 2000-char cap here
    assert US.split_typed(long) == (long, "")
    assert US.split_typed("") == ("", "") and US.split_typed(None) == ("", "")


def test_split_is_idempotent_on_the_words():
    for text, words, _ in MIXED:
        assert US.split_typed(words) == (words, "")


# ------------------------------------------------------------------------------------------------------------ the page's port (web_app/bind.js splitTyped)
_JS = r"""
const B = require(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map((t) => { const r = B.splitTyped(t); return [r.words, r.direction]; })));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_bind_js_splitter_agrees_with_python():
    cases = PURE_DIRECTION + WORDS + [m[0] for m in MIXED] + [
        "", "   ", "a crimson, gold and teal palette", "RISE UP in red and gold", "Use palette: teal", "in red, gold and teal, bigger text", "IN GOLD", "WITHOUT WORDS", "never give up",
        "colour of magic", "STYLE: PLAIN", "text: HELLO", "#ff8800", "make it #ff8800", "HELLO\r\nWORLD\r\nbigger text", "“KEEP IT PLAIN” palette: red", "RISE UP and palette: teal",
        "go gold", "make it gold", "show me the words", "please make the text bigger", "I want the text in blue and white", "Make the background teal", "turquoise, lavender / rose & mint palette",
    ]
    r = subprocess.run([NODE, "-e", _JS, str(ROOT / "web_app" / "bind.js"), json.dumps(cases)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    bad = [(c, tuple(g), US.split_typed(c)) for c, g in zip(cases, got) if tuple(g) != US.split_typed(c)]
    assert not bad, bad


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_bind_js_colour_names_are_the_pythons():
    src = (ROOT / "web_app" / "bind.js").read_text(encoding="utf-8")
    i = src.index("const TYPED_NAMES = [")
    names = json.loads(src[src.index("[", i):src.index("]", i) + 1])
    assert sorted(names) == sorted(US.NAMES) and names == sorted(US.NAMES, key=len, reverse=True)


# ------------------------------------------------------------------------------------------------------------ the READ (media_read)
def _media(text):
    return json.dumps(dict(video="motngrfx_clip.mp4", images=[], audio=None, text=text, length=12, format="9:16"))


def _info():
    return dict(v=1, length=12, items=[dict(id="V1", kind="video", file="motngrfx_clip.mp4", duration=24.0, w=540, h=960, tiles=[1.0, 2.0], palette_px=["#112233", "#445566"])],
                music=None, music_detail=None, audio_from=None)


def test_read_prompt_keeps_the_direction_out_of_the_text_markers():
    med = MR.parse_media(_media(TYPED))
    assert med["text"] == "" and med["direction"] == TYPED
    p = MR.build_prompt(med, _info()["items"], None, None)
    text_block = p[p.index("<<<TEXT"):p.index(">>>TEXT")]
    assert "legible" not in text_block and "crimson" not in text_block and "(the user pasted no text)" in text_block
    assert "<<<DIRECTION" in p and TYPED in p[p.index("<<<DIRECTION"):p.index(">>>DIRECTION")]
    mixed = MR.build_prompt(MR.parse_media(_media("RISE UP\npalette: teal, orange")), _info()["items"], None, None)
    assert "RISE UP" in mixed[mixed.index("<<<TEXT"):mixed.index(">>>TEXT")] and "palette: teal" not in mixed[mixed.index("<<<TEXT"):mixed.index(">>>TEXT")]
    plain = MR.build_prompt(MR.parse_media(_media("GOLDEN HOUR")), _info()["items"], None, None)
    assert "<<<DIRECTION" not in plain and "GOLDEN HOUR" in plain


def test_read_of_pure_direction_has_no_on_screen_words():
    bad_reply = json.dumps(dict(on_screen=["MORE LEGIBLE TEXT,", "CRIMSON, GOLD", "more legible text"], concept="legible text", look="crimson and gold", media=[]))
    r = MR.parse(bad_reply, _info(), _media(TYPED))
    assert r["on_screen"] == [] and r["concept"] == "" and r["look"] == "crimson and gold"
    r = MR.parse("sorry, I cannot do that", _info(), _media(TYPED))                                 # no usable reply: the rule split of the words (none), the direction becomes the look
    assert r["on_screen"] == [] and r["concept"] == "" and r["look"] == TYPED
    r = MR.parse(json.dumps(dict(on_screen=["RISE UP", "teal, orange"], concept="a climb")), _info(), _media("RISE UP\npalette: teal, orange"))
    assert r["on_screen"] == ["RISE UP"]


# ------------------------------------------------------------------------------------------------------------ the STORY request
def _story_req(text, on_screen=()):
    media = [dict(id="I1", kind="image", file="a.png", shows="a crane", window=None, mood="calm", palette=["#112233", "#445566"])]
    return dict(read=dict(v=1, on_screen=list(on_screen), concept="", look="", media=media, music=None, lyrics=[], notes=""), length=8, seed=3, format="9:16", text=text)


def _story_reply(n, lines):
    return json.dumps(dict(title="The Crane Wakes Slowly Today Again", logline="A crane crosses a snowy forest.", look="slow, cold", music="ambient, piano",
                           shots=[dict(n=k, beat=f"Beat {k} of the crane story in the snow, moving on.", line=lines[k - 1]) for k in range(1, n + 1)], notes=""))


def test_story_request_of_pure_direction_takes_no_words_from_it():
    s = ST.parse(_story_reply(2, ["RISE", "FALL"]), _story_req(TYPED, on_screen=["MORE LEGIBLE TEXT,", "CRIMSON, GOLD"]))
    lines = [x["line"] for x in s["shots"]]
    assert "MORE LEGIBLE TEXT" not in " ".join(lines).upper() and "CRIMSON" not in " ".join(lines).upper()
    assert not s.get("text_select") and not (s.get("text_candidates") or {}).get("words")           # the box text offers no candidate words
    s = ST.parse(_story_reply(2, ["RISE", "FALL"]), _story_req("RISE UP\nFALL DOWN\npalette: teal, orange"))
    assert (s.get("text_candidates") or {}).get("words") == 4                                          # only RISE UP FALL DOWN are candidates


# ------------------------------------------------------------------------------------------------------------ the concept's brief
def test_brief_on_screen_drops_direction_clauses_and_keeps_the_direction():
    read = dict(on_screen=["MORE LEGIBLE TEXT,", "CRIMSON, GOLD"], concept="a city at dusk", look="warm", direction=TYPED)
    b = CO.parse_brief(json.dumps(read))
    assert b["on_screen"] == [] and b["direction"] == TYPED and b["concept"] == "a city at dusk"
    assert CO.brief_lines(b) == []
    b = CO.parse_brief(json.dumps(dict(read, on_screen=["RISE UP", "palette: teal, orange", "GOLDEN HOUR"])))
    assert b["on_screen"] == ["RISE UP", "GOLDEN HOUR"]
    b = CO.parse_brief(json.dumps(dict(concept="x", story={"shots": [{"n": 1, "line": "more legible text"}, {"n": 2, "line": "STAY"}]})))
    assert [ln for ln in b["on_screen"]] == ["STAY"]
