"""2026-11-01 the user's typed "Words or direction" text reaches the piece as piece.direction + piece.user_style (tools/fusion/user_style.py), in BOTH routes, and only the
user's own typed text counts: lyrics, auto-written words, the model's concept / look never set a palette or a legibility wish.
IMAGES: page brief.direction -> concept.parse_brief -> apply_brief (c.direction) -> request_extras (owners.direction) -> MographRequest -> MographPiecePlan (user_style.stamp);
        piece.request.direction rides the design step (LOCKED_KEYS has `request`), parse_direction(piece) reads it there.
VIDEO:  page brief.direction -> own_video_job.read_brief -> stamp_direction_json(piece_designed.json) -> retime box / render read the piece.
Run: python -m pytest tests/test_user_direction_route.py -q"""
from __future__ import annotations

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

from tools.app import own_video_job as J  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import still_preview as SP  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

TYPED = "more legible text, palette: crimson, gold"
CRIM, GOLD = list(US.NAMES["crimson"]), list(US.NAMES["gold"])
NOTHING = {"palette": None, "legible_boost": False, "plain": False}
READ = {"on_screen": ["GOLDEN HOUR", "STAY"], "concept": "a city at dusk", "look": "warm, slow", "media": [{"id": "V1", "kind": "video", "shows": "a street", "window": [0, 8]}]}


def _style(piece):
    return US.parse_direction(piece)


# ------------------------------------------------------------------------------------------------------------ the parser is conservative
@pytest.mark.parametrize("text", ["GOLDEN HOUR", "golden hour in the city", "RED SKY\nBLUE MOON", "BORN IN GOLD", "ROSE", "the sun rose in gold", "PLAIN SIGHT", "# 123 and #bad",
                                  "midnight in green park"])
def test_ordinary_on_screen_words_set_nothing(text):
    assert US.parse_direction(text) == NOTHING


@pytest.mark.parametrize("text,palette", [
    (TYPED, [CRIM, GOLD]), ("palette: crimson, gold", [CRIM, GOLD]), ("make it in red and gold", [list(US.NAMES["red"]), GOLD]), ("a crimson and gold palette", [CRIM, GOLD]),
    ("use #ff8800 and #224466", [[255, 136, 0], [34, 68, 102]]), ("GOLDEN HOUR\npalette: #f80, teal", [[255, 136, 0], list(US.NAMES["teal"])]),
])
def test_explicit_palette_phrases(text, palette):
    assert US.parse_direction(text)["palette"] == palette


def test_the_typed_wishes():
    st = US.parse_direction(TYPED)
    assert st["legible_boost"] and st["plain"] and st["palette"] == [CRIM, GOLD]
    assert US.parse_direction("keep it plain please")["plain"] and not US.parse_direction("keep it plain please")["legible_boost"]
    assert US.parse_direction("bigger text")["legible_boost"] and US.parse_direction("no echo")["plain"]


def test_a_pieces_own_words_and_the_models_text_are_never_read():
    assert _style({"request": {"words": "GOLDEN HOUR\npalette: red\nSTYLE: PLAIN"}}) == NOTHING                      # the request / piece words are the piece's own (auto-written) lines
    assert _style({"brief": {"concept": "palette: red, legible, plain", "look": "make it simple", "direction": {"palette": "x"}}}) == NOTHING     # the model's brief (concept / look / direction dict)
    assert _style({"type": {"words": [{"text": "PALETTE: RED"}]}, "request": {"words": "x"}}) == NOTHING
    assert US.parse_direction({"words": "RISE\npalette: crimson, gold"})["palette"] == [CRIM, GOLD]                      # a RAW words box dict: its directive lines count


# ------------------------------------------------------------------------------------------------------------ IMAGES route
def _images_chain(read: dict):
    """page brief -> concept -> request extras -> the MographRequest raw record -> the planned piece (what the nodes do, minus ComfyUI)."""
    b = CO.parse_brief(json.dumps(read))
    c = CO.apply_brief({"words": ["X"], "idea": "i"}, b)
    extras = CO.request_extras(c)
    direction = US.direction_of(extras)
    raw = SP.request_record({"words": "\n".join(c["words"]), "owners": extras, "direction": direction or None, "seed": 3})
    piece = SP.prepare_piece({"id": "P", "type": {"words": [{"text": w} for w in c["words"]]}}, raw, False)
    return b, c, extras, US.stamp(piece, direction)


def test_images_typed_text_reaches_the_piece():
    b, c, extras, piece = _images_chain(dict(READ, direction=TYPED))
    assert b["direction"] == TYPED and c["direction"] == TYPED and json.loads(extras)["direction"] == TYPED and "owners" in json.loads(extras)
    assert piece["direction"] == TYPED
    assert piece["user_style"] == {"palette": [CRIM, GOLD], "legible_boost": True, "plain": True} == _style(piece)
    assert piece["request"]["direction"] == TYPED
    designed = {"request": piece["request"], "type": piece["type"]}                                  # what survives the design step (design_step.LOCKED_KEYS keeps `request`, not user_style)
    assert _style(designed) == piece["user_style"], "the recipe still sees the user's wishes after the design step"


def test_images_words_and_lyrics_alone_never_set_a_style():
    typed_words_only = _images_chain(dict(READ, direction="GOLDEN HOUR"))
    assert typed_words_only[3]["user_style"] == NOTHING and _style({"request": typed_words_only[3]["request"]}) == NOTHING
    none_typed = _images_chain(dict(READ, lyrics=[{"t": 0, "text": "palette: red, make it plain, more legible"}], notes="palette: red"))
    b, c, extras, piece = none_typed
    assert "direction" not in b and "direction" not in c and US.direction_of(extras) == ""
    assert "user_style" not in piece and "direction" not in piece and "direction" not in piece["request"]
    assert _style(piece) == NOTHING                                                                  # on_screen "GOLDEN HOUR" / "STAY" + concept + look: nothing
    assert "direction" not in json.loads(extras), "no typed text = request_extras carries no direction key"


def test_images_model_reply_cannot_inject_a_direction():
    # the page builds the brief from normalizeRead (drops unknown fields) + withDirection(typed); a model-written `direction` in the READ never reaches the browser's brief,
    # and a brief without typed text carries none; the server side only reads `direction` as a string
    assert "direction" not in CO.parse_brief(json.dumps(dict(READ, direction={"pace": "slow"})))
    assert "direction" not in CO.parse_brief(json.dumps(dict(READ, direction="   ")))
    assert CO.parse_brief(json.dumps({"direction": "palette: red"})) == {}, "a direction alone is not a usable brief"


@pytest.fixture(scope="module")
def nodes():
    for cand in (os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI"), str(Path.home() / "Documents" / "Dev" / "ComfyUI")):
        if cand and os.path.isfile(os.path.join(cand, "comfy_api", "latest", "__init__.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import comfy_api.latest  # noqa: F401
        import folder_paths  # noqa: F401
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"comfy_api not importable: {e}")
    import nodes_fusion.plan as plan
    return plan


def test_images_nodes_stamp_the_planned_piece(nodes):
    from test_fusion_plan import synth_music
    _, c, extras, _ = _images_chain(dict(READ, direction=TYPED))
    req = nodes.MographRequest.execute("HELLO\nWORLD", "", "", 12, "auto", 3, "draft", format="9:16", owners=extras).args[0]
    assert req["direction"] == TYPED and req["raw"]["direction"] == TYPED
    piece = nodes.MographPiecePlan.execute(req, synth_music(100.0, 12.0)).args[0]
    assert piece["direction"] == TYPED and piece["user_style"]["palette"] == [CRIM, GOLD] and piece["user_style"]["legible_boost"] and piece["request"]["direction"] == TYPED
    plain = nodes.MographRequest.execute("GOLDEN HOUR\nSTAY", "", "", 12, "auto", 3, "draft", format="9:16").args[0]
    p2 = nodes.MographPiecePlan.execute(plain, synth_music(100.0, 12.0)).args[0]
    assert "direction" not in plain and "user_style" not in p2 and "direction" not in p2 and _style(p2) == NOTHING


# ------------------------------------------------------------------------------------------------------------ VIDEO route
DESIGNED = json.dumps({"id": "P", "type": {"words": [{"text": "GOLDEN HOUR"}]}, "request": {"words": "GOLDEN HOUR"}})


def test_video_typed_text_reaches_the_designed_piece():
    b = J.read_brief(dict(READ, direction=TYPED))
    assert b["direction"] == TYPED
    piece = json.loads(J.stamp_direction_json(DESIGNED, b["direction"]))
    assert piece["direction"] == TYPED
    assert piece["user_style"] == {"palette": [CRIM, GOLD], "legible_boost": True, "plain": True} == _style(piece)      # what own_video_v2_retime_box / recipe_eval read
    assert piece["type"] == json.loads(DESIGNED)["type"], "nothing else on the piece is touched"


def test_video_lyrics_words_and_the_storyboard_never_set_a_style():
    lyr = dict(READ, lyrics=[{"t": 0, "text": "palette: red, make it plain"}], story={"hits": [{"text": "GOLDEN HOUR", "land": 1.0}, {"text": "legible text", "land": 3.0}]}, notes="plain")
    b = J.read_brief(lyr)
    assert b["direction"] == "" and J.stamp_direction_json(DESIGNED, b["direction"]) == DESIGNED, "no typed text = the piece JSON is returned byte-identical"
    only_words = J.read_brief(dict(lyr, direction="GOLDEN HOUR"))
    piece = json.loads(J.stamp_direction_json(DESIGNED, only_words["direction"]))
    assert piece["direction"] == "GOLDEN HOUR" and piece["user_style"] == NOTHING and _style(piece) == NOTHING
    assert J.read_brief(dict(READ, direction={"palette": "red"}))["direction"] == "", "only a typed string counts"
    assert J.stamp_direction_json("not json", "palette: red") == "not json"


# ------------------------------------------------------------------------------------------------------------ the page
def test_the_page_sends_the_typed_text_in_the_brief():
    app = (ROOT / "web_app" / "app.js").read_text(encoding="utf-8")
    bind = (ROOT / "web_app" / "bind.js").read_text(encoding="utf-8")
    assert "B.withDirection(" in app and "const base = readBrief(), typed = S.media.text" in app
    assert "function withDirection(briefJson, text)" in bind and "o.direction = t.slice(0, 2000)" in bind and "briefFromStory, withDirection," in bind
    # the READ the model writes is normalised first: `direction` is not one of its fields
    norm = bind[bind.index("function normalizeRead"):bind.index("function briefString")]
    assert "direction" not in norm
