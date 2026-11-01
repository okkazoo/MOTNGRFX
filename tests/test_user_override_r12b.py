"""2026-11-01 regression, real GPU render R12b (images mode, 9:16, 8 s): the typed direction "more legible text, palette: crimson, gold" reached the piece and the colour stage, but the
words were still ROLLED 90 deg at a cap of ~10 % (FINAL ECHO) and the outline was the COMPLEMENT of the palette (teal / navy), so the user saw "a pink outline, a blue / white pixel
font, no crimson / gold, no bigger type". A piece shaped like that archive's piece.json, minimised: top-level `direction` + `user_style`, `request.direction`, two words (a grid
SACRED NOISE in PassionOne, a rolled FINAL ECHO in Jacquard12), the model's compose chain (flow_wake -> tint_type -> sandwich).
Pins: (1) parse_direction on the piece (top level and the designed piece that keeps only `request`); (2) tools/fusion/user_layout.py re-fits the rolled word upright and bigger and leaves the
grid word and every piece WITHOUT a legibility wish alone; (3) legible_color.support_color never gives a user palette a complement outline; (4) the compose recipe carries the palette,
the x1.2 target + forced outline, the 12 % floor, no echo and no glint.
Run: python -m pytest tests/test_user_override_r12b.py -q"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import legible_color as LC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import user_layout as UL  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

TYPED = "more legible text, palette: crimson, gold"
CRIM, GOLD = list(US.NAMES["crimson"]), list(US.NAMES["gold"])
H, W = 160, 96


def _piece(direction=TYPED, stamped=True):
    ops = [{"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$flow_m", "trail": 10, "state_key": "A:type_matte", "u": "$u"}},
           {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
           {"op": "tint_type", "out": "tcol", "args": {"footage": "$footage", "type_a": "$wm"}},
           {"op": "sandwich", "out": "comp", "args": {"footage": "$footage", "matte": "$m", "type_a": "$wm", "type_rgb": "$tcol", "matte_threshold": 0.5}},
           {"op": "vignette", "out": "image", "args": {"frame": "$comp", "strength": 0.3}}]
    words = [{"text": "SACRED NOISE", "font": "PassionOne-Black.ttf", "font_pin": "PassionOne-Black.ttf", "lines": ["SAC", "RED", "NOI", "SE"], "ar_min": 0.85, "module": "pipework",
              "placement": {"dx": 0.0, "dy": 0.0, "hf": 0.878, "pitch": 0.0, "roll": 0.0, "wf": 0.9, "yaw": 0.0}, "t0": 0.0, "t1": 3.9},
             {"text": "FINAL ECHO", "font": "Jacquard12-Regular.ttf", "font_pin": "Jacquard12-Regular.ttf", "lines": ["FINAL ECHO"], "module": "beat_bars",
              "placement": {"dx": 0.0, "dy": 0.0, "hf": 0.253, "pitch": 0.0, "roll": 90.0, "wf": 0.88, "yaw": 0.0}, "t0": 4.5, "t1": 8.0}]
    p = {"id": "A859730280", "fps": 24, "format": "9:16", "quality": "draft", "sizes": {"compose": [W, H]}, "slug": "sacred-noise-final-echo",
         "shots": [{"id": "s1", "frames": 192, "matte": True}], "technique": "behind", "type_passes": ["matte", "depth", "line"],
         "type": {"bpm": 106.081, "beats": [0.006, 0.575, 1.143, 1.695, 2.275, 2.843, 3.409, 3.989, 4.545, 5.097, 5.67, 6.234, 6.801, 7.369, 7.934], "duration": 8.0, "words": words},
         "compose": {"crossfade": 0.0, "default": {"ops": ops}}, "request": {"format": "9:16", "length": 8, "direction": direction}}
    if stamped:
        US.stamp(p, direction)                                  # what MographPiecePlan does (top-level direction + user_style)
    return p


def _designed(p):
    """what survives the design step: LOCKED_KEYS keeps `request`, not the top-level direction / user_style."""
    q = copy.deepcopy(p)
    q.pop("direction", None)
    q.pop("user_style", None)
    return q


def _hue(rgb):
    lab = LC.lin_to_oklab(LC.srgb_to_lin(np.asarray(rgb, np.float32)))
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


# ------------------------------------------------------------------------------------------------------------ 1 the typed direction is on the piece
def test_the_archive_shaped_piece_parses():
    for p in (_piece(), _designed(_piece())):
        st = US.parse_direction(p)
        assert st["palette"] == [CRIM, GOLD] and st["legible_boost"] and st["plain"]
    assert _piece()["user_style"] == {"palette": [CRIM, GOLD], "legible_boost": True, "plain": True}


# ------------------------------------------------------------------------------------------------------------ 2 the layout half
def test_legible_boost_refits_the_rolled_word_upright_and_bigger():
    p, fixes = _piece(), []
    n = UL.apply(p, fixes)
    # 2026-11-01 (R17b): a mid-word split grid is now re-laid as whole words and a risky face (Jacquard 12, bitmap blackletter) is swapped; tests/test_legible_boost_layout.py pins that part
    assert n == 2 and any("type.words[1] layout" in x for x in fixes) and any("type.words[0] layout" in x for x in fixes)
    sacred, final = p["type"]["words"]
    assert final["placement"]["roll"] == 0.0 and final["lines"] == ["FINAL", "ECHO"], "a rolled word reads upright"
    assert abs(final["placement"]["wf"] - 0.9) < 0.01 and final["placement"]["hf"] > 0.2
    from tools.fusion import stable_type as ST
    assert ST.fit_box("FINAL ECHO", final["font"], None, "stack")["cap"] > ST.fit_box("FINAL ECHO", final["font"], None, "rot90")["cap"], "the new layout has the bigger cap"
    assert sacred["lines"] == ["SACRED", "NOISE"] and sacred["placement"]["roll"] == 0.0 and sacred["ar_min"] == 0.85, "a mid-word split grid is re-laid as whole words (ar_min kept)"


def test_the_designed_piece_that_keeps_only_request_is_refitted_too():
    p = _designed(_piece())
    assert UL.apply(p) == 2 and p["type"]["words"][1]["placement"]["roll"] == 0.0


def test_no_legibility_wish_changes_nothing():
    for direction in ("", "palette: crimson, gold", "keep it plain", "GOLDEN HOUR"):
        p = _piece(direction)
        before = copy.deepcopy(p["type"])
        UL.apply(p)
        # 2026-11-01 (render R19): without the wish only the RARE mid-word-split pass runs (tests/test_midword_rare.py): the split grid word may be re-laid as whole words; the rolled
        # whole word (FINAL ECHO) and every face stay as they were
        assert p["type"]["words"][1] == before["words"][1] and [w["font"] for w in p["type"]["words"]][1] == before["words"][1]["font"], direction
        w0, b0 = p["type"]["words"][0], before["words"][0]
        assert w0 == b0 or " ".join(w0["lines"]).split() == w0["text"].split(), direction
        assert {k: v for k, v in p["type"].items() if k != "words"} == {k: v for k, v in before.items() if k != "words"}, direction


def test_apply_json_roundtrip():
    p = _piece()
    out, fixes = UL.apply_json(json.dumps(p))
    assert fixes and json.loads(out)["type"]["words"][1]["placement"]["roll"] == 0.0
    same, none = UL.apply_json(json.dumps(_piece("palette: crimson, gold")))
    assert json.loads(same)["type"]["words"][1]["placement"]["roll"] == 90.0 and all("mid-word" in x for x in none)      # no wish: only the rare-split pass may report
    assert UL.apply_json("not json") == ("not json", [])


# ------------------------------------------------------------------------------------------------------------ 3 the outline stays in the palette
def test_a_user_palette_never_gets_a_complement_outline():
    hues = LC.hues_of_colors([[v / 255.0 for v in CRIM], [v / 255.0 for v in GOLD]])
    pal_h = [h["h"] for h in hues]
    for ink_rgb, light in (([1.0, 0.55, 0.55], False), ([1.0, 0.89, 0.66], False), ([0.45, 0.05, 0.1], True)):
        ink = LC.srgb_to_lin(np.array(ink_rgb, np.float32))
        sup = LC.support_color(ink, hues, light=light)
        h = _hue(LC.lin_to_srgb(sup))
        assert min(LC.hue_dist(h, ph) for ph in pal_h) < 30, f"outline hue {h:.0f} must be a palette hue {pal_h}, not the complement"
        assert LC.contrast_ratio(float(LC.rel_lum(ink)), float(LC.rel_lum(sup))) >= 3.0
    shot = [dict(h=_hue([0.05, 0.55, 0.55]), c=0.1)]                                  # no user flag (a shot's own hues): the complement rule of today's colour review is unchanged
    ink = LC.srgb_to_lin(np.array([0.95, 0.75, 0.2], np.float32))
    assert LC.hue_dist(_hue(LC.lin_to_srgb(LC.support_color(ink, shot, light=False))), _hue(LC.lin_to_srgb(ink))) >= 90


# ------------------------------------------------------------------------------------------------------------ 4 the compose recipe carries every wish
def test_recipe_carries_palette_boost_floor_and_no_echo(tmp_path):
    (tmp_path / "beauty").mkdir()
    for p in (_piece(), _designed(_piece())):
        rec = RE.build_recipe(p, "s1", type_dir=str(tmp_path))
        lc = next(o for o in rec["ops"] if o["op"] == "legible_color")["args"]
        assert lc["user_palette"] == [[v / 255.0 for v in CRIM], [v / 255.0 for v in GOLD]]
        assert lc["boost"] is True and lc["echo_alpha"] == 0.0 and "life" not in lc, "x1.2 target + forced outline, no echo, no hold glint"
        assert all(o["args"].get("gate") == 0.0 for o in rec["ops"] if o["op"] == "flow_wake"), "the wake is gated off"
    plain = _piece("palette: crimson, gold")
    lc2 = next(o for o in RE.build_recipe(plain, "s1", type_dir=str(tmp_path))["ops"] if o["op"] == "legible_color")["args"]
    assert lc2["user_palette"] and "boost" not in lc2 and "echo_alpha" not in lc2, "a palette alone keeps the default echo / glint"
