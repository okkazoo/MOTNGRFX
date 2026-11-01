"""Hybrid free planner: creative brief (mograph.brief) -> planner_rules.plan_from_brief."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mograph import brief as B  # noqa: E402
from mograph.analyze import analyze_inputs  # noqa: E402
from mograph.plan import FORMATS, validate_plan  # noqa: E402
from mograph.planner_rules import _wcag, plan_from_brief  # noqa: E402

SEA = "Somewhere between here and the sea."


def ana(text: str, dur: float = 10.0) -> dict:
    return analyze_inputs(text, [], None, dur)


def text_layers(plan):
    return [(s, L) for s in plan["scenes"] for L in s["layers"] if L["type"] == "text"]


def words_of(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.casefold())


# ---------------------------------------------------------------- prompt

@pytest.mark.parametrize("text,n", [("Midnight ramen. Open till 3am.", 0), (SEA + "\nStyle: filmic, fast and hectic.", 0),
                                    ("NEON HEART — the new album", 3), ("", 0)])
def test_prompt_short_and_without_example_values(text, n):
    p = B.brief_prompt(text, n, seed=5)
    assert len(p.split()) <= 250
    assert not re.search(r"#[0-9a-fA-F]{6}", p)            # no colours to copy
    assert "{" not in p and "e.g." not in p.lower() and "example" not in p.lower()
    if text:
        for f in B.FAMILIES:
            assert f in p
        assert ("hero_image" in p) == (n > 0)


def test_prompt_enumeration_order_varies_with_seed():
    def fam_order(seed):
        line = next(ln for ln in B.brief_prompt("Hello world", 0, seed).splitlines() if ln.startswith("family:"))
        return [f for f in re.findall(r"(\w+) \(", line)]
    orders = {tuple(fam_order(s)) for s in range(6)}
    assert len(orders) >= 5
    assert sorted(fam_order(0)) == sorted(B.FAMILIES)


# ---------------------------------------------------------------- parsing / normalisation

def test_parse_keeps_verbatim_lines_and_fixes_fields():
    raw = ('```json\n{"family": "Film Hectic", "palette": ["#ffffff", "#ff3b1f", "#0b0b0b"], "energy": 0.2, '
           '"bpm": 300, "arc": "bursts", "lines": ["somewhere between here", "and the sea"], '
           '"mood": ["fast", "x"], "text_region": "Bottom"}\n```')
    log = []
    b = B.parse_brief(raw, SEA + "\nStyle: filmic, fast and hectic.", 0, 1, log)
    assert b["family"] == "film_hectic"
    assert b["lines"] == ["Somewhere between here", "and the sea."]       # the user's own spelling
    assert b["palette"][0] == "#0b0b0b" and b["palette"][1] == "#ffffff" and b["palette"][2] == "#ff3b1f"
    lo, hi = B.FAMILY_SPECS["film_hectic"]["energy"]
    assert lo <= b["energy"] <= hi and b["bpm"] <= B.FAMILY_SPECS["film_hectic"]["bpm"][1]
    assert b["arc"] == "hold_drop_bursts" and b["text_region"] == "bottom" and b["mood"] == ["hype"]
    assert log


def test_invented_words_fall_back_to_auto_split():
    log = []
    b = B.parse_brief('{"family": "type_soft", "lines": ["BREAK THE LIMIT"]}', "FASTER THAN YESTERDAY", 0, 0, log)
    assert b["lines"] == ["FASTER THAN YESTERDAY"]
    assert any("auto split" in s for s in log)
    b = B.parse_brief('{"lines": ["Midnight ramen."]}', "Midnight ramen. Open till 3am. Late noodles, hot broth.",
                      0, 0, [])
    assert b["lines"] == B.auto_lines("Midnight ramen. Open till 3am. Late noodles, hot broth.")   # dropped words


def test_auto_lines_are_the_users_words():
    assert B.auto_lines("Midnight ramen. Open till 3am.") == ["Midnight ramen.", "Open till 3am."]
    assert B.auto_lines("NEON HEART — the new album") == ["NEON HEART", "the new album"]
    assert B.auto_lines(SEA + "\nStyle: filmic, slow and soft.") == ["Somewhere between here", "and the sea."]
    for t in ("Grandma's garden, summer 1979", "We go where light doesn't.", "a b c d e f g h i j k l m"):
        assert words_of(" ".join(B.auto_lines(t))) == words_of(t)


def test_garbage_gives_rule_defaults_and_normalise_is_idempotent():
    log = []
    b = B.parse_brief("I can't do that", "Breathe in. Breathe out.", 0, 3, log)
    assert b["family"] in B.FAMILIES and b["lines"] == ["Breathe in.", "Breathe out."]
    assert B.normalize_brief(b, "Breathe in. Breathe out.", 0, 3) == b
    b2 = B.parse_brief('{"family": "neon_loop", "palette": ["#0a0a0a", "#1a1a1a", "#2a2a2a", "#ffffff"], '
                       '"mood": 3}', "x y", 0, 3, [])
    assert all(max(B._rgb(c)) - min(B._rgb(c)) >= 0.2 for c in b2["palette"][2:])   # grey accents replaced


def test_slot_machine_variety():
    fams = [B.slot_machine(s)["family"] for s in range(10)]
    assert len(set(fams)) >= 6
    for start in (7001, 123456):
        assert len({B.slot_machine(s)["family"] for s in range(start, start + 10)}) >= 6
    assert len({tuple(B.slot_machine(s)["palette"]) for s in range(10)}) == 10


# ---------------------------------------------------------------- plan_from_brief

def test_slot_machine_plans_vary_and_validate():
    a = ana("")
    fams, pals, heads = set(), set(), set()
    for seed in range(10):
        log = []
        p = plan_from_brief(a, None, seed=seed, log=log)
        assert validate_plan(p, a) == []
        fams.add(log[-1].split(" / ")[0].split(": ")[1])
        pals.add(tuple(p["style"]["palette"]))
        heads.add(json.dumps(text_layers(p)[0][1]["text"]))
    assert len(fams) >= 6 and len(pals) >= 8 and len(heads) >= 6


def test_hectic_cuts_fast_and_soft_holds():
    a = ana(SEA)
    hec = plan_from_brief(a, {"family": "type_hectic", "arc": "hold_drop_bursts"}, seed=2)
    lens = [s["t1"] - s["t0"] for s in hec["scenes"]]
    assert sum(1 for x in lens if 0.18 <= x <= 0.5) >= 4                 # a burst of flash-length shots
    assert hec["style"]["energy"] >= 0.85 and hec["audio"]["bpm"] >= 128
    strobes = [L for s in hec["scenes"] for L in s["layers"] if L.get("primitive") == "rect" and L["t1"] - L["t0"] < 0.1]
    assert strobes                                                       # exposure strobe frames
    soft = plan_from_brief(a, {"family": "film_soft", "arc": "chapters"}, seed=2)
    lens = [s["t1"] - s["t0"] for s in soft["scenes"]]
    assert min(lens) >= 1.4 and all(x >= 2.0 for x in lens[:-1])
    assert {s["transition_out"]["type"] for s in soft["scenes"][:-2]} <= {"crossfade", "luma_wipe"}   # then a hard cut onto the end card
    assert [t["type"] for t in soft["style"]["treatment"]][:2] == ["film_grade", "light_leak"]
    assert soft["style"]["energy"] <= 0.35


@pytest.mark.parametrize("family", sorted(B.FAMILIES))
@pytest.mark.parametrize("arc", sorted(B.ARCS))
def test_every_family_and_arc_is_valid_verbatim_and_readable(family, arc):
    text = "Midnight ramen. Open till 3am."
    a = ana(text)
    p = plan_from_brief(a, {"family": family, "arc": arc}, seed=11)
    assert validate_plan(p, a) == []
    user = set(words_of(text))
    pal = p["style"]["palette"]
    for sc, L in text_layers(p):
        for t in (L["text"].values() if isinstance(L["text"], dict) else [L["text"]]):
            assert set(words_of(t)) <= user                             # never new copy
        if L["role"] == "headline":
            x, y, w, h = L["box"]["9:16"]
            assert h * FORMATS["9:16"][1] >= 0.5 * w * FORMATS["9:16"][0] - 1e-6   # tall: auto-fit stacks
        bg = sc["background"]
        if bg["type"] == "color" and L.get("highlight") is None and not L.get("window"):
            ref = lambda c: B._rgb(pal[int(c.split(":")[1])]) if c.startswith("palette:") else B._rgb(c)  # noqa: E731
            assert _wcag(ref(L["color"]), ref(bg["color"])) >= 3.0


def test_brief_json_string_images_and_determinism():
    from PIL import Image
    fx = os.path.join(ROOT, "tests", "fixtures")
    imgs = [np.asarray(Image.open(os.path.join(fx, f)).convert("RGB"), np.float32) / 255.0
            for f in ("blob_gradient.png", "stripes.png")]
    a = analyze_inputs("Midnight ramen. Open till 3am.", imgs, None, 10.0)
    raw = json.dumps({"family": "film_type_hectic", "hero_image": 1, "lines": ["Midnight ramen.", "Open till 3am."]})
    p1 = plan_from_brief(a, raw, seed=4)
    p2 = plan_from_brief(a, raw, seed=4)
    assert p1 == p2 and validate_plan(p1, a) == []
    assert p1["scenes"][0]["background"]["src"] == "img1"                 # hero first
    assert any(L.get("window") for _, L in text_layers(p1))               # type as a window onto the photo
