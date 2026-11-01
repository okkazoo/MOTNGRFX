"""2026-11-01, user decision after render R19 (SURFER laid out as SU / RF / ER): a MID-WORD split (grid / column rows that break inside a word) stays only as a RARE option in every
piece, image and video mode. tools/fusion/user_layout.py apply() (hooked in nodes_fusion/plan.py after US.stamp, nodes_fusion/design.py MographDesignRepair and
tools/app/own_video_v2_retime_box.py after the facefix / grid3 layouts): at most ONE word per piece keeps its split, and only in about 1 piece in 4 (seeded per piece); every other
split word is re-laid as whole words (line / stack, the bigger cap that passes the size floor). A legible_boost piece keeps none. A pinned font stays.
Run: python -m pytest tests/test_midword_rare.py -q"""
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import stable_type as ST  # noqa: E402
from tools.fusion import user_layout as UL  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

FONT = "PassionOne-Black.ttf"
TEXTS = ["SURFER", "SACRED NOISE", "MIDNIGHT RUNNER"]


def _word(text, font=FONT, **extra):
    b = ST.fit_box(text, font, None, "grid")
    assert b.get("split"), text
    w = {"text": text, "font": font, "axes": None, "lines": list(b["lines"]), "module": "beat_bars",
         "placement": {"dx": 0.0, "dy": 0.0, "hf": b["hf"], "pitch": 0.0, "roll": 0.0, "wf": b["wf"], "yaw": 0.0}, "t0": 0.0, "t1": 3.0}
    w.update(extra)
    return w


def _piece(seed, direction="", texts=TEXTS, **word_extra):
    p = {"id": f"P{seed}", "fps": 24, "format": "9:16", "quality": "draft", "plan": {"seed": seed},
         "type": {"duration": 9.0, "words": [_word(t, **word_extra) for t in texts]}, "request": {"format": "9:16", "length": 9}}
    if direction:
        US.stamp(p, direction)
    return p


def _split(p):
    return [w["text"] for w in p["type"]["words"] if UL._mid_split(w, w["text"].split())]


def test_at_most_one_split_word_per_piece_and_the_rest_are_whole_words():
    kept = 0
    for seed in range(240):
        p = _piece(seed)
        assert len(_split(p)) == 3
        UL.apply(p, [])
        assert len(_split(p)) <= 1, (seed, _split(p))
        kept += len(_split(p))
        for w in p["type"]["words"]:
            if w["text"] not in _split(p):
                assert " ".join(w["lines"]).split() == w["text"].split()
    assert kept > 0


def test_about_one_piece_in_four_keeps_one_split_word():
    n = 800
    keeps = 0
    for seed in range(n):
        p = _piece(seed, texts=["SURFER"])
        UL.apply(p, [])
        keeps += len(_split(p))
    assert 0.18 <= keeps / n <= 0.32, keeps / n               # about 25 %


def test_the_draw_is_seeded_per_piece_and_stable_across_the_hooks():
    for seed in range(40):
        a, b = _piece(seed), _piece(seed)
        UL.apply(a, [])
        UL.apply(b, [])
        assert a == b, seed                                   # same seed, same result
        again = copy.deepcopy(a)
        assert UL.apply(again, []) == 0 and again == a, seed  # the second hook (design repair, retime) changes nothing: the kept word stays kept
    out, fixes = UL.apply_json(json.dumps(_piece(1)))        # the design nodes' string form
    assert fixes and len(_split(json.loads(out))) <= 1
    # the own-video route passes its own seed: it, not the piece's plan.seed, decides
    kept = set()
    for seed in range(60):
        p = _piece(0, texts=["SURFER"])
        UL.apply(p, [], seed=seed)
        if _split(p):
            kept.add(seed)
    assert 5 <= len(kept) <= 30


def test_a_boost_piece_has_no_split_at_all():
    for seed in range(120):
        p = _piece(seed, "more legible text, palette: crimson, gold")
        UL.apply(p, [])
        assert _split(p) == [], seed


def test_whole_word_relay_passes_the_size_floor():
    seen = 0
    for seed in range(60):
        fixes = []
        p = _piece(seed)
        UL.apply(p, fixes)
        for w in p["type"]["words"]:
            if UL._mid_split(w, w["text"].split()):
                continue
            seen += 1
            pl = w["placement"]
            boxes = [ST.fit_box(w["text"], w["font"], w.get("axes"), c) for c in ("line", "stack") if c in ST.allowed(w["text"])]
            b = next(b for b in boxes if b["lines"] == w["lines"])
            assert b["ok"] and b["cap"] >= DD.CAP_FLOOR, (w["text"], b["cap"])
            assert abs(b["hf"] - pl["hf"]) < 1e-6 and abs(b["wf"] - pl["wf"]) < 1e-6 and pl["roll"] == 0.0
            best = max((x for x in boxes if x["ok"]), key=lambda x: x["cap"])
            assert b["cap"] >= best["cap"] - 1e-9, "the bigger cap of line / stack"
    assert seen > 0


def test_the_change_is_logged_and_a_pinned_font_stays():
    p = _piece(3, font_pin=FONT)
    p["type"]["font_pins"] = [{"font": FONT, "text": t} for t in TEXTS]
    fixes = []
    UL.apply(p, fixes)
    assert fixes and p["user_style_log"] == fixes and all("layout" in x for x in fixes)
    assert all(w["font"] == FONT and w["font_pin"] == FONT for w in p["type"]["words"])
    assert len(_split(p)) <= 1


def test_a_piece_without_a_split_is_untouched():
    p = _piece(5)
    for w in p["type"]["words"]:
        w["lines"] = w["text"].split() if len(w["text"].split()) > 1 else []
        w.pop("lines") if not w["lines"] else None
    before = copy.deepcopy(p)
    assert UL.apply(p, []) == 0 and p == before
