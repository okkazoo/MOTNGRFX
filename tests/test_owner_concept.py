"""Per-word owner, round 2 (2026-10-28): the model chooses the owners in the CONCEPT step, at most one LTX word per shot, a clean prompt clause, and the compose
fallback when LTX formed no letters (tools/fusion/ltx_check.py). Fake replies, tiny synthetic frames and the REAL batch-3060 v005 r01 clips (skipped when absent): no
model, no network, no GPU.
Run: python -m pytest tests/test_owner_concept.py -q"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from test_fusion_core import synth_music  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import ltx_check as LC  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

REAL = ROOT / "21_batch-3060" / "v005" / "out" / "motion-graphic_batch-3060_v005_r01_out_shots"
OBJ = dict(title="x", subject="old fisherman", scene="an old fisherman on a quiet harbour at dawn with a small boat and a lantern in the mist",
           music="folk, guitar, soft drums, warm, airy", bpm=90, mood="tender", look="strand", shots="cut")


# ------------------------------------------------------------------------------------------------ 1. the concept chooses the owners
def test_concept_prompt_asks_for_owners_and_says_ltx_letters_are_fragile():
    p = CO.concept_prompt(3, 16, [])
    assert '"owners"' in p and "flat | ltx | handoff" in p and "expensive and fragile" in p
    assert "matched to the scene" in p and "vary it between words" in p
    cut = CO.concept_prompt(3, 16, [{"shots": "single"}, {"shots": "chain"}, {"shots": "chain"}])        # -> a cut piece (the least used mode)
    assert "the DEFAULT is flat" in cut and "never all LTX" in cut
    single = CO.concept_prompt(3, 16, [], mode="single")
    assert "AT MOST 2 key lines" in single and "the DEFAULT is flat" in single                                                               # a 16 s single shot: the long-shot cap
    guide = (ROOT / "16_app" / "DESIGN_GUIDE_open_model.md").read_text(encoding="utf-8")
    assert "ONE key word per shot" in guide and "concept" in guide.lower()


def test_concept_owners_are_mended_field_by_field():
    owners = [{"render": "hand-off", "ltx_at": "30%", "material": "wet harbour stone letters"}, {"render": "flat", "material": "ignored"}, "ltx", {"render": "banana"}]
    c, errors, fixes = CO.normalize(dict(OBJ, words=["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME"], owners=owners), 3, [], 16)
    assert not errors and c["words"] == ["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME"]
    assert c["owners"] == [{"render": "handoff", "ltx_at": 0.3, "material": "wet harbour stone"}, {"render": "flat"}, {"render": "ltx"}, {}]
    assert any("'hand-off' -> 'handoff'" in f for f in fixes) and any("'30%' -> 0.3" in f for f in fixes) and any("'banana'" in f for f in fixes)
    c, _, fixes = CO.normalize(dict(OBJ, words=["A", "B", "C", "D"], owners=[{"render": "handoff", "ltx_at": 5, "material": "x " * 12}]), 3, [], 16)
    assert c["owners"][0]["ltx_at"] == 0.2 and len(c["owners"][0]["material"].split()) == 8 and c["owners"][1:] == [{}, {}, {}]     # 5 is read as 5 %: clamped to 0.2
    c, _, fixes = CO.normalize(dict(OBJ, words=["A", "B", "C", "D"]), 3, [], 16)                          # none in the reply: the rules decide, logged
    assert "owners" not in c and any(f.startswith("owners: the reply has none") for f in fixes)
    c, _, fixes = CO.normalize(dict(OBJ, words=["A", "B", "C", "D"], owners="all handoff"), 3, [], 16)
    assert "owners" not in c and any("not a list" in f for f in fixes)


def test_concept_owners_follow_the_lines_through_merge_and_split():
    own = [{"render": "flat"}, {"render": "handoff", "material": "ice"}, {"render": "ltx"}, {"render": "flat"}, {"render": "handoff"}, {"render": "ltx"}]
    c, errors, fixes = CO.normalize(dict(OBJ, words=["A B", "C", "D E", "F G", "H", "I J"], owners=own), 3, [], 16)       # 6 lines merged to 4 for 4 shots
    assert not errors and len(c["words"]) == 4 and len(c["owners"]) == 4
    assert c["words"] == ["A B C", "D E", "F G H", "I J"] and [o["render"] for o in c["owners"]] == ["flat", "ltx", "flat", "ltx"]     # a merged line takes the owner of its first line
    c, errors, fixes = CO.normalize(dict(OBJ, words=["AFTER HOURS", "THE CITY KEEPS", "DREAMING"], owners=[{"render": "ltx"}, {"render": "flat"}, {"render": "handoff"}]), 3, [], 16)
    assert len(c["words"]) == 4 and len(c["owners"]) == 4 and [o["render"] for o in c["owners"]].count("ltx") >= 1      # a split line: both halves keep the owner (see remap_owners below)
    assert OW.remap_owners([{"render": "ltx"}, {"render": "flat"}], ["AFTER HOURS", "DREAMING"], ["AFTER", "HOURS", "DREAMING"]) == [{"render": "ltx"}, {"render": "ltx"}, {"render": "flat"}]


@pytest.fixture(scope="module")
def music():
    return synth_music(120, 16)


REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket",
       "music": "calm piano", "length": 16, "look": "auto", "seed": 5, "quality": "preview", "format": "9:16"}


def test_plan_uses_the_concept_owners_and_the_rules_only_where_it_gave_none(music):
    base = pr.plan_piece(REQ, music)
    n = len(base["type"]["words"])
    assert n >= 3 and base["plan"]["owner"]["source"] == "rules"
    assert [w["render"] for w in base["type"]["words"]] == ["flat"] * n                      # the rules' default is the flat graphic (2026-10-29; it was handoff)
    own = [{"render": "flat"}, {"render": "ltx", "material": "wet harbour stone, barnacles"}, {"render": "handoff", "ltx_at": 0.3, "material": "brushed steel plates"}, {}][:n]
    own += [{"render": "flat"}] * (n - len(own))
    req = dict(REQ, words=" / ".join(w["text"] for w in base["type"]["words"]), owners=json.dumps(own))          # the concept already fitted the lines: one owner per line
    plan = pr.plan_piece(req, music)
    from tools.fusion import user_style as US
    US.stamp(plan, "fill the letters with a material")                  # 2026-11-01: an LTX material word is flat unless the user asks for a textured letter (letter_fill.py); the typed text lands on the piece, the owner pass runs again
    OW.apply_to_plan(plan)
    ws = plan["type"]["words"]
    assert plan["plan"]["owner"]["source"] == "concept"
    assert ws[0]["render"] == "flat" and "material" not in ws[0] and "ltx_at" in ws[0]
    assert ws[1]["render"] == "ltx" and ws[1]["material"] == "wet harbour stone, barnacles"
    assert ws[2]["render"] == "handoff" and ws[2]["ltx_at"] == 0.3 and ws[2]["material"] == "brushed steel plates"
    if n > 3:
        assert ws[3]["render"] == "flat" and "material" not in ws[3]                           # the entry {} -> the rules' default for that word only (flat)
    for w, s in zip(ws, plan["shots"]):                                                         # the shots carry the chosen owners: flat shots get no clause and guide 0
        assert (w["render"] == "flat") == (s["control"]["guide"] == 0.0)
        assert ('formed as' in s["prompt"]) == (w["render"] != "flat")
    assert pr.normalize_request(dict(REQ, owners="not json"))["owners"] is None                  # unreadable: the rules decide
    assert pr.normalize_request(pr.normalize_request(dict(REQ, owners=own)))["owners"] == own    # survives a second normalisation


# ------------------------------------------------------------------------------------------------ 2. cap + clean clause
def _shot_piece(texts, dur_s, render="handoff"):
    n = int(dur_s * 24)
    step = dur_s / len(texts)
    words = [{"text": t, "t0": round(i * step, 3), "t1": round((i + 1) * step - 0.2, 3), "render": render, "ltx_at": 0.5, "material": "wind-worn sandstone slabs, crumbling edges"}
             for i, t in enumerate(texts)]
    return {"fps": 24, "layout": {"s1": {"start": 0, "frames": n, "xf": 0}}, "plan": {"notes": []},
            "user_style": {"textured": True},                                    # 2026-11-01: letters are solid unless the user asks for a material (letter_fill.py); these pieces are the ones that asked
            "type": {"words": words},
            "shots": [{"id": "s1", "frames": n, "prompt": "a potter at a wheel, slow camera rise, cinematic", "control": {"pass": "depth", "guide": 1.0}, "guide": 1.0}]}


def test_cap_keeps_one_ltx_word_in_a_short_shot_and_two_in_a_long_one():
    p = _shot_piece(["HANDS", "REMIND", "CLAY", "SPIN", "ON"], 6.0)
    fx = OW.cap_per_shot(p)
    ws = p["type"]["words"]
    assert [w["render"] for w in ws] == ["flat", "handoff", "flat", "flat", "flat"] and len(fx) == 4        # REMIND = the longest key word
    assert all("ltx_at" not in ws[i] and "material" not in ws[i] for i in (0, 2, 3, 4)) and "REMIND" in fx[0] and "flat" in fx[0]
    p = _shot_piece(["HANDS", "REMIND", "CLAY", "SPIN", "ON"], 16.0)
    OW.cap_per_shot(p)
    assert [w["render"] for w in p["type"]["words"]] == ["handoff", "handoff", "flat", "flat", "flat"]            # the two longest words
    assert sum(w["render"] != "flat" for w in p["type"]["words"]) == 2 and OW.shot_cap(7.9) == 1 and OW.shot_cap(8.0) == 2
    p = _shot_piece(["A", "B"], 6.0, render="flat")
    assert OW.cap_per_shot(p) == []                                                                    # nothing owned: nothing to cap
    assert OW.cap_per_shot({"fps": 24, "type": {"words": [{"text": "A", "t0": 0, "t1": 2}]}, "shots": []}) == []        # an old piece


def test_ltx_word_is_at_most_six_letters_longer_ones_go_flat():
    p = _shot_piece(["REMEMBER", "SPIN"], 16.0)
    fx = OW.cap_per_shot(p)
    ws = p["type"]["words"]
    assert [w["render"] for w in ws] == ["flat", "handoff"] and len(fx) == 1 and "REMEMBER" in fx[0] and "8 letters" in fx[0]
    assert "ltx_at" not in ws[0] and "material" not in ws[0]
    p = _shot_piece(["STORM BREAKER", "NO WAY", "ICE-COLD"], 24.0)                       # letters only: 'NO WAY' = 5, 'ICE-COLD' = 7, 'STORM BREAKER' = 11
    OW.cap_per_shot(p)
    assert [w["render"] for w in p["type"]["words"]] == ["flat", "handoff", "flat"] and OW.MAX_LTX_LETTERS == 6
    p = _shot_piece(["REMEMBER"], 6.0)
    OW.apply_to_plan(p)
    assert p["type"]["words"][0]["render"] == "flat" and "formed as" not in p["shots"][0]["prompt"] and p["shots"][0]["control"]["guide"] == OW.OFF_GUIDE
    from tools.fusion import concept as CO
    assert "at most 6 letters" in CO.concept_prompt(1, 16, [])


def test_apply_to_plan_names_only_the_kept_word_and_logs_the_fix():
    p = _shot_piece(["HANDS", "REMIND", "CLAY"], 6.0)
    p["shots"][0]["control"]["guide"] = p["shots"][0]["guide"] = 0.4                                    # the planner's recipe guide
    OW.apply_to_plan(p)
    pr_ = p["shots"][0]["prompt"]
    assert pr_.count("formed as") == 1 and 'the word "REMIND" formed as wind-worn sandstone slabs, crumbling edges letters, extruded 3D lettering' in pr_
    assert '"HANDS"' not in pr_ and '"CLAY"' not in pr_ and pr_.endswith("cinematic") and p["shots"][0]["control"]["guide"] == OW.OWNED_GUIDE
    assert len(p["plan"]["owner"]["fixes"]) == 2 and any("HANDS" in n for n in p["plan"]["notes"])
    assert OW.apply_to_plan(p)["shots"][0]["prompt"] == pr_                                            # idempotent


def test_clause_wording_and_clean_prompt_strip_it():
    from tools.app import layered_shots as LS
    assert OW.clause("spin", "thick carved ice, extruded, frosted edges") == 'the word "SPIN" formed as thick carved ice, extruded, frosted edges letters, extruded 3D lettering'
    assert OW.clause("AFTER HOURS", "brushed steel") == 'the words "AFTER HOURS" formed as brushed steel letters, extruded 3D lettering'
    assert "letters letters" not in OW.clause("SPIN", "carved ice letters") and "letters letters" not in OW.clause("SPIN", "carved ice lettering")
    base = "a potter at a wheel, slow camera rise, cinematic"
    one = OW.prompt_with_clause(base, [dict(start=0.0, end=4.0, next=None, render="handoff", ltx_at=0.5, ltx_t=None, text="SPIN", material="carved ice")], 0.0, 4.0)
    assert one == 'a potter at a wheel, slow camera rise, the word "SPIN" formed as carved ice letters, extruded 3D lettering, cinematic'
    assert LS.clean_prompt(one) == base and OW.strip_clause(one) == "a potter at a wheel, slow camera rise, cinematic"
    old = 'a potter, the words "HANDS" formed as wind-worn sandstone slabs, crumbling edges letters, the words "SPIN" formed as carved ice letters, cinematic'     # the 2026-10-28 format
    assert LS.clean_prompt(old) == "a potter, cinematic"
    two = 'a potter, the word "A" formed as ice letters, extruded 3D lettering, the word "B" formed as steel letters, extruded 3D lettering, cinematic'
    assert LS.clean_prompt(two) == "a potter, cinematic"


# ------------------------------------------------------------------------------------------------ 3. fallback when LTX formed no letters
def _word_mask(text, hw, y_frac=0.5):
    H, W = hw
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)
    sc = 0.8 * W / tw
    m = np.zeros((H, W), np.uint8)
    cv2.putText(m, text, (int(0.1 * W), int(y_frac * H)), cv2.FONT_HERSHEY_DUPLEX, sc, 255, max(2, int(sc * 3)))
    return m


def _write_matte_dir(tmp, piece, hw, n, y_by_word=None):
    """a type pass dir: matte/matte_NNNNN.png = the word of the piece's window the frame falls in, drawn big in the centre (a stand-in for the type matte)."""
    wins = OW.windows(piece)
    d = Path(tmp) / "matte"
    d.mkdir(parents=True, exist_ok=True)
    blank = np.zeros(hw, np.uint8)
    cache = {}
    for f in range(n):
        t = f / 24.0
        img = blank
        for i, w in enumerate(wins):
            if w["start"] <= t < w["end"]:
                img = cache.setdefault(i, _word_mask(w["text"], hw, (y_by_word or {}).get(i, 0.5)))
        cv2.imwrite(str(d / f"matte_{f:05d}.png"), img)
    return str(tmp)


class _Lazy:
    """a frame sequence read on demand (the real clips are 385 frames: never decode them all). extra(i, frame) -> frame draws on top."""

    def __init__(self, path, extra=None):
        from tools.fusion import compose as C
        self.src = C.FrameSource(str(path), "rgb", None)
        self.shape = (len(self.src),) + tuple(np.asarray(self.src[0]).shape)
        self.extra = extra

    def __getitem__(self, i):
        f = np.asarray(self.src[int(i)], np.float32).copy()
        return self.extra(int(i), f) if self.extra else f


def test_score_frames_separates_letters_from_scene_drift():
    rng = np.random.default_rng(1)
    clean = [np.clip(0.5 + 0.1 * rng.standard_normal((64, 48, 3)), 0, 1).astype(np.float32) for _ in range(6)]
    mask = _word_mask("HI", (64, 48))
    drift = [np.clip(c + 0.03 + 0.01 * rng.standard_normal(c.shape), 0, 1).astype(np.float32) for c in clean]       # global grade shift + noise, no letters
    r = LC.score_frames((c, t, mask) for c, t in zip(clean, drift))
    assert r["n"] == 6 and r["excess"] < LC.MIN_EXCESS * 0.5
    lettered = []
    for t in drift:
        t = t.copy()
        t[mask > 0] = 0.95
        lettered.append(t)
    r2 = LC.score_frames((c, t, mask) for c, t in zip(clean, lettered))
    assert r2["excess"] > LC.MIN_EXCESS * 2
    assert LC.score_frames((c, t, np.zeros((64, 48), np.uint8)) for c, t in zip(clean, lettered)) is None          # an empty mask: unknown, not "missing"


needs_real = pytest.mark.skipif(not (REAL / "s1_clean.mp4").exists(), reason="batch 3060 v005 r01 clips not on disk")


@needs_real
def test_real_v005_r01_clips_have_no_letters_so_every_handoff_word_falls_back_to_flat(tmp_path, capsys):
    piece = json.loads((REAL / "piece.json").read_text(encoding="utf-8"))
    assert [w["render"] for w in piece["type"]["words"]] == ["handoff"] * 5
    td = _write_matte_dir(tmp_path / "type", piece, (640, 360), 385, y_by_word={0: 0.55, 1: 0.35, 2: 0.7, 3: 0.45, 4: 0.6})
    clean, text = _Lazy(REAL / "s1_clean.mp4"), _Lazy(REAL / "s1_textbg.mp4")
    rows = LC.check_shot(piece, "s1", td, clean, text)
    print("\nMEASURED real v005 r01 (no letters):", [(r["text"], r["inside"], r["outside"], r["excess"], r["n"]) for r in rows])
    assert len(rows) == 5 and all(r["formed"] is False and r["n"] >= LC.MIN_FRAMES for r in rows)
    assert max(r["excess"] for r in rows) < LC.MIN_EXCESS * 0.6                                               # a wide margin below the threshold
    msgs = LC.apply_fallback(piece, "s1", td, clean, text)
    assert len(msgs) == 5 and all(m.startswith("ltx letters missing -> flat") for m in msgs)
    assert all(w["render"] == "flat" and w["ltx_missing"] and w["render_asked"] == "handoff" and "ltx_at" not in w for w in piece["type"]["words"])
    assert len(piece["plan"]["owner"]["fallback"]) == 5 and len(piece["plan"]["owner"]["ltx_check"]) == 5
    assert OW.windows(piece) is not None and all(OW.flat_gain([w], 8.0) == 1.0 for w in OW.windows(piece))   # the flat type is on for the whole span
    assert LC.apply_fallback(piece, "s1", td, clean, text) == []                                              # nothing left to check


@needs_real
def test_fallback_does_not_trigger_where_letters_are_drawn_into_the_text_clip(tmp_path):
    piece = json.loads((REAL / "piece.json").read_text(encoding="utf-8"))
    ys = {0: 0.55, 1: 0.35, 2: 0.7, 3: 0.45, 4: 0.6}
    td = _write_matte_dir(tmp_path / "type", piece, (640, 360), 385, ys)
    wins = OW.windows(piece)
    masks = {i: _word_mask(w["text"], (640, 360), ys[i]) > 0 for i, w in enumerate(wins)}

    def draw(i, f):                                      # letters (stone grey) formed in the text clip for word 1 and 3 only, from their handoff on
        t = i / 24.0
        for k in (1, 3):
            w = wins[k]
            if OW.handoff_t(w) <= t < w["end"]:
                f[masks[k]] = (0.62, 0.58, 0.50) if f[masks[k]].mean() > 0.4 else (0.9, 0.85, 0.75)
        return f

    clean, text = _Lazy(REAL / "s1_clean.mp4"), _Lazy(REAL / "s1_textbg.mp4", draw)
    rows = {r["i"]: r for r in LC.check_shot(piece, "s1", td, clean, text)}
    print("\nMEASURED real clips + letters drawn into words 1 and 3:", {i: (r["inside"], r["outside"], r["excess"]) for i, r in rows.items()})
    assert rows[1]["formed"] is True and rows[3]["formed"] is True and all(rows[i]["formed"] is False for i in (0, 2, 4))
    msgs = LC.apply_fallback(piece, "s1", td, clean, text)
    ws = piece["type"]["words"]
    assert len(msgs) == 3 and [w["render"] for w in ws] == ["flat", "handoff", "flat", "handoff", "flat"]      # the words with letters keep their handoff


def test_fallback_skips_what_it_cannot_judge(tmp_path):
    p = _shot_piece(["HANDS", "REMEMBER"], 6.0)
    clean = np.zeros((144, 64, 48, 3), np.uint8)
    text = np.zeros_like(clean)
    lax = copy.deepcopy(p)
    lax["layers"] = {"ltx_strict": False}
    assert LC.apply_fallback(lax, "s1", str(tmp_path), clean, text) == [] and all(w["render"] == "handoff" for w in lax["type"]["words"])      # no matte pass dir: unknown; trusted only when the piece says so
    assert LC.apply_fallback(p, "s1", str(tmp_path), None, text) == [] and LC.apply_fallback(p, "s1", str(tmp_path), clean, None) == []      # not layered
    off = copy.deepcopy(p)
    off["layers"] = {"ltx_check": False}
    assert LC.apply_fallback(off, "s1", str(tmp_path), clean, text) == []
    assert all(w["render"] == "handoff" for w in off["type"]["words"])
    msgs = LC.apply_fallback(p, "s1", str(tmp_path), clean, text)             # 2026-10-29: the text pass is the background only when its word PASSED; a word that cannot be verified is flat
    assert len(msgs) == 2 and all("not verified" in m for m in msgs) and all(w["render"] == "flat" for w in p["type"]["words"])
    draft = copy.deepcopy(p)
    draft["quality"] = "draft"
    assert LC.check_shot(draft, "s1", str(tmp_path), clean, text) == []                                        # a draft has no LTX letters at all
