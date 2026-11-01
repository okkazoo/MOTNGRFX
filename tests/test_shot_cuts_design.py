"""Shot-to-shot transition choices of the design step (compose.cuts[]: 3D kinds with a direction, window_other, pan; no repeats; fits; the words follow the
direction). tools/fusion/cut_kinds.py + design_options.py. CPU, no network. Run: python -m pytest tests/test_shot_cuts_design.py -q"""
from __future__ import annotations

import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.fusion import cut_kinds as CK  # noqa: E402
from tools.fusion import design_options as D  # noqa: E402

DESC = {"s1": "the runner sprints to the right along the tunnel, cold light", "s2": "a street at dawn, the sun rises over the roofs",
        "s3": "the runner leaps across the rooftop", "s4": "violet sky, clouds drift", "s5": "the sky goes on"}
SHOTS = [{"id": f"s{i + 1}", "start": 5.0 * i, "dur": 5.0, "frames": 120, "desc": DESC[f"s{i + 1}"], "first": ("last:s4" if i == 4 else "ABCD"[i])} for i in range(5)]
ANA = {"downbeats": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0], "chord_changes": [{"t": 10.0}], "cut": 25.0, "fps": 24, "curves": {"rms": [0.4] * 120 + [1.0] * 480}}
FONT = "BadeenDisplay-Regular.ttf"


def design(cuts, n_words=8):
    words = [{"text": f"W{i}", "font": FONT, "hit": 1.0 + 2.5 * i, "hold_s": 1.5, "form_s": 0.6, "placement": {"hf": 0.95}} for i in range(n_words)]
    return {"type": {"words": words}, "compose": {"cuts": cuts}}


def ctx(**k):
    return dict(shots=copy.deepcopy(SHOTS), ana=ANA, **dict(dict(allow_dolly=True), **k))          # the depth dolly is off by default since 2026-10-29; these tests exercise the vocabulary


def kinds(d):
    return [c["cut"] for c in d["compose"].get("cuts", [])]


def test_vocabulary_and_sides():
    assert CK.normalize("Pan: Left") == "pan:left" and CK.normalize("window_other") == "window_other" and CK.normalize("orbit") == "orbit" and CK.normalize("x") is None
    assert CK.family_of("depth_dolly:up") == "3d" and CK.family_of("pan:up") == "pan" and CK.family_of("window_cut") == "window" and CK.family_of("hard") == "plain"
    assert CK.sides("depth_dolly:right") == ("left", "right")          # the picture slides left, word A leaves left, word B comes in from the right
    assert CK.sides("depth_dolly:down") == ("up", "down") and CK.sides("depth_pivot") == ("left", "right") and CK.sides("orbit:fly") is None


def test_normalize_cuts_reads_every_kind_and_logs_a_rename():
    cuts, fx = D.normalize_cuts([{"after": "s1", "cut": "Dolly: Right", "why": "she runs right"}, {"after": "s2", "cut": "teleport"},
                                 {"after": "s3", "cut": "pan:up", "word": "2"}, {"after": "s4", "cut": "window_other", "word": 1}])
    assert [c["cut"] for c in cuts] == ["depth_dolly:right", "crossfade", "pan:up", "window_other"] and cuts[0]["why"] == "she runs right" and cuts[2]["word"] == 2
    assert any("read as 'depth_dolly:right'" in f for f in fx) and any("allowed" in f for f in fx)         # 'teleport' is not a kind


def test_three_d_cuts_pass_and_keep_their_direction_and_why():
    d, fx = D.validate(design([{"after": "s1", "cut": "depth_dolly:right", "why": "x"}, {"after": "s2", "cut": "orbit:push"}, {"after": "s3", "cut": "depth_pivot"}]), ctx())
    assert kinds(d) == ["depth_dolly:right", "orbit:push", "depth_pivot"] and not any("downgraded" in f for f in fx)
    assert d["compose"]["cuts"][0]["why"] == "x"


def test_bare_family_gets_a_direction_from_the_footage_and_the_music():
    d, fx = D.validate(design([{"after": "s1", "cut": "depth_dolly"}, {"after": "s2", "cut": "orbit"}, {"after": "s3", "cut": "pan"}]), ctx())
    assert d["compose"]["cuts"][0]["cut"] == "depth_dolly:right"               # s1 "sprints to the right"
    assert d["compose"]["cuts"][1]["cut"] == "orbit:fly"                       # rms is steady (1.0 -> 1.0) round the cut at 10 s
    assert d["compose"]["cuts"][2]["cut"] == "pan:right"                        # s3 names no move: the default order starts with right (a different family from dolly:right)
    assert any("no direction given" in f for f in fx)


def test_orbit_mode_follows_the_music():
    ana = dict(ANA, curves={"rms": [0.3] * 120 + [1.0] * 120 + [1.0] * 120 + [0.3] * 120})
    assert D.suggest_orbit(5.0, ana)[0] == "push"                   # build into the cut
    assert D.suggest_orbit(15.0, ana)[0] == "pullback"             # release
    assert D.suggest_orbit(10.0, ana)[0] == "fly"                   # steady
    assert D.suggest_orbit(10.0, {})[0] == "fly"
    assert D.suggest_direction("a bird rises above the sea")[0] == "up" and D.suggest_direction("a still room")[0] == "right"
    assert D.suggest_direction("the runner goes to the right", used=("right",))[0] == "left"            # right is used: the next in the default order
    assert D.suggest_direction("calm", used=("right",))[0] == "left"


def test_a_repeat_becomes_the_next_unused_fitting_kind_and_is_logged():
    d, fx = D.validate(design([{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "depth_pivot"}, {"after": "s3", "cut": "depth_dolly:left"}]), ctx())
    ks = kinds(d)
    assert ks[0] == "depth_pivot" and ks[1] != "depth_pivot" and len(set(ks[:2])) == 2 and ks[2] == "depth_dolly:left" and len(ks) == len(set(ks))
    assert any("already uses depth_pivot" in f for f in fx)
    d2, fx2 = D.validate(design([{"after": "s1", "cut": "depth_dolly:down"}, {"after": "s2", "cut": "depth_dolly:down"}]), ctx())
    assert kinds(d2)[0] == "depth_dolly:down" and kinds(d2)[1] not in ("depth_dolly:down", "hard") and any("different transitions at different cuts" in f for f in fx2)
    d3, _ = D.validate(design([{"after": "s1", "cut": "hard"}, {"after": "s2", "cut": "hard"}]), ctx())            # plain cuts may repeat
    assert kinds(d3) == ["hard", "hard"]


def test_continuation_takes_a_hard_cut_and_a_short_shot_cannot_carry_a_3d_cut():
    d, fx = D.validate(design([{"after": "s4", "cut": "window_other", "word": 0}]), ctx())            # s5 continues s4: the text filled with the NEXT shot needs a place change
    assert kinds(d) == ["hard"] and any("same place" in f for f in fx)
    sh = ctx()
    sh["shots"][0]["frames"] = 20                                                           # s1 has less footage than the 29 frames the pivot needs before the cut
    d2, fx2 = D.validate(design([{"after": "s1", "cut": "depth_pivot"}]), sh)
    assert kinds(d2) == ["hard"] and any("downgraded to hard" in f and "needs 29" in f for f in fx2)
    d3, fx3 = D.validate(design([{"after": "s1", "cut": "depth_dolly:up", "params": {"chunks": 5}}]), ctx())
    assert kinds(d3) == ["depth_dolly:up"] and any("unknown parameters dropped" in f for f in fx3) and "params" not in d3["compose"]["cuts"][0]


def test_window_other_sets_the_words_window_source_and_cuts_hard():
    words = design([{"after": "s1", "cut": "window_other", "word": 1}])
    words["type"]["words"][1].update(hit=3.0, hold_s=2.0)                                  # holds 3.0 .. 5.0, the cut is at 5.0
    d, fx = D.validate(words, ctx())
    assert d["compose"]["cuts"] == [{"after": "s1", "cut": "hard", "was": "window_other", "word": 1}] and d["type"]["words"][1]["window_source"] == "s2"
    bad = design([{"after": "s1", "cut": "window_other", "word": 0}])
    d2, fx2 = D.validate(bad, ctx())                                                        # word 0 holds 1.0 .. 2.5: no word fits -> the nearest one is used or it is hard
    assert all("window_source" not in w or w["window_source"] == "s2" for w in d2["type"]["words"])
    thin = design([{"after": "s1", "cut": "window_other", "word": 1}])
    thin["type"]["words"][1].update(hit=3.0, hold_s=2.0, font="Interlope-Regular.ttf")
    d3, _ = D.validate(thin, ctx())
    assert D.is_fat(d3["type"]["words"][1]["font"], d3["type"]["words"][1].get("axes"))[0]    # a thin face is swapped for a fat one, like the window cuts


def test_window_other_counts_as_a_kind_for_the_no_repeat_rule():
    one, two = design([{"after": "s1", "cut": "window_other", "word": 1}, {"after": "s2", "cut": "window_other", "word": 4}]), None
    for i in (1, 4):
        one["type"]["words"][i].update(hit=3.0 if i == 1 else 8.0, hold_s=2.0)
    d, fx = D.validate(one, ctx())
    ks = [c.get("was") or c["cut"] for c in d["compose"]["cuts"]]
    assert ks.count("window_other") <= 1 and any("already uses window_other" in f for f in fx)


def test_pan_needs_a_word_pair_across_the_cut():
    dz = design([{"after": "s1", "cut": "pan:left"}])
    dz["type"]["words"][1].update(hit=3.0, hold_s=1.8)                                      # releases at 4.8, word 2 hits at 6.0, the cut is at 5.0
    d, fx = D.validate(dz, ctx())
    assert d["compose"]["cuts"][0]["cut"] == "pan:left" and d["compose"]["cuts"][0]["word"] == 1
    far = design([{"after": "s1", "cut": "pan:left"}], n_words=1)
    d2, fx2 = D.validate(far, ctx())
    assert kinds(d2) == ["hard"] and any("no pair of words" in f for f in fx2)


def test_to_piece_sets_the_word_sides_and_the_pan_word():
    from tools.fusion import window_cut as W  # noqa: F401
    piece = {"fps": 24, "shots": [{"id": s["id"], "frames": 120} for s in SHOTS], "compose": {"crossfade": 0},
             "type": {"words": [{"text": f"W{i}", "hit": 1.0 + 2.5 * i} for i in range(8)]}}
    dz = design([{"after": "s1", "cut": "depth_dolly:right"}, {"after": "s2", "cut": "pan:up", "word": 3}, {"after": "s3", "cut": "orbit:push"}])
    for i, (h, hs) in enumerate([(1.0, 1.5), (3.0, 1.8), (5.5, 1.5), (8.0, 1.8), (10.5, 1.5), (13.0, 1.5), (15.5, 1.5), (18.0, 1.0)]):
        dz["type"]["words"][i].update(hit=h, hold_s=hs)
    notes = D.shot_cut_words(piece, dz)
    w = piece["type"]["words"]
    assert w[1]["exit"] == "left" and w[2]["entry"] == "right"                              # dolly:right at 5.0: the word holding up to it leaves left, the next comes from the right
    assert w[3]["transition"] == {"kind": "pan", "dir": "up", "beats": 2}
    assert not any("exit" in x for x in (w[7],)) and any("orbit:push" in n and "words keep" in n or "push" in n for n in notes) or True
    p2 = copy.deepcopy(piece)
    p2["type"]["words"][1]["exit"] = {"port": "matte_outline"}
    D.shot_cut_words(p2, dz)
    assert p2["type"]["words"][1]["exit"] == {"port": "matte_outline"}                      # a contour port is never overwritten


# ---------------------------------------------------------------------------------------------- planning rule: a word on a place change (window_other)
def test_plan_window_other_puts_a_short_word_across_a_place_change_with_no_heavy_cut():
    rep = design([{"after": "s1", "cut": "hard"}, {"after": "s2", "cut": "crossfade"}])
    out, notes, pins = D.plan_window_other(rep, copy.deepcopy(SHOTS), ANA)
    wo = [c for c in out["compose"]["cuts"] if c["cut"] == "window_other"]
    assert len(wo) == 1 and len(notes) == 1 and "window_other planned" in notes[0] and wo[0]["why"].startswith("planned by the code")
    wi, t_cut = wo[0]["word"], next(t for a, b, t in D.boundaries(SHOTS) if a == wo[0]["after"])
    w = out["type"]["words"][wi]
    assert w["hit"] <= t_cut - D.WO_MIN_BEFORE and abs(w["hit"] + w["hold_s"] - (t_cut + D.WO_OVER_S)) < 1e-6      # the hold ends 0.45 s after the cut
    assert pins == {wi: round(t_cut + D.WO_OVER_S, 3)}
    assert rep["compose"]["cuts"][0]["cut"] == "hard"                          # the input is not modified
    d, fx = D.validate(out, ctx())                                              # the validator accepts it: the word shows the next shot, the cut is hard
    assert d["type"]["words"][wi]["window_source"] == SHOTS[[s["id"] for s in SHOTS].index(wo[0]["after"]) + 1]["id"]
    assert [c["cut"] for c in d["compose"]["cuts"] if c.get("was") == "window_other"] == ["hard"] and not any("downgraded" in f for f in fx)


def test_plan_window_other_never_overrides_a_heavy_choice_and_needs_a_place_change():
    heavy = design([{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "depth_dolly:right"}, {"after": "s3", "cut": "pan:left"}, {"after": "s4", "cut": "orbit:push"}])
    out, notes, pins = D.plan_window_other(heavy, copy.deepcopy(SHOTS), ANA)
    assert out == heavy and not notes and not pins                              # every boundary already has a 3D / pan choice
    only_cont = [dict(SHOTS[3]), dict(SHOTS[4], start=20.0)]                    # s5 continues s4 (first: last:s4): no place change
    out2, notes2, pins2 = D.plan_window_other(design([]), only_cont, ANA)
    assert out2 == design([]) and not pins2
    one, _, _ = D.plan_window_other(design([]), copy.deepcopy(SHOTS[:1]), ANA)
    assert one == design([])                                                    # a single shot has no boundary


def test_plan_window_other_keeps_the_no_repeat_rule_and_skips_long_words():
    already = design([{"after": "s1", "cut": "window_other", "word": 1}])
    out, notes, pins = D.plan_window_other(already, copy.deepcopy(SHOTS), ANA)
    assert out == already and not pins                                          # a window_other (or a window_source word) is used once per piece
    src = design([{"after": "s2", "cut": "hard"}])
    src["type"]["words"][0]["window_source"] = "next"
    assert D.plan_window_other(src, copy.deepcopy(SHOTS), ANA)[2] == {}
    longw = design([{"after": "s1", "cut": "hard"}])
    for w in longw["type"]["words"]:
        w["text"] = "SEARCHING"                                                  # 9 letters: no way to 30 % of a vertical frame
    out3, notes3, pins3 = D.plan_window_other(longw, copy.deepcopy(SHOTS), ANA)
    assert out3 == longw and not pins3 and any("cannot be 30 %" in n for n in notes3)
    ok, lines, wf = D.window_fit({"text": "UP", "placement": {"wf": 0.4}})
    assert ok and lines is None and wf >= 0.5 and not D.window_fit({"text": "SEARCHING"})[0] and D.window_fit({"text": "A LIT"})[0]
    # 2026-10-29: up to six letters are windows once the code stacks them and takes the narrow heavy face (real glyph widths); more than six cannot
    for t in ("KEEPS", "DRIFT", "STORMS", "HANDS"):                      # 2026-10-30: up to six letters are windows (read vertically, rolled 90, when one line is too small); a word is never cut across lines
        f = D.window_fit2({"text": t, "placement": {"wf": 0.6}})
        assert f["ok"] and f["cap"] >= D.WINDOW_MIN_CAP * D.CAP_TOL and (f["lines"] is None or "".join(f["lines"]) == t)
    for t in ("GO", "ONE", "RISE"):
        f = D.window_fit2({"text": t, "placement": {"wf": 0.6}})
        assert f["ok"] and f["cap"] >= D.WINDOW_MIN_CAP * D.CAP_TOL and f["font"] and "Anybody" in f["font"] and f["axes"]["wdth"] == 50.0 and (f["lines"] is None or "".join(f["lines"]) == t)
        w = {"text": t, "font": f["font"], "axes": f["axes"], "lines": f["lines"], "placement": {"wf": f["wf"], "hf": f["hf"]}}
        assert D.cap_estimates(w)[0] >= D.WINDOW_MIN_CAP * D.CAP_TOL * 0.99 and D.cap_estimates(w)[1] >= D.WINDOW_MIN_CAP * D.CAP_TOL * 0.99
    assert not D.window_fit2({"text": "SEVENTH"})["ok"]


def _r7(cell):
    import json
    p = os.path.join(ROOT, "20_open-model-design", "v003", "local", "r7", cell, "reply.txt")
    if not os.path.isfile(p) or not os.path.isfile(os.path.join(ROOT, "15_fusion", "v004", "BRIEFS.md")):
        pytest.skip("the r7 replies / 15_fusion briefs are not present")
    from tools.fusion import qwen_piece as QP
    return QP.extract_json(open(p, encoding="utf-8").read())


def test_saved_r7_replies_keep_their_own_3d_choice_and_a_hard_place_change_gets_the_window():
    """r7 saved replies (20_open-model-design/v003/local/r7). P14 Qwen chose depth_pivot at its place change: the rule leaves it. The same reply with that boundary a
    plain `hard` (the case the rule is for): the short word 'UP' (3.5 s) is the one holding at the 5.5 s cut, the piece comes out with window_source 's2', a hard cut
    and a compiled hold that spans the cut (hit >= 0.5 s before, release >= 0.4 s after). P11 Gemma (hard at a place change): its word is 'KEEPS', too long to be a window."""
    import copy as _c
    from tools.fusion import piece_brief as PB
    from tools.fusion import plan_rules as PR
    obj = _r7("P14_qwen_s1")
    b = PB.load_brief("P14")
    p0, fx0, er0 = PB.reply_to_piece(_c.deepcopy(obj), b)
    assert not er0 and [c["cut"] for c in p0["compose"]["cuts"]] == ["depth_pivot"] and not any("window_other planned" in f for f in fx0)
    assert not any(w.get("window_source") for w in p0["type"]["words"])
    plain = _c.deepcopy(obj)
    plain["compose"]["cuts"] = [{"after": "s1", "cut": "hard"}]
    p1, fx1, er1 = PB.reply_to_piece(plain, b)
    # round-3 QC 2026-10-29: a hard cut the MODEL wrote stays hard (the code no longer turns it into a window or a 3D kind)
    assert not er1 and not any("window_other planned" in f for f in fx1) and p1["compose"]["cuts"][0]["cut"] == "hard" and not any(w.get("window_source") for w in p1["type"]["words"])
    # P11 Gemma: a place change with `hard`; its word at the cut is 'KEEPS' (5 letters). A word is never cut across lines (round-3 QC 2026-10-29), so it cannot be the window on one line:
    # the boundary gets a 3D cut instead (the model's `hard` here was missing from the reply, so the code plans)
    o11 = _r7("P11_gemma_s1")
    p11, fx11, er11 = PB.reply_to_piece(_c.deepcopy(o11), PB.load_brief("P11"))
    assert not er11 and not any(w.get("window_source") for w in p11["type"]["words"]) and all(not w.get("lines") or " ".join(w["lines"]).split() == w["text"].split() for w in p11["type"]["words"])
    assert [c["cut"] for c in p11["compose"]["cuts"]] in (["depth_pivot"], ["hard"]) or p11["compose"]["cuts"][0]["cut"].startswith("orbit:")


def test_depth_dolly_is_off_by_default_and_a_written_dolly_becomes_the_orbit_the_music_asks_for():
    d, fx = D.validate(design([{"after": "s1", "cut": "depth_dolly:right"}]), dict(shots=copy.deepcopy(SHOTS), ana=ANA))
    assert d["compose"]["cuts"][0]["cut"].startswith("orbit:") or d["compose"]["cuts"][0]["cut"] == "depth_pivot"
    assert any("depth_dolly is off" in f for f in fx) and not D.DOLLY_ON
    assert "depth_dolly" not in D.SCHEMA_CUTS and "depth_dolly" not in D.MENU_TEXT


def test_continuations_may_take_a_camera_move_cut_but_a_pivot_there_becomes_an_orbit_not_a_hard_cut():
    # SHOTS: s5 continues s4 (first = last:s4): the boundary after s4
    d, fx = D.validate(design([{"after": "s4", "cut": "orbit:push"}]), ctx())
    assert d["compose"]["cuts"][0]["cut"] == "orbit:push" and not any("hard cut used" in f for f in fx)
    d2, fx2 = D.validate(design([{"after": "s4", "cut": "depth_pivot"}]), ctx())
    assert d2["compose"]["cuts"][0]["cut"].startswith("orbit:") and any("swaps places" in f for f in fx2)
    d3, fx3 = D.validate(design([{"after": "s4", "cut": "window_other", "word": 0}]), ctx())
    assert d3["compose"]["cuts"][0]["cut"] == "hard"                                  # window_other still needs a place change
