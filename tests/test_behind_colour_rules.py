"""2026-11-02 regressions from an own-video overlay review (a word behind a man standing in the middle of the frame, letters showing through his jacket, and a light peach fill on most words):
(1) readable-behind rule: every glyph >= 70 % and the word >= 85 % visible, else the word is FRONT for its whole hold; a nudge only where it passes with the subject still overlapping;
(2) the occluder is the subject's SOLID Alpha-Gen matte (holes filled), never a luminosity key, and the guard / apply_behind see it (the old job: `chains: 0`);
(3) colour: hues come from the shot WITHOUT the subject's pixels, the subject's / skin hue is a last resort, no peach ink, varied across words, contrast vs the pixels the word sits on, outline vs fill.
Run: python -m pytest --rootdir=tests -q -p no:cacheprovider tests/test_behind_colour_rules.py     (CPU, synthetic data only)"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.app import overlay_mode as OM  # noqa: E402
from tools.fusion import compose as C  # noqa: E402
from tools.fusion import legible_color as LC  # noqa: E402

W, H = 90, 160


def word_stack(n=5, y0=60, y1=84, x0=8, x1=82):
    """n equal letters in a row (a synthetic word) -> (n, H, W) bool."""
    st = np.zeros((n, H, W), bool)
    cw = (x1 - x0) / n
    for i in range(n):
        st[i, y0:y1, int(x0 + i * cw + 2):int(x0 + (i + 1) * cw - 2)] = True
    return st


def man(frames=8, x0=30, x1=60, y0=20, y1=H):
    """a subject standing in the middle of the frame (full height body)."""
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return [m] * frames


# ------------------------------------------------------------------------------------------------ 1 the readable-behind rule
def test_rule_numbers_are_one_set_everywhere():
    assert OM.VIS_RULE == {"letter": 0.70, "word": 0.85, "most": 0.25}
    assert OM.BEHIND_FLOOR == dict(C.LEG_FLOOR) == {"word": 0.15, "letter": 0.70, "most": 0.25}
    assert C.LEG_BEHIND["word"] == 0.15 and C.LEG_BEHIND["letter"] == 0.70            # no relaxed floor for the behind words any more


def test_a_man_in_the_middle_hides_the_middle_letters_so_the_word_is_front_and_stays_centred():
    st, occ = word_stack(), man()
    v = OM.rule_verdict(OM.word_visibility(st, occ))
    assert not v["ok"] and v["letter_min"] == 0.0 and v["word_mean"] < 0.85           # the middle glyph is gone for the whole hold
    res = OM.plan_word(st, occ, cap=0.15)
    assert res["status"] == "front" and (res["dx"], res["dy"], res["scale"]) == (0.0, 0.0, 1.0)        # no nudge makes a full-height body leave the word readable


def test_the_subject_over_an_end_letter_only_reads_behind():
    st = word_stack()
    occ = man(x0=0, x1=11)                                                              # the man's arm over the first letter by a third
    v = OM.rule_verdict(OM.word_visibility(st, occ))
    vis = OM.word_visibility(st, occ)["vis"][0]
    assert 0.0 < vis[0] < 0.7 or vis[0] >= 0.7                                          # sanity: the first glyph is the one touched
    assert vis[1:].min() == 1.0
    covered = man(x0=0, x1=9)
    v2 = OM.rule_verdict(OM.word_visibility(st, covered))
    assert v2["ok"] and v2["letter_min"] >= 0.7 and v2["word_mean"] >= 0.85


def test_one_glyph_under_70_percent_fails_even_when_the_word_is_mostly_visible():
    st = word_stack(n=8)
    m = np.zeros((H, W), np.float32)
    x0 = int(np.nonzero(st[3].any(0))[0].min())
    m[:, x0:x0 + 5] = 1.0                                                               # eats the left half of glyph 3 only (word loses ~6 %)
    wv = OM.word_visibility(st, [m] * 6)
    v = OM.rule_verdict(wv)
    assert wv["word"].min() > 0.85 and v["letter_min"] < 0.7 and not v["ok"] and v["letter_worst"] == 3


def test_a_brief_crossing_is_tolerated_but_most_of_the_hold_is_not():
    st = word_stack()
    clear = [np.zeros((H, W), np.float32)] * 6
    brief = man()[:2] + clear + clear                                                   # 2 of 14 frames (14 %)
    assert OM.rule_verdict(OM.word_visibility(st, brief))["ok"]
    most = man()[:6] + clear                                                            # 6 of 12 frames
    assert not OM.rule_verdict(OM.word_visibility(st, most))["ok"]


def test_a_nudge_that_clears_the_subject_but_leaves_it_overlapping_is_taken():
    st = word_stack(y0=70, y1=94)
    body = np.zeros((H, W), np.float32)
    body[90:H, 10:80] = 1.0                                                             # the subject's shoulders: the bottom 17 % of the letters (word 83 % visible: fails)
    assert not OM.rule_verdict(OM.word_visibility(st, [body] * 8))["ok"]
    res = OM.plan_word(st, [body] * 8, cap=0.15)
    assert res["status"] == "nudged" and (res["dx"], res["dy"], res["scale"]) != (0.0, 0.0, 1.0)       # a smaller word / a small move clears the shoulders
    assert res["verdict"]["ok"] and res["overlap"] > OM.OVERLAP_MIN                                    # and the subject still overlaps it: a real behind moment
    assert abs(res["dy"]) <= 0.2 and abs(res["dx"]) <= 0.08 and res["scale"] >= 0.9


def test_choose_behind_never_picks_a_failing_word_to_fill_the_quota():
    sc = [dict(frames=10, overlap=1.0, cover=0.5, ok=False), dict(frames=10, overlap=1.0, cover=0.6, ok=False)]
    assert OM.choose_behind(sc) == []
    sc = [dict(frames=10, overlap=1.0, cover=0.05, ok=True), dict(frames=10, overlap=1.0, cover=0.6, ok=False), dict(frames=10, overlap=0.0, cover=0.0, ok=True)]
    assert OM.choose_behind(sc) == [0]                                                  # ok but the subject is nowhere near: plain front type


def test_glyph_stack_without_a_font_gives_one_mask_per_letter():
    st = OM.glyph_stack("AB CD", "no-such-font.ttf", None, ["AB", "CD"], 0.1, (W, H), "line", (0.5, 0.5))
    assert st.shape == (4, H, W) and st.any(axis=(1, 2)).all()


def test_front_words_are_in_front_for_the_whole_hold_behind_words_obey_the_floor():
    n = 60
    ta = np.zeros((H, W), np.float32)
    ta[60:84, 8:82] = 1.0
    light = np.zeros((H, W), np.float32)
    light[:, 10:14] = 1.0                                                               # a sliver of the subject over the word (~5 %)
    fr = C.legible_plan(lambda f: ta, lambda f: light, [(0.5, 2.0)], fps=24, floor=dict(OM.FRONT_FLOOR), letters=False)["holds"][0]
    assert fr["floor"] and fr["boxes"], "a FRONT word is fully in front even when the subject hardly touches it"
    bh = C.legible_plan(lambda f: ta, lambda f: light, [(0.5, 2.0)], fps=24, floor=dict(OM.BEHIND_FLOOR), letters=False)["holds"][0]
    assert not bh["floor"] and not bh["boxes"], "a sliver is readable behind"
    big = np.zeros((H, W), np.float32)
    big[:, 30:60] = 1.0
    bf = C.legible_plan(lambda f: ta, lambda f: big, [(0.5, 2.0)], fps=24, floor=dict(OM.BEHIND_FLOOR), letters=False)["holds"][0]
    assert bf["floor"] and bf["boxes"], "a body over the middle brings the word in front whole"


# ------------------------------------------------------------------------------------------------ 2 the occluder is the solid subject matte
def _lumkey_chain():
    return {"ops": [
        {"op": "luminosity_key", "out": "sk", "args": {"img": "$footage", "low": 0.18, "high": 0.55}},
        {"op": "over", "out": "image_t", "args": {"base": "$footage", "fg": "$tface", "alpha": "$hl0_front"}},
        {"op": "over", "out": "image", "args": {"base": "$image_t", "fg": "$footage", "alpha": "$sk"}}]}


def test_a_luminosity_key_occluder_is_replaced_by_the_solid_matte_and_then_guarded():
    piece = {"fps": 24, "compose": {"size": [540, 960], "default": _lumkey_chain()}}
    assert OM.apply_behind(copy.deepcopy(piece), [(1.0, 2.0), (3.0, 4.0)], [0])["chains"] == 0          # the old job: the key is not a matte, nothing was guarded
    rep = OM.solid_occluder(piece)
    assert rep["replaced"] == 1 and rep["size"] == [540, 960]
    ops = piece["compose"]["default"]["ops"]
    names = [o["op"] for o in ops]
    assert names[:3] == ["luminosity_key", "upscale_matte", "fill_holes"]
    up = ops[1]
    assert up["args"]["matte"] == "$matte" and up["args"]["guide"] == "$footage" and up["args"]["size"] == [540, 960]
    assert ops[2]["args"]["close"] == OM.SOLID_CLOSE and ops[-1]["args"]["alpha"] == "$" + ops[2]["out"]
    r2 = OM.apply_behind(piece, [(1.0, 2.0), (3.0, 4.0)], [0])
    assert r2["chains"] == 1
    legs = [o for o in piece["compose"]["default"]["ops"] if o["op"] == "legible"]
    assert len(legs) == 2 and legs[0]["args"]["floor"] == OM.FRONT_FLOOR and legs[1]["args"]["floor"] == OM.BEHIND_FLOOR


def test_solid_mask_fills_holes_inside_the_body_but_not_outside():
    m = np.zeros((H, W), np.float32)
    m[20:150, 25:65] = 1.0
    m[60:66, 40:48] = 0.0                                                               # a hole in the jacket (a patch a luminosity key would let letters show through)
    m[100:103, 30:34] = 0.15                                                            # a soft noisy patch
    s = OM.solid_mask(m)
    assert s[60:66, 40:48].min() > 0.9
    assert s[:15].max() == 0.0 and s[:, :20].max() == 0.0


def test_guard_ops_treats_a_luminosity_key_alpha_like_a_matte():
    from tools.fusion import recipe_eval as RE
    piece = {"fps": 24, "plan": {"flow_wake": {"holds": [[1.0, 2.0]]}}, "compose": {"holds": [], "default": _lumkey_chain()}, "type": {"words": [{"text": "A"}]}}
    ops = RE.guard_ops(piece, copy.deepcopy(piece["compose"]["default"]["ops"]))
    assert [o for o in ops if o["op"] == "legible"], "the guard wrapped the luminosity key"


# ------------------------------------------------------------------------------------------------ 3 colour from the shot, not the skin
def scene(skin=(0.78, 0.55, 0.40)):
    """a 40x72 frame: left third = the subject (skin / peach), the rest = a dark set with a teal window and a sodium-yellow lamp. -> (frame sRGB 0..1, subject mask)."""
    f = np.zeros((72, 40, 3), np.float32) + np.array([0.12, 0.12, 0.11], np.float32)
    f[:, :14] = np.array(skin, np.float32)
    f[10:30, 28:38] = np.array([0.10, 0.55, 0.55], np.float32)                          # teal window
    f[40:52, 18:26] = np.array([0.95, 0.80, 0.25], np.float32)                          # sodium lamp
    m = np.zeros((72, 40), bool)
    m[:, :14] = True
    return f, m


def test_hues_leave_the_subjects_pixels_out():
    f, m = scene()
    with_subject = [h["h"] for h in LC.shot_hues(LC.srgb_to_lin(f))]
    without = [h["h"] for h in LC.shot_hues(LC.srgb_to_lin(f), valid=~m)]
    skin_h = LC.shot_hues(LC.srgb_to_lin(f), valid=m)[0]["h"]
    assert any(LC.hue_dist(h, skin_h) < 15 for h in with_subject)
    assert not any(LC.hue_dist(h, skin_h) < 15 for h in without)
    assert len(without) >= 2                                                              # teal and yellow stay


def test_the_subjects_hue_goes_last_and_a_peach_ink_is_never_picked_while_anything_else_fits():
    f, m = scene()
    lin = LC.srgb_to_lin(f)
    hues = LC.shot_hues(lin)                                                              # the old way: the skin is a hue of the shot
    subj = [h for h in LC.shot_hues(lin, valid=m) if h["c"] >= 0.04][:2]
    pref, skin, n_fresh = LC.prefer_hues(hues, subj)
    assert skin and all(LC.hue_dist(s["h"], subj[0]["h"]) < LC.LC_SUBJ_NEAR for s in skin[:1])
    assert all(LC.hue_dist(p["h"], subj[0]["h"]) >= LC.LC_SUBJ_NEAR for p in pref)
    bg = LC.srgb_to_lin(np.full((20, 20, 3), 0.15, np.float32))
    hist = LC._hist(LC.rel_lum(bg).reshape(-1), np.ones(400))
    ch, info = LC.pick_ink(hist, pref, skin, n_fresh=n_fresh)
    assert ch is not None and not info["skin"] and not LC.is_peach_ink(ch["lin"])
    assert LC.hue_dist(ch["hue"], subj[0]["h"]) >= LC.LC_SUBJ_NEAR
    assert LC.contrast_ratio(float(LC.rel_lum(ch["lin"])), 0.15 ** 2.2) >= 3.0
    lab = LC.lin_to_oklab(ch["lin"])
    assert float(np.hypot(lab[1], lab[2])) >= LC.LC_INK_MIN_C - 1e-3                        # a colour, not a near-white with a trace of hue


def test_without_a_matte_the_generic_skin_band_stands_in_and_nothing_else_left_falls_back_to_it():
    hues = [dict(h=55.0, c=0.07, L=0.7, share=0.3, score=1.0), dict(h=200.0, c=0.05, L=0.6, share=0.1, score=0.5)]
    pref, skin, _ = LC.prefer_hues(hues, None)
    assert [p["h"] for p in pref] == [200.0] and [s["h"] for s in skin] == [55.0]
    bg = LC.srgb_to_lin(np.full((10, 10, 3), 0.1, np.float32))
    hist = LC._hist(LC.rel_lum(bg).reshape(-1), np.ones(100))
    ch, info = LC.pick_ink(hist, [], skin)
    assert ch is not None and info["skin"], "only the skin hue exists: it is used as the last resort (never a crash / white)"
    assert LC.is_peach_ink(LC.srgb_to_lin(np.array([0.93, 0.74, 0.58], np.float32))) and not LC.is_peach_ink(LC.srgb_to_lin(np.array([0.3, 0.8, 0.8], np.float32)))


def test_inks_vary_across_words_and_outlines_contrast_the_fill():
    hues = [dict(h=200.0, c=0.06, L=0.6, share=0.2, score=2.0), dict(h=95.0, c=0.07, L=0.8, share=0.1, score=1.5), dict(h=310.0, c=0.05, L=0.5, share=0.1, score=1.0)]
    bg = LC.srgb_to_lin(np.full((10, 10, 3), 0.12, np.float32))
    hist = LC._hist(LC.rel_lum(bg).reshape(-1), np.ones(100))
    used, inks = [], []
    for i in range(3):
        pref, skin, nf = LC.prefer_hues(hues, None, used=used[-2:])
        ch, _ = LC.pick_ink(hist, pref, skin, n_fresh=nf)
        used.append(ch["hue"])
        inks.append(ch["lin"])
        sup = LC.support_color(ch["lin"], pref, light=float(LC.rel_lum(ch["lin"])) <= 0.18, pick=i)
        assert LC.contrast_ratio(float(LC.rel_lum(ch["lin"])), float(LC.rel_lum(sup))) >= 3.0 - 1e-6
    hs = [LC._hue_deg(LC.lin_to_oklab(x)) for x in inks]
    assert all(LC.hue_dist(hs[i], hs[i + 1]) >= LC.LC_USED_NEAR for i in range(2)), hs     # a word never repeats the last ink's hue when a fresh hue passes


def test_a_mixed_background_still_gets_a_shot_hue_ink_not_the_palette_fallback():
    """the job's failure: a dark jacket + dark ceiling + a white tee in the box: no ink reaches 85 % of those pixels -> every word fell to the whole-clip palette. The ladder keeps the shot's hue."""
    pix = np.concatenate([np.full(45, 0.06), np.full(30, 0.5), np.full(25, 0.92)]).astype(np.float32)      # dark / mid grey / white
    lin = LC.srgb_to_lin(np.stack([pix, pix, pix], -1).reshape(1, -1, 3))
    hist = LC._hist(LC.rel_lum(lin).reshape(-1), np.ones(100))
    hues = [dict(h=200.0, c=0.05, L=0.6, share=0.2, score=2.0)]
    assert LC.choose_shot(hist, None, hues, 4.5, need=0.85) is None and LC.choose_shot(hist, None, hues, 3.0, need=0.85) is None
    ch, info = LC.pick_ink(hist, hues, [])
    assert ch is not None and info["rung"] >= 2 and not info["skin"] and info["need"] < 0.85


def test_front_words_are_cleared_over_their_whole_appearance_by_the_full_type_alpha():
    """first box render of the rule: the subject solid over the FORMING letters, the whole word snapped in front at the hit (a 27-level frame jump = a false cut). A front word is in front over its span
    (forming + hold + exit) and the legible op is shaped by the full type alpha (the echo copy too), not only the crisp front copy."""
    chain = {"ops": [
        {"op": "mask_gate", "out": "_wake_back", "args": {"a": "$hl0", "gate": 0.4}},
        {"op": "over", "out": "_wake_lay", "args": {"base": "$footage", "alpha": "$_wake_back", "fg": [0.5, 0.2, 0.3]}},
        {"op": "over", "out": "image_t", "args": {"base": "$_wake_lay", "fg": "$tface", "alpha": "$hl0_front"}},
        {"op": "over", "out": "image", "args": {"base": "$image_t", "fg": "$footage", "alpha": "$matte"}}]}
    piece = {"fps": 24, "compose": {"size": [540, 960], "default": chain}}
    holds, spans = [(2.0, 3.0), (6.0, 7.0)], [(1.0, 4.0), (5.0, 8.0)]
    rep = OM.apply_behind(piece, holds, [1], 24, spans=spans)
    legs = [o for o in piece["compose"]["default"]["ops"] if o["op"] == "legible"]
    assert rep["chains"] == 1 and len(legs) == 2
    lf, lb = legs
    assert lf["args"]["holds"] == [[1.0, 4.0]] and lf["args"]["type_a"] == "$hl0" and lf["args"]["floor"] == OM.FRONT_FLOOR      # front word 0: its whole span, the full alpha
    assert lb["args"]["holds"] == [[6.0, 7.0]] and lb["args"]["floor"] == OM.BEHIND_FLOOR                                          # behind word 1: its hold, the readable rule
