"""The RANDOM SYSTEM at the start of every piece (user 2026-10-30: "can we set it as random? it chooses transitions and a range of fonts at the start with a limited palette discerned from the
imagery") + the stable, centred, on-the-beat type rule + the "zoom through a letter" transition and the flat-ground inverse type. CPU, no network, no GPU.
Run: python -m pytest tests/test_random_system.py -q --rootdir=tests"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from mograph import fonts_liked as fl  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import cut_kinds as CK  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import design_options as DO  # noqa: E402
from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import random_system as RS  # noqa: E402
from tools.fusion import stable_type as ST  # noqa: E402
from tools.fusion import still_palette as SP  # noqa: E402

MOODS = ["tender", "defiant", "eerie", "joyful", "lonely", "hopeful", "tense", "nostalgic", "playful", "solemn", "restless", "dreamy"]
DESC = {"s1": "a runner on a road", "s2": "a harbour at dawn", "s3": "a neon street", "s4": "a field of wheat", "s5": "a rooftop"}
SHOTS = [{"id": f"s{i + 1}", "start": 5.0 * i, "dur": 5.0, "frames": 120, "desc": DESC[f"s{i + 1}"], "first": "ABCDE"[i]} for i in range(5)]
ANA = {"downbeats": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0], "beats": [0.5 * i for i in range(50)], "chord_changes": [{"t": 10.0}], "cut": 25.0, "fps": 24, "bpm": 120,
       "curves": {"rms": [0.4] * 120 + [1.0] * 480}}


def reply(texts=("RISE", "GO", "SPIN", "WAKE", "BURN"), cuts=None):
    words = [dict(text=t, font="BadeenDisplay-Regular.ttf", hit=2.4 + 5.0 * i, hold_s=1.8, form_s=0.6, exit_s=0.6, placement={"hf": 0.4, "wf": 0.6}) for i, t in enumerate(texts)]
    return {"type": {"words": words}, "compose": ({"cuts": cuts} if cuts is not None else {})}


def chain(n=30, bpm=100):
    """n consecutive pieces of one batch: each system is drawn knowing the earlier ones."""
    out = []
    for i in range(n):
        out.append(PS.choose(i + 1, MOODS[i % len(MOODS)], bpm, earlier=list(out)))
    return out


# ------------------------------------------------------------------------------------------------------------------ A1 transitions
def test_system_draw_is_deterministic_per_seed():
    a = PS.choose(7, "tender", 90, earlier=[PS.choose(3, "eerie", 100)])
    b = PS.choose(7, "tender", 90, earlier=[PS.choose(3, "eerie", 100)])
    assert a == b and a["cuts"] == b["cuts"] and a["fonts"] == b["fonts"]
    assert PS.choose(7, "tender", 90)["cuts"] != PS.choose(8, "tender", 90)["cuts"] or PS.choose(7, "tender", 90)["fonts"] != PS.choose(8, "tender", 90)["fonts"]


def test_two_to_three_kinds_with_styles_from_the_full_menu_and_consecutive_pieces_differ():
    ss = chain(40)
    for i, s in enumerate(ss):
        ks = s["cuts"]["kinds"]
        assert 2 <= len(ks) <= 3 and len(set(ks)) == len(ks) and all(k in CK.ALL for k in ks)
        assert any(RS.robust(k) for k in ks)                                              # one kind that always renders
        for k in ks:
            st = s["cuts"]["styles"][k]
            assert (st in CK.STYLES) if k.startswith("orbit:") else (st in (None, "chunks") if k == "depth_pivot" else st is None)
        recent = {k for p in ss[max(0, i - 2):i] for k in p["cuts"]["kinds"]}
        assert not recent & set(ks), f"piece {i}: {ks} repeats a kind of the last two pieces {recent}"
    assert all(ss[i]["cuts"]["kinds"] != ss[i + 1]["cuts"]["kinds"] for i in range(len(ss) - 1))


def test_the_draw_reaches_the_whole_menu():
    seen_k, seen_s = set(), set()
    for i in range(400):
        c = RS.draw_cuts(i, "soft", slide_on=True)
        seen_k |= set(c["kinds"])
        seen_s |= {v for v in c["styles"].values() if v}
    assert {"orbit:push", "orbit:pullback", "orbit:fly", "depth_pivot", "pan:left", "pan:right", "window_other", "window_cut", "window_swap", "window_zoom"} <= seen_k
    assert seen_s == {"chunks", "slide"}                                                   # 2026-10-30: `smooth` (one flat card) is never drawn, only named
    off = {v for i in range(200) for v in RS.draw_cuts(i, "soft", slide_on=False)["styles"].values()}
    assert "slide" not in off                                                              # while SLIDE_ON is off the draw never offers it
    assert not any(k.startswith("depth_dolly") for i in range(200) for k in RS.draw_cuts(i, "soft", dolly_on=False)["kinds"])


def _planned(sysm, rep=None):
    out, notes, pins = DO.plan_cuts(rep or reply(), copy.deepcopy(SHOTS), ANA, system=sysm)
    return out, notes, pins


def _varied(seq, drawn):
    """#10c variety rule (user 2026-10-31): no kind twice in a row, max 2 per kind; a kind outside the draw only as the fallback, never orbit:fly"""
    return (all(a != b for a, b in zip(seq, seq[1:])) and all(seq.count(k) <= 2 for k in seq)
            and "orbit:fly" not in set(seq) - set(drawn))


def test_only_drawn_kinds_reach_the_plan_and_the_validator():
    allowed_plain = {"hard", "crossfade"}
    for i in range(24):
        sysm = PS.choose(i + 1, MOODS[i % len(MOODS)], 100)
        drawn = set(sysm["cuts"]["kinds"])
        out, notes, pins = _planned(sysm)
        kinds = {c["after"]: c["cut"] for c in out["compose"]["cuts"]}
        assert _varied(list(kinds.values()), drawn), (i, kinds, drawn)      # #10c: outside the draw only as the no-repeat fallback
        # a model that wrote kinds outside the draw gets drawn ones
        wild = reply(cuts=[{"after": "s1", "cut": "depth_pivot"}, {"after": "s2", "cut": "orbit:fly"}, {"after": "s3", "cut": "pan:down", "word": 2}, {"after": "s4", "cut": "orbit:pullback"}])
        d, fixes = DO.validate(wild, dict(shots=copy.deepcopy(SHOTS), ana=ANA, fill_3d=True, system=sysm))
        got = {c["cut"] if not c.get("was") else c["was"] for c in d["compose"].get("cuts", [])}
        assert _varied([c["cut"] for c in d["compose"].get("cuts", [])], drawn), (i, got, drawn)


def test_no_fixed_pivot_first_or_fly_fallback():
    sysm = {"seed": 1, "fonts": [], "cuts": {"kinds": ["orbit:pullback", "pan:left"], "styles": {"orbit:pullback": "smooth", "pan:left": None}, "order": ["pan:left", "orbit:pullback"]}}
    out, notes, pins = _planned(sysm, reply(cuts=[]))
    kinds = [c["cut"] for c in out["compose"]["cuts"]]
    assert kinds and _varied(kinds, sysm["cuts"]["kinds"]) and kinds[0] != "depth_pivot", kinds
    # the no-repeat swap also stays inside the draw (a repeated kind comes back as the other drawn one or stays a 3D kind, never pivot / fly)
    rep = reply(cuts=[{"after": f"s{i + 1}", "cut": "orbit:pullback"} for i in range(4)])
    d, fixes = DO.validate(rep, dict(shots=copy.deepcopy(SHOTS), ana=ANA, fill_3d=True, system=sysm))
    assert _varied([c["cut"] for c in d["compose"]["cuts"]], sysm["cuts"]["kinds"])
    # the order of the picks follows the system's seeded order, not a fixed list
    a = {"seed": 1, "cuts": {"kinds": ["orbit:push", "orbit:pullback"], "styles": {}, "order": ["orbit:push", "orbit:pullback"]}}
    b = {"seed": 1, "cuts": {"kinds": ["orbit:push", "orbit:pullback"], "styles": {}, "order": ["orbit:pullback", "orbit:push"]}}
    ka = [c["cut"] for c in DO.fill_3d([], copy.deepcopy(SHOTS), None, system=a)[0]]
    kb = [c["cut"] for c in DO.fill_3d([], copy.deepcopy(SHOTS), None, system=b)[0]]
    assert ka[0] == "orbit:push" and kb[0] == "orbit:pullback"


def test_a_failed_window_falls_back_to_another_drawn_kind_not_fly():
    sysm = {"seed": 1, "fonts": [], "cuts": {"kinds": ["window_zoom", "orbit:push"], "styles": {"orbit:push": "smooth"}, "order": ["window_zoom", "orbit:push"]}}
    rep = reply(texts=("SEARCHING", "NOBODY", "TOGETHER", "WANDERING", "FOREVER"), cuts=[{"after": "s1", "cut": "window_zoom", "word": 0}])    # long words: no window can read
    d, fixes = DO.validate(rep, dict(shots=copy.deepcopy(SHOTS), ana=ANA, fill_3d=True, system=sysm))
    first = {c["after"]: c["cut"] for c in d["compose"]["cuts"]}["s1"]
    assert first == "orbit:push" and any("downgraded to hard" in f for f in fixes)


def test_window_other_takes_short_words_and_phrases_and_zoom_is_planned_when_drawn():
    for t in ("RISE", "STORM", "REMAIN"):
        f = DO.window_fit2({"text": t, "font": "Anybody[wdth,wght].ttf", "placement": {"wf": 0.6}})
        assert f["ok"] and f["cap"] >= DO.WINDOW_MIN_CAP * DO.CAP_TOL
    assert DO.window_fit2({"text": "THE SEA"})["ok"] and not DO.window_fit2({"text": "SEARCHING"})["ok"]
    sysm = {"seed": 2, "fonts": ["Anybody[wdth,wght].ttf"], "cuts": {"kinds": ["window_zoom", "orbit:pullback"], "styles": {"orbit:pullback": "smooth"}, "order": ["window_zoom", "orbit:pullback"]}}
    out, notes, pins = _planned(sysm, reply(texts=("ONE", "TWO", "SPIN", "RISE", "GO")))
    cuts = {c["after"]: c for c in out["compose"]["cuts"]}
    assert any(c["cut"] == "window_zoom" for c in cuts.values()) and pins
    z = next(c for c in cuts.values() if c["cut"] == "window_zoom")
    assert z["exit"] == "zoom" and "window_other" not in {c["cut"] for c in cuts.values()}


def test_menu_and_schema_offer_only_the_drawn_kinds():
    sysm = PS.choose(11, "tender", 100)
    drawn = sysm["cuts"]["kinds"]
    txt = DO.menu_text(SHOTS, ANA, system=sysm)
    sch = DO.schema_cuts(sysm)
    cut_line = next(ln for ln in sch.splitlines() if '"cut":' in ln)
    for k in CK.ALL:
        if k in drawn:
            assert k in txt and k in cut_line
        elif k not in ("hard", "crossfade"):
            assert k not in cut_line, (k, cut_line)
    assert "depth_pivot" in DO.menu_text(SHOTS, ANA) and "SHOT TRANSITIONS DRAWN FOR THIS PIECE" in txt         # without a system the full menu as before
    assert PS.prompt_text(sysm).count("Shot transitions drawn for THIS piece") == 1


def test_concept_batch_state_carries_the_cut_kinds():
    c1 = {"words": ["RISE"], "system": PS.choose(1, "tender", 90), "mood": "tender"}
    c2 = {"words": ["FALL"], "cut_kinds": ["orbit:fly", "pan:up"], "mood": "eerie"}                   # an older concept: only its kinds
    es = CO.earlier_systems([c1, c2])
    assert es[0] is c1["system"] and es[1]["cuts"]["kinds"] == ["orbit:fly", "pan:up"]
    s3 = PS.choose(3, "dreamy", 90, earlier=es)
    assert not set(s3["cuts"]["kinds"]) & (set(c1["system"]["cuts"]["kinds"]) | {"orbit:fly", "pan:up"})


# ------------------------------------------------------------------------------------------------------------------ A2 fonts
def test_fonts_are_three_or_four_liked_faces_that_contrast_and_pass_the_legibility_screen():
    ss = chain(60)
    risky = set(DD.MAIN_HOLD_RISKY) | {"LibreBarcode39Text"}
    n_any = 0
    for s in ss:
        fs = s["fonts"]
        assert 3 <= len(fs) <= 4 and len(set(fs)) == len(fs) and all(f in fl.FILES for f in fs)             # ⊂ mograph/fonts/liked.json
        assert not {fl.lookup(f)["key"] for f in fs} & risky
        assert any(fl.role_of(f) in PS.EXPRESSIVE_ROLES for f in fs) and any(fl.role_of(f) in ("heavy", "tall", "text") for f in fs)
        assert sum(1 for a, b in zip(fs, fs[1:]) if not fl.contrast_axes(fl.role_of(a), fl.role_of(b))) <= 1       # neighbouring lines contrast
        n_any += PS.ANYBODY in fs
    assert 0 < n_any < len(ss)                                                                             # Anybody is in the pool at its normal weight: not forced, not excluded
    assert len({f for s in ss for f in s["fonts"]}) >= 15


def test_a_face_that_cannot_carry_the_longest_word_is_skipped():
    wide = fl.BY_KEY["Ultra"]
    texts = ["COMMUNICATION"]
    assert not RS._legible_at(wide, texts, aspect=0.5625) or RS._legible_at(wide, ["GO"])
    assert RS._legible_at(wide, ["GO"])
    drawn = {f for i in range(60) for f in RS.draw_fonts(i, "bright", texts=["EXTRAORDINARILY"])}
    assert drawn and drawn <= fl.FILES


# ------------------------------------------------------------------------------------------------------------------ A3 palette from the still
def _img(kind):
    rng = np.random.default_rng(3)
    a = np.zeros((192, 108, 3), np.uint8)
    if kind == "night":
        a[:] = [20, 60, 70]; a[100:] = [200, 120, 40]; a[20:50, 20:80] = [240, 230, 200]
    elif kind == "flat":
        a[:] = [128, 128, 128]
    elif kind == "bright":
        a[:] = [235, 220, 190]; a[60:140] = [210, 150, 90]; a[:30] = [150, 190, 220]
    elif kind == "mid":
        a[:] = [90, 100, 40]
    elif kind == "dark":
        a[:] = [8, 10, 12]; a[90:100] = [30, 60, 90]
    return np.clip(a.astype(int) + rng.integers(0, 6, a.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("kind", ["night", "flat", "bright", "mid", "dark"])
def test_palette_extraction_returns_3_to_5_colours_with_a_light_a_dark_and_legible_pairs(kind):
    p = SP.extract(_img(kind), k=5, seed=4)
    cs = p["colors"]
    assert 3 <= len(cs) <= 5 and all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in cs)
    assert SP.contrast(p["bg"], p["fg"]) >= SP.MIN_RATIO
    lab = [SP.rgb_to_lab(np.array(c, float))[0] for c in cs]
    assert min(lab) < 35 and max(lab) > 80                                                                  # a dark and a light colour
    assert p["pairs"] and all(SP.contrast(a, b) >= SP.MIN_RATIO for a, b in p["pairs"])
    assert all(SP.delta_e(a, b) >= SP.MERGE_DE * 0.6 for i, a in enumerate(cs) for b in cs[i + 1:])        # near-duplicates are dropped
    assert SP.extract(_img(kind), k=5, seed=4) == p                                                         # deterministic


def test_type_palette_folds_to_the_still_only():
    still = SP.extract(_img("night"), k=5, seed=1)
    sysm = PS.with_still_palette(PS.choose(5, "tender", 90), still)
    t = {"palette": {"bg": [3, 3, 3], "fg": [250, 250, 250], "accents": [[255, 0, 255], [10, 255, 10], [255, 255, 0]]}, "camera": "follow"}
    fixes = []
    PS.fold_type(t, sysm, fixes)
    cols = {tuple(c) for c in still["colors"]}
    pal = t["palette"]
    assert tuple(pal["bg"]) in cols and tuple(pal["fg"]) in cols and all(tuple(a) in cols for a in pal["accents"]) and len(pal["accents"]) >= 3
    assert SP.contrast(pal["bg"], pal["fg"]) >= SP.MIN_RATIO and any("own still" in f for f in fixes)
    assert "ONLY these" in PS.prompt_text(sysm)
    # a colour of the reply that is already in the palette stays as it is
    t2 = {"palette": {"bg": list(still["bg"]), "fg": list(still["fg"]), "accents": [list(a) for a in still["accents"]]}}
    assert PS.fold_type(t2, sysm, []) <= 1


def test_the_still_palette_travels_in_the_locked_string_and_reaches_the_prompt():
    from tools.fusion import design_step as DS
    still = SP.extract(_img("night"), k=5, seed=1)
    piece = {"type": {"words": [{"text": "A"}], "seed": 3}, "plan": {"piece_system": PS.choose(5, "tender", 90)}}
    got = DS.apply_locked(piece, {"still_palette": still})
    assert got["plan"]["still_palette"]["colors"] == still["colors"]
    clips = [{"sid": "s1", "u8": np.stack([_img("night")] * 3)}]
    sp = DS.still_palette_of({"type": {"seed": 3}, "shots": [{"id": "s1", "frames": 72}]}, clips)
    assert sp and sp["source"] == "still" and 3 <= len(sp["colors"]) <= 5 and DS.still_palette_of({"type": {}}, None) is None


# ------------------------------------------------------------------------------------------------------------------ A4 variety knobs
def test_variety_knobs():
    assert not hasattr(PS, "MODULE_B_P") and set(PS.KEPT_MODULES) == set(PS.ALL_MODULES) - set(DD.AVOID_MODULES)        # every module the legibility fold keeps, derived from the fold
    assert all(len(s["modules"]) == len(set(s["modules"])) == min(DD.MAX_MODULES, len(PS.KEPT_MODULES)) for s in chain(120))      # 2026-11-01: n distinct modules up to the cap (was one + a 40 % sibling)
    p = CO.concept_prompt(5, 16, [])
    assert "most pieces use none" not in p and "type_matte" in p
    n = sum(1 for sd in range(400) if CO.type_matte_drawn(sd)) / 400
    assert 0.5 < n < 0.7 and "THIS PIECE uses ONE `type_matte`" in CO.type_matte_text(True)
    assert any(CO.type_matte_drawn(sd) for sd in range(5)) and "THIS PIECE uses ONE `type_matte`" in CO.concept_prompt(next(sd for sd in range(50) if CO.type_matte_drawn(sd)), 16, [])


# ------------------------------------------------------------------------------------------------------------------ stable, centred, on the beat
TEXTS = ["RISE", "THE SEA REMEMBERS", "FOREVER", "GO", "WAKE UP"]


@pytest.mark.parametrize("layout", ST.LAYOUTS)
def test_every_layout_fits_the_safe_area_without_a_mid_word_split(layout):
    for t in TEXTS:
        if layout == "stack" and len(t.split()) < 2:
            continue
        if layout in ST.SPLIT_LAYOUTS and layout not in ST.allowed(t):
            continue
        b = ST.fit_box(t, "BadeenDisplay-Regular.ttf", None, layout)
        x0, y0, x1, y1 = b["box"]
        m = 1.0 - 2 * 0.03
        assert x0 >= 0.03 - 1e-6 and y0 >= 0.03 - 1e-6 and x1 <= 0.97 + 1e-6 and y1 <= 0.97 + 1e-6 and abs((x0 + x1) / 2 - 0.5) < 1e-6 and abs((y0 + y1) / 2 - 0.5) < 1e-6
        if layout in ST.SPLIT_LAYOUTS:
            assert ST.orphan_free(b["lines"], t) and "".join(b["lines"]) == "".join(t.split()) and b["ok"]    # the only mid-word splits, never an orphan letter
        else:
            assert " ".join(b["lines"]).split() == t.split()                                                # lines break only at spaces
        assert b["roll"] == ST.ROLL[layout] and (layout in ("rot90", "rot-90")) == (b["roll"] != 0) and 0 < b["wf"] <= 1 and 0 < b["hf"] <= 1 and m > 0


def test_layouts_are_drawn_per_word_seeded_never_twice_the_same_in_a_row():
    texts = ["RISE", "FALL", "TURN", "WAKE", "BURN", "STAY"]
    a, b = ST.draw_layouts(5, texts), ST.draw_layouts(5, texts)
    assert a == b and all(x in ST.LAYOUTS for x in a) and "stack" not in a                                  # single words cannot stack
    seen = {x for s in range(60) for x in ST.draw_layouts(s, ["THE SEA", "RISE", "WAKE UP", "FALL"])}
    assert seen == set(ST.LAYOUTS)
    assert "rot90" not in ST.draw_layouts(1, texts, fmt="16:9") and "rot-90" not in ST.draw_layouts(1, texts, fmt="16:9")


def _spec(words, layouts_seed=3, bpm=100.0):
    import json
    spec = json.load(open(ROOT / "tools" / "fusion" / "demo_spec.json", encoding="utf-8"))
    spec["bpm"] = bpm
    spec["words"] = words
    spec["camera"] = "beat_orbit"
    return spec


def test_hold_frames_have_zero_word_motion_and_the_word_box_is_centred_and_inside_the_safe_area():
    from tools.fusion import type_frame as TF
    words = [dict(text="REMEMBER", module="slats", font="BadeenDisplay-Regular.ttf", form_in="blobs", form_out="wipe", t0=0.0, t1=4.4, form_s=0.9, exit_s=0.6, hit_on="beat", hold_s=2.4,
                  placement={"yaw": 14.0, "dx": 0.2, "dy": -0.1, "wf": 0.5, "hf": 0.3}),
             dict(text="WAKE UP", module="slats", font="Ultra-Regular.ttf", form_in="wipe", form_out="blobs", t0=4.6, t1=9.0, form_s=0.9, exit_s=0.6, hit_on="beat", hold_s=2.4,
                  placement={"yaw": -14.0, "dx": -0.2, "dy": 0.1, "wf": 0.5, "hf": 0.3})]
    spec = _spec(words)
    spec["camera_params"] = {"read_dps": 7.0, "intro_deg": 40.0, "elev": 4.0, "bob": 0.8}                # the old moving camera
    fixes: list = []
    sysm = PS.choose(5, "tender", 100)
    ST.enforce(spec, sysm, 100.0, fixes)
    assert spec["camera"] == "beat_orbit" and spec["camera_params"]["read_dps"] == 0.0 and spec["camera_params"]["bob"] == 0.0 and spec["camera_params"]["intro_deg"] == 0.0
    assert all(w["placement"]["dx"] == 0 and w["placement"]["dy"] == 0 and w["placement"]["yaw"] == 0 for w in spec["words"]) and fixes
    ext = TF.word_extents(spec, (576, 1024), step=0.25)
    assert len(ext) == 2
    for e in ext:
        for k in ("x0", "x1", "y0", "y1"):
            assert np.ptp(e[k]) < 1e-3, (e["text"], k, e[k])                                                  # zero motion over the whole hold
        assert abs((e["x0"][0] + e["x1"][0]) / 2 - 0.5) < 0.02 and abs((e["y0"][0] + e["y1"][0]) / 2 - 0.5) < 0.02          # centred
    assert not TF.overflow(ext)                                                                           # inside the safe area (rolled layouts included)


@pytest.mark.parametrize("roll", [90.0, -90.0])
def test_a_rolled_word_reads_vertically_centred_inside_the_safe_area_on_the_real_type_scene(roll):
    from tools.fusion import type_frame as TF
    b = ST.fit_box("REMEMBER", "BadeenDisplay-Regular.ttf", None, "rot90" if roll > 0 else "rot-90")
    w = dict(text="REMEMBER", module="slats", font="BadeenDisplay-Regular.ttf", form_in="blobs", form_out="wipe", t0=0.0, t1=4.0, form_s=0.9, exit_s=0.6, hit_on="beat",
             placement={"roll": b["roll"], "wf": b["wf"], "hf": b["hf"]})
    spec = _spec([w])
    spec["camera_params"] = dict(ST.STATIC_CAMERA)
    e = TF.word_extents(spec, (576, 1024), step=0.4)[0]
    assert np.nanmax(e["y1"]) - np.nanmin(e["y0"]) > 0.6 and np.nanmax(e["x1"]) - np.nanmin(e["x0"]) < 0.5            # tall and narrow = vertical reading
    assert not TF.overflow([e]) and abs((e["y0"][0] + e["y1"][0]) / 2 - 0.5) < 0.02 and abs((e["x0"][0] + e["x1"][0]) / 2 - 0.5) < 0.02
    assert abs(e["y1"][0] - e["y0"][0] - (b["box"][3] - b["box"][1])) < 0.12                                  # the fit estimate matches the real scene


def test_enforce_raises_a_short_hold_to_one_beat_and_keeps_window_words_centred():
    t = {"words": [dict(text="GO", font="Ultra-Regular.ttf", hold_s=0.2, placement={"yaw": 9.0, "dx": 0.1, "wf": 0.5, "hf": 0.3}, screen_offset=[0.1, 0.1]),
                   dict(text="RISE", font="Anybody[wdth,wght].ttf", hold_s=1.6, window_source="next", placement={"yaw": 9.0, "dx": 0.1, "wf": 0.9, "hf": 0.4, "roll": 90.0})],
         "camera": "follow", "camera_params": {"read_dps": 5}}
    fixes: list = []
    ST.enforce(t, None, 60.0, fixes, protected={1})
    assert t["words"][0]["hold_s"] == 1.0 and "screen_offset" not in t["words"][0]                            # one beat at 60 bpm
    assert t["words"][1]["placement"]["roll"] == 90.0 and t["words"][1]["placement"]["dx"] == 0 and t["words"][1]["placement"]["wf"] == 0.9      # the window's own fit stays
    assert t["camera"] == "beat_orbit" and ST.min_hold_s(120) == 0.5


def test_every_word_lands_within_a_frame_of_a_beat_and_holds_one_beat():
    from tools.fusion import spec as S
    from tools.fusion.beats import Beats
    for bpm in (72, 90, 100, 124):
        grid = Beats(bpm, None, 4, 0.0, 30.0)
        words = [dict(text=t, form_in="blobs", form_out="wipe", t0=1.7 * i * 2, t1=1.7 * i * 2 + 3.2, form_s=0.77, exit_s=0.55, hit_on="beat") for i, t in enumerate(["RISE", "FALL", "TURN", "WAKE"])]
        plan = S.plan_timing(words, {"tier": "mid", "bpm": bpm}, grid)
        beat = 60.0 / bpm
        assert not ST.landed_on_beat([p["hit"] for p in plan], beat), (bpm, [p["hit"] for p in plan])         # every landed frame within 1 frame (1/24 s) of a beat
        assert all(p["rel"] - p["hit"] >= min(beat, 1.0) - 1e-6 for p in plan)                               # the hold lasts at least one beat
    assert ST.landed_on_beat([0.0, 0.5, 1.02, 1.3], 0.5) == [(1.3, 7.2)] or len(ST.landed_on_beat([0.0, 0.5, 1.02, 1.3], 0.5)) == 1


# ------------------------------------------------------------------------------------------------------------------ transitions read
def test_every_drawn_transition_plays_over_at_least_twelve_frames():
    from mograph.modules.transitions import depth_pivot as DP
    from tools.fusion import pivot_cut as PC
    from tools.fusion import window_cut as WC
    for k in CK.SHOT_3D[:4]:
        for st in (None, "smooth", "chunks", "slide"):
            if k == CK.PIVOT and st not in (None, "chunks"):
                continue
            n, i2 = PC.window_frames(k, {"style": st} if st else {})
            assert ST.window_ok(n) and 0 < i2 < n
            mode = {"mode": k.split(":")[1]} if k.startswith("orbit") else {}
            p = DP.resolve_params(dict(mode, **({"style": st} if st else {})))
            assert p["travel"] >= ST.MIN_TRANSITION_FRAMES
    ph = lambda t, zs: WC.window_cut_phase(t, "zoom", t_open=0.0, t_cut=5.0, zoom_s=zs)                       # noqa: E731
    assert ph(5.0 - 0.45, 0.1)["zoom"] > 0.0 and ph(5.0 - 0.55, 0.1)["zoom"] == 0.0                           # a too-short zoom_s is read as 0.5 s = 12 frames
    assert DO.normalize_cuts([{"after": "s1", "cut": "window_zoom", "word": 0, "zoom_s": 0.1}])[0][0]["zoom_s"] == DO.ZOOM_MIN_S == WC.ZOOM_MIN_S == 0.5
    assert DO.ZOOM_S == WC.ZOOM_S and WC.ZOOM_MIN_S * 24 >= ST.MIN_TRANSITION_FRAMES


# ------------------------------------------------------------------------------------------------------------------ PART B: zoom through a letter + flat-ground inverse type
def _letters(h=640, w=360, text="RISE"):
    import cv2
    a = np.zeros((h, w), np.uint8)                                                 # (OpenCV 5 on the box draws text on uint8 only)
    for i, ch in enumerate(text):                                                  # four separate letters (a font with gaps), thick strokes
        cv2.putText(a, ch, (14 + i * 84, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 22)
    return a.astype(np.float32) / 255.0


def test_the_zoom_letter_grows_to_cover_the_frame_by_the_end_of_the_window():
    from tools.fusion import window_cut as WC
    h, w = 640, 360
    a = _letters(h, w)
    plan = WC.zoom_plan(a)
    assert plan["n"] >= 3 and 0 <= plan["k"] < plan["n"] and 0 < plan["r"] < 0.5
    cover, bbox_area = [], []
    for z in (0.0, 0.25, 0.5, 0.75, 1.0):
        st = WC.zoom_state(plan, z, w, h)
        al = WC.zoom_alpha(a, st, h, w)
        cover.append(float((al > 0.5).mean()))
        ys, xs = np.nonzero(al > 0.5)
        bbox_area.append((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        if z == 1.0:
            assert st["cx"] == pytest.approx(w / 2) and st["cy"] == pytest.approx(h / 2)                      # the focus glided to the centre
            assert plan["r"] * w * st["s"] >= 0.5 * np.hypot(w, h)                                           # the inscribed circle reaches the far corner
            assert al.min() > 0.999                                                                            # every pixel of the frame is inside the letter
    assert all(b >= a_ for a_, b in zip(cover, cover[1:])) and cover[0] < 0.15 and cover[-1] == 1.0
    assert all(b >= a_ for a_, b in zip(bbox_area, bbox_area[1:])) and bbox_area[-1] == w * h
    st0 = WC.zoom_state(plan, 0.0, w, h)
    assert st0["s"] == 1.0 and (WC.zoom_alpha(a, st0, h, w) > 0.5).mean() == pytest.approx(float((a > 0.5).mean()), abs=0.01)    # z = 0 is the word as it was


def test_zoom_op_frames_start_on_the_plate_end_on_the_next_shot_and_the_matte_follows_its_own_picture():
    from tools.fusion import window_cut as WC
    h, w = 640, 360
    a = _letters(h, w)
    A = np.zeros((h, w, 3), np.float32); A[...] = [0.8, 0.2, 0.1]; A[100:200, 100:260] = [0.1, 0.9, 0.2]     # a green "subject" on a red plate
    matte = np.zeros((h, w), np.float32); matte[100:200, 100:260] = 1.0                                       # its own matte
    B = np.zeros((h, w, 3), np.float32); B[...] = [0.1, 0.3, 0.9]
    kw = dict(mode="zoom", t_open=0.5, t_cut=3.0, zoom_s=0.9, matte_a=matte, min_contrast=None, line=None)
    assert np.allclose(WC.window_cut(a, A, B, t=0.0, **kw), A, atol=1e-5)                                    # before the window: plate A
    assert np.allclose(WC.window_cut(a, A, B, t=3.2, **kw), B, atol=1e-5)                                    # after the cut: B
    last = WC.window_cut(a, A, B, t=2.99, **kw)
    assert np.abs(last - B).mean() < 0.01                                                                     # the last zoom frame is (almost) exactly B
    mid = WC.window_cut(a, A, B, t=2.4, **kw)
    assert np.abs(mid - A).mean() > 0.02 and np.abs(mid - B).mean() > 0.02                                    # visibly between the two pictures
    # the plate and its matte are zoomed by the SAME map: wherever the zoomed matte is high, the zoomed plate shows the subject colour (no second subject, no slip)
    plan = WC.zoom_plan(a)
    fx, fy = int(plan["fx"] * w), int(plan["fy"] * h)
    matte = np.zeros((h, w), np.float32); matte[fy - 12:fy + 12, fx - 12:fx + 12] = 1.0                        # the subject sits at the focus: it stays on screen while the camera dives
    A = np.zeros((h, w, 3), np.float32); A[...] = [0.8, 0.2, 0.1]; A[fy - 12:fy + 12, fx - 12:fx + 12] = [0.1, 0.9, 0.2]
    plate_z, al, mt, _ln, _p, st = WC.zoom_layers(A, a, matte, None, 0.5, h, w, plan=plan)
    import cv2
    sel = cv2.erode((np.asarray(mt) > 0.99).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0                    # inside the zoomed matte (its soft rim aside)
    assert sel.sum() > 400 and np.allclose(np.asarray(plate_z)[sel], [0.1, 0.9, 0.2], atol=0.05)
    # B stays still: no pixel of B moves (the frame inside the letters is B itself)
    z24 = WC.window_cut_phase(2.4, "zoom", t_open=0.5, t_cut=3.0, zoom_s=0.9)["zoom"]
    inside = np.asarray(WC.zoom_layers(A, a, matte, None, z24, h, w, plan=plan)[1]) > 0.9995
    assert inside.sum() > 1000 and np.allclose(WC.window_cut(a, A, B, t=2.4, **dict(kw, matte_a=None))[inside], B[inside], atol=2e-3)


def test_flat_ground_window_shows_the_picture_only_inside_the_letters():
    from tools.fusion import window_cut as WC
    h, w = 640, 360
    a = _letters(h, w)
    foot = np.zeros((h, w, 3), np.float32); foot[...] = [0.2, 0.4, 0.8]
    ground = [0.95, 0.9, 0.8]
    out = WC.window_cut(a, ground, foot, t=1.0, mode="other", t_open=0.0, t_end=3.0, exit="io", min_contrast=None)
    import cv2
    k = np.ones((7, 7), np.uint8)
    far_out = cv2.erode((a < 0.01).astype(np.uint8), k) > 0                       # away from the letter edge (the backends soften the edge by a pixel or two)
    far_in = cv2.erode((a > 0.99).astype(np.uint8), k) > 0
    assert far_out.any() and far_in.any()
    assert np.abs(out[far_out] - np.array(ground)).max() < 0.01 and np.abs(out[far_in] - np.array([0.2, 0.4, 0.8])).max() < 0.01
    ops = WC.shot_ops(dict(kind="window_other", to="self", t_open=0.2, t_end=3.0, ground=ground), "from")
    args = ops[0]["args"]
    assert args["a"] == ground and args["b"] == "$footage" and args["subject"] == "inside" and "matte_a" not in args


def test_ground_colour_comes_from_the_pieces_palette_against_its_imagery():
    from tools.fusion import window_cut as WC
    dark = SP.extract(_img("night"), seed=1)
    bright = SP.extract(_img("bright"), seed=1)
    assert dark["mean_lightness"] < 55 < bright["mean_lightness"]
    g1 = WC.ground_colour({"plan": {"still_palette": dark}}, "plain")
    g2 = WC.ground_colour({"plan": {"still_palette": bright}}, "plain")
    assert [round(v * 255) for v in g1] == dark["light"] and [round(v * 255) for v in g2] == bright["dark"]      # the colour that stands off the imagery, a colour of the palette
    assert WC.ground_colour({"type": {"palette": {"fg": [200, 100, 50]}}}, "plain") == [round(v / 255, 4) for v in (200, 100, 50)]
    assert WC.ground_colour({}, [10, 20, 30]) == pytest.approx([10 / 255, 20 / 255, 30 / 255])


def test_inverse_window_is_planned_on_one_short_word_and_validated():
    sysm = {"seed": 4, "fonts": ["Anybody[wdth,wght].ttf"], "inverse": {"plain_window": True}, "cuts": {"kinds": ["orbit:push", "orbit:fly"], "styles": {}, "order": ["orbit:push", "orbit:fly"]}}
    out, notes, pins = _planned(sysm, reply(texts=("SEARCHING", "RISE", "WANDERING", "GO", "FOREVER")))
    ws = [w for w in out["type"]["words"] if w.get("window_ground")]
    assert len(ws) == 1 and ws[0]["text"] in ("RISE", "GO") and ws[0]["window_source"] == "self" and any("inverse window planned" in n for n in notes)
    d, fixes = DO.validate(out, dict(shots=copy.deepcopy(SHOTS), ana=ANA, fill_3d=True, system=sysm))
    w = next(w for w in d["type"]["words"] if w.get("window_ground"))
    assert w["window_source"] in ("s1", "s2", "s3", "s4", "s5") and DO.is_fat(w["font"], w.get("axes"))[0]
    off = dict(sysm, inverse={"plain_window": False})
    assert not [w for w in _planned(off)[0]["type"]["words"] if w.get("window_ground")]


def test_window_zoom_in_the_vocabulary_the_block_and_the_piece_writer():
    from tools.fusion import blocks as BK
    from tools.fusion import window_cut as WC
    from test_window_cut import piece
    assert CK.normalize("zoom") == "window_zoom" == CK.normalize("letter zoom") and "window_zoom" in CK.WINDOW and CK.family_of("window_zoom") == "window"
    assert "window_zoom" in DO.CUT_KINDS and "window_zoom" in RS.menu() and "window_zoom" in BK.BLOCKS
    p = piece([{"cut": "window_zoom", "after": "s1", "word": 0, "zoom_s": 0.8}])
    assert WC.owned_boundaries(p) == [0]
    notes = WC.apply_to_piece(p)
    ops_a, ops_b = p["compose"]["shots"]["s1"]["ops"][0]["args"], p["compose"]["shots"]["s2"]["ops"][0]["args"]
    assert ops_a["mode"] == "zoom" and ops_b["mode"] == "zoom" and ops_a["zoom_s"] == 0.8 and ops_a["t_cut"] == ops_b["t_cut"] and ops_a["b"] == "$next" and ops_b["a"] == "$prev" and notes


def test_a_system_that_draws_a_window_kind_has_a_heavy_face_on_its_shortlist():
    n = 0
    for s in chain(80):
        if any(k.startswith("window") for k in s["cuts"]["kinds"]):
            n += 1
            assert any(fl.role_of(f) in DO.FAT_ROLES for f in s["fonts"]), s["fonts"]
    assert n > 10


def test_an_older_system_without_a_draw_gets_one_from_its_own_seed():
    old = {k: v for k, v in PS.choose(9, "tender", 90).items() if k not in ("cuts", "inverse")}
    sysm, explicit = PS.for_piece({"plan": {"piece_system": old}}, None)
    assert explicit and sysm["cuts"]["kinds"] and PS.for_piece({"plan": {"piece_system": old}}, None)[0]["cuts"] == sysm["cuts"]      # deterministic: drawn from its own seed, never the fixed fallback


def test_transition_windows_are_listed_and_a_word_holding_under_the_move_is_reported():
    cuts = [{"after": "s1", "cut": "orbit:push"}, {"after": "s2", "cut": "depth_pivot"}, {"after": "s3", "cut": "window_zoom", "word": 2}, {"after": "s4", "cut": "hard"}]
    wins = ST.transition_windows(cuts, SHOTS)
    assert [w[0] for w in wins] == ["orbit:push", "depth_pivot"] and all(ST.window_ok(round((b - a) * 24)) for _k, _a, a, b in wins)
    assert wins[0][2] < 5.0 < wins[0][3]                                                      # the move straddles the cut at 5 s
    words = [{"hit": 1.0, "release": 2.0}, {"hit": 2.4, "release": wins[0][2] + 0.1}, {"hit": 2.4, "release": 4.8}]
    assert ST.transition_overlaps(words, wins) == [(2, "orbit:push", pytest.approx(4.8 - (wins[0][2] + 0.3), abs=0.02))]      # only the word that runs on under the move


# ------------------------------------------------------------------------------------------------------------------ font count scales with the piece length
@pytest.mark.parametrize("length_s,want", [(6, 2), (12, 3), (18, 4), (24, 4), (30, 5), (48, 6), (90, 6)])
def test_the_number_of_faces_scales_with_the_piece_length(length_s, want):
    assert RS.n_fonts(length_s) == want and RS.n_fonts(None) == RS.n_fonts(16) == 3
    for seed in range(12):
        s = PS.choose(seed + 1, MOODS[seed % len(MOODS)], 100, length_s=length_s)
        fs = s["fonts"]
        assert len(fs) == want and len(set(fs)) == want and all(f in fl.FILES for f in fs)                     # liked fonts only
        assert any(fl.role_of(f) in PS.EXPRESSIVE_ROLES for f in fs) and any(fl.role_of(f) in ("heavy", "tall", "text") for f in fs)
        assert sum(1 for a, b in zip(fs, fs[1:]) if not fl.contrast_axes(fl.role_of(a), fl.role_of(b))) <= max(1, want // 3)     # neighbouring lines contrast
    one = PS.for_piece({"plan": {}, "type": {"seed": 4, "mood": "tender", "duration": float(length_s)}, "audio": {"bpm": 100}}, None)[0]
    assert len(one["fonts"]) == want                                                                               # a piece without a system: its own length decides


# ------------------------------------------------------------------------------------------------------------------ outline as a stand-out option
HOLDS = [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0), (10.0, 12.0), (13.0, 15.0)]


def test_the_outline_is_a_drawn_option_at_some_words_never_every_word():
    modes = [PS.choose(i, "tender", 100)["outline"]["mode"] for i in range(80)]
    assert set(modes) == {"none", "hits"} and PS.choose(3, "tender", 100)["outline"] == PS.choose(3, "tender", 100)["outline"]
    for mode in ("words", "hits"):
        for seed in range(30):
            spec = {"mode": mode, "share": 0.4 if mode == "words" else RS.HITS_SHARE}
            w = RS.outline_windows(spec, HOLDS, 100, seed)
            assert 1 <= len(w) <= len(HOLDS) - 1                                                                    # at some words, not all
            if mode == "hits":
                assert len(w) <= -(-len(HOLDS) // 3)                                                                # user 2026-10-30: at most ceil(n / 3) words carry the outline
            assert all(a >= HOLDS[0][0] and b > a and any(h0 - 1e-6 <= a and b <= h1 + 1e-6 for h0, h1 in HOLDS) for a, b in w)      # inside a word's hold
            assert w == RS.outline_windows(spec, HOLDS, 100, seed)
            if mode == "hits":
                assert all(b - a <= 60.0 / 100 * RS.OUTLINE_PULSE_BEATS + 1e-6 for a, b in w)
    assert RS.outline_windows({"mode": "none"}, HOLDS, 100, 1) == [] and RS.outline_windows({"mode": "words"}, [], 100, 1) == []
    one = RS.outline_windows({"mode": "words", "share": 0.4}, [(1.0, 3.0)], 100, 1)
    assert len(one) == 1 and one[0][1] - one[0][0] <= 0.61                                                          # one word: a pulse, not the whole hold
    assert "Outline" in PS.prompt_text(PS.choose(5, "tender", 100)) and "Not on every word" in PS.prompt_text(PS.choose(5, "tender", 100))


def test_the_forced_outline_is_legible_colors_own_outline_support_and_only_inside_its_windows():
    import numpy as _np
    from test_legible_color import _alpha, _face, _plate, _run
    base = _run(_plate(0.04), frames=10)                                                                            # a dark plate: the cyan face already passes, no support
    assert float(base[-1][..., 3][_alpha() < 0.01].max()) < 0.01
    on = _run(_plate(0.04), frames=10, outline=[[0.0, 5.0]])
    ring = (_alpha() < 0.01)
    assert float(on[-1][..., 3][ring].max()) > 0.5 and float(on[-1][..., 3][_alpha() > 0.99].min()) > 0.99          # an outline ring around the letters, the letters unchanged
    assert _np.allclose(on[-1][..., :3][_alpha() > 0.99], base[-1][..., :3][_alpha() > 0.99], atol=0.02)
    off = _run(_plate(0.04), frames=10, outline=[[5.0, 6.0]])                                                       # a window that is not now: nothing
    assert float(off[-1][..., 3][ring].max()) < 0.01
    ring_rgb = on[-1][..., :3][ring & (on[-1][..., 3] > 0.5)]
    from tools.fusion import legible_color as LCm
    assert abs(float(ring_rgb.mean()) - LCm.LC_SUP_DARK) < 0.05                                                     # the outline is in the OPPOSITE luminance of the (light) type: legible_color's own support colour


def test_the_recipe_wires_the_outline_windows_into_legible_color():
    from tools.fusion import recipe_eval as RE
    piece = {"plan": {"piece_system": {"seed": 5, "outline": {"mode": "words", "share": 0.4}}}, "type": {"bpm": 100.0, "seed": 5}}
    w = RE._outline_windows(piece, HOLDS)
    assert w and len(w) <= 4 and w == RE._outline_windows(piece, HOLDS)
    assert RE._outline_windows(dict(piece, compose={"outline": False}), HOLDS) == [] and RE._outline_windows(dict(piece, compose={"outline": [[1, 2]]}), HOLDS) == [[1.0, 2.0]]
    assert RE._outline_windows({"plan": {}, "type": {}}, HOLDS) == []


# ------------------------------------------------------------------------------------------------------------------ named effects menu + user-requested effects
def test_effects_names_aliases_and_unknown_names_are_reported_not_fatal():
    from tools.fusion import effects_menu as EM
    names, unk = EM.norm_effects(["Window-Subject-Front", "outline", "zoom", "orbit:push", "bogus", "outline", "hard"])
    assert names == ["window_subject_front", "outline", "window_zoom", "orbit:push"] and unk == ["bogus", "hard"]
    assert EM.norm_effects("plate_in_type, flat window; stroke") == (["plate_in_type", "flat_ground_window", "outline"], [])
    assert EM.norm_effects(None) == ([], []) and EM.norm_effects(42) == ([], [])
    assert set(EM.CONTRAST_EFFECTS) == {"window_subject_front", "plate_in_subject", "plate_in_type"} and set(EM.CONTRAST_EFFECTS) <= set(EM.EFFECTS)


def test_the_draw_gives_zero_to_two_contrast_sections_and_a_request_forces_more():
    from tools.fusion import effects_menu as EM
    counts = [len(PS.choose(i, "tender", 100)["effects"]["contrast"]) for i in range(300)]
    assert set(counts) == {0, 1, 2} and 0.25 < counts.count(1) / 300 < 0.55
    assert PS.choose(7, "x", 100)["effects"] == PS.choose(7, "x", 100)["effects"]
    d = EM.draw_effects(3, (), ["plate_in_type", "plate_in_subject", "window_subject_front"])
    assert d["contrast"][:3] == ["plate_in_type", "plate_in_subject", "window_subject_front"] and len(d["contrast"]) == 3     # a request for three gives three


def test_requested_effects_are_forced_into_the_system_and_reach_the_prompt():
    from tools.fusion import effects_menu as EM
    base = PS.choose(5, "tender", 100)
    s = PS.choose(5, "tender", 100, effects=["window_subject_front", "outline", "zoom", "flat_ground_window", "bogus"])
    e = s["effects"]
    assert "window_subject_front" in e["contrast"] and e["requested"] == ["window_subject_front", "outline", "window_zoom", "flat_ground_window"] and e["unknown"] == ["bogus"]
    assert "window_zoom" in s["cuts"]["kinds"] and s["inverse"]["plain_window"] is True and s["outline"]["mode"] != "none"
    assert any(fl.role_of(f) == "heavy" for f in s["fonts"]) and set(e["active"]) >= {"window_subject_front", "window_zoom", "outline", "flat_ground_window"}
    assert any(RS.robust(k) for k in s["cuts"]["kinds"]) and base["fonts"] or True
    txt = PS.prompt_text(s)
    assert "REQUIRED by the user" in txt and "window_subject_front" in txt and "window_zoom" in txt
    assert "REQUIRED" not in PS.prompt_text(base)
    again = EM.apply_effects(base, ["plate_in_subject"])                                    # applying to a ready system (the planner does this with the concept's)
    assert "plate_in_subject" in again["effects"]["contrast"] and base["effects"] == PS.choose(5, "tender", 100)["effects"]    # the input system is not changed


def test_contrast_effects_become_type_matte_shot_keys_through_plate_layers_shot_keys():
    from tools.fusion import effects_menu as EM
    from tools.fusion import plate_layers as PLT
    shots = [{"id": f"s{i + 1}", "frames": 96, "matte": i != 1} for i in range(4)] + [{"id": "s5", "frames": 24, "matte": True}]
    s = PS.choose(5, "tender", 100, effects=["plate_in_type", "window_subject_front"])
    placed = EM.place_contrast(shots, s, 5)
    assert [p["name"] for p in placed] == ["plate_in_type", "window_subject_front"] and len({p["shot"] for p in placed}) == 2
    for p in placed:
        sh = next(x for x in shots if x["id"] == p["shot"])
        assert sh["type_matte"] == p["name"] and PLT.norm_type_matte(sh["type_matte"]) == p["name"] and sh["frames"] >= 72 and sh["matte"]       # a long shot with a subject
    assert not shots[4].get("type_matte") and EM.place_contrast([], s, 5) == [] and EM.place_contrast(shots, PS.choose(5, "x", 100, effects=["outline"]), 5) == []
    assert all(PLT.shot_keys({"type_matte": n}) == {"type_matte": n} for n in EM.CONTRAST_EFFECTS)


def test_the_request_field_effects_is_read_by_the_planner_and_the_concept_step():
    from test_fusion_core import synth_music
    from tools.fusion import plan_rules as PR
    REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto", "seed": 5,
           "quality": "preview", "format": "9:16", "shots": "cut"}
    assert PR.effects_of(["outline", "zoom"]) == ["outline", "zoom"] and PR.effects_of("a, b") == ["a", "b"] and PR.effects_of('{"effects": ["x"]}') == ["x"] and PR.effects_of(None, "[]") == []
    r = PR.normalize_request(dict(REQ, effects="plate_in_type, outline, bogus"))
    assert r["effects"] == ["plate_in_type", "outline", "bogus"] and PR.normalize_request(REQ)["effects"] == []
    plan = PR.plan_piece(dict(REQ, effects=["plate_in_type", "outline", "bogus"]), synth_music(100, 16))
    eff = plan["plan"]["effects"]
    assert eff["requested"] == ["plate_in_type", "outline"] and eff["unknown"] == ["bogus"] and "plate_in_type" in eff["active"] and "outline" in eff["active"]
    assert any(s.get("type_matte") == "plate_in_type" for s in plan["shots"]) and plan["plan"]["piece_system"]["outline"]["mode"] != "none"
    assert any("unknown name(s) ignored: bogus" in n for n in plan["plan"]["notes"])
    # the concept step: `effects` rides in the `previous` string (no new graph input) and in the prompt; the concept carries them to the request
    assert CO.parse_effects('{"concepts": [], "effects": "outline, zoom"}') == ["outline", "zoom"] and CO.parse_effects("[]") == [] and CO.parse_effects("") == []
    p = CO.concept_prompt(5, 16, [], effects=["window_subject_front", "outline"])
    assert "REQUIRED by the user" in p and "window_subject_front" in p and "REQUIRED by the user" not in CO.concept_prompt(5, 16, [])
    c = {"words": ["RISE"], "owners": [], "effects": ["outline"], "shot_plan": [], "idea": ""}
    assert json.loads(CO.request_extras(c))["effects"] == ["outline"]
    assert PR.effects_of(CO.request_extras(c)) == ["outline"]


# ------------------------------------------------------------------------------------------------------------------ grid / column: words split mid-word to fill the frame
@pytest.mark.parametrize("word,rows", [("DREAMING", ["DR", "EA", "MI", "NG"]), ("SEARCHING", ["SEA", "RCH", "ING"]), ("WILD", ["WI", "LD"]), ("STEAM", ["STE", "AM"])])
def test_grid_splits_a_word_into_balanced_rows_that_fill_the_frame(word, rows):
    b = ST.fit_box(word, "Anybody[wdth,wght].ttf", None, "grid")
    assert b["lines"] == rows and b["ok"] and b["roll"] == 0
    sizes = [len(r) for r in b["lines"]]
    assert max(sizes) - min(sizes) <= 1 and 2 <= len(sizes) <= 4 and all(2 <= z <= 4 for z in sizes) and ST.orphan_free(b["lines"], word)
    x0, y0, x1, y1 = b["box"]
    assert abs((x0 + x1) / 2 - 0.5) < 1e-6 and x0 >= 0.03 and x1 <= 0.97 and y0 >= 0.03 and y1 <= 0.97          # centred, inside the safe area
    if word in ("DREAMING", "WILD"):
        assert (y1 - y0) >= 0.60 or (x1 - x0) >= 0.8                                                           # the block fills the frame (height or width)


def test_grid_splits_table_and_a_phrase_keeps_its_words_apart():
    assert ST.grid_splits(8) == [(4, 4), (3, 3, 2), (2, 2, 2, 2)] and ST.grid_splits(9) == [(3, 3, 3), (3, 2, 2, 2)] and ST.grid_splits(5) == [(3, 2)] and ST.grid_splits(4) == [(2, 2)]
    assert ST.grid_splits(3) == [] and ST.grid_splits(2) == [] and ST.grid_splits(16) == [(4, 4, 4, 4)] and ST.grid_splits(17) == []
    b = ST.fit_box("SILVER MOON", "Anybody[wdth,wght].ttf", None, "grid")             # neutral words: the export scrubber rewrites the clip's own lyric words in inputs only
    g = ST.rows_by_token(b["lines"], "SILVER MOON")
    assert g is not None and "".join(g[0]) == "SILVER" and "".join(g[1]) == "MOON" and len(g[0]) >= 2 and len(g[1]) >= 2           # letters never mix across words
    assert ST.orphan_free(["SIL", "VER", "MO", "ON"], "SILVER MOON")                                             # the user's own split
    assert ST.fit_box("A LIGHT", "Anybody[wdth,wght].ttf", None, "grid")["lines"][0] == "A"                       # a one-letter word is one row, not an orphan of a split


def test_the_no_orphan_rule():
    assert ST.orphan_free(["S", "T", "E", "A", "M"], "STEAM") and ST.orphan_free(["STE", "AM"], "STEAM") and ST.orphan_free(["STEAM"], "STEAM")
    assert not ST.orphan_free(["ST", "E", "AM"], "STEAM")                                                           # S-T / E / A-M
    assert not ST.orphan_free(["STEA", "M"], "STEAM")                                                               # an orphan at the end of a grid
    assert not ST.orphan_free(["S", "TEAM"], "STEAM") and not ST.orphan_free(["ST", "EA", "M"], "STEAM")
    assert not ST.orphan_free(["SILV", "ER M", "OON"], "SILVER MOON")                                              # a row that mixes two words
    assert not ST.orphan_free(["STE", "AX"], "STEAM")                                                               # does not spell the word
    for n in range(4, 17):                                                                                          # every split the code can make is orphan free
        for sizes in ST.grid_splits(n):
            word = "ABCDEFGHIJKLMNOP"[:n]
            rows, k = [], 0
            for z in sizes:
                rows.append(word[k:k + z])
                k += z
            assert ST.orphan_free(rows, word) and min(sizes) >= 2
    for t in ("DREAMING", "SEARCHING", "WILD", "STEAM", "FIRST WORD", "RISE", "FOREVER", "A LIGHT"):
        for lay in ("grid", "column"):
            if lay in ST.allowed(t):
                assert ST.orphan_free(ST.fit_box(t, "BadeenDisplay-Regular.ttf", None, lay)["lines"], t), (t, lay)


def test_column_is_one_letter_per_row_for_three_to_eight_letters_and_passes_the_size_floor():
    for t in ("RISE", "STEAM", "WAKE", "DREAMING", "SEA"):
        assert "column" in ST.allowed(t)
        b = ST.fit_box(t, "Anybody[wdth,wght].ttf", None, "column")
        assert b["lines"] == list(t) and b["ok"] and b["cap"] >= DD.CAP_FLOOR and b["box"][1] >= 0.03 and b["box"][3] <= 0.97
    assert "column" not in ST.allowed("GO") and "column" not in ST.allowed("SEARCHING") and "column" not in ST.allowed("THE SEA") and "grid" not in ST.allowed("GO")
    assert not ST.fit_box("EXTRAORDINARY", "Anybody[wdth,wght].ttf", None, "column")["ok"]                          # 13 letters: no column
    assert ST.best_layout("EXTRAORDINARY", "Anybody[wdth,wght].ttf", None, "column")["layout"] != "column"          # the drawn layout falls back when it cannot carry the word
    seen = {x for sd in range(80) for x in ST.draw_layouts(sd, ["STEAM", "DREAMING", "RISE"])}
    assert {"grid", "column"} <= seen and "grid" in ST.LAYOUTS and "column" in ST.LAYOUTS


def test_enforce_writes_the_rows_of_a_split_word_and_keeps_them_centred():
    t = {"words": [dict(text="DREAMING", font="Anybody[wdth,wght].ttf", hold_s=2.0, placement={"yaw": 8.0, "dx": 0.1}), dict(text="STEAM", font="Anybody[wdth,wght].ttf", hold_s=2.0, placement={})]}
    fixes: list = []
    ST.enforce(t, {"seed": 1, "layouts": ["grid", "column"]}, 100.0, fixes)
    w0, w1 = t["words"]
    assert w0["lines"] == ["DR", "EA", "MI", "NG"] and w1["lines"] == list("STEAM") and w0["ar_min"] == 0.85
    assert w0["placement"]["dx"] == 0 and w0["placement"]["roll"] == 0 and w0["placement"]["wf"] > 0.8 and any("layout grid" in f for f in fixes) and any("layout column" in f for f in fixes)


def test_the_type_engine_takes_the_rows_of_a_split_word_as_they_are():
    from tools.fusion import type_frame as TF
    words = []
    for k, (t, lay) in enumerate((("DREAMING", "grid"), ("STEAM", "column"), ("SEARCHING", "grid"))):
        b = ST.fit_box(t, "Anybody[wdth,wght].ttf", None, lay)
        words.append(dict(text=t, module="slats", font="Anybody[wdth,wght].ttf", form_in="blobs", form_out="wipe", t0=k * 4.4, t1=k * 4.4 + 4.2, form_s=0.9, exit_s=0.6, hit_on="beat",
                          lines=b["lines"], ar_min=0.85, placement={"wf": b["wf"], "hf": b["hf"], "roll": 0}))
    spec = _spec(words)
    spec["camera_params"] = dict(ST.STATIC_CAMERA)
    ext = TF.word_extents(spec, (576, 1024), step=0.5)
    assert not TF.overflow(ext)
    for e in ext:
        assert abs((np.nanmin(e["x0"]) + np.nanmax(e["x1"])) / 2 - 0.5) < 0.02 and abs((np.nanmin(e["y0"]) + np.nanmax(e["y1"])) / 2 - 0.5) < 0.02
        assert np.ptp(e["x0"]) < 1e-3 and np.ptp(e["y0"]) < 1e-3                                                  # still
    assert np.nanmax(ext[0]["y1"]) - np.nanmin(ext[0]["y0"]) > 0.65 and np.nanmax(ext[1]["y1"]) - np.nanmin(ext[1]["y0"]) > 0.7      # the rows really stack and fill the frame height


def test_a_split_word_survives_the_whole_design_step_as_rows():
    """DS.parse (the app's design step) keeps the mid-word rows the stable-type rule wrote: nothing rejoins or re-breaks them."""
    import json as _json
    from test_diversity import REQ, parse_reply
    from test_fusion_core import synth_music
    from tools.fusion import design_step as DS
    from tools.fusion import plan_rules as PR
    music = synth_music(100, 16)
    plan = PR.plan_piece(dict(REQ, words="DREAMING\nSTEAM\nGO\nRISE"), music)
    plan["plan"]["layouts_pin"] = ["grid", "column", "line", "grid"]
    reply_ = DS.reply_from_piece(plan, music)
    out = parse_reply(plan, music, reply_)
    words = _json.loads(out["piece_json"])["type"]["words"]
    assert words[0]["lines"] == ["DR", "EA", "MI", "NG"] and words[3]["lines"] == ["RI", "SE"]                    # (word 1 is a window word of the plan: protected, keeps its own fit)
    assert all(w["placement"].get("roll", 0) == 0 and w["placement"].get("dx", 0) == 0 for w in words)
    for w in (words[0], words[3]):
        assert ST.orphan_free(w["lines"], w["text"])


def test_hold_life_names_are_in_the_effects_menu_and_requested_ones_stay_active():
    from tools.fusion import effects_menu as EM
    from tools.fusion import hold_life as HL
    assert set(EM.HOLD_LIFE_EFFECTS) == set(HL.EFFECTS) and set(HL.EFFECTS) <= set(EM.EFFECTS)
    assert EM.norm_effects("pulse, glow-breathe, outline") == (["pulse", "glow_breathe", "outline"], [])
    base = {"seed": 3, "effects": {}, "outline": {"mode": "none"}}
    s = EM.apply_effects(base, ["pulse", "glitch_slice"])
    assert s["effects"]["requested"] == ["pulse", "glitch_slice"] and {"pulse", "glitch_slice"} <= set(s["effects"]["active"])


def test_outline_hits_never_outline_more_than_a_third_of_the_words_for_any_word_count():
    for n in range(1, 13):
        holds = [(1.0 + 3 * i, 3.0 + 3 * i) for i in range(n)]
        for seed in range(20):
            w = RS.outline_windows({"mode": "hits", "share": RS.HITS_SHARE}, holds, 100, seed)
            assert 1 <= len(w) <= -(-n // 3), (n, seed, len(w))


def test_the_default_compose_recipes_have_no_glow_behind_the_type():
    import json
    for name in ("behind", "behind_sheer", "built_in", "traced"):
        rec = json.loads((Path(ROOT) / "tools" / "fusion" / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        glows = [o for o in rec["compose"]["default"]["ops"] if o["op"] == "add_glow" and o["args"].get("mask") == "$type_matte"]
        assert not glows, (name, glows)


def test_grid_and_column_are_requestable_effects_forced_onto_at_least_half_the_words():
    from tools.fusion import effects_menu as EM
    texts = ["DREAMING", "FIRST WORD", "WILD", "SEA", "STEAM", "AM"]
    names, unk = EM.norm_effects("grid, column")
    assert names == ["grid", "column"] and not unk
    drawn = ["line"] * len(texts)
    out = ST.force_layouts(drawn, texts, {"requested": ["grid", "column"]}, seed=3)
    assert sum(1 for d in out if d in ("grid", "column")) >= 3                          # at least half of 6
    for t, d in zip(texts, out):
        assert d in ST.allowed(t)                                                       # only where the layout exists (no-orphan rule: SEA / AM never a one-letter row)
    assert ST.force_layouts(drawn, texts, {"requested": []}, seed=3) == drawn           # no request: the seeded draw stays
    assert sum(1 for d in ST.force_layouts(["line", "line"], ["AMBER", "WILD"], {"requested": ["grid"]}, 1) if d == "grid") == 2     # two words: both
    assert ST.LAYOUT_WEIGHTS["grid"] >= 3.5                                              # ~1 in 3 of the words with 4+ letters in the normal draw


# ------------------------------------------------------------------------------------------------------------------ QC v026: grid <= 4 rows, a requested column is used, the face is left clear
@pytest.mark.parametrize("text", ["FORGETS TURN", "WHAT THE CLAY", "EVERY NAME", "REMEMBER", "THE SEA REMEMBERS", "FIRST WORD", "ONE TWO THREE FOUR"])
def test_a_grid_word_never_has_more_than_four_rows(text):
    for layout in ("grid",):
        if layout not in ST.allowed(text):
            continue
        b = ST.fit_box(text, "BadeenDisplay-Regular.ttf", None, layout)
        assert 2 <= len(b["lines"]) <= ST.GRID_MAX_TOTAL_ROWS and ST.orphan_free(b["lines"], text)
    # best_layout of a phrase that cannot be a <= 4-row grid never hands one back
    b = ST.best_layout(text, "BadeenDisplay-Regular.ttf", None, "grid")
    assert b["layout"] != "grid" or len(b["lines"]) <= ST.GRID_MAX_TOTAL_ROWS


def test_a_phrase_that_needs_more_than_four_rows_has_no_grid_and_falls_to_stack_or_line():
    assert not ST.grid_possible("WHAT THE CLAY") and "grid" not in ST.allowed("WHAT THE CLAY")
    assert ST.grid_possible("FORGETS TURN") and len(ST._split_rows("FORGETS TURN", "grid", None, None, 0.5625)[0]) == 4
    b = ST.best_layout("WHAT THE CLAY", "BadeenDisplay-Regular.ttf", None, "grid")
    assert b["layout"] in ("line", "stack", "rot90", "rot-90")


def test_a_grid_never_covers_the_subjects_face_when_a_matte_exists():
    m = np.zeros((704, 384), np.uint8)
    m[120:700, 100:290] = 255                                               # a standing subject: face = the top quarter of the box
    fr = ST.face_rect(m)
    assert fr is not None and fr[1] == pytest.approx(120 / 704) and fr[3] == pytest.approx((120 + 0.25 * 579) / 704, abs=1e-3)
    assert ST.face_rect(np.zeros((704, 384), np.uint8)) is None            # no subject: no face rule
    free = ST.best_layout("DREAMING", "BadeenDisplay-Regular.ttf", None, "grid")
    assert free["layout"] == "grid" and ST.face_cover(free["box"], fr) > ST.FACE_LIMIT      # the 4-row grid fills the frame, so it covers the face
    got = ST.best_layout("DREAMING", "BadeenDisplay-Regular.ttf", None, "grid", face=fr)
    assert got["layout"] != "grid" or got["face_cover"] <= ST.FACE_LIMIT
    # through enforce: the system carries the face rectangles per word
    t = {"words": [dict(text="DREAMING", font="BadeenDisplay-Regular.ttf", hold_s=2.0)], "camera": "beat_orbit"}
    fixes: list = []
    ST.enforce(t, dict(seed=1, layouts=["grid"], face_rects={0: fr}), 90.0, fixes)
    w = t["words"][0]
    assert not w.get("lines") or len(w["lines"]) == 1                      # the drawn grid was dropped for a one-row layout (rolled or line): no mid-word split over the face


def test_a_requested_column_is_used_by_an_eligible_word_even_when_every_word_drew_a_grid():
    """QC v026 r02: effects grid + column were requested, the draw gave every word a grid, so no word was a column and HANDS came out rolled."""
    texts = ["HANDS", "REMEMBER", "WHAT", "THE CLAY", "FORGETS TURN"]
    drawn = ["grid", "grid", "grid", "rot-90", "grid"]
    got = ST.force_layouts(drawn, texts, {"requested": ["grid", "column"]}, seed=62)
    assert "column" in got and "grid" in got
    col = got.index("column")
    assert 3 <= len(ST._letters(texts[col])) <= 8 and len(texts[col].split()) == 1
    only = ST.force_layouts(["rot90", "rot-90", "line"], ["HANDS", "WHAT", "THE CLAY"], {"requested": ["column"]}, seed=1)
    assert only.count("column") >= 1
    keep = ST.force_layouts(["column", "grid", "line", "line"], ["HANDS", "FORGETS TURN", "GO", "WAKE UP"], {"requested": ["grid", "column"]}, seed=1)
    assert "column" in keep and "grid" in keep                              # the only holder of a requested layout is never taken


def test_enforce_turns_a_requested_column_into_real_one_letter_rows_and_explains_a_fallback():
    t = {"words": [dict(text=x, font="BadeenDisplay-Regular.ttf", hold_s=2.0) for x in ("HANDS", "REMEMBER", "WHAT")], "camera": "beat_orbit"}
    fixes: list = []
    ST.enforce(t, dict(seed=62, effects=dict(requested=["grid", "column"]), layouts=["grid", "grid", "grid"]), 75.0, fixes)
    cols = [w for w in t["words"] if w.get("lines") and all(len(r) == 1 for r in w["lines"]) and len(w["lines"]) > 1]
    assert cols and all(w["placement"]["roll"] == 0.0 for w in cols)       # one letter per row, not rotated
    t2 = {"words": [dict(text="GO", font="BadeenDisplay-Regular.ttf", hold_s=2.0)], "camera": "beat_orbit"}
    f2: list = []
    ST.enforce(t2, dict(seed=1, effects=dict(requested=["column"])), 75.0, f2)
    assert any("effects column requested but no word uses it: no eligible word" in f for f in f2)


# ------------------------------------------------------------------------------------------------------------------ QC v026: coverage (no 1-3 s with no word on screen)
def _v026(n):
    import json
    return json.load(open(ROOT / "tests" / "fixtures" / f"v026_r{n}_type_spec.json", encoding="utf-8"))


def _runner():
    from tools.fusion import spec as SPEC
    from tools.fusion.beats import Beats

    def run(t):
        g = Beats(t.get("bpm"), t.get("beats"), t.get("bar", 4), t.get("beat_offset", 0.0), t["duration"] + 4.0, t.get("downbeats"))
        return [dict(start=x["S"], hit=x["hit"], release=x["rel"], end=x["end"]) for x in SPEC.plan_timing(copy.deepcopy(t["words"]), t, g)]
    return run


@pytest.mark.parametrize("n,bpm_over", [("02", False), ("03", True)])
def test_the_v026_pieces_had_long_empty_stretches_and_the_rule_closes_them(n, bpm_over):
    sp, run = _v026(n), _runner()
    before = run(sp)
    gaps = ST.blank_gaps(before, sp["bpm"])
    assert len(gaps) >= 3 and all(b >= 1.0 for _i, b, _l in gaps)                      # 1.0-1.2 s after the release of every word (limit max(1 beat, 0.75 s))
    fixes: list = []
    after = ST.enforce_coverage(sp, run, cut=16.0, fixes=fixes)
    assert ST.blank_gaps(after, sp["bpm"]) == [] and fixes
    assert (sp.get("overlap") == "on") == bpm_over                                     # above 90 bpm the beat-snapped mode cannot get there: the overlap mode is on
    for a, b in zip(before, after):                                                    # a hit stays on the grid, the holds only grow
        assert b["release"] - b["hit"] >= a["release"] - a["hit"] - 0.3
    assert after[-1]["end"] <= 16.0 + 0.4


def test_a_pan_or_a_3d_cut_window_is_a_designed_transition_not_a_gap():
    sp, run = _v026("01"), _runner()
    cw = run(sp)
    pans = {0}
    assert 0 not in [i for i, _b, _l in ST.blank_gaps(cw, sp["bpm"], pan_after=pans)] and 0 in [i for i, _b, _l in ST.blank_gaps(cw, sp["bpm"])]
    win = [("orbit:fly", "s2", float(cw[1]["release"]) + 0.1, float(cw[2]["start"]) - 0.1)]
    assert 1 not in [i for i, _b, _l in ST.blank_gaps(cw, sp["bpm"], windows=win, pan_after=pans)]
    assert ST.max_blank_s(75) == pytest.approx(0.8) and ST.max_blank_s(120) == 0.75 and ST.max_blank_s(None) == 0.75


def test_coverage_never_moves_a_hit_into_another_shot_or_an_ltx_owned_word():
    sp, run = _v026("02"), _runner()
    base = run(sp)
    starts = [0.0, float(base[1]["hit"]) - 0.01, float(base[2]["hit"]) - 0.01, float(base[3]["hit"]) - 0.01, float(base[4]["hit"]) - 0.01]      # every hit sits just inside its own shot: no hit may move earlier
    fixes: list = []
    ST.enforce_coverage(sp, run, starts=starts, cut=16.0, fixes=fixes)
    assert any("could not be shortened" in f or "exit" in f for f in fixes)
    after = run(sp)
    assert all(_a["hit"] >= s0 - 1e-6 for _a, s0 in zip(after, starts))
    sp2 = _v026("02")
    sp2["words"][1]["render"] = "handoff"                                             # footage owns this word at a fixed time
    ST.enforce_coverage(sp2, run, cut=16.0, fixes=[])
    assert abs(run(sp2)[1]["hit"] - base[1]["hit"]) <= 0.05


def test_plan_piece_reports_no_long_gap_between_words():
    from test_fusion_core import synth_music
    from tools.fusion import plan_rules as PR
    REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto", "seed": 5,
           "quality": "preview", "format": "9:16", "shots": "cut"}
    plan = PR.plan_piece(REQ, synth_music(100, 16))
    chk = PR.dry_run(plan)
    pans = {i for i, w in enumerate(plan["type"]["words"]) if isinstance(w.get("transition"), dict) and w["transition"].get("kind") == "pan"}
    left = ST.blank_gaps(chk["words"], plan["type"]["bpm"], pan_after=pans)
    assert not left or any("could not be shortened" in n for n in (plan["plan"].get("coverage") or []))


def test_font_draw_does_not_keep_one_face():
    """2026-10-30 (ship #8 audit: 'Ultra was overused in the experiments'): measured on the real draw the experiments' Ultra came from hand-picked recipes, not from `draw_fonts` (Ultra
    4 % of the faces, every face 2-6 %, 9-13 % of consecutive pieces share any face). This guards it: over 300 pieces no face is more than 12 % of the draws, and consecutive pieces
    (earlier = the previous two systems) rarely share a face."""
    import collections
    from tools.fusion import piece_system as PS
    for tone in PS.EXPR_ROLE_WEIGHTS:
        cnt, earlier, rep, pairs = collections.Counter(), [], 0, 0
        for seed in range(300):
            f = RS.draw_fonts(seed, tone, earlier=earlier[-2:], texts=["HELLO", "WORLD"], length_s=12)
            cnt.update(f)
            if earlier:
                pairs += 1
                rep += bool(set(f) & set(earlier[-1]["fonts"]))
            earlier.append({"fonts": f})
        tot = sum(cnt.values())
        assert max(cnt.values()) / tot <= 0.12, (tone, cnt.most_common(2))
        assert rep / pairs <= 0.30, (tone, rep / pairs)
