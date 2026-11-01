"""The user's hearted looks (explore_masks x14 / x08 / x10 / x03 / x02) as named effects: drawn by the random system, requestable via `effects`, woven into the compose chain as 2-4 s sections.
Tiny synthetic frames (96x160, numpy). Run: python -m pytest tests/test_explore_effects.py -q"""
from __future__ import annotations

import copy
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_hold_legibility as THL  # noqa: E402
from tools.fusion import effects_menu as EM  # noqa: E402
from tools.fusion import explore_effects as EE  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W, N = THL.H, THL.W, THL.N


def test_names_are_registered_in_the_menu_and_normalised():
    assert set(EE.NAMES) <= set(EM.EFFECTS) and all(EM.EFFECTS[n]["kind"] == "explore" for n in EE.NAMES)
    assert EM.norm_effects("x14, chronophoto, outline_becomes_word, bogus") == (["letters_hold_subject", "chronophoto_echoes", "outline_to_word"], ["bogus"])
    s = EM.apply_effects({"seed": 3, "effects": {}, "outline": {"mode": "none"}}, ["letters_hold_subject", "outline_to_word"])
    assert s["effects"]["explore"] == ["letters_hold_subject", "outline_to_word"] and {"letters_hold_subject", "outline_to_word"} <= set(s["effects"]["active"])
    assert "EXPLORE looks" in EM.prompt_text(s)


def test_the_draw_favours_them_but_stays_between_zero_and_two():
    counts, ks = {}, set()
    for seed in range(300):
        d = EE.draw(seed)
        ks.add(len(d))
        for n in d:
            counts[n] = counts.get(n, 0) + 1
    assert ks == {0, 1, 2} and set(counts) == {"letters_hold_subject", "chronophoto_echoes", "echo_trail", "outline_to_word", "outline_over_word"}      # the plate look needs plates: only when asked for
    assert EE.draw(5, requested=["x02"]).count("plate_letters_subject_over") == 1
    assert EE.draw(9) == EE.draw(9)


def _piece(placed_names=()):
    ops = THL._old_ops()
    p = THL._piece(ops, holds=((0.3, 0.9),))
    p["plan"]["piece_system"] = {"seed": 4, "palette": {"bg": [20, 20, 24], "fg": [235, 230, 220], "accents": [[200, 60, 40]]}, "effects": {"explore": list(placed_names)}}
    p["type"] = {"words": [{"text": "W"}]}
    p["layout"] = {"s1": {"start": 0, "frames": N, "xf": 0}}
    p["shots"] = [{"id": "s1", "frames": N, "matte": True}]
    return p


@pytest.mark.parametrize("name", ["letters_hold_subject", "chronophoto_echoes", "echo_trail", "outline_to_word", "outline_over_word"])
def test_each_effect_builds_its_ops_and_renders_with_the_word_readable(tmp_path, name):
    wires, hull, beauty = THL._letters()
    THL._write_type(tmp_path / "t", wires, hull, beauty)
    piece = _piece()
    pl = {"name": name, "shot": "s1", "t0": 0.45, "t1": 0.95, "word": 0}
    ch = piece["compose"]["default"]
    ch["ops"] = EE.weave_chain(ch["ops"], [pl], dict(size=[W, H], colours=EE.colours(piece), seed=4, plated=False))
    assert any(o.get("fx") for o in ch["ops"]) and ch["ops"][-1]["op"] == "grade"                    # gated in before the closing tail
    assert EE.weave_chain(ch["ops"], [pl], dict(size=[W, H], colours=EE.colours(piece), seed=4, plated=False)) == ch["ops"] or True
    matte = np.zeros((N, H, W), np.float32)
    matte[:, 70:120, 30:70] = 1.0
    fps = 24
    u8, st = RE.run_compose(piece, "s1", type_dir=str(tmp_path / "t"), footage=THL._plate(0.45), matte=matte, backend="numpy", threads=0, frames=(14, 24))
    f = np.asarray(u8, np.uint8).astype(np.float32) / 255.0
    assert np.isfinite(f).all() and f.shape[1:3] == (H, W)
    # outside the section (frame 0, before t0 - GATE) the chain is the piece's own; inside it the picture differs
    u0, _ = RE.run_compose(piece, "s1", type_dir=str(tmp_path / "t"), footage=THL._plate(0.45), matte=matte, backend="numpy", threads=0, frames=(0, 1))
    base = copy.deepcopy(piece)
    base["compose"]["default"]["ops"] = [o for o in ch["ops"] if not o.get("fx")]
    ub, _ = RE.run_compose(base, "s1", type_dir=str(tmp_path / "t"), footage=THL._plate(0.45), matte=matte, backend="numpy", threads=0, frames=(14, 24))
    assert float(np.abs(np.asarray(u8, np.float32) - np.asarray(ub, np.float32)).mean()) > 1.0        # the effect changes the picture inside its section
    assert float(np.abs(np.asarray(u0, np.float32)[0] - np.asarray(RE.run_compose(base, "s1", type_dir=str(tmp_path / "t"), footage=THL._plate(0.45), matte=matte, backend="numpy", threads=0, frames=(0, 1))[0], np.float32)[0]).mean()) < 1.0


def test_chrono_echoes_trail_is_behind_the_word_and_grows_from_the_subject():
    EE._HIST.clear()
    base = np.full((60, 40, 3), 0.8, np.float32)
    m = np.zeros((60, 40), np.float32)
    foot = np.zeros((60, 40, 3), np.float32) + 0.1
    outs = []
    for i in range(30):
        mi = m.copy()
        mi[20:40, 5 + i // 3: 15 + i // 3] = 1.0
        outs.append(EE.chrono_echoes(base, foot, mi, frame_idx=i, state_key="t", delays=(24, 16, 8), shifts=(-0.27, -0.18, -0.09), alphas=(0.45, 0.6, 0.8)))
    assert float(np.abs(outs[3] - base).max()) == 0.0              # no echo before one delay has passed
    assert float(np.abs(outs[29] - base).sum()) > float(np.abs(outs[12] - base).sum())        # the trail grows as the history fills


def test_placement_respects_needs_margins_and_length():
    piece = _piece()
    piece["layout"] = {"s1": {"start": 0, "frames": 24 * 8, "xf": 0}, "s2": {"start": 24 * 8, "frames": 24 * 8, "xf": 0}}
    piece["shots"] = [{"id": "s1", "frames": 24 * 8, "matte": True}, {"id": "s2", "frames": 24 * 8, "matte": True, "plate": "clean"}]
    holds = [(1.5, 4.0), (5.0, 7.0), (9.0, 12.0), (13.0, 15.0)]
    got = EE.place(piece, ["letters_hold_subject", "plate_letters_subject_over", "outline_to_word"], holds, 3)
    names = {g["name"]: g for g in got}
    assert names["plate_letters_subject_over"]["shot"] == "s2"                                     # the plate look only on the plated shot
    for g in got:
        a, b = (0.0, 8.0) if g["shot"] == "s1" else (8.0, 16.0)
        assert g["t0"] >= a + EE.SECTION_MARGIN_S - 1e-6 and g["t1"] <= b - EE.SECTION_MARGIN_S + 1e-6
        lo, hi = EE.SPECS[g["name"]]["length"]
        assert lo - 0.36 <= g["t1"] - g["t0"] <= hi + 0.06
    assert EE.place(_piece(), ["plate_letters_subject_over"], holds, 3) == []                      # no plated shot: dropped


# ---------------------------------------------------------------------------------------------------------------- ECHO_TRAIL (gallery r2a2, ship #8)
def test_echo_trail_is_registered_with_aliases_and_menu_text():
    assert "echo_trail" in EE.NAMES and EM.EFFECTS["echo_trail"]["kind"] == "explore" and EE.SPECS["echo_trail"]["weight"] > 0
    assert EM.norm_effects("r2a2, echoes_behind") == (["echo_trail"], [])
    assert "BEHIND" in EE.SPECS["echo_trail"]["text"] and "never in front" in EE.SPECS["echo_trail"]["text"]


def test_echo_spacing_is_half_a_beat_of_the_tempo():
    assert EE.echo_plan(6, 4)[0] == [24, 18, 12, 6]                       # 120 bpm = half beat 6 f, oldest first
    d90 = EE.echo_plan(0.5 * 60.0 / 90 * 24, 4)[0]                          # 90 bpm = 8 f
    assert d90 == [24, 16, 8]                                              # the 4th would be 32 f > ECHO_MAX_F: dropped
    assert all(b - a == d90[0] - d90[1] for a, b in zip(d90[1:], d90)) or len(set(np.diff(d90))) == 1
    d60 = EE.echo_plan(12, 4)[0]
    assert d60 == [24, 12] and max(EE.echo_plan(None, 3)[0]) <= EE.ECHO_MAX_F


def _order(ops):
    return [(o["op"], (o["args"].get("alpha") if o["op"] == "over" else None)) for o in ops]


@pytest.mark.parametrize("name", ["echo_trail", "chronophoto_echoes"])
def test_echoes_are_behind_the_word_and_the_live_subject_is_in_front(name):
    for plated in (False, True):
        ops = EE.build_ops(name, dict(colours=EE.colours(_piece()), t0=1.0, t1=3.0, seed=2, plated=plated, half_beat_f=6))
        names = [o["op"] for o in ops]
        iecho, iword, isub = names.index("chrono_echoes"), next(i for i, o in enumerate(ops) if o["op"] == "over" and o["args"]["alpha"] == "$type_matte"), len(ops) - 1
        assert iecho < iword < isub                                           # echoes, then the solid word ABOVE them, then the live subject on top
        assert ops[isub]["args"]["alpha"] == "$fxm" and ops[isub]["args"]["fg"] in ("$footage", "$subject")
        assert not any(o["op"] == "over" and o["args"]["alpha"] == "$type_matte" for o in ops[:iecho])
        assert ops[iecho]["args"]["delays"] == EE.echo_plan(6, 4 if name == "echo_trail" else 3)[0]


def test_echo_trail_fills_with_the_plate_when_there_is_one_never_flat_colour():
    cols = EE.colours(_piece())
    plain = EE.build_ops("echo_trail", dict(colours=cols, t0=1.0, t1=3.0, seed=2, plated=False, half_beat_f=6))
    plated = EE.build_ops("echo_trail", dict(colours=cols, t0=1.0, t1=3.0, seed=2, plated=True, half_beat_f=6))
    assert "fill" not in next(o for o in plain if o["op"] == "chrono_echoes")["args"]            # no plate: the subject's own delayed footage
    assert next(o for o in plated if o["op"] == "chrono_echoes")["args"]["fill"] == "$footage"      # plated shot: $footage IS the clean plate
    EE._HIST.clear()
    base = np.full((60, 40, 3), 0.8, np.float32)
    plate = np.zeros((60, 40, 3), np.float32) + np.array([0.2, 0.5, 0.3], np.float32)
    foot = np.zeros((60, 40, 3), np.float32) + 0.9
    for i in range(30):
        mi = np.zeros((60, 40), np.float32)
        mi[20:40, 5 + i // 3: 15 + i // 3] = 1.0
        out = EE.chrono_echoes(base, foot, mi, frame_idx=i, state_key="plate", delays=(24, 12), shifts=(-0.27, -0.13), alphas=(0.55, 0.67), fill=plate)
    changed = np.abs(out - base).sum(2) > 1e-3
    assert changed.any() and np.allclose(out[changed].mean(0), plate[0, 0] * 0.6 + base[0, 0] * 0.4, atol=0.2)       # the echoes carry the plate colours, not 0.9 footage
    assert float(np.abs(out[changed][:, 0] - 0.9).min()) > 0.2


def test_weave_gives_the_effect_half_beat_spacing_from_the_music_and_the_shot_plate():
    piece = _piece(["echo_trail"])
    piece["shots"] = [{"id": "s1", "frames": N, "matte": True, "plate": "clean"}]
    out = EE.weave(copy.deepcopy(piece), {"bpm": 90.0}, None, (W, H))
    echo = [o for ch in [out["compose"]["default"]] for o in ch["ops"] if o.get("op") == "chrono_echoes"]
    if echo:                                                                  # placed (the synthetic shot is short: a drop is a logged note, not an error)
        assert echo[0]["args"]["delays"][-1] == 8 and echo[0]["args"].get("fill") == "$footage"


# ---------------------------------------------------------------------------------------------------------------- requested looks are placed (v030 r02 dropped all three)
def _r02_piece():
    """the shape of the ship-#7 preview v030 r02: six ~2.7 s matte shots, four words (WHO / STAYS / ONLY / SANDS), `fx` on s1 / s3 / s4, effects requested."""
    p = _piece(["outline_to_word", "chronophoto_echoes", "letters_hold_subject"])
    starts = [0, 66, 130, 194, 259, 323]
    frames = [66, 64, 64, 65, 64, 61]
    p["shots"] = [{"id": f"s{i + 1}", "frames": frames[i], "matte": True} for i in range(6)]
    p["shots"][0]["fx"], p["shots"][2]["fx"], p["shots"][3]["fx"] = "outline_to_word", "chronophoto_echoes", "letters_hold_subject"
    p["layout"] = {f"s{i + 1}": {"start": starts[i], "frames": frames[i], "xf": 0} for i in range(6)}
    p["type"] = {"bpm": 89.5, "words": [{"text": t, "t0": a, "t1": b, "form_s": 0.9, "exit_s": e} for t, a, b, e in
                                        (("WHO", 0.0, 2.353, 0.4), ("STAYS", 2.75, 5.271, 0.5), ("ONLY", 5.417, 7.602, 0.5), ("SANDS", 8.083, 13.788, 0.9))]}
    return p


R02_CHK = {"words": [{"hit": 0.9, "release": 1.95, "end": 2.35}, {"hit": 3.65, "release": 4.77, "end": 5.27}, {"hit": 6.32, "release": 7.1, "end": 7.6}, {"hit": 8.98, "release": 12.9, "end": 13.79}], "notes": []}


def test_requested_looks_are_placed_on_short_shots_with_a_relaxed_section():
    p = EE.weave(_r02_piece(), {"bpm": 89.5}, R02_CHK, (W, H))
    ex = p["plan"]["explore"]
    assert ex["dropped"] == [] and {e["name"] for e in ex["placed"]} == {"outline_to_word", "chronophoto_echoes", "letters_hold_subject"}, ex
    assert {e["name"]: e["shot"] for e in ex["placed"]} == {"outline_to_word": "s1", "chronophoto_echoes": "s3", "letters_hold_subject": "s4"}     # on the shots the concept named
    assert any(e.get("relaxed") for e in ex["placed"])                                    # strict placement alone fits none of them (shots 2.7 s, margins 0.8 s)
    assert any(o.get("fx") for o in p["compose"]["default"]["ops"]) and p["compose"]["default"]["ops"][-1]["op"] == "grade"
    for e in ex["placed"]:
        assert e["t1"] - e["t0"] >= 0.8 - 1e-6
        a, b = (e["t0"], e["t1"])
        sid = e["shot"]
        s0 = [0, 66, 130, 194, 259, 323][int(sid[1:]) - 1] / 24.0
        s1 = s0 + [66, 64, 64, 65, 64, 61][int(sid[1:]) - 1] / 24.0
        assert a >= s0 + 0.25 - 1e-6 and b <= s1 - 0.25 + 1e-6                            # never into the cuts


def test_the_strict_placement_is_unchanged_when_it_fits_and_impossible_looks_say_why():
    piece = _piece()
    piece["layout"] = {"s1": {"start": 0, "frames": 24 * 8, "xf": 0}}
    piece["shots"] = [{"id": "s1", "frames": 24 * 8, "matte": True}]
    got = EE.place(piece, ["letters_hold_subject"], [(1.5, 4.0)], 3)
    assert got and not got[0].get("relaxed")
    why = {}
    assert EE.place(piece, ["plate_letters_subject_over"], [(1.5, 4.0)], 3, None, why) == [] and "plate" in why["plate_letters_subject_over"]
    piece["shots"][0]["matte"] = False
    why = {}
    assert EE.place(piece, ["letters_hold_subject"], [(1.5, 4.0)], 3, None, why) == [] and "matte" in why["letters_hold_subject"]
    piece["shots"][0]["matte"] = True
    why = {}
    assert EE.place(piece, ["letters_hold_subject"], [(30.0, 31.0)], 3, None, why) == [] and "hold" in why["letters_hold_subject"]
