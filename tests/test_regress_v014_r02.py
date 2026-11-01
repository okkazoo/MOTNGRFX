"""Three regressions of the draft piece 22_batch-24gb/v014 r02 ('Concrete Command', Gemma design, 6 shots), each with the data of that run (tests/fixtures/v014_r02_concept.json = the concept,
v014_r02_design_reply.txt = the design reply the model returned):
  1. every word overflowed the frame by ~1 letter, alternating sides (type camera `follow`: ink at ~0.89 of the frame width + a composed hold 9 % off centre + a 6 % push-in);
  2. all six stills / shots were the same medium shot (the framing sat at the END of a prompt that began with the subject description; 4 camera entries for 6 shots);
  3. a half-transparent second figure at the 3D cuts (a shot between two cuts lost the matte patch of the first: the shot's own matte over the transformed picture).
CPU only. Run: python -m pytest tests/test_regress_v014_r02.py -q"""
from __future__ import annotations

import copy
import json
import os
import re
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

from test_fusion_plan import synth_music  # noqa: E402
from tools.fusion import angle_still as AS  # noqa: E402
from tools.fusion import concept as CO  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import pivot_cut as PC  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import shot_camera as SC  # noqa: E402
from tools.fusion import type_frame as TF  # noqa: E402

FX = HERE / "fixtures"
CONCEPT = json.loads((FX / "v014_r02_concept.json").read_text(encoding="utf-8"))
REPLY = (FX / "v014_r02_design_reply.txt").read_text(encoding="utf-8")


def request(quality):
    return PR.normalize_request({"words": "\n".join(CONCEPT["words"]), "scene": CONCEPT["scene"], "music": CONCEPT["music"], "length": 16, "look": CONCEPT["look"], "seed": 602,
                                 "quality": quality, "format": "9:16", "shots": CONCEPT["shots"], "owners": CO.request_extras(CONCEPT)})


@pytest.fixture(scope="module")
def music():
    return synth_music(92.0, 16.0, 0.2)


def _plan(quality, music):
    return PR.plan_piece(request(quality), music)


@pytest.fixture(scope="module")
def plans(music):
    return {q: _plan(q, music) for q in ("draft", "preview")}


@pytest.fixture(scope="module")
def designed(plans, music):
    """the model's design reply parsed into the final piece, at both qualities (the type camera of the reply / system is `follow`)."""
    out = {}
    for q, plan in plans.items():
        r = DS.parse(REPLY, "", plan, music, "", "Gemma")
        out[q] = json.loads(r["piece_json"])
    return out


# ------------------------------------------------------------------------------------------------------------------------ 1. text inside the frame
@pytest.mark.parametrize("quality", ["draft", "preview"])
def test_every_word_stays_inside_the_frame_over_its_whole_hold(designed, quality):
    piece = designed[quality]
    size = tuple(piece["sizes"]["type"])
    ext = TF.word_extents(piece["type"], size)
    assert [e["text"] for e in ext] == ["BREAK", "CLAIM", "STRIKE", "OWN"]
    bad = TF.overflow(ext)
    assert not bad, f"{quality}: words leave the safe area: {bad}"
    for e in ext:                                                       # the full glyph box, every sampled hold frame, both axes
        assert np.nanmin(e["x0"]) >= TF.SAFE_MARGIN - 1e-6 and np.nanmax(e["x1"]) <= 1 - TF.SAFE_MARGIN + 1e-6, e["text"]
        assert np.nanmin(e["y0"]) >= TF.SAFE_MARGIN - 1e-6 and np.nanmax(e["y1"]) <= 1 - TF.SAFE_MARGIN + 1e-6, e["text"]


def test_the_cause_is_the_composed_hold_camera_and_the_guard_replaces_it(plans):
    """with the camera `follow` the same design DOES overflow (the regression); the guard measures it and falls back to beat_orbit."""
    piece = copy.deepcopy(plans["draft"])
    piece["type"]["camera"] = "follow"
    piece["type"].pop("camera_params", None)
    piece["type"]["words"][0]["placement"].update(wf=0.9, hf=0.4, yaw=10.5)
    size = tuple(piece["sizes"]["type"])
    assert TF.overflow(TF.word_extents(piece["type"], size)), "follow no longer overflows: the regression test lost its subject"
    notes = TF.guard(piece, size=size)
    assert piece["type"]["camera"] == "beat_orbit" and notes and "follow" in notes[0]
    assert not TF.overflow(TF.word_extents(piece["type"], size))


def test_a_designed_move_other_than_the_composed_holds_is_never_swapped(plans):
    piece = copy.deepcopy(plans["draft"])
    piece["type"]["camera"] = "beat_orbit"
    assert TF.guard(piece, size=tuple(piece["sizes"]["type"]), only_risky=True) == [] and piece["type"]["camera"] == "beat_orbit"


# ------------------------------------------------------------------------------------------------------------------------ 2. a different camera in every shot
FRAMING_LEADS = [v for v in AS.STILL_LEADS.values()]


def _lead_of(prompt):
    return next((v for v in FRAMING_LEADS if prompt.lower().startswith(v.lower())), None)


@pytest.mark.parametrize("quality", ["draft", "preview"])
def test_every_still_and_shot_prompt_leads_with_its_own_framing(plans, quality):
    piece = plans[quality]
    assert len(piece["shots"]) == 6 and len(piece["stills"]) == 6
    leads = [_lead_of(s["prompt"]) for s in piece["stills"]]
    assert all(leads), [s["prompt"][:60] for s in piece["stills"]]
    assert len(set(leads)) == 6, f"the six stills share framings: {leads}"
    for s in piece["stills"]:                                           # the shared scene text no longer carries a contradicting framing
        assert not re.search(r"mid-to-wide|medium-wide", AS.strip_lead(s["prompt"]), re.I), s["prompt"]
    shot_leads = [next((v for v in SC.LEADS.values() if sh["prompt"].startswith(v)), None) for sh in piece["shots"]]
    assert all(shot_leads) and len(set(shot_leads)) == 6, shot_leads
    frs = [SC.parse(sh["prompt"])[0] for sh in piece["shots"]]
    assert len(set(frs)) == 6, frs
    moves = [SC.parse(sh["prompt"])[1] for sh in piece["shots"]]
    assert all(a != b for a, b in zip(moves, moves[1:])), moves


@pytest.mark.parametrize("n_shots,n_entries", [(4, 4), (5, 3), (6, 4), (6, 0), (8, 2)])
def test_distinct_pads_and_enforces_a_framing_per_shot(n_shots, n_entries):
    base = [{"prompt": "the dancer holds a pose in the empty pool at dusk", "camera": "medium, slow push in"} for _ in range(n_entries)]     # every entry the SAME camera
    ents, fixes = SC.distinct(base, n_shots)
    assert len(ents) == n_shots
    frs = [SC.parse(e["camera"])[0] for e in ents]
    assert len(set(frs)) == n_shots, frs
    mv = [SC.parse(e["camera"])[1] for e in ents]
    assert all(a != b for a, b in zip(mv, mv[1:]))


def test_lead_framing_moves_the_framing_to_the_front_and_keeps_the_subject_first_for_the_edit_prompt():
    p = "athletic woman, charcoal grey jacket, An empty concrete swimming pool; mid-to-wide shots., soft pastel colour negative photograph"
    led = AS.lead_framing(p, "close-up")
    assert led.startswith(AS.STILL_LEADS["close-up"]) and "mid-to-wide" not in led
    assert AS.split_still_prompt(led)["subject"] == "athletic woman"            # the anchored angle edit still quotes the subject, not the lead
    assert AS.lead_framing(led, "wide").startswith(AS.STILL_LEADS["wide"]) and AS.STILL_LEADS["close-up"] not in AS.lead_framing(led, "wide")


# ------------------------------------------------------------------------------------------------------------------------ 3. a matte only over its own picture
H, W = 24, 16


def _clip(sid, n, v):
    return {"sid": sid, "u8": np.full((n, H, W, 3), v, np.uint8)}


def test_a_shot_between_two_cuts_keeps_the_matte_of_BOTH_transitions():
    s2 = _clip("s2", 100, 10)
    own = np.full((100, H, W), 0.5, np.float32)                          # the shot's own matte: the subject standing still
    first = (0, np.full((43, H, W), 0.0, np.float32))                    # cut 1 (s1 -> s2): s2 is B, frames 0-43 show the transition
    second = (36, np.full((27, H, W), 1.0, np.float32))                  # cut 2 (s2 -> s3): s2 is A, frames 36-63 (7 shared with cut 1)
    PC.add_matte_patch(s2, first)
    PC.add_matte_patch(s2, second)
    assert len(PC.clip_patches(s2)) == 2 and s2["matte_patch"] is second
    m = PC.patched_mattes([s2], {"s2": own})["s2"]
    assert np.allclose(m[0:36], 0.0)                                      # the first transition's matte (the old code left 0.5 here: the shot's own matte over the chunk picture)
    assert np.allclose(m[36:63], 1.0)                                     # the second one, also over the frames both cuts share
    assert np.allclose(m[63:], 0.5)                                       # outside both windows the shot's own matte, untouched
    assert np.allclose(own, 0.5)                                          # the input is not modified in place


def test_patched_mattes_leaves_clips_without_a_cut_alone_and_the_patch_survives_a_reference_geometry():
    a = _clip("s1", 10, 0)
    ref = np.full((10, 12, 9), 0.25, np.float32)                          # the matte is a cover-resized reference, smaller than the footage
    assert PC.patched_mattes([a], {"s1": ref})["s1"] is ref
    PC.add_matte_patch(a, (2, np.ones((3, H, W), np.float32)))
    out = PC.patched_mattes([a], {"s1": ref})["s1"]
    assert out.shape == ref.shape and out[2:5].mean() > 0.9 and np.allclose(out[:2], 0.25) and np.allclose(out[5:], 0.25)


def test_the_second_cut_renders_from_the_matte_of_the_picture_it_receives():
    """the node feeds cut 2 the mattes with cut 1's patch already in (the clip it gets is the spliced one)."""
    s2 = _clip("s2", 100, 10)
    own = np.full((100, H, W), 0.5, np.float32)
    PC.add_matte_patch(s2, (0, np.zeros((43, H, W), np.float32)))
    seen = PC.patch_matte_all(own, PC.clip_patches(s2))
    assert np.allclose(seen[:43], 0.0) and np.allclose(seen[43:], 0.5)


def test_the_planned_cuts_of_the_run_do_overlap_so_the_patches_must_stack(designed):
    """the real plan: cut 1's window on s2 [67, 110) and cut 2's window on s2 [103, 130) share frames, so a single patch per shot cannot be right."""
    p = designed["draft"]
    wins = [PC.plan_cut(p, sp) for sp in PC.specs(p)]
    ab = {(w["a"], w["b"]): w for w in wins}
    assert ab[("s1", "s2")]["window_b"][1] > ab[("s2", "s3")]["window_a"][0]


# ------------------------------------------------------------------------------------------------------------------------ 0. draft is the REAL single-stage LTX tier; there is no stand-in footage
@pytest.mark.parametrize("quality", ["preview", "draft", "final"])
def test_no_quality_takes_a_still_or_stand_in_path_by_itself(quality):
    """v014 / v015 (24 GB tier, quality draft): MographFootageGather asked for NO clip, so six Ken Burns moves over the stills were composed: static identical framing, no LTX, no
    motion for the Alpha-Gen pass. The stand-in is gone (2026-10-30); the only footage without LTX is the STILL PREVIEW, and only the app quality `still` (a flag on the piece) asks for it."""
    from tools.fusion import still_preview as SP
    assert SP.is_still({"quality": quality}) is False and SP.is_still({"quality": quality}, {"quality": quality}) is False
    assert SP.is_still({"quality": quality, "still_preview": True}) is True and SP.is_still({"quality": quality}, {"still_preview": True}) is True
    assert SP.split_quality(quality) == ((SP.APP_FINAL_TIER if quality == "final" else quality), False) and SP.split_quality("still") == ("draft", True)


def test_the_gather_nodes_do_not_shortcut_on_quality_draft():
    for fname in ("footage.py", "layers.py"):
        src = (ROOT / "nodes_fusion" / fname).read_text(encoding="utf-8")
        code = [l for l in src.splitlines() if not l.strip().startswith("#")]
        assert not any(re.search(r"_is_draft|_is_standin|standin", l) for l in code), fname


@pytest.mark.parametrize("quality,control,typ,compose", [("preview", [384, 704], [384, 704], [360, 640]), ("draft", [576, 1024], [576, 1024], [540, 960])])
def test_draft_and_preview_plans_carry_their_real_sizes(plans, quality, control, typ, compose):
    s = plans[quality]["sizes"]
    assert (s["control"], s["type"], s["compose"]) == (control, typ, compose) and s["shot"] == typ
