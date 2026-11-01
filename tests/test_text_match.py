"""The rendered text matches the graphic, and the video model never invents its own text (user 2026-10-29: "I want the rendered text to match what the graphic text is - not do two different
versions on top of each other"; "LTX or whatever renderer should never invent its own text - it should be based off a canny or depth of the graphic with the words told to the model").
Case: 22_batch-24gb v005 r02, word DRIFT (Codystar dotted, lanes, handoff -> read NONE / II / II -> flat) in shot s2 (tests/fixtures/v005_r02_piece.json). Synthetic frames, no GPU.
Run: python -m pytest tests/test_text_match.py -q"""
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
from tools.app import layered_shots as LS  # noqa: E402
from tools.fusion import design_step as DS  # noqa: E402
from tools.fusion import ltx_check as LC  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import text_guard as TG  # noqa: E402

FX = HERE / "fixtures"
REQ = {"words": "SLIDE\nLAUGH\nWIND\nUP\nHI\nGO", "scene": "a wide frozen lake under pale winter sun, a child in a red jacket", "music": "calm piano", "length": 16, "look": "auto",
       "seed": 5, "quality": "preview", "format": "9:16"}


def v005(n):
    return json.loads((FX / f"v005_r0{n}_piece.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def music():
    return synth_music(120, 16)


TEXTURED_ASK = "fill the letters with carved ice"      # 2026-11-01: letters are solid (every LTX material word flat) unless the user asks for a material (letter_fill.py)


def _plan_textured(req, music):
    """plan_piece, then what MographPiecePlan does: the user's typed text lands on the piece (user_style.stamp) and the owner pass runs again, which brings back the LTX words the solid-letters default had flattened."""
    from tools.fusion import user_style as US
    p = PR.plan_piece(req, music)
    US.stamp(p, TEXTURED_ASK)
    return OW.apply_to_plan(p)


# ------------------------------------------------------------------------------------------------ (a) the prompt describes the actual graphic
def test_clause_names_the_graphics_look_and_layout():
    c = OW.clause("DRIFT", "glowing wire strands, thin", OW.font_look("Codystar-Regular.ttf"), ["DRI", "FT"])
    assert c == 'the word "DRIFT" in dotted dot-matrix capitals stacked on 2 lines formed as glowing wire strands, thin letters, extruded 3D lettering'
    assert OW.clause("SPIN", "carved ice") == 'the word "SPIN" formed as carved ice letters, extruded 3D lettering'            # no look, no lines: the plain old form
    assert OW.font_look("Anybody[wdth,wght].ttf") == "heavy block" and OW.font_look("BungeeOutline-Regular.ttf") == "outlined hollow" and OW.font_look("nope.ttf") == ""
    base = "a kayak crosses a lake, camera: slow push in, cinematic"
    wins = [dict(start=0.0, end=4.0, next=None, ltx_t=None, render="handoff", ltx_at=0.5, material="carved ice", text="DRIFT", look="dotted dot-matrix", lines=["DRI", "FT"])]
    p = OW.prompt_with_clause(base, wins, 0.0, 4.0)
    assert 'in dotted dot-matrix capitals stacked on 2 lines formed as carved ice letters' in p and OW.strip_clause(p) == "a kayak crosses a lake, camera: slow push in, cinematic"     # CLAUSE_RE still strips it
    assert OW.prompt_with_clause(p, wins, 0.0, 4.0) == p                                                                     # idempotent


def test_planned_handoff_word_prompt_carries_its_font_look(music):
    own = [{"render": "handoff", "material": "carved ice"}] + [{"render": "flat"}] * 5
    p = _plan_textured(dict(REQ, owners=json.dumps(own)), music)
    ws = p["type"]["words"]
    k = next(i for i, w in enumerate(ws) if w["render"] == "handoff")
    s = next(s for s in p["shots"] if f'"{ws[k]["text"]}"' in s["prompt"])
    assert OW.font_look(ws[k]["font"]) in s["prompt"] and f'the word "{ws[k]["text"]}"' in s["prompt"]


# ------------------------------------------------------------------------------------------------ (b) the control pass follows the glyph weight
def test_control_pass_is_picked_from_the_glyph_weight():
    assert OW.stroke_class("Codystar-Regular.ttf") == "thin" and OW.stroke_class("BungeeOutline-Regular.ttf") == "thin" and OW.stroke_class("Doto[ROND,wght].ttf") == "thin"
    assert OW.stroke_class("Anybody[wdth,wght].ttf") == "heavy" and OW.stroke_class("BadeenDisplay-Regular.ttf") == "heavy" and OW.stroke_class("Jacquard24-Regular.ttf") == "mid"
    assert OW.control_pass_for(["Codystar-Regular.ttf"]) == "line" and OW.control_pass_for(["Anybody[wdth,wght].ttf"]) == "depth" and OW.control_pass_for(["Jacquard24-Regular.ttf"]) == "union"
    assert OW.control_pass_for(["Anybody[wdth,wght].ttf", "Codystar-Regular.ttf"]) == "line"             # the thinnest face of the shot decides
    assert OW.control_pass_for([]) == "depth"


def test_plan_sets_the_control_pass_of_the_ltx_shot_and_the_passes_it_needs(music):
    own = [{"render": "handoff", "material": "carved ice"}] + [{"render": "flat"}] * 5
    p = _plan_textured(dict(REQ, owners=json.dumps(own)), music)
    w0 = p["type"]["words"][0]
    want = OW.control_pass_for([w0["font"]], [w0.get("axes")])
    assert p["shots"][0]["control"]["pass"] == want
    assert all(s["control"]["guide"] == 0.0 for s in p["shots"][1:])                                    # the flat shots have no control pass at all
    need = {"line": {"line"}, "depth": {"depth"}, "union": {"depth", "line"}}[want]
    assert need <= set(p["type_passes"])
    from tools.fusion import recipe_eval as RE
    assert "union" not in RE.need_passes(dict(p, shots=[dict(p["shots"][0], control={"pass": "union", "guide": 1.0})])) and "union" not in RE.control_passes(dict(p, shots=[dict(p["shots"][0], control={"pass": "union"})]))


def test_union_control_frames_are_depth_with_the_line_pass_drawn_over(tmp_path):
    pytest.importorskip("torch")
    try:
        from nodes_fusion import footage as FO
    except Exception as e:      # noqa: BLE001  ComfyUI not importable here
        pytest.skip(f"nodes_fusion.footage not importable: {type(e).__name__}")
    for name, v in (("depth", 90), ("line", 0)):
        d = tmp_path / name
        d.mkdir()
        for i in range(2):
            im = np.full((32, 18), v, np.uint8)
            if name == "line":
                im[10:12, :] = 255
            cv2.imwrite(str(d / f"{name}_{i:05d}.png"), im)
    u = FO.read_control_frames(str(tmp_path), "union", 0, 2, (18, 32))
    assert u.shape[0] == 2 and u[0][..., 0].max() > 0.95 and abs(float(u[0][..., 0][0, 0]) - 90 / 255) < 0.02


# ------------------------------------------------------------------------------------------------ (c) the layout check
def _frames(w=108, h=192):
    clean = np.full((h, w, 3), 0.3, np.float32)
    flat = np.zeros((h, w), np.uint8)
    cv2.putText(flat, "DRIFT", (4, 100), cv2.FONT_HERSHEY_DUPLEX, 1.3, 255, 8)
    return clean, flat


def test_layout_iou_is_high_for_letters_on_the_graphic_and_low_for_letters_elsewhere():
    clean, flat = _frames()
    on = clean.copy()
    on[flat > 0] = 0.95                                                      # the video model drew its letters where the graphic is
    off = clean.copy()
    stacked = np.zeros_like(flat)
    for k, ch in enumerate("DRIFT"):                                         # v005 r02: its OWN vertical stack, not the horizontal graphic
        cv2.putText(stacked, ch, (40, 30 + 32 * k), cv2.FONT_HERSHEY_DUPLEX, 1.1, 255, 6)
    off[stacked > 0] = 0.95
    hi = LC.layout_iou_frames([(clean, on, flat)] * 4)
    lo = LC.layout_iou_frames([(clean, off, flat)] * 4)
    assert hi["iou"] >= LC.LAYOUT_IOU_MIN and lo["iou"] < LC.LAYOUT_IOU_MIN and hi["iou"] > lo["iou"] + 0.3
    assert LC.layout_iou_frames([(clean, on, flat)] * 2) is None             # fewer than MIN_FRAMES usable frames: unknown


def _layout_scene(tmp_path, drawn):
    from test_owner_concept import _word_mask, _write_matte_dir
    from test_ltx_read import _piece
    p = _piece(("HANDS",), 6.0)
    hw = (256, 144)
    td = _write_matte_dir(tmp_path / "type", p, hw, 144)
    clean = np.full((144,) + hw + (3,), 90, np.uint8)
    text = clean.copy()
    w = OW.windows(p)[0]
    m = _word_mask("HANDS", hw) > 0
    if drawn == "elsewhere":
        m = np.roll(np.roll(m, 70, axis=0), 60, axis=1)
    for f in range(144):
        if w["start"] <= f / 24.0 < w["end"]:
            text[f][m] = 240
    return p, td, clean, text


def test_a_word_counts_as_ltx_only_when_the_reading_and_the_layout_both_pass(tmp_path):
    reads = [dict(i=0, text="HANDS", frames=[1], readings=["HANDS", "HANDS", "HANDS"])]
    ok_p, td, clean, text = _layout_scene(tmp_path / "a", "on")
    q = copy.deepcopy(ok_p)
    assert LC.apply_fallback(q, "s1", td, clean, text, reads=reads) == [] and q["type"]["words"][0]["render"] == "handoff"
    assert q["plan"]["owner"]["ltx_check"][0]["layout_ok"] is True and q["plan"]["owner"]["ltx_check"][0]["iou"] >= LC.LAYOUT_IOU_MIN
    bad_p, td2, clean2, text2 = _layout_scene(tmp_path / "b", "elsewhere")
    q2 = copy.deepcopy(bad_p)
    msgs = LC.apply_fallback(q2, "s1", td2, clean2, text2, reads=reads)                  # the reading passes, the letters are not where the graphic is
    assert len(msgs) == 1 and "not the graphic's layout" in msgs[0] and q2["type"]["words"][0]["render"] == "flat" and q2["type"]["words"][0]["ltx_missing"]
    q3 = copy.deepcopy(bad_p)
    q3["layers"] = {"ltx_iou": False}
    assert LC.apply_fallback(q3, "s1", td2, clean2, text2, reads=reads) == []            # the check can be switched off


# ------------------------------------------------------------------------------------------------ (d) one text at any moment
def test_handoff_is_one_text_flat_until_ltx_at_then_a_short_crossfade():
    wins = OW.windows({"type": {"words": [{"text": "DRIFT", "t0": 0.0, "t1": 4.0, "render": "handoff", "ltx_at": 0.5}]}})
    w = wins[0]
    t_h = OW.handoff_t(w)
    assert OW.flat_gain_of(w, t_h - 0.01) > 0.99 and OW.flat_gain_of(w, t_h + OW.FLAT_FADE_S) < 0.01 and OW.FLAT_FADE_S <= OW.HANDOFF_MAX_FADE_S and OW.CONTROL_RAMP_S <= OW.HANDOFF_MAX_FADE_S
    for t in np.arange(0.0, 4.0, 1 / 24):
        assert OW.flat_gain_of(w, t) + OW.control_gain_of(w, t) == pytest.approx(1.0, abs=1e-6)       # never both held, never neither
    piece = {"fps": 24, "layout": {"s1": {"start": 0, "frames": 96, "xf": 0}}, "shots": [{"id": "s1"}], "type": {"words": [{"text": "DRIFT", "t0": 0.0, "t1": 4.0, "render": "handoff", "ltx_at": 0.5}]}}
    assert OW.check_one_text(piece, {"s1": True}) == []
    probs = OW.check_one_text(piece, {"s1": False})                                    # no text-rendered background: the flat word would fade to nothing
    assert len(probs) == 1 and "no text-rendered background" in probs[0]
    assert OW.check_one_text({"type": {"words": [{"text": "A", "t0": 0, "t1": 2}]}}) == []        # an old piece has one owner (flat)


def test_after_a_fallback_the_shot_uses_the_clean_clip_and_one_text_remains_v005_r02(tmp_path):
    """the box piece: DRIFT read NONE / II / II -> flat, but the shot kept the text-pass clip (the LTX layout of its own) under the flat word = two versions. Now the clean clip is the background."""
    p = v005("2")
    for w in p["type"]["words"]:                                    # the piece as planned, before compose mended it
        if w["text"] == "DRIFT":
            w.update(render="handoff", ltx_at=0.5, material="glowing wire strands, thin")
            w.pop("ltx_missing", None)
            w.pop("render_asked", None)
    clean = np.full((6, 64, 36, 3), 90, np.uint8)
    text = clean.copy()
    text[:, 10:50, 10:25] = 200                                      # the video model's own letters
    clip = dict(sid="s2", u8=text, subject=clean, textbg=True)
    ws = p["type"]["words"]
    for w in ws:
        if w["text"] != "DRIFT":
            w["render"] = "flat"
    # the VLM read failed -> the word is flat (this is what apply_fallback records)
    LC_msgs = LC.demote_shot_words(p, [1], "the vision model read ['NONE', 'II', 'II']")
    assert LC_msgs and ws[1]["render"] == "flat"
    out, notes = LC.choose_background(p, "s2", clip, None, None)
    assert out["u8"] is clean and out["textbg"] is False and any("no LTX-owned word is left" in n for n in notes)
    assert OW.check_one_text(p, {"s2": False}) == []


# ------------------------------------------------------------------------------------------------ the renderer never invents its own text
def test_strip_text_and_spelled_words():
    clean, removed = TG.strip_text("a kayak on a lake, the word \"DRIFT\" formed as wire, a neon sign above the door, camera: slow push in, cinematic")
    assert clean == "a kayak on a lake, camera: slow push in, cinematic" and len(removed) == 2
    assert TG.spelled_words('x, the word "DRIFT" formed as wire, the words "A B" formed as ice') == ["DRIFT", "A B"]
    assert TG.strip_text("a runner passes a billboard, cinematic")[0] == "cinematic"


def test_v005_r02_shot_without_a_control_pass_names_no_text_and_stills_rule_text_out():
    p = v005("2")
    for s in p["shots"]:
        if s["id"] == "s2":                                           # the box piece's shot 2 as the plan wrote it: the word named, guide 1.0
            s["prompt"] = s["prompt"]
    broken = copy.deepcopy(p)
    for w in broken["type"]["words"]:
        w["render"] = "flat"                                          # every word flat -> no shot has a control pass
        for k in ("ltx_at", "material", "ltx_missing", "render_asked"):
            w.pop(k, None)
    broken["shots"][1]["control"]["guide"] = 0.0
    broken["shots"][1]["guide"] = 0.0
    broken["shots"][1]["prompt"] = 'The kayak approaches a ripple, camera: slow push in, the word "DRIFT" formed as glowing wire strands, thin letters, extruded 3D lettering, cinematic'
    broken["stills"][0]["prompt"] += ", a sign reading LAKE"
    probs = TG.plan_problems(broken)
    assert any(x.startswith("s2:") and "names text" in x for x in probs) and any(x.startswith("still A") and "sign" in x for x in probs)
    fx = TG.fix_plan(broken)
    assert TG.plan_problems(broken) == [] and fx
    assert "DRIFT" not in broken["shots"][1]["prompt"] and "letters" not in broken["shots"][1]["prompt"] and "sign" not in broken["stills"][0]["prompt"]
    assert broken["shots"][1]["negative"] == TG.NEG_NO_TEXT and all(t in broken["stills"][0]["negative"] for t in TG.STILL_NEG_TOKENS)
    assert TG.fix_plan(broken) == []                                  # idempotent


def test_a_shot_with_an_ltx_word_spells_exactly_the_control_passs_word(music):
    own = [{"render": "handoff", "material": "carved ice"}] + [{"render": "flat"}] * 5
    p = _plan_textured(dict(REQ, owners=json.dumps(own)), music)
    assert TG.plan_problems(p) == []
    ws = p["type"]["words"]
    assert TG.spelled_words(p["shots"][0]["prompt"]) == [ws[0]["text"].upper()] and all(not TG.spelled_words(s["prompt"]) for s in p["shots"][1:])
    bad = copy.deepcopy(p)
    bad["shots"][0]["prompt"] = bad["shots"][0]["prompt"].replace(ws[0]["text"].upper(), "SOMETHING ELSE")
    assert any("spells" in x for x in TG.plan_problems(bad))
    assert TG.fix_plan(bad) and TG.plan_problems(bad) == [] and TG.spelled_words(bad["shots"][0]["prompt"]) == [ws[0]["text"].upper()]
    bad2 = copy.deepcopy(p)
    bad2["shots"][0]["control"]["guide"] = 0.0
    assert any("control guide is 0" in x for x in TG.plan_problems(bad2))
    default = PR.plan_piece(REQ, music)                               # the default: all flat -> no shot names text
    assert TG.plan_problems(default) == [] and all(not TG.TEXT_RE.search(s["prompt"]) for s in default["shots"])


def test_old_pieces_and_drafts_keep_their_control_prompts(music):
    draft = PR.plan_piece(dict(REQ, quality="draft"), music)
    assert TG.plan_problems(draft) == [] or all("control" not in x for x in TG.plan_problems(draft))
    legacy = {"layout": {"s1": {"start": 0, "frames": 96}}, "shots": [{"id": "s1", "prompt": "glowing letters built into the scene, cinematic", "control": {"pass": "depth", "guide": 0.4}}],
              "type": {"words": [{"text": "A", "t0": 0, "t1": 2}]}, "stills": []}
    assert TG.plan_problems(legacy) == [] and TG.fix_plan(legacy) == [] and "letters" in legacy["shots"][0]["prompt"]


def test_layered_graph_gives_the_clean_chain_a_text_free_negative():
    base = ROOT / "app" / "graphs" / "fusion_design_qwen_auto.json"
    if not base.is_file():
        pytest.skip("base graph not present")
    g0 = json.loads(base.read_text(encoding="utf-8"))
    g, _ = LS.convert(g0, None)
    n1 = g["l1.8"]["inputs"]["negative"]
    assert g[n1[0]]["inputs"]["text"] == TG.NEG_NO_TEXT and g[g["t1.8"]["inputs"]["negative"][0]]["inputs"]["text"] == TG.NEG_TEXT_PASS
    assert LS.clean_prompt('a runner, the word "SPIN" formed as ice letters, extruded 3D lettering, a neon sign, camera: slow push, cinematic') == "a runner, camera: slow push, cinematic"
    assert LS.check(g) == []


# ------------------------------------------------------------------------------------------------ the designed word keeps the plan's graphic
def test_design_step_locks_the_ltx_words_graphic_to_the_control_pass_it_was_made_from(music):
    own = [{"render": "handoff", "material": "carved ice"}] + [{"render": "flat"}] * 5
    plan = _plan_textured(dict(REQ, owners=json.dumps(own)), music)
    reply = DS.reply_from_piece(plan, music)
    w0 = reply["type"]["words"][0]
    w0["font"] = "Barrio-Regular.ttf" if plan["type"]["words"][0]["font"] != "Barrio-Regular.ttf" else "Honk-Regular.ttf"
    w0["module"] = "slats" if plan["type"]["words"][0]["module"] != "slats" else "lanes"
    w0["placement"] = dict(w0.get("placement") or {}, wf=0.5)
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(plan)), plan, music, "P" * 500)
    got = json.loads(out["piece_json"])["type"]["words"][0]
    pw = plan["type"]["words"][0]
    assert got["font"] == pw["font"] and got["module"] == pw["module"] and got.get("placement", {}).get("wf") == pw.get("placement", {}).get("wf")
    assert any("one version" in x for x in out["report"].splitlines()) or "one version" in out["report"]
    assert TG.plan_problems(out["piece"]) == []


# ------------------------------------------------------------------------------------------------ the box probe builds offline
def test_text_pass_probe_builds_the_control_pass_sweep_and_scores_a_render(tmp_path):
    from tools.app import text_pass_probe as TP
    piece = v005("2")
    base = ROOT / "app" / "graphs" / "fusion_design_qwen_auto_layers.json"
    if not base.is_file():
        pytest.skip("layered graph not present")
    for nm, (start, guide, pm, *ps) in TP.SWEEP_PASSES.items():
        q = json.loads(TP.with_pass(piece, 2, ps[0]))
        assert q["shots"][1]["control"]["pass"] == ps[0] and OW.windows(q)[1]["render"] == "handoff"        # DRIFT is asked for again (the archived piece had the fallback applied)
        g = TP.build(piece, json.dumps(q), "s2_clean.mp4", start=start, guide=guide, prompt=None, k=2, name=nm, downscale=(ps[1] if len(ps) > 1 else None))
        assert "t2.32" in g and g["lay_sig"]["inputs"]["sigmas"].startswith(str(start)) and g["t2.8"]["inputs"]["negative"] == ["lay_negt", 0]
        if len(ps) > 1:
            assert g["t2.32"]["inputs"]["latent_downscale_factor"] == ps[1]
    # score: letters drawn on the graphic pass, letters elsewhere do not
    clean, flat = _frames()
    mdir = tmp_path / "matte"
    mdir.mkdir()
    for i in range(8):
        cv2.imwrite(str(mdir / f"matte_{i:05d}.png"), flat)

    def write(path, drawn):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (clean.shape[1], clean.shape[0]))
        for _ in range(8):
            f = clean.copy()
            f[drawn > 0] = 0.95
            vw.write(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()
    write(tmp_path / "clean.mp4", np.zeros_like(flat))
    write(tmp_path / "on.mp4", flat)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TP.score(str(tmp_path / "on.mp4"), str(tmp_path / "clean.mp4"), str(mdir))
    assert json.loads(buf.getvalue())["passes_layout"] is True


# ------------------------------------------------------------------------------------------------ the text clip is never the background unless its word passed (24 GB QC 2026-10-29)
def test_prepare_backgrounds_judges_every_shot_before_choosing_and_never_keeps_an_unpassed_text_clip(tmp_path):
    """v005 r01: CARVE failed its reading, HOLD (a word that spans three shots) failed in the last of them; the text clips of those shots were kept under the flat words = two versions."""
    from test_ltx_read import _piece
    p = _piece(("HANDS", "CLAY"), 6.0)
    clean = np.full((144, 64, 48, 3), 90, np.uint8)
    text = clean.copy()
    text[:, 10:30, 10:30] = 200
    clips = [dict(sid="s1", u8=text, subject=clean, textbg=True)]
    log = []
    out = LC.prepare_backgrounds(p, str(tmp_path), clips, None, log=log.append)          # no matte pass: nothing can be verified
    assert [c["textbg"] for c in out] == [False] and all(c["u8"] is clean for c in out)
    assert all(w["render"] == "flat" and w.get("ltx_missing") for w in p["type"]["words"]) and any("not verified" in m for m in log)
    assert OW.check_one_text(p, {c["sid"]: bool(c["textbg"]) for c in out}) == []
    lax = copy.deepcopy(_piece(("HANDS", "CLAY"), 6.0))
    lax["layers"] = {"ltx_strict": False, "drift_check": False}                            # the old trusting behaviour is an explicit opt-out
    out2 = LC.prepare_backgrounds(lax, str(tmp_path), [dict(sid="s1", u8=text, subject=clean, textbg=True)], None, log=lambda m: None)
    assert out2[0]["textbg"] is True


# ------------------------------------------------------------------------------------------------ a string where a pass list belongs never crashes the type stage (box v008 r01)
def test_a_string_type_passes_or_control_pass_never_reaches_set():
    """`unknown pass 'a'; known: (...)`: a model-written type_passes "all" was split into letters by set(). Every reader is tolerant, the design step mends the field and logs it."""
    from tools.fusion import recipe_eval as RE
    from tools.fusion import type_passes as TPS
    assert set(RE.KNOWN_PASSES) == set(TPS.PASSES)
    base = {"shots": [{"id": "s1", "kind": "control", "control": {"pass": "depth", "guide": 1.0}}], "compose": {}}
    for bad, want in (("all", set(RE.DEFAULT_PASSES)), ("auto", set(RE.DEFAULT_PASSES)), ("matte", {"matte"}), ("depth+line", {"depth", "line", "matte"}), ("matte, depth", {"matte", "depth"}),
                      (["matte", "nonsense"], {"matte"}), ("zzz", set(RE.DEFAULT_PASSES)), (None, set(RE.DEFAULT_PASSES))):
        ps = RE.need_passes(dict(base, type_passes=bad))
        assert set(ps) <= set(RE.KNOWN_PASSES), (bad, ps)
        assert want <= set(ps), (bad, ps)
    for pas in ("depth+line", "all", "line", "union", "zzz", "depth, line"):
        piece = dict(base, shots=[{"id": "s1", "kind": "control", "control": {"pass": pas}}], type_passes=["matte"])
        assert set(RE.need_passes(piece)) <= set(RE.KNOWN_PASSES) and set(RE.control_passes(piece)) <= set(RE.KNOWN_PASSES), pas
    assert RE.norm_control_pass("depth+line") == "union" and RE.norm_control_pass("all") == "depth" and RE.norm_control_pass("line") == "line" and RE.norm_control_pass("zzz") == "depth"
    notes: list = []
    p = {"type_passes": "all", "shots": [{"id": "s1", "control": {"pass": "depth+line"}}, {"id": "s2", "control": {"pass": "bogus"}}]}
    got = DS.normalize_passes(p)
    assert p["type_passes"] == list(RE.DEFAULT_PASSES) and [s["control"]["pass"] for s in p["shots"]] == ["union", "depth"] and got and any("type_passes" in x for x in got)
    assert DS.normalize_passes(p) == []                                                                   # idempotent
    # the type stage's own assertion never sees a letter
    import inspect
    assert "assert p in PASSES" in inspect.getsource(TPS) and all(len(x) > 1 for x in RE.need_passes(dict(base, type_passes="all")))


def test_design_parse_mends_a_string_type_passes_of_the_plan(music):
    plan = PR.plan_piece(REQ, music)
    reply = DS.reply_from_piece(plan, music)
    bad = copy.deepcopy(plan)
    bad["type_passes"] = "all"
    bad["shots"][0]["control"]["pass"] = "depth+line"
    out = DS.parse(json.dumps(reply), json.dumps(DS.make_locked(bad)), bad, music, "P" * 500)
    p = json.loads(out["piece_json"])
    from tools.fusion import recipe_eval as RE
    assert isinstance(p["type_passes"], list) and set(p["type_passes"]) <= set(RE.KNOWN_PASSES) and p["shots"][0]["control"]["pass"] == "union"
    assert set(RE.need_passes(p)) <= set(RE.KNOWN_PASSES) and any("type_passes" in n for n in p["plan"]["notes"])
