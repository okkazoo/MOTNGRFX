"""2026-11-01 colour review (tools/fusion/legible_color.py shot inks / support / tones / echo front, tools/fusion/user_style.py, recipe_eval wiring).
Synthetic frames, numpy backend, CPU: the ink comes from THE SHOT's hues and still passes the contrast + ring floor, falls back when no hue passes, the outline
is a complementary shot hue (not black), consecutive shots do not repeat the hue, the wake sits behind a crisp front copy at a low opacity in another colour,
literal flat grounds / inverse type take the shot's hues at >= 4.5:1, the user's palette / legibility wish wins, outline weights vary per word."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import legible_color as LC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
from tools.fusion import user_style as US  # noqa: E402

H, W = 160, 96
WORD = [[0.0, 0.0, 1.0, 1.0, False]]
NAVY = np.array([0.08, 0.12, 0.26], np.float32)
CRIMSON = np.array([0.80, 0.08, 0.16], np.float32)
TEAL = np.array([0.05, 0.55, 0.55], np.float32)


def _alpha():
    a = np.zeros((H, W), np.float32)
    a[60:100, 18:78] = 1.0
    return a


def _plate(base, accent=None, accent_share=0.08):
    p = np.zeros((H, W, 3), np.float32) + base
    if accent is not None:
        n = int(H * accent_share)
        p[:n] = accent                                                     # a band of the accent away from the word
    return p


def _hue_deg(rgb):
    lab = LC.lin_to_oklab(LC.srgb_to_lin(np.asarray(rgb, np.float32)))
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


def _run(plate, key="t:sc", frames=6, face=None, alpha=None, **kw):
    LC.reset(key)
    face = np.zeros((H, W, 3), np.float32) + 0.95 if face is None else face
    alpha = _alpha() if alpha is None else alpha
    out = None
    for i in range(frames):
        out = LC.legible_color(face, alpha, plate, t=0.5, frame_idx=i, u=0.5, fps=24, state_key=key, words=WORD, **kw)
    return out


def _ink(rgba, alpha=None):
    m = (_alpha() if alpha is None else alpha) > 0.5
    return np.median(rgba[..., :3][m], axis=0)


def test_shot_hues_rank_a_small_saturated_accent_first():
    hues = LC.shot_hues(LC.srgb_to_lin(_plate(NAVY, CRIMSON)))
    assert hues and LC.hue_dist(hues[0]["h"], _hue_deg(CRIMSON)) < 20, hues
    assert any(LC.hue_dist(h["h"], _hue_deg(NAVY)) < 20 for h in hues)
    assert LC.shot_hues(LC.srgb_to_lin(np.full((H, W, 3), 0.5, np.float32))) == [], "a grey plate has no hue"


def test_ink_is_the_shots_hue_and_passes_the_contrast_floor():
    plate = _plate(NAVY, CRIMSON)
    out = _run(plate, shot_palette=True)
    ink = _ink(out)
    rec = LC.LC_LOG["t:sc"][5][0]
    assert rec["kind"] in ("shot", "kept"), rec
    assert LC.hue_dist(_hue_deg(ink), _hue_deg(CRIMSON)) < 25, (ink, rec)
    yi, yb = float(LC.rel_lum(LC.srgb_to_lin(ink))), float(LC.rel_lum(LC.srgb_to_lin(NAVY)))
    assert LC.contrast_ratio(yi, yb) >= LC.LC_TARGET_LARGE, "legibility wins over colour"
    assert rec["cover"] >= LC.LC_NEED - 0.05


def test_no_passing_hue_falls_back_to_the_old_chain():
    plate = np.full((H, W, 3), 0.46, np.float32)                           # mid grey Y 0.18: no ink reaches 4.5 x 1.1 (white ~4.6, black ~4.6)
    hues = LC.shot_hues(LC.srgb_to_lin(_plate(NAVY, CRIMSON)))
    hist = LC._hist(LC.rel_lum(LC.srgb_to_lin(plate)).reshape(-1), np.ones(H * W))
    assert LC.choose_shot(hist, hist, hues, 4.5) is None
    out = _run(plate, shot_palette=True, target=4.5)
    assert LC.LC_LOG["t:sc"][5][0]["kind"] not in ("shot",)
    assert out.shape == (H, W, 4)


def test_support_colour_is_complementary_not_black():
    ink = LC.srgb_to_lin(np.array([0.95, 0.75, 0.2], np.float32))         # a light gold ink: dark support
    hues = [dict(h=_hue_deg(TEAL), c=0.1), dict(h=_hue_deg(CRIMSON), c=0.2)]
    sup = LC.support_color(ink, hues, light=False)
    assert LC.hue_dist(_hue_deg(LC.lin_to_srgb(sup)), _hue_deg(LC.lin_to_srgb(ink))) >= 90
    assert LC.contrast_ratio(float(LC.rel_lum(ink)), float(LC.rel_lum(sup))) >= 3.0
    s8 = LC.lin_to_srgb(sup)
    assert float(s8.max() - s8.min()) > 0.04, "a hue, not near-black"
    dark_ink = LC.srgb_to_lin(np.array([0.1, 0.05, 0.3], np.float32))
    light = LC.support_color(dark_ink, hues, light=True)
    assert LC.contrast_ratio(float(LC.rel_lum(dark_ink)), float(LC.rel_lum(light))) >= 3.0


def test_forced_outline_draws_in_the_support_hue():
    plate = _plate(NAVY, CRIMSON)
    out = _run(plate, shot_palette=True, outline=[[0.0, 1.0]])
    rec = LC.LC_LOG["t:sc"][5][0]
    assert "outline" in rec["support"] and rec["sup_hex"] not in ("#080808", "#f7f7f7")
    ring = (out[..., 3] > 0.3) & (_alpha() < 0.5)
    assert ring.any()


def test_consecutive_shots_do_not_repeat_the_ink_hue():
    plate = _plate(NAVY, CRIMSON, accent_share=0.15)
    plate[-int(H * 0.15):] = TEAL                                          # two accents: crimson + teal
    LC.reset()
    hues = []
    for k, sid in enumerate(("s1", "s2", "s3")):
        out = None
        for i in range(4):
            out = LC.legible_color(np.zeros((H, W, 3), np.float32) + 0.95, _alpha(), plate, t=0.5, frame_idx=i, u=0.5, fps=24,
                                   state_key=f"P:{sid}:lc0", words=WORD, shot_palette=True, piece_key="P", shot_t0=float(k * 4))
        hues.append(_hue_deg(_ink(out)))
    assert LC.hue_dist(hues[0], hues[1]) >= LC.LC_VARY_DEG - 5, hues
    assert LC.hue_dist(hues[1], hues[2]) >= LC.LC_VARY_DEG - 5, hues


def test_echo_sits_behind_a_crisp_front_copy():
    front = _alpha()
    wake = front.copy()
    wake[60:100, 78:92] = 1.0                                              # trail to the right of the word
    plate = _plate(NAVY, CRIMSON)
    out = _run(plate, alpha=wake, front=front, front_face=np.zeros((H, W, 3), np.float32) + 0.95, shot_palette=True)
    a = out[..., 3]
    assert float(a[front > 0.5].min()) > 0.99, "the front copy is fully opaque"
    trail = (wake > 0.5) & (front < 0.5)
    trail[:, 78:80] = False
    assert float(a[trail].max()) <= LC.LC_ECHO_ALPHA + 0.02, "the echo is subdued"
    ink, echo = _ink(out), np.median(out[..., :3][trail], axis=0)
    assert float(np.abs(ink - echo).max()) > 0.05, "the echo is not the same ink"
    out0 = _run(plate, alpha=wake, front=front, echo_alpha=0.0, shot_palette=True)
    assert float(out0[..., 3][trail].max()) < 0.02, "echo_alpha 0 = no wake (the user asked for plain text)"


def test_tone_pair_tints_the_ground_and_keeps_the_ink_legible():
    hues = LC.shot_hues(LC.srgb_to_lin(_plate(NAVY, CRIMSON)))
    g, i = LC.tone_pair(hues, [0.913, 0.914, 0.909], [0.035, 0.039, 0.035])
    yg, yi = (float(LC.rel_lum(LC.srgb_to_lin(np.asarray(x, np.float32)))) for x in (g, i))
    assert LC.contrast_ratio(yg, yi) >= 4.5
    assert float(np.asarray(i).max() - np.asarray(i).min()) > 0.05, "the inverse type is a hue, not black"
    assert LC.tone_pair([], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0])[1].tolist() == [0.0, 0.0, 0.0]


def test_style_ops_turn_literal_ground_and_inverse_type_into_shot_tones():
    ops = [{"op": "fill", "out": "fx", "args": {"color": [0.91, 0.91, 0.91], "like": "$footage"}},
           {"op": "over", "out": "fx", "args": {"alpha": "$tmat", "base": "$fx", "fg": [0.03, 0.03, 0.03]}},
           {"op": "fill", "out": "warm", "args": {"color": [1.0, 0.8, 0.55], "like": "$footage"}}]
    out = RE.style_ops({"id": "P"}, ops, "s2")
    assert out[0]["op"] == "shot_tone" and out[0]["args"]["part"] == "ground"
    assert out[1]["op"] == "shot_tone" and out[1]["args"]["part"] == "ink" and out[2]["args"]["fg"] == "$" + out[1]["out"]
    assert out[3]["op"] == "fill", "a fill that is never a ground stays"
    assert RE.style_ops({"id": "P", "compose": {"shot_colour": False}}, ops, "s2") == ops
    img = LC.shot_tone(np.zeros((H, W, 3), np.float32), _plate(NAVY, CRIMSON), [0.91, 0.91, 0.91], [0.03, 0.03, 0.03], part="ink", state_key="t:tone")
    assert img.shape == (H, W, 3)


def test_user_direction_palette_and_legibility():
    st = US.parse_direction({"words": "RISE\nFALL\npalette: crimson, gold", "direction": "please make the text more legible"})
    assert st["palette"] == [list(US.NAMES["crimson"]), list(US.NAMES["gold"])] and st["legible_boost"] and st["plain"]
    assert US.parse_direction({"request": {"words": "RED SKY\nBLUE MOON"}})["palette"] is None, "lyrics are never read as colours"
    assert US.parse_direction({"brief": {"direction": "use #ff8800 and #224466"}})["palette"] == [[255, 136, 0], [34, 68, 102]]
    assert US.parse_direction({"user_style": {"palette": ["#00ff00"], "legible_boost": True}})["palette"] == [[0, 255, 0]]
    assert US.parse_direction("")["palette"] is None


def test_user_palette_wins_over_the_shot():
    plate = _plate(NAVY, CRIMSON)
    out = _run(plate, shot_palette=True, user_palette=[[0.2, 0.9, 0.3]])
    assert LC.hue_dist(_hue_deg(_ink(out)), _hue_deg([0.2, 0.9, 0.3])) < 25


def test_boost_raises_the_target_and_forces_the_outline():
    out = _run(_plate(NAVY, CRIMSON), shot_palette=True, boost=True)
    rec = LC.LC_LOG["t:sc"][5][0]
    assert rec["target"] >= LC.LC_TARGET_SMALL * 1.2 - 1e-6 and "outline" in rec["support"]
    assert out.shape == (H, W, 4)


def test_outline_weights_vary_per_word_and_are_seeded():
    p = {"id": "A1", "plan": {"piece_system": {"seed": 7}}}
    w = RE.outline_weights(p, 6)
    assert w == RE.outline_weights(p, 6) and len(set(w)) > 2
    assert all(RE.OUTLINE_W[0] <= x <= RE.OUTLINE_W[1] for x in w)
    assert all(abs(a - b) >= RE.OUTLINE_W_STEP - 1e-6 for a, b in zip(w, w[1:]))
    thin = _run(_plate(NAVY, CRIMSON), shot_palette=True, outline=[[0.0, 1.0]], outline_w=0.6)
    bold = _run(_plate(NAVY, CRIMSON), shot_palette=True, outline=[[0.0, 1.0]], outline_w=2.2)
    assert float((bold[..., 3] > 0.3).sum()) > float((thin[..., 3] > 0.3).sum())


def test_hold_glint_breathes_on_the_beat_and_is_subtle():
    plate = _plate(NAVY, CRIMSON)
    LC.reset("t:g")
    a_on, a_off = None, None
    for i, t in enumerate((0.40, 0.45, 0.50, 0.70)):
        o = LC.legible_color(np.zeros((H, W, 3), np.float32) + 0.95, _alpha(), plate, t=t, frame_idx=i, u=0.5, fps=24, state_key="t:g", words=WORD,
                             shot_palette=True, life={"beats": [0.5], "amp": 0.4})
        if t == 0.50:
            a_on = o[..., 3]
        if t == 0.70:
            a_off = o[..., 3]
    halo = _alpha() < 0.5
    assert 0.0 < float(a_on[halo].max()) <= 0.45, "a hairline glint, never a solid outline"
    assert float(a_on[halo].sum()) > float(a_off[halo].sum()), "it breathes with the beat"


def test_recipe_draws_a_front_copy_and_later_wake_drawers_follow_it(tmp_path):
    (tmp_path / "beauty").mkdir()
    ops = [{"op": "flow_wake", "out": "wm", "args": {"type_a": "$type_matte", "flow": "$flow", "matte": "$matte", "trail": 6}},
           {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
           {"op": "sandwich", "out": "image", "args": {"footage": "$footage", "matte": "$m", "type_a": "$wm", "type_rgb": "$tcol", "matte_threshold": 0.5}},
           {"op": "mask_gate", "out": "wa", "args": {"a": "$wm", "gate": 1.0}},
           {"op": "knockout", "out": "image", "args": {"footage": "$image", "type_a": "$wa", "fg": "$footage", "fg_matte": "$m"}}]
    piece = {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]}, "shots": [{"id": "s1", "frames": 24, "matte": True}],
             "compose": {"crossfade": 0.0, "default": {"ops": ops}}, "plan": {"flow_wake": {"holds": [[0.3, 0.9]]}}, "type": {"words": [{"text": "ONE"}]}}
    rec = RE.build_recipe(piece, "s1", type_dir=str(tmp_path))
    lc = next(o for o in rec["ops"] if o["op"] == "legible_color")
    assert lc["args"]["alpha"] == "$wm" and lc["args"]["front"] == "$tmat" and lc["args"]["front_face"] == "$tface"
    assert lc["args"]["shot_palette"] is True and lc["args"]["piece_key"] == "T"
    gate = next(o for o in rec["ops"] if o.get("out") == "wa")
    ef = next(o for o in rec["ops"] if o["op"] == "screen" and o["out"] == gate["args"]["a"][1:])
    eb = next(o for o in rec["ops"] if o["out"] == ef["args"]["b"][1:])
    assert ef["args"]["a"] == "$tmat" and eb["args"]["a"] == "$wm" and eb["args"]["gate"] == LC.LC_ECHO_ALPHA, "a later pass draws the front copy + subdued echo, never the raw wake"
    piece["compose"]["echo_front"] = False
    lc2 = next(o for o in RE.build_recipe(piece, "s1", type_dir=str(tmp_path))["ops"] if o["op"] == "legible_color")
    assert "front" not in lc2["args"]


def test_echo_copies_drawn_by_the_type_pass_are_subdued_behind_the_bright_copy():
    a = np.zeros((H, W), np.float32)
    a[30:60, 18:78] = 1.0                                                   # the front copy
    a[100:130, 18:78] = 1.0                                                 # an echo copy the type pass drew below it, same alpha
    face = np.zeros((H, W, 3), np.float32)
    face[30:60, 18:78] = 0.95
    face[100:130, 18:78] = 0.30                                             # ... but dimmer in the beauty
    plate = _plate(NAVY, CRIMSON)
    out = _run(plate, alpha=a, face=face, shot_palette=True, echo_face=True)
    assert float(out[40:50, 30:60, 3].min()) > 0.99 and float(out[110:120, 30:60, 3].max()) <= LC.LC_ECHO_ALPHA + 0.02
    plain = _run(plate, alpha=a, face=face, shot_palette=True)
    assert float(plain[110:120, 30:60, 3].min()) > 0.99, "off: the matte is drawn as it is"
    one = a.copy()
    one[100:130] = 0.0
    shaded = face.copy()
    shaded[30:33, 18:78] = 0.35                                             # a dark bevel edge next to the face is not an echo
    o2 = _run(plate, alpha=one, face=shaded, shot_palette=True, echo_face=True)
    assert float(o2[30:58, 30:60, 3].min()) > 0.99
