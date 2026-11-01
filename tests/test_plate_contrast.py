"""plate_layers contrast modes (plate_in_subject / plate_in_type): pure numpy, run on the box (ComfyUI venv) or anywhere."""
import numpy as np

from tools.fusion import plate_layers as PLT

H, W, T = 64, 48, 3
PLATE = np.zeros((T, H, W, 3), np.uint8)
PLATE[...] = (30, 120, 200)            # the clean plate: blue room
CLEAN = np.zeros((T, H, W, 3), np.uint8)
CLEAN[...] = (30, 120, 200)
CLEAN[:, 20:50, 10:30] = (220, 180, 60)    # the subject (yellow) stands in the room
MATTE = np.zeros((T, H, W), np.float32)
MATTE[:, 20:50, 10:30] = 1.0
LET = np.zeros((H, W), np.float32)
LET[8:16, 4:44] = 1.0                   # one word line near the top, clear of the subject
LET2 = np.zeros((H, W), np.float32)
LET2[30:38, 4:44] = 1.0                 # a word line crossing the subject
FLAT, WORDC = (196, 48, 40), (244, 238, 224)
IDENT = dict(move={"dx": 0.0, "dy": 0.0, "scale": 1.0}, pan={"dx": 0.0, "dy": 0.0, "scale": 1.0})


def _bake(mode, letters, **kw):
    kw.setdefault("fill_source", "plate")          # these tests are about the picture rules, not the fill choice (the flat test plate is forced; the fill tests below pick freely)
    return PLT.bake(PLATE, CLEAN, MATTE, mode=mode, letters=lambda j: letters, color=False, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, guard=False, edge_line=False, **IDENT, **kw)


def test_modes_are_registered():
    assert PLT.CONTRAST_MODES == ("plate_in_subject", "plate_in_type", "window_subject_front", "text_as_mask")
    for m in PLT.CONTRAST_MODES:
        assert m in PLT.TYPE_MATTE_MODES and m in PLT.TYPE_MATTE_ALL
        assert PLT.norm_type_matte(m) == m
        assert PLT.shot_keys({"type_matte": m})["type_matte"] == m
        assert PLT.wants_plate({"shots": [{"id": "s1", "type_matte": m}]}, "s1")


def test_plate_in_subject_picture():
    r = _bake("plate_in_subject", LET)
    f = r["footage"][1].astype(int)
    assert np.abs(f[40, 20] - np.array((30, 120, 200))).max() <= 2            # inside the silhouette: the CLEAN PLATE, not the yellow subject
    assert np.abs(f[40, 40] - np.array(FLAT)).max() <= 2                      # outside both: the flat ground
    assert np.abs(f[12, 24] - np.array(WORDC)).max() <= 2                     # the word, flat colour
    assert r["matte"].max() == 0                                              # nothing stands in front of the type


def test_plate_in_subject_silhouette_covers_the_word():
    r = _bake("plate_in_subject", LET2)
    f = r["footage"][0].astype(int)
    assert np.abs(f[34, 20] - np.array((30, 120, 200))).max() <= 2            # the silhouette is in front of the word: plate where they overlap
    assert np.abs(f[34, 40] - np.array(WORDC)).max() <= 2                     # the rest of the word reads


def test_plate_in_type_picture():
    r = _bake("plate_in_type", LET2)
    f = r["footage"][2].astype(int)
    assert np.abs(f[34, 40] - np.array((30, 120, 200))).max() <= 2            # letters filled with the clean plate
    assert np.abs(f[45, 20] - np.array((220, 180, 60))).max() <= 2            # the subject (clean clip x matte) behind the word
    assert np.abs(f[5, 5] - np.array(FLAT)).max() <= 2                        # flat ground elsewhere
    assert r["matte"].max() == 0


def test_matte_follows_the_subject_in_contrast_modes():
    r = _bake("plate_in_type", np.zeros((H, W), np.float32))                  # no letters: the picture is flat + the subject exactly where the matte is
    f = r["footage"][0].astype(int)
    ys, xs = np.where(np.abs(f - np.array(FLAT)).max(axis=2) > 8)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (20, 49, 10, 29) or abs(ys.min() - 20) <= 1 and abs(xs.min() - 10) <= 1


def test_outline_is_dark_and_hugs_the_letters():
    r = _bake("plate_in_type", LET)
    f = r["footage"][0].astype(int)
    assert f[16, 24].sum() < np.array(FLAT).sum() * 0.6                       # just below the letters: darkened (outline)
    assert np.abs(f[40, 44] - np.array(FLAT)).max() <= 2                      # far from letters: untouched flat


def test_window_subject_front_picture():
    r = PLT.bake(PLATE, CLEAN, MATTE, mode="window_subject_front", letters=lambda j: LET2, color=False, flat_rgb=(10, 10, 12), word_rgb=WORDC, fill_tint=0.0, guard=False, edge_line=False, **IDENT)
    f = r["footage"][1].astype(int)
    assert np.abs(f[34, 40] - np.array((30, 120, 200))).max() <= 2            # letters outside the subject: windows onto the footage (plate)
    assert np.abs(f[5, 5] - np.array((10, 10, 12))).max() <= 2                 # flat dark ground
    assert np.abs(f[34, 20] - np.array((220, 180, 60))).max() <= 2            # the subject stands IN FRONT of the letters (clean clip x matte)
    assert np.abs(f[24, 20] - np.array((220, 180, 60))).max() <= 2             # and in front of the flat ground
    assert r["matte"].max() == 0
    assert f[38, 40].sum() > 3 * 10 + 60                                       # a light outline hugs the letters (ground is near black)


# ---- occlusion guard + fill tint (2026-10-30)
LETS = np.zeros((H, W), np.float32)
LETS[8:16, 4:12] = 1.0                  # three letters, the middle one behind the subject block (cols 10:30 x rows 20:50 -> move the block up to row 8 below)
LETS[8:16, 16:24] = 1.0
LETS[8:16, 32:40] = 1.0
MATTE_TOP = np.zeros((T, H, W), np.float32)
MATTE_TOP[:, 6:30, 16:24] = 1.0         # covers exactly the middle letter (100 %), the word share is 1/3


def test_occlusion_share_measures_word_and_worst_letter():
    sh = PLT.occlusion_share([LETS] * 3, [MATTE_TOP[0]] * 3)
    assert abs(sh["word"] - 1.0 / 3.0) < 0.01 and sh["letter"] > 0.99 and sh["frames"] == 3
    clear = PLT.occlusion_share([LET] * 2, [MATTE[0]] * 2)                     # a word above the subject: nothing covered
    assert clear["word"] == 0.0 and clear["letter"] == 0.0
    assert PLT.occlusion_share([np.zeros((H, W), np.float32)], [MATTE[0]]) == {"word": 0.0, "letter": 0.0, "frames": 0}


def test_occlusion_decision_keep_relayout_swap():
    ok = PLT.occlusion_decision({"word": 0.2, "letter": 0.5}, "window_subject_front")
    assert ok["action"] == "keep" and ok["mode"] == "window_subject_front"
    bad = {"word": 0.4, "letter": 0.5}
    d = PLT.occlusion_decision(bad, "window_subject_front", can_relayout=True)
    assert d["action"] == "bigger_layout" and d["layout"] == "grid_fill" and d["mode"] == "window_subject_front" and "40%" in d["reason"]
    d = PLT.occlusion_decision(bad, "plate_in_type")
    assert d["action"] == "swap" and d["mode"] == "plate_in_subject"
    d = PLT.occlusion_decision({"word": 0.1, "letter": 0.8}, "plate_in_type")         # one letter over 70 % is enough
    assert d["action"] == "swap"
    assert PLT.occlusion_decision({"word": 0.9, "letter": 0.9}, "plate_in_subject")["action"] == "keep"     # not a guarded mode


def test_bake_guard_swaps_a_covered_word_and_records_it():
    r = PLT.bake(PLATE, CLEAN, MATTE, mode="window_subject_front", letters=lambda j: LET2, color=False, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, edge_line=False, **IDENT)
    occ = r["info"]["occlusion"]
    assert occ["action"] == "swap" and occ["mode"] == "plate_in_subject" and occ["share"]["word"] > 0.35
    f = r["footage"][1].astype(int)
    assert np.abs(f[34, 20] - np.array((30, 120, 200))).max() <= 2             # baked as plate_in_subject: the silhouette shows the plate
    assert np.abs(f[34, 40] - np.array(WORDC)).max() <= 2                      # the word is flat colour, not a window
    r2 = PLT.bake(PLATE, CLEAN, MATTE, mode="window_subject_front", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, edge_line=False, **IDENT)
    assert r2["info"]["occlusion"]["action"] == "keep"                          # a word clear of the subject stays a window section
    r3 = PLT.bake(PLATE, CLEAN, MATTE, mode="plate_in_type", letters=lambda j: LET2, color=False, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, can_relayout=True, edge_line=False, **IDENT)
    assert r3["info"]["occlusion"]["action"] == "bigger_layout" and r3["info"]["type_matte"] == "plate_in_type"


def test_fill_tint_mixes_the_ink_into_the_fills():
    plain = PLT.bake(PLATE, CLEAN, MATTE, mode="plate_in_subject", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=(200, 100, 50), fill_tint=0.0, guard=False, edge_line=False, **IDENT)
    tint = PLT.bake(PLATE, CLEAN, MATTE, mode="plate_in_subject", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=(200, 100, 50), fill_tint=0.2, guard=False, edge_line=False, **IDENT)
    a, b = plain["footage"][0][40, 20].astype(float), tint["footage"][0][40, 20].astype(float)       # inside the silhouette: the plate (30,120,200)
    want = np.array((30, 120, 200)) * 0.8 + np.array((200, 100, 50)) * 0.2
    assert np.abs(a - np.array((30, 120, 200))).max() <= 2 and np.abs(b - want).max() <= 2
    default = PLT.bake(PLATE, CLEAN, MATTE, mode="plate_in_subject", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=(200, 100, 50), guard=False, edge_line=False, **IDENT)
    assert abs(PLT.FILL_TINT - 0.2) < 1e-9 and np.abs(default["footage"][0][40, 20].astype(float) - want).max() <= 2        # default on
    assert tint["info"]["fill_tint"] == 0.2


def test_shot_keys_flat_and_fill_tint():
    k = PLT.shot_keys({"type_matte": "window_subject_front", "flat": "Light", "fill_tint": 0.9})
    assert k["flat"] == "light" and k["fill_tint"] == 0.5
    assert "flat" not in PLT.shot_keys({"type_matte": "reveal_subject", "flat": "light"})
    assert {"flat", "fill_tint"} <= set(PLT.SHOT_KEYS)


# ---- animated edge line + text_as_mask (own-video v006)
def test_edge_line_follows_the_matte_is_crisp_and_travels():
    sq = np.zeros((H, W), np.float32)
    sq[20:50, 10:30] = 1.0
    a0 = PLT.edge_line_alpha(sq, 0)
    a4 = PLT.edge_line_alpha(sq, 4)
    assert a0.max() > 0.9 and np.abs(a0 - a4).sum() > 5.0                       # drawn, and the dash pattern moved with the frame number
    ys, xs = np.where(a0 > 0.2)
    assert ys.min() >= 18 and ys.max() <= 51 and xs.min() >= 8 and xs.max() <= 31   # hugs the contour of the mask, 2 px wide: nothing far from it
    assert a0[30:40, 14:26].max() == 0.0                                        # nothing inside the silhouette
    assert int((a0[14:27, 15] > 0.05).sum()) <= 5                               # crisp: a 2 px line (+ anti-aliased rim) across the top edge, no blur ramp
    big = np.zeros((H, W), np.float32)
    big[10:40, 5:25] = 1.0                                                      # the matte moved: the line moved with it
    assert PLT.edge_line_alpha(big, 0)[44:50, :].max() == 0.0


def test_edge_line_is_drawn_on_the_contrast_modes_by_default():
    """the default edge animation is the brush-stroke `sweep` (contour_sweep.edge_alpha, 2026-10-30); `edge_mode="dash"` keeps the earlier travelling dashes."""
    kw = dict(mode="plate_in_subject", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=(250, 250, 250), fill_tint=0.0, guard=False, **IDENT)
    r = PLT.bake(PLATE, CLEAN, MATTE, edge_mode="dash", **kw)
    f = r["footage"][0].astype(int)
    assert f[19:22, 10:30].min(axis=2).max() > 200                              # the silhouette's top edge carries the (white) ink line (a dash of it)
    r2 = PLT.bake(PLATE, CLEAN, MATTE, edge_line=False, **kw)
    assert np.abs(r2["footage"][0][24, 20].astype(int) - np.array((30, 120, 200))).max() <= 2
    assert not np.array_equal(r["footage"][0], r["footage"][2])                 # animated: the dashes slid between frame 0 and 2
    T2 = 40
    P2, C2, M2 = np.repeat(PLATE[:1], T2, 0), np.repeat(CLEAN[:1], T2, 0), np.repeat(MATTE[:1], T2, 0)
    rs = PLT.bake(P2, C2, M2, **kw)                                             # default = sweep
    near = lambda fr: int((fr.astype(int)[16:54, 6:34].min(axis=2) > 200).sum())
    on = [near(rs["footage"][j]) for j in range(T2)]
    assert max(on[12:28]) >= 12, "brush stroke ink hugs the silhouette mid-section"
    assert on[0] < max(on[12:28]) and on[-1] < max(on[12:28]), "it draws on at the section start and off at its end"
    assert not np.array_equal(rs["footage"][14], rs["footage"][20])             # animated


def test_text_as_mask_letters_show_the_footage_and_no_subject_on_top():
    r = PLT.bake(PLATE, CLEAN, MATTE, mode="text_as_mask", letters=lambda j: LET2, color=False, flat_rgb=(10, 10, 12), word_rgb=WORDC, fill_tint=0.0, guard=True, edge_line=False, outline=False, **IDENT)
    f = r["footage"][1].astype(int)
    assert r["info"]["occlusion"] is None                                       # not a guarded mode: nothing stands in front of the letters
    assert np.abs(f[34, 20] - np.array((220, 180, 60))).max() <= 2              # letter interior over the subject: the FOOTAGE (the man shows through the letter)
    assert np.abs(f[34, 40] - np.array((30, 120, 200))).max() <= 2              # letter interior elsewhere: the footage (room)
    assert np.abs(f[45, 20] - np.array((10, 10, 12))).max() <= 2                # the subject outside the letters is NOT drawn: flat ground
    rp = PLT.bake(PLATE, CLEAN, MATTE, mode="text_as_mask", letters=lambda j: LET2, color=False, flat_rgb=(10, 10, 12), word_rgb=WORDC, fill_tint=0.0, edge_line=False, outline=False, mask_fill="plate", **IDENT)
    assert np.abs(rp["footage"][1][34, 20].astype(int) - np.array((30, 120, 200))).max() <= 2     # mask_fill = plate: the clean plate in the letters
    re_ = PLT.bake(PLATE, CLEAN, MATTE, mode="text_as_mask", letters=lambda j: LET2, color=False, flat_rgb=(10, 10, 12), word_rgb=(250, 250, 250), fill_tint=0.0, outline=False, **IDENT)
    assert re_["footage"][1][29:31, 20].max() > 200                              # the edge line runs along the letter edge (top of the letter row 30)


# ---------------------------------------------------------------- no flat fills inside the subject's silhouette (2026-10-30)
def _moving(seed, T_=T + 6, std=40.0):
    r = np.random.RandomState(seed)
    return np.clip(128 + r.randn(T_, H, W, 3) * std, 0, 255).astype(np.uint8)      # new noise every frame = textured AND moving


def _bake_fill(plate, clean, **kw):
    return PLT.bake(plate, clean, MATTE_T, mode="plate_in_subject", letters=lambda j: LET, color=False, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, guard=False, edge_line=False, **IDENT, **kw)


MATTE_T = np.zeros((T + 6, H, W), np.float32)
MATTE_T[:, 20:50, 10:30] = 1.0


def test_flat_plate_falls_back_to_the_footage():
    flat_plate = np.full((T + 6, H, W, 3), 200, np.uint8)                 # pale, still: reads as a flat fill
    clean = _moving(1)
    r = _bake_fill(flat_plate, clean)
    fi = r["info"]["fill"]
    assert fi["fill_source"] == "footage", fi
    assert fi["measured"]["plate"]["texture"] < PLT.FILL_MIN_TEXTURE and fi["measured"]["plate"]["motion"] < PLT.FILL_MIN_MOTION
    assert fi["measured"]["footage"]["texture"] >= PLT.FILL_MIN_TEXTURE and fi["measured"]["footage"]["motion"] >= PLT.FILL_MIN_MOTION
    f = r["footage"][2].astype(int)
    assert np.abs(f[40, 20] - clean[2, 40, 20].astype(int)).max() <= 2     # inside the silhouette: the shot's own moving footage, NOT the pale plate, not a palette colour
    assert np.abs(f[40, 20] - np.array((200, 200, 200))).max() > 10


def test_textured_moving_plate_is_kept():
    plate = _moving(2)
    r = _bake_fill(plate, _moving(3))
    fi = r["info"]["fill"]
    assert fi["fill_source"] == "plate", fi
    f = r["footage"][2].astype(int)
    assert np.abs(f[40, 20] - plate[2, 40, 20].astype(int)).max() <= 2


def test_still_footage_and_flat_plate_use_the_other_clip():
    flat_plate = np.full((T + 6, H, W, 3), 200, np.uint8)
    still = np.repeat(_moving(4, 1), T + 6, axis=0)                       # textured but not moving
    other = _moving(5)
    r = _bake_fill(flat_plate, still, other_clip=other)
    assert r["info"]["fill"]["fill_source"] == "other_clip", r["info"]["fill"]
    assert "other_clip" in r["info"]["fill"]["measured"]


def test_fill_source_can_be_forced_and_is_never_a_flat_colour():
    r = _bake_fill(_moving(6), _moving(7), fill_source="footage")
    assert r["info"]["fill"]["fill_source"] == "footage"
    pk = PLT.pick_fill({"plate": {"texture": 0.0, "motion": 0.0}, "footage": {"texture": 1.0, "motion": 0.1}})
    assert pk["fill_source"] == "footage" and pk["fill_source"] in PLT.FILL_SOURCES      # nothing above the floors: the best candidate, never a palette fill


def test_subject_is_drawn_at_full_strength_in_the_contrast_modes():
    clean = CLEAN.copy()
    plate = PLATE.copy()
    plate[...] = (200, 40, 40)                                            # a very different plate: the colour fit would gain/offset the subject toward it
    out = PLT.bake(plate, clean, MATTE, mode="window_subject_front", letters=lambda j: LET, color=True, flat_rgb=FLAT, word_rgb=WORDC, fill_tint=0.0, guard=False, edge_line=False, outline=False, **IDENT)
    assert out["info"]["subject_strength"] == "full"
    assert np.abs(out["footage"][1][40, 20].astype(int) - np.array((220, 180, 60))).max() <= 1     # the subject's own pixel value, no dimming / desaturation
