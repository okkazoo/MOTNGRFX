"""the page's frame design -> render plan (tools/app/frame_render.py): layout, word windows, moves. No GPU, no Blender."""
import numpy as np
import pytest

from tools.app import frame_render as R

FONT = "EricaOne-Regular.ttf"


def shot(n, a, b, line="WORD", **tl):
    return dict(n=n, line=line, font=FONT, src_start=a / 25, src_end=b / 25, tl=tl)


def test_has_frame_design():
    assert not R.has_frame_design(None)
    assert not R.has_frame_design(dict(shots=[dict(line="A")]))          # words, but nothing laid out
    assert not R.has_frame_design(dict(shots=[dict(line="", tl=dict(s=2))]))
    assert R.has_frame_design(dict(shots=[dict(line="A", tl=dict(s=2))]))
    assert R.has_frame_design(dict(shots=[dict(line="A", src_start=0.0, src_end=2.0)]))     # an untouched storyboard shot renders as its frame shows it


def test_word_lands_when_sung():
    shots = [shot(1, 0, 100, "SUNG"), shot(2, 100, 200, "FREE")]
    alone, _ = R.plan_words(shots, 25.0, 200)
    words, _ = R.plan_words(shots, 25.0, 200, hits=[dict(land=2.0)])
    assert words[0]["start"] == 50 and words[0]["end"] == alone[0]["end"]     # lands when sung, holds to the end of its shot
    assert words[1]["start"] == alone[1]["start"]                            # no hit in the shot: starts with it
    late, _ = R.plan_words(shots, 25.0, 200, hits=[dict(land=3.99)])
    assert late[0]["start"] == alone[0]["start"]                             # too late to read: the word keeps the whole shot
    clock, _ = R.plan_words([dict(shots[0], start=10.0)], 25.0, 100, hits=[dict(land=12.0)])
    assert clock[0]["start"] == 50                                           # hits are on the piece clock, the frames on the clip's


def test_unliked_font_is_refused():
    with pytest.raises(ValueError):
        R.font_path("Arial.ttf")


@pytest.mark.parametrize("old,new", [("inverse", "mask"), ("nonsense", "front"), (None, "front"), ("maskb", "maskb")])
def test_layer_of(old, new):
    assert R.layer_of(dict(layer=old)) == new


def test_layout_fits_the_page_rule():
    L = R.layout("HOWL", R.font_path(FONT), 1920, 1080, dict())
    assert L["fs"] <= 1920 * 0.26 + 1e-6
    assert L["box"][0] <= 0.88 * 1920 + 1 and L["box"][1] <= 0.84 * 1080 + 1
    assert (L["cx"], L["cy"], L["rot"], L["s"]) == (960, 540, 0, 1)
    assert [l["ch"] for l in L["letters"]] == list("HOWL")


def test_rotation_sizes_along_the_height():
    a, b = R.layout("HOWLING", R.font_path(FONT), 1920, 1080, dict()), R.layout("HOWLING", R.font_path(FONT), 1920, 1080, dict(rot=90))
    assert b["fs"] < a["fs"]                                            # the frame's shorter side now limits it
    assert R.to_frame(dict(rot=90, s=1, cx=100, cy=100), 10, 0) == pytest.approx((100, 110))      # clockwise, like the page's CSS


def test_spread_reaches_the_frame_width():
    L = R.layout("ABC", R.font_path(FONT), 1920, 1080, dict(align="j", s=0.5))
    assert L["box"][0] == pytest.approx(0.88 * 1920 / 0.5)
    assert L["letters"][0]["x"] == pytest.approx(-L["box"][0] / 2)


def test_grid_rows():
    L = R.layout("DUTY", R.font_path(FONT), 1080, 1920, dict(cols=2))
    ys = sorted({round(l["y"]) for l in L["letters"]})
    assert len(ys) == 2 and len(L["letters"]) == 4


def test_anchor_left_against_the_edge():
    L = R.layout("ABC", R.font_path(FONT), 1920, 1080, dict(anch=True, align="l", x=0.9))
    assert L["cx"] < 960


def test_words_stay_clear_of_the_moves_and_merge_across_cuts():
    s = [shot(1, 0, 60, "A"), shot(2, 60, 120, "B"), shot(3, 120, 130, "C"), shot(4, 130, 140, "C"), shot(5, 140, 200, "")]
    words, plan = R.plan_words(s, 25, 200)
    assert 60 in plan and plan[60]["h"] == 7
    assert 130 not in plan                                              # inside the held word
    a, b, c = words
    assert a["end"] <= 60 - plan[60]["h"] and b["start"] >= 60 + plan[60]["h"]
    assert (c["start"], c["end"]) == (120, 140) and c["text"] == "C"


@pytest.mark.parametrize("kind", R.KINDS_LONG)
def test_moves_keep_size_and_land_on_the_pictures(kind):
    A, B = np.full((90, 160, 3), 40, np.uint8), np.full((90, 160, 3), 200, np.uint8)
    for u in (0.02, 0.3, 0.49, 0.51, 0.8, 0.98):
        assert R.move_frame(kind, u, A, B).shape == A.shape
    assert abs(float(R.move_frame(kind, 0.001, A, B).mean()) - 40) < 8
    assert abs(float(R.move_frame(kind, 0.999, A, B).mean()) - 200) < 16
