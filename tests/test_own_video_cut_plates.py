"""own-video plan step: clean plates for the shots either side of a 3D cut that splits the subject (pure python)."""
from tools.app import own_video_v3 as V3

SPANS = [78, 77, 77, 77, 75]


def _piece(*cuts):
    return {"compose": {"cuts": [dict(after=a, cut=c) for a, c in cuts]}}


def test_3d_cuts_get_a_plate_window_on_both_sides():
    w = V3.cut_plate_windows(_piece(("s1", "orbit:pullback"), ("s2", "orbit:push"), ("s3", "depth_pivot"), ("s4", "orbit:fly")), SPANS)
    assert [x["after"] for x in w] == ["s1", "s2", "s3", "s4"]
    x = w[0]
    assert x["shots"] == ["s1", "s2"] and x["boundary"] == 78
    assert x["a"] == 78 - round(V3.CUT_PLATE_PRE_S * 24) and x["b"] == 78 + round(V3.CUT_PLATE_POST_S * 24)
    assert all(0 <= y["a"] < y["boundary"] < y["b"] <= 384 for y in w)      # clipped to the two shots


def test_hard_and_window_cuts_need_no_cut_plate():
    assert V3.cut_plate_windows(_piece(("s1", "hard"), ("s2", "window_cut"), ("s3", "pan:right")), SPANS) == []


def test_pieces_for_moments_takes_the_pieces_under_the_cut_window():
    segs = [dict(person=True, job=1, a=0, b=78), dict(person=True, job=2, a=78, b=81), dict(person=True, job=3, a=81, b=155), dict(person=True, job=9, a=324, b=384)]
    w = V3.cut_plate_windows(_piece(("s1", "orbit:push")), SPANS)
    assert [s["job"] for s in V3.pieces_for_moments(segs, w)] == [1, 2, 3]
