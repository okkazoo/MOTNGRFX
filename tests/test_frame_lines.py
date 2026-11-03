"""the library strands between the frames' words (tools/app/frame_lines.py): synthetic 540x960 gradient frames, no footage, one synthetic matte. CPU, no GPU, no Blender."""
import cv2
import numpy as np
import pytest

from tools.app import frame_lines as FL
from tools.app import frame_render as R
from tools.app import overlay_mode as OM

FPS, N, SIZE = 25, 150, (540, 960)
PS = dict(width=SIZE[0], height=SIZE[1], fps=FPS, frames=N, fps_str="25/1")
ERICA, BUNGEE = "EricaOne-Regular.ttf", "BungeeInline-Regular.ttf"


def bg(f):
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    return np.stack([60 + 100 * x / SIZE[0], 40 + 120 * y / SIZE[1], 90 + f % 40 + 0 * x], 2).astype(np.uint8)


class HalfMatte:
    """a synthetic subject: the left half of the frame, every frame."""
    def get(self, f):
        m = np.zeros((SIZE[1], SIZE[0]), np.float32)
        m[:, :SIZE[0] // 2] = 1.0
        return m


def shot(n, a, b, line, font=ERICA, **tl):
    return dict(n=n, line=line, font=font, src_start=a / FPS, src_end=b / FPS, tl=tl)


def build(shots, lines, cuts=None, hits=None):
    words, _ = R.plan_words(shots, FPS, N, hits, moves=False)
    objs = [R.Word(w, R.layout(w["text"], R.font_path(w["font"]), *SIZE, w["tl"]), SIZE) for w in words]
    if lines:
        FL.plan(objs, FPS, SIZE, cuts or {}, seed=5)
    return objs


@pytest.fixture(autouse=True)
def synthetic_video(monkeypatch):
    monkeypatch.setattr(OM, "_frames", lambda video, w, h, limit=None, bt601=False: (bg(f) for f in range(N)))


def render(objs, mattes=None):
    return [im for _, im in R.compose_frames(objs, "synthetic", PS, mattes)]


TWO = [shot(1, 0, 70, "SOLAR", rot=0, s=1.0), shot(2, 70, 150, "RISE", font=BUNGEE, rot=90, s=0.8, y=0.4)]


def test_window_keeps_frame_count_and_the_land_frame():
    on, off = build(TWO, True), build(TWO, False)
    g = on[1].gin
    from mograph.modules.sequence import OUTLINE_KINDS
    assert g.kind in OUTLINE_KINDS
    assert on[1].w["start"] == off[1].w["start"] == g.L == 70                       # the sung frame never moves
    assert len(render(on)) == N
    assert 70 - g.ws - g.ex >= round(FL.T_MIN * FPS) and 70 - g.ws - g.ex <= round(FL.T_MAX * FPS)
    assert g.ws >= on[0].w["start"] + round(FL.HOLD_S * FPS)                          # A held still before it released


def test_after_the_form_in_it_is_the_plain_word():
    on, off = build(TWO, True), build(TWO, False)
    a, b = render(on), render(off)
    g = on[1].gin
    for f in list(range(g.we, on[-1].gout.ws)) + list(range(on[0].gin.we, g.ws)):
        assert np.array_equal(a[f], b[f]), f
    assert any(not np.array_equal(a[f], b[f]) for f in range(g.ws, g.we))


def _near(mask_a, mask_b, tol):
    d = cv2.distanceTransform((mask_b < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    return float((d[mask_a] <= tol).mean())


def test_strands_start_on_a_and_land_on_b():
    on = build(TWO, True)
    g = on[1].gin
    frames = render(on)
    cap = 0.72 * on[0].lay["fs"] * on[0].lay["s"]
    first = (frames[g.ws + g.ex] != bg(g.ws + g.ex)).any(2)                             # A's fill is gone, the strands left
    assert first.sum() > 200
    assert _near(first, on[0].flat.mask(*SIZE), 0.06 * cap + 6) > 0.9
    last = (frames[g.L] != bg(g.L)).any(2)                                              # B's land frame: the arcs just off its outline, no fill yet
    assert last.sum() > 200
    assert _near(last, on[1].flat.mask(*SIZE), 0.06 * on[1].lay["fs"] * on[1].lay["s"] + 6) > 0.9
    def centre(f):
        m = (frames[f] != bg(f)).any(2)
        return np.argwhere(m).mean(0), int(m.sum())
    c0, _ = centre(g.ws + g.ex)
    c1, n1 = centre(g.ws + g.ex + 22)
    assert n1 > 100 and np.linalg.norm(c1 - c0) > 30                                    # on the way: the strands have moved off A's outline


def test_too_short_a_gap_stays_a_plain_cut():
    s = [shot(1, 0, 20, "ONE"), shot(2, 20, 40, "TWO", font=BUNGEE), shot(3, 40, 150, "THREE", rot=90)]
    on, off = build(s, True), build(s, False)
    assert on[1].gin.kind == "cut"
    a, b = render(on), render(off)
    assert all(np.array_equal(a[f], b[f]) for f in range(20, 40))


def test_deterministic_and_neighbouring_gaps_never_repeat():
    s = [shot(1, 0, 60, "ONE"), shot(2, 60, 120, "TWO", font=BUNGEE), shot(3, 120, 150, "THREE", rot=90)]
    runs = [build(s, True) for _ in range(2)]
    kinds = [[g.kind for g in FL.gaps_of(o)] for o in runs]
    assert kinds[0] == kinds[1] and runs[0][1].gin.kind != runs[0][2].gin.kind
    fa, fb = render(runs[0]), render(runs[1])
    assert all(np.array_equal(x, y) for x, y in zip(fa, fb))


def test_first_word_forms_and_last_word_fades_out_in_place():
    on = build(TWO, True)
    gaps = FL.gaps_of(on)
    assert gaps[0].kind in ("form", "enter") and gaps[0].b is on[0]
    assert gaps[-1].kind == "exit" and gaps[-1].we == on[-1].w["end"] and gaps[-1].ws >= on[1].gin.we + round(FL.HOLD_S * FPS)
    frames, off = render(on), render(build(TWO, False))
    g = gaps[-1]
    assert not np.array_equal(frames[g.we - 2], off[g.we - 2])


def test_strands_stay_behind_the_subject_matte():
    s = [dict(TWO[0], tl=dict(layer="behind")), dict(TWO[1], tl=dict(layer="behind", rot=90, s=0.8))]
    on = build(s, True)
    g = on[1].gin
    f = g.ws + g.ex + 8
    im = render(on, HalfMatte())[f]
    assert np.array_equal(im[:, :SIZE[0] // 2], bg(f)[:, :SIZE[0] // 2])                # nothing over the subject
    assert (im[:, SIZE[0] // 2:] != bg(f)[:, SIZE[0] // 2:]).any()                      # strands still run on the other half


def test_a_picture_move_drops_the_strands_for_that_gap():
    near = build(TWO, True, cuts={70: dict(kind="flip", h=7)})
    assert near[1].gin.kind == "cut" and "move" in near[1].gin.why
    far = build(TWO, True, cuts={140: dict(kind="flip", h=3)})
    assert far[1].gin.kind not in ("cut",)


def test_report_lists_every_gap_and_the_cli_flag():
    on = build(TWO, True)
    rep = R.report(on, None, {})
    assert [x["kind"] for x in rep["lines"]] == [g.kind for g in FL.gaps_of(on)] and rep["lines"][1]["window"] == [on[1].gin.ws, on[1].gin.we]
    with pytest.raises(SystemExit):
        R.main(["--help"])
