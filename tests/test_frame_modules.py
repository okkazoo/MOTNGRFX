"""the word modules build / take apart the frames' words (tools/app/frame_modules.py): synthetic 540x960 gradient frames, no footage. CPU, no GPU, no Blender."""
import numpy as np
import pytest

from tools.app import frame_lines as FL
from tools.app import frame_modules as FM
from tools.app import frame_render as R
from tools.app import overlay_mode as OM

FPS, N, SIZE = 25, 200, (540, 960)
PS = dict(width=SIZE[0], height=SIZE[1], fps=FPS, frames=N, fps_str="25/1")
ERICA, BUNGEE = "EricaOne-Regular.ttf", "BungeeInline-Regular.ttf"


def bg(f):
    y, x = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    return np.stack([60 + 100 * x / SIZE[0], 40 + 120 * y / SIZE[1], 90 + f % 40 + 0 * x], 2).astype(np.uint8)


def shot(n, a, b, line, font=ERICA, **tl):
    return dict(n=n, line=line, font=font, src_start=a / FPS, src_end=b / FPS, tl=tl)


@pytest.fixture(autouse=True)
def synthetic_video(monkeypatch):
    monkeypatch.setattr(OM, "_frames", lambda video, w, h, limit=None, bt601=False: (bg(f) for f in range(N)))


def build(shots, modules=True, cuts=None, seed=5):
    words, _ = R.plan_words(shots, FPS, N, None, moves=False)
    objs = [R.Word(w, R.layout(w["text"], R.font_path(w["font"]), *SIZE, w["tl"]), SIZE) for w in words]
    for o in objs:
        o.module = o.fm_in = o.fm_out = None
        o.colours(bg(o.w["start"]))                                                        # the colours of the word's own first frame, in both versions
    if modules:
        FM.plan(objs, "synthetic", PS, cuts or {}, seed)
    FL.plan(objs, FPS, SIZE, cuts or {}, seed)
    return objs


def render(objs, frames=None):
    return {f: im for f, im in R.compose_frames(objs, "synthetic", PS, None, frames) if im is not None}


THREE = [shot(1, 30, 70, "SOLAR", rot=0, s=1.0), shot(2, 70, 140, "RISE", font=BUNGEE, rot=90, s=0.8, y=0.4), shot(3, 140, 200, "ROOT", s=0.9, y=0.6)]


def test_every_word_gets_a_module_that_matches_its_gui_letters_and_neighbours_differ():
    on = build(THREE)
    assert all(o.module in FM.POOL for o in on)
    assert all(on[i].module != on[i + 1].module for i in range(2))
    assert all(o.fm["iou"] >= FM.IOU_MIN for o in on)


def test_tl_module_is_a_user_override():
    s = [dict(x, tl=dict(x["tl"], module="slats")) for x in THREE[:1]] + THREE[1:]
    on = build(s)
    assert on[0].module == "slats"


def test_construction_ends_on_the_land_frame_and_the_formed_word_is_the_gui_fill():
    on = build(THREE)
    off = build(THREE, modules=False)
    gaps = FM.gaps_of(on)
    assert gaps[0].kind == "enter" and gaps[0].b is on[0]
    for g in gaps:
        if g.b is not None:
            assert g.we == g.L == g.b.w["start"]                                          # the sung frame never moves
    win = FM.index(on)
    held = [f for f in range(0, N, 7) if f not in win and f not in FL.index(on) and f not in FL.index(off)]
    inwin = sorted(win)[::6]
    fa, fb = render(on, set(held) | set(inwin)), render(off, set(held) | set(inwin))
    assert held and all(np.array_equal(fa[f], fb[f]) for f in held)                        # outside the windows the GUI word is untouched
    assert any(not np.array_equal(fa[f], fb[f]) for f in inwin)


def test_window_frames_are_not_confined_to_the_gui_box():
    on = build(THREE)
    off = build(THREE, modules=False)
    g = [g for g in FM.gaps_of(on) if g.kind == "pair"][0]
    f = g.ws + (g.L - g.ws) // 2
    fa, fb = render(on, {f}), render(off, {f})
    diff = (fa[f] != fb[f]).any(2)
    flat = on[1].flat.mask(*SIZE) > 0.05
    assert diff.sum() > 100 and (diff & ~flat).sum() > 100                                # strands run outside the word's own box


def test_deterministic_and_report_lists_modules():
    a, b = build(THREE), build(THREE)
    assert [o.module for o in a] == [o.module for o in b]
    assert FM.describe(a) == FM.describe(b)
    rep = R.report(a, None, {})
    assert [w["module"] for w in rep["words"]] == [o.module for o in a] and rep["modules"]


def test_a_picture_move_drops_the_gap_and_metal_stays_flat():
    near = build(THREE, cuts={70: dict(kind="flip", h=7)})
    assert not [g for g in FM.gaps_of(near) if g.kind == "pair" and g.b is near[1]]
    s = [THREE[0], dict(THREE[1], tl=dict(THREE[1]["tl"], metal="gold")), THREE[2]]
    m = build(s)
    assert m[1].module is None


def test_modules_off_is_the_flat_route():
    off = build(THREE, modules=False)
    assert FM.gaps_of(off) == [] and all(o.module is None for o in off)
