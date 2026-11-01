"""v005 round 3 (2026-10-27): the type colour is CHOSEN FROM THE MEASURED BACKGROUND (tools/fusion/legible_color.py).
Synthetic frames (numpy backend, one torch-CPU parity test), no GPU: bright bg -> dark text, dark bg -> light text, mid-grey passes via a lightness shift,
a busy checkerboard fires support, the designed hue is kept when it already passes, the colour does not flicker on a flickering plate, transitions ease,
the beauty shading survives, the opt-out, the report, and the recipe wiring (flat + multiplane)."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import compose as C  # noqa: E402
from tools.fusion import legible_color as LC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

H, W, N = 160, 96, 24
CYAN = np.array([0.15, 0.85, 0.95], np.float32)
KEY = "t:lc"
ONE_WORD = [[0.0, 0.0, 1.0, 1.0, False]]          # [start, hit, release, end, fixed]: one word, so the outline budget ceil(1 / 3) = 1


def _alpha():
    a = np.zeros((H, W), np.float32)
    a[50:110, 20:76] = 1.0
    return a


def _face(col=CYAN):
    return np.zeros((H, W, 3), np.float32) + np.asarray(col, np.float32)


def _plate(v):
    return np.full((H, W, 3), v, np.float32)


def _run(plate, face=None, alpha=None, frames=8, key=KEY, plates=None, **kw):
    """n frames through the op -> last RGBA; LC_LOG[key] holds the per-frame readout."""
    LC.reset(key)
    face = _face() if face is None else face
    alpha = _alpha() if alpha is None else alpha
    outs = []
    for i in range(frames):
        p = plate if plates is None else plates[i]
        outs.append(LC.legible_color(face, alpha, p, t=i / 24.0, frame_idx=i, u=0.5, fps=24, state_key=key, **kw))
    return outs


def _core_lum(rgba, alpha=None):
    m = (_alpha() if alpha is None else alpha) > 0.5
    lin = LC.srgb_to_lin(rgba[..., :3][m])
    return float(LC.rel_lum(lin).mean())


def _ratio(rgba, plate_v):
    return float(LC.contrast_ratio(_core_lum(rgba), float(LC.rel_lum(LC.srgb_to_lin(np.full(3, plate_v))))))


def _hue(rgb):
    lab = LC.lin_to_oklab(LC.srgb_to_lin(np.asarray(rgb, np.float32)))
    return float(np.degrees(np.arctan2(lab[2], lab[1])))


# ----------------------------------------------------------------------------------------------------------- the maths
def test_wcag_contrast_and_luminance():
    assert LC.contrast_ratio(1.0, 0.0) == pytest.approx(21.0, abs=1e-4)
    assert float(LC.rel_lum(LC.srgb_to_lin(np.array([1.0, 1.0, 1.0])))) == pytest.approx(1.0, abs=1e-4)
    assert float(LC.rel_lum(LC.srgb_to_lin(np.array([0.5, 0.5, 0.5])))) == pytest.approx(0.2140, abs=2e-3)
    rt = LC.oklab_to_lin(LC.lin_to_oklab(np.array([0.2, 0.5, 0.7])))
    assert np.allclose(rt, [0.2, 0.5, 0.7], atol=1e-5)
    assert LC.target_for(60.0) == 3.0 and LC.target_for(20.0) == 4.5


# ----------------------------------------------------------------------------------------------------------- choice
def test_bright_background_gets_dark_text():
    out = _run(_plate(0.92))[-1]
    assert _core_lum(out) < 0.25
    assert _ratio(out, 0.92) >= 3.0
    assert LC.LC_LOG[KEY][7][0]["kind"] in ("lightness", "hue", "kept", "extreme")


def test_dark_background_gets_light_text():
    face = _face([0.10, 0.12, 0.30])                       # a dark designed navy on a near-black plate
    out = _run(_plate(0.04), face=face)[-1]
    assert _core_lum(out) > 0.1, "lightened from the dark designed navy (the least shift that reaches the target)"
    assert _ratio(out, 0.04) >= 3.0


def test_mid_grey_passes_through_a_lightness_shift_keeping_the_hue():
    out = _run(_plate(0.45))[-1]
    assert _ratio(out, 0.45) >= 3.0, "WCAG large-text target reached on mid-grey"
    rec = LC.LC_LOG[KEY][7][0]
    assert rec["kind"] in ("lightness", "kept") and not rec["support"]
    dh = abs((_hue(out[80, 48, :3]) - _hue(CYAN) + 180) % 360 - 180)
    assert dh < 12, f"the designed hue is kept while the lightness shifts ({dh:.1f} deg off)"
    assert _core_lum(out) != pytest.approx(_core_lum(np.concatenate([_face(), _alpha()[..., None]], -1)), abs=0.05)


def test_designed_colour_is_kept_when_it_already_passes():
    out = _run(_plate(0.02))[-1]
    assert LC.LC_LOG[KEY][7][0]["kind"] == "design"
    assert np.abs(out[..., :3][_alpha() > 0.5] - CYAN).max() < 0.02
    assert not LC.LC_LOG[KEY][7][0]["support"]


def test_busy_checkerboard_fires_support_instead_of_recolouring_everything():
    y, x = np.mgrid[0:H, 0:W]
    levels = np.array([0.0, 0.30, 0.58, 1.0], np.float32)
    plate = np.repeat(levels[((y // 6) + (x // 6) * 2) % 4][..., None], 3, -1).astype(np.float32)
    out = _run(plate, frames=10, words=ONE_WORD)[-1]
    rec = LC.LC_LOG[KEY][9][0]
    assert rec["kind"] == "support" and rec["support"], rec
    a = _alpha()
    outside = (cv2.dilate(a, np.ones((9, 9), np.uint8)) > 0) & (a < 0.5)
    assert float(out[..., 3][outside].max()) > 0.2, "the support lives in the extended alpha of the type layer"
    assert float(out[..., 3][a > 0.5].min()) == pytest.approx(1.0)
    # only the minimum support: the scrim / shadow does not reach across the whole frame
    far = cv2.dilate(a, np.ones((61, 61), np.uint8)) == 0
    assert float(out[..., 3][far].max()) < 0.01
    # a plain plate fires nothing
    _run(_plate(0.9))
    assert not LC.LC_LOG[KEY][7][0]["support"]


def test_support_escalates_outline_shadow_scrim_by_the_residual():
    cn = LC._centres()
    design = LC.srgb_to_lin(np.array([0.5, 0.5, 0.5]))
    mid = int(np.argmin(np.abs(cn - 0.17)))
    kinds = []
    dflt = []
    for dark, bright, midp in ((0.8, 0.2, 0.0), (0.55, 0.45, 0.0), (0.34, 0.33, 0.33)):      # residual (best colour fails) ~ 0.2, 0.45, 0.67
        hist = np.zeros(LC.LC_NB, np.float32)
        hist[0], hist[-1], hist[mid] = dark, bright, midp
        ch = LC.choose(hist, design, np.array([0.3, 0.3, 0.3]), 4.5, soft=True)          # soft support only when asked for (effect soft_support)
        kinds.append(ch["support"])
        dflt.append(LC.choose(hist, design, np.array([0.3, 0.3, 0.3]), 4.5)["support"])      # default (user 2026-10-30: pure type): never a soft shadow / scrim; the outline only when even the extreme ink fails 3:1 over > 40 %
    assert kinds == [None, "shadow", "scrim"], kinds
    assert dflt == [None, "outline", None], dflt


def test_small_type_targets_4_5_and_large_type_3():
    h2, w2 = 340, 192
    a = np.zeros((h2, w2), np.float32)
    for x0 in range(20, 160, 14):
        a[100:104, x0:x0 + 8] = 1.0                        # cap height 4 px -> ~22 px at 1080: small
    LC.reset(KEY)
    LC.legible_color(np.zeros((h2, w2, 3), np.float32) + CYAN, a, np.full((h2, w2, 3), 0.3, np.float32), t=0.0, frame_idx=0, fps=24, state_key=KEY)
    assert LC.LC_LOG[KEY][0][0]["target"] == 4.5
    _run(_plate(0.3), frames=1)
    assert LC.LC_LOG[KEY][0][0]["target"] == 3.0


# ----------------------------------------------------------------------------------------------------------- time
def test_flickering_background_does_not_flicker_the_text():
    plates = [_plate(0.30 if i % 2 else 0.55) for i in range(24)]
    lum = lambda outs: np.array([_core_lum(o) for o in outs[8:]])       # noqa: E731
    smooth = lum(_run(None, frames=24, plates=plates))
    raw = lum(_run(None, frames=24, plates=plates, smooth=0.0))
    assert np.abs(np.diff(smooth)).max() < 0.03, np.abs(np.diff(smooth)).max()
    assert np.abs(np.diff(raw)).max() > 0.2, "without smoothing the colour flips with the plate (the test would not prove anything otherwise)"


def test_a_background_change_eases_the_colour_no_pop():
    plates = [_plate(0.04)] * 8 + [_plate(0.92)] * 16
    lum = np.array([_core_lum(o) for o in _run(None, frames=24, plates=plates)])
    step = np.abs(np.diff(lum))
    assert lum[7] > 0.4 and lum[-1] < 0.25, "light on dark, then dark on bright"
    assert step.max() < 0.75 * abs(lum[7] - lum[-1]), "spread over several frames, not one pop"
    # but it arrives: the final colour reaches the target ratio
    assert _ratio(_run(None, frames=24, plates=plates)[-1], 0.92) >= 3.0


def test_state_restarts_on_a_frame_jump_and_warm_up_matches_a_sequential_run():
    plates = [_plate(0.30 if i % 3 else 0.6) for i in range(30)]
    seq = _run(None, frames=30, plates=plates)
    LC.reset(KEY)
    a, f = _alpha(), _face()
    for i in range(16, 30):                                           # a chunk started at 16 with LC_WARM frames of run-in
        o = LC.legible_color(f, a, plates[i], t=i / 24, frame_idx=i, u=0.5, fps=24, state_key=KEY)
    assert abs(_core_lum(o) - _core_lum(seq[-1])) < 0.05


# ----------------------------------------------------------------------------------------------------------- apply
def test_the_beauty_shading_survives_the_recolour():
    a = _alpha()
    ramp = np.linspace(0.55, 1.0, W, dtype=np.float32)[None, :, None]       # a bevel-like horizontal gradient
    face = (CYAN[None, None, :] * ramp + 0 * _face()).astype(np.float32)
    out = _run(_plate(0.92), face=face)[-1]
    m = a > 0.5
    lf = C._luma(face)[m]
    lo = C._luma(out[..., :3])[m]
    assert np.corrcoef(lf, lo)[0, 1] > 0.97, "the beauty's luma pattern is mapped onto the chosen colour"
    assert lo.std() > 0.01
    assert abs(float(lo.mean()) - float(lf.mean())) > 0.08, "...and the colour itself did change"


def test_opt_out_passes_the_face_through():
    for kw in (dict(enable=False), dict(words=[[0.0, 0.1, 0.3, 0.5, 1]])):
        out = _run(_plate(0.92), **kw)[-1]
        assert np.allclose(out[..., :3], _face()) and np.allclose(out[..., 3], _alpha()), kw
    assert LC.LC_LOG.get(KEY) is None or not LC.LC_LOG[KEY]


def test_no_word_on_screen_is_a_noop():
    out = _run(_plate(0.5), alpha=np.zeros((H, W), np.float32))[-1]
    assert np.allclose(out[..., :3], _face()) and out[..., 3].max() == 0


def test_two_words_get_their_own_colour():
    a = np.zeros((H, W), np.float32)
    a[10:40, 10:86] = 1.0
    a[120:150, 10:86] = 1.0
    plate = np.zeros((H, W, 3), np.float32)
    plate[:80] = 0.92
    plate[80:] = 0.03
    out = _run(plate, alpha=a)[-1]
    top, bot = _core_lum(out, np.pad(a[:80], ((0, 80), (0, 0)))), _core_lum(out, np.pad(a[80:], ((80, 0), (0, 0))))
    assert top < 0.25 < 0.4 < bot, (top, bot)


def test_torch_backend_gives_the_same_answer():
    torch = pytest.importorskip("torch")
    ref = _run(_plate(0.45))[-1]
    with RE.use_backend("torch"):
        raw = LC.legible_color.raw
        LC.reset(KEY)
        for i in range(8):
            o = raw(C._in(_face()), C._in(_alpha()), C._in(_plate(0.45)), t=i / 24.0, frame_idx=i, u=0.5, fps=24, state_key=KEY)
        assert isinstance(o, torch.Tensor)
        o = C._out(o)
    assert np.abs(o - ref).max() < 0.02


# ----------------------------------------------------------------------------------------------------------- report
def test_report_table_has_the_per_word_numbers():
    _run(_plate(0.92), frames=24)
    words = [[0.0, 0.2, 0.8, 0.95, 0]]
    rows = LC.lc_report(KEY, words, 24, ["HELLO"])
    r = rows[0]
    assert r["measured"] and r["word"] == "HELLO" and r["frames"] > 5
    for k in ("bg_lum", "color", "ratio_min", "ratio_mean", "cover_min", "support_fired", "how", "target", "life_ratio_min"):
        assert k in r
    assert r["bg_lum"] > 0.7 and r["ratio_min"] >= 3.0 and r["ratio_mean"] >= r["ratio_min"] and r["support_fired"] is False
    txt = LC.format_table(rows + [dict(word="FIXED", hit=1.0, release=2.0, fixed=True, measured=False)])
    assert "HELLO" in txt and "fixed (opt-out)" in txt and r["color"] in txt


def test_cli_prints_the_table_from_a_compose_report(tmp_path, capsys):
    _run(_plate(0.92), frames=24)
    rows = LC.lc_report(KEY, [[0.0, 0.2, 0.8, 0.95, 0]], 24, ["HELLO"])
    rp = tmp_path / "compose_report.json"
    rp.write_text(json.dumps({"shots": {"s1": {"hold_net": [], "legible_color": rows}}}))
    assert LC.main([str(rp)]) == 0
    assert "HELLO" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------------- recipe wiring
def _write_type(tmp, beauty, matte, n=N):
    for k, im in (("beauty", beauty), ("matte", matte), ("line", matte)):
        d = Path(tmp) / k
        d.mkdir(parents=True, exist_ok=True)
        u8 = np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)
        if u8.ndim == 3:
            u8 = u8[..., ::-1]
        for i in range(n):
            cv2.imwrite(str(d / f"{k}_{i:05d}.png"), u8)


def _ops():
    return [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 1.0}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": [W, H], "guide": "$footage"}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "matte": "$m", "type_a": "$type_matte", "type_rgb": "$tcol", "matte_threshold": 0.5,
                                                    "type_opacity": 1.0}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0}},
    ]


def _piece(ops=None, **comp):
    c = {"crossfade": 0.0, "default": {"ops": ops or _ops()}}
    c.update(comp)
    return {"id": "T", "fps": 24, "format": "9:16", "sizes": {"compose": [W, H]}, "shots": [{"id": "s1", "frames": N, "matte": True}],
            "compose": c, "plan": {"flow_wake": {"holds": [[0.3, 0.9]]}}, "type": {"words": [{"text": "ONE"}]}}


def _type_dir(tmp, beauty_col=CYAN):
    a = _alpha()
    beauty = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.03, 0.05], np.float32)
    beauty = beauty * (1 - a[..., None]) + np.asarray(beauty_col, np.float32) * a[..., None]
    _write_type(tmp, beauty, a)
    return tmp


def _compose(tmp, plate_v, piece=None):
    td = _type_dir(tmp)
    plate = np.clip(np.full((N, H, W, 3), plate_v, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)
    u8, st = RE.run_compose(piece or _piece(), "s1", type_dir=str(td), footage=plate, matte=np.zeros((N, H, W), np.float32), backend="numpy", threads=0)
    return np.asarray(u8, np.uint8).astype(np.float32) / 255.0, st


def test_flat_recipe_inserts_the_op_before_the_drawer_and_draws_its_output(tmp_path):
    (tmp_path / "beauty").mkdir()
    rec = RE.build_recipe(_piece(), "s1", type_dir=str(tmp_path))
    names = [o["op"] for o in rec["ops"]]
    assert names.count("legible_color") == 1
    assert names.index("type_face") < names.index("legible_color") < names.index("sandwich") < names.index("hold_net")
    lc = next(o for o in rec["ops"] if o["op"] == "legible_color")
    assert lc["args"]["face"] == "$tface" and lc["args"]["alpha"] == "$tmat" and lc["args"]["plate"] == "$foot" and lc["args"]["mode"] == "layer"
    sw = next(o for o in rec["ops"] if o["op"] == "sandwich")
    assert sw["args"]["type_rgb"] == "$lcrgb0" and sw["args"]["type_a"] == "$lca0"
    net = next(o for o in rec["ops"] if o["op"] == "hold_net")
    assert net["args"]["face"] == "$lcrgb0" and net["args"]["alpha"] == "$tmat", "the net sees the recoloured face and the plain alpha"
    assert rec["lc"]["key"].startswith("T:s1:lc") and rec["lc"]["words"]


def test_piece_and_word_opt_out_keys(tmp_path):
    (tmp_path / "beauty").mkdir()
    for p in (_piece(color="fixed"), _piece(legible_color=False)):
        assert "legible_color" not in [o["op"] for o in RE.build_recipe(p, "s1", type_dir=str(tmp_path))["ops"]]
    p = _piece()
    p["color"] = "fixed"
    assert "legible_color" not in [o["op"] for o in RE.build_recipe(p, "s1", type_dir=str(tmp_path))["ops"]]
    p = _piece()
    p["type"]["words"] = [{"text": "ONE", "color": "fixed"}]
    rec = RE.build_recipe(p, "s1", type_dir=str(tmp_path))
    lc = next(o for o in rec["ops"] if o["op"] == "legible_color")
    assert [w[4] for w in lc["args"]["words"]] == [1], "the word is passed through, the op stays for the other words"


def test_end_to_end_bright_plate_draws_dark_text_and_reports(tmp_path):
    img, st = _compose(tmp_path / "t", 0.9)
    f = img[int(0.6 * 24)]
    m = _alpha() > 0.5
    assert float(LC.rel_lum(LC.srgb_to_lin(f[m])).mean()) < 0.25, "the cyan beauty is drawn dark on the bright plate"
    rows = st["legible_color"]
    assert rows and rows[0]["measured"] and rows[0]["ratio_min"] >= 3.0 and rows[0]["bg_lum"] > 0.7
    assert rows[0]["word"] == 0 or rows[0]["word"] is not None
    # the same piece on a dark plate keeps the designed cyan
    img2, st2 = _compose(tmp_path / "t2", 0.03)
    f2 = img2[int(0.6 * 24)]
    assert np.abs(f2[m] - CYAN).max() < 0.1
    assert st2["legible_color"][0]["how"] == "design"
    # opt-out: draws the beauty colour on the bright plate
    img3, st3 = _compose(tmp_path / "t3", 0.9, _piece(color="fixed"))
    assert float(LC.rel_lum(LC.srgb_to_lin(img3[int(0.6 * 24)][m])).mean()) > 0.45 and "legible_color" not in st3


def _mp_piece():
    p = _piece([
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "gain": 0.7}},
        {"op": "fill", "out": "tcol", "args": {"like": "$footage", "color": [0.95, 0.95, 0.95]}},
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "matte": "$s1_matte", "type_a": "$type_matte", "type_rgb": "$tcol",
                                                    "matte_threshold": 0.5, "type_opacity": 1.0}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx"}},
    ])
    p["shots"] = [{"id": "s1", "frames": N, "matte": True, "depth": True}]
    p["type"] = {"camera": {"chain": "auto", "plate": "multiplane", "cards": 4}, "words": [], "palette": {"bg": [5, 9, 13]}}
    return p


def test_multiplane_recipe_includes_the_op_and_the_cards_draw_the_recoloured_face(tmp_path):
    td = tmp_path / "type"
    (td / "beauty").mkdir(parents=True)
    (td / "manifest.json").write_text(json.dumps({"spec": {"palette": {"bg": [5, 9, 13]}}, "size": [W, H], "depth": {"near": 500, "far": 1500}}))
    rec = RE.build_recipe(_mp_piece(), "s1", type_dir=str(td))
    names = [o["op"] for o in rec["ops"]]
    assert names.count("legible_color") == 1 and names.count("multiplane") == 1 and names.count("legible_support") == 1
    assert names.index("type_face") < names.index("legible_color") < names.index("multiplane") < names.index("legible_support") < names.index("hold_net")
    lc = next(o for o in rec["ops"] if o["op"] == "legible_color")
    assert lc["args"]["mode"] == "post" and lc["args"]["face"] == "$tface" and lc["args"]["alpha"] == "$tmat" and lc["args"]["state_key"].endswith(":lcmp")
    mp = next(o for o in rec["ops"] if o["op"] == "multiplane")
    assert mp["args"]["type_rgb"] == "$lcmp_beauty" and mp["args"]["type_a"] == "$tmat"
    lay = next(o for o in rec["ops"] if o.get("out") == "lcmp_beauty")
    assert lay["op"] == "over" and lay["args"]["alpha"] == "$tmat" and lay["args"]["fg"] == "$lcmp_face"
    bgfill = next(o for o in rec["ops"] if o.get("out") == "_lcmp_bg")
    assert np.allclose(bgfill["args"]["color"], np.array([5, 9, 13]) / 255.0)
    net = next(o for o in rec["ops"] if o["op"] == "hold_net")
    assert net["args"]["face"] == "$lcmp_face"
    assert not any(o["op"] in ("sandwich", "flow_wake") for o in rec["ops"]) and not any(o.get("out") in ("lcrgb0", "lca0") for o in rec["ops"])
    assert rec["lc"]["key"].endswith(":lcmp")
    # a flat fallback (no depth pass) keeps the flat colour ops intact
    p = _mp_piece()
    p["shots"] = [{"id": "s1", "frames": N, "matte": True}]
    names2 = [o["op"] for o in RE.build_recipe(p, "s1", type_dir=str(td))["ops"]]
    assert "multiplane" not in names2 and "legible_color" in names2 and "sandwich" in names2


def test_legible_support_paints_under_the_type_only():
    key = "sup:lc"
    y, x = np.mgrid[0:H, 0:W]
    levels = np.array([0.0, 0.30, 0.58, 1.0], np.float32)
    plate = np.repeat(levels[((y // 6) + (x // 6) * 2) % 4][..., None], 3, -1).astype(np.float32)
    LC.reset(key)
    for i in range(10):
        LC.legible_color(_face(), _alpha(), plate, t=i / 24, frame_idx=i, fps=24, state_key=key, mode="post", words=ONE_WORD)
    img = np.full((H, W, 3), 0.5, np.float32)
    out = LC.legible_support(img, _alpha(), state_key=key, frame_idx=9)
    assert np.abs(out - img).max() > 0.05, "something was painted near the word"
    assert np.allclose(out[_alpha() > 0.5], 0.5), "never over the type itself"
    assert np.allclose(LC.legible_support(img, _alpha(), state_key=key, frame_idx=3), img), "no decision for that frame: untouched"


# ----------------------------------------------------------------------------------------------------------- 2026-10-30: outline only on the drawn words / as a capped last resort
def _hist_at(*pairs):
    h = np.zeros(LC.LC_NB, np.float32)
    for y, w in pairs:
        h[int(np.clip(np.sqrt(y) * LC.LC_NB, 0, LC.LC_NB - 1))] += w
    return h / h.sum()


def test_a_failing_ink_picks_the_lightness_extreme_without_an_outline():
    design = LC.srgb_to_lin(np.array([0.5, 0.5, 0.5]))
    hist = _hist_at((0.0, 0.8), (1.0, 0.2))                       # light ink reads on 80 % of the surround (>= 3:1), not all of it: the design ink fails the 85 % need
    ch = LC.choose(hist, design, np.array([0.3, 0.3, 0.3]), 4.5)
    assert ch["kind"] == "extreme" and ch["support"] is None, ch
    y = float(LC.rel_lum(ch["lin"]))
    assert y > 0.6 or y < 0.03, f"an extreme of the design hue, got luminance {y:.3f}"


def test_the_outline_is_the_last_resort_only_when_even_the_extreme_ink_fails_the_floor():
    design = LC.srgb_to_lin(np.array([0.5, 0.5, 0.5]))
    hist = _hist_at((0.0, 0.5), (1.0, 0.5))                       # half the surround fails 3:1 whichever extreme is used
    ch = LC.choose(hist, design, np.array([0.3, 0.3, 0.3]), 4.5)
    assert ch["kind"] == "support" and ch["support"] == "outline", ch


def test_a_palette_colour_is_tried_before_the_lightness_shift():
    design = LC.srgb_to_lin(np.array([0.5, 0.5, 0.5]))
    hist = _hist_at((0.02, 1.0))                                   # dark ground
    pal = [LC.srgb_to_lin(np.array([0.05, 0.05, 0.05])), LC.srgb_to_lin(np.array([1.0, 0.9, 0.4]))]      # one palette ink fails, one passes
    ch = LC.choose(hist, design, np.array([0.02, 0.02, 0.02]), 4.5, palette=pal)
    assert ch["kind"] == "palette" and np.allclose(ch["lin"], pal[1], atol=1e-5) and ch["support"] is None
    assert LC.choose(hist, design, np.array([0.02, 0.02, 0.02]), 4.5)["kind"] == "lightness"


def _words(n):
    return [[i + 0.0, i + 0.2, i + 0.8, i + 1.0, False] for i in range(n)]


def _checker():
    y, x = np.mgrid[0:H, 0:W]
    levels = np.array([0.0, 0.30, 0.58, 1.0], np.float32)
    return np.repeat(levels[((y // 6) + (x // 6) * 2) % 4][..., None], 3, -1).astype(np.float32)


def _outlined_words(n, windows, key):
    """every word's hold frame on a plate that defeats any ink: -> the word indices that carry an outline (forced window or adaptive)."""
    LC.reset(key)
    out = []
    for i in range(n):
        f = i * 24 + 12
        LC.legible_color(_face(), _alpha(), _checker(), t=i + 0.5, frame_idx=f, u=0.5, fps=24, state_key=key, words=_words(n), outline=windows, smooth=0.0)
        rec = LC.LC_LOG[key][f][0]
        if "outline" in rec["support"]:
            out.append(i)
        else:
            assert rec["kind"] == "extreme" and not rec["support"], rec          # cap full: the extreme ink stands alone
    return out


def test_outlined_words_never_exceed_ceil_n_over_3_even_on_a_plate_nothing_reads_on():
    for n in (1, 2, 3, 4, 6, 7):
        got = _outlined_words(n, [], f"cap{n}")
        assert len(got) == -(-n // 3), (n, got)


def test_drawn_outline_windows_count_against_the_cap_and_come_first():
    from tools.fusion import random_system as RSY
    n = 7
    wins = RSY.outline_windows({"mode": "hits", "share": 0.34}, [(w[1], w[2]) for w in _words(n)], 100, 5)
    drawn = {int(w[0]) for w in wins}
    got = _outlined_words(n, wins, "capd")
    assert drawn <= set(got) and len(got) <= -(-n // 3), (drawn, got)


def test_no_outline_without_a_failing_plate_or_a_drawn_window():
    LC.reset("plain")
    for i in range(4):
        f = i * 24 + 12
        LC.legible_color(_face(), _alpha(), _plate(0.9), t=i + 0.5, frame_idx=f, u=0.5, fps=24, state_key="plain", words=_words(4), outline=[])
        assert not LC.LC_LOG["plain"][f][0]["support"]
