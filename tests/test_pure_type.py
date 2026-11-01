"""User 2026-10-30: "Can we not use any soft blur behind the type. Just have it pure type". Legibility comes from the ink colour (palette contrast floor) plus the occasional thin OUTLINE only: no
soft shadow, no local scrim, no hold-net darkening scrim, no blur / shadow / glow op on the type mask in a default piece. The soft versions stay reachable: effect `soft_support`."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("FUSION_BACKEND", "numpy")

from tools.fusion import effects_menu as EM  # noqa: E402
from tools.fusion import legible_color as LC  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402
import test_hold_legibility as THL  # noqa: E402

SOFT_OPS = ("blur", "drop_shadow", "shadow", "scrim", "add_glow", "halation")


def _recipe(tmp_path, soft=False):
    (tmp_path / "beauty").mkdir(exist_ok=True)
    piece = THL._piece(THL._old_ops())
    if soft:
        piece["compose"]["soft_support"] = True
    return RE.build_recipe(piece, "s1", type_dir=str(tmp_path))


def test_default_recipe_has_no_blur_shadow_scrim_or_glow_on_the_type_mask(tmp_path):
    rec = _recipe(tmp_path)
    type_refs = ("$ta", "$tmat", "$type_matte", "$type_a", "$wm")
    for o in rec["ops"]:
        a = o.get("args", {})
        on_type = any(v in type_refs for k, v in a.items() if k in ("mask", "alpha", "type_a", "img") and isinstance(v, str)) or o["op"] in ("blur", "drop_shadow", "shadow", "scrim")
        assert not (o["op"] in SOFT_OPS and on_type), o
    net = [o for o in rec["ops"] if o["op"] == "hold_net"]
    assert net and not any(o["args"].get("scrim") for o in net)                                            # the hold net restores the plain type but never darkens the surround
    assert not any(o["args"].get("soft_support") for o in rec["ops"] if o["op"] == "legible_color")


def test_soft_support_is_reachable_when_requested(tmp_path):
    rec = _recipe(tmp_path, soft=True)
    assert any(o["args"].get("scrim") for o in rec["ops"] if o["op"] == "hold_net")
    s = EM.apply_effects({"seed": 1, "effects": {}, "outline": {"mode": "none"}}, ["soft_support"])
    assert "soft_support" in s["effects"]["requested"] and "soft_support" in s["effects"]["active"] and EM.EFFECTS["soft_support"]["kind"] == "legibility"


def test_the_support_is_never_more_than_the_thin_outline_whatever_the_residual():
    cn = LC._centres()
    design = LC.srgb_to_lin(np.array([0.5, 0.5, 0.5]))
    mid = int(np.argmin(np.abs(cn - 0.17)))
    for dark, bright, midp in ((0.8, 0.2, 0.0), (0.55, 0.45, 0.0), (0.34, 0.33, 0.33)):                    # residuals ~0.2 / 0.45 / 0.67
        hist = np.zeros(LC.LC_NB, np.float32)
        hist[0], hist[-1], hist[mid] = dark, bright, midp
        assert LC.choose(hist, design, np.array([0.3, 0.3, 0.3]), 4.5)["support"] in (None, "outline")      # never a shadow / scrim; the outline itself only as the last resort (test_legible_color)


def test_a_model_written_glow_behind_the_type_is_dropped_and_the_chain_stays_wired():
    from tools.fusion import design_defaults as DD
    piece = {"compose": {"default": {"ops": [
        {"op": "sandwich", "out": "sw", "args": {"footage": "$f", "matte": "$m", "type_a": "$wm", "type_rgb": "$tcol"}},
        {"op": "add_glow", "out": "glow", "args": {"img": "$sw", "mask": "$wm", "radius": 24.0, "intensity": 0.3}},
        {"op": "add_glow", "out": "hlg", "hl": True, "args": {"img": "$glow", "mask": "$wm", "radius": 10.0, "intensity": 0.2}},
        {"op": "grain", "out": "gr", "args": {"frame": "$hlg"}}]}}}
    notes = DD.enforce_compose(piece, [(1.0, 3.0)])
    ops = piece["compose"]["default"]["ops"]
    assert any("add_glow glow" in n for n in notes) and [o["op"] for o in ops] == ["sandwich", "add_glow", "grain"]
    assert ops[1]["hl"] and ops[1]["args"]["img"] == "$sw"                                  # the hold-life glow (an effect) stays, re-wired to the sandwich
    keep = {"compose": {"type_glow": True, "default": {"ops": [{"op": "add_glow", "out": "g", "args": {"img": "$sw", "mask": "$wm"}}]}}}
    DD.enforce_compose(keep, [(1.0, 3.0)])
    assert keep["compose"]["default"]["ops"][0]["op"] == "add_glow"                         # asked for: kept


def test_the_default_draw_outlines_at_most_ceil_n_over_3_words():
    """2026-10-30 (user: "we don't need to outline them all"): the drawn outline (none | hits) covers at most ceil(n / 3) words, whatever the seed and the word count."""
    from tools.fusion import random_system as RSY
    for n in range(1, 13):
        holds = [(2.0 * i + 0.4, 2.0 * i + 1.8) for i in range(n)]
        for seed in range(40):
            wins = RSY.outline_windows(RSY.draw_outline(seed), holds, 100, seed)
            hit_words = {i for i, (a, b) in enumerate(holds) if any(w[0] < b and w[1] > a for w in wins)}
            assert len(hit_words) <= -(-n // 3), (n, seed, hit_words)
            assert len(wins) == len(hit_words)


def test_the_recipe_hands_legible_color_the_palette_and_only_the_drawn_windows(tmp_path):
    import json
    import test_legible_color as TLC
    td = tmp_path / "type"
    (td / "beauty").mkdir(parents=True)
    words = [dict(text=f"w{i}", start=2.0 * i, hit=2.0 * i + 0.4, release=2.0 * i + 1.6, end=2.0 * i + 2.0) for i in range(7)]
    (td / "manifest.json").write_text(json.dumps({"spec": {"palette": {"bg": [5, 9, 13]}}, "size": [TLC.W, TLC.H], "depth": {"near": 500, "far": 1500}, "info": {"words": words}}))
    p = TLC._mp_piece()
    p["type"]["palette"] = {"bg": [5, 9, 13], "fg": [240, 230, 200], "accents": [[200, 60, 40], [40, 120, 200], [250, 200, 30]]}
    p["plan"] = {"piece_system": {"seed": 5, "outline": {"mode": "hits", "share": 0.34}}}
    rec = RE.build_recipe(p, "s1", type_dir=str(td))
    lc = [o for o in rec["ops"] if o["op"] == "legible_color"]
    assert lc
    for o in lc:
        a = o["args"]
        n = len(a["words"])
        wins = a.get("outline") or []
        covered = {i for i, w in enumerate(a["words"]) if any(x[0] < w[2] and x[1] > w[1] for x in wins)}
        assert 1 <= len(covered) <= -(-n // 3), (n, wins)
        pal = a.get("ink_palette") or []
        assert len(pal) >= 4 and all(0.0 <= c <= 1.0 for col in pal for c in col), pal
