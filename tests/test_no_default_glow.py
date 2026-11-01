"""No glow behind the type unless it is hold-life (user 2026-10-30; the ship-#7 previews v030 r01 / r02 had `add_glow` ops `$_wi2g`, `$_gl0c`, `$_wi4g` in their chains = the matte treatments
`wipe` ring and `glow` outline). A designed piece's chains hold no add_glow except hold-life's (`hl`). Run: python -m pytest tests/test_no_default_glow.py -q"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_hold_legibility as THL  # noqa: E402
from tools.fusion import matte_treatments as MT  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402

TRS = [{"matte": "wipe", "t0": 1.0, "t1": 2.0}, {"matte": "glow", "t0": 3.0, "t1": 4.0}, {"matte": "front", "t0": 5.0, "t1": 6.0}, {"matte": "window", "t0": 7.0, "t1": 8.0}]


def _glows(ops):
    return [o for o in ops if o.get("op") == "add_glow"]


def test_wipe_and_glow_treatments_carry_no_add_glow_by_default():
    ops = MT.ops_for(TRS)
    assert _glows(ops) == []
    assert any(o["op"] == "matte_ring" for o in ops) and any(o["op"] == "contour_lines" for o in ops)         # the rest of the treatment (ring, outline) is unchanged
    on = MT.ops_for([dict(t, edge_glow=True) for t in TRS])
    assert len(_glows(on)) == 2                                                                          # an explicit request still gets it
    assert [o["op"] for o in on if o["op"] != "add_glow"] == [o["op"] for o in ops]


def test_a_woven_piece_chain_has_no_glow_except_hold_life():
    ops = THL._old_ops()
    p = THL._piece(ops, holds=((0.3, 0.9), (1.2, 1.8)))
    p["type"] = {"bpm": 90.0, "words": [{"text": "A", "t0": 0.0, "t1": 1.0, "matte": "wipe"}, {"text": "B", "t0": 1.1, "t1": 2.0, "matte": "glow"}]}
    chk = {"words": [{"hit": 0.3, "release": 0.9, "end": 1.0}, {"hit": 1.2, "release": 1.8, "end": 2.0}], "notes": []}
    PR.weave_matte(p, {"bpm": 90.0}, chk)
    ops = p["compose"]["default"]["ops"]
    assert any(str(o.get("out", "")).startswith(("_wi", "_gl")) for o in ops)                              # the treatments are woven
    assert [o for o in _glows(ops) if not o.get("hl")] == []


def test_planned_pieces_have_no_add_glow_outside_hold_life():
    from test_fusion_plan import REQ, synth_music
    m = synth_music(110, 12)
    for seed in range(8):
        p = PR.plan_piece(dict(REQ, seed=seed, scene="a rider on a beach"), m, dry=False)
        comp = p["compose"]
        chains = ([comp["default"]] if comp.get("default") else []) + list((comp.get("shots") or {}).values())
        bad = [o for ch in chains for o in ch.get("ops", []) if o.get("op") == "add_glow" and not o.get("hl")]
        assert bad == [], (seed, bad[:1])
