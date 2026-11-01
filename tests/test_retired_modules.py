"""RETIRED WORD MODULES (user 2026-11-02, review of a 16:9 piece where SKIP used `outflow`: "I'm not a fan of these echos coming in from one side and then going down. I'd remove this type of echo from the
library of choices"). `outflow` is retired the way `particles` was (2026-11-01): the module file and its registration stay, but no choosable set offers it, PS.choose never draws it, and a piece / reply that
names it is swapped at plan time (owner.fix_speckle_forms via text_guard.fix_plan; the own-video route runs the same swap). NOT the liked time echoes (x08 / echo effects / flow_wake echo cap).
Run: python -m pytest tests/test_retired_modules.py -q"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tools.app import overlay_mode as OM  # noqa: E402
from tools.fusion import batch_plan as bp  # noqa: E402
from tools.fusion import design_defaults as DD  # noqa: E402
from tools.fusion import design_salvage as SV  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import qwen_piece as QP  # noqa: E402
from tools.fusion import text_guard as TG  # noqa: E402

OUTFLOW_PARAMS = dict(tilt=0.9, rings=14, n_arcs=3, width=0.2, speed=1.5)
MOODS = ("tender", "eerie", "defiant", "joyful")      # soft / tense / driven / bright


def test_library_is_intact():
    bp.word_modules()                                      # imports mograph.modules (registers the words)
    from mograph.modules.core import list_modules
    assert "outflow" in list_modules("word")              # registered and importable: only the choice is removed


def test_no_choosable_set_offers_outflow():
    assert "outflow" in DD.RETIRED_MODULES and "outflow" in DD.BANNED_MODULES and "outflow" in DD.AVOID_MODULES
    assert "outflow" not in PS.KEPT_MODULES and "outflow" not in DD.EFFECT_MODULES and "outflow" not in DD.SPECKLE_MODULE_FALLBACK
    assert "outflow" not in sum(PS.FAMILY.values(), ())
    assert not any("outflow" in w for w in PS.MODULE_WEIGHTS.values())
    assert "outflow" not in QP.WORD_MODULES and "outflow" not in bp.word_modules()
    assert "outflow" not in sum(PR.LOOKS.values(), [])
    assert '"outflow"' not in open(os.path.join(ROOT, "nodes_fusion", "routes.py"), encoding="utf-8").read().split("LOOK_FAMILIES")[1].split("}")[0]      # the page's look -> modules table
    assert "outflow" not in OM.module_pool(("slats", "outflow", "gradient_fade"))
    assert "outflow" not in SV.SYN_MODULE.values()


def test_the_liked_time_echoes_are_untouched():
    for m in ("depth_echo", "bold_thin_echo", "fly_into"):          # x08 / echo effects / the other ring module stay in the library as before
        assert m in QP.WORD_MODULES
    assert "fly_into" in PS.KEPT_MODULES


def test_choose_never_draws_outflow():
    for mood in MOODS:
        for seed in range(40):
            for words in ((), ("ONE",), ("ONE", "TWO", "THREE"), ("ONE", "TWO", "THREE", "FOUR", "FIVE")):
                s = PS.choose(seed, mood, 70 + seed * 2, words=words)
                assert "outflow" not in s["modules"] and s["module"] != "outflow" and "outflow" not in s["word_modules"], (mood, seed, words)
    assert "outflow" not in PS.choose(3, "tender", 80, pin_modules=["outflow", "lanes"])["modules"]          # even a pinned pick is not honoured


def _piece(mods=("outflow", "outflow", "outflow")):
    words = [dict(text=t, module=m, form_in="lines", form_out="wipe", params=dict(OUTFLOW_PARAMS, bpm=120)) for t, m in zip(("SKIP", "THE", "BEAT"), mods)]
    return {"type": {"words": words}, "plan": {"piece_system": {"module": "outflow", "modules": ["outflow", "lanes"], "word_modules": list(mods),
                                                                 "io": {"form_in": ["lines", "wipe"], "form_out": ["wipe", "flicker"]}}}}


def test_a_piece_that_asks_for_outflow_is_swapped_in_the_image_route():
    p = _piece()
    fixes = TG.fix_plan(p)
    ws = p["type"]["words"]
    assert not any(w["module"] in DD.BANNED_MODULES for w in ws)
    assert len({w["module"] for w in ws}) == len(ws) <= DD.MAX_MODULES      # module variety kept: a distinct module per word, within the cap
    assert all(not (set(OUTFLOW_PARAMS) & set(w["params"])) and w["params"].get("bpm") == 120 for w in ws)      # outflow's own params dropped, the rest stays
    ps = p["plan"]["piece_system"]
    assert "outflow" not in ps["modules"] and ps["module"] != "outflow" and "outflow" not in ps["word_modules"]
    assert ps["word_modules"] == [w["module"] for w in ws]
    assert any("outflow" in f for f in fixes)
    assert OW.fix_speckle_forms(p) == []                                     # idempotent


def test_a_mixed_piece_keeps_its_other_modules():
    p = _piece(mods=("lanes", "outflow", "slats"))
    OW.fix_speckle_forms(p)
    ms = [w["module"] for w in p["type"]["words"]]
    assert ms[0] == "lanes" and ms[2] == "slats" and ms[1] not in DD.BANNED_MODULES
    assert len(set(ms)) <= DD.MAX_MODULES


def test_swap_respects_the_module_cap():
    p = _piece(mods=("lanes", "slats", "circuit", "outflow"))
    OW.fix_speckle_forms(p)
    ms = [w["module"] for w in p["type"]["words"]]
    assert "outflow" not in ms and len(set(ms)) <= DD.MAX_MODULES


def test_a_clean_piece_is_untouched():
    p = _piece(mods=("lanes", "slats", "circuit"))
    p["plan"]["piece_system"] = {"module": "lanes", "modules": ["lanes", "slats", "circuit"], "word_modules": ["lanes", "slats", "circuit"],
                                 "io": {"form_in": ["lines", "wipe"], "form_out": ["wipe", "flicker"]}}
    for w in p["type"]["words"]:
        w["params"] = dict(bpm=120)
    before = copy.deepcopy(p)
    assert OW.fix_speckle_forms(p) == [] and p == before


def test_a_particles_piece_still_swaps_as_before():
    p = _piece(mods=("particles", "outflow", "lanes"))
    for w in p["type"]["words"]:
        w["params"] = dict(n=3000, size=2, tilt=0.9, bpm=120)
    OW.fix_speckle_forms(p)
    assert not any(w["module"] in DD.BANNED_MODULES for w in p["type"]["words"])
    w0, w1, w2 = p["type"]["words"]
    assert "n" not in w0["params"] and "size" not in w0["params"] and w0["params"]["bpm"] == 120      # each swapped word loses its old module's own params
    assert "tilt" not in w1["params"] and w1["params"]["bpm"] == 120
    assert w2["module"] == "lanes" and w2["params"]["tilt"] == 0.9                                      # an untouched word keeps its params


def test_own_video_route_swaps_outflow():
    assert "outflow" not in OM.module_pool(("slats", "gradient_fade", "staircase", "beat_bars", "particles", "outflow"))
    p = _piece(mods=("outflow", "slats", "outflow"))
    fixes = OM.mend_speckle(p)                                              # the own-video parse path (own_video_v2_retime_box -> OVM.mend_speckle)
    assert fixes and not any(w["module"] in DD.BANNED_MODULES for w in p["type"]["words"])
    assert p["type"]["words"][0]["module"] != p["type"]["words"][2]["module"]
    assert OM.mend_speckle(p) == []                                         # idempotent


def test_a_model_reply_naming_outflow_lands_on_a_kept_module():
    check = SV.v_enum(QP.WORD_MODULES, SV.SYN_MODULE)
    for v in ("outflow", "flow", "Outflow"):
        got, _kind, _msg = check(v)
        assert got in QP.WORD_MODULES and got not in DD.BANNED_MODULES, (v, got)
