"""render R17: voxel_roll rendered FINAL / ECHO (Bungee Inline, two lines, wf 0.9) as one merged black block. The module measures its own cell grid
(mograph/modules/words/voxel_roll.cells_legible) and tools/fusion/owner.fix_illegible_modules moves a word that fails to a readable module at plan time.
Pure python (layout + cv2). Run: python -m pytest tests/test_voxel_roll_legible.py -q"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("FUSION_BACKEND", "numpy")

from mograph.modules.words.voxel_roll import cells_legible  # noqa: E402
from tools.fusion import owner as OW  # noqa: E402
from tools.fusion import text_guard as TG  # noqa: E402

BUNGEE = "BungeeInline-Regular.ttf"


def _piece(modules=("voxel_roll", "voxel_roll")):
    return {"format": "9:16", "sizes": {"type": [576, 1024]}, "shots": [], "type": {"words": [
        {"text": "CRASHING FAITH", "font": "MoiraiOne-Regular.ttf", "lines": ["CRAS", "HING", "FAI", "TH"], "module": modules[0], "placement": {"wf": 0.9, "hf": 0.627}},
        {"text": "FINAL ECHO", "font": BUNGEE, "lines": ["FINAL", "ECHO"], "module": modules[1], "placement": {"wf": 0.9, "hf": 0.283}, "params": {"rows": 6, "lean": 10}}]}}


def test_the_r17_word_is_measured_as_merged():
    r = cells_legible(BUNGEE, ["FINAL", "ECHO"], wf=0.9, hf=0.283)
    assert not r["ok"] and r["letters"] == 9 and r["pieces"] <= 3 and r["cells"] == 242, r       # 2 pieces for 9 letters: lines and neighbours share cubes
    assert "merge" in r["reason"]


def test_short_one_line_words_and_a_pixel_face_stay_on_the_module():
    for font, lines in ((BUNGEE, ["ROLL"]), ("silkscreen", ["ROLL"]), ("silkscreen", ["HELLO"]), (BUNGEE, ["GO"])):
        r = cells_legible(font, lines, wf=0.9, hf=0.25)
        assert r["ok"] and r["pieces"] >= 0.75 * r["letters"], (font, lines, r)


def test_a_hairline_face_that_voxelises_to_nothing_is_rejected():
    r = cells_legible("MoiraiOne-Regular.ttf", ["CRAS", "HING", "FAI", "TH"], wf=0.9, hf=0.627)
    assert not r["ok"] and r["cells"] < 8 and "too few cubes" in r["reason"], r


def test_plan_time_swap_to_a_readable_module():
    p = _piece()
    fixes = OW.fix_illegible_modules(p)
    ws = p["type"]["words"]
    assert len(fixes) == 2 and ws[0]["module"] == "lanes" and ws[1]["module"] == "lanes", fixes
    assert "rows" not in ws[1]["params"] and "lean" not in ws[1]["params"]                  # the cube module's own params go with it
    assert p["plan"]["module_legibility"] == fixes and OW.fix_illegible_modules(p) == []      # idempotent
    q = _piece(("voxel_roll", "voxel_roll"))
    q["type"]["words"].append({"text": "ROLL", "font": BUNGEE, "lines": ["ROLL"], "module": "slats", "placement": {"wf": 0.9, "hf": 0.25}})
    OW.fix_illegible_modules(q)
    assert [w["module"] for w in q["type"]["words"]] == ["slats", "slats", "slats"]          # the piece's own readable module is preferred over the default
    ok = {"format": "9:16", "type": {"words": [{"text": "ROLL", "font": BUNGEE, "lines": ["ROLL"], "module": "voxel_roll", "placement": {"wf": 0.9, "hf": 0.25}}]}}
    assert OW.fix_illegible_modules(ok) == [] and ok["type"]["words"][0]["module"] == "voxel_roll"


def test_every_design_source_goes_through_text_guard_fix_plan():
    p = _piece()
    fixes = TG.fix_plan(p)
    assert any("module voxel_roll -> lanes" in f for f in fixes) and p["type"]["words"][1]["module"] == "lanes"
