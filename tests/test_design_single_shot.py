"""REGRESSION (batch 3060 v005 r01 / v007 r01, both SINGLE-shot pieces): the model's design reply was valid JSON, yet `parse: type = rules, compose = rules, design valid = False,
fields: 0 from the model (0 %)` - the whole design was thrown away. Two causes, both fixed (2026-10-28):
  1. `qwen_piece.compile_timing` made "the 5 words do not fit in 16 s even with every hold at 1.2 s" a hard ERROR, so the type section was rejected before `fit_scene`
     (hits moved earlier, form / exit tightened) could fit it. It is a logged fix now; fit_scene decides.
  2. the plan's own technique of a single shot (`built_in`, allowed by the design context) was not in `qwen_piece.TECHNIQUES`, so `compose.technique 'built_in'` was an error
     (and design_salvage's enum rejected it): the whole compose section fell to the rules.
Fixtures: the real replies + the r01 piece of each run (the plan piece the design step is called with). No model, no GPU.
Run: python -m pytest tests/test_design_single_shot.py -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import design_step as DS  # noqa: E402

FIX = HERE / "fixtures"


@pytest.mark.parametrize("run", ["v005", "v007"])
def test_single_shot_design_reply_is_kept_not_thrown_away(run):
    piece = json.loads((FIX / f"single_shot_{run}_piece.json").read_text(encoding="utf-8"))
    reply = (FIX / f"single_shot_{run}_design_reply.txt").read_text(encoding="utf-8")
    assert len(piece["shots"]) == 1 and len(piece["type"]["words"]) == 5
    r = DS.parse(reply, None, piece, None, "")
    src = r["piece"]["design"]["src"]
    assert src["type"] == "model" and src["compose"] == "model", r["report"][:600]
    f = r["fields"]              # 40 %, not 60: these OLD plans own all five words (handoff), and an LTX-owned word keeps the plan's face / module / lines / box (design_step.lock_graphic, 2026-10-29)
    assert f["sections"] == {"type": "model", "compose": "model"} and f["pct_model"] >= 40.0 and not f["fields_from_rules"] and not f["pending"], f["pct_model"]
    assert "design valid = True" in r["report"] and r["ok"]
    assert len(r["piece"]["type"]["words"]) == 5 and r["piece"]["type"]["words"][-1]["t1"] <= 16.0 + 1e-6      # the timing fit moved the hits inside the 16 s


def test_plan_technique_outside_the_global_list_is_allowed():
    piece = json.loads((FIX / "single_shot_v007_piece.json").read_text(encoding="utf-8"))
    reply = (FIX / "single_shot_v007_design_reply.txt").read_text(encoding="utf-8")
    r = DS.parse(reply, None, piece, None, "")
    assert r["piece"]["technique"] == "built_in" and "compose.technique 'built_in'" not in r["report"] and "not one of" not in r["report"]


def test_timing_that_overruns_the_music_is_fitted_not_rejected():
    from tools.fusion import qwen_piece as QP
    import inspect
    src = inspect.getsource(QP.compile_timing)
    assert "errors.append(f\"the {len(done)} words do not fit" not in src          # no hard error: fit_scene decides
