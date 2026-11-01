"""Storyboard random font default must never be a hard-to-read ("risky") face: a default goes into the brief as the user's own font choice (font_pins) and the
legibility rules then keep it (GPU render R2 2026-11-01: BungeeHairline kept at ~6% cap height). CPU only.
Run: python -m pytest --rootdir=tests tests/test_font_risky.py -q"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.app import fonts_route as fr  # noqa: E402

NODE = shutil.which("node")


def test_liked_list_flags_risky_faces():
    by = {e["key"]: e for e in fr.liked_list()}
    assert all(isinstance(e["risky"], bool) for e in by.values())
    for k in ("BungeeHairline", "Doto", "FacadeVF", "Monofett"):
        assert by[k]["risky"] is True, k
    assert by["Anybody"]["risky"] is False
    assert any(not e["risky"] for e in by.values())


def test_risky_falls_back_when_design_defaults_does_not_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "tools.fusion.design_defaults", None)        # import raises ImportError
    assert "BungeeHairline" in fr._risky_names() and "Doto" in fr._risky_names()
    by = {e["key"]: e for e in fr.liked_list()}
    assert by["BungeeHairline"]["risky"] and by["Doto"]["risky"] and not by["Anybody"]["risky"]


_JS = r"""
const B = require(process.argv[1]);
const fonts = JSON.parse(process.argv[2]);
let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const picked = new Set();
for (let i = 0; i < 3000; i++) picked.add(B.fontPick(fonts, [], [], rnd));
for (let i = 0; i < 300; i++) for (const f of B.fontPickAll(fonts, 6, rnd)) picked.add(f);
const reach = new Set(); let cur = fonts[0].file;
for (let i = 0; i < fonts.length; i++) { cur = B.fontStep(fonts, cur, 1); reach.add(cur); }
const allRisky = fonts.map((f) => ({ ...f, risky: true }));
console.log(JSON.stringify({ picked: [...picked], reach: [...reach], fallback: B.fontPick(allRisky, [], [], rnd) }));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_fontpick_never_risky_but_fontstep_reaches_it():
    fonts = [{"file": e["file"], "role": e["role"], "risky": e["risky"]} for e in fr.liked_list()]
    r = subprocess.run([NODE, "-e", _JS, os.path.join(ROOT, "web_app", "bind.js"), json.dumps(fonts)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    risky = {f["file"] for f in fonts if f["risky"]}
    assert risky and not (set(out["picked"]) & risky)
    assert not [f for f in out["picked"] if "barcode" in f.lower()]
    assert len(out["picked"]) > 10
    assert risky <= set(out["reach"])                      # the arrows still reach every face
    assert out["fallback"]                                 # all-risky pool falls back to the full list
