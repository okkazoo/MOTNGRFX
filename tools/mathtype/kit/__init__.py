"""mathtype kit: shared building blocks for the 11_mathtype pieces (GPU-first via xp / torch, CPU fallback).

    import kit
    be = kit.backend()                    # torch cuda if available, else torch cpu (kit.backend("numpy") also works)
    from kit import text, draw, camera, timeline, palette, io, qc

A new piece is a `frame(t, W, H) -> uint8 (H, W, 3)` plus its maths; `kit.io.main(piece)` does the CLI,
generation numbering, TIMINGS rows, stills, score + mux and the automatic QC. See README.md.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATHTYPE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MATHTYPE))
for _p in (MATHTYPE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .core import backend, device_label, torch_or_none  # noqa: E402,F401
from . import camera, draw, io, palette, qc, text, timeline  # noqa: E402,F401

# load torch through xp now: xp.is_t() only recognises tensors once xp.torch() has run, and callers that
# import torch directly (contours.py) would otherwise hand tensors to the numpy branches.
torch_or_none()
