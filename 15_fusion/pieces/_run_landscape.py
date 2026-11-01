"""piece_runner for a 16:9 piece ("format": "16:9" in the P*.json) WITHOUT editing tools/fusion/piece_runner.py: the runner's draft
path is portrait-only (type 576x1024, compose 540x960, depth / alpha reference clips cropped to 540x960 / 576x1024). This wrapper
swaps those sizes for landscape pieces and then runs the normal CLI (same arguments).

    python 15_fusion/pieces/_run_landscape.py 15_fusion/pieces/P13.json --exec box ... --draft --out-version v004

Remove once piece_runner reads the piece format itself (reported as a blocker in the v004 P11/P13/P15 run)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.fusion import piece_runner as PRN  # noqa: E402

_init, _box, _ref = PRN.Piece.__init__, PRN.Piece.piece_box, PRN.Piece._ref_clip


def _land(self):
    return self.j.get("format") == "16:9"


def __init__(self, *a, **k):
    _init(self, *a, **k)
    if _land(self):
        self.type_size = (1024, 576) if self.draft else (1920, 1088)


def piece_box(self):
    j = _box(self)
    if _land(self):
        j.setdefault("sizes", {})["compose"] = [960, 540] if self.draft else [1920, 1080]
    return j


def _ref_clip(self, sid, tag, w, h):
    if _land(self) and h > w:
        w, h = h, w
    return _ref(self, sid, tag, w, h)


PRN.Piece.__init__, PRN.Piece.piece_box, PRN.Piece._ref_clip = __init__, piece_box, _ref_clip

if __name__ == "__main__":
    j = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    assert j.get("format") == "16:9", "use tools/fusion/piece_runner.py for portrait pieces"
    sys.exit(PRN.main(sys.argv[1:]))
