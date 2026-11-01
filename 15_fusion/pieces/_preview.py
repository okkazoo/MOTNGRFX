"""piece-local helper: compose a few piece frames with the ops in <id>.json (or an override ops json) -> review/<id>_prev_<tag>.png
   python 15_fusion/pieces/_preview.py P09 --t 1.0,2.5,5.8 [--ops ops.json] [--tag a] [--w 360] [--cols 4]
   --ops: json list of ops used for every shot (replaces compose.default and per-shot ops). Needs the piece's shots/depth/type done."""
import argparse, copy, json, os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["FUSION_BACKEND"] = "numpy"
import cv2, numpy as np  # noqa: E402
from tools.fusion import piece_runner as PR, compose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pid"); ap.add_argument("--t", required=True); ap.add_argument("--ops"); ap.add_argument("--tag", default="a")
ap.add_argument("--w", type=int, default=360); ap.add_argument("--cols", type=int, default=4); ap.add_argument("--full", action="store_true"); ap.add_argument("--type-dir")
a = ap.parse_args()
P = PR.Piece(ROOT / "15_fusion" / "pieces" / f"{a.pid}.json", 1)
if a.ops:
    ops = json.loads(Path(a.ops).read_text(encoding="utf-8"))
    P.j["compose"]["default"] = {"ops": ops}
    P.j["compose"]["shots"] = {}
if a.type_dir:
    P.typ = lambda: dict(dir=a.type_dir, nf=10 ** 6, passes=["matte", "depth", "line", "beauty"])
lay, _ = P.layout()
tiles = []
for tt in a.t.split(","):
    t = float(tt)
    f = int(round(t * 24))
    sid = [s for s in lay if lay[s]["start"] <= f][-1]
    L = lay[sid]
    loc = min(max(f - L["start"], 0), L["frames"] - 1)
    rec = P.recipe(sid)
    d = Path(tempfile.mkdtemp())
    PR._chunk((rec, loc, loc + 1, str(d)))
    im = cv2.imread(str(d / f"{loc:05d}.png"))
    if a.full:
        cv2.imwrite(str(ROOT / "15_fusion" / "v002" / "review" / f"{a.pid}_full_{a.tag}_{t:.2f}.png"), im)
    im = cv2.resize(im, (a.w, int(im.shape[0] * a.w / im.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"{t:.2f}s {sid}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 60), 1, cv2.LINE_AA)
    tiles.append(im)
while len(tiles) % a.cols:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)])
out = ROOT / "15_fusion" / "v002" / "review" / f"{a.pid}_prev_{a.tag}.png"
cv2.imwrite(str(out), sheet)
print("wrote", out)
