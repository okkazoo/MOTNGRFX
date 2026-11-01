"""piece-local helper: render the type passes of a piece JSON into a scratch dir (never touches state.json / the piece's own type dir).
   python 15_fusion/pieces/_tune_type.py P09 [--frames a:b] [--passes matte,beauty] [--tag t1] [--size 576x1024] [--procs 3]
Writes 15_fusion/v002/local/pieces/<id>/type_<tag>/ plus a contact sheet review/<id>_tune_<tag>.png"""
import argparse, copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.fusion import type_passes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pid"); ap.add_argument("--frames"); ap.add_argument("--passes", default="matte,depth,line,beauty")
ap.add_argument("--tag", default="t1"); ap.add_argument("--size", default="1088x1920"); ap.add_argument("--procs", type=int, default=3)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
j = json.loads((ROOT / "15_fusion" / "pieces" / f"{a.pid}.json").read_text(encoding="utf-8"))
d = ROOT / "15_fusion" / "v002" / "local" / "pieces" / a.pid
ana = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
spec = copy.deepcopy(j["type"])
spec.update(fps=24, duration=float(ana["cut"]), bpm=float(ana["bpm"]), beats=ana["beats"], downbeats=ana["downbeats"])
spec.setdefault("bar", 4)
w, h = map(int, a.size.split("x"))
spec["size"] = [w, h]
out = d / f"type_{a.tag}"
fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
man = type_passes.render_passes(spec, str(out), size=(w, h), passes=a.passes.split(","), frames=fr, procs=a.procs, overwrite=True, mp4=False, verbose=False)
sheet = ROOT / "15_fusion" / "v002" / "review" / f"{a.pid}_tune_{a.tag}.png"
type_passes.make_sheet(str(out), str(sheet), n=a.n, passes=a.passes.split(",")[:4])
for wd in man["info"]["words"]:
    print(f"word {wd['text']!r:14} start {wd['start']:.2f} hit {wd['hit']:.2f} release {wd['release']:.2f} end {wd['end']:.2f} {wd['lines']}")
for n in man["info"]["notes"]:
    print("NOTE", n)
print("sheet", sheet, "wall", man["timing"]["wall_s"])
