"""regenerates the compose block of P07.json. python 15_fusion/pieces/_gen_P07.py (type passes come from P07_anamorph.py)"""
import json
from pathlib import Path
p = Path(__file__).with_name("P07.json")
j = json.loads(p.read_text(encoding="utf-8"))
DEP = {"footage_depth": "$depth", "plane": "$type_depth", "softness": 0.14}
ops = [
 {"op": "blur", "out": "mm", "args": {"img": "$type_matte", "sigma": 1.1}},
 {"op": "multiply", "out": "mm", "args": {"a": "$mm", "b": 1.4}},
 {"op": "tint_type", "out": "tcol", "args": {"type_a": "$mm", "footage": "$footage", "mode": "sample", "sample_blur": 40, "sat": 0.8, "lum_target": 1.0, "lum_amount": 1.0}},
 {"op": "blur", "out": "shadow", "args": {"img": "$mm", "sigma": 3.0}},
 {"op": "depth_occlude", "out": "image", "args": dict(DEP, footage="$footage", type_rgb=[0.05, 0.02, 0.07], type_a="$shadow", type_opacity=0.5)},
 {"op": "depth_occlude", "out": "vis", "args": dict(DEP, footage="$black", type_rgb=[1, 1, 1], type_a="$mm", type_opacity=1.0)},
 {"op": "depth_occlude", "out": "image", "args": dict(DEP, footage="$image", type_rgb="$tcol", type_a="$mm", type_opacity=1.0)},
 {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$vis", "radius": 10, "intensity": {"curve": "low", "lo": 0.10, "hi": 0.70}, "color": [1.0, 0.82, 0.55]}},
 {"op": "halation", "out": "image", "args": {"frame": "$image", "threshold": 0.8, "radius": 20, "strength": 0.2, "tint": [1.0, 0.55, 0.3]}},
 {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.0, "gamma": 1.0, "gain": 1.0, "sat": 1.05}},
 {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 7, "amount": 0.02, "size": 1.2}},
 {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.25}}]
ops.insert(0, {"op": "fill", "out": "black", "args": {"like": "$footage", "color": [0, 0, 0]}})
j["compose"]["default"] = {"ops": ops}
j["compose"]["shots"] = {}
p.write_text(json.dumps(j, indent=1), encoding="utf-8")
json.dump(ops, open(p.with_name("_p07_ops.json"), "w"))
print("ok")
