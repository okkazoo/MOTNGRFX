"""P06.json -> P06_v003.json: the same piece (stills, shot prompts, seeds, audio, words, grade) with the type driven by the footage's
optical flow. Each compose chain gets a `flow_wake` op right after the matte upscale; everything downstream that read $type_matte
(tint_type, the sandwich, the glow, the s3 halo) reads the displaced alpha $wa instead, so the displaced type is what gets sandwiched.

    python 15_fusion/pieces/_gen_P06_v003.py            # rewrites P06_v003.json from P06.json
    import _gen_P06_v003; _gen_P06_v003.build(wake={...}) -> piece dict (the preview script overrides params this way)
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).parent

# hold windows (hit .. release) of the six words, piece seconds: the wake relaxes there so the word reads
HOLDS = [(0.75, 1.125), (3.0, 4.125), (6.0, 7.125), (9.0, 10.125), (12.0, 13.125), (15.0, 16.875)]

WAKE = {                                                      # tuned on the s1a draft (16_app/p06_flow g02_wake-a)
    "strength": {"curve": "onset", "lo": 3.0, "hi": 6.0},      # beat-reactive: the song's onset envelope scales the push
    "radius": 70, "decay": 0.9, "trail": 10, "trail_gain": 1.0, "bg_drag": 0.12, "max_disp": 0.2, "smooth": 1.0,
}


def settle_timeline(holds=HOLDS, ramp=0.12):
    keys = [[0.0, 0.0]]
    for a, b in holds:
        keys += [[round(a - ramp, 3), 0.0], [round(a + ramp, 3), 1.0], [round(b - ramp, 3), 1.0], [round(b + ramp, 3), 0.0]]
    return {"timeline": keys, "ease": "smooth"}


def wake_op(wake=None):
    args = {"type_a": "$type_matte", "flow": "$flow", "matte": "$m"}
    args.update(copy.deepcopy(dict(WAKE, **(wake or {}))))
    args.setdefault("settle", settle_timeline())
    args.update({"frame_idx": "$frame_idx", "u": "$u", "state_key": "P06"})
    return {"op": "flow_wake", "out": "wa", "args": args}


def _swap(obj):
    if isinstance(obj, str):
        return "$wa" if obj == "$type_matte" else obj
    if isinstance(obj, dict):
        return {k: _swap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap(v) for v in obj]
    return obj


def wake_ops(ops, wake=None):
    out, done = [], False
    for o in ops:
        out.append(_swap(o))
        if o["op"] == "upscale_matte" and not done:       # needs $m (the upscaled runner matte)
            out.append(wake_op(wake))
            done = True
    assert done, "no upscale_matte op to hang flow_wake on"
    return out


def build(wake=None):
    j = json.loads((HERE / "P06.json").read_text(encoding="utf-8"))
    comp = j["compose"]
    comp["default"]["ops"] = wake_ops(comp["default"]["ops"], wake)
    for sid, spec in comp.get("shots", {}).items():
        spec["ops"] = wake_ops(spec["ops"], wake)
    for s in j["shots"]:
        s["flow"] = True                                   # footage optical flow (tools/fusion/footage_flow.py) -> $flow
        if s["id"] in ("s1b", "s2b", "s3b"):               # runner pulls apart into threads (16_masktrack v002: rain d1.0 stable)
            s["masktrack"] = {"style": "rain", "start": "auto", "denoise": 1.0, "corridor": False, "subject": "the runner"}
    j["version"] = "v003"
    j["parent"] = {"piece": "P06", "version": "v002",
                   "reuse": [],                                                  # full re-render on the box (speed test)
                   "why": "all heavy work on the GPU box + type interacts with the footage motion (compose.flow_wake) "
                          "+ runner pulled apart with masked Motion-Track (rain) in s1b/s2b/s3b"}
    return j


if __name__ == "__main__":
    out = HERE / "P06_v003.json"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)
