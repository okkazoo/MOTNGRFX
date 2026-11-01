"""P13 Highland (v004): orchestral, landscape 16:9, 75 bpm score grid (brief 80, snapped to whole bars), 16 s, ONE continuous i2v
shot (385 f, single slow lateral sweep). Run with 15_fusion/pieces/_run_landscape.py (the runner's draft path is portrait-only).

Words ABOVE / THE RIDGE / THE MORNING / WAITS. Hits 1.6 (beat 3) / 6.4 (downbeat) / 9.6 (downbeat) / 14.4 (beat 3), holds 2 beats.
Transitions (no repeats): ABOVE -> THE RIDGE = `twist` (slow, 2-beat gap); THE RIDGE -> THE MORNING = ridge-contour carry (engine cut:
the letters dissolve into the ridge iso-contours of the depth and THE MORNING re-forms out of them, P08's idea, stronger);
THE MORNING -> WAITS = `braid`. Letters are built from the ridge contours (type_from_edges): they condense out of them at the start,
hold solid, and dissolve back at the end. A light pulse travels through the type on every beat (depth iso band whose phase runs
0 -> 1 across each beat, masked by the letters, glowed). Weave: the nearest ridges (depth above `near_thr`, solid threshold) cover
the type; valley fog (bright luma key away from the sky, sheer) drifts over it. flow_wake pushes the letters outside the holds.

    python 15_fusion/pieces/_gen_P13.py [key=val ...]   -> P13.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _v004_lib import reformat, window_ops, hold_gate, song_paths, cli_song, load_grid, retime, rel, replan_state, analysis_for, tl, pulse, ramp_keys  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402

PID = "P13"
DESIGN_BPM = 75.0          # every time below is written on this grid (beat 0 at 0 s); retime() maps it onto the loaded grid
T = dict(near_thr=0.66, fog_lo=0.62, fog_hi=0.9, levels=8)


def build(tune=None, song=None, grid_path=None, fmt="16:9"):
    land = fmt == "16:9"
    pid = PID + ("" if land else "_v")
    T.update(tune or {})
    wav, gp = song_paths(PID, song, grid_path)
    g = load_grid(gp)
    at, dur = retime(g, DESIGN_BPM, anchor=1.6)      # ABOVE's hit on the first downbeat (t2m song starts on beat 4)
    cut, bpm = float(g["cut"]), float(g["bpm"])
    iv = 60.0 / bpm
    beats, dbs = [x for x in g["beats"] if x <= cut + 1e-6], g["downbeats"]
    font = "BigShouldersDisplay[wght].ttf"   # tall; THE RIDGE = Moirai One hairline outline, WAITS = Ultra fat solid (contrast rule)
    W = lambda **k: dict(dict(font=font, module="lanes", axes={"wght": 800}), **k)  # noqa: E731
    words = [
        W(text="ABOVE", lines=["ABOVE"], t0=0.0, t1=at(4.0), form_s=at(1.6), exit_s=dur(0.8), hit_on="bar",
          form_in="lines", form_out="lines", placement={"yaw": -8, "dy": -0.08, "w": 900 if land else 520}),
        W(text="THE RIDGE", lines=["THE RIDGE"] if land else ["THE", "RIDGE"], t0=at(5.6), t1=at(8.8), form_s=dur(0.8), exit_s=dur(0.8), hit_on="beat",
          form_in="lines", form_out="wipe", placement={"yaw": 8, "dy": -0.10, "w": 560 if land else 360}, font="MoiraiOne-Regular.ttf", axes=None),
        W(text="THE MORNING", lines=["THE MORNING"] if land else ["THE", "MORNING"], t0=at(9.2), t1=at(12.0), form_s=dur(0.4), exit_s=dur(0.8), hit_on="beat",
          form_in="wipe", form_out="lines", placement={"yaw": -6, "dy": -0.06, "w": 900 if land else 520}),
        W(text="WAITS", lines=["WAITS"], t0=at(12.8), t1=at(16.0), form_s=dur(0.8), exit_s=dur(0.8), hit_on="beat",
          form_in="lines", form_out="lines", placement={"yaw": 6, "dy": -0.08, "w": 820 if land else 500}, font="Ultra-Regular.ttf", axes=None),
    ]
    type_spec = {"seed": 13, "energy": 0.35, "mood": "calm", "tier": "calm",
                 "palette": {"bg": [20, 16, 12], "fg": [255, 238, 210], "accents": [[255, 200, 120], [230, 150, 80], [190, 200, 220]]},
                 "camera": "beat_orbit", "camera_params": {"read_dps": 3.0, "intro_deg": 16.0, "fast_dps": 12.0, "elev": 3.0, "bob": 0.5},
                 "transitions": [{"module": "twist"}, None, {"module": "braid"}], "words": words, "line": {"px": 3.0}}
    holds = [(at(a), at(b)) for a, b in [(1.6, 3.2), (6.4, 8.0), (9.6, 11.2), (13.6, 15.2)]]
    hg = hold_gate(holds)

    pz = pulse(beats, dbs, amp=1.0, amp_db=1.6, decay=0.3)
    fill = ramp_keys([(at(a), v) for a, v in [(0, 0), (1.2, 0), (1.6, 1), (5.6, 1), (6.2, 0), (8.0, 0), (8.5, 0), (9.2, 0), (9.6, 1), (16, 1)]])
    grow = ramp_keys([(at(a), v) for a, v in [(0, 24), (1.6, 3), (8.0, 3), (8.8, 22), (9.2, 22), (9.6, 3), (16, 3)]])
    carry = ramp_keys([(at(a), v) for a, v in [(0, 0.6), (1.6, 0.0), (8.0, 0.0), (8.4, 1.0), (9.4, 1.0), (9.9, 0.0), (16, 0.0)]])
    clgate = lambda t: min(1.0, 0.18 + 0.55 * carry(t) + 0.12 * pz(t))  # noqa: E731
    beat_phase = lambda t: (((t - float(g["beats"][0])) % iv) / iv)  # noqa: E731     light band runs front -> back once per beat

    ops = [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 1.0, "gain": 0.97, "sat": 1.05}},
        {"op": "contour_lines", "out": "cl", "args": {"src": "$depth", "mode": "iso", "levels": T["levels"],
                                                      "phase": {"timeline": [[0.0, 0.0], [cut, 1.0]]}, "width": 1, "smooth": 2.5,
                                                      "lo": 0.05, "hi": 0.95, "min_len": 40, "frame_idx": "$frame_idx"}},
        {"op": "multiply", "out": "clv", "args": {"a": "$cl", "b": tl(clgate, cut)}},
        {"op": "over", "out": "image", "args": {"base": "$foot", "fg": [1.0, 0.9, 0.72], "alpha": "$clv"}},
        {"op": "type_from_edges", "out": "tfe", "args": {"edges": "$cl", "type_a": "$type_matte", "grow": tl(grow, cut, nd=1),
                                                         "line_width": 1, "density": 2.5, "fill": tl(fill, cut), "softness": 3}},
        {"op": "multiply", "out": "front", "args": {"a": "$tfe", "b": tl(hg, cut)}},      # in front of every matte while read
        {"op": "sandwich", "out": "image", "args": {"footage": "$image", "type_rgb": [1.0, 0.95, 0.86], "type_a": "$tfe",
                                                    "matte": "$depth", "matte_threshold": T["near_thr"], "matte_soft": 0.08,
                                                    "front": "$front"}},
        # valley fog (sheer): bright luma, but not the sky (far depth)
        {"op": "luminosity_key", "out": "fog", "args": {"img": "$foot", "low": T["fog_lo"], "high": T["fog_hi"]}},
        {"op": "multiply", "out": "fog", "args": {"a": "$fog", "b": "$depth"}},     # sky (far depth ~0) keeps no fog

        {"op": "multiply", "out": "fog", "args": {"a": "$fog", "b": tl(lambda t: 0.7 * (1.0 - 0.9 * hg(t)), cut)}},
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$foot", "alpha": "$fog"}},
        # light pulse through the type on every beat
        {"op": "contour_lines", "out": "band", "args": {"src": "$depth", "mode": "iso", "levels": 2, "phase": tl(beat_phase, cut),
                                                        "width": 10, "smooth": 4.0, "lo": 0.0, "hi": 1.0, "min_len": 20,
                                                        "frame_idx": "$frame_idx"}},
        {"op": "blur", "out": "band", "args": {"img": "$band", "sigma": 8}},
        {"op": "multiply", "out": "lit", "args": {"a": "$band", "b": "$tfe"}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$lit", "radius": 10, "intensity": tl(lambda t: 0.25 + 0.6 * pz(t), cut),
                                                    "color": [1.0, 0.82, 0.5]}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$tfe", "radius": 18, "intensity": tl(lambda t: 0.06 + 0.16 * pz(t), cut),
                                                    "color": [1.0, 0.8, 0.5]}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.02, "sat": 1.05}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 13, "amount": 0.02, "size": 1.2}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.25}},
    ]
    # g02 redesign (P10 bar): ABOVE / THE MORNING are WINDOWS (night-blue plate, then the ground flips to dawn gold), the near
    # ridges step out in front of the plate; THE RIDGE = Moirai One hairline outline built from the ridge contours; WAITS = Ultra.
    gi = next(i for i, o in enumerate(ops) if o["op"] == "add_glow")
    outline = ramp_keys([(at(a), v) for a, v in [(0, 0), (5.6, 0), (6.4, 1), (8.0, 1), (8.6, 0), (16, 0)]])
    extra = [{"op": "multiply", "out": "oline", "args": {"a": "$type_line", "b": tl(outline, cut)}},
             {"op": "over", "out": "image", "args": {"base": "$image", "fg": [1.0, 0.94, 0.82], "alpha": "$oline"}}]
    extra += window_ops([(at(1.6), at(4.0), [0.05, 0.07, 0.12], [1.0, 0.86, 0.6]),
                         (at(9.6), at(12.0), [0.98, 0.86, 0.62], [0.12, 0.08, 0.05])], cut, T["near_thr"],
                        type_a="$type_matte", line="$type_line")
    ops[gi:gi] = extra
    ops = PR.wake_ops(ops, holds, pid, {"curve": "onset", "lo": 2.5, "hi": 5.0})

    still = ("dawn over layered mountain ridges, valley fog between the ranges, golden first light raking across the ridges, "
             "receding layers of blue and gold mountains, clear sky, wide landscape photograph, 35mm, cinematic, no people")
    piece = {
        "id": PID, "slug": "highland", "fps": 24, "version": "v004", "format": "16:9",
        "audio": {"src": rel(wav), "bpm": int(round(bpm)), "analysis": analysis_for(wav, gp), "grid": rel(gp),
                  "cut": cut, "fade": 0.4},
        "type": type_spec, "type_passes": ["matte", "line"],
        "stills": [{"id": "A", "model": "moss", "seed": 1301, "size": [1920, 1088], "prompt": still,
                    "negative": "text, watermark, logo, caption, people, cartoon, illustration, 3d render, oversaturated, blurry"}],
        "shots": [{"id": "s1", "kind": "i2v", "first": "A", "frames": 385, "span": min(int(g["nframes"]), 385), "seed": 131,
                   "size": [1920, 1088], "depth": True, "flow": True,
                   "prompt": "one continuous slow lateral camera sweep to the right across the layered mountain ridges at dawn, the near "
                             "ridges slide past faster than the far ones, valley fog drifts slowly between the ranges, golden first "
                             "light spreads over the peaks, epic, cinematic, smooth steady camera"}],
        "compose": {"crossfade": 0, "crf": 18, "default": {"ops": ops}},
        "signoff": {"text": "", "off": True},
        "parent": {"piece": None, "why": "v004 brief P13 (15_fusion/v004/BRIEFS.md) + reel-017 visual decisions"},
    }
    PR.fill_type_timing(piece, {"cut": cut, "bpm": bpm, "beats": g["beats"], "downbeats": g["downbeats"]})
    return piece


if __name__ == "__main__":
    # python _gen_P13.py [--song X.wav] [--grid X_grid.json] [--replan] [key=val ...]
    #   default song: 15_fusion/v004/audio/t2m/P13.wav + _grid.json when present, else the s1 score song.
    #   --replan: drop audio / type / compose / final from the runner state so the next run re-plans type + compose on the new grid
    #             and reuses the cached LTX footage, depth and flow.
    song, gridp, rest = cli_song(sys.argv[1:])
    alt = "--alt" in rest           # the other-aspect version (_v): own still + shot, same song / grid / plan
    tune = {k: float(v) for k, v in (a.split("=", 1) for a in rest if "=" in a)}
    p = build(tune, song, gridp)
    if alt:
        p = reformat(p, "9:16", "_v",
            ("tall vertical photograph from a high peak at dawn, a near rocky ridge in the foreground at the bottom of the frame, layered "
             "mountain ridges receding upward through the middle with valley fog between the ranges, golden first light raking across "
             "the ridges, a big clear sky at the top, 35mm, cinematic, no people, tall vertical frame"),
            ("one continuous slow lateral camera sweep to the right across the layered mountain ridges at dawn, the near ridge slides past "
             "faster than the far ones, valley fog drifts slowly between the ranges, golden first light spreads over the peaks, epic, "
             "cinematic, smooth steady camera"))
    out = HERE / f"{p['id']}.json"
    out.write_text(json.dumps(p, indent=1), encoding="utf-8")
    print("song", p["audio"]["src"], "grid", p["audio"]["grid"], "bpm", p["type"]["bpm"])
    chk = PR.dry_run(p, size=(1024, 576) if p.get("format") == "16:9" else (576, 1024))
    for w in chk["words"]:
        print(f"  {w['text']!r:14} start {w['start']:.2f} hit {w['hit']:.2f} rel {w['release']:.2f} end {w['end']:.2f} {w['lines']}")
    print("notes:", chk["notes"], "last_end", round(chk["last_end"], 2))
    if "--replan" in rest:
        print("state: dropped", replan_state(p["id"]))
    print("wrote", out)
