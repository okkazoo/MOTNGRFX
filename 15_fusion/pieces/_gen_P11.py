"""P11 Tidewater (v004): B/W, slow, 75 bpm score grid, 16 s, 9:16, ONE continuous i2v shot (385 f, slow push in).

Words THE TIDE / KEEPS / WHAT WE / LET GO. Hits: 1.6 (beat 3) / 4.8 (beat 3) / 9.6 (downbeat) / 12.8 (downbeat), holds 2-3 beats.
Weave: wave contours (contour_lines iso on the shot depth, phase flowing toward camera) are on the sea the whole piece; letters are
BUILT from those contours (type_from_edges): they condense out of the waves (fill 0 -> 1), hold solid, then dissolve back into the
waves (fill -> 0, halo grows) and the next line condenses out of them (THE TIDE -> KEEPS, WHAT WE -> LET GO; the engine side is a wipe
= cut, the carry is the contours). KEEPS -> WHAT WE = the `ripple` transition (letters expand into puddle rings). Type sits behind
the rock + figure (solid matte = near depth, thresholded) and in front of the sea; white spray is re-drawn over it (sheer luma key).
Pulses: contour brightness + type glow kick on every beat (downbeats harder), from the score grid. flow_wake: the sea pushes the
letters outside the holds (settle 0.75 inside). No caption.

    python 15_fusion/pieces/_gen_P11.py [--tune key=val ...]   -> P11.json
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _v004_lib import reformat, window_ops, hold_gate, song_paths, cli_song, load_grid, retime, rel, replan_state, analysis_for, tl, pulse, ramp_keys  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402

PID = "P11"
DESIGN_BPM = 75.0          # every time below is written on this grid (beat 0 at 0 s); retime() maps it onto the loaded grid
T = dict(near_thr=0.62, spray_lo=0.62, spray_hi=0.92, levels=9, cl_lo=0.06, cl_hi=0.94)


def build(tune=None, song=None, grid_path=None, fmt="9:16"):
    land = fmt == "16:9"
    T.update(tune or {})
    wav, gp = song_paths(PID, song, grid_path)
    g = load_grid(gp)
    at, dur = retime(g, DESIGN_BPM)
    cut, bpm = float(g["cut"]), float(g["bpm"])
    beats, dbs = [b for b in g["beats"] if b <= cut + 1e-6], g["downbeats"]
    font = "FacadeVF.ttf"              # liked tall face (user font rule); was Bodoni Moda in g01/g02
    ax = {"wght": 900}
    W = lambda **k: dict(font=font, module="lanes", axes=dict(ax), **k)  # noqa: E731
    words = [
        W(text="THE TIDE", lines=["THE TIDE"] if land else ["THE", "TIDE"], t0=at(0.0), t1=at(4.0), form_s=dur(1.6), exit_s=dur(0.8),
          hit_on="beat", form_in="lines", form_out="wipe", placement={"yaw": -8, "dy": -0.12, "w": 860 if land else 520}),
        dict(W(text="KEEPS", lines=["KEEPS"], t0=at(4.4), t1=at(7.2), form_s=dur(0.4), exit_s=dur(0.8), hit_on="beat",
          form_in="wipe", form_out="lines", placement={"yaw": 8, "dy": -0.10, "w": 420 if land else 340}), font="MoiraiOne-Regular.ttf", axes=None),
        W(text="WHAT WE", lines=["WHAT WE"] if land else ["WHAT", "WE"], t0=at(8.8), t1=at(12.0), form_s=dur(0.8), exit_s=dur(0.8),
          hit_on="bar", form_in="lines", form_out="wipe", placement={"yaw": -8, "dy": -0.15, "w": 860 if land else 520}),
        dict(W(text="LET GO", lines=["LET GO"] if land else ["LET", "GO"], t0=at(12.4), t1=at(16.0), form_s=dur(0.4), exit_s=dur(0.8),
          hit_on="bar", form_in="wipe", form_out="lines", placement={"yaw": 6, "dy": -0.17, "w": 900 if land else 520}), font="Ultra-Regular.ttf",
          axes=None),
    ]
    type_spec = {"seed": 11, "energy": 0.25, "mood": "calm", "tier": "calm",
                 "palette": {"bg": [8, 8, 10], "fg": [240, 242, 246], "accents": [[200, 205, 215], [150, 155, 165], [255, 255, 255]]},
                 "camera": "beat_orbit", "camera_params": {"read_dps": 3.0, "intro_deg": 18.0, "fast_dps": 14.0, "elev": 3.0, "bob": 0.5},
                 "transitions": [None, {"module": "ripple", "span": 0.95, "rings": 4}, None], "words": words, "line": {"px": 3.0}}
    holds = [(at(a), at(b)) for a, b in [(1.6, 3.2), (4.8, 6.4), (9.6, 11.2), (12.8, 15.2)]]
    hg = hold_gate(holds)

    # ---- timelines (piece seconds), sampled per frame
    pz = pulse(beats, dbs, amp=1.0, amp_db=1.7, decay=0.25)
    fill = ramp_keys([(at(a), v) for a, v in [(0, 0), (1.2, 0), (1.6, 1), (3.2, 1), (3.7, 0), (4.4, 0), (6.5, 0), (7.2, 1), (11.2, 1), (11.7, 0), (12.4, 0),
                      (12.8, 1), (15.2, 1), (15.7, 0), (16, 0)]])
    grow = ramp_keys([(at(a), v) for a, v in [(0, 26), (1.6, 3), (3.2, 3), (4.0, 24), (4.4, 24), (4.8, 3), (11.2, 3), (12.0, 24), (12.4, 24), (12.8, 3),
                      (15.2, 3), (16, 30)]])
    carry = ramp_keys([(at(a), v) for a, v in [(0, 0.55), (1.6, 0.0), (3.2, 0.0), (3.6, 1.0), (4.6, 1.0), (5.1, 0.0), (6.6, 0.0), (7.2, 0.8), (8.8, 0.8), (9.4, 0.0), (11.2, 0.0), (11.6, 1.0), (12.6, 1.0),
                       (13.1, 0.0), (15.2, 0.0), (15.8, 1.0), (16, 1.0)]])
    clgate = lambda t: min(1.0, 0.30 + 0.50 * carry(t) + 0.20 * pz(t))  # noqa: E731
    glow = lambda t: 0.05 + 0.22 * pz(t)  # noqa: E731
    phase = {"timeline": [[0.0, 0.0], [cut, 2.4]]}                     # 2.4 level steps over the piece: waves roll in slowly

    ops = [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 0.95, "gain": 0.95, "sat": 0.0}},
        {"op": "contour_lines", "out": "cld", "args": {"src": "$depth", "mode": "iso", "levels": T["levels"], "phase": phase, "width": 1,
                                                       "smooth": 3.0, "lo": T["cl_lo"], "hi": T["cl_hi"], "min_len": 30,
                                                       "frame_idx": "$frame_idx"}},
        # g02: + iso-lines of the smoothed footage luma = wave crests / foam edges that move with the sea itself
        {"op": "contour_lines", "out": "clw", "args": {"src": "$foot", "mode": "iso", "levels": 5, "width": 1, "smooth": 9.0,
                                                       "lo": 0.3, "hi": 0.8, "min_len": 80, "frame_idx": "$frame_idx"}},
        {"op": "screen", "out": "cl", "args": {"a": "$cld", "b": "$clw"}},
        {"op": "multiply", "out": "clv", "args": {"a": "$cl", "b": tl(clgate, cut)}},
        {"op": "over", "out": "image", "args": {"base": "$foot", "fg": [0.93, 0.95, 0.98], "alpha": "$clv"}},
        {"op": "type_from_edges", "out": "tfe", "args": {"edges": "$cl", "type_a": "$type_matte", "grow": tl(grow, cut, nd=1),
                                                         "line_width": 1, "density": 2.5, "fill": tl(fill, cut), "softness": 3}},
        {"op": "multiply", "out": "front", "args": {"a": "$tfe", "b": tl(hg, cut)}},      # in front of every matte while read
        {"op": "sandwich", "out": "image", "args": {"footage": "$image", "type_rgb": [0.96, 0.97, 1.0], "type_a": "$tfe",
                                                    "matte": "$depth", "matte_threshold": T["near_thr"], "matte_soft": 0.08,
                                                    "front": "$front"}},
        {"op": "luminosity_key", "out": "spray", "args": {"img": "$foot", "low": T["spray_lo"], "high": T["spray_hi"]}},
        {"op": "multiply", "out": "spray", "args": {"a": "$spray", "b": tl(lambda t: 0.85 * (1.0 - 0.9 * hg(t)), cut)}},
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$foot", "alpha": "$spray"}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$tfe", "radius": 14, "intensity": tl(glow, cut),
                                                    "color": [0.88, 0.92, 1.0]}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.02, "sat": 0.0}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 11, "amount": 0.03, "size": 1.3}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.32}},
    ]
    # g04 redesign (P10 bar): THE TIDE / WHAT WE are WINDOWS (dark plate, then the ground flips light), rock + figure step out in
    # front of the plate; KEEPS = Moirai One hairline outline over the sea, LET GO = Ultra (fat serif) solid, huge. Consecutive lines contrast.
    gi = next(i for i, o in enumerate(ops) if o["op"] == "add_glow")
    outline = ramp_keys([(at(a), v) for a, v in [(0, 0), (4.3, 0), (4.6, 1), (6.6, 1), (7.0, 0), (16, 0)]])
    extra = [{"op": "multiply", "out": "oline", "args": {"a": "$type_line", "b": tl(outline, cut)}},
             {"op": "over", "out": "image", "args": {"base": "$image", "fg": [0.97, 0.98, 1.0], "alpha": "$oline"}}]
    extra += window_ops([(at(1.6), at(4.0), [0.03, 0.03, 0.035], [0.92, 0.93, 0.95]),
                         (at(9.6), at(12.0), [0.88, 0.88, 0.86], [0.05, 0.05, 0.05])], cut, T["near_thr"],
                        type_a="$type_matte", line="$type_line")
    ops[gi:gi] = extra
    ops = PR.wake_ops(ops, holds, PID, {"curve": "onset", "lo": 3.0, "hi": 6.0})

    still = ("black and white photograph, a lone figure in a long dark coat standing on a dark sea rock in the lower middle of the "
             "frame, waves breaking around the rock, white spray in the air, overcast sky, the open sea filling the frame behind, "
             "horizon high in the frame, long lens, grainy silver gelatin print, tall vertical frame")
    piece = {
        "id": PID, "slug": "tidewater", "fps": 24, "version": "v004",
        "audio": {"src": rel(wav), "bpm": int(round(bpm)), "analysis": analysis_for(wav, gp), "grid": rel(gp),
                  "cut": cut, "fade": 0.4},
        "type": type_spec, "type_passes": ["matte", "line"],
        "stills": [{"id": "A", "model": "moss", "seed": 1101, "size": [1088, 1920], "prompt": still,
                    "negative": "text, watermark, logo, caption, colour, color, cartoon, illustration, 3d render, extra limbs, deformed, blurry"}],
        "shots": [{"id": "s1", "kind": "i2v", "first": "A", "frames": 385, "span": min(int(g["nframes"]), 385), "seed": 111,
                   "size": [1088, 1920], "depth": True, "flow": True,
                   "prompt": "black and white film, the camera slowly and steadily pushes in toward the lone figure standing on the sea "
                             "rock, one continuous slow dolly in, waves roll in and break around the rock, white spray drifts in the "
                             "wind, the figure stands still, the coat moves softly in the wind, overcast light, cinematic, filmic grain"}],
        "compose": {"crossfade": 0, "crf": 18, "default": {"ops": ops}},
        "signoff": {"text": "", "off": True},
        "parent": {"piece": None, "why": "v004 brief P11 (15_fusion/v004/BRIEFS.md) + reel-017 visual decisions"},
    }
    PR.fill_type_timing(piece, {"cut": cut, "bpm": bpm, "beats": g["beats"], "downbeats": g["downbeats"]})
    return piece


if __name__ == "__main__":
    # python _gen_P11.py [--song X.wav] [--grid X_grid.json] [--replan] [key=val ...]
    #   default song: 15_fusion/v004/audio/t2m/P11.wav + _grid.json when present, else the s1 score song.
    #   --replan: drop audio / type / compose / final from the runner state so the next run re-plans type + compose on the new grid
    #             and reuses the cached LTX footage, depth and flow.
    song, gridp, rest = cli_song(sys.argv[1:])
    alt = "--alt" in rest           # the other-aspect version (_w): own still + shot, same song / grid / plan
    tune = {k: float(v) for k, v in (a.split("=", 1) for a in rest if "=" in a)}
    p = build(tune, song, gridp)
    if alt:
        p = reformat(p, "16:9", "_w",
            ("black and white photograph, a wide seascape, a lone figure in a long dark coat standing on a dark sea rock on the right third "
             "of the frame, waves breaking around the rock, white spray in the air, the open sea and an overcast sky stretching away to the "
             "left, horizon in the upper third, long lens, grainy silver gelatin print, wide frame"),
            ("black and white film, the camera slowly and steadily pushes in toward the lone figure standing on the sea rock at the right of "
             "the frame, one continuous slow dolly in, waves roll in and break around the rock, white spray drifts in the wind, the figure "
             "stands still, the coat moves softly in the wind, overcast light, cinematic, filmic grain"))
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
