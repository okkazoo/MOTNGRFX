"""P15 Reeds (v004): golden, gentle, 90 bpm score grid, 16 s, 9:16, ONE continuous i2v shot (385 f, slow forward push through reeds).

Words THROUGH / THE TALL / GOLD / GRASS. Hits 2.67 (downbeat) / 8.0 (downbeat) / 10.67 (downbeat) / 14.67 (beat 3), holds 2-3 beats.
Transitions (no repeats): THROUGH -> THE TALL = `ripple` (leads, P11 leads with the contour exit); THE TALL -> GOLD = scribble carry
(engine cut; the letters dissolve into the hand-drawn scribbles around the near stalks and GOLD condenses out of them);
GOLD -> GRASS = `fan`. Weave: the type sits BETWEEN the reeds: near stalks (depth above `near_thr`, solid threshold) pass in front,
the glowing seed heads (bright luma key, sheer) drift over it; scribble lines (contour_lines scribble on the near-depth outline,
re-drawn every 4 frames) flash on every beat (downbeats harder). Camera track: the type rides the footage's global drift (cam_ops),
flow_wake pushes it outside the holds. No caption.

    python 15_fusion/pieces/_gen_P15.py [key=val ...]   -> P15.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _v004_lib import reformat, window_ops, camtrack, cam_rebase, hold_gate, song_paths, cli_song, load_grid, retime, rel, replan_state, analysis_for, tl, pulse, ramp_keys  # noqa: E402
from tools.fusion import plan_rules as PR  # noqa: E402
from tools.fusion import recipe_eval as RE  # noqa: E402

PID = "P15"
DESIGN_BPM = 90.0          # every time below is written on this grid (beat 0 at 0 s); retime() maps it onto the loaded grid
T = dict(near_thr=0.86, seed_lo=0.80, seed_hi=0.97, scr_thr=0.45, tr_alpha=0.6)   # g02: near 0.70 hid most of the frame


def build(tune=None, song=None, grid_path=None, fmt="9:16"):
    land = fmt == "16:9"
    pid = PID + ("_w" if land else "")
    T.update(tune or {})
    wav, gp = song_paths(PID, song, grid_path)
    g = load_grid(gp)
    at, dur = retime(g, DESIGN_BPM)
    cut, bpm = float(g["cut"]), float(g["bpm"])
    b = 60.0 / bpm
    beats, dbs = [x for x in g["beats"] if x <= cut + 1e-6], g["downbeats"]
    font = "BadeenDisplay-Regular.ttf"   # liked tall face; THE TALL / GRASS in fat faces (Erica One / Ultra): tall vs puffy (font rule)
    W = lambda **k: dict(dict(font=font, module="lanes"), **k)  # noqa: E731
    words = [
        W(text="THROUGH", lines=["THROUGH"], t0=at(0.0), t1=at(5.333), form_s=dur(2.667), exit_s=dur(0.667), hit_on="bar",
          form_in="lines", form_out="lines", placement={"yaw": -8, "dy": -0.04, "dx": 0.03, "w": 880 if land else 520}),
        W(text="THE TALL", lines=["THE", "TALL"], t0=at(6.667), t1=at(10.0), form_s=dur(1.333), exit_s=dur(0.667), hit_on="bar",
          form_in="lines", form_out="wipe", placement={"yaw": 8, "dy": -0.08, "dx": 0.08, "w": 420 if land else 300}, font="EricaOne-Regular.ttf"),
        dict(W(text="GOLD", lines=["GOLD"], t0=at(10.333), t1=at(12.667), form_s=dur(0.333), exit_s=dur(0.667), hit_on="bar",
          form_in="wipe", form_out="lines", placement={"yaw": -6, "dy": -0.02, "dx": 0.03, "w": 860 if land else 520})),
        W(text="GRASS", lines=["GRASS"], t0=at(14.0), t1=at(16.667), form_s=dur(0.667), exit_s=dur(0.667), hit_on="beat",
          form_in="lines", form_out="lines", placement={"yaw": 6, "dy": -0.04, "dx": 0.05, "w": 760 if land else 500}, font="Ultra-Regular.ttf"),
    ]
    type_spec = {"seed": 15, "energy": 0.3, "mood": "calm", "tier": "calm",
                 "palette": {"bg": [30, 20, 8], "fg": [255, 240, 205], "accents": [[255, 196, 110], [240, 160, 70], [255, 225, 160]]},
                 "camera": "beat_orbit", "camera_params": {"read_dps": 3.0, "intro_deg": 16.0, "fast_dps": 12.0, "elev": 3.0, "bob": 0.5},
                 "transitions": [{"module": "ripple"}, None, {"module": "spiral_around"}], "words": words, "line": {"px": 3.0}}
    holds = [(at(a), at(b)) for a, b in [(2.667, 4.667), (8.0, 9.333), (10.667, 12.0), (14.667, 16.0)]]
    hg = hold_gate(holds)

    pz = pulse(beats, dbs, amp=1.0, amp_db=1.6, decay=0.2)
    fill = ramp_keys([(at(a), v) for a, v in [(0, 0), (2.2, 0), (2.667, 1), (6.6, 1), (7.4, 0), (9.333, 0), (9.8, 0), (10.333, 0), (10.667, 1), (16, 1)]])
    grow = ramp_keys([(at(a), v) for a, v in [(0, 22), (2.667, 3), (9.333, 3), (10.0, 22), (10.333, 22), (10.667, 3), (16, 3)]])
    carry = ramp_keys([(at(a), v) for a, v in [(0, 0.6), (2.667, 0.0), (9.333, 0.0), (9.7, 1.0), (10.6, 1.0), (11.0, 0.0), (16, 0.0)]])
    scr_gate = lambda t: min(1.0, 0.10 + 0.55 * pz(t) + 0.7 * carry(t))  # noqa: E731
    glow = lambda t: 0.08 + 0.22 * pz(t)  # noqa: E731
    gap = ramp_keys([(at(a), v) for a, v in [(0, 0), (4.667, 0), (5.333, 1), (6.667, 1), (8.0, 0), (12.0, 0), (12.667, 1), (14.0, 1),
                                             (14.667, 0), (16, 0)]])
    trgate = lambda t: 1.0 - (1.0 - T["tr_alpha"]) * gap(t)  # noqa: E731

    ops = RE.cam_ops(layers=("type_matte", "type_line"), gain=1.0, ref="$flow", prefix="cam_")
    tr = camtrack(pid)                                 # g02: re-base the track at every hit (g01 drifted ~90 px off the left edge)
    if tr:
        hits = [h for h, _ in holds]
        bounds = [at(6.0), at(10.167), at(13.333)]     # mid-gaps between the words
        f = cam_rebase(tr, hits, bounds)
        for lay in ("cam_type_matte", "cam_type_line"):
            ops.append({"op": "translate", "out": lay, "args": {"img": "$" + lay,
                        "dx": tl(lambda t: f(t, 0), cut, nd=1), "dy": tl(lambda t: f(t, 1), cut, nd=1)}})
    body = [
        {"op": "grade", "out": "foot", "args": {"frame": "$footage", "lift": 0.0, "gamma": 1.0, "gain": 0.98, "sat": 1.08}},
        {"op": "contour_lines", "out": "scr", "args": {"src": "$foot", "mode": "scribble", "thresh": T["scr_thr"], "max_contours": 7,
                                                       "strokes": 3, "jitter": 3.0, "boil": 4, "width": 1, "smooth": 2.5,
                                                       "min_len": 40, "seed": 15, "frame_idx": "$frame_idx"}},
        {"op": "multiply", "out": "tmg", "args": {"a": "$type_matte", "b": tl(trgate, cut)}},
        {"op": "type_from_edges", "out": "tfe", "args": {"edges": "$scr", "type_a": "$tmg", "grow": tl(grow, cut, nd=1),
                                                         "line_width": 1, "density": 3.0, "fill": tl(fill, cut), "softness": 3}},
        {"op": "multiply", "out": "front", "args": {"a": "$tfe", "b": tl(hg, cut)}},      # in front of every matte while read
        {"op": "sandwich", "out": "image", "args": {"footage": "$foot", "type_rgb": [1.0, 0.95, 0.84], "type_a": "$tfe",
                                                    "matte": "$depth", "matte_threshold": T["near_thr"], "matte_soft": 0.08,
                                                    "front": "$front"}},
        {"op": "luminosity_key", "out": "seeds", "args": {"img": "$foot", "low": T["seed_lo"], "high": T["seed_hi"]}},
        {"op": "multiply", "out": "seeds", "args": {"a": "$seeds", "b": tl(lambda t: 0.75 * (1.0 - 0.9 * hg(t)), cut)}},
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": "$foot", "alpha": "$seeds"}},
        {"op": "multiply", "out": "scrv", "args": {"a": "$scr", "b": tl(scr_gate, cut)}},
        {"op": "over", "out": "image", "args": {"base": "$image", "fg": [1.0, 0.86, 0.55], "alpha": "$scrv"}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$tfe", "radius": 16, "intensity": tl(glow, cut),
                                                    "color": [1.0, 0.78, 0.42]}},
        {"op": "add_glow", "out": "image", "args": {"img": "$image", "mask": "$scrv", "radius": 8, "intensity": 0.25,
                                                    "color": [1.0, 0.75, 0.4]}},
        {"op": "grade", "out": "image", "args": {"frame": "$image", "lift": 0.01, "gamma": 1.0, "gain": 1.02, "sat": 1.05}},
        {"op": "grain", "out": "image", "args": {"frame": "$image", "frame_idx": "$frame_idx", "seed": 15, "amount": 0.02, "size": 1.2}},
        {"op": "vignette", "out": "image", "args": {"frame": "$image", "strength": 0.25}},
    ]
    # g03 redesign (P10 bar): THROUGH / GOLD are WINDOWS (cream plate, then a near-black plate: the ground flips), the near reed
    # wall steps out in front of the plate, outline in the plate's contrast colour; THE TALL = Erica One, small, outline over the
    # reeds; GRASS = Ultra solid between the reeds. Consecutive lines contrast in face, scale and fill.
    gi = next(i for i, o in enumerate(body) if o["op"] == "add_glow")
    outline = ramp_keys([(at(a), v) for a, v in [(0, 0), (6.6, 0), (7.4, 1), (9.4, 1), (9.8, 0), (16, 0)]])
    extra = [{"op": "multiply", "out": "oline", "args": {"a": "$type_line", "b": tl(outline, cut)}},
             {"op": "over", "out": "image", "args": {"base": "$image", "fg": [1.0, 0.93, 0.8], "alpha": "$oline"}}]
    extra += window_ops([(at(2.667), at(5.333), [0.96, 0.90, 0.78], [0.25, 0.14, 0.05]),
                         (at(10.667), at(12.667), [0.06, 0.04, 0.02], [1.0, 0.8, 0.45])], cut, T["near_thr"],
                        type_a="$type_matte", line="$type_line")
    body[gi:gi] = extra
    body = PR.wake_ops(body, holds, pid, {"curve": "onset", "lo": 2.5, "hi": 5.0})
    for o in body:                                     # the wake displaces the camera-tracked type
        if o["op"] == "flow_wake":
            o["args"]["type_a"] = "$cam_type_matte" if o["out"] == "wm" else "$cam_type_line"
    ops += body

    still = ("photograph from inside a field of tall golden reeds at sunset, the camera low among the stalks, out of focus stalks close "
             "to the lens at the left and right edges, seed heads glowing in the low backlit sun, warm golden haze, soft bokeh, a gap "
             "between the reeds leading into the distance, cinematic 35mm, tall vertical frame")
    piece = {
        "id": PID, "slug": "reeds", "fps": 24, "version": "v004",
        "audio": {"src": rel(wav), "bpm": int(round(bpm)), "analysis": analysis_for(wav, gp), "grid": rel(gp),
                  "cut": cut, "fade": 0.4},
        "type": type_spec, "type_passes": ["matte", "line"],
        "stills": [{"id": "A", "model": "moss", "seed": 1501, "size": [1088, 1920], "prompt": still,
                    "negative": "text, watermark, logo, caption, people, person, cartoon, illustration, 3d render, oversaturated, blurry"}],
        "shots": [{"id": "s1", "kind": "i2v", "first": "A", "frames": 385, "span": min(int(g["nframes"]), 385), "seed": 151,
                   "size": [1088, 1920], "depth": True, "flow": True,
                   "prompt": "the camera pushes slowly and steadily forward through the tall golden reeds at sunset, one continuous "
                             "forward move, close stalks slide past the lens on both sides, seed heads glow in the backlight and sway "
                             "softly in the breeze, warm golden haze, floating seeds, cinematic, shallow depth of field"}],
        "compose": {"crossfade": 0, "crf": 18, "default": {"ops": ops}},
        "signoff": {"text": "", "off": True},
        "parent": {"piece": None, "why": "v004 brief P15 (15_fusion/v004/BRIEFS.md) + reel-017 visual decisions"},
    }
    PR.fill_type_timing(piece, {"cut": cut, "bpm": bpm, "beats": g["beats"], "downbeats": g["downbeats"]})
    return piece


if __name__ == "__main__":
    # python _gen_P15.py [--song X.wav] [--grid X_grid.json] [--replan] [key=val ...]
    #   default song: 15_fusion/v004/audio/t2m/P15.wav + _grid.json when present, else the s1 score song.
    #   --replan: drop audio / type / compose / final from the runner state so the next run re-plans type + compose on the new grid
    #             and reuses the cached LTX footage, depth and flow.
    song, gridp, rest = cli_song(sys.argv[1:])
    alt = "--alt" in rest           # the other-aspect version (_w): own still + shot, same song / grid / plan
    tune = {k: float(v) for k, v in (a.split("=", 1) for a in rest if "=" in a)}
    p = build(tune, song, gridp)
    if alt:
        p = reformat(p, "16:9", "_w",
            ("wide photograph from inside a field of tall golden reeds at sunset, the camera low among the stalks, out of focus stalks close "
             "to the lens at the left edge and on the right third, a wide gap through the reeds opening onto the low sun on the horizon, "
             "seed heads glowing in the backlight, warm golden haze, soft bokeh, cinematic 35mm, wide frame"),
            ("the camera pushes slowly and steadily forward through the tall golden reeds at sunset, one continuous forward move, close "
             "stalks slide past the lens on both sides, seed heads glow in the backlight and sway softly in the breeze, warm golden haze, "
             "floating seeds, cinematic, shallow depth of field"))
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
