"""Shared by _gen_P16.py .. _gen_P20.py (v005 designs, 15_fusion/v005/DESIGNS.md; media + measured grids 15_fusion/v005/MEDIA.md).

A piece is written as DATA (words with their landing time, module, font, io, hold layout, contour ports, matte treatment, depth slot, camera
move, transition) and assembled with the planner's own machinery, so nothing here re-implements what the app already does:
  * timing   solve(): every word's HIT lands on the design's beat (design seconds -> the measured grid of the generated song), holds as long as the
             next landing allows (spec.plan_timing is the judge); warnings are printed when a hit or a readable hold cannot be kept
  * layout   tools/fusion/hold_layouts (placement wf / hf, screen_offset, camera fill) + weave_layout (backing shapes as the compose op)
  * ports    mograph/modules/contour_ports keys on the words (entry / exit / via dicts) + `plates` so spec.build_scene can bind them
  * matte    "matte": front | window | fill | wipe | glow -> plan_rules.weave_matte (tools/fusion/matte_treatments)
  * plate    "camera": {"chain": [...], "plate": "multiplane", "cards": 4} + word "slot" (mograph/render/multiplane)
  * flow     plan_rules.weave_flow (flow_wake in every chain), settle on the holds
  * blocks   the lego passages the piece is built on, recorded in plan.blocks (tools/fusion/blocks.py)
Deviations from a design that the machinery forces are written to piece["plan"]["deferred"] AND printed.

    python 15_fusion/pieces/_gen_P16.py [--check]          writes 15_fusion/pieces/P16.json   (--check = dry-run scene build, ~1 min)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from mograph import fonts_liked as fl  # noqa: E402
from mograph import fusion_brief as fb  # noqa: E402
from tools.fusion import blocks as bk  # noqa: E402
from tools.fusion import hold_layouts as hl  # noqa: E402
from tools.fusion import plan_rules as pr  # noqa: E402
from tools.fusion import spec as sp  # noqa: E402
from tools.fusion.beats import Beats  # noqa: E402

V5 = ROOT / "15_fusion" / "v005"
FOOT = "15_fusion/v005/out/footage"
FPS = 24
PORTS = {"matte": "matte_outline", "iso": "depth_iso", "edge": "luma_edges"}      # design words -> contour_ports kinds
NEG = ("text, letters, words, watermark, logo, caption, cartoon, illustration, 3d render, oversaturated, blurry, deformed")


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def load_grid(pid):
    g = json.loads((V5 / "audio" / f"{pid}_grid.json").read_text(encoding="utf-8"))
    return dict(bpm=float(g["bpm"]), beats=[float(b) for b in g["beats"]], downbeats=[float(d) for d in g["downbeats"]],
                bar_len=float(g["bar_len"]), cut=float(g["cut"]), nframes=int(g["nframes"]), fade=float(g.get("fade", 0.5)),
                key=g.get("key", ""), chord_changes=g.get("chord_changes", []), music_prompt=g.get("music_prompt", ""), src=f"15_fusion/v005/audio/{pid}.wav")


def analysis(pid, grid, write=True):
    """music_analysis curves (onset / low bands) on the song + its measured grid -> repo-relative path (None when the analysis cannot run)."""
    p = V5 / "local" / "audio" / f"{pid}_analysis.json"
    if write or not p.exists():
        try:
            from tools.fusion import music_analysis
            a = music_analysis.analyse(str(ROOT / grid["src"]), grid["bpm"], 4, FPS)
            a.update(bpm=grid["bpm"], bar_len=grid["bar_len"], beats=grid["beats"], downbeats=grid["downbeats"],
                     chord_changes=grid["chord_changes"] or a.get("chord_changes", []), first_bar_start=grid["downbeats"][0])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(a, indent=1), encoding="utf-8")
        except Exception as e:                    # noqa: BLE001
            print(f"  (no analysis json: {type(e).__name__}: {e})")
            return None
    return rel(p)


def lay(name, fmt="9:16", variant=None, seed=0, tag="v5"):
    """the hold-layout record `name`; with `variant` the first seed whose record has it (corner / strip kind / step pattern)."""
    for s in range(0, 400 if variant else 1):
        r = hl.build(name, seed + s, tag, fmt)
        if variant is None or r["variant"] == variant:
            return r
    raise ValueError(f"layout {name!r} has no variant {variant!r}")


HOLD_CAM = ("hold_drift", dict(fill=0.58, comp=[0.0, 0.0]))     # v005 round 2: a full-frame hold keeps the WHOLE word (3D depth, yaw) inside the frame: fill 0.7 + the off-centre comp clipped letters


def port(kind, d="up", **kw):
    """a contour port dict: kind = matte | iso | edge; d = up | along | around; kw = lead, glow, persist, pulse, travel ..."""
    return dict({"port": PORTS[kind], "dir": d}, **kw)


# ======================================================================================================== timing
class Infeasible(Exception):
    """overlap solver: the cycle between word `i` and `i + 1` cannot hold a readable line even with the compressed exit / transition / form."""

    def __init__(self, i, msg):
        super().__init__(msg)
        self.i = i


def solve_overlap(cfgs, grid, mfun, iv_ratio, tier, mood="mid", overlap="auto"):
    """Slow tempo (<= 90 bpm): every word's HIT on its design beat, hold >= max(1.2 s, readable), found in closed form.
    Cycle between hit i and hit i+1 = hold_i + exit_i + gap_i + form_{i+1}. The hold gets all the room the three others do not need; the three
    shrink from their preferred length toward the compressed floors (spec.OVERLAP_FLOORS) only as far as the readable hold requires.
    Raises Infeasible(i) when even the floors leave < the readable hold between hit i and hit i+1."""
    beats = Beats(grid["bpm"], grid["beats"], 4, 0.0, grid["cut"] + 4.0, grid["downbeats"])
    iv = beats.iv
    FL, NF = sp.OVERLAP_FLOORS, sp.NORMAL_FLOORS
    H = [beats.snap(mfun(c["hit"])) for c in cfgs]
    n = len(cfgs)
    io = [c.get("io", ("lines", "lines")) for c in cfgs]
    nomat = lambda i: io[i][0] in sp.NO_MATERIAL
    fmin = [FL["form_nomat"] if nomat(i) else FL["form"] for i in range(n)]
    fnorm = [NF["form_nomat"] if nomat(i) else NF["form"] for i in range(n)]
    hmin = [max(sp.HOLD_FLOOR, sp.min_hold(dict(text=c["text"], lines=c.get("lines")))) for c in cfgs]
    form, exit_, gap, hold, ovl = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    form[0] = H[0]                                               # the first word starts at 0
    for i in range(n - 1):
        C = H[i + 1] - H[i]
        carries = io[i][1] not in sp.NO_MATERIAL and io[i + 1][0] not in sp.NO_MATERIAL
        g_fix = None if carries else 0.5 * iv                    # plan_timing puts a cut there: half a beat, not ours to shrink
        pref = dict(f=min(0.8, 0.9 * iv), x=min(float(cfgs[i].get("exit_s", 0.6)), 0.9 * iv, 0.7),
                    g=g_fix if g_fix is not None else max(float(cfgs[i].get("gap", 0.5)), 0.4))
        flo = dict(f=fmin[i + 1], x=FL["exit"], g=g_fix if g_fix is not None else FL["gap"])
        room = C - hmin[i]
        if sum(flo.values()) > room + 1e-9:
            raise Infeasible(i, f"{cfgs[i]['text']} -> {cfgs[i + 1]['text']}: {C:.2f} s between the hits, readable hold {hmin[i]:.2f} s + "
                                f"compressed exit/transition/form {sum(flo.values()):.2f} s do not fit")
        if sum(pref.values()) <= room:
            use = dict(pref)
        else:                                                    # compress each phase toward its floor in proportion to its slack
            slack = {k: pref[k] - flo[k] for k in pref}
            need = sum(pref.values()) - room
            use = {k: pref[k] - (need * slack[k] / sum(slack.values()) if sum(slack.values()) > 0 else 0.0) for k in pref}
        form[i + 1], exit_[i], gap[i] = use["f"], use["x"], use["g"]
        hold[i] = C - use["f"] - use["x"] - use["g"]
        norm_sum = fnorm[i + 1] + NF["exit"] + (g_fix if g_fix is not None else NF["gap"])
        ovl[i] = max(0.0, norm_sum - (use["f"] + use["x"] + use["g"]))
        if ovl[i] > iv + 1e-9:
            raise Infeasible(i, f"{cfgs[i]['text']} -> {cfgs[i + 1]['text']}: overlap {ovl[i]:.2f} s exceeds one beat {iv:.2f} s")
    last = n - 1
    exit_[last] = min(float(cfgs[last].get("exit_s", 0.6)), 0.7)
    room = grid["cut"] - 0.05 - exit_[last] - H[last]
    if room < hmin[last]:                                        # the last exit may compress to its floor before a line is dropped
        exit_[last] = FL["exit"]
        room = grid["cut"] - 0.05 - exit_[last] - H[last]
    want = float(cfgs[last].get("hold", 2.0)) * iv_ratio
    hold[last] = max(hmin[last], min(max(want, hmin[last]), room))
    if room < hmin[last] - 1e-9:
        raise Infeasible(last - 1, f"{cfgs[last]['text']}: only {room:.2f} s between its hit and the cut for a {hmin[last]:.2f} s hold")
    words, S = [], 0.0
    for i in range(n):
        t1 = S + form[i] + hold[i] + exit_[i]
        words.append(dict(text=cfgs[i]["text"], t0=round(S, 4), t1=round(t1, 4), form_s=round(form[i], 4), exit_s=round(exit_[i], 4), hit_on="beat"))
        S = t1 + gap[i]
    base = {"tier": tier, "mood": mood, "bpm": grid["bpm"], "overlap": overlap}
    probe = [dict(w, form_in=io[i][0], form_out=io[i][1]) for i, w in enumerate(words)]
    tm = sp.plan_timing(probe, base, beats)
    report, warns = [], []
    for i, c in enumerate(cfgs):
        report.append((c["text"], round(tm[i]["hit"], 2), round(tm[i]["rel"], 2), round(tm[i]["end"], 2), round(H[i], 2)))
        if abs(tm[i]["hit"] - H[i]) > 1.0 / FPS:
            warns.append(f"{c['text']}: lands {tm[i]['hit']:.3f} s, design beat {H[i]:.3f} s (overlap solver off by {tm[i]['hit'] - H[i]:+.3f} s)")
    solve_overlap.last = dict(overlap=[round(x, 3) for x in ovl], hold=[round(x, 3) for x in hold])
    return [{k: w[k] for k in ("t0", "t1", "form_s", "exit_s", "hit_on")} for w in words], report, warns


def solve(cfgs, grid, mfun, iv_ratio, tier, mood="mid", overlap="auto"):
    """cfgs (design dicts) -> spec word timing {t0, t1, form_s, exit_s, hit_on}. mfun(design s) -> measured s on the song's grid.
    Slow tempo (spec.overlap_on): solve_overlap (closed form, hits on the design beats, hold >= 1.2 s)."""
    beats = Beats(grid["bpm"], grid["beats"], 4, 0.0, grid["cut"] + 4.0, grid["downbeats"])
    iv = beats.iv
    base = {"tier": tier, "mood": mood, "bpm": grid["bpm"], "overlap": overlap}
    if sp.overlap_on(base, beats):
        return solve_overlap(cfgs, grid, mfun, iv_ratio, tier, mood, overlap)
    words, report, warns = [], [], []
    n = len(cfgs)
    hits = [beats.snap(mfun(c["hit"])) for c in cfgs]
    for i, c in enumerate(cfgs):
        io = c.get("io", ("lines", "lines"))
        gap_prev = cfgs[i - 1].get("gap", 0.5) if i else 0.0
        t0 = (words[-1]["t1"] + max(gap_prev, 0.36)) if i else 0.0
        probe = dict(text=c["text"], t0=t0, t1=t0 + 3.0, form_s=0.5, exit_s=0.5, hit_on="beat", form_in=io[0], form_out=io[1])
        S = sp.plan_timing(words + [probe], base, beats)[i]["S"]
        form = max(hits[i] - S, 0.2)
        ex = min(float(c.get("exit_s", 0.6)), iv)             # an exit longer than a beat costs a whole extra beat once the engine snaps it
        if i + 1 < n:
            rel_max = hits[i + 1] - 0.55 - max(c.get("gap", 0.5), 0.36) - ex
        else:
            rel_max = grid["cut"] - ex - 0.25
        want = float(c.get("hold", 2.0)) * iv_ratio
        rel_top = max(min(hits[i] + want, (hits[i + 1] - 0.5 - ex) if i + 1 < n else grid["cut"] - ex - 0.25), hits[i] + 0.6)
        tries = [rel_top - m * iv / 2.0 for m in range(0, 40) if rel_top - m * iv / 2.0 >= hits[i] + 0.6 - 1e-9] or [hits[i] + 0.6]
        chosen = None
        for rel_t in tries:
            best = None
            for k in range(-10, 11):
                span = form + (rel_t - hits[i]) + ex + k * iv / 4.0
                cand = dict(probe, t1=t0 + max(span, 1.0), form_s=round(form, 4), exit_s=ex)
                tm = sp.plan_timing(words + [cand], base, beats)[i]
                err = 3.0 * abs(tm["hit"] - hits[i]) + abs(tm["rel"] - rel_t) + (2.0 if tm["end"] > grid["cut"] + 0.5 else 0.0)
                err += 5.0 if tm["rel"] - tm["hit"] < 0.8 * iv else 0.0                # never a zero hold (one beat is the least the engine plays)
                if best is None or err < best[0]:
                    best = (err, cand, tm)
            if i + 1 >= n:
                chosen = best
                break
            nio = cfgs[i + 1].get("io", ("lines", "lines"))
            nxt = dict(text=cfgs[i + 1]["text"], t0=best[1]["t1"] + max(c.get("gap", 0.5), 0.36), t1=best[1]["t1"] + 4.0, form_s=0.5, exit_s=0.5,
                       hit_on="beat", form_in=nio[0], form_out=nio[1])
            s_next = sp.plan_timing(words + [best[1], nxt], base, beats)[i + 1]["S"]
            chosen = best
            if s_next <= hits[i + 1] - (0.2 if nio[0] in sp.NO_MATERIAL else 0.46):
                break
        best = chosen
        _, cand, tm = best
        words.append(dict(text=c["text"], form_in=io[0], form_out=io[1], t0=round(cand["t0"], 4), t1=round(cand["t1"], 4), form_s=cand["form_s"],
                          exit_s=ex, hit_on="beat"))
        report.append((c["text"], round(tm["hit"], 2), round(tm["rel"], 2), round(tm["end"], 2), round(hits[i], 2)))
        if abs(tm["hit"] - hits[i]) > 0.3:
            warns.append(f"{c['text']}: lands {tm['hit']:.2f} s, the design beat is {hits[i]:.2f} s (exit + transition + form of the previous word need the room)")
        if tm["rel"] - tm["hit"] < max(1.0, sp.min_hold(dict(text=c["text"]))) - 0.25:
            warns.append(f"{c['text']}: hold {tm['rel'] - tm['hit']:.2f} s is shorter than the readable {sp.min_hold(dict(text=c['text'])):.1f} s (the design overlaps exit and transition)")
    return [{k: w[k] for k in ("t0", "t1", "form_s", "exit_s", "hit_on")} for w in words], report, warns


# ======================================================================================================== assembly
def build(d, check=False):
    """design dict -> piece dict (also written by the generator)."""
    pid, fmt = d["id"], "9:16"
    grid = load_grid(pid)
    music = dict(cut=grid["cut"], nframes=grid["nframes"], bpm=grid["bpm"], bar_len=grid["bar_len"], beats=grid["beats"], downbeats=grid["downbeats"],
                 fade=grid["fade"], src=grid["src"], key=grid["key"])
    apath = analysis(pid, grid, write=not d.get("no_audio_analysis"))
    if apath:
        try:
            music["curves"] = json.loads((ROOT / apath).read_text(encoding="utf-8")).get("curves") or {}
        except Exception:                          # noqa: BLE001
            pass
    iv_d = 60.0 / d["bpm_design"]
    iv_m = 60.0 / grid["bpm"]
    ad, am = d["anchor"]

    def mfun(t):
        return am + (t - ad) / iv_d * iv_m

    tier = fb.PACE_TIER[d["pace"]]
    ov_mode = d.get("overlap", "auto")                  # auto = slow tempo; a design may force True / False
    while True:
        try:
            timing, report, _ = solve(d["words"], grid, mfun, iv_m / iv_d, tier, d.get("mood", "mid"), ov_mode)
            if ov_mode == "auto" and min(r[2] - r[1] for r in report) < sp.HOLD_FLOOR - 0.02:      # beat-snapped timing cannot keep the readable
                ov_mode = True                          # hold at this tempo: switch the overlap mode on (and re-solve)
                continue
            break
        except Infeasible as e:                          # never go below the readable hold: drop the later line of the failing pair, say so
            drop = min(e.i + 1, len(d["words"]) - 1)
            gone = d["words"].pop(drop)
            d.setdefault("deferred", []).append(f"timing: line {gone['text']!r} dropped, {e} (merged into the previous line's hold)")
            if len(d["words"]) < 2:
                raise
    pal_hex = pr.palette_for(d.get("mood", "calm"), d.get("seed", 0), fb.DIRECTION_PALETTES[d["palette"]])
    rgb = [pr._hex_rgb(c) for c in pal_hex]
    cut = grid["cut"]

    words, cams, records = [], [], []
    for i, (c, tm) in enumerate(zip(d["words"], timing)):
        rec = (lay(c["layout"][0], fmt=fmt, variant=c["layout"][1]) if isinstance(c.get("layout"), tuple) else lay(c.get("layout", "full"), fmt=fmt))
        io = c.get("io", ("lines", "lines"))
        lf = fl.lookup(c["font"])
        assert lf is not None, f"{c['font']} is not a liked font"
        w = {"text": c["text"], "lines": c.get("lines") or [c["text"]], "font": lf["file"], "module": c["module"], **tm,
             "form_in": io[0], "form_out": io[1], "placement": dict({"yaw": c.get("yaw", -14 if i % 2 == 0 else 14)}, **hl.placement(rec))}
        if c.get("axes") is not None:
            w["axes"] = c["axes"]
        if c.get("params"):
            w["params"] = c["params"]
        if rec.get("box"):
            w["screen_offset"] = rec["offset"]
            w["ar_min"] = rec["ar_min"]
            if rec.get("max_lines"):
                w["max_lines"] = rec["max_lines"]
        w["hold_layout"] = rec
        for key in ("entry", "exit", "via", "edge_run"):
            if c.get(key):
                w[key] = c[key]
        if c.get("matte"):
            w["matte"] = c["matte"]
        if c.get("slot") is not None:
            w["slot"] = c["slot"]
        words.append(w)
        records.append(rec)
        mv = c.get("camera", "hold_drift")           # every line gets a move with its layout's framing (fill / comp), else the auto tiling would drop the layout
        fill_used = 0.7
        if mv:
            name, kw = (mv, {}) if isinstance(mv, str) else (mv[0], dict(mv[1]))
            fill_used = float(kw.get("fill", rec.get("fill") or 0.7)) if not rec.get("box") else 0.7
            if rec.get("anchored") and name != "hold_drift":
                d.setdefault("deferred", []).append(f"{c['text']}: camera {name} -> hold_drift (backing shapes are screen space)")
                name, kw = "hold_drift", {}
            cams.append(dict(module=name, word=i, **{**hl.camera_params(rec, name), **kw}))
        cap = hl.cap_frac(c["text"], rec, lf["file"], fmt, c.get("lines") or [c["text"]], fill_used)      # v005 round 2: the hold's letter height (face-on, flat) >= 7 % of the frame (45 px at 640)
        need = hl.MIN_CAP_FRAC if c.get("edge_run") else hl.min_cap(rec["name"])         # ... 4 % only for the edge_run block's line
        w["hold_cap_frac"] = round(cap, 4)
        print(f"  size: {c['text']:16s} {rec['name']}/{rec['variant']:8s} cap {cap * 100:.1f} % of the frame height ({cap * 640:.0f} px at 640), needs {need * 100:.0f} %")
        if cap < need - 1e-9 and not c.get("allow_small"):
            raise ValueError(f"{pid}: {c['text']!r} on layout {rec['name']} has a letter height of {cap * 100:.1f} % of the frame (< {need * 100:.0f} %): pick a bigger layout")
    cams.extend(copy.deepcopy(d.get("extra_chain", [])))        # explicit extra moves (window given): e.g. follow in a transition
    transitions = [c.get("transition") for c in d["words"][:-1]]

    shots, stills, plates = [], [], []
    for s in d["shots"]:
        sid = s["id"]
        fdir = f"{FOOT}/{pid}"
        media = {"shot": f"{fdir}/{pid}_{sid}_g01.mp4", "depth": f"{fdir}/{pid}_{sid}_depth_g01.mp4", "matte": f"{fdir}/{pid}_{sid}_matte_g01.mp4"}
        shots.append({"id": sid, "kind": "i2v", "first": s["still"], "frames": s["frames"], "span": s["span"], "seed": s["seed"], "size": [384, 672],
                      "matte": True, "depth": True, "flow": True, "prompt": s["prompt"], "media": media})
        plates.append({"plate": media["shot"], "matte": media["matte"], "depth": media["depth"], "frames": s["span"]})
    for st in d["stills"]:
        stills.append({"id": st["id"], "model": "moss", "seed": st["seed"], "size": [1088, 1920], "negative": NEG + st.get("neg", ""), "prompt": st["prompt"],
                       "media": f"{FOOT}/{pid}/{pid}_still_{st['id']}_g01.png"})

    camera = dict({"chain": cams, "plate": "multiplane", "cards": d.get("cards", 4)}, **d.get("camera_extra", {}))
    ts = {"seed": d.get("seed", 0), "energy": pr.energy_of(music), "mood": d["pace"], "tier": tier,
          "palette": {"bg": rgb[0], "fg": rgb[1], "accents": rgb[2:]}, "camera": camera, "transitions": transitions,
          "transition_names": pr.transition_names(), "words": words, "fps": FPS, "duration": round(cut, 4), "bpm": round(grid["bpm"], 3), "bar": 4,
          "beats": [round(b, 4) for b in grid["beats"]], "downbeats": [round(b, 4) for b in grid["downbeats"]], "plates": plates, "line": {"px": 4.0}}
    if ov_mode is True:
        ts["overlap"] = True                            # plan_timing then uses the same overlap mode as the solver (slow tempos switch it on by themselves)
    tech = d["technique"]
    recipe = pr.load_recipe(tech)
    sizes = pr.sizes_for("preview", fmt)
    beats = pr.make_grid(music, cut)
    comp = pr.apply_recipe(recipe, cut, pal_hex, sizes["compose"], beats)
    for ch in [comp["default"]] + list(comp.get("shots", {}).values()):
        if not any(o.get("out") == "m" for o in ch["ops"]):
            ch["ops"].insert(0, {"op": "upscale_matte", "out": "m", "args": {"matte": "$matte", "size": list(sizes["compose"]), "guide": "$footage"}})
    piece = {"id": pid, "slug": d["slug"], "fps": FPS, "version": "v005",
             "audio": {"src": grid["src"], "bpm": round(grid["bpm"], 3), "analysis": apath, "cut": round(cut, 4), "fade": round(grid["fade"], 3)},
             "type": ts, "type_passes": list(recipe["type_passes"]), "stills": stills, "shots": shots,
             "compose": {"size": list(sizes["compose"]), "crossfade": 0, "crf": 20, **comp},
             "signoff": {"off": True, "text": "", "t0": round(cut - 1.0, 2), "font": "Space Grotesk", "px": 38, "color": rgb[1]},
             "technique": tech, "quality": "preview", "format": fmt, "sizes": sizes,
             "brief": {"scene": d["scene"], "mood": [d["pace"]], "palette": pal_hex},
             "plan": {"version": 1, "tier": tier, "shots": "cut" if len(shots) > 1 else "single", "energy": ts["energy"], "look": "hand-assembled from lego",
                      "bars": len(grid["downbeats"]), "notes": [], "seed": d.get("seed", 0)},
             "parent": {"piece": None, "why": d["why"]}}
    piece = pr.finalize_piece(piece, music)
    names = [b for b in d["blocks"] if b in bk.BLOCKS]
    piece["plan"]["blocks"] = bk.describe(names, d.get("seed", 0), len(words), tag=pid)
    piece["plan"]["layouts"] = [r["name"] + "/" + r["variant"] for r in records]
    piece["plan"]["direction"] = {"pace": d["pace"], "palette": d["palette"], "type_role": d["type_role"], "camera": d["camera_lead"], "shots": piece["plan"]["shots"]}
    # the REAL timeline: the type scene built with the contour rides in (their durations move later words by a beat or two), no camera (fast)
    scene = sp.build_scene(dict(copy.deepcopy(piece["type"]), camera=None), size=tuple(sizes["type"]))
    chk = {"words": [{k: w[k] for k in ("start", "hit", "release", "end")} for w in scene.info["words"]], "notes": scene.info["notes"],
           "last_end": max(w["end"] for w in scene.info["words"])}
    report = [(c["text"], w["hit"], w["release"], w["end"], round(beats.snap(mfun(c["hit"])), 2)) for c, w in zip(d["words"], chk["words"])]
    ovm = sp.overlap_on({"bpm": grid["bpm"], "overlap": ov_mode}, beats)
    for c, w in zip(d["words"], chk["words"]):
        if abs(w["hit"] - beats.snap(mfun(c["hit"]))) > (1.5 / FPS if ovm else 0.3):
            d.setdefault("deferred", []).append(f"timing: {c['text']} lands {w['hit']:.2f} s, the design beat is {beats.snap(mfun(c['hit'])):.2f} s "
                                                f"(exit + transition + form of the previous word need the room)")
        need = max(sp.HOLD_FLOOR if ovm else 1.0, sp.min_hold(dict(text=c["text"], lines=c.get("lines"))))
        if w["release"] - w["hit"] < need - (0.02 if ovm else 0.25):
            d.setdefault("deferred", []).append(f"timing: {c['text']} holds {w['release'] - w['hit']:.2f} s, shorter than the readable {need:.1f} s "
                                                f"(the design overlaps exit and transition)")
    if chk["last_end"] > cut + 0.05:
        d.setdefault("deferred", []).append(f"timing: the last word ends {chk['last_end']:.2f} s, after the cut {cut:.2f} s (its exit is cut off)")
    pr.weave_layout(piece, music, chk)
    pr.weave_matte(piece, music, chk)
    pr.weave_flow(piece, music, chk)
    piece["plan"]["contour"] = {"available": True, "subject": True, "lines": [{k: w[k] for k in ("entry", "exit", "via", "matte") if k in w} for w in words]}
    piece["plan"]["timing"] = [dict(zip(("text", "hit", "release", "end", "wanted_hit"), r)) for r in report]
    for hook in d.get("post", []):                         # block hooks that need the woven recipe + timing: strip_resnap.weave, edge_run.weave
        hook(piece, music, chk)
    piece["plan"]["deferred"] = list(d.get("deferred", []))
    for msg in piece["plan"]["deferred"]:
        print("  DEFERRED:", msg)
    piece["plan"]["scene_notes"] = chk["notes"]
    print("  scene: notes", chk["notes"] or "none", "| last word ends", round(chk["last_end"], 2), "of", round(cut, 2))
    return piece


def legible_camera(piece, music, chk, modules=("push_through", "graze", "tilt_to_plan", "ease_reversal", "follow"), lead=0.5, settle=0.6):
    """post hook (d["post"]): camera MOVES that are not at rest on the hold pose are kept OFF the hold (v005 QC: P17 CLIMB 'MB' / THE SPIRAL cropped
    because push_through dives through the counter at the middle of the hold; P18 CARVE graze had the word off-frame at the hold midpoint).
    graze / tilt_to_plan / ease_reversal: window [word start, hit] (the move runs while the word forms, hold_drift frames the hold);
    follow on a word: kept as the move of the transition INTO that word (its default job; it never runs on a hold);
    push_through (v005 round 2): only on the LAST word, window [release - lead, end of the piece], one dive through the counter of the final
    word. On any other word the dive ends BEHIND the plane the next word sits on (every word of a piece sits on the same plane, slot z), so the
    route to the next hold pose U-turns through the type (P17 g03: THE SPIRAL mirrored, then seen from inside a letter); such an entry is dropped
    and the default tiling (follow through transitions, hold_drift on the holds) stays."""
    chain = piece["type"]["camera"]["chain"]
    ws = chk["words"]
    last = len(ws) - 1
    for e in list(chain):
        i, m = e.get("word"), e.get("module")
        if i is None or m not in modules or "window" in e:
            continue
        if m == "follow":
            e.pop("word")
            e["window"] = f"trans:{max(i - 1, 0)}"
            piece["plan"].setdefault("notes", []).append(f"camera follow word {i}: window {e['window']} (the transition into the word, legible_camera)")
            continue
        if m == "push_through":
            if i != last:
                chain.remove(e)
                piece["plan"].setdefault("notes", []).append(f"camera push_through word {i}: dropped (no word beyond the dived plane, legible_camera)")
                continue
            a, b = ws[i]["release"] - lead, chk.get("T") or piece["type"]["duration"]
            e["to_next"] = False
            e.setdefault("beyond", 0.2)
            for e2 in chain:                              # the hold_drift of this word ends where the dive starts (overlapping windows would fight)
                if e2 is not e and e2.get("word") == i and e2.get("module") == "hold_drift":
                    e2.pop("word")
                    e2["window"] = [round(ws[i]["start"], 3), round(a, 3)]
        else:
            a, b = ws[i]["start"], ws[i]["hit"] - settle                # the move ends `settle` s before the hit: the join to the hold pose relaxes over ~1-2 s
        if b - a < 0.5:
            continue
        e.pop("word")
        e["window"] = [round(a, 3), round(b, 3)]
        piece["plan"].setdefault("notes", []).append(f"camera {m} word {i}: window {e['window']} (off the hold, legible_camera)")
        if m != "push_through":                            # the hold itself: a composed hold_drift with the layout's framing (the default tiling would use fill 0.7 + the off-centre comp)
            rec = piece["type"]["words"][i].get("hold_layout") or {}
            kw = hl.camera_params(rec, "hold_drift") if rec.get("box") else dict(HOLD_CAM[1])
            e.setdefault("fill", kw.get("fill", 0.7))              # the move frames the word like the hold, so the join has little to relax
            chain.append(dict(module="hold_drift", window=[round(b, 3), round(ws[i]["release"], 3)], **kw))


def main(design, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry-run scene build (timeline notes)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    piece = build(design, check=a.check)
    out = Path(a.out) if a.out else HERE / f"{design['id']}.json"
    out.write_text(pr.piece_json(piece), encoding="utf-8")
    print("wrote", out, "|", [r[0] + "@" + str(r[1]) for r in [(t["text"], t["hit"]) for t in piece["plan"]["timing"]]])
    return piece
