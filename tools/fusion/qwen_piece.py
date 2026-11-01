"""qwen_piece: an open model designs the TYPE + COMPOSE of an existing piece; the engine renders it over the existing footage.

Design phase only (20_open-model-design v002). No Claude in the loop: code + the model.

    python tools/fusion/qwen_piece.py package P06 --seed 1 --out DIR            # input package only (prompt + images), no network
    python tools/fusion/qwen_piece.py design  P06 --model qwen/design_llm_8b --seed 1 --version v004 --gen 01 --out DIR
    python tools/fusion/qwen_piece.py seed-state 15_fusion/pieces/P06_v004.json   # state.json seed so the stock runner sees every adopted kind as done
    python tools/fusion/qwen_piece.py media-check 15_fusion/pieces/P06_v004.json

What the model gets (build_package): the words + lyrics, the music analysis as text (bpm, key, chords, downbeats, dropouts), the shot list
(start, duration, one-line description), a 6-frame contact sheet per shot (from the existing colour clips) and a matte sheet, the fixed
design guide (16_app/DESIGN_GUIDE_open_model.md), the module menu, the liked fonts (mograph/fonts/liked.json) and the type / compose schema.

What it may change (TYPE + COMPOSE only): type.{mood, energy, palette, camera, camera_params, transitions, line, words[]} and a compact
compose description (technique, plate grade, glow, grain, vignette, type opacity, freeze holds) that code turns into compose ops from the
repo's recipes (plan_rules.load_recipe / apply_recipe) plus the per-word matte treatments (plan_rules.weave_matte). LOCKED, copied back from
the original JSON by merge_locked(): audio, fps, stills, shots (only `media` is ADDED), type_passes, type.seed, the word texts and their order,
compose size / crossfade / crf. The caption (signoff) is switched off (user rule: no title / BPM caption).

Validation: the repo's own loaders (fonts_liked.require_liked, spec.build_scene via plan_rules.dry_run, plan_rules timing) plus range checks.
Errors go back to the SAME model for up to 2 repair turns.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import design_options as DO
from tools.fusion import openings as OP      # noqa: E402  text-as-window choices: schema, menu, validator, piece writer
from tools.fusion import owner as OW               # noqa: E402  one owner per word (flat | ltx | handoff): fields, defaults, windows
from tools.fusion import still_palette as SP        # noqa: E402  the palette extracted from the piece's own first still
from tools.fusion import stable_type as ST          # noqa: E402  every word lands centred and holds still for >= 1 beat (stacked / rolled layouts)
from tools.fusion import piece_system as PS        # noqa: E402  the piece's seeded system (module, io pair, faces, palette, camera, yaw band): written into the prompt, outliers folded into it
from tools.fusion import why_fields as WF          # noqa: E402  a `why` per decision (par round r6): rules text, scoring; the compilers ignore the fields

GUIDE = ROOT / "16_app" / "DESIGN_GUIDE_open_model.md"
FPS = 24

# ----------------------------------------------------------------------------------------------------------- the pieces
# src = the JSON whose footage is reused; ana = the analysis the real run used; media = what exists on disk (repo-relative).
PIECES = {
    "P06": dict(
        src="15_fusion/pieces/P06_v003.json", parent_version="v003", title="Song Chords", mood_brief="kinetic and emotional",
        ana="15_fusion/v003/local/pieces/P06/analysis.json", state="15_fusion/v003/local/pieces/P06/state.json",
        key_text="C minor (the analysis tool reports its relative major, D# major)",
        music_text="instrumental excerpt of a Song2 rhythmic-lead track (user audio), one continuous piece, no sung lyrics",
        lyrics="none (instrumental)", seed_shift=0,
        ing="15_fusion/v003/ingredients/P06", clip="motion-graphic_fusion_v003_P06_{sid}_1_00001_.mp4",
        still="15_fusion/v003/out/motion-graphic_fusion_v003_P06_still_{sid}_1_00001_.png",
        matte_dir="15_fusion/v003/ingredients/P06/matte_{sid}",
        masktrack="15_fusion/v003/local/pieces/P06/masktrack/mt_{sid}_composite.mp4",
        first_version=4,
        fps_note="runner subject in a lit tunnel, then a street at dawn, then a rooftop at twilight"),
    "P10": dict(
        src="15_fusion/pieces/P10.json", parent_version="v002", title="Noir Step", mood_brief="kinetic, tense, noir",
        ana="15_fusion/v002/local/pieces/P10/analysis.json", state="15_fusion/v002/local/pieces/P10/state.json",
        key_text="B minor (the analysis tool reports B major; the chords are B, F#sus4 and Bm)",
        music_text="trip-hop / film-noir instrumental: dusty swung breakbeat, upright bass walking line, muted trumpet, smoky Rhodes, vinyl crackle, cool and tense, steady lurching groove",
        lyrics="none (instrumental)", seed_shift=3,
        ing="15_fusion/v002/ingredients/P10", clip="motion-graphic_fusion_v002_P10_{sid}_1_00001_.mp4",
        still="15_fusion/v002/ingredients/P10/motion-graphic_fusion_v002_P10_still_{sid}_1_00001_.png",
        matte_dir="15_fusion/v002/ingredients/P10/matte_{sid}", masktrack=None,
        first_version=3,
        fps_note="black and white film noir"),
}
MODELS = [  # fixed order = version order inside a piece: two seeds per model
    ("qwen/design_llm_8b", "design_llm_8b", "design_llm_8b"),
    ("qwen/qwen3.5-9b", "qwen3.5-9b", "Qwen3.5-9B"),
    ("google/gemma-4-31b-it", "gemma-4-31b", "Gemma 4 31B"),
    ("google/gemma-3-12b-it", "gemma-3-12b", "Gemma 3 12B"),
]

# ----------------------------------------------------------------------------------------------------------- allowed values
WORD_MODULES = ["lanes", "pipework", "circuit", "depth_echo", "hollow_tubes", "topo_tower", "hinge_unfold", "xray_unfold", "candy_orbit",
                "depth_tunnel", "beat_bars", "bold_thin_echo", "gradient_fade", "scale_clash", "slats", "solid_hollow", "staircase", "spectrum",
                "tangent_spray", "fly_into", "voxel_orbit", "voxel_roll"]        # `particles` (a dot-cloud word) retired 2026-11-01: letters are never a speckle (design_defaults.SPECKLE_MODULES); `outflow` retired 2026-11-02 (design_defaults.RETIRED_MODULES)
MODULE_NOTES = {
    "lanes": "strand: hairline lanes peel off, draw the outline, fill",
    "pipework": "strand: shaded tubes with chamfered bends", "circuit": "strand: PCB traces with pulse heads (pairs with circuit_board)",
    "depth_echo": "mech 3D: outline copies fan out behind the word", "hollow_tubes": "mech 3D: letters rise as open tubes",
    "topo_tower": "mech 3D: word on the floor, contour copies stack as a topographic tower", "hinge_unfold": "mech 3D: letters fold open on hinges",
    "xray_unfold": "mech 3D: 14 depth layers unfold", "candy_orbit": "mech 3D: letters orbit like candy", "depth_tunnel": "mech 3D: each letter a shaft into the page (letter as window)",
    "beat_bars": "slab: bars swing and jump on the beats", "bold_thin_echo": "slab: bold word with thin echoes", "gradient_fade": "slab: comet-like arrival",
    "scale_clash": "slab: one huge and one tiny word on one axis", "slats": "slab: speed-line bars", "solid_hollow": "slab: fill and outline alternate",
    "staircase": "slab: stepped letters", "spectrum": "stack: spectrum bars form the word", "tangent_spray": "stack: tangent spray",
    "fly_into": "stack: letters are tunnels the camera flies into (letter as window)", "outflow": "ring: outline rings stream in",
    "voxel_orbit": "voxel: strands condense to cubes", "voxel_roll": "voxel: cubes roll into place", "particles": "particles: 2000 streaks land on glyph sites",
}
IO_STYLES = ["lines", "blobs", "flicker", "wipe"]        # `particles` retired 2026-11-01 (design_defaults.SPECKLE_IO)
TRANSITIONS = ["braid", "circuit_board", "fan", "ripple", "spiral_around", "twist", "cut", "contour_morph"]       # contour_morph (ship #8) = the default word-to-word move on a shot with a subject matte (tools/fusion/contour_morph.py); design_step._finish turns it into null for the type engine
CAMERAS = ["beat_orbit", "hold_drift", "follow", "orbit_even", "spin_read_spin", "push_through", "tilt_to_plan", "graze", "ease_reversal", "pullback_reveal"]
TECHNIQUES = ["behind", "behind_sheer", "traced"]
MATTES = ["front", "window", "fill", "wipe", "glow"]
HIT_ON = ["bar", "beat"]
WORD_KEYS = ("text", "font", "module", "lines", "hit_on", "form_in", "form_out", "entry", "exit", "t0", "t1", "form_s", "exit_s", "placement", "axes", "matte",
             "screen_offset", "render", "ltx_at", "ltx_t", "material")      # keys of a word in the piece JSON; the model gives hit / hold_s / form_s / exit_s and compile_timing() derives t0 / t1 / hit_on
# (lo, hi) clamps, applied silently and logged as fixes
RANGES = {"fast_dps": (0.0, 90.0), "pitch": (-30.0, 30.0), "roll": (-90.0, 90.0), "dx": (-0.5, 0.5), "dy": (-0.5, 0.5), "energy": (0.0, 1.0), "line.px": (2.0, 8.0), "read_dps": (2.0, 10.0), "intro_deg": (0.0, 60.0), "elev": (0.0, 10.0), "bob": (0.0, 2.0),
          "form_s": (0.45, 2.0), "exit_s": (0.5, 1.5), "hold_s": (0.5, 6.0), "yaw": (-25.0, 25.0), "wf": (0.3, 1.0), "hf": (0.15, 1.0), "so": (-0.3, 0.3),
          "plate_gain": (0.4, 1.2), "plate_sat": (0.0, 1.3), "glow_lo": (0.0, 0.3), "glow_hi": (0.0, 0.6), "grain": (0.0, 0.06),
          "vignette": (0.0, 0.5), "type_opacity": (0.4, 1.0)}
MIN_HOLD_S = 1.0          # planned hit .. release below this is an error (guide says 1.6 s)
MIN_HOLD_IN = 1.2         # the model's own hold_s below this is an error


def liked_fonts() -> list[dict]:
    from mograph import font_registry as fr, fonts_liked as fl
    byfile = {v["file"]: v for v in fr.REGISTRY.values()}
    out = []
    for e in fl.LIKED:
        vary = (byfile.get(e["file"]) or {}).get("vary") or {}
        out.append(dict(file=e["file"], family=e["family"], role=e["role"], axes={k: list(v) for k, v in vary.items()}))
    return out


# ----------------------------------------------------------------------------------------------------------- inputs
_CTX: dict = {}      # in-graph pieces (tools/fusion/design_step.py, ComfyUI app graph): name -> {"orig": piece, "ana": music, "techniques": [..]}


def load_original(name: str) -> dict:
    if name in _CTX:
        return copy.deepcopy(_CTX[name]["orig"])
    return json.loads((ROOT / PIECES[name]["src"]).read_text(encoding="utf-8"))


def load_analysis(name: str) -> dict:
    if name in _CTX:
        return copy.deepcopy(_CTX[name]["ana"])
    return json.loads((ROOT / PIECES[name]["ana"]).read_text(encoding="utf-8"))


def allowed_techniques(name: str) -> list[str]:
    """compose techniques the footage of piece `name` supports (in-graph pieces: the shots exist already, so a recipe that reads a
    matte / depth the graph did not make is not allowed); the file-based pieces allow all."""
    return list((_CTX.get(name) or {}).get("techniques") or TECHNIQUES)


def dropouts(ana: dict, thr: float = 0.3, min_s: float = 0.3) -> list[tuple[float, float]]:
    """(start s, length s) of stretches where the rms envelope is under `thr` x its median for >= min_s (music nearly stops)."""
    import numpy as np
    r = np.asarray((ana.get("curves") or {}).get("rms") or [], float)
    if not len(r):
        return []
    fps = float(ana.get("fps") or FPS)
    low = r < thr * float(np.median(r))
    out, i = [], 0
    while i < len(r):
        if low[i]:
            j = i
            while j < len(r) and low[j]:
                j += 1
            if (j - i) / fps >= min_s:
                out.append((round(i / fps, 2), round((j - i) / fps, 2)))
            i = j
        else:
            i += 1
    return out


def chord_blocks(ana: dict) -> list[tuple[float, float, str]]:
    cc = [c if isinstance(c, dict) else {"t": float(c)} for c in ana.get("chord_changes") or []]     # bare seconds (t2m grid) carry no chord names
    for k, c in enumerate(cc):
        c.setdefault("from", f"harmony {k + 1}")
        c.setdefault("to", f"harmony {k + 2}")
    cut = float(ana["cut"])
    if not cc:
        return []
    blocks, t0, ch = [], 0.0, cc[0]["from"]
    for c in cc:
        blocks.append((round(t0, 2), round(float(c["t"]), 2), ch))
        t0, ch = float(c["t"]), c["to"]
    blocks.append((round(t0, 2), round(cut, 2), ch))
    return blocks


def music_text(name: str, ana: dict) -> str:
    P = PIECES[name]
    cut = float(ana["cut"])
    downs = [round(float(d), 2) for d in ana["downbeats"] if float(d) < cut - 0.2]
    lines = [f"- length {cut:g} s ({int(ana['nframes'])} frames at 24 fps), tempo {float(ana['bpm']):.1f} bpm, beat {60.0 / float(ana['bpm']):.3f} s, bar {float(ana['bar_len']):.3f} s (4/4)",
             f"- key: {P['key_text']}",
             f"- character: {P['music_text']}; mood brief: {P['mood_brief']}",
             f"- downbeats (s): {', '.join(f'{d:g}' for d in downs)}",
             "- tracked beats (s): " + ", ".join(f"{float(b):g}" for b in ana["beats"] if float(b) < cut - 0.2)[:700]]
    cb = chord_blocks(ana)
    if cb:
        lines.append("- chord blocks (start-end s: chord): " + "; ".join(f"{a:g}-{b:g}: {c}" for a, b, c in cb))
        lines.append("- chord changes at (s): " + ", ".join(f"{b:g}" for a, b, c in cb[:-1]))
    do = dropouts(ana)
    do = [d for d in do if d[0] + d[1] < cut - 0.1] or []
    lines.append("- dropouts (the music nearly stops; start s, length s): " + (", ".join(f"{a:g} ({b:g})" for a, b in do) if do else "none inside the piece"))
    lines.append(f"- the audio fades out over the last 0.5 s")
    return "\n".join(lines)


def shot_rows(orig: dict) -> list[dict]:
    from tools.fusion import plan_rules as PR
    lay = PR.layout(orig)
    rows = []
    for s in orig["shots"]:
        L = lay[s["id"]]
        desc = re.sub(r"\s+", " ", s.get("prompt", "")).strip()
        rows.append(dict(id=s["id"], start=round(L["start"] / FPS, 2), dur=round(L["frames"] / FPS, 2), frames=L["frames"],
                         matte=bool(s.get("matte")), desc=desc, first=str(s.get("first") or "")))      # first: "A" (a place) | "last:<id>" (a continuation): the shot transitions fit by it
    return rows


def words_of(orig: dict) -> list[str]:
    return [w["text"] for w in orig["type"]["words"]]


# ----------------------------------------------------------------------------------------------------------- images
def _label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def contact_sheet(mp4: Path, out_png: Path, label: str, n: int = 6, tile_w: int = 190, span: int | None = None) -> Path:
    """n frames evenly over the first `span` frames of the clip, 3 columns, with the time in the shot on each tile."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = min(total, span or total) - 1
    idx = [int(round(i * last / (n - 1))) for i in range(n)]
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((int(tile_w * 16 / 9), tile_w, 3), np.uint8)
        h = int(round(fr.shape[0] * tile_w / fr.shape[1]))
        t = cv2.resize(fr, (tile_w, h), interpolation=cv2.INTER_AREA)
        tiles.append(_label(t, f"{label} t={i / FPS:.1f}s"))
    cap.release()
    h = tiles[0].shape[0]
    rows = [np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(n // 3)]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return out_png


def matte_sheet(items: list[tuple[str, Path]], out_png: Path, tile_w: int = 150) -> Path:
    """one mid-shot matte frame per shot (white = subject), 4 columns."""
    import cv2
    import numpy as np
    tiles = []
    for sid, d in items:
        files = sorted(Path(d).glob("*.png"))
        f = cv2.imread(str(files[len(files) // 2]), cv2.IMREAD_GRAYSCALE)
        h = int(round(f.shape[0] * tile_w / f.shape[1]))
        t = cv2.cvtColor(cv2.resize(f, (tile_w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        tiles.append(_label(t, f"matte {sid}"))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return out_png


def build_images(name: str, outdir: Path) -> list[Path]:
    P = PIECES[name]
    orig = load_original(name)
    rows = shot_rows(orig)
    imgs, mattes = [], []
    for r in rows:
        clip = ROOT / P["ing"] / P["clip"].format(sid=r["id"])
        imgs.append(contact_sheet(clip, outdir / f"sheet_{name}_{r['id']}.png", f"{r['id']}", span=r["frames"]))
        mattes.append((r["id"], ROOT / P["matte_dir"].format(sid=r["id"])))
    imgs.append(matte_sheet(mattes, outdir / f"sheet_{name}_mattes.png"))
    return imgs


# ----------------------------------------------------------------------------------------------------------- the prompt
EXPRESSIVE = ("pixel", "ornamental", "decon", "blackletter")          # the characterful roles: the key words of a piece use them
ROLE_ORDER = EXPRESSIVE + ("heavy", "tall", "fine", "mono", "text")   # order of the FONTS list in the prompt
ROLE_NOTE = {"pixel": "pixel grid", "ornamental": "ornamental display", "decon": "deconstructed, broken or inline", "blackletter": "blackletter gothic",
             "heavy": "heavy expressive", "tall": "tall condensed, plain", "fine": "fine hairline", "mono": "monospace", "text": "plain text face"}
NOT_PICKED = ("LibreBarcode39Text",)                 # liked, but its glyphs are barcodes: the code never swaps a word to it (the model may still pick it)
MIN_FACES = 3                                        # a piece uses at least this many distinct liked faces (one per word when it has fewer words)


def font_note(f: dict) -> str:
    """3-6 words of character for a liked font: its role and the registry's mood tags (mograph/font_registry.py)."""
    from mograph import font_registry as fr
    reg = next((v for v in fr.REGISTRY.values() if v["file"] == f["file"]), {})
    moods = [m for m in (reg.get("mood") or []) if m not in ("calm",)][:2]
    return ", ".join([ROLE_NOTE.get(f["role"], f["role"])] + moods)


def fonts_block(seed: int | None = None, only=None) -> str:
    """the liked fonts grouped by role, the expressive roles first; inside a role shuffled by `seed` (no list-order bias; None = file order). `only` = a shortlist of font files (the piece's
    system): just those faces are listed. A face with a variable axis shows its DEFAULT instance too (Anybody: wdth 100 / wght 100 = thin)."""
    import random
    rows = []
    byrole: dict = {}
    for f in liked_fonts():
        if only is not None and f["file"] not in set(only):
            continue
        byrole.setdefault(f["role"], []).append(f)
    for role in ROLE_ORDER + tuple(r for r in byrole if r not in ROLE_ORDER):
        fs = byrole.get(role, [])
        if seed is not None:
            fs = sorted(fs, key=lambda f: f["file"])
            random.Random(f"{seed}:{role}").shuffle(fs)
        for f in fs:
            ax = ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in f["axes"].items())
            dflt = ""
            if f["file"] == PS.ANYBODY:
                d = PS.default_axes(f["file"])
                dflt = f" | default instance (axes omitted = the heavy designed look): " + ", ".join(f"{k} {v:g}" for k, v in d.items()) + " (thin, even strokes: a clean reader in plain flat white)"
            rows.append(f"- {f['file']} | {f['family']} | role {f['role']} | {font_note(f)}" + (f" | axes: {ax}" if ax else "") + dflt)
    return "\n".join(rows)


def modules_block(seed: int = 0) -> str:
    rows = [f"- {m}: {MODULE_NOTES[m]}" for m in _rot(WORD_MODULES, 3 * seed)]
    rows.append("form_in / form_out styles: " + ", ".join(_rot(IO_STYLES, seed)) + " (flicker and wipe carry no strands, the next gap becomes a cut); `entry` = the side the strands of a `lines` word come from: " + ", ".join(_rot(OP.ORDER, seed)))
    rows.append("transitions: " + ", ".join(TRANSITIONS))
    rows.append("cameras (type world): " + ", ".join(_rot(CAMERAS, seed)) + " (beat_orbit takes camera_params read_dps / intro_deg / fast_dps / elev / bob)")
    return "\n".join(rows)


def _rot(seq, k: int):
    k = int(k) % max(len(seq), 1)
    return list(seq[k:]) + list(seq[:k])


def tempo_texts(bpm=None, system: dict | None = None, seed: int = 0) -> dict:
    """the tempo-driven pieces of the schema text: the yaw band, the beat_orbit parameter bands and the form / exit times as whole beats mapped to seconds (the schema used to say
    `yaw 14-22`, `fast_dps 40-60`, `form_s 0.45..2.0`: the models copied the example numbers and the floors, diversity audit 2026-10-29)."""
    if system:
        mo = dict(yaw=system["motion"]["yaw"], bands=system["camera"]["bands"], bpm=system["motion"]["bpm"], sign=system["motion"]["start_sign"])
    elif bpm:
        m = PS.motion(bpm, seed)
        mo = dict(yaw=m["yaw"], bands=m["bands"], bpm=m["bpm"], sign=m["start_sign"])
    else:
        mo = None
    if mo is None:
        return dict(yaw="-25..25 degrees, alternating in sign from word to word; small for a slow piece, larger for a fast one", read="2..10 drift while a word is read",
                    intro="0..60 how far off face-on a word starts", fast="0..90 turn speed in the gaps", bob="0..2", form="0.45..2.0 seconds of form-in before the hit",
                    exit="0.5..1.5 seconds of exit after the hold")
    b = mo["bands"]
    bt = mo["bpm"]
    if system and (system.get("camera") or {}).get("stable"):                 # 2026-10-30: every word lands centred and holds still (stable_type.py)
        return dict(yaw="0 (every word is centred and still)", read="0 (the camera does not drift while a word is read)", intro="0", fast="0", bob="0",
                    form=f"seconds of form-in before the hit, a whole number of beats at {bt:g} bpm so that the hit lands ON a beat: {PS.beat_text(bt, 0.45, 2.0)}",
                    exit=f"seconds of exit after the hold, a whole number of beats at {bt:g} bpm: {PS.beat_text(bt, 0.5, 1.5, (1, 2, 3))}")
    return dict(yaw=f"{mo['yaw'][0]:g}..{mo['yaw'][1]:g} degrees at {bt:g} bpm, alternating in sign from word to word", read=f"{b['read_dps'][0]:g}..{b['read_dps'][1]:g} drift while a word is read",
                intro=f"{b['intro_deg'][0]:g}..{b['intro_deg'][1]:g} how far off face-on a word starts", fast=f"{b['fast_dps'][0]:g}..{b['fast_dps'][1]:g} turn speed in the gaps", bob=f"{b['bob'][0]:g}..{b['bob'][1]:g}",
                form=f"seconds of form-in before the hit, a whole number of beats at {bt:g} bpm: {PS.beat_text(bt, 0.45, 2.0)}",
                exit=f"seconds of exit after the hold, a whole number of beats at {bt:g} bpm: {PS.beat_text(bt, 0.5, 1.5, (1, 2, 3))}")


def schema_text(words: list[str], cut: float, techniques: list[str] | None = None, seed: int = 0, bpm=None, system: dict | None = None, cuts_system: dict | None = None) -> str:
    n = len(words)
    tt = tempo_texts(bpm, system, seed)
    io_enum = ", ".join(_rot(IO_STYLES, seed))                  # the enumerations are rotated per design seed: "lines" / "lanes" first in every prompt was a bias (every piece opened with strands from the top)
    mod_enum = ", ".join(_rot(WORD_MODULES, 3 * seed))
    techniques = techniques or TECHNIQUES
    return f"""Reply with ONE JSON object and nothing else (no markdown fence, no commentary). Keys:
{{
 "concept": "<one sentence: the single idea of this design>",
 "idea_why": "<why this idea for THIS music and footage, one sentence>",
 "treatment_notes": "<one sentence: where the type sits relative to the subject and how it behaves>",
 "system": {{"rule": "<the ONE rule that organises the type of the whole piece>", "roles": {{"A": "<font file>", "B": "<font file>"}}, "pattern": ["A", "B"], "yaw": [<-25..25>, <-25..25>], "axis": {{"font": "<font file>", "tag": "<axis tag>", "windows": [[<lo>, <hi>], [<lo>, <hi>]]}}, "progression": "<what changes from the first word to the last>", "why": "<why this rule for these words>"}},     // optional but strongly wanted. The pattern is cycled over the words (A, B, A, B ...), yaw and the axis windows too; a word that omits font / yaw / axes gets them from here, a word that writes its own keeps it. A system uses fonts as ROLES that recur; at least three distinct liked faces in the piece.
 "type": {{
  "mood": "<one or two words>", "energy": <0..1>,
  "palette": {{"bg": [r,g,b], "fg": [r,g,b], "accents": [[r,g,b],[r,g,b],[r,g,b]]}},     // integers 0..255
  "camera": "<one of: {', '.join(_rot(CAMERAS, seed))}>",
  "camera_params": {{"read_dps": <{tt['read']}>, "intro_deg": <{tt['intro']}>, "fast_dps": <{tt['fast']}>, "elev": <0..10>, "bob": <{tt['bob']}>}},     // only used with beat_orbit; may be omitted
  "transitions": "auto" | [<{n - 1} entries, one per gap between words, each EXACTLY one of: {', '.join(TRANSITIONS)} (or null to let the engine pick that gap; on a shot with a subject matte a null gap or `contour_morph` plays CONTOUR_MORPH, the word dissolves into the subject's contour and the contour re-forms into the next word: name another transition only when you want that one instead); no name twice except cut and contour_morph>],
  "line": {{"px": <2..8>}},          // outline line width in px
  "words": [                          // EXACTLY {n} entries, in this order, text spelled exactly as given: {', '.join(words)}
   {{"text": "<as given>", "why": "<why this font role, module, form in/out, hit and owner (render / material) for THIS word>", "font": "<a font file from the FONTS list, exact>", "module": "<EXACTLY one of: {mod_enum}>",
     "lines": ["<optional line breaks; the pieces joined without spaces must spell the word>"],
     "hit": <seconds: the downbeat or beat from the music analysis where the word finishes forming and hits>,
     "hold_s": <seconds the finished word stays still: 1.6 or more where the time allows, 1.2 is the absolute minimum>, "form_s": <{tt['form']}>, "exit_s": <{tt['exit']}>,
     "form_in": "<EXACTLY one of: {io_enum}; each word enters differently: no two words share a form_in and an entry side, and the piece uses at least two styles>", "form_out": "<EXACTLY one of: {io_enum}>",
     "entry": "<optional: the side the strands of a `lines` word come from, EXACTLY one of: {', '.join(_rot(OP.ORDER, seed))}; name it, the silent default is the top>",
     "placement": {{"yaw": <{tt['yaw']}>, "pitch": <-30..30>, "roll": <-30..30>, "dx": <-0.5..0.5>, "dy": <-0.5..0.5 frame fractions>, "wf": <0.3..1.0 box width as a fraction of the frame>, "hf": <0.15..1.0 box height fraction>}},
     "screen_offset": [<x -0.3..0.3>, <y -0.3..0.3>],      // optional: shifts the word on screen in frame fractions (y down)
     "axes": {{"<axis tag of THAT font>": <number> | [<lo>, <hi>, "beat"]}},      // optional; only the axis tags in that font's line of the FONTS list ("axes: wght 100-900"), values inside the range shown; a font with no axes list takes none
     "matte": "<optional, omit unless you want a treatment: EXACTLY one of {', '.join(MATTES)}; never write behind or none>",
     "render": "<who draws this word, EXACTLY one of: {', '.join(OW.RENDERS)} (default flat: leave it out or write flat unless the word is the one short key word whose material letters are the idea; see the guide section 12)>",
     "ltx_at": <{OW.LTX_AT_RANGE[0]:g}..{OW.LTX_AT_RANGE[1]:g}: handoff only, the fraction of the word's time where the video model's letters take over>,
     "material": "<at most {OW.MATERIAL_MAX_WORDS} words: what the letters are made of in the footage, e.g. thick carved ice, extruded, frosted edges; needed unless render is flat>",
     {DO.SCHEMA_WORD}}}
  ]
 }},
 "compose": {{
  "technique": "<EXACTLY one of: {', '.join(techniques)}>",
  "plate_gain": <0.4..1.2 brightness of the footage under the type>, "plate_sat": <0..1.3 colour of the footage>,
  "glow_lo": <0..0.3>, "glow_hi": <0..0.6>, "glow_color": [r,g,b],        // floats 0..1 (NOT 0..255); glow follows the music envelope between lo and hi
  "grain": <0..0.06>, "vignette": <0..0.5>, "type_opacity": <0.4..1.0>,
  "default": {{"why": "<why this chain for this footage>", "ops": [{{"op": "<a building block from COMPOSE BUILDING BLOCKS>", "out": "<buffer>", "args": {{...}}}}, ...]}},      // optional: the op chain every shot uses; write it yourself, the last op writes `image`
  "shots": {{"<shot id>": {{"ops": [...]}}}},                                                                                     // optional: a chain of its own for one shot
  "holds": [[<time s>, <freeze length s 0.2..1.0>], ...],      // optional freeze-frames of footage and type, at most 6, e.g. on dropouts; pairs of numbers, not objects
  {DO.schema_cuts(cuts_system or system)}
 }}
}}
{WF.APP_RULES}
Timing: you give each word the beat it hits on (`hit`), how long it holds still (`hold_s`, 1.6 or more where the time allows, never under 1.2), the form-in time before the hit and the exit time after the hold.
The engine plans the words one after the other: a word starts after the previous one has finished its exit plus a gap of at least 0.4 s, so the hit of word k
must be later than the end of word k-1 (its hit + hold_s + exit_s) plus that gap plus form_s. The first word starts at 0 s, so its hit must be within 2 s of the start.
Hits of later words may be any downbeat or beat that leaves room. Choose the hits on musical moments (a chord change, the bar a section starts, the first downbeat after a dropout) and let the holds and the form / exit times differ from word to word; the same gap and the same hold on every word reads as a metronome. The last word's exit must end at least 0.2 s before {cut:g} s. The code places the words for you; you do not give start or end times."""


def _blocks_text(cut: float) -> str:
    from tools.fusion import compose_blocks as CB
    return CB.prompt_text(cut)


def build_prompt(name: str, design_seed: int) -> str:
    P = PIECES[name]
    orig = load_original(name)
    ana = load_analysis(name)
    cut = float(ana["cut"])
    rows = shot_rows(orig)
    shots = "\n".join(f"- {r['id']}: starts {r['start']:g} s, lasts {r['dur']:g} s ({r['frames']} f); matte: {'yes' if r['matte'] else 'no'}; {r['desc']}" for r in rows)
    guide = Path(P.get("guide") or GUIDE).read_text(encoding="utf-8")
    words = words_of(orig)
    csys = PS.for_piece(orig, ana)[0]                                   # the piece's system for the CUTS menu (explicit, or drawn from the piece's seed: the same one normalize_design uses)
    win = DO.menu_text(rows, ana, system=csys)
    win_section = f"=== TEXT AS WINDOW ===\n{win}\n\n" if win else ""
    sw = shot_windows(orig, len(words))
    bpm_ = float(ana.get("bpm") or 0) or None
    psys, _explicit = PS.for_piece(orig, ana, seed=design_seed)       # the piece's system: from the concept (plan.piece_system); a piece without one (a typed request, an old plan) is designed as before: the full font list, no system section
    if not _explicit:
        psys = None
    sp_ = (orig.get("plan") or {}).get("still_palette")
    if psys and isinstance(sp_, dict):                                 # the type palette is the one extracted from the piece's own first still
        psys = PS.with_still_palette(psys, sp_)
    fmin = min(sec for _b, sec in PS.beat_menu(bpm_ or 100, 0.45, 2.0))
    xmin = min(sec for _b, sec in PS.beat_menu(bpm_ or 100, 0.5, 1.5, (1, 2, 3)))
    if len(words) * (fmin + 1.2 + xmin + 0.4) > cut:                   # a tight piece: the floors are what the engine accepts
        fmin, xmin = 0.45, 0.5
    align = ("Words and shots go together: word k belongs to shot k (same order). Its form-in starts with its shot, its `hit` is a beat INSIDE its shot, and its hold and exit end "
             f"{SHOT_GAP_S:g} s before the cut into the next shot (the code moves any hit, hold or exit that does not fit and logs it). Shots: "
             + "; ".join(f"word {k + 1} = {sid} {a:g}..{b:g} s" for k, (a, b, sid) in enumerate(sw)) + ".\n") if sw else ""
    return f"""You design the type and the compose look of one short {P.get('shape', 'vertical')} music video, "{P['title']}". Your reply is one JSON object (schema at the end).

=== DESIGN GUIDE (follow it) ===
{guide}

{win_section}
=== THE PIECE ===
Title: {P['title']}. Frame: {P.get('frame', '1080x1920 vertical (9:16)')}, 24 fps, {cut:g} s. The footage plays continuously and cuts between shots; the type is yours.
{("THE FOOTAGE IDEA (written by the concept step together with the shot prompts; the footage is already made from it, serve it): " + P["idea"] + chr(10)) if P.get("idea") else ""}Words, in order, each shown once: {' / '.join(words)}
Lyrics: {P['lyrics']}. The words above are the text the piece shows.
Time budget: {len(words)} words in {cut:g} s = {cut / len(words):.1f} s per word. One word needs at least form-in {fmin:g} + hold 1.2 (1.6 is better) + exit {xmin:g} + gap 0.4 = {fmin + 1.2 + xmin + 0.4:.2f} s, so with short words the hits
sit about {fmin + 1.2 + xmin + 0.4:.1f} s or more apart; a word that is long to read (many letters) needs a longer hold. Form-in and exit are asked in whole beats (the schema lists the seconds).
{align}Design seed: {design_seed} (use it to pick the concept, see the guide section 9).

=== MUSIC ANALYSIS ===
{music_text(name, ana)}

=== SHOTS (footage, fixed) ===
{shots}
{P.get('images_text') or "Images attached, in order: one 6-frame contact sheet per shot (same order as the list; each tile is labelled with the time inside the shot) and one matte sheet" + chr(10) + "(one frame per shot, white = the subject the type can go behind or in front of)."}

{("=== YOUR SYSTEM FOR THIS PIECE ===" + chr(10) + PS.prompt_text(psys) + chr(10)) if psys else ""}
=== WORD MODULES, STYLES, TRANSITIONS, CAMERAS (all approved{"; your system above picks from them" if psys else ""}) ===
{modules_block(design_seed)}
{(OP.prompt_text((orig.get("plan") or {}).get("earlier_openings")) + chr(10)) if OP.prompt_text((orig.get("plan") or {}).get("earlier_openings")) else ""}

=== FONTS (the only fonts you may use{": your system's shortlist" if psys else ""}; write the file name exactly; expressive roles first; use at least three different faces) ===
{fonts_block(design_seed, only=psys["fonts"] if psys else None)}

=== COMPOSE BUILDING BLOCKS (optional: write the op chain of a shot yourself; invalid ops are dropped one by one, a chain that draws no type is replaced by the rules' chain) ===
{_blocks_text(cut)}

=== OUTPUT SCHEMA ===
{schema_text(words, cut, allowed_techniques(name), design_seed, bpm_, psys, csys)}
"""


def build_package(name: str, design_seed: int, outdir: Path) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(name, design_seed)
    imgs = build_images(name, outdir)
    (outdir / f"package_{name}_seed{design_seed}.txt").write_text(prompt, encoding="utf-8")
    return dict(prompt=prompt, images=[str(p) for p in imgs])


# ----------------------------------------------------------------------------------------------------------- parse + validate
def extract_json(text: str):
    """the LARGEST balanced {...} object of a reply that parses as JSON (fences, prose, // comments tolerated; an inner object never wins over its parent)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:       # noqa: BLE001
        pass
    best, best_len = None, 0
    s = t.find("{")
    while s >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = re.sub(r"//[^\n\"]*$", "", t[s:i + 1], flags=re.M)
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict) and len(cand) > best_len:
                            best, best_len = obj, len(cand)
                    except Exception:       # noqa: BLE001
                        pass
                    break
        s = t.find("{", s + 1)
    cands = []                                                # (size, preference, object): the largest wins, a tie goes to the schema-aware mender
    for pref, obj_ in ((0, best), (1, _close_brackets(t)), (2, _mend_brackets(t))):     # `_close_brackets`: a missing / extra closer at the end (Gemma 4: `type` never closed); `_mend_brackets`: a slip in the middle
        if obj_ is not None:
            cands.append((len(json.dumps(obj_)), pref, obj_))
    return max(cands, key=lambda x: (x[0], x[1]))[2] if cands else None


_CHILD_KEYS = {"palette": {"bg", "fg", "accents", "panel"}, "camera_params": {"read_dps", "intro_deg", "fast_dps", "elev", "bob", "lead", "fill", "radius_scale"}, "line": {"px"},
               "placement": {"yaw", "pitch", "roll", "dx", "dy", "wf", "hf", "bleed", "w", "h", "z"}, "system": {"rule", "roles", "pattern", "yaw", "axis", "progression", "why"}}


def _mend_brackets(t: str):
    """a long reply with a bracket slip in the MIDDLE (a `]` typed for a `}`, an extra `]`, a missing `}`): walk the text with a stack of expected closers and, at a
    closer that does not match the top, try each policy for the whole reply in turn: `replace` (the closer is a typo for the expected one), `drop` (it is a stray),
    `insert` (the closers above it are missing). The policy whose text parses to the largest object wins; what is still open at the end is closed. None when no policy does."""
    s = t.find("{")
    if s < 0:
        return None
    body = re.sub(r"//[^\n\"]*$", "", t[s:], flags=re.M)
    found = []
    for policy in ("replace", "drop", "insert"):
        out, stack, names, in_str, esc, last_key = [], [], [], False, False, ""
        for pos, c in enumerate(body):
            if in_str:
                out.append(c)
                if esc:
                    esc = False
                elif c == chr(92):
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                km = re.match(r'"([^"]{1,80})"\s*:', body[pos:pos + 120])
                if km:
                    closers = []
                    if stack and stack[-1] == "]" and "}" in stack:
                        while stack and stack[-1] != "}":          # a `"key":` inside an array: the arrays above it were never closed
                            closers.append(stack.pop())
                            names.pop()
                    while stack and stack[-1] == "}" and names[-1] in _CHILD_KEYS and km.group(1) not in _CHILD_KEYS[names[-1]]:
                        closers.append(stack.pop())                  # a key that is not a child of `palette` / `camera_params` ...: that object was never closed
                        names.pop()
                    if closers:
                        while out and (out[-1].isspace() or out[-1] == ","):        # the comma that was meant for the next key goes after the closers
                            out.pop()
                        out.extend(closers)
                        out.append(",")
                    last_key = km.group(1)
                in_str = True
                out.append(c)
            elif c in "{[":
                stack.append("}" if c == "{" else "]")
                names.append(last_key if c == "{" else "")
                out.append(c)
            elif c in "}]":
                if stack and stack[-1] == c:
                    stack.pop()
                    names.pop()
                    out.append(c)
                elif policy == "replace" and stack:
                    out.append(stack.pop())
                    names.pop()
                elif policy == "insert" and c in stack:
                    while stack and stack[-1] != c:
                        out.append(stack.pop())
                        names.pop()
                    stack.pop()
                    names.pop()
                    out.append(c)
                if not stack and out:
                    break
            else:
                out.append(c)
        text = re.sub(r",\s*$", "", "".join(out).rstrip()) + "".join(reversed(stack))
        try:
            obj = json.loads(text)
        except Exception:       # noqa: BLE001
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return max(found, key=lambda o: len(json.dumps(o))) if found else None


def _close_brackets(t: str):
    """a reply object whose brackets do not balance (a missing closer at the end, `compose` written inside an unclosed `type`): the text from the first `{`
    with the missing `}` / `]` appended in stack order (and a trailing comma removed). None when that does not parse to an object either."""
    s = t.find("{")
    if s < 0:
        return None
    body = re.sub(r"//[^\n\"]*$", "", t[s:], flags=re.M)
    stack, in_str, esc, end = [], False, False, len(body)
    for i, c in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack[-1] != c:
                return None
            stack.pop()
            if not stack:
                end = i + 1
                break
    if not stack:
        return None
    text = re.sub(r",\s*$", "", body[:end].rstrip()) + "".join(reversed(stack))
    try:
        obj = json.loads(text)
    except Exception:       # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _clamp(v, key, fixes, where):
    lo, hi = RANGES[key]
    try:
        x = float(v)
    except (TypeError, ValueError):
        fixes.append(f"{where}: {v!r} is not a number")
        return None
    y = min(max(x, lo), hi)
    if abs(y - x) > 1e-9:
        fixes.append(f"{where}: {x:g} clamped to {y:g} ({lo:g}..{hi:g})")
    return y


def _rgb(v, scale, where, fixes, errors):
    try:
        a = [float(x) for x in v]
        assert len(a) == 3
    except Exception:       # noqa: BLE001
        errors.append(f"{where}: needs [r,g,b]")
        return None
    out = [min(max(x, 0.0), scale) for x in a]
    if out != a:
        fixes.append(f"{where}: clamped to 0..{scale:g}")
    return [int(round(x)) for x in out] if scale > 1 else [round(x, 3) for x in out]


def _carries(prev: dict, nxt: dict) -> bool:
    """does the boundary between two words carry material (a transition whose length follows the stated gap)? A flicker / wipe side carries nothing: the engine
    puts a fixed half-beat cut there, so the next word's start cannot be moved by the gap and the form time absorbs the difference."""
    from tools.fusion import spec as SP
    return SP._style_name(prev.get("form_out")) not in SP.NO_MATERIAL and SP._style_name(nxt.get("form_in")) not in SP.NO_MATERIAL


def _compile(t: dict, ana: dict, fixes: list, dh: float = 0.0, beat_only: bool = False, last_only: bool = False):
    """one pass of the timing solver with every hold shortened by `dh` (floor MIN_HOLD_IN). Returns (done words with t0 / t1 / form_s / hit_on, end of the last word)."""
    from tools.fusion import plan_rules as PR
    from tools.fusion import spec as SP
    probe = {"type": copy.deepcopy(t)}
    PR.fill_type_timing(probe, ana)
    spec = probe["type"]
    grid = PR.make_grid(ana, float(ana["cut"]))
    downs = [float(d) for d in ana["downbeats"]]
    beats = [float(b) for b in ana["beats"]]
    done, end = [], 0.0
    for i, w in enumerate(t["words"]):
        D = float(w["hit"])
        if w.get("_pin_rel") is not None:                   # a planned window_other word: it hits no earlier than PIN_HOLD_S before its pinned release (the fit moves hits earlier, a 6 s hold would follow)
            D = max(D, float(w["_pin_rel"]) - PIN_HOLD_S)
        nd = min(downs, key=lambda d: abs(d - D)) if downs else D
        if abs(nd - D) <= 0.06 and not beat_only:
            D, hit_on = nd, "bar"
        else:
            nb = min(beats, key=lambda b: abs(b - D)) if beats else D
            if abs(nb - D) > 0.06 and not beat_only:
                fixes.append(f"type.words[{i}].hit {D:g} moved to the beat {nb:g}")
            D, hit_on = nb, "beat"
        f, x = float(w["form_s"]), float(w["exit_s"])
        h = max(MIN_HOLD_IN, float(w["hold_s"]) - (dh if (not last_only or i == len(t["words"]) - 1) else 0.0))     # last_only: only the last word gives time
        base = {k: v for k, v in w.items() if k not in ("hit", "hold_s")}
        base["hit_on"] = hit_on
        prev_t1 = done[-1]["t1"] if done else 0.0
        g = max(0.4, D - f - prev_t1) if done else 0.0
        got, cand = None, None
        for _ in range(10):
            t0 = 0.0 if not done else prev_t1 + g
            cand = dict(base, form_s=round(f, 3), t0=round(t0, 3), t1=round(t0 + f + h + x, 3))
            got = SP.plan_timing(done + [cand], spec, grid)[-1]
            err = D - got["hit"]
            if abs(err) < 0.02:
                break
            if done and g + err >= 0.36 and _carries(done[-1], base):
                g += err
            else:                                            # the start cannot move: change the form time instead
                f2 = min(max(f + err, 0.45), 2.0)
                if abs(f2 - f) < 1e-6:
                    break
                f = f2
        if abs(D - got["hit"]) >= 0.02 and dh == 0.0:
            fixes.append(f"type.words[{i}]: the hit at {D:g} s is not reachable (the engine puts it at {got['hit']:.2f} s: the previous word ends too late, or the first word starts at 0)")
        pin = base.get("_pin_rel")                          # a planned window_other word: its release stays where the cut needs it (design_options.plan_window_other)
        if pin is not None:
            h2 = min(float(pin) - got["hit"], 6.0)
            if h2 > h + 1e-6:
                h = h2
                cand = dict(cand, t1=round(cand["t0"] + f + h + x, 3))
                got = SP.plan_timing(done + [cand], spec, grid)[-1]
        done.append(dict(base, form_s=round(f, 3), t0=cand["t0"], t1=cand["t1"], _h=h, _D=D))
        end = got["rel"]                                    # end of the last HOLD: the last exit may be cut by the end of the music (as in the original pieces)
    return done, end


def cut_pre_windows(c_in, shots) -> dict:
    """{word index: seconds}: how long before the end of its shot a word must be GONE because a 3D cut follows (round-3 QC 2026-10-29: the "CAETCH" doubling was the word's own exit animation
    running inside the cut's transition window). The cut's window A is `i2` frames before the boundary (pivot_cut.window_frames); a word either finishes its exit before it or is a window
    word. shots = shot_windows [(start, end, id)]; only 3D kinds count."""
    out: dict = {}
    if not shots or not isinstance(c_in, dict):
        return out
    try:
        from tools.fusion import pivot_cut as PC
    except Exception:       # noqa: BLE001
        return out
    ids = [sid for _a, _b, sid in shots]
    for c in c_in.get("cuts") or []:
        if not isinstance(c, dict) or c.get("after") not in ids:
            continue
        kind = PC.normalize(c.get("cut"))
        k = ids.index(c["after"])
        if kind is None or k + 1 >= len(ids):
            continue
        try:
            n, i2 = PC.window_frames(kind, c.get("params") if isinstance(c.get("params"), dict) else None)
        except Exception:       # noqa: BLE001
            continue
        out[k] = round(i2 / float(FPS), 3)
    return out


def compile_timing(t: dict, ana: dict, fixes: list, shots: list | None = None, free: set = frozenset(), pre: dict | None = None) -> list[str]:
    """the model gives, per word, the beat it HITS on (`hit`), `hold_s`, `form_s`, `exit_s`. The type engine plans words in a chain
    (spec.plan_timing: a word starts where the previous one ends plus a gap; t0 of the next word minus t1 of this word is that gap, the span
    t1 - t0 = form + hold + exit; hits snap to the grid). This solves t0 / t1 / hit_on so every hit lands where the model asked, using the
    engine's own plan_timing as the simulator. When the chain overruns the music (the last word must end by the cut) every hold is
    shortened together, 0.1 s at a time, down to MIN_HOLD_IN, and logged as a fix; if it still does not fit it is an error.
    shots = [(start, end, sid), ...] (one per word, `shot_windows`): word k lives inside shot k, `fit_to_shots` first moves its hit / hold / form / exit
    inside the shot (each change a fix) so the chain below starts every word at its shot start and ends it SHOT_GAP_S before the cut.
    Mutates t["words"] (adds t0, t1, hit_on; drops hit / hold_s). -> errors."""
    limit = float(ana["cut"]) - 0.2            # the last hold must finish before the end of the music; its exit may be cut (as in the original pieces)
    errors = []
    if shots and len(shots) == len(t["words"]):
        fit_to_shots(t, ana, shots, fixes, free, pre)
    fx0: list = []
    done, end = _compile(t, ana, fx0, 0.0)
    dh, last_dh = 0.0, 0.0
    if end > limit + 1e-9:                                  # first: only the LAST word's hold gives time (the fewest fields changed)
        d2 = 0.0
        while d2 < 6.0:                                     # hold_s goes up to 6 s: the last hold may give up to 4.8 s
            d2 = round(d2 + 0.1, 3)
            d_l, e_l = _compile(t, ana, [], d2, last_only=True)
            if e_l <= limit + 1e-9:
                done, end, last_dh = d_l, e_l, d2
                break
            if d_l[-1]["_h"] <= MIN_HOLD_IN + 1e-9:
                break
    while end > limit + 1e-9 and dh < 3.0:
        dh = round(dh + 0.1, 3)
        done, end = _compile(t, ana, [], dh)
        if all(d["_h"] <= MIN_HOLD_IN + 1e-9 for d in done):
            break
    fixes.extend(fx0 if dh == 0.0 else [f for f in fx0 if "not reachable" not in f])
    if end > limit + 1e-9:
        # NOT an error any more (batch 3060 v005 / v007 single-shot pieces: the model's hits were fine for the music but late for 5 words, the whole type section was
        # thrown away): keep the model's timing as compiled, `fit_scene` (stages B and C: tighter form / exit, hits moved to the earliest beats) fits it to the music
        # and logs every change; only if that fails too is it an error (fit_scene returns it).
        fixes.append(f"the {len(done)} words do not fit in {float(ana['cut']):g} s even with every hold at {MIN_HOLD_IN:g} s (the last hold would end at {end:.2f} s): "
                     f"the timing fit moves the hits earlier and tightens form / exit")
        dh = 0.0
        done, end = _compile(t, ana, [], 0.0)
    elif last_dh > 0.0:
        fixes.append(f"type.words[{len(done) - 1}].hold_s shortened by {last_dh:g} s (floor {MIN_HOLD_IN:g} s) so the {len(done)} words end before the music fades; the other holds are untouched")
    elif dh > 0.0:
        fixes.append(f"all holds shortened by {dh:g} s (hold_s floor {MIN_HOLD_IN:g} s) so the {len(done)} words end before the music fades")
    for w, d in zip(t["words"], done):
        w["t0"], w["t1"], w["form_s"], w["hit_on"] = d["t0"], d["t1"], d["form_s"], d["hit_on"]
        w["_hold_in"] = float(w["hold_s"])
    for w, d in zip(t["words"], done):
        w["_D"], w["_h0"] = round(float(d["_D"]), 3), round(float(w["_hold_in"]), 3)
        w.pop("_hold_in", None)
    for w in t["words"]:
        w.pop("hit", None)
        w.pop("hold_s", None)
    return errors


def _morph_names_to_null(probe: dict) -> dict:
    """the engine probes below run the type engine on a COPY of the design: `contour_morph` is not a type-engine module (KeyError: unknown module), so the probe sees what
    design_step._finish gives the engine (null gaps). The design itself keeps the name: _finish records plan.contour_morph_asked from it. (design_step is imported lazily: it imports this module.)"""
    from tools.fusion import design_step as DS
    DS._morph_names_to_null(probe)
    return probe


def fit_scene(design: dict, name: str, fixes: list) -> list[str]:
    """DETERMINISTIC TIMING FIT (code, not the model; the planner's own tail rule: the last hold must end before the music does). The real timeline
    (spec.build_scene through plan_rules.dry_run) is the judge, because words joined by cuts or explicit transitions run later than plan_timing
    predicts. If the last hold ends after cut - 0.2 s, in this order, each step re-planned and re-checked on the real scene:
      A. every hold shortened together, 0.1 s at a time, down to MIN_HOLD_IN (the model's hits stay where the engine can put them);
      B. form_s -> 0.45 and exit_s -> 0.5 on every word (the engine's floors);
      C. hits moved EARLIER: every word hits at the earliest beat after the previous word (hit_on beat), holds at the floor.
    Each change is logged in `fixes`. Needs the `_D` / `_h0` keys compile_timing leaves on the words. -> errors."""
    from tools.fusion import plan_rules as PR
    ana, orig = load_analysis(name), load_original(name)
    limit = float(ana["cut"]) - 0.2
    t = design["type"]
    words = t["words"]
    if not words or "_D" not in words[0]:
        return []
    src = [dict(w) for w in words]

    def scene_end():
        probe = copy.deepcopy(orig)
        probe["type"] = dict(copy.deepcopy(t), seed=orig["type"].get("seed", 1))
        PR.fill_type_timing(_morph_names_to_null(probe), ana)
        chk = PR.dry_run(probe)
        return float(chk["words"][-1]["release"]) if chk.get("words") else 0.0

    def apply(dh, tight_form_exit, earliest, scale=None, last_only=False):
        tt = {"words": [], **{k: v for k, v in t.items() if k != "words"}}
        for w in src:
            x = {k: v for k, v in w.items() if k not in ("t0", "t1", "_D", "_h0")}
            x["hit"] = 0.0 if earliest else w["_D"]
            x["hold_s"] = w["_h0"] if scale is None else MIN_HOLD_IN + scale * max(w["_h0"] - MIN_HOLD_IN, 0.0)
            if last_only and w is src[-1]:                  # A0: only the last word gives time (the fewest fields changed)
                x["hold_s"] = max(MIN_HOLD_IN, w["_h0"] - dh)
            if tight_form_exit:
                x["form_s"], x["exit_s"] = 0.45, 0.5
            tt["words"].append(x)
        done, _ = _compile(tt, ana, [], 0.0 if last_only else dh, beat_only=earliest)
        for w, d, x in zip(words, done, tt["words"]):
            w["t0"], w["t1"], w["form_s"], w["hit_on"], w["exit_s"] = d["t0"], d["t1"], d["form_s"], d["hit_on"], x["exit_s"]

    try:
        end0 = scene_end()
    except Exception:       # noqa: BLE001  the engine's own error is reported by timing_errors
        return []
    if end0 <= limit:
        return []
    h_min_steps = max(int(round((max(w["_h0"] for w in src) - MIN_HOLD_IN) * 10)), 0)
    last_steps = max(int(round((src[-1]["_h0"] - MIN_HOLD_IN) * 10)), 0)

    def smallest(n, last_only):
        """the smallest shortening step k in 1..n (x 0.1 s) that brings the REAL scene in time, by bisection (the end only moves earlier as the holds shrink: ~6
        dry runs instead of up to 48); None when even n steps do not."""
        def ok(k):
            apply(k / 10.0, False, False, None, last_only)
            try:
                return scene_end() <= limit
            except Exception:       # noqa: BLE001
                return False
        if n < 1 or not ok(n):
            return None
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    for label, n, last_only in (("A0: only the last word's hold shortened", last_steps, True), ("A: every hold shortened", h_min_steps, False)):
        k = smallest(n, last_only)
        if k:
            apply(k / 10.0, False, False, None, last_only)
            end = scene_end()
            hits = ", ".join(f"{w['text']} {h:.2f}" for w, h in zip(words, _scene_hits(t, ana, orig)))
            fixes.append(f"TIMING FIT (code): the model's timeline put the last hold at {end0:.2f} s of {float(ana['cut']):g} s; stage {label} by {k / 10.0:g} s "
                         f"brings it to {end:.2f} s. Planned hits now: {hits}")
            return []
    stages = []
    stages += [("B: form 0.45 s / exit 0.5 s on every word, holds shortened", dh / 10.0, True, False) for dh in range(0, h_min_steps + 1, max(h_min_steps, 1))]
    stages += [(f"C: every word hits at the earliest beat after the previous one, form 0.45 s / exit 0.5 s, holds scaled to {k * 10} % of the way from {MIN_HOLD_IN:g} s to the model's hold",
                0.0, True, True, k / 10.0) for k in range(10, -1, -1)]
    for label, dh, tf, early, *sc in stages:
        apply(dh, tf, early, sc[0] if sc else None, bool(len(sc) > 1 and sc[1]))
        try:
            end = scene_end()
        except Exception:       # noqa: BLE001
            continue
        if end <= limit:
            hits = ", ".join(f"{w['text']} {h:.2f}" for w, h in zip(words, _scene_hits(t, ana, orig)))
            fixes.append(f"TIMING FIT (code): the model's timeline put the last hold at {end0:.2f} s of {float(ana['cut']):g} s; stage {label}"
                         f"{f' by {dh:g} s' if label.startswith('A') else ''} brings it to {end:.2f} s. Planned hits now: {hits}")
            return []
    return [f"the {len(words)} words cannot be fitted before the music ends even with minimum holds, forms and exits and the earliest hits"]


def _scene_hits(t, ana, orig):
    from tools.fusion import plan_rules as PR
    probe = copy.deepcopy(orig)
    probe["type"] = dict(copy.deepcopy(t), seed=orig["type"].get("seed", 1))
    PR.fill_type_timing(_morph_names_to_null(probe), ana)
    return [float(w["hit"]) for w in PR.dry_run(probe)["words"]]


def _scale_of(w: dict) -> float:
    return float((w.get("placement") or {}).get("wf", 1.0))


def window_words(design: dict) -> set:
    """indices of the words that are windows (window_source on the word, or named by a window cut): their face is a fat face on purpose."""
    t = (design.get("type") or {}).get("words") or []
    idx = {i for i, w in enumerate(t) if w.get("window_source") not in (None, "", "self") or w.get("window_ground")}
    for c in ((design.get("compose") or {}).get("cuts") or []):
        if c.get("cut") in DO.WINDOW_KINDS and isinstance(c.get("word"), int):
            idx.add(c["word"])
    return idx


def diversify_fonts(t: dict, fixes: list, seed, protected=(), only=None) -> None:
    """code fixes on the model's fonts (all picks from mograph/fonts/liked.json, seeded, logged in `fixes`; the model's own choices are kept
    wherever they already meet the rules):
      a. no word uses an expressive role (pixel / ornamental / decon / blackletter): the hero word (longest hold, else the biggest box) gets one;
      b. at least MIN_FACES distinct faces (one per word when fewer words) from at least 2 roles: duplicates of non-hero words first are swapped to
         roles not used yet, keeping the contrast with the neighbouring lines;
      c. consecutive words with the same face and a similar scale: the second (the first when the second is the hero) gets a contrasting role.
    No upper limit on the number of faces. `only` = the piece system's shortlist (font files): the swaps are drawn from it."""
    import random
    from mograph import fonts_liked as fl
    W = t.get("words") or []
    n = len(W)
    if not n or any(not w.get("font") for w in W):
        return
    protected = set(protected)
    free = [i for i in range(n) if i not in protected] or list(range(n))
    hero = max(free, key=lambda i: (float(W[i].get("hold_s", W[i].get("_h0", 0.0))), _scale_of(W[i]) * float((W[i].get("placement") or {}).get("hf", 1.0)), -i))
    pool = [e for e in fl.LIKED if e["key"] not in NOT_PICKED and (not only or e["file"] in set(only))]

    def faces():
        return [w["font"] for w in W]

    def roles():
        return [fl.role_of(w["font"]) for w in W]

    def pick(i, allowed, tag):
        ex = set(faces()) - {W[i]["font"]}
        cand = [e for e in pool if e["role"] in allowed and e["file"] not in ex and e["file"] != W[i]["font"]]
        nb = [fl.role_of(W[j]["font"]) for j in (i - 1, i + 1) if 0 <= j < n]
        good = [e for e in cand if all(fl.contrast_axes(e["role"], r) for r in nb)]
        cand = sorted(good or cand, key=lambda e: e["key"])
        return random.Random(f"{seed}:{tag}:{i}").choice(cand) if cand else None

    def swap(i, e, why):
        old = W[i]["font"]
        fixes.append(f"type.words[{i}].font {old} swapped to {e['file']} ({e['role']}): {why} (code fix)")
        W[i]["font"] = e["file"]
        if W[i].pop("axes", None) is not None:
            fixes.append(f"type.words[{i}].axes dropped (they belonged to {old})")

    if not any(r in EXPRESSIVE for r in roles()) and hero not in protected:
        e = pick(hero, EXPRESSIVE, "hero")
        if e:
            swap(hero, e, "no word used a pixel / ornamental / decon / blackletter face; the key word gets one")
    target = min(MIN_FACES, n)
    for step in range(2 * n + 2):
        fs, rs = faces(), roles()
        few_faces, few_roles = len(set(fs)) < target, len(set(rs)) < min(2, n)
        if not (few_faces or few_roles):
            break
        others = [i for i in range(n) if i != hero and i not in protected] or [i for i in range(n) if i != hero]
        dups = [i for i in reversed(others) if fs.count(fs[i]) > 1]
        i = (dups or list(reversed(others)) or [hero])[0]
        unused = [r for r in ROLE_ORDER if r not in set(rs)] or list(ROLE_ORDER)
        e = pick(i, unused, "mix")
        if not e:
            break
        swap(i, e, f"the piece needs at least {target} different faces from at least 2 roles")
    for i in range(1, n):
        if W[i]["font"] == W[i - 1]["font"] and abs(_scale_of(W[i]) - _scale_of(W[i - 1])) < 0.15:
            j = i if (i != hero and i not in protected) else i - 1
            if j in protected:
                continue
            cur = fl.role_of(W[j]["font"])
            e = pick(j, [r for r in ROLE_ORDER if r != cur], "run")
            if e:
                swap(j, e, "the same face twice in a row at a similar scale")


def scene_text(orig: dict) -> str:
    """the scene the footage shows, for the rules' default material: the brief's scene, else the first shot prompt without any letter clause."""
    sc = str((orig.get("brief") or {}).get("scene") or "")
    if sc:
        return sc
    return OW.strip_clause(str(((orig.get("shots") or [{}])[0]).get("prompt") or ""))


def owner_fields(w: dict, o: dict, wh: str, fixes: list, t: dict, orig: dict) -> None:
    """the owner fields of word `w` (model reply) -> `o` (clean word): render (invalid -> the nearest allowed name, else handoff), ltx_at (handoff only, clamped
    0.2..0.8), material (<= 8 words; missing on a word that is not flat -> the rules' default from the mood / palette / scene). Never an error: a bad field is
    mended alone and logged, the valid ones stay."""
    rd = w.get("render")
    r, how = OW.norm_render(rd, OW.DEFAULT_RENDER)
    if rd is None:
        pass                                                  # not mentioned: the default (handoff), silently
    elif how is None:
        fixes.append(f"{wh}.render {rd!r}: not one of {', '.join(OW.RENDERS)}; {r} used (the rules' default)")
    elif how == "fix":
        fixes.append(f"{wh}.render {rd!r} -> {r!r} (nearest allowed)")
    o["render"] = r
    if r == "handoff":
        la = w.get("ltx_at")
        v, how = OW.clamp_ltx_at(la if la is not None else OW.DEFAULT_LTX_AT)
        if la is None:
            pass
        elif v is None:
            v = OW.DEFAULT_LTX_AT
            fixes.append(f"{wh}.ltx_at {la!r}: not a number; {v:g} used")
        elif how == "fix":
            fixes.append(f"{wh}.ltx_at {la!r} clamped to {v:g} ({OW.LTX_AT_RANGE[0]:g}..{OW.LTX_AT_RANGE[1]:g})")
        o["ltx_at"] = v
    if r != "flat":
        m = w.get("material")
        v, how = OW.clean_material(m)
        if m is None or v is None:
            v = OW.default_material(t.get("mood"), t.get("palette"), scene_text(orig))
            if m is not None:
                fixes.append(f"{wh}.material {m!r} unusable: {v!r} (the rules' default from the mood / palette / scene)")
        elif how == "fix":
            fixes.append(f"{wh}.material {m!r} -> {v!r} (at most {OW.MATERIAL_MAX_WORDS} words, no 'letters')")
        o["material"] = v


def np_median_diff(v) -> float:
    d = sorted(b - a for a, b in zip(v, v[1:]))
    return d[len(d) // 2] if d else 0.5


def shot_windows(orig: dict, n_words: int):
    """[(start s, end s, shot id)] when the piece has exactly as many shots as words (> 1): word k belongs to shot k. None otherwise (a single shot, or an old
    piece whose lockups do not match its shots): the timing solver then works as it always did."""
    shots = orig.get("shots") or []
    if len(shots) != n_words or n_words < 2:
        return None
    from tools.fusion import plan_rules as PR
    lay = PR.layout(orig)
    out = []
    for sh in shots:
        L = lay[sh["id"]]
        n = int(sh.get("span") or L["frames"])
        out.append((L["start"] / FPS, (L["start"] + n) / FPS, sh["id"]))
    return out


PIN_HOLD_S = 3.6          # a pinned (planned window_other) word holds at most this long to reach its release (design_options.plan_window_other)
SHOT_GAP_S = 0.35         # a carrying boundary needs this much between a word's end and the next word's start (the engine's smallest stated gap); a flicker / wipe boundary is a cut and needs none


def window_free(t_in_words, c_in) -> set:
    """indices of the words that are text-as-window words (window_source, or named by a window_cut / window_swap): they must span the cut on purpose, so the shot
    fit leaves their hold and exit alone (their hit still lands inside their shot)."""
    free = set()
    for i, w in enumerate(t_in_words or []):
        if isinstance(w, dict) and w.get("window_source") not in (None, "", "self"):
            free.add(i)
    for c in (c_in.get("cuts") if isinstance(c_in, dict) else None) or []:
        if isinstance(c, dict) and str(c.get("cut", "")).strip().lower() in ("window_cut", "window_swap", "window", "swap", "window_other"):
            try:
                free.add(int(c.get("word", 0)))
            except (TypeError, ValueError):
                free.add(0)
    return free


def hold_floor_room(w: dict) -> float:
    """the least a word needs between its hit and the end of its time in the shot: the hold floor and the exit."""
    return MIN_HOLD_IN + float(w.get("exit_s") or 0.5)


def fit_to_shots(t: dict, ana: dict, shots: list, fixes: list, free: set = frozenset(), pre: dict | None = None) -> None:
    """word k lives inside shot k: its hit is a beat inside the shot (after the cut), and hit + hold + exit end SHOT_GAP_S before the next shot starts (0 for a
    flicker / wipe boundary: that is a cut; the last word: the hold ends before the cut - 0.2 s, its exit may be cut). form_s is kept unless the word would start more than 0.6 s after its shot start, then it becomes hit - shot start (clamped 0.45..2.0). The model's hit / hold_s / exit_s are kept when they
    already fit; otherwise the nearest values that fit replace them (hit -> the nearest beat inside the window, hold_s shortened to the floor MIN_HOLD_IN, exit_s
    to 0.5), each change logged. `free` = window words: only their hit is fitted. Mutates t["words"] (hit, hold_s, form_s, exit_s)."""
    beats = sorted(float(b) for b in ana["beats"])
    downs = [float(d) for d in ana["downbeats"]]
    cut = float(ana["cut"])
    n = len(t["words"])
    iv = float(np_median_diff(beats)) if len(beats) > 1 else 0.5
    for i, (w, (a, b, sid)) in enumerate(zip(t["words"], shots)):
        wh = f"type.words[{i}]"
        last = i == n - 1
        gap = 0.0 if last or not _carries(w, t["words"][i + 1]) else min(SHOT_GAP_S, 0.5 * iv)        # a carrying boundary snaps to at least half a beat
        end_lim = cut - 0.2 if (last or i in free) else b - gap + min(0.2, 0.5 * iv)        # the engine snaps an end to a beat: up to half a beat of slack
        pw = float((pre or {}).get(i, 0.0)) if not (last or i in free) and _carries(w, t["words"][i + 1]) else 0.0      # (a flicker / wipe boundary is a cut whose next word starts at this word's end: ending it early would pull the next word's form-in into the old shot)
        if pw and end_lim - pw - hold_floor_room(w) >= a + 0.2:                               # a 3D cut follows: the word's exit is over before the cut's transition window opens
            end_lim -= pw                                                                     # (a hold or hit moved for it is logged below like any other fit)
        x = float(w["exit_s"])
        hit0, hold0, f0 = float(w["hit"]), float(w["hold_s"]), float(w["form_s"])
        lo = a + max(0.2, f0 - 0.3)                        # the word's form-in starts no earlier than 0.3 s before its own shot (with the previous word gone early for a 3D cut it could start inside the cut)
        hi = end_lim - MIN_HOLD_IN - (0.0 if (last or i in free) else 0.5) + 0.05         # the last exit may run past the cut; +0.05 s: the rules' own plan sits on this edge
        hi = max(min(hi, a + RANGES["form_s"][1]), lo)
        hit = min(max(hit0, lo), hi)
        if abs(hit - hit0) > 1e-6:
            cand = [bb for bb in beats if lo - 1e-6 <= bb <= hi + 1e-6]
            if cand:
                pref = [d for d in downs if lo - 1e-6 <= d <= hi + 1e-6]
                hit = min(pref or cand, key=lambda c: abs(c - hit0))
            hit = round(hit, 3)
            fixes.append(f"{wh}.hit {hit0:g} moved to {hit:g}: word {i + 1} lives in shot {sid} ({a:g}..{b:g} s), its hit must land inside it")
        room = end_lim - hit
        hold = min(hold0, room - (0.0 if (last or i in free) else x))
        if hold < MIN_HOLD_IN:
            x2 = x if (last or i in free) else max(0.5, min(x, room - MIN_HOLD_IN))
            if x2 < x - 1e-6:
                fixes.append(f"{wh}.exit_s {x:g} shortened to {x2:g} to keep the hold inside shot {sid}")
                x = x2
            hold = max(MIN_HOLD_IN, min(hold0, room - (0.0 if (last or i in free) else x)))
        if hold < hold0 - 0.05:
            fixes.append(f"{wh}.hold_s {hold0:g} shortened to {hold:g}: word {i + 1} must end before the " +
                         (f"cut into shot {i + 2}" if not last else "music does"))
        f = f0
        if hit - f0 > a + 0.6:                          # a late start would leave the cut without type; an early one is absorbed by the engine's chain
            f = min(max(hit - a, 0.45), RANGES["form_s"][1])
            fixes.append(f"{wh}.form_s {f0:g} -> {f:g}: the word starts forming with its shot ({a:g} s)")
        w["hit"], w["hold_s"], w["form_s"], w["exit_s"] = hit, round(max(hold, MIN_HOLD_IN), 3), round(f, 3), round(x, 3)


def pad_accents(pal: dict, n: int = 3) -> dict:
    """palette.accents padded to `n` colours (the 3D word modules read accents[2]): the fg, then the midpoint of bg and fg, then the accents again. In place."""
    acc = [list(a) for a in pal.get("accents") or []]
    fg, bg = list(pal.get("fg") or [235, 235, 235]), list(pal.get("bg") or [8, 8, 8])
    spare = [fg, [(int(x) + int(y)) // 2 for x, y in zip(bg, fg)]] + [list(a) for a in acc]
    k = 0
    while len(acc) < n:
        acc.append(spare[k % len(spare)])
        k += 1
    pal["accents"] = acc
    return pal


def normalize_design(obj, name: str, diversify: bool = True, plan_windows: bool = False, pins: dict | None = None):      # `diversify` is kept for old callers: the font rules now run in finish_design()
    """model reply object -> (clean design dict, fixes [str], errors [str]). Errors need a repair turn; fixes are silent clamps.
    plan_windows: the planning rule of design_options.plan_window_other runs on the reply first (a place change where the design chose nothing heavy gets a word whose hold
    spans the cut and the cut `window_other`; the app's design step turns it on, the offline callers that hand in a stub reply leave it off)."""
    from mograph import fonts_liked as fl
    fixes, errors = [], []
    orig = load_original(name)
    ana = load_analysis(name)
    cut = float(ana["cut"])
    want = words_of(orig)
    if not isinstance(obj, dict):
        return None, fixes, ["the reply is not a JSON object"]
    csys = PS.for_piece(orig, ana)[0]                                            # the piece's system (explicit, or drawn from its seed): its DRAWN cut kinds are the only ones used
    if plan_windows and len(orig.get("shots") or []) > 1:
        obj, wnotes, pins = DO.plan_cuts(obj, shot_rows(orig), ana, windows=plan_windows != "cuts_only", system=csys)          # a drawn window cut (window_other / window_zoom ..) + a drawn 3D cut at every other place change
        fixes.extend(f"compose.cuts: {n}" for n in wnotes)
    t_in, c_in = obj.get("type"), obj.get("compose")
    if not isinstance(t_in, dict):
        return None, fixes, ["missing object `type`"]
    psys = (orig.get("plan") or {}).get("piece_system")
    psys = psys if isinstance(psys, dict) and psys.get("module") else None            # the concept's piece system: a choice inside it stays, an outlier is folded into it (piece_system.fold_*)
    still_pal = (orig.get("plan") or {}).get("still_palette")                        # the palette extracted from the piece's own first still (design_step.package): type colours snap to it
    if psys and isinstance(still_pal, dict):
        psys = PS.with_still_palette(psys, still_pal)
    elif isinstance(still_pal, dict) and still_pal.get("colors"):
        t_in = copy.deepcopy(t_in)
        if not isinstance(t_in.get("palette"), dict):
            t_in["palette"] = {}
        SP.fold_palette(t_in["palette"], still_pal, fixes)
    if psys:
        t_in = copy.deepcopy(t_in)
        PS.fold_type(t_in, psys, fixes)
        if isinstance(t_in.get("words"), list):
            PS.fold_words(t_in["words"], psys, fixes, protected=window_free(t_in["words"], c_in), seed=psys.get("seed", 0))
    if not isinstance(c_in, dict):
        errors.append("missing object `compose`")
        c_in = {}
    t: dict = {}
    t["mood"] = str(t_in.get("mood") or "").strip()[:40] or "driving"
    e = _clamp(t_in.get("energy", 0.7), "energy", fixes, "type.energy")
    t["energy"] = 0.7 if e is None else round(e, 3)
    pal = t_in.get("palette") if isinstance(t_in.get("palette"), dict) else {}
    p = {}
    for k in ("bg", "fg"):
        v = _rgb(pal.get(k), 255.0, f"type.palette.{k}", fixes, errors)
        if v:
            p[k] = v
    acc = [_rgb(a, 255.0, "type.palette.accents[]", fixes, errors) for a in (pal.get("accents") or [])[:4]]
    p["accents"] = [a for a in acc if a] or [p.get("fg", [235, 235, 235])]
    if len(p["accents"]) < 3:
        fixes.append(f"type.palette.accents: {len(p['accents'])} given, padded to 3 (the 3D modules read accents[2])")
        pad_accents(p)
    t["palette"] = p
    cam = t_in.get("camera", "beat_orbit")
    if cam not in CAMERAS:
        errors.append(f"type.camera {cam!r}: allowed {', '.join(CAMERAS)}")
        cam = "beat_orbit"
    t["camera"] = cam
    cp = t_in.get("camera_params") if isinstance(t_in.get("camera_params"), dict) else {}
    cpo = {}
    for k in ("read_dps", "intro_deg", "fast_dps", "elev", "bob"):
        if k in cp:
            v = _clamp(cp[k], k, fixes, f"type.camera_params.{k}")
            if v is not None:
                cpo[k] = v
    if cam == "beat_orbit" and cpo:
        t["camera_params"] = cpo
    tr = t_in.get("transitions", "auto")
    if tr != "auto":
        if not isinstance(tr, list):
            errors.append("type.transitions: 'auto' or a list")
            tr = "auto"
        else:
            bad = [x for x in tr if x is not None and x not in TRANSITIONS]      # null = the engine picks that gap (the planner writes it too)
            if bad:
                errors.append(f"type.transitions: unknown {bad}; allowed {', '.join(TRANSITIONS)}")
            nm = [x for x in tr if x not in (None, "cut", "contour_morph")]
            if len(nm) != len(set(nm)):
                errors.append("type.transitions: a transition (other than cut) is used twice")
            if len(tr) != len(want) - 1:
                errors.append(f"type.transitions: needs {len(want) - 1} entries (one per gap between words) or the string \"auto\", got {len(tr)}")
    t["transitions"] = tr
    lp = _clamp((t_in.get("line") or {}).get("px", 4.0), "line.px", fixes, "type.line.px")
    t["line"] = {"px": 4.0 if lp is None else round(lp, 2)}
    words_in = t_in.get("words")
    out_words = []
    if not isinstance(words_in, list) or len(words_in) != len(want):
        errors.append(f"type.words: needs exactly {len(want)} entries in order ({', '.join(want)}), got {len(words_in) if isinstance(words_in, list) else 'none'}")
        words_in = words_in if isinstance(words_in, list) else []
    from mograph import font_registry as fr
    byfile = {v["file"]: v for v in fr.REGISTRY.values()}
    for i, w in enumerate(words_in[:len(want)]):
        wh = f"type.words[{i}]"
        if not isinstance(w, dict):
            errors.append(f"{wh}: not an object")
            continue
        o: dict = {}
        if str(w.get("text", "")).strip().upper() != want[i].upper():
            errors.append(f"{wh}.text {w.get('text')!r}: must be {want[i]!r}")
        o["text"] = want[i]
        font = w.get("font")
        try:
            ent = fl.require_liked(font)
            o["font"] = ent["file"]
        except Exception:       # noqa: BLE001
            errors.append(f"{wh}.font {font!r}: not in the liked font list; use a file name from the FONTS list")
            o["font"] = None
        mod = w.get("module")
        if mod is None:                                             # not written: one of the piece's own modules, else a rotation (never "lanes" as the silent default)
            own = [x.get("module") for x in words_in if isinstance(x, dict) and x.get("module") in WORD_MODULES]
            mod = own[i % len(own)] if own else OP.default_module(i, int(orig["type"].get("seed", 1) or 0))
        if mod not in WORD_MODULES:
            errors.append(f"{wh}.module {mod!r}: allowed {', '.join(WORD_MODULES)}")
        o["module"] = mod
        if w.get("lines"):
            ls = [str(x) for x in w["lines"]]
            if "".join(ls).replace(" ", "").upper() != want[i].replace(" ", "").upper():
                errors.append(f"{wh}.lines {ls}: joined without spaces they must spell {want[i]!r}")
            elif not DO.word_aligned(want[i], ls):
                fixes.append(f"{wh}.lines {ls}: cuts a word; dropped (lines break only at spaces, a word stays on one line and is scaled to fit)")
            else:
                o["lines"] = [x.upper() for x in ls]
        for k in ("form_in", "form_out"):
            v = w.get(k)
            if v is None:                                       # not written: one of the piece's own styles, else a rotation (never "lines from the top" as the silent default, openings.py)
                v = OP.default_style(i, [x.get(k) for x in words_in if isinstance(x, dict) and x.get(k) in IO_STYLES], int(orig["type"].get("seed", 1) or 0))
            if v not in IO_STYLES:
                errors.append(f"{wh}.{k} {v!r}: allowed {', '.join(IO_STYLES)}")
                v = OP.default_style(i, [], int(orig["type"].get("seed", 1) or 0))             # an error goes back to the model; the placeholder is not `lines`
            o[k] = v
        for k in ("entry", "exit"):                             # the side the strands come from / leave to (module ports); a contour port dict is not offered to the design
            v = w.get(k)
            if isinstance(v, str) and v.strip().lower() in OP.SIDES:
                o[k] = v.strip().lower()
            elif v is not None:
                fixes.append(f"{wh}.{k} {v!r}: not one of {', '.join(OP.SIDES)}, dropped")
        try:
            o["hit"] = float(w["hit"])
        except Exception:       # noqa: BLE001
            errors.append(f"{wh}: `hit` (seconds, a downbeat or beat from the music analysis) is required")
            o["hit"] = 0.0
        try:
            hold_in = float(w["hold_s"])
        except Exception:       # noqa: BLE001
            errors.append(f"{wh}: `hold_s` (seconds the finished word stays still) is required")
            hold_in = 1.6
        if hold_in < MIN_HOLD_IN:
            errors.append(f"{wh}.hold_s {hold_in:g}: a word must hold at least {MIN_HOLD_IN:g} s (the guide asks 1.6 s where the time allows)")
        v = _clamp(hold_in, "hold_s", fixes, f"{wh}.hold_s")
        o["hold_s"] = round(v if v is not None else 1.6, 3)
        for k in ("form_s", "exit_s"):
            v = _clamp(w.get(k, 0.75 if k == "form_s" else 0.5), k, fixes, f"{wh}.{k}")
            o[k] = round(v if v is not None else 0.6, 3)
        pl = w.get("placement") if isinstance(w.get("placement"), dict) else {}
        po = {}
        for k in ("yaw", "wf", "hf", "pitch", "roll", "dx", "dy"):
            if k in pl:
                v = _clamp(pl[k], k, fixes, f"{wh}.placement.{k}")
                if v is not None:
                    po[k] = round(v, 3)
        if po:
            o["placement"] = po
        so = w.get("screen_offset")
        if so is not None:
            try:
                a = [_clamp(x, "so", fixes, f"{wh}.screen_offset") for x in so]
                assert len(a) == 2 and None not in a
                if any(abs(x) > 1e-9 for x in a):
                    o["screen_offset"] = [round(x, 3) for x in a]
            except Exception:       # noqa: BLE001
                errors.append(f"{wh}.screen_offset: [x, y] numbers")
        ax = w.get("axes")
        if isinstance(ax, dict) and o.get("font"):
            vary = (byfile.get(o["font"]) or {}).get("vary") or {}
            axo = {}
            for tag, val in ax.items():
                if tag not in vary:
                    fixes.append(f"{wh}.axes.{tag}: not an axis of {o['font']}, dropped")
                    continue
                lo, hi = vary[tag]
                try:
                    if isinstance(val, (list, tuple)):
                        a0, a1 = float(val[0]), float(val[1])
                        axo[tag] = [min(max(a0, lo), hi), min(max(a1, lo), hi), "beat"]
                    else:
                        axo[tag] = min(max(float(val), lo), hi)
                except Exception:       # noqa: BLE001
                    fixes.append(f"{wh}.axes.{tag}: unreadable, dropped")
            if axo:
                o["axes"] = axo
        mt = w.get("matte")
        if isinstance(mt, str) and mt.strip().lower() in ("behind", "none", "null", "no", "plain", "off"):
            fixes.append(f"{wh}.matte {mt!r}: not a treatment (plain behind is the technique), dropped")
            mt = None
        if mt:
            if mt in MATTES:
                o["matte"] = mt
            else:
                errors.append(f"{wh}.matte {mt!r}: allowed {', '.join(MATTES)} (or omit)")
        if w.get("window_source") not in (None, "", "self"):
            o["window_source"] = str(w["window_source"])
        if w.get("window_ground"):                                     # inverse type on a flat ground (planned by the code, design_options.plan_flat_window): the letters show the picture, the ground is a palette colour
            o["window_ground"] = w["window_ground"] if isinstance(w["window_ground"], list) else "plain"
            o.setdefault("window_source", "self")
        owner_fields(w, o, wh, fixes, t, orig)
        out_words.append(o)
    t["words"] = out_words
    if plan_windows:                                         # the app's design step: the modules are folded into the piece's one system BEFORE the timing is compiled and fitted (a module changes a word's span)
        from tools.fusion import design_defaults as DD
        DD.one_module_system(out_words, fixes)
        DD.one_treatment(out_words, fixes)                   # io styles change the form / exit spans too (v007: one fold of them overran the last hold)
        OP.enforce(out_words, fixes, ((orig.get("plan") or {}).get("earlier_openings")) or (), int(orig["type"].get("seed", 1) or 0), system=psys)      # openings AFTER the fold (the fold caps the io styles, this makes the entries differ within the cap): no two words enter the same way, >= 2 io styles, word 1 does not open like the piece before it
    for i_, r_ in (pins or {}).items():
        if 0 <= int(i_) < len(out_words):
            out_words[int(i_)]["_pin_rel"] = float(r_)             # private key: _compile keeps this word's release there; dropped with the other "_" keys (design_step._clean_type)
    if out_words and not errors:
        sw_ = shot_windows(orig, len(out_words))
        errors.extend(compile_timing(t, ana, fixes, shots=sw_, free=window_free(words_in, c_in), pre=cut_pre_windows(c_in, sw_)))
    c: dict = {}
    tech = c_in.get("technique", "behind")
    if tech not in TECHNIQUES and tech not in allowed_techniques(name):      # the plan's own technique ('built_in' of a single shot, 'traced') is allowed though it is no TECHNIQUES entry: the model is told to pick from the allowed list
        errors.append(f"compose.technique {tech!r}: allowed {', '.join(allowed_techniques(name))}")
        tech = "behind"
    elif tech not in allowed_techniques(name):
        ok_t = allowed_techniques(name)
        fixes.append(f"compose.technique {tech!r} needs a matte / depth the footage does not have; {ok_t[0]!r} used instead")
        tech = ok_t[0]
    c["technique"] = tech
    for k in ("plate_gain", "plate_sat", "glow_lo", "glow_hi", "grain", "vignette", "type_opacity"):
        if k in c_in:
            v = _clamp(c_in[k], k, fixes, f"compose.{k}")
            if v is not None:
                c[k] = round(v, 4)
    if "glow_color" in c_in:
        v = _rgb(c_in["glow_color"], 1.0, "compose.glow_color", fixes, errors)
        if v:
            c["glow_color"] = v
    cuts, cf = DO.normalize_cuts(c_in.get("cuts"))
    fixes += cf
    if cuts:
        c["cuts"] = cuts
    hs = []
    for h in (c_in.get("holds") or [])[:6]:
        try:
            ht, hd = float(h[0]), float(h[1])
            if 0.0 < ht < cut - 0.3:
                hs.append([round(ht, 2), round(min(max(hd, 0.2), 1.0), 2)])
            else:
                fixes.append(f"compose.holds {h}: time outside the piece, dropped")
        except Exception:       # noqa: BLE001
            fixes.append(f"compose.holds {h}: unreadable, dropped")
    if hs:
        c["holds"] = hs
    d = dict(concept=str(obj.get("concept") or "").strip()[:300], treatment_notes=str(obj.get("treatment_notes") or "").strip()[:300], type=t, compose=c)
    if not d["concept"]:
        errors.append("`concept` is empty")
    return d, fixes, errors


def _fmt_of(orig: dict) -> str:
    try:
        from tools.fusion import plan_rules as PR
        return PR.format_of(orig)
    except Exception:       # noqa: BLE001
        return "9:16"


def finish_design(design: dict, name: str, fixes: list, diversify: bool = True) -> dict:
    """after normalize_design + fit_scene (the holds are final): 1. the text-as-window validator (design_options.validate: boundaries, fat faces, cap
    height, beat fit, adjacency), 2. the font rules (diversify_fonts: expressive hero, >= 3 faces), so a window word's heavy face counts and is never
    swapped away. Fonts / axes / placement / window_source / compose.cuts come back; the timing keys stay as they were. -> the design (changed in place)."""
    orig, ana = load_original(name), load_analysis(name)
    ctx_ = _CTX.get(name) or {}
    seed = orig["type"].get("seed", 1)
    words = design["type"]["words"]
    if orig.get("shots"):               # also with ONE shot: the validator drops cuts / window_source there (a window_cut `after s1` crashed apply_to_piece)
        from tools.fusion import plan_rules as PR
        probe = copy.deepcopy(orig)
        probe["type"] = dict(copy.deepcopy(design["type"]), seed=seed)
        PR.fill_type_timing(_morph_names_to_null(probe), ana)
        try:
            chk = PR.dry_run(probe)["words"]
        except Exception:       # noqa: BLE001  the engine's own error is reported by timing_errors
            chk = []
        view = copy.deepcopy(design)
        cp_ = (orig.get("plan") or {}).get("cut_pins") or []
        if cp_:                                                                   # the user's camera move per shot (storyboard): an explicit entry at that boundary, the model's entry there is replaced
            rows_ = shot_rows(orig)
            ents = {c.get("after"): c for c in ((view.get("compose") or {}).get("cuts") or []) if isinstance(c, dict)}
            for k, kd in enumerate(cp_[:max(len(rows_) - 1, 0)]):
                if kd:
                    ents[rows_[k]["id"]] = {"after": rows_[k]["id"], "cut": kd, "why": "the user's own camera move for this shot (storyboard)"}
                    fixes.append(f"compose.cuts after {rows_[k]['id']}: {kd} (the user's choice)")
            view.setdefault("compose", {})["cuts"] = list(ents.values())
        fixes.extend("timing: " + e for e in shot_hold_errors(chk, shot_windows(orig, len(chk)), window_free(design["type"].get("words"), design.get("compose") or {})))      # reported, not rejected: the fit already tried
        for w, c in zip(view["type"]["words"], chk):
            w["t0"], w["hit"], w["release"] = float(c["start"]), float(c["hit"]), float(c["release"])
        type_dir = ctx_.get("type_dir")
        caps = None
        if type_dir:
            try:
                caps = DO.caps_from_manifest(type_dir)
            except Exception:       # noqa: BLE001
                caps = None
        new, fx = DO.validate(view, dict(shots=shot_rows(orig), ana=ana, caps=caps, fill_3d=True, system=PS.for_piece(orig, ana)[0]))
        fixes.extend(fx)
        try:                                                                      # 2026-10-30: a transition must be SEEN: report the words whose hold runs on under a 3D cut's move
            ov_ = ST.transition_overlaps(view["type"]["words"], ST.transition_windows((new.get("compose") or {}).get("cuts"), shot_rows(orig)))
            fixes.extend(f"timing: word {i} {view['type']['words'][i].get('text', '')!r} still holds {sec:g} s under the {kd} move (the cut should be seen: shorten the hold or move the hit)" for i, kd, sec in ov_)
        except Exception:       # noqa: BLE001  the engine is not importable here: the check is skipped
            pass
        for w, nw in zip(words, new["type"]["words"]):
            for k in ("font", "axes", "placement", "window_source", "window_ground"):
                if k in nw:
                    w[k] = nw[k]
                else:
                    w.pop(k, None) if k in ("axes", "window_source", "window_ground") else None
        cuts = (new.get("compose") or {}).get("cuts")
        design.setdefault("compose", {})
        if cuts:
            design["compose"]["cuts"] = cuts
        else:
            design["compose"].pop("cuts", None)
    from tools.fusion import font_pins as FP                    # the user's per-line fonts (storyboard) win over every font rule below
    pins = (orig.get("type") or {}).get("font_pins") or []
    if pins:
        design["type"]["font_pins"] = copy.deepcopy(pins)
        FP.apply(words, pins, fixes)                            # first, so the rules below see them (contrast / >= 3 faces are built AROUND a chosen face) ...
    if diversify:
        psys_ = (orig.get("plan") or {}).get("piece_system")
        diversify_fonts(design["type"], fixes, seed, window_words(design) | FP.pinned_indices(words, pins), only=(psys_ or {}).get("fonts") if isinstance(psys_, dict) else None)
        from tools.fusion import design_defaults as DD          # the draft strips' qualities as defaults: one type system (<= 2 modules, <= 3 faces, one treatment family), no echo modules, the size floor
        DD.enforce_type(design["type"]["words"], fixes, protected=window_words(design), caps=None, font_protected=FP.pinned_indices(words, pins))
    FP.apply(words, pins, fixes)                                # ... and last, so no later swap (window validator, legibility / pixel guard) can undo them
    FP.contrast(words, pins, fixes)                             # the lines nobody chose still contrast with the chosen ones
    ST.enforce(design["type"], PS.for_piece(orig, ana)[0], ana.get("bpm"), fixes, protected=window_words(design), seed=seed, fmt=_fmt_of(orig))      # centred, still >= 1 beat, stacked / rolled layout, held camera
    return design


def shot_hold_errors(ws: list, sw, skip=frozenset()) -> list[str]:
    """word k lives in shot k: it must be on screen, held, for MIN_HOLD_S inside its OWN shot (round-3 QC 2026-10-29: TIDE never appeared). ws = the dry-run words (text, hit, release),
    sw = shot_windows [(start, end, id)] or None, skip = window words (they span the cut on purpose). -> error texts."""
    errs = []
    for k, (w, (a0, b0, sid)) in enumerate(zip(ws, sw or [])):
        seen = min(float(w["release"]), b0) - max(float(w["hit"]), a0)
        if k not in skip and seen < MIN_HOLD_S - 1e-6:
            errs.append(f"{w['text']!r} is held only {max(seen, 0.0):.2f} s inside its own shot {sid} ({a0:g}..{b0:g} s); it needs at least {MIN_HOLD_S:g} s: move the hit earlier in the shot or shorten the form time")
    return errs


def timing_errors(piece: dict, ana: dict) -> tuple[list[str], dict]:
    """the repo's own validator: spec.build_scene through plan_rules.dry_run on a type spec with the analysis timing filled in."""
    from tools.fusion import plan_rules as PR
    p = copy.deepcopy(piece)
    PR.fill_type_timing(_morph_names_to_null(p), ana)       # contour_morph is a legitimate transition name; the type engine only knows the null gap (the copy is normalised, the caller's piece keeps the name)
    try:
        chk = PR.dry_run(p)
    except Exception as e:      # noqa: BLE001
        return [f"the type engine rejected the design: {type(e).__name__}: {str(e)[:300]}"], {}
    errs = []
    for n in chk.get("notes") or []:
        pass                                                 # engine notes (a transition replaced by a cut between flat modules, an exit cut by the end of the music) are handled by the engine
    for w in chk["words"]:
        hold = float(w["release"]) - float(w["hit"])
        if hold < MIN_HOLD_S:
            errs.append(f"{w['text']!r}: the planned hold is {hold:.2f} s (hit {w['hit']:.2f} -> release {w['release']:.2f}); it needs at least 1.0 s: raise hold_s")
    ws = chk["words"]
    for a, b in zip(ws, ws[1:]):
        if float(b["start"]) < float(a["end"]) - 0.05:
            errs.append(f"{a['text']!r} (ends {a['end']:.2f}) overlaps {b['text']!r} (starts {b['start']:.2f}) after beat snapping")
    if ws and float(ws[-1]["release"]) > float(ana["cut"]) - 0.2:
        errs.append(f"the last word's hold ends at {ws[-1]['release']:.2f} s, too close to the end of the music (cut {float(ana['cut']):g} s): move hits earlier or shorten holds / exits")
    if errs:
        plan = "; ".join(f"{w['text']} start {w['start']:.2f} hit {w['hit']:.2f} release {w['release']:.2f} end {w['end']:.2f}" for w in ws)
        errs.append(f"the engine's planned timeline was: {plan}. Downbeats: " + ", ".join(f"{float(d):g}" for d in ana["downbeats"] if float(d) < float(ana["cut"])))
    return errs, chk


# ----------------------------------------------------------------------------------------------------------- merge + compose
def _hex(c):
    return "#%02x%02x%02x" % tuple(int(x) for x in c)


def build_compose(orig: dict, design: dict, ana: dict) -> dict:
    """the model's compose description -> compose section: the repo recipe for `technique` (colours from the design palette, timelines scaled
    to the cut) with the grade / glow / grain / vignette / opacity values patched in. Locked: size, crossfade, crf. No flow_wake (the local CPU
    render has no $flow; plan_rules.weave_flow adds it on the box)."""
    from tools.fusion import plan_rules as PR
    c = design["compose"]
    pal = design["type"]["palette"]
    cols = [_hex(pal.get("bg", [8, 8, 8])), _hex(pal.get("fg", [235, 235, 235]))] + [_hex(a) for a in pal.get("accents", [])]
    rec = PR.load_recipe(c["technique"])
    size = orig["compose"].get("size", [1080, 1920])
    comp = PR.apply_recipe(rec, float(ana["cut"]), cols, size=size)
    ops = comp["default"]["ops"]
    for o in ops:
        a = o["args"]
        if o["op"] == "grade" and o.get("out") == "foot":
            if "plate_gain" in c:
                a["gain"] = c["plate_gain"]
            if "plate_sat" in c:
                a["sat"] = c["plate_sat"]
        elif o["op"] == "add_glow":
            it = a.get("intensity")
            if isinstance(it, dict) and ("glow_lo" in c or "glow_hi" in c):
                it = dict(it)
                it["lo"] = c.get("glow_lo", it.get("lo", 0.05))
                it["hi"] = max(c.get("glow_hi", it.get("hi", 0.3)), it["lo"])
                a["intensity"] = it
            if "glow_color" in c:
                a["color"] = c["glow_color"]
        elif o["op"] == "grain" and "grain" in c:
            a["amount"] = c["grain"]
        elif o["op"] == "vignette" and "vignette" in c:
            a["strength"] = c["vignette"]
        elif o["op"] == "sandwich" and "type_opacity" in c:
            a["type_opacity"] = c["type_opacity"]
    out = {"size": size, "crossfade": orig["compose"].get("crossfade", 0.0), "crf": orig["compose"].get("crf", 16), **comp}
    if c.get("holds"):
        out["holds"] = c["holds"]
        out["holds_type"] = True
    return out


def media_for(name: str, orig: dict) -> tuple[dict, dict]:
    """(stills {id: path}, shots {id: media dict}): repo-relative media declarations for every existing render."""
    P = PIECES[name]
    stills = {s["id"]: P["still"].format(sid=s["id"]) for s in orig.get("stills", [])}
    shots = {}
    for s in orig["shots"]:
        m = {"shot": f"{P['ing']}/{P['clip'].format(sid=s['id'])}"}
        if s.get("matte"):
            m["matte"] = P["matte_dir"].format(sid=s["id"])
        shots[s["id"]] = m
    return stills, shots


def merge_locked(name: str, design: dict, version: str, model_title: str, seed: int, usage: dict | None = None, repairs: int = 0, type_dir=None) -> dict:
    """original JSON with everything the model may not touch copied back, the model's type / compose merged in, media + version + parent added."""
    P = PIECES[name]
    orig = load_original(name)
    ana = load_analysis(name)
    out = copy.deepcopy(orig)
    out["id"] = f"{name}_{version}"
    out["title"] = f"{P['title']} {version} — {model_title} s{seed}"
    out["version"] = version
    reuse = ["stills", "shots", "alpha", "audio"] + (["masktrack"] if any(s.get("masktrack") for s in orig["shots"]) else [])
    out["parent"] = {"piece": name, "version": P["parent_version"], "reuse": reuse,
                     "why": f"Qwen-only design ({model_title}, seed {seed}) from the fixed design guide; reuses {name} {P['parent_version']} footage"}
    t = copy.deepcopy(design["type"])
    for w in t["words"]:
        for k in [k for k in w if k.startswith("_")]:
            w.pop(k)
    t["seed"] = orig["type"].get("seed", 1)
    if orig["type"].get("font_pins"):                           # the user's per-line fonts (tools/fusion/font_pins.py): locked, the model cannot change them
        from tools.fusion import font_pins as FP
        t["font_pins"] = copy.deepcopy(orig["type"]["font_pins"])
        FP.apply(t["words"], t["font_pins"])
    out["type"] = t
    out["compose"] = build_compose(orig, design, ana)
    out["signoff"] = dict(orig.get("signoff") or {}, off=True)
    stills, shots = media_for(name, orig)
    for s in out.get("stills", []):
        s["media"] = stills[s["id"]]
    for s in out["shots"]:
        s["media"] = shots[s["id"]]
    out["design"] = {"model": model_title, "seed": seed, "concept": design["concept"], "treatment_notes": design["treatment_notes"],
                     "repair_turns": repairs, "technique": design["compose"]["technique"], "usage": usage or {}}
    # matte treatments (per word) woven into the compose chain with the repo's own weaver
    from tools.fusion import plan_rules as PR
    probe = copy.deepcopy(out)
    PR.fill_type_timing(probe, ana)
    chk = PR.dry_run(probe)
    PR.weave_matte(probe, ana, chk)
    out["compose"] = probe["compose"]
    if probe.get("plan"):
        out["plan"] = {"matte_treatments": probe["plan"].get("matte_treatments")}
    notes = DO.to_piece(out, design, type_dir)            # window choices: compose.cuts + the words' window_source + the shots' window_cut ops
    if notes:
        out.setdefault("plan", {})["window_notes"] = list(notes)
    return out


# ----------------------------------------------------------------------------------------------------------- chat
def _client():
    raise SystemExit("no hosted-model client in this export: pass chat_fn (the app runs the design model through the Gemma GGUF graph nodes)")


def chat(OR, key, model, messages, temperature, seed, max_tokens):
    raise SystemExit("no hosted-model client in this export")


def user_message(prompt: str, images) -> dict:
    import base64
    import mimetypes

    def part(path):
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")}}
    return {"role": "user", "content": [{"type": "text", "text": prompt}] + [part(i) for i in images]}


def repair_text(errors: list[str]) -> str:
    return ("The design was checked by the engine and has these errors:\n" + "\n".join(f"- {e}" for e in errors[:14]) +
            "\nReturn the COMPLETE corrected JSON object again (same schema, nothing else). Keep the concept; change only what the errors require.")


def design_once(name: str, model: str, seed: int, design_seed: int, outdir: Path, tag: str, temperature=0.9, max_tokens=4500, max_repairs=2,
                package: dict | None = None, chat_fn=None) -> dict:
    """package -> model -> parse -> validate -> up to `max_repairs` repair turns with the SAME model. Writes raw replies + parsed design into outdir.
    -> result dict (valid, design, fixes, errors, repairs, usage list, cost)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pkg = package or build_package(name, design_seed, outdir / "package")
    ana = load_analysis(name)
    orig = load_original(name)
    if chat_fn is None:
        OR, key = _client()
        chat_fn = lambda msgs: chat(OR, key, model, msgs, temperature, seed, max_tokens)      # noqa: E731
    msgs = [user_message(pkg["prompt"], pkg["images"])] if (pkg["images"] or pkg["prompt"] != "P") else [{"role": "user", "content": pkg["prompt"]}]
    usages, design, fixes, errors, repairs = [], None, [], [], 0
    for turn in range(max_repairs + 1):
        text, u = chat_fn(msgs)
        usages.append(u)
        (outdir / f"{tag}_raw_turn{turn}.txt").write_text(text, encoding="utf-8")
        obj = extract_json(text)
        if obj is None:
            design, fixes, errors = None, [], ["the reply contained no parseable JSON object; return only the JSON object"]
        else:
            design, fixes, errors = normalize_design(obj, name)
            if design is not None and not errors:
                errors = fit_scene(design, name, fixes)
            if design is not None and not errors:
                finish_design(design, name, fixes)
            if design is not None and not errors:
                probe = copy.deepcopy(orig)
                probe["type"] = dict(design["type"], seed=orig["type"].get("seed", 1))
                errors, _ = timing_errors(probe, ana)
        if not errors:
            break
        if turn < max_repairs:
            repairs += 1
            if turn == 0 and len(msgs) == 1 and isinstance(msgs[0]["content"], list):      # repair turns carry the text package only (images are re-sent at ~3.5k tokens a turn)
                msgs = [{"role": "user", "content": [msgs[0]["content"][0]]}]
            msgs = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": repair_text(errors)}]
    cost = sum(float(x["cost"]) for x in usages if x.get("cost") is not None)
    res = dict(valid=not errors and design is not None, design=design, fixes=fixes, errors=errors, repairs=repairs, usage=usages, cost=round(cost, 6))
    (outdir / f"{tag}_result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------------------------------------- state seed + checks
def _probe(path: Path):
    from tools.fusion import compose
    w, h, nf, fps = compose._probe_video(path)
    return int(w), int(h), int(nf)


def seed_state(piece_json: Path, fusion_ver: str, log=print) -> Path:
    """15_fusion/<fusion_ver>/local/pieces/<id>/state.json with a done record for every adopted kind (audio, stills, shots, alpha, masktrack), the
    runner's own resume mechanism. Needed because piece_runner._media cannot adopt PNG-folder mattes or masktrack results; with it the stock
    `--dry` / run shows only type -> compose -> final."""
    pj = Path(piece_json)
    j = json.loads(pj.read_text(encoding="utf-8"))
    name = j["parent"]["piece"]
    P = PIECES[name]
    d = ROOT / "15_fusion" / fusion_ver / "local" / "pieces" / j["id"]
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "state.json"
    if sp.exists():
        raise FileExistsError(f"{sp} exists; never overwritten")
    orig_state = json.loads((ROOT / P["state"]).read_text(encoding="utf-8"))
    z = dict(wall_s=0.0, gpu_s=0.0, gpu_exec_s=0.0)
    items = {"audio": copy.deepcopy(orig_state["items"]["audio"])}
    for s in j.get("stills", []):
        p = str(ROOT / s["media"])
        items[f"still:{s['id']}"] = dict(out=dict(paths=[p], local=p, media=True), **z)
    for s in j["shots"]:
        m = s["media"]
        clip = ROOT / m["shot"]
        w, h, nf = _probe(clip)
        items[f"shot:{s['id']}"] = dict(out=dict(paths=[str(clip)], local=str(clip), frames=nf, size=[w, h], media=True), **z)
        if s.get("matte"):
            fl = [str(f) for f in sorted((ROOT / m["matte"]).glob("*.png"))]
            items[f"alpha:{s['id']}"] = dict(out=dict(paths=fl, files=fl, n=len(fl), media=True), **z)
        if s.get("masktrack"):
            mt = ROOT / P["masktrack"].format(sid=s["id"])
            w2, h2, nf2 = _probe(mt)
            items[f"masktrack:{s['id']}"] = dict(out=dict(ok=True, paths=[str(mt)], local=str(mt), frames=nf2, size=[w2, h2], media=True), **z)
    st = {"piece": j["id"], "items": items, "attempts": {}, "pending": {}, "stages": {}, "seeded_from": P["state"]}
    sp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    log(f"seeded {sp} with {len(items)} records")
    return sp


def media_check(piece_json: Path) -> list[str]:
    j = json.loads(Path(piece_json).read_text(encoding="utf-8"))
    P = PIECES[j["parent"]["piece"]]
    probs = []
    for s in j.get("stills", []):
        if not (ROOT / s["media"]).is_file():
            probs.append(f"still {s['id']}: missing {s['media']}")
    for s in j["shots"]:
        m = s["media"]
        if not (ROOT / m["shot"]).is_file():
            probs.append(f"shot {s['id']}: missing {m['shot']}")
        else:
            try:
                _, _, nf = _probe(ROOT / m["shot"])
                if nf < int(s["frames"]):
                    probs.append(f"shot {s['id']}: clip has {nf} frames, piece wants {s['frames']}")
            except Exception as e:      # noqa: BLE001
                probs.append(f"shot {s['id']}: unreadable ({e})")
        if m.get("matte"):
            n = len(list((ROOT / m["matte"]).glob("*.png")))
            if n < int(s["frames"]):
                probs.append(f"matte {s['id']}: {n} PNG frames, shot has {s['frames']}")
        if s.get("masktrack") and not (ROOT / P["masktrack"].format(sid=s["id"])).is_file():
            probs.append(f"masktrack {s['id']}: composite missing")
    return probs


def locked_diff(name: str, piece: dict) -> list[str]:
    """fields that must equal the original; returns differences (empty = locked fields intact)."""
    o = load_original(name)
    bad = []
    for k in ("audio", "fps", "type_passes"):
        if piece.get(k) != o.get(k):
            bad.append(k)
    strip = lambda L: [{k: v for k, v in x.items() if k != "media"} for x in L]       # noqa: E731
    if strip(piece["shots"]) != strip(o["shots"]):
        bad.append("shots")
    if strip(piece.get("stills", [])) != strip(o.get("stills", [])):
        bad.append("stills")
    if [w["text"] for w in piece["type"]["words"]] != [w["text"] for w in o["type"]["words"]]:
        bad.append("type.words text/order")
    if piece["type"].get("seed") != o["type"].get("seed"):
        bad.append("type.seed")
    for k in ("size", "crossfade", "crf"):
        if piece["compose"].get(k) != o["compose"].get(k):
            bad.append(f"compose.{k}")
    return bad


# ----------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("piece")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p = sub.add_parser("design")
    p.add_argument("piece")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--version", required=True)
    p.add_argument("--gen", default="01")
    p.add_argument("--out", required=True, help="version local/ folder")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-repairs", type=int, default=2)
    p = sub.add_parser("replay", help="re-validate the SAVED raw replies of a design under the current validator (no API call); writes the piece if valid")
    p.add_argument("piece")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--version", required=True)
    p.add_argument("--gen", default="01")
    p.add_argument("--out", required=True)
    p = sub.add_parser("seed-state")
    p.add_argument("piece_json")
    p.add_argument("--fusion-version", default="v006")
    p = sub.add_parser("media-check")
    p.add_argument("piece_json")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        r = build_package(a.piece, a.seed + PIECES[a.piece]["seed_shift"], Path(a.out))
        print(f"prompt {len(r['prompt'])} chars, {len(r['images'])} images -> {a.out}")
        return 0
    if a.cmd == "seed-state":
        seed_state(Path(a.piece_json), a.fusion_version)
        return 0
    if a.cmd == "media-check":
        pr = media_check(Path(a.piece_json))
        print("\n".join(pr) if pr else "media ok")
        return 1 if pr else 0
    if a.cmd == "replay":
        mt = next(m for m in MODELS if m[0] == a.model)
        out = Path(a.out)
        tag = f"{a.piece}_{mt[1]}_g{a.gen}"
        live = json.loads((out / f"{tag}_result.json").read_text(encoding="utf-8"))
        raws = sorted(out.glob(f"{tag}_raw_turn*.txt"))
        it = iter(zip(raws, live["usage"]))
        replay_fn = lambda msgs: (lambda rp, u: (rp.read_text(encoding="utf-8"), u))(*next(it))      # noqa: E731
        rdir = out / "replay"
        res = design_once(a.piece, a.model, a.seed, a.seed + PIECES[a.piece]["seed_shift"], rdir, tag, package={"prompt": "P", "images": []}, chat_fn=replay_fn)
        res["cost_actual"] = live["cost"]
        res["replayed_from"] = f"{tag}_raw_turn*.txt under the validator of {time.strftime('%Y-%m-%d')}"
        (rdir / f"{tag}_result.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"{tag}: replay valid={res['valid']} repairs={res['repairs']} (live run: repairs={live['repairs']} cost=${live['cost']})")
        if res["valid"]:
            pj = ROOT / "15_fusion" / "pieces" / f"{a.piece}_{a.version}.json"
            if pj.exists():
                print(f"{pj} exists, never overwritten")
                return 2
            piece = merge_locked(a.piece, res["design"], a.version, mt[2], a.seed, usage=dict(cost=live["cost"], turns=live["usage"]), repairs=res["repairs"])
            (out / f"{tag}.json").write_text(json.dumps(res["design"], indent=1), encoding="utf-8")
            pj.write_text(json.dumps(piece, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {pj}")
        return 0 if res["valid"] else 1
    if a.cmd == "design":
        mt = next(m for m in MODELS if m[0] == a.model)
        out = Path(a.out)
        tag = f"{a.piece}_{mt[1]}_g{a.gen}"
        dseed = a.seed + PIECES[a.piece]["seed_shift"]
        pk = out / f"package_{a.piece}_s{a.seed}"
        res = design_once(a.piece, a.model, a.seed, dseed, out, tag, a.temperature, max_repairs=a.max_repairs, package=build_package(a.piece, dseed, pk))
        print(f"{tag}: valid={res['valid']} repairs={res['repairs']} cost=${res['cost']} fixes={len(res['fixes'])} errors={res['errors'][:3]}")
        if res["valid"]:
            pj = ROOT / "15_fusion" / "pieces" / f"{a.piece}_{a.version}.json"
            if pj.exists():
                print(f"{pj} exists, never overwritten")
                return 2
            piece = merge_locked(a.piece, res["design"], a.version, mt[2], a.seed, usage=dict(cost=res["cost"], turns=res["usage"]), repairs=res["repairs"])
            (out / f"{tag}.json").write_text(json.dumps(res["design"], indent=1), encoding="utf-8")
            pj.write_text(json.dumps(piece, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {pj}")
        return 0 if res["valid"] else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
