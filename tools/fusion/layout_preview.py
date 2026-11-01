"""layout_preview: the RANDOMISE LAYOUT PREVIEW (user 2026-10-30: "an option on the preview to create a random seed so we can see a preview of the text layout on the single frames and hit a
randomise button to quickly create a new one"). Pure functions + a small CPU render; no ComfyUI import, no Gemma, no LTX, no GPU.

    redraw_piece(piece, seed, stills, effects)  -> (new piece, info): the per-piece random system DRAWN AGAIN with `seed` on the SAME words / timings / shots:
                                                   fonts (n by the piece length, one face per word by length / legibility), layouts (line / stack / rot90 / grid / column, no-orphan rule),
                                                   the palette (3-5 colours of the first still, extracted with the seed + a seeded legible ink / ground pair), the outline choice,
                                                   the hold-life choice, the cut kinds (compose.cuts remapped onto the drawn kinds)
    render_tiles(piece, stills, ...)            -> one frame per word at its landed hold: the type pass for just those frames (beauty + matte, CPU subprocesses) composed over the
                                                   word's own shot still through legible_color (the same colour / outline logic as the real compose), 360x640
    contact_sheet(tiles, header)                -> the IMAGE: word / layout / font under each tile, the seed + palette + cuts + outline + hold life in the header
    preview(...)                                -> redraw + render + sheet + the files `<stem>_layout_s<seed>.png|.json` (never overwritten: `_b`, `_c` ...)
    with_layout_seed(system, seed, ...)         -> the PLANNER hook: a system whose layout side (fonts, cuts, outline, inverse, effects, seed) is the draw of `seed` and whose module / io /
                                                   camera / photographic style (everything the stills and footage depend on) is kept. plan_rules._build calls it for a request `layout_seed`,
                                                   design_step.still_palette_of uses `still_palette_for` (same seeded palette pick), tools/app/upgrade_still.py applies a saved lock.

Same seed -> same draw (every random choice is a function of (seed, piece), none of the process state). Doc: 16_app/STILL_PREVIEW.md, `tests/test_layout_preview.py`.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import piece_system as PS  # noqa: E402
from tools.fusion import random_system as RS  # noqa: E402
from tools.fusion import stable_type as ST  # noqa: E402
from tools.fusion import still_palette as SP  # noqa: E402

VERSION = 1
TILE_W, TILE_H = 360, 640                  # preview size of one tile (the type pass itself renders at TYPE_SIZE: the engine takes multiples of 64 only)
TYPE_SIZE = (576, 1024)                    # the draft type size (multiples of 64), resized into the tile
LAYOUT_KEYS = ("fonts", "expressive", "anybody_default", "cuts", "outline", "inverse", "effects")     # the layout side of a piece system: what a layout seed re-draws
MAX_PROCS = 8                              # type workers at once (one subprocess per word)
TASK_TIMEOUT_S = 11.0                      # a word whose scene is not built + drawn in this time (a pixel face on the `circuit` module takes 30 s) is drawn flat instead
FPS = 24
LOCK_SUFFIX = "_layout_lock.json"
LAYOUT_LABEL = {"line": "line", "stack": "stack", "rot90": "rot +90", "rot-90": "rot -90", "grid": "grid", "column": "column"}


# ---------------------------------------------------------------------------------------------------------------------------------------------- the request key
def layout_seed_of(v) -> int | None:
    """a request's `layout_seed` -> int >= 1, else None (0 / empty / not a number = no layout seed: the planner's own draw)."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n & 0xFFFFFFFF if n > 0 else None


def effects_list(v) -> list:
    """the `effects` input (a comma / semicolon / newline separated string, or a list) -> the names as written (effects_menu.norm_effects normalises them later)."""
    if isinstance(v, str):
        v = re.split(r"[,;|\n]+", v)
    return [str(x).strip() for x in (v or []) if str(x).strip()]


# ---------------------------------------------------------------------------------------------------------------------------------------------- the palette (3-5 colours of the still + a seeded pair)
def still_palette_for(img, seed: int, k: int | None = None) -> dict:
    """the limited palette of a still for a layout seed: SP.extract with the seed (a different k-means start, k = 3 + seed % 3 colours) + a seeded legible ink / ground pair out of
    `pairs` (the first pair - lightest on darkest - is the most likely, the others give the accent-ink looks). The result is what plan.still_palette holds."""
    seed = int(seed)
    kk = int(k) if k else 3 + seed % 3
    pal = SP.extract(img, k=kk, seed=seed)
    pairs = [p for p in (pal.get("pairs") or []) if len(p) == 2]
    if len(pairs) > 1:
        rng = random.Random(f"lpal:{seed}")
        w = [4.0] + [2.0 if i < 3 else 1.0 for i in range(1, len(pairs))]
        ink, ground = PS._wpick(rng, pairs, w)
        dark_ground = SP.luminance(ground) < SP.luminance(ink)
        if dark_ground == (SP.luminance(pal["bg"]) < SP.luminance(pal["fg"])):              # keep the picture's own dark / light logic: a pale ink on a dark ground in a dark picture
            pal = dict(pal, bg=[int(v) for v in ground], fg=[int(v) for v in ink])
            pal["accents"] = [c for c in pal["colors"] if c not in (pal["bg"], pal["fg"])]
    return pal


# ---------------------------------------------------------------------------------------------------------------------------------------------- the planner hook
def with_layout_seed(system: dict | None, seed: int, words=(), length_s=None, effects=None, still: dict | None = None, mood: str = "", bpm=None) -> dict:
    """the piece system whose LAYOUT side is the draw of `seed`: fonts (n from the length, legible at the longest word), cut kinds + styles, outline mode, inverse option, effects, and `seed` itself
    (stable_type.draw_layouts, the hold-life draw and the outline windows are seeded by system["seed"]). Module / io / camera / palette family / photographic style - what the stills and the
    footage were made from - stay as the base system had them. No base system (a typed request): the full draw of the seed. `still` = a still palette: the type palette of the system."""
    seed = int(seed)
    base = copy.deepcopy(system) if isinstance(system, dict) and system.get("module") else None
    m = (base or {}).get("mood") or mood or ""
    b = ((base or {}).get("motion") or {}).get("bpm") or bpm
    new = PS.choose(seed, m, b, None, (), None, tuple(words or ()), length_s, effects)
    if base is None:
        out = new
    else:
        from tools.fusion import effects_menu as EM
        out = base
        keep_contrast = list(((base.get("effects") or {}).get("contrast")) or [])          # the contrast sections are placed on SHOTS (type_matte keys, plate / matte ingredients): footage side, kept
        for k in LAYOUT_KEYS:
            out[k] = copy.deepcopy(new[k])
        out["seed"] = seed
        out["effects"] = dict(out["effects"], contrast=list(dict.fromkeys(keep_contrast + [n for n in (out["effects"].get("requested") or []) if n in EM.CONTRAST_EFFECTS])))
        out = EM.apply_effects(out, effects)                                                # recomputes effects.active
    out["layout_seed"] = seed
    if still:
        out = PS.with_still_palette(out, still)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- reading an archive
def _read_rgb(path) -> np.ndarray | None:
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def archive_stem(folder) -> str | None:
    """the run stem of a MographArchive folder (`<stem>_piece.json`), None for a folder with only the short legacy names."""
    for n in sorted(os.listdir(folder)):
        if n.endswith("_piece.json"):
            return n[:-len("_piece.json")]
    return None


def load_archive(folder) -> dict:
    """-> {stem, piece (dict), stills {k: RGB uint8}} of a MographArchive folder: `<stem>_piece.json` (else piece.json) and `<stem>_still_<k>.png` per place (else `<stem>_still.png` /
    still.png as place 1). Raises FileNotFoundError without a piece."""
    folder = Path(folder)
    stem = archive_stem(folder)
    pj = folder / (f"{stem}_piece.json" if stem else "piece.json")
    if not pj.is_file():
        raise FileNotFoundError(f"{folder}: no piece.json (is it a MographArchive folder?)")
    piece = json.loads(pj.read_text(encoding="utf-8"))
    stills: dict[int, np.ndarray] = {}
    for k in range(1, 7):
        for n in ([f"{stem}_still_{k}.png"] if stem else []) + [f"still_{k}.png"]:
            if (folder / n).is_file():
                stills[k] = _read_rgb(folder / n)
                break
    if 1 not in stills:
        for n in ([f"{stem}_still.png"] if stem else []) + ["still.png"]:
            if (folder / n).is_file():
                stills[1] = _read_rgb(folder / n)
                break
    return dict(stem=stem or str(piece.get("id") or piece.get("slug") or "piece"), piece=piece, stills=stills)


def stills_from_batch(batch) -> dict:
    """an IMAGE batch (torch [N,H,W,3] float 0..1 or numpy) -> {k: RGB uint8}, row 0 = place 1."""
    arr = batch.detach().cpu().numpy() if hasattr(batch, "detach") else np.asarray(batch)
    return {i + 1: np.clip(np.rint(arr[i, ..., :3] * 255.0), 0, 255).astype(np.uint8) for i in range(arr.shape[0])}


def still_for_time(piece: dict, t: float, stills: dict):
    """(place number k, RGB still) of the shot playing at piece time t: the shot of the layout, its `first` place -> `stills[k]`; the first still without a match."""
    from tools.fusion import places as PL
    lay = piece.get("layout") or {}
    rows = sorted(((int(v.get("start", 0)), int(v.get("frames", 0)), sid) for sid, v in lay.items()), key=lambda r: r[0])
    sid = None
    for a, n, s in rows:
        if a / FPS - 1e-6 <= t < (a + n) / FPS:
            sid = s
    sid = sid or (rows[-1][2] if rows and t >= rows[-1][0] / FPS else (rows[0][2] if rows else None))
    shot = next((s for s in (piece.get("shots") or []) if s.get("id") == sid), {})
    idx = PL.still_index(piece, shot.get("first")) if shot.get("first") is not None else None
    k = (idx + 1) if idx is not None else 1
    img = stills.get(k) if stills.get(k) is not None else (stills.get(1) if stills.get(1) is not None else next(iter(stills.values()), None))
    return k, img


# ---------------------------------------------------------------------------------------------------------------------------------------------- the re-draw
def _fmt(piece: dict) -> str:
    f = str(piece.get("format") or "9:16")
    return "16:9" if f == "16:9" else "9:16"


def _protected(piece: dict) -> set:
    """words that are windows (their face stays: window_source / window_ground on the word, or a window cut that names the word)."""
    try:
        from tools.fusion import qwen_piece as QP
        return set(QP.window_words(piece))
    except Exception:       # noqa: BLE001
        t = (piece.get("type") or {}).get("words") or []
        return {i for i, w in enumerate(t) if w.get("window_source") not in (None, "", "self") or w.get("window_ground")}


def assign_fonts(words: list, system: dict, seed: int, protected=()) -> list:
    """one shortlist face per word (in place; -> the files). Seeded; never the face of the neighbour word; the least used face first so the whole shortlist appears; a face that cannot carry the
    word at the legibility floor is skipped (random_system._legible_at); a word that is long takes the face with the narrowest letters of the candidates (the bigger cap). A face of a window
    word is kept. An Anybody pick gets its default (thin) instance as axes; a changed face drops the old axes (they belonged to the old face)."""
    from mograph import fonts_liked as fl
    from tools.fusion import design_options as DO
    short = [f for f in (system.get("fonts") or []) if fl.lookup(f)]
    if not short:
        return [w.get("font") for w in words]
    used = {f: 0 for f in short}
    out = []
    for i, w in enumerate(words):
        if i in protected or not isinstance(w, dict):
            out.append(w.get("font") if isinstance(w, dict) else None)
            continue
        text = str(w.get("text") or "")
        rng = random.Random(f"lfont:{int(seed)}:{i}:{text}")
        prev = out[-1] if out else None
        cand = [f for f in short if f != prev and RS._legible_at(fl.lookup(f), [text])] or [f for f in short if f != prev] or short
        low = min(used[f] for f in cand)
        cand = [f for f in cand if used[f] == low] or cand
        wts = []
        for f in cand:
            r = DO.line_width_per_cap(f, None, "".join(ch for ch in text if ch.isalnum()) or "A")
            wts.append(1.0 if r is None else (1.0 / max(r, 0.5)) ** (1.0 if len(text) >= 7 else 0.3))        # long words lean to narrow faces, short ones are free
        pick = PS._wpick(rng, cand, wts)
        used[pick] += 1
        if w.get("font") != pick:
            w.pop("axes", None)
        w["font"] = pick
        if pick == PS.ANYBODY:
            w["axes"] = PS.default_axes(PS.ANYBODY)
        out.append(pick)
    return out


def _reset_split(w: dict) -> None:
    """a word whose old `lines` split it MID-WORD (grid / column) goes back to plain text so the new draw lays it out freely; space breaks of a phrase stay as rows."""
    text = str(w.get("text") or "")
    lines = w.get("lines")
    if not lines:
        return
    toks = [ST._letters(t) for t in text.split() if ST._letters(t)]
    rows = [ST._letters(x) for x in lines if ST._letters(x)]
    if rows != toks:
        w.pop("lines", None)
    w.pop("ar_min", None)


def layout_of(w: dict) -> str:
    """the layout name of a drawn word: line | stack | rot +90 | rot -90 | grid | column (from its roll and `lines`)."""
    roll = float(((w.get("placement") or {}).get("roll")) or 0.0)
    text = str(w.get("text") or "")
    toks = [ST._letters(t) for t in text.split() if ST._letters(t)]
    rows = [ST._letters(x) for x in (w.get("lines") or []) if ST._letters(x)]
    if abs(abs(roll) - 90.0) < 1.0:
        return LAYOUT_LABEL["rot90"] if roll > 0 else LAYOUT_LABEL["rot-90"]
    if rows and rows != toks:
        return LAYOUT_LABEL["column"] if all(len(r) == 1 for r in rows) else LAYOUT_LABEL["grid"]
    if len(rows) > 1:
        return LAYOUT_LABEL["stack"]
    return LAYOUT_LABEL["line"]


def font_label(f) -> str:
    s = os.path.splitext(str(f or ""))[0]
    return re.sub(r"\[.*?\]", "", s).replace("_", " ")


def _remap_cuts(piece: dict, system: dict, fixes: list) -> int:
    """compose.cuts whose kind is not one of the DRAWN kinds take a drawn 3D kind (a window / pan kind needs a fitting word and can fail: only the robust kinds are put at a boundary
    in a re-drawn piece), the style of the draw, the old parameters / why dropped. -> number remapped."""
    from tools.fusion import cut_kinds as CK
    comp = piece.get("compose") or {}
    cuts = comp.get("cuts")
    kinds = RS.kinds_of(system)
    if not isinstance(cuts, list) or not kinds:
        return 0
    n, used = 0, []
    for c in cuts:
        if not isinstance(c, dict):
            continue
        k = CK.normalize(c.get("cut")) if c.get("cut") else None
        if k in kinds:
            used.append(k)
            continue
        new = RS.pick_next(system, used, feasible=RS.robust) or RS.pick_next(system, used)
        if not new:
            continue
        fixes.append(f"compose.cuts after {c.get('after')}: {c.get('cut')} -> {new} (the layout seed draws {', '.join(kinds)})")
        for key in ("params", "why", "word", "style", "dir", "mode"):
            c.pop(key, None)
        c["cut"] = new
        st = RS.style_for(system, new)
        if st:
            c["style"] = st
        c["why"] = f"re-drawn by layout seed {system.get('seed')}"
        used.append(new)
        n += 1
    return n


def redraw_piece(piece: dict, seed: int, stills: dict | None = None, effects=None, size=None) -> tuple[dict, dict]:
    """the piece with its random system DRAWN AGAIN by `seed` on the same words / timings / shots. -> (new piece, info). `stills` {k: RGB uint8} (the first one gives the palette; none = the
    system's palette stays). `effects` = named effects forced in (effects_menu). Deterministic in (piece, seed, stills, effects)."""
    from tools.fusion import effects_menu as EM
    from tools.fusion import hold_life as HL
    from tools.fusion import plan_rules as PR
    seed = int(seed)
    p = copy.deepcopy(piece)
    t = p.setdefault("type", {})
    words = t.get("words") or []
    texts = [str(w.get("text") or "") for w in words if isinstance(w, dict)]
    pl = p.setdefault("plan", {})
    audio = p.get("audio") or {}
    bpm = t.get("bpm") or audio.get("bpm") or 100
    length = audio.get("cut") or t.get("duration")
    eff = effects_list(effects)
    pal = None
    if stills:
        first = stills.get(1) if stills.get(1) is not None else next(iter(stills.values()))
        pal = still_palette_for(first, seed)
        pl["still_palette"] = copy.deepcopy(pal)
    base = pl.get("piece_system") if isinstance(pl.get("piece_system"), dict) else None
    system = with_layout_seed(base, seed, texts, length, eff, pal, mood=str(t.get("mood") or ""), bpm=bpm)
    pl["piece_system"] = copy.deepcopy(system)
    pl["layout_seed"] = seed
    fixes: list = []
    # ---- the type: palette from the still, a face per word, the layout per word (centred, still, no-orphan), the held camera
    if pal:
        t["palette"] = dict(SP.type_palette(pal))
    prot = _protected(p)
    for i, w in enumerate(words):
        if isinstance(w, dict) and i not in prot:
            _reset_split(w)
    assign_fonts(words, system, seed, prot)
    ST.enforce(t, system, bpm, fixes, protected=prot, seed=seed, fmt=_fmt(p))
    # ---- effects as the planner records them (plan_rules._build), outline + hold life, cut kinds
    ef = system.get("effects") or {}
    pl["effects"] = dict(placed=(pl.get("effects") or {}).get("placed", []), active=list(ef.get("active") or []), requested=list(ef.get("requested") or []), unknown=list(ef.get("unknown") or []))
    comp = p.setdefault("compose", {})
    comp.pop("outline", None)                                                    # recomputed from the new system (recipe_eval._outline_windows / hold_life._outline_of)
    hl_req = [n for n in (ef.get("requested") or []) if n in EM.HOLD_LIFE_EFFECTS]
    if hl_req:
        comp["hold_life"] = hl_req
    elif isinstance(comp.get("hold_life"), list):
        comp.pop("hold_life")
    n_cuts = _remap_cuts(p, system, fixes)
    holds = [(float(r["hit"]), float(r["rel"])) for r in word_timing(p)]
    try:
        HL.weave(p, None, None, size or (p.get("sizes") or {}).get("compose") or (540, 960), holds=holds)
    except Exception as e:      # noqa: BLE001  the hook must not break the preview: report the failure, the other choices stand
        pl["hold_life"] = dict(effects=[], error=f"{type(e).__name__}: {e}")
    from tools.fusion import recipe_eval as RE
    outline_w = [list(w) for w in RE._outline_windows(p, holds)]                # exactly the windows the render would force (the drawn ones, none | hits | the requested `outline`), nothing added
    info = dict(version=VERSION, seed=seed, fixes=fixes, holds=[[round(a, 3), round(b, 3)] for a, b in holds], outline_windows=outline_w, cuts_remapped=n_cuts,
                summary=system_summary(p, system, outline_w))
    pl["layout_preview"] = dict(version=VERSION, seed=seed, effects=eff)
    return p, info


def system_summary(piece: dict, system: dict, outline_w=None) -> dict:
    """the drawn system in a few fields (the PreviewAny text and the header of the sheet)."""
    words = (piece.get("type") or {}).get("words") or []
    pal = (system.get("palette") or {})
    hl = ((piece.get("plan") or {}).get("hold_life") or {})
    cuts = system.get("cuts") or {}
    return dict(seed=system.get("seed"), fonts=[font_label(f) for f in system.get("fonts") or []],
                cuts=[(k + (f":{cuts['styles'][k]}" if (cuts.get("styles") or {}).get(k) else "")) for k in cuts.get("kinds") or []],
                outline=(system.get("outline") or {}).get("mode", "none"), outline_windows=outline_w or [], hold_life=list(hl.get("effects") or []), effects=list((system.get("effects") or {}).get("active") or []),
                palette=dict(source=pal.get("source"), colors=pal.get("colors"), bg=pal.get("bg"), fg=pal.get("fg"), accents=pal.get("accents")),
                words=[dict(text=w.get("text"), layout=layout_of(w), lines=w.get("lines"), font=font_label(w.get("font"))) for w in words if isinstance(w, dict)])


# ---------------------------------------------------------------------------------------------------------------------------------------------- the type frames (CPU subprocesses)
def type_spec(piece: dict) -> dict:
    """the type spec handed to type_passes (as nodes_fusion/type.py build_spec without a music dict): piece['type'] + fps / duration / bar."""
    spec = copy.deepcopy(piece["type"])
    spec["fps"] = int(piece.get("fps", FPS))
    if "duration" not in spec:
        lay = piece.get("layout") or {}
        cut = (piece.get("audio") or {}).get("cut")
        spec["duration"] = float(cut) if cut else max((v["start"] + v["frames"] for v in lay.values()), default=0) / spec["fps"]
    spec.setdefault("bar", 4)
    return spec


def word_timing(piece: dict) -> list:
    """the engine's own timing of every word (spec.plan_timing on the beat grid, as plan_rules.word_holds): [{S, hit, rel, end, ...}] piece seconds. S = where the word's module starts."""
    from tools.fusion import plan_rules as PR
    t = piece["type"]
    grid = PR.make_grid({"bpm": t["bpm"], "beats": t["beats"], "downbeats": t["downbeats"]}, float(t["duration"]))
    return PR._spec().plan_timing(copy.deepcopy(t["words"]), t, grid)


def isolate_word(spec: dict, i: int, S: float, end: float, tail: float = 0.8) -> dict:
    """a type spec holding ONLY word i, moved so that its module starts at 0 (the beat grid shifted by S, which keeps every snap of plan_timing: S is where the word starts on the full
    timeline). The camera plan of a scene walks the whole timeline (a 30 s piece of six `circuit` words takes 90 s to build); one word of ~5 s builds in a few seconds. Spec keys that name
    a word by index follow it (`camera_override` word:i -> word:0); a dict camera (multiplane chain) or a pan transition cannot be cut out: the full spec comes back (index 0 mapping kept by the caller)."""
    if not isinstance(spec.get("camera"), (str, type(None))) or any((w.get("transition") or {}).get("kind") == "pan" for w in spec["words"] if isinstance(w, dict)):
        return spec
    sp = copy.deepcopy(spec)
    w = copy.deepcopy(sp["words"][i])
    t0 = float(w.get("t0", S))
    t1 = float(w.get("t1", t0 + 2.0))
    w["t0"], w["t1"] = 0.0, round(max(t1 - t0, 1.0), 4)
    w.pop("transition", None)
    w.pop("slot", None)
    sp["words"] = [w]
    sp["transitions"] = []
    sp["beats"] = [round(float(b) - S, 6) for b in sp.get("beats") or [] if float(b) - S >= -1e-6]
    sp["downbeats"] = [round(float(d) - S, 6) for d in sp.get("downbeats") or [] if float(d) - S >= -1e-6]
    sp["duration"] = round(float(end) - S + tail, 3)
    co = sp.get("camera_override")
    if isinstance(co, dict):
        sp["camera_override"] = {"word:0": v for k, v in co.items() if k == f"word:{i}"}
    return sp


_FOCUS_HZ = [6.0]


def _coarse_focus_init(self, seq):
    """FocusLook.__init__ at `_FOCUS_HZ` instead of 60 Hz (module-level so the scene pickles on every Python: a locally defined subclass failed to pickle on the box's Python 3.12)."""
    self.seq = seq
    self.ts = np.arange(0.0, seq.duration + seq.tail + 0.5, 1.0 / _FOCUS_HZ[0])
    self.xs = np.array([seq.focus(float(t)) for t in self.ts])


def _coarse_focus(hz: float = 6.0) -> None:
    """the camera plan tabulates the sequence's focus point at 60 Hz (mograph.modules.sequence.FocusLook), one `layers()` call each: 90 s on a 30 s piece of a pixel face. The preview camera is the
    held beat_orbit (it frames the word boxes and never follows the focus), so the table is taken at `hz` instead: the same frames, a tenth of the build time."""
    from mograph.modules import sequence as MS
    _FOCUS_HZ[0] = float(hz)
    if MS.FocusLook.__init__ is not _coarse_focus_init:
        MS.FocusLook.__init__ = _coarse_focus_init


def _worker(tasks_path: str, out: str, size: tuple, ids: list) -> None:
    """render the beauty + matte pass of tasks `ids` (tasks.json: [{spec, frame}]; one scene build per task, S = 2) into out/<id>/beauty, out/<id>/matte; the body of a subprocess."""
    import pickle
    from tools.fusion import spec as S_
    from tools.fusion import type_passes as TP
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    _coarse_focus()
    for k in ids:
        scene = S_.build_scene(tasks[k]["spec"], tuple(size))
        scene.camera = scene.chain if scene.chain is not None else TP._fixed(scene)
        ww, wh = scene.world
        scene.ground_y = max(float(m.placement.origin[1] + m.placement.extent_along(np.array([0.0, 1.0, 0.0]))) for m in scene.seq.words)
        scene.pivot = (ww / 2.0, wh / 2.0, 0.0)
        sp = scene.spec
        d = f"{out}/{k}"
        for sub in ("beauty", "matte"):
            os.makedirs(f"{d}/{sub}", exist_ok=True)
        job = dict(scene=scene, out=d, passes=["beauty", "matte"], S=2, near=500.0, far=1500.0, depth_mode="linear", line_px=1.5, matte_soft=float((sp.get("matte") or {}).get("soft", 0.6)),
                   clay={}, flow_mode="hybrid", flow_bg="camera")
        TP._init(pickle.dumps(job))
        f = int(min(max(int(tasks[k]["frame"]), 0), scene.nf - 1))
        TP._frame(f)
        (Path(d) / "frame.txt").write_text(str(f), encoding="utf-8")


def render_frames(tasks: list, out: str, size: tuple, procs: int | None = None, timeout: float | None = None) -> dict:
    """tasks = [{spec, frame}]: beauty + matte PNG of `frame` of each spec in out/<task index>/beauty|matte (frame.txt = the frame actually written). procs 0 = in this process (no deadline);
    else ONE subprocess per task, at most `procs` at once (OMP threads 1, no CUDA), each with a deadline of `timeout` s (TASK_TIMEOUT_S): a task that fails or runs out of time is killed and
    reported in `failed` (the caller draws that word flat). -> {wall_s, procs, failed: [task index], timeouts: [task index]}."""
    t0 = time.time()
    out = str(out)
    os.makedirs(out, exist_ok=True)
    tp = os.path.join(out, "tasks.json")
    Path(tp).write_text(json.dumps(tasks), encoding="utf-8")
    n_proc = MAX_PROCS if procs is None else int(procs)
    n_proc = min(n_proc, len(tasks), os.cpu_count() or 1)
    if n_proc <= 0:
        _worker(tp, out, size, list(range(len(tasks))))
        return dict(wall_s=round(time.time() - t0, 2), procs=0, failed=[], timeouts=[])
    limit = TASK_TIMEOUT_S if timeout is None else float(timeout)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    pending, running, failed, timeouts = list(range(len(tasks))), {}, [], []
    while pending or running:
        while pending and len(running) < n_proc:
            k = pending.pop(0)
            lp = os.path.join(out, f"worker{k}.log")
            log = open(lp, "w", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", tp, "--out", out, "--size", f"{size[0]}x{size[1]}", "--frames", str(k)]
            running[k] = (subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), time.time(), log)
        for k, (pr, t_start, log) in list(running.items()):
            rc = pr.poll()
            if rc is None and time.time() - t_start > limit:
                pr.kill()
                pr.wait()
                timeouts.append(k)
                rc = -9
            if rc is not None:
                log.close()
                running.pop(k)
                if rc != 0:
                    failed.append(k)
        time.sleep(0.03)
    return dict(wall_s=round(time.time() - t0, 2), procs=n_proc, failed=sorted(failed), timeouts=sorted(timeouts))


# ---------------------------------------------------------------------------------------------------------------------------------------------- the flat fallback
LAYOUT_KEY = {"line": "line", "stack": "stack", "rot +90": "rot90", "rot -90": "rot-90", "grid": "grid", "column": "column"}


def _set_axes(font, axes):
    """variable-font axes of a PIL font (wdth / wght ...), as the engine and explore_cuts2._font set them; a font without axes is left alone."""
    if not axes:
        return font
    try:
        names = [a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"]) for a in font.get_variation_axes()]
        tags = {"Width": "wdth", "Weight": "wght"}
        font.set_variation_by_axes([float(axes.get(tags.get(nm, nm.lower()[:4]), a.get("default", a["minimum"]))) for a, nm in zip(font.get_variation_axes(), names)])
    except Exception:   # noqa: BLE001
        pass
    return font


def flat_frame(word: dict, fg=(255, 255, 255), size=TYPE_SIZE, letters: bool = False) -> tuple:
    """a word drawn FLAT with PIL in its own font file, in the layout the draw gave it (stable_type.fit_box: rows, cap height, +-90 roll), centred. The preview's fallback for a word whose engine
    scene is too slow to build in the time of a Randomise (or failed): the layout, face, size and position are the real ones, the module's own shapes are not. -> (face RGB uint8 HxWx3, alpha uint8 HxW)."""
    from PIL import Image, ImageDraw, ImageFont
    from mograph import font_registry as fr
    from tools.fusion import design_options as DO
    W, H = size
    text = " ".join(str(word.get("text") or "").split())
    key = LAYOUT_KEY.get(layout_of(word), "line")
    box = ST.fit_box(text, word.get("font"), word.get("axes"), key, W / float(H))
    lines = [str(x) for x in (box.get("lines") or [text])]
    cap_px = max(float(box.get("cap") or 0.1) * H, 8.0)
    from mograph import fonts_liked as _fl
    path = _fl.font_file(word.get("font"))                                  # the liked face or FontNotLiked: no generic fallback (user 2026-10-30)
    f200 = _set_axes(ImageFont.truetype(path, 200), word.get("axes"))
    bb = f200.getbbox("H")
    size_px = max(int(round(200.0 * cap_px / max(bb[3] - bb[1], 1))), 6)
    font = _set_axes(ImageFont.truetype(path, size_px), word.get("axes"))
    hb = font.getbbox("H")
    cap_h = hb[3] - hb[1]
    split = key in ("grid", "column")
    pitch = cap_px * (ST.LEAD if split else (1.12 if len(lines) > 1 else 1.0))
    widths = [font.getlength(x) for x in lines]
    cw, ch = int(max(widths) + 8), int(pitch * (len(lines) - 1) + cap_h + 8 + size_px * 0.4)
    canvas = Image.new("L", (max(cw, 8), max(ch, 8)), 0)
    d = ImageDraw.Draw(canvas)
    for r, (ln, wd) in enumerate(zip(lines, widths)):
        d.text(((cw - wd) / 2.0, 4 + cap_h + r * pitch), ln, font=font, fill=255, anchor="ls")
    roll = float((word.get("placement") or {}).get("roll") or 0.0)

    def place(cv):
        """the canvas as the stage lands it: rolled +-90, shrunk into 94 % of the frame, pasted centred -> uint8 HxW."""
        if abs(abs(roll) - 90.0) < 1.0:
            cv = cv.rotate(-roll, expand=True)
        k = min(0.94 * W / max(cv.width, 1), 0.94 * H / max(cv.height, 1), 1.0)       # a column of tall-ascender letters (QC v026 / Alien Block: TIDE) must stay inside the frame
        if k < 1.0:
            cv = cv.resize((max(int(round(cv.width * k)), 8), max(int(round(cv.height * k)), 8)), Image.LANCZOS)
        al = Image.new("L", (W, H), 0)
        al.paste(cv, ((W - cv.width) // 2, (H - cv.height) // 2), cv)
        return np.asarray(al, np.uint8)

    a = place(canvas)
    face = np.empty((H, W, 3), np.uint8)
    face[:] = np.asarray(fg, np.uint8)[:3]
    if not letters:
        return face, a
    lid = np.full((H, W), -1, np.int16)                       # letters=True: + the letter-id map (reading order of the drawn glyphs, rows top to bottom), for the cuts that form a word letter by letter
    idx = 0
    for r, (ln, wd) in enumerate(zip(lines, widths)):
        for k, glyph in enumerate(ln):
            if glyph == " ":
                continue
            cv = Image.new("L", canvas.size, 0)
            ImageDraw.Draw(cv).text(((cw - wd) / 2.0 + font.getlength(ln[:k]), 4 + cap_h + r * pitch), glyph, font=font, fill=255, anchor="ls")
            m = place(cv)
            lid[(m > 127) & (lid < 0)] = idx
            idx += 1
    miss = (a > 127) & (lid < 0)                              # inked pixels no single glyph drawing claimed (kerning of the whole line vs the glyph-by-glyph advance): the nearest letter owns them
    if miss.any() and (lid >= 0).any():
        src = lid >= 0
        ys, xs = np.nonzero(src)
        _, lab0 = cv2.distanceTransformWithLabels((~src).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)      # label = the nearest letter pixel (a zero pixel of the input)
        table = np.full(int(lab0.max()) + 1, -1, np.int16)
        table[lab0[ys, xs]] = lid[ys, xs]
        lid[miss] = table[lab0[miss]]
    return face, a, lid


# ---------------------------------------------------------------------------------------------------------------------------------------------- tiles + sheet
def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(im[y0:y0 + h, x0:x0 + w])


def sample_times(piece: dict, holds: list, bpm) -> list:
    """the frame time of every word: its landed hold, half a beat after the hit (inside a one-beat outline pulse, past the first read frames), never past the release."""
    beat = 60.0 / float(bpm) if bpm else 0.6
    out = []
    for hit, rel in holds:
        t = hit + min(0.5 * beat, max(0.5 * (rel - hit), 0.0))
        out.append(min(max(t, hit + 3.0 / FPS), max(rel - 1.0 / FPS, hit)))
    return out


def render_tiles(piece: dict, stills: dict, holds: list, outline_w: list, work: str, procs: int | None = None, tile=(TILE_W, TILE_H), type_size=TYPE_SIZE) -> tuple[list, dict]:
    """one composed tile per word. -> (tiles [RGB uint8 HxWx3], timing dict). The type frame of word i at its landed hold over the still of ITS shot, legible_color (single frame, no smoothing, the
    drawn outline windows forced) between them. 2026-10-30 (user: "we don't need to outline them all"): the tile shows what the render does: an outline ONLY on the words the random system drew (`outline_w`)
    or, as a last resort where even the extreme ink fails, within the piece-wide ceil(n / 3) cap (ONE colour-stage state and word list for all tiles, so the cap counts across tiles); the palette inks are tried first."""
    from tools.fusion import legible_color as LC
    from tools.fusion import recipe_eval as RE
    t = piece.get("type") or {}
    words = t.get("words") or []
    spec = type_spec(piece)
    rows = word_timing(piece)
    times = sample_times(piece, holds, t.get("bpm") or (piece.get("audio") or {}).get("bpm"))
    tasks, idx = [], []
    for i, tt in enumerate(times):
        r = rows[i]
        sp_i = isolate_word(spec, i, float(r["S"]), float(r["end"]))
        iso = sp_i is not spec
        f = int(round((tt - (float(r["S"]) if iso else 0.0)) * FPS))
        tasks.append(dict(spec=sp_i, frame=f))
        idx.append(f)
    tm = {}
    t0 = time.time()
    tm["render"] = render_frames(tasks, work, type_size, procs)
    tiles, flat, color_errors = [], [], []
    bad = set(tm["render"].get("failed") or [])
    lc_rows = [[float(w.get("t0", h_)), float(h_), float(r_), float(w.get("t1", r_)), False] for w, (h_, r_) in zip(words, holds)]
    ink_pal = RE._ink_palette(piece)
    key = f"lp:{(piece.get('plan') or {}).get('layout_seed')}"
    LC.reset(key)                                                               # one state key: the outline cap (ceil(n / 3)) counts across the tiles; every call below starts a fresh track (frame 0)
    t1 = time.time()
    tw, th = tile
    for i, (w, f) in enumerate(zip(words, idx)):
        beauty = matte = None
        if i not in bad:
            f = int(Path(f"{work}/{i}/frame.txt").read_text(encoding="utf-8")) if Path(f"{work}/{i}/frame.txt").is_file() else f
            beauty = cv2.imread(f"{work}/{i}/beauty/beauty_{f:05d}.png", cv2.IMREAD_COLOR)
            matte = cv2.imread(f"{work}/{i}/matte/matte_{f:05d}.png", cv2.IMREAD_GRAYSCALE)
        if beauty is None or matte is None:                                    # too slow / failed: the word flat in its own face and layout
            fg = ((t.get("palette") or {}).get("fg")) or (255, 255, 255)
            fb_, ma = flat_frame(w, tuple(int(v) for v in fg), type_size)
            beauty, matte = cv2.cvtColor(fb_, cv2.COLOR_RGB2BGR), ma
            flat.append(i)
        face = cv2.resize(cv2.cvtColor(beauty, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        alpha = cv2.resize(matte, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        _k, still = still_for_time(piece, times[i], stills)
        plate = (_cover(still, tw, th).astype(np.float32) / 255.0) if still is not None else np.full((th, tw, 3), 0.08, np.float32)
        try:
            LC._SHOT_PAL.pop(key, None)                                          # 2026-11-01: each tile takes the hues of ITS still (one colour state: the outline cap counts across tiles)
            rgba = LC.legible_color(face, alpha, plate, t=times[i], frame_idx=0, u=0.5, fps=FPS, words=lc_rows, state_key=key, mode="layer", smooth=0.0, outline=outline_w,
                                    shot_palette=(piece.get("compose") or {}).get("shot_colour") is not False, piece_key=f"{key}:prev", shot_t0=float(times[i]),   # 2026-11-01: the shot's own hues, like compose
                                    **({"user_palette": [[v / 255.0 for v in c] for c in RE.user_style(piece)["palette"]]} if RE.user_style(piece)["palette"] else {}),
                                    **({"ink_palette": ink_pal} if ink_pal else {}), **({"soft_support": True} if (piece.get("compose") or {}).get("soft_support") else {}))
            rgb, a = np.asarray(rgba)[..., :3], np.asarray(rgba)[..., 3:4]
        except Exception as e:      # noqa: BLE001  the colour stage must not stop the preview: the drawn face over the plate (reported in the timing)
            rgb, a = face, alpha[..., None]
            color_errors.append(f"word {i}: {type(e).__name__}: {e}")
        comp = plate * (1.0 - a) + rgb * a
        tiles.append(np.clip(np.rint(comp * 255.0), 0, 255).astype(np.uint8))
    tm["flat"] = flat
    tm["color_errors"] = color_errors
    tm["compose_s"] = round(time.time() - t1, 2)
    tm["total_s"] = round(time.time() - t0, 2)
    return tiles, tm


def _put(img, text, x, y, scale=0.45, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)      # a 1 px shadow (a thicker halo has a wider advance in OpenCV: a ghost tail)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def contact_sheet(tiles: list, piece: dict, summary: dict, holds: list, seed: int, cols: int | None = None) -> np.ndarray:
    """RGB uint8 sheet: a header band (seed, cut kinds, outline, hold life, palette swatches), the tiles in rows, and under each tile the word, its layout and its font."""
    words = (piece.get("type") or {}).get("words") or []
    n = len(tiles)
    cols = cols or max(1, math.ceil(n / math.ceil(n / 4.0)))
    rows = math.ceil(n / cols)
    th, tw = tiles[0].shape[:2]
    cap_h, pad, head_h = 52, 8, 78
    W = cols * tw + (cols + 1) * pad
    H = head_h + rows * (th + cap_h) + (rows + 1) * pad
    sheet = np.full((H, W, 3), 22, np.uint8)
    _put(sheet, f"layout seed {seed}", pad + 2, 26, 0.8, (255, 255, 255), 2)
    _put(sheet, "cuts: " + (", ".join(summary.get("cuts") or []) or "-"), pad + 2, 48, 0.45)
    _put(sheet, f"outline: {summary.get('outline')}   hold life: {', '.join(summary.get('hold_life') or []) or '-'}   fonts: {', '.join(summary.get('fonts') or [])}"[:int(W / 7.2)], pad + 2, 68, 0.45)
    cols_pal = (summary.get("palette") or {}).get("colors") or []
    for j, c in enumerate(cols_pal[:5]):
        x0 = W - pad - (j + 1) * 36
        cv2.rectangle(sheet, (x0, 10), (x0 + 30, 40), tuple(int(v) for v in c), -1)
        cv2.rectangle(sheet, (x0, 10), (x0 + 30, 40), (200, 200, 200), 1)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = head_h + pad + r * (th + cap_h + pad)
        sheet[y:y + th, x:x + tw] = tile
        w = words[i] if i < len(words) else {}
        _put(sheet, f"{i + 1}. {w.get('text', '')}", x + 2, y + th + 17, 0.55, (255, 255, 255), 1)
        _put(sheet, (f"{layout_of(w)}   hold {holds[i][0]:.1f}-{holds[i][1]:.1f}s" if i < len(holds) else layout_of(w)) + ("   (flat)" if i in (summary.get("flat_words") or []) else ""), x + 2, y + th + 33, 0.42)
        _put(sheet, font_label(w.get("font")), x + 2, y + th + 48, 0.42, (170, 220, 255))
    return sheet


# ---------------------------------------------------------------------------------------------------------------------------------------------- files
def unique_base(folder, stem: str, seed: int, ext=(".png", ".json")) -> str:
    """`<stem>_layout_s<seed>`, then `..._b`, `..._c` ... until none of the extensions exists in `folder` (a render is never overwritten)."""
    folder = Path(folder)
    base = f"{stem}_layout_s{int(seed)}"
    cand, k = base, 0
    while any((folder / (cand + e)).exists() for e in ext):
        k += 1
        cand = f"{base}_{chr(ord('a') + k)}" if k < 26 else f"{base}_{k}"
    return cand


def preview(piece: dict, stills: dict, seed: int, effects=None, out_dir=None, stem: str = "piece", procs: int | None = None, work: str | None = None, size=None) -> dict:
    """redraw + render + sheet. -> dict(sheet (RGB uint8), piece, summary, info, tiles, files {png, json}, timing). Files only when `out_dir` is given: `<stem>_layout_s<seed>.png` and `.json`
    (the drawn system, the words with layout / font, the fixes) next to the archive, never overwriting."""
    import shutil
    import tempfile
    if not ((piece.get("type") or {}).get("words") or []):
        raise ValueError("the piece has no words: nothing to preview")
    t0 = time.time()
    newp, info = redraw_piece(piece, seed, stills, effects, size)
    t1 = time.time()
    holds = [tuple(h) for h in info["holds"]]
    wd = work or tempfile.mkdtemp(prefix=f"lp_{int(seed)}_")
    try:
        tiles, tm = render_tiles(newp, stills, holds, info["outline_windows"], wd, procs)
    finally:
        if not work:                                                    # our own scratch folder of type frames: gone after the sheet
            shutil.rmtree(wd, ignore_errors=True)
    info["summary"]["flat_words"] = list(tm.get("flat") or [])                 # words drawn flat (engine scene too slow / failed)
    sheet = contact_sheet(tiles, newp, info["summary"], holds, seed)
    files = {}
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = unique_base(out_dir, stem, seed)
        cv2.imwrite(str(out_dir / (base + ".png")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        (out_dir / (base + ".json")).write_text(json.dumps(dict(seed=int(seed), effects=list(effects) if isinstance(effects, (list, tuple)) else effects, summary=info["summary"], fixes=info["fixes"],
                                                                holds=info["holds"], piece=newp), indent=1, default=str), encoding="utf-8")
        files = dict(png=str(out_dir / (base + ".png")), json=str(out_dir / (base + ".json")), base=base)
    return dict(sheet=sheet, piece=newp, summary=info["summary"], info=info, tiles=tiles, files=files,
                timing=dict(redraw_s=round(t1 - t0, 2), render=tm.get("render"), compose_s=tm.get("compose_s"), tiles_s=tm.get("total_s"), flat=tm.get("flat"), color_errors=tm.get("color_errors"), total_s=round(time.time() - t0, 2)))


# ---------------------------------------------------------------------------------------------------------------------------------------------- the lock ("Use this layout")
def lock_path(folder, stem: str) -> Path:
    return Path(folder) / f"{stem}{LOCK_SUFFIX}"


def write_lock(folder, stem: str, seed: int, effects=None) -> Path:
    p = lock_path(folder, stem)
    p.write_text(json.dumps(dict(version=VERSION, stem=stem, layout_seed=int(seed), effects=list(effects or [])), indent=1), encoding="utf-8")
    return p


def read_lock(folder, stem: str) -> dict | None:
    p = lock_path(folder, stem)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return d if layout_seed_of(d.get("layout_seed")) else None


def apply_lock(piece: dict, lock: dict, stills: dict | None = None) -> dict:
    """the piece with a saved lock applied: the layout seed's draw (what the user saw on the sheet) on the same words. Used by tools/app/upgrade_still.py before the upgrade graph is built."""
    newp, _info = redraw_piece(piece, int(lock["layout_seed"]), stills, lock.get("effects"))
    return newp


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="layout_preview: --worker (internal) or a preview of an archive folder")
    ap.add_argument("--worker")
    ap.add_argument("--out")
    ap.add_argument("--size", default="576x1024")
    ap.add_argument("--frames", default="")
    ap.add_argument("--archive")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--effects", default="")
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args(argv)
    if a.worker:
        w, h = (int(v) for v in a.size.split("x"))
        _worker(a.worker, a.out, (w, h), [int(f) for f in a.frames.split(",") if f != ""])
        return
    arc = load_archive(a.archive)
    r = preview(arc["piece"], arc["stills"], a.seed, [e for e in a.effects.split(",") if e], out_dir=a.out or a.archive, stem=arc["stem"], procs=a.procs)
    print(json.dumps(dict(files=r["files"], timing=r["timing"]), indent=1))


if __name__ == "__main__":
    main()
