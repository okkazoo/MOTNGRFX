"""design_step: the open-model DESIGN step of the app graph, as pure functions (no ComfyUI imports; the nodes are in nodes_fusion/design.py).

    package(request, music, piece, clips, mattes)  -> prompt + contact / matte sheets + locked fields        (MographDesignPackage)
    parse(reply, locked, piece, music, prompt)     -> piece_json, ok, repair_prompt, report                  (MographDesignParse)
    repair(piece_json, ok, repair_reply, piece)    -> piece_json (always valid), report                      (MographDesignRepair)

NO CLAUDE: the model is design_llm_8b (DesignLLM_Advanced in the graph); everything else is code. The design logic is
``tools/fusion/qwen_piece.py`` (prompt, schema, extract_json, normalize_design with its clamps / liked fonts, the timing compiler and the
deterministic timing fit); this module only feeds it an IN-GRAPH piece instead of a P*.json on disk (``qwen_piece._CTX``) and adds the part
earlier hosted-model experiments did not have: a reply that is wrong in whole sections must still give a valid piece.

Sections, resolved one by one (``resolve``):
  type     model design (all checks pass)  >  model's mood / palette / camera / transitions + the rules' words (only the words were
           wrong)  >  the rules' type (reply has no `type` at all, or nothing in it works)
  compose  model design (built from the repo's recipe for its technique, restricted to techniques the generated footage supports)
           >  the rules' compose
FIELD-LEVEL (v002, ``design_salvage.py``): the reply is read field by field. A valid field is the model's as written; an invalid one is replaced ALONE (the nearest
allowed value when there is an obvious one, else the rules' value for that field of that word); only a section the reply lacks entirely comes from the rules.
The piece-level rules (>= 3 liked faces incl. one expressive, consecutive-line contrast, the timing fit to the cut, window-cut eligibility) then change the fewest
fields that make the piece comply and log each change; every design reports fields_from_model / fields_fixed / fields_from_rules with the field paths
(``plan.design.fields`` of the piece, ``parse:`` report text). The repair call gets only the fields that stay invalid and its reply is merged field by field.

The rules are ``plan_rules.plan_piece`` = the `piece` input (already planned, dry-run, tail-filled, flow_wake woven). Locked and never the
model's: audio, fps, stills, shots, type_passes, sizes, word texts and order, type.seed, compose size / crossfade / crf. The caption is off.
After the sections are chosen the matte treatments and flow_wake are woven with the repo's own weavers, the result is dry-run on the real
type scene, and any failure there returns the rules' piece unchanged: the output is ALWAYS a valid piece.
"""
from __future__ import annotations

import contextlib
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(ROOT.as_posix())

from tools.fusion import compose_blocks as CB    # noqa: E402
from tools.fusion import design_salvage as SV   # noqa: E402
from tools.fusion import owner as OW             # noqa: E402
from tools.fusion import plan_rules as PR        # noqa: E402
from tools.fusion import qwen_piece as QP        # noqa: E402
from tools.fusion import recipe_eval as RE       # noqa: E402

APP = "_app"                       # the name the in-graph piece has inside qwen_piece (PIECES / _CTX)
GUIDES_DIR = ROOT / "16_app"
DEFAULT_GUIDE = "DESIGN_GUIDE_open_model"
LOCKED_KEYS = ("audio", "fps", "stills", "shots", "type_passes", "sizes", "quality", "format", "slug", "id", "still_preview", "request")     # still_preview / request: the still-preview flag and the app inputs ride on the designed piece (tools/fusion/still_preview.py)
SHEET_TILE_W = 160                 # contact-sheet tile width (preview footage is 384 wide)
MATTE_TILE_W = 120
SHEET_COLS = 2                     # sheets per row of the one combined image


# --------------------------------------------------------------------------------------------- guides, music, locked
def guide_options() -> list[str]:
    """design guides in 16_app (DESIGN_GUIDE*.md), stems; the open-model guide first."""
    names = sorted(p.stem for p in GUIDES_DIR.glob("DESIGN_GUIDE*.md"))
    return sorted(names, key=lambda n: (n != DEFAULT_GUIDE, n)) or [DEFAULT_GUIDE]


def guide_path(guide: str | None) -> Path:
    p = GUIDES_DIR / f"{guide or DEFAULT_GUIDE}.md"
    return p if p.is_file() else QP.GUIDE


def ana_from_piece(piece: dict) -> dict:
    """a music analysis rebuilt from a planned piece (cut / bpm / beats / downbeats live in piece.audio and piece.type); the graph passes the
    real MOGRAPH_MUSIC, this is for a Parse / Repair node that was not given one."""
    t, a = piece.get("type") or {}, piece.get("audio") or {}
    cut = float(a.get("cut") or t.get("duration") or 0.0)
    bpm = float(t.get("bpm") or a.get("bpm") or 120.0)
    return dict(cut=cut, nframes=int(np.ceil(cut * PR.FPS - 1e-6)), bpm=bpm, bar_len=4 * 60.0 / bpm, key="", beats=list(t.get("beats") or []),
                downbeats=list(t.get("downbeats") or []), chord_changes=[], curves={}, fps=PR.FPS)


def make_locked(piece: dict) -> dict:
    """the fields the model may not touch, as a plain dict (the `locked` STRING of the nodes is its JSON)."""
    out = {k: copy.deepcopy(piece[k]) for k in LOCKED_KEYS if k in piece}
    out["words"] = [w["text"] for w in piece["type"]["words"]]
    out["type_seed"] = piece["type"].get("seed", 1)
    c = piece.get("compose") or {}
    out["compose_locked"] = {k: c.get(k) for k in ("size", "crossfade", "crf")}
    out["technique"] = piece.get("technique")
    return out


def apply_locked(piece: dict, locked: dict | None) -> dict:
    """the piece with the locked fields written back (a no-op when `locked` is what make_locked made from the same piece)."""
    p = copy.deepcopy(piece)
    if not locked:
        return p
    for k in LOCKED_KEYS:
        if k in locked:
            p[k] = copy.deepcopy(locked[k])
    if "type_seed" in locked:
        p["type"]["seed"] = locked["type_seed"]
    for k, v in (locked.get("compose_locked") or {}).items():
        if v is not None:
            p.setdefault("compose", {})[k] = copy.deepcopy(v)
    if isinstance(locked.get("still_palette"), dict):        # the limited palette of the piece's own first still (package): the design prompt states it, the fold snaps the type colours to it
        p.setdefault("plan", {})["still_palette"] = copy.deepcopy(locked["still_palette"])
    return p


def still_palette_of(piece: dict, clips: list[dict] | None, k: int = 5) -> dict | None:
    """3-5 colours from the piece's FIRST still: frame 0 of the first shot's clip (the I2V clip starts from its MOSS still). None without clips. Seeded by the piece seed, so a re-run is exact."""
    from tools.fusion import still_palette as SP
    if not clips:
        return None
    by = {c["sid"]: c for c in clips}
    rows = QP.shot_rows(piece)
    first = next((by[r["id"]] for r in rows if r["id"] in by), None) or clips[0]
    try:
        fr = _u8(first["u8"][:1])[0]
    except Exception:       # noqa: BLE001  no frames: no palette, the family palette stays
        return None
    ls = (piece.get("plan") or {}).get("layout_seed")                       # a layout seed (Randomise layout -> Use this layout): the palette pick of that draw (layout_preview.still_palette_for)
    if ls:
        from tools.fusion import layout_preview as LPV
        return LPV.still_palette_for(fr, int(ls))
    return SP.extract(fr, k=k, seed=int((piece.get("type") or {}).get("seed") or 0))


def _piece_needs(piece: dict) -> dict:
    return {s["id"]: RE.shot_needs(piece, s["id"]) for s in piece["shots"]}


def allowed_techniques(piece: dict, ana: dict) -> list[str]:
    """techniques whose recipe reads no matte / depth the graph did not make for this piece (Gather and the alpha / depth chains follow the PLAN's
    recipe, and the footage exists before the design). The plan's own technique is first."""
    have = _piece_needs(piece)
    own = piece.get("technique") or PR.infer_technique(piece)
    out = [own]
    for tech in QP.TECHNIQUES:
        if tech in out:
            continue
        try:
            stub = {"compose": {"technique": tech}}
            cand = copy.deepcopy(piece)
            cand["compose"] = QP.build_compose(piece, {"compose": stub["compose"], "type": {"palette": piece["type"].get("palette") or {}}}, ana)
            need = _piece_needs(cand)
        except Exception:       # noqa: BLE001
            continue
        if all(not need[sid][k] or have.get(sid, {}).get(k) for sid in need for k in ("matte", "depth")):
            out.append(tech)
    return out


@contextlib.contextmanager
def app_context(piece: dict, ana: dict, locked: dict | None = None, *, request: dict | None = None, guide: str | None = None,
                images_text: str | None = None, footage: dict | None = None):
    """register the in-graph piece under qwen_piece.APP so build_prompt / normalize_design / fit_scene / timing_errors work on it.
    `footage` = the piece the shots were generated for (the plan): it decides which compose techniques are allowed."""
    orig = apply_locked(piece, locked)
    fmt = PR.format_of(orig)
    size = (orig.get("sizes") or {}).get("compose") or (orig.get("compose") or {}).get("size") or [1080, 1920]
    words = [w["text"] for w in orig["type"]["words"]]
    req = request or {}
    music_prompt = str(req.get("music") or "").strip()
    meta = dict(src=None, title=(orig.get("slug") or " ".join(words)).replace("-", " ").title()[:48] or "Untitled",
                parent_version="", mood_brief=str(orig["type"].get("mood") or "driving"),
                key_text=str(ana.get("key") or "not measured"),
                music_text=("generated instrumental music from the prompt: " + music_prompt) if music_prompt else "generated instrumental music, no sung lyrics",
                lyrics="none (instrumental)", seed_shift=0, guide=str(guide_path(guide)),
                frame=f"{size[0]}x{size[1]} {'horizontal (16:9)' if fmt == '16:9' else 'vertical (9:16)'}",
                shape="horizontal" if fmt == "16:9" else "vertical")
    if images_text:
        meta["images_text"] = images_text
    idea = ((orig.get("plan") or {}).get("footage") or {}).get("idea")
    if idea:
        meta["idea"] = str(idea)
    techniques = allowed_techniques(apply_locked(footage, locked) if footage is not None else orig, ana)
    QP.PIECES[APP] = meta
    QP._CTX[APP] = dict(orig=orig, ana=ana, techniques=techniques)
    try:
        yield APP
    finally:
        QP.PIECES.pop(APP, None)
        QP._CTX.pop(APP, None)


# --------------------------------------------------------------------------------------------- images
def _u8(frames) -> np.ndarray:
    """torch / numpy frames [T,H,W,C] (uint8 or float 0..1) -> uint8 numpy."""
    a = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return a


def contact_arrays(frames, label: str, n: int = 6, tile_w: int = SHEET_TILE_W, span: int | None = None) -> np.ndarray:
    """n frames evenly over the first `span` frames, 3 columns, the time in the shot on every tile (same layout as qwen_piece.contact_sheet,
    from in-memory frames). -> uint8 RGB."""
    a = _u8(frames)
    last = max(min(a.shape[0], span or a.shape[0]) - 1, 0)
    tiles = []
    for i in range(n):
        j = int(round(i * last / (n - 1)))
        fr = np.ascontiguousarray(a[j][..., :3])
        h = int(round(fr.shape[0] * tile_w / fr.shape[1]))
        t = np.ascontiguousarray(cv2.resize(fr, (tile_w, h), interpolation=cv2.INTER_AREA))
        tiles.append(QP._label(t, f"{label} t={j / PR.FPS:.1f}s"))
    return np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(n // 3)])


def matte_arrays(items: list[tuple[str, object]], tile_w: int = MATTE_TILE_W, cols: int = 4) -> np.ndarray:
    """one mid-shot matte frame per shot (white = subject). items = [(sid, frames [T,H,W,C])]. -> uint8 RGB."""
    tiles = []
    for sid, m in items:
        a = _u8(m)
        f = a[a.shape[0] // 2]
        g = f.mean(axis=-1).astype(np.uint8) if f.ndim == 3 else f
        h = int(round(g.shape[0] * tile_w / g.shape[1]))
        t = cv2.cvtColor(cv2.resize(g, (tile_w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        tiles.append(QP._label(np.ascontiguousarray(t), f"matte {sid}"))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])


def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w, 3), np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def combine(sheets: list[np.ndarray], cols: int = SHEET_COLS) -> np.ndarray:
    """all sheets in one image, reading order left to right, top to bottom, padded with black to a common cell size."""
    h = max(s.shape[0] for s in sheets)
    w = max(s.shape[1] for s in sheets)
    cells = [pad_to(s, h, w) for s in sheets]
    while len(cells) % cols:
        cells.append(np.zeros((h, w, 3), np.uint8))
    return np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(len(cells) // cols)])


def sheets_for(piece: dict, clips: list[dict], mattes: dict | None = None) -> tuple[list[np.ndarray], list[str]]:
    """[contact sheet per shot ..., matte sheet] + one reading-order description per sheet."""
    by = {c["sid"]: c for c in clips or []}
    rows = QP.shot_rows(piece)
    sheets, desc = [], []
    for r in rows:
        c = by.get(r["id"])
        if c is None:
            continue
        sheets.append(contact_arrays(c["u8"], r["id"], span=r["frames"]))
        desc.append(f"contact sheet of {r['id']}")
    items = [(r["id"], (mattes or {}).get(r["id"])) for r in rows]
    items = [(sid, m) for sid, m in items if m is not None and getattr(m, "shape", (0,))[0] > 1]
    if items:
        sheets.append(matte_arrays(items))
        desc.append("matte sheet (one frame per shot, white = the subject the type can go behind or in front of)")
    return sheets, desc


def images_text(desc: list[str]) -> str:
    if not desc:
        return "No images are attached."
    return ("One image is attached: a grid of sheets in reading order (left to right, top to bottom): " + "; ".join(f"{i + 1}. {d}" for i, d in enumerate(desc)) +
            ". Contact-sheet tiles are labelled with the time inside the shot.")


# --------------------------------------------------------------------------------------------- package
def package(request: dict, music: dict, piece: dict, clips: list[dict], mattes: dict | None = None, guide: str | None = None,
            seed: int = 1) -> dict:
    """-> dict(prompt, images [uint8 HxWx3 each, padded to one size], sheet [uint8, all sheets in one image], locked [JSON str],
    techniques [allowed], desc)."""
    sheets, desc = sheets_for(piece, clips, mattes)
    locked = make_locked(piece)
    sp = still_palette_of(piece, clips)
    if sp:
        locked["still_palette"] = sp
    with app_context(piece, music, locked, request=request, guide=guide, images_text=images_text(desc)) as name:
        prompt = QP.build_prompt(name, int(seed))
        techniques = list(QP._CTX[name]["techniques"])
    if sheets:
        h, w = max(s.shape[0] for s in sheets), max(s.shape[1] for s in sheets)
        images = [pad_to(s, h, w) for s in sheets]
        sheet = combine(sheets)
    else:
        images, sheet = [np.zeros((16, 16, 3), np.uint8)], np.zeros((16, 16, 3), np.uint8)
    return dict(prompt=prompt, images=images, sheet=sheet, locked=json.dumps(locked, ensure_ascii=False), techniques=techniques, desc=desc)


# --------------------------------------------------------------------------------------------- reply -> design
def reply_from_piece(piece: dict, ana: dict) -> dict:
    """a model-style reply object that reproduces the rules' own plan (hit / hold_s / form_s / exit_s from its dry run). Used as the stub that lets
    the compose section of a reply without `type` be checked, and by the tests."""
    probe = copy.deepcopy(piece)
    PR.fill_type_timing(probe, ana)
    chk = PR.dry_run(probe)
    t = piece["type"]
    words = []
    for w, c in zip(t["words"], chk["words"]):
        d = {"text": w["text"], "font": w["font"], "module": w["module"], "form_in": w.get("form_in", "lines"), "form_out": w.get("form_out", "lines"),
             "hit": round(float(c["hit"]), 3), "hold_s": round(max(float(c["release"]) - float(c["hit"]), QP.MIN_HOLD_IN), 3),
             "form_s": w.get("form_s", 0.75), "exit_s": w.get("exit_s", 0.5)}
        if w.get("placement"):
            d["placement"] = {k: v for k, v in w["placement"].items() if k in ("yaw", "wf", "hf")}
        d.update(OW.word_fields(w, t.get("mood"), t.get("palette"), QP.scene_text(piece)))        # the rules' owner (the plan's own, else handoff + the default material)
        words.append(d)
    return {"concept": "the rules' own plan", "treatment_notes": "type behind the subject on the beat grid",
            "type": {"mood": t.get("mood", "driving"), "energy": t.get("energy", 0.7), "palette": copy.deepcopy(t.get("palette") or {}),
                     "camera": t.get("camera") if t.get("camera") in QP.CAMERAS else "beat_orbit", "transitions": "auto",
                     "line": {"px": 4.0}, "words": words},
            "compose": {"technique": piece.get("technique") or PR.infer_technique(piece)}}


def _split(errors: list[str]) -> tuple[list[str], list[str]]:
    """(type errors, compose errors); a missing `concept` is not an error of the design."""
    typ, comp = [], []
    for e in errors:
        if e.startswith("`concept`"):
            continue
        (comp if e.startswith("compose") or e.startswith("missing object `compose`") else typ).append(e)
    return typ, comp


def _probe(orig: dict, t: dict) -> dict:
    p = copy.deepcopy(orig)
    p["type"] = dict(copy.deepcopy(t), seed=orig["type"].get("seed", 1))
    return p


def type_errors_of(t: dict, name: str) -> list[str]:
    """the engine's checks on a type section (hold >= 1.0 s, no overlap after beat snapping, the last hold ends before the music does)."""
    errs, _ = QP.timing_errors(_probe(QP.load_original(name), t), QP.load_analysis(name))
    return errs


def builds(t: dict, name: str) -> bool:
    """does the type engine accept the section at all (spec.build_scene through plan_rules.dry_run)?"""
    try:
        p = _probe(QP.load_original(name), t)
        PR.fill_type_timing(p, QP.load_analysis(name))
        PR.dry_run(p)
        return True
    except Exception:       # noqa: BLE001
        return False


def _timing(d: dict) -> list[dict] | None:
    """the words' compiled timing (t0 / t1 / form_s / exit_s / hit_on) of a normalised design, or None before compile_timing has run."""
    ws = (d.get("type") or {}).get("words") or []
    if not ws or "t0" not in ws[0]:
        return None
    return [dict(t0=float(w.get("t0", 0.0)), t1=float(w.get("t1", 0.0)), form_s=float(w.get("form_s", 0.0)), exit_s=float(w.get("exit_s", 0.0)),
                 hit_on=w.get("hit_on")) for w in ws]


def evaluate(obj, name: str, base: dict) -> dict:
    """_evaluate with the planned window_other; when the music cannot take its hold (the timeline does not fit) the same reply is evaluated again without it (the 3D cuts stay)."""
    ev = _evaluate(obj, name, base, True)
    if ev["type_errors"] and any("window_other planned" in f for f in ev["fixes"]):
        ev2 = _evaluate(obj, name, base, "cuts_only")
        if len(ev2["type_errors"]) < len(ev["type_errors"]):
            ev2["fixes"].append("compose.cuts: window_other dropped: the hold across the cut could not be fitted to the music (" + ev["type_errors"][0][:90] + ")")
            return ev2
    return ev


def _evaluate(obj, name: str, base: dict, plan_windows=True) -> dict:
    """normalize_design + the timing fit + the engine's timing checks on a parsed reply. -> dict(design, fixes, type_errors, compose_errors, pre, post):
    pre / post = the words' timing right after the model's own timing was compiled / after the piece-level rules (timing fit, window validator, font
    rules) ran: design_salvage.account() compares them to find the fields those rules changed."""
    if obj is None:
        e = ["the reply contained no parseable JSON object; return only the JSON object"]
        return dict(design=None, fixes=[], type_errors=e, compose_errors=[], pre=None, post=None)
    stub = False
    if not isinstance(obj.get("type"), dict):                    # evaluate the rest (compose) against the rules' own type
        obj = dict(obj, type=reply_from_piece(base, QP.load_analysis(name))["type"])
        stub = True
    d, fixes, errors = QP.normalize_design(obj, name, plan_windows=plan_windows)         # + the planning rule: a word on a place change, window_other (design_options.plan_window_other)
    typ, comp = _split(errors)
    pre = _timing(d) if d is not None else None
    if stub:
        fixes = [f for f in fixes if not f.startswith("type.")]           # those belong to the stand-in type
        typ = ["missing object `type` (the reply has no `type` object: palette, camera, transitions and the words with their hit / hold_s / fonts / modules)"] + typ
    if d is not None and not typ:
        typ += QP.fit_scene(d, name, fixes)
    if d is not None and not typ:
        QP.finish_design(d, name, fixes)                                       # window validator, then the font rules (>= 3 faces, expressive hero)
        typ += type_errors_of(d["type"], name)
    return dict(design=d, fixes=fixes, type_errors=typ, compose_errors=comp, stub=stub, pre=pre, post=_timing(d) if d is not None else None)


GRAPHIC_KEYS = ("font", "axes", "module", "lines")
GRAPHIC_PLACEMENT = ("yaw", "pitch", "roll", "dx", "dy", "wf", "hf")


def lock_graphic(i: int, w: dict, pw: dict, fixes: list) -> None:
    """an LTX-owned word is drawn by the video model FROM the plan's control pass (a depth / canny render of the plan's graphic: its face, module, lines and box). If the design then drew
    the flat word differently, the flat graphic and the rendered letters would be two different versions of the word (user 2026-10-29: "not two different versions on top of each other").
    So the designed word keeps the plan's face / axes / module / lines / placement. Every difference is logged."""
    diff = []
    for k in GRAPHIC_KEYS:
        if pw.get(k) is None and k in ("axes", "lines"):
            if k in w:
                diff.append(k)
                w.pop(k)
            continue
        if pw.get(k) is not None and w.get(k) != pw.get(k):
            diff.append(k)
            w[k] = copy.deepcopy(pw[k])
    ppl, pl = pw.get("placement") or {}, w.setdefault("placement", {})
    for k in GRAPHIC_PLACEMENT:
        if ppl.get(k) is not None and pl.get(k) != ppl.get(k):
            diff.append(f"placement.{k}")
            pl[k] = ppl[k]
        elif ppl.get(k) is None and k in pl:                                   # the plan's render used the engine default for this key
            diff.append(f"placement.{k}")
            pl.pop(k)
    if diff:
        fixes.append(f"type.words[{i}] {', '.join(diff)} -> the plan's: the control pass the footage was made from shows that graphic, so the flat word and the rendered letters must be one version")


def lock_owner(design_words: list, plan_words: list, plan_rows, fixes: list) -> None:
    """the shots are generated BEFORE the design step, from the PLAN's control pass: where the plan words carry owner fields the footage already holds LTX's
    letters from the plan's handoff time on, so the designed word keeps that owner (render, material) and the absolute handoff time `ltx_t` (the flat fade must
    land where LTX's letters begin). The model's own choice is kept in `asked` only in the fix text. No owner fields in the plan (an old plan / offline
    qwen_piece over footage made without owners): nothing is locked, the model's owner fields stand."""
    if not OW.has_owner(plan_words):
        return
    wins = OW.windows({"type": {"words": plan_words}}, plan_rows)
    for i, (w, pw) in enumerate(zip(design_words, wins or [])):
        asked = (w.get("render"), w.get("ltx_at"), w.get("material"))
        w["render"] = pw["render"]
        if pw["render"] == "handoff":
            w["ltx_at"] = pw["ltx_at"]
            w["ltx_t"] = round(OW.handoff_t(pw), 3)
        else:
            w.pop("ltx_at", None)
            w.pop("ltx_t", None)
        if pw["render"] != "flat":
            w["material"] = pw["material"] or w.get("material") or OW.default_material()
        else:
            w.pop("material", None)
        if pw["render"] != "flat":
            lock_graphic(i, w, plan_words[i], fixes)
        now = (w.get("render"), w.get("ltx_at"), w.get("material"))
        if asked[0] != now[0] or (now[0] == "handoff" and asked[1] != now[1]) or (now[0] != "flat" and asked[2] != now[2]):
            fixes.append(f"type.words[{i}].render {asked[0]!r} -> {now[0]!r}: the shot footage was generated from the plan's control pass with that owner "
                         f"(material {now[2]!r}, handoff at {w.get('ltx_t', '-')} s); the owner cannot change after the footage exists")


def _plan_rows(plan: dict, ana: dict):
    """the plan's TRUE word timeline (the same build_scene the type passes use): info.words rows, or None when the dry run fails."""
    try:
        probe = copy.deepcopy(plan)
        PR.fill_type_timing(probe, ana)
        return PR.dry_run(probe)["words"]
    except Exception:       # noqa: BLE001
        return None


def _clean_type(t: dict, seed) -> dict:
    t = copy.deepcopy(t)
    for w in t.get("words", []):
        for k in [k for k in w if k.startswith("_")]:
            w.pop(k)
    t["seed"] = seed
    return t


MODEL = "design_llm_8b"                 # the label written into plan.design.model (the redesign graph passes the arm's name)
TIMING_KEYS = ("hit", "hold_s", "form_s", "exit_s")


def _vary() -> dict:
    """{font file: {axis tag: (lo, hi)}} of the font registry (the axes the salvage may keep)."""
    from mograph import font_registry as fr
    return {v["file"]: {k: (x[0], x[1]) for k, x in (v.get("vary") or {}).items()} for v in fr.REGISTRY.values()}


def rules_flat(piece: dict, ana: dict, want: list[str]) -> dict:
    """the rules planner's value for every design field of `piece` as {path: value} (same paths as design_salvage.flatten): what an invalid or missing
    field of the model's reply is replaced with, field by field."""
    rr = reply_from_piece(piece, ana)
    flat = SV.flatten(rr, want)
    for i, w in enumerate(piece["type"]["words"]):
        if w.get("lines"):
            flat[f"type.words[{i}].lines"] = list(w["lines"])
    flat["concept"], flat["treatment_notes"] = rr["concept"], rr["treatment_notes"]
    return flat


def _all_rules(sv, section: str, why: str) -> None:
    for p in list(sv.status):
        if p == section or p.startswith(section + ".") or p.startswith(section + "["):
            sv.status[p] = ("rules", why)


def _section_src(sv, section: str, present: bool, used: bool) -> str:
    if not present or not used:
        return "rules"
    st = [s for p, (s, _) in SV.counted(sv).items() if p.startswith(section + ".") or p.startswith(section + "[")]
    n_rules = sum(1 for s in st if s == "rules")
    return "model" if n_rules == 0 else ("model fields + rules fields" if n_rules < len(st) else "rules")


def _minimal_timing(clean: dict, rules: dict, want: list, name: str, orig: dict):
    """the model's timing does not fit the music (the engine rejects it): take the FEWEST timing fields from the rules, the last word first (it is the one that
    runs out of music): its hit alone, then its four timing fields, then the same for the word before. -> (clean reply, evaluation, [paths taken]) or None."""
    clean2, taken = copy.deepcopy(clean), []
    for k in reversed(range(len(want))):
        for keys in (("hit",), TIMING_KEYS):
            for key in keys:
                pth = SV._w(k, key)
                if pth in rules and pth not in taken:
                    clean2["type"]["words"][k][key] = rules[pth]
                    taken.append(pth)
            ev2 = evaluate(clean2, name, orig)
            if not ev2["type_errors"]:
                return clean2, ev2, taken
    return None


def resolve(obj, base: dict, plan: dict, ana: dict, locked: dict | None = None, base_label: str = "rules", *, guide: str | None = None,
            request: dict | None = None, raw: dict | None = None, from_repair=(), model: str = MODEL) -> tuple[dict, dict]:
    """parsed reply (or None) + the piece to fall back to (`base`) -> (valid piece, report dict). FIELD-LEVEL (design_salvage.py): every word and every compose
    field is read on its own; a valid field is the model's as written, an invalid one is replaced alone (nearest allowed value, else the rules' value for that
    field); only a whole missing section comes from the rules. `raw` = the flat model values of an earlier pass (+ the repair reply's), used instead of `obj`
    by the repair step. See the module docstring for the sections and design_salvage for the rules."""
    with app_context(base, ana, locked, request=request, guide=guide, footage=plan) as name:
        orig = QP.load_original(name)
        seed = orig["type"].get("seed", 1)
        want = [w["text"] for w in orig["type"]["words"]]
        rules = rules_flat(orig, ana, want)
        no_json = obj is None and raw is None
        raw = dict(raw) if raw is not None else SV.flatten(obj, want)
        techniques = list(QP._CTX[name]["techniques"])
        sv = SV.salvage(raw, rules, dict(want=want, techniques=techniques, plan_technique=techniques[0], cut=float(ana["cut"]), vary=_vary(),
                                         from_repair=list(from_repair), scene=QP.scene_text(orig)))
        has_t, has_c = bool(raw.get("@type")), bool(raw.get("@compose"))
        extra_fixes: list[str] = []
        ev = dict(design=None, fixes=[], type_errors=[], compose_errors=[], pre=None, post=None)
        if has_t or has_c:
            ev = evaluate(sv.clean, name, orig)
            if ev["design"] is not None and ev["type_errors"] and has_t:      # the model's timing does not fit the music: the FEWEST timing fields come from the rules
                first = ev["type_errors"][0]
                got = _minimal_timing(sv.clean, rules, want, name, orig)
                ev2, clean2, taken = (got[1], got[0], got[2]) if got else (None, None, [])
                if ev2 is not None and not ev2["type_errors"]:
                    sv.clean, ev = clean2, ev2
                    for pth in taken:
                        sv.status[pth] = ("rules", "the model's timing could not be fitted to the music: " + first[:120])
                    extra_fixes.append(f"timing: {len(taken)} field(s) ({', '.join(p.split('.', 2)[2] if p.startswith('type.words') else p for p in taken)}) taken from the rules, "
                                       f"the rest of the type kept; the model's timing failed: {first[:160]}")
        d = ev["design"]
        src = {}
        type_ok = bool(d is not None and has_t and not ev["type_errors"])
        if type_ok and OW.has_owner(orig["type"]["words"]):
            lock_owner(d["type"]["words"], orig["type"]["words"], _plan_rows(orig, ana), ev["fixes"])
            from tools.fusion import design_defaults as DD                    # the locked graphics may have brought faces back: the one-system rules run once more around them
            from tools.fusion import font_pins as FP                          # the user's per-line fonts stay through this second pass too
            pins_ = (orig.get("type") or {}).get("font_pins") or []
            DD.enforce_type(d["type"]["words"], ev["fixes"], protected=set(QP.window_words(d)) | {i for i, w_ in enumerate(orig["type"]["words"]) if w_.get("render") not in (None, "flat")},
                            font_protected=FP.pinned_indices(d["type"]["words"], pins_))
            FP.apply(d["type"]["words"], pins_, ev["fixes"])
            FP.contrast(d["type"]["words"], pins_, ev["fixes"])
        if type_ok:
            SV.account(sv, ev["fixes"], ev["pre"], ev["post"], d)
            t_sec = _clean_type(d["type"], seed)
        else:
            t_sec = copy.deepcopy(base["type"])
            _all_rules(sv, "type", "the type section could not be used" if has_t else "section missing")
        src["type"] = _section_src(sv, "type", has_t, type_ok)
        comp, comp_model = None, False
        if d is not None and has_c and not ev["compose_errors"]:
            try:
                comp = QP.build_compose(orig, {"compose": d["compose"], "type": {"palette": t_sec.get("palette") or d["type"]["palette"]}}, ana)
                comp_model = True
            except Exception as e:       # noqa: BLE001
                ev["compose_errors"].append(f"compose: could not be built ({type(e).__name__}: {str(e)[:120]})")
        if comp is None:
            comp = copy.deepcopy(base["compose"])
            _all_rules(sv, "compose", "the compose section could not be used" if has_c else "section missing")
        src["compose"] = _section_src(sv, "compose", has_c, comp_model)
        ops_rep = None
        if has_c and raw.get("compose.ops"):                       # the model's own op chains (compose building blocks): each op validated alone, a chain that draws no type is left out
            comp, ops_rep = apply_model_chains(comp if comp is not None else copy.deepcopy(base["compose"]), raw["compose.ops"], orig, float(ana["cut"]), sv)
        out = copy.deepcopy(orig)
        out["type"], out["compose"] = t_sec, comp
        if d is not None and (comp_model or type_ok):                          # text-as-window choices: compose.cuts + words' window_source + the shots' window_cut ops
            from tools.fusion import design_options as DO
            wd = {"type": {"words": t_sec.get("words", [])}, "compose": {"cuts": d["compose"].get("cuts")} if comp_model else {}}
            if src["type"] == "rules":
                wd["type"] = {"words": []}                                     # window words of a type that fell back to the rules are not the model's
            wnotes = DO.to_piece(out, wd, None)
            if wnotes:
                out.setdefault("plan", {})["window_notes"] = list(wnotes)
        if comp_model:
            out["technique"] = d["compose"]["technique"]
        out["signoff"] = dict(orig.get("signoff") or {}, off=True)
        fields = SV.report(sv)
        fields["sections"] = dict(src)
        if isinstance(raw.get("system"), dict):
            out.setdefault("plan", {})["system"] = raw["system"]
        if ops_rep:
            out.setdefault("plan", {}).setdefault("compose_ops", ops_rep)
        out["design"] = dict(src=src, concept=(d or {}).get("concept", ""), treatment_notes=(d or {}).get("treatment_notes", ""),
                             technique=(d or {}).get("compose", {}).get("technique") if comp_model else None,
                             salvage=dict(raw=sv.raw, pending=sv.pending), fields=fields)
        final = _finish(out, ana, model)
        from tools.fusion import text_guard as TG                            # the renderer never invents its own text: mend the prompts / negatives, then judge what is left
        def _tg_kind(x):                                                     # (where, what): the wording of a problem moves with the timeline, its shot and kind do not
            return (x.split(":")[0], "spells" if "spells" in x else "guide" if "guide is" in x else "names text" if "names text" in x else "negative" if "negative" in x else x)
        tg_inherited = {_tg_kind(x) for x in TG.plan_problems(orig)}         # problems the PLAN already had (its footage exists, the design cannot change them) are warnings; a NEW kind fails the design check
        tg_fixes = TG.fix_plan(final)
        tg_all = TG.plan_problems(final)
        tg_probs = [x for x in tg_all if _tg_kind(x) not in tg_inherited]
        tg_fixes += [f"text_guard (inherited from the plan, the footage is already made): {x}" for x in tg_all if _tg_kind(x) in tg_inherited]
    errors_t = list(ev["type_errors"])
    errors_c = list(ev["compose_errors"])
    if no_json:
        errors_t = list(ev["type_errors"]) or ["the reply contained no parseable JSON object; return only the JSON object"]
    elif not has_t:
        errors_t = ["missing object `type`"] + errors_t
    if not has_c and not no_json:
        errors_c = ["missing object `compose`"] + errors_c
    errors_c += [f"text_guard: {x}" for x in tg_probs]
    pending = list(sv.pending)
    report = dict(src=src, ok=bool(type_ok and has_c and not errors_c and not errors_t and not pending), fixes=extra_fixes + ev["fixes"] + list((final.get("plan") or {}).get("clear_text") or []) + tg_fixes, type_errors=errors_t,
                  compose_errors=errors_c, pending=pending, fields=fields, want=want,
                  mode="short" if (has_t and has_c and pending and not errors_t and not errors_c) else "long")
    return final, report


def apply_model_chains(comp: dict, raw_ops: dict, orig: dict, cut: float, sv=None) -> tuple[dict, dict]:
    """the model's compose.default / compose.shots.<id> op chains over `comp` (the rules' / technique compose): validated by compose_blocks (names, arguments, $references and
    picture kinds, each op alone), restricted to what the GENERATED footage has (the shots are made before the design: `$matte` / `$depth` only on shots with that pass; the type passes the plan
    asked for). A chain that survives replaces the chain of its shot (default = every shot without an entry); one that does not leaves the rules' chain there. -> (comp, report)."""
    have_tp = set(RE.pass_names(orig.get("type_passes"), default=())) | {"matte", "beauty"}
    unavail = {}
    for sh in orig["shots"]:
        u = set()
        if not sh.get("matte"):
            u.add("matte")
        if not sh.get("depth"):
            u.add("depth")
        u |= {buf for buf, ps in CB.PASS_FOR.items() if ps not in have_tp}
        unavail[sh["id"]] = u
    chains, fixes = CB.normalize_compose(raw_ops, [x["id"] for x in orig["shots"]], cut, unavail)
    rep = dict(source="rules", chains=0, fixes=fixes[:30], n_fixes=len(fixes))
    comp = copy.deepcopy(comp)
    if chains.get("default"):
        comp["default"] = chains["default"]
        rep["chains"] += 1
    for sid, ch in (chains.get("shots") or {}).items():
        comp.setdefault("shots", {})[sid] = ch
        rep["chains"] += 1
    for pid in ("default",) + tuple((raw_ops.get("shots") or {}) if isinstance(raw_ops.get("shots"), dict) else ()):
        got = chains.get("default") if pid == "default" else (chains.get("shots") or {}).get(pid)
        if sv is not None:
            raw_given = raw_ops.get("default") if pid == "default" else (raw_ops.get("shots") or {}).get(pid)
            if raw_given:
                clean = not [f for f in fixes if f.startswith(pid + ".")]
                sv.put(f"compose.ops.{pid}", ("model" if clean else "fixed") if got else "rules", "" if clean else ("; ".join(f for f in fixes if f.startswith(pid + "."))[:200]))
    if rep["chains"]:
        rep["source"] = "model"
    return comp, rep


def weave_pan(probe: dict, out: dict, ana: dict, chk: dict | None = None) -> list[str]:
    """a pan the design chose (`pan:<dir>` cut -> the word's `transition`, design_options.to_piece) needs the same post-step the planner runs
    (plan_rules._finish -> pan_plan.weave): the plate_pan ops in every compose chain and the matching LTX camera phrase. No-op without a pan (and cheap: the camera scene
    is only built when there is one). When the weave fails the pans are removed from `out` (the piece renders without them, a note says so). -> notes."""
    from tools.fusion import pan_plan
    if not pan_plan.boundaries(probe.get("type")):
        return []
    trial = copy.deepcopy(probe)                       # a failed weave leaves the probe untouched
    try:
        pan_plan.weave(trial, ana, chk)
        probe["compose"], probe["shots"], probe["plan"] = trial["compose"], trial["shots"], trial["plan"]
        return []
    except Exception as e:       # noqa: BLE001
        for w in out["type"].get("words", []):
            if pan_plan._is_pan(w.get("transition")):
                w.pop("transition", None)
        note = f"pan dropped: pan_plan.weave failed ({type(e).__name__}: {str(e)[:160]})"
        probe.setdefault("plan", {}).setdefault("window_notes", []).append(note)       # `plan` is copied from the probe by _finish
        return [note]


def normalize_passes(piece: dict) -> list[str]:
    """`type_passes` must be a list of known pass names and every shot's control pass ONE known name (a string such as "all" / "matte" / "depth+line" used to reach set() and crash the type
    stage with `unknown pass 'a'`). Mended in place, every change returned as a note."""
    notes: list[str] = []
    want = RE.pass_names(piece.get("type_passes"), warn=notes.append)
    if piece.get("type_passes") != want:
        notes.append(f"type_passes {piece.get('type_passes')!r} -> {want}")
        piece["type_passes"] = want
    for s in piece.get("shots") or []:
        ctl = s.get("control")
        if isinstance(ctl, dict) and "pass" in ctl:
            new = RE.norm_control_pass(ctl["pass"], warn=notes.append)
            if new != ctl["pass"]:
                notes.append(f"{s.get('id')}: control pass {ctl['pass']!r} -> {new!r}")
                ctl["pass"] = new
    return notes


def _morph_names_to_null(out: dict) -> None:
    """`contour_morph` in type.transitions / a word's transition is not a type-engine module: the engine picks that gap (null) and contour_morph.weave plays the morph on a matte shot."""
    t = out.get("type") or {}
    named: set = set()
    if isinstance(t.get("transitions"), list):
        named |= {i for i, x in enumerate(t["transitions"]) if x == "contour_morph"}
        t["transitions"] = [None if x == "contour_morph" else x for x in t["transitions"]]
    for i, w in enumerate(t.get("words") or []):
        if isinstance(w, dict) and w.get("transition") == "contour_morph":
            named.add(i)
            w.pop("transition", None)
    if named:                                                  # contour_morph.plan keeps a specific `look`'s own transitions except at the gaps the model named
        out.setdefault("plan", {})["contour_morph_asked"] = sorted(named)


def _finish(out: dict, ana: dict, model: str = MODEL) -> dict:
    """matte treatments + flow_wake woven (the repo's weavers), the whole piece dry-run on the real type scene, layout filled."""
    _morph_names_to_null(out)
    pnotes = normalize_passes(out)
    if pnotes:
        out.setdefault("plan", {}).setdefault("notes", []).extend(pnotes)
    probe = copy.deepcopy(out)
    PR.fill_type_timing(probe, ana)
    chk = PR.dry_run(probe)
    try:                                                                       # coverage rule: no long stretch with no word on screen (QC v026); the type timing it moves is written back to the piece
        chk = PR.enforce_coverage(probe, chk)
        for w_o, w_p in zip(out["type"].get("words", []), probe["type"].get("words", [])):
            for k_ in ("t0", "t1", "exit_s"):
                if k_ in w_p:
                    w_o[k_] = w_p[k_]
        if probe["type"].get("overlap") is not None:
            out["type"]["overlap"] = probe["type"]["overlap"]
    except Exception as e:      # noqa: BLE001  never break a piece for it
        probe.setdefault("plan", {}).setdefault("notes", []).append(f"coverage skipped: {type(e).__name__}: {str(e)[:120]}")
    PR.weave_matte(probe, ana, chk)
    PR.weave_flow(probe, ana, chk)
    weave_pan(probe, out, ana, chk)
    try:
        PR.weave_hold_life(probe, ana, chk, (probe.get("sizes") or {}).get("compose") or PR.sizes_for(probe.get("quality") or "final", PR.format_of(probe))["compose"])      # hold life on the DESIGNED chains too (the plan stage's weave was replaced with the model's compose)
    except Exception as e:      # noqa: BLE001  never break a piece for it
        probe.setdefault("plan", {}).setdefault("notes", []).append(f"hold life skipped: {type(e).__name__}: {str(e)[:120]}")
    from tools.fusion import explore_effects  # the hearted explore looks on the DESIGNED chains too (weave never raises; placement is kept in plan.explore)
    explore_effects.weave(probe, ana, chk, (probe.get("sizes") or {}).get("compose") or PR.sizes_for(probe.get("quality") or "final", PR.format_of(probe))["compose"])
    from tools.fusion import contour_morph  # CONTOUR_MORPH = the default word-to-word move on a matte shot, unless the model named another transition for the gap (weave never raises)
    contour_morph.weave(probe, ana, chk, (probe.get("sizes") or {}).get("compose") or PR.sizes_for(probe.get("quality") or "final", PR.format_of(probe))["compose"], designed=True)
    from tools.fusion import word_rules  # 2026-10-31 WORD RULES (same rule as the cuts): no word transition on two consecutive gaps, at most 2 per piece (before the sweep, whose word events follow the transition names)
    word_rules.enforce(probe, ana, chk, (probe.get("sizes") or {}).get("compose") or PR.sizes_for(probe.get("quality") or "final", PR.format_of(probe))["compose"], designed=True)
    from tools.fusion import contour_sweep  # contour sweep on the DESIGNED chains too (weave never raises)
    contour_sweep.weave(probe, ana, chk, (probe.get("sizes") or {}).get("compose") or PR.sizes_for(probe.get("quality") or "final", PR.format_of(probe))["compose"])
    from tools.fusion import design_defaults as DD
    DD.enforce_compose(probe, PR.word_holds(probe, ana, chk))            # clear holds: flow_wake fully settled inside every word hold, the legibility guard on (logged in plan.clear_text)
    out["compose"], out["shots"], out["plan"] = probe["compose"], probe["shots"], probe["plan"]
    if probe["type"].get("transitions") != out["type"].get("transitions"):      # word_rules wrote explicit transitions into the probe
        out["type"]["transitions"] = probe["type"]["transitions"]
    from tools.fusion import type_frame                                    # every word's ink box inside the frame over its hold, measured with the piece's own type camera (v014 r02: 'follow' cut a letter off every word)
    q_ = out.get("quality") or "final"
    out["plan"].setdefault("notes", []).extend(type_frame.guard(out, size=(out.get("sizes") or PR.sizes_for(q_, PR.format_of(out)))["type"]))
    out["plan"]["design"] = {"model": model, "src": out["design"]["src"], "salvage": out["design"].get("salvage"), "fields": out["design"].get("fields")}
    RE.need_passes(out)
    return PR.finalize_piece(out, ana)


def safe_resolve(obj, base: dict, plan: dict, ana: dict, locked=None, base_label: str = "rules", **kw) -> tuple[dict, dict]:
    """resolve(), and when even that raises, the rules' piece unchanged (planned, dry-run, valid by construction)."""
    try:
        return resolve(obj, base, plan, ana, locked, base_label, **kw)
    except Exception as e:       # noqa: BLE001
        rep = dict(src={"type": "rules", "compose": "rules"}, ok=False, fixes=[], type_errors=[f"design step failed: {type(e).__name__}: {str(e)[:200]}"],
                   compose_errors=[], pending=[], fields=None, want=[], mode="long")
        p = copy.deepcopy(plan)
        p.setdefault("plan", {})["design"] = {"model": kw.get("model", MODEL), "src": rep["src"], "error": rep["type_errors"][0]}
        return PR.finalize_piece(p, ana), rep


# --------------------------------------------------------------------------------------------- parse / repair
def repair_prompt(prompt: str, reply: str, errors: list[str]) -> str:
    """the LONG repair prompt (the whole design prompt + the rejected reply + the errors): only for a reply with no usable JSON / a missing section."""
    return ((prompt or "").rstrip() + "\n\n=== YOUR PREVIOUS REPLY (rejected) ===\n" + (reply or "(empty)")[:6000] + "\n\n=== CHECK ===\n" +
            QP.repair_text(errors))


def short_repair_prompt(rep: dict, reply: str, piece: dict) -> str:
    """the SHORT repair prompt: only the invalid fields (path, what was written, what is allowed) + the reply format; the fonts list only when a font is asked for."""
    ns = SimpleNamespace(pending=rep["pending"], want=rep.get("want") or [w["text"] for w in piece["type"]["words"]])
    return SV.repair_prompt(ns, reply, QP.fonts_block(piece["type"].get("seed", 1)), ns.want)


def piece_summary(piece: dict) -> str:
    """one line the batch driver reads back from the report: shots mode, shots, stills, the real cut length, words, technique."""
    pl = piece.get("plan") or {}
    cut = float((piece.get("audio") or {}).get("cut") or (piece.get("type") or {}).get("duration") or 0.0)
    from tools.fusion import openings as OP
    return (f"piece: mode={pl.get('shots', '?')} shots={len(piece.get('shots') or [])} stills={len(piece.get('stills') or [])} cut={cut:.3f}s "
            f"frames={int(round(cut * PR.FPS))} words={len((piece.get('type') or {}).get('words') or [])} technique={piece.get('technique')} quality={piece.get('quality')} "
            f"{OP.summary(OP.opening_of(piece))}".rstrip())


def report_text(stage: str, rep: dict, extra: str = "") -> str:
    lines = [f"{stage}: type = {rep['src']['type']}, compose = {rep['src']['compose']}, design valid = {rep['ok']}"]
    f = rep.get("fields")
    if f:
        lines.append(f"fields: {len(f['fields_from_model'])} from the model ({f['pct_model']:g} %), {len(f['fields_fixed'])} fixed, {len(f['fields_from_rules'])} from the rules "
                     f"of {f['n']}; {len(rep.get('pending') or [])} pending for the repair call")
        lines += [f"  fixed {x['path']}: {x['why']}" for x in f["fields_fixed"][:30]]
        lines += [f"  rules {x['path']}: {x['why']}" for x in f["fields_from_rules"][:30]]
    if extra:
        lines.append(extra)
    for k in ("type_errors", "compose_errors"):
        lines += [f"  {k[:-7]} error: {e}" for e in rep[k]]
    lines += [f"  fix: {x}" for x in rep["fixes"][:40]]
    return "\n".join(lines)


def parse(reply: str, locked: str | dict | None, piece: dict, music: dict | None = None, prompt: str = "", model: str = MODEL) -> dict:
    """-> dict(piece_json, ok, repair_prompt, report, piece, fields). piece_json is ALWAYS a valid piece (the model's fields as written, an invalid field
    mended alone, only a missing section from the rules); ok says every field was usable (nothing is pending for a repair call). repair_prompt is SHORT
    (the invalid fields only) unless the reply had no usable JSON / a missing section."""
    ana = music or ana_from_piece(piece)
    lk = json.loads(locked) if isinstance(locked, str) and locked.strip() else (locked or None)
    obj = QP.extract_json(reply or "")
    final, rep = safe_resolve(obj, piece, piece, ana, lk, "rules", model=model)
    errors = rep["type_errors"] + rep["compose_errors"]
    ok = bool(rep["ok"]) and not errors
    if ok:
        rp = ""
    elif rep.get("mode") == "short":
        rp = short_repair_prompt(rep, reply, piece)
    else:
        rp = repair_prompt(prompt, reply, errors + [f"{p['path']}: {p['why']}" for p in rep["pending"] if p["path"] not in ("type", "compose")])
    return dict(piece_json=PR.piece_json(final), ok=ok, repair_prompt=rp, report=report_text("parse", rep, piece_summary(final)), piece=final, fields=rep.get("fields"))


def repair(piece_json: str, ok: bool, repair_reply: str | None, piece: dict, music: dict | None = None, locked: str | dict | None = None,
           model: str = MODEL) -> dict:
    """-> dict(piece_json, report, piece, fields). ok: the parse result is kept as it is. Otherwise the repair reply (a {field path: value} map for the pending
    fields, or a design-shaped object) is merged FIELD BY FIELD into the first reply's fields (kept in the piece: plan.design.salvage) and the design is
    resolved again: a field the repair reply gets right replaces the rules' value, one it gets wrong stays with the rules."""
    ana = music or ana_from_piece(piece)
    try:
        base = json.loads(piece_json)
    except Exception:       # noqa: BLE001
        base = None
    if base is None:                       # not even JSON: the rules' plan
        base, ok = copy.deepcopy(piece), False
    if ok:
        return dict(piece_json=piece_json, report="repair: not needed (the first reply passed)\n" + piece_summary(base), piece=base,
                    fields=((base.get("plan") or {}).get("design") or {}).get("fields"))
    lk = json.loads(locked) if isinstance(locked, str) and locked.strip() else (locked or None)
    obj = QP.extract_json(repair_reply or "")
    state = ((base.get("plan") or {}).get("design") or {}).get("salvage")
    if isinstance(state, dict) and isinstance(state.get("raw"), dict):
        want = [w["text"] for w in piece["type"]["words"]]
        merged = SV.repair_flat(obj, SimpleNamespace(pending=state.get("pending") or [], want=want))
        raw = dict(state["raw"])
        raw.update(merged)
        final, rep = safe_resolve(None, piece, piece, ana, lk, "rules", raw=raw, from_repair=[k for k in merged if not k.startswith("@")], model=model)
    else:                                  # a piece_json that did not come from parse(): the repair reply is resolved against it as a whole
        final, rep = safe_resolve(obj, base, piece, ana, lk, "parse result", model=model)
    return dict(piece_json=PR.piece_json(final), report=report_text("repair", rep, "repair reply: " + (f"{len(repair_reply or '')} chars" if repair_reply else "none") + "\n" + piece_summary(final)),
                piece=final, fields=rep.get("fields"))
