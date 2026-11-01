"""piece_brief: level B of the open-model par work (20_open-model-design/v003): BRIEF -> PIECE JSON, the artist level.

Claude wrote `15_fusion/pieces/P*.json` from a brief (15_fusion/PIECES.md, 15_fusion/v004/BRIEFS.md: the shared rules at the top + the per-piece paragraph) and the
repo docs. This module gives an open model the same inputs and checks what comes back against the engine and against Claude's JSON for the same brief:

    load_brief(pid)                 -> dict(shared rules, brief paragraph, words, music analysis, stills, Claude's piece)
    build_prompt(pid, variant)      -> (prompt text, image paths)   variant = the round's context (see VARIANTS)
    reply_to_piece(obj, brief)      -> (piece, fixes, errors)       the reply (idea, system, stills, shots, type, compose) -> a piece JSON the engine reads
    check_piece(piece, ana)         -> dict                          engine checks: type scene dry-run, timing, per-shot recipe build, every op executed on a tiny
                                                                     synthetic scene (CPU, 54x96, 3 frames) -> the first error per shot
    groups(piece) / compare(...)    -> the per-field-group summaries and the numeric rubric of the six gaps (a guide, not the judge)

The reply schema is deliberately the model's side of the work: it chooses the idea, the shot prompts, the words' fonts / modules / axes / camera, the timing as
`hit` / `hold_s` (the code compiles t0 / t1 with the engine's own plan_timing: models cannot do that arithmetic, BUILT.md), and the compose op chains (building blocks,
tools/fusion/compose_blocks.py). Everything the engine owns (seeds, sizes, frames of the footage, flags of the passes a chain reads, state keys, the legibility guard)
is filled by code.
"""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import compose_blocks as CB   # noqa: E402
from tools.fusion import design_options as DO   # noqa: E402  r7: shot transitions (compose.cuts)
from tools.fusion import design_step as DS      # noqa: E402
from tools.fusion import stable_type as ST     # noqa: E402  centred, still, on the beat
from tools.fusion import piece_system as PS     # noqa: E402  r9: the seeded piece system (module, io pair, faces, palette, camera, yaw band) written into the prompt, outliers folded into it
from tools.fusion import plan_rules as PR       # noqa: E402
from tools.fusion import qwen_piece as QP       # noqa: E402
from tools.fusion import recipe_eval as RE      # noqa: E402
from tools.fusion import why_fields as WF       # noqa: E402

FPS = 24
_CTX_LOCK = threading.RLock()        # qwen_piece keeps ONE in-graph piece context (QP._CTX["_app"]): parses in threads must take turns
FUSION = ROOT / "15_fusion"
STOP_WORDS = {"THE", "A", "AN", "OF", "AND", "TO", "IN", "ON", "IS", "AT", "WE", "IT", "OR"}

# --------------------------------------------------------------------------------------------- the test briefs
# words = the lines the brief names, in order (P06: the three chord chapters' pairs). ana = the analysis the real run used. stills = the footage of the brief
# (the first frame / still; the model sees them as the image input, never Claude's type). key / character = what the brief's "Music" line says.
BRIEFS = {
    "P06": dict(src="15_fusion/PIECES.md", words=["STILL", "SEARCHING", "A LIGHT", "AHEAD", "KEEP", "GOING"], ana="15_fusion/v003/local/pieces/P06/analysis.json",
                stills=["15_fusion/v002/out/motion-graphic_fusion_v002_P06_still_A_1_00001_.png", "15_fusion/v002/out/motion-graphic_fusion_v002_P06_still_B_1_00001_.png",
                        "15_fusion/v002/out/motion-graphic_fusion_v002_P06_still_C_1_00001_.png"],
                key="C minor", character="a rhythmic lead track (user audio), instrumental, 160 bpm, chords Cm 0-6 s, Ab 6-12 s, Gm 12-18 s", claude="15_fusion/pieces/P06.json",
                title="Song chords", mood="kinetic and emotional"),
    "P10": dict(src="15_fusion/PIECES.md", words=["WHO", "WAS", "SHE"], ana="15_fusion/v002/local/pieces/P10/analysis.json",
                stills=["15_fusion/v002/ingredients/P10/motion-graphic_fusion_v002_P10_still_A_1_00001_.png"], key="B minor",
                character="trip-hop / film-noir instrumental: dusty swung breakbeat, upright bass, muted trumpet, smoky Rhodes, vinyl crackle, cool and tense; dropouts at 4.9 / 6.6 / 10.1 / 14.7 s",
                claude="15_fusion/pieces/P10.json", title="Noir Step", mood="kinetic noir"),
    "P11": dict(src="15_fusion/v004/BRIEFS.md", words=["THE TIDE", "KEEPS", "WHAT WE", "LET GO"], ana="15_fusion/v004/local/pieces/P11/analysis.json",
                stills=["15_fusion/v004/out/motion-graphic_fusion_v004_P11_s1_1_00001_.mp4"], key="F major",
                character="felt piano with a sustained string pad and cello root, slow swell; calm", claude="15_fusion/pieces/P11.json", title="Tidewater", mood="slow, filmic, B/W"),
    "P14": dict(src="15_fusion/v004/BRIEFS.md", words=["RISE", "UP", "INTO", "THE LIGHT"], ana="15_fusion/v004/local/pieces/P14/analysis.json",
                stills=["15_fusion/v004/local/pieces/P14/mp/s1.mp4", "15_fusion/v004/local/pieces/P14/mp/s2raw.mp4"], key="C minor",
                character="bright house-leaning groove, four-on-the-floor, claps, airy pads", claude="15_fusion/pieces/P14.json", title="Lift", mood="kinetic, a dancer"),
}


def _section(text: str, pid: str) -> tuple[str, str]:
    """(shared header = everything before the first '## P', the '## <pid> ...' section)."""
    parts = re.split(r"(?m)^(?=## P\d+)", text)
    head = parts[0].rstrip()
    sec = next((p.rstrip() for p in parts[1:] if p.startswith(f"## {pid} ")), "")
    return head, sec


def load_brief(pid: str) -> dict:
    b = dict(BRIEFS[pid])
    text = (ROOT / b["src"]).read_text(encoding="utf-8")
    head, sec = _section(text, pid)
    ana = json.loads((ROOT / b["ana"]).read_text(encoding="utf-8"))
    ana.setdefault("nframes", int(math.ceil(float(ana["cut"]) * FPS - 1e-6)))
    ana.setdefault("fps", FPS)
    claude = json.loads((ROOT / b["claude"]).read_text(encoding="utf-8"))
    return dict(b, pid=pid, shared=head, brief=sec, ana=ana, claude_piece=claude)


def music_text(b: dict) -> str:
    ana = b["ana"]
    cut = float(ana["cut"])
    downs = [round(float(d), 2) for d in ana["downbeats"] if float(d) < cut - 0.2]
    beats = [round(float(x), 3) for x in ana["beats"] if float(x) < cut - 0.2]
    lines = [f"- length {cut:g} s ({int(ana['nframes'])} frames at 24 fps), tempo {float(ana['bpm']):.1f} bpm, beat {60.0 / float(ana['bpm']):.3f} s, bar {float(ana['bar_len']):.3f} s (4/4)",
             f"- key: {b['key']}", f"- character: {b['character']}; mood: {b['mood']}",
             f"- downbeats (s): {', '.join(f'{d:g}' for d in downs)}", "- beats (s): " + ", ".join(f"{x:g}" for x in beats)]
    cb = QP.chord_blocks(ana)
    if cb:
        lines.append("- chord blocks (start-end s: chord): " + "; ".join(f"{a:g}-{c:g}: {n}" for a, c, n in cb))
    do = [d for d in QP.dropouts(ana) if d[0] + d[1] < cut - 0.1]
    lines.append("- dropouts (the music nearly stops; start s, length s): " + (", ".join(f"{a:g} ({c:g})" for a, c in do) if do else "none inside the piece"))
    lines.append(f"- envelope curves usable in compose args as {{\"curve\": \"sub|low|mid|high|rms|onset\", \"lo\": a, \"hi\": b}}; the audio fades out over the last {float(ana.get('fade', 0.5)):g} s")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- the prompt
def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _runner_schema() -> str:
    """the piece JSON schema the engine accepts = the docstring of tools/fusion/piece_runner.py (what Claude read) + the type spec docstring of spec.py."""
    from tools.fusion import piece_runner, spec
    return ("=== PIECE JSON, as the engine reads it (tools/fusion/piece_runner.py) ===\n" + (piece_runner.__doc__ or "").split("Piece JSON (all paths relative to the motion-graphic repo root):")[-1].strip() +
            "\n\n=== TYPE SPEC (tools/fusion/spec.py) ===\n" + (spec.__doc__ or "").strip())


def _visual_decisions() -> str:
    t = _read("16_app/PLAN.md")
    i = t.index("## Visual decisions")
    j = t.index("\n## ", i + 5)
    return t[i:j].strip()


def system_for(b: dict) -> dict:
    """the piece system of a brief (r9): seeded by the brief id (the harness has no run seed), tied to the brief's mood and the music's tempo."""
    return PS.choose(PS.seed_of(b["pid"]), b.get("mood", ""), (b.get("ana") or {}).get("bpm"))


def reply_schema(b: dict, why: bool = False, transitions: bool = False, system: dict | None = None) -> str:
    n = len(b["words"])
    tt = QP.tempo_texts((b.get("ana") or {}).get("bpm") if system else None, system, PS.seed_of(b["pid"]))
    w_idea = WHY_IDEA if why else ""
    w_sys = WHY_SYS if why else ""
    w_shot = WHY_SHOT if why else ""
    w_word = WHY_WORD if why else ""
    w_def = WHY_CHAIN if why else ""
    w_sh = WHY_CHAIN_SHOT if why else ""
    cuts_line = (("  " + DO.schema_cuts(system).replace("\n", "\n  ") + ",\n") if transitions else "")
    return f"""Reply with ONE JSON object and nothing else (no markdown fence, no commentary). The code turns it into the piece JSON above; you decide the creative content.
{{
 "idea": "<one or two sentences: the ONE filmable idea, the camera move chosen FOR the type of footage, and how the words ride it>",{w_idea}
 "system": {{"rule": "<the one rule that organises the type of the whole piece>", "roles": {{"<role name>": "<font file(s) and why>"}}, "progression": "<what changes from the first word to the last>"{w_sys}}},
 "stills": [{{"id": "A", "prompt": "<photographic still prompt: subject, place, light, framing, lens>"}}, {{"id": "B", "angle_of": "A", "framing": "<close-up | low | profile | top-down | wide: the SAME subject and place from another camera; the code writes the prompt>"}}],
 "shots": [{{"id": "s1", "first": "<the id of a still of its own: every shot starts from its own still, never from another shot's last frame>", "until": <piece time in s where this shot ends; the last shot ends at the piece length; cuts sit on downbeats>,
            "prompt": "<what happens in THIS shot and ONE camera move, concrete; later shots progress the earlier ones>",{w_shot}
            "masktrack": {{"style": "rise", "start": <s>, "subject": "<the subject>"}}}}],        // masktrack optional
 "type": {{
  "mood": "<word>", "energy": <0..1>, "palette": {{"bg": [r,g,b], "fg": [r,g,b], "accents": [[r,g,b], ...]}},       // integers 0..255
  "camera": "<one of: {', '.join(QP._rot(QP.CAMERAS, PS.seed_of(b["pid"])))}>", "camera_params": {{"read_dps": <{tt['read'].split(' ')[0]}>, "intro_deg": <{tt['intro'].split(' ')[0]}>, "fast_dps": <{tt['fast'].split(' ')[0]}>, "elev": <0..10>, "bob": <{tt['bob']}>}},
  "transitions": "auto" | [<{n - 1} entries, one per gap: {', '.join(QP.TRANSITIONS)} or null>],
  "line": {{"px": <2..8>}},
  "words": [      // EXACTLY {n} entries, in order, text exactly: {' / '.join(b['words'])}
   {{"text": "..."{w_word}, "lines": ["optional line breaks, joined without spaces they spell the word"], "font": "<font file from the FONTS list, exact>", "module": "<one of: {', '.join(QP.WORD_MODULES)}>",
     "axes": {{"<axis tag of THAT font>": <number> | [<lo>, <hi>, "beat"]}},
     "hit": <s: a downbeat or beat where the word finishes forming>, "hold_s": <s the finished word stays still>, "form_s": <{tt['form'] if system else '0.45..2.0'}>, "exit_s": <{tt['exit'] if system else '0.5..1.5'}>,
     "form_in": "<{', '.join(QP._rot(QP.IO_STYLES, 1))}: each word enters differently>", "form_out": "<same list>", "entry": "<optional, the side the strands of a `lines` word come from: {', '.join(QP._rot(QP.OP.ORDER, 1))}>",
     "placement": {{"yaw": <{tt['yaw'] if system else '-25..25'}>, "pitch": <deg>, "roll": <deg>, "dx": <frame fraction>, "dy": <frame fraction>, "wf": <0.3..1.0>, "hf": <0.15..1.0>}},
     "screen_offset": [<x>, <y>]}} ]
 }},
 "compose": {{
  "crossfade": <seconds, or a list with one entry per boundary between shots; 0 = hard cut>,
  "holds": [[<time s>, <freeze length s>], ...],        // optional freeze-frames (at most 6)
{cuts_line}  "default": {{{w_def}"ops": [{{"op": "<block>", "out": "<buffer name>", "args": {{...}}}}, ...]}},        // the chain every shot without its own entry uses
  "shots": {{"<shot id>": {{{w_sh}"ops": [...]}}}}                                                          // optional per-shot chains
 }}
}}
The last op of every chain writes `image`. Every `hit` is a downbeat or beat of the MUSIC block; the code places the words one after the other from your hit / hold_s / form_s / exit_s.{WHY_RULES if why else ""}"""


# r6: a short reason per decision (the user wants to see WHY the model chose its route). The keys are added to the same schema; the code ignores them when it compiles the piece.
WHY_IDEA = '\n "idea_why": "<why this idea for THIS music and footage>",'
WHY_SYS = ', "why": "<why this one rule for these words>"'
WHY_SHOT = '\n            "why": "<why this place, this camera move, this continuation>",'
WHY_WORD = ', "why": "<why this font role, module, form in/out and hit for THIS word>"'
WHY_CHAIN = '"why": "<why this chain for this footage and music>", '
WHY_CHAIN_SHOT = '"why": "<why>", '
WHY_RULES = """

REASONS. Next to every decision write a short `why` (one or two sentences, at most ~35 words), specific to THIS piece: name the music moment (a chord change, a dropout, the tempo), the footage (what is in the shot or the still) or the word it serves. Say what the choice does for this piece, not what the rule says: never paste or paraphrase a sentence of the reference docs ("behind puts the type behind the subject" is a rule, not a reason). Two decisions never share one reason.
 - `idea_why` (top level, next to `idea`): why THIS idea for THIS music and footage.
 - `system.why`: why this one rule for these words.
 - `shots[].why`: why this place, this ONE camera move, and (when the shot shows the same place as an earlier one: an angle still) why this camera on that place instead of a new place.
 - `type.words[].why`: why this font role, module, form_in / form_out, and why the hit lands where it does (the music moment it sits on) for THIS word.
 - `compose.default.why` and `compose.shots.<id>.why`: why this chain (what it does to this footage, which timeline answers which music event)."""


TASK = ("You are the artist-designer of MOTNGRFX, a pipeline that makes short vertical music videos with kinetic typography over generated footage. Below are the brief for ONE piece, "
        "the shared rules every piece follows, the music analysis, the repo's reference docs and the piece JSON the engine accepts. Write the piece: the footage idea with its still and shot "
        "prompts, the type design (words, fonts, modules, axes, camera, timing, palette) and the compose op chains. Your output is judged against the piece the lead designer wrote from this "
        "same brief, so aim for the same level of craft: one idea carried through everything, not a collection of effects.")

# what each round adds to the context. r0 = what Claude read (brief, shared rules, visual decisions, module menu, piece schema, op docs, fonts, music) and nothing else.
VARIANTS = {
    "r0": dict(principles=False, syntax=False, selfcheck=False, menu=True, visual=True),
    "r1": dict(principles=True, syntax=True, selfcheck=False, menu=True, visual=True),
    "r2": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True),
    "r3": dict(principles=True, syntax=True, selfcheck=True, menu=False, visual=False),       # ablation: the module menu and the user's review notes dropped (-7 k tokens)
    "r5": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True),          # r4 + shot length follows the tempo (principle 9): the FINAL prompt
    "r5a": dict(principles=True, syntax=True, selfcheck=True, menu=False, visual=False),       # the same without the module menu / visual notes (the first r5 run fell back to this by accident)
    "r4": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True),          # r2 + the shot-structure principle (9) and check; the files are read at build time, r4 is the final prompt
    "r6a": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True, why=True, p9="r4"),   # r4 + a `why` per decision (thinking off); principle 9 as r4 had it (no tempo rule): r6 differs from r4 by the why fields only
    "r6b": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True, why=True, p9="r4"),   # the same prompt as r6a, thinking ON (design_par.ROUND_OPTS)
    "r7": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True, why=True, p9="r4", transitions=True),   # r6a (thinking off, whys) + the transitions reference, menu and `compose.cuts` in the schema
    "r8": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True, why=True, p9="r4", transitions=True, defaults=True),   # r7 + the clear-text / one-type-system / designed-cut / camera-variety defaults told up front (guide sections 20-21)
    "r9": dict(principles=True, syntax=True, selfcheck=True, menu=True, visual=True, why=True, p9="r4", transitions=True, defaults=True, system=True),   # r8 + the seeded piece system (diversity round 2026-10-29): module, io pair, font shortlist, palette family, camera, yaw band, beat-based timing
}


def _guide_section(num: int) -> str:
    """section `num` ('## 20. ...') of the open-model design guide, heading included."""
    txt = _read("16_app/DESIGN_GUIDE_open_model.md")
    m = re.search(rf"^## {num}\. .*?(?=^## \d+\. |\Z)", txt, re.S | re.M)
    return m.group(0).strip() if m else ""


DEFAULTS_EXTRA = ("Every shot starts from a still of its own (`first` is a still id, never `last:<id>`: a shot that starts from another shot's last frame carries its drift and its framing forward). "
                  "Either several shots, each with its own still (a fresh still for a new place; for the same subject from another camera a still with `angle_of` and a `framing`), or ONE single shot for the whole piece. "
                  "A shot prompt names no text, letters, words or signs (the type is drawn by the engine, never by the footage model). Words are drawn flat.")


def _extra(name: str) -> str:
    p = ROOT / "16_app" / name
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


def build_prompt(pid: str, variant: str = "r0") -> tuple[str, list[str]]:
    b = load_brief(pid)
    v = VARIANTS[variant]
    ana = b["ana"]
    seed = 1
    sysm = system_for(b) if v.get("system") else None
    parts = [TASK, "=== SHARED RULES (top of the brief file) ===\n" + b["shared"], "=== THE BRIEF ===\n" + b["brief"],
             f"Words, in order, each shown once: {' / '.join(b['words'])}", "=== MUSIC ANALYSIS (the grid your hits must land on) ===\n" + music_text(b),
             ("=== VISUAL DECISIONS (the user's review notes, 16_app/PLAN.md) ===\n" + _visual_decisions()) if v.get("visual", True) else "",
             ("=== MODULE MENU (16_app/MODULES_MENU.md) ===\n" + _read("16_app/MODULES_MENU.md").strip()) if v.get("menu", True) else "",
             "=== WORD MODULES, STYLES, TRANSITIONS, CAMERAS ===\n" + QP.modules_block(),
             ("=== YOUR SYSTEM FOR THIS PIECE ===\n" + PS.prompt_text(sysm)) if sysm else "",
             "=== FONTS (the only fonts you may use" + (": your system's shortlist, use at least three different faces" if sysm else "") + "; write the file name exactly; liked fonts only, a hard rule) ===\n" + QP.fonts_block(seed, only=sysm["fonts"] if sysm else None),
             _runner_schema(),
             "=== COMPOSE OPS: the building blocks you may chain (signature + doc from tools/fusion/compose.py) ===\n"
             "Buffers readable before any op has made them: $footage (the shot), $matte (the subject cut-out of the shot, 0..1), $depth, $flow and $flow_m (footage motion; give the shot its "
             "motion by reading them), $type_matte / $type_line / $type_depth / $type_beauty (the type passes, aligned to piece time), $frame_idx, $t, $u. Every op writes its result to `out`; a later "
             "op reads it as \"$out\". The final image is the buffer `image`.\n" + CB.menu(float(ana["cut"]))]
    if v["syntax"]:
        parts.append(_extra("PIECE_REFERENCE_syntax.md"))
    if v["principles"]:
        parts.append(_extra("PIECE_REFERENCE_principles.md"))
    if v.get("defaults"):
        parts.append("=== DEFAULTS THE CODE ENFORCES: CLEAR TEXT, ONE TYPE SYSTEM, A DESIGNED CUT AT EVERY PLACE CHANGE, A DIFFERENT CAMERA IN EVERY SHOT (r8) ===\n" + _guide_section(20) + "\n\n" +
                     _guide_section(21) + "\n\n" + DEFAULTS_EXTRA)
    if v.get("transitions"):
        parts.append(_extra("PIECE_REFERENCE_transitions.md"))
        parts.append("=== SHOT TRANSITIONS: THE MENU (tools/fusion/design_options.py) ===\n" + (DO.drawn_menu_text(sysm) or DO.MENU_TEXT))
    parts.append("=== OUTPUT ===\n" + reply_schema(b, bool(v.get("why")), bool(v.get("transitions")), sysm))
    if v["selfcheck"]:
        parts.append(_extra("PIECE_REFERENCE_selfcheck.md"))
    parts.append(f"Piece length {float(ana['cut']):g} s, vertical 9:16, 24 fps. Design seed {seed}. Reply with the JSON object only.")
    prompt = "\n\n".join(x for x in parts if x)
    if v.get("p9") == "r4":      # controlled comparison with r4: principle 9 as r4 had it (the tempo rule came after r4); the text is the one saved in r4's prompts
        f = ROOT / "20_open-model-design" / "v003" / "review" / "r4_principle9.txt"
        i = prompt.find("9. SHOT STRUCTURE FOLLOWS THE STORY AND THE TEMPO")
        if i >= 0 and f.is_file():
            j = prompt.find("\n\n", i)
            prompt = prompt[:i] + f.read_text(encoding="utf-8") + (prompt[j:] if j >= 0 else "")
    return prompt, [str(ROOT / s) if not Path(s).is_absolute() else s for s in b["stills"]]


def stills_image(paths: list[str], out: Path, tile_w: int = 270) -> Path:
    """one image with the brief's footage (stills side by side; a clip contributes its first frame)."""
    import cv2
    import numpy as np
    tiles = []
    for p in paths:
        p = str(p)
        if p.lower().endswith((".mp4", ".mkv")):
            cap = cv2.VideoCapture(p)
            ok, fr = cap.read()
            cap.release()
            if not ok:
                continue
        else:
            fr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        h = int(round(fr.shape[0] * tile_w / fr.shape[1]))
        t = cv2.resize(fr, (tile_w, h), interpolation=cv2.INTER_AREA)
        tiles.append(QP._label(t, f"footage {len(tiles) + 1}"))
    if not tiles:
        raise FileNotFoundError("no footage images for the brief")
    h = max(t.shape[0] for t in tiles)
    row = np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# --------------------------------------------------------------------------------------------- reply -> piece
def _snap(span_frames: int) -> int:
    return min(PR.snap8(span_frames), 385)


def _skeleton(obj: dict, b: dict, fixes: list, errors: list) -> dict:
    """the locked scaffolding of a piece from the reply's stills + shots: ids, frames, spans, seeds, sizes; type words are placeholders until normalize_design."""
    ana = b["ana"]
    cut = float(ana["cut"])
    comp_in = obj.get("compose") if isinstance(obj.get("compose"), dict) else {}
    shots_in = [s for s in (obj.get("shots") or []) if isinstance(s, dict)]
    if not shots_in:
        errors.append("shots: at least one shot is needed")
        shots_in = [{"id": "s1", "first": "A", "until": cut, "prompt": ""}]
    cf = comp_in.get("crossfade", 0)
    if isinstance(cf, list):
        cfl = [max(float(x), 0.0) if isinstance(x, (int, float)) else 0.0 for x in cf]
    else:
        cfl = [max(float(cf), 0.0) if isinstance(cf, (int, float)) and not isinstance(cf, bool) else 0.0] * (len(shots_in) - 1)
    cfl = (cfl + [0.0] * len(shots_in))[:max(len(shots_in) - 1, 0)]
    ids, prev_until, shots, seen = [], 0.0, [], set()
    for k, s in enumerate(shots_in):
        sid = re.sub(r"[^A-Za-z0-9_]", "", str(s.get("id") or f"s{k + 1}")) or f"s{k + 1}"
        if sid in seen:
            sid = f"{sid}_{k + 1}"
        seen.add(sid)
        try:
            until = float(s.get("until"))
        except (TypeError, ValueError):
            until = cut * (k + 1) / len(shots_in)
            fixes.append(f"shots[{k}].until missing: {until:g} s")
        if k == len(shots_in) - 1 and abs(until - cut) > 0.05:
            fixes.append(f"shots[{k}].until {until:g} -> the piece length {cut:g}")
            until = cut
        until = min(max(until, prev_until + 1.0), cut)
        xf = int(round(cfl[k - 1] * FPS)) if k else 0
        span = int(round((until - prev_until) * FPS)) + xf
        frames = _snap(span + 1)
        if span + 1 > 385:
            errors.append(f"shots[{k}] lasts {span / FPS:.1f} s, longer than the 385-frame (16 s) i2v shot: split it (chain with first = last:<id>)")
        first = str(s.get("first") or ("A" if k == 0 else f"last:{ids[-1]}"))
        sh = {"id": sid, "kind": "i2v", "first": first, "frames": frames, "seed": 100 + k, "size": [1088, 1920], "prompt": re.sub(r"\s+", " ", str(s.get("prompt") or "")).strip()}
        if frames != span:
            sh["span"] = span
        mt = s.get("masktrack")
        if isinstance(mt, dict) and mt.get("style") in ("rise", "rain", "unravel", "pull_apart"):
            sh["masktrack"] = {"style": mt["style"], "start": mt.get("start", "auto"), "denoise": 1.0, "corridor": False, "subject": str(mt.get("subject") or "the subject")[:60]}
        shots.append(sh)
        ids.append(sid)
        prev_until = until
    stills = []
    for k, st in enumerate(x for x in (obj.get("stills") or []) if isinstance(x, dict)):
        stills.append({"id": str(st.get("id") or chr(65 + k)), "model": "moss", "seed": 10 + k, "size": [1088, 1920], "prompt": str(st.get("prompt") or ""), "negative": PR._STILL_NEG,
                       **{kk: st[kk] for kk in ("angle_of", "framing", "angle_fallback") if kk in st}})
    need = {s["first"] for s in shots if not s["first"].startswith("last:")}
    have = {s["id"] for s in stills}
    for sid in sorted(need - have):
        errors.append(f"shots: first = {sid!r} but there is no still with that id")
    for s in shots:
        if s["first"].startswith("last:") and s["first"][5:] not in ids[:ids.index(s["id"])]:
            errors.append(f"shot {s['id']}: first = {s['first']!r} names a shot that does not come earlier")
    words = [{"text": w, "font": "Anybody[wdth,wght].ttf", "module": "lanes", "t0": 0.0, "t1": 1.0} for w in b["words"]]
    piece = {"id": b["pid"] + "_par", "slug": b["pid"].lower() + "-par", "fps": FPS, "audio": {"src": "", "bpm": int(round(float(ana["bpm"]))), "cut": cut, "fade": float(ana.get("fade", 0.5))},
             "type": {"seed": 1, "palette": {"bg": [8, 8, 8], "fg": [235, 235, 235], "accents": [[235, 235, 235]]}, "words": words}, "type_passes": ["matte"], "stills": stills,
             "shots": shots, "compose": {"size": [1080, 1920], "crossfade": cfl if len(cfl) != 1 or len(shots) == 2 else cfl, "crf": 16}, "signoff": {"text": "", "off": True},
             "quality": "final", "format": "9:16"}
    return piece


def _brief_seed(b: dict) -> int:
    """a seed per piece (the brief's id), so the entry rotation of openings.enforce differs from piece to piece (the app uses the piece's own seed)."""
    import zlib
    return zlib.crc32(str(b.get("pid", "")).encode()) % 97


def _no_chain(obj: dict, fixes: list) -> dict:
    """2026-10-29 (user: "why do all these videos use the last frame to start the next video's first frame? we should either be doing different shots completely or a single long prompted
    video"): no shot starts from another shot's last frame. A `first: last:<id>` (or a missing `first` on a later shot) becomes a NEW still of its own: an ANGLE still of the place's first
    still (`angle_of`, `framing` read from the shot prompt or rotated; tools/fusion/angle_still.py: until the graph wires the MOSS edit it is a fresh still of the same subject description
    with a different framing). A still the model wrote with `angle_of` gets the same treatment (its prompt is the code's). Every conversion is logged."""
    from tools.fusion import angle_still as AS
    from tools.fusion import shot_camera as SC
    shots = [x for x in (obj.get("shots") or []) if isinstance(x, dict)]
    stills = [dict(x) for x in (obj.get("stills") or []) if isinstance(x, dict)]
    if not shots:
        return obj
    by_id = {str(st.get("id")): st for st in stills}
    used = set(by_id)
    root_of: dict = {}
    for k, sh in enumerate(shots):
        sid = str(sh.get("id") or f"s{k + 1}")
        first = str(sh.get("first") or ("A" if k == 0 else "last:"))
        if first.startswith("last:") and k > 0:
            prev = shots[k - 1]
            prev_id = str(prev.get("id") or f"s{k}")
            ref = first[5:] or prev_id
            root = root_of.get(ref) or root_of.get(prev_id) or (str(shots[0].get("first") or "A"))
            new_id = next((c for c in "BCDEFGHIJ" if c not in used), None)
            if new_id is None:
                sh["first"] = root
                fixes.append(f"shots[{k}].first {first!r} -> {root!r}: no free still id left, it starts from the place's first still")
                root_of[sid] = root
                continue
            used.add(new_id)
            fr = AS.normalize_framing(SC.framing_of({"prompt": sh.get("prompt", "")}) or SC.FRAMING_ORDER[k % len(SC.FRAMING_ORDER)])
            base_p = str((by_id.get(root) or {}).get("prompt") or sh.get("prompt") or "")
            st = {"id": new_id, "angle_of": root, "framing": fr, "angle_fallback": True, "prompt": f"{AS.lead_framing(base_p, fr)}, the same subject and scene as above, {AS.framing_clause(fr)}".strip(", ")}
            stills.append(st)
            by_id[new_id] = st
            sh["first"] = new_id
            root_of[sid] = root
            fixes.append(f"shots[{k}].first {first!r} -> {new_id!r}: no shot starts from another shot's last frame; the same place from another camera gets its own angle still ({fr}) of {root}")
        else:
            root_of[sid] = first
    for st in stills:
        if st.get("angle_of") and not st.get("angle_fallback"):
            src = by_id.get(str(st["angle_of"]))
            if src is None or src.get("angle_of"):
                fixes.append(f"stills[{st.get('id')}].angle_of {st.get('angle_of')!r}: unknown or itself an angle still (an edit of an edit); a fresh still is used")
                st.pop("angle_of", None)
                st.pop("framing", None)
                continue
            fr = AS.normalize_framing(st.get("framing") or "wide")
            st["framing"], st["angle_fallback"] = fr, True
            st["prompt"] = f"{AS.lead_framing(src.get('prompt', ''), fr)}, the same subject and scene as above, {AS.framing_clause(fr)}".strip(", ")
    obj["stills"], obj["shots"] = stills, shots
    return obj


def reply_to_piece(obj, b: dict, plan_windows: bool = True) -> tuple[dict | None, list[str], list[str]]:
    """-> (piece | None, fixes, errors). errors = what makes the piece unusable; fixes = silent repairs (clamps, plumbing, dropped ops)."""
    fixes: list[str] = []
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, fixes, ["the reply is not a JSON object"]
    obj = WF.strip_why(obj)          # the `why` fields (r6) are for the reader, never for the engine
    obj = _no_chain(copy.deepcopy(obj), fixes)
    obj_in = copy.deepcopy(obj)
    ana = copy.deepcopy(b["ana"])
    sk = _skeleton(obj, b, fixes, errors)
    sk = PR.finalize_piece(sk, ana)
    pins_: dict = {}
    if len(sk["shots"]) > 1:      # the planning rule (a word on a place change, window_other): the app's design step runs it inside normalize_design; here it runs on the reply itself
        lay_ = sk["layout"]
        rows_ = [dict(id=s_["id"], start=round(lay_[s_["id"]]["start"] / FPS, 2), dur=round(lay_[s_["id"]]["frames"] / FPS, 2), frames=lay_[s_["id"]]["frames"],
                      first=s_.get("first", "")) for s_ in sk["shots"]]
        obj, wn_, pins_ = DO.plan_cuts(obj, rows_, ana, windows=plan_windows, system=b.get("system") if isinstance(b.get("system"), dict) else None)
        fixes += [f"compose.cuts: {x}" for x in wn_]
    t_in = obj.get("type") if isinstance(obj.get("type"), dict) else None
    if t_in is None:
        errors.append("missing object `type`")
        return None, fixes, errors
    t_in = copy.deepcopy(t_in)
    for k_, w_ in enumerate(t_in.get("words") or []):        # a hold under the app's floor is a repairable value (the app clamps it), not a failed reply
        if isinstance(w_, dict) and isinstance(w_.get("hold_s"), (int, float)) and w_["hold_s"] < QP.MIN_HOLD_IN:
            fixes.append(f"type.words[{k_}].hold_s {w_['hold_s']:g} -> {QP.MIN_HOLD_IN:g} (the floor)")
            w_["hold_s"] = QP.MIN_HOLD_IN
    tr_in = t_in.get("transitions")
    if isinstance(tr_in, list):        # repeated / unknown / wrong-count transitions are repaired field by field (the engine picks that gap), as the app's salvage does
        nw, seen = [], set()
        for k_, x_ in enumerate(tr_in[:max(len(b["words"]) - 1, 0)]):
            if x_ is None or (x_ in QP.TRANSITIONS and (x_ == "cut" or x_ not in seen)):
                nw.append(x_)
                seen.add(x_)
            else:
                nw.append(None)
                fixes.append(f"type.transitions[{k_}] {x_!r} -> null (unknown or already used)")
        nw += [None] * (max(len(b["words"]) - 1, 0) - len(nw))
        t_in["transitions"] = nw
    cp_in = t_in.get("camera_params")
    stub_comp = {"technique": "behind"}
    holds_in = (obj.get("compose") or {}).get("holds") if isinstance(obj.get("compose"), dict) else None
    if holds_in:
        stub_comp["holds"] = holds_in
    wo_ = [c_ for c_ in ((obj.get("compose") or {}).get("cuts") or []) if isinstance(c_, dict) and DO.CK.normalize(c_.get("cut")) == "window_other"] if isinstance(obj.get("compose"), dict) else []
    if wo_:                      # the word of a window_other cut spans the cut on purpose: the shot fit leaves its hold alone (qwen_piece.window_free)
        stub_comp["cuts"] = wo_
    from tools.fusion import design_defaults as DD
    sysm = b.get("system") if isinstance(b.get("system"), dict) else None
    if sysm:                                                  # r9: the piece's system: what is inside it stays, an outlier is folded into it (logged)
        PS.fold_type(t_in, sysm, fixes)
        PS.fold_words(t_in.get("words") or [], sysm, fixes, protected={c_.get("word") for c_ in wo_ if isinstance(c_.get("word"), int)}, seed=sysm.get("seed", 0))
    DD.one_module_system([w_ for w_ in (t_in.get("words") or []) if isinstance(w_, dict)], fixes)         # before the timing is compiled: a module changes a word's span
    DD.one_treatment([w_ for w_ in (t_in.get("words") or []) if isinstance(w_, dict)], fixes)             # ... and so do the io styles
    from tools.fusion import openings as OP
    OP.enforce([w_ for w_ in (t_in.get("words") or []) if isinstance(w_, dict)], fixes, (b.get("earlier_openings") or ()), _brief_seed(b), system=sysm)      # openings: no two words enter the same way, >= 2 styles, an `entry` named
    nd_in = {"concept": str(obj.get("idea") or "x"), "treatment_notes": "", "type": t_in, "compose": stub_comp}
    with _CTX_LOCK, DS.app_context(sk, ana) as name:
        d, fx, errs = QP.normalize_design(nd_in, name, pins=pins_)
        errs = [e for e in errs if not e.startswith("`concept`")]
        if d is not None and not errs:
            errs = QP.fit_scene(d, name, fx)           # the app's deterministic timing fit (the models overrun the music; code fits it and logs every change)
    fixes += fx
    errors += errs
    if errs and pins_ and plan_windows:          # a planned window_other holds a word across the cut: when the music cannot take that hold, the piece is planned without it (3D cuts stay)
        p2, f2, e2 = reply_to_piece(obj_in, b, plan_windows=False)
        if len(e2) < len(errors):
            return p2, f2 + ["compose.cuts: window_other dropped: the hold across the cut could not be fitted to the music (" + errs[0][:90] + ")"], e2
    if d is None:
        return None, fixes, errors
    seed = 1
    typ = DS._clean_type(d["type"], seed)
    # camera_params beyond the four normalize_design knows (fast_dps) and placement keys beyond yaw / wf / hf: kept within sane ranges (gap 3: Claude uses both)
    if isinstance(cp_in, dict) and typ.get("camera") == "beat_orbit":
        cp = dict(typ.get("camera_params") or {})
        if "fast_dps" in cp_in and isinstance(cp_in["fast_dps"], (int, float)):
            cp["fast_dps"] = round(min(max(float(cp_in["fast_dps"]), 0.0), 90.0), 2)
        if cp:
            typ["camera_params"] = cp
    for w_in, w in zip(t_in.get("words") or [], typ.get("words") or []):
        pl_in = w_in.get("placement") if isinstance(w_in, dict) and isinstance(w_in.get("placement"), dict) else {}
        for k, (lo, hi) in (("pitch", (-30, 30)), ("roll", (-30, 30)), ("dx", (-0.5, 0.5)), ("dy", (-0.5, 0.5))):
            if k in pl_in and isinstance(pl_in[k], (int, float)) and not isinstance(pl_in[k], bool):
                w.setdefault("placement", {})[k] = round(min(max(float(pl_in[k]), lo), hi), 3)
    piece = copy.deepcopy(sk)
    piece["type"] = typ
    piece["type"]["mood"] = typ.get("mood", "driving")
    PR.fill_type_timing(piece, ana)
    # compose: the model's chains, validated block by block
    comp_out, cfx = CB.normalize_compose(obj.get("compose"), [s["id"] for s in piece["shots"]], float(ana["cut"]))
    fixes += cfx
    c = piece["compose"]
    if comp_out.get("default"):
        c["default"] = comp_out["default"]
    if comp_out.get("shots"):
        c["shots"] = comp_out["shots"]
    rules_for = []
    for s in piece["shots"]:
        if not (comp_out.get("shots", {}).get(s["id"]) or comp_out.get("default")):
            rules_for.append(s["id"])
    if rules_for:
        pal = [QP._hex(typ["palette"].get("bg", [8, 8, 8])), QP._hex(typ["palette"].get("fg", [235, 235, 235]))] + [QP._hex(a) for a in typ["palette"].get("accents", [])]
        rec = PR.apply_recipe(PR.load_recipe("behind"), float(ana["cut"]), pal, size=c["size"])
        c.setdefault("default", rec["default"])
        fixes.append(f"compose: no valid chain for {', '.join(rules_for)} -> the rules' 'behind' chain is used there (the model's compose did not survive)")
        piece["plan"] = {"compose_source": "rules"}
    else:
        piece["plan"] = {"compose_source": "model"}
    if sysm:
        piece["plan"]["piece_system"] = copy.deepcopy(sysm)
    if d["compose"].get("holds"):
        c["holds"] = d["compose"]["holds"]
        c["holds_type"] = True
    # state keys: flow_wake keeps a state per call; every one gets its own
    for sid, chain in [("default", c.get("default"))] + list((c.get("shots") or {}).items()):
        for i, o in enumerate((chain or {}).get("ops") or []):
            if o["op"] == "flow_wake":
                o["args"]["state_key"] = f"{piece['id']}:{sid}:{o.get('out', 'image')}:{i}"
    # flags for the passes the chains read
    for s in piece["shots"]:
        need = RE.shot_needs(piece, s["id"])
        s["matte"] = bool(need["matte"]) or bool(s.get("masktrack"))
        s["depth"] = bool(need["depth"])
        s["flow"] = bool(RE.shot_needs_flow(piece, s["id"]))
        for k in ("matte", "depth", "flow"):
            if not s[k]:
                del s[k]
    piece["type_passes"] = RE.need_passes(piece)
    piece["idea"] = str(obj.get("idea") or "")
    piece["system"] = obj.get("system") if isinstance(obj.get("system"), dict) else {}
    piece["layout"] = PR.layout(piece)
    _apply_cuts(piece, obj, ana, fixes)
    _apply_defaults(piece, ana, fixes)
    return piece, fixes, errors


def _apply_defaults(piece: dict, ana: dict, fixes: list) -> None:
    """2026-10-29 (r8): the draft strips' qualities as defaults, the same functions the app's design step runs (tools/fusion/design_defaults.py): one type system, the size floor, clear holds."""
    from tools.fusion import design_defaults as DD
    from tools.fusion import text_guard as TG
    fixes += TG.fix_plan(piece)                   # the footage model draws no text of its own: shot / still prompts name none, the negatives say so
    words = piece["type"]["words"]
    prot = {i for i, w in enumerate(words) if w.get("window_source")}
    QP.diversify_fonts(piece["type"], fixes, 1, prot, only=((piece.get("plan") or {}).get("piece_system") or {}).get("fonts"))                                      # the app's font rules (>= 3 faces, one expressive) run in finish_design; the harness runs them too
    DD.enforce_type(words, fixes, protected=prot)
    ps_ = (piece.get("plan") or {}).get("piece_system")
    if isinstance(ps_, dict) and ps_.get("module"):                 # 2026-10-30: centred, still for >= 1 beat, stacked / rolled layout, held camera (the app's design step does the same)
        ST.enforce(piece["type"], ps_, ana.get("bpm"), fixes, protected=prot)
    try:
        chk = PR.dry_run(copy.deepcopy(piece))["words"]
        holds = [(float(c["hit"]), float(c["release"])) for c in chk]
    except Exception:       # noqa: BLE001  the engine's own error is reported by check_piece
        holds = []
    fixes += DD.enforce_compose(piece, holds)


def _apply_cuts(piece: dict, obj: dict, ana: dict, fixes: list) -> None:
    """r7: compose.cuts of the reply -> design_options.normalize_cuts + validate (boundary, word, fat face, fit, no repeat, direction) -> piece.compose.cuts + the
    words' window_source / exit / entry sides (DO.to_piece). Every change is a fix. A reply without `cuts` leaves the piece as it was."""
    comp_in = obj.get("compose") if isinstance(obj.get("compose"), dict) else {}
    cuts_in = comp_in.get("cuts")
    if cuts_in is None or len(piece["shots"]) < 2:
        return
    lay = piece["layout"]
    rows = [dict(id=s["id"], start=round(lay[s["id"]]["start"] / FPS, 2), dur=round(lay[s["id"]]["frames"] / FPS, 2), frames=lay[s["id"]]["frames"],
                 desc=s.get("prompt", ""), first=s.get("first", ""), matte=bool(s.get("matte"))) for s in piece["shots"]]
    try:
        chk = PR.dry_run(copy.deepcopy(piece))["words"]
    except Exception:       # noqa: BLE001  the engine's own error is reported by check_piece
        chk = []
    words = copy.deepcopy(piece["type"]["words"])
    for w, c in zip(words, chk):
        w["t0"], w["hit"], w["release"] = float(c["start"]), float(c["hit"]), float(c["release"])
    design = {"type": {"words": words}, "compose": {"cuts": copy.deepcopy(cuts_in)}}
    new, fx = DO.validate(design, dict(shots=rows, ana=ana, fill_3d=True, system=(piece.get("plan") or {}).get("piece_system")))
    fixes += [f"compose.cuts: {x}" for x in fx]
    for w, nw in zip(piece["type"]["words"], new["type"]["words"]):
        for k in ("font", "axes", "placement", "window_source", "window_ground"):
            if k in nw:
                w[k] = nw[k]
            elif k in ("axes", "window_source", "window_ground"):
                w.pop(k, None)
    cuts = (new.get("compose") or {}).get("cuts")
    if cuts:
        for note in DO.to_piece(piece, new, None):
            fixes.append(f"compose.cuts: {note}")


# --------------------------------------------------------------------------------------------- the engine check
SZ = (54, 96)            # synthetic compose size of the execution test (every op size argument is rewritten to it)
NSYN = 8


def _small(o, w, h):
    """op args with a literal [W, H] size argument rewritten to the synthetic size."""
    if isinstance(o, dict):
        return {k: ([w, h] if k == "size" and isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else _small(v, w, h)) for k, v in o.items()}
    if isinstance(o, list):
        return [_small(x, w, h) for x in o]
    return o


def _synthetic_type_dir(td: Path, n: int = NSYN) -> Path:
    import cv2
    import numpy as np
    W, H = SZ
    for p in ("matte", "line", "depth", "beauty"):
        (td / p).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        m = np.zeros((H, W), np.uint8)
        cv2.putText(m, "AB", (4 + i % 3, H // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 6)
        ln = cv2.Canny(m, 50, 150)
        dep = (np.tile(np.linspace(60, 200, W, dtype=np.uint8), (H, 1)) * (m > 0)).astype(np.uint8)
        bea = np.zeros((H, W, 3), np.uint8)
        bea[:] = (10, 10, 12)
        bea[m > 0] = (235, 235, 240)
        for name, im in (("matte", m), ("line", ln), ("depth", dep), ("beauty", bea)):
            cv2.imwrite(str(td / name / f"{i:05d}.png"), im)
    return td


def _synthetic_media(n: int = NSYN):
    import cv2
    import numpy as np
    W, H = SZ
    foot = np.zeros((n, H, W, 3), np.uint8)
    matte = np.zeros((n, H, W, 3), np.uint8)
    depth = np.zeros((n, H, W, 3), np.uint8)
    for i in range(n):
        g = np.tile(np.linspace(30, 220, H, dtype=np.uint8)[:, None], (1, W))
        foot[i] = np.stack([g, np.roll(g, 7 * i, axis=0), 255 - g], -1)
        cv2.ellipse(foot[i], (W // 2 + 2 * i, H // 2), (10, 26), 0, 0, 360, (200, 160, 120), -1)
        cv2.ellipse(matte[i], (W // 2 + 2 * i, H // 2), (10, 26), 0, 0, 360, (255, 255, 255), -1)
        depth[i] = np.stack([g, g, g], -1)
    flow = np.zeros((n, H // 2, W // 2, 2), np.float32)
    flow[..., 0] = 1.5
    flow[:, 10:24, 8:20, 1] = 2.0
    return foot, matte, depth, flow


def _others_for(q, sid, foot, matte):
    """window cuts read the neighbour shot ($next / $prev): the synthetic scene stands in for it (a flipped copy, so the two pictures differ)."""
    from tools.fusion import window_cut
    return window_cut.collect_others(q, sid, lambda s_: foot[..., ::-1].copy(), lambda s_: matte)


def check_piece(piece: dict, ana: dict, tmp: Path | None = None, exec_ops: bool = True) -> dict:
    """does the engine take this piece? type scene dry-run + timing, per shot: recipe build and (exec_ops) every op executed on a tiny synthetic scene."""
    import tempfile
    rep: dict = dict(type=dict(ok=False, error=None, timing_errors=[], words=[]), shots={})
    p = copy.deepcopy(piece)
    PR.fill_type_timing(p, ana)
    try:
        errs, chk = QP.timing_errors(p, ana)
        hard = [e for e in errs if "overlaps" in e]
        rep["type"].update(ok=bool(chk) and not hard, timing_errors=errs[:4], soft=[e for e in errs if e not in hard and not e.startswith("the engine's planned timeline")][:4], words=[(w["text"], round(float(w["hit"]), 2), round(float(w["release"]) - float(w["hit"]), 2)) for w in chk.get("words", [])])
        if not chk:
            rep["type"]["error"] = errs[0] if errs else "dry run failed"
    except Exception as e:      # noqa: BLE001
        rep["type"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    W, H = SZ
    q = copy.deepcopy(p)
    q["compose"] = _small(q["compose"], W, H)
    q["sizes"] = dict(q.get("sizes") or {}, compose=[W, H])
    q["compose"]["size"] = [W, H]
    q["layout"] = PR.layout(q)
    q["signoff"] = {"text": "", "off": True}
    _orig_call = RE.call_op

    def _named(entry, args):
        try:
            return _orig_call(entry, args)
        except Exception as e_:      # noqa: BLE001
            raise type(e_)(f"op {getattr(entry[0], '__name__', '?')}({', '.join(k for k in args)}): {e_}") from e_
    RE.call_op = _named
    td_ctx = tempfile.TemporaryDirectory() if tmp is None else None
    td = Path(tmp) if tmp is not None else Path(td_ctx.name)
    try:
        if exec_ops:
            _synthetic_type_dir(td / "type")
            foot, matte, depth, flow = _synthetic_media()
        done: dict = {}
        for s in q["shots"]:
            sid = s["id"]
            r = dict(build=False, exec=False, error=None, ops=0)
            sig = json.dumps([_chain_of(q, sid), bool(s.get("flow")), bool(s.get("depth")), bool(s.get("matte"))], sort_keys=True)
            if exec_ops and sig in done:
                rep["shots"][sid] = dict(done[sig], same_as=done[sig].get("first_id"))
                continue
            try:
                rec = RE.build_recipe(q, sid, type_dir=td / "type", cut=float(ana["cut"]))
                r["build"] = True
                r["ops"] = len(rec["ops"])
                if exec_ops:
                    n = int(rec["frames"])
                    for a in sorted({0, n // 2, max(n - 1, 0)}):
                        RE.run_compose(q, sid, type_dir=td / "type", footage=foot, matte=matte, depth=depth, flow=flow, frames=(a, a + 1), backend="numpy", threads=0, cut=float(ana["cut"]),
                                       curves={k: v for k, v in (ana.get("curves") or {}).items() if isinstance(v, list)},
                                       others=_others_for(q, sid, foot, matte))
                    r["exec"] = True
            except Exception as e:      # noqa: BLE001
                import traceback
                tb = traceback.extract_tb(e.__traceback__)
                where = f" [{tb[-1].name}:{tb[-1].lineno}]" if tb else ""
                r["error"] = f"{type(e).__name__}: {str(e)[:220]}{where}"
            rep["shots"][sid] = r
            done[sig] = dict(r, first_id=sid)
    finally:
        RE.call_op = _orig_call
        if td_ctx is not None:
            try:
                td_ctx.cleanup()
            except Exception:       # noqa: BLE001
                pass
    rep["ok"] = bool(rep["type"]["ok"] and rep["shots"] and all(v["build"] and (v["exec"] or not exec_ops) for v in rep["shots"].values()))
    return rep


# --------------------------------------------------------------------------------------------- field groups + rubric (a guide, not the judge)
def _chain_of(piece: dict, sid: str) -> list:
    c = piece.get("compose") or {}
    return ((c.get("shots") or {}).get(sid) or c.get("default") or {}).get("ops") or []


def _jacc(a: str, b: str) -> float:
    A, B = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(A & B) / len(A | B) if A | B else 0.0


CAMERA_WORDS = ("camera", "tracks", "orbit", "dolly", "push", "pull", "pan", "tilt", "follows", "glides", "sweep", "crane", "handheld", "drone")


def structure(piece: dict) -> dict:
    """the SHOT STRUCTURE of a piece: shots, places (shots that start from a still of their own), continuations (first = last:<id>) and a label: single | chain | cut | hybrid
    (places of several shots: a new place, then its shot continues it)."""
    shots = piece.get("shots") or []
    firsts = [str(s.get("first", "")) for s in shots]
    cont = sum(1 for f in firsts if f.startswith("last:"))
    places = len({f for f in firsts if not f.startswith("last:")})
    label = "single" if len(shots) <= 1 else ("chain" if places == 1 else ("cut" if cont == 0 else "hybrid"))
    return dict(structure=label, places=places, continues=cont)


def structure_vs(piece: dict, ref: dict) -> dict:
    """the model's shot structure next to the lead designer's: the same number of places, continuation used when the reference uses it, shots within one."""
    a, b = structure(piece), structure(ref)
    na, nb = len(piece.get("shots") or []), len(ref.get("shots") or [])
    return dict(shots=na, ref_shots=nb, places=a["places"], ref_places=b["places"], continues=a["continues"], ref_continues=b["continues"], label=a["structure"], ref_label=b["structure"],
                match=bool(a["places"] == b["places"] and (a["continues"] > 0) == (b["continues"] > 0) and abs(na - nb) <= 1), multi_shot_kept=bool(na >= 2 or nb < 2))


def groups(piece: dict, ana: dict) -> dict:
    """numeric summary per field group of a piece (Claude's or a model's), the same function for both."""
    t = piece.get("type") or {}
    ws = t.get("words") or []
    shots = piece.get("shots") or []
    comp = piece.get("compose") or {}
    cut = float(ana["cut"])
    downs = [float(d) for d in ana.get("downbeats") or []]
    beats = [float(x) for x in ana.get("beats") or []]
    out: dict = {}
    # 1 footage
    prompts = [s.get("prompt", "") for s in shots]
    pair = [_jacc(a, c) for i, a in enumerate(prompts) for c in prompts[i + 1:]]
    out["footage"] = dict(**structure(piece), shots=len(shots), prompt_words=round(statistics.mean(len(p.split()) for p in prompts), 1) if prompts else 0,
                          distinct=round(1 - (statistics.mean(pair) if pair else 0), 2), camera_in_prompt=sum(1 for p in prompts if any(w in p.lower() for w in CAMERA_WORDS)),
                          continuity=sum(1 for s in shots if str(s.get("first", "")).startswith("last:")), idea=bool(piece.get("idea")), stills=len(piece.get("stills") or []))
    # 2 system
    fonts = [w.get("font") for w in ws]
    mods = [w.get("module") for w in ws]
    nf = len(set(fonts))
    rec = sum(1 for i, f in enumerate(fonts) if f in fonts[:i])
    alt = all(fonts[i] != fonts[i + 1] for i in range(len(fonts) - 1)) if len(fonts) > 1 else False
    period = next((p for p in range(1, len(fonts)) if all(fonts[i] == fonts[i % p] for i in range(len(fonts)))), None) if len(fonts) > 1 else None
    out["system"] = dict(words=len(ws), faces=nf, recurring_words=rec, modules=len(set(mods)), forms=len(set((w.get("form_in"), w.get("form_out")) for w in ws)),
                         font_period=period if period and period < len(fonts) else None, alternates=alt, system_block=bool(piece.get("system")))
    # 3 motion
    cam = t.get("camera")
    cp = t.get("camera_params") or {}
    yaws = [(w.get("placement") or {}).get("yaw", 0) for w in ws]
    animated = [w for w in ws if any(isinstance(v, list) for v in (w.get("axes") or {}).values())]
    prog = 0
    for tag in {k for w in animated for k in (w.get("axes") or {})}:
        los = [(w.get("axes") or {}).get(tag) for w in ws if isinstance((w.get("axes") or {}).get(tag), list)]
        if len(los) > 2 and all(los[i][0] <= los[i + 1][0] for i in range(len(los) - 1)) and los[-1][0] > los[0][0]:
            prog += 1
    out["motion"] = dict(camera=cam, intro_deg=cp.get("intro_deg"), fast_dps=cp.get("fast_dps"), bob=cp.get("bob"), yaw_abs=round(statistics.mean(abs(y) for y in yaws), 1) if yaws else 0,
                         yaw_alternates=bool(len(yaws) > 1 and all(yaws[i] * yaws[i + 1] < 0 for i in range(len(yaws) - 1))), axes_animated=len(animated), axes_progression=prog,
                         transitions=("auto" if t.get("transitions") == "auto" else len([x for x in t.get("transitions") or [] if x])))
    # 4 timing
    hits = []
    holds = []
    for w in ws:
        hits.append(float(w.get("t0", 0)) + float(w.get("form_s", 0)))
        holds.append(float(w.get("t1", 0)) - float(w.get("t0", 0)) - float(w.get("form_s", 0)) - float(w.get("exit_s", 0)))
    gaps = [hits[i + 1] - hits[i] for i in range(len(hits) - 1)]
    out["timing"] = dict(hit_gap_cv=round(statistics.pstdev(gaps) / statistics.mean(gaps), 2) if len(gaps) > 1 and statistics.mean(gaps) else 0.0,
                         hold_cv=round(statistics.pstdev(holds) / statistics.mean(holds), 2) if len(holds) > 1 and statistics.mean(holds) else 0.0, hold_mean=round(statistics.mean(holds), 2) if holds else 0,
                         form_distinct=len({round(float(w.get("form_s", 0)), 2) for w in ws}), on_downbeat=sum(1 for w in ws if w.get("hit_on") == "bar"), freeze_holds=len(comp.get("holds") or []))
    # 5 compose
    chains = [(sid, _chain_of(piece, sid)) for sid in [s["id"] for s in shots]]
    all_ops = [o for _, ch in chains for o in ch]
    uniq = {json.dumps(ch, sort_keys=True) for _, ch in chains}
    out["compose"] = dict(chains=len(uniq), ops=max((len(ch) for _, ch in chains), default=0), distinct_ops=len({o["op"] for o in all_ops}), timelines=sum(CB.count_timelines(ch) for _, ch in chains) // max(len(chains), 1),
                          beat_curves=sum(1 for o in all_ops if "curve" in json.dumps(o.get("args"))), bg_ops=sum(1 for o in all_ops if o["op"] in ("blur", "fill", "crossfade", "refract", "split_region", "knockout")),
                          drawers=sum(1 for o in all_ops if o["op"] in CB.DRAWERS), source=(piece.get("plan") or {}).get("compose_source", "claude"))
    # 6 palette + lockups
    colour_tl = 0
    for o in all_ops:
        a = o.get("args") or {}
        for v in a.values():
            if isinstance(v, dict) and "timeline" in v and v["timeline"] and isinstance(v["timeline"][0][1], list):
                colour_tl += 1
    stop_alone = sum(1 for w in ws if str(w.get("text", "")).upper() in STOP_WORDS)
    out["palette_lockups"] = dict(accents=len((t.get("palette") or {}).get("accents") or []), colour_timelines=colour_tl, stopword_lockups=stop_alone,
                                  words_per_lockup=round(statistics.mean(len(str(w.get("text", "")).split()) for w in ws), 1) if ws else 0)
    return out


def rubric(g: dict, ana: dict) -> dict:
    """six 0..2 scores (0 = absent, 1 = partial, 2 = like the lead designer's pieces); documented in 16_app/DESIGN_GUIDE_open_model.md section 13."""
    f, s, m, t, c, p = (g[k] for k in ("footage", "system", "motion", "timing", "compose", "palette_lockups"))
    r = {}
    r["1_footage"] = (1 if f["camera_in_prompt"] >= max(f["shots"] - 1, 1) else 0) + (1 if f["shots"] == 1 or f["distinct"] >= 0.45 else 0)
    r["2_system"] = (1 if s["faces"] >= 3 and s["modules"] <= 3 else 0) + (1 if s["recurring_words"] >= 1 and (s["font_period"] or s["alternates"] or s["faces"] <= max(s["words"] // 2, 3)) else 0)
    r["3_motion"] = (1 if m["camera"] not in (None, "hold_drift") and (m["intro_deg"] or 0) >= 25 and (m["fast_dps"] or 0) >= 30 else 0) + (1 if m["yaw_abs"] >= 8 and m["axes_animated"] >= max(s["words"] // 2, 1) else 0)
    r["4_timing"] = (1 if t["hold_cv"] >= 0.1 or t["hit_gap_cv"] >= 0.1 else 0) + (1 if t["form_distinct"] >= 2 or t["freeze_holds"] >= 1 else 0)
    r["5_compose"] = (1 if c["source"] != "rules" and c["ops"] >= 7 and c["drawers"] >= 1 else 0) + (1 if c["timelines"] >= 2 and (c["bg_ops"] >= 1 or c["chains"] > 1) else 0)
    r["6_palette_lockups"] = (1 if p["colour_timelines"] >= 1 or p["accents"] >= 3 else 0) + (1 if p["stopword_lockups"] == 0 and p["words_per_lockup"] >= 1.0 else 0)
    r["total"] = sum(r.values())
    return r


# --------------------------------------------------------------------------------------------- craft checks (the rubric above only asks "is it there"; these ask "is it used the way the lead designer uses it")
def _musical_points(ana: dict) -> list[float]:
    cc = [float(c["t"]) if isinstance(c, dict) else float(c) for c in ana.get("chord_changes") or []]
    do = QP.dropouts(ana)
    pts = cc + [a for a, _ in do] + [a + b for a, b in do] + [0.0]
    downs = [float(d) for d in ana.get("downbeats") or []]
    return sorted(set(pts + [min([d for d in downs if d >= p - 0.05] or [p]) for p in pts]))


def _timeline_values(ops, args_keys=None):
    """every timeline dict found in the ops' args -> [(arg name, keys)]"""
    out = []

    def walk(o, name=""):
        if isinstance(o, dict):
            if "timeline" in o and isinstance(o["timeline"], list):
                out.append((name, o["timeline"], bool(o.get("step"))))
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, name)
    walk([x.get("args", {}) for x in ops])
    return out


def craft(piece: dict, ana: dict) -> dict:
    """six 0/1 craft checks (a guide, same function for Claude's piece and a model's): the lead designer's habits that a presence check misses."""
    t = piece.get("type") or {}
    ws = t.get("words") or []
    shots = piece.get("shots") or []
    from mograph import fonts_liked as FL
    role = {e["file"]: e["role"] for e in FL.LIKED}
    files = []
    for w in ws:
        try:
            files.append(FL.require_liked(w.get("font"))["file"])
        except Exception:       # noqa: BLE001
            files.append(str(w.get("font")))
    faces = len(set(files))
    expressive = any(role.get(f) in QP.EXPRESSIVE or f in ("Anybody[wdth,wght].ttf", "Ultra-Regular.ttf", "BadeenDisplay-Regular.ttf") for f in files)
    hits = [float(w.get("t0", 0)) + float(w.get("form_s", 0)) for w in ws]
    holds = [float(w.get("t1", 0)) - float(w.get("t0", 0)) - float(w.get("form_s", 0)) - float(w.get("exit_s", 0)) for w in ws]
    floor = sum(1 for h in holds if abs(h - QP.MIN_HOLD_IN) < 0.06)
    pts = _musical_points(ana)
    on_music = sum(1 for h in hits if pts and min(abs(h - p) for p in pts) <= 0.4)
    chains = {sid: _chain_of(piece, sid) for sid in [s["id"] for s in shots]}
    flips = 0
    colour_steps = 0
    for ch in chains.values():
        for name, keys, step in _timeline_values(ch):
            vals = {json.dumps(k[1]) for k in keys}
            if name in ("matte_gain", "plate_rgb", "front", "type_opacity", "fg", "alpha") and len(keys) >= 3 and len(vals) >= 2:
                flips += 1
            if isinstance(keys[0][1], list) and step and len(vals) >= 2:
                colour_steps += 1
    n_sect = len(ana.get("chord_changes") or []) + 1
    firsts = {s.get("first") for s in shots if not str(s.get("first", "")).startswith("last:")}
    distinct = len({json.dumps(ch, sort_keys=True) for ch in chains.values()})
    r = dict(hard_faces=faces >= 3, hard_expressive=bool(expressive), faces=faces,
             c1_hold_varies=bool(holds) and floor <= len(holds) / 2 and (statistics.pstdev(holds) / max(statistics.mean(holds), 1e-6) >= 0.1 if len(holds) > 1 else True),
             c2_hits_on_music=bool(ws) and on_music >= max(1, math.ceil(len(ws) / 2)), c3_behind_front_or_ground_flip=flips >= 1,
             c4_colour_per_section=colour_steps >= 1 and (colour_steps >= 1 if n_sect <= 2 else True), c5_fonts_hard_rule=faces >= 3 and bool(expressive),
             c6_chains_follow_scenes=(distinct >= 2) if len(firsts) >= 2 and len(shots) >= 2 else True, holds_at_floor=floor, hits_on_music=on_music, flips=flips, colour_steps=colour_steps)
    r["craft_total"] = sum(int(bool(r[k])) for k in ("c1_hold_varies", "c2_hits_on_music", "c3_behind_front_or_ground_flip", "c4_colour_per_section", "c5_fonts_hard_rule", "c6_chains_follow_scenes"))
    return r
