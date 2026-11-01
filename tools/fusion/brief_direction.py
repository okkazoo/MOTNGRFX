"""brief_direction: level A of the open-model par work (20_open-model-design/v003): DIRECTION -> BRIEF, the concept level.

The lead designer's briefs (15_fusion/v004/BRIEFS.md: `## P11 Tidewater - ...` then Words / Music / Footage / Weave) are what the app's concept step has to produce. The model gets only what a
user would say (title, mood, tempo, length, the words, a subject hint), the shared rules at the top of BRIEFS.md, the repo menus and the music grid, plus the FORMAT shown on three briefs that are not
in the test set. It writes a brief; the harness checks the format and scores the content against the lead designer's brief for the same slot (a guide, not the judge).

    python tools/app/design_par.py direction --round a0
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion import compose_blocks as CB   # noqa: E402
from tools.fusion import piece_brief as PB      # noqa: E402
from tools.fusion import qwen_piece as QP       # noqa: E402

DIRECTIONS = {
    "P06": "Title: Song chords. Mood: kinetic and emotional. 160 bpm, 18 seconds, vertical 9:16, a rhythmic instrumental lead track in C minor with three chord sections (Cm, Ab, Gm). "
           "Words: STILL / SEARCHING / A LIGHT / AHEAD / KEEP / GOING. Subject hint: a runner, three places, a journey from dark to light.",
    "P10": "Title: Noir Step. Mood: kinetic noir. 90 bpm, about 15 seconds, vertical 9:16, a trip-hop / film-noir instrumental (swung breakbeat, upright bass, muted trumpet, smoky Rhodes) with dropouts. "
           "Words: WHO / WAS / SHE. Subject hint: a woman in a trench coat in the rain at night, black and white, 1940s.",
    "P11": "Title: Tidewater. Mood: slow, filmic, black and white. 72 bpm, 16 seconds, vertical 9:16, felt piano with a string pad and cello, slow swell. "
           "Words: THE TIDE / KEEPS / WHAT WE / LET GO. Subject hint: a lone figure by the sea.",
    "P14": "Title: Lift. Mood: kinetic, a dancer. 124 bpm (the music is 120), 12 seconds, vertical 9:16, a bright house-leaning groove, four-on-the-floor, claps, airy pads. "
           "Words: RISE / UP / INTO / THE LIGHT. Subject hint: a contemporary dancer in an empty studio.",
}
EXAMPLES = ("P12", "P13", "P15")        # format examples (not test briefs)

FORMAT = """Write ONE brief in exactly this format and nothing else (no commentary, no code fence):
## <id> <Title> - <feel words> (<bpm> bpm, <length> s, <9:16 | 16:9>)
Words: <the lines, separated by " / ">.
Music: <the music, one line: instruments, mood, master>.
Footage: <the MOSS still prompt in quotes, then how many shots, what each shot is (one continuous camera move each) and where the cuts fall (on a downbeat)>.
Weave: <how the type lives IN the footage: behind / in front of which real elements, which compose ops and passes (sandwich, knockout, depth_occlude, flow_wake, contour_lines, tint_type, add_glow ...), how the lines
enter and exit, which transition (each used once), the camera of the type, what pulses on the beat, the colours>."""


def _brief_section(src: str, pid: str) -> str:
    text = (ROOT / src).read_text(encoding="utf-8")
    return PB._section(text, pid)[1]


def build_prompt(pid: str) -> str:
    b = PB.load_brief(pid)
    ex = "\n\n".join(_brief_section("15_fusion/v004/BRIEFS.md", e) for e in EXAMPLES)
    shared = PB._section((ROOT / "15_fusion/v004/BRIEFS.md").read_text(encoding="utf-8"), "P12")[0]
    return "\n\n".join([
        "You are the concept designer of MOTNGRFX, a pipeline that makes short vertical music videos: generated footage with kinetic typography that lives inside it. From the user's direction below you write the "
        "BRIEF that the piece designer will turn into a piece: the footage idea (a still prompt and shots with one camera move each), the music, and the weave (how the type sits in the footage). Your brief is judged "
        "against the brief the lead designer wrote for the same direction.",
        "=== USER DIRECTION ===\n" + DIRECTIONS[pid],
        "=== SHARED RULES (top of the brief file) ===\n" + shared, "=== EXAMPLE BRIEFS (the FORMAT and the level of detail; make a totally new idea, never reuse their footage or words) ===\n" + ex,
        "=== VISUAL DECISIONS (16_app/PLAN.md) ===\n" + PB._visual_decisions(), "=== MODULE MENU ===\n" + PB._read("16_app/MODULES_MENU.md").strip(),
        "=== COMPOSE OPS available for the weave ===\n" + ", ".join(CB.ALLOWED), "=== MUSIC GRID of the music that will be generated ===\n" + PB.music_text(b),
        "=== OUTPUT ===\n" + FORMAT.replace("<id>", pid)])


WEAVE_TERMS = ["sandwich", "behind", "in front", "knockout", "window", "depth_occlude", "flow_wake", "contour", "scribble", "tint_type", "glow", "matte", "halation", "refract", "blur",
               "crossfade", "ripple", "twist", "braid", "fan", "spiral", "circuit_board", "hold", "freeze", "dropout", "chord", "downbeat", "kick", "beat"]
RULE_TERMS = ["downbeat", "continuous", "behind", "one camera", "hard cut", "cut"]


def parse(text: str, pid: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```\w*\n|\n```$", "", t)
    lines = {k: "" for k in ("header", "Words", "Music", "Footage", "Weave")}
    cur = None
    for ln in t.splitlines():
        if ln.startswith("## "):
            lines["header"], cur = ln, None
            continue
        m = re.match(r"^(Words|Music|Footage|Weave):\s*(.*)$", ln)
        if m:
            cur = m.group(1)
            lines[cur] = m.group(2)
        elif cur and ln.strip():
            lines[cur] += " " + ln.strip()
    return lines


def score(lines: dict, pid: str, claude: dict) -> dict:
    b = PB.BRIEFS[pid]
    want = [w.strip().upper() for w in b["words"]]
    got = [w.strip().upper() for w in re.split(r"\s*/\s*", lines["Words"].rstrip(". ")) if w.strip()]
    fmt = all(lines[k] for k in lines)
    foot, weave = lines["Footage"].lower(), lines["Weave"].lower()
    shots = len(re.findall(r"\bshots?\b", foot))
    n_terms = len({w for w in WEAVE_TERMS if w in weave})
    ct = len({w for w in WEAVE_TERMS if w in claude["Weave"].lower()})
    has_still = '"' in lines["Footage"] or "'" in lines["Footage"]
    camera = len(re.findall(r"\b(camera|dolly|push|pull|orbit|track|sweep|pan|tilt|glide)\w*", foot))
    r = dict(format_ok=fmt, words_exact=got == want, still_prompt=has_still, shot_mentions=shots, camera_terms=camera, cut_on_downbeat=("downbeat" in foot or "downbeat" in weave),
             weave_terms=n_terms, weave_terms_claude=ct, weave_chars=len(lines["Weave"]), weave_chars_claude=len(claude["Weave"]), footage_chars=len(lines["Footage"]), footage_chars_claude=len(claude["Footage"]),
             behind_and_front=("behind" in weave and ("in front" in weave or "front" in weave)), colour_plan=bool(re.search(r"(amber|blue|violet|warm|teal|orange|silver|gold|palette|colou?r)", weave)),
             one_camera_per_shot=("continuous" in foot or "one " in foot))
    r["score"] = (int(fmt) + int(r["words_exact"]) + int(has_still) + int(camera >= 2 and r["cut_on_downbeat"]) + int(n_terms >= max(ct - 3, 5)) + int(r["behind_and_front"]) + int(r["colour_plan"]) +
                  int(r["weave_chars"] >= 0.6 * r["weave_chars_claude"]))
    return r


def run(a, LOCAL, ask, MODELS) -> int:
    import concurrent.futures as cf
    jobs = [(pid, m, s) for pid in a.briefs.split(",") for m in a.models.split(",") for s in range(1, a.seeds + 1)]

    def one(pid, m, s):
        d = LOCAL / a.round / f"{pid}_{'qwen' if 'qwen' in m else 'gemma'}_s{s}"
        d.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(pid)
        (d / "prompt.txt").write_text(prompt, encoding="utf-8")
        if a.reuse and (d / "reply.txt").is_file():
            text, usage = (d / "reply.txt").read_text(encoding="utf-8"), json.loads((d / "usage.json").read_text(encoding="utf-8"))
        else:
            try:
                text, usage = ask(m, prompt, None, s, max_tokens=6000, temperature=0.7)
            except Exception as e:      # noqa: BLE001
                text, usage = "", dict(model=m, error=str(e)[:300])
            (d / "reply.txt").write_text(text, encoding="utf-8")
            (d / "usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
        claude = parse(PB._section((ROOT / PB.BRIEFS[pid]["src"]).read_text(encoding="utf-8"), pid)[1], pid)
        lines = parse(text, pid)
        sc = score(lines, pid, claude)
        res = dict(brief=pid, model=m, seed=s, round=a.round, tokens_in=usage.get("tin"), tokens_out=usage.get("tout"), secs=usage.get("secs"), valid=sc["format_ok"] and sc["words_exact"], rubric=dict(total=sc["score"]),
                   direction=sc, lines=lines)
        (d / "score.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        return res
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in cf.as_completed([ex.submit(one, *j) for j in jobs]):
            r = f.result()
            print(f"{a.round} {r['brief']} {'qwen' if 'qwen' in r['model'] else 'gemma'}: format+words={r['valid']} score={r['rubric']['total']}/8 tin={r['tokens_in']} tout={r['tokens_out']} {r['secs']}s", flush=True)
    return 0
