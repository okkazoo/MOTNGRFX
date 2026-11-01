"""concept: design_llm_8b invents the REQUEST of a piece (pure functions; the nodes are in nodes_fusion/concept.py). NO CLAUDE.

    concept_prompt(seed, length, previous)        -> the prompt for the model
    parse(reply, previous, seed, length)          -> dict(concept, errors, clashes, fixes)        (validate + force uniqueness)
    final(reply2, first, previous, seed, length)  -> concept (the repair reply if it is clean, else the seeded rules variation)

The request fields come from the model (words, scene, music tags, look, shots mode, mood); the LENGTH is set by the caller (the 10-run batch uses one
length so the timings are comparable) and the number of shots stays code-decided (plan_rules.cut_spans / shot_frames from the length and the beat grid).
ONE LOCKUP PER SHOT (2026-10-28, tools/fusion/owner.py): in a `cut` / `chain` piece the model is asked for exactly `lockup_count(length, mode)` lockups of 1-3 words
(the shot count of that length); too many / too few are merged / split deterministically (plan_rules.fit_lockups, logged as a fix, never an error) and the planner
then times lockup k to shot k. A `single` piece keeps 5-7 free lines.
OWNERS (2026-10-28): the model also chooses, per lockup, WHO draws it (`owners`: render flat | ltx | handoff, ltx_at, material; tools/fusion/owner.py). The plan locks
the owners BEFORE the footage exists (the shots are generated from them), so the concept is the only place the model's choice can count. Same enums / clamps / trims
as the design schema (owner.norm_owner); a missing / unreadable entry is left out and the planner uses the rules' owner for that word; at most one LTX word per shot
is enforced in the plan (owner.cap_per_shot).
The still prompt and the shot prompts are written by the existing planner (`MographPiecePlan` builds them from this request), so the model's request
drives every generation prompt downstream.

Uniqueness against `previous` (a list of earlier concepts): rejected when
  - the words are the same (same set of lines, or >= WORDS_OVERLAP of the lines shared),
  - the scene's subject is the same (token overlap >= SUBJECT_OVERLAP) or the scene text overlaps >= SCENE_OVERLAP,
  - the music tags overlap >= TAGS_OVERLAP (share of the smaller tag set),
  - the mood word is the same,
  - the subject CATEGORY (child / young person, adult person, animal, landscape only, vehicle, object, crowd: `subject_category`) or the scene SETTING
    (forest, water, urban, open land, indoor, space: `scene_setting`) was already used by an earlier run: rejected until MIN_CATEGORIES (4) different
    categories have been used, then repeats are fine (the 3060 batch v001 made seven pieces about "a lone child / young person").
The shots mode: the model's own valid answer stands (par round 2026-10-29: the series' spread used to replace it with the least used mode, which made two of four 3060 pieces one long `single` shot);
the series only SUGGESTS the least used multi-shot mode (cut | chain) in the prompt, `single` is never suggested. SHOT PLAN (same round): the reply also carries `idea` + `shot_plan` (a prompt, a camera move and, from shot 2 on,
`continues` per shot: true = the same place carried on from the previous shot's last frame, false = a new place); plan_rules writes those as the shot prompts / `first` refs and uses its templates only for a shot without a usable entry.
A clash goes back to the model once (lazy repair with the clash list); if that fails too the seeded rules variation (BANK) is used, skipping every bank
entry that clashes.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re

from tools.fusion import owner as OW
from tools.fusion import subject_guard as SG
from tools.fusion import user_style as US
from tools.fusion import effects_menu as EM
from tools.fusion import piece_system as PS
from tools.fusion import plate_layers as PLT
from tools.fusion import plan_rules as PR
from tools.fusion import why_fields as WF
from tools.fusion import qwen_piece as QP

MODES = list(PR.SHOT_MODES)                     # cut | chain | single
LOOKS = list(PR.LOOK_NAMES)
WORDS_OVERLAP, SUBJECT_OVERLAP, SCENE_OVERLAP, TAGS_OVERLAP = 0.5, 0.6, 0.5, 0.6
MIN_LINES, MAX_LINES, MAX_LINE_WORDS, MAX_LINE_CHARS = 5, 7, 4, 26     # a single-shot piece: 5-7 free lines
LOCKUP_WORDS = 3                                                          # a multi-shot piece: ONE lockup per shot, 1-3 words each
BPM_RANGE = (60, 160)

# starting cues (seeded): the model invents the concept; the cues keep ten runs from starting at the same place. DIVERSITY ROUND 2026-10-29 (16_app/DIVERSITY_AUDIT.md): the old cue was ONE copied
# example ("a quiet laundromat at 3 am", "a wheat field at dusk") that the models pasted nearly literally, and a third of the examples carried a time of day, so 72 % of the pieces were
# dawn / dusk / night and 68 % misty. Now every ingredient is its own seeded axis (place, light, weather, colour, mood, music, word register), none carries another's content, and an axis
# value an earlier concept of the batch used is skipped.
PLACE_ITEMS = [("harbour", "a working harbour"), ("salt flat", "a desert salt flat"), ("rooftop", "a rooftop garden"), ("flooded", "a flooded nave"), ("mountain pass", "a high mountain pass"),
               ("market", "a street market"), ("funfair", "an abandoned funfair"), ("pine forest", "a pine forest"), ("test clip", "a test clip platform"), ("cliff", "a sea cliff"),
               ("train", "a train carriage"), ("greenhouse", "a greenhouse"), ("frozen lake", "a frozen lake"), ("bus stop", "a bus stop"), ("wheat", "a wheat field"),
               ("skate park", "a concrete skate park"), ("ferry", "a ferry deck"), ("river gorge", "a river gorge"), ("car wash", "a self-service car wash"), ("boxing", "a boxing gym"),
               ("parking", "a multi-storey car park"), ("botanical", "a botanical garden"), ("pier", "a fishing pier"), ("salt marsh", "a salt marsh"), ("stairwell", "a concrete stairwell"),
               ("drive-in", "a drive-in cinema"), ("swimming pool", "an empty swimming pool"), ("market hall", "an indoor market hall"), ("vineyard", "a hillside vineyard"), ("quarry", "a stone quarry"),
               ("lighthouse", "a lighthouse gallery"), ("tennis", "a clay tennis court"), ("bakery", "a bakery kitchen"), ("rice terrace", "a rice terrace"), ("bridge", "under a motorway bridge"),
               ("ice rink", "an ice rink"), ("barber", "a barbershop"), ("bowling", "a bowling alley"), ("cemetery", "a hillside cemetery"), ("ski lift", "a ski lift"), ("airport", "an airport gate"),
               ("dock", "a container dock"), ("orchard", "an apple orchard"), ("sand dune", "a coastal sand dune"), ("tunnel", "a road tunnel"), ("stadium", "an empty stadium"),
               ("diner", "a roadside diner"), ("ballroom", "a faded ballroom"), ("mine", "a disused mine entrance"), ("lagoon", "a tidal lagoon"), ("rooftop pool", "a hotel rooftop pool"),
               ("tram", "a tram stop"), ("allotment", "a village allotment"), ("fishing village", "a fishing village slipway"), ("record shop", "a record shop"), ("steam", "a bathhouse steam room"),
               ("bamboo", "a bamboo grove"), ("elevator", "a glass elevator"), ("playground", "an empty playground"), ("scrapyard", "a scrapyard"), ("temple", "a hillside temple stair"),
               ("laundromat", "a laundromat"), ("canyon", "a slot canyon"), ("library", "a reading room")]
PLACES = [p for _k, p in PLACE_ITEMS]
OVERUSED_PLACES = ("laundromat", "canyon", "library")                     # 3, 2 and 2 of the 25 app concepts: never drawn as a cue, and a clash when the scene names one
LIGHTS = ["hard midday sun with short shadows", "flat overcast daylight", "blue hour after sunset", "late golden-hour backlight", "night with practical lights only", "cold fluorescent interior light",
          "storm light under heavy cloud", "low winter sun from the side", "neon and sodium streetlight", "soft window light from one side", "firelight and candles", "moonlight",
          "bright snow light, no shadows", "mixed daylight and tungsten", "headlights in the dark", "sun through trees in patches"]
WEATHERS = ["clear and dry"] * 3 + ["light rain", "strong wind", "falling snow", "heat shimmer", "drizzle", "dust in the air", "humid and heavy", "calm and overcast", "sudden downpour", "light breeze",
                                    "thin haze"]
COLOURS = ["deep red and black", "teal and orange", "pastel pink and mint", "yellow and navy", "monochrome green", "bone white and ink blue", "burnt orange and brown", "violet and acid lime",
           "faded denim blue and cream", "rust and sea green", "black and a single hot red", "peach and powder blue"]
REGISTERS = ["each lockup is one concrete noun", "short commands (imperatives)", "a question and its answer", "fragments of one spoken sentence", "names and numbers (a place, a time, a count)",
             "verbs ending in -ing", "two opposites against each other", "one repeated word with a single change", "plain everyday objects", "a short list of three things"]
GENRES = ["cinematic orchestral", "lo-fi hip hop", "desert rock", "solo piano and strings", "ambient electronic", "breakbeat and funk bass", "dark ambient drone",
          "gypsy jazz", "post-rock", "uplifting house", "boom bap hip hop", "meditative flute and drums", "synthwave", "acoustic folk", "dub with deep bass",
          "minimal techno", "latin percussion groove", "slow blues guitar"]
MOODS = ["tender", "defiant", "eerie", "joyful", "lonely", "hopeful", "tense", "nostalgic", "playful", "solemn", "restless", "dreamy"]
OVERUSED_WORDS = ("AWAKE", "DRIFT", "HOLD", "BREATHE", "SILENT", "GOLD", "SPIN", "HOME")   # the audit: AWAKE in 4 of 7 Gemma concepts, HOLD 3 / BREATHE 2 / DRIFT 2 / SPIN 2 / HOME 2 of 8 Qwen ones
RECENT_N = 8                                                                             # a word or a place used by one of the last RECENT_N concepts of the batch is a clash
_WORD_STOP = {"THE", "AND", "FOR", "WITH", "YOUR", "THAT", "THIS", "FROM", "INTO", "ARE", "WAS", "NOT", "BUT", "ALL", "OUR", "YOU", "ITS", "HIS", "HER", "THEY", "THEM", "WILL", "WHAT",
              "WHEN", "THEN", "THAN", "EVER", "EACH", "ONCE", "SOME", "ANOTHER", "NEVER", "AGAIN"}


# --------------------------------------------------------------------------------------------- helpers
def _tok(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2 and t not in _STOP}


_STOP = {"the", "and", "with", "from", "that", "this", "for", "are", "its", "into", "his", "her", "their", "over", "under", "near", "through"}


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _share(a: set, b: set) -> float:
    """overlap as a share of the SMALLER set (a tag list that is a subset of another is a clash)."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _tags(music: str) -> set[str]:
    out = set()
    for t in re.split(r"[,;/]", music or ""):
        t = re.sub(r"\b\d+\s*bpm\b", "", t.lower()).strip()
        t = re.sub(r"[^a-z0-9 \-]", "", t).strip()
        if t:
            out.add(t)
    return out


def _lines(c: dict) -> set[str]:
    return {re.sub(r"\s+", " ", w.upper()).strip() for w in c.get("words", []) if str(w).strip()}


# --------------------------------------------------------------------------------------------- categories (v001 batch: 7 concepts were all "a lone child")
MIN_CATEGORIES = 4        # once this many different categories have been used, a repeat of a category is fine
_PEOPLE = ("man woman person lady gentleman men women elder fisherman runner dancer drummer cyclist skateboarder astronaut monk cook keeper gardener "
           "farmer sailor pilot singer musician guitarist pianist painter artist soldier chef baker miner hiker climber swimmer surfer rider traveller "
           "traveler stranger figure silhouette wanderer worker driver potter weaver blacksmith grandmother grandfather mother father queen king shepherd")
SUBJECT_WORDS = {         # keyword -> subject category; the FIRST mapped word of the subject wins (then the scene's words)
    **{w: "crowd" for w in "crowd crowds people audience dancers group family team children kids pair couple townspeople protesters swarm flock".split()},
    **{w: "child" for w in "child kid boy girl teen teenager toddler youth baby schoolboy schoolgirl infant adolescent tween".split()},
    **{w: "adult" for w in _PEOPLE.split()},
    **{w: "animal" for w in ("horse dog cat bird fox wolf deer bear fish whale owl crow raven heron stag rabbit hare elephant lion tiger dragon butterfly moth "
                             "jellyfish octopus seal otter goat sheep cow pony foal puppy kitten animal creature beast").split()},
    **{w: "landscape" for w in ("pine sapling tree trees forest mountain mountains ridge valley river lake ocean sea waves wave desert dune dunes canyon cliff "
                                "glacier waterfall storm clouds sky field meadow flower flowers plant plants moss fog mist landscape horizon island").split()},
    **{w: "vehicle" for w in ("boat ship train car truck bus bicycle motorbike motorcycle plane aeroplane helicopter tram van tractor rocket submarine canoe "
                              "sailboat taxi").split()},
    **{w: "object" for w in ("robot lantern lighthouse stall kite balloon clock piano guitar radio boombox umbrella door window mirror candle lamp telephone "
                             "typewriter bottle cup chair bridge tower house building statue machine drum violin bell key book letter doll mask").split()},
}
SUBJECT_CLASS_NAME = {"child": "child / young person", "adult": "adult person", "animal": "animal", "landscape": "landscape only (no figure)",
                      "vehicle": "vehicle", "object": "object", "crowd": "crowd / group", "other": "other"}
_MODIFIER_YOUNG = {"young", "younger", "little"}
SETTING_WORDS = {         # keyword -> scene setting category (the most frequent in the scene text)
    **{w: "forest" for w in "forest woods pine trees jungle grove orchard".split()},
    **{w: "water" for w in "sea ocean harbour harbor lake river beach shore waves pier ferry dock underwater canal flooded pond".split()},
    **{w: "urban" for w in ("city street neon test clip market rooftop alley bridge highway skyline platform downtown traffic laundromat station").split()},
    **{w: "open land" for w in ("desert dune dunes salt canyon mountain cliff field meadow prairie plain valley hill snow snowy glacier ridge pass wheat "
                                "steppe").split()},
    **{w: "indoor" for w in ("room kitchen church cathedral theatre theater library greenhouse house farmhouse stage studio hall attic warehouse tunnel "
                             "corridor bedroom cabin museum cafe").split()},
    **{w: "space" for w in "space moon moons planet orbit stars galaxy nebula mars asteroid cosmic".split()},
}


def subject_category(subject: str, scene: str = "") -> str:
    """the CATEGORY of the footage's main subject: child (child / kid / boy / girl / teen / young person) | adult | animal | landscape (no figure) |
    vehicle | object | crowd | other. The first mapped word of the subject wins; `young` takes the category of the noun after it unless that is a person
    (a young man is a young person); the scene's words are read only when the subject has none."""
    for text in (subject, scene):
        toks = re.findall(r"[a-z]+", (text or "").lower())
        for i, t in enumerate(toks):
            if t in _MODIFIER_YOUNG:
                nxt = next((SUBJECT_WORDS[u] for u in toks[i + 1:i + 4] if u in SUBJECT_WORDS), None)
                if nxt in ("animal", "landscape", "vehicle", "object", "crowd"):
                    return nxt
                if t != "little" or nxt in ("child", "adult"):
                    return "child"
                continue
            if t in SUBJECT_WORDS:
                return SUBJECT_WORDS[t]
    return "other"


def scene_setting(scene: str, subject: str = "") -> str:
    """the SETTING category of the scene: forest | water | urban | open land | indoor | space | other (the most frequent keyword; ties: the first)."""
    toks = re.findall(r"[a-z]+", ((scene or "") + " " + (subject or "")).lower())
    n: dict = {}
    first: dict = {}
    for i, t in enumerate(toks):
        c = SETTING_WORDS.get(t)
        if c:
            n[c] = n.get(c, 0) + 1
            first.setdefault(c, i)
    return max(n, key=lambda c: (n[c], -first[c])) if n else "other"


def used_categories(previous: list[dict]) -> dict:
    """{"subject": {category: [run numbers]}, "setting": {...}} of the earlier concepts."""
    out: dict = {"subject": {}, "setting": {}}
    for i, p in enumerate(previous, 1):
        out["subject"].setdefault(subject_category(p.get("subject", ""), p.get("scene", "")), []).append(i)
        st = scene_setting(p.get("scene", ""), p.get("subject", ""))
        if st != "other":
            out["setting"].setdefault(st, []).append(i)
    return out


def categories_text(previous: list[dict]) -> str:
    """for the concept prompt: which subject / setting categories are used up (a repeat is rejected until MIN_CATEGORIES have been used)."""
    if not previous:
        return ""
    u = used_categories(previous)
    subj = ", ".join(f"{SUBJECT_CLASS_NAME.get(c, c)} (runs {', '.join(map(str, r))})" for c, r in u["subject"].items())
    sett = ", ".join(f"{c} (runs {', '.join(map(str, r))})" for c, r in u["setting"].items())
    free_s = [SUBJECT_CLASS_NAME[c] for c in SUBJECT_CLASS_NAME if c not in u["subject"] and c != "other"]
    free_t = [c for c in ("forest", "water", "urban", "open land", "indoor", "space") if c not in u["setting"]]
    need = "" if len(u["subject"]) >= MIN_CATEGORIES else f" Categories not used yet, pick one: subject - {', '.join(free_s)}; setting - {', '.join(free_t)}."
    return (f"Subject categories already used: {subj}. Scene settings already used: {sett}. A subject of a category that was already used is rejected "
            f"(a boy and a girl and a teenager are ONE category: young person).{need}")


def mode_counts(previous: list[dict]) -> dict:
    n = {m: 0 for m in MODES}
    for p in previous:
        if p.get("shots") in n:
            n[p["shots"]] += 1
    return n


MULTI_MODES = ("cut", "chain")           # the modes that make SEVERAL shots; `single` is only ever the model's own answer for a piece that is genuinely one scene


def target_mode(previous: list[dict], seed: int, only: tuple = MULTI_MODES) -> str:
    """the SUGGESTED shots mode: the least used multi-shot mode so far (ties: seeded). `single` is never suggested (the 3060 round made two of four pieces one long shot; the lead designer's
    pieces are multi-shot unless the piece is genuinely one scene with one camera move, BRIEFS.md)."""
    n = mode_counts(previous)
    low = min(n[m] for m in only)
    cands = sorted(m for m in only if n[m] == low)
    return random.Random(f"mode:{seed}").choice(cands)


def _draw(seed, axis: str, items, previous: list[dict]):
    """one cue axis: its own seeded draw, skipping the values the earlier concepts of the batch were given (their `cues`)."""
    used = {(p.get("cues") or {}).get(axis) for p in previous if isinstance(p, dict)}
    free = [x for x in items if x not in used] or list(items)
    return random.Random(f"cues:{seed}:{axis}").choice(free)


def cues(seed: int, previous: list[dict]) -> dict:
    """the starting cues of one run: place / light / weather / colour / mood / music / word register, each its OWN seeded axis (never one copied example), plus the piece system's look,
    footage camera moves and photographic style (piece_system.choose with the cue mood; the final system is drawn again from the model's mood and look)."""
    used_m = {p.get("mood") for p in previous}
    mood = random.Random(f"cues:{seed}:mood").choice([m for m in MOODS if m not in used_m] or MOODS)
    sysm = PS.choose(seed, mood, None, earlier=earlier_systems(previous))      # the piece's one palette is drawn here: the scene states it, the system keeps it
    return dict(place=_draw(seed, "place", [p for k, p in PLACE_ITEMS if k not in OVERUSED_PLACES], previous), light=_draw(seed, "light", LIGHTS, previous), weather=_draw(seed, "weather", WEATHERS, previous),
                colour=sysm["palette"]["footage"], genre=_draw(seed, "genre", GENRES, previous), mood=mood, register=_draw(seed, "register", REGISTERS, previous),
                look=sysm["look"], moves=list(sysm["shot_moves"]), photo=sysm["photo"], type_matte=type_matte_drawn(seed))


TYPE_MATTE_P = 0.6            # chance that a piece is asked for ONE type-matte / window shot (2026-10-30: it was "most pieces use none", so it was never chosen)


def type_matte_drawn(seed) -> bool:
    return random.Random(f"cues:{seed}:type_matte").random() < TYPE_MATTE_P


def type_matte_text(drawn: bool) -> str:
    """the prompt sentence about `type_matte` for this piece: drawn (60 %) = ask for ONE on a short word in a fat face; otherwise it stays optional."""
    if drawn:
        return ("THIS PIECE uses ONE `type_matte` (reveal_subject | reveal_plate | subject_through) on a SHORT word (up to 6 letters) in a fat face in ONE shot: put it where the picture can carry it, "
                "in the shot with a clear subject")
    return "`type_matte` is optional this time; use it only if the idea asks for it"


def earlier_systems(previous: list[dict]) -> list[dict]:
    """the piece systems of the earlier concepts of the batch, in order: a concept's `system`, else (an older concept) a stub that only carries its `cut_kinds`, so the next pieces do not
    draw the cut kinds of the last two again (random_system.draw_cuts)."""
    out = []
    for p in previous:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("system"), dict):
            out.append(p["system"])
        elif p.get("cut_kinds"):
            out.append({"cuts": {"kinds": list(p["cut_kinds"])}})
    return out


def place_keys(text: str) -> set:
    t = (text or "").lower()
    return {k for k, _p in PLACE_ITEMS if k in t}


def sig_words(c: dict) -> set:
    """the significant words of a concept's lockups (UPPER CASE, 4+ letters, no function words)."""
    return {t for line in c.get("words", []) for t in re.findall(r"[A-Z']+", str(line).upper()) if len(t) >= 4 and t not in _WORD_STOP}


def recent_words(previous: list[dict]) -> list[str]:
    """the words the last RECENT_N concepts of the batch used, for the prompt's ban list."""
    out: list = []
    for p in previous[-RECENT_N:]:
        for w in sorted(sig_words(p)):
            if w not in out:
                out.append(w)
    return out


def clashes(c: dict, previous: list[dict]) -> list[str]:
    """why `c` is not unique against the earlier concepts (empty = unique)."""
    out = []
    for i, p in enumerate(previous, 1):
        tag = f"run {i}" if len(previous) > 1 else "an earlier run"
        a, b = _lines(c), _lines(p)
        if a and b and (a == b or len(a & b) / min(len(a), len(b)) >= WORDS_OVERLAP):
            out.append(f"{tag}: the words are the same ({', '.join(sorted(a & b))})")
        sa, sb = _tok(c.get("subject", "")), _tok(p.get("subject", ""))
        if sa and sb and (sa == sb or _jacc(sa, sb) >= SUBJECT_OVERLAP):
            out.append(f"{tag}: the same scene subject ({p.get('subject')!r})")
        elif _jacc(_tok(c.get("scene", "")), _tok(p.get("scene", ""))) >= SCENE_OVERLAP:
            out.append(f"{tag}: the scene reads almost the same")
        ta, tb = _tags(c.get("music", "")), _tags(p.get("music", ""))
        if ta and tb and _share(ta, tb) >= TAGS_OVERLAP:
            out.append(f"{tag}: the music tags overlap ({', '.join(sorted(ta & tb))})")
        if c.get("mood") and c.get("mood") == p.get("mood"):
            out.append(f"{tag}: the same mood ({p.get('mood')!r})")
    sw = sig_words(c)
    for i, p in enumerate(previous, 1):
        if i <= len(previous) - RECENT_N:
            continue
        shared = sorted(sw & sig_words(p))
        if shared:
            out.append(f"run {i}: the word{'s' if len(shared) > 1 else ''} {', '.join(shared)} {'were' if len(shared) > 1 else 'was'} already used in this batch; use other words")
        kp = place_keys(c.get("scene", "")) & (place_keys(p.get("scene", "")) | place_keys((p.get("cues") or {}).get("place", "")))
        if kp:
            out.append(f"run {i}: the same place ({', '.join(sorted(kp))}); choose another place")
    over = sorted(sw & set(OVERUSED_WORDS))
    if over:
        out.append(f"the word{'s' if len(over) > 1 else ''} {', '.join(over)} {'are' if len(over) > 1 else 'is'} overused by these models (AWAKE, DRIFT, HOLD, BREATHE ...); write other words")
    bad = sorted(place_keys(c.get("scene", "")) & set(OVERUSED_PLACES))
    if bad:
        out.append(f"the place ({', '.join(bad)}) is overused by these models; choose another place")
    out += category_clashes(c, previous)
    return out


def category_clashes(c: dict, previous: list[dict]) -> list[str]:
    """a subject category / scene setting that an earlier run already used is rejected, until MIN_CATEGORIES different ones have been used."""
    if not previous:
        return []
    u = used_categories(previous)
    out = []
    cs = subject_category(c.get("subject", ""), c.get("scene", ""))
    if cs in u["subject"] and len(u["subject"]) < MIN_CATEGORIES:
        r0 = u["subject"][cs][0]
        out.append(f"run {r0}: the same kind of subject ({SUBJECT_CLASS_NAME.get(cs, cs)}: {c.get('subject')!r} vs {previous[r0 - 1].get('subject')!r}); "
                   f"choose another category (animal, landscape without a figure, object, vehicle, adult person, crowd, child / young person)")
    ct = scene_setting(c.get("scene", ""), c.get("subject", ""))
    if ct != "other" and ct in u["setting"] and len(u["setting"]) < MIN_CATEGORIES:
        out.append(f"run {u['setting'][ct][0]}: the same scene setting ({ct}); choose another (forest, water, urban, open land, indoor, space)")
    return out


# --------------------------------------------------------------------------------------------- prompt
def previous_text(previous: list[dict]) -> str:
    if not previous:
        return "(nothing yet: this is the first piece)"
    return "\n".join(f"{i}. words: {' / '.join(p.get('words', []))} | subject: {p.get('subject')} | scene: {str(p.get('scene', ''))[:110]} | mood: {p.get('mood')} | "
                     f"music: {p.get('music')} | shots: {p.get('shots')}" for i, p in enumerate(previous, 1))


def story_n(b: dict, length: int) -> int | None:
    """the storyboard's shot count (brief.story.shots, STORYBOARD v12: never more shots than pictures) when the planner can honour it at this length (plan_rules.story_shot_count), else None = today's rule."""
    shots = ((b or {}).get("story") or {}).get("shots") or []
    if not shots:
        return None
    return PR.story_shot_count(len(shots), int(length) * PR.FPS)[0]


def lockup_count(length: int, mode: str, story_shots: int | None = None) -> int | None:
    """how many lockups (text blocks) the piece shows: ONE per shot, timed to the cuts (tools/fusion/owner.py, plan_rules.fit_lockups), so the number of shots
    of a `cut` / `chain` piece of this length (plan_rules.cut_spans starts from mograph.fusion_brief.shot_count). None for `single` (one continuous shot: free lines)."""
    if mode == "single":
        return None
    if story_shots:
        return int(story_shots)                         # the storyboard's count (story_n: validated), the planner cuts exactly that many shots (plan_rules._build)
    return max(1, PR.fb.shot_count(int(math.ceil(int(length) * PR.FPS - 1e-6))))


def avoid_text(previous: list[dict]) -> str:
    """the prompt's ban lists: words the models overuse, words and places the earlier concepts of this batch already used."""
    rw = recent_words(previous)
    rp = []
    for p in previous[-RECENT_N:]:
        for k in sorted(place_keys(p.get("scene", "")) | place_keys((p.get("cues") or {}).get("place", ""))):
            if k not in rp:
                rp.append(k)
    return (f"Do NOT use these words (overused by earlier runs): {', '.join(OVERUSED_WORDS)}" + (f"; nor any of these, already used in this batch: {', '.join(rw)}" if rw else "") + ". "
            f"Do NOT use these places: {', '.join(OVERUSED_PLACES)}" + (f"; nor these, already used in this batch: {', '.join(rp)}" if rp else "") + ".")


def effects_text(seed: int, previous: list[dict], effects=()) -> str:
    """the concept prompt's sentence about the named effects (effects_menu): the ones the user REQUIRES and the contrast looks drawn for this piece. '' without either."""
    d = EM.draw_effects(seed, earlier_systems(previous), effects)
    active = list(dict.fromkeys(d["contrast"] + d["requested"]))
    return EM.prompt_text({"effects": dict(d, active=active)})


# --------------------------------------------------------------------------------------------- the USER BRIEF (READ JSON v1, 16_app/MEDIA_BOX.md)
BRIEF_MAX_LINES = 12                  # on-screen lines kept from the brief
BRIEF_MAX_IMAGES = 5


def _ws(v, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


STORY_MAX_SHOTS = 6                   # = mograph.fusion_brief.MAX_SHOTS


def liked_font_file(v) -> str | None:
    """a font choice of the page (liked file / key / family) -> its liked.json file, else None (a font that is not liked is never a user choice)."""
    if not isinstance(v, str) or not v.strip():
        return None
    from mograph import fonts_liked as fl
    e = fl.lookup(v.strip())
    return e["file"] if e else None


def story_camera(v) -> str | None:
    """a camera move of the page / story (the cut into the next shot) -> 'hold' or a canonical cut_kinds kind (depth_pivot, orbit:push, pan:left, ...), else None."""
    if not isinstance(v, str) or not v.strip():
        return None
    if v.strip().lower() in ("hold", "none", "cut", "hard"):
        return "hold"
    from tools.fusion import cut_kinds as CK
    k = CK.normalize(v)
    return k if k and k not in CK._BARE else None


def story_cuts(c: dict) -> list:
    """the user's per-shot camera moves (brief.story.shots[].camera = the cut after shot k) as the cut kind the planner may pin: [kind | None, ..] one per shot, [] when the story chose none.
    Only the 3D kinds (depth_pivot / orbit:* / depth_dolly:*) and `hold` (= an explicit hard cut) are pinned; a pan is a type-camera travel the planner lays out itself (kept in the brief only) and
    window / explore kinds need a word, so they are never forced. The last shot has no cut."""
    shots = ((c.get("brief") or {}).get("story") or {}).get("shots") or []
    from tools.fusion import cut_kinds as CK
    out = []
    for k, x in enumerate(shots):
        cam = x.get("camera")
        out.append(("hard" if cam == "hold" else cam) if cam and k < len(shots) - 1 and (cam == "hold" or cam in CK.SHOT_3D) else None)
    return out if any(out) else []


def story_fonts(c: dict) -> list:
    """the user's per-line fonts (brief.story.shots[].font) aligned with the concept's `words`: [file | None, ..], [] when the user chose none (tools/fusion/font_pins.py matches by line text,
    then by position)."""
    shots = ((c.get("brief") or {}).get("story") or {}).get("shots") or []
    if not any(x.get("font") and x.get("font_user") is True for x in shots):          # R17b: a font without `font_user` is the page's random default, never a pin
        return []
    from tools.fusion import font_pins as FP
    pins = [{"text": x.get("line") or "", "font": x.get("font") if x.get("font_user") is True else None} for x in shots]          # ALL shots (a line without a choice stays None; the position fallback needs the full list)
    words = list(c.get("words") or [])
    out = [FP.pin_for(w, i, len(words), pins) for i, w in enumerate(words)]
    return out if any(out) else []


def _clean_story(st) -> dict:
    """the STORY JSON v1 the page carries in the brief ({title, logline, look, music, shots[{n, image, start, end, beat, line}], notes}; tools/fusion/story.py) -> a cleaned copy, or {} when it is
    absent / carries nothing. Kept so the planner and the archive see the approved story; absent = the brief is what it was before the story step."""
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except ValueError:
            return {}
    if not isinstance(st, dict):
        return {}
    shots = []
    for x in st.get("shots") if isinstance(st.get("shots"), list) else []:
        if not isinstance(x, dict):
            continue
        t = {}
        for k in ("start", "end"):
            try:
                t[k] = round(float(x.get(k)), 3)
            except (TypeError, ValueError):
                t[k] = None
        shot = {"n": len(shots) + 1, "image": _ws(x.get("image"), 8) or None, "start": t["start"], "end": t["end"], "beat": _ws(x.get("beat"), 320), "line": _ws(x.get("line"), 160)}
        ff = liked_font_file(x.get("font"))
        if ff:
            shot["font"] = ff                                  # the font shown for this line (storyboard STORY v2): a liked.json file, or dropped
            if x.get("font_user") is True:
                shot["font_user"] = True                       # the USER changed it in the page (R17b): only this makes it a pin; the page's random default font is not one
        cam = story_camera(x.get("camera"))
        if cam:
            shot["camera"] = cam                               # the cut INTO the next shot (storyboard STORY v2): hold | a cut_kinds kind, or dropped
        shots.append(shot)
    out = {"title": _ws(st.get("title"), 60), "logline": _ws(st.get("logline"), 300), "look": _ws(st.get("look"), 300), "music": _ws(st.get("music"), 200),
           "shots": shots[:STORY_MAX_SHOTS], "notes": _ws(st.get("notes"), 400)}
    return out if (out["title"] or out["logline"] or any(x["beat"] or x["line"] for x in out["shots"])) else {}


def story_text(st: dict, beat_chars: int = 140) -> str:
    """the story as one paragraph (title, logline, beat k per shot) for `idea` and the concept prompt; '' without a story."""
    if not st:
        return ""
    bits = [f"{st['title']}." if st.get("title") else "", st.get("logline") or ""] + [f"Shot {x['n']}: {x['beat'][:beat_chars]}" for x in st.get("shots") or [] if x.get("beat")]
    return " ".join(b.rstrip() if b.endswith((".", "!", "?")) else b + "." for b in bits if b)


def story_in_concept(b: dict) -> bool:
    """did the page already put the story's logline into the brief's concept (MEDIA_BOX.md: concept = title + logline + beats)?"""
    st = (b or {}).get("story") or {}
    return bool(st.get("logline") and st["logline"] in (b.get("concept") or "")) or bool(b.get("concept") and not st.get("logline") and st.get("title") and st["title"] in b["concept"])


def parse_brief(text) -> dict:
    """the `brief` input (READ JSON v1: on_screen, concept, look, media, music, notes; a dict, its JSON text, or plain text = a concept) -> the cleaned brief, or {} when it carries nothing
    the concept can use (empty / blank / only `notes`): {} means today's fully automatic behaviour, byte for byte."""
    if isinstance(text, dict):
        v = text
    else:
        t = str(text or "").strip()
        if not t:
            return {}
        try:
            v = json.loads(t)
        except ValueError:
            v = None
        if not isinstance(v, dict):
            v = {"concept": t}                                    # not READ JSON: free text is a direction
    on = v.get("on_screen")
    if isinstance(on, str):
        on = re.split(r"[\n|]", on)
    lines: list[str] = []
    for x in on if isinstance(on, list) else []:
        t = _ws(US.words_of(x) if isinstance(x, str) else x, 160)          # a direction clause ("palette: ...", "more legible text") is never an on-screen line (user_style.split_typed)
        if t and (not lines or lines[-1] != t):
            lines.append(t)
    media = []
    for m in v.get("media") if isinstance(v.get("media"), list) else []:
        if not isinstance(m, dict) or str(m.get("kind") or "").lower() not in ("video", "image", "audio"):
            continue
        w = m.get("window")
        try:
            w = [round(float(w[0]), 2), round(float(w[1]), 2)] if isinstance(w, (list, tuple)) and len(w) == 2 else None
        except (TypeError, ValueError):
            w = None
        pal = [str(c)[:9] for c in (m.get("palette") or []) if isinstance(c, str)][:6]
        media.append({"id": _ws(m.get("id"), 8), "kind": str(m["kind"]).lower(), "shows": _ws(m.get("shows"), 240), "mood": _ws(m.get("mood"), 60), "window": w, "palette": pal})
    mu = v.get("music")
    music = None
    if isinstance(mu, dict):
        try:
            bpm = float(mu.get("bpm"))
            bpm = round(bpm, 1) if 20 <= bpm <= 300 else None
        except (TypeError, ValueError):
            bpm = None
        key = _ws(mu.get("key"), 24)
        music = {"bpm": bpm, "key": key} if bpm or key else None
    story = _clean_story(v.get("story"))
    if story and not lines:                                   # a story without on_screen: its lines (the user saw them in the storyboard) are the words
        lines = [x["line"] for x in story["shots"] if x["line"] and (not lines or lines[-1] != x["line"])]
        lines = [t for t in (_ws(US.words_of(x), 160) for x in lines) if t]
    if isinstance(v.get("direction"), str) and v["direction"].strip():
        lines = US.drop_direction_lines(lines, v["direction"])      # a line the READ copied out of the typed direction ("CRIMSON, GOLD" from "palette: crimson, gold") is not a word
    b = {"on_screen": lines[:BRIEF_MAX_LINES], "concept": _ws(v.get("concept"), 600), "look": _ws(v.get("look"), 300), "media": media[:7], "music": music, "notes": _ws(v.get("notes"), 300)}
    if story:
        b["story"] = story                                    # only when there is one: a brief without a story is byte-identical to before
    usable = bool(b["on_screen"] or b["concept"] or b["look"] or b["music"] or story or any(m["shows"] for m in b["media"]))
    dr = v.get("direction")                                   # the text the USER typed in the page's "Words or direction" box (app.js withDirection), never the model's; rides to user_style.stamp
    if usable and isinstance(dr, str) and dr.strip():
        b["direction"] = dr.replace("\r\n", "\n").replace("\r", "\n").strip()[:2000]
    return b if usable else {}


def brief_lines(b: dict) -> list[str]:
    """the user's on-screen lines as the concept's `words` (UPPER CASE, the same character clean-up as the model's lines; never merged, split or rewritten)."""
    out: list[str] = []
    for x in (b or {}).get("on_screen") or []:
        t = re.sub(r"\s+", " ", _CLEAN.sub("", str(x).upper())).strip()
        if t and (not out or out[-1] != t):
            out.append(t)
    return out


def brief_extra_images(b: dict) -> int:
    """how many images the user gave BEYOND the first (image 1 = the first frame of shot 1, `IN image`; image k = the still of shot k, `IN images`)."""
    return max(0, min(sum(1 for m in (b or {}).get("media") or [] if m["kind"] == "image"), BRIEF_MAX_IMAGES) - 1)


def brief_bpm(b: dict) -> int | None:
    bpm = ((b or {}).get("music") or {}).get("bpm")
    return int(min(max(round(bpm), BPM_RANGE[0]), BPM_RANGE[1])) if bpm else None


def brief_block(b: dict, length: int, n_shots: int) -> str:
    """the USER BRIEF section of the concept prompt ('' without a brief)."""
    if not b:
        return ""
    lines = brief_lines(b)
    out = ["=== USER BRIEF (written by the person making this piece: it OVERRIDES any cue below that contradicts it; the ALREADY MADE list and the ban lists do not apply to it) ==="]
    if lines:
        out.append(f"- ON-SCREEN LINES, used VERBATIM and in this order: {json.dumps(lines, ensure_ascii=False)}. `words` is exactly these {len(lines)} line{'s' if len(lines) != 1 else ''}, copied letter for letter "
                   f"(the code restores them anyway): do not translate, reword, add, merge or drop one. The words register cue is ignored for them; invent the scene, subject, shots and music so that these words belong. "
                   f"The piece has {n_shots} shot{'s' if n_shots != 1 else ''}: line k appears in shot k; when there are fewer lines than shots the last line stays.")
    if b.get("concept"):
        out.append(f"- CONCEPT (the direction to follow; `idea`, `scene` and `shot_plan` must realise it, do not replace it with your own): {b['concept']}")
    if b.get("look"):
        out.append(f"- LOOK / FEEL (pace, palette, mood, light: it steers the TYPE colours, the grade and the lighting mood, in `mood`, `music` and the camera moves, in place of the colour / light / mood / photographic cues. "
                   f"It NEVER changes the subject, what the subject wears or holds, or the colour of any object in a user image, in `subject_desc` or in `shot_plan`): {b['look']}")
    if b.get("story") and not story_in_concept(b):
        out.append(f"- STORY (the storyboard the user approved; `idea` tells it and `shot_plan` entry k realises beat k, in this order): {story_text(b['story'], 200)}")
    media = b.get("media") or []
    imgs = [m for m in media if m["kind"] == "image"][:BRIEF_MAX_IMAGES]
    desc = []
    for k, m in enumerate(imgs, 1):
        bits = [m["shows"] or "(not described)"] + ([f"mood {m['mood']}"] if m["mood"] else []) + ([f"palette {', '.join(m['palette'])}"] if m["palette"] else [])
        role = ("the FIRST FRAME of shot 1: `scene` and shot 1 describe exactly this picture" if k == 1 else
                f"the STILL of shot {k}: the first frame of shot {k}; the `shot_plan` entry {k} continues what this picture shows (same subject and place, nothing contradicting it) with `same_place` false")
        desc.append(f"  * user image {k} ({m['id'] or 'I' + str(k)}) is {role}. It shows: {'; '.join(bits)}")
    if imgs:
        desc.append("  * RULE for every shot that starts from a user image: its `shot_plan` prompt restates ONLY what that image shows (the 'It shows' text above) and then says what MOVES; "
                    "it invents no clothing, colour, jewellery, object, animal or other person, and names no framing (the camera field carries it). The video model sees only that first frame, so a changed subject turns into a different clip.")
    for m in media:
        if m["kind"] == "video":
            desc.append(f"  * user video ({m['id'] or 'V1'}): {m['shows'] or '(not described)'}" + (f"; window {m['window'][0]}-{m['window'][1]} s" if m["window"] else "") + (f"; mood {m['mood']}" if m["mood"] else ""))
        elif m["kind"] == "audio":
            desc.append(f"  * user audio ({m['id'] or 'A1'}): {m['shows'] or 'the soundtrack'}" + (f"; mood {m['mood']}" if m["mood"] else ""))
    if desc:
        out.append("- MEDIA the user supplied (the footage and the stills must match what they show; treat them as fixed facts):\n" + "\n".join(desc))
    mu = b.get("music")
    if mu:
        out.append("- MUSIC: the user's own song is the soundtrack and the cut follows it" + (f"; tempo {mu['bpm']:g} bpm (`bpm` = {brief_bpm(b)})" if mu.get("bpm") else "") + (f"; key {mu['key']}" if mu.get("key") else "")
                   + ". `music` still lists style tags that describe that song's feel.")
    if b.get("notes"):
        out.append(f"- NOTES: {b['notes']}")
    return "\n".join(out) + "\n\n"


def brief_images(b: dict) -> list[dict]:
    """the user's image media of a brief (image k = the still of shot k), at most BRIEF_MAX_IMAGES."""
    return [m for m in (b or {}).get("media") or [] if m["kind"] == "image"][:BRIEF_MAX_IMAGES]


def user_shows(b: dict) -> list[str]:
    """the media-read `shows` of each user image, in order (shot k starts from image k); '' for an undescribed image; [] without images or without any text."""
    out = [str(m.get("shows") or "").strip() for m in brief_images(b)]
    return out if any(out) else []


def guard_user_shots(sp: list[dict], shows: list[str], allowed: str = "", framing: bool = False) -> tuple[list[dict], int]:
    """SUBJECT FROM THE USER'S PICTURE (tools/fusion/subject_guard.py): in a shot that starts from a user image the subject text is that image's `shows`, not the model's `subject_desc`, and the
    clauses that introduce clothing / colours / people / animals the image does not show are removed. framing=True also strips the framing phrases of the model text where the shot's camera
    lead will name the framing (shot_camera.lead): 'Wide establishing shot, ..., A close-up of X' contradicted itself. Shots without a user image are returned untouched. -> (entries, number changed)."""
    from tools.fusion import shot_camera as SC
    out, n = [], 0
    for k, e in enumerate(sp):
        s = shows[k] if k < len(shows) else ""
        if e and s and str(e.get("prompt") or ""):
            p0 = str(e["prompt"])
            p = SG.guard_prompt(p0, s, allowed)
            if framing and SC.lead(e):
                p = SG.strip_framing(p)
            if p != p0:
                e = dict(e, prompt=p)
                n += 1
        out.append(e)
    return out, n


def apply_brief(c: dict, b: dict) -> dict:
    """the code side of the brief on a finished concept (the model's, or the rules variation's): the on-screen lines verbatim, the user's concept / look kept in `idea`, the song's bpm,
    the shots that start from a user image are new places. Idempotent. Returns c."""
    if not b:
        return c
    lines = brief_lines(b)
    if lines and list(c.get("words") or []) != lines:
        if c.get("owners") and len(c["owners"]) != len(lines):
            c.pop("owners", None)                                   # an owner per line of the OLD words: the rules' owners (all flat) are used
        c["words"] = lines
    bpm = brief_bpm(b)
    if bpm and c.get("bpm") != bpm:
        c["bpm"] = bpm
        tags = list(c.get("tags") or [])
        mood = c.get("mood") or ""
        c["music"] = ", ".join(tags + ([mood] if mood and mood not in [t.lower() for t in tags] else [])) + f", {bpm} bpm"
    pre = [f"USER CONCEPT: {b['concept']}"] if b.get("concept") else []
    pre += [f"USER LOOK: {b['look']}"] if b.get("look") else []
    st = b.get("story") or {}
    if st and not story_in_concept(b):
        pre += [f"USER STORY: {story_text(st)}"]                 # the page normally put title + logline + beats into the concept; if not, the story goes into the idea here
    if pre and not str(c.get("idea") or "").startswith(pre[0]):
        c["idea"] = (" ".join(pre) + (" | " + c["idea"] if c.get("idea") else ""))[:1800 if st else 900]
    sp = c.get("shot_plan")
    if isinstance(sp, list):
        for k in range(1, min(brief_extra_images(b) + 1, len(sp))):
            sp[k] = dict(sp[k] or {}, continues=False)              # the shot starts from the user's image: a place of its own, never an angle edit of another place
    shows = user_shows(b)
    if shows:
        c["shows"] = shows                                          # -> request_extras -> MographRequest -> piece.request.shows (the layered_shots scrub reads it)
        if isinstance(c.get("shot_plan"), list):
            allowed = " ".join(brief_lines(b))
            sp2, _ = guard_user_shots(c["shot_plan"], shows, allowed, framing=False)
            c["shot_plan"], _ = guard_user_shots(sp2, shows, allowed, framing=True)       # idempotent; the framing strip only touches entries that have a camera lead
    c["brief"] = {k: b[k] for k in ("on_screen", "concept", "look", "music", "story") if b.get(k)}
    c["brief"]["images"] = brief_extra_images(b) + (1 if any(m["kind"] == "image" for m in b.get("media") or []) else 0)
    if b.get("direction"):
        c["direction"] = b["direction"]                           # -> request_extras -> MographRequest -> MographPiecePlan (user_style.stamp)
    return c


def concept_prompt(seed: int, length: int, previous: list[dict], mode: str | None = None, effects=(), brief="") -> str:
    cu = cues(seed, previous)
    cu["effects_text"] = effects_text(seed, previous, effects)
    cu["type_matte_text"] = type_matte_text(cu["type_matte"])
    want = mode if mode in MODES else target_mode(previous, seed)
    b = parse_brief(brief)                                  # {} = no brief: the prompt below is then byte-identical to the one without this argument
    n = lockup_count(length, want, story_n(b, length))
    ulines = brief_lines(b)
    cap = OW.shot_cap(length) if n is None else OW.MAX_LTX_PER_SHOT
    if n is None:
        words_spec = f'["<{MIN_LINES} to {MAX_LINES} lines of text shown on screen, UPPER CASE, 1-{MAX_LINE_WORDS} words each, together one small story or feeling>", "..."]'
        words_rule = ""
    else:
        words_spec = "[" + ", ".join(f'"<lockup {k}: ONE short word, UPPER CASE (two or three words only when they are one fixed phrase)>"' for k in range(1, n + 1)) + "]"
        words_rule = (f"`words` has EXACTLY {n} entries: the piece is cut into {n} shots and each shot shows ONE lockup of text, in story order, "
                      f"1-{LOCKUP_WORDS} words each (short words read best); ")
    if ulines:                                             # the user's lines ARE the words: the model copies them, the code restores them (apply_brief)
        words_spec = json.dumps(ulines, ensure_ascii=False)
        words_rule = f"`words` is EXACTLY the USER BRIEF's {len(ulines)} on-screen line{'s' if len(ulines) != 1 else ''} below, copied letter for letter, in order; "
    owners_spec = ('[' + ", ".join('{"render": "flat | ltx | handoff", "ltx_at": <0.2..0.8, handoff only>, "material": "<at most %d words>", "why": "<why this owner and material for this line>"}' % OW.MATERIAL_MAX_WORDS
                                   for _ in range(2)) + ", ... one per entry of words]")
    owners_rule = (f"the DEFAULT is flat; give LTX AT MOST {cap} key line{'s' if cap > 1 else ''} of the whole piece (handoff or ltx) and only when material letters are the idea of the piece" if n is None else
                   "the DEFAULT is flat: every lockup is flat unless material letters are the idea of THAT shot; at most one handoff or ltx lockup per shot and usually none or one in the whole piece (never all LTX); small or long lockups are always flat")
    ns = 1 if n is None else n
    sdesc_rule = ("<the subject exactly as it looks, 8-20 words: kind, age or surface, clothes and colours, one distinguishing object; it is repeated word for word in every shot and still>" if not brief_images(b) else
                  "<the subject of the USER IMAGES, 8-20 words, using ONLY what their 'It shows' text says: no invented clothing, colours, jewellery, objects or animals; a shot that starts from a user image uses that image's own text instead>")
    shot_spec = "[" + ", ".join('{"prompt": "<what happens in shot %d: what the subject does, 12-35 words, concrete and photographic; the subject is named with its subject_desc>", "camera": "<ONE camera move, 2-6 words>"%s, "why": "<why this place / camera move / continuation>"}' % (k, ', "same_place": <true = the SAME place and subject from another camera (it gets its own angle still, never the last frame of the previous shot) | false = a new place with a fresh still>' if k > 1 else "") for k in range(1, ns + 1)) + "]"
    return f"""You invent the REQUEST for one short music-video piece: kinetic typography over generated footage, with a generated song. Everything after this is
made from your answer (the still image, the video shots, the music, the type), so be concrete, visual and original.
The piece lasts {int(length)} seconds.

Reply with ONE JSON object and nothing else (no markdown fence, no commentary):
{{
 "title": "<2-4 words>",
 "words": {words_spec},
 "subject": "<the ONE main subject of the footage, 2-4 words, e.g. 'old fisherman' or 'neon noodle stall'>",
 "subject_desc": "{sdesc_rule}",
 "scene": "<one sentence of 25-50 words: what the footage shows: the subject, the place, the light, the weather, how far the camera is; photographic; no text, no logos>",
 "music": "<8-14 comma separated style tags for a text-to-music model: genre, groove, real instruments, mood, texture; no artist names, no bpm>",
 "bpm": <{BPM_RANGE[0]}..{BPM_RANGE[1]}>,
 "mood": "<one word>",
 "look": "<one of: {', '.join(LOOKS)}; the cue above unless the idea needs another>",
 "shots": "<cut | chain | single>",
 "idea": "<ONE filmable idea in one or two sentences: what happens on screen, the camera move chosen FOR this kind of subject, and how the words ride it>",
 "idea_why": "<why this idea fits this subject, mood and music, one sentence>",
 "shot_plan": {shot_spec},
 "owners": {owners_spec}
}}
look: the family of 3D type: strand = fine lines, mech = machined 3D, slab = heavy blocks, stack = layered bars, ring = rings and tunnels, voxel = cubes, particles = dust and streaks, auto = decided later.
shots: cut = a different still for every shot with hard cuts on the beat; chain = one place seen from a different camera in every shot (each shot starts from its OWN still of that place, never from the previous shot's last frame); single = one long continuous shot.
owners: who draws each entry of `words` (same order, one object each). render "flat" = clean flat type that compose draws (cheap, always readable); "handoff" = the flat type forms in, then the
footage model's own 3D letters take over at ltx_at and hold; "ltx" = the footage model draws the letters the whole time. LTX letters are expensive and fragile: {owners_rule}. LTX can only spell a SHORT word: an entry you give to handoff or ltx must be ONE short key word of at most {OW.MAX_LTX_LETTERS} letters (e.g. "STORM", "RISE", "FLARE"); a longer entry stays flat.
material = what that word's letters are physically made of, matched to the scene and different from the other words ("thick carved ice, extruded, frosted edges", "wet harbour stone, barnacles", "brushed steel plates, riveted");
leave material out for a flat word; ltx_at only for handoff (0.2 = early, 0.8 = late; vary it between words).
idea / shot_plan: the footage and the words are ONE idea. `shot_plan` has EXACTLY {ns} entr{'y' if ns == 1 else 'ies'}, one per shot, in story order; each shot prompt says what the subject does in THAT shot, progresses the previous shot
(place, light, action) and contains no text, letters or captions; the shots are different from each other. The camera move is chosen FOR the subject: a runner or walker is tracked so that the subject stays at the same place
in the frame (the cut-out and the type stay stable), a person facing the camera is met by a camera that backs away and keeps them in frame, a landscape gets one slow lateral sweep. One continuous move per shot.
why: every `why` is one short sentence specific to THIS piece (its subject, light, music or the line it serves), never a rule restated; it may be omitted but is read.
OPTIONAL footage options per shot (guide section 23; {cu['type_matte_text']}): in a `shot_plan` entry `same_place: true` + `framing` (close-up | low | profile | top-down | wide) + `same_subject_as` (the first shot's id) makes the shot an angle still of the SAME subject; `plate: "clean"` (+ optional `subject_move`, `plate_pan`) puts the type between a clean background and the subject, for ONE whole subject clearly apart from the background (at most two per piece); `type_matte` (reveal_subject | reveal_plate | subject_through) on a SHORT word in a fat face (one or two shots).
EXPLORE LOOKS (guide section 24): a `shot_plan` entry may carry `"fx"` = letters_hold_subject | chronophoto_echoes | echo_trail | outline_to_word | outline_over_word | plate_letters_subject_over, for a shot whose subject moves clearly across the frame (a matte-cut subject on a plain ground with the word); at most one or two per piece, on different shots; leave it out when the idea does not ask for it.
SUBJECT CONTINUITY: one subject, one description. `subject_desc` is copied word for word into every shot prompt and every still (the code adds it where it is missing), so the footage never changes who or what it shows (no second person, no other car); the subject fills at least a quarter of the frame height unless that shot is a wide establishing shot; one palette for the whole piece (the colours cue).
DEFAULT SHAPE (the pieces the user liked best share it; vary the subject, place, light and colours from piece to piece, never inside one piece): ONE subject or one coherent world in every shot (the same subject from another framing, or a few places of one world, never a different person or object per shot); short single words, one per lockup; a designed 3D cut at every shot boundary (the code adds them).
Rules: {words_rule}one clear subject that a person-matte can cut out (a person, an animal, a vehicle, an object); the words must be a story or a feeling, not a description of the footage;
the music must be playable by real instruments (no toy sounds).

{brief_block(b, length, 1 if n is None else n)}Cues for THIS piece (each is one separate ingredient drawn for this run so that no two pieces start from the same place: build the concept FROM them, and do not add a time of day, dawn, dusk, sunrise, mist, fog or haze that a cue does not name):
 place - {cu['place']}; light - {cu['light']}; weather - {cu['weather']}; ONE colour palette for the whole piece (footage and light) - {cu['colour']}; mood - {cu['mood']}; music direction - {cu['genre']};
 the words are written in this register: {cu['register']}; look (3D type family) - {cu['look']}; camera moves for the shots come from: {', '.join(cu['moves'])}; photographic style of the footage - {cu['photo']}.
{cu['effects_text']}
{avoid_text(previous)}
Shot structure (the `shots` answer): `cut` = a different place or angle per shot, hard cuts on downbeats; `chain` = one place and subject from a different camera in every shot (every shot has its own still; no shot starts from the last frame of another: that carries drift and the same framing forward);
`single` = ONE long shot, only when the whole piece is genuinely one scene with one camera move. Several beats mean several shots: a strong default is places of TWO shots each (a new place, then the same subject from another camera: `same_place` false, true, false, true ...),
at most 6 shots of 3-5 s, hard cuts on downbeats. Suggested for variety in this series: shots = "{want}"; with `same_place` flags a `cut` piece can show a place from two cameras; answer `single` only if the idea really is one scene.
Design seed: {int(seed)}.

=== ALREADY MADE (yours must differ from every one: other words, other subject, other mood, other music tags) ===
{previous_text(previous)}
{categories_text(previous)}
"""


def repair_prompt(prompt: str, reply: str, problems: list[str]) -> str:
    return ((prompt or "").rstrip() + "\n\n=== YOUR PREVIOUS REPLY (rejected) ===\n" + (reply or "(empty)")[:3000] + "\n\n=== PROBLEMS ===\n" +
            "\n".join(f"- {p}" for p in problems[:14]) +
            "\nReturn the COMPLETE new JSON object (same schema, nothing else). Change the words, the subject, the mood and the music so that none of the problems remains.")


# --------------------------------------------------------------------------------------------- parse
_CLEAN = re.compile(r"[^A-Z0-9 '!?.,&\-]")


def owners_of(v, src_idx: list[int]) -> tuple[list[dict] | None, list[str]]:
    """the reply's `owners` list -> (one clean entry per kept line of `words` (src_idx = the reply index of each kept line), fixes). None when the reply has none /
    not a list (the rules' owners are used, logged by the caller). Each entry is mended alone (owner.norm_owner)."""
    if v is None:
        return None, ["owners: the reply has none; the rules' owners are used"]
    if isinstance(v, dict):                                 # {"0": {...}, "1": {...}} or {"THE SEA": {...}} is not read; a single object is one entry
        v = [v] if any(k in v for k in ("render", "ltx_at", "material")) else None
    if not isinstance(v, list):
        return None, [f"owners: {type(v).__name__} is not a list; the rules' owners are used"]
    out, fixes = [], []
    if len(v) != len(src_idx):
        fixes.append(f"owners: {len(v)} entries for {len(src_idx)} lines of words; matched in order, a missing one is the rules' owner")
    for k, j in enumerate(src_idx):
        e, fx = OW.norm_owner(v[j] if j < len(v) else None, f"owners[{j}]")
        out.append(e)
        fixes += fx
    return out, fixes


SHOT_MIN_WORDS, SHOT_MAX_WORDS = 5, 70
_NO_TEXT_RE = re.compile(r"\b(?:text|letters?|words?|captions?|titles?|subtitles?|logos?|typography|lettering|writing|signs? reading)\b", re.I)


PAD_STAGES = ("", "closer on the subject, the same place", "a wider view of the same place, later light", "a later moment of the same scene, new angle", "the same place at the end of the action",
              "a last moment, the camera pulled back")


def with_subject(sp: list[dict], desc: str, shows: list[str] | None = None) -> tuple[list[dict], int]:
    """ONE subject, ONE description (round-3 QC 2026-10-29: v010 r01 had 3-4 different people, v009 r02 asked for a car and got a man in 3 shots): every shot prompt of the plan carries the
    subject description word for word; where it is missing it is put in front (kept within SHOT_MAX_WORDS). -> (entries, number changed)."""
    if not desc and not any(shows or []):
        return sp, 0
    out, n = [], 0
    for k, e in enumerate(sp):
        p = str((e or {}).get("prompt") or "")
        own = SG.subject_text(shows[k]) if shows and k < len(shows) else ""         # a shot that starts from a USER image: that image's text is the subject, never the model's `subject_desc`
        d = own or desc
        if e and p and d and d.lower() not in p.lower():
            room = max(SHOT_MAX_WORDS - len(d.split()), 10)
            tail = " ".join(p.split()[:room])
            e = dict(e, prompt=f"{d}, {tail}")
            n += 1
        out.append(e)
    return out, n


def shot_plan_of(v, n: int, pad_from: str = "") -> tuple[list[dict], list[str]]:
    """the reply's `shot_plan` -> (clean entries [{"prompt", "camera"}], fixes): EXACTLY `n` entries, one per shot. Each entry is judged alone (5-70 words, no text / letters in the picture).
    A missing or bad entry used to become {} (the planner's TEMPLATE for that shot, and trailing empties were cut, so the last shot often got a template prompt). With `pad_from` (the
    reply's own `scene` text) the entry is PADDED from it instead (scene + a stage phrase, the previous entry's camera) and the fix says so; more entries than shots are trimmed (logged).
    Without `pad_from` the old behaviour stays (a bad entry becomes {}, trailing empties are cut). [] when the reply has none."""
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.strip() else None
        except ValueError:
            v = None
    if isinstance(v, dict):
        v = v.get("shot_plan")
    if not isinstance(v, list) or not v:
        return [], ["shot_plan: the reply has none; the planner's shot templates are used"]
    out, fixes = [], []
    pad = re.sub(r"\s+", " ", str(pad_from or "")).strip().rstrip(" ,;.")
    if pad and _NO_TEXT_RE.search(pad):
        pad = ""
    if len(v) > n:
        fixes.append(f"shot_plan: {len(v)} entries for {n} shot{'s' if n != 1 else ''}; the first {n} kept")
    last_cam = ""
    for k in range(n):
        e = v[k] if k < len(v) else None
        e = {"prompt": e} if isinstance(e, str) else e
        p = re.sub(r"\s+", " ", str((e or {}).get("prompt") or "")).strip().rstrip(" ,;")
        cam = re.sub(r"\s+", " ", str((e or {}).get("camera") or "")).strip().rstrip(" ,;.")[:60]
        nw = len(p.split())
        why = None
        if not p:
            why = "missing"
        elif nw < SHOT_MIN_WORDS or nw > SHOT_MAX_WORDS:
            why = f"{nw} words ({SHOT_MIN_WORDS}-{SHOT_MAX_WORDS} wanted)"
        elif _NO_TEXT_RE.search(p):
            why = "names text / letters in the picture"
        if why is not None:
            if pad:
                stage = PAD_STAGES[min(k, len(PAD_STAGES) - 1)]
                out.append({"prompt": pad + (", " + stage if stage else "")})                 # no camera: shot_camera.enforce gives it its own (a padded shot must not repeat the one before)
                
                fixes.append(f"shot_plan[{k}]: {why}; padded from the scene text for shot {k + 1} (no template prompt)")
            else:
                out.append({})
                fixes.append(f"shot_plan[{k}]: {why}; the template is used for shot {k + 1}")
        else:
            ent = {"prompt": p, "camera": cam} if cam else {"prompt": p}
            ent.update(PLT.shot_keys(e))                                                          # optional footage options: plate "clean", type_matte, subject_move, plate_pan (guide section 23)
            from tools.fusion import explore_effects as _EE
            if isinstance(e, dict) and _EE.norm(e.get("fx")):
                ent["fx"] = _EE.norm(e.get("fx"))                                                 # one of the hearted explore looks on this shot (guide section 24)
            for ok_ in ("framing", "same_subject_as"):
                if isinstance(e, dict) and str(e.get(ok_) or "").strip():
                    ent[ok_] = str(e[ok_]).strip()[:40]                                           # angle still: the same subject from another framing
            cv = (e or {}).get("same_place", (e or {}).get("continues"))          # `continues` is the old name: the same place from another camera (an angle still), never a last-frame chain
            cv = {"true": True, "yes": True, "false": False, "no": False}.get(cv.strip().lower(), None) if isinstance(cv, str) else cv
            if isinstance(cv, bool) and k > 0:
                ent["continues"] = cv                                                                    # kept under its old key: plan_rules reads it as "same place, another camera"
            out.append(ent)
            last_cam = cam or last_cam
    if not pad:
        while out and not out[-1]:
            out.pop()
    return out, fixes


def normalize(obj, seed: int, previous: list[dict], length: int, effects=(), brief="") -> tuple[dict | None, list[str], list[str]]:
    """parsed reply -> (clean concept, errors, fixes). Errors = unusable fields; fixes = silent repairs (logged)."""
    errors, fixes = [], []
    if not isinstance(obj, dict):
        return None, ["the reply is not a JSON object"], fixes
    b = parse_brief(brief)
    c: dict = {}
    words = obj.get("words")
    if isinstance(words, str):
        words = [w for w in re.split(r"[\n/|]", words) if w.strip()]
    lines, src_idx = [], []
    for j, w in enumerate(words if isinstance(words, list) else []):
        t = re.sub(r"\s+", " ", _CLEAN.sub("", str(w).upper())).strip()
        if t != str(w).strip():
            fixes.append(f"words: {str(w)!r} cleaned to {t!r}")
        if t and (not lines or lines[-1] != t):
            lines.append(t)
            src_idx.append(j)
    ulines = brief_lines(b)
    if ulines:                                              # the USER's on-screen lines are the words, whatever the model wrote (verbatim, in order; owners are matched to them in order)
        fixes.append(f"words: the user's {len(ulines)} on-screen line(s) used verbatim ({' / '.join(ulines)}); the model's {len(lines)} line(s) dropped")
        lines, src_idx = list(ulines), list(range(len(ulines)))
    mode = str(obj.get("shots") or "").strip().lower()
    want = target_mode(previous, seed)
    if mode not in MODES:
        fixes.append(f"shots {mode!r}: not one of {', '.join(MODES)}; {want} used (the suggested multi-shot mode)")
        mode = want                                       # a valid answer of the model is never replaced by the series' spread: that turned multi-shot ideas into single shots
    owners_in, owner_fixes = owners_of(obj.get("owners"), src_idx)       # per cleaned line, before the lines are trimmed / merged / split
    n_lock = lockup_count(length, mode, story_n(b, length))
    lines_in = list(lines)
    if ulines:                                              # verbatim: no 5-7 line rule, no length limit, no merge / split (the planner fits lockups to the shots itself, plan_rules.fit_lockups)
        if n_lock is not None and len(lines) != n_lock:
            fixes.append(f"words: {len(lines)} user line(s) for {n_lock} shots; the planner spreads them over the shots in order")
    elif n_lock is None:                                      # one continuous shot: free lines as before
        if len(lines) > MAX_LINES:
            fixes.append(f"words: {len(lines)} lines, the first {MAX_LINES} kept")
            lines = lines[:MAX_LINES]
        if len(lines) < MIN_LINES:
            errors.append(f"words: {len(lines)} usable lines, {MIN_LINES} to {MAX_LINES} needed (UPPER CASE, 1-{MAX_LINE_WORDS} words each)")
        for t in lines:
            if len(t.split()) > MAX_LINE_WORDS or len(t) > MAX_LINE_CHARS:
                errors.append(f"words: {t!r} is too long ({MAX_LINE_WORDS} words / {MAX_LINE_CHARS} letters at most per line)")
    else:                                                   # ONE lockup per shot: too many / too few are merged / split deterministically (a fix, not an error)
        if not lines:
            errors.append(f"words: no usable lockups, {n_lock} needed (UPPER CASE, 1-{LOCKUP_WORDS} words each)")
        for t in lines:                                     # the prompt asks for 1-3 words; a 4th is tolerated (a merge of two lockups can be longer, it is not checked)
            if len(t.split()) > MAX_LINE_WORDS or len(t) > MAX_LINE_CHARS:
                errors.append(f"words: {t!r} is too long ({LOCKUP_WORDS} words / {MAX_LINE_CHARS} letters at most per lockup)")
        lines, note = PR.fit_lockups(lines, n_lock) if lines else (lines, None)
        if note:
            fixes.append(note)
    c["words"] = lines
    fixes += owner_fixes
    subj0 = re.sub(r"\s+", " ", str(obj.get("subject") or "").strip())[:50]
    desc = re.sub(r"\s+", " ", str(obj.get("subject_desc") or "").strip()).strip(" .,;")
    if len(desc.split()) < 3 or len(desc.split()) > 30:
        if desc:
            fixes.append(f"subject_desc: {len(desc.split())} words (8-20 wanted); the subject name is used")
        desc = subj0
    shows = user_shows(b)                                   # the pictures the user gave: their text IS the subject of the shots that start from them (SG, tools/fusion/subject_guard.py)
    allowed_txt = " ".join(ulines)
    if shows:
        fl = SG.flagged(desc, SG.tokens(" ".join(shows)) | SG.tokens(allowed_txt))
        if fl:
            fixes.append(f"subject_desc: {', '.join(fl[:6])} are not in what the user's image shows; the image's text is used")
            desc = SG.subject_text(next((s for s in shows if s), desc))
    scene_raw = " ".join(re.sub(r"\s+", " ", str(obj.get("scene") or "")).strip().split()[:45])
    if shows and shows[0] and SG.flagged(scene_raw, SG.tokens(" ".join(shows)) | SG.tokens(allowed_txt)):
        fixes.append("scene: names clothing / colours / a subject the user's image does not show; rewritten from the image's text")
        scene_raw = " ".join(SG.guard_prompt(scene_raw, shows[0], allowed_txt).split()[:45])
    sp, sp_fixes = shot_plan_of(obj.get("shot_plan"), 1 if n_lock is None else n_lock, pad_from=scene_raw)
    fixes += sp_fixes
    if sp:
        from tools.fusion import shot_camera as SC                  # a different camera in every shot (no two consecutive share framing or move, no pair repeats)
        sp, nd = with_subject(sp, desc, shows)                # ONE subject, ONE description: verbatim in every shot prompt (the still of a new place is made from it); a user-image shot takes its image's text
        if nd:
            fixes.append(f"shot_plan: the subject description added to {nd} shot prompt{'s' if nd != 1 else ''} (one subject, one description in every shot)")
        if shows:
            sp, ng = guard_user_shots(sp, shows, allowed_txt)
            if ng:
                fixes.append(f"shot_plan: {ng} user-image shot prompt{'s' if ng != 1 else ''} rewritten from the image's own text (no clothing / colour / subject the image does not show)")
        sp, cam_fixes = SC.enforce(sp, 1 if n_lock is None else n_lock, scene_raw)
        fixes += cam_fixes
        if shows:
            sp, nf = guard_user_shots(sp, shows, allowed_txt, framing=True)      # the camera lead names the framing: no second one in the model text
            if nf:
                fixes.append(f"shot_plan: framing phrase removed from {nf} user-image shot prompt{'s' if nf != 1 else ''} (the camera lead carries it)")
        c["shot_plan"] = sp
    idea = re.sub(r"\s+", " ", str(obj.get("idea") or "").strip())[:300]
    if idea:
        c["idea"] = idea
    own = OW.remap_owners(owners_in, lines_in, lines) if owners_in else None       # one entry per final line (merged / split lines take the owner of their first word)
    if own and any(o.get("render") for o in own):
        c["owners"] = [dict(o) for o in own]
    elif owners_in is not None:
        fixes.append("owners: no usable render in the reply; the rules' owners are used (every word flat)")
    c["title"] = re.sub(r"\s+", " ", str(obj.get("title") or "").strip())[:40] or "Untitled"
    subj = re.sub(r"\s+", " ", str(obj.get("subject") or "").strip())[:50]
    if len(_tok(subj)) < 1:
        errors.append("subject: the one main subject of the footage (2-4 words) is missing")
    c["subject"] = subj
    c["subject_desc"] = desc
    scene = re.sub(r"\s+", " ", str(obj.get("scene") or "").strip())
    n = len(scene.split())
    if n < 10:
        errors.append(f"scene: {n} words, 25-50 needed (what the footage shows: subject, place, light, camera distance)")
    elif n > 80:
        scene = " ".join(scene.split()[:60])
        fixes.append("scene: cut to 60 words")
    if shows and shows[0] and SG.flagged(scene, SG.tokens(" ".join(shows)) | SG.tokens(allowed_txt)):
        scene = SG.guard_prompt(scene, shows[0], allowed_txt)           # the scene text makes the stills of the other places: it keeps the picture's subject too
    if desc and desc.lower() not in scene.lower():                  # the scene text makes the first still: it names the same subject, with the same words
        scene = f"{desc}, {scene}"
        fixes.append("scene: the subject description put in front (one subject, one description)")
    c["scene"] = scene
    music = obj.get("music")
    raw = [t.strip() for t in re.split(r"[,;]", ", ".join(music) if isinstance(music, list) else str(music or "")) if t.strip()]
    keep = []
    for t in raw:
        tl = re.sub(r"\b\d+\s*bpm\b", "", t, flags=re.I).strip(" ,")
        if tl and tl.lower() not in [k.lower() for k in keep]:
            keep.append(tl)
    if len(keep) < 3:
        errors.append(f"music: {len(keep)} style tags, 8-14 comma separated tags needed (genre, groove, instruments, mood, texture)")
    keep = keep[:16]
    try:
        bpm = int(round(float(obj.get("bpm"))))
        lo, hi = BPM_RANGE
        if not lo <= bpm <= hi:
            fixes.append(f"bpm {bpm} clamped to {lo}..{hi}")
            bpm = min(max(bpm, lo), hi)
    except (TypeError, ValueError):
        bpm = 100
        fixes.append("bpm missing: 100")
    if brief_bpm(b):
        bpm = brief_bpm(b)                                  # the user's song sets the tempo
    words_m = re.sub(r"[^a-z\- ]", "", str(obj.get("mood") or "").strip().lower()).split()
    mood = words_m[0][:24] if words_m else ""
    if not mood:
        errors.append("mood: one word is missing")
    c["mood"], c["bpm"], c["tags"] = mood, bpm, keep
    c["music"] = ", ".join(keep + ([mood] if mood and mood not in [k.lower() for k in keep] else [])) + f", {bpm} bpm"
    look = str(obj.get("look") or "auto").strip().lower().replace(" ", "_")
    if look not in LOOKS:
        fixes.append(f"look {look!r} unknown: auto")
        look = "auto"
    earlier_sys = earlier_systems(previous)
    sysm = PS.choose(seed, mood, bpm, look=look if look in PS.FAMILY else None, earlier=earlier_sys, palette_mood=cues(seed, previous)["mood"], words=c.get("words") or [], length_s=length, effects=effects)      # the piece's SYSTEM (module, io pair, font shortlist, palette, camera, yaw band, photographic style): drawn here, held by the design
    if look not in PS.FAMILY:                                # auto / mech / ring ...: the type family follows the system (the code keeps only the readable strand / slab modules anyway)
        if look != "auto":
            fixes.append(f"look {look!r} -> {sysm['look']!r} (the piece's system keeps the readable module families)")
        look = sysm["look"]
    c["look"] = look
    c["system"] = sysm
    req_eff, unk_eff = EM.norm_effects(effects)
    if req_eff or unk_eff:
        c["effects"] = list(req_eff)                                  # the user's required effects ride with the concept (request_extras -> plan_rules.effects_of)
        if unk_eff:
            fixes.append("effects: unknown name(s) ignored: " + ", ".join(unk_eff) + " (known: " + ", ".join(EM.EFFECTS) + " and the cut kinds)")
    c["cut_kinds"] = list((sysm.get("cuts") or {}).get("kinds") or [])         # carried to the next pieces of the batch (their draw excludes the last two pieces' kinds)
    c["cues"] = {k: v for k, v in cues(seed, previous).items() if k in ("place", "light", "weather", "colour", "genre", "register")}
    c["shots"] = mode
    c["length"] = int(length)
    c["seed"] = int(seed)
    eo = [p["opening"] for p in previous if isinstance(p, dict) and isinstance(p.get("opening"), dict) and p["opening"].get("form_in")]
    if eo:
        c["earlier_openings"] = eo                                 # how the earlier pieces of the batch opened: the design step tells the model to open differently and swaps a repeated opening (openings.py)
    wh = WF.collect(obj)                    # the model's reasons (idea_why, shot_plan[].why, owners[].why): kept for the reader, never an error when missing, not used by the planner
    if wh:
        c["whys"] = wh
    if b and c["words"] is not None:
        apply_brief(c, b)
    return (None if errors and not c["words"] else c), errors, fixes


def request_extras(c: dict) -> str:
    """the 8th output of MographConceptFinal (-> MographRequest.owners): the owners list as before when the concept has nothing else; otherwise the object
    {"owners": [...], "shot_plan": [...], "idea": "..."} (plan_rules.normalize_request reads both forms, so no graph wiring changes)."""
    own, sp, idea = c.get("owners") or [], c.get("shot_plan") or [], c.get("idea") or ""
    eo = c.get("earlier_openings") or []
    sy = c.get("system") if isinstance(c.get("system"), dict) else None
    ef = [x for x in (c.get("effects") or []) if isinstance(x, str)]
    fo = story_fonts(c)                                           # the user's per-line fonts from the storyboard: win over the planner's / designer's font rules (tools/fusion/font_pins.py)
    cu = story_cuts(c)                                            # the user's per-shot cut moves (3D kinds / hold): pinned in the design step (qwen_piece.finish_design)
    sn = story_n(c.get("brief") or {}, c.get("length") or 0) if c.get("length") else None     # the storyboard's shot count: the planner cuts exactly that many shots (plan_rules.story_shot_count)
    dr = c.get("direction") if isinstance(c.get("direction"), str) else ""        # the user's typed direction (apply_brief)
    sh = [str(x) for x in c.get("shows") or []] if isinstance(c.get("shows"), list) else []     # the media-read text of each user image (apply_brief)
    if not sp and not idea and not eo and not sy and not ef and not fo and not cu and not sn and not dr and not sh:
        return json.dumps(own, ensure_ascii=False)
    return json.dumps({"owners": own, "shot_plan": sp, "idea": idea, **({"earlier_openings": eo} if eo else {}), **({"system": sy} if sy else {}), **({"effects": ef} if ef else {}), **({"fonts": fo} if fo else {}), **({"cuts": cu} if cu else {}), **({"story_shots": sn} if sn else {}), **({"direction": dr} if dr else {}), **({"shows": sh} if sh else {})}, ensure_ascii=False)


def parse(reply: str, previous: list[dict], seed: int, length: int, effects=(), brief="") -> dict:
    """-> dict(concept | None, errors, clashes, fixes, ok). ok = a clean, unique concept. With a user `brief` the words are the user's (no word / place / category clash: the user chose them)."""
    obj = QP.extract_json(reply or "")
    if obj is None:
        return dict(concept=None, errors=["the reply contained no parseable JSON object; return only the JSON object"], clashes=[], fixes=[], ok=False)
    c, errors, fixes = normalize(obj, seed, previous, length, effects, brief)
    cl = clashes(c, previous) if c is not None and not errors and not parse_brief(brief) else []
    return dict(concept=c, errors=errors, clashes=cl, fixes=fixes, ok=c is not None and not errors and not cl)


# --------------------------------------------------------------------------------------------- the seeded rules variation
BANK = [
    dict(title="Keeper of the light", subject="lighthouse keeper", mood="solemn", look="slab", shots="chain", bpm=88,
         words=["THE SEA", "REMEMBERS", "EVERY SHIP", "EVERY NAME", "KEEP THE LIGHT"],
         scene="an old lighthouse keeper in a yellow coat standing on the gallery of a stone lighthouse as a storm breaks over a dark sea, wind-torn spray, one beam of light cutting the clouds, wide shot from below",
         tags=["cinematic orchestral", "low strings", "taiko drums", "brass swells", "epic", "dark", "big reverb"]),
    dict(title="Late order", subject="noodle stall cook", mood="nostalgic", look="strand", shots="cut", bpm=80,
         words=["LATE", "NIGHT", "STEAM RISES", "NOBODY SLEEPS", "ORDER ANOTHER"],
         scene="a cook behind a tiny neon noodle stall on a rainy city street at night, steam rising into pink and green neon light, wet pavement reflections, medium shot through the rain",
         tags=["lo-fi hip hop", "dusty drums", "warm rhodes piano", "vinyl crackle", "mellow bass", "relaxed"]),
    dict(title="Salt flat", subject="wild horse", mood="free", look="mech", shots="single", bpm=120,
         words=["RUN", "FOR NOTHING", "FOR EVERYTHING", "DUST", "FOREVER"],
         scene="a wild white horse galloping across a vast white salt flat at golden hour, a long plume of dust behind it, low camera tracking alongside, heat haze on the horizon",
         tags=["desert rock", "twangy electric guitar", "driving drums", "fuzz bass", "hypnotic groove", "wide open"]),
    dict(title="Last turn", subject="ballet dancer", mood="tender", look="ring", shots="chain", bpm=70,
         words=["CURTAIN", "UP", "ONE LAST", "TURN", "APPLAUSE ECHOES"],
         scene="a ballet dancer in a white dress turning alone on the bare stage of an empty old theatre, a single spotlight, dust in the beam, red velvet seats in the dark, medium-wide shot",
         tags=["solo piano", "slow strings", "soft harp", "intimate", "emotional", "minimal", "spacious"]),
    dict(title="Reboot", subject="robot gardener", mood="hopeful", look="voxel", shots="cut", bpm=100,
         words=["SEED", "SIGNAL", "BLOOM", "REBOOT", "GROW"],
         scene="a small rusty robot gardener watering glowing blue plants inside a cracked greenhouse at dawn, soft mist, light shafts through broken glass, camera at plant height",
         tags=["ambient electronic", "warm synth pads", "soft glitchy percussion", "gentle arpeggios", "dreamy", "airy"]),
    dict(title="Under the bridge", subject="skateboarder", mood="defiant", look="slab", shots="cut", bpm=128,
         words=["PUSH", "OFF", "THE GROUND", "WE ROLL", "UNTIL DARK"],
         scene="a skateboarder pushing along the concrete under a highway bridge at dusk, orange sodium lights flickering on, long shadows, low camera at wheel level, side tracking shot",
         tags=["driving breakbeat", "slap bass", "bright synth stabs", "funky guitar", "energetic", "punchy drums"]),
    dict(title="Fog line", subject="fishing boat", mood="eerie", look="stack", shots="single", bpm=62,
         words=["SLOW", "WATER", "NOTHING", "TO SEE", "ONLY LISTEN"],
         scene="a small fishing boat drifting through thick grey fog on a still harbour, one yellow lantern at the bow, the water barely moving, distant shapes of masts, static wide shot",
         tags=["dark ambient", "sub bass drone", "distant bells", "sparse piano notes", "tense", "cold texture"]),
    dict(title="Sunday kitchen", subject="old woman dancing", mood="joyful", look="particles", shots="chain", bpm=140,
         words=["SUNDAY", "MORNING", "FLOUR", "AND SONG", "DANCE ANYWAY"],
         scene="an old woman in an apron dancing in a sunny farmhouse kitchen with flour in the air, light through a window, copper pots on the wall, medium shot, warm morning light",
         tags=["gypsy jazz", "acoustic guitar", "upright bass", "violin", "swing", "cheerful", "bouncy"]),
    dict(title="Red dune", subject="astronaut", mood="lonely", look="ring", shots="cut", bpm=85,
         words=["ONE", "STEP", "THEN", "ANOTHER", "THE WAY BACK"],
         scene="an astronaut in a white suit walking up a steep red sand dune under a pale pink sky, two small moons, footprints behind, long shot from below, thin dust blowing off the ridge",
         tags=["space ambient", "slow synth arpeggio", "deep pads", "soft kick", "cinematic", "vast"]),
    dict(title="Louder", subject="drummer", mood="restless", look="mech", shots="chain", bpm=110,
         words=["HIT", "THE ROOM", "AGAIN", "THE ROOM HITS BACK", "LOUDER"],
         scene="a drummer playing a single huge drum kit in a flooded stone church, ankle-deep water splashing with every hit, shafts of light from high windows, medium shot from the front",
         tags=["post-rock", "huge reverb drums", "tremolo guitars", "building crescendo", "epic", "raw"]),
    dict(title="Gutter river", subject="paper boat", mood="playful", look="strand", shots="cut", bpm=95,
         words=["SMALL", "BOATS", "BIG", "RIVER", "SAIL ON"],
         scene="a folded paper boat floating down a rainy street gutter past leaves and bottle caps, water rushing, a child's rubber boots at the edge of the frame, very low macro camera",
         tags=["indie folk", "acoustic guitar", "upright bass", "brushed drums", "warm", "gentle", "light"]),
    dict(title="Sunrise climb", subject="cyclist", mood="determined", look="slab", shots="single", bpm=124,
         words=["CLIMB", "LUNGS BURN", "THE SUMMIT", "IS A", "SUNRISE"],
         scene="a road cyclist climbing a switchback mountain pass at sunrise, breath steaming in cold air, golden light on the peaks, a drone shot circling slowly from the side",
         tags=["uplifting electronic", "four-on-the-floor kick", "pulsing bass", "soaring synth lead", "driving", "bright"]),
    dict(title="Platform", subject="street dancer", mood="electric", look="stack", shots="cut", bpm=92,
         words=["FLOOR", "IS", "THE STAGE", "MOVE", "LOUD"],
         scene="a street dancer freezing mid-move on a test clip platform as a train blurs past, green fluorescent light, tiled walls, a beat-up boombox at their feet, medium-wide shot",
         tags=["boom bap hip hop", "heavy 808 bass", "snappy snare", "brass stabs", "gritty", "swagger"]),
    dict(title="Snow stair", subject="monk on stairs", mood="calm", look="particles", shots="chain", bpm=64,
         words=["QUIET", "SNOW", "EACH STEP", "A BREATH", "BEGIN AGAIN"],
         scene="a monk in an orange robe slowly climbing a long snowy stone stair toward a wooden temple, snow falling, soft grey light, lanterns glowing, wide shot from the bottom of the stair",
         tags=["meditative", "shakuhachi flute", "low frame drum", "singing bowl", "minimal", "spacious"]),
    dict(title="Kite weather", subject="kid with a kite", mood="curious", look="particles", shots="chain", bpm=108,
         words=["WIND", "TUGS", "THE STRING", "LET GO", "LOOK UP"],
         scene="a small kid in a red raincoat running along a windy grass dune with a huge yellow kite, grey sky breaking into light, sea grass bending, low camera behind the child, wide shot",
         tags=["indie pop", "bright glockenspiel", "handclaps", "bouncy bass", "light strings", "playful", "sunny"]),
    dict(title="Lantern night", subject="festival crowd", mood="wonder", look="ring", shots="cut", bpm=96,
         words=["THOUSAND", "SMALL LIGHTS", "RISE TOGETHER", "NO ONE", "ALONE"],
         scene="a dense crowd on a river bank at night releasing paper lanterns that drift over the black water, warm orange glow on faces, bridge and city lights far behind, wide shot from the water",
         tags=["world fusion", "frame drums", "oud", "choir pads", "warm bass", "ceremonial", "glowing"]),
    dict(title="Stag", subject="red stag", mood="haunted", look="mech", shots="single", bpm=66,
         words=["HE STANDS", "BREATH", "IN BLUE", "SNOW REMEMBERS", "STEPS"],
         scene="a red stag standing motionless in a snow-covered birch wood at dawn, breath steaming, pale blue light between the trunks, a long lens compresses the trees, very still frame",
         tags=["neo-classical", "solo cello", "felt piano", "air textures", "sparse", "frozen", "slow"]),
    dict(title="Ice cave", subject="glacier cave", mood="serene", look="stack", shots="chain", bpm=72,
         words=["BLUE", "FROM", "INSIDE", "THE ICE SINGS", "DRIP BY DRIP"],
         scene="the inside of a glacier ice cave, translucent turquoise walls, a thin stream running across the floor, sun filtering through the ceiling, no people, slow forward glide through the cave",
         tags=["ambient", "glass pads", "singing bowls", "water textures", "sub bass", "weightless", "clear"]),
    dict(title="Last tram", subject="night tram", mood="bittersweet", look="slab", shots="cut", bpm=102,
         words=["LAST", "TRAM", "DEPOT", "WINDOWS FOGGED", "SOMEONE WAVES"],
         scene="a lit yellow tram gliding through an empty rainy city boulevard at midnight, reflections stretching on the tracks, fogged windows glowing, camera low at the kerb, medium-wide shot",
         tags=["jazz noir", "muted trumpet", "brushed drums", "upright bass", "rhodes", "smoky", "late"]),
    dict(title="Owl hour", subject="barn owl", mood="fierce", look="strand", shots="single", bpm=118,
         words=["QUIET", "WINGS", "SEE", "EVERYTHING", "STRIKE"],
         scene="a barn owl on a beam inside a dusty old wooden barn, shafts of dusty light through the slats, hay and rusted tools below, the owl turns its head toward the camera, close shot",
         tags=["dark electronic", "pulsing sub", "metallic percussion", "tense strings", "stealth", "sharp"]),
    dict(title="Clay", subject="potter", mood="wistful", look="mech", shots="chain", bpm=76,
         words=["HANDS", "REMEMBER", "WHAT THE CLAY", "FORGETS", "TURN"],
         scene="an elderly potter shaping a bowl on a spinning wheel in a sunlit workshop, wet clay on his hands, shelves of finished pots, dust floating in a window beam, medium close shot from the side",
         tags=["acoustic folk", "fingerpicked guitar", "soft accordion", "light shaker", "warm", "earthy"]),
    dict(title="Balloon", subject="hot air balloon", mood="triumphant", look="voxel", shots="cut", bpm=124,
         words=["CUT", "THE ROPES", "RISE", "ABOVE THE FOG", "FREE AT LAST"],
         scene="a single striped hot air balloon lifting out of a valley of thick morning fog toward a bright orange sunrise, mountain tops poking through the clouds, wide shot from below the basket",
         tags=["epic indie rock", "driving drums", "clean electric guitar", "soaring synths", "anthemic", "bright"]),
]


def _fit(c: dict) -> dict:
    """the bank's lines fitted to one lockup per shot of the concept's mode and length (merge / split, plan_rules.fit_lockups); the note goes to c["lockups"]."""
    n = lockup_count(c["length"], c["shots"])
    if n is not None:
        c["words"], note = PR.fit_lockups(list(c["words"]), n)
        if note:
            c["lockups"] = note
    return c


def _from_bank(e: dict, seed: int, length: int) -> dict:
    c = copy.deepcopy(e)
    c["music"] = ", ".join(c["tags"] + [c["mood"]]) + f", {c['bpm']} bpm"
    c["length"], c["seed"] = int(length), int(seed)
    return c


def rules_variation(seed: int, previous: list[dict], length: int) -> dict:
    """a bank concept that clashes with none of the earlier ones, starting at a seeded index, shots mode spread like the model's; when the bank
    is used up the least clashing entry is taken and its words / mood are rotated so it is still not a copy."""
    n = len(BANK)
    start = random.Random(f"bank:{seed}").randrange(n)
    want = target_mode(previous, seed)
    order = [BANK[(start + i) % n] for i in range(n)]
    best, best_n = None, 99
    for e in order:
        c = _from_bank(e, seed, length)
        k = len(clashes(c, previous))
        if k == 0:
            c["shots"] = want if want in MODES else c["shots"]
            c["source"] = "rules"
            return _fit(c)
        k = k + 0.5 * len(category_clashes(c, previous))        # between two clashing entries the one that repeats only a category / setting is the better fallback
        if k < best_n:
            best, best_n = c, k
    best["words"] = best["words"][1:] + best["words"][:1]
    best["mood"] = next((m for m in MOODS if m not in {p.get("mood") for p in previous}), best["mood"])
    best["music"] = ", ".join(best["tags"] + [best["mood"]]) + f", {best['bpm']} bpm"
    best["shots"], best["source"] = want, "rules (bank used up)"
    return _fit(best)


def final(reply2: str | None, first: dict, previous: list[dict], seed: int, length: int, effects=(), brief="") -> tuple[dict, str]:
    """the concept to use + a report line. `first` = parse() of the first reply. ok -> as is; else the repair reply if it is clean, else the rules."""
    if first.get("ok"):
        c = dict(first["concept"], source="model")
        return c, "concept: the model's first reply was clean and unique"
    problems = first["errors"] + first["clashes"]
    repair_note = ""
    if reply2:
        r2 = parse(reply2, previous, seed, length, effects, brief)
        if r2["ok"]:
            return dict(r2["concept"], source="model (repair)"), "concept: the repair reply fixed: " + "; ".join(problems[:4])
        repair_note = "; repair reply: " + "; ".join((r2["errors"] + r2["clashes"])[:3])
    c = rules_variation(seed, previous, length)
    if EM.norm_effects(effects)[0]:
        c["effects"] = EM.norm_effects(effects)[0]
    apply_brief(c, parse_brief(brief))                       # the user's words / concept / look / bpm hold over the bank concept too
    return c, "concept: seeded rules variation (" + c["source"] + "), because: " + "; ".join(problems[:5]) + repair_note


def parse_effects(text) -> list:
    """the user-requested effects of a concept step: `{"concepts": [...], "effects": ["outline", ...]}` in the `previous` STRING (no new graph input), or a bare `{"effects": ...}`; [] otherwise."""
    try:
        v = json.loads(text) if isinstance(text, str) and text.strip() else text
    except ValueError:
        return []
    if isinstance(v, dict) and v.get("effects") is not None:
        e = v["effects"]
        return [x.strip() for x in re.split(r"[,;|\n]+", e) if x.strip()] if isinstance(e, str) else [str(x) for x in e] if isinstance(e, list) else []
    return []


def parse_previous(text: str) -> list[dict]:
    """the `previous` STRING input: a JSON list of earlier concepts (or {"concepts": [...]}); anything unreadable = none."""
    try:
        v = json.loads(text) if str(text or "").strip() else []
    except ValueError:
        return []
    if isinstance(v, dict):
        v = v.get("concepts", [])
    return [p for p in v if isinstance(p, dict)] if isinstance(v, list) else []
