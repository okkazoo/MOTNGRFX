"""plan_rules.py - the rule-based piece planner of the Mograph app (DESIGN.md section 2). Pure Python, no ComfyUI.

    plan_piece(request, music, brief_text=None)  ->  piece dict (the 15_fusion/pieces/P*.json schema + layout / sizes /
                                                      technique / quality / brief / plan)
    finalize_piece(piece, music=None)            ->  the same fields for a hand-written P*.json (MographPieceFromJSON)
    music_plan(request, brief)                   ->  tags / bpm / key / seconds for the ACE-Step node
    assign_batch(pieces, seed) / plan_batch(requests, musics, seed)  ->  EVERY module / transition / io style / camera move / hold layout
                                                      used across a batch (tools/fusion/batch_plan.py); plan_piece(.., assignment=) applies one piece's slice

Rules write every field; the Qwen brief (mograph/fusion_brief.py) only supplies creative text (scene, still prompt, motion
lines, music tags, technique, mood, palette, signoff). A missing, empty or broken brief changes nothing but that text: the
spec is a pure function of (request, music analysis, brief). Seeds per sub-task: zlib.crc32(f"{seed}:{tag}").

Timing: lines = auto_lines(words); every word's HIT lands on a downbeat (hit_on 'bar'), windows are bars proportional to the
letter count (>= 1 bar each), the last ~1.2 s are left for the signoff. Spans are found by running the real
``spec.plan_timing`` on the beat grid, so the plan is exact, and ``dry_run`` builds the scene at 576x1024 to catch what the
estimate cannot (notes such as "timeline longer than the clip"); on a note the bar budget shrinks and the plan is rebuilt (up
to 3 times).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mograph import font_registry as fr  # noqa: E402
from mograph import fonts_liked as fl  # noqa: E402  (hard allowlist: mograph/fonts/liked.json)
from mograph import fusion_brief as fb  # noqa: E402
from mograph.brief import PALETTES, _MOOD_SYN, order_palette  # noqa: E402
from tools.fusion import blocks as bk  # noqa: E402  (lego passages of the v005 designs)
from tools.fusion import hold_layouts as hl  # noqa: E402  (named hold compositions from 16_app/LAYOUT_NOTES.md)
from tools.fusion import owner as OW  # noqa: E402  (one owner per word: flat | ltx | handoff; tools/fusion/owner.py)
from tools.fusion import plate_layers as PLT  # noqa: E402  (optional per-shot plate / type-matte / angle keys of the concept's shot plan)
from tools.fusion import effects_menu as EM  # noqa: E402  (named effects: drawn + requested)
from tools.fusion import stable_type as ST  # noqa: E402  (every word lands centred and holds still for >= 1 beat: stacked / rolled layouts)
from tools.fusion import piece_system as PS  # noqa: E402  (the piece's seeded system: module, io pair, faces, palette, camera, yaw band, photographic style)

RECIPES = Path(__file__).resolve().parent / "recipes"
FPS = 24
LENGTHS = (8, 12, 16, 24)               # the app's quick picks; any whole second in MIN_LENGTH..MAX_LENGTH is accepted
MIN_LENGTH, MAX_LENGTH = 6, 60
XFADE = fb.XFADE
SHOT_FRAMES = fb.SHOT_FRAMES
MAX_SHOTS = fb.MAX_SHOTS
LONG_SHOT = 193            # longest planned shot (frames): the planner emits IC-control shots (cap 193, ltx_ic.KINDS); plain i2v is proven to 385
RESERVE_S = 0.9            # kept free at the end for the signoff (the tail is then filled by hold / signoff, see fill_tail)
SIGNOFF_LEAD_S = 1.15      # signoff starts this long before the cut (P02: 10.6 s of 11.75 s) unless fill_tail moves it
LAG_S = 0.35              # a lockup may start this much after its shot (the engine's cut / transition gaps) before the previous hold is pulled in
TAIL_MAX_S = 0.9           # the last word may leave at most this long before the cut ...
TAIL_TARGET_S = 0.6        # ... fill_tail grows its hold until about this much is left (signoff covers it)
SIGNOFF_OVERLAP_S = 0.45   # the signoff fades in this long before the last word is gone: no dead air between them
SIGNOFF_MIN_S = 1.0        # the signoff is on screen at least this long
FORMATS = ("9:16", "16:9")
DRY_SIZE = (576, 1024)
MAX_RETRIES = 3
NO_VOX = "instrumental, no vocals, no singing, no choir"      # as tools/fusion/songs.py
TECHNIQUES = tuple(fb.TECHNIQUES)
OWNERS = True               # a new plan gives every word an owner (render / ltx_at / material), the shot prompts the letter clause and the control guide; False = the pre-owner plan
SHOT_GAP_S = 0.35           # one lockup per shot: a word's exit ends this long before the next shot starts (the transition lives in the cut's last 0.35 s; the engine's smallest stated gap)

# look = family of word modules (by base class); "auto" lets energy + seed choose
LOOKS: dict[str, list[str]] = {
    "strand": ["lanes", "pipework", "circuit"],
    "mech": ["depth_echo", "hollow_tubes", "topo_tower", "hinge_unfold", "xray_unfold", "candy_orbit", "depth_tunnel"],
    "slab": ["beat_bars", "bold_thin_echo", "gradient_fade", "scale_clash", "slats", "solid_hollow", "staircase"],
    "stack": ["spectrum", "tangent_spray"],
    "ring": ["fly_into"],           # `outflow` retired 2026-11-02 (design_defaults.RETIRED_MODULES)
    "voxel": ["voxel_orbit", "voxel_roll"],
    "particles": ["particles"],
}
LOOK_NAMES = ["auto"] + list(LOOKS)
_AUTO_FAMILIES = {"calm": ["strand", "mech", "ring", "stack"], "mid": ["strand", "mech", "slab", "stack"],
                  "kinetic": ["slab", "strand", "stack", "ring"]}


def sub_seed(seed, tag: str) -> int:
    """seed for one sub-task: zlib.crc32(f"{seed}:{tag}") (like mograph.brief._salt)."""
    return zlib.crc32(f"{int(seed)}:{tag}".encode("utf-8"))


def _pick(seed, tag: str, items):
    items = list(items)
    return items[sub_seed(seed, tag) % len(items)]


def _r(x, nd=3):
    return float(round(float(x), nd))


# =================================================================================================== request
def _owners_of(v) -> list | None:
    """the request's `owners` (the concept step's per-lockup choice: [{render, ltx_at, material}, ...] one per line of `words`; a list or its JSON text) -> a list
    of dicts, or None (none given / unreadable: every word then gets the rules' owner)."""
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.strip() else None
        except ValueError:
            return None
    if isinstance(v, dict):
        v = v.get("owners")
    if not isinstance(v, list) or not v:
        return None
    return [e if isinstance(e, dict) else {} for e in v]


def shot_plan_of(v) -> list:
    """the request's per-shot footage plan (the concept step's `shot_plan`: [{prompt, camera}, ...], {} = use the template for that shot) from a list, its JSON text, or the
    owners object {"owners", "shot_plan", "idea"} -> a list of dicts, [] when there is none."""
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.strip() else None
        except ValueError:
            return []
    if isinstance(v, dict):
        v = v.get("shot_plan")
    if not isinstance(v, list):
        return []
    return [e if isinstance(e, dict) else {} for e in v]


def openings_of(*vs) -> list:
    """the request's `earlier_openings` (how the earlier pieces of a batch opened, openings.py) from the owners object {"owners", "shot_plan", "idea", "earlier_openings"}; [] when there are none."""
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict) and isinstance(v.get("earlier_openings"), list):
            return [e for e in v["earlier_openings"] if isinstance(e, dict) and e.get("form_in")]
    return []


def cut_pins_of(*vs) -> list:
    """the user's per-shot cut kinds (storyboard camera moves): a list [kind | None, ..] under `cuts` in the owners extras object (or `cut_pins` in the request); kinds validated by cut_kinds, else None. [] = none chosen."""
    from tools.fusion import cut_kinds as CK
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            v = v.get("cuts") if v.get("cuts") is not None else v.get("cut_pins")
        if isinstance(v, (list, tuple)) and v and all(x is None or isinstance(x, str) for x in v):
            out = [CK.normalize(x) if x else None for x in v]
            if any(out):
                return out
    return []


def story_shots_of(*vs):
    """the storyboard's shot count (STORYBOARD v12): `story_shots` in the request or in the owners extras object (concept.request_extras). int >= 1, else None (= today's rule); validated in _build."""
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            v = v.get("story_shots")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1:
            return int(v)
    return None


def FP_pin_for(text, i, n, pins):
    from tools.fusion import font_pins as FP
    return FP.pin_for(text, i, n, pins)


def FP_apply(words, pins, fixes=None):
    from tools.fusion import font_pins as FP
    return FP.apply(words, pins, fixes)


def effects_of(*vs) -> list:
    """the request's `effects` (user-requested named effects, tools/fusion/effects_menu.py): a list or a comma separated string under `effects` in the request, or the same key inside the owners
    extras object; [] when there are none (the draw is random as before). Unknown names are kept as written: they are reported by the planner, never fatal."""
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            v = v.get("effects")
        if isinstance(v, str) and v.strip() and not v.strip().startswith(("{", "[")):
            return [x.strip() for x in re.split(r"[,;|\n]+", v) if x.strip()]
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            return [x.strip() for x in v if x.strip()]
    return []


def system_of(*vs) -> dict | None:
    """the request's piece system (tools/fusion/piece_system.py, drawn at the concept step) from the owners object {"owners", ..., "system"} or the `system` key itself; None when there is none."""
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            sy = v.get("system") if isinstance(v.get("system"), dict) else v
            if isinstance(sy, dict) and sy.get("module") and isinstance(sy.get("io"), dict) and isinstance(sy.get("fonts"), list):
                return sy
    return None


def idea_of(*vs) -> str:
    """the concept's one-line visual idea from the request's `idea` or the owners object."""
    for v in vs:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict) and str(v.get("idea") or "").strip():
            return " ".join(str(v["idea"]).split())[:300]
        if isinstance(v, str) and v.strip() and not v.strip().startswith(("{", "[")):
            return " ".join(v.split())[:300]
    return ""


SHOT_PROMPT_WORDS = (5, 70)


def model_shot_prompt(entry) -> str:
    """a concept shot-plan entry -> the shot prompt body ("<prompt>, camera: <camera>"), or "" when it is not usable (the template is then used for that shot)."""
    if not isinstance(entry, dict):
        return ""
    p = " ".join(str(entry.get("prompt") or "").split()).strip(" ,;")
    n = len(p.split())
    if not (SHOT_PROMPT_WORDS[0] <= n <= SHOT_PROMPT_WORDS[1]) or re.search(r"\b(?:text|letters?|captions?|logos?|typography|lettering)\b", p, re.I):
        return ""
    cam = " ".join(str(entry.get("camera") or "").split()).strip(" ,;.")[:60]
    if cam and not set(re.findall(r"[a-z]+", cam.lower())) <= set(re.findall(r"[a-z]+", p.lower())):
        p = f"{p}, camera: {cam}"
    return p


def place_still(sp_plan: list, shot_index: int, fmt: str = "9:16", style: str | None = None, framing: str | None = None) -> str:
    """the still prompt of a NEW PLACE from the concept's shot plan entry of the shot that starts it (what the subject does THERE, the place, the light), without the
    camera clause; "" when that shot has no usable entry (the planner then varies the first still: `_STILL_ANGLES`). One still per place, each its own place."""
    if not (0 <= shot_index < len(sp_plan)):
        return ""
    body = model_shot_prompt(sp_plan[shot_index])
    if not body:
        return ""
    body = re.sub(r",\s*camera:.*$", "", body).strip(" ,;")
    from tools.fusion import angle_still as AS
    from tools.fusion import shot_camera as SC
    fr = framing or SC.framing_of(sp_plan[shot_index])           # the place's still is framed like its first shot (a new place = a new framing, shot_camera.distinct)
    if fr:
        body = AS.lead_framing(body, fr)                         # the framing LEADS the prompt (an image model reads the front first; at the end it lost to the subject description)
    return f"{body}, {fr + ' shot, ' if fr else ''}{(style or _STILL_STYLE) + ', ' if (style or _STILL_STYLE) else ''}{_FRAME_WORDS.get(fmt, _FRAME_WORDS['9:16'])}"


def normalize_request(req: dict) -> dict:
    """Validate the app inputs. Raises ValueError with a message for the page."""
    r = dict(req or {})
    words = str(r.get("words") or "").strip()
    if not fb.words_lines(words).strip():
        raise ValueError("words are empty: type the words the video should show")
    try:
        length = int(r.get("length") or 12)
    except (TypeError, ValueError):
        raise ValueError(f"length {r.get('length')!r} is not a number") from None
    length = max(MIN_LENGTH, min(MAX_LENGTH, length))          # the requested length is honoured; the music is made at it (whole bars, t2m_music.fit_bpm)
    look = str(r.get("look") or "auto").strip().lower().replace(" ", "_")
    if look not in LOOK_NAMES and look not in {m for ms in LOOKS.values() for m in ms}:
        raise ValueError(f"look {look!r} unknown; options: {', '.join(LOOK_NAMES)} or a module name")
    quality = str(r.get("quality") or "final").strip().lower()
    if quality not in ("preview", "draft", "final"):
        quality = "final"
    fmt = str(r.get("format") or "9:16").strip().lower().replace("x", ":").replace("/", ":")
    fmt = {"portrait": "9:16", "vertical": "9:16", "landscape": "16:9", "horizontal": "16:9"}.get(fmt, fmt)
    if fmt not in FORMATS:
        raise ValueError(f"format {r.get('format')!r} unknown; options: {', '.join(FORMATS)}")
    try:
        seed = int(r.get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    shots_given = bool(r["shots_given"]) if "shots_given" in r else bool(str(r.get("shots") or "").strip())      # a normalised request keeps the flag
    shots = str(r.get("shots") or "cut").strip().lower()
    if shots not in SHOT_MODES:
        raise ValueError(f"shots {r.get('shots')!r} unknown; options: {', '.join(SHOT_MODES)}")
    owners = _owners_of(r.get("owners"))
    from tools.fusion import font_pins as FP     # the user's per-line fonts (storyboard): `font_pins` (made) or `fonts` (aligned with the lines) in the request / the owners extras object
    pins = FP.pins_of(r.get("font_pins"), r.get("fonts"), r.get("owners"), lines=[x for x in fb.words_lines(words).splitlines() if x.strip()])
    return {"words": words, "owners": owners, "font_pins": pins, "cut_pins": cut_pins_of(r.get("cut_pins"), r.get("owners")), "story_shots": story_shots_of(r.get("story_shots"), r.get("owners")), "scene": " ".join(str(r.get("scene") or "").split()), "shots": shots, "signoff": bool(r.get("signoff", False)),
            "music": " ".join(str(r.get("music") or "").split()), "length": length, "look": look, "seed": seed & 0xFFFFFFFF,
            "quality": quality, "format": fmt, "image": r.get("image"), "audio": r.get("audio"), "shots_given": shots_given,
            "shot_plan": shot_plan_of(r.get("shot_plan")) or shot_plan_of(r.get("owners")), "idea": idea_of(r.get("idea"), r.get("owners")), "earlier_openings": openings_of(r.get("earlier_openings"), r.get("owners")),
            "system": system_of(r.get("system"), r.get("owners")), "legacy_chain": bool(r.get("legacy_chain")),
            "effects": effects_of(r.get("effects"), r.get("owners")), "layout_seed": _layout_seed_of(r.get("layout_seed"))}


def _layout_seed_of(v):
    """the request's `layout_seed` (the Randomise-layout preview's \"Use this layout\", tools/fusion/layout_preview.py): an int >= 1 re-draws the LAYOUT side of the piece system (fonts, cut kinds, outline,
    layouts, hold life, palette pick); None / 0 = the planner's own draw."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return (n & 0xFFFFFFFF) if n > 0 else None


# =================================================================================================== music side
def energy_of(music: dict) -> float:
    """0..1 from the tempo (75 %) and the onset density of the cut (25 %)."""
    bpm = float(music.get("bpm") or 100.0)
    n = int(music.get("nframes") or 0)
    on = (music.get("curves") or {}).get("onset") or []
    on = on[:n] if n else on
    dens = sum(on) / len(on) if on else 0.15
    e = 0.75 * min(1.0, max(0.0, (bpm - 60.0) / 90.0)) + 0.25 * min(1.0, dens / 0.3)
    return _r(min(1.0, max(0.0, e)), 3)


def tier_of(energy: float) -> str:
    return "calm" if energy < 0.40 else ("mid" if energy < 0.72 else "kinetic")


_MOOD_BPM = {"calm": 72, "warm": 90, "luxury": 84, "nostalgic": 96, "dark": 100, "techno": 124, "playful": 110, "hype": 132}
_MOOD_KEY = {"calm": "D minor", "warm": "G major", "luxury": "Bb major", "nostalgic": "A minor", "dark": "F minor",
             "techno": "F minor", "playful": "A major", "hype": "E minor"}
_MOOD_TAGS = {
    "calm": "ambient cinematic, soft felt piano, slow swelling pads, airy strings, spacious reverb, floating, wistful",
    "warm": "warm acoustic groove, soft kick, round bass, felt piano, hand percussion, sunlit and intimate",
    "luxury": "slow elegant downtempo, velvet Rhodes chords, deep soft bass, brushed drums, glossy and refined",
    "nostalgic": "dusty lo-fi groove, tape-warbled keys, soft vinyl texture, mellow bass, wistful and warm",
    "dark": "dark deep house, round sub bass, soft four-on-the-floor kick, brooding analog pad, late-night, hypnotic",
    "techno": "driving techno, locked four-on-the-floor kick, rolling bassline, arpeggiated analog sequencer, hypnotic",
    "playful": "bright melodic house, bouncy plucked synth hook, round bass, crisp claps, light percussion, uplifting",
    "hype": "cinematic breakbeat, punchy kick and snare, rolling distorted bass, urgent staccato strings, energetic",
}
_MOOD_ENERGY = {"calm": 0.25, "warm": 0.4, "luxury": 0.3, "nostalgic": 0.4, "dark": 0.5, "techno": 0.7, "playful": 0.6,
                "hype": 0.9}


_WORD_MOOD = {"city": "dark", "hours": "dark", "midnight": "dark", "rain": "dark", "shadow": "dark", "neon": "techno",
              "machine": "techno", "signal": "techno", "sea": "calm", "ocean": "calm", "salt": "calm", "silver": "calm",
              "mist": "calm", "tide": "calm", "sun": "warm", "gold": "warm", "home": "warm", "summer": "warm",
              "wild": "hype", "run": "hype", "fire": "hype", "storm": "hype", "bloom": "playful", "candy": "playful",
              "dance": "playful", "memory": "nostalgic", "yesterday": "nostalgic", "remember": "nostalgic"}


def moods_from_text(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z']+", (text or "").lower()):
        m = t if t in fr.MOOD_TAGS else (_MOOD_SYN.get(t) or _WORD_MOOD.get(t))
        if m and m not in out:
            out.append(m)
    return out


def parse_bpm(text: str) -> int | None:
    """'... 122 bpm ...' in a music prompt -> 122 (50..180) or None."""
    m = re.search(r"(\d{2,3})\s*bpm", text or "", re.I)
    return int(min(180, max(50, int(m.group(1))))) if m else None


def music_plan(request: dict, brief: dict | None = None) -> dict:
    """What MographMusicEncode asks ACE-Step for: tags (+ NO_VOX), lyrics, bpm, keyscale, seconds, seed.
    Priority: the user's music prompt, then the brief, then rules from the mood words of scene / words / look.
    ACE ends its music 1-10 s before the requested duration (FINDINGS.md), so ask for 1.7x + 2 s and cut at a downbeat."""
    req = normalize_request(request)
    brief = brief or fb.empty_brief()
    text = " ".join([req["music"], req["scene"], req["words"], brief.get("scene", "")])
    moods = list(brief.get("mood") or []) + [m for m in moods_from_text(text) if m not in (brief.get("mood") or [])]
    mood = moods[0] if moods else "dark"
    bpm = parse_bpm(req["music"]) or brief.get("bpm") or _MOOD_BPM[mood]
    bpm = int(min(180, max(50, bpm)))
    tags = req["music"] or brief.get("music") or _MOOD_TAGS[mood]
    if not re.search(r"\bno vocals\b", tags, re.I):
        tags = tags.rstrip(" ,.") + ", " + NO_VOX
    if not re.search(r"\b\d{2,3}\s*bpm\b", tags, re.I):
        tags = tags + f", {bpm} BPM"
    km = re.search(r"\b([A-G][b#]?)\s*(major|minor)\b", req["music"])
    key = f"{km.group(1)} {km.group(2)}" if km else _MOOD_KEY[mood]
    seconds = min(120.0, round(req["length"] * 1.7 + 2.0, 1))
    return {"tags": tags, "lyrics": "[Instrumental]", "bpm": bpm, "keyscale": key, "timesignature": "4",
            "seconds": seconds, "seed": sub_seed(req["seed"], "music"), "mood": mood}


# =================================================================================================== text helpers
_SUBJECTS = ("person", "figure", "man", "woman", "girl", "boy", "child", "dog", "cat", "horse", "bird", "dancer", "runner",
             "rider", "couple", "people", "silhouette", "statue", "lighthouse", "boat", "ship", "car", "swimmer", "skater")
_BEHIND_WORDS = ("mist", "fog", "smoke", "glass", "silk", "haze", "steam", "curtain")          # sheer: a grey, see-through matte
_SHEER_WORDS = _BEHIND_WORDS + ("veil", "lace", "water", "spray")
_OPAQUE_WORDS = ("flower", "wheat", "reed", "grass", "lantern", "rock", "fold", "leaf", "leaves", "tree", "crowd") + _SUBJECTS   # solid matte
_SHEER_RE = re.compile(r"\b(?:" + "|".join(_SHEER_WORDS) + r")\w*")                  # prefix: misty, foggy, glassy, waterfall
_OPAQUE_RE = re.compile(r"\b(?:" + "|".join(_OPAQUE_WORDS) + r")(?:s|es)?\b")
_BEHINDS = ("behind", "behind_sheer")
_TRACED_WORDS = ("contour", "landscape", "terrain", "topographic", "ridge")
_MOOD_SCENE = {
    "calm": "a quiet misty coastline at dawn, soft silver light, still water, long exposure",
    "hype": "a dense neon city street at night, light trails, wet asphalt, strong reflections",
    "warm": "a sunlit room at golden hour, warm dust floating in the light, wooden textures",
    "nostalgic": "an empty street at dusk lit like old film, faded colours, soft halation, fine grain",
    "techno": "a dark industrial hall with laser haze, metal and glass, cold cyan light",
    "luxury": "a dark marble hall with soft gold light, polished reflections, slow drifting dust",
    "playful": "a bright pastel studio set, soft shadows, floating shapes, clean daylight",
    "dark": "a rainy night street, deep shadows, pools of neon light on wet ground",
}
_STILL_STYLE = ""            # was the fixed "cinematic 35mm film still, shallow depth of field" on 100 % of the stills (diversity audit 2026-10-29): the style is now drawn per piece (piece_system.photo_style)
_STILL_NEG = ("nsfw, worst quality, bad anatomy, jpeg artifact, text, letters, words, watermark, logo, caption, signature, signage, signs, typography, lettering, "
              "blurry, deformed, oversaturated, cartoon")
_FRAME_WORDS = {"9:16": "tall vertical frame", "16:9": "wide horizontal frame"}
_MOTION = {
    "calm": ["the air drifts slowly and the light shifts, very slow gentle push in",
             "everything breathes softly, the camera glides slowly sideways",
             "light slides slowly across the scene, the camera drifts forward",
             "soft movement settles, a slow pull back, calm"],
    "mid": ["things move steadily through the scene, the camera drifts slowly",
            "light flickers and slides across surfaces, the camera pushes in slowly",
            "movement picks up with the pulse, the camera glides along",
            "the scene keeps flowing, one continuous movement, camera drifts on"],
    "kinetic": ["fast movement everywhere, light streaks past, the camera pushes in quickly",
                "the scene surges with the beat, the camera whips sideways",
                "bursts of light and motion, the camera orbits fast",
                "relentless forward motion, the camera drives ahead"],
}
# one shot = one continuous generation = ONE camera move (shots "cut" / "single"): never a push in AND a pull back
_SWEEP = {
    "calm": ["one slow continuous camera sweep from left to right, the air drifts softly",
             "one slow continuous camera glide forward through the scene, everything breathes",
             "one slow continuous camera sweep from right to left, light shifts gently",
             "one slow continuous camera rise upward, soft movement in the air"],
    "mid": ["one steady continuous camera sweep from left to right, things move through the scene",
            "one steady continuous camera push forward, light slides across surfaces",
            "one steady continuous camera sweep from right to left, the scene keeps flowing",
            "one steady continuous camera pan upward, movement picks up with the pulse"],
    "kinetic": ["one fast continuous camera sweep from left to right, light streaks past",
                "one fast continuous camera drive forward, relentless motion",
                "one fast continuous camera sweep from right to left, bursts of light and motion",
                "one fast continuous camera orbit around the scene, the scene surges"],
}
SHOT_MODES = ("cut", "chain", "single")
_STILL_ANGLES = ["", "a different wide view of the same place", "a closer view of the same place from a new angle",
                 "a low angle view of the same place", "a high angle view of the same place", "a close detail of the same place"]
_LETTER_PHRASE = {
    "behind_sheer": "the subject stands perfectly still while the air moves around it",
    "built_in": "glowing letters built into the scene as part of the world",
    "behind": "the subject stands perfectly still while the air moves around it",
    "traced": "long flowing strands draw out and trace the lines",
}


def _letters(s: str) -> int:
    return max(1, len(re.sub(r"[\W_]+", "", s)))


def _first_words(s: str, n: int) -> str:
    w = s.split()
    return " ".join(w[:n]).rstrip(" ,;")


def choose_technique(brief: dict, scene: str) -> str:
    if brief.get("technique") in TECHNIQUES:
        return brief["technique"]
    low = scene.lower()
    if _SHEER_RE.search(low):                         # mist / glass / silk ...: grey see-through matte (solid "behind" if the recipe is missing)
        return "behind_sheer" if (RECIPES / "behind_sheer.json").exists() else "behind"
    if any(w in low for w in _TRACED_WORDS):
        return "traced"
    if _OPAQUE_RE.search(low):                        # flowers, wheat, rocks, people ...: a SOLID matte
        return "behind"
    return "built_in"


def merge_lines(lines: list[str], max_lines: int) -> list[str]:
    """merge the shortest neighbouring pair until at most ``max_lines`` remain (the user's words, in order)."""
    lines = list(lines)
    while len(lines) > max(1, max_lines):
        j = min(range(len(lines) - 1), key=lambda k: _letters(lines[k]) + _letters(lines[k + 1]))
        lines[j:j + 2] = [lines[j] + " " + lines[j + 1]]
    return lines


def split_for_bars(lines: list[str], bars: int, min_bars: int = 3) -> list[str]:
    """few lines for many bars would hold each word for ages: split the longest multi-word line in two (balanced, user's
    words in order) until every line has about ``min_bars`` bars or no line can be split."""
    from mograph.brief import _balanced_split
    lines = list(lines)
    while len(lines) < 6 and bars // max(len(lines), 1) > min_bars:
        j = max((k for k in range(len(lines)) if len(lines[k].split()) >= 2), key=lambda k: _letters(lines[k]), default=None)
        if j is None:
            break
        ws = lines[j].split()
        parts = _balanced_split(ws, max_words=int(math.ceil(len(ws) / 2.0)))
        if len(parts) != 2:
            parts = [ws[: len(ws) // 2], ws[len(ws) // 2:]]
        lines[j:j + 1] = [" ".join(p) for p in parts]
    return lines


# =================================================================================================== palette / fonts / look
def _hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


_MOOD_PALETTE = {"calm": "type_soft", "warm": "film_soft", "luxury": "film_type_soft", "nostalgic": "paper_assembly",
                 "techno": "neon_loop", "playful": "soft_loop", "dark": "ambient_glow", "hype": "type_hectic"}


def _pad_accents(pal: list[str], n: int = 3) -> list[str]:
    """word modules read accents[0..2]: repeat the given accents (or the light colour) up to three."""
    pal = list(pal)
    acc = pal[2:] or [pal[1]]
    while len(pal) < 2 + n:
        pal.append(acc[(len(pal) - 2) % len(acc)])
    return pal


def palette_for(mood: str, seed: int, hint: list | None) -> list[str]:
    """[dark, light, accent, accent, accent(, ...)] as hex."""
    if hint and len(hint) >= 3:
        pal = order_palette(list(hint))
        if len(pal) >= 3:
            return _pad_accents(pal)
    bank = PALETTES[_MOOD_PALETTE.get(mood, "type_soft")]
    return _pad_accents(bank[sub_seed(seed, "palette") % len(bank)])


def pick_family(look: str, energy: float, seed: int) -> tuple[str, list[str]]:
    """-> (family, module pool). ``look`` is 'auto', a family or a single module name."""
    if look in LOOKS:
        return look, LOOKS[look]
    for fam, mods in LOOKS.items():
        if look in mods:
            return fam, [look]
    fam = _pick(seed, "look", _AUTO_FAMILIES[tier_of(energy)])
    return fam, LOOKS[fam]


def pick_modules(look: str, energy: float, seed: int) -> list[str]:
    fam, pool = pick_family(look, energy, seed)
    a = sub_seed(seed, "mod_a") % len(pool)
    if len(pool) == 1:
        return [pool[0], pool[0]]
    b = (a + 1 + sub_seed(seed, "mod_b") % (len(pool) - 1)) % len(pool)
    return [pool[a], pool[b]]


def _liked_display() -> set[str]:
    """registry names of the liked fonts the planner may use as a display face."""
    return {e["family"] for e in fl.LIKED
            if e["family"] in fr.REGISTRY and "display" in fr.REGISTRY[e["family"]].get("role", ())}


def pick_fonts(moods: list[str], energy: float, seed: int, n: int) -> list[str]:
    """2-3 LIKED display faces (file names) drawn only from mograph/fonts/liked.json. Each differs in role from the
    others (tall / heavy / fine / mono / ornamental ...), so consecutive lines (fonts cycle in order) never share a role."""
    disp = _liked_display()
    mo = moods or ["hype"]
    first, _ = fr.pick_fonts(mo, energy, seed, "f1", allowed=disp)
    out, used = [first], {fl.role_of(first)}
    want = 3 if n >= 5 else 2
    foils = [f for f in fr.foils(first) if f in disp and fl.role_of(f) not in used]
    if foils:
        f = foils[sub_seed(seed, "foil") % len(foils)]
        out.append(f)
        used.add(fl.role_of(f))
    for k in range(2, 12):
        if len(out) >= want:
            break
        pool = {f for f in disp if fl.role_of(f) not in used}
        if not pool:
            break
        cand, _ = fr.pick_fonts(mo, energy, seed, f"f{k}", allowed=pool)
        out.append(cand)
        used.add(fl.role_of(cand))
    return [fr.REGISTRY[f]["file"] for f in out[:3]]


LINE_SCALE_ALT = 0.78       # placement.bleed of a line whose font has no weight / fill contrast with the previous line


def line_scale(prev_file: str | None, cur_file: str, prev_bleed: float = 1.0) -> float:
    """Per-line contrast (user rule): consecutive lines differ in ROLE (pick_fonts guarantees it) AND in at least one of
    scale / weight / fill-vs-outline. Weight and fill follow from the role pair; when neither differs the scale does:
    -> the placement bleed (1.0 = full frame) for this line."""
    if prev_file is None:
        return 1.0
    if fl.contrast_axes(fl.role_of(prev_file), fl.role_of(cur_file)):
        return 1.0
    return 1.0 if prev_bleed < 1.0 else LINE_SCALE_ALT


def _font_axes(font_file: str, tier: str, cut: float) -> dict | None:
    name = next((n for n, f in fr.REGISTRY.items() if f["file"] == font_file), None)
    vary = (fr.REGISTRY.get(name) or {}).get("vary") or {}
    if "wght" not in vary:
        return None
    lo, hi = vary["wght"]
    lo2 = int(round(lo + 0.3 * (hi - lo)))
    if tier == "calm":
        return {"wght": {"keys": [[0.0, lo2], [_r(cut, 2), int(hi)]], "ease": "smooth"}}
    return {"wght": [lo2, int(hi), "beat"]}


# =================================================================================================== timing
def _spec():
    from tools.fusion import spec
    return spec


def transition_names() -> list[str]:
    """every registered transition module except the plain cut (a new transition is picked up as soon as it is registered)."""
    _spec()                                           # imports mograph.modules (registers the transitions)
    from mograph.modules.core import list_modules
    return [n for n in list_modules("transition") if n != "cut"]


def make_grid(music: dict, cut: float):
    from tools.fusion.beats import Beats
    return Beats(music.get("bpm"), music.get("beats"), 4, 0.0, float(cut) + 4.0, music.get("downbeats"))


def _word_io(tier: str, i: int, style: str) -> tuple[str, str]:
    if tier == "calm":
        return style, style
    if tier == "mid":
        return (style if i == 0 else "flicker"), "flicker"                 # the piece's seeded style opens it (was always `lines`)
    return "flicker", "flicker"


def _lead_beats(tier: str) -> int:
    return {"calm": 4, "mid": 1, "kinetic": 1}[tier]


def overlap_gap(iv: float, carries: bool) -> float:
    """transition gap (s) of the slow-tempo overlap mode (spec.overlap_on): the compressed floor between two linked lines, half a beat at a cut."""
    return _spec().OVERLAP_FLOORS["gap"] if carries else 0.5 * iv


def overlap_t0s(spans, ios, iv: float) -> list[float]:
    """cumulative t0 per line for the overlap mode: spec.plan_timing reads the transition gap as next.t0 - this.t1, so the planner states it
    (overlap_gap) instead of the calm 4-beat default. ios: [(form_in, form_out)] per line."""
    NM = _spec().NO_MATERIAL
    t, out = 0.0, []
    for j, sp_ in enumerate(spans):
        out.append(round(t, 4))
        carries = j + 1 < len(spans) and ios[j][1] not in NM and ios[j + 1][0] not in NM
        t += float(sp_) + overlap_gap(iv, carries)
    return out


OV_END_RESERVE = 0.31      # overlap mode: the last exit ends 0.31 s before the cut (the planner's dead-tail rule: last_end <= cut - 0.3)
OV_FORM_PREF, OV_EXIT_PREF = 0.8, 0.7


def _schedule_overlap(lines, tier, grid, cut, wio, max_lines):
    """Slow tempo (spec.overlap_on): the user's lines are kept (merged only when even the compressed exit / transition / form cannot give a
    1.2 s hold, then the fewest). Hits land on the first beat >= 0.8 s, then on downbeats; each line's cycle (form + hold + exit + gap)
    is split in closed form by spec.overlap_split. -> same tuple as schedule_words: ([(span, form_s, exit_s, hit_on)], positions, lines)."""
    sp = _spec()
    F, iv, NM = sp.OVERLAP_FLOORS, float(grid.iv), sp.NO_MATERIAL
    end_target = float(cut) - OV_END_RESERVE
    first = float(grid.snap(0.8, mode="ceil"))
    P = [first] + [float(d) for d in grid.d if first + 0.5 * iv < float(d) < end_target]

    def hold(t):
        return max(sp.HOLD_FLOOR, sp.min_hold({"text": t}))

    def f_floor(j):
        return F["form_nomat"] if wio(j)[0] in NM else F["form"]

    def gap(j):
        return overlap_gap(iv, wio(j)[1] not in NM and wio(j + 1)[0] not in NM)

    def steps(ls):
        """minimal position index per line (None: the lines do not fit)."""
        idx = [0]
        for j, t in enumerate(ls[:-1]):
            need = hold(t) + F["exit"] + gap(j) + f_floor(j + 1)
            m = idx[-1] + 1
            while m < len(P) and P[m] - P[idx[-1]] < need - 1e-6:
                m += 1
            idx.append(m)
        last = len(ls) - 1
        if idx[-1] >= len(P) or P[idx[-1]] + hold(ls[last]) + F["exit"] > end_target + 1e-6:
            return None
        return idx

    base = merge_lines(split_for_bars(list(lines), len(P)), min(len(lines), len(P), int(max_lines or 10 ** 6)))
    ls = list(base)
    while steps(ls) is None and len(ls) > 1:            # a readable hold is never given up: merge the fewest lines until they fit
        ls = merge_lines(ls, len(ls) - 1)
    idx = steps(ls) or [0] * len(ls)
    L = len(ls)
    max_last = max([m for m in range(len(P)) if P[m] + hold(ls[-1]) + F["exit"] <= end_target + 1e-6] or [idx[-1]])
    spare = max(max_last - idx[-1], 0)                  # extra downbeats: longer holds on the longer lines, by letter count
    if L > 1 and spare:
        w = [_letters(t) ** 0.8 for t in ls[:-1]]
        raw = [spare * x / sum(w) for x in w]
        add = [int(math.floor(r)) for r in raw]
        for j in sorted(range(L - 1), key=lambda j: -(raw[j] - math.floor(raw[j])))[: spare - sum(add)]:
            add[j] += 1
        shift = 0
        for j in range(L - 1):
            shift += add[j]
            idx[j + 1] += shift
    hits = [P[min(m, len(P) - 1)] for m in idx]
    out, form_prev = [], hits[0]                        # the first line starts at 0
    for j in range(L):
        if j < L - 1:
            sp_ = sp.overlap_split(hits[j + 1] - hits[j], hold(ls[j]), OV_FORM_PREF, OV_EXIT_PREF, gap(j), f_floor(j + 1), g_fix=gap(j))
            x, h, f_next = (sp_["x"], sp_["hold"], sp_["f"]) if sp_ else (F["exit"], hold(ls[j]), f_floor(j + 1))
        else:
            x = OV_EXIT_PREF if end_target - hits[j] - OV_EXIT_PREF >= hold(ls[j]) else F["exit"]
            h, f_next = max(end_target - hits[j] - x, hold(ls[j])), 0.0
        out.append((round(form_prev + h + x, 4), round(form_prev, 4), round(x, 4), "beat"))
        form_prev = f_next
    return out, len(P), ls


def split_lines(lines: list[str], n: int) -> list[str]:
    """split the longest multi-word line in two (balanced, the words in order) until there are n lines or none can be split."""
    from mograph.brief import _balanced_split
    lines = list(lines)
    while len(lines) < n:
        j = max((k for k in range(len(lines)) if len(lines[k].split()) >= 2), key=lambda k: _letters(lines[k]), default=None)
        if j is None:
            break
        ws = lines[j].split()
        parts = _balanced_split(ws, max_words=int(math.ceil(len(ws) / 2.0)))
        if len(parts) != 2:
            parts = [ws[: len(ws) // 2], ws[len(ws) // 2:]]
        lines[j:j + 1] = [" ".join(p) for p in parts]
    return lines


def fit_lockups(lines: list[str], n: int) -> tuple[list[str], str | None]:
    """ONE lockup per shot: too many lines are merged (the shortest neighbouring pair first), too few are split (the longest multi-word line, balanced); the
    words and their order never change. -> (lines, a note describing the fix or None). Fewer lines than shots after the split (single words only) stay
    short: the last lockup then runs to the end of the piece."""
    before = list(lines)
    out = list(lines)
    if len(out) > n:
        out = merge_lines(out, n)
    elif len(out) < n:
        out = split_lines(out, n)
    if out == before and len(out) >= n:
        return out, None
    verb = "merged" if len(before) > len(out) else "split"
    note = f"lockups: {len(before)} lines {verb} to {len(out)} for {n} shots (one lockup per shot): {' / '.join(before)} -> {' / '.join(out)}"
    if len(out) < n:
        note += f"; only {len(out)} lockups can be made of single words, the last one runs to the end"
    return out, note


def schedule_shot_words(lines: list[str], starts: list[float], tier: str, grid, cut: float, io, style: str, bars_adjust: int = 0):
    """word k starts with shot k: t0 = the shot start (a cut on a downbeat), hit on a downbeat when the shot has room, else on a beat ~0.9 s later, the exit
    ends before the next shot (a flicker / wipe boundary is a cut, a carrying one keeps a transition gap; engine gap = t0(k+1) - t1(k)); the last word ends
    before the cut (OV_END_RESERVE). The engine's own plan_timing is the
    simulator (as in schedule_words): the span is searched so that end + gap lands on the next shot start with a readable hold.
    -> [(t0, span, form_s, exit_s, hit_on)] one per line (`starts` has at least len(lines) entries; the last word runs to `cut`)."""
    sp = _spec()
    L = len(lines)
    iv = float(grid.iv)
    bar_len = iv * grid.bar
    end_last = float(cut) - OV_END_RESERVE + iv * min(bars_adjust, 0)      # a dry-run retry pulls the last exit in by a beat per attempt (the engine's real end runs up to ~1 s past this simulator's)

    def wio(j):
        return tuple(io[j]) if io and j < len(io) else _word_io(tier, j, style)

    def wd(j, t0, span, form_s, exit_s, hit_on):
        fi, fo = wio(j)
        return {"t0": t0, "t1": t0 + span, "form_s": form_s, "exit_s": exit_s, "hit_on": hit_on, "form_in": fi, "form_out": fo}

    words, out = [], []
    for i in range(L):
        s0 = float(starts[i])
        nxt = float(starts[i + 1]) if i < L - 1 else None
        shot_len = (nxt if nxt is not None else float(cut)) - s0
        hold_need = max(sp.min_hold({"text": lines[i]}), 1.0)
        if bars_adjust < 0 and nxt is None:
            hold_need *= 0.9                    # last resort of the dry-run retries: the last word's hold gives up a tenth rather than run past the cut
        look = None
        best = None
        for need in (hold_need,):
            best = None
            # hit on a downbeat when the shot has room for it, else on a beat; form / exit options, longest first: the first option whose best span reads (hold)
            # and lands (next start / before the cut) wins, else the least bad. The spans are offset by 13 ms so no candidate sits exactly between two beats
            # (a tie snaps either way after the 3-decimal rounding of t1).
            for hit_on in (("bar", "beat") if bars_adjust >= 0 else ("beat",)):                 # a dry-run retry drops the bar-landed hit (it can wait a whole bar)
                look = wd(i + 1, nxt, 2.0, 1.0, 0.5, hit_on) if nxt is not None else None
                for form_s, exit_s in ((0.9, 0.9 if nxt is None else 0.5), (0.75, 0.5), (0.6, 0.5), (0.45, 0.5), (0.35, 0.4)):
                    n_c = int(max(shot_len, 1.0) / (iv / 4.0)) + 5
                    for q in range(n_c):
                        span = 1.013 + q * iv / 4.0
                        cand = wd(i, s0, span, form_s, exit_s, hit_on)
                        tm = sp.plan_timing(words + [cand] + ([look] if look else []), {"tier": tier}, grid)[i]
                        ok_hold = (tm["rel"] - tm["hit"]) >= need - 1e-6
                        if nxt is not None:
                            nx = tm["end"] + tm["gap"]
                            err = abs(nx - nxt)
                            # never start the next word before its shot (a word must not form over the previous shot's last frames), then the nearest start
                            key = (not ok_hold, nx < nxt - 0.05, round(err, 4), -span)
                        else:
                            key = (not ok_hold, tm["end"] > end_last + 1e-6, round(max(0.0, tm["end"] - end_last), 3), -span)      # none fits: the least overshoot, never the longest span
                        if best is None or key < best[0]:
                            best = (key, span, form_s, exit_s, hit_on)
                    if not best[0][0] and not best[0][1] and (nxt is None or best[0][2] <= 0.5 * iv + 1e-6):
                        break
                if not best[0][0] and not best[0][1] and (nxt is None or best[0][2] <= 0.5 * iv + 1e-6):
                    break
            if not best[0][0] and not best[0][1] and (nxt is None or best[0][2] <= 0.5 * iv + 1e-6):
                break
        _, span, form_s, exit_s, hit_on = best
        words.append(wd(i, s0, span, form_s, exit_s, hit_on))
        out.append((s0, span, form_s, exit_s, hit_on))
    return out


def schedule_words(lines: list[str], tier: str, grid, cut: float, bar_len: float, style: str, bars_adjust: int = 0, io=None, max_lines=None):
    """Spans + form times so every hit sits on a downbeat. -> (list of (span, form_s, exit_s, hit_on), bars, lines used).
    io: optional [(form_in, form_out)] per line (a batch assignment); lines past its end use the tier rule (_word_io).
    max_lines: never more lines than this (a batch assignment has that many; the plan may only merge, never split further)."""
    sp = _spec()

    def wio(j):
        return tuple(io[j]) if io and j < len(io) else _word_io(tier, j, style)
    if sp.overlap_on({"tier": tier}, grid):
        return _schedule_overlap(lines, tier, grid, cut, wio, max_lines)
    D = [float(d) for d in grid.d]
    iv = float(grid.iv)
    end_target = float(cut) - RESERVE_S
    slow = bar_len > 3.0                    # bar > 3 s (< 80 bpm): half-bar windows, hits on the beat grid (every 2nd beat)
    U = sorted(set(D) | {d + 2 * iv for d in D}) if slow else D
    unit = bar_len / 2.0 if slow else bar_len
    hit_on = "beat" if slow else "bar"
    H0 = U[1] if len(U) > 1 else unit
    B = int(math.floor((end_target - H0) / unit + 1e-9)) + bars_adjust
    lines = split_for_bars(lines, B)
    pace = sp.PACE[sp.pace_of({"tier": tier}, grid)]

    def exit_of(j, n):
        return (1.4 if tier == "calm" else 0.9) if (tier == "calm" or j == n - 1) else 0.45

    def need_units(ls, relax=False):
        """units each word but the last needs (a readable hold + its exit + the transition gap + the next word's lead-in), and the
        seconds the last word needs (readable hold + exit)."""
        n, out = len(ls), []
        hold = (lambda t: 1.0) if relax else (lambda t: sp.min_hold({"text": t}))      # relax: 1 s flat (keeps two lines in a short piece)
        for j, t in enumerate(ls[:-1]):
            need = hold(t) + exit_of(j, n)
            carries = wio(j)[1] not in sp.NO_MATERIAL and wio(j + 1)[0] not in sp.NO_MATERIAL
            gap = min(max(pace[0] * iv, pace[1]), pace[2]) if carries else 0.5 * iv
            need += gap + _lead_beats(tier) * iv + 0.5 * iv
            out.append(max(1, int(math.ceil(need / unit - 1e-9))))
        return out, hold(ls[-1]) + exit_of(n - 1, n)

    avail = (end_target - H0) + bars_adjust * unit
    base = merge_lines(lines, min(len(lines), max(1, B), int(max_lines or 10 ** 6)))

    def fit(relax, floor_n):
        ls = list(base)
        nd, lt = need_units(ls, relax)
        while sum(nd) * unit + lt > avail + 1e-9 and len(ls) > floor_n:      # fewer, longer-held lines until every one can be read
            ls = merge_lines(ls, len(ls) - 1)
            nd, lt = need_units(ls, relax)
        return ls, nd, lt

    lines, need, last_t = fit(False, 1)
    if len(lines) < min(2, len(base)):                  # one long hold is all that reads: two lines with a 1 s hold are better
        l2, n2, t2 = fit(True, 2)
        if len(l2) >= 2 and sum(n2) * unit + t2 <= avail + 1e-9:
            lines, need, last_t = l2, n2, t2
    L = len(lines)
    need = need + [1]
    B = max(B, sum(need))
    # bars (units) per word: what it needs + a share of the spare bars by letter count (largest remainder)
    w = [_letters(s) ** 0.8 for s in lines]
    extra = B - sum(need)
    raw = [extra * x / sum(w) for x in w]
    k = [need[j] + int(math.floor(r)) for j, r in enumerate(raw)]
    order = sorted(range(L), key=lambda j: -(raw[j] - math.floor(raw[j])))
    for j in order[: B - sum(k)]:
        k[j] += 1
    hits, c = [], 1
    for j in range(L):
        hits.append(U[min(c, len(U) - 1)])
        c += k[j]
    lead = _lead_beats(tier) * iv
    exits = [exit_of(j, L) for j in range(L)]
    out = []
    words: list[dict] = []
    S = 0.0

    def wd(j, span, form_s):
        fi, fo = wio(j)
        return {"t0": 0.0, "t1": span, "form_s": form_s, "exit_s": exits[j], "hit_on": hit_on, "form_in": fi, "form_out": fo}

    for i in range(L):
        form_s = max(hits[i] - S, 0.2)
        if i < L - 1:
            target = grid.snap(hits[i + 1] - lead)
            best = None
            n_c = int(((k[i] + 3) * unit - 1.0) / (iv / 4.0)) + 1
            for q in range(n_c):
                span = 1.0 + q * iv / 4.0
                tm = sp.plan_timing(words + [wd(i, span, form_s), wd(i + 1, 2.0, 1.0)], {"tier": tier}, grid)
                nxt = tm[i]["end"] + tm[i]["gap"]
                err = abs(nxt - target)
                key = (err > 1e-6 and nxt > target + 1e-6, err, span)       # never land past the target if avoidable
                if best is None or key < best[0]:
                    best = (key, span, nxt)
            span, S_next = best[1], best[2]
        else:
            span, best = 1.0, None
            n_c = int(((k[i] + 3) * unit - 1.0) / (iv / 8.0)) + 1
            for q in range(n_c):
                s2 = 1.0 + q * iv / 8.0
                tm = sp.plan_timing(words + [wd(i, s2, form_s)], {"tier": tier}, grid)
                if tm[i]["end"] <= end_target + 1e-6:
                    span = s2
            S_next = None
        words.append(wd(i, span, form_s))
        out.append((span, form_s, exits[i], hit_on))
        if S_next is not None:
            S = S_next
    return out, B, lines


# =================================================================================================== recipe
def load_recipe(technique: str) -> dict:
    return json.loads((RECIPES / f"{technique}.json").read_text(encoding="utf-8"))


def _scale_timelines(obj, f: float):
    if isinstance(obj, dict):
        o = {}
        for k, v in obj.items():
            if k == "timeline" and isinstance(v, list):
                o[k] = [[_r(float(p[0]) * f, 3)] + list(p[1:]) for p in v]
            else:
                o[k] = _scale_timelines(v, f)
        return o
    if isinstance(obj, list):
        return [_scale_timelines(v, f) for v in obj]
    return obj


def _snap_timelines(obj, grid):
    """every `timeline` key time -> the nearest beat (keys at t <= 0 stay), strictly increasing: pulses land on beats."""
    if isinstance(obj, dict):
        o = {}
        for k, v in obj.items():
            if k == "timeline" and isinstance(v, list):
                keys, prev = [], -1.0
                for p in v:
                    t = float(p[0])
                    if t > 1e-9:
                        t = grid.snap(t, minimum=max(prev, 0.0) + 0.25 * grid.iv)
                    t = _r(max(t, prev + 0.001), 3)
                    keys.append([t] + list(p[1:]))
                    prev = t
                o[k] = keys
            else:
                o[k] = _snap_timelines(v, grid)
        return o
    if isinstance(obj, list):
        return [_snap_timelines(v, grid) for v in obj]
    return obj


def _subst_colors(obj, cols: dict):
    if isinstance(obj, str) and obj.startswith("@") and obj[1:] in cols:
        return list(cols[obj[1:]])
    if isinstance(obj, dict):
        return {k: _subst_colors(v, cols) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst_colors(v, cols) for v in obj]
    return obj


def _fit_sizes(obj, size):
    """ops that carry a literal compose size (upscale_matte 'size') get the piece's compose size, whatever the format."""
    if isinstance(obj, dict):
        o = {k: (list(size) if k == "size" and isinstance(v, list) and len(v) == 2 else _fit_sizes(v, size)) for k, v in obj.items()}
        return o
    if isinstance(obj, list):
        return [_fit_sizes(v, size) for v in obj]
    return obj


def apply_recipe(recipe: dict, cut: float, palette: list[str], size=None, grid=None) -> dict:
    """compose section of the recipe with timeline keys rescaled to the cut, '@accent0' style colours filled in and literal
    sizes set to the compose ``size`` (format aware). With a beat ``grid`` every timeline key is snapped to a beat."""
    cols = {"bg": palette[0], "fg": palette[1]}
    for i, a in enumerate(palette[2:] or [palette[1]]):
        cols[f"accent{i}"] = a
    cols.setdefault("accent1", cols["accent0"])
    rgb = {k: [_r(c / 255.0, 3) for c in _hex_rgb(v)] for k, v in cols.items()}
    comp = copy.deepcopy(recipe["compose"])
    comp = _scale_timelines(comp, float(cut) / float(recipe.get("ref_cut") or cut))
    if grid is not None and grid:
        comp = _snap_timelines(comp, grid)
    if size:
        comp = _fit_sizes(comp, size)
    return _subst_colors(comp, rgb)


# =================================================================================================== layout / shots
def snap8(n: int) -> int:
    """smallest 8k+1 >= n (>= 9)."""
    n = int(math.ceil(n))
    return max(9, 8 * int(math.ceil((n - 1) / 8.0)) + 1)


def shot_frames(nframes: int, xf: int = XFADE, grid=None, fps: int = FPS) -> list[int]:
    """frames per shot: N = ceil((nframes - xf) / (121 - xf)) shots of equal 8k+1 length, the last trimmed to what is left.
    xf == 0 with a beat grid (hard cuts): the LTX length (8k+1 >= span) of every shot of cut_spans; the shot is cut to its span."""
    if xf == 0 and grid is not None:
        return [snap8(sp) for sp in cut_spans(nframes, grid, fps)]
    n = fb.shot_count(nframes, SHOT_FRAMES, xf, MAX_SHOTS)
    raw = (nframes - xf) / n + xf
    f = snap8(raw)
    if f > SHOT_FRAMES:
        f = min(LONG_SHOT, f)           # more than 6 x 121 frames needed: longer shots, still 6 slots
    out = [f] * n
    last_start = (n - 1) * (f - xf)
    out[-1] = min(f, snap8(nframes - last_start))
    return out


MIN_SPAN = 24              # frames: a hard-cut shot is on screen at least 1 s


def story_shot_count(n, nframes: int):
    """the storyboard's shot count (brief.story.shots, STORYBOARD v12: never more shots than pictures) validated against the planner's limits: -> (n | None, note | None).
    Valid = 1..MAX_SHOTS and an even span of nframes / n between MIN_SPAN and SHOT_FRAMES (one LTX clip per shot). None = the caller keeps today's rule (note says why when n was given)."""
    if n is None:
        return None, None
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None, f"story shot count {n!r} is not a number: the planner's own shot count is used"
    if n < 1 or n > MAX_SHOTS:
        return None, f"story has {n} shots (1..{MAX_SHOTS} allowed): the planner's own shot count is used"
    per = nframes / float(n)
    if per > SHOT_FRAMES + 1e-9 or per < MIN_SPAN:
        return None, f"story has {n} shots for {nframes / FPS:g} s ({per / FPS:.1f} s each; {MIN_SPAN / FPS:g}..{SHOT_FRAMES / FPS:.2f} s allowed): the planner's own shot count is used"
    return n, None


def cut_spans(nframes: int, grid, fps: int = FPS, n: int | None = None, exact: bool = False) -> list[int]:
    """Hard cuts on downbeats: n shot spans (piece frames, sum = nframes); every cut is round(downbeat * fps), as even as the
    bars allow and never longer than SHOT_FRAMES (LONG_SHOT when 6 slots cannot cover the piece). Without a downbeat grid: even cuts."""
    n = int(n or fb.shot_count(nframes, SHOT_FRAMES, XFADE, MAX_SHOTS))
    if n <= 1:
        return [int(nframes)]
    downs = sorted({int(round(float(d) * fps)) for d in (grid.d if grid else [])})
    downs = [c for c in downs if MIN_SPAN <= c <= nframes - MIN_SPAN]
    tries = [(SHOT_FRAMES, n)] if exact else [(SHOT_FRAMES, m) for m in range(n, MAX_SHOTS + 1)] + [(LONG_SHOT, n)]     # exact (the storyboard's shot count): exactly n, else even cuts
    for cap, n_try in tries:    # more shots before longer ones
        cuts, prev, ok = [], 0, True
        for k in range(1, n_try):
            target = nframes * k / n_try
            left = n_try - k
            cand = [c for c in downs if prev + MIN_SPAN <= c <= prev + cap and MIN_SPAN * left <= nframes - c <= cap * left]
            if not cand:
                ok = False
                break
            prev = min(cand, key=lambda c: (abs(c - target), c))
            cuts.append(prev)
        if ok and MIN_SPAN <= nframes - prev <= cap:
            b = [0] + cuts + [int(nframes)]
            return [b[i + 1] - b[i] for i in range(n_try)]
    b = [int(round(nframes * k / n)) for k in range(n + 1)]              # no usable downbeats: even cuts
    return [b[i + 1] - b[i] for i in range(n)]


def layout(piece: dict, fps: int = FPS) -> dict:
    """{sid: {start, frames, xf}} in piece frames; xf = crossfade frames INTO the shot (0 for the first), as Piece.layout."""
    cf = piece.get("compose", {}).get("crossfade", 0.5)
    shots = piece["shots"]
    cfl = cf if isinstance(cf, list) else [cf] * max(len(shots) - 1, 0)
    out, start = {}, 0
    for k, s in enumerate(shots):
        n = int(s.get("span") or s["frames"])
        c = int(round(cfl[k - 1] * fps)) if k else 0
        if k:
            start -= c
        out[s["id"]] = {"start": start, "frames": n, "xf": c}
        start += n
    return out


def orient(size, fmt: str = "9:16") -> list[int]:
    """portrait (w, h) -> the same size in the requested format (16:9 swaps the sides)."""
    w, h = int(size[0]), int(size[1])
    lo, hi = min(w, h), max(w, h)
    return [hi, lo] if fmt == "16:9" else [lo, hi]


PREVIEW_I2V = [384, 672]                 # plain i2v single-stage size of the preview tier (portrait; orient() for 16:9)


def sizes_for(quality: str, fmt: str = "9:16") -> dict:
    """control = LTX stage-1 + type control pass; shot = LTX output; type = compose layers / still; compose = output.
    9:16 / 16:9 only swaps the sides. final: stage 1 576x1024, stage 2 = x2 latent upscale 1152x2048, type 1088x1920, compose
    1080x1920 (shot scaled x0.9375). The IC-LoRA control graphs (ref downscale 2) need a stage-1 side that is a multiple of 64:
    544x960 -> 1088x1920 fails on the box (latent grid 17 not divisible by 2), so 1152x2048 stays for IC shots.
    draft: ONE stage at 576x1024 (no stage 2, no x2 decode; ltx_ic.plan_size drops two_stage), type 576x1024 = shot size
    (exact alignment), compose 540x960 (x0.9375). Plain i2v shots may use 544x960 draft / 1088x1920 final (stage 1 544x960).
    preview: the small fast iteration tier, ONE stage like draft. IC control / motiontrack / type layers 384x704 (multiples of 64), plain
    i2v 384x672 (multiples of 32, the closest to 9:16 near 360 wide, see PREVIEW_I2V), compose 360x640: every layer is cover-fitted
    (compose.fit_cover: x0.9375 then a centred 10 px crop top and bottom for 384x704; x0.9524 + 3 px side crop for 384x672), identically
    for footage, type and matte, so control-video alignment stays exact. Nothing is upscaled anywhere."""
    if quality == "preview":
        d = {"control": [384, 704], "shot": [384, 704], "type": [384, 704], "compose": [360, 640]}
    elif quality == "draft":
        d = {"control": [576, 1024], "shot": [576, 1024], "type": [576, 1024], "compose": [540, 960]}
    else:
        d = {"control": [576, 1024], "shot": [1152, 2048], "type": [1088, 1920], "compose": [1080, 1920]}
    return {k: orient(v, fmt) for k, v in d.items()}


def format_of(piece: dict) -> str:
    """'9:16' | '16:9' of a piece (its `format`, else from sizes.compose)."""
    f = piece.get("format")
    if f in FORMATS:
        return f
    sz = (piece.get("sizes") or {}).get("compose")
    return "16:9" if sz and int(sz[0]) > int(sz[1]) else "9:16"


def _slug(words: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", fb.words_lines(words).lower()).strip("-")
    return "-".join(s.split("-")[:4]) or "piece"


# =================================================================================================== the planner
def _scene_text(req: dict, brief: dict, moods: list[str], technique: str, style: str = "", concept: bool = False) -> tuple[str, str]:
    """-> (scene sentence, still prompt). `style` = the piece's photographic style (piece_system.photo_style); the scene never gets a lone figure added (a subject word is only asked for in the still)."""
    frame = _FRAME_WORDS[req.get("format", "9:16")]
    user_scene = req["scene"]
    scene = user_scene or brief.get("scene") or ""
    if not scene:
        scene = _MOOD_SCENE[moods[0] if moods else "dark"]
    still = (brief.get("still") if not user_scene else "") or f"{scene}, {style or _STILL_STYLE}".rstrip(" ,")
    if req.get("format") == "16:9" and "tall vertical frame" in still.lower():
        still = re.sub("tall vertical frame", frame, still, flags=re.I)
    if not any(w in still.lower() for w in ("vertical", "horizontal")):
        still = still.rstrip(" .,") + ", " + frame
    if technique in _BEHINDS and not concept and not any(w in still.lower() for w in _SUBJECTS):      # never for a concept's stills: its subject is named, and the vague words drew a man for a car (round-3 QC)
        still = still.rstrip(" .,") + ", one clear subject in the foreground"
    return scene, still


# 2026-10-29: no shot starts from another shot's last frame (request key `legacy_chain` = the old behaviour, tests only): every shot has its own still (fresh or an angle still of its place)


def _build(req: dict, music: dict, brief: dict, bars_adjust: int, notes: list, assignment: dict | None = None) -> dict:
    seed = req["seed"]
    cut = float(music["cut"])
    nframes = int(music.get("nframes") or math.ceil(cut * FPS - 1e-6))
    bpm = float(music["bpm"])
    bar_len = float(music.get("bar_len") or 4 * 60.0 / bpm)
    energy = energy_of(music)
    tier = tier_of(energy)
    dirn = brief.get("direction") or {}                                   # the Qwen brief's direction (pace / palette / type role / camera / shots / music)
    tier = fb.PACE_TIER.get(dirn.get("pace"), tier)
    blocks_names = [n for n in ((assignment or {}).get("blocks") or []) if n in bk.BLOCKS]
    notes = notes + list((assignment or {}).get("block_notes") or [])
    quality = req["quality"]
    fmt = req["format"]

    text_for_mood = " ".join([req["scene"], brief.get("scene", ""), req["music"], req["words"]])
    moods = list(brief.get("mood") or []) + [m for m in moods_from_text(text_for_mood) if m not in (brief.get("mood") or [])]
    for m in {"calm": ["calm", "luxury"], "mid": ["dark", "nostalgic"], "kinetic": ["hype", "dark"]}[tier]:
        if m not in moods:
            moods.append(m)
    moods = moods[:3]
    technique = choose_technique(brief, req["scene"] or brief.get("scene", ""))
    recipe = load_recipe(technique)
    sizes = sizes_for(quality, fmt)

    psys = req.get("system") if isinstance(req.get("system"), dict) and req["system"].get("module") else None      # the concept's piece system (None: a typed request, the rules' own defaults)
    eff_req = list(req.get("effects") or [])
    if eff_req:                                                           # USER-REQUESTED EFFECTS (request `effects`): forced into the piece's system; unknown names are reported, never fatal
        _known, _unk = EM.norm_effects(eff_req)
        psys = EM.apply_effects(psys, eff_req) if psys else PS.choose(seed, moods[0], bpm, effects=eff_req, length_s=cut)
        notes = list(notes) + [f"effects: forced {', '.join(_known) or 'none'}" + (f"; unknown name(s) ignored: {', '.join(_unk)}" if _unk else "")]
    tone_ = (psys or {}).get("tone") or PS.tone_of(moods[0], bpm)
    photo_ = (psys or {}).get("photo") or PS.photo_style(seed, tone_)[1]                  # the stills' photographic style, per piece (was one fixed "cinematic 35mm film still, shallow depth of field")
    foot_ = ((psys or {}).get("palette") or {}).get("footage")
    if foot_:
        photo_ = f"{photo_}, {foot_} colour palette"                                     # ONE palette per piece, stated in every still (round-3 QC: the liked pieces are teal + amber night, gold hour)
    pal_hint = None
    if psys and not (brief.get("palette") or fb.DIRECTION_PALETTES.get(dirn.get("palette"))):
        pf = psys["palette"]
        pal_hint = ["#%02x%02x%02x" % tuple(pf["bg"]), "#%02x%02x%02x" % tuple(pf["fg"])] + ["#%02x%02x%02x" % tuple(a) for a in pf["accents"]]
    palette = palette_for(moods[0], seed, brief.get("palette") or fb.DIRECTION_PALETTES.get(dirn.get("palette")) or pal_hint)
    lines = fb.lines_of(req["words"])
    lines0 = list(lines)                                                   # the request's lines: the concept's owners are one per line of these
    if req.get("layout_seed"):                                             # "Use this layout" of the Randomise-layout preview: the layout side of the system is the draw of that seed (stills / footage side kept)
        from tools.fusion import layout_preview as LPV
        psys = LPV.with_layout_seed(psys, req["layout_seed"], lines, cut, eff_req, mood=moods[0], bpm=bpm)
        notes = list(notes) + [f"layout seed {req['layout_seed']}: fonts, cut kinds, outline, layouts and palette pick drawn from it"]
    grid = make_grid(music, cut)
    style = psys["io"]["form_in"][0] if psys else _pick(seed, "io", ["lines"])      # material-carrying styles only: the rules' own gaps (strand transitions, pans) need strands. 2026-11-02: `blobs` / `particles` are banned (design_defaults.SPECKLE_IO), so only `lines` carries strands; before, the draw named a banned style and fix_plan turned it into a flicker, the whole piece became cuts and the last word left 2.4 s before the cut
    mode = req.get("shots", "cut")
    if not req.get("shots_given"):                                          # the brief's shot structure / a block's, unless the user chose
        bmode = next((((bk.BLOCKS[n].get("slots") or {}).get("shots") or {}).get("mode") for n in blocks_names
                      if ((bk.BLOCKS[n].get("slots") or {}).get("shots") or {}).get("mode")), None)
        mode = dirn.get("shots") if dirn.get("shots") in SHOT_MODES else (bmode if bmode in SHOT_MODES else mode)
    al = list((assignment or {}).get("lines") or [])                      # a batch assignment (batch_plan.assign_batch): per-line choices
    io_ = [(a["io_in"], a["io_out"]) for a in al] or None
    if io_ is None and psys:                                              # the piece's own io pair, alternating (the rules' tier defaults were flicker / lines for every plan)
        io_ = [(psys["io"]["form_in"][k % 2], psys["io"]["form_out"][k % 2]) for k in range(24)]
    st_n, st_note = story_shot_count(req.get("story_shots"), nframes)             # the storyboard's shot count (never more shots than pictures): exactly that many shots / lockups; invalid -> today's rule + a note
    if st_note and mode != "single":
        notes = notes + [st_note]
    spans = None if mode == "single" else cut_spans(nframes, grid, FPS, n=st_n, exact=bool(st_n))
    t0_fixed = None
    if OWNERS and spans is not None and len(spans) > 1:   # ONE lockup per shot, timed to the cuts (the words of a single shot keep the beat-grid schedule)
        lines, lock_note = fit_lockups(lines, len(spans))
        if lock_note:
            notes = notes + [lock_note]
        cum = [0]
        for sp_ in spans:
            cum.append(cum[-1] + int(sp_))
        sh = schedule_shot_words(lines, [c / float(FPS) for c in cum[:len(lines)]], tier, grid, cut, io_, style, bars_adjust)
        t0_fixed, sched, bars = [x[0] for x in sh], [x[1:] for x in sh], 0
    else:
        sched, bars, lines = schedule_words(lines, tier, grid, cut, bar_len, style, bars_adjust, io=io_, max_lines=len(al) or None)
    if al and len(al) != len(lines):
        notes = notes + [f"assignment has {len(al)} lines, the plan {len(lines)}: lines past the assignment use the rules"]
    mods = pick_modules(req["look"], energy, seed)
    fonts = pick_fonts(moods, energy, seed, len(lines))
    yaw = 10 if tier == "calm" else 16
    ysign = -1
    if psys:                                                              # the system's module(s), three of its shortlist faces, a yaw from the tempo band (alternating, start sign seeded)
        mods = list(psys.get("word_modules") or psys["modules"])          # 2026-11-01 user: a different module per word (was modules[0] / [-1] alternating)
        fonts = PS.three_faces(psys)
        yaw = round(sum(psys["motion"]["yaw"]) / 2.0, 1)
        ysign = int(psys["motion"]["start_sign"])
        if psys["camera"].get("stable"):                                  # 2026-10-30 (user): every word lands centred and holds still (stable_type.py): no yaw
            yaw = 0

    words = []
    prev_bleed = 1.0
    t0s = [0.0] * len(lines)
    if t0_fixed is not None:                             # one lockup per shot: t0 = the shot start
        t0s = list(t0_fixed)
    elif _spec().overlap_on({"tier": tier}, grid):       # slow tempo: the transition gap is stated (next.t0 - this.t1), not the calm default
        t0s = overlap_t0s([x[0] for x in sched], [((al[i]["io_in"], al[i]["io_out"]) if i < len(al) else (tuple(io_[i]) if io_ and i < len(io_) else _word_io(tier, i, style)))
                                                  for i in range(len(lines))], float(grid.iv))
    for i, (text, (span, form_s, exit_s, hit_on)) in enumerate(zip(lines, sched)):
        a = al[i] if i < len(al) else None
        fi, fo = (a["io_in"], a["io_out"]) if a else (tuple(io_[i]) if io_ and i < len(io_) else _word_io(tier, i, style))
        w = {"text": text, "font": a["font_file"] if a else fonts[i % len(fonts)], "module": a["module"] if a else mods[i % len(mods)],
             "t0": t0s[i], "t1": _r(t0s[i] + span), "form_s": _r(form_s),
             "exit_s": _r(exit_s), "hit_on": hit_on, "form_in": fi, "form_out": fo,
             "placement": {"yaw": ysign * yaw if i % 2 == 0 else -ysign * yaw}}
        if tier == "calm":
            w["ar_min"] = 0.8
        if a:                                                           # contrast comes from the assignment (fonts, layouts), not from bleed
            rec = copy.deepcopy(a["layout_record"])
            if rec.get("box") and not hl.eligible(rec["name"], text, fmt, w["module"]):     # the plan's line differs from the assigned one
                rec = hl.build(hl.DEFAULT, seed, f"{i}", fmt)
            w["placement"].update(hl.placement(rec))
            if rec.get("box"):
                w["screen_offset"] = rec["offset"]
                w["ar_min"] = rec["ar_min"]
                if rec.get("max_lines"):
                    w["max_lines"] = rec["max_lines"]
            w["hold_layout"] = rec
            w["assigned"] = {k: a[k] for k in ("module", "io_in", "io_out", "transition", "camera", "font", "layout") if k in a}
        else:
            bleed = prev_bleed = line_scale(words[-1]["font"] if words else None, w["font"], prev_bleed)
            if bleed < 1.0:
                w["placement"]["bleed"] = bleed
        pin_ = FP_pin_for(text, i, len(lines), req.get("font_pins"))                  # the user's own font for this line wins over the system / assignment / contrast rule
        if pin_:
            w["font"], w["font_pin"] = pin_, pin_
        ax = _font_axes(w["font"], tier, cut)
        if ax:
            w["axes"] = ax
        words.append(w)
    mo_ = PS.motion(bpm, seed)
    camera_params = {k: mo_["suggest"][k] for k in ("read_dps", "intro_deg", "fast_dps", "elev", "bob")} if tier == "calm" else None      # the tempo band, seeded (was one fixed set with elev 3.0)
    if psys and psys["camera"]["type"] == "beat_orbit":
        camera_params = dict(psys["camera"]["params"])
        if psys["camera"].get("stable"):
            camera_params = ST.static_camera_params(camera_params)
    if al:
        camera_params = None
    type_spec = {"seed": sub_seed(seed, "type") % 100000, "energy": energy, "mood": moods[0],
                 "palette": {"bg": _hex_rgb(palette[0]), "fg": _hex_rgb(palette[1]), "accents": [_hex_rgb(a) for a in palette[2:]]},
                 "camera": (psys["camera"]["type"] if psys else "beat_orbit"), "transitions": "auto", "transition_names": transition_names(), "tier": tier, "words": words,
                 "fps": FPS, "duration": _r(cut, 4), "bpm": _r(bpm, 3), "bar": 4,
                 "beats": [_r(b, 4) for b in music["beats"]], "downbeats": [_r(d, 4) for d in music["downbeats"]]}
    if camera_params:
        type_spec["camera_params"] = camera_params
    if req.get("font_pins"):
        type_spec["font_pins"] = copy.deepcopy(req["font_pins"])                   # rides on the piece: the design step, the type passes and spec.build_scene re-apply it
    if psys and psys["camera"].get("stable") and not al:                  # centred, still >= 1 beat, stacked / rolled layout (the design step does the same on its own words)
        ST.enforce(type_spec, psys, bpm, notes, seed=seed, fmt=fmt)
    if al:                                            # every word its assigned camera move (the auto tiling keeps follow between the words) + transitions
        type_spec["camera"] = "auto"
        type_spec["camera_override"] = {f"word:{i}": dict({"module": a["camera"]}, **hl.camera_params(words[i]["hold_layout"], a["camera"]))
                                        for i, a in enumerate(al[:len(words)])}
        type_spec["transitions"] = [(al[i].get("transition") or None) for i in range(min(len(al), len(words)) - 1)]
        type_spec["transitions"] += [None] * (len(words) - 1 - len(type_spec["transitions"]))
        for i in range(min(len(al), len(words)) - 1):     # block camera_pan_transition: the camera travel is the transition (spec key on the word, see pan_plan)
            if al[i].get("transition") == "pan":
                words[i]["transition"] = dict(kind="pan", **(al[i].get("pan") or {"dir": "down"}))
                type_spec["transitions"][i] = None
        if "chain_through_cards" in blocks_names:         # the block: the same moves travel through multiplane depth cards, every line between two cards
            cards = bk.resolve_params("chain_through_cards", seed, f"{seed}.cards")["cards"]
            type_spec["camera"] = {"chain": [dict({"module": a["camera"], "word": i}, **hl.camera_params(words[i]["hold_layout"], a["camera"]))
                                             for i, a in enumerate(al[:len(words)])], "plate": "multiplane", "cards": int(cards)}
            type_spec.pop("camera_override", None)
            for w in words:
                w["slot"] = 1
    if technique in _BEHINDS:
        type_spec["line"] = {"px": 4.0}

    scene, still = _scene_text(req, brief, moods, technique, photo_, concept=bool(req.get("shot_plan") or req.get("idea")))
    core = _first_words(scene, 28)
    if mode == "single":
        spans = [nframes]
        frames = [snap8(nframes)]
    else:                                                 # cut AND chain: hard cuts on downbeats (chain = each shot starts from the previous one's last frame)
        spans = spans or cut_spans(nframes, grid, FPS)
        frames = [snap8(sp) for sp in spans]
    shots = []
    pool = _MOTION[tier] if mode == "chain" else _SWEEP[tier]
    rot = sub_seed(seed, "motion") % len(pool)
    sp_plan = req.get("shot_plan") or []
    multi_cam = mode != "single" and len(frames) > 1 and bool(sp_plan)      # a concept piece (the app path); a typed request with no shot plan keeps the rules' templates
    if multi_cam:                                       # a distinct framing + move for EVERY shot (the concept wrote cameras for its lockups only: v014 r02 had 4 entries for 6 shots, six medium stills)
        from tools.fusion import shot_camera as _SCd
        sp_plan, _cam_fixes = _SCd.distinct(sp_plan, len(frames))
        if _cam_fixes:
            notes = notes + ["shot cameras: " + "; ".join(_cam_fixes)]
    n_model = 0
    # SHOT STRUCTURE: which shot continues the previous one (first = last:sK) and which starts a new place (a still of its own). The pure modes: chain = every shot continues, cut = none does;
    # the concept's shot plan may say it per shot (`continues`): places of two shots each (a new place, then its shot continues it) as the lead designer's pieces do.
    explicit = mode != "single" and any(isinstance(e, dict) and isinstance(e.get("continues"), bool) for e in sp_plan[1:])
    cont = [bool(k and ((sp_plan[k].get("continues") if (k < len(sp_plan) and isinstance(sp_plan[k], dict) and isinstance(sp_plan[k].get("continues"), bool)) else (mode == "chain")))) for k in range(len(frames))]         if explicit else [mode == "chain" and k > 0 for k in range(len(frames))]
    # 2026-10-29 (user: "why do all these videos use the last frame to start the next video's first frame? we should either be doing different shots completely or a single long prompted video"):
    # NO shot starts from another shot's last frame (it carries the drift / morph forward and keeps the camera: the doubled skater, the multiplying paddles). `cont[k]` now means "the SAME place
    # and subject from another camera": that shot gets its own ANGLE still (an edit of the place's first still, tools/fusion/angle_still.py; until the graph wires the MOSS edit it is a fresh
    # still of the same subject description with a different framing); every other shot a fresh still. Every shot has its own still, every boundary is a real cut. `legacy_chain` keeps the old behaviour.
    legacy_chain = bool(req.get("legacy_chain"))
    same = list(cont)
    if not legacy_chain:
        if any(same) and mode != "single":
            notes = notes + [f"structure: {sum(same)} shot(s) show the same place from another camera: each starts from its own angle still (angle_of the place's first still), none from the previous shot's last frame"]
        cont = [False] * len(frames)
    new_i = -1
    place_first: list[int] = []                       # index of the shot that starts each place (its model prompt, if any, describes the place: the place's still)
    for k, nfr in enumerate(frames):
        chained = cont[k] if (explicit or not legacy_chain) else (mode == "chain")
        motion = (brief.get("motions") or [])[k] if k < len(brief.get("motions") or []) else (_MOTION[tier] if chained else _SWEEP[tier])[(rot + k) % len(pool)]
        if not chained and k < len(brief.get("motions") or []) and "camera" not in motion.lower():
            motion = motion + ", " + _SWEEP[tier][(rot + k) % len(_SWEEP[tier])]       # one scene movement + ONE camera move
        parts = [core, _LETTER_PHRASE[technique], motion]                  # no trailing "cinematic" (it ended every shot prompt: diversity audit 2026-10-29)
        mp = model_shot_prompt(sp_plan[k]) if k < len(sp_plan) else ""
        lead_k = ""
        if multi_cam and k < len(sp_plan):
            from tools.fusion import shot_camera as _SCl
            lead_k = _SCl.lead(sp_plan[k])                       # the shot's framing + move LEAD its prompt ("Tight close-up, slow push in, ...")
        if mp:                                                   # the concept wrote THIS shot (what the subject does + one camera move chosen for it): the templates are only the salvage
            if lead_k:
                mp = re.sub(r",\s*camera:.*$", "", mp).strip(" ,;")
            parts = [mp]
            n_model += 1
        if lead_k:
            parts = [lead_k] + parts
        sid = f"s{k + 1}"
        if not cont[k]:
            new_i += 1
            place_first.append(k)
        first_ref = f"last:s{k}" if cont[k] else chr(ord("A") + (new_i if (mode == "cut" or explicit or not legacy_chain) else 0))
        shots.append({"id": sid, "kind": "control", "first": first_ref,
                      "frames": nfr, "seed": sub_seed(seed, f"shot{k + 1}"), "size": sizes["shot"], "guide": recipe["guide"],
                      "depth": bool(recipe["shot"].get("depth")), "matte": bool(recipe["shot"].get("matte")),
                      "prompt": ", ".join(p for p in parts if p), "control": {"pass": recipe["control_pass"]}})
        if k < len(sp_plan):
            shots[-1].update(PLT.shot_keys(sp_plan[k]))                  # optional plate / type_matte / subject_move / plate_pan keys the concept asked for (none = the plain shot)
            if isinstance(sp_plan[k], dict) and sp_plan[k].get("fx") in EM.EE.NAMES:
                shots[-1]["fx"] = sp_plan[k]["fx"]                        # an explore look the concept put on this shot (explore_effects.weave places it there)
        if spans[k] is not None and spans[k] != nfr:
            shots[-1]["span"] = int(spans[k])          # piece frames the shot occupies: generated slightly longer, the tail is cut
    effects_placed = EM.place_contrast(shots, psys, seed) if psys else []         # the contrast looks of the piece's system -> `type_matte: <name>` on a shot (plate_layers.shot_keys), one section each
    if mode == "single" and frames[0] > LONG_SHOT:
        shots[0]["long"] = True                        # longer than the LONG_SHOT-frame LTX cap: the footage code must extend / loop it
    for s in shots:                                   # keep P*.json style: drop false flags
        for key in ("depth", "matte"):
            if not s[key]:
                del s[key]

    compose = {"size": sizes["compose"], "crossfade": 0, "crf": 16 if quality == "final" else 20,
               **apply_recipe(recipe, cut, palette, sizes["compose"], grid)}
    from tools.fusion import angle_still as AS
    from tools.fusion import shot_camera as SC
    still_plain = still                                  # the shared scene text of every still; each still puts its own framing in FRONT of it (AS.lead_framing)
    frs = [SC.framing_of(sp_plan[k]) if (multi_cam and k < len(sp_plan)) else None for k in range(len(frames))]
    if sp_plan:
        fr0 = SC.framing_of(sp_plan[0])
        if multi_cam and fr0:
            still = AS.lead_framing(still_plain, fr0)
        elif fr0 and fr0 not in still.lower():
            still = f"{still}, {fr0} shot"
    stills = [{"id": "A", "model": "moss", "seed": sub_seed(seed, "still"), "size": sizes["type"], "prompt": still, "negative": _STILL_NEG}]
    n_places = sum(1 for c_ in cont if not c_) if (mode == "cut" or explicit or not legacy_chain) else 1
    if not legacy_chain and mode != "single":         # every SHOT has its own still: a fresh one for a new place, an angle still (same subject, other camera) for the same place
        root = 0
        for k in range(1, len(frames)):
            sid_k = chr(ord("A") + k)
            if same[k]:
                fr = SC.framing_of(sp_plan[k]) if k < len(sp_plan) else None
                fr = AS.normalize_framing(fr or SC.FRAMING_ORDER[k % len(SC.FRAMING_ORDER)])
                stills.append({"id": sid_k, "model": "moss", "angle_of": chr(ord("A") + root), "framing": fr, "seed": sub_seed(seed, f"still{k}"), "size": sizes["type"],
                               "prompt": f"{AS.lead_framing(stills[root]['prompt'], fr)}, the same subject and scene as above, {AS.framing_clause(fr)}", "negative": _STILL_NEG, "angle_fallback": True})
            else:
                root = k
                stills.append({"id": sid_k, "model": "moss", "seed": sub_seed(seed, f"still{k}"), "size": sizes["type"],
                               "prompt": place_still(sp_plan, k, fmt, photo_, frs[k]) or (AS.lead_framing(still_plain, frs[k]) if frs[k] else still) + (f", {_STILL_ANGLES[k % len(_STILL_ANGLES)] or _STILL_ANGLES[1]}" if not frs[k] else ""),
                               "negative": _STILL_NEG})
    elif mode == "cut" or explicit:                   # legacy: every NEW PLACE starts from its own still; a continued shot has none
        for k in range(1, n_places):
            stills.append({"id": chr(ord("A") + k), "model": "moss", "seed": sub_seed(seed, f"still{k}"), "size": sizes["type"],
                           "prompt": place_still(sp_plan, place_first[k] if k < len(place_first) else -1, fmt, photo_)
                           or f"{still}, {_STILL_ANGLES[k % len(_STILL_ANGLES)] or _STILL_ANGLES[1]}", "negative": _STILL_NEG})
    first = lines[0].lower()
    so_text = brief.get("signoff") or ("{a}", "{a} / {b} bpm", "{b} bpm")[sub_seed(seed, "signoff") % 3].format(a=_first_words(first, 4), b=int(round(bpm)))
    piece = {"id": f"A{seed}", "slug": _slug(req["words"]), "fps": FPS,
             "audio": {"src": music.get("src", ""), "bpm": int(round(bpm)), "cut": _r(cut, 4), "fade": _r(music.get("fade", 0.5), 2)},
             "type": type_spec, "type_passes": list(recipe["type_passes"]),
             "stills": stills,
             "shots": shots, "compose": compose,
             "signoff": {"text": so_text, "font": "Space Grotesk", "t0": _r(max(cut - SIGNOFF_LEAD_S, 0.0), 2), "px": 30 + sub_seed(seed, "signoff_px") % 15,
                         "color": _hex_rgb(palette[1]), "off": not req.get("signoff", False)},
             "technique": technique, "quality": quality, "format": fmt, "sizes": sizes,
             "brief": {k: v for k, v in brief.items() if v not in (None, "", [], {})},
             "plan": {"version": 1, "tier": tier, "shots": mode, "energy": energy, "look": req["look"], "bars": bars, "notes": list(notes), "seed": seed}}
    if req.get("layout_seed"):
        piece["plan"]["layout_seed"] = int(req["layout_seed"])
    if req.get("cut_pins"):
        piece["plan"]["cut_pins"] = list(req["cut_pins"])                       # the user's cut kind after shot k (the design step pins them in compose.cuts)
    if sp_plan or req.get("idea"):
        piece["plan"]["footage"] = {"idea": req.get("idea", ""), "shot_prompts": {"model": n_model, "rules": len(shots) - n_model}}
    if t0_fixed is not None:
        piece["plan"]["lockup_starts"] = [_r(x, 4) for x in t0_fixed]         # one lockup per shot: word k should start with shot k (pull_in_lags checks the dry run against it)
    if req.get("earlier_openings"):
        piece["plan"]["earlier_openings"] = list(req["earlier_openings"])
    if psys:
        piece["plan"]["piece_system"] = copy.deepcopy(psys)                  # the design step designs inside it and folds outliers into it (piece_system.fold_*)
        if effects_placed or (psys.get("effects") or {}).get("active"):
            piece["plan"]["effects"] = dict(placed=effects_placed, active=list((psys.get("effects") or {}).get("active") or []), requested=list((psys.get("effects") or {}).get("requested") or []),
                                            unknown=list((psys.get("effects") or {}).get("unknown") or []))
    piece["plan"]["structure"] = {"mode": mode, "explicit": bool(explicit), "places": n_places, "first": [s_["first"] for s_ in shots],
                                  "angle": {s_["id"]: s_["angle_of"] for s_ in stills if s_.get("angle_of")}, "last_frame_chaining": bool(legacy_chain and any(cont))}
    if blocks_names:
        piece["plan"]["blocks"] = bk.describe(blocks_names, seed, len(words), tag=str(seed))
    if dirn:
        piece["plan"]["direction"] = dict(dirn)
    if al:
        piece["plan"]["assignment"] = [{k: v for k, v in a.items() if k not in ("layout_record", "camera_params")} for a in al]
        piece["plan"]["layouts"] = [(w["hold_layout"]["name"] + "/" + w["hold_layout"]["variant"]) if w.get("hold_layout") else "full"
                                    for w in words]
    if OWNERS and quality != "draft":                    # one owner per word: render / ltx_at / material (owner.py); the shots get the letter clause and the guide (a draft's stand-in footage has no letters: all flat)
        given = OW.remap_owners(req.get("owners"), lines0, lines)          # the concept's choice per lockup (fitted to the final lines); rules defaults where it gave none
        for i, w in enumerate(words):
            o = (given[i] if given and i < len(given) else None) or {}
            for k in ("render", "ltx_at", "material"):
                if o.get(k) is not None:
                    w[k] = o[k]
            w.update(OW.word_fields(w, moods[0], type_spec["palette"], scene))
            if w["render"] == "flat":
                w.pop("material", None)
        piece["plan"]["owner"] = {"source": "concept" if given and any(o.get("render") for o in given) else "rules"}
    piece["layout"] = layout(piece)
    if OWNERS and quality != "draft":
        OW.apply_to_plan(piece)
        for sh_ in piece["shots"]:
            if OW.CLAUSE_RE.search(sh_.get("prompt", "")):          # the clause names the letters: the generic built_in / traced letter phrases would contradict it
                for ph in (_LETTER_PHRASE["built_in"], _LETTER_PHRASE["traced"]):
                    sh_["prompt"] = sh_["prompt"].replace(", " + ph, "")
    from tools.fusion import text_guard as TG                       # the renderer never invents its own text: a shot with no control pass names no text (and its negative says so), a shot with an LTX word spells exactly that word
    TG.fix_plan(piece)
    return piece


def dry_run(piece: dict, size=None, camera: bool = False) -> dict:
    """spec.build_scene at 576x1024 on the piece's type spec -> {'notes': [...], 'last_end': s, 'words': [...]}.
    The camera plan (CameraChain clearance, ~30 s) does not touch the timeline notes, so it is skipped unless camera=True."""
    sp = _spec()
    size = tuple(size) if size else tuple(orient(DRY_SIZE, format_of(piece)))
    ts = copy.deepcopy(piece["type"])
    for w in ts.get("words", []):                      # contour-port keys do not change the timeline (and spec.build_scene may not know them yet)
        for k in ("entry", "exit"):
            if isinstance(w.get(k), dict):
                del w[k]
        w.pop("via", None)
        w.pop("matte", None)
    if not camera:
        ts["camera"] = None
    from mograph.render import cam3d
    world = (cam3d.W, cam3d.H)                      # build_scene sets the global world plane; give it back afterwards
    try:
        scene = sp.build_scene(ts, size=size)
    finally:
        cam3d.set_world(*world)
    words = scene.info["words"]
    from mograph.modules.core import WordModule
    trs = [it for it, _a, _b in scene.seq.timeline if not isinstance(it, WordModule)]                # the transition between each word pair, in order (WORD RULES: tools/fusion/word_rules.py reads these)
    return {"notes": list(scene.info.get("notes") or []), "words": words,
            "last_end": max((w["end"] for w in words), default=0.0), "duration": scene.duration,
            "transitions": [getattr(t, "name", type(t).__name__) for t in trs],
            "trans_rank": [[r[0] for r in (getattr(getattr(t, "engine", None), "ranking", None) or [])] for t in trs]}


# =================================================================================================== footage flow (flow_wake)
# The footage's motion pushes / drags / streaks the letters (compose.flow_wake). It is part of EVERY generation: weave_flow puts
# one flow_wake op per type alpha the recipe reads in front of the recipe's own ops and points those ops at the displaced alpha.
# Influence = a motion matte derived from the footage flow itself ($flow_m, camera drift removed), so no Alpha-Gen is needed and
# it works for every technique. `settle` (0 = full effect, 1 = relax fast) follows the planned word holds: the type is still
# when a word hits and holds, and free to be pushed while it forms and exits.
WAKE = {"radius": 70, "decay": 0.9, "trail": 10, "trail_gain": 1.0, "bg_drag": 0.12, "max_disp": 0.2, "smooth": 1.0}
WAKE_STRENGTH = 4.0                                             # without an onset curve
WAKE_ALPHAS = ("type_matte", "type_line")                       # type alphas a recipe may composite (rgb passes are left alone)
HOLD_MIN_S = 0.2
SETTLE_RAMP_S = 0.12
SETTLE_HOLD = 1.0                                               # settle inside a hold: FULLY settled (2026-10-29, clear text: no smear / echo / residual push while a word is read; it was 0.75 = a subtle residual movement)


def settle_timeline(holds, ramp: float = SETTLE_RAMP_S) -> dict:
    """{'timeline': keys} 0 outside the holds, SETTLE_HOLD inside (piece seconds); overlapping windows merge."""
    ws = []
    for a, b in sorted((float(a), float(b)) for a, b in holds if float(b) - float(a) >= HOLD_MIN_S):
        r = min(ramp, (b - a) / 4.0)
        if ws and a - r <= ws[-1][1] + ws[-1][2]:
            ws[-1][1], ws[-1][2] = max(ws[-1][1], b), max(ws[-1][2], r)
        else:
            ws.append([a, b, r])
    keys = [[0.0, 0.0]]
    for a, b, r in ws:
        for t, v in ((a - r, 0.0), (a + r, SETTLE_HOLD), (b - r, SETTLE_HOLD), (b + r, 0.0)):
            t = _r(max(t, keys[-1][0] + 0.001), 3)
            keys.append([t, v])
    return {"timeline": keys, "ease": "smooth"}


def _swap_refs(obj, mp: dict):
    if isinstance(obj, str):
        return mp.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _swap_refs(v, mp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap_refs(v, mp) for v in obj]
    return obj


def _arg_refs(ops) -> set:
    found: set = set()

    def walk(o):
        if isinstance(o, str):
            if o.startswith("$"):
                found.add(o[1:])
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk([op.get("args", {}) for op in ops])
    return found


def wake_ops(ops: list, holds, key_prefix: str, strength=None, wake: dict | None = None) -> list:
    """ops of one recipe chain -> the same chain with a flow_wake op per type alpha it reads, the readers re-pointed at it."""
    refs = _arg_refs(ops)
    use = [a for a in WAKE_ALPHAS if a in refs and not any(o.get("out") == a for o in ops)]
    if not use:
        return copy.deepcopy(ops)
    out_name = {a: "w" + a.split("_")[1][0] for a in use}                   # type_matte -> wm, type_line -> wl
    mp = {f"${a}": f"${out_name[a]}" for a in use}
    settle = settle_timeline(holds)
    pre = []
    for a in use:
        args = {"type_a": f"${a}", "flow": "$flow", "matte": "$flow_m"}
        args.update(copy.deepcopy(dict(WAKE, **(wake or {}))))
        args["strength"] = copy.deepcopy(strength) if strength is not None else WAKE_STRENGTH
        args.update({"settle": copy.deepcopy(settle), "frame_idx": "$frame_idx", "u": "$u", "state_key": f"{key_prefix}:{a}"})
        pre.append({"op": "flow_wake", "out": out_name[a], "args": args})
    return pre + [_swap_refs(copy.deepcopy(o), mp) for o in ops]


def word_holds(piece: dict, music: dict | None, chk: dict | None = None) -> list[tuple[float, float]]:
    """(hit, release) of every planned word, piece seconds: the windows where the word must read. From the dry-run readout when
    given, else from the planner's own timing (spec.plan_timing on the beat grid)."""
    if chk and chk.get("words"):
        return [(float(w["hit"]), float(w["release"])) for w in chk["words"]]
    t = piece["type"]
    grid = make_grid(music or {"bpm": t["bpm"], "beats": t["beats"], "downbeats": t["downbeats"]}, float(t["duration"]))
    tm = _spec().plan_timing(copy.deepcopy(t["words"]), t, grid)
    return [(float(x["hit"]), float(x["rel"])) for x in tm]


def weave_flow(piece: dict, music: dict | None = None, chk: dict | None = None, wake: dict | None = None) -> dict:
    """add flow_wake to every compose chain of the piece (in place; returns it). Idempotent: a chain that already has one is kept."""
    comp = piece.get("compose") or {}
    holds = word_holds(piece, music, chk)
    curves = (music or {}).get("curves") or piece.get("curves") or {}
    strength = {"curve": "onset", "lo": 3.0, "hi": 6.0} if curves.get("onset") else None
    n = 0
    chains = [comp["default"]] if comp.get("default") else []
    chains += list((comp.get("shots") or {}).values())
    for ch in chains:
        if any(o.get("op") == "flow_wake" for o in ch.get("ops", [])):
            continue
        ch["ops"] = wake_ops(ch.get("ops", []), holds, str(piece.get("id", "piece")), strength, wake)
        n += 1
    piece.setdefault("plan", {})["flow_wake"] = {"chains": n, "holds": [[_r(a, 3), _r(b, 3)] for a, b in holds]}
    for s in piece.get("shots", []):
        s["flow"] = True                              # footage optical flow (MographFootageFlow) -> $flow
    return piece


# =================================================================================================== hold layouts (backing shapes)
# A layout with shapes (hold_layouts.LAYOUTS: edge_L, solid_strips, split_sides, type_circle, construction_triangle, cells) paints flat shapes
# behind its line: compose op `layout_shapes` (tools/fusion/compose.py), screen space, visible from the word's start to its end (fade 0.3 s).
# The op sits in front of the recipe's plate users (sandwich / depth_occlude / knockout footage, the grade of the plate), tint_type keeps
# sampling the real plate; with a subject matte ($m of upscale_matte) the shapes never cover the subject. The type itself is positioned in the
# type stage (placement wf / hf + screen_offset), so every type pass and the LTX control video agree with the shapes.
SHAPE_LUM = {"dark": 0.22, "ink2": 0.26, "accent": 0.42}          # luminance caps: the type is tinted to ~0.9, a flat backing stays dark
SHAPE_FADE_S = 0.3
_PLATE_USERS = {("sandwich", "footage"), ("depth_occlude", "footage"), ("grade", "frame"), ("knockout", "footage"), ("knockout", "plate_rgb")}
_SUBJECT_RE = re.compile(r"\b(?:" + "|".join(_SUBJECTS) + r")s?\b")


def _lum(c) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def shape_color(role: str, palette: dict) -> list[float]:
    """colour role (hold_layouts.COLOR_ROLES) -> [r, g, b] 0..1 from the piece palette ({bg, fg, accents} as 0..255): dark = the background
    colour, ink2 = accent 0 and accent = accent 1, each scaled to its luminance cap (SHAPE_LUM); ink2 is kept visibly apart from dark."""
    acc = [list(a) for a in (palette.get("accents") or [palette["fg"]])]
    base = {"dark": palette["bg"], "ink2": acc[0], "accent": acc[1 % len(acc)]}[role]
    c = [v / 255.0 for v in base]
    cap = SHAPE_LUM[role]
    L = _lum(c)
    target = min(L, cap) if role == "dark" else min(max(L, 0.6 * cap), cap)
    c = [v * target / L for v in c] if L > 1e-6 else [target] * 3
    if role == "ink2":
        d = shape_color("dark", palette)
        if max(abs(a - b) for a, b in zip(c, d)) < 0.06:
            c = [min(1.0, v + 0.09) for v in c]
    return [_r(min(1.0, max(0.0, v)), 3) for v in c]


def word_spans(piece: dict, music: dict | None = None, chk: dict | None = None) -> list[tuple[float, float]]:
    """(start, end) of every word's module window in piece seconds (dry-run readout when given, else the planner's own timing)."""
    if chk and chk.get("words") and all("start" in w and "end" in w for w in chk["words"]):
        return [(float(w["start"]), float(w["end"])) for w in chk["words"]]
    t = piece["type"]
    grid = make_grid(music or {"bpm": t["bpm"], "beats": t["beats"], "downbeats": t["downbeats"]}, float(t["duration"]))
    return [(float(x["S"]), float(x["end"])) for x in _spec().plan_timing(copy.deepcopy(t["words"]), t, grid)]


def piece_has_subject(piece: dict) -> bool:
    """does the shot have a subject a contour / matte can be taken from (a matte technique, or a subject word in the scene)?"""
    if piece.get("technique") in _BEHINDS:
        return True
    st = " ".join(str(st.get("prompt", "")) for st in piece.get("stills", [])[:1]) + " " + str((piece.get("brief") or {}).get("scene", ""))
    return bool(_SUBJECT_RE.search(st.lower()))


def weave_layout(piece: dict, music: dict | None = None, chk: dict | None = None) -> dict:
    """add the `layout_shapes` op to every compose chain of a piece whose words carry layouts with shapes (in place; idempotent).
    Words without shapes cost nothing: no op is added when no word has any."""
    words = piece["type"]["words"]
    have = [(i, w["hold_layout"]) for i, w in enumerate(words) if (w.get("hold_layout") or {}).get("shapes")]
    plan = piece.setdefault("plan", {})
    if not have:
        plan["layout_shapes"] = {"words": 0}
        return piece
    spans = word_spans(piece, music, chk)
    pal = piece["type"]["palette"]
    persist = next((blk for blk in plan.get("blocks", []) if blk["name"] == "strips_persist_across_cuts"), None)
    shapes = []
    for i, rec in have:
        a, b = spans[i]
        if persist and i + 1 < len(spans):                                # strips_persist_across_cuts: the strip stays up over the cut
            b = max(b, min(b + float(persist["params"].get("persist_s", 1.0)), spans[i + 1][0] + 0.3))
        for sh in rec["shapes"]:
            d = copy.deepcopy(sh)
            d["color"] = shape_color(sh["color"], pal)
            d["t0"], d["t1"] = _r(a, 3), _r(b, 3)
            shapes.append(d)
    comp = piece.get("compose") or {}
    chains = [comp["default"]] if comp.get("default") else []
    chains += list((comp.get("shots") or {}).values())
    n = 0
    for ch in chains:
        ops = ch.get("ops", [])
        if any(o.get("op") == "layout_shapes" for o in ops):
            continue
        k = next((j + 1 for j, o in enumerate(ops) if o.get("op") == "upscale_matte" and o.get("out") == "m"), 0)
        args = {"img": "$footage", "shapes": copy.deepcopy(shapes), "t": "$t", "fade": SHAPE_FADE_S}
        if k:
            args["matte"] = "$m"                          # the subject stays in front of the strips
        used = 0
        tail = []
        for o in ops[k:]:
            o = copy.deepcopy(o)
            for key, v in list(o.get("args", {}).items()):
                if v == "$footage" and (o["op"], key) in _PLATE_USERS:
                    o["args"][key] = "$plate"
                    used += 1
            tail.append(o)
        if used:
            ch["ops"] = ops[:k] + [{"op": "layout_shapes", "out": "plate", "args": args}] + tail
            n += 1
    plan["layout_shapes"] = {"words": len(have), "shapes": len(shapes), "chains": n}
    return piece


_TAIL_OPS = ("grade", "grain", "vignette")
_RGB_OUT = ("sandwich", "depth_occlude", "knockout", "tint_type", "add_glow", "grade", "halation", "grain", "vignette", "fill", "refract", "over", "crossfade",
            "screen", "multiply", "overlay", "soft_light", "blur", "translate")


def _running_picture(head_ops: list) -> str:
    """name of the picture buffer a chain has made by the end of `head_ops`: `image` when an op wrote it, else the out of the last picture-making op
    (a model-written chain names its steps; `image` is only its last op), else `image` (rules chains / nothing made yet)."""
    if any(o.get("out", "image") == "image" for o in head_ops):
        return "image"
    for o in reversed(head_ops):
        if o.get("op") in _RGB_OUT and isinstance(o.get("out"), str):
            return o["out"]
    return "image"


def weave_matte(piece: dict, music: dict | None = None, chk: dict | None = None) -> dict:
    """the matte treatments of the words' `"matte"` keys (tools/fusion/matte_treatments.py: front | window | fill | wipe | glow) as compose ops
    in every chain, over each word's hit .. release window, before the closing grade / grain / vignette. A treatment that needs the subject matte
    is only woven into a chain that has one ($m of upscale_matte); `window` runs without (no subject-steps-out). Idempotent."""
    plan = piece.setdefault("plan", {})
    words = piece["type"]["words"]
    if not any(w.get("matte") for w in words):
        plan["matte_treatments"] = {"words": 0}
        return piece
    try:
        from tools.fusion import matte_treatments as MT
    except Exception:                                  # noqa: BLE001
        plan["matte_treatments"] = {"words": 0, "note": "matte_treatments not importable"}
        return piece
    holds = word_holds(piece, music, chk)
    trs = MT.from_words(words, holds)
    comp = piece.get("compose") or {}
    chains = [comp["default"]] if comp.get("default") else []
    chains += list((comp.get("shots") or {}).values())
    n = 0
    for ch in chains:
        ops = ch.get("ops", [])
        if any(str(o.get("out", "")).startswith("_") and o["out"][1:3] in ("fr", "wi", "fi", "wp", "gl") for o in ops):
            continue
        outs = {o.get("out") for o in ops}
        has_m = "m" in outs
        keep = [dict(t, subject_out=False) if (t["matte"] == "window" and not has_m) else t for t in trs if has_m or not MT.NEEDS_MATTE[t["matte"]]]
        if not keep:
            continue
        k = len(ops)
        while k > 0 and ops[k - 1].get("op") in _TAIL_OPS:
            k -= 1
        base = _running_picture(ops[:k])             # a model-written chain makes `image` only at its last op: the treatments paint on the picture made so far
        new = MT.ops_for(keep, base=base, plate="$foot" if "foot" in outs else "$footage", matte="$m",
                         type_a="$type_matte", type_rgb="$tcol" if "tcol" in outs else (1.0, 0.92, 0.78))
        ch["ops"] = ops[:k] + new + ops[k:]
        n += 1
    plan["matte_treatments"] = {"words": sum(1 for w in words if w.get("matte")), "treatments": len(trs), "chains": n}
    return piece


def contour_keys_of(piece: dict) -> list[dict]:
    """the contour-port keys (entry / exit / via / matte) of every line: the batch assignment's when present, else chosen here by the same
    rules (batch_plan._contour_keys). [] for a piece without a subject."""
    if not piece_has_subject(piece):
        return []
    asg = (piece.get("plan") or {}).get("assignment")
    if asg and all("entry" in a for a in asg):
        return [{k: a[k] for k in ("entry", "exit", "via", "matte") if k in a} for a in asg]
    from tools.fusion import batch_plan
    n = len(piece["type"]["words"])
    ports, mattes = batch_plan.piece_ports(piece)
    keys = batch_plan._contour_keys([{"subject": True, "n": n, "ports": ports, "mattes": mattes}], [(0, i) for i in range(n)],
                                    batch_plan.contour_port_names(), (piece.get("plan") or {}).get("seed", 0))
    return [keys[(0, i)] for i in range(n)]


def apply_contour_keys(piece: dict) -> dict:
    """record the contour keys in ``plan.contour`` and, when tools/fusion/contour_ports.py exists, write them into the spec words
    (entry / exit / via / matte; mograph/modules/contour_ports.py reads them, spec.build_scene falls back to the plain module sides
    when no plate data is bound). Without the module the spec words stay as they are. Idempotent."""
    from tools.fusion import batch_plan
    keys = contour_keys_of(piece)
    avail = batch_plan.contour_ports_available()
    ports, mattes = batch_plan.piece_ports(piece)
    names = [b["name"] for b in piece.get("plan", {}).get("blocks", [])]
    for i, mode in bk.matte_for_lines(names, len(keys), mattes).items():          # a block that asks for a matte treatment on a line gets it
        if 0 <= i < len(keys):                          # (a piece without a subject has no keys; `last` of 0 lines is -1)
            keys[i]["matte"] = mode
    piece.setdefault("plan", {})["contour"] = {"available": avail, "subject": bool(keys), "lines": keys}
    if avail:
        for w, k in zip(piece["type"]["words"], keys):
            w.update(copy.deepcopy(k))
    return piece


# =================================================================================================== batch (tools/fusion/batch_plan.py)
def block_ctx(piece: dict) -> dict:
    """the context the lego blocks are matched against (blocks.eligible): shot mode / count, technique, lines, format, subject, shortest hold."""
    hs = [b - a for a, b in word_holds(piece, None)]
    mh = min(hs, default=1.6)
    return bk.piece_ctx(holds=hs, shots=(piece.get("plan") or {}).get("shots", "cut"), technique=piece.get("technique", "built_in"),
                        n_lines=len(piece["type"]["words"]), fmt=format_of(piece), subject=piece_has_subject(piece), min_hold_s=mh,
                        n_shots=len(piece.get("shots", [])) or None, tier=(piece.get("plan") or {}).get("tier", "mid"))


def plan_with_blocks(request: dict, music: dict, brief_text=None, seed: int = 0, **kw) -> dict:
    """ONE piece with lego blocks: request + music analysis + Qwen brief (blocks + direction + creative text) -> the full piece. Same as a
    one-piece plan_batch (blocks chosen from what the plan offers, then the assignment of modules / io / transitions / cameras / fonts / layouts)."""
    pieces, _ = plan_batch([request], [music], seed, briefs=[brief_text], **kw)
    return pieces[0]


def assign_batch(pieces, seed: int = 0, **kw) -> dict:
    """a batch of pieces together uses every word module / transition / io style / camera move / hold layout; see batch_plan.assign_batch."""
    from tools.fusion import batch_plan
    return batch_plan.assign_batch(pieces, seed, **kw)


def plan_batch(requests, musics, seed: int = 0, **kw):
    """plan_piece for every request with the batch assignment applied -> (pieces, assignment); see batch_plan.plan_batch."""
    from tools.fusion import batch_plan
    return batch_plan.plan_batch(requests, musics, seed, **kw)


# =================================================================================================== tail
def fill_tail(piece: dict, music: dict, chk: dict, size=None, max_iter: int = 3) -> dict:
    """No dead air: when the last word leaves more than TAIL_MAX_S before the cut, its hold grows (the planned span is checked by
    the same dry run). The signoff then starts SIGNOFF_OVERLAP_S before the last word is gone and runs to the cut, so the type
    layer is never empty. -> the dry-run readout that is valid for the final piece."""
    cut = float(music["cut"])
    words = piece["type"]["words"]
    last = words[-1]
    if cut - chk["last_end"] > TAIL_MAX_S:
        base_t1, base_x = float(last["t1"]), float(last["exit_s"])
        goal = cut - TAIL_TARGET_S
        grid = make_grid(music, cut)
        cands = []
        # the word's release and end snap to the beat grid: scan hold growth (t1) x exit length with the cheap timing function,
        # prefer the end closest to the goal with the hold (not the exit) doing the filling, then confirm with the dry run
        for k in range(0, 13):
            for x in (base_x, 0.45, 0.6, 0.9, 1.2):
                w = copy.deepcopy(words)
                w[-1].update(t1=_r(base_t1 + 0.3 * k), exit_s=_r(x))
                t = _spec().plan_timing(w, piece["type"], grid)[-1]
                if t["end"] > cut - 0.3 + 1e-6:
                    continue
                cost = abs(t["end"] - goal) + 0.4 * abs((t["end"] - t["rel"]) - 0.6) + 0.02 * k
                cands.append((cost, w[-1]["t1"], w[-1]["exit_s"]))
        for _, t1, x in sorted(cands)[:3]:
            last["t1"], last["exit_s"] = t1, x
            new = dry_run(piece, size)
            if not new["notes"] and new["last_end"] <= cut - 0.3 and new["last_end"] > chk["last_end"] + 1e-6:
                chk = new
                break
        else:
            last["t1"], last["exit_s"] = base_t1, base_x
    so = piece["signoff"]
    so["t0"] = _r(min(max(chk["last_end"] - SIGNOFF_OVERLAP_S, 0.0), max(cut - SIGNOFF_MIN_S, 0.0)), 2)
    piece["plan"]["tail"] = {"last_end": _r(chk["last_end"], 2), "cut": _r(cut, 2), "gap": _r(cut - chk["last_end"], 2),
                             "signoff_t0": so["t0"]}
    return chk


def weave_hold_life(piece: dict, music: dict, chk: dict | None, size) -> dict:
    """a request's hold-life / soft_support names -> compose, then tools/fusion/hold_life.weave (1-2 of letter_flicker / glitch_slice / pulse / glow_breathe / outline_shimmer on the drawn type in
    each hold). Called by _finish (rules pieces) AND by design_step._finish (a Gemma-designed piece replaces the rules' compose, so the weave has to run again on it: r14/r15 pieces had none)."""
    _hl = [n for n in (((piece.get("plan") or {}).get("effects") or {}).get("requested") or []) if n in EM.HOLD_LIFE_EFFECTS]       # a request's hold-life names -> compose.hold_life (exactly those)
    if _hl and (piece.get("compose") or {}).get("hold_life") is None:
        piece.setdefault("compose", {})["hold_life"] = _hl
    if "soft_support" in (((piece.get("plan") or {}).get("effects") or {}).get("requested") or []):
        piece.setdefault("compose", {})["soft_support"] = True            # requested: the soft shadow / scrim may back the type (default: pure type)
    from tools.fusion import hold_life
    return hold_life.weave(piece, music, chk, size)


def _finish(piece: dict, music: dict, chk: dict | None, size) -> dict:
    chk_dry = chk is not None
    FP_apply(piece["type"]["words"], piece["type"].get("font_pins"), (piece.get("plan") or {}).get("notes"))      # whatever a batch / block / system step wrote, the user's fonts stay
    if chk is None:                                   # no dry run: the planner's own timing
        tm = _spec().plan_timing(copy.deepcopy(piece["type"]["words"]), piece["type"], make_grid(music, float(music["cut"])))
        chk = {"words": [{"hit": x["hit"], "release": x["rel"], "end": x["end"]} for x in tm], "notes": [],
               "last_end": tm[-1]["end"]}
        so = piece["signoff"]
        so["t0"] = _r(min(max(chk["last_end"] - SIGNOFF_OVERLAP_S, 0.0), max(float(music["cut"]) - SIGNOFF_MIN_S, 0.0)), 2)
    else:
        chk = fill_tail(piece, music, chk, size)
    apply_contour_keys(piece)
    if chk_dry:                                       # a composed-hold type camera (follow / hold_drift) framed words past the frame edge (v014 r02): measured on the real scene, beat_orbit if so
        from tools.fusion import type_frame
        type_frame.guard(piece, piece.setdefault("plan", {}).setdefault("notes", []), size=(piece.get("sizes") or {}).get("type") or sizes_for(piece.get("quality", "final"), format_of(piece))["type"], only_risky=True)
    weave_layout(piece, music, chk)
    weave_matte(piece, music, chk)
    weave_flow(piece, music, chk)
    from tools.fusion import edge_run, strip_resnap  # lego blocks, compose side (no-ops without their block)
    strip_resnap.weave(piece, music, chk)
    edge_run.weave(piece, music, chk)
    from tools.fusion import pan_plan  # block camera_pan_transition: the plate travels with the type camera (no-op without a pan)
    pan_plan.weave(piece, music, chk, size)
    weave_hold_life(piece, music, chk, size)
    from tools.fusion import explore_effects; explore_effects.weave(piece, music, chk, size)           # 2026-10-30 the hearted explore looks (x14 / x08 / x10 / x03 / x02) as 2-4 s sections
    from tools.fusion import contour_morph; contour_morph.weave(piece, music, chk, size)   # 2026-10-30 ship #8: CONTOUR_MORPH = the default word-to-word move on a shot with a subject matte (before the sweep, which then skips those pairs)
    from tools.fusion import word_rules; word_rules.enforce(piece, music, chk, size)         # 2026-10-31 WORD RULES: never the same word transition on two consecutive gaps, at most 2 per piece (same rule as the cuts)
    from tools.fusion import contour_sweep; contour_sweep.weave(piece, music, chk, size)   # 2026-10-30 contour sweep: on when requested (effects: contour_sweep) or drawn (seeded 40 %); compose.contour_sweep = false opts out
    piece["layout"] = layout(piece)
    return piece


def pull_in_lags(piece: dict, chk: dict, size, tries: int = 3) -> dict:
    """one lockup per shot: the engine starts a word after the previous one's end + gap, which runs later than the planner's simulator when the words are
    joined by cuts / transitions (qwen_piece does the same). Where word k starts more than LAG_S after its shot start, the previous word's hold is cut by
    that lag (its readable minimum is kept) and the dry run repeated. -> the dry-run readout valid for the final piece."""
    starts = (piece.get("plan") or {}).get("lockup_starts")
    words = piece["type"]["words"]
    if not starts or len(starts) != len(words):
        return chk
    sp = _spec()
    for _ in range(tries):
        lag = [(float(c["start"]) - float(starts[k]), k) for k, c in enumerate(chk["words"]) if k > 0]
        worst = max(lag, default=(0.0, 0))
        if worst[0] <= LAG_S:
            break
        k = worst[1]
        prev = words[k - 1]
        cur = float(prev["t1"])
        floor = float(prev["t0"]) + float(prev.get("form_s", 0.9)) + max(sp.min_hold({"text": prev["text"]}), 1.0)
        new = max(cur - worst[0], floor)
        if new >= cur - 1e-6:
            break
        prev["t1"] = _r(new)
        trial = dry_run(piece, size)
        if max([float(c["start"]) - float(starts[j]) for j, c in enumerate(trial["words"]) if j > 0], default=0.0) >= worst[0] - 1e-6:
            prev["t1"] = _r(cur)                                # no gain: keep the planned hold
            break
        chk = trial
    return chk


def enforce_coverage(piece: dict, chk: dict, size=None) -> dict:
    """COVERAGE RULE (QC v026: 1-3 s with no word on screen): the stretch between a word's release and the next word's form-in start is at most max(1 beat, 0.75 s), except across a designed 3D
    cut's move window or a pan (those show their own transition). Shortens the exit / pulls the next word in / switches the overlap mode on (stable_type.enforce_coverage, every try is a dry run on the
    real type scene and is undone when it breaks a hold, a cut or a shot's hit). In place on piece["type"] (t0 / t1 / exit_s / overlap); the notes go to plan.coverage. -> the dry-run readout valid for the final piece."""
    t = piece.get("type") or {}
    if len(t.get("words") or []) < 2:
        return chk
    from tools.fusion import pan_plan
    lay = layout(piece)
    rows = [dict(id=sid, start=round(L["start"] / FPS, 3)) for sid, L in lay.items()]
    windows = ST.transition_windows((piece.get("compose") or {}).get("cuts"), rows)
    pans = {i for i, w in enumerate(t["words"]) if pan_plan._is_pan(w.get("transition"))}
    notes: list = []

    def run(tt):
        return dry_run(dict(piece, type=tt), size)["words"]
    try:
        before = copy.deepcopy(t)
        ST.enforce_coverage(t, run, windows, pans, [r["start"] for r in rows], float((piece.get("audio") or {}).get("cut") or t.get("duration") or 0.0) or None, notes)
    except Exception as e:       # noqa: BLE001  never break a piece for it
        notes.append(f"coverage skipped: {type(e).__name__}: {str(e)[:120]}")
        piece["type"] = before
        t = piece["type"]
    if notes:
        piece.setdefault("plan", {})["coverage"] = notes
    return dry_run(piece, size) if t != before else chk


def plan_piece(request: dict, music: dict, brief_text=None, dry: bool = True, size=None, assignment: dict | None = None) -> dict:
    """request + music analysis (+ brief text) -> piece dict. Deterministic. The dry run is built at the request's format
    (576x1024 or 1024x576); the footage flow (flow_wake) and the tail fill are part of every plan.
    assignment: one piece of ``assign_batch(...)["pieces"]`` (module / io / transition / camera / font / layout per line, contour keys);
    without it the rules choose as before."""
    req = normalize_request(request)
    size = tuple(size) if size else tuple(orient(DRY_SIZE, req["format"]))
    log: list[str] = []
    brief = fb.parse_brief(brief_text, req, log)
    notes = [f"brief: {m}" for m in log]
    if fb.brief_is_empty(brief):
        notes.append("brief: empty -> rules only")
    piece = None
    for attempt in range(MAX_RETRIES + 1):
        piece = _build(req, music, brief, -attempt, notes, assignment)
        if not dry:
            return _finish(piece, music, None, size)
        chk = pull_in_lags(piece, dry_run(piece, size), size)
        chk = enforce_coverage(piece, chk, size)                          # no long stretch with no word on screen (stable_type.enforce_coverage)
        bad = list(chk["notes"])
        if chk["last_end"] > float(music["cut"]) - 0.3:
            bad.append(f"last word ends at {chk['last_end']:.2f}s of {float(music['cut']):.2f}s")
        if not bad:
            if attempt:
                piece["plan"]["notes"] = notes + [f"windows shrunk by {attempt} bar(s) after the dry run"]
            piece["plan"]["tries"] = attempt + 1
            return _finish(piece, music, chk, size)
        notes = notes + [f"dry run {attempt + 1}: " + "; ".join(bad)]
    piece["plan"]["notes"] = notes + ["WARNING: dry run still reports notes after the retries"]
    piece["plan"]["tries"] = MAX_RETRIES + 1
    return _finish(piece, music, chk, size)


# =================================================================================================== hand-written pieces
def infer_technique(piece: dict) -> str:
    if any(s.get("matte") for s in piece.get("shots", [])):
        return "behind"
    if any((s.get("control") or {}).get("pass") == "line" for s in piece.get("shots", [])):
        return "traced"
    return "built_in"


def fill_type_timing(piece: dict, music: dict | None) -> None:
    """the runner fills duration / fps / bpm / beats / downbeats into the type spec; do it here so the piece is self-contained."""
    t = piece.setdefault("type", {})
    t.setdefault("fps", piece.get("fps", FPS))
    t.setdefault("bar", 4)
    if music:
        t.setdefault("duration", _r(music["cut"], 4))
        t.setdefault("bpm", _r(music["bpm"], 3))
        t.setdefault("beats", [_r(b, 4) for b in music["beats"]])
        t.setdefault("downbeats", [_r(d, 4) for d in music["downbeats"]])


def finalize_piece(piece: dict, music: dict | None = None) -> dict:
    """add layout / sizes / technique / quality / brief to a P*.json dict (kept when present)."""
    p = copy.deepcopy(piece)
    p.setdefault("fps", FPS)
    p.setdefault("id", "PIECE")
    p.setdefault("quality", "final")
    p.setdefault("format", format_of(p))
    p.setdefault("sizes", sizes_for(p["quality"], p["format"]))
    p.setdefault("technique", infer_technique(p))
    p.setdefault("brief", {})
    p.setdefault("compose", {"size": p["sizes"]["compose"], "crossfade": 0.5})
    if not p.get("shots"):
        raise ValueError("piece has no shots")
    if len(p["shots"]) > MAX_SHOTS:
        raise ValueError(f"piece has {len(p['shots'])} shots; the app graph has {MAX_SHOTS} slots")
    fill_type_timing(p, music)
    p["layout"] = layout(p)
    return p


def piece_json(piece: dict) -> str:
    return json.dumps(piece, indent=1, sort_keys=True, ensure_ascii=False)


def piece_hash(piece: dict) -> str:
    return hashlib.sha256(json.dumps(piece, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
