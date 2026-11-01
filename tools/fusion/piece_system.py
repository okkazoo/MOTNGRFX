"""piece_system: ONE seeded SYSTEM per piece, drawn at the concept step (seed = the run seed) and held over every word of the design (diversity round 2026-10-29,
`16_app/DIVERSITY_AUDIT.md`). Pure functions: no ComfyUI, no network, no Claude.

Why: the open models (Qwen3.8-27B, Gemma 4 31B) kept landing on one value per field (lines in, lanes, one font, yaw +-18, beat_orbit ...) because of enum order, "default" wording in the docs,
copied example values and hard code defaults. Randomising per WORD would bring back the five-modules / five-faces / five-io-styles mess, so the variety is PER PIECE:

    choose(seed, mood, bpm, look, earlier)  -> the system dict: tone, module (+ optional second of the same family), look, io pair (form_in / form_out with one alternate each, entry sides),
                                               font shortlist (3-4 liked faces, random_system.draw_fonts), CUTS (2-3 cut kinds + a style each from the full menu, random_system.draw_cuts),
                                               palette family (the footage palette of the stills; the TYPE palette comes from the first still, still_palette.py), the held type camera
                                               (stable_type.py: centred, still for >= 1 beat), yaw band from the tempo, footage camera moves, photographic style
    for_piece(orig, ana, seed)              -> (system, explicit): the system a piece carries in plan.piece_system (explicit = True), else one computed from its seed / mood / bpm
    prompt_text(system, ...)                -> the design prompt section "YOUR SYSTEM FOR THIS PIECE"
    fold_type(t_in, system, fixes)          -> camera / camera_params / palette of a RAW reply type object folded into the system (outliers only)
    fold_words(words, system, fixes, ...)   -> module / form_in / form_out / font / yaw of RAW reply words: a value INSIDE the system is the model's and stays, an outlier is folded in
    beat_menu(bpm, ...)                     -> form_s / exit_s as whole beats mapped to seconds (the floors 0.45 / 0.5 s stop being the modal answer)
    photo_style(seed, tone) / frame_words   -> the stills' photographic style (replaces the fixed "cinematic 35mm film still, shallow depth of field, tall vertical frame")
    default_axes(font_file)                 -> the font file's own default instance (fvar defaults), e.g. Anybody wdth 100 / wght 100 = thin

The choices are tied to the concept MOOD through a small mood -> tone table (soft / tense / driven / bright) and per-tone weight tables, not uniform noise; `earlier` (the systems of the
earlier pieces of a batch) lowers the weight of what was already used so a batch spreads. Legibility stays the user's rule: only the modules the legibility fold keeps are ever drawn
(design_defaults.AVOID_MODULES are folded away by the code anyway), and the faces drawn exclude the hard-to-read ones.
"""
from __future__ import annotations

import random
import re
import zlib

VERSION = 1
TONES = ("soft", "tense", "driven", "bright")

# ---------------------------------------------------------------------------------------------------------------------------------------------- mood -> tone
MOOD_TONE: dict[str, str] = {}
for _tone, _ws in {
    "soft": "tender dreamy hopeful nostalgic lonely calm warm serene wistful bittersweet curious wonder gentle luxury quiet slow filmic lullaby",
    "tense": "eerie tense solemn dark noir haunted fierce moody ominous brooding cold bleak",
    "driven": "defiant restless kinetic hype techno electric determined triumphant energetic aggressive urgent fast",
    "bright": "joyful playful uplifting free happy sunny cheerful warmth bouncy festive",
}.items():
    for _w in _ws.split():
        MOOD_TONE[_w] = _tone
DEFAULT_BPM = {"soft": 76, "tense": 92, "bright": 106, "driven": 124}


def tone_of(mood: str = "", bpm=None) -> str:
    """soft | tense | driven | bright from the mood word(s) (the first known word wins), else from the tempo (< 85 soft, > 112 driven, else bright)."""
    for t in re.findall(r"[a-z]+", str(mood or "").lower()):
        if t in MOOD_TONE:
            return MOOD_TONE[t]
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return "bright"
    return "soft" if b < 85 else ("driven" if b > 112 else "bright")


# ---------------------------------------------------------------------------------------------------------------------------------------------- helpers
def _rng(seed, field: str, tone: str = "") -> random.Random:
    return random.Random(f"psys:{int(seed)}:{tone}:{field}")


def _wpick(rng: random.Random, items, weights):
    items, weights = list(items), [max(float(w), 0.0) for w in weights]
    tot = sum(weights)
    if not items:
        return None
    if tot <= 0:
        return items[rng.randrange(len(items))]
    x = rng.random() * tot
    for it, w in zip(items, weights):
        x -= w
        if x <= 0:
            return it
    return items[-1]


def _lerp(a, b, u):
    return a + (b - a) * u


def _used(earlier, *path):
    """values at `path` of the earlier systems (a list / scalar flattened)."""
    out = []
    for s in earlier or []:
        v = s if isinstance(s, dict) else {}
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (list, tuple)):
            out += list(v)
        elif v is not None:
            out.append(v)
    return out


REPEAT_PENALTY = 0.3          # weight multiplier of a value that an earlier piece of the batch already used


# ---------------------------------------------------------------------------------------------------------------------------------------------- modules
# only the modules the legibility fold keeps = every module of the 24 that design_defaults.AVOID_MODULES does not fold away (echo / spray, two-style, 3D texture-first: a draw of one of
# them would be undone). Derived from the fold, not a fixed list: a module the fold learns to keep is drawn from then on. Weights per mood tone: no module is first for every tone.
from tools.fusion import design_defaults as _DD  # noqa: E402
from tools.fusion import effects_menu as EM  # noqa: E402
from tools.fusion import random_system as RS  # noqa: E402
from tools.fusion import still_palette as SP  # noqa: E402
FAMILY = {"strand": ("lanes", "pipework", "circuit", "voxel_orbit", "voxel_roll", "tangent_spray"), "slab": ("slats", "gradient_fade", "staircase", "beat_bars", "spectrum", "fly_into")}
MODULE_WEIGHTS = {
    "soft": {"slats": 2.5, "gradient_fade": 2.5, "pipework": 2, "lanes": 1.5, "staircase": 1, "circuit": 1, "beat_bars": 0.4, "particles": 2.5, "voxel_orbit": 1.5, "voxel_roll": 1, "tangent_spray": 1.5, "spectrum": 1, "fly_into": 1},
    "tense": {"circuit": 2.5, "pipework": 2, "slats": 2, "staircase": 1.5, "lanes": 1.2, "gradient_fade": 0.8, "beat_bars": 0.6, "tangent_spray": 2.5, "particles": 2, "voxel_orbit": 2, "voxel_roll": 1.5, "spectrum": 1.5, "fly_into": 1.5},
    "driven": {"slats": 2.5, "beat_bars": 1.5, "staircase": 2.5, "circuit": 1.5, "pipework": 1, "lanes": 1, "gradient_fade": 0.8, "spectrum": 2.5, "voxel_roll": 2.5, "particles": 2, "tangent_spray": 2, "voxel_orbit": 1.5, "fly_into": 1.5},
    "bright": {"slats": 2.5, "beat_bars": 1, "gradient_fade": 2, "staircase": 1.5, "pipework": 1.5, "lanes": 1.5, "circuit": 0.8, "particles": 2.5, "voxel_orbit": 2, "voxel_roll": 1.5, "fly_into": 2, "spectrum": 1.5, "tangent_spray": 1},
}
ALL_MODULES = ("lanes", "pipework", "circuit", "depth_echo", "hollow_tubes", "topo_tower", "hinge_unfold", "xray_unfold", "candy_orbit", "depth_tunnel", "beat_bars", "bold_thin_echo", "gradient_fade",
               "scale_clash", "slats", "solid_hollow", "staircase", "spectrum", "tangent_spray", "fly_into", "outflow", "voxel_orbit", "voxel_roll", "particles")      # the 24 word modules (qwen_piece.WORD_MODULES)
KEPT_MODULES = tuple(m for m in ALL_MODULES if m not in _DD.AVOID_MODULES)          # 2026-10-30: the 7 reading-first (lanes pipework circuit beat_bars gradient_fade slats staircase) + the 7 effect modules (particles tangent_spray outflow voxel_orbit voxel_roll spectrum fly_into) = 14 of the 24
INVERSE_P = 0.5               # chance that a piece draws one inverse-type word on a plain ground (user 2026-10-30: more inverse imagery inside the texts)
# 2026-11-01 (user, "always carry the expressive fonts ... varied modules"; design_defaults.MAX_MODULES 2 -> 4): a piece draws min(number of its words, MAX_MODULES) DISTINCT modules, word i takes
# modules[i % n] (system["word_modules"]). Was ONE module + a 40 % (MODULE_B_P) sibling of the same family, so R18's three words were all `particles`.


def family_of(module: str) -> str | None:
    return next((f for f, ms in FAMILY.items() if module in ms), None)


def module_count(n_words=None) -> int:
    """how many distinct modules a piece of `n_words` words draws: min(n_words, design_defaults.MAX_MODULES); the cap when the word count is not known yet."""
    cap = max(1, int(_DD.MAX_MODULES))
    try:
        n = int(n_words)
    except (TypeError, ValueError):
        return cap
    return cap if n <= 0 else min(n, cap)


def draw_modules(seed, tone: str, look: str | None = None, earlier=(), n_words=None, pinned=()) -> list[str]:
    """the piece's modules, seeded: `pinned` (the user's own picks, kept as written and first) then weighted draws WITHOUT replacement from the kept modules (the tone weights, a module of an earlier
    piece of the batch drawn less, `look` strand | slab limits the pool to that family) until there are module_count(n_words). The first draw uses the same rng as before, so the lead module of a seed
    is unchanged. A module repeats on a word only when the words outnumber the modules (the cap) or the pool runs out."""
    n = module_count(n_words)
    used_m = set(_used(earlier, "modules"))
    cand = [m for m in KEPT_MODULES if look not in FAMILY or m in FAMILY[look]]

    def wt(m):
        return MODULE_WEIGHTS[tone].get(m, 1.0) * (REPEAT_PENALTY if m in used_m else 1.0)
    out = [m for m in dict.fromkeys(pinned or ()) if m in ALL_MODULES and m not in _DD.BANNED_MODULES][:max(1, int(_DD.MAX_MODULES))]
    if not out:
        out = [_wpick(_rng(seed, "module", tone), cand, [wt(m) for m in cand])]
    more = _rng(seed, "module_more", tone)
    while len(out) < n:
        pool = [m for m in cand if m not in out]
        if not pool:
            break
        out.append(_wpick(more, pool, [wt(m) for m in pool]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- io styles
# (form_in, form_in alternate, form_out, form_out alternate, weight): the two styles per direction are the piece's own (design_defaults.MAX_FORMS = 2 per direction). `lines` carries a side
# (`entry`), the others do not; flicker / wipe carry no strands (the next gap becomes a cut), so every row keeps some strand-carrying style.
IO_ROWS = {
    "soft": [("blobs", "wipe", "blobs", "particles", 3), ("wipe", "lines", "wipe", "blobs", 2.5), ("particles", "blobs", "blobs", "wipe", 2.5), ("lines", "blobs", "blobs", "wipe", 1.5),
             ("blobs", "particles", "particles", "blobs", 2), ("wipe", "particles", "blobs", "particles", 2)],
    "tense": [("flicker", "wipe", "wipe", "flicker", 1.2), ("particles", "flicker", "flicker", "blobs", 2.5), ("lines", "flicker", "wipe", "flicker", 1.5), ("wipe", "flicker", "particles", "wipe", 2),
              ("blobs", "flicker", "flicker", "particles", 2), ("lines", "particles", "wipe", "blobs", 1.5)],
    "driven": [("flicker", "particles", "flicker", "wipe", 1.2), ("particles", "lines", "particles", "flicker", 2.5), ("wipe", "blobs", "flicker", "blobs", 2), ("lines", "flicker", "particles", "flicker", 1.5),
               ("blobs", "wipe", "wipe", "particles", 2), ("particles", "flicker", "wipe", "blobs", 2)],
    "bright": [("blobs", "particles", "particles", "blobs", 3), ("particles", "wipe", "blobs", "wipe", 2.5), ("wipe", "blobs", "blobs", "particles", 2), ("lines", "blobs", "particles", "blobs", 1.5),
               ("blobs", "lines", "wipe", "particles", 1.5), ("particles", "blobs", "blobs", "wipe", 2.5)],
}
_SPECKLE_SWAP = ("wipe", "flicker", "lines")      # 2026-11-02: `blobs` is banned with `particles` (design_defaults.SPECKLE_IO), so only these three io styles are drawn; rotated so none takes over the first slot (tests/test_diversity.py)


def _no_speckle_rows(rows, off=0):
    """2026-11-01 (user, R18 / R19, no dot-cloud letters at any stage): a `particles` / `blobs` io style in a row becomes the first of wipe / flicker / lines the row's pair does not already hold."""
    out, k = [], 0
    for r in rows:
        r = list(r)
        for a, b in ((0, 1), (2, 3)):
            for j, o in ((a, b), (b, a)):
                if r[j] in _DD.SPECKLE_IO:
                    cyc = [_SPECKLE_SWAP[(k + q) % len(_SPECKLE_SWAP)] for q in range(len(_SPECKLE_SWAP))]      # rotate, so no one style takes over every replaced slot
                    r[j] = next((x for x in cyc if x != r[o] and x not in _DD.SPECKLE_IO), "lines")
                    k += 1
        out.append(tuple(_align_strands(r, len(out) + off)))
    return out


def _align_strands(r, k):
    """with `blobs` / `particles` gone `lines` is the only io style that carries strands, and a word-to-word transition needs strands on BOTH sides of the gap (a flicker / wipe side turns the gap into a cut). Words alternate the
    two styles of each direction (word i takes [i % 2]), so a row keeps one gap strand-carrying on both sides: word 0 leaves on form_out[0] and word 1 enters on form_in[1] (a row that already carries strands on the
    next gap, form_out[1] / form_in[0], is left as it is). The pair of a direction stays two different styles."""
    a_out, a_in, b_out, b_in = 2, 1, 3, 0
    if (r[a_out] == "lines" and r[a_in] == "lines") or (r[b_out] == "lines" and r[b_in] == "lines"):
        return r
    r[a_out] = r[a_in] = "lines"                                       # the first gap (never `lines` as the FIRST entry: tests/test_diversity.py)
    for j in (b_out, b_in):                                     # the other slot of each pair differs from the `lines` just set: a side-less style, alternated row by row so neither takes over
        r[j] = ("wipe", "flicker")[(k + j + 1) % 2]
    return r


_ROW_PARITY = {"soft": 1, "tense": 0, "driven": 0, "bright": 0}      # which side-less style (wipe / flicker) each tone's rows start from: tuned so neither takes over the first entry (tests/test_diversity.py)
IO_ROWS = {k: _no_speckle_rows(v, _ROW_PARITY[k]) for k, v in IO_ROWS.items()}
ENTRY_ORDER = ("left", "right", "down", "in", "out", "up")        # the side a `lines` word comes from; "up" (the module default) is last in the rotation


# ---------------------------------------------------------------------------------------------------------------------------------------------- fonts
# roles of the liked fonts, read from mograph/fonts/liked.json at call time. EXPRESSIVE roles carry the key word; READER roles carry the rest.
EXPRESSIVE_ROLES = ("pixel", "ornamental", "decon", "blackletter")
EXPR_ROLE_WEIGHTS = {"soft": {"ornamental": 3, "blackletter": 2, "pixel": 2, "decon": 1}, "tense": {"blackletter": 3, "decon": 2, "pixel": 2, "ornamental": 1},
                     "driven": {"pixel": 3, "decon": 2, "ornamental": 2, "blackletter": 1}, "bright": {"ornamental": 3, "pixel": 3, "decon": 1, "blackletter": 1}}
READER_ROLE_WEIGHTS = {"soft": {"heavy": 2, "tall": 2, "fine": 2, "text": 2, "mono": 1}, "tense": {"tall": 3, "heavy": 2, "mono": 2, "text": 2, "fine": 1},
                       "driven": {"heavy": 3, "tall": 3, "text": 2, "mono": 1, "fine": 1}, "bright": {"heavy": 3, "text": 2, "tall": 2, "fine": 1, "mono": 1}}
ANYBODY = "Anybody[wdth,wght].ttf"
ANYBODY_P = {"soft": 0.45, "tense": 0.4, "driven": 0.55, "bright": 0.55}      # the user's liked clean reader: offered in about half the shortlists, never forced
SHORTLIST = (8, 11)           # faces in a shortlist (lo, hi); the design picks three of them


def _font_pool():
    from mograph import fonts_liked as fl
    from tools.fusion import design_defaults as DD
    skip = set(DD.MAIN_HOLD_RISKY) | {"LibreBarcode39Text"}      # nothing the legibility screen would swap away again
    return [e for e in fl.LIKED if e["key"] not in skip]


def _pick_fonts(seed, tone, earlier, texts=()) -> list[str]:
    """the piece's 3-4 faces (random_system.draw_fonts; Anybody is in the pool at its normal weight, never forced)."""
    return RS.draw_fonts(seed, tone, earlier, texts=texts)


def _pick_fonts_old(seed, tone, earlier) -> list[str]:                  # the 2026-10-29 shortlist of 8-11 faces (kept for the diversity audit's comparison)
    pool = [e for e in _font_pool() if e["file"] != ANYBODY]            # Anybody is offered by its own draw below, never by the role weights
    used = set(_used(earlier, "fonts"))
    rng = _rng(seed, "fonts", tone)

    def w(e):
        return REPEAT_PENALTY if e["file"] in used else 1.0
    n = rng.randint(*SHORTLIST)
    want_anybody = rng.random() < ANYBODY_P[tone]
    n_take = n - 1 if want_anybody else n
    chosen: list[dict] = []

    def take(role_w, k):
        for _ in range(k):
            cand = [e for e in pool if e not in chosen and e["role"] in role_w]
            if not cand:
                return
            chosen.append(_wpick(rng, cand, [role_w[e["role"]] * w(e) for e in cand]))
    take(EXPR_ROLE_WEIGHTS[tone], 2)                                   # two characterful faces, drawn by role weight (different roles first)
    if len(chosen) > 1 and chosen[0]["role"] == chosen[1]["role"]:
        other = [e for e in pool if e["role"] in EXPRESSIVE_ROLES and e["role"] != chosen[0]["role"] and e not in chosen]
        if other:
            chosen[1] = _wpick(rng, other, [w(e) for e in other])
    take(READER_ROLE_WEIGHTS[tone], n_take - len(chosen))
    files = [e["file"] for e in chosen] + ([ANYBODY] if want_anybody else [])
    rng.shuffle(files)
    return files


def role_of_file(f: str) -> str:
    from mograph import fonts_liked as fl
    return fl.role_of(f)


def three_faces(system: dict) -> list[str]:
    """three faces of the shortlist for the rules' own plan: a heavy / tall reader, the expressive one, and a supporting face of a third role (the design picks its own three from the same list)."""
    from mograph import fonts_liked as fl
    fs = list(system["fonts"])
    roles = {f: fl.role_of(f) for f in fs}
    exp = [f for f in fs if roles[f] in EXPRESSIVE_ROLES]
    head = [f for f in fs if roles[f] in ("heavy", "tall")]
    a = head[0] if head else next((f for f in fs if f not in exp), fs[0])
    b = exp[0] if exp else next((f for f in fs if f != a), a)
    c = next((f for f in fs if f not in (a, b) and roles[f] not in (roles[a], roles[b])), next((f for f in fs if f not in (a, b)), a))
    return [a, b, c]


def default_axes(font_file: str) -> dict:
    """the font file's own DEFAULT instance as {axis tag: value} (the fvar defaults; Anybody = wdth 100, wght 100, the thin instance the user likes). {} for a static font.
    The registry's `axes` (Anybody wdth 120 / wght 900) is the heavy designed look; this is the font as its designer set it."""
    import os
    from PIL import ImageFont
    from mograph import font_registry as fr
    try:
        f = ImageFont.truetype(os.path.join(fr.FONT_DIR, font_file), 40)
        info = f.get_variation_axes()
    except (OSError, AttributeError, ValueError):
        return {}
    rev = {v: k for k, v in fr.AXIS_NAMES.items()}
    out = {}
    for a in info:
        nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
        out[rev.get(nm.lower(), nm.lower())] = float(a["default"])
    return {k: (int(v) if float(v).is_integer() else v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------------------------------------- palette families
# `footage` = the colour palette the STILL prompts state (one palette per piece, round-3 QC 2026-10-29: the liked pieces are teal + amber night and gold hour); the type palette above it matches it.
PALETTES = {
    "teal_amber_night": dict(bg=[6, 18, 24], fg=[232, 240, 236], accents=[[240, 160, 50], [40, 160, 160], [250, 210, 120]], note="pale ink on teal-black; amber and teal accents", footage="teal and amber night"),
    "gold_hour": dict(bg=[22, 14, 6], fg=[252, 240, 206], accents=[[250, 190, 70], [220, 130, 40], [255, 225, 150]], note="warm ivory ink on brown-black; gold and honey accents", footage="warm gold hour light"),
    "warm_cream_umber": dict(bg=[24, 16, 12], fg=[242, 232, 214], accents=[[214, 140, 70], [168, 84, 52], [240, 200, 120]], note="warm cream ink on umber black; amber and rust accents", footage="warm amber and brown"),
    "ice_on_ink": dict(bg=[8, 14, 26], fg=[226, 238, 250], accents=[[90, 190, 220], [60, 120, 200], [170, 225, 240]], note="ice-white ink on blue-black; cyan and steel-blue accents", footage="cold blue and steel grey"),
    "bone_oxblood": dict(bg=[28, 8, 12], fg=[238, 228, 220], accents=[[200, 40, 52], [240, 110, 90], [150, 24, 40]], note="bone ink on oxblood black; crimson and coral accents", footage="deep red and black"),
    "mint_forest": dict(bg=[6, 20, 14], fg=[222, 242, 230], accents=[[90, 200, 130], [200, 230, 80], [40, 140, 100]], note="pale mint ink on forest black; emerald and lime accents", footage="deep green and black"),
    "violet_dusk": dict(bg=[16, 10, 30], fg=[236, 230, 250], accents=[[170, 110, 240], [240, 100, 200], [110, 90, 220]], note="lavender-white ink on violet black; magenta and violet accents", footage="violet and magenta dusk"),
    "sodium_navy": dict(bg=[8, 16, 40], fg=[250, 240, 222], accents=[[255, 150, 40], [255, 200, 90], [230, 110, 30]], note="warm white ink on navy; sodium orange and gold accents", footage="navy blue with sodium orange"),
    "signal_red_black": dict(bg=[10, 10, 10], fg=[245, 245, 245], accents=[[235, 40, 30], [255, 120, 60], [250, 210, 200]], note="white ink on black with ONE hot red accent family", footage="black with a single hot red"),
    "acid_charcoal": dict(bg=[18, 20, 18], fg=[236, 240, 228], accents=[[200, 240, 40], [255, 90, 160], [70, 230, 200]], note="off-white ink on charcoal; acid lime, pink and aqua accents", footage="charcoal grey with acid lime"),
    "dust_rose": dict(bg=[30, 20, 24], fg=[246, 232, 230], accents=[[236, 150, 160], [200, 110, 130], [250, 200, 180]], note="blush ink on plum black; dusty rose and peach accents", footage="dusty rose and peach"),
}
PALETTE_WEIGHTS = {
    "soft": {"gold_hour": 3, "teal_amber_night": 2.5, "warm_cream_umber": 3, "dust_rose": 3, "mint_forest": 2, "ice_on_ink": 2, "violet_dusk": 1.5, "sodium_navy": 1, "bone_oxblood": 0.5, "signal_red_black": 0.3, "acid_charcoal": 0.3},
    "tense": {"teal_amber_night": 3.5, "gold_hour": 1, "ice_on_ink": 3, "bone_oxblood": 3, "violet_dusk": 2, "signal_red_black": 2, "mint_forest": 1.5, "sodium_navy": 1.5, "warm_cream_umber": 1, "acid_charcoal": 0.8, "dust_rose": 0.5},
    "driven": {"teal_amber_night": 3, "gold_hour": 1.5, "signal_red_black": 3, "acid_charcoal": 3, "sodium_navy": 2, "violet_dusk": 2, "bone_oxblood": 1.5, "ice_on_ink": 1.5, "warm_cream_umber": 1, "mint_forest": 1, "dust_rose": 0.5},
    "bright": {"gold_hour": 3.5, "teal_amber_night": 2, "sodium_navy": 3, "acid_charcoal": 2, "warm_cream_umber": 2, "mint_forest": 2, "dust_rose": 2, "violet_dusk": 1.5, "ice_on_ink": 1.5, "signal_red_black": 1, "bone_oxblood": 0.5},
}


def _spread(rgb) -> float:
    try:
        return float(max(rgb) - min(rgb))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------------------------------------------------------------------------- camera
# the type-world camera: reader-safe cameras only (the strongly moving ones belong on a single non-reading word, guide section 7). beat_orbit is ONE of three, not the default.
STABLE_CAMERA = "beat_orbit"          # the held camera: beat_orbit with read_dps 0 / bob 0 / elev 0 (no motion on a landed word); see stable_type.static_camera_params
CAMERA_WEIGHTS = {"soft": {"hold_drift": 3, "follow": 3, "beat_orbit": 2}, "tense": {"follow": 3, "beat_orbit": 3, "hold_drift": 2},
                  "driven": {"beat_orbit": 4, "follow": 3, "hold_drift": 0.5}, "bright": {"beat_orbit": 3, "follow": 3, "hold_drift": 1.5}}
# the footage cameras for the concept's shots (tools/fusion/shot_camera.MOVES vocabulary), a family per tone: the shots of one piece share a language
SHOT_MOVES = {
    "soft": ("push in", "pull back", "static", "tilt up", "handheld drift", "pan right", "crane up"),
    "tense": ("push in", "static", "pull back", "handheld drift", "tilt down", "pan left", "tracking"),
    "driven": ("tracking", "push in", "orbit", "crane up", "pan right", "handheld drift", "pull back"),
    "bright": ("orbit", "tracking", "crane up", "pull back", "pan left", "push in", "handheld drift"),
}


def motion(bpm, seed=0) -> dict:
    """tempo-driven motion numbers (guide section 14 as a function): yaw band, beat_orbit camera params. Slow = small, fast = big. A suggested value is drawn inside each band by the seed so two
    pieces of the same tempo differ; the band itself is what the fold enforces."""
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        b = 100.0
    u = min(max((b - 85.0) / 30.0, 0.0), 1.0)                       # 0 at <= 85 bpm (slow, tender), 1 at >= 115 bpm (kinetic)
    r = _rng(seed, "motion")
    yaw_lo, yaw_hi = round(_lerp(5, 14, u), 1), round(_lerp(10, 22, u), 1)
    bands = {"intro_deg": (round(_lerp(12, 35, u), 1), round(_lerp(25, 55, u), 1)), "fast_dps": (round(_lerp(8, 40, u), 1), round(_lerp(20, 60, u), 1)),
             "read_dps": (round(_lerp(3, 5, u), 1), round(_lerp(4, 8, u), 1)), "bob": (round(_lerp(0.4, 0.9, u), 2), round(_lerp(0.6, 1.2, u), 2))}
    sug = {k: round(r.uniform(*v), 1 if k != "bob" else 2) for k, v in bands.items()}
    sug["elev"] = round(r.uniform(1.0, 6.0), 1)                       # the audit: elev 2-3 in 94 % of pieces
    return dict(bpm=round(b, 1), u=round(u, 3), yaw=[yaw_lo, yaw_hi], start_sign=1 if r.random() < 0.5 else -1, bands={k: list(v) for k, v in bands.items()}, suggest=sug)


# ---------------------------------------------------------------------------------------------------------------------------------------------- photographic style (stills)
PHOTO = [   # (key, still phrase, tones that favour it): lens / film / light / medium; no "cinematic", no fixed depth of field
    ("35mm", "shot on 35mm film, natural grain, available light", {"tense": 3, "soft": 1, "driven": 1}),
    ("85mm", "85mm portrait lens, shallow depth of field, soft background", {"soft": 3, "bright": 1}),
    ("medium_format", "medium format photograph, fine grain, even tonal range", {"soft": 2, "bright": 2}),
    ("anamorphic", "anamorphic lens photograph, soft streaking lens flares, wide framing", {"tense": 2, "driven": 2}),
    ("documentary", "documentary photograph, available light, candid, slight motion blur", {"tense": 2, "driven": 2, "bright": 1}),
    ("instant", "instant film photograph, faded colours, soft vignette", {"soft": 2, "bright": 2}),
    ("large_format", "large format photograph, deep focus, crisp detail from front to back", {"tense": 1, "soft": 1, "bright": 1}),
    ("telephoto", "200mm telephoto photograph, compressed layers of depth", {"driven": 2, "bright": 1, "tense": 1}),
    ("wide_24", "24mm wide angle photograph, deep perspective, sharp throughout", {"driven": 3, "bright": 2}),
    ("expired_film", "expired colour film photograph, warm colour cast, heavy grain", {"soft": 2, "tense": 1}),
    ("flash", "direct on-camera flash photograph, harsh falloff, saturated colour", {"driven": 2, "bright": 1}),
    ("long_exposure", "long exposure photograph, smooth motion trails, steady tripod frame", {"soft": 2, "tense": 1}),
    ("snapshot", "candid snapshot, flat natural light, unposed", {"bright": 2, "soft": 1}),
    ("film_noir", "high-contrast fast film photograph, deep shadows, hard light", {"tense": 3}),
    ("pastel_film", "soft pastel colour negative photograph, airy light", {"soft": 2, "bright": 2}),
]
FRAME_WORDS = {"9:16": "vertical 9:16 composition", "16:9": "wide horizontal 16:9 composition"}


def photo_style(seed, tone: str = "bright", earlier=()) -> tuple[str, str]:
    """(key, phrase) of the piece's photographic style, tone-weighted, seeded."""
    used = set(_used(earlier, "photo_key"))
    r = _rng(seed, "photo", tone)
    w = [(base.get(tone, 0.4) if isinstance(base, dict) else 1.0) * (REPEAT_PENALTY if key in used else 1.0) for key, _p, base in PHOTO]
    pick = _wpick(r, PHOTO, w)
    return pick[0], pick[1]


def frame_words(fmt: str = "9:16") -> str:
    return FRAME_WORDS.get(fmt, FRAME_WORDS["9:16"])


# ---------------------------------------------------------------------------------------------------------------------------------------------- the draw
def choose(seed, mood: str = "", bpm=None, look: str | None = None, earlier=(), palette_mood: str | None = None, words=(), length_s=None, effects=None, pin_modules=()) -> dict:
    """the piece's system. seed = the run seed (deterministic: the same seed + mood + bpm + look gives the same system). `look` (strand | slab, the concept's own valid answer) keeps
    the module in that family; any other look (auto, mech ...) leaves the choice to the tone. `earlier` = the systems of the earlier pieces of the batch (their cut kinds are not drawn again by
    the next two pieces). `words` = the piece's lines (a face that cannot carry the longest word at the size floor is not drawn). `length_s` = the piece length: it sets how many faces are drawn (random_system.n_fonts). `effects` = names a request FORCES into the system (effects_menu.EFFECTS; unknown names are reported in `system.effects.unknown`). Modules: min(len(words), design_defaults.MAX_MODULES) distinct ones
    (the cap when `words` is empty), word i takes `word_modules[i]`; `pin_modules` = the user's own module picks, kept as written and first."""
    seed = int(seed or 0)
    tone = tone_of(mood, bpm)
    modules = draw_modules(seed, tone, look, earlier, len(words or ()), pin_modules)
    module = modules[0]
    ri = _rng(seed, "io", tone)
    used_io = set(_used(earlier, "io", "form_in"))
    rows = IO_ROWS[tone]
    row = _wpick(ri, rows, [r[4] * (REPEAT_PENALTY if r[0] in used_io else 1.0) for r in rows])
    io = dict(form_in=[row[0], row[1]], form_out=[row[2], row[3]])
    side0 = ri.randrange(5)                                           # the first side is any but "up" ("up" is the module default, not a design)
    entries = [ENTRY_ORDER[(side0 + k) % 5] for k in range(3)]
    primary = alt = STABLE_CAMERA                                        # 2026-10-30 (user): every word lands centred and holds STILL for >= 1 beat; the camera moves only in the form-in / exit (stable_type.py)
    mo = motion(bpm if bpm is not None else DEFAULT_BPM[tone], seed)
    ptone = tone_of(palette_mood, bpm) if palette_mood else tone               # the concept's scene states the palette drawn at the CUE mood: a changed mood must not change it
    rp = _rng(seed, "palette", ptone)
    used_p = set(_used(earlier, "palette", "family"))
    pf = list(PALETTES)
    pal = _wpick(rp, pf, [PALETTE_WEIGHTS[ptone].get(k, 0.4) * (REPEAT_PENALTY if k in used_p else 1.0) for k in pf])
    rs = _rng(seed, "shotmoves", tone)
    moves = list(SHOT_MOVES[tone])
    rs.shuffle(moves)
    pk, pp = photo_style(seed, tone, earlier)
    cuts = RS.draw_cuts(seed, tone, earlier)
    pool_ = None
    if any(m not in _DD.PIXEL_SAFE_MODULES for m in modules):                                   # QC v026: a pixel / bitmap face never meets an extruding / ray / spray module (design_defaults.pixel_problem)
        pool_ = [e for e in _font_pool() if e["key"] not in _DD.PIXEL_DOT_FACES]
    fonts = RS.draw_fonts(seed, tone, earlier, texts=words, pool=pool_, need_heavy=any(k.startswith("window") for k in cuts["kinds"]), length_s=length_s)       # a window kind needs a heavy face on the shortlist
    outline = RS.draw_outline(seed, tone)                                                       # the stand-out outline at some words / hits (legible_color's outline support, forced on)
    inverse = dict(plain_window=_rng(seed, "inverse", tone).random() < INVERSE_P)             # an inverse-type word (the picture only inside the letters, over a flat palette ground)
    out = dict(version=VERSION, seed=seed, mood=str(mood or ""), tone=tone, modules=modules, module=module, look=family_of(module), word_modules=[modules[i % len(modules)] for i in range(len(words or ()))],
                io=dict(io, entries=entries), fonts=fonts, expressive=[f for f in fonts if role_of_file(f) in EXPRESSIVE_ROLES],
                anybody_default=ANYBODY in fonts, camera=dict(type=primary, alt=alt, params=mo["suggest"], bands=mo["bands"], stable=True),
                motion=dict(bpm=mo["bpm"], yaw=mo["yaw"], start_sign=mo["start_sign"]), palette=dict(family=pal, **PALETTES[pal]),
                shot_moves=moves[:4], photo_key=pk, photo=pp, cuts=cuts, inverse=inverse, outline=outline,
                effects=EM.draw_effects(seed, earlier, effects))
    return EM.apply_effects(out, effects)


def _pin_cuts(system: dict, orig: dict) -> dict:
    """`plan.cut_kinds` (a piece author's own list) pins the cut kinds of the system instead of the seeded draw (a hand-written piece, a test)."""
    kinds = [k for k in ((orig.get("plan") or {}).get("cut_kinds") or []) if isinstance(k, str)]
    lays = [k for k in ((orig.get("plan") or {}).get("layouts_pin") or []) if isinstance(k, str)]
    if lays:
        system = dict(system, layouts=lays)                    # `plan.layouts_pin`: the layout of word i (stable_type.LAYOUTS) instead of the seeded draw
    pins = [k for k in ((orig.get("plan") or {}).get("cut_pins") or []) if isinstance(k, str) and k not in ("hard", "crossfade")]
    if pins and not kinds and isinstance(system.get("cuts"), dict):          # the user's camera moves (storyboard) are part of the piece's drawn kinds: the validator never swaps them away
        c = dict(system["cuts"])
        c["kinds"] = list(c.get("kinds") or []) + [k for k in dict.fromkeys(pins) if k not in (c.get("kinds") or [])]
        c["order"] = list(c.get("order") or c["kinds"]) + [k for k in c["kinds"] if k not in (c.get("order") or [])]
        system = dict(system, cuts=c)
    if not kinds:
        return system
    return dict(system, cuts=dict(version=RS.VERSION, kinds=kinds, styles={}, order=list(kinds), recent_excluded=[], pinned=True))


def for_piece(orig: dict, ana: dict | None = None, seed=None) -> tuple[dict, bool]:
    """(system, explicit): the system a piece carries (plan.piece_system, written by the planner from the concept's), else one computed from its seed, mood and tempo.
    `plan.cut_kinds` pins the cut kinds."""
    sysm, explicit = _for_piece(orig, ana, seed)
    return _pin_cuts(sysm, orig), explicit


def _for_piece(orig: dict, ana: dict | None = None, seed=None) -> tuple[dict, bool]:
    ps = (orig.get("plan") or {}).get("piece_system")
    if isinstance(ps, dict) and ps.get("module"):
        if not ps.get("cuts"):                                           # a system drawn before the random-cuts round: its cuts are drawn now from its own seed (never the fixed pivot-first fallback)
            ps = dict(ps, cuts=RS.draw_cuts(ps.get("seed", 0), ps.get("tone", "")), inverse=ps.get("inverse") or dict(plain_window=False))
        return ps, True
    t = orig.get("type") or {}
    sd = seed if seed is not None else (t.get("seed") if t.get("seed") is not None else (orig.get("plan") or {}).get("seed", 0))
    bpm = (ana or {}).get("bpm") or (orig.get("audio") or {}).get("bpm")
    mood = t.get("mood") or (orig.get("brief") or {}).get("mood") or ""
    if isinstance(mood, list):
        mood = " ".join(map(str, mood))
    length = (ana or {}).get("cut") or (orig.get("audio") or {}).get("cut") or t.get("duration")
    return choose(sd, mood, bpm, earlier=((orig.get("plan") or {}).get("earlier_systems") or ()), length_s=length), False


def seed_of(key: str) -> int:
    """a small stable seed from a string (a brief id) for callers that have no run seed."""
    return zlib.crc32(str(key).encode()) % 997


# ---------------------------------------------------------------------------------------------------------------------------------------------- timing in beats
def beat_menu(bpm, lo: float, hi: float, beats=(1, 2, 3, 4)) -> list[tuple[int, float]]:
    """[(beats, seconds)] of the whole beats that fit lo..hi s at this tempo (the floors 0.45 s / 0.5 s were the modal answer of both models; asking in beats maps to seconds here)."""
    try:
        iv = 60.0 / float(bpm)
    except (TypeError, ValueError, ZeroDivisionError):
        iv = 0.5
    return [(b, round(b * iv, 2)) for b in beats if lo - 1e-9 <= b * iv <= hi + 1e-9] or [(1, round(min(max(iv, lo), hi), 2))]


def beat_text(bpm, lo: float, hi: float, beats=(1, 2, 3, 4)) -> str:
    return ", ".join(f"{b} beat{'s' if b != 1 else ''} = {s:g} s" for b, s in beat_menu(bpm, lo, hi, beats))


# ---------------------------------------------------------------------------------------------------------------------------------------------- cuts + palette lines
CUT_WORDS = {"orbit:push": "orbit:push (the camera dives into the outgoing shot, the next opens behind it: a build)", "orbit:pullback": "orbit:pullback (the camera backs out, the next shot flies in past the lens: a release)",
             "orbit:fly": "orbit:fly (an S-curve with a bank through both shots: free, driving)", "depth_pivot": "depth_pivot (the outgoing shot pivots away, the next swings in over the top: a place change with a subject in both)",
             "window_other": "window_other (the held word's letters show the NEXT shot, then the hard cut)", "window_cut": "window_cut (the word is a window onto the next shot and the window grows until it fills the frame)",
             "window_swap": "window_swap (next shot inside the letters, swapping with the plate on the cut beat)", "window_zoom": "window_zoom (the camera scales INTO one letter of the holding word until the next shot fills the frame)",
             "hard": "hard (a hard cut on the beat)"}


def _cut_label(kind: str, style) -> str:
    base = CUT_WORDS.get(kind) or (f"{kind} (the type camera travels {kind.split(':')[1]} from the word before the cut to the one after it)" if kind.startswith("pan:") else kind)
    return base + (f", style {style}" if style else "")


def _cuts_line(s: dict) -> str:
    c = s.get("cuts") or {}
    kinds = c.get("kinds") or []
    if not kinds:
        return "- Shot transitions: choose from the menu below."
    st = c.get("styles") or {}
    return ("- Shot transitions drawn for THIS piece (the ONLY cut kinds you may write in compose.cuts; give each place change one of them, vary them along the piece, a style is part of the draw: `style` on the cut): "
            + "; ".join(_cut_label(k, st.get(k)) for k in kinds) + ". Anything else is replaced by one of these.")


def _outline_line(s: dict) -> str:
    o = s.get("outline") or {}
    mode = o.get("mode", "none")
    return ("- Outline (the code draws it, you do not write it): " + {"words": "a thin outline in the opposite luminance stands the type out on some of the words, for their whole hold",
                                                                 "hits": "a thin outline pulses for one beat as some words land",
                                                                 }.get(mode, "only where the footage makes a word hard to read") + ". Not on every word.")


def palette_line(s: dict) -> str:
    pal = s["palette"]
    if pal.get("source") == "still":
        cs = pal["colors"]
        return (f"- Palette (taken from this piece's own first still, {len(cs)} colours; type, plain grounds and strands use ONLY these): " + ", ".join(str(list(c)) for c in cs)
                + f". Ink {pal['fg']} on ground {pal['bg']} (contrast {SP.contrast(pal['bg'], pal['fg']):.1f}:1); accents {pal['accents']}. Write them as type.palette bg / fg / accents; the code snaps any other colour to the nearest of these.")
    return (f"- Palette family `{pal['family']}`: {pal['note']}; the footage and every still state ONE palette: {pal['footage']}. bg about {pal['bg']}, fg about {pal['fg']}, accents about {pal['accents']}. "
            f"Take the darkest/lightest tones from the footage, but keep this temperature and give at least one accent of this hue family (not neutral grey).")


def with_still_palette(system: dict, still: dict | None) -> dict:
    """a copy of `system` whose palette is the one extracted from the piece's first still (still_palette.extract): bg / fg / accents / colours; the footage words stay. The still palette is
    what fold_type snaps a reply's type palette to. A falsy `still` returns the system unchanged."""
    import copy as _copy
    if not still or not still.get("colors"):
        return system
    out = _copy.deepcopy(system)
    old = out["palette"]
    tp = SP.type_palette(still)
    out["palette"] = dict(old, family="still", bg=tp["bg"], fg=tp["fg"], accents=tp["accents"], colors=[list(c) for c in still["colors"]], source="still", min_ratio=still.get("min_ratio", SP.MIN_RATIO),
                          note="colours extracted from the piece's own first still", still=dict(still))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- prompt text
def prompt_text(system: dict, fonts_text: str = "", bpm=None) -> str:
    """the section of the design prompt that hands the piece its system. `fonts_text` is the shortlist block (qwen_piece.fonts_block) when the caller prints it elsewhere."""
    s = system
    mod = s["modules"]
    io = s["io"]
    cam = s["camera"]
    mo = s["motion"]
    lines = [
        f"This system was drawn for THIS piece (mood '{s.get('mood') or s['tone']}', tone {s['tone']}, tempo {mo['bpm']:g} bpm) so that no two pieces look alike. Design INSIDE it: a choice of yours that is inside it stays as you wrote it, "
        f"an outlier is folded into it by the code and logged. It is one system for all words: vary the words by the pattern below, not by new styles.",
        (f"- Word modules (varied: a different one on each word, in this order, repeating only when the piece has more words than modules): {', '.join('`' + m + '`' for m in (s.get('word_modules') or mod))}" if len(mod) > 1
         else f"- Word module: `{mod[0]}` (one module for every word)") + f". Family of the first: {s['look']}. Do not use any other module.",
        f"- Form in / form out: form_in `{io['form_in'][0]}` (alternate `{io['form_in'][1]}`), form_out `{io['form_out'][0]}` (alternate `{io['form_out'][1]}`). Use the first on the first word and alternate between the two on the following words; "
        f"no other io styles. A `lines` word names its `entry` side from: {', '.join(io['entries'])} (never `up`, never the same side twice in a row).",
        f"- Faces: your faces come ONLY from this shortlist of {len(s['fonts'])} liked faces (drawn for this piece; use all of them if you can, one of them expressive on the key word; neighbouring words use contrasting faces): " + ", ".join(s["fonts"]) + ".",
        f"- Placement and camera (user rule: the text lands stable and centred): every word lands CENTRED in the frame (dx 0, dy 0, yaw 0, pitch 0) and HOLDS STILL for at least one full beat ({60.0 / mo['bpm']:.2f} s at {mo['bpm']:g} bpm) "
        f"before its exit; its form-in ends ON a beat. Type camera `{cam['type']}` held still: write camera_params read_dps 0, bob 0, elev 0 (motion lives only in the form-in and the exit). "
        f"Layout per word (the code draws it from: one line; stacked lines that break at spaces; rolled +90 / -90 degrees reading vertically; `grid` = the word split MID-WORD into rows of 2-4 letters that fill the frame, DR / EA / MI / NG, SEA / RCH / ING, STE / AM; `column` = one letter per row, S / T / E / A / M; "
        f"all centred inside the safe area, all letters one size). Never leave a single letter on a row of a split word unless EVERY row is one letter (S-T / E / A-M is wrong, S/T/E/A/M and STE / AM are right). The code writes the rows: keep your own `lines` at spaces and do not set roll yourself.",
        _cuts_line(s),
        _outline_line(s),
        EM.prompt_text(s),
        palette_line(s),
        f"- Footage camera moves of this piece's shots (concept): {', '.join(s['shot_moves'])}. Photographic style of its stills: {s['photo']}.",
    ]
    if s.get("anybody_default"):
        d = default_axes(ANYBODY)
        lines.append(f"- {ANYBODY} is on the shortlist as a CLEAN READER. At its DEFAULT instance (axes wdth {d.get('wdth', 100):g}, wght {d.get('wght', 100):g}: thin, even strokes) in plain flat white it reads cleanly "
                     f"and calmly; write `\"axes\": {{\"wdth\": {d.get('wdth', 100):g}, \"wght\": {d.get('wght', 100):g}}}` for that look. Optional: use it where a quiet reading word fits, not as a rule.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------------------------------- the fold (RAW reply objects)
def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def fold_type(t_in: dict, system: dict, fixes: list) -> int:
    """camera, camera_params and palette of a RAW reply `type` object (in place). A camera of the system stays; another one becomes the system's type camera. beat_orbit parameters outside the tempo
    bands are clamped to them. A palette whose bg and fg are both neutral (the near-white-on-near-black default) takes the family's; neutral accents take the family's. -> fields changed."""
    n = 0
    cams = {system["camera"]["type"]}                       # the piece's camera is its primary (the alternate stays in the dict for a caller that wants a second choice): r9 showed 7 of 8 models keep beat_orbit when it is merely allowed
    cam = t_in.get("camera")
    if isinstance(cam, str) and cam in {"beat_orbit", "hold_drift", "follow", "orbit_even", "spin_read_spin", "push_through", "tilt_to_plan", "graze", "ease_reversal", "pullback_reveal"} and cam not in cams:
        new = system["camera"]["type"]
        fixes.append(f"type.camera {cam!r} -> {new!r} (the piece's system: {', '.join(sorted(cams))})")
        t_in["camera"] = new
        if new != "beat_orbit":
            t_in.pop("camera_params", None)
        n += 1
    cp = t_in.get("camera_params")
    if isinstance(cp, dict) and t_in.get("camera", "beat_orbit") == "beat_orbit":
        for k, (lo, hi) in system["camera"]["bands"].items():
            v = _num(cp.get(k))
            if v is None:
                continue
            slack = 0.15 * (hi - lo)
            y = min(max(v, lo - slack), hi + slack)
            if abs(y - v) > 1e-9:
                fixes.append(f"type.camera_params.{k} {v:g} -> {y:g} (the tempo band of this piece: {lo:g}-{hi:g})")
                cp[k] = round(y, 2)
                n += 1
    pal = t_in.get("palette")
    fam = system["palette"]
    if fam.get("source") == "still":                          # the piece's own still decides the colours (a limited palette discerned from the imagery): every colour of the reply snaps to it
        if not isinstance(pal, dict):
            pal = t_in["palette"] = {}
        n += SP.fold_palette(pal, dict(fam, colors=fam["colors"], bg=fam["bg"], fg=fam["fg"]), fixes)
    elif isinstance(pal, dict):
        bg, fg = pal.get("bg"), pal.get("fg")
        if isinstance(bg, list) and isinstance(fg, list) and len(bg) == 3 and len(fg) == 3 and _spread(bg) <= 14 and _spread(fg) <= 14:
            fixes.append(f"type.palette bg/fg both neutral {bg}/{fg} -> {fam['bg']}/{fam['fg']} (palette family {fam['family']})")
            pal["bg"], pal["fg"] = list(fam["bg"]), list(fam["fg"])
            n += 1
        acc = pal.get("accents")
        if isinstance(acc, list) and all(isinstance(a, list) and len(a) == 3 and _spread(a) <= 20 for a in acc):
            fixes.append(f"type.palette.accents neutral -> the family's ({fam['family']})")
            pal["accents"] = [list(a) for a in fam["accents"]]
            n += 1
    return n


def _file_of(name):
    from mograph import fonts_liked as fl
    e = fl.lookup(name) if name else None
    return e["file"] if e else None


def fold_words(words: list, system: dict, fixes: list, protected=(), seed=0) -> int:
    """module / form_in / form_out / font / placement.yaw of RAW reply words (dicts, in place). Inside the system = the model's and kept; an outlier is folded in, each change one fix string.
      module    not one of the system's -> the system's module of the same family if it has one, else the first
      form_*    not one of the system's two styles for that direction -> the first/second by position (alternating), so the whole piece follows its pair; `entry` is dropped when the new style is not `lines`
      font      not on the shortlist -> a shortlist face of the same role (not the neighbour's), else of a contrasting role; its axes go with it
      yaw       magnitude above the tempo band -> clamped to it (sign kept); 0 / absent stays
    Words in `protected` (window words) keep their face and io styles (the window's) but follow the module and the yaw band. -> number of fields changed."""
    from mograph import fonts_liked as fl
    n = 0
    mods = system["modules"]
    short = system["fonts"]
    yaw_lo, yaw_hi = system["motion"]["yaw"]
    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        m = w.get("module")
        if isinstance(m, str) and m in ALL_MODULES and m not in mods:
            fam = family_of(m)
            new = next((x for x in mods if family_of(x) == fam), mods[0])
            fixes.append(f"type.words[{i}].module {m} -> {new} (the piece's system: {', '.join(mods)})")
            w["module"] = new
            n += 1
        for k in ("form_in", "form_out"):
            if i in protected:                                            # a window word keeps its face and its io (the window's), but follows the module and the yaw band
                break
            v = w.get(k)
            allowed = system["io"][k]
            if isinstance(v, str) and v in {"lines", "blobs", "flicker", "particles", "wipe"} and v not in allowed:
                new = allowed[i % len(allowed)]
                fixes.append(f"type.words[{i}].{k} {v!r} -> {new!r} (the piece's system: {', '.join(allowed)})")
                w[k] = new
                if k == "form_in" and new != "lines":
                    w.pop("entry", None)
                n += 1
        f = w.get("font")
        if isinstance(f, str) and f and i not in protected:
            fe = _file_of(f)
            if fe and fe not in short:
                role = fl.role_of(fe)
                nb = {_file_of(words[j].get("font")) for j in (i - 1, i + 1) if 0 <= j < len(words) and isinstance(words[j], dict)}
                same = [x for x in short if fl.role_of(x) == role and x not in nb]
                other = [x for x in short if x not in nb and (fl.contrast_axes(role, fl.role_of(x)) or role in EXPRESSIVE_ROLES)]
                pool = same or other or list(short)
                new = random.Random(f"fold:{seed}:{i}").choice(sorted(pool))
                fixes.append(f"type.words[{i}].font {fe} -> {new} (not on the piece's shortlist of {len(short)} faces)")
                w["font"] = new
                if w.pop("axes", None) is not None:
                    fixes.append(f"type.words[{i}].axes dropped (they belonged to {fe})")
                n += 1
        pl = w.get("placement")
        if isinstance(pl, dict):
            y = _num(pl.get("yaw"))
            if y is not None and abs(y) > yaw_hi + 1e-9:
                new = round((1 if y > 0 else -1) * random.Random(f"yaw:{seed}:{i}").uniform(yaw_lo, yaw_hi), 1)       # a magnitude inside the band, drawn per word (not one value for all)
                fixes.append(f"type.words[{i}].placement.yaw {y:g} -> {new:g} (the tempo band of this piece: {yaw_lo:g}-{yaw_hi:g})")
                pl["yaw"] = new
                n += 1
    return n
