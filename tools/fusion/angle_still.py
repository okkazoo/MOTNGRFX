"""Same-subject NEW-ANGLE stills (idea 23_moss-angles, 2026-10-29). Pure functions, no ComfyUI, no models: the anchored edit prompt, the graph fragment for one angle
still, and the consistency scorer that picks the best of a seed sweep. `tests/test_angle_still.py` covers it; `16_app/ANGLE_STILLS.md` says how the app wires it.

What it is: a place whose still is `{"angle_of": "A", "framing": "close-up"}` is NOT a fresh text-to-image still. It is an EDIT of place A's still with the MOSS-Image
SFT + the `moss_image_edit_chinoll` LoRA (TextEncodeMOSSImageEdit, edit_format "Moss-Image-Edit", the reference as `images.image_0`, vae connected, same node for the
negative). The edit is always made from the ORIGINAL still (memory rule: edits darken when chained, never edit an edit). v002 finding: a bare "show the same X from
a close-up" loses the person; an ANCHORED prompt (the subject's clothing + colour + light restated, "change only the camera") keeps palette, clothes and kayak.
"""
from __future__ import annotations

import re

import numpy as np

# --------------------------------------------------------------------------------------------------------------------------------------------------- framings
# the camera clause per framing, subject-neutral (the subject is anchored separately). Keys are the app vocabulary; `FRAMING_ALIASES` maps the design / shot_camera
# words (tools/fusion/shot_camera.py FRAMINGS) onto them. An unknown framing is used verbatim as the camera clause.
FRAMING_CLAUSES = {
    "close-up": "a close-up, the camera close to the subject, framing the subject's head and shoulders and what they hold",
    "extreme close-up": "an extreme close-up on one detail of the subject, filling the frame",
    "low": "a very low angle just above the ground or water at the subject's side, looking up at the subject against the sky",
    "high": "a high angle looking down on the subject from above and slightly behind",
    "profile": "a side view in profile, the camera level with the subject, the whole subject visible",
    "top-down": "directly overhead, a top-down view of the subject and the ground or water around it",
    "wide": "a wide shot from further back, the subject small in the surroundings",
    "medium": "a medium shot at eye level, the subject framed from the waist or the middle up",
    "behind": "from directly behind the subject, level with it, looking the way the subject looks",
    "over-the-shoulder": "over the subject's shoulder, looking past it at what it faces",
}
FRAMING_ALIASES = {
    "closeup": "close-up", "close up": "close-up", "tight": "close-up", "ecu": "extreme close-up", "extreme close up": "extreme close-up", "macro": "extreme close-up",
    "low angle": "low", "low-angle": "low", "worm": "low", "high angle": "high", "high-angle": "high", "birds eye": "top-down", "bird's eye": "top-down", "overhead": "top-down",
    "top down": "top-down", "topdown": "top-down", "side": "profile", "side view": "profile", "establishing": "wide", "long": "wide", "mid": "medium", "waist": "medium",
    "rear": "behind", "back": "behind", "ots": "over-the-shoulder", "over the shoulder": "over-the-shoulder", "over-shoulder": "over-the-shoulder",
}
# framings where the subject's face is normally visible from the new camera; for these `hide_face` is the way to avoid an invented face
FACE_RISK = ("close-up", "extreme close-up", "low", "medium", "over-the-shoulder", "high")
FACE_HIDDEN = "The person is seen from behind or from the side, the face not visible, the face in shadow."
SECOND_REF = ("The second reference image shows the same person from another angle: keep the person's hair, build, clothes and skin identical to it; "
              "the first reference image is the scene and the light.")
KEEP_TAIL = "keep the same colours, light and mood, do not change the time of day, the weather or the colour grade."

_LIGHT_RE = re.compile(r"\b(light|lit|golden|warm|cool|cold|blue|teal|orange|amber|pink|pastel|dawn|sunrise|sunset|dusk|twilight|moonlit?|overcast|haze|hazy|mist|misty|fog|foggy|"
                       r"glow|glowing|shadow|shadows|neon|backlit|rim|palette|colou?rs?|hues?|tones?|grade|dark|bright|night|daylight|morning|evening)\b", re.I)
_STYLE_RE = re.compile(r"\b(film still|35 ?mm|depth of field|bokeh|lens|vertical frame|horizontal frame|photograph(?:y)?|photo|cinematic|kodak|grain|8k|4k|hd|"
                       r"shot from|shot on|camera|angle|framing|frame|close[- ]?up|wide shot|medium shot|low angle|high angle|top[- ]down|overhead|profile|view)\b", re.I)
_EDGE_PUNCT = " ,;:.\n\t"


def normalize_framing(framing) -> str:
    """the key of FRAMING_CLAUSES for a framing word ("Close up", "low angle", "overhead" ...); an unknown phrase comes back stripped and unchanged (used verbatim)."""
    f = re.sub(r"\s+", " ", str(framing or "").strip().lower().replace("_", " "))
    f = re.sub(r"\s+(shot|view)$", "", f)
    if f in FRAMING_CLAUSES:
        return f
    if f in FRAMING_ALIASES:
        return FRAMING_ALIASES[f]
    f2 = f.replace(" ", "-")
    if f2 in FRAMING_CLAUSES:
        return f2
    return str(framing or "").strip()


def framing_clause(framing) -> str:
    """the camera clause for a framing (the known clause, else the phrase itself)."""
    k = normalize_framing(framing)
    return FRAMING_CLAUSES.get(k, k)


# The framing a still LEADS with (regression 2026-10-29, draft v014 r02: all six stills were the same medium woman: the framing sat at the END of a prompt that began with the subject
# description and carried the scene's own "mid-to-wide shots", and an image model reads the front of a prompt first). One clause, no comma inside (split_still_prompt reads clauses).
STILL_LEADS = {
    "close-up": "tight close-up with the subject's face and shoulders filling the frame",
    "extreme close-up": "extreme close-up on one small detail filling the whole frame",
    "low": "very low camera angle on the ground looking steeply up at the subject",
    "high": "high camera angle looking down on the subject from above",
    "profile": "side profile view with the camera level with the subject",
    "top-down": "directly overhead top-down view",
    "wide": "wide establishing shot from far back with the subject small in a large space",
    "medium": "medium shot at eye level with the subject framed from the waist up",
    "behind": "view from directly behind the subject",
    "over-the-shoulder": "over-the-shoulder view looking past the subject at what it faces",
}
_FRAMING_WORDS = re.compile(
    r"\b(?:an? |the )?(?:(?:extreme |tight |very )?close[- ]?ups?|(?:mid|medium|wide|long|full|establishing)(?:[- ]to[- ](?:mid|medium|wide|close))*[- ]?(?:wide|medium|range)?[- ]?(?:shots?|views?|framing)"
    r"|mid[- ]to[- ]wide|low[- ]angle|high[- ]angle|top[- ]down|bird'?s[- ]eye)\b", re.I)


def strip_lead(prompt: str) -> str:
    """the prompt without a framing lead clause that lead_framing put in front."""
    p = str(prompt or "").strip()
    for v in STILL_LEADS.values():
        if p.lower().startswith(v.lower()):
            return p[len(v):].lstrip(" ,;")
    return p


def strip_framing(prompt: str) -> str:
    """the prompt without its own framing words ("mid-to-wide shots", "medium shot", "close-up" ...): a framing stated in the shared scene text fights the lead of the still."""
    p = strip_lead(prompt)
    p = _FRAMING_WORDS.sub("", p)
    p = re.sub(r"\s{2,}", " ", p)
    p = re.sub(r"(?:\s*[,;]\s*){2,}", ", ", p)
    p = re.sub(r"[,;]\s*(?=\.)", "", p)
    p = re.sub(r"\s+([,;.])", r"\1", p)
    return p.strip(" ,;")


def lead_framing(prompt: str, framing) -> str:
    """the still prompt with its framing FIRST: '<lead>, <prompt without its own framing words>'. An unknown framing phrase leads verbatim."""
    k = normalize_framing(framing)
    lead = STILL_LEADS.get(k, k)
    base = strip_framing(prompt)
    return f"{lead}, {base}" if lead and base else (lead or base)


_PERSON_RE = re.compile(r"\b(person|people|man|men|woman|women|girl|boy|child|children|kid|lady|guy|human|figure|runner|skater|dancer|swimmer|climber|paddler|rider|cyclist|"
                        r"biker|surfer|hiker|walker|artist|singer|musician|drummer|pianist|dj|athlete|model|portrait|face|he|she|his|her|they)\b", re.I)


def is_person(still_prompt: str) -> bool:
    """does the still prompt's subject look like a person (so the face-hidden wording applies)? Word list heuristic over the first two clauses of the prompt (no model)."""
    return bool(_PERSON_RE.search(", ".join(_clauses(still_prompt)[:2])))


def face_risk(framing) -> bool:
    """True when the new camera would normally see the person's face (the low-angle edit in v002 invented one): the app should default `hide_face` on."""
    return normalize_framing(framing) in FACE_RISK


# ------------------------------------------------------------------------------------------------------------------------------------ still prompt -> anchor
def _clauses(text: str) -> list[str]:
    return [c.strip(_EDGE_PUNCT) for c in re.split(r"[,;]|\s+-\s+", str(text or "")) if c.strip(_EDGE_PUNCT)]


def split_still_prompt(still_prompt: str) -> dict:
    """pull the pieces an anchored edit prompt needs out of a place's still prompt (the planner's `stills[i].prompt`: subject clause first, then the place and the light,
    then framing / style words). -> {"subject": first clause, "light": [clauses with light / colour words], "rest": [other clauses], "dropped": [framing + style clauses]}.
    Pure text heuristics (no model); a design that names the subject itself passes `subject=` to `build_edit_prompt` instead."""
    cl = _clauses(strip_lead(still_prompt))                       # a framing lead (lead_framing) is not the subject
    out = {"subject": "", "light": [], "rest": [], "dropped": []}
    for i, c in enumerate(cl):
        if i == 0:
            out["subject"] = c
        elif _STYLE_RE.search(c):
            out["dropped"].append(c)
        elif _LIGHT_RE.search(c):
            out["light"].append(c)
        else:
            out["rest"].append(c)
    return out


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


def _strip_article(s: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", s.strip(), flags=re.I)


def subject_anchor(still_prompt: str, subject: str | None = None) -> str:
    """the 'same subject' phrase: the design's own `subject` text when given, else the first clause of the still prompt."""
    s = (subject or "").strip(_EDGE_PUNCT) or split_still_prompt(still_prompt)["subject"]
    return _strip_article(_lower_first(s)) if s else "subject"


def palette_phrase(still_prompt: str, palette: str | None = None) -> str:
    """the colour + light phrase restated in the edit prompt: the given `palette` text, else the still prompt's light / colour clauses (at most 3), else ''."""
    if palette and palette.strip():
        return palette.strip(_EDGE_PUNCT)
    return ", ".join(split_still_prompt(still_prompt)["light"][:3])


def build_edit_prompt(still_prompt: str, framing, *, subject: str | None = None, palette: str | None = None, hide_face: bool | None = None,
                      second_ref: bool = False, keep: str | None = None, person: bool = True) -> str:
    """the ANCHORED edit prompt for one angle still (the v002 shape, which kept palette / clothes / kayak): "Change only the camera: <framing clause>. Show the same <subject>,
    as in the reference image; <palette phrase>; keep the same colours, light and mood ...".
    still_prompt: the source place's still prompt (subject clause + light words are taken from it).  framing: close-up | low | profile | top-down | wide | ... (aliases ok).
    subject / palette: override the extracted text (a design that has looked at the still can name the clothing: "dark long-sleeve top, red life vest, dark hair").
    keep: extra things that must not change ("the red kayak with dark-red paddle blades").  hide_face: add the face-hidden wording; None = on for the framings in FACE_RISK.
    person: False when the subject is not a person (no face line by default).  second_ref: the graph gives the model a second reference image of the person (`angle_graph(second_ref=...)`): add the line that says what it is for."""
    parts = [f"Change only the camera: {framing_clause(framing)}."]
    if subject and subject.strip():
        same = f"Show the same {subject_anchor(still_prompt, subject)}, seen in the reference image"
    else:                                                       # the first clause of a still prompt is a whole sentence ("A red kayak glides across a lake"): quote it
        same = f"Keep the same subject and scene as in the reference image ({subject_anchor(still_prompt)})"
    if keep and keep.strip():
        same += f", with {keep.strip(_EDGE_PUNCT)}"
    pal = palette_phrase(still_prompt, palette)
    if pal:
        same += f"; the same {pal}"
    parts.append(f"{same}; {KEEP_TAIL}")
    if hide_face is None:
        hide_face = person and face_risk(framing)
    if hide_face:
        parts.append(FACE_HIDDEN)
    if second_ref:
        parts.append(SECOND_REF)
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------------------------------------------------------------- graph
NEGATIVE = ("nsfw, worst quality, bad anatomy, jpeg artifact, text, letters, words, watermark, logo, caption, signature, signage, signs, typography, lettering, "
            "blurry, deformed, oversaturated, cartoon")
UNET = "moss_image_sft_bf16.safetensors"
CLIP = "qwen3.5_9b_bf16.safetensors"
VAE = "flux2-vae_small.safetensors"
LORA = "moss_image_edit_chinoll.safetensors"
EDIT_FORMAT = "Moss-Image-Edit"
STEPS, CFG, SAMPLER, SCHEDULER = 30, 4.0, "er_sde", "simple"        # the edit sampler (NOT the text-to-image 25 / 2.5); the app graph (build_graphs.add_angle_chain) reads these
DEFAULT_SIZE = (448, 832)                  # w, h of the probes (draft); the app uses the place's still size


def _ref_input(ref) -> tuple[dict | None, list]:
    """(node dict | None, link) for a reference: a str = LoadImage filename (a node is made), a [node_id, slot] list = an existing IMAGE output (no node)."""
    if isinstance(ref, (list, tuple)):
        return None, [str(ref[0]), int(ref[1])]
    return {"class_type": "LoadImage", "inputs": {"image": str(ref)}}, []


def angle_graph(ref, prompt: str, seeds, *, prefix: str, second_ref=None, size: tuple[int, int] = DEFAULT_SIZE, negative: str = NEGATIVE, steps: int = STEPS, cfg: float = CFG,
                sampler: str = SAMPLER, scheduler: str = SCHEDULER, lora_strength: float = 1.0, lora: str = LORA, unet: str = UNET, clip: str = CLIP, vae: str = VAE,
                id_prefix: str = "", loaders: dict | None = None) -> dict:
    """ComfyUI API-format graph (dict node_id -> node) for ONE angle still with one sampler branch per seed: the loaders, the LoRA, both text encodes, the empty latent and
    the reference are shared (ComfyUI runs them once), each seed adds KSampler -> VAEDecode -> SaveImage `<prefix>_s<seed>`.
    ref: LoadImage filename (str) OR an existing IMAGE link [node_id, slot] (the app passes place A's decoded still).  second_ref: same, the optional 2nd reference
    (image_1; the base is always `ref`, the original still).  loaders: {"unet": [id, 0], "clip": [id, 0], "vae": [id, 0]} existing links to reuse the app graph's loaders
    (then no loader nodes are added; the LoRA node is still added).  id_prefix: prepended to every node id so the fragment can be merged into a bigger graph.
    -> the graph; its SaveImage node ids are `seed_save_ids(seeds, id_prefix)`."""
    seeds = [int(s) for s in (seeds if isinstance(seeds, (list, tuple)) else [seeds])]
    if not seeds:
        raise ValueError("angle_graph needs at least one seed")
    p = id_prefix
    g: dict = {}
    if loaders:
        mdl, clp, vae_l = (list(loaders[k]) for k in ("unet", "clip", "vae"))
    else:
        g[p + "unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        g[p + "clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "moss_image", "device": "default"}}
        g[p + "vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
        mdl, clp, vae_l = [p + "unet", 0], [p + "clip", 0], [p + "vae", 0]
    g[p + "lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": mdl, "lora_name": lora, "strength_model": float(lora_strength)}}
    images: dict = {}
    for i, r in enumerate([ref] + ([second_ref] if second_ref is not None else [])):
        node, link = _ref_input(r)
        if node is not None:
            g[f"{p}ref{i}"] = node
            link = [f"{p}ref{i}", 0]
        images[f"images.image_{i}"] = link
    for name, text in (("pos", prompt), ("neg", negative)):
        g[p + name] = {"class_type": "TextEncodeMOSSImageEdit",
                       "inputs": {"clip": clp, "prompt": text, "edit_format": EDIT_FORMAT, "image_strength": 1.0, "vae": vae_l, **{k: list(v) for k, v in images.items()}}}
    w, h = size
    g[p + "latent"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": int(w), "height": int(h), "batch_size": 1}}
    for i, s in enumerate(seeds):                     # ids sort in seed order (progress_standin names several SaveImage nodes image, image2 ... by sorted node id)
        g[f"{p}ks_{i:02d}"] = {"class_type": "KSampler", "inputs": {"seed": s, "steps": int(steps), "cfg": float(cfg), "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
                                                                   "model": [p + "lora", 0], "positive": [p + "pos", 0], "negative": [p + "neg", 0], "latent_image": [p + "latent", 0]}}
        g[f"{p}dec_{i:02d}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{p}ks_{i:02d}", 0], "vae": vae_l}}
        g[f"{p}save_{i:02d}"] = {"class_type": "SaveImage", "inputs": {"images": [f"{p}dec_{i:02d}", 0], "filename_prefix": f"{prefix}_s{s}"}}
    return g


def seed_save_ids(seeds, id_prefix: str = "") -> list[str]:
    """SaveImage node ids of `angle_graph`, in seed order."""
    return [f"{id_prefix}save_{i:02d}" for i in range(len(seeds) if isinstance(seeds, (list, tuple)) else 1)]


def seed_list(base_seed: int, n: int = 4) -> list[int]:
    """n distinct sweep seeds derived from the place's seed (the first is the place seed itself, so n=1 = a plain edit)."""
    base = int(base_seed)
    return [base] + [(base * 2654435761 + 40503 * k) % (2**31 - 1) for k in range(1, max(1, int(n)))]


# ----------------------------------------------------------------------------------------------------------------------------------------------------- pick
# No embedding model is on the box (no CLIP-vision / SigLIP / DINO file, no node), the design VLM is too heavy for a per-image pick, so the pick is a colour score:
# a palette histogram of the whole frame (light + grade) and a histogram of the SUBJECT pixels (the pixels furthest from the frame's median colour: the vest, the kayak,
# the dark clothes against pale water). Both are histogram intersections in 0..1. It measures palette + clothing colour, NOT faces or hair: it cannot tell two faces apart.
# A caller that has a real subject matte (Alpha-Gen) passes `ref_mask` / `cand_mask` and the score uses those pixels instead of the heuristic.
SUBJECT_FRACTION = 0.2          # share of pixels taken as the subject by the heuristic mask
W_PALETTE, W_SUBJECT = 0.4, 0.6


def _rgb01(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3].astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def _downscale(a: np.ndarray, side: int = 96) -> np.ndarray:
    h, w = a.shape[:2]
    st = max(1, int(round(max(h, w) / side)))
    return a[::st, ::st]


def _hist(px: np.ndarray, bins: int = 6) -> np.ndarray:
    """normalised joint RGB histogram (bins^3) of an (N, 3) pixel array in 0..1."""
    if len(px) == 0:
        return np.zeros(bins**3, np.float32)
    q = np.clip((px * bins).astype(np.int32), 0, bins - 1)
    idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
    h = np.bincount(idx, minlength=bins**3).astype(np.float32)
    return h / h.sum()


def subject_mask(img, fraction: float = SUBJECT_FRACTION) -> np.ndarray:
    """heuristic subject region (bool, on the 96-px downscale): the `fraction` of pixels furthest (RGB distance) from the frame's median colour."""
    a = _downscale(_rgb01(img))
    d = np.linalg.norm(a - np.median(a.reshape(-1, 3), axis=0), axis=-1)
    return d >= np.quantile(d, 1.0 - float(fraction))


def _small_mask(mask, shape) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    h, w = shape[:2]
    ys = (np.arange(h) * m.shape[0] // h).clip(0, m.shape[0] - 1)
    xs = (np.arange(w) * m.shape[1] // w).clip(0, m.shape[1] - 1)
    return m[np.ix_(ys, xs)]


def consistency_score(ref, cand, *, ref_mask=None, cand_mask=None) -> float:
    """0..1, how much `cand` looks like the same subject in the same light as `ref` (higher = more consistent). ref / cand: HxWx3 arrays (uint8 or 0..1 float).
    ref_mask / cand_mask: optional subject mattes (HxW, any size); without them the heuristic `subject_mask` is used."""
    a, b = _downscale(_rgb01(ref)), _downscale(_rgb01(cand))
    ma = _small_mask(ref_mask, a.shape) if ref_mask is not None else subject_mask(ref)
    mb = _small_mask(cand_mask, b.shape) if cand_mask is not None else subject_mask(cand)
    pal = float(np.minimum(_hist(a.reshape(-1, 3)), _hist(b.reshape(-1, 3))).sum())
    sub = float(np.minimum(_hist(a[ma]), _hist(b[mb])).sum())
    return W_PALETTE * pal + W_SUBJECT * sub


def pick_best(ref, cands, *, ref_mask=None, cand_masks=None) -> tuple[int, list[float]]:
    """index of the most consistent candidate of a seed sweep + all scores (ties: the earlier seed)."""
    if not len(cands):
        raise ValueError("pick_best needs at least one candidate")
    scores = [consistency_score(ref, c, ref_mask=ref_mask, cand_mask=(cand_masks[i] if cand_masks else None)) for i, c in enumerate(cands)]
    return int(np.argmax(scores)), scores
