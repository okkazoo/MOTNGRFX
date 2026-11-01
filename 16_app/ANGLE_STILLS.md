# Same-subject new-angle stills (hand-off, 2026-10-29)

For whoever wires it AFTER the design worker finishes (this file touches none of `qwen_piece.py`, `design_options.py`, `design_salvage.py`, `design_step.py`,
`DESIGN_GUIDE_open_model.md`, `app/graphs/*`, `tools/app/build_graphs.py`: every change to those is proposed below). Evidence: `23_moss-angles` v001-v003
(`v003/review/sheet_*.png`, `version.json`). The pure code is ready and tested: `tools/fusion/angle_still.py`, `tests/test_angle_still.py` (22 tests).

User ask (2026-10-29): "I like what you are getting with moss edit - I don't mind if the person is slightly different but if we can keep them consistent someway that
would be great - for multi shots we don't always have to use the same image from different angles but it could be effective as a choice that is available."
=> an OPTION per place, not the default. A place is either a fresh text-to-image still (today) or an edit of another place's still from a new camera.

## 1. What it is

MOSS-Image SFT + LoRA `moss_image_edit_chinoll` (`LoraLoaderModelOnly`), `TextEncodeMOSSImageEdit` (`edit_format` "Moss-Image-Edit", the reference as `images.image_0`, vae
connected, the SAME node type for the negative), KSampler 30 steps, cfg 4, er_sde / simple. The reference is ALWAYS the original still (or the user's own first image): never
an edit (edits darken / drift when chained, memory `edit-models-darken`). `angle_still.build_edit_prompt` writes the anchored prompt, `angle_still.angle_graph` the nodes.

## 2. Schema for a place (piece JSON `stills[]`)

Today: `{"id": "B", "model": "moss", "seed": 123, "size": [w, h], "prompt": "...", "negative": "..."}`. Proposed, one extra form:

```json
{"id": "B", "model": "moss_edit", "angle_of": "A", "framing": "close-up", "seed": 123, "size": [w, h],
 "prompt": "<the anchored edit prompt, written by the code>", "negative": "<same as A>",
 "hide_face": true, "sweep": 4, "ref_person": null}
```

- `angle_of`: id of ANOTHER place that is a normal text-to-image still, or the user's own first image (row 0). Never an angle place (no edit of an edit). The code downgrades an
  angle place whose `angle_of` is unknown or is itself an angle place to a normal still (logged, `why`).
- `framing`: `close-up | extreme close-up | low | high | profile | top-down | wide | medium | behind | over-the-shoulder` (aliases such as "low angle", "overhead" are normalised by
  `angle_still.normalize_framing`; any other phrase is used verbatim as the camera clause).
- `prompt`: NOT written by the design model. `plan_rules` builds it: `angle_still.build_edit_prompt(stills[A].prompt, framing, subject=<concept subject>, hide_face=...)`, so
  `MographStillPrompt` (prompt, seed) needs no change. `hide_face` defaults on for the framings in `angle_still.FACE_RISK` (close-up, low, medium, high, over-the-shoulder)
  when the subject is a person.
- `sweep`: seeds tried (default 1 in the app preview, up to 4); the pick is `angle_still.pick_best` (see 5).
- `ref_person`: optional id of an earlier ANGLE place whose image becomes `image_1` (see 3 verdict B). Default null. Only use a hidden-face image for it.
- `stills` is a locked key in the design step (`design_step.LOCKED_KEYS`): the choice therefore belongs to the CONCEPT / plan step, not to the design reply.

## 3. Probe results (idea `23_moss-angles/v003`, box 58159210, RTX 4090, 448x832, one reference still: a backlit paddler in a red vest, seen from behind)

Judged by me (the worker) from the contact sheets, no scout. Palette: the prompt now restates the still's own light ("warm golden light ..."); v002 restated "cool pale-blue",
which was NOT the reference's light, so v002's "palette kept" verdict was against a different light than the reference (the reference is a pale-gold sunrise, v003 follows it).

- A, seed sweep (4 seeds x close-up / low / profile / top-down, neutral wording; `sheet_a_seed_sweep.png`): palette, vest, kayak and paddle blades held in 16/16; the camera
  was followed in 16/16. The PERSON is not consistent across seeds: close-up (4/4) and low (2/4; the other two show the back) invent a different woman's face every seed (hair length and parting change too);
  profile and top-down show the person small, so the drift is hardly visible there. The edit also re-renders the light stronger and more golden than the reference
  (the reference is dim and muted): "do not change the colour grade" is only partly obeyed. Seeds differ mostly in pose and framing, so a sweep is useful to PICK A COMPOSITION,
  not to fix identity.
- C, face-hidden wording ("The person is seen from behind or from the side, the face not visible, the face in shadow." appended; `sheet_c_face_hidden.png`): close-up 4/4 and low
  4/4 show the person from behind or turned away, no face invented; dark shoulder-length hair, dark top, red vest the same in every seed. Neutral wording gave an invented face in 6/8 (close-up 4/4, low 2/4). This is the setting that gives "slightly different person, consistent enough". It is the default for the face-risk framings.
- B, second reference (`image_0` = the original, `image_1` = the best close-up of A, picked by the scorer = neutral close-up seed 42, a face-visible woman;
  `sheet_b_two_refs.png`, `sheet_b2_topdown_hidden.png`): the person IS held (same face, same bun in all 4 seeds of low and profile) but the camera instruction is lost:
  "low" and "profile" came out as near copies of the close-up pose. Top-down was fine (the person is small). Rewording the second-reference line to "identity only, do not
  copy the camera" did not fix it (`sheet_b3_second_ref_wording.png`; profile got worse). Reason: in the Moss-Image-Edit format the text encoder does NOT see the images (vae is
  connected, `EDIT_FORMATS["Moss-Image-Edit"] = (..., False)`), so the text cannot tell the model what the second image is for; it acts as a strong pose prior.
  With the HIDDEN-face close-up as `image_1` (seed 2718 of the hidden sweep) plus hidden-face wording, low gave 4/4 the same back view with the same hair / build /
  vest, i.e. the most consistent set of the probe, and the low angle was still followed (camera on the deck, over the shoulder). Verdict: second reference = an OPTION,
  useful only with a hidden-face reference; a face-visible reference makes every angle a clone of itself. Not tested: profile / top-down with a hidden-face reference.
- Pick scorer (`angle_still.consistency_score`, colour only; no CLIP-vision / SigLIP / DINO file or node exists on the box, the design VLM is too heavy per image): it separates
  face-visible (0.28-0.52) from back views (0.57-0.74) and palette-shifted seeds from faithful ones, but is nearly flat among seeds of the same kind (0.571-0.602 for the four
  hidden close-ups), and it cannot tell two faces apart. Treat it as a colour / clothing gate and a tiebreak, not an identity measure. Its picks on the sheets are marked PICK.

## 4. When the concept step should choose an angle place (proposed guide text, for `DESIGN_GUIDE_open_model.md` section "Shot structure (concept step)")

Add after the sentence about `continues`:

> A new place may instead be a NEW CAMERA ON THE SAME SUBJECT: give the `shot_plan` entry `"same_subject_as": "<place id, normally A>"` and `"framing": "close-up" | "low" | "profile" | "top-down" | "wide"`.
> Use it when (1) the piece is about one person or object and the next beat is the same subject from another side, (2) a shot continues the action but the camera changes
> (a hard cut to the same subject at a new angle, not a new location), (3) an intimate beat wants the person closer (framing close-up, the face stays hidden). Do not use it
> for a new location, a landscape or an abstract beat, or for a place whose subject is not clearly one person or object. At most two such places per piece, never one that
> names another such place. Fresh places (a new text-to-image still) stay the default; most pieces mix both.

Code side (plan_rules, not a design file): read `same_subject_as` + `framing` from `sp_plan[k]` where `place_still(...)` is called (about line 1221); emit the schema in section 2.

## 5. Graph wiring (proposed; the files below are the design worker's or generated, so nothing was edited)

Existing chain per place k (2..6), `tools/app/build_graphs.py add_place_stills`: `MographStillPrompt` -> `CLIPTextEncode` -> `EmptyFlux2LatentImage` -> `KSampler` -> `VAEDecode` ->
`MographFirstFrames.still_k` (lazy). The batch rows are read by `places.batch_rows` / `MographShotSlot` (`places.pick_row`): a row is just an image, so nothing downstream changes.

Add per place k an ANGLE chain next to it (same loaders: `still_unet`, `still_clip`, `still_vae`; the LoRA node is the only new loader-side node):

1. `LoraLoaderModelOnly` (`still_unet`, `moss_image_edit_chinoll.safetensors`, 1.0).
2. reference: the image of place `angle_of` = `link("first")` when `angle_of` is A (MographFirstFrame = the user's image or still A), else `still{j}_dec`. Scale it to the
   still size first (`TextEncodeMOSSImageEdit` uses the reference at the size it comes in; the probe used 384x704 against a 448x832 latent and was fine).
3. `TextEncodeMOSSImageEdit` x2: positive prompt = `still{k}_prompt` output 0, negative = the plan negative; both with the same reference. `angle_graph(ref=[node, slot], ..., loaders=
   {"unet":..., "clip":..., "vae":...}, id_prefix=f"ang{k}_")` returns exactly this fragment (ids `ang{k}_...`), 1 KSampler + decode per seed.
4. `KSampler` per seed: seed = `still{k}_prompt` output 1 (+ `angle_still.seed_list(seed, sweep)`), steps 30, cfg 4, er_sde / simple (NOT the t2i 25 / 2.5).
5. pick: new node `MographAnglePick` (nodes_fusion/inputs.py): inputs = reference IMAGE + N decoded IMAGEs (or one batch), output = `angle_still.pick_best` winner. N = 1 in the
   preview, so `sweep: 1` needs no node.
6. `MographFirstFrames`: add lazy inputs `angle_2 .. angle_6` next to `still_2 .. still_6`; `check_lazy_status` asks for `angle_k` when `stills[k-1].angle_of` is set and
   `still_k` otherwise; `execute` passes whichever arrived as that place's row (`places.batch_rows` unchanged, it takes `stills{i: image}`). Add `places.angle_source(piece, idx)
   -> index | None` and have `needed_stills` also return the source place index (A is always row 0, so it is already run).
7. `ref_person` (second reference): `angle_graph(..., second_ref=<link to that angle place's picked image>)`. Order matters: that place must be decoded first (it is a plain
   dependency; ComfyUI orders it).

Where it does NOT apply: the 12 GB (3060) card uses `moss_image_sft_int8_convrot` + w4a8 text encoder; the probe ran only on bf16 + Qwen3.5-9B bf16 (the 24 GB card's stills
set). Whether the LoRA patches the int8 convrot DiT and what it does to the picture is NOT tested, so: offer angle places on the 24 GB (bf16) tier only until someone runs it on
16 GB / 12 GB. The 16 GB tier (MOSS bf16) is expected to work, not run. The LoRA is in the 24 GB card only (`card-library/cards/COMPETITIONS/motion-graphic/card.yml`).

## 6. Timings and VRAM (RTX 4090 24 GB, warm models, 448x832, 30 steps; ComfyUI `/history` execution times)

- one reference, 4-seed sweep (shared encoders, 4 KSampler branches): 36.6 s per graph (6 runs, 36.56-36.98 s) = 9.1 s per image. One image alone: 12.2-13.6 s (v002 graphs, text
  encode + LoRA patch included, nothing cached).
- two references, 4 seeds: 40.1-40.6 s (4 runs) = 10.1 s per image (+10 %).
- peak VRAM reported by `progress_standin` 23,528-23,784 MiB of 24,564 against an idle 22,344-22,376 MiB: that box keeps other models resident, so the edit's own demand is not isolated; the
  step over idle was +1.2-1.4 GiB (reference latent + 297 MB LoRA). Peak RAM 23-31 GiB anon. Not measured on 16 GB / 12 GB / any other GPU.
- First graph after a cold start loads the DiT + 18 GB text encoder: not measured here (box was warm).
- NOT tested: one KSampler with `batch_size` = N for the sweep (cheaper graph, probably the same time); the LoRA with the cards' other MOSS builds; chained use inside the full app
  graph (the probe was standalone graphs through `progress_standin`).

## 7. Licence

See the report of 2026-10-29 and the comment block in the card: the LoRA weights carry NO declared licence (HF card has no `license` field, the GitHub repo is GPLv3 for CODE only
and its guide says the weights "use their own licence" without naming one); base MOSS-Image SFT/Base and the ConceptEdit-12M data it credits are Apache-2.0. The card downloads
from the author's repo to the user's own box; we do not redistribute. Remove the card row if undeclared is not acceptable for the contest entry.


## 8. WIRED into the app graph (2026-10-29, 24 GB tier; the box has NOT run the full graph)
What exists now: `build_graphs.add_angle_chain` (per place k = 2..6: `ang{k}_prompt` = MographStillPrompt with `angle: true`, `ang{k}_ref` = MographAngleRef, `ang{k}_pos` / `ang{k}_neg` = TextEncodeMOSSImageEdit, `ang{k}_lat`, `ang{k}_ks`, `ang{k}_dec`, one shared `ang_lora` = LoraLoaderModelOnly on `still_unet`), `MographFirstFrames` with lazy inputs `angle_2..6` and the switch `angles` (default false; `llm_arms.TIERS[24].nodes` sets it true, the 16 GB and 3060 tiers keep it false), `places.angle_source / angle_sources / angle_prompt`, `angle_still.is_person`, tests `tests/test_angle_graph.py`.
- A place with `angle_of` (+ `framing`; plan_rules writes both and keeps `angle_fallback` + a fresh-still prompt) is the EDIT of its source place's ORIGINAL still when `angles` is on, else the fresh still (the old behaviour). The source is `first` for place A (the user's own image or still A), another place's own still (lazy `still_j` of MographAngleRef) otherwise. An edit of an edit, an unknown id or itself falls back to the fresh still.
- The `last:sK` conversion (no last-frame chaining, plan_rules) already writes `angle_of` stills; `test_a_chained_plan_gets_angle_places_that_the_graph_can_edit` checks that each of them has a real angle chain and an anchored prompt.
- Prompt: `angle_still.build_edit_prompt(source still prompt, framing)`, face-hidden wording for the face-risk framings when the source subject is a person (`is_person`, word list); a place may carry `subject`, `palette`, `keep`, `hide_face`, `person` to override. Settings: 30 steps, cfg 4, er_sde / simple (not the t2i 25 / 2.5), seed = the place's seed, ONE seed per place (sweep 1; the colour pick of section 5.5 is not in the graph).
- The reference is cover-fitted to the still size (`sizes` 6 / 7) like the t2i latent; the negative is the plan's negative text. Loaders are the first still's: no extra model load, only the LoRA patch.
- Static check without a server: `python tools/app/graph_signatures.py app/graphs/<graph>.json --prefix ang` compares every node class with the graphs the box ran (23_moss-angles/v003).
- NOT tested: the full graph on the box, the int8 convrot (3060) DiT with the LoRA (so `angles` stays off there), sweep > 1.
