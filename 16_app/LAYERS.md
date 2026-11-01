# LAYERS - the three-layer shot route (clean footage, subject matte, text-rendered background)

Built 2026-10-28, UNCOMMITTED, not run on a GPU. Open models only, NO Claude. Code: `tools/app/layered_shots.py` (graph transform), `nodes_fusion/layers.py` (4 glue nodes),
`tools/fusion/layers_recipe.py` (compose side), additive hooks in `tools/fusion/recipe_eval.py`, `tools/fusion/window_cut.py`, `tools/fusion/compose.py`
(`depth_occlude(fg=)`), `nodes_fusion/compose.py`, `nodes_fusion/archive.py`. Tests `tests/test_layered_shots.py` (20). Graphs `app/graphs/fusion_design_qwen[_auto]_layers.json`.

## 1. The bug
Every app shot was ONE union-control LTX shot: the type control pass (MographTypePasses) guides `LTXAddVideoICLoRAGuide` (`ltx-2.3-22b-ic-lora-union-control-ref0.5`), and the
Alpha-Gen matte (`ltx-2.5-22b-ic-lora-alpha-gen-0.9`) was made from THAT clip. Batch-3060 v001 run 3 shot 2: `s2_clip.mp4` is giant stone letters and no subject, `s2_matte/` is the
letters. Compose cannot put type between the subject and the background when the "subject" is the letters. User requirement: "They need to stay separate, to be able to put text
between the subject and the background."

## 2. The route (per shot k)
```
slot k (MographShotSlot, unchanged: prompt, seed, frames, control slice, first frame, guide)
 |
 +-- l{k}.*  CLEAN      plain LTX image-to-video, base model (node 1, no LoRA), first frame = MOSS still / previous CLEAN clip's last frame,
 |                      prompt = slot prompt minus the "letters built into the scene" / "strands trace the lines" clauses (MographLayerPrompt)
 |                      -> l{k}.dec = clean clip
 +-- a{k}.*  MATTE      the existing Alpha-Gen chain; its reference comes from MographLayerGather = the CLEAN clip (not the text clip)
 +-- t{k}.*  TEXT BG    only when the shot has a text pass (control guide > 0): union-control V2V of the clean clip
 |                      -> t{k}.dec = text-rendered background (same scene, same camera)
 gather (MographLayerGather) -> per shot: u8 = text clip if made else clean clip (the BACKGROUND), subject = clean u8, textbg = bool
 compose: background -> crisp type layers -> SUBJECT (clean pixels, cut by the clean matte) on top
```
**Text-pass wiring (V2V).** ComfyUI-LTXVideo @ 3bf3ca6 has no "text V2V" graph. Its 2.5 `V2V_ICLoRA_Single_Stage` and `ICLoRA_Union_Control` examples both use the source frames only as
`LTXAddVideoICLoRAGuide` reference tokens on an empty latent (the same pattern as our Alpha-Gen chain), and the repo's own refine pass (stage 2 of every two-stage graph) is the other V2V
mechanism: start from an existing latent, noise it to sigma `--v2v-start` and run the tail of the distilled schedule (the refine pass starts at 0.909). The text pass combines the two, on the nodes the app graph already has:
`t{k}.31 LTXVImgToVideoInplace(latent = clean latent)` -> `t{k}.32 LTXAddVideoICLoRAGuide(image = type control slice, strength = slot guide, latent_downscale_factor 2)` ->
sampler with `lay_sig` = `0.975, 0.909375, 0.725, 0.421875, 0.0` (DEFAULT since the 2026-10-28 probe, section 8: 4 steps; `--v2v-start 0.909375` = the refine-pass tail, 3 steps, but the letters stay ghost outlines).
The latent is the clean chain's `LTXVSeparateAVLatent` output taken directly (`l{k}.40[0]`): no VAE decode/encode round trip, no 360x640 -> 384x704 resize. `MographLayerGate` passes it
through and depends on the clean barrier, so ALL clean shots run before the first union-control pass, and `MographLayerGather` waits for all text passes before Alpha-Gen starts: the
LoRA patch changes twice (base -> union -> alpha), not per shot. The first frame is pinned by `t{k}.31` as before.
Why not the union LoRA on RGB frames: the union IC-LoRA was trained on depth/canny/pose control, not RGB, so the clean clip is the INIT latent, never the guide.

**guide == 0.** `wants_text(piece, sid)` = control guide > 0 (read exactly as MographShotSlot reads it) and not `piece["layers"]["text"] == False`. `MographLayerGather.check_lazy_status`
requests `text_k` only for those shots, so the `t{k}.*` chain of a guide-0 shot is never executed (same lazy gating as the existing gather): no V2V pass, background = clean clip.
The planner's `behind` recipe uses guide 0.4 for every shot, `built_in` 1.0; `traced` runs a line control with no matte (no Alpha-Gen, text pass only).

## 3. Compose
`MographCompose` hands `clip["subject"]` to `recipe_eval.run_compose(subject=)`. With a subject the recipe is built `layered=True`: `$footage` stays the BACKGROUND (grade, tint, legible
colour, hold net, glow, knockout window content), and `tools/fusion/layers_recipe.layerize_ops` rewrites the ops that take "the subject":

| op | change |
|---|---|
| `sandwich` (not inverted) | `fg` = subject version of the plate: the `grade` chain that made `$foot` is cloned onto `$subject` (same grade), so the subject is graded like the plate was |
| `sandwich` with `invert` | untouched: its `fg` is the BACKGROUND drawn over the type |
| `knockout` with `fg_matte` | `fg` = subject version of `fg` / `footage` |
| `window_cut` | explicit `fg` cloned; otherwise new args `a_subject` / `b_subject` (`$footage`->`$subject`, `$next`->`$next_subject`, `$prev`->`$prev_subject`, `$shot:<id>`->`$shot:<id>:subject`); the op takes the subject of whichever shot is the plate / the window |
| `depth_occlude` | new arg `fg`: pixels shown where the footage covers the type = clean clip. The depth ref is cut from the clean clip too |
| `upscale_matte` | `guide` = `$subject` (the matte was cut from the clean clip) |
| `crossfade(composite, plate)` (`wipe` brighten) | the plate is first re-composited with the subject (`sandwich` with type_opacity 0) so the background's own subject pixels are not blended back in |

Added to the recipe inputs: `subject` (+ `next_subject` / `prev_subject` / `shot:<id>:subject` via `window_cut.other_shot_ref`; `collect_others(subject_of=)`). Single-clip is the default:
`layered=False`, `subject=None` leave every existing recipe, test and the old graphs byte-identical (tests `test_single_clip_recipe_is_unchanged_by_the_layer_support`).
Not layered, reported in `rec["layers"]["skipped"]`: a plate chain through a stateful op (flow_wake, hold_net, legible_color, trace_edges, legible, multiplane); multiplane pieces compose
single-clip. Not rewritten: `refract`, `luminosity_key`, `type_from_edges` (read the footage as the thing the type is made of, not as the subject).

## 4. Archive
`MographArchive` writes per layered shot `sK_clean.mp4` (always), `sK_textbg.mp4` (only when a text pass ran), `sK_matte/NNNNN.png` (Alpha-Gen on the clean clip); `sK_clip.mp4` stays for
single-clip shots. `index.json` has `shots.sK.layers = {clean, textbg}`. `tools/app/run_timing.py` STAGES: "Clean shots" (`l{k}.*`, `lay_clean`, `lay_prompt`), "Text shots" (`t{k}.*`, `lay_gate`, `lay_sig`).

## 5. THE HOOK for `tools/app/build_graphs.py` (APPLIED 2026-10-28: `python tools/app/build_graphs.py --layers 3` writes `fusion_design_qwen[_auto]_layers.json`; `--check --layers 3` compares them)
Standalone it already works: `python tools/app/layered_shots.py` reads `app/graphs/fusion_design_qwen[_auto].json` and writes `*_layers.json` (+ `.app.json`); `--check` for staleness.
To get it from `build_graphs.py` two additions in `main()`:
```python
    ap.add_argument("--layers", type=int, choices=(1, 3), default=1,
                    help="3 = three-layer shots (clean + Alpha-Gen matte + text V2V), tools/app/layered_shots.py; writes <name>_layers.json")
    ...
        g, side = make(draft, oi if a.fill else None, a.still, design, auto)       # existing line
        if a.layers == 3 and design:                                              # NEW (after make, before text = dump(g))
            from tools.app import layered_shots
            g, side, suffix = layered_shots.build_graphs_hook(g, side, design, auto)
            name += suffix
```
`build_graphs.check(g, side)` accepts the result (test `test_passes_the_repo_graph_checker`; the new classes register themselves in `ltx_ic.CLASSES` on import). To make three layers the default for
the design graphs, call the hook unconditionally instead of behind `--layers`.

## 6. Cost per piece (from measured numbers, NOT measured for this route)
Baselines (BUILT.md Speeds, RUN_3060.md section 10, 7 pieces of 16 s): LTX shots mean 352 s (1 shot 206-217 s, 4 shots 378-423 s, i.e. ~150 s fixed + ~60-70 s per extra shot), Alpha-Gen 220 s,
wall mean 1302 s. The clean shot replaces the old control shot at the same size and length (no guide tokens: equal or slightly cheaper than 352 s). Alpha-Gen is unchanged. The only
extra is the text V2V pass, once per shot with guide > 0:
- conservative (the brief: "a V2V pass ~ the same per text shot"): **+352 s per piece** (+27 % of the 1302 s mean wall, ~1650 s);
- from step count (4 of the 8 distilled steps; the 3-step default before the probe was ~25 % cheaper in the sampler, decode and guide encode unchanged, no model load): **about +130 to +200 s** (+10 to +15 %);
- guide 0 on every shot (or `layers.text = false`): **+0 s** (clean replaces control).
Unmeasured extras: one more LoRA patch change (base -> union), the second archive mp4 per shot (~3-7 s), compose is unchanged (a few extra graded `subject` frames per sandwich, GPU).
Peak VRAM should not change (same model, same sizes; the clean latent is a few MB). Peak RAM: one more uint8 clip per shot (preview 360 x 640 x 3 B x 121 frames ~ 84 MB per shot).

## 7. Untested until a box runs it
- Everything on a GPU. The static tests cover the graph, the glue nodes with fake clips (local ComfyUI checkout) and the compose maths on tiny frames.
- **Whether the text pass forms letters at sigma 0.909 / 3 steps while keeping the scene: ANSWERED by the probe in section 8 (no; 0.975 / 4 steps does).** Still open: the full 385-frame shot (the probe cut 49 frames). If the scene drifts, the sigma is the only knob (the IC guide strength is the slot guide, 0.4 for `behind`).
- Frame 0 of the text pass is pinned to the first frame (`t{k}.31`), which is also the clean clip's frame 0: no conflict expected.
- A latent coming out of `LTXVSeparateAVLatent` into `LTXVImgToVideoInplace` / `LTXAddVideoICLoRAGuide` (noise mask handling): the nodes read `samples` and `noise_mask` generically, not run.
- Subject edges: the matte is cut from the clean clip, the background is the text clip; the camera is shared only as far as the V2V keeps it.
- `wipe` / `front` / `window` treatments on a real layered shot, `window_cut` with a neighbour (`$next_subject`), `depth_occlude` (built_in) with a clean-clip depth.
- Two-stage (`fusion.json`) layered graph converts and validates; the text pass starts from the clean STAGE-1 latent there.

## 8. Text-pass probe (2026-10-28, RTX 3060, `21_batch-3060/v006`, `tools/app/text_pass_probe.py`)
Question: which V2V start sigma / guide forms legible letters on a CLEAN clip while subject and camera hold. Setup: the archived v005 r01 `s1_clean.mp4` (potter), first 49 frames at 384x704, the shot's real control
pass (depth, word HANDS visible from frame ~25), the app's own `t1.*` text chain cut out of `fusion_design_qwen_auto_layers.json`, clean latent = VAE-encoded clip (instead of `l1.40`), union-control guide 1.0, seed of the shot.
Strips (frames 0 10 19 29 38 48): `21_batch-3060/v006/strips/<variant>.png`, `clean.png` = the input.

| variant | steps | warm run s* | letters legible | subject / camera match the clean clip |
|---|---|---|---|---|
| start 0.909 (old default), slot prompt | 3 | n/a (cold 143) | no: faint ghost outlines (`HANS` thin line) | yes (bowl form changes slightly) |
| start 0.909, ONE-word prompt | 3 | 40 | no: thin outlines, 3 letters | yes |
| **start 0.975, slot prompt (all words)** | 4 | 44 | yes-ish: solid sandstone H-A-N-D-S stack, some garbling | yes (hands / bowl differ in detail, same place) |
| **start 0.975, ONE-word prompt** | 4 | 81 (queue overlap; sampler 4 steps ~ same) | **yes: cleanest letters** | yes |
| start 0.975, guide 0.65, slot prompt | 4 | 37 | weaker / thinner, garbled | yes |
| start 0.99375 | 7 | 63 | yes, solid | NO: subject slides out of frame |
| start 1.0 (full from noise, first frame pinned) | 8 | 57 | yes | NO: scene and subject lost (blur) |
*total wall seconds of the ComfyUI run on a warm box (models loaded; includes the ~5-10 s CPU type pass, VAE encode, tiled decode, mp4 save). Sampler ~5 s per step at 49 frames / 384x704 with the guide tokens.
The cold first run (143 s) paid the model load.
Result: **0.975 / 4 steps is the cheapest start that forms letters and keeps the scene**; guide 1.0 beats 0.65; naming ONE word in the text-pass prompt reads better than the 3-5 word slot prompt (NOT wired: `t{k}.6` still takes the slot prompt;
a per-window one-word prompt is the next step). The letters are as legible as the control pass's font allows (script face, stacked vertically). The previous real run (385 frames, 0.909) showed no letters; the probe at the same sigma
shows only ghost outlines, so the sigma was the cause. Not tested: other seeds, other shots, the full 16 s shot, a latent straight from `l{k}.40` (no VAE round trip).

## 9. VLM legibility check + why the type sat over the climber (2026-10-28, batch 3060 v007 r01)
**Reading check (decider).** `MographLtxRead` (`nodes_fusion/layers.py`, wired as `ltx_read` between `flow` and `compose` by `layered_shots.convert`; `llm_arms.convert` gives it the design model) crops 3 frames of the text-rendered clip per ltx / handoff word (after handoff + 0.25 s, cut to the dilated type-matte box), the design VLM answers `Read the text in this image exactly. Reply with the text only, or NONE.`, `ltx_check.apply_fallback(reads=)` compares letters-only upper case, EXACT by default (`piece.layers.ltx_read_dist`: int or "fuzzy"), majority of crops; fail = the word is flat for its whole span, the readings are logged in `plan.owner.fallback` / `ltx_check`. The pixel excess stays in the log (it passed icicles at 0.097 and `BRAAKER` at 0.112). Base graphs (Qwen transformers) have `model = ""` = node off, pixel test decides. Probe on the box with the app's own `MographGGUFVL` (`tools/app/ltx_read_probe.py`, crops in `16_app/core_check/layer_check/read_probe/`): STILL STANDING -> `STILL STANDING` x2 + `NONE` (pass), STORM BREAKER -> `STORM BREAKING`, `STORM BRAAKER`, `NONE` (fail -> flat). Qwen3.x arms: text-only calls use the llama.cpp JSON grammar (`json_mode`).

**Layering diagnosis + fix.** Compose DOES put the clean subject above the type in the layered route (`layers_recipe.layerize_ops` gives `depth_occlude` a `fg` = the clean clip, `nodes_fusion/compose.py` passes `clip["subject"]`). But in v007 r01 (and every layered `depth_occlude` shot: the `behind` look, plane 0.55) "the subject" was only what the DEPTH map calls nearer than the plane: the shot's recipe never reads `$matte`, so `shot_needs(matte)` was false, the Alpha-Gen chain `a<k>.*` was never requested (relay log `16_app/bench/batch_3060_v007/run_01.relay.log`: `d1.*` ran, no `a1.*`), the gather gave placeholder alpha refs and the archive wrote no `sK_matte/` (the wiring `a<k>.dec[1]` -> compose / archive `matte_k` is fine, it is lazy). A small figure on a summit against sky is easy for a depth model to call far (stand-in Depth-Anything-V2-Small: climber 0.09, mountain 0.72; the real Video-Depth-Anything depth was not archived), so the cover was 0 over him and the type was drawn across his torso. A second, intentional factor: the legibility guard (`compose.legible`, v005 user rule) clears the cover over letters it would hide > 20 % of a hold; with a matte-less depth cover it can also clear him.
FIX (2026-10-28): in a layered shot a `depth_occlude` recipe now needs the matte (`recipe_eval.shot_needs(piece, sid, layered=True)`, called by `MographLayerGather`, `MographCompose` and `MographArchive` when the clip has a subject), so Alpha-Gen runs on the clean clip, and `build_recipe(layered=True)` adds `subject_matte=$sK_matte` to the op: `compose.depth_occlude(subject_matte=)` makes the cover at least the matte AFTER the guard, so the clean subject stays in front of the type whatever the depth map says (the guard still protects letters from scenery such as the mountain). Opt out: `piece.layers.subject_matte = false`. Cost: one Alpha-Gen pass per `depth_occlude` shot (BUILT.md Speeds: Alpha-Gen 220 s per 16 s piece). Sheet `16_app/core_check/layer_check/sheet.png` (CPU, real piece + clips + type passes, row 2 = before, row 3 = after with a stand-in silhouette matte for the climber since no matte was archived); tests `tests/test_layered_shots.py` (2 new). Not run on the box.

## 9. Background choice, negatives, control pass, one text (2026-10-29)
- **Which picture is the background** (`ltx_check.choose_background`, run for every layered clip before any shot is composed: `nodes_fusion/compose.py prepare_backgrounds`): the text-rendered clip only while the shot still has an LTX-owned word AFTER the reading + layout checks and the subject of that clip sits within `matte_align.DRIFT_MAX` (1.2 % of the short side) of the clean clip the matte was cut from. Otherwise the CLEAN clip is the background (the shot's remaining LTX words become flat, logged in `plan.owner.background` / `fallback`). This is the fix for v005 r02 s2 (DRIFT fell back to flat but the shot kept the text pass under it: two versions at once). Plate chains the layers cannot clone (flow_wake, plate_pan, hold_net ...) also compose on the clean clip alone (`run_compose`, `stats.layers_fallback`).
- **No invented text**: the clean chain `l{k}.*` has its own negative prompt `text_guard.NEG_NO_TEXT` (node `lay_neg{k}`), its prompt loses every text mention (`layered_shots.clean_prompt`); the text chain `t{k}.*` has `NEG_TEXT_PASS` (node `lay_negt`) and its prompt spells exactly the control pass's word(s) in the graphic's look (`owner.clause(look=, lines=)`). Piece-level validator `text_guard.plan_problems` (design check), mender `text_guard.fix_plan`.
- **Control pass per face** (`owner.control_pass_for`, set by `owner.apply_to_plan`): thin / dotted / outline faces -> `line` (canny), mid -> `union` (depth with the line drawn over it, `footage.read_control_frames`), heavy -> `depth`. Box probe: `python tools/app/text_pass_probe.py build --passes ...` (depth / line / union at draft size), `... score` (layout IoU).
- **Layout check** (`ltx_check.layout_iou_frames`): IoU of the letters the video model drew with the flat type matte >= 0.5, else the word is flat. **One text**: the flat graphic holds until `ltx_at`, then a <= 0.25 s crossfade (`owner.check_one_text`, warned in compose).
