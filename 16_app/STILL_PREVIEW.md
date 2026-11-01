# Preview (app quality `still`, the still-based preview) and the upgrade to Final

Decision 2026-10-30 (user): no stand-in mode any more. A quick first look is built from the shots' own STILLS (one-frame matte and depth), and only after that are the stills animated with LTX
(clean i2v, Alpha-Gen over the shot, depth over the shot, ...). The stand-in (cached clips / Ken Burns over still 1 / luma matte + depth / `mograph_cache`) is deleted.

## What `still` is
* App Quality option "Still preview" (value `still`; `IN quality`). `MographRequest` turns it into `quality draft` + `still_preview: true` (`tools/fusion/still_preview.py`, `split_quality`), so every
  `== "draft"` rule of the code base applies unchanged: draft sizes (control / shot / type 576x1024, compose 540x960 -> the stills are the DRAFT-size stills), draft type job, flat words only (no LTX
  letters, no text pass), draft encode. `MographPiecePlan` puts `still_preview` and `request` (the app inputs as given) on the piece; `design_step.LOCKED_KEYS` keeps both through the design.
* The app shows TWO Quality choices (user 2026-10-30): **Preview** (value `still`, this document) and **Final** (value `final` -> the internal tier `still_preview.APP_FINAL_TIER = "draft"`: LTX 576x1024 single stage, compose 540x960; change that one line to "final" once the two-stage final has run as a whole piece). `draft` / `preview` stay internal tiers (batch drivers, tests). Tier graphs default to Final, the plain design graphs to Preview.
* Per stage (Gemma concept, Gemma design, MOSS stills, music, type, 3D cuts, compose, assemble, archive: exactly as a draft run):

| stage | still preview |
|---|---|
| concept, design, MOSS stills, music | as normal; stills saved at the draft still size (`<stem>_still.png`, `<stem>_still_<k>.png`) |
| footage per shot | the shot's OWN still (row of `MographFirstFrames` named by the shot's `first`) held for the shot length at the compose size, no zoom (`footage.still_outputs`) |
| clean i2v, union-control text pass, clean plate | not requested (`check_lazy_status` of the three gather barriers returns nothing) |
| matte | the existing LTX Alpha-Gen IC-LoRA on a 9-frame clip of THAT still (`a{k}.*` as before; no other matting model is installed - `comfyui_controlnet_aux` is on the 3060 card but marked "remove from card", no BiRefNet / rembg); `MographDecodeClip(gray)` repeats frame 0 over the shot as a view |
| depth | Video Depth Anything Small on that still, 1 frame |
| footage flow | zero (no RAFT / DIS over identical frames) |
| 3D cuts | normal: they project the held still + its depth + its matte (matte follows its own picture) |
| text | graphic type only |

* Why no zoom ease: the matte and depth are cut from the unzoomed still; a 2-3 % ease would move the picture but not the matte (it would need the matte warped per frame: 2.7 GB of float32 per shot at 576x1024). Held stills keep the subject exactly on its matte.
* Labels: `run_batch.py --quality still` ends every part label in `_still` (`r01_gemma_ltx_still`). The graph's own `archive/motngrfx` and `video/motngrfx` prefixes are literals that progress_standin re-points per part, so the part label is the tag.
* NOT MEASURED. Estimates only: the LTX clean + text stages disappear; Alpha-Gen on 9 frames instead of up to 145 is roughly 1/16 of the Alpha-Gen stage (BUILT.md Speeds: Alpha-Gen 461 s on a draft piece) plus its model load; VDA on 1 frame is seconds; compose is unchanged.

## Lock: upgrade to the real render with the SAME stills and the SAME design
Everything is already in the preview run's archive folder (`<stem>_out_shots`): `<stem>_piece.json` (designed piece incl. `request`, `still_preview`), `<stem>_still.png` + `<stem>_still_<k>.png`, `<stem>_music.wav`.

    python tools/app/run_batch.py --upgrade COMPETITIONS/motion-graphic/app/v012/r01_gemma_ltx_still --iid <iid> --port <port> [--graph <the app graph>] [--dry-run]
    python tools/app/upgrade_still.py --target COMPETITIONS/motion-graphic/app/v012 --part r01_gemma_ltx_still --iid <iid> --port <port>

`upgrade_still.upgrade_graph` (pure) turns the app graph into the upgrade graph: `quality final` (= `APP_FINAL_TIER`); `request` = the saved words / scene / music / look / shots / owners literals (concept chain removed);
MOSS chains -> `LoadImage` of the saved PNGs (same pixels); `plan` and `design_piece` -> `MographPieceFromJSON` of the saved designed piece (no design model, no repair); the saved wav as user audio.
Real LTX footage, Alpha-Gen and depth over the whole shot, 3D cuts, compose. Only while `APP_FINAL_TIER` is the draft tier (the stills are draft-size; `upgraded_piece` refuses otherwise).

**Nothing is overwritten.** The upgrade READS the preview's files and runs under a NEW part label: `r01_gemma_ltx_still` -> `r01_gemma_ltx_up540` (a second upgrade `..._up540b`), so its stills (re-saved
by the archive as the new run's own `<stem>_still*.png`), clips, mattes, depth, `piece.json`, wav carry another stem; the ComfyUI input copies carry the new part too (`tests/test_still_preview.py`).

Manual steps (if the script cannot be used): (1) copy the preview's `<stem>_still*.png` and `<stem>_music.wav` into the box's ComfyUI `input/` under NEW names; (2) load the app graph, set `IN quality` draft, `IN audio` = the wav,
`IN seed` / `IN length` / `IN format` and the request literals from `piece.json -> request`; (3) replace the `still_dec` / `still<k>_dec` VAEDecode nodes by LoadImage nodes of the PNGs; (4) replace nodes `plan` and
`design_piece` by `MographPieceFromJSON` with the preview's `piece.json` (delete `still_preview`, set `quality` draft); delete the concept / design-model nodes; (5) submit through `progress_standin.py submit ... --part <new part>`.

## Not done / open
* Alpha-Gen is still an LTX sampler (9 frames per shot). A truly LTX-free matte needs a segmentation node (BiRefNet / rembg) installed on the card.
* `plan_rules.py` still tests `quality != "draft"` for the owner fields; a still preview is a draft so this is right.
* The graphs on disk must be regenerated (`build_graphs.py`, then `layered_shots`, `llm_arms`) for the `first_set` gather input, the `Still preview` option and the removed draft nodes.

## Randomise layout (the Layout panel of the Preview result)
User 2026-10-30: "an option on the preview to create a random seed so we can see a preview of the text layout on the single frames and hit a randomise button to quickly create a new one".
* Node `MographLayoutPreview` (`nodes_fusion/layout_preview.py`, code `tools/fusion/layout_preview.py`): input = the archive folder of a Preview run (`<stem>_piece.json` + `<stem>_still*.png`; or `piece` + `stills`), `seed` (randomize control), optional `effects`.
  It draws the per-piece random system AGAIN with the seed on the SAME words / timings / shots: fonts (n from the length, one face per word), layouts (line / stack / rot +-90 / grid / column, no-orphan rule), the still palette (extracted with the seed + a seeded legible pair),
  the outline mode, the hold-life choice, the cut kinds (`compose.cuts` remapped onto the drawn kinds). Then ONE frame per word at its landed hold (half a beat after the hit): the type pass for those frames only (beauty + matte, CPU subprocesses) composed over THAT word's own shot still through
  `legible_color`. Output: a contact sheet IMAGE (360x640 tiles, word / layout / font under each, the seed + cuts + outline + hold life + palette in the header), the drawn system text, the re-drawn piece. No Gemma, no LTX, no GPU.
  Files next to the archive: `<stem>_layout_s<seed>.png` + `.json` (new name every time: `_b`, `_c`). Each word is its own short scene (the camera plan of one scene walks the whole timeline); one subprocess per word, 11 s deadline: a word whose scene is too slow (dot / pixel faces on the `circuit` module) or fails is drawn flat in its own face and layout and marked `(flat)`. Measured on the box CPU (shared with a render): 2.4-3.4 s for a 4-word piece, 5.4-11.7 s for a 6-word 30 s piece (BUILT.md).
* Graph: `tools/app/layout_preview_graph.py` (`--write-template` writes `app/graphs/_layout_preview.json`; `--archive <dir> --seed N --out g.json` for a standalone graph; `--via-load` puts MographArchiveLoad in front, which decodes every clip + matte: slower). The page fills `layout.inputs.archive / seed` and posts it to /prompt (the `_` prefix keeps it out of the template dropdown).
* Page (`web_app/app.js`, `app.css`): in the result view a Layout panel: the sheet, the seed, **Randomise** (new seed -> queue the graph over /prompt + /ws, history polling as a fallback -> /view), **Use this layout** (POST `/mograph/layout_lock` writes `<archive>/<stem>_layout_lock.json`; the page keeps the seed
  and sends it as `MographRequest.layout_seed` of the next run; "clear" drops it). The archive folder + stem come from MographArchive's ui text.
* Honoured by the planner: request `layout_seed` (>= 1; 0 / empty = none) -> `plan_rules._build` -> `layout_preview.with_layout_seed`: the LAYOUT side of the piece system (fonts, cut kinds, outline, inverse option, effects, and `seed`, which seeds `stable_type.draw_layouts`, the hold-life draw and the outline windows) is the draw of that seed; module / io / camera / photographic style /
  footage palette family / the contrast sections placed on shots stay, so the stills and footage prompts do not change. `design_step.still_palette_of` extracts the palette with the same seed. `plan.layout_seed` is recorded. **Preview -> Final (`tools/app/upgrade_still.py load_saved`)** applies a saved lock to the saved designed piece (same words / timings, the layout seed's draw), no model call.
* Hook for the graph builder (`tools/app/build_graphs.py` is another worker's): one line in its main after the app graphs are written: `from tools.app import layout_preview_graph; layout_preview_graph.main(["--write-template"])`. The app graphs themselves need NO regeneration for `layout_seed`: the page sets the optional input on the MographRequest node of the prompt.
* Not done: the tile shows the drawn type over the still (legible colour, forced outline windows), not the hold-life animation, window / inverse words or the 3D cuts; word timing is not re-planned (a re-drawn layout may need a longer hold for a long word: the real design step re-checks it).

* Measured 2026-10-30 (integration round): the still-based Preview was NOT rendered in this round, so its wall time is still unmeasured. What was measured instead is the real low-res tier (`--quality preview`, LTX 384x704, compose 360x640) with the same designer: 1058.7 s and 932.9 s of ComfyUI time for a 16 s piece (22_batch-24gb v024 r01 / r02, BUILT.md Speeds). The Preview should be well under those: no LTX clean / text passes, Alpha-Gen on 9 frames, VDA on 1 frame.
