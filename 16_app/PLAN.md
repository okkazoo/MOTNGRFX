# 16_app — Mograph app (contest entry) — plan (2026-10-26)

Goal: one app that turns words (+ optional image / audio) into a 9:16 type-in-footage video, with every heavy stage
on the GPU box inside ComfyUI. Doubles as phase 5 (contest rule 1, "built with ComfyUI") and the speed benchmark.
Deadline 31 Oct 05:00 NZDT.

## Decisions (user, 2026-10-26)
- Entry name: **MOTNGRFX** (not "Mograph"; not a "music video" — a motion graphic, vertical or horizontal).
- Footage motion affecting the type (`flow_wake`) = most successful effect -> woven into EVERY generation by default.
- Masked Motion-Track: downward `rain` on a person reads as collapsing -> not used. Try a reversed style `rise` (threads
  stream UP off the subject, accelerating, thinning at the top) in tools/fusion/mask_tracks.py at the next box session.
  "Flow lines around the subject looked great" — answered in the reel review: motion trails around subjects in general
  (MTunravel, P06), never downward on people; see Visual decisions.
- Music: commercially clean only (Song2 = CC BY-NC -> not in the entry). Exact-length route: composed timing skeleton at
  the exact frame count + ACE-Step 1.5 audio-to-audio (MIT); known beat grid, no beat tracking.
- Priority now: core system clean + efficient, app in good shape; visual tuning later (after user review of artifact-017).
- Repo: GitHub repo -> rename to MOTNGRFX (pending user go-ahead); local folder / project id stay `motion-graphic`.
  Keep the repo PRIVATE until submission. Before going public: scan the full git history for secrets (API keys, HF
  tokens), confirm `_inspiration/` and licence-restricted media were never committed, clean-clone install test; then
  make public (user's go) and submit by 31 Oct 05:00 NZDT.
- Order: build the node pack + app first, then ONE box session benchmarks it (per-node timings). No box spent timing the old piece_runner path.
- Inputs (Run panel): words (multiline), scene prompt (empty = written from the words), optional image upload,
  optional audio upload, music prompt, length (8/12/16/24 s), look (auto + module families), seed, quality (draft/final).
- Outputs: final 9:16 MP4 with audio + previews (still, type-pass preview) as stages finish.
- UX: like comfy.org/hub/apps/openjutsu (screenshots `16_app/ref/openjutsu_*.jpg`); inputs/outputs are ours.

## Visual decisions (user review of reel artifact-017, 2026-10-26)
Source: `_info/artifacts/artifact-017-fusion-reel/db/` (overall/main.json + reviews/*.json). Liked: P01, P02. Called most
successful: P06, P10. No notes yet on APP1, MTP06s1b, MTrain, MTunravel2, MTpull.
- 2026-10-30 TYPE LANDS STABLE, CENTRED (user: "I want the text to land pretty stable for a beat - centre frame for now as they work the best ... the words can be stacked or rotated 90 or -90 degrees to read vertically - but we should clearly see the transitions happening"): every word forms ON THE BEAT (fully landed on a beat), lands centred and holds still >= 1 beat (motion only in form-in / exit); layouts = one line, stacked lines (no mid-word splits) or rotated +/-90 vertical; transitions clearly visible (>= ~0.5 s), never hidden under a word hold.
- 2026-10-30 RANDOM SYSTEM PER PIECE (user: "set it as random? it chooses transitions and a range of fonts at the start with a limited palette discerned from the imagery"): seeded draw at the start = 2-3 cut kinds + styles from the FULL menu (recent kinds excluded), 3-4 liked fonts, 3-5 colours extracted from the piece's own still; Gemma designs inside it. Wanted more: inverse imagery inside letters on a plain ground, and as a transition (zoom through a letter).
- 2026-10-30 FONT COUNT FROM LENGTH (user: "the length of the clip should dictate the amount of fonts to use - longer ones use more"): n_fonts = clamp(1 + ceil(length_s / 8), 2, 6): 6 s 2, 12 s 3, 24 s 4, 30 s 5, 48 s+ 6.
- 2026-10-30 OUTLINE LIKED: the outline that makes the graphics stand out at points in own-video v002 r1; an option at some words/moments, not on every word.
- 2026-10-30 CONTRAST MOMENTS (user: "use the matte to mask the clean plate into it and have the text in the background - also have the text matte filled with the clean plate and the matte of the main element behind it on a flat background"): `plate_in_subject` = flat ground, word behind, subject silhouette filled with the clean plate; `plate_in_type` = flat ground, matte-cut subject behind, letters filled with the clean plate. 1-3 moments per piece on hits; first test = own-video v003.
- 2026-10-30 MASKING SECTIONS + REQUESTABLE EFFECTS (user, two reference frames: flat black / pale grey ground, letters as windows showing the footage, thin letter outline, matte-cut subject IN FRONT): new mode `window_subject_front` beside `plate_in_subject` / `plate_in_type`, used as a SECTION (one or two word holds, 2-4 s), 0-2 per piece; all are named entries in the per-piece effects menu, and a request field `effects` forces chosen ones in (user: "can we ask for specific effects - can this be a module for it to choose from"). Render side `plate_layers.py` (own-video worker), selection + `effects` field (random-system worker); first test own-video v003.
- 2026-10-30 WORD-TO-WORD TRANSITIONS, NOT FLAT WORDS (user on own-video v003: "just words flat as is not what we want - we should still be able to see the text effects a bit and we need to see clear transitions like how the text goes from A to B", ref = approved g04 `depth_slide_type.py` SEARCHING -> A LIGHT): consecutive words are joined by a designed type transition (g04 lines circuit/lanes/scan, or circuit_board / braid / fan / ripple / spiral_around / twist), >= 0.5 s, no two adjacent pairs alike, even over a single clip; designed word modules stay (v003's finalize had flattened all words to beat_bars).
- 2026-10-30 HOLD LIFE (user: "when the text resolves it shouldnt be static it should have some glitch or pulse or glow or something to keep it alive - it should be legible for a beat but in that beat still be active"): position stays centred and still, the word is ALIVE: per piece 1-2 of letter_flicker / glitch_slice / pulse / glow_breathe / outline_shimmer, on beats / sub-beats; guard: >= 85 % letters visible every frame, a letter off <= 2 frames, never two adjacent off, the landing + on-beat frames clean. `tools/fusion/hold_life.py`, `16_app/HOLD_LIFE.md`.
- 2026-10-30 LETTER GRID + LETTER COLUMN LAYOUTS (user: "text to be able to fill the space - where we split the word like FIRST WORD as SOD / IUM ... and single letters of the word stacked vertically"; refs DR/EA/MI/NG, WI/LD, SA/LT, SEA/RCH/ING): `grid` = mid-word split into balanced rows filling the frame (equal counts preferred, short row centred, words never mixed), `column` = one letter per row. NO-ORPHAN RULE: a split word never leaves a single-letter row unless every row is one letter (S-T / E / A-M invalid; S/T/E/A/M and STE / AM valid).
- 2026-10-30 TYPE COLOUR FROM THE FOOTAGE (user: "why is the text all white? ... can we get colour inspired by the surroundings or maybe contrasted with it?"): 3-5 colour palette from the footage's keyframes; per piece the draw picks harmonious (sampled from the surroundings) or contrast (complement / darkest-lightest); legibility floor ~3:1 against the background behind the word, else the other pick, then the outline; white only when the palette gives white.
- 2026-10-30 TRANSITIONS ROUND 2 (user: "I dont like any of the volume changes really with the circles but I would like to try and create lines at the depth points instead and have them roughly track into the resolve of the planes but they form the words"; "a grid of circles becoming mattes for the transition - blinking off and into existence"): volume_reveal / circle_union dropped; try depth-iso brush lines that track the planes and morph into B's word, and a circle-grid matte that blinks A off / B on (beat / seeded / radial / directional order). Gallery `16_app/core_check/explore_cuts/GALLERY_round2.md`.
- 2026-10-30 SINGLE-SHOT FILL / OVERLAP EXPERIMENTS (user: "different experiments with how you use text and masks and image in different fills and overlaps - just practice/experiment on single shots without transitioning"): 12-16 held-shot variations of text x matte x image (fills from same / other shot / plate / duotone / halftone / depth band, layer order and overlaps, knockouts, inversions, blend modes, outline styles), gallery + side-by-side sheet `16_app/core_check/explore_masks/`.
- 2026-10-30 TRANSITION EXPERIMENTS (user: "being in a volume that is only revealed as you start travelling forward into it - or better you are already in a volume and the translation shows you the parallax of the individual panels - ... some of them as circles - the squares transition into circles with the hard outlines that merge into overlapping circles like a boolean join and we then travel through that in either direction"): experimental styles volume_reveal / volume_parallax_lateral / circle_union (+ artist variants) in `tools/fusion/explore_cuts.py`, gallery `16_app/core_check/explore_cuts/GALLERY.md`; promote the best into cut_kinds after the user's look.
- 2026-10-30 CONTOURS NEVER ON THE FRAME + NO FLAT ROTATING PLANES (user: "I dont like seeing the edge of the frame outlines ... I am more interested in the contours around the things in the image never around the frame"; "we shouldnt see simple flat planes rotating like this in frame - they should be broken up into smaller chunks"): contour strokes only on objects / subject / depth shapes, never on frame or card borders; 3D cuts default to chunked depth-aware fragments, a single-card `smooth` move only on request.
- 2026-10-30 3D CUTS: PLATE MOVES, SUBJECT GLITCHES IN (user on own-video v003 s2: "it should have done the movement transition just with the background and the text coming on with no silhouette and then had the matted foreground element glitch into existence once the frame had settled"): a subject-split cut moves the clean plate (playing) with the text; no blurred hole / silhouette; after the settle the matte-cut subject glitches in (~6-10 frames, slices + RGB split + flicker) and stays; no clean plate -> no split.
- 2026-10-30 SILHOUETTES NEVER FLAT (user: "I dont want frames inside the silhouette to go flat colour", liked v024 r01 ~10 s where the skater's own footage plays inside his matte): any subject silhouette is filled with moving, textured imagery (its own footage, or a clean plate / other clip only if it passes a texture + motion floor), never a flat or palette fill; no default dimming of the subject.
- 2026-10-30 NEVER STILL THROUGH A TRANSITION (user on v024 r02: "the horse goes to a freeze frame then transitions ... it never should go still"): every cut kind plays A's own frames on through the transition and B from its first frame (ping-pong / slow-down at a short clip's end, never a hold); mattes follow the same frame index. pivot_cut `_dolly_render` was a stills version.
- 2026-10-30 CONTOUR SWEEP (user: "I want the contour lines to animate over the forms of the elements it is based on and move on and off in the direction of the transform - they could be then input and output of the transforms"): contour lines (Alpha-Gen matte edge + depth / luma contours) draw on and off ALONG the forms, gated by a front sweeping in the transition's direction; at every transition A's lines draw off and B's draw on from the same side, then resolve onto B's subject edge before B's word lands (read frames clean). Op `contour_sweep`, effect name `contour_sweep`, default animation of the contrast-moment edge line. `16_app/CONTOUR_SWEEP.md`. LINE STYLE (user: "I dont want dashed lines around them like they are a photoshop selection - I want the edge to give a vector and movement for an animated line of differing width and agility and messyness to surround it"): no dashes / marching ants; a live brush-vector stroke with variable width (tapers, thick-thin), agility (flows along the tangent, overshoots corners), messiness (2-3 offset sketchy strokes, smooth wobble, small breaks); crisp, no glow.
- 2026-10-30 PURE TYPE + ANIMATED MATTE EDGE + TEXT-AS-MASK (user on own-video v005: "Can we not use any soft blur behind the type. Just have it pure type"; "when doing masking the background into the mattes can we use an animated outline around the mattes derived from LTX alpha so it creates a cool edge"; "versions where we have the internals of the text be the mask"): no blur / shadow / scrim / dark plate behind type by default (legibility = ink colour + occasional outline); contrast moments get an animated crisp edge line on the Alpha-Gen matte contour (draw-on / travelling dash / beat pulse, no glow); text-as-mask sections = flat ground, letters filled with footage or plate, no subject on top. Grid words max 4 rows and never over the subject's face for the hold.
- 2026-10-30 RANDOMISE LAYOUT PREVIEW (user: "an option on the preview to create a random seed so we can see a preview of the text layout on the single frames and hit a randomise button to quickly create a new one"): in the app's Preview view, a Layout panel = one frame per word at its landed hold over its still; Randomise = new seed re-draws the random system (layouts incl. grid/column, fonts, palette, outline, hold life, cut kinds) on the same words, no Gemma / LTX call, target < 15 s; Use this layout locks the seed for Preview / Final. Node `MographLayoutPreview`.
- 2026-10-30 NO GLOW BEHIND TYPE, OUTLINE ONLY SOMETIMES (user on own-video v004: "I dont like the use of the glow behind all the text - we dont need to outline everything"): no default add_glow on the word matte; glow only as a drawn effect (hold life glow_breathe, low weight); outline on at most ~1 in 3 words (hits / busy background), all-words outline only when requested via `effects`.
- 2026-10-30 FINAL UPSCALE ROUTE = REAL-ESRGAN ON THE FINISHED VIDEO (user, after v016 r01 x4plus / v018 r03 + v019 r03 x2plus at 1080x1920: "these are good enough - that should be our upscale route"): Final = the 540x960 render + RealESRGAN_x2plus (x4plus from 360x640) -> lanczos to 1080x1920, audio/fps unchanged; TiledFusion / Pixel Spatial Upscaler / native-1080 type not needed. Stage `tools/app/final_upscale.py`, card rows, BSD-3 licence line.
- 2026-10-30 TWO TIERS IN THE APP (user: "we dont have draft anymore do we - i thought we have preview and then final?"): the app shows Preview (still-based) and Final only. Final = the internal `draft` sizes for now (LTX 576x1024 -> 540x960, measured ~20 min / 16 s piece) via one constant; switch to 1080x1920 (two-stage final or the LTX Pixel Spatial Upscaler x2 on picks) once tested on the 24 GB box. `draft` stays an internal tier, not in the menu.
- 2026-10-30 FAVOURITE FONTS ONLY, EXPRESSIVE FIRST (user: "some of the fonts used look a little plain and not part of the favourites I want used", then "always use our favourited fonts, not generic ones"): every face on every text path resolves through `mograph/fonts_liked.font_file` (raises, no PIL default / DejaVu / Arial / system font anywhere; test greps the source), ~60 % of a piece's faces are the characterful liked ones, the plain liked ones (SpaceGrotesk, MajorMono) are rare contrast lines, the hero (longest-holding) word wears a characterful face.
- 2026-10-30 HEARTED LOOKS AS DESIGNER CHOICES (user hearts on explore_masks x14 / x08 / x10 / x03 / x02, v027 r02, v028 r01 / r02): `explore_effects` = letters_hold_subject, chronophoto_echoes, outline_to_word (form-in), outline_over_word, plate_letters_subject_over; 0-2 per piece drawn by the random system, requestable via `effects` or a `shot_plan` `fx` key, each a 2-4 s section; words still read >= 1 beat (echoes behind the word, window letters through knockout min_contrast); guide section 24.
- 2026-10-30 ONE SHOT NOW, ON-BOX REVISIONS LATER (user: "maybe add revisions later (even though I liked the first one the best) but those revisions all need to stay on the box - not claude or any api in any way"): tests are one render per variant. ONE SHOT IS THE DEFAULT (user: "revisions arent default - one shot is default with the option to tick on revisions - but that can be a later v02 of the app but keep it in the code but not exposed"): the `auto_revise` code lives in the repo but app v1 does NOT expose it (no UI tick, no graph input; internal flag only, default False); app v02 adds the tick. The revision step: box review sheet -> Gemma 31B vision (MographGGUFVL) checks legibility / on-beat landings / holds / visible transitions / matte fit -> schema-limited patch (timing, placement, cut choice, grade) -> re-render; HARD REVISION LIMIT (user: "it should have a maximum revision limit"): `max_revisions` default 2, ceiling 3 enforced in code (a higher value is clamped, never honoured), the loop also stops early when Gemma reports nothing to fix; r1 always kept, unique names + Gemma's why, replies archived. No Claude or any API.
- 2026-10-30 ~17:30 DEPTH-LINE SCALE LIKED (user on explore_cuts `round2_r3/r3_dl_window_lighthouse_direct.mp4`: "i like the line scale on these"; earlier on r2: "transition straight to the text not to an intermediary outline ... the lines should be thinner but still varied"): line/contour-to-word strokes at `line_w` 0.006 x W, thick-thin `var` 1.25, `direct=True` (no outline stage; letters fill as their strokes land). Default for any promoted depth_lines / contour-to-word style.
- 2026-10-30 ~19:00 OUTLINE MORPHS STRAIGHT INTO THE LETTER OUTLINE (user on explore_cuts `promoted/promoted_depth_lines_horse.mp4`: "I actually want the graphic outlines to transition straight to the text outlines ... not through the lines just by themselves then the text"): depth_lines cut and contour_morph have no free-lines stage; each image-form outline morphs continuously into a letter outline (point-for-point / SDF), then the letters fill and the outline thins away.
- 2026-10-30 ~19:05 CIRCLE GRID WORD APPROVED as a transition (user on explore_cuts `promoted/promoted_circle_grid_word_lamp.mp4`: "this is great lets add this as a transtion"): cut kind `circle_grid` with the word form locked to that look (B's word built from dots blinking on inside the letters on the beat, solid word lands); weighted in the random cut draw, nameable by Gemma.
- 2026-10-30 ~19:50 DEPTH-LINES OUTLINE MORPH APPROVED (user on `promoted/promoted_depth_lines_horse_outline_morph.mp4` + `_lighthouse_outline_morph.mp4`: "these are great ... this transition should be added"): THE `depth_lines` cut. Shipping transitions are collected in the dashboard Outputs set **motngrfx-transitions** (id 8cac26e6ef35).
- 2026-10-30 ~20:00 NO CONTOUR SWEEP ON SHOT CUTS (user on `core_check/contour_sweep/cut_s1-s2_sweep.mp4`: "dont use this - just do a simple straight push in or pull out"): contour sweep only on word-to-word moves inside a shot; a shot cut that would have had it is a simple straight camera push in or pull out (no orbit, no rotation, no lines).
- 2026-10-30 ~20:20 TRANSITIONS SET APPROVED (user: "ok lets go with these", dashboard set motngrfx-transitions). Look templates (request `look` families strand/mech/slab/stack/ring/voxel/particles) are unchanged; a chosen family keeps its own word-to-word moves; contour_morph is the default only on look = auto (or when Gemma names it).
- 2026-10-30 ~20:30 CIRCLE GRID RADIAL (user on `explore_cuts/round2_r1/r2_cg_radial_horse.mp4`: "is this in there? ... if its too destructive to add then dont"): order variant of the word-less circle_grid (dots spread from the subject matte centroid) on matte shots; circle_grid_word stays the locked lamp look.
- 2026-10-30 ~22:10 RESOLVE OVERLAPS (user on batch v031 r01, depth_pivot s2->s3: "why does it stop the video between f175-193? the transition should resolve with the contour lines drawing at the same time"): a 3D cut's settle never holds a still (B keeps playing); the subject glitch-in and B's word / contour lines draw DURING the settle, no pause then snap.
- 2026-10-30 ~00:05 TEXT VARIETY MATCHES THE CUTS (user: "just the variety of text choices and transitions can match the cuts"; v028 r03 used orbit:fly on all 4 cuts): no cut kind, word transition, layout or hold-life kind twice in a row; max 2 uses of a cut kind / word transition per piece; the failed-cut fill (design_options.fill_3d / suggest_orbit) no longer defaults to fly. Draw counts unchanged; look-family rule kept.
- Fonts: hard allowlist `mograph/fonts/liked.json` (42, from the fonts artifact); new fonts are added there explicitly. 2026-10-30 user: "always use our favourited fonts not generic ones": every text path (incl. fallbacks, hold life, explore ops, layout preview) goes through one liked.json helper that raises instead of falling back to a generic/system face; a test fails on generic-font usage.
- BEST SO FAR (user 2026-10-29: "the best you've made so far - very promising"): `22_batch-24gb/v009/out/..._v009_r02_video_00001_.mp4` (Gemma 31B design, LTX: one silver car from 4 framings, slats module, neon palette, depth_pivot / orbit:push / orbit:fly cuts) and `v010/out/..._v010_r01_video_00001_.mp4` (Qwen3.8 design, LTX: skater, 4 places, slats, single words, orbit push / fly / pullback). Code state = round 3 (clear text + designed cuts + no chaining). Every output is now labelled `<rNN>_<gemma>_<ltx>`.
- Font LIKED as drawn (user 2026-10-29 on `24_clean-plate/work/c_w_independent.mp4`: "I really like that font"): Anybody (`mograph/fonts/Anybody[wdth,wght].ttf`) at its DEFAULT variable instance (thin), plain flat white, centred, between plate and subject. Offer it as a go-to clean reader.
- Clean plate + independent subject layer LIKED (user 2026-10-29 on `24_clean-plate/work/c_k_independent.mp4`: "this is way cleaner"): LTX clean plate of the shot + the subject (clip x its own matte) moved/scaled independently, type between them; make it a route the design can choose (or always-on if cheap). Next: separate elements per named object (SAM 3 masks splitting the Alpha-Gen matte).
- No last-frame chaining (user 2026-10-29: "we should either be doing different shots completely or a single long prompted video creation"): multi-shot = every shot from its own still (fresh MOSS still or a MOSS angle edit of an earlier place); a continuous moment = one long LTX generation from one prompt; never `last:sK`.
- Same subject, new angle as an OPTION (user 2026-10-29: "I like what you are getting with moss edit - I don't mind if the person is slightly different but if we can keep them consistent someway that would be great - for multi shots we don't always have to use the same image from different angles but it could be effective as a choice that is available"): a place may be a MOSS edit of an earlier still from a new framing (23_moss-angles v002: anchored prompt, Moss-Image-Edit format); different cameras per shot stay required either way.
- Camera varies per shot (user 2026-10-29: "when we do multi shot I'd like to have the camera be quite different on each shot"): every shot of a multi-shot piece gets a distinct framing + move (wide / close-up / low / top-down ... x static / push / pull / pan / orbit / track / crane); no two consecutive shots share framing or move; continuations keep the place but change the camera.
- Openings vary (user 2026-10-29: "why do all the compositions start with the graphics coming in with a similar transition from the top?"): every word names how it enters (`form_in` + `entry` side up / down / left / right / in / out), no two words enter the same way, a piece uses >= 2 io styles,
  word 1 never repeats the opening of an earlier piece of the batch (`tools/fusion/openings.py`; the design prompt lists the earlier openings); `lines` from the top is the engine's silent default, not a design.
- Matte follows its video (user 2026-10-29: "if we are morphing anything that has an alpha we need to do a seperate morph on the apha that matches that morph - so we dont get double ups where we see two people"): a subject matte is applied only to the picture it was cut from, frame for frame; any transform of that picture (3D cut, pan, crossfade, warp, V2V regenerate) is applied identically to the matte, or the matte is re-extracted from the result.
- Clear text and transitions are the default (user 2026-10-29: "can we get more of this stuff happening - clear text and transitions?", pointing at `16_app/core_check/transition_cut_check/out/`): holds are clean (no smear / echo /
  flicker / particles, flow_wake fully settled in a hold), ONE type system per piece (<= 2 modules, 3 faces, one treatment family), every word >= 9 % cap height, words are FLAT unless one short key word is meant as
  material letters, a designed cut (3D kind or window_other) at EVERY place change and >= 2 places per piece. Enforced in code (`tools/fusion/design_defaults.py`, `design_options.plan_cuts`), told to the models in guide sections 20-21.
- The renderer never invents text (user 2026-10-29: "LTX or whatever renderer should never invent its own text - it should be based off a canny or depth of the graphic with the words told to the model so it knows what it spells"):
  an LTX word is a depth / canny / union control render of THE GRAPHIC (the pass follows the face's stroke weight) and the prompt spells exactly that word in the graphic's look; the rendered text and the flat graphic are ONE version
  (flat until `ltx_at`, a <= 0.25 s crossfade, never both); every shot without a control pass names no text and its negative prompt rules it out (`tools/fusion/text_guard.py`).
- Shot transitions (user 2026-10-29, on `20_open-model-design/v001/out/..._design_llm_8b.mp4`: "worked really well with the choice of
  separating out the main subject against the transforming background"): in a 3D shot transition (depth pivot / dolly) the SUBJECT stays
  crisp and in front (matte) while the BACKGROUND transforms; the type lives between them; word A out, word B in and the strands share ONE
  direction. Multi-shot pieces: a different transition at each cut (user: "transition them in different ways"); the 3D transitions
  (v020 g04 depth pivot, v004 depth dolly down/right) are options between shots. Directions vary too, incl. in/out (dolly
  up/down/left/right, camera push / pullback / fly, pan). Text filled with the FOLLOWING shot is a liked option
  (`16_app/core_check/window_cut_check/out/window_other.mp4`, `window_cut` mode `other`).

Timing / music
- Pace every transition from the tempo. Cuts and pulses land on beats, correctly syncopated (P06 pulses were off-beat).
  Slow/filmic pieces get slower transitions. Current transitions are too fast in nearly every piece.
- Read the tempo reliably from the settled music before the video stage (the known beat grid from the ACE a2a route).
- Hold: once a line forms, hold it long enough to read, then deconstruct. (2026-10-29: the hold is fully settled, no leftover movement, echo or smear: clear text first.)
- Audio sounds small/tinny (P04 asks whether that's the step count): fix. Piano is always accompanied by a pad or strings;
  orchestral for landscape imagery (P08).
- Theme: one chosen theme drives the music, visuals and words; push it harder than v002 did.
- Mood for the transition pace comes from the generation prompt (or footage analysis if the prompt doesn't carry it).

Type in the footage
- Weave type into the footage: hidden between elements (flowers, wheat, folds), in front of and behind the subject at once
  (P06 runners; P01 behind the subject, in front of the rock), then coming forward with a hold to read. Keep mattes
  mostly solid; use a grayscale matte only for sheer/transparent material.
- The bar is v002 P10 (user: "I want ones as interesting as this"; P12_g02 = "very generic ... just one after the other").
  Principles taken from P10 (learn, don't copy): type is a WINDOW (letters reveal a second image / the footage inside,
  flat ground outside); type built AROUND the subject at its scale (letters stacked along the body, front/behind); the
  ground/contrast flips per section; transitions share the type's graphic language (strands/lines matching outline
  letters), never generic fills; letters fill the frame, negative space is designed. A caption-style line over
  footage fails the bar.
- CONTOUR PORTS = the core idea (user, 2026-10-26: "all of these have an entry and an exit and that is what I want to
  join up with contours in our videos"). Every approved gallery module's `entry_bundle()` / `exit_bundle()` and every
  transition's a_exit -> b_entry joins the footage's contours: tracked contour bundles (subject matte outline, depth
  iso-lines, edges; moving with the plate) become valid entries (word forms out of the contour), exit targets (word
  dissolves onto it) and mid-points of transitions (strands ride the contour between words). Built once in the module
  core (Bundle/Port/adapt), GPU-native, so all 24 modules / 13 chains / 5 io styles join the video unchanged.
- Contour-born transitions (user, on P12_g03): transitions must be CREATED FROM THE CONTOURS OF THE IMAGE and feed into
  the creation of the next text — words dissolve into the footage's (tracked, animated) contours, travel along them,
  and the next word forms out of them. Not "generic type on top -> twisty chain graphic -> next generic pulsing type".
  Use the whole module library to the max (strand/mech/slab/stack/ring/voxel/particles modules, mathtype renderers,
  transitions, contour_lines/type_from_edges, flow_wake, cam track, masked trails): each piece combines several.
- Type behind elements wherever possible (user, on P11_g03: "better, but it should use text behind elements where
  possible to integrate it with the video"): every line sits behind at least one real element of the plate (subject,
  rock, wave crest, lanterns, reeds, ridges) for part of its life, via sandwich/depth_occlude/knockout; `legible`
  keeps holds readable. Menu of everything available: `16_app/MODULES_MENU.md`.
- Contrast between lines (user): consecutive lines always differ — contrasting face family (tall vs puffy vs mono vs
  ornamental), scale (huge vs small), weight, fill vs outline. Never two lines in a row in the same treatment.
- Both formats (user, 2026-10-26): every piece must work vertical (9:16) AND horizontal (16:9). Each format gets its own
  generated still (reframed for that aspect, never a crop of the other) and its own LTX shot; the type is re-laid out
  for that frame (line breaks, safe area, legibility) and checked, same song + beat grid.
- Legibility (user, on P15_g01_draft): a matte must never hide a word's letters for the whole time the word is
  showing. Occlusion is for forming/exits and for things passing in front briefly; during the hold every letter is
  readable (type comes to the front, or the occluder clears). Enforced by an automatic per-letter visibility check.
- Flat masking and masking inside the text (P10) in more pieces. Idea: contour off half the frame, fill it with flat
  colour, run the type across it. Colour too, not only B/W.
- Contours (new GPU capability): animated contour outlines that track the subject, used as type in/out paths. Type
  decomposes into the contours and travels to the next line (P08 dissipating into the mountains is the model). Wrap
  type over land contours with a light pulse on the rhythm.
- Simple, fast camera track on the generated footage so type moves with the scene (P09).
- Motion trails/flow lines around subjects: yes, more (MTunravel, P06). Never downward on people (use `rise` or another
  flow field). Don't run trails and footage-flow-on-type at the same moment (P06v3: one or the other). Also try lines
  drawn over the flow-warped background and scribble lines around matted objects.
- Motion artifacting on the masked / inverse-mask areas (LTX masked motion tracks).

Transitions
- Vary them; no in-between transition repeats within a piece (P01's twist did). Plain tilts or simple fades as the main
  animation read as boring (P03, P05, P09); a fade is fine occasionally.
- New: letters expand and slow-glitch-fade into overlapping outlines, like drops in a puddle, until off screen (P01).
- P07 idea: whoosh into the particle cloud -> fast spin-up reveals the words -> slow-mo spin past camera, readable ->
  whoosh out. Particle streaks glitch black into the background or distort it in thin horizontal lines.

Layout
- No title / BPM caption on any piece.
- Line breaks: never leave a single letter alone on a line (AN-D, HO-U-RS, BL-O-OM, GO-I-NG) unless every line is single
  letters.
- Type stays readable in the font (P03's sand formation became too realistic).

Footage
- One continuous generation per shot: a single camera sweep, no FFLF, no blending clips together (P02, P07, P09).
- The background always has some movement (P04: fabric blowing softly).
- P10 stop-start playback stuttered: speed changes only on beats, or not at all.

## Openjutsu UX (reference)
- Dark ink canvas with faint dot grid; floating rounded left panel (~210 px) with stacked inputs; top-left app name + BETA pill;
  top-right GitHub + Download buttons.
- Inputs: dashed drop zones (icon + title + one-line hint "MP4 or MOV · up to 15 s is used"); once filled they show a thumbnail
  with filename · duration and an x to clear. Optional textarea with an example placeholder. Compact dropdown (resolution) and
  seed field with a dice button.
- Primary button full width, yellow: disabled text says what's missing ("Add a video to start"), enabled = the verb
  ("Swap character"). Under it: cost line + small grey note.
- Empty canvas: "Start with an example" card, 2x2 grid of example videos with the input image inset top-right; click loads
  the example's inputs.
- Result: big player in the canvas, yellow play/scrub/fullscreen/mute controls; zoom % control bottom-right.

## Architecture
- Node pack = this repo (ComfyUI-Mograph, comfy_api v3 `io.ComfyNode`, `nodes.py`). Front-end served BY the pack at
  `/mograph` (aiohttp route on PromptServer), so the same page works on local ComfyUI and on the box. It uses the server's
  own `/upload/image`, `/prompt`, `/ws` (progress, executed previews), `/view`, `/history`.
- The page loads an API-format workflow template and fills nodes by `_meta.title` (`IN words`, `IN image`, ...); outputs are
  read from nodes titled `OUT final`, `OUT still`, `OUT type`. Swapping the backend graph never touches the page.
- Templates: `app/graphs/local_v1.json` (existing CPU v1 nodes: Analyze -> Plan(Rules) -> Soundtrack -> Render; runs on
  this PC today) and `app/graphs/fusion.json` (phase 5 nodes + LTX/ACE/alpha/depth, box only). Quality=draft on the
  fusion graph swaps LTX footage for a still/cached clip so type + compose can be developed locally (Quadro P600 4 GB:
  no LTX/ACE here).
- Also export an App Mode workflow (ComfyUI front-end >= 1.41.13) from the same graph as a bonus.

## Speed (baseline: 15_fusion/v002 state.json, 10 pieces)
- wall min: shot 572 (gpu_s 164, includes queue), alpha 282 (80), local CPU type+compose+assemble+final 583, depth+stills 121.
- gpu_s sums to more than the box lifetime, so it includes queue waits; true per-node GPU time is unknown -> the box
  benchmark reads ComfyUI history execution timestamps per node.
- Levers: everything on the box (no scp/PNG round-trips; tensors between nodes), compose on CUDA, NVENC, models kept
  resident / ordered to avoid reloads, fp8 LTX + sage attention, draft vs final resolution.
- Audio fix ("tiny" — user overall note on artifact-017): mastering pass in the graph (`mograph/mastering.py`).

## Box log
- 2026-10-26: box 57698102 (RTX 5090, UK, $0.735/h, machine 144208) launched for the P01 masked Motion-Track probe
  (`16_app/mask_ltx/`). Destroy after the probe unless the node-pack benchmark is ready to run in the same session.
  DESTROYED 2026-10-26 after masktrack v001/v002 probe, app box test and P06 v003 re-render. Lifetime ~85 min
  (42 min of it provisioning), ~$1.08 + $0.23 setup.
- 2026-10-26: box 57782175 (RTX 5090, Portugal, $0.576/h, machine 48379, auto-launch job al_5b67652e84) launched for
  fusion v004 (5 new pieces from the reel-017 notes). Destroy when the 5 finals are down.

## Box test results (2026-10-26, box 57698102)
- App end to end on the box (`app/smoke_local.py --port 9010 --graph fusion_noqwen`): words only -> 11.7 s 1080x1920 MP4
  + AAC in 217.5 s (ACE ~8 s, plan ~3 s, type ~9 s, LTX 3 two-stage shots 87 s sampling + 48 s decode,
  depth 11 s, assemble 5 s). Output `tests/out/smoke_local_final.mp4`. No Qwen node on the box (DesignLLM missing) ->
  test used a no-brief variant; the brief is optional (rules fallback).
- Linux packaging fixes needed for a clean install (put in the README setup):
  - pack must include `tools/mathtype/`, `tools/ltx_graph.py`, `11_mathtype/engine/` (cam3d imports `ol`),
    `14_modules/library/manifest*.json`; pip extras `fonttools soundfile contourpy symusic`.
  - `11_mathtype/engine/dimtype/ol.py` FD was a hard-coded Windows path -> now relative (fixed).
  - after adding files, restart ComfyUI (a failed import caches the missing path).
  - ComfyUI flags: `--mmap-torch-files --cache-ram 16` (without them ComfyUI was OOM-killed at LTX load on 62 GB RAM,
    with ACE/LTX/Gemma all held in RAM) + `--use-sage-attention`.
- Masked Motion-Track: see `16_masktrack/` v001/v002 verdicts (rain d1.0 works; in-graph full-res float32 recomposite OOM).

## Work packages
1. Fusion node-graph design (read-only design pass) -> `16_app/DESIGN.md`.
2. Front-end app + local v1 template, pack installed into local ComfyUI (`Dev/ComfyUI/custom_nodes`).
3. Fusion nodes (type passes, compose, assemble, music trim + mastering) per DESIGN.md.
4. Box session: install pack, run fusion graph on P02 + one heavy piece, per-node timings -> `16_app/BENCH.md`.
5. Demo video + repo setup notes + post.

## Core pipeline pass (2026-10-26, WP core)
- **Footage flow in every generation.** `MographFootageFlow` (gather -> flow -> compose in both graphs): RAFT on CUDA, DIS on CPU/draft or when RAFT
  fails; clips stay uint8, flow is fp16 at half res and rides on the clip (`clip["flow"]`). `plan_rules.weave_flow` puts one `flow_wake` per type alpha
  (`$type_matte`, `$type_line`) in front of each recipe chain (built_in / behind / traced); influence = motion matte derived from the flow itself
  (`$flow_m`, `recipe_eval.FlowMatteSource`, camera drift = median flow removed) so it needs no Alpha-Gen; `settle` follows the planned (hit, release)
  word holds. The masked Motion-Track "rain" is not used (masktrack nodes stay in the pack, not in the graphs). Tuning later (WAKE in plan_rules).
- **Format.** `IN format` (9:16 | 16:9) -> request.format -> `piece.format` / `piece.sizes` (`plan_rules.sizes_for`): final 1080x1920 / 1920x1080,
  LTX stage 1 576x1024 / 1024x576 (x2), type layers 1088x1920 / 1920x1088, draft halves. `MographPieceSizes` feeds the LTX / alpha / still size
  primitives; ShotSlot, DecodeClip, Gather, FirstFrame, TypePasses read `piece.sizes`. Landscape type scenes build with no notes for modules of every family.
- **One graph with or without Qwen.** Sidecar `optional_nodes` (`tools/app/optional_nodes.py`, applied by `app_routes.load_graph`): nodes of a missing
  class are removed and the links to them dropped. `fusion_noqwen` is no longer needed. `--check` verifies the pruned graph.
- **Tail / length.** Planner fills the tail (hold / exit grows until the last word ends <= 0.9 s before the cut; the signoff starts as it leaves
  and runs to the cut; px 38). Music = exact-length skeleton (16_app/MUSIC.md): output duration == requested length (draft smoke: 12.000 s).
- **Efficiency.** Gather builds the float32 alpha / depth reference videos only for shots whose recipe reads a matte / depth (else a 16x16 placeholder);
  flow is fp16 end to end. Box flags (also in the sidecar `server_flags`): `--mmap-torch-files --cache-ram 16 --use-sage-attention`.
- **Stills.** MOSS-Image (Apache 2.0) is the default still chain.
- **16:9 draft crop (fixed 2026-10-26).** The type layers were always laid out for landscape (1024x576, world 960x540, 2-3 line wide stacks, inside a
  >= 5 % margin at every hold; `test_type_alpha_stays_inside_the_safe_area_at_every_hold`). The cropped "AF/TER" in `draft_16x9.mp4` came from the draft
  stand-in footage (`mograph_cache` clips, portrait P02 LTX shots with the neon type baked in, cover-cropped into 16:9). The stand-in footage was removed 2026-10-30
  (the quick first look is now the Still preview, `16_app/STILL_PREVIEW.md`).
- Checks: `tests/test_fusion_core.py`, `python tools/app/build_graphs.py --check [--server ...]`, `16_app/core_check/` (A/B wake sheets, draft 9:16 / 16:9 mp4s).
