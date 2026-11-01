# MOTNGRFX module menu (2026-10-26, scout inventory)

APPROVED SET (user, 2026-10-26: "all the ones on that page are good"): the Module Gallery, artifact-016
https://claude.ai/artifact/XxYDqUePBVatpR73pvzrwY (archive `_info/artifacts/artifact-016-module-gallery/`): all 24 word
modules, 13 chains (chain_flow_bends_demo/_follow, chain_signal_bends_flow_demo/_follow, chain_signal_circuit_bends_follow,
chain_bends_spiral_flow_follow, chain_signal_braid_bends_follow, chain_bends_fan_flow_follow, engine_chain_1_calm,
engine_chain_3_hype, io_chain_flicker_blobs_twist, io_chain_wipe_particles_braid, ...) and 5 io styles (blobs, lines,
flicker, particles, wipe). Build pieces from these as shown there (the chains with the follow camera are the reference
for how modules + transitions + camera combine). Library builder `14_modules/library/build_library.py`, gallery
`14_modules/gallery/build_gallery.py`.

Purpose: a checklist for piece designers. Every piece combines entries from at least four of sections A-E, and every
word-to-word hand-off is born from the footage's contours (`PLAN.md` Visual decisions: "bar is v002 P10", "Contour-born
transitions", "Contrast between lines", "Type behind elements"). "Type on top + one transition + next word" fails the bar.
Status: LIVE (used in a shipped piece), DEMO (library demo render only), UNIT (unit test), NEW (no piece or test).
History (2026-10-26): the early fusion pieces P01-P15 happened to use one word module; that is a habit of those pieces, not a preference, and every module the code keeps (beat_bars, circuit, gradient_fade, lanes, pipework, slats, staircase) reads equally well on a hold. Each piece is given its own module (DESIGN_GUIDE section 22). No piece has used the type as LTX control (type-driven footage).

## A. Word modules (`mograph/modules/words/*.py`; families `tools/fusion/plan_rules.py:56-64`)
Contract: `WordModule(words, font, placement, entry, exit, timing)` -> `entry_bundle()`, `layers(t)`, `exit_bundle()`
(`mograph/modules/core.py:634`). Spec key `words[i].module` (`tools/fusion/spec.py:285`). All export passes
(beauty/matte/depth/line/clay/flow) via `tools/fusion/type_passes.py`.
- Strand: lanes (hairline lanes peel off, draw outline, fill; LIVE in the early pieces) · pipework (shaded tubes, chamfered bends; DEMO) · circuit (PCB traces, pulse heads; DEMO, pairs with circuit_board)
- Mech (3D): depth_echo (outline copies fan out behind; LIVE P01) · hollow_tubes (letters rise as open tubes) · topo_tower (word on floor, contour copies stack as a topo tower) · hinge_unfold · xray_unfold (14 depth layers unfold) · candy_orbit · depth_tunnel (each letter a shaft into the page = letter-as-window) — DEMO
- Slab (flat graphic): beat_bars (bars swing/jump on beats) · bold_thin_echo · gradient_fade (comet arrival) · scale_clash (huge + tiny word on one axis = contrast rule) · slats (speed-line bars) · solid_hollow (fill/outline alternate) · staircase — DEMO
- Stack/ring/voxel/particles: spectrum · tangent_spray · fly_into (letters are tunnels, letter-as-window) · outflow (outline rings stream in; RETIRED 2026-11-02 by the user, still registered, never offered or kept) · voxel_orbit (strands condense to cubes) · voxel_roll · particles (2000 streaks land on glyph sites; the P07 "whoosh") — DEMO
- Form-in/out styles (`mograph/modules/io/`): blobs / flicker / lines / particles / wipe (alphabetical; none is the default: every word names its own, and `entry` = the side the strands of a `lines` word come from; `tools/fusion/openings.py`). flicker and wipe carry no strands (Sequence swaps the transition for `cut`).
- Cameras (`mograph/modules/camera/*.py`): follow, pullback_reveal, hold_drift (LIVE P08/P09), orbit_even, spin_read_spin, push_through, tilt_to_plan, graze, ease_reversal. These move the type world, not the plate (plate: cam track, section D).

## B. Transitions (`mograph/modules/transitions/*.py`)
Contract: transition.layers(0) draws A's exit bundle, layers(dur) draws B's entry bundle; `adapt(a, b)` (core.py:411) pairs
strands. Bundle = `P (N,K,3)` polylines + direction + colour + width + speed (core.py:310). Spec `transitions:` (spec.py:299-312).
- twist (cable twist; LIVE P13/P14; UNIT) · braid (three plaits) · fan (DEMO only) · spiral_around (strands spiral round an object; object is a free parameter -> could be a footage subject) · circuit_board · ripple (puddle-ring outlines; UNIT) · cut (hairline flash)
- pan (v006, hidden from the engine pool): the camera travel IS the transition, 4 directions (camera down / up / right / left); A holds, the `pan` camera (`camera/pan_dir.py`) travels one frame-span, A leaves as a world-locked object, B arrives as it settles on a downbeat; any word module, io styles at the frame edge; the plate travels with it (compose op `plate_pan`). Spec `"transition": {"kind": "pan", "dir": "down", "beats": 2}` on a word, or camera `{"move": "pan", "dir": ..}`; block `camera_pan_transition`. See `mograph/modules/README.md` (Pan transitions).
- depth_pivot (2026-10-27, hidden, IMAGE transition, not strands): the depth pivot-chunk look approved in 17_depth-dolly v013 as a module + ComfyUI node + lego block; see section B2 below. `mograph/modules/transitions/depth_pivot.py`, node `nodes_fusion/depth_transition.py`, block `depth_pivot_chunks`.
- Rule: no repeated transition within a piece.

## B2. Depth pivot-chunk image transition (`depth_pivot`; node MOTNGRFX_DepthPivotTransition; block `depth_pivot_chunks`)
What it is: two shots become depth scenes of chunky camera-facing cards (the subject = ONE cut-out card, the background = `chunks` cards), one camera pivots round
A's volume, B swings in over the top turned 45 deg and lands flat (quintic, no spring, continuous per-card blur). Engine: `tools/fusion/depth_orbit.py` +
`depth_dolly.py` (not edited); the module is the adapter. Frame 0 = A exactly, the settled last frame = B exactly. CPU only; ~170 s for 72 frames at 432x762 on
a laptop CPU (render it on the box for anything bigger; never locally for finals).
- Node inputs: `image_a`, `mask_a`, `image_b`, `mask_b` (MASK = the Alpha-Gen matte output, one mask per frame), optional `depth_a` / `depth_b` (IMAGE, bright =
  near; empty = Depth-Anything-V2-Small on the CPU from the first frame), widgets `chunks` (default 32), `video` (auto), `in_angle` 45, `out_angle` 40, `travel` 36,
  `hold1` 12, `hold2` 24, `depth_ratio` 6, `blur` continuous, `subject_card`, `short` 432, `seed`, advanced `size_lo` 1.5 / `size_hi` 4, `spring`, optional `params_json`.
  Outputs: `frames` (IMAGE [N,H,W,3] float, N = hold1 + travel + hold2 = 72) and `params_json` (STRING, the full resolved parameters).
- LIVE VIDEO IS THE DEFAULT (user rule 2026-10-27): an image batch with more than one frame is a clip and plays live on the cards (A from output frame 0, B from the
  frame it first becomes visible; a finished clip holds its last frame), the mask batch is the per-frame subject matte. A single image is a still, used only when no
  clip exists. Always pass the shot's motion clip + its per-frame mattes when they exist. `video` = on / off force it. The block's `needs.prefer` says the same.
- Density: ONE control, `chunks` = background cards per image (+ the runner as one card). 32 = v009's look (card scale ref = sqrt(W*H/50) x U(1.5, 4) = ~124-419 px at
  432x762); the card scale follows the count (engine max_chunks = round(chunks x 50 / 32), bg_chunks = chunks), so 100 = smaller cards, full coverage.
- Reproducible: seeded (`seed`), and `params_json` out is the complete resolved dict (also the engine arguments). Paste it into the node's `params_json` (or pass it as
  `params=` to `depth_pivot()`) to re-make the very same frames.
- Python: `from mograph.modules.transitions.depth_pivot import depth_pivot, depth_pivot_render, resolve_params`;
  `depth_pivot(a, matte_a, b, matte_b, depth_a=None, depth_b=None, params=None)` -> uint8 RGB (N,H,W,3).
- Reproduce 17_depth-dolly v013 g01 (the `prompt.txt` command: `--max-chunks 50 --size-range 1.5,4 --depth-ratio 6 --in-angle 45 --out-angle 40 --no-spring --blur continuous
  --subject-cards 1 --video`, density v009) through the node: image_a / image_b = the s1a / s2a clips decoded as frame batches (73 frames, 1088x1920), mask_a / mask_b =
  their 73 Alpha-Gen mattes, depth_a / depth_b = the first-frame depth (or empty), every widget at its default, `short` 432, seed 0. Differences from v013: the
  engine pipeline is identical; v013's depth came from the cached `v001/local/depth_*_raw.npy` (same model, same first frame), and v013 sized the cards with 50 chunks
  total where the node's default is the v009 count (32 background + 1 runner).
- Block `depth_pivot_chunks` (status partial): opt-in, needs 2+ shots with mattes (`cut` / `chain`), prefers the shot's video + per-frame mattes; slot `image_transition`
  between consecutive shots. Since 2026-10-29 the splice stage exists: `tools/fusion/pivot_cut.py` + node `MographPivotCut` (cut kinds `depth_pivot`, `orbit:push|pullback|fly`, `depth_dolly:<dir>` in `compose.cuts[]`, matte rendered through the transition, `build_graphs.py --pivot-cut`); still missing: the node in the generated graphs and a GPU-box run. All shot transitions: `16_app/TRANSITIONS.md`.

## C. Math-type / procedural renderers (GPU torch/CUDA via xp.py; footage-driven ones already turn the plate's own
contours/flow into type: videotrace E6, tracegeo E3, contours E1, outlinepath E2. Gap = an adapter so they emit type
layers into compose, being built 2026-10-26. Check here BEFORE building any "new" contour/type effect.)
- `11_mathtype/engine/dimtype/ol.py` (glyph contours, extrusion, travelling strands; `split_strands` :196) · `vox.py` (voxels) · `dD_shader.py` (chrome) · `varword/vw.py` (per-letter variable-font word: weight/width vary per letter -> contrast rule)
- `tools/mathtype/`: contours.py (iso fields from a text SDF), silkwind.py / matte_wipe.py (wind/silk trails, MTunravel liked), particles, harmonograph, moire, chladni, growth, follow*, relay, videotrace.py / outlinepath.py (trace video contours, but write their own video), dotfield; GPU via `xp.py`; shared `kit/`.

## D. Compose ops (`tools/fusion/compose.py` `_ops_registry` :1723; recipes `tools/fusion/recipes/{behind,behind_sheer,built_in,traced}.json`)
- Occlusion: sandwich (matte_threshold solid / none for sheer; `front=` puts one layer in front AND behind; chain two) · depth_occlude / depth_cover · legible (pulls flagged letters in front during holds; auto-inserted by build_recipe; UNIT) · knockout (plate with type cut out = type as window; LIVE P10) · window_cut (letters as windows between TWO shots: other / cut / swap, see F11; CPU-checked, `tools/fusion/window_cut.py`) · fill_holes · luminosity_key, upscale_matte, fit_cover
- Contours into type: trace_edges / EdgeTracer (held Sobel edges) · contour_lines (iso/outline/scribble, phase, reveal, stagger; UNIT) · type_from_edges (letters made of edges; P08 dissipation) · edges_in_type (footage lines inside letters)
- Motion: flow_wake (footage flow pushes type, settles; P06 liked) · translate + cam_ops (type follows the plate's camera track)
- Look: refract, tint_type, add_glow, halation, grade, grain, vignette, blends, crossfade, blur, fill (complementary tint + subtle glow = P06 look)

## E. Footage-side generators
- LTX IC control (`tools/fusion/ltx_ic.py` alpha|control|motiontrack): use the type's `line`/`depth` pass as control -> footage grows around the type (NEW)
- Alpha-Gen matte · Video Depth Anything depth · RAFT flow (`footage_flow.py`) · masked motion tracks (`mask_tracks.py`: pull_apart, unravel, rain, rise; never downward on people)

## F. Combos that meet the bar
1. Contour birth -> hold -> contour death: contour_lines(depth iso, phase on beat, stagger) -> type_from_edges forms the word -> hold with flow_wake settle -> contour_lines reveal 1->0 while the next word's type_from_edges ramps in. (Chaining into module strands needs GAP 1.)
2. Window type over a second plate (P10 in colour): lanes/scale_clash lines -> knockout(plate B or flat colour) -> sandwich(matte_threshold, front=) so the subject passes in front of the window letters -> contour_lines(outline) glow on the subject; flip ground per section.
3. Letters built around the subject at its scale, front and behind: matte -> sandwich(front=) + legible -> depth_echo/topo_tower -> add_glow. (Matte-driven placement = GAP 6.)
4. Contour-born hand-off with strands: word A exit_bundle -> "contour" transition whose b_entry is the footage contour bundle -> strands travel along the contour -> word B entry_bundle. (GAP 1, 2.)
5. Mask-track rise/unravel carries the word: masked LTX v2v on the subject -> flow_wake on type from the new flow -> type_from_edges from trace_edges of the deformed subject.
6. Particle/voxel word seeded from the matte outline: outline samples -> entry positions of particles/voxel_orbit -> condense into glyphs; reverse onto the next contour. (GAP 3.)
7. Whoosh through the type (P07 brief): particles + spin_read_spin/push_through; plate cam_ops + flow_wake; thin-line glitch via contour_lines(scribble)/refract tied to type speed. (GAP 5.)
8. Type-driven footage: word A's line/depth pass as LTX control + first frame -> footage grows out of the letters; sandwich/knockout on the same alpha for exact registration. (Box test first, GAP 7.)
9. Topographic stack on real terrain: topo_tower/xray_unfold copies synced to contour_lines(iso, phase) of video depth; beat pulse of light along the outline. (GAP 4.)
10. Alternating treatment per line: scale_clash huge fill -> bold_thin_echo hairline -> solid_hollow / varword; ground flips knockout vs sandwich; every gap a different transition (ripple, spiral_around the subject, contour hand-off, cut on a downbeat).
11. Text as window onto another shot, P10-style (`window_cut` op, blocks `window_other` / `window_cut` / `window_swap`, design keys `words[].window_source` and `compose.cuts[]`, `tools/fusion/window_cut.py` + `design_options.py`; heavy liked face, cap >= ~30 % of the frame, two shots that differ in luma / colour; opt-in, max 2 per piece):
    a. window_other: while a word holds, the letters show the NEXT (or a named) shot, the word's own shot stays outside, the plate's subject steps out over both. Mode of the op where the outside is a flat colour = P10's knockout (bit-identical frames).
    b. window_cut (text-window reveal cut): over shot A the word is a window onto shot B, the letters grow (distance-field dilation) until B fills the frame = the cut; the boundary is a HARD cut, no crossfade.
    c. window_swap (the swap cut): B inside / A outside, ON the cut beat (downbeat, chord change or dropout) they swap (B whole picture, A in the letters), then the letters leave (fade / shrink / io) and only B remains; optional freeze on the swap (P10's freeze-on-dropout); hard cut, no crossfade.

## G. Gaps
1. Footage contours as strand bundles: `contours_to_bundle(matte|depth, frame, n)` -> `Bundle` (contour_lines' cv2 path already has the polylines). Blocks combos 1, 4, 6.
2. Transition whose a_exit/b_entry can be a footage-contour bundle with time-varying positions (contour moves with the subject).
3. Word-module entry from arbitrary points (matte outline samples) instead of a Port.
4. Footage-driven module params (depth/iso heights into topo_tower, depth_echo).
5. Type-pass `flow` wired into plate ops (refract/displacement) so type speed distorts the plate.
6. Auto-placement of lines from matte bounds/regions (lines in free zones, around and behind the subject).
7. Type-driven LTX control path through the app; `rise` mask style (now in mask_tracks.py; verify end to end).
8. Format parity: per-format placement for 9:16 and 16:9 in modules (verify).
9. Tests: only lanes, twist, ripple have unit tests; run a draft hold-check (`15_fusion/pieces/_v004_holdcheck.py`) before using any other module in a piece.

## 2026-10-30 additions (see `16_app/TRANSITIONS.md` section 10)
- Cut kind `window_zoom` (zoom through a letter) = block `window_zoom` (`tools/fusion/blocks.py`), compose op `window_cut` mode `zoom`; flat-ground inverse type = word keys `window_source: self` + `window_ground: plain` (compose op `window_cut` mode `other` with a colour as `a`).
- The cut kinds, the fonts (3-4) and the type palette (3-5 colours from the first still) of a piece are DRAWN per piece (`tools/fusion/random_system.py`, `still_palette.py`); the word modules stay the 7 the legibility fold keeps.
- Type layouts per word: one line / stacked at spaces / rolled +-90 reading vertically, all centred and still (`tools/fusion/stable_type.py`).
