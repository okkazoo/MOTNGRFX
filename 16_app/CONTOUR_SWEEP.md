# CONTOUR SWEEP (2026-10-30)

**UPDATE 2026-10-30 (late): NO SWEEP ON SHOT CUTS.** User on `16_app/core_check/contour_sweep/cut_s1-s2_sweep.mp4`: "don't use this - just do a simple straight push in or pull out". `contour_sweep.plan` / `weave` now make no event for any
3D / pan / window shot cut (`SHOT_CUT_EVENTS = False` in `tools/fusion/contour_sweep.py`; tests `tests/test_contour_sweep_no_shot_cuts.py`). The sweep stays ONLY on word-to-word transitions inside a shot (where `contour_morph`,
ship #8, owns the gap when the word has a matte). Shot cuts that want a plain move use `orbit:push` (straight push in) / `orbit:pullback` (straight pull out): one rigid camera along its axis through the depth layers, zero
orbit / yaw / pitch / roll, no mid-way rise, no lines, chunked reveal of B like the other 3D cuts; the random cut draw weights them up. Everything below that describes cut directions is the disabled path.

User: "I want the contour lines to animate over the forms of the elements it is based on and move on and off in the direction of the transform - they could be then input and output of the transforms".
Line style (same day): "I dont want dashed lines around them like they are a photoshop selection - I want the edge to give a vector and movement for an animated line of differing width and agility and messyness to surround it".

Code: `tools/fusion/contour_sweep.py` (pure plan + compose op `contour_sweep`, registered in `compose._ops_registry`), tests `tests/test_contour_sweep.py` (26), check `16_app/core_check/contour_sweep/` (`make_check.py`, `strip_shot_cut.png`, `strip_word_transition.png`, the two mp4s, `plans.json`).

## What is drawn
The contours of the shot's Alpha-Gen subject matte (iso 0.5; `contour_ports.matte_outlines`, re-traced on every frame, so the lines follow the forms; the chain's own matte is used, so in a 3D cut the lines take the clip's transform, "matte follows its clip"). Optional `sources`: `depth` (depth iso-lines, `contour_ports.iso_lines`), `luma` (iso-lines of the picture luminance); a shot without a matte falls back to luma contours.
Every contour is `n_strokes` BRUSH STROKES, never dashes (anti-aliased filled ribbons, crisp, no glow, no blur, palette colour):
- width: base x (0.3 .. 1.6) from smooth noise over the arc length (drifting with time), tapered ends (pressure-like). Base 2 px at 360 wide (scales with the frame width), so about 0.6-3.2 px at 360;
- movement: the visible run travels ALONG the contour and grows / shrinks with the sweep; the lead end runs ahead of the tail (different easings); a short thin tip overshoots the front; stroke j starts `stagger` later than stroke j-1;
- messiness: 2 strokes (1-3) per contour, offset from each other, a low-frequency wobble off the edge (`wobble` = 1.8 % of the subject size, smooth over arc AND time), occasional small breaks. Seeded: (piece seed, event, contour, stroke).

Params (`PARAMS`, override with `compose.contour_sweep = {..}` or `plan(..., params=)`):
| key | default | meaning |
|---|---|---|
| `width` | 2/360 | base stroke width, share of the frame width |
| `width_range` | (0.3, 1.6) | multiplier band along the stroke |
| `n_strokes` | 2 | strokes per contour (1-3) |
| `wobble` | 0.018 | offset off the edge, share of the subject size (use 0.01-0.03) |
| `speed` | 1.0 | drift speed of width / wobble noise and of the idle runs |
| `gaps` | 1.0 | break rate multiplier (0 = none) |
| `max_contours` | 4 | largest contours drawn |
| `sources` | ("matte",) | matte / depth / luma |
| `taper` | 0.16 | share of a run that tapers at each end |
| `stagger` | 0.14 | later start of each extra stroke (share of the phase) |
| `overshoot` | 0.035 | lead tip past the front |
| `border` | 0.02 | no stroke within this share of the frame border (+ 1.2 % margin for the wobble; hard clip at 2 %) |
| `straight` | 0.25 | a near-straight run longer than this share of the frame side is a frame / card edge: not drawn (0 = off) |
| `straight_dev` | 0.005 | 'near-straight' = within this share of the side of the run's chord |
| `guard` | True | False = every contour point (used for letters: plate_layers `text_as_mask`) |

## Never the frame, never a card (user 2026-10-30 on the orbit:fly cut: "I dont like seeing the edge of the frame outlines ... I am more interested in the contours around the things in the image never around the frame")
`reject_mask` drops, per resampled contour: (1) points within `border` of the frame border (a matte cut by the frame has its contour there; the stroke buffer is also hard-clipped in the border band); (2) every near-straight run longer than 25 % of the frame side (a 3D-cut card edge, a frame line), dilated by 3 % of the side, then any short stub left between two rejected runs (the card / frame corner); (3) points within 3 px of an explicit card quad edge when the op gets `quads` (corner arrays in frame px, not supplied by the cut engine today). Contours are fetched 2 x `max_contours` (largest first) and the first `max_contours` that still have drawable points are kept, so a card-sized contour does not use a slot. The subject extent that scales the sweep front is measured on the drawable points only. In a 3D cut the matte is the cut's patched matte (the object warped with its card), so the strokes go around the objects inside the moving cards; only where that matte touches the card border is the border run removed. Contour sources stay matte iso, depth iso-lines and luma iso-lines (the last two only when asked, or when the shot has no matte).

## How it sweeps
The transition has a direction `f`. A point has s' = its position along `f` scaled to the subject's own extent (0 entry side, 1 exit side) and u2 = its arc distance from the contour's entry point (the two arms of a closed contour grow towards each other).
- OFF (A, first half): drawn while s' > F(t) and u2 > G(t): the lines are consumed from the entry side, in `f`, along the forms (alpha ramps in over the first 12 % so nothing pops).
- ON (B, second half): drawn while s' <= F(t) + tip and u2 <= H(t) (H leads F): the lines grow from the same side.
- RESOLVE: the full contour, the extra strokes fade first, the wobble goes to 0 (the stroke sits ON the subject edge), then everything fades to nothing at `END_GUARD_S` (0.10 s) before B's word lands.

Direction `f` = from A's exit side to B's entry side (g04 `depth_slide_type`: A leaves left, B comes in from the right, the lines flow left -> right):
| transition | f |
|---|---|
| `pan:<d>`, `depth_dolly:<d>` | the named direction d (= camera travel; A exits the opposite side, B enters from d) |
| `depth_pivot` | right (cut_kinds.sides: A left, B right) |
| `orbit:push` / `window_zoom` | radial OUT from the centre |
| `orbit:pullback` | radial IN |
| `orbit:fly`, orbit `slide` | its `dir` (default right) |
| `window_cut` / `window_swap` / `window_other` | a seeded side per cut (no direction of its own) |
| crossfade / hard | none (nothing to ride) |
| word-to-word | the words' `exit` / `entry` sides when named, `out`/`in` = radial, else a seeded side (never the same twice in a row) |

## Timing (`plan(piece, seed) -> dict`, JSON, deterministic)
- shot cut: 3D kinds = `pivot_cut.window_frames` (A window = the `i2` frames before the cut, B window = the rest; defaults 72 / 27 when the engine cannot be imported), pan = `plan.pan.boundaries` (the travel is A's draw-off; B lands at its end so there is NO B phase), window kinds = +-0.6 s around the cut (`window_zoom` -0.5 / +0.9; an ESTIMATE, the window engine does not publish its span).
- word-to-word transition (a word's `transition` / `type.transitions`, not `cut`): from A's release to B's hit - 0.10 s; A off 40 %, B on 40 %, resolve 20 %; dropped when under 0.45 s or when it overlaps a shot-cut event (the cut owns its span).
- B phase = cut -> `min(window end, B's landing - 0.10)`; shorter than 0.30 s = dropped (A's draw-off only).

## Legibility guard (tested)
- `clean_windows`: no line from 2 frames before every word's landing to landing + max(1 beat, 0.5 s) (capped at the release), 0.12 s ramps on both sides; hold_life's clean landing / beat frames sit inside it.
- B's lines are gone 0.10 s before B's landing; the `avoid` input (the type alpha) is cut out of the lines, so a stroke never crosses a letter.
- Note: a line may be on screen during the LATE part of a word's hold (after the clean window) when a transition starts there; it never touches letters.

## How it gets into a piece
1. Compose key, NO planner edit: `compose.contour_sweep = true` (or `{..params..}`; `false` = off) or the request effect `contour_sweep` (`effects_menu.EFFECTS`, so `--effects contour_sweep` forces it; it lands in `plan.effects.requested`). `contour_sweep.recipe_ops(piece, ops, sid, shots)` puts the op after the last type drawer (sandwich / depth_occlude / knockout / multiplane / hold_net) of each shot's chain, before the closing grade / vignette, reading the drawer's own matte and `type_a` as `avoid`. Called from `recipe_eval.build_recipe` and `piece_runner.Piece.recipe` (next to the multiplane hook).
2. Planner hook (NOT added by me: the QC fix-pack worker is editing the planner files). One line, in `tools/fusion/plan_rules.py` `_finish` right after the `weave_hold_life(...)` call, AND in `tools/fusion/design_step.py` next to the `PR.weave_hold_life(probe, ...)` call (line ~727):
```python
    from tools.fusion import contour_sweep; contour_sweep.weave(piece, music, chk, size)   # 2026-10-30 contour sweep: on when requested (effects: contour_sweep) or drawn (seeded 40 %); compose.contour_sweep = false opts out
```
`weave` never raises, sets `compose.contour_sweep = True` and writes the event summary into `plan.contour_sweep`. `draw=False` = only when requested. For a piece with no hook the effect still works through (1).

## Contrast-moment edge line
`plate_layers.EDGE_MODE = "sweep"` (default): `bake(..., edge_mode=None | "sweep" | "dash", edge_seed=0)`; the sweep is `contour_sweep.edge_alpha` = the same brush strokes, an idle form (runs flowing along the contour, growing / shrinking) that draws on from the left over the first 12 frames of the section and off over the last 12 (ramps shrink for short sections). `edge_mode="dash"` keeps the earlier travelling dashes (`edge_line_alpha`, unchanged). `tests/test_plate_contrast.py::test_edge_line_is_drawn_on_the_contrast_modes_by_default` was rewritten for this (dash mode keeps the old assertion).

## Check (one shot, no tuning)
`16_app/core_check/contour_sweep/make_check.py` on the box through `recipe_eval.compose_frames` (the real engine; chain = sandwich, then the op put in by `recipe_ops`) on the v026 r01 archive (r03's archive is not on the box / PC; its cut files are not kept): `strip_shot_cut.png` = the orbit:fly cut s1 -> s2 (the engine's own transition footage + its patched matte): 8 frames 2.46-5.04 s: A's lamp outline draws off, B's face / hair outline draws on from the left and resolves. `strip_word_transition.png` = a `lanes` word transition (exit left, entry right) in shot s4, 10.83-12.71 s: the lamp outline draws off, on, resolves, and is gone before the next word forms. Colour: the palette accent, swapped to the dark / light extreme when it sits within 0.28 luminance of the picture under the line (dark on the pale pink plate).

## Limits / open
- A pan has no B phase (B lands at the end of the travel: no room before the clean window).
- Window cut kinds: the window span is an estimate (+-0.6 s).
- Words LTX owns (`render: ltx / handoff`): the lines are not suppressed over the LTX-rendered word's frames (no `avoid` for those); only the flat type alpha is avoided.
- Contours are traced from the matte, not the depth, by default (`sources`); depth / luma extraction was only unit-checked on the matte path.
- The sweep normalises s' to the subject's extent per frame, so a fast-moving subject shifts the front slightly.
