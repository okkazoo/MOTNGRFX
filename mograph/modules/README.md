# mograph.modules: lego type modules

Vector type that forms from nothing, holds, and deconstructs, as interchangeable parts. Everything is `cam3d` Layers in one 3D
world (x right, y down, +z away from the viewer, the 960x540 frame plane at z = 0). The camera is separate.

```
word module  ->  transition module  ->  word module  ->  ...      Sequence([...]).render(path, camera)
   forms words      carries the strands     forms words
   out of a Bundle  between any two points
```

## The hand-off: `Bundle`
`Bundle(P (N,K,3) head-first, D (N,3) head direction, col (N,3), w (N,), t, v (N,) head speed, tube 0|1, rim)`.
Contract that makes parts chain: `A.layers(A.duration)` draws exactly `A.exit_bundle()`; a transition's `layers(0)` draws exactly
its `a_exit` and `layers(dur)` exactly its `b_entry`; `B.layers(0)` draws exactly `B.entry_bundle()`.
`adapt(a, b)` pairs strands (optimal assignment, `scipy.linear_sum_assignment`, so streams do not cross); N != M splits / merges.

## Word module (`words/<name>.py`)
```python
WordModule(words, font=None, placement=None, entry="left", exit="right", timing=None, palette=None, **params)
  words      str ("\n" = new line) or list of lines;  font: liked-font key / (key, loc)
  placement  Placement.centred() (library default) | Placement.at(origin, yaw, pitch, roll, w, h) | Placement(origin, u, v, w, h)
  entry/exit Port: 'left'|'right'|'up'|'down'|'in'|'out', angle in degrees (0 right, 90 up), or a 3-vector. Port(x, spread=0.0).
             The port is the SIDE the strands come from (entry) or go to (exit).
  timing     Timing(form, hold, exit);  .snap(beat)
  entry_bundle()   exit_bundle()   layers(t)  (t in [0, duration]; beyond the end the strands keep coasting out)
  focus(t)   hold_pose(dist)   (camera hints only)   duration
```
Built-ins: `lanes` (hairline lanes, solid fill), `pipework` (shaded tubes, chamfered bends, tube letters). Both are
`words/_strand.py: StrandWord`: outline cut into N arcs = N strands; override `connect()` (how a lane joins an arc), `TUBE`,
`FILL`, `ARC_PER_CAP`, `DEFAULT_FONT`. Params: `n_strands`, `width`, `tail_len`, `run`, `speed`, `bend` (pipework).

## Transition module (`transitions/<name>.py`)
```python
TransitionModule(a_exit=None, b_entry=None, dur=None, **params)   # bare (params only) inside a Sequence
  bind(a_exit, b_entry, dur=None)  -> build()      # Sequence does this from the neighbours' bundles
  layers(t)  duration  focus(t)  bounds_points()   # dur=None: auto_dur() from distance / hand-off speed
```
Built-in: `twist(turns, handed=+1|-1, radius, pitch, stiff, squeeze, stretch, piece)`.

## Add a module
```python
from mograph.modules.core import register
from mograph.modules.words._strand import StrandWord

@register("mything")                      # id used by the library CLI and manifest
class MyThing(StrandWord):
    description = "one or two plain sentences"
    DEMO_WORDS = "THING"
    TUBE = 0.0
```
Then import it in `words/__init__.py` (or `transitions/__init__.py`). For a different mechanism subclass `WordModule` directly:
implement `build()`, `entry_bundle()`, `exit_bundle()`, `layers(t)`. A new transition subclasses `TransitionModule`: implement
`build()` (uses `self.a_exit`, `self.b_entry`, `self.dur`) and `layers(t)`; use `adapt()` for the pairing and `core.Slide` /
`core.Route3` / `core.strand_layers` for the moving windows.

## Form-in / form-out styles and Material (phase 3a)
```python
Lanes("FORM", form_in="blobs", form_out=("flicker", {"beat": 0.5}))      # any word module, any style
mograph.modules.io: lines (default) | blobs | flicker | particles | wipe       # io/<name>.py, IOStyle subclasses
Material(kind: strands | points | blobs | none)  .to(kind, like=None)  .as_bundle(like=None)   # material.py
module.entry_material()  module.exit_material()   # strands for 'lines', blobs / points / none otherwise
module.hold_shape()  module.body_layers(t)  module.reveal_layers(q) / conceal_layers(q)         # what a style works from
```
A style owns the form window (reveal, p = 0..1) and the exit window (conceal; p > 1 = material still coasting away). The module's own
look and mechanic run in the hold. `STYLE_MECH` (MechWord family, 0.45) keeps that share of the window for the body's own volume
mechanic: style, then mechanic; on exit the mechanic folds back first. Contract: `layers(0)` draws exactly `entry_material()`,
`layers(duration)` exactly `exit_material()`; `entry_bundle()` / `exit_bundle()` return that material as strands (blobs become capsules, points
short dashes), so every transition runs on any style (`adapt()` and `TransitionModule` accept any Material). `flicker` and `wipe` carry
nothing (`none`): Sequence puts the `cut` transition there (match-cut + frame-exact hairline flash) and lists it in `seq.notes`.
Style options: blobs `max_blobs, r0, soft, solid, seed`; flicker `fps, frag, jitter, slices, strobes, beat, seed`; particles `pitch,
carriers, mark, accent, seed`; wipe `soft, edge, steps`. Add a style: subclass `io.base.IOStyle`, `@register_style("name")`, implement
`layers(t, p)` and `material()`, import it in `io/__init__.py`. Checks: `python 14_modules/v004/io_test.py`; layer-for-layer regression
of the default look: `14_modules/v004/io_regress.py`.

## Sequence and camera
```python
seq = Sequence([lanes_A, Twist(turns=2), pipework_B])          # two words with no transition get the default 'twist'
seq.render(path, camera=None | 'demo' | Camera | FlowPath | seq.follow_camera(), fps=30, size=(960, 540), S=2, audio=None)
seq.timeline  seq.duration  seq.layers(t)  seq.frame(t, camera)  seq.demo_camera()
```

## Library
`python -m mograph.modules.library <module> [--words ..] [--font ..] [--entry left] [--exit right] [--check-dirs]`
renders the clip (centred, demo camera, ~4 s) + contact sheet (or the direction-check sheet) into
`14_modules/v001/local/` (gNN, never overwritten) and updates `14_modules/library/manifest.json`.

## Camera modules (`camera/`, phase 3b)
One file per move; `camera/base.py` has `CameraModule`, `CameraChain`, `Hints`, `Segment`.
```python
from mograph.modules.camera import CameraChain, Follow, OrbitEven, SpinReadSpin, PushThrough, TiltToPlan, Graze, EaseReversal, PullbackReveal, HoldDrift
chain = CameraChain.auto(seq)                                    # rules below
chain = CameraChain.auto(seq, words="hold_drift", override={(4.0, 7.3): OrbitEven()})   # default + one move in a window
chain = CameraChain(seq, [(Follow(), 0, 3.4), OrbitEven(rate=24).at(3.4, 7.2), (HoldDrift(), "word:2")])
seq.render(path, camera=chain)        # or camera="auto";   chain.at(t) -> cam3d.Camera;  chain.report(verbose=True)
```
- `CameraModule(**params)`; `.at(t0, t1 | 'word:i' | 'hold:i' | 'trans:i')`; `segment(seq, t0, t1, hints) -> Segment` (eye / look / roll / log F / right-hint
  curves on a 240 Hz grid, plus declared `rests` and `fast` windows); `curves(hints, ts, t0, t1)` is what a subclass implements.
- `Hints` (`seq.camera_hints()`): `words[i]` (module, placement, `fit()`, `pose()`, `counters()`, hold times), `trans[i]`, `focus(t)`, `window(name)`,
  `enter` (eye / look / velocity the chain has reached when the segment starts), `next_module`.
- Joins: every segment starts exactly on the position, velocity and acceleration the chain has reached (a quintic Hermite correction that relaxes into
  the module's own curve; the relax time is picked per join from 0.5-2.6 s by minimising heading change and speed dips). So position, velocity and
  acceleration are continuous (C2). The chain starts from and ends at rest. `clear=140` keeps the eye 140+ px away from strand lines (not while
  diving through a counter). Gates (fix pass v006): (1) a join whose route doubles back or turns more than `rest_head` (3.5 deg/frame) becomes an
  explicit ease-to-rest turn: the camera brakes along its own path to rest (quintic, C2), then leaves on the new module's route (a quintic Hermite chord
  or the module's route plus a decaying offset, whichever turns less); `max_relax` on a module keeps a sustained move (orbit, spin, dive) on its own arc.
  (2) the turn governor `_govern`: wherever the composed path would still turn faster than `turn_deg` (2.4 deg/frame) in heading, view or image roll
  (outside declared rests / fast spins), the whole camera is re-timed along its own geometry (slowed into the turn, catching up on straights, max rate
  1.35; `chain.gov` has max_lag / slowed_s). The view settles on a new target faster than the eye (`look_relax` 1 s).
  (3) the image-right hint: `Segment.right_v` defaults to the horizon-level right of the view azimuth; only `tilt_to_plan` fades it in (`right_w`).
  A word's camera normal is `WordInfo.vn` (the plane normal, clamped to 58 deg elevation: floor / ceiling words are read at a raked angle).
- `chain.report()`: speed_report-style numbers on the cameras the renderer gets (stops outside declared rests, heading jump, view turn, roll step
  as the parallel-transport roll about the view axis, jerk) and the v003 gates (heading <= 3 deg/frame outside declared rests / fast spins, spins
  reported separately); `CameraChain.speed_line(summary)` is the one-liner used in the library descriptions.
- Moves: `follow`, `orbit_even`, `spin_read_spin`, `push_through`, `tilt_to_plan`, `graze`, `ease_reversal`, `pullback_reveal`, `hold_drift`.
- `CameraChain.auto` rules: follow through every transition (from the previous word's exit to the next word's form); a word on a tilted plane
  (|yaw| >= 12 deg, hold >= 0.6 s) gets `orbit_even`, any other word `hold_drift` (`words='hold_drift'` forces it everywhere). Same gates as any chain.
- `orbit_even` is an arc of `sweep` (100 deg) at one rate, pivot-locked, starting at `entry_az` (the follow before it arrives there; same hook as
  `spin_read_spin`). `push_through` with `to_next` ends at rest on the next word's hold pose, which must lie beyond the dived-through plane (no U-turn).
  `tilt_to_plan` fades in the image-right hint of the track's own azimuth (no roll) and backs off to take in the transition. `pullback_reveal`
  never ends smaller than `min_word` of the frame. `ease_reversal` looks on to the strands for its last second.
- Library clips: `14_modules/v006/camera_clips.py <id|all> [--preview] [--cover] [--final]` on two standard chains: `planar`
  (`14_modules/v004/camera_test_chain.py`: follow, hold_drift, graze, ease_reversal, spin_read_spin, pullback_reveal) and `depth`
  (`14_modules/v006/camera_test_chain_depth.py`: tilted word -> twist -> wall word -> circuit board -> floor word, wall and floor joined at a crease:
  orbit_even, push_through, tilt_to_plan, auto). `diag.py` lists where a clip fails a gate.

## Transition engine (phase 4, `engine.py`)
```python
seq = Sequence.auto([w0, w1, w2, w3, w4], seed=3, energy=0.4, mood="calm", bpm=96, duration=None)   # engine picks every free gap
gap = engine.analyse(a, b, others=words)          # GapInfo: dist, heading, detour, plane ('same'|'fold'|'opposite'), ortho, size_ratio, kinds / counts, obstacle ...
engine.rank(gap)                                  # [(name, score, parts)] rules only
tcls, params, dur, camera_hint = engine.choose(gap, history, seed, energy)     # Choice (a tuple + .name .score .why .gap)
seq.gaps, seq.engine, tr.camera_hint              # readouts; engine.describe(seq) prints them
```
Rules: every transition class has `SUITS` ({feature: (trapezoid | {category: fit}, weight)}, `must`, `moods`, `dur`, `brief`); score = weighted
mean x must-fits; variety (repeat x0.35, two back x0.75); seeded weighted pick among the top 3; params from the gap; duration from the
hand-off speed, energy, range, `avail`, beat grid. Side with material `none` -> `cut`. Explicit transitions in the list are kept. Tests:
`python -m pytest tests/test_modules_engine.py`; driver `14_modules/v005/engine_lib.py` (decision sheet, auto chains). A new transition only
needs a `SUITS` (and optionally a params function in `engine.PARAMS`).

## Portrait world plane and animated font axes (tools/fusion)
- `cam3d.set_world(540, 960)` / `with cam3d.world(540, 960):` makes the frame plane portrait for this process (default 960x540 unchanged; call it
  in worker processes too; `Sequence.render(..., world=(w, h))` does). `cam3d.FIT_HK / FIT_FILLK` tell `WordInfo.fit` / `follow_camera` to frame tall
  boxes by their full height and fill more of the frame in portrait.
- `layout_word(font, ...)`: `font` may be `{'key': .., 'axes': {'wght': 700, 'ELSH': 50}, 'ar': 0.6}`: variable-font axes go through fontTools
  (`ol.word(loc=axes)`, one glyph set per axes dict, cached) so every word module gets them; `ar < 1` squeezes the face horizontally (tall faces).
  Letters whose contours all wind the same way (pixel / dot fonts) are flagged `ConList.union`, and `Layer.union` fills each contour on its own so
  overlapping squares merge instead of cancelling (even-odd).
- `tools/fusion/type_passes.py`: JSON spec -> beauty / matte / depth / line / clay / flow passes (frame aligned, one camera). See its docstring.

## Contour ports and matte treatments (v005, `contour_ports.py`, `tools/fusion/matte_treatments.py`)
Footage contours as entry / exit / via ports of every word module and as the path between two words. Sources (`PlateData` = plate + matte +
depth per frame): `matte_outline` (subject outline, closed, s = 0 lowest point, 0.5 top, left side first), `depth_iso` (floor / background
iso-lines, subject painted out so they run on behind it), `luma_edges` (iso-lines of the blurred plate luminance). `ContourTrack` keeps the
lines in correspondence over time; `ContourContext` projects them through the piece camera (`bind_camera(chain)`), so they stay pinned to the picture.
Piece JSON keys, on a word spec (a dict under `entry` / `exit` is a contour port, a plain string stays the module side):
```
"entry": {"port": "matte_outline|depth_iso|luma_edges", "dir": "up|along|around", "lead": 0.6, "glow": 0.9, "persist": 0.0,
          "pulse": 120 | "beat", "travel": 1.0, "strand_len": 0.16, "lines": null, "levels": 5, "width": 1.5, "colour": null, "side": "up"}
"exit":  {same keys}          strands travel to the contour and lie down on it (glow = contour left glowing, persist = s it stays)
"via":   {same keys}          the transition AFTER this word: A exit -> contour ride -> B entry (replaces the planned transition, same duration)
"matte": "front|window|fill|wipe|glow"  or {"mode": .., ..options}   one matte treatment for this line (hit .. release)
```
`dir`: `up` = both sides of an outline ride up to the head, open lines toward their higher end (never downward on people; there is no `down`);
`along` = clockwise / left to right; `around` = half a turn / serpentine over the lines.
Planner hooks: `contour_ports.available(piece)` -> `{"ports": [kinds every shot supports], "shots": {sid: {kind: bool}}, "matte": [treatments]}`
(`matte_outline` needs the shot's `"matte": true`, `depth_iso` its `"depth": true`, `luma_edges` always); `split_port_spec(word)` -> (module_entry,
module_exit, ports) so spec.build_scene can strip the dicts before it builds the module; `apply_ports(items, words, ctx)` wraps the words
(`PortedWord`: entry / exit rides are carved out of the module's own form / exit window, so the beat grid does not move) and swaps in `ContourRide`
transitions for `via`; then `Sequence(items)`, `attach(seq, ctx)`. Bind the camera first (`ctx.bind_camera(chain)`); rebuild after a camera that
depends on the transitions changes. `ContourRide` is not in the engine pool (needs a context). Matte treatments:
`matte_treatments.ops_for(from_words(words, windows))` -> compose ops after the plain image (legible guard stays on: front / wipe end in `sandwich`,
window carries an explicit `legible`). Check render on the box: `python -m tools.fusion.contour_ports_check --plates DIR --out DIR --set ports|via|matte`.

## Multiplane plate: gallery camera chains through the footage and the type (v005, `mograph/render/multiplane.py`)
The plate frame is cut into 3-5 depth cards (background bands by depth, the subject on its own card from the matte, a foreground card for what is
nearer than the subject), placed in the same cam3d world as the type, so a CameraChain move (push_through, orbit_even, graze, follow, ...) travels
through the type AND the footage. The type is composited per pixel by depth (it can sit between cards or cross them).
Spec keys: `"camera": {"chain": "orbit_even" | "auto" | [{"module": "push_through", "word": 0, ..params}, {"module": "orbit_even", "window": "word:2", "sweep": 56}],
"plate": "multiplane", "cards": 4, "params": {..}, "z": [far..near], "ref": 1428, "subject_z": 1000}` and, on a word, `"slot": k` (type sits between card k and k + 1;
-1 behind all, cards - 1 in front of all; or an explicit `placement.z`). Functions: `plan(spec)` (card depths + slot depths), `apply_slots(words, cfg)` (before
spec.build_scene), `camera_for(seq, cam)` (the CameraChain), `MultiPlane(data, world, size, cards, z, ref).render(frame, cam, type_rgb, type_a, type_depth)`
(torch; CPU when the shared box GPU is full). `ContourContext.set_plane(z, ref)` puts contour ports ON the subject card (world-fixed, parallax with the cards).
The camera moves are sized by the words (about 1400 px), so keep the subject card ~1000 behind the home plane (`subject_z`) when a push_through should end in front
of her. Check render on the box: `python -m tools.fusion.multiplane_check --plates DIR --out DIR`. Magnification is clamped per card to [min_mag 0.82, max_mag 1.35] of the source pixels (the eye is held back / brought in along the view axis for that card only, so a dive passes
through the type while the plate only parallaxes), cards have a 15% reflected overscan and the sampling window is kept on the canvas (`edge_report()`), and
`plan_legible(mp, chain, fps, type_at, depth_at, holds)` + `compose.legible_clear` draw letters that the cards would hide for their whole hold in front. In a piece: `tools/fusion/multiplane_recipe.apply(piece, ops, sid)` (called by `piece_runner.recipe`) replaces the flat type ops by the `multiplane` compose op
(`compose_op`: plate cards + the type passes beauty / matte / depth + the piece's camera, rebuilt from the type manifest; one op per core, CPU torch); the matte treatments
front / window / fill / wipe are dropped there. All cards share ONE camera (held-back eye, eased look turn), so a card is never rescaled on its own. Limits: 360x640 draft;
the cards are flat (floor / walls are bands, not slopes); a strong pan inside a push_through briefly distorts the subject card.

## Pan transitions: the camera travel IS the transition (v006, `camera/pan_dir.py`, `transitions/pan.py`, `tools/fusion/pan_plan.py`, `plate_pan.py`)
Four directional transitions (camera down / up / right / left) where nothing is animated per word: word A holds, the `pan` camera (`PanDir`) travels
one frame-span along D with a quintic ease (pure translation, no zoom / roll, rest on both words), A leaves the frame on the opposite side exactly as a
world-locked object would, B (placed one frame-span further along D in the same face-on plane) arrives as the camera settles, B's hit on a downbeat.
`Pan` (transition, hidden from the engine pool / `list_modules('transition')`, `get_module('pan')`) only DRAWS both words in the world: A frozen on its
hold, B formed (`WordModule.carry` freezes A's exit window / B's form window; `layers_uncarried` bypasses it), so it composes with ANY word module;
`exit_io` / `entry_io` map the io style of A's exit / B's entry onto the travel (the frame edge); flicker / wipe sides are turned into lines (a cut would
replace the travel). Geometry shared with the planner: `pan_dir.fit_dist` (= `WordInfo.fit`), `frame_span`, `DIR_VEC` (camera direction in image axes).
```
"transition": {"kind": "pan", "dir": "down", "beats": 2}      on a word: the transition AFTER it   (also "transitions": [.., {"module": "pan", ..}])
"camera": {"move": "pan", "dir": "down" | ["down", "right"], "beats": 2}    every gap without its own transition becomes a pan
keys: dir, beats (2) | dur (s), ease io|soft, gap (extra air, share of a span), fill (0.7), exit_io, entry_io, land downbeat|beat, blur (1), parallax (0), plate (1 = gain of the plate pan)
```
`spec.plan_timing` (via `pan_plan.boundaries`): A's hold is kept, A's exit / B's form are `EDGE_S` (0.05 s) and frozen, the travel starts after the hold and ENDS on
B's hit on the first downbeat the readable hold allows. `spec.build_scene` places the run's words (face-on, screen offset / axes animation / contour ports off on
the pan words) and carves one `PanDir` over the run into the auto chain (`pan_plan.window_overrides`; also in `multiplane.camera_for`). The plate travels with it:
`pan_plan.plate_track(scene)` = the on-screen displacement of the type's plane from the REAL camera, `pan_plan.weave(piece)` bakes it into compose op
`plate_pan` (numpy, `tools/fusion/plate_pan.py`: wrap-tiled plate, motion blur taps proportional to the speed, soft seam while moving, optional depth parallax) before the
plate buffers of every chain (`$footage` / `$matte` / `$depth` -> `*_p`), settles `flow_wake` during the travel and rewrites the shots' LTX prompts with
`pan_plan.ltx_phrase` (tilt down / tilt up / pan right / pan left). `pan_plan.dir_from_flow` / `retarget` pick D from the footage's own camera motion.
Planner: lego block `camera_pan_transition` (opt-in: named by the brief or the request key `pan`: True | n | 'down' | [dirs]), batch coverage `pan_dirs` (4).
Check: `python -m tools.fusion.pan_check` -> `16_app/core_check/v006_pan_dirs_local.png`; tests `tests/test_pan_transitions.py`.
