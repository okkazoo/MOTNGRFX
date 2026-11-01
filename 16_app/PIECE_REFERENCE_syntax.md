=== ENGINE SYNTAX NOTES (facts the engine checks; the code mends what it can and logs it against you) ===
References and kinds
- Every reference to a buffer starts with `$`: `"type_rgb": "$tcol"`, `"img": "$image"`. A bare name (`"tcol"`) is wrong. A name you give to `out` is written WITHOUT the `$`, read WITH it.
- Pictures are either RGB (a colour image) or GRAY (a 0..1 mask). Ops that need a mask (`type_a`, `matte`, `mask`, `front`, `edges`, `fg_matte`, `cover`) must be given a gray buffer; ops that need a colour image
  (`footage`, `frame`, `img`, `type_rgb`, `plate_rgb`, `base`) an RGB one. RGB: `$footage`, `$type_beauty`, and the output of tint_type, grade, sandwich, depth_occlude, knockout, add_glow, halation,
  grain, vignette, fill, refract. GRAY: `$matte`, `$depth`, `$flow_m`, `$type_matte`, `$type_line`, `$type_depth`, and the output of upscale_matte, luma, luminosity_key, fill_holes, flow_wake,
  contour_lines, edges_in_type, type_from_edges, mask_gate, mask_mul, matte_reach, matte_ring, split_region. `over`, `crossfade`, `blur`, `translate` give back the kind of their first picture.
  `screen`, `multiply`, `overlay`, `soft_light` need two pictures of the SAME kind (or a number for `b`).
- `flow_wake` returns the displaced ALPHA (gray): use it as `type_a` of the next op, never as `type_rgb`. The type colour follows the same displacement automatically.
- `sandwich(footage, type_rgb, type_a, matte)`: the type sits between `footage` and the subject; `matte` is the subject cut-out at the footage size (`upscale_matte` of `$matte` -> a buffer you name, e.g. `m`).
  `matte_threshold` 0.5 + `matte_soft` 0.08 make a solid subject; `front` is an optional gray REGION (not the type itself) where the type is drawn IN FRONT of the subject (e.g. `luminosity_key` of the type depth).
  `matte_gain` (a number or a timeline) scales the matte: 1 = the subject covers the type, 0 = the type is in front of the subject.
- The chain ends with an op that writes `image`. Colours inside ops are floats 0..1 (`[0.35, 0.65, 1.0]`); only the type palette in `type.palette` uses integers 0..255.
Timelines (time in PIECE seconds, not shot time)
- number over time, linear: `{"timeline": [[0, 0.4], [6.0, 0.9], [12.0, 0.5]]}`;  smooth: add `"ease": "smooth"`;  jump at each key (colours, on/off): add `"step": true`.
- colour over time (a colour per section, switching on the chord changes): `{"timeline": [[0, [0.35, 0.65, 1.0]], [6.0, [1.0, 0.68, 0.28]], [12.0, [0.7, 0.5, 1.0]]], "step": true}`.
- on/off per word: `{"timeline": [[0, 1], [0.75, 0], [1.5, 1], [3.0, 0], ...], "step": true}` (a value per time span; keys in increasing time).
- driven by the music: `{"curve": "onset", "lo": 0.08, "hi": 0.45}` (curves: sub, low, mid, high, rms, onset).
- A timeline may be used for any numeric argument of any op (intensity, radius, gain, sat, opacity, strength, plate colour of `knockout`, `t` of `crossfade`, `reveal` of contour_lines ...).
Type and timing
- Every `hit` is a downbeat or beat of the MUSIC block. The first word starts forming at 0 s, so its hit is at least `form_s` after 0 (a first hit at 0.0 is not reachable). Words are placed one after the other: a word starts
  after the previous one has finished its exit plus a gap of at least 0.4 s. `hold_s` is at least 1.2 s. The last word's hold ends before the music does (cut length - 0.2 s).
- `transitions`: a list with one entry per gap between words; each entry is one name or null; a name (other than `cut`) may be used only once in the whole piece.
- `axes` only with the tags listed in that font's line of the FONTS list, values inside the range shown; `[lo, hi, "beat"]` makes the axis move between lo and hi on the beat.
- `placement.yaw` is degrees the word is turned about the vertical axis (-25..25); `camera_params` apply to `beat_orbit`: `intro_deg` is how far off face-on the word starts, `fast_dps` the turn speed in the gaps,
  `read_dps` the drift while it is read, `bob` the up-down breathing (defaults of the camera: read_dps 7, intro_deg 42, fast_dps 35, bob 0.8, elev 4).
Shots
- The app plans at most 6 shots (3-6 s each); an i2v shot is one continuous generation of at most 385 frames (16 s); `until` is where it ends in the piece. `first: "last:<id>"` continues from the last frame of an earlier shot (same scene, no cut); a new place
  needs its own still. Hard cuts: `compose.crossfade` 0. Passes (matte, depth, flow) are switched on by the buffers the chain reads (`$matte`, `$depth`, `$flow`).
