# INGEST: any dropped video / still -> the base format (2026-10-30)

Module `tools/app/ingest.py` (stdlib planning, numpy/cv2 only in the pixel helpers; standalone, no repo imports), tests `tests/test_ingest.py` (59, local, ~1 s).
User asks: "take their ratio and fit it or crop it to work with our base ratio ... so everything is the same" and "how do we handle the difference in frame rate?" (a 30 fps clip forced to 24 fps stuttered: 1 frame in 5 dropped).

## Frame-rate policy (`timeline_fps`)
Integer frame ratios only. Never a fractional resample (30 -> 24 is impossible by construction; a test asserts src/out is an integer ratio for every rate).
| source | timeline | how |
|---|---|---|
| 23.976 / 24 / 25 / 29.97 / 30 and anything < 46 fps (15, 20 ...) | same, exact rational (`30000/1001`) | `fps=<rate>` (identity on a constant-rate stream) |
| 47.95 / 48 / 50 / 59.94 / 60 (>= 46) | exactly half (90 -> 30 every 3rd, 120 -> 30 every 4th) | `select='not(mod(n,2))',setpts=N/((out)*TB),fps=out`, no blending |
| VFR (`r_frame_rate` and `avg_frame_rate` differ > 1 %) | constant 24 / 25 / 30, nearest to the average (avg >= 46 halved first) | `fps=<rate>` (drops / duplicates by timestamp) |

Audio is `-c:a copy` (re-encoded to AAC 256k only when the codec cannot live in mp4, flagged in `audio.copy`); video starts at 0 so it stays in sync.

**Open for the route:** the pipeline FPS constant (`plan_rules.FPS = 24`) and LTX are fixed at 24. A 30 / 25 / 29.97 clip leaves ingest at its own rate. The route must either run the own-video timeline at `plan.fps.fps` (assemble + matte frames at that rate) or, if 24 is forced, interpolate (RAFT flow / RIFE) instead of dropping frames. Not decided or built here.

## Aspect policy (`plan_shot`, per shot)
Target = base ratio of the tier: `plan_rules.sizes_for(tier)["compose"]` = 360x640 preview / 540x960 draft / 1080x1920 final (a test checks the table against plan_rules). 16:9 via `--format 16:9`. Sizes are display sizes after rotation (side_data rotation / tags.rotate) and SAR.
- Crop window = the largest exact 9:16 window with integer sides (9k x 16k): 1920x1080 -> 603x1072, so the ratio is exact.
- `crop` (cover): near-target (ratio within 10 % of 9:16) -> light crop; or a subject track whose union (+8 % per side) fits the window; or no track and a portrait source -> centre crop. The window centre follows the subject centroid with a 0.5 s centred moving average (edges replicated), clamped to the frame, kept so the padded subject box stays inside, pushed off caption boxes during their time span (spans dilated by half the smoothing window so the window is clear when the caption appears). Unavoidable captions are listed with `residual_overlap`.
- `fit` (contain): subject union wider/taller than the window, or no track and a landscape source. The whole frame is scaled to the target and centred on a flat ground = per-channel median of the border strip of a downscaled frame at the shot start (`ground`, '#rrggbb'; not the k-means palette helper in compose.py, which pulls in the compose backend).
- Never upscale: `max_tier` per shot and overall (crop: window px >= tier width; fit: scale factor <= 1). Render clips the requested tier to `max_tier`; `warnings` says so. A source too small even for preview is flagged (`max_tier: null`, rendered at preview).
- Stills: same logic, one frame, no fps.

## Inputs (data; the matte / caption detection is the GPU part)
- `--track track.json`: `{"fps": 30, "boxes": [[x0,y0,x1,y1] | null, ...]}` normalised, one per frame at `fps` (null fps = per output frame), or `{"samples": [{"t": s, "box": [...]}]}`. Nulls are filled from the nearest known box.
- `--captions caps.json`: `[{"box": [x0,y0,x1,y1], "t0": s, "t1": s}]` normalised; missing t0/t1 = whole clip.
- `--cuts cuts.json | detect`: list of cut times (s) or `{"cuts": [...]}`; each span is a shot with its own mode / path / ground. `detect` = ffmpeg scene score 0.35 (slow). No cuts = one shot.

## What the own-video route must call
```python
import sys; sys.path.insert(0, "tools/app"); import ingest
pr   = ingest.probe(path)                                   # display_w/h, fps, vfr, rotation, duration, audio
plan = ingest.build_plan(pr, tier, track=track, captions=caps, cuts=cuts,
                         ground_fn=lambda t: ingest.ground_colour(ingest.grab_frame(path, t, pr)))
json.dump(plan, open("ingest.json", "w"))                   # = ingest.json
cmds = ingest.render_commands(plan, tier, outdir)           # tier is clipped to plan["max_tier"]
ingest.execute(plan, cmds)                                  # -> <outdir>/<stem>_ingest_<tier>.mp4 | .png
```
CLI: `python tools/app/ingest.py plan <video> [--track t.json] [--captions c.json] [--cuts c.json|detect] --tier preview -o ingest.json`, then `... render ingest.json --tier preview -o outdir/` (`--dry-run` prints the commands). Run it on the box (frame work).

Render path: a static single crop / fit / still = one ffmpeg command (`crop=...:exact=1,scale=..:flags=lanczos`, or `scale,pad=..:color=0x..`); a moving crop or several shots = ffmpeg decode (fps filter) -> cv2 per-frame crop / fit (INTER_AREA) -> ffmpeg x264 crf 14 with the source audio copied. Output = compose size of the tier (360x640 / 540x960 / 1080x1920); the draft 576x1024 shot size is the route's own scale step, as in `own_video.py` today.

Wiring into `own_video.py` is NOT done (another worker owns it): replace its 30 -> 24 resample + scale by `build_plan` + `render_commands`, and use `plan["fps"]["fps"]` for the matte / assemble rate.

## Box check (vast 58200117, `/workspace/mg_test_ingest/`, ffmpeg 6.1, preview tier, CPU only)
| clip | plan | output | frames | audio | mpdecimate unique src -> out |
|---|---|---|---|---|---|
| `own_video/your_clip.mp4` (portrait 30 fps 24 s) | crop(native), fps 30 kept | 360x640 30 fps | 720 = 24 s x 30 | 24.000 s, 1125 AAC frames both | 717 -> 716 |
| synthetic 1920x1080 60 fps 10 s + moving box + track + caption | crop(subject_fits), window 603x1072 follows the box, fps 60 -> 30 | 360x640 30 fps | 300 = 10 s x 30 | 10.000 s, 431 both | 600 -> 300 (exactly half) |
| same, no track | fit(default_fit), ground #62504a | 360x640 30 fps | 300 | 10.000 s | 600 -> 300 |
| 1080x1920 59.94 fps 5 s (draft) | crop(native), 60000/1001 -> 30000/1001 | 540x960 30000/1001 | 150 of 300 | 5.000 s | n/a |
| `own_video/src.mp4` 1080x1920 30 fps 48 s (final) | crop(native) | 1080x1920 30 fps | 1440 | 48.000 s, 2250 vs 2251 AAC frames | n/a |
| synthetic VFR (30 fps, every 5th frame missing, avg 24.1) | fit, VFR -> 24 | 360x640 24 fps | 192 = 8 s x 24 | 8.000 s | 192 -> 144 (see limits) |

Frame review (`f_b1_*.png`): the moving box is centred in the 360x640 crop at t = 8 s and clamped at the left edge early on; the fit frame is letterboxed on the ground colour. In the caption clip the caption sits on the subject's side and cannot be avoided (reported `avoided: false`, residual 58 %); the push logic is covered by the unit tests, not by a box clip.

Limits: VFR -> constant drops and duplicates frames by timestamp (the synthetic gap pattern gives 48 duplicates); fit of a 16:9 clip leaves ~68 % of the 9:16 frame as flat ground (as specified); 1 AAC frame of difference on the 48 s clip; captions and the matte-to-box step are inputs, not built here.
