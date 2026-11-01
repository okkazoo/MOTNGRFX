# tools/mathtype/kit

Shared building blocks for the 11_mathtype pieces. A new piece is only its maths: a `frame(t, W, H)` plus
setup. The kit covers text, drawing, camera, timing, palettes, output and QC. Per-frame work runs in
torch on CUDA when available, else torch CPU (`kit.backend()`). Setup (glyph rasters, SDFs) runs once
on the CPU.

```python
import kit
from kit import text, draw, camera, timeline, palette, io, qc
be = kit.backend()            # "auto" | "cuda" | "cpu" | "numpy"
```

## Modules

| module | what | key functions |
|---|---|---|
| `text` | liked fonts by role, phrases, layout, SDFs, outlines | `LIKED`, `fonts(role)`, `load_font(name, size, axes)` (cached, all axes incl. CTST), `split_phrases(sentence, n)` → `Phrase(text, words, read)`, `read_time(text)` (0.5 s + 0.3 s/word, ≤ 20 cps), `split_lines(text, max_lines, max_chars)`, `key_word`, `auto_mix(lines, key, tall, puffy, morph_to)`, `layout(lines, W, H, scale, text_w, cap_max, track, gap, align, axes)` → `Layout` (`.draw/.mask/.sdf/.outlines/.slots/.centres`), `sdf(mask)`, `sdf_morph`, `smin`, `resample_closed`, `cached(dir, key, make)` |
| `draw` | per-pixel primitives (torch; numpy where cheap) | `canvas`, `paint`, `tone`, `to_u8`, `box_down(img, ss)`, `blur`; `gradmag`, `fill(f)`, `iso_line(f, level, w)`, `iso_bands(f, spacing, w)`; `disc`, `box`, `tri`, `discs`, `coverage`; `splat(xy, w)`, `densify`, `splat_polylines(P, I)`; `fill_tris(tri, vals)`; `square_wave(phi)`, `hash01`, `value_noise`, `sample`, `upsample` |
| `camera` | eases, springs, paths, projection | `ease(role, u)` (arrive = power4.out, arrive_expo, camera/flood = power2.inOut, exit = power3.in, linear, smooth), `cubic_bezier(...)`, `smoothstep(a, b, x)`, `smootherstep`, `bump`; `spring(tau, zeta, freq, x0, v0)` (closed form, any zeta), `SPRINGS` by role, `spring_to`, `spring_decay`, `log_zoom`; `bezier`, `bezier_path(keys, t, role)`, `hermite(keys, t)`; `screen_to_world`, `world_to_screen`; `rot`, `look_at`, `project`, `focal_px`; `homography(src, dst)`, `apply_h`, `warp(src, H, W, H)` |
| `timeline` | landings, read and transition windows | `Timeline.build(phrases, bpm, start, form, trans, tail)` (landings snap to the beat), `Timeline.fixed(phrases, land, trans, dur)` (existing pieces); `.land .read .trans .dur .nframes`, `.state(t)` → (a, b, s), `.phase(t)` → (form/read/trans, i, u), `.reading(t)`, `.read_moments()`, `.changes()`, `.check()`, `.summary()` |
| `palette` | named palettes, colour-area rule | `PALETTES` (mono, e1, ink, dawn, chalk, e4, nature, sea), `get(name)` → `Palette(.ground .ink .accent .accents .limit)`, `rgb(hex)`, `area(img, pal)`, `overruns(img, pal)` |
| `io` | render, encode, audio, logs, review images, CLI | `render(frame_fn, path, W, H, n, fps, batch, frames_fn, nvenc)`, `Encoder` (h264_nvenc on CUDA, else libx264, CRF 16, never overwrites), `score(preset, dur, changes, wav)`, `mux`, `timing_row(local, ...)`, `make_minutes`, `contact_sheet`, `strip`, `still`, `run(piece, kind)`, `main(make_piece)` |
| `qc` | first-pass review, ~1-2 s | `review(video, timeline, palette, mask_fn, out_prefix)` → JSON + strip; `legibility(Y, mask)`, `decode`; `python -m kit.qc VIDEO` |
| `parity` | migration check | `python tools/mathtype/kit/parity.py e4 --ref <final.mp4>` (old path vs `--kit` path vs the delivered final) |

`kit.io.main(piece)` gives every piece the same CLI: `--draft`, `--final` (adds the score and mux), `--still T..`, `--size`,
`--qc VIDEO`, `--device`, `--batch`, `--x264`, `--no-audio`, `--parent/--trigger/--change`, and
`--work-start` (feeds make_min). Each run records:
- the gNN generation (naming rule; a code hash that includes the kit);
- a GENERATIONS row and a TIMINGS row (timing rule);
- two stills;
- the contact sheet;
- the QC JSON and strip.

## A new piece in 10 minutes

1. Copy `tools/mathtype/harmonograph.py` (136 lines, the demo) to `tools/mathtype/<name>.py`.
2. Set `SENTENCE`, `VDIR` (create the version with `proj.py version-start`), `test`, `stem`, `score_preset`, and pick a palette.
3. In `__init__`, build:
   - the phrases: `text.split_phrases(SENTENCE, n=3)`;
   - the timeline: `timeline.Timeline.build(phrases, bpm=96, form=1.1, trans=0.9)`, whose read windows satisfy the readability rule by construction;
   - the type: `text.layout(text.auto_mix(text.split_lines(p.text)), DW, DH, scale=2, text_w=560)`, with `.sdf()` for fields or `.outlines()` for pen paths.
4. Write `frame(t, W, H)` with the maths only:
   - use `timeline.phase(t)` / `state(t)` to know where you are;
   - drive arrivals with `camera.ease("arrive", u)` and exits with `camera.ease("exit", u)`;
   - draw with `draw.splat_polylines`, `draw.iso_bands`, `draw.fill`, `draw.discs` or `draw.fill_tris`;
   - composite with `draw.paint` on a `draw.canvas` at `ss = 2` and `draw.box_down`.
5. Add `text_mask(i, W, H)` (the glyph mask), so the QC measures legibility against the real letters.
6. Run `python tools/mathtype/<name>.py --draft --work-start HH:MM`. Read the sheet in `review/`, and fix whatever the QC flags print.
7. Run `--final`. On the box, add `--device cuda --batch 4` if you wrote `frames(ts, W, H)`.

The skeleton:

```python
class Piece:
    test, stem, score_preset = "E8a", "E8a_name", "E3"
    def __init__(self, a, be):
        self.be, self.version_dir, self.palette = be, VDIR, palette.get("mono")
        ph = text.split_phrases(SENTENCE, n=3)
        self.timeline = timeline.Timeline.build(ph, bpm=96, form=1.0, trans=0.9)
        self.lays = [text.layout(text.auto_mix(text.split_lines(p.text)), 720, 900, scale=2, text_w=560) for p in ph]
    def frame(self, t, W, H):
        a, b, s = self.timeline.state(t)
        ...                                  # the idea
        return draw.to_u8(img)
if __name__ == "__main__":
    io.main(lambda a, be: Piece(a, be))
```

## Conventions

- **Units.** Design space is 720x900 for 4:5. Layouts are rendered at `scale` px per design px. SDFs are negative inside; `Layout.sdf()` is in design px.
- **Determinism.** Every frame is a pure function of t, and all randomness comes from a seeded RNG at setup.
  - The springs are closed form.
  - Splats use `index_put_(accumulate=True)` and the triangle fill uses `scatter_reduce(amax)`; both are deterministic under `xp.Backend`'s CUDA settings.
- **Fonts.** `load_font` applies the registry axes, including Facade's `CTST`, which the older scripts never mapped. To match those old renders, `layout(..., axes={"Facade": {"CTST": 800}})`.
- **Migrated pieces.**
  - `moire.py --kit` (E4) and `contours.py --kit` (E1) route their shared pieces through the kit. The old path is still the default.
  - The parity results are in `11_mathtype/v014/review/parity_*.json`.
