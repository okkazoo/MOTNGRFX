# HOLD LIFE (2026-10-30)

User: "when the text resolves it shouldnt be static it should have some glitch or pulse or glow or something to keep it alive - it should be legible for a beat but in that beat still be active or subtly glitching or each letter is glitching on and off".
The landed word stays centred and still (stable_type rule 1, no camera move); it lands on the beat and reads for >= 1 beat; life is added on the DRAWN type layer only, never to LTX-rendered words or the footage.

Code: `tools/fusion/hold_life.py` (pure planning + compose op `hold_life`, registered in `compose._ops_registry`), tests `tests/test_hold_life.py`, check `16_app/core_check/hold_life/` (make_check.py, strips, metrics.json).

## Menu (`hold_life.EFFECTS`)
| name | what |
|---|---|
| `letter_flicker` | single letters blink off (65 %) / dim to 0.3 (35 %) for 1-2 frames, a few per beat, off-beat first then 16ths, seeded; needs a word of >= 7 letters (see guard) |
| `glitch_slice` | 2-4 frame bursts on sub-beats: 1-3 horizontal bands shifted 0.3-0.9 % of the frame width + RGB split 0.19-0.55 % (1-3 px at 540 wide) |
| `pulse` | scale 1.00 -> up to 1.03 -> 1.00 + brightness lift (5-15 %) on each beat after the landing, about the ink-box centre; attack ends ON the beat frame, decay 0.4 beat; a one-beat hold pulses on the off-beat |
| `glow_breathe` | an `add_glow` on the type alpha after the last drawer: intensity timeline = beat envelope (0 outside holds, ramps 0.2 s, starts after the read frames), radius follows the music `rms` (else `onset`) curve when the piece has one |
| `outline_shimmer` | the piece's forced outline windows (`compose.outline`, from random_system.draw_outline) get 2-3 frame dips on off-beats; legible_color eases the outline (0.15 s) so it breathes (measured: dip to 57 % of the full outline in a 2-beat hold, one dip at 0.75 s). Width stays legible_color's (not modulated). Needs an outline in the piece |

Per piece the seeded draw (`draw_effects`) picks 1-2 (45 % one), never `letter_flicker` + `glitch_slice` together; each word draws its own intensity 0.6-1.0 (event density, slice amplitude, pulse amplitude, lift, glow level).
A request can force effects: `compose.hold_life = ["pulse", ...]` (list or comma string; exactly those, feasible ones only); `compose.hold_life = false` switches the stage off. (The effects menu owner can list the five names in `effects_menu.EFFECTS` and copy the request's names into `compose.hold_life`; nothing here edits effects_menu.py.)

## Legibility guard (hard; tests/test_hold_life.py)
- every frame of a hold: >= 85 % of the letters fully visible -> at most floor(0.15 n) letters off/dim at once. Consequence: a word with n < 7 letters gets NO letter_flicker (one of 6 is 83 %); the draw drops the effect when no word has 7+ letters and the plan logs `skipped`.
- a letter off/dim for <= 2 frames at 24 fps (`max_off_frames(fps)` scales: 4 at 48 fps), never two adjacent letters off in a frame, enforced again in the op on the real letter count.
- displacement <= 1.5 % of the frame width (slice + split, and pulse growth: 0.03 x half word width <= 1.5 %).
- the first 3 frames after the landing are the identity (no dropout, offset, scale, lift); dropouts / slices / split never touch a beat frame +-1. Only the pulse sits on the beat (scale <= 1.03 about the ink-box centre, every letter fully visible).
- effects live in [hit + 3 frames, release): nothing in the exit, nothing outside holds; a handoff word's window stops at its handoff time, an `ltx` word gets nothing.

## Letter masks
The type passes (beauty / matte / depth / line / clay / flow) expose NO per-glyph alpha, so letters are the CONNECTED COMPONENTS of the type alpha (core > 0.35, every other drawn pixel takes the nearest component via distance-transform labels, marks over a letter - dots of i/j, accents - merge into it, reading order = rows top to bottom / left to right, vertical words sorted along y). Cached per hold. Touching letters merge into one component; the plan names letters by text index and is mapped proportionally, then the guard is applied again on the component count (fewer than 7 components = no letter flicker for that word). Making the type engine emit a per-glyph id pass would remove this approximation.

## How it is wired
`hold_life.weave(piece, music, chk, size)` plans (`plan(words, piece, seed)` -> JSON in `piece["plan"]["hold_life"]` summary + the full plan inside the ops) and, per compose chain, puts op `hold_life` (img = ref = the drawer's `type_a`) in front of every `sandwich` / `depth_occlude` / `knockout` and rewires `type_a`; drawers fed a line pass are left alone. `recipe_eval.legibility_pass` twins the op onto the beauty face exactly like `translate` (one token added to its tuple): the alpha twin takes dropouts / slices / split / scale, the face twin the same warp + RGB split + lift, both from the same reference alpha. `glow_breathe` adds `add_glow` after the last drawer; `outline_shimmer` rewrites `compose.outline`. Idempotent (re-weave replaces).

## HOOK for the other worker (plan_rules.py / stable_type.py were being edited, so NOT added by me)
In `tools/fusion/plan_rules.py`, `_finish`, right after the `pan_plan.weave(...)` line (and before `piece["layout"] = layout(piece)`):

```python
    from tools.fusion import hold_life; hold_life.weave(piece, music, chk, size)   # 2026-10-30 hold life: 1-2 of letter_flicker / glitch_slice / pulse / glow_breathe / outline_shimmer on the drawn type in each hold (no-op without word holds)
```

`stable_type.py` needs no change (the word stays centred and still; hold life is compose-side). `weave` never raises for missing word timing (it records `plan.hold_life.error` and returns the piece). Seed = `plan.piece_system.seed`, else `type.seed`.
