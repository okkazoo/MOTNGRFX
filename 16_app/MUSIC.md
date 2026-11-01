# Exact-length, beat-locked music (score skeleton -> ACE-Step audio-to-audio)

> SUPERSEDED 2026-10-27: the skeleton + audio-to-audio route below sounded like "circus / tinker toy" and is out of the app graph. The app now makes music with ACE-Step 1.5 XL base text-to-music (no LM, 50 steps, cfg 6, length + 4 s), cuts it to the exact length, measures the beat grid on the song, then plans: see `_info/BUILT.md` Music, `tools/fusion/t2m_music.py`, `nodes_fusion/music_t2m.py`.

Problem (15_fusion/FINDINGS.md): ACE-Step text-to-music ends 1-10 s before the requested length with digital silence, and the beat
tracker misreads tempo on ACE / user audio (P02: 79 bpm read for a 122 bpm song). Fix: compose a CPU **skeleton** of exactly the
piece length on a known grid, restyle it with ACE-Step 1.5 audio-to-audio (MIT, commercially clean; Song2 is CC BY-NC and not used),
and **keep the skeleton grid as the analysis**. No beat tracking anywhere on this path.

```
MographScoreSkeleton -> AUDIO (exact samples) + MOGRAPH_MUSIC (grid)
   AUDIO -> VAEEncodeAudio(ace_1.5_vae) -> KSampler(denoise 0.3-0.7, 8 steps) -> VAEDecodeAudio
   MographMusicStyleEncode(clip, music, tags) -> positive; ConditioningZeroOut -> negative
   VAEDecodeAudio -> MographMusicFit(audio, music, skeleton) -> AUDIO (exact) + MOGRAPH_MUSIC (skeleton grid kept) -> MographMaster -> PiecePlan
```

## Files
- `mograph/score_skeleton.py` - `compose()`, `plan_grid()`, `fit_exact()`, `curves_of()`, `music_dict()`; CLI `python -m mograph.score_skeleton`.
- `nodes_fusion/music_score.py` (registered in `nodes_fusion/__init__.py` SUBMODULES): `MographScoreSkeleton`, `MographMusicStyleEncode`,
  `MographMusicFit`, `MographMusicRoute`.
- `tools/fusion/music_graph.py` - builds `app/graphs/music_exact_test.json` + `.app.json` (`write` / `check`); `a2a_chain()` is the reusable chain.
- `tests/test_score_skeleton.py` (32 tests), drafts in `16_app/music_check/`.

## The skeleton
`compose(length_s | n_frames, bpm, meter, key, mood, energy, sections, hits, seed)` -> `(wave float32 (2, n), grid)`.

- **Length.** `n_frames / 24` seconds, `n_samples = round(length * sr)` (48 kHz default). The wav has exactly that many samples.
- **Tempo.** `bars * meter * 60 / bpm == length` exactly. Every bar count whose tempo is 40-200 is a candidate; the nearest to the
  hint (log distance, small preference for even / phrase-sized bar counts) wins. The tempo moves, the audio is never trimmed.
  At 24 fps lengths 8 / 12 / 16 / 24 s give integer bpm (30 / 20 / 15 / 10 x bars): hint 122 -> 120 (16 s, 8 bars, and 24 s, 12 bars),
  80 -> 80 (12 s, 4 bars), 100 -> 90 (8 s, 3 bars). `grid["in_window"]` says whether it stayed within +-15 % of the hint.
- **Arc.** Default sections on bar lines: intro / build / drop / outro (8 bars: 2 + 2 + 3 + 1). `sections=[(start, end, role)]` with roles
  intro build drop break outro (aliases main/chorus -> drop, bridge/breakdown -> break), snapped to bars. The last bar is a landing
  bar: final chord + bass + crash, rings out, 0.5 bar (max 0.8 s) fade, so it never ends on a hard cut.
- **Parts.** Drums, bass, chords, pad, arp. SoundFont via `mograph/sampled.py` (GeneralUser GS + FreePats upright, symusic):
  `electronic` (drums kit 25 808/909, synth bass 1 + numpy sub, Tine e-piano stabs with the 5.3 / 6.3 kHz notch from v006, warm pad,
  vibes arp; moods driving / dark / bright / playful), `soft` (room kit, finger bass, upright piano strum, harp arp; calm),
  `filmic` (orchestral kit, double bass, slow strings, choir pad, harp; epic). Drum / bass / arp patterns and chord progressions
  come from `synth.py` tables, so moods agree with the old CPU bed. If symusic or the bank is missing it falls back to
  `synth.render_soundtrack` on the same grid (`grid["backend"]`).
- **Hits.** Video cut times, each snapped to the nearest beat (`grid["hits"]`, originals in `hits_requested`; out-of-range dropped, duplicates
  merged). On a hit: riser into it, crash + impact + kick + chord / bass accent, drums out for half a beat before.
- **Key.** `key` or mood + seed picks tonic and mode (minor, dorian, major ...). `grid["key"]` keeps the mode; `grid["key_ace"]` is the
  ACE `keyscale` (dorian / phrygian count as minor).
- **Deterministic.** Same arguments -> same bytes (tests compare arrays). Seed changes the key / voicing / humanise, never the grid.
- **Level.** About -16 LUFS / -1.6 dBTP, bass-heavy like the ACE songs (see the draft numbers below). It is a stem, not a master.

`grid` keys: `bpm beat_s bar_len bar_s meter n_bars bars[{i,start,end}] beats downbeats sections[{start,end,role,bar0,bar1}] hits hits_requested
key key_ace tonic mode mood energy palette chords[{bar,degree,root,pcs}] chord_changes length_s n_frames fps n_samples sr seed bpm_hint in_window backend tail_fade_s`.
`beats` / `downbeats` are strictly inside the piece (`downbeats[k] == k * bar_len`).

## MOGRAPH_MUSIC (what PiecePlan, TypePasses, Compose, Master read)
Same keys as `MographMusicCut`: `cut nframes bpm bar_len key beats downbeats chord_changes curves music_end fade fps src`, with
`cut = length`, `nframes = n_frames`, `beats` / `downbeats` extended 4 s past the end like MusicCut's, `curves` (rms sub low mid high onset,
`nframes + 1` values each) measured on the audio, `src = "score"` (Fit: `"score+ace"`). Extras: `meter n_bars sections hits n_samples sr grid`
(Fit adds `fit`). PiecePlan runs on it unchanged (tested). `chord_changes` are the composed progression's, not detected.

## Nodes
| node | inputs | outputs |
|---|---|---|
| `MographScoreSkeleton` | request?, length_s 12, music_prompt, bpm 0 (auto), mood auto, energy -1 (auto), key, meter 4, seed, hits "3.5, 8", sections "intro:0-3, build:3-6", brief? | AUDIO, MOGRAPH_MUSIC, grid_json STRING |
| `MographMusicStyleEncode` | clip, music, tags, request?, brief? | CONDITIONING |
| `MographMusicFit` | audio, music, skeleton?, xfade_ms 80, tail_fade_ms 30, match_loudness false | AUDIO (exact), MOGRAPH_MUSIC |
| `MographMusicRoute` | request, user_audio / user_music / score_audio / score_music (lazy) | AUDIO, MOGRAPH_MUSIC |

- With a request, Skeleton takes length, seed (`sub_seed(seed, "score")`), and bpm / key / mood from `plan_rules.music_plan` (music prompt, then
  brief, then rules from the words), the same priority as `MographMusicEncode`. Explicit bpm / key / mood override.
- `MographMusicStyleEncode` is `TextEncodeAceStepAudio1.5` for audio-to-audio: tags = the tags field, else the request's music tags, else the
  mood's; any "NNN bpm" in the prompt is replaced by the grid's tempo; `no vocals` is added; bpm, duration, time signature and keyscale come from
  the grid; `generate_audio_codes` off (audio reference), LM settings as `workflows/api/acestep_a2a.json`. A new node because `keyscale` is a Combo.
- `MographMusicFit`: longer than the target -> cut + `tail_fade_ms` fade. Shorter, or silent in the last 0.25 s (ACE padding) -> the skeleton's
  tail takes over through an equal-power `xfade_ms` crossfade, gain-matched to the ACE audio just before the join. No skeleton input -> fade + silence.
  Resamples to the grid's rate. Grid fields are kept; only `curves` are re-measured. `match_loudness` matches integrated loudness to the skeleton
  (+-12 dB, peak <= -1 dBFS); off by default because `MographMaster` normalises to -14 LUFS anyway.
- `MographMusicRoute` is the switch the core graph needs (below): the user's audio branch (MusicCut) when `request.audio` exists, else the score
  branch; lazy, so the other branch never executes (skeleton and ACE never run for user audio; MusicCut never runs without it).

## Choosing denoise
KSampler always runs 8 steps; `denoise` sets where on the schedule it starts. With ModelSamplingAuraFlow shift 3 the start noise level is about
`sigma = 3d / (1 + 2d)` (estimate from the sampler code, not measured on the box): 0.3 -> 0.56, 0.5 -> 0.75, 0.7 -> 0.88. Flow matching mixes
`(1 - sigma) * skeleton + sigma * noise`, so the skeleton keeps about 44 % / 25 % / 12 % of its amplitude at the start.

| denoise | expect | use when |
|---|---|---|
| 0.3 | timing and chords of the skeleton survive almost exactly; timbre only partly ACE (SoundFont character shows through) | the video cuts are placed on the beats and must land |
| 0.5 (default in the test graph) | grid held, drums / bass re-voiced, ACE arrangement detail and fills | start here |
| 0.7 | ACE sound and detail, tempo still near the grid but individual hits drift, chords may change | the skeleton is a loose guide; check the result against the grid |
| > 0.8 | mostly ACE text-to-music; the grid guarantee is gone | do not use |

The audio length is fixed by the latent whichever denoise is used. Raise `cfg` / change `steps` only after the first listening round.

## What to listen for (box test, `app/graphs/music_exact_test.json`)
1. **Downbeat check.** Play the "OUT restyled raw" file against a click at `grid.beats`: do kicks sit within ~20 ms of the beats at 0.3 / 0.5 / 0.7? Where do they start to slide?
2. **Tempo and phase.** The tracker (`music_analysis`) given the grid bpm should return it (it does on the skeleton: 119.98 for 120, 80.15 for 80); on the ACE output a half / double reading is the old failure, the grid makes it irrelevant.
3. **Tail.** "OUT music" must end at exactly L (`fit` mode `cut` in the node log). Mode `crossfade` means ACE went silent early: listen at the join (about `live end`) for a jump in timbre.
4. **Hits and the build.** Does the riser land on the hit? Does the drop (bar 4 of 8) arrive as a change, or has ACE smoothed it?
5. **Loudness / stereo.** ACE output vs skeleton: is Master pushing hard (check the Master log line)? Is the intro too quiet at high denoise?
6. **Style.** Does the music prompt show (instruments, genre) while the groove stays? If the SoundFont character dominates at 0.3, raise to 0.4-0.5 or sharpen the tags.
7. **Seed.** Same seed twice must give the same skeleton; ACE itself is stochastic per seed.

Unknowns only the box can settle: the real denoise-to-timing curve above, how ACE treats a 48 kHz VAE-encoded stem with `generate_audio_codes` off,
and the decode length (latent hop is 1920 samples = 40 ms at 48 kHz, so +-1 frame mismatches are expected; Fit absorbs them).

## Core graph change (not applied; `tools/app/build_graphs.py` and `app/graphs/fusion*.json` belong to another worker)
In `build_graphs.py`, non-draft branch ("music: ACE -> cut -> master"), the `fusion.json` ids become:
```
add skel       MographScoreSkeleton   request=request, brief=<qwen> (omit if none); hits "" / sections "" (the plan needs the music first)
replace ace_enc  MographMusicEncode -> MographMusicStyleEncode   clip=ace_clip, music=[skel,1], tags="", request=request, brief=<qwen>
delete  ace_lat (EmptyAceStep1.5LatentAudio)
add     ace_venc VAEEncodeAudio       audio=[skel,0], vae=ace_vae
ace_ks  latent_image=[ace_venc,0]  denoise=0.5  (seed, steps 8, cfg 1, euler/simple unchanged)  title "ACE sampler (a2a)"
ace_zero, ace_dec unchanged (ace_zero.conditioning=[ace_enc,0])
add     ace_fit MographMusicFit       audio=[ace_dec,0], music=[skel,1], skeleton=[skel,0]
cut     MographMusicCut: drop ace_audio; music_bpm = 100 (constant; the user path takes its hint from the music prompt); user_audio=audio
add     route   MographMusicRoute     request=request, user_audio=[cut,1], user_music=[cut,0], score_audio=[ace_fit,0], score_music=[ace_fit,1]
master  audio=[route,0], music=[route,1]
plan.music, type.music, compose.music: [cut,0] -> [route,1]       (assemble.audio stays [master,0])
```
Same helpers exist in `tools/fusion/music_graph.py` (`ace_loaders()`, `a2a_chain(skel, seed_link, denoise_link, tags_link)`). Sidecar `stage_labels`: "Music style
encode (a2a)", "Score skeleton", "ACE sampler (a2a)", "Music fit". The `seconds` / `bpm` outputs of the old MographMusicEncode are no longer used.
`MographMusicEncode` stays for the text-to-music fallback. Draft graph (no ACE, local): `score_audio=[skel,0]`, `score_music=[skel,1]` gives exact-length music
with a known grid instead of the cached `default.flac` and needs no GPU. Hits: a skeleton cannot take the plan's cut times without a loop (plan needs
the music first); the planner already aligns its windows to the skeleton's bars. For pieces planned from JSON (`MographPieceFromJSON`) pass `hits` explicitly.

## Local drafts (CPU, no ACE)
`16_app/music_check/` (24-bit WAV 48 kHz, new files, never overwritten; `python -m mograph.score_skeleton --out ...` adds `_r02` instead of overwriting):

| file | length | grid | key / mood | skeleton LUFS / dBTP | mastered |
|---|---|---|---|---|---|
| `skeleton_calm_12s_80bpm.wav` (+ `_master.wav` film preset, `_grid.json`) | 12.000 s, 576000 samples | 4 bars at 80 bpm, hit at 6.0 s | F major, calm / soft | -16.05 / -1.6 | -14.05 / -1.1 |
| `skeleton_energetic_16s_120bpm.wav` (+ `_master.wav` club preset, `_grid.json`) | 16.000 s, 768000 samples | 8 bars at 120 bpm (hint 122), hit at 8.0 s | G minor, driving / electronic | -16.07 / -1.6 | -14.01 / -1.1 |

Measured only (nobody has listened yet): both are bass-heavy (low 59 % / 72 % of the power, >4 kHz about 1.5-2 %, like the ACE songs in AUDIO.md),
tracker recovers the tempo within 0.2 %. Check by ear: the intro is quiet but audible (bar RMS -20 dB vs -16 dB in the drop), the 8-bar build ends with a snare
roll and riser into the hit at 8.0 s, the calm piece's piano strum and harp arp are sparse at 80 bpm, and the electronic e-piano has no whistle (notched).

## Run
`python -m pytest tests/test_score_skeleton.py -q` (about 100 s: SoundFont renders). Local ComfyUI check on 8190: graph skeleton -> Fit -> Master -> PreviewAudio
gave exactly 576000 samples for 12 s. Box: submit `app/graphs/music_exact_test.json` through `progress_standin.py submit` (fields: length, bpm, music, denoise, seed;
outputs: skeleton, ACE raw, fitted + mastered). The graph uses `SaveAudioAdvanced`, which the local ComfyUI 0.19.3 lacks (the box has it, as in `acestep_a2a.json`).
