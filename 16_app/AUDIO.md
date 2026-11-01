# Audio: the "tiny" fix (MographMaster)

User note on the v002 songs: "a bit tiny". What the measurements say about the source files
(`15_fusion/v002/local/pieces/P02/audio.wav`, `P05/audio.wav`, the cut songs that went into the videos):

- **Peaks over full scale.** True peak +0.3 dBTP (P02) and +0.7 dBTP (P05): the songs clip on playback and in any
  loudness-normalising platform.
- **Not at a common level.** P02 -16.3 LUFS (quiet), P05 -12.9 LUFS (loud): the pieces differ by 3.4 LU.
- **Nearly mono.** L/R correlation 0.95 / 0.93; side level is 16 / 15 dB under the mid.
- **Dull top end.** Above 4 kHz only 1.5 % (P02) / 3.4 % (P05) of the power; high/low ratio -17 / -14 dB.

## Chain (`tools/fusion/audio_cut.py: master_audio`, node `MographMaster`)

EQ (highpass 25-28 Hz, low-end weight, less mud around 320-350 Hz, presence 3-3.5 kHz, air shelf 9-10 kHz) ->
glue compressor (`mastering.glue`, release from the tempo) -> mid/side width 1.25 (`mastering.widen`, new) -> mono below
120 Hz (`mastering.mono_below`) -> gain + look-ahead true-peak limiter (`mastering.master`) to **-14 LUFS integrated,
<= -1.0 dBTP**. Presets (EQ + glue only): `club` (default: +1 dB at 55 Hz, -2 dB at 320 Hz, +2 dB at 3.2 kHz, +3 dB
shelf at 9 kHz, glue thr -16 dB / 2:1), `film` (+1 dB shelf 90 Hz, +1.5 dB 3 kHz, +2 dB shelf 10 kHz, 1.8:1),
`neutral` (+1 dB shelf 100 Hz, +1 dB 3.5 kHz, +1.5 dB shelf 10 kHz, 1.5:1). Targets and width are node inputs.

## Before / after (width 1.25, target -14 LUFS, tp_max -1.0)

LUFS and true peak are `mograph.mix.loudness_lufs` / `mastering.true_peak_db`; the figure in brackets is ffmpeg
`ebur128=peak=true` (independent meter, true peak). Band shares are Welch power in 20-250 Hz /
250-4000 Hz / >4000 Hz; "high/low" is 10*log10 of the high share over the low share.

| piece | version | LUFS | true peak dBTP | LRA LU | L/R corr | side/mid dB | low / mid / high % | high/low dB |
|---|---|---|---|---|---|---|---|---|
| P02 | original | -16.3 (-16.2) | +0.3 (+0.3) | 3.1 | 0.951 | -16.0 | 79.1 / 19.4 / 1.5 | -17.2 |
| P02 | club | -14.1 (-14.0) | -1.1 (-1.1) | 2.6 | 0.902 | -12.9 | 79.7 / 18.8 / 1.5 | -17.3 |
| P02 | film | -14.1 (-14.0) | -1.1 (-1.1) | 2.5 | 0.892 | -12.4 | 77.8 / 20.8 / 1.4 | -17.4 |
| P02 | neutral | -14.1 (-14.0) | -1.1 (-1.1) | 2.7 | 0.886 | -12.2 | 76.3 / 22.4 / 1.3 | -17.7 |
| P05 | original | -12.8 (-12.8) | +0.7 (+0.7) | 0.9 | 0.933 | -14.6 | 90.7 / 6.0 / 3.4 | -14.3 |
| P05 | club | -14.1 (-14.0) | -1.1 (-1.1) | 0.9 | 0.865 | -11.4 | 85.9 / 8.1 / 6.0 | -11.6 |
| P05 | film | -14.1 (-14.0) | -1.1 (-1.1) | 0.9 | 0.871 | -11.6 | 86.4 / 8.4 / 5.2 | -12.2 |
| P05 | neutral | -14.1 (-14.0) | -1.1 (-1.1) | 0.9 | 0.881 | -12.0 | 87.4 / 7.9 / 4.7 | -12.7 |

Listen: `16_app/audio_check/P02_master_{club,film,neutral}.wav` and `P05_master_{club,film,neutral}.wav` (24-bit
WAV, 48 kHz; the originals are untouched).

## Reading the numbers

- Fixed for certain: no more clipping (+0.3 / +0.7 -> -1.1 dBTP), the same loudness for every piece (-14.0 +-0.1 by both
  meters), a wider image (correlation 0.95 -> 0.90 / 0.93 -> 0.87, side 3 dB closer to the mid).
- P02 gains +2.2 LU (it was the quiet one), which is the most likely "tiny"; P05 loses 1.2 LU because it was above the
  target and over full scale (it will sound a little less loud, but cleaner and no longer clipped).
- The top end barely moves on P02 (high/low -17.2 -> -17.3 dB): the ACE render has almost nothing above 4 kHz to lift, and the
  limiter pulls the (bass-dominated) mix down about as much as the shelf lifts the air. P05 gets +2.7 dB high/low. A bolder EQ
  (+3.5 dB presence, +6 dB shelf at 7 kHz) was tried: P02 only reaches -16.0 dB and the limiter works 2.5 dB harder, so the
  default stays moderate. If the user still hears "tiny" the next lever is the music prompt (brighter, more mid-range
  instruments), not mastering.
- Bass: sub-60 Hz share on P02 rises 37 -> 41 % (the +1 dB at 55 Hz plus makeup gain), on P05 it falls 42 -> 37 %.
