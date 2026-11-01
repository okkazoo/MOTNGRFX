# tools/mathtype

Math-driven type and motion tests (11_mathtype). Each script's docstring has its own usage; outputs never
overwrite (gNN generations, `local/GENERATIONS.md`). Install: `pip install -r requirements-mathtype.txt`
(plus ffmpeg on PATH).

New pieces: start from the shared kit (`kit/README.md`: text, draw, camera, timeline, palette, io, qc) and
the 136-line demo `harmonograph.py`. `moire.py --kit` and `contours.py --kit` run E4 / E1 on the kit.

## Running on GPU (torch backend)

`silkwind.py` and `matte_wipe.py` take `--backend numpy|torch` and `--device cpu|cuda`. numpy is the default
and the reference: its output is byte-identical to the pre-port code. On torch everything runs in float32:
- the noise banks, curl flow, domain warp, M(t), projection and trail samples;
- the splat (scatter-add of the densely sampled segments), the per-class Gaussian blurs (cv2's kernel and
  border), the tone map and the area downsample.

The bubble-head coverage stamps are still drawn by cv2 on the CPU (about 400 small patches per frame) and
composited on the device. `--frame-batch N` renders N frames per batch. Use 4 on an 8 GB card and 8 on a
16-24 GB card; a full-size frame peaks at roughly 0.6 GB.

Measured against numpy (`xp_bench.py`, torch 2.9 CPU): PSNR 69-77 dB for silkwind (plain, beaty, matte;
preview and 1080x1350) and 98-102 dB for the matte_wipe compositor. Visually identical.

```bash
DEV="--backend torch --device cuda"
WAV=11_mathtype/v004/local/beat_D_silkwind_beaty_g01.wav
EV=11_mathtype/v004/local/beat_D_silkwind_beaty_g01.json

# silk wind, 10 s loop (v003): preview (360x450, N/4), draft (360x450, 3x ss), final (1080x1350, 2x ss)
python tools/mathtype/silkwind.py --preview $DEV
python tools/mathtype/silkwind.py --draft   $DEV --frame-batch 8
python tools/mathtype/silkwind.py --final   $DEV --frame-batch 4

# audio-reactive (v004): preview / final
python tools/mathtype/silkwind.py --audio $WAV --events $EV --mode beaty --preview $DEV
python tools/mathtype/silkwind.py --audio $WAV --events $EV --mode beaty --final   $DEV --frame-batch 4

# matte for the sea wipe (v005), then the compositor
python tools/mathtype/silkwind.py --audio $WAV --events $EV --mode beaty --matte $DEV \
    --wipe-at 9.6 --wipe-dur 1.2 --wipe-base 3 --wipe-far 0.1 --wipe-light 0.8
python tools/mathtype/matte_wipe.py --a 11_mathtype/v005/local/D_wipe_sideA_g01.mp4 \
    --b 11_mathtype/v005/local/D_wipe_sideB_g01.mp4 --matte 11_mathtype/v005/local/D_wipe_beaty_gNN_matte.mp4 \
    --audio $WAV --k-pre 0.55 --gain 1.8 $DEV

# benchmark + parity check against numpy (writes 11_mathtype/v006/review/xpv_rNN_*)
python tools/mathtype/xp_bench.py --device cuda --frame-batch 8 --throughput 50
```

Determinism: on CUDA, `xp.Backend` turns on `torch.use_deterministic_algorithms` (warn only), sets
`CUBLAS_WORKSPACE_CONFIG`, and disables TF32 for matmul and cuDNN. TF32 would cost about 0.01 rad in the
noise phases. Noise parameters come from the same seeded numpy RNG on both backends. The splat accumulates
with `index_put_(accumulate=True)`, which is deterministic under that flag. CUDA results agree with
torch-cpu to float32 rounding (sin/cos ulps), not bit for bit.
