# TIERS - the best model per stage for a 16 GB and a 24 GB GPU (open weights only, no API), 2026-10-28

**NOTHING HERE HAS RUN ON A 16 GB OR 24 GB GPU.** Measured data exists only for the RTX 3060 12 GB (`tools/app/llm_arms.py`, `_info/BUILT.md` "Speeds"). Every VRAM figure below marked *est.* is an
estimate from those probes; every pick is "best quality that the sizes say fits", not a benchmark result. File sizes are bytes from the Hugging Face tree API (`/api/models/<repo>/tree/main`, 2026-10-28).
Cards: `card-library:motion-graphic-16gb` 0.1.0 (new) and `card-library:motion-graphic` 0.3.0 (24 GB, upgraded). Graphs: `app/graphs/fusion_design_qwen_auto_layers_<arm>.json`, built by
`python tools/app/llm_arms.py build --arm <arm> --base fusion_design_qwen_auto_layers` (`TIERS` + 4 new `ARMS` in `tools/app/llm_arms.py`; tests `tests/test_llm_arms.py`).


> **Qwen removed 2026-10-30 (user decision: Gemma only).** Qwen3.8-27B is no longer a design model on any tier: 24 GB and 16 GB defaults are Gemma (31B QAT / 12B Q8_0), the 3060 stays Gemma 12B Q6_K. The Qwen3.8-27B rows and reasoning below are kept as history only.

## 1. Picks at a glance
| stage | 12 GB (3060, built) | **16 GB** | **24 GB** |
|---|---|---|---|
| design + concept LLM (llama.cpp GGUF, KV q8_0, vision; **n_ctx 16384 on the 3060, 24576 on 16 GB, 32768 on 24 GB**, section 9) | Gemma 4 12B Q6_K (measured) | **Gemma 4 12B Q8_0** (default) | **Gemma 4 31B QAT UD-Q4_K_XL** (default) |
| stills DiT / encoder | MOSS SFT int8 convrot / Qwen3.5-9B w4a8 gptq | MOSS SFT **bf16** / Qwen3.5-9B **int8 convrot** | MOSS SFT **bf16** / Qwen3.5-9B **bf16** |
| video | LTX 2.5 distilled int8 convrot, streamed, preview 384x704 | same file, streamed (~8 GB/step est.), **draft 576x1024** | same file, resident at draft (tight), **draft 576x1024** |
| LTX text encoder | gemma4-12b int8 convrot, streamed | same, resident while encoding | same, resident |
| music | ACE-Step 1.5 XL base bf16 | same | same |
| depth / flow | Video-Depth-Anything-Small / RAFT | same | same |
| ComfyUI flags | `--mmap-torch-files --cache-ram 16 --reserve-vram 1.5` | same | `--reserve-vram 1.0` |

Why no change for music / depth / flow: ACE XL base bf16 (9.29 GiB) is the approved route and already the largest ACE; Video-Depth-Anything Base (`video_depth_anything_vitb.pth`, 458,247,082 B) and
Large (`..._vitl.pth`, 1,538,392,012 B) are **CC-BY-NC-4.0** (only Small is Apache-2.0; `build_graphs.check()` asserts Small); RAFT self-downloads.

## 2. Design LLM candidates (all `unsloth/*-GGUF`, Apache-2.0, vision = F16 mmproj)
Estimate: peak = file + 1.45 GiB (KV q8_0 at 16384, compute buffer, CUDA context; the token-embedding table stays on the CPU) + projector 0.5 GiB + its size when it runs on the GPU. Calibrated on the
3060: Qwen3.8-27B IQ3_XXS 10.18 -> 11.6 measured (projector on the CPU), Gemma 12B Q6_K 9.11+0.16 -> 10.82 measured. Usable VRAM: 16 GB cards report 15.99 GiB, 24 GB cards 23.99 GiB (L4: 22.5 GiB).
| model | file | bytes | GiB | est. peak GiB | 16 GB | 24 GB | note |
|---|---|---:|---:|---:|---|---|---|
| Qwen3.8-27B | UD-IQ3_XXS | 10,934,860,704 | 10.18 | 11.6 *measured* | yes | yes | the 3060 arm: 94.6 % fields, 3/3 valid, 113 s/call |
| Qwen3.8-27B | UD-Q3_K_XL | 13,146,393,504 | 12.24 | 13.7 | yes | yes | 16 GB fallback if IQ4_XS runs out of memory |
| **Qwen3.8-27B** | **UD-IQ4_XS** | **14,252,845,984** | **13.27** | **14.7** | **yes (1.3 spare), pick** | yes | projector on the CPU as measured |
| Qwen3.8-27B | UD-Q4_K_M | 16,464,440,224 | 15.33 | 16.8 | no | yes | |
| **Qwen3.8-27B** | **UD-Q5_K_M** | **19,771,509,664** | **18.41** | **21.2 (projector GPU)** | no | **yes (2.8 spare), pick** | |
| Qwen3.8-27B | UD-Q6_K | 21,983,677,344 | 20.47 | 21.9 (projector CPU) / 23.3 | no | borderline | < 1 GiB spare with the projector on the GPU |
| Gemma 4 12B | Q8_0 | 12,669,647,680 | 11.80 | 13.5 | yes (alt) | yes | measured Q6_K: 93.3 % fields, 54 s/call, 27 tok/s |
| Gemma 4 12B | UD-Q8_K_XL | 13,637,327,680 | 12.70 | 14.4 | yes | yes | |
| Gemma 4 26B-A4B (MoE) | qat UD-Q4_K_XL | 14,249,047,104 | 13.27 | ~14.5 | tight | yes | not chosen: the only measured 26B arm (IQ3_S) kept 84.9 % of fields (9.3 fixed per reply) |
| Gemma 4 26B-A4B | UD-Q4_K_M | 16,947,541,728 | 15.78 | 17.2 | no | yes | |
| Gemma 4 26B-A4B | UD-Q5_K_M / UD-Q6_K | 21,150,365,408 / 23,172,478,688 | 19.70 / 21.58 | 21.2+ / 23+ | no | Q5 yes, Q6 no | |
| Gemma 4 31B (dense) | UD-IQ3_XXS | 11,837,782,976 | 11.02 | ~14 | yes | yes | untested alt for 16 GB |
| **Gemma 4 31B** | **qat UD-Q4_K_XL** | **17,287,670,048** | **16.10** | **~20 (projector 1.12 GPU)** | no | **yes, alt** | quantization-aware 4-bit; no 31B probe exists |
| Gemma 4 31B | UD-Q5_K_M / Q6_K | 21,658,401,728 / 25,201,486,784 | 20.17 / 23.47 | 24+ | no | no | |
| Qwen3.6-27B | UD-Q4_K_XL | 17,612,564,704 | 16.40 | 18+ | no | yes | predecessor of Qwen3.8, same mmproj size; skipped |
Why Qwen3.8-27B on both tiers: it is the arm with the best measured design quality (94.6 % of fields kept, 3/3 valid first pass, least guide-copying) even at 3 bits, so more bits of the same model is the
only upgrade backed by data; Gemma 4 12B (93.3 %) is within the seed noise (27B 89-97 %, 12B 91-96 %) and 2x faster, hence the alternate arm. Rendered taste was never compared (`LLM_ARMS.md` section 8).

## 3. Video (LTX 2.5 distilled 22B) and its text encoder
| candidate | file | bytes | GiB | note |
|---|---|---:|---:|---|
| **int8 convrot (official)** | `Lightricks/LTX-2.5` `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21,504,034,224 | 20.03 | **pick, both tiers.** MAE 4.10 / PSNR 29.7 dB vs bf16 (JoaoZaokk README, one 3090, 512x512, 3 steps); measured working on sm_86 (3060) |
| bf16 | `...-distilled-transformer-bf16.safetensors` | 42,018,190,584 | 39.13 | fits neither tier |
| nvfp4 | `...-distilled-transformer-nvfp4.safetensors` | 18,721,548,408 | 17.44 | native FP4 only on Blackwell; > 16 GB anyway, no quality measure; skipped |
| W4A8 (JoaoZaokk) | `ltx-2.5-22b-distilled-transformer-w4a8.safetensors` | 12,520,267,816 | 11.66 | fits 16 GB whole, but its own README: MAE 7.81 (1.9x worse picture, 2.9x worse sound than int8); IC-LoRA use untested. Commented fallback row |
| W4A8 mixed (tsolful) | `ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors` | 15,412,311,232 | 14.35 | "80 % INT8 quality" is the uploader's claim; does not fit 16 GB with activations either |
| GGUF (unsloth/LTX-2.5-GGUF `distilled/`) | Q4_K_M / UD-Q4_K_XL / Q5_K_M / UD-Q5_K_XL / Q6_K / Q8_0 | 13,584,703,872 / 14,000,234,880 / 15,631,524,224 / 16,000,033,152 / 17,806,270,848 / 22,779,666,816 | 12.65 / 13.04 / 14.56 / 14.90 / 16.58 / 21.22 | needs `UnetLoaderGGUF` in every LTX graph, dequantises on the fly, not measured against int8 convrot; Q5 is not whole in 16 GB with latents either. Q4_K_M kept as a commented row |
| text encoder gemma4-12b int8 convrot | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15,372,969,374 | 14.32 | **pick**; the bf16 file is 26,263,858,182 B (24.46 GiB) and the GGUF encoder (elix3r, Q5_K_M 8.86 GiB) needs a ComfyUI-GGUF patch |
16 GB: the 20.03 GiB transformer cannot be resident next to latents and the VAE decode; Dynamic VRAM keeps ~10-12 GiB and streams the rest per step (*est.*; the 3060 streams ~8-9 GiB, measured), so the card
asks for PCIe >= 8 GB/s. "Nothing good fits fully" -> best-quality streamed (int8 convrot), as the task said. 24 GB: ~20 GiB + draft latents/decode *est.* ~23 GiB, i.e. resident but with ~1 GiB spare;
the longest draft shots (IC cap 289 f) may still stream a little.
**Quality options that exist:** `plan_rules.sizes_for`: preview 384x704 (single stage), draft 576x1024 (single stage), final 576x1024 stage 1 + x2 upscale to 1152x2048. The design / layers graphs are built
single-stage (`add_ltx_shared(single=True)`), so only preview and draft run in them. 2026-10-30: the app shows two choices, Preview (held stills, `16_app/STILL_PREVIEW.md`) and Final, and Final renders at `still_preview.APP_FINAL_TIER = "draft"` (one line to switch to "final"); the tier graphs default to Final. A two-stage (final) three-layer route
is NOT built (clean / text / alpha chains would each need the stage-2 sampler); the x2 upscaler file is in the cards anyway (`fusion.json` uses it). That is the gap between 24 GB and a real "final" tier.

## 4. Stills
Comfy-Org/MOSS-Image: `moss_image_sft_bf16` 10,164,655,808 B (9.47 GiB), `moss_image_sft_int8_convrot` 5,413,698,958 (5.04); encoder `qwen3.5_9b_bf16` 19,306,312,328 (17.98), `qwen3.5_9b_int8_convrot`
9,748,121,532 (9.08), `qwen3.5_9b_w4a8_gptq` 7,343,591,088 (6.84); VAE `flux2-vae` 336,211,292 / `flux2-vae_small` 249,521,340 (kept: box-proven; whether the full VAE is better is untested).
16 GB: DiT bf16 + encoder int8 convrot (each whole, one after the other). 24 GB: both bf16 (17.98 GiB encoder alone fits). models.json already listed the bf16 DiT as a 32 GB item. The stills only feed the
first frame of LTX, so the expected gain is small; **the two swapped files have not been run in the app's stills chain** (tier graphs set `still_unet` / `still_clip`). 2026-10-29: a piece with several places runs one MOSS still per place (up to 6, `fusion_design_qwen_auto_layers*` etc. `still{k}_*` chains, lazy: only the places the plan has), one after the other, on the SAME loaders; on 24 GB the bf16 encoder (18 GiB) and the DiT (9.5 GiB) do not sit together, so ComfyUI may swap them per still: NOT MEASURED, each still is the 25-step sampler + a decode.

## 5. Expected VRAM per stage (*est.*, GiB) and quality
| stage | 16 GB | 24 GB |
|---|---|---|
| design / concept / repair / legibility read (one llama.cpp load each, unloaded after) | 15.0 (Qwen IQ4_XS, n_ctx 24576) / 13.8 (Gemma Q8) | 21.8 (Qwen Q5_K_M, n_ctx 32768) / 20.4 (Gemma 31B QAT) |
| ACE-Step XL base + qwen_4b encoder | ~10 / 8 whole | whole |
| MOSS stills | encoder 9.1 then DiT 9.5 whole | encoder 18.0 then DiT 9.5 whole |
| LTX encode | gemma int8 14.3 whole | whole |
| LTX sample (clean i2v, union V2V, Alpha-Gen matte) at draft 576x1024 | ~11 resident + ~9 streamed per step | ~20 + latents, ~1 spare |
| VDA-Small, RAFT, compose (torch), type passes (CPU) | small; `--reserve-vram` 1.5 | small; `--reserve-vram` 1.0 |
Quality setting: **draft** (576x1024 LTX, compose 540x960, `crf 20`) on both; preview (384x704) stays selectable. KV q8_0, V2V start 0.975 (4 steps) unchanged; n_ctx per tier (24576 / 32768, section 9). `tests/test_llm_arms.py` asserts all three.

## 6. GPUs
- **16 GB** (`gpu_types`, `min_vram_gb 16`, PCIe >= 8): RTX 4060 Ti 16G, RTX 5060 Ti 16G, RTX 4070S Ti, RTX 5070 Ti, RTX 4080, RTX 4080S, RTX 5080, RTX A4000. The 8 GB 4060 Ti is excluded by `min_vram_gb`. Vast, 2026-10-28
  (1 GPU, >= 64 GB RAM, reliability > 0.95): 5060 Ti $0.15-0.20/h, 4060 Ti 16G $0.16-0.18, 4070S Ti $0.27-0.44, 4080S $0.31, 5080 $0.38; card `max_price` 0.45.
- **24 GB** (`min_vram_gb 24`): RTX 3090, RTX 4090, RTX A5000 and the card's existing 5090 / L40S / A6000. The L4 (22.5 GiB usable) and a 22 GiB card do not qualify. Vast: 3090 $0.21-0.40/h; `max_price` 0.80 kept.
- Blackwell consumer cards (5060 Ti / 5070 Ti / 5080 / 5090): the llama-cpp-python wheel's CUDA architecture list is not published; the cards' GPU smoke test (load the pick, generate 8 tokens) fails loudly at provisioning if there is no kernel image.

## 7. Cards and graphs
| tier | card | disk | downloads | default graph | alternate graph |
|---|---|---|---|---|---|
| 12 GB | `motion-graphic-3060` 0.2.0 | 130 GB | 90.18 GiB / 96.83 GB, 20 rows | `..._auto_layers_gemma12b_q6` | (none; Qwen removed 2026-10-30) |
| 16 GB | `motion-graphic-16gb` 0.1.0 | 150 GB | 102.62 GiB / 110.19 GB, 20 rows | `fusion_design_qwen_auto_layers_gemma12b_q8` | (none; Qwen removed 2026-10-30) |
| 24 GB | `motion-graphic` 0.3.0 | 170 GB | 121.92 GiB / 130.91 GB, 20 rows | `fusion_design_qwen_auto_layers_pivot_gemma31b_qat` (Gemma 4 31B QAT, DEFAULT since 2026-10-29, user: "use gemma as the default design model") | (none; Qwen removed 2026-10-30) |
Mechanism (smallest that works): `llm_arms.ARMS` entries with `tier` / `measured=False` / `vram_est_gib` / `bases`, and `llm_arms.TIERS[16|24]` = stills loader files (`still_unet`, `still_clip`), default Quality (draft,
"final" removed from the options), `server_flags`, `why`; `convert()` applies them and writes `s["tier"]` into the `.app.json`. `check()` allows exactly those loader inputs to differ from the base graph.
The video, music and depth loaders are the same files on every tier. Nothing picks the tier graph by `/system_stats` yet (EXTRAS item 1): the page's graph list / the card's default graph does it by hand.
Cards were produced by a one-shot script (not committed: `motion-graphic-16gb` from the 3060 card, `motion-graphic` 0.3.0 from 0.2.1 + the same flags / GGUF / verify / reaper blocks); `tools/make_card.py` was NOT
edited (0.2.0 / 0.2.1 are already hand edits it would drop). Both cards: llama-cpp-python 0.4.5+cu130 wheel (cp312, `pip install --no-deps`, diskcache), verify
`llama_supports_gpu_offload() == True`, GGUF file-size check, a GPU smoke generation, `--mmap-torch-files --cache-ram 16 --reserve-vram N [--use-sage-attention]` patched into `comfyui.sh` and checked against
`/system_stats` argv, idle reaper 60 + 60 min + container-CPU guard (`/workspace/NO_REAP`, `KEEPALIVE`). `models.json` got the two new encoder entries.

## 8. UNVERIFIED (everything not on the 3060)
- All 16 / 24 GB VRAM peaks, speeds, and that Qwen3.8-27B IQ4_XS (16 GB) / Q5_K_M with the projector on the GPU (24 GB) load without an out-of-memory; the Gemma 31B QAT (no probe at all).
- LTX int8 on 16 / 24 GB: how much streams, step times, draft shot caps (289 f IC) at 16 GB; Alpha-Gen + union-control + the V2V text pass at 576x1024 (only 384x704 was probed).
- MOSS bf16 DiT + int8-convrot / bf16 encoder inside the app's stills chain; Blackwell and Ada kernels of the llama.cpp wheel and of the int8 convrot path (only sm_86 seen).
- `scripts/_common.py` MODELS lacks the GGUF / new MOSS names, so `tests/test_install_scripts.py::test_every_model_in_the_gpu_graphs_has_a_download_entry` fails (it already failed for the 3060 arms before this work).
- No rendered comparison of design quality across models (taste), no tier selection from `/system_stats`, no two-stage "final" for the three-layer design graphs.

## 9. Context the par-quality prompts need (2026-10-29, par round `20_open-model-design/v003/review/PAR.md`; the n_ctx / KV numbers are ESTIMATES, nothing here ran on a 16 / 24 GB card)
Measured prompt sizes (`usage.prompt_tokens`, text + one image): the piece-from-brief prompt (shared rules, brief, music grid, PLAN visual decisions, module menu, piece schema, op blocks, fonts) is **15.7-16.3 k tokens** with Claude's own inputs only (round r0),
**18.1-18.7 k** with the principles + syntax notes (r1), **18.7-19.7 k** with the self-check and the shot-structure principle (r2 / r4 / r5); without the module menu and the visual-decisions notes (r3) **12.6-12.9 k**, with no quality loss on the rubric (Qwen 9.5 = r2, Gemma 8.5 vs 9.0, inside the seed noise of 0.5). Replies: 2.0-2.4 k tokens (Gemma median) and 3.2-4.5 k (Qwen, up to 8.7 k)
for a 3-6 word piece. The APP's design prompt after this work (guide sections 13-17 + schema with the system block / op blocks) is **12.5 k tokens of text + the 960x1704 sheet (~1.5-2 k)** against 8.8 k + sheet before; the concept prompt is ~1.5 k. So: n_ctx 16384 no longer holds the app's design call (14-18 k prompt + 1.5-5 k reply); 24576 does, 32768 holds the full par prompt (19.7 k + 9 k).
KV cost, from the 3060 probes (LLM_ARMS section 5): Qwen3.8-27B q8_0 KV = 544 MiB at n_ctx 16384 (16 full-attention layers) + 150 MiB recurrent state, so ~0.8 GiB at 24576 and ~1.1 GiB at 32768 (+0.3 / +0.6 GiB over the table above): the 24 GB pick
(Q5_K_M, est. peak 21.2 GiB at 16 k) would be ~21.5 / 21.8 GiB of 23.99, 2.2 GiB spare. Gemma 4 31B QAT: sliding-window layers make the KV grow slowly (12B: 391 MiB at 16 k), est. 20 -> ~20.4 GiB at 32 k. The 16 GB pick (Qwen IQ4_XS, 14.7 GiB at 16 k)
would be ~15.0 GiB at 24576 and ~15.3 of 15.99 at 32768: use 24576 there. Prompt processing is the cost: the 3060 measured 225 tok/s for the 27B (IQ3_XXS), i.e. ~65 s to first token for a 14.5 k prompt and ~85 s for 19 k; the 24 GB card is **not measured**.
**Decision 2026-10-29 (graphs built):** 24 GB arms (`qwen38_27b_q5km`, `gemma31b_qat`) `n_ctx` **32768** on EVERY llama.cpp node (concept, design, both repairs, the legibility read `ltx_read`; the repair carries the whole guide, so it is not capped at 16384 on a tier arm), KV q8_0, projector stays ON the GPU: est. 21.8 of 23.99 GiB for Qwen Q5_K_M (2.2 GiB spare) and 20.4 for Gemma 31B QAT, so there was no reason to move the projector to the CPU. 16 GB arms (`qwen38_27b_iq4xs`, `gemma12b_q8`) `n_ctx` **24576** (est. 15.0 of 15.99 with the projector on the CPU as before; Gemma 12B Q8 13.8), because 32768 would leave 0.7 GiB. The 3060 arms keep 16384 (measured there); the app design prompt (12.5 k text + sheet) plus a reply no longer fits that, so a 3060 run of the new prompts is unverified. `llm_arms.CTX_16 / CTX_24`, `TIERS[t]["n_ctx"]`, `ctx_for()`; `check()` and the tests assert it per tier. All VRAM numbers are ESTIMATES, nothing ran on a 16 / 24 GB card.

## 10. FINAL = whole-video Real-ESRGAN upscale to 1080x1920 (user decision 2026-10-30; hand-off for build_graphs.py)
The app's Final render stays at draft sizes (`still_preview.APP_FINAL_TIER = "draft"`, LTX 576x1024, compose 540x960) and the FINISHED video is upscaled whole, type included, to exactly 1080x1920 (1920x1080 for 16:9). Not generative: Real-ESRGAN (BSD-3-Clause) is a feed-forward SR net, the type keeps its shapes. Rejected after the v022 / v023 tests: a native 1080 re-render of the piece JSON with upscaled footage (17 min per piece and a different word schedule from the current type engine), TiledFusion / LTX upscale (dropped by the user, too expensive).
- **Code**: `tools/app/final_upscale.py` (`tests/test_final_upscale.py`, 12 pass). Fragment = core ComfyUI nodes only (no restart): `GetVideoComponents` (video -> images, audio, fps) -> `UpscaleModelLoader` + `ImageUpscaleWithModel` -> `ImageScale` (lanczos, exact width x height, crop disabled) -> `CreateVideo` (fps and audio from the split, unchanged) -> the existing `SaveVideo`.
- **Model by source size** (`pick_model`): 540x960 (draft) -> `RealESRGAN_x2plus.pth` (exactly 1080x1920, no resize step); 360x640 (preview) -> `RealESRGAN_x4plus.pth` (1440x2560, then lanczos down to 1080x1920); 16:9 swaps the sides; quality final (already 1080 high) -> no-op. Files: card rows `upscale_models/RealESRGAN_x2plus.pth` (67,061,725 B) and `RealESRGAN_x4plus.pth` (67,040,989 B) in the 24 GB, 16 GB, 3060 cards (already added).
- **Hook for the integration worker** (build_graphs.py: do it where the tier graphs get their SaveVideo, i.e. after `assemble` / `out_final` exist, only for the Final choice of the app, not for Preview stills):
  ```python
  from tools.app.final_upscale import add_final_upscale
  add_final_upscale(graph, quality="draft", fmt="9:16", video="assemble", save="out_final", prefix="up", keep_draft=False, model=None, src=None) -> graph
  ```
  It adds `up_comp, up_model, up_run, up_fit, up_vid` and re-points `graph["out_final"]["inputs"]["video"]` at `["up_vid", 0]`; `keep_draft=True` also keeps a second SaveVideo `out_final_draft` with the 540x960 video. `MographArchive.video` should keep linking `assemble` (the archive records the draft video, the ingredients stay draft size). Pass the request's quality / format (`fmt` = `plan_rules.format_of(piece)`); `src=(w, h)` overrides the tier size. The Final graph's outputs (dashboard tile, project version, motngrfx-tests) are unchanged: the same `out_final` node saves the 1080 file.
- **Existing mp4** (a finished piece on the box): `python tools/app/final_upscale.py plan --size 540x960`, `... graph --video "projects/.../x_video_00001_.mp4 [output]" --size 540x960 --prefix projects/.../x_up1080 --out g.json`, then submit `g.json` with `progress_standin.py submit` (the `[output]` suffix makes core LoadVideo read from the output folder). Streaming runner without ComfyUI (what was measured): `tools/app/upscale_finished.py run --src X.mp4 --model .../RealESRGAN_x2plus.pth --out OUT.mp4 --lanczos-out BASE.mp4 --report R.json` (waits for >= 6 GiB free GPU, ~1-2 GB used).
- **Measured** (RTX 4090 box, 384 frames, card shared with other renders): x2plus 540x960 -> 1080x1920 230 s / 194 s per piece = 1.7-2.0 frames/s, process VRAM 0.95 GB (torch alloc 213 MiB); x4plus 360x640 185 s = 2.1 frames/s, 1.9 GB; lanczos-only baseline 8 s. Sheets: `22_batch-24gb/v023/review/sheet_p1..p3.png` (1:1 crops: type edge, subject, flat area; lanczos left, Real-ESRGAN right). The ComfyUI-graph form ran once on a lone 81-frame 540x960 clip (core LoadVideo `[output]`): ComfyUI 38.5 s (upscale 33.0 s = 2.45 frames/s, save 5.0 s), peak VRAM 2004 MiB, result 1080x1920 24 fps 81 frames; a full 384-frame piece through the graph is not measured. Memory of the graph form: `ImageUpscaleWithModel` returns the whole clip as one float32 CPU tensor (384 frames at 1440x2560 = 17 GB before the lanczos step, 9.5 GB for x2); on a 64 GB box next to other jobs use the streaming runner for long pieces or chunk the video.
- **Known limits**: a draft video is already h264 (crf 20) and the net sharpens its block edges and grain too (the grain of the grade shows as fine noise on flat areas); frames are upscaled independently, no temporal model: over 6 consecutive frames (v018 r03 f172-177, 1:1 crops) the mean frame-to-frame change is +30 to +50 % over lanczos (flat sky 0.17 -> 0.22, hull 3.0 -> 4.0, type edge 5.1 -> 7.0; v019 flat 0.48 -> 0.71), part of it from the sharper edges moving, part added shimmer; watch for crawling grain and edge shimmer in motion, not measured by eye on playback; type edges inside the video are redrawn only as sharpening, a native type render at 1088x1920 is crisper (v022 sheets) but costs a full compose.
