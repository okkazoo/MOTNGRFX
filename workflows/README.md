# Workflows

## Which graph do I use?

Pick the graph for your GPU. Each one is an API-format workflow in `app/graphs/`; the tier graphs have a `.app.json` sidecar. Open it through the web app (`/motngrfx`, `template` dropdown) or queue it with ComfyUI's `/prompt`.

| GPU VRAM | Graph | Design model |
|---|---|---|
| 12 GB (RTX 3060) | `app/graphs/motngrfx_3060.json` | Gemma 4 12B Q6_K |
| 16 GB | `app/graphs/motngrfx_16gb.json` | Gemma 4 12B Q8_0 |
| 24 GB (RTX 4090 / 3090) | `app/graphs/motngrfx_default.json` | Gemma 4 31B QAT (GGUF, llama.cpp) |
| No GPU | `app/graphs/local_v1.json` | none (rules-only CPU renderer, no models) |

`motngrfx_default` is the graph the web app runs by default and the one the examples were made with. Gemma designs the piece, ACE-Step 1.5 makes the music, MOSS-Image the stills, and LTX 2.5 22B image-to-video the footage with Alpha-Gen mattes, clean plates and union-control text passes. Then come 3D depth cuts between shots, compose and assemble. The 12 GB and 16 GB graphs are the same pipeline with a 12B design model and smaller model files, but they have plain cuts: the 3D depth cuts (`MographPivotCut`) are only in `motngrfx_default`. They were not run end to end on the current graphs.

Gemma is the only design model. All graphs need `llama-cpp-python`, the GGUF files and the other models described in the top-level [`README.md`](../README.md) (Install). The old `fusion_design_qwen_*` file names are a name prefix from earlier builds; the graphs listed below run Gemma.

## Other graphs in `app/graphs/`

The graphs are generated (`tools/app/build_graphs.py`, `tools/app/llm_arms.py`); do not edit them by hand.

- **Aliases of the tier graphs:** `motngrfx_default` has the same content as `fusion_design_qwen_auto_layers_pivot_gemma31b_qat`, `motngrfx_3060` as `fusion_design_qwen_auto_layers_gemma12b_q6`, and `motngrfx_16gb` as `fusion_design_qwen_auto_layers_gemma12b_q8`.
- **Schema variant:** `fusion_design_qwen_auto_layers_pivot_gemma31b_qat_schema` is the default graph with the reply-schema grammar turned on (`16_app/REPLY_SCHEMA.md`).
- **Plain cuts:** `fusion_design_qwen_auto_layers_gemma31b_qat` is the 24 GB graph without the 3D cuts: three-layer shots (clean clip, Alpha-Gen matte, text background) with plain cuts. The 12B tier graphs work the same way.
- **Earlier generations:** `fusion_design_qwen_gemma12b_q6`, `fusion_design_qwen_gemma26b_a4b` and the `_auto` ones (one footage chain per shot, or the model also invents the concept). The Gemma 26B A4B arm comes from the design-model comparison in `tools/app/llm_arms.py`.
- **Video route:** `fusion_redesign` replays a saved design reply over existing footage. The app's video route loads it.
- **CPU and test graphs:** `local_v1` (no models) and `music_exact_test` (music-length test).
- **Helper graphs** (names start with `_`): the app runs these behind the page. `_media_read` reads your media, `_story` writes the storyboard, `_layout_preview` previews layouts and `_final_upscale` is the Final (1080) upscale.

## Other folders

- `workflows/api/`: single-purpose API graphs used by the tools (ACE-Step, LTX image-to-video / IC-LoRA control, MOSS text-to-image). Not templates for the app.
- `example_workflows/mograph_basic.json`: the old CPU graph (Analyze -> Plan -> Soundtrack -> Render, no generative models), a UI-format workflow for loading in the ComfyUI canvas. It predates the fusion pipeline.
