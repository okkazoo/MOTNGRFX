# Third-party licences

The repository code is Apache-2.0 (`../LICENSE`). Everything below is third-party and keeps its own terms.

| What | Where the licence text / notice is |
|---|---|
| Bundled fonts (SIL Open Font License 1.1; Ultra is Apache-2.0) | `../mograph/fonts/OFL-*.txt`, `../mograph/fonts/LICENSE-Ultra.txt`, registry with sources in `../mograph/fonts/fonts.json` |
| SoundFont banks for the optional sampled-instrument engine (downloaded at run time, never committed) | `banks/README.md` |
| Model weights (LTX 2.x, Gemma 4, ACE-Step 1.5, MOSS-Image, Qwen3.5 (MOSS text encoder), Video Depth Anything Small) | not in this repository; audit with sources, conditions and the attribution block in `../_info/licences.md`, summary in the README "License" section |
| Custom node packs the GPU template needs (ComfyUI-LTXVideo, ComfyUI-Video-Depth-Anything, ComfyUI-VideoHelperSuite) | cloned by the user from their own repositories; see their licences there |

Gated downloads (`Lightricks/LTX-2.5` and the Alpha-Gen IC-LoRA) require accepting the licence on Hugging Face and an `HF_TOKEN`; see the README "Models" step.
