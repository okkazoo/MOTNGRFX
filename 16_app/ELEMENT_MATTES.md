# Per-element mattes (man / buoy) from LTX clips: research

Researched 2026-10-29. Research only; nothing installed or run. "UNVERIFIED" = not confirmed from a primary source or not testable without running it.

Goal: Alpha-Gen gives one soft matte for all foreground. We want one matte per named element, tracked across the clip, to split the Alpha-Gen alpha into layers over the Clean-Plate background.

## 0. Bottom line

- Use SAM 3.1 through ComfyUI core nodes (no custom pack needed). Weights are in an ungated Comfy-Org repo, 1.75 GB repo total, fp16.
- SAM 3 licence is not a blocker for this app: worldwide grant, no non-commercial clause, no EU/UK/KR/US exclusion. Only limits: trade controls, ITAR/military/nuclear/espionage/weapons end uses, no reverse engineering, ship the licence with redistributed weights.
- SAM 3 gives instance masks (labels + tracking). Alpha-Gen stays the alpha. Split alpha by instance with a normalised weight so the layers sum back to the original matte.
- Core node caveat: the core video template does per-frame detection, not tracking. For tracked, named instances build the graph by hand: SAM3 Detect (per label, frame 0, individual masks) -> Run SAM3 Video Track (initial_mask) -> SAM3 Track to Mask (object_indices).
- Fallback if SAM 3 is ever ruled out: SAM 2.1 (Apache-2.0) + Grounding DINO or OWLv2 (Apache-2.0) / Florence-2 (MIT). MatAnyone is non-commercial: do not use it in the product.

## 1. SAM 3

| Item | Finding | Source |
|---|---|---|
| SAM 3 release | Announced 2025-11-19 (SAM 3 + SAM 3D). Paper arXiv v1 2025-11-20, v2 revised 2026-03-28. Meta's own blog/README give no explicit SAM 3 date; 2025-11-19 is from press coverage and the licence date. | https://wandb.ai/byyoung3/ml-news/reports/Meta-Launches-SAM-3-and-SAM-3D--VmlldzoxNTEyMTQyNA , https://arxiv.org/abs/2511.16719 , https://github.com/facebookresearch/sam3 |
| SAM 3.1 ("Object Multiplex") | 2026-03-27. "Drop-in replacement"; tracks up to 16 objects per forward pass; 16 -> 32 FPS on one H100 at a medium object count; ~7x speedup at 128 objects vs SAM 3. | https://ai.meta.com/blog/segment-anything-model-3/ , https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md |
| Accuracy 3 vs 3.1 | Mixed on video benchmarks: SA-Co/VEval YT-Temporal cgF1 50.8 -> 52.9, MOSEv2 J&F 60.3 -> 62.3, LVVIS mAP 36.3 -> 34.3. 3.1 is not strictly better. | RELEASE_SAM3p1.md (above) |
| Meta repos (gated, request access + HF login) | `facebook/sam3`: `sam3.pt` 3.45 GB, `model.safetensors` 3.44 GB, plus tokenizer/config files (small). `facebook/sam3.1`: `sam3.1_multiplex.pt` 3.5 GB. | https://huggingface.co/facebook/sam3/tree/main , https://huggingface.co/facebook/sam3.1/tree/main |
| ComfyUI repack (NOT gated per page; licence tag "sam-license") | `Comfy-Org/sam3.1`, file `checkpoints/sam3.1_multiplex_fp16.safetensors`; repo total 1.75 GB. Go in `ComfyUI/models/checkpoints/`. | https://huggingface.co/Comfy-Org/sam3.1 , https://huggingface.co/Comfy-Org/sam3.1/tree/main , https://docs.comfy.org/tutorials/utility/video-segment-sam3 |
| Code requirements (Meta reference code) | Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6. Not needed if using ComfyUI core nodes. | https://github.com/facebookresearch/sam3 |
| Text prompts | Short noun phrases ("man", "buoy"), max 32 tokens in the Comfy docs. Comma-separated for several; `:N` caps instances per prompt ("eye:2, window panels:4"). | https://docs.comfy.org/tutorials/utility/video-segment-sam3 |
| Internal resolution | 1008x1008, input auto-resized. A 576x1024 clip is therefore resampled; output masks are resized back. Aspect handling UNVERIFIED. | https://github.com/comfyanonymous/ComfyUI/pull/13408 |

### Licence ("SAM License", Last Updated November 19, 2025)

Read from the LICENSE in the SAM 3 repo (identical 7.35 kB LICENSE file is in facebook/sam3, facebook/sam3.1 and Comfy-Org/sam3.1).

| Question | Answer | Quote / section |
|---|---|---|
| Commercial use? | Allowed. No non-commercial clause exists; grant is general. | s1(a): "non-exclusive, worldwide, non-transferable and royalty-free limited license ... to use, reproduce, distribute, copy, create derivative works of, and make modifications to the SAM Materials." |
| Redistribution? | Allowed, under the same agreement, with a copy of the agreement shipped. | s1(b)(i) |
| Derivative works | You own your derivatives/modifications; Meta keeps the SAM Materials. | s5(a) (per page summary) |
| Territory limits? | None. "Worldwide" grant. Contracting entity differs only: Meta Platforms Ireland Ltd for EEA/Switzerland, Meta Platforms, Inc. elsewhere. No EU/UK/KR/US exclusion. | s1(a) and "Meta" definition |
| Use limits (AUP-like) | Must comply with Trade Control Laws; licensee must not be a Trade Controls target; no ITAR activities; no end uses "related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons". | s1(b)(iii), s1(b)(v) |
| Reverse engineering | Prohibited ("reverse engineer, decompile or discover the underlying components"). | s1(b)(iv) |
| Research publication | Must acknowledge SAM Materials if you publish research results using them. | s1(b)(ii) |
| Termination / law | Meta can terminate for breach; California law. | s6, s7 (per page summary) |

Sources: https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE , https://huggingface.co/facebook/sam3/blob/main/LICENSE

Implications for the app: bundling the licence text beside any cached/redistributed weights is enough; downloading from `Comfy-Org/sam3.1` on rented boxes avoids needing an HF token (confirm at first download: UNVERIFIED that no gate appears on the file itself). Contest-entry use is not restricted by anything above (not legal advice).

### VRAM / speed at ~576x1024, 100-400 frames

No primary-source figure exists. Treat all of this as UNVERIFIED until benchmarked on a 24 GB box.

| Claim | Source | Quality |
|---|---|---|
| SAM 3: 30 ms/image with 100+ objects on H200. Video latency scales with object count; near real time for ~5 objects. | https://ai.meta.com/blog/segment-anything-model-3/ | Meta, primary |
| SAM 3.1: ~4 GB fp16 vs ~8 GB SAM 3 (model footprint) | https://www.mixpeek.com/model/facebook/sam3.1 , https://medium.com/@priyadarshinichavan/sam-3-1-is-here-and-it-changes-everything-about-real-time-object-tracking-7336ad0bd748 | third-party, unverified |
| Memory grows with frame count and object count; reduce resolution, objects, clip length | https://codersera.com/blog/how-to-run-sam-3-locally-2026/amp/ | third-party |
| Core node can try ~100 GB RAM if `detection_threshold` < ~0.4 and `max_objects` = 0; cap `max_objects` (PR #13767 lowers peak). Set `max_objects` to a small non-zero number. | https://github.com/comfyanonymous/ComfyUI/pull/13408 | ComfyUI PR discussion |
| Core track masks are stored bit-packed (1 bit per pixel per object per frame), so 400 frames x 2 objects at 1008 res is small. | https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_sam3.py | code |

Rough expectation (my estimate, not sourced): 2-4 objects, 400 frames at 576x1024 fits a 24 GB card; the input frame batch (400 x 576 x 1024 x 3 float) is ~2.7 GB in RAM/VRAM, so process in chunks if needed.

## 2. ComfyUI support

### 2a. Core (recommended)

ComfyUI core ships SAM 3.1 nodes. PR #13408 "feat: SAM (segment anything) 3.1 support (CORE-34)", author kijai, merged 2026-04-23. Files: `comfy/ldm/sam3/*`, `comfy_extras/nodes_sam3.py`, `comfy/text_encoders/sam3_clip.py`. https://github.com/comfyanonymous/ComfyUI/pull/13408

Whether the original SAM 3 weights (not 3.1) load in core: UNVERIFIED; the docs and templates only name `sam3.1_multiplex_fp16.safetensors`.

Nodes (from `comfy_extras/nodes_sam3.py`, https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_sam3.py):

| Node id (display) | Key inputs | Outputs |
|---|---|---|
| `SAM3_Detect` ("SAM3 Detect") | model, image (batch OK), conditioning (CLIPTextEncode text), bboxes, positive_coords/negative_coords (JSON), threshold 0.5, refine_iterations 2, `individual_masks` (bool: per-object masks instead of union) | `masks` (MASK), `bboxes` |
| `SAM3_VideoTrack` ("Run SAM3 Video Track") | images (frame batch), model, `initial_mask` (one mask per object, first frame), conditioning (optional, detects NEW objects during tracking), detection_threshold 0.5, `max_objects` (UI default 4, 0 = internal cap 64, initial masks count), detect_interval 1 | `track_data` (SAM3_TRACK_DATA) |
| `SAM3_TrackPreview` | track_data, images, opacity, fps | preview mp4 with object ids/scores |
| `SAM3_TrackToMask` ("SAM3 Track to Mask") | track_data, `object_indices` (e.g. "0,2"; empty = all) | `masks` [N,H,W], OR of the selected objects, bilinear-resized to original size (edges can be fractional) |

Loading: `CheckpointLoaderSimple` with `sam3.1_multiplex_fp16.safetensors`; MODEL goes to SAM3 nodes, CLIP goes to `CLIPTextEncode` for the label text. Templates: Workflow > Browse Templates > Utility > "SAM3: Video Segmentation" (`utility_video_segment_sam3`). https://docs.comfy.org/tutorials/utility/video-segment-sam3 , https://github.com/Comfy-Org/workflow_templates/blob/main/templates/utility_video_segment_sam3.json

Important finding: the stock video template uses only `SAM3_Detect` on the whole frame batch (no VideoTrack node), i.e. per-frame detection, which can flicker identity. For tracked instances use `SAM3_VideoTrack`.

Per-element wiring (my design; node behaviour UNVERIFIED until run):
1. Frame 0 (or the clearest frame): `SAM3_Detect` once per label, conditioning = that single label, `individual_masks` true, `max_objects`/count 1 ("man:1"). One run per label gives a known label -> mask mapping and avoids relying on output order.
2. Stack the per-label masks into one `initial_mask` batch in label order (index 0 = man, 1 = buoy).
3. `SAM3_VideoTrack` with images = full clip, `initial_mask` = stack, conditioning empty (so no unnamed extra objects appear), `max_objects` = number of labels.
4. One `SAM3_TrackToMask` per label with `object_indices` = that label's index.
Starting from a frame other than 0: not exposed in the core node signature (initial mask is "first frame"); trim/reverse the batch to start at the best frame. UNVERIFIED.

### 2b. Community pack: PozzettiAndrea/ComfyUI-SAM3

| Item | Finding | Source |
|---|---|---|
| Repo | https://github.com/PozzettiAndrea/ComfyUI-SAM3 (330 commits; install via Manager, runs `install.py`, uses experimental `comfy-env` + downloads pixi) | repo page |
| Licence | Wrapper GPL-3.0; wrapped model under Meta SAM License | https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-SAM3/main/LICENSE |
| Nodes | Load SAM3 Model, SAM3 Text/Point Segmentation, Create/Collect/Combine Point and Box nodes, `SAM3VideoSegmentation`, `SAM3Propagate`, `SAM3VideoOutput` | https://www.runcomfy.com/comfyui-nodes/ComfyUI-SAM3 |
| Video prompt | `prompt_mode` text/point/box (exclusive). Text: comma-separated, each phrase gets its own object id ("person, dog, car"). `frame_idx` selects the prompt frame, `score_threshold` default 0.3. | https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-SAM3/main/nodes/sam3_video_nodes.py |
| Propagate | start_frame, end_frame (-1 = last), direction forward/backward/both, `offload_model` | https://www.runcomfy.com/comfyui-nodes/ComfyUI-SAM3/sam3-propagate |
| Output | `SAM3VideoOutput`: `obj_id` (-1 = union, else selects one object), outputs `masks` (MASK [frames,h,w] float32), `frames`, `visualization` | nodes file above |
| Model | "automatic model downloads" from Meta's SAM 3 (gated on Meta's side; token needs UNVERIFIED) | runcomfy page |
| Pros / cons vs core | Pro: text prompt per label with ids, direction both, prompt on any frame. Con: GPL-3.0 wrapper, heavier install (comfy-env/pixi), non-core, 3.1 support not stated. | |

Choice: core nodes for the app (no extra dependency, ungated weights, licence cleaner). Keep Pozzetti as a fallback if core `SAM3_VideoTrack` initial-mask handling proves awkward.

### 2c. Community LTX clean-plate workflow using SAM3

- "LTX Cleanplate for VFX" by Doug Hogan, posted ~4 months before 2026-10 (no exact date): https://comfy.org/workflows/8f2cf0df5da6-8f2cf0df5da6/ (JSON: https://comfy.org/workflows/download/8f2cf0df5da6.json). Uses SAM3 (`LoadSAM3Model`, `SAM3VideoSegmentation`, `SAM3Propagate`, `SAM3VideoOutput` = the Pozzetti pack), `LTXVDilateVideoMask`, `GrowMaskWithBlur`, `LTXVInpaintPreprocess`, and the LTX **2.3 In-Outpainting** IC-LoRA (mask-conditioned, green fill). It is a 2.3 workflow, not 2.5, and not the Clean-Plate LoRA.
- Tips from it: dilate and feather the mask for motion blur; re-seed SAM3 on a clearer frame if the mask drifts; prompt background only.
- No SAM3 workflow for the LTX **2.5 Clean-Plate LoRA** was found. This is expected: the 2.5 Clean-Plate LoRA "does not take a user-supplied region mask" (model card). https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate
- Clean-Plate LoRA facts: gated, LTX-2.x Community License, `ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors` (0.33 GB per https://ltxworkflow.com/models/ltx25-ic-lora-clean-plate), trained at 1024x576 / 576x1024, 49 frames, 25 fps; dims divisible by 32, frames 8k+1; weak when the subject fills most of the frame (card recommends mask inpainting then). 24 GB is "tight" (third-party).

## 3. Alternatives (if SAM 3 is excluded)

| Component | Licence | Role | ComfyUI node | Notes / source |
|---|---|---|---|---|
| SAM 2.1 | Apache-2.0 (code + checkpoints); released 2024-09-29/30; tiny 38.9M .. large 224.4M params; large 39.5 FPS on A100 | Video mask tracking from point/box/mask prompts; multi-object, objects can be added mid-clip | `kijai/ComfyUI-segment-anything-2` (Apache-2.0; weights `Kijai/sam2-safetensors` into `models/sam2`; node names UNVERIFIED) | https://github.com/facebookresearch/sam2 , https://github.com/kijai/ComfyUI-segment-anything-2 |
| Grounding DINO | Apache-2.0 (`IDEA-Research/grounding-dino-base`, 0.2B) | Text -> boxes (e.g. "a man. a buoy.") | `storyicon/comfyui_segment_anything` (GroundingDino + SAM, per search result; licence UNVERIFIED) | https://huggingface.co/IDEA-Research/grounding-dino-base |
| OWLv2 | Apache-2.0 (`google/owlv2-base-patch16-ensemble`, 0.2B) | Text -> boxes | node pack UNVERIFIED | https://huggingface.co/google/owlv2-base-patch16-ensemble |
| Florence-2-large | MIT (0.8B) | Caption-to-phrase grounding / detection -> boxes. Zero-shot referring segmentation weak (RefCOCO RES 35.8 mIoU zero-shot). | `rdancer/ComfyUI_Florence2SAM2` (Florence-2 + SAM 2; licence UNVERIFIED) | https://huggingface.co/microsoft/Florence-2-large , https://github.com/rdancer/ComfyUI_Florence2SAM2 |
| MatAnyone / MatAnyone 2 | **NTU S-Lab License 1.0: non-commercial; commercial use needs permission from the authors.** | Video matting from a first-frame mask; MatAnyone 2 is the newer version (CVPR 2026 highlight), loads `PeiqingYang/MatAnyone2` | `FuouM/ComfyUI-MatAnyone` (nodes MatAnyone, MatAnyone2, Solid Color Batched; repo licence UNVERIFIED) | https://raw.githubusercontent.com/pq-yang/MatAnyone/main/LICENSE , https://github.com/pq-yang/MatAnyone2 , https://www.runcomfy.com/comfyui-nodes/ComfyUI-MatAnyone |

Verdict: the Apache/MIT path (Grounding DINO or OWLv2 boxes -> SAM 2.1 video tracking per object) is a sound fallback, but needs two models and a hand-made glue step (boxes -> prompts). SAM 3 replaces both with one text-prompted model. MatAnyone is human-focused and non-commercial, and we already have a soft matte from Alpha-Gen, so it is not needed; skip it.

## 4. Recommended pipeline

Inputs per clip: RGB frames I, Alpha-Gen soft matte A (union of all foreground), Clean-Plate background B (from the Clean-Plate LoRA), design labels L = {man, buoy}.

1. Labels -> frame-0 masks: `SAM3_Detect` per label (section 2a). Check each returns a mask whose area overlaps A (reject detections with <50% of the mask inside dilate(A)). If a label is empty, lower threshold to ~0.3, else fall back to box/point prompt from the design model.
2. Track: `SAM3_VideoTrack` with the stacked initial masks; `SAM3_TrackToMask` per label -> instance masks M_i (binary-ish, 576x1024 after resize).
3. Make instance masks consistent with A: `M_i' = M_i * dilate(A, ~3 px)`; fill holes; soften with a small blur (2-4 px) to get weights `w_i`.
4. Split the soft alpha by normalised weight (my formula; sums back to A so recompositing equals the original matte):
   `a_i = A * (w_i + eps*[i is nearest]) / (sum_j w_j + eps)`
   where pixels with A > 0 but all w_j = 0 (fringe, hair, motion blur outside the SAM mask) go to the nearest instance by distance transform. Leftover foreground not claimed by any label becomes an extra "other" layer.
5. Per-layer colour: with the clean plate, un-premultiply the colour: `F = (I - (1 - A) * B) / max(A, eps)` clipped 0..1, then layer_i RGBA = (F, a_i). Without B use I directly (halo risk at soft edges).
6. Stack layers over B in z-order given by the design model; move each layer independently.

Failure cases and mitigations (SAM 3 behaviour for the first three is general knowledge of trackers, UNVERIFIED for this setup; test on a clip):

| Case | What happens | Mitigation |
|---|---|---|
| Occlusion (man passes in front of buoy) | Back object's mask disappears or ID swaps on reappearance; and the occluded pixels do not exist in the video. | Run tracking forward and backward and merge by confidence; re-detect with the label every N frames (`detect_interval`, or Pozzetti `direction=both`); for the back layer, inpaint or accept a hole (animate only within allowed motion). |
| Touching objects (man holding buoy) | Boundary between masks is a guess; alpha split becomes arbitrary along the contact line. | Weighted split in step 4 keeps the sum correct; add negative points; check seam visually; consider merging into one layer if contact is permanent. |
| Thin parts (rope, antenna, hair) | SAM masks are blobby, miss thin structure at 1008 res; thin parts get lost or assigned to the wrong instance. | A already contains the thin structure; dilate the instance mask 3-6 px and use nearest-instance assignment so thin parts follow the nearest object. |
| Semi-transparent / motion blur | SAM mask is binary; the blur falls outside it. | Step 3/4 handle via A. |
| Missing or extra instances | Core `max_objects`/count semantics: default one instance per text prompt ("apron" returns one unless `:N`). Low threshold + `max_objects` 0 may spike RAM. | Specify counts per label; keep `max_objects` small. https://github.com/comfyanonymous/ComfyUI/pull/13408 |
| Clean plate wrong where subject fills frame | Clean-Plate LoRA falls back poorly. | Per model card, use mask inpainting for those clips. |
| Resolution/frames | Clean-Plate wants dims %32 and 8k+1 frames; SAM runs at 1008 internally. | 576x1024 fits both; pad frame count to 8k+1. |

## 5. Open items (UNVERIFIED, test first)

1. Core `SAM3_VideoTrack` with `initial_mask` stacked from several `SAM3_Detect` outputs: index order and behaviour with 2-4 masks.
2. Peak VRAM/RAM and seconds for 576x1024 x 100/250/400 frames, 2-4 objects, on a 24 GB card (A5000/4090/3090), core nodes, fp16.
3. Whether original SAM 3 (not 3.1) weights load in core.
4. Whether `Comfy-Org/sam3.1` downloads without an HF token on a fresh rented box.
5. Aspect handling at 1008x1008 for portrait 576x1024 input (letterbox vs stretch).
6. LTX-2.5 Alpha-Gen IC-LoRA was not researched here (outside the question); no 2.5 Alpha-Gen repo appeared in the Lightricks listing page I fetched (https://huggingface.co/Lightricks), so confirm the repo name before wiring.
