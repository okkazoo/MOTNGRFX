# Ingredients of a composition under one name (2026-10-29)

Rule: EVERYTHING one finished composition is made of is kept under its `<stem>_` name, plus the final video. Rows marked *full_archive* are the heavy extras; `MographArchive full_archive=False` (or `MOTNGRFX_ARCHIVE_FULL=0` on the box) leaves them out and keeps the light set. `<stem>_MANIFEST.md` starts with the size of the composition and the share that is full_archive extras. Per-shot compose layers: background / plate = `_sK_clip|textbg|plate.mp4`, subject = `_sK_clean.mp4` + `_sK_matte/`, type = `_type_beauty.mp4` + `_type_matte/` (PNG), composited = `_sK_comp.mp4`.

Goal: one search for the run stem (`motion-graphic_batch-24gb_v002_r01`) in Outputs finds EVERYTHING that composition is made of, together, and the mattes show as one stack.

## Where it ends up
`<version>/ingredients/<stem>/` (hardlinks only, never copies; `<stem>_MANIFEST.md` lists every entry: what, origin node, frames / size / fps):

| name | what |
|---|---|
| `<stem>_video.mp4` | the finished piece (hardlink of `out/<stem>_video_00001_.mp4`) |
| `<stem>_music.wav` | the music mix (box: from the assembled video's audio; old runs: cut from the final mp4) |
| `<stem>_still.png`, `<stem>_still_<k>.png` | first frame / one still per place |
| `<stem>_sK_clip.mp4`, `_sK_clean.mp4`, `_sK_textbg.mp4` | shot footage (single clip / clean + text-rendered background of three-layer shots) |
| `<stem>_sK_matte/<stem>_sK_matte_00001.png` | Alpha-Gen matte, PNG per frame, one folder per shot (stack members; PNG is never packed) |
| `<stem>_mattes_stacked.mp4` | all mattes stacked into one gray clip (hardlink of the `review/` preview from `matte_preview.py`) |
| `<stem>_sK_depth.mp4`, `_sK_flow.mp4` | depth / optical-flow previews |
| `<stem>_type_beauty|matte|depth|line|clay|flow.mp4` | the type passes compose read = the graphic-design animation alone; `_type_manifest.json` |
| `<stem>_sK_contour_<kind>.mp4` + `.npz` | contour tracing (matte_outline / depth_iso / luma_edges) the type words use as entry / exit / via ports |
| `<stem>_ltx_read_sK_<j>.png`, `_ltx_reads.json` | crops the VLM read in the LTX legibility check |
| `<stem>_cut_s1-s2_<kind>.mp4` | the 3D shot cut as rendered (A's tail then B's head; `<kind>` = depth_pivot, orbit_push, depth_dolly_down ...) *full_archive* |
| `<stem>_cut_s1-s2_matte/<stem>_cut_s1-s2_matte_00001.png` | the subject matte THROUGH that cut, PNG per frame (stack member) *full_archive* |
| `<stem>_cut_s1-s2_<kind>.json` | the cut's plan, resolved params, windows, which depth the engine used (given / estimated) |
| `<stem>_sK_depth.png` | depth of the first frame of shot K (what the cuts / parallax read), 8-bit gray PNG |
| `<stem>_sK_depth/<stem>_sK_depth_00001.png` | per-frame depth PNG sequence (stack member) *full_archive* |
| `<stem>_sK_comp.mp4` | the composited shot as compose made it (type woven in, graded), before assemble *full_archive* |
| `<stem>_sK_plate.mp4` | the clean plate of a clean-plate shot (u8 of the clip; the subject layer is `_sK_clean.mp4` + `_sK_matte/`) |
| `<stem>_sK_sources.json` | what shot K's compose ops read from OTHER shots (window_other / `$next` / `$prev`) and which files here hold it |
| `<stem>_type_matte/<stem>_type_matte_00001.png` | the type matte pass as PNG (the `_type_matte.mp4` is only the preview) *full_archive* |
| `<stem>_design_sheet.png` | the sheet image the vision model was shown for the design |
| `<stem>_places.json` | per place: id, prompt, angle-still source place + anchored edit prompt (`_still_<k>.png` = row k; an angle place's still IS the edit when the graph's `angles` switch was on) |
| `<stem>_piece.json`, `_index.json`, `_concept.json`, `_concept_reply.txt`, `_design_reply.txt`, `_design_check.txt`, `_design_final.txt`, prompts | design + model text (from `review/model_replies/` or the run record) |

Dashboard rule (vastai-api-manager `core/output_collapse.py`): a folder with >= 2 sub-folders that each hold a >= 16-frame numbered PNG sequence is ONE `.stack` tile; its subtree (mp4, json, wav, stills) is the stack's members. `python tools/app/ingredients.py check <folder>` says whether it will collapse (a one-shot run, or a rules run without mattes, will not).

## Box side (MographArchive)
Needs a ComfyUI restart on the box. Every file in `<prefix>_shots/` gets `<stem>_` in front (stem = filename_prefix basename without `_out`, `tools/fusion/ingredient_names.py`). The short names (`still.png`, `sK_clip.mp4`, `sK_matte/NNNNN.png`, `piece.json`, `index.json`) stay as hardlinks (`LEGACY_NAMES = True`) because `MographArchiveLoad` and `tools/fusion/generations.py` read them; the dashboard sync copies both names into `out/`, so flip the flag once those readers (plus `tools/app/matte_preview.py` and `run_batch.collect_files`, which look for `sK_matte` / `NNNNN.png`) take the prefixed names. The extras (`nodes_fusion/archive_ingredients.py`) are best effort and cost about the type-pass encodes (ffmpeg veryfast, 3 in parallel); `MOTNGRFX_ARCHIVE_EXTRAS=0` switches them off. The graph needs the new optional archive inputs `places` (first_set), `type` (type_final), `depth_1..6` (`build_graphs.py` wires them; regenerate the graph JSONs).

Whole-composition set (same day, second step): new optional archive inputs `full_archive` (bool, default ON), `composed` (MographCompose output 0), `cuts` (MographPivotCut output: only in the `--pivot-cut` graphs, `insert_pivot_cut` wires it), `sheet` (MographDesignPackage output 3); `build_graphs.py` wires `composed` + `sheet` in the design graphs and the redesign graphs (regenerate the graph JSONs). MographPivotCut puts a record of every rendered cut on A's clip (`clip['cut_renders']`, `pivot_cut.cut_record`: views of the new clips, no copy); the archive writes them, so nothing is re-rendered. Cost: the cut / composed mp4s are x264 veryfast encodes of frames already in memory (about one more type-pass round of encodes); the PNG folders are written at compression 3; the type matte PNGs are hardlinks of the type render's own files. Size is in `index.json` (`bytes`: total / heavy) and in the MANIFEST. Limits: a cut whose depth was not wired (`depth_k` absent) uses the engine's own CPU-estimated depth, which is not exposed, so only its source is recorded in the cut json; the archive's `clips` come from before the cut / compose nodes, so `_sK_clip.mp4` is the footage as generated and the cut's frames are in the `_cut_` files; a light (redesign) archive writes only the design sheet and the composed shots besides the json; the per-composition `_places.json` cannot say whether the `angles` switch was on, only which places ask for an edit. The contour traces (`_sK_contour_<kind>`) are the kinds the piece uses; with full_archive a piece with a `handoff` word that names no port gets matte_outline + luma_edges traced.

## Local side
```
python tools/app/ingredients.py pull --iid 58135115 --version COMPETITIONS/motion-graphic/22_batch-24gb/v002 [--run r01] [--state <bench>/state.json]
python tools/app/ingredients.py pull --no-box --version COMPETITIONS/motion-graphic/21_batch-3060/v005     # destroyed box: only what is local
```
The box address comes from the dashboard API (`/api/instances/<iid>`), the key is `~/.ssh/vastai_comfyui_rsa`, the box is only listed (`find`) and read (`tar -c`). Files the dashboard sync already put in `out/<stem>_out_shots/` (same size) are linked, not fetched; the rest (json, wav, whatever is not synced yet) comes with `tar | ssh` into `<version>/local/ingredients_pull/<stem>/`. `run_batch.finalize` runs it for every run after the model replies are saved (a failure only logs). Old archives (short names) are mapped to the prefixed names; what they never saved (type passes, place stills, previews, contours) is absent, and the wav is cut from the final mp4.

The pull needs no change for the new files: every `<stem>_`-prefixed file or folder in the archive dir is listed, fetched (or linked from the dashboard sync) and hardlinked, PNG folders stay folders (they count as extra stack members; the stack rule is unchanged). The MANIFEST describes the new names and starts with a size line (`ingredients.size_report`: total, by kind, full_archive share).

Not covered: the type passes / wav of a run whose ComfyUI temp folder is gone (not reachable after the fact), and any file a hardlink cannot reach (different volume: reported in the manifest, never copied).
