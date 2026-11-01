# Reply schema: wrong types and values made impossible at generation time

Built 2026-10-29. Code: `tools/fusion/reply_schema.py`, wiring `tools/app/llm_gguf.py` (`Runner.chat(schema=)`, `response_format_for`) + `nodes_fusion/llm_gguf.py`
(input `schema`: none | design | concept), speed probe `tools/app/schema_probe.py`, tests `tests/test_reply_schema.py` (19). Status: **built, OFF by default, NOT run on a model**
(the box was rendering; llama-cpp-python is not installed on this PC). Measurements below are of the saved replies only.

## Why

The open design models (Qwen3.8-27B UD-Q5_K_M, Gemma 4 31B QAT via llama.cpp) run with `response_format {"type":"json_object"}` (the generic JSON grammar) only in the concept / short
repair calls, and with no grammar at all in the design call. The generic grammar guarantees syntax, nothing else: a model can still write a string where a list belongs, an io style
that does not exist (`gradient_fade`, `fade`), a font that is not on the liked list (`Big ShouldersDisplay[wght].ttf`), `matte: "behind"`, a `holds` entry as an object, or
a bracket slip in the middle of a 5 k-token reply. `design_salvage.py` repairs these after the fact (and `recipe_eval.pass_names` mends a string `type_passes`), but every repair is a
field the model did not really design, and a few reached the engine (box v008 r01: `type_passes: "all"` -> `set("all")` -> `unknown pass 'a'`).
With a JSON-schema grammar the sampler can only pick tokens that keep the reply inside the schema: the wrong type or value cannot be produced.

## What is constrained

Schemas are generated from the code's own constants (nothing is typed twice; `tests/test_reply_schema.py::test_enums_come_from_the_code`):

| field | source of truth | constraint |
|---|---|---|
| `type.words[].module` | `qwen_piece.WORD_MODULES` | enum (24) |
| `form_in`, `form_out` | `qwen_piece.IO_STYLES` | enum (5) |
| `entry` | `openings.ORDER` | enum (left right down in out up), optional |
| `font`, `system.roles.*`, `system.axis.font` | `qwen_piece.liked_fonts()` (= `mograph/fonts/liked.json`, 42) | enum of the exact file names |
| `matte` | `qwen_piece.MATTES` | enum (`behind` / `none` cannot be written), optional |
| `render` | `owner.RENDERS` | enum, optional; `ltx_at` number (0.2..0.8); `material` string |
| `type.camera` | `qwen_piece.CAMERAS` | enum |
| `type.transitions` | `qwen_piece.TRANSITIONS` | `"auto"` or a list of exactly (words - 1) entries, each a transition name or `null` |
| `type.palette.bg/fg`, `accents` | rgb | arrays of 3 INTEGERS 0..255 (floats cannot be written), accents 3-4 entries |
| `compose.glow_color` | rgb | array of 3 numbers |
| `compose.technique` | techniques of the prompt (`"technique": "<EXACTLY one of: ...>"`) | enum |
| `compose.holds` | | list (max 6) of `[time, freeze]` number pairs (no objects) |
| `compose.cuts[]` | `design_options.CUT_KINDS` (minus `depth_dolly:*` while `DOLLY_ON` is off), `EXITS`, shot ids of the prompt's SHOTS list | `after` enum (all shots but the last), `cut` enum, `exit` enum, `freeze_s`, `word` integer |
| `compose.default/shots.*.ops[]` | `compose_blocks.ALLOWED` | `op` enum, `out` string, `args` a free JSON object |
| `type.words` | words of the prompt (`Words, in order, each shown once: A / B / C`) | exactly N entries in order, each `text` fixed to its word (`lock_text`) |
| numbers (`hit`, `hold_s`, `form_s`, `exit_s`, `yaw`, `wf`, `hf`, `energy`, ... `plate_gain`) | `qwen_piece.RANGES` | `number` with the clamp range written in (see "Not enforced") |
| `why`, `idea_why`, `concept`, `treatment_notes`, `material`, `window_source` | | plain strings |
| `type_passes` | `spec.PASSES` | not a design-reply key at all (closed objects); `type_passes_schema()` = list of known pass names, for any place a model writes a piece |
| concept: `look`, `shots` | `plan_rules.LOOK_NAMES`, `SHOT_MODES` | enum |
| concept: `bpm` | `concept.BPM_RANGE` | integer 60..160 |
| concept: `words[]` | `concept._CLEAN`, `MAX_LINE_CHARS` | string matching `^[A-Z0-9 '!?.,&-]{1,26}$` (upper case, no slash lists) |
| concept: `words`, `shot_plan`, `owners` | `EXACTLY n entries` of the prompt | exactly n entries; `shot_plan[k>0]` needs the place flag the prompt asks for (`same_place`, older prompts `continues`) as a boolean |
| concept: `owners[].render` | `owner.RENDERS` | enum |

Every object with declared keys is **closed**: a key the schema does not list cannot be written (`why_owner`, `why_material`, `type_passes`, a second `compose` inside `type` ...). The
only open objects are `args` of a compose op, the axis map of a word and the role map of `system`. Required keys are generated first, in the schema's order, then the optional ones;
the `why` fields and `system` are REQUIRED by default (`require_why`, `require_system`), so a model reasons before it chooses and `system` comes before the words it should organise
(the prompt calls `system` optional; with an optional key the grammar would put it last, after the words). `design_schema(require_why=False, require_system=False)` is the lenient form.

Counts come from the prompt text itself (`design_context`, `concept_context`), so the graph needs no extra wiring: the node reads its own prompt.

## Not enforced (the converter cannot, so the code still clamps)

llama.cpp's json-schema-to-grammar converter (JamePeng llama-cpp-python 0.4.5, `llama_cpp/llama_grammar.py`, read 2026-10-29; `response_format {"type":"json_object","schema":...}` goes through
`_grammar_for_response_format` in both the plain and the MTMD (image) chat handlers) enforces `integer` minimum / maximum but **ignores `minimum` / `maximum` of a `number`**. So float ranges
(`hold_s` 0.5..6, `yaw`, `glow_color` 0..1, `plate_gain`, ...) are still written by the model and clamped by the code; they are in the schema for the measurement and the docs. Also not
expressible: transitions used twice, `lines` joined spelling the word, a `hit` on a real beat, the timing arithmetic (all stay in `qwen_piece` / `design_salvage`). The schemas use only
`type properties required additionalProperties items minItems maxItems enum const anyOf minimum maximum pattern` (`reply_schema.unsupported_keywords` = [], tested), no `$ref`, `allOf`, `oneOf`, `format`.

## Measured: how inconsistent are the saved replies (no model run)

All 136 index rows of `20_open-model-design/replies.jsonl` were scanned: 75 replies were scoreable (41 design, 34 concept) (`python tools/fusion/reply_schema.py measure`; raw: `16_app/reply_schema_check/measure.json`, table `measure.md`).
Used: the 36 app-run rows (67 design / concept reply texts, both GGUF models and the dropped design_llm_8b / Gemma 12B arms) and 8 hosted-API cells that used the APP prompts
(`par_app`, `par_concept`: Qwen3.8-27B and Gemma 4 31B). The 84 `par_brief` cells (whole-piece JSON of the par harness) have a different schema and are not scored; 8 `par_direction` cells are prose.
Columns: **JSON syntax ok** = parses as written (fence stripped); **types + values ok** = every written field has the right type / enum value / integer range / pattern (the reply is parsed with the
code's own mender when its syntax is broken); **fully valid as written** = also valid syntax, no missing / undeclared keys, right list lengths.
Caveats: app-run prompts are *reconstructed* with today's code (not what the run saw), so counts / texts are not checked for them (free counts) and "keys / lengths" of old runs reflect OLDER prompts
(design_llm_8b and Gemma 12B concepts predate `idea` / `shot_plan` / `owners`: those columns say nothing about the model). Small n for the two models in use today.

| model | reply | n | JSON syntax ok | types + values ok | fully valid as written | top-5 wrong type / value (replies) | top-3 keys / lengths (replies) |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (app, GGUF) | design | 10 | 100 % | 80 % | 80 % | `compose.cuts[].cut` enum x2 (`depth_dolly:right`), `type.words[].font` enum x1, `system.roles.B` enum x1 (font name misspelt) | none |
| Qwen3.8-27B (app, GGUF) | concept | 11 | 82 % | 82 % | 82 % | none | none (2 replies not parseable even after mending) |
| Gemma 4 31B QAT (app, GGUF) | design | 8 | 75 % | 25 % | 25 % | `compose.cuts[].cut` enum x4, `type.words[].form_out` enum x3 (`fade`), `type.words[].font` enum x1, `type.words[].matte` enum x1, `system.roles.B` enum x1 | `type.words[].why_material` extra x1, `system.axis.progression` extra x1, `type.words[].why_owner` extra x1 |
| Gemma 4 31B QAT (app, GGUF) | concept | 8 | 62 % | 100 % | 62 % | none | one reply missing most keys (wrong object extracted) |
| design_llm_8b (app, dropped arm) | design | 15 | 100 % | 0 % | 0 % | `type.words[].matte` enum x15 (`behind`), `form_in` enum x8 / `form_out` x7 (`gradient_fade`), `compose.holds[]` type x1 (objects) | `type.words[].lines` length x3 |
| design_llm_8b (app, dropped arm) | concept | 7 | 100 % | 86 % | 0 % | `words[]` pattern x1 (one string `A / B / C`) | `shot_plan`, `idea`, `owners` missing x7 (older prompt) |
| Gemma 4 12B (app, dropped arm) | design | 4 | 75 % | 50 % | 50 % | `type.words[].font` enum x2 | `type.compose` extra x1, `compose` missing x1 |
| Gemma 4 12B (app, dropped arm) | concept | 4 | 100 % | 100 % | 0 % | none | `shot_plan`, `idea` missing x4 (older prompt) |

Read: for the two models in use, ONE reply in five (Qwen3.8-27B design) to THREE in four (Gemma 4 31B QAT design) carries at least one wrong value that salvage had to repair; the recurring
ones are made-up enum values (cut kinds, io styles, fonts spelt wrongly) and extra keys, not wrong container types. Syntax breaks (a `]` for a `}` in the middle, an unclosed `type`) hit 25 % (design) and 38 % (concept) of the Gemma 31B QAT
replies as written and are absorbed today by `qwen_piece._mend_brackets`. The generic grammar already removes those; the schema grammar removes the rest. Soft (not
enforceable) range violations: `compose.glow_color` outside 0..1 in 10 of 15 design_llm_8b and 1 of 2 Gemma API replies. `type_passes`: the string occurs in none of the 75
archived design / concept replies (the v008 r01 reply that crashed was not archived), so that crash is not measured here; a design reply can no longer contain the key.

## Speed (NOT measured)

Grammar-constrained sampling costs time per token and a grammar-compile once per call (it lands in `prompt_s`). Sizes (this PC, converter of llama-cpp-python 0.4.5): design schema 17.8 kB JSON ->
GBNF 40 kB / 305 rules at 3 words, 65 kB / 473 rules at 6 words (every word entry is its own tuple slot because `lock_text` fixes each `text`); without the lock (`schema_nolock` in the probe) 23.6 kB / 192 rules;
concept 4.7 kB / 55 rules. There is no number for tok/s: no GPU was free. Run the probe when the box is idle (it loads the model itself; do not run it while ComfyUI is rendering):

```
python tools/app/schema_probe.py --arm qwen38_27b_q5km --kind design \
    --prompt-file 20_open-model-design/v003/local/app0/new_qwen_s1/prompt.txt \
    --seeds 1,2,3 --max-tokens 6000 --out 16_app/reply_schema_check/probe_qwen38_design
python tools/app/schema_probe.py --arm gemma31b_qat --kind design --prompt-file 20_open-model-design/v003/local/app0/new_qwen_s1/prompt.txt \
    --seeds 1,2,3 --max-tokens 6000 --out 16_app/reply_schema_check/probe_gemma31_design
python tools/app/schema_probe.py --arm qwen38_27b_q5km --kind concept --prompt-file 20_open-model-design/v003/local/app0/concept_qwen_s1/prompt.txt --seeds 1,2,3 --out 16_app/reply_schema_check/probe_qwen38_concept
```

(`--dry` builds the schema and the GBNF and prints the sizes without a model.) Per arm (`free`, `json`, `schema`, `schema_nolock`) x seed it prints gen tok/s, tokens, prompt_s (the schema arm
pays the grammar compile there), total_s, finish reason, syntax / schema validity and the violating fields, then a markdown table; `probe.json` + each reply text are written to `--out`. Text-only prompt
(the generation speed does not depend on the image; add `--image sheet.png --mmproj auto` to send the sheet). Put the table into BUILT.md "Speeds" when it exists. Decision rule: turn the schema on for a
call when valid % rises and tok/s stays within ~20 % of `free` (or total_s does not grow beyond what the saved salvage / repair calls cost today: a repair call is 85-104 s on the 3060).

## Hand-off (graph builder; files owned by the other worker: `tools/app/build_graphs.py`, `tools/app/llm_arms.py`, `app/graphs/*`, `design_step.py`)

Nothing is switched on. To turn it on per call (the node reads the counts from its own prompt, no other wiring):

1. `MographGGUFVL` has a new optional input `schema` (`none` default | `design` | `concept`). It implies the JSON grammar, so `thinking` is ignored with it (as with `json_mode`).
2. Per call: the MAIN design call (the one with the sheet image): `schema: "design"`. The concept call: `schema: "concept"` (replaces `json_mode`; `json_mode` may stay). The LONG design repair (image-less, carries the
   whole prompt + the rejected reply): `schema: "design"`. The SHORT repair (`design_step.short_repair_prompt`; its reply is a `{path: value}` map): NO schema, `json_mode` as now. A design arm with `thinking: True`
   must keep `schema: "none"` on the design call.
3. In `llm_arms.convert` (the place that writes `json_mode` for image-less nodes, ~line 193) add an arm flag `reply_schema` (default False until the probe says so) that writes the `schema` input for those roles.
   Role detection by node title is already there for `json_mode`.
4. Measure: `design_step` already reports `fields_from_model / fields_fixed / fields_from_rules`; the schema should push `fields_fixed` towards 0 for enum / type fixes. Compare a schema-on batch to the last schema-off batch on the same arm.
5. DRIFT: the schemas are closed, so a key added to `qwen_piece.schema_text`, `design_options.SCHEMA_WORD / SCHEMA_CUTS`, `concept.concept_prompt` without `tools/fusion/reply_schema.py` becomes impossible for the model
   to write. `tests/test_reply_schema.py::test_every_key_the_prompts_ask_for_is_in_the_schema` fails when that happens (it already caught the `entry` word key and the `same_place` shot flag added during this work; both are
   in the schema now). Lists the schema reads live: `WORD_MODULES`, `IO_STYLES`, `MATTES`, `CAMERAS`, `TRANSITIONS`, `TECHNIQUES`, `RANGES`, `openings.ORDER`, `owner.RENDERS`, `design_options.CUT_KINDS / EXITS / DOLLY_ON`, `compose_blocks.ALLOWED`,
   liked fonts, `concept.LOOKS / MODES / BPM_RANGE`.
6. Prompt text: `qwen_piece.schema_text` calls `system` optional; the schema requires it (set `require_system=False` in `design_schema` if the guide should keep it optional). The prompt's "EXACTLY one of" enum lists and the
   schema are the same constants, so they cannot disagree.
7. Open decision for the concept: `shots` allows `single` although the prompt fixes the number of lockups for the suggested mode; a model answering `single` then still writes n lockups (as today; `PR.fit_lockups` merges).


## WIRED (2026-10-29): arm flag `reply_schema`, OFF by default; A/B graphs `..._schema`
- `tools/app/llm_arms.py`: arm key `reply_schema` (default absent = false) or `convert(..., reply_schema=True)` / CLI `--reply-schema` write the `schema` input of the converted `MographGGUFVL` nodes by ROLE (read from the node that supplies the prompt): concept call (`concept_qwen`) -> `concept`; main design call (`design_qwen`, with the sheet) -> `design`; design repair (`design_qwen2`) -> `design_repair`; the concept repair (`concept_qwen2`) gets none (json_mode as before; `PROMPT_ROLES` has the one-line change if you want `concept` there too). A `thinking` arm keeps `none` on the design call (the grammar forbids the reasoning). `json_mode` stays where it was.
- `design_repair` is a new schema NAME (`reply_schema.NAMES`): ONE graph node serves both repair kinds and cannot know at build time which prompt `MographDesignParse` will hand it (`design_step.parse`: long repair = the whole design prompt + the rejected reply, reply = a whole design object; short repair = the invalid fields only, reply = a `{path: value}` map). `schema_for("design_repair", prompt)` returns the design schema when the prompt contains `=== YOUR PREVIOUS REPLY (rejected) ===` and the design prompt's words, else None (then `json_mode` alone, as today). So the long repair gets the grammar and the short one never does, without a second node.
- A/B graphs (written next to the plain ones; the plain graphs are byte-identical to before): `app/graphs/fusion_design_qwen_auto_layers_pivot_qwen38_27b_q5km_schema.json` and `..._pivot_gemma31b_qat_schema.json` (+ `.app.json`, title ends "reply schema grammar", sidecar key `reply_schema`). Rebuild: `python tools/app/llm_arms.py --reply-schema --base fusion_design_qwen_auto_layers_pivot --arm qwen38_27b_q5km --arm gemma31b_qat`; a default `llm_arms.py` run and every `--check` also rewrite / verify an existing twin.
- Optional concept keys: the concept `shot_plan` schema now also lists (optional, never required) `same_subject_as`, `framing`, `plate`, `type_matte`, `subject_move`, `plate_pan`, `plate_subject` (`reply_schema.plate_angle_keys`, a separate block), so a concept prompt that asks for them can be generated under the grammar; the prompt does not ask yet. The drift guard (`test_every_key_the_prompts_ask_for_is_in_the_schema`) stays green.
- Tests: `tests/test_reply_schema_flag.py`. NOT run on a model.
