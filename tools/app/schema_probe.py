#!/usr/bin/env python
"""schema_probe: does the reply-schema grammar cost speed, and does it make the reply valid? ONE loaded GGUF, the same design (or concept) prompt, three arms x N seeds:

    free    no grammar (what the design call does today)
    json    llama.cpp's generic JSON grammar (response_format json_object; what the concept / repair calls use today)
    schema  the reply schema grammar (response_format json_object + schema; tools/fusion/reply_schema.py, 16_app/REPLY_SCHEMA.md)
    schema_nolock  (design only) the same without locking each word entry's `text` (one shared word schema instead of one per word: a 2.5x smaller grammar at 6 words)

per run: prompt_s (time to first token; the schema arm also pays the grammar compile here), gen_tokens, gen_tok_s, total_s, finish reason, VRAM / RSS peaks, then
the reply is checked: JSON syntax as written, parsed (the code's own extractor), schema violations by kind (type / enum / const / pattern / int_range = a wrong type or value;
missing / extra / length = wrong keys or list sizes). Prints a markdown summary and writes <out>/probe.json + one reply file per run. NO API, no ComfyUI: a fresh process loads
the model with the arm's settings (tools/app/llm_arms.py) and frees it at the end. Run it on the box ONLY when ComfyUI is idle (the model and ComfyUI's models compete for VRAM).

    python tools/app/schema_probe.py --arm qwen38_27b_q5km --kind design \\
        --prompt-file 20_open-model-design/v003/local/app0/new_qwen_s1/prompt.txt --seeds 1,2,3 --out 16_app/reply_schema_check/probe_qwen
    python tools/app/schema_probe.py --arm gemma31b_qat --kind design --prompt-file ... --seeds 1,2,3 --out 16_app/reply_schema_check/probe_gemma
    python tools/app/schema_probe.py --arm qwen38_27b_q5km --kind concept --prompt-file 20_open-model-design/v003/local/app0/concept_qwen_s1/prompt.txt --seeds 1,2,3 --out ...
    python tools/app/schema_probe.py --dry --kind design --prompt-file ...       # no model: builds the schema, the GBNF (needs llama_cpp) and prints their sizes

--model <gguf path relative to --root> overrides --arm (then give --ctx etc. by hand). The prompt is text only (the speed of the generation does not depend on the image); add --image
sheet.png to send the combined sheet the way the graph does (the model needs its projector: --mmproj auto).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.app import llm_gguf as LG        # noqa: E402
from tools.fusion import reply_schema as RS  # noqa: E402

ARMS = ("free", "json", "schema", "schema_nolock")


def check_reply(text: str, kind: str, prompt: str) -> dict:
    """syntax as written, parsed by the code's extractor, schema violations (counts by kind and the field list). The schema is the one the schema arm used (counts from the prompt)."""
    from tools.fusion import qwen_piece as QP
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        syntax = isinstance(json.loads(t), dict)
    except Exception:       # noqa: BLE001
        syntax = False
    obj = QP.extract_json(text or "")
    out: dict = dict(syntax_ok=syntax, parsed=obj is not None, valid=False, value_violations=None, struct_violations=None, fields=[])
    if obj is None:
        return out
    sch = RS.schema_for(kind, prompt)
    vs = RS.violations(obj, sch)
    val = [v for v in vs if v["kind"] in RS.VALUE]
    st = [v for v in vs if v["kind"] in RS.STRUCT]
    out.update(valid=syntax and not val and not st, value_violations=len(val), struct_violations=len(st), fields=[f'{v["path"]} [{v["kind"]}]' for v in val + st][:20])
    return out


def summarize(runs: list[dict]) -> str:
    rows = ["| arm | runs | gen tok/s (mean) | gen tokens (mean) | prompt_s (mean) | total_s (mean) | JSON syntax ok | schema-valid | value violations (mean) |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = [x for x in runs if x["arm"] == arm]
        if not r:
            continue
        m = lambda k: sum(float(x.get(k) or 0) for x in r) / len(r)          # noqa: E731
        vv = [x["check"]["value_violations"] for x in r if x["check"]["value_violations"] is not None]
        rows.append(f"| {arm} | {len(r)} | {m('gen_tok_s'):.1f} | {m('gen_tokens'):.0f} | {m('prompt_s'):.1f} | {m('total_s'):.1f} | {sum(x['check']['syntax_ok'] for x in r)}/{len(r)} | "
                    f"{sum(x['check']['valid'] for x in r)}/{len(r)} | {(sum(vv) / len(vv)) if vv else float('nan'):.1f} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=("design", "concept"), default="design")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--arm", help="a key of tools/app/llm_arms.ARMS (model file, ctx, sampler settings)")
    ap.add_argument("--model", help="GGUF relative to --root (instead of --arm)")
    ap.add_argument("--root", default=LG.DEFAULT_ROOT)
    ap.add_argument("--mmproj", default="none", help="none (text only, default) | auto | path")
    ap.add_argument("--image")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of free,json,schema")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="16_app/reply_schema_check/probe")
    ap.add_argument("--dry", action="store_true", help="no model: build the schema + GBNF and print sizes")
    a = ap.parse_args(argv)
    prompt = Path(a.prompt_file).read_text(encoding="utf-8")
    sch = RS.schema_for(a.kind, prompt)
    g = RS.to_gbnf(sch)
    arms_wanted = [x for x in a.arms.split(",") if x in ARMS and not (x == "schema_nolock" and a.kind != "design")]
    ctx_info = RS.design_context(prompt) if a.kind == "design" else RS.concept_context(prompt)
    print(f"schema {a.kind}: {len(json.dumps(sch))} chars, context from the prompt: {json.dumps(ctx_info)[:300]}; unsupported keywords: {RS.unsupported_keywords(sch) or 'none'}; "
          + ("GBNF not available (llama_cpp not importable)" if g is None else f"GBNF {len(g)} chars / {g.count(chr(10)) + 1} rules"))
    if a.dry:
        return 0
    arm: dict = {}
    if a.arm:
        from tools.app import llm_arms as LA
        arm = LA.ARMS[a.arm]
        model = Path(a.root) / arm["dir"] / arm["file"]
    elif a.model:
        model = Path(a.root) / a.model
    else:
        ap.error("--arm or --model")
    mm = None if a.mmproj == "none" else (LG.mmproj_for(model) if a.mmproj == "auto" else a.mmproj)
    img = None
    if a.image and mm:
        import numpy as np
        from PIL import Image
        img = LG.png_b64(np.asarray(Image.open(a.image).convert("RGB")))
    n_ctx = a.ctx or int(arm.get("n_ctx") or 24576)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smp = LG.Sampler()
    smp.start()
    r = LG.Runner()
    runs: list[dict] = []
    t_all = time.perf_counter()
    try:
        ls = r.load(str(model), mm, handler=arm.get("handler", "auto"), n_gpu_layers=arm.get("n_gpu_layers", -1), n_ctx=n_ctx, kv=arm.get("kv_cache", "q8_0"),
                    flash_attn=True, n_cpu_moe=arm.get("n_cpu_moe", 0), mmproj_gpu=arm.get("mmproj_on_gpu", True))
        print("loaded", json.dumps(ls))
        for seed in [int(x) for x in a.seeds.split(",") if x.strip()]:
            for name in arms_wanted:
                sc = a.kind if name == "schema" else RS.schema_for("design", prompt, lock_text=False) if name == "schema_nolock" else None
                kw = dict(system="", max_tokens=a.max_tokens, temperature=a.temperature, top_p=arm.get("top_p", 0.9), top_k=arm.get("top_k", 40),
                          repeat_penalty=arm.get("repeat_penalty", 1.0), seed=seed, json_mode=(name == "json"), schema=sc)
                text, st = r.chat(prompt, img, **kw)
                answer, _ = LG.split_think(text)
                chk = check_reply(answer, a.kind, prompt)
                (out / f"{a.kind}_{name}_seed{seed}.txt").write_text(text, encoding="utf-8")
                runs.append(dict(arm=name, seed=seed, **st, check=chk))
                print(f"{name:7s} seed {seed}: {st['gen_tokens']} tok, {st['gen_tok_s']} tok/s, prompt {st['prompt_s']} s, total {st['total_s']} s, finish {st['finish']}, "
                      f"syntax {chk['syntax_ok']}, valid {chk['valid']}, value violations {chk['value_violations']}, fields {chk['fields'][:4]}", flush=True)
    finally:
        mem = smp.stop()
        r.close()
    summary = summarize(runs)
    print("\n" + summary + f"\n\nVRAM peak {mem['vram_peak_mib']} MiB (base {mem['vram_base_mib']}), RSS peak {mem['rss_peak_gib']} GiB, wall {time.perf_counter() - t_all:.0f} s")
    (out / "probe.json").write_text(json.dumps(dict(kind=a.kind, model=str(model), arm=a.arm, n_ctx=n_ctx, schema_chars=len(json.dumps(sch)), gbnf_chars=len(g) if g else None,
                                                   runs=runs, memory=mem, summary_md=summary), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
