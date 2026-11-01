#!/usr/bin/env python
"""Benchmark inputs and scoring for the LLM arms (tools/app/llm_arms.py). Local, CPU, no GPU, no network, no model.

    python tools/app/llm_probe.py package --shots <batch run out/..._rNN_out_shots> --out <dir>
        rebuilds the design package of a batch run from its archive (piece.json + sK_clip.mp4 + sK_matte/): writes <dir>/prompt.txt, sheet.png,
        locked.json, piece.json (the rules-side piece the parse falls back to), music.json. The music analysis is rebuilt from the piece
        (bpm, beats, downbeats; the real run also had chord changes / curves), so the prompt is the run's prompt minus those music details.
    python tools/app/llm_probe.py score --dir <dir> --reply reply.txt [--label NAME] [--json out.json]
        design_step.parse() of a model reply against the package: valid?, fields from the model / fixed / from the rules (design_salvage), sections,
        and the variety of the design (distinct fonts / modules / io styles / transitions), as one json line `SCORE {...}`.

The package is the same code path the graph's MographDesignPackage node runs (`design_step.package`), fed from the archive instead of ComfyUI memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _load_clip(path: Path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def _load_matte(folder: Path, step: int = 1):
    import cv2
    import numpy as np
    files = sorted(folder.glob("*.png"))[::step]
    return np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]) if files else None


def package_from_archive(shots_dir: str | Path, guide: str | None = None, seed: int | None = None) -> dict:
    """design_step.package() for a batch run's archive. -> the dict of package() + piece, music, request."""
    import numpy as np
    from tools.fusion import design_step as DS
    d = Path(shots_dir)
    piece = json.loads((d / "piece.json").read_text(encoding="utf-8"))
    music = DS.ana_from_piece(piece)
    sids = [s["id"] for s in piece["shots"]]
    clips = [dict(sid=sid, u8=_load_clip(d / f"{sid}_clip.mp4")) for sid in sids if (d / f"{sid}_clip.mp4").is_file()]
    mattes = {sid: _load_matte(d / f"{sid}_matte", 4) for sid in sids if (d / f"{sid}_matte").is_dir()}
    request = dict(words="\n".join(w["text"] for w in piece["type"]["words"]), scene=(piece.get("stills") or [{}])[0].get("prompt", ""), music="",
                   seed=int((piece.get("plan") or {}).get("seed") or 1))
    pk = DS.package(request, music, piece, clips, mattes, guide, int(seed if seed is not None else request["seed"]))
    return dict(pk, piece=piece, music=music, request=request)


def write_package(pk: dict, out: str | Path) -> dict:
    import cv2
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "prompt.txt").write_text(pk["prompt"], encoding="utf-8")
    cv2.imwrite(str(o / "sheet.png"), cv2.cvtColor(pk["sheet"], cv2.COLOR_RGB2BGR))
    (o / "locked.json").write_text(pk["locked"], encoding="utf-8")
    (o / "piece.json").write_text(json.dumps(pk["piece"], ensure_ascii=False), encoding="utf-8")
    (o / "music.json").write_text(json.dumps(pk["music"], ensure_ascii=False), encoding="utf-8")
    info = dict(prompt_chars=len(pk["prompt"]), sheet_hw=list(pk["sheet"].shape[:2]), techniques=pk["techniques"], desc=pk["desc"])
    (o / "package.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def variety(piece: dict) -> dict:
    """how varied the DESIGN is (the final piece after salvage): distinct fonts / modules / io styles / transitions / palette keys and the technique."""
    ws = (piece.get("type") or {}).get("words") or []
    tr = [t for t in ((piece.get("type") or {}).get("transitions") or []) if isinstance(t, str)]
    comp = piece.get("compose") or {}
    place = {json.dumps(w.get("placement"), sort_keys=True) for w in ws}
    return dict(words=len(ws), fonts=len({w.get("font") for w in ws}), modules=len({w.get("module") for w in ws}),
                placements=len(place), holds=len(comp.get("holds") or []), cuts=len((piece.get("plan") or {}).get("window_notes") or []),
                matte_treatments=sum(1 for w in ws if w.get("matte")), axes_words=sum(1 for w in ws if w.get("axes")),
                form_in=len({w.get("form_in") for w in ws}), form_out=len({w.get("form_out") for w in ws}), transitions=len(set(tr)),
                camera=(piece.get("type") or {}).get("camera"), technique=piece.get("technique"),
                fonts_list=sorted({str(w.get("font")).split("[")[0].replace(".ttf", "") for w in ws}), modules_list=sorted({str(w.get("module")) for w in ws}))


def score(dir_: str | Path, reply: str, label: str = "") -> dict:
    from tools.fusion import design_step as DS
    d = Path(dir_)
    piece = json.loads((d / "piece.json").read_text(encoding="utf-8"))
    music = json.loads((d / "music.json").read_text(encoding="utf-8"))
    locked = (d / "locked.json").read_text(encoding="utf-8")
    prompt = (d / "prompt.txt").read_text(encoding="utf-8")
    from tools.fusion import qwen_piece as QP
    obj = QP.extract_json(reply or "")
    r = DS.parse(reply, locked, piece, music, prompt, model=label or DS.MODEL)
    f = dict(r.get("fields") or {})
    if f and '"ltx_at"' not in prompt:                      # a package built before the per-word render / ltx_at / material schema: those fields were never asked for
        new_keys = ("ltx_at", "material", "render")
        keep = [x for x in f.get("fields_from_rules") or [] if not (x["path"].rsplit(".", 1)[-1] in new_keys and "missing" in x.get("why", ""))]
        dropped = len(f.get("fields_from_rules") or []) - len(keep)
        f["fields_from_rules"], f["n"] = keep, f["n"] - dropped
        f["pct_model"] = round(100.0 * len(f.get("fields_from_model") or []) / max(f["n"], 1), 1)
    concept = str((obj or {}).get("concept") or "")
    import difflib
    best = max((difflib.SequenceMatcher(None, concept.lower(), ln.lower()).ratio() for ln in prompt.splitlines() if len(ln) > 30), default=0.0) if concept else 0.0
    out = dict(label=label, reply_chars=len(reply or ""), json_found=obj is not None, concept=concept[:200], concept_copied_from_prompt=round(best, 2), valid_first_pass=bool(r["ok"]),
               fields_total=f.get("n"), fields_from_model=len(f.get("fields_from_model") or []), fields_fixed=len(f.get("fields_fixed") or []),
               fields_from_rules=len(f.get("fields_from_rules") or []), pct_model=f.get("pct_model"),
               sections=((r["piece"].get("plan") or {}).get("design") or {}).get("src"), repair_needed=not r["ok"],
               variety=variety(r["piece"]), report=r["report"].splitlines()[:40])
    return out


STANDIN = Path(os.environ.get("PROGRESS_STANDIN") or Path.home() / "Documents/Dev/comfyui-projects/tools/progress_standin.py")
SSH = ["ssh", "-o", "ConnectTimeout=20", "-p", "60181", "-i", str(Path.home() / ".ssh/vastai_comfyui_rsa"), "root@<box-ip>"]


def run_graph(graph: dict, name: str, out_dir: Path, port: int = 9000, iid: str = "57999748") -> dict:
    """submit a probe graph through progress_standin (the dashboard tile; global rule), read the PreviewAny texts from /history.
    -> dict(rc, result_line, reply, stats, prompt_id)."""
    import subprocess
    import urllib.request
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gp, pidf = out_dir / f"{name}.graph.json", out_dir / f"{name}.pid"
    gp.write_text(json.dumps(graph), encoding="utf-8")
    pidf.unlink(missing_ok=True)
    cp = subprocess.run([sys.executable, str(STANDIN), "--iid", iid, "--port", str(port), "--no-project", "--name", name, "--prompt-id-file", str(pidf),
                         "submit", str(gp)], capture_output=True, text=True, cwd=str(STANDIN.parents[2]), encoding="utf-8", errors="replace")
    (out_dir / f"{name}.standin.log").write_text(cp.stdout + "\n" + cp.stderr, encoding="utf-8")
    res = next((l for l in cp.stdout.splitlines()[::-1] if l.startswith("RESULT")), "")
    pid = pidf.read_text().strip() if pidf.is_file() else ""
    reply, stats = "", ""
    if pid:
        import time
        h = {}
        for _ in range(30):                                   # the history entry can lag the DONE event by a few seconds
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30)).get(pid, {})
            if h.get("outputs"):
                break
            time.sleep(2)
        for nid, o in (h.get("outputs") or {}).items():
            txt = "\n".join(o.get("text") or [])
            if nid == "out_reply":
                reply = txt
            elif nid == "out_stats":
                stats = txt
        if not reply:
            (out_dir / f"{name}.history.json").write_text(json.dumps(h)[:20000], encoding="utf-8")
    return dict(rc=cp.returncode, result_line=res, reply=reply, stats=stats, prompt_id=pid)


def table(probes_dir: str | Path, pkg_dir: str | Path | None = None) -> str:
    """markdown rows (one per arm, mean over its seeds) from <probes_dir>/llmprobe_*.json: the LLM_ARMS.md table."""
    rows: dict[str, list[dict]] = {}
    for f in sorted(Path(probes_dir).glob("llmprobe_*.json")):
        if f.name.endswith(".graph.json") or f.name.endswith(".history.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        rf = f.with_name(f.stem + ".reply.txt")
        if pkg_dir and rf.is_file() and rf.stat().st_size:                  # re-score with the current scorer (new metrics apply to old replies)
            d["score"] = {k: v for k, v in score(pkg_dir, rf.read_text(encoding="utf-8"), d.get("arm", "")).items() if k != "report"}
        rows.setdefault(d.get("arm", f.stem), []).append(d)

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 1) if xs else None
    lines = ["| arm | seeds | valid 1st pass | % fields from model | fixed | from rules | fonts / modules / placements | concept copied | load s | TTFT s | prompt tok/s | gen tok/s | total s | VRAM peak MiB | RAM peak GiB |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, ds in rows.items():
        sc = [d.get("score") or {} for d in ds]
        st = [d["stats"] if isinstance(d.get("stats"), dict) else {} for d in ds]
        v = [x.get("variety") or {} for x in sc]
        vram = [int(m.group(1)) for d in ds for m in [re.search(r"peakVRAM\s+(\d+)", d.get("result") or "")] if m]
        ram = [float(m.group(1)) for d in ds for m in [re.search(r"peakRAM\s+([\d.]+)", d.get("result") or "")] if m]
        lines.append(f"| {arm} | {len(ds)} | {sum(1 for x in sc if x.get('valid_first_pass'))}/{len(ds)} | {mean([x.get('pct_model') for x in sc])} | "
                     f"{mean([x.get('fields_fixed') for x in sc])} | {mean([x.get('fields_from_rules') for x in sc])} | "
                     f"{mean([y.get('fonts') for y in v])} / {mean([y.get('modules') for y in v])} / {mean([y.get('placements') for y in v])} | "
                     f"{mean([x.get('concept_copied_from_prompt') for x in sc])} | {mean([y.get('load_s') for y in st])} | {mean([y.get('prompt_s') for y in st])} | {mean([y.get('prompt_tok_s') for y in st])} | "
                     f"{mean([y.get('gen_tok_s') for y in st])} | {mean([y.get('total_s') for y in st])} | {max(vram) if vram else None} | {max(ram) if ram else None} |")
    return "\n".join(lines)


def _overrides(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = True if v.lower() == "true" else False if v.lower() == "false" else int(v) if re.fullmatch(r"-?\d+", v) else v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("--shots", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--guide")
    p.add_argument("--seed", type=int)
    r = sub.add_parser("run", help="build + submit the one-model probe graph, save reply / stats / score under --out")
    r.add_argument("--dir", required=True)
    r.add_argument("--arm", help="arm key (llm_arms.ARMS); omit with --baseline")
    r.add_argument("--baseline", action="store_true")
    r.add_argument("--n-gpu-layers", type=int)
    r.add_argument("--n-ctx", type=int)
    r.add_argument("--tag", default="")
    r.add_argument("--seed", type=int, default=3)
    r.add_argument("--verbose", action="store_true", help="llama.cpp load log into the ComfyUI log (layers offloaded)")
    r.add_argument("--set", action="append", default=[], help="node input override k=v (kv_cache=f16, mmproj_on_gpu=false, n_cpu_moe=4 ...)")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--pkg", help="package dir: re-score every saved reply with the current scorer")
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--reply", required=True)
    s.add_argument("--label", default="")
    s.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        info = write_package(package_from_archive(a.shots, a.guide, a.seed), a.out)
        print(json.dumps(info))
        return 0
    if a.cmd == "table":
        print(table(a.dir, a.pkg))
        return 0
    if a.cmd == "run":
        from tools.app import llm_arms as LA
        prompt = (Path(a.dir) / "prompt.txt").read_text(encoding="utf-8")
        g = (LA.baseline_probe_graph(prompt, seed=a.seed) if a.baseline else
             LA.probe_graph(a.arm, prompt, n_gpu_layers=a.n_gpu_layers, n_ctx=a.n_ctx, seed=a.seed, verbose=a.verbose, **_overrides(a.set)))
        label = "baseline_design_llm_8b_4bit" if a.baseline else a.arm
        name = f"llmprobe_{label}{a.tag}_s{a.seed}"
        out = Path(a.out)
        r = run_graph(g, name, out)
        (out / f"{name}.reply.txt").write_text(r["reply"], encoding="utf-8")
        sc = score(a.dir, r["reply"], label) if r["reply"] else {}
        rec = dict(arm=label, tag=a.tag, seed=a.seed, n_gpu_layers=a.n_gpu_layers, result=r["result_line"], rc=r["rc"], stats=json.loads(r["stats"]) if r["stats"].strip().startswith("{") else r["stats"],
                   score={k: v for k, v in sc.items() if k != "report"}, report=sc.get("report"))
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in rec.items() if k != "report"}, ensure_ascii=False)[:3000])
        return 0 if r["reply"] else 1
    sc = score(a.dir, Path(a.reply).read_text(encoding="utf-8", errors="replace"), a.label)
    if a.json:
        Path(a.json).write_text(json.dumps(sc, indent=1, ensure_ascii=False), encoding="utf-8")
    print("SCORE " + json.dumps({k: v for k, v in sc.items() if k != "report"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
