"""End-to-end smoke test for the MOTNGRFX web app against a running ComfyUI (does what the page does).

    python app/smoke_local.py --port 8190
    python app/smoke_local.py --port 8190 --example after-hours --poster      # also saves app/examples/<id>.mp4

Steps: GET /mograph, GET /mograph/graphs + /mograph/graphs/<name> (template + sidecar), upload the fixture image
(and audio), fill the fields by node title, POST /prompt with a client_id, follow /ws until done, print per-node
timings, then confirm the "OUT final" file exists via /history and /view (ffprobe if available).

Needs only aiohttp (ships with ComfyUI). Run it with ComfyUI's python.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "app" / "examples"
FIXTURES = ROOT / "tests" / "fixtures"


# ---- binding: Python port of web_app/bind.js (buildPrompt) --------------------------------------------------------

def title_map(graph: dict) -> dict:
    return {n["_meta"]["title"]: i for i, n in graph.items() if n.get("_meta", {}).get("title")}


def opt_value(o):
    return o["value"] if isinstance(o, dict) else o


def is_empty(f: dict, v) -> bool:
    if f["kind"] in ("image", "audio", "seed"):
        return v in (None, "")
    return v is None or str(v).strip() == ""


def coerce(f: dict, v):
    if f["kind"] == "seed":
        return max(0, min(4294967295, int(v)))
    if f["kind"] in ("select", "number"):
        match = [opt_value(o) for o in f.get("options", []) if str(opt_value(o)) == str(v)]
        out = match[0] if match else v
        if isinstance(out, (int, float)):
            if "max" in f:
                out = min(out, f["max"])
            if "min" in f:
                out = max(out, f["min"])
        return out
    return str(v)


def build_prompt(graph: dict, app: dict, values: dict, uploads: dict) -> tuple[dict, dict]:
    prompt = json.loads(json.dumps(graph))
    tm = title_map(prompt)
    drop: list[str] = []
    used: dict = {}
    for f in app.get("fields", []):
        if f.get("node_title") not in tm:          # a media field (16_app/MEDIA_BOX.md) has no node_title of its own
            continue
        node = prompt[tm[f["node_title"]]]
        is_file = f["kind"] in ("image", "audio")
        raw = uploads.get(f["id"]) if is_file else values.get(f["id"])
        if is_empty(f, raw) and not is_file and f.get("default") not in (None, ""):
            raw = f["default"]  # same fallback as the page
        if is_empty(f, raw):
            if f["kind"] == "seed":
                node["inputs"][f["input_name"]] = used[f["id"]] = random.randrange(2 ** 32)
            elif not f.get("optional"):
                raise SystemExit(f"missing required field: {f['id']}")
            elif f.get("remove_when_empty"):
                drop += f["remove_when_empty"]
            elif f["kind"] in ("textarea", "text"):
                node["inputs"][f["input_name"]] = ""
            continue
        node["inputs"][f["input_name"]] = used[f["id"]] = coerce(f, raw)
    ids = {tm[t] for t in drop if t in tm}
    for i in ids:
        prompt.pop(i, None)
    for n in prompt.values():
        for k in [k for k, v in n["inputs"].items() if isinstance(v, list) and len(v) == 2 and str(v[0]) in ids]:
            del n["inputs"][k]
    return prompt, used


# ---- run --------------------------------------------------------------------------------------------------------

async def upload(sess: aiohttp.ClientSession, base: str, path: Path) -> str:
    fd = aiohttp.FormData()
    fd.add_field("image", path.read_bytes(), filename="mograph_" + path.name)
    fd.add_field("type", "input")
    fd.add_field("overwrite", "true")
    async with sess.post(f"{base}/upload/image", data=fd) as r:
        r.raise_for_status()
        j = await r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


async def amain(a) -> int:
    base = f"http://{a.host}:{a.port}"
    t_all = time.perf_counter()
    async with aiohttp.ClientSession() as sess:
        # 1. page + static assets
        for path, needle in (("/mograph", "<title>MOTNGRFX</title>"), ("/motngrfx", "<title>MOTNGRFX</title>"), ("/mograph/static/node_packs.json", "packs"), ("/mograph/static/app.js", "MographBind"),
                             ("/mograph/static/app.css", "--yellow"), ("/mograph/static/bind.js", "createTracker")):
            async with sess.get(base + path) as r:
                body = await r.text()
                ok = r.status == 200 and needle in body
                print(f"GET {path}: {r.status} {'ok' if ok else 'MISSING ' + needle}")
                if not ok:
                    return 1
        # 2. template + sidecar
        async with sess.get(f"{base}/mograph/graphs") as r:
            graphs = await r.json()
        print("templates:", [g["name"] for g in graphs])
        async with sess.get(f"{base}/mograph/graphs/{a.graph}") as r:
            tpl = await r.json()
        graph, app = tpl["graph"], tpl["app"]
        tm = title_map(graph)
        print("visible fields:", [f["id"] for f in app["fields"] if f.get("node_title") in tm])

        # 3. inputs (optionally from an example)
        values = {"words": a.words, "length": a.length, "quality": a.quality, "seed": a.seed, "format": a.format}
        image, audio = Path(a.image) if a.image else None, Path(a.audio) if a.audio else None
        if a.example:
            ex = json.loads((EXAMPLES / f"{a.example}.json").read_text(encoding="utf-8"))
            for k, v in ex["inputs"].items():
                if k == "image":
                    image = EXAMPLES / v
                elif k == "audio":
                    audio = EXAMPLES / v
                else:
                    values[k] = v
        t0 = time.perf_counter()
        uploads = {}
        if image:
            uploads["image"] = await upload(sess, base, image)
        if audio:
            uploads["audio"] = await upload(sess, base, audio)
        t_up = time.perf_counter() - t0
        prompt, used = build_prompt(graph, app, values, uploads)
        print("inputs used:", used, "| uploads:", uploads, f"| upload {t_up:.2f}s")
        print("prompt nodes:", {i: n["_meta"]["title"] for i, n in prompt.items()})

        # 4. websocket + POST /prompt
        client_id = uuid.uuid4().hex
        timings: dict[str, dict] = {}
        titles = {i: n["_meta"]["title"] for i, n in prompt.items()}
        running: str | None = None
        outputs: dict[str, dict] = {}
        pid = None
        t_start = None
        async with sess.ws_connect(f"{base}/ws?clientId={client_id}", max_msg_size=0) as ws:
            async with sess.post(f"{base}/prompt", json={"prompt": prompt, "client_id": client_id}) as r:
                body = await r.json()
                if r.status != 200:
                    print("POST /prompt rejected:", json.dumps(body, indent=1)[:2000])
                    return 1
            pid = body["prompt_id"]
            print("prompt_id:", pid)
            deadline = time.perf_counter() + a.timeout
            ok = False

            def end(node, t):
                if node in timings and timings[node].get("end") is None:
                    timings[node]["end"] = t

            while time.perf_counter() < deadline:
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                if m.type != aiohttp.WSMsgType.TEXT:
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                msg = json.loads(m.data)
                d, t = msg.get("data", {}), time.perf_counter()
                if d.get("prompt_id") not in (None, pid):
                    continue
                typ = msg["type"]
                if typ == "execution_start":
                    t_start = t
                elif typ == "execution_cached":
                    for n in d.get("nodes", []):
                        timings[str(n)] = {"start": t, "end": t, "cached": True}
                elif typ == "executing":
                    node = d.get("node")
                    if running is not None and running != node:
                        end(running, t)
                    if node is not None:
                        timings.setdefault(str(node), {"start": t, "end": None})
                        running = str(node)
                        print(f"  [{t - (t_start or t):6.2f}s] executing {titles.get(running, running)}")
                elif typ == "progress":
                    print(f"  progress {titles.get(str(d.get('node')), d.get('node'))}: {d['value']}/{d['max']}")
                elif typ == "executed":
                    outputs[str(d["node"])] = d.get("output", {})
                    end(str(d["node"]), t)
                elif typ == "execution_success":
                    end(running, t) if running else None
                    ok = True
                    t_end = t
                    break
                elif typ in ("execution_error", "execution_interrupted"):
                    print("EXECUTION FAILED:", json.dumps(d, indent=1)[:2500])
                    return 1
            if not ok:                  # silent or closed socket: /history is the truth (a finished prompt must not read as a timeout)
                try:
                    async with sess.get(f"{base}/history/{pid}", timeout=aiohttp.ClientTimeout(total=15)) as r:
                        h = (await r.json()).get(pid)
                except Exception:
                    h = None
                if h and (h.get("status") or {}).get("status_str") == "success":
                    print("socket silent/closed, but /history says the prompt succeeded")
                    ok, t_end = True, time.perf_counter()
                    t_start = t_start or t_end
                else:
                    print("timed out / socket closed before execution_success")
                    return 1

        # 5. timings
        print("\nper-node timings (client side, from /ws events):")
        rows = []
        for nid in sorted(timings, key=lambda k: timings[k]["start"]):
            tt = timings[nid]
            sec = None if tt.get("end") is None else tt["end"] - tt["start"]
            rows.append({"id": nid, "title": titles.get(nid, nid), "seconds": None if sec is None else round(sec, 3),
                         "cached": bool(tt.get("cached"))})
            print(f"  {titles.get(nid, nid):<14} {('%.3f s' % sec) if sec is not None else '-':>10}"
                  f"{'  (cached)' if tt.get('cached') else ''}")
        total = t_end - t_start
        print(f"  {'TOTAL':<14} {total:>8.3f} s   (click-to-done incl. upload {time.perf_counter() - t_all:.2f} s)")
        print("MOGRAPH_TIMINGS " + json.dumps({"template": a.graph, "total_seconds": round(total, 3), "nodes": rows}))

        # 6. confirm OUT final via /history and fetch it through /view
        hist = None
        for _ in range(40):   # the history entry is written just after execution_success
            async with sess.get(f"{base}/history/{pid}") as r:
                hist = (await r.json()).get(pid)
            if hist:
                break
            await asyncio.sleep(0.25)
        if not hist:
            print("FAIL: prompt never appeared in /history")
            return 1
        status = hist["status"]["status_str"]
        out_id = title_map(prompt).get("OUT final")
        files = []
        for k in ("images", "gifs", "video", "videos"):
            files += hist["outputs"].get(out_id, {}).get(k, [])
        print(f"history status: {status}; OUT final node {out_id}: {files}")
        if status != "success" or not files:
            print("FAIL: no OUT final output in history")
            return 1
        f = files[0]
        async with sess.get(f"{base}/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                                    "type": f.get("type", "output")}) as r:
            data = await r.read()
            print(f"GET /view -> {r.status}, {len(data)} bytes, {r.headers.get('Content-Type')}")
            if r.status != 200 or len(data) < 1000:
                return 1
        out = Path(a.out) if a.out else ROOT / "tests" / "out" / "smoke_local_final.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print("saved", out)
        if a.poster and a.example:
            dest = EXAMPLES / f"{a.example}.mp4"
            dest.write_bytes(data)
            print("poster saved", dest)
            if shutil.which("ffmpeg"):   # still frame shown in the examples grid before hover/play
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.2", "-i", str(dest), "-frames:v", "1",
                                "-q:v", "4", str(EXAMPLES / f"{a.example}.jpg")], check=False)
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            p = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                                "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,duration",
                                "-of", "json", str(out)], capture_output=True, text=True)
            print("ffprobe:", p.stdout.strip() or p.stderr.strip())
        else:
            print("ffprobe not on PATH; skipped")
    print("SMOKE OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--graph", default="local_v1")
    ap.add_argument("--words", default="AFTER\nHOURS\nTHE CITY\nKEEPS\nDREAMING")
    ap.add_argument("--image", default=str(FIXTURES / "blob_gradient.png"))
    ap.add_argument("--audio", default="")
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--quality", default="draft")
    ap.add_argument("--format", default="9:16", choices=["9:16", "16:9"])
    ap.add_argument("--out", default="", help="save the OUT final file here (default tests/out/smoke_local_final.mp4)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--example", default="", help="use inputs of app/examples/<id>.json")
    ap.add_argument("--poster", action="store_true", help="with --example: save the MP4 as the example poster")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    return asyncio.run(amain(a))


if __name__ == "__main__":
    sys.exit(main())
