"""`/mograph/own_video`: the own-video route of the app (16_app/MEDIA_BOX.md "Own-video route").

    POST /mograph/own_video   {video, brief, length, seed, format, quality, soundtrack, audio?, edit?, scale?, client_id?}  -> {"job": id}      (409 while another job runs)
        scale (2026-11-03) = "full" (DEFAULT, the page's "Original") | "half" ("Half scale": compose, type and the final at half the source size, even sides, same fps / frame count); anything
        else is a 400. Passed to the runner as --scale and written to request.json. `format` is the clip's orientation only ("9:16" | "16:9"); the runner takes the real size from the clip.
        edit (2026-11-01) = "overlay" (DEFAULT: the picture is untouched, only type is composited over / behind it) | "redesign" (with transitions: 3D cuts, effects). Read from the body, else from
        brief.edit; the runner gets it as --edit and the brief.json it reads carries it (own_video_job.read_brief). Both modes keep the source size, fps and frame count.
        soundtrack (v19) = "video" (the clip's own sound; the old "keep" is accepted) | "track" (the user's audio file `audio`, a name in the input folder; required) | "generate" (new music)
    GET  /mograph/own_video/<job>  -> {job, stage, stage_n, stages, log: [last 20 lines], prompt_ids, outputs: [{kind, url}], error, done, times, cached: {stage: true}}
    POST /mograph/own_video/<job>/cancel  -> {job, cancelled, killed, interrupted, dequeued}   (2026-11-01 "cancel + restart with reuse": the runner's process group is killed, the job's queued
        ComfyUI prompts are deleted and the running one interrupted, job.json gets cancelled: true / stage "cancelled" / status "cancelled", the 409 lock is free again. What the stages
        already made stays in the stage cache (tools/app/stage_cache.py) and is reused by the next job whose inputs have not changed.)

The job runs in a background subprocess (`tools/app/own_video_job.py`, the ComfyUI venv python) so the server stays responsive; its state is the file
`<output>/motngrfx_app/own/<job>/job.json`, which the runner rewrites at every step (the GET just reads it). One job at a time. `client_id` (the page's websocket id)
makes the runner's /prompt submits carry it, so the page gets the progress events of every queued graph.

This module has no ComfyUI import at module level: `JobManager` is plain Python (tests mock `popen`); `register(routes, ...)` wires it into aiohttp.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB_SCRIPT = ROOT / "tools" / "app" / "own_video_job.py"
OUT_SUB = "motngrfx_app/own"        # App label: everything the app makes lives under output/motngrfx_app/ (16_app/MEDIA_BOX.md "App label")
SOUNDTRACKS = ("video", "track", "generate")
EDITS = ("overlay", "redesign")


def norm_edit(v) -> str:
    """"overlay" (default) | "redesign" ("with transitions"); same rule as tools/app/overlay_mode.norm_edit (kept here so the route imports nothing heavy)."""
    s = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "redesign" if s in ("redesign", "with_transitions", "transitions", "full", "edit") else "overlay"
SOUNDTRACK_ALIASES = {"keep": "video"}          # v19: the old name of "video"
QUALITIES = ("still", "preview", "draft", "final")
FORMATS = ("9:16", "16:9")      # the clip's orientation, sent by the page; the runner ignores it (the source-size route takes the clip's own aspect)
SCALES = ("full", "half")
JOB_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
LOG_LINES = 20
STAGES = ["queued", "ingest", "audio", "music", "analyse", "words", "prep", "cut", "matte", "archive", "design", "parse", "retime", "render", "finish", "done"]


class BadRequest(ValueError):
    pass


class Busy(RuntimeError):
    pass


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def parse_params(body: dict, input_dir: Path) -> dict:
    """validate the POST body -> the runner's parameters. Raises BadRequest."""
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    video = str(body.get("video") or "").strip()
    if not video or "/" in video or "\\" in video or video.startswith("."):
        raise BadRequest("video: a file name in the ComfyUI input folder is required")
    path = Path(input_dir) / video
    if not path.is_file():
        raise BadRequest(f"video: {video} is not in the input folder")
    brief = body.get("brief") or {}
    if isinstance(brief, str):
        try:
            brief = json.loads(brief) if brief.strip() else {}
        except ValueError:
            raise BadRequest("brief: not valid JSON") from None
    if not isinstance(brief, dict):
        raise BadRequest("brief: the READ JSON must be an object")
    try:
        length = float(body.get("length") or 12)
    except (TypeError, ValueError):
        raise BadRequest("length: not a number") from None
    if not 0 < length < 86400:                                                  # the page sends the video's own length; the job's window stops at the clip's end, so nothing is capped here (2026-11-03)
        raise BadRequest("length: not a length")
    length = int(length) if length == int(length) else length
    try:
        seed = int(body.get("seed") or 1)
    except (TypeError, ValueError):
        raise BadRequest("seed: not a number") from None
    fmt = str(body.get("format") or "9:16").strip()
    if fmt not in FORMATS:
        raise BadRequest(f"format: only {', '.join(FORMATS)} for the own-video route")
    quality = str(body.get("quality") or "final").strip().lower()
    if quality not in QUALITIES:
        raise BadRequest(f"quality: one of {', '.join(QUALITIES)}")
    soundtrack = str(body.get("soundtrack") or "video").strip().lower()
    soundtrack = SOUNDTRACK_ALIASES.get(soundtrack, soundtrack)
    if soundtrack not in SOUNDTRACKS:
        raise BadRequest(f"soundtrack: one of {', '.join(SOUNDTRACKS)}")
    audio = str(body.get("audio") or "").strip()
    audio_path = ""
    if audio:
        if "/" in audio or "\\" in audio or audio.startswith("."):
            raise BadRequest("audio: a file name in the ComfyUI input folder is required")
        if not (Path(input_dir) / audio).is_file():
            raise BadRequest(f"audio: {audio} is not in the input folder")
        audio_path = str(Path(input_dir) / audio)
    elif soundtrack == "track":
        raise BadRequest("audio: soundtrack \"track\" needs the audio file (a name in the ComfyUI input folder)")
    scale = str(body.get("scale") if body.get("scale") not in (None, "") else "full").strip().lower()
    if scale not in SCALES:
        raise BadRequest(f"scale: one of {', '.join(SCALES)}")
    cid = str(body.get("client_id") or "")
    if cid and not re.fullmatch(r"[\w\-]{1,80}", cid):
        cid = ""
    edit = norm_edit(body.get("edit") if body.get("edit") not in (None, "") else brief.get("edit"))
    brief = dict(brief, edit=edit)
    return dict(edit=edit, video=video, video_path=str(path), brief=brief, length=length, seed=seed, format=fmt, scale=scale, quality=quality, soundtrack=soundtrack, audio=audio, audio_path=audio_path, client_id=cid)


class JobManager:
    def __init__(self, output_dir, input_dir, comfy_dir, port: int = 8188, popen=subprocess.Popen, python: str | None = None, script: Path | None = None):
        self.output_dir, self.input_dir, self.comfy_dir = Path(output_dir), Path(input_dir), Path(comfy_dir)
        self.port, self.popen = int(port), popen
        self.python = python or sys.executable
        self.script = Path(script or JOB_SCRIPT)
        self.procs: dict[str, object] = {}

    # ---------------------------------------------------------------------------------------------------- paths
    def job_dir(self, job: str) -> Path:
        return self.output_dir / OUT_SUB / job

    def state_path(self, job: str) -> Path:
        return self.job_dir(job) / "job.json"

    def _read_state(self, job: str) -> dict | None:
        try:
            return json.loads(self.state_path(job).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------------------------------------------- busy
    def running(self) -> str | None:
        """id of the job that is still running (own children first, then any job.json that is not finished and whose runner pid is alive)."""
        for job, p in list(self.procs.items()):
            if p.poll() is None and not (self._read_state(job) or {}).get("cancelled"):
                return job
        base = self.output_dir / OUT_SUB
        if base.is_dir():
            for d in sorted(base.iterdir(), reverse=True)[:20]:
                st = self._read_state(d.name) if JOB_RE.match(d.name) else None
                if st and not st.get("done") and not st.get("error") and not st.get("cancelled") and _alive(st.get("pid")):
                    return d.name
        return None

    # ---------------------------------------------------------------------------------------------------- start
    def start(self, body: dict) -> str:
        p = parse_params(body, self.input_dir)
        busy = self.running()
        if busy:
            raise Busy(busy)
        job = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        d = self.job_dir(job)
        d.mkdir(parents=True, exist_ok=True)
        (d / "brief.json").write_text(json.dumps(p["brief"], ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "request.json").write_text(json.dumps({k: v for k, v in p.items() if k != "brief"}, indent=1), encoding="utf-8")
        argv = [self.python, str(self.script), "--job", job, "--video", p["video_path"], "--brief-file", str(d / "brief.json"), "--length", str(p["length"]),
                "--seed", str(p["seed"]), "--format", p["format"], "--scale", p["scale"], "--quality", p["quality"], "--soundtrack", p["soundtrack"], "--edit", p["edit"],
                "--out-dir", str(d), "--comfy-dir", str(self.comfy_dir), "--port", str(self.port)]
        if p["audio_path"] and p["soundtrack"] == "track":
            argv += ["--audio", p["audio_path"]]
        if p["client_id"]:
            argv += ["--client-id", p["client_id"]]
        state = dict(job=job, stage="queued", stage_n=0, stages=STAGES, log=[], prompt_ids=[], outputs=[], error=None, done=False, times={}, pid=None, started=time.time())
        self.write_state(job, state)
        log = open(d / "runner.log", "ab")
        try:
            kw = {"start_new_session": True} if os.name == "posix" else {}              # its own process group: cancel() kills the runner and every ffmpeg / python it started
            proc = self.popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1", "COMFY_DIR": str(self.comfy_dir)}, **kw)
        except OSError as e:
            state.update(error=f"could not start the runner: {e}", stage="error")
            self.write_state(job, state)
            return job
        self.procs[job] = proc
        if isinstance(getattr(proc, "pid", None), int):
            state["pid"] = proc.pid
            self.write_state(job, state)
        return job

    # ---------------------------------------------------------------------------------------------------- cancel
    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 10):
        """one JSON request to the ComfyUI of this server (tests replace this)."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8") or "null"
        return json.loads(raw)

    @staticmethod
    def _kill_runner(proc, pid, grace: float = 5.0) -> bool:
        """kill the runner's whole process group (SIGTERM, then SIGKILL after `grace` s); never our own group. True if the runner is gone."""
        pid = pid or getattr(proc, "pid", None)
        if not isinstance(pid, int) or pid <= 1:
            return True

        def gone() -> bool:
            if proc is not None and hasattr(proc, "poll"):
                return proc.poll() is not None
            return not _alive(pid)
        if os.name != "posix":
            subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"], capture_output=True)
            return gone()
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return True
        group = pgid != os.getpgid(0) and pgid == pid            # started by start(): the runner leads its own group
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if group:
                    os.killpg(pgid, sig)
                else:
                    os.kill(pid, sig)
            except OSError:
                pass
            t_end = time.time() + (grace if sig == signal.SIGTERM else 2.0)
            while time.time() < t_end and not gone():
                time.sleep(0.05)
            if gone():
                if group:                                         # children that ignored SIGTERM with the leader gone
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                return True
        return gone()

    def _comfy_cancel(self, prompt_ids: list) -> tuple[list, list]:
        """delete OUR queued prompts from ComfyUI and interrupt the running one only if it is ours (another client's work is never touched). -> (dequeued ids, interrupted ids)."""
        ids = [str(i) for i in prompt_ids or []]
        if not ids:
            return [], []
        try:
            q = self._req("GET", "/queue")
        except (urllib.error.URLError, OSError, ValueError):
            return [], []
        running = {str(it[1]) for it in (q.get("queue_running") or []) if isinstance(it, (list, tuple)) and len(it) > 1}
        pending = {str(it[1]) for it in (q.get("queue_pending") or []) if isinstance(it, (list, tuple)) and len(it) > 1}
        dequeued = [i for i in ids if i in pending]
        interrupted = [i for i in ids if i in running]
        try:
            if dequeued:
                self._req("POST", "/queue", {"delete": dequeued})
            for i in interrupted:
                self._req("POST", "/interrupt", {"prompt_id": i})
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return dequeued, interrupted

    def cancel(self, job: str) -> dict | None:
        """stop a running job: kill the runner (group), delete / interrupt its ComfyUI prompts, mark job.json cancelled. None = unknown job."""
        if not JOB_RE.match(job or ""):
            return None
        st = self._read_state(job)
        if st is None:
            return None
        if st.get("done") or st.get("error") or st.get("cancelled"):
            return dict(job=job, cancelled=bool(st.get("cancelled")), killed=False, interrupted=[], dequeued=[], already=True)
        proc = self.procs.get(job)
        killed = self._kill_runner(proc, st.get("pid"))              # runner first: it must not queue another graph while we clear ComfyUI
        dequeued, interrupted = self._comfy_cancel(st.get("prompt_ids"))
        st = self._read_state(job) or st                             # the runner may have written once more before it died
        st["log"] = (list(st.get("log") or []) + [f"[{time.strftime('%H:%M:%S')}] cancelled by the user (stage {st.get('stage')})"])[-300:]
        st.update(cancelled=True, status="cancelled", stage="cancelled", error=None, done=False, ended=time.time())
        self.write_state(job, st)
        self.procs.pop(job, None)
        return dict(job=job, cancelled=True, killed=killed, interrupted=interrupted, dequeued=dequeued)

    def write_state(self, job: str, st: dict) -> None:
        p = self.state_path(job)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ---------------------------------------------------------------------------------------------------- status
    def status(self, job: str) -> dict | None:
        if not JOB_RE.match(job or ""):
            return None
        st = self._read_state(job)
        if st is None:
            return None
        proc = self.procs.get(job)
        if not st.get("done") and not st.get("error") and not st.get("cancelled"):
            dead = (proc is not None and proc.poll() is not None) or (proc is None and st.get("pid") and not _alive(st["pid"]))
            if dead:                                       # the runner died without writing its end state
                st = self._read_state(job) or st
                if not st.get("done") and not st.get("error"):
                    code = proc.poll() if proc is not None else "?"
                    st["error"] = f"the runner exited (code {code}) during stage {st.get('stage')}; see runner.log"
                    st["stage"] = "error"
        if not st.get("log"):
            st["log"] = self._tail(job)
        st["log"] = list(st.get("log") or [])[-LOG_LINES:]
        base = urllib.parse.quote(f"{OUT_SUB}/{job}", safe="")
        outs = []
        for o in st.get("outputs") or []:
            o = dict(o)
            if "file" in o and "url" not in o:
                o["url"] = f"/view?filename={urllib.parse.quote(o['file'])}&subfolder={base}&type=output"
            is_final = bool(o.get("final")) or o.get("file") == "final.mp4"
            label = o.get("label") or ("Final video" if is_final else "Generated soundtrack" if o.get("kind") == "audio" else o.get("file", ""))
            outs.append({"kind": o.get("kind", "video"), "url": o["url"], "label": label, "final": is_final, **({"file": o["file"]} if "file" in o else {})})
        st["outputs"] = outs
        return st

    def _tail(self, job: str) -> list[str]:
        try:
            return (self.job_dir(job) / "runner.log").read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_LINES:]
        except OSError:
            return []


def register(routes, output_dir=None, input_dir=None, comfy_dir=None, port: int | None = None) -> JobManager:
    """wire the two endpoints into ComfyUI's aiohttp `routes` (called from app_routes.register())."""
    from aiohttp import web
    import folder_paths
    if not port:                    # the port ComfyUI was started with (--port): PromptServer.instance.port is not set yet at registration and a proxy (vast Caddy) may sit on 8188
        try:
            from comfy.cli_args import args as _cli
            port = int(_cli.port)
        except Exception:   # noqa: BLE001
            try:
                from server import PromptServer
                port = getattr(PromptServer.instance, "port", None)
            except Exception:   # noqa: BLE001
                port = None
        port = port or 8188
    out = Path(output_dir or folder_paths.get_output_directory())
    inp = Path(input_dir or folder_paths.get_input_directory())
    mgr = JobManager(out, inp, comfy_dir or out.parent, port)

    frame_mattes = None

    @routes.get("/mograph/own_video/matte_frame")                     # before /{job}. One frame's Alpha-Gen matte for the storyboard: the PNG, 202 while it is being made (tools/app/frame_matte.py)
    async def own_video_matte_frame(request):
        nonlocal frame_mattes
        import asyncio
        from tools.app import frame_matte as FM
        if frame_mattes is None:
            frame_mattes = FM.FrameMattes(out, inp, port)
        q = request.rel_url.query
        try:
            state, got = await asyncio.get_running_loop().run_in_executor(None, frame_mattes.request, q.get("video"), q.get("t"), q.get("end"), q.get("peek") == "1")
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:   # noqa: BLE001  (ffprobe could not read the clip)
            return web.json_response({"error": str(e)[:600]}, status=500)
        if state == "none":
            return web.json_response({"error": "no matte for this frame yet"}, status=404)
        if state == "ready":
            return web.FileResponse(got, headers={"Content-Type": "image/png", "Cache-Control": "no-store"})
        if state == "making":
            return web.json_response({"state": "making"}, status=202, headers={"Cache-Control": "no-store"})
        return web.json_response({"error": got}, status=500)

    @routes.post("/mograph/own_video")
    async def own_video_start(request):
        try:
            body = await request.json()
        except Exception:   # noqa: BLE001
            return web.json_response({"error": "bad json"}, status=400)
        try:
            job = mgr.start(body)
        except BadRequest as e:
            return web.json_response({"error": str(e)}, status=400)
        except Busy as e:
            return web.json_response({"error": "an own-video job is already running", "job": str(e)}, status=409)
        return web.json_response({"job": job})

    @routes.get("/mograph/own_video/{job}")
    async def own_video_status(request):
        st = mgr.status(request.match_info["job"])
        if st is None:
            return web.json_response({"error": "unknown job"}, status=404)
        return web.json_response(st, headers={"Cache-Control": "no-store"})

    @routes.post("/mograph/own_video/{job}/cancel")
    async def own_video_cancel(request):
        import asyncio
        res = await asyncio.get_running_loop().run_in_executor(None, mgr.cancel, request.match_info["job"])      # kill + wait must not block the server's loop
        if res is None:
            return web.json_response({"error": "unknown job"}, status=404)
        return web.json_response(res, headers={"Cache-Control": "no-store"})

    return mgr
