#!/usr/bin/env python
"""piece_runner: one piece JSON -> finished 1080x1920 / 24 fps / H.264 + audio, resumable.

    python tools/fusion/piece_runner.py 15_fusion/pieces/P02.json [--stage all|audio,stills,...]
                                        [--force <stage>] [--procs N] [--dry]
                                        [--exec local|box] [--out-version v003] [--iid ID --host H --ssh-port N --tunnel-port N]
                                        [--box-sim DIR] [--draft | --preview]

GPU work (stills, i2v / control / motiontrack shots, Alpha-Gen mattes, Video Depth Anything) is submitted to the Vast box through
progress_standin.py (live tile in the dashboard, outputs land in 15_fusion/v002/out/ via the sync, never copied by hand). Type passes
and compositing run on the local CPU (multiprocessing). Everything lands in 15_fusion/v002/local/pieces/<id>/ and state.json records
each finished item + its outputs, so a killed run resumes where it stopped (a submitted-but-unfinished render is adopted from out/ when
its files appear). A stage pulls its dependencies in on demand (shots need stills + type for control shots, ...); --stage all runs
audio -> (type || stills) -> shots/alpha/depth pipelined per shot -> compose -> final. --force X forgets stage X (and what depends on
it); old renders stay in out/ (new attempt = new part name, never overwritten).

--exec box: type passes, compose, assemble and the final contact sheet run ON the rented GPU box (code synced once to /workspace/mograph,
shots / mattes / depth read straight from /workspace/ComfyUI/output, only the final mp4 + sheet come back); ComfyUI graphs still go
through progress_standin.py. Type passes start as soon as the audio analysis exists and overlap stills / shots / alpha. Stage and item
timings (wall + GPU execution seconds from ComfyUI /history) go to <piece dir>/timings.json and 16_app/BENCH_<id>.md. --box-sim DIR runs
the "remote" commands as local subprocesses in DIR (tests the whole path without a box; stills / shots / mattes are adopted from v002).
--exec local (default) is the original all-local behaviour. Box connection: --iid / --host / --ssh-port / --tunnel-port, env
MOGRAPH_BOX_IID / _HOST / _SSH_PORT / _TUNNEL_PORT, else the vast dashboard (GET /api/instances/<iid>) or Instance_Data ssh.json.

Piece JSON (all paths relative to the motion-graphic repo root):
{
 "id": "P02", "slug": "after-hours", "fps": 24,                      # fps fixed at 24 (LTX graphs read 24 fps)
 "audio": {"src": "<flac/wav>", "bpm": 122,                          # bpm = hint for the beat tracker
           "analysis": "<existing *_analysis.json>",                 # optional, else 15_fusion/v002/local/audio/<stem>_analysis.json, else computed
           "cut": 11.75,                                             # optional: total length in s. default = last downbeat before the music
           "fade": 0.5},                                             #   ends + `ring` (0.5 s); the last `fade` s fade out
 "type":  { ...tools/fusion/spec.py spec: palette, words[], camera, transitions, ... },   # runner fills duration/fps/bpm/beats/downbeats
 "type_passes": ["matte", "depth", "line", "beauty"],                # optional extra passes (compose/control needs are added automatically)
 "stills": [ {"id": "A", "model": "moss", "prompt": "...", "negative": "...", "lora": "rainywindow", "strength": 1.0,
              "trigger": true, "seed": 11, "size": [1088, 1920]} ],  # straight at 1088x1920 (graphs accept it); `size` may be smaller
 "shots": [ {"id": "s1", "kind": "i2v" | "control" | "motiontrack",
             "first": "A" | "last:s0",                               # still id, or last frame of an earlier shot (continuity)
             "prompt": "...", "frames": 105,                         # 8k+1, i2v <= 385, IC kinds <= 193 (draft control <= 289)
             "seed": 1,
             "size": [1088, 1920],                                   # i2v final size (two-stage: 544x960 + exact x2 latent upscale); draft runs
                                                                     # (--draft / quality draft): i2v single stage at size/2 = 544x960, no stage 2
                                                                     # control/motiontrack: ltx_ic.plan_size -> [1152, 2048] two-stage, draft 576x1024 single stage
             "control": {"pass": "depth" | "line" | "clay" | "matte"},                # control: type pass rendered first -> reference mp4
             "tracks":  {"n": 4, "seed": 1},                         # motiontrack: points sampled on the type, moved by the type flow
             "matte": false,                                         # true -> Alpha-Gen matte (576x1024 ref, PNG sequence)
             "depth": true,                                          # true -> Video Depth Anything frames (540x960 ref)
             "depth_invert": false,                                  # flip polarity if the node gave near = dark
             "flow": true,                                           # --exec box: footage optical flow (RAFT CUDA) -> $flow in compose ops (flow_wake);
                                                                     #   computed from the final footage (masktrack result if any); {"from": "shot"} = original
             "masktrack": {"style": "rise" | "rain" | "unravel" | "pull_apart",    # optional stage after alpha: masked Motion-Track LTX pass (default rise; rain on a person -> rise)
                           "start": "auto" | <piece-time seconds>,        #   (16_app/mask_ltx; box probe 16_masktrack/v001) dissolves the subject
                           "denoise": 1.0, "corridor": false,             #   along vector tracks; start = when the tracks leave the subject (beat-snapped)
                                                                         #   only denoise 1.0 shows an effect; the recomposite runs uint8 in the runner
                           "subject": "the runner", "prompt": "...", "seed": 1, "n_points": 40, "distance": 300, "dilate": 1, "feather": 0.75}}
                                                                         #   needs "matte": true; the result replaces $footage for that shot in compose
 "compose": {"size": [1080, 1920], "crossfade": 0.5,                  # crossfade s (scalar or per boundary); shot k starts crossfade early
             "shots": {"s1": {"ops": [ {"op": "tint_type", "out": "tcol", "args": {...}}, ... ]}},   # compose.py ops, {"op","out","args"}
             "default": {"ops": [...]},                              # used for shots without own entry
             "palette": {"source": "footage", "k": 5},               # optional -> $palette
             "crf": 16},
 "signoff": {"text": "after hours  /  122 bpm", "font": "Space Grotesk", "t0": 10.6, "color": [235, 230, 225], "px": 30, "off": true}   # small last line; "off": true = not drawn (the planner default: no title / BPM caption)
}
Op arguments may reference "$name": the shot ("$s1" or "$footage"), "$s1_matte" / "$s1_depth" (shot-local frames, native size),
"$type_matte|depth|line|beauty|clay" (aligned to piece time automatically), "$signoff", ops outputs ("out"), "$frame_idx", "$palette".
Scalars may be {"anim": [[u, v], ...]} (u = position in the shot, compose.py) or {"curve": "sub|low|mid|high|rms|onset", "lo": a, "hi": b}
(the song's envelope at that piece frame, lo..hi). Shot frames are composed as 1080x1920 (type 1088 wide is cropped 4 px each side;
1152x2048 control shots scale by 0.9375 exactly). Draft: 576x1024 single-stage shots and type scale by 0.9375 to 540x960.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FUSION = ROOT / "15_fusion"
VER = FUSION / "v002"
OUT = VER / "out"
REVIEW = VER / "review"
PIECES_DIR = VER / "local" / "pieces"
STANDIN = Path(os.environ.get("PROGRESS_STANDIN") or Path.home() / "Documents/Dev/comfyui-projects/tools/progress_standin.py")
# the box: defaults only (the 2026-10-25 box); --iid/--host/--ssh-port/--tunnel-port, MOGRAPH_BOX_* env or the dashboard override them
BOX = dict(iid=os.environ.get("MOGRAPH_BOX_IID", "57643895"), host=os.environ.get("MOGRAPH_BOX_HOST", "<box-ip>"),
           ssh_port=int(os.environ.get("MOGRAPH_BOX_SSH_PORT", 40095)), tunnel_port=int(os.environ.get("MOGRAPH_BOX_TUNNEL_PORT", 9000)),
           input_dir="/workspace/ComfyUI/input", idea="COMPETITIONS/motion-graphic/fusion/v002",
           comfy="/workspace/ComfyUI", root="/workspace/mograph")
DASH = "http://127.0.0.1:8000"          # vast dashboard backend
INSTANCE_DATA = Path(os.environ.get("VAST_INSTANCE_DATA") or Path.home() / "Documents/Dev/vastai-api-manager/Instance_Data")
PROJ_PY = Path.home() / ".claude" / "skills" / "project" / "proj.py"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
BOX_SSH_PERSIST = os.environ.get("MOGRAPH_BOX_SSH", "persist") != "plain"       # plain = one ssh / scp process per call (the old way)
FPS = 24
TYPE_SIZE = (1088, 1920)
I2V_MAX_FRAMES = 385            # plain i2v two-stage proven to 385 f (16_app/longshot/RESULTS.md); draft single stage is lighter
IC_MAX_FRAMES = 193             # control / motiontrack two-stage (ltx_ic.KINDS max_frames)
PREVIEW_TYPE = (384, 704)       # preview tier: type layers = the IC shot size (spec.py: multiples of 64); compose 360x640 (plan_rules.sizes_for)
IC_DRAFT_MAX_FRAMES = 289       # control single-stage draft 576x1024: 289 f ran on the box (51 s, 16_app/longshot/RESULTS.md); two-stage IC above 193 is untested
STAGES = ["audio", "stills", "type", "shots", "flow", "alpha", "depth", "masktrack", "compose", "final"]
DOWNSTREAM = {                        # --force X forgets X and these
    "audio": STAGES[1:], "stills": ["shots", "flow", "alpha", "depth", "masktrack", "compose", "final"],
    "type": ["shots", "flow", "alpha", "depth", "masktrack", "compose", "final"],
    "shots": ["flow", "alpha", "depth", "masktrack", "compose", "final"],
    "flow": ["compose", "final"], "alpha": ["masktrack", "flow", "compose", "final"], "depth": ["compose", "final"],
    "masktrack": ["flow", "compose", "final"], "compose": ["final"], "final": [],
}
MASK_STYLE_TEXT = {      # appended to the shot prompt for the masked Motion-Track pass ({s} = subject)
    "rain": "{s} dissolves downward into glowing vertical threads of light that fall like rain",
    "rise": "{s} dissolves upward into glowing vertical threads of light that stream up and away, accelerating and thinning out toward the top",
    "unravel": "{s} unravels into luminous threads of light that spiral around the body and dissolve into the air",
    "pull_apart": "{s} splits down the middle and pulls apart into long fine threads of light that stream away to the left and to the "
                  "right in parallel lines and dissolve into the air",           # box probe: reads as locomotion (figure walks away); not a default
}

PERSON_RE = re.compile(r"\b(person|man|woman|runner|runners|dancer|dancers|skater|figure|girl|boy|swimmer|people|child|rider|couple)\b")


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def snap8(n):
    return max(9, ((int(n) - 1) // 8) * 8 + 1)


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))[:200]} -> rc {r.returncode}\n{(r.stderr or r.stdout)[-800:]}")
    return r


def stage_of(key):
    k = key.split(":")[0]
    return {"audio": "audio", "still": "stills", "type": "type", "shot": "shots", "alpha": "alpha", "depth": "depth",
            "masktrack": "masktrack", "flow": "flow", "compose": "compose", "assemble": "compose", "final": "final"}[k]


def set_version(ver, root=None):
    """point the module-level output locations at 15_fusion/<ver> (or at a --box-sim directory `root`)."""
    global VER, OUT, REVIEW, PIECES_DIR
    VER = FUSION / ver
    OUT = VER / "out"
    REVIEW = VER / "review"
    PIECES_DIR = VER / "local" / "pieces"
    if root is not None:
        root = Path(root)
        OUT, REVIEW, PIECES_DIR = root / "out", root / "review", root / "pieces"
    BOX["idea"] = f"COMPETITIONS/motion-graphic/fusion/{ver}"


# =================================================================================================== box connection / remote
def parse_probe(text):
    """ffprobe -of json (stream=width,height,nb_frames,r_frame_rate,duration:format=duration) -> (w, h, nframes, fps); as compose._probe_video."""
    j = json.loads(text)
    s = j["streams"][0]
    num, den = (s.get("r_frame_rate") or "24/1").split("/")
    fps = float(num) / float(den or 1)
    n = s.get("nb_frames")
    if n in (None, "N/A", "0"):
        dur = s.get("duration")
        dur = float(dur) if dur not in (None, "N/A") else float(j["format"]["duration"])
        n = int(round(dur * fps))
    return int(s["width"]), int(s["height"]), int(n), fps


def _http_json(url, timeout=8):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_box(iid=None, host=None, ssh_port=None, tunnel_port=None, need_box=False):
    """fill BOX from, in order: the arguments, MOGRAPH_BOX_* env (already in BOX), the vast dashboard (GET /api/instances/<iid>:
    ssh_host / ssh_port / tunnel_ports / started_running_at), Instance_Data/machine_*/<iid>/connection/ssh.json. With need_box and no
    iid, the dashboard's only running instance is taken. -> dict of what was found (also stored as BOX['info'])."""
    iid = iid or os.environ.get("MOGRAPH_BOX_IID")
    info = {}
    if not iid and need_box:
        try:
            run_ = [i for i in _http_json(f"{DASH}/api/instances/") if i.get("state") == "running" and i.get("ssh_host")]
        except Exception as e:      # noqa: BLE001
            raise SystemExit(f"--iid is required (dashboard not reachable: {e})")
        if len(run_) != 1:
            raise SystemExit("--iid is required: the dashboard lists %d running instance(s): %s" % (
                len(run_), ", ".join(str(i["instance_id"]) for i in run_)))
        iid = str(run_[0]["instance_id"])
    if iid:
        BOX["iid"] = str(iid)
    explicit = host or os.environ.get("MOGRAPH_BOX_HOST")
    if iid and not explicit:
        try:
            d = _http_json(f"{DASH}/api/instances/{iid}")
            if d.get("ssh_host") and d.get("ssh_port"):
                BOX["host"], BOX["ssh_port"] = d["ssh_host"], int(d["ssh_port"])
            if d.get("tunnel_ports") and tunnel_port is None and not os.environ.get("MOGRAPH_BOX_TUNNEL_PORT"):
                BOX["tunnel_port"] = int(d["tunnel_ports"][0])
            info = {k: d.get(k) for k in ("gpu_name", "price_per_hour", "started_running_at", "created_at", "state", "ssh_ready")}
        except Exception:       # noqa: BLE001
            for cj in INSTANCE_DATA.glob(f"machine_*/{iid}/connection/ssh.json"):
                try:
                    c = json.loads(cj.read_text())
                    BOX["host"], BOX["ssh_port"] = c["ipaddr"], int(c["port"])
                    break
                except Exception:       # noqa: BLE001
                    continue
    if host:
        BOX["host"] = host
    if ssh_port:
        BOX["ssh_port"] = int(ssh_port)
    if tunnel_port:
        BOX["tunnel_port"] = int(tunnel_port)
    BOX["info"] = info
    return info


class Remote:
    """The box as the runner sees it: ssh / scp to the real machine, or (sim) local subprocesses in a --box-sim directory where
    /workspace/ComfyUI and /workspace/mograph become <dir>/ComfyUI and <dir>/mograph. dry = log the command, run nothing."""

    def __init__(self, piece_id, sim=None, dry=False, log=print, backend="torch", onbox=False):
        import shlex
        self.q = shlex.quote
        self.id, self.dry, self.log, self.backend = piece_id, dry, log, backend
        self.onbox = bool(onbox)          # runner is ON the box: "remote" commands are local subprocesses on the real /workspace paths
        self.sim = Path(sim).resolve() if sim else (Path(BOX["root"]).parent if self.onbox else None)
        if self.onbox:
            self.root, self.comfy = BOX["root"], BOX["comfy"]
            self.py = Path(sys.executable).as_posix()
        elif self.sim:
            base = self.sim.as_posix()
            self.root, self.comfy = f"{base}/mograph", f"{base}/ComfyUI"
            self.py = Path(sys.executable).as_posix()
        else:
            self.root, self.comfy = BOX["root"], BOX["comfy"]
            self.py = "python3"
        self.input, self.output = f"{self.comfy}/input", f"{self.comfy}/output"
        self.work, self.out = f"{self.root}/work/{piece_id}", f"{self.root}/out/{piece_id}"
        self.info = dict(cores=12, nvenc=False, missing=[])
        self.ready = threading.Event()
        self.prep_error = None
        self.prep_s = None
        self.t_first = None
        self.cmd_log = []
        self.pool = None                  # persistent ssh sessions (tools/fusion/box_ssh.py); None in sim / dry / plain mode
        self._pool_lock = threading.Lock()
        self.n_calls = dict(session=0, plain=0)

    # ---- persistent ssh (one handshake for the whole run instead of ~4 s per ssh / scp)
    def _sessions(self):
        if self.sim or self.dry or not BOX_SSH_PERSIST:
            return None
        with self._pool_lock:
            if self.pool is None:
                from tools.fusion import box_ssh
                import atexit
                argv = ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6",
                        "-T", "-p", str(BOX["ssh_port"]), f"root@{BOX['host']}", "bash -s"]
                self.pool = box_ssh.SessionPool(argv)
                atexit.register(self.pool.close)
            return None if self.pool.broken else self.pool

    def _sess_call(self, cmd, timeout, data=None):
        """-> (rc, out bytes, err bytes) through a persistent session, or None when sessions are unusable (caller falls back to plain ssh)."""
        pool = self._sessions()
        if pool is None:
            return None
        from tools.fusion import box_ssh
        try:
            r = pool.call(cmd, timeout=timeout, data=data)
            self.n_calls["session"] += 1
            return r
        except box_ssh.BoxSSHError as e:
            pool.broken = True
            self.log(f"  WARNING persistent ssh unusable ({e}); falling back to one ssh per call")
            return None

    def close(self):
        if self.pool is not None:
            self.pool.close()

    # ---- plumbing
    def _bash(self):
        for c in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
            if Path(c).exists():
                return c
        return shutil.which("bash") or "bash"

    def _note(self, what, cmd):
        if self.t_first is None and not self.dry:
            self.t_first = time.time()
        self.cmd_log.append(f"{what} {cmd}")
        self.log(f"  [{'dry ' if self.dry else ''}{'onbox' if self.onbox else 'sim' if self.sim else 'box'}] {what} {cmd if self.dry else cmd[:220]}")

    def _ssh(self):
        return ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=20", "-p", str(BOX["ssh_port"]), f"root@{BOX['host']}"]

    def sh(self, cmd, timeout=120, check=True):
        self._note("sh", cmd)
        if self.dry:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        sr = None if self.sim else self._sess_call(cmd, timeout)
        if sr is not None:
            r = subprocess.CompletedProcess(cmd, sr[0], sr[1].decode("utf-8", "replace"), sr[2].decode("utf-8", "replace"))
            if check and r.returncode != 0:
                raise RuntimeError(f"remote command failed rc {r.returncode}: {cmd[:200]}\n{(r.stderr or r.stdout)[-800:]}")
            return r
        for attempt in range(4):
            if self.sim:
                self.sim.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([self._bash(), "-c", cmd], capture_output=True, text=True, timeout=timeout, cwd=str(self.sim),
                                   encoding="utf-8", errors="replace")
            else:
                r = subprocess.run(self._ssh() + [cmd], capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
            if r.returncode != 255 or self.sim or attempt == 3:       # 255 = ssh itself failed: retry
                break
            time.sleep(5)
        if check and r.returncode != 0:
            raise RuntimeError(f"remote command failed rc {r.returncode}: {cmd[:200]}\n{(r.stderr or r.stdout)[-800:]}")
        return r

    def put_many(self, pairs, timeout=900):
        """pairs: [(local path | bytes, absolute remote path)] -> ONE tar stream through the persistent session (one round trip, no per-file
        ssh mkdir). Falls back to put() per file when sessions are unusable (sim / plain ssh)."""
        pairs = list(pairs)
        for src, remote in pairs:
            self._note("put", f"{src if isinstance(src, (str, Path)) else f'<{len(src)} bytes>'} -> {remote}")
        if self.dry or not pairs:
            return
        if not self.sim and self._sessions() is not None:
            from tools.fusion import box_ssh
            blob = box_ssh.tar_bytes([(r.lstrip("/"), src if isinstance(src, (bytes, bytearray)) else str(src)) for src, r in pairs])
            dirs = sorted({str(PurePosixPath(r).parent) for _, r in pairs})
            cmd = f"mkdir -p {' '.join(self.q(d) for d in dirs)} && head -c {len(blob)} | tar xzf - -C /"
            sr = self._sess_call(cmd, timeout, data=blob)
            if sr is not None:
                if sr[0] != 0:
                    raise RuntimeError(f"put_many failed rc {sr[0]}: {sr[2].decode('utf-8', 'replace')[-400:]}")
                return
        for src, remote in pairs:
            if isinstance(src, (bytes, bytearray)):
                import tempfile
                with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
                    f.write(src)
                try:
                    self._put_one(f.name, remote, timeout)
                finally:
                    os.unlink(f.name)
            else:
                self._put_one(src, remote, timeout)

    def put(self, local, remote, timeout=900):
        self.put_many([(local, remote)], timeout)

    def _put_one(self, local, remote, timeout=900):
        if self.dry:
            return
        if self.sim:
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, remote)
        else:
            self.sh(f"mkdir -p {self.q(str(PurePosixPath(remote).parent))}")
            run(["scp", *SSH_OPTS, "-P", str(BOX["ssh_port"]), str(local), f"root@{BOX['host']}:{remote}"], timeout=timeout)

    def get_many(self, pairs, timeout=900):
        """pairs: [(absolute remote path, local path)] -> ONE tar stream back through the persistent session."""
        pairs = list(pairs)
        for remote, local in pairs:
            self._note("get", f"{remote} -> {local}")
        if self.dry or not pairs:
            return
        if not self.sim and self._sessions() is not None:
            from tools.fusion import box_ssh
            sr = self._sess_call(f"tar cf - -C / {' '.join(self.q(r.lstrip('/')) for r, _ in pairs)}", timeout)
            if sr is not None:
                if sr[0] != 0:
                    raise RuntimeError(f"get_many failed rc {sr[0]}: {sr[2].decode('utf-8', 'replace')[-400:]}")
                box_ssh.untar_to(sr[1], [(r.lstrip("/"), str(l)) for r, l in pairs])
                return
        for remote, local in pairs:
            self._get_one(remote, local, timeout)

    def get(self, remote, local, timeout=900):
        self.get_many([(remote, local)], timeout)

    def _get_one(self, remote, local, timeout=900):
        if self.dry:
            return
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if self.sim:
            shutil.copyfile(remote, local)
        else:
            run(["scp", *SSH_OPTS, "-P", str(BOX["ssh_port"]), f"root@{BOX['host']}:{remote}", str(local)], timeout=timeout)

    def put_json(self, obj, remote):
        import tempfile
        if self.dry:
            self._note("put", f"<json {len(json.dumps(obj))} bytes> -> {remote}")
            return
        if not self.sim and self._sessions() is not None:
            return self.put_many([(json.dumps(obj).encode("utf-8"), remote)])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(obj, f)
        try:
            self.put(f.name, remote)
        finally:
            os.unlink(f.name)

    def py_cmd(self, sub, args="", env=""):
        return f"cd {self.root} && {env}{self.py} -m tools.fusion.box_stages {sub} {args}".rstrip()

    def bg(self, name, cmd):
        """run `cmd` detached (nohup) on the box; it must accept --marker (box_stages does). Poll with wait(name)."""
        marker, logf = f"{self.work}/{name}.done.json", f"{self.work}/{name}.log"
        full = f"mkdir -p {self.work} && rm -f {marker} && (nohup bash -c {self.q(cmd + ' --marker ' + marker)} > {logf} 2>&1 < /dev/null &) ; echo started"
        self.sh(full)
        return marker

    def wait(self, name, timeout=7200, poll=4.0, progress=None):
        """poll the done-marker of a bg() command. -> its result dict; raises with the remote error."""
        marker, logf = f"{self.work}/{name}.done.json", f"{self.work}/{name}.log"
        if self.dry:
            self._note("wait", f"{marker}  (poll every {poll:g}s, log {logf})")
            return {}
        t0, last = time.time(), 0.0
        # one ssh round trip waits up to 20 s ON the box (marker checked every 0.5 s), so the finish is noticed within 0.5 s and a long
        # stage costs ~1 call per 20 s instead of one 4 s ssh handshake per poll
        while time.time() - t0 < timeout:
            r = self.sh(f"for i in $(seq 1 40); do [ -f {marker} ] && break; sleep 0.5; done; "
                        f"cat {marker} 2>/dev/null || (echo __RUNNING__; tail -n 2 {logf} 2>/dev/null)", check=False, timeout=90)
            txt = (r.stdout or "").strip()
            if txt and not txt.startswith("__RUNNING__"):
                d = json.loads(txt.splitlines()[-1])
                if not d.get("ok"):
                    raise RuntimeError(f"{name} failed on the box: {d.get('error')}  (log {logf})")
                return dict(d["result"], remote_wall_s=d.get("wall_s"))
            if progress and time.time() - last > 45:
                last = time.time()
                progress("  " + " | ".join(txt.splitlines()[1:])[:200])
            time.sleep(0.2)          # the box-side loop above already waited
        raise TimeoutError(f"{name} did not finish in {timeout}s (marker {marker})")

    # ---- output lookup
    def find_out(self, idea, part, exts, expect_n=1, timeout=180):
        """ComfyUI output files of a submitted part, straight on the box (the dashboard sync is not waited for). -> sorted PurePosixPaths."""
        pat = rf".*/[^/]*_{re.escape(part)}(_[a-z]+)?_[0-9]+_?\.({'|'.join(exts)})"
        base = f"{self.output}/projects/{idea}"
        if self.dry:
            self._note("find", f"{base} -regex {pat}  (expect {expect_n})")
            return [PurePosixPath(f"{base}/<stem>_{part}_{k + 1:05d}_.{exts[0]}") for k in range(max(expect_n, 1))]
        t0 = time.time()
        files = []
        while True:
            r = self.sh(f"find {self.q(base)} -type f -regextype posix-extended -regex {self.q(pat)} 2>/dev/null", check=False)
            files = sorted((PurePosixPath(x.strip()) for x in (r.stdout or "").splitlines() if x.strip()), key=lambda p: natural(p.name))
            if len(files) >= expect_n or time.time() - t0 > timeout:
                return files
            time.sleep(3)

    # ---- one-time code sync + dependencies
    SYNC_DIRS = ["tools", "mograph", "nodes_fusion", "15_fusion/pieces", "workflows/api", "11_mathtype/engine/dimtype"]   # dimtype = ol.py (cam3d)
    SYNC_FILES = ["requirements.txt", "requirements-mathtype.txt", "requirements-music.txt"]

    def _sync_files(self):
        out = []
        for d in self.SYNC_DIRS:
            for dp, dn, fn in os.walk(ROOT / d):
                dn[:] = [x for x in dn if x not in ("__pycache__", ".git") and not x.startswith("tmpclaude")]
                for f in fn:
                    if not f.endswith((".pyc", ".pyo")) and not f.startswith("tmpclaude"):
                        out.append(Path(dp) / f)
        out += [ROOT / f for f in self.SYNC_FILES if (ROOT / f).exists()]
        return sorted(out)

    def sync_code(self):
        """delta sync: sha1 of every synced file vs the manifest the last sync left on the box (<root>/.sync_manifest.json); only changed
        files go up (one tar.gz stream), files deleted locally are deleted there. No manifest (first sync, old box) = everything."""
        import hashlib
        import io
        import tarfile
        import tempfile
        files = self._sync_files()
        local = {f.relative_to(ROOT).as_posix(): hashlib.sha1(f.read_bytes()).hexdigest() for f in files}
        digest = hashlib.sha1(json.dumps(local, sort_keys=True).encode()).hexdigest()[:16]
        have = {}
        if not self.dry:
            r = self.sh(f"cat {self.root}/.sync_manifest.json 2>/dev/null", check=False)
            try:
                have = json.loads(r.stdout) if r.stdout.strip() else {}
            except ValueError:
                have = {}
        changed = sorted(k for k, v in local.items() if have.get(k) != v)
        removed = sorted(k for k in have if k not in local)
        if not changed and not removed and not self.dry:
            self.log(f"  code on the box is current ({digest}, {len(files)} files)")
            return digest
        tgz = f"{self.root}/.code.tgz"
        if self.dry:
            self._note("tar", f"{len(files)} files from {', '.join(self.SYNC_DIRS + self.SYNC_FILES)} -> {tgz}; extract to {self.root}")
            return digest
        with tempfile.TemporaryDirectory() as td:
            loc = Path(td) / "code.tgz"
            with tarfile.open(loc, "w:gz", compresslevel=6) as t:
                for k in changed:
                    t.add(ROOT / k, arcname=k)
                for name, data in ((".sync_manifest.json", json.dumps(local).encode()), (".sync_hash", (digest + "\n").encode())):
                    ti = tarfile.TarInfo(name)
                    ti.size, ti.mtime = len(data), time.time()
                    t.addfile(ti, io.BytesIO(data))
            self.log(f"  code delta: {len(changed)} of {len(files)} files changed, {len(removed)} removed, {loc.stat().st_size // 1024} KB")
            self.put(loc, tgz)
        rm = (" && rm -f " + " ".join(self.q(k) for k in removed)) if removed else ""
        self.sh(f"mkdir -p {self.root} && cd {self.root} && tar xzf .code.tgz && rm -f .code.tgz{rm}", timeout=300)
        return digest

    def prepare(self):
        """mkdirs, python detection, one-time code sync, pip install of what is missing, probe (cores, NVENC). Sets self.ready."""
        t0 = time.time()
        try:
            self.sh(f"mkdir -p {self.root} {self.work} {self.out} {self.input} && echo ok")
            if not self.sim:
                r = self.sh("for p in /venv/main/bin/python python3 python; do command -v $p >/dev/null 2>&1 && { command -v $p; break; }; done", check=False)
                if r.stdout.strip() and not self.dry:
                    self.py = r.stdout.strip().splitlines()[-1]
            if not self.onbox:                      # on the box the launcher already synced the code (box_run.py)
                self.sync_code()
            r = self.sh(self.py_cmd("probe"), timeout=180)
            if not self.dry:
                d = json.loads(r.stdout.strip().splitlines()[-1])
                if not d.get("ok"):
                    raise RuntimeError(f"probe failed: {d.get('error')}")
                self.info = d["result"]
                if self.info.get("missing"):
                    names = " ".join(self.info["missing"])
                    self.log(f"  pip install {names}")
                    self.sh(f"{self.py} -m pip install -q {names}", timeout=900)
                self.log(f"  box: {self.info.get('cores')} cores, NVENC {self.info.get('nvenc')}, torch {self.info.get('torch')} "
                         f"cuda {self.info.get('cuda')} {self.info.get('gpu') or ''}")
            else:
                self._note("sh", f"{self.py} -m pip install -q <whatever probe reports missing: opencv-python-headless scipy ...>")
        except BaseException as e:      # noqa: BLE001
            self.prep_error = e
        finally:
            self.prep_s = round(time.time() - t0, 1)
            self.ready.set()

    def wait_ready(self):
        self.ready.wait()
        if self.prep_error:
            raise RuntimeError(f"box preparation failed: {self.prep_error}")


# =================================================================================================== the piece
def check_liked_fonts(j: dict) -> None:
    """HARD rule: a piece spec may only name fonts in mograph/fonts/liked.json (type words, and the signoff unless it is off).
    Raises mograph.fonts_liked.FontNotLiked at load time, before any GPU / render work."""
    from mograph.fonts_liked import require_liked
    for k, w in enumerate((j.get("type") or {}).get("words") or []):
        if w.get("font") is not None:
            require_liked(w["font"])
    so = j.get("signoff")
    if so and not so.get("off") and so.get("font") is not None:
        require_liked(so["font"])


class Piece:
    def __init__(self, json_path, procs=None, exec_mode="local", sim=None, dry=False, draft=False, backend=None, preview=False,
                 onbox=False, relay=None):
        if draft and preview:
            raise ValueError("--draft and --preview are mutually exclusive")
        if onbox:
            exec_mode = "box"
        self.baton, self._have_baton, self._baton_mx = None, False, threading.Lock()      # batch: see run_batch
        self.onbox, self.relay = bool(onbox), relay        # onbox: this runner lives ON the box; GPU submits go through `relay` (box_run.py)
        self.exec_mode, self.dry, self.draft, self.preview = exec_mode, dry, draft, preview
        self.jpath = Path(json_path).resolve()
        self.j = json.loads(self.jpath.read_text(encoding="utf-8"))
        check_liked_fonts(self.j)
        self.id = self.j["id"]
        self.fps = FPS
        self.dir = PIECES_DIR / self.id
        if not dry:
            self.dir.mkdir(parents=True, exist_ok=True)
            for d in ("graphs", "uploads", "type", "compose"):
                (self.dir / d).mkdir(exist_ok=True)
        self.procs_arg = procs
        self.procs = procs or max(1, min(8, (os.cpu_count() or 2) - 1))
        self.type_size = (576, 1024) if draft else TYPE_SIZE        # draft: 576x1024 type layers (= the draft shot size), 540x960 compose (box mode)
        if preview:                                                 # preview: 384x704 type layers (= the IC shot size), 360x640 compose
            from tools.fusion import plan_rules
            self.type_size = tuple(plan_rules.orient(PREVIEW_TYPE, plan_rules.format_of(self.j)))
        self.sim = Path(sim).resolve() if sim else None
        self._push_lock = threading.Lock()
        self._flow_lock = threading.Lock()
        self._pushed = False
        self.lock = threading.RLock()
        self.futs = {}
        self.tls = threading.local()
        self.sp = self.dir / "state.json"
        self.state = json.loads(self.sp.read_text(encoding="utf-8")) if self.sp.exists() else {}
        self.state.setdefault("piece", self.id)
        for k in ("items", "attempts", "pending", "stages"):
            self.state.setdefault(k, {})
        if self.onbox:
            self.state["items"].pop("audio", None)      # its record holds the PC's wav / analysis paths; re-cutting the audio takes < 1 s
        self.shots = {s["id"]: s for s in self.j.get("shots", [])}
        self.stills = {s["id"]: s for s in self.j.get("stills", [])}
        self.logf = None if dry else open(self.dir / "run.log", "a", encoding="utf-8")
        self.rx = (Remote(self.id, self.sim, dry, self.log, backend or ("auto" if self.sim else "torch"), onbox=self.onbox)
                   if exec_mode == "box" else None)

    # ---------------------------------------------------------------- log / state
    def log(self, msg):
        line = f"[{self.id} {ts()}] {msg}"
        with self.lock:
            print(line, flush=True)
            if self.logf:
                self.logf.write(line + "\n")
                self.logf.flush()

    def save(self):
        if self.dry:
            return
        with self.lock:
            tmp = self.sp.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")
            for k in range(8):                    # Windows: replace can hit a transient "Access is denied" (indexer / virus scanner holds the file)
                try:
                    os.replace(tmp, self.sp)
                    break
                except PermissionError:
                    if k == 7:
                        raise
                    time.sleep(0.25)

    def forget(self, stage):
        with self.lock:
            for st in [stage] + DOWNSTREAM[stage]:
                for k in [k for k in self.state["items"] if stage_of(k) == st]:
                    del self.state["items"][k]
                self.state["stages"].pop(st, None)
            self.save()

    def _valid(self, rec):
        if self.rx is not None:                       # box mode: the files live on the box, trust the record (ok=False = retry next run)
            return rec.get("out", {}).get("ok", True)
        return all(Path(p).exists() for p in rec.get("out", {}).get("paths", []))

    def item(self, key, fn):
        """memoised, resumable unit of work: state -> done record, else run `fn` once in its own thread (dependencies call item() too)."""
        with self.lock:
            rec = self.state["items"].get(key)
            if rec and self._valid(rec):
                return rec["out"]
            f = self.futs.get(key)
            start = f is None
            if start:
                f = self.futs[key] = Future()
        if start:
            def work():
                self.tls.gpu = 0.0
                self.tls.exec = 0.0
                t0 = time.time()
                try:
                    self.log(f"start {key}")
                    out = fn()
                    rec = dict(out=out, wall_s=round(time.time() - t0, 1), gpu_s=round(self.tls.gpu, 1),
                               gpu_exec_s=round(self.tls.exec, 1), t_start=round(t0, 2), t_end=round(time.time(), 2),
                               at=datetime.datetime.now().isoformat(timespec="seconds"))
                    with self.lock:
                        self.state["items"][key] = rec
                        self.state["pending"].pop(key, None)
                        self.save()
                    self.log(f"done {key}  wall {rec['wall_s']}s  gpu {rec['gpu_s']}s")
                    f.set_result(out)
                except BaseException as e:      # noqa: BLE001
                    self.log(f"FAILED {key}: {e}")
                    f.set_exception(e)
            threading.Thread(target=work, daemon=True, name=key).start()
        return f.result()

    # ---------------------------------------------------------------- box
    def ssh(self, remote_cmd, timeout=60):
        return run(["ssh", *SSH_OPTS, "-o", "ConnectTimeout=20", "-p", str(BOX["ssh_port"]),
                    f"root@{BOX['host']}", remote_cmd], timeout=timeout)

    def _take_baton(self):
        """batch mode: the first GPU submit of a piece takes the ComfyUI baton and keeps it until its on-box flow / compose are done."""
        if self.baton is None:
            return
        with self._baton_mx:
            if not self._have_baton:
                if self.baton.locked():
                    self.log("waiting for the previous piece to finish its on-box GPU jobs (ComfyUI baton)")
                self.baton.acquire()
                self._have_baton = True
                self.log("ComfyUI baton taken")

    def _drop_baton(self):
        with self._baton_mx:
            if self._have_baton:
                self._have_baton = False
                self.baton.release()
                self.log("ComfyUI baton released")

    def keepalive(self):
        if self.sim or self.dry:
            return
        if self.onbox:
            Path("/workspace/KEEPALIVE").touch()
            return
        try:
            self.ssh("touch /workspace/KEEPALIVE")
        except Exception as e:      # noqa: BLE001
            self.log(f"WARNING keepalive failed: {e}")

    def upload(self, local, name=None):
        name = name or Path(local).name
        if not name.startswith(self.id + "_"):
            name = f"{self.id}_{name}"
        tmp = self.dir / "uploads" / name
        if Path(local).resolve() != tmp.resolve():
            shutil.copyfile(local, tmp)
        run(["scp", *SSH_OPTS, "-P", str(BOX["ssh_port"]), str(tmp),
             f"root@{BOX['host']}:{BOX['input_dir']}/{name}"], timeout=600)
        self.log(f"uploaded {name} ({tmp.stat().st_size // 1024} KB)")
        return name

    def _find_out(self, part, exts):
        if self.rx is not None:
            return self.rx.find_out(BOX["idea"], part, exts, 1, timeout=0)
        rx = re.compile(rf".*_{re.escape(part)}(?:_[a-z]+)?_(\d+)_?\.({'|'.join(exts)})$")
        files = [p for p in OUT.iterdir() if rx.match(p.name)]
        return sorted(files, key=lambda p: natural(p.name))

    def _wait_files(self, part, exts, expect_n=1, timeout=1500):
        if self.rx is not None:
            files = self.rx.find_out(BOX["idea"], part, exts, expect_n, timeout=180)
            if not files:
                raise TimeoutError(f"outputs of {part} not found on the box under {self.rx.output}/projects/{BOX['idea']}")
            return files
        t0 = time.time()
        last, since = None, time.time()
        while time.time() - t0 < timeout:
            files = self._find_out(part, exts)
            sig = [(p.name, p.stat().st_size) for p in files]
            if sig != last:
                last, since = sig, time.time()
            settled = time.time() - since > 4
            if files and settled and (len(files) >= expect_n or time.time() - since > 90):
                return files
            time.sleep(3)
        raise TimeoutError(f"outputs of {part} did not appear in {OUT} within {timeout}s")

    def fetch_seq(self, part, expect_n):
        """PNG sequences sync to out/ at ~5 s/file: tar them on the box and scp the tar down instead. -> sorted local Paths."""
        if self.rx is not None:
            files = self.rx.find_out(BOX["idea"], part, ("png",), expect_n, timeout=180)
            if len(files) < expect_n:
                self.log(f"WARNING {part}: {len(files)} of {expect_n} png found on the box")
            return files
        import tarfile
        rd = f"/workspace/ComfyUI/output/projects/{BOX['idea']}"
        d = self.dir / "seq" / part
        have = sorted(d.glob("*.png"), key=lambda p: natural(p.name)) if d.exists() else []
        if len(have) >= expect_n:
            return have
        d.mkdir(parents=True, exist_ok=True)
        pat = rf"_{part}(_[a-z]+)?_[0-9]+_?\.png$"
        for attempt in range(10):
            r = self.ssh(f"cd {rd} && ls | grep -E '{pat}' > /tmp/{part}.lst; tar cf /tmp/{part}.tar -T /tmp/{part}.lst; wc -l < /tmp/{part}.lst",
                         timeout=300)
            n = int(r.stdout.strip().splitlines()[-1])
            if n >= expect_n:
                break
            time.sleep(6)
        tar = self.dir / "seq" / f"{part}.tar"
        run(["scp", *SSH_OPTS, "-P", str(BOX["ssh_port"]), f"root@{BOX['host']}:/tmp/{part}.tar", str(tar)], timeout=900)
        with tarfile.open(tar) as t:
            t.extractall(d, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(d)
        tar.unlink()
        files = sorted(d.glob("*.png"), key=lambda p: natural(p.name))
        self.log(f"  fetched {len(files)} png of {part} by tar")
        return files

    def submit(self, graph, slug, why, exts=("mp4",), expect_n=1, key=None, seq=False):
        """queue a graph through progress_standin (blocks). -> list of local output Paths (synced into out/)."""
        with self.lock:
            n = self.state["attempts"].get(slug, 0) + 1
            part = f"{self.id}_{slug}_{n}"
            pend = self.state["pending"].get(key) if key else None
        if pend and self.rx is not None:                  # box: adopt straight from the box's output dir
            files = self.rx.find_out(BOX["idea"], pend["part"], ("png",) if seq else exts, expect_n, timeout=0)
            files = files if len(files) >= expect_n else []
            if files:
                self.log(f"adopted {len(files)} file(s) of {pend['part']}")
                return files
        elif pend:                                        # adopt a render a killed run left behind
            files = self.fetch_seq(pend["part"], expect_n) if seq and (self.dir / "seq" / pend["part"]).exists()                 else ([] if seq else self._find_out(pend["part"], exts))
            if files:
                self.log(f"adopted {len(files)} file(s) of {pend['part']}")
                return files
        with self.lock:
            self.state["attempts"][slug] = n
            if key:
                self.state["pending"][key] = dict(part=part, at=ts())
            self.save()
        gpath = self.dir / "graphs" / f"{part}.json"
        pidf = self.dir / "graphs" / f"{part}.prompt_id"
        cmd = [sys.executable, str(STANDIN), "submit", str(gpath), "--iid", BOX["iid"], "--port", str(BOX["tunnel_port"]),
               "--idea", BOX["idea"], "--part", part, "--why", why, "--linger", "5", "--prompt-id-file", str(pidf)]
        if self.dry:
            self.log(f"[dry] GPU submit {part}: {' '.join(map(str, cmd))}")
            return self.rx.find_out(BOX["idea"], part, ("png",) if seq else exts, expect_n)
        gpath.write_text(json.dumps(graph, indent=1), encoding="utf-8")
        if self.sim:                                      # --box-sim: no GPU; v002's render of this item stands in for the new one
            self.log(f"sim submit {part}: {why}")
            self._sim_submit(slug, part, exts, seq)
            return self.rx.find_out(BOX["idea"], part, ("png",) if seq else exts, expect_n, timeout=5)
        self.keepalive()
        self.log(f"submit {part}: {why}")
        t0 = time.time()
        if self.onbox:
            self._take_baton()
            t0 = time.time()
            ok, tail = self._relay_submit(part, why, graph, pidf)
            el = time.time() - t0
            self.tls.gpu = getattr(self.tls, "gpu", 0.0) + el
            if not ok:
                raise RuntimeError(f"submit {part} failed:\n" + "\n".join(tail))
            self.tls.exec = getattr(self.tls, "exec", 0.0) + self._gpu_exec(key or slug, pidf)
            files = self.fetch_seq(part, expect_n) if seq else self._wait_files(part, exts, expect_n)
            self.log(f"  {part}: {len(files)} file(s) in out/ after {time.time() - t0:.0f}s")
            return files
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        ok, tail = None, []
        for line in p.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-12:]
            if re.search(r"\b(DONE|FAILED|CANCELLED)\b", line) and re.match(r"\[\s*\d+s\]", line):
                ok = " DONE " in line or line.split("]")[1].strip().startswith("DONE")
                self.log("  " + line[:200])
            elif line.startswith("RESULT"):
                self.log("  " + line[:200])
        p.wait()
        el = time.time() - t0
        self.tls.gpu = getattr(self.tls, "gpu", 0.0) + el
        if not ok:
            raise RuntimeError(f"submit {part} failed:\n" + "\n".join(tail))
        self.tls.exec = getattr(self.tls, "exec", 0.0) + self._gpu_exec(key or slug, pidf)
        files = self.fetch_seq(part, expect_n) if seq else self._wait_files(part, exts, expect_n)
        self.log(f"  {part}: {len(files)} file(s) in out/ after {time.time() - t0:.0f}s")
        return files

    def _relay_submit(self, part, why, graph, pidf):
        """on-box runner: hand the graph to the PC launcher (box_run.py), which queues it through progress_standin over the tunnel so the
        dashboard gets its live tile, and wait for its answer file. -> (ok, tail lines)."""
        rd = Path(self.relay)
        rd.mkdir(parents=True, exist_ok=True)
        resp = rd / f"{part}.resp.json"
        resp.unlink(missing_ok=True)
        tmp = rd / f"{part}.req.tmp"
        tmp.write_text(json.dumps(dict(part=part, piece=self.id, why=why, graph=graph, at=ts())), encoding="utf-8")
        os.replace(tmp, rd / f"{part}.req.json")
        while not resp.exists():
            time.sleep(0.5)
        time.sleep(0.05)
        d = json.loads(resp.read_text(encoding="utf-8"))
        if d.get("prompt_id"):
            pidf.write_text(str(d["prompt_id"]), encoding="utf-8")
        for line in d.get("log", []):
            self.log("  " + line)
        return bool(d.get("ok")), d.get("tail", [])

    # ---------------------------------------------------------------- audio
    def audio(self):
        return self.item("audio", self._audio)

    def _audio(self):
        if self.dry:
            a = self.j["audio"]
            cut = float(a.get("cut") or 0)
            self.log(f"[dry] audio: ffmpeg cut/fade of {a['src']} -> {self.dir / 'audio.wav'}, analysis.json (cut {cut or '?'} s)")
            return dict(paths=[], wav=str(self.dir / "audio.wav"), analysis=str(self.dir / "analysis.json"), cut=cut,
                        nframes=int(math.ceil(cut * self.fps)))
        import numpy as np
        import soundfile as sf
        a = self.j["audio"]
        src = (ROOT / a["src"]).resolve() if not Path(a["src"]).is_absolute() else Path(a["src"])
        cand = [Path(a["analysis"])] if a.get("analysis") else []
        cand += [VER / "local" / "audio" / f"{src.stem}_analysis.json"]
        ana = None
        for c in cand:
            c = c if c.is_absolute() else ROOT / c
            if c.exists():
                ana = json.loads(c.read_text(encoding="utf-8"))
                self.log(f"analysis {c.name}")
                break
        if ana is None:
            from tools.fusion import music_analysis
            ana = music_analysis.analyse(src, a.get("bpm"), 4, self.fps)
        x, sr = sf.read(str(src), always_2d=True)
        peak = np.abs(x).max(1)
        win = int(0.05 * sr)
        env = np.array([peak[i:i + win].max() for i in range(0, len(peak) - win, win)])
        live = np.nonzero(env > 0.01)[0]
        music_end = float((live[-1] + 1) * win / sr) if len(live) else len(x) / sr
        bar = float(ana["bar_len"])
        db = [float(d) for d in ana["downbeats"]]
        while db[-1] + bar < music_end + 0.1:             # the tracker stops early on tails: extend the grid
            db.append(db[-1] + bar)
        ring, fade = float(a.get("ring", 0.5)), float(a.get("fade", 0.5))
        if a.get("cut"):
            cut = float(a["cut"])
        else:
            last = max(d for d in db if d <= music_end + 0.1)
            cut = min(last + ring, len(x) / sr)
        nf = int(math.ceil(cut * self.fps - 1e-6))
        wav = self.dir / "audio.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", f"{cut:.4f}", "-af",
             f"afade=t=out:st={max(cut - fade, 0):.4f}:d={fade:.4f}", "-ar", "48000", "-ac", "2", str(wav)])
        tail = db[-1] + bar
        db_ext = db + [tail + bar * i for i in range(int(4 / bar) + 2)]
        beats = [float(b) for b in ana["beats"]]
        iv = float(np.median(np.diff(beats))) if len(beats) > 1 else 60.0 / float(ana["bpm"])
        while beats[-1] < cut + 4:
            beats.append(beats[-1] + iv)
        curves = {k: v[:nf + 1] for k, v in ana["curves"].items() if isinstance(v, list)}
        out = dict(src=str(src), music_end=round(music_end, 3), cut=round(cut, 4), nframes=nf, fade=fade, bpm=ana["bpm"],
                   bar_len=bar, key=ana["key"]["estimate"], beats=beats, downbeats=db_ext,
                   chord_changes=ana.get("chord_changes", []), curves=curves, fps=self.fps)
        (self.dir / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        self.log(f"audio: music ends {music_end:.2f}s, cut {cut:.2f}s ({nf} frames), bpm {ana['bpm']}, key {out['key']}")
        return dict(paths=[str(wav), str(self.dir / "analysis.json")], wav=str(wav), analysis=str(self.dir / "analysis.json"),
                    cut=cut, nframes=nf)

    def ana(self):
        if self.dry:
            return dict(bpm=float(self.j["audio"].get("bpm", 120)), beats=[], downbeats=[], curves={})
        return json.loads(Path(self.audio()["analysis"]).read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- pre-rendered media (v005)
    def _media(self, rel, what):
        """v005: stills / shots / depth / mattes rendered AHEAD (15_fusion/v005/MEDIA.md) are attached to the piece JSON as `media` paths (repo
        relative): the runner adopts them and never regenerates. Only where the file system is this machine's (local / --on-box); a declared file
        that is missing is an error, not a silent re-render."""
        if not rel or (self.rx is not None and not self.onbox):
            return None
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"{self.id}: media for {what} declared but missing: {p}")
        self.log(f"{what}: adopting pre-rendered media {rel}")
        return p

    def _media_frames(self, sid, key, mp4):
        """PNG folder of a media mp4 (depth / matte), cut by ffmpeg into work/<id>/media/<sid>_<key>/ -> sorted file list."""
        d = self.dir / "media" / f"{sid}_{key}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), str(d / "f_%05d.png")])
        return [str(f) for f in sorted(d.glob("*.png"), key=lambda f: natural(f.name))]

    # ---------------------------------------------------------------- stills
    def still(self, sid):
        return self.item(f"still:{sid}", lambda: self._still(sid))

    def _still(self, sid):
        sys.path.insert(0, str(ROOT / "workflows" / "api"))
        import t2i_patch
        s = self.stills[sid]
        m = self._media(s.get("media"), f"still {sid}")
        if m:
            return dict(paths=[str(m)], local=str(m), media=True)
        model = s.get("model", "moss")
        gname = {"moss": "moss_t2i"}[model]
        if model == "moss" and s.get("lora"):
            gname = "moss_t2i_lora"
        g = json.loads((ROOT / "workflows" / "api" / f"{gname}.json").read_text(encoding="utf-8"))
        w, h = s.get("size", [1088, 1920])
        lora = s.get("lora")
        g = t2i_patch.patch(g, s["prompt"], int(s.get("seed", 1)), w, h, lora, s.get("strength"), s.get("negative"), None,
                            bool(s.get("trigger", True)))
        files = self.submit(g, f"still_{sid}", f"{self.id} still {sid}: {s['prompt'][:70]}", exts=("png",), key=f"still:{sid}")
        return dict(paths=[str(files[0])], local=str(files[0]))

    # ---------------------------------------------------------------- type
    def need_passes(self):
        from tools.fusion import recipe_eval as _RE
        ps = set(_RE.pass_names(self.j.get("type_passes"))) | {"matte"}
        from tools.fusion import multiplane_recipe
        ps |= multiplane_recipe.extra_passes(self.j)               # plate = multiplane: the cards composite the type pass by depth
        refs = set()
        _refs(self.j.get("compose", {}).get("shots", {}), refs)
        _refs(self.j.get("compose", {}).get("default", {}), refs)
        for r in refs:
            m = re.match(r"type_(\w+?)(?:@-?\d+)?$", r)
            if m:
                ps.add(m.group(1))
        if refs & {"type_matte", "type_line"} and self.j.get("compose", {}).get("beauty_face") is not False:
            ps.add("beauty")                                       # recipe_eval.legibility_pass draws the type from the beauty pass
        for s in self.j.get("shots", []):
            if s["kind"] == "control":
                ps.add({"union": "depth"}.get(s.get("control", {}).get("pass", "depth"), s.get("control", {}).get("pass", "depth")))      # `union` (owner.control_pass_for) is an app-graph pass (depth + line); this runner reads the depth pass
            if s["kind"] == "motiontrack":
                ps |= {"flow", "matte"}
        return sorted(ps)

    def typ(self):
        return self.item("type", self._type)

    def _type(self):
        if self.rx is not None:
            return self._type_box()
        from tools.fusion import type_passes
        a = self.audio()
        an = self.ana()
        spec = copy.deepcopy(self.j["type"])
        spec["fps"] = self.fps
        spec["duration"] = float(a["cut"])
        spec["bpm"] = float(an["bpm"])
        spec["beats"] = an["beats"]
        spec["downbeats"] = an["downbeats"]
        spec.setdefault("bar", 4)
        spec["size"] = list(TYPE_SIZE)
        (self.dir / "type_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        passes = self.need_passes()
        out = self.dir / "type"
        man = type_passes.render_passes(spec, str(out), size=TYPE_SIZE, passes=passes, procs=self.procs, overwrite=True, mp4=False,
                                        verbose=False)
        sheet = REVIEW / f"{self.id}_type_sheet.png"
        type_passes.make_sheet(str(out), str(sheet), n=6, passes=[p for p in passes if p != "flow"][:4])
        self.log(f"type: {man['nf']} frames {TYPE_SIZE}, passes {passes}, wall {man['timing']['wall_s']:.0f}s on {man['timing']['procs']} procs")
        for w in man["info"]["words"]:
            self.log(f"   word {w['text']!r:14} start {w['start']:.2f} hit {w['hit']:.2f} release {w['release']:.2f} end {w['end']:.2f}  {w['lines']}")
        for n in man["info"]["notes"]:
            self.log(f"   NOTE {n}")
        return dict(paths=[str(out / "manifest.json")], dir=str(out), passes=passes, nf=man["nf"], sheet=str(sheet),
                    info=man["info"]["words"])

    # ---------------------------------------------------------------- shot layout (piece time)
    def layout(self):
        """start frame (piece time) + length of every shot; shot k starts `crossfade` before the previous ends."""
        cf = self.j.get("compose", {}).get("crossfade", 0.5)
        ids = [s["id"] for s in self.j["shots"]]
        cfl = cf if isinstance(cf, list) else [cf] * max(len(ids) - 1, 0)
        out, start = {}, 0
        for k, sid in enumerate(ids):
            n = int(self.shots[sid].get("span") or self.shots[sid]["frames"])   # optional "span": piece frames the shot occupies (default = its frames)
            c = int(round(cfl[k - 1] * self.fps)) if k else 0
            if k:
                start = start - c
            out[sid] = dict(start=start, frames=n, xf=c)
            start += n
        return out, cfl

    # ---------------------------------------------------------------- shots
    def shot(self, sid):
        return self.item(f"shot:{sid}", lambda: self._shot(sid))

    def _on_screen_frames(self, sid, prev):
        """frames of shot `sid` that are on screen (its `span`; hard cuts generate a few frames more): the continuity frame of a chained shot is the
        last frame the viewer sees before the cut, not the last generated one."""
        sh = next((x for x in self.j.get("shots", []) if x.get("id") == sid), {})
        return int(min(sh.get("span") or prev["frames"], prev["frames"]))

    def first_frame(self, s):
        ref = s.get("first")
        if ref is None:
            return None
        if ref.startswith("last:"):
            prev = self.shot(ref[5:])
            if self.rx is not None:
                return self._first_frame_box(s, ref, prev)
            png = self.dir / f"{s['id']}_first_from_{ref[5:]}.png"
            from tools.fusion import compose
            fs = compose.FrameSource(prev["local"])
            xf = self.layout()[0][s["id"]]["xf"]                   # frame on screen when this shot's crossfade starts -> seamless blend
            img = fs[max(self._on_screen_frames(ref[5:], dict(frames=len(fs))) - 1 - xf, 0)]
            fs.close()
            compose.imwrite(png, img)
            return self.upload(png, f"{s['id']}_f0.png")
        st = self.still(ref)
        if self.rx is not None:                       # box: the still is already on the box, copy it into ComfyUI/input
            name = f"{self.id}_still_{ref}.png"
            self.rx.sh(f"cp {self.rx.q(st['local'])} {self.rx.input}/{name}")
            return name
        return self.upload(st["local"], f"still_{ref}.png")

    def _shot(self, sid):
        s = self.shots[sid]
        kind = s["kind"]
        n = int(s["frames"])
        m = self._media((s.get("media") or {}).get("shot"), f"shot {sid}")
        if m:
            from tools.fusion import compose
            w0, h0, nf, fps = compose._probe_video(m)
            self.log(f"shot {sid}: {m.name} {w0}x{h0} {nf}f @ {fps:g} (media)")
            return dict(paths=[str(m)], local=str(m), frames=nf, size=[w0, h0], media=True)
        assert n == snap8(n), f"shot {sid}: frames {n} is not 8k+1"
        preview = self.preview or self.j.get("quality") == "preview"
        draft = self.draft or self.j.get("quality") == "draft"
        single = draft or preview                          # one-stage LTX (no stage 2 / x2 decode)
        cap = I2V_MAX_FRAMES if kind == "i2v" else (IC_DRAFT_MAX_FRAMES if single else IC_MAX_FRAMES)
        assert n <= cap, f"shot {sid}: too many frames for {kind} ({n} > {cap})"
        why = f"{self.id} shot {sid} ({kind}, {n}f): {s.get('prompt', '')[:60]}"
        f0 = self.first_frame(s)
        if kind == "i2v":
            w, h = s.get("size", [1088, 1920])
            if preview:                                     # preview: plain i2v at the closest 32-multiple to 9:16 near 360 wide
                from tools.fusion import plan_rules
                w, h = plan_rules.orient(plan_rules.PREVIEW_I2V, plan_rules.format_of(self.j))
            elif draft and max(w, h) > 1024:                # draft: the plan's final size halved = the stage-1 size
                w, h = w // 2, h // 2
            g = i2v_graph(s["prompt"], f0, n, w, h, int(s.get("seed", 1)), self.fps, two_stage=not single)
            files = self.submit(g, sid, why, key=f"shot:{sid}")
        else:
            from tools.fusion import ltx_ic
            w, h = s.get("size", [1088, 1920])
            if preview:                                     # preview: 384x704 whatever the plan's size says (orientation kept by plan_size)
                w, h = (704, 384) if w > h else (384, 704)
            elif draft and max(w, h) > 1024:
                w, h = w // 2, h // 2
            (w, h), two = ltx_ic.plan_size(kind, (w, h), draft, preview)      # motiontrack: 1152x2048 two-stage / 576x1024 draft single
            lay, _ = self.layout()
            start = lay[sid]["start"]
            if kind == "control":
                ref = self.control_video(sid, s, start, n)
                g = ltx_ic.build("control", ref_video=ref, first_frame=f0, prompt=s["prompt"], size=(w, h), frames=n,
                                 seed=int(s.get("seed", 1)), two_stage=two, out_prefix=f"fusion/{self.id}_{sid}")
            else:
                tr = self.make_tracks(sid, s, start, n, (w, h))
                g = ltx_ic.build("motiontrack", tracks=tr, first_frame=f0, prompt=s["prompt"], size=(w, h), frames=n,
                                 seed=int(s.get("seed", 1)), two_stage=two, out_prefix=f"fusion/{self.id}_{sid}")
            files = self.submit(g, sid, why, key=f"shot:{sid}")
        mp4 = [p for p in files if p.suffix == ".mp4"][0]
        if self.rx is not None:
            w0, h0, nf, fps = self.rx_probe(mp4, (s.get("size", [1088, 1920])[0], s.get("size", [1088, 1920])[1], n))
        else:
            from tools.fusion import compose
            w0, h0, nf, fps = compose._probe_video(mp4)
        self.log(f"shot {sid}: {mp4.name} {w0}x{h0} {nf}f @ {fps:g}")
        return dict(paths=[str(mp4)], local=str(mp4), frames=nf, size=[w0, h0])

    def control_video(self, sid, s, start, n):
        t = self.typ()
        p = {"union": "depth"}.get(s.get("control", {}).get("pass", "depth"), s.get("control", {}).get("pass", "depth"))
        if self.rx is not None:                       # box: cut the control reference from the type PNGs on the box, straight into input/
            name = f"{self.id}_ctl_{sid}.mp4"
            self.rx.sh(f"ffmpeg -v error -y -framerate {self.fps} -start_number {start} -i {t['dir']}/{p}/{p}_%05d.png "
                       f"-vf 'tpad=stop_mode=clone:stop=400,format=yuv420p' -frames:v {n} -c:v libx264 -crf 10 -r {self.fps} "
                       f"{self.rx.input}/{name}", timeout=900)
            return name
        dst = self.dir / "uploads" / f"{self.id}_ctl_{sid}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-framerate", str(self.fps), "-start_number", str(start), "-i",
             str(Path(t["dir"]) / p / f"{p}_%05d.png"), "-vf", "tpad=stop_mode=clone:stop=400,format=yuv420p",
             "-frames:v", str(n), "-c:v", "libx264", "-crf", "10", "-r", str(self.fps), str(dst)])
        return self.upload(dst, dst.name)

    def make_tracks(self, sid, s, start, n, size=None):
        if self.rx is not None:
            raise NotImplementedError("motiontrack shots (type-flow tracks) are not supported with --exec box yet; use --exec local")
        import numpy as np
        from tools.fusion import compose
        t = self.typ()
        d = Path(t["dir"])
        tk = s.get("tracks", {})
        npts = int(tk.get("n", 4))
        rng = np.random.default_rng(int(tk.get("seed", 1)))
        m = compose.imread(d / "matte" / f"matte_{start:05d}.png", "gray")
        ys, xs = np.nonzero(m > 0.5)
        if len(xs) < npts:
            raise RuntimeError(f"tracks for {sid}: type matte empty at frame {start}")
        sel = rng.choice(len(xs), min(len(xs), 400), replace=False)
        pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float64)
        chosen = [pts[0]]
        for _ in range(npts - 1):                                        # farthest-point sampling
            dd = np.min([np.hypot(*(pts - c).T) for c in chosen], axis=0)
            chosen.append(pts[int(np.argmax(dd))])
        pos = np.array(chosen)
        tracks = [[] for _ in range(npts)]
        W, H = TYPE_SIZE
        fw, fh = size or s.get("size", [1152, 2048])             # the motiontrack graph size (ltx_ic.plan_size)
        for f in range(n):
            for i in range(npts):                                        # type px (1088 wide, 4 px cropped) -> final px
                tracks[i].append({"x": (pos[i, 0] - 4) * fw / 1080.0, "y": pos[i, 1] * fh / 1920.0})
            fp = d / "flow" / f"flow_{min(start + f, t['nf'] - 1):05d}.npy"
            fl = np.load(fp).astype(np.float32)                           # half-res, dx dy in half-res px
            for i in range(npts):
                x, y = pos[i] / 2.0
                xi, yi = int(np.clip(round(x), 0, fl.shape[1] - 1)), int(np.clip(round(y), 0, fl.shape[0] - 1))
                pos[i] += 2.0 * fl[yi, xi]
                pos[i] = np.clip(pos[i], 0, [W - 1, H - 1])
        return tracks

    # ---------------------------------------------------------------- alpha / depth
    def _ref_clip(self, sid, tag, w, h):
        sh = self.shot(sid)
        if self.rx is not None:                       # box: scale/crop the shot on the box, straight into ComfyUI/input
            name = f"{self.id}_{sid}_{tag}.mp4"
            self.rx.sh(f"ffmpeg -v error -y -i {self.rx.q(sh['local'])} -vf "
                       f"{self.rx.q(f'scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}')} -r {self.fps} -an "
                       f"-c:v libx264 -crf 10 -pix_fmt yuv420p {self.rx.input}/{name}", timeout=900)
            return name, sh["frames"]
        dst = self.dir / "uploads" / f"{self.id}_{sid}_{tag}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-i", sh["local"], "-vf",
             f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}", "-r", str(self.fps), "-an",
             "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(dst)])
        return self.upload(dst, dst.name), sh["frames"]

    def alpha(self, sid):
        return self.item(f"alpha:{sid}", lambda: self._alpha(sid))

    def _alpha(self, sid):
        m = self._media((self.shots[sid].get("media") or {}).get("matte"), f"matte {sid}")
        if m:
            fl = self._media_frames(sid, "matte", m)
            return dict(paths=fl, files=fl, n=len(fl), media=True)
        from tools.fusion import ltx_ic
        aw, ah = 576, 1024
        if self.preview or self.j.get("quality") == "preview":       # preview: the matte runs at the footage's own size (multiple of 32),
            aw, ah = (int(v) for v in self.shot(sid)["size"])         # never rescaled up; the LoRA's recipe is 576x1024 (softer matte)
        name, n = self._ref_clip(sid, f"a{aw}", aw, ah)
        nn = min(snap8(n), 145)
        g = ltx_ic.build("alpha", ref_video=name, size=(aw, ah), frames=nn, seed=int(self.shots[sid].get("seed", 1)),
                         out_prefix=f"fusion/{self.id}_{sid}_alpha")
        files = self.submit(g, f"alpha_{sid}", f"{self.id} alpha matte of {sid} ({aw}x{ah}, {nn}f)", exts=("png",), expect_n=nn,
                            key=f"alpha:{sid}", seq=True)
        return dict(paths=[str(p) for p in files], files=[str(p) for p in files], n=len(files))

    def depth(self, sid):
        return self.item(f"depth:{sid}", lambda: self._depth(sid))

    def _depth(self, sid):
        m = self._media((self.shots[sid].get("media") or {}).get("depth"), f"depth {sid}")
        if m:
            fl = self._media_frames(sid, "depth", m)
            return dict(paths=fl, files=fl, n=len(fl), media=True)
        dw, dh = 540, 960
        if self.preview or self.j.get("quality") == "preview":       # preview: the compose size (downscale only), 16:9 swaps the sides
            dw, dh = (640, 360) if self.j.get("format") == "16:9" else (360, 640)
        name, n = self._ref_clip(sid, f"d{dw}", dw, dh)
        g = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {"class_type": "LoadVideoDepthAnythingModel", "inputs": {"model": "video_depth_anything_vits.pth"}},
            "4": {"class_type": "VideoDepthAnythingProcess",
                  "inputs": {"vda_model": ["3", 0], "images": ["2", 0], "input_size": 518, "max_res": 1280, "precision": "fp16"}},
            "5": {"class_type": "VideoDepthAnythingOutput", "inputs": {"depths": ["4", 0], "colormap": "gray"}},
            "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": f"fusion/{self.id}_{sid}_depth"}},
        }
        files = self.submit(g, f"depth_{sid}", f"{self.id} video depth of {sid} ({n}f)", exts=("png",), expect_n=n,
                            key=f"depth:{sid}", seq=True)
        return dict(paths=[str(p) for p in files], files=[str(p) for p in files], n=len(files))

    # ---------------------------------------------------------------- compose
    def composed(self, sid):
        return self.item(f"compose:{sid}", lambda: self._compose(sid))

    def recipe(self, sid):
        comp = self.j["compose"]
        spec = comp.get("shots", {}).get(sid) or comp.get("default") or {"ops": []}
        ops = _subst(copy.deepcopy(spec["ops"]), {"$matte": f"${sid}_matte", "$depth": f"${sid}_depth", "$shot": f"${sid}"})
        from tools.fusion import multiplane_recipe
        ops = multiplane_recipe.apply(self, ops, sid)              # plate = multiplane: the `multiplane` op renders plate + type through the cards
        from tools.fusion import contour_sweep as _CS              # 2026-10-30: compose.contour_sweep = true / effect `contour_sweep` -> the sweep op after the last type drawer
        ops = _CS.recipe_ops(self.j, ops, sid, self.shots)
        lay, _ = self.layout()
        L = lay[sid]
        sh = self.shot(sid)
        fpath = self.footage_path(sid)                # the shot, or its masked Motion-Track result when the shot has one that worked
        size = tuple(comp.get("size", [1080, 1920]))
        refs = set()
        _refs([o.get("args", {}) for o in ops], refs)
        if self.j.get("signoff") and not self.j["signoff"].get("off") and sid:       # signoff.off (planner default): no title / BPM caption
            so = self.j["signoff"]
            t0f = int(round(float(so["t0"]) * self.fps))
            if L["start"] + L["frames"] > t0f:
                ops.append({"op": "over", "out": spec.get("output", "image"),
                            "args": {"base": f"${spec.get('output', 'image')}", "fg": list(c / 255.0 for c in so.get("color", [235, 230, 225])),
                                     "alpha": "$signoff"}})
                refs.add("signoff")
        inputs = {"footage": dict(path=fpath, kind="rgb")}
        produced = {o.get("out", "image") for o in ops}
        builtin = {"frame_idx", "t", "u", "n", "palette", "piece_frame", "footage"}
        for r in sorted(refs - produced - builtin):
            if r == sid:
                inputs[r] = dict(path=fpath, kind="rgb")
            elif r == f"{sid}_matte":
                inputs[r] = dict(files=self.alpha(sid)["files"], kind="gray", native=True)
            elif r == f"{sid}_depth":
                fl = self.depth(sid)["files"]
                inputs[r] = dict(files=fl, kind="gray", native=True, invert=bool(self.shots[sid].get("depth_invert")),
                                 norm=_depth_norm(fl))
            elif r.startswith("type_"):
                p, _, lag = r[5:].partition("@")          # optional "@-3" = frame offset (echo: the type 3 frames ago)
                inputs[r] = dict(dir=str(Path(self.typ()["dir"]) / p), kind="gray" if p in ("matte", "depth", "line") else "rgb",
                                 offset=L["start"] + (int(lag) if lag else 0))
            elif _other_shot_ref(r, sid, list(self.shots)) is not None:      # window_cut.py: $next / $prev / $shot:<id> (+ _matte) = another shot, aligned in piece time
                tgt, kd = _other_shot_ref(r, sid, list(self.shots))
                off = L["start"] - lay[tgt]["start"]
                inputs[r] = (dict(path=self.footage_path(tgt), kind="rgb", offset=off) if kd == "footage"
                             else dict(files=self.alpha(tgt)["files"], kind="gray", native=True, offset=off))
            elif r == "signoff":
                so = self.j["signoff"]
                inputs[r] = dict(signoff=dict(text=so["text"], font=so.get("font", "Space Grotesk"), t0=float(so["t0"]),
                                              t1=float(self.audio()["cut"]), px=int(so.get("px", 30)), fade=float(so.get("fade", 0.35))),
                                 offset=L["start"])
            else:
                raise ValueError(f"compose {sid}: unknown reference ${r}")
        inputs["footage"] = dict(path=fpath, kind="rgb")
        rec = dict(name=sid, size=list(size), fps=self.fps, ops=ops, output=spec.get("output", "image"), inputs=inputs,
                   piece_offset=L["start"], frames=L["frames"], crf=comp.get("crf", 16),
                   curves={k: v for k, v in self.ana()["curves"].items() if isinstance(v, list)})
        if comp.get("holds"):                       # optional freeze-frames, piece time: [[t_s, dur_s], ...] (see _hold_index)
            rec["holds"] = comp["holds"]
            rec["holds_type"] = bool(comp.get("holds_type", True))
        if comp.get("palette"):
            from tools.fusion import compose
            pr = comp["palette"]
            fs = compose.FrameSource(fpath, "rgb", size)
            rec["palette_arr"] = compose.palette_from_footage(fs, k=int(pr.get("k", 5))).tolist()
            fs.close()
        return rec

    def _compose(self, sid):
        rec = self.recipe(sid)
        out = self.dir / "compose" / sid
        if out.exists():
            shutil.rmtree(out)
        (out / "frames").mkdir(parents=True)
        (out / "recipe.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        t0 = time.time()
        spf = compose_parallel(rec, out / "frames", self.procs)
        self.log(f"compose {sid}: {rec['frames']} frames in {time.time() - t0:.0f}s ({spf:.2f} s/frame/core)")
        return dict(paths=[str(out / "frames" / "00000.png")], frames_dir=str(out / "frames"), n=rec["frames"], s_per_frame=spf)

    def assemble(self):
        return self.item("assemble", self._assemble)

    def _assemble(self):
        if self.rx is not None:
            return self._assemble_box()
        from tools.fusion import compose
        lay, cfl = self.layout()
        a = self.audio()
        parts = []
        for sid in self.shots:
            c = self.composed(sid)
            L = lay[sid]
            parts.append(dict(src=c["frames_dir"], cut=(L["start"] + L["xf"] / 2.0) / self.fps if L["xf"] or L["start"] else 0.0))
        from tools.fusion import window_cut                       # a window cut owns its boundary: HARD cut at the incoming shot's piece start
        _own = window_cut.owned_boundaries(self.j)
        if _own:
            _ids = list(self.shots)
            cfl = list(cfl)
            for k in _own:
                if k < len(cfl):
                    cfl[k] = 0.0
                    parts[k + 1]["cut"] = lay[_ids[k + 1]]["start"] / self.fps
        size = tuple(self.j.get("compose", {}).get("size", [1080, 1920]))
        dst = self.dir / f"{self.id}_final.mp4"
        r = compose.assemble(parts, a["wav"], str(dst), crossfades=[c for c in cfl], fps=self.fps, size=size,
                             crf=int(self.j.get("compose", {}).get("crf", 16)))
        self.log(f"assembled {dst.name}: {r['frames']} frames, {r['seconds']:.2f}s")
        return dict(paths=[str(dst)], mp4=str(dst), frames=r["frames"], seconds=r["seconds"])

    # ---------------------------------------------------------------- final
    def final(self):
        return self.item("final", self._final)

    def _final(self):
        if self.rx is not None:
            return self._final_box()
        import cv2
        import numpy as np
        from tools.fusion import compose
        mp4 = self.assemble()["mp4"]
        fs = compose.FrameSource(mp4)
        n = len(fs)
        REVIEW.mkdir(parents=True, exist_ok=True)

        def tile(i, w):
            f = fs[i]
            im = cv2.cvtColor(np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{i / self.fps:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 60), 1, cv2.LINE_AA)
            return np.pad(im, ((0, 3), (0, 3), (0, 0)), constant_values=40)

        idx = np.unique(np.linspace(0, n - 1, 12).round().astype(int))
        tiles = [tile(int(i), 270) for i in idx]
        sheet = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(2)])
        sp = REVIEW / f"{self.id}_sheet.png"
        cv2.imwrite(str(sp), sheet)
        step = max(1, int(round(self.fps / 6)))
        idx6 = list(range(0, n, step))
        t6 = [tile(i, 144) for i in idx6]
        cols = 12
        while len(t6) % cols:
            t6.append(np.zeros_like(t6[0]))
        strip = np.vstack([np.hstack(t6[r:r + cols]) for r in range(0, len(t6), cols)])
        stp = REVIEW / f"{self.id}_strip6fps.png"
        cv2.imwrite(str(stp), strip)
        fs.close()
        pr = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration", "-of", "csv=p=0",
                  mp4]).stdout.strip().replace("\n", " | ")
        self.log(f"final: {mp4}  [{pr}]  sheet {sp}")
        return dict(paths=[str(sp), str(stp)], sheet=str(sp), strip=str(stp), mp4=mp4, probe=pr)

    # ---------------------------------------------------------------- box mode (--exec box): everything below runs through self.rx
    def rx_probe(self, path, dry=(1088, 1920, 73)):
        """(w, h, nframes, fps) of an mp4 on the box (ffprobe over ssh)."""
        if self.dry:
            return dry[0], dry[1], dry[2], 24.0
        r = self.rx.sh("ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,duration:format=duration "
                       f"-of json {self.rx.q(str(path))}")
        return parse_probe(r.stdout)

    def piece_box(self):
        """the piece dict the box-side compose reads (recipe_eval): P*.json, plus the draft switches."""
        j = copy.deepcopy(self.j)
        j["fps"] = self.fps
        if (j.get("type") or {}).get("plates"):
            j["type"]["plates"] = self.box_plates(j["type"]["plates"], ship=False)
        if self.draft:
            j["quality"] = "draft"
            j.setdefault("sizes", {})["compose"] = [540, 960]
        elif self.preview:
            from tools.fusion import plan_rules
            j["quality"] = "preview"
            j.setdefault("sizes", {})["compose"] = plan_rules.orient([360, 640], plan_rules.format_of(j))
        return j

    def box_plates(self, plates, ship=True):
        """type.plates -> absolute BOX paths (box_ssh.rewrite_plates). A PC-driven box run ships the files first (hash-skipping delta); on the
        box / in sim they must already be there. Contour ports declared + a plate missing = PlatesMissing, never a silent fallback to module sides."""
        from tools.fusion import box_ssh
        if not box_ssh.plate_strings(plates):
            return plates
        new = box_ssh.rewrite_plates(plates, self.rx.root)
        if not ship or self.dry:
            return new
        ports = box_ssh.declares_ports(self.j.get("type"))
        if not self.rx.onbox:
            pairs, missing = box_ssh.plate_pairs(plates, ROOT)
            if missing and ports:
                raise box_ssh.PlatesMissing(f"{self.id}: contour ports declared but plate files missing: {', '.join(missing)}")
            d = box_ssh.delta_put(self.rx, [(l, f"{self.rx.root}/{r}") for l, r in pairs])
            self.log(f"plates -> box: {d['sent']} sent ({d['bytes'] // 1024} KB), {d['skipped']} already there")
        gone = box_ssh.missing_on_disk(new) if (self.rx.onbox or self.rx.sim) else []
        if gone and ports:
            raise box_ssh.PlatesMissing(f"{self.id}: contour ports declared but plates missing on the box: {', '.join(gone)}")
        return new

    def push_piece_files(self):
        """once: audio.wav, analysis.json and the piece json -> <root>/work/<id>/ on the box (after the audio stage)."""
        rx = self.rx
        with self._push_lock:
            if self._pushed:
                return
            rx.wait_ready()
            a = self.audio()
            rx.put_many([(a["wav"], f"{rx.work}/audio.wav"), (a["analysis"], f"{rx.work}/analysis.json"),
                         (json.dumps(self.piece_box()).encode("utf-8"), f"{rx.work}/piece.json")])
            self._pushed = True

    def _type_box(self):
        """type passes ON the box: spec + audio analysis go up, procs = box cores - 2 single-threaded workers, nohup + done marker, polled.
        Started by the first thread that gets here, i.e. as soon as the audio analysis exists, so it overlaps stills / shots / alpha."""
        rx = self.rx
        a = self.audio()
        an = self.ana()
        spec = copy.deepcopy(self.j["type"])
        spec["fps"] = self.fps
        spec["duration"] = float(a["cut"])
        spec["bpm"] = float(an["bpm"])
        spec["beats"] = an["beats"] if "beats" in an else []
        spec["downbeats"] = an["downbeats"] if "downbeats" in an else []
        spec.setdefault("bar", 4)
        spec["size"] = list(self.type_size)
        if spec.get("plates"):
            spec["plates"] = self.box_plates(spec["plates"])
        if not self.dry:
            (self.dir / "type_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        passes = self.need_passes()
        rx.wait_ready()
        self.push_piece_files()
        # box cores - 2 single-threaded workers, capped at 20: each holds a 2x-supersampled 1088x1920 raster (~1 GB) and ComfyUI's LTX models
        # already sit in the same 62 GB of RAM (the in-graph recomposite OOM-killed it once); --procs N overrides
        procs = self.procs_arg or max(1, min(int(rx.info.get("cores", 12)) - 2, 20))
        rx.put_json(spec, f"{rx.work}/type_spec.json")
        env = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 "
        out = f"{rx.out}/type"
        sheet = f"{rx.out}/{self.id}_type_sheet.png"
        t0 = time.time()
        rx.bg("type", rx.py_cmd("type", f"--spec {rx.work}/type_spec.json --out {out} --passes {','.join(passes)} --procs {procs} "
                                         f"--sheet {sheet}", env))
        self.log(f"type: started on the box ({procs} procs, passes {passes}, {self.type_size[0]}x{self.type_size[1]})")
        res = rx.wait("type", progress=self.log)
        self.log(f"type: {res.get('nf')} frames, wall {res.get('render_wall_s', 0):.0f}s on {res.get('procs')} procs "
                 f"(waited {time.time() - t0:.0f}s)")
        for w in res.get("info", []):
            self.log(f"   word {w['text']!r:14} start {w['start']:.2f} hit {w['hit']:.2f} release {w['release']:.2f} end {w['end']:.2f}  {w['lines']}")
        for n in res.get("notes", []):
            self.log(f"   NOTE {n}")
        return dict(paths=[], dir=out, passes=passes, nf=res.get("nf"), sheet=sheet, info=res.get("info", []), procs=procs,
                    remote_s=res.get("remote_wall_s"), size=list(self.type_size))

    def _first_frame_box(self, s, ref, prev):
        """continuity first frame (`last:s1a`): extract it ON the box with ffmpeg straight into ComfyUI/input (no download / upload)."""
        rx = self.rx
        xf = self.layout()[0][s["id"]]["xf"]                       # frame on screen when this shot's crossfade starts -> seamless blend
        idx = max(self._on_screen_frames(ref[5:], prev) - 1 - xf, 0)
        name = f"{self.id}_{s['id']}_f0.png"
        rx.sh(f"ffmpeg -v error -y -i {rx.q(prev['local'])} -vf {rx.q(f'select=eq(n\\,{idx})')} -frames:v 1 {rx.input}/{name}")
        self.log(f"first frame of {s['id']}: frame {idx} of {ref[5:]} -> input/{name}")
        return name

    def _assemble_box(self):
        """compose (torch CUDA, FUSION_BACKEND=torch) of every shot + assemble + encode in ONE process on the box; reads the shot mp4s,
        alpha / depth PNGs and the type layers where they are. Per-shot compose times come back as sub-items compose:<sid>."""
        from tools.fusion import recipe_eval as RE
        rx = self.rx
        piece = self.piece_box()
        shots = []
        self.gpu_quiet()
        for sid in self.shots:
            need = RE.shot_needs(piece, sid)
            ent = dict(sid=sid, footage=self.footage_path(sid))
            if RE.shot_needs_flow(piece, sid):
                ent["flow"] = self.flow(sid)["path"]
            if need["matte"]:
                ent["matte"] = list(self.alpha(sid)["files"])
            if need["depth"]:
                ent["depth"] = list(self.depth(sid)["files"])
            shots.append(ent)
        t = self.typ()
        self.audio()
        self.push_piece_files()
        out_mp4 = f"{rx.out}/{self.id}_final.mp4"
        job = dict(piece=f"{rx.work}/piece.json", analysis=f"{rx.work}/analysis.json", audio=f"{rx.work}/audio.wav", type_dir=t["dir"],
                   out_mp4=out_mp4, backend=rx.backend, encoder="auto", crf=int(self.j.get("compose", {}).get("crf", 16)),
                   nvenc=bool(rx.info.get("nvenc")), shots=shots, report=f"{rx.out}/compose_report.json")
        rx.put_json(job, f"{rx.work}/compose_job.json")
        rx.bg("compose", rx.py_cmd("compose", f"--job {rx.work}/compose_job.json"))
        self.log(f"compose+assemble: started on the box ({len(shots)} shots, backend {rx.backend})")
        res = rx.wait("compose", progress=self.log, poll=3.0)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        for sid, d in res.get("shots", {}).items():
            for hh in d.get("hold_net") or []:                 # compose.hold_net: legibility score per word hold + whether the net fired
                self.log(f"  hold_net {sid} {hh['hit']:.2f}-{hh['release']:.2f}s: mid score {hh['mid_score']} min {hh['min_score']} "
                         f"fired {hh['fired']} (max k {hh['max_k']}, {hh['fired_frames']}/{hh['frames']} frames)")
            if d.get("legible_color"):                         # tools/fusion/legible_color.py: the type colour chosen from the measured background, per word
                from tools.fusion import legible_color as LCOL
                for ln in LCOL.format_table(d["legible_color"]).splitlines():
                    self.log(f"  legible_color {sid} {ln}")
            with self.lock:
                self.state["items"][f"compose:{sid}"] = dict(out=dict(paths=[], remote=d.get("footage")), wall_s=d["wall_s"], gpu_s=0.0,
                                                              gpu_exec_s=0.0, sub=True, s_per_frame=d["s_per_frame"], at=now)
        self.save()
        asm = res.get("assemble", {})
        self.log(f"assembled {Path(out_mp4).name}: {asm.get('frames')} frames, {asm.get('seconds')}s, {asm.get('encoder')}, "
                 f"compose {res.get('compose_wall_s')}s + encode {asm.get('wall_s')}s on {res.get('backend')}")
        return dict(paths=[], mp4=out_mp4, remote=out_mp4, frames=asm.get("frames"), seconds=asm.get("seconds"), encoder=asm.get("encoder"),
                    compose_wall_s=res.get("compose_wall_s"), assemble_wall_s=asm.get("wall_s"), backend=res.get("backend"))

    def _final_box(self):
        """contact sheet on the box, then download ONLY the final mp4 and the sheet."""
        rx = self.rx
        asm = self.assemble()
        mp4 = asm["mp4"]
        sheet, strip = f"{rx.out}/{self.id}_sheet.png", f"{rx.out}/{self.id}_strip6fps.png"
        r = rx.sh(rx.py_cmd("final", f"--mp4 {mp4} --sheet {sheet} --strip {strip} --fps {self.fps}"), timeout=900)
        probe = ""
        if not self.dry:
            d = json.loads(r.stdout.strip().splitlines()[-1])
            if not d.get("ok"):
                raise RuntimeError(f"final failed on the box: {d.get('error')}")
            probe = d["result"]["probe"]
        t0 = time.time()
        dst = self.dir / f"{self.id}_final.mp4"
        sp = REVIEW / f"{self.id}_sheet.png"
        rx.get_many([(mp4, dst), (sheet, sp)])
        dl = round(time.time() - t0, 1)
        self.log(f"final: {dst}  [{probe}]  sheet {sp}  (downloaded in {dl}s; 6 fps strip stays on the box: {strip})")
        return dict(paths=[str(sp)], sheet=str(sp), mp4=str(dst), probe=probe, download_s=dl, remote_strip=strip)

    # ---------------------------------------------------------------- flow (box mode): footage optical flow for compose's $flow
    def gpu_quiet(self):
        """block until every ComfyUI render of this piece is done (shots, alpha, depth, masktrack), then free ComfyUI's VRAM once, so the
        on-box RAFT / compose jobs do not fight the LTX model for memory."""
        for sid in self.shots:
            self.shot(sid)
        for sid in self.shots:
            if self.shots[sid].get("matte"):
                self.alpha(sid)
            if self.shots[sid].get("depth"):
                self.depth(sid)
            if self.shots[sid].get("masktrack"):
                self.masktrack(sid)
        with self.lock:
            if getattr(self, "_freed", False):
                return
            self._freed = True
        if self.dry or self.sim:
            self.log("[dry/sim] POST /free {unload_models, free_memory} to the box's ComfyUI before the on-box GPU jobs")
            return
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{BOX['tunnel_port']}/free", method="POST", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"unload_models": True, "free_memory": True}).encode())
            urllib.request.urlopen(req, timeout=30).read()
            time.sleep(3)
            self.log("freed ComfyUI VRAM (unload_models, free_memory) for the on-box RAFT / compose jobs")
        except Exception as e:      # noqa: BLE001
            self.log(f"  (could not POST /free: {e})")

    def flow(self, sid):
        return self.item(f"flow:{sid}", lambda: self._flow(sid))

    def _flow(self, sid):
        """dense forward optical flow of the shot's final footage (the masktrack result when there is one; "flow": {"from": "shot"} = the
        original shot), RAFT on CUDA (DIS on CPU when RAFT fails), cached as out/<id>/flow/<sid>_flow.npy [T,H/2,W/2,2] float16."""
        rx = self.rx
        if rx is None:
            raise NotImplementedError("the flow stage (compose $flow) needs --exec box")
        spec = self.shots[sid].get("flow")
        self.gpu_quiet()
        src = self.shot(sid)["local"] if isinstance(spec, dict) and spec.get("from") == "shot" else self.footage_path(sid)
        rx.wait_ready()
        out = f"{rx.out}/flow/{sid}_flow.npy"
        job = dict(video=str(src), out=out, model=(spec.get("model", "raft_large") if isinstance(spec, dict) else "raft_large"), batch=4)
        rx.put_json(job, f"{rx.work}/flow_{sid}_job.json")
        with self._flow_lock:                         # one RAFT job on the GPU at a time
            r = rx.sh(rx.py_cmd("flow", f"--job {rx.work}/flow_{sid}_job.json"), timeout=1800)
        res = {}
        if not self.dry:
            d = json.loads(r.stdout.strip().splitlines()[-1])
            if not d.get("ok"):
                raise RuntimeError(f"flow failed on the box: {d.get('error')}")
            res = d["result"]
        self.log(f"flow {sid}: {res.get('method')} {res.get('shape')} {res.get('s_per_frame')} s/frame {'(RAFT failed, DIS fallback: ' + res['fallback'] + ')' if res.get('fallback') else ''}")
        return dict(paths=[], path=out, remote_s=res.get("wall_s"), method=res.get("method"), shape=res.get("shape"), s_per_frame=res.get("s_per_frame"))

    # ---------------------------------------------------------------- masktrack (optional stage, after alpha)
    def masktrack(self, sid):
        return self.item(f"masktrack:{sid}", lambda: self._masktrack(sid))

    def footage_path(self, sid):
        """the mp4 compose reads as $footage for shot `sid`: its masked Motion-Track result when the shot asks for one and it worked,
        else the original shot (also when the masktrack stage failed or was not run)."""
        loc = self.shot(sid)["local"]
        if self.shots[sid].get("masktrack"):
            m = self.masktrack(sid)
            if m.get("ok") and m.get("local"):
                return m["local"]
            self.log(f"masktrack {sid}: not available ({m.get('error', 'no result')}) - compose uses the original shot")
        return loc

    def mask_start(self, spec, L, n):
        """shot-local frame where the tracks leave the subject. spec['start'] = 'auto' (40 % into the shot) or piece-time seconds;
        snapped to the nearest beat of the song."""
        beats = [float(b) for b in self.ana().get("beats", [])]
        t0 = L["start"] / self.fps
        want = t0 + 0.4 * n / self.fps if spec.get("start", "auto") in ("auto", None) else float(spec["start"])
        if beats:
            want = min(beats, key=lambda b: abs(b - want))
        return int(max(0, min(round((want - t0) * self.fps), n - 2)))

    def _masktrack(self, sid):
        spec = self.shots[sid]["masktrack"]
        t_wall = time.time()
        try:
            if not self.shots[sid].get("matte"):
                raise RuntimeError('masktrack needs the shot\'s Alpha-Gen matte ("matte": true)')
            style = spec.get("style", "rise")
            subj_text = f"{spec.get('subject', 'the runner')} {self.shots[sid].get('prompt', '')}".lower()
            if style == "rain" and PERSON_RE.search(subj_text):          # downward trails on a person read as collapsing
                self.log(f"masktrack {sid}: 'rain' on a person -> 'rise'")
                style = "rise"
            if style not in MASK_STYLE_TEXT:
                raise ValueError(f"masktrack style {style!r}: one of {sorted(MASK_STYLE_TEXT)}")
            sh = self.shot(sid)
            al = self.alpha(sid)
            n = int(sh["frames"])
            L = self.layout()[0][sid]
            f0 = self.mask_start(spec, L, n)
            names = dict(footage=f"{self.id}_{sid}_mtfoot.mp4", matte=f"{self.id}_{sid}_mtmatte.mp4", corridor=f"{self.id}_{sid}_mtcorr.mp4")
            prompt = spec.get("prompt") or f"{self.shots[sid].get('prompt', '')}, {MASK_STYLE_TEXT[style].format(s=spec.get('subject', 'the runner'))}"
            job = dict(id=self.id, sid=sid, style=style, start_frame=f0, frames=n, matte=[str(p) for p in al["files"]],
                       footage=str(sh["local"]), names=names, prompt=prompt, denoise=float(spec.get("denoise", 1.0)),
                       corridor=bool(spec.get("corridor", False)), seed=int(spec.get("seed", 1)), n_points=int(spec.get("n_points", 40)),
                       distance=float(spec.get("distance", 300.0)), dilate=int(spec.get("dilate", 1 if spec.get("corridor", False) else 2)),
                       feather=float(spec.get("feather", 0.75)), out_prefix=f"mask_ltx/{self.id}_{sid}", copy_footage=True)
            self.log(f"masktrack {sid}: {style}, tracks leave the subject at shot frame {f0} ({f0 / self.fps:.2f}s), denoise {job['denoise']}")
            from tools.fusion import box_stages
            import argparse as _ap
            if self.rx is not None:
                rx = self.rx
                rx.wait_ready()
                job["input_dir"], job["graph_out"] = rx.input, f"{rx.work}/mt_{sid}_graph.json"
                job["region_out"] = f"{rx.work}/mt_{sid}_region.npy"
                rx.put_json(job, f"{rx.work}/mt_{sid}_job.json")
                r = rx.sh(rx.py_cmd("masktrack", f"--job {rx.work}/mt_{sid}_job.json"), timeout=900)
                graph, meta = {}, {}
                if not self.dry:
                    d = json.loads(r.stdout.strip().splitlines()[-1])
                    if not d.get("ok"):
                        raise RuntimeError(f"masktrack prep failed on the box: {d.get('error')}")
                    meta = d["result"]
                    graph = json.loads(rx.sh(f"cat {job['graph_out']}").stdout)
            else:
                job["input_dir"], job["graph_out"] = str(self.dir / "uploads"), str(self.dir / "graphs" / f"{self.id}_mt_{sid}_graph.json")
                job["region_out"] = str(self.dir / "graphs" / f"{self.id}_mt_{sid}_region.npy")
                jp = self.dir / "graphs" / f"{self.id}_mt_{sid}_job.json"
                jp.write_text(json.dumps(job), encoding="utf-8")
                meta = box_stages.cmd_masktrack(_ap.Namespace(job=str(jp)))
                graph = json.loads(Path(job["graph_out"]).read_text(encoding="utf-8"))
                for nm in [names["footage"], names["matte"]] + ([names["corridor"]] if job["corridor"] else []):
                    self.upload(self.dir / "uploads" / nm, nm)
            if not self.dry and not self.sim:
                try:
                    if not _http_json(f"http://127.0.0.1:{BOX['tunnel_port']}/object_info/MographVideoLatentMask", 20):
                        raise RuntimeError("node MographVideoLatentMask is missing on the box: copy 16_app/mask_ltx/box_pack/mograph_latent_mask "
                                           "to ComfyUI/custom_nodes and restart ComfyUI")
                except RuntimeError:
                    raise
                except Exception as e:      # noqa: BLE001
                    self.log(f"  (could not check MographVideoLatentMask via the tunnel: {e})")
            why = f"{self.id} masked Motion-Track {sid} ({style}, denoise {job['denoise']}, tracks from frame {f0})"
            files = self.submit(graph, f"masktrack_{sid}", why, key=f"masktrack:{sid}")
            raw = [p for p in files if p.suffix == ".mp4"][0]                       # 576x1024 decode, subject regenerated
            rjob = dict(footage=str(sh["local"]), raw=str(raw), region=job["region_out"], crf=12)
            if self.rx is not None:                                              # uint8 recomposite on the box, one frame at a time
                rjob["out_mp4"] = f"{self.rx.out}/mt_{sid}_composite.mp4"
                self.rx.put_json(rjob, f"{self.rx.work}/mt_{sid}_recomp.json")
                self.rx.sh(self.rx.py_cmd("recomposite", f"--job {self.rx.work}/mt_{sid}_recomp.json"), timeout=900)
                mp4 = PurePosixPath(rjob["out_mp4"])
                w0, h0, nf, fps = self.rx_probe(mp4, (1088, 1920, n))
            else:
                from tools.fusion import compose
                rjob["out_mp4"] = str(self.dir / f"{self.id}_mt_{sid}_composite.mp4")
                jp2 = self.dir / "graphs" / f"{self.id}_mt_{sid}_recomp.json"
                jp2.write_text(json.dumps(rjob), encoding="utf-8")
                box_stages.cmd_recomposite(_ap.Namespace(job=str(jp2)))
                mp4 = Path(rjob["out_mp4"])
                w0, h0, nf, fps = compose._probe_video(mp4)
            self.log(f"masktrack {sid}: {mp4.name} {w0}x{h0} {nf}f @ {fps:g}")
            return dict(ok=True, paths=[str(mp4)], local=str(mp4), raw=str(raw), frames=nf, size=[w0, h0], start_frame=f0, style=style, meta=meta)
        except Exception as e:      # noqa: BLE001
            self.log(f"WARNING masktrack {sid} failed after {time.time() - t_wall:.0f}s: {e} - compose falls back to the original shot")
            return dict(ok=False, error=str(e)[:400], paths=[str(self.dir / f"masktrack_{sid}_FAILED")])

    # ---------------------------------------------------------------- timing
    def _gpu_exec(self, label, pidf):
        """seconds ComfyUI spent EXECUTING the prompt (/history status.messages: execution_start -> execution_success), not queue time."""
        try:
            pid = Path(pidf).read_text().strip()
            h = _http_json(f"http://127.0.0.1:{BOX['tunnel_port']}/history/{pid}", 20)[pid]
            msgs = {m[0]: (m[1] or {}).get("timestamp") for m in h["status"]["messages"]}
            a = msgs.get("execution_start")
            b = msgs.get("execution_success") or msgs.get("execution_error") or msgs.get("execution_interrupted")
            if a and b:
                with self.lock:
                    self.state.setdefault("gpu_runs", []).append(dict(key=label, prompt_id=pid, start=a / 1000.0, end=b / 1000.0,
                                                                      exec_s=round((b - a) / 1000.0, 1)))
                    self.save()
                return (b - a) / 1000.0
            self.log(f"  (no execution_start/success in /history for {label}: {sorted(msgs)})")
        except Exception as e:      # noqa: BLE001
            self.log(f"  (GPU execution time unavailable for {label}: {e})")
        return 0.0

    def _sim_submit(self, slug, part, exts, seq):
        """--box-sim: stand in for a GPU render by copying the matching v002 output into the simulated ComfyUI output dir."""
        sp = FUSION / "v002" / "local" / "pieces" / self.id / "state.json"
        items = json.loads(sp.read_text(encoding="utf-8"))["items"]
        if slug.startswith("still_"):
            srcs = [items[f"still:{slug[6:]}"]["out"]["local"]]
        elif slug.startswith("alpha_"):
            srcs = items[f"alpha:{slug[6:]}"]["out"]["files"]
        elif slug.startswith("depth_"):
            srcs = items[f"depth:{slug[6:]}"]["out"]["files"]
        elif slug.startswith("masktrack_"):
            raise RuntimeError("sim: v002 has no masked Motion-Track render to stand in")
        else:
            srcs = [items[f"shot:{slug}"]["out"]["local"]]
        dst = Path(self.rx.output) / "projects" / BOX["idea"]
        dst.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(srcs, 1):
            ext = Path(src).suffix
            name = f"sim_{part}_image_{i:05d}_{ext}" if seq else f"sim_{part}_{i:05d}_{ext}"
            shutil.copyfile(src, dst / name)

    def write_bench(self):
        """<piece dir>/timings.json (stages, items, GPU execution, idle estimate) + 16_app/BENCH_<id>.md vs the v002 numbers."""
        with self.lock:
            items = {k: v for k, v in self.state["items"].items()}
            runs = list(self.state.get("gpu_runs", []))
        main = {k: v for k, v in items.items() if not v.get("sub")}
        stages = {}
        for st in STAGES:
            ks = [k for k in main if stage_of(k) == st]
            if ks:
                stages[st] = dict(wall_s=round(sum(main[k]["wall_s"] for k in ks), 1), gpu_exec_s=round(sum(main[k].get("gpu_exec_s", 0.0) for k in ks), 1),
                                  gpu_wait_s=round(sum(main[k].get("gpu_s", 0.0) for k in ks), 1), items=sorted(ks))
        t0s = [v["t_start"] for v in main.values() if "t_start" in v]
        t1s = [v["t_end"] for v in main.values() if "t_end" in v]
        window = (max(t1s) - min(t0s)) if t0s and t1s else 0.0
        exec_total = sum(r["exec_s"] for r in runs)
        iv = sorted((r["start"], r["end"]) for r in runs)
        busy, cur = 0.0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                busy += (cur[1] - cur[0]) if cur else 0.0
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        busy += (cur[1] - cur[0]) if cur else 0.0
        info = BOX.get("info") or {}
        life = None
        if info.get("started_running_at") and not self.sim:
            try:
                st_ = datetime.datetime.fromisoformat(str(info["started_running_at"]).replace("Z", "+00:00"))
                life = (datetime.datetime.now(datetime.timezone.utc) - st_).total_seconds()
            except Exception:       # noqa: BLE001
                life = None
        gpu = dict(exec_s=round(exec_total, 1), busy_union_s=round(busy, 1), run_window_s=round(window, 1),
                   idle_in_run_s=round(max(window - exec_total, 0.0), 1), idle_pct=round(100 * max(window - exec_total, 0.0) / window, 1) if window else None,
                   box_lifetime_s=round(life, 1) if life else None, idle_in_lifetime_s=round(max(life - exec_total, 0.0), 1) if life else None,
                   n_prompts=len(runs))
        rx = self.rx
        out = dict(piece=self.id, exec=self.exec_mode + (" (sim)" if self.sim else ""), version=VER.name if not self.sim else "sim", draft=self.draft,
                   written=datetime.datetime.now().isoformat(timespec="seconds"),
                   box=dict(iid=BOX["iid"], host=BOX["host"] if not self.sim else "sim", gpu=info.get("gpu_name") or rx.info.get("gpu"),
                            price_per_hour=info.get("price_per_hour"), cores=rx.info.get("cores"), nvenc=rx.info.get("nvenc"),
                            prepare_s=rx.prep_s, backend=rx.backend),
                   stages=stages, gpu=gpu,
                   items={k: dict(stage=stage_of(k), wall_s=v["wall_s"], gpu_exec_s=v.get("gpu_exec_s", 0.0), gpu_wait_s=v.get("gpu_s", 0.0),
                                  **({"sub_of": "assemble"} if v.get("sub") else {}),
                                  **({"s_per_frame": v["s_per_frame"]} if "s_per_frame" in v else {}),
                                  **({"remote_s": v["out"].get("remote_s")} if isinstance(v.get("out"), dict) and v["out"].get("remote_s") else {}))
                          for k, v in sorted(items.items())})
        (self.dir / "timings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        base = v002_baseline(self.id)
        md = bench_markdown(out, base, self.id)
        bench = (self.sim / f"BENCH_{self.id}_sim.md") if self.sim else (ROOT / "16_app" / f"BENCH_{self.id}.md")
        bench.parent.mkdir(parents=True, exist_ok=True)
        bench.write_text(md, encoding="utf-8")
        self.log(f"timings -> {self.dir / 'timings.json'}, bench -> {bench}")

    def plan_local(self, stage="all"):
        """--exec local --dry: which items would run (and which are cached in state.json); nothing is executed."""
        stages = STAGES if stage == "all" else stage.split(",")
        print(f"plan {self.id} exec=local out={VER.name} dir={self.dir}\n  box iid {BOX['iid']} {BOX['host']}:{BOX['ssh_port']} tunnel :{BOX['tunnel_port']}  "
              f"procs {self.procs}  backend numpy")
        for st in STAGES:
            if st in stages:
                for it in self.want_items(st):
                    key = ":".join(it)
                    rec = self.state["items"].get(key)
                    print(f"  {st:9} {key:16} {'cached' if rec and self._valid(rec) else 'run'}")

    # ---------------------------------------------------------------- driver
    def want_items(self, stage):
        shots = list(self.shots)
        return {"audio": [("audio",)], "stills": [("still", s) for s in self.stills], "type": [("type",)],
                "shots": [("shot", s) for s in shots], "alpha": [("alpha", s) for s in shots if self.shots[s].get("matte")],
                "depth": [("depth", s) for s in shots if self.shots[s].get("depth")],
                "masktrack": [("masktrack", s) for s in shots if self.shots[s].get("masktrack")],
                "flow": [("flow", s) for s in shots if self.rx is not None and self.shots[s].get("flow")],       # box mode only
                "compose": [("assemble",)], "final": [("final",)]}[stage]

    def call(self, it):
        return getattr(self, {"type": "typ"}.get(it[0], it[0]))(*it[1:])

    def run_stage(self, stage):
        return [self.call(it) for it in self.want_items(stage)]

    def go(self, stage="all"):
        t0 = time.time()
        self.log(f"=== {self.jpath.name} stage={stage} ===")
        stages = STAGES if stage == "all" else stage.split(",")
        errs = []
        threads = []
        results = {}
        if self.rx is not None:                       # box: code sync / pip / probe in the background while the GPU work starts
            if self.dry:
                self.rx.prepare()
            else:
                threading.Thread(target=self.rx.prepare, daemon=True, name="box-prepare").start()
            if self.dry:                              # --dry: walk the stages one after the other = a readable, ordered plan
                for st in STAGES:
                    if st in stages:
                        for it in self.want_items(st):
                            self.call(it)
                return True

        def runner(st, it):
            try:
                self.call(it)
            except BaseException as e:      # noqa: BLE001
                errs.append((st, it, e))

        # everything is lazy+memoised: start all leaf items concurrently, dependencies resolve themselves
        for st in stages:
            if st in ("compose", "final"):
                continue
            for it in self.want_items(st):
                th = threading.Thread(target=runner, args=(st, it), daemon=True)
                th.start()
                threads.append(th)
        for th in threads:
            th.join()
        if not errs and any(s in stages for s in ("compose", "final")):
            for st in ("compose", "final"):
                if st in stages:
                    for it in self.want_items(st):
                        try:
                            self.call(it)
                        except BaseException as e:      # noqa: BLE001
                            errs.append((st, it, e))
                            break
                if st == "compose":
                    self._drop_baton()                    # batch: the next piece may start rendering; final is CPU / io only
        self._drop_baton()
        self.summarise()
        self.log(f"=== finished in {time.time() - t0:.0f}s, {len(errs)} error(s) ===")
        for st, it, e in errs:
            self.log(f"ERROR {st} {it}: {e}")
        return not errs

    def summarise(self):
        with self.lock:
            for st in STAGES:
                recs = {k: v for k, v in self.state["items"].items() if stage_of(k) == st and not v.get("sub")}
                if recs:
                    self.state["stages"][st] = dict(
                        items=sorted(recs), wall_s=round(sum(v["wall_s"] for v in recs.values()), 1),
                        gpu_s=round(sum(v["gpu_s"] for v in recs.values()), 1),
                        outputs=[p for v in recs.values() for p in v["out"].get("paths", [])[:3]])
            self.save()
            (self.dir / "timings.json").write_text(json.dumps(
                {k: {"wall_s": v["wall_s"], "gpu_s": v["gpu_s"]} for k, v in self.state["stages"].items()}, indent=1), encoding="utf-8")
        if self.rx is not None and not self.dry:
            self.write_bench()


# =================================================================================================== timing report
def v002_baseline(piece_id):
    """summed wall seconds per stage (+ elapsed) of the v002 run of this piece, from its state.json. None when there is none."""
    sp = FUSION / "v002" / "local" / "pieces" / piece_id / "state.json"
    if not sp.exists():
        return None
    st = json.loads(sp.read_text(encoding="utf-8"))
    stages, t0s, t1s = {}, [], []
    for k, v in st["items"].items():
        d = stages.setdefault(stage_of(k), dict(wall_s=0.0, gpu_wait_s=0.0, n=0))
        d["wall_s"] += v["wall_s"]
        d["gpu_wait_s"] += v.get("gpu_s", 0.0)
        d["n"] += 1
        try:
            end = datetime.datetime.fromisoformat(v["at"])
            t1s.append(end)
            t0s.append(end - datetime.timedelta(seconds=v["wall_s"]))
        except Exception:       # noqa: BLE001
            pass
    elapsed = (max(t1s) - min(t0s)).total_seconds() if t0s else None
    return dict(stages=stages, elapsed_s=elapsed)


def bench_markdown(t, base, pid):
    """16_app/BENCH_<id>.md: this run vs v002, per stage and per item, plus the GPU idle estimate."""
    def mn(x):
        return "-" if x is None else f"{x / 60:.1f}"

    box = t["box"]
    L = [f"# {pid} bench: all heavy work on the box ({t['version']}) vs local CPU type + compose (v002)", "",
         f"Written {t['written']} by `tools/fusion/piece_runner.py --exec box`" + (" (--box-sim: stills / shots / mattes adopted from v002, "
                                                                                   "no GPU, local CPU stands in for the box)" if "sim" in t["exec"] else "") + ".",
         f"Box {box['iid']} {box.get('gpu') or ''} ({box.get('cores')} cores, NVENC {box.get('nvenc')}, compose backend {box.get('backend')}), "
         f"{'draft size (single-stage LTX + type 576x1024, compose 540x960)' if t.get('draft') else 'full size (type 1088x1920, compose 1080x1920)'}. "
         f"Box preparation (code sync + pip + probe, overlaps the first GPU work): {box.get('prepare_s')} s.", "",
         "Wall = summed over the stage's items (items overlap, so the sum is not elapsed time). v002 'GPU' = time the submit call waited "
         "(queue + run); v003 GPU exec = ComfyUI /history execution_start -> execution_success.", "",
         "| stage | v002 wall (min) | v003 wall (min) | v003 GPU exec (min) | v003 / v002 |", "|---|---:|---:|---:|---:|"]
    tot0 = tot1 = 0.0
    for st in STAGES:
        b = (base or {}).get("stages", {}).get(st)
        n = t["stages"].get(st)
        if not b and not n:
            continue
        w0, w1 = (b["wall_s"] if b else None), (n["wall_s"] if n else None)
        tot0 += w0 or 0.0
        tot1 += w1 or 0.0
        ratio = f"{w1 / w0:.2f}x" if w0 and w1 else "-"
        L.append(f"| {st} | {mn(w0)} | {mn(w1)} | {mn(n['gpu_exec_s']) if n and n['gpu_exec_s'] else '-'} | {ratio} |")
    L.append(f"| **sum of stage walls** | **{mn(tot0)}** | **{mn(tot1)}** | **{mn(t['gpu']['exec_s'])}** | **{tot1 / tot0:.2f}x** |" if tot0 else
             f"| **sum of stage walls** | - | **{mn(tot1)}** | **{mn(t['gpu']['exec_s'])}** | - |")
    g = t["gpu"]
    L += ["", "## Elapsed and GPU idle", "",
          "| | v002 | v003 |", "|---|---:|---:|",
          f"| elapsed, first item start -> last item end (min) | {mn((base or {}).get('elapsed_s'))} | {mn(g['run_window_s'])} |",
          f"| GPU executing, summed prompts (min) | - | {mn(g['exec_s'])} ({g['n_prompts']} prompts) |",
          f"| GPU busy, union of intervals (min) | - | {mn(g['busy_union_s'])} |",
          f"| GPU idle inside the run window = window - exec (min) | - | {mn(g['idle_in_run_s'])} ({g['idle_pct']}%) |"]
    if g.get("box_lifetime_s"):
        L.append(f"| box lifetime so far (dashboard started_running_at -> now) / idle = lifetime - exec (min) | - | "
                 f"{mn(g['box_lifetime_s'])} / {mn(g['idle_in_lifetime_s'])} |")
    L += ["", "FINDINGS.md: v002 GPU was busy ~1.5-2 h of 4 h 25 m box time (waiting on local type passes + compose).", "",
          "## v003 items", "", "| item | stage | wall (s) | GPU exec (s) | note |", "|---|---|---:|---:|---|"]
    for k, v in t["items"].items():
        note = []
        if v.get("sub_of"):
            note.append(f"inside {v['sub_of']}" + (f", {v['s_per_frame']} s/frame" if "s_per_frame" in v else ""))
        if v.get("remote_s"):
            note.append(f"on-box {v['remote_s']} s")
        L.append(f"| {k} | {v['stage']} | {v['wall_s']} | {v['gpu_exec_s'] or '-'} | {'; '.join(note)} |")
    return "\n".join(L) + "\n"


def ensure_version(ver):
    """15_fusion/<ver> must be a PROJECTS version before progress_standin --idea .../<ver> can join it: create it with proj.py when missing."""
    if (FUSION / ver / "version.json").exists():
        return
    prev = sorted(p.name for p in FUSION.glob("v[0-9][0-9][0-9]") if p.name < ver)
    cmd = [sys.executable, str(PROJ_PY), "version-start", "COMPETITIONS/motion-graphic/fusion", "--why",
           "all-on-box run of the fusion pieces (type passes, compose, assemble on the GPU box) + stage timing bench"]
    if prev:
        cmd += ["--parent", prev[-1]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:       # noqa: BLE001
        d = {}
    if not d.get("ok") or d.get("version") != ver:
        raise SystemExit(f"cannot start {ver} with proj.py (got {d.get('version')}): {(r.stdout or r.stderr)[-400:]}")
    print(f"started PROJECTS version {ver}: {d.get('dir')}", flush=True)


# =================================================================================================== graph helpers
def i2v_graph(prompt, image, frames, w, h, seed, fps=24, prefix=None, two_stage=True):
    """workflows/api/ltx_i2v.json patched via tools/ltx_graph.py; `frames` any 8k+1 (the length expression is rewritten).
    two_stage False (draft): one stage at w x h, no x2 upscale."""
    from tools import ltx_graph
    secs = (frames - 1) // fps if (frames - 1) % fps == 0 else 1
    g = ltx_graph.build("i2v", prompt, image, None, secs, fps, "9x16", w, h, seed, prefix, two_stage)
    if (frames - 1) % fps:
        g["398:362"]["inputs"]["value"] = frames - 1          # Duration node carries the frame count - 1 ...
        g["398:378"]["inputs"]["expression"] = "a * b / b + 1"  # ... and the expression is length = a + 1 (b = fps, used)
    return g


# =================================================================================================== compose (parallel)
def _refs(obj, acc):
    if isinstance(obj, str) and obj.startswith("$"):
        acc.add(obj[1:])
    elif isinstance(obj, dict):
        for v in obj.values():
            _refs(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _refs(v, acc)


def _subst(obj, m):
    if isinstance(obj, str):
        return m.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _subst(v, m) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, m) for v in obj]
    return obj


def _other_shot_ref(ref, sid, ids):
    """(target shot id, 'footage' | 'matte') for a two-shot compose input ($next, $prev, $shot:<id>; tools/fusion/window_cut.py), else None."""
    from tools.fusion import window_cut
    return window_cut.other_shot_ref(ref, sid, ids)


def _depth_norm(files, n=10):
    """(lo, hi) = 2nd / 98th percentile of the clip's depth values (VDA output only spans a slice of 0..1 per clip)."""
    import numpy as np
    from tools.fusion import compose
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int)
    v = np.concatenate([compose.imread(files[i], "gray")[::4, ::4].ravel() for i in idx])
    lo, hi = np.percentile(v, [2, 98])
    return [float(lo), float(max(hi, lo + 1e-3))]


class _ListSource:
    def __init__(self, files, kind, size, invert=False, norm=None):
        self.files, self.kind, self.size, self.invert, self.norm = files, kind, size, invert, norm
        self.n = len(files)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from tools.fusion import compose
        f = compose.imread(self.files[i], self.kind)
        if self.size is not None and f.shape[:2] != (self.size[1], self.size[0]):
            f = compose.fit_cover(f, self.size)
        if self.norm:
            f = (f - self.norm[0]) / (self.norm[1] - self.norm[0])
            f = f.clip(0.0, 1.0)
        return 1.0 - f if self.invert else f

    def close(self):
        pass


class _SignoffSource:
    """alpha (HxW) of a small mono caps line at the bottom, faded in/out over the window [t0, t1] (piece time)."""

    def __init__(self, spec, size, fps):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from mograph.modules import core
        self.size, self.fps, self.t0, self.t1, self.fade = size, fps, spec["t0"], spec["t1"], spec.get("fade", 0.35)
        key, _ = core.resolve_font(spec.get("font", "Space Grotesk"))
        font = ImageFont.truetype(core.ol.font_path(key), int(spec.get("px", 30)))
        text = spec["text"].upper()
        track = int(spec.get("px", 30) * 0.28)
        wds = [font.getlength(ch) for ch in text]
        tw = sum(wds) + track * (len(text) - 1)
        im = Image.new("L", size, 0)
        d = ImageDraw.Draw(im)
        x, y = (size[0] - tw) / 2.0, size[1] * 0.925
        for ch, w in zip(text, wds):
            d.text((x, y), ch, fill=255, font=font)
            x += w + track
        self.mask = np.asarray(im, np.float32) / 255.0
        self.n = int(1e6)
        self.zero = np.zeros_like(self.mask)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t = i / self.fps
        a = min(1.0, max(0.0, (t - self.t0) / self.fade)) * min(1.0, max(0.0, (self.t1 - t) / self.fade))
        return self.mask * (a * a * (3 - 2 * a)) if a > 0 else self.zero

    def close(self):
        pass


def _open_inputs(recipe):
    from tools.fusion import compose
    size = tuple(recipe["size"])
    srcs, offs, modes = {}, {}, {}
    for nm, sp in recipe["inputs"].items():
        fit = None if sp.get("native") else size
        if "signoff" in sp:
            srcs[nm] = _SignoffSource(sp["signoff"], size, recipe["fps"])
        elif "files" in sp:
            srcs[nm] = _ListSource(sp["files"], sp.get("kind", "gray"), fit, sp.get("invert", False), sp.get("norm"))
        elif "dir" in sp:
            files = sorted((str(p) for p in Path(sp["dir"]).glob("*.png")), key=natural)
            srcs[nm] = _ListSource(files, sp.get("kind", "gray"), fit, sp.get("invert", False))
        else:
            srcs[nm] = compose.FrameSource(sp["path"], sp.get("kind", "rgb"), fit)
        offs[nm] = int(sp.get("offset", 0))
        modes[nm] = sp.get("mode", "hold")
    return srcs, offs, modes


def _hold_index(recipe, nm, i, po, off):
    """freeze-frames (recipe["holds"] = [[t_s, dur_s], ..] in piece time). Footage-like inputs (path / files) stop dead for the hold
    and resume where they stopped; type-pass inputs (dir) hold the frame of the hold start and then jump back onto the music grid
    (holds_type, default true). Signoff inputs are untouched."""
    sp = recipe["inputs"][nm]
    if "signoff" in sp:
        return i + off
    fps = recipe["fps"]
    hs = [(int(round(float(a) * fps)), int(round(float(b) * fps))) for a, b in recipe["holds"]]
    p = i + po
    if "dir" in sp:
        if recipe.get("holds_type", True):
            for h0, hn in hs:
                if h0 <= p < h0 + hn:
                    return h0 + (off - po)
        return i + off
    fz = lambda q: sum(min(max(q - h0, 0), hn) for h0, hn in hs)
    return i + off - (fz(p) - fz(po))


def _chunk(job):
    """render frames [i0, i1) of a recipe (shot-local indices). trace_edges is warmed up on the 4 frames before i0."""
    recipe, i0, i1, out_dir = job
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from tools.fusion import compose
    reg = compose._ops_registry()
    srcs, offs, modes = _open_inputs(recipe)
    n = int(recipe["frames"])
    ops = recipe["ops"]
    states = {k: compose.EdgeTracer() for k, o in enumerate(ops) if o["op"] == "trace_edges"}
    pal = np.asarray(recipe["palette_arr"], np.float32) if recipe.get("palette_arr") else None
    po = int(recipe.get("piece_offset", 0))
    curves = recipe.get("curves", {})
    out_buf = recipe.get("output", "image")
    warm = 4 if states else 0
    t_acc = 0.0
    for i in range(max(0, i0 - warm), i1):
        t0 = time.perf_counter()
        ctx = {}
        for nm, s in srcs.items():
            j = i + offs[nm]
            if recipe.get("holds"):
                j = _hold_index(recipe, nm, i, po, offs[nm])
            L = len(s)
            j = j % L if modes[nm] == "loop" else min(max(j, 0), L - 1)
            ctx[nm] = s[j]
        ctx.update(frame_idx=i + po, t=(i + po) / recipe["fps"], u=i / max(n - 1, 1), n=n, piece_frame=i + po)
        if pal is not None:
            ctx["palette"] = pal
        u = ctx["u"]

        def res(v):
            if isinstance(v, str) and v.startswith("$"):
                return ctx[v[1:]]
            if isinstance(v, dict) and "anim" in v:
                return compose._anim(v, u)
            if isinstance(v, dict) and "timeline" in v:        # piece-time keys [[sec, value], ..]; "step": true -> hold value
                return compose._timeline(v, (i + po) / recipe["fps"])
            if isinstance(v, dict) and "curve" in v:
                c = curves[v["curve"]]
                return float(v.get("lo", 0.0) + (v.get("hi", 1.0) - v.get("lo", 0.0)) * c[min(max(i + po, 0), len(c) - 1)])
            if isinstance(v, list):
                return [res(x) for x in v]
            return v
        for k, op in enumerate(ops):
            args = {a: res(b) for a, b in op.get("args", {}).items()}
            if op["op"] == "trace_edges":
                fo = args.pop("footage")
                r = states[k].step(fo, **args)
            else:
                r = reg[op["op"]](**args)
            ctx[op.get("out", "image")] = r
        if i < i0:
            continue
        img = ctx[out_buf]
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        compose.imwrite(Path(out_dir) / f"{i:05d}.png", img)
        t_acc += time.perf_counter() - t0
    for s in srcs.values():
        s.close()
    return i1 - i0, t_acc


def _worker_init():
    os.environ["FUSION_BACKEND"] = "numpy"


def compose_parallel(recipe, out_dir, procs):
    """contiguous frame chunks over `procs` processes (numpy backend, spawn). -> mean busy seconds per frame per core."""
    import multiprocessing as mp
    n = int(recipe["frames"])
    size = max(4, int(math.ceil(n / (procs * 3))))
    jobs = [(recipe, a, min(a + size, n), str(out_dir)) for a in range(0, n, size)]
    done, busy = 0, 0.0
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_worker_init) as pool:
        for k, t in pool.imap_unordered(_chunk, jobs):
            done += k
            busy += t
    return busy / max(done, 1)


# =================================================================================================== on-box batch
_COMFY_BATON = threading.Lock()         # one piece at a time owns ComfyUI's VRAM: the on-box flow / compose jobs of piece N must not fight piece N+1's LTX model


def run_batch(a):
    """--on-box: run the pieces of `a.piece` pipelined. Piece N+1 starts (audio, stills and shots queue behind N in ComfyUI, type passes use the CPU)
    while N is still in compose / final; the GPU-heavy on-box jobs (flow, compose) of a piece run under _COMFY_BATON after ITS ComfyUI renders are
    done, and the next piece's first submit waits for that baton, so VRAM is never shared. -> all ok."""
    results, threads = {}, []

    def one(path, first):
        try:
            P = Piece(path, a.procs, exec_mode="box", dry=a.dry, draft=a.draft, backend=a.backend or "torch", preview=a.preview,
                      onbox=True, relay=a.relay)
            for f in a.force:
                P.forget(f)
            P.baton = _COMFY_BATON
            results[P.id] = P.go(a.stage)
        except BaseException as e:      # noqa: BLE001
            print(f"[batch] {path}: FAILED {type(e).__name__}: {e}", flush=True)
            results[str(path)] = False

    for i, path in enumerate(a.piece):
        th = threading.Thread(target=one, args=(path, i == 0), name=f"piece-{i}")
        th.start()
        threads.append(th)
        time.sleep(1.0)
    for th in threads:
        th.join()
    print("[batch] " + json.dumps(results), flush=True)
    return all(results.values())


# =================================================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("piece", nargs="+", help="piece json(s); several = a batch (--on-box pipelines them: footage of piece N+1 renders while N composes)")
    ap.add_argument("--on-box", action="store_true",
                    help="this runner IS on the box (started by tools/fusion/box_run.py): remote commands are local, ComfyUI at localhost:18188, "
                         "GPU submits go through --relay to the PC (progress_standin tiles)")
    ap.add_argument("--relay", default=None, help="--on-box: directory the submit requests / answers go through (default <root>/work/relay)")
    ap.add_argument("--stage", default="all", help="all | comma list of: " + ",".join(STAGES))
    ap.add_argument("--force", action="append", default=[], choices=STAGES, help="forget this stage (and its dependants) first")
    ap.add_argument("--procs", type=int, default=None, help="local: compose / type processes; box: type processes (default box cores - 2)")
    ap.add_argument("--dry", action="store_true", help="print the plan (box: every remote command, in order) and run nothing")
    ap.add_argument("--exec", dest="exec_mode", choices=["local", "box"], default="local",
                    help="local (default): type / compose / assemble on this PC; box: all of them on the GPU box")
    ap.add_argument("--out-version", default=None, help="15_fusion/<ver> for outputs (default v002 for --exec local, v003 for --exec box)")
    ap.add_argument("--iid", help="vast instance id (env MOGRAPH_BOX_IID; box mode without it: the dashboard's only running instance)")
    ap.add_argument("--host", help="box ssh host (env MOGRAPH_BOX_HOST; default: dashboard / Instance_Data lookup by --iid)")
    ap.add_argument("--ssh-port", type=int, help="box ssh port (env MOGRAPH_BOX_SSH_PORT)")
    ap.add_argument("--tunnel-port", type=int, help="local port of the ssh tunnel to the box's ComfyUI (env MOGRAPH_BOX_TUNNEL_PORT, default 9000)")
    ap.add_argument("--box-sim", metavar="DIR", help="simulate the box in DIR (remote commands run as local subprocesses; implies --exec box)")
    ap.add_argument("--draft", action="store_true", help="draft sizes (single-stage LTX 576x1024, type 576x1024, compose 540x960, x264 veryfast)")
    ap.add_argument("--preview", action="store_true", help="preview sizes (single-stage LTX 384x704 IC / 384x672 i2v, type 384x704, compose 360x640); not with --draft")
    ap.add_argument("--backend", default=None, help="box compose backend: torch (default) | auto | numpy ('auto' in --box-sim)")
    a = ap.parse_args(argv)
    if a.draft and a.preview:
        ap.error("--draft and --preview are mutually exclusive")
    if a.on_box:
        a.exec_mode = "box"
        BOX.update(tunnel_port=18188, iid=a.iid or os.environ.get("MOGRAPH_BOX_IID", BOX["iid"]))
        a.relay = a.relay or f"{BOX['root']}/work/relay"
    if len(a.piece) > 1 and not a.on_box:
        ap.error("several pieces need --on-box (use tools/fusion/box_run.py from the PC)")
    if a.box_sim:
        a.exec_mode = "box"
    ver = a.out_version or ("v003" if a.exec_mode == "box" else "v002")
    set_version(ver, root=(Path(a.box_sim).resolve() / "runner") if a.box_sim else None)
    if a.on_box:
        pass                                              # box facts are known; the PC launcher did resolve_box / ensure_version
    elif a.exec_mode == "box" and not a.box_sim and not a.dry:
        resolve_box(a.iid, a.host, a.ssh_port, a.tunnel_port, need_box=True)
    elif a.iid or a.host or a.ssh_port or a.tunnel_port or os.environ.get("MOGRAPH_BOX_IID"):
        resolve_box(a.iid, a.host, a.ssh_port, a.tunnel_port)
    if not a.dry and not a.box_sim and not a.on_box and ver != "v002":
        ensure_version(ver)
    os.environ.setdefault("FUSION_BACKEND", "numpy")
    if a.on_box:
        sys.exit(0 if run_batch(a) else 1)
    P = Piece(a.piece[0], a.procs, exec_mode=a.exec_mode, sim=a.box_sim, dry=a.dry, draft=a.draft, backend=a.backend, preview=a.preview)
    if a.dry and a.exec_mode == "local":
        P.plan_local(a.stage)
        sys.exit(0)
    if a.dry:
        P.log(f"=== DRY RUN: plan of {a.exec_mode} execution, out {ver}; box {BOX['iid']} root@{BOX['host']}:{BOX['ssh_port']}, tunnel 127.0.0.1:{BOX['tunnel_port']} ===")
    for f in a.force:
        P.log(f"force {f}")
        P.forget(f)
    ok = P.go(a.stage)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
