"""box_ssh persistent sessions, the delta code sync, box_poll (no box needed: a local `bash -s` stands in for `ssh host bash -s`)."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fusion import box_poll, box_ssh  # noqa: E402

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("tar") is None or shutil.which("base64") is None,
                                reason="needs bash + tar + base64 (Git Bash / Linux)")


@pytest.fixture()
def sess():
    s = box_ssh.Session([BASH, "-s"], start_timeout=30)
    yield s
    s.close()


def test_call_returns_rc_stdout_stderr(sess):
    rc, out, err = sess.call("echo hi; echo oops >&2; exit 3")
    assert (rc, out, err) == (3, b"hi\n", b"oops\n")


def test_quoting_survives_and_state_does_not_leak(sess, tmp_path):
    rc, out, _ = sess.call("cd " + tmp_path.as_posix() + " && printf '%s' \"a 'b' $HOME \\\"c\\\"\"")
    assert rc == 0 and b"a 'b'" in out and b'"c"' in out
    rc, out, _ = sess.call("pwd")
    assert out.decode().strip() != tmp_path.as_posix()            # each call is a fresh subshell


def test_commands_never_read_the_session_stdin(sess):
    rc, out, _ = sess.call("cat; echo after")                     # would eat the next command line without </dev/null
    assert (rc, out) == (0, b"after\n")
    assert sess.call("echo next")[1] == b"next\n"


def test_binary_output_is_byte_exact(sess, tmp_path):
    blob = bytes(range(256)) * 300
    f = tmp_path / "b.bin"
    f.write_bytes(blob)
    rc, out, _ = sess.call(f"cat {f.as_posix()}")
    assert rc == 0 and out == blob


def test_put_tar_stream_and_get_back(sess, tmp_path):
    big = os.urandom(300_000)
    blob = box_ssh.tar_bytes([("x/y/a.bin", big), ("x/b.json", b"{}")])
    dst = tmp_path / "dst"
    rc, _, err = sess.call(f"mkdir -p {dst.as_posix()} && head -c {len(blob)} | tar xzf - -C {dst.as_posix()}", data=blob)
    assert rc == 0, err
    assert (dst / "x" / "y" / "a.bin").read_bytes() == big
    assert sess.call("echo still-in-sync")[1] == b"still-in-sync\n"      # the payload did not desync the protocol
    rc, tarred, _ = sess.call(f"tar cf - -C {dst.as_posix()} x/y/a.bin")
    out = tmp_path / "got" / "a.bin"
    box_ssh.untar_to(tarred, [("x/y/a.bin", str(out))])
    assert out.read_bytes() == big


def test_timeout_kills_the_session_and_the_pool_replaces_it():
    pool = box_ssh.SessionPool([BASH, "-s"])
    try:
        assert pool.call("echo a")[1] == b"a\n"
        with pytest.raises(subprocess.TimeoutExpired):
            pool.call("sleep 5", timeout=0.5)
        assert pool.call("echo b")[1] == b"b\n"                   # a fresh session
    finally:
        pool.close()


def test_pool_runs_parallel_calls_on_separate_sessions():
    import threading
    pool = box_ssh.SessionPool([BASH, "-s"], max_sessions=4)
    res = []
    try:
        ts = [threading.Thread(target=lambda i=i: res.append(pool.call(f"sleep 0.4; echo {i}")[1])) for i in range(4)]
        t0 = time.time()
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert sorted(res) == [b"0\n", b"1\n", b"2\n", b"3\n"]
        assert time.time() - t0 < 6.0                             # not 4 x 0.4 serial + startup
    finally:
        pool.close()


def test_dead_ssh_raises_boxsshError():
    with pytest.raises(box_ssh.BoxSSHError):
        box_ssh.Session([sys.executable, "-c", "pass"], start_timeout=5)


def test_delta_sync_sends_only_changed_files(tmp_path, monkeypatch):
    from tools.fusion import piece_runner as PR
    src = tmp_path / "repo"
    for rel, txt in (("tools/a.py", "A"), ("mograph/b.py", "B"), ("requirements.txt", "r")):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(txt)
    monkeypatch.setattr(PR, "ROOT", src)
    logs = []
    rx = PR.Remote("t", sim=tmp_path / "sim", log=logs.append)
    monkeypatch.setattr(PR.Remote, "SYNC_DIRS", ["tools", "mograph"])
    monkeypatch.setattr(PR.Remote, "SYNC_FILES", ["requirements.txt"])
    rx.sync_code()
    assert any("3 of 3 files changed" in m for m in logs)
    assert (Path(rx.root) / "tools" / "a.py").read_text() == "A"
    logs.clear()
    rx.sync_code()
    assert any("is current" in m for m in logs)
    (src / "tools" / "a.py").write_text("A2")
    (src / "mograph" / "b.py").unlink()
    logs.clear()
    rx.sync_code()
    assert any("1 of 2 files changed, 1 removed" in m for m in logs), logs
    assert (Path(rx.root) / "tools" / "a.py").read_text() == "A2"
    assert not (Path(rx.root) / "mograph" / "b.py").exists()


def test_box_poll_returns_a_request_once_and_new_log_text(tmp_path, capsys):
    rd = tmp_path / "relay"
    rd.mkdir()
    (rd / "P1_x_1.req.json").write_text(json.dumps(dict(part="P1_x_1", graph={})))
    log, done = tmp_path / "run.log", tmp_path / "run.done"
    log.write_bytes(b"line1\n")
    argv = ["--relay", str(rd), "--log", str(log), "--done", str(done), "--timeout", "1"]
    box_poll.main(argv)
    d = json.loads(capsys.readouterr().out)
    assert [q["name"] for q in d["reqs"]] == ["P1_x_1"] and d["log"] == "line1\n" and d["done"] is None
    assert (rd / "P1_x_1.req.taken").exists() and not (rd / "P1_x_1.req.json").exists()
    done.write_text("0\n")
    log.write_bytes(b"line1\nline2\n")
    box_poll.main(argv + ["--offset", str(d["offset"])])
    d2 = json.loads(capsys.readouterr().out)
    assert d2["reqs"] == [] and d2["log"] == "line2\n" and d2["done"] == 0


def test_put_many_and_get_many_in_sim_mode(tmp_path):
    from tools.fusion import piece_runner as PR
    rx = PR.Remote("t", sim=tmp_path / "sim", log=lambda m: None)
    a = tmp_path / "a.txt"
    a.write_text("hello")
    rx.put_many([(a, f"{rx.work}/a.txt"), (b'{"k": 1}', f"{rx.work}/sub/j.json")])
    assert Path(rx.work, "a.txt").read_text() == "hello" and json.loads(Path(rx.work, "sub", "j.json").read_text()) == {"k": 1}
    rx.get_many([(f"{rx.work}/a.txt", tmp_path / "out" / "a2.txt")])
    assert (tmp_path / "out" / "a2.txt").read_text() == "hello"


def test_relay_submit_roundtrip_and_baton(tmp_path):
    """the on-box runner writes <part>.req.json and blocks on <part>.resp.json; a second piece waits for the ComfyUI baton."""
    import threading
    from tools.fusion import piece_runner as PR
    logs = []

    def piece(pid):
        P = PR.Piece.__new__(PR.Piece)
        P.id, P.relay, P.log, P.onbox = pid, str(tmp_path / "relay"), logs.append, True
        P.baton, P._have_baton, P._baton_mx = PR._COMFY_BATON, False, threading.Lock()
        return P

    A, B = piece("PA"), piece("PB")
    pidf = tmp_path / "x.prompt_id"

    def answer():
        rd = tmp_path / "relay"
        for _ in range(100):
            req = rd / "PA_still_A_1.req.json"
            if req.exists():
                d = json.loads(req.read_text())
                assert d["graph"] == {"1": {}} and d["piece"] == "PA"
                (rd / "PA_still_A_1.resp.json").write_text(json.dumps(dict(ok=True, tail=[], log=["DONE x"], prompt_id="abc")))
                return
            time.sleep(0.05)
    th = threading.Thread(target=answer)
    th.start()
    A._take_baton()
    ok, tail = A._relay_submit("PA_still_A_1", "why", {"1": {}}, pidf)
    th.join()
    assert ok and pidf.read_text() == "abc" and any("DONE x" in m for m in logs)
    got = []
    tb = threading.Thread(target=lambda: (B._take_baton(), got.append(1)))
    tb.start()
    time.sleep(0.3)
    assert not got                                         # B waits while A holds the baton
    A._drop_baton()
    tb.join(timeout=5)
    assert got == [1]
    B._drop_baton()


# ----------------------------------------------------------------------------------------------- plates on the box (contour ports)
def _plates(tmp, names=("a", "b")):
    out = []
    for n in names:
        d = tmp / "repo" / "footage"
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for k in ("plate", "matte", "depth"):
            (d / f"{n}_{k}.mp4").write_bytes(f"{n}-{k}".encode() * 50)
            files[k] = f"footage/{n}_{k}.mp4"
        out.append(dict(files, frames=24))
    return out


def test_plate_paths_map_to_box_paths_independent_of_the_host():
    r = box_ssh.plate_box_rel
    assert r("15_fusion/v005/out/footage/P18/P18_s1_g01.mp4") == "15_fusion/v005/out/footage/P18/P18_s1_g01.mp4"      # repo-relative: same place
    a = r("C:/Users/someone/Documents/x/plate.mp4")
    assert a.startswith("work/plates_ext/") and a.endswith("/plate.mp4") and a == r(r"C:\Users\someone\Documents\x\plate.mp4")
    assert r("/data/other/plate.mp4") != a and r("../up/plate.mp4").startswith("work/plates_ext/")
    plates = [dict(plate="p/a.mp4", matte="C:/m/b.mp4", depth=None, frames=9)]
    new = box_ssh.rewrite_plates(plates, "/workspace/mograph")
    assert new[0]["plate"] == "/workspace/mograph/p/a.mp4" and new[0]["matte"].startswith("/workspace/mograph/work/plates_ext/")
    assert new[0]["depth"] is None and new[0]["frames"] == 9 and plates[0]["plate"] == "p/a.mp4"          # a copy, frames kept


def test_declares_ports_and_missing_plates(tmp_path):
    assert box_ssh.declares_ports({"words": [{"entry": {"port": "luma_edges"}}]}) and box_ssh.declares_ports({"words": [{"via": {"port": "x"}}]})
    assert not box_ssh.declares_ports({"words": [{"entry": "up", "exit": "down"}]})
    pl = _plates(tmp_path)
    pairs, missing = box_ssh.plate_pairs(pl, tmp_path / "repo")
    assert len(pairs) == 6 and missing == [] and pairs[0][1] == "footage/a_plate.mp4"
    (tmp_path / "repo" / "footage" / "b_depth.mp4").unlink()
    pairs, missing = box_ssh.plate_pairs(pl, tmp_path / "repo")
    assert len(pairs) == 5 and missing == ["footage/b_depth.mp4"]


def test_plates_ship_once_by_hash_and_the_runner_rewrites_the_paths(tmp_path, monkeypatch):
    from tools.fusion import piece_runner as PR
    repo = tmp_path / "repo"
    pl = _plates(tmp_path)
    logs = []
    rx = PR.Remote("t", sim=tmp_path / "sim", log=logs.append)
    pairs, _ = box_ssh.plate_pairs(pl, repo)
    put = [(loc, f"{rx.root}/{rel}") for loc, rel in pairs]
    d1 = box_ssh.delta_put(rx, put)
    assert d1["sent"] == 6 and d1["skipped"] == 0
    assert (Path(rx.root) / "footage" / "a_plate.mp4").read_bytes() == b"a-plate" * 50
    d2 = box_ssh.delta_put(rx, put)                                              # second run: nothing travels
    assert d2 == dict(sent=0, skipped=6, bytes=0)
    (repo / "footage" / "a_matte.mp4").write_bytes(b"changed")
    (Path(rx.root) / "footage" / "b_depth.mp4").unlink()                         # gone from the box: re-sent even though the hash matches
    d3 = box_ssh.delta_put(rx, put)
    assert d3["sent"] == 2 and d3["skipped"] == 4
    assert (Path(rx.root) / "footage" / "a_matte.mp4").read_bytes() == b"changed"

    class Stub:                                                                  # Piece.box_plates needs only these
        id, dry = "T", False
        log = staticmethod(lambda m: logs.append(m))
        j = {"type": {"words": [{"entry": {"port": "luma_edges", "dir": "up"}}]}}
    Stub.rx = rx
    monkeypatch.setattr(PR, "ROOT", repo)
    new = PR.Piece.box_plates(Stub, pl)                                          # PC-driven box run: ships (nothing new to send), rewrites
    assert new[0]["plate"] == f"{rx.root}/footage/a_plate.mp4" and all(os.path.exists(p) for _, _, p in box_ssh.plate_strings(new))
    assert any("plates -> box: 0 sent" in m for m in logs)
    (repo / "footage" / "b_plate.mp4").unlink()
    with pytest.raises(box_ssh.PlatesMissing, match="b_plate.mp4"):              # ports declared + a plate missing on the PC: loud, before anything runs
        PR.Piece.box_plates(Stub, pl)

    class OnBox:                                                                 # the on-box runner: files must already be there
        onbox, sim, root = True, None, rx.root
    Stub.rx = OnBox
    (Path(rx.root) / "footage" / "b_plate.mp4").unlink()
    with pytest.raises(box_ssh.PlatesMissing, match="b_plate.mp4"):              # ... missing on the box: loud, not a silent fallback to module sides
        PR.Piece.box_plates(Stub, pl)
    Stub.j = {"type": {"words": [{"entry": "up"}]}}
    assert PR.Piece.box_plates(Stub, pl)[0]["plate"].endswith("a_plate.mp4")     # no ports declared: no complaint
