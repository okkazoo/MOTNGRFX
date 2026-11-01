"""Runtime-path audit: everything the ComfyUI nodes import must live in the repo and must not use absolute Windows paths.

Two checks, both run with cwd != repo:

1. ``test_no_absolute_paths_in_runtime_code``: static. Starting from the node entry points (``nodes.py``, ``nodes_fusion/*``,
   ``app_routes.py``) plus the dynamic-import packages (``mograph/modules``, ``mograph/render``) and the dimtype engine ``ol.py``
   (put on ``sys.path`` by ``mograph/render/cam3d.py``), follow ``import`` statements to every repo module and fail on any
   string literal that is an absolute drive path (``C:/...``, ``C:\\...``), a Git-Bash drive path (``/c/Users``) or a home path.
   Docstrings and comments are ignored (they may mention where something came from).
2. ``test_runtime_modules_import_from_foreign_cwd``: dynamic. In a subprocess with a temp cwd, import every non-ComfyUI runtime
   module (``mograph.*``, ``tools.fusion`` modules the nodes use, ``ol``) and assert all of them resolve inside the repo.
   The ComfyUI-dependent ``nodes_fusion`` modules are imported too when a ComfyUI checkout is found
   (``MOGRAPH_COMFYUI_PATH``, ``../..`` or sibling ``ComfyUI``), else that part is skipped.

Also asserts that the data files the nodes read at run time (module manifests, fonts, graphs) exist in the repo.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOP_PACKAGES = ("mograph", "tools", "nodes_fusion")
DIMTYPE = ROOT / "11_mathtype" / "engine" / "dimtype"

# string literals that mean "an absolute path of somebody's machine"
ABS_PATH = re.compile(r"""(?ix)
    (?:^|[\s"'=(,])[a-z]:[\\/]{1,2}(?:users|windows|program|dev|documents|temp)     # C:/Users/..., C:\\Program Files
  | ^[a-z]:[\\/]                                                                    # any literal starting with a drive
  | /c/users/ | /mnt/c/ | /home/[a-z0-9_.-]+/ | /Users/[a-z0-9_.-]+/
""")

# Not runtime: command-line / box-orchestration scripts and one-off study code that nodes never import.
NOT_RUNTIME = {
    "tools/fusion/piece_runner.py", "tools/fusion/cut_p06.py", "tools/fusion/box_stages.py", "tools/fusion/ltx_mask.py",
    "tools/fusion/ltx_ic.py", "tools/fusion/selfcheck_compose.py", "tools/fusion/songs.py", "tools/fusion/mask_preview.py",
}


def _py_files(sub: str):
    for p in sorted((ROOT / sub).rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def _module_file(name: str) -> Path | None:
    """Repo file of a top-level dotted module name (mograph.x.y, tools.fusion.z, nodes_fusion.a), else None."""
    parts = name.split(".")
    if parts[0] not in TOP_PACKAGES:
        return None
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _package_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        return ".".join(parts[:-1])
    return ".".join(parts[:-1])


def _imports(path: Path) -> set[str]:
    """Dotted repo-module names imported anywhere in ``path`` (also inside functions / try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pkg = _package_of(path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg.split(".") if pkg else []
                base_parts = base_parts[: len(base_parts) - (node.level - 1)] if node.level > 1 else base_parts
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                out.add(base)
                for a in node.names:
                    out.add(f"{base}.{a.name}")
    return out


def _closure(seeds: list[Path]) -> set[Path]:
    seen: set[Path] = set()
    todo = list(seeds)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        # importing a.b.c also runs a/__init__.py and a/b/__init__.py
        for name in _imports(p):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                f = _module_file(".".join(parts[:i]))
                if f is not None and f not in seen:
                    todo.append(f)
    return seen


def _runtime_files() -> set[Path]:
    seeds = [ROOT / "nodes.py", ROOT / "app_routes.py", ROOT / "__init__.py"]
    seeds += list(_py_files("nodes_fusion"))
    seeds += list(_py_files("mograph/modules")) + list(_py_files("mograph/render"))
    seeds += [DIMTYPE / "ol.py"]
    seeds += [ROOT / "tools" / "mathtype" / "audiofeat.py"]      # loaded by path in tools/fusion/music_analysis.py
    files = _closure([p for p in seeds if p.is_file()])
    spike = ROOT / "nodes_fusion" / "_spike.py"          # only loaded with MOGRAPH_SPIKE=1, still part of the pack
    files |= _closure([spike]) if spike.is_file() else set()
    return {p for p in files if p.relative_to(ROOT).as_posix() not in NOT_RUNTIME}


def _string_literals(path: Path):
    """(lineno, value) of every non-docstring string constant (f-string literal parts included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_ids.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            yield node.lineno, node.value


def test_runtime_closure_is_inside_repo_and_nonempty():
    files = _runtime_files()
    rel = {p.relative_to(ROOT).as_posix() for p in files}
    for must in ("nodes.py", "nodes_fusion/type.py", "mograph/render/cam3d.py", "11_mathtype/engine/dimtype/ol.py",
                 "tools/fusion/type_passes.py", "tools/fusion/plan_rules.py"):
        assert must in rel, f"{must} not reached by the import closure"


def test_no_absolute_paths_in_runtime_code():
    bad = []
    for p in sorted(_runtime_files()):
        for lineno, s in _string_literals(p):
            if len(s) < 200 and ABS_PATH.search(s):
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{lineno}: {s[:100]!r}")
    assert not bad, "absolute machine paths used at runtime (make them repo-relative):\n  " + "\n  ".join(bad)


def test_runtime_data_files_ship_in_repo():
    """Files the nodes read at run time and that an earlier hand-made pack was missing on the Linux box."""
    must = [
        "tools/ltx_graph.py", "tools/mathtype/__init__.py", "tools/mathtype/kit/__init__.py",
        "11_mathtype/engine/dimtype/ol.py", "14_modules/library/manifest.json",
        "mograph/fonts", "licenses", "LICENSE", "requirements.txt",
        "app/graphs/local_v1.json", "web_app/index.html",          # fusion.json / fusion.app.json are dev-only (the app loads motngrfx_* and _* graphs): not shipped in the public export
    ]
    missing = [m for m in must if not (ROOT / m).exists()]
    assert not missing, f"missing from the repo: {missing}"
    assert list((ROOT / "mograph" / "fonts").glob("*.[to]tf")), "no fonts in mograph/fonts"
    json.loads((ROOT / "14_modules" / "library" / "manifest.json").read_text(encoding="utf-8"))
    for p in (ROOT / "14_modules" / "library" / "manifest.d").glob("*.json"):
        json.loads(p.read_text(encoding="utf-8"))


def _comfy_path() -> str | None:
    cands = [os.environ.get("MOGRAPH_COMFYUI_PATH"), str(ROOT.parent.parent), str(ROOT.parent / "ComfyUI")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "comfy_api", "latest", "__init__.py")):
            return c
    return None


_CHILD = textwrap.dedent('''
    import importlib, json, os, pkgutil, sys
    root, comfy, with_comfy = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    assert os.path.realpath(os.getcwd()) != os.path.realpath(root), "cwd must not be the repo"
    sys.path.insert(0, root)                      # as ComfyUI would for a custom node dir
    if with_comfy:
        sys.path.insert(0, comfy)
    names = ["mograph"]
    import mograph
    for pkg in ("mograph", "mograph.render", "mograph.modules", "mograph.modules.words"):
        try:
            m = importlib.import_module(pkg)
        except ModuleNotFoundError:
            continue
        for _, n, _ in pkgutil.walk_packages(m.__path__, pkg + "."):
            names.append(n)
    names += ["tools.fusion.plan_rules", "tools.fusion.audio_cut", "tools.fusion.compose", "tools.fusion.recipe_eval",
              "tools.fusion.type_passes", "tools.fusion.latent_mask", "tools.ltx_graph"]
    failed = {}
    for n in names:
        try:
            importlib.import_module(n)
        except Exception as e:
            failed[n] = f"{type(e).__name__}: {e}"
    if with_comfy:
        for n in ("plan", "audio", "type", "compose", "inputs", "footage", "latent_mask", "routes"):
            try:
                importlib.import_module("nodes_fusion." + n)
            except Exception as e:
                failed["nodes_fusion." + n] = f"{type(e).__name__}: {e}"
    outside = sorted({os.path.abspath(m.__file__) for m in sys.modules.values()
                      if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(os.path.abspath(root))} )
    ol = sys.modules.get("ol")
    print(json.dumps({"failed": failed, "ol": getattr(ol, "__file__", None), "n": len(outside),
                      "fd": getattr(ol, "FD", None), "outdir": getattr(ol, "OUTDIR", None)}))
''')


@pytest.mark.parametrize("with_comfy", [False, True])
def test_runtime_modules_import_from_foreign_cwd(tmp_path, with_comfy):
    comfy = _comfy_path()
    if with_comfy and not comfy:
        pytest.skip("no ComfyUI checkout (set MOGRAPH_COMFYUI_PATH) - nodes_fusion modules need comfy_api")
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    work = tmp_path / "cwd"
    work.mkdir()
    proc = subprocess.run([sys.executable, str(script), str(ROOT), comfy or "", "1" if with_comfy else "0"],
                          cwd=work, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    failed = info["failed"]
    if not with_comfy:      # modules that need torch / comfy are allowed to fail on a bare interpreter
        failed = {k: v for k, v in failed.items() if "torch" not in v and "comfy" not in v and "folder_paths" not in v}
    assert not failed, f"imports failed with cwd != repo: {failed}"
    assert info["n"] > 10
    if info["ol"]:
        assert Path(info["ol"]).resolve() == (DIMTYPE / "ol.py").resolve()
        assert Path(info["fd"]).resolve() == (ROOT / "mograph" / "fonts").resolve()
        assert Path(info["outdir"]).resolve().is_relative_to(ROOT.resolve()), info["outdir"]


# Tools that used to name the author's home folder and so were once left out of the public export. They ship now:
# their defaults are built from Path.home() / env vars, and no source line may hard-code a user's home path again.
FORMERLY_PERSONAL = (
    "tools/fusion/piece_runner.py", "tools/fusion/cut_p06.py", "tools/finals_pipeline.py",
    "tools/qwen_brief_eval.py", "tools/sound_jobs.py",
)
_AUTHOR = "cra" + "ig"          # split so this file does not itself contain the path it forbids
HOME_PATH = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+" + _AUTHOR + r"|/c/users/" + _AUTHOR)


@pytest.mark.parametrize("rel", [pytest.param(r, marks=pytest.mark.skipif(r == "tools/qwen_brief_eval.py" and not (ROOT / r).exists(),
                                                                         reason=f"dev fixture not shipped in the public export: {r}"))
                                 for r in FORMERLY_PERSONAL])
def test_formerly_personal_tools_have_no_home_path(rel):
    path = ROOT / rel
    assert path.is_file(), f"{rel} is missing from the repo"
    hits = [f"{i}: {line.strip()[:90]}" for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if HOME_PATH.search(line)]
    assert not hits, f"{rel} hard-codes the author's home path:\n  " + "\n  ".join(hits)
