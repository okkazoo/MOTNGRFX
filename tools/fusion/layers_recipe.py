"""layers_recipe - compose side of the THREE-LAYER shot route (16_app/LAYERS.md).

A layered shot has two footage clips that share one scene and one camera:
    $footage  the BACKGROUND: the text-rendered clip (LTX union-control V2V of the clean clip), or the clean clip itself when the shot has no text pass
    $subject  the CLEAN clip (plain image-to-video, no type guide): the pixels the Alpha-Gen matte was made from
Every compose recipe in the repo is written against ONE clip (`$footage`) and takes "the subject" from the plate it draws the type onto
(`sandwich` with no `fg=` shows `footage` again over the type; `knockout` / `window_cut` step the subject out of the plate; `depth_occlude`
leaves the covering plate pixels). With a text-rendered background that would put the LETTERS-IN-THE-SCENE subject on top of the type, and the
matte (cut from the clean clip) would not even fit it. `layerize_ops` rewrites a built op list so that every such op takes the subject
from the clean clip:

    sandwich (not inverted)   fg = subject version of the plate (the `grade` chain that made `$foot` is cloned onto `$subject`, same grade)
    depth_occlude             fg = subject version of the plate (new arg): covering pixels come from the clean clip
    knockout (fg_matte)       fg = subject version of fg / footage
    window_cut                fg_matte -> fg as above; otherwise a_subject / b_subject (new args) = subject versions of a / b
                              (`$next` -> `$next_subject`, `$prev` -> `$prev_subject`, `$shot:<id>` -> `$shot:<id>:subject`)
    upscale_matte             guide = subject version (the matte was cut from the clean clip, so its edges follow those pixels)

Not touched on purpose: an INVERTED sandwich (its `fg` is the background drawn over the type), composite buffers (`$image` ...: they already
carry the subject from an earlier layered op), type colour sampling (`tint_type`, `legible_color`: the background is what the type sits on).
Not supported (reported in `report["skipped"]`): plate chains through stateful ops (flow_wake, hold_net, legible_color, trace_edges, legible,
multiplane) cannot be cloned, the op keeps its single-clip behaviour.

The result is a pure function of the op list: `layerize_ops(ops) -> (ops2, report)`; an op list without a subject-taking op comes back unchanged.
"""
from __future__ import annotations

import copy

SUBJECT = "subject"
# plate processing ops that are cloned onto the subject (stateless, input -> same-size image)
CLONABLE = {"grade", "halation", "vignette", "grain", "blur", "translate", "fit_cover", "crossfade", "add_glow"}
# ops whose output already carries the subject (a compositing op ran before) and ops with per-frame state (cannot be run twice on one clip)
COMPOSITE = {"sandwich", "knockout", "window_cut", "depth_occlude", "over", "overlay", "screen", "multiply", "soft_light", "refract", "luminosity_key",
             "type_from_edges", "edges_in_type", "rgba_part"}
STATEFUL = {"flow_wake", "hold_net", "legible_color", "trace_edges", "multiplane", "legible", "rigid_pose", "plate_pan"}
OTHER_SUBJECT = ("$next", "$prev")


class _NoClone(Exception):
    pass


def _walk_refs(obj, acc: set) -> None:
    if isinstance(obj, str) and obj.startswith("$"):
        acc.add(obj[1:])
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_refs(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_refs(v, acc)


def other_subject_ref(ref):
    """'$next' -> '$next_subject', '$prev' -> '$prev_subject', '$shot:<id>' -> '$shot:<id>:subject'; None for anything else."""
    if not isinstance(ref, str):
        return None
    if ref in OTHER_SUBJECT:
        return ref + "_subject"
    if ref.startswith("$shot:") and ref.count(":") == 1:
        return ref + ":subject"
    return None


class _Layerizer:
    def __init__(self, ops):
        self.ops = ops
        self.after: dict[int, list] = {}           # def index -> cloned ops inserted right after it
        self.before: dict[int, list] = {}          # op index -> ops inserted right before it (subject re-insertion for a plate blended into a composite)
        self.last_sandwich = None                  # (matte, matte_threshold, matte_soft, type_a) of the last layered sandwich seen
        self.clone_of: dict[int, str] = {}
        self.skipped: list[str] = []
        self._lin: dict = {}

    # ------------------------------------------------------------------ lineage
    def last_def(self, name: str, j: int):
        for d in range(j - 1, -1, -1):
            if self.ops[d].get("out", "image") == name:
                return d
        return None

    def lineage(self, name: str, j: int) -> bool:
        """does buffer `name` as defined before op j derive from $footage?"""
        if name == "footage":
            return True
        d = self.last_def(name, j)
        if d is None:
            return False
        if d in self._lin:
            return self._lin[d]
        acc: set = set()
        _walk_refs(self.ops[d].get("args", {}), acc)
        r = any(self.lineage(a, d) for a in acc)
        self._lin[d] = r
        return r

    # ------------------------------------------------------------------ subject version of a plate buffer
    def subject_of(self, name: str, j: int):
        """name of the buffer holding `name` computed from $subject instead of $footage (cloned grade chain), or None when not possible."""
        if name == "footage":
            return SUBJECT
        d = self.last_def(name, j)
        if d is None or not self.lineage(name, j):
            return None
        if d in self.clone_of:
            return self.clone_of[d]
        op = self.ops[d]
        if op["op"] not in CLONABLE:
            return None
        args = copy.deepcopy(op.get("args", {}))

        def rn(v):
            if isinstance(v, str) and v.startswith("$"):
                r = v[1:]
                if self.lineage(r, d):
                    s = self.subject_of(r, d)
                    if s is None:
                        raise _NoClone(r)
                    return "$" + s
                return v
            if isinstance(v, dict):
                return {k: rn(x) for k, x in v.items()}
            if isinstance(v, list):
                return [rn(x) for x in v]
            return v
        try:
            args = rn(args)
        except _NoClone:
            return None
        out = f"{name}__s{d}"
        self.clone_of[d] = out
        self.after.setdefault(d, []).append({"op": op["op"], "out": out, "args": args})
        return out

    def ref_subject(self, v, j: int):
        """arg value (a '$ref' string) -> the '$ref' of its subject version, or None."""
        if not (isinstance(v, str) and v.startswith("$")):
            return None
        o = other_subject_ref(v)
        if o:
            return o
        s = self.subject_of(v[1:], j)
        return "$" + s if s else None

    # ------------------------------------------------------------------ per op
    def edit(self, j: int, op: dict) -> dict:
        name, a = op["op"], op.get("args", {})
        new = {}
        if name == "sandwich":
            inv = a.get("invert", False)
            if inv is False or inv is None or inv == 0:
                src = a.get("fg") if a.get("fg") is not None else a.get("footage")
                s = self.ref_subject(src, j)
                if s and s != src:
                    new["fg"] = s
                    self.last_sandwich = (a.get("matte"), a.get("matte_threshold"), a.get("matte_soft"), a.get("type_a"))
                elif s is None:
                    why = self._why_not(src, j)
                    if why:
                        self.skipped.append(f"op {j} sandwich: plate {src} not layered ({why})")
        elif name == "depth_occlude":
            s = self.ref_subject(a.get("footage"), j)
            if s and a.get("fg") is None:
                new["fg"] = s
        elif name == "knockout":
            if a.get("fg_matte") is not None:
                src = a.get("fg") if a.get("fg") is not None else a.get("footage")
                s = self.ref_subject(src, j)
                if s and s != src:
                    new["fg"] = s
        elif name == "window_cut":
            if a.get("fg") is not None:                      # explicit subject pixels: clone their grade chain
                s = self.ref_subject(a["fg"], j)
                if s and s != a["fg"]:
                    new["fg"] = s
            else:                                            # the op picks plate / window by phase: give it the subject version of both
                for side in ("a", "b"):
                    s = self.ref_subject(a.get(side), j)
                    if s and a.get(side + "_subject") is None:
                        new[side + "_subject"] = s
        elif name == "crossfade":
            # a plate (graded background) blended into a composite (`wipe` brighten): the blend would put the background's own subject pixels back
            # over the clean subject. Re-insert the subject into the plate first: sandwich(plate, no type, matte, fg=subject version of the plate).
            b = a.get("b")
            if self.last_sandwich and isinstance(b, str) and b.startswith("$") and self._why_not(a.get("a"), j) is None and self._composite(a.get("a"), j):
                sb = self.ref_subject(b, j)
                if sb and sb != b:
                    m, thr, soft, ta = self.last_sandwich
                    args = {"footage": b, "fg": sb, "type_rgb": [0.0, 0.0, 0.0], "type_a": ta, "type_opacity": 0.0, "matte": m}
                    if thr is not None:
                        args.update(matte_threshold=thr, matte_soft=soft if soft is not None else 0.08)
                    nm = f"{b[1:]}__L{j}"
                    self.before.setdefault(j, []).append({"op": "sandwich", "out": nm, "args": args})
                    new["b"] = "$" + nm
        elif name == "upscale_matte":
            s = self.ref_subject(a.get("guide"), j)
            if s and s != a.get("guide"):
                new["guide"] = s
        if not new:
            return op
        op2 = copy.deepcopy(op)
        op2["args"].update(new)
        return op2

    def _composite(self, ref, j: int) -> bool:
        d = self.last_def(ref[1:], j) if isinstance(ref, str) and ref.startswith("$") else None
        return d is not None and self.ops[d]["op"] in COMPOSITE

    def _why_not(self, ref, j: int):
        """why `ref` has no subject version: None (it is a composite buffer, which already carries the subject, or not footage-derived) or a reason."""
        if not (isinstance(ref, str) and ref.startswith("$")) or not self.lineage(ref[1:], j):
            return None
        d = self.last_def(ref[1:], j)
        if d is None:
            return None
        o = self.ops[d]["op"]
        if o in COMPOSITE:
            return None
        if o in STATEFUL:
            return f"{o} keeps per-frame state"
        return f"op {o} is not a clonable plate op"


def layerize_ops(ops: list) -> tuple[list, dict]:
    """op list of ONE shot (after build_recipe's passes) -> (ops with the subject taken from `$subject`, report). `$subject` is a new recipe
    input (the clean clip); the op list is returned unchanged (and report['changed'] is 0) when no op takes the subject."""
    L = _Layerizer(ops)
    edited = [L.edit(j, op) for j, op in enumerate(ops)]
    out = []
    for j, op in enumerate(edited):
        out.extend(L.before.get(j, []))
        out.append(op)
        out.extend(L.after.get(j, []))
    n_changed = sum(1 for a, b in zip(ops, edited) if a is not b)
    return out, {"changed": n_changed, "clones": sum(len(v) for v in L.after.values()) + sum(len(v) for v in L.before.values()), "skipped": L.skipped}


def uses_subject(ops: list) -> bool:
    acc: set = set()
    for o in ops:
        _walk_refs(o.get("args", {}), acc)
    return any(r == SUBJECT or r.endswith("_subject") or r.endswith(":subject") for r in acc)
