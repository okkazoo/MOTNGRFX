"""word_rules: the cut REPEAT RULE applied to the word-to-word transitions (user 2026-10-30: "just the variety of text choices and transitions can match the cuts").

    never the same transition on two consecutive word changes, and at most MAX_PER (2) uses of one transition per piece.

A "transition" of a gap = `contour_morph` when contour_morph.weave placed a morph there, else the module the type engine plays (the engine's own pick, or the explicit `type.transitions[i]` of the
model / a block): braid, circuit_board, fan, ripple, spiral_around, twist ... Plain `cut` and `pan` gaps are exempt (a pan is a cut kind and follows the cut rules). The word events of the contour
sweep carry the name of their gap's transition, so they follow the same list.

    enforce(piece, music, chk, size, designed=False)    THE HOOK: called by plan_rules._finish and design_step._finish right after contour_morph.weave (before contour_sweep.weave). Never raises.

A gap that breaks the rule takes, in order: (1) a morph there is dropped (plan.contour_morph_skip; contour_morph.plan honours it) and the engine's own pick for the gap plays; (2) when that still
breaks the rule, another module from the gap's engine RANKING (best fit first, so a specific `look` keeps its family: the engine ranks the transitions that suit those words) that is not the
neighbour's transition and is used least; written to `type.transitions[i]`. Fonts, layouts, hold life are not touched here. `plan.word_rules` keeps the result and the fixes.
"""
from __future__ import annotations

NAME = "word_rules"
MAX_PER = 2
MORPH = "contour_morph"
EXEMPT = (None, "", "cut", "pan", "none")


def _kind(tr):
    if isinstance(tr, dict):
        return tr.get("kind") or tr.get("module")
    return tr


def effective(piece: dict, names: list) -> list:
    """the transition of every gap: contour_morph where the morph plays, else the engine/explicit name from the dry run (`names`, one per gap)."""
    n = max(len(((piece.get("type") or {}).get("words") or [])) - 1, 0)
    morph = {int(x) for x in ((((piece.get("plan") or {}).get("contour_morph")) or {}).get("words") or [])}
    return [MORPH if i in morph else (names[i] if i < len(names) else None) for i in range(n)]


def violations(kinds: list) -> list:
    """[(gap index, why)] where a non-exempt kind repeats the gap before or is the third use in the piece."""
    out, seen = [], {}
    for i, k in enumerate(kinds):
        if k in EXEMPT:
            continue
        if i and kinds[i - 1] == k:
            out.append((i, "same as the gap before"))
        elif seen.get(k, 0) >= MAX_PER:
            out.append((i, f"third use of {k}"))
        seen[k] = seen.get(k, 0) + 1
    return out


def _dry(piece, chk):
    from tools.fusion import plan_rules as PR
    n = len(((piece.get("type") or {}).get("words") or [])) - 1
    if chk and chk.get("transitions") is not None and len(chk["transitions"]) == n:
        return chk
    return PR.dry_run(piece)


def _set_transition(piece, i, name):
    t = piece["type"]
    n = len(t.get("words") or []) - 1
    trs = t.get("transitions")
    if not isinstance(trs, list):
        trs = [None] * n
    trs = list(trs) + [None] * (n - len(trs))
    trs[i] = name
    t["transitions"] = trs


def enforce(piece: dict, music: dict | None = None, chk: dict | None = None, size=None, designed: bool = False) -> dict:
    pl = piece.setdefault("plan", {})
    try:
        words = (piece.get("type") or {}).get("words") or []
        if len(words) < 3:
            return piece
        from tools.fusion import contour_morph as CM
        from tools.fusion import plan_rules as PR
        pool = [n for n in ((piece.get("type") or {}).get("transition_names") or PR.transition_names()) if n != "cut"]
        fixes: list = []
        cur = chk
        for _ in range(4):
            dry = _dry(piece, cur)
            names = list(dry.get("transitions") or [])
            rank = list(dry.get("trans_rank") or [])
            kinds = effective(piece, names)
            if not violations(kinds):
                break
            changed = False
            skip = set(pl.get("contour_morph_skip") or [])
            for i in range(len(kinds)):
                k = kinds[i]
                if k in EXEMPT:
                    continue
                prev = kinds[i - 1] if i else None
                before = kinds[:i]
                if not (k == prev or before.count(k) >= MAX_PER):
                    continue
                if k == MORPH:                                     # (1) drop the morph: the engine's own pick for the gap plays
                    skip.add(i)
                    k = names[i] if i < len(names) else None
                    kinds[i] = k
                    fixes.append(f"gap {i}: contour_morph dropped (rule: not twice in a row, at most {MAX_PER} per piece); the engine's {k} plays")
                    changed = True
                    if k in EXEMPT or not (k == prev or before.count(k) >= MAX_PER):
                        continue
                nxt = kinds[i + 1] if i + 1 < len(kinds) else None
                others = kinds[:i] + kinds[i + 1:]
                cands = [c for c in ((rank[i] if i < len(rank) and rank[i] else []) + pool) if c not in EXEMPT]
                cands = list(dict.fromkeys(cands))
                good = [c for c in cands if c != prev and c != nxt and others.count(c) < MAX_PER] or [c for c in cands if c != prev and others.count(c) < MAX_PER]
                if not good:
                    continue
                new = min(good, key=lambda c: (others.count(c), good.index(c)))
                _set_transition(piece, i, new)
                skip.add(i)
                fixes.append(f"gap {i}: {k} -> {new} (rule: not twice in a row, at most {MAX_PER} per piece)")
                kinds[i] = new
                changed = True
            if not changed:
                break
            pl["contour_morph_skip"] = sorted(skip)
            CM.weave(piece, music, chk, size, designed)                # idempotent: the dropped morphs' ops go
            cur = None                                                 # the engine's later picks see the explicit names: read them again
        dry = _dry(piece, cur)
        final = effective(piece, list(dry.get("transitions") or []))
        pl["word_rules"] = dict(transitions=final, fixes=fixes, left=[f"gap {i}: {w}" for i, w in violations(final)])
    except Exception as e:      # noqa: BLE001  never break a piece for a rule
        pl["word_rules"] = dict(error=f"{type(e).__name__}: {str(e)[:160]}")
    return piece
