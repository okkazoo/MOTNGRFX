"""user_style: the USER's explicit style wishes, read from what the user wrote (2026-11-01, user: "if people are specifically writing different desires as edits
to the initial default creation, asking for more legible text and a certain design colour palette, then those override the underlying default").

    parse_direction(src) -> {"palette": [[r, g, b] 0..255, ..] | None, "legible_boost": bool, "plain": bool}
    stamp(piece, text)   -> piece with piece["direction"] = the typed text and piece["user_style"] = parse_direction(text)   (the ONE place a route lands the user's text on a piece)
    split_typed(text) -> (words_text, direction_text): the ONE splitter of the typed text (directive clauses are never on-screen words); words_of(text) = its words
    direction_of(owners) -> the typed text carried in the concept's request extras (image mode: concept.request_extras -> MographRequest.owners)

src: a piece (dict with `user_style` / `direction` / `request` / `brief`), a brief dict, or plain text. Only USER-AUTHORED text counts, never the model's reading of the media,
never lyrics or auto-written words:
    piece["user_style"] (an explicit dict, wins) | request / brief fields `direction`, `user_text`, `text` | the directive lines of a raw words box (top-level `words` only: the
    `words` of a piece / request are the piece's own on-screen words, never read).
CONSERVATIVE on purpose: the typed text is often the words to show too ("GOLDEN HOUR"), so a colour NAME counts only inside an explicit palette phrase:
    "palette: crimson, gold" / "colours: ..." lines, "palette of red and gold", "crimson and gold palette|tones|scheme", "make it / text / words ... in red and gold".
    hex: 6-digit anywhere ("#ff8800"); 3-digit only on a palette line. A bare "GOLDEN HOUR", "golden hour in the city", "ROSE" or "red sky" never sets a palette.
legible_boost: "legible / legibility / readable / more readable / clear(er) text / bigger|larger text / easier to read" -> run time raises the floors (recipe_eval: bigger min cap,
contrast 4.5 x 1.2, outline on, no echo / wake, no hold-life shimmer). plain: "keep|make it plain|simple|minimal|clean", "plain|simple|minimal text|look|style|design", "no effects / echo / trails / shimmer".
Pure python, no numpy."""
from __future__ import annotations

import json
import re

MAX_DIRECTION = 2000
NAMES = {
    "red": (214, 40, 40), "crimson": (190, 20, 50), "scarlet": (230, 40, 30), "orange": (245, 130, 30), "amber": (255, 180, 40), "gold": (230, 180, 40),
    "golden": (230, 180, 40), "yellow": (250, 220, 50), "lime": (190, 235, 50), "green": (50, 170, 80), "emerald": (20, 160, 110), "teal": (20, 150, 150),
    "cyan": (40, 210, 230), "aqua": (70, 230, 210), "turquoise": (60, 210, 200), "blue": (40, 100, 220), "navy": (20, 35, 90), "cobalt": (0, 70, 170),
    "indigo": (75, 40, 160), "purple": (130, 50, 170), "violet": (150, 80, 220), "magenta": (220, 40, 160), "pink": (250, 120, 180), "rose": (235, 90, 120),
    "coral": (250, 120, 90), "peach": (255, 190, 150), "cream": (245, 235, 210), "ivory": (250, 245, 230), "white": (245, 245, 245), "black": (12, 12, 14),
    "silver": (190, 195, 200), "grey": (128, 128, 128), "gray": (128, 128, 128), "brown": (120, 75, 40), "rust": (170, 70, 30), "olive": (120, 130, 40),
    "mint": (150, 235, 190), "lavender": (190, 170, 235), "neon": (60, 255, 120),
}
_HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_HEX6 = re.compile(r"#[0-9a-fA-F]{6}\b")
_NAME = "(?:" + "|".join(sorted(NAMES, key=len, reverse=True)) + ")"
_NAME_RE = re.compile(rf"\b({_NAME})\b", re.I)
_SEP = r"(?:\s*,\s*(?:and\s+)?|\s*/\s*|\s*&\s*|\s*\+\s*|\s+and\s+|\s+with\s+)"
_LIST = rf"\b{_NAME}\b(?:{_SEP}\b{_NAME}\b)*"
_P_LINE = re.compile(r"^[ \t]*(?:palette|colou?rs?)[ \t]*:[ \t]*(.+)$", re.I | re.M)                                   # "palette: crimson, gold" (a whole line)
_P_CUE = re.compile(rf"\b(?:palette|colou?rs?|tones?|hues?)\b\s*(?:of|is|are|being|=|:|-)?\s*(?:a\s+|the\s+|only\s+)?({_LIST})", re.I)   # "palette of red", "colours: red, gold"
_P_TAIL = re.compile(rf"({_LIST})\s+(?:colou?r\s+)?(?:palette|colou?rs?|scheme|tones?|theme)\b", re.I)                     # "a crimson and gold palette"
_P_IN = re.compile(rf"\b(?:make|do|render|set|keep|show|write|type|use|using|go|style|styled|colou?red|paint|painted|text|words?|letters)\b[^.\n]{{0,24}}?\bin\s+(?:an?\s+)?({_LIST})", re.I)   # "make it in red and gold"
_P_BARE = re.compile(rf"(?:^|[,;.]\s*)(?:in|using|with)\s+({_NAME}(?:{_SEP}{_NAME})+)[ \t]*[.!]?[ \t]*$", re.I | re.M)                   # a clause that is only "in red and gold" (two colours or more)
_DIRECTIVE = re.compile(r"^\s*(palette|colou?rs?|style|look|text|type|font|direction)\s*:", re.I)
_LEGIBLE = re.compile(r"\b(more\s+)?(legib\w*|readab\w*|clear(er)?\s+(text|type|words|letters)|(bigger|larger)\s+(text|type|words|letters|font)|easier\s+to\s+read)\b", re.I)
_PLAIN = re.compile(r"\b((keep|make|go|stay|want|prefer|have)\s+(it\s+|this\s+|the\s+\w+\s+|everything\s+)?(very\s+|really\s+|more\s+)?(plain|simple|simpler|minimal(ist)?|clean)\b"
                    r"|(plain|simple|minimal(ist)?)\s+(text|type|look|style|design|version)\b|no\s+(effects?|echo(es)?|trails?|shimmer)\b|clean\s+(text|type)\b)", re.I)
FIELDS = ("direction", "user_text", "text")


def _hex(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _texts(src) -> list:
    """the user-authored text pieces of src."""
    if src is None:
        return []
    if isinstance(src, str):
        return [src]
    if not isinstance(src, dict):
        return []
    out = []
    for i, blk in enumerate((src, src.get("request") if isinstance(src.get("request"), dict) else None, src.get("brief") if isinstance(src.get("brief"), dict) else None)):
        if not blk:
            continue
        for k in FIELDS:
            if isinstance(blk.get(k), str) and blk[k].strip():
                out.append(blk[k])
        w = blk.get("words")
        if i == 0 and isinstance(w, str):                                  # a raw words box: only its directive lines (the rest are the words / lyrics); a piece's / request's `words` are never read
            out += [ln for ln in w.splitlines() if _DIRECTIVE.match(ln)]
    return out


def colours_in(text: str) -> list:
    """[[r, g, b], ..] the text names, in reading order, duplicates dropped: 6-digit hex anywhere; colour names (and 3-digit hex) only inside an explicit palette phrase (see the module doc)."""
    text = text or ""
    found = []                                                            # (position, rgb)
    for m in _HEX6.finditer(text):
        found.append((m.start(), _hex(m.group(0))))
    for m in _P_LINE.finditer(text):
        for h in _HEX.finditer(m.group(1)):
            found.append((m.start(1) + h.start(), _hex(h.group(0))))
        for nm in _NAME_RE.finditer(text, m.start(1), m.end(1)):
            found.append((nm.start(), list(NAMES[nm.group(1).lower()])))
    for rx in (_P_CUE, _P_TAIL, _P_IN, _P_BARE):
        for m in rx.finditer(text):
            for nm in _NAME_RE.finditer(text, m.start(1), m.end(1)):
                found.append((nm.start(), list(NAMES[nm.group(1).lower()])))
    found.sort(key=lambda x: x[0])
    seen, res = set(), []
    for _, c in found:
        if tuple(c) not in seen:
            seen.add(tuple(c))
            res.append(c)
    return res


def parse_direction(src) -> dict:
    """see the module doc."""
    res = {"palette": None, "legible_boost": False, "plain": False}
    if isinstance(src, dict) and isinstance(src.get("user_style"), dict):
        u = src["user_style"]
        pal = [list(c)[:3] for c in (u.get("palette") or []) if isinstance(c, (list, tuple)) and len(c) >= 3]
        pal += [_hex(c) for c in (u.get("palette") or []) if isinstance(c, str) and _HEX.fullmatch(c.strip())]
        return {"palette": pal or None, "legible_boost": bool(u.get("legible_boost")), "plain": bool(u.get("plain") or u.get("legible_boost"))}
    pal = []
    for t in _texts(src):
        pal += [c for c in colours_in(t) if c not in pal]
        if _LEGIBLE.search(t):
            res["legible_boost"] = True
        if _PLAIN.search(t):
            res["plain"] = True
    res["palette"] = pal or None
    if res["legible_boost"]:
        res["plain"] = True
    return res


# 2026-11-01 user: "I'd like the text not to be filled with that textured pattern": letter interiors are a SOLID shot-palette ink by default (tools/fusion/letter_fill.py).
# A material / footage / plate / pattern INSIDE the letters only when the user's own words ask for it ("marble letters", "footage inside the letters", "text as a window",
# "textured text", "fill the letters with the image", "x14" ...). A colour name ("gold", "crimson") is never a material request.
_TEXTURED = re.compile(
    r"\b(textur\w*|material|marble|metal(lic)?|chrome|stone|wood(en)?|glass|brass|copper|steel|iron|concrete|fabric|liquid|molten|extruded|3d\s+letter\w*)\b[^.\n]{0,30}\b(text|type|letters?|lettering|words?|font)\b"
    r"|\b(text|type|letters?|lettering|words?)\b[^.\n]{0,30}\b(textur\w*|material|marble|metal(lic)?|chrome|stone|wood(en)?|glass|brass|copper|steel|iron|concrete|fabric|liquid|molten)\b"
    r"|\b(footage|video|image|picture|photo|plate|scene|shot|background)\s+(in|inside|within|through)\s+(the\s+)?(text|type|letters?|lettering|words?)\b"
    r"|\b(text|type|letters?|words?)\s+(as|like)\s+(a\s+)?(windows?|masks?|cut-?outs?)\b"
    r"|\bfill(ed|ing)?\s+(the\s+)?(text|type|letters?|words?)\s+with\b|\b(knock-?out|see-?through)\s+(text|type|letters?)\b|\bx14\b|\bplate[_ ]in[_ ]type\b", re.I)


def wants_textured_letters(src) -> bool:
    """True when the USER's own text (same sources as parse_direction; piece.user_style.textured / letter_fill = "texture" wins) asks for a material / footage / pattern inside
    the letters. Default False = solid letters."""
    if isinstance(src, dict) and isinstance(src.get("user_style"), dict) and "textured" in src["user_style"]:
        return bool(src["user_style"]["textured"])
    if isinstance(src, dict) and str(((src.get("compose") or {}).get("letter_fill")) or "").lower() in ("texture", "textured", "footage", "material"):
        return True
    return any(_TEXTURED.search(t) for t in _texts(src))


def clean_direction(text) -> str:
    """the user's typed text as the routes carry it: str, line breaks kept (directive lines need them), trimmed, capped."""
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()[:MAX_DIRECTION]


# ------------------------------------------------------------------------------------------------------------ split_typed: words vs direction
# 2026-11-01 (real run: "more legible text, palette: crimson, gold" became the on-screen words ["MORE LEGIBLE TEXT,", "CRIMSON, GOLD"]). ONE splitter for the READ, the story, the
# concept's brief and the page (web_app/bind.js splitTyped is a port, pinned to this by tests/test_split_typed.py): directive clauses / lines are the DIRECTION and never words.
# CONSERVATIVE like parse_direction: only an explicit palette phrase, a legibility / plain wish or a clear instruction ("make the text ...", "use gold") is direction; "GOLDEN HOUR",
# "RED SKY", "BORN IN GOLD", "MAKE IT COUNT", "KEEP GOING", "NO FEAR" stay words. A "double quoted" run is ALWAYS words.
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
_LEAD = r"(?:(?:use|using|with|in|a|an|the|set|keep|go|choose|pick)\s+)*"
_CUT = re.compile(rf"{_LEAD}\b(?:palette\s*(?::|=|of\b|is\b|are\b)|colou?rs?\s*[:=])[^\n]*", re.I)                      # "palette: ..." / "palette of ..." / "colours: ..." to the end of the (unquoted) line
_TAIL_LEAD = re.compile(rf"{_LEAD}$", re.I)
_VERB = r"(?:make|use|using|keep|set|render|show|give|put|add|try|have|write|go|style|paint|colou?r|apply|choose|pick)"
_PRE = r"(?:(?:please|also|and|then|just|now|maybe|can\s+you|could\s+you|let'?s|we\s+want|it\s+should|should|must|i\s+want(?:\s+to)?|i\s+need|i'?d\s+like(?:\s+to)?|i\s+would\s+like(?:\s+to)?)\s+)*"
_OBJ = r"(?:text|type|typeface|fonts?|letters?|lettering|colou?rs?|palette|background|effects?|echo(?:es)?|trails?|shimmer|glow|outline|shadow|contrast|animation|captions?|titles?|serif|sans|bold|italic|hues?|tones?)"
_INSTR = re.compile(rf"^\s*{_PRE}{_VERB}\b(?:\s+\S+){{0,5}}?\s+(?:the\s+)?(?:{_OBJ}|words?)\b|^\s*{_PRE}(?:make|use|using|set|paint|apply)\s+(?:\S+\s+){{0,3}}?{_NAME}\b", re.I)
_NEG = re.compile(rf"^\s*{_PRE}(?:don'?t|do\s+not|never|avoid|without)\b(?:\s+\S+){{0,4}}?\s+{_OBJ}\b", re.I)
_DELIM = re.compile(r"([,;.!?]+\s*|\n)")


def _is_directive(clause: str) -> bool:
    c = clause.strip()
    if not c:
        return False
    return bool(colours_in(c) or _LEGIBLE.search(c) or _PLAIN.search(c) or _INSTR.search(c) or _NEG.search(c))


def _split_chunk(chunk: str) -> tuple[str, list]:
    """an unquoted run inside one line -> (its words, [direction pieces])."""
    words, dirs, pos = [], [], 0
    pieces = []                                                           # (text, is_direction), in order
    for m in _CUT.finditer(chunk):                                        # the palette cue swallows the rest of the line
        pieces.append((chunk[pos:m.start()], False))
        pieces.append((m.group(0), True))
        pos = m.end()
    pieces.append((chunk[pos:], False))
    for txt, d in pieces:
        if d:
            dirs.append(txt.strip())
            continue
        spans = []                                                        # colour phrases that run across commas ("a crimson, gold and teal palette", "make it in red and gold")
        for rx in (_P_TAIL, _P_IN, _P_BARE):
            for m in rx.finditer(txt):
                s = m.start()
                if rx is _P_TAIL:
                    s = len(txt[:s]) - len(_TAIL_LEAD.search(txt[:s]).group(0)) if _TAIL_LEAD.search(txt[:s]) else s
                spans.append((s, m.end()))
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        p = 0
        segs = []
        for a, b in merged:
            segs.append((txt[p:a], False))
            segs.append((txt[a:b], True))
            p = b
        segs.append((txt[p:], False))
        for seg, sd in segs:
            if sd:
                dirs.append(seg.strip(" ,;.\t"))
                continue
            parts = _DELIM.split(seg)                                     # clause, delimiter, clause, ...
            for i in range(0, len(parts), 2):
                cl, dl = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
                if _is_directive(cl):
                    dirs.append(cl.strip())
                else:
                    words.append(cl + dl)
    return "".join(words), [d for d in dirs if d]


_EDGE = re.compile(r"^[\s,;.:\-–—]+|(?:[\s,;.:\-–—]|\b(?:and|with|plus|in)\b|&)+$", re.I)


def split_typed(text) -> tuple:
    """-> (words_text, direction_text): the typed "Words or direction" text split line by line and clause by clause. Directive clauses go to direction_text only (palette: / colours: lines,
    "palette of ...", "in red and gold", legibility asks, plain / simple / minimal asks, "make the text ...", "use ..."), everything else is the words. Pure direction -> words_text ""
    (the model writes its own words, as with an empty box). Quoted text is always words. parse_direction(direction_text) reads the same wishes as parse_direction(text)."""
    t = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()          # no length cap here: pasted lyrics are long
    wl, dl = [], []
    for line in t.split("\n"):
        if not line.strip():
            wl.append("")
            continue
        if _DIRECTIVE.match(line):
            dl.append(line.strip())
            continue
        q = list(_QUOTED.finditer(line))
        if not q:
            w, d = _split_chunk(line)
        else:                                                             # quoted runs are words; the rest of the line is split around them
            w, d, pos = "", [], 0
            for m in q:
                cw, cd = _split_chunk(line[pos:m.start()])
                w, d, pos = w + cw + m.group(0), d + cd, m.end()
            cw, cd = _split_chunk(line[pos:])
            w, d = w + cw, d + cd
        if not d:
            wl.append(line)                                               # nothing directive on the line: the line is untouched
            continue
        w = _EDGE.sub("", w.strip()).strip() if not q else w.strip()
        w = w if re.search(r"\w", w) else ""
        dl.append(line.strip() if not w else " ".join(d))
        if w:
            wl.append(w)
    if not dl:
        return t, ""                                                      # no direction anywhere: the text exactly as typed
    words = re.sub(r"\n{3,}", "\n\n", "\n".join(wl)).strip()
    return words, "\n".join(dl)


def _norm_words(s) -> str:
    return " ".join(re.findall(r"\w+", str(s or "").lower()))


def drop_direction_lines(lines, text) -> list:
    """lines minus those that are only a piece of the typed text's DIRECTION part (a stale READ whose on_screen copied "CRIMSON, GOLD" out of "palette: crimson, gold"); words stay."""
    d = " " + _norm_words(split_typed(text)[1]) + " "
    if d.strip() == "":
        return list(lines)
    return [x for x in lines if not (_norm_words(x) and " " + _norm_words(x) + " " in d)]


def words_of(text) -> str:
    """the part of text that may be shown on screen (split_typed's words_text)."""
    return split_typed(text)[0]


def stamp(piece, text):
    """THE landing point of the user's typed "Words or direction" text on a piece: piece["direction"] = the raw text, piece["user_style"] = parse_direction(text).
    Nothing typed = the piece is left untouched (byte-identical to before). Returns the piece."""
    t = clean_direction(text)
    if t and isinstance(piece, dict):
        piece["direction"] = t
        piece["user_style"] = parse_direction(t)
    return piece


def shows_in(owners) -> list:
    """the media-read text of each user image carried in the concept's request extras (concept.request_extras `shows`); [] when there is none."""
    v = owners
    if isinstance(v, str):
        if not v.strip().startswith("{"):
            return []
        try:
            v = json.loads(v)
        except ValueError:
            return []
    sh = v.get("shows") if isinstance(v, dict) else None
    return [str(x or "").strip()[:240] for x in sh] if isinstance(sh, list) and any(str(x or "").strip() for x in sh) else []


def direction_of(owners) -> str:
    """the typed text carried in the concept's request extras (the owners object {"owners": .., "direction": ..}: concept.request_extras); '' when there is none."""
    v = owners
    if isinstance(v, str):
        if not v.strip().startswith("{"):
            return ""
        try:
            v = json.loads(v)
        except ValueError:
            return ""
    return clean_direction(v.get("direction")) if isinstance(v, dict) and isinstance(v.get("direction"), str) else ""
