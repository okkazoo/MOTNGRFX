"""WORD PICK (tools/fusion/word_pick.py + story.py + media_read.py + web_app/bind.js): which sung words go on a shot. Made-up lyrics only."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fusion import media_read as MR  # noqa: E402
from tools.fusion import story as ST  # noqa: E402
from tools.fusion import word_pick as WP  # noqa: E402


def line(t, words, step=0.4, hold=None):
    """a sung line: words start `step` apart; hold = {index: seconds} for held notes -> the READ's lyric line with `ts` and `words`."""
    hold = hold or {}
    ws, x = [], t
    for i, w in enumerate(words.split()):
        d = hold.get(i, step * 0.8)
        ws.append(dict(text=w, t=round(x, 2), end=round(x + d, 2), p=0.9))
        x += max(step, d + 0.05)
    return dict(t=ws[0]["t"], end=ws[-1]["end"], text=words, ts=[w["t"] for w in ws], words=ws)


# a made-up song: "golden hour" and "fire" come back like a chorus, one note is held
SONG = [
    line(0.5, "we walk the golden hour"),
    line(3.0, "fire in the night"),
    line(5.0, "and the river runs on"),
    line(8.0, "golden hour", hold={1: 1.6}),
    line(10.0, "fire fire burning bright"),
    line(13.0, "to the sky"),
]


def prep(lines=SONG, beats=None):
    toks, ts, es, ln = [], [], [], []
    for l in lines:
        a = len(toks)
        for w in l["words"]:
            toks.append(w["text"]), ts.append(w["t"]), es.append(w["end"])
        ln.append((a, len(toks) - 1))
    return WP.prepare(toks, ts, es, ln, beats), toks


def words_of(P, idx):
    return " ".join(P["tokens"][i] for i in idx)


def test_scoring_prefers_repeated_content_words_over_stopwords():
    P, toks = prep()
    sc = WP.word_scores(P)
    fire, the = toks.index("fire"), toks.index("the")
    assert sc[fire] > sc[the] + 3                                  # a hook word vs a function word
    assert all(sc[i] < 0 for i, t in enumerate(toks) if t in ("the", "in", "and", "on", "to", "we"))
    once = toks.index("river")                                     # a content word sung once still beats every function word
    assert sc[fire] > sc[once] > sc[the]
    assert WP.word_parts(P, fire)["hook"] > 0 and WP.word_parts(P, once)["hook"] == 0


def test_held_note_beat_and_line_end_raise_a_word():
    P, toks = prep()
    held = [i for i, t in enumerate(toks) if t == "hour" and P["dur"][i] > 1.0]
    assert held and WP.word_parts(P, held[0])["held"] > 1.0
    assert WP.word_parts(P, toks.index("sky"))["end"] >= 0.9       # last word of its line
    g = toks.index("golden")                                       # sung at 1.3 s? the grid below has a beat there
    Pb, _ = prep(beats=[P["times"][g], 3.0])
    assert WP.word_parts(Pb, g)["beat"] > 0.9 and WP.word_parts(P, g)["beat"] == 0


def test_picks_fall_inside_the_shots_span():
    P, toks = prep()
    for a, b in [(0, 3), (3, 5), (5, 8), (8, 10), (10, 13), (13, 16)]:
        run = WP.pick(P, a, b)
        assert run, (a, b)
        assert a <= P["times"][run[0]] < b
        alts = WP.alternatives(P, a, b)
        assert alts and alts[0]["t"] == run and all(a <= P["times"][x["t"][0]] < b for x in alts)
        assert len({tuple(x["t"]) for x in alts}) == len(alts)     # different ideas
    assert WP.pick(P, 20, 30) == []                                # nothing sung there


def test_a_function_word_never_starts_or_ends_a_pick_and_no_line_is_crossed():
    P, toks = prep()
    for a, b in [(0, 3), (3, 5), (5, 8), (8, 10), (10, 13), (13, 16), (0, 16)]:
        for s, run in WP.runs_in_span(P, a, b):
            assert not WP.is_weak(toks[run[0]]) and not WP.is_weak(toks[run[-1]]) and len(run) <= 3
            assert len({P["line_of"][i] for i in run}) == 1
            assert run == list(range(run[0], run[-1] + 1))


def test_phrases_are_not_split():
    P, toks = prep()
    for a, b in [(0, 3), (8, 10), (0, 16)]:
        for _, run in WP.runs_in_span(P, a, b):
            assert words_of(P, run) not in ("golden", "hour"), (a, b, run)      # "golden hour" is sung twice: never cut into its halves
    assert words_of(P, WP.pick(P, 8, 10)) == "golden hour"
    assert words_of(P, WP.pick(P, 0, 3)) == "golden hour"
    assert words_of(P, WP.pick(P, 10, 13)).startswith("fire")


def test_legibility_fragments_and_long_words_score_low():
    P = WP.prepare(["'cause", "a", "extraordinarily", "fire", "hmm", "x"], [0, 1, 2, 3, 4, 5], None, None, None)
    sc = WP.word_scores(P)
    assert sc[3] > sc[0] and sc[3] > sc[2] and sc[3] > sc[5]


def test_deterministic_per_seed():
    P, _ = prep()
    a = [WP.pick(P, x, x + 3, 0, 7) for x in (0, 3, 6, 9, 12)]
    b = [WP.pick(P, x, x + 3, 0, 7) for x in (0, 3, 6, 9, 12)]
    assert a == b
    P2, _ = prep()
    assert WP.alternatives(P2, 0, 16, 7) == WP.alternatives(P, 0, 16, 7)
    # the jitter only breaks ties: a clear winner stays the winner for every seed
    assert {tuple(WP.pick(P, 10, 13, 0, s)) for s in range(20)} == {tuple(WP.pick(P, 10, 13, 0, 0))}


def test_untimed_text_has_no_timed_picks():
    P = WP.prepare(["hello", "world"], [None, None], None, None, None)
    assert not P["timed"] and WP.runs_in_span(P, 0, 10) == [] and WP.pick(P, 0, 10) == []


def test_model_pick_guard():
    P, toks = prep()
    ok = [toks.index("river")]
    assert WP.model_pick_ok(P, ok, 5, 8) and not WP.model_pick_ok(P, ok, 0, 3)        # sung at 5.x, not inside 0-3
    assert not WP.model_pick_ok(P, [toks.index("the")], 0, 16)                          # only a function word
    assert not WP.model_pick_ok(P, [], 0, 16) and not WP.model_pick_ok(P, [999], 0, 16)


# ------------------------------------------------------------------------------------------------- story.py
def story_req(length=12, n_images=3, seed=3, lyrics=SONG, select=None):
    media = [dict(id=f"I{k}", kind="image", file=f"p{k}.png", shows=f"picture {k}", window=None, mood="calm", palette=["#112233"]) for k in range(1, n_images + 1)]
    media.append(dict(id="A1", kind="audio", file="song.wav", shows="song", window=None, mood="", palette=[]))
    d = dict(read=dict(v=1, on_screen=[], concept="", look="", media=media, music=dict(bpm=90.0, key="D minor"), lyrics=[dict(l) for l in lyrics], notes=""), length=length, seed=seed, format="9:16")
    if select:
        d["select"] = select
    return d


def reply(n, picks=None):
    return json.dumps(dict(title="Night Fire", logline="fire in the night", look="warm", music="pop, drums, bass, piano", notes="",
                           shots=[dict(n=k, beat=f"Beat {k} of the story, moving on.", pick=(picks[k - 1] if picks else [])) for k in range(1, n + 1)]))


def shot_lines(s):
    return [x["line"] for x in s["shots"]]


def test_story_rule_pick_comes_from_what_is_sung_in_each_shot():
    r = story_req(12, 3)                                       # shots 0-4 / 4-8 / 8-12
    s = ST.parse(reply(3), r)
    tc = s["text_candidates"]["tokens"]
    spans = [(x["start"], x["end"]) for x in s["shots"]]
    assert all(shot_lines(s)) and s["text_select"]
    for d in s["text_select"]:
        a, b = spans[d["shot"] - 1]
        assert a <= tc[d["i"]]["t"] < b + 1e-6                 # every chosen word is sung while its shot is on
    assert "word score" in s["notes"]
    assert shot_lines(s)[2] == "GOLDEN HOUR"                   # the repeated, held phrase of shot 3 (8-12 s)
    assert s == ST.parse(reply(3), story_req(12, 3))           # deterministic


def test_story_model_pick_sung_elsewhere_is_rejected_and_a_good_one_kept():
    r = story_req(12, 3)
    flat = [w for l in r["read"]["lyrics"] for w in l["text"].split()]
    river, fire = flat.index("river"), flat.index("fire")      # river is sung at 5.x (shot 2), the first fire at 3.0 (shot 1)
    s = ST.parse(reply(3, picks=[[river], [], [flat.index("fire")]]), r)
    assert shot_lines(s)[0] != "RIVER"                          # not sung during shot 1: rejected, replaced by the score pick
    assert "not sung inside" in s["notes"]
    s2 = ST.parse(reply(3, picks=[[fire], [], []]), r)
    assert shot_lines(s2)[0] == "FIRE"                          # "fire" is sung at 3.0 s inside shot 1 (0-4 s): the model's choice stays


def test_story_word_alts_per_shot_are_inside_the_span():
    s = ST.parse(reply(3), story_req(12, 3))
    alts = s["word_alts"]
    assert set(alts) <= {"1", "2", "3"} and alts["3"]
    tc = s["text_candidates"]["tokens"]
    for k, lst in alts.items():
        a, b = s["shots"][int(k) - 1]["start"], s["shots"][int(k) - 1]["end"]
        assert lst[0]["t"] and all(a <= tc[x["t"][0]]["t"] < b for x in lst) and lst == sorted(lst, key=lambda x: -x["s"])
    chosen = [d["i"] for d in s["text_select"] if d["shot"] == 3]
    assert chosen == alts["3"][0]["t"]                          # the shot's pick = the best alternative when the model gave none


def test_story_user_select_is_still_honoured_verbatim():
    flat = [w for l in SONG for w in l["text"].split()]
    s = ST.parse(reply(3), story_req(12, 3, select=[dict(i=flat.index("river"), phrase=1, shot=3)]))
    assert shot_lines(s) == ["", "", "RIVER"]


def test_story_pasted_text_keeps_the_old_rule_pick():
    r = story_req(12, 3)
    r["read"]["lyrics"] = []
    r["text"] = "hold on to the light and let go of the night again and again"
    s = ST.parse(reply(3), r)
    assert "word_alts" not in s and all(shot_lines(s))


def test_video_mode_word_alts_use_the_cuts_source_spans():
    media = [dict(id="V1", kind="video", file="v.mp4", shows="a clip", window=[0.0, 12.0], mood="", palette=[])]
    rd = dict(v=1, on_screen=[], concept="", look="", media=media, music=dict(bpm=90.0, key="D"), lyrics=[dict(l) for l in SONG], notes="", video_duration=24.0,
              video_cuts=[dict(start=0.0, end=3.0), dict(start=3.0, end=8.0), dict(start=8.0, end=12.0)], beat_grid=dict(bpm=90.0, beats=[round(0.667 * k, 2) for k in range(1, 40)], downbeats=[0.0, 2.67]))
    s = ST.parse(json.dumps(dict(title="x", logline="y", look="z", music="m", notes="")), dict(read=rd, length=12, seed=3, format="9:16"))
    assert s["word_alts"] and s["hits"]
    tc = s["text_candidates"]["tokens"]
    for k, lst in s["word_alts"].items():
        sh = s["shots"][int(k) - 1]
        assert all(sh["src_start"] <= tc[x["t"][0]]["t"] < sh["src_end"] for x in lst)


# ------------------------------------------------------------------------------------------------- Whisper words carried through
def test_asr_words_carry_start_end_probability():
    j = dict(segments=[dict(start=0.0, end=2.0, text="x", words=[dict(w="golden", s=0.5, e=1.1, p=0.9), dict(w="hour", s=1.2, e=2.4, p=0.8), dict(w="la", s=2.5, e=2.6, p=0.1)]),
                       dict(start=3.0, end=4.0, text="y", words=[dict(w="fire", s=3.0, e=3.5, p=0.95), dict(w="night", s=3.6, e=4.2, p=0.9), dict(w="burning", s=4.3, e=4.9, p=0.9)])])
    lines, note = MR.lyrics_from_asr(j)
    assert note is None and lines[0]["text"] == "golden hour" and lines[0]["ts"] == [0.5, 1.2]                  # a low-probability word is dropped
    assert lines[0]["words"] == [dict(text="golden", t=0.5, end=1.1, p=0.9), dict(text="hour", t=1.2, end=2.4, p=0.8)]
    out = MR.lyric_out(lines[0], 10.0)                                                                           # put on the video's timeline
    assert out["t"] == 10.5 and out["ts"] == [10.5, 11.2] and out["words"][1]["end"] == 12.4 and out["end"] == 12.4
    assert set(MR.lyric_out(dict(t=1.0, text="a b"))) == {"t", "text"}


def test_story_uses_word_ends_for_held_notes():
    R = ST.norm_req(story_req(12, 3))
    assert R["wp"] is not None and R["wp"]["real_ends"] and len(R["tok_e"]) == len(R["tok_t"])
    plain = story_req(12, 3)
    for l in plain["read"]["lyrics"]:
        l.pop("words"), l.pop("end")
    R2 = ST.norm_req(plain)
    assert R2["tok_t"] == R["tok_t"] and not R2["wp"]["real_ends"]                       # ts alone still times the words


# ------------------------------------------------------------------------------------------------- bind.js (node, no DOM)
NODE = shutil.which("node")


def run_node(js: str):
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(not NODE, reason="no node on this machine")
def test_bind_lyric_tokens_use_the_word_timings():
    lyr = [dict(t=1.0, end=4.0, text="a b c d", words=[dict(text="a", t=1.0, end=1.2, p=.9), dict(text="b", t=2.5, end=3.3, p=.8), dict(text="c", t=3.4, end=3.5), dict(text="d", t=3.9, end=4.0)]),
           dict(t=6.0, end=6.0, text="e f", ts=[6.0, 6.7]), dict(t=9.0, end=11.0, text="g h")]
    got = run_node("const B=require('./web_app/bind.js');console.log(JSON.stringify(B.lyricTokens(" + json.dumps(lyr) + ")))")
    assert got["tokens"] == list("abcdefgh")
    assert got["times"][:4] == [1.0, 2.5, 3.4, 3.9] and got["times"][4:6] == [6.0, 6.7]            # the real word times, not an even spread
    assert got["times"][6:] == [9.0, 10.0] and got["ends"][1] == 3.3 and got["probs"][:2] == [0.9, 0.8]   # no word detail: even spread over t..end (fallback)


@pytest.mark.skipif(not NODE, reason="no node on this machine")
def test_bind_word_editor_helpers():
    js = """
const B=require('./web_app/bind.js');
const shots=[{n:1,start:0,end:4},{n:2,start:4,end:8},{n:3,start:8,end:12}];
const times=[0.5,1,3.9,4.0,6,11.9,12.0,13];
const g=B.sungByShot(shots,times);
let sel=[{i:2,phrase:1,shot:1},{i:4,phrase:2,shot:2}];
const story={shots,text_select:sel,word_alts:{'1':[{t:[0,1],s:3},{t:[2],s:2}],'2':[{t:[4],s:2},{t:[3],s:1}]}};
const c1=B.cycleShotWords(story,1);
const c2=B.cycleShotWords(Object.assign({},story,{text_select:c1}),1);
const c3=B.cycleShotWords(story,2);
console.log(JSON.stringify({by:[...g.by.entries()],out:g.outside,range:B.rangeWordShot(sel,0,2,1).map(t=>t.i+':'+t.shot),clear:B.clearShotWords(sel,1).map(t=>t.i),
  c1:c1.map(t=>t.i+':'+t.shot),c2:c2.map(t=>t.i+':'+t.shot),c3:c3&&c3.map(t=>t.i+':'+t.shot),none:B.cycleShotWords({shots,text_select:[]},1)}));
"""
    got = run_node(js)
    assert got["by"] == [[1, [0, 1, 2]], [2, [3, 4]], [3, [5, 6]]] and got["out"] == [7]          # the last shot owns a word at its very end
    assert got["range"] == ["0:1", "1:1", "2:1", "4:2"] and got["clear"] == [4]
    assert got["c1"] == ["0:1", "1:1", "4:2"]                       # current words (2) are not alt 0: the first alternative
    assert got["c2"] == ["2:1", "4:2"]                              # then the next one
    assert got["c3"] == ["2:1", "3:2"] and got["none"] is None             # shot 2 holds 4 = alt 0 -> alt 1; no alternatives -> null


@pytest.mark.skipif(not NODE, reason="no node on this machine")
def test_bind_changed_video_line_lands_when_it_is_sung():
    js = """
const B=require('./web_app/bind.js');
const sh={n:2,start:2,end:6,src_start:12,src_end:16,image:'V1'};
console.log(JSON.stringify([B.landHit(sh,'GOLDEN',[0],[13.1],[2.6,3.0,3.2]),B.landHit(sh,'X',[0],[30],[]),B.landHit(sh,'Y',[0,1],[15.5,15.9],[]),B.landHit(sh,'Z',[],[],[])]));
"""
    a, b, c, d = run_node(js)
    assert a["land"] == 3.2 and a["sung"] == 13.1 and a["hold_until"] >= a["land"] + 0.69          # 13.1 -> piece 3.1, snapped forward to the 3.2 beat
    assert b == dict(text="X", land=2, hold_until=6)                                               # sung outside this cut: the shot's start
    assert c["land"] == 5.3 and c["hold_until"] <= 6                                               # clamped inside the shot
    assert d["land"] == 2


@pytest.mark.skipif(not NODE, reason="no node on this machine")
def test_bind_normalize_story_keeps_word_alts():
    js = """
const B=require('./web_app/bind.js');
const n=B.normalizeStory({v:2,title:'t',shots:[{n:1,start:0,end:4}],word_alts:{'1':[{t:[3,4],s:2.5},{t:[],s:1},{x:1}],'2':[]}});
console.log(JSON.stringify(n.word_alts));
"""
    assert run_node(js) == {"1": [{"t": [3, 4], "s": 2.5}]}
