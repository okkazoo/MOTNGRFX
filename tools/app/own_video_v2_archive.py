#!/usr/bin/env python
"""(runs ON THE BOX, stdlib only) assemble the v002 archive dir MographArchiveLoad reads: shot clips + CUT-AWARE mattes.

    python own_video_v2_archive.py --work /tmp/w --alpha-dir /workspace/ComfyUI/output/projects/.../v002 --out /workspace/ComfyUI/output/own_video/arch_v002

--work holds segments.json, spans.json, piece.json, index.json (tools/app/own_video_v2.py prep). The Alpha-Gen PNG frames of every PERSON segment (job h<j>, file
`<stem>_alpha_h<j>_image_NNNNN_.png`, frame 1 = the segment's first picture) are laid onto the planned shots on the 24 fps timeline: shot k, frame i  <->  timeline frame
start_k + i  <->  the segment that frame belongs to  <->  its matte frame (i - segment start). Frames of insert segments (train, graffiti, street, close-up, crowd) get an empty
(black) matte, so a matte is never shown on a picture it was not cut from. sK_clip.mp4 = the planned shot's clip (8k+1 frames, from the same 24 fps timeline).
"""
import argparse, glob, json, os, shutil, struct, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--alpha-dir", required=True)
ap.add_argument("--seg", default="/workspace/own_video/v2/seg")
ap.add_argument("--stem", default="motion-graphic_own-video_v002")
ap.add_argument("--size", default="576x1024", help="WxH of the (empty) matte frames: the Alpha-Gen size of the version")
ap.add_argument("--out", required=True)
a = ap.parse_args()

spans = json.load(open(f"{a.work}/spans.json"))
segs = json.load(open(f"{a.work}/segments.json"))
os.makedirs(a.out, exist_ok=True)


def black_png(path, w=576, h=1024):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = (b"\x00" + b"\x00" * (w * 3)) * h
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def seg_of(j):
    for s in segs:
        if s["a"] <= j < s["b"]:
            return s
    return None


black = f"{a.work}/black_576x1024.png"
black_png(black, *[int(x) for x in a.size.split("x")])
mp = {}
for k, n in enumerate(spans["spans"], 1):
    start = spans["starts"][k - 1]
    na = ((n - 1 + 7) // 8) * 8 + 1
    shutil.copy(f"{a.seg}/s{k}_clip.mp4", f"{a.out}/s{k}_clip.mp4")
    md = f"{a.out}/s{k}_matte"
    shutil.rmtree(md, ignore_errors=True)
    os.makedirs(md)
    rows = []
    for i in range(na):
        j = start + i
        last = sum(spans["spans"]) - 1
        if j > last:                                # the pad of the last shot repeats the window's last frame (clip: tpad clone): its matte is the last frame's matte
            j = last
        s = seg_of(j)
        src, why = black, "beyond the clip"
        if s is not None:
            why = s["label"]
            if s["person"]:
                p = f"{a.alpha_dir}/{a.stem}_alpha_h{s['job']}_image_{j - s['a'] + 1:05d}_.png"
                if not os.path.isfile(p):
                    raise SystemExit(f"missing matte frame {p}")
                src = p
        shutil.copy(src, f"{md}/{i + 1:05d}.png")
        rows.append([j, s["i"] if s else None, "matte" if src != black else "empty"])
    mp[f"s{k}"] = rows
    print(f"s{k}: {na} clip frames (timeline {start}..{start + na - 1}), mattes from segments {sorted({r[1] for r in rows if r[2] == 'matte'})}, empty on {sum(1 for r in rows if r[2] == 'empty')} frames")
shutil.copy(f"{a.seg}/s1_first.png", f"{a.out}/still.png")
shutil.copy(f"{a.work}/piece.json", f"{a.out}/piece.json")
shutil.copy(f"{a.work}/index.json", f"{a.out}/index.json")
json.dump(mp, open(f"{a.out}/matte_map.json", "w"))
print("archive", a.out)
