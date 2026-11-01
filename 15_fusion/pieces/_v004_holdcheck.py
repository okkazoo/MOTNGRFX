"""hold readability check (user rule 2026-10-26): during each word's hold every letter must be >= 0.8 visible for >= 80 % of the hold.
Runs ON THE BOX (cd /workspace/mograph; python 15_fusion/pieces/_v004_holdcheck.py <PID> --holds a,b a,b ...), on the piece's real inputs:
  letters   = connected components of the type matte pass (out/<PID>/type/matte) at each hold frame (> 0.5)
  occluder  = the sandwich matte (shot depth, normalised like compose: 2/98 %, smoothstep(thr -+ soft/2)) x (1 - front), front = type
              matte x the `front` gate timeline of the piece  +  the sheer overlay (luma key of the graded footage x its gate) after it
  visible   = sum(type * (1 - occ)) / sum(type) per letter.
Prints per word: frames checked, worst letter fraction, % of hold frames where every letter >= 0.8; PASS / FAIL. Approximation: the
compose order (flow_wake displacement, camera translate) is ignored, letters come from the undisplaced matte pass."""
import glob
import os
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from tools.fusion import compose as C  # noqa: E402


def gate_of(piece, out_name):
    for o in piece["compose"]["default"]["ops"]:
        if o.get("out") == out_name and o["op"] == "multiply":
            return o["args"]["b"]
    return None


def val(spec, t):
    if isinstance(spec, dict) and "timeline" in spec:
        return float(C._timeline(spec, t))
    return float(spec) if isinstance(spec, (int, float)) else 1.0


def main():
    pid = sys.argv[1]
    holds = [tuple(float(x) for x in a.split(",")) for a in sys.argv[sys.argv.index("--holds") + 1:]]
    piece = json.load(open(f"work/{pid}/piece.json"))
    ops = piece["compose"]["default"]["ops"]
    sw = [o for o in ops if o["op"] == "sandwich"][0]["args"]
    thr, soft = float(sw.get("matte_threshold", 0.5)), float(sw.get("matte_soft", 0.08))
    front_gate = gate_of(piece, "front")
    sheer = [o for o in ops if o["op"] == "luminosity_key" and o["out"] in ("spray", "fog", "seeds")]
    sheer = sheer[0] if sheer else None
    sheer_gate = gate_of(piece, sheer["out"]) if sheer else None
    comfy = "/workspace/ComfyUI/output/projects/COMPETITIONS/motion-graphic/fusion/v004"
    dfiles = sorted(glob.glob(f"{comfy}/motion-graphic_fusion_v004_{pid}_depth_s1_*_*.png"))
    shot = sorted(glob.glob(f"{comfy}/motion-graphic_fusion_v004_{pid}_s1_*_*.mp4"))[-1]
    idx = np.linspace(0, len(dfiles) - 1, 10).round().astype(int)
    v = np.concatenate([cv2.imread(dfiles[i], 0)[::4, ::4].ravel() / 255.0 for i in idx])
    lo, hi = np.percentile(v, [2, 98])
    cap = cv2.VideoCapture(shot)
    allf = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        allf.append(f)
    # camera track (compose translate ops on the type matte): shift the letters like compose does; off-frame letter area = hidden
    shifts = [o for o in ops if o["op"] == "translate" and o["out"].startswith("cam_")]
    cam = None
    if shifts:
        from tools.fusion import recipe_eval as RE
        fl = np.load(f"out/{pid}/flow/s1_flow.npy", mmap_mode="r")
        sz = (960, 540) if piece.get("format") == "16:9" else (540, 960)
        cam = RE.camera_track(fl, sz)

    def dxy(pf):
        x = y = 0.0
        for o in shifts:
            for ax, key in ((0, "dx"), (1, "dy")):
                a = o["args"][key]
                d = float(cam[min(pf, len(cam) - 1)][ax]) * float(a.get("gain", 1.0)) if isinstance(a, dict) and "cam" in a else val(a, pf / 24.0)
                if ax == 0:
                    x += d
                else:
                    y += d
        return x, y
    ok_all = True
    for a, b in holds:
        fr = list(range(int(round(a * 24)), int(round(b * 24))))
        good, worst = 0, 1.0
        for pf in fr:
            tm = cv2.imread(f"out/{pid}/{os.environ.get('CHECK_TYPE', 'type')}/matte/matte_{pf:05d}.png", 0)
            if tm is None:
                continue
            H, W = 960, 540
            if tm.shape[1] > tm.shape[0]:
                H, W = 540, 960
            tm = cv2.resize(tm, (W, H)).astype(np.float32) / 255.0
            if shifts:
                sx, sy = dxy(pf)
                big = cv2.copyMakeBorder(tm, H, H, W, W, cv2.BORDER_CONSTANT, value=0)
                M = np.float32([[1, 0, sx], [0, 1, sy]])
                big = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]))
                tm_full = big
                tm = big[H:2 * H, W:2 * W]
            else:
                tm_full = None
            d = cv2.resize(cv2.imread(dfiles[min(pf, len(dfiles) - 1)], 0), (W, H)).astype(np.float32) / 255.0
            d = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1)
            m = np.clip((d - (thr - soft / 2)) / soft, 0, 1)
            m = m * m * (3 - 2 * m)
            fg = tm * val(front_gate, pf / 24.0)
            occ = m * (1 - np.clip(fg, 0, 1))
            if sheer is not None:
                img = cv2.resize(allf[min(pf, len(allf) - 1)], (W, H))[:, :, ::-1].astype(np.float32) / 255.0
                l = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
                k = np.clip((l - sheer["args"]["low"]) / (sheer["args"]["high"] - sheer["args"]["low"]), 0, 1)
                k = k * k * (3 - 2 * k) * val(sheer_gate, pf / 24.0)
                occ = 1 - (1 - occ) * (1 - k)
            src = tm_full if tm_full is not None else tm
            n, lab = cv2.connectedComponents((src > 0.5).astype(np.uint8))
            occ_full = np.ones(src.shape, np.float32)
            if tm_full is not None:
                occ_full[H:2 * H, W:2 * W] = occ
            else:
                occ_full = occ
            fr_min = 1.0
            for c in range(1, n):
                sel = lab == c
                if sel.sum() < 30:
                    continue
                vis = float((src[sel] * (1 - occ_full[sel])).sum() / max(src[sel].sum(), 1e-6))
                fr_min = min(fr_min, vis)
            worst = min(worst, fr_min)
            good += fr_min >= 0.8
        pct = good / max(len(fr), 1)
        res = pct >= 0.8
        ok_all &= res
        print(f"{pid} hold {a:.2f}-{b:.2f}: {len(fr)} frames, worst letter {worst:.2f}, all letters >=0.8 in {100 * pct:.0f}% -> "
              f"{'PASS' if res else 'FAIL'}")
    print(f"{pid}: {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
