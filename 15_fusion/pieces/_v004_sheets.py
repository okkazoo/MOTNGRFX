"""self-check sheets for a v004 draft: 6 fps contact sheet (time-stamped) + 6-frame strips across given times.

    python 15_fusion/pieces/_v004_sheets.py <mp4> <out_prefix> [t1 t2 ...]   -> <prefix>_sheet6fps.png, <prefix>_strip_<t>.png
"""
import subprocess
import sys

import cv2
import numpy as np


def frames(mp4):
    cap = cv2.VideoCapture(mp4)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def label(img, txt):
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    mp4, pre = sys.argv[1], sys.argv[2]
    ts = [float(x) for x in sys.argv[3:]]
    fr = frames(mp4)
    h, w = fr[0].shape[:2]
    tw = 135 if h > w else 240
    th = int(round(tw * h / w))
    idx = list(range(0, len(fr), 4))
    cols = 12 if h > w else 8
    rows = (len(idx) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, i in enumerate(idx):
        t = cv2.resize(fr[i], (tw, th), interpolation=cv2.INTER_AREA)
        label(t, f"{i / 24:.2f}")
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(f"{pre}_sheet6fps.png", sheet)
    sw = 270 if h > w else 480
    sh = int(round(sw * h / w))
    for t in ts:
        c = int(round(t * 24))
        ii = [min(max(c + d, 0), len(fr) - 1) for d in (-8, -4, 0, 4, 8, 12)]
        strip = np.concatenate([label(cv2.resize(fr[i], (sw, sh), interpolation=cv2.INTER_AREA), f"{i / 24:.2f}") for i in ii], 1)
        cv2.imwrite(f"{pre}_strip_{t:.2f}.png", strip)
    print("frames", len(fr), "sheet", f"{pre}_sheet6fps.png")


if __name__ == "__main__":
    main()
