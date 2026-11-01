"""Cover-crop and coordinate mapping shared by the planner and the renderer.

All rectangles are normalised ``[x, y, w, h]`` with a top-left origin. A *crop*
is a rectangle in normalised image space; mapping a point through a crop gives
its normalised position in the frame that the crop fills.
"""
from __future__ import annotations

FORMATS = {"9:16": (1080, 1920), "16:9": (1920, 1080)}

DRAFT_SHORT_SIDE = 360


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def frame_size(fmt: str, quality: str) -> tuple[int, int]:
    """(width, height) for a format. "full" is FORMATS[fmt]; "draft" has a short side of 360 px."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(FORMATS)}")
    w, h = FORMATS[fmt]
    if quality == "full":
        return w, h
    if quality != "draft":
        raise ValueError(f"unknown quality {quality!r}; expected 'draft' or 'full'")
    short, long_ = min(w, h), max(w, h)
    s = DRAFT_SHORT_SIDE
    l = _even(s * long_ / short)
    return (s, l) if w <= h else (l, s)


def cover_crop(img_w: int, img_h: int, frame_w: int, frame_h: int,
               focus: tuple[float, float] = (0.5, 0.5), zoom: float = 1.0) -> tuple[float, float, float, float]:
    """Normalised image-space crop [x, y, w, h] that covers the frame at ``zoom``.

    The crop has the frame's aspect ratio, is centred on ``focus`` (normalised image
    space) and is clamped inside the image. ``zoom`` < 1 is treated as 1 (anything
    smaller could not cover the frame).
    """
    img_w = max(1.0, float(img_w))
    img_h = max(1.0, float(img_h))
    frame_w = max(1.0, float(frame_w))
    frame_h = max(1.0, float(frame_h))
    zoom = max(1.0, float(zoom))
    a_img = img_w / img_h
    a_frm = frame_w / frame_h
    if a_img > a_frm:          # image wider than frame: use full height
        cw, ch = a_frm / a_img, 1.0
    else:                      # image taller: use full width
        cw, ch = 1.0, a_img / a_frm
    cw /= zoom
    ch /= zoom
    fx = float(focus[0]) if focus is not None else 0.5
    fy = float(focus[1]) if focus is not None else 0.5
    x = min(max(fx - cw / 2.0, 0.0), 1.0 - cw)
    y = min(max(fy - ch / 2.0, 0.0), 1.0 - ch)
    return (x, y, cw, ch)


def image_to_frame(pt: tuple[float, float], crop: tuple[float, float, float, float]) -> tuple[float, float]:
    """Normalised image-space point -> normalised frame-space point (may fall outside 0..1)."""
    cx, cy, cw, ch = crop
    return ((pt[0] - cx) / cw, (pt[1] - cy) / ch)


def frame_to_image(pt, crop) -> tuple[float, float]:
    """Normalised frame-space point -> normalised image-space point."""
    cx, cy, cw, ch = crop
    return (cx + pt[0] * cw, cy + pt[1] * ch)


def rect_image_to_frame(rect, crop) -> tuple[float, float, float, float]:
    """Image-space rect -> frame-space rect, clipped to 0..1 (w/h may become 0)."""
    x, y, w, h = rect
    x0, y0 = image_to_frame((x, y), crop)
    x1, y1 = image_to_frame((x + w, y + h), crop)
    x0, x1 = min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)
    y0, y1 = min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))
