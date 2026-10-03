from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

MARGIN_BLOCKS = 2
SMOOTH = 3
CLIP = 96
SEAM = 12
LOOKBACK = 12
BLOCK_FRAC = 1 / 36


@dataclass
class ClipCanvas:
    frames: list[int]
    rects: list[Box]
    scale_hw: tuple[float, float]
    sizes: list[tuple[int, int]]
    pads: list[tuple[int, int]]
    crops: np.ndarray


def mosaic_block(frame_h: int) -> int:
    return max(2, int(round(frame_h * BLOCK_FRAC)))


def frame_window(track, i: int, frame_hw: tuple[int, int], canvas: int) -> Box:
    fh, fw = frame_hw
    hist = [track.boxes[j] for j in range(max(track.start, i - LOOKBACK), i + 1) if j in track.boxes]
    big = max(max(b[2] - b[0], b[3] - b[1]) for b in hist)
    side = int(min(max(big + 2 * MARGIN_BLOCKS * mosaic_block(fh), canvas), max(fh, fw)))
    sx, sy = min(side, fw), min(side, fh)
    near = hist[-SMOOTH:]
    cx = sum((b[0] + b[2]) / 2 for b in near) / len(near)
    cy = sum((b[1] + b[3]) / 2 for b in near) / len(near)
    x0 = min(max(int(round(cx - sx / 2)), 0), fw - sx)
    y0 = min(max(int(round(cy - sy / 2)), 0), fh - sy)
    return (x0, y0, x0 + sx, y0 + sy)


def _fit(rect: Box, s: float, size: int):
    x0, y0, x1, y1 = rect
    th, tw = min(size, max(1, int(round((y1 - y0) * s)))), min(size, max(1, int(round((x1 - x0) * s))))
    return (th, tw), ((size - th) // 2, (size - tw) // 2)


def _windows_and_scale(track, frames_idx, frame_hw, size):
    rects = [frame_window(track, i, frame_hw, size) for i in frames_idx]
    return rects, size / max(max(r[2] - r[0], r[3] - r[1]) for r in rects)


def build_clip_canvas(track, frames_idx: list[int], frames, size: int) -> ClipCanvas:
    hw = getattr(frames, "hw", None) or frames[0].shape[:2]
    rects, s = _windows_and_scale(track, frames_idx, hw, size)
    sizes, pads, crops = [], [], []
    for i, (x0, y0, x1, y1) in zip(frames_idx, rects):
        (th, tw), (top, left) = _fit((x0, y0, x1, y1), s, size)
        crop = cv2.resize(frames[i][y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        crops.append(cv2.copyMakeBorder(crop, top, size - th - top, left, size - tw - left, cv2.BORDER_REPLICATE))
        sizes.append((th, tw))
        pads.append((top, left))
    return ClipCanvas(list(frames_idx), rects, (s, s), sizes, pads, np.stack(crops))


def build_clip_canvas_gpu(track, frames_idx: list[int], frames, size: int) -> ClipCanvas:
    import torch
    import torch.nn.functional as F

    hw = getattr(frames, "hw", None) or tuple(frames[0].shape[:2])
    rects, s = _windows_and_scale(track, frames_idx, hw, size)
    sizes, pads, crops = [], [], []
    for i, (x0, y0, x1, y1) in zip(frames_idx, rects):
        (th, tw), (top, left) = _fit((x0, y0, x1, y1), s, size)
        crop = frames[i][y0:y1, x0:x1].permute(2, 0, 1)[None].float()
        if s < 1:
            crop = F.interpolate(crop, size=(th, tw), mode="area")
        else:
            crop = F.interpolate(crop, size=(th, tw), mode="bicubic", align_corners=False)
        crop = F.pad(crop, (left, size - tw - left, top, size - th - top), mode="replicate")
        crops.append(crop[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0))
        sizes.append((th, tw))
        pads.append((top, left))
    return ClipCanvas(list(frames_idx), rects, (s, s), sizes, pads, torch.stack(crops))


def uncanvas_frame(cc: ClipCanvas, k: int, restored: np.ndarray, scale: int = 1) -> np.ndarray:
    top, left = cc.pads[k]
    th, tw = cc.sizes[k]
    x0, y0, x1, y1 = cc.rects[k]
    crop = restored[top * scale : (top + th) * scale, left * scale : (left + tw) * scale]
    up = (x1 - x0) > crop.shape[1]
    return cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)


def clip_spans(start: int, end: int, clip: int = CLIP, seam: int = SEAM) -> list[list[int]]:
    spans, s = [], start
    while True:
        e = min(s + clip - 1, end)
        spans.append(list(range(s, e + 1)))
        if e == end:
            return spans
        s = e + 1 - seam


def _ramp(n: int) -> list[float]:
    return [0.5 - 0.5 * float(np.cos(np.pi * (j + 0.5) / n)) for j in range(n)]


def seam_weights(sp: list[int], has_parent: bool, has_child: bool, seam: int = SEAM) -> dict[int, float]:
    w = {f: 1.0 for f in sp}
    r = _ramp(seam)
    if has_parent:
        for j, f in enumerate(sp[:seam]):
            w[f] *= r[j]
    if has_child:
        for j, f in enumerate(sp[-seam:]):
            w[f] *= 1.0 - r[j]
    return w


def clip_weights(spans: list[list[int]], seam: int = SEAM) -> list[dict[int, float]]:
    return [seam_weights(sp, k > 0, k < len(spans) - 1, seam) for k, sp in enumerate(spans)]


def _region_under(mask_full: np.ndarray, rect: Box, frame_hw: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    fh, fw = frame_hw
    gh, gw = mask_full.shape[:2]
    if (gh, gw) == (fh, fw):
        return mask_full[y0:y1, x0:x1].astype(np.float32)
    sx, sy = gw / fw, gh / fh
    gx0, gy0 = max(0, int(np.floor(x0 * sx)) - 1), max(0, int(np.floor(y0 * sy)) - 1)
    gx1, gy1 = min(gw, int(np.ceil(x1 * sx)) + 1), min(gh, int(np.ceil(y1 * sy)) + 1)
    sub = mask_full[gy0:gy1, gx0:gx1].astype(np.float32)
    size = (max(1, int(round((gx1 - gx0) / sx))), max(1, int(round((gy1 - gy0) / sy))))
    up = cv2.resize(sub, size, interpolation=cv2.INTER_LINEAR)
    ox, oy = int(round(x0 - gx0 / sx)), int(round(y0 - gy0 / sy))
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    part = up[oy : oy + (y1 - y0), ox : ox + (x1 - x0)]
    out[: part.shape[0], : part.shape[1]] = part
    return out


def blend_mask(mask_full: np.ndarray | None, rect: Box, frame_hw: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if mask_full is None:
        return np.ones((y1 - y0, x1 - x0, 1), np.float32)
    b = mosaic_block(frame_hw[0])
    m = (_region_under(mask_full, rect, frame_hw) > 0.5).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1)))
    alpha = np.clip(cv2.distanceTransform(m, cv2.DIST_L2, 5) / b, 0.0, 1.0)
    return alpha[..., None].astype(np.float32)
