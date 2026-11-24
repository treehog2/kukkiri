from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

GENITAL_LABEL = 1


def block_grid(detections, hw: tuple[int, int], block: int) -> np.ndarray | None:
    h, w = hw
    m = np.zeros((h, w), np.uint8)
    for d in detections:
        if d.label != GENITAL_LABEL:
            continue
        if d.mask is not None:
            m |= cv2.resize(d.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            m[max(0, y0):y1, max(0, x0):x1] = 1
    if not m.any():
        return None
    m = cv2.dilate(m, np.ones((block // 4 * 2 + 1,) * 2, np.uint8))
    gy, gx = -(-h // block), -(-w // block)
    pad = np.zeros((gy * block, gx * block), np.float32)
    pad[:h, :w] = m
    on = pad.reshape(gy, block, gx, block).mean((1, 3)) > 0.3
    return on if on.any() else None


def pixelate(frame: np.ndarray, on: np.ndarray, block: int) -> np.ndarray:
    out = frame.copy()
    for by, bx in zip(*np.nonzero(on)):
        cell = out[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
        cell[:] = cell.reshape(-1, 3).mean(0)
    return out


def _batches(src: Path, ffmpeg: str, batch: int):
    from kukkiri.video_io import iter_frames

    chunk = []
    for f in iter_frames(src, ffmpeg, chunk=batch):
        chunk.append(f)
        if len(chunk) == batch:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def add_mosaic(src: str | Path, dst: str | Path, detector, block: int | None = None, hold: int = 6, ffmpeg: str = "ffmpeg",
           batch: int = 16, progress=None) -> tuple[int, int]:
    from kukkiri.video_io import encode_frames, extract_audio, probe

    src, dst = Path(src), Path(dst)
    info = probe(src)
    h, w = info["height"], info["width"]
    expected = info["nb_frames"] or round(info["fps"] * info["duration"])
    block = block or max(8, round(h / 36))

    grids: dict[int, np.ndarray] = {}
    n = 0
    for chunk in _batches(src, ffmpeg, batch):
        for k, dets in enumerate(detector.detect_frames(chunk, batch_size=batch)):
            g = block_grid(dets, (h, w), block)
            if g is not None:
                grids[n + k] = g
        n += len(chunk)
        if progress and expected:
            progress(min(49.0, 50 * n / expected))
    if not n:
        raise RuntimeError(f"no frames could be read from {src}")
    fps = info["fps"]
    if info["duration"] and abs(n / info["duration"] - fps) > 0.02 * fps:
        fps = n / info["duration"]
    found = sorted(grids)

    def grid_for(i: int, j: list[int]) -> np.ndarray | None:
        if i in grids or not found:
            return grids.get(i)
        while j[0] + 1 < len(found) and found[j[0] + 1] <= i:
            j[0] += 1
        near = min(found[j[0]:j[0] + 2], key=lambda k: abs(k - i))
        return grids[near] if abs(near - i) <= hold else None

    done = 0

    def out_frames():
        nonlocal done
        i, j = 0, [0]
        for chunk in _batches(src, ffmpeg, batch):
            for f in chunk:
                g = grid_for(i, j)
                if g is not None:
                    done += 1
                    f = pixelate(f, g, block)
                yield f
                i += 1
            if progress:
                progress(min(99.0, 50 + 50 * i / n))

    with tempfile.TemporaryDirectory(prefix="kukkiri_mosaic_", ignore_cleanup_errors=True) as tmp:
        audio = extract_audio(src, Path(tmp) / "audio.mka")
        encode_frames(out_frames(), fps, dst, audio=audio, crf=16, ffmpeg=ffmpeg)
    if progress:
        progress(100.0)
    return done, n
