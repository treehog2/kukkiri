from __future__ import annotations

import math

import numpy as np

LAYOUTS, LENSES = ("sbs",), ("equirect", "fisheye")
THRESHOLD = 0.15
SAME = 0.5


def eye_size(frame_w: int) -> int:
    return frame_w // 2


def eye_bounds(box, frame_w: int) -> tuple[int, int]:
    n = eye_size(frame_w)
    return (0, n) if (box[0] + box[2]) / 2 < n else (n, 2 * n)


def detect_eyes(detector, frames: list, batch_size: int = 32, fisheye: bool = False) -> list[list]:
    from kukkiri.detector import Detection

    n = eye_size(frames[0].shape[1])
    left = detector.detect_frames([f[:, :n] for f in frames], batch_size=batch_size)
    right = detector.detect_frames([f[:, n : 2 * n] for f in frames], batch_size=batch_size)
    out = []
    for dl, dr in zip(left, right):
        dets = []
        for d, side in [(d, 0) for d in dl] + [(d, 1) for d in dr]:
            m = d.mask
            if m is not None:
                z = np.zeros_like(m)
                m = np.concatenate([m, z] if side == 0 else [z, m], axis=1)
            x0, y0, x1, y1 = d.box
            dets.append(Detection((x0 + side * n, y0, x1 + side * n, y1), d.score, d.label, m))
        out.append(dets)
    if fisheye:
        for side in (0, 1):
            strips = [low_strip(f[:, side * n : (side + 1) * n]) for f in frames]
            for dets, found in zip(out, detector.detect_frames(strips, batch_size=batch_size)):
                for d in found:
                    b = strip_box_to_eye(d.box, n)
                    if b is not None:
                        _keep_stronger(dets, Detection((b[0] + side * n, b[1], b[2] + side * n, b[3]), d.score, d.label, None))
    return out


def _keep_stronger(dets: list, fd) -> None:
    def same(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return small > 0 and ix * iy >= SAME * small

    rivals = [d for d in dets if d.label == fd.label and d.mask is not None and same(d.box, fd.box)]
    if any(d.score >= fd.score for d in rivals):
        return
    for d in rivals:
        dets.remove(d)
    dets.append(fd)


def fisheye_share(track) -> float:
    return sum(f not in track.masks for f in track.seen) / max(len(track.seen), 1)


def _strip_origin(n: int) -> tuple[int, int, int]:
    m = n // 2
    return n // 4, n // 2, m


_GRIDS: dict = {}


def low_strip(eye):
    import torch
    import torch.nn.functional as F

    n = eye.shape[0]
    key = (n, eye.device)
    if key not in _GRIDS:
        u0, v0, m = _strip_origin(n)
        t = np.arange(m) + 0.5
        gu, gv = np.meshgrid(u0 + t, v0 + t)
        x, y = fisheye_to_equirect(gu, gv, n)
        inside = np.hypot(gu - n / 2, gv - n / 2) <= n / 2
        grid = np.stack([x / n * 2 - 1, y / n * 2 - 1], -1).astype(np.float32)
        _GRIDS[key] = (torch.from_numpy(grid)[None].to(eye.device), torch.from_numpy(inside).to(eye.device))
    grid, inside = _GRIDS[key]
    out = F.grid_sample(eye.permute(2, 0, 1)[None].float(), grid, mode="bilinear", padding_mode="border",
                        align_corners=False)[0]
    return (out * inside).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()


def strip_box_to_eye(box, n: int):
    u0, v0, _ = _strip_origin(n)
    xs = np.linspace(box[0], box[2], 9) + u0
    ys = np.linspace(box[1], box[3], 9) + v0
    gu, gv = np.meshgrid(xs, ys)
    keep = np.hypot(gu - n / 2, gv - n / 2) <= n / 2
    if not keep.any():
        return None
    x, y = fisheye_to_equirect(gu[keep], gv[keep], n)
    x0, y0 = int(max(np.floor(x.min()), 0)), int(max(np.floor(y.min()), 0))
    x1, y1 = int(min(np.ceil(x.max()), n)), int(min(np.ceil(y.max()), n))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None




def _xp(a):
    if isinstance(a, np.ndarray) or np.isscalar(a):
        return np
    import torch

    return torch


def equirect_to_fisheye(x, y, n: int):
    xp = _xp(x)
    lon = (x / n - 0.5) * math.pi
    lat = (0.5 - y / n) * math.pi
    dx, dy, dz = xp.cos(lat) * xp.sin(lon), xp.sin(lat), xp.cos(lat) * xp.cos(lon)
    theta = xp.arccos(xp.clip(dz, -1.0, 1.0))
    r = theta / (math.pi / 2) * (n / 2)
    phi = xp.arctan2(dy, dx)
    return n / 2 + r * xp.cos(phi), n / 2 - r * xp.sin(phi)


def fisheye_to_equirect(u, v, n: int):
    xp = _xp(u)
    px, py = u - n / 2, n / 2 - v
    theta = xp.hypot(px, py) / (n / 2) * (math.pi / 2)
    phi = xp.arctan2(py, px)
    dx, dy, dz = xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi), xp.cos(theta)
    lon = xp.arctan2(dx, dz)
    lat = xp.arcsin(xp.clip(dy, -1.0, 1.0))
    return (lon / math.pi + 0.5) * n, (0.5 - lat / math.pi) * n


def fisheye_box(rect, n: int) -> tuple[float, float, float]:
    e0 = (rect[0] // n) * n
    xs = np.linspace(rect[0] - e0, rect[2] - e0, 33)
    ys = np.linspace(rect[1], rect[3], 33)
    gx, gy = np.meshgrid(xs, ys)
    u, v = equirect_to_fisheye(gx, gy, n)
    side = max(u.max() - u.min(), v.max() - v.min(), 1.0)
    cu, cv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    return cu - side / 2, cv - side / 2, side


def to_canvas(frame, rect, size: int, n: int):
    import torch
    import torch.nn.functional as F

    u0, v0, side = box = fisheye_box(rect, n)
    e0 = (rect[0] // n) * n
    m = max(size, int(math.ceil(side)))
    t = (torch.arange(m, device=frame.device, dtype=torch.float64) + 0.5) / m * side
    gv, gu = torch.meshgrid(v0 + t, u0 + t, indexing="ij")
    x, y = fisheye_to_equirect(gu, gv, n)
    x = x.clamp(0, n - 1e-3) + e0
    y = y.clamp(0, frame.shape[0] - 1e-3)
    bx0, by0 = max(int(x.min()) - 2, 0), max(int(y.min()) - 2, 0)
    bx1, by1 = min(int(x.max()) + 3, frame.shape[1]), min(int(y.max()) + 3, frame.shape[0])
    src = frame[by0:by1, bx0:bx1].permute(2, 0, 1)[None].float()
    h, w = by1 - by0, bx1 - bx0
    grid = torch.stack([(x - bx0) / w * 2 - 1, (y - by0) / h * 2 - 1], -1).float()
    out = F.grid_sample(src, grid[None], mode="bicubic", padding_mode="border", align_corners=False)
    if m != size:
        out = F.interpolate(out, size=(size, size), mode="area")
    return out[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0), box


def from_canvas(canvas, rect, box, n: int):
    import torch
    import torch.nn.functional as F

    u0, v0, side = box
    e0 = (rect[0] // n) * n
    dev = canvas.device
    xs = torch.arange(rect[0], rect[2], device=dev, dtype=torch.float64) + 0.5 - e0
    ys = torch.arange(rect[1], rect[3], device=dev, dtype=torch.float64) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u, v = equirect_to_fisheye(gx, gy, n)
    grid = torch.stack([(u - u0) / side * 2 - 1, (v - v0) / side * 2 - 1], -1).float()
    src = canvas.permute(2, 0, 1)[None].float()
    return F.grid_sample(src, grid[None], mode="bicubic", padding_mode="border", align_corners=False)[0]
