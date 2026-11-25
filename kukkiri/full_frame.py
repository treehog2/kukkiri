from __future__ import annotations

from kukkiri.geometry import CLIP, SEAM, seam_weights

SIDE = 32


def _restore_clip(ltx, clip):
    import torch
    import torch.nn.functional as F

    h, w = clip.shape[1:3]
    x = F.pad(clip.permute(0, 3, 1, 2).float(), (0, -w % SIDE, 0, -h % SIDE), mode="replicate")
    try:
        out = ltx.restore_canvas(x.permute(0, 2, 3, 1).to(torch.uint8))
    except torch.OutOfMemoryError as e:
        raise MemoryError(f"not enough GPU memory to restore the full frame at {w}x{h}: use a smaller video (or "
                          "--max-1080p above 1080p), or restore without --full-frame") from e
    return torch.from_numpy(out[:, :h, :w]).to(clip.device).float()


def restore(frames, ltx, sink) -> int:
    import torch

    held, tail = [], None
    start = 0

    def emit(clip, last: bool):
        nonlocal tail, start
        sp = list(range(len(clip)))
        w = seam_weights(sp, tail is not None, not last)
        y = _restore_clip(ltx, torch.stack(clip))
        y *= torch.tensor([w[i] for i in sp], device=y.device).view(-1, 1, 1, 1)
        if tail is not None:
            y[:SEAM] += tail
        n = len(clip) if last else len(clip) - SEAM
        tail = None if last else y[n:].clone()
        done = (y[:n] + 0.5).clamp(0, 255).to(torch.uint8)
        for i in range(n):
            sink(start + i, done[i])
            frames.release(start + i)
        start += n
        return clip[n:]

    for batch in frames.iter_batches():
        held += batch
        while len(held) > CLIP:
            rest = held[CLIP:]
            held = emit(held[:CLIP], False) + rest
    if held:
        emit(held, True)
    return start
