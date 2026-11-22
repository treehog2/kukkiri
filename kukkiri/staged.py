from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from kukkiri.streaming import JobSource, routes_to_ltx


class _Recorder(JobSource):

    def __init__(self, n_frames: int, frames):
        super().__init__(n_frames)
        self.frames, self.events, self._seen = frames, [], 0

    def put(self, job: tuple) -> None:
        self.events.append(("job", job))
        self.n_jobs += 1

    def advance(self, frontier: int) -> None:
        self.events.append(("advance", frontier))
        self.frames.release(self._seen)
        self._seen += 1

    def close(self, n_frames: int) -> None:
        self.events.append(("close", n_frames))


def plan(produce, frames, n_frames: int) -> list:
    rec = _Recorder(n_frames, frames)
    produce(rec)
    return rec.events


def ltx_pass(events: list, frames, ltx, min_size: float, work: Path, progress=None) -> dict:
    from kukkiri.geometry import build_fixed_canvas_gpu

    jobs = sorted(((sp[0], sp[-1], tr, sp) for kind, job in events if kind == "job"
                   for _, tr, sp, _ in [job] if routes_to_ltx(tr, sp, min_size, frames.hw[0])),
                  key=lambda j: (j[0], j[1]))
    out, k, head = {}, 0, 0
    work.mkdir(parents=True, exist_ok=True)

    def run_ready(final: bool):
        nonlocal k
        while k < len(jobs) and (final or jobs[k][1] < head):
            _, _, tr, sp = jobs[k]
            cc = build_fixed_canvas_gpu(tr, sp, frames, ltx.canvas)
            p = work / f"ltx_{tr.track_id}_{sp[0]}.npy"
            np.save(p, ltx.restore_canvas(cc.crops))
            out[(tr.track_id, sp[0])] = p
            k += 1
            if progress:
                progress(k, len(jobs))
        low = jobs[k][0] if k < len(jobs) else head
        for i in range(released[0], low):
            frames.release(i)
        released[0] = max(released[0], low)

    released = [0]
    if jobs:
        for batch in frames.iter_batches():
            head += len(batch)
            run_ready(False)
            if k == len(jobs):
                break
        run_ready(True)
    return out


class ReplayLtx:

    canvas = 512
    needs_pixels = False

    def __init__(self, results: dict):
        self.results = results

    def warm(self, n_frames: int) -> None:
        pass

    def restore_clip(self, tr, sp, cc):
        return np.load(self.results[(tr.track_id, sp[0])])


def replay(events: list, frames, source: JobSource) -> None:
    it = iter(events)
    for batch in frames.iter_batches():
        for _ in batch:
            for kind, payload in it:
                if kind == "job":
                    source.put(payload)
                    continue
                if kind == "advance":
                    source.advance(payload)
                    break
                source.close(payload)
                return
    for kind, payload in it:
        if kind == "job":
            source.put(payload)
        elif kind == "close":
            source.close(payload)
            return


def release_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
