from __future__ import annotations

import os

import numpy as np

from kukkiri.constants import RESTORE_CROP_SIZE
from kukkiri.geometry import (
    CLIP,
    SEAM,
    ClipCanvas,
    blend_mask,
    build_clip_canvas,
    build_clip_canvas_gpu,
    seam_weights,
    uncanvas_frame,
)
from kukkiri.tracker import Track

CANVAS_SIZE = RESTORE_CROP_SIZE

__all__ = ["CANVAS_SIZE", "ClipCanvas", "JobSource", "SpanEmitter", "restore_stream", "uncanvas_frame"]


class SpanEmitter:

    def __init__(self, clip: int = CLIP, seam: int = SEAM, min_length: int = 1):
        self.clip, self.seam, self.min_length = clip, seam, min_length
        self._next: dict[int, int] = {}

    def next_start(self, tr: Track) -> int:
        return self._next.get(tr.track_id, tr.start)

    def poll(self, tr: Track, confirmed: int) -> list[tuple]:
        jobs = []
        if confirmed - tr.start + 1 < self.min_length:
            return jobs
        while True:
            s = self.next_start(tr)
            e = s + self.clip - 1
            if confirmed <= e:
                return jobs
            sp = list(range(s, e + 1))
            jobs.append((e, tr, sp, seam_weights(sp, tr.track_id in self._next, True, self.seam)))
            self._next[tr.track_id] = e + 1 - self.seam

    def close(self, tr: Track) -> list[tuple]:
        s = self.next_start(tr)
        has_parent = tr.track_id in self._next
        self._next.pop(tr.track_id, None)
        e = min(s + self.clip - 1, tr.end)
        sp = list(range(s, e + 1))
        return [(e, tr, sp, seam_weights(sp, has_parent, False, self.seam))]


def _accumulate_job_gpu(track: Track, sp, wts, cc, restored, scale, acc: dict) -> None:
    import torch.nn.functional as F

    for k, fi in enumerate(sp):
        w = wts.get(fi, 0.0)
        if w <= 0:
            continue
        top, left = cc.pads[k]
        th, tw = cc.sizes[k]
        x0, y0, x1, y1 = cc.rects[k]
        crop = restored[k, top * scale : (top + th) * scale, left * scale : (left + tw) * scale]
        crop = crop.permute(2, 0, 1)[None].float()
        crop = F.interpolate(crop, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False, antialias=False)[0]
        key = (track.track_id, fi)
        if key in acc:
            s, tot, rect = acc[key]
            acc[key] = (s + w * crop, tot + w, rect)
        else:
            acc[key] = (w * crop, w, cc.rects[k])


def _blend_frame_gpu(fi: int, tracks: "list[Track]", acc: dict, frame: np.ndarray, frame_hw) -> bool:
    import torch

    touched = False
    for tr in tracks:
        key = (tr.track_id, fi)
        if key not in acc:
            continue
        s, tot, rect = acc.pop(key)
        crop = s / max(tot, 1e-6)
        x0, y0, x1, y1 = rect
        m = torch.from_numpy(blend_mask(tr.masks.get(fi), rect, frame_hw)).to(crop.device).permute(2, 0, 1)
        if isinstance(frame, torch.Tensor):
            region = frame[y0:y1, x0:x1].permute(2, 0, 1).float()
            frame[y0:y1, x0:x1] = (region * (1.0 - m) + crop * m).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            touched = True
            continue
        region = torch.from_numpy(np.ascontiguousarray(frame[y0:y1, x0:x1])).to(crop.device).permute(2, 0, 1).float()
        out = (region * (1.0 - m) + crop * m).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        frame[y0:y1, x0:x1] = out.cpu().numpy()
        touched = True
    return touched


class JobSource:

    def __init__(self, n_frames: int, maxsize: int = 0):
        import queue
        import threading

        self.q: queue.Queue = queue.Queue(maxsize)
        self.lock = threading.Lock()
        self.pending = np.zeros(n_frames, np.int32)
        self.by_frame: dict[int, list[Track]] = {}
        self.frontier = 0
        self.n_jobs = 0

    def put(self, job: tuple) -> None:
        _, tr, sp, _ = job
        with self.lock:
            for fi in sp:
                self.pending[fi] += 1
                lst = self.by_frame.setdefault(fi, [])
                if tr not in lst:
                    lst.append(tr)
        self.n_jobs += 1
        self.q.put(job)

    def advance(self, frontier: int) -> None:
        self.frontier = max(self.frontier, frontier)

    def close(self, n_frames: int) -> None:
        self.frontier = n_frames
        self.q.put(None)


class StreamStalled(RuntimeError):
    pass


def restore_stream(source: JobSource, frames, restorer, sink) -> set[int]:
    pending, by_frame = source.pending, source.by_frame
    frame_hw = getattr(frames, "hw", None) or frames[0].shape[:2]
    acc: dict = {}
    touched: set[int] = set()
    nxt = 0

    import time as _time

    tstats: dict = {
        "q2_wait": 0.0,
        "accumulate": 0.0,
        "flush": 0.0,
        "blend": 0.0,
        "sink": 0.0,
        "frames": 0,
        "groups": {},
    }
    release = getattr(frames, "release", None)

    import torch

    def flush() -> None:
        nonlocal nxt
        while nxt < source.frontier and pending[nxt] == 0:
            f = frames[nxt]
            if isinstance(f, torch.Tensor):
                f.record_stream(torch.cuda.current_stream())
            t0 = _time.time()
            with source.lock:
                trs = by_frame.pop(nxt, ())
            if trs and _blend_frame_gpu(nxt, sorted(trs, key=lambda t: (t.start, t.track_id)), acc, f, frame_hw):
                touched.add(nxt)
            t1 = _time.time()
            sink(nxt, f)
            tstats["blend"] += t1 - t0
            tstats["sink"] += _time.time() - t1
            tstats["frames"] += 1
            if release is not None:
                release(nxt)
            nxt += 1

    import queue
    import threading

    refiner = getattr(restorer, "refiner", None)
    device = getattr(restorer, "device", "cuda")
    q1: queue.Queue = queue.Queue(maxsize=2)
    q2: queue.Queue = queue.Queue(maxsize=2)
    err: list[BaseException] = []

    def refine_stage() -> None:
        try:
            stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            while not err:
                try:
                    item = q1.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    while not err:
                        try:
                            q2.put(None, timeout=5)
                            return
                        except queue.Full:
                            continue
                    return
                tr, sp, wts, cc, restored = item
                scale = 1
                if refiner is not None:
                    with torch.cuda.stream(stream):
                        restored = refiner.refine_clip(restored, to_numpy=False)
                        stream.synchronize()
                    scale = refiner.scale
                else:
                    with torch.cuda.stream(stream):
                        restored = torch.from_numpy(restored).to(device)
                        stream.synchronize()
                while not err:
                    try:
                        q2.put((tr, sp, wts, cc, restored, scale), timeout=5)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            err.append(e)
            try:
                q2.put(None, timeout=1)
            except queue.Full:
                pass

    def tail_stage() -> None:
        try:
            tail_stream = torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
            with torch.cuda.stream(tail_stream):
                while True:
                    t0 = _time.time()
                    try:
                        item = q2.get(timeout=0.2)
                    except queue.Empty:
                        tstats["q2_wait"] += _time.time() - t0
                        if err:
                            return
                        flush()
                        continue
                    tstats["q2_wait"] += _time.time() - t0
                    if item is None or err:
                        torch.cuda.current_stream().synchronize()
                        return
                    tr, sp, wts, cc, restored, scale = item
                    restored.record_stream(torch.cuda.current_stream())
                    t0 = _time.time()
                    _accumulate_job_gpu(tr, sp, wts, cc, restored, scale, acc)
                    tstats["accumulate"] += _time.time() - t0
                    with source.lock:
                        for fi in sp:
                            pending[fi] -= 1
                    t0 = _time.time()
                    flush()
                    tstats["flush"] += _time.time() - t0
        except BaseException as e:
            err.append(e)

    stall_s = float(os.environ.get("KUKKIRI_STALL_S", "900"))
    stop_watch = threading.Event()

    def watchdog() -> None:
        last, since = -1, _time.time()
        while not stop_watch.wait(5):
            if err:
                return
            if nxt != last:
                last, since = nxt, _time.time()
            elif _time.time() - since > stall_s:
                import faulthandler
                import sys

                msg = (
                    f"restore stalled: no frame finished for {stall_s:.0f} s at frame {nxt} (frontier "
                    f"{source.frontier}, decoded {getattr(frames, 'decoded', '?')}, frames {len(frames)})"
                )
                if sys.stderr is not None:
                    print(msg + "; thread stacks follow", file=sys.stderr, flush=True)
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                err.append(StreamStalled(msg))
                return

    threads = [threading.Thread(target=refine_stage, daemon=True), threading.Thread(target=tail_stage, daemon=True)]
    for th in threads:
        th.start()
    threading.Thread(target=watchdog, daemon=True).start()
    B = max(1, int(os.environ.get("KUKKIRI_PRIMARY_BATCH", "8")))
    if hasattr(restorer, "graph_batch"):
        restorer.graph_batch = B

    max_ahead = max(
        2 * CLIP, int(float(os.environ.get("KUKKIRI_FRAME_BUDGET_GB", "6")) * 2**30 // (frame_hw[0] * frame_hw[1] * 3))
    )

    def run_group(group) -> None:
        _bcc = (
            build_clip_canvas_gpu
            if getattr(frames, "device", None) is not None or isinstance(frames[group[0][2][0]], torch.Tensor)
            else build_clip_canvas
        )
        ccs = [_bcc(tr, sp, frames, CANVAS_SIZE) for _, tr, sp, _ in group]
        tstats["groups"][len(group)] = tstats["groups"].get(len(group), 0) + 1
        if hasattr(restorer, "restore_canvases"):
            outs = restorer.restore_canvases([cc.crops for cc in ccs])
        else:
            outs = [restorer.restore_canvas(cc.crops) for cc in ccs]
        for (_, tr, sp, wts), cc, restored in zip(group, ccs, outs):
            while not err:
                try:
                    q1.put((tr, sp, wts, cc, restored), timeout=5)
                    break
                except queue.Full:
                    continue

    full: list = []

    def run_full() -> None:
        nonlocal full
        grp, full = full, []
        while grp[0][0] - nxt > max_ahead and not err:
            _time.sleep(0.02)
        run_group(grp)

    def make_room(end: int) -> None:
        while end - nxt > max_ahead and not err:
            if full:
                run_full()
            else:
                _time.sleep(0.02)

    try:
        while not err:
            try:
                job = source.q.get(timeout=0.5)
            except queue.Empty:
                if full:
                    run_full()
                continue
            if job is None:
                break
            if len(job[2]) == CLIP:
                full.append(job)
                if len(full) == B:
                    run_full()
            else:
                make_room(job[0])
                run_group([job])
        if full and not err:
            run_full()
    except BaseException as e:
        err.append(e)
        raise
    while True:
        try:
            q1.put(None, timeout=5)
            break
        except queue.Full:
            if err:
                break
    for th in threads:
        while th.is_alive() and not err:
            th.join(timeout=1)
        th.join(timeout=30 if err else None)
    stop_watch.set()
    if err:
        raise err[0]
    flush()
    if os.environ.get("KUKKIRI_TIMING"):
        print("tail timing", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tstats.items()}, flush=True)
    return touched
