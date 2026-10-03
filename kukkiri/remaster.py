
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)

MOSAIC_LABEL = 0


@dataclass(frozen=True)
class RemasterConfig:
    detector_size: str = "seg_medium"
    batch: int = 8
    max_pixels: int | None = None
    gap: int = 6
    min_length: int = 2
    qp: int = 14
    bitrate_factor: float = 1.5
    min_bitrate: int = 2_000_000
    max_bitrate: int = 50_000_000
    preset: str = "P7"
    cut_tv: float = 0.3


@dataclass
class ProgressEvent:

    pct: float
    stage: str
    frames_done: int = 0
    frames_total: int = 0
    eta_s: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]
LiveCallback = Callable[[Path], None]


@dataclass
class RemasterResult:
    output: str
    duration_s: float
    frames: int
    frames_touched: int
    n_tracks: int
    n_clips: int
    detect_s: float
    remaster_s: float
    total_s: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    source_bitrate: int = 0
    output_bitrate: int = 0
    output_bytes: int = 0
    tracks: list[tuple[int, int, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_models(weights_dir: Path, cfg: RemasterConfig, threshold: float, batch: int):
    from kukkiri.detector import CensorDetector, RFDETRSize
    from kukkiri.restorer import VideoRestorer
    from kukkiri.weights import detector_checkpoint, restorer_weights

    detector = CensorDetector(detector_checkpoint(weights_dir), size=RFDETRSize(cfg.detector_size), threshold=threshold)
    detector.fast = True
    detector.trt = True
    restorer = VideoRestorer(restorer_weights(weights_dir), device="cuda", trt_steps=True, engine_dir=weights_dir)
    restorer.graph_batch = batch
    os.environ.setdefault("KUKKIRI_PRIMARY_BATCH", str(batch))
    return detector, restorer


def _output_size(height: int, width: int, max_pixels: int | None) -> tuple[int, int] | None:
    if not max_pixels or height * width <= max_pixels:
        return None
    f = (1920 * 1080 / (height * width)) ** 0.5
    return int(height * f / 2) * 2, int(width * f / 2) * 2


def _cut_flags(batch, prev, cut_tv: float):
    import torch
    import torch.nn.functional as F

    x = F.interpolate(torch.stack(list(batch)).permute(0, 3, 1, 2).float(), size=(36, 64), mode="area")
    q = (x / 32).long().clamp(0, 7)
    idx = (q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]).view(len(batch), -1)
    h = torch.zeros(len(batch), 512, device=x.device).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
    h = h / idx.shape[1]
    prevs = torch.cat([h[:1] if prev is None else prev[None], h[:-1]])
    tv = 0.5 * (h - prevs).abs().sum(1)
    return (tv > cut_tv).tolist(), h[-1]


def _produce_jobs(detector, frames, source, gap: int, min_length: int, stats: dict, cut_tv: float) -> None:
    from kukkiri.tracker import GappedTracker
    from kukkiri.streaming import SpanEmitter

    bq: queue.Queue = queue.Queue(maxsize=6)

    def feed():
        try:
            for batch in frames.iter_batches():
                bq.put(batch)
        except BaseException as e:
            stats.setdefault("error", e)
        finally:
            bq.put(None)

    th = threading.Thread(target=feed, daemon=True)
    th.start()
    import torch

    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            tk = GappedTracker(max_gap=gap, min_length=min_length)
            _detect_loop(detector, bq, tk, SpanEmitter(min_length=min_length), source, stats, cut_tv)
    except BaseException as e:
        stats.setdefault("error", e)
    finally:
        while th.is_alive():
            try:
                bq.get(timeout=0.5)
            except queue.Empty:
                pass
        th.join()
        source.close(stats["n_frames"])


def _track_span(tr) -> tuple[int, int, float]:
    area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in tr.boxes.values()) / max(len(tr.boxes), 1)
    return tr.start, tr.end, float(area)


def _detect_loop(detector, bq, tk, em, source, stats, cut_tv: float) -> None:
    from kukkiri.tracker import FACE_LABEL, is_face_track

    fi, prev_hist = 0, None
    stats.setdefault("n_cuts", 0)

    def close(tracks):
        for tr in tracks:
            if is_face_track(tr):
                stats["n_face_tracks"] = stats.get("n_face_tracks", 0) + 1
                continue
            stats["n_tracks"] += 1
            stats["tracks"].append(_track_span(tr))
            for job in em.close(tr):
                source.put(job)

    try:
        while True:
            batch = bq.get()
            if batch is None or fi >= len(source.pending):
                break
            cuts, prev_hist = _cut_flags(batch, prev_hist, cut_tv)
            for cut, dets in zip(cuts, detector.detect_frames(batch, batch_size=32)):
                dets = [d for d in dets if d.label in (MOSAIC_LABEL, FACE_LABEL)]
                if fi >= len(source.pending):
                    break
                if cut and fi > 0:
                    stats["n_cuts"] += 1
                    close(tk.finish())
                stats["n_detections"] += len(dets)
                close(tk.push(fi, dets))
                for tr in tk.active:
                    if is_face_track(tr):
                        continue
                    for job in em.poll(tr, tk.confirmed[tr.track_id]):
                        source.put(job)
                source.advance(min([em.next_start(tr) for tr in tk.active] + [fi + 1]))
                fi += 1
        close(tk.finish())
    finally:
        stats["n_frames"] = fi


def remaster(
    src: str | Path,
    dst: str | Path,
    weights_dir: str | Path,
    *,
    config: RemasterConfig = RemasterConfig(),
    threshold: float | None = None,
    ffmpeg: str = "ffmpeg",
    progress: ProgressCallback | None = None,
    scratch: str | Path | None = None,
    live_dir: str | Path | None = None,
    on_live_file: LiveCallback | None = None,
) -> RemasterResult:
    from kukkiri.mux import LiveMuxer
    from kukkiri.streaming import (
        CANVAS_SIZE,
        JobSource,
        restore_stream,
    )
    from kukkiri.video_io import probe
    from kukkiri.video_io_gpu import (
        GpuVideoWriter,
        SharedFrameStore,
        gpu_readable_source,
    )

    t0 = time.time()
    cfg = config
    batch, gap, min_length, qp = cfg.batch, cfg.gap, cfg.min_length, cfg.qp
    bitrate_factor, min_bitrate, max_bitrate = cfg.bitrate_factor, cfg.min_bitrate, cfg.max_bitrate
    src, dst, weights_dir = Path(src), Path(dst), Path(weights_dir)
    work = Path(tempfile.mkdtemp(prefix="kukkiri_", dir=str(scratch) if scratch else None))
    readable = gpu_readable_source(src, work, ffmpeg=ffmpeg)
    info = probe(readable)
    input_bitrate = info.get("bit_rate") if readable == src else probe(src).get("bit_rate")
    src = readable
    fps = float(info.get("fps") or 30.0)
    n_frames = int(info.get("nb_frames") or 0)
    height = int(info.get("height") or 0)
    if threshold is None:
        threshold = 0.2 if 0 < height < 720 else 0.35
    if progress:
        progress(ProgressEvent(0.0, "preparing", 0, n_frames))
    detector, restorer = _load_models(weights_dir, cfg, threshold, batch)
    logger.info(
        f"remaster {src.name}: {n_frames} frames @ {fps:.3f} fps, {height}p, thr {threshold}, batch {batch}"
    )

    t2 = time.time()
    scale_to = _output_size(height, int(info.get("width") or 0), cfg.max_pixels)
    frames = SharedFrameStore(src, scale_to=scale_to)
    n_frames = n_frames or len(frames)
    if scale_to:
        logger.info(f"source {height}p served at {scale_to[0]}p ({scale_to[1]}x{scale_to[0]})")
    src_bitrate = input_bitrate
    if src_bitrate and scale_to:
        src_bitrate = src_bitrate * (scale_to[0] * scale_to[1]) / (height * int(info.get("width") or 1))
    cap = int(min(max(bitrate_factor * src_bitrate, min_bitrate), max_bitrate)) if src_bitrate else None
    logger.info(f"encode: source {(input_bitrate or 0) / 1e6:.2f} Mbps -> cap {cap / 1e6 if cap else None} Mbps")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mux = LiveMuxer(src, dst, fps, n_frames / fps, live_dir=Path(live_dir) if live_dir else None, ffmpeg=ffmpeg)
    writer = GpuVideoWriter(
        work / "video.mp4", frames.hw[1], frames.hw[0], fps, qp=qp, ffmpeg=ffmpeg, max_bitrate=cap, preset=cfg.preset,
        stream=mux.stdin, idr_period=mux.idr_period(fps) if live_dir else None,
    )
    done = 0
    last = [0.0]

    def publish_live():
        if on_live_file:
            for path in mux.new_files():
                on_live_file(path)

    def sink(fi, f):
        nonlocal done
        writer.write(f)
        done += 1
        now = time.time()
        if now - last[0] > 2:
            last[0] = now
            publish_live()
            if progress and n_frames:
                rate = done / max(now - t2, 1e-3)
                eta = (n_frames - done) / rate * 1.05 if done > 32 else None
                progress(ProgressEvent(100 * done / n_frames, "remaster", done, n_frames, eta))

    try:
        if detector.trt and not hasattr(detector, "_trt_ctx"):
            detector._trt_init()
        restorer.warm(CANVAS_SIZE)
        pstats = {"n_frames": 0, "n_detections": 0, "n_tracks": 0, "tracks": []}
        source = JobSource(len(frames), maxsize=256)
        producer = threading.Thread(
            target=_produce_jobs, args=(detector, frames, source, gap, min_length, pstats, cfg.cut_tv), daemon=True
        )
        producer.start()
        touched = restore_stream(source, frames, restorer, sink)
        producer.join()
        if "error" in pstats:
            raise pstats["error"]
        writer.close()
    except BaseException:
        writer.abort()
        mux.kill()
        raise
    remaster_s = time.time() - t2
    n_frames = pstats["n_frames"]
    d = {
        **{k: v for k, v in pstats.items() if k != "tracks"},
        "n_windows": source.n_jobs,
        "n_frames_touched": len(touched),
    }
    logger.info(f"remaster (detect + restore + encode): {remaster_s:.1f} s, {json.dumps(d)}")

    if progress:
        progress(ProgressEvent(99.0, "remux", n_frames, n_frames, 5.0))
    mux.close()
    publish_live()
    shutil.rmtree(work, ignore_errors=True)
    if progress:
        progress(ProgressEvent(100.0, "remux", n_frames, n_frames, 0.0))
    return RemasterResult(
        output=str(dst),
        duration_s=float(info.get("duration") or (n_frames / fps if fps else 0.0)),
        frames=n_frames,
        frames_touched=int(d.get("n_frames_touched", 0)),
        n_tracks=int(d.get("n_tracks", 0)),
        n_clips=int(d.get("n_windows", 0)),
        detect_s=0.0,
        remaster_s=round(remaster_s, 1),
        total_s=round(time.time() - t0, 1),
        width=frames.hw[1],
        height=frames.hw[0],
        fps=round(fps, 3),
        source_bitrate=int(input_bitrate or 0),
        output_bitrate=int(probe(dst).get("bit_rate") or 0),
        output_bytes=dst.stat().st_size,
        tracks=pstats["tracks"],
    )
