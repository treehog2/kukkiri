
from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Callable

from kukkiri.errors import SourceTruncated

logger = logging.getLogger(__name__)

MOSAIC_LABEL = 0


@dataclass(frozen=True)
class RemasterConfig:
    detector_size: str = "seg_medium"
    batch: int = 8
    max_pixels: int | None = None
    threshold_scale: float = 1.0
    gap: int = 6
    min_length: int = 2
    qp: int = 14
    bitrate_factor: float = 1.5
    min_bitrate: int = 2_000_000
    max_bitrate: int = 50_000_000
    preset: str = "P7"
    cut_tv: float = 0.3
    ltx: bool = False
    ltx_min_size: float = 0.09
    max_fps: float | None = None
    vr: str | None = None
    vr_lens: str = "equirect"


VR_MAX_BITRATE = 80_000_000


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


LTX_LOW_VRAM_GB = 32
LTX_MIN_VRAM_GB = 16


def _frame_budget() -> int:
    return int(float(os.environ.get("KUKKIRI_FRAME_BUDGET_GB", "6")) * 2**30)


LTX_STAGED_GB = 15


def _ltx_staged() -> bool:
    import torch

    env = os.environ.get("KUKKIRI_LTX_STAGED")
    if env:
        return env == "1"
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gb < LTX_STAGED_GB:
        os.environ.setdefault("KUKKIRI_FRAME_BUDGET_GB", "1.5")
        logger.info(f"kiyo-ltx staged mode ({gb:.0f} GB GPU): detection, LTX and kiyo run as separate passes (slower)")
        return True
    return False


def _ltx_resident_blocks() -> int | None:
    import torch

    env = os.environ.get("KUKKIRI_LTX_RESIDENT_BLOCKS")
    if env:
        return int(env)
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gb >= LTX_LOW_VRAM_GB:
        return None
    keep = max(0, min(48, int(3 * (gb - LTX_MIN_VRAM_GB))))
    os.environ.setdefault("KUKKIRI_FRAME_BUDGET_GB", "2")
    logger.info(f"kiyo-ltx low-VRAM mode ({gb:.0f} GB GPU): {keep} of 48 transformer blocks stay on the GPU, the rest "
                "are streamed from system memory (slower)")
    return keep


def _load_models(weights_dir: Path, cfg: RemasterConfig, threshold: float, batch: int):
    from kukkiri.detector import CensorDetector, RFDETRSize
    from kukkiri.restorer import VideoRestorer
    from kukkiri.weights import detector_checkpoint, restorer_weights

    compile_ok = cfg.ltx and importlib.util.find_spec("triton") is not None and os.environ.get("KUKKIRI_LTX_COMPILE", "1") != "0"
    if compile_ok:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir

        cur = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if cur is None or os.path.abspath(cur) == os.path.abspath(default_cache_dir()):
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(weights_dir) / "inductor")
    det_dir = Path(weights_dir) / "vr" if cfg.vr else weights_dir
    detector = CensorDetector(detector_checkpoint(det_dir), size=RFDETRSize(cfg.detector_size), threshold=threshold)
    detector.fast = True
    detector.trt = True
    restorer = VideoRestorer(restorer_weights(weights_dir), device="cuda", trt_steps=True, engine_dir=weights_dir)
    restorer.graph_batch = batch
    os.environ.setdefault("KUKKIRI_PRIMARY_BATCH", str(batch))
    if cfg.ltx:
        from kukkiri.ltx import LtxRestorer

        if _ltx_staged():
            restorer.ltx = None
            restorer.make_ltx = lambda: LtxRestorer(weights_dir, compile=compile_ok, resident_blocks=0)
        else:
            restorer.ltx = LtxRestorer(weights_dir, compile=compile_ok, resident_blocks=_ltx_resident_blocks())
        restorer.ltx_min_size = cfg.ltx_min_size
    return detector, restorer


ENCODE_MAX_SIDE = 4096


def _output_size(height: int, width: int, max_pixels: int | None) -> tuple[int, int] | None:
    f = (1920 * 1080 / (height * width)) ** 0.5 if max_pixels and height * width > max_pixels else 1.0
    f = min(f, ENCODE_MAX_SIDE / max(height, width, 1))
    if f >= 1.0:
        return None
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


def _produce_jobs(detector, frames, source, gap: int, min_length: int, stats: dict, cut_tv: float, vr=None,
                  fisheye: bool = False) -> None:
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
            _detect_loop(detector, bq, tk, SpanEmitter(min_length=min_length), source, stats, cut_tv, vr, fisheye)
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


def _detect_loop(detector, bq, tk, em, source, stats, cut_tv: float, vr=None, fisheye: bool = False) -> None:
    from kukkiri.tracker import FACE_LABEL, is_face_track

    fi, prev_hist = 0, None
    stats.setdefault("n_cuts", 0)

    def place(tr):
        if not vr:
            return
        from kukkiri.vr import eye_bounds, eye_size, fisheye_share

        w = source_w[0]
        if not hasattr(tr, "bounds"):
            tr.bounds = eye_bounds(tr.boxes[min(tr.boxes)], w)
        tr.fisheye = eye_size(w) if fisheye and fisheye_share(tr) > 0.5 else None

    source_w = [0]

    def close(tracks):
        for tr in tracks:
            place(tr)
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
            if vr:
                from kukkiri.vr import detect_eyes

                source_w[0] = batch[0].shape[1]
                per_frame = detect_eyes(detector, batch, batch_size=32, fisheye=fisheye)
            else:
                per_frame = detector.detect_frames(batch, batch_size=32)
            for cut, dets in zip(cuts, per_frame):
                dets = [d for d in dets if d.label in (MOSAIC_LABEL, FACE_LABEL)]
                if fi >= len(source.pending):
                    break
                if cut and fi > 0:
                    stats["n_cuts"] += 1
                    close(tk.finish())
                stats["n_detections"] += len(dets)
                close(tk.push(fi, dets))
                for tr in tk.active:
                    place(tr)
                    for job in em.poll(tr, tk.confirmed[tr.track_id]):
                        if not is_face_track(tr):
                            source.put(job)
                source.advance(min([em.next_start(tr) for tr in tk.active] + [fi + 1]))
                fi += 1
        close(tk.finish())
    finally:
        stats["n_frames"] = fi


def _eta_s(done: int, n_frames: int, since_first_s: float) -> float | None:
    if done <= 32:
        return None
    return (n_frames - done) / ((done - 1) / max(since_first_s, 1e-3)) * 1.05


def remaster(src: str | Path, dst: str | Path, weights_dir: str | Path, **kwargs) -> RemasterResult:
    try:
        return _remaster_once(src, dst, weights_dir, **kwargs)
    except SourceTruncated as e:
        logger.warning(f"{e}: retrying from a repaired copy")
        from kukkiri.video_io import probe
        from kukkiri.video_io_gpu import repair_source

        work = Path(tempfile.mkdtemp(prefix="repair_", dir=str(kwargs["scratch"]) if kwargs.get("scratch") else None))
        try:
            repaired = repair_source(Path(src), work, ffmpeg=kwargs.get("ffmpeg", "ffmpeg"))
            got = int(probe(repaired).get("nb_frames") or 0)
            if got < e.expected - max(2, round(e.fps)):
                raise SourceTruncated(got, e.expected, e.fps) from e
            if kwargs.get("live_dir"):
                shutil.rmtree(kwargs["live_dir"], ignore_errors=True)
            return _remaster_once(repaired, dst, weights_dir, original=src, **kwargs)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _remaster_once(
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
    original: str | Path | None = None,
) -> RemasterResult:
    from kukkiri.mux import LiveMuxer
    from kukkiri.geometry import CLIP
    from kukkiri.streaming import (
        CANVAS_SIZE,
        JobSource,
        restore_stream,
    )
    from kukkiri.video_io import frame_repeats, probe
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
    input_bitrate = info.get("bit_rate") if readable == src and not original else probe(original or src).get("bit_rate")
    src = readable
    fps = float(info.get("fps") or 30.0)
    n_frames = int(info.get("nb_frames") or 0)
    height = int(info.get("height") or 0)
    if threshold is None and cfg.vr:
        from kukkiri.vr import THRESHOLD

        threshold = round(THRESHOLD * cfg.threshold_scale, 3)
    if threshold is None:
        threshold = round((0.2 if 0 < height < 720 else 0.35) * cfg.threshold_scale, 3)
    if cfg.vr:
        max_bitrate = max(max_bitrate, VR_MAX_BITRATE)
    if progress:
        progress(ProgressEvent(0.0, "preparing", 0, n_frames))
    detector, restorer = _load_models(weights_dir, cfg, threshold, batch)
    logger.info(
        f"remaster {src.name}: {n_frames} frames @ {fps:.3f} fps, {height}p, thr {threshold}, batch {batch}"
    )
    width = int(info.get("width") or 0)
    if not cfg.vr and height >= 960 and width == 2 * height:
        logger.warning(f"{src.name} looks like a side-by-side VR video: restore it with --vr sbs for VR's small "
                       "mosaics, adding --vr-lens fisheye for studios that pixelate the fisheye image (in the app: "
                       "Advanced, VR video)")

    t2 = time.time()
    scale_to = None if cfg.vr else _output_size(height, int(info.get("width") or 0), cfg.max_pixels)
    max_fps = cfg.max_fps if cfg.max_fps is not None else (30.0 if cfg.ltx else 0.0)
    stride = 1
    while max_fps and fps / stride > max_fps * 1.05:
        stride += 1
    if stride > 1:
        logger.info(f"{fps:.3f} fps source: restoring 1 frame in {stride} ({fps / stride:.3f} fps out)")
        fps, n_frames = fps / stride, -(-n_frames // stride)
    try:
        frames = SharedFrameStore(src, scale_to=scale_to, stride=stride, budget_bytes=_frame_budget())
    except Exception as e:
        import PyNvVideoCodec as nvc

        shutil.rmtree(work, ignore_errors=True)
        if not isinstance(e, nvc.PyNvVCException):
            raise
        logger.warning(f"{src.name}: GPU decoder cannot open the source: {' | '.join(str(e).strip().splitlines()[:3])}")
        raise SourceTruncated(0, n_frames, fps) from e
    n_frames = n_frames or len(frames)
    repeats = frame_repeats(src, fps, n_frames, float(info.get("duration") or 0)) if stride == 1 else None
    n_out = sum(repeats) if repeats else n_frames
    if repeats:
        logger.info(f"variable frame rate: {n_frames} frames over {info.get('duration')} s -> {n_out} at {fps:.3f} fps")
    if scale_to:
        logger.info(f"source {height}p served at {scale_to[0]}p ({scale_to[1]}x{scale_to[0]})")
    src_bitrate = input_bitrate
    if src_bitrate and scale_to:
        src_bitrate = src_bitrate * (scale_to[0] * scale_to[1]) / (height * int(info.get("width") or 1))
    cap = int(min(max(bitrate_factor * src_bitrate, min_bitrate), max_bitrate)) if src_bitrate else None
    logger.info(f"encode: source {(input_bitrate or 0) / 1e6:.2f} Mbps -> cap {cap / 1e6 if cap else None} Mbps")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mux = LiveMuxer(src, dst, fps, n_out / fps, live_dir=Path(live_dir) if live_dir else None, ffmpeg=ffmpeg)
    writer = GpuVideoWriter(
        work / "video.mp4", frames.hw[1], frames.hw[0], fps, qp=qp, ffmpeg=ffmpeg, max_bitrate=cap, preset=cfg.preset,
        stream=mux.stdin, idr_period=mux.idr_period(fps) if live_dir else None,
    )
    done = 0
    last = [0.0]
    first = [0.0]

    def publish_live():
        if on_live_file:
            for path in mux.new_files():
                on_live_file(path)

    def sink(fi, f):
        nonlocal done
        for _ in range(repeats[fi] if repeats and fi < len(repeats) else 1):
            writer.write(f)
        done += 1
        now = time.time()
        if done == 1:
            first[0] = now
        if now - last[0] > 2:
            last[0] = now
            publish_live()
            if progress and n_frames:
                eta = _eta_s(done, n_frames, now - first[0])
                progress(ProgressEvent(100 * done / n_frames, "remaster", done, n_frames, eta))

    try:
        if detector.trt and not hasattr(detector, "_trt_ctx"):
            detector._trt_init()
        pstats = {"n_frames": 0, "n_detections": 0, "n_tracks": 0, "tracks": []}
        fisheye = cfg.vr == "sbs" and cfg.vr_lens == "fisheye"
        if getattr(restorer, "make_ltx", None) is not None:
            from kukkiri import staged

            if progress:
                progress(ProgressEvent(0.0, "detect", 0, n_frames))
            plan_frames = SharedFrameStore(src, scale_to=scale_to, stride=stride, budget_bytes=0, min_window=64)
            events = staged.plan(lambda rec: _produce_jobs(detector, plan_frames, rec, gap, min_length, pstats,
                                                           cfg.cut_tv, cfg.vr, fisheye), plan_frames, len(frames))
            if "error" in pstats:
                raise pstats["error"]
            detector = plan_frames = None
            staged.release_gpu()
            ltx = restorer.make_ltx()
            ltx.warm(CLIP)

            def ltx_progress(k, n):
                if progress:
                    progress(ProgressEvent(100 * k / n, "ltx", k, n))

            results = staged.ltx_pass(events, SharedFrameStore(src, scale_to=scale_to, stride=stride, budget_bytes=0,
                                                               min_window=CLIP + 64), ltx,
                                      restorer.ltx_min_size, work / "ltx", ltx_progress)
            ltx = None
            staged.release_gpu()
            restorer.ltx = staged.ReplayLtx(results)
            restorer.warm(CANVAS_SIZE)
            source = JobSource(len(frames), maxsize=256)

            def replay():
                try:
                    staged.replay(events, frames, source)
                except BaseException as e:
                    pstats["error"] = e
                    source.close(len(frames))

            producer = threading.Thread(target=replay, daemon=True)
        else:
            restorer.warm(CANVAS_SIZE)
            if getattr(restorer, "ltx", None) is not None:
                restorer.ltx.warm(CLIP)
            source = JobSource(len(frames), maxsize=256)
            producer = threading.Thread(
                target=_produce_jobs, args=(detector, frames, source, gap, min_length, pstats, cfg.cut_tv, cfg.vr,
                                            fisheye), daemon=True
            )
        producer.start()
        touched = restore_stream(source, frames, restorer, sink)
        producer.join()
        if "error" in pstats:
            raise pstats["error"]
        if n_frames and pstats["n_frames"] < n_frames - max(2, round(fps)):
            raise SourceTruncated(pstats["n_frames"], n_frames, fps)
        writer.close()
    except BaseException:
        logger.error(
            f"remaster failed after {done} frames written ({getattr(frames, 'decoded', '?')} decoded, "
            f"{n_frames} expected)"
        )
        writer.abort()
        mux.kill()
        raise
    remaster_s = time.time() - t2
    n_frames = pstats["n_frames"]
    tracks = pstats["tracks"]
    if repeats:
        off = list(accumulate(repeats, initial=0))
        tracks = [(off[min(s, len(repeats))], off[min(e, len(repeats))], a) for s, e, a in tracks]
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
        tracks=tracks,
    )
