
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kukkiri.errors import AlreadyRemasteredError, NotRemasterableError
from kukkiri.video_io import probe

GATE_MIN_MOSAIC = 0.01
GATE_MAX_GENITAL_RATIO = 0.05
GATE_GENITAL_SEEN = 0.005
GATE_THRESHOLD = 0.5
GATE_MAX_WIDTH = 1280
GATE_BATCH = 32
GATE_MAX_SAMPLES = 1000
GATE_SLICES = 8
GATE_QUEUE = 64
GATE_MIN_KEYFRAMES = 50
LABEL_MOSAIC, LABEL_GENITAL = (
    0,
    1,
)


@dataclass
class RestorableScan:
    keyframes: int
    mosaic_frac: float
    genital_frac: float
    seconds: float

    @property
    def restorable(self) -> bool:
        if self.keyframes < GATE_MIN_KEYFRAMES:
            return True
        if self.mosaic_frac < GATE_MIN_MOSAIC:
            return False
        return self.genital_frac < GATE_MAX_GENITAL_RATIO * self.mosaic_frac

    @property
    def _nothing_here(self) -> bool:
        return self.mosaic_frac < GATE_MIN_MOSAIC and self.genital_frac < GATE_GENITAL_SEEN

    @property
    def reason(self) -> str:
        if self.restorable:
            return ""
        what = "nothing to remaster" if self._nothing_here else "already remastered"
        return (
            f"{what}: mosaic on {self.mosaic_frac:.1%} of {self.keyframes} keyframes,"
            f" genital on {self.genital_frac:.1%}"
        )

    @property
    def error(self) -> type[Exception]:
        return NotRemasterableError if self._nothing_here else AlreadyRemasteredError

    def to_dict(self) -> dict:
        return asdict(self)


def _reader(video: Path, ffmpeg: str, w: int, h: int, start: float, length: float | None) -> subprocess.Popen:
    cmd = [ffmpeg, "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-skip_frame", "nokey", "-i", str(video), "-vf", f"scale={w}:{h}", "-vsync", "0"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def _keyframes(video: Path, ffmpeg: str, slices: int = GATE_SLICES):
    info = probe(video)
    w, h = info["width"], info["height"]
    if w > GATE_MAX_WIDTH:
        h = int(round(h * GATE_MAX_WIDTH / w / 2)) * 2
        w = GATE_MAX_WIDTH
    duration = info["duration"]
    n_slices = max(1, min(slices, os.cpu_count() or 1))
    if duration <= 0:
        n_slices = 1
    span = duration / n_slices
    procs = [_reader(video, ffmpeg, w, h, i * span, None if n_slices == 1 else span + 0.5) for i in range(n_slices)]
    q: queue.Queue = queue.Queue(maxsize=GATE_QUEUE * n_slices)
    nbytes = w * h * 3

    def pump(proc: subprocess.Popen) -> None:
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                q.put(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        finally:
            q.put(None)

    threads = [threading.Thread(target=pump, args=(p,), daemon=True) for p in procs]
    for t in threads:
        t.start()
    try:
        done = 0
        while done < n_slices:
            frame = q.get()
            if frame is None:
                done += 1
                continue
            yield frame
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.stdout.close()
            p.wait()


def scan_keyframes(video: Path, detector_weights: Path, ffmpeg: str = "ffmpeg", device: str = "cuda",
                   size: str = "seg_medium") -> RestorableScan:
    from kukkiri.detector import CensorDetector, RFDETRSize

    t0 = time.time()
    det = CensorDetector(detector_weights, size=RFDETRSize(size), threshold=GATE_THRESHOLD, device=device)
    kept: list[tuple[bool, bool]] = []
    stride = 1
    seen = 0
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal stride
        for dets in det.detect_frames_fast(batch, batch_size=GATE_BATCH):
            labels = {d.label for d in dets}
            kept.append((LABEL_MOSAIC in labels, LABEL_GENITAL in labels))
        batch.clear()
        if len(kept) >= 2 * GATE_MAX_SAMPLES:
            kept[:] = kept[::2]
            stride *= 2

    for frame in _keyframes(video, ffmpeg):
        if seen % stride == 0:
            batch.append(frame)
            if len(batch) == GATE_BATCH:
                flush()
        seen += 1
    if batch:
        flush()
    a = np.array(kept, dtype=bool) if kept else np.zeros((0, 2), dtype=bool)
    mosaic = float(a[:, 0].mean()) if len(a) else 0.0
    genital = float(a[:, 1].mean()) if len(a) else 0.0
    return RestorableScan(seen, mosaic, genital, round(time.time() - t0, 1))
