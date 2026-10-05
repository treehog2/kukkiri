import os
import queue
import subprocess
import sys

import torch

from kukkiri.detector import Detection
from kukkiri.mux import exact_frame_times
from kukkiri.remaster import ENCODE_MAX_SIDE, _detect_loop, _output_size
from kukkiri.streaming import JobSource, SpanEmitter
from kukkiri.tracker import FACE_LABEL, GappedTracker


class _Detector:
    def __init__(self, label: int):
        self.label = label

    def detect_frames(self, batch, batch_size=32):
        return [[Detection((10, 10, 30, 30), 0.9, self.label)] for _ in batch]


def _run_detect(label: int, n: int = 400):
    bq: queue.Queue = queue.Queue()
    for i in range(0, n, 32):
        bq.put([torch.zeros(36, 64, 3, dtype=torch.uint8)] * min(32, n - i))
    bq.put(None)
    source = JobSource(n)
    stats = {"n_frames": 0, "n_detections": 0, "n_tracks": 0, "tracks": []}
    seen = []

    def advance(f):
        seen.append(f)
        JobSource.advance(source, f)

    source.advance = advance
    _detect_loop(_Detector(label), bq, GappedTracker(), SpanEmitter(min_length=2), source, stats, cut_tv=0.3)
    return source, stats, seen


def test_face_track_does_not_pin_the_frontier():
    source, stats, seen = _run_detect(FACE_LABEL)
    assert source.n_jobs == 0 and stats["n_face_tracks"] == 1
    assert seen[-1] > 300


def test_mosaic_track_still_emits_jobs():
    source, stats, _ = _run_detect(0)
    assert source.n_jobs > 0 and stats["n_tracks"] == 1


def test_output_size_caps_each_side():
    assert _output_size(1080, 1920, None) is None
    assert _output_size(4320, 7680, None) == (2304, 4096)
    assert max(_output_size(4096, 8192, None)) <= ENCODE_MAX_SIDE
    assert _output_size(2160, 3840, 1920 * 1088) == (1080, 1920)


def test_exact_frame_times_mux(tmp_path):
    rate, stamp = exact_frame_times(30000 / 1001)
    assert rate == "30000/1001"
    raw = tmp_path / "v.h264"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30000/1001", "-frames:v", "300",
         "-c:v", "libx264", "-bf", "0", "-f", "h264", str(raw)],
        check=True,
    )
    out = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-r", rate, "-i", str(raw), "-c", "copy", *stamp, str(out)],
                   check=True)
    pts = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert len(pts) == 300 and abs(float(pts[-1]) - 299 * 1001 / 30000) < 1e-4


def test_stalled_stream_fails(tmp_path):
    code = (
        "import numpy as np\n"
        "from kukkiri.streaming import JobSource, StreamStalled, restore_stream\n"
        "try:\n"
        "    restore_stream(JobSource(10), [np.zeros((8, 8, 3), np.uint8)] * 10, object(), lambda fi, f: None)\n"
        "except StreamStalled:\n"
        "    print('stalled')\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "KUKKIRI_STALL_S": "1"}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=60)
    assert run.stdout.strip() == "stalled", run.stderr[-2000:]
