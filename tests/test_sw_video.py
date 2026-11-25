import shutil
import subprocess

import numpy as np
import pytest
import torch

from kukkiri.video_io import iter_frames, probe
from kukkiri.video_io_sw import FfmpegFrameStore, FfmpegVideoWriter

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30000/1001:duration=3",
                    "-pix_fmt", "yuv420p", "-crf", "4", "-colorspace", "bt709", str(p)], check=True)
    return p


@pytest.mark.parametrize("stride", [1, 3])
def test_store_reads_every_frame_in_order(clip, stride):
    s = FfmpegFrameStore(clip, batch=8, stride=stride)
    n = sum(len(b) for b in s.iter_batches())
    total = probe(clip)["nb_frames"]
    assert n == len(s) == -(-total // stride) and s.hw == (240, 320)
    want = list(iter_frames(clip))[::stride]
    for i in (0, n // 2, n - 1):
        assert np.abs(s[i].numpy().astype(int) - want[i]).mean() < 12
    s.release(0)
    with pytest.raises(KeyError):
        s[0]
    with pytest.raises(IndexError):
        s[n]


def test_read_and_write_keep_the_stored_colours(clip, tmp_path):
    s = FfmpegFrameStore(clip)
    raw = tmp_path / "out.h264"
    with open(raw, "wb") as f:
        w = FfmpegVideoWriter(320, 240, 30000 / 1001, 4, f)
        for b in s.iter_batches():
            for t in b:
                w.write(t)
        w.close()
    def planes(p):
        o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], capture_output=True).stdout
        return np.frombuffer(o, np.uint8).reshape(-1, 360, 320).astype(int)
    a, b = planes(clip), planes(raw)
    assert len(a) == len(b) == w.n
    for part in (slice(0, 240), slice(240, 300), slice(300, 360)):
        assert abs((b - a)[:, part].mean()) < 0.5


def test_a_failing_encoder_raises_instead_of_hanging(tmp_path):
    with open(tmp_path / "x", "wb") as f:
        w = FfmpegVideoWriter(321, 241, 30.0, 14, f, ffmpeg="ffmpeg")
        with pytest.raises(Exception):
            for _ in range(200):
                w.write(torch.zeros(241, 321, 3, dtype=torch.uint8))
            w.close()


def test_frame_duration_is_only_stamped_on_a_mac(monkeypatch):
    import sys

    from kukkiri.mux import exact_frame_times

    for platform in ("linux", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        assert exact_frame_times(30000 / 1001) == ("30000/1001", ["-bsf:v", "setts=ts=N*1001/(30000*TB)"])
    monkeypatch.setattr(sys, "platform", "darwin")
    assert exact_frame_times(25.0)[1][1] == "setts=ts=N*1/(25*TB):duration=1/(25*TB)"
