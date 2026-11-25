import shutil
import subprocess

import numpy as np
import pytest

from kukkiri.add_mosaic import add_mosaic
from kukkiri.detector import Detection
from kukkiri.video_io import iter_frames, probe


class OneBox:
    def detect_frames(self, frames, batch_size=0):
        return [[Detection(box=(40, 40, 90, 80), score=0.9, label=1)] for _ in frames]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("posix", [True, False])
def test_a_video_is_mosaicked_with_its_length_and_audio(tmp_path, monkeypatch, posix):
    if not posix:
        import os
        import sys

        monkeypatch.setitem(sys.modules, "fcntl", None)
        monkeypatch.delattr(os, "readv", raising=False)
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    assert add_mosaic(src, dst, OneBox()) == (50, 50)
    assert probe(dst)["nb_frames"] == 50 and abs(probe(dst)["duration"] - 2) < 0.1
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of",
                        "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip()
    assert a == "aac"
    before, after = next(iter_frames(src)), next(iter_frames(dst))
    assert abs(before[48:72, 48:80].astype(int) - after[48:72, 48:80]).mean() > abs(before[90:, 100:].astype(int) - after[90:, 100:]).mean() + 3


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("matrix", ["bt709", "smpte170m"])
def test_untouched_colours_are_kept(tmp_path, matrix):
    class Nothing:
        def detect_frames(self, frames, batch_size=0):
            return [[] for _ in frames]

    def planes(p):
        o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], capture_output=True).stdout
        return np.frombuffer(o, np.uint8).reshape(-1, 360, 320).astype(int)

    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=25", "-frames:v", "20",
                    "-vf", "eq=saturation=0.4:contrast=0.7", "-pix_fmt", "yuv420p", "-crf", "4", "-colorspace", matrix, str(src)], check=True)
    add_mosaic(src, dst, Nothing())
    a, b = planes(src), planes(dst)
    for part in (slice(0, 240), slice(240, 300), slice(300, 360)):
        assert abs((b - a)[:, part].mean()) < 0.5
