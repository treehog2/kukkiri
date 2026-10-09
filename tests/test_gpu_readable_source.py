import json
import subprocess
from pathlib import Path

import pytest

from kukkiri import video_io_gpu
from kukkiri.video_io_gpu import gpu_readable_source, repair_source


def _format(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["format"]["format_name"]


def _clip(path: Path, pix_fmt: str = "yuv420p", codec: str = "libx264") -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=2", "-c:v",
         codec, "-pix_fmt", pix_fmt, str(path)],
        check=True,
    )
    return path


def _audio_codecs(path: Path) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.split()


def _two_audio_mkv(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=2", "-f", "lavfi",
         "-i", "sine=duration=2", "-f", "lavfi", "-i", "sine=f=880:duration=2", "-map", "0", "-map", "1", "-map", "2",
         "-c:v", "libx264", "-c:a:0", "aac", "-c:a:1", "flac", str(path)],
        check=True,
    )
    return path


def test_mp4_is_read_as_is(tmp_path):
    src = _clip(tmp_path / "in.mp4")
    assert gpu_readable_source(src, tmp_path) == src


@pytest.mark.parametrize("pix_fmt", ["yuv420p10le", "yuv422p", "yuv444p"])
def test_h264_beyond_8bit_420_is_reencoded(tmp_path, pix_fmt):
    out = gpu_readable_source(_clip(tmp_path / "in.mp4", pix_fmt), tmp_path)
    pix = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "csv=p=0",
         str(out)],
        capture_output=True, text=True, check=True,
    )
    assert out.name == "source_readable.mp4" and pix.stdout.strip() == "yuv420p"


def test_full_range_420_is_read_as_is(tmp_path):
    src = _clip(tmp_path / "in.mp4", "yuvj420p")
    assert gpu_readable_source(src, tmp_path) == src


def test_10bit_hevc_is_read_as_is(tmp_path):
    src = _clip(tmp_path / "in.mp4", "yuv420p10le", "libx265")
    assert gpu_readable_source(src, tmp_path) == src


def test_h264_mkv_is_remuxed_to_mp4(tmp_path):
    out = gpu_readable_source(_clip(tmp_path / "in.mkv"), tmp_path)
    assert out.name == "source_readable.mp4" and "mp4" in _format(out).split(",")


def test_mkv_remux_keeps_every_audio_track(tmp_path):
    assert _audio_codecs(gpu_readable_source(_two_audio_mkv(tmp_path / "two_audio.mkv"), tmp_path)) == ["aac", "aac"]


def test_repaired_copy_is_mp4_with_every_audio_track(tmp_path):
    out = repair_source(_two_audio_mkv(tmp_path / "two_audio.mkv"), tmp_path)
    assert "mp4" in _format(out).split(",") and _audio_codecs(out) == ["aac", "aac"]


def _probe_result(stdout: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ('{"streams": [{"nb_read_packets": "120", "nb_read_frames": "60"}]}', True),
        ('{"streams": [{"nb_read_packets": "60", "nb_read_frames": "60"}]}', False),
        ('{"streams": [{"nb_read_packets": "120", "nb_read_frames": "0"}]}', True),
        ('{"streams": [{}]}', False),
        ('{"streams": []}', False),
        ("", False),
    ],
)
def test_field_coded(monkeypatch, stdout, expected):
    monkeypatch.setattr(video_io_gpu.subprocess, "run", lambda *a, **k: _probe_result(stdout))
    assert video_io_gpu._field_coded(Path("in.ts")) is expected


def test_field_coded_probe_timeout_counts_as_frame_coded(monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    monkeypatch.setattr(video_io_gpu.subprocess, "run", hang)
    assert video_io_gpu._field_coded(Path("in.ts")) is False


def test_field_coded_h264_is_reencoded(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io_gpu, "_field_coded", lambda video: True)
    assert gpu_readable_source(_clip(tmp_path / "in.mp4"), tmp_path).name == "source_readable.mp4"


def test_field_coding_is_only_checked_for_h264(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io_gpu, "_field_coded", lambda video: pytest.fail("probed a non-h264 source"))
    src = _clip(tmp_path / "in.mp4", "yuv420p", "libx265")
    assert gpu_readable_source(src, tmp_path) == src
