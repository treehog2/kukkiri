import json
import subprocess
from pathlib import Path

from kukkiri.video_io_gpu import gpu_readable_source, repair_source


def _format(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["format"]["format_name"]


def _clip(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=2", "-c:v",
         "libx264", str(path)],
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


def test_h264_mkv_is_remuxed_to_mp4(tmp_path):
    out = gpu_readable_source(_clip(tmp_path / "in.mkv"), tmp_path)
    assert out.name == "source_readable.mp4" and "mp4" in _format(out).split(",")


def test_mkv_remux_keeps_every_audio_track(tmp_path):
    assert _audio_codecs(gpu_readable_source(_two_audio_mkv(tmp_path / "two_audio.mkv"), tmp_path)) == ["aac", "aac"]


def test_repaired_copy_is_mp4_with_every_audio_track(tmp_path):
    out = repair_source(_two_audio_mkv(tmp_path / "two_audio.mkv"), tmp_path)
    assert "mp4" in _format(out).split(",") and _audio_codecs(out) == ["aac", "aac"]
