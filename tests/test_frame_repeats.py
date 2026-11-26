import subprocess
from pathlib import Path

from kukkiri.video_io import frame_repeats, plausible_fps, probe


def _clip(path: Path, drop_every: int | None) -> Path:
    vf = f"select='not(eq(mod(n\\,{drop_every})\\,{drop_every - 1}))'" if drop_every else "null"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=20", "-vf", vf,
         "-fps_mode", "passthrough", "-c:v", "libx264", "-video_track_timescale", "30", str(path)],
        check=True,
    )
    return path


def _repeats(path: Path) -> list[int] | None:
    info = probe(path)
    return frame_repeats(path, info["fps"], info["nb_frames"], info["duration"])


def test_constant_rate_keeps_plain_stamping(tmp_path):
    assert _repeats(_clip(tmp_path / "cfr.mp4", None)) is None


def test_dropped_frames_are_filled(tmp_path):
    repeats = _repeats(_clip(tmp_path / "vfr.mp4", 30))
    assert repeats is not None and sum(repeats) == 600 - 1 and repeats.count(2) == 19


def test_a_declared_rate_far_above_the_real_one_falls_back_to_the_average():
    assert abs(plausible_fps(60000 / 1001, 184737, 6164.0) - 30000 / 1001) < 1e-9
    assert abs(plausible_fps(479.52, 138942, 4636.0) - 30000 / 1001) < 1e-9
    assert plausible_fps(30.0, 13980, 600.0) == 30.0
    assert plausible_fps(30.0, 18000, 600.0) == 30.0
    assert abs(plausible_fps(1000.0, 1300, 100.0) - 13.0) < 1e-9


def test_audio_running_past_the_picture_keeps_the_declared_rate(tmp_path):
    src = tmp_path / "long_audio.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30:duration=10", "-f", "lavfi",
                    "-i", "sine=duration=40", "-c:v", "libx264", "-c:a", "aac", str(src)], check=True)
    info = probe(src)
    assert info["duration"] > 39 and abs(info["video_duration"] - 10) < 0.1
    assert plausible_fps(info["fps"], info["nb_frames"], info["video_duration"]) == 30.0
