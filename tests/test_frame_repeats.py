import subprocess
from pathlib import Path

from kukkiri.video_io import frame_repeats, probe


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
