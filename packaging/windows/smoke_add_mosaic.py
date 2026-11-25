import subprocess
import tempfile
from pathlib import Path

from kukkiri.add_mosaic import add_mosaic
from kukkiri.detector import Detection
from kukkiri.video_io import probe


class OneBox:
    def detect_frames(self, frames, batch_size=0):
        return [[Detection(box=(40, 40, 90, 80), score=0.9, label=1)] for _ in frames]


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    assert add_mosaic(src, dst, OneBox()) == (50, 50)
    assert probe(dst)["nb_frames"] == 50
    print("add-mosaic smoke test ok")
