import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kukkiri.tools import tool

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    tmp = Path(tmp)
    for name in ("ffmpeg", "ffprobe"):
        exe = tool(name)
        assert Path(exe).is_absolute() and Path(exe).parent == Path(sys.executable).parent / "ffmpeg", exe
        subprocess.run([exe, "-version"], check=True, capture_output=True)
    src = tmp / "in.mp4"
    subprocess.run([tool("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=2", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    decoys = tmp / "decoys"
    decoys.mkdir()
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        (decoys / name).write_text("not a program")
    os.environ["PATH"] = str(decoys) + os.pathsep + os.environ["PATH"]
    os.chdir(decoys)

    from kukkiri.video_io import cut_section, probe
    from kukkiri.video_io_gpu import gpu_readable_source

    assert probe(src)["nb_frames"] == 50
    assert gpu_readable_source(src, tmp) == src
    cut = cut_section(src, tmp / "cut.mp4", 0.5, 1.0, ffmpeg=tool("ffmpeg"))
    assert probe(cut)["nb_frames"] > 0
    print("bundled ffmpeg / ffprobe smoke test ok")
