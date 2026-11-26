import shutil
import subprocess
import time

import pytest

from kukkiri.mux import LiveMuxer

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_the_video_reaches_the_disk_while_encoding_and_the_audio_lines_up(tmp_path):
    src, raw = tmp_path / "src.mp4", tmp_path / "v.h264"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=60",
                    "-f", "lavfi", "-i", "sine=duration=60", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", str(src), "-t", "20", "-an", "-c:v", "libx264", "-bf", "0",
                    "-f", "h264", str(raw)], check=True)
    data = raw.read_bytes()
    m = LiveMuxer(src, tmp_path / "o.mp4", 25.0, 20.0)
    m.stdin.write(data[: len(data) // 2])
    m.stdin.flush()
    time.sleep(2)
    written = sum(p.stat().st_size for p in tmp_path.glob("o.mp4*"))
    m.stdin.write(data[len(data) // 2 :])
    m.close()
    assert written > 50_000
    assert sorted(p.name for p in tmp_path.glob("o.mp4*")) == ["o.mp4"]
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0", str(tmp_path / "o.mp4")],
                         capture_output=True, text=True, check=True).stdout.split()
    durations = dict(line.split(",") for line in out)
    assert abs(float(durations["video"]) - 20) < 0.1 and abs(float(durations["audio"]) - 20) < 0.1
    packets = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=stream_index,pts_time", "-of", "csv=p=0",
                              str(tmp_path / "o.mp4")], capture_output=True, text=True, check=True).stdout.split()
    last, worst = {}, 0.0
    for line in packets:
        s, t = line.split(",")[:2]
        last[s] = float(t)
        if len(last) == 2:
            worst = max(worst, abs(last["0"] - last["1"]))
    assert worst < 2


@pytest.mark.parametrize("sar", ["4:3", "1:1"])
def test_the_source_display_shape_survives_the_mux(tmp_path, sar):
    from kukkiri.video_io import probe

    w, h, fps = 320, 180, 30
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r={fps}", "-f", "lavfi", "-i", "sine",
                    "-t", "5", "-vf", f"setsar={sar.replace(':', '/')}", "-c:v", "libx264", "-bf", "0", "-c:a", "aac", str(src)],
                   check=True)
    info = probe(src)
    assert info["sar"] == ((4, 3) if sar == "4:3" else None)
    enc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", "setsar=1", "-c:v", "libx264", "-bf", "0", "-f", "h264",
                          "pipe:1"], capture_output=True, check=True).stdout
    m = LiveMuxer(src, tmp_path / "out.mp4", fps, 5.0, sar=info["sar"], size=(w, h))
    m.stdin.write(enc)
    m.close()
    shape = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=sample_aspect_ratio,display_aspect_ratio", "-of", "csv=p=0", str(tmp_path / "out.mp4")],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert shape == ("4:3,64:27" if sar == "4:3" else "1:1,16:9")


def test_a_phone_video_keeps_its_rotation(tmp_path):
    from kukkiri.video_io import probe

    plain, src = tmp_path / "plain.mp4", tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30", "-f", "lavfi", "-i", "sine",
                    "-t", "3", "-c:v", "libx264", "-bf", "0", "-c:a", "aac", str(plain)], check=True)
    if subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(plain), "-c", "copy", str(src)]).returncode:
        pytest.skip("this ffmpeg cannot tag a rotation")
    info = probe(src)
    assert info["rotation"] == 90
    enc = subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-i", str(src), "-c:v", "libx264", "-bf", "0", "-f", "h264",
                          "pipe:1"], capture_output=True, check=True).stdout
    m = LiveMuxer(src, tmp_path / "out.mp4", 30, 3.0, rotation=info["rotation"])
    m.stdin.write(enc)
    m.close()
    assert probe(tmp_path / "out.mp4")["rotation"] == 90
