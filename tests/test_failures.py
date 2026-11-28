import os
import shutil
import subprocess
import sys
import time

import pytest

from kukkiri.add_mosaic import add_mosaic
from kukkiri.errors import FFmpegError, SourceTruncated
from kukkiri.video_io import probe, upright
from kukkiri.video_io_sw import FfmpegFrameStore

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


class Nothing:
    def detect_frames(self, frames, batch_size=0):
        return [[] for _ in frames]


def make(path, *video, audio=("-c:a", "aac"), size="320x240", seconds=2):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=duration={seconds}", "-pix_fmt", "yuv420p", *video, *audio, str(path)], check=True)
    return path


def seconds(path, stream="v"):
    return float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True).stdout.strip())


def test_a_turned_video_comes_out_upright(tmp_path):
    src = make(tmp_path / "a.mp4")
    turned = tmp_path / "turned.mp4"
    new = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-display_rotation", "90", "-i", str(src), "-c", "copy", str(turned)]).returncode
    if new:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", str(src), "-c", "copy", "-metadata:s:v", "rotate=90", str(turned)], check=True)
    assert upright(turned)[:2] == (240, 320)
    assert add_mosaic(turned, tmp_path / "o.mp4", Nothing()) == (0, 50)
    info = probe(tmp_path / "o.mp4")
    assert (info["width"], info["height"]) == (240, 320)


def test_odd_sizes_and_pcm_audio(tmp_path):
    src = make(tmp_path / "odd.mkv", "-vf", "format=yuv444p,crop=319:239", "-pix_fmt", "yuv444p", "-c:v", "libx264", audio=("-c:a", "pcm_s16le"))
    assert add_mosaic(src, tmp_path / "o.mp4", Nothing()) == (0, 50)
    info = probe(tmp_path / "o.mp4")
    assert (info["width"], info["height"], info["nb_frames"]) == (320, 240, 50)
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                            str(tmp_path / "o.mp4")], capture_output=True, text=True).stdout.strip()
    assert codec == "aac"


def test_shorter_audio_does_not_cut_the_video(tmp_path):
    a = make(tmp_path / "a.mp4")
    src = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", str(a), "-f", "lavfi", "-i", "sine=duration=1", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", str(src)], check=True)
    add_mosaic(src, tmp_path / "o.mp4", Nothing())
    assert probe(tmp_path / "o.mp4")["nb_frames"] == 50


def test_variable_frame_rate_keeps_its_length(tmp_path):
    src = make(tmp_path / "vfr.mp4", "-vf", "select='not(between(mod(n,20),5,9))'", "-vsync", "0", seconds=4)
    n = add_mosaic(src, tmp_path / "o.mp4", Nothing())[1]
    assert 0 <= seconds(tmp_path / "o.mp4") - seconds(src) < 3 / 18 and n == probe(tmp_path / "o.mp4")["nb_frames"]


def test_an_encoder_that_cannot_write_says_why(tmp_path):
    src = make(tmp_path / "a.mp4")
    with pytest.raises(FFmpegError, match="could not write"):
        add_mosaic(src, tmp_path / "no" / "such" / "o.mp4", Nothing())


def test_unreadable_and_cut_files_raise(tmp_path):
    src = make(tmp_path / "a.mp4", "-g", "10", seconds=6)
    d = bytearray(src.read_bytes())
    i = d.find(b"mdat")
    d[i + 2000:i + 2000 + (len(d) - i) // 2] = bytes((len(d) - i) // 2)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(d)
    got = 0
    try:
        got = sum(len(b) for b in FfmpegFrameStore(bad).iter_batches())
        assert got == probe(bad)["nb_frames"]
    except (SourceTruncated, FFmpegError):
        pass
    s = FfmpegFrameStore(src)
    s.ffmpeg = "false"
    with pytest.raises(FFmpegError):
        list(s.iter_batches())


def run(env, *args):
    return subprocess.run([sys.executable, "-m", "kukkiri", *args], capture_output=True, text=True, env={**os.environ, **env})


def test_messages_instead_of_tracebacks(tmp_path):
    r = run({"KUKKIRI_DEVICE": "cuda:0"}, "check")
    assert r.returncode == 1 and "KUKKIRI_DEVICE" in r.stderr and "Traceback" not in r.stderr
    assert run({"KUKKIRI_DEVICE": "cuda:0"}, "--help").returncode == 0


def test_a_missing_ffmpeg_is_named(monkeypatch, capsys):
    from kukkiri import cli

    def gone(args):
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")

    monkeypatch.setattr(cli, "_restore", gone)
    with pytest.raises(SystemExit) as e:
        cli.main(["restore", "x.mp4"])
    assert e.value.code == 1 and "ffprobe was not found" in capsys.readouterr().err


def test_the_muxer_says_why_it_ended(tmp_path):
    from kukkiri.mux import LiveMuxer

    m = LiveMuxer(tmp_path / "nothing.mp4", tmp_path / "o.mp4", 25.0, 1.0)
    assert m.failure() is None
    m.stdin.close()
    for _ in range(200):
        if m.failure():
            break
        time.sleep(0.1)
    assert "mux failed" in m.failure()
