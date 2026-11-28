import shutil
import subprocess
import time

import pytest

from kukkiri.mux import LiveMuxer, partial_path
from kukkiri.watch import FragmentIndex, playlist_path

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_the_live_playlist_follows_the_growing_restore_and_plays(tmp_path):
    src, raw = tmp_path / "src.mp4", tmp_path / "v.h264"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=30",
                    "-f", "lavfi", "-i", "sine=duration=30", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", "-i", str(src), "-an", "-c:v", "libx264", "-bf", "0",
                    "-f", "h264", str(raw)], check=True)
    data = raw.read_bytes()
    m = LiveMuxer(src, tmp_path / "o.mp4", 25.0, 30.0)
    part = partial_path(tmp_path / "o.mp4")
    index, playlist = FragmentIndex(part), playlist_path(part)
    m.stdin.write(data[: len(data) // 3])
    m.stdin.flush()
    time.sleep(3)
    assert index.write(playlist)
    first = playlist.read_text().count("#EXTINF")
    m.stdin.write(data[len(data) // 3 : 2 * len(data) // 3])
    m.stdin.flush()
    time.sleep(3)
    assert index.write(playlist, finished=True)
    text = playlist.read_text()
    assert text.count("#EXTINF") > first > 0 and text.rstrip().endswith("#EXT-X-ENDLIST")
    got = subprocess.run(["ffprobe", "-v", "error", "-allowed_extensions", "ALL", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(playlist)], capture_output=True, text=True)
    assert 10 < float(got.stdout.strip()) < 21, got.stderr[-300:]
    m.stdin.write(data[2 * len(data) // 3 :])
    m.close()


def test_a_restore_written_again_from_the_start_is_read_again(tmp_path):
    part = tmp_path / "x.mp4.part"
    part.write_bytes(b"\0\0\0\x10ftypisom\0\0\0\0" + b"\0" * 64)
    index = FragmentIndex(part)
    index.update()
    part.write_bytes(b"")
    index.update()
    assert index.pos == 0 and index.frags == []


def test_playlist_entries_survive_awkward_file_names(tmp_path):
    from kukkiri.watch import FragmentIndex

    index = FragmentIndex(tmp_path / "#clip 복원 café.mp4.part")
    index.init_end, index.track = 100, (1, 1000)
    index.frags = [[100, 50, 0, 1], [150, 50, 2000, 1], [200, 50, 4000, 1]]
    text = index.playlist(finished=True)
    uris = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert len(uris) == 3 and all(u.startswith("%23clip%20") for u in uris)
    out = tmp_path / "p.m3u8"
    index.update = lambda: None
    assert index.write(out, finished=True)
    assert "%EB%B3%B5%EC%9B%90" in out.read_bytes().decode("utf-8")


def test_a_rewritten_restore_with_other_first_bytes_is_read_again(tmp_path):
    from kukkiri.watch import FragmentIndex

    part = tmp_path / "x.mp4.part"
    part.write_bytes(b"\0\0\0\x10ftypisom\0\0\0\0" + b"\1" * 64)
    index = FragmentIndex(part)
    index.update()
    index.frags = [[16, 10, 0, 1]]
    part.write_bytes(b"\0\0\0\x10ftypmp42\0\0\0\0" + b"\2" * 200)
    index.update()
    assert index.frags == []

