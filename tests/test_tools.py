import sys

from kukkiri import tools


def test_bundled_copy_by_full_path_on_windows(monkeypatch, tmp_path):
    (tmp_path / "ffmpeg").mkdir()
    (tmp_path / "ffmpeg" / "ffprobe.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert tools.tool("ffprobe") == str(tmp_path / "ffmpeg" / "ffprobe.exe")
    assert tools.tool("ffmpeg") == "ffmpeg"


def test_name_elsewhere(monkeypatch, tmp_path):
    (tmp_path / "ffmpeg").mkdir()
    (tmp_path / "ffmpeg" / "ffprobe.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools.tool("ffprobe") == "ffprobe"
