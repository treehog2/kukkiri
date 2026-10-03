
from __future__ import annotations

import subprocess
from pathlib import Path

SEGMENT_S = 4
HLS_TIME = SEGMENT_S - 0.1


class LiveMuxer:
    def __init__(
        self,
        src: Path,
        dst: Path,
        fps: float,
        duration_s: float,
        live_dir: Path | None = None,
        ffmpeg: str = "ffmpeg",
    ):
        self.src, self.dst, self.live_dir = Path(src), Path(dst), Path(live_dir) if live_dir else None
        self._seen: set[Path] = set()
        self._playlist_dirty = False
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-hide_banner"]
        cmd += ["-framerate", f"{fps:.6f}", "-f", "h264", "-i", "pipe:0", "-i", str(self.src)]
        cmd += ["-max_interleave_delta", "0"]
        cut = ["-t", f"{duration_s:.3f}"]
        if self.live_dir:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            cmd += [
                "-map", "0:v:0", "-map", "1:a?", "-c", "copy", *cut,
                "-f", "hls", "-hls_time", str(HLS_TIME), "-hls_playlist_type", "event", "-hls_list_size", "0",
                "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
                "-hls_segment_filename", str(self.live_dir / "seg_%05d.m4s"),
                "-hls_flags", "temp_file+independent_segments", str(self.playlist),
            ]
        cmd += ["-map", "0:v:0", "-map", "1:a?", "-c", "copy", *cut, "-movflags", "+faststart", str(self.dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stdin = self.proc.stdin

    @staticmethod
    def idr_period(fps: float) -> int:
        return max(1, round(SEGMENT_S * fps))

    @property
    def playlist(self) -> Path:
        assert self.live_dir is not None
        return self.live_dir / "index.m3u8"

    def new_files(self) -> list[Path]:
        if not self.live_dir:
            return []
        out: list[Path] = []
        init = self.live_dir / "init.mp4"
        segs = sorted(p for p in self.live_dir.glob("seg_*.m4s") if p not in self._seen)
        if init not in self._seen:
            if not (segs and init.exists() and init.stat().st_size > 0):
                return []
            self._seen.add(init)
            out.append(init)
        for p in segs:
            self._seen.add(p)
            out.append(p)
        if (segs or self._playlist_dirty) and self.playlist.exists():
            out.append(self._playlist_snapshot())
            self._playlist_dirty = False
        return out

    def _playlist_snapshot(self) -> Path:
        lines = self.playlist.read_text().splitlines()
        kept: list[str] = []
        for line in lines:
            if line.startswith("seg_") and self.live_dir / line not in self._seen:
                if kept and kept[-1].startswith("#EXTINF:"):
                    kept.pop()
                continue
            kept.append(line)
        snap = self.live_dir / "upload" / self.playlist.name
        snap.parent.mkdir(exist_ok=True)
        snap.write_text("\n".join(kept) + "\n")
        return snap

    def close(self) -> None:
        try:
            _, err = self.proc.communicate()
        except BrokenPipeError:
            _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed ({self.proc.returncode}): {err.decode(errors='replace')[-2000:]}")
        self._playlist_dirty = True

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
