
from __future__ import annotations

import logging
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SEGMENT_S = 4
HLS_TIME = SEGMENT_S - 0.1
logger = logging.getLogger(__name__)


def exact_frame_times(fps: float) -> tuple[str, list[str]]:
    r = Fraction(fps).limit_denominator(1001)
    setts = f"setts=ts=N*{r.denominator}/({r.numerator}*TB)"
    if sys.platform == "darwin":
        setts += f":duration={r.denominator}/({r.numerator}*TB)"
    return f"{r.numerator}/{r.denominator}", ["-bsf:v", setts]


def partial_path(dst: Path) -> Path:
    return Path(dst).with_name(Path(dst).name + ".part")


class LiveMuxer:
    def __init__(
        self,
        src: Path,
        dst: Path,
        fps: float,
        duration_s: float,
        live_dir: Path | None = None,
        ffmpeg: str = "ffmpeg",
        sar: tuple[int, int] | None = None,
        size: tuple[int, int] = (0, 0),
        rotation: float = 0,
    ):
        self.src, self.dst, self.live_dir = Path(src), Path(dst), Path(live_dir) if live_dir else None
        self.ffmpeg, self.rotation = ffmpeg, rotation
        self._seen: set[Path] = set()
        self._playlist_dirty = False
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-hide_banner"]
        rate, stamp = exact_frame_times(fps)
        if sar:
            (n, d), (w, h) = sar, size
            stamp = ["-bsf:v", f"{stamp[1]},h264_metadata=sample_aspect_ratio={n}/{d}", "-aspect", f"{w * n}:{h * d}"]
        self.cut = cut = ["-t", f"{duration_s:.3f}"]
        self.video_only = None if self.live_dir else partial_path(self.dst)
        if self.video_only:
            cmd += ["-framerate", rate, "-f", "h264", "-i", "pipe:0", "-map", "0:v:0", "-c", "copy", *stamp]
            cmd += ["-f", "mp4", str(self.video_only)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self.stdin = self.proc.stdin
            return
        cmd += ["-framerate", rate, "-f", "h264", "-i", "pipe:0", *cut, "-i", str(self.src)]
        cmd += ["-max_interleave_delta", "0"]
        if self.live_dir:
            self.live_dir.mkdir(parents=True, exist_ok=True)
            cmd += [
                "-map", "0:v:0", "-map", "1:a?", "-c", "copy", *stamp,
                "-f", "hls", "-hls_time", str(HLS_TIME), "-hls_playlist_type", "event", "-hls_list_size", "0",
                "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
                "-hls_segment_filename", str(self.live_dir / "seg_%05d.m4s"),
                "-hls_flags", "temp_file+independent_segments", str(self.playlist),
            ]
        cmd += ["-map", "0:v:0", "-map", "1:a?", "-c", "copy", *stamp, "-movflags", "+faststart", str(self.dst)]
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
        if self.video_only:
            cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-hide_banner", "-nostdin"]
            rest = ["-i", str(self.video_only), *self.cut, "-i", str(self.src), "-map", "0:v:0", "-map", "1:a?", "-c", "copy"]
            rest += ["-movflags", "+faststart", str(self.dst)]
            try:
                rotate = ["-display_rotation", f"{self.rotation:g}"] if self.rotation else []
                r = subprocess.run(cmd + rotate + rest, capture_output=True)
                if r.returncode != 0 and b"display_rotation" in r.stderr:
                    logger.warning(f"could not keep the video's {self.rotation:g} degree rotation: {r.stderr[-300:]!r}")
                    r = subprocess.run(cmd + rest, capture_output=True)
            finally:
                self.video_only.unlink(missing_ok=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed ({r.returncode}): {r.stderr.decode(errors='replace')[-2000:]}")
        self._playlist_dirty = True

    def failure(self) -> str | None:
        if self.proc.poll() in (None, 0):
            return None
        return f"ffmpeg mux failed ({self.proc.returncode}): {self.proc.stderr.read().decode(errors='replace')[-2000:]}"

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        if self.video_only:
            self.proc.wait()
            self.video_only.unlink(missing_ok=True)
