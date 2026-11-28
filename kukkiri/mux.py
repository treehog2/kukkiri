
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import threading
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


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"could not remove {path.name}: {e}")


class _PipeReader(io.RawIOBase):

    def __init__(self, fd: int):
        self._f = os.fdopen(fd, "rb", buffering=0)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, b) -> int:
        return self._f.readinto(b)

    def close(self) -> None:
        self._f.close()
        super().close()


class GrowingMp4:

    def __init__(self, src: Path, out: Path, fps: float, duration_s: float, codec: str = "h264"):
        rfd, wfd = os.pipe()
        self.stdin = os.fdopen(wfd, "wb")
        self.frames = 0
        self.error: BaseException | None = None
        self._stopping = False
        self._read = _PipeReader(rfd)
        self._thread = threading.Thread(target=self._run, args=(Path(src), Path(out), fps, duration_s, codec), daemon=True)
        self._thread.start()

    def _run(self, src: Path, out_path: Path, fps: float, duration_s: float, codec: str) -> None:
        try:
            import av

            r = Fraction(fps).limit_denominator(1001)
            tb = 1 / r
            vin = av.open(self._read, format=codec, options={"framerate": f"{r.numerator}/{r.denominator}"})
            out = av.open(str(out_path), "w", format="mp4",
                          options={"movflags": "delay_moov+default_base_moof", "frag_duration": "2000000"})
            vs = out.add_stream_from_template(vin.streams.video[0])
            vs.time_base = tb
            if codec == "hevc":
                vs.codec_tag = "hvc1"
            ain = av.open(str(src))
            audio = {s.index: out.add_stream_from_template(s) for s in ain.streams.audio}
            start = (ain.start_time or 0) / 1e6
            packets = ain.demux(*ain.streams.audio) if audio else iter(())
            pending = next(packets, None)

            def audio_until(t: float) -> None:
                nonlocal pending
                while pending is not None:
                    if pending.pts is None or pending.size == 0:
                        pending = next(packets, None)
                        continue
                    at = float(pending.pts * pending.time_base) - start
                    if at >= min(t, duration_s):
                        return
                    if at < 0:
                        pending = next(packets, None)
                        continue
                    shift = round(start / pending.time_base)
                    pending.pts -= shift
                    if pending.dts is not None:
                        pending.dts -= shift
                    pending.stream = audio[pending.stream.index]
                    out.mux(pending)
                    pending = next(packets, None)

            for pkt in vin.demux(vin.streams.video[0]):
                if pkt.size == 0:
                    continue
                pkt.stream, pkt.time_base = vs, tb
                pkt.pts = pkt.dts = self.frames
                pkt.duration = 1
                out.mux(pkt)
                self.frames += 1
                audio_until(float(self.frames * tb))
            if not self._stopping:
                audio_until(duration_s)
            out.close()
            vin.close()
            ain.close()
        except BaseException as e:
            self.error = e
        finally:
            self._read.close()

    def close(self, stopping: bool = False) -> None:
        self._stopping = stopping
        try:
            self.stdin.close()
        except OSError:
            pass
        self._thread.join()


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
        codec: str = "h264",
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
            stamp = ["-bsf:v", f"{stamp[1]},{codec}_metadata=sample_aspect_ratio={n}/{d}", "-aspect", f"{w * n}:{h * d}"]
        self.cut = cut = ["-t", f"{duration_s:.3f}"]
        self.video_only = None if self.live_dir else partial_path(self.dst)
        if self.video_only:
            self.final = ["-bsf:v", f"{codec}_metadata=sample_aspect_ratio={sar[0]}/{sar[1]}", "-aspect",
                          f"{size[0] * sar[0]}:{size[1] * sar[1]}"] if sar else []
            if codec == "hevc":
                self.final += ["-tag:v", "hvc1"]
            self.growing = GrowingMp4(self.src, self.video_only, fps, duration_s, codec)
            self.stdin = self.growing.stdin
            return
        cmd += ["-framerate", rate, "-f", codec, "-i", "pipe:0", *cut, "-i", str(self.src)]
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
        if self.video_only:
            self._finish()
            return
        try:
            _, err = self.proc.communicate()
        except BrokenPipeError:
            _, err = self.proc.communicate()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed ({self.proc.returncode}): {err.decode(errors='replace')[-2000:]}")
        self._playlist_dirty = True

    def _finish(self) -> None:
        self.growing.close()
        if self.growing.error:
            _unlink(self.video_only)
            raise RuntimeError(f"mux failed: {self.growing.error!r}") from self.growing.error
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-hide_banner", "-nostdin"]
        rest = ["-i", str(self.video_only), *self.cut, "-i", str(self.src), "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
                *self.final, "-movflags", "+faststart", str(self.dst)]
        rotate = ["-display_rotation", f"{self.rotation:g}"] if self.rotation else []
        r = subprocess.run(cmd + rotate + rest, capture_output=True)
        if r.returncode != 0 and b"display_rotation" in r.stderr:
            logger.warning(f"could not keep the video's {self.rotation:g} degree rotation: {r.stderr[-300:]!r}")
            r = subprocess.run(cmd + rest, capture_output=True)
        if r.returncode != 0:
            kept = self.dst.with_name(self.dst.stem + ".unfinished.mp4")
            try:
                self.video_only.replace(kept)
            except OSError:
                kept = self.video_only
            raise RuntimeError(f"ffmpeg mux failed ({r.returncode}); the restored video is kept as {kept}: "
                               f"{r.stderr.decode(errors='replace')[-2000:]}")
        if os.environ.get("KUKKIRI_KEEP_PART") != "1":
            _unlink(self.video_only)

    def failure(self) -> str | None:
        if self.video_only:
            return f"mux failed: {self.growing.error!r}" if self.growing.error else None
        if self.proc.poll() in (None, 0):
            return None
        return f"ffmpeg mux failed ({self.proc.returncode}): {self.proc.stderr.read().decode(errors='replace')[-2000:]}"

    def kill(self) -> None:
        if self.video_only:
            self.growing.close(stopping=True)
            _unlink(self.video_only)
            return
        if self.proc.poll() is None:
            self.proc.kill()
