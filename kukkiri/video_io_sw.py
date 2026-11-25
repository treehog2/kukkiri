from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kukkiri.errors import FFmpegError, SourceTruncated
from kukkiri.video_io import FROM_BGR, TO_BGR, probe
from kukkiri.video_io_gpu import SharedFrameStore

class FfmpegFrameStore(SharedFrameStore):

    def __init__(self, video: Path, batch: int = 8, scale_to: tuple[int, int] | None = None,
                 budget_bytes: int = 6 << 30, min_window: int = 256, stride: int = 1, ffmpeg: str = "ffmpeg"):
        info = probe(video)
        self.video, self.ffmpeg, self.stride, self.device, self.batch = video, ffmpeg, stride, "cpu", batch
        self._src_hw = self.hw = (info["height"], info["width"])
        self._expected, self._src_fps = info["nb_frames"], info["fps"]
        self.n = -(-(info["nb_frames"] or round(info["fps"] * info["duration"])) // stride)
        self.fps = info["fps"] / stride
        self.scale_to = None if scale_to is None or tuple(scale_to) == self.hw else tuple(scale_to)
        if self.scale_to is not None:
            self.hw = self.scale_to
        self.max_window = max(min_window, int(budget_bytes // (self.hw[0] * self.hw[1] * 3)))
        self._buf: dict = {}
        self._cv = threading.Condition()
        self._head = self._low = self.decoded = 0
        self._done = False

    def _batches(self):
        h, w = self._src_hw
        hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
        cmd = [self.ffmpeg, "-loglevel", "error", *hw, "-noautorotate", "-i", str(self.video), "-map", "0:v:0", "-vsync",
               "0", "-vf", TO_BGR, "-f", "rawvideo", "-"]
        err = tempfile.TemporaryFile()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0)
        got = 0
        try:
            while True:
                buf = np.empty((self.batch, h, w, 3), np.uint8)
                mv, off = memoryview(buf).cast("B"), 0
                while off < buf.nbytes:
                    k = p.stdout.readinto(mv[off:])
                    if not k:
                        break
                    off += k
                n = off // (h * w * 3)
                got += n
                if n:
                    yield buf[:n]
                if n < self.batch:
                    break
            rc = p.wait()
            err.seek(0)
            why = err.read().decode(errors="replace")[-2000:]
            if not got:
                raise FFmpegError(f"no frames could be read from {self.video.name} ({rc}): {why}")
            if (rc or why) and got < self._expected:
                raise SourceTruncated(got, self._expected, self._src_fps)
        finally:
            if p.poll() is None:
                p.kill()
            p.stdout.close()
            p.wait()
            err.close()

    def iter_batches(self):
        src = 0
        try:
            for batch in self._batches():
                keep = [k for k in range(len(batch)) if (src + k) % self.stride == 0]
                src += len(batch)
                if not keep:
                    continue
                with self._cv:
                    while self._head - self._low > self.max_window:
                        self._cv.wait(timeout=0.5)
                out = [torch.from_numpy(batch[k].copy()) for k in keep]
                if self.scale_to is not None:
                    out = [F.interpolate(t[None].permute(0, 3, 1, 2).float(), size=self.scale_to, mode="area").round().clamp(0, 255)
                           .to(torch.uint8)[0].permute(1, 2, 0).contiguous() for t in out]
                with self._cv:
                    for t in out:
                        self._buf[self._head] = t
                        self._head += 1
                    self.decoded = self._head
                    self._cv.notify_all()
                yield out
        finally:
            with self._cv:
                self._done = True
                self.n = self._head
                self._cv.notify_all()


class FfmpegVideoWriter:

    def __init__(self, width: int, height: int, fps: float, qp: int, stream, max_bitrate: int | None = None,
                 idr_period: int | None = None, ffmpeg: str = "ffmpeg"):
        if sys.platform == "darwin":
            enc = ["-c:v", "h264_videotoolbox", "-profile:v", "high", *(["-b:v", str(max_bitrate)] if max_bitrate else ["-q:v", "80"])]
        else:
            cap = ["-maxrate", str(max_bitrate), "-bufsize", str(max_bitrate)] if max_bitrate else []
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(qp), *cap]
        cmd = [ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
               "-i", "-", "-vf", FROM_BGR, *enc,
               "-bf", "0", *(["-g", str(idr_period)] if idr_period else []), "-f", "h264", "-"]
        self._err = tempfile.TemporaryFile()
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stream, stderr=self._err)
        self.n = 0

    def write(self, frame_bgr: torch.Tensor) -> None:
        try:
            self._p.stdin.write(np.ascontiguousarray(frame_bgr.numpy()).data)
        except OSError:
            raise FFmpegError(self._errors()) from None
        self.n += 1

    def close(self) -> None:
        try:
            self._p.stdin.close()
        except OSError:
            pass
        if self._p.wait() != 0:
            raise FFmpegError(self._errors())

    def _errors(self) -> str:
        rc = self._p.wait()
        self._err.seek(0)
        return f"ffmpeg encoder ended ({rc}): " + self._err.read().decode(errors="replace")[-2000:]

    def abort(self) -> None:
        self._p.kill()
        self._p.wait()
