from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path

import torch

from kukkiri.errors import FFmpegError

logger = logging.getLogger(__name__)

_NVC_LOCK = (
    threading.Lock()
)


GPU_READABLE_CONTAINERS = {"mp4", "mov", "matroska"}
GPU_READABLE_CODECS = {"h264", "hevc"}
MUXABLE_AUDIO = {"aac", "mp3", "ac3", "eac3"}


def gpu_readable_source(video: Path, out_dir: Path, ffmpeg: str = "ffmpeg") -> Path:
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name:stream=codec_type,codec_name", "-of", "json",
         str(video)],
        capture_output=True, text=True,
    )
    if streams.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {Path(video).name} ({streams.returncode}): {streams.stderr[-500:]}")
    j = json.loads(streams.stdout)
    codecs = {}
    for st in j.get("streams", []):
        codecs.setdefault(st.get("codec_type"), st.get("codec_name"))
    container_ok = bool(GPU_READABLE_CONTAINERS & set(j.get("format", {}).get("format_name", "").split(",")))
    video_ok = codecs.get("video") in GPU_READABLE_CODECS
    audio_ok = codecs.get("audio") is None or codecs.get("audio") in MUXABLE_AUDIO
    if container_ok and video_ok and audio_ok:
        return Path(video)
    out = Path(out_dir) / "source_readable.mp4"
    vcodec = ["-c:v", "copy"] if video_ok else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14"]
    vcodec += [] if video_ok else ["-pix_fmt", "yuv420p", "-vsync", "cfr"]
    acodec = ["-c:a", "copy"] if audio_ok else ["-c:a", "aac", "-b:a", "192k"]
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?", *vcodec, *acodec]
    cmd += ["-movflags", "+faststart", str(out)]
    t = time.time()
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise FFmpegError(f"could not make {Path(video).name} readable ({run.returncode}): {run.stderr[-500:]}")
    logger.info(
        f"{Path(video).name} ({j.get('format', {}).get('format_name')}, {codecs.get('video')}/{codecs.get('audio')})"
        f" -> {'remux' if video_ok else 'h264 re-encode'} in {time.time() - t:.1f} s"
    )
    return out


class _CAI:
    def __init__(self, d: dict):
        self.__cuda_array_interface__ = d


def _frame_to_tensor(fr, device: str) -> torch.Tensor:
    view = fr.cuda()[0]
    t = torch.as_tensor(_CAI(view.__cuda_array_interface__), device=device)
    return t.flip(-1).contiguous()


class GpuFrameStore:

    def __init__(
        self,
        video: Path,
        device: str = "cuda",
        gpu_id: int = 0,
        batch: int = 32,
        scale_to: tuple[int, int] | None = None,
    ):
        import PyNvVideoCodec as nvc
        from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder

        self.dec = SimpleDecoder(str(video), gpu_id, 0, 0, True, 0, 0, 0, 4, nvc.OutputColorType.RGB)
        meta = self.dec.get_stream_metadata()
        self.n = int(meta.num_frames)
        self.hw = (int(meta.height), int(meta.width))
        self.scale_to = None if scale_to is None or tuple(scale_to) == self.hw else tuple(scale_to)
        if self.scale_to is not None:
            self.hw = self.scale_to
        self.fps = float(meta.average_fps)
        self.device, self.batch = device, batch
        self._buf: dict = {}
        self._next = 0
        self._lock = threading.Lock()
        self.decoded = 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        with self._lock:
            while i >= self._next:
                with _NVC_LOCK:
                    frames = self.dec.get_batch_frames(self.batch)
                if not frames:
                    self.n = self._next
                    raise IndexError(i)
                for t in self._convert(frames):
                    self._buf[self._next] = t
                    self._next += 1
                torch.cuda.current_stream().synchronize()
                self.decoded = self._next
            return self._buf[i]

    def release(self, i: int) -> None:
        with self._lock:
            self._buf.pop(i, None)

    def iter_batches(self):
        while True:
            with _NVC_LOCK:
                frames = self.dec.get_batch_frames(self.batch)
            if not frames:
                return
            batch = self._convert(frames)
            torch.cuda.current_stream().synchronize()
            yield batch

    def _convert(self, frames) -> list[torch.Tensor]:
        ts = [_frame_to_tensor(fr, self.device) for fr in frames]
        if self.scale_to is None:
            return ts
        x = torch.stack(ts).permute(0, 3, 1, 2).half()
        x = torch.nn.functional.interpolate(x, size=self.scale_to, mode="area")
        return list(x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous())


class SharedFrameStore:

    def __init__(
        self,
        video: Path,
        device: str = "cuda",
        gpu_id: int = 0,
        batch: int = 32,
        scale_to: tuple[int, int] | None = None,
        budget_bytes: int = 6 << 30,
        min_window: int = 256,
    ):
        import PyNvVideoCodec as nvc
        from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder

        self.dec = SimpleDecoder(str(video), gpu_id, 0, 0, True, 0, 0, 0, 4, nvc.OutputColorType.RGB)
        meta = self.dec.get_stream_metadata()
        self.n = int(meta.num_frames)
        self.hw = (int(meta.height), int(meta.width))
        self.fps = float(meta.average_fps)
        self.device, self.batch = device, batch
        self.scale_to = None if scale_to is None or tuple(scale_to) == self.hw else tuple(scale_to)
        if self.scale_to is not None:
            self.hw = self.scale_to
        self.max_window = max(min_window, int(budget_bytes // (self.hw[0] * self.hw[1] * 3)))
        self._buf: dict = {}
        self._cv = threading.Condition()
        self._head = 0
        self._low = 0
        self._done = False
        self.decoded = 0

    def __len__(self) -> int:
        return self.n

    def iter_batches(self):
        try:
            while True:
                with self._cv:
                    while self._head - self._low > self.max_window:
                        self._cv.wait(timeout=0.5)
                with _NVC_LOCK:
                    frames = self.dec.get_batch_frames(self.batch)
                if not frames:
                    return
                batch = GpuFrameStore._convert(self, frames)
                torch.cuda.current_stream().synchronize()
                with self._cv:
                    for t in batch:
                        self._buf[self._head] = t
                        self._head += 1
                    self.decoded = self._head
                    self._cv.notify_all()
                yield batch
        finally:
            with self._cv:
                self._done = True
                self.n = self._head
                self._cv.notify_all()

    def __getitem__(self, i: int) -> torch.Tensor:
        with self._cv:
            while i >= self._head and not self._done:
                self._cv.wait(timeout=0.5)
            if i >= self._head:
                raise IndexError(i)
            return self._buf[i]

    def release(self, i: int) -> None:
        with self._cv:
            self._buf.pop(i, None)
            self._low = max(self._low, i + 1)
            self._cv.notify_all()


class GpuVideoWriter:

    def __init__(
        self,
        out: Path,
        width: int,
        height: int,
        fps: float,
        qp: int = 14,
        preset: str = "P4",
        ffmpeg: str = "ffmpeg",
        max_bitrate: int | None = None,
        multipass: bool = False,
        stream=None,
        idr_period: int | None = None,
    ):
        import PyNvVideoCodec as nvc

        self.out, self.fps, self.ffmpeg = Path(out), fps, ffmpeg
        self.raw = self.out.with_suffix(".h264")
        self._stream = stream
        if max_bitrate:
            cap = int(max_bitrate)
            rc = dict(rc="vbr", bitrate=cap, maxbitrate=cap, vbvbufsize=cap)
            if multipass:
                rc["multipass"] = "fullres"
        else:
            rc = dict(rc="constqp", constqp=qp)
        self.enc = nvc.CreateEncoder(
            width,
            height,
            "ABGR",
            False,
            codec="h264",
            preset=preset,
            tuning_info="high_quality",
            fps=fps,
            bf=0,
            **({"idrperiod": int(idr_period)} if idr_period else {}),
            **rc,
        )
        self._f = stream if stream is not None else open(self.raw, "wb")
        self._alpha = torch.full((height, width, 1), 255, dtype=torch.uint8, device="cuda")
        self._nbuf = 8
        self._bufs = [torch.empty((height, width, 4), dtype=torch.uint8, device="cuda") for _ in range(self._nbuf)]
        self._q: queue.Queue = queue.Queue(maxsize=self._nbuf - 3)
        self._err: list = []
        self.n = 0
        self._th = threading.Thread(target=self._encode_loop, daemon=True)
        self._th.start()

    def _encode_loop(self) -> None:
        try:
            stream = torch.cuda.Stream()
            n = 0
            while True:
                item = self._q.get()
                if item is None:
                    return
                frame, ev = item
                x = self._bufs[n % self._nbuf]
                with torch.cuda.stream(stream):
                    stream.wait_event(ev)
                    frame.record_stream(stream)
                    x[..., :3] = frame.flip(-1)
                    x[..., 3] = self._alpha[..., 0]
                stream.synchronize()
                n += 1
                with _NVC_LOCK:
                    pkts = self.enc.Encode(x)
                for p in pkts:
                    self._f.write(p["data"])
        except BaseException as e:
            self._err.append(e)

    def write(self, frame_bgr: torch.Tensor) -> None:
        if self._err:
            raise RuntimeError(f"encoder thread died: {self._err[0]!r}")
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self._q.put((frame_bgr, ev))
        self.n += 1

    def abort(self) -> None:
        try:
            self._q.put(None, timeout=5)
        except queue.Full:
            return
        self._th.join(timeout=10)

    def close(self) -> Path:
        self._q.put(None)
        self._th.join()
        if self._err:
            raise RuntimeError(f"encoder thread died: {self._err[0]!r}")
        with _NVC_LOCK:
            pkts = self.enc.EndEncode()
        for p in pkts:
            self._f.write(p["data"])
        if self._stream is not None:
            self._f.flush()
            return self.out
        self._f.close()
        proc = subprocess.run(
            [
                self.ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                str(self.raw),
                "-c",
                "copy",
                str(self.out),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        self.raw.unlink(missing_ok=True)
        return self.out
