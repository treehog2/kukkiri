from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from kukkiri.errors import FFmpegError
from kukkiri.constants import DEFAULT_CRF, FRAME_EXT


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def probe(video: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,bit_rate:format=duration,bit_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {Path(video).name}: {proc.stderr.strip()[-300:]}")
    j = json.loads(proc.stdout)
    s, f = j["streams"][0], j.get("format", {})
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "nb_frames": int(s.get("nb_frames") or 0),
        "duration": float(f.get("duration") or 0.0),
        "bit_rate": int(s.get("bit_rate") or f.get("bit_rate") or 0),
    }


def frame_repeats(video: Path, fps: float, n_frames: int, duration: float) -> list[int] | None:
    if not (fps and n_frames and duration) or abs(n_frames - duration * fps) <= fps / 2:
        return None
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(video)],
        capture_output=True, text=True, check=True,
    )
    pts = np.sort([float(t) for t in proc.stdout.split() if t not in ("", "N/A")])
    if len(pts) != n_frames:
        return None
    slots = np.round((pts - pts[0]) * fps).astype(np.int64)
    if slots[-1] + 1 > 2 * n_frames:
        return None
    repeats = np.diff(slots, append=slots[-1] + 1)
    return None if (repeats == 1).all() else repeats.tolist()


def extract_frames(video: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vsync", "0", str(frames_dir / f"%06d{FRAME_EXT}")])
    return sorted(frames_dir.glob(f"*{FRAME_EXT}"))


def extract_audio(video: Path, audio_out: Path) -> Path | None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-acodec", "copy", str(audio_out)],
        capture_output=True,
        text=True,
    )
    return audio_out if proc.returncode == 0 and audio_out.exists() and audio_out.stat().st_size > 0 else None


def encode_video(frames_dir: Path, fps: float, out: Path, audio: Path | None = None, crf: int = DEFAULT_CRF) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}", "-i", str(frames_dir / f"%06d{FRAME_EXT}")]
    if audio is not None:
        cmd += ["-i", str(audio), "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)]
    _run(cmd)
    return out


def cut_section(video: Path, out: Path, start: float, duration: float | None, ffmpeg: str = "ffmpeg") -> Path:
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(video)]
    cmd += ["-t", f"{duration:.3f}"] if duration else []
    cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    _run(cmd)
    return out


def start_decode(video: Path, ffmpeg: str = "ffmpeg") -> dict:
    import fcntl

    info = probe(video)
    proc = subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None
    try:
        fcntl.fcntl(proc.stdout.fileno(), 1031, 1 << 20)
    except OSError:
        pass
    handle = {"proc": proc, "info": info, "result": None, "error": None}
    th = threading.Thread(target=_drain_decode, args=(handle,), daemon=True)
    th.start()
    handle["thread"] = th
    return handle


def _drain_decode(handle: dict) -> None:
    import os

    try:
        proc, info = handle["proc"], handle["info"]
        w, h = int(info["width"]), int(info["height"])
        n = int(info.get("nb_frames") or 0)
        fsz = h * w * 3
        fd = proc.stdout.fileno()
        buf = np.empty((n + 1 if n else 1024, h, w, 3), np.uint8)
        mv = memoryview(buf).cast("B")
        off, total = 0, buf.nbytes
        while True:
            k = os.readv(fd, [mv[off : min(off + (64 << 20), total)]])
            if not k:
                break
            off += k
            if off == total:
                buf = np.concatenate([buf, np.empty_like(buf)])
                mv = memoryview(buf).cast("B")
                total = buf.nbytes
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise FFmpegError(err)
        handle["result"] = buf[: off // fsz]
    except BaseException as e:
        handle["error"] = e


def finish_decode(handle: dict) -> np.ndarray:
    handle["thread"].join()
    if handle["error"] is not None:
        raise handle["error"]
    return handle["result"]


def iter_frames(video: Path, ffmpeg: str = "ffmpeg", chunk: int = 16):
    import fcntl
    import os

    info = probe(video)
    w, h = int(info["width"]), int(info["height"])
    fsz = h * w * 3
    proc = subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    try:
        fcntl.fcntl(fd, 1031, 1 << 20)
    except OSError:
        pass
    try:
        while True:
            buf = np.empty((chunk, h, w, 3), np.uint8)
            mv = memoryview(buf).cast("B")
            off, total = 0, buf.nbytes
            while off < total:
                k = os.readv(fd, [mv[off:total]])
                if not k:
                    break
                off += k
            n = off // fsz
            for i in range(n):
                yield buf[i]
            if n < chunk:
                break
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise FFmpegError(err)
    finally:
        if proc.poll() is None:
            proc.kill()


class FrameStore:

    def __init__(self, video: Path, ffmpeg: str = "ffmpeg", prefetch: int = 720):
        info = probe(video)
        self.n = int(info.get("nb_frames") or 0)
        self.hw = (int(info["height"]), int(info["width"]))
        self._it = iter_frames(video, ffmpeg)
        self._buf: dict = {}
        self._next = 0
        self.decoded = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._released = 0
        self._prefetch = prefetch
        self._eof = False
        self._th = threading.Thread(target=self._pump, daemon=True)
        self._th.start()

    def _pump(self) -> None:
        while True:
            with self._cv:
                while not self._eof and self._next - self._released >= self._prefetch:
                    self._cv.wait()
                if self._eof:
                    return
            try:
                f = next(self._it)
            except StopIteration:
                with self._cv:
                    self._eof = True
                    self.n = self._next
                    self._cv.notify_all()
                return
            with self._cv:
                self._buf[self._next] = f
                self._next += 1
                self.decoded = self._next
                self._cv.notify_all()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> np.ndarray:
        with self._cv:
            while i >= self._next:
                if self._eof:
                    raise IndexError(i)
                self._cv.wait()
            return self._buf[i]

    def release(self, i: int) -> None:
        with self._cv:
            self._buf.pop(i, None)
            if i >= self._released:
                self._released = i + 1
            self._cv.notify_all()


def decode_frames(video: Path, ffmpeg: str = "ffmpeg") -> np.ndarray:
    return finish_decode(start_decode(video, ffmpeg))


def encode_frames(
    frames,
    fps: float,
    out: Path,
    audio: Path | None = None,
    crf: int = DEFAULT_CRF,
    nvenc: bool = False,
    ffmpeg: str = "ffmpeg",
) -> Path:
    first = None
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
    ]
    if audio is not None:
        cmd += ["-i", str(audio), "-c:a", "copy", "-shortest"]
    if nvenc:
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(np.ascontiguousarray(first).tobytes())
    for f in it:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise FFmpegError(err)
    return out


def read_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FFmpegError(f"unreadable frame {path}")
    return img


def write_frame(path: Path, frame: np.ndarray) -> None:
    cv2.imwrite(str(path), frame)
