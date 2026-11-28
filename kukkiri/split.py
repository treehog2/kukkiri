
from __future__ import annotations

import os
import sys
from pathlib import Path

SEARCH_S = (30.0, 120.0)


def find_cut(src: Path, target: int, fps: float, n_frames: int, cut_tv: float, stride: int = 1,
             scale_to: tuple[int, int] | None = None) -> int | None:
    for s in SEARCH_S:
        try:
            cut = _cut_within(src, target, round(s * fps), n_frames, cut_tv, stride, scale_to)
        except Exception:
            cut = None
        if cut is not None:
            return cut
    return None


def _cut_within(src: Path, target: int, reach: int, n_frames: int, cut_tv: float, stride: int,
                scale_to: tuple[int, int] | None) -> int | None:
    import torch

    from kukkiri.remaster import _cut_flags
    from kukkiri.video_io_gpu import SharedFrameStore

    a, b = max(1, target - reach), min(n_frames, target + reach)
    if b - a < 2:
        return None
    frames = SharedFrameStore(src, scale_to=scale_to, stride=stride, budget_bytes=0, min_window=64, start=a, end=b)
    first = -(-a // stride) * stride
    cuts, prev, i = [], None, 0
    for batch in frames.iter_batches():
        flags, prev = _cut_flags(batch, prev, cut_tv)
        cuts += [first + (i + k) * stride for k, f in enumerate(flags) if f and i + k > 0]
        for k in range(len(batch)):
            frames.release(i + k)
        i += len(batch)
    del frames
    torch.cuda.empty_cache()
    return min(cuts, key=lambda c: abs(c - target)) if cuts else None


def plan(src: Path, n_parts: int, fps: float, n_frames: int, cut_tv: float, stride: int = 1,
         scale_to: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    bounds = [0]
    for k in range(1, n_parts):
        cut = find_cut(src, round(n_frames * k / n_parts), fps, n_frames, cut_tv, stride, scale_to)
        if cut is not None and cut > bounds[-1]:
            bounds.append(cut)
    bounds.append(n_frames)
    return list(zip(bounds, bounds[1:]))



JOB_SUFFIX = ".kukkiri-job"
TRUNCATED_EXIT = 86


def job_dir(dst: Path) -> Path:
    return Path(dst).with_name(Path(dst).name + JOB_SUFFIX)


def _write_json(path: Path, obj) -> None:
    import json

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _fsync(path: Path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def lock_path(job: Path) -> Path:
    return job.with_name(job.name + ".lock")


def _lock(job: Path):
    import errno

    job.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path(job), "a+")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                same = os.stat(lock_path(job)).st_ino == os.fstat(f.fileno()).st_ino
            except FileNotFoundError:
                same = False
            if not same:
                raise BlockingIOError(errno.EWOULDBLOCK, "lock file replaced")
    except OSError as e:
        f.close()
        if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, getattr(errno, "EDEADLOCK", errno.EDEADLK)):
            raise JobBusy(f"another kukkiri is already restoring to {job.name[: -len(JOB_SUFFIX)]}") from None
        raise
    return f


def _unlock(job: Path, f) -> None:
    if os.name != "nt":
        lock_path(job).unlink(missing_ok=True)
    f.close()
    if os.name == "nt":
        try:
            lock_path(job).unlink(missing_ok=True)
        except OSError:
            pass


def _clear(job: Path) -> None:
    import shutil

    left = []
    for p in job.iterdir():
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
        except OSError:
            left.append(p.name)
    if left:
        raise RuntimeError(f"could not start {job.name[: -len(JOB_SUFFIX)]} over: {', '.join(left)} in {job} is in "
                           "use (close any program that has it open) or cannot be removed")


def _identity(src: Path, cfg, threshold: float | None, weights_dir: Path, section=None) -> dict:
    from dataclasses import asdict

    from kukkiri import __version__

    import json

    st = Path(src).stat()
    ident = {"source": str(Path(src).resolve()), "size": st.st_size, "mtime": int(st.st_mtime), "config": asdict(cfg),
             "threshold": threshold, "weights": str(Path(weights_dir).resolve()), "section": section,
             "kukkiri": __version__}
    return json.loads(json.dumps(ident))


def _assign(bounds: list[list[int]], gpus: list[int]) -> dict[int, list[int]]:
    load = {g: 0 for g in gpus}
    out: dict[int, list[int]] = {g: [] for g in gpus}
    for i in sorted(range(len(bounds)), key=lambda i: bounds[i][0] - bounds[i][1]):
        g = min(load, key=load.get)
        out[g].append(i)
        load[g] += bounds[i][1] - bounds[i][0]
    return {g: sorted(v) for g, v in out.items() if v}


def part_path(job: Path, i: int, codec: str) -> Path:
    return job / f"part{i:03d}.{codec}"


COORDINATOR_ENV = "KUKKIRI_COORDINATOR_PID"


def _exit_with_parent() -> None:
    import time

    parent = int(os.environ.get(COORDINATOR_ENV) or os.getppid())
    if os.name == "nt":
        if not _wait_for_windows_process(parent):
            os._exit(1)
    else:
        while os.getppid() == parent:
            time.sleep(1)
    os._exit(1)


def _wait_for_windows_process(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype, k.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.WaitForSingleObject.restype, k.WaitForSingleObject.argtypes = wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]

    def started(h) -> int | None:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(h, *(ctypes.byref(x) for x in t)):
            return None
        return t[0].dwHighDateTime << 32 | t[0].dwLowDateTime

    handle = k.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return False
    theirs, ours = started(handle), started(k.GetCurrentProcess())
    if theirs is None or ours is None or theirs > ours:
        return False
    k.WaitForSingleObject(handle, 0xFFFFFFFF)
    return True


def run_parts(job: Path, parts: list[int], weights_dir: Path, ffmpeg: str = "ffmpeg") -> None:
    import json
    import sys
    import threading

    from kukkiri.errors import SourceTruncated
    from kukkiri.remaster import RemasterConfig, _remaster_once

    threading.Thread(target=_exit_with_parent, daemon=True).start()
    meta = json.loads((job / "job.json").read_text())
    cfg = RemasterConfig(**meta["identity"]["config"])
    for i in parts:
        out = part_path(job, i, cfg.codec)
        if (job / f"part{i:03d}.json").exists():
            continue

        def progress(ev, i=i):
            if ev.stage == "remaster":
                print(json.dumps({"part": i, "done": ev.frames_done}), flush=True)

        a, b = meta["parts"][i]
        try:
            r = _remaster_once(meta["readable"], out, weights_dir, config=cfg, threshold=meta["identity"]["threshold"],
                               ffmpeg=ffmpeg, progress=progress, frame_range=(a, b), part=True,
                               original=meta["cap_source"])
            want = len(range(-(-a // meta["stride"]) * meta["stride"], b, meta["stride"]))
            if r.frames != want and not _short_end_ok(r.frames, want, i, meta):
                raise SourceTruncated(r.frames, want, meta["fps"])
        except SourceTruncated as e:
            print(f"part {i}: {e}", file=sys.stderr, flush=True)
            sys.exit(TRUNCATED_EXIT)
        _fsync(out)
        _write_json(job / f"part{i:03d}.json", r.to_dict())
        print(json.dumps({"part": i, "finished": True}), flush=True)


def _short_end_ok(got: int, want: int, i: int, meta: dict) -> bool:
    return i == len(meta["parts"]) - 1 and want - max(2, round(meta["fps"])) <= got <= want


class JobBusy(RuntimeError):
    pass


BUSY_EXIT = 4


class _Truncated(Exception):

    def __init__(self, expected: int):
        super().__init__(expected)
        self.expected = expected


def remaster_split(src, dst, weights_dir, *, config, gpus: list[int], part_minutes: float = 10.0,
                   threshold: float | None = None, ffmpeg: str = "ffmpeg", progress=None, original=None,
                   section=None, restart: bool = False, **_):
    import shutil

    job = job_dir(dst)

    def seen():
        try:
            st = Path(dst).stat()
            return st.st_mtime_ns, st.st_size
        except FileNotFoundError:
            return None

    before = seen()
    lock = _lock(job)
    held = seen()
    if held != before:
        _unlock(job, lock)
        raise JobBusy(f"another kukkiri has just restored to {Path(dst).name}")
    try:
        job.mkdir(exist_ok=True)
        if restart:
            _clear(job)
        try:
            r = _split_once(src, dst, weights_dir, config, gpus, part_minutes, threshold, ffmpeg, progress, original,
                            section, repaired=None)
        except _Truncated as t:
            print("a part ended early (a damaged source): restoring again from a repaired copy", file=sys.stderr,
                  flush=True)
            r = _split_once(src, dst, weights_dir, config, gpus, part_minutes, threshold, ffmpeg, progress, original,
                            section, repaired=t.expected)
        shutil.rmtree(job, ignore_errors=True)
        return r
    except BaseException:
        if held is None:
            Path(dst).unlink(missing_ok=True)
        raise
    finally:
        _unlock(job, lock)


def _split_once(src, dst, weights_dir, cfg, gpus, part_minutes, threshold, ffmpeg, progress, original, section,
                repaired: int | None):
    import json
    import shutil
    import subprocess
    import threading
    import time

    from kukkiri.device import Device, pick
    from kukkiri.mux import LiveMuxer
    from kukkiri.remaster import ProgressEvent, RemasterResult, _output_size, remaster, stream_rate
    from kukkiri.video_io import frame_repeats, probe
    from kukkiri.video_io_gpu import gpu_readable_source, repair_source

    if pick() is not Device.CUDA:
        raise RuntimeError("--gpus needs NVIDIA GPUs")
    t0 = time.time()
    src, dst, weights_dir = Path(src), Path(dst), Path(weights_dir)
    given = Path(original) if original else src
    job = job_dir(dst)
    ident = _identity(given, cfg, threshold, weights_dir, section)
    meta_path = job / "job.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    except ValueError:
        meta = None
    if meta is None:
        _clear(job)
    if meta is not None and (meta.get("identity") != ident or (repaired is not None and not meta.get("repaired"))):
        _clear(job)
        meta = None
    if progress:
        progress(ProgressEvent(0.0, "preparing", 0, 0))
    readable = Path(meta["readable"]) if meta is not None else None
    if readable is not None and readable.parent != job.resolve() and readable != Path(ident["source"]):
        readable = None
    if readable is None or not readable.exists():
        base = src
        if section is not None:
            base = job / "section.mp4"
            if not base.exists():
                shutil.copyfile(src, job / "section.mp4.tmp")
                (job / "section.mp4.tmp").replace(base)
        if repaired is not None or (meta is not None and meta.get("repaired")):
            base = repair_source(base, job, ffmpeg=ffmpeg)
        readable = Path(gpu_readable_source(base, job, ffmpeg=ffmpeg)).resolve()
    info = probe(readable)
    fps, stride = stream_rate(info, cfg, given.name)
    n_src = int(info.get("nb_frames") or 0)
    if repaired is not None and n_src < repaired - max(2, round(fps)):
        from kukkiri.errors import SourceTruncated

        raise SourceTruncated(n_src, repaired, fps)
    if not n_src:
        _clear(job)
        return remaster(src, dst, weights_dir, config=cfg, threshold=threshold, ffmpeg=ffmpeg, progress=progress,
                        **({"original": original} if original else {}))
    height, width = int(info.get("height") or 0), int(info.get("width") or 0)
    scale_to = None if cfg.vr else _output_size(height, width, cfg.max_pixels)
    if meta is None:
        n_parts = max(len(gpus), round(n_src / fps / 60 / part_minutes))
        bounds = plan(readable, n_parts, fps, n_src, cfg.cut_tv, stride, scale_to)
        meta = {"identity": ident, "readable": str(readable), "parts": [list(b) for b in bounds],
                "repaired": repaired is not None}
    elif meta["readable"] != str(readable):
        meta["readable"] = str(readable)
    meta["cap_source"] = str((job / "section.mp4").resolve()) if section is not None else str(given.resolve())
    meta["fps"], meta["stride"] = fps, stride
    _write_json(meta_path, meta)
    bounds = meta["parts"]
    _check_disk(dst, given, job)
    fps_out = fps / stride
    repeats = frame_repeats(readable, fps_out, n_src, float(info.get("duration") or 0)) if stride == 1 else None

    def n_kept(a: int, b: int) -> int:
        return len(range(-(-a // stride) * stride, b, stride))

    def n_out(a: int, b: int) -> int:
        return sum(repeats[a:b]) if repeats else n_kept(a, b)

    total_out = n_out(0, n_src)
    def whole(i: int) -> bool:
        try:
            r = json.loads((job / f"part{i:03d}.json").read_text())
            return part_path(job, i, cfg.codec).stat().st_size == r["output_bytes"]
        except (OSError, ValueError, KeyError):
            return False

    done = {i: n_out(*bounds[i]) for i in range(len(bounds)) if whole(i)}
    for i in set(range(len(bounds))) - set(done):
        (job / f"part{i:03d}.json").unlink(missing_ok=True)
        for stale in (part_path(job, i, cfg.codec), part_path(job, i, cfg.codec).with_suffix(f".{cfg.codec}.tmp")):
            stale.unlink(missing_ok=True)
    todo = [i for i in range(len(bounds)) if i not in done]
    if done:
        print(f"resuming: {len(done)} of {len(bounds)} parts already restored", file=sys.stderr, flush=True)
    finished = {i: threading.Event() for i in range(len(bounds))}
    for i in done:
        finished[i].set()
    started = threading.Event()
    procs: list = []
    plock = threading.Lock()
    stop = threading.Event()
    errors: list[str] = []
    truncated: list[int] = []
    progress_parts: dict[int, int] = {}

    def fail(msg: str) -> None:
        errors.append(msg)
        started.set()
        for e in finished.values():
            e.set()

    def worker(g: int, ids: list[int]) -> None:
        log_path = job / f"gpu{'' if g is None else g}.log"
        name = "the GPU" if g is None else f"GPU {g}"
        try:
            env = {**os.environ, "PYTHONWARNINGS": "ignore", "KUKKIRI_KEEP_MODELS": "1",
                   COORDINATOR_ENV: str(os.getpid())}
            if g is not None:
                env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g))
            cmd = [sys.executable, "-m", "kukkiri", "_parts", str(job), ",".join(map(str, ids)), "--weights",
                   str(weights_dir), "--ffmpeg", ffmpeg]
            with open(log_path, "a") as log:
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
                with plock:
                    procs.append(p)
                    if stop.is_set():
                        p.kill()
                for line in p.stdout:
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(msg, dict) or "part" not in msg:
                        continue
                    started.set()
                    if msg.get("finished"):
                        done[msg["part"]] = n_out(*bounds[msg["part"]])
                        finished[msg["part"]].set()
                    elif "done" in msg:
                        progress_parts[msg["part"]] = msg["done"]
                p.wait()
            if p.returncode == TRUNCATED_EXIT:
                truncated.append(g)
            if p.returncode != 0:
                fail(f"{name} failed (exit {p.returncode}): {log_path.read_text(errors='replace')[-1500:]}")
        except BaseException as e:
            fail(f"{name}: {e!r}")

    jobs = [(g, [todo[k] for k in ks]) for g, ks in _assign([bounds[i] for i in todo], gpus).items()]
    threads = [threading.Thread(target=worker, args=(g, ids), daemon=True) for g, ids in jobs]

    def launch() -> None:
        if not threads:
            return
        threads[0].start()
        started.wait()
        for t in threads[1:]:
            if not errors and not stop.is_set():
                t.start()

    launcher = threading.Thread(target=launch, daemon=True)
    launcher.start()

    mux, written, results, first, mux_size = None, 0, [], None, None

    def report() -> None:
        if progress:
            nonlocal first
            got = sum(done.values()) + sum(v for k, v in progress_parts.items() if k not in done)
            first = first or (time.time(), got)
            rate = (got - first[1]) / max(time.time() - first[0], 1e-3)
            eta = (total_out - got) / rate if rate > 0 else None
            progress(ProgressEvent(min(99.0, 100.0 * got / max(total_out, 1)), "remaster", got, total_out, eta))

    try:
        for i in range(len(bounds)):
            path, f, fed, held = part_path(job, i, cfg.codec), None, 0, []
            try:
                while True:
                    ended = finished[i].wait(timeout=0.5)
                    if errors:
                        if truncated and repaired is None:
                            raise _Truncated(n_src)
                        raise RuntimeError(errors[0])
                    want, r = None, None
                    if ended:
                        r = json.loads((job / f"part{i:03d}.json").read_text())
                        if r["frames"] != n_kept(*bounds[i]) and not _short_end_ok(r["frames"], n_kept(*bounds[i]), i, meta):
                            (job / f"part{i:03d}.json").unlink(missing_ok=True)
                            raise RuntimeError(f"part {i} restored {r['frames']} frames, expected {n_kept(*bounds[i])}")
                        want = r["output_bytes"]
                    if f is None and path.exists():
                        f = open(path, "rb")
                    if mux is None and held and r:
                        mux = LiveMuxer(readable, dst, fps_out, total_out / fps_out, ffmpeg=ffmpeg, sar=info.get("sar"),
                                        size=(r["width"], r["height"]), rotation=info.get("rotation", 0),
                                        codec=cfg.codec)
                        mux_size = (r["width"], r["height"])
                        for d in held:
                            mux.stdin.write(d)
                        held = []
                    while f is not None:
                        data = f.read(8 << 20 if want is None else min(8 << 20, want - fed))
                        if not data:
                            break
                        fed += len(data)
                        if mux is None:
                            held.append(data)
                            size = (r["width"], r["height"]) if r else _stream_size(path) if fed >= 1 << 18 else None
                            if size:
                                mux = LiveMuxer(readable, dst, fps_out, total_out / fps_out, ffmpeg=ffmpeg,
                                                sar=info.get("sar"), size=size, rotation=info.get("rotation", 0),
                                                codec=cfg.codec)
                                mux_size = size
                                for d in held:
                                    mux.stdin.write(d)
                                held = []
                        else:
                            mux.stdin.write(data)
                    if ended:
                        if fed != want or mux is None:
                            raise RuntimeError(f"part {i}: joined {fed} bytes of {want}")
                        if mux_size and tuple(mux_size) != (r["width"], r["height"]):
                            raise RuntimeError(f"part {i} is {r['width']}x{r['height']}, the output {mux_size}")
                        break
                    report()
            finally:
                if f is not None:
                    f.close()
            results.append((written, r))
            written += n_out(*bounds[i])
        if mux is None:
            raise RuntimeError("nothing to join")
        mux.stdin.flush()
    except BaseException:
        stop.set()
        started.set()
        with plock:
            running = list(procs)
        for p in running:
            if p.poll() is None:
                p.kill()
        launcher.join()
        with plock:
            running = list(procs)
        for p in running:
            if p.poll() is None:
                p.kill()
            p.wait()
        for t in threads:
            if t.is_alive():
                t.join()
        if mux is not None:
            mux.kill()
        raise
    if progress:
        progress(ProgressEvent(99.0, "remux", total_out, total_out, 5.0))
    mux.close()
    for t in threads:
        if t.is_alive():
            t.join()
    if progress:
        progress(ProgressEvent(100.0, "remux", total_out, total_out, 0.0))
    first_r = results[0][1]
    return RemasterResult(
        output=str(dst), duration_s=float(info.get("duration") or 0), frames=sum(r["frames"] for _, r in results),
        frames_touched=sum(r["frames_touched"] for _, r in results), n_tracks=sum(r["n_tracks"] for _, r in results),
        n_clips=sum(r["n_clips"] for _, r in results), detect_s=0.0, remaster_s=round(time.time() - t0, 1),
        total_s=round(time.time() - t0, 1), width=first_r["width"], height=first_r["height"], fps=round(fps_out, 3),
        source_bitrate=first_r["source_bitrate"], output_bitrate=int(probe(dst).get("bit_rate") or 0),
        output_bytes=dst.stat().st_size,
        tracks=[(s + off, e + off, a) for off, r in results for s, e, a in r["tracks"]],
    )


def _stream_size(path: Path) -> tuple[int, int] | None:
    from kukkiri.video_io import probe

    try:
        info = probe(path)
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        return (w, h) if w and h else None
    except Exception:
        return None


def _check_disk(dst: Path, src: Path, job: Path) -> None:
    import shutil

    free = shutil.disk_usage(job).free
    need = 3 * Path(src).stat().st_size
    if free < need:
        print(f"warning: {free / 1e9:.0f} GB free next to the output; a split restore of this video may need about "
              f"{need / 1e9:.0f} GB there while it runs", file=sys.stderr, flush=True)
