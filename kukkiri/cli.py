from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from kukkiri.errors import FFmpegError
from kukkiri.weights import MODELS


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".ts", ".m2ts", ".mpg", ".mpeg", ".m4v", ".webm", ".flv"}
OUTPUT_TAG = ".kukkiri"


def collect_videos(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            found = (f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
            out += sorted(f for f in found if OUTPUT_TAG not in f.stem)
        else:
            out.append(p)
    return out


def _seconds(text: str) -> float:
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def _windows_nvidia_adapters() -> list[tuple[str, float]] | None:
    import json
    import subprocess

    cmd = "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | ConvertTo-Json"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=20)
        rows = json.loads(out.stdout)
    except Exception:
        return None
    found = []
    for r in [rows] if isinstance(rows, dict) else rows:
        if "nvidia" in (r.get("Name") or "").lower():
            digits = (r.get("DriverVersion") or "").replace(".", "")[-5:]
            found.append((r["Name"], int(digits) / 100 if digits.isdigit() else 0.0))
    return found


def _gpu_problem() -> str | None:
    import torch

    if not torch.cuda.is_available():
        need = "kukkiri needs an NVIDIA GTX 16-series / RTX 20-series or newer GPU and driver 580 or newer."
        found = _windows_nvidia_adapters() if sys.platform == "win32" else None
        if found == []:
            return f"no NVIDIA GPU found. {need}"
        if found:
            name, version = found[0]
            update = " Newer drivers: nvidia.com/drivers" if version < 580 else ""
            return f"found {name} with driver {version:.2f}, which kukkiri cannot use. {need}{update}"
        return "no usable NVIDIA GPU found. kukkiri needs an NVIDIA GPU and driver 580 or newer."
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 5):
        return (f"{torch.cuda.get_device_name()} (compute capability {major}.{minor}) is not supported. "
                "kukkiri needs an NVIDIA GTX 16-series / RTX 20-series or newer GPU.")
    return None


def _restore(args) -> int:
    problem = _gpu_problem()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1
    from kukkiri.weights import default_dir, download, missing, version_of

    try:
        version = version_of(args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    args.model = f"kiyo-{version}"
    weights = Path(args.weights) if args.weights else default_dir(version)
    if missing(weights, version):
        print(f"downloading the {args.model} model weights into {weights}", file=sys.stderr)
        try:
            download(weights, log=lambda m: print(m, file=sys.stderr), version=version)
        except Exception as e:
            print(f"error: could not download the weights: {e}", file=sys.stderr)
            return 1
    args.weights = str(weights)
    videos = collect_videos(args.input)
    if not videos:
        print("error: no videos found", file=sys.stderr)
        return 1
    if len(args.input) == 1 and len(videos) == 1 and not Path(args.input[0]).is_dir():
        src = videos[0]
        dst = Path(args.output) if args.output else src.with_name(f"{src.stem}{_tag(args)}.mp4")
        try:
            return _restore_one(src, dst, args)
        except FFmpegError as e:
            print(f"failed: {src}: {e}", file=sys.stderr)
            return 1
    outdir = Path(args.output) if args.output else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    results, used = {}, set()
    for i, src in enumerate(videos, 1):
        dst = (outdir or src.parent) / f"{src.stem}{_tag(args)}.mp4"
        n = 2
        while dst in used:
            dst = dst.with_name(f"{src.stem}{_tag(args)} ({n}).mp4")
            n += 1
        used.add(dst)
        print(f"[{i}/{len(videos)}] {src}", file=sys.stderr)
        try:
            results[src] = "restored" if _restore_one(src, dst, args) == 0 else "declined"
        except Exception as e:
            print(f"\nfailed: {src}: {e!r}", file=sys.stderr)
            results[src] = "failed"
    counts = {k: sum(v == k for v in results.values()) for k in ("restored", "declined", "failed")}
    print(f"batch done: {counts['restored']} restored, {counts['declined']} declined, {counts['failed']} failed",
          file=sys.stderr)
    for src, r in results.items():
        if r != "restored":
            print(f"  {r}: {src}", file=sys.stderr)
    return 1 if counts["failed"] else 0


def _tag(args) -> str:
    tag = f"{OUTPUT_TAG}-{args.model}"
    if args.start or args.duration:
        tag += "." + "_".join(t.replace(":", "-") for t in (args.start or "0", args.duration) if t)
    return tag


def _restore_one(src: Path, dst: Path, args) -> int:
    from kukkiri.errors import AlreadyRemasteredError, NotRemasterableError
    from kukkiri.gate import scan_keyframes
    from kukkiri.remaster import remaster
    from kukkiri.video_io import cut_section
    from kukkiri.weights import detector_checkpoint

    weights = Path(args.weights)
    if args.start or args.duration:
        with tempfile.TemporaryDirectory(prefix="kukkiri_cut_") as tmp:
            start, duration = _seconds(args.start or "0"), _seconds(args.duration) if args.duration else None
            until = f"+ {args.duration}" if args.duration else "to the end"
            print(f"cutting {args.start or '0'} {until}", file=sys.stderr)
            section = cut_section(src, Path(tmp) / "section.mp4", start, duration, ffmpeg=args.ffmpeg)
            return _run_remaster(remaster, section, dst, weights, args)
    if not args.skip_check:
        from kukkiri.weights import detector_size, version_of

        scan = scan_keyframes(src, detector_checkpoint(weights), ffmpeg=args.ffmpeg, size=detector_size(version_of(args.model)))
        print(f"check: mosaic on {scan.mosaic_frac:.1%} of keyframes, bare genitals on {scan.genital_frac:.1%}"
              f" ({scan.keyframes} keyframes, {scan.seconds:.0f} s)", file=sys.stderr)
        if not scan.restorable:
            err = scan.error(scan.reason)
            kind = "already uncensored or restored" if isinstance(err, AlreadyRemasteredError) else "nothing to restore"
            assert isinstance(err, (AlreadyRemasteredError, NotRemasterableError))
            print(f"skipped: {kind} ({scan.reason}). Use --skip-check to restore anyway.", file=sys.stderr)
            return 2
    return _run_remaster(remaster, src, dst, weights, args)


def _run_remaster(remaster, src: Path, dst: Path, weights: Path, args) -> int:
    def progress(ev):
        eta = f", {ev.eta_s / 60:.1f} min left" if ev.eta_s else ""
        print(f"\r{ev.stage}: {ev.pct:5.1f}%{eta}   ", end="", file=sys.stderr, flush=True)

    from kukkiri.remaster import RemasterConfig

    from kukkiri.weights import detector_size, version_of

    cfg = RemasterConfig(detector_size=detector_size(version_of(args.model)),
                         max_pixels=1920 * 1088 if args.max_1080p else None)
    existed = Path(dst).exists()
    try:
        r = remaster(src, dst, weights, config=cfg, threshold=args.threshold, ffmpeg=args.ffmpeg, progress=progress)
    except BaseException:
        if not existed:
            Path(dst).unlink(missing_ok=True)
        raise
    print(f"\ndone: {r.output} ({r.frames_touched} of {r.frames} frames restored, {r.total_s:.0f} s)", file=sys.stderr)
    if args.previews:
        from kukkiri.preview import make_previews

        make_previews(src, Path(r.output), r, ffmpeg=args.ffmpeg, log=lambda m: print(m, file=sys.stderr))
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kukkiri", description="Restore mosaic-censored video.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("restore", help="restore videos")
    r.add_argument("input", nargs="+", help="video files and/or folders (searched recursively)")
    r.add_argument("-o", "--output", help="output mp4 for one input file, else an output folder "
                   "(default: <name>.kukkiri-<model>.mp4 next to each input)")
    r.add_argument("--model", help=f"model to restore with: {', '.join(MODELS)} (default: the latest, {MODELS[0]}); "
                   "restored files are named after it")
    r.add_argument("--previews", action="store_true",
                   help="also write up to 3 watermarked before/after clips of the most restored moments")
    r.add_argument("--weights", help="weights directory (default: KUKKIRI_WEIGHTS or a per-user folder); "
                   "missing weights are downloaded on first use")
    r.add_argument("--threshold", type=float, default=None, help="detector score threshold (default: by resolution)")
    r.add_argument("--start", help="restore only from this time (seconds, mm:ss or hh:mm:ss)")
    r.add_argument("--duration", help="restore only this long (seconds, mm:ss or hh:mm:ss)")
    r.add_argument("--skip-check", action="store_true", help="restore even if the input looks uncensored")
    r.add_argument("--max-1080p", action="store_true", help="serve sources above 1080p at 1080p (faster, smaller)")
    r.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    r.add_argument("-v", "--verbose", action="store_true")
    d = sub.add_parser("download-weights", help="download and verify the model weights")
    d.add_argument("dest", nargs="?", help="weights directory (default: KUKKIRI_WEIGHTS or a per-user folder)")
    d.add_argument("--model", help=f"which model: {', '.join(MODELS)} (default: the latest)")
    d.add_argument("--base-url", help="download location, if not the default")
    args = p.parse_args(argv)
    if args.cmd == "download-weights":
        from kukkiri.weights import default_dir, download, version_of

        try:
            version = version_of(args.model)
            dest = Path(args.dest) if args.dest else default_dir(version)
            download(dest, args.base_url, log=lambda m: print(m, file=sys.stderr), version=version)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")
    sys.exit(_restore(args))
