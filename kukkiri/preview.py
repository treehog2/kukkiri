from __future__ import annotations

import subprocess
from pathlib import Path

PREVIEW_S = 30
PREVIEW_COUNT = 3
PREVIEW_SPACING_S = 300
PREVIEW_MIN_AREA = 0.01
PREVIEW_MIN_LEN_S = 1.0
PREVIEW_MIN_SCORE = 0.25
PREVIEW_SIDE_W = 640
WATERMARK = Path(__file__).with_name("watermark.png")
WATERMARK_OPACITY = 0.5
WATERMARK_HEIGHT = 0.03


def preview_windows(tracks: list, fps: float, duration_s: float, frame_px: int) -> list[float]:
    if duration_s <= PREVIEW_S or not fps:
        return [0.0]
    tracks = [t for t in tracks if t[2] >= PREVIEW_MIN_AREA * frame_px and t[1] - t[0] + 1 >= PREVIEW_MIN_LEN_S * fps]
    if not tracks:
        return []
    n = int(duration_s) + 1
    area = [0.0] * n
    count = [0] * n
    for start, end, a in tracks:
        for sec in range(int(start / fps), min(int(end / fps), n - 1) + 1):
            area[sec] += a
            count[sec] += 1
    single = [a if c == 1 else 0.0 for a, c in zip(area, count)]
    last = int(duration_s - PREVIEW_S)
    spacing = min(PREVIEW_SPACING_S, max(PREVIEW_S, duration_s / (PREVIEW_COUNT + 1)))

    def best(per_sec: list[float], picked: list[float]) -> tuple[float, float] | None:
        scores = sorted(((sum(per_sec[t : t + PREVIEW_S]), t) for t in range(last + 1)), key=lambda x: -x[0])
        for score, t in scores:
            if score <= 0:
                return None
            if all(abs(t - p) >= spacing for p in picked):
                return score, float(t)
        return None

    picked: list[float] = []
    floor = 0.0
    for i in range(PREVIEW_COUNT):
        hit = best(single if i == PREVIEW_COUNT - 1 else area, picked)
        if hit is None and i == PREVIEW_COUNT - 1:
            hit = best(area, picked)
        if hit is None:
            break
        t = int(hit[1])
        score = sum(area[t : t + PREVIEW_S])
        if score < floor:
            break
        floor = floor or PREVIEW_MIN_SCORE * score
        picked.append(hit[1])
    return sorted(picked)


def render_preview(src: Path, restored: Path, start: float, fps: float, width: int, height: int, out: Path,
                   ffmpeg: str = "ffmpeg", src_offset: float = 0.0) -> None:
    half_h = max(2, round(PREVIEW_SIDE_W * height / max(width, 1) / 2) * 2)
    wm_h = max(16, round(2 * PREVIEW_SIDE_W * WATERMARK_HEIGHT))
    margin = max(6, round(wm_h * 0.4))
    rate = f"fps={fps:.3f}," if fps else ""
    side = f"scale={PREVIEW_SIDE_W}:{half_h}"
    graph = (f"[0:v]{rate}{side}[b];[1:v]{rate}{side}[a];[b][a]hstack=inputs=2[v];"
             f"[2:v]scale=-1:{wm_h},format=rgba,colorchannelmixer=aa={WATERMARK_OPACITY}[w];"
             f"[v][w]overlay=W-w-{margin}:{margin}")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-hide_banner"]
    cmd += ["-ss", f"{start + src_offset:.3f}", "-t", str(PREVIEW_S), "-i", str(src)]
    cmd += ["-ss", f"{start:.3f}", "-t", str(PREVIEW_S), "-i", str(restored)]
    cmd += ["-i", str(WATERMARK)]
    cmd += ["-filter_complex", graph, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}-{s // 60 % 60:02d}-{s % 60:02d}" if s >= 3600 else f"{s // 60}-{s % 60:02d}"


def make_previews(src: Path, restored: Path, result, ffmpeg: str = "ffmpeg", src_offset: float = 0.0, log=print) -> list[Path]:
    try:
        starts = preview_windows(result.tracks or [], result.fps, result.duration_s, result.width * result.height)
    except Exception as e:
        log(f"previews skipped: {e}")
        return []
    if not starts:
        log("previews: nothing restored that is worth showing")
        return []
    written = []
    for start in starts:
        out = Path(restored).with_name(f"{Path(restored).stem}.preview-{_stamp(start)}.mp4")
        try:
            render_preview(Path(src), Path(restored), start, result.fps, result.width, result.height, out, ffmpeg, src_offset)
            written.append(out)
            log(f"preview: {out}")
        except Exception as e:
            detail = getattr(e, "stderr", "") or ""
            log(f"preview at {_stamp(start)} failed: {e} {detail[-300:]}")
            out.unlink(missing_ok=True)
    return written
