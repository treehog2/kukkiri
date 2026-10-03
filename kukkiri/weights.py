from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

VERSION = "v4"
BASE_URL: str | None = f"https://huggingface.co/treehog/kukkiri/resolve/main/{VERSION}"
ONNX = "export_b32_fp16/rfdetr-seg-medium.h16.onnx"
FILES = {
    f"kiyo-{VERSION}-detector.safetensors": (
        "detector.safetensors", 141176868, "43a37ca65c3d6a6d8cca9c2feb8e5621b908bb085a718208f790a62ccb821327"
    ),
    f"kiyo-{VERSION}-detector.json": (
        "detector.json", 2326, "c9367acbefd8df6a6e526e4f84f920e1eb0a0f3c091436474533131bb85a4429"
    ),
    f"kiyo-{VERSION}-restorer.safetensors": (
        "restorer.safetensors", 39085084, "c7480977eb35246682fe6256959f7afe79f6db1f17ae8f6c2c16488a5fc6f766"
    ),
    f"kiyo-{VERSION}-detector-b32-fp16.onnx": (
        ONNX, 65103632, "a9371a3a7899cc22e6be127cb1fb7b61e7cc79e2eeb5523a3eb13afbd2c107e0"
    ),
}


def default_dir() -> Path:
    if os.environ.get("KUKKIRI_WEIGHTS"):
        return Path(os.environ["KUKKIRI_WEIGHTS"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "kukkiri" / "weights" / VERSION


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(dest: Path) -> list[str]:
    bad = []
    for name, (local, size, sha) in FILES.items():
        p = Path(dest) / local
        if not p.is_file() or p.stat().st_size != size or _sha256(p) != sha:
            bad.append(name)
    return bad


def download(dest: Path, base_url: str | None = None, log=print) -> None:
    base = (base_url or BASE_URL or "").rstrip("/")
    if not base:
        raise RuntimeError("no download location; pass it with --base-url")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    todo = verify(dest)
    for name in todo:
        local, size, sha = FILES[name]
        if local == ONNX:
            _drop_stale_derived(dest)
        part = dest / f"{local}.part"
        part.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {name} ({size / 1e6:.0f} MB)")
        h, got = hashlib.sha256(), 0
        with urllib.request.urlopen(f"{base}/{name}") as r, open(part, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
                h.update(block)
                got += len(block)
                print(f"\rdownload: {100 * got / size:5.1f}%", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if got != size or h.hexdigest() != sha:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: download does not match the release (size {got}, sha256 {h.hexdigest()})")
        part.replace(dest / local)
    if todo and base.endswith(f"/{VERSION}"):
        try:
            urllib.request.urlopen(f"{base[: -len(VERSION) - 1]}/config.json", timeout=10).close()
        except OSError:
            pass
    log(f"weights {VERSION} ready in {dest}")


def missing(weights_dir: Path) -> list[str]:
    d = Path(weights_dir)
    out = []
    release = (d / "detector.safetensors").is_file() and (d / "detector.json").is_file()
    if release and not (d / ONNX).is_file():
        out.append("detector")
    elif not ((d / "detector.pth").is_file() or release):
        out.append("detector")
    if not ((d / "primary.pth").is_file() or (d / "restorer.safetensors").is_file()):
        out.append("restorer")
    return out


STAMP = "detector.sha256"


def _drop_stale_derived(d: Path) -> None:
    src = d / "detector.safetensors"
    if not src.is_file():
        return
    digest = _sha256(src)
    stamp = d / STAMP
    if stamp.is_file() and stamp.read_text().strip() == digest:
        return
    import shutil

    (d / "detector.pth").unlink(missing_ok=True)
    for exp in d.glob("export_b*"):
        shutil.rmtree(exp, ignore_errors=True)
    stamp.write_text(digest + "\n")


def detector_checkpoint(weights_dir: Path) -> Path:
    d = Path(weights_dir)
    _drop_stale_derived(d)
    ckpt = d / "detector.pth"
    if not ckpt.is_file():
        import torch
        from safetensors.torch import load_file

        meta = json.loads((d / "detector.json").read_text())
        tmp = d / "detector.pth.part"
        torch.save({"model": load_file(str(d / "detector.safetensors")), **meta}, tmp)
        tmp.replace(ckpt)
    return ckpt


def restorer_weights(weights_dir: Path) -> Path:
    d = Path(weights_dir)
    return d / "primary.pth" if (d / "primary.pth").is_file() else d / "restorer.safetensors"
