from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

VERSION = "v4"
BASE = "https://huggingface.co/treehog/kukkiri/resolve/main"
BASE_URL: str | None = f"{BASE}/{VERSION}"
RELEASES = {
    "v4": ("seg_medium", {
        "detector.safetensors": ("detector.safetensors", 141176868, "43a37ca65c3d6a6d8cca9c2feb8e5621b908bb085a718208f790a62ccb821327"),
        "detector.json": ("detector.json", 2326, "c9367acbefd8df6a6e526e4f84f920e1eb0a0f3c091436474533131bb85a4429"),
        "restorer.safetensors": ("restorer.safetensors", 39085084, "c7480977eb35246682fe6256959f7afe79f6db1f17ae8f6c2c16488a5fc6f766"),
        "detector-b32-fp16.onnx": ("export_b32_fp16/rfdetr-seg-medium.h16.onnx", 65103632, "a9371a3a7899cc22e6be127cb1fb7b61e7cc79e2eeb5523a3eb13afbd2c107e0"),
    }),
    "ltx-v1": ("seg_medium", {
        "detector.safetensors": ("detector.safetensors", 141176868, "43a37ca65c3d6a6d8cca9c2feb8e5621b908bb085a718208f790a62ccb821327"),
        "detector.json": ("detector.json", 2326, "c9367acbefd8df6a6e526e4f84f920e1eb0a0f3c091436474533131bb85a4429"),
        "restorer.safetensors": ("restorer.safetensors", 39085084, "c7480977eb35246682fe6256959f7afe79f6db1f17ae8f6c2c16488a5fc6f766"),
        "transformer.safetensors": ("ltx-transformer.safetensors", 13364075432, "9712079e1b0ee20a9a47efbd0278cc4fb0eb7b88f9571e9d4d9fdfec70ccc94e"),
        "vae.safetensors": ("ltx-vae.safetensors", 1452269922, "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d"),
        "prompt.safetensors": ("ltx-prompt.safetensors", 8388704, "46f8652e075d12b501f6a21bda80e373dd956aa0f1ce897ca0c82ff7b21af511"),
        "detector-b32-fp16.onnx": ("export_b32_fp16/rfdetr-seg-medium.h16.onnx", 65103632, "a9371a3a7899cc22e6be127cb1fb7b61e7cc79e2eeb5523a3eb13afbd2c107e0"),
    }),
    "v3": ("seg_small", {
        "detector.safetensors": ("detector.safetensors", 135624836, "fbdd397d840844568c3a566c27ecda2ea5ad820bd9668bf6a6d73e3ef6982b31"),
        "detector.json": ("detector.json", 2323, "4515f9d18d358b3069b46f4a8b47e2162e4d2d278a577b06010cc17ce643852e"),
        "restorer.safetensors": ("restorer.safetensors", 39085084, "c7480977eb35246682fe6256959f7afe79f6db1f17ae8f6c2c16488a5fc6f766"),
        "detector-b32-fp16.onnx": ("export_b32_fp16/rfdetr-seg-small.h16.onnx", 63087963, "d224477c56d52d4b71d598ed408d5cc7fb04cf915480115356c7f0037ca9268a"),
    }),
}
VR_VERSION = "vr-v1"
RELEASES[VR_VERSION] = ("seg_medium", {
    "detector.safetensors": ("detector.safetensors", 143142972, "e82394f9f8d0bf2acc412379e060c28da96cbcb40d4f31855ab77171ae20c769"),
    "detector.json": ("detector.json", 2326, "d1ab8ba5c3449f0f6317aca9e316d7e33cc012fd81ece2e0664a8741a560dcf9"),
    "detector-b32-fp16.onnx": ("export_b32_fp16/rfdetr-seg-medium.h16.onnx", 66250578, "e1e3bffbfa80190c651e75dd52b38b86a901891ee77ca23e044190f55d89a8fe"),
})
MODELS = [f"kiyo-{v}" for v in RELEASES if v != VR_VERSION]


def version_of(model: str | None) -> str:
    v = (model or f"kiyo-{VERSION}").removeprefix("kiyo-")
    if v not in RELEASES or v == VR_VERSION:
        raise ValueError(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")
    return v


def download_size(version: str = VERSION) -> str:
    n = sum(size for _, size, _ in RELEASES[version][1].values())
    return f"{n / 1e9:.0f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def is_ltx(version: str) -> bool:
    return version.startswith("ltx-")


def detector_size(version: str = VERSION) -> str:
    return RELEASES[version][0]


def files(version: str = VERSION) -> dict:
    return {f"kiyo-{version}-{k}": v for k, v in RELEASES[version][1].items()}


def onnx_name(version: str = VERSION) -> str:
    return RELEASES[version][1]["detector-b32-fp16.onnx"][0]


ONNX = onnx_name()
FILES = files()


def default_dir(version: str = VERSION) -> Path:
    if os.environ.get("KUKKIRI_WEIGHTS"):
        return Path(os.environ["KUKKIRI_WEIGHTS"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "kukkiri" / "weights" / version


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(dest: Path, version: str = VERSION) -> list[str]:
    bad = []
    for name, (local, size, sha) in files(version).items():
        p = Path(dest) / local
        if not p.is_file() or p.stat().st_size != size or _sha256(p) != sha:
            bad.append(name)
    return bad


def download(dest: Path, base_url: str | None = None, log=print, version: str = VERSION) -> None:
    base = (base_url or f"{BASE}/{version}").rstrip("/")
    if not base:
        raise RuntimeError("no download location; pass it with --base-url")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    todo = verify(dest, version)
    table = files(version)
    for name in todo:
        local, size, sha = table[name]
        if local == onnx_name(version):
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
    if todo and base.endswith(f"/{version}"):
        try:
            urllib.request.urlopen(f"{base[: -len(version) - 1]}/config.json", timeout=10).close()
        except OSError:
            pass
    log(f"weights kiyo-{version} ready in {dest}")


def missing(weights_dir: Path, version: str = VERSION) -> list[str]:
    d = Path(weights_dir)
    out = []
    release = (d / "detector.safetensors").is_file() and (d / "detector.json").is_file()
    if release and not (d / onnx_name(version)).is_file():
        out.append("detector")
    elif not ((d / "detector.pth").is_file() or release):
        out.append("detector")
    if not ((d / "primary.pth").is_file() or (d / "restorer.safetensors").is_file()):
        out.append("restorer")
    if is_ltx(version) and not all(p.is_file() for p in ltx_paths(d)):
        out.append("ltx")
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


def vr_dir(weights_dir: Path) -> Path:
    return Path(weights_dir) / "vr"


def vr_missing(weights_dir: Path) -> bool:
    d = vr_dir(weights_dir)
    return not all((d / local).is_file() for local, _, _ in RELEASES[VR_VERSION][1].values())


LTX_FILES = ("ltx-transformer.safetensors", "ltx-vae.safetensors", "ltx-prompt.safetensors")


def ltx_paths(weights_dir: Path) -> tuple[Path, Path, Path]:
    d = Path(weights_dir)
    return tuple(d / n for n in LTX_FILES)


def restorer_weights(weights_dir: Path) -> Path:
    d = Path(weights_dir)
    return d / "primary.pth" if (d / "primary.pth").is_file() else d / "restorer.safetensors"
