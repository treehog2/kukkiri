
from __future__ import annotations

import torch


def trt_major(trt) -> int:
    return int(trt.__version__.split(".")[0])


def gpu_tag() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def adopt_untagged(path, tagged: str) -> None:
    from pathlib import Path

    path = Path(path)
    old = path.with_name(path.name.replace(f"_{tagged}", ""))
    if not path.exists() and old != path and old.exists():
        try:
            old.replace(path)
        except OSError:
            pass


def strongly_typed(trt) -> bool:
    return not hasattr(trt.BuilderFlag, "FP16")


def network_flags(trt) -> int:
    if strongly_typed(trt):
        return 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    return 1 << int(flag) if flag is not None else 0


def configure_fp16(trt, config) -> None:
    if not strongly_typed(trt):
        config.set_flag(trt.BuilderFlag.FP16)


def io_buffers(trt, engine, device) -> dict[str, torch.Tensor]:
    out = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        dt = torch.float16 if trt.nptype(engine.get_tensor_dtype(n)).__name__ == "float16" else torch.float32
        out[n] = torch.empty(tuple(engine.get_tensor_shape(n)), dtype=dt, device=device)
    return out
