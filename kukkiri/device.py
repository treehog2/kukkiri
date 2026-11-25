from __future__ import annotations

import contextlib
import os
import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Device(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def env_problem() -> str | None:
    env = os.environ.get("KUKKIRI_DEVICE")
    if env and env not in [d.value for d in Device]:
        return f"KUKKIRI_DEVICE is {env}: it must be one of {', '.join(d.value for d in Device)}, or unset"
    return None


def pick() -> Device:
    env = os.environ.get("KUKKIRI_DEVICE")
    if env:
        return Device(env)
    import torch

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        return Device.MPS
    return Device.CUDA


_lock = threading.RLock()


def gpu_lock():
    return contextlib.nullcontext() if pick() is Device.CUDA else _lock


def cuda_stream(**kw):
    import torch

    on = pick() is Device.CUDA and torch.cuda.is_available()
    return torch.cuda.stream(torch.cuda.Stream(**kw)) if on else contextlib.nullcontext()


def on_stream(stream):
    import torch

    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def cuda_record(t: torch.Tensor) -> None:
    if t.is_cuda:
        import torch

        t.record_stream(torch.cuda.current_stream())
