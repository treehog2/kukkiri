import os
import types

import pytest

torch = pytest.importorskip("torch")

from kukkiri import cli, remaster


@pytest.mark.parametrize("gpu, ram, staged", [(22.5, 15.9, True), (22.5, 32, False), (48, 15.9, False), (12, 64, True), (22.5, None, False)])
def test_staged_when_ram_is_short(monkeypatch, gpu, ram, staged):
    monkeypatch.delenv("KUKKIRI_LTX_STAGED", raising=False)
    monkeypatch.setenv("KUKKIRI_FRAME_BUDGET_GB", "1")
    monkeypatch.setattr(remaster, "_windows_ram_gb", lambda: ram)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=gpu * 2**30))
    assert remaster._ltx_staged() is staged


def test_out_of_ram_is_told_from_out_of_gpu_memory(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2**30, 24 * 2**30))
    assert cli._out_of_ram(torch.OutOfMemoryError("x"))
    assert cli._out_of_ram(RuntimeError("Returning 2 from cuMemHostAlloc"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2**28, 24 * 2**30))
    assert not cli._out_of_ram(torch.OutOfMemoryError("x"))
    assert not cli._out_of_ram(RuntimeError("other"))


def test_not_on_other_systems():
    if os.name != "nt":
        assert remaster._windows_ram_gb() is None
        assert not cli._out_of_ram(RuntimeError("cuMemHostAlloc"))
