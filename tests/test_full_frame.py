
import pytest

torch = pytest.importorskip("torch")

from kukkiri import cli, full_frame


class _Frames:
    def __init__(self, n, hw, batch=32):
        self.f = [torch.full((*hw, 3), i % 200, dtype=torch.uint8) for i in range(n)]
        self.batch, self.released = batch, []

    def iter_batches(self):
        for i in range(0, len(self.f), self.batch):
            yield self.f[i : i + self.batch]

    def release(self, i):
        self.released.append(i)


class _Ltx:

    def __init__(self):
        self.shapes = []

    def restore_canvas(self, x):
        self.shapes.append(tuple(x.shape))
        return x.numpy()


@pytest.mark.parametrize("n", [1, 12, 13, 95, 96, 97, 108, 109, 180, 181, 300])
def test_every_frame_once_in_order_and_unchanged(n):
    frames, ltx, out = _Frames(n, (30, 50)), _Ltx(), []
    assert full_frame.restore(frames, ltx, lambda i, f: out.append((i, f))) == n
    assert [i for i, _ in out] == list(range(n)) == frames.released
    for i, f in out:
        assert torch.equal(f, frames.f[i])
    assert all(t <= 96 and (h, w) == (32, 64) for t, h, w, _ in ltx.shapes)


def test_too_little_gpu_memory_says_so():
    class Full:
        def restore_canvas(self, x):
            raise torch.OutOfMemoryError("x")

    with pytest.raises(MemoryError, match="50x30"):
        full_frame.restore(_Frames(3, (30, 50)), Full(), lambda i, f: None)


@pytest.mark.parametrize("extra", [["--model", "kiyo-v4"], ["--model", "kiyo-ltx-v1", "--vr", "sbs"]])
def test_needs_kiyo_ltx_and_no_vr(monkeypatch, capsys, tmp_path, extra):
    monkeypatch.setattr(cli, "_gpu_problem", lambda: None)
    monkeypatch.setattr(cli, "pick", lambda: cli.Device.CUDA)
    import types

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=96 * 2**30))
    (tmp_path / "a.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        cli.main(["restore", str(tmp_path / "a.mp4"), "--full-frame", *extra])
    assert e.value.code == 1 and "--full-frame needs" in capsys.readouterr().err
