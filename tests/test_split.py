from kukkiri.remaster import RemasterConfig
from kukkiri.split import _assign, _identity, job_dir


def test_parts_are_shared_out_by_length_and_kept_in_film_order():
    bounds = [[0, 100], [100, 400], [400, 450], [450, 700]]
    out = _assign(bounds, [0, 1])
    assert sorted(i for ids in out.values() for i in ids) == [0, 1, 2, 3]
    load = {g: sum(bounds[i][1] - bounds[i][0] for i in ids) for g, ids in out.items()}
    assert max(load.values()) - min(load.values()) <= 100
    assert all(ids == sorted(ids) for ids in out.values())


def test_one_gpu_takes_every_part():
    assert _assign([[0, 10], [10, 20]], [3]) == {3: [0, 1]}


def test_a_job_is_tied_to_its_source_and_settings(tmp_path):
    src, w = tmp_path / "film.mp4", tmp_path / "weights"
    src.write_bytes(b"x" * 10)
    a = _identity(src, RemasterConfig(), None, w)
    assert a == _identity(src, RemasterConfig(), None, w)
    assert a != _identity(src, RemasterConfig(codec="hevc"), None, w)
    assert a != _identity(src, RemasterConfig(), 0.5, w)
    assert a != _identity(src, RemasterConfig(), None, tmp_path / "other")
    assert a != _identity(src, RemasterConfig(), None, w, section=(60.0, 30.0))
    src.write_bytes(b"x" * 11)
    assert a != _identity(src, RemasterConfig(), None, w)
    assert job_dir(tmp_path / "out.mp4").name == "out.mp4.kukkiri-job"


def test_folder_scans_skip_a_split_job_folder(tmp_path):
    from kukkiri.cli import collect_videos

    (tmp_path / "film.mp4").write_bytes(b"x")
    job = tmp_path / "film.kukkiri-kiyo-v4.mp4.kukkiri-job"
    job.mkdir()
    (job / "source_readable.mp4").write_bytes(b"x")
    assert collect_videos([tmp_path]) == [tmp_path / "film.mp4"]


def test_a_saved_job_identity_matches_a_new_one(tmp_path):
    import json

    src = tmp_path / "film.mp4"
    src.write_bytes(b"x")
    a = _identity(src, RemasterConfig(), None, tmp_path, section=(240.0, 360.0))
    assert json.loads(json.dumps(a)) == _identity(src, RemasterConfig(), None, tmp_path, section=(240.0, 360.0))


def test_a_second_restore_of_the_same_output_is_refused(tmp_path):
    import shutil
    import subprocess
    import sys

    from kukkiri.split import _clear, _lock, _unlock, lock_path

    job = job_dir(tmp_path / "out.mp4")
    held = _lock(job)
    job.mkdir()
    probe = f"from kukkiri.split import _lock; _lock(__import__('pathlib').Path({str(job)!r}))"
    (job / "part000.h264").write_bytes(b"x")
    other = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert other.returncode != 0 and "already restoring to out.mp4" in other.stderr
    _clear(job)
    assert list(job.iterdir()) == []
    shutil.rmtree(job)
    again = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert again.returncode != 0 and "already restoring" in again.stderr
    _unlock(job, held)
    assert not lock_path(job).exists()
    _unlock(job, _lock(job))


def test_only_the_last_part_may_end_a_little_short():
    from kukkiri.split import TRUNCATED_EXIT, _short_end_ok

    meta = {"parts": [[0, 100], [100, 200], [200, 300]], "fps": 30.0}
    assert _short_end_ok(70, 100, 2, meta) and _short_end_ok(100, 100, 2, meta)
    assert not _short_end_ok(69, 100, 2, meta)
    assert not _short_end_ok(99, 100, 1, meta)
    assert TRUNCATED_EXIT not in (1, 3)


def test_a_lock_on_a_removed_lock_file_counts_as_busy(tmp_path, monkeypatch):
    import os

    import pytest

    import kukkiri.split as split

    if os.name == "nt":
        pytest.skip("Windows cannot remove a file another process has open")
    job = job_dir(tmp_path / "out.mp4")
    a = split._lock(job)
    real_open = open
    opened = {}

    def pausing_open(path, mode="r", *args, **kw):
        f = real_open(path, mode, *args, **kw)
        if str(path).endswith(".lock") and "b_opened" not in opened:
            opened["b_opened"] = True
            split._unlock(job, a)
            opened["c"] = split._lock(job)
        return f

    monkeypatch.setattr("builtins.open", pausing_open)
    with pytest.raises(split.JobBusy):
        split._lock(job)
    monkeypatch.setattr("builtins.open", real_open)
    opened["c"].close()


def test_an_output_finished_by_another_run_while_waiting_is_left_alone(tmp_path, monkeypatch):
    import pytest

    import kukkiri.split as split

    dst = tmp_path / "out.mp4"
    real_lock = split._lock

    def lock_after_other_finished(job):
        dst.write_bytes(b"finished by the other run")
        return real_lock(job)

    monkeypatch.setattr(split, "_lock", lock_after_other_finished)
    with pytest.raises(split.JobBusy):
        split.remaster_split(tmp_path / "src.mp4", dst, tmp_path, config=None, gpus=[0])
    assert dst.read_bytes() == b"finished by the other run"
    assert not split.job_dir(dst).exists()


def test_a_failed_split_removes_only_an_output_it_could_have_written(tmp_path, monkeypatch):
    import pytest

    import kukkiri.split as split

    dst = tmp_path / "out.mp4"

    def fail(*a, **k):
        dst.write_bytes(b"half written")
        raise RuntimeError("GPU failed")

    monkeypatch.setattr(split, "_split_once", fail)
    with pytest.raises(RuntimeError):
        split.remaster_split(tmp_path / "src.mp4", dst, tmp_path, config=None, gpus=[0])
    assert not dst.exists()
    dst.write_bytes(b"done earlier")
    with pytest.raises(RuntimeError):
        split.remaster_split(tmp_path / "src.mp4", dst, tmp_path, config=None, gpus=[0])
    assert dst.exists()



def test_a_worker_exits_when_its_coordinator_is_killed(tmp_path):
    import os
    import subprocess
    import sys
    import time

    import pytest

    if os.name == "nt":
        pytest.skip("checked on Windows by a real split restore")
    worker = "import threading, time; from kukkiri.split import _exit_with_parent; threading.Thread(target=_exit_with_parent, daemon=True).start(); time.sleep(60)"
    parent = f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', {worker!r}]); print(p.pid, flush=True); time.sleep(60)"
    coord = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True)
    pid = int(coord.stdout.readline())
    time.sleep(1)
    os.kill(pid, 0)
    coord.kill()
    coord.wait()
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        with open(f"/proc/{pid}/stat") as f:
            assert f.read().split()[2] == "Z", "the worker outlived its coordinator"
    except FileNotFoundError:
        pass


def test_a_worker_whose_coordinator_died_before_it_looked_exits(tmp_path):
    import os
    import subprocess
    import sys

    import pytest

    if os.name == "nt":
        pytest.skip("checked on Windows by a real split restore")
    worker = "from kukkiri.split import _exit_with_parent; _exit_with_parent()"
    r = subprocess.run([sys.executable, "-c", worker], env={**os.environ, "KUKKIRI_COORDINATOR_PID": "999999999"},
                       timeout=20)
    assert r.returncode == 1

