import pytest

gui = pytest.importorskip("kukkiri.gui")


def test_only_minor_and_major_releases_prompt():
    assert gui.worth_updating("v0.7.0", "0.6.2")
    assert gui.worth_updating("v1.0.0", "0.9.4")
    assert not gui.worth_updating("v0.6.3", "0.6.2")
    assert not gui.worth_updating("v0.7.0", "0.7.0")
    assert not gui.worth_updating("v0.6.9", "0.7.0")
    assert not gui.worth_updating("v0.7.0", "0.0.0.dev0")


def test_a_warning_run_on_after_progress_is_its_own_line():
    line = "preparing:   0.0%   [2026-11-24 14:33:04] [WARNING] rf-detr - Using patch size 12 instead of 14"
    assert gui.split_progress(line) == ["preparing:   0.0%", "[2026-11-24 14:33:04] [WARNING] rf-detr - Using patch size 12 instead of 14"]
    assert gui.split_progress("ltx:  27.2%, 3.5 min left") == ["ltx:  27.2%, 3.5 min left"]
    assert gui.split_progress("done: out.mp4 (543 of 899 frames restored, 499 s)") == ["done: out.mp4 (543 of 899 frames restored, 499 s)"]


def test_known_library_noise_is_left_out():
    for line in ("nanobind: leaked 2 instances!", "- leaked function \"repr\"", "[WARN ][14:01:46] INVALID INDEX 900",
                 "[2026-11-24 14:33:04] [WARNING] rf-detr - Checkpoint has 3 classes but model is configured for 90.",
                 "[11/24/2026-14:40:00] [TRT] [W] Functionality provided through tensorrt.plugin module is experimental.",
                 "W1124 14:35:06.889000 87876 utils.py:1806] [1/0] Not enough SMs to use max_autotune_gemm mode"):
        assert gui._LOG_NOISE_RE.search(line), line
    for line in ("done: out.mp4 (543 of 899 frames restored, 499 s)", "cutting 1:30 + 00:30", "error: out of memory"):
        assert not gui._LOG_NOISE_RE.search(line), line
