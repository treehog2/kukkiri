import pytest

gui = pytest.importorskip("kukkiri.gui")


def test_only_minor_and_major_releases_prompt():
    assert gui.worth_updating("v0.7.0", "0.6.2")
    assert gui.worth_updating("v1.0.0", "0.9.4")
    assert not gui.worth_updating("v0.6.3", "0.6.2")
    assert not gui.worth_updating("v0.7.0", "0.7.0")
    assert not gui.worth_updating("v0.6.9", "0.7.0")
    assert not gui.worth_updating("v0.7.0", "0.0.0.dev0")
