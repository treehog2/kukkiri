from kukkiri import weights


def test_weights_follow_the_install_folder_unless_already_downloaded(tmp_path, monkeypatch):
    monkeypatch.delenv("KUKKIRI_WEIGHTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    monkeypatch.setattr(weights.sys, "platform", "linux")
    monkeypatch.setenv("KUKKIRI_WEIGHTS_ROOT", str(tmp_path / "install" / "models"))
    assert weights.default_dir("ltx-v1") == tmp_path / "install" / "models" / "ltx-v1"
    old = tmp_path / "cache" / "kukkiri" / "weights" / "ltx-v1"
    old.mkdir(parents=True)
    (old / "kiyo-ltx-v1-vae.safetensors").write_bytes(b"x")
    assert weights.default_dir("ltx-v1") == old
    monkeypatch.delenv("KUKKIRI_WEIGHTS_ROOT")
    assert weights.default_dir("v6") == tmp_path / "cache" / "kukkiri" / "weights" / "v6"

