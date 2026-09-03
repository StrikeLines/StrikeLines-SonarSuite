from pathlib import Path

from sidescantools import windows_app
from sidescantools.windows_app import startup_sonar_file


def test_windows_app_starts_empty_without_arguments():
    assert startup_sonar_file([]) is None


def test_windows_app_accepts_existing_jsf_or_xtf(tmp_path):
    jsf = tmp_path / "line.jsf"
    xtf = tmp_path / "line.XTF"
    jsf.touch()
    xtf.touch()

    assert startup_sonar_file([str(jsf)]) == jsf.resolve()
    assert startup_sonar_file([str(xtf)]) == xtf.resolve()


def test_windows_app_ignores_missing_or_unsupported_paths(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.touch()

    assert startup_sonar_file([str(tmp_path / "missing.jsf")]) is None
    assert startup_sonar_file([str(text_file)]) is None


def test_windows_app_smoke_test_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(windows_app, "_run_smoke_test", lambda: calls.append(True))

    assert windows_app.main(["--smoke-test"]) == 0
    assert calls == [True]
    assert not (tmp_path / "sidescantools-smoke-error.txt").exists()
