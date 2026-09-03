from pathlib import Path

import pytest

from sidescantools.contact_picker_app import build_parser, main


def test_launcher_defaults_are_safe_for_large_jsf():
    arguments = build_parser().parse_args(["survey.jsf"])

    assert arguments.chunk_size == 256
    assert arguments.downsampling_factor == 32
    assert arguments.vertical_beam_angle == 60.0
    assert arguments.viewer == "auto"


def test_launcher_rejects_missing_input(tmp_path: Path):
    with pytest.raises(SystemExit, match="2"):
        main([str(tmp_path / "missing.jsf")])


def test_sonar_file_is_optional_at_the_argparse_level():
    arguments = build_parser().parse_args([])

    assert arguments.sonar_file is None


def test_launcher_rejects_missing_input_for_napari_viewer():
    # Napari has no "prompt for a file on launch" path, unlike Qt -- an
    # explicit sonar_file is still required for it.
    with pytest.raises(SystemExit, match="2"):
        main(["--viewer", "napari"])


def test_launcher_allows_missing_input_for_qt_viewer(monkeypatch):
    # The desktop shortcut launches with no arguments at all; the Qt viewer
    # is expected to open an idle workspace instead of erroring here.
    calls = {}

    def fake_run_qt_contact_picker(sonar_file, **kwargs):
        calls["sonar_file"] = sonar_file
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "sidescantools.qt_contact_picker_ui.run_qt_contact_picker",
        fake_run_qt_contact_picker,
    )

    main(["--viewer", "qt"])

    assert calls["sonar_file"] is None
    # Neither can be defaulted from a file that isn't known yet -- must stay
    # None so run_qt_contact_picker resolves both after the user opens a file.
    assert calls["kwargs"]["work_dir"] is None
    assert calls["kwargs"]["contacts_db_path"] is None


def test_launcher_without_arguments_selects_idle_qt_workspace(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sidescantools.qt_contact_picker_ui.run_qt_contact_picker",
        lambda sonar_file, **kwargs: calls.append(sonar_file),
    )

    main([])

    assert calls == [None]
