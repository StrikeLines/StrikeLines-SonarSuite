import json

import pytest

from sidescantools.gain_settings import (
    SCHEMA_VERSION,
    SonarGainSettings,
    gain_settings_path,
    load_gain_settings,
    portable_egn_table_path,
    resolve_egn_table_path,
    save_gain_settings,
)


def _settings(**changes):
    values = {
        "source_file": "line.jsf",
        "overall_gain_db": -7.0,
        "tvg_spreading_db_per_decade": 18.0,
        "tvg_absorption_db_per_m": 0.12,
        "auto_tvg_brightness_target_percent": 30,
        "auto_tvg_active": True,
        "auto_tvg_gain_db": (-1.25, 0.0, 2.5, 1.0),
        "speed_correction_px_per_ping": 2.75,
        "processing_mode": "egn",
        "egn_table_path": "tables/survey.npz",
        "destripe_active": True,
        "slant_range_correction_active": True,
    }
    values.update(changes)
    return SonarGainSettings(**values)


def test_gain_settings_round_trip_as_versioned_json(tmp_path):
    sonar_path = tmp_path / "line.jsf"
    settings = _settings()

    output_path = save_gain_settings(sonar_path, settings)

    assert output_path == tmp_path / "line.jsf.tvg_gain.cfg"
    assert load_gain_settings(sonar_path) == settings
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["processing"]["mode"] == "egn"
    assert raw["processing"]["destripe_active"] is True
    assert raw["processing"]["slant_range_correction_active"] is True
    assert raw["display"]["auto_tvg_gain_db"] == [-1.25, 0.0, 2.5, 1.0]
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_gain_settings_returns_none(tmp_path):
    assert load_gain_settings(tmp_path / "line.xtf") is None


def test_older_sidecar_without_optional_processing_fields_loads_disabled(tmp_path):
    sonar_path = tmp_path / "line.jsf"
    path = gain_settings_path(sonar_path)
    values = _settings(destripe_active=False).to_dict()
    del values["processing"]["destripe_active"]
    del values["processing"]["slant_range_correction_active"]
    path.write_text(json.dumps(values), encoding="utf-8")

    loaded = load_gain_settings(sonar_path)

    assert loaded.destripe_active is False
    assert loaded.slant_range_correction_active is False


def test_gain_settings_cannot_be_attached_to_a_different_sonar_file(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        save_gain_settings(tmp_path / "other.jsf", _settings())


def test_gain_settings_reject_invalid_batch_inputs():
    with pytest.raises(ValueError, match="raw.*egn"):
        _settings(processing_mode="bac")
    with pytest.raises(ValueError, match="between 1 and 100"):
        _settings(auto_tvg_brightness_target_percent=0)
    with pytest.raises(ValueError, match="finite"):
        _settings(overall_gain_db=float("nan"))


def test_egn_paths_are_portable_and_resolve_from_the_sonar_directory(tmp_path):
    survey_dir = tmp_path / "survey"
    table_dir = tmp_path / "tables"
    survey_dir.mkdir()
    table_dir.mkdir()
    sonar_path = survey_dir / "line.xtf"
    table_path = table_dir / "egn.npz"

    stored_path = portable_egn_table_path(table_path, sonar_path)
    settings = _settings(
        source_file="line.xtf",
        egn_table_path=stored_path,
    )

    assert not gain_settings_path(sonar_path).exists()
    assert resolve_egn_table_path(settings, sonar_path) == table_path.resolve()
