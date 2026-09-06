from pathlib import Path

import numpy as np
import pytest

from sidescantools.readers import (
    JSFReader,
    SonarDataset,
    SonarReaderRegistry,
    UnsupportedSonarFormatError,
    XTFReader,
    is_supported_sonar_path,
    register_sonar_reader,
    sonar_file_dialog_filter,
    sonar_reader_registry,
    supported_sonar_suffixes,
    unregister_sonar_reader,
)
from sidescantools.sidescan_file import SidescanFile


def _dataset(path: Path, *, format_id: str = "synthetic") -> SonarDataset:
    ping_count = 3
    channel_count = 2
    data = np.arange(channel_count * ping_count * 4).reshape(
        channel_count, ping_count, 4
    )
    per_ping = np.arange(ping_count, dtype=float)
    return SonarDataset(
        filepath=path,
        format_id=format_id,
        data=data,
        ping_x_axis=np.linspace(0, 12, 4),
        timestamp=[None] * ping_count,
        sound_velocity=np.full(ping_count, 1500.0),
        starting_depth=np.zeros(ping_count),
        gain_adc=np.zeros(ping_count),
        longitude=-80.0 + per_ping / 1000,
        latitude=30.0 + per_ping / 1000,
        depth=np.full(ping_count, 5.0),
        packet_no=np.arange(100, 100 + ping_count),
        sensor_heading=np.full(ping_count, 90.0),
        sensor_pitch=np.zeros(ping_count),
        sensor_primary_altitude=np.full(ping_count, 10.0),
        sensor_aux_altitude=np.full(ping_count, np.nan),
        sensor_roll=np.zeros(ping_count),
        sensor_speed=np.full(ping_count, 2.0),
        seconds_per_ping=np.full((channel_count, ping_count), 0.01),
        slant_range=np.full((channel_count, ping_count), 12.0),
        layback_m=np.full(ping_count, np.nan),
        cable_out_m=np.full(ping_count, np.nan),
    )


class _SyntheticReader:
    format_id = "synthetic"
    suffixes = ("syn", ".sonar")

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        dataset = _dataset(filepath)
        dataset.choose_subsys = choose_subsys
        return dataset


def test_builtin_readers_are_registered_as_independent_adapters():
    assert isinstance(sonar_reader_registry.reader_for("line.JSF"), JSFReader)
    assert isinstance(sonar_reader_registry.reader_for("line.xtf"), XTFReader)
    assert supported_sonar_suffixes() == (".jsf", ".xtf")


def test_registry_dispatches_case_insensitively_and_normalizes_suffixes(tmp_path):
    registry = SonarReaderRegistry()
    reader = _SyntheticReader()
    registry.register(reader)

    result = registry.read(tmp_path / "line.SYN", choose_subsys=2)

    assert registry.supported_suffixes() == (".syn", ".sonar")
    assert result.format_id == "synthetic"
    assert result.choose_subsys == 2
    assert result.data.shape == (2, 3, 4)


def test_registry_rejects_duplicate_extensions_and_unknown_files():
    registry = SonarReaderRegistry()
    registry.register(_SyntheticReader())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_SyntheticReader())
    with pytest.raises(UnsupportedSonarFormatError, match="not supported"):
        registry.reader_for("line.unknown")


def test_dataset_contract_rejects_misaligned_metadata(tmp_path):
    values = vars(_dataset(tmp_path / "line.syn")).copy()
    for derived in ("num_ch", "num_ping", "ping_len"):
        values.pop(derived)
    values["longitude"] = np.zeros(2)

    with pytest.raises(ValueError, match="longitude length"):
        SonarDataset(**values)


def test_global_registration_updates_facade_discovery_and_dialog_filter(tmp_path):
    reader = _SyntheticReader()
    register_sonar_reader(reader)
    try:
        path = tmp_path / "line.SYN"
        path.touch()

        loaded = SidescanFile(path, choose_subsys=1)

        assert loaded.format_id == "synthetic"
        assert loaded.choose_subsys == 1
        assert is_supported_sonar_path(path)
        assert "*.syn" in sonar_file_dialog_filter()
    finally:
        unregister_sonar_reader(reader)

    assert not is_supported_sonar_path(tmp_path / "another.syn")
