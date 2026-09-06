"""Normalized sonar dataset contract and reader registry.

File-format readers stop at this boundary. Everything downstream in
SonarSuite consumes :class:`SonarDataset`, so a new input format does not need
to know about the waterfall, bottom tracker, contact picker, or GeoTIFF code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class UnsupportedSonarFormatError(NotImplementedError):
    """Raised when no registered reader owns a source file's extension."""


@dataclass
class SonarDataset:
    """Format-neutral representation used by SonarSuite's processing core.

    Readers normalize navigation to WGS84 decimal degrees, measurements to SI
    units, headings/attitude to degrees, and samples to
    ``[channel, ping, sample]`` order. Port is channel 0, stored outer-range to
    nadir; starboard is channel 1, stored nadir to outer-range. Optional
    per-ping metadata uses ``NaN`` where the processing field does not have an
    established legacy zero sentinel.
    """

    filepath: Path
    format_id: str
    data: np.ndarray
    ping_x_axis: np.ndarray
    timestamp: list[Any]
    sound_velocity: np.ndarray
    starting_depth: np.ndarray
    gain_adc: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    depth: np.ndarray
    packet_no: np.ndarray
    sensor_heading: np.ndarray
    sensor_pitch: np.ndarray
    sensor_primary_altitude: np.ndarray
    sensor_aux_altitude: np.ndarray
    sensor_roll: np.ndarray
    sensor_speed: np.ndarray
    seconds_per_ping: np.ndarray
    slant_range: np.ndarray
    layback_m: np.ndarray
    cable_out_m: np.ndarray
    choose_subsys: int = 0
    subsys_num: int = 1
    subsys_names: list[Any] = field(default_factory=list)
    coord_units: int = 2
    # Existing XTF bottom-line sidecars store pings in reverse order. Keeping
    # this detail on the dataset prevents it leaking into core processing as
    # an extension check.
    bottom_line_storage_reversed: bool = False
    reader_metadata: dict[str, Any] = field(default_factory=dict)
    num_ch: int = field(init=False)
    num_ping: int = field(init=False)
    ping_len: int = field(init=False)

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.format_id = str(self.format_id).strip().casefold()
        if not self.format_id:
            raise ValueError("format_id must not be blank")

        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError("sonar data must have [channel, ping, sample] dimensions")
        self.num_ch, self.num_ping, self.ping_len = self.data.shape
        if self.num_ch < 2:
            raise ValueError("SonarSuite currently requires port and starboard channels")
        if self.num_ping < 1 or self.ping_len < 1:
            raise ValueError("sonar data must contain at least one ping and one sample")

        self.ping_x_axis = np.asarray(self.ping_x_axis, dtype=float)
        if self.ping_x_axis.shape != (self.ping_len,):
            raise ValueError(
                "ping_x_axis length must match the number of samples per ping"
            )

        if len(self.timestamp) != self.num_ping:
            raise ValueError("timestamp length must match the sonar ping count")
        self.timestamp = list(self.timestamp)

        per_ping_fields = (
            "sound_velocity",
            "starting_depth",
            "gain_adc",
            "longitude",
            "latitude",
            "depth",
            "packet_no",
            "sensor_heading",
            "sensor_pitch",
            "sensor_primary_altitude",
            "sensor_aux_altitude",
            "sensor_roll",
            "sensor_speed",
            "layback_m",
            "cable_out_m",
        )
        for name in per_ping_fields:
            values = np.asarray(getattr(self, name))
            if values.shape != (self.num_ping,):
                raise ValueError(f"{name} length must match the sonar ping count")
            setattr(self, name, values)

        for name in ("seconds_per_ping", "slant_range"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (self.num_ch, self.num_ping):
                raise ValueError(
                    f"{name} must have [channel, ping] dimensions matching sonar data"
                )
            setattr(self, name, values)


@runtime_checkable
class SonarReader(Protocol):
    """Interface implemented by one adapter for each sonar file format."""

    format_id: str
    suffixes: tuple[str, ...]

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        """Parse and normalize ``filepath`` into a :class:`SonarDataset`."""


class SonarReaderRegistry:
    """Maps file extensions to independent format reader adapters."""

    def __init__(self) -> None:
        self._readers_by_suffix: dict[str, SonarReader] = {}

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        normalized = str(suffix).strip().casefold()
        if normalized and not normalized.startswith("."):
            normalized = f".{normalized}"
        if len(normalized) < 2:
            raise ValueError("reader suffixes must include a file extension")
        return normalized

    def register(self, reader: SonarReader, *, replace: bool = False) -> None:
        if not isinstance(reader, SonarReader):
            raise TypeError("reader must implement the SonarReader protocol")
        format_id = str(reader.format_id).strip().casefold()
        if not format_id:
            raise ValueError("reader format_id must not be blank")
        suffixes = tuple(self._normalize_suffix(value) for value in reader.suffixes)
        if not suffixes:
            raise ValueError("reader must advertise at least one file extension")
        for suffix in suffixes:
            owner = self._readers_by_suffix.get(suffix)
            if owner is not None and owner is not reader and not replace:
                raise ValueError(
                    f"extension {suffix} is already registered by {owner.format_id}"
                )
        for suffix in suffixes:
            self._readers_by_suffix[suffix] = reader

    def unregister(self, reader_or_format_id: SonarReader | str) -> None:
        format_id = (
            reader_or_format_id.format_id
            if not isinstance(reader_or_format_id, str)
            else reader_or_format_id
        )
        normalized = str(format_id).strip().casefold()
        self._readers_by_suffix = {
            suffix: reader
            for suffix, reader in self._readers_by_suffix.items()
            if str(reader.format_id).strip().casefold() != normalized
        }

    def reader_for(self, filepath: str | Path) -> SonarReader:
        path = Path(filepath)
        reader = self._readers_by_suffix.get(path.suffix.casefold())
        if reader is None:
            extensions = ", ".join(self.supported_suffixes()) or "none"
            raise UnsupportedSonarFormatError(
                f"File type {path.suffix.casefold() or '<none>'} is not supported. "
                f"Supported extensions: {extensions}."
            )
        return reader

    def supports(self, filepath: str | Path) -> bool:
        return Path(filepath).suffix.casefold() in self._readers_by_suffix

    def supported_suffixes(self) -> tuple[str, ...]:
        return tuple(self._readers_by_suffix)

    def format_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(reader.format_id).strip().casefold()
                for reader in self._readers_by_suffix.values()
            )
        )

    def read(self, filepath: str | Path, *, choose_subsys: int = 0) -> SonarDataset:
        path = Path(filepath)
        reader = self.reader_for(path)
        dataset = reader.read(path, choose_subsys=choose_subsys)
        if not isinstance(dataset, SonarDataset):
            raise TypeError("sonar readers must return a SonarDataset")
        if dataset.format_id != str(reader.format_id).strip().casefold():
            raise ValueError(
                "reader returned a dataset with a different format_id: "
                f"{dataset.format_id!r}"
            )
        return dataset
