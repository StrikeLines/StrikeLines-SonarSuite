"""Reusable, channel-oriented sidescan swath geometry contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from operator import index

import numpy as np

from sidescantools.contact_model import Channel


class GeometryUnavailable(ValueError):
    """Raised when geometry cannot provide a coordinate for an acoustic anchor."""


@dataclass(frozen=True, slots=True)
class GeometrySettings:
    """Every processing setting that can affect a derived contact coordinate."""

    vertical_beam_angle: float
    cable_out_m: float = 0.0
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    navigation_smoothing_version: str = "legacy-v1"
    geometry_algorithm_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "vertical_beam_angle",
            "cable_out_m",
            "x_offset_m",
            "y_offset_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))

        if not isinstance(self.navigation_smoothing_version, str) or not (
            self.navigation_smoothing_version.strip()
        ):
            raise ValueError("navigation_smoothing_version must not be blank")
        if (
            isinstance(self.geometry_algorithm_version, bool)
            or not isinstance(self.geometry_algorithm_version, int)
            or self.geometry_algorithm_version < 1
        ):
            raise ValueError("geometry_algorithm_version must be a positive integer")

    def to_json(self) -> str:
        """Serialize deterministically for persistence and stale detection."""

        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def settings_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SwathGeometry:
    """Prepared per-ping nadir/outer geometry for one sonar channel.

    Arrays remain aligned to original/global source ping indices. Invalid pings
    are retained in those arrays and excluded from the legacy-compatible bulk
    coordinate output.
    """

    channel: Channel
    sample_count: int
    valid_ping_mask: np.ndarray
    nadir_lon: np.ndarray
    nadir_lat: np.ndarray
    outer_lon: np.ndarray
    outer_lat: np.ndarray
    slant_range_m: np.ndarray
    ground_range_m: np.ndarray
    geometry_settings: GeometrySettings

    def __post_init__(self) -> None:
        try:
            channel = Channel(self.channel)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel must be port (0) or starboard (1)") from exc
        object.__setattr__(self, "channel", channel)

        try:
            sample_count = index(self.sample_count)
        except TypeError as exc:
            raise ValueError("sample_count must be an integer") from exc
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        object.__setattr__(self, "sample_count", sample_count)

        names = (
            "valid_ping_mask",
            "nadir_lon",
            "nadir_lat",
            "outer_lon",
            "outer_lat",
            "slant_range_m",
            "ground_range_m",
        )
        arrays: dict[str, np.ndarray] = {}
        for name in names:
            dtype = bool if name == "valid_ping_mask" else float
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            if array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional array")
            array.setflags(write=False)
            arrays[name] = array
            object.__setattr__(self, name, array)

        ping_count = len(arrays["valid_ping_mask"])
        if any(len(arrays[name]) != ping_count for name in names[1:]):
            raise ValueError("all geometry arrays must align to the same ping count")

        valid = arrays["valid_ping_mask"]
        coordinate_arrays = (
            arrays["nadir_lon"],
            arrays["nadir_lat"],
            arrays["outer_lon"],
            arrays["outer_lat"],
        )
        if any(not np.all(np.isfinite(array[valid])) for array in coordinate_arrays):
            raise ValueError("valid geometry rows must contain finite coordinates")
        if np.any(np.abs(arrays["nadir_lon"][valid]) > 180) or np.any(
            np.abs(arrays["outer_lon"][valid]) > 180
        ):
            raise ValueError("valid geometry longitude must be in [-180, 180]")
        if np.any(np.abs(arrays["nadir_lat"][valid]) > 90) or np.any(
            np.abs(arrays["outer_lat"][valid]) > 90
        ):
            raise ValueError("valid geometry latitude must be in [-90, 90]")

    @property
    def ping_count(self) -> int:
        return len(self.valid_ping_mask)

    def coordinate_for_fraction(
        self, global_ping_index: int, sample_fraction: float
    ) -> tuple[float, float]:
        """Return WGS 84 longitude/latitude in O(1) without a bulk matrix."""

        try:
            ping_index = index(global_ping_index)
        except TypeError as exc:
            raise GeometryUnavailable("global_ping_index must be an integer") from exc
        if not 0 <= ping_index < self.ping_count:
            raise GeometryUnavailable("global_ping_index is outside the source file")
        if not self.valid_ping_mask[ping_index]:
            raise GeometryUnavailable("navigation is unavailable for this ping")
        if isinstance(sample_fraction, bool) or not isinstance(
            sample_fraction, (int, float)
        ):
            raise GeometryUnavailable("sample_fraction must be a number")
        fraction = float(sample_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise GeometryUnavailable("sample_fraction must be in [0.0, 1.0]")

        longitude = self.nadir_lon[ping_index] + fraction * (
            self.outer_lon[ping_index] - self.nadir_lon[ping_index]
        )
        latitude = self.nadir_lat[ping_index] + fraction * (
            self.outer_lat[ping_index] - self.nadir_lat[ping_index]
        )
        return float(longitude), float(latitude)

    def coordinate_for_sample(
        self,
        global_ping_index: int,
        sample_index: int,
        sample_count: int | None = None,
    ) -> tuple[float, float]:
        """Convert a near-to-far sample to a fraction and perform one lookup."""

        count = self.sample_count if sample_count is None else sample_count
        try:
            count = index(count)
            sample_index = index(sample_index)
        except TypeError as exc:
            raise GeometryUnavailable("sample index and count must be integers") from exc
        if count < 2:
            raise GeometryUnavailable("sample_count must be at least 2")
        if not 0 <= sample_index < count:
            raise GeometryUnavailable("sample_index is outside the channel")
        return self.coordinate_for_fraction(
            global_ping_index, sample_index / (count - 1)
        )

    def coordinates_for_all_samples(
        self, sample_count: int | None = None
    ) -> np.ndarray:
        """Return valid-ping coordinates in legacy ping-major/sample-major order."""

        count = self.sample_count if sample_count is None else sample_count
        try:
            count = index(count)
        except TypeError as exc:
            raise GeometryUnavailable("sample_count must be an integer") from exc
        if count < 2:
            raise GeometryUnavailable("sample_count must be at least 2")

        valid = self.valid_ping_mask
        lon = np.linspace(
            self.nadir_lon[valid], self.outer_lon[valid], count, axis=1
        )
        lat = np.linspace(
            self.nadir_lat[valid], self.outer_lat[valid], count, axis=1
        )
        return np.column_stack((lon.ravel(), lat.ravel()))
