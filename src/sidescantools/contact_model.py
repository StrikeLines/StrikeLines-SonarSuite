"""Core contact-picker domain types and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
from uuid import uuid4


class ContactValidationError(ValueError):
    """Raised when a contact domain object violates its persistence contract."""


class Channel(IntEnum):
    """Sidescan channel using the source-file channel indices."""

    PORT = 0
    STARBOARD = 1

    @property
    def label(self) -> str:
        return "port" if self is Channel.PORT else "starboard"


class CoordinateStatus(str, Enum):
    VALID = "valid"
    STALE = "stale"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


def _require_integer(name: str, value: int, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContactValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ContactValidationError(f"{name} must be at least {minimum}")


def _require_optional_nonnegative_finite(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContactValidationError(f"{name} must be a number or None")
    if not math.isfinite(value) or value < 0:
        raise ContactValidationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ContactAnchor:
    """Immutable, resolution-independent acoustic location of a contact."""

    source_file_id: int
    global_ping_index: int
    ping_number: int | None
    channel: Channel
    source_sample_index: int
    sample_fraction: float
    display_chunk: int
    display_ping_index: int
    display_sample_index: int

    def __post_init__(self) -> None:
        _require_integer("source_file_id", self.source_file_id, minimum=1)
        _require_integer("global_ping_index", self.global_ping_index, minimum=0)
        if self.ping_number is not None:
            _require_integer("ping_number", self.ping_number)
        _require_integer("source_sample_index", self.source_sample_index, minimum=0)
        _require_integer("display_chunk", self.display_chunk, minimum=0)
        _require_integer("display_ping_index", self.display_ping_index, minimum=0)
        _require_integer("display_sample_index", self.display_sample_index, minimum=0)

        try:
            channel = Channel(self.channel)
        except (TypeError, ValueError) as exc:
            raise ContactValidationError("channel must be port (0) or starboard (1)") from exc
        object.__setattr__(self, "channel", channel)

        if isinstance(self.sample_fraction, bool) or not isinstance(
            self.sample_fraction, (int, float)
        ):
            raise ContactValidationError("sample_fraction must be a number")
        fraction = float(self.sample_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ContactValidationError("sample_fraction must be in [0.0, 1.0]")
        object.__setattr__(self, "sample_fraction", fraction)


@dataclass(frozen=True, slots=True)
class ContactCoordinate:
    """WGS 84 contact coordinate derived from a specific geometry profile."""

    longitude: float
    latitude: float
    slant_range_m: float | None
    ground_range_m: float | None
    geometry_profile_id: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.longitude, bool)
            or not isinstance(self.longitude, (int, float))
            or not math.isfinite(self.longitude)
        ):
            raise ContactValidationError("longitude must be finite")
        if (
            isinstance(self.latitude, bool)
            or not isinstance(self.latitude, (int, float))
            or not math.isfinite(self.latitude)
        ):
            raise ContactValidationError("latitude must be finite")
        if not -180.0 <= self.longitude <= 180.0:
            raise ContactValidationError("longitude must be in [-180, 180]")
        if not -90.0 <= self.latitude <= 90.0:
            raise ContactValidationError("latitude must be in [-90, 90]")
        _require_optional_nonnegative_finite("slant_range_m", self.slant_range_m)
        _require_optional_nonnegative_finite("ground_range_m", self.ground_range_m)
        _require_integer("geometry_profile_id", self.geometry_profile_id, minimum=1)

        object.__setattr__(self, "longitude", float(self.longitude))
        object.__setattr__(self, "latitude", float(self.latitude))
        if self.slant_range_m is not None:
            object.__setattr__(self, "slant_range_m", float(self.slant_range_m))
        if self.ground_range_m is not None:
            object.__setattr__(self, "ground_range_m", float(self.ground_range_m))


@dataclass(frozen=True, slots=True)
class ContactDraft:
    """Complete contact payload ready for one transactional database insert."""

    anchor: ContactAnchor
    coordinate: ContactCoordinate
    name: str
    notes: str = ""
    classification: str | None = None
    timestamp_iso: str | None = None
    timestamp_basis: str | None = None
    intensity_source: float | None = None
    intensity_display: float | None = None
    intensity_pipeline: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ContactValidationError("name must be text")
        if not isinstance(self.notes, str):
            raise ContactValidationError("notes must be text")
        if self.classification is not None and not isinstance(
            self.classification, str
        ):
            raise ContactValidationError("classification must be text or None")
        if not isinstance(self.uuid, str) or not self.uuid.strip():
            raise ContactValidationError("uuid must not be blank")
        for name in ("intensity_source", "intensity_display"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ContactValidationError(f"{name} must be finite or None")


@dataclass(frozen=True, slots=True)
class ContactThumbnail:
    image_bytes: bytes
    width_px: int
    height_px: int
    ping_radius: int
    sample_radius: int
    mime_type: str = "image/png"
    display_pipeline: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise ContactValidationError("image_bytes must not be empty")
        for name in ("width_px", "height_px"):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in ("ping_radius", "sample_radius"):
            _require_integer(name, getattr(self, name), minimum=0)
        if not isinstance(self.mime_type, str) or not self.mime_type.strip():
            raise ContactValidationError("mime_type must not be blank")


@dataclass(frozen=True, slots=True)
class ContactRecord:
    id: int
    draft: ContactDraft
    coordinate_status: CoordinateStatus
    coordinate_error: str | None
    source_display_path: str | None
    geometry_settings_hash: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _require_integer("id", self.id, minimum=1)
        try:
            status = CoordinateStatus(self.coordinate_status)
        except (TypeError, ValueError) as exc:
            raise ContactValidationError("invalid coordinate status") from exc
        object.__setattr__(self, "coordinate_status", status)
