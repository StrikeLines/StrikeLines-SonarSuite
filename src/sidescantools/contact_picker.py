"""Pure display-pixel conversion for the sidescan contact picker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from operator import index
from typing import Callable, Mapping

from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactDraft,
    ContactRecord,
    ContactThumbnail,
)
from sidescantools.contact_store import ContactStore
from sidescantools.swath_geometry import SwathGeometry


class InvalidContactPixel(ValueError):
    """Raised when a displayed pixel cannot identify a real sonar sample."""


def _integer(name: str, value: int) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise InvalidContactPixel(f"{name} must be an integer") from exc
    return result


def _positive_integer(name: str, value: int, *, minimum: int = 1) -> int:
    result = _integer(name, value)
    if result < minimum:
        raise InvalidContactPixel(f"{name} must be at least {minimum}")
    return result


def anchor_from_display_pixel(
    *,
    source_file_id: int,
    ping_number: int | None,
    chunk_index: int,
    local_ping_index: int,
    display_x: int,
    chunk_size: int,
    display_channel_width: int,
    source_ping_count: int,
    source_sample_count: int,
) -> ContactAnchor:
    """Convert one integer Napari waterfall pixel to an acoustic anchor.

    The visible layout contract is ``PORT outer..nadir | STARBOARD nadir..outer``.
    The returned sample fraction has one orientation for both channels:
    ``0.0`` is nadir and ``1.0`` is the outer edge.
    """

    chunk_index = _integer("chunk_index", chunk_index)
    local_ping_index = _integer("local_ping_index", local_ping_index)
    display_x = _integer("display_x", display_x)
    chunk_size = _positive_integer("chunk_size", chunk_size)
    display_channel_width = _positive_integer(
        "display_channel_width", display_channel_width, minimum=2
    )
    source_ping_count = _positive_integer("source_ping_count", source_ping_count)
    source_sample_count = _positive_integer(
        "source_sample_count", source_sample_count, minimum=2
    )

    if chunk_index < 0:
        raise InvalidContactPixel("chunk_index is outside the displayed array")
    if not 0 <= local_ping_index < chunk_size:
        raise InvalidContactPixel("local_ping_index is outside the displayed chunk")
    if not 0 <= display_x < 2 * display_channel_width:
        raise InvalidContactPixel("display_x is outside the displayed waterfall")

    global_ping_index = chunk_index * chunk_size + local_ping_index
    if global_ping_index >= source_ping_count:
        raise InvalidContactPixel("pixel belongs to last-chunk padding, not a real ping")

    if display_x < display_channel_width:
        channel = Channel.PORT
        normalized_display_sample = display_channel_width - 1 - display_x
    else:
        channel = Channel.STARBOARD
        normalized_display_sample = display_x - display_channel_width

    sample_fraction = normalized_display_sample / (display_channel_width - 1)
    source_sample_index = round(sample_fraction * (source_sample_count - 1))
    source_sample_index = min(max(source_sample_index, 0), source_sample_count - 1)

    return ContactAnchor(
        source_file_id=source_file_id,
        global_ping_index=global_ping_index,
        ping_number=ping_number,
        channel=channel,
        source_sample_index=source_sample_index,
        sample_fraction=sample_fraction,
        display_chunk=chunk_index,
        display_ping_index=local_ping_index,
        display_sample_index=display_x,
    )


def display_position_for_anchor(
    anchor: ContactAnchor,
    *,
    chunk_size: int,
    display_channel_width: int,
) -> tuple[int, int, int]:
    """Reconstruct a Napari marker position from an authoritative anchor."""

    chunk_size = _positive_integer("chunk_size", chunk_size)
    display_channel_width = _positive_integer(
        "display_channel_width", display_channel_width, minimum=2
    )

    chunk_index, local_ping_index = divmod(anchor.global_ping_index, chunk_size)
    normalized_sample = round(anchor.sample_fraction * (display_channel_width - 1))
    normalized_sample = min(max(normalized_sample, 0), display_channel_width - 1)

    if anchor.channel is Channel.PORT:
        display_x = display_channel_width - 1 - normalized_sample
    else:
        display_x = display_channel_width + normalized_sample

    return chunk_index, local_ping_index, display_x


def source_array_sample_for_anchor(
    anchor: ContactAnchor, *, source_sample_count: int
) -> int:
    """Map normalized near-to-far sample order back to ``SidescanFile.data``."""

    source_sample_count = _positive_integer(
        "source_sample_count", source_sample_count, minimum=2
    )
    if anchor.source_sample_index >= source_sample_count:
        raise InvalidContactPixel("source sample is outside the source channel")
    if anchor.channel is Channel.PORT:
        return source_sample_count - 1 - anchor.source_sample_index
    return anchor.source_sample_index


@dataclass(frozen=True, slots=True)
class PickContactResult:
    contact: ContactRecord
    thumbnail_warning: str | None = None


class ContactPickerService:
    """Validate, derive, and persist one contact without any Napari dependency."""

    def __init__(
        self,
        *,
        sidescan_file,
        preprocessor,
        source_file_id: int,
        geometry_profile_id: int,
        geometry_by_channel: Mapping[Channel | int, SwathGeometry],
        store: ContactStore,
        thumbnail_factory: Callable[[ContactAnchor], ContactThumbnail] | None = None,
        display_intensity_provider: Callable[[ContactAnchor], float] | None = None,
        display_pipeline: str | Callable[[], str] = "napari-waterfall-v1",
    ):
        self.sidescan_file = sidescan_file
        self.preprocessor = preprocessor
        self.source_file_id = source_file_id
        self.geometry_profile_id = geometry_profile_id
        self.geometry_by_channel = {
            Channel(channel): geometry for channel, geometry in geometry_by_channel.items()
        }
        self.store = store
        self.thumbnail_factory = thumbnail_factory
        self.display_intensity_provider = display_intensity_provider
        self.display_pipeline = display_pipeline

    def pick_display_pixel(
        self,
        *,
        chunk_index: int,
        local_ping_index: int,
        display_x: int,
        name: str | None = None,
        notes: str = "",
        classification: str | None = None,
    ) -> PickContactResult:
        """Immediately save a valid displayed return and return its record."""

        chunk_index = _integer("chunk_index", chunk_index)
        local_ping_index = _integer("local_ping_index", local_ping_index)
        display_x = _integer("display_x", display_x)
        global_ping = chunk_index * self.preprocessor.chunk_size + local_ping_index
        ping_number = None
        if 0 <= global_ping < self.sidescan_file.num_ping:
            raw_ping_number = self.sidescan_file.packet_no[global_ping]
            ping_number = int(raw_ping_number) if raw_ping_number is not None else None

        anchor = anchor_from_display_pixel(
            source_file_id=self.source_file_id,
            ping_number=ping_number,
            chunk_index=chunk_index,
            local_ping_index=local_ping_index,
            display_x=display_x,
            chunk_size=self.preprocessor.chunk_size,
            display_channel_width=self.preprocessor.ping_len,
            source_ping_count=self.sidescan_file.num_ping,
            source_sample_count=self.sidescan_file.ping_len,
        )
        coordinate = self.coordinate_for_anchor(anchor)
        source_data_sample = source_array_sample_for_anchor(
            anchor, source_sample_count=self.sidescan_file.ping_len
        )
        timestamp_iso, timestamp_basis = self._timestamp_for_ping(
            anchor.global_ping_index
        )
        draft = ContactDraft(
            anchor=anchor,
            coordinate=coordinate,
            name=name if name is not None else self.store.next_default_contact_name(),
            notes=notes,
            classification=classification,
            timestamp_iso=timestamp_iso,
            timestamp_basis=timestamp_basis,
            intensity_source=float(
                self.sidescan_file.data[
                    int(anchor.channel),
                    anchor.global_ping_index,
                    source_data_sample,
                ]
            ),
            intensity_display=self._display_intensity(anchor),
            intensity_pipeline=self._display_pipeline_description(),
        )

        thumbnail = None
        thumbnail_warning = None
        if self.thumbnail_factory is not None:
            try:
                thumbnail = self.thumbnail_factory(anchor)
            except Exception as exc:
                thumbnail_warning = f"thumbnail unavailable: {exc}"

        return PickContactResult(
            contact=self.store.create_contact(draft, thumbnail),
            thumbnail_warning=thumbnail_warning,
        )

    def coordinate_for_anchor(self, anchor: ContactAnchor) -> ContactCoordinate:
        """Derive a coordinate from the currently active geometry profile."""

        geometry = self.geometry_by_channel.get(anchor.channel)
        if geometry is None:
            raise InvalidContactPixel(
                f"geometry is not prepared for the {anchor.channel.label} channel"
            )
        if geometry.geometry_settings.settings_hash != self._active_settings_hash:
            raise InvalidContactPixel("prepared geometry does not match the active profile")

        longitude, latitude = geometry.coordinate_for_fraction(
            anchor.global_ping_index, anchor.sample_fraction
        )
        return ContactCoordinate(
            longitude=longitude,
            latitude=latitude,
            slant_range_m=self._fractional_range(
                geometry.slant_range_m[anchor.global_ping_index],
                anchor.sample_fraction,
            ),
            ground_range_m=self._fractional_range(
                geometry.ground_range_m[anchor.global_ping_index],
                anchor.sample_fraction,
            ),
            geometry_profile_id=self.geometry_profile_id,
        )

    def _display_intensity(self, anchor: ContactAnchor) -> float:
        if self.display_intensity_provider is not None:
            return float(self.display_intensity_provider(anchor))
        return float(
            self.preprocessor.napari_fullmat[
                anchor.display_chunk,
                anchor.display_ping_index,
                anchor.display_sample_index,
            ]
        )

    def _display_pipeline_description(self) -> str:
        description = (
            self.display_pipeline()
            if callable(self.display_pipeline)
            else self.display_pipeline
        )
        return str(description)

    @property
    def _active_settings_hash(self) -> str:
        hashes = {
            geometry.geometry_settings.settings_hash
            for geometry in self.geometry_by_channel.values()
        }
        if len(hashes) != 1:
            raise InvalidContactPixel("channel geometries use different settings")
        active_hash = next(iter(hashes))
        if active_hash != self.store.geometry_profile_hash(self.geometry_profile_id):
            raise InvalidContactPixel("prepared geometry does not match the active profile")
        return active_hash

    @staticmethod
    def _fractional_range(value, sample_fraction: float) -> float | None:
        try:
            result = float(value) * sample_fraction
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result >= 0 else None

    def _timestamp_for_ping(self, global_ping_index: int) -> tuple[str | None, str | None]:
        timestamps = getattr(self.sidescan_file, "timestamp", None)
        if timestamps is None:
            return None, None
        value = timestamps[global_ping_index]
        if value is None:
            return None, None
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.isoformat(), "unknown"
            basis = "utc" if value.utcoffset().total_seconds() == 0 else "explicit-offset"
            return value.isoformat(), basis
        return str(value), "unknown"
