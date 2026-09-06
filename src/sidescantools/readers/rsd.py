"""Garmin RSD input adapter.

Garmin RSD files use a compact, protobuf-like envelope around a sequence of
sonar records.  This module intentionally keeps the binary decoder separate
from the format-neutral :class:`~sidescantools.readers.base.SonarDataset`
adapter so framing and channel-selection behavior can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
import logging
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

import numpy as np

from sidescantools.readers.base import SonarDataset


_FILE_MAGIC = 0xD9264B7C
_RECORD_MAGIC = 0xB7E9DA86
_TRAILER_MAGIC = 0xF98EACBC
_RECORDS_OFFSET = 0x5000
_GARMIN_EPOCH = datetime(1989, 12, 31)
_MAX_FIELD_COUNT = 4096
_MAX_FIELD_LENGTH = 256 * 1024 * 1024


logger = logging.getLogger(__name__)


class RSDFormatError(ValueError):
    """Raised when an RSD file is truncated or has invalid framing."""


@dataclass(frozen=True)
class RSDChannelInfo:
    channel_id: int
    first_chunk_offset: int
    transducer_port: int | None = None
    frequency_mode: int | None = None
    frequency_start_hz: int | None = None
    frequency_end_hz: int | None = None
    capabilities: int | None = None


@dataclass(frozen=True)
class RSDHeader:
    version: int
    unit_id: int | None
    product: int | None
    recording_start: datetime | None
    channels: tuple[RSDChannelInfo, ...]


@dataclass(frozen=True)
class RSDRecord:
    offset: int
    channel_id: int
    sequence_count: int
    recording_time_ms: int
    samples: np.ndarray
    sample_count: int
    bottom_depth_m: float
    first_sample_depth_m: float
    last_sample_depth_m: float
    latitude: float
    longitude: float
    gain: int | None
    beam: int | None
    beam_info: dict[int, bytes] | None


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise RSDFormatError(f"Truncated RSD {context}")
    return value


def _read_varuint(stream: BinaryIO, *, allow_eof: bool = False) -> int | None:
    value = 0
    for shift in range(0, 35, 7):
        byte = stream.read(1)
        if not byte:
            if allow_eof and shift == 0:
                return None
            raise RSDFormatError("Truncated RSD variable-length integer")
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
    raise RSDFormatError("RSD variable-length integer exceeds 32 bits")


def _read_structure(
    stream: BinaryIO, *, allow_eof: bool = False
) -> dict[int, bytes] | None:
    field_count = _read_varuint(stream, allow_eof=allow_eof)
    if field_count is None:
        return None
    if field_count > _MAX_FIELD_COUNT:
        raise RSDFormatError(f"Unreasonable RSD field count: {field_count}")

    fields: dict[int, bytes] = {}
    for _ in range(field_count):
        key = _read_varuint(stream)
        assert key is not None
        field_number = key >> 3
        value_length = key & 0x07
        if value_length == 0:
            raise RSDFormatError("RSD field key has a zero value length")
        if value_length == 7:
            value_length = _read_varuint(stream)
            assert value_length is not None
        if value_length > _MAX_FIELD_LENGTH:
            raise RSDFormatError(f"Unreasonable RSD field length: {value_length}")
        fields[field_number] = _read_exact(
            stream, value_length, context=f"field {field_number}"
        )
    return fields


def _parse_structure(value: bytes) -> dict[int, bytes]:
    stream = BytesIO(value)
    fields = _read_structure(stream)
    assert fields is not None
    if stream.read(1):
        raise RSDFormatError("Unexpected bytes after nested RSD structure")
    return fields


def _parse_varray(value: bytes) -> list[bytes]:
    stream = BytesIO(value)
    count = _read_varuint(stream)
    assert count is not None
    items: list[bytes] = []
    for _ in range(count):
        size = _read_varuint(stream)
        assert size is not None
        items.append(_read_exact(stream, size, context="variable-array item"))
    if stream.read(1):
        raise RSDFormatError("Unexpected bytes after RSD variable array")
    return items


def _parse_structure_array(value: bytes) -> list[dict[int, bytes]]:
    """Parse Garmin's structure-array variant (items have no size prefix)."""

    stream = BytesIO(value)
    count = _read_varuint(stream)
    assert count is not None
    items: list[dict[int, bytes]] = []
    for _ in range(count):
        item = _read_structure(stream)
        assert item is not None
        items.append(item)
    if stream.read(1):
        raise RSDFormatError("Unexpected bytes after RSD structure array")
    return items


def _uint(value: bytes) -> int:
    return int.from_bytes(value, "little", signed=False)


def _varuint(value: bytes) -> int:
    stream = BytesIO(value)
    decoded = _read_varuint(stream)
    assert decoded is not None
    if stream.read(1):
        raise RSDFormatError("Unexpected bytes after RSD variable integer")
    return decoded


def _zigzag32(value: bytes) -> int:
    encoded = _varuint(value)
    return (encoded >> 1) ^ -(encoded & 1)


def _millimeters(value: bytes | None) -> float:
    if value is None:
        return float("nan")
    return _zigzag32(value) / 1000.0


def _map_units(value: bytes | None) -> float:
    if value is None:
        return float("nan")
    signed = int.from_bytes(value, "little", signed=True)
    return signed * 360.0 / (2**32)


def _first_varray_uint(value: bytes) -> int:
    items = _parse_varray(value)
    if not items:
        raise RSDFormatError("RSD channel has no data identifier")
    return _varuint(items[0])


def read_rsd_header(filepath: str | Path) -> RSDHeader:
    """Read the fixed-area RSD file header and channel directory."""

    path = Path(filepath)
    with path.open("rb") as stream:
        fields = _read_structure(stream)
    assert fields is not None
    if _uint(fields.get(0, b"")) != _FILE_MAGIC:
        raise RSDFormatError(f"{path.name} is not a Garmin RSD file")

    version = _uint(fields.get(1, b"\0\0"))
    file_info = _parse_structure(fields[5]) if 5 in fields else {}
    unit_id = _uint(file_info[1]) if 1 in file_info else None
    product = _uint(file_info[2]) if 2 in file_info else None
    gdate = _uint(file_info[3]) if 3 in file_info else None
    recording_start = None
    if gdate not in (None, 0, 0xFFFFFFFF):
        recording_start = _GARMIN_EPOCH + timedelta(seconds=gdate)

    channels: list[RSDChannelInfo] = []
    for channel in _parse_structure_array(fields.get(6, b"\0")):
        channel_id = _first_varray_uint(channel[0])
        first_chunk_offset = _uint(channel.get(1, b"\0"))
        transducer_port = frequency_mode = frequency_start = frequency_end = None
        capabilities = None
        dps_items = _parse_varray(channel[2]) if 2 in channel else []
        if dps_items:
            dps = _parse_structure(dps_items[0])
            transducer_port = _varuint(dps[0]) if 0 in dps else None
            if 1 in dps:
                frequency = _parse_structure(dps[1])
                frequency_mode = _varuint(frequency[0]) if 0 in frequency else None
                frequency_start = _uint(frequency[1]) if 1 in frequency else None
                frequency_end = _uint(frequency[2]) if 2 in frequency else None
            capabilities = _uint(dps[2]) if 2 in dps else None
        channels.append(
            RSDChannelInfo(
                channel_id=channel_id,
                first_chunk_offset=first_chunk_offset,
                transducer_port=transducer_port,
                frequency_mode=frequency_mode,
                frequency_start_hz=frequency_start,
                frequency_end_hz=frequency_end,
                capabilities=capabilities,
            )
        )

    declared_count = _uint(fields.get(2, b"\0"))
    if declared_count != len(channels):
        raise RSDFormatError(
            f"RSD channel directory declares {declared_count} channels, "
            f"but contains {len(channels)}"
        )
    return RSDHeader(version, unit_id, product, recording_start, tuple(channels))


def _parse_record_body(value: bytes) -> tuple[dict[int, bytes], np.ndarray]:
    stream = BytesIO(value)
    fields = _read_structure(stream)
    assert fields is not None
    sample_count = _uint(fields.get(7, b"\0"))
    sample_bytes = sample_count * 2
    samples = np.frombuffer(
        _read_exact(stream, sample_bytes, context="sonar samples"), dtype="<u2"
    ).copy()
    shade_available = bool(_uint(fields.get(8, b"\0")))
    trailing = stream.read()
    expected_shade = sample_count if shade_available else 0
    if len(trailing) not in (0, expected_shade):
        raise RSDFormatError(
            "RSD sonar body has an unexpected payload length "
            f"({len(trailing)} trailing bytes for {sample_count} samples)"
        )
    return fields, samples


def iter_rsd_records(filepath: str | Path) -> Iterator[RSDRecord]:
    """Yield decoded sonar records while reading the file sequentially."""

    path = Path(filepath)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(_RECORDS_OFFSET)
        while stream.tell() < file_size:
            offset = stream.tell()
            header = _read_structure(stream, allow_eof=True)
            if header is None:
                return
            if _uint(header.get(0, b"")) != _RECORD_MAGIC:
                raise RSDFormatError(f"Invalid RSD record magic at offset {offset}")
            _read_exact(stream, 4, context="record-header CRC")

            data_size = _uint(header.get(4, b"\0\0"))
            body = _read_exact(stream, data_size, context="record body")
            trailer = _read_exact(stream, 12, context="record trailer")
            trailer_magic, chunk_size, _trailer_crc = struct.unpack("<III", trailer)
            if trailer_magic != _TRAILER_MAGIC:
                raise RSDFormatError(f"Invalid RSD trailer magic at offset {offset}")
            actual_size = stream.tell() - offset
            if chunk_size not in (actual_size, actual_size - 12):
                raise RSDFormatError(
                    f"Invalid RSD chunk size {chunk_size} at offset {offset}; "
                    f"decoded {actual_size} bytes"
                )

            state = _parse_structure(header[1]) if 1 in header else {}
            state_ids = _parse_varray(state[1]) if 1 in state else []
            header_channel = _varuint(state_ids[0]) if state_ids else 0
            if not body:
                continue
            fields, samples = _parse_record_body(body)
            body_channel = _varuint(fields[0]) if 0 in fields else header_channel
            if header_channel and body_channel != header_channel:
                raise RSDFormatError(
                    f"RSD channel mismatch at offset {offset}: "
                    f"header {header_channel}, body {body_channel}"
                )
            yield RSDRecord(
                offset=offset,
                channel_id=body_channel,
                sequence_count=_uint(header.get(2, b"\0")),
                recording_time_ms=_uint(header.get(5, b"\0")),
                samples=samples,
                sample_count=len(samples),
                bottom_depth_m=_millimeters(fields.get(1)),
                first_sample_depth_m=_millimeters(fields.get(3)),
                last_sample_depth_m=_millimeters(fields.get(4)),
                gain=_uint(fields[5]) if 5 in fields else None,
                latitude=_map_units(fields.get(9)),
                longitude=_map_units(fields.get(10)),
                beam=_varuint(fields[12]) if 12 in fields else None,
                beam_info=_parse_structure(fields[13]) if 13 in fields else None,
            )


def _fill_samples(
    output: np.ndarray, records: list[RSDRecord], *, reverse: bool = False
) -> None:
    """Fill a preallocated channel, resampling only on sample-count changes."""

    width = output.shape[1]
    target = np.linspace(0.0, 1.0, width)
    for row, record in enumerate(records):
        if record.sample_count == width:
            samples = record.samples
        elif record.sample_count == 1:
            output[row].fill(record.samples[0])
            continue
        else:
            source = np.linspace(0.0, 1.0, record.sample_count)
            samples = np.rint(np.interp(target, source, record.samples)).astype(
                np.uint16
            )
        output[row] = samples[::-1] if reverse else samples


def _mean_finite(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.column_stack((first, second))
    count = np.sum(np.isfinite(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(
        total,
        count,
        out=np.full(len(values), np.nan, dtype=float),
        where=count > 0,
    )


def _track_heading_and_speed(
    latitude: np.ndarray, longitude: np.ndarray, time_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive course and speed from consecutive WGS84 fixes."""

    count = len(latitude)
    heading = np.zeros(count, dtype=float)
    speed = np.zeros(count, dtype=float)
    if count < 2:
        return heading, speed

    earth_radius_m = 6_371_008.8
    lat_radians = np.deg2rad(latitude)
    lon_radians = np.deg2rad(longitude)
    delta_lat = np.diff(lat_radians)
    delta_lon = np.diff(lon_radians)
    mean_lat = (lat_radians[:-1] + lat_radians[1:]) / 2.0
    north_m = delta_lat * earth_radius_m
    east_m = delta_lon * np.cos(mean_lat) * earth_radius_m
    course = np.mod(np.rad2deg(np.arctan2(east_m, north_m)), 360.0)
    distance_m = np.hypot(east_m, north_m)
    delta_time_s = np.diff(time_ms.astype(float)) / 1000.0
    segment_speed = np.divide(
        distance_m,
        delta_time_s,
        out=np.zeros_like(distance_m),
        where=delta_time_s > 0,
    )

    valid_course = distance_m > 1e-4
    for index in range(1, count):
        heading[index] = (
            course[index - 1] if valid_course[index - 1] else heading[index - 1]
        )
        speed[index] = segment_speed[index - 1]
    heading[0] = heading[1]
    speed[0] = speed[1]
    return heading, speed


class RSDReader:
    format_id = "rsd"
    suffixes = (".rsd",)

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        path = Path(filepath)
        if choose_subsys != 0:
            raise ValueError("Garmin RSD files expose one SideVü subsystem")

        header = read_rsd_header(path)
        port: list[RSDRecord] = []
        starboard: list[RSDRecord] = []
        for record in iter_rsd_records(path):
            # Garmin beam IDs 2 and 3 are the port and starboard SideVü beams.
            if record.beam == 2:
                port.append(record)
            elif record.beam == 3:
                starboard.append(record)

        if not port or not starboard:
            raise ValueError(f"No paired SideVü beams found in {path}")
        num_ping = min(len(port), len(starboard))
        if len(port) != len(starboard):
            logger.warning(
                "%s: SideVü channel counts differ (port=%d, starboard=%d); "
                "using the first %d paired pings",
                path.name,
                len(port),
                len(starboard),
                num_ping,
            )
        port = port[:num_ping]
        starboard = starboard[:num_ping]
        ping_len = max(
            max(record.sample_count for record in port),
            max(record.sample_count for record in starboard),
        )
        data = np.empty((2, num_ping, ping_len), dtype=np.uint16)
        _fill_samples(data[0], port, reverse=True)
        _fill_samples(data[1], starboard)

        port_time = np.asarray([record.recording_time_ms for record in port])
        star_time = np.asarray([record.recording_time_ms for record in starboard])
        time_ms = np.rint((port_time.astype(float) + star_time) / 2.0).astype(np.int64)
        recording_start = header.recording_start or _GARMIN_EPOCH
        timestamp = [
            recording_start + timedelta(milliseconds=int(value)) for value in time_ms
        ]

        port_lat = np.asarray([record.latitude for record in port], dtype=float)
        star_lat = np.asarray([record.latitude for record in starboard], dtype=float)
        port_lon = np.asarray([record.longitude for record in port], dtype=float)
        star_lon = np.asarray([record.longitude for record in starboard], dtype=float)
        latitude = _mean_finite(port_lat, star_lat)
        longitude = _mean_finite(port_lon, star_lon)
        sensor_heading, sensor_speed = _track_heading_and_speed(
            latitude, longitude, time_ms
        )

        port_range = np.asarray(
            [record.last_sample_depth_m for record in port], dtype=float
        )
        star_range = np.asarray(
            [record.last_sample_depth_m for record in starboard], dtype=float
        )
        slant_range = np.vstack((port_range, star_range))
        if not np.all(np.isfinite(slant_range)) or np.any(slant_range <= 0):
            raise RSDFormatError("RSD SideVü records contain invalid range values")
        sound_speed = 1500.0
        seconds_per_ping = 2.0 * slant_range / (sound_speed * ping_len)

        bottom_depth = _mean_finite(
            np.asarray([record.bottom_depth_m for record in port], dtype=float),
            np.asarray(
                [record.bottom_depth_m for record in starboard], dtype=float
            ),
        )
        starting_depth = _mean_finite(
            np.asarray(
                [record.first_sample_depth_m for record in port], dtype=float
            ),
            np.asarray(
                [record.first_sample_depth_m for record in starboard], dtype=float
            ),
        )
        gain_adc = _mean_finite(
            np.asarray(
                [record.gain if record.gain is not None else np.nan for record in port],
                dtype=float,
            ),
            np.asarray(
                [
                    record.gain if record.gain is not None else np.nan
                    for record in starboard
                ],
                dtype=float,
            ),
        )

        return SonarDataset(
            filepath=path,
            format_id=self.format_id,
            data=data,
            ping_x_axis=np.linspace(0.0, float(np.nanmedian(slant_range)), ping_len),
            timestamp=timestamp,
            sound_velocity=np.full(num_ping, sound_speed),
            starting_depth=starting_depth,
            gain_adc=gain_adc,
            longitude=longitude,
            latitude=latitude,
            # RSD provides bottom depth but not a reliable transducer depth.
            depth=np.zeros(num_ping),
            packet_no=np.arange(num_ping, dtype=np.int64),
            sensor_heading=sensor_heading,
            sensor_pitch=np.zeros(num_ping),
            sensor_primary_altitude=bottom_depth,
            sensor_aux_altitude=np.zeros(num_ping),
            sensor_roll=np.zeros(num_ping),
            sensor_speed=sensor_speed,
            seconds_per_ping=seconds_per_ping,
            slant_range=slant_range,
            layback_m=np.full(num_ping, np.nan),
            cable_out_m=np.full(num_ping, np.nan),
            choose_subsys=0,
            subsys_num=1,
            subsys_names=["SideVü"],
            reader_metadata={
                "unit_id": header.unit_id,
                "product": header.product,
                "rsd_version": header.version,
                "port_channel_id": port[0].channel_id,
                "starboard_channel_id": starboard[0].channel_id,
                # The GUI's legacy 32x reduction leaves only 64 samples from
                # a typical 2048-sample Garmin ping. Preserve enough bins for
                # contact shapes and bottom overlays to remain well resolved.
                "minimum_processed_samples_per_channel": 512,
                "port_sequence_count": np.asarray(
                    [record.sequence_count for record in port], dtype=np.uint32
                ),
                "starboard_sequence_count": np.asarray(
                    [record.sequence_count for record in starboard], dtype=np.uint32
                ),
            },
        )


__all__ = [
    "RSDChannelInfo",
    "RSDFormatError",
    "RSDHeader",
    "RSDReader",
    "RSDRecord",
    "iter_rsd_records",
    "read_rsd_header",
]
