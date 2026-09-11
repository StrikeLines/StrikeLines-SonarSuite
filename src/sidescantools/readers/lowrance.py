"""Navico (Lowrance, Simrad, B&G) SLG/SL2/SL3 input adapter.

Navico sonar logs are a short file header followed by a gapless list of
frames.  Three on-disk layouts exist and this module decodes all of them:

* format 1 (``.slg``) -- variable-size frame header whose optional fields are
  present only when the matching validity flag is set;
* format 2 (``.sl2``) -- fixed 144-byte frame header; and
* format 3 (``.sl3``) -- fixed 168-byte frame header (128 bytes for the two
  undocumented debug channel types).

As in :mod:`sidescantools.readers.rsd`, the binary decoder is kept separate
from the :class:`~sidescantools.readers.base.SonarDataset` adapter so framing
and channel selection can be tested independently.

The layout follows Herbert Oppmann's "Navico Sonar Log File Format"
(2021-06-02) notes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

import numpy as np

from sidescantools.readers.base import SonarDataset


logger = logging.getLogger(__name__)


class LowranceFormatError(ValueError):
    """Raised when a Navico log is truncated or has invalid framing."""


# Channel types ("ChannelType" in the format notes). Format 1 does not record
# one, so frames decoded from a ``.slg`` file carry CHANNEL_UNKNOWN.
CHANNEL_UNKNOWN = -1
CHANNEL_PRIMARY = 0
CHANNEL_SECONDARY = 1
CHANNEL_DOWNSCAN = 2
CHANNEL_LEFT_SIDESCAN = 3
CHANNEL_RIGHT_SIDESCAN = 4
CHANNEL_COMPOSITE_SIDESCAN = 5
CHANNEL_3D = 9

CHANNEL_NAMES = {
    CHANNEL_UNKNOWN: "unrecorded",
    CHANNEL_PRIMARY: "primary",
    CHANNEL_SECONDARY: "secondary",
    CHANNEL_DOWNSCAN: "downscan",
    CHANNEL_LEFT_SIDESCAN: "left sidescan",
    CHANNEL_RIGHT_SIDESCAN: "right sidescan",
    CHANNEL_COMPOSITE_SIDESCAN: "composite sidescan",
    7: "debug (7)",
    8: "debug (8)",
    CHANNEL_3D: "3D",
    10: "debug digital",
    11: "debug noise",
}

FREQUENCY_NAMES = {
    0: "200 kHz",
    1: "50 kHz",
    2: "83 kHz",
    3: "455 kHz",
    4: "800 kHz",
    5: "38 kHz",
    6: "28 kHz",
    7: "130-210 kHz",
    8: "90-150 kHz",
    9: "40-60 kHz",
    10: "25-45 kHz",
    160: "83/800 kHz",
}

_SIDESCAN_CHANNELS = frozenset(
    {CHANNEL_LEFT_SIDESCAN, CHANNEL_RIGHT_SIDESCAN, CHANNEL_COMPOSITE_SIDESCAN}
)

_FILE_HEADER_SIZE = {1: 10, 2: 8, 3: 8}
_FRAME_HEADER_SIZE = {2: 144, 3: 168}
# Format 3 stores the two undocumented debug channels with a short header.
_SL3_SHORT_HEADER_SIZE = 128
_SL3_SHORT_HEADER_CHANNELS = frozenset({7, 8})

# Format 1 frame flags.
_SLG_UNKNOWN0 = 0x0001
_SLG_UNKNOWN1 = 0x0002
_SLG_ALTITUDE_VALID = 0x0004
_SLG_UPPER_LIMIT_VALID = 0x0008
_SLG_TEMPERATURE_VALID = 0x0010
_SLG_TEMPERATURE2_VALID = 0x0020
_SLG_TEMPERATURE3_VALID = 0x0040
_SLG_WATER_SPEED_VALID = 0x0080
_SLG_POSITION_VALID = 0x0100
_SLG_DEPTH_INVALID = 0x0200
_SLG_SURFACE_DEPTH_VALID = 0x0400
_SLG_TOP_OF_BOTTOM_VALID = 0x0800
_SLG_SPEED_AND_TRACK_VALID = 0x4000

# Format 2 frame flags. Format 3 documents the same offset as an unknown
# value, so format 3 frames are read without validity gating.
_SL2_GPS_SPEED_VALID = 0x0002
_SL2_TEMPERATURE_VALID = 0x0004
_SL2_POSITION_VALID = 0x0010
_SL2_WATER_SPEED_VALID = 0x0040
_SL2_TRACK_VALID = 0x0080
_SL2_HEADING_VALID = 0x0100
_SL2_ALTITUDE_VALID = 0x0200

_FEET_TO_METERS = 0.3048
_KNOTS_TO_METERS_PER_SECOND = 0.514444
# Navico serializes positions as spherical Mercator meters on the polar radius.
_POLAR_EARTH_RADIUS_M = 6356752.3142
_INVALID_ALTITUDE_FEET = -10000.0
_UNIX_EPOCH = datetime(1970, 1, 1)
# Some units leave CreationDateTime holding a device uptime counter instead of
# a wall clock, so a decoded date outside this window is not a real timestamp.
_EARLIEST_PLAUSIBLE_LOG = datetime(2000, 1, 1)
_FUTURE_CLOCK_SLACK = timedelta(days=366)
_SOUND_VELOCITY_MS = 1500.0
# The GUI's legacy 32x reduction would leave 48 samples of a 1536-sample
# composite sidescan half. Keep enough bins for contacts to stay resolved.
_MINIMUM_PROCESSED_SAMPLES = 512

_NAN = float("nan")


@dataclass(frozen=True)
class LowranceHeader:
    """Decoded Navico file header."""

    format: int
    version: int
    bytes_per_sounding: int
    debug: bool

    @property
    def file_header_size(self) -> int:
        return _FILE_HEADER_SIZE[self.format]


@dataclass(frozen=True)
class LowranceFrame:
    """One decoded sonar frame with SI-unit telemetry."""

    offset: int
    channel_type: int
    frame_index: int
    frame_size: int
    packet_size: int
    header_size: int
    time_offset_ms: float
    creation_time: datetime | None
    upper_limit_m: float
    lower_limit_m: float
    water_depth_m: float
    keel_depth_m: float
    frequency: int | None
    gps_speed_ms: float
    water_speed_ms: float
    water_temperature_c: float
    longitude: float
    latitude: float
    track_deg: float
    heading_deg: float
    altitude_m: float
    samples: np.ndarray

    @property
    def channel_name(self) -> str:
        return CHANNEL_NAMES.get(self.channel_type, f"unknown ({self.channel_type})")


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise LowranceFormatError(f"Truncated Navico {context}")
    return value


def _feet(value: float) -> float:
    return value * _FEET_TO_METERS


def _knots(value: float) -> float:
    return value * _KNOTS_TO_METERS_PER_SECOND


def _longitude(easting: int) -> float:
    return np.rad2deg(easting / _POLAR_EARTH_RADIUS_M)


def _latitude(northing: int) -> float:
    mercator = np.exp(northing / _POLAR_EARTH_RADIUS_M)
    return np.rad2deg(2.0 * np.arctan(mercator) - np.pi / 2)


def _posix_time(value: int) -> datetime | None:
    """Convert a Navico CreationDateTime to a naive UTC timestamp.

    The field is documented as a POSIX time with -1 meaning "not set", but
    some units leave a device uptime counter there instead. Only a date that
    could plausibly be a recording is accepted; anything else is reported as
    missing so the caller can fall back to the file's own timestamp.
    """

    if value <= 0 or value == 0xFFFFFFFF:
        return None
    decoded = _UNIX_EPOCH + timedelta(seconds=int(value))
    latest = datetime.now(timezone.utc).replace(tzinfo=None) + _FUTURE_CLOCK_SLACK
    if not _EARLIEST_PLAUSIBLE_LOG <= decoded <= latest:
        return None
    return decoded


def _altitude_m(value_feet: float) -> float:
    if value_feet == _INVALID_ALTITUDE_FEET or not np.isfinite(value_feet):
        return _NAN
    return _feet(value_feet)


def _slg_time_offset_ms(buffer: bytes, cursor: int) -> float:
    """Decode a format 1 TimeOffset, which files store as uint or float.

    The published notes describe TimeOffset as a float, but the sample files
    seen so far store a plain millisecond count. The two are unambiguous in
    practice: a millisecond count read as a float is subnormal until the log
    passes twelve days, and a genuine float millisecond value is never below
    one for anything but the first tick.
    """

    as_float = struct.unpack_from("<f", buffer, cursor)[0]
    if np.isfinite(as_float) and as_float >= 1.0:
        return float(as_float)
    return float(struct.unpack_from("<I", buffer, cursor)[0])


def read_lowrance_header(filepath: str | Path) -> LowranceHeader:
    """Read the Navico file header that precedes the frame list."""

    path = Path(filepath)
    with path.open("rb") as stream:
        raw = stream.read(10)
    if len(raw) < 8:
        raise LowranceFormatError(f"{path.name} is too short to be a Navico sonar log")

    file_format, version, bytes_per_sounding, debug, reserved = struct.unpack_from(
        "<HHHBB", raw, 0
    )
    if file_format not in _FILE_HEADER_SIZE:
        raise LowranceFormatError(
            f"{path.name} declares Navico format {file_format}; "
            "only 1 (.slg), 2 (.sl2) and 3 (.sl3) are supported"
        )
    if file_format == 1 and len(raw) < 10:
        raise LowranceFormatError(f"{path.name} has a truncated format 1 file header")
    if bytes_per_sounding <= 0:
        raise LowranceFormatError(
            f"{path.name} declares an invalid BytesPerSounding of {bytes_per_sounding}"
        )
    if reserved:
        logger.debug("%s: reserved file-header byte is %d", path.name, reserved)

    return LowranceHeader(
        format=file_format,
        version=version,
        bytes_per_sounding=bytes_per_sounding,
        debug=bool(debug),
    )


def _decode_format1_frame(
    buffer: bytes, offset: int, header: LowranceHeader, frame_index: int
) -> LowranceFrame:
    flags = struct.unpack_from("<H", buffer, 0)[0]
    cursor = 2

    def take_float() -> float:
        nonlocal cursor
        value = struct.unpack_from("<f", buffer, cursor)[0]
        cursor += 4
        return float(value)

    def take_int() -> int:
        nonlocal cursor
        value = struct.unpack_from("<i", buffer, cursor)[0]
        cursor += 4
        return int(value)

    lower_limit = take_float()
    water_depth = take_float()
    upper_limit = take_float() if flags & _SLG_UPPER_LIMIT_VALID else 0.0
    temperature = take_float() if flags & _SLG_TEMPERATURE_VALID else _NAN
    water_speed = take_float() if flags & _SLG_WATER_SPEED_VALID else _NAN
    longitude = latitude = _NAN
    if flags & _SLG_POSITION_VALID:
        northing = take_int()
        easting = take_int()
        longitude = _longitude(easting)
        latitude = _latitude(northing)
    if flags & _SLG_SURFACE_DEPTH_VALID:
        take_float()
    if flags & _SLG_TOP_OF_BOTTOM_VALID:
        take_float()
    if flags & _SLG_TEMPERATURE2_VALID:
        take_float()
    if flags & _SLG_TEMPERATURE3_VALID:
        take_float()
    if flags & _SLG_UNKNOWN0 and flags & _SLG_SPEED_AND_TRACK_VALID:
        take_float()
    if flags & _SLG_UNKNOWN1:
        take_float()
        take_float()
    time_offset_ms = _slg_time_offset_ms(buffer, cursor)
    cursor += 4
    gps_speed = track = _NAN
    if flags & _SLG_SPEED_AND_TRACK_VALID:
        gps_speed = take_float()
        track = take_float()
    altitude = take_float() if flags & _SLG_ALTITUDE_VALID else _NAN

    packet_size = struct.unpack_from("<H", buffer, cursor)[0]
    cursor += 2
    if cursor + packet_size > header.bytes_per_sounding:
        raise LowranceFormatError(
            f"Navico frame at offset {offset} declares {packet_size} sounding bytes "
            f"after a {cursor}-byte header, which does not fit in the "
            f"{header.bytes_per_sounding}-byte frame"
        )

    return LowranceFrame(
        offset=offset,
        channel_type=CHANNEL_UNKNOWN,
        frame_index=frame_index,
        frame_size=header.bytes_per_sounding,
        packet_size=packet_size,
        header_size=cursor,
        time_offset_ms=time_offset_ms,
        creation_time=None,
        upper_limit_m=_feet(upper_limit),
        lower_limit_m=_feet(lower_limit),
        water_depth_m=_NAN if flags & _SLG_DEPTH_INVALID else _feet(water_depth),
        keel_depth_m=_NAN,
        frequency=None,
        gps_speed_ms=_knots(gps_speed),
        water_speed_ms=_knots(water_speed),
        water_temperature_c=temperature,
        longitude=longitude,
        latitude=latitude,
        track_deg=np.rad2deg(track),
        # Format 1 has no heading field; course over ground is the only bearing.
        heading_deg=_NAN,
        altitude_m=_altitude_m(altitude),
        samples=np.frombuffer(buffer, dtype=np.uint8, count=packet_size, offset=cursor),
    )


def _decode_format2_frame(buffer: bytes, offset: int) -> LowranceFrame:
    (
        frame_size,
        _previous_frame_size,
        channel_type,
        packet_size,
        frame_index,
        upper_limit,
        lower_limit,
    ) = struct.unpack_from("<HHHHIff", buffer, 28)
    frequency = struct.unpack_from("<B", buffer, 53)[0]
    creation_time = _posix_time(struct.unpack_from("<i", buffer, 60)[0])
    water_depth, keel_depth = struct.unpack_from("<ff", buffer, 64)
    (
        gps_speed,
        temperature,
        easting,
        northing,
        water_speed,
        track,
        altitude,
        heading,
        flags,
    ) = struct.unpack_from("<ffiiffffH", buffer, 100)
    time_offset_ms = struct.unpack_from("<I", buffer, 140)[0]

    positioned = bool(flags & _SL2_POSITION_VALID)
    return LowranceFrame(
        offset=offset,
        channel_type=channel_type,
        frame_index=frame_index,
        frame_size=frame_size,
        packet_size=packet_size,
        header_size=_FRAME_HEADER_SIZE[2],
        time_offset_ms=float(time_offset_ms),
        creation_time=creation_time,
        upper_limit_m=_feet(upper_limit),
        lower_limit_m=_feet(lower_limit),
        water_depth_m=_feet(water_depth),
        keel_depth_m=_feet(keel_depth),
        frequency=frequency,
        gps_speed_ms=_knots(gps_speed) if flags & _SL2_GPS_SPEED_VALID else _NAN,
        water_speed_ms=_knots(water_speed) if flags & _SL2_WATER_SPEED_VALID else _NAN,
        water_temperature_c=(
            temperature if flags & _SL2_TEMPERATURE_VALID else _NAN
        ),
        longitude=_longitude(easting) if positioned else _NAN,
        latitude=_latitude(northing) if positioned else _NAN,
        track_deg=np.rad2deg(track) if flags & _SL2_TRACK_VALID else _NAN,
        heading_deg=np.rad2deg(heading) if flags & _SL2_HEADING_VALID else _NAN,
        altitude_m=_altitude_m(altitude) if flags & _SL2_ALTITUDE_VALID else _NAN,
        samples=np.empty(0, dtype=np.uint8),
    )


def _decode_format3_frame(buffer: bytes, offset: int) -> LowranceFrame:
    frame_size, _previous_frame_size, channel_type = struct.unpack_from(
        "<HHH", buffer, 8
    )
    frame_index = struct.unpack_from("<I", buffer, 16)[0]
    upper_limit, lower_limit = struct.unpack_from("<ff", buffer, 20)
    creation_time = _posix_time(struct.unpack_from("<i", buffer, 40)[0])
    packet_size = struct.unpack_from("<H", buffer, 44)[0]
    water_depth = struct.unpack_from("<f", buffer, 48)[0]
    frequency = struct.unpack_from("<B", buffer, 52)[0]
    (
        gps_speed,
        temperature,
        easting,
        northing,
        water_speed,
        track,
        altitude,
        heading,
    ) = struct.unpack_from("<ffiiIfff", buffer, 84)
    time_offset_ms = struct.unpack_from("<I", buffer, 124)[0]

    header_size = (
        _SL3_SHORT_HEADER_SIZE
        if channel_type in _SL3_SHORT_HEADER_CHANNELS
        else _FRAME_HEADER_SIZE[3]
    )
    return LowranceFrame(
        offset=offset,
        channel_type=channel_type,
        frame_index=frame_index,
        frame_size=frame_size,
        packet_size=packet_size,
        header_size=header_size,
        time_offset_ms=float(time_offset_ms),
        creation_time=creation_time,
        upper_limit_m=_feet(upper_limit),
        lower_limit_m=_feet(lower_limit),
        water_depth_m=_feet(water_depth),
        # Format 3 dropped the keel-depth field.
        keel_depth_m=_NAN,
        frequency=frequency,
        # Format 3 documents no validity flags, so every field is taken as read.
        gps_speed_ms=_knots(gps_speed),
        water_speed_ms=_knots(float(water_speed)),
        water_temperature_c=float(temperature),
        longitude=_longitude(easting),
        latitude=_latitude(northing),
        track_deg=np.rad2deg(track),
        heading_deg=np.rad2deg(heading),
        altitude_m=_altitude_m(altitude),
        samples=np.empty(0, dtype=np.uint8),
    )


def _iter_format1_frames(
    stream: BinaryIO, path: Path, header: LowranceHeader, file_size: int
) -> Iterator[LowranceFrame]:
    # Format 1 frames are a variable-size header followed by sounding bytes,
    # zero-padded so that every frame occupies exactly BytesPerSounding.
    stride = header.bytes_per_sounding
    offset = header.file_header_size
    frame_index = 0
    while offset + stride <= file_size:
        buffer = _read_exact(stream, stride, context=f"frame at offset {offset}")
        yield _decode_format1_frame(buffer, offset, header, frame_index)
        offset += stride
        frame_index += 1
    if offset < file_size:
        logger.info(
            "%s: ignoring %d trailing bytes after the last complete frame",
            path.name,
            file_size - offset,
        )


def _iter_indexed_frames(
    stream: BinaryIO,
    path: Path,
    header: LowranceHeader,
    file_size: int,
    *,
    included_channels: frozenset[int] | None = None,
    channel_counts: dict[int, int] | None = None,
) -> Iterator[LowranceFrame]:
    # Formats 2 and 3 carry the frame size in each header, so the walk follows
    # the declared chain rather than a fixed stride.
    minimum_header = (
        _FRAME_HEADER_SIZE[2] if header.format == 2 else _SL3_SHORT_HEADER_SIZE
    )
    offset = header.file_header_size
    while offset + minimum_header <= file_size:
        buffer = _read_exact(
            stream, minimum_header, context=f"frame header at offset {offset}"
        )
        if header.format == 2:
            frame_size, _previous_frame_size, channel_type = struct.unpack_from(
                "<HHH", buffer, 28
            )
            frame_header_size = _FRAME_HEADER_SIZE[2]
        else:
            frame_size, _previous_frame_size, channel_type = struct.unpack_from(
                "<HHH", buffer, 8
            )
            frame_header_size = (
                _SL3_SHORT_HEADER_SIZE
                if channel_type in _SL3_SHORT_HEADER_CHANNELS
                else _FRAME_HEADER_SIZE[3]
            )

        if frame_size <= frame_header_size:
            raise LowranceFormatError(
                f"Navico frame at offset {offset} declares a {frame_size}-byte "
                f"size, which cannot hold its {frame_header_size}-byte header"
            )
        if offset + frame_size > file_size:
            logger.info(
                "%s: dropping the final frame at offset %d, which is truncated",
                path.name,
                offset,
            )
            return

        if channel_counts is not None:
            channel_counts[channel_type] = channel_counts.get(channel_type, 0) + 1

        # SL2/SL3 files interleave every active transducer. The application
        # only needs sidescan frames, so seek over other channels before doing
        # their full telemetry decode or allocating payload bytes.
        if included_channels is not None and channel_type not in included_channels:
            stream.seek(frame_size - minimum_header, 1)
            offset += frame_size
            continue

        if header.format == 2:
            frame = _decode_format2_frame(buffer, offset)
        else:
            frame = _decode_format3_frame(buffer, offset)

        remaining_header = frame.header_size - minimum_header
        if remaining_header:
            _read_exact(
                stream, remaining_header, context=f"frame header at offset {offset}"
            )
        payload = _read_exact(
            stream,
            frame.frame_size - frame.header_size,
            context=f"frame payload at offset {offset}",
        )
        packet_size = frame.packet_size
        if packet_size > len(payload):
            logger.debug(
                "%s: frame at offset %d declares %d sounding bytes but only %d fit; "
                "using the frame size",
                path.name,
                offset,
                packet_size,
                len(payload),
            )
            packet_size = len(payload)
        yield replace(
            frame,
            packet_size=packet_size,
            samples=np.frombuffer(payload, dtype=np.uint8, count=packet_size),
        )
        offset += frame.frame_size


def iter_lowrance_frames(filepath: str | Path) -> Iterator[LowranceFrame]:
    """Yield decoded sonar frames while reading the log sequentially."""

    path = Path(filepath)
    header = read_lowrance_header(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(header.file_header_size)
        walk = _iter_format1_frames if header.format == 1 else _iter_indexed_frames
        yield from walk(stream, path, header, file_size)


def _iter_sidescan_frames(
    filepath: str | Path, channel_counts: dict[int, int]
) -> Iterator[LowranceFrame]:
    """Yield only SL2/SL3 sidescan frames while counting all channels."""

    path = Path(filepath)
    header = read_lowrance_header(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(header.file_header_size)
        yield from _iter_indexed_frames(
            stream,
            path,
            header,
            file_size,
            included_channels=_SIDESCAN_CHANNELS,
            channel_counts=channel_counts,
        )


def _fill_samples(
    output: np.ndarray, pings: list[np.ndarray], *, reverse: bool = False
) -> None:
    """Fill a preallocated channel, resampling only on sample-count changes."""

    width = output.shape[1]
    target = np.linspace(0.0, 1.0, width)
    for row, samples in enumerate(pings):
        count = len(samples)
        if count == width:
            row_samples = samples
        elif count == 1:
            output[row].fill(samples[0])
            continue
        elif count == 0:
            output[row].fill(0)
            continue
        else:
            source = np.linspace(0.0, 1.0, count)
            row_samples = np.rint(np.interp(target, source, samples)).astype(np.uint8)
        output[row] = row_samples[::-1] if reverse else row_samples


def _mean_finite(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.column_stack((first, second))
    count = np.sum(np.isfinite(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(
        total,
        count,
        out=np.full(len(values), _NAN, dtype=float),
        where=count > 0,
    )


def _frame_values(frames: list[LowranceFrame], attribute: str) -> np.ndarray:
    return np.asarray([getattr(frame, attribute) for frame in frames], dtype=float)


def _paired_values(
    port: list[LowranceFrame], starboard: list[LowranceFrame], attribute: str
) -> np.ndarray:
    return _mean_finite(
        _frame_values(port, attribute), _frame_values(starboard, attribute)
    )


def _coalesce(primary: np.ndarray, *fallbacks: np.ndarray) -> np.ndarray:
    """Return ``primary`` with non-finite entries taken from the fallbacks."""

    result = np.array(primary, dtype=float, copy=True)
    for fallback in fallbacks:
        missing = ~np.isfinite(result)
        if not missing.any():
            break
        result[missing] = np.asarray(fallback, dtype=float)[missing]
    return result


class LowranceReader:
    """Read Navico ``.slg``/``.sl2``/``.sl3`` sidescan logs.

    Navico logs interleave every active transducer channel in one file. This
    adapter keeps the sidescan channels: a paired left/right recording when
    both are present, otherwise the composite channel, whose packet holds the
    port half followed by the starboard half.

    Format 1 (``.slg``) does not record a channel type at all, so a sidescan
    log cannot be told apart from a down-looking one. Such files are rejected
    by default rather than split into a port/starboard pair that may be
    fabricated from a single down-looking beam. Callers who know their format 1
    logs are composite sidescan can opt in::

        register_sonar_reader(
            LowranceReader(slg_is_composite_sidescan=True), replace=True
        )
    """

    format_id = "lowrance"
    suffixes = (".slg", ".sl2", ".sl3")

    def __init__(self, *, slg_is_composite_sidescan: bool = False) -> None:
        self.slg_is_composite_sidescan = slg_is_composite_sidescan

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        path = Path(filepath)
        if choose_subsys != 0:
            raise ValueError("Navico sonar logs expose one sidescan subsystem")

        header = read_lowrance_header(path)
        if header.format == 1 and not self.slg_is_composite_sidescan:
            raise LowranceFormatError(
                f"{path.name} is a Navico format 1 (.slg) log, which does not record "
                "which channel each frame belongs to. Re-register LowranceReader with "
                "slg_is_composite_sidescan=True to read it as composite sidescan."
            )

        frames_by_channel, channel_counts = self._collect_sidescan_frames(path, header)
        port, starboard, layout = self._select_channels(
            path, frames_by_channel, channel_counts
        )
        return self._build_dataset(path, header, port, starboard, layout)

    @staticmethod
    def _collect_sidescan_frames(
        path: Path, header: LowranceHeader
    ) -> tuple[dict[int, list[LowranceFrame]], dict[int, int]]:
        frames_by_channel: dict[int, list[LowranceFrame]] = {}
        channel_counts: dict[int, int] = {}
        frame_iterator = (
            iter_lowrance_frames(path)
            if header.format == 1
            else _iter_sidescan_frames(path, channel_counts)
        )
        for frame in frame_iterator:
            channel = frame.channel_type
            if header.format == 1:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                # Opted-in format 1 logs are read as one composite channel.
                channel = CHANNEL_COMPOSITE_SIDESCAN
            frames_by_channel.setdefault(channel, []).append(frame)
        return frames_by_channel, channel_counts

    @staticmethod
    def _select_channels(
        path: Path,
        frames_by_channel: dict[int, list[LowranceFrame]],
        channel_counts: dict[int, int],
    ) -> tuple[list[LowranceFrame], list[LowranceFrame], str]:
        left = frames_by_channel.get(CHANNEL_LEFT_SIDESCAN, [])
        right = frames_by_channel.get(CHANNEL_RIGHT_SIDESCAN, [])
        composite = frames_by_channel.get(CHANNEL_COMPOSITE_SIDESCAN, [])

        if left and right:
            return left, right, "left_right"
        if composite:
            return composite, composite, "composite"

        present = ", ".join(
            f"{CHANNEL_NAMES.get(channel, f'unknown ({channel})')} x{count}"
            for channel, count in sorted(channel_counts.items())
        )
        raise LowranceFormatError(
            f"No sidescan channels found in {path.name}; it contains "
            f"{present or 'no decodable frames'}"
        )

    @staticmethod
    def _channel_samples(
        frames: list[LowranceFrame], layout: str, *, starboard: bool
    ) -> list[np.ndarray]:
        if layout == "composite":
            # The composite packet is stored in display order: far port first,
            # nadir in the middle, far starboard last. Each half therefore
            # already runs in the direction SonarDataset expects.
            halves = []
            for frame in frames:
                middle = len(frame.samples) // 2
                halves.append(
                    frame.samples[middle : middle * 2]
                    if starboard
                    else frame.samples[:middle]
                )
            return halves
        # Separate left/right channels each start at nadir. The caller
        # reverses port so it reaches SonarDataset outer-range to nadir.
        return [frame.samples for frame in frames]

    def _build_dataset(
        self,
        path: Path,
        header: LowranceHeader,
        port_frames: list[LowranceFrame],
        starboard_frames: list[LowranceFrame],
        layout: str,
    ) -> SonarDataset:
        num_ping = min(len(port_frames), len(starboard_frames))
        if len(port_frames) != len(starboard_frames):
            logger.warning(
                "%s: sidescan channel counts differ (port=%d, starboard=%d); "
                "using the first %d paired pings",
                path.name,
                len(port_frames),
                len(starboard_frames),
                num_ping,
            )
        port_frames = port_frames[:num_ping]
        starboard_frames = starboard_frames[:num_ping]

        port_samples = self._channel_samples(port_frames, layout, starboard=False)
        starboard_samples = self._channel_samples(
            starboard_frames, layout, starboard=True
        )
        ping_len = max(
            max((len(samples) for samples in port_samples), default=0),
            max((len(samples) for samples in starboard_samples), default=0),
        )
        if ping_len < 1:
            raise LowranceFormatError(
                f"Sidescan frames in {path.name} contain no sounding samples"
            )
        data = np.zeros((2, num_ping, ping_len), dtype=np.uint8)
        _fill_samples(data[0], port_samples, reverse=layout == "left_right")
        _fill_samples(data[1], starboard_samples)

        latitude = _paired_values(port_frames, starboard_frames, "latitude")
        longitude = _paired_values(port_frames, starboard_frames, "longitude")
        track_deg = _paired_values(port_frames, starboard_frames, "track_deg")
        heading_deg = _paired_values(port_frames, starboard_frames, "heading_deg")
        gps_speed = _paired_values(port_frames, starboard_frames, "gps_speed_ms")
        water_speed = _paired_values(port_frames, starboard_frames, "water_speed_ms")
        water_depth = _paired_values(port_frames, starboard_frames, "water_depth_m")
        keel_depth = _paired_values(port_frames, starboard_frames, "keel_depth_m")
        temperature = _paired_values(
            port_frames, starboard_frames, "water_temperature_c"
        )
        altitude = _paired_values(port_frames, starboard_frames, "altitude_m")

        slant_range = np.vstack(
            (
                _frame_values(port_frames, "lower_limit_m"),
                _frame_values(starboard_frames, "lower_limit_m"),
            )
        )
        if not np.all(np.isfinite(slant_range)) or np.any(slant_range <= 0):
            raise LowranceFormatError(
                f"Sidescan frames in {path.name} declare an invalid range window"
            )
        seconds_per_ping = 2.0 * slant_range / (_SOUND_VELOCITY_MS * ping_len)

        time_offset_ms = _paired_values(port_frames, starboard_frames, "time_offset_ms")
        recording_start, timestamp_source = self._recording_start(
            path, port_frames, time_offset_ms
        )
        timestamp = [
            recording_start + timedelta(milliseconds=float(offset))
            for offset in time_offset_ms
        ]

        # Each composite half spans nadir to the outer range regardless of the
        # window the frame reports for the full display width.
        starting_depth = (
            np.zeros(num_ping)
            if layout == "composite"
            else _paired_values(port_frames, starboard_frames, "upper_limit_m")
        )
        frequencies = sorted(
            {
                frame.frequency
                for frame in port_frames + starboard_frames
                if frame.frequency is not None
            }
        )

        return SonarDataset(
            filepath=path,
            format_id=self.format_id,
            data=data,
            ping_x_axis=np.linspace(0.0, float(np.nanmedian(slant_range)), ping_len),
            timestamp=timestamp,
            sound_velocity=np.full(num_ping, _SOUND_VELOCITY_MS),
            starting_depth=np.nan_to_num(starting_depth),
            # Navico logs carry no ADC gain setting.
            gain_adc=np.zeros(num_ping),
            longitude=longitude,
            latitude=latitude,
            # The transducer sits at keel depth, which only format 2 records.
            depth=np.nan_to_num(keel_depth),
            packet_no=np.asarray(
                [frame.frame_index for frame in port_frames], dtype=np.int64
            ),
            # Heading is often flagged invalid on hull-mounted units, in which
            # case course over ground is the only bearing the log provides.
            sensor_heading=np.nan_to_num(_coalesce(heading_deg, track_deg)),
            sensor_pitch=np.zeros(num_ping),
            # A hull-mounted transducer's height above the bottom is the
            # sounded water depth.
            sensor_primary_altitude=np.nan_to_num(water_depth),
            sensor_aux_altitude=np.zeros(num_ping),
            sensor_roll=np.zeros(num_ping),
            sensor_speed=np.nan_to_num(_coalesce(gps_speed, water_speed)),
            seconds_per_ping=seconds_per_ping,
            slant_range=slant_range,
            layback_m=np.full(num_ping, _NAN),
            cable_out_m=np.full(num_ping, _NAN),
            choose_subsys=0,
            subsys_num=1,
            subsys_names=["SideScan"],
            reader_metadata={
                "navico_format": header.format,
                "navico_version": header.version,
                "bytes_per_sounding": header.bytes_per_sounding,
                "debug_channels_enabled": header.debug,
                "sidescan_layout": layout,
                "frequency_names": [
                    FREQUENCY_NAMES.get(value, f"unknown ({value})")
                    for value in frequencies
                ],
                "timestamp_source": timestamp_source,
                "water_temperature_c": temperature,
                "track_deg": track_deg,
                "gps_altitude_m": altitude,
                "minimum_processed_samples_per_channel": _MINIMUM_PROCESSED_SAMPLES,
            },
        )

    @staticmethod
    def _recording_start(
        path: Path, frames: list[LowranceFrame], time_offset_ms: np.ndarray
    ) -> tuple[datetime, str]:
        """Return the log start time and how it was established.

        Formats 2 and 3 stamp each frame with the log's creation time. Format 1
        has no absolute clock, so the file's modification time -- which is when
        the recording was closed -- is walked back by the log duration.
        """

        for frame in frames:
            if frame.creation_time is not None:
                return frame.creation_time, "frame_creation_time"

        modified = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).replace(tzinfo=None)
        duration_ms = float(np.nanmax(time_offset_ms)) if len(time_offset_ms) else 0.0
        logger.info(
            "%s: no usable creation time in the log; deriving timestamps from the "
            "file's modification time",
            path.name,
        )
        return modified - timedelta(milliseconds=duration_ms), "file_modified_time"


__all__ = [
    "CHANNEL_3D",
    "CHANNEL_COMPOSITE_SIDESCAN",
    "CHANNEL_DOWNSCAN",
    "CHANNEL_LEFT_SIDESCAN",
    "CHANNEL_NAMES",
    "CHANNEL_PRIMARY",
    "CHANNEL_RIGHT_SIDESCAN",
    "CHANNEL_SECONDARY",
    "CHANNEL_UNKNOWN",
    "FREQUENCY_NAMES",
    "LowranceFormatError",
    "LowranceFrame",
    "LowranceHeader",
    "LowranceReader",
    "iter_lowrance_frames",
    "read_lowrance_header",
]
