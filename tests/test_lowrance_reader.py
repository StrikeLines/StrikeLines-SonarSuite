from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from sidescantools.readers import LowranceReader, sonar_reader_registry
from sidescantools.readers.lowrance import (
    CHANNEL_COMPOSITE_SIDESCAN,
    CHANNEL_DOWNSCAN,
    CHANNEL_LEFT_SIDESCAN,
    CHANNEL_PRIMARY,
    CHANNEL_RIGHT_SIDESCAN,
    LowranceFormatError,
    iter_lowrance_frames,
    read_lowrance_header,
)


POLAR_EARTH_RADIUS_M = 6356752.3142
FEET_TO_METERS = 0.3048
KNOTS_TO_METERS_PER_SECOND = 0.514444
CREATION_TIME = 1_412_432_468  # 2014-10-04 14:21:08 UTC

# Format 2 validity flags for a fix with GPS speed, temperature, position and
# course over ground, but no compass heading.
SL2_FLAGS = 0x0002 | 0x0004 | 0x0010 | 0x0080 | 0x0200

# Format 1 validity flags matching the same set of populated fields.
SLG_FLAGS = 0x0004 | 0x0010 | 0x0100 | 0x2000 | 0x4000


def _easting(longitude: float) -> int:
    return round(math.radians(longitude) * POLAR_EARTH_RADIUS_M)


def _northing(latitude: float) -> int:
    mercator = math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    return round(POLAR_EARTH_RADIUS_M * mercator)


def _file_header(
    file_format: int, *, bytes_per_sounding: int = 3200, version: int = 1
) -> bytes:
    header = struct.pack("<HHHBB", file_format, version, bytes_per_sounding, 0, 0)
    return header + b"\x00\x00" if file_format == 1 else header


def _sl2_frame(
    *,
    offset: int,
    channel: int,
    index: int,
    packet: bytes,
    previous_size: int = 0,
    lower_limit_ft: float = 100.0,
    upper_limit_ft: float = 0.0,
    depth_ft: float = 20.0,
    keel_depth_ft: float = 2.0,
    frequency: int = 3,
    gps_speed_knots: float = 4.0,
    temperature_c: float = 18.5,
    longitude: float = -84.0,
    latitude: float = 30.0,
    track_deg: float = 45.0,
    heading_deg: float = 90.0,
    altitude_ft: float = 30.0,
    flags: int = SL2_FLAGS,
    time_ms: int = 1000,
    creation_time: int = CREATION_TIME,
) -> bytes:
    header = bytearray(144)
    struct.pack_into("<I", header, 0, offset)
    struct.pack_into(
        "<HHHHI",
        header,
        28,
        len(packet) + 144,
        previous_size,
        channel,
        len(packet),
        index,
    )
    struct.pack_into("<ff", header, 40, upper_limit_ft, lower_limit_ft)
    header[53] = frequency
    struct.pack_into("<i", header, 60, creation_time)
    struct.pack_into("<ff", header, 64, depth_ft, keel_depth_ft)
    struct.pack_into(
        "<ffiiffffH",
        header,
        100,
        gps_speed_knots,
        temperature_c,
        _easting(longitude),
        _northing(latitude),
        0.0,
        math.radians(track_deg),
        altitude_ft,
        math.radians(heading_deg),
        flags,
    )
    struct.pack_into("<I", header, 140, time_ms)
    return bytes(header) + packet


def _sl3_frame(
    *,
    offset: int,
    channel: int,
    index: int,
    packet: bytes,
    previous_size: int = 0,
    lower_limit_ft: float = 100.0,
    upper_limit_ft: float = 0.0,
    depth_ft: float = 20.0,
    frequency: int = 4,
    gps_speed_knots: float = 4.0,
    temperature_c: float = 18.5,
    longitude: float = -84.0,
    latitude: float = 30.0,
    track_deg: float = 45.0,
    heading_deg: float = 90.0,
    altitude_ft: float = 30.0,
    time_ms: int = 1000,
    creation_time: int = CREATION_TIME,
) -> bytes:
    header = bytearray(168)
    struct.pack_into("<II", header, 0, offset, 10)
    struct.pack_into("<HHH", header, 8, len(packet) + 168, previous_size, channel)
    struct.pack_into("<I", header, 16, index)
    struct.pack_into("<ff", header, 20, upper_limit_ft, lower_limit_ft)
    struct.pack_into("<i", header, 40, creation_time)
    struct.pack_into("<H", header, 44, len(packet))
    struct.pack_into("<f", header, 48, depth_ft)
    header[52] = frequency
    struct.pack_into(
        "<ffiiIfff",
        header,
        84,
        gps_speed_knots,
        temperature_c,
        _easting(longitude),
        _northing(latitude),
        0,
        math.radians(track_deg),
        altitude_ft,
        math.radians(heading_deg),
    )
    struct.pack_into("<I", header, 124, time_ms)
    return bytes(header) + packet


def _slg_frame(
    *,
    packet: bytes,
    bytes_per_sounding: int,
    lower_limit_ft: float = 100.0,
    depth_ft: float = 20.0,
    temperature_c: float = 18.5,
    longitude: float = -84.0,
    latitude: float = 30.0,
    gps_speed_knots: float = 4.0,
    track_deg: float = 45.0,
    altitude_ft: float = 30.0,
    time_ms: int = 1000,
    flags: int = SLG_FLAGS,
) -> bytes:
    body = bytearray(struct.pack("<Hff", flags, lower_limit_ft, depth_ft))
    if flags & 0x0008:
        body += struct.pack("<f", 0.0)
    if flags & 0x0010:
        body += struct.pack("<f", temperature_c)
    if flags & 0x0080:
        body += struct.pack("<f", 0.0)
    if flags & 0x0100:
        body += struct.pack("<ii", _northing(latitude), _easting(longitude))
    if flags & 0x0400:
        body += struct.pack("<f", 1.0)
    if flags & 0x0800:
        body += struct.pack("<f", 2.0)
    if flags & 0x0020:
        body += struct.pack("<f", 3.0)
    if flags & 0x0040:
        body += struct.pack("<f", 4.0)
    if flags & 0x0001 and flags & 0x4000:
        body += struct.pack("<f", 5.0)
    if flags & 0x0002:
        body += struct.pack("<ff", 6.0, 7.0)
    body += struct.pack("<I", time_ms)
    if flags & 0x4000:
        body += struct.pack("<ff", gps_speed_knots, math.radians(track_deg))
    if flags & 0x0004:
        body += struct.pack("<f", altitude_ft)
    body += struct.pack("<H", len(packet))
    body += packet
    return bytes(body).ljust(bytes_per_sounding, b"\x00")


def _packet(start: int, length: int) -> bytes:
    return bytes((start + index) % 256 for index in range(length))


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _composite_sl2(
    tmp_path: Path, *, ping_count: int = 4, packet_len: int = 64
) -> Path:
    payload = bytearray(_file_header(2))
    offset = 8
    previous = 0
    for index in range(ping_count):
        frame = _sl2_frame(
            offset=offset,
            channel=CHANNEL_COMPOSITE_SIDESCAN,
            index=index,
            packet=_packet(index * 8, packet_len),
            previous_size=previous,
            time_ms=1000 * (index + 1),
            latitude=30.0 + index / 1000,
        )
        payload += frame
        previous = len(frame)
        offset += len(frame)
    return _write(tmp_path / "composite.sl2", bytes(payload))


def test_read_header_reports_the_declared_layout(tmp_path):
    path = _write(tmp_path / "line.sl3", _file_header(3, bytes_per_sounding=1600))

    header = read_lowrance_header(path)

    assert (header.format, header.version, header.bytes_per_sounding) == (3, 1, 1600)
    assert header.file_header_size == 8


def test_format_1_header_is_two_bytes_longer(tmp_path):
    path = _write(tmp_path / "line.slg", _file_header(1, version=0))

    assert read_lowrance_header(path).file_header_size == 10


def test_unknown_format_is_rejected(tmp_path):
    path = _write(tmp_path / "line.sl2", _file_header(4))

    with pytest.raises(LowranceFormatError, match="Navico format 4"):
        read_lowrance_header(path)


def test_short_file_is_rejected(tmp_path):
    path = _write(tmp_path / "line.sl2", b"\x02\x00\x01")

    with pytest.raises(LowranceFormatError, match="too short"):
        read_lowrance_header(path)


def test_registry_owns_every_navico_extension():
    for suffix in (".slg", ".sl2", ".sl3"):
        reader = sonar_reader_registry.reader_for(f"line{suffix}")
        assert isinstance(reader, LowranceReader)
    assert isinstance(sonar_reader_registry.reader_for("LINE.SL2"), LowranceReader)


def test_composite_frames_split_into_port_and_starboard(tmp_path):
    path = _composite_sl2(tmp_path, ping_count=3, packet_len=64)

    dataset = LowranceReader().read(path)

    assert dataset.data.shape == (2, 3, 32)
    assert dataset.data.dtype == np.uint8
    for ping in range(3):
        packet = np.frombuffer(_packet(ping * 8, 64), dtype=np.uint8)
        # The composite packet is already in display order, so the port half
        # runs outer-range to nadir and needs no reversal.
        np.testing.assert_array_equal(dataset.data[0, ping], packet[:32])
        np.testing.assert_array_equal(dataset.data[1, ping], packet[32:])
    assert dataset.reader_metadata["sidescan_layout"] == "composite"


def test_composite_metadata_and_units_are_normalized(tmp_path):
    path = _composite_sl2(tmp_path, ping_count=2, packet_len=64)

    dataset = LowranceReader().read(path)

    assert dataset.format_id == "lowrance"
    # Positions round-trip through integer Mercator meters.
    assert dataset.latitude[0] == pytest.approx(30.0, abs=1e-5)
    assert dataset.longitude[0] == pytest.approx(-84.0, abs=1e-5)
    # Positions round-trip through integer Mercator meters.
    assert dataset.latitude[1] == pytest.approx(30.001, abs=1e-5)
    assert dataset.slant_range[0, 0] == pytest.approx(100.0 * FEET_TO_METERS)
    assert dataset.depth[0] == pytest.approx(2.0 * FEET_TO_METERS)
    assert dataset.sensor_primary_altitude[0] == pytest.approx(20.0 * FEET_TO_METERS)
    assert dataset.sensor_speed[0] == pytest.approx(4.0 * KNOTS_TO_METERS_PER_SECOND)
    # Heading is flagged invalid, so course over ground supplies the bearing.
    assert dataset.sensor_heading[0] == pytest.approx(45.0)
    assert dataset.reader_metadata["water_temperature_c"][0] == pytest.approx(18.5)
    assert dataset.reader_metadata["frequency_names"] == ["455 kHz"]
    # Each composite half starts at nadir regardless of the display window.
    np.testing.assert_array_equal(dataset.starting_depth, np.zeros(2))
    assert dataset.subsys_names == ["SideScan"]
    assert dataset.reader_metadata["minimum_processed_samples_per_channel"] == 512


def test_compass_heading_is_preferred_when_flagged_valid(tmp_path):
    payload = _file_header(2) + _sl2_frame(
        offset=8,
        channel=CHANNEL_COMPOSITE_SIDESCAN,
        index=0,
        packet=_packet(0, 32),
        flags=SL2_FLAGS | 0x0100,
    )
    path = _write(tmp_path / "heading.sl2", payload)

    dataset = LowranceReader().read(path)

    assert dataset.sensor_heading[0] == pytest.approx(90.0)


def test_timestamps_start_at_the_logged_creation_time(tmp_path):
    path = _composite_sl2(tmp_path, ping_count=3)

    dataset = LowranceReader().read(path)

    start = datetime(1970, 1, 1) + timedelta(seconds=CREATION_TIME)
    assert dataset.timestamp[0] == start + timedelta(milliseconds=1000)
    assert dataset.timestamp[2] == start + timedelta(milliseconds=3000)
    assert dataset.reader_metadata["timestamp_source"] == "frame_creation_time"


def test_uptime_counter_in_the_creation_field_is_not_used_as_a_clock(tmp_path):
    # Some units leave a device uptime counter in CreationDateTime, which
    # decodes to a 1970 date rather than when the survey was run.
    payload = bytearray(_file_header(2))
    offset = 8
    for index in range(2):
        frame = _sl2_frame(
            offset=offset,
            channel=CHANNEL_COMPOSITE_SIDESCAN,
            index=index,
            packet=_packet(0, 32),
            time_ms=5000 * (index + 1),
            creation_time=8_271_513,
        )
        payload += frame
        offset += len(frame)
    path = _write(tmp_path / "uptime.sl2", bytes(payload))
    modified = datetime(2014, 12, 11, 23, 55, 0, tzinfo=timezone.utc)
    os.utime(path, (modified.timestamp(), modified.timestamp()))

    dataset = LowranceReader().read(path)

    assert dataset.reader_metadata["timestamp_source"] == "file_modified_time"
    assert dataset.timestamp[-1] == modified.replace(tzinfo=None)


def test_separate_left_and_right_channels_store_port_outer_to_nadir(tmp_path):
    payload = bytearray(_file_header(2))
    offset = 8
    for index in range(2):
        for channel, start in (
            (CHANNEL_LEFT_SIDESCAN, index * 8),
            (CHANNEL_RIGHT_SIDESCAN, 128 + index * 8),
        ):
            frame = _sl2_frame(
                offset=offset,
                channel=channel,
                index=index,
                packet=_packet(start, 16),
                time_ms=1000 * (index + 1),
            )
            payload += frame
            offset += len(frame)
    path = _write(tmp_path / "paired.sl2", bytes(payload))

    dataset = LowranceReader().read(path)

    assert dataset.data.shape == (2, 2, 16)
    assert dataset.reader_metadata["sidescan_layout"] == "left_right"
    for ping in range(2):
        left = np.frombuffer(_packet(ping * 8, 16), dtype=np.uint8)
        right = np.frombuffer(_packet(128 + ping * 8, 16), dtype=np.uint8)
        np.testing.assert_array_equal(dataset.data[0, ping], left[::-1])
        np.testing.assert_array_equal(dataset.data[1, ping], right)


def test_unpaired_side_channels_are_truncated_to_the_shorter_one(tmp_path, caplog):
    payload = bytearray(_file_header(2))
    offset = 8
    channels = [CHANNEL_LEFT_SIDESCAN, CHANNEL_RIGHT_SIDESCAN, CHANNEL_LEFT_SIDESCAN]
    for index, channel in enumerate(channels):
        frame = _sl2_frame(
            offset=offset, channel=channel, index=index, packet=_packet(index, 16)
        )
        payload += frame
        offset += len(frame)
    path = _write(tmp_path / "unpaired.sl2", bytes(payload))

    with caplog.at_level("WARNING"):
        dataset = LowranceReader().read(path)

    assert dataset.num_ping == 1
    assert "channel counts differ" in caplog.text


def test_varying_sample_counts_are_resampled_to_a_common_width(tmp_path):
    payload = bytearray(_file_header(2))
    offset = 8
    for index, length in enumerate((64, 32)):
        frame = _sl2_frame(
            offset=offset,
            channel=CHANNEL_COMPOSITE_SIDESCAN,
            index=index,
            packet=_packet(0, length),
            time_ms=1000 * (index + 1),
        )
        payload += frame
        offset += len(frame)
    path = _write(tmp_path / "ragged.sl2", bytes(payload))

    dataset = LowranceReader().read(path)

    assert dataset.data.shape == (2, 2, 32)
    # The short ping is stretched across the full width, keeping its endpoints.
    assert dataset.data[1, 1, 0] == 16
    assert dataset.data[1, 1, -1] == 31


def test_missing_sidescan_channels_report_what_the_log_contains(tmp_path):
    payload = bytearray(_file_header(2))
    offset = 8
    channels = (CHANNEL_PRIMARY, CHANNEL_DOWNSCAN, CHANNEL_PRIMARY)
    for index, channel in enumerate(channels):
        frame = _sl2_frame(
            offset=offset, channel=channel, index=index, packet=_packet(0, 16)
        )
        payload += frame
        offset += len(frame)
    path = _write(tmp_path / "downlooking.sl2", bytes(payload))

    with pytest.raises(LowranceFormatError) as error:
        LowranceReader().read(path)

    assert "primary x2" in str(error.value)
    assert "downscan x1" in str(error.value)


def test_truncated_final_frame_is_dropped(tmp_path):
    complete = _file_header(2) + _sl2_frame(
        offset=8, channel=CHANNEL_COMPOSITE_SIDESCAN, index=0, packet=_packet(0, 32)
    )
    partial = _sl2_frame(
        offset=len(complete),
        channel=CHANNEL_COMPOSITE_SIDESCAN,
        index=1,
        packet=_packet(0, 32),
    )[:100]
    path = _write(tmp_path / "truncated.sl2", complete + partial)

    dataset = LowranceReader().read(path)

    assert dataset.num_ping == 1


def test_frame_smaller_than_its_header_is_rejected(tmp_path):
    frame = bytearray(
        _sl2_frame(
            offset=8, channel=CHANNEL_COMPOSITE_SIDESCAN, index=0, packet=_packet(0, 32)
        )
    )
    struct.pack_into("<H", frame, 28, 100)
    path = _write(tmp_path / "invalid.sl2", _file_header(2) + bytes(frame))

    with pytest.raises(LowranceFormatError, match="cannot hold its 144-byte header"):
        list(iter_lowrance_frames(path))


def test_format_3_composite_frames_are_read(tmp_path):
    payload = bytearray(_file_header(3, bytes_per_sounding=3200))
    offset = 8
    for index in range(2):
        frame = _sl3_frame(
            offset=offset,
            channel=CHANNEL_COMPOSITE_SIDESCAN,
            index=index,
            packet=_packet(index * 4, 32),
            time_ms=500 * (index + 1),
        )
        payload += frame
        offset += len(frame)
    path = _write(tmp_path / "line.sl3", bytes(payload))

    dataset = LowranceReader().read(path)

    assert dataset.data.shape == (2, 2, 16)
    assert dataset.reader_metadata["navico_format"] == 3
    assert dataset.reader_metadata["frequency_names"] == ["800 kHz"]
    # Format 3 dropped the keel-depth field, so the transducer depth is unset.
    np.testing.assert_array_equal(dataset.depth, np.zeros(2))
    assert dataset.sensor_heading[0] == pytest.approx(90.0)
    np.testing.assert_array_equal(
        dataset.data[1, 0], np.frombuffer(_packet(0, 32), dtype=np.uint8)[16:]
    )


def test_format_3_short_header_channels_are_skipped(tmp_path):
    payload = bytearray(_file_header(3))
    offset = 8
    debug = bytearray(
        _sl3_frame(offset=offset, channel=8, index=0, packet=_packet(0, 32))
    )
    # ChannelType 7 and 8 use a 128-byte header, so the frame is 40 bytes shorter.
    del debug[128:168]
    struct.pack_into("<H", debug, 8, len(debug))
    payload += debug
    offset += len(debug)
    payload += _sl3_frame(
        offset=offset,
        channel=CHANNEL_COMPOSITE_SIDESCAN,
        index=0,
        packet=_packet(0, 32),
    )
    path = _write(tmp_path / "debug.sl3", bytes(payload))

    frames = list(iter_lowrance_frames(path))

    assert [frame.header_size for frame in frames] == [128, 168]
    assert LowranceReader().read(path).num_ping == 1


def test_format_1_is_rejected_without_an_explicit_channel_assumption(tmp_path):
    payload = _file_header(1, bytes_per_sounding=256) + _slg_frame(
        packet=_packet(0, 64), bytes_per_sounding=256
    )
    path = _write(tmp_path / "line.slg", payload)

    with pytest.raises(LowranceFormatError, match="does not record"):
        LowranceReader().read(path)


def test_format_1_reads_as_composite_sidescan_when_opted_in(tmp_path):
    payload = bytearray(_file_header(1, bytes_per_sounding=256))
    for index in range(3):
        payload += _slg_frame(
            packet=_packet(index * 4, 64),
            bytes_per_sounding=256,
            time_ms=1000 * (index + 1),
            latitude=30.0 + index / 1000,
        )
    path = _write(tmp_path / "line.slg", bytes(payload))

    dataset = LowranceReader(slg_is_composite_sidescan=True).read(path)

    assert dataset.data.shape == (2, 3, 32)
    # Positions round-trip through integer Mercator meters.
    assert dataset.latitude[1] == pytest.approx(30.001, abs=1e-5)
    assert dataset.slant_range[0, 0] == pytest.approx(100.0 * FEET_TO_METERS)
    assert dataset.sensor_heading[0] == pytest.approx(45.0)
    # Format 1 records no keel depth and no compass heading.
    np.testing.assert_array_equal(dataset.depth, np.zeros(3))
    np.testing.assert_array_equal(
        dataset.data[0, 0], np.frombuffer(_packet(0, 64), dtype=np.uint8)[:32]
    )


def test_format_1_optional_fields_shift_the_header(tmp_path):
    # Adding the two "unknown" blocks and the remaining conditional fields
    # lengthens the header without changing the frame stride.
    flags = SLG_FLAGS | 0x0001 | 0x0002 | 0x0008 | 0x0020
    flags |= 0x0040 | 0x0080 | 0x0400 | 0x0800
    payload = _file_header(1, bytes_per_sounding=256) + _slg_frame(
        packet=_packet(9, 64), bytes_per_sounding=256, flags=flags
    )
    path = _write(tmp_path / "wide.slg", payload)

    (frame,) = list(iter_lowrance_frames(path))

    assert frame.header_size == 76
    assert frame.frame_size == 256
    assert frame.latitude == pytest.approx(30.0, abs=1e-5)
    np.testing.assert_array_equal(
        frame.samples, np.frombuffer(_packet(9, 64), dtype=np.uint8)
    )


def test_format_1_sounding_data_must_fit_in_the_frame(tmp_path):
    frame = bytearray(_slg_frame(packet=_packet(0, 64), bytes_per_sounding=256))
    struct.pack_into("<H", frame, 38, 4096)
    path = _write(
        tmp_path / "overflow.slg",
        _file_header(1, bytes_per_sounding=256) + bytes(frame),
    )

    with pytest.raises(LowranceFormatError, match="does not fit"):
        list(iter_lowrance_frames(path))


def test_format_1_timestamps_fall_back_to_the_file_modification_time(tmp_path):
    payload = bytearray(_file_header(1, bytes_per_sounding=256))
    for index in range(2):
        payload += _slg_frame(
            packet=_packet(0, 64),
            bytes_per_sounding=256,
            time_ms=5000 * (index + 1),
        )
    path = _write(tmp_path / "line.slg", bytes(payload))
    modified = datetime(2014, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    os.utime(path, (modified.timestamp(), modified.timestamp()))

    dataset = LowranceReader(slg_is_composite_sidescan=True).read(path)

    assert dataset.reader_metadata["timestamp_source"] == "file_modified_time"
    # The log closes at the modification time, so the last ping lands there.
    assert dataset.timestamp[-1] == modified.replace(tzinfo=None)
    assert dataset.timestamp[0] == modified.replace(tzinfo=None) - timedelta(seconds=5)


def test_subsystem_selection_is_rejected(tmp_path):
    path = _composite_sl2(tmp_path, ping_count=2)

    with pytest.raises(ValueError, match="one sidescan subsystem"):
        LowranceReader().read(path, choose_subsys=1)
