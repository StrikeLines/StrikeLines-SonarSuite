from __future__ import annotations

from datetime import datetime
from pathlib import Path
import struct

import numpy as np
import pytest

from sidescantools.readers.rsd import (
    RSDFormatError,
    RSDReader,
    iter_rsd_records,
    read_rsd_header,
)


FILE_MAGIC = 0xD9264B7C
RECORD_MAGIC = 0xB7E9DA86
TRAILER_MAGIC = 0xF98EACBC


def _varuint(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _zigzag(value: int) -> bytes:
    return _varuint((value << 1) ^ (value >> 31))


def _structure(fields: list[tuple[int, bytes]]) -> bytes:
    output = bytearray(_varuint(len(fields)))
    for field_number, value in fields:
        if 1 <= len(value) <= 6:
            output.extend(_varuint((field_number << 3) | len(value)))
        else:
            output.extend(_varuint((field_number << 3) | 7))
            output.extend(_varuint(len(value)))
        output.extend(value)
    return bytes(output)


def _varray(values: list[bytes]) -> bytes:
    output = bytearray(_varuint(len(values)))
    for value in values:
        output.extend(_varuint(len(value)))
        output.extend(value)
    return bytes(output)


def _map_units(degrees: float) -> bytes:
    value = round(degrees * (2**32) / 360.0)
    return int(value).to_bytes(4, "little", signed=True)


def _channel_info(channel_id: int, offset: int) -> bytes:
    return _structure(
        [
            (0, _varray([_varuint(channel_id)])),
            (1, offset.to_bytes(8, "little")),
        ]
    )


def _record(
    *, channel_id: int, beam: int, sequence: int, time_ms: int, samples: list[int]
) -> bytes:
    body_fields = _structure(
        [
            (0, _varuint(channel_id)),
            (1, _zigzag(8_000)),
            (3, _zigzag(0)),
            (4, _zigzag(40_000)),
            (5, b"\xff"),
            (7, len(samples).to_bytes(4, "little")),
            (8, b"\x00"),
            (9, _map_units(30.0 + sequence / 10_000)),
            (10, _map_units(-80.0 + sequence / 10_000)),
            (12, _varuint(beam)),
        ]
    ) + np.asarray(samples, dtype="<u2").tobytes()
    state = _structure([(1, _varray([_varuint(channel_id)]))])
    header = _structure(
        [
            (0, RECORD_MAGIC.to_bytes(4, "little")),
            (1, state),
            (2, sequence.to_bytes(4, "little")),
            (3, b"\0\0\0\0"),
            (4, len(body_fields).to_bytes(2, "little")),
            (5, time_ms.to_bytes(4, "little")),
        ]
    )
    without_trailer = header + b"\0\0\0\0" + body_fields
    size = len(without_trailer) + 12
    return without_trailer + struct.pack("<III", TRAILER_MAGIC, size, 0)


def _write_rsd(path: Path) -> None:
    file_info = _structure(
        [
            (0, (1).to_bytes(2, "little")),
            (1, (123).to_bytes(4, "little")),
            (2, (4040).to_bytes(2, "little")),
            # 2025-03-25 00:00:00 relative to Garmin's epoch.
            (3, (1_111_795_200).to_bytes(4, "little")),
        ]
    )
    channels = _varuint(2) + _channel_info(10, 0x5000) + _channel_info(11, 0x5000)
    header = _structure(
        [
            (0, FILE_MAGIC.to_bytes(4, "little")),
            (1, (0).to_bytes(2, "little")),
            (2, (2).to_bytes(4, "little")),
            (3, b"\x02"),
            (5, file_info),
            (6, channels),
        ]
    )
    records = b"".join(
        [
            _record(channel_id=10, beam=2, sequence=1, time_ms=100, samples=[1, 2, 3]),
            _record(channel_id=11, beam=3, sequence=7, time_ms=102, samples=[4, 5, 6]),
            _record(channel_id=10, beam=2, sequence=2, time_ms=200, samples=[7, 8, 9]),
            _record(
                channel_id=11,
                beam=3,
                sequence=9,
                time_ms=204,
                samples=[10, 11, 12],
            ),
        ]
    )
    path.write_bytes(header + b"\0" * (0x5000 - len(header)) + records)


def test_rsd_header_and_record_decoder(tmp_path):
    path = tmp_path / "line.rsd"
    _write_rsd(path)

    header = read_rsd_header(path)
    records = list(iter_rsd_records(path))

    assert header.product == 4040
    assert [channel.channel_id for channel in header.channels] == [10, 11]
    assert [(record.beam, record.sequence_count) for record in records] == [
        (2, 1),
        (3, 7),
        (2, 2),
        (3, 9),
    ]
    np.testing.assert_array_equal(records[0].samples, [1, 2, 3])
    assert records[0].bottom_depth_m == 8.0
    assert records[0].last_sample_depth_m == 40.0
    assert records[0].latitude == pytest.approx(30.0001, abs=1e-7)


def test_rsd_reader_normalizes_sidevu_orientation_and_metadata(tmp_path):
    path = tmp_path / "line.RSD"
    _write_rsd(path)

    dataset = RSDReader().read(path)

    assert dataset.format_id == "rsd"
    assert dataset.data.shape == (2, 2, 3)
    np.testing.assert_array_equal(dataset.data[0, 0], [3, 2, 1])
    np.testing.assert_array_equal(dataset.data[1, 0], [4, 5, 6])
    np.testing.assert_allclose(dataset.slant_range, 40.0)
    np.testing.assert_allclose(dataset.sensor_primary_altitude, 8.0)
    assert dataset.timestamp[0] == datetime(2025, 3, 25, 0, 0, 0, 101000)
    assert dataset.reader_metadata["port_channel_id"] == 10
    assert dataset.reader_metadata["starboard_channel_id"] == 11


def test_rsd_reader_rejects_non_sidevu_and_bad_magic(tmp_path):
    path = tmp_path / "bad.rsd"
    path.write_bytes(b"not an rsd file")

    with pytest.raises(RSDFormatError):
        read_rsd_header(path)
    with pytest.raises(ValueError, match="one SideVü subsystem"):
        RSDReader().read(path, choose_subsys=1)
