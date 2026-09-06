"""JSF input adapter."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import numpy as np

from sidescantools.jsf import JSFFile, JSFSystemInformation, JSFSonarDataPacket
from sidescantools.readers.base import SonarDataset


logger = logging.getLogger(__name__)


def _finite_or_nan(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def jsf_tow_data(message) -> tuple[float, float]:
    """Return JSF direct layback and valid cable counter in meters."""

    layback_m = _finite_or_nan(getattr(message, "layback", np.nan))
    validity = int(getattr(message, "validity", 0))
    cable_out_m = float("nan")
    if validity & (1 << 11):
        raw_cable_out = _finite_or_nan(getattr(message, "cable_out", np.nan))
        if np.isfinite(raw_cable_out):
            cable_out_m = raw_cable_out / 10.0
    return layback_m, cable_out_m


class JSFReader:
    format_id = "jsf"
    suffixes = (".jsf",)

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        jsf_file = JSFFile(filepath)

        subsys_names: list[int] = []
        subsys_num = 0
        num_ch = 0
        for packet in jsf_file.packets:
            if type(packet) is JSFSystemInformation:
                subsys_num = packet.num_subsystems
            if type(packet) is JSFSonarDataPacket:
                if packet.header.subsys_no not in subsys_names:
                    subsys_names.append(packet.header.subsys_no)

        if len(subsys_names) != subsys_num:
            logger.info(
                "Mismatch on subsystem names: %s and expected count: %s",
                subsys_names,
                subsys_num,
            )
            subsys_num = len(subsys_names)
        if not subsys_names:
            raise ValueError(f"No sidescan sonar packets found in {filepath}")
        if not 0 <= choose_subsys < len(subsys_names):
            raise ValueError(
                f"Subsystem index {choose_subsys} is unavailable; "
                f"this file contains {len(subsys_names)} subsystem(s)"
            )

        data_port = []
        data_starboard = []
        expected_index = np.zeros(2, dtype=int)
        selected_subsystem = subsys_names[choose_subsys]
        for packet in jsf_file.packets:
            if type(packet) is not JSFSonarDataPacket:
                continue
            if packet.header.subsys_no != selected_subsystem:
                continue
            if packet.header.channel >= num_ch:
                num_ch = packet.header.channel + 1
            if expected_index[packet.header.channel] != packet.message.ping_no:
                logger.info(
                    "Expected ping mismatch: subsystem=%s channel=%s ping=%s expected=%s",
                    packet.header.subsys_no,
                    packet.header.channel,
                    packet.message.ping_no,
                    expected_index[packet.header.channel],
                )
                expected_index[packet.header.channel] = packet.message.ping_no + 1
            else:
                expected_index[packet.header.channel] += 1
            if packet.header.channel == 0:
                data_port.append(packet.message.data)
            elif packet.header.channel == 1:
                data_starboard.append(packet.message.data)

        if num_ch != 2:
            raise NotImplementedError(
                f"Expected 2 channels in {filepath}, but found {num_ch}"
            )
        data = np.array((np.fliplr(data_port), data_starboard))
        _num_ch, num_ping, ping_len = data.shape

        sound_velocity = np.empty(num_ping)
        timestamp = [datetime] * num_ping
        sensor_heading = np.empty(num_ping)
        sensor_pitch = np.empty(num_ping)
        sensor_primary_altitude = np.empty(num_ping)
        sensor_roll = np.empty(num_ping)
        sensor_speed = np.empty(num_ping)
        longitude = np.empty(num_ping)
        latitude = np.empty(num_ping)
        depth = np.empty(num_ping)
        packet_no = np.empty(num_ping)
        seconds_per_ping = np.empty((num_ch, num_ping))
        slant_range = np.empty((num_ch, num_ping))
        layback_m = np.full(num_ping, np.nan)
        cable_out_m = np.full(num_ping, np.nan)
        starting_depth = np.zeros(num_ping)
        gain_adc = np.zeros(num_ping)

        ping_index = 0
        for packet in jsf_file.packets:
            if type(packet) is not JSFSonarDataPacket:
                continue
            if (
                packet.header.subsys_no != selected_subsystem
                or packet.header.channel != 0
            ):
                continue
            message = packet.message
            sound_velocity[ping_index] = message.SOS
            timestamp[ping_index] = datetime.fromtimestamp(message.time)
            sensor_heading[ping_index] = message.compass_heading / 100
            sensor_pitch[ping_index] = message.pitch / 32768 * 180
            sensor_primary_altitude[ping_index] = message.altitude / 1e3
            sensor_roll[ping_index] = message.roll / 32768 * 180
            sensor_speed[ping_index] = message.speed / 10 / 1.944
            layback_m[ping_index], cable_out_m[ping_index] = jsf_tow_data(message)
            depth[ping_index] = message.depth / 1e3
            packet_no[ping_index] = message.ping_no
            seconds_per_ping[:, ping_index] = message.sampling_interval_ns / 1e9
            sound_speed = message.SOS or 1500
            slant_range[:, ping_index] = np.round(
                seconds_per_ping[0, ping_index] * ping_len * sound_speed / 2,
                decimals=2,
            )
            starting_depth[ping_index] = message.starting_depth
            gain_adc[ping_index] = message.gain_adc

            if message.coord_units != 2:
                raise NotImplementedError(
                    f"JSF coordinate unit {message.coord_units} is not supported"
                )
            longitude[ping_index] = message.longitude / 10000 / 60
            latitude[ping_index] = message.latitude / 10000 / 60
            ping_index += 1

        return SonarDataset(
            filepath=filepath,
            format_id=self.format_id,
            data=data,
            ping_x_axis=np.linspace(0, slant_range[0, 0], ping_len),
            timestamp=timestamp,
            sound_velocity=sound_velocity,
            starting_depth=starting_depth,
            gain_adc=gain_adc,
            longitude=longitude,
            latitude=latitude,
            depth=depth,
            packet_no=packet_no,
            sensor_heading=sensor_heading,
            sensor_pitch=sensor_pitch,
            sensor_primary_altitude=sensor_primary_altitude,
            sensor_aux_altitude=np.zeros(num_ping),
            sensor_roll=sensor_roll,
            sensor_speed=sensor_speed,
            seconds_per_ping=seconds_per_ping,
            slant_range=slant_range,
            layback_m=layback_m,
            cable_out_m=cable_out_m,
            choose_subsys=choose_subsys,
            subsys_num=subsys_num,
            subsys_names=subsys_names,
        )
