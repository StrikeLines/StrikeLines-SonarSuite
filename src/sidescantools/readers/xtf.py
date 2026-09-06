"""XTF input adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyxtf

from sidescantools.readers.base import SonarDataset
from sidescantools.xtf_wrapper import XTFWrapper


def _finite_or_nan(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def xtf_tow_data(packet) -> tuple[float, float]:
    """Return XTF direct layback and cable-out values in meters."""

    layback_m = _finite_or_nan(getattr(packet, "Layback", np.nan))
    cable_out_m = _finite_or_nan(getattr(packet, "CableOut", np.nan))
    hundredths = _finite_or_nan(getattr(packet, "CableOutHundredths", 0.0))
    if np.isfinite(cable_out_m) and np.isfinite(hundredths):
        cable_out_m += hundredths / 100.0
    return layback_m, cable_out_m


class XTFReader:
    format_id = "xtf"
    suffixes = (".xtf",)

    def read(self, filepath: Path, *, choose_subsys: int = 0) -> SonarDataset:
        xtf = XTFWrapper(file_path=filepath)
        data = np.asarray(xtf.sonar_data)
        num_ch, num_ping, _ping_len = data.shape

        sound_velocity = np.empty(num_ping)
        timestamp = [datetime] * num_ping
        sensor_heading = np.empty(num_ping)
        sensor_pitch = np.empty(num_ping)
        sensor_aux_altitude = np.empty(num_ping)
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

        sonar_packets = xtf.packets[pyxtf.XTFHeaderType.sonar]
        for ping_index in range(num_ping):
            ping = sonar_packets[ping_index]
            layback_m[ping_index], cable_out_m[ping_index] = xtf_tow_data(ping)
            sound_velocity[ping_index] = ping.SoundVelocity * 2
            timestamp[ping_index] = datetime(
                ping.Year,
                ping.Month,
                ping.Day,
                ping.Hour,
                ping.Minute,
                ping.Second,
            )
            sensor_heading[ping_index] = ping.SensorHeading
            sensor_pitch[ping_index] = ping.SensorPitch
            sensor_primary_altitude[ping_index] = ping.SensorPrimaryAltitude
            sensor_roll[ping_index] = ping.SensorRoll
            sensor_speed[ping_index] = ping.SensorSpeed / 1.943844
            sensor_aux_altitude[ping_index] = ping.SensorAuxAltitude
            longitude[ping_index] = ping.SensorXcoordinate
            latitude[ping_index] = ping.SensorYcoordinate
            depth[ping_index] = ping.SensorDepth
            packet_no[ping_index] = ping.PingNumber
            channel_headers = ping.ping_chan_headers
            for channel_index in range(num_ch):
                header = (
                    channel_headers[channel_index]
                    if len(channel_headers) == num_ch
                    else channel_headers[0]
                )
                seconds_per_ping[channel_index, ping_index] = header.SecondsPerPing
                slant_range[channel_index, ping_index] = header.SlantRange

        return SonarDataset(
            filepath=filepath,
            format_id=self.format_id,
            data=data,
            ping_x_axis=xtf.x_axis_m,
            timestamp=timestamp,
            sound_velocity=sound_velocity,
            starting_depth=np.zeros(num_ping),
            gain_adc=np.zeros(num_ping),
            longitude=longitude,
            latitude=latitude,
            depth=depth,
            packet_no=packet_no,
            sensor_heading=sensor_heading,
            sensor_pitch=sensor_pitch,
            sensor_primary_altitude=sensor_primary_altitude,
            sensor_aux_altitude=sensor_aux_altitude,
            sensor_roll=sensor_roll,
            sensor_speed=sensor_speed,
            seconds_per_ping=seconds_per_ping,
            slant_range=slant_range,
            layback_m=layback_m,
            cable_out_m=cable_out_m,
            choose_subsys=choose_subsys,
            subsys_names=["default"],
            bottom_line_storage_reversed=True,
        )
