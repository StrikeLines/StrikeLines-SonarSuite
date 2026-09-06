"""Backward-compatible facade over SonarSuite's modular reader registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from sidescantools.readers import (
    SonarDataset,
    UnsupportedSonarFormatError,
    is_supported_sonar_path,
    jsf_tow_data,
    load_sonar_dataset,
    register_sonar_reader,
    sonar_file_dialog_filter,
    supported_sonar_format_ids,
    supported_sonar_suffixes,
    unregister_sonar_reader,
    xtf_tow_data,
)


class SidescanFile:
    """Load a sonar file through its registered format adapter.

    The public attributes intentionally match the historical ``SidescanFile``
    API. Existing processing code therefore receives the new normalized
    dataset without needing a coordinated rewrite.
    """

    filepath: Path
    format_id: str
    choose_subsys: int
    ping_len: int
    num_ping: int
    num_ch: int
    subsys_num: int
    data: np.ndarray
    subsys_names: list[Any]
    sound_velocity: np.ndarray
    ping_x_axis: np.ndarray
    timestamp: list[Any]
    starting_depth: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    coord_units: int
    gain_adc: np.ndarray
    depth: np.ndarray
    packet_no: np.ndarray
    sensor_heading: np.ndarray
    sensor_pitch: np.ndarray
    sensor_primary_altitude: np.ndarray
    sensor_roll: np.ndarray
    sensor_speed: np.ndarray
    sensor_aux_altitude: np.ndarray
    seconds_per_ping: np.ndarray
    slant_range: np.ndarray
    layback_m: np.ndarray
    cable_out_m: np.ndarray
    bottom_line_storage_reversed: bool

    def __init__(self, filepath: str | os.PathLike, choose_subsys: int = 0):
        dataset = load_sonar_dataset(filepath, choose_subsys=choose_subsys)
        self.__dict__.update(vars(dataset))
        self._dataset = dataset


__all__ = [
    "SidescanFile",
    "SonarDataset",
    "UnsupportedSonarFormatError",
    "is_supported_sonar_path",
    "jsf_tow_data",
    "register_sonar_reader",
    "sonar_file_dialog_filter",
    "supported_sonar_format_ids",
    "supported_sonar_suffixes",
    "unregister_sonar_reader",
    "xtf_tow_data",
]
