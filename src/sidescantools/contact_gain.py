"""Built-in SidescanTools gain processing for an anchor-safe waterfall view."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Callable

import numpy as np

from sidescantools.aux_functions import convert_to_dB, hist_equalization
from sidescantools.destripe import destripe_waterfall


class BuiltInGainMode(str, Enum):
    RAW = "raw"
    SLANT = "slant"
    BAC = "bac"
    EGN = "egn"


@dataclass(frozen=True, slots=True)
class BuiltInGainRequest:
    mode: BuiltInGainMode = BuiltInGainMode.RAW
    egn_table_path: Path | None = None
    bac_angle_count: int = 360
    energy_normalization: bool = True
    convert_db: bool = False
    clahe: bool = False
    nadir_angle: float = 0.0
    use_internal_altitude: bool = False
    destripe: bool = False
    slant_range_correction: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", BuiltInGainMode(self.mode))
        if self.bac_angle_count < 8:
            raise ValueError("BAC angle count must be at least 8")
        if not math.isfinite(float(self.nadir_angle)):
            raise ValueError("nadir angle must be finite")
        if self.mode is BuiltInGainMode.EGN:
            if self.egn_table_path is None:
                raise ValueError("EGN mode requires an EGN table")
            object.__setattr__(self, "egn_table_path", Path(self.egn_table_path))


@dataclass(frozen=True, slots=True)
class BuiltInGainResult:
    display_data: np.ndarray
    pipeline_description: str


def normalize_processed_waterfall(values: np.ndarray) -> np.ndarray:
    """Robustly normalize processed intensities to a finite [0, 1] display."""

    values = np.asarray(values, dtype=float)
    finite_nonzero = np.isfinite(values) & (values != 0)
    output = np.zeros(values.shape, dtype=float)
    if not np.any(finite_nonzero):
        return output
    low, high = np.percentile(values[finite_nonzero], (1.0, 99.5))
    if high <= low:
        low = float(np.min(values[finite_nonzero]))
        high = float(np.max(values[finite_nonzero]))
    if high <= low:
        return output
    finite = np.isfinite(values)
    output[finite] = np.clip((values[finite] - low) / (high - low), 0.0, 1.0)
    return output


def ground_to_slant_display(
    ground_display: np.ndarray,
    depth_by_channel: np.ndarray,
) -> np.ndarray:
    """Inverse-project corrected ground pixels to original acoustic samples.

    Both input and output use ``PORT outer..nadir | STARBOARD nadir..outer``.
    Water-column samples are left at zero because no ground correction exists
    for them.
    """

    ground = np.asarray(ground_display, dtype=float)
    depths = np.asarray(depth_by_channel, dtype=float)
    if ground.ndim != 2 or ground.shape[1] % 2:
        raise ValueError("ground waterfall must contain two equal-width channels")
    ping_count, total_width = ground.shape
    channel_width = total_width // 2
    if depths.shape != (2, ping_count):
        raise ValueError("depth information must have shape [2, ping_count]")

    result = np.zeros_like(ground, dtype=float)
    slant_samples = np.arange(channel_width)
    for ping in range(ping_count):
        for channel in (0, 1):
            depth = float(depths[channel, ping])
            if not math.isfinite(depth):
                continue
            valid = slant_samples > max(0.0, depth)
            if not np.any(valid):
                continue
            slant = slant_samples[valid]
            ground_sample = np.rint(
                np.sqrt(np.maximum(slant.astype(float) ** 2 - depth**2, 0.0))
            ).astype(int)
            ground_sample = np.clip(ground_sample, 0, channel_width - 1)
            if channel == 0:
                result[ping, channel_width - 1 - slant] = ground[
                    ping, channel_width - 1 - ground_sample
                ]
            else:
                result[ping, channel_width + slant] = ground[
                    ping, channel_width + ground_sample
                ]
    return result


class BuiltInGainProcessor:
    """Run existing SidescanPreprocessor operations on an isolated data copy."""

    def __init__(self, preprocessor, raw_logical_waterfall: np.ndarray):
        self.preprocessor = preprocessor
        self.raw = np.asarray(raw_logical_waterfall, dtype=float)

    def process(
        self,
        request: BuiltInGainRequest,
        progress: Callable[[int, str], None] | None = None,
    ) -> BuiltInGainResult:
        request = BuiltInGainRequest(**{
            field: getattr(request, field)
            for field in request.__dataclass_fields__
        })
        notify = progress or (lambda percent, text: None)
        notify(2, "Preparing processing data")

        show_ground_range = (
            request.slant_range_correction
            or request.mode is BuiltInGainMode.SLANT
        )
        if request.mode is BuiltInGainMode.RAW and not show_ground_range:
            display = np.clip(
                np.nan_to_num(self.raw, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        else:
            processor = self._processing_copy()
            notify(10, "Applying slant-range correction")
            processor.slant_range_correction(
                active_interpolation=True,
                nadir_angle=request.nadir_angle,
                use_intern_altitude=request.use_internal_altitude,
                active_mult_slant_range_resampling=False,
            )
            ground = processor.slant_corrected_mat
            if request.mode is BuiltInGainMode.BAC:
                notify(45, "Estimating and applying BAC")
                processor.apply_beam_pattern_correction(
                    angle_num=request.bac_angle_count
                )
                if request.energy_normalization:
                    notify(70, "Applying energy normalization")
                    processor.apply_energy_normalization()
                ground = np.hstack(
                    (
                        np.fliplr(processor.sonar_data_proc[0]),
                        processor.sonar_data_proc[1],
                    )
                )
            elif request.mode is BuiltInGainMode.EGN:
                if not request.egn_table_path.is_file():
                    raise FileNotFoundError(
                        f"EGN table does not exist: {request.egn_table_path}"
                    )
                self._validate_egn_table(request.egn_table_path)
                notify(45, "Applying EGN table")
                processor.do_EGN_correction(request.egn_table_path)
                ground = processor.egn_corrected_mat

            if show_ground_range:
                notify(78, "Removing water column")
                display = ground
            else:
                notify(78, "Restoring acoustic sample positions")
                display = ground_to_slant_display(ground, processor.dep_info)
            display = normalize_processed_waterfall(display)

        stages = [request.mode.value]
        if show_ground_range and request.mode is not BuiltInGainMode.SLANT:
            stages.append("slant-range-corrected")
        if request.mode is BuiltInGainMode.BAC and request.energy_normalization:
            stages.append("energy-normalized")
        if request.destripe:
            notify(83, "Applying destripe filter")
            display = destripe_waterfall(display)
            stages.append("destripe")
        if request.convert_db:
            notify(86, "Converting display to dB")
            converted = np.array(display, dtype=float, copy=True)
            if np.any(converted > 0):
                display = normalize_processed_waterfall(convert_to_dB(converted))
            stages.append("dB")
        if request.clahe:
            notify(92, "Applying CLAHE")
            if np.any(display > 0):
                display = hist_equalization(np.array(display, copy=True))
            stages.append("CLAHE")

        notify(100, "Gain processing ready")
        return BuiltInGainResult(
            display_data=np.clip(
                np.nan_to_num(display, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            ),
            pipeline_description="sidescantools-built-in-v1|" + "|".join(stages),
        )

    def _processing_copy(self):
        processor = copy.copy(self.preprocessor)
        processor.sonar_data_proc = np.array(
            self.preprocessor.sonar_data_proc, dtype=float, copy=True
        )
        for name in (
            "portside_bottom_dist",
            "starboard_bottom_dist",
            "napari_portside_bottom",
            "napari_starboard_bottom",
        ):
            if hasattr(self.preprocessor, name):
                setattr(
                    processor,
                    name,
                    np.array(getattr(self.preprocessor, name), copy=True),
                )
        return processor

    @staticmethod
    def _validate_egn_table(path: Path) -> None:
        required = {
            "egn_table",
            "egn_hit_cnt",
            "angle_range",
            "angle_num",
            "angle_stepsize",
            "ping_len",
            "r_size",
            "r_reduc_factor",
        }
        with np.load(path) as table:
            missing = sorted(required.difference(table.files))
        if missing:
            raise ValueError(
                "Invalid EGN table; missing " + ", ".join(missing)
            )
