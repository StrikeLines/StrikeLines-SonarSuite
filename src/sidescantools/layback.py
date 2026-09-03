"""Resolve embedded and user-supplied towfish layback consistently."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from sidescantools.swath_geometry import GeometrySettings


@dataclass(frozen=True, slots=True)
class TowDataSummary:
    recorded_layback_m: float | None
    recorded_cable_out_m: float | None


def _positive_median(values) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    valid = array[np.isfinite(array) & (array > 0)]
    return float(np.median(valid)) if valid.size else None


def summarize_tow_data(sidescan_file) -> TowDataSummary:
    """Summarize per-ping metadata as stable file-level median values."""

    return TowDataSummary(
        recorded_layback_m=_positive_median(
            getattr(sidescan_file, "layback_m", None)
        ),
        recorded_cable_out_m=_positive_median(
            getattr(sidescan_file, "cable_out_m", None)
        ),
    )


def resolve_geometry_layback(
    base: GeometrySettings,
    tow_data: TowDataSummary,
    *,
    manual_layback_m: float | None = None,
) -> tuple[GeometrySettings, str]:
    """Apply manual/file tow data using one deterministic precedence order."""

    cable_out_m = (
        tow_data.recorded_cable_out_m
        if tow_data.recorded_cable_out_m is not None
        else base.cable_out_m
    )
    if manual_layback_m is not None:
        value = float(manual_layback_m)
        if not math.isfinite(value) or value < 0:
            raise ValueError("manual layback must be finite and nonnegative")
        return (
            replace(base, cable_out_m=cable_out_m, layback_m=value),
            "Manual layback override",
        )
    if tow_data.recorded_layback_m is not None:
        return (
            replace(
                base,
                cable_out_m=cable_out_m,
                layback_m=tow_data.recorded_layback_m,
            ),
            "Recorded layback from sonar file",
        )
    if tow_data.recorded_cable_out_m is not None:
        return (
            replace(base, cable_out_m=tow_data.recorded_cable_out_m, layback_m=None),
            "Derived from recorded cable out at 45°",
        )
    if base.cable_out_m > 0:
        return base, "Derived from command-line cable out at 45°"
    return base, "No recorded layback or cable out"
