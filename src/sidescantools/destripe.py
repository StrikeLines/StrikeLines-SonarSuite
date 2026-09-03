"""Fast suppression of ping-wise sidescan brightness stripes."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter


DEFAULT_BASELINE_WINDOW_PINGS = 51
DEFAULT_MAX_CORRECTION_DB = 6.0


def destripe_waterfall(
    waterfall: np.ndarray,
    *,
    baseline_window_pings: int = DEFAULT_BASELINE_WINDOW_PINGS,
    max_correction_db: float = DEFAULT_MAX_CORRECTION_DB,
) -> np.ndarray:
    """Suppress roll-related horizontal stripes without flattening the swath.

    A towed sonar's roll can make a whole port or starboard ping brighter or
    darker.  Estimate that per-ping gain robustly in log-amplitude space and
    compare it with a median-filtered along-track baseline.  Applying one
    scalar correction per side and ping preserves the across-track texture
    and TVG profile while removing abrupt row-to-row gain changes.
    """

    source = np.asarray(waterfall, dtype=float)
    if source.ndim != 2 or source.shape[1] < 4 or source.shape[1] % 2:
        raise ValueError("waterfall must contain two equal-width channels")
    if baseline_window_pings < 3:
        raise ValueError("destripe baseline window must be at least 3 pings")
    if not math.isfinite(float(max_correction_db)) or max_correction_db <= 0:
        raise ValueError("maximum destripe correction must be positive and finite")

    output = np.clip(
        np.nan_to_num(source, nan=0.0, posinf=1.0, neginf=0.0, copy=True),
        0.0,
        1.0,
    )
    ping_count = output.shape[0]
    if ping_count < 3:
        return output

    window = min(int(baseline_window_pings), ping_count)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)
    channel_width = output.shape[1] // 2

    for start, stop in ((0, channel_width), (channel_width, 2 * channel_width)):
        side = output[:, start:stop]
        usable = np.isfinite(side) & (side > 0.0)
        minimum_samples = min(
            channel_width, max(2, int(math.ceil(channel_width * 0.05)))
        )
        valid_rows = np.count_nonzero(usable, axis=1) >= minimum_samples
        if np.count_nonzero(valid_rows) < 3:
            continue

        # A fixed tiny floor only protects log10; zeros remain zero after the
        # multiplicative correction and do not influence the row estimate.
        log_amplitude = np.full(side.shape, np.nan, dtype=float)
        log_amplitude[usable] = 20.0 * np.log10(
            np.maximum(side[usable], 1e-8)
        )
        row_level_db = np.full(ping_count, np.nan, dtype=float)
        row_level_db[valid_rows] = np.nanmedian(
            log_amplitude[valid_rows], axis=1
        )

        row_indices = np.arange(ping_count)
        filled_level_db = np.interp(
            row_indices,
            row_indices[valid_rows],
            row_level_db[valid_rows],
        )
        baseline_db = median_filter(filled_level_db, size=window, mode="reflect")
        correction_db = np.clip(
            baseline_db - filled_level_db,
            -float(max_correction_db),
            float(max_correction_db),
        )
        correction_db[~valid_rows] = 0.0
        side *= np.power(10.0, correction_db / 20.0)[:, None]

    return np.clip(output, 0.0, 1.0)
