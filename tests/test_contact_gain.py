from pathlib import Path

import numpy as np
import pytest

from sidescantools.contact_gain import (
    BuiltInGainMode,
    BuiltInGainRequest,
    ground_to_slant_display,
    normalize_processed_waterfall,
)
from sidescantools.sidescan_preproc import SidescanPreprocessor


def test_egn_requires_a_table_path():
    with pytest.raises(ValueError, match="requires an EGN table"):
        BuiltInGainRequest(mode=BuiltInGainMode.EGN)


def test_processed_normalization_is_finite_and_robust_to_outlier():
    values = np.arange(100, dtype=float).reshape(10, 10)
    values[0, 0] = np.nan
    values[-1, -1] = 1_000_000

    result = normalize_processed_waterfall(values)

    assert np.all(np.isfinite(result))
    assert np.min(result) == 0.0
    assert np.max(result) == 1.0


def test_ground_to_slant_preserves_display_orientation_and_water_column():
    ground = np.tile(np.arange(8, dtype=float), (1, 1))
    depths = np.array([[1.0], [1.0]])

    result = ground_to_slant_display(ground, depths)

    # Slant samples 0 and 1 are water column and remain zero on both sides.
    assert result[0, 3] == 0.0
    assert result[0, 2] == 0.0
    assert result[0, 4] == 0.0
    assert result[0, 5] == 0.0
    # Port remains outer-to-nadir; starboard remains nadir-to-outer.
    assert result[0, 1] == ground[0, 1]
    assert result[0, 6] == ground[0, 6]


def test_request_normalizes_path_type():
    request = BuiltInGainRequest(
        mode="egn", egn_table_path=Path("table.npz")
    )

    assert request.mode is BuiltInGainMode.EGN
    assert request.egn_table_path == Path("table.npz")


def test_egn_validation_reports_missing_fields(tmp_path: Path):
    path = tmp_path / "invalid.npz"
    np.savez(path, egn_table=np.ones((2, 2)))

    from sidescantools.contact_gain import BuiltInGainProcessor

    with pytest.raises(ValueError, match="missing"):
        BuiltInGainProcessor._validate_egn_table(path)


def test_builtin_bac_corrects_both_port_and_starboard_channels():
    processor = object.__new__(SidescanPreprocessor)
    processor.ping_len = 8
    processor.sonar_data_proc = np.stack(
        (
            np.full((4, 8), 2.0, dtype=float),
            np.full((4, 8), 4.0, dtype=float),
        )
    )
    processor.dep_info = np.full((2, 4), 2.0, dtype=float)
    original_port = processor.sonar_data_proc[0].copy()
    original_starboard = processor.sonar_data_proc[1].copy()

    processor.apply_beam_pattern_correction(angle_num=72)

    assert not np.array_equal(processor.sonar_data_proc[0], original_port)
    assert not np.array_equal(
        processor.sonar_data_proc[1], original_starboard
    )
    assert np.all(np.isfinite(processor.sonar_data_proc))
