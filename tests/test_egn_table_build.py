import numpy as np
import pytest

from sidescantools.egn_table_build import (
    fill_missing_slant_ranges,
    generate_egn_table_from_infos,
)


def test_fill_missing_slant_ranges_uses_previous_value_without_mutating_input():
    source = np.array([[0.0, 10.0, 0.0, 12.0, 0.0], [20.0, 0.0, 22.0, 0.0, 24.0]])

    result = fill_missing_slant_ranges(source)

    np.testing.assert_array_equal(result[0], [10.0, 10.0, 10.0, 12.0, 12.0])
    np.testing.assert_array_equal(result[1], [20.0, 20.0, 22.0, 22.0, 24.0])
    np.testing.assert_array_equal(source[0], [0.0, 10.0, 0.0, 12.0, 0.0])


def test_generate_egn_table_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="At least one EGN info file"):
        generate_egn_table_from_infos([], tmp_path / "table.npz")
