"""Characterization test: accumulate_egn_bins() must match the pre-vectorization
per-sample Python loop it replaced in generate_egn_info(), bit-for-bit.

The golden fixture in tests/fixtures/egn_accumulation_golden.npz was produced
by running a verbatim copy of the OLD loop body (see
scripts in the commit that added this test) on hand-crafted r_idx/alpha_idx/
values arrays chosen to force every interesting case: duplicate bins that
must accumulate rather than overwrite, negative and >=size out-of-range
indices on both axes, and an exact-zero value that must be skipped.
"""

import unittest
from pathlib import Path

import numpy as np

from sidescantools.egn_table_build import accumulate_egn_bins

FIXTURES = Path(__file__).parent / "fixtures"


class EGNAccumulationParityTests(unittest.TestCase):
    def test_matches_legacy_loop_output(self):
        golden = np.load(FIXTURES / "egn_accumulation_golden.npz")
        r_size = int(golden["r_size"])
        angle_num = int(golden["angle_num"])

        egn_mat = np.zeros((r_size, angle_num))
        egn_hit_cnt = np.zeros((r_size, angle_num))
        accumulate_egn_bins(
            egn_mat,
            egn_hit_cnt,
            golden["r_idx"],
            golden["alpha_idx"],
            golden["values"],
            r_size,
            angle_num,
        )

        np.testing.assert_array_equal(egn_mat, golden["egn_mat"])
        np.testing.assert_array_equal(egn_hit_cnt, golden["egn_hit_cnt"])

    def test_duplicate_bins_accumulate_not_overwrite(self):
        egn_mat = np.zeros((3, 3))
        egn_hit_cnt = np.zeros((3, 3))
        # Three samples all landing on the same (1, 1) bin.
        r_idx = np.array([1, 1, 1])
        alpha_idx = np.array([1, 1, 1])
        values = np.array([2.0, 3.0, 4.0])

        accumulate_egn_bins(egn_mat, egn_hit_cnt, r_idx, alpha_idx, values, 3, 3)

        self.assertEqual(egn_mat[1, 1], 9.0)
        self.assertEqual(egn_hit_cnt[1, 1], 3)
        self.assertEqual(np.count_nonzero(egn_mat), 1)

    def test_zero_values_and_out_of_range_indices_are_skipped(self):
        egn_mat = np.zeros((4, 4))
        egn_hit_cnt = np.zeros((4, 4))
        r_idx = np.array([0, -1, 4, 2])
        alpha_idx = np.array([0, 0, 0, -1])
        values = np.array([0.0, 5.0, 5.0, 5.0])

        accumulate_egn_bins(egn_mat, egn_hit_cnt, r_idx, alpha_idx, values, 4, 4)

        self.assertEqual(egn_mat.sum(), 0.0)
        self.assertEqual(egn_hit_cnt.sum(), 0.0)

    def test_accumulates_into_pre_populated_arrays(self):
        # accumulate_egn_bins mutates in place and must add to, not replace,
        # whatever the caller already put there (generate_egn_info calls it
        # once per ping into the same egn_mat/egn_hit_cnt across the file).
        egn_mat = np.zeros((2, 2))
        egn_mat[0, 0] = 10.0
        egn_hit_cnt = np.zeros((2, 2))
        egn_hit_cnt[0, 0] = 1

        accumulate_egn_bins(
            egn_mat, egn_hit_cnt, np.array([0]), np.array([0]), np.array([1.5]), 2, 2
        )

        self.assertEqual(egn_mat[0, 0], 11.5)
        self.assertEqual(egn_hit_cnt[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
