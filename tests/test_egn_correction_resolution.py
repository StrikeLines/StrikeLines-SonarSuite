"""Tests for do_EGN_correction(): both a characterization test proving the
vectorized rewrite doesn't regress the one case that already worked (table
and data at the same resolution), and new coverage for the case that used to
crash outright -- applying a table built at one resolution to data at a
different resolution ("index N is out of bounds for axis 0 with size N").
"""

import unittest
from pathlib import Path

import numpy as np

from sidescantools.sidescan_preproc import SidescanPreprocessor

FIXTURES = Path(__file__).parent / "fixtures"


def build_preproc(ping_len, num_ping, slant_corrected_mat, depths):
    preproc = SidescanPreprocessor.__new__(SidescanPreprocessor)
    preproc.ping_len = ping_len
    preproc.slant_corrected_mat = slant_corrected_mat
    preproc.dep_info = [depths, depths]
    return preproc


def build_table(path, ping_len, egn_table, r_reduc_factor=1.0, angle_range=None):
    r_size, angle_num = egn_table.shape
    angle_range = angle_range if angle_range is not None else [-np.pi / 2, np.pi / 2]
    np.savez(
        path,
        egn_table=egn_table,
        egn_hit_cnt=np.ones_like(egn_table),
        angle_range=np.array(angle_range),
        angle_num=angle_num,
        angle_stepsize=(angle_range[1] - angle_range[0]) / angle_num,
        ping_len=ping_len,
        r_size=r_size,
        r_reduc_factor=r_reduc_factor,
        nadir_angle=0,
    )


class SameResolutionParityTests(unittest.TestCase):
    """table_ping_len == self.ping_len: the case that already worked before
    the fix, because the old code's ping_len-mixing bug is invisible when
    both values happen to coincide. Must still match exactly."""

    def test_matches_legacy_output(self):
        golden = np.load(FIXTURES / "egn_correction_golden_same_res.npz")
        table_path = FIXTURES / "egn_correction_table_same_res.npz"

        preproc = build_preproc(
            ping_len=10,
            num_ping=golden["slant_corrected_mat"].shape[0],
            slant_corrected_mat=golden["slant_corrected_mat"],
            depths=golden["dep_info"][0],
        )
        preproc.do_EGN_correction(table_path)

        np.testing.assert_allclose(
            preproc.egn_corrected_mat, golden["egn_corrected_mat"]
        )


class MismatchedResolutionTests(unittest.TestCase):
    """table_ping_len != self.ping_len: the case that used to raise
    'index N is out of bounds for axis 0 with size N' outright."""

    def test_does_not_crash_and_produces_correct_shape(self):
        # A table built at 100 samples/ping applied to data downsampled to
        # 30 samples/ping -- roughly the user's real 29296-vs-916 situation,
        # scaled down for a fast test.
        table_ping_len = 100
        self_ping_len = 30
        r_size, angle_num = 60, 12
        rng = np.random.default_rng(3)
        table_path = FIXTURES / "_tmp_mismatched_table.npz"
        build_table(table_path, table_ping_len, rng.random((r_size, angle_num)) + 0.1)

        num_ping = 5
        slant_corrected_mat = rng.random((num_ping, 2 * self_ping_len)) * 10.0
        depths = rng.integers(2, self_ping_len - 1, size=num_ping).astype(float)
        preproc = build_preproc(self_ping_len, num_ping, slant_corrected_mat, depths)

        preproc.do_EGN_correction(table_path)  # must not raise

        self.assertEqual(
            preproc.egn_corrected_mat.shape, (num_ping, 2 * self_ping_len)
        )
        table_path.unlink()

    def test_r_idx_is_rescaled_to_the_tables_build_resolution(self):
        # Hand-verified case: table built at ping_len=40, applied at
        # ping_len=10 -> resolution_scale = 40/10 = 4.0. One ping, depth=6,
        # r_reduc_factor=1, angle_num=8 over [-pi/2, pi/2] (stepsize=pi/8).
        #
        # For sample offset (index - ping_len) = -8, 0, +8 (array indices
        # 2, 10, 18 of a 20-wide ping): r = sqrt(offset**2 + depth**2) is,
        # respectively, 10, 6, 10 at the CURRENT resolution. The fix must
        # multiply r by resolution_scale (4.0) before binning, landing at
        # r_idx 40, 24, 40 -- not 10, 6, 10, which is what the pre-fix code
        # (or a fix that forgot the rescale) would have produced, and not
        # what would even fit in a table this small if the r_size were
        # sized for the *unscaled* case.
        table_ping_len = 40
        self_ping_len = 10
        r_size, angle_num = 50, 8
        depth = 6.0

        egn_table = np.zeros((r_size, angle_num))
        # Distinct, identifiable values at the three expected bins.
        egn_table[40, 2] = 2.0   # offset -8
        egn_table[24, 4] = 5.0   # offset  0
        egn_table[40, 6] = 10.0  # offset +8
        table_path = FIXTURES / "_tmp_hand_verified_table.npz"
        build_table(table_path, table_ping_len, egn_table, r_reduc_factor=1.0)

        slant_corrected_mat = np.full((1, 2 * self_ping_len), 100.0)
        preproc = build_preproc(
            self_ping_len, 1, slant_corrected_mat, np.array([depth])
        )
        preproc.do_EGN_correction(table_path)

        EPS = np.finfo(float).eps
        self.assertAlmostEqual(
            preproc.egn_corrected_mat[0, 2], 100.0 / (2.0 + EPS), places=6
        )
        self.assertAlmostEqual(
            preproc.egn_corrected_mat[0, 10], 100.0 / (5.0 + EPS), places=6
        )
        self.assertAlmostEqual(
            preproc.egn_corrected_mat[0, 18], 100.0 / (10.0 + EPS), places=6
        )
        table_path.unlink()

    def test_same_physical_range_maps_to_same_bin_at_two_different_resolutions(self):
        # The actual point of the fix: build one table, then confirm that a
        # ping at TWO different resolutions -- but describing the same
        # physical geometry (same depth-to-ping_len ratio) -- selects the
        # same table bin. This is the resolution-independence property the
        # user actually needs (build once per survey, apply at any zoom).
        table_ping_len = 40
        r_size, angle_num = 50, 8
        egn_table = np.zeros((r_size, angle_num))
        egn_table[24, 4] = 7.0  # the nadir bin, from the previous test's math
        table_path = FIXTURES / "_tmp_scale_invariance_table.npz"
        build_table(table_path, table_ping_len, egn_table, r_reduc_factor=1.0)

        EPS = np.finfo(float).eps

        # Resolution A: ping_len=10, depth=6 (depth/ping_len ratio = 0.6).
        preproc_a = build_preproc(10, 1, np.full((1, 20), 50.0), np.array([6.0]))
        preproc_a.do_EGN_correction(table_path)

        # Resolution B: ping_len=20, depth=12 -- same physical ratio (0.6),
        # just described at 2x the sample density.
        preproc_b = build_preproc(20, 1, np.full((1, 40), 50.0), np.array([12.0]))
        preproc_b.do_EGN_correction(table_path)

        # Both nadir samples must land on the same table bin (value 7.0) and
        # therefore produce the same corrected value.
        expected = 50.0 / (7.0 + EPS)
        self.assertAlmostEqual(preproc_a.egn_corrected_mat[0, 10], expected, places=6)
        self.assertAlmostEqual(preproc_b.egn_corrected_mat[0, 20], expected, places=6)
        table_path.unlink()


if __name__ == "__main__":
    unittest.main()
