"""Characterization test: the vectorized slant_range_correction() must match
the pre-vectorization implementation bit-for-bit on the same input.

The golden fixtures in tests/fixtures/ were captured by running the OLD,
unvectorized per-sample-loop implementation (see the case construction below,
which must stay in sync with how the fixtures were generated) before it was
replaced with a vectorized flat-bottom projection for performance.
"""

import unittest
from pathlib import Path

import numpy as np

from sidescantools.sidescan_preproc import SidescanPreprocessor

FIXTURES = Path(__file__).parent / "fixtures"


class SyntheticSidescanFile:
    def __init__(self, num_ping, ping_len, seed=42):
        generator = np.random.default_rng(seed)
        self.data = generator.integers(
            1, 1000, size=(2, num_ping, ping_len), dtype=np.int16
        )
        self.ping_len = ping_len
        self.num_ping = num_ping
        self.slant_range = np.full((2, num_ping), 50.0)
        self.longitude = np.zeros(num_ping)
        self.latitude = np.zeros(num_ping)


def build_case(num_ping, ping_len, depths_port, depths_star, seed=42):
    source = SyntheticSidescanFile(num_ping, ping_len, seed=seed)
    preproc = SidescanPreprocessor(source, chunk_size=num_ping, downsampling_factor=1)
    preproc.portside_bottom_dist = np.array(depths_port, dtype=int)
    preproc.starboard_bottom_dist = np.array(depths_star, dtype=int)
    return preproc


class SlantRangeCorrectionParityTests(unittest.TestCase):
    # Deliberately covers: depth=0 (no blanking at all), depth=1, a typical
    # mid-ping depth, depth on the very last sample, and small depths that
    # force many samples to collide onto the same early ground-range index
    # -- exactly the case where "last dep_idx wins" ordering matters.
    ping_len = 40
    depths_port = [0, 1, 15, ping_len - 1, 3, 20]
    depths_star = [0, 2, 18, ping_len - 2, 4, 25]

    def test_matches_legacy_output_with_interpolation(self):
        preproc = build_case(
            len(self.depths_port), self.ping_len, self.depths_port, self.depths_star
        )
        preproc.slant_range_correction(
            active_interpolation=True,
            nadir_angle=0,
            use_intern_altitude=False,
            active_mult_slant_range_resampling=False,
        )

        golden = np.load(FIXTURES / "slant_range_correction_golden.npz")
        np.testing.assert_array_equal(preproc.sonar_data_proc[0], golden["channel_0"])
        np.testing.assert_array_equal(preproc.sonar_data_proc[1], golden["channel_1"])

    def test_matches_legacy_output_without_interpolation(self):
        preproc = build_case(
            len(self.depths_port), self.ping_len, self.depths_port, self.depths_star
        )
        preproc.slant_range_correction(
            active_interpolation=False,
            nadir_angle=0,
            use_intern_altitude=False,
            active_mult_slant_range_resampling=False,
        )

        golden = np.load(FIXTURES / "slant_range_correction_golden_no_interp.npz")
        np.testing.assert_array_equal(preproc.sonar_data_proc[0], golden["channel_0"])
        np.testing.assert_array_equal(preproc.sonar_data_proc[1], golden["channel_1"])

    def test_depth_zero_places_every_sample_without_blanking(self):
        # depth=0 means no sample is excluded near the transducer; dep_idx=0
        # is skipped (dep_idx must be > depth), but every dep_idx from 1
        # upward should still land somewhere.
        preproc = build_case(1, self.ping_len, [0], [0])
        preproc.slant_range_correction(
            active_interpolation=False,
            nadir_angle=0,
            use_intern_altitude=False,
            active_mult_slant_range_resampling=False,
        )
        # With no blanking, essentially the whole ping should be filled
        # (a few high-index samples can land out of bounds and stay NaN).
        nan_fraction = np.isnan(preproc.sonar_data_proc[1][0]).mean()
        self.assertLess(nan_fraction, 0.5)

    def test_depth_at_last_sample_blanks_almost_everything(self):
        # depth == ping_len - 1 means dep_idx > depth has no valid values at
        # all (the loop's dep_idx range is empty), so the whole ping stays
        # NaN before interpolation. Effective depth is starboard_bottom_dist
        # directly, but ping_len - portside_bottom_dist for port (port data
        # is mirrored before this projection), so drive each side with the
        # value that actually produces depth == ping_len - 1 for that side.
        preproc = build_case(1, self.ping_len, [1], [self.ping_len - 1])
        preproc.slant_range_correction(
            active_interpolation=False,
            nadir_angle=0,
            use_intern_altitude=False,
            active_mult_slant_range_resampling=False,
        )
        self.assertTrue(np.isnan(preproc.sonar_data_proc[0][0]).all())
        self.assertTrue(np.isnan(preproc.sonar_data_proc[1][0]).all())


if __name__ == "__main__":
    unittest.main()
