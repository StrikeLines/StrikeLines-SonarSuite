import unittest

import numpy as np
from scipy import signal

from sidescantools.sidescan_preproc import SidescanPreprocessor


class SyntheticSidescanFile:
    def __init__(self):
        generator = np.random.default_rng(42)
        self.data = generator.integers(1, 1000, size=(2, 11, 128), dtype=np.int16)
        self.ping_len = self.data.shape[2]


class PreprocessorDownsamplingTests(unittest.TestCase):
    def test_chunked_decimation_matches_legacy_bulk_operation(self):
        source = SyntheticSidescanFile()
        expected = signal.decimate(source.data.astype(float), 4, axis=2)
        expected = np.clip(expected, np.min(source.data), None)

        preprocessor = SidescanPreprocessor(
            source,
            chunk_size=3,
            downsampling_factor=4,
        )

        np.testing.assert_allclose(preprocessor.sonar_data_proc, expected)
        self.assertEqual(preprocessor.ping_len, 32)
        self.assertEqual(preprocessor.num_chunk, 4)

    def test_bottom_edge_tracking_accepts_a_clicked_start_position(self):
        edges = np.zeros((3, 20), dtype=bool)
        edges[:, 12] = True

        result = SidescanPreprocessor.edges_to_bottom_dist(
            None,
            edges,
            threshold_bin=0.5,
            data_is_port_side=False,
            click_pos=10,
            dist_at_ends=2,
        )

        np.testing.assert_array_equal(result, [12, 12, 12])

    def test_bottom_edge_tracking_handles_an_empty_chunk(self):
        result = SidescanPreprocessor.edges_to_bottom_dist(
            None,
            np.zeros((0, 20), dtype=bool),
            threshold_bin=0.5,
            data_is_port_side=False,
            dist_at_ends=2,
        )

        self.assertEqual(result.size, 0)


if __name__ == "__main__":
    unittest.main()
