import unittest

import numpy as np
from scipy import signal

from sidescantools.sidescan_preproc import SidescanPreprocessor
from sidescantools.sidescan_preproc import resolve_downsampling_factor


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

    def test_reader_can_preserve_a_minimum_processed_sample_width(self):
        source = SyntheticSidescanFile()
        generator = np.random.default_rng(7)
        source.data = generator.integers(
            1, 1000, size=(2, 3, 2048), dtype=np.int16
        )
        source.ping_len = 2048
        source.reader_metadata = {
            "minimum_processed_samples_per_channel": 512
        }

        preprocessor = SidescanPreprocessor(
            source,
            chunk_size=3,
            downsampling_factor=32,
        )

        self.assertEqual(preprocessor.downsampling_factor, 4)
        self.assertEqual(preprocessor.ping_len, 512)
        self.assertEqual(preprocessor.sonar_data_proc.shape, (2, 3, 512))

    def test_target_resolution_selects_nearest_integer_factor(self):
        source = SyntheticSidescanFile()
        source.ping_len = 4096

        self.assertEqual(
            resolve_downsampling_factor(
                source, 32, target_samples_per_channel=1024
            ),
            4,
        )

        source.ping_len = 5000
        self.assertEqual(
            resolve_downsampling_factor(
                source, 32, target_samples_per_channel=1024
            ),
            5,
        )

        # Nearest output width is not always obtained by merely rounding the
        # ideal factor: 1500 / 1 is 1500, while 1500 / 2 is 750 and closer to
        # the requested 1024 samples.
        source.ping_len = 1500
        self.assertEqual(
            resolve_downsampling_factor(
                source, 32, target_samples_per_channel=1024
            ),
            2,
        )

    def test_target_resolution_supports_native_and_small_inputs(self):
        source = SyntheticSidescanFile()
        self.assertEqual(
            resolve_downsampling_factor(source, 32, target_samples_per_channel=0),
            1,
        )
        self.assertEqual(
            resolve_downsampling_factor(source, 32, target_samples_per_channel=512),
            1,
        )

    def test_reader_minimum_still_wins_over_lower_user_target(self):
        source = SyntheticSidescanFile()
        source.ping_len = 2048
        source.reader_metadata = {"minimum_processed_samples_per_channel": 512}

        self.assertEqual(
            resolve_downsampling_factor(source, 32, target_samples_per_channel=256),
            4,
        )

    def test_target_resolution_rejects_negative_values(self):
        source = SyntheticSidescanFile()
        with self.assertRaisesRegex(ValueError, "target_samples_per_channel"):
            resolve_downsampling_factor(source, 32, target_samples_per_channel=-1)

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
