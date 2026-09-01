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


if __name__ == "__main__":
    unittest.main()
