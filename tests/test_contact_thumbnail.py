from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from sidescantools.contact_picker import anchor_from_display_pixel
from sidescantools.contact_thumbnail import ContactThumbnailExtractor


class SyntheticSidescanFile:
    num_ping = 7
    ping_len = 4


class SyntheticPreprocessor:
    chunk_size = 3
    ping_len = 4

    def __init__(self):
        logical = np.arange(9 * 8, dtype=float).reshape(9, 8)
        self.napari_fullmat = logical.reshape(3, 3, 8)


class ContactThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SyntheticPreprocessor()
        self.sidescan_file = SyntheticSidescanFile()
        self.extractor = ContactThumbnailExtractor(
            preprocessor=self.preprocessor,
            sidescan_file=self.sidescan_file,
            ping_radius=2,
            sample_radius=2,
            crosshair_radius=1,
        )

    def anchor(self, *, global_ping, display_x):
        return anchor_from_display_pixel(
            source_file_id=1,
            ping_number=100 + global_ping,
            chunk_index=global_ping // 3,
            local_ping_index=global_ping % 3,
            display_x=display_x,
            chunk_size=3,
            display_channel_width=4,
            source_ping_count=7,
            source_sample_count=4,
        )

    @staticmethod
    def decode(thumbnail):
        return np.asarray(Image.open(BytesIO(thumbnail.image_bytes)).convert("RGB"))

    def test_center_crop_spans_chunk_boundary(self):
        thumbnail = self.extractor(self.anchor(global_ping=3, display_x=5))

        self.assertEqual((thumbnail.width_px, thumbnail.height_px), (5, 5))
        image = self.decode(thumbnail)
        np.testing.assert_array_equal(image[2, 2], (255, 0, 0))

    def test_file_and_image_edges_are_clipped_cleanly(self):
        first = self.extractor(self.anchor(global_ping=0, display_x=0))
        last = self.extractor(self.anchor(global_ping=6, display_x=7))

        self.assertEqual((first.width_px, first.height_px), (3, 3))
        self.assertEqual((last.width_px, last.height_px), (3, 3))
        np.testing.assert_array_equal(self.decode(first)[0, 0], (255, 0, 0))
        np.testing.assert_array_equal(self.decode(last)[2, 2], (255, 0, 0))

    def test_last_chunk_padding_is_excluded(self):
        thumbnail = self.extractor(self.anchor(global_ping=6, display_x=4))

        self.assertEqual(thumbnail.height_px, 3)

    def test_crop_can_cross_port_starboard_seam(self):
        thumbnail = self.extractor(self.anchor(global_ping=3, display_x=3))

        self.assertEqual(thumbnail.width_px, 5)
        self.assertEqual(self.decode(thumbnail).shape, (5, 5, 3))

    def test_png_encoding_is_deterministic(self):
        anchor = self.anchor(global_ping=3, display_x=5)

        first = self.extractor(anchor)
        second = self.extractor(anchor)

        self.assertEqual(first.image_bytes, second.image_bytes)
        self.assertTrue(first.image_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_active_display_provider_drives_thumbnail_and_pipeline_metadata(self):
        displayed = np.full((7, 8), 0.25, dtype=float)
        displayed[3, 5] = 1.0
        extractor = ContactThumbnailExtractor(
            preprocessor=self.preprocessor,
            sidescan_file=self.sidescan_file,
            ping_radius=1,
            sample_radius=1,
            crosshair_radius=0,
            logical_waterfall_provider=lambda: displayed,
            display_pipeline_provider=lambda: "active-bac|gain=6dB",
        )

        thumbnail = extractor(self.anchor(global_ping=3, display_x=5))

        self.assertEqual(thumbnail.display_pipeline, "active-bac|gain=6dB")
        self.assertEqual((thumbnail.width_px, thumbnail.height_px), (3, 3))
        np.testing.assert_array_equal(self.decode(thumbnail)[1, 1], (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
