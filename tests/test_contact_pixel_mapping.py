import unittest

from sidescantools.contact_model import Channel
from sidescantools.contact_picker import (
    InvalidContactPixel,
    anchor_from_display_pixel,
    display_position_for_anchor,
    source_array_sample_for_anchor,
)


class ContactPixelMappingTests(unittest.TestCase):
    def make_anchor(self, *, chunk=0, local_ping=0, display_x=0, **overrides):
        arguments = {
            "source_file_id": 1,
            "ping_number": 10_000 + chunk * 4 + local_ping,
            "chunk_index": chunk,
            "local_ping_index": local_ping,
            "display_x": display_x,
            "chunk_size": 4,
            "display_channel_width": 5,
            "source_ping_count": 10,
            "source_sample_count": 9,
        }
        arguments.update(overrides)
        return anchor_from_display_pixel(**arguments)

    def test_port_outer_and_nadir_orientation(self):
        outer = self.make_anchor(display_x=0)
        nadir = self.make_anchor(display_x=4)

        self.assertIs(outer.channel, Channel.PORT)
        self.assertEqual(outer.sample_fraction, 1.0)
        self.assertEqual(outer.source_sample_index, 8)
        self.assertEqual(nadir.sample_fraction, 0.0)
        self.assertEqual(nadir.source_sample_index, 0)

    def test_starboard_nadir_and_outer_orientation(self):
        nadir = self.make_anchor(display_x=5)
        outer = self.make_anchor(display_x=9)

        self.assertIs(nadir.channel, Channel.STARBOARD)
        self.assertEqual(nadir.sample_fraction, 0.0)
        self.assertEqual(nadir.source_sample_index, 0)
        self.assertEqual(outer.sample_fraction, 1.0)
        self.assertEqual(outer.source_sample_index, 8)

    def test_exact_channel_boundary_is_starboard_nadir(self):
        anchor = self.make_anchor(display_x=5)

        self.assertIs(anchor.channel, Channel.STARBOARD)
        self.assertEqual(anchor.sample_fraction, 0.0)

    def test_downsampled_mapping_preserves_endpoints(self):
        port_outer = self.make_anchor(
            display_x=0, display_channel_width=4, source_sample_count=10
        )
        starboard_outer = self.make_anchor(
            display_x=7, display_channel_width=4, source_sample_count=10
        )

        self.assertEqual(port_outer.source_sample_index, 9)
        self.assertEqual(starboard_outer.source_sample_index, 9)

    def test_global_ping_uses_chunk_and_local_index(self):
        anchor = self.make_anchor(chunk=2, local_ping=1, display_x=5)

        self.assertEqual(anchor.global_ping_index, 9)
        self.assertEqual(anchor.display_chunk, 2)
        self.assertEqual(anchor.display_ping_index, 1)

    def test_first_padded_row_in_last_chunk_is_rejected(self):
        with self.assertRaisesRegex(InvalidContactPixel, "padding"):
            self.make_anchor(chunk=2, local_ping=2, display_x=5)

    def test_negative_and_out_of_range_coordinates_are_rejected(self):
        invalid_arguments = (
            {"chunk": -1},
            {"local_ping": -1},
            {"local_ping": 4},
            {"display_x": -1},
            {"display_x": 10},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(InvalidContactPixel):
                    self.make_anchor(**arguments)

    def test_channel_width_must_support_a_fraction(self):
        with self.assertRaisesRegex(InvalidContactPixel, "display_channel_width"):
            self.make_anchor(display_channel_width=1)

    def test_positions_must_already_be_integer_data_coordinates(self):
        with self.assertRaisesRegex(InvalidContactPixel, "display_x"):
            self.make_anchor(display_x=1.5)

    def test_marker_reconstruction_uses_current_chunk_and_display_width(self):
        anchor = self.make_anchor(chunk=1, local_ping=3, display_x=2)

        marker = display_position_for_anchor(
            anchor, chunk_size=3, display_channel_width=9
        )

        self.assertEqual(marker, (2, 1, 4))

    def test_marker_round_trip_is_exact_at_same_resolution(self):
        for display_x in range(10):
            with self.subTest(display_x=display_x):
                anchor = self.make_anchor(display_x=display_x)
                marker = display_position_for_anchor(
                    anchor, chunk_size=4, display_channel_width=5
                )
                self.assertEqual(marker, (0, 0, display_x))

    def test_normalized_sample_maps_back_to_source_array_orientation(self):
        port = self.make_anchor(display_x=1)
        starboard = self.make_anchor(display_x=8)

        self.assertEqual(
            source_array_sample_for_anchor(port, source_sample_count=9), 2
        )
        self.assertEqual(
            source_array_sample_for_anchor(starboard, source_sample_count=9), 6
        )


if __name__ == "__main__":
    unittest.main()
