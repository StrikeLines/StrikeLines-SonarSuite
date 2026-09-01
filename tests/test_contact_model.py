import unittest

from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactValidationError,
)


class ContactModelTests(unittest.TestCase):
    def test_anchor_coerces_channel_index_to_enum(self):
        anchor = ContactAnchor(1, 0, 100, 0, 5, 0.5, 0, 0, 5)

        self.assertIs(anchor.channel, Channel.PORT)
        self.assertEqual(anchor.channel.label, "port")

    def test_anchor_rejects_invalid_fraction(self):
        with self.assertRaisesRegex(ContactValidationError, "sample_fraction"):
            ContactAnchor(1, 0, 100, Channel.PORT, 5, 1.01, 0, 0, 5)

    def test_anchor_is_immutable(self):
        anchor = ContactAnchor(1, 0, 100, Channel.PORT, 5, 0.5, 0, 0, 5)

        with self.assertRaises((AttributeError, TypeError)):
            anchor.global_ping_index = 2

    def test_coordinate_accepts_wgs84_bounds_and_nullable_ranges(self):
        coordinate = ContactCoordinate(-180, 90, None, None, 1)

        self.assertEqual(coordinate.longitude, -180.0)
        self.assertEqual(coordinate.latitude, 90.0)

    def test_coordinate_rejects_out_of_bounds_wgs84(self):
        with self.assertRaisesRegex(ContactValidationError, "longitude"):
            ContactCoordinate(180.1, 0, None, None, 1)
        with self.assertRaisesRegex(ContactValidationError, "latitude"):
            ContactCoordinate(0, -90.1, None, None, 1)

    def test_coordinate_rejects_unknown_range_encoded_as_negative(self):
        with self.assertRaisesRegex(ContactValidationError, "slant_range_m"):
            ContactCoordinate(0, 0, -1, None, 1)


if __name__ == "__main__":
    unittest.main()
