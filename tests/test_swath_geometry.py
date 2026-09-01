import unittest

import numpy as np

from sidescantools.contact_model import Channel
from sidescantools.swath_geometry import (
    GeometrySettings,
    GeometryUnavailable,
    SwathGeometry,
)


class GeometrySettingsTests(unittest.TestCase):
    def test_serialization_and_hash_are_stable(self):
        first = GeometrySettings(60, cable_out_m=10, x_offset_m=2, y_offset_m=-1)
        second = GeometrySettings(
            vertical_beam_angle=60.0,
            y_offset_m=-1.0,
            x_offset_m=2.0,
            cable_out_m=10.0,
        )

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.settings_hash, second.settings_hash)
        self.assertEqual(len(first.settings_hash), 64)

    def test_coordinate_setting_change_changes_hash(self):
        original = GeometrySettings(60, cable_out_m=10)
        changed = GeometrySettings(60, cable_out_m=11)

        self.assertNotEqual(original.settings_hash, changed.settings_hash)


class SwathGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = SwathGeometry(
            channel=Channel.PORT,
            sample_count=3,
            valid_ping_mask=np.array([True, False, True]),
            nadir_lon=np.array([10.0, np.nan, 30.0]),
            nadir_lat=np.array([1.0, np.nan, 3.0]),
            outer_lon=np.array([12.0, np.nan, 34.0]),
            outer_lat=np.array([2.0, np.nan, 5.0]),
            slant_range_m=np.array([20.0, np.nan, 40.0]),
            ground_range_m=np.array([15.0, np.nan, 30.0]),
            geometry_settings=GeometrySettings(60),
        )

    def test_fraction_lookup_interpolates_nadir_to_outer(self):
        self.assertEqual(self.geometry.coordinate_for_fraction(0, 0), (10.0, 1.0))
        self.assertEqual(self.geometry.coordinate_for_fraction(0, 0.5), (11.0, 1.5))
        self.assertEqual(self.geometry.coordinate_for_fraction(0, 1), (12.0, 2.0))

    def test_sample_lookup_delegates_to_same_fraction_contract(self):
        self.assertEqual(self.geometry.coordinate_for_sample(2, 1), (32.0, 4.0))
        self.assertEqual(
            self.geometry.coordinate_for_sample(2, 1, sample_count=5),
            self.geometry.coordinate_for_fraction(2, 0.25),
        )

    def test_invalid_original_ping_is_not_compacted_to_another_row(self):
        with self.assertRaisesRegex(GeometryUnavailable, "unavailable"):
            self.geometry.coordinate_for_fraction(1, 0.5)

    def test_bulk_coordinates_keep_legacy_ping_then_sample_order(self):
        expected = np.array(
            [
                [10.0, 1.0],
                [11.0, 1.5],
                [12.0, 2.0],
                [30.0, 3.0],
                [32.0, 4.0],
                [34.0, 5.0],
            ]
        )

        np.testing.assert_allclose(
            self.geometry.coordinates_for_all_samples(), expected
        )

    def test_prepared_arrays_cannot_be_mutated_after_validation(self):
        with self.assertRaises(ValueError):
            self.geometry.nadir_lon[0] = 99


if __name__ == "__main__":
    unittest.main()
