import unittest

import numpy as np

from sidescantools.georef_thread import Georeferencer
from sidescantools.swath_geometry import GeometrySettings


class SyntheticSidescanFile:
    """Deterministic metadata fixture large enough for legacy smoothing windows."""

    def __init__(self):
        ping_count = 401
        sample_count = 7
        ping = np.arange(ping_count, dtype=float)

        self.packet_no = np.arange(50_000, 50_000 + ping_count)
        self.longitude = -70.0 + ping * 0.00001
        self.latitude = 40.0 + ping * 0.000005 + np.sin(ping / 35.0) * 0.00001
        self.longitude[[50, 250]] = 0.0
        self.latitude[[50, 250]] = 0.0
        self.sensor_heading = (350.0 + ping * 0.2) % 360.0
        self.slant_range = np.vstack(
            (30.0 + ping * 0.002, 32.0 + ping * 0.002)
        )
        self.data = np.zeros((2, ping_count, sample_count), dtype=float)


def prepared_georeferencer(channel):
    georeferencer = Georeferencer.__new__(Georeferencer)
    georeferencer.sidescan_file = SyntheticSidescanFile()
    georeferencer.channel = channel
    georeferencer.vertical_beam_angle = 60
    georeferencer.active_proc_data = False
    georeferencer.cable_out = 12.5
    georeferencer.x_offset = 1.25
    georeferencer.y_offset = -0.75
    georeferencer.geometry_settings = GeometrySettings(
        vertical_beam_angle=60,
        cable_out_m=12.5,
        x_offset_m=1.25,
        y_offset_m=-0.75,
    )
    georeferencer.swath_geometry = None
    georeferencer.PING = georeferencer.sidescan_file.packet_no
    georeferencer.LALO_OUTER = []
    georeferencer.prep_data()
    return georeferencer


class GeoreferencerCharacterizationTests(unittest.TestCase):
    @staticmethod
    def legacy_bulk_coordinates(georeferencer):
        nadir = georeferencer.LOLA_plt
        outer_lat, outer_lon = map(np.asarray, zip(*georeferencer.LALO_OUTER))
        width = georeferencer.sidescan_file.data.shape[2]
        longitude = [
            np.linspace(lon, lon_outer, width)
            for (lon, lon_outer) in zip(nadir[:, 0], outer_lon)
        ]
        latitude = [
            np.linspace(lat, lat_outer, width)
            for (lat, lat_outer) in zip(nadir[:, 1], outer_lat)
        ]
        return np.column_stack(
            (np.asarray(longitude).ravel(), np.asarray(latitude).ravel())
        )

    def test_legacy_nav_shape_and_invalid_ping_filtering(self):
        georeferencer = prepared_georeferencer(channel=0)

        self.assertEqual(georeferencer.nav.shape, (399 * 7, 2))
        self.assertTrue(np.all(np.isfinite(georeferencer.nav)))

    def test_constructor_accepts_already_loaded_file_and_geometry_settings(self):
        source = SyntheticSidescanFile()
        settings = GeometrySettings(
            vertical_beam_angle=55,
            cable_out_m=9,
            x_offset_m=2,
            y_offset_m=-3,
        )

        georeferencer = Georeferencer(
            filepath="synthetic.jsf",
            sidescan_file=source,
            geometry_settings=settings,
            output_folder=".",
        )

        self.assertIs(georeferencer.sidescan_file, source)
        self.assertIs(georeferencer.geometry_settings, settings)
        self.assertEqual(georeferencer.cable_out, 9)

    def test_channels_share_nadir_and_have_different_outer_coordinates(self):
        port = prepared_georeferencer(channel=0).nav.reshape(-1, 7, 2)
        starboard = prepared_georeferencer(channel=1).nav.reshape(-1, 7, 2)

        np.testing.assert_allclose(port[:, 0], starboard[:, 0], atol=1e-12)
        self.assertFalse(np.allclose(port[:, -1], starboard[:, -1]))

    def test_shared_geometry_bulk_output_matches_legacy_linspace_order(self):
        for channel in (0, 1):
            with self.subTest(channel=channel):
                georeferencer = prepared_georeferencer(channel)

                np.testing.assert_allclose(
                    georeferencer.nav,
                    self.legacy_bulk_coordinates(georeferencer),
                    rtol=0,
                    atol=1e-14,
                )

    def test_single_lookup_uses_original_ping_index_after_filtering(self):
        georeferencer = prepared_georeferencer(channel=0)
        valid_original_index = 251
        compact_index = int(
            np.count_nonzero(
                georeferencer.sidescan_file.longitude[:valid_original_index]
            )
        )

        coordinate = georeferencer.swath_geometry.coordinate_for_sample(
            valid_original_index, 3
        )

        np.testing.assert_allclose(
            coordinate,
            georeferencer.nav.reshape(-1, 7, 2)[compact_index, 3],
            rtol=0,
            atol=1e-14,
        )

    def test_forced_geometry_preparation_is_repeatable(self):
        georeferencer = prepared_georeferencer(channel=1)
        first = georeferencer.nav.copy()

        returned = georeferencer.prepare_swath_geometry(force=True)

        self.assertIs(returned, georeferencer.swath_geometry)
        np.testing.assert_array_equal(georeferencer.nav, first)

    def test_contact_only_preparation_does_not_allocate_bulk_navigation(self):
        georeferencer = Georeferencer.__new__(Georeferencer)
        georeferencer.sidescan_file = SyntheticSidescanFile()
        georeferencer.channel = 0
        georeferencer.vertical_beam_angle = 60
        georeferencer.active_proc_data = False
        georeferencer.cable_out = 12.5
        georeferencer.x_offset = 1.25
        georeferencer.y_offset = -0.75
        georeferencer.geometry_settings = GeometrySettings(60, 12.5, 1.25, -0.75)
        georeferencer.PING = georeferencer.sidescan_file.packet_no
        georeferencer.LALO_OUTER = []
        georeferencer.swath_geometry = None
        georeferencer.nav = []

        geometry = georeferencer.prepare_swath_geometry()

        self.assertEqual(geometry.ping_count, 401)
        self.assertEqual(georeferencer.nav, [])


if __name__ == "__main__":
    unittest.main()
