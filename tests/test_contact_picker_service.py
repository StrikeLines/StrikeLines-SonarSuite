from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sidescantools.contact_model import Channel, CoordinateStatus
from sidescantools.contact_picker import ContactPickerService, InvalidContactPixel
from sidescantools.contact_store import ContactStore
from sidescantools.swath_geometry import GeometrySettings, SwathGeometry


class SyntheticSidescanFile:
    num_ping = 5
    ping_len = 5

    def __init__(self):
        self.packet_no = np.arange(100, 105)
        self.timestamp = [
            datetime(2026, 8, 26, 12, index, tzinfo=timezone.utc)
            for index in range(self.num_ping)
        ]
        self.data = np.zeros((2, self.num_ping, self.ping_len), dtype=float)
        self.data[0] = 100 + np.arange(self.ping_len)
        self.data[1] = 200 + np.arange(self.ping_len)


class SyntheticPreprocessor:
    chunk_size = 3
    ping_len = 3

    def __init__(self):
        self.napari_fullmat = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)


class ContactPickerServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ContactStore(
            Path(self.temporary_directory.name) / "contacts.sqlite"
        )
        self.source_file = SyntheticSidescanFile()
        self.preprocessor = SyntheticPreprocessor()
        self.source = self.store.register_source_file(
            Path(self.temporary_directory.name) / "synthetic.jsf",
            format="jsf",
            ping_count=self.source_file.num_ping,
            source_sample_count=self.source_file.ping_len,
        )
        self.settings = GeometrySettings(60)
        self.profile_id = self.store.get_or_create_geometry_profile(self.settings)
        valid = np.array([True, True, False, True, True])
        self.geometry = {
            Channel.PORT: self.make_geometry(
                Channel.PORT, valid, outer_lon=np.full(5, -70.01)
            ),
            Channel.STARBOARD: self.make_geometry(
                Channel.STARBOARD, valid, outer_lon=np.full(5, -69.99)
            ),
        }
        self.service = ContactPickerService(
            sidescan_file=self.source_file,
            preprocessor=self.preprocessor,
            source_file_id=self.source.id,
            geometry_profile_id=self.profile_id,
            geometry_by_channel=self.geometry,
            store=self.store,
        )

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def make_geometry(self, channel, valid, *, outer_lon):
        nan_for_invalid = np.where(valid, 40.0, np.nan)
        return SwathGeometry(
            channel=channel,
            sample_count=5,
            valid_ping_mask=valid,
            nadir_lon=np.where(valid, -70.0, np.nan),
            nadir_lat=nan_for_invalid,
            outer_lon=np.where(valid, outer_lon, np.nan),
            outer_lat=np.where(valid, 40.01, np.nan),
            slant_range_m=np.where(valid, 20.0, np.nan),
            ground_range_m=np.where(valid, 16.0, np.nan),
            geometry_settings=self.settings,
        )

    def test_port_pick_uses_shared_coordinate_and_inverse_source_orientation(self):
        result = self.service.pick_display_pixel(
            chunk_index=0, local_ping_index=0, display_x=0
        )

        contact = result.contact
        self.assertEqual(contact.coordinate_status, CoordinateStatus.VALID)
        self.assertIs(contact.draft.anchor.channel, Channel.PORT)
        self.assertEqual(contact.draft.anchor.sample_fraction, 1.0)
        self.assertEqual(contact.draft.coordinate.longitude, -70.01)
        self.assertEqual(contact.draft.intensity_source, 100.0)
        self.assertEqual(contact.draft.intensity_display, 0.0)
        self.assertEqual(contact.draft.name, "Target 0001")
        self.assertEqual(contact.draft.timestamp_basis, "utc")

    def test_starboard_pick_uses_near_to_far_source_sample_directly(self):
        result = self.service.pick_display_pixel(
            chunk_index=0, local_ping_index=1, display_x=5
        )

        self.assertIs(result.contact.draft.anchor.channel, Channel.STARBOARD)
        self.assertEqual(result.contact.draft.intensity_source, 204.0)
        self.assertEqual(result.contact.draft.coordinate.slant_range_m, 20.0)
        self.assertEqual(result.contact.draft.coordinate.ground_range_m, 16.0)

    def test_invalid_navigation_and_padding_create_no_rows(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.service.pick_display_pixel(
                chunk_index=0, local_ping_index=2, display_x=3
            )
        with self.assertRaisesRegex(InvalidContactPixel, "padding"):
            self.service.pick_display_pixel(
                chunk_index=1, local_ping_index=2, display_x=3
            )

        self.assertEqual(self.store.list_contacts(), [])

    def test_thumbnail_failure_is_nonfatal_and_reported(self):
        def fail_thumbnail(anchor):
            raise RuntimeError("encoder failed")

        service = ContactPickerService(
            sidescan_file=self.source_file,
            preprocessor=self.preprocessor,
            source_file_id=self.source.id,
            geometry_profile_id=self.profile_id,
            geometry_by_channel=self.geometry,
            store=self.store,
            thumbnail_factory=fail_thumbnail,
        )

        result = service.pick_display_pixel(
            chunk_index=0, local_ping_index=0, display_x=1
        )

        self.assertIn("encoder failed", result.thumbnail_warning)
        self.assertIsNone(self.store.get_thumbnail(result.contact.id))

    def test_display_gain_provider_is_recorded_without_changing_source_intensity(self):
        service = ContactPickerService(
            sidescan_file=self.source_file,
            preprocessor=self.preprocessor,
            source_file_id=self.source.id,
            geometry_profile_id=self.profile_id,
            geometry_by_channel=self.geometry,
            store=self.store,
            display_intensity_provider=lambda anchor: 0.625,
            display_pipeline=lambda: "qt-continuous-waterfall|gain=6dB",
        )

        result = service.pick_display_pixel(
            chunk_index=0, local_ping_index=0, display_x=0
        )

        self.assertEqual(result.contact.draft.intensity_source, 100.0)
        self.assertEqual(result.contact.draft.intensity_display, 0.625)
        self.assertEqual(
            result.contact.draft.intensity_pipeline,
            "qt-continuous-waterfall|gain=6dB",
        )

    def test_geometry_profile_mismatch_is_rejected_before_insert(self):
        other_profile = self.store.get_or_create_geometry_profile(
            GeometrySettings(60, cable_out_m=1)
        )
        service = ContactPickerService(
            sidescan_file=self.source_file,
            preprocessor=self.preprocessor,
            source_file_id=self.source.id,
            geometry_profile_id=other_profile,
            geometry_by_channel=self.geometry,
            store=self.store,
        )

        with self.assertRaisesRegex(InvalidContactPixel, "active profile"):
            service.pick_display_pixel(
                chunk_index=0, local_ping_index=0, display_x=0
            )

        self.assertEqual(self.store.list_contacts(), [])


if __name__ == "__main__":
    unittest.main()
