from pathlib import Path
import sqlite3
import tempfile
import unittest

from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactDraft,
    ContactThumbnail,
    CoordinateStatus,
)
from sidescantools.contact_store import (
    ContactStore,
    ContactStoreError,
    DuplicateContactAnchor,
)
from sidescantools.swath_geometry import GeometrySettings


class ContactStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "contacts.sqlite"
        self.store = ContactStore(self.database_path)
        self.source = self.store.register_source_file(
            Path(self.temporary_directory.name) / "survey.jsf",
            format="JSF",
            ping_count=20,
            source_sample_count=9,
            file_size_bytes=1234,
            mtime_ns=5678,
            content_fingerprint="sha256:example",
        )
        self.profile_id = self.store.get_or_create_geometry_profile(
            GeometrySettings(60, cable_out_m=10)
        )

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def draft(self, *, sample=4, name="Target 0001"):
        anchor = ContactAnchor(
            source_file_id=self.source.id,
            global_ping_index=3,
            ping_number=50_003,
            channel=Channel.PORT,
            source_sample_index=sample,
            sample_fraction=sample / 8,
            display_chunk=0,
            display_ping_index=3,
            display_sample_index=4,
        )
        coordinate = ContactCoordinate(
            longitude=-70.12345678,
            latitude=40.12345678,
            slant_range_m=15.5,
            ground_range_m=12.25,
            geometry_profile_id=self.profile_id,
        )
        return ContactDraft(
            anchor=anchor,
            coordinate=coordinate,
            name=name,
            notes="Possible debris — café",
            classification="debris",
            timestamp_iso="2026-08-26T15:20:30-05:00",
            timestamp_basis="explicit-offset",
            intensity_source=123.5,
            intensity_display=0.75,
            intensity_pipeline="normalized-v1",
        )

    @staticmethod
    def thumbnail():
        return ContactThumbnail(
            image_bytes=b"not-a-real-png-for-storage-test",
            width_px=32,
            height_px=24,
            ping_radius=10,
            sample_radius=15,
            display_pipeline="normalized-v1+crosshair",
        )

    def test_initial_migration_and_reopen(self):
        self.assertEqual(self.store.schema_version, 1)
        self.assertEqual(
            self.store.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1
        )
        self.store.close()

        self.store = ContactStore(self.database_path)

        self.assertEqual(self.store.schema_version, 1)
        migration_count = self.store.connection.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0]
        self.assertEqual(migration_count, 1)

    def test_source_and_geometry_profile_registration_are_idempotent(self):
        source_again = self.store.register_source_file(
            self.source.display_path,
            format="jsf",
            ping_count=21,
            source_sample_count=9,
        )
        profile_again = self.store.get_or_create_geometry_profile(
            GeometrySettings(60, cable_out_m=10)
        )

        self.assertEqual(source_again.id, self.source.id)
        self.assertEqual(source_again.ping_count, 21)
        self.assertEqual(profile_again, self.profile_id)

    def test_contact_and_thumbnail_round_trip_with_unicode(self):
        created = self.store.create_contact(self.draft(), self.thumbnail())

        reopened = self.store.get_contact(created.id)

        self.assertEqual(reopened.draft, created.draft)
        self.assertEqual(reopened.coordinate_status, CoordinateStatus.VALID)
        self.assertEqual(reopened.draft.notes, "Possible debris — café")
        self.assertEqual(self.store.get_thumbnail(created.id), self.thumbnail())

    def test_update_list_and_delete_cascade(self):
        created = self.store.create_contact(self.draft(), self.thumbnail())

        updated = self.store.update_contact_text(
            created.id,
            name="Anomaly α",
            notes="reviewed",
            classification=None,
        )
        listed = self.store.list_contacts(
            source_file_id=self.source.id, statuses=[CoordinateStatus.VALID]
        )

        self.assertEqual(updated.draft.name, "Anomaly α")
        self.assertEqual([record.id for record in listed], [created.id])
        self.store.delete_contact(created.id)
        self.assertIsNone(self.store.get_thumbnail(created.id))
        with self.assertRaises(KeyError):
            self.store.get_contact(created.id)

    def test_duplicate_anchor_is_rejected_without_partial_row(self):
        self.store.create_contact(self.draft())

        with self.assertRaises(DuplicateContactAnchor):
            self.store.create_contact(self.draft(name="duplicate"))

        self.assertEqual(len(self.store.list_contacts()), 1)

    def test_thumbnail_failure_rolls_back_contact_insert(self):
        self.store.connection.execute(
            """
            CREATE TRIGGER force_thumbnail_failure
            BEFORE INSERT ON contact_thumbnails
            BEGIN
                SELECT RAISE(ABORT, 'forced thumbnail failure');
            END
            """
        )

        with self.assertRaises(ContactStoreError):
            self.store.create_contact(self.draft(), self.thumbnail())

        self.assertEqual(self.store.list_contacts(), [])

    def test_stale_detection_and_recompute_preserve_acoustic_data(self):
        created = self.store.create_contact(self.draft(), self.thumbnail())
        changed_profile_id = self.store.get_or_create_geometry_profile(
            GeometrySettings(60, cable_out_m=25)
        )

        changed_count = self.store.mark_stale_for_profile(
            self.source.id, changed_profile_id
        )
        stale = self.store.get_contact(created.id)
        recomputed_coordinate = ContactCoordinate(
            longitude=-70.22222222,
            latitude=40.22222222,
            slant_range_m=16,
            ground_range_m=13,
            geometry_profile_id=changed_profile_id,
        )
        recomputed = self.store.recompute_contact(
            created.id, recomputed_coordinate
        )

        self.assertEqual(changed_count, 1)
        self.assertEqual(stale.coordinate_status, CoordinateStatus.STALE)
        self.assertEqual(recomputed.coordinate_status, CoordinateStatus.VALID)
        self.assertEqual(recomputed.draft.anchor, created.draft.anchor)
        self.assertEqual(recomputed.draft.name, created.draft.name)
        self.assertEqual(self.store.get_thumbnail(created.id), self.thumbnail())
        history = self.store.list_coordinate_history(created.id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["coordinate_status"], "stale")
        self.assertEqual(history[0]["longitude"], -70.12345678)

    def test_recompute_failure_retains_last_known_coordinate(self):
        created = self.store.create_contact(self.draft())

        failed = self.store.record_recompute_error(
            created.id, "navigation unavailable"
        )

        self.assertEqual(failed.coordinate_status, CoordinateStatus.ERROR)
        self.assertEqual(failed.coordinate_error, "navigation unavailable")
        self.assertEqual(failed.draft.coordinate, created.draft.coordinate)
        self.assertEqual(len(self.store.list_coordinate_history(created.id)), 1)


class FutureSchemaTests(unittest.TestCase):
    def test_newer_database_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "future.sqlite"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO schema_migrations VALUES (99, 'future')"
            )
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(ContactStoreError, "newer"):
                ContactStore(path)


if __name__ == "__main__":
    unittest.main()
