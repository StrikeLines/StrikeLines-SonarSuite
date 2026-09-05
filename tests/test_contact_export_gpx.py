from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from sidescantools.contact_export import (
    GPXExporter,
    GPX_NAMESPACE,
    SIDESCAN_NAMESPACE,
)
from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactDraft,
    ContactRecord,
    CoordinateStatus,
)


class GPXExporterTests(unittest.TestCase):
    def test_export_identifies_sonarsuite_as_the_creator(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "contacts.gpx"
            GPXExporter().export([self.record(1, 1)], destination)

            root = ET.parse(destination).getroot()

            self.assertEqual(root.attrib["creator"], "SonarSuite Contact Picker")

    def record(
        self,
        contact_id,
        global_ping,
        *,
        status=CoordinateStatus.VALID,
        name=None,
        notes="",
        timestamp="2026-08-26T15:20:30-05:00",
        timestamp_basis="explicit-offset",
    ):
        anchor = ContactAnchor(
            source_file_id=1,
            global_ping_index=global_ping,
            ping_number=50_000 + global_ping,
            channel=Channel.STARBOARD,
            source_sample_index=4,
            sample_fraction=0.5,
            display_chunk=0,
            display_ping_index=global_ping,
            display_sample_index=8,
        )
        coordinate = ContactCoordinate(
            longitude=-87.123456789,
            latitude=29.123456789,
            slant_range_m=10.5,
            ground_range_m=8.5,
            geometry_profile_id=2,
        )
        draft = ContactDraft(
            anchor=anchor,
            coordinate=coordinate,
            name=f"Target {contact_id:04d}" if name is None else name,
            notes=notes,
            timestamp_iso=timestamp,
            timestamp_basis=timestamp_basis,
            intensity_source=42.5,
            intensity_display=0.8,
            uuid=f"00000000-0000-0000-0000-{contact_id:012d}",
        )
        return ContactRecord(
            id=contact_id,
            draft=draft,
            coordinate_status=status,
            coordinate_error=None,
            source_display_path="survey & line.jsf",
            geometry_settings_hash="a" * 64,
            created_at="2026-08-26T20:20:31Z",
            updated_at="2026-08-26T20:20:31Z",
        )

    def test_valid_gpx_namespace_precision_order_and_escaping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contacts.gpx"
            later = self.record(2, 10)
            earlier = self.record(
                1,
                3,
                name="Rock & <wreck>",
                notes="Possible debris & cable",
            )

            result = GPXExporter().export([later, earlier], destination)
            root = ET.parse(destination).getroot()
            waypoints = root.findall(f"{{{GPX_NAMESPACE}}}wpt")

            self.assertEqual(root.tag, f"{{{GPX_NAMESPACE}}}gpx")
            self.assertEqual(result.exported_count, 2)
            self.assertEqual(waypoints[0].attrib["lat"], "29.12345679")
            self.assertEqual(waypoints[0].attrib["lon"], "-87.12345679")
            self.assertEqual(
                waypoints[0].findtext(f"{{{GPX_NAMESPACE}}}name"),
                "Rock & <wreck>",
            )
            self.assertEqual(
                waypoints[0].findtext(f"{{{GPX_NAMESPACE}}}desc"),
                "Possible debris & cable",
            )
            self.assertEqual(
                waypoints[0].findtext(
                    f"{{{GPX_NAMESPACE}}}extensions/"
                    f"{{{SIDESCAN_NAMESPACE}}}source_filename"
                ),
                "survey & line.jsf",
            )

    def test_status_filter_and_blank_default_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contacts.gpx"
            valid = self.record(7, 1, name="   ")
            stale = self.record(8, 2, status=CoordinateStatus.STALE)
            error = self.record(9, 3, status=CoordinateStatus.ERROR)

            result = GPXExporter().export([valid, stale, error], destination)
            root = ET.parse(destination).getroot()

            self.assertEqual(result.exported_count, 1)
            self.assertEqual(result.skipped_by_status, {"error": 1, "stale": 1})
            self.assertEqual(
                root.findtext(
                    f"{{{GPX_NAMESPACE}}}wpt/{{{GPX_NAMESPACE}}}name"
                ),
                "Target 0007",
            )

    def test_include_stale_exports_last_known_coordinate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contacts.gpx"
            stale = self.record(1, 1, status=CoordinateStatus.STALE)

            result = GPXExporter(include_stale=True).export([stale], destination)

            self.assertEqual(result.exported_count, 1)
            self.assertEqual(result.skipped_count, 0)

    def test_ambiguous_timestamp_is_omitted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contacts.gpx"
            contact = self.record(
                1,
                1,
                timestamp="2026-08-26T15:20:30",
                timestamp_basis="unknown",
            )

            GPXExporter().export([contact], destination)
            waypoint = ET.parse(destination).getroot().find(
                f"{{{GPX_NAMESPACE}}}wpt"
            )

            self.assertIsNone(waypoint.find(f"{{{GPX_NAMESPACE}}}time"))

    def test_existing_file_requires_explicit_overwrite_and_temp_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contacts.gpx"
            destination.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                GPXExporter().export([self.record(1, 1)], destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing")

            GPXExporter().export([self.record(1, 1)], destination, overwrite=True)

            self.assertTrue(destination.read_bytes().startswith(b"<?xml"))
            self.assertEqual(
                list(Path(temporary_directory).glob(".contacts.gpx.*.tmp")), []
            )


if __name__ == "__main__":
    unittest.main()
