"""Standards-compliant contact exporters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4
import xml.etree.ElementTree as ET

from sidescantools.contact_model import ContactRecord, CoordinateStatus


GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"
SIDESCAN_NAMESPACE = "https://github.com/sonoware/sidescantools/contact/1"
ET.register_namespace("", GPX_NAMESPACE)
ET.register_namespace("sct", SIDESCAN_NAMESPACE)


class ContactExporter(Protocol):
    suffixes: tuple[str, ...]

    def export(
        self,
        contacts: Sequence[ContactRecord],
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> "ExportResult": ...


@dataclass(frozen=True, slots=True)
class ExportResult:
    destination: Path
    exported_count: int
    skipped_by_status: dict[str, int]

    @property
    def skipped_count(self) -> int:
        return sum(self.skipped_by_status.values())


class GPXExporter:
    suffixes = (".gpx",)

    def __init__(self, *, include_stale: bool = False):
        self.include_stale = include_stale

    def export(
        self,
        contacts: Sequence[ContactRecord],
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> ExportResult:
        destination = Path(destination)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        if not destination.parent.exists():
            raise FileNotFoundError(destination.parent)

        included_statuses = {CoordinateStatus.VALID}
        if self.include_stale:
            included_statuses.add(CoordinateStatus.STALE)
        eligible = []
        skipped = Counter()
        for contact in contacts:
            if contact.coordinate_status in included_statuses:
                eligible.append(contact)
            else:
                skipped[contact.coordinate_status.value] += 1
        eligible.sort(
            key=lambda contact: (
                contact.draft.anchor.source_file_id,
                contact.draft.anchor.global_ping_index,
                contact.id,
            )
        )

        root = ET.Element(
            ET.QName(GPX_NAMESPACE, "gpx"),
            {"version": "1.1", "creator": "SidescanTools Contact Picker"},
        )
        for contact in eligible:
            self._append_waypoint(root, contact)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")

        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as output:
                tree.write(output, encoding="utf-8", xml_declaration=True)
                output.flush()
                os.fsync(output.fileno())
            if destination.exists() and not overwrite:
                raise FileExistsError(destination)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        return ExportResult(
            destination=destination,
            exported_count=len(eligible),
            skipped_by_status=dict(sorted(skipped.items())),
        )

    @staticmethod
    def _append_waypoint(root: ET.Element, contact: ContactRecord) -> None:
        draft = contact.draft
        coordinate = draft.coordinate
        waypoint = ET.SubElement(
            root,
            ET.QName(GPX_NAMESPACE, "wpt"),
            {
                "lat": f"{coordinate.latitude:.8f}",
                "lon": f"{coordinate.longitude:.8f}",
            },
        )
        if draft.timestamp_iso and draft.timestamp_basis in {
            "utc",
            "explicit-offset",
        }:
            ET.SubElement(waypoint, ET.QName(GPX_NAMESPACE, "time")).text = (
                draft.timestamp_iso
            )
        name = draft.name.strip() or f"Target {contact.id:04d}"
        ET.SubElement(waypoint, ET.QName(GPX_NAMESPACE, "name")).text = name
        if draft.notes:
            ET.SubElement(waypoint, ET.QName(GPX_NAMESPACE, "desc")).text = draft.notes
        ET.SubElement(waypoint, ET.QName(GPX_NAMESPACE, "type")).text = "contact"

        extensions = ET.SubElement(
            waypoint, ET.QName(GPX_NAMESPACE, "extensions")
        )
        values = {
            "uuid": draft.uuid,
            "source_filename": (
                Path(contact.source_display_path).name
                if contact.source_display_path
                else None
            ),
            "ping_number": draft.anchor.ping_number,
            "global_ping_index": draft.anchor.global_ping_index,
            "channel": draft.anchor.channel.label,
            "source_sample_index": draft.anchor.source_sample_index,
            "sample_fraction": f"{draft.anchor.sample_fraction:.12g}",
            "slant_range_m": draft.coordinate.slant_range_m,
            "ground_range_m": draft.coordinate.ground_range_m,
            "intensity_source": draft.intensity_source,
            "intensity_display": draft.intensity_display,
            "geometry_settings_hash": contact.geometry_settings_hash,
            "coordinate_status": contact.coordinate_status.value,
        }
        for key, value in values.items():
            if value is not None:
                ET.SubElement(
                    extensions, ET.QName(SIDESCAN_NAMESPACE, key)
                ).text = str(value)
