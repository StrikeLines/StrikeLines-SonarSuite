"""Run the contact pipeline against a real JSF without launching Napari."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from sidescantools.contact_export import GPXExporter, GPX_NAMESPACE
from sidescantools.contact_picker import ContactPickerService
from sidescantools.contact_store import ContactStore
from sidescantools.contact_thumbnail import ContactThumbnailExtractor
from sidescantools.georef_thread import Georeferencer
from sidescantools.sidescan_file import SidescanFile
from sidescantools.swath_geometry import GeometrySettings


class SmokePreprocessor:
    """Endpoint-preserving lightweight waterfall used only by this smoke test."""

    def __init__(self, sidescan_file, *, chunk_size: int, reduction_factor: int):
        self.chunk_size = chunk_size
        self.ping_len = max(2, math.ceil(sidescan_file.ping_len / reduction_factor))
        source_indices = np.rint(
            np.linspace(0, sidescan_file.ping_len - 1, self.ping_len)
        ).astype(int)
        port = np.take(sidescan_file.data[0], source_indices, axis=1)
        starboard = np.take(sidescan_file.data[1], source_indices, axis=1)
        logical = np.hstack((port, starboard))
        chunk_count = math.ceil(sidescan_file.num_ping / chunk_size)
        self.napari_fullmat = np.zeros(
            (chunk_count, chunk_size, 2 * self.ping_len), dtype=logical.dtype
        )
        self.napari_fullmat.reshape(-1, 2 * self.ping_len)[
            : sidescan_file.num_ping
        ] = logical


def run_smoke(path: Path, *, reduction_factor: int = 32) -> dict[str, object]:
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        sidescan_file = SidescanFile(path)

    preprocessor = SmokePreprocessor(
        sidescan_file, chunk_size=256, reduction_factor=reduction_factor
    )
    settings = GeometrySettings(vertical_beam_angle=60)
    geometry = {}
    for channel in (0, 1):
        georeferencer = Georeferencer(
            path,
            channel=channel,
            sidescan_file=sidescan_file,
            geometry_settings=settings,
            output_folder=path.parent,
        )
        geometry[channel] = georeferencer.prepare_swath_geometry()
        if georeferencer.nav != []:
            raise AssertionError("contact geometry unexpectedly allocated bulk nav")

    selected_ping = sidescan_file.num_ping // 2
    display_row = preprocessor.napari_fullmat.reshape(
        -1, 2 * preprocessor.ping_len
    )[selected_ping]
    port_x = int(np.argmax(display_row[: preprocessor.ping_len]))
    starboard_x = preprocessor.ping_len + int(
        np.argmax(display_row[preprocessor.ping_len :])
    )

    with tempfile.TemporaryDirectory(prefix="sidescantools-contact-smoke-") as temp:
        temporary = Path(temp)
        with ContactStore(temporary / "contacts.sqlite") as store:
            stat = path.stat()
            source = store.register_source_file(
                path,
                format="jsf",
                ping_count=sidescan_file.num_ping,
                source_sample_count=sidescan_file.ping_len,
                file_size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
            profile_id = store.get_or_create_geometry_profile(settings)
            thumbnails = ContactThumbnailExtractor(
                preprocessor=preprocessor,
                sidescan_file=sidescan_file,
                ping_radius=20,
                sample_radius=30,
            )
            picker = ContactPickerService(
                sidescan_file=sidescan_file,
                preprocessor=preprocessor,
                source_file_id=source.id,
                geometry_profile_id=profile_id,
                geometry_by_channel=geometry,
                store=store,
                thumbnail_factory=thumbnails,
            )
            chunk, local = divmod(selected_ping, preprocessor.chunk_size)
            port = picker.pick_display_pixel(
                chunk_index=chunk,
                local_ping_index=local,
                display_x=port_x,
                name="Real JSF port smoke contact",
            ).contact
            starboard = picker.pick_display_pixel(
                chunk_index=chunk,
                local_ping_index=local,
                display_x=starboard_x,
                name="Real JSF starboard smoke contact",
            ).contact
            contacts = store.list_contacts()
            destination = temporary / "contacts.gpx"
            export = GPXExporter().export(contacts, destination)
            waypoints = ET.parse(destination).getroot().findall(
                f"{{{GPX_NAMESPACE}}}wpt"
            )
            thumbnail = store.get_thumbnail(port.id)

            if len(contacts) != 2 or export.exported_count != 2 or len(waypoints) != 2:
                raise AssertionError("contact persistence or GPX round trip failed")
            if port.draft.anchor.channel == starboard.draft.anchor.channel:
                raise AssertionError("port/starboard mapping collapsed to one channel")

            return {
                "source": str(path),
                "subsystems": sidescan_file.subsys_names,
                "pings": sidescan_file.num_ping,
                "source_samples_per_channel": sidescan_file.ping_len,
                "display_samples_per_channel": preprocessor.ping_len,
                "valid_navigation_pings": int(np.count_nonzero(sidescan_file.longitude)),
                "selected_global_ping": selected_ping,
                "selected_ping_number": port.draft.anchor.ping_number,
                "port": {
                    "source_sample": port.draft.anchor.source_sample_index,
                    "longitude": port.draft.coordinate.longitude,
                    "latitude": port.draft.coordinate.latitude,
                    "intensity": port.draft.intensity_source,
                },
                "starboard": {
                    "source_sample": starboard.draft.anchor.source_sample_index,
                    "longitude": starboard.draft.coordinate.longitude,
                    "latitude": starboard.draft.coordinate.latitude,
                    "intensity": starboard.draft.intensity_source,
                },
                "thumbnail_size": [thumbnail.width_px, thumbnail.height_px],
                "gpx_waypoints": len(waypoints),
                "bulk_nav_allocated": False,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsf", type=Path)
    parser.add_argument("--reduction-factor", type=int, default=32)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_smoke(arguments.jsf, reduction_factor=arguments.reduction_factor),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
