"""Command-line launcher for the Napari contact-picking workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sidescantools.swath_geometry import GeometrySettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sidescantools-contacts",
        description="Open a JSF or XTF waterfall for bottom editing and contact picking.",
    )
    parser.add_argument(
        "sonar_file",
        type=Path,
        nargs="?",
        default=None,
        help="input .jsf or .xtf file (omit to pick one from a file dialog; "
        "only supported with --viewer qt)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="output directory (defaults to the input file's directory)",
    )
    parser.add_argument(
        "--contacts-db",
        type=Path,
        help="SQLite project database (defaults to WORK_DIR/contacts.sqlite)",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--downsampling-factor", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--vertical-beam-angle", type=float, default=60.0)
    parser.add_argument("--cable-out", type=float, default=0.0, metavar="METERS")
    parser.add_argument("--x-offset", type=float, default=0.0, metavar="METERS")
    parser.add_argument("--y-offset", type=float, default=0.0, metavar="METERS")
    parser.add_argument("--db-display", action="store_true")
    parser.add_argument("--histogram-equalization", action="store_true")
    parser.add_argument(
        "--viewer",
        choices=("auto", "qt", "napari"),
        default="auto",
        help="display backend; auto uses the OpenGL-free Qt viewer on Windows",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    viewer_backend = arguments.viewer
    if viewer_backend == "auto":
        viewer_backend = "qt" if sys.platform == "win32" else "napari"

    sonar_file = arguments.sonar_file.resolve() if arguments.sonar_file is not None else None
    if sonar_file is not None:
        if not sonar_file.is_file():
            parser.error(f"input file does not exist: {sonar_file}")
        if sonar_file.suffix.casefold() not in {".jsf", ".xtf"}:
            parser.error("input file must have a .jsf or .xtf extension")
    elif viewer_backend != "qt":
        parser.error("sonar_file is required unless --viewer qt is used")
    if arguments.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if arguments.downsampling_factor < 1:
        parser.error("--downsampling-factor must be positive")

    # With no sonar_file, work_dir/contacts_db default relative to whatever
    # file the user picks from the Open-file dialog on launch -- resolved
    # later, inside run_qt_contact_picker, once that file is known.
    work_dir = arguments.work_dir.resolve() if arguments.work_dir is not None else None
    if work_dir is None and sonar_file is not None:
        work_dir = sonar_file.parent
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)

    contacts_db = arguments.contacts_db.resolve() if arguments.contacts_db is not None else None
    if contacts_db is None and work_dir is not None:
        contacts_db = work_dir / "contacts.sqlite"
    if contacts_db is not None:
        contacts_db.parent.mkdir(parents=True, exist_ok=True)

    settings = GeometrySettings(
        vertical_beam_angle=arguments.vertical_beam_angle,
        cable_out_m=arguments.cable_out,
        x_offset_m=arguments.x_offset,
        y_offset_m=arguments.y_offset,
    )

    common_arguments = dict(
        chunk_size=arguments.chunk_size,
        default_threshold=arguments.threshold,
        downsampling_factor=arguments.downsampling_factor,
        work_dir=work_dir,
        active_dB=arguments.db_display,
        active_hist_equal=arguments.histogram_equalization,
        contacts_db_path=contacts_db,
        geometry_settings=settings,
    )
    if viewer_backend == "qt":
        from sidescantools.qt_contact_picker_ui import run_qt_contact_picker

        run_qt_contact_picker(sonar_file, **common_arguments)
        return

    # Napari is imported only after arguments have been validated, keeping
    # `--help` useful in non-GUI environments.
    from sidescantools.bottom_detection_napari_ui import run_napari_btm_line

    run_napari_btm_line(
        sonar_file,
        **common_arguments,
    )


if __name__ == "__main__":
    main()
