"""Dedicated entry point for the frozen Windows Qt application."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sys
import traceback


SUPPORTED_SONAR_SUFFIXES = {".jsf", ".xtf"}


def startup_sonar_file(arguments: list[str]) -> Path | None:
    """Return a valid file-association argument, otherwise start empty."""

    if not arguments:
        return None
    candidate = Path(arguments[0]).expanduser()
    if candidate.is_file() and candidate.suffix.casefold() in SUPPORTED_SONAR_SUFFIXES:
        return candidate.resolve()
    return None


def _run_smoke_test() -> None:
    """Exercise the native libraries and idle window in a frozen build."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import numpy as np
    import pyxtf  # noqa: F401 - verifies that the packaged reader imports
    import rasterio
    from PyQt5.QtWidgets import QApplication
    from pyproj import Transformer
    from rasterio._env import set_proj_data_search_path
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    from sidescantools.geotiff_export import _configure_pyproj_data
    from sidescantools.qt_contact_picker_ui import run_qt_contact_picker

    proj_data = _configure_pyproj_data(
        Path(rasterio.__file__).resolve().parent / "proj_data"
    )
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)
    set_proj_data_search_path(str(proj_data))
    east, north = Transformer.from_crs(4326, 3857, always_xy=True).transform(
        -122.0, 47.0
    )
    if not np.isfinite((east, north)).all():
        raise RuntimeError("PROJ returned invalid coordinates")

    pixels = np.arange(4, dtype=np.uint8).reshape(2, 2)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype=pixels.dtype,
            crs="EPSG:4326",
            transform=from_origin(-122.0, 47.0, 0.0001, 0.0001),
        ) as dataset:
            dataset.write(pixels, 1)
        with memory_file.open() as dataset:
            if dataset.crs is None or dataset.crs.to_epsg() != 4326:
                raise RuntimeError("GDAL did not preserve the test raster CRS")

    application = QApplication.instance() or QApplication([])
    window = run_qt_contact_picker(None, block=False)
    application.processEvents()
    window.close()
    application.processEvents()
    application.quit()


def main(argv: list[str] | None = None) -> int:
    """Launch the Qt-only desktop build, optionally opening one sonar file."""

    multiprocessing.freeze_support()
    os.environ.setdefault("QT_API", "pyqt5")
    arguments = sys.argv[1:] if argv is None else argv

    if arguments == ["--smoke-test"]:
        error_file = Path.cwd() / "sidescantools-smoke-error.txt"
        try:
            _run_smoke_test()
        except Exception:
            error_file.write_text(traceback.format_exc(), encoding="utf-8")
            return 1
        error_file.unlink(missing_ok=True)
        return 0

    from sidescantools.qt_contact_picker_ui import run_qt_contact_picker

    run_qt_contact_picker(startup_sonar_file(arguments), block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
