"""Configure native geospatial data paths before the frozen app imports."""

from __future__ import annotations

import os
from pathlib import Path


def _existing_directory(*candidates) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).resolve()
        if path.is_dir():
            return path
    return None


def _configure_geospatial_data() -> None:
    import pyproj
    import rasterio
    from rasterio.env import GDALDataFinder

    rasterio_directory = Path(rasterio.__file__).resolve().parent
    proj_directory = _existing_directory(
        rasterio_directory / "proj_data",
        pyproj.datadir.get_data_dir(),
        Path(pyproj.__file__).resolve().parent / "proj_dir" / "share" / "proj",
    )
    if proj_directory is not None and (proj_directory / "proj.db").is_file():
        pyproj.datadir.set_data_dir(str(proj_directory))
        os.environ["PROJ_DATA"] = str(proj_directory)
        os.environ["PROJ_LIB"] = str(proj_directory)
        from rasterio._env import set_proj_data_search_path

        set_proj_data_search_path(str(proj_directory))

    gdal_directory = _existing_directory(
        GDALDataFinder().search(),
        rasterio_directory / "gdal_data",
    )
    if gdal_directory is not None:
        os.environ["GDAL_DATA"] = str(gdal_directory)


_configure_geospatial_data()
