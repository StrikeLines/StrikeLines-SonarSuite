"""Fast, settings-aware GeoTIFF export for sidescan waterfall data.

The original SidescanTools exporter writes an XYZ point cloud for each channel
and invokes GMT gridding separately.  This module keeps the shared swath
geometry, but bins both channels directly into one north-up raster and writes
the result with Rasterio.  That avoids large intermediate files and preserves
the exact RGB palette used by the Qt waterfall.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
import pyproj
from pyproj import Geod, Transformer
from scipy.ndimage import binary_dilation, maximum_filter

from sidescantools.bottom_line_io import compute_depth_info, load_bottom_info
from sidescantools.contact_gain import (
    BuiltInGainMode,
    BuiltInGainProcessor,
    BuiltInGainRequest,
)
from sidescantools.gain_settings import (
    SonarGainSettings,
    load_gain_settings,
    resolve_egn_table_path,
)
from sidescantools.georef_thread import Georeferencer
from sidescantools.sidescan_file import SidescanFile
from sidescantools.sidescan_preproc import SidescanPreprocessor
from sidescantools.swath_geometry import GeometrySettings, SwathGeometry


SUPPORTED_EPSG_CODES = (4326, 3857)
DEFAULT_MAX_RASTER_PIXELS = 40_000_000
_TVG_FLOOR_FRACTION = 0.02
_GEOD = Geod(ellps="WGS84")


def _configure_pyproj_data(*extra_candidates: Path) -> Path:
    """Point PyProj at a real ``proj.db`` despite broken Windows globals.

    QGIS installers sometimes leave a quoted ``PROJ_LIB`` value in the
    process environment. PROJ treats those quote characters as part of the
    path, so EPSG lookups fail even though both PyProj and Rasterio ship a
    compatible database. Resolve and set an application-local path before
    creating any transformer; do not alter the user's environment variables.
    """

    # Prefer files installed alongside the running Python/Rasterio packages;
    # an external QGIS database can target a different PROJ/GDAL version.
    candidates: list[Path] = [Path(path) for path in extra_candidates if path]
    try:
        candidates.append(Path(pyproj.datadir.get_data_dir().strip('"')))
    except Exception:
        pass
    candidates.extend(
        (
            Path(pyproj.__file__).resolve().parent / "proj_dir" / "share" / "proj",
            Path(sys.prefix) / "Library" / "share" / "proj",
            Path(sys.prefix) / "share" / "proj",
        )
    )
    # A sanitized system path is the final fallback only.
    for variable in ("PROJ_DATA", "PROJ_LIB"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value.strip().strip('"')))
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if (resolved / "proj.db").is_file():
            pyproj.datadir.set_data_dir(str(resolved))
            return resolved
    raise RuntimeError(
        "Cannot find the PROJ coordinate database (proj.db). Reinstall "
        "PyProj/Rasterio or the full SidescanTools environment."
    )


@dataclass(frozen=True, slots=True)
class PreparedSonarExport:
    rgb: np.ndarray
    geometry_by_channel: Mapping[int, SwathGeometry]
    pipeline_description: str
    used_default_settings: bool = False


@dataclass(frozen=True, slots=True)
class GeoTiffExportResult:
    destination: Path
    epsg: int
    width: int
    height: int
    valid_pixel_count: int
    resolution_m: float
    used_default_settings: bool = False


def geotiff_output_path(sonar_path: str | os.PathLike) -> Path:
    """Return ``<source directory>/<source stem>.tif``."""

    return Path(sonar_path).with_suffix(".tif")


def default_gain_settings(sonar_path: str | os.PathLike) -> SonarGainSettings:
    """Settings used for batch files that have never been opened in the UI."""

    source = Path(sonar_path)
    return SonarGainSettings(
        source_file=source.name,
        overall_gain_db=-5.0,
        tvg_spreading_db_per_decade=5.0,
        tvg_absorption_db_per_m=0.08,
        auto_tvg_brightness_target_percent=30,
        auto_tvg_active=False,
        auto_tvg_gain_db=(),
        speed_correction_px_per_ping=3.0,
        processing_mode=BuiltInGainMode.RAW.value,
        egn_table_path=None,
        slant_range_correction_active=False,
    )


def render_rgb_with_gain_settings(
    waterfall: np.ndarray,
    *,
    slant_range_m: float,
    settings: SonarGainSettings,
) -> np.ndarray:
    """Apply a sidecar's gain curve and the Qt waterfall's warm RGB palette."""

    source = np.asarray(waterfall, dtype=float)
    if source.ndim != 2 or source.shape[1] < 4 or source.shape[1] % 2:
        raise ValueError("continuous waterfall must have two equal-width channels")
    if not math.isfinite(float(slant_range_m)) or slant_range_m <= 0:
        raise ValueError("slant range must be positive and finite")

    channel_width = source.shape[1] // 2
    sample_fraction = np.concatenate(
        (
            np.linspace(1.0, 0.0, channel_width),
            np.linspace(0.0, 1.0, channel_width),
        )
    )
    range_m = sample_fraction * float(slant_range_m)
    floor_m = float(slant_range_m) * _TVG_FLOOR_FRACTION
    gain_db = (
        settings.overall_gain_db
        + settings.tvg_spreading_db_per_decade
        * np.log10(np.maximum(range_m, floor_m) / floor_m)
        + settings.tvg_absorption_db_per_m * range_m
    )
    if settings.auto_tvg_active:
        residual = np.asarray(settings.auto_tvg_gain_db, dtype=float)
        if residual.shape != (source.shape[1],):
            raise ValueError(
                "saved Auto TVG correction does not match the waterfall width"
            )
        gain_db = gain_db + residual

    corrected = np.nan_to_num(
        source, nan=0.0, posinf=1.0, neginf=0.0, copy=True
    )
    corrected *= np.power(10.0, gain_db / 20.0)[None, :]
    gray = np.rint(np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack(
        (gray, (gray.astype(float) * 0.78).astype(np.uint8), gray // 3),
        axis=2,
    )


def prepare_sonar_export(
    sonar_path: str | os.PathLike,
    *,
    chunk_size: int,
    default_threshold: float,
    downsampling_factor: int,
    active_db: bool,
    active_hist_equal: bool,
    geometry_settings: GeometrySettings,
    progress: Callable[[int, str], None] | None = None,
) -> PreparedSonarExport:
    """Load one file and reproduce its persisted waterfall for batch export."""

    notify = progress or (lambda percent, message: None)
    _configure_pyproj_data()
    source = Path(sonar_path).resolve()
    settings = load_gain_settings(source)
    used_defaults = settings is None
    if settings is None:
        settings = default_gain_settings(source)

    notify(2, "Reading sonar data")
    sidescan_file = SidescanFile(source)
    preprocessor = SidescanPreprocessor(
        sidescan_file=sidescan_file,
        chunk_size=chunk_size,
        downsampling_factor=downsampling_factor,
    )
    depth_info = compute_depth_info(sidescan_file, downsampling_factor)
    preprocessor.init_napari_bottom_detect(
        default_threshold,
        active_dB=active_db,
        active_hist_equal=active_hist_equal,
        depth_info=depth_info,
    )
    bottom_path = source.parent / f"{source.stem}_bottom_info.npz"
    if bottom_path.is_file():
        load_bottom_info(bottom_path, preprocessor, sidescan_file)

    raw = _logical_waterfall(
        preprocessor.napari_fullmat,
        preprocessor.ping_len,
        sidescan_file.num_ping,
    )
    mode = BuiltInGainMode(settings.processing_mode)
    if (
        mode is BuiltInGainMode.RAW
        and not settings.destripe_active
        and not settings.slant_range_correction_active
    ):
        display_data = raw
        pipeline = "qt-continuous-waterfall-v1|raw"
    else:
        egn_path = resolve_egn_table_path(settings, source)
        request = BuiltInGainRequest(
            mode=mode,
            egn_table_path=egn_path,
            nadir_angle=_egn_table_nadir_angle(egn_path),
            destripe=settings.destripe_active,
            slant_range_correction=settings.slant_range_correction_active,
        )
        result = BuiltInGainProcessor(preprocessor, raw).process(
            request,
            progress=lambda percent, message: notify(
                10 + int(percent * 0.45), message
            ),
        )
        display_data = result.display_data
        pipeline = result.pipeline_description

    notify(58, "Applying saved TVG and display colors")
    slant_range_m = float(np.median(sidescan_file.slant_range))
    rgb = render_rgb_with_gain_settings(
        display_data,
        slant_range_m=slant_range_m,
        settings=settings,
    )
    notify(65, "Preparing swath geometry")
    geometry = {
        channel: Georeferencer(
            source,
            channel=channel,
            sidescan_file=sidescan_file,
            geometry_settings=geometry_settings,
            output_folder=source.parent,
        ).prepare_swath_geometry()
        for channel in (0, 1)
    }
    return PreparedSonarExport(
        rgb=rgb,
        geometry_by_channel=geometry,
        pipeline_description=(
            pipeline
            + f"|gain={settings.overall_gain_db:.1f}dB"
            f"|tvg-spreading={settings.tvg_spreading_db_per_decade:.1f}dB/decade"
            f"|tvg-absorption={settings.tvg_absorption_db_per_m:.2f}dB/m"
            + (
                f"|swath-equalized={settings.auto_tvg_brightness_target_percent}pct"
                if settings.auto_tvg_active
                else ""
            )
        ),
        used_default_settings=used_defaults,
    )


def export_sonar_file(
    sonar_path: str | os.PathLike,
    *,
    epsg: int,
    chunk_size: int,
    default_threshold: float,
    downsampling_factor: int,
    active_db: bool,
    active_hist_equal: bool,
    geometry_settings: GeometrySettings,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
) -> GeoTiffExportResult:
    prepared = prepare_sonar_export(
        sonar_path,
        chunk_size=chunk_size,
        default_threshold=default_threshold,
        downsampling_factor=downsampling_factor,
        active_db=active_db,
        active_hist_equal=active_hist_equal,
        geometry_settings=geometry_settings,
        progress=progress,
    )
    result = export_prepared_waterfall(
        sonar_path,
        prepared.rgb,
        prepared.geometry_by_channel,
        epsg=epsg,
        pipeline_description=prepared.pipeline_description,
        overwrite=overwrite,
        progress=progress,
    )
    return GeoTiffExportResult(
        destination=result.destination,
        epsg=result.epsg,
        width=result.width,
        height=result.height,
        valid_pixel_count=result.valid_pixel_count,
        resolution_m=result.resolution_m,
        used_default_settings=prepared.used_default_settings,
    )


def export_prepared_waterfall(
    sonar_path: str | os.PathLike,
    rgb: np.ndarray,
    geometry_by_channel: Mapping[int, SwathGeometry],
    *,
    epsg: int,
    pipeline_description: str,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
    max_raster_pixels: int = DEFAULT_MAX_RASTER_PIXELS,
) -> GeoTiffExportResult:
    """Rasterize one prepared full-swath RGB waterfall into one GeoTIFF."""

    try:
        import rasterio
        from rasterio._env import set_proj_data_search_path
        from rasterio.enums import ColorInterp
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise RuntimeError(
            "GeoTIFF export requires Rasterio. Install the full SidescanTools "
            "environment before exporting."
        ) from exc

    proj_data = _configure_pyproj_data(
        Path(rasterio.__file__).resolve().parent / "proj_data"
    )
    # Rasterio/GDAL has its own PROJ context, separate from PyProj's.
    set_proj_data_search_path(str(proj_data))

    notify = progress or (lambda percent, message: None)
    epsg = int(epsg)
    if epsg not in SUPPORTED_EPSG_CODES:
        raise ValueError("GeoTIFF CRS must be EPSG:4326 or EPSG:3857")
    if max_raster_pixels < 1:
        raise ValueError("max_raster_pixels must be positive")

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[1] % 2:
        raise ValueError("waterfall RGB must have shape [pings, 2*samples, 3]")
    channel_width = image.shape[1] // 2
    geometries = {
        int(channel): geometry for channel, geometry in geometry_by_channel.items()
    }
    if set(geometries) != {0, 1}:
        raise ValueError("port and starboard swath geometry are required")
    for geometry in geometries.values():
        if geometry.ping_count != image.shape[0]:
            raise ValueError("waterfall and swath geometry ping counts differ")

    destination = geotiff_output_path(sonar_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    notify(70, "Sizing output raster")
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    bounds_x: list[np.ndarray] = []
    bounds_y: list[np.ndarray] = []
    projected_across_spacing: list[np.ndarray] = []
    geodesic_across_spacing: list[np.ndarray] = []
    for geometry in geometries.values():
        valid = geometry.valid_ping_mask
        if not np.any(valid):
            continue
        nadir_lon = geometry.nadir_lon[valid]
        nadir_lat = geometry.nadir_lat[valid]
        outer_lon = geometry.outer_lon[valid]
        outer_lat = geometry.outer_lat[valid]
        nadir_x, nadir_y = transformer.transform(nadir_lon, nadir_lat)
        outer_x, outer_y = transformer.transform(outer_lon, outer_lat)
        bounds_x.extend((np.asarray(nadir_x), np.asarray(outer_x)))
        bounds_y.extend((np.asarray(nadir_y), np.asarray(outer_y)))
        projected_across_spacing.append(
            np.hypot(np.asarray(outer_x) - nadir_x, np.asarray(outer_y) - nadir_y)
            / max(1, channel_width - 1)
        )
        _, _, distance_m = _GEOD.inv(
            nadir_lon, nadir_lat, outer_lon, outer_lat
        )
        geodesic_across_spacing.append(
            np.asarray(distance_m) / max(1, channel_width - 1)
        )
    if not bounds_x:
        raise ValueError("sonar file has no valid navigation coordinates")

    all_x = np.concatenate(bounds_x)
    all_y = np.concatenate(bounds_y)
    finite_bounds = np.isfinite(all_x) & np.isfinite(all_y)
    if not np.any(finite_bounds):
        raise ValueError("projected navigation contains no finite coordinates")
    xmin = float(np.min(all_x[finite_bounds]))
    xmax = float(np.max(all_x[finite_bounds]))
    ymin = float(np.min(all_y[finite_bounds]))
    ymax = float(np.max(all_y[finite_bounds]))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("navigation extent is too small to create a raster")

    resolution_m = _positive_median(np.concatenate(geodesic_across_spacing))
    if epsg == 3857:
        resolution_x = resolution_y = _positive_median(
            np.concatenate(projected_across_spacing)
        )
    else:
        center_lon = (xmin + xmax) / 2.0
        center_lat = (ymin + ymax) / 2.0
        lon_east, _, _ = _GEOD.fwd(center_lon, center_lat, 90.0, resolution_m)
        _, lat_north, _ = _GEOD.fwd(center_lon, center_lat, 0.0, resolution_m)
        resolution_x = abs(lon_east - center_lon)
        resolution_y = abs(lat_north - center_lat)
    if min(resolution_x, resolution_y) <= 0:
        raise ValueError("could not determine a positive GeoTIFF resolution")

    width = max(1, int(math.ceil((xmax - xmin) / resolution_x)) + 1)
    height = max(1, int(math.ceil((ymax - ymin) / resolution_y)) + 1)
    pixel_count = width * height
    if pixel_count > max_raster_pixels:
        scale = math.sqrt(pixel_count / max_raster_pixels)
        resolution_x *= scale
        resolution_y *= scale
        resolution_m *= scale
        width = max(1, int(math.ceil((xmax - xmin) / resolution_x)) + 1)
        height = max(1, int(math.ceil((ymax - ymin) / resolution_y)) + 1)

    # Half-pixel padding keeps endpoint samples centered inside the raster.
    origin_x = xmin - resolution_x / 2.0
    origin_y = ymax + resolution_y / 2.0
    intensity = np.zeros((height, width), dtype=np.uint8)
    valid_pixels = np.zeros((height, width), dtype=bool)
    fractions = np.linspace(0.0, 1.0, channel_width)
    ping_chunk = max(1, min(512, 2_000_000 // max(1, channel_width)))

    notify(75, "Rasterizing port and starboard swaths")
    for channel in (0, 1):
        geometry = geometries[channel]
        # Geometry is near-to-far. The Qt port half is displayed far-to-near;
        # starboard already runs near-to-far.
        channel_gray = image[:, :channel_width, 0]
        if channel == 0:
            channel_gray = channel_gray[:, ::-1]
        else:
            channel_gray = image[:, channel_width:, 0]

        valid_indices = np.flatnonzero(geometry.valid_ping_mask)
        for start in range(0, len(valid_indices), ping_chunk):
            indices = valid_indices[start : start + ping_chunk]
            lon = geometry.nadir_lon[indices, None] + fractions[None, :] * (
                geometry.outer_lon[indices, None]
                - geometry.nadir_lon[indices, None]
            )
            lat = geometry.nadir_lat[indices, None] + fractions[None, :] * (
                geometry.outer_lat[indices, None]
                - geometry.nadir_lat[indices, None]
            )
            x, y = transformer.transform(lon, lat)
            columns = np.floor((np.asarray(x) - origin_x) / resolution_x).astype(int)
            rows = np.floor((origin_y - np.asarray(y)) / resolution_y).astype(int)
            values = channel_gray[indices]
            inside = (
                np.isfinite(x)
                & np.isfinite(y)
                & (rows >= 0)
                & (rows < height)
                & (columns >= 0)
                & (columns < width)
            )
            np.maximum.at(intensity, (rows[inside], columns[inside]), values[inside])
            valid_pixels[rows[inside], columns[inside]] = True

    if not np.any(valid_pixels):
        raise ValueError("no sonar samples fell inside the output raster")

    # Fill only tiny one-cell rasterization gaps. This is the direct-gridding
    # counterpart of the upstream near-neighbour search radius, without
    # creating large XYZ intermediates or bleeding across separated swaths.
    neighbours = binary_dilation(valid_pixels, iterations=1) & ~valid_pixels
    nearby_max = maximum_filter(intensity, size=3, mode="constant")
    intensity[neighbours] = nearby_max[neighbours]
    valid_pixels[neighbours] = True

    red = intensity
    green = (intensity.astype(float) * 0.78).astype(np.uint8)
    blue = intensity // 3
    alpha = np.where(valid_pixels, 255, 0).astype(np.uint8)
    transform = from_origin(origin_x, origin_y, resolution_x, resolution_y)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    notify(94, "Writing tiled GeoTIFF")
    raster_crs = rasterio.crs.CRS.from_epsg(epsg)
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=4,
            dtype="uint8",
            crs=raster_crs,
            transform=transform,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dataset:
            dataset.write(red, 1)
            dataset.write(green, 2)
            dataset.write(blue, 3)
            dataset.write(alpha, 4)
            dataset.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
            dataset.update_tags(
                SOURCE_FILE=Path(sonar_path).name,
                SIDESCANTOOLS_DISPLAY_PIPELINE=pipeline_description,
                TARGET_CRS=f"EPSG:{epsg}",
            )
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    notify(100, f"Saved {destination.name}")
    return GeoTiffExportResult(
        destination=destination,
        epsg=epsg,
        width=width,
        height=height,
        valid_pixel_count=int(np.count_nonzero(valid_pixels)),
        resolution_m=float(resolution_m),
    )


def _logical_waterfall(
    chunks: np.ndarray, ping_len: int, source_ping_count: int
) -> np.ndarray:
    chunks = np.asarray(chunks)
    expected_width = 2 * ping_len
    if chunks.ndim != 3 or chunks.shape[2] != expected_width:
        raise ValueError("chunked waterfall dimensions are inconsistent")
    return chunks.reshape(-1, expected_width)[:source_ping_count]


def _egn_table_nadir_angle(table_path: Path | None) -> float:
    if table_path is None:
        return 0.0
    try:
        with np.load(table_path) as table:
            return float(table["nadir_angle"]) if "nadir_angle" in table else 0.0
    except Exception:
        return 0.0


def _positive_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    usable = values[np.isfinite(values) & (values > 0.0)]
    if not usable.size:
        raise ValueError("navigation does not provide a usable pixel resolution")
    return float(np.median(usable))
