from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import utm
import math
from pyproj import CRS, Transformer
from scipy.signal import savgol_filter
from scipy import interpolate
from decimal import Decimal
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from sidescantools.swath_geometry import GeometrySettings, SwathGeometry

if TYPE_CHECKING:
    from sidescantools.sidescan_file import SidescanFile


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedTrackGeometry:
    """Channel-independent navigation prepared once for a two-sided swath."""

    valid_ping_mask: np.ndarray
    source_ping: np.ndarray
    original_lon: np.ndarray
    original_lat: np.ndarray
    original_heading: np.ndarray
    ping_unique: np.ndarray
    cog_smooth: np.ndarray
    nadir_lon: np.ndarray
    nadir_lat: np.ndarray
    nadir_east: np.ndarray
    nadir_north: np.ndarray
    inverse_transformer: Transformer
    epsg_code: str


def _safe_savgol_filter(values, preferred_window: int, polyorder: int):
    """Smooth a track without exceeding the number of available pings."""

    array = np.asarray(values)
    sample_count = array.shape[-1]
    window_length = min(int(preferred_window), sample_count)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        return array.copy()
    return savgol_filter(array, window_length, polyorder)


class Georeferencer:
    filepath: str | os.PathLike
    sidescan_file: SidescanFile
    channel: int
    active_utm: bool
    active_export_navdata: bool
    active_blockmedian: bool
    proc_data: np.array
    output_folder: str | os.PathLike
    active_proc_data: bool
    LALO_OUTER: list
    PING: list
    cog_smooth: np.ndarray
    vertical_beam_angle: int
    epsg_code: str
    pix_size: float
    resolution: float
    search_radius: float
    LOLA_plt: np.ndarray
    HEAD_plt: np.ndarray
    LOLA_plt_ori: np.ndarray
    HEAD_plt_ori: np.ndarray
    cable_out: float
    x_offset: float
    y_offset: float

    def __init__(
        self,
        filepath: str | os.PathLike,
        channel: int = 0,
        active_utm: bool = True,
        active_export_navdata: bool = False,
        active_blockmedian: bool = True,
        proc_data=None,
        nav=[],
        output_folder: str | os.PathLike = "./georef_out",
        vertical_beam_angle: int = 60,
        pix_size: float = 0.0,
        resolution: float = 0.0,
        search_radius: float = 0.0,
        cable_out: float = 0.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        geometry_settings: GeometrySettings | None = None,
        sidescan_file: SidescanFile | None = None,
        prepared_track: PreparedTrackGeometry | None = None,
    ):
        self.filepath = Path(filepath)
        if sidescan_file is None:
            from sidescantools.sidescan_file import SidescanFile

            sidescan_file = SidescanFile(self.filepath)
        self.sidescan_file = sidescan_file
        self.channel = channel
        self.active_utm = active_utm
        self.active_export_navdata = active_export_navdata
        self.active_blockmedian = active_blockmedian
        self.output_folder = Path(output_folder)
        self.geometry_settings = geometry_settings or GeometrySettings(
            vertical_beam_angle=vertical_beam_angle,
            cable_out_m=cable_out,
            x_offset_m=x_offset,
            y_offset_m=y_offset,
        )
        self.vertical_beam_angle = self.geometry_settings.vertical_beam_angle
        self.active_proc_data = False
        self.nav = nav
        self.pix_size = pix_size
        self.resolution = resolution
        self.search_radius = search_radius
        self.LALO_OUTER = []
        self.PING = []
        self.cog_smooth = np.empty_like(proc_data)
        self.LOLA_plt = np.empty_like(proc_data)
        self.HEAD_plt = np.empty_like(proc_data)
        self.LOLA_plt_ori = np.empty_like(proc_data)
        self.HEAD_plt_ori = np.empty_like(proc_data)
        self.cable_out = self.geometry_settings.cable_out_m
        self.x_offset = self.geometry_settings.x_offset_m
        self.y_offset = self.geometry_settings.y_offset_m
        self.swath_geometry: SwathGeometry | None = None
        self.prepared_track = prepared_track
        if proc_data is not None:
            self.proc_data = proc_data
            self.active_proc_data = True
        self.setup_output_folder()
        self.PING = self.sidescan_file.packet_no

    def setup_output_folder(self):
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=False, exist_ok=True)
            if not self.output_folder.exists():
                print(
                    f"Error setting up output folder. Path might be invalid: {self.output_folder}"
                )
                raise FileNotFoundError

    def get_pix_size(self, lo, la, res_factor):
        """
        Calculate distance between pings
        Calculate distance between coordinates in m
        from a middle subset of coordinates array (else it takes very
        long and the distances should be similar throughout the interp. coords)


        Parameters
        -----------
        lo: np.ndarray
            Array of interpolated(!) longitudes
        la: np.ndarray
            Array of interpolated(!) latitudes
        """
        import geopy.distance

        # define subset if array length is larger than 300 pings.
        if len(lo) > 300:
            start = int(len(lo) / 2 - 100)
            stop = int(len(lo) / 2 + 100)
            lo = lo[start:stop]
            la = la[start:stop]

        DIST = np.ones_like(lo)

        for i, (lon, lat, dst) in enumerate(zip(lo, la, DIST)):
            c_a = (lat, lon)
            c_b = (la[i - 1], lo[i - 1])
            DIST[i] = geopy.distance.distance(c_a, c_b).meters

        DIST[0] = np.nan
        # Round pixel resolution to 3 decimals and multiply by *factor* else too small
        self.pix_size = np.round(np.nanmedian(DIST), 3) * res_factor

        # Set first value to avoid jumps

    def calculate_cog(self, lo, la, ping_unique, ping_uniform):
        """
        Calculate Course over Ground (COG)/true heading based on difference between single coordinates.
        Note that coordinates must be unique!

        Parameters
        ----------
        lo: np.ndarray
            Longitude or Easting, unique and smoothed (savgol filteres) if neccessary
        la: np.ndarray
            Latitude or Northing, unique and smoothed (savgol filteres) if neccessary
        ping_unique:  np.ndarray
            Array of unique pings (without duplicates) to build spline
        ping_uniform:  np.ndarray
            Ping array for original length with monotonous ping numbers to evaluate spline
        """
        lo = np.asarray(lo, dtype=float)
        la = np.asarray(la, dtype=float)
        ping_unique = np.asarray(ping_unique, dtype=float)
        if len(ping_unique) < 2:
            raise ValueError("At least two unique navigation fixes are required.")

        longitude_difference = np.diff(lo, prepend=np.nan)
        latitude_difference = np.diff(la, prepend=np.nan)
        cog = np.arctan2(latitude_difference, longitude_difference)
        cog[0] = cog[1]
        cog = np.unwrap(cog)
        cog = np.rad2deg(cog)

        # Garmin and some modern systems store a unique navigation fix at
        # almost every ping. Fitting FITPACK's global smoothing spline through
        # tens of thousands of already-dense fixes takes tens of seconds and
        # adds no missing positions. Use linear interpolation for dense tracks;
        # the Savitzky-Golay pass below still provides the established heading
        # smoothing. Retain the legacy spline for genuinely sparse navigation.
        dense_navigation = (
            len(ping_unique) >= 1024
            and len(ping_unique) >= 0.8 * len(ping_uniform)
        )
        if dense_navigation:
            cog_intp = np.interp(ping_uniform, ping_unique, cog)
        else:
            spline_degree = min(3, len(ping_unique) - 1)
            cog_spl = interpolate.UnivariateSpline(
                ping_unique, cog, k=spline_degree, s=len(ping_unique) / 2
            )
            cog_intp = cog_spl(ping_uniform)
        self.cog_smooth = _safe_savgol_filter(cog_intp, 100, 3)

    def _prepare_track_geometry(self) -> PreparedTrackGeometry:
        """Prepare navigation shared by port and starboard geometry."""

        source_ping = np.asarray(self.sidescan_file.packet_no).reshape(-1)
        longitude = np.asarray(self.sidescan_file.longitude, dtype=float).reshape(-1)
        latitude = np.asarray(self.sidescan_file.latitude, dtype=float).reshape(-1)
        heading = np.asarray(self.sidescan_file.sensor_heading, dtype=float).reshape(-1)
        coordinate_candidate = (
            np.isfinite(longitude)
            & np.isfinite(latitude)
            & (longitude != 0)
            & (latitude != 0)
        )
        valid_ping_mask = (
            coordinate_candidate
            & (np.abs(longitude) <= 180)
            & (latitude >= -80)
            & (latitude <= 84)
        )
        conversion_failures = int(
            np.count_nonzero(coordinate_candidate & ~valid_ping_mask)
        )
        if conversion_failures:
            logger.warning(
                "%s: dropped %d navigation fix(es) outside the UTM domain",
                Path(self.filepath).name,
                conversion_failures,
            )

        original_lon = longitude[valid_ping_mask]
        original_lat = latitude[valid_ping_mask]
        original_heading = heading[valid_ping_mask]
        filtered_ping = source_ping[valid_ping_mask]
        if len(original_lon) < 2:
            raise ValueError("At least two valid navigation fixes are required.")

        unique_mask = np.ones(len(original_lon), dtype=bool)
        unique_mask[1:] = (original_lon[1:] != original_lon[:-1]) | (
            original_lat[1:] != original_lat[:-1]
        )
        lon_unique = original_lon[unique_mask]
        lat_unique = original_lat[unique_mask]
        ping_unique = np.flatnonzero(unique_mask)
        if len(lon_unique) < 2:
            raise ValueError("At least two valid, unique navigation fixes are required.")

        # Determine one projected CRS for the complete survey, then transform
        # every coordinate in compiled PROJ code. Garmin commonly stores a new
        # fix for every ping; the prior Python-level utm.from_latlon/to_latlon
        # loops dominated file loading for those surveys.
        try:
            _east, _north, zone, letter = utm.from_latlon(
                float(lat_unique[0]), float(lon_unique[0])
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError("Unable to determine a UTM zone for sonar navigation") from exc
        crs = CRS.from_dict(
            {"proj": "utm", "zone": zone, "south": letter < "N"}
        )
        epsg = crs.to_authority()
        if epsg is None:
            raise ValueError("Unable to determine a projected CRS for sonar navigation.")
        epsg_code = f"{epsg[0]}:{epsg[1]}"
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        east, north = forward.transform(lon_unique, lat_unique)
        east = np.asarray(east, dtype=float)
        north = np.asarray(north, dtype=float)
        valid_projected = np.isfinite(east) & np.isfinite(north)
        if not np.all(valid_projected):
            dropped = int(np.count_nonzero(~valid_projected))
            logger.warning(
                "%s: dropped %d navigation fix(es) that could not be projected",
                Path(self.filepath).name,
                dropped,
            )
            east = east[valid_projected]
            north = north[valid_projected]
            ping_unique = ping_unique[valid_projected]
        if len(east) < 2:
            raise ValueError("At least two valid, unique navigation fixes are required.")

        ping_uniform = np.arange(len(filtered_ping), dtype=float)
        ping_uniform = np.clip(ping_uniform, ping_unique[0], ping_unique[-1])
        self.calculate_cog(east, north, ping_unique, ping_uniform)
        cog_smooth = np.asarray(self.cog_smooth, dtype=float)

        heading_radians = np.deg2rad(cog_smooth[ping_unique])
        layback = self.geometry_settings.effective_layback_m
        east_offset = (
            east
            - layback * np.sin(heading_radians)
            + self.x_offset * np.cos(heading_radians)
        )
        north_offset = (
            north
            - layback * np.cos(heading_radians)
            + self.y_offset * np.sin(heading_radians)
        )
        lon_offset, lat_offset = inverse.transform(east_offset, north_offset)

        spline_degree = min(3, len(ping_unique) - 1)
        lon_spline = interpolate.make_interp_spline(
            ping_unique, lon_offset, k=spline_degree
        )
        lat_spline = interpolate.make_interp_spline(
            ping_unique, lat_offset, k=spline_degree
        )
        east_spline = interpolate.make_interp_spline(
            ping_unique, east_offset, k=spline_degree
        )
        north_spline = interpolate.make_interp_spline(
            ping_unique, north_offset, k=spline_degree
        )
        nadir_lon = _safe_savgol_filter(lon_spline(ping_uniform), 100, 2)
        nadir_lat = _safe_savgol_filter(lat_spline(ping_uniform), 100, 2)
        nadir_east = _safe_savgol_filter(east_spline(ping_uniform), 100, 2)
        nadir_north = _safe_savgol_filter(north_spline(ping_uniform), 100, 2)

        return PreparedTrackGeometry(
            valid_ping_mask=valid_ping_mask,
            source_ping=filtered_ping,
            original_lon=original_lon,
            original_lat=original_lat,
            original_heading=original_heading,
            ping_unique=ping_unique,
            cog_smooth=cog_smooth,
            nadir_lon=np.asarray(nadir_lon),
            nadir_lat=np.asarray(nadir_lat),
            nadir_east=np.asarray(nadir_east),
            nadir_north=np.asarray(nadir_north),
            inverse_transformer=inverse,
            epsg_code=epsg_code,
        )

    def prep_data(self, *, build_bulk_nav=True):
        if self.active_proc_data:
            swath_width = len(self.proc_data[0])
        else:
            swath_width = len(self.sidescan_file.data[self.channel][0])

        track = getattr(self, "prepared_track", None)
        if track is None:
            track = self._prepare_track_geometry()
            self.prepared_track = track
        self.PING = track.source_ping
        self.cog_smooth = track.cog_smooth
        self.epsg_code = track.epsg_code

        slant_range = np.asarray(
            self.sidescan_file.slant_range[self.channel], dtype=float
        ).reshape(-1)[track.valid_ping_mask]
        ground_range = (
            math.sin(math.radians(self.vertical_beam_angle)) * slant_range
        )
        heading_radians = np.deg2rad(track.cog_smooth)
        direction = 1.0 if self.channel == 1 else -1.0
        east_outer = (
            track.nadir_east
            + direction * ground_range * np.sin(heading_radians)
        )
        north_outer = (
            track.nadir_north
            - direction * ground_range * np.cos(heading_radians)
        )
        east_outer = _safe_savgol_filter(east_outer, 300, 2)
        north_outer = _safe_savgol_filter(north_outer, 300, 2)
        outer_lon, outer_lat = track.inverse_transformer.transform(
            east_outer, north_outer
        )
        outer_lon = np.asarray(outer_lon, dtype=float)
        outer_lat = np.asarray(outer_lat, dtype=float)
        self.LALO_OUTER = list(zip(outer_lat, outer_lon))

        x = np.arange(len(track.cog_smooth))
        x_original = np.arange(len(track.original_heading))
        self.HEAD_plt = np.column_stack((x, track.cog_smooth))
        self.HEAD_plt_ori = np.column_stack((x_original, track.original_heading))
        self.LOLA_plt = np.column_stack((track.nadir_lon, track.nadir_lat))
        self.LOLA_plt_ori = np.column_stack(
            (track.original_lon, track.original_lat)
        )

        def align_to_original(values):
            aligned = np.full(len(track.valid_ping_mask), np.nan, dtype=float)
            aligned[track.valid_ping_mask] = values
            return aligned

        self.swath_geometry = SwathGeometry(
            channel=self.channel,
            sample_count=swath_width,
            valid_ping_mask=track.valid_ping_mask,
            nadir_lon=align_to_original(track.nadir_lon),
            nadir_lat=align_to_original(track.nadir_lat),
            outer_lon=align_to_original(outer_lon),
            outer_lat=align_to_original(outer_lat),
            slant_range_m=align_to_original(slant_range),
            ground_range_m=align_to_original(ground_range),
            geometry_settings=self.geometry_settings,
        )
        if build_bulk_nav:
            self.nav = self.swath_geometry.coordinates_for_all_samples()

    def prepare_swath_geometry(self, *, force=False, build_bulk_nav=False):
        """Prepare reusable geometry, optionally materializing raster coordinates."""

        if force or self.swath_geometry is None:
            self.prep_data(build_bulk_nav=build_bulk_nav)
        elif build_bulk_nav:
            self.nav = self.swath_geometry.coordinates_for_all_samples()
        return self.swath_geometry

    def channel_stack(self):
        """
        Work on raw or processed data, depending on `self.active_proc_data`
        - Norm data to max 255 for pic generation
        """

        # check whether processed data is present
        if self.active_proc_data:
            ch_stack = self.proc_data
        else:
            ch_stack = self.sidescan_file.data[self.channel]

        # Extract metadata for each ping in sonar channel, also longitude to mask invalid values
        lon = self.sidescan_file.longitude
        lon = np.ndarray.flatten(np.array(lon))
        mask_x = lon != 0
        mask_y = np.ones_like(ch_stack[0])

        # 'expand' to match ch_stack shape
        mask = mask_x[:, np.newaxis] * mask_y
        mask = mask.astype(bool)

        # Extract valid pings (same like ZERO mask for coordinates)
        ch_stack = ch_stack[np.all(mask, axis=1)]
        swath_len = len(lon)
        swath_width = len(ch_stack[0])
        print(f"swath_len: {swath_len}, swath_width: {swath_width}")

        ch_stack = np.array(ch_stack, dtype=float)

        # Hack for alter transparency
        ch_stack /= np.max(np.abs(ch_stack)) / 254
        ch_stack = np.clip(ch_stack, 1, 255)

        # Channel 0 is normalized as port by every registered reader.
        if self.channel == 0:
            ch_stack = np.flip(ch_stack, axis=1)

        ch_stack_flat = np.ndarray.flatten(ch_stack)

        return ch_stack_flat.astype(np.uint8)

    @staticmethod
    def write_img(im_path, data, alpha=None):
        # flip data to show first ping at bottom
        data = np.flipud(data)
        image_to_write = Image.fromarray(data)
        if alpha is not None:
            alpha = Image.fromarray(alpha)
            image_to_write.putalpha(alpha)
        png_info = PngInfo()
        png_info.add_text("Info", "Generated by SonarSuite")
        image_to_write.save(im_path, pnginfo=png_info)

    def georeference(self, bs_data, progress_signal=None):
        """
        Method to georeference point cloud data.
        Uses pygmt to get region from xyz data by using coordinate precision as spacing.
        Runs blockmedian to reduce data size and nearneighbor on blockmedian output
        to produce final interpolated grid.
        Output from nearneighbor is of type xarray so it can directly
        be used with rioxarray to assign CRS and save to geotiff.

        Parameters
        ----------
        bs_data: np.ndarray
            1D array of backscatter data (can be amplitudes or greyscales)
        """

        import pygmt
        import rioxarray  # noqa: F401 - registers the xarray ``.rio`` accessor

        # Determine pixel size based on minimum distance between coordinates
        self.get_pix_size(self.nav[:, 0], self.nav[:, 1], res_factor=1)

        resolution = f"{self.resolution}e"
        search_radius = f"{self.search_radius}e"

        # Define output file names
        # Convert resolution to mm to avoid "." in file name
        out_median = (
            self.output_folder / f"outmedian_{self.filepath.stem}_ch{self.channel}.xyz"
        )
        if self.resolution < 1.0:
            res_name = str(int(self.resolution * 100)) + "mm"
        else:
            res_name = str(int(self.resolution)) + "m"
        if self.active_utm:
            epsg_name = str(self.epsg_code).replace(":", "")
            out_tiff = (
                self.output_folder
                / f"{self.filepath.stem}_{res_name}_ch{self.channel}_{epsg_name}.tif"
            )
        else:
            out_tiff = (
                self.output_folder
                / f"{self.filepath.stem}_{res_name}_ch{self.channel}_EPSG4326.tif"
            )

        xybs = np.column_stack((self.nav, bs_data))
        crd = Decimal(xybs[0, 0])
        dgts = len(str(crd).split(".")[1]) - 2
        prec = 1 / (10**dgts)
        region = pygmt.info(self.nav, per_column=True, spacing=(prec, prec))

        if self.active_utm:
            print(
                f"Georeferencing with resolution {str(resolution).strip('e')}m and {str(self.search_radius).strip('e')}m in {self.epsg_code}."
            )
        else:
            print(
                f"Georeferencing with resolution {str(resolution).strip('e')}m and {str(self.search_radius).strip('e')}m in WGS84/EPSG:4326."
            )

        if self.active_blockmedian:
            print("Applying GMT Blockmedian...")
            pygmt.blockmedian(
                data=xybs,
                outfile=out_median,
                output_type="file",
                coltypes="fg",
                spacing=resolution,
                region=region,
                binary="o3d",
            )

            if progress_signal is not None:
                progress_signal.emit((1000 / len(bs_data)) * 0.05)

            print("Applying GMT Nearneighbour alg...")
            data_nn = pygmt.nearneighbor(
                data=out_median,
                coltypes="fg",
                region=region,
                binary="i3d",
                spacing=resolution,
                search_radius=search_radius,
            )
        else:
            print("Blockmedian off, using nearneighbor only")
            data_nn = pygmt.nearneighbor(
                data=xybs,
                coltypes="fg",
                region=region,
                binary="i3d",
                spacing=resolution,
                search_radius=search_radius,
            )

        # Clip data to range between 0 - 256
        data_clp = data_nn.clip(min=0.0, max=255.0)

        # Reproject to utm if applied and save to geotiff
        if self.active_utm:
            print(f"Saving to: {out_tiff}")
            data_pr = data_clp.rio.write_crs("EPSG:4326", inplace=True)
            data_rpr = data_pr.rio.reproject(self.epsg_code)
            data_rpr.rio.to_raster(out_tiff, compress="deflate", tiled=True)

        # Save as geotiff (set epsg_code for filename)
        else:
            print(f"Saving to: {out_tiff}")
            self.epsg_code = "EPSG:4326"
            data_pr = data_clp.rio.write_crs(self.epsg_code, inplace=True)
            data_pr.rio.to_raster(out_tiff, compress="deflate", tiled=True)

        if progress_signal is not None:
            progress_signal.emit(0.5)

    def process(self, progress_signal=None):
        # Check if enough data are present, otherwise quit
        if len(self.PING) >= 2:
            self.prepare_swath_geometry(force=True, build_bulk_nav=True)
            chan_stack_flat = self.channel_stack()

            self.georeference(bs_data=chan_stack_flat, progress_signal=progress_signal)

            # Export navigation data
            if self.active_export_navdata:
                xyz = np.column_stack((self.nav, chan_stack_flat))
                nav_ch = (
                    self.output_folder
                    / f"Navigation_{self.filepath.stem}_ch{self.channel}.csv"
                )
                print(f"Saving navinfo to {nav_ch}")

                np.savetxt(
                    nav_ch,
                    xyz,
                    fmt="%s",
                    delimiter=";",
                    header="Nadir Longitude; Nadir Latitude; BS",
                )
        else:
            raise ValueError(
                f"At least two pings are required; found {len(self.PING)}."
            )


def main():
    parser = argparse.ArgumentParser(description="Tool to process sidescan sonar data")
    parser.add_argument("xtf", metavar="FILE", help="Path to a supported sonar file")
    parser.add_argument(
        "channel",
        type=int,
        default=0,
        help="Channel number (can be 0 or 1, default: 0)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="Output raster resolution",
    )
    parser.add_argument(
        "--search_radius",
        type=float,
        default=0.2,
        help="Search Radius for output raster creation. Usually 2 * resolution.",
    )
    parser.add_argument(
        "--UTM",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uses UTM projection rather than WGS84. Default is UTM",
    )
    parser.add_argument(
        "--navdata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, exports navigation data to csv",
    )
    parser.add_argument(
        "--blockmedian",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, uses blockmedian before nearneighbour alg. for gridding to reduce noise and data size. Default True.",
    )

    args = parser.parse_args()
    print("args:", args)

    georeferencer = Georeferencer(
        filepath=args.xtf,
        channel=args.channel,
        active_utm=args.UTM,
        active_export_navdata=args.navdata,
        active_blockmedian=args.blockmedian,
        resolution=args.resolution,
        search_radius=args.search_radius,
    )
    georeferencer.process()


if __name__ == "__main__":
    main()
