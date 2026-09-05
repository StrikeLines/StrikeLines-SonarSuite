"""Bottom-line persistence and altitude-seeding shared by every interactive
bottom-tracking UI (Napari and Qt). Extracted from the Napari-only
implementation so both windows read and write the exact same
``<file>_bottom_info.npz`` format that ``egn_table_build.py`` and the CLI
also depend on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sidescantools.sidescan_file import SidescanFile
from sidescantools.sidescan_preproc import SidescanPreprocessor


def compute_depth_info(
    sidescan_file: SidescanFile, downsampling_factor: int
) -> np.ndarray | None:
    """Convert the sonar's logged depth (meters) to a per-ping sample index
    at the given downsampling resolution, for seeding the bottom line via
    ``SidescanPreprocessor.set_depth_from_info``. Returns ``None`` when no
    depth was logged.
    """
    if downsampling_factor <= 0:
        raise ValueError("downsampling_factor must be greater than zero")

    # Work on a copy: converting meters to indices must never overwrite the
    # source metadata used elsewhere for altitude and geometry calculations.
    depth_m = np.asarray(sidescan_file.depth, dtype=float).copy()
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid_depth):
        return None

    ping_axis = np.asarray(sidescan_file.ping_x_axis, dtype=float)
    if ping_axis.size == 0:
        raise ValueError("The sonar sample-distance axis is empty.")

    depth_info = np.zeros(depth_m.shape, dtype=int)
    if ping_axis.size == 1:
        return depth_info

    valid_values = depth_m[valid_depth]
    if np.all(np.diff(ping_axis) >= 0):
        upper = np.clip(
            np.searchsorted(ping_axis, valid_values), 1, len(ping_axis) - 1
        )
        nearest = np.where(
            np.abs(valid_values - ping_axis[upper - 1])
            <= np.abs(valid_values - ping_axis[upper]),
            upper - 1,
            upper,
        )
    else:
        # Unexpected axes still get a correct answer, without mutating the
        # source data, even though the fast searchsorted path is unavailable.
        nearest = np.array(
            [np.argmin(np.abs(value - ping_axis)) for value in valid_values]
        )
    depth_info[valid_depth] = np.round(nearest / downsampling_factor).astype(int)
    return depth_info


def save_bottom_info(
    path: str | os.PathLike,
    preproc: SidescanPreprocessor,
    sidescan_file: SidescanFile,
) -> None:
    """Save the current chunk-granularity bottom line to ``path`` in the
    shared 3-key ``.npz`` format (``bottom_info_port``, ``bottom_info_star``,
    ``downsampling_factor``) that ``egn_table_build.py`` and the CLI also
    read. Outlier removal here only cleans up the persisted copy -- it does
    not feed back into the live preprocessor.
    """
    info_port = preproc.napari_portside_bottom.flatten()[: sidescan_file.num_ping]
    info_star = preproc.napari_starboard_bottom.flatten()[: sidescan_file.num_ping]

    # detect and remove outliers
    sample_thresh = 5
    for ping_idx in range(sidescan_file.num_ping):
        if 0 < ping_idx < sidescan_file.num_ping - 1:
            dist1 = info_port[ping_idx] - info_port[ping_idx - 1]
            dist2 = info_port[ping_idx] - info_port[ping_idx + 1]
            if (
                np.abs(dist1) > sample_thresh
                and np.abs(dist2) > sample_thresh
                and dist1 * dist2 > 0
            ):
                info_port[ping_idx] = int(
                    (info_port[ping_idx - 1] + info_port[ping_idx + 1]) / 2
                )
            dist1 = info_star[ping_idx] - info_star[ping_idx - 1]
            dist2 = info_star[ping_idx] - info_star[ping_idx + 1]
            if (
                np.abs(dist1) > sample_thresh
                and np.abs(dist2) > sample_thresh
                and dist1 * dist2 > 0
            ):
                info_star[ping_idx] = int(
                    (info_star[ping_idx - 1] + info_star[ping_idx + 1]) / 2
                )
    # flip order for xtf files to contain backwards compability
    if sidescan_file.filepath.suffix.casefold() == ".xtf":
        info_port = np.flip(info_port)
        info_star = np.flip(info_star)
    np.savez(
        path,
        bottom_info_port=info_port,
        bottom_info_star=info_star,
        downsampling_factor=preproc.downsampling_factor,
    )


def load_bottom_info(
    path: str | os.PathLike,
    preproc: SidescanPreprocessor,
    sidescan_file: SidescanFile,
) -> None:
    """Load a bottom line saved by ``save_bottom_info`` into ``preproc``,
    overwriting whatever chunk-granularity state (automatic or manual) is
    currently there. Also resyncs the flat arrays so slant-range/BAC/EGN
    processing picks up the loaded line immediately.
    """
    path = Path(path)
    if not (path.exists() and path.suffix == ".npz"):
        raise FileNotFoundError(f"no bottom-line file at {path}")
    with np.load(path) as bottom_info:
        napari_portside_bottom = bottom_info["bottom_info_port"].flatten().copy()
        napari_starboard_bottom = bottom_info["bottom_info_star"].flatten().copy()
    # flip order for xtf files to contain backwards compability
    if sidescan_file.filepath.suffix.casefold() == ".xtf":
        napari_portside_bottom[: sidescan_file.num_ping] = np.flip(
            napari_portside_bottom[: sidescan_file.num_ping]
        )
        napari_starboard_bottom[: sidescan_file.num_ping] = np.flip(
            napari_starboard_bottom[: sidescan_file.num_ping]
        )

    for chunk_idx in range(preproc.num_chunk):
        start = chunk_idx * preproc.chunk_size
        stop = (chunk_idx + 1) * preproc.chunk_size
        port_chunk = napari_portside_bottom[start:stop]
        preproc.napari_portside_bottom[chunk_idx, : len(port_chunk)] = port_chunk
        star_chunk = napari_starboard_bottom[start:stop]
        preproc.napari_starboard_bottom[chunk_idx, : len(star_chunk)] = star_chunk
        preproc.update_bottom_map_napari(chunk_idx, add_line_width=1)
    preproc.sync_chunked_bottom_to_flat()
