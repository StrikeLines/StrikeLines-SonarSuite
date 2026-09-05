from sidescantools.sidescan_file import SidescanFile
from sidescantools.sidescan_preproc import SidescanPreprocessor
import numpy as np
import os
import pathlib


def fill_missing_slant_ranges(slant_ranges: np.ndarray) -> np.ndarray:
    """Return slant ranges with missing entries filled from nearby valid data.

    Interior and trailing gaps use the preceding range. Leading gaps use the
    first valid range in that channel. The input array is not modified.
    """

    filled = np.asarray(slant_ranges, dtype=float).copy()
    if filled.ndim != 2:
        raise ValueError("slant_ranges must be a channels-by-pings array")

    for channel in filled:
        valid = np.isfinite(channel) & (channel > 0)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            continue
        first_valid = int(valid_indices[0])
        channel[:first_valid] = channel[first_valid]
        for index in range(first_valid + 1, len(channel)):
            if not np.isfinite(channel[index]) or channel[index] <= 0:
                channel[index] = channel[index - 1]
    return filled


def accumulate_egn_bins(egn_mat, egn_hit_cnt, r_idx, alpha_idx, values, r_size, angle_num):
    """Scatter-accumulate one ping's samples into the EGN mat/hit-count bins.

    Equivalent to, for each sample i: if 0 <= r_idx[i] < r_size and
    0 <= alpha_idx[i] < angle_num and values[i] != 0, then
    egn_mat[r_idx[i], alpha_idx[i]] += values[i] and
    egn_hit_cnt[r_idx[i], alpha_idx[i]] += 1. Vectorized with np.add.at
    (unbuffered) rather than plain fancy-index += so that samples that land
    in the same bin within one ping are correctly summed instead of only the
    last one being kept -- a Python for-loop over every sample here was the
    dominant cost of building an EGN table on a full-resolution survey file.
    """

    valid = (
        (r_idx >= 0)
        & (r_idx < r_size)
        & (alpha_idx >= 0)
        & (alpha_idx < angle_num)
        & (values != 0)
    )
    if np.any(valid):
        rows = r_idx[valid]
        cols = alpha_idx[valid]
        np.add.at(egn_mat, (rows, cols), values[valid])
        np.add.at(egn_hit_cnt, (rows, cols), 1)


# Generate the EGN info for a specific file
def generate_egn_info(
    filename: str,
    bottom_file: str,
    out_path: str,
    chunk_size: int,
    nadir_angle: int,
    active_intern_depth: bool,
    active_bottom_detection_downsampling: bool,
    progress_signal=None,
    egn_table_parameters=[360, 2],
):
    print("---")
    print(f"Reading file: {filename}")

    sidescan_file = SidescanFile(filename)

    # check if bottom_file exists otherwise switch to intern altitude
    if bottom_file is not None and active_intern_depth == False:
        if pathlib.Path(bottom_file).exists():
            with np.load(bottom_file) as archive:
                bottom_info = {key: archive[key].copy() for key in archive.files}
        else:
            active_intern_depth = True
    else:
        active_intern_depth = True

    # Slant ranges might be incomplete; do not replace an interior gap with
    # the final range from the file, which creates an artificial discontinuity.
    sidescan_file.slant_range = fill_missing_slant_ranges(
        sidescan_file.slant_range
    )

    # Check if downsampling was applied
    downsampling_factor = 1
    if active_intern_depth == False:
        try:
            downsampling_factor = bottom_info["downsampling_factor"]
        except KeyError:
            downsampling_factor = 1

        portside_bottom_dist = bottom_info["bottom_info_port"].flatten()[
            : sidescan_file.num_ping
        ]
        starboard_bottom_dist = bottom_info["bottom_info_star"].flatten()[
            : sidescan_file.num_ping
        ]
        # flip order for xtf files to contain backwards compability
        filepath = pathlib.Path(filename)
        if filepath.suffix.casefold() == ".xtf":
            portside_bottom_dist = np.flip(portside_bottom_dist)
            starboard_bottom_dist = np.flip(starboard_bottom_dist)

        # If a bottom line distance value is 0, just use val from other side (solve that bug in bottom line detection)
        num_btm_line = len(portside_bottom_dist)
        for btm_idx in range(num_btm_line):
            if portside_bottom_dist[btm_idx] == 0:
                portside_bottom_dist[btm_idx] = starboard_bottom_dist[btm_idx]
            elif starboard_bottom_dist[btm_idx] == 0:
                starboard_bottom_dist[btm_idx] = portside_bottom_dist[btm_idx]

        if len(portside_bottom_dist) != len(starboard_bottom_dist):
            print(
                f"Reading bottom info {bottom_file}: detected bottom line lengths don't match!"
            )
            return False

        # check that data length and bottom detection length match
        if sidescan_file.num_ping != len(portside_bottom_dist):
            # if lengths don't match, bottom line might be padded to fill the last last chunk
            try:
                bottom_chunk_size = bottom_info["chunk_size"]
            except KeyError:
                bottom_chunk_size = chunk_size
            expected_full_chunk_size = int(
                np.ceil(sidescan_file.num_ping / bottom_chunk_size) * bottom_chunk_size
            )
            if len(portside_bottom_dist) != expected_full_chunk_size:
                print(
                    f"Sizes of NUM ping ({sidescan_file.num_ping}) and bottom line info ({len(portside_bottom_dist)}) don't match!"
                )
                print(f"Sonar file: {filename}")
                print(f"Bottom line detection: {bottom_file}")
                return False

    # Check whether data shall be downsampled using the bottom line detection factor
    if active_bottom_detection_downsampling:
        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=chunk_size,
            downsampling_factor=downsampling_factor,
        )
    else:
        preproc = SidescanPreprocessor(
            sidescan_file=sidescan_file,
            chunk_size=chunk_size,
            downsampling_factor=1,
        )
        if active_intern_depth == False:
            # rescale bottom info
            portside_bottom_dist = portside_bottom_dist * downsampling_factor
            starboard_bottom_dist = starboard_bottom_dist * downsampling_factor

    if active_intern_depth == False:
        preproc.portside_bottom_dist = portside_bottom_dist
        preproc.starboard_bottom_dist = starboard_bottom_dist

    # slant range correction
    preproc.slant_range_correction(
        active_interpolation=True,
        nadir_angle=nadir_angle,
        use_intern_altitude=active_intern_depth,
        progress_signal=progress_signal,
    )

    # EGN parameters - these are not displayed in the UI to keep it simple
    angle_range = [-1 * np.pi / 2, np.pi / 2]
    angle_num = egn_table_parameters[0]
    r_reduc_factor = egn_table_parameters[1]
    r_size = int(preproc.ping_len * 1.1 / r_reduc_factor)
    angle_stepsize = (angle_range[1] - angle_range[0]) / angle_num
    egn_mat = np.zeros((r_size, angle_num))
    egn_hit_cnt = np.zeros((r_size, angle_num))

    # either use depth from annotation file or intern depth
    if active_intern_depth:
        stepsize = sidescan_file.slant_range[0, :] / preproc.ping_len
        mean_depth = sidescan_file.depth / stepsize  # is the same for both sides
    else:
        mean_depth = np.array(np.round(np.mean(preproc.dep_info, 0)), dtype=int)

    num_data = np.shape(preproc.slant_corrected_mat)[0]

    dd = mean_depth**2
    EPS = np.finfo(float).eps

    for vector_idx in range(num_data):
        if vector_idx % 1000 == 0:
            if progress_signal is not None:
                progress_signal.emit(1000 / num_data * 0.5)
            print(f"EGN Progress: {vector_idx/num_data:.2%}")
        r = np.sqrt(
            (
                np.linspace(0, 2 * preproc.ping_len - 1, 2 * preproc.ping_len)
                - preproc.ping_len
            )
            ** 2
            + dd[vector_idx]
        )
        r_idx = np.array(np.round(r / r_reduc_factor), dtype=int)
        alpha = np.sign(
            np.linspace(0, 2 * preproc.ping_len - 1, 2 * preproc.ping_len)
            - preproc.ping_len
        ) * np.arccos(mean_depth[vector_idx] / (r + EPS))
        alpha_idx = np.array(
            np.round(alpha / angle_stepsize) + angle_num / 2, dtype=int
        )
        accumulate_egn_bins(
            egn_mat,
            egn_hit_cnt,
            r_idx,
            alpha_idx,
            preproc.slant_corrected_mat[vector_idx],
            r_size,
            angle_num,
        )
    np.savez(
        out_path,
        egn_mat=egn_mat,
        egn_hit_cnt=egn_hit_cnt,
        angle_range=angle_range,
        angle_num=angle_num,
        angle_stepsize=angle_stepsize,
        ping_len=preproc.ping_len,
        r_size=r_size,
        r_reduc_factor=r_reduc_factor,
        nadir_angle=nadir_angle,
    )


def generate_egn_table_from_infos(egn_path_list: list, egn_table_path: os.PathLike):
    if not egn_path_list:
        raise ValueError("At least one EGN info file is required.")

    do_init = True
    for egn_file in egn_path_list:
        with np.load(egn_file) as egn_info:
            if do_init:
                # Copy arrays before the NPZ handle closes.
                full_mat = egn_info["egn_mat"].copy()
                full_hit_cnt = egn_info["egn_hit_cnt"].copy()
                angle_range_init = egn_info["angle_range"].copy()
                angle_num_init = egn_info["angle_num"].copy()
                angle_stepsize_init = egn_info["angle_stepsize"].copy()
                ping_len_init = egn_info["ping_len"].copy()
                r_size_init = egn_info["r_size"].copy()
                r_reduc_factor_init = egn_info["r_reduc_factor"].copy()
                nadir_angle = egn_info["nadir_angle"].copy()

                do_init = False

            elif (
                np.array_equal(angle_range_init, egn_info["angle_range"])
                and angle_num_init == egn_info["angle_num"]
                and angle_stepsize_init == egn_info["angle_stepsize"]
                and ping_len_init == egn_info["ping_len"]
                and r_size_init == egn_info["r_size"]
                and r_reduc_factor_init == egn_info["r_reduc_factor"]
            ):
                full_mat += egn_info["egn_mat"]
                full_hit_cnt += egn_info["egn_hit_cnt"]
            else:
                print(f"EGN Parameter mismatch! Skipping file: {egn_file}")

    # build final egn table
    egn_table = np.divide(
        full_mat, full_hit_cnt, out=np.zeros_like(full_mat), where=full_hit_cnt != 0
    )
    egn_table[np.where(full_hit_cnt == 0)] = np.nan
    print("Saving " + str(egn_table_path))
    np.savez(
        egn_table_path,
        egn_table=egn_table,
        egn_hit_cnt=full_hit_cnt,
        angle_range=angle_range_init,
        angle_num=angle_num_init,
        angle_stepsize=angle_stepsize_init,
        ping_len=ping_len_init,
        r_size=r_size_init,
        r_reduc_factor=r_reduc_factor_init,
        nadir_angle=nadir_angle,
    )
