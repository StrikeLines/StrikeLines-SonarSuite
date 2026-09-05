"""save_bottom_info/load_bottom_info are a compatibility contract: the same
<file>_bottom_info.npz format is read independently by egn_table_build.py
and custom_threading.py, and now by both the Napari and Qt interactive
editors. These tests pin the 3-key format, the .xtf flip, and the
flat/chunked resync load_bottom_info must perform."""

from pathlib import Path

import numpy as np
import pytest

from sidescantools.bottom_line_io import (
    compute_depth_info,
    load_bottom_info,
    save_bottom_info,
)
from sidescantools.sidescan_preproc import SidescanPreprocessor


class _SyntheticSidescanFile:
    def __init__(self, filepath, num_ping=6, ping_len=5):
        generator = np.random.default_rng(1)
        self.data = generator.integers(
            1, 1000, size=(2, num_ping, ping_len), dtype=np.int16
        )
        self.ping_len = ping_len
        self.num_ping = num_ping
        self.filepath = Path(filepath)
        self.ping_x_axis = np.linspace(0.0, 10.0, ping_len)
        self.depth = np.full(num_ping, 4.0)


def _preprocessor(sidescan_file, chunk_size=4):
    preproc = SidescanPreprocessor(
        sidescan_file, chunk_size=chunk_size, downsampling_factor=1
    )
    preproc.napari_portside_bottom = np.zeros(
        (preproc.num_chunk, chunk_size), dtype=int
    )
    preproc.napari_starboard_bottom = np.zeros(
        (preproc.num_chunk, chunk_size), dtype=int
    )
    preproc.bottom_map = np.zeros(
        (preproc.num_chunk, chunk_size, 2 * preproc.ping_len)
    )
    return preproc


def test_save_and_load_round_trip_preserves_values_for_jsf(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf", num_ping=6)
    preproc = _preprocessor(sidescan_file)
    preproc.napari_portside_bottom[:] = [[1, 2, 3, 4], [5, 6, 0, 0]]
    preproc.napari_starboard_bottom[:] = [[10, 20, 30, 40], [50, 60, 0, 0]]
    npz_path = tmp_path / "line_bottom_info.npz"

    save_bottom_info(npz_path, preproc, sidescan_file)

    reloaded = _preprocessor(sidescan_file)
    load_bottom_info(npz_path, reloaded, sidescan_file)

    np.testing.assert_array_equal(
        reloaded.napari_portside_bottom.flatten()[:6], [1, 2, 3, 4, 5, 6]
    )
    np.testing.assert_array_equal(
        reloaded.napari_starboard_bottom.flatten()[:6], [10, 20, 30, 40, 50, 60]
    )
    # load_bottom_info must resync the flat arrays too -- this is exactly
    # what slant_range_correction() reads.
    np.testing.assert_array_equal(reloaded.portside_bottom_dist, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(
        reloaded.starboard_bottom_dist, [10, 20, 30, 40, 50, 60]
    )


def test_save_and_load_flips_order_for_xtf_files(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.xtf", num_ping=6)
    preproc = _preprocessor(sidescan_file)
    preproc.napari_portside_bottom[:] = [[1, 2, 3, 4], [5, 6, 0, 0]]
    preproc.napari_starboard_bottom[:] = [[10, 20, 30, 40], [50, 60, 0, 0]]
    npz_path = tmp_path / "line_bottom_info.npz"

    save_bottom_info(npz_path, preproc, sidescan_file)
    with np.load(npz_path) as saved:
        # Saved to disk already flipped for .xtf ("backwards compatibility").
        np.testing.assert_array_equal(saved["bottom_info_port"], [6, 5, 4, 3, 2, 1])

    reloaded = _preprocessor(sidescan_file)
    load_bottom_info(npz_path, reloaded, sidescan_file)

    # Flipped back on load -- round trip through an .xtf file is transparent.
    np.testing.assert_array_equal(
        reloaded.napari_portside_bottom.flatten()[:6], [1, 2, 3, 4, 5, 6]
    )


def test_load_bottom_info_raises_for_a_missing_file(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf")
    preproc = _preprocessor(sidescan_file)

    with pytest.raises(FileNotFoundError):
        load_bottom_info(tmp_path / "does_not_exist.npz", preproc, sidescan_file)


def test_compute_depth_info_returns_none_when_no_depth_logged(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf")
    sidescan_file.depth = np.zeros(sidescan_file.num_ping)

    assert compute_depth_info(sidescan_file, downsampling_factor=1) is None


def test_compute_depth_info_converts_meters_to_sample_index(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf", num_ping=3)
    sidescan_file.ping_x_axis = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    sidescan_file.depth = np.array([4.0, 6.0, 0.1])

    depth_info = compute_depth_info(sidescan_file, downsampling_factor=1)

    np.testing.assert_array_equal(depth_info, [2, 3, 0])


def test_compute_depth_info_does_not_mutate_logged_depth(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf", num_ping=3)
    sidescan_file.depth = np.array([4.0, 6.0, 2.0])
    original = sidescan_file.depth.copy()

    compute_depth_info(sidescan_file, downsampling_factor=1)

    np.testing.assert_array_equal(sidescan_file.depth, original)


def test_compute_depth_info_uses_later_valid_depths(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf", num_ping=3)
    sidescan_file.ping_x_axis = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    sidescan_file.depth = np.array([0.0, 6.0, 4.0])

    depth_info = compute_depth_info(sidescan_file, downsampling_factor=1)

    np.testing.assert_array_equal(depth_info, [0, 3, 2])


def test_compute_depth_info_rejects_invalid_downsampling_factor(tmp_path):
    sidescan_file = _SyntheticSidescanFile(tmp_path / "line.jsf")

    with pytest.raises(ValueError, match="downsampling_factor"):
        compute_depth_info(sidescan_file, downsampling_factor=0)
