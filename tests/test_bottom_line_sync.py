"""Regression coverage for the flat (portside_bottom_dist/starboard_bottom_dist)
vs. chunked (napari_portside_bottom/napari_starboard_bottom) bottom-line
representations staying in sync. slant_range_correction() reads only the
flat arrays; every interactive/chunk-granularity mutator writes only the
chunked ones -- without sync_chunked_bottom_to_flat()/
sync_flat_bottom_to_chunked(), a correction never reaches the corrected
output even though the on-screen overlay looks right.
"""

import numpy as np

from sidescantools.sidescan_preproc import SidescanPreprocessor


class _SyntheticSidescanFile:
    def __init__(self, num_ping=7, ping_len=6):
        generator = np.random.default_rng(0)
        self.data = generator.integers(
            1, 1000, size=(2, num_ping, ping_len), dtype=np.int16
        )
        self.ping_len = ping_len
        self.num_ping = num_ping


def _preprocessor(num_ping=7, ping_len=6, chunk_size=3):
    sidescan_file = _SyntheticSidescanFile(num_ping=num_ping, ping_len=ping_len)
    preproc = SidescanPreprocessor(
        sidescan_file, chunk_size=chunk_size, downsampling_factor=1
    )
    # Bottom-line arrays are normally allocated by init_napari_bottom_detect();
    # allocated directly here so these tests exercise only the sync methods,
    # not the (expensive, threshold-dependent) detection algorithms.
    preproc.napari_portside_bottom = np.zeros((preproc.num_chunk, chunk_size), dtype=int)
    preproc.napari_starboard_bottom = np.zeros((preproc.num_chunk, chunk_size), dtype=int)
    preproc.bottom_map = np.zeros((preproc.num_chunk, chunk_size, 2 * ping_len))
    return preproc


def test_sync_chunked_bottom_to_flat_rebuilds_whole_file_by_default():
    preproc = _preprocessor(num_ping=7, chunk_size=3)
    preproc.napari_portside_bottom[:] = [[1, 2, 3], [4, 5, 6], [7, 0, 0]]
    preproc.napari_starboard_bottom[:] = [[10, 20, 30], [40, 50, 60], [70, 0, 0]]

    preproc.sync_chunked_bottom_to_flat()

    np.testing.assert_array_equal(preproc.portside_bottom_dist, [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(
        preproc.starboard_bottom_dist, [10, 20, 30, 40, 50, 60, 70]
    )


def test_sync_chunked_bottom_to_flat_updates_only_the_given_chunk():
    preproc = _preprocessor(num_ping=7, chunk_size=3)
    preproc.portside_bottom_dist = np.array([9, 9, 9, 9, 9, 9, 9])
    preproc.starboard_bottom_dist = np.array([9, 9, 9, 9, 9, 9, 9])
    preproc.napari_portside_bottom[1] = [4, 5, 6]
    preproc.napari_starboard_bottom[1] = [40, 50, 60]

    preproc.sync_chunked_bottom_to_flat(chunk_idx=1)

    np.testing.assert_array_equal(preproc.portside_bottom_dist, [9, 9, 9, 4, 5, 6, 9])
    np.testing.assert_array_equal(
        preproc.starboard_bottom_dist, [9, 9, 9, 40, 50, 60, 9]
    )


def test_sync_flat_bottom_to_chunked_rebuilds_chunked_arrays_and_bottom_map():
    # Values must stay within [0, ping_len) -- they're sample indices into a
    # ping_len-wide array, not arbitrary integers.
    preproc = _preprocessor(num_ping=7, ping_len=6, chunk_size=3)
    preproc.portside_bottom_dist = np.array([1, 2, 3, 4, 5, 4, 3])
    preproc.starboard_bottom_dist = np.array([1, 1, 1, 1, 1, 1, 1])

    preproc.sync_flat_bottom_to_chunked()

    np.testing.assert_array_equal(preproc.napari_portside_bottom[0], [1, 2, 3])
    np.testing.assert_array_equal(preproc.napari_portside_bottom[1], [4, 5, 4])
    assert preproc.napari_portside_bottom[2, 0] == 3
    # bottom_map is rebuilt from the flat arrays via build_bottom_line_map();
    # the sample column itself is always marked regardless of whether that
    # helper widens the line by +/-1 near an edge.
    for ping in range(7):
        port_column = int(preproc.portside_bottom_dist[ping])
        star_column = preproc.ping_len + int(preproc.starboard_bottom_dist[ping])
        chunk_idx, local_idx = divmod(ping, preproc.chunk_size)
        assert preproc.bottom_map[chunk_idx, local_idx, port_column] == 1
        assert preproc.bottom_map[chunk_idx, local_idx, star_column] == 1


def test_recalculating_whole_file_bottom_line_is_visible_after_resync():
    """Regression test for the flat/chunked divergence: a whole-file
    automatic-detection result (which only ever touches the flat arrays)
    must reach the chunked overlay via sync_flat_bottom_to_chunked(),
    otherwise the interactive view keeps showing the stale line even
    though slant_range_correction() would already see the new one."""
    preproc = _preprocessor(num_ping=4, ping_len=5, chunk_size=2)
    preproc.napari_portside_bottom[:] = [[9, 9], [9, 9]]  # stale chunked state
    preproc.portside_bottom_dist = np.array([1, 1, 1, 1])
    preproc.starboard_bottom_dist = np.array([2, 2, 2, 2])

    preproc.sync_flat_bottom_to_chunked()

    np.testing.assert_array_equal(
        preproc.napari_portside_bottom.flatten()[:4], [1, 1, 1, 1]
    )
