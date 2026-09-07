"""Tests for the path-search bottom detector.

The threshold detector these replace could collapse the whole track onto a
constant at one end of the ping (see
``test_detector_never_collapses_onto_a_constant``), so the sweeps here are
deliberately wider than the values the GUI exposes.
"""

from __future__ import annotations

import numpy as np
import pytest

from sidescantools.sidescan_preproc import (
    SidescanPreprocessor,
    bottom_candidates,
    bottom_dp_path,
    bottom_step_scores,
)


PING_LEN = 256
PING_COUNT = 120
RANGE_M = 25.6  # 10 cm per sample, so a sample index doubles as decimeters


class SyntheticSidescanFile:
    """A waterfall with a known bottom, built the way a reader would hand it over."""

    def __init__(self, altitudes, water=8.0, bottom=200.0, noise=2.0, seed=0):
        generator = np.random.default_rng(seed)
        self.ping_len = PING_LEN
        altitudes = np.asarray(altitudes, dtype=int)
        ping_count = len(altitudes)

        # Build both channels nadir-first, then flip port into storage order.
        nadir_first = np.empty((2, ping_count, PING_LEN))
        for channel in range(2):
            for ping, altitude in enumerate(altitudes):
                row = np.full(PING_LEN, water)
                row[altitude:] = bottom
                nadir_first[channel, ping] = row
        nadir_first += generator.normal(0.0, noise, nadir_first.shape)
        nadir_first = np.clip(nadir_first, 1.0, None)

        self.data = np.stack(
            (np.fliplr(nadir_first[0]), nadir_first[1])
        ).astype(np.int16)
        self.slant_range = np.full((2, ping_count), RANGE_M)
        self.altitudes = altitudes

    def add_water_column_blob(self, pings, sample, width=6, level=255.0):
        """Paint a bright false target high in the water column."""

        for ping in pings:
            self.data[1, ping, sample : sample + width] = level
            self.data[0, ping, PING_LEN - sample - width : PING_LEN - sample] = level


def _preprocessor(source):
    return SidescanPreprocessor(source, chunk_size=40, downsampling_factor=1)


def _flat(altitude=70, count=PING_COUNT):
    return SyntheticSidescanFile(np.full(count, altitude))


def test_step_score_peaks_at_the_water_to_bottom_transition():
    source = _flat(altitude=70)
    scores = bottom_step_scores(source.data[1].astype(float))

    peak = int(np.median(np.argmax(scores, axis=1)))
    assert abs(peak - 70) <= 2


def test_step_score_ignores_samples_without_a_full_window_each_side():
    source = _flat()
    scores = bottom_step_scores(source.data[1].astype(float))

    # Without this the bright nadir return itself would win, because nothing
    # sits behind it for the score to be brighter than.
    assert not np.isfinite(scores[:, 0]).any()
    assert not np.isfinite(scores[:, -1]).any()
    assert np.isfinite(scores[:, PING_LEN // 2]).all()


def test_candidates_stay_outside_the_blanked_span():
    source = _flat()
    scores = bottom_step_scores(source.data[1].astype(float))

    indices, _costs = bottom_candidates(scores, blank_samples=90)

    assert indices.min() >= 90


def test_candidates_survive_blanking_that_covers_the_whole_ping():
    source = _flat()
    scores = bottom_step_scores(source.data[1].astype(float))

    indices, costs = bottom_candidates(scores, blank_samples=PING_LEN + 50)

    # Nowhere legal left to look, but every ping must still yield an answer.
    assert indices.shape[0] == PING_COUNT
    assert np.isfinite(costs).all()


def test_path_search_prefers_the_cheaper_candidate_when_smoothing_is_off():
    indices = np.array([[10, 200], [10, 200], [10, 200]])
    costs = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    path, _cost = bottom_dp_path(indices, costs, smoothing_weight=0.0)

    np.testing.assert_array_equal(path, [10, 200, 10])


def test_path_search_rides_over_a_single_outlier_when_smoothing_is_on():
    indices = np.array([[10, 200], [10, 200], [10, 200]])
    # The middle ping's evidence favours the far candidate, but only just.
    costs = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    path, _cost = bottom_dp_path(indices, costs, smoothing_weight=50.0)

    np.testing.assert_array_equal(path, [10, 10, 10])


def test_detector_finds_a_flat_bottom_on_both_channels():
    source = _flat(altitude=70)
    preprocessor = _preprocessor(source)

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=2.0)

    # Starboard counts from nadir; port counts from the outer range.
    assert abs(int(np.median(preprocessor.starboard_bottom_dist)) - 70) <= 3
    assert abs(int(np.median(preprocessor.portside_bottom_dist)) - (PING_LEN - 70)) <= 3


def test_detector_follows_a_sloping_bottom():
    altitudes = np.linspace(50, 150, PING_COUNT).astype(int)
    preprocessor = _preprocessor(SyntheticSidescanFile(altitudes))

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=2.0)

    error = preprocessor.starboard_bottom_dist - altitudes
    assert np.abs(np.median(error)) <= 3
    assert np.percentile(np.abs(error), 90) <= 6


def test_smoothing_suppresses_a_water_column_false_target():
    altitudes = np.full(PING_COUNT, 120)
    source = SyntheticSidescanFile(altitudes)
    # A bright blob well inside the water column on a handful of pings.
    source.add_water_column_blob(pings=range(58, 63), sample=40)
    preprocessor = _preprocessor(source)

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.0)
    unsmoothed = preprocessor.starboard_bottom_dist[58:63].copy()

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=60.0)
    smoothed = preprocessor.starboard_bottom_dist[58:63]

    # Without smoothing the blob captures the line; with it the line holds.
    assert np.abs(unsmoothed - 120).max() > 40
    assert np.abs(smoothed - 120).max() <= 6


def test_blanking_lifts_the_line_past_a_near_nadir_false_target():
    altitudes = np.full(PING_COUNT, 120)
    source = SyntheticSidescanFile(altitudes)
    source.add_water_column_blob(pings=range(PING_COUNT), sample=30)
    preprocessor = _preprocessor(source)

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=1.0)
    assert int(np.median(preprocessor.starboard_bottom_dist)) < 60

    # 6 m at 10 cm per sample clears the blob but not the real bottom.
    preprocessor.detect_bottom_line(blanking_m=6.0, smoothing=1.0)
    assert abs(int(np.median(preprocessor.starboard_bottom_dist)) - 120) <= 6


def test_detector_never_collapses_onto_a_constant():
    """Regression test for the threshold detector's end-of-ping collapse.

    That detector seeded the whole track from a constant chosen by
    ``threshold < 0.5``, so crossing 0.5 slammed the line to one end of the
    ping. The path search has no such seed; sweep hard and confirm the line
    stays on the bottom.
    """

    altitudes = np.full(PING_COUNT, 90)
    preprocessor = _preprocessor(SyntheticSidescanFile(altitudes))

    for smoothing in (0.0, 0.1, 0.49, 0.5, 0.51, 1.0, 5.0, 25.0, 100.0, 1000.0):
        preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=smoothing)
        starboard = preprocessor.starboard_bottom_dist
        port = preprocessor.portside_bottom_dist

        assert abs(int(np.median(starboard)) - 90) <= 6, smoothing
        assert abs(int(np.median(port)) - (PING_LEN - 90)) <= 6, smoothing
        # Neither channel may sit against an end of the ping.
        assert starboard.min() > 3 and starboard.max() < PING_LEN - 3, smoothing
        assert port.min() > 3 and port.max() < PING_LEN - 3, smoothing


def test_every_strategy_is_distinct_and_mirrors_correctly():
    """The Qt panel offered four strategies but only ever ran two."""

    altitudes = np.full(PING_COUNT, 100)
    source = SyntheticSidescanFile(altitudes)
    # Corrupt starboard only, so a port-only strategy must beat a starboard-only one.
    source.data[1, :, 30:45] = 255
    preprocessor = _preprocessor(source)
    choices = preprocessor.bottom_strategy_choices

    results = {}
    for choice in choices:
        preprocessor.detect_bottom_line(
            blanking_m=0.0, smoothing=1.0, bottom_strategy_choice=choice
        )
        results[choice] = (
            preprocessor.portside_bottom_dist.copy(),
            preprocessor.starboard_bottom_dist.copy(),
        )

    port_only = results[choices[2]]
    star_only = results[choices[3]]
    # "Only use portside" mirrors the clean port answer onto starboard.
    assert abs(int(np.median(port_only[1])) - 100) <= 6
    # "Only use starboard" inherits the corrupted side's mistake.
    assert int(np.median(star_only[1])) < 60
    # Which makes the two strategies genuinely different.
    assert not np.array_equal(port_only[1], star_only[1])


def test_data_cost_flags_pings_chosen_on_continuity():
    altitudes = np.full(PING_COUNT, 120)
    source = SyntheticSidescanFile(altitudes)
    source.add_water_column_blob(pings=range(58, 63), sample=40)
    preprocessor = _preprocessor(source)

    cost = preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=60.0)

    assert cost.shape == (2, PING_COUNT)
    # The pings whose own evidence was overruled cost more than quiet ones.
    assert cost[1, 58:63].mean() > cost[1, :40].mean()


def test_blanking_is_measured_in_meters_not_samples():
    """Two files at different ranges must blank the same physical distance."""

    coarse = _flat(altitude=100)
    fine = _flat(altitude=100)
    fine.slant_range = np.full((2, PING_COUNT), RANGE_M / 2)  # 5 cm per sample

    coarse_preprocessor = _preprocessor(coarse)
    fine_preprocessor = _preprocessor(fine)
    coarse_preprocessor.detect_bottom_line(blanking_m=5.0, smoothing=1.0)
    fine_preprocessor.detect_bottom_line(blanking_m=5.0, smoothing=1.0)

    # 5 m is 50 samples on the coarse file and 100 on the fine one.
    assert coarse_preprocessor.starboard_bottom_dist.min() >= 50
    assert fine_preprocessor.starboard_bottom_dist.min() >= 100


@pytest.mark.parametrize("altitude", [40, 70, 120, 180])
def test_detector_is_accurate_across_depths(altitude):
    preprocessor = _preprocessor(_flat(altitude=altitude))

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=2.0)

    assert abs(int(np.median(preprocessor.starboard_bottom_dist)) - altitude) <= 4
