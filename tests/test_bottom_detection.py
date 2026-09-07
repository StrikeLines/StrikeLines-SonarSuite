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
    bottom_dp_path,
    bottom_sample_costs,
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


def test_blanked_samples_cost_more_than_any_real_one():
    source = _flat()
    scores = bottom_step_scores(source.data[1].astype(float))

    costs = bottom_sample_costs(scores, blank_samples=90)

    window = max(3, int(round(0.01 * PING_LEN)))
    assert (costs[:, :90] > 1.0).all()
    # The last `window` samples carry no score either, so compare the interior.
    assert costs[:, 90 : PING_LEN - window].max() <= 1.0


def test_costs_stay_finite_when_blanking_covers_the_whole_ping():
    source = _flat()
    scores = bottom_step_scores(source.data[1].astype(float))

    costs = bottom_sample_costs(scores, blank_samples=PING_LEN + 50)

    # Nowhere legal left to look, but the search must still be solvable.
    assert costs.shape == (PING_COUNT, PING_LEN)
    assert np.isfinite(costs).all()


def _two_well_costs():
    """Three pings, cheap at sample 0 except the middle one, cheap at sample 9."""

    costs = np.ones((3, 10))
    costs[0, 0] = costs[2, 0] = 0.0
    costs[1, 9] = 0.0
    return costs


def test_path_search_follows_the_data_when_smoothing_is_off():
    path, _cost = bottom_dp_path(_two_well_costs(), smoothing_weight=0.0)

    np.testing.assert_array_equal(path, [0, 9, 0])


def test_path_search_rides_over_a_single_outlier_when_smoothing_is_on():
    path, _cost = bottom_dp_path(_two_well_costs(), smoothing_weight=1.0)

    np.testing.assert_array_equal(path, [0, 0, 0])


def test_path_search_matches_an_exhaustive_search():
    """The linear-time min-convolution must agree with comparing every pair."""

    generator = np.random.default_rng(11)
    for _ in range(60):
        pings = int(generator.integers(2, 12))
        samples = int(generator.integers(2, 25))
        costs = generator.random((pings, samples)) * generator.choice([1.0, 10.0])
        weight = float(generator.choice([0.0, 0.01, 0.2, 1.0, 5.0]))

        fast, _cost = bottom_dp_path(costs, weight)

        index = np.arange(samples)
        total = costs[0].copy()
        back = np.zeros((pings, samples), dtype=int)
        for ping in range(1, pings):
            step = total[:, None] + weight * np.abs(index[None, :] - index[:, None])
            previous = np.argmin(step, axis=0)
            total = step[previous, index] + costs[ping]
            back[ping] = previous
        slow = np.zeros(pings, dtype=int)
        cursor = int(np.argmin(total))
        for ping in range(pings - 1, -1, -1):
            slow[ping] = cursor
            cursor = int(back[ping, cursor])

        def path_cost(path):
            return costs[np.arange(pings), path].sum() + weight * np.abs(
                np.diff(path)
            ).sum()

        assert path_cost(fast) == pytest.approx(path_cost(slow))


def test_detector_finds_a_flat_bottom_on_both_channels():
    source = _flat(altitude=70)
    preprocessor = _preprocessor(source)

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.10)

    # Starboard counts from nadir; port counts from the outer range.
    assert abs(int(np.median(preprocessor.starboard_bottom_dist)) - 70) <= 3
    assert abs(int(np.median(preprocessor.portside_bottom_dist)) - (PING_LEN - 70)) <= 3


def test_detector_follows_a_sloping_bottom():
    altitudes = np.linspace(50, 150, PING_COUNT).astype(int)
    preprocessor = _preprocessor(SyntheticSidescanFile(altitudes))

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.10)

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

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.5)
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

    for smoothing in (0.0, 0.01, 0.1, 0.49, 0.5, 0.51, 1.0, 5.0, 100.0, 1000.0):
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

    cost = preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.5)

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

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=0.10)

    assert abs(int(np.median(preprocessor.starboard_bottom_dist)) - altitude) <= 4


def test_maximum_smoothing_still_follows_a_real_slope():
    """The top of the slider must not iron out genuine depth change.

    A weight of 1 makes one sample of movement cost the whole data-cost range,
    which is where the line stops tracking the seafloor. The slider stops
    there, so the maximum has to remain usable.
    """

    altitudes = np.linspace(50, 150, PING_COUNT).astype(int)
    preprocessor = _preprocessor(SyntheticSidescanFile(altitudes))

    preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=1.0)

    error = preprocessor.starboard_bottom_dist - altitudes
    assert np.percentile(np.abs(error), 90) <= 6


def test_smoothing_reduces_jitter_monotonically():
    """The control has to have a visible, ordered effect -- it previously did not."""

    generator = np.random.default_rng(3)
    altitudes = 100 + generator.integers(-4, 5, PING_COUNT)
    preprocessor = _preprocessor(SyntheticSidescanFile(altitudes, noise=6.0))

    jitter = []
    for smoothing in (0.0, 0.05, 0.2, 1.0):
        preprocessor.detect_bottom_line(blanking_m=0.0, smoothing=smoothing)
        line = preprocessor.starboard_bottom_dist
        jitter.append(float(np.mean(np.abs(np.diff(line)))))

    assert jitter == sorted(jitter, reverse=True)
    assert jitter[-1] < jitter[0] / 3
