import numpy as np
import pytest

from sidescantools.destripe import destripe_waterfall


def striped_waterfall(ping_count=151, channel_width=64):
    across_track = np.linspace(0.05, 0.4, channel_width)
    roll_gain = np.tile((0.7, 1.0, 1.3), ping_count // 3 + 1)[:ping_count]
    port = np.tile(across_track[::-1], (ping_count, 1)) * roll_gain[:, None]
    starboard = np.tile(across_track, (ping_count, 1)) * (
        2.0 - roll_gain
    )[:, None]
    return np.hstack((port, starboard))


def test_destripe_suppresses_ping_wise_roll_banding():
    source = striped_waterfall()

    result = destripe_waterfall(source)

    before = np.median(source, axis=1)
    after = np.median(result, axis=1)
    assert np.std(after) < np.std(before) * 0.05


def test_destripe_preserves_each_pings_across_track_profile():
    source = striped_waterfall()

    result = destripe_waterfall(source)

    # The correction is one scalar per ping and side, so texture and the TVG
    # slope across a row remain unchanged unless clipping is reached.
    np.testing.assert_allclose(
        result[70, :64] / result[70, 10],
        source[70, :64] / source[70, 10],
        rtol=1e-12,
        atol=1e-12,
    )


def test_destripe_is_finite_and_does_not_invent_returns_in_zero_water_column():
    source = striped_waterfall()
    source[:, 55:73] = 0.0
    source[10, 3] = np.nan

    result = destripe_waterfall(source)

    assert np.all(np.isfinite(result))
    assert np.all(result[:, 55:73] == 0.0)


def test_destripe_rejects_invalid_waterfall_shape():
    with pytest.raises(ValueError, match="two equal-width channels"):
        destripe_waterfall(np.ones((10, 5)))
