from types import SimpleNamespace

import numpy as np

from sidescantools import xtf_wrapper
from sidescantools.xtf_wrapper import XTFWrapper


def _ping(samples, *, num_samples=3, slant_range=12.0):
    channel_header = SimpleNamespace(
        SecondsPerPing=0.01,
        NumSamples=num_samples,
        SlantRange=slant_range,
    )
    return SimpleNamespace(
        data=samples,
        ping_chan_headers=[channel_header],
        SoundVelocity=750.0,
    )


def test_single_ping_xtf_uses_the_first_available_header(monkeypatch, tmp_path):
    packet = _ping([np.array([1, 2, 3])])
    header = SimpleNamespace(NumberOfSonarChannels=1)
    packets = {xtf_wrapper.pyxtf.XTFHeaderType.sonar: [packet]}
    monkeypatch.setattr(
        xtf_wrapper.pyxtf,
        "xtf_read",
        lambda _path, **_kwargs: (header, packets),
    )
    monkeypatch.setattr(
        xtf_wrapper.pyxtf,
        "concatenate_channel",
        lambda *_args, **_kwargs: np.array([[1, 2, 3]]),
    )

    wrapped = XTFWrapper(tmp_path / "single.xtf")

    assert wrapped.num_sample_per_ping == 3
    np.testing.assert_array_equal(wrapped.sonar_data[0], [[1, 2, 3]])
    assert wrapped.x_axis_m[-1] == 12.0


def test_missing_first_ping_is_backfilled_from_first_valid_ping(monkeypatch, tmp_path):
    packets = {
        xtf_wrapper.pyxtf.XTFHeaderType.sonar: [
            _ping([]),
            _ping([np.array([4, 5, 6])]),
        ]
    }
    header = SimpleNamespace(NumberOfSonarChannels=1)
    monkeypatch.setattr(
        xtf_wrapper.pyxtf,
        "xtf_read",
        lambda _path, **_kwargs: (header, packets),
    )

    def fail_concatenation(*_args, **_kwargs):
        raise ValueError("missing channel")

    monkeypatch.setattr(xtf_wrapper.pyxtf, "concatenate_channel", fail_concatenation)

    wrapped = XTFWrapper(tmp_path / "missing.xtf")

    np.testing.assert_array_equal(wrapped.sonar_data[0], [[4, 5, 6], [4, 5, 6]])


def test_variable_ping_width_uses_assembled_width_and_largest_range(
    monkeypatch, tmp_path
):
    packets = {
        xtf_wrapper.pyxtf.XTFHeaderType.sonar: [
            _ping([np.array([1, 2, 3])], num_samples=3, slant_range=20.0),
            _ping([np.array([1, 2, 3, 4, 5])], num_samples=5, slant_range=30.0),
        ]
    }
    header = SimpleNamespace(NumberOfSonarChannels=1)
    monkeypatch.setattr(
        xtf_wrapper.pyxtf,
        "xtf_read",
        lambda _path, **_kwargs: (header, packets),
    )
    monkeypatch.setattr(
        xtf_wrapper.pyxtf,
        "concatenate_channel",
        lambda *_args, **_kwargs: np.array(
            [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]]
        ),
    )

    wrapped = XTFWrapper(tmp_path / "variable.xtf")

    assert wrapped.num_sample_per_ping == 5
    assert wrapped.x_axis_m.shape == (5,)
    assert wrapped.x_axis_m[-1] == 30.0
