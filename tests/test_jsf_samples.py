import numpy as np
import pytest

from sidescantools.jsf import JSFSonarDataMessage


def _message(*, samples, data_format, weighting=0):
    message = JSFSonarDataMessage.__new__(JSFSonarDataMessage)
    message.samples = samples
    message.data_format = data_format
    message.weighting = weighting
    return message


def test_envelope_samples_decode_directly_to_weighted_float32():
    message = _message(samples=4, data_format=0, weighting=2)
    encoded = np.array([-8, 0, 12, 20], dtype="<i2").tobytes()

    message.load_data(encoded)

    assert message.data.dtype == np.float32
    np.testing.assert_array_equal(message.data, [-2.0, 0.0, 3.0, 5.0])


@pytest.mark.parametrize(
    ("data_format", "values"),
    [
        (1, [-3, 4, -5, 6]),
        (2, [-3, 4]),
        (9, [-3, 4, -5, 6]),
    ],
)
def test_non_envelope_samples_preserve_legacy_flat_int16_values(
    data_format, values
):
    message = _message(samples=2, data_format=data_format)

    message.load_data(np.array(values, dtype="<i2").tobytes())

    assert message.data.dtype == np.int16
    np.testing.assert_array_equal(message.data, values)


def test_unknown_jsf_sample_format_is_rejected():
    message = _message(samples=1, data_format=99)

    with pytest.raises(NotImplementedError, match="format 99"):
        message.load_data(b"\x00\x00")
