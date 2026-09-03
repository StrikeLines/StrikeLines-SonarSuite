from types import SimpleNamespace

import numpy as np
import pytest

from sidescantools.layback import (
    TowDataSummary,
    resolve_geometry_layback,
    summarize_tow_data,
)
from sidescantools.sidescan_file import jsf_tow_data, xtf_tow_data
from sidescantools.swath_geometry import GeometrySettings


def test_xtf_tow_data_combines_cable_out_hundredths():
    message = SimpleNamespace(
        Layback=17.25,
        CableOut=42,
        CableOutHundredths=35,
    )

    layback_m, cable_out_m = xtf_tow_data(message)

    assert layback_m == 17.25
    assert cable_out_m == 42.35


def test_jsf_tow_data_converts_valid_cable_counter_from_decimeters():
    valid = SimpleNamespace(layback=11.5, cable_out=432, validity=1 << 11)
    invalid = SimpleNamespace(layback=11.5, cable_out=432, validity=0)

    assert jsf_tow_data(valid) == (11.5, 43.2)
    assert jsf_tow_data(invalid)[0] == 11.5
    assert np.isnan(jsf_tow_data(invalid)[1])


def test_tow_data_summary_uses_median_of_positive_finite_values():
    sidescan_file = SimpleNamespace(
        layback_m=np.array([0.0, np.nan, 10.0, 20.0, 30.0]),
        cable_out_m=np.array([np.inf, 40.0, 60.0]),
    )

    summary = summarize_tow_data(sidescan_file)

    assert summary.recorded_layback_m == 20.0
    assert summary.recorded_cable_out_m == 50.0


def test_layback_resolution_precedence_and_cable_fallback():
    base = GeometrySettings(60, cable_out_m=10)
    recorded = TowDataSummary(recorded_layback_m=25, recorded_cable_out_m=80)

    manual, manual_source = resolve_geometry_layback(
        base, recorded, manual_layback_m=12.5
    )
    direct, direct_source = resolve_geometry_layback(base, recorded)
    cable, cable_source = resolve_geometry_layback(
        base, TowDataSummary(recorded_layback_m=None, recorded_cable_out_m=80)
    )

    assert manual.effective_layback_m == 12.5
    assert manual.cable_out_m == 80
    assert "Manual" in manual_source
    assert direct.effective_layback_m == 25
    assert "Recorded layback" in direct_source
    assert cable.effective_layback_m == pytest.approx(np.sin(np.pi / 4) * 80)
    assert "cable out" in cable_source


def test_invalid_manual_layback_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_geometry_layback(
            GeometrySettings(60),
            TowDataSummary(None, None),
            manual_layback_m=-0.1,
        )
