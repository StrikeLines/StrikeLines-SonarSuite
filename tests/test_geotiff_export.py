from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

from sidescantools.contact_model import Channel
from sidescantools.gain_settings import SonarGainSettings
from sidescantools.gain_settings import save_gain_settings
from sidescantools.geotiff_export import (
    _configure_pyproj_data,
    export_prepared_waterfall,
    geotiff_output_path,
    prepare_sonar_export,
    render_rgb_with_gain_settings,
)
from sidescantools.qt_contact_picker_ui import WaterfallGainModel
from sidescantools.swath_geometry import GeometrySettings, SwathGeometry


def gain_settings(
    width,
    *,
    auto_tvg=False,
    slant_range_correction=False,
    layback_override_m=None,
):
    return SonarGainSettings(
        source_file="line.jsf",
        overall_gain_db=-7.0,
        tvg_spreading_db_per_decade=18.0,
        tvg_absorption_db_per_m=0.12,
        auto_tvg_brightness_target_percent=30,
        auto_tvg_active=auto_tvg,
        auto_tvg_gain_db=(
            tuple(np.linspace(-2.0, 3.0, width)) if auto_tvg else ()
        ),
        speed_correction_px_per_ping=3.0,
        processing_mode="raw",
        egn_table_path=None,
        slant_range_correction_active=slant_range_correction,
        layback_override_m=layback_override_m,
    )


def geometry(channel):
    ping_count = 4
    nadir_lon = np.full(ping_count, -70.0)
    nadir_lat = 40.0 + np.arange(ping_count) * 0.0001
    direction = -1.0 if channel == Channel.PORT else 1.0
    return SwathGeometry(
        channel=channel,
        sample_count=4,
        valid_ping_mask=np.ones(ping_count, dtype=bool),
        nadir_lon=nadir_lon,
        nadir_lat=nadir_lat,
        outer_lon=nadir_lon + direction * 0.0004,
        outer_lat=nadir_lat,
        slant_range_m=np.full(ping_count, 40.0),
        ground_range_m=np.full(ping_count, 30.0),
        geometry_settings=GeometrySettings(60),
    )


def test_output_path_is_beside_source_and_uses_same_basename(tmp_path):
    assert geotiff_output_path(tmp_path / "survey-line.jsf") == (
        tmp_path / "survey-line.tif"
    )
    assert geotiff_output_path(tmp_path / "survey-line.xtf") == (
        tmp_path / "survey-line.tif"
    )


def test_proj_database_prefers_active_python_environment_over_quoted_qgis_path(
    monkeypatch, tmp_path
):
    from sidescantools import geotiff_export

    environment_data = tmp_path / "python-environment-proj"
    environment_data.mkdir()
    (environment_data / "proj.db").touch()
    external_qgis_data = tmp_path / "external-qgis-proj"
    external_qgis_data.mkdir()
    (external_qgis_data / "proj.db").touch()
    selected = []
    monkeypatch.setenv("PROJ_LIB", f'"{external_qgis_data}"')
    monkeypatch.setattr(
        geotiff_export.pyproj.datadir,
        "get_data_dir",
        lambda: str(environment_data),
    )
    monkeypatch.setattr(
        geotiff_export.pyproj.datadir,
        "set_data_dir",
        lambda path: selected.append(Path(path)),
    )

    result = _configure_pyproj_data()

    assert result == environment_data.resolve()
    assert selected == [environment_data.resolve()]


def test_saved_gain_render_exactly_matches_qt_waterfall_model():
    source = np.linspace(0.01, 0.6, 5 * 8).reshape(5, 8)
    settings = gain_settings(source.shape[1], auto_tvg=True)
    model = WaterfallGainModel(source, slant_range_m=42.0)
    model.overall_gain_db = settings.overall_gain_db
    model.tvg_spreading_db_per_decade = settings.tvg_spreading_db_per_decade
    model.tvg_absorption_db_per_m = settings.tvg_absorption_db_per_m
    model.restore_auto_tvg_gain(settings.auto_tvg_gain_db)

    exported_rgb = render_rgb_with_gain_settings(
        source, slant_range_m=42.0, settings=settings
    )

    np.testing.assert_array_equal(exported_rgb, model.render_rgb())


def test_prepare_export_applies_saved_slant_range_setting(monkeypatch, tmp_path):
    from sidescantools import geotiff_export

    source = tmp_path / "line.jsf"
    source.touch()
    save_gain_settings(
        source,
        gain_settings(8, slant_range_correction=True),
    )
    waterfall = np.linspace(0.05, 0.8, 4 * 8).reshape(1, 4, 8)
    sidescan_file = SimpleNamespace(
        num_ping=4,
        slant_range=np.full((2, 4), 40.0),
    )
    preprocessor = SimpleNamespace(
        ping_len=4,
        napari_fullmat=waterfall,
        init_napari_bottom_detect=lambda *args, **kwargs: None,
    )
    requests = []

    class CapturingProcessor:
        def __init__(self, current_preprocessor, raw):
            assert current_preprocessor is preprocessor
            self.raw = raw

        def process(self, request, progress=None):
            requests.append(request)
            return SimpleNamespace(
                display_data=self.raw,
                pipeline_description="raw|slant-range-corrected",
            )

    class FakeGeoreferencer:
        def __init__(self, *args, channel, **kwargs):
            self.channel = channel

        def prepare_swath_geometry(self):
            return geometry(self.channel)

    monkeypatch.setattr(geotiff_export, "_configure_pyproj_data", lambda: None)
    monkeypatch.setattr(geotiff_export, "SidescanFile", lambda path: sidescan_file)
    monkeypatch.setattr(
        geotiff_export, "SidescanPreprocessor", lambda **kwargs: preprocessor
    )
    monkeypatch.setattr(
        geotiff_export,
        "compute_depth_info",
        lambda current_file, factor: np.zeros(4),
    )
    monkeypatch.setattr(geotiff_export, "BuiltInGainProcessor", CapturingProcessor)
    monkeypatch.setattr(geotiff_export, "Georeferencer", FakeGeoreferencer)

    prepared = prepare_sonar_export(
        source,
        chunk_size=4,
        default_threshold=0.5,
        downsampling_factor=1,
        active_db=False,
        active_hist_equal=False,
        geometry_settings=GeometrySettings(60),
    )

    assert len(requests) == 1
    assert requests[0].slant_range_correction is True
    assert "slant-range-corrected" in prepared.pipeline_description


def test_prepare_export_applies_saved_manual_layback(monkeypatch, tmp_path):
    from sidescantools import geotiff_export

    source = tmp_path / "line.jsf"
    source.touch()
    save_gain_settings(source, gain_settings(8, layback_override_m=73.5))
    sidescan_file = SimpleNamespace(
        num_ping=4,
        slant_range=np.full((2, 4), 40.0),
        layback_m=np.full(4, 12.0),
        cable_out_m=np.full(4, 50.0),
    )
    preprocessor = SimpleNamespace(
        ping_len=4,
        napari_fullmat=np.linspace(0.05, 0.8, 4 * 8).reshape(1, 4, 8),
        init_napari_bottom_detect=lambda *args, **kwargs: None,
    )
    captured_settings = []

    class FakeGeoreferencer:
        def __init__(self, *args, channel, geometry_settings, **kwargs):
            self.channel = channel
            captured_settings.append(geometry_settings)

        def prepare_swath_geometry(self):
            return geometry(self.channel)

    monkeypatch.setattr(geotiff_export, "_configure_pyproj_data", lambda: None)
    monkeypatch.setattr(geotiff_export, "SidescanFile", lambda path: sidescan_file)
    monkeypatch.setattr(
        geotiff_export, "SidescanPreprocessor", lambda **kwargs: preprocessor
    )
    monkeypatch.setattr(
        geotiff_export,
        "compute_depth_info",
        lambda current_file, factor: np.zeros(4),
    )
    monkeypatch.setattr(geotiff_export, "Georeferencer", FakeGeoreferencer)

    prepare_sonar_export(
        source,
        chunk_size=4,
        default_threshold=0.5,
        downsampling_factor=1,
        active_db=False,
        active_hist_equal=False,
        geometry_settings=GeometrySettings(60),
    )

    assert len(captured_settings) == 2
    assert all(item.effective_layback_m == 73.5 for item in captured_settings)


@pytest.mark.parametrize("epsg", [4326, 3857])
def test_export_is_a_georeferenced_rgba_geotiff(epsg, tmp_path):
    source = tmp_path / "line.jsf"
    source.touch()
    gray = np.array(
        [
            [10, 20, 30, 40, 50, 60, 70, 80],
            [15, 25, 35, 45, 55, 65, 75, 85],
            [20, 30, 40, 50, 60, 70, 80, 90],
            [25, 35, 45, 55, 65, 75, 85, 95],
        ],
        dtype=np.uint8,
    )
    rgb = np.stack(
        (gray, (gray.astype(float) * 0.78).astype(np.uint8), gray // 3), axis=2
    )

    result = export_prepared_waterfall(
        source,
        rgb,
        {0: geometry(Channel.PORT), 1: geometry(Channel.STARBOARD)},
        epsg=epsg,
        pipeline_description="test-pipeline",
    )

    assert result.destination == tmp_path / "line.tif"
    assert result.destination.is_file()
    assert result.valid_pixel_count > 0
    with rasterio.open(result.destination) as dataset:
        assert dataset.driver == "GTiff"
        assert dataset.crs.to_epsg() == epsg
        assert dataset.count == 4
        assert dataset.colorinterp[-1].name == "alpha"
        assert dataset.tags()["SOURCE_FILE"] == "line.jsf"
        assert dataset.tags()["SIDESCANTOOLS_DISPLAY_PIPELINE"] == "test-pipeline"
        red, green, blue, alpha = dataset.read()
        assert np.any(alpha == 255)
        valid = alpha == 255
        np.testing.assert_array_equal(green[valid], (red[valid] * 0.78).astype(np.uint8))
        np.testing.assert_array_equal(blue[valid], red[valid] // 3)
        assert dataset.transform.a > 0
        assert dataset.transform.e < 0


def test_existing_geotiff_requires_explicit_overwrite(tmp_path):
    source = tmp_path / "line.jsf"
    source.touch()
    destination = tmp_path / "line.tif"
    destination.write_bytes(b"existing")
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)

    with pytest.raises(FileExistsError, match="line.tif"):
        export_prepared_waterfall(
            source,
            rgb,
            {0: geometry(Channel.PORT), 1: geometry(Channel.STARBOARD)},
            epsg=4326,
            pipeline_description="test",
        )


def test_batch_worker_exports_every_file_with_its_own_loader_path(monkeypatch, tmp_path):
    from sidescantools import qt_contact_picker_ui

    sources = [tmp_path / "b.jsf", tmp_path / "a.xtf"]
    calls = []

    def fake_export(source, **kwargs):
        calls.append((Path(source), kwargs["epsg"], kwargs["downsampling_factor"]))
        return SimpleNamespace(used_default_settings=False)

    monkeypatch.setattr(qt_contact_picker_ui, "export_sonar_file", fake_export)
    worker = qt_contact_picker_ui.GeoTiffExportWorker(
        sources,
        epsg=3857,
        loader_settings=SimpleNamespace(
            chunk_size=256,
            default_threshold=0.7,
            downsampling_factor=32,
            active_dB=False,
            active_hist_equal=False,
            geometry_settings=GeometrySettings(60),
        ),
        overwrite=False,
    )
    completed = []
    worker.signals.finished.connect(lambda results, failures: completed.append((results, failures)))

    worker.run()

    assert [call[0] for call in calls] == [path.resolve() for path in sources]
    assert all(call[1:] == (3857, 32) for call in calls)
    assert len(completed[0][0]) == 2
    assert completed[0][1] == []
