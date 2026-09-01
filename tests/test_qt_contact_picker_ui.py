import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QEvent, QObject, QPoint, QRunnable, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsView,
    QLabel,
    QPushButton,
    QWidget,
)

from sidescantools import qt_contact_picker_ui
from sidescantools.contact_store import ContactStore
from sidescantools.interaction_mode import InteractionModeController
from sidescantools.qt_contact_picker_ui import (
    EGNTableBuildCoordinator,
    EGNTableBuilderDialog,
    QtContactPickerWindow,
    SonarFileContext,
    SonarLoaderSettings,
    WaterfallGainModel,
    WaterfallView,
    logical_bottom_overlay,
    logical_waterfall,
    sonar_files_in_directory,
    waterfall_rgb,
)
from sidescantools.sidescan_preproc import SidescanPreprocessor
from sidescantools.swath_geometry import GeometrySettings


def test_waterfall_rgb_is_uint8_and_preserves_shape():
    source = np.array([[0.0, 1.0, 2.0], [np.nan, 3.0, 4.0]])

    result = waterfall_rgb(source)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8
    assert tuple(result[1, 0]) == (0, 0, 0)


def test_logical_waterfall_stitches_chunks_and_removes_padding():
    class Preprocessor:
        ping_len = 2
        napari_fullmat = np.arange(3 * 2 * 4).reshape(3, 2, 4)

    result = logical_waterfall(Preprocessor(), source_ping_count=5)

    assert result.shape == (5, 4)
    np.testing.assert_array_equal(result, np.arange(20).reshape(5, 4))


def test_logical_bottom_overlay_uses_the_same_reshape_as_logical_waterfall():
    class Preprocessor:
        ping_len = 2
        bottom_map = np.arange(3 * 2 * 4).reshape(3, 2, 4)

    result = logical_bottom_overlay(Preprocessor(), source_ping_count=5)

    assert result.shape == (5, 4)
    np.testing.assert_array_equal(result, np.arange(20).reshape(5, 4))


def test_requested_gain_and_scale_defaults_are_applied(qtbot):
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=10.0)
    view = WaterfallView()
    qtbot.addWidget(view)

    assert model.overall_gain_db == -5.0
    assert model.tvg_spreading_db_per_decade == 5.0
    assert model.tvg_absorption_db_per_m == 0.08
    assert view._along_track_scale == 3.0


def test_tvg_spreading_profile_is_symmetric_and_strongest_at_outer_ranges():
    # slant_range_m=10 -> floor_m = 10 * 0.02 = 0.2. With only the spreading
    # term active at 20 dB/decade, gain_profile() reduces to a clean
    # closed form: gain = max(range_m, floor_m) / floor_m.
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=10.0)
    model.overall_gain_db = 0.0
    model.tvg_spreading_db_per_decade = 20.0
    model.tvg_absorption_db_per_m = 0.0

    profile = model.gain_profile()

    np.testing.assert_allclose(profile, profile[::-1])
    np.testing.assert_allclose(profile[2], 1.0)  # nadir, clamped to the floor
    np.testing.assert_allclose(profile[3], 1.0)
    np.testing.assert_allclose(profile[1], 5.0 / 0.2)  # sample_fraction 0.5
    np.testing.assert_allclose(profile[0], 10.0 / 0.2)  # outer edge, full range


def test_tvg_absorption_grows_linearly_with_range():
    # Isolate the absorption term (spreading=0): gain_db = absorption * range_m,
    # linear and un-floored (no log singularity to guard against).
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=10.0)
    model.overall_gain_db = 0.0
    model.tvg_spreading_db_per_decade = 0.0
    model.tvg_absorption_db_per_m = 1.0

    profile = model.gain_profile()

    np.testing.assert_allclose(profile[2], 1.0)  # nadir: 0 dB boost
    np.testing.assert_allclose(profile[0], 10 ** (10.0 / 20))  # 1 dB/m * 10 m
    np.testing.assert_allclose(profile, profile[::-1])


def test_uncalibrated_range_falls_back_to_a_unitless_reference():
    # No slant_range_m given: the curve shape must still work sensibly
    # rather than raising or dividing by zero.
    model = WaterfallGainModel(np.full((2, 6), 0.1))
    model.tvg_spreading_db_per_decade = 20.0

    profile = model.gain_profile()

    assert np.all(np.isfinite(profile))
    assert profile[0] > profile[2]  # still stronger at the outer edge


def test_set_source_without_slant_range_m_preserves_existing_calibration():
    # Simulates a processing-mode change on the SAME file: must not reset
    # calibration back to the uncalibrated fallback.
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=200.0)

    model.set_source(np.full((2, 6), 0.2), base_pipeline="reprocessed")

    assert model._reference_range_m == 200.0


def test_set_source_with_slant_range_m_updates_calibration_for_a_new_file():
    # Simulates navigating to a different file with a different range
    # setting: calibration must update, not stay pinned to the old file's.
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=200.0)

    model.set_source(np.full((2, 6), 0.2), base_pipeline="raw", slant_range_m=75.0)

    assert model._reference_range_m == 75.0


def test_overall_gain_changes_display_without_mutating_source():
    source = np.full((2, 4), 0.25)
    model = WaterfallGainModel(source)
    model.overall_gain_db = 6.020599913
    model.tvg_spreading_db_per_decade = 0.0
    model.tvg_absorption_db_per_m = 0.0

    corrected = model.corrected()

    np.testing.assert_allclose(corrected, 0.5)
    np.testing.assert_allclose(source, 0.25)
    assert model.corrected_value(0, 0) == corrected[0, 0]


def test_tvg_changes_outer_pixels_more_than_nadir_pixels():
    source = np.full((3, 10), 0.05)
    model = WaterfallGainModel(source, slant_range_m=10.0)
    before = model.render_rgb().astype(float)

    model.tvg_spreading_db_per_decade = 30.0
    after = model.render_rgb().astype(float)
    delta = np.mean(np.abs(after - before), axis=(0, 2))

    assert delta[0] > delta[4]
    assert delta[-1] > delta[5]
    assert delta[0] > 0


def test_normalize_tvg_recovers_known_range_attenuation_and_flattens_brightness():
    rows = 80
    channel_width = 128
    template = WaterfallGainModel(
        np.ones((rows, 2 * channel_width)), slant_range_m=100.0
    )
    template.overall_gain_db = 0.0
    template.tvg_spreading_db_per_decade = 18.0
    template.tvg_absorption_db_per_m = 0.12
    attenuation = template.gain_profile()
    source = np.full((rows, 2 * channel_width), 0.35) / attenuation[None, :]
    model = WaterfallGainModel(source, slant_range_m=100.0)
    model.overall_gain_db = -3.0

    overall, spreading, absorption = model.normalize_tvg()
    corrected = model.corrected()

    assert overall == -10.0
    assert spreading == pytest.approx(18.0, abs=0.25)
    assert absorption == pytest.approx(0.12, abs=0.005)
    assert np.median(corrected) == pytest.approx(0.5, abs=0.02)
    assert np.ptp(np.median(corrected, axis=0)) < 0.06


def test_normalize_tvg_is_robust_to_bad_values_and_local_bright_targets():
    rows = 100
    channel_width = 96
    template = WaterfallGainModel(
        np.ones((rows, 2 * channel_width)), slant_range_m=60.0
    )
    template.overall_gain_db = 0.0
    template.tvg_spreading_db_per_decade = 12.0
    template.tvg_absorption_db_per_m = 0.08
    source = np.full((rows, 2 * channel_width), 0.25) / template.gain_profile()[None, :]
    source[0, 0] = np.nan
    source[1, -1] = np.inf
    source[10:30, 35:41] = 1.0
    original = source.copy()
    model = WaterfallGainModel(source, slant_range_m=60.0)

    overall, spreading, absorption = model.normalize_tvg()

    assert math.isfinite(overall)
    assert spreading == pytest.approx(12.0, abs=1.0)
    assert absorption == pytest.approx(0.08, abs=0.02)
    np.testing.assert_array_equal(model.source, original)


def test_normalize_tvg_raises_dark_areas_and_reduces_blown_out_areas():
    rows = 60
    channel_width = 96
    fraction = np.linspace(0.0, 1.0, channel_width)
    port_near_to_far = np.clip(
        0.08 + 0.82 * fraction + 0.06 * np.sin(4.0 * np.pi * fraction),
        0.02,
        1.0,
    )
    starboard_near_to_far = np.clip(
        0.92 - 0.78 * fraction + 0.05 * np.sin(3.0 * np.pi * fraction),
        0.02,
        1.0,
    )
    across_swath = np.concatenate(
        (port_near_to_far[::-1], starboard_near_to_far)
    )
    source = np.tile(across_swath, (rows, 1))
    model = WaterfallGainModel(source, slant_range_m=80.0)
    dark_column = int(np.argmin(across_swath))
    bright_column = int(np.argmax(across_swath))

    before = np.median(model.corrected(), axis=0)
    model.normalize_tvg()
    after = np.median(model.corrected(), axis=0)

    assert after[dark_column] > before[dark_column]
    assert after[bright_column] < before[bright_column]
    assert np.std(after) < np.std(before) * 0.2
    assert np.percentile(after, 5) > 0.43
    assert np.percentile(after, 95) < 0.57
    assert model.overall_gain_db == WaterfallGainModel.NORMALIZE_OVERALL_GAIN_DB
    assert "swath-equalized=50pct" in model.pipeline_description


def test_clear_normalization_removes_the_empirical_gain_curve():
    model = WaterfallGainModel(np.full((20, 40), 0.2), slant_range_m=30.0)
    model.normalize_tvg()

    model.clear_normalization()

    assert not model._normalization_active
    np.testing.assert_array_equal(model._normalization_gain_db, 0.0)
    assert "swath-equalized" not in model.pipeline_description


def test_normalize_tvg_rejects_a_waterfall_without_returns():
    model = WaterfallGainModel(np.zeros((10, 20)), slant_range_m=20.0)

    with pytest.raises(ValueError, match="not enough finite sonar returns"):
        model.normalize_tvg()


def test_range_at_column_reports_uncalibrated_before_any_slant_range_given():
    model = WaterfallGainModel(np.full((2, 6), 0.1))

    _, calibrated = model.range_at_column(0)

    assert calibrated is False


def test_range_at_column_matches_reference_range_once_calibrated():
    model = WaterfallGainModel(np.full((2, 6), 0.1), slant_range_m=30.0)

    range_m, calibrated = model.range_at_column(0)  # outer port edge: full range

    assert calibrated is True
    assert range_m == pytest.approx(30.0)
    assert model.range_at_column(2)[0] == pytest.approx(0.0)  # nadir


class _FakeWheelEvent:
    """Duck-typed stand-in for QWheelEvent so tests don't depend on the
    active Qt binding's (PyQt5 vs PySide6) constructor signature."""

    def __init__(self, delta_y: int):
        self._delta_y = delta_y
        self.accepted = False

    def angleDelta(self):
        return QPoint(0, self._delta_y)

    def accept(self):
        self.accepted = True


def _shown_view(qtbot, *, width=200, height=5000):
    view = WaterfallView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()
    qtbot.waitExposed(view)
    view.set_image(np.zeros((height, width, 3), dtype=np.uint8), fit=True)
    return view


def test_view_defaults_to_a_crosshair_cursor_not_the_scroll_hand(qtbot):
    # ScrollHandDrag normally shows an open/closed hand, unhelpful for
    # pixel-precise contact picking -- must be overridden to a crosshair.
    view = WaterfallView()
    qtbot.addWidget(view)

    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor


def test_waterfall_view_fits_width_and_applies_along_track_slider(qtbot):
    view = _shown_view(qtbot, width=200)

    view.set_along_track_scale(2.5)

    transform = view.transform()
    expected_sx = view.viewport().width() / 200
    assert transform.m11() == pytest.approx(expected_sx, rel=1e-3)
    assert transform.m22() == pytest.approx(2.5)


def test_resize_reapplies_width_fit_and_keeps_along_track_scale(qtbot):
    view = _shown_view(qtbot, width=200)
    view.set_along_track_scale(3.0)

    view.resize(500, 600)
    qtbot.wait(50)

    transform = view.transform()
    assert transform.m11() == pytest.approx(view.viewport().width() / 200, rel=1e-3)
    assert transform.m22() == pytest.approx(3.0)


def test_wheel_scrolls_through_the_survey_instead_of_zooming(qtbot):
    view = _shown_view(qtbot)
    before_transform = view.transform()
    before_scroll = view.verticalScrollBar().value()

    view.wheelEvent(_FakeWheelEvent(delta_y=-120))

    assert view.transform() == before_transform
    assert view.verticalScrollBar().value() != before_scroll


def test_wheel_direction_matches_scroll_convention(qtbot):
    view = _shown_view(qtbot)
    view.verticalScrollBar().setValue(view.verticalScrollBar().maximum() // 2)
    start = view.verticalScrollBar().value()

    view.wheelEvent(_FakeWheelEvent(delta_y=120))  # scroll "up" / toward the start
    up_value = view.verticalScrollBar().value()
    view.verticalScrollBar().setValue(start)

    view.wheelEvent(_FakeWheelEvent(delta_y=-120))  # scroll "down" / toward the end
    down_value = view.verticalScrollBar().value()

    assert up_value < start < down_value


def test_markers_ignore_the_independent_axis_scales(qtbot):
    view = _shown_view(qtbot, width=200)
    view.set_along_track_scale(6.0)  # deliberately different from width-fit sx

    view.set_markers([(100, 20, "Target 0001")])

    assert len(view._marker_items) == 1
    item = view._marker_items[0]
    assert bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
    assert item.pos().x() == pytest.approx(20)
    assert item.pos().y() == pytest.approx(100)


def test_set_image_preserves_scroll_position_on_refresh(qtbot):
    view = _shown_view(qtbot, width=200, height=5000)
    view.verticalScrollBar().setValue(view.verticalScrollBar().maximum() // 2)
    center_row_before = view.mapToScene(view.viewport().rect().center()).y()

    # Simulate a gain/processing refresh: same-sized image, fit=False.
    view.set_image(np.ones((5000, 200, 3), dtype=np.uint8) * 50, fit=False)

    center_row_after = view.mapToScene(view.viewport().rect().center()).y()
    assert center_row_after == pytest.approx(center_row_before, abs=2.0)


def test_hover_inside_the_image_emits_pixel_hovered_with_row_and_column(qtbot):
    view = _shown_view(qtbot, width=200, height=5000)
    viewport_position = view.viewport().rect().center()
    scene_point = view.mapToScene(viewport_position)
    expected_row = math.floor(scene_point.y())
    expected_column = math.floor(scene_point.x())

    with qtbot.waitSignal(view.pixel_hovered, timeout=1000) as blocker:
        view._emit_hover_at(viewport_position)

    assert blocker.args == [expected_row, expected_column]


def test_hover_outside_the_image_emits_hover_cleared_instead(qtbot):
    view = _shown_view(qtbot, width=200, height=5000)

    with qtbot.waitSignal(view.hover_cleared, timeout=1000):
        view._emit_hover_at(QPoint(-5, -5))


def test_mouse_leaving_the_view_emits_hover_cleared(qtbot):
    view = _shown_view(qtbot, width=200, height=5000)

    with qtbot.waitSignal(view.hover_cleared, timeout=1000):
        view.leaveEvent(QEvent(QEvent.Type.Leave))


def _rgba(color):
    return (color.red(), color.green(), color.blue(), color.alpha())


def test_set_bottom_overlay_marks_active_pixels_and_leaves_others_transparent(qtbot):
    view = _shown_view(qtbot, width=4, height=3)
    mask = np.zeros((3, 4))
    mask[1, 2] = 1

    view.set_bottom_overlay(mask)

    image = view._overlay_pixmap.toImage()
    # Compare channels rather than QColor.__eq__ -- it also compares
    # internal color-spec state that differs by construction path even
    # when every channel value already matches.
    assert _rgba(image.pixelColor(2, 1)) == _rgba(view.BOTTOM_LINE_COLOR)
    assert image.pixelColor(0, 0).alpha() == 0


def test_patch_bottom_overlay_row_touches_only_that_row(qtbot):
    view = _shown_view(qtbot, width=4, height=3)
    view.set_bottom_overlay(np.zeros((3, 4)))

    view.patch_bottom_overlay_row(1, np.array([0, 1, 0, 0]))

    image = view._overlay_pixmap.toImage()
    assert _rgba(image.pixelColor(1, 1)) == _rgba(view.BOTTOM_LINE_COLOR)
    assert image.pixelColor(1, 0).alpha() == 0  # untouched row stays clear


def test_edit_mode_press_emits_bottom_edited_at_the_pressed_pixel(qtbot):
    view = _shown_view(qtbot, width=200, height=5000)
    view.edit_mode = True
    viewport_position = view.viewport().rect().center()
    scene_point = view.mapToScene(viewport_position)
    expected_row = math.floor(scene_point.y())
    expected_column = math.floor(scene_point.x())

    with qtbot.waitSignal(view.bottom_edited, timeout=1000) as blocker:
        view._emit_edit_at(viewport_position)

    assert blocker.args == [expected_row, expected_column]


def test_drag_holds_the_new_column_across_rows_a_fast_drag_skips(qtbot):
    # Verified against the reference Napari bottom editor
    # (bottom_detection_napari_ui.py's mouse-drag callback): a fast drag
    # holds the *destination* column across every skipped row -- it does
    # not ramp from the old column to the new one.
    view = _shown_view(qtbot, width=200, height=5000)
    view.edit_mode = True
    start_position = QPoint(20, 100)
    end_position = QPoint(150, 130)
    start_row = math.floor(view.mapToScene(start_position).y())
    start_column = math.floor(view.mapToScene(start_position).x())
    end_row = math.floor(view.mapToScene(end_position).y())
    end_column = math.floor(view.mapToScene(end_position).x())
    assert end_row - start_row > 1, "test setup must actually skip rows"

    emitted = []
    view.bottom_edited.connect(lambda row, column: emitted.append((row, column)))
    view._emit_edit_at(start_position)
    view._emit_edit_at(end_position)

    expected = [(start_row, start_column)]
    expected += [(row, end_column) for row in range(start_row + 1, end_row)]
    expected.append((end_row, end_column))
    assert emitted == expected


class _FakeEGNSignals(QObject):
    finished = Signal()
    progress = Signal(float)
    error_signal = Signal(Exception)


def _fake_egn_worker_factory(fail_filenames: set):
    """Build a drop-in replacement for EGNTableProcessingWorker that
    completes instantly instead of parsing a real sonar file, so the
    coordinator's own orchestration can be tested without real JSF/XTF data.
    """

    class _FakeEGNWorker(QRunnable):
        def __init__(
            self,
            filename,
            bottom_file,
            out_path,
            chunk_size,
            nadir_angle,
            active_intern_depth,
            active_bottom_detection_downsampling,
            egn_table_parameters,
        ):
            super().__init__()
            self.filename = filename
            self.out_path = Path(out_path)
            self.signals = _FakeEGNSignals()

        def run(self):
            if self.filename in fail_filenames:
                self.signals.error_signal.emit(RuntimeError(f"boom: {self.filename}"))
            else:
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                self.out_path.write_bytes(b"fake-egn-info")
            self.signals.finished.emit()

    return _FakeEGNWorker


def test_coordinator_combines_only_successful_files_and_cleans_up(
    qtbot, tmp_path, monkeypatch
):
    files = [tmp_path / "a.jsf", tmp_path / "b.jsf", tmp_path / "c.jsf"]
    for f in files:
        f.write_bytes(b"")
    monkeypatch.setattr(
        qt_contact_picker_ui,
        "EGNTableProcessingWorker",
        _fake_egn_worker_factory({str(files[1])}),
    )
    combine_calls = []
    monkeypatch.setattr(
        qt_contact_picker_ui,
        "generate_egn_table_from_infos",
        lambda info_paths, output_path: combine_calls.append(
            (list(info_paths), output_path)
        ),
    )
    output_path = tmp_path / "out" / "egn_table.npz"

    coordinator = EGNTableBuildCoordinator(
        files,
        output_path,
        chunk_size=1000,
        nadir_angle=0,
        use_internal_altitude=True,
        apply_bottom_downsampling=True,
    )
    temp_dir = Path(coordinator._temp_dir.name)
    with qtbot.waitSignal(coordinator.finished, timeout=5000) as blocker:
        coordinator.start()

    succeeded, failures, returned_output_path = blocker.args
    assert succeeded == 2
    assert [source.name for source, _ in failures] == ["b.jsf"]
    assert returned_output_path == output_path
    assert len(combine_calls) == 1
    combined_info_paths, combined_output = combine_calls[0]
    assert len(combined_info_paths) == 2
    assert combined_output == str(output_path)
    # The temp directory holding per-file intermediates must not survive.
    assert not temp_dir.exists()


def test_coordinator_reports_failed_when_every_file_fails(qtbot, tmp_path, monkeypatch):
    files = [tmp_path / "a.jsf", tmp_path / "b.jsf"]
    for f in files:
        f.write_bytes(b"")
    monkeypatch.setattr(
        qt_contact_picker_ui,
        "EGNTableProcessingWorker",
        _fake_egn_worker_factory({str(f) for f in files}),
    )
    combine_calls = []
    monkeypatch.setattr(
        qt_contact_picker_ui,
        "generate_egn_table_from_infos",
        lambda info_paths, output_path: combine_calls.append(info_paths),
    )

    coordinator = EGNTableBuildCoordinator(
        files,
        tmp_path / "egn_table.npz",
        chunk_size=1000,
        nadir_angle=0,
        use_internal_altitude=True,
        apply_bottom_downsampling=True,
    )
    with qtbot.waitSignal(coordinator.failed, timeout=5000) as blocker:
        coordinator.start()

    assert "every file failed" in blocker.args[0]
    assert combine_calls == []


def test_coordinator_with_no_files_fails_immediately(qtbot, tmp_path):
    coordinator = EGNTableBuildCoordinator(
        [],
        tmp_path / "egn_table.npz",
        chunk_size=1000,
        nadir_angle=0,
        use_internal_altitude=True,
        apply_bottom_downsampling=True,
    )
    with qtbot.waitSignal(coordinator.failed, timeout=1000) as blocker:
        coordinator.start()
    assert "No files" in blocker.args[0]


def test_dialog_add_folder_finds_files_recursively_without_duplicates(qtbot, tmp_path):
    nested = tmp_path / "day1" / "line2"
    nested.mkdir(parents=True)
    jsf_file = tmp_path / "day1" / "a.jsf"
    jsf_file.write_bytes(b"")
    xtf_file = nested / "b.XTF"  # case-insensitive suffix match
    xtf_file.write_bytes(b"")
    (nested / "notes.txt").write_bytes(b"")  # must be ignored

    dialog = EGNTableBuilderDialog(initial_directory=tmp_path)
    qtbot.addWidget(dialog)

    dialog._add_paths([jsf_file, xtf_file])
    assert dialog.file_list.count() == 2

    # Adding the same files again (as a folder scan would re-discover them)
    # must not create duplicate entries.
    dialog._add_paths([jsf_file, xtf_file])
    assert dialog.file_list.count() == 2

    dialog._clear_files()
    assert dialog.file_list.count() == 0
    assert dialog._files == []


def test_dialog_remove_selected_keeps_list_and_files_in_sync(qtbot, tmp_path):
    files = [tmp_path / "a.jsf", tmp_path / "b.jsf", tmp_path / "c.jsf"]
    for f in files:
        f.write_bytes(b"")
    dialog = EGNTableBuilderDialog(initial_directory=tmp_path)
    qtbot.addWidget(dialog)
    dialog._add_paths(files)

    dialog.file_list.setCurrentRow(1)  # b.jsf
    dialog._remove_selected()

    assert dialog.file_list.count() == 2
    remaining = {path.name for path in dialog._files}
    assert remaining == {"a.jsf", "c.jsf"}
    assert {dialog.file_list.item(i).text() for i in range(dialog.file_list.count())} == {
        str(files[0]),
        str(files[2]),
    }


def test_sonar_files_in_directory_filters_sorts_and_is_not_recursive(tmp_path):
    (tmp_path / "line003.jsf").write_bytes(b"")
    (tmp_path / "line001.JSF").write_bytes(b"")  # case-insensitive suffix match
    (tmp_path / "line002.xtf").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")  # must be excluded
    nested = tmp_path / "subfolder"
    nested.mkdir()
    (nested / "line099.jsf").write_bytes(b"")  # must be excluded, not recursive

    result = sonar_files_in_directory(tmp_path)

    assert [p.name for p in result] == ["line001.JSF", "line002.xtf", "line003.jsf"]


def test_sonar_files_in_directory_empty_for_a_folder_with_no_sonar_files(tmp_path):
    (tmp_path / "readme.txt").write_bytes(b"")

    assert sonar_files_in_directory(tmp_path) == []


class _FakeHoverWindow:
    """Stand-in exposing just the attributes _show_hover_stats and
    _clear_hover_stats touch, so the hover-readout formatting is tested
    without constructing a full QtContactPickerWindow (store, picker,
    docks, a loaded sonar file, ...)."""

    _show_hover_stats = QtContactPickerWindow._show_hover_stats
    _clear_hover_stats = QtContactPickerWindow._clear_hover_stats

    def __init__(self, *, sidescan_file, preprocessor, display):
        self.sidescan_file = sidescan_file
        self.preprocessor = preprocessor
        self.display = display
        self.hover_stats_label = QLabel()


def _fake_sidescan_file(num_ping=5):
    return SimpleNamespace(
        num_ping=num_ping,
        sensor_primary_altitude=np.full(num_ping, 12.5),
        sensor_speed=np.full(num_ping, 2.0),  # m/s
        sensor_heading=np.full(num_ping, 47.0),
        latitude=np.full(num_ping, 32.7),
        longitude=np.full(num_ping, -117.2),
    )


def _hover_window(*, sidescan_file=None, ping_len=4, slant_range_m=20.0):
    display = WaterfallGainModel(
        np.full((5, 2 * ping_len), 0.5), slant_range_m=slant_range_m
    )
    return _FakeHoverWindow(
        sidescan_file=sidescan_file or _fake_sidescan_file(),
        preprocessor=SimpleNamespace(ping_len=ping_len),
        display=display,
    )


def test_hover_stats_report_ping_amplitude_and_sensor_values():
    window = _hover_window()

    window._show_hover_stats(2, 1)  # column 1 < ping_len 4 -> port side

    text = window.hover_stats_label.text()
    assert "Ping 2/4" in text
    assert "Port range" in text
    assert "Amplitude" in text
    assert "Altitude 12.5 m" in text
    assert "3.9 kn" in text  # 2.0 m/s * 1.943844 knots/(m/s)
    assert "Heading 047" in text
    assert "32.700000, -117.200000" in text


def test_hover_stats_label_starboard_columns_correctly():
    window = _hover_window()

    window._show_hover_stats(0, 6)  # column 6 >= ping_len 4 -> starboard side

    assert "Stbd range" in window.hover_stats_label.text()


def test_hover_stats_omit_range_when_uncalibrated():
    window = _hover_window(slant_range_m=None)

    window._show_hover_stats(0, 0)

    text = window.hover_stats_label.text()
    assert "range" not in text
    assert "Ping 0/4" in text  # other stats are still shown


def test_hover_stats_skip_non_finite_sensor_fields():
    sidescan_file = _fake_sidescan_file()
    sidescan_file.sensor_primary_altitude[1] = float("nan")
    window = _hover_window(sidescan_file=sidescan_file)

    window._show_hover_stats(1, 0)

    assert "Altitude" not in window.hover_stats_label.text()


def test_clear_hover_stats_empties_the_label():
    window = _hover_window()
    window._show_hover_stats(0, 0)
    assert window.hover_stats_label.text() != ""

    window._clear_hover_stats()

    assert window.hover_stats_label.text() == ""


class _FakeBottomLineWindow:
    """Stand-in exposing just what _apply_bottom_edit/_toggle_bottom_edit/
    _apply_interaction_mode touch, so the edit-write and mode-toggle logic
    is tested without constructing a full QtContactPickerWindow (store,
    picker, docks, a loaded sonar file, ...) -- same technique as
    _FakeHoverWindow above."""

    _apply_bottom_edit = QtContactPickerWindow._apply_bottom_edit
    _apply_interaction_mode = QtContactPickerWindow._apply_interaction_mode
    _toggle_bottom_edit = QtContactPickerWindow._toggle_bottom_edit

    def __init__(self, *, preprocessor, view, strategy_text, edit_bottom_button):
        self.preprocessor = preprocessor
        self.view = view
        self.bottom_strategy_combo = SimpleNamespace(currentText=lambda: strategy_text)
        self.edit_bottom_button = edit_bottom_button
        self.interaction_modes = InteractionModeController()
        self.dirty_mark_count = 0

    def _mark_bottom_line_dirty(self) -> None:
        # Real autosave (writing <file>_bottom_info.npz, debounced by a
        # QTimer) is exercised separately -- here just record that an edit
        # correctly triggered it.
        self.dirty_mark_count += 1


class _FakeSidescanFileForBottomEdit:
    def __init__(self, num_ping, ping_len):
        generator = np.random.default_rng(3)
        self.data = generator.integers(
            1, 1000, size=(2, num_ping, ping_len), dtype=np.int16
        )
        self.ping_len = ping_len
        self.num_ping = num_ping


def _bottom_edit_window(
    qtbot, *, num_ping=6, ping_len=5, chunk_size=3, strategy_text="Each side individually"
):
    preprocessor = SidescanPreprocessor(
        _FakeSidescanFileForBottomEdit(num_ping, ping_len),
        chunk_size=chunk_size,
        downsampling_factor=1,
    )
    preprocessor.napari_portside_bottom = np.zeros(
        (preprocessor.num_chunk, chunk_size), dtype=int
    )
    preprocessor.napari_starboard_bottom = np.zeros(
        (preprocessor.num_chunk, chunk_size), dtype=int
    )
    preprocessor.bottom_map = np.zeros(
        (preprocessor.num_chunk, chunk_size, 2 * ping_len)
    )
    preprocessor.portside_bottom_dist = np.zeros(num_ping, dtype=int)
    preprocessor.starboard_bottom_dist = np.zeros(num_ping, dtype=int)

    view = WaterfallView()
    qtbot.addWidget(view)
    view.set_image(np.zeros((num_ping, 2 * ping_len, 3), dtype=np.uint8), fit=True)

    edit_bottom_button = QPushButton()
    edit_bottom_button.setCheckable(True)
    qtbot.addWidget(edit_bottom_button)

    return _FakeBottomLineWindow(
        preprocessor=preprocessor,
        view=view,
        strategy_text=strategy_text,
        edit_bottom_button=edit_bottom_button,
    )


def test_apply_bottom_edit_writes_portside_and_flattens_to_dist(qtbot):
    window = _bottom_edit_window(qtbot, num_ping=6, ping_len=5, chunk_size=3)

    window._apply_bottom_edit(4, 2)  # ping 4, column 2 (< ping_len 5 -> port)

    chunk_idx, local_idx = divmod(4, 3)
    assert window.preprocessor.napari_portside_bottom[chunk_idx, local_idx] == 2
    # Regression guard for the flat/chunked divergence: the write must also
    # reach portside_bottom_dist, which slant_range_correction() reads.
    assert window.preprocessor.portside_bottom_dist[4] == 2


def test_apply_bottom_edit_writes_starboard_for_columns_past_ping_len(qtbot):
    window = _bottom_edit_window(qtbot, num_ping=6, ping_len=5, chunk_size=3)

    window._apply_bottom_edit(1, 7)  # column 7 -> starboard sample 7 - 5 = 2

    assert window.preprocessor.napari_starboard_bottom[0, 1] == 2
    assert window.preprocessor.starboard_bottom_dist[1] == 2


def test_apply_bottom_edit_combine_strategy_mirrors_the_other_side(qtbot):
    window = _bottom_edit_window(
        qtbot, num_ping=6, ping_len=5, chunk_size=3, strategy_text="Combine both sides"
    )

    window._apply_bottom_edit(0, 2)  # port sample 2

    assert window.preprocessor.napari_portside_bottom[0, 0] == 2
    assert window.preprocessor.napari_starboard_bottom[0, 0] == 5 - 2  # ping_len - column


def test_apply_bottom_edit_ignores_a_row_past_the_last_chunk(qtbot):
    window = _bottom_edit_window(qtbot, num_ping=6, ping_len=5, chunk_size=3)

    window._apply_bottom_edit(999, 2)  # far past num_chunk * chunk_size

    # Must not raise, and must not touch anything.
    assert not window.preprocessor.napari_portside_bottom.any()


def test_toggling_edit_button_switches_drag_mode_and_view_edit_flag(qtbot):
    window = _bottom_edit_window(qtbot)
    window.interaction_modes.add_listener(window._apply_interaction_mode)
    window.edit_bottom_button.toggled.connect(window._toggle_bottom_edit)

    window.edit_bottom_button.setChecked(True)

    assert window.view.edit_mode is True
    assert window.view.dragMode() == QGraphicsView.DragMode.NoDrag

    window.edit_bottom_button.setChecked(False)

    assert window.view.edit_mode is False
    assert window.view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag


def test_apply_bottom_edit_marks_the_bottom_line_dirty(qtbot):
    window = _bottom_edit_window(qtbot)

    window._apply_bottom_edit(0, 2)

    assert window.dirty_mark_count == 1


class _FakeAutosaveWindow:
    """Stand-in exposing just what the autosave debounce touches, so it's
    tested against real save_bottom_info() I/O and a real QTimer without
    constructing a full QtContactPickerWindow."""

    _mark_bottom_line_dirty = QtContactPickerWindow._mark_bottom_line_dirty
    _autosave_bottom_line = QtContactPickerWindow._autosave_bottom_line
    _flush_pending_bottom_line_save = QtContactPickerWindow._flush_pending_bottom_line_save
    _default_bottom_info_path = QtContactPickerWindow._default_bottom_info_path

    def __init__(self, *, filepath, preprocessor, sidescan_file):
        self.filepath = filepath
        self.preprocessor = preprocessor
        self.sidescan_file = sidescan_file
        self.bottom_status_label = QLabel()
        self.bottom_autosave_timer = QTimer()
        self.bottom_autosave_timer.setSingleShot(True)
        self.bottom_autosave_timer.setInterval(50)  # short, so tests stay fast


def _autosave_window(tmp_path):
    filepath = tmp_path / "line.jsf"
    sidescan_file = _FakeSidescanFileForBottomEdit(num_ping=4, ping_len=5)
    sidescan_file.filepath = filepath
    preprocessor = SidescanPreprocessor(sidescan_file, chunk_size=2, downsampling_factor=1)
    preprocessor.napari_portside_bottom = np.zeros((preprocessor.num_chunk, 2), dtype=int)
    preprocessor.napari_starboard_bottom = np.zeros((preprocessor.num_chunk, 2), dtype=int)
    preprocessor.bottom_map = np.zeros((preprocessor.num_chunk, 2, 10))

    window = _FakeAutosaveWindow(
        filepath=filepath, preprocessor=preprocessor, sidescan_file=sidescan_file
    )
    window.bottom_autosave_timer.timeout.connect(window._autosave_bottom_line)
    return window


def test_mark_bottom_line_dirty_debounces_then_writes_the_ancillary_file(qtbot, tmp_path):
    window = _autosave_window(tmp_path)
    expected_path = tmp_path / "line_bottom_info.npz"

    window._mark_bottom_line_dirty()
    assert not expected_path.exists()  # debounced -- not written yet

    with qtbot.waitSignal(window.bottom_autosave_timer.timeout, timeout=1000):
        pass

    assert expected_path.exists()
    assert "auto-saved" in window.bottom_status_label.text().lower()


def test_repeated_dirty_marks_restart_the_debounce_instead_of_writing_each_time(
    qtbot, tmp_path
):
    window = _autosave_window(tmp_path)
    expected_path = tmp_path / "line_bottom_info.npz"

    for _ in range(5):
        window._mark_bottom_line_dirty()
        qtbot.wait(10)  # well under the 50ms interval -- keeps restarting it

    assert not expected_path.exists()

    with qtbot.waitSignal(window.bottom_autosave_timer.timeout, timeout=1000):
        pass
    assert expected_path.exists()


def test_flush_pending_bottom_line_save_writes_immediately(qtbot, tmp_path):
    window = _autosave_window(tmp_path)
    expected_path = tmp_path / "line_bottom_info.npz"
    window._mark_bottom_line_dirty()

    window._flush_pending_bottom_line_save()

    assert expected_path.exists()
    assert not window.bottom_autosave_timer.isActive()


def test_flush_pending_bottom_line_save_is_a_no_op_when_nothing_is_pending(
    qtbot, tmp_path
):
    window = _autosave_window(tmp_path)
    expected_path = tmp_path / "line_bottom_info.npz"

    window._flush_pending_bottom_line_save()

    assert not expected_path.exists()


def _fake_loaded_context(tmp_path):
    real_file = tmp_path / "line.jsf"
    real_file.write_bytes(b"")
    return SonarFileContext(
        filepath=real_file,
        sidescan_file=SimpleNamespace(num_ping=10, ping_len=9),
        preprocessor="unchanged-marker",
        raw_waterfall="unchanged-marker",
        built_in_processor="unchanged-marker",
        geometry="unchanged-marker",
        source_file_id=-1,
        geometry_profile_id=-1,
        slant_range_m=42.0,
        bottom_info_status="unchanged-marker",
    )


def _fake_loader_settings(tmp_path):
    return SonarLoaderSettings(
        chunk_size=256,
        default_threshold=0.7,
        downsampling_factor=32,
        active_dB=False,
        active_hist_equal=False,
        output_directory=tmp_path,
        geometry_settings=GeometrySettings(60),
    )


def test_register_in_store_assigns_new_ids_without_reloading_anything(tmp_path):
    # This is what a database switch uses instead of _load_sonar_context --
    # it must NOT re-parse the file or recompute geometry, only register
    # the already-loaded file into the new store.
    context = _fake_loaded_context(tmp_path)
    settings = _fake_loader_settings(tmp_path)

    with ContactStore(tmp_path / "new.sqlite") as store:
        result = qt_contact_picker_ui._register_in_store(
            context, settings=settings, store=store
        )

    assert result.source_file_id != -1
    assert result.geometry_profile_id != -1
    assert result.filepath == context.filepath
    # Every already-loaded piece is carried over untouched, proving no
    # reload/recompute happened.
    assert result.preprocessor == "unchanged-marker"
    assert result.raw_waterfall == "unchanged-marker"
    assert result.built_in_processor == "unchanged-marker"
    assert result.geometry == "unchanged-marker"
    assert result.slant_range_m == 42.0


def test_register_in_store_reuses_the_row_for_an_already_known_file(tmp_path):
    context = _fake_loaded_context(tmp_path)
    settings = _fake_loader_settings(tmp_path)

    with ContactStore(tmp_path / "shared.sqlite") as store:
        first = qt_contact_picker_ui._register_in_store(
            context, settings=settings, store=store
        )
        second = qt_contact_picker_ui._register_in_store(
            context, settings=settings, store=store
        )

    assert first.source_file_id == second.source_file_id


def test_confirm_replace_existing_database_true_when_nothing_is_there(qtbot, tmp_path):
    widget = QWidget()
    qtbot.addWidget(widget)

    result = QtContactPickerWindow._confirm_replace_existing_database(
        widget, tmp_path / "new.sqlite"
    )

    assert result is True


def test_confirm_replace_existing_database_respects_the_users_answer(
    qtbot, tmp_path, monkeypatch
):
    existing = tmp_path / "existing.sqlite"
    existing.write_bytes(b"")
    widget = QWidget()
    qtbot.addWidget(widget)

    monkeypatch.setattr(
        qt_contact_picker_ui.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: qt_contact_picker_ui.QMessageBox.StandardButton.No),
    )
    assert (
        QtContactPickerWindow._confirm_replace_existing_database(widget, existing)
        is False
    )

    monkeypatch.setattr(
        qt_contact_picker_ui.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: qt_contact_picker_ui.QMessageBox.StandardButton.Yes),
    )
    assert (
        QtContactPickerWindow._confirm_replace_existing_database(widget, existing)
        is True
    )
