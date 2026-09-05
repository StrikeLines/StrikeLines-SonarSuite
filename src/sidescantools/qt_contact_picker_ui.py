"""Raster Qt fallback for contact picking on systems without working OpenGL."""

from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sidescantools.bottom_line_io import (
    compute_depth_info,
    load_bottom_info,
    save_bottom_info,
)
from sidescantools.contact_picker import ContactPickerService
from sidescantools.contact_gain import (
    BuiltInGainMode,
    BuiltInGainProcessor,
    BuiltInGainRequest,
)
from sidescantools.contact_store import ContactStore, DuplicateContactAnchor
from sidescantools.contact_thumbnail import ContactThumbnailExtractor
from sidescantools.contact_ui import ContactDock
from sidescantools.custom_threading import EGNTableProcessingWorker
from sidescantools.egn_table_build import generate_egn_table_from_infos
from sidescantools.gain_settings import (
    SonarGainSettings,
    gain_settings_path,
    load_gain_settings,
    portable_egn_table_path,
    resolve_egn_table_path,
    save_gain_settings,
)
from sidescantools.georef_thread import Georeferencer
from sidescantools.geotiff_export import (
    PreparedSonarExport,
    export_prepared_waterfall,
    export_sonar_file,
    geotiff_output_path,
)
from sidescantools.interaction_mode import InteractionMode, InteractionModeController
from sidescantools.layback import (
    TowDataSummary,
    resolve_geometry_layback,
    summarize_tow_data,
)
from sidescantools.sidescan_file import SidescanFile
from sidescantools.sidescan_preproc import SidescanPreprocessor
from sidescantools.swath_geometry import GeometrySettings


class GainProcessingSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)


class GainProcessingWorker(QRunnable):
    def __init__(self, processor: BuiltInGainProcessor, request: BuiltInGainRequest):
        super().__init__()
        self.processor = processor
        self.request = request
        self.signals = GainProcessingSignals()

    def run(self) -> None:
        try:
            result = self.processor.process(
                self.request,
                progress=lambda percent, message: self.signals.progress.emit(
                    percent, message
                ),
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)


class GeoTiffExportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object, object)


class GeoTiffExportWorker(QRunnable):
    """Export one prepared file or a directory batch off the GUI thread."""

    def __init__(
        self,
        files: list[Path],
        *,
        epsg: int,
        loader_settings,
        overwrite: bool,
        prepared_current: PreparedSonarExport | None = None,
    ):
        super().__init__()
        self.files = [Path(path).resolve() for path in files]
        self.epsg = int(epsg)
        self.loader_settings = loader_settings
        self.overwrite = overwrite
        self.prepared_current = prepared_current
        self.signals = GeoTiffExportSignals()

    def run(self) -> None:
        results = []
        failures = []
        total = len(self.files)
        for file_index, source in enumerate(self.files):
            def report(file_percent: int, message: str) -> None:
                overall = int(
                    ((file_index + max(0, min(100, file_percent)) / 100.0) / total)
                    * 100
                )
                self.signals.progress.emit(
                    overall, f"{source.name}: {message}"
                )

            try:
                if self.prepared_current is not None and total == 1:
                    result = export_prepared_waterfall(
                        source,
                        self.prepared_current.rgb,
                        self.prepared_current.geometry_by_channel,
                        epsg=self.epsg,
                        pipeline_description=(
                            self.prepared_current.pipeline_description
                        ),
                        overwrite=self.overwrite,
                        progress=report,
                    )
                else:
                    result = export_sonar_file(
                        source,
                        epsg=self.epsg,
                        chunk_size=self.loader_settings.chunk_size,
                        default_threshold=(
                            self.loader_settings.default_threshold
                        ),
                        downsampling_factor=(
                            self.loader_settings.downsampling_factor
                        ),
                        active_db=self.loader_settings.active_dB,
                        active_hist_equal=(
                            self.loader_settings.active_hist_equal
                        ),
                        geometry_settings=(
                            self.loader_settings.geometry_settings
                        ),
                        overwrite=self.overwrite,
                        progress=report,
                    )
                results.append(result)
            except Exception as exc:
                failures.append((source, str(exc)))
        self.signals.finished.emit(results, failures)


def _copy_preprocessor_for_bottom_line(
    preprocessor: SidescanPreprocessor,
) -> SidescanPreprocessor:
    """Shallow-copy the preprocessor and deep-copy just the bottom-line
    arrays a whole-file automatic-detection algorithm reads or writes, so a
    background worker running that algorithm never touches the live
    instance the GUI thread renders from (mirrors ``BuiltInGainProcessor.
    _processing_copy()`` in contact_gain.py, scoped to bottom-line fields).
    """

    processor_copy = copy.copy(preprocessor)
    for name in (
        "sonar_data_proc",
        "portside_bottom_dist",
        "starboard_bottom_dist",
        "napari_portside_bottom",
        "napari_starboard_bottom",
        "bottom_map",
    ):
        if hasattr(preprocessor, name):
            setattr(
                processor_copy, name, np.array(getattr(preprocessor, name), copy=True)
            )
    return processor_copy


class BottomLineWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class BottomLineRecalcWorker(QRunnable):
    """Runs one whole-file automatic bottom-line algorithm on a private
    preprocessor copy built on the GUI thread (see
    ``_copy_preprocessor_for_bottom_line``), so the only thing crossing
    back to the GUI thread is a finished copy, swapped in by whole-object
    attribute reassignment -- never an in-place mutation of the live
    preprocessor's arrays while the GUI thread might be rendering or
    handling a manual edit concurrently.
    """

    def __init__(self, preprocessor_copy: SidescanPreprocessor, run_algorithm):
        super().__init__()
        self.preprocessor_copy = preprocessor_copy
        self.run_algorithm = run_algorithm
        self.signals = BottomLineWorkerSignals()

    def run(self) -> None:
        try:
            self.run_algorithm(self.preprocessor_copy)
            self.preprocessor_copy.sync_flat_bottom_to_chunked()
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(self.preprocessor_copy)


# EGN parameters that egn_table_build.py itself keeps out of the UI to stay
# simple: number of beam-angle bins and the range reduction factor.
EGN_TABLE_RESOLUTION = [360, 2]


class EGNTableBuildCoordinator(QObject):
    """Runs ``generate_egn_info`` for many files in parallel on the shared
    thread pool, then combines whatever succeeded into one EGN table.

    A single unreadable or oddly-formatted file in a folder full of survey
    data must not sink the whole batch, so per-file failures are collected
    and reported rather than aborting the build.
    """

    progress = Signal(int, int)  # (completed, total)
    finished = Signal(int, list, object)  # (succeeded_count, failures, output_path)
    failed = Signal(str)  # nothing could be built

    def __init__(
        self,
        files: list[Path],
        output_path: Path,
        *,
        chunk_size: int,
        nadir_angle: int,
        use_internal_altitude: bool,
        apply_bottom_downsampling: bool,
        thread_pool: QThreadPool | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.files = list(files)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.nadir_angle = nadir_angle
        self.use_internal_altitude = use_internal_altitude
        self.apply_bottom_downsampling = apply_bottom_downsampling
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="sidescantools-egn-")
        self._completed = 0
        self._succeeded_info_paths: list[Path] = []
        self._failures: list[tuple[Path, str]] = []
        self._workers: list[EGNTableProcessingWorker] = []

    def start(self) -> None:
        if not self.files:
            self._temp_dir.cleanup()
            self.failed.emit("No files were selected.")
            return
        temp_dir = Path(self._temp_dir.name)
        for index, source in enumerate(self.files):
            # Matches the convention used everywhere else in this project:
            # bottom-line detection is saved next to the source file. When it
            # isn't there, generate_egn_info() already falls back to internal
            # altitude on its own -- no pre-filtering needed here.
            bottom_path = source.parent / f"{source.stem}_bottom_info.npz"
            info_path = temp_dir / f"{index:04d}_{source.stem}_egn_info.npz"
            worker = EGNTableProcessingWorker(
                str(source),
                str(bottom_path),
                str(info_path),
                self.chunk_size,
                self.nadir_angle,
                self.use_internal_altitude,
                self.apply_bottom_downsampling,
                egn_table_parameters=EGN_TABLE_RESOLUTION,
            )
            worker.signals.error_signal.connect(
                lambda err, source=source: self._on_file_error(source, err)
            )
            worker.signals.finished.connect(
                lambda source=source, info_path=info_path: self._on_file_finished(
                    source, info_path
                )
            )
            self._workers.append(worker)
            self.thread_pool.start(worker)

    def _on_file_error(self, source: Path, error: Exception) -> None:
        self._failures.append((source, str(error)))

    def _on_file_finished(self, source: Path, info_path: Path) -> None:
        self._completed += 1
        already_failed = any(failed_source == source for failed_source, _ in self._failures)
        if not already_failed:
            if info_path.exists():
                self._succeeded_info_paths.append(info_path)
            else:
                self._failures.append((source, "no EGN info was produced"))
        self.progress.emit(self._completed, len(self.files))
        if self._completed == len(self.files):
            self._combine()

    def _combine(self) -> None:
        try:
            if self._succeeded_info_paths:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                generate_egn_table_from_infos(
                    [str(path) for path in self._succeeded_info_paths],
                    str(self.output_path),
                )
        except Exception as exc:
            self._temp_dir.cleanup()
            self.failed.emit(f"could not combine the EGN table: {exc}")
            return
        self._temp_dir.cleanup()
        if self._succeeded_info_paths:
            self.finished.emit(
                len(self._succeeded_info_paths), self._failures, self.output_path
            )
        else:
            self.failed.emit("every file failed; no EGN table was produced")


class EGNTableBuilderDialog(QDialog):
    """Pick individual files or a whole folder from disk and build a new EGN
    table from them, without first adding those files to a project."""

    def __init__(self, parent=None, *, initial_directory: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Build EGN Table")
        self.resize(640, 520)
        self._initial_directory = initial_directory or Path.home()
        self._files: list[Path] = []
        self._coordinator: EGNTableBuildCoordinator | None = None
        self.result_table_path: Path | None = None

        self.file_list = QListWidget()
        add_files_button = QPushButton("Add Files…")
        add_files_button.clicked.connect(self._add_files)
        add_folder_button = QPushButton("Add Folder…")
        add_folder_button.clicked.connect(self._add_folder)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_files)
        file_buttons = QHBoxLayout()
        for button in (add_files_button, add_folder_button, remove_button, clear_button):
            file_buttons.addWidget(button)

        form = QFormLayout()
        self.nadir_angle_spin = QSpinBox()
        self.nadir_angle_spin.setRange(0, 89)
        self.nadir_angle_spin.setSuffix("°")
        self.nadir_angle_spin.setToolTip(
            "Angle between perpendicular and first bottom return; leave 0 if unsure."
        )
        form.addRow("Nadir angle", self.nadir_angle_spin)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1, 100_000)
        self.chunk_size_spin.setValue(1000)
        form.addRow("Chunk size", self.chunk_size_spin)

        self.internal_altitude_checkbox = QCheckBox(
            "Use internal altitude (automatic per file when no bottom-line "
            "detection is found next to it)"
        )
        form.addRow("", self.internal_altitude_checkbox)

        self.apply_downsampling_checkbox = QCheckBox(
            "Apply the downsampling used for bottom-line detection, when available"
        )
        self.apply_downsampling_checkbox.setChecked(True)
        form.addRow("", self.apply_downsampling_checkbox)

        default_name = f"egn_table_{datetime.now():%Y-%m-%d_%H-%M-%S}.npz"
        self.output_path_edit = QLineEdit(str(self._initial_directory / default_name))
        browse_output_button = QPushButton("Browse…")
        browse_output_button.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path_edit, 1)
        output_row.addWidget(browse_output_button)
        form.addRow("Output table", output_row)

        self.status_label = QLabel(
            "Add sonar files or a folder, then Build. Files without a matching "
            "<name>_bottom_info.npz next to them use internal altitude instead."
        )
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.build_button = QPushButton("Build")
        self.build_button.clicked.connect(self._start_build)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.build_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sonar files (.jsf / .xtf)"))
        layout.addWidget(self.file_list, 1)
        layout.addLayout(file_buttons)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_row)

    def _add_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select sidescan files",
            str(self._initial_directory),
            "Sidescan files (*.jsf *.xtf);;All files (*)",
        )
        self._add_paths(Path(name) for name in filenames)

    def _add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select a folder of sidescan files", str(self._initial_directory)
        )
        if not directory:
            return
        found = sorted(
            path
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.casefold() in (".jsf", ".xtf")
        )
        if not found:
            self.status_label.setText(f"No .jsf/.xtf files found under {directory}")
            return
        self._add_paths(found)

    def _add_paths(self, paths) -> None:
        added = 0
        for path in paths:
            path = path.resolve()
            if path in self._files:
                continue
            self._files.append(path)
            self.file_list.addItem(str(path))
            added += 1
        if added:
            self.status_label.setText(f"{len(self._files)} file(s) queued.")

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            del self._files[row]
        self.status_label.setText(f"{len(self._files)} file(s) queued.")

    def _clear_files(self) -> None:
        self._files.clear()
        self.file_list.clear()
        self.status_label.setText("Add sonar files or a folder, then Build.")

    def _browse_output(self) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Save EGN table as",
            self.output_path_edit.text(),
            "NumPy tables (*.npz)",
        )
        if destination:
            self.output_path_edit.setText(destination)

    def _start_build(self) -> None:
        if not self._files:
            self.status_label.setText("Add at least one sonar file first.")
            return
        output_text = self.output_path_edit.text().strip()
        if not output_text:
            self.status_label.setText("Choose an output path for the EGN table first.")
            return
        output_path = Path(output_text)
        if output_path.exists():
            answer = QMessageBox.question(
                self, "Overwrite EGN table", f"Replace {output_path.name}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_inputs_enabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Processing 0/{len(self._files)} file(s)…")

        self._coordinator = EGNTableBuildCoordinator(
            self._files,
            output_path,
            chunk_size=self.chunk_size_spin.value(),
            nadir_angle=self.nadir_angle_spin.value(),
            use_internal_altitude=self.internal_altitude_checkbox.isChecked(),
            apply_bottom_downsampling=self.apply_downsampling_checkbox.isChecked(),
        )
        self._coordinator.progress.connect(self._on_progress)
        self._coordinator.finished.connect(self._on_finished)
        self._coordinator.failed.connect(self._on_failed)
        self._coordinator.start()

    def _on_progress(self, completed: int, total: int) -> None:
        self.progress_bar.setValue(round(100 * completed / max(total, 1)))
        self.status_label.setText(f"Processing {completed}/{total} file(s)…")

    def _on_finished(self, succeeded: int, failures: list, output_path: Path) -> None:
        self._set_inputs_enabled(True)
        self.progress_bar.setValue(100)
        self.result_table_path = output_path
        message = f"Built {output_path.name} from {succeeded} of {len(self._files)} file(s)."
        if failures:
            failed_names = ", ".join(source.name for source, _ in failures)
            message += f" {len(failures)} file(s) skipped: {failed_names}."
        self.status_label.setText(message)

    def _on_failed(self, message: str) -> None:
        self._set_inputs_enabled(True)
        self.status_label.setText(f"Build failed: {message}")

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (
            self.build_button,
            self.close_button,
            self.file_list,
            self.nadir_angle_spin,
            self.chunk_size_spin,
            self.internal_altitude_checkbox,
            self.apply_downsampling_checkbox,
            self.output_path_edit,
        ):
            widget.setEnabled(enabled)


def waterfall_rgb(chunk: np.ndarray) -> np.ndarray:
    """Convert one waterfall chunk to deterministic grayscale RGB."""

    values = np.asarray(chunk, dtype=float)
    finite = np.isfinite(values)
    gray = np.zeros(values.shape, dtype=np.uint8)
    if np.any(finite):
        low, high = np.percentile(values[finite], (1.0, 99.5))
        if high <= low:
            low = float(np.min(values[finite]))
            high = float(np.max(values[finite]))
        if high > low:
            scaled = (values[finite] - low) * (255.0 / (high - low))
            gray[finite] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    # A warm grayscale remains readable while preserving sonar contrast.
    return np.stack((gray, (gray * 0.78).astype(np.uint8), gray // 3), axis=2)


def _reshape_chunked(
    array: np.ndarray, ping_len: int, source_ping_count: int
) -> np.ndarray:
    """Flatten a ``[num_chunk, chunk_size, 2*ping_len]`` per-chunk array (as
    produced by ``SidescanPreprocessor`` for both the sonar waterfall and the
    bottom-line overlay) into a continuous ``[num_ping, 2*ping_len]`` array,
    dropping the final chunk's padding."""

    array = np.asarray(array)
    expected_width = 2 * ping_len
    if array.ndim != 3 or array.shape[2] != expected_width:
        raise ValueError("chunked array dimensions are inconsistent")
    available_pings = array.shape[0] * array.shape[1]
    if not 0 < source_ping_count <= available_pings:
        raise ValueError("source ping count is inconsistent with the chunked array")
    return array.reshape(available_pings, expected_width)[:source_ping_count]


def logical_waterfall(preprocessor, source_ping_count: int) -> np.ndarray:
    """Return the continuous 2D waterfall while excluding final-chunk padding."""

    return _reshape_chunked(
        preprocessor.napari_fullmat, preprocessor.ping_len, source_ping_count
    )


def logical_bottom_overlay(preprocessor, source_ping_count: int) -> np.ndarray:
    """Return the continuous 2D bottom-line mask matching ``logical_waterfall``'s
    row indexing. ``bottom_map`` is ``float64`` (0.0/1.0), not boolean --
    callers treat it as truthy where > 0."""

    return _reshape_chunked(
        preprocessor.bottom_map, preprocessor.ping_len, source_ping_count
    )


SONAR_FILE_SUFFIXES = (".jsf", ".xtf")


def sonar_files_in_directory(directory: str | os.PathLike) -> list[Path]:
    """List .jsf/.xtf files directly inside a directory, sorted by name.

    Not recursive -- survey exports are conventionally one flat folder of
    line files, and this backs the Qt picker's "next/previous file"
    navigation, which stays within the folder the current file was opened
    from rather than searching subfolders.
    """

    # Resolved so entries compare equal to a resolved current filepath
    # regardless of whether either was originally given as relative.
    directory = Path(directory).resolve()
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in SONAR_FILE_SUFFIXES
    )


def egn_table_nadir_angle(table_path: Path | None) -> float:
    """Read the EGN table's nadir angle without another UI setting."""

    if table_path is None:
        return 0.0
    try:
        with np.load(table_path) as table:
            return float(table["nadir_angle"]) if "nadir_angle" in table else 0.0
    except Exception:
        # The processing worker validates the selected table and reports a
        # useful error. A missing optional angle should retain the old default.
        return 0.0


@dataclass
class SonarLoaderSettings:
    """Session-level settings that stay fixed while navigating between files
    in a directory -- as opposed to per-file state, which is rebuilt by
    _load_sonar_context() on every file open/switch."""

    chunk_size: int
    default_threshold: float
    downsampling_factor: int
    active_dB: bool
    active_hist_equal: bool
    output_directory: Path
    geometry_settings: GeometrySettings


@dataclass
class SonarFileContext:
    """Everything the Qt picker needs to display and pick contacts on one
    sonar file. Deliberately excludes the WaterfallGainModel ('display') and
    ContactPickerService ('picker') -- those are built by the caller so
    display's gain/TVG settings and object identity survive a file switch
    (see QtContactPickerWindow.load_file())."""

    filepath: Path
    sidescan_file: SidescanFile
    preprocessor: SidescanPreprocessor
    raw_waterfall: np.ndarray
    built_in_processor: BuiltInGainProcessor
    geometry: dict
    source_file_id: int
    geometry_profile_id: int
    slant_range_m: float
    bottom_info_status: str
    geometry_settings: GeometrySettings | None = None
    tow_data: TowDataSummary = TowDataSummary(None, None)
    layback_source: str = "No recorded layback or cable out"
    layback_override_m: float | None = None


def _prepare_file_geometry(
    filepath: Path,
    sidescan_file: SidescanFile,
    geometry_settings: GeometrySettings,
    output_directory: Path,
) -> dict:
    return {
        channel: Georeferencer(
            filepath,
            channel=channel,
            sidescan_file=sidescan_file,
            geometry_settings=geometry_settings,
            output_folder=output_directory,
        ).prepare_swath_geometry()
        for channel in (0, 1)
    }


def _load_sonar_context(
    filepath: Path,
    *,
    settings: SonarLoaderSettings,
    store: ContactStore,
) -> SonarFileContext:
    """Load one sonar file and prepare everything the Qt picker needs for
    it. Used both for the file run_qt_contact_picker() opens initially and
    for every subsequent QtContactPickerWindow.load_file() call when the
    processor navigates to another file in the same directory -- every
    per-file preparation step lives here exactly once.
    """

    # Resolved so this always matches sonar_files_in_directory()'s entries
    # for next/previous lookups, even if the caller passed a relative path.
    filepath = Path(filepath).resolve()
    print(f"Loading {filepath.name}…")
    sidescan_file = SidescanFile(filepath)
    tow_data = summarize_tow_data(sidescan_file)
    try:
        saved_settings = load_gain_settings(filepath)
    except Exception:
        # The window's normal settings restore path reports malformed sidecars.
        saved_settings = None
    layback_override_m = (
        saved_settings.layback_override_m if saved_settings is not None else None
    )
    geometry_settings, layback_source = resolve_geometry_layback(
        settings.geometry_settings,
        tow_data,
        manual_layback_m=layback_override_m,
    )
    preprocessor = SidescanPreprocessor(
        sidescan_file=sidescan_file,
        chunk_size=settings.chunk_size,
        downsampling_factor=settings.downsampling_factor,
    )
    depth_info = compute_depth_info(sidescan_file, settings.downsampling_factor)
    preprocessor.init_napari_bottom_detect(
        settings.default_threshold,
        active_dB=settings.active_dB,
        active_hist_equal=settings.active_hist_equal,
        depth_info=depth_info,
    )
    # A prior manual correction (or a previous automatic run) saved next to
    # the file takes priority over the fresh automatic guess above -- same
    # convention egn_table_build.py already relies on for this file.
    bottom_info_path = filepath.parent / f"{filepath.stem}_bottom_info.npz"
    if bottom_info_path.is_file():
        load_bottom_info(bottom_info_path, preprocessor, sidescan_file)
        bottom_info_status = f"Loaded bottom line from {bottom_info_path.name}"
    else:
        # No ancillary file next to this sonar file yet -- the fresh
        # automatic guess becomes the working line, and is written out
        # immediately so one now exists (same file the CLI, EGN table
        # builder, and Napari bottom editor all read/write).
        save_bottom_info(bottom_info_path, preprocessor, sidescan_file)
        bottom_info_status = (
            f"Automatic bottom-line detection (saved to {bottom_info_path.name})"
        )
    raw_waterfall = logical_waterfall(preprocessor, sidescan_file.num_ping)
    slant_range_m = float(np.median(sidescan_file.slant_range))
    built_in_processor = BuiltInGainProcessor(preprocessor, raw_waterfall)

    print("Preparing contact geometry…")
    geometry = _prepare_file_geometry(
        filepath,
        sidescan_file,
        geometry_settings,
        settings.output_directory,
    )
    source_stat = filepath.stat()
    source = store.register_source_file(
        filepath,
        format=filepath.suffix.lstrip("."),
        ping_count=sidescan_file.num_ping,
        source_sample_count=sidescan_file.ping_len,
        file_size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    profile_id = store.get_or_create_geometry_profile(geometry_settings)
    store.mark_stale_for_profile(source.id, profile_id)

    return SonarFileContext(
        filepath=filepath,
        sidescan_file=sidescan_file,
        preprocessor=preprocessor,
        raw_waterfall=raw_waterfall,
        built_in_processor=built_in_processor,
        geometry=geometry,
        source_file_id=source.id,
        geometry_profile_id=profile_id,
        slant_range_m=slant_range_m,
        bottom_info_status=bottom_info_status,
        geometry_settings=geometry_settings,
        tow_data=tow_data,
        layback_source=layback_source,
        layback_override_m=layback_override_m,
    )


def _register_in_store(
    context: SonarFileContext, *, settings: SonarLoaderSettings, store: ContactStore
) -> SonarFileContext:
    """Re-register an already-loaded file's context into a *different*
    store -- used when the processor switches the active contacts database
    without changing which sonar file is open. Deliberately skips every
    expensive step in _load_sonar_context (no file reparse, no waterfall
    stitching, no georeferencing recompute): only new source_file_id /
    geometry_profile_id values scoped to the new database are needed.
    """

    filepath = context.filepath
    source_stat = filepath.stat()
    source = store.register_source_file(
        filepath,
        format=filepath.suffix.lstrip("."),
        ping_count=context.sidescan_file.num_ping,
        source_sample_count=context.sidescan_file.ping_len,
        file_size_bytes=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
    )
    active_geometry_settings = context.geometry_settings or settings.geometry_settings
    profile_id = store.get_or_create_geometry_profile(active_geometry_settings)
    store.mark_stale_for_profile(source.id, profile_id)
    return replace(context, source_file_id=source.id, geometry_profile_id=profile_id)


def _build_contact_picker(
    context: SonarFileContext,
    *,
    store: ContactStore,
    display: "WaterfallGainModel",
) -> ContactPickerService:
    """Build the per-file ContactPickerService, wired to the *persistent*
    display object so its callbacks stay valid across a file switch (see
    QtContactPickerWindow.load_file())."""

    return ContactPickerService(
        sidescan_file=context.sidescan_file,
        preprocessor=context.preprocessor,
        source_file_id=context.source_file_id,
        geometry_profile_id=context.geometry_profile_id,
        geometry_by_channel=context.geometry,
        store=store,
        thumbnail_factory=ContactThumbnailExtractor(
            preprocessor=context.preprocessor,
            sidescan_file=context.sidescan_file,
            logical_waterfall_provider=display.corrected,
            display_pipeline_provider=lambda: display.pipeline_description,
        ),
        display_intensity_provider=lambda anchor: display.corrected_value(
            anchor.global_ping_index, anchor.display_sample_index
        ),
        display_pipeline=lambda: display.pipeline_description,
    )


class WaterfallGainModel:
    """Non-destructive display gain: a broadband offset plus a standard
    time-variable-gain (TVG) curve, the simple across-track normalization
    sonar hardware (including EdgeTech's own acquisition-time TVG) applies to
    compensate for spreading and absorption loss increasing with range --
    much cheaper to reason about and tune by eye than building/applying an
    EGN table, at the cost of not being empirically fit to the actual data.
    """

    # Near nadir, range is ~0 and log10(range) diverges; TVG curves
    # conventionally reference the spreading term to a small starting range
    # instead. Expressed as a fraction of the waterfall's own representative
    # range so it scales sensibly with whatever data is loaded.
    _TVG_FLOOR_FRACTION = 0.02
    DEFAULT_OVERALL_GAIN_DB = -5.0
    MIN_OVERALL_GAIN_DB = -30
    MAX_OVERALL_GAIN_DB = 20
    DEFAULT_TVG_SPREADING_DB_PER_DECADE = 5.0
    DEFAULT_TVG_ABSORPTION_DB_PER_M = 0.08
    MIN_TVG_SPREADING_DB_PER_DECADE = -20
    MAX_TVG_SPREADING_DB_PER_DECADE = 34
    MIN_TVG_ABSORPTION_DB_PER_M = 0.0
    MAX_TVG_ABSORPTION_DB_PER_M = 0.2
    _NORMALIZE_MAX_SAMPLED_VALUES = 4_000_000
    _NORMALIZE_RANGE_BINS = 128
    DEFAULT_NORMALIZE_TARGET_PERCENT = 30
    MIN_NORMALIZE_TARGET_PERCENT = 1
    MAX_NORMALIZE_TARGET_PERCENT = 100
    NORMALIZE_OVERALL_GAIN_DB = -10.0
    _NORMALIZE_MIN_TOTAL_GAIN_DB = -30.0
    _NORMALIZE_MAX_TOTAL_GAIN_DB = 60.0

    def __init__(
        self,
        source: np.ndarray,
        *,
        slant_range_m: float | None = None,
        base_pipeline: str = "qt-continuous-waterfall-v1|raw",
    ):
        self.overall_gain_db = self.DEFAULT_OVERALL_GAIN_DB
        self.tvg_spreading_db_per_decade = (
            self.DEFAULT_TVG_SPREADING_DB_PER_DECADE
        )
        self.tvg_absorption_db_per_m = self.DEFAULT_TVG_ABSORPTION_DB_PER_M
        self.normalize_target_percent = self.DEFAULT_NORMALIZE_TARGET_PERCENT
        # A single representative max range for the whole waterfall, not a
        # per-ping value -- deliberately simple, matching how a processor
        # tunes TVG by eye rather than from a precise per-ping model. Falls
        # back to a unitless "1.0" reference when no calibrated slant range
        # is available, so the curve shape still works, just without
        # physically meaningful dB/m units on the absorption slider. This is
        # a property of the source *file*, set once here and deliberately
        # not re-derived by set_source(), which is called on every
        # processing-mode change and must not silently reset it.
        self._reference_range_m = 1.0
        self._range_is_calibrated = False
        self.set_source(source, base_pipeline=base_pipeline, slant_range_m=slant_range_m)

    def set_source(
        self,
        source: np.ndarray,
        *,
        base_pipeline: str,
        slant_range_m: float | None = None,
    ) -> None:
        source = np.asarray(source, dtype=float)
        if source.ndim != 2 or source.shape[1] < 4 or source.shape[1] % 2:
            raise ValueError("continuous waterfall must have two equal-width channels")
        self.source = source
        self.base_pipeline = base_pipeline
        # Only update calibration when the caller actually has a new value
        # to give (switching to a different file). Callers that are just
        # swapping display data for the *same* file (a processing-mode
        # change) omit this and keep whatever calibration is already set.
        if slant_range_m and slant_range_m > 0:
            self._reference_range_m = float(slant_range_m)
            self._range_is_calibrated = True
        channel_width = source.shape[1] // 2
        self._sample_fraction = np.concatenate(
            (
                np.linspace(1.0, 0.0, channel_width),
                np.linspace(0.0, 1.0, channel_width),
            )
        )
        self._range_m = self._sample_fraction * self._reference_range_m
        # A fitted equalization curve belongs to the exact data being shown.
        # Processing-mode and file changes call set_source(), so never carry a
        # residual correction over to a different waterfall.
        self._normalization_gain_db = np.zeros(source.shape[1], dtype=float)
        self._normalization_active = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def pipeline_description(self) -> str:
        description = (
            self.base_pipeline
            + f"|gain={self.overall_gain_db:.1f}dB"
            f"|tvg-spreading={self.tvg_spreading_db_per_decade:.1f}dB/decade"
            f"|tvg-absorption={self.tvg_absorption_db_per_m:.2f}dB/m"
        )
        if self._normalization_active:
            description += f"|swath-equalized={self.normalize_target_percent}pct"
        return description

    def gain_profile(self) -> np.ndarray:
        floor_m = self._reference_range_m * self._TVG_FLOOR_FRACTION
        spreading_db = self.tvg_spreading_db_per_decade * np.log10(
            np.maximum(self._range_m, floor_m) / floor_m
        )
        absorption_db = self.tvg_absorption_db_per_m * self._range_m
        gain_db = (
            self.overall_gain_db
            + spreading_db
            + absorption_db
            + self._normalization_gain_db
        )
        return np.power(10.0, gain_db / 20.0)

    def clear_normalization(self) -> None:
        self._normalization_gain_db = np.zeros(self.source.shape[1], dtype=float)
        self._normalization_active = False

    @property
    def auto_tvg_active(self) -> bool:
        return self._normalization_active

    @property
    def auto_tvg_gain_db(self) -> tuple[float, ...]:
        if not self._normalization_active:
            return ()
        return tuple(float(value) for value in self._normalization_gain_db)

    def restore_auto_tvg_gain(self, gain_db: tuple[float, ...]) -> None:
        values = np.asarray(gain_db, dtype=float)
        if values.shape != (self.source.shape[1],):
            raise ValueError(
                "saved Auto TVG correction does not match the waterfall width"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("saved Auto TVG correction contains invalid values")
        self._normalization_gain_db = values.copy()
        self._normalization_active = True

    def set_normalize_target_percent(self, target_percent: int) -> None:
        target_percent = int(target_percent)
        if not (
            self.MIN_NORMALIZE_TARGET_PERCENT
            <= target_percent
            <= self.MAX_NORMALIZE_TARGET_PERCENT
        ):
            raise ValueError("normalize target brightness must be between 1% and 100%")
        if target_percent != self.normalize_target_percent:
            self.clear_normalization()
        self.normalize_target_percent = target_percent

    def normalize_tvg(self) -> tuple[float, float, float]:
        """Equalize typical brightness across the swath around the selected target.

        Broad loss is exposed through the normal overall/spreading/absorption
        controls. A smooth per-column residual handles port/starboard
        imbalance and non-monotonic dark or blown-out areas that three scalar
        controls cannot represent. The fit uses representative range bands so
        individual contacts and shadows do not dictate the result.
        """

        channel_width = self.source.shape[1] // 2
        row_limit = max(
            1,
            self._NORMALIZE_MAX_SAMPLED_VALUES // self.source.shape[1],
        )
        row_count = min(self.source.shape[0], row_limit)
        row_indices = np.linspace(
            0, self.source.shape[0] - 1, row_count, dtype=int
        )
        sampled = self.source[row_indices]
        finite_sample = sampled[np.isfinite(sampled)]
        positive_sample = finite_sample[finite_sample > 0.0]
        if positive_sample.size < 4:
            raise ValueError("not enough finite sonar returns to normalize TVG")
        low_signal = float(np.percentile(positive_sample, 1.0))
        amplitude_floor = max(1e-6, min(1e-3, low_signal * 0.1))

        # Port is displayed outer-to-nadir and starboard nadir-to-outer. Work
        # on each independently in near-to-far order: sharing one empirical
        # curve would preserve a port/starboard brightness mismatch.
        port_near_to_far = sampled[:, :channel_width][:, ::-1]
        starboard_near_to_far = sampled[:, channel_width:]
        range_near_to_far = self._range_m[channel_width:]

        band_count = min(self._NORMALIZE_RANGE_BINS, channel_width)
        band_edges = np.linspace(0, channel_width, band_count + 1, dtype=int)
        band_centers = np.asarray(
            [(start + stop - 1) / 2.0 for start, stop in zip(band_edges[:-1], band_edges[1:])]
        )
        brightness_profiles = []

        for side in (port_near_to_far, starboard_near_to_far):
            side_db = np.full(band_count, np.nan, dtype=float)
            for band_index, (start, stop) in enumerate(
                zip(band_edges[:-1], band_edges[1:])
            ):
                band = side[:, start:stop].reshape(-1)
                usable = band[np.isfinite(band)]
                if usable.size < 4:
                    continue
                # Zeros are dark data, not missing data. Give them a small
                # floor so they request more gain without producing infinity.
                representative = float(np.median(np.maximum(usable, amplitude_floor)))
                side_db[band_index] = 20.0 * math.log10(representative)

            valid = np.isfinite(side_db)
            if np.count_nonzero(valid) < 2:
                raise ValueError("not enough finite sonar returns to normalize TVG")
            side_db = np.interp(band_centers, band_centers[valid], side_db[valid])
            if band_count >= 5:
                smoothing_kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0)) / 9.0
                side_db = np.convolve(
                    np.pad(side_db, (2, 2), mode="edge"),
                    smoothing_kernel,
                    mode="valid",
                )
            brightness_profiles.append(side_db)

        representative_range = np.interp(
            band_centers, np.arange(channel_width), range_near_to_far
        )
        floor_m = self._reference_range_m * self._TVG_FLOOR_FRACTION
        one_side_log_range = np.log10(
            np.maximum(representative_range, floor_m) / floor_m
        )
        brightness_db = np.concatenate(brightness_profiles)
        log_range = np.tile(one_side_log_range, 2)
        range_m = np.tile(representative_range, 2)
        weights = np.ones_like(brightness_db)
        spreading = self.tvg_spreading_db_per_decade
        absorption = self.tvg_absorption_db_per_m

        # Iteratively reweighted least squares prevents a wreck, shadow, or
        # seam present in a handful of bands from steering the full-width TVG.
        for _ in range(6):
            spreading, absorption = self._bounded_tvg_fit(
                brightness_db, log_range, range_m, weights
            )
            flattened = brightness_db + spreading * log_range + absorption * range_m
            residual = flattened - np.median(flattened)
            robust_scale = 1.4826 * float(np.median(np.abs(residual)))
            if robust_scale <= 1e-9:
                break
            cutoff = 1.5 * robust_scale
            weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))

        target_amplitude = self.normalize_target_percent / 100.0
        target_db = 20.0 * math.log10(target_amplitude)
        # Match the precision the UI can actually retain; the empirical
        # residual below is calculated after rounding, so setting the visible
        # controls cannot subtly undo the equalization.
        overall = self.NORMALIZE_OVERALL_GAIN_DB
        spreading = float(round(spreading))
        absorption = float(round(absorption, 2))

        desired_total_gain = np.clip(
            target_db - brightness_db,
            self._NORMALIZE_MIN_TOTAL_GAIN_DB,
            self._NORMALIZE_MAX_TOTAL_GAIN_DB,
        )
        parametric_gain = overall + spreading * log_range + absorption * range_m
        residual_profiles = np.split(desired_total_gain - parametric_gain, 2)
        full_resolution_residuals = [
            np.interp(np.arange(channel_width), band_centers, residual)
            for residual in residual_profiles
        ]

        self.overall_gain_db = overall
        self.tvg_spreading_db_per_decade = spreading
        self.tvg_absorption_db_per_m = absorption
        self._normalization_gain_db = np.concatenate(
            (full_resolution_residuals[0][::-1], full_resolution_residuals[1])
        )
        self._normalization_active = True
        return (
            self.overall_gain_db,
            self.tvg_spreading_db_per_decade,
            self.tvg_absorption_db_per_m,
        )

    @classmethod
    def _bounded_tvg_fit(
        cls,
        brightness_db: np.ndarray,
        log_range: np.ndarray,
        range_m: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """Solve the two-parameter weighted fit within the UI's limits."""

        weights = np.asarray(weights, dtype=float)
        weights /= np.sum(weights)
        y = brightness_db - np.sum(weights * brightness_db)
        x = log_range - np.sum(weights * log_range)
        r = range_m - np.sum(weights * range_m)

        s_min = cls.MIN_TVG_SPREADING_DB_PER_DECADE
        s_max = cls.MAX_TVG_SPREADING_DB_PER_DECADE
        a_min = cls.MIN_TVG_ABSORPTION_DB_PER_M
        a_max = cls.MAX_TVG_ABSORPTION_DB_PER_M
        candidates: list[tuple[float, float]] = []

        design = np.column_stack((x, r))
        root_weights = np.sqrt(weights)
        unconstrained, *_ = np.linalg.lstsq(
            design * root_weights[:, None], -y * root_weights, rcond=None
        )

        def add_candidate(spreading_value: float, absorption_value: float) -> None:
            candidates.append(
                (
                    float(np.clip(spreading_value, s_min, s_max)),
                    float(np.clip(absorption_value, a_min, a_max)),
                )
            )

        add_candidate(float(unconstrained[0]), float(unconstrained[1]))
        rr = float(np.sum(weights * r * r))
        xx = float(np.sum(weights * x * x))
        for spreading_bound in (s_min, s_max):
            best_absorption = (
                -float(np.sum(weights * r * (y + spreading_bound * x))) / rr
                if rr > 0.0
                else a_min
            )
            add_candidate(spreading_bound, best_absorption)
        for absorption_bound in (a_min, a_max):
            best_spreading = (
                -float(np.sum(weights * x * (y + absorption_bound * r))) / xx
                if xx > 0.0
                else 0.0
            )
            add_candidate(best_spreading, absorption_bound)

        def error(candidate: tuple[float, float]) -> float:
            spreading_value, absorption_value = candidate
            residual = y + spreading_value * x + absorption_value * r
            return float(np.sum(weights * residual * residual))

        return min(candidates, key=error)

    def corrected(self) -> np.ndarray:
        values = np.nan_to_num(
            self.source, nan=0.0, posinf=1.0, neginf=0.0, copy=True
        )
        values *= self.gain_profile()[None, :]
        return np.clip(values, 0.0, 1.0)

    def corrected_value(self, global_ping: int, display_sample: int) -> float:
        if not 0 <= global_ping < self.source.shape[0]:
            raise IndexError("global ping is outside the continuous waterfall")
        if not 0 <= display_sample < self.source.shape[1]:
            raise IndexError("display sample is outside the continuous waterfall")
        value = float(self.source[global_ping, display_sample])
        if not math.isfinite(value):
            return 0.0
        return float(
            np.clip(value * self.gain_profile()[display_sample], 0.0, 1.0)
        )

    def range_at_column(self, display_sample: int) -> tuple[float, bool]:
        """Slant range in meters at one display column, and whether that
        range is a real calibrated value. Before any slant_range_m has ever
        been supplied, ``_range_m`` is only a 0-1 fraction of the waterfall
        width (see ``_reference_range_m``'s fallback above) -- not physically
        meaningful, so callers must check the second value before labeling
        the number as meters.
        """
        return float(self._range_m[display_sample]), self._range_is_calibrated

    def render_rgb(self) -> np.ndarray:
        gray = np.rint(self.corrected() * 255.0).astype(np.uint8)
        return np.stack(
            (gray, (gray.astype(float) * 0.78).astype(np.uint8), gray // 3),
            axis=2,
        )


class WaterfallView(QGraphicsView):
    """Continuously scrolling waterfall: width always fits the window, height
    (pings) is user-scaled and scrolled -- like a seismic/sonar section viewer
    rather than a generic pannable/zoomable image.
    """

    pixel_clicked = Signal(int, int)
    pixel_hovered = Signal(int, int)
    hover_cleared = Signal()
    bottom_edited = Signal(int, int)

    DEFAULT_ALONG_TRACK_SCALE = 3.0
    PINGS_PER_WHEEL_NOTCH = 40
    BOTTOM_LINE_COLOR = QColor(255, 40, 40, 210)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        # ScrollHandDrag normally shows an open/closed hand; a crosshair is
        # far more useful for pixel-precise contact picking, but Qt's own
        # drag-mode handling reasserts the hand cursor on every press/release
        # (see mousePressEvent/mouseReleaseEvent below), so it has to be
        # re-applied after each one rather than set just once here.
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self._image_size = (0, 0)
        self._press_position = None
        self._pixmap_item = None
        self._marker_items = []
        self._along_track_scale = self.DEFAULT_ALONG_TRACK_SCALE
        # Bottom-edit mode: dragging paints the bottom line instead of
        # panning/picking (see QtContactPickerWindow._apply_interaction_mode,
        # which also switches setDragMode so the two never fight over a drag).
        self.edit_mode = False
        self._last_edit_position = None
        self._overlay_pixmap = None
        self._overlay_item = None

    def set_image(self, rgb: np.ndarray, *, fit: bool = False) -> None:
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        scene = self.scene()
        pixmap = QPixmap.fromImage(image)
        had_content = self._image_size != (0, 0)
        if self._pixmap_item is None:
            self._pixmap_item = scene.addPixmap(pixmap)
            self._pixmap_item.setZValue(-1)
        else:
            self._pixmap_item.setPixmap(pixmap)
        scene.setSceneRect(0, 0, width, height)
        self._image_size = (width, height)
        # A gain/processing refresh replaces the same-sized pixmap in place;
        # keep the processor's current scroll position instead of jumping
        # back to the top. Only the very first load (fit=True) skips this.
        self._apply_transform(preserve_center=had_content and not fit)

    def set_along_track_scale(self, scale: float) -> None:
        """Set vertical pixels-per-ping, independent of the width-fit scale.

        Lets the processor stretch or compress the along-track axis to
        correct for vessel-speed changes between pings, so a contact that is
        actually round on the seabed renders round here instead of stretched
        or squashed.
        """

        self._along_track_scale = max(0.01, float(scale))
        self._apply_transform(preserve_center=True)

    def set_markers(self, markers: list[tuple[int, int, str]]) -> None:
        scene = self.scene()
        for item in self._marker_items:
            scene.removeItem(item)
        self._marker_items = []
        pen = QPen(QColor("cyan"))
        pen.setWidthF(1.5)
        for row, column, label in markers:
            item = scene.addEllipse(
                -6, -6, 12, 12, pen, QBrush(QColor(0, 255, 255, 55))
            )
            # Keep marker circles a fixed, readable screen size regardless of
            # the independent horizontal/vertical waterfall scale below --
            # otherwise they'd stretch into ellipses along with the image.
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPos(column, row)
            item.setZValue(2)
            item.setToolTip(label)
            self._marker_items.append(item)

    def center_on(self, row: int, column: int) -> None:
        self.centerOn(column, row)

    def set_bottom_overlay(self, mask: np.ndarray) -> None:
        """Translucent-red overlay showing the current bottom line, layered
        above the base waterfall and below contact markers. ``mask`` is
        ``bottom_map`` reshaped to ``[num_ping, 2*ping_len]`` (see
        ``logical_bottom_overlay``) -- truthy, not necessarily boolean,
        where the bottom line passes through that pixel. Kept as its own
        graphics item (like ``set_markers``' circles) so a gain/TVG change
        never has to rebuild it and an edit never triggers a gain re-render.
        """
        height, width = mask.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        active = np.asarray(mask) > 0
        rgba[active] = (
            self.BOTTOM_LINE_COLOR.red(),
            self.BOTTOM_LINE_COLOR.green(),
            self.BOTTOM_LINE_COLOR.blue(),
            self.BOTTOM_LINE_COLOR.alpha(),
        )
        image = QImage(
            rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888
        ).copy()
        self._overlay_pixmap = QPixmap.fromImage(image)
        scene = self.scene()
        if self._overlay_item is None:
            self._overlay_item = scene.addPixmap(self._overlay_pixmap)
            self._overlay_item.setZValue(0)
        else:
            self._overlay_item.setPixmap(self._overlay_pixmap)

    def patch_bottom_overlay_row(self, row: int, mask_row: np.ndarray) -> None:
        """Update a single row of the existing bottom-line overlay in
        place -- used while dragging so per-event cost stays independent of
        file length instead of rebuilding the whole-frame overlay on every
        ``mouseMoveEvent``."""
        if self._overlay_pixmap is None or not (0 <= row < self._image_size[1]):
            return
        width = self._image_size[0]
        rgba = np.zeros((1, width, 4), dtype=np.uint8)
        active = np.asarray(mask_row) > 0
        rgba[0, active] = (
            self.BOTTOM_LINE_COLOR.red(),
            self.BOTTOM_LINE_COLOR.green(),
            self.BOTTOM_LINE_COLOR.blue(),
            self.BOTTOM_LINE_COLOR.alpha(),
        )
        strip = QImage(
            rgba.data, width, 1, rgba.strides[0], QImage.Format.Format_RGBA8888
        ).copy()
        painter = QPainter(self._overlay_pixmap)
        # Source (not SourceOver) so pixels the edit just cleared become
        # transparent again instead of leaving a stale red streak blended in.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawImage(0, row, strip)
        painter.end()
        self._overlay_item.setPixmap(self._overlay_pixmap)

    def mousePressEvent(self, event) -> None:
        if self.edit_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_edit_position = None
                self._emit_edit_at(self._event_position(event))
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = self._event_position(event)
        super().mousePressEvent(event)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event) -> None:
        if self.edit_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_edit_position = None
            return
        release_position = self._event_position(event)
        press_position = self._press_position
        self._press_position = None
        super().mouseReleaseEvent(event)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and press_position is not None
            and (release_position - press_position).manhattanLength() <= 4
        ):
            point = self.mapToScene(release_position)
            width, height = self._image_size
            column, row = math.floor(point.x()), math.floor(point.y())
            if 0 <= column < width and 0 <= row < height:
                self.pixel_clicked.emit(row, column)

    def mouseMoveEvent(self, event) -> None:
        if self.edit_mode:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._emit_edit_at(self._event_position(event))
            return
        super().mouseMoveEvent(event)
        self._emit_hover_at(self._event_position(event))

    def _emit_edit_at(self, viewport_position) -> None:
        point = self.mapToScene(viewport_position)
        width, height = self._image_size
        column, row = math.floor(point.x()), math.floor(point.y())
        if not (0 <= column < width and 0 <= row < height):
            return
        if self._last_edit_position is not None:
            last_row, last_column = self._last_edit_position
            if last_row != row:
                # Zero-order hold at the *new* column across any rows a fast
                # drag skipped, matching the reference Napari bottom editor
                # exactly (it holds the destination sample, not a ramp).
                step = 1 if row > last_row else -1
                for held_row in range(last_row + step, row, step):
                    self.bottom_edited.emit(held_row, column)
        self._last_edit_position = (row, column)
        self.bottom_edited.emit(row, column)

    def _emit_hover_at(self, viewport_position) -> None:
        point = self.mapToScene(viewport_position)
        width, height = self._image_size
        column, row = math.floor(point.x()), math.floor(point.y())
        if 0 <= column < width and 0 <= row < height:
            self.pixel_hovered.emit(row, column)
        else:
            self.hover_cleared.emit()

    def leaveEvent(self, event) -> None:
        self.hover_cleared.emit()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        # Move through the survey by a fixed number of pings per notch,
        # rather than zooming -- the along-track slider owns zoom now.
        notches = event.angleDelta().y() / 120.0
        view_pixels = notches * self.PINGS_PER_WHEEL_NOTCH * self._along_track_scale
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(round(scrollbar.value() - view_pixels))
        event.accept()

    @staticmethod
    def _event_position(event):
        return (
            event.position().toPoint()
            if hasattr(event, "position")
            else event.pos()
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_transform(preserve_center=True)

    def _apply_transform(self, *, preserve_center: bool) -> None:
        width, height = self._image_size
        if width <= 0 or height <= 0:
            return
        viewport_width = max(1, self.viewport().width())
        center_row = None
        if preserve_center and not self.scene().sceneRect().isEmpty():
            center_row = self.mapToScene(self.viewport().rect().center()).y()
        self.setTransform(
            QTransform().scale(viewport_width / width, self._along_track_scale)
        )
        if center_row is not None:
            self.centerOn(width / 2, center_row)


class QtContactPickerWindow(QMainWindow):
    def __init__(
        self,
        *,
        context: SonarFileContext,
        store: ContactStore,
        picker: ContactPickerService,
        contacts_db_path: Path,
        display: WaterfallGainModel,
        loader_settings: SonarLoaderSettings,
    ):
        super().__init__()
        self._apply_context(context)
        self.store = store
        self.picker = picker
        self.display = display
        self.loader_settings = loader_settings
        self.contacts_db_path = contacts_db_path
        self._directory_files = sonar_files_in_directory(context.filepath.parent)
        self.thread_pool = QThreadPool.globalInstance()
        self.processing_worker = None
        self.bottom_worker = None
        self._bottom_worker_source: Path | None = None
        self._pending_full_bottom_recalc = False
        self.geotiff_worker = None
        self._restoring_gain_settings = False
        self._pending_restored_gain_settings = None
        self.gain_settings_save_timer = QTimer(self)
        self.gain_settings_save_timer.setSingleShot(True)
        self.gain_settings_save_timer.setInterval(500)
        self.gain_settings_save_timer.timeout.connect(self._save_file_gain_settings)
        self.interaction_modes = InteractionModeController()
        self.interaction_modes.add_listener(self._apply_interaction_mode)
        self.setWindowTitle("SidescanTools - Contact picker (Qt raster)")
        self.resize(1400, 820)

        self.view = WaterfallView()
        self.view.pixel_clicked.connect(self.pick_contact)
        self.view.pixel_hovered.connect(self._show_hover_stats)
        self.view.hover_cleared.connect(self._clear_hover_stats)
        self.view.bottom_edited.connect(self._apply_bottom_edit)
        self.hover_stats_label = QLabel("")
        self.statusBar().addPermanentWidget(self.hover_stats_label)
        self.gain_slider, self.gain_spin = self._gain_control(
            WaterfallGainModel.MIN_OVERALL_GAIN_DB,
            WaterfallGainModel.MAX_OVERALL_GAIN_DB,
            WaterfallGainModel.DEFAULT_OVERALL_GAIN_DB,
        )
        self.tvg_spreading_slider, self.tvg_spreading_spin = self._gain_control(
            WaterfallGainModel.MIN_TVG_SPREADING_DB_PER_DECADE,
            WaterfallGainModel.MAX_TVG_SPREADING_DB_PER_DECADE,
            WaterfallGainModel.DEFAULT_TVG_SPREADING_DB_PER_DECADE,
        )
        self.tvg_absorption_slider, self.tvg_absorption_spin = self._fine_control(
            WaterfallGainModel.MIN_TVG_ABSORPTION_DB_PER_M,
            WaterfallGainModel.MAX_TVG_ABSORPTION_DB_PER_M,
            WaterfallGainModel.DEFAULT_TVG_ABSORPTION_DB_PER_M,
            step=0.01,
            decimals=2,
            suffix=" dB/m",
        )
        self.gain_spin.setToolTip(
            "Display-only gain applied uniformly to both channels"
        )
        self.tvg_spreading_spin.setToolTip(
            "Time-variable-gain spreading-loss term: dB boost per decade of "
            "range beyond nadir. This is the same kind of range compensation "
            "EdgeTech and other sonar hardware apply during acquisition -- "
            "tune by eye until brightness looks even from near to far range."
        )
        self.tvg_absorption_spin.setToolTip(
            "Time-variable-gain absorption term: dB boost per meter of range, "
            "linear. Compensates the frequency-dependent acoustic absorption "
            "component of range loss, on top of the spreading term."
        )
        self.gain_spin.valueChanged.connect(self._overall_gain_changed)
        self.tvg_spreading_spin.valueChanged.connect(self._tvg_spreading_changed)
        self.tvg_absorption_spin.valueChanged.connect(self._tvg_absorption_changed)
        # Connect sliders directly as well as through the editable spin boxes.
        # This avoids backend-specific signal overload issues in QtPy.
        self.gain_slider.valueChanged.connect(self._overall_gain_changed)
        self.tvg_spreading_slider.valueChanged.connect(self._tvg_spreading_changed)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(90)
        self.render_timer.timeout.connect(self.render_gain)
        self.reset_gain_button = QPushButton("Reset TVG")
        self.reset_gain_button.setMinimumHeight(36)
        self.reset_gain_button.clicked.connect(self.reset_gain)
        self.normalize_tvg_button = QPushButton("Auto TVG")
        self.normalize_tvg_button.setMinimumHeight(36)
        self.normalize_tvg_button.setToolTip(
            "Equalize typical brightness across the full swath around the "
            "selected target. "
            "Automatically adjusts overall gain and TVG, then applies a "
            "smooth residual correction to dark and blown-out areas."
        )
        self.normalize_tvg_button.clicked.connect(self.normalize_tvg)
        self.normalize_target_button = QPushButton(
            "Auto TVG Brightness Target: "
            f"{self.display.normalize_target_percent}%…"
        )
        self.normalize_target_button.setMinimumHeight(36)
        self.normalize_target_button.setToolTip(
            "Choose the desired typical waterfall brightness, then normalize "
            "the current swath to that target."
        )
        self.normalize_target_button.clicked.connect(self.set_normalize_target)

        self.along_track_slider, self.along_track_spin = self._along_track_control(
            0.1, 8.0, WaterfallView.DEFAULT_ALONG_TRACK_SCALE
        )
        self.along_track_spin.setToolTip(
            "Vertical pixels per ping along the survey track. The waterfall's "
            "width always fits the window; adjust this to stretch or compress "
            "the ping axis so a contact's true shape isn't distorted by "
            "changes in vessel speed."
        )
        # The spin box is the single source of truth for the real value; the
        # slider only drives it (see _along_track_control), so one connection
        # here is enough to apply every change live -- this is a cheap view
        # transform, not a re-render, so no debounce timer is needed.
        self.along_track_spin.valueChanged.connect(self.view.set_along_track_scale)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.setMinimumHeight(36)
        self.reset_view_button.clicked.connect(self.reset_view)

        instructions = QLabel(
            "Left-click a sonar return to save a contact. Drag to pan; scroll "
            "the wheel to move up/down the survey. Width always fits the "
            "window -- use \"Speed Correction\" to fix contact shapes "
            "distorted by vessel-speed changes."
        )

        open_button = QPushButton("Open…")
        open_button.setToolTip("Open a different sonar file")
        open_button.clicked.connect(self.open_file)
        self.previous_file_button = QPushButton("◀ Previous file")
        self.previous_file_button.clicked.connect(lambda: self._go_to_relative_file(-1))
        self.next_file_button = QPushButton("Next file ▶")
        self.next_file_button.setToolTip(
            "Move to the next .jsf/.xtf file in this folder. Gain, TVG, and "
            "along-track scale carry over; the same contacts database is used."
        )
        self.next_file_button.clicked.connect(lambda: self._go_to_relative_file(1))
        self.file_position_label = QLabel()
        file_nav = QHBoxLayout()
        file_nav.addWidget(open_button)
        file_nav.addSpacing(14)
        file_nav.addWidget(self.previous_file_button)
        file_nav.addWidget(self.next_file_button)
        file_nav.addWidget(self.file_position_label, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(file_nav)
        layout.addWidget(instructions)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(central)

        new_database_button = QPushButton("New Database…")
        new_database_button.setToolTip(
            "Create a new, empty contacts database and switch to it. The "
            "currently open file is registered into it right away, and any "
            "other file navigated to from here will share it too -- use "
            "this to start a project spanning many survey files."
        )
        new_database_button.clicked.connect(self.new_database)
        open_database_button = QPushButton("Open Database…")
        open_database_button.setToolTip(
            "Switch to an existing contacts database, e.g. one you built "
            "earlier for this survey -- files from any folder can share it."
        )
        open_database_button.clicked.connect(self.open_database)
        save_database_as_button = QPushButton("Save Database As…")
        save_database_as_button.setToolTip(
            "Copy the current database, including every contact in it, to "
            "a new file and switch to using that copy."
        )
        save_database_as_button.clicked.connect(self.save_database_as)
        self.database_label = QLabel()
        self.database_label.setWordWrap(True)
        database_row = QHBoxLayout()
        database_row.addWidget(new_database_button)
        database_row.addWidget(open_database_button)
        database_row.addWidget(save_database_as_button)

        self.contact_dock = ContactDock(
            self.store,
            self.source_file_id,
            export_directory=self.contacts_db_path.parent,
        )
        active_geometry_settings = (
            self.context.geometry_settings or self.loader_settings.geometry_settings
        )
        self.contact_dock.set_geometry_status(
            f"Geometry ready · layback "
            f"{active_geometry_settings.effective_layback_m:.1f} m",
            ready=True,
        )

        contacts_panel = QWidget()
        contacts_panel_layout = QVBoxLayout(contacts_panel)
        contacts_panel_layout.setContentsMargins(0, 0, 0, 0)
        contacts_panel_layout.addLayout(database_row)
        contacts_panel_layout.addWidget(self.database_label)
        contacts_panel_layout.addWidget(self._build_layback_group())
        contacts_panel_layout.addWidget(self.contact_dock, 1)
        contacts_panel_layout.addWidget(self._build_geotiff_export_group())

        dock = QDockWidget("Sonar Contacts", self)
        dock.setWidget(contacts_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        processing_dock = QDockWidget("Processing and gain", self)
        processing_dock.setWidget(self._build_processing_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, processing_dock)
        bottom_line_dock = QDockWidget("Bottom Line", self)
        bottom_line_dock.setWidget(self._build_bottom_line_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, bottom_line_dock)
        self.tabifyDockWidget(processing_dock, bottom_line_dock)
        # tabifyDockWidget normally leaves the most recently added dock on
        # top. Processing and Gain is the primary startup workflow.
        processing_dock.raise_()
        self._connect_gain_settings_autosave()
        gain_settings_notice = self._restore_file_gain_settings()
        self.contact_dock.contact_deleted.connect(self.refresh_chunk)
        self.contact_dock.contact_updated.connect(self.refresh_chunk)
        self.contact_dock.table.selectionModel().selectionChanged.connect(
            self.center_selected_contact
        )
        self.view.set_image(self.display.render_rgb(), fit=True)
        self.refresh_chunk()
        self._refresh_bottom_overlay()
        self._apply_interaction_mode(self.interaction_modes.mode)
        self._update_file_position()
        self._update_database_label()
        self._update_status()
        if gain_settings_notice:
            self.statusBar().showMessage(gain_settings_notice, 8000)

    @staticmethod
    def _gain_control(minimum: int, maximum: int, value: float):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(round(value))
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" dB")
        spin.setValue(float(value))
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(lambda current: slider.setValue(round(current)))
        return slider, spin

    @staticmethod
    def _fine_control(
        minimum: float,
        maximum: float,
        value: float,
        *,
        step: float,
        decimals: int,
        suffix: str,
    ):
        # QSlider is integer-only; a fixed-point scale gives the spin box's
        # finer sub-1.0 steps a matching integer slider range.
        fixed_point_scale = 10**decimals
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(round(minimum * fixed_point_scale), round(maximum * fixed_point_scale))
        slider.setSingleStep(max(1, round(step * fixed_point_scale)))
        slider.setValue(round(value * fixed_point_scale))
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        slider.valueChanged.connect(lambda raw: spin.setValue(raw / fixed_point_scale))
        spin.valueChanged.connect(
            lambda current: slider.setValue(round(current * fixed_point_scale))
        )
        return slider, spin

    @classmethod
    def _along_track_control(cls, minimum: float, maximum: float, value: float):
        return cls._fine_control(
            minimum, maximum, value, step=0.05, decimals=2, suffix=" px/ping"
        )

    def reset_view(self) -> None:
        self.along_track_spin.setValue(WaterfallView.DEFAULT_ALONG_TRACK_SCALE)
        self.view.verticalScrollBar().setValue(0)

    def _connect_gain_settings_autosave(self) -> None:
        for spin in (
            self.gain_spin,
            self.tvg_spreading_spin,
            self.tvg_absorption_spin,
            self.along_track_spin,
        ):
            spin.valueChanged.connect(self._schedule_gain_settings_save)
        self.processing_mode.currentIndexChanged.connect(
            self._schedule_gain_settings_save
        )
        self.egn_path.textChanged.connect(self._schedule_gain_settings_save)

    def _schedule_gain_settings_save(self, *args) -> None:
        if not self._restoring_gain_settings:
            self.gain_settings_save_timer.start()

    def _current_file_gain_settings(self) -> SonarGainSettings:
        egn_path = self.egn_path.text().strip()
        return SonarGainSettings(
            source_file=self.filepath.name,
            overall_gain_db=self.display.overall_gain_db,
            tvg_spreading_db_per_decade=(
                self.display.tvg_spreading_db_per_decade
            ),
            tvg_absorption_db_per_m=self.display.tvg_absorption_db_per_m,
            auto_tvg_brightness_target_percent=(
                self.display.normalize_target_percent
            ),
            auto_tvg_active=self.display.auto_tvg_active,
            auto_tvg_gain_db=tuple(
                round(value, 6) for value in self.display.auto_tvg_gain_db
            ),
            speed_correction_px_per_ping=self.along_track_spin.value(),
            processing_mode=str(self.processing_mode.currentData()),
            egn_table_path=portable_egn_table_path(egn_path, self.filepath),
            destripe_active=self.destripe_button.isChecked(),
            slant_range_correction_active=(
                self.slant_range_checkbox.isChecked()
            ),
            layback_override_m=getattr(self, "_layback_override_m", None),
        )

    def _save_file_gain_settings(self) -> bool:
        if self._restoring_gain_settings:
            return False
        try:
            save_gain_settings(self.filepath, self._current_file_gain_settings())
        except Exception as exc:
            self.statusBar().showMessage(
                f"Could not save TVG gain settings: {exc}", 8000
            )
            return False
        return True

    def _flush_pending_gain_settings_save(self) -> None:
        if self.gain_settings_save_timer.isActive():
            self.gain_settings_save_timer.stop()
            self._save_file_gain_settings()

    def _reset_file_gain_settings(self) -> None:
        self.display.clear_normalization()
        self.display.set_normalize_target_percent(
            WaterfallGainModel.DEFAULT_NORMALIZE_TARGET_PERCENT
        )
        self.gain_spin.setValue(WaterfallGainModel.DEFAULT_OVERALL_GAIN_DB)
        self.tvg_spreading_spin.setValue(
            WaterfallGainModel.DEFAULT_TVG_SPREADING_DB_PER_DECADE
        )
        self.tvg_absorption_spin.setValue(
            WaterfallGainModel.DEFAULT_TVG_ABSORPTION_DB_PER_M
        )
        self.along_track_spin.setValue(WaterfallView.DEFAULT_ALONG_TRACK_SCALE)
        self.normalize_target_button.setText(
            "Auto TVG Brightness Target: "
            f"{self.display.normalize_target_percent}%…"
        )
        self.egn_path.clear()
        self.destripe_button.setChecked(False)
        self.slant_range_checkbox.setChecked(False)
        context = getattr(self, "context", None)
        self._layback_override_m = getattr(context, "layback_override_m", None)
        if hasattr(self, "_update_layback_controls"):
            self._update_layback_controls()
        raw_index = self.processing_mode.findData(BuiltInGainMode.RAW.value)
        self.processing_mode.setCurrentIndex(raw_index)

    def _apply_file_gain_settings(
        self,
        settings: SonarGainSettings,
        *,
        restore_auto_tvg: bool,
    ) -> None:
        self.display.clear_normalization()
        self.gain_spin.setValue(settings.overall_gain_db)
        self.tvg_spreading_spin.setValue(settings.tvg_spreading_db_per_decade)
        self.tvg_absorption_spin.setValue(settings.tvg_absorption_db_per_m)
        self.along_track_spin.setValue(settings.speed_correction_px_per_ping)
        self.display.set_normalize_target_percent(
            settings.auto_tvg_brightness_target_percent
        )
        self.normalize_target_button.setText(
            "Auto TVG Brightness Target: "
            f"{self.display.normalize_target_percent}%…"
        )
        egn_path = resolve_egn_table_path(settings, self.filepath)
        self.egn_path.setText(str(egn_path) if egn_path is not None else "")
        self.destripe_button.setChecked(settings.destripe_active)
        self.slant_range_checkbox.setChecked(
            settings.slant_range_correction_active
        )
        self._layback_override_m = settings.layback_override_m
        if hasattr(self, "_update_layback_controls"):
            self._update_layback_controls()
        mode_index = self.processing_mode.findData(settings.processing_mode)
        if mode_index < 0:
            raise ValueError(
                f"unsupported saved processing mode: {settings.processing_mode}"
            )
        self.processing_mode.setCurrentIndex(mode_index)
        if settings.auto_tvg_active and restore_auto_tvg:
            self.display.restore_auto_tvg_gain(settings.auto_tvg_gain_db)

    def _restore_file_gain_settings(self) -> str | None:
        settings_path = gain_settings_path(self.filepath)
        self._pending_restored_gain_settings = None
        try:
            settings = load_gain_settings(self.filepath)
        except Exception as exc:
            self._restoring_gain_settings = True
            try:
                self._reset_file_gain_settings()
            finally:
                self._restoring_gain_settings = False
            return f"Could not load {settings_path.name}: {exc}"

        self._restoring_gain_settings = True
        try:
            if settings is None:
                self._reset_file_gain_settings()
            else:
                if settings.source_file != self.filepath.name:
                    raise ValueError(
                        "saved source filename does not match the sonar file"
                    )
                needs_processing = (
                    settings.processing_mode == BuiltInGainMode.EGN.value
                    or settings.destripe_active
                    or settings.slant_range_correction_active
                )
                self._apply_file_gain_settings(
                    settings, restore_auto_tvg=not needs_processing
                )
                if needs_processing:
                    self._pending_restored_gain_settings = settings
        except Exception as exc:
            self._reset_file_gain_settings()
            return f"Could not apply {settings_path.name}: {exc}"
        finally:
            self._restoring_gain_settings = False

        if settings is None:
            if not self._save_file_gain_settings():
                return f"Could not create {settings_path.name}"
        elif (
            settings.processing_mode == BuiltInGainMode.EGN.value
            or settings.destripe_active
            or settings.slant_range_correction_active
        ):
            QTimer.singleShot(0, self.apply_builtin_processing)
        return None

    def _apply_context(self, context: SonarFileContext) -> None:
        # Kept in full (not just the fields picked out below) so a database
        # switch can later re-register this same already-loaded file into a
        # different store via _register_in_store(), without re-parsing it.
        self.context = context
        self.filepath = context.filepath
        self.sidescan_file = context.sidescan_file
        self.preprocessor = context.preprocessor
        self.raw_waterfall = context.raw_waterfall
        self.built_in_processor = context.built_in_processor
        self.source_file_id = context.source_file_id
        self._layback_override_m = context.layback_override_m

    def open_file(self) -> None:
        start_dir = str(self.filepath.parent)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open sonar file",
            start_dir,
            "Sidescan files (*.jsf *.xtf);;All files (*)",
        )
        if filename:
            self.load_file(Path(filename))

    def _go_to_relative_file(self, offset: int) -> None:
        if not self._directory_files:
            return
        try:
            current_index = self._directory_files.index(self.filepath)
        except ValueError:
            # The current file was opened from outside this folder's own
            # listing (e.g. via Open... on a file in a different directory,
            # or the file was renamed/moved since navigation was built).
            current_index = 0
        new_index = current_index + offset
        if not 0 <= new_index < len(self._directory_files):
            return
        self.load_file(self._directory_files[new_index])

    def _update_file_position(self) -> None:
        try:
            index = self._directory_files.index(self.filepath)
        except ValueError:
            self.file_position_label.setText(self.filepath.name)
            self.previous_file_button.setEnabled(False)
            self.next_file_button.setEnabled(False)
            return
        total = len(self._directory_files)
        self.file_position_label.setText(
            f"File {index + 1} of {total}: {self.filepath.name}"
        )
        self.previous_file_button.setEnabled(index > 0)
        self.next_file_button.setEnabled(index < total - 1)

    def load_file(self, filepath: Path) -> None:
        """Switch the whole window to a different sonar file in place.

        Gain, TVG, Speed Correction, and processing mode are saved for the
        old file and restored from the new file's sidecar. The same
        ContactStore/database is reused, so contacts picked across every file
        in a folder land in one shared project.
        """

        filepath = Path(filepath)
        self.bottom_recalc_timer.stop()
        self._pending_full_bottom_recalc = False
        # The current file's preprocessor/sidescan_file are about to be
        # replaced below -- an edit sitting in the autosave debounce would
        # otherwise never get written.
        self._flush_pending_bottom_line_save()
        self._flush_pending_gain_settings_save()
        self.statusBar().showMessage(f"Loading {filepath.name}…")
        QApplication.processEvents()
        try:
            context = _load_sonar_context(
                filepath, settings=self.loader_settings, store=self.store
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Could not open {filepath.name}: {exc}", 8000)
            return

        self._apply_context(context)
        # Compare resolved paths (context.filepath, like every entry in
        # _directory_files) rather than the possibly-relative local
        # `filepath` -- otherwise this looks like a folder change, and
        # re-scans, on every single call.
        if (
            not self._directory_files
            or context.filepath.parent != self._directory_files[0].parent
        ):
            self._directory_files = sonar_files_in_directory(context.filepath.parent)
        self.picker = _build_contact_picker(context, store=self.store, display=self.display)
        self.display.set_source(
            context.raw_waterfall,
            base_pipeline="qt-continuous-waterfall-v1|raw",
            slant_range_m=context.slant_range_m,
        )
        self.processing_mode.setCurrentIndex(0)
        self.processing_progress.setValue(0)
        self.processing_status.setText("Raw display")
        gain_settings_notice = self._restore_file_gain_settings()

        self.setWindowTitle(f"SidescanTools - Contact picker (Qt raster) — {filepath.name}")
        self.contact_dock.set_source_file(context.source_file_id)
        active_geometry_settings = (
            context.geometry_settings or self.loader_settings.geometry_settings
        )
        self.contact_dock.set_geometry_status(
            f"Geometry ready · layback "
            f"{active_geometry_settings.effective_layback_m:.1f} m",
            ready=True,
        )
        self._update_layback_controls()
        # Editing is per-file state; switching files with it still on risks
        # painting corrections onto the wrong survey line.
        self.edit_bottom_button.setChecked(False)
        self.bottom_status_label.setText(context.bottom_info_status)
        self.refine_altitude_button.setEnabled(
            bool(np.max(self.sidescan_file.sensor_primary_altitude) > 0)
        )
        self.view.set_image(self.display.render_rgb(), fit=True)
        self.refresh_chunk()
        self._refresh_bottom_overlay()
        self._update_file_position()
        self._update_status(f"Opened {filepath.name}")
        if gain_settings_notice:
            self.statusBar().showMessage(gain_settings_notice, 8000)

    def _build_layback_group(self) -> QGroupBox:
        group = QGroupBox("Towfish Layback")
        self._emphasize_sidebar_group(group)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.recorded_layback_label = QLabel()
        self.recorded_cable_out_label = QLabel()
        form.addRow("File layback", self.recorded_layback_label)
        form.addRow("File cable out", self.recorded_cable_out_label)
        self.layback_spin = QDoubleSpinBox()
        self.layback_spin.setRange(0.0, 20_000.0)
        self.layback_spin.setDecimals(1)
        self.layback_spin.setSingleStep(1.0)
        self.layback_spin.setSuffix(" m")
        self.layback_spin.setToolTip(
            "Layback used to move the towfish position behind the navigation "
            "track. Apply stores a per-file manual override."
        )
        form.addRow("Layback used", self.layback_spin)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.apply_layback_button = QPushButton("Apply Layback")
        self.apply_layback_button.clicked.connect(self.apply_manual_layback)
        self.use_file_layback_button = QPushButton("Use File Value")
        self.use_file_layback_button.clicked.connect(self.use_file_layback)
        buttons.addWidget(self.apply_layback_button)
        buttons.addWidget(self.use_file_layback_button)
        layout.addLayout(buttons)
        self.layback_status_label = QLabel()
        self.layback_status_label.setWordWrap(True)
        layout.addWidget(self.layback_status_label)
        self._update_layback_controls()
        return group

    @staticmethod
    def _tow_value_text(value: float | None) -> str:
        return "Not recorded" if value is None else f"{value:.1f} m"

    def _update_layback_controls(self) -> None:
        if not hasattr(self, "layback_spin"):
            return
        self.recorded_layback_label.setText(
            self._tow_value_text(self.context.tow_data.recorded_layback_m)
        )
        self.recorded_cable_out_label.setText(
            self._tow_value_text(self.context.tow_data.recorded_cable_out_m)
        )
        self.layback_spin.blockSignals(True)
        active_geometry_settings = (
            self.context.geometry_settings or self.loader_settings.geometry_settings
        )
        self.layback_spin.setValue(active_geometry_settings.effective_layback_m)
        self.layback_spin.blockSignals(False)
        self.layback_status_label.setText(self.context.layback_source)
        self.use_file_layback_button.setEnabled(
            self._layback_override_m is not None
        )

    def apply_manual_layback(self) -> None:
        self._rebuild_geometry_for_layback(self.layback_spin.value())

    def use_file_layback(self) -> None:
        self._rebuild_geometry_for_layback(None)

    def _rebuild_geometry_for_layback(
        self, manual_layback_m: float | None
    ) -> None:
        geometry_settings, source = resolve_geometry_layback(
            self.loader_settings.geometry_settings,
            self.context.tow_data,
            manual_layback_m=manual_layback_m,
        )
        self.apply_layback_button.setEnabled(False)
        self.use_file_layback_button.setEnabled(False)
        self.layback_status_label.setText("Recalculating contact and mosaic geometry…")
        QApplication.processEvents()
        try:
            self.contact_dock.flush_pending_edit()
            geometry = _prepare_file_geometry(
                self.filepath,
                self.sidescan_file,
                geometry_settings,
                self.loader_settings.output_directory,
            )
            profile_id = self.store.get_or_create_geometry_profile(geometry_settings)
            self.store.mark_stale_for_profile(self.source_file_id, profile_id)
            updated_context = replace(
                self.context,
                geometry=geometry,
                geometry_profile_id=profile_id,
                geometry_settings=geometry_settings,
                layback_source=source,
                layback_override_m=manual_layback_m,
            )
            self._apply_context(updated_context)
            self.picker = _build_contact_picker(
                updated_context, store=self.store, display=self.display
            )
            failures = 0
            for record in self.store.list_contacts(
                source_file_id=self.source_file_id
            ):
                try:
                    coordinate = self.picker.coordinate_for_anchor(
                        record.draft.anchor
                    )
                    self.store.recompute_contact(record.id, coordinate)
                except Exception as exc:
                    self.store.record_recompute_error(record.id, str(exc))
                    failures += 1
            self._save_file_gain_settings()
            self.contact_dock.refresh()
            self.contact_dock.set_geometry_status(
                f"Geometry ready · layback "
                f"{geometry_settings.effective_layback_m:.1f} m",
                ready=True,
            )
            self._update_layback_controls()
            self.refresh_chunk()
            message = "Layback geometry updated"
            if failures:
                message += f"; {failures} contact(s) could not be recomputed"
            self.statusBar().showMessage(message, 8000)
        except Exception as exc:
            self.layback_status_label.setText(f"Could not apply layback: {exc}")
            self._update_layback_controls()
        finally:
            self.apply_layback_button.setEnabled(True)
            self.use_file_layback_button.setEnabled(
                self._layback_override_m is not None
            )

    def _build_geotiff_export_group(self) -> QGroupBox:
        group = QGroupBox("GeoTIFF Export")
        self._emphasize_sidebar_group(group)
        layout = QVBoxLayout(group)
        crs_row = QFormLayout()
        self.geotiff_crs = QComboBox()
        self.geotiff_crs.addItem("WGS 84 (EPSG:4326)", 4326)
        self.geotiff_crs.addItem("Web Mercator (EPSG:3857)", 3857)
        self.geotiff_crs.setToolTip(
            "Choose the coordinate reference system embedded in the output raster."
        )
        crs_row.addRow("Output CRS", self.geotiff_crs)
        layout.addLayout(crs_row)

        self.export_current_geotiff_button = QPushButton("Export Current File")
        self.export_current_geotiff_button.setMinimumHeight(36)
        self.export_current_geotiff_button.setToolTip(
            "Export the open waterfall beside its source sonar file, using "
            "the currently displayed processing, TVG, and colors."
        )
        self.export_current_geotiff_button.clicked.connect(
            self.export_current_geotiff
        )
        self.export_directory_geotiff_button = QPushButton(
            "Batch Export Directory…"
        )
        self.export_directory_geotiff_button.setMinimumHeight(36)
        self.export_directory_geotiff_button.setToolTip(
            "Export every .jsf/.xtf file in one folder using each file's own "
            ".tvg_gain.cfg sidecar."
        )
        self.export_directory_geotiff_button.clicked.connect(
            self.export_geotiff_directory
        )
        layout.addWidget(self.export_current_geotiff_button)
        layout.addWidget(self.export_directory_geotiff_button)

        self.geotiff_progress = QProgressBar()
        self.geotiff_progress.setRange(0, 100)
        self.geotiff_progress.setValue(0)
        self.geotiff_status = QLabel(
            "Outputs are saved beside each source file as <basename>.tif."
        )
        self.geotiff_status.setWordWrap(True)
        layout.addWidget(self.geotiff_progress)
        layout.addWidget(self.geotiff_status)
        return group

    def export_current_geotiff(self) -> None:
        if self.processing_worker is not None:
            QMessageBox.warning(
                self,
                "Processing still running",
                "Wait for waterfall processing to finish before exporting.",
            )
            return
        # Write even if the debounce timer is idle: the file on disk becomes
        # the durable record of the exact display being exported.
        self.gain_settings_save_timer.stop()
        if not self._save_file_gain_settings():
            QMessageBox.warning(
                self,
                "Could not save settings",
                "GeoTIFF export was not started because the TVG settings "
                "sidecar could not be saved.",
            )
            return
        destination = geotiff_output_path(self.filepath)
        overwrite = self._confirm_geotiff_overwrite([destination])
        if overwrite is None:
            return
        prepared = PreparedSonarExport(
            rgb=np.array(self.display.render_rgb(), copy=True),
            geometry_by_channel=dict(self.context.geometry),
            pipeline_description=self.display.pipeline_description,
        )
        self._start_geotiff_export(
            [self.filepath],
            overwrite=overwrite,
            prepared_current=prepared,
        )

    def export_geotiff_directory(self) -> None:
        # If the open file is part of this batch, do not let a pending slider
        # edit miss the sidecar snapshot the worker is about to load.
        self._flush_pending_gain_settings_save()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose sonar directory to batch export",
            str(self.filepath.parent),
        )
        if not directory:
            return
        files = sonar_files_in_directory(directory)
        if not files:
            QMessageBox.information(
                self,
                "No sonar files",
                "The selected directory contains no .jsf or .xtf files.",
            )
            return

        destinations: dict[str, list[Path]] = {}
        for source in files:
            output = geotiff_output_path(source)
            destinations.setdefault(output.name.casefold(), []).append(source)
        collisions = [sources for sources in destinations.values() if len(sources) > 1]
        if collisions:
            names = ", ".join(source.name for source in collisions[0])
            QMessageBox.warning(
                self,
                "Duplicate output basename",
                f"These files would write the same GeoTIFF name: {names}",
            )
            return

        overwrite = self._confirm_geotiff_overwrite(
            [geotiff_output_path(source) for source in files]
        )
        if overwrite is None:
            return
        self._start_geotiff_export(files, overwrite=overwrite)

    def _confirm_geotiff_overwrite(self, destinations: list[Path]) -> bool | None:
        existing = [path for path in destinations if path.exists()]
        if not existing:
            return False
        noun = existing[0].name if len(existing) == 1 else f"{len(existing)} GeoTIFFs"
        answer = QMessageBox.question(
            self,
            "Replace existing GeoTIFF?",
            f"{noun} already exist. Replace the existing output?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return True if answer == QMessageBox.StandardButton.Yes else None

    def _start_geotiff_export(
        self,
        files: list[Path],
        *,
        overwrite: bool,
        prepared_current: PreparedSonarExport | None = None,
    ) -> None:
        if self.geotiff_worker is not None:
            QMessageBox.information(
                self, "Export in progress", "A GeoTIFF export is already running."
            )
            return
        self.export_current_geotiff_button.setEnabled(False)
        self.export_directory_geotiff_button.setEnabled(False)
        self.geotiff_crs.setEnabled(False)
        self.geotiff_progress.setValue(0)
        self.geotiff_status.setText(
            f"Starting export of {len(files)} sonar file(s)…"
        )
        worker = GeoTiffExportWorker(
            files,
            epsg=int(self.geotiff_crs.currentData()),
            loader_settings=self.loader_settings,
            overwrite=overwrite,
            prepared_current=prepared_current,
        )
        self.geotiff_worker = worker
        worker.signals.progress.connect(self._geotiff_export_progressed)
        worker.signals.finished.connect(self._geotiff_export_finished)
        self.thread_pool.start(worker)

    def _geotiff_export_progressed(self, percent: int, message: str) -> None:
        self.geotiff_progress.setValue(percent)
        self.geotiff_status.setText(message)

    def _geotiff_export_finished(self, results: list, failures: list) -> None:
        self.export_current_geotiff_button.setEnabled(True)
        self.export_directory_geotiff_button.setEnabled(True)
        self.geotiff_crs.setEnabled(True)
        self.geotiff_worker = None
        if results:
            self.geotiff_progress.setValue(100)
        default_count = sum(result.used_default_settings for result in results)
        message = f"Exported {len(results)} GeoTIFF(s)"
        if default_count:
            message += f"; {default_count} used default settings (no sidecar found)"
        if failures:
            message += f"; {len(failures)} failed"
        self.geotiff_status.setText(message)
        self.statusBar().showMessage(message, 10000)
        if failures:
            details = "\n".join(
                f"{path.name}: {error}" for path, error in failures[:8]
            )
            if len(failures) > 8:
                details += f"\n…and {len(failures) - 8} more"
            QMessageBox.warning(self, "GeoTIFF export incomplete", details)

    def _update_database_label(self) -> None:
        self.database_label.setText(f"Database: {self.contacts_db_path.name}")
        self.database_label.setToolTip(str(self.contacts_db_path))

    def _confirm_replace_existing_database(self, path: Path) -> bool:
        """True if it's OK to (re)create an empty database at `path` --
        either nothing is there yet, or the user explicitly accepted
        wiping whatever survey data already is."""

        if not path.exists():
            return True
        answer = QMessageBox.question(
            self,
            "Replace existing database?",
            f"{path.name} already exists and may hold contacts from a "
            "previous survey.\n\nReplace it with a new, empty database? "
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _switch_database(self, path: Path, *, store_factory) -> None:
        """Point the whole window (store, dock, picker) at a different
        database file, re-registering the currently loaded sonar file into
        it so it's immediately available there too.

        ``store_factory()`` must return a ready ``ContactStore`` open on
        `path`. The previous store is only closed after that succeeds, so a
        failed switch (a bad file, an unreadable schema, ...) leaves the
        window on the database that was already working.
        """

        try:
            new_store = store_factory()
        except Exception as exc:
            QMessageBox.critical(self, "Could not open database", str(exc))
            return
        try:
            context = _register_in_store(
                self.context, settings=self.loader_settings, store=new_store
            )
        except Exception as exc:
            new_store.close()
            QMessageBox.critical(self, "Could not open database", str(exc))
            return

        # The dock's own set_store() must flush any pending edit through
        # self.store while it's still the OLD store, so it must run before
        # that connection is closed -- not after, like every other field
        # updated below. (new_database() may have already closed self.store
        # and discarded any pending edit itself, when overwriting the file
        # it's currently open on -- closing an already-closed connection is
        # a documented no-op, so this stays safe either way.)
        old_store = self.store
        self.contact_dock.set_store(new_store, context.source_file_id)
        old_store.close()

        self.store = new_store
        self.contacts_db_path = path
        self._apply_context(context)
        self.picker = _build_contact_picker(context, store=self.store, display=self.display)
        self.contact_dock.export_directory = path.parent
        self.refresh_chunk()
        self._update_database_label()
        self._update_status(f"Switched to database {path.name}")

    def new_database(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "New contacts database",
            str(self.contacts_db_path.parent),
            "SQLite database (*.sqlite)",
        )
        if not filename:
            return
        path = Path(filename)
        if not self._confirm_replace_existing_database(path):
            return

        if path.resolve() == self.contacts_db_path.resolve():
            # Overwriting the database currently open: close it before the
            # factory below deletes its file. Windows will not safely allow
            # deleting a file this process still has an open handle on --
            # the failure mode isn't a clean exception but a native crash,
            # so this has to happen up front, not left to _switch_database's
            # normal old-store handling (which runs after the factory).
            # Nothing is worth flushing into a database about to be wiped.
            self.contact_dock.discard_pending_edit()
            self.store.close()

        def factory():
            if path.exists():
                path.unlink()
            return ContactStore(path)

        self._switch_database(path, store_factory=factory)

    def open_database(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open contacts database",
            str(self.contacts_db_path.parent),
            "SQLite database (*.sqlite *.db);;All files (*)",
        )
        if not filename:
            return
        self._switch_database(Path(filename), store_factory=lambda: ContactStore(filename))

    def save_database_as(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save database as",
            str(self.contacts_db_path.parent),
            "SQLite database (*.sqlite)",
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.resolve() == self.contacts_db_path.resolve():
            self._update_status("Already using this database")
            return
        if not self._confirm_replace_existing_database(destination):
            return

        # self.store is deliberately left open through the copy: every write
        # already commits (and releases SQLite's write lock) inside its own
        # "with self.connection:" block, so there's never a transaction
        # in progress here to make an idle-connection copy unsafe -- and
        # closing it early would pull the connection out from under
        # self.contact_dock (the same ContactStore object) before
        # _switch_database gets a chance to flush a pending edit through it.
        try:
            shutil.copy2(self.contacts_db_path, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save database", str(exc))
            return
        self._switch_database(
            destination, store_factory=lambda: ContactStore(destination)
        )

    def _build_processing_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        gain_group = QGroupBox("Gain & TVG")
        self._emphasize_sidebar_group(gain_group)
        gain_layout = QVBoxLayout(gain_group)
        gain_layout.addWidget(
            self._sidebar_gain_control("Overall gain", self.gain_slider, self.gain_spin)
        )
        gain_layout.addWidget(
            self._sidebar_gain_control(
                "TVG spreading", self.tvg_spreading_slider, self.tvg_spreading_spin
            )
        )
        gain_layout.addWidget(
            self._sidebar_gain_control(
                "TVG absorption", self.tvg_absorption_slider, self.tvg_absorption_spin
            )
        )
        gain_layout.addWidget(self.normalize_tvg_button)
        gain_layout.addWidget(self.normalize_target_button)
        gain_layout.addWidget(self.reset_gain_button)
        layout.addWidget(gain_group)

        view_group = QGroupBox("View")
        self._emphasize_sidebar_group(view_group)
        view_layout = QVBoxLayout(view_group)
        view_layout.addWidget(
            self._sidebar_gain_control(
                "Speed Correction", self.along_track_slider, self.along_track_spin
            )
        )
        view_layout.addWidget(self.reset_view_button)
        layout.addWidget(view_group)

        slant_group = QGroupBox("Slant Range Correction")
        self._emphasize_sidebar_group(slant_group)
        slant_layout = QVBoxLayout(slant_group)
        self.slant_range_checkbox = QCheckBox("Apply Slant Range Correction")
        self.slant_range_checkbox.setMinimumHeight(30)
        self.slant_range_checkbox.setToolTip(
            "Remove the water column and project each side onto ground range, "
            "using the saved bottom-tracking line as the new nadir. This "
            "setting also applies to GeoTIFF exports."
        )
        self.slant_range_checkbox.clicked.connect(self.apply_builtin_processing)
        slant_layout.addWidget(self.slant_range_checkbox)
        layout.addWidget(slant_group)

        destripe_group = QGroupBox("Destripe Filter")
        self._emphasize_sidebar_group(destripe_group)
        destripe_layout = QVBoxLayout(destripe_group)
        self.destripe_button = QPushButton("Apply Destripe Filter")
        self.destripe_button.setCheckable(True)
        self.destripe_button.setMinimumHeight(36)
        self.destripe_button.setToolTip(
            "Suppress horizontal brightness stripes caused by small changes "
            "in towfish roll. Checked means the filter is active for this "
            "sonar file and for its GeoTIFF export."
        )
        self.destripe_button.clicked.connect(self.apply_builtin_processing)
        destripe_layout.addWidget(self.destripe_button)
        layout.addWidget(destripe_group)

        egn_group = QGroupBox("EGN Settings & Options")
        self._emphasize_sidebar_group(egn_group)
        egn_layout = QVBoxLayout(egn_group)
        form = QFormLayout()

        self.processing_mode = QComboBox()
        for label, mode in (
            ("Raw waterfall", BuiltInGainMode.RAW),
            ("Empirical Gain Normalization (EGN)", BuiltInGainMode.EGN),
        ):
            self.processing_mode.addItem(label, mode.value)
        self.processing_mode.currentIndexChanged.connect(
            self._processing_mode_changed
        )
        form.addRow("Processing", self.processing_mode)

        self.egn_path = QLineEdit()
        self.egn_path.setPlaceholderText("Select an EGN .npz table")
        browse_egn = QPushButton("Browse…")
        browse_egn.clicked.connect(self.browse_egn_table)
        build_egn = QPushButton("Build…")
        build_egn.setToolTip(
            "Build a new EGN table from sonar files or a whole folder on disk"
        )
        build_egn.clicked.connect(self.open_egn_table_builder)
        egn_row = QHBoxLayout()
        egn_row.addWidget(self.egn_path, 1)
        egn_row.addWidget(browse_egn)
        egn_row.addWidget(build_egn)
        form.addRow("EGN table", egn_row)
        self.egn_browse_button = browse_egn
        self.egn_build_button = build_egn
        egn_layout.addLayout(form)

        processing_buttons = QHBoxLayout()
        self.apply_processing_button = QPushButton("Apply")
        self.apply_processing_button.clicked.connect(self.apply_builtin_processing)
        reset_button = QPushButton("Show raw")
        reset_button.clicked.connect(self.show_raw_waterfall)
        processing_buttons.addWidget(self.apply_processing_button)
        processing_buttons.addWidget(reset_button)
        egn_layout.addLayout(processing_buttons)

        self.processing_progress = QProgressBar()
        self.processing_progress.setRange(0, 100)
        self.processing_progress.setValue(0)
        self.processing_status = QLabel("Raw display")
        self.processing_status.setWordWrap(True)
        egn_layout.addWidget(self.processing_progress)
        egn_layout.addWidget(self.processing_status)
        layout.addWidget(egn_group)
        layout.addStretch(1)
        self._processing_mode_changed()
        return panel

    @staticmethod
    def _sidebar_gain_control(
        label: str, slider: QSlider, spin: QDoubleSpinBox
    ) -> QWidget:
        control = QWidget()
        layout = QVBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setMinimumHeight(36)
        spin.setMinimumWidth(90)
        minus = QToolButton()
        minus.setText("−")
        plus = QToolButton()
        plus.setText("+")
        for button in (minus, plus):
            button.setFixedSize(36, 36)
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(400)
            button.setAutoRepeatInterval(70)
        minus.clicked.connect(lambda: spin.stepBy(-1))
        plus.clicked.connect(lambda: spin.stepBy(1))

        value_row = QHBoxLayout()
        value_row.addWidget(QLabel(label))
        value_row.addStretch(1)
        value_row.addWidget(minus)
        value_row.addWidget(spin)
        value_row.addWidget(plus)
        layout.addLayout(value_row)
        layout.addWidget(slider)
        return control

    @staticmethod
    def _emphasize_sidebar_group(group: QGroupBox) -> None:
        group.setStyleSheet(
            "QGroupBox {"
            " border: 2px solid #202020;"
            " border-radius: 4px;"
            " margin-top: 10px;"
            " padding-top: 8px;"
            " font-weight: 600;"
            "}"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " left: 8px;"
            " padding: 0 4px;"
            "}"
        )

    def _processing_mode_changed(self, *args) -> None:
        mode = BuiltInGainMode(self.processing_mode.currentData())
        is_egn = mode is BuiltInGainMode.EGN
        self.egn_path.setEnabled(is_egn)
        self.egn_browse_button.setEnabled(is_egn)

    def open_egn_table_builder(self) -> None:
        dialog = EGNTableBuilderDialog(self, initial_directory=self.filepath.parent)
        dialog.exec()
        if dialog.result_table_path is not None:
            self.egn_path.setText(str(dialog.result_table_path))

    def browse_egn_table(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select EGN table",
            str(self.filepath.parent),
            "NumPy tables (*.npz)",
        )
        if not filename:
            return
        self.egn_path.setText(filename)

    def apply_builtin_processing(self) -> None:
        if self.processing_worker is not None:
            return
        try:
            mode = BuiltInGainMode(self.processing_mode.currentData())
            egn_path = (
                Path(self.egn_path.text().strip())
                if self.egn_path.text().strip()
                else None
            )
            request = BuiltInGainRequest(
                mode=mode,
                egn_table_path=egn_path,
                nadir_angle=egn_table_nadir_angle(egn_path),
                destripe=self.destripe_button.isChecked(),
                slant_range_correction=(
                    self.slant_range_checkbox.isChecked()
                ),
            )
        except Exception as exc:
            self.processing_status.setText(str(exc))
            return

        self.apply_processing_button.setEnabled(False)
        self.destripe_button.setEnabled(False)
        self.slant_range_checkbox.setEnabled(False)
        self.processing_progress.setValue(0)
        self.processing_status.setText("Starting processing…")
        worker = GainProcessingWorker(self.built_in_processor, request)
        self.processing_worker = worker
        worker.signals.progress.connect(self._processing_progressed)
        worker.signals.finished.connect(self._processing_finished)
        worker.signals.failed.connect(self._processing_failed)
        self.thread_pool.start(worker)

    def _processing_progressed(self, percent: int, message: str) -> None:
        self.processing_progress.setValue(percent)
        self.processing_status.setText(message)

    def _processing_finished(self, result) -> None:
        self.display.set_source(
            result.display_data,
            base_pipeline=result.pipeline_description,
        )
        restore_warning = None
        if self._pending_restored_gain_settings is not None:
            settings = self._pending_restored_gain_settings
            self._pending_restored_gain_settings = None
            self._restoring_gain_settings = True
            try:
                self._apply_file_gain_settings(settings, restore_auto_tvg=True)
            except ValueError as exc:
                restore_warning = (
                    f"Processing ready; saved Auto TVG was not restored: {exc}"
                )
            finally:
                self._restoring_gain_settings = False
        self.view.set_image(self.display.render_rgb())
        self._refresh_bottom_overlay()
        self.processing_progress.setValue(100)
        self.processing_status.setText(
            restore_warning
            or "Active: " + result.pipeline_description.replace("|", " · ")
        )
        self.apply_processing_button.setEnabled(True)
        self.destripe_button.setEnabled(True)
        self.slant_range_checkbox.setEnabled(True)
        self.processing_worker = None
        self._schedule_gain_settings_save()
        self._update_status()

    def _processing_failed(self, message: str) -> None:
        self._pending_restored_gain_settings = None
        self.processing_status.setText(f"Processing failed: {message}")
        self.apply_processing_button.setEnabled(True)
        self.destripe_button.setEnabled(True)
        self.slant_range_checkbox.setEnabled(True)
        self.processing_worker = None

    def show_raw_waterfall(self) -> None:
        self._pending_restored_gain_settings = None
        self.display.set_source(
            self.raw_waterfall,
            base_pipeline="qt-continuous-waterfall-v1|raw",
        )
        self.destripe_button.setChecked(False)
        self.slant_range_checkbox.setChecked(False)
        self.processing_mode.setCurrentIndex(0)
        self.processing_progress.setValue(0)
        self.processing_status.setText("Raw display")
        self.view.set_image(self.display.render_rgb())
        self._refresh_bottom_overlay()
        self._schedule_gain_settings_save()
        self._update_status()

    def _build_bottom_line_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()

        self.bottom_threshold_slider, self.bottom_threshold_spin = self._fine_control(
            0.0,
            1.0,
            self.loader_settings.default_threshold,
            step=0.01,
            decimals=2,
            suffix="",
        )
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.bottom_threshold_slider, 1)
        threshold_row.addWidget(self.bottom_threshold_spin)
        form.addRow("Threshold", threshold_row)

        self.bottom_strategy_combo = QComboBox()
        self.bottom_strategy_combo.addItems(self.preprocessor.bottom_strategy_choices)
        self.bottom_strategy_combo.setCurrentText(
            self.preprocessor.bottom_strategy_choices[1]
        )
        form.addRow("Strategy", self.bottom_strategy_combo)
        layout.addLayout(form)

        # Threshold and strategy describe one file-level bottom track. Apply
        # them to the whole file after the controls settle, rather than
        # allowing different chunks to retain different detector settings.
        self.bottom_recalc_timer = QTimer(self)
        self.bottom_recalc_timer.setSingleShot(True)
        self.bottom_recalc_timer.setInterval(400)
        self.bottom_recalc_timer.timeout.connect(self.recalc_bottom_whole_file)
        self.bottom_threshold_spin.valueChanged.connect(
            lambda _value: self.bottom_recalc_timer.start()
        )
        self.bottom_strategy_combo.currentIndexChanged.connect(
            lambda _index: self.bottom_recalc_timer.start()
        )

        recalc_row = QHBoxLayout()
        self.recalc_all_button = QPushButton("Recalculate Whole File…")
        self.recalc_all_button.setToolTip(
            "Explicitly rerun the current threshold and strategy across the "
            "entire file. Changes to either control already do this "
            "automatically after a short pause."
        )
        self.recalc_all_button.clicked.connect(self.recalc_bottom_whole_file)
        recalc_row.addWidget(self.recalc_all_button)
        layout.addLayout(recalc_row)

        altitude_row = QHBoxLayout()
        self.bottom_search_range_spin = QDoubleSpinBox()
        self.bottom_search_range_spin.setRange(0.01, 1.0)
        self.bottom_search_range_spin.setSingleStep(0.01)
        self.bottom_search_range_spin.setDecimals(2)
        self.bottom_search_range_spin.setValue(0.06)
        self.bottom_search_range_spin.setToolTip(
            "Fraction of the ping width to search around the logged "
            "altitude (matches the CLI's 'Bottom line refinement search "
            "range' parameter)."
        )
        self.refine_altitude_button = QPushButton("Refine Using Altitude…")
        self.refine_altitude_button.setToolTip(
            "Use the sonar's logged altitude to constrain automatic "
            "detection to a narrow search window per chunk -- more robust "
            "than plain thresholding when altitude is logged, but requires "
            "it. Runs in the background."
        )
        self.refine_altitude_button.clicked.connect(self.refine_bottom_with_altitude)
        self.refine_altitude_button.setEnabled(
            bool(np.max(self.sidescan_file.sensor_primary_altitude) > 0)
        )
        altitude_row.addWidget(QLabel("Search range"))
        altitude_row.addWidget(self.bottom_search_range_spin)
        altitude_row.addWidget(self.refine_altitude_button)
        layout.addLayout(altitude_row)

        self.edit_bottom_button = QPushButton("Edit Bottom Line")
        self.edit_bottom_button.setCheckable(True)
        self.edit_bottom_button.setToolTip(
            "Drag across the waterfall to manually correct the bottom line. "
            "Panning and contact-picking are unavailable while this is on."
        )
        self.edit_bottom_button.toggled.connect(self._toggle_bottom_edit)
        layout.addWidget(self.edit_bottom_button)

        autosave_note = QLabel(
            "Saved automatically to <file>_bottom_info.npz beside the sonar "
            "file -- the same file the CLI, EGN table builder, and Napari "
            "bottom editor use."
        )
        autosave_note.setWordWrap(True)
        layout.addWidget(autosave_note)

        # Debounced like bottom_recalc_timer above, but longer -- this one
        # writes to disk, so a burst of drag-edits or slider ticks should
        # settle before it fires rather than saving on every single change.
        self.bottom_autosave_timer = QTimer(self)
        self.bottom_autosave_timer.setSingleShot(True)
        self.bottom_autosave_timer.setInterval(800)
        self.bottom_autosave_timer.timeout.connect(self._autosave_bottom_line)

        self.bottom_status_label = QLabel(self.context.bottom_info_status)
        self.bottom_status_label.setWordWrap(True)
        layout.addWidget(self.bottom_status_label)
        layout.addStretch(1)
        return panel

    def _toggle_bottom_edit(self, checked: bool) -> None:
        self.interaction_modes.set_mode(
            InteractionMode.BOTTOM_EDIT if checked else InteractionMode.PAN_ZOOM
        )

    def _apply_interaction_mode(self, mode: InteractionMode) -> None:
        editing = mode is InteractionMode.BOTTOM_EDIT
        self.view.edit_mode = editing
        self.view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if editing
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if self.edit_bottom_button.isChecked() != editing:
            self.edit_bottom_button.setChecked(editing)

    def _current_center_row(self) -> int:
        point = self.view.mapToScene(self.view.viewport().rect().center())
        return max(0, min(self.sidescan_file.num_ping - 1, math.floor(point.y())))

    def _refresh_bottom_overlay(self) -> None:
        mask = logical_bottom_overlay(self.preprocessor, self.sidescan_file.num_ping)
        # Slant-range correction collapses the tracked bottom onto the new
        # nadir. The original red tracking line would otherwise be drawn at
        # stale acoustic-sample positions over the corrected seabed.
        if self.slant_range_checkbox.isChecked():
            mask = np.zeros_like(mask)
        self.view.set_bottom_overlay(mask)

    def _apply_bottom_edit(self, row: int, column: int) -> None:
        preproc = self.preprocessor
        chunk_idx, local_ping_idx = divmod(row, preproc.chunk_size)
        if chunk_idx >= preproc.num_chunk:
            return
        combine_both_sides = (
            self.bottom_strategy_combo.currentText()
            == preproc.bottom_strategy_choices[1]
        )

        def clamp_bottom(value):
            return max(1, min(preproc.ping_len - 1, value))

        if column < preproc.ping_len:
            port_sample = clamp_bottom(column)
            preproc.napari_portside_bottom[chunk_idx, local_ping_idx] = port_sample
            if combine_both_sides:
                preproc.napari_starboard_bottom[chunk_idx, local_ping_idx] = (
                    clamp_bottom(preproc.ping_len - port_sample)
                )
        else:
            starboard_sample = clamp_bottom(column - preproc.ping_len)
            preproc.napari_starboard_bottom[chunk_idx, local_ping_idx] = (
                starboard_sample
            )
            if combine_both_sides:
                preproc.napari_portside_bottom[chunk_idx, local_ping_idx] = (
                    clamp_bottom(preproc.ping_len - starboard_sample)
                )
        preproc.update_bottom_map_napari(chunk_idx, add_line_width=0)
        preproc.sync_chunked_bottom_to_flat(chunk_idx)
        self.view.patch_bottom_overlay_row(
            row, preproc.bottom_map[chunk_idx, local_ping_idx]
        )
        self._mark_bottom_line_dirty()

    def recalc_bottom_whole_file(self) -> None:
        if self.bottom_worker is not None:
            self._pending_full_bottom_recalc = True
            self.bottom_status_label.setText(
                "Current calculation finishing; latest threshold queued…"
            )
            return
        self._pending_full_bottom_recalc = False
        threshold = self.bottom_threshold_spin.value()
        combine_both_sides = (
            self.bottom_strategy_combo.currentText()
            == self.preprocessor.bottom_strategy_choices[1]
        )

        def run_algorithm(processor: SidescanPreprocessor) -> None:
            processor.detect_bottom_line_t(
                threshold_bin=threshold, combine_both_sides=combine_both_sides
            )

        self._start_bottom_worker(run_algorithm, "Recalculating bottom line…")

    def refine_bottom_with_altitude(self) -> None:
        search_range = self.bottom_search_range_spin.value()

        def run_algorithm(processor: SidescanPreprocessor) -> None:
            processor.refine_detected_bottom_line(search_range)

        self._start_bottom_worker(
            run_algorithm, "Refining bottom line using altitude…"
        )

    def _start_bottom_worker(self, run_algorithm, status_message: str) -> None:
        # The private copy is built here, on the GUI thread, before the
        # worker starts -- the worker thread only ever touches this copy,
        # never self.preprocessor (see _copy_preprocessor_for_bottom_line).
        processor_copy = _copy_preprocessor_for_bottom_line(self.preprocessor)
        self._set_bottom_controls_enabled(False)
        self.bottom_status_label.setText(status_message)
        worker = BottomLineRecalcWorker(processor_copy, run_algorithm)
        self.bottom_worker = worker
        self._bottom_worker_source = self.filepath
        worker.signals.finished.connect(self._bottom_recalc_finished)
        worker.signals.failed.connect(self._bottom_recalc_failed)
        self.thread_pool.start(worker)

    def _bottom_recalc_finished(self, processor_copy: SidescanPreprocessor) -> None:
        worker_source = self._bottom_worker_source
        self.bottom_worker = None
        self._bottom_worker_source = None
        if self._pending_full_bottom_recalc:
            # The user changed the threshold/strategy while this calculation
            # was running. Discard its now-obsolete result and launch one
            # calculation with the newest values.
            self._pending_full_bottom_recalc = False
            self._set_bottom_controls_enabled(True)
            self.bottom_status_label.setText("Applying latest bottom settings…")
            QTimer.singleShot(0, self.recalc_bottom_whole_file)
            return
        if worker_source != self.filepath:
            # A file switch occurred while the worker was running. Never let
            # an old file's arrays overwrite the newly loaded preprocessor.
            self._set_bottom_controls_enabled(True)
            return
        # Whole-object attribute reassignment, never in-place mutation --
        # reassignment is atomic, an in-place slice write on the array the
        # GUI thread might concurrently be reading/rendering is not.
        self.preprocessor.portside_bottom_dist = processor_copy.portside_bottom_dist
        self.preprocessor.starboard_bottom_dist = processor_copy.starboard_bottom_dist
        self.preprocessor.napari_portside_bottom = processor_copy.napari_portside_bottom
        self.preprocessor.napari_starboard_bottom = (
            processor_copy.napari_starboard_bottom
        )
        self.preprocessor.bottom_map = processor_copy.bottom_map
        self._refresh_bottom_overlay()
        self._set_bottom_controls_enabled(True)
        self.bottom_status_label.setText("Bottom line updated")
        self._mark_bottom_line_dirty()

    def _bottom_recalc_failed(self, message: str) -> None:
        self.bottom_worker = None
        self._bottom_worker_source = None
        self._set_bottom_controls_enabled(True)
        if self._pending_full_bottom_recalc:
            self._pending_full_bottom_recalc = False
            self.bottom_status_label.setText("Applying latest bottom settings…")
            QTimer.singleShot(0, self.recalc_bottom_whole_file)
            return
        self.bottom_status_label.setText(f"Bottom line recalculation failed: {message}")

    def _set_bottom_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.recalc_all_button,
            self.refine_altitude_button,
            self.edit_bottom_button,
        ):
            widget.setEnabled(enabled)
        if enabled:
            # Stays disabled with no altitude logged, independent of
            # whether a worker just finished.
            self.refine_altitude_button.setEnabled(
                bool(np.max(self.sidescan_file.sensor_primary_altitude) > 0)
            )

    def _default_bottom_info_path(self) -> Path:
        return self.filepath.parent / f"{self.filepath.stem}_bottom_info.npz"

    def _mark_bottom_line_dirty(self) -> None:
        """Restart the autosave debounce -- called from every bottom-line
        mutation (manual edit, chunk/whole-file/altitude recalculation) so
        the ancillary file next to the sonar file never falls out of sync
        with what's on screen, without writing to disk on every single
        ping edited mid-drag."""
        self.bottom_autosave_timer.start()

    def _autosave_bottom_line(self) -> None:
        path = self._default_bottom_info_path()
        save_bottom_info(path, self.preprocessor, self.sidescan_file)
        self.bottom_status_label.setText(f"Bottom line auto-saved to {path.name}")

    def _flush_pending_bottom_line_save(self) -> None:
        """Save immediately if an edit is still waiting on the debounce --
        called before switching files or closing the window, so a change
        made just before either can never be silently lost."""
        if self.bottom_autosave_timer.isActive():
            self.bottom_autosave_timer.stop()
            self._autosave_bottom_line()

    def _overall_gain_changed(self, value) -> None:
        self.display.overall_gain_db = float(value)
        if self.gain_spin.value() != float(value):
            self.gain_spin.setValue(float(value))
        self.render_timer.start()

    def _tvg_spreading_changed(self, value) -> None:
        self.display.tvg_spreading_db_per_decade = float(value)
        if self.tvg_spreading_spin.value() != float(value):
            self.tvg_spreading_spin.setValue(float(value))
        self.render_timer.start()

    def _tvg_absorption_changed(self, value) -> None:
        self.display.tvg_absorption_db_per_m = float(value)
        self.render_timer.start()

    def reset_gain(self) -> None:
        self.display.clear_normalization()
        self.gain_spin.setValue(WaterfallGainModel.DEFAULT_OVERALL_GAIN_DB)
        self.tvg_spreading_spin.setValue(
            WaterfallGainModel.DEFAULT_TVG_SPREADING_DB_PER_DECADE
        )
        self.tvg_absorption_spin.setValue(
            WaterfallGainModel.DEFAULT_TVG_ABSORPTION_DB_PER_M
        )
        self._schedule_gain_settings_save()
        self.render_gain()

    def normalize_tvg(self) -> None:
        try:
            overall, spreading, absorption = self.display.normalize_tvg()
        except ValueError as error:
            self.statusBar().showMessage(f"TVG normalization failed: {error}")
            return
        self.gain_spin.setValue(overall)
        self.tvg_spreading_spin.setValue(spreading)
        self.tvg_absorption_spin.setValue(absorption)
        self.render_gain()
        self.statusBar().showMessage(
            "Swath brightness equalized around "
            f"{self.display.normalize_target_percent}%; "
            f"gain {overall:+.1f} dB, "
            f"spreading {spreading:+.1f} dB/decade, "
            f"absorption {absorption:+.2f} dB/m"
        )
        self._schedule_gain_settings_save()

    def set_normalize_target(self) -> None:
        target_percent, accepted = QInputDialog.getInt(
            self,
            "Auto TVG Brightness Target",
            "Target brightness:",
            self.display.normalize_target_percent,
            WaterfallGainModel.MIN_NORMALIZE_TARGET_PERCENT,
            WaterfallGainModel.MAX_NORMALIZE_TARGET_PERCENT,
            1,
        )
        if not accepted:
            return
        self.display.set_normalize_target_percent(target_percent)
        self.normalize_target_button.setText(
            f"Auto TVG Brightness Target: {target_percent}%…"
        )
        self.normalize_tvg()

    def render_gain(self) -> None:
        self.view.set_image(self.display.render_rgb())
        self._update_status()

    def _update_status(self, message: str | None = None) -> None:
        detail = (
            f"{self.sidescan_file.num_ping:,} pings × "
            f"{2 * self.preprocessor.ping_len:,} display samples; "
            f"gain {self.display.overall_gain_db:+.1f} dB; "
            f"TVG {self.display.tvg_spreading_db_per_decade:+.1f} dB/decade, "
            f"{self.display.tvg_absorption_db_per_m:+.2f} dB/m"
        )
        self.statusBar().showMessage(f"{message + '; ' if message else ''}{detail}")

    def _show_hover_stats(self, row: int, column: int) -> None:
        sidescan_file = self.sidescan_file
        side = "Port" if column < self.preprocessor.ping_len else "Stbd"
        parts = [f"Ping {row:,}/{sidescan_file.num_ping - 1:,}"]

        range_m, calibrated = self.display.range_at_column(column)
        if calibrated:
            parts.append(f"{side} range {range_m:.1f} m")

        amplitude = self.display.corrected_value(row, column)
        parts.append(f"Amplitude {amplitude * 100:.0f}%")

        altitude = sidescan_file.sensor_primary_altitude[row]
        if math.isfinite(altitude):
            parts.append(f"Altitude {altitude:.1f} m")

        speed_kn = sidescan_file.sensor_speed[row] * 1.943844
        if math.isfinite(speed_kn):
            parts.append(f"Speed {speed_kn:.1f} kn")

        heading = sidescan_file.sensor_heading[row]
        if math.isfinite(heading):
            parts.append(f"Heading {heading:03.0f}°")

        latitude, longitude = sidescan_file.latitude[row], sidescan_file.longitude[row]
        if math.isfinite(latitude) and math.isfinite(longitude):
            parts.append(f"{latitude:.6f}, {longitude:.6f}")

        self.hover_stats_label.setText("    ".join(parts))

    def _clear_hover_stats(self) -> None:
        self.hover_stats_label.setText("")

    def refresh_chunk(self, *args) -> None:
        markers = []
        for record in self.store.list_contacts(source_file_id=self.source_file_id):
            anchor = record.draft.anchor
            if anchor.channel.value == 0:
                column = round(
                    (1.0 - anchor.sample_fraction)
                    * (self.preprocessor.ping_len - 1)
                )
            else:
                column = self.preprocessor.ping_len + round(
                    anchor.sample_fraction * (self.preprocessor.ping_len - 1)
                )
            markers.append(
                (anchor.global_ping_index, column, record.draft.name)
            )
        self.view.set_markers(markers)

    def center_selected_contact(self, *args) -> None:
        record = self.contact_dock.selected_record()
        if record is None:
            return
        anchor = record.draft.anchor
        if anchor.channel.value == 0:
            column = round(
                (1.0 - anchor.sample_fraction)
                * (self.preprocessor.ping_len - 1)
            )
        else:
            column = self.preprocessor.ping_len + round(
                anchor.sample_fraction * (self.preprocessor.ping_len - 1)
            )
        self.view.center_on(anchor.global_ping_index, column)

    def pick_contact(self, row: int, column: int) -> None:
        chunk_index, local_ping_index = divmod(
            row, self.preprocessor.chunk_size
        )
        try:
            result = self.picker.pick_display_pixel(
                chunk_index=chunk_index,
                local_ping_index=local_ping_index,
                display_x=column,
            )
        except DuplicateContactAnchor:
            self.statusBar().showMessage(
                "A contact already exists at this sonar sample", 5000
            )
            return
        except Exception as exc:
            self.statusBar().showMessage(f"Contact not saved: {exc}", 8000)
            return
        self.contact_dock.refresh_and_focus_name(select_contact_id=result.contact.id)
        self.refresh_chunk()
        self._update_status(
            result.thumbnail_warning or f"Saved {result.contact.draft.name}"
        )

    def closeEvent(self, event) -> None:
        self._flush_pending_bottom_line_save()
        self._flush_pending_gain_settings_save()
        self.store.close()
        super().closeEvent(event)


class QtContactPickerStartWindow(QMainWindow):
    """Idle Qt workspace shown until the user selects a sonar file."""

    def __init__(
        self,
        open_selected_file: Callable[[Path], QtContactPickerWindow],
        *,
        initial_directory: Path | None = None,
    ):
        super().__init__()
        self._open_selected_file = open_selected_file
        self._initial_directory = initial_directory
        self._loaded_window: QtContactPickerWindow | None = None
        self.setWindowTitle("SidescanTools - Contact picker (Qt raster)")
        self.resize(1400, 820)

        self.open_button = QPushButton("Open…")
        self.open_button.setToolTip("Open a sonar file")
        self.open_button.clicked.connect(self.open_file)
        self.previous_file_button = QPushButton("◀ Previous file")
        self.next_file_button = QPushButton("Next file ▶")
        self.previous_file_button.setEnabled(False)
        self.next_file_button.setEnabled(False)
        self.file_position_label = QLabel("No sonar file selected")

        file_nav = QHBoxLayout()
        file_nav.addWidget(self.open_button)
        file_nav.addSpacing(14)
        file_nav.addWidget(self.previous_file_button)
        file_nav.addWidget(self.next_file_button)
        file_nav.addWidget(self.file_position_label, 1)

        prompt = QLabel(
            "Open a JSF or XTF sidescan file to display its waterfall and "
            "enable processing, contacts, and export tools."
        )
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setWordWrap(True)
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addLayout(file_nav)
        central_layout.addWidget(prompt, 1)
        self.setCentralWidget(central)

        processing_dock = QDockWidget("Processing and gain", self)
        processing_message = QLabel("Open a sonar file to enable these controls.")
        processing_message.setWordWrap(True)
        processing_message.setMargin(10)
        processing_dock.setWidget(processing_message)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, processing_dock)

        contacts_dock = QDockWidget("Sonar Contacts", self)
        contact_group = QGroupBox("Contact List")
        contact_group.setStyleSheet(
            "QGroupBox { border: 2px solid #111; border-radius: 3px; "
            "margin-top: 0.7em; padding-top: 0.5em; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
            "padding: 0 4px; }"
        )
        contact_layout = QVBoxLayout(contact_group)
        contact_layout.addWidget(QLabel("No sonar file selected"))
        contact_layout.addStretch(1)
        contacts_dock.setWidget(contact_group)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, contacts_dock)
        self.statusBar().showMessage("Ready — select Open… to choose a sonar file")

    def open_file(self) -> None:
        start_dir = str(self._initial_directory or "")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open sonar file",
            start_dir,
            "Sidescan files (*.jsf *.xtf);;All files (*)",
        )
        if not filename:
            return
        filepath = Path(filename)
        if filepath.suffix.casefold() not in {".jsf", ".xtf"}:
            QMessageBox.warning(
                self,
                "Unsupported sonar file",
                "Select a JSF or XTF sidescan file.",
            )
            return
        self.statusBar().showMessage(f"Loading {filepath.name}…")
        QApplication.processEvents()
        try:
            loaded_window = self._open_selected_file(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "Could not open sonar file", str(exc))
            self.statusBar().showMessage("No sonar file selected")
            return
        self._loaded_window = loaded_window
        loaded_window.show()
        self.close()


def run_qt_contact_picker(
    filepath: str | os.PathLike | None = None,
    *,
    chunk_size: int = 256,
    default_threshold: float = 0.7,
    downsampling_factor: int = 32,
    work_dir: str | os.PathLike | None = None,
    active_dB: bool = False,
    active_hist_equal: bool = False,
    contacts_db_path: str | os.PathLike | None = None,
    geometry_settings: GeometrySettings | None = None,
    block: bool = True,
):
    """Open the no-OpenGL contact picker using Qt's raster paint engine.

    ``filepath=None`` (the desktop-shortcut launch path) opens an idle main
    workspace. The user can then choose a file with the normal Open button.
    """
    application = QApplication.instance() or QApplication([])
    icon_path = Path(__file__).resolve().parent / "res" / "icon.ico"
    if icon_path.is_file():
        application.setWindowIcon(QIcon(str(icon_path)))

    if filepath is None:
        initial_directory = Path(work_dir) if work_dir is not None else None

        def open_selected_file(selected: Path) -> QtContactPickerWindow:
            return run_qt_contact_picker(
                selected,
                chunk_size=chunk_size,
                default_threshold=default_threshold,
                downsampling_factor=downsampling_factor,
                work_dir=work_dir,
                active_dB=active_dB,
                active_hist_equal=active_hist_equal,
                contacts_db_path=contacts_db_path,
                geometry_settings=geometry_settings,
                block=False,
            )

        window = QtContactPickerStartWindow(
            open_selected_file,
            initial_directory=initial_directory,
        )
        window.show()
        if block:
            application.exec()
        return window

    filepath = Path(filepath)
    output_directory = Path(work_dir) if work_dir is not None else filepath.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    database = (
        Path(contacts_db_path)
        if contacts_db_path is not None
        else output_directory / "contacts.sqlite"
    )
    loader_settings = SonarLoaderSettings(
        chunk_size=chunk_size,
        default_threshold=default_threshold,
        downsampling_factor=downsampling_factor,
        active_dB=active_dB,
        active_hist_equal=active_hist_equal,
        output_directory=output_directory,
        geometry_settings=geometry_settings or GeometrySettings(vertical_beam_angle=60),
    )

    store = ContactStore(database)
    context = _load_sonar_context(filepath, settings=loader_settings, store=store)
    display = WaterfallGainModel(context.raw_waterfall, slant_range_m=context.slant_range_m)
    picker = _build_contact_picker(context, store=store, display=display)

    window = QtContactPickerWindow(
        context=context,
        store=store,
        picker=picker,
        contacts_db_path=database,
        display=display,
        loader_settings=loader_settings,
    )
    window.show()
    if block:
        application.exec()
    return window
