"""Redesigned Qt UI for the SidescanTools contact picker.

Same backend as ``qt_contact_picker_ui.py`` -- file loading, the waterfall
gain model, contact picking, and the SQLite store are all reused unchanged.
Only the window's layout, styling, and widget choices are new, so this can
be run side-by-side with the classic UI for comparison:

    sidescantools-contacts <file> --ui classic
    sidescantools-contacts <file> --ui v2

Design notes for anyone diffing this against the classic window:

- The TVG/gain row's spin-box step buttons were tiny native Qt arrows
  squeezed into a single overcrowded QHBoxLayout with four sliders fighting
  for space. ``GainControl`` below replaces each with a label + big +/-
  QToolButtons + numeric readout on one line, and a full-width slider on the
  line beneath -- both bigger step targets and more room for the slider.
- File navigation (Open / Previous / Next) moves to the top of the
  "Processing and Gain" dock instead of a thin bar above the canvas.
- A QSS stylesheet (Fusion style, forced below) gives every control a
  consistent dark theme with a clear primary/secondary/danger button
  hierarchy, since the classic window has no stylesheet at all and just
  renders whatever the platform's native Qt style happens to do.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableView,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sidescantools.contact_export import GPXExporter
from sidescantools.contact_gain import BuiltInGainMode, BuiltInGainRequest
from sidescantools.contact_store import ContactStore, DuplicateContactAnchor
from sidescantools.contact_ui import ContactTableModel
from sidescantools.custom_widgets import ErrorWarnDialog
from sidescantools.swath_geometry import GeometrySettings

# Everything below is shared, unmodified backend/UI plumbing -- reused
# rather than duplicated so the redesigned window can never drift from the
# classic one's file-loading, gain-model, or contact-picking behavior.
from sidescantools.qt_contact_picker_ui import (
    EGNTableBuilderDialog,
    GainProcessingWorker,
    SonarFileContext,
    SonarLoaderSettings,
    WaterfallGainModel,
    WaterfallView,
    _build_contact_picker,
    _load_sonar_context,
    _register_in_store,
    sonar_files_in_directory,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG_APP = "#181b20"
BG_PANEL = "#1f232a"
BG_ELEVATED = "#262b33"
BG_ELEVATED_2 = "#2d333c"
BG_INPUT = "#20242b"
BORDER = "#343b45"
BORDER_STRONG = "#454e5a"
TEXT_PRIMARY = "#e8ecf1"
TEXT_SECONDARY = "#9aa5b3"
TEXT_MUTED = "#6b7480"
ACCENT = "#37c2b7"
ACCENT_HOVER = "#4ad2c7"
ACCENT_PRESSED = "#2aa89e"
ACCENT_TEXT_ON = "#08211f"
DANGER = "#e2585f"
DANGER_HOVER = "#ea6c72"
WARN = "#e3ad4f"

STYLE_SHEET = """
QMainWindow, QWidget#centralArea { background: #181b20; }
QWidget { color: #e8ecf1; font-size: 13px; }
QMainWindow::separator { background: #343b45; width: 4px; height: 4px; }
QMainWindow::separator:hover { background: #37c2b7; }

QToolTip {
    background: #2d333c; color: #e8ecf1; border: 1px solid #454e5a;
    padding: 4px 8px; border-radius: 4px;
}

QDockWidget { color: #e8ecf1; }
QDockWidget::title {
    background: #1f232a; color: #e8ecf1; padding: 9px 10px; font-weight: 600;
    border-bottom: 1px solid #343b45;
}

QLabel#fileName { font-weight: 700; font-size: 14px; color: #e8ecf1; }
QLabel#caption { color: #6b7480; font-size: 11px; }
QLabel#dbLabel { color: #9aa5b3; font-size: 12px; }
QLabel#hintText { color: #9aa5b3; font-size: 12px; }
QLabel#statusPill { color: #9aa5b3; font-size: 12px; padding: 2px 0; }
QLabel#hoverStats { color: #9aa5b3; font-family: Consolas, "Courier New", monospace; }

QFrame#fileCard, QFrame#dbCard, QFrame#detailCard, QLabel#thumbCard {
    background: #262b33; border: 1px solid #343b45; border-radius: 8px; color: #6b7480;
}
QFrame#hintBanner {
    background: #262b33; border: 1px solid #343b45; border-left: 3px solid #37c2b7;
    border-radius: 6px;
}
QToolButton#hintClose {
    background: transparent; border: none; color: #6b7480; font-weight: 700;
}
QToolButton#hintClose:hover { color: #e8ecf1; }

QGroupBox {
    background: #1f232a; border: 1px solid #343b45; border-radius: 6px;
    margin-top: 16px; padding: 12px 10px 10px 10px; font-weight: 600; color: #9aa5b3;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #37c2b7;
}

QPushButton, QToolButton {
    background: #262b33; color: #e8ecf1; border: 1px solid #454e5a;
    border-radius: 5px; padding: 6px 12px;
}
QPushButton:hover, QToolButton:hover { background: #2d333c; border-color: #37c2b7; }
QPushButton:pressed, QToolButton:pressed { background: #20242b; }
QPushButton:disabled, QToolButton:disabled { color: #6b7480; background: #1f232a; border-color: #343b45; }

QPushButton[class="primary"] { background: #37c2b7; color: #08211f; border: 1px solid #37c2b7; font-weight: 700; }
QPushButton[class="primary"]:hover { background: #4ad2c7; }
QPushButton[class="primary"]:pressed { background: #2aa89e; }
QPushButton[class="primary"]:disabled { background: #262b33; color: #6b7480; border-color: #343b45; }

QPushButton[class="danger"] { background: transparent; color: #e2585f; border: 1px solid #e2585f; }
QPushButton[class="danger"]:hover { background: rgba(226,88,95,0.14); color: #ea6c72; border-color: #ea6c72; }
QPushButton[class="danger"]:disabled { color: #6b7480; border-color: #343b45; }

QPushButton[class="ghost"], QToolButton[class="ghost"] { background: transparent; border: 1px solid transparent; color: #9aa5b3; }
QPushButton[class="ghost"]:hover, QToolButton[class="ghost"]:hover { background: #262b33; color: #e8ecf1; border-color: #343b45; }

QToolButton#stepBtn { font-weight: 700; font-size: 15px; padding: 0; }
QToolButton#navBtn, QToolButton#navBtnPrimary { border-radius: 18px; }
QToolButton#navBtnPrimary { background: #37c2b7; border: 1px solid #37c2b7; color: #08211f; }
QToolButton#navBtnPrimary:hover { background: #4ad2c7; }
QToolButton#navBtnPrimary:disabled { background: #262b33; border-color: #343b45; }
QToolButton#navBtn:disabled { color: #6b7480; }

QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #20242b; border: 1px solid #343b45; border-radius: 4px;
    padding: 4px 6px; color: #e8ecf1; selection-background-color: #37c2b7;
    min-height: 22px;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #37c2b7;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #6b7480; background: #1f232a;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #20242b; color: #e8ecf1; selection-background-color: #37c2b7;
    selection-color: #08211f; border: 1px solid #454e5a;
}

QCheckBox { spacing: 8px; padding: 2px 0; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #454e5a; border-radius: 3px; background: #20242b;
}
QCheckBox::indicator:checked { background: #37c2b7; border-color: #37c2b7; }
QCheckBox:disabled { color: #6b7480; }

QSlider::groove:horizontal { height: 6px; background: #20242b; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #37c2b7; border-radius: 3px; }
QSlider::add-page:horizontal { background: #20242b; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
    background: #37c2b7; border: 2px solid #181b20;
}
QSlider::handle:horizontal:hover { background: #4ad2c7; }
QSlider:disabled::groove:horizontal { background: #1f232a; }
QSlider:disabled::handle:horizontal { background: #454e5a; border-color: #181b20; }

QTableView {
    background: #1f232a; alternate-background-color: #232830; gridline-color: #343b45;
    selection-background-color: rgba(55,194,183,0.22); selection-color: #e8ecf1;
    border: 1px solid #343b45; border-radius: 6px;
}
QHeaderView::section {
    background: #262b33; color: #9aa5b3; padding: 6px; border: none;
    border-bottom: 1px solid #454e5a; border-right: 1px solid #343b45; font-weight: 600;
}

QProgressBar {
    background: #20242b; border: 1px solid #343b45; border-radius: 4px;
    text-align: center; color: #9aa5b3; max-height: 8px;
}
QProgressBar::chunk { background: #37c2b7; border-radius: 3px; }

QStatusBar { background: #1f232a; color: #9aa5b3; border-top: 1px solid #343b45; }
QStatusBar::item { border: none; }

QScrollBar:vertical { background: #181b20; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #454e5a; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #37c2b7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #181b20; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #454e5a; border-radius: 5px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: #37c2b7; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


# ---------------------------------------------------------------------------
# Small hand-painted icon set -- avoids depending on any icon theme/assets
# ---------------------------------------------------------------------------


def _icon(kind: str, color: str = TEXT_PRIMARY, size: int = 18) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.4, size * 0.09))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    s = float(size)
    m = s * 0.18

    if kind == "chevron-left":
        painter.drawLine(QPointF(s * 0.62, m), QPointF(s * 0.34, s * 0.5))
        painter.drawLine(QPointF(s * 0.34, s * 0.5), QPointF(s * 0.62, s - m))
    elif kind == "chevron-right":
        painter.drawLine(QPointF(s * 0.38, m), QPointF(s * 0.66, s * 0.5))
        painter.drawLine(QPointF(s * 0.66, s * 0.5), QPointF(s * 0.38, s - m))
    elif kind == "folder":
        path = QPainterPath()
        path.moveTo(m, s * 0.32)
        path.lineTo(s * 0.42, s * 0.32)
        path.lineTo(s * 0.5, s * 0.42)
        path.lineTo(s - m, s * 0.42)
        path.lineTo(s - m, s - m)
        path.lineTo(m, s - m)
        path.closeSubpath()
        painter.drawPath(path)
    elif kind == "refresh":
        rect = QRectF(m, m, s - 2 * m, s - 2 * m)
        painter.drawArc(rect, 35 * 16, 280 * 16)
        painter.drawLine(QPointF(s - m, m + s * 0.02), QPointF(s - m * 1.7, m - s * 0.05))
        painter.drawLine(QPointF(s - m, m + s * 0.02), QPointF(s - m * 0.55, m + s * 0.2))
    elif kind == "activity":
        path = QPainterPath()
        path.moveTo(m, s * 0.52)
        path.lineTo(s * 0.34, s * 0.52)
        path.lineTo(s * 0.43, s * 0.25)
        path.lineTo(s * 0.58, s * 0.75)
        path.lineTo(s * 0.67, s * 0.52)
        path.lineTo(s - m, s * 0.52)
        painter.drawPath(path)
    elif kind == "save":
        painter.drawRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), 2, 2)
        painter.drawRect(QRectF(s * 0.32, m, s * 0.36, s * 0.24))
        painter.drawRect(QRectF(s * 0.3, s * 0.56, s * 0.4, s * 0.3))
    elif kind == "delete":
        painter.drawLine(QPointF(s * 0.22, s * 0.28), QPointF(s * 0.78, s * 0.28))
        painter.drawRect(QRectF(s * 0.38, s * 0.12, s * 0.24, s * 0.16))
        path = QPainterPath()
        path.moveTo(s * 0.28, s * 0.28)
        path.lineTo(s * 0.33, s * 0.86)
        path.lineTo(s * 0.67, s * 0.86)
        path.lineTo(s * 0.72, s * 0.28)
        painter.drawPath(path)
        painter.drawLine(QPointF(s * 0.42, s * 0.4), QPointF(s * 0.42, s * 0.76))
        painter.drawLine(QPointF(s * 0.58, s * 0.4), QPointF(s * 0.58, s * 0.76))
    elif kind == "export":
        painter.drawRect(QRectF(m, s * 0.58, s - 2 * m, s * 0.26))
        painter.drawLine(QPointF(s * 0.5, m), QPointF(s * 0.5, s * 0.56))
        painter.drawLine(QPointF(s * 0.32, s * 0.28), QPointF(s * 0.5, m))
        painter.drawLine(QPointF(s * 0.68, s * 0.28), QPointF(s * 0.5, m))
    elif kind == "database":
        painter.drawEllipse(QRectF(m, m, s - 2 * m, s * 0.22))
        painter.drawLine(QPointF(m, m + s * 0.11), QPointF(m, s - m - s * 0.11))
        painter.drawLine(QPointF(s - m, m + s * 0.11), QPointF(s - m, s - m - s * 0.11))
        painter.drawArc(QRectF(m, s - m - s * 0.22, s - 2 * m, s * 0.22), 180 * 16, 180 * 16)
    elif kind == "check":
        painter.drawLine(QPointF(s * 0.22, s * 0.52), QPointF(s * 0.42, s * 0.72))
        painter.drawLine(QPointF(s * 0.42, s * 0.72), QPointF(s * 0.8, s * 0.28))
    elif kind == "eye":
        painter.drawEllipse(QRectF(m, s * 0.32, s - 2 * m, s * 0.36))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.07, s * 0.07)
    elif kind == "info":
        painter.drawEllipse(QRectF(m, m, s - 2 * m, s - 2 * m))
        painter.drawLine(QPointF(s * 0.5, s * 0.46), QPointF(s * 0.5, s * 0.72))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(s * 0.5, s * 0.32), s * 0.045, s * 0.045)
    elif kind == "target":
        painter.drawEllipse(QRectF(m, m, s - 2 * m, s - 2 * m))
        painter.drawEllipse(QRectF(s * 0.38, s * 0.38, s * 0.24, s * 0.24))
        painter.drawLine(QPointF(s * 0.5, 0), QPointF(s * 0.5, m))
        painter.drawLine(QPointF(s * 0.5, s - m), QPointF(s * 0.5, s))
        painter.drawLine(QPointF(0, s * 0.5), QPointF(m, s * 0.5))
        painter.drawLine(QPointF(s - m, s * 0.5), QPointF(s, s * 0.5))
    painter.end()
    return QIcon(pixmap)


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------


class GainControl(QWidget):
    """Label + big +/- steppers on one line, full-width slider beneath.

    Replaces the classic window's tiny native spin-box arrows and a slider
    squeezed to a handful of pixels by four controls sharing one row.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        step: float,
        decimals: int,
        suffix: str,
        slider_scale: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self._scale = slider_scale

        name = QLabel(label)
        name.setObjectName("controlLabel")

        self.minus = QToolButton()
        self.minus.setObjectName("stepBtn")
        self.minus.setText("−")
        self.plus = QToolButton()
        self.plus.setObjectName("stepBtn")
        self.plus.setText("+")
        for button in (self.minus, self.plus):
            button.setFixedSize(28, 28)
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(400)
            button.setAutoRepeatInterval(70)

        self.spin = QDoubleSpinBox()
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin.setFixedWidth(100)

        self.minus.clicked.connect(lambda: self.spin.stepBy(-1))
        self.plus.clicked.connect(lambda: self.spin.stepBy(1))

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(name)
        top.addStretch(1)
        top.addWidget(self.minus)
        top.addWidget(self.spin)
        top.addWidget(self.plus)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * slider_scale), round(maximum * slider_scale))
        self.slider.setSingleStep(max(1, round(step * slider_scale)))
        self.slider.setValue(round(value * slider_scale))

        self.slider.valueChanged.connect(self._slider_to_spin)
        self.spin.valueChanged.connect(self._spin_to_slider)
        self.spin.valueChanged.connect(self.valueChanged)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.slider)

    def _slider_to_spin(self, raw: int) -> None:
        value = raw / self._scale
        if abs(self.spin.value() - value) > 1e-9:
            self.spin.setValue(value)

    def _spin_to_slider(self, value: float) -> None:
        raw = round(value * self._scale)
        if self.slider.value() != raw:
            self.slider.setValue(raw)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt override
        super().setToolTip(text)
        self.spin.setToolTip(text)
        self.slider.setToolTip(text)


class Stepper(QWidget):
    """Compact "− [value] +" control for a QFormLayout row.

    Used instead of a bare QSpinBox/QDoubleSpinBox: once the spin button
    box is given any custom background via QSS, Qt's built-in up/down
    arrow glyphs stop rendering visibly on a dark palette (confirmed via a
    real screenshot while building this window -- the button box painted
    but with no visible arrow at all). Big painted +/- buttons sidestep
    that entirely and match the same stepper pattern used for gain/TVG.
    """

    valueChanged = Signal(object)

    def __init__(
        self,
        *,
        minimum,
        maximum,
        value,
        step,
        decimals: int = 0,
        suffix: str = "",
        integer: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.minus = QToolButton()
        self.minus.setObjectName("stepBtn")
        self.minus.setText("−")
        self.plus = QToolButton()
        self.plus.setObjectName("stepBtn")
        self.plus.setText("+")
        for button in (self.minus, self.plus):
            button.setFixedSize(26, 26)
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(400)
            button.setAutoRepeatInterval(70)

        self.spin = QSpinBox() if integer else QDoubleSpinBox()
        if not integer:
            self.spin.setDecimals(decimals)
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.minus.clicked.connect(lambda: self.spin.stepBy(-1))
        self.plus.clicked.connect(lambda: self.spin.stepBy(1))
        # Routed through a lambda rather than connected signal-to-signal:
        # PyQt5 rejects a direct connection from a double-typed signal to a
        # Signal(object) receiver (int-typed QSpinBox.valueChanged connects
        # fine, but this widget wraps both), even though relaying the same
        # value through a plain callable works for either.
        self.spin.valueChanged.connect(lambda value: self.valueChanged.emit(value))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.minus)
        layout.addWidget(self.spin, 1)
        layout.addWidget(self.plus)

    def value(self):
        return self.spin.value()

    def setValue(self, value) -> None:
        self.spin.setValue(value)


class HintBanner(QFrame):
    """Dismissible instruction banner -- present by default, but out of an
    expert user's way for the rest of the session once closed."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("hintBanner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 6, 8)
        layout.setSpacing(8)

        icon_label = QLabel()
        icon_label.setPixmap(_icon("info", ACCENT, 16).pixmap(16, 16))
        message = QLabel(text)
        message.setObjectName("hintText")
        message.setWordWrap(True)
        close_button = QToolButton()
        close_button.setObjectName("hintClose")
        close_button.setText("✕")
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.setToolTip("Dismiss")
        close_button.clicked.connect(self.hide)

        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(message, 1)
        layout.addWidget(close_button, 0, Qt.AlignmentFlag.AlignTop)


# ---------------------------------------------------------------------------
# Contacts dock (redesigned view; same ContactTableModel/store underneath)
# ---------------------------------------------------------------------------


class ContactDockV2(QWidget):
    """Contact list/editor -- same store and model as the classic
    ``ContactDock``, rebuilt with a clearer visual hierarchy: a status pill,
    a cleaner table, a smaller/nicer thumbnail well, and Save/Delete/Export
    given distinct primary/danger/secondary weight instead of four identical
    buttons in a row.
    """

    contact_updated = Signal(int)
    contact_deleted = Signal(int)

    CLASSIFICATIONS = ["", "Debris", "Wreck", "Rock / Reef", "Anomaly", "Unknown"]

    def __init__(self, store: ContactStore, source_file_id: int, *, export_directory, parent=None):
        super().__init__(parent)
        self.store = store
        self.source_file_id = source_file_id
        self.export_directory = Path(export_directory)
        self.model = ContactTableModel(store, source_file_id, self)
        self._loaded_contact_id: int | None = None

        self.status = QLabel("Geometry not prepared")
        self.status.setObjectName("statusPill")

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.selectionModel().selectionChanged.connect(self._load_selection)

        self.thumbnail_label = QLabel("No thumbnail")
        self.thumbnail_label.setObjectName("thumbCard")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(110)
        self.thumbnail_label.setMaximumHeight(150)

        self.name_edit = QLineEdit()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.classification_edit = QComboBox()
        self.classification_edit.setEditable(True)
        self.classification_edit.addItems(self.CLASSIFICATIONS)
        self.classification_edit.setCurrentText("")

        detail_card = QFrame()
        detail_card.setObjectName("detailCard")
        form = QFormLayout(detail_card)
        form.addRow("Name", self.name_edit)
        form.addRow("Notes", self.notes_edit)
        form.addRow("Classification", self.classification_edit)

        self.save_button = QPushButton(" Save")
        self.save_button.setIcon(_icon("save", ACCENT_TEXT_ON))
        self.save_button.setProperty("class", "primary")
        self.delete_button = QPushButton(" Delete")
        self.delete_button.setIcon(_icon("delete", DANGER))
        self.delete_button.setProperty("class", "danger")
        self.export_selected_button = QPushButton(" Export Selected")
        self.export_selected_button.setIcon(_icon("export", TEXT_PRIMARY))
        self.export_all_button = QPushButton(" Export All")
        self.export_all_button.setIcon(_icon("export", TEXT_PRIMARY))
        self.save_button.clicked.connect(self.save_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.export_selected_button.clicked.connect(self.export_selected)
        self.export_all_button.clicked.connect(self.export_all)

        edit_row = QHBoxLayout()
        edit_row.addWidget(self.save_button, 2)
        edit_row.addWidget(self.delete_button, 1)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_selected_button)
        export_row.addWidget(self.export_all_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.status)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.thumbnail_label)
        layout.addWidget(detail_card)
        layout.addLayout(edit_row)
        layout.addLayout(export_row)
        self._set_editor_enabled(False)

    def selected_record(self):
        rows = self.table.selectionModel().selectedRows()
        return self.model.record_at(rows[0].row()) if rows else None

    def refresh(self, *, select_contact_id: int | None = None) -> None:
        row = self.model.refresh(select_contact_id=select_contact_id)
        if row is not None:
            self.table.selectRow(row)
        elif not self.model.records:
            self._clear_editor()

    def refresh_and_focus_name(self, *, select_contact_id: int) -> None:
        self.refresh(select_contact_id=select_contact_id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def set_geometry_status(self, text: str, *, ready: bool = False) -> None:
        color = ACCENT if ready else WARN
        self.status.setText(f"<span style='color:{color};'>●</span>&nbsp;&nbsp;{text}")

    def set_source_file(self, source_file_id: int) -> None:
        self._autosave_if_dirty()
        self.source_file_id = source_file_id
        self._loaded_contact_id = None
        self.model.set_source_file(source_file_id)
        self._clear_editor()

    def discard_pending_edit(self) -> None:
        self._loaded_contact_id = None

    def set_store(self, store: ContactStore, source_file_id: int) -> None:
        self._autosave_if_dirty()
        self._loaded_contact_id = None
        self.store = store
        self.model.store = store
        self.set_source_file(source_file_id)

    def save_selected(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        updated = self.store.update_contact_text(
            record.id,
            name=self.name_edit.text(),
            notes=self.notes_edit.toPlainText(),
            classification=self.classification_edit.currentText().strip() or None,
        )
        self.refresh(select_contact_id=updated.id)
        self.contact_updated.emit(updated.id)

    def delete_selected(self, *, confirm=True) -> None:
        record = self.selected_record()
        if record is None:
            return
        if confirm:
            answer = QMessageBox.question(
                self, "Delete contact", f"Delete {record.draft.name or 'this contact'}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        contact_id = record.id
        self.store.delete_contact(contact_id)
        self.refresh()
        self.contact_deleted.emit(contact_id)

    def export_selected(self) -> None:
        record = self.selected_record()
        if record is not None:
            self._choose_and_export([record])

    def export_all(self) -> None:
        self._choose_and_export(self.model.records)

    def _choose_and_export(self, contacts) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export contacts",
            str(self.export_directory / "contacts.gpx"),
            "GPX files (*.gpx)",
        )
        if not destination:
            return
        path = Path(destination)
        overwrite = False
        if path.exists():
            answer = QMessageBox.question(self, "Overwrite GPX", f"Replace {path.name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        result = GPXExporter().export(contacts, path, overwrite=overwrite)
        self.status.setText(f"Exported {result.exported_count}; skipped {result.skipped_count}")

    def _load_selection(self, selected, deselected) -> None:
        self._autosave_if_dirty()
        record = self.selected_record()
        if record is None:
            self._loaded_contact_id = None
            self._clear_editor()
            return
        self.name_edit.setText(record.draft.name)
        self.notes_edit.setPlainText(record.draft.notes)
        self.classification_edit.setCurrentText(record.draft.classification or "")
        self._loaded_contact_id = record.id
        self._load_thumbnail(record.id)
        self._set_editor_enabled(True)

    def _autosave_if_dirty(self) -> None:
        if self._loaded_contact_id is None:
            return
        try:
            previous = self.store.get_contact(self._loaded_contact_id)
        except KeyError:
            return
        name = self.name_edit.text()
        notes = self.notes_edit.toPlainText()
        classification = self.classification_edit.currentText().strip() or None
        if (
            name == previous.draft.name
            and notes == previous.draft.notes
            and classification == previous.draft.classification
        ):
            return
        try:
            updated = self.store.update_contact_text(
                self._loaded_contact_id, name=name, notes=notes, classification=classification
            )
        except Exception as exc:
            dialog = ErrorWarnDialog(title="Autosave failed", message=str(exc))
            dialog.exec()
            return
        self.model.update_record(updated)
        self.set_geometry_status(f"Saved {updated.draft.name}", ready=True)
        self.contact_updated.emit(updated.id)

    def _load_thumbnail(self, contact_id: int) -> None:
        thumbnail = self.store.get_thumbnail(contact_id)
        if thumbnail is None:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("No thumbnail")
            return
        pixmap = QPixmap()
        pixmap.loadFromData(thumbnail.image_bytes)
        self.thumbnail_label.setText("")
        self.thumbnail_label.setPixmap(
            pixmap.scaled(
                240,
                150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _clear_editor(self) -> None:
        self.name_edit.clear()
        self.notes_edit.clear()
        self.classification_edit.setCurrentText("")
        self.thumbnail_label.setPixmap(QPixmap())
        self.thumbnail_label.setText("No thumbnail")
        self._set_editor_enabled(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for widget in (
            self.name_edit,
            self.notes_edit,
            self.classification_edit,
            self.save_button,
            self.delete_button,
            self.export_selected_button,
        ):
            widget.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class QtContactPickerWindowV2(QMainWindow):
    def __init__(
        self,
        *,
        context: SonarFileContext,
        store: ContactStore,
        picker,
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
        self.setWindowTitle("SidescanTools — Contact Picker (Redesign Preview)")
        self.setWindowIcon(_icon("target", ACCENT, 40))
        self.resize(1640, 960)
        self.setStyleSheet(STYLE_SHEET)

        self.view = WaterfallView()
        self.view.pixel_clicked.connect(self.pick_contact)
        self.view.pixel_hovered.connect(self._show_hover_stats)
        self.view.hover_cleared.connect(self._clear_hover_stats)
        self.hover_stats_label = QLabel("")
        self.hover_stats_label.setObjectName("hoverStats")
        self.statusBar().addPermanentWidget(self.hover_stats_label)

        self.gain_control = GainControl(
            "Overall gain", WaterfallGainModel.MIN_OVERALL_GAIN_DB,
            WaterfallGainModel.MAX_OVERALL_GAIN_DB,
            WaterfallGainModel.DEFAULT_OVERALL_GAIN_DB,
            step=1.0, decimals=1, suffix=" dB",
        )
        self.tvg_spreading_control = GainControl(
            "TVG spreading", WaterfallGainModel.MIN_TVG_SPREADING_DB_PER_DECADE,
            WaterfallGainModel.MAX_TVG_SPREADING_DB_PER_DECADE,
            WaterfallGainModel.DEFAULT_TVG_SPREADING_DB_PER_DECADE,
            step=1.0, decimals=1, suffix=" dB",
        )
        self.tvg_absorption_control = GainControl(
            "TVG absorption", WaterfallGainModel.MIN_TVG_ABSORPTION_DB_PER_M,
            WaterfallGainModel.MAX_TVG_ABSORPTION_DB_PER_M,
            WaterfallGainModel.DEFAULT_TVG_ABSORPTION_DB_PER_M,
            step=0.01, decimals=2, suffix=" dB/m", slider_scale=100,
        )
        self.along_track_control = GainControl(
            "Along-track scale", 0.1, 8.0, WaterfallView.DEFAULT_ALONG_TRACK_SCALE,
            step=0.05, decimals=2, suffix=" px/ping", slider_scale=100,
        )
        self.gain_control.setToolTip("Display-only gain applied uniformly to both channels")
        self.tvg_spreading_control.setToolTip(
            "Time-variable-gain spreading-loss term: dB boost per decade of "
            "range beyond nadir. This is the same kind of range compensation "
            "EdgeTech and other sonar hardware apply during acquisition -- "
            "tune by eye until brightness looks even from near to far range."
        )
        self.tvg_absorption_control.setToolTip(
            "Time-variable-gain absorption term: dB boost per meter of range, "
            "linear. Compensates the frequency-dependent acoustic absorption "
            "component of range loss, on top of the spreading term."
        )
        self.along_track_control.setToolTip(
            "Vertical pixels per ping along the survey track. The waterfall's "
            "width always fits the window; adjust this to stretch or compress "
            "the ping axis so a contact's true shape isn't distorted by "
            "changes in vessel speed."
        )
        self.gain_control.valueChanged.connect(self._overall_gain_changed)
        self.tvg_spreading_control.valueChanged.connect(self._tvg_spreading_changed)
        self.tvg_absorption_control.valueChanged.connect(self._tvg_absorption_changed)
        self.along_track_control.valueChanged.connect(self.view.set_along_track_scale)

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(90)
        self.render_timer.timeout.connect(self.render_gain)

        reset_gain_button = QPushButton(" Reset gain")
        reset_gain_button.setIcon(_icon("refresh", TEXT_SECONDARY))
        reset_gain_button.setProperty("class", "ghost")
        reset_gain_button.clicked.connect(self.reset_gain)
        normalize_tvg_button = QPushButton(" Normalize")
        normalize_tvg_button.setIcon(_icon("activity", ACCENT))
        normalize_tvg_button.setProperty("class", "ghost")
        normalize_tvg_button.setToolTip(
            "Equalize typical brightness across the full swath around 50%. "
            "Automatically adjusts overall gain and TVG, then applies a "
            "smooth residual correction to dark and blown-out areas."
        )
        normalize_tvg_button.clicked.connect(self.normalize_tvg)
        reset_view_button = QPushButton(" Reset view")
        reset_view_button.setIcon(_icon("refresh", TEXT_SECONDARY))
        reset_view_button.setProperty("class", "ghost")
        reset_view_button.clicked.connect(self.reset_view)

        gain_group = QGroupBox("Gain && TVG")
        gain_layout = QVBoxLayout(gain_group)
        gain_layout.setSpacing(10)
        gain_layout.addWidget(self.gain_control)
        gain_layout.addWidget(self.tvg_spreading_control)
        gain_layout.addWidget(self.tvg_absorption_control)
        reset_gain_row = QHBoxLayout()
        reset_gain_row.addStretch(1)
        reset_gain_row.addWidget(normalize_tvg_button)
        reset_gain_row.addWidget(reset_gain_button)
        gain_layout.addLayout(reset_gain_row)

        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)
        view_layout.setSpacing(10)
        view_layout.addWidget(self.along_track_control)
        reset_view_row = QHBoxLayout()
        reset_view_row.addStretch(1)
        reset_view_row.addWidget(reset_view_button)
        view_layout.addLayout(reset_view_row)
        view_layout.addStretch(1)

        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(gain_group, 2)
        toolbar_row.addWidget(view_group, 1)

        self.hint_banner = HintBanner(
            "Left-click a sonar return to save a contact. Drag to pan; scroll "
            "the wheel to move up/down the survey. Width always fits the "
            "window — use “Along-track scale” to fix contact shapes "
            "distorted by vessel-speed changes."
        )

        central = QWidget()
        central.setObjectName("centralArea")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.hint_banner)
        layout.addLayout(toolbar_row)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(central)

        processing_dock = QDockWidget("Processing and Gain", self)
        processing_dock.setObjectName("processingDock")
        processing_dock.setWidget(self._build_processing_panel())
        processing_dock.widget().setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, processing_dock)

        self.contact_dock = ContactDockV2(
            self.store, self.source_file_id, export_directory=self.contacts_db_path.parent
        )
        self.contact_dock.set_geometry_status("Geometry ready", ready=True)

        contacts_dock = QDockWidget("Sonar Contacts", self)
        contacts_dock.setObjectName("contactsDock")
        contacts_dock.setWidget(self._build_contacts_panel())
        contacts_dock.widget().setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, contacts_dock)

        self.contact_dock.contact_deleted.connect(self.refresh_chunk)
        self.contact_dock.contact_updated.connect(self.refresh_chunk)
        self.contact_dock.table.selectionModel().selectionChanged.connect(
            self.center_selected_contact
        )
        self.view.set_image(self.display.render_rgb(), fit=True)
        self.refresh_chunk()
        self._update_file_position()
        self._update_database_label()
        self._update_status()

    # -- panel builders ----------------------------------------------------

    def _build_processing_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        file_card = QFrame()
        file_card.setObjectName("fileCard")
        file_layout = QVBoxLayout(file_card)
        file_layout.setSpacing(6)

        self.file_name_label = QLabel()
        self.file_name_label.setObjectName("fileName")
        self.file_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_name_label.setWordWrap(True)
        self.file_position_label = QLabel()
        self.file_position_label.setObjectName("caption")
        self.file_position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.previous_file_button = QToolButton()
        self.previous_file_button.setObjectName("navBtn")
        self.previous_file_button.setIcon(_icon("chevron-left", TEXT_PRIMARY, 16))
        self.previous_file_button.setToolTip("Previous file in this folder")
        self.previous_file_button.setFixedSize(36, 36)
        self.previous_file_button.clicked.connect(lambda: self._go_to_relative_file(-1))

        self.next_file_button = QToolButton()
        self.next_file_button.setObjectName("navBtnPrimary")
        self.next_file_button.setIcon(_icon("chevron-right", ACCENT_TEXT_ON, 16))
        self.next_file_button.setToolTip(
            "Move to the next .jsf/.xtf file in this folder. Gain, TVG, and "
            "along-track scale carry over; the same contacts database is used."
        )
        self.next_file_button.setFixedSize(36, 36)
        self.next_file_button.clicked.connect(lambda: self._go_to_relative_file(1))

        open_button = QPushButton(" Open…")
        open_button.setIcon(_icon("folder", TEXT_PRIMARY))
        open_button.setProperty("class", "ghost")
        open_button.setToolTip("Open a different sonar file")
        open_button.clicked.connect(self.open_file)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.previous_file_button)
        nav_row.addWidget(open_button, 1)
        nav_row.addWidget(self.next_file_button)

        file_layout.addWidget(self.file_name_label)
        file_layout.addWidget(self.file_position_label)
        file_layout.addLayout(nav_row)
        outer.addWidget(file_card)

        mode_group = QGroupBox("Processing mode")
        mode_form = QFormLayout(mode_group)
        self.processing_mode = QComboBox()
        for label, mode in (
            ("Raw waterfall", BuiltInGainMode.RAW),
            ("Slant-range corrected", BuiltInGainMode.SLANT),
            ("Beam Angle Correction (BAC)", BuiltInGainMode.BAC),
            ("Empirical Gain Normalization (EGN)", BuiltInGainMode.EGN),
        ):
            self.processing_mode.addItem(label, mode.value)
        self.processing_mode.currentIndexChanged.connect(self._processing_mode_changed)
        mode_form.addRow("Mode", self.processing_mode)

        self.nadir_angle = Stepper(minimum=0.0, maximum=89.0, value=0.0, step=1.0, decimals=1, suffix="°")
        mode_form.addRow("Nadir angle", self.nadir_angle)
        outer.addWidget(mode_group)

        egn_group = QGroupBox("Empirical Gain Normalization")
        egn_layout = QVBoxLayout(egn_group)
        self.egn_path = QLineEdit()
        self.egn_path.setPlaceholderText("Select an EGN .npz table")
        self.egn_browse_button = QPushButton("Browse…")
        self.egn_browse_button.setProperty("class", "ghost")
        self.egn_browse_button.clicked.connect(self.browse_egn_table)
        self.egn_build_button = QPushButton("Build…")
        self.egn_build_button.setProperty("class", "ghost")
        self.egn_build_button.setToolTip(
            "Build a new EGN table from sonar files or a whole folder on disk"
        )
        self.egn_build_button.clicked.connect(self.open_egn_table_builder)
        egn_row = QHBoxLayout()
        egn_row.addWidget(self.egn_path, 1)
        egn_row.addWidget(self.egn_browse_button)
        egn_row.addWidget(self.egn_build_button)
        egn_layout.addLayout(egn_row)
        outer.addWidget(egn_group)

        bac_group = QGroupBox("Beam Angle Correction")
        bac_form = QFormLayout(bac_group)
        self.bac_resolution = Stepper(minimum=36, maximum=1440, value=360, step=36, integer=True)
        bac_form.addRow("Angle bins", self.bac_resolution)
        self.energy_normalization = QCheckBox("Normalize ping energy after BAC")
        self.energy_normalization.setChecked(True)
        bac_form.addRow("", self.energy_normalization)
        outer.addWidget(bac_group)

        enhance_group = QGroupBox("Enhancements")
        enhance_layout = QVBoxLayout(enhance_group)
        self.internal_altitude = QCheckBox("Use logged sensor altitude")
        self.convert_db = QCheckBox("Convert processed intensity to dB")
        self.apply_clahe = QCheckBox("Apply adaptive histogram equalization (CLAHE)")
        for checkbox in (self.internal_altitude, self.convert_db, self.apply_clahe):
            enhance_layout.addWidget(checkbox)
        outer.addWidget(enhance_group)

        self.apply_processing_button = QPushButton(" Apply")
        self.apply_processing_button.setIcon(_icon("check", ACCENT_TEXT_ON))
        self.apply_processing_button.setProperty("class", "primary")
        self.apply_processing_button.clicked.connect(self.apply_builtin_processing)
        show_raw_button = QPushButton(" Show raw")
        show_raw_button.setIcon(_icon("eye", TEXT_SECONDARY))
        show_raw_button.setProperty("class", "ghost")
        show_raw_button.clicked.connect(self.show_raw_waterfall)
        processing_buttons = QHBoxLayout()
        processing_buttons.addWidget(self.apply_processing_button, 2)
        processing_buttons.addWidget(show_raw_button, 1)
        outer.addLayout(processing_buttons)

        self.processing_progress = QProgressBar()
        self.processing_progress.setRange(0, 100)
        self.processing_progress.setValue(0)
        self.processing_progress.setTextVisible(False)
        self.processing_status = QLabel("Raw display")
        self.processing_status.setObjectName("caption")
        self.processing_status.setWordWrap(True)
        outer.addWidget(self.processing_progress)
        outer.addWidget(self.processing_status)
        outer.addStretch(1)
        self._processing_mode_changed()
        return panel

    def _build_contacts_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        db_card = QFrame()
        db_card.setObjectName("dbCard")
        db_layout = QVBoxLayout(db_card)
        self.database_label = QLabel()
        self.database_label.setObjectName("dbLabel")
        self.database_label.setWordWrap(True)

        db_buttons = QHBoxLayout()
        new_database_button = QToolButton()
        new_database_button.setIcon(_icon("database", TEXT_PRIMARY))
        new_database_button.setText(" New")
        new_database_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        new_database_button.setProperty("class", "ghost")
        new_database_button.setToolTip(
            "Create a new, empty contacts database and switch to it. The "
            "currently open file is registered into it right away, and any "
            "other file navigated to from here will share it too -- use "
            "this to start a project spanning many survey files."
        )
        new_database_button.clicked.connect(self.new_database)

        open_database_button = QToolButton()
        open_database_button.setIcon(_icon("folder", TEXT_PRIMARY))
        open_database_button.setText(" Open")
        open_database_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        open_database_button.setProperty("class", "ghost")
        open_database_button.setToolTip(
            "Switch to an existing contacts database, e.g. one you built "
            "earlier for this survey -- files from any folder can share it."
        )
        open_database_button.clicked.connect(self.open_database)

        save_database_as_button = QToolButton()
        save_database_as_button.setIcon(_icon("export", TEXT_PRIMARY))
        save_database_as_button.setText(" Save As")
        save_database_as_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        save_database_as_button.setProperty("class", "ghost")
        save_database_as_button.setToolTip(
            "Copy the current database, including every contact in it, to "
            "a new file and switch to using that copy."
        )
        save_database_as_button.clicked.connect(self.save_database_as)

        for button in (new_database_button, open_database_button, save_database_as_button):
            db_buttons.addWidget(button)

        db_layout.addWidget(self.database_label)
        db_layout.addLayout(db_buttons)
        layout.addWidget(db_card)
        layout.addWidget(self.contact_dock, 1)
        return panel

    # -- pure logic, ported unchanged from the classic window --------------

    def reset_view(self) -> None:
        self.along_track_control.setValue(WaterfallView.DEFAULT_ALONG_TRACK_SCALE)
        self.view.verticalScrollBar().setValue(0)

    def _apply_context(self, context: SonarFileContext) -> None:
        self.context = context
        self.filepath = context.filepath
        self.sidescan_file = context.sidescan_file
        self.preprocessor = context.preprocessor
        self.raw_waterfall = context.raw_waterfall
        self.built_in_processor = context.built_in_processor
        self.source_file_id = context.source_file_id

    def open_file(self) -> None:
        start_dir = str(self.filepath.parent)
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open sonar file", start_dir, "Sidescan files (*.jsf *.xtf);;All files (*)"
        )
        if filename:
            self.load_file(Path(filename))

    def _go_to_relative_file(self, offset: int) -> None:
        if not self._directory_files:
            return
        try:
            current_index = self._directory_files.index(self.filepath)
        except ValueError:
            current_index = 0
        new_index = current_index + offset
        if not 0 <= new_index < len(self._directory_files):
            return
        self.load_file(self._directory_files[new_index])

    def _update_file_position(self) -> None:
        try:
            index = self._directory_files.index(self.filepath)
        except ValueError:
            self.file_name_label.setText(self.filepath.name)
            self.file_position_label.setText("Standalone file")
            self.previous_file_button.setEnabled(False)
            self.next_file_button.setEnabled(False)
            return
        total = len(self._directory_files)
        self.file_name_label.setText(self.filepath.name)
        self.file_name_label.setToolTip(str(self.filepath))
        self.file_position_label.setText(f"File {index + 1} of {total}")
        self.previous_file_button.setEnabled(index > 0)
        self.next_file_button.setEnabled(index < total - 1)

    def load_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        self.statusBar().showMessage(f"Loading {filepath.name}…")
        QApplication.processEvents()
        try:
            context = _load_sonar_context(filepath, settings=self.loader_settings, store=self.store)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not open {filepath.name}: {exc}", 8000)
            return

        self._apply_context(context)
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

        self.setWindowTitle(
            f"SidescanTools — Contact Picker (Redesign Preview) — {filepath.name}"
        )
        self.contact_dock.set_source_file(context.source_file_id)
        self.contact_dock.set_geometry_status("Geometry ready", ready=True)
        self.view.set_image(self.display.render_rgb(), fit=True)
        self.refresh_chunk()
        self._update_file_position()
        self._update_status(f"Opened {filepath.name}")

    def _update_database_label(self) -> None:
        self.database_label.setText(
            f"<span style='color:{ACCENT};'>●</span>&nbsp;&nbsp;{self.contacts_db_path.name}"
        )
        self.database_label.setToolTip(str(self.contacts_db_path))

    def _confirm_replace_existing_database(self, path: Path) -> bool:
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
        try:
            new_store = store_factory()
        except Exception as exc:
            QMessageBox.critical(self, "Could not open database", str(exc))
            return
        try:
            context = _register_in_store(self.context, settings=self.loader_settings, store=new_store)
        except Exception as exc:
            new_store.close()
            QMessageBox.critical(self, "Could not open database", str(exc))
            return

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
            self, "New contacts database", str(self.contacts_db_path.parent), "SQLite database (*.sqlite)"
        )
        if not filename:
            return
        path = Path(filename)
        if not self._confirm_replace_existing_database(path):
            return

        if path.resolve() == self.contacts_db_path.resolve():
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
            self, "Save database as", str(self.contacts_db_path.parent), "SQLite database (*.sqlite)"
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.resolve() == self.contacts_db_path.resolve():
            self._update_status("Already using this database")
            return
        if not self._confirm_replace_existing_database(destination):
            return
        try:
            shutil.copy2(self.contacts_db_path, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save database", str(exc))
            return
        self._switch_database(destination, store_factory=lambda: ContactStore(destination))

    def _processing_mode_changed(self, *args) -> None:
        mode = BuiltInGainMode(self.processing_mode.currentData())
        is_egn = mode is BuiltInGainMode.EGN
        is_bac = mode is BuiltInGainMode.BAC
        self.egn_path.setEnabled(is_egn)
        self.egn_browse_button.setEnabled(is_egn)
        self.bac_resolution.setEnabled(is_bac)
        self.energy_normalization.setEnabled(is_bac)

    def open_egn_table_builder(self) -> None:
        dialog = EGNTableBuilderDialog(self, initial_directory=self.filepath.parent)
        dialog.exec()
        if dialog.result_table_path is not None:
            self.egn_path.setText(str(dialog.result_table_path))
            self._load_egn_nadir_angle(dialog.result_table_path)

    def browse_egn_table(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EGN table", str(self.filepath.parent), "NumPy tables (*.npz)"
        )
        if not filename:
            return
        self.egn_path.setText(filename)
        self._load_egn_nadir_angle(Path(filename))

    def _load_egn_nadir_angle(self, table_path: Path) -> None:
        try:
            with np.load(table_path) as table:
                if "nadir_angle" in table:
                    self.nadir_angle.setValue(float(table["nadir_angle"]))
        except Exception:
            pass

    def apply_builtin_processing(self) -> None:
        try:
            mode = BuiltInGainMode(self.processing_mode.currentData())
            egn_path = (
                Path(self.egn_path.text().strip()) if self.egn_path.text().strip() else None
            )
            request = BuiltInGainRequest(
                mode=mode,
                egn_table_path=egn_path,
                bac_angle_count=self.bac_resolution.value(),
                energy_normalization=self.energy_normalization.isChecked(),
                convert_db=self.convert_db.isChecked(),
                clahe=self.apply_clahe.isChecked(),
                nadir_angle=self.nadir_angle.value(),
                use_internal_altitude=self.internal_altitude.isChecked(),
            )
        except Exception as exc:
            self.processing_status.setText(str(exc))
            return

        self.apply_processing_button.setEnabled(False)
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
        self.display.set_source(result.display_data, base_pipeline=result.pipeline_description)
        self.view.set_image(self.display.render_rgb())
        self.processing_progress.setValue(100)
        self.processing_status.setText(
            "Active: " + result.pipeline_description.replace("|", " · ")
        )
        self.apply_processing_button.setEnabled(True)
        self.processing_worker = None
        self._update_status()

    def _processing_failed(self, message: str) -> None:
        self.processing_status.setText(f"Processing failed: {message}")
        self.apply_processing_button.setEnabled(True)
        self.processing_worker = None

    def show_raw_waterfall(self) -> None:
        self.display.set_source(self.raw_waterfall, base_pipeline="qt-continuous-waterfall-v1|raw")
        self.processing_mode.setCurrentIndex(0)
        self.processing_progress.setValue(0)
        self.processing_status.setText("Raw display")
        self.view.set_image(self.display.render_rgb())
        self._update_status()

    def _overall_gain_changed(self, value) -> None:
        self.display.overall_gain_db = float(value)
        self.render_timer.start()

    def _tvg_spreading_changed(self, value) -> None:
        self.display.tvg_spreading_db_per_decade = float(value)
        self.render_timer.start()

    def _tvg_absorption_changed(self, value) -> None:
        self.display.tvg_absorption_db_per_m = float(value)
        self.render_timer.start()

    def reset_gain(self) -> None:
        self.display.clear_normalization()
        self.gain_control.setValue(WaterfallGainModel.DEFAULT_OVERALL_GAIN_DB)
        self.tvg_spreading_control.setValue(WaterfallGainModel.DEFAULT_TVG_SPREADING_DB_PER_DECADE)
        self.tvg_absorption_control.setValue(WaterfallGainModel.DEFAULT_TVG_ABSORPTION_DB_PER_M)
        self.render_gain()

    def normalize_tvg(self) -> None:
        try:
            overall, spreading, absorption = self.display.normalize_tvg()
        except ValueError as error:
            self.statusBar().showMessage(f"TVG normalization failed: {error}")
            return
        self.gain_control.setValue(overall)
        self.tvg_spreading_control.setValue(spreading)
        self.tvg_absorption_control.setValue(absorption)
        self.render_gain()
        self.statusBar().showMessage(
            "Swath brightness equalized around 50%; "
            f"gain {overall:+.1f} dB, "
            f"spreading {spreading:+.1f} dB/decade, "
            f"absorption {absorption:+.2f} dB/m"
        )

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
                column = round((1.0 - anchor.sample_fraction) * (self.preprocessor.ping_len - 1))
            else:
                column = self.preprocessor.ping_len + round(
                    anchor.sample_fraction * (self.preprocessor.ping_len - 1)
                )
            markers.append((anchor.global_ping_index, column, record.draft.name))
        self.view.set_markers(markers)

    def center_selected_contact(self, *args) -> None:
        record = self.contact_dock.selected_record()
        if record is None:
            return
        anchor = record.draft.anchor
        if anchor.channel.value == 0:
            column = round((1.0 - anchor.sample_fraction) * (self.preprocessor.ping_len - 1))
        else:
            column = self.preprocessor.ping_len + round(
                anchor.sample_fraction * (self.preprocessor.ping_len - 1)
            )
        self.view.center_on(anchor.global_ping_index, column)

    def pick_contact(self, row: int, column: int) -> None:
        chunk_index, local_ping_index = divmod(row, self.preprocessor.chunk_size)
        try:
            result = self.picker.pick_display_pixel(
                chunk_index=chunk_index, local_ping_index=local_ping_index, display_x=column
            )
        except DuplicateContactAnchor:
            self.statusBar().showMessage("A contact already exists at this sonar sample", 5000)
            return
        except Exception as exc:
            self.statusBar().showMessage(f"Contact not saved: {exc}", 8000)
            return
        self.contact_dock.refresh_and_focus_name(select_contact_id=result.contact.id)
        self.refresh_chunk()
        self._update_status(result.thumbnail_warning or f"Saved {result.contact.draft.name}")

    def closeEvent(self, event) -> None:
        self.store.close()
        super().closeEvent(event)


def run_qt_contact_picker_v2(
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
    """Open the redesigned contact picker. Same backend/arguments as
    ``run_qt_contact_picker`` -- only the window class differs."""

    application = QApplication.instance() or QApplication([])
    QApplication.setStyle("Fusion")

    if filepath is None:
        selected, _ = QFileDialog.getOpenFileName(
            None, "Open sonar file", "", "Sidescan files (*.jsf *.xtf);;All files (*)"
        )
        if not selected:
            return None
        filepath = selected

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

    window = QtContactPickerWindowV2(
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
