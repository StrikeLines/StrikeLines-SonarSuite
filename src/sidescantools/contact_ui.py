"""Qt model/view components for contact management in Napari."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sidescantools.contact_export import GPXExporter
from sidescantools.contact_store import ContactStore
from sidescantools.custom_widgets import ErrorWarnDialog


class ContactTableModel(QAbstractTableModel):
    columns = (
        ("Name", lambda record: record.draft.name),
        ("Source", lambda record: Path(record.source_display_path or "").name),
        ("Ping", lambda record: record.draft.anchor.ping_number),
        ("Side", lambda record: record.draft.anchor.channel.label),
        ("Latitude", lambda record: f"{record.draft.coordinate.latitude:.8f}"),
        ("Longitude", lambda record: f"{record.draft.coordinate.longitude:.8f}"),
        ("Status", lambda record: record.coordinate_status.value),
    )

    def __init__(self, store: ContactStore, source_file_id: int, parent=None):
        super().__init__(parent)
        self.store = store
        self.source_file_id = source_file_id
        self.records = []
        self.refresh()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.columns[index.column()][1](self.records[index.row()])
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section][0]
        return super().headerData(section, orientation, role)

    def set_source_file(self, source_file_id: int) -> None:
        self.source_file_id = source_file_id
        self.refresh()

    def refresh(self, *, select_contact_id: int | None = None) -> int | None:
        self.beginResetModel()
        self.records = self.store.list_contacts(source_file_id=self.source_file_id)
        self.endResetModel()
        if select_contact_id is None:
            return None
        return next(
            (
                row
                for row, record in enumerate(self.records)
                if record.id == select_contact_id
            ),
            None,
        )

    def record_at(self, row: int):
        return self.records[row] if 0 <= row < len(self.records) else None

    def update_record(self, record) -> None:
        """Patch one already-loaded row in place, without resetting selection."""

        for row, existing in enumerate(self.records):
            if existing.id == record.id:
                self.records[row] = record
                self.dataChanged.emit(
                    self.index(row, 0), self.index(row, self.columnCount() - 1)
                )
                return


class ContactDock(QWidget):
    """Initial contact list/editor dock backed directly by ``ContactStore``."""

    contact_updated = Signal(int)
    contact_deleted = Signal(int)

    def __init__(
        self,
        store: ContactStore,
        source_file_id: int,
        *,
        export_directory: str | Path,
        parent=None,
    ):
        super().__init__(parent)
        self.store = store
        self.source_file_id = source_file_id
        self.export_directory = Path(export_directory)
        self.model = ContactTableModel(store, source_file_id, self)
        self._loaded_contact_id: int | None = None

        self.status = QLabel("Geometry not prepared")
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.selectionModel().selectionChanged.connect(self._load_selection)

        self.thumbnail_label = QLabel("No thumbnail")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(150)
        self.thumbnail_label.setStyleSheet("border: 1px solid palette(mid);")

        self.name_edit = QLineEdit()
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(90)
        self.classification_edit = QLineEdit()
        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Notes", self.notes_edit)
        form.addRow("Classification", self.classification_edit)

        self.save_button = QPushButton("Save")
        self.delete_button = QPushButton("Delete")
        self.export_selected_button = QPushButton("Export Selected")
        self.export_all_button = QPushButton("Export All")
        self.save_button.clicked.connect(self.save_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.export_selected_button.clicked.connect(self.export_selected)
        self.export_all_button.clicked.connect(self.export_all)
        buttons = QHBoxLayout()
        for button in (
            self.save_button,
            self.delete_button,
            self.export_selected_button,
            self.export_all_button,
        ):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.table)
        layout.addWidget(self.thumbnail_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
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
        """Select a just-picked contact and put the cursor in its name field.

        Used after a fresh pick so the processor can type a real name
        immediately, without an extra click into the field first.
        """

        self.refresh(select_contact_id=select_contact_id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def set_geometry_status(self, text: str, *, ready: bool = False) -> None:
        self.status.setText(text)
        self.status.setProperty("ready", ready)

    def set_source_file(self, source_file_id: int) -> None:
        """Point this dock at a different file's contacts (e.g. after the
        viewer navigates to the next survey file).

        Explicitly flushes any unsaved edit on the previously active contact
        first -- a model reset does clear the table's selection, but doesn't
        reliably emit selectionChanged synchronously, so it can't be relied
        on to trigger _load_selection()'s autosave path the way switching
        rows within one file does.
        """

        self._autosave_if_dirty()
        self.source_file_id = source_file_id
        self._loaded_contact_id = None
        self.model.set_source_file(source_file_id)
        self._clear_editor()

    def discard_pending_edit(self) -> None:
        """Forget any unsaved edit on the currently loaded contact without
        persisting it -- used right before the database backing it is about
        to be destroyed (e.g. "New Database" overwriting the file currently
        open), where flushing it first would be pointless and touching
        self.store at that point may no longer be safe.
        """

        self._loaded_contact_id = None

    def set_store(self, store: ContactStore, source_file_id: int) -> None:
        """Point this dock at a different database file entirely -- e.g. the
        processor switched the whole session's active contacts database --
        not just a different file's contacts within the same database.

        Must flush any pending edit to the OLD store before self.store is
        reassigned: _autosave_if_dirty() always saves through self.store, so
        just delegating straight to set_source_file() here would flush (or
        worse, misdirect, if the contact id happens to collide) the edit
        into the *new* database instead of the one it belongs to.
        """

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
            classification=self.classification_edit.text().strip() or None,
        )
        self.refresh(select_contact_id=updated.id)
        self.contact_updated.emit(updated.id)

    def delete_selected(self, *, confirm=True) -> None:
        record = self.selected_record()
        if record is None:
            return
        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete contact",
                f"Delete {record.draft.name or 'this contact'}?",
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
            answer = QMessageBox.question(
                self, "Overwrite GPX", f"Replace {path.name}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        result = GPXExporter().export(contacts, path, overwrite=overwrite)
        self.status.setText(
            f"Exported {result.exported_count}; skipped {result.skipped_count}"
        )

    def _load_selection(self, selected, deselected) -> None:
        self._autosave_if_dirty()
        record = self.selected_record()
        if record is None:
            self._loaded_contact_id = None
            self._clear_editor()
            return
        self.name_edit.setText(record.draft.name)
        self.notes_edit.setPlainText(record.draft.notes)
        self.classification_edit.setText(record.draft.classification or "")
        self._loaded_contact_id = record.id
        self._load_thumbnail(record.id)
        self._set_editor_enabled(True)

    def _autosave_if_dirty(self) -> None:
        """Persist edits on the previously loaded contact before it's replaced.

        ``_load_selection`` overwrites the name/notes/classification fields
        every time the table selection changes (including programmatically,
        e.g. after a new pick). Without this, an edit the user hasn't
        explicitly saved is silently discarded the moment they click another
        row -- unacceptable for a workflow built around carefully labeling
        one contact at a time.
        """

        if self._loaded_contact_id is None:
            return
        try:
            previous = self.store.get_contact(self._loaded_contact_id)
        except KeyError:
            return
        name = self.name_edit.text()
        notes = self.notes_edit.toPlainText()
        classification = self.classification_edit.text().strip() or None
        if (
            name == previous.draft.name
            and notes == previous.draft.notes
            and classification == previous.draft.classification
        ):
            return
        try:
            updated = self.store.update_contact_text(
                self._loaded_contact_id,
                name=name,
                notes=notes,
                classification=classification,
            )
        except Exception as exc:
            dialog = ErrorWarnDialog(title="Autosave failed", message=str(exc))
            dialog.exec()
            return
        self.model.update_record(updated)
        self.status.setText(f"Saved {updated.draft.name}")
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
                280,
                280,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _clear_editor(self) -> None:
        self.name_edit.clear()
        self.notes_edit.clear()
        self.classification_edit.clear()
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
