from pathlib import Path

from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactDraft,
)
from sidescantools.contact_store import ContactStore
from sidescantools.contact_ui import ContactDock
from sidescantools.swath_geometry import GeometrySettings
from qtpy.QtWidgets import QLineEdit


def create_contact(
    store,
    source_id,
    profile_id,
    *,
    global_ping_index=3,
    name="Target 0001",
    thumbnail=None,
):
    return store.create_contact(
        ContactDraft(
            anchor=ContactAnchor(
                source_file_id=source_id,
                global_ping_index=global_ping_index,
                ping_number=1000 + global_ping_index,
                channel=Channel.PORT,
                source_sample_index=4,
                sample_fraction=0.5,
                display_chunk=0,
                display_ping_index=global_ping_index,
                display_sample_index=4,
            ),
            coordinate=ContactCoordinate(-87.1, 30.1, 10, 8, profile_id),
            name=name,
            notes="initial",
        ),
        thumbnail,
    )


def test_contact_dock_edits_refreshes_and_deletes(qtbot, tmp_path):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        contact = create_contact(store, source.id, profile_id)
        dock = ContactDock(
            store,
            source.id,
            export_directory=tmp_path,
        )
        qtbot.addWidget(dock)

        assert dock.model.rowCount() == 1
        dock.table.selectRow(0)
        assert dock.name_edit.text() == "Target 0001"

        dock.name_edit.setText("Wreck candidate")
        dock.notes_edit.setText("reviewed")
        dock.save_selected()

        assert store.get_contact(contact.id).draft.name == "Wreck candidate"
        assert dock.model.data(dock.model.index(0, 0)) == "Wreck candidate"

        dock.delete_selected(confirm=False)

        assert dock.model.rowCount() == 0
        assert store.list_contacts() == []


def test_waypoint_exports_have_their_own_dark_outlined_group(qtbot, tmp_path):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)

        assert dock.waypoint_export_group.title() == "Waypoint Export"
        assert "border: 2px solid #111" in dock.waypoint_export_group.styleSheet()
        assert dock.export_selected_button in dock.waypoint_export_group.findChildren(
            type(dock.export_selected_button)
        )
        assert dock.export_all_button in dock.waypoint_export_group.findChildren(
            type(dock.export_all_button)
        )


def test_contact_list_has_dark_outline_single_line_notes_and_no_thumbnail(
    qtbot, tmp_path
):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)

        assert dock.contact_list_group.title() == "Contact List"
        assert "border: 2px solid #111" in dock.contact_list_group.styleSheet()
        assert isinstance(dock.notes_edit, QLineEdit)
        assert not hasattr(dock, "thumbnail_label")


def test_contact_dock_autosaves_edit_before_switching_selection(qtbot, tmp_path):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        first = create_contact(store, source.id, profile_id)
        create_contact(
            store, source.id, profile_id, global_ping_index=6, name="Target 0002"
        )
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)

        dock.table.selectRow(0)
        assert dock.name_edit.text() == "Target 0001"
        dock.notes_edit.setText("edited but never pressed Save")

        # Switching rows without an explicit Save must not discard the edit.
        dock.table.selectRow(1)

        assert store.get_contact(first.id).draft.notes == "edited but never pressed Save"
        assert "Saved" in dock.status.text()

        # The table itself should reflect the autosave without a manual refresh.
        dock.table.selectRow(0)
        assert dock.notes_edit.text() == "edited but never pressed Save"


def test_contact_dock_does_not_resave_unchanged_selection(qtbot, tmp_path):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        first = create_contact(store, source.id, profile_id)
        create_contact(
            store, source.id, profile_id, global_ping_index=6, name="Target 0002"
        )
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)

        dock.table.selectRow(0)
        before = store.get_contact(first.id).updated_at
        dock.table.selectRow(1)
        dock.table.selectRow(0)

        assert store.get_contact(first.id).updated_at == before


def test_refresh_and_focus_name_selects_the_default_name(qtbot, tmp_path):
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf",
            format="jsf",
            ping_count=10,
            source_sample_count=9,
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        contact = create_contact(store, source.id, profile_id)
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)
        dock.show()
        qtbot.waitExposed(dock)

        dock.refresh_and_focus_name(select_contact_id=contact.id)

        assert dock.name_edit.selectedText() == "Target 0001"
        qtbot.waitUntil(lambda: dock.name_edit.hasFocus(), timeout=2000)


def test_discard_pending_edit_drops_unsaved_changes_without_saving(qtbot, tmp_path):
    # Used right before a database is about to be destroyed (e.g. "New
    # Database" overwriting the file currently open) -- flushing a pending
    # edit into data that's about to be wiped is pointless, and touching
    # the store at that point may not even be safe.
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source = store.register_source_file(
            tmp_path / "survey.jsf", format="jsf", ping_count=10, source_sample_count=9
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        contact = create_contact(store, source.id, profile_id)
        dock = ContactDock(store, source.id, export_directory=tmp_path)
        qtbot.addWidget(dock)
        dock.table.selectRow(0)
        dock.notes_edit.setText("typed but never saved")

        dock.discard_pending_edit()

        assert store.get_contact(contact.id).draft.notes == "initial"
        # Nothing should try to autosave it later either -- a subsequent
        # refresh (which normally flushes a dirty edit first) must not
        # resurrect the discarded text.
        dock.model.refresh()
        assert store.get_contact(contact.id).draft.notes == "initial"


def test_set_source_file_autosaves_pending_edit_then_switches_scope(qtbot, tmp_path):
    # Models the "Next file" navigation in the Qt picker: the dock is
    # re-pointed at a different file's contacts mid-session, and any unsaved
    # edit on the file being left must not be silently lost.
    with ContactStore(tmp_path / "contacts.sqlite") as store:
        source_a = store.register_source_file(
            tmp_path / "line_a.jsf", format="jsf", ping_count=10, source_sample_count=9
        )
        source_b = store.register_source_file(
            tmp_path / "line_b.jsf", format="jsf", ping_count=10, source_sample_count=9
        )
        profile_id = store.get_or_create_geometry_profile(GeometrySettings(60))
        contact_a = create_contact(store, source_a.id, profile_id)
        create_contact(store, source_b.id, profile_id, name="Target 0001 on B")

        dock = ContactDock(store, source_a.id, export_directory=tmp_path)
        qtbot.addWidget(dock)
        dock.table.selectRow(0)
        dock.notes_edit.setText("edited on file A, never pressed Save")

        dock.set_source_file(source_b.id)

        # The edit on A's contact must have been flushed before the switch.
        assert store.get_contact(contact_a.id).draft.notes == (
            "edited on file A, never pressed Save"
        )
        # The dock is now scoped to B's contacts, not A's.
        assert dock.source_file_id == source_b.id
        assert dock.model.rowCount() == 1
        assert dock.model.record_at(0).draft.name == "Target 0001 on B"


def test_set_store_autosaves_then_switches_to_a_different_database(qtbot, tmp_path):
    # Models "New Database.../Open Database..." in the Qt picker: the whole
    # dock is re-pointed at a different database *file*, not just a
    # different file's contacts within the same one.
    with ContactStore(tmp_path / "a.sqlite") as store_a, ContactStore(
        tmp_path / "b.sqlite"
    ) as store_b:
        source_a = store_a.register_source_file(
            tmp_path / "line.jsf", format="jsf", ping_count=10, source_sample_count=9
        )
        source_b = store_b.register_source_file(
            tmp_path / "line.jsf", format="jsf", ping_count=10, source_sample_count=9
        )
        profile_a = store_a.get_or_create_geometry_profile(GeometrySettings(60))
        profile_b = store_b.get_or_create_geometry_profile(GeometrySettings(60))
        contact_a = create_contact(store_a, source_a.id, profile_a)
        create_contact(store_b, source_b.id, profile_b, name="Target in B")

        dock = ContactDock(store_a, source_a.id, export_directory=tmp_path)
        qtbot.addWidget(dock)
        dock.table.selectRow(0)
        dock.notes_edit.setText("edited in database A, never pressed Save")

        dock.set_store(store_b, source_b.id)

        # The edit in A must have been flushed before switching away from it.
        assert store_a.get_contact(contact_a.id).draft.notes == (
            "edited in database A, never pressed Save"
        )
        # The dock -- and its table model -- now both point at database B.
        assert dock.store is store_b
        assert dock.model.store is store_b
        assert dock.source_file_id == source_b.id
        assert dock.model.rowCount() == 1
        assert dock.model.record_at(0).draft.name == "Target in B"
