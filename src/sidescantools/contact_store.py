"""SQLite persistence and numbered migrations for sonar contacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterable

from sidescantools.contact_model import (
    Channel,
    ContactAnchor,
    ContactCoordinate,
    ContactDraft,
    ContactRecord,
    ContactThumbnail,
    CoordinateStatus,
)
from sidescantools.swath_geometry import GeometrySettings


class ContactStoreError(RuntimeError):
    pass


class DuplicateContactAnchor(ContactStoreError):
    """Raised when an exact acoustic anchor is already stored."""


@dataclass(frozen=True, slots=True)
class SourceFileRecord:
    id: int
    canonical_path: str
    display_path: str
    format: str
    subsystem_index: int
    ping_count: int
    source_sample_count: int
    file_size_bytes: int | None
    mtime_ns: int | None
    content_fingerprint: str | None


_MIGRATION_1 = (
    """
    CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE source_files (
        id INTEGER PRIMARY KEY,
        canonical_path TEXT NOT NULL,
        display_path TEXT NOT NULL,
        file_size_bytes INTEGER,
        mtime_ns INTEGER,
        content_fingerprint TEXT,
        format TEXT NOT NULL,
        subsystem_index INTEGER NOT NULL DEFAULT 0,
        ping_count INTEGER NOT NULL,
        source_sample_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        UNIQUE(canonical_path, subsystem_index)
    )
    """,
    """
    CREATE TABLE geometry_profiles (
        id INTEGER PRIMARY KEY,
        settings_json TEXT NOT NULL,
        settings_hash TEXT NOT NULL UNIQUE,
        algorithm_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE contacts (
        id INTEGER PRIMARY KEY,
        uuid TEXT NOT NULL UNIQUE,
        source_file_id INTEGER NOT NULL REFERENCES source_files(id),
        geometry_profile_id INTEGER REFERENCES geometry_profiles(id),
        global_ping_index INTEGER NOT NULL CHECK(global_ping_index >= 0),
        ping_number INTEGER,
        channel_index INTEGER NOT NULL CHECK(channel_index IN (0, 1)),
        channel_name TEXT NOT NULL CHECK(channel_name IN ('port', 'starboard')),
        source_sample_index INTEGER NOT NULL CHECK(source_sample_index >= 0),
        sample_fraction REAL NOT NULL
            CHECK(sample_fraction >= 0.0 AND sample_fraction <= 1.0),
        display_chunk INTEGER,
        display_ping_index INTEGER,
        display_sample_index INTEGER,
        longitude REAL,
        latitude REAL,
        coordinate_status TEXT NOT NULL DEFAULT 'valid'
            CHECK(coordinate_status IN ('valid', 'stale', 'error', 'unavailable')),
        coordinate_error TEXT,
        timestamp_iso TEXT,
        timestamp_basis TEXT,
        slant_range_m REAL,
        ground_range_m REAL,
        intensity_source REAL,
        intensity_display REAL,
        intensity_pipeline TEXT,
        name TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        classification TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(source_file_id, global_ping_index, channel_index, source_sample_index)
    )
    """,
    "CREATE INDEX idx_contacts_source_ping ON contacts(source_file_id, global_ping_index)",
    "CREATE INDEX idx_contacts_name ON contacts(name)",
    "CREATE INDEX idx_contacts_coordinate_status ON contacts(coordinate_status)",
    """
    CREATE TABLE contact_thumbnails (
        contact_id INTEGER PRIMARY KEY REFERENCES contacts(id) ON DELETE CASCADE,
        mime_type TEXT NOT NULL DEFAULT 'image/png',
        image_bytes BLOB NOT NULL,
        width_px INTEGER NOT NULL,
        height_px INTEGER NOT NULL,
        ping_radius INTEGER NOT NULL,
        sample_radius INTEGER NOT NULL,
        display_pipeline TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE contact_coordinate_history (
        id INTEGER PRIMARY KEY,
        contact_id INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
        geometry_profile_id INTEGER REFERENCES geometry_profiles(id),
        longitude REAL,
        latitude REAL,
        coordinate_status TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
    """,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ContactStore:
    """Project-local contact repository with explicit transactional writes."""

    latest_schema_version = 1

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=busy_timeout_ms / 1000)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        try:
            self._migrate()
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        self.connection.close()

    @property
    def schema_version(self) -> int:
        row = self.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return 0 if row is None else int(row["version"])

    def _migrate(self) -> None:
        has_migrations = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        version = self.schema_version if has_migrations else 0
        if version > self.latest_schema_version:
            raise ContactStoreError(
                f"database schema {version} is newer than supported "
                f"schema {self.latest_schema_version}"
            )
        if version == 0:
            applied_at = _utc_now()
            try:
                with self.connection:
                    for statement in _MIGRATION_1:
                        self.connection.execute(statement)
                    self.connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (1, applied_at),
                    )
            except sqlite3.DatabaseError as exc:
                raise ContactStoreError("failed to apply contact schema migration 1") from exc

    def register_source_file(
        self,
        path: str | Path,
        *,
        format: str,
        ping_count: int,
        source_sample_count: int,
        subsystem_index: int = 0,
        file_size_bytes: int | None = None,
        mtime_ns: int | None = None,
        content_fingerprint: str | None = None,
    ) -> SourceFileRecord:
        display_path = str(Path(path))
        canonical_path = str(Path(path).resolve(strict=False))
        now = _utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO source_files(
                    canonical_path, display_path, file_size_bytes, mtime_ns,
                    content_fingerprint, format, subsystem_index, ping_count,
                    source_sample_count, created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_path, subsystem_index) DO UPDATE SET
                    display_path=excluded.display_path,
                    file_size_bytes=excluded.file_size_bytes,
                    mtime_ns=excluded.mtime_ns,
                    content_fingerprint=excluded.content_fingerprint,
                    format=excluded.format,
                    ping_count=excluded.ping_count,
                    source_sample_count=excluded.source_sample_count,
                    last_seen_at=excluded.last_seen_at
                """,
                (
                    canonical_path,
                    display_path,
                    file_size_bytes,
                    mtime_ns,
                    content_fingerprint,
                    format.lower(),
                    subsystem_index,
                    ping_count,
                    source_sample_count,
                    now,
                    now,
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM source_files WHERE canonical_path=? AND subsystem_index=?",
            (canonical_path, subsystem_index),
        ).fetchone()
        return self._source_from_row(row)

    def get_or_create_geometry_profile(self, settings: GeometrySettings) -> int:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO geometry_profiles(
                    settings_json, settings_hash, algorithm_version, created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(settings_hash) DO NOTHING
                """,
                (
                    settings.to_json(),
                    settings.settings_hash,
                    settings.geometry_algorithm_version,
                    _utc_now(),
                ),
            )
        row = self.connection.execute(
            "SELECT id FROM geometry_profiles WHERE settings_hash=?",
            (settings.settings_hash,),
        ).fetchone()
        return int(row["id"])

    def geometry_profile_hash(self, geometry_profile_id: int) -> str:
        row = self.connection.execute(
            "SELECT settings_hash FROM geometry_profiles WHERE id=?",
            (geometry_profile_id,),
        ).fetchone()
        if row is None:
            raise KeyError(geometry_profile_id)
        return str(row["settings_hash"])

    def create_contact(
        self,
        draft: ContactDraft,
        thumbnail: ContactThumbnail | None = None,
    ) -> ContactRecord:
        now = _utc_now()
        anchor = draft.anchor
        coordinate = draft.coordinate
        try:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    INSERT INTO contacts(
                        uuid, source_file_id, geometry_profile_id,
                        global_ping_index, ping_number, channel_index, channel_name,
                        source_sample_index, sample_fraction, display_chunk,
                        display_ping_index, display_sample_index, longitude, latitude,
                        coordinate_status, timestamp_iso, timestamp_basis,
                        slant_range_m, ground_range_m, intensity_source,
                        intensity_display, intensity_pipeline, name, notes,
                        classification, created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'valid', ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        draft.uuid,
                        anchor.source_file_id,
                        coordinate.geometry_profile_id,
                        anchor.global_ping_index,
                        anchor.ping_number,
                        int(anchor.channel),
                        anchor.channel.label,
                        anchor.source_sample_index,
                        anchor.sample_fraction,
                        anchor.display_chunk,
                        anchor.display_ping_index,
                        anchor.display_sample_index,
                        coordinate.longitude,
                        coordinate.latitude,
                        draft.timestamp_iso,
                        draft.timestamp_basis,
                        coordinate.slant_range_m,
                        coordinate.ground_range_m,
                        draft.intensity_source,
                        draft.intensity_display,
                        draft.intensity_pipeline,
                        draft.name,
                        draft.notes,
                        draft.classification,
                        now,
                        now,
                    ),
                )
                contact_id = int(cursor.lastrowid)
                if thumbnail is not None:
                    self.connection.execute(
                        """
                        INSERT INTO contact_thumbnails(
                            contact_id, mime_type, image_bytes, width_px, height_px,
                            ping_radius, sample_radius, display_pipeline, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            contact_id,
                            thumbnail.mime_type,
                            thumbnail.image_bytes,
                            thumbnail.width_px,
                            thumbnail.height_px,
                            thumbnail.ping_radius,
                            thumbnail.sample_radius,
                            thumbnail.display_pipeline,
                            now,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "contacts.source_file_id" in message or "contacts.uuid" in message:
                raise DuplicateContactAnchor("contact anchor or UUID already exists") from exc
            raise ContactStoreError("contact could not be saved") from exc
        return self.get_contact(contact_id)

    def next_default_contact_name(self) -> str:
        next_id = self.connection.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM contacts"
        ).fetchone()[0]
        return f"Target {int(next_id):04d}"

    def get_contact(self, contact_id: int) -> ContactRecord:
        row = self.connection.execute(
            """
            SELECT contacts.*, source_files.display_path AS source_display_path,
                   geometry_profiles.settings_hash AS geometry_settings_hash
            FROM contacts
            JOIN source_files ON source_files.id = contacts.source_file_id
            LEFT JOIN geometry_profiles
                ON geometry_profiles.id = contacts.geometry_profile_id
            WHERE contacts.id=?
            """,
            (contact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(contact_id)
        return self._contact_from_row(row)

    def list_contacts(
        self,
        *,
        source_file_id: int | None = None,
        statuses: Iterable[CoordinateStatus | str] | None = None,
    ) -> list[ContactRecord]:
        clauses = []
        parameters: list[object] = []
        if source_file_id is not None:
            clauses.append("source_file_id=?")
            parameters.append(source_file_id)
        if statuses is not None:
            values = [CoordinateStatus(status).value for status in statuses]
            if not values:
                return []
            placeholders = ",".join(["?"] * len(values))
            clauses.append(f"coordinate_status IN ({placeholders})")
            parameters.extend(values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            "SELECT contacts.*, source_files.display_path AS source_display_path, "
            "geometry_profiles.settings_hash AS geometry_settings_hash "
            "FROM contacts "
            "JOIN source_files ON source_files.id = contacts.source_file_id "
            "LEFT JOIN geometry_profiles "
            "ON geometry_profiles.id = contacts.geometry_profile_id"
            + where
            + " ORDER BY contacts.source_file_id, contacts.global_ping_index, contacts.id",
            parameters,
        ).fetchall()
        return [self._contact_from_row(row) for row in rows]

    def update_contact_text(
        self,
        contact_id: int,
        *,
        name: str,
        notes: str,
        classification: str | None,
    ) -> ContactRecord:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE contacts
                SET name=?, notes=?, classification=?, updated_at=?
                WHERE id=?
                """,
                (name, notes, classification, _utc_now(), contact_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(contact_id)
        return self.get_contact(contact_id)

    def mark_stale_for_profile(
        self, source_file_id: int, active_geometry_profile_id: int
    ) -> int:
        """Mark valid derived coordinates made with another profile as stale."""

        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE contacts
                SET coordinate_status='stale', updated_at=?
                WHERE source_file_id=?
                  AND coordinate_status='valid'
                  AND (
                      geometry_profile_id IS NULL
                      OR geometry_profile_id != ?
                  )
                """,
                (_utc_now(), source_file_id, active_geometry_profile_id),
            )
        return cursor.rowcount

    def recompute_contact(
        self, contact_id: int, coordinate: ContactCoordinate
    ) -> ContactRecord:
        """Replace derived geometry while preserving and historizing the anchor."""

        now = _utc_now()
        with self.connection:
            previous = self.connection.execute(
                "SELECT * FROM contacts WHERE id=?", (contact_id,)
            ).fetchone()
            if previous is None:
                raise KeyError(contact_id)
            self._insert_coordinate_history(previous, now)
            self.connection.execute(
                """
                UPDATE contacts
                SET geometry_profile_id=?, longitude=?, latitude=?,
                    slant_range_m=?, ground_range_m=?, coordinate_status='valid',
                    coordinate_error=NULL, updated_at=?
                WHERE id=?
                """,
                (
                    coordinate.geometry_profile_id,
                    coordinate.longitude,
                    coordinate.latitude,
                    coordinate.slant_range_m,
                    coordinate.ground_range_m,
                    now,
                    contact_id,
                ),
            )
        return self.get_contact(contact_id)

    def record_recompute_error(self, contact_id: int, error: str) -> ContactRecord:
        """Retain the last coordinate and record a concise recomputation error."""

        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must not be blank")
        now = _utc_now()
        with self.connection:
            previous = self.connection.execute(
                "SELECT * FROM contacts WHERE id=?", (contact_id,)
            ).fetchone()
            if previous is None:
                raise KeyError(contact_id)
            self._insert_coordinate_history(previous, now)
            self.connection.execute(
                """
                UPDATE contacts
                SET coordinate_status='error', coordinate_error=?, updated_at=?
                WHERE id=?
                """,
                (error.strip(), now, contact_id),
            )
        return self.get_contact(contact_id)

    def list_coordinate_history(self, contact_id: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM contact_coordinate_history
            WHERE contact_id=? ORDER BY id
            """,
            (contact_id,),
        ).fetchall()

    def delete_contact(self, contact_id: int) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM contacts WHERE id=?", (contact_id,)
            )
        if cursor.rowcount != 1:
            raise KeyError(contact_id)

    def get_thumbnail(self, contact_id: int) -> ContactThumbnail | None:
        row = self.connection.execute(
            "SELECT * FROM contact_thumbnails WHERE contact_id=?", (contact_id,)
        ).fetchone()
        if row is None:
            return None
        return ContactThumbnail(
            image_bytes=bytes(row["image_bytes"]),
            width_px=row["width_px"],
            height_px=row["height_px"],
            ping_radius=row["ping_radius"],
            sample_radius=row["sample_radius"],
            mime_type=row["mime_type"],
            display_pipeline=row["display_pipeline"],
        )

    def _insert_coordinate_history(self, previous: sqlite3.Row, recorded_at: str) -> None:
        self.connection.execute(
            """
            INSERT INTO contact_coordinate_history(
                contact_id, geometry_profile_id, longitude, latitude,
                coordinate_status, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                previous["id"],
                previous["geometry_profile_id"],
                previous["longitude"],
                previous["latitude"],
                previous["coordinate_status"],
                recorded_at,
            ),
        )

    @staticmethod
    def _source_from_row(row: sqlite3.Row) -> SourceFileRecord:
        return SourceFileRecord(
            id=row["id"],
            canonical_path=row["canonical_path"],
            display_path=row["display_path"],
            format=row["format"],
            subsystem_index=row["subsystem_index"],
            ping_count=row["ping_count"],
            source_sample_count=row["source_sample_count"],
            file_size_bytes=row["file_size_bytes"],
            mtime_ns=row["mtime_ns"],
            content_fingerprint=row["content_fingerprint"],
        )

    @staticmethod
    def _contact_from_row(row: sqlite3.Row) -> ContactRecord:
        anchor = ContactAnchor(
            source_file_id=row["source_file_id"],
            global_ping_index=row["global_ping_index"],
            ping_number=row["ping_number"],
            channel=Channel(row["channel_index"]),
            source_sample_index=row["source_sample_index"],
            sample_fraction=row["sample_fraction"],
            display_chunk=row["display_chunk"],
            display_ping_index=row["display_ping_index"],
            display_sample_index=row["display_sample_index"],
        )
        coordinate = ContactCoordinate(
            longitude=row["longitude"],
            latitude=row["latitude"],
            slant_range_m=row["slant_range_m"],
            ground_range_m=row["ground_range_m"],
            geometry_profile_id=row["geometry_profile_id"],
        )
        draft = ContactDraft(
            anchor=anchor,
            coordinate=coordinate,
            name=row["name"],
            notes=row["notes"],
            classification=row["classification"],
            timestamp_iso=row["timestamp_iso"],
            timestamp_basis=row["timestamp_basis"],
            intensity_source=row["intensity_source"],
            intensity_display=row["intensity_display"],
            intensity_pipeline=row["intensity_pipeline"],
            uuid=row["uuid"],
        )
        return ContactRecord(
            id=row["id"],
            draft=draft,
            coordinate_status=CoordinateStatus(row["coordinate_status"]),
            coordinate_error=row["coordinate_error"],
            source_display_path=row["source_display_path"],
            geometry_settings_hash=row["geometry_settings_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
