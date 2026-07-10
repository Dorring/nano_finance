"""Phase 34 tests: reusable SQLite migration helpers."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.session_manager import SessionManager
from services.sqlite_migrations import (
    ensure_column,
    get_component_version,
    run_component_migrations,
    table_columns,
)


def test_ensure_column_is_idempotent(tmp_path):
    db_path = tmp_path / "migrations.db"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

        assert ensure_column(conn, "items", "metadata_json", "metadata_json TEXT") is True
        assert ensure_column(conn, "items", "metadata_json", "metadata_json TEXT") is False
        assert "metadata_json" in table_columns(conn, "items")


def test_run_component_migrations_updates_version_once(tmp_path):
    db_path = tmp_path / "migrations.db"
    calls = []

    def migrate_to_v2(conn):
        calls.append("v2")
        ensure_column(conn, "items", "metadata_json", "metadata_json TEXT")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE items (id INTEGER PRIMARY KEY);
            CREATE TABLE schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            """
        )

        run_component_migrations(conn, "items", 2, {2: migrate_to_v2})
        run_component_migrations(conn, "items", 2, {2: migrate_to_v2})

        assert calls == ["v2"]
        assert get_component_version(conn, "items") == 2
        assert "metadata_json" in table_columns(conn, "items")


def test_session_manager_uses_shared_migration_helper_static():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "services", "session_manager.py")
    content = open(path, encoding="utf-8").read()

    assert "run_component_migrations" in content
    assert "ensure_column" in content
    assert "def _migrate_to_v2" in content


def test_session_manager_still_migrates_legacy_schema(tmp_path):
    db_path = tmp_path / "legacy_sessions.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            INSERT INTO schema_version VALUES ('session_manager', 1);
            """
        )

    manager = SessionManager(db_path=str(db_path))
    manager.add_message("s1", 1, "assistant", "A", metadata={"trace": "t1"})

    assert manager.get_recent_messages("s1", 1)[0]["metadata"] == {"trace": "t1"}
