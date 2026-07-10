"""Small SQLite migration helpers for FinQuery service-local databases."""
from __future__ import annotations

from collections.abc import Callable
import sqlite3


Migration = Callable[[sqlite3.Connection], None]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True when a table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for a table."""
    return {
        row["name"] if hasattr(row, "keys") else row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> bool:
    """Add a column if missing. Return True when a column was added."""
    if column_name in table_columns(conn, table_name):
        return False
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    return True


def get_component_version(conn: sqlite3.Connection, component: str) -> int:
    """Read component version from schema_version.

    Supports both normalized schema_version(component, version) and legacy
    single-column schema_version(version) tables.
    """
    if not table_exists(conn, "schema_version"):
        return 0

    columns = table_columns(conn, "schema_version")
    if "component" in columns:
        row = conn.execute(
            "SELECT version FROM schema_version WHERE component = ?",
            (component,),
        ).fetchone()
    else:
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        return 0
    return int(row["version"] if hasattr(row, "keys") else row[0])


def set_component_version(conn: sqlite3.Connection, component: str, version: int) -> None:
    """Store component version while preserving existing schema_version shape."""
    columns = table_columns(conn, "schema_version")
    if "component" in columns:
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (component, version) VALUES (?, ?)",
            (component, int(version)),
        )
    else:
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        count = row[0] if not hasattr(row, "keys") else row[0]
        if count:
            conn.execute("UPDATE schema_version SET version = ?", (int(version),))
        else:
            conn.execute("INSERT INTO schema_version VALUES (?)", (int(version),))


def run_component_migrations(
    conn: sqlite3.Connection,
    component: str,
    target_version: int,
    migrations: dict[int, Migration],
) -> int:
    """Run migrations newer than current version through target_version.

    Migration dict keys are target versions. For example, key 2 migrates
    schema from <2 to 2. Each migration must be idempotent.
    """
    current_version = get_component_version(conn, component)
    for version in sorted(migrations):
        if current_version < version <= target_version:
            migrations[version](conn)
            current_version = version
    set_component_version(conn, component, target_version)
    conn.commit()
    return current_version
