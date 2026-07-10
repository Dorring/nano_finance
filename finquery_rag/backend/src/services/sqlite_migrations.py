"""Small SQLite migration helpers for FinQuery service-local databases."""
from __future__ import annotations

from collections.abc import Callable
import sqlite3


Migration = Callable[[sqlite3.Connection], None]


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
    """Read component version from schema_version(component, version)."""
    row = conn.execute(
        "SELECT version FROM schema_version WHERE component = ?",
        (component,),
    ).fetchone()
    if row is None:
        return 0
    return int(row["version"] if hasattr(row, "keys") else row[0])


def set_component_version(conn: sqlite3.Connection, component: str, version: int) -> None:
    """Upsert component version into schema_version(component, version)."""
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (component, version) VALUES (?, ?)",
        (component, int(version)),
    )


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
