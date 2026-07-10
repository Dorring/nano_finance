"""Small SQLite migration helpers for FinQuery service-local databases."""
from __future__ import annotations

from collections.abc import Callable
import sqlite3


Migration = Callable[[sqlite3.Connection], None]


def validate_identifier(identifier: str) -> str:
    """Return identifier when safe for SQLite DDL fragments."""
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("SQLite identifier must be a non-empty string")
    if not (identifier[0].isalpha() or identifier[0] == "_"):
        raise ValueError("Unsafe SQLite identifier: %r" % (identifier,))
    for char in identifier:
        if not (char.isalnum() or char == "_"):
            raise ValueError("Unsafe SQLite identifier: %r" % (identifier,))
    return identifier


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True when a table exists."""
    table_name = validate_identifier(table_name)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for a table."""
    table_name = validate_identifier(table_name)
    return {
        row["name"] if hasattr(row, "keys") else row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> bool:
    """Add a column if missing. Return True when a column was added."""
    table_name = validate_identifier(table_name)
    column_name = validate_identifier(column_name)
    if column_name in table_columns(conn, table_name):
        return False
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    return True


def get_component_version(
    conn: sqlite3.Connection,
    component: str,
    version_table: str = "schema_version",
) -> int:
    """Read component version from a schema version table.

    Supports both normalized version_table(component, version) and legacy
    single-column version_table(version) tables.
    """
    version_table = validate_identifier(version_table)
    if not table_exists(conn, version_table):
        return 0

    version_table = validate_identifier(version_table)
    columns = table_columns(conn, version_table)
    if "component" in columns:
        row = conn.execute(
            f"SELECT version FROM {version_table} WHERE component = ?",
            (component,),
        ).fetchone()
    else:
        row = conn.execute(f"SELECT version FROM {version_table} LIMIT 1").fetchone()
    if row is None:
        return 0
    return int(row["version"] if hasattr(row, "keys") else row[0])


def set_component_version(
    conn: sqlite3.Connection,
    component: str,
    version: int,
    version_table: str = "schema_version",
) -> None:
    """Store component version while preserving existing version table shape."""
    version_table = validate_identifier(version_table)
    columns = table_columns(conn, version_table)
    if "component" in columns:
        conn.execute(
            f"INSERT OR REPLACE INTO {version_table} (component, version) VALUES (?, ?)",
            (component, int(version)),
        )
    else:
        row = conn.execute(f"SELECT COUNT(*) FROM {version_table}").fetchone()
        count = row[0] if not hasattr(row, "keys") else row[0]
        if count:
            conn.execute(f"UPDATE {version_table} SET version = ?", (int(version),))
        else:
            conn.execute(f"INSERT INTO {version_table} VALUES (?)", (int(version),))


def run_component_migrations(
    conn: sqlite3.Connection,
    component: str,
    target_version: int,
    migrations: dict[int, Migration],
    version_table: str = "schema_version",
) -> int:
    """Run migrations newer than current version through target_version.

    Migration dict keys are target versions. For example, key 2 migrates
    schema from <2 to 2. Each migration must be idempotent.
    """
    current_version = get_component_version(conn, component, version_table=version_table)
    for version in sorted(migrations):
        if current_version < version <= target_version:
            migrations[version](conn)
            current_version = version
    set_component_version(conn, component, target_version, version_table=version_table)
    conn.commit()
    return current_version
