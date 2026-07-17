"""Tenant-scoped editable memory profile for query planning.

The profile is deliberately small and preference-oriented. It is used to make
follow-up query rewriting less ambiguous, not as a factual source for answers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any


ALLOWED_PROFILE_FIELDS = {
    "preferred_language",
    "preferred_currency",
    "preferred_unit",
    "default_period",
    "default_company",
    "watchlist",
    "focus_metrics",
}

MAX_STRING_CHARS = 80
MAX_LIST_ITEMS = 20


class UserMemoryStore:
    """SQLite-backed editable user memory profile with tenant isolation."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.getenv("MEMORY_PROFILE_DB_PATH", "memory_profile.db")
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memory_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()

    def get_profile(self, user_id: int) -> dict[str, Any]:
        if user_id is None:
            return {}
        row = self._get_conn().execute(
            "SELECT profile_json, updated_at FROM user_memory_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {}
        try:
            profile = json.loads(row["profile_json"])
        except (TypeError, json.JSONDecodeError):
            profile = {}
        if isinstance(profile, dict):
            profile["updated_at"] = row["updated_at"]
            return profile
        return {"updated_at": row["updated_at"]}

    def upsert_profile(self, user_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        if user_id is None:
            return {}
        sanitized_patch = sanitize_profile_patch(patch)
        existing = self.get_profile(user_id)
        existing.pop("updated_at", None)
        existing.update(sanitized_patch)
        compact = {key: value for key, value in existing.items() if value not in (None, "", [])}
        updated_at = time.time()
        self._get_conn().execute(
            """
            INSERT INTO user_memory_profiles(user_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (user_id, json.dumps(compact, ensure_ascii=False, sort_keys=True), updated_at),
        )
        self._get_conn().commit()
        compact["updated_at"] = updated_at
        return compact

    def clear_profile(self, user_id: int) -> bool:
        if user_id is None:
            return False
        cursor = self._get_conn().execute(
            "DELETE FROM user_memory_profiles WHERE user_id = ?",
            (user_id,),
        )
        self._get_conn().commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def sanitize_profile_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Whitelist and bound profile fields."""
    if not isinstance(patch, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in ALLOWED_PROFILE_FIELDS:
            continue
        if isinstance(value, str):
            cleaned = _clean_string(value)
            if cleaned:
                sanitized[key] = cleaned
        elif isinstance(value, list):
            items = []
            for item in value[:MAX_LIST_ITEMS]:
                if isinstance(item, str):
                    cleaned = _clean_string(item)
                    if cleaned and cleaned not in items:
                        items.append(cleaned)
            if items:
                sanitized[key] = items
    return sanitized


def build_memory_profile_context(profile: dict[str, Any] | None) -> str:
    """Return a compact prompt block for query rewriting."""
    if not isinstance(profile, dict):
        return ""
    profile = sanitize_profile_patch(profile)
    if not profile:
        return ""
    lines = []
    for key in sorted(profile):
        value = profile[key]
        if isinstance(value, list):
            rendered = ", ".join(value)
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)


def _clean_string(value: str) -> str:
    value = " ".join(str(value or "").split())
    value = value.replace("\x00", "")
    if len(value) > MAX_STRING_CHARS:
        value = value[:MAX_STRING_CHARS].rstrip()
    return value
