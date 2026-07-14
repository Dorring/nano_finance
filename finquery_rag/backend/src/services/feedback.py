"""Tenant-scoped answer feedback storage for RAG quality loops."""
import os
import sqlite3
import time
import uuid

from .sqlite_migrations import run_component_migrations


class FeedbackStore:
    """SQLite-backed feedback store keyed by tenant and trace."""

    SCHEMA_VERSION = 2
    VALID_RATINGS = {"up", "down"}
    MAX_TRACE_ID_LENGTH = 128
    MAX_COMMENT_CHARS = 2000

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.getenv("FEEDBACK_DB_PATH", "feedback.db")
        self.db_path = db_path
        self._init_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_schema_version (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS answer_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    tenant_id INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_tenant_created
                    ON answer_feedback(tenant_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_trace
                    ON answer_feedback(tenant_id, trace_id);
                """
            )
            run_component_migrations(
                conn,
                "feedback_store",
                self.SCHEMA_VERSION,
                {2: self._migrate_to_v2},
                version_table="feedback_schema_version",
            )

    def _migrate_to_v2(self, conn):
        self._dedupe_latest_feedback(conn)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_tenant_trace_unique
                ON answer_feedback(tenant_id, trace_id)
            """
        )

    def _dedupe_latest_feedback(self, conn):
        """Keep only the newest feedback row for each tenant/trace before enabling uniqueness."""
        conn.execute(
            """
            DELETE FROM answer_feedback
            WHERE EXISTS (
                SELECT 1
                FROM answer_feedback AS newer
                WHERE newer.tenant_id = answer_feedback.tenant_id
                  AND newer.trace_id = answer_feedback.trace_id
                  AND (
                      newer.created_at > answer_feedback.created_at
                      OR (
                          newer.created_at = answer_feedback.created_at
                          AND newer.rowid > answer_feedback.rowid
                      )
                  )
            )
            """
        )

    def _is_valid_trace_id(self, trace_id) -> bool:
        return isinstance(trace_id, str) and 0 < len(trace_id) <= self.MAX_TRACE_ID_LENGTH

    def _clean_comment(self, comment):
        if not isinstance(comment, str):
            return None
        cleaned = " ".join(comment.replace("\x00", " ").split())
        if not cleaned:
            return None
        return cleaned[: self.MAX_COMMENT_CHARS]

    def _normalize_bounds(self, limit=50, offset=0):
        try:
            normalized_limit = int(limit or 50)
            normalized_offset = int(offset or 0)
        except (TypeError, ValueError):
            return None
        return max(0, min(normalized_limit, 1000)), max(0, normalized_offset)

    def submit(self, tenant_id, trace_id, rating, comment=None, now=None):
        """Store latest feedback for one tenant/trace; fail closed on invalid identifiers."""
        if tenant_id is None or not self._is_valid_trace_id(trace_id):
            return None
        if rating not in self.VALID_RATINGS:
            return None

        feedback_id = uuid.uuid4().hex
        created_at = time.time() if now is None else float(now)
        cleaned_comment = self._clean_comment(comment)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO answer_feedback (feedback_id, tenant_id, trace_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, trace_id) DO UPDATE SET
                    feedback_id = excluded.feedback_id,
                    rating = excluded.rating,
                    comment = excluded.comment,
                    created_at = excluded.created_at
                """,
                (feedback_id, tenant_id, trace_id, rating, cleaned_comment, created_at),
            )
            conn.commit()

        return {
            "feedback_id": feedback_id,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "rating": rating,
            "comment": cleaned_comment,
            "created_at": created_at,
        }

    def list_for_tenant(self, tenant_id, limit=50, offset=0, rating=None):
        """Return tenant-scoped feedback rows ordered newest first."""
        if tenant_id is None:
            return []

        bounds = self._normalize_bounds(limit, offset)
        if bounds is None:
            return []
        limit, offset = bounds
        where = ["tenant_id = ?"]
        params = [tenant_id]
        if rating:
            if rating not in self.VALID_RATINGS:
                return []
            where.append("rating = ?")
            params.append(rating)

        sql = (
            "SELECT feedback_id, tenant_id, trace_id, rating, comment, created_at "
            "FROM answer_feedback WHERE "
            + " AND ".join(where)
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    def count_for_tenant(self, tenant_id, rating=None):
        """Count tenant-scoped feedback rows, optionally filtered by rating."""
        if tenant_id is None:
            return 0
        where = ["tenant_id = ?"]
        params = [tenant_id]
        if rating:
            if rating not in self.VALID_RATINGS:
                return 0
            where.append("rating = ?")
            params.append(rating)

        sql = "SELECT COUNT(*) FROM answer_feedback WHERE " + " AND ".join(where)
        with self._conn() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def summary_for_tenant(self, tenant_id):
        """Return compact tenant-scoped feedback diagnostics."""
        if tenant_id is None:
            return {"total": 0, "up": 0, "down": 0, "latest_created_at": None}

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT rating, COUNT(*) AS count, MAX(created_at) AS latest_created_at
                FROM answer_feedback
                WHERE tenant_id = ?
                GROUP BY rating
                """,
                (tenant_id,),
            ).fetchall()

        summary = {"total": 0, "up": 0, "down": 0, "latest_created_at": None}
        for row in rows:
            rating = row["rating"]
            count = int(row["count"] or 0)
            if rating in self.VALID_RATINGS:
                summary[rating] = count
                summary["total"] += count
            latest = row["latest_created_at"]
            if latest is not None and (
                summary["latest_created_at"] is None or latest > summary["latest_created_at"]
            ):
                summary["latest_created_at"] = latest
        return summary
