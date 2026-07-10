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

    def submit(self, tenant_id, trace_id, rating, comment=None, now=None):
        """Store latest feedback for one tenant/trace; fail closed on invalid identifiers."""
        if tenant_id is None or not trace_id:
            return None
        if rating not in self.VALID_RATINGS:
            return None

        feedback_id = uuid.uuid4().hex
        created_at = time.time() if now is None else float(now)
        cleaned_comment = comment.strip() if isinstance(comment, str) and comment.strip() else None
        if cleaned_comment and len(cleaned_comment) > 2000:
            cleaned_comment = cleaned_comment[:2000]

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

        limit = max(0, min(int(limit or 50), 1000))
        offset = max(0, int(offset or 0))
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
