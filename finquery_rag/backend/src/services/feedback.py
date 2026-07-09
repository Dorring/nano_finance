"""Tenant-scoped answer feedback storage for RAG quality loops."""
import os
import sqlite3
import time
import uuid


class FeedbackStore:
    """SQLite-backed feedback store keyed by tenant and trace."""

    SCHEMA_VERSION = 1
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
            row = conn.execute("SELECT version FROM feedback_schema_version LIMIT 1").fetchone()
            if row is None:
                conn.execute("INSERT INTO feedback_schema_version VALUES (?)", (self.SCHEMA_VERSION,))
            elif row["version"] < self.SCHEMA_VERSION:
                conn.execute("UPDATE feedback_schema_version SET version = ?", (self.SCHEMA_VERSION,))
            conn.commit()

    def submit(self, tenant_id, trace_id, rating, comment=None, now=None):
        """Store one feedback row and return it; fail closed on invalid identifiers."""
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
                "INSERT INTO answer_feedback (feedback_id, tenant_id, trace_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
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
