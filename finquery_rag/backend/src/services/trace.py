"""
Structured tracing for RAG queries.

Logs query lifecycle: rewrite, filter, candidates, scores, context, model info, timing.
Supports content sanitization and sampling.
"""
import sqlite3
import json
import os
import time
import uuid
import re

from src.evaluation.evaluation import write_jsonl
from .sqlite_migrations import ensure_column, run_component_migrations, table_exists

SCHEMA_VERSION = 2
MAX_TRACE_ID_LENGTH = 128
MAX_TEXT_CHARS = 50000
MAX_JSON_CHARS = 50000

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trace_log (
    trace_id      TEXT PRIMARY KEY,
    tenant_id     INTEGER NOT NULL,
    query_original TEXT NOT NULL,
    query_rewritten TEXT,
    intent        TEXT,
    filter_conditions TEXT,
    candidates_json TEXT,
    final_context TEXT,
    answer        TEXT,
    sources_json  TEXT,
    diagnostics_json TEXT,
    model_name    TEXT,
    prompt_version TEXT,
    index_version TEXT,
    latency_ms    REAL,
    error_message TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_tenant ON trace_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_log(created_at);
"""

# Default sensitive pattern - SSN, phone, etc.
_SANITIZE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

class TraceLogger:
    def __init__(self, db_path=None, sample_rate=1.0, redact_content=True):
        if db_path is None:
            db_path = os.getenv("TRACE_DB_PATH", "trace_log.db")
        self.db_path = db_path
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.redact_content = redact_content
        self._last_created_at = 0.0
        self._init_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            if not table_exists(conn, "schema_version"):
                conn.executescript(_SCHEMA_SQL)
            run_component_migrations(
                conn,
                "trace_logger",
                SCHEMA_VERSION,
                {
                    2: self._migrate_to_v2,
                },
            )

    @staticmethod
    def _migrate_to_v2(conn):
        ensure_column(conn, "trace_log", "diagnostics_json", "diagnostics_json TEXT")

    def _should_sample(self):
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        import random
        return random.random() < self.sample_rate

    @staticmethod
    def sanitize(text):
        return _SANITIZE_RE.sub("[REDACTED]", str(text))

    @staticmethod
    def _is_valid_trace_id(trace_id):
        return isinstance(trace_id, str) and 0 < len(trace_id) <= MAX_TRACE_ID_LENGTH

    @staticmethod
    def _bounded_text(value):
        if value is None:
            return None
        text = str(value)
        if len(text) <= MAX_TEXT_CHARS:
            return text
        return text[:MAX_TEXT_CHARS] + "\n[truncated]"

    @staticmethod
    def _bounded_json(value):
        if not value:
            return None
        payload = json.dumps(value)
        if len(payload) <= MAX_JSON_CHARS:
            return payload
        return json.dumps({"truncated": True})

    @staticmethod
    def _normalize_query_bounds(limit=20, offset=0):
        try:
            normalized_limit = int(limit or 20)
            normalized_offset = int(offset or 0)
        except (TypeError, ValueError):
            return None
        return max(0, min(normalized_limit, 1000)), max(0, normalized_offset)

    def log(self, tenant_id, query_original, query_rewritten=None, intent=None,
            filter_conditions=None, candidates=None, final_context=None,
            answer=None, sources=None, diagnostics=None, model_name=None, prompt_version=None,
            index_version=None, latency_ms=None, error_message=None):
        if not self._should_sample():
            return None

        trace_id = uuid.uuid4().hex
        now = max(time.time(), self._last_created_at + 0.000001)
        self._last_created_at = now

        q_orig = self.sanitize(query_original) if self.redact_content else query_original
        q_rewr = self.sanitize(query_rewritten) if self.redact_content and query_rewritten else query_rewritten
        ctx = self.sanitize(final_context) if self.redact_content and final_context else final_context
        ans = self.sanitize(answer) if self.redact_content and answer else answer
        q_orig = self._bounded_text(q_orig)
        q_rewr = self._bounded_text(q_rewr)
        ctx = self._bounded_text(ctx)
        ans = self._bounded_text(ans)
        err = self._bounded_text(error_message)

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO trace_log (trace_id, tenant_id, query_original, query_rewritten, intent, filter_conditions, candidates_json, final_context, answer, sources_json, diagnostics_json, model_name, prompt_version, index_version, latency_ms, error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trace_id, tenant_id, q_orig, q_rewr, intent,
                 self._bounded_json(filter_conditions),
                 self._bounded_json(candidates),
                 ctx, ans,
                 self._bounded_json(sources),
                 self._bounded_json(diagnostics),
                 self._bounded_text(model_name), self._bounded_text(prompt_version), self._bounded_text(index_version),
                 latency_ms, err, now)
            )
            conn.commit()
        return trace_id

    def get_trace(self, trace_id):
        if not self._is_valid_trace_id(trace_id):
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM trace_log WHERE trace_id = ?", (trace_id,)).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_trace_for_tenant(self, tenant_id, trace_id):
        """Return a trace only when it belongs to the requested tenant."""
        if tenant_id is None or not self._is_valid_trace_id(trace_id):
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trace_log WHERE tenant_id = ? AND trace_id = ?",
                (tenant_id, trace_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_recent(self, tenant_id, limit=20):
        return self.query_traces(tenant_id=tenant_id, limit=limit)

    def query_traces(
        self,
        tenant_id,
        limit=20,
        offset=0,
        created_after=None,
        created_before=None,
        error_only=False,
    ):
        """Tenant-scoped trace query with basic filters for replay/export."""
        if tenant_id is None:
            return []
        bounds = self._normalize_query_bounds(limit, offset)
        if bounds is None:
            return []
        limit, offset = bounds

        where = ["tenant_id = ?"]
        params = [tenant_id]
        if created_after is not None:
            where.append("created_at >= ?")
            params.append(float(created_after))
        if created_before is not None:
            where.append("created_at <= ?")
            params.append(float(created_before))
        if error_only:
            where.append("error_message IS NOT NULL")

        sql = (
            "SELECT * FROM trace_log WHERE "
            + " AND ".join(where)
            + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_traces(
        self,
        tenant_id,
        created_after=None,
        created_before=None,
        error_only=False,
    ):
        """Count tenant-scoped traces matching the same filters as query_traces."""
        if tenant_id is None:
            return 0

        where = ["tenant_id = ?"]
        params = [tenant_id]
        if created_after is not None:
            where.append("created_at >= ?")
            params.append(float(created_after))
        if created_before is not None:
            where.append("created_at <= ?")
            params.append(float(created_before))
        if error_only:
            where.append("error_message IS NOT NULL")

        sql = "SELECT COUNT(*) FROM trace_log WHERE " + " AND ".join(where)
        with self._conn() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def summary_for_tenant(self, tenant_id):
        """Return compact tenant-scoped trace diagnostics without query content."""
        if tenant_id is None:
            return {
                "total": 0,
                "errors": 0,
                "latest_created_at": None,
                "avg_latency_ms": None,
            }

        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                    MAX(created_at) AS latest_created_at,
                    AVG(latency_ms) AS avg_latency_ms
                FROM trace_log
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

        avg_latency = row["avg_latency_ms"]
        return {
            "total": int(row["total"] or 0),
            "errors": int(row["errors"] or 0),
            "latest_created_at": row["latest_created_at"],
            "avg_latency_ms": float(avg_latency) if avg_latency is not None else None,
        }
    def export_traces_jsonl(self, tenant_id, output_path, **query_kwargs):
        """Export tenant-scoped traces as JSONL and return exported count."""
        rows = self.query_traces(tenant_id=tenant_id, **query_kwargs)
        write_jsonl(output_path, rows)
        return len(rows)

    def cleanup_older_than(self, cutoff_created_at, tenant_id=None):
        """Delete traces older than cutoff_created_at and return deleted row count.

        If tenant_id is provided, cleanup is tenant-scoped. If tenant_id is None,
        cleanup is global and intended for operator/admin maintenance.
        """
        cutoff = float(cutoff_created_at)
        with self._conn() as conn:
            if tenant_id is None:
                cursor = conn.execute(
                    "DELETE FROM trace_log WHERE created_at < ?",
                    (cutoff,),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM trace_log WHERE tenant_id = ? AND created_at < ?",
                    (tenant_id, cutoff),
                )
            conn.commit()
            return cursor.rowcount

    def cleanup_by_ttl(self, ttl_seconds, tenant_id=None, now=None):
        """Delete traces older than now - ttl_seconds and return cleanup metadata."""
        ttl = max(0, int(ttl_seconds or 0))
        if now is None:
            now = time.time()
        cutoff = float(now) - ttl
        deleted = self.cleanup_older_than(cutoff, tenant_id=tenant_id)
        return {
            "deleted": deleted,
            "tenant_id": tenant_id,
            "ttl_seconds": ttl,
            "cutoff_created_at": cutoff,
        }

    @staticmethod
    def _row_to_dict(row):
        if hasattr(row, "keys"):
            return {key: row[key] for key in row.keys()}
        columns = ["trace_id", "tenant_id", "query_original", "query_rewritten", "intent",
                    "filter_conditions", "candidates_json", "final_context", "answer",
                    "sources_json", "diagnostics_json", "model_name", "prompt_version", "index_version",
                    "latency_ms", "error_message", "created_at"]
        return dict(zip(columns, row))
