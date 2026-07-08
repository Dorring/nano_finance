"""
Structured tracing for RAG queries.

Logs query lifecycle: rewrite, filter, candidates, scores, context, model info, timing.
Supports content sanitization and sampling.
"""
import sqlite3
import hashlib
import json
import time
import uuid
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

SCHEMA_VERSION = 1

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
    def __init__(self, db_path="trace_log.db", sample_rate=1.0, redact_content=True):
        self.db_path = db_path
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.redact_content = redact_content
        self._init_schema()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_schema(self):
        with self._conn() as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
            if row is None:
                conn.executescript(_SCHEMA_SQL)
                conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
                conn.commit()
            else:
                ver = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if ver and ver[0] < SCHEMA_VERSION:
                    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
                    conn.commit()

    def _should_sample(self):
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        import random
        return random.random() < self.sample_rate

    @staticmethod
    def sanitize(text):
        return _SANITIZE_RE.sub("[REDACTED]", text)

    def log(self, tenant_id, query_original, query_rewritten=None, intent=None,
            filter_conditions=None, candidates=None, final_context=None,
            answer=None, sources=None, model_name=None, prompt_version=None,
            index_version=None, latency_ms=None, error_message=None):
        if not self._should_sample():
            return None

        trace_id = uuid.uuid4().hex
        now = time.time()

        q_orig = self.sanitize(query_original) if self.redact_content else query_original
        q_rewr = self.sanitize(query_rewritten) if self.redact_content and query_rewritten else query_rewritten
        ctx = self.sanitize(final_context) if self.redact_content and final_context else final_context
        ans = self.sanitize(answer) if self.redact_content and answer else answer

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO trace_log (trace_id, tenant_id, query_original, query_rewritten, intent, filter_conditions, candidates_json, final_context, answer, sources_json, model_name, prompt_version, index_version, latency_ms, error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trace_id, tenant_id, q_orig, q_rewr, intent,
                 json.dumps(filter_conditions) if filter_conditions else None,
                 json.dumps(candidates) if candidates else None,
                 ctx, ans,
                 json.dumps(sources) if sources else None,
                 model_name, prompt_version, index_version,
                 latency_ms, error_message, now)
            )
            conn.commit()
        return trace_id

    def get_trace(self, trace_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM trace_log WHERE trace_id = ?", (trace_id,)).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_trace_for_tenant(self, tenant_id, trace_id):
        """Return a trace only when it belongs to the requested tenant."""
        if tenant_id is None or not trace_id:
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
        limit = max(0, min(int(limit or 20), 1000))
        offset = max(0, int(offset or 0))

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
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        rows = self._conn().execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def export_traces_jsonl(self, tenant_id, output_path, **query_kwargs):
        """Export tenant-scoped traces as JSONL and return exported count."""
        rows = self.query_traces(tenant_id=tenant_id, **query_kwargs)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

    @staticmethod
    def _row_to_dict(row):
        columns = ["trace_id", "tenant_id", "query_original", "query_rewritten", "intent",
                    "filter_conditions", "candidates_json", "final_context", "answer",
                    "sources_json", "model_name", "prompt_version", "index_version",
                    "latency_ms", "error_message", "created_at"]
        return dict(zip(columns, row))
