"""Operational health and configuration checks for FinQuery RAG.

The checks in this module are intentionally lightweight:
- no outbound LLM call
- no embedding request
- no Chroma collection creation
- no tenant data exposure
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

BM25_DB_PATH = os.getenv("BM25_DB_PATH", "rag_bm25.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
GLOBAL_COLLECTION_NAME = "rag_global_knowledge_base"


def _safe_int(value: str | None, default: int) -> tuple[int, str | None]:
    if value is None:
        return default, None
    try:
        return int(value), None
    except (TypeError, ValueError):
        return default, f"invalid integer: {value!r}"


def _path_check(path: str, *, expect_dir: bool = False) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    parent = p if expect_dir else p.parent
    parent_exists = parent.exists()
    writable = os.access(parent, os.W_OK) if parent_exists else False
    ok = parent_exists and writable
    if expect_dir and exists and not p.is_dir():
        ok = False
    return {
        "ok": ok,
        "path": str(p),
        "exists": exists,
        "parent_exists": parent_exists,
        "parent_writable": writable,
    }


def _sqlite_check(path: str, *, required_tables: tuple[str, ...] = ()) -> dict[str, Any]:
    p = Path(path)
    base = _path_check(str(p))
    result: dict[str, Any] = {
        **base,
        "kind": "sqlite",
        "required_tables": list(required_tables),
        "missing_tables": [],
    }
    if not base["ok"]:
        return result

    if not p.exists():
        result["ok"] = False
        result["error"] = "database file does not exist"
        return result

    try:
        with sqlite3.connect(str(p), timeout=2) as conn:
            conn.execute("SELECT 1").fetchone()
            if required_tables:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
                tables = {row[0] for row in rows}
                missing = [table for table in required_tables if table not in tables]
                result["missing_tables"] = missing
                if missing:
                    result["ok"] = False
                    result["error"] = "missing required tables"
    except Exception as exc:  # pragma: no cover - exact sqlite errors vary by platform
        result["ok"] = False
        result["error"] = str(exc)
    return result


def collect_config_snapshot() -> dict[str, Any]:
    """Return non-secret runtime configuration relevant to RAG operations."""
    candidate_multiplier, candidate_error = _safe_int(
        os.getenv("RAG_CANDIDATE_MULTIPLIER"), 2
    )
    session_ttl, ttl_error = _safe_int(os.getenv("SESSION_TTL_SECONDS"), 0)

    errors = [err for err in (candidate_error, ttl_error) if err]
    if candidate_multiplier < 1:
        errors.append("RAG_CANDIDATE_MULTIPLIER must be >= 1")
    if session_ttl < 0:
        errors.append("SESSION_TTL_SECONDS must be >= 0")

    return {
        "ok": not errors,
        "errors": errors,
        "llm": {
            "base_url_configured": bool(os.getenv("LLM_API_BASE_URL")),
            "api_key_configured": bool(os.getenv("LLM_API_KEY")),
            "model_name": os.getenv("LLM_MODEL_NAME", "nanochat"),
        },
        "retrieval": {
            "hybrid_enabled": True,
            "candidate_multiplier": candidate_multiplier,
            "reranker": os.getenv("RAG_RERANKER") or "none",
            "reranker_model_configured": bool(os.getenv("RAG_RERANKER_MODEL")),
        },
        "storage": {
            "chroma_path": CHROMA_PATH,
            "chroma_collection": GLOBAL_COLLECTION_NAME,
            "bm25_db_path": BM25_DB_PATH,
            "sessions_db_path": os.getenv("SESSIONS_DB_PATH", "sessions.db"),
            "trace_db_path": os.getenv("TRACE_DB_PATH", "trace_log.db"),
        },
        "sessions": {
            "ttl_seconds": session_ttl,
        },
    }


def collect_health_snapshot(
    *,
    document_registry: Any | None = None,
    session_manager: Any | None = None,
    bm25_db_path: str | None = None,
    trace_db_path: str | None = None,
) -> dict[str, Any]:
    """Return a readiness snapshot without reading tenant content."""
    config = collect_config_snapshot()
    registry_path = getattr(document_registry, "db_path", "document_registry.db")
    session_path = getattr(
        session_manager, "db_path", os.getenv("SESSIONS_DB_PATH", "sessions.db")
    )
    bm25_path = bm25_db_path or BM25_DB_PATH
    trace_path = trace_db_path or os.getenv("TRACE_DB_PATH", "trace_log.db")

    checks = {
        "config": config,
        "chroma_path": {
            **_path_check(CHROMA_PATH, expect_dir=True),
            "kind": "directory",
            "required": False,
        },
        "document_registry": {
            **_sqlite_check(registry_path, required_tables=("document_registry",)),
            "required": True,
        },
        "bm25": {
            **_sqlite_check(bm25_path, required_tables=("chunk_store", "fts_index")),
            "required": True,
        },
        "sessions": {
            **_sqlite_check(session_path, required_tables=("conversations",)),
            "required": True,
        },
        "trace": {
            **_sqlite_check(trace_path, required_tables=("trace_log",)),
            "required": False,
        },
    }

    required_ok = all(
        check.get("ok", False)
        for check in checks.values()
        if isinstance(check, dict) and check.get("required", True)
    )
    ready = bool(config["ok"] and required_ok)

    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checked_at": time.time(),
        "checks": checks,
    }
