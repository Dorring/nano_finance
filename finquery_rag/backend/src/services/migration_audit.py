"""Read-only migration readiness audit for FinQuery local RAG stores.

The audit is designed for production cutovers from older unscoped indexes. It
never reads chunk content or tenant document text; it only inspects identifiers,
metadata fields, table presence, and aggregate counts.
"""
from __future__ import annotations

from pathlib import Path
import os
import sqlite3
from typing import Any

from .chunk_id import is_scoped_chunk_id


DEFAULT_BM25_DB_PATH = "rag_bm25.db"
DEFAULT_REGISTRY_DB_PATH = "document_registry.db"
DEFAULT_CHROMA_PATH = "./chroma_db"


def audit_migration_readiness(
    *,
    bm25_db_path: str | None = None,
    registry_db_path: str | None = None,
    chroma_path: str | None = None,
) -> dict[str, Any]:
    """Return a non-content migration readiness report for local stores."""
    bm25_path = bm25_db_path or os.getenv("BM25_DB_PATH", DEFAULT_BM25_DB_PATH)
    registry_path = registry_db_path or os.getenv("DOCUMENT_REGISTRY_DB_PATH", DEFAULT_REGISTRY_DB_PATH)
    chroma_dir = chroma_path or os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)

    stores = {
        "bm25": _audit_bm25_store(bm25_path),
        "document_registry": _audit_registry_store(registry_path),
        "chroma": _audit_chroma_store(chroma_dir),
    }
    risks = []
    for store_name, store in stores.items():
        for issue in store.get("issues", []):
            risks.append({"store": store_name, **issue})

    high_risks = [risk for risk in risks if risk.get("severity") == "high"]
    medium_risks = [risk for risk in risks if risk.get("severity") == "medium"]
    low_risks = [risk for risk in risks if risk.get("severity") == "low"]
    return {
        "passed": not high_risks,
        "summary": {
            "risk_count": len(risks),
            "high_risk_count": len(high_risks),
            "medium_risk_count": len(medium_risks),
            "low_risk_count": len(low_risks),
        },
        "stores": stores,
        "risks": risks,
        "recommendations": _recommendations(risks),
    }


def _audit_bm25_store(path: str) -> dict[str, Any]:
    result = _base_store_result(path, kind="sqlite")
    if not result["exists"]:
        result["issues"].append(_issue("medium", "bm25_missing", "BM25 database does not exist; rebuild before production queries"))
        return result

    try:
        with sqlite3.connect(path, timeout=2) as conn:
            tables = _sqlite_tables(conn)
            result["tables"] = sorted(tables)
            required = {"chunk_store", "fts_index"}
            missing = sorted(required - tables)
            result["missing_tables"] = missing
            if missing:
                result["issues"].append(_issue("high", "bm25_missing_tables", "BM25 database is missing required tables", {"missing_tables": missing}))
                return result
            columns = _table_columns(conn, "chunk_store")
            result["chunk_store_columns"] = columns
            for column in ("doc_id", "user_id", "doc_name"):
                if column not in columns:
                    result["issues"].append(_issue("high", "bm25_missing_column", f"chunk_store is missing {column!r}", {"column": column}))
            if any(column not in columns for column in ("doc_id", "user_id", "doc_name")):
                return result

            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN doc_id LIKE 'user_%' THEN 1 ELSE 0 END) AS scoped,
                    SUM(CASE WHEN doc_id NOT LIKE 'user_%' THEN 1 ELSE 0 END) AS legacy,
                    SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) AS missing_user,
                    SUM(CASE WHEN doc_name IS NULL OR doc_name = '' THEN 1 ELSE 0 END) AS missing_doc_name
                FROM chunk_store
                """
            ).fetchone()
            total, scoped, legacy, missing_user, missing_doc_name = (int(value or 0) for value in row)
            result["counts"].update({
                "chunk_store_rows": total,
                "scoped_doc_ids": scoped,
                "legacy_unscoped_doc_ids": legacy,
                "missing_user_id_rows": missing_user,
                "missing_doc_name_rows": missing_doc_name,
                "fts_rows": int(conn.execute("SELECT COUNT(*) FROM fts_index").fetchone()[0] or 0),
            })
            samples = [row[0] for row in conn.execute(
                "SELECT doc_id FROM chunk_store WHERE doc_id NOT LIKE 'user_%' ORDER BY doc_id LIMIT 5"
            ).fetchall()]
            result["samples"]["legacy_doc_ids"] = samples

            mismatch_count = 0
            for doc_id, user_id in conn.execute("SELECT doc_id, user_id FROM chunk_store WHERE user_id IS NOT NULL LIMIT 10000"):
                if not is_scoped_chunk_id(str(doc_id), user_id):
                    mismatch_count += 1
            result["counts"]["scope_user_mismatch_sampled_rows"] = mismatch_count

            if legacy:
                result["issues"].append(_issue("high", "bm25_legacy_doc_ids", "BM25 chunk_store contains unscoped doc IDs that current tenant filters may not retrieve", {"count": legacy, "samples": samples}))
            if missing_user:
                result["issues"].append(_issue("high", "bm25_missing_user_id", "BM25 chunk_store rows are missing user_id", {"count": missing_user}))
            if missing_doc_name:
                result["issues"].append(_issue("medium", "bm25_missing_doc_name", "BM25 chunk_store rows are missing doc_name", {"count": missing_doc_name}))
            if mismatch_count:
                result["issues"].append(_issue("high", "bm25_scope_user_mismatch", "BM25 doc_id user prefix does not match user_id on sampled rows", {"sampled_mismatch_count": mismatch_count}))
    except Exception as exc:  # pragma: no cover - platform sqlite errors vary
        result["issues"].append(_issue("high", "bm25_unreadable", f"BM25 database cannot be audited: {exc}"))
    return result


def _audit_registry_store(path: str) -> dict[str, Any]:
    result = _base_store_result(path, kind="sqlite")
    if not result["exists"]:
        result["issues"].append(_issue("medium", "registry_missing", "Document registry database does not exist; uploads may need registry rebuild"))
        return result
    try:
        with sqlite3.connect(path, timeout=2) as conn:
            tables = _sqlite_tables(conn)
            result["tables"] = sorted(tables)
            if "document_registry" not in tables:
                result["issues"].append(_issue("high", "registry_missing_table", "document_registry table is missing"))
                return result
            columns = _table_columns(conn, "document_registry")
            result["document_registry_columns"] = columns
            required_columns = {"document_id", "tenant_id", "filename", "status", "chunk_count"}
            missing_columns = sorted(required_columns - set(columns))
            result["missing_columns"] = missing_columns
            if missing_columns:
                result["issues"].append(_issue("high", "registry_missing_columns", "document_registry is missing required columns", {"missing_columns": missing_columns}))
                return result
            total = int(conn.execute("SELECT COUNT(*) FROM document_registry").fetchone()[0] or 0)
            ready = int(conn.execute("SELECT COUNT(*) FROM document_registry WHERE status = 'ready'").fetchone()[0] or 0)
            missing_tenant = int(conn.execute("SELECT COUNT(*) FROM document_registry WHERE tenant_id IS NULL").fetchone()[0] or 0)
            missing_filename = int(conn.execute("SELECT COUNT(*) FROM document_registry WHERE filename IS NULL OR filename = ''").fetchone()[0] or 0)
            zero_ready_chunks = int(conn.execute("SELECT COUNT(*) FROM document_registry WHERE status = 'ready' AND COALESCE(chunk_count, 0) <= 0").fetchone()[0] or 0)
            result["counts"].update({
                "registry_rows": total,
                "ready_rows": ready,
                "missing_tenant_id_rows": missing_tenant,
                "missing_filename_rows": missing_filename,
                "ready_rows_with_zero_chunks": zero_ready_chunks,
            })
            status_counts = {row[0] or "unknown": int(row[1]) for row in conn.execute("SELECT status, COUNT(*) FROM document_registry GROUP BY status")}
            result["status_counts"] = status_counts
            if missing_tenant:
                result["issues"].append(_issue("high", "registry_missing_tenant", "Registry rows are missing tenant_id", {"count": missing_tenant}))
            if missing_filename:
                result["issues"].append(_issue("high", "registry_missing_filename", "Registry rows are missing filename", {"count": missing_filename}))
            if zero_ready_chunks:
                result["issues"].append(_issue("medium", "registry_ready_zero_chunks", "Ready registry rows have zero chunks", {"count": zero_ready_chunks}))
    except Exception as exc:  # pragma: no cover
        result["issues"].append(_issue("high", "registry_unreadable", f"Document registry cannot be audited: {exc}"))
    return result


def _audit_chroma_store(path: str) -> dict[str, Any]:
    result = _base_store_result(path, kind="directory")
    chroma_dir = Path(path)
    if not chroma_dir.exists():
        result["issues"].append(_issue("low", "chroma_missing", "Chroma directory does not exist; vector index may need rebuild"))
        return result
    if not chroma_dir.is_dir():
        result["issues"].append(_issue("high", "chroma_not_directory", "Chroma path exists but is not a directory"))
        return result
    sqlite_path = chroma_dir / "chroma.sqlite3"
    result["sqlite_path"] = str(sqlite_path)
    result["sqlite_exists"] = sqlite_path.exists()
    if not sqlite_path.exists():
        result["issues"].append(_issue("medium", "chroma_sqlite_missing", "Chroma sqlite metadata file was not found; vector index may be incomplete"))
        return result
    try:
        with sqlite3.connect(str(sqlite_path), timeout=2) as conn:
            tables = _sqlite_tables(conn)
            result["tables"] = sorted(tables)
            table, column = _find_chroma_id_column(conn, tables)
            result["id_table"] = table
            result["id_column"] = column
            if not table or not column:
                result["issues"].append(_issue("medium", "chroma_id_column_unknown", "Could not locate Chroma embedding ID column for scoped ID audit"))
                return result
            total = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
            legacy = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} NOT LIKE 'user_%'").fetchone()[0] or 0)
            scoped = total - legacy
            result["counts"].update({
                "embedding_id_rows": total,
                "scoped_embedding_ids": scoped,
                "legacy_unscoped_embedding_ids": legacy,
            })
            samples = [row[0] for row in conn.execute(f"SELECT {column} FROM {table} WHERE {column} NOT LIKE 'user_%' ORDER BY {column} LIMIT 5").fetchall()]
            result["samples"]["legacy_embedding_ids"] = samples
            if legacy:
                result["issues"].append(_issue("high", "chroma_legacy_embedding_ids", "Chroma embedding IDs appear unscoped and may not match current chunk IDs", {"count": legacy, "samples": samples}))
    except Exception as exc:  # pragma: no cover
        result["issues"].append(_issue("medium", "chroma_unreadable", f"Chroma sqlite metadata cannot be audited: {exc}"))
    return result


def _base_store_result(path: str, *, kind: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "kind": kind,
        "exists": p.exists(),
        "counts": {},
        "samples": {},
        "issues": [],
    }


def _issue(severity: str, code: str, message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"severity": severity, "code": code, "message": message}
    if extra:
        payload.update(extra)
    return payload


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _find_chroma_id_column(conn: sqlite3.Connection, tables: set[str]) -> tuple[str | None, str | None]:
    for table in ("embeddings", "embedding_metadata"):
        if table not in tables:
            continue
        columns = _table_columns(conn, table)
        for column in ("embedding_id", "id"):
            if column in columns:
                return table, column
    return None, None


def _recommendations(risks: list[dict[str, Any]]) -> list[str]:
    codes = {risk.get("code") for risk in risks}
    recs: list[str] = []
    if {"bm25_legacy_doc_ids", "bm25_missing_user_id", "bm25_scope_user_mismatch"} & codes:
        recs.append("Rebuild BM25 from tenant-scoped chunks before enabling production queries.")
    if "chroma_legacy_embedding_ids" in codes:
        recs.append("Rebuild Chroma/vector index so embedding IDs match current user-scoped chunk IDs.")
    if {"registry_missing", "registry_ready_zero_chunks"} & codes:
        recs.append("Rebuild or backfill document_registry before relying on lifecycle/status UI.")
    if not recs:
        recs.append("No high-risk legacy index patterns detected by the non-content audit.")
    return recs
