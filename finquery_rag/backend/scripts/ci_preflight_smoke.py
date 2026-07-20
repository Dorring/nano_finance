"""Run a commit-safe FinQuery deployment preflight smoke.

The script builds a temporary ready local runtime (SQLite stores + Chroma metadata)
and runs `src.eval_cli preflight` against the checked smoke eval fixtures. It does
not call LLMs, embeddings, or external services.
"""
from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.eval_cli import main as eval_cli_main  # noqa: E402
from src.services.document_registry import DocumentRegistry  # noqa: E402
from src.services.session_manager import SessionManager  # noqa: E402
from src.services.trace import TraceLogger  # noqa: E402


def main() -> int:
    artifact_root = os.getenv("FINQUERY_PREFLIGHT_ARTIFACT_DIR")
    artifacts = Path(artifact_root) if artifact_root else Path(tempfile.gettempdir()) / "finquery_preflight_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finquery-preflight-", ignore_cleanup_errors=True) as tmp:
        runtime = Path(tmp)
        bm25 = runtime / "bm25.db"
        registry = runtime / "registry.db"
        sessions = runtime / "sessions.db"
        trace = runtime / "trace.db"
        feedback = runtime / "feedback.db"
        chroma = runtime / "chroma"
        _create_bm25(bm25)
        _create_registry(registry)
        _create_sessions(sessions)
        TraceLogger(db_path=str(trace))
        _create_feedback(feedback)
        _create_chroma(chroma)
        old_env = {
            "CHROMA_PATH": os.environ.get("CHROMA_PATH"),
            "DOCUMENT_REGISTRY_DB_PATH": os.environ.get("DOCUMENT_REGISTRY_DB_PATH"),
            "SESSIONS_DB_PATH": os.environ.get("SESSIONS_DB_PATH"),
        }
        os.environ["CHROMA_PATH"] = str(chroma)
        os.environ["DOCUMENT_REGISTRY_DB_PATH"] = str(registry)
        os.environ["SESSIONS_DB_PATH"] = str(sessions)
        try:
            return eval_cli_main([
                "preflight",
                "--cases",
                str(ROOT / "eval" / "golden_smoke.jsonl"),
                "--predictions",
                str(ROOT / "eval" / "predictions_smoke.jsonl"),
                "--baseline",
                str(ROOT / "eval" / "baseline_smoke_report.json"),
                "--bm25-db",
                str(bm25),
                "--registry-db",
                str(registry),
                "--chroma-path",
                str(chroma),
                "--trace-db",
                str(trace),
                "--feedback-db",
                str(feedback),
                "--min-pass-rate",
                "1.0",
                "--max-missing",
                "0",
                "--tolerance",
                "0.0",
                "--min-cases",
                "12",
                "--required-tag",
                "smoke",
                "--required-tag",
                "citation",
                "--required-tag",
                "no_answer",
                "--required-tag",
                "calculation",
                "--require-expected-intent",
                "--out",
                str(artifacts / "preflight_smoke.json"),
            ])
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _create_bm25(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE chunk_store (
                doc_id TEXT PRIMARY KEY,
                content TEXT,
                metadata_json TEXT,
                user_id INTEGER,
                doc_name TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE fts_index USING fts5(
                content,
                doc_id UNINDEXED,
                tokenize='unicode61'
            )
        """)
        rows = [
            ("user_1_q3.pdf::1", "redacted", "{}", 1, "q3.pdf"),
        ]
        for doc_id, content, metadata, user_id, doc_name in rows:
            conn.execute(
                "INSERT INTO chunk_store(doc_id, content, metadata_json, user_id, doc_name) VALUES (?, ?, ?, ?, ?)",
                (doc_id, content, metadata, user_id, doc_name),
            )
            conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", (content, doc_id))
        conn.commit()


def _create_registry(path: Path) -> None:
    registry = DocumentRegistry(db_path=str(path))
    document_id = "user_1_q3.pdf"
    registry.register(document_id, 1, "q3.pdf", "hash", chunk_count=1, status="pending")
    registry.transition(document_id, "parsing")
    registry.mark_indexing(document_id)
    registry.mark_ready(document_id, chunk_count=1, content_hash="content-hash")
    del registry


def _create_sessions(path: Path) -> None:
    manager = SessionManager(db_path=str(path))
    manager.close()


def _create_feedback(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE answer_feedback (
                feedback_id TEXT PRIMARY KEY,
                tenant_id INTEGER,
                trace_id TEXT,
                rating TEXT,
                comment TEXT,
                created_at REAL
            )
        """)
        conn.commit()


def _create_chroma(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE embeddings (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO embeddings(id) VALUES (?)", ("user_1_q3.pdf::1",))
        conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())
