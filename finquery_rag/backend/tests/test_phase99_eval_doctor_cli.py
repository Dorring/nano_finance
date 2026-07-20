import json
import sqlite3

from src.services.document_registry import DocumentRegistry
from src.services.session_manager import SessionManager
from src.evaluation.eval_cli import main as eval_cli_main


def _create_bm25_db(path):
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


def _prepare_ready_runtime(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    registry_path = tmp_path / "registry.db"
    sessions_path = tmp_path / "sessions.db"
    bm25_path = tmp_path / "bm25.db"
    registry = DocumentRegistry(db_path=str(registry_path))
    sessions = SessionManager(db_path=str(sessions_path))
    _create_bm25_db(bm25_path)
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", str(registry_path))
    monkeypatch.setenv("SESSIONS_DB_PATH", str(sessions_path))
    try:
        sessions.close()
    except Exception:
        pass
    return registry, bm25_path


def test_eval_cli_doctor_returns_zero_when_required_runtime_is_ready(tmp_path, monkeypatch, capsys):
    _prepare_ready_runtime(tmp_path, monkeypatch)
    out_path = tmp_path / "doctor.json"

    code = eval_cli_main([
        "doctor",
        "--bm25-db",
        str(tmp_path / "bm25.db"),
        "--out",
        str(out_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert captured.err == ""
    assert payload["ready"] is True
    assert written["status"] == "ready"
    assert "secret" not in captured.out.lower()


def test_eval_cli_doctor_returns_one_when_required_store_is_missing(tmp_path, monkeypatch, capsys):
    _prepare_ready_runtime(tmp_path, monkeypatch)

    code = eval_cli_main([
        "doctor",
        "--bm25-db",
        str(tmp_path / "missing_bm25.db"),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["ready"] is False
    assert payload["checks"]["bm25"]["ok"] is False
    assert "FinQuery doctor detected degraded readiness:" in captured.err
    assert "bm25 (required)" in captured.err


def test_eval_cli_doctor_warn_only_returns_zero_for_degraded_runtime(tmp_path, monkeypatch, capsys):
    _prepare_ready_runtime(tmp_path, monkeypatch)

    code = eval_cli_main([
        "doctor",
        "--bm25-db",
        str(tmp_path / "missing_bm25.db"),
        "--warn-only",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == "degraded"
    assert "bm25 (required)" in captured.err
