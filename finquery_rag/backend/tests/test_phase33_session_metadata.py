"""Phase 33 tests: session history preserves assistant UI metadata."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.session_manager import SessionManager


def test_session_manager_round_trips_message_metadata(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    metadata = {
        "sources": [{"filename": "report.pdf", "chunk_id": "chunk-1"}],
        "diagnostics": {"traceId": "trace-1", "retrievalConfidence": 0.81},
    }

    mgr.add_message("s1", 1, "assistant", "Answer", metadata=metadata)
    messages = mgr.get_recent_messages("s1", 1)

    assert messages == [{
        "role": "assistant",
        "content": "Answer",
        "metadata": metadata,
    }]


def test_session_manager_migrates_legacy_schema_for_metadata(tmp_path):
    db_path = tmp_path / "legacy_sessions.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            INSERT INTO schema_version VALUES ('session_manager', 1);
            """
        )

    mgr = SessionManager(db_path=str(db_path))
    mgr.add_message("s1", 1, "assistant", "Answer", metadata={"diagnostics": {"traceId": "t"}})
    message = mgr.get_recent_messages("s1", 1)[0]

    assert message["metadata"]["diagnostics"]["traceId"] == "t"
    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM schema_version WHERE component = 'session_manager'"
        ).fetchone()[0]
    assert version == 2


def test_main_and_dashboard_wire_session_metadata_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    dashboard = open(os.path.join(root, "..", "frontend", "src", "pages", "Dashboard.jsx"), encoding="utf-8").read()

    assert "def _assistant_session_metadata" in main
    assert "metadata=_assistant_session_metadata" in main
    assert "metadata.sources || message.sources || []" in dashboard
    assert "metadata.diagnostics || message.diagnostics || null" in dashboard
