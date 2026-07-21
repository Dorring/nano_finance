"""Phase 31 tests: trace diagnostics are persisted for replay/debugging."""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluation import trace_to_replay_case
from services.trace import TraceLogger


def test_trace_logger_persists_diagnostics_json(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(
        tenant_id=1,
        query_original="Q",
        diagnostics={"confidence": 0.8, "context_sufficient": True},
    )

    row = logger.get_trace_for_tenant(1, trace_id)

    assert json.loads(row["diagnostics_json"]) == {
        "confidence": 0.8,
        "context_sufficient": True,
    }


def test_trace_logger_migrates_legacy_schema_with_diagnostics(tmp_path):
    db_path = tmp_path / "legacy_trace.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (1);
            CREATE TABLE trace_log (
                trace_id TEXT PRIMARY KEY,
                tenant_id INTEGER NOT NULL,
                query_original TEXT NOT NULL,
                query_rewritten TEXT,
                intent TEXT,
                filter_conditions TEXT,
                candidates_json TEXT,
                final_context TEXT,
                answer TEXT,
                sources_json TEXT,
                model_name TEXT,
                prompt_version TEXT,
                index_version TEXT,
                latency_ms REAL,
                error_message TEXT,
                created_at REAL NOT NULL
            );
            """
        )

    logger = TraceLogger(db_path=str(db_path), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(tenant_id=1, query_original="Q", diagnostics={"confidence": 0.5})
    row = logger.get_trace_for_tenant(1, trace_id)

    assert json.loads(row["diagnostics_json"]) == {"confidence": 0.5}
    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    assert version == 2


def test_query_paths_pass_diagnostics_to_trace_static():
    """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

    Diagnostics are constructed inside RAGOrchestrator.answer() and logged
    via trace_logger. The stream endpoint delegates to engine.query() which
    runs the full orchestrator including trace logging. This static test
    verifies the orchestrator still emits diagnostics and the trace lookup
    helper still exists.
    """
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    orchestrator_path = os.path.join(os.path.dirname(__file__), "..", "src", "application", "rag_orchestrator.py")
    main = open(main_path, encoding="utf-8").read()
    orchestrator = open(orchestrator_path, encoding="utf-8").read()

    # Orchestrator must still construct diagnostics and log trace.
    assert '"diagnostics": {' in orchestrator
    assert '"context_sufficient": is_sufficient' in orchestrator
    # main.py trace lookup helper must still exist.
    assert 'trace["diagnostics"] = _json_field(row.get("diagnostics_json")) or {}' in main



def test_trace_to_replay_case_keeps_diagnostics_metadata():
    case = trace_to_replay_case({
        "trace_id": "trace-1",
        "tenant_id": 1,
        "query_original": "Q",
        "filter_conditions": "{}",
        "sources_json": "[]",
        "diagnostics_json": json.dumps({
            "confidence": 0.77,
            "context_sufficient": True,
        }),
        "answer": "A",
    })

    assert case.metadata["diagnostics"] == {
        "confidence": 0.77,
        "context_sufficient": True,
    }



def test_trace_logger_uses_shared_migration_helper_static():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "services", "trace.py")
    content = open(path, encoding="utf-8").read()

    assert "run_component_migrations" in content
    assert "ensure_column" in content
    assert "def _migrate_to_v2" in content
