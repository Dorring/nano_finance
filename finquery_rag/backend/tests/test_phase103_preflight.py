import json
import sqlite3

from src.evaluation.eval_cli import main as eval_cli_main
from src.services.document_registry import DocumentRegistry
from src.services.preflight import build_preflight_report
from src.services.session_manager import SessionManager
from src.services.trace import TraceLogger


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def _create_bm25(path):
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
        conn.execute(
            "INSERT INTO chunk_store(doc_id, content, metadata_json, user_id, doc_name) VALUES (?, ?, ?, ?, ?)",
            ("user_1_report.pdf::1", "redacted", "{}", 1, "report.pdf"),
        )
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("redacted", "user_1_report.pdf::1"))
        conn.commit()


def _create_chroma(path):
    path.mkdir()
    with sqlite3.connect(path / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE embeddings (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO embeddings(id) VALUES (?)", ("user_1_report.pdf::1",))
        conn.commit()


def _create_eval_files(tmp_path):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    baseline = tmp_path / "baseline.json"
    _write_jsonl(cases, [{
        "id": "c1",
        "question": "What was revenue?",
        "expected_sources": [{"filename": "report.pdf", "page": 1}],
        "expected_answer_contains": ["Revenue"],
        "expected_intent": "document_qa",
        "tags": ["smoke", "citation"],
    }])
    _write_jsonl(predictions, [{
        "id": "c1",
        "answer": "Revenue was 10.",
        "intent": "document_qa",
        "sources": [{"filename": "report.pdf", "page": 1}],
        "retrieved_chunks": [{"filename": "report.pdf", "page": 1}],
    }])
    report = {
        "summary": {
            "pass_rate": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "retrieval_precision": 1.0,
            "retrieval_recall": 1.0,
            "answer_contains": 1.0,
            "number_accuracy": 1.0,
            "no_answer_accuracy": 1.0,
            "calculation_accuracy": 1.0,
            "answer_calculation_consistency": 1.0,
            "intent_accuracy": 1.0,
            "missing_predictions": 0,
        },
        "cases": [{"id": "c1", "passed": True}],
    }
    baseline.write_text(json.dumps(report), encoding="utf-8")
    return cases, predictions, baseline


def _prepare_ready_runtime(tmp_path, monkeypatch):
    bm25 = tmp_path / "bm25.db"
    registry = tmp_path / "registry.db"
    sessions = tmp_path / "sessions.db"
    trace = tmp_path / "trace.db"
    feedback = tmp_path / "feedback.db"
    chroma = tmp_path / "chroma"
    _create_bm25(bm25)
    _create_chroma(chroma)
    DocumentRegistry(db_path=str(registry)).register("doc1", 1, "report.pdf", "hash", chunk_count=1, status="ready")
    session_manager = SessionManager(db_path=str(sessions))
    session_manager.close()
    TraceLogger(db_path=str(trace))
    with sqlite3.connect(feedback) as conn:
        conn.execute("CREATE TABLE answer_feedback (feedback_id TEXT PRIMARY KEY)")
    monkeypatch.setenv("CHROMA_PATH", str(chroma))
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", str(registry))
    monkeypatch.setenv("SESSIONS_DB_PATH", str(sessions))
    return bm25, registry, chroma, trace, feedback


def test_build_preflight_report_passes_ready_runtime(tmp_path, monkeypatch):
    bm25, registry, chroma, trace, feedback = _prepare_ready_runtime(tmp_path, monkeypatch)
    cases, predictions, baseline = _create_eval_files(tmp_path)

    report = build_preflight_report(
        cases_path=cases,
        predictions_path=predictions,
        baseline_path=baseline,
        bm25_db_path=str(bm25),
        registry_db_path=str(registry),
        chroma_path=str(chroma),
        trace_db_path=str(trace),
        feedback_db_path=str(feedback),
        min_cases=1,
        required_tags=("smoke", "citation"),
        require_expected_intent=True,
    )

    assert report["passed"] is True
    assert all(report["sections"].values())
    assert report["summary"]["eval_pass_rate"] == 1.0
    assert report["summary"]["retrieval_recall_at_5"] == 1.0


def test_eval_cli_preflight_warn_only_writes_degraded_report(tmp_path, capsys):
    cases, predictions, baseline = _create_eval_files(tmp_path)
    out = tmp_path / "preflight.json"

    code = eval_cli_main([
        "preflight",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--baseline",
        str(baseline),
        "--bm25-db",
        str(tmp_path / "missing_bm25.db"),
        "--out",
        str(out),
        "--warn-only",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["passed"] is False
    assert "health" in payload["summary"]["failed_sections"]
    assert written["summary"] == payload["summary"]
    assert "FinQuery preflight failed:" in captured.err


def test_eval_cli_preflight_rejects_bad_threshold(tmp_path, capsys):
    cases, predictions, baseline = _create_eval_files(tmp_path)

    code = eval_cli_main([
        "preflight",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--baseline",
        str(baseline),
        "--min-pass-rate",
        "1.5",
    ])

    captured = capsys.readouterr()
    assert code == 2
    assert "min_pass_rate must be between 0 and 1" in captured.err
