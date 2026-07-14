"""Phase 83 tests: replay export hardening."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import (
    export_replay_cases_from_feedback,
    export_replay_cases_from_traces,
    trace_to_replay_case,
)
from services.feedback import FeedbackStore
from services.trace import TraceLogger
from src.eval_cli import main as eval_cli_main


def _trace(trace_id="t1", **overrides):
    payload = {
        "trace_id": trace_id,
        "tenant_id": 1,
        "query_original": "What was revenue?",
        "filter_conditions": json.dumps({"doc_names": ["report.pdf"]}),
        "sources_json": json.dumps([{"filename": "report.pdf", "page": 1}]),
        "diagnostics_json": json.dumps({"best_score": 0.9}),
        "answer": "Revenue was $10M.",
    }
    payload.update(overrides)
    return payload


def test_trace_to_replay_case_rejects_invalid_filter_json():
    with pytest.raises(ValueError, match="filter_conditions must be valid JSON"):
        trace_to_replay_case(_trace(filter_conditions="{bad json"))


def test_trace_to_replay_case_rejects_wrong_json_field_types():
    with pytest.raises(ValueError, match="filter_conditions must be a JSON object"):
        trace_to_replay_case(_trace(filter_conditions=json.dumps(["not", "object"])))
    with pytest.raises(ValueError, match="sources_json must be a JSON array"):
        trace_to_replay_case(_trace(sources_json=json.dumps({"not": "array"})))
    with pytest.raises(ValueError, match="diagnostics_json must be a JSON object"):
        trace_to_replay_case(_trace(diagnostics_json=json.dumps(["not", "object"])))


def test_export_replay_cases_rejects_duplicate_case_ids_without_writing(tmp_path):
    out = tmp_path / "replay.jsonl"

    with pytest.raises(ValueError, match="duplicate replay case id 'dup'"):
        export_replay_cases_from_traces([_trace("dup"), _trace("dup")], out)

    assert not out.exists()


def test_feedback_replay_export_rejects_duplicate_case_ids_without_writing(tmp_path):
    out = tmp_path / "feedback_replay.jsonl"
    feedback_rows = [
        {"feedback_id": "fb1", "trace_id": "dup", "rating": "down"},
        {"feedback_id": "fb2", "trace_id": "dup", "rating": "down"},
    ]

    with pytest.raises(ValueError, match="duplicate replay case id 'dup'"):
        export_replay_cases_from_feedback(feedback_rows, lambda _tid: _trace("dup"), out)

    assert not out.exists()


def test_eval_cli_replay_from_traces_reports_bad_trace_without_output(tmp_path, capsys):
    trace_db = tmp_path / "trace.db"
    out = tmp_path / "replay.jsonl"
    logger = TraceLogger(db_path=str(trace_db), sample_rate=1.0, redact_content=True)
    logger.log(
        tenant_id=1,
        query_original="Q",
        filter_conditions="{bad json",
        answer="A",
    )

    code = eval_cli_main([
        "replay-from-traces",
        "--db",
        str(trace_db),
        "--tenant-id",
        "1",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "filter_conditions must be a JSON object" in captured.err
    assert not out.exists()


def test_eval_cli_feedback_to_replay_reports_bad_trace_without_output(tmp_path, capsys):
    trace_db = tmp_path / "trace.db"
    feedback_db = tmp_path / "feedback.db"
    out = tmp_path / "feedback_replay.jsonl"
    logger = TraceLogger(db_path=str(trace_db), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(
        tenant_id=1,
        query_original="Q",
        filter_conditions="{bad json",
        answer="A",
    )
    feedback = FeedbackStore(db_path=str(feedback_db))
    feedback.submit(tenant_id=1, trace_id=trace_id, rating="down", comment="bad", now=1)

    code = eval_cli_main([
        "feedback-to-replay",
        "--feedback-db",
        str(feedback_db),
        "--trace-db",
        str(trace_db),
        "--tenant-id",
        "1",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "filter_conditions must be a JSON object" in captured.err
    assert not out.exists()


