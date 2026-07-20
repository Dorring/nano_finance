"""Phase 18C tests: feedback-linked replay export."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluation import (
    export_replay_cases_from_feedback,
    feedback_to_replay_case,
    load_jsonl_cases,
)
from services.feedback import FeedbackStore
from services.trace import TraceLogger
from src.evaluation.eval_cli import main as eval_cli_main


def _trace_logger(tmp_path):
    return TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=True)


def test_feedback_to_replay_case_adds_feedback_metadata(tmp_path):
    trace_logger = _trace_logger(tmp_path)
    trace_id = trace_logger.log(
        tenant_id=7,
        query_original="What was revenue?",
        filter_conditions={"doc_names": ["report.pdf"]},
        sources=[{"filename": "report.pdf", "page": 1}],
        answer="Revenue was $10M.",
    )
    trace = trace_logger.get_trace_for_tenant(7, trace_id)
    feedback = {
        "feedback_id": "fb1",
        "trace_id": trace_id,
        "rating": "down",
        "comment": "wrong period",
        "created_at": 123.0,
    }

    case = feedback_to_replay_case(feedback, trace)
    payload = case.to_dict()

    assert payload["id"] == trace_id
    assert "feedback_down" in payload["tags"]
    assert "feedback_replay" in payload["tags"]
    assert payload["metadata"]["feedback_id"] == "fb1"
    assert payload["metadata"]["feedback_comment"] == "wrong period"
    assert payload["metadata"]["trace_id"] == trace_id


def test_export_replay_cases_from_feedback_skips_missing_traces(tmp_path):
    trace_logger = _trace_logger(tmp_path)
    trace_id = trace_logger.log(tenant_id=1, query_original="Q", answer="A")
    feedback_rows = [
        {"feedback_id": "fb1", "trace_id": trace_id, "rating": "down", "comment": None, "created_at": 1.0},
        {"feedback_id": "fb2", "trace_id": "missing", "rating": "down", "comment": None, "created_at": 2.0},
    ]
    out = tmp_path / "feedback_replay.jsonl"

    cases = export_replay_cases_from_feedback(
        feedback_rows,
        lambda tid: trace_logger.get_trace_for_tenant(1, tid),
        out,
    )

    assert len(cases) == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metadata"]["feedback_id"] == "fb1"
    assert load_jsonl_cases(out)[0].case_id == trace_id


def test_feedback_replay_export_uses_latest_feedback_per_trace(tmp_path):
    trace_logger = _trace_logger(tmp_path)
    trace_id = trace_logger.log(tenant_id=1, query_original="Q", answer="A")
    feedback_store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    feedback_store.submit(tenant_id=1, trace_id=trace_id, rating="up", comment="first", now=1)
    feedback_store.submit(tenant_id=1, trace_id=trace_id, rating="down", comment="latest", now=2)
    out = tmp_path / "feedback_replay.jsonl"

    cases = export_replay_cases_from_feedback(
        feedback_store.list_for_tenant(1, rating="down"),
        lambda tid: trace_logger.get_trace_for_tenant(1, tid),
        out,
    )

    assert len(cases) == 1
    assert cases[0].metadata["feedback_comment"] == "latest"


def test_eval_cli_exposes_feedback_to_replay_command():
    cli_path = os.path.join(os.path.dirname(__file__), "..", "src", "evaluation", "eval_cli.py")
    content = open(cli_path, encoding="utf-8").read()

    assert 'feedback-to-replay' in content
    assert 'FeedbackStore' in content
    assert 'export_replay_cases_from_feedback' in content



def test_eval_cli_rejects_invalid_trace_export_bounds(tmp_path, capsys):
    out = tmp_path / "traces.jsonl"

    code = eval_cli_main([
        "traces",
        "--db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "1",
        "--limit",
        "0",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "limit must be >= 1" in captured.err
    assert not out.exists()


def test_eval_cli_rejects_invalid_trace_time_range(tmp_path, capsys):
    out = tmp_path / "traces.jsonl"

    code = eval_cli_main([
        "traces",
        "--db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "1",
        "--created-after",
        "20",
        "--created-before",
        "10",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "created-after must be <= created-before" in captured.err
    assert not out.exists()


def test_eval_cli_caps_trace_export_limit(tmp_path, capsys):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(tenant_id=1, query_original="Q", answer="A")
    out = tmp_path / "traces.jsonl"

    code = eval_cli_main([
        "traces",
        "--db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "1",
        "--limit",
        "999999",
        "--out",
        str(out),
    ])

    assert code == 0
    assert trace_id in out.read_text(encoding="utf-8")


def test_eval_cli_rejects_invalid_feedback_replay_offset(tmp_path, capsys):
    out = tmp_path / "feedback_replay.jsonl"

    code = eval_cli_main([
        "feedback-to-replay",
        "--feedback-db",
        str(tmp_path / "feedback.db"),
        "--trace-db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "1",
        "--offset",
        "-1",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "offset must be >= 0" in captured.err
    assert not out.exists()


def test_eval_cli_rejects_negative_trace_cleanup_ttl(tmp_path, capsys):
    code = eval_cli_main([
        "traces-cleanup",
        "--db",
        str(tmp_path / "trace.db"),
        "--ttl-seconds",
        "-1",
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "ttl-seconds must be >= 0" in captured.err
