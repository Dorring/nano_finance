"""Phase 86 tests: trace and feedback eval CLI diagnostics."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.feedback import FeedbackStore
from services.trace import TraceLogger
from src.eval_cli import main as eval_cli_main


def test_trace_export_uses_jsonl_writer_and_creates_parent(tmp_path):
    db = tmp_path / "trace.db"
    out = tmp_path / "nested" / "traces.jsonl"
    logger = TraceLogger(db_path=str(db), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(tenant_id=1, query_original="Q", answer="A")

    count = logger.export_traces_jsonl(tenant_id=1, output_path=out)

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["trace_id"] == trace_id
    assert list(out.parent.glob(f".{out.name}.*.tmp")) == []


def test_eval_cli_traces_rejects_invalid_tenant_id(tmp_path, capsys):
    out = tmp_path / "traces.jsonl"

    code = eval_cli_main([
        "traces",
        "--db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "0",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "tenant-id must be >= 1" in captured.err
    assert not out.exists()


def test_eval_cli_replay_from_traces_supports_offset_and_error_filter(tmp_path, capsys):
    db = tmp_path / "trace.db"
    out = tmp_path / "replay.jsonl"
    logger = TraceLogger(db_path=str(db), sample_rate=1.0, redact_content=True)
    logger.log(tenant_id=1, query_original="normal", answer="A")
    error_trace = logger.log(tenant_id=1, query_original="failed", answer="", error_message="boom")

    code = eval_cli_main([
        "replay-from-traces",
        "--db",
        str(db),
        "--tenant-id",
        "1",
        "--error-only",
        "--offset",
        "0",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 0
    assert "exported 1 replay cases" in captured.out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == error_trace
    assert rows[0]["question"] == "failed"


def test_eval_cli_replay_from_traces_rejects_invalid_bounds(tmp_path, capsys):
    out = tmp_path / "replay.jsonl"

    code = eval_cli_main([
        "replay-from-traces",
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


def test_eval_cli_feedback_replay_rejects_invalid_tenant_id_before_output(tmp_path, capsys):
    out = tmp_path / "feedback_replay.jsonl"

    code = eval_cli_main([
        "feedback-to-replay",
        "--feedback-db",
        str(tmp_path / "feedback.db"),
        "--trace-db",
        str(tmp_path / "trace.db"),
        "--tenant-id",
        "0",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "tenant-id must be >= 1" in captured.err
    assert not out.exists()


def test_eval_cli_feedback_replay_uses_validated_tenant_id(tmp_path, capsys):
    trace_db = tmp_path / "trace.db"
    feedback_db = tmp_path / "feedback.db"
    out = tmp_path / "feedback_replay.jsonl"
    trace_logger = TraceLogger(db_path=str(trace_db), sample_rate=1.0, redact_content=True)
    trace_id = trace_logger.log(tenant_id=2, query_original="Q", answer="A")
    feedback_store = FeedbackStore(db_path=str(feedback_db))
    feedback_store.submit(tenant_id=2, trace_id=trace_id, rating="down", comment="bad", now=1)

    code = eval_cli_main([
        "feedback-to-replay",
        "--feedback-db",
        str(feedback_db),
        "--trace-db",
        str(trace_db),
        "--tenant-id",
        "2",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 0
    assert "exported 1 feedback replay cases" in captured.out
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["metadata"]["tenant_id"] == 2


def test_eval_cli_trace_cleanup_rejects_invalid_optional_tenant_id(tmp_path, capsys):
    code = eval_cli_main([
        "traces-cleanup",
        "--db",
        str(tmp_path / "trace.db"),
        "--ttl-seconds",
        "1",
        "--tenant-id",
        "0",
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "tenant-id must be >= 1" in captured.err
