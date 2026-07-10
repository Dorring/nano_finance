"""Trace store input and size hardening regression checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.trace import MAX_JSON_CHARS, MAX_TEXT_CHARS, TraceLogger


def test_trace_logger_rejects_invalid_trace_id_bounds(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0)
    trace_id = logger.log(tenant_id=1, query_original="q")

    assert logger.get_trace("") is None
    assert logger.get_trace("t" * 129) is None
    assert logger.get_trace_for_tenant(1, None) is None
    assert logger.get_trace_for_tenant(1, "t" * 129) is None
    assert logger.get_trace_for_tenant(1, trace_id)["trace_id"] == trace_id


def test_trace_logger_query_bounds_fail_closed_for_bad_inputs(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0)
    logger.log(tenant_id=1, query_original="q")

    assert logger.query_traces(tenant_id=1, limit="bad") == []
    assert logger.query_traces(tenant_id=1, offset="bad") == []
    assert logger.query_traces(tenant_id=1, limit=-1) == []
    assert logger.query_traces(tenant_id=1, offset=-5) == logger.query_traces(tenant_id=1, offset=0)


def test_trace_logger_bounds_large_text_fields(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=False)
    trace_id = logger.log(
        tenant_id=1,
        query_original="q" * (MAX_TEXT_CHARS + 10),
        final_context="c" * (MAX_TEXT_CHARS + 10),
        answer="a" * (MAX_TEXT_CHARS + 10),
        error_message="e" * (MAX_TEXT_CHARS + 10),
    )
    row = logger.get_trace(trace_id)

    assert row["query_original"].endswith("\n[truncated]")
    assert row["final_context"].endswith("\n[truncated]")
    assert row["answer"].endswith("\n[truncated]")
    assert row["error_message"].endswith("\n[truncated]")


def test_trace_logger_bounds_large_json_fields(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0)
    trace_id = logger.log(
        tenant_id=1,
        query_original="q",
        filter_conditions={"blob": "x" * (MAX_JSON_CHARS + 10)},
        candidates=[{"blob": "x" * (MAX_JSON_CHARS + 10)}],
        sources=[{"blob": "x" * (MAX_JSON_CHARS + 10)}],
        diagnostics={"blob": "x" * (MAX_JSON_CHARS + 10)},
    )
    row = logger.get_trace(trace_id)

    assert row["filter_conditions"] == '{"truncated": true}'
    assert row["candidates_json"] == '{"truncated": true}'
    assert row["sources_json"] == '{"truncated": true}'
    assert row["diagnostics_json"] == '{"truncated": true}'


def test_trace_store_hardening_static_contract():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "services", "trace.py")
    content = open(path, encoding="utf-8").read()

    assert "MAX_TRACE_ID_LENGTH = 128" in content
    assert "MAX_TEXT_CHARS = 50000" in content
    assert "MAX_JSON_CHARS = 50000" in content
    assert "def _is_valid_trace_id" in content
    assert "def _normalize_query_bounds" in content
