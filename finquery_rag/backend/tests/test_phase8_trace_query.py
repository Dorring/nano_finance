"""Phase 8B tests: tenant-scoped trace query/export."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.trace import TraceLogger


def _logger(tmp_path):
    return TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=True)


def _log(logger, tenant_id, query, latency=10, error=None):
    return logger.log(
        tenant_id=tenant_id,
        query_original=query,
        answer="answer",
        sources=[{"filename": "doc.pdf", "page": 1}],
        latency_ms=latency,
        error_message=error,
    )


def test_get_trace_for_tenant_allows_owner(tmp_path):
    logger = _logger(tmp_path)
    trace_id = _log(logger, 1, "q1")

    row = logger.get_trace_for_tenant(1, trace_id)

    assert row["trace_id"] == trace_id
    assert row["tenant_id"] == 1


def test_get_trace_for_tenant_denies_other_tenant(tmp_path):
    logger = _logger(tmp_path)
    trace_id = _log(logger, 1, "q1")

    assert logger.get_trace_for_tenant(2, trace_id) is None


def test_query_traces_is_tenant_scoped_and_ordered(tmp_path):
    logger = _logger(tmp_path)
    _log(logger, 1, "old")
    _log(logger, 2, "other")
    _log(logger, 1, "new")

    rows = logger.query_traces(tenant_id=1, limit=10)

    assert [row["query_original"] for row in rows] == ["new", "old"]
    assert all(row["tenant_id"] == 1 for row in rows)


def test_query_traces_supports_limit_and_offset(tmp_path):
    logger = _logger(tmp_path)
    for i in range(5):
        _log(logger, 1, f"q{i}")

    rows = logger.query_traces(tenant_id=1, limit=2, offset=1)

    assert len(rows) == 2
    assert rows[0]["query_original"] == "q3"


def test_query_traces_filters_error_only(tmp_path):
    logger = _logger(tmp_path)
    _log(logger, 1, "ok")
    _log(logger, 1, "bad", error="timeout")

    rows = logger.query_traces(tenant_id=1, error_only=True)

    assert len(rows) == 1
    assert rows[0]["query_original"] == "bad"


def test_query_traces_filters_created_range(tmp_path):
    logger = _logger(tmp_path)
    _log(logger, 1, "early")
    _log(logger, 1, "late")
    rows = logger.query_traces(tenant_id=1, limit=10)
    late_ts = rows[0]["created_at"]

    filtered = logger.query_traces(tenant_id=1, created_after=late_ts)

    assert len(filtered) == 1
    assert filtered[0]["query_original"] == "late"


def test_query_traces_fail_closed_missing_tenant(tmp_path):
    logger = _logger(tmp_path)
    _log(logger, 1, "q1")

    assert logger.query_traces(tenant_id=None) == []


def test_export_traces_jsonl(tmp_path):
    logger = _logger(tmp_path)
    _log(logger, 1, "q1")
    _log(logger, 2, "q2")
    out = tmp_path / "traces.jsonl"

    count = logger.export_traces_jsonl(tenant_id=1, output_path=out)

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["tenant_id"] == 1
    assert rows[0]["query_original"] == "q1"
