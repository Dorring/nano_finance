"""Phase 16A tests: trace HTTP API surface."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.trace import TraceLogger


def test_trace_logger_supports_http_lookup_contract(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=True)
    trace_id = logger.log(
        tenant_id=1,
        query_original="What was revenue?",
        filter_conditions={"doc_names": ["report.pdf"]},
        candidates=[{"doc_id": "user_1_report.pdf::1", "score": 0.9}],
        answer="Revenue was $10M.",
        sources=[{"filename": "report.pdf", "page": 1}],
        latency_ms=12.5,
    )

    row = logger.get_trace_for_tenant(1, trace_id)

    assert row["trace_id"] == trace_id
    assert row["tenant_id"] == 1
    assert '"report.pdf"' in row["sources_json"]
    assert logger.get_trace_for_tenant(2, trace_id) is None


def test_main_exposes_authenticated_trace_routes():
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert '@app.get("/traces")' in content
    assert '@app.get("/traces/{trace_id}")' in content
    assert "Depends(get_current_user)" in content
    assert "get_trace_for_tenant(current_user.id, trace_id)" in content
    public_trace_block = content[content.index("def _public_trace"):content.index("######################### API Endpoints")]
    assert '"tenant_id"' not in public_trace_block
