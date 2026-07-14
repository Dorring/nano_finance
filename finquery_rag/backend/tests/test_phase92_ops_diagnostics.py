"""Phase 92 tests: tenant-scoped operations diagnostics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.feedback import FeedbackStore
from services.trace import TraceLogger


def test_trace_logger_counts_and_summary_are_tenant_scoped(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), redact_content=False)
    logger.log(tenant_id=1, query_original="q1", answer="a1", latency_ms=10, error_message=None)
    logger.log(tenant_id=1, query_original="q2", answer="a2", latency_ms=30, error_message="boom")
    logger.log(tenant_id=2, query_original="q3", answer="a3", latency_ms=99, error_message="other")

    assert logger.count_traces(1) == 2
    assert logger.count_traces(1, error_only=True) == 1
    assert logger.count_traces(2) == 1
    assert logger.count_traces(None) == 0

    summary = logger.summary_for_tenant(1)
    assert summary["total"] == 2
    assert summary["errors"] == 1
    assert summary["avg_latency_ms"] == 20.0
    assert summary["latest_created_at"] is not None
    assert "q1" not in repr(summary)
    assert "a1" not in repr(summary)


def test_feedback_store_counts_and_summary_are_tenant_scoped(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    store.submit(tenant_id=1, trace_id="t1", rating="up", now=1)
    store.submit(tenant_id=1, trace_id="t2", rating="down", comment="bad answer", now=2)
    store.submit(tenant_id=2, trace_id="t3", rating="down", now=3)

    assert store.count_for_tenant(1) == 2
    assert store.count_for_tenant(1, rating="up") == 1
    assert store.count_for_tenant(1, rating="down") == 1
    assert store.count_for_tenant(1, rating="bad") == 0
    assert store.count_for_tenant(None) == 0

    summary = store.summary_for_tenant(1)
    assert summary == {
        "total": 2,
        "up": 1,
        "down": 1,
        "latest_created_at": 2.0,
    }
    assert "bad answer" not in repr(summary)


def test_ops_summary_endpoint_contract_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()

    assert '@app.get("/ops/summary")' in main
    assert "async def get_ops_summary(current_user: User = Depends(get_current_user))" in main
    assert "return _tenant_ops_summary(current_user.id)" in main
    assert "document_registry.status_summary(user_id)" in main
    assert "session_manager.storage_summary(user_id)" in main
    assert "trace_logger.summary_for_tenant(user_id)" in main
    assert "feedback_store.summary_for_tenant(user_id)" in main


def test_trace_and_feedback_list_contract_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    traces_block = main[main.index('@app.get("/traces")'):main.index('@app.get("/traces/{trace_id}")')]
    feedback_block = main[main.index('@app.get("/feedback")'):main.index('@app.get("/documents/{doc_name}")')]

    assert "count_traces(" in traces_block
    assert '"total_traces": total_traces' in traces_block
    assert '"has_more": normalized_offset + len(rows) < total_traces' in traces_block
    assert "count_for_tenant(current_user.id, rating=rating)" in feedback_block
    assert '"total_feedback": total_feedback' in feedback_block
    assert '"has_more": normalized_offset + len(rows) < total_feedback' in feedback_block