"""Phase 18A tests: answer feedback storage/API surface."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.feedback import FeedbackStore


def test_feedback_store_is_tenant_scoped_and_ordered(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    first = store.submit(tenant_id=1, trace_id="t1", rating="up", comment="good", now=1)
    second = store.submit(tenant_id=1, trace_id="t2", rating="down", comment="bad", now=2)
    store.submit(tenant_id=2, trace_id="t3", rating="down", now=3)

    rows = store.list_for_tenant(1)

    assert [row["feedback_id"] for row in rows] == [second["feedback_id"], first["feedback_id"]]
    assert {row["tenant_id"] for row in rows} == {1}


def test_feedback_store_fail_closed_and_filters_rating(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    store.submit(tenant_id=1, trace_id="t1", rating="up", now=1)
    store.submit(tenant_id=1, trace_id="t2", rating="down", now=2)

    assert store.submit(tenant_id=None, trace_id="t3", rating="up") is None
    assert store.submit(tenant_id=1, trace_id="", rating="up") is None
    assert store.submit(tenant_id=1, trace_id="t4", rating="bad") is None
    assert [row["rating"] for row in store.list_for_tenant(1, rating="down")] == ["down"]
    assert store.list_for_tenant(1, rating="bad") == []
    assert store.list_for_tenant(None) == []


def test_feedback_store_truncates_comment(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    row = store.submit(tenant_id=1, trace_id="t1", rating="down", comment="x" * 2500)

    assert len(row["comment"]) == 2000


def test_main_exposes_authenticated_feedback_routes():
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert '@app.post("/feedback", response_model=FeedbackResponse)' in content
    assert '@app.get("/feedback")' in content
    assert "Depends(get_current_user)" in content
    assert "get_trace_for_tenant(current_user.id, request.trace_id)" in content
    public_feedback_block = content[content.index("def _public_feedback"):content.index("def _public_trace")]
    assert '"tenant_id"' not in public_feedback_block
