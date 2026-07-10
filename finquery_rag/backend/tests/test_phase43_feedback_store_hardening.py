"""Feedback store input hardening regression checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.feedback import FeedbackStore


def test_feedback_store_rejects_invalid_trace_id_bounds(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))

    assert store.submit(tenant_id=1, trace_id="", rating="up") is None
    assert store.submit(tenant_id=1, trace_id=None, rating="up") is None
    assert store.submit(tenant_id=1, trace_id="t" * 129, rating="up") is None
    assert store.submit(tenant_id=1, trace_id="t" * 128, rating="up") is not None


def test_feedback_store_normalizes_comment_whitespace_and_control_chars(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))

    row = store.submit(
        tenant_id=1,
        trace_id="trace-1",
        rating="down",
        comment="  wrong\x00\n\tperiod   ",
    )

    assert row["comment"] == "wrong period"
    assert store.list_for_tenant(1)[0]["comment"] == "wrong period"


def test_feedback_store_comment_limit_uses_constant(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    row = store.submit(tenant_id=1, trace_id="trace-1", rating="down", comment="x" * 2500)

    assert len(row["comment"]) == store.MAX_COMMENT_CHARS


def test_feedback_store_list_bounds_fail_closed_for_bad_inputs(tmp_path):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    store.submit(tenant_id=1, trace_id="trace-1", rating="up")

    assert store.list_for_tenant(1, limit="bad") == []
    assert store.list_for_tenant(1, offset="bad") == []
    assert store.list_for_tenant(1, limit=-1) == []
    assert store.list_for_tenant(1, offset=-5) == store.list_for_tenant(1, offset=0)


def test_feedback_store_hardening_static_contract():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "services", "feedback.py")
    content = open(path, encoding="utf-8").read()

    assert "MAX_TRACE_ID_LENGTH = 128" in content
    assert "MAX_COMMENT_CHARS = 2000" in content
    assert "def _is_valid_trace_id" in content
    assert "def _clean_comment" in content
    assert "def _normalize_bounds" in content
