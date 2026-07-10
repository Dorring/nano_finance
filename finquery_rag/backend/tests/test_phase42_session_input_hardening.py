"""Session input hardening regression checks."""
import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.schemas import QueryRequest
from services.session_manager import SessionManager


def _close(mgr):
    try:
        mgr.close()
    except Exception:
        pass


def test_query_request_bounds_session_id_length():
    QueryRequest(question="hello?", session_id="s" * 128)

    with pytest.raises(ValidationError):
        QueryRequest(question="hello?", session_id="")

    with pytest.raises(ValidationError):
        QueryRequest(question="hello?", session_id="s" * 129)


def test_session_manager_rejects_oversized_session_ids(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("s" * 129, 1, "user", "hello")
        assert mgr.get_session_count("s" * 129, 1) == 0
        assert mgr.get_recent_messages("s" * 129, 1) == []
        assert mgr.clear_session("s" * 129, 1) is False
    finally:
        _close(mgr)


def test_session_manager_truncates_oversized_content(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("s1", 1, "user", "x" * (mgr.MAX_CONTENT_CHARS + 20))
        message = mgr.get_recent_messages("s1", 1)[0]

        assert len(message["content"]) <= mgr.MAX_CONTENT_CHARS + len("\n[truncated]")
        assert message["content"].endswith("\n[truncated]")
    finally:
        _close(mgr)


def test_session_manager_bounds_oversized_metadata(tmp_path):
    mgr = SessionManager(db_path=str(tmp_path / "sessions.db"))
    try:
        mgr.add_message("s1", 1, "assistant", "ok", metadata={"blob": "x" * (mgr.MAX_METADATA_JSON_CHARS + 20)})
        message = mgr.get_recent_messages("s1", 1)[0]

        assert message["metadata"] == {"truncated": True}
    finally:
        _close(mgr)


def test_session_endpoints_validate_session_id_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()

    assert "def _validate_session_id" in main
    assert 'raise api_error(400, "invalid_session_id", "session_id must be 1-128 characters")' in main
    assert "session_id = _validate_session_id(request.session_id)" in main
    assert "session_id = _validate_session_id(session_id)" in main
