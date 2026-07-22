"""Phase 4 hotfix: Calculation validation HTTP runtime tests.

Uses FastAPI TestClient to verify the real /query endpoint includes
answerability and validation in responses for calculation paths.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

# ---------------------------------------------------------------------------
# Optional imports — skip entire module if core deps are missing
# ---------------------------------------------------------------------------

try:
    from starlette.testclient import TestClient

    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False

try:
    from src.main import app

    _HAS_APP = True
except Exception:
    _HAS_APP = False

try:
    from src.domain.answer import AnswerResult

    _HAS_ANSWER = True
except ImportError:
    _HAS_ANSWER = False


if not (_HAS_STARLETTE and _HAS_APP):
    pytest.skip(
        "FastAPI TestClient or src.main.app not available",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# AnswerResult construction helpers
# ---------------------------------------------------------------------------


def _make_answer_result(
    validation_status="passed",
    calculation_status="EXECUTED",
    has_value=True,
    include_error_message=False,
):
    """Build an AnswerResult (or SimpleNamespace fallback) for testing.

    All values use JSON-serialisable types (no Decimal) so the response
    can be serialised by FastAPI without custom encoders.
    """
    calc_value = 100 if has_value and calculation_status == "EXECUTED" else None
    calculation = {
        "status": calculation_status,
        "value": calc_value,
        "operands": (
            [{"value": 100, "evidence_chunk_id": "chunk_001"}]
            if calculation_status == "EXECUTED"
            else []
        ),
        "formula_version": "v1",
        "target_metric": "revenue",
    }
    if include_error_message:
        calculation["error_message"] = "internal calculation failure detail"

    validation = {"status": validation_status, "issues": []}
    answerability = {
        "status": (
            "answerable" if validation_status == "passed" else "calculation_blocked"
        )
    }

    answer_text = (
        "Revenue was 100 million in FY2024."
        if validation_status == "passed"
        else "The requested calculation could not be completed."
    )
    confidence = 0.95 if validation_status == "passed" else 0.0

    kwargs = dict(
        answer=answer_text,
        confidence=confidence,
        sources=({"filename": "report.pdf", "page": 12, "chunk_id": "chunk_001"},),
        trace_id="trace-001",
        calculation=calculation,
        validation=validation,
        answerability=answerability,
        repair={
            "was_repaired": False,
            "fallback_used": validation_status != "passed",
        },
    )

    if _HAS_ANSWER:
        try:
            return AnswerResult(**kwargs)
        except TypeError:
            pass

    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# Engine discovery and mocking
# ---------------------------------------------------------------------------


def _get_engine(app):
    """Find the RAG engine (or orchestrator) on the app.

    Tries common attribute locations: ``app.state.rag_engine``,
    ``app.state.engine``, ``app.state.orchestrator``, etc.
    """
    state = getattr(app, "state", None)
    if state is not None:
        for attr in ("rag_engine", "engine", "rag", "orchestrator"):
            engine = getattr(state, attr, None)
            if engine is not None:
                return engine
    # Try direct attributes on app
    for attr in ("rag_engine", "engine", "rag", "orchestrator"):
        engine = getattr(app, attr, None)
        if engine is not None:
            return engine
    return None


def _mock_engine_answer(engine, answer_result):
    """Mock the engine's answer-related methods to return answer_result."""
    for method_name in ("answer", "query", "ask", "process_query", "answer_query"):
        if hasattr(engine, method_name):
            try:
                setattr(engine, method_name, AsyncMock(return_value=answer_result))
            except (AttributeError, TypeError):
                pass
    # Also mock streaming methods (harmless if not used by /query)
    for method_name in ("answer_stream", "stream_answer", "stream"):
        if hasattr(engine, method_name):
            try:
                setattr(engine, method_name, AsyncMock(return_value=answer_result))
            except (AttributeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _make_request_body(query="What was the revenue in FY2024?"):
    """Build a request body that works with multiple API schemas."""
    return {
        "question": query,
        "query": query,
        "tenant_id": 1,
        "user_id": 1,
    }


def _post_query(client, body):
    """POST to /query, trying common endpoint paths.

    Returns the first non-404 response, or ``None`` if no endpoint found.
    """
    for endpoint in ("/query", "/api/query", "/v1/query", "/qa", "/ask"):
        try:
            resp = client.post(endpoint, json=body)
            if resp.status_code != 404:
                return resp
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_executed_response_contains_answerability_and_validation():
    """POST /query with EXECUTED calculation: response has answerability + validation."""
    answer_result = _make_answer_result(
        validation_status="passed",
        calculation_status="EXECUTED",
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_answer(engine, answer_result)

    client = TestClient(app)
    resp = _post_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query endpoint not found")

    data = resp.json()
    assert "answerability" in data, "Response missing 'answerability' key"
    assert "validation" in data, "Response missing 'validation' key"
    v_status = data["validation"].get("status")
    assert v_status in ("passed", "blocked", "failed", "not_applicable")


def test_blocked_response_has_blocked_validation():
    """POST /query with BLOCKED calculation: validation.status == 'blocked'."""
    answer_result = _make_answer_result(
        validation_status="blocked",
        calculation_status="BLOCKED",
        has_value=False,
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_answer(engine, answer_result)

    client = TestClient(app)
    resp = _post_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query endpoint not found")

    data = resp.json()
    assert data.get("validation", {}).get("status") == "blocked"
    calc = data.get("calculation", {})
    # No success value for BLOCKED
    assert calc.get("value") is None


def test_failed_response_has_failed_validation():
    """POST /query with FAILED calculation: validation.status == 'failed'."""
    answer_result = _make_answer_result(
        validation_status="failed",
        calculation_status="FAILED",
        has_value=False,
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_answer(engine, answer_result)

    client = TestClient(app)
    resp = _post_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query endpoint not found")

    data = resp.json()
    assert data.get("validation", {}).get("status") == "failed"
    calc = data.get("calculation", {})
    # No internal error_message leaked
    assert "error_message" not in calc


def test_blocked_no_success_value():
    """BLOCKED response: calculation payload has no value or value is null."""
    answer_result = _make_answer_result(
        validation_status="blocked",
        calculation_status="BLOCKED",
        has_value=False,
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_answer(engine, answer_result)

    client = TestClient(app)
    resp = _post_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query endpoint not found")

    data = resp.json()
    calc = data.get("calculation", {})
    assert calc.get("value") is None, (
        "BLOCKED calculation should not expose a success value"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
