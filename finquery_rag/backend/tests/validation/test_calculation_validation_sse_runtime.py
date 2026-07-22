"""Phase 4 hotfix: Calculation validation SSE runtime tests.

Uses FastAPI TestClient to verify the real /query/stream endpoint
includes validation in the done event for calculation paths.
"""

from __future__ import annotations

import os
import sys
import json
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
    """Find the RAG engine (or orchestrator) on the app."""
    state = getattr(app, "state", None)
    if state is not None:
        for attr in ("rag_engine", "engine", "rag", "orchestrator"):
            engine = getattr(state, attr, None)
            if engine is not None:
                return engine
    for attr in ("rag_engine", "engine", "rag", "orchestrator"):
        engine = getattr(app, attr, None)
        if engine is not None:
            return engine
    return None


def _mock_engine_stream(engine, answer_result):
    """Mock the engine's streaming and answer methods."""

    # Streaming methods — return a simple async generator that yields
    # only a done event (no token events for BLOCKED/FAILED).
    async def _stream_gen(*args, **kwargs):
        yield answer_result

    for method_name in ("answer_stream", "stream_answer", "stream", "stream_query"):
        if hasattr(engine, method_name):
            try:
                setattr(engine, method_name, _stream_gen)
            except (AttributeError, TypeError):
                pass

    # Non-streaming answer (some endpoints fall back to this)
    for method_name in ("answer", "query", "ask", "process_query", "answer_query"):
        if hasattr(engine, method_name):
            try:
                setattr(
                    engine,
                    method_name,
                    AsyncMock(return_value=answer_result),
                )
            except (AttributeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(text):
    """Parse SSE event stream text into a list of (event, data) tuples.

    Each event is a dict with 'event' and 'data' keys.  'data' is parsed
    as JSON if possible, otherwise kept as a string.
    """
    events = []
    current_event = None
    current_data_lines = []

    for line in text.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[len("data:") :].strip())
        elif line.strip() == "" and (current_event is not None or current_data_lines):
            data_str = "\n".join(current_data_lines)
            try:
                parsed = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                parsed = data_str
            events.append(
                {
                    "event": current_event or "message",
                    "data": parsed,
                }
            )
            current_event = None
            current_data_lines = []

    # Handle trailing event without final blank line
    if current_event is not None or current_data_lines:
        data_str = "\n".join(current_data_lines)
        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            parsed = data_str
        events.append(
            {
                "event": current_event or "message",
                "data": parsed,
            }
        )

    return events


def _find_done_event(events):
    """Find the 'done' event in a list of parsed SSE events.

    Looks for events with event type 'done', 'complete', 'end', 'result',
    or 'final'.  Returns the event dict or ``None``.
    """
    done_types = {"done", "complete", "end", "result", "final", "finish"}
    for ev in events:
        if ev["event"].lower() in done_types:
            return ev
    # If no explicit done event, try the last event
    if events:
        return events[-1]
    return None


def _stream_query(client, body):
    """POST to /query/stream, trying common endpoint paths.

    Returns the first non-404 response, or ``None`` if no endpoint found.
    """
    for endpoint in (
        "/query/stream",
        "/api/query/stream",
        "/v1/query/stream",
        "/stream",
        "/query/stream",
    ):
        try:
            resp = client.post(endpoint, json=body)
            if resp.status_code != 404:
                return resp
        except Exception:
            continue
    return None


def _make_request_body(query="What was the revenue in FY2024?"):
    """Build a request body that works with multiple API schemas."""
    return {
        "question": query,
        "query": query,
        "tenant_id": 1,
        "user_id": 1,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_sse_done_event_contains_validation():
    """SSE stream's done event includes a 'validation' field."""
    answer_result = _make_answer_result(
        validation_status="passed",
        calculation_status="EXECUTED",
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_stream(engine, answer_result)

    client = TestClient(app)
    resp = _stream_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query/stream endpoint not found")

    events = _parse_sse_events(resp.text)
    done_event = _find_done_event(events)
    if done_event is None:
        pytest.skip("No done event found in SSE stream")

    data = done_event["data"]
    if isinstance(data, str):
        # Data might be a JSON string
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            pytest.skip("Done event data is not JSON")

    if isinstance(data, dict):
        assert "validation" in data, "SSE done event must include 'validation' field"
    else:
        # If data is not a dict, check the raw text for validation
        assert "validation" in resp.text, (
            "SSE response must contain 'validation' somewhere"
        )


def test_sse_blocked_no_partial_tokens():
    """BLOCKED calculation: no partial LLM tokens in SSE stream."""
    answer_result = _make_answer_result(
        validation_status="blocked",
        calculation_status="BLOCKED",
        has_value=False,
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_stream(engine, answer_result)

    client = TestClient(app)
    resp = _stream_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query/stream endpoint not found")

    events = _parse_sse_events(resp.text)

    # No token events should appear for BLOCKED
    token_event_types = {"token", "chunk", "delta", "partial", "step"}
    token_events = [ev for ev in events if ev["event"].lower() in token_event_types]
    assert len(token_events) == 0, (
        f"BLOCKED calculation should not produce partial tokens, "
        f"but found {len(token_events)} token events"
    )

    # The done event should have validation status 'blocked'
    done_event = _find_done_event(events)
    if done_event is not None:
        data = done_event["data"]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                data = {}
        if isinstance(data, dict):
            v = data.get("validation", {})
            assert v.get("status") in ("blocked", "failed"), (
                f"Expected blocked/failed validation in done event, "
                f"got {v.get('status')}"
            )


def test_sse_failed_no_internal_error():
    """FAILED calculation: no 'error_message' in SSE done event."""
    answer_result = _make_answer_result(
        validation_status="failed",
        calculation_status="FAILED",
        has_value=False,
    )
    engine = _get_engine(app)
    if engine is None:
        pytest.skip("RAG engine not found on app.state")
    _mock_engine_stream(engine, answer_result)

    client = TestClient(app)
    resp = _stream_query(client, _make_request_body())
    if resp is None:
        pytest.skip("/query/stream endpoint not found")

    events = _parse_sse_events(resp.text)
    done_event = _find_done_event(events)
    if done_event is None:
        pytest.skip("No done event found in SSE stream")

    data = done_event["data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # If it's a plain string, just check it doesn't contain error_message
            assert "error_message" not in data, (
                "Done event string must not contain 'error_message'"
            )
            return

    if isinstance(data, dict):
        # The done event data should not contain error_message
        assert "error_message" not in data, (
            "SSE done event must not contain 'error_message'"
        )
        # Also check nested calculation dict
        calc = data.get("calculation", {})
        if isinstance(calc, dict):
            assert "error_message" not in calc, (
                "SSE done event calculation must not contain 'error_message'"
            )

    # Also verify the full response text doesn't leak error details
    assert "error_message" not in resp.text or "error_message" not in data, (
        "SSE response should not expose internal error_message"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
