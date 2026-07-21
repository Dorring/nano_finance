"""Phase 3 SSE contract tests.

Verifies that the SSE streaming response structure remains compatible
when calculation results are present or absent. Tests the
``make_stream_done_event`` / ``make_stream_error_event`` helpers that
produce the final SSE event consumed by the frontend.
"""
import json
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.services.streaming import (
    make_stream_done_event,
    make_stream_error_event,
)


def _calc_result(status=CalculationStatus.EXECUTED, operation=CalculationOperation.SUM):
    return CalculationResult(
        status=status,
        operation=operation,
        value=Decimal("42") if status is CalculationStatus.EXECUTED else None,
        unit="base" if status is CalculationStatus.EXECUTED else None,
        formula="a + b" if status is CalculationStatus.EXECUTED else None,
        formula_version="sum.v1",
        target_metric=operation.value,
        operands=(
            CalculationOperand(
                name="a", value=Decimal("20"), source_text="20", evidence_chunk_id="c1",
            ),
        ) if status is CalculationStatus.EXECUTED else (),
        error_code=None if status is CalculationStatus.EXECUTED else "SOME_ERROR",
        error_message=None if status is CalculationStatus.EXECUTED else "error",
    )


def _parse_sse_event(sse_str: str) -> dict:
    """Parse one SSE 'data: ...' line into a dict."""
    assert sse_str.startswith("data: ")
    assert sse_str.endswith("\n\n")
    payload = sse_str[len("data: "):].rstrip("\n")
    return json.loads(payload)


class TestSSEDoneEventCalculations:
    """Points 1-4, 10-12: SSE done event carries calculation results."""

    def test_ordinary_done_event_no_calculations(self):
        """Point 1: ordinary done event omits 'calculations'."""
        ev = _parse_sse_event(
            make_stream_done_event(sources=[], context_sufficient=True, trace_id="t1")
        )
        assert ev["type"] == "done"
        assert "calculations" not in ev

    def test_done_event_with_executed_calculation(self):
        """Point 2 + 10: SSE final event carries EXECUTED calculation."""
        calc = _calc_result().to_dict()
        ev = _parse_sse_event(
            make_stream_done_event(
                sources=[], context_sufficient=True, trace_id="t1",
                calculations=[calc],
            )
        )
        assert ev["type"] == "done"
        assert "calculations" in ev
        assert len(ev["calculations"]) == 1
        assert ev["calculations"][0]["status"] == "executed"

    def test_done_event_with_blocked_calculation(self):
        """Point 3: SSE final event carries BLOCKED structured status."""
        calc = _calc_result(CalculationStatus.BLOCKED).to_dict()
        ev = _parse_sse_event(
            make_stream_done_event(calculations=[calc])
        )
        assert ev["calculations"][0]["status"] == "blocked"
        assert ev["calculations"][0]["error_code"] == "SOME_ERROR"

    def test_done_event_with_failed_calculation(self):
        """Point 4: SSE final event carries FAILED safe error."""
        calc = _calc_result(CalculationStatus.FAILED).to_dict()
        ev = _parse_sse_event(
            make_stream_done_event(calculations=[calc])
        )
        assert ev["calculations"][0]["status"] == "failed"
        assert ev["calculations"][0]["error_code"] == "SOME_ERROR"
        # Safe: no internal stack trace exposed
        assert "traceback" not in str(ev["calculations"][0]).lower()

    def test_done_event_all_legacy_fields_present(self):
        """Point 5: SSE done event retains all original fields."""
        ev = _parse_sse_event(
            make_stream_done_event(
                sources=[{"doc": "a"}],
                confidence=0.9,
                context_sufficient=True,
                intent="numeric",
                intent_confidence=0.8,
                trace_id="t-1",
            )
        )
        expected = {
            "type", "sources", "confidence", "context_sufficient",
            "intent", "intent_confidence", "trace_id",
        }
        assert expected.issubset(set(ev.keys()))

    def test_done_event_field_types_unchanged(self):
        """Point 6: field types in SSE done event are unchanged."""
        ev = _parse_sse_event(
            make_stream_done_event(
                sources=[{"doc": "a"}],
                confidence=0.9,
                context_sufficient=True,
                intent="numeric",
                intent_confidence=0.8,
                trace_id="t-1",
            )
        )
        assert isinstance(ev["type"], str)
        assert isinstance(ev["sources"], list)
        assert isinstance(ev["confidence"], (int, float))
        assert isinstance(ev["context_sufficient"], bool)
        assert isinstance(ev["intent"], str)
        assert isinstance(ev["intent_confidence"], (int, float))
        assert isinstance(ev["trace_id"], str)

    def test_done_event_trace_id_present(self):
        """Point 7: Trace ID is present in SSE done event."""
        ev = _parse_sse_event(make_stream_done_event(trace_id="trace-xyz"))
        assert ev["trace_id"] == "trace-xyz"

    def test_done_event_sources_compatible(self):
        """Point 8: Sources in SSE done event are compatible (list)."""
        src = [{"document_name": "doc1", "page": 1}]
        ev = _parse_sse_event(make_stream_done_event(sources=src))
        assert isinstance(ev["sources"], list)
        assert ev["sources"][0]["document_name"] == "doc1"

    def test_calculations_optional_in_sse(self):
        """Point 9: calculations is Optional in SSE done event."""
        ev_no_calc = _parse_sse_event(make_stream_done_event())
        assert "calculations" not in ev_no_calc
        calc = _calc_result().to_dict()
        ev_with_calc = _parse_sse_event(
            make_stream_done_event(calculations=[calc])
        )
        assert "calculations" in ev_with_calc

    def test_old_frontend_ignores_calculations(self):
        """Point 11: old frontend ignoring 'calculations' does not error.

        Both events with and without 'calculations' are valid SSE
        data payloads that can be JSON-parsed without error.
        """
        ev1 = _parse_sse_event(make_stream_done_event(trace_id="t1"))
        assert "calculations" not in ev1
        calc = _calc_result().to_dict()
        ev2 = _parse_sse_event(
            make_stream_done_event(trace_id="t2", calculations=[calc])
        )
        assert "calculations" in ev2

    def test_token_event_format_unchanged(self):
        """Point 12: normal streaming token event format is unchanged.

        Token events are 'data: {"type":"token","content":"..."}\n\n' and
        do not carry calculation fields.
        """
        token_event = f"data: {json.dumps({'type': 'token', 'content': 'hello'})}\n\n"
        parsed = _parse_sse_event(token_event)
        assert parsed["type"] == "token"
        assert parsed["content"] == "hello"
        assert "calculations" not in parsed


class TestSSEErrorEvent:
    """Error events remain compatible (no calculations in error path)."""

    def test_error_event_structure(self):
        ev = _parse_sse_event(
            make_stream_error_event("ERR_001", "something broke", trace_id="t1")
        )
        assert ev["type"] == "error"
        assert ev["detail"]["error_code"] == "ERR_001"
        assert ev["detail"]["message"] == "something broke"
        assert ev["trace_id"] == "t1"
        assert "calculations" not in ev

    def test_error_event_retryable_flag(self):
        ev = _parse_sse_event(
            make_stream_error_event("ERR_002", "fail", retryable=False)
        )
        assert ev["retryable"] is False
