"""Helpers for streaming query responses."""
from __future__ import annotations

import json


def safe_log_query_trace(engine, trace_data: dict) -> str | None:
    """Persist trace data without allowing tracing failures to affect answers."""
    try:
        return engine.trace_logger.log(**trace_data)
    except Exception:
        return None


def make_stream_done_event(**payload) -> str:
    """Build one SSE done event with stable response fields."""
    defaults = {
        "sources": [],
        "confidence": None,
        "context_sufficient": None,
        "intent": None,
        "intent_confidence": None,
        "trace_id": None,
    }
    defaults.update(payload)
    return f"data: {json.dumps({'type': 'done', **defaults})}\n\n"
