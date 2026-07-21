"""Phase 29 tests: stream responses expose retrieval confidence."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.streaming import make_stream_done_event


def test_stream_done_event_can_carry_retrieval_confidence():
    event = make_stream_done_event(
        sources=[],
        context_sufficient=True,
        confidence=0.73,
        intent="document_qa",
        intent_confidence=0.91,
        trace_id="trace-1",
    )

    payload = json.loads(event.removeprefix("data: ").strip())

    assert payload["type"] == "done"
    assert payload["confidence"] == 0.73


def test_stream_query_endpoint_computes_and_returns_confidence_static():
    """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

    Confidence is computed inside the orchestrator (engine.query →
    RAGOrchestrator.answer) and returned in the result dict. The stream
    endpoint reads it from ``result.get("confidence")`` and passes it to
    ``make_stream_done_event``. This static test verifies the unified path.
    """
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    # The unified stream path must call engine.query() and propagate confidence.
    assert "confidence = result.get(\"confidence\")" in content
    assert "confidence=confidence" in content
