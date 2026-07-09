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
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert "confidence = engine._compute_confidence(chunks)" in content
    assert "context_sufficient=False, confidence=confidence" in content
    assert "context_sufficient=True, confidence=confidence" in content
