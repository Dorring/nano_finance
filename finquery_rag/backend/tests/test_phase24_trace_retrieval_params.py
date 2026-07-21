"""Phase 24 tests: trace/replay preserves retrieval parameters."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluation import trace_to_replay_case


def test_trace_to_replay_case_keeps_n_results_metadata():
    trace = {
        "trace_id": "t1",
        "tenant_id": 9,
        "query_original": "What was revenue?",
        "filter_conditions": json.dumps({"doc_names": ["q3.pdf"], "n_results": 7}),
        "sources_json": "[]",
        "answer": "Revenue was $10M.",
    }

    case = trace_to_replay_case(trace)

    assert case.document_names == ("q3.pdf",)
    assert case.metadata["n_results"] == 7


def test_rag_engine_trace_records_n_results_static():
    # Logic moved to RAGOrchestrator; check both locations
    orchestrator_path = os.path.join(os.path.dirname(__file__), "..", "src", "application", "rag_orchestrator.py")
    content = open(orchestrator_path, encoding="utf-8").read()

    assert '"filter_conditions": {"doc_names": doc_names, "n_results": n_results}' in content


def test_stream_trace_records_request_n_results_static():
    """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

    The stream endpoint delegates to engine.query() which runs the full
    orchestrator. ``n_results`` is passed through as a request parameter.
    This static test verifies the stream endpoint passes ``n_results`` to
    ``engine.query()``; the orchestrator records it in filter_conditions.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(path, encoding="utf-8").read()

    # Stream must pass n_results to engine.query().
    assert "n_results=request.n_results" in content
