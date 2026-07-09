"""Phase 24 tests: trace/replay preserves retrieval parameters."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import trace_to_replay_case


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
    path = os.path.join(os.path.dirname(__file__), "..", "src", "services", "rag_engine.py")
    content = open(path, encoding="utf-8").read()

    assert '"filter_conditions": {"doc_names": doc_names, "n_results": n_results}' in content


def test_stream_trace_records_request_n_results_static():
    path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(path, encoding="utf-8").read()

    assert '"filter_conditions": {"doc_names": doc_names or [], "n_results": request.n_results}' in content
