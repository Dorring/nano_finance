"""Phase 93 tests: tenant-scoped replay export APIs."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluation import feedback_to_replay_case, trace_to_replay_case


def _trace(trace_id="trace-1", question="What is revenue?"):
    return {
        "trace_id": trace_id,
        "tenant_id": 1,
        "query_original": question,
        "filter_conditions": json.dumps({"doc_names": ["annual.pdf"], "n_results": 5}),
        "sources_json": json.dumps([{"filename": "annual.pdf", "page": 1, "chunk_id": "user_1_annual.pdf::1"}]),
        "diagnostics_json": json.dumps({"confidence": 0.9}),
        "answer": "Revenue was 123.",
        "intent": "financial_qa",
        "model_name": "test-model",
        "created_at": 10.0,
    }


def test_trace_replay_case_payload_is_content_minimized():
    case = trace_to_replay_case(_trace("ok")).to_dict()

    assert case["id"] == "ok"
    assert case["document_names"] == ["annual.pdf"]
    assert case["expected_sources"][0]["filename"] == "annual.pdf"
    assert case["metadata"]["n_results"] == 5
    assert "Revenue was 123" not in repr(case)
    assert "final_context" not in repr(case)


def test_feedback_replay_case_adds_feedback_metadata_without_expected_answer_text():
    feedback = {"feedback_id": "fb1", "trace_id": "t1", "rating": "down", "comment": "wrong", "created_at": 11.0}
    case = feedback_to_replay_case(feedback, _trace("t1")).to_dict()

    assert case["id"] == "t1"
    assert "feedback_down" in case["tags"]
    assert "feedback_replay" in case["tags"]
    assert case["metadata"]["feedback_id"] == "fb1"
    assert case["metadata"]["feedback_comment"] == "wrong"
    assert "Revenue was 123" not in repr(case)


def test_replay_api_contract_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    traces_block = main[main.index('@app.get("/replay/traces")'):main.index('@app.get("/replay/feedback")')]
    feedback_block = main[main.index('@app.get("/replay/feedback")'):main.index('@app.post("/feedback"')]

    assert 'feedback_to_replay_case' in main
    assert 'trace_to_replay_case' in main
    assert "def _replay_cases_payload_from_traces(rows: list[dict])" in main
    assert "def _replay_cases_payload_from_feedback(feedback_rows: list[dict], trace_lookup)" in main
    assert "current_user: User = Depends(get_current_user)" in traces_block
    assert "logger.query_traces(" in traces_block
    assert "logger.count_traces(" in traces_block
    assert "_replay_cases_payload_from_traces(rows)" in traces_block
    assert '"has_more": normalized_offset + len(rows) < total_traces' in traces_block
    assert "feedback_store.list_for_tenant(" in feedback_block
    assert "feedback_store.count_for_tenant(current_user.id, rating=rating)" in feedback_block
    assert "logger.get_trace_for_tenant(current_user.id, trace_id)" in feedback_block
    assert "_replay_cases_payload_from_feedback(" in feedback_block