"""Phase 94 tests: in-memory evaluation score/compare APIs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.evaluation import compare_reports, evaluate_payload


def _case(case_id="c1"):
    return {
        "id": case_id,
        "question": "What was revenue?",
        "expected_answer_contains": ["123"],
        "expected_sources": [{"filename": "annual.pdf", "page": 1}],
    }


def _prediction(case_id="c1", answer="Revenue was 123."):
    return {
        "id": case_id,
        "answer": answer,
        "sources": [{"filename": "annual.pdf", "page": 1}],
        "latency_ms": 25,
    }


def test_eval_payload_scores_predictions_without_file_io():
    report = evaluate_payload([_case()], [_prediction()])

    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["scored_cases"] == 1
    assert report["summary"]["pass_rate"] == 1.0
    assert report["cases"][0]["passed"] is True


def test_eval_payload_rejects_duplicate_predictions():
    try:
        evaluate_payload([_case()], [_prediction(), _prediction()])
    except ValueError as exc:
        assert "duplicate prediction id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_eval_comparison_flags_regression():
    baseline = evaluate_payload([_case()], [_prediction()])
    candidate = evaluate_payload([_case()], [_prediction(answer="wrong")])

    comparison = compare_reports(baseline, candidate, 0.0)

    assert comparison["passed"] is False
    assert comparison["metric_deltas"]["pass_rate"]["delta"] < 0


def test_eval_api_contract_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    schemas = open(os.path.join(root, "src", "models", "schemas.py"), encoding="utf-8").read()
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    score_block = main[main.index('@app.post("/eval/score")'):main.index('@app.post("/eval/compare")')]
    compare_block = main[main.index('@app.post("/eval/compare")'):main.index('@app.post("/feedback"')]

    assert "class EvalScoreRequest" in schemas
    assert "class EvalCompareRequest" in schemas
    assert "compare_reports, evaluate_payload" in main
    assert "current_user: User = Depends(get_current_user)" in score_block
    assert "return _eval_report_from_payload(request.cases, request.predictions)" in score_block
    assert "current_user: User = Depends(get_current_user)" in compare_block
    assert "_eval_comparison_from_payload(" in compare_block