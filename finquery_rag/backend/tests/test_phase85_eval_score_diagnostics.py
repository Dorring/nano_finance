"""Phase 85 tests: eval scoring diagnostics and report schema hardening."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import EvaluationCase, Prediction, compare_reports, evaluate_predictions
from src.eval_cli import main as eval_cli_main


def test_evaluate_predictions_reports_extra_prediction_ids():
    cases = [EvaluationCase.from_dict({"id": "c1", "question": "Q", "expected_answer_contains": ["A"]})]
    predictions = {
        "c1": Prediction.from_dict({"id": "c1", "answer": "A"}),
        "stale": Prediction.from_dict({"id": "stale", "answer": "old"}),
    }

    report = evaluate_predictions(cases, predictions)

    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["total_predictions"] == 2
    assert report["summary"]["extra_predictions"] == 1
    assert report["extra_prediction_ids"] == ["stale"]
    assert "1 predictions did not match any evaluation case" in report["warnings"]


def test_evaluate_predictions_warns_on_empty_case_set():
    report = evaluate_predictions([], {"p1": Prediction.from_dict({"id": "p1", "answer": "A"})})

    assert report["summary"]["total_cases"] == 0
    assert report["summary"]["scored_cases"] == 0
    assert report["summary"]["extra_predictions"] == 1
    assert "no evaluation cases loaded" in report["warnings"]


def test_eval_cli_score_returns_2_for_invalid_cases_without_output(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    out = tmp_path / "report.json"
    cases.write_text(json.dumps({"id": "bad"}) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps({"id": "bad", "answer": "A"}) + "\n", encoding="utf-8")

    code = eval_cli_main([
        "score",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "missing question" in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_eval_cli_score_outputs_extra_prediction_diagnostics(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    out = tmp_path / "report.json"
    cases.write_text(json.dumps({"id": "c1", "question": "Q", "expected_answer_contains": ["A"]}) + "\n", encoding="utf-8")
    predictions.write_text(
        json.dumps({"id": "c1", "answer": "A"}) + "\n" + json.dumps({"id": "extra", "answer": "B"}) + "\n",
        encoding="utf-8",
    )

    code = eval_cli_main([
        "score",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["extra_prediction_ids"] == ["extra"]
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["extra_predictions"] == 1


def test_compare_reports_rejects_bad_report_schema():
    with pytest.raises(ValueError, match="baseline report summary must be an object"):
        compare_reports({"summary": [], "cases": []}, {"summary": {}, "cases": []})
    with pytest.raises(ValueError, match="candidate report cases must be a list"):
        compare_reports({"summary": {}, "cases": []}, {"summary": {}, "cases": {}})
    with pytest.raises(ValueError, match=r"candidate report cases\[0\] missing id"):
        compare_reports({"summary": {}, "cases": []}, {"summary": {}, "cases": [{"passed": True}]})


def test_eval_cli_compare_returns_2_for_bad_report_schema(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out = tmp_path / "comparison.json"
    baseline.write_text(json.dumps({"summary": {}, "cases": []}), encoding="utf-8")
    candidate.write_text(json.dumps({"summary": {}, "cases": [{"passed": True}]}), encoding="utf-8")

    code = eval_cli_main([
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "candidate report cases[0] missing id" in captured.err
    assert captured.out == ""
    assert not out.exists()

