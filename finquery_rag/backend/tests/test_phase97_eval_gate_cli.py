import json

from src.evaluation.eval_cli import main as eval_cli_main


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _case(case_id="c1"):
    return {
        "id": case_id,
        "question": "What was revenue?",
        "expected_answer_contains": ["revenue"],
        "expected_numbers": ["100"],
        "expected_sources": [{"filename": "report.pdf", "page": 1}],
    }


def _prediction(case_id="c1", *, answer="Revenue was 100."):
    return {
        "id": case_id,
        "answer": answer,
        "sources": [{"filename": "report.pdf", "page": 1}],
        "retrieved_chunks": [{"filename": "report.pdf", "page": 1}],
    }


def test_eval_cli_gate_passes_and_writes_report_and_junit(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    junit_path = tmp_path / "gate.xml"
    _write_jsonl(cases_path, [_case()])
    _write_jsonl(predictions_path, [_prediction()])

    code = eval_cli_main([
        "gate",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--out",
        str(report_path),
        "--junit-out",
        str(junit_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    junit = junit_path.read_text(encoding="utf-8")
    assert code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert payload["summary"]["pass_rate"] == 1.0
    assert report["summary"]["missing_predictions"] == 0
    assert '<testsuite name="finquery-eval-gate" tests="2" failures="0">' in junit


def test_eval_cli_gate_fails_on_threshold_and_missing_predictions(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    junit_path = tmp_path / "gate.xml"
    _write_jsonl(cases_path, [_case("c1"), _case("c2")])
    _write_jsonl(predictions_path, [_prediction("c1", answer="No matching value.")])

    code = eval_cli_main([
        "gate",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--min-pass-rate",
        "0.75",
        "--max-missing",
        "0",
        "--junit-out",
        str(junit_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    junit = junit_path.read_text(encoding="utf-8")
    assert code == 1
    assert payload["passed"] is False
    assert [item["name"] for item in payload["failed_checks"]] == [
        "min_pass_rate",
        "max_missing_predictions",
    ]
    assert payload["missing_case_ids"] == ["c2"]
    assert "FinQuery eval gate failed:" in captured.err
    assert "Missing predictions: c2" in captured.err
    assert 'failures="2"' in junit


def test_eval_cli_gate_fails_on_baseline_regression(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    baseline_path = tmp_path / "baseline.json"
    comparison_path = tmp_path / "comparison.json"
    _write_jsonl(cases_path, [_case()])
    _write_jsonl(predictions_path, [_prediction(answer="No supporting answer.")])
    baseline_path.write_text(json.dumps({
        "summary": {"pass_rate": 1.0, "citation_recall": 1.0},
        "cases": [{"id": "c1", "passed": True}],
    }), encoding="utf-8")

    code = eval_cli_main([
        "gate",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--baseline",
        str(baseline_path),
        "--min-pass-rate",
        "0",
        "--comparison-out",
        str(comparison_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["failed_checks"][-1]["name"] == "baseline_regression"
    assert "baseline_regression" in captured.err
    assert comparison["passed"] is False
    assert "pass_rate" in comparison["regressions"]


def test_eval_cli_gate_rejects_invalid_thresholds(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(cases_path, [_case()])
    _write_jsonl(predictions_path, [_prediction()])

    code = eval_cli_main([
        "gate",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--min-pass-rate",
        "1.1",
    ])

    captured = capsys.readouterr()
    assert code == 2
    assert "min-pass-rate must be <= 1" in captured.err
