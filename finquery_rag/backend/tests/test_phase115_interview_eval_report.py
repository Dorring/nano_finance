import json

from src.evaluation.eval_cli import main as eval_cli_main
from src.evaluation.evaluation import (
    EvaluationCase,
    Prediction,
    build_interview_report,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_interview_report_groups_resume_metrics_and_weak_cases():
    cases = [
        EvaluationCase.from_dict({
            "id": "title",
            "question": "What is the title?",
            "expected_answer_contains": ["Rethinking Crack Segmentation"],
            "expected_sources": [{"filename": "ECCV_2026.pdf", "page": 1}],
            "expected_intent": "document_qa",
            "tags": ["front_matter", "citation"],
        }),
        EvaluationCase.from_dict({
            "id": "missing_metric",
            "question": "What is revenue?",
            "expected_answer_contains": ["$10M"],
            "expected_numbers": ["10"],
            "expected_sources": [{"filename": "q3.pdf", "page": 2}],
            "tags": ["number", "citation"],
        }),
        EvaluationCase.from_dict({
            "id": "no_answer",
            "question": "What is the CEO salary?",
            "expected_no_answer": True,
            "tags": ["no_answer"],
        }),
    ]
    predictions = {
        "title": Prediction.from_dict({
            "id": "title",
            "answer": "Rethinking Crack Segmentation is the title.",
            "sources": [{"filename": "ECCV_2026.pdf", "page": 1}],
            "retrieved_chunks": [{"filename": "ECCV_2026.pdf", "page": 1}],
            "intent": "document_qa",
            "latency_ms": 120,
        }),
        "missing_metric": Prediction.from_dict({
            "id": "missing_metric",
            "answer": "Revenue was not disclosed.",
            "sources": [{"filename": "wrong.pdf", "page": 9}],
            "retrieved_chunks": [{"filename": "q3.pdf", "page": 2}],
            "latency_ms": 200,
        }),
        "no_answer": Prediction.from_dict({
            "id": "no_answer",
            "answer": "Could not find a sufficiently relevant answer in the provided documents.",
            "latency_ms": 80,
        }),
    }

    report = build_interview_report(cases, predictions, ks=(1, 5))

    assert report["summary"]["total_cases"] == 3
    assert report["summary"]["missing_predictions"] == 0
    assert report["summary"]["retrieval_recall_at_k"] == {"1": 1.0, "5": 1.0}
    assert report["case_groups"]["no_answer"] == ["no_answer"]
    assert report["case_groups"]["citation"] == ["title", "missing_metric"]
    assert report["case_groups"]["calculation_or_number"] == ["missing_metric"]
    assert report["resume_metrics"][0] == {
        "name": "Golden answer pass rate",
        "value": "66.7%",
        "source": "offline JSONL eval",
    }
    assert any(item["id"] == "missing_metric" for item in report["weak_cases"])


def test_eval_cli_interview_report_writes_json(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    out_path = tmp_path / "interview_report.json"
    _write_jsonl(cases_path, [{
        "id": "c1",
        "question": "What was revenue?",
        "expected_answer_contains": ["$10M"],
        "expected_numbers": ["10"],
        "expected_sources": [{"filename": "q3.pdf", "page": 2}],
    }])
    _write_jsonl(predictions_path, [{
        "id": "c1",
        "answer": "Revenue was $10M.",
        "sources": [{"filename": "q3.pdf", "page": 2}],
        "retrieved_chunks": [{"filename": "q3.pdf", "page": 2}],
    }])

    code = eval_cli_main([
        "interview-report",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--k",
        "1",
        "--out",
        str(out_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["summary"]["answer_pass_rate"] == 1.0
    assert written["resume_metrics"][2]["name"] == "Retrieval Recall@1"
