import json

from src.eval_cli import main as eval_cli_main
from src.services.evaluation import (
    EvaluationCase,
    Prediction,
    build_failure_analysis_markdown,
    build_interview_report,
    score_prediction,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_score_prediction_classifies_retrieval_and_includes_diagnostics():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "expected_answer_contains": ["$10M"],
        "expected_numbers": ["10"],
        "expected_sources": [{"filename": "report.pdf", "page": 2}],
        "expected_intent": "document_qa",
    })
    prediction = Prediction.from_dict({
        "id": "c1",
        "answer": "Revenue was not available.",
        "sources": [],
        "retrieved_chunks": [],
        "intent": "document_qa",
    })

    score = score_prediction(case, prediction)

    assert score["failure_category"] == "retrieval_miss"
    assert score["expected_answer_contains"] == ["$10M"]
    assert score["expected_numbers"] == ["10"]
    assert score["expected_sources"] == [{"filename": "report.pdf", "page": 2, "chunk_id": None}]
    assert "Revenue was not available" in score["actual_answer_preview"]


def test_number_score_accepts_commas_and_percent_suffixes():
    case = EvaluationCase.from_dict({
        "id": "num",
        "question": "Q",
        "expected_numbers": ["98000", "76.6"],
    })
    prediction = Prediction.from_dict({
        "id": "num",
        "answer": "The values were 98,000 and 76.6%.",
    })

    score = score_prediction(case, prediction)

    assert score["number_accuracy"] == 1.0


def test_interview_report_weak_cases_include_actual_and_expected_details():
    cases = [EvaluationCase.from_dict({
        "id": "weak",
        "question": "Q",
        "expected_answer_contains": ["target"],
        "expected_sources": [{"filename": "a.pdf", "page": 1}],
    })]
    predictions = {"weak": Prediction.from_dict({
        "id": "weak",
        "answer": "wrong answer",
        "sources": [{"filename": "a.pdf", "page": 1}],
        "retrieved_chunks": [{"filename": "a.pdf", "page": 1}],
    })}

    report = build_interview_report(cases, predictions)
    weak = report["weak_cases"][0]

    assert weak["failure_category"] == "answer_mismatch"
    assert weak["expected_answer_contains"] == ["target"]
    assert weak["actual_answer_preview"] == "wrong answer"
    assert weak["actual_sources"] == [{"filename": "a.pdf", "page": 1}]


def test_failure_analysis_markdown_and_cli(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    out = tmp_path / "failure.md"
    _write_jsonl(cases_path, [{
        "id": "c1",
        "question": "What was revenue?",
        "expected_answer_contains": ["$10M"],
        "expected_sources": [{"filename": "report.pdf", "page": 2}],
    }])
    _write_jsonl(predictions_path, [{
        "id": "c1",
        "answer": "No matching revenue was found.",
        "sources": [],
        "retrieved_chunks": [],
    }])

    code = eval_cli_main([
        "failure-analysis",
        "--cases", str(cases_path),
        "--predictions", str(predictions_path),
        "--out", str(out),
    ])

    assert code == 0
    content = out.read_text(encoding="utf-8")
    assert "# FinQuery eval failure analysis" in content
    assert "### c1" in content
    assert "retrieval_miss" in content
    assert "No matching revenue was found" in content

    markdown = build_failure_analysis_markdown(
        [EvaluationCase.from_dict({"id": "c2", "question": "Q", "expected_answer_contains": ["A"]})],
        {"c2": Prediction.from_dict({"id": "c2", "answer": "B"})},
    )
    assert "answer_mismatch" in markdown
