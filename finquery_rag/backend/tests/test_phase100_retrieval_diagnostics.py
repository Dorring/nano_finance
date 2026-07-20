import json

from src.evaluation.evaluation import EvaluationCase, Prediction, diagnose_retrieval
from src.evaluation.eval_cli import main as eval_cli_main


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def test_diagnose_retrieval_reports_ranked_source_coverage():
    cases = [
        EvaluationCase.from_dict({
            "id": "c1",
            "question": "Q1",
            "expected_sources": [{"filename": "a.pdf", "page": 1}],
            "tags": ["smoke"],
        }),
        EvaluationCase.from_dict({
            "id": "c2",
            "question": "Q2",
            "expected_sources": [{"filename": "b.pdf", "page": 2}],
        }),
    ]
    predictions = {
        "c1": Prediction.from_dict({
            "id": "c1",
            "answer": "A",
            "retrieved_chunks": [
                {"filename": "other.pdf", "page": 9},
                {"filename": "a.pdf", "page": 1},
            ],
        }),
        "c2": Prediction.from_dict({
            "id": "c2",
            "answer": "B",
            "retrieved_chunks": [{"filename": "wrong.pdf", "page": 2}],
        }),
    }

    report = diagnose_retrieval(cases, predictions, ks=(1, 2, 5))

    assert report["summary"]["total_expected_sources"] == 2
    assert report["summary"]["recall_at_k"] == {"1": 0.0, "2": 0.5, "5": 0.5}
    assert report["summary"]["mrr"] == 0.25
    assert report["summary"]["full_recall_rate"] == 0.5
    assert report["cases"][0]["best_rank"] == 2
    assert report["cases"][1]["missed_expected_sources"] == [
        {"filename": "b.pdf", "page": 2, "chunk_id": None}
    ]
    assert [item["id"] for item in report["worst_cases"]] == ["c2", "c1"]


def test_diagnose_retrieval_handles_missing_predictions_and_no_source_cases():
    cases = [
        EvaluationCase.from_dict({
            "id": "missing",
            "question": "Q1",
            "expected_sources": [{"filename": "a.pdf"}],
        }),
        EvaluationCase.from_dict({"id": "no_source", "question": "Q2"}),
    ]

    report = diagnose_retrieval(cases, {}, ks=(3,))

    assert report["summary"]["missing_predictions"] == 2
    assert report["summary"]["cases_without_expected_sources"] == 1
    assert report["summary"]["recall_at_k"] == {"3": 0.0}
    assert report["missing_case_ids"] == ["missing", "no_source"]
    assert report["no_expected_source_case_ids"] == ["no_source"]


def test_eval_cli_retrieval_diagnostics_writes_report(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    out_path = tmp_path / "diag.json"
    _write_jsonl(cases_path, [{
        "id": "c1",
        "question": "Q1",
        "expected_sources": [{"filename": "a.pdf", "page": 1}],
    }])
    _write_jsonl(predictions_path, [{
        "id": "c1",
        "answer": "A",
        "retrieved_chunks": [{"filename": "a.pdf", "page": 1}],
    }])

    code = eval_cli_main([
        "retrieval-diagnostics",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--k",
        "1",
        "--k",
        "3",
        "--out",
        str(out_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["summary"]["recall_at_k"] == {"1": 1.0, "3": 1.0}
    assert written["cases"][0]["full_recall"] is True


def test_eval_cli_retrieval_diagnostics_rejects_bad_k(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(cases_path, [{"id": "c1", "question": "Q1"}])
    _write_jsonl(predictions_path, [])

    code = eval_cli_main([
        "retrieval-diagnostics",
        "--cases",
        str(cases_path),
        "--predictions",
        str(predictions_path),
        "--k",
        "0",
    ])

    captured = capsys.readouterr()
    assert code == 2
    assert "k values must be >= 1" in captured.err
