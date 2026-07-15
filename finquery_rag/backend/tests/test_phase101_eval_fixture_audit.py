import json

from src.eval_cli import main as eval_cli_main
from src.services.evaluation import EvaluationCase, audit_evaluation_fixtures


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def test_audit_evaluation_fixtures_reports_coverage_and_tags():
    cases = [
        EvaluationCase.from_dict({
            "id": "c1",
            "question": "Q1",
            "expected_sources": [{"filename": "a.pdf"}],
            "expected_intent": "document_qa",
            "tags": ["smoke", "citation"],
        }),
        EvaluationCase.from_dict({
            "id": "c2",
            "question": "Q2",
            "expected_no_answer": True,
            "expected_intent": "document_qa",
            "tags": ["smoke", "no_answer"],
        }),
    ]

    report = audit_evaluation_fixtures(cases, min_cases=2, required_tags=["citation", "no_answer"])

    assert report["passed"] is True
    assert report["summary"]["tag_counts"] == {"citation": 1, "no_answer": 1, "smoke": 2}
    assert report["summary"]["intent_counts"] == {"document_qa": 2}
    assert report["summary"]["coverage_counts"]["expected_sources"] == 1
    assert report["summary"]["coverage_rates"]["expected_no_answer"] == 0.5


def test_audit_evaluation_fixtures_fails_required_policy():
    cases = [EvaluationCase.from_dict({"id": "c1", "question": "Q1", "tags": ["smoke"]})]

    report = audit_evaluation_fixtures(
        cases,
        min_cases=2,
        required_tags=["citation"],
        require_expected_source=True,
        require_expected_intent=True,
    )

    assert report["passed"] is False
    assert {issue["code"] for issue in report["errors"]} == {
        "missing_expected_source",
        "missing_expected_intent",
        "min_cases_not_met",
        "missing_required_tag",
    }
    assert report["warnings"][0]["code"] == "missing_expected_signal"


def test_eval_cli_audit_fixtures_writes_report(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "audit.json"
    _write_jsonl(cases_path, [{
        "id": "c1",
        "question": "Q1",
        "expected_intent": "document_qa",
        "expected_answer_contains": ["revenue"],
        "tags": ["smoke"],
    }])

    code = eval_cli_main([
        "audit-fixtures",
        "--cases",
        str(cases_path),
        "--required-tag",
        "smoke",
        "--require-expected-intent",
        "--out",
        str(out_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert captured.err == ""
    assert payload["passed"] is True
    assert written["summary"]["total_cases"] == 1


def test_eval_cli_audit_fixtures_returns_one_on_policy_failure(tmp_path, capsys):
    cases_path = tmp_path / "cases.jsonl"
    _write_jsonl(cases_path, [{"id": "c1", "question": "Q1"}])

    code = eval_cli_main([
        "audit-fixtures",
        "--cases",
        str(cases_path),
        "--min-cases",
        "2",
        "--required-tag",
        "citation",
        "--require-expected-source",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["passed"] is False
    assert "FinQuery fixture audit failed:" in captured.err
    assert "minimum is 2" in captured.err
