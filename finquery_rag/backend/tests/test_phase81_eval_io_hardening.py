"""Phase 81 tests: evaluation JSON/JSONL IO hardening."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import load_jsonl_cases, write_json_file, write_jsonl
from src.eval_cli import main as eval_cli_main


def test_write_jsonl_rejects_non_object_without_clobbering_existing_file(tmp_path):
    out = tmp_path / "predictions.jsonl"
    out.write_text('{"id":"existing"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSONL row 2 must be an object"):
        write_jsonl(out, [{"id": "ok"}, ["bad"]])

    assert out.read_text(encoding="utf-8") == '{"id":"existing"}\n'
    assert list(tmp_path.glob(".predictions.jsonl.*.tmp")) == []


def test_write_jsonl_creates_parent_and_loads_back(tmp_path):
    out = tmp_path / "nested" / "cases.jsonl"

    write_jsonl(out, [{"id": "c1", "question": "Q"}])

    cases = load_jsonl_cases(out)
    assert cases[0].case_id == "c1"
    assert out.read_text(encoding="utf-8").endswith("\n")


def test_write_json_file_rejects_non_object_payload(tmp_path):
    out = tmp_path / "report.json"

    with pytest.raises(ValueError, match="JSON payload must be an object"):
        write_json_file(out, ["not", "an", "object"])

    assert not out.exists()


def test_eval_cli_score_writes_complete_json_report(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    report = tmp_path / "report.json"
    cases.write_text(json.dumps({"id": "c1", "question": "Q", "expected_answer_contains": ["A"]}) + "\n", encoding="utf-8")
    predictions.write_text(json.dumps({"id": "c1", "answer": "A"}) + "\n", encoding="utf-8")

    code = eval_cli_main([
        "score",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--out",
        str(report),
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(report.read_text(encoding="utf-8"))["summary"]["pass_rate"] == 1.0
    assert json.loads(captured.out)["summary"]["pass_rate"] == 1.0
    assert list(tmp_path.glob(".report.json.*.tmp")) == []
