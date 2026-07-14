"""Phase 84 tests: eval runner input hardening."""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.eval_runner import run_case, run_jsonl_cases, validate_n_results
from services.evaluation import EvaluationCase
from src.eval_cli import main as eval_cli_main


class GuardedRAGEngine:
    def __init__(self):
        self.calls = []

    async def query(self, question, doc_names=None, user_id=None, n_results=5):
        self.calls.append({
            "question": question,
            "doc_names": doc_names,
            "user_id": user_id,
            "n_results": n_results,
        })
        return {"answer": "A", "sources": [], "retrieved_chunks": []}


def test_validate_n_results_enforces_eval_runner_bounds():
    assert validate_n_results("20") == 20
    with pytest.raises(ValueError, match="n_results must be >= 1"):
        validate_n_results(0)
    with pytest.raises(ValueError, match="n_results must be <= 20"):
        validate_n_results(21)
    with pytest.raises(ValueError, match="n_results must be an integer"):
        validate_n_results("bad")


def test_run_case_rejects_invalid_user_id_before_engine_call():
    engine = GuardedRAGEngine()
    case = EvaluationCase.from_dict({"id": "c1", "question": "Q"})

    with pytest.raises(ValueError, match="user_id must be >= 1"):
        asyncio.run(run_case(case, engine, user_id=0, n_results=1))

    assert engine.calls == []


def test_run_jsonl_cases_rejects_invalid_n_results_without_writing(tmp_path):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    cases.write_text(json.dumps({"id": "c1", "question": "Q"}) + "\n", encoding="utf-8")
    engine = GuardedRAGEngine()

    with pytest.raises(ValueError, match="n_results must be <= 20"):
        asyncio.run(run_jsonl_cases(str(cases), str(out), engine, user_id=1, n_results=99))

    assert engine.calls == []
    assert not out.exists()


def test_eval_cli_run_rejects_invalid_user_id_before_importing_app(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    cases.write_text(json.dumps({"id": "c1", "question": "Q"}) + "\n", encoding="utf-8")

    code = eval_cli_main([
        "run",
        "--cases",
        str(cases),
        "--out",
        str(out),
        "--user-id",
        "0",
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "user-id must be >= 1" in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_eval_cli_run_rejects_invalid_n_results_before_importing_app(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "predictions.jsonl"
    cases.write_text(json.dumps({"id": "c1", "question": "Q"}) + "\n", encoding="utf-8")

    code = eval_cli_main([
        "run",
        "--cases",
        str(cases),
        "--out",
        str(out),
        "--user-id",
        "1",
        "--n-results",
        "21",
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "n_results must be <= 20" in captured.err
    assert captured.out == ""
    assert not out.exists()
