"""Phase 5B tests: evaluation runner and baseline/candidate comparison."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.eval_runner import run_case, run_jsonl_cases
from services.evaluation import EvaluationCase, compare_reports


class FakeRAGEngine:
    def __init__(self):
        self.calls = []

    async def query(self, question, doc_names=None, user_id=None, n_results=5):
        self.calls.append({
            "question": question,
            "doc_names": doc_names,
            "user_id": user_id,
            "n_results": n_results,
        })
        return {
            "answer": "Revenue was $10M.",
            "sources": [{"filename": "q3.pdf", "page": 2}],
            "searched_docs": doc_names or ["q3.pdf"],
            "confidence": 0.9,
            "context_sufficient": True,
            "intent": "document_qa",
            "intent_confidence": 0.82,
            "retrieved_chunks": [{"doc_id": "q3.pdf::1", "filename": "q3.pdf", "page": 2}],
            "retrieval_debug": {"reranker": "heuristic", "candidate_count": 4, "returned_count": 1},
        }


def test_run_case_calls_rag_engine_with_case_filters():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "document_names": ["q3.pdf"],
    })
    engine = FakeRAGEngine()

    prediction = asyncio.run(run_case(case, engine, user_id=7, n_results=3))

    assert engine.calls == [{
        "question": "What was revenue?",
        "doc_names": ["q3.pdf"],
        "user_id": 7,
        "n_results": 3,
    }]
    assert prediction["id"] == "c1"
    assert prediction["answer"] == "Revenue was $10M."
    assert prediction["sources"][0]["filename"] == "q3.pdf"
    assert prediction["retrieved_chunks"][0]["doc_id"] == "q3.pdf::1"
    assert prediction["retrieval_debug"]["reranker"] == "heuristic"
    assert prediction["intent"] == "document_qa"
    assert prediction["intent_confidence"] == 0.82
    assert prediction["latency_ms"] >= 0


def test_run_jsonl_cases_writes_predictions(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "preds.jsonl"
    cases_path.write_text(
        json.dumps({"id": "c1", "question": "Q", "document_names": ["d.pdf"]}) + "\n",
        encoding="utf-8",
    )

    predictions = asyncio.run(
        run_jsonl_cases(
            str(cases_path),
            str(out_path),
            FakeRAGEngine(),
            user_id=1,
            n_results=2,
        )
    )

    assert len(predictions) == 1
    row = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert row["id"] == "c1"
    assert row["searched_docs"] == ["d.pdf"]


def test_compare_reports_passes_when_candidate_improves():
    baseline = {
        "summary": {"pass_rate": 0.5, "citation_recall": 0.5, "intent_accuracy": 0.5, "p95_latency_ms": 100},
        "cases": [{"id": "c1", "passed": False}],
    }
    candidate = {
        "summary": {"pass_rate": 1.0, "citation_recall": 1.0, "intent_accuracy": 1.0, "p95_latency_ms": 120},
        "cases": [{"id": "c1", "passed": True}],
    }

    comparison = compare_reports(baseline, candidate)

    assert comparison["passed"] is True
    assert comparison["metric_deltas"]["pass_rate"]["delta"] == 0.5
    assert comparison["newly_passed"] == ["c1"]
    assert comparison["p95_latency_delta_ms"] == 20


def test_compare_reports_fails_on_regression():
    baseline = {
        "summary": {"pass_rate": 1.0, "citation_recall": 1.0},
        "cases": [{"id": "c1", "passed": True}],
    }
    candidate = {
        "summary": {"pass_rate": 0.0, "citation_recall": 0.0},
        "cases": [{"id": "c1", "passed": False}],
    }

    comparison = compare_reports(baseline, candidate)

    assert comparison["passed"] is False
    assert "pass_rate" in comparison["regressions"]
    assert comparison["newly_failed"] == ["c1"]


def test_compare_reports_respects_tolerance():
    baseline = {"summary": {"pass_rate": 0.91}, "cases": []}
    candidate = {"summary": {"pass_rate": 0.90}, "cases": []}

    comparison = compare_reports(baseline, candidate, regression_tolerance=0.02)

    assert comparison["passed"] is True
    assert comparison["regressions"] == []



def test_compare_reports_includes_intent_accuracy_regression():
    baseline = {"summary": {"intent_accuracy": 1.0}, "cases": []}
    candidate = {"summary": {"intent_accuracy": 0.0}, "cases": []}

    comparison = compare_reports(baseline, candidate)

    assert comparison["passed"] is False
    assert "intent_accuracy" in comparison["regressions"]
