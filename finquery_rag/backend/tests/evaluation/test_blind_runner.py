"""Tests for the Phase 5 blind runner.

Verifies that the runner extracts all Phase 3/4 envelope fields, passes
only the allowed parameters to the RAG engine, measures latency, records
engine errors without crashing, and never loads labels.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

from src.evaluation.blind_runner import (
    run_blind_query,
    run_blind_queries,
)
from src.evaluation.schemas import EvaluationQuery


class FakeEngine:
    """Records calls and returns a configurable result envelope."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result or {}
        self._raises = raises

    async def query(self, question, doc_names=None, user_id=None, n_results=5):
        self.calls.append(
            {
                "question": question,
                "doc_names": doc_names,
                "user_id": user_id,
                "n_results": n_results,
            }
        )
        if self._raises is not None:
            raise self._raises
        return dict(self._result)


def _make_query(**overrides: Any) -> EvaluationQuery:
    data: dict[str, Any] = {"case_id": "c1", "question": "What is revenue?"}
    data.update(overrides)
    return EvaluationQuery.from_dict(data)


def test_blind_query_extracts_phase3_4_fields() -> None:
    """calculations/answerability/validation/warnings must be extracted."""
    engine = FakeEngine(
        result={
            "answer": "Revenue is 1000.",
            "calculations": [{"operation": "sum", "value": "1000"}],
            "answerability": {"status": "answerable"},
            "validation": {"status": "passed"},
            "warnings": ["minor issue"],
        }
    )
    pred = asyncio.run(run_blind_query(_make_query(), engine, user_id=1, n_results=3))

    assert len(pred.calculations) == 1
    assert pred.calculations[0]["operation"] == "sum"
    assert pred.answerability == {"status": "answerable"}
    assert pred.validation == {"status": "passed"}
    assert pred.warnings == ("minor issue",)


def test_blind_query_passes_correct_params() -> None:
    """query() must be called with only question/doc_names/user_id/n_results."""
    engine = FakeEngine()
    asyncio.run(
        run_blind_query(
            _make_query(document_names=["a.pdf", "b.pdf"]),
            engine,
            user_id=7,
            n_results=4,
        )
    )

    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert set(call.keys()) == {"question", "doc_names", "user_id", "n_results"}
    assert call["question"] == "What is revenue?"
    assert call["doc_names"] == ["a.pdf", "b.pdf"]
    assert call["user_id"] == 7
    assert call["n_results"] == 4


def test_blind_query_measures_latency() -> None:
    """latency_ms must be a non-negative number."""
    engine = FakeEngine(result={"answer": "ok"})
    pred = asyncio.run(run_blind_query(_make_query(), engine, user_id=1, n_results=2))

    assert isinstance(pred.latency_ms, float)
    assert pred.latency_ms >= 0


def test_blind_query_handles_engine_error() -> None:
    """Engine exceptions must be caught and recorded as error_code."""
    engine = FakeEngine(raises=RuntimeError("boom"))
    pred = asyncio.run(run_blind_query(_make_query(), engine, user_id=1, n_results=2))

    assert pred.error_code == "RuntimeError"
    assert pred.answer == ""


def test_blind_query_empty_doc_names_becomes_none() -> None:
    """Empty document_names must be passed to the engine as None."""
    engine = FakeEngine()
    asyncio.run(run_blind_query(_make_query(), engine, user_id=1, n_results=2))

    assert engine.calls[0]["doc_names"] is None


def test_blind_queries_runs_multiple() -> None:
    """run_blind_queries must run every query in order."""
    engine = FakeEngine(result={"answer": "ok"})
    queries = [_make_query(case_id=f"c{i}") for i in range(3)]

    preds = asyncio.run(
        run_blind_queries(queries, engine, user_id=1, n_results=2)
    )

    assert len(preds) == 3
    assert [p.case_id for p in preds] == ["c0", "c1", "c2"]
    assert len(engine.calls) == 3


def test_blind_runner_never_loads_labels() -> None:
    """The runner module source must not import labels or label loaders."""
    source = Path(inspect.getfile(run_blind_query)).read_text(encoding="utf-8")

    assert "EvaluationLabel" not in source
    assert "load_labels" not in source
