"""Tests enforcing label isolation between the blind runner and labels.

These tests verify, structurally and behaviorally, that the blind runner
cannot access expected fields: the ``EvaluationQuery`` schema carries no
``expected_*`` fields, the runner source never imports labels or opens
label files, and the RAG engine call only receives question/document/user
parameters.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any

from src.evaluation.blind_runner import run_blind_query
from src.evaluation.schemas import EvaluationQuery


def test_evaluation_query_has_no_expected_fields() -> None:
    """EvaluationQuery dataclass must not declare any expected_* fields."""
    field_names = {f.name for f in fields(EvaluationQuery)}
    expected_fields = {name for name in field_names if name.startswith("expected_")}
    assert expected_fields == set()


def test_blind_runner_does_not_import_labels() -> None:
    """blind_runner.py source must not import EvaluationLabel."""
    source = Path(inspect.getfile(run_blind_query)).read_text(encoding="utf-8")
    assert "EvaluationLabel" not in source


def test_blind_runner_does_not_open_label_files() -> None:
    """blind_runner.py source must not reference label paths at all."""
    source = Path(inspect.getfile(run_blind_query)).read_text(encoding="utf-8")
    assert "label" not in source.lower()


class _SpyEngine:
    """Captures the exact kwargs passed to query()."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def query(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = dict(kwargs)
        return {"answer": "ok"}


def test_rag_engine_call_has_no_expected_fields() -> None:
    """The engine query() call must only pass question/doc_names/user_id/
    n_results plus explicit cross-case isolation params (conversation_history
    and memory_profile). No expected_* fields may leak."""
    spy = _SpyEngine()
    query = EvaluationQuery.from_dict(
        {"case_id": "c1", "question": "Q", "document_names": ["d.pdf"]}
    )

    asyncio.run(run_blind_query(query, spy, user_id=1, n_results=2))

    assert spy.kwargs is not None
    assert set(spy.kwargs.keys()) == {
        "question",
        "doc_names",
        "user_id",
        "n_results",
        "conversation_history",
        "memory_profile",
    }
    assert spy.kwargs["question"] == "Q"
    assert spy.kwargs["doc_names"] == ["d.pdf"]
    assert spy.kwargs["user_id"] == 1
    assert spy.kwargs["n_results"] == 2
    assert spy.kwargs["conversation_history"] == []
    assert spy.kwargs["memory_profile"] is None
    # No expected_* fields may be passed to the engine.
    assert not any(k.startswith("expected_") for k in spy.kwargs)
