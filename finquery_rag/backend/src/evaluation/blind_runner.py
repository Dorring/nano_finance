"""Phase 5 blind evaluation runner.

Runs the RAG engine against ``EvaluationQuery`` objects WITHOUT access to
any expected fields. The runner only sees the question, document names,
tags, and metadata. It captures the full Phase 3/4 result envelope
including calculations, answerability, validation, and warnings, plus
latency and error diagnostics.

Structural isolation is enforced: the only inputs are ``EvaluationQuery``
objects and a RAG engine handle. Expected fields are never imported or
loaded by this module.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Iterable

from .dataset_loader import load_queries, write_jsonl
from .schemas import EvaluationPrediction, EvaluationQuery


async def run_blind_query(
    query: EvaluationQuery,
    rag_engine: Any,
    *,
    user_id: int,
    n_results: int,
) -> EvaluationPrediction:
    """Run one ``EvaluationQuery`` through the RAG engine and capture the result.

    The engine is called with only the question, document names, user id,
    and ``n_results``. No expected fields are passed. The returned
    prediction captures all Phase 3/4 envelope fields (calculations,
    answerability, validation, warnings) when present, plus latency and
    error diagnostics.

    Both sync and async ``query()`` methods are supported via
    ``inspect.isawaitable``. Engine exceptions are caught and recorded as
    ``error_code`` so a single failing query never crashes the whole run.
    """
    start = time.time()
    error_code: str | None = None
    result: dict[str, Any] = {}
    try:
        raw = rag_engine.query(
            question=query.question,
            doc_names=list(query.document_names) or None,
            user_id=user_id,
            n_results=n_results,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        if isinstance(raw, dict):
            result = raw
        else:
            result = {}
    except Exception as exc:  # noqa: BLE001 - record error, do not crash.
        error_code = type(exc).__name__

    latency_ms = (time.time() - start) * 1000
    return _prediction_from_result(
        query.case_id,
        result,
        latency_ms=latency_ms,
        error_code=error_code,
    )


async def run_blind_queries(
    queries: Iterable[EvaluationQuery],
    rag_engine: Any,
    *,
    user_id: int,
    n_results: int,
) -> list[EvaluationPrediction]:
    """Run multiple ``EvaluationQuery`` objects through the RAG engine sequentially."""
    predictions: list[EvaluationPrediction] = []
    for query in queries:
        predictions.append(
            await run_blind_query(
                query,
                rag_engine,
                user_id=user_id,
                n_results=n_results,
            )
        )
    return predictions


async def run_blind_jsonl(
    questions_path: str | Path,
    output_path: str | Path,
    rag_engine: Any,
    *,
    user_id: int,
    n_results: int,
) -> list[EvaluationPrediction]:
    """Load questions JSONL, run the RAG engine, and write predictions JSONL.

    The predictions file is written atomically: either the complete file
    appears or the previous file remains untouched.
    """
    queries = load_queries(questions_path)
    predictions = await run_blind_queries(
        queries,
        rag_engine,
        user_id=user_id,
        n_results=n_results,
    )
    write_jsonl(output_path, (p.to_dict() for p in predictions))
    return predictions


def _prediction_from_result(
    case_id: str,
    result: dict[str, Any],
    *,
    latency_ms: float,
    error_code: str | None,
) -> EvaluationPrediction:
    """Build an ``EvaluationPrediction`` from a RAG engine result dict.

    Extracts all Phase 3/4 envelope fields when present. Missing fields
    default to empty/None so the prediction is always well-formed.
    """
    answerability = result.get("answerability")
    validation = result.get("validation")
    return EvaluationPrediction(
        case_id=case_id,
        answer=str(result.get("answer", "")),
        sources=tuple(dict(s) for s in result.get("sources", []) or []),
        retrieved_chunks=tuple(
            dict(c) for c in result.get("retrieved_chunks", []) or []
        ),
        calculations=tuple(
            dict(c) for c in result.get("calculations", []) or []
        ),
        answerability=dict(answerability) if isinstance(answerability, dict) else None,
        validation=dict(validation) if isinstance(validation, dict) else None,
        warnings=tuple(str(w) for w in result.get("warnings", []) or []),
        intent=result.get("intent"),
        intent_confidence=result.get("intent_confidence"),
        context_sufficient=result.get("context_sufficient"),
        retrieval_debug=dict(result.get("retrieval_debug", {}) or {}),
        trace_id=result.get("trace_id"),
        latency_ms=latency_ms,
        error_code=error_code,
    )
