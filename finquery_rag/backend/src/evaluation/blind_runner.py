"""Phase 5 blind evaluation runner.

Runs the RAG engine against ``EvaluationQuery`` objects WITHOUT access to
any expected fields. The runner only sees the question, document names,
tags, and metadata. It captures the full Phase 3/4 result envelope
including calculations, answerability, validation, and warnings, plus
latency and error diagnostics.

Structural isolation is enforced: the only inputs are ``EvaluationQuery``
objects and a RAG engine handle. Expected fields are never imported or
loaded by this module. Cross-case isolation is enforced by passing empty
``conversation_history`` and ``None`` ``memory_profile`` on every call.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any, Iterable

from .dataset_loader import load_queries, write_jsonl
from .schemas import EvaluationPrediction, EvaluationQuery

_DEFAULT_TIMEOUT_SECONDS = 120.0


def clear_engine_caches(rag_engine: Any) -> None:
    """Clear retriever caches on the engine between cases.

    This is a no-op if the engine has no caches. It safely attempts to
    clear the ``_query_processor`` cache via ``getattr`` so that
    cross-case state does not leak.
    """
    query_processor = getattr(rag_engine, "_query_processor", None)
    if query_processor is None:
        return
    cache_clear = getattr(query_processor, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
        return
    cache = getattr(query_processor, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()


async def run_blind_query(
    query: EvaluationQuery,
    rag_engine: Any,
    *,
    user_id: int,
    n_results: int,
    session_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> EvaluationPrediction:
    """Run one ``EvaluationQuery`` through the RAG engine and capture the result.

    The engine is called with only the question, document names, user id,
    ``n_results``, and explicit empty ``conversation_history`` / ``None``
    ``memory_profile`` to enforce cross-case isolation. No expected fields
    are passed. The returned prediction captures all Phase 3/4 envelope
    fields (calculations, answerability, validation, warnings) when
    present, plus latency and error diagnostics.

    Both sync and async ``query()`` methods are supported via
    ``inspect.isawaitable``. Async calls are wrapped with
    ``asyncio.wait_for`` using the configured ``timeout``. On timeout the
    ``error_code`` is set to ``"QUERY_TIMEOUT"``. Engine exceptions are
    caught and recorded as ``error_code`` so a single failing query never
    crashes the whole run.
    """
    if session_id is None:
        session_id = f"eval_session_{query.case_id}"

    start_ns = time.perf_counter_ns()
    error_code: str | None = None
    result: dict[str, Any] = {}
    try:
        raw = rag_engine.query(
            question=query.question,
            doc_names=list(query.document_names) or None,
            user_id=user_id,
            n_results=n_results,
            conversation_history=[],
            memory_profile=None,
        )
        if inspect.isawaitable(raw):
            raw = await asyncio.wait_for(raw, timeout=timeout)
        if isinstance(raw, dict):
            result = raw
        else:
            result = {}
    except asyncio.TimeoutError:
        error_code = "QUERY_TIMEOUT"
    except Exception as exc:  # noqa: BLE001 - record error, do not crash.
        error_code = type(exc).__name__

    end_ns = time.perf_counter_ns()
    latency_ms = (end_ns - start_ns) / 1_000_000
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
    session_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> list[EvaluationPrediction]:
    """Run multiple ``EvaluationQuery`` objects through the RAG engine sequentially.

    If ``session_id`` is not provided, each case auto-generates a unique
    ``f"eval_session_{case_id}"``. Engine caches are cleared between
    cases via ``clear_engine_caches`` to prevent cross-case state leakage.
    """
    predictions: list[EvaluationPrediction] = []
    for query in queries:
        case_session = (
            session_id if session_id is not None else f"eval_session_{query.case_id}"
        )
        predictions.append(
            await run_blind_query(
                query,
                rag_engine,
                user_id=user_id,
                n_results=n_results,
                session_id=case_session,
                timeout=timeout,
            )
        )
        clear_engine_caches(rag_engine)
    return predictions


async def run_blind_jsonl(
    questions_path: str | Path,
    output_path: str | Path,
    rag_engine: Any,
    *,
    user_id: int,
    n_results: int,
    session_id: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
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
        session_id=session_id,
        timeout=timeout,
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
    Streaming contract is always ``False`` for the blind runner because
    it is non-streaming — contract verification must happen via a
    dedicated SSE test suite.
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
        calculations=tuple(dict(c) for c in result.get("calculations", []) or []),
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
        system_error_category=result.get("system_error_category"),
        generation_mode=result.get("generation_mode", "llm"),
        streaming_contract_checked=False,
        context_token_count=result.get("context_token_count"),
        llm_generation_call_count=result.get("llm_generation_call_count", 0),
        llm_rewrite_call_count=result.get("llm_rewrite_call_count", 0),
    )
