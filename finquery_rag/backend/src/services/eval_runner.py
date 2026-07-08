"""Run FinQuery RAG against evaluation cases and persist predictions."""
from __future__ import annotations

import inspect
import time
from typing import Any, Iterable

from .evaluation import EvaluationCase, load_jsonl_cases, write_jsonl


async def run_case(
    case: EvaluationCase,
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> dict[str, Any]:
    """Run one EvaluationCase through a RAGEngine-compatible object."""
    start = time.time()
    result = rag_engine.query(
        question=case.question,
        doc_names=list(case.document_names) or None,
        user_id=user_id,
        n_results=n_results,
    )
    if inspect.isawaitable(result):
        result = await result
    latency_ms = (time.time() - start) * 1000

    sources = result.get("sources", []) or []
    prediction = {
        "id": case.case_id,
        "question": case.question,
        "answer": result.get("answer", ""),
        "sources": sources,
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "searched_docs": result.get("searched_docs", []),
        "confidence": result.get("confidence"),
        "context_sufficient": result.get("context_sufficient"),
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "retrieval_debug": result.get("retrieval_debug", {}),
        "latency_ms": latency_ms,
    }
    if "rewritten_question" in result:
        prediction["rewritten_question"] = result.get("rewritten_question")
    return prediction


async def run_cases(
    cases: Iterable[EvaluationCase],
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    predictions = []
    for case in cases:
        predictions.append(
            await run_case(
                case=case,
                rag_engine=rag_engine,
                user_id=user_id,
                n_results=n_results,
            )
        )
    return predictions


async def run_jsonl_cases(
    cases_path: str,
    output_path: str,
    rag_engine: Any,
    user_id: int,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Load JSONL cases, run the RAG engine, and write predictions JSONL."""
    cases = load_jsonl_cases(cases_path)
    predictions = await run_cases(cases, rag_engine, user_id=user_id, n_results=n_results)
    write_jsonl(output_path, predictions)
    return predictions
