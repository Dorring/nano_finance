"""Oracle context builder for offline evaluation upper-bound measurement.

This module provides perfect-evidence answers for measuring the GENERATION upper
bound. It is NOT a production module and must NEVER be imported by:

- src/main.py (FastAPI routes)
- src/services/rag_engine.py (RAGEngine)
- src/services/retrieval.py (BM25 retriever)
- src/services/reranker.py (Reranker)
- src/services/vector_store.py (ChromaDB wrapper)
- Any production code path

Usage (offline eval CLI only):
    from src.evaluation.oracle_context import build_oracle_context
    context, sources = build_oracle_context(case)
    # Measure: given perfect evidence, does the model generate the right answer?

Metrics produced:
    - Oracle Context Answer Accuracy
    - Oracle Generation Upper Bound

WARNING: These must NOT be called:
    - RAG Accuracy
    - Retrieval Accuracy
    - Production Accuracy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OracleEvidence:
    """Evidence item provided by the oracle (from golden labels, not retrieval)."""
    document_name: str
    page: int | None
    content: str
    chunk_id: str


def build_oracle_context(case: dict[str, Any]) -> tuple[str, list[dict]]:
    """Build oracle context and source list from an evaluation case.

    Args:
        case: An evaluation case dict with expected_sources and expected_answer fields.

    Returns:
        (context_text, sources_list) where context_text is the concatenated
        perfect evidence and sources_list is structured source metadata.

    Raises:
        ValueError: If the case lacks expected_sources or cannot build oracle context.
    """
    expected_sources = case.get("expected_sources", [])
    if not expected_sources:
        raise ValueError("Case has no expected_sources; cannot build oracle context")

    context_parts: list[str] = []
    sources: list[dict] = []

    for idx, source in enumerate(expected_sources):
        if not isinstance(source, dict):
            continue
        filename = source.get("filename") or source.get("doc_name") or f"source_{idx}"
        page = source.get("page")
        content = source.get("content") or ""
        chunk_id = source.get("chunk_id") or source.get("doc_id") or f"oracle_{idx}"

        context_parts.append(
            f"[Oracle Evidence {idx + 1}: {filename}"
            + (f", page {page}" if page is not None else "")
            + f"]\n{content}"
        )
        sources.append({
            "filename": filename,
            "page": page,
            "chunk_id": chunk_id,
            "oracle": True,
        })

    return "\n\n".join(context_parts), sources
