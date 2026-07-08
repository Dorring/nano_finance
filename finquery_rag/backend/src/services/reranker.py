"""Reranking interfaces for FinQuery retrieval.

The default production path keeps reranking disabled. This module provides a
small dependency-free interface so cross-encoder reranking can be added later
without changing the RAG pipeline shape.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol


class Reranker(Protocol):
    """Protocol implemented by all rerankers."""

    name: str

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        """Return chunks sorted by reranker relevance."""


@dataclass
class NoopReranker:
    """Preserve retrieval order. Useful as the default / disabled reranker."""

    name: str = "noop"

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        limit = top_k if top_k is not None else len(chunks)
        return list(chunks)[:limit]


@dataclass
class HeuristicReranker:
    """Dependency-free lexical reranker for deterministic tests and fallback.

    Score combines original retrieval score with query-token overlap. It is not
    a substitute for a cross-encoder, but gives the pipeline a stable reranker
    contract without model downloads or new dependencies.
    """

    original_score_weight: float = 0.7
    lexical_weight: float = 0.3
    name: str = "heuristic"

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        if not chunks:
            return []

        query_terms = _tokenize(query)
        scored = []
        for index, chunk in enumerate(chunks):
            original_score = _safe_float(chunk.get("score", 0.0))
            lexical_score = _lexical_overlap(query_terms, chunk.get("content", ""))
            rerank_score = (
                self.original_score_weight * original_score
                + self.lexical_weight * lexical_score
            )
            item = dict(chunk)
            item["rerank_score"] = rerank_score
            item["reranker"] = self.name
            scored.append((rerank_score, original_score, -index, item))

        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        ordered = [item for _, _, _, item in scored]
        if top_k is not None:
            return ordered[:top_k]
        return ordered


def build_reranker(name: str | None) -> Reranker | None:
    """Build a reranker from config name.

    `None`, empty, "none", and "noop" all mean disabled/no-op.
    """
    normalized = (name or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized == "noop":
        return NoopReranker()
    if normalized == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"Unknown reranker: {name}")


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text or "")
        if token.strip()
    }


def _lexical_overlap(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = _tokenize(content)
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    return overlap / math.sqrt(len(query_terms) * len(content_terms))


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
