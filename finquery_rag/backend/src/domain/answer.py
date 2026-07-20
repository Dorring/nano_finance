"""Answer result and retrieval result domain objects."""
from dataclasses import dataclass
from typing import Any

from src.domain.evidence import EvidenceItem


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable representation of a retrieval operation's output."""

    query: str
    retrieval_query: str
    evidence: tuple[EvidenceItem, ...]
    debug: dict[str, Any]


@dataclass(frozen=True)
class AnswerResult:
    """Immutable representation of a final answer."""

    answer: str
    sources: tuple[dict[str, Any], ...]
    confidence: float
    intent: str
    trace_id: str | None
    warnings: tuple[str, ...] = ()

    def to_legacy_dict(self) -> dict:
        """Convert to the legacy API response dict format."""
        result = {
            "answer": self.answer,
            "sources": list(self.sources),
            "confidence": self.confidence,
            "intent": self.intent,
        }
        if self.trace_id is not None:
            result["trace_id"] = self.trace_id
        if self.warnings:
            result["warnings"] = list(self.warnings)
        return result
