"""Evidence item domain object."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    """Immutable representation of a single retrieved evidence chunk."""

    chunk_id: str
    content: str
    document_name: str | None
    page: int | None
    content_type: str | None
    score: float
    rerank_score: float | None
    metadata: dict[str, Any]

    @classmethod
    def from_chunk(cls, chunk: dict) -> "EvidenceItem":
        """Create an EvidenceItem from the legacy chunk dict format."""
        return cls(
            chunk_id=chunk.get("chunk_id", ""),
            content=chunk.get("content", ""),
            document_name=chunk.get("document_name") or chunk.get("doc_name"),
            page=chunk.get("page"),
            content_type=chunk.get("content_type"),
            score=chunk.get("score", 0.0),
            rerank_score=chunk.get("rerank_score"),
            metadata={
                k: v
                for k, v in chunk.items()
                if k
                not in {
                    "chunk_id",
                    "content",
                    "document_name",
                    "doc_name",
                    "page",
                    "content_type",
                    "score",
                    "rerank_score",
                }
            },
        )

    def to_chunk(self) -> dict:
        """Convert back to the legacy chunk dict format."""
        result = {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_name": self.document_name,
            "page": self.page,
            "content_type": self.content_type,
            "score": self.score,
            "rerank_score": self.rerank_score,
        }
        result.update(self.metadata)
        return result
