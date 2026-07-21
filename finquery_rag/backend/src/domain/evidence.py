"""Evidence item domain object."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    """Immutable representation of a single retrieved evidence chunk.

    ``from_chunk`` tolerates the multiple shapes the legacy chunk dict can
    take: top-level fields, fields nested under ``metadata``, or both.
    ``doc_id`` is accepted as a fallback for ``chunk_id`` because the
    pre-refactor pipeline used ``doc_id`` as the primary identifier.
    """

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
        """Create an EvidenceItem from the legacy chunk dict format.

        Reads top-level fields first, then falls back to nested ``metadata``.
        ``page=0`` is preserved (truthy checks would silently drop it).
        ``chunk_id`` falls back to ``doc_id`` because the retrieval pipeline
        populates ``doc_id`` rather than ``chunk_id``.
        """
        metadata = dict(chunk.get("metadata") or {})

        document_name = (
            chunk.get("document_name")
            or chunk.get("doc_name")
            or metadata.get("document_name")
            or metadata.get("doc_name")
            or metadata.get("filename")
        )

        # ``page`` may legitimately be 0; use ``is not None`` instead of truthy.
        page = chunk.get("page")
        if page is None:
            page = metadata.get("page")

        content_type = (
            chunk.get("content_type")
            or chunk.get("type")
            or metadata.get("content_type")
            or metadata.get("type")
        )

        chunk_id = chunk.get("chunk_id") or chunk.get("doc_id") or ""

        consumed_top_level = {
            "chunk_id",
            "doc_id",
            "content",
            "document_name",
            "doc_name",
            "page",
            "content_type",
            "type",
            "score",
            "rerank_score",
        }
        extra: dict[str, Any] = {}
        for key, value in chunk.items():
            if key in consumed_top_level:
                continue
            extra[key] = value
        for key, value in metadata.items():
            if key in extra:
                continue
            extra[key] = value

        return cls(
            chunk_id=chunk_id,
            content=chunk.get("content", "") or "",
            document_name=document_name,
            page=page,
            content_type=content_type,
            score=chunk.get("score", 0.0) or 0.0,
            rerank_score=chunk.get("rerank_score"),
            metadata=extra,
        )

    def to_chunk(self) -> dict:
        """Convert back to the legacy chunk dict format.

        Round-trip invariant: ``EvidenceItem.from_chunk(c).to_chunk()``
        preserves all original fields (top-level extras and ``metadata``).
        """
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.chunk_id,
            "content": self.content,
            "document_name": self.document_name,
            "page": self.page,
            "content_type": self.content_type,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": dict(self.metadata),
        }
        # Mirror nested metadata back to the top level so legacy consumers
        # that read e.g. ``chunk["type"]`` or ``chunk["doc_name"]`` still work.
        for key in ("type", "doc_name", "filename", "page"):
            if key in self.metadata and key not in result:
                result[key] = self.metadata[key]
        return result
