"""Phase 5 evaluation domain objects.

Separates evaluation data into three distinct objects to enforce label isolation:

- EvaluationQuery: what the blind runner sees (no expected_* fields)
- EvaluationLabel: what the scorer sees (all expected_* fields)
- EvaluationPrediction: what the blind runner produces

The old EvaluationCase (which combined question + expected fields) remains
available for backward compatibility but must NOT be used by the sealed runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExpectedSource:
    """A citation or retrieval target expected for a case."""

    filename: str | None = None
    page: int | str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedSource":
        return cls(
            filename=data.get("filename") or data.get("doc_name"),
            page=data.get("page"),
            chunk_id=data.get("chunk_id") or data.get("doc_id"),
        )

    def matches(self, candidate: dict[str, Any]) -> bool:
        """Return True when candidate satisfies all fields set on this source."""
        candidate_id = candidate.get("chunk_id") or candidate.get("doc_id")
        if self.chunk_id and candidate_id != self.chunk_id:
            return False
        if self.filename:
            cand_filename = candidate.get("filename") or candidate.get("doc_name")
            if cand_filename != self.filename:
                return False
        if self.page is not None:
            cand_page = candidate.get("page")
            if str(cand_page) != str(self.page):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "page": self.page, "chunk_id": self.chunk_id}


@dataclass(frozen=True)
class ExpectedCalculation:
    """Expected deterministic financial calculation for a case."""

    calc_id: str
    operation: str
    args: dict[str, Any]
    expected_value: str
    tolerance: str = "0"
    unit: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedCalculation":
        calc_id = data.get("id") or data.get("calc_id") or data.get("operation")
        operation = data.get("operation")
        if not calc_id:
            raise ValueError("expected calculation missing id/calc_id")
        if not operation:
            raise ValueError(f"expected calculation {calc_id!r} missing operation")
        return cls(
            calc_id=str(calc_id),
            operation=str(operation),
            args=dict(data.get("args", {})),
            expected_value=str(data.get("expected_value")),
            tolerance=str(data.get("tolerance", "0")),
            unit=data.get("unit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.calc_id,
            "operation": self.operation,
            "args": self.args,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class EvaluationQuery:
    """What the blind runner sees — NO expected_* fields allowed.

    This object is passed to the RAG engine. It must never contain
    expected sources, expected numbers, expected calculations, or any
    other label information.
    """

    case_id: str
    question: str
    document_names: tuple[str, ...]
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationQuery":
        case_id = data.get("case_id") or data.get("id")
        question = data.get("question")
        if not case_id:
            raise ValueError("evaluation query missing case_id/id")
        if not question:
            raise ValueError(f"evaluation query {case_id!r} missing question")
        # Reject any expected_* fields to enforce isolation
        for key in data:
            if key.startswith("expected_"):
                raise ValueError(
                    f"evaluation query {case_id!r} must not contain label field {key!r}"
                )
        return cls(
            case_id=str(case_id),
            question=str(question),
            document_names=tuple(str(d) for d in data.get("document_names", [])),
            tags=tuple(str(t) for t in data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "document_names": list(self.document_names),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationLabel:
    """What the scorer sees — all expected_* fields.

    This object is NEVER passed to the RAG engine. It is only loaded
    by the sealed scorer after predictions are generated and sealed.
    """

    case_id: str
    expected_sources: tuple[ExpectedSource, ...]
    expected_numbers: tuple[str, ...]
    expected_calculations: tuple[ExpectedCalculation, ...]
    expected_intent: str | None
    expected_answerability: str | None
    expected_validation_status: str | None
    expected_no_answer: bool
    required_answer_terms: tuple[str, ...]
    forbidden_answer_terms: tuple[str, ...]
    slice_tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationLabel":
        case_id = data.get("case_id") or data.get("id")
        if not case_id:
            raise ValueError("evaluation label missing case_id/id")
        return cls(
            case_id=str(case_id),
            expected_sources=tuple(
                ExpectedSource.from_dict(s) for s in data.get("expected_sources", [])
            ),
            expected_numbers=tuple(
                str(n) for n in data.get("expected_numbers", [])
            ),
            expected_calculations=tuple(
                ExpectedCalculation.from_dict(c)
                for c in data.get("expected_calculations", [])
            ),
            expected_intent=data.get("expected_intent"),
            expected_answerability=data.get("expected_answerability"),
            expected_validation_status=data.get("expected_validation_status"),
            expected_no_answer=bool(data.get("expected_no_answer", False)),
            required_answer_terms=tuple(
                str(t) for t in data.get("required_answer_terms", [])
            ),
            forbidden_answer_terms=tuple(
                str(t) for t in data.get("forbidden_answer_terms", [])
            ),
            slice_tags=tuple(str(t) for t in data.get("slice_tags", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_sources": [s.to_dict() for s in self.expected_sources],
            "expected_numbers": list(self.expected_numbers),
            "expected_calculations": [c.to_dict() for c in self.expected_calculations],
            "expected_intent": self.expected_intent,
            "expected_answerability": self.expected_answerability,
            "expected_validation_status": self.expected_validation_status,
            "expected_no_answer": self.expected_no_answer,
            "required_answer_terms": list(self.required_answer_terms),
            "forbidden_answer_terms": list(self.forbidden_answer_terms),
            "slice_tags": list(self.slice_tags),
        }


@dataclass(frozen=True)
class EvaluationPrediction:
    """What the blind runner produces — captures all Phase 3/4 fields.

    Unlike the old Prediction, this object records calculations,
    answerability, validation, and warnings from the RAG engine result.
    """

    case_id: str
    answer: str
    sources: tuple[dict[str, Any], ...]
    retrieved_chunks: tuple[dict[str, Any], ...]
    calculations: tuple[dict[str, Any], ...]
    answerability: dict[str, Any] | None
    validation: dict[str, Any] | None
    warnings: tuple[str, ...]
    intent: str | None
    intent_confidence: float | None
    context_sufficient: bool | None
    retrieval_debug: dict[str, Any]
    trace_id: str | None
    latency_ms: float
    error_code: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationPrediction":
        case_id = data.get("case_id") or data.get("id")
        if not case_id:
            raise ValueError("evaluation prediction missing case_id/id")
        answerability = data.get("answerability")
        validation = data.get("validation")
        return cls(
            case_id=str(case_id),
            answer=str(data.get("answer", "")),
            sources=tuple(dict(s) for s in data.get("sources", [])),
            retrieved_chunks=tuple(dict(c) for c in data.get("retrieved_chunks", [])),
            calculations=tuple(dict(c) for c in data.get("calculations", [])),
            answerability=dict(answerability) if isinstance(answerability, dict) else None,
            validation=dict(validation) if isinstance(validation, dict) else None,
            warnings=tuple(str(w) for w in data.get("warnings", [])),
            intent=data.get("intent"),
            intent_confidence=data.get("intent_confidence"),
            context_sufficient=data.get("context_sufficient"),
            retrieval_debug=dict(data.get("retrieval_debug", {})),
            trace_id=data.get("trace_id"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            error_code=data.get("error_code"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "answer": self.answer,
            "sources": list(self.sources),
            "retrieved_chunks": list(self.retrieved_chunks),
            "calculations": list(self.calculations),
            "answerability": self.answerability,
            "validation": self.validation,
            "warnings": list(self.warnings),
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "context_sufficient": self.context_sufficient,
            "retrieval_debug": self.retrieval_debug,
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
        }


# ---------------------------------------------------------------------------
# Dataset partition identifiers
# ---------------------------------------------------------------------------

VALID_PARTITIONS = ("dev", "calibration", "sealed")


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest describing a dataset partition."""

    partition: str
    case_count: int
    questions_sha256: str
    labels_sha256: str | None  # None for sealed public manifest
    created_at: str
    slices: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        partition = data.get("partition")
        if partition not in VALID_PARTITIONS:
            raise ValueError(
                f"partition must be one of {VALID_PARTITIONS}, got {partition!r}"
            )
        return cls(
            partition=partition,
            case_count=int(data.get("case_count", 0)),
            questions_sha256=str(data.get("questions_sha256", "")),
            labels_sha256=data.get("labels_sha256"),
            created_at=str(data.get("created_at", "")),
            slices=tuple(str(s) for s in data.get("slices", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "case_count": self.case_count,
            "questions_sha256": self.questions_sha256,
            "labels_sha256": self.labels_sha256,
            "created_at": self.created_at,
            "slices": list(self.slices),
        }
