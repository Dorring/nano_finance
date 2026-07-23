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
    """A citation or retrieval target expected for a case.

    At least one of ``filename``/``document_name``, ``page``, or
    ``chunk_id`` must be set. An empty source ``{}`` is rejected.
    """

    filename: str | None = None
    page: int | str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedSource":
        filename = (
            data.get("filename")
            or data.get("document_name")
            or data.get("doc_name")
        )
        page = data.get("page", data.get("page_number"))
        chunk_id = data.get("chunk_id") or data.get("doc_id")
        if not any([filename, page is not None, chunk_id]):
            raise ValueError(
                "ExpectedSource must have at least one of "
                "filename/document_name, page, or chunk_id"
            )
        return cls(filename=filename, page=page, chunk_id=chunk_id)

    def matches(self, candidate: dict[str, Any]) -> bool:
        """Return True when candidate satisfies all fields set on this source."""
        candidate_id = candidate.get("chunk_id") or candidate.get("doc_id")
        if self.chunk_id and candidate_id != self.chunk_id:
            return False
        if self.filename:
            cand_filename = (
                candidate.get("filename")
                or candidate.get("document_name")
                or candidate.get("doc_name")
            )
            if cand_filename != self.filename:
                return False
        if self.page is not None:
            cand_page = candidate.get("page")
            if str(cand_page) != str(self.page):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "page": self.page, "chunk_id": self.chunk_id}


_VALID_OPERATIONS = frozenset({
    "difference", "growth_rate", "percentage_share", "sum", "average",
    "gross_margin", "net_margin", "debt_ratio", "scale_conversion",
})

_VALID_UNITS = frozenset({
    "ratio", "percent", "percentage_point",
    "元", "万元", "百万元", "亿元",
    "million", "billion", "currency",
})

_VALID_FORMULA_VERSIONS = frozenset({"v1", "v2", "v3"})


@dataclass(frozen=True)
class ExpectedCalculation:
    """Expected deterministic financial calculation for a case.

    ``expected_value`` is mandatory — a missing value must raise, never
    silently become the string ``"None"``.
    """

    calc_id: str
    operation: str
    args: dict[str, Any]
    expected_value: str
    tolerance: str = "0"
    unit: str | None = None
    metric: str | None = None
    period: str | None = None
    currency: str | None = None
    scale: str | None = None
    formula_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedCalculation":
        calc_id = data.get("id") or data.get("calc_id") or data.get("operation")
        operation = data.get("operation")
        if not calc_id:
            raise ValueError("expected calculation missing id/calc_id")
        if not operation:
            raise ValueError(f"expected calculation {calc_id!r} missing operation")
        raw_value = data.get("expected_value")
        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(
                f"expected calculation {calc_id!r} missing expected_value "
                "(must not be None or empty)"
            )
        return cls(
            calc_id=str(calc_id),
            operation=str(operation),
            args=dict(data.get("args", {})),
            expected_value=str(raw_value),
            tolerance=str(data.get("tolerance", "0")),
            unit=data.get("unit"),
            metric=data.get("metric"),
            period=data.get("period"),
            currency=data.get("currency"),
            scale=data.get("scale"),
            formula_version=data.get("formula_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.calc_id,
            "operation": self.operation,
            "args": self.args,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "metric": self.metric,
            "period": self.period,
            "currency": self.currency,
            "scale": self.scale,
            "formula_version": self.formula_version,
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
    annotation_evidence: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationLabel":
        case_id = data.get("case_id") or data.get("id")
        if not case_id:
            raise ValueError("evaluation label missing case_id/id")
        annotation = data.get("annotation_evidence")
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
            annotation_evidence=dict(annotation) if isinstance(annotation, dict) else None,
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
            "annotation_evidence": self.annotation_evidence,
        }


@dataclass(frozen=True)
class EvaluationPrediction:
    """What the blind runner produces — captures all Phase 3/4 fields.

    Unlike the old Prediction, this object records calculations,
    answerability, validation, and warnings from the RAG engine result.

    Runtime observability fields (v2):
        context_token_count: Real token count from the context builder or
            tokenizer — never estimated via ``len(answer)/4``.
        llm_generation_call_count: Number of LLM generation calls.
        llm_rewrite_call_count: Number of LLM query-rewrite calls.
        system_error_category: Structured error category (``auth_*``,
            ``env_*``, ``model_*``, ``retrieval_*``, etc.) or ``None``.
        generation_mode: ``"llm"`` / ``"deterministic"`` / ``"hybrid"``.
        streaming_contract_checked: Whether streaming contract was
            verified (only via dedicated SSE test, not blind runner).
        validator_internal_failure_count: Validator internal failures.
        validator_internal_failure_blocked_count: Cases where a
            validator internal failure caused a safe block.
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
    context_token_count: int | None = None
    llm_generation_call_count: int = 0
    llm_rewrite_call_count: int = 0
    system_error_category: str | None = None
    generation_mode: str = "llm"
    streaming_contract_checked: bool = False
    validator_internal_failure_count: int = 0
    validator_internal_failure_blocked_count: int = 0

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
            context_token_count=data.get("context_token_count"),
            llm_generation_call_count=int(data.get("llm_generation_call_count", 0)),
            llm_rewrite_call_count=int(data.get("llm_rewrite_call_count", 0)),
            system_error_category=data.get("system_error_category"),
            generation_mode=str(data.get("generation_mode", "llm")),
            streaming_contract_checked=bool(data.get("streaming_contract_checked", False)),
            validator_internal_failure_count=int(
                data.get("validator_internal_failure_count", 0)
            ),
            validator_internal_failure_blocked_count=int(
                data.get("validator_internal_failure_blocked_count", 0)
            ),
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
            "context_token_count": self.context_token_count,
            "llm_generation_call_count": self.llm_generation_call_count,
            "llm_rewrite_call_count": self.llm_rewrite_call_count,
            "system_error_category": self.system_error_category,
            "generation_mode": self.generation_mode,
            "streaming_contract_checked": self.streaming_contract_checked,
            "validator_internal_failure_count": self.validator_internal_failure_count,
            "validator_internal_failure_blocked_count": self.validator_internal_failure_blocked_count,
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


# ---------------------------------------------------------------------------
# Evaluation Feature Flags (evaluation-only, never production default)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationFeatureFlags:
    """Evaluation-only feature flags for ablation control.

    All flags default to ``True`` so that the default behaviour matches
    production exactly. Only the evaluation composition root may inject
    a non-default flags object — production code must NEVER read these
    flags from user input or config files.

    Constraints enforced by the evaluation runner:
        - Disabling ``citation_validation_enabled`` must NOT disable
          other validators.
        - Disabling ``answerability_enabled`` must NOT disable
          ``post_validation_enabled``.
        - ``dense_enabled=False`` and ``bm25_enabled=False`` must produce
          two genuinely different retrieval modes (Dense-only vs BM25-only).
    """

    dense_enabled: bool = True
    bm25_enabled: bool = True
    reranker_enabled: bool = True
    query_rewrite_enabled: bool = True
    hierarchical_context_enabled: bool = True
    calculator_enabled: bool = True
    answerability_enabled: bool = True
    post_validation_enabled: bool = True
    citation_validation_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_enabled": self.dense_enabled,
            "bm25_enabled": self.bm25_enabled,
            "reranker_enabled": self.reranker_enabled,
            "query_rewrite_enabled": self.query_rewrite_enabled,
            "hierarchical_context_enabled": self.hierarchical_context_enabled,
            "calculator_enabled": self.calculator_enabled,
            "answerability_enabled": self.answerability_enabled,
            "post_validation_enabled": self.post_validation_enabled,
            "citation_validation_enabled": self.citation_validation_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationFeatureFlags":
        return cls(
            dense_enabled=bool(data.get("dense_enabled", True)),
            bm25_enabled=bool(data.get("bm25_enabled", True)),
            reranker_enabled=bool(data.get("reranker_enabled", True)),
            query_rewrite_enabled=bool(data.get("query_rewrite_enabled", True)),
            hierarchical_context_enabled=bool(
                data.get("hierarchical_context_enabled", True)
            ),
            calculator_enabled=bool(data.get("calculator_enabled", True)),
            answerability_enabled=bool(data.get("answerability_enabled", True)),
            post_validation_enabled=bool(data.get("post_validation_enabled", True)),
            citation_validation_enabled=bool(
                data.get("citation_validation_enabled", True)
            ),
        )


# ---------------------------------------------------------------------------
# Unified Case Scoring (single source of truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseCheck:
    """One atomic check within a case score."""

    name: str
    passed: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "passed": self.passed}
        if self.detail is not None:
            d["detail"] = self.detail
        return d


@dataclass(frozen=True)
class CaseScore:
    """Unified case-level score produced by the single canonical scorer.

    ``passed`` is True only when every applicable check passes.
    ``primary_failure`` is the first failing check name (or None).
    ``secondary_failures`` are the remaining failing check names.
    """

    case_id: str
    passed: bool
    checks: tuple[CaseCheck, ...]
    primary_failure: str | None
    secondary_failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "primary_failure": self.primary_failure,
            "secondary_failures": list(self.secondary_failures),
        }
