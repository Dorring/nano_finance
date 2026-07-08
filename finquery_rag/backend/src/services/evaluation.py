"""Offline evaluation and trace replay helpers for FinQuery RAG.

The module intentionally avoids network, model, and vector-store dependencies.
It scores saved predictions and can convert trace records into replay cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
from statistics import mean
from typing import Any, Iterable


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


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
            if not cand_filename and candidate_id:
                cand_filename = _filename_from_doc_id(str(candidate_id))
            if cand_filename != self.filename:
                return False

        if self.page is not None:
            cand_page = candidate.get("page")
            if str(cand_page) != str(self.page):
                return False

        return True


@dataclass(frozen=True)
class EvaluationCase:
    """A golden/replay RAG evaluation case."""

    case_id: str
    question: str
    expected_sources: tuple[ExpectedSource, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_numbers: tuple[str, ...] = ()
    expected_no_answer: bool = False
    document_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        case_id = data.get("id") or data.get("case_id") or data.get("trace_id")
        question = data.get("question")
        if not case_id:
            raise ValueError("evaluation case missing id/case_id")
        if not question:
            raise ValueError(f"evaluation case {case_id!r} missing question")

        return cls(
            case_id=str(case_id),
            question=str(question),
            expected_sources=tuple(
                ExpectedSource.from_dict(item)
                for item in data.get("expected_sources", [])
            ),
            expected_answer_contains=tuple(
                str(item) for item in data.get("expected_answer_contains", [])
            ),
            expected_numbers=tuple(
                _normalize_number(str(item)) for item in data.get("expected_numbers", [])
            ),
            expected_no_answer=bool(data.get("expected_no_answer", False)),
            document_names=tuple(str(item) for item in data.get("document_names", [])),
            tags=tuple(str(item) for item in data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "question": self.question,
            "document_names": list(self.document_names),
            "expected_sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "chunk_id": s.chunk_id,
                }
                for s in self.expected_sources
            ],
            "expected_answer_contains": list(self.expected_answer_contains),
            "expected_numbers": list(self.expected_numbers),
            "expected_no_answer": self.expected_no_answer,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Prediction:
    """A saved RAG output for an EvaluationCase."""

    case_id: str
    answer: str
    sources: tuple[dict[str, Any], ...] = ()
    retrieved_chunks: tuple[dict[str, Any], ...] = ()
    latency_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        case_id = data.get("id") or data.get("case_id")
        if not case_id:
            raise ValueError("prediction missing id/case_id")
        return cls(
            case_id=str(case_id),
            answer=str(data.get("answer", "")),
            sources=tuple(dict(item) for item in data.get("sources", [])),
            retrieved_chunks=tuple(dict(item) for item in data.get("retrieved_chunks", [])),
            latency_ms=_optional_float(data.get("latency_ms")),
        )


def load_jsonl_cases(path: str | Path) -> list[EvaluationCase]:
    return [EvaluationCase.from_dict(item) for item in _read_jsonl(path)]


def load_jsonl_predictions(path: str | Path) -> dict[str, Prediction]:
    predictions = [Prediction.from_dict(item) for item in _read_jsonl(path)]
    return {pred.case_id: pred for pred in predictions}


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_prediction(case: EvaluationCase, prediction: Prediction) -> dict[str, Any]:
    """Score one prediction against one case."""
    answer = prediction.answer or ""
    answer_lower = answer.lower()

    source_stats = _score_sources(case.expected_sources, prediction.sources)
    retrieval_stats = _score_sources(case.expected_sources, prediction.retrieved_chunks)
    contains_score = _contains_score(answer_lower, case.expected_answer_contains)
    number_score = _number_score(answer, case.expected_numbers)
    no_answer_score = _no_answer_score(answer_lower, case.expected_no_answer)

    required_scores = []
    if case.expected_sources:
        required_scores.append(source_stats["recall"])
    if case.expected_answer_contains:
        required_scores.append(contains_score)
    if case.expected_numbers:
        required_scores.append(number_score)
    if case.expected_no_answer:
        required_scores.append(no_answer_score)

    passed = bool(required_scores) and all(score >= 1.0 for score in required_scores)
    if not required_scores:
        passed = True

    return {
        "id": case.case_id,
        "passed": passed,
        "citation_precision": source_stats["precision"],
        "citation_recall": source_stats["recall"],
        "retrieval_precision": retrieval_stats["precision"],
        "retrieval_recall": retrieval_stats["recall"],
        "answer_contains": contains_score,
        "number_accuracy": number_score,
        "no_answer_accuracy": no_answer_score,
        "latency_ms": prediction.latency_ms,
        "tags": list(case.tags),
    }


def evaluate_predictions(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Aggregate case-level scores into a stable report."""
    case_scores: list[dict[str, Any]] = []
    missing: list[str] = []

    for case in cases:
        pred = predictions.get(case.case_id)
        if pred is None:
            missing.append(case.case_id)
            continue
        case_scores.append(score_prediction(case, pred))

    latencies = [
        score["latency_ms"]
        for score in case_scores
        if isinstance(score.get("latency_ms"), (int, float))
    ]

    return {
        "summary": {
            "total_cases": len(case_scores) + len(missing),
            "scored_cases": len(case_scores),
            "missing_predictions": len(missing),
            "pass_rate": _avg(score["passed"] for score in case_scores),
            "citation_precision": _avg(score["citation_precision"] for score in case_scores),
            "citation_recall": _avg(score["citation_recall"] for score in case_scores),
            "retrieval_precision": _avg(score["retrieval_precision"] for score in case_scores),
            "retrieval_recall": _avg(score["retrieval_recall"] for score in case_scores),
            "answer_contains": _avg(score["answer_contains"] for score in case_scores),
            "number_accuracy": _avg(score["number_accuracy"] for score in case_scores),
            "no_answer_accuracy": _avg(score["no_answer_accuracy"] for score in case_scores),
            "p95_latency_ms": _percentile(latencies, 95),
        },
        "missing_case_ids": missing,
        "cases": case_scores,
    }


def trace_to_replay_case(trace: dict[str, Any]) -> EvaluationCase:
    """Convert one TraceLogger row dict into a replay case.

    The generated case intentionally stores references to sources and documents,
    but not the full retrieved context. This keeps replay fixtures small and
    avoids copying potentially sensitive document content into golden files.
    """
    trace_id = trace.get("trace_id")
    question = trace.get("query_original")
    if not trace_id or not question:
        raise ValueError("trace row missing trace_id or query_original")

    filters = _loads_json(trace.get("filter_conditions")) or {}
    sources = _loads_json(trace.get("sources_json")) or []

    return EvaluationCase(
        case_id=str(trace_id),
        question=str(question),
        document_names=tuple(str(item) for item in filters.get("doc_names", []) or []),
        expected_sources=tuple(ExpectedSource.from_dict(item) for item in sources),
        expected_answer_contains=(),
        expected_numbers=tuple(_extract_numbers(trace.get("answer") or "")),
        expected_no_answer=_looks_like_no_answer(trace.get("answer") or ""),
        tags=("trace_replay",),
        metadata={
            "trace_id": trace_id,
            "tenant_id": trace.get("tenant_id"),
            "model_name": trace.get("model_name"),
            "created_at": trace.get("created_at"),
        },
    )


def export_replay_cases_from_traces(
    traces: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> list[EvaluationCase]:
    cases = [trace_to_replay_case(trace) for trace in traces]
    write_jsonl(output_path, (case.to_dict() for case in cases))
    return cases


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _score_sources(
    expected_sources: tuple[ExpectedSource, ...],
    candidates: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    if not expected_sources:
        return {"precision": 1.0, "recall": 1.0}
    if not candidates:
        return {"precision": 0.0, "recall": 0.0}

    matched_expected = {
        idx
        for idx, expected in enumerate(expected_sources)
        if any(expected.matches(candidate) for candidate in candidates)
    }
    matched_candidates = sum(
        1 for candidate in candidates
        if any(expected.matches(candidate) for expected in expected_sources)
    )
    return {
        "precision": matched_candidates / len(candidates),
        "recall": len(matched_expected) / len(expected_sources),
    }


def _contains_score(answer_lower: str, expected_phrases: tuple[str, ...]) -> float:
    if not expected_phrases:
        return 1.0
    hits = sum(1 for phrase in expected_phrases if phrase.lower() in answer_lower)
    return hits / len(expected_phrases)


def _number_score(answer: str, expected_numbers: tuple[str, ...]) -> float:
    if not expected_numbers:
        return 1.0
    found = set(_extract_numbers(answer))
    hits = sum(1 for number in expected_numbers if number in found)
    return hits / len(expected_numbers)


def _no_answer_score(answer_lower: str, expected_no_answer: bool) -> float:
    if not expected_no_answer:
        return 1.0
    return 1.0 if _looks_like_no_answer(answer_lower) else 0.0


def _looks_like_no_answer(answer: str) -> bool:
    text = answer.lower()
    markers = [
        "couldn't find",
        "could not find",
        "sufficiently relevant",
        "no relevant",
        "not found",
        "无法",
        "没有找到",
        "未找到",
        "不足以",
    ]
    return any(marker in text for marker in markers)


def _extract_numbers(text: str) -> list[str]:
    return [_normalize_number(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def _normalize_number(value: str) -> str:
    return value.replace(",", "").strip()


def _filename_from_doc_id(doc_id: str) -> str | None:
    prefix = doc_id.split("::", 1)[0]
    if prefix.startswith("user_"):
        parts = prefix.split("_")
        if len(parts) > 2:
            return "_".join(parts[2:])
    return prefix or None


def _loads_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Iterable[Any]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return mean(items)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
