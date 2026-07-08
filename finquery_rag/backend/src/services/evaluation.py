"""Offline evaluation and trace replay helpers for FinQuery RAG.

The module intentionally avoids network, model, and vector-store dependencies.
It scores saved predictions and can convert trace records into replay cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import math
import re
from statistics import mean
from typing import Any, Iterable

from .answer_validation import validate_answer_calculations
from .financial_tools import (
    convert_scale,
    format_ratio_percent,
    growth_rate,
    percentage_share,
    sum_values,
    verify_sum,
)


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
class EvaluationCase:
    """A golden/replay RAG evaluation case."""

    case_id: str
    question: str
    expected_sources: tuple[ExpectedSource, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_numbers: tuple[str, ...] = ()
    expected_no_answer: bool = False
    expected_calculations: tuple[ExpectedCalculation, ...] = ()
    expected_intent: str | None = None
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
            expected_calculations=tuple(
                ExpectedCalculation.from_dict(item)
                for item in data.get("expected_calculations", [])
            ),
            expected_intent=(
                str(data["expected_intent"]) if data.get("expected_intent") else None
            ),
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
            "expected_calculations": [calc.to_dict() for calc in self.expected_calculations],
            "expected_intent": self.expected_intent,
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
    calculations: tuple[dict[str, Any], ...] = ()
    intent: str | None = None
    intent_confidence: float | None = None
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
            calculations=tuple(dict(item) for item in data.get("calculations", [])),
            intent=str(data["intent"]) if data.get("intent") else None,
            intent_confidence=_optional_float(data.get("intent_confidence")),
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
    calculation_score = _calculation_score(case.expected_calculations, prediction.calculations)
    answer_calculation_consistency = _answer_calculation_consistency(
        answer,
        prediction.calculations,
    )
    intent_accuracy = _intent_score(case.expected_intent, prediction.intent)

    required_scores = []
    if case.expected_sources:
        required_scores.append(source_stats["recall"])
    if case.expected_answer_contains:
        required_scores.append(contains_score)
    if case.expected_numbers:
        required_scores.append(number_score)
    if case.expected_no_answer:
        required_scores.append(no_answer_score)
    if case.expected_calculations:
        required_scores.append(calculation_score)
    if prediction.calculations:
        required_scores.append(answer_calculation_consistency)
    if case.expected_intent:
        required_scores.append(intent_accuracy)

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
        "calculation_accuracy": calculation_score,
        "answer_calculation_consistency": answer_calculation_consistency,
        "intent_accuracy": intent_accuracy,
        "expected_intent": case.expected_intent,
        "predicted_intent": prediction.intent,
        "intent_confidence": prediction.intent_confidence,
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
            "calculation_accuracy": _avg(score["calculation_accuracy"] for score in case_scores),
            "answer_calculation_consistency": _avg(
                score["answer_calculation_consistency"] for score in case_scores
            ),
            "intent_accuracy": _avg(score["intent_accuracy"] for score in case_scores),
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
        expected_intent=str(trace.get("intent")) if trace.get("intent") else None,
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



def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two evaluation reports and flag metric regressions.

    The comparison is intentionally deterministic and schema-stable so it can
    be used as a CI quality gate once project-specific thresholds are defined.
    """
    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    metric_names = [
        "pass_rate",
        "citation_precision",
        "citation_recall",
        "retrieval_precision",
        "retrieval_recall",
        "answer_contains",
        "number_accuracy",
        "no_answer_accuracy",
        "calculation_accuracy",
        "answer_calculation_consistency",
        "intent_accuracy",
    ]

    metrics = {}
    regressions = []
    for name in metric_names:
        base_value = _optional_float(baseline_summary.get(name)) or 0.0
        cand_value = _optional_float(candidate_summary.get(name)) or 0.0
        delta = cand_value - base_value
        metrics[name] = {
            "baseline": base_value,
            "candidate": cand_value,
            "delta": delta,
        }
        if delta < -regression_tolerance:
            regressions.append(name)

    latency_delta = None
    base_latency = _optional_float(baseline_summary.get("p95_latency_ms"))
    cand_latency = _optional_float(candidate_summary.get("p95_latency_ms"))
    if base_latency is not None and cand_latency is not None:
        latency_delta = cand_latency - base_latency

    baseline_cases = {
        item["id"]: item
        for item in baseline.get("cases", [])
        if "id" in item
    }
    candidate_cases = {
        item["id"]: item
        for item in candidate.get("cases", [])
        if "id" in item
    }
    newly_failed = sorted(
        case_id
        for case_id, base_case in baseline_cases.items()
        if base_case.get("passed") is True
        and candidate_cases.get(case_id, {}).get("passed") is False
    )
    newly_passed = sorted(
        case_id
        for case_id, cand_case in candidate_cases.items()
        if cand_case.get("passed") is True
        and baseline_cases.get(case_id, {}).get("passed") is False
    )

    return {
        "passed": not regressions and not newly_failed,
        "regression_tolerance": regression_tolerance,
        "metric_deltas": metrics,
        "regressions": regressions,
        "newly_failed": newly_failed,
        "newly_passed": newly_passed,
        "p95_latency_delta_ms": latency_delta,
        "baseline_missing_predictions": baseline_summary.get("missing_predictions", 0),
        "candidate_missing_predictions": candidate_summary.get("missing_predictions", 0),
    }

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




def _intent_score(expected_intent: str | None, predicted_intent: str | None) -> float:
    if not expected_intent:
        return 1.0
    return 1.0 if predicted_intent == expected_intent else 0.0


def _answer_calculation_consistency(answer: str, calculations: tuple[dict[str, Any], ...]) -> float:
    if not calculations:
        return 1.0
    validation = validate_answer_calculations(answer, list(calculations))
    return 1.0 if validation.ok else 0.0

def _calculation_score(
    expected_calculations: tuple[ExpectedCalculation, ...],
    predicted_calculations: tuple[dict[str, Any], ...],
) -> float:
    if not expected_calculations:
        return 1.0
    if not predicted_calculations:
        return 0.0

    predicted_by_id = {
        str(item.get("id") or item.get("calc_id") or item.get("operation")): item
        for item in predicted_calculations
    }
    hits = 0
    for expected in expected_calculations:
        predicted = predicted_by_id.get(expected.calc_id)
        if not predicted:
            continue
        if _prediction_matches_expected_calculation(expected, predicted):
            hits += 1
    return hits / len(expected_calculations)


def _prediction_matches_expected_calculation(
    expected: ExpectedCalculation,
    predicted: dict[str, Any],
) -> bool:
    if predicted.get("operation") and predicted.get("operation") != expected.operation:
        return False
    if expected.unit and predicted.get("unit") and predicted.get("unit") != expected.unit:
        return False

    predicted_value = _decimal_or_none(predicted.get("value"))
    if predicted_value is None:
        return False

    expected_value = _decimal_or_none(expected.expected_value)
    tolerance = _decimal_or_none(expected.tolerance)
    if expected_value is None or tolerance is None:
        return False

    deterministic = _run_expected_calculation(expected)
    if deterministic is None:
        return False

    return (
        abs(predicted_value - expected_value) <= abs(tolerance)
        and abs(deterministic - expected_value) <= abs(tolerance)
    )


def _run_expected_calculation(expected: ExpectedCalculation) -> Decimal | None:
    args = expected.args
    operation = expected.operation
    if operation == "growth_rate":
        result = growth_rate(args.get("current"), args.get("previous"))
    elif operation == "percentage_share":
        result = percentage_share(args.get("part"), args.get("total"))
    elif operation == "sum_values":
        result = sum_values(list(args.get("values", [])))
    elif operation == "verify_sum":
        result = verify_sum(
            list(args.get("components", [])),
            args.get("reported_total"),
            tolerance=args.get("tolerance", "0.01"),
        )
    elif operation == "convert_scale":
        result = convert_scale(
            args.get("value"),
            args.get("from_scale", ""),
            args.get("to_scale", ""),
        )
    elif operation == "format_ratio_percent":
        result = format_ratio_percent(args.get("value"))
    else:
        return None
    return result.value if result.ok else None


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

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
