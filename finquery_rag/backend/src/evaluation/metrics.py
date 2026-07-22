"""Phase 5 evaluation metrics for FinQuery RAG.

This module computes retrieval, grounding, safety, utility, and system
metrics for offline RAG evaluation. All functions are deterministic and
offline — no network, no model calls.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from statistics import mean
from typing import Any

from src.evaluation.schemas import (
    EvaluationLabel,
    EvaluationPrediction,
    ExpectedCalculation,
    ExpectedSource,
)

__all__ = [
    # Retrieval
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "document_coverage",
    "expected_page_recall",
    # Grounding
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "numeric_accuracy",
    "metric_value_accuracy",
    "period_value_accuracy",
    "unit_scale_accuracy",
    "calculation_accuracy",
    "answer_calculation_consistency",
    "formula_version_accuracy",
    # Safety
    "answerability_accuracy",
    "answerability_macro_f1",
    "no_answer_precision",
    "no_answer_recall",
    "no_answer_f1",
    "unsupported_numeric_release_rate",
    "invalid_citation_release_rate",
    "calculation_mismatch_release_rate",
    "false_block_rate",
    "unsafe_answer_rate",
    "validator_fail_closed_rate",
    # Utility
    "strict_case_pass",
    "macro_strict_pass_rate",
    "supported_answer_coverage",
    "partial_answer_utility",
    "correct_refusal_rate",
    "answered_case_rate",
    # System
    "p50_latency",
    "p95_latency",
    "avg_retrieved_chunks",
    "avg_context_tokens",
    "avg_sources",
    "llm_call_rate",
    "validation_block_rate",
    "calculation_bypass_rate",
    "system_error_rate",
    # Aggregate
    "compute_all_metrics",
]

_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")

_NO_ANSWER_MARKERS = [
    "couldn't find",
    "could not find",
    "no relevant",
    "not found",
    "unable to find",
    "cannot answer",
    "no information",
    "insufficient",
    "无法",
    "没有找到",
    "未找到",
    "不足以",
    "无法回答",
]

_BLOCKED_STATUSES = frozenset({"blocked", "rejected", "failed", "error"})
_NO_ANSWER_STATUSES = frozenset({"no_answer", "not_answerable", "insufficient_context"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_number(value: str) -> str:
    """Normalize a number string by removing commas and trailing percent."""
    return str(value).replace(",", "").strip().rstrip("%")


def _to_decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal, returning None if invalid."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _numbers_in_text(text: str) -> list[Decimal]:
    """Extract all numbers from text as Decimal values."""
    numbers: list[Decimal] = []
    for match in _NUMBER_RE.finditer(text or ""):
        normalized = _normalize_number(match.group(0))
        decimal = _to_decimal(normalized)
        if decimal is not None:
            numbers.append(decimal)
    return numbers


def _number_in_answer(answer: str, expected: str, tolerance: str = "0") -> bool:
    """Check if expected number appears in answer within tolerance."""
    expected_dec = _to_decimal(_normalize_number(expected))
    if expected_dec is None:
        return False
    tol = _to_decimal(tolerance) or Decimal("0")
    for num in _numbers_in_text(answer):
        if abs(num - expected_dec) <= abs(tol):
            return True
    return False


def _get_answerability_status(answerability: dict[str, Any] | None) -> str | None:
    """Extract the answerability status from a prediction's answerability dict."""
    if answerability is None:
        return None
    return answerability.get("status") or answerability.get("answerability")


def _get_validation_status(validation: dict[str, Any] | None) -> str | None:
    """Extract the validation status from a prediction's validation dict."""
    if validation is None:
        return None
    return validation.get("status") or validation.get("result")


def _looks_like_no_answer(answer: str) -> bool:
    """Check if the answer text looks like a refusal."""
    text = (answer or "").lower()
    return any(marker in text for marker in _NO_ANSWER_MARKERS)


def _is_no_answer_prediction(prediction: EvaluationPrediction) -> bool:
    """Check if the prediction represents a refusal/no-answer."""
    if not (prediction.answer or "").strip():
        return True
    status = _get_answerability_status(prediction.answerability)
    if status in _NO_ANSWER_STATUSES:
        return True
    return _looks_like_no_answer(prediction.answer)


def _is_blocked_prediction(prediction: EvaluationPrediction) -> bool:
    """Check if the prediction was blocked by validation."""
    val_status = _get_validation_status(prediction.validation)
    if val_status in _BLOCKED_STATUSES:
        return True
    if not (prediction.answer or "").strip() and prediction.error_code is None:
        return True
    return False


def _percentile(values: list[float], percentile: int) -> float:
    """Compute the percentile of a list of values using linear interpolation."""
    if not values:
        return 0.0
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


def _avg(values: list[float]) -> float:
    """Return the mean of values, or 0.0 if empty."""
    if not values:
        return 0.0
    return mean(values)


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    k: int,
) -> float:
    """Fraction of expected sources found in the top-k retrieved chunks.

    Returns 1.0 when there are no expected sources.
    """
    if not expected_sources:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = list(retrieved_chunks)[:k]
    matched = sum(
        1
        for expected in expected_sources
        if any(expected.matches(chunk) for chunk in top_k)
    )
    return matched / len(expected_sources)


def precision_at_k(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    k: int,
) -> float:
    """Fraction of top-k retrieved chunks that match an expected source.

    Returns 0.0 when k <= 0 or no chunks are retrieved.
    """
    if k <= 0:
        return 0.0
    top_k = list(retrieved_chunks)[:k]
    if not top_k:
        return 0.0
    matched = sum(
        1
        for chunk in top_k
        if any(expected.matches(chunk) for expected in expected_sources)
    )
    return matched / len(top_k)


def mrr(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Mean reciprocal rank of the first matching retrieved chunk.

    Returns 1.0 when there are no expected sources.
    """
    if not expected_sources:
        return 1.0
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if any(expected.matches(chunk) for expected in expected_sources):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    k: int,
) -> float:
    """Normalized discounted cumulative gain with binary relevance.

    Returns 1.0 when there are no expected sources.
    """
    if not expected_sources:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = list(retrieved_chunks)[:k]
    dcg = 0.0
    for i, chunk in enumerate(top_k, 1):
        rel = 1.0 if any(expected.matches(chunk) for expected in expected_sources) else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(expected_sources), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def document_coverage(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of expected documents (by filename) found in retrieved chunks.

    Returns 1.0 when no expected sources have a filename.
    """
    expected_docs = {s.filename for s in expected_sources if s.filename}
    if not expected_docs:
        return 1.0
    retrieved_docs = {
        chunk.get("filename") or chunk.get("doc_name") for chunk in retrieved_chunks
    }
    matched = len(expected_docs & retrieved_docs)
    return matched / len(expected_docs)


def expected_page_recall(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    retrieved_chunks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of expected (filename, page) pairs found in retrieved chunks.

    Returns 1.0 when no expected sources have both filename and page set.
    """
    expected_pages = [
        s for s in expected_sources if s.filename is not None and s.page is not None
    ]
    if not expected_pages:
        return 1.0
    matched = 0
    for expected in expected_pages:
        if any(expected.matches(chunk) for chunk in retrieved_chunks):
            matched += 1
    return matched / len(expected_pages)


# ---------------------------------------------------------------------------
# Grounding Metrics
# ---------------------------------------------------------------------------


def citation_precision(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of cited sources that match an expected source.

    Returns 1.0 when there are no expected sources and no citations.
    """
    if not sources:
        return 1.0 if not expected_sources else 0.0
    matched = sum(
        1
        for src in sources
        if any(expected.matches(src) for expected in expected_sources)
    )
    return matched / len(sources)


def citation_recall(
    expected_sources: tuple[ExpectedSource, ...] | list[ExpectedSource],
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of expected sources that appear in cited sources.

    Returns 1.0 when there are no expected sources.
    """
    if not expected_sources:
        return 1.0
    matched = sum(
        1
        for expected in expected_sources
        if any(expected.matches(src) for src in sources)
    )
    return matched / len(expected_sources)


def citation_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def numeric_accuracy(
    answer: str,
    expected_numbers: tuple[str, ...] | list[str],
) -> float:
    """Fraction of expected numbers found in the answer.

    Returns 1.0 when there are no expected numbers.
    """
    if not expected_numbers:
        return 1.0
    answer_numbers = _numbers_in_text(answer)
    hits = 0
    for expected in expected_numbers:
        expected_dec = _to_decimal(_normalize_number(expected))
        if expected_dec is None:
            continue
        if any(abs(n - expected_dec) == 0 for n in answer_numbers):
            hits += 1
    return hits / len(expected_numbers)


def metric_value_accuracy(
    answer: str,
    expected_calculations: tuple[ExpectedCalculation, ...] | list[ExpectedCalculation],
) -> float:
    """Fraction of expected calculation values found in the answer.

    Returns 1.0 when there are no expected calculations.
    """
    if not expected_calculations:
        return 1.0
    answer_numbers = _numbers_in_text(answer)
    hits = 0
    for calc in expected_calculations:
        expected_val = _to_decimal(_normalize_number(calc.expected_value))
        if expected_val is None:
            continue
        tol = _to_decimal(calc.tolerance) or Decimal("0")
        if any(abs(n - expected_val) <= abs(tol) for n in answer_numbers):
            hits += 1
    return hits / len(expected_calculations)


def period_value_accuracy(
    answer: str,
    expected_calculations: tuple[ExpectedCalculation, ...] | list[ExpectedCalculation],
) -> float:
    """Fraction of period-specific expected values found in the answer.

    Only calculations with a 'period' key in args are considered.
    Returns 1.0 when no period-specific calculations exist.
    """
    period_calcs = [
        c for c in expected_calculations if any("period" in key.lower() for key in c.args)
    ]
    if not period_calcs:
        return 1.0
    answer_numbers = _numbers_in_text(answer)
    hits = 0
    for calc in period_calcs:
        expected_val = _to_decimal(_normalize_number(calc.expected_value))
        if expected_val is None:
            continue
        tol = _to_decimal(calc.tolerance) or Decimal("0")
        if any(abs(n - expected_val) <= abs(tol) for n in answer_numbers):
            hits += 1
    return hits / len(period_calcs)


def unit_scale_accuracy(
    answer: str,
    expected_calculations: tuple[ExpectedCalculation, ...] | list[ExpectedCalculation],
) -> float:
    """Fraction of calculations where the expected unit/scale appears in the answer.

    Returns 1.0 when no calculations have a unit defined.
    """
    unit_calcs = [c for c in expected_calculations if c.unit]
    if not unit_calcs:
        return 1.0
    answer_lower = (answer or "").lower()
    hits = 0
    for calc in unit_calcs:
        if calc.unit and calc.unit.lower() in answer_lower:
            hits += 1
    return hits / len(unit_calcs)


def _calculation_matches(
    expected: ExpectedCalculation,
    predicted: dict[str, Any],
) -> bool:
    """Check if a prediction calculation matches an expected calculation."""
    pred_op = predicted.get("operation")
    if pred_op and pred_op != expected.operation:
        return False
    pred_val = _to_decimal(predicted.get("value"))
    expected_val = _to_decimal(_normalize_number(expected.expected_value))
    if pred_val is None or expected_val is None:
        return False
    tol = _to_decimal(expected.tolerance) or Decimal("0")
    if abs(pred_val - expected_val) > abs(tol):
        return False
    if expected.unit and predicted.get("unit"):
        if predicted.get("unit") != expected.unit:
            return False
    return True


def _match_prediction_calc(
    expected: ExpectedCalculation,
    prediction_calculations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the prediction calculation matching an expected one by id or operation."""
    pred_by_id = {
        str(p.get("id") or p.get("calc_id") or p.get("operation") or ""): p
        for p in prediction_calculations
    }
    pred = pred_by_id.get(expected.calc_id)
    if pred is not None:
        return pred
    return next(
        (p for p in prediction_calculations if p.get("operation") == expected.operation),
        None,
    )


def calculation_accuracy(
    expected_calculations: tuple[ExpectedCalculation, ...] | list[ExpectedCalculation],
    prediction_calculations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of expected calculations matched by prediction calculations.

    Returns 1.0 when there are no expected calculations.
    """
    if not expected_calculations:
        return 1.0
    if not prediction_calculations:
        return 0.0
    hits = 0
    for expected in expected_calculations:
        pred = _match_prediction_calc(expected, prediction_calculations)
        if pred is not None and _calculation_matches(expected, pred):
            hits += 1
    return hits / len(expected_calculations)


def answer_calculation_consistency(
    answer: str,
    prediction_calculations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of prediction calculations whose values appear in the answer.

    Returns 1.0 when there are no prediction calculations.
    """
    if not prediction_calculations:
        return 1.0
    answer_numbers = _numbers_in_text(answer)
    hits = 0
    for calc in prediction_calculations:
        val = _to_decimal(_normalize_number(str(calc.get("value", ""))))
        if val is None:
            continue
        if any(abs(n - val) == 0 for n in answer_numbers):
            hits += 1
    return hits / len(prediction_calculations)


def formula_version_accuracy(
    expected_calculations: tuple[ExpectedCalculation, ...] | list[ExpectedCalculation],
    prediction_calculations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> float:
    """Fraction of expected calculations using the correct formula/operation.

    Returns 1.0 when there are no expected calculations.
    """
    if not expected_calculations:
        return 1.0
    if not prediction_calculations:
        return 0.0
    hits = 0
    for expected in expected_calculations:
        pred = _match_prediction_calc(expected, prediction_calculations)
        if pred is not None and pred.get("operation") == expected.operation:
            hits += 1
    return hits / len(expected_calculations)


# ---------------------------------------------------------------------------
# Safety Metrics
# ---------------------------------------------------------------------------


def answerability_accuracy(
    expected_answerability: str | None,
    prediction_answerability: dict[str, Any] | None,
) -> float:
    """1.0 if prediction answerability status matches expected, else 0.0.

    Returns 1.0 when no expected answerability is defined.
    """
    if not expected_answerability:
        return 1.0
    pred_status = _get_answerability_status(prediction_answerability)
    return 1.0 if pred_status == expected_answerability else 0.0


def answerability_macro_f1(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Macro-averaged F1 over all answerability labels.

    Returns 1.0 when no answerability labels are defined.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    statuses: set[str] = set()
    for label in labels:
        if label.expected_answerability:
            statuses.add(label.expected_answerability)
        pred = pred_by_id.get(label.case_id)
        if pred:
            status = _get_answerability_status(pred.answerability)
            if status:
                statuses.add(status)
    if not statuses:
        return 1.0
    f1_scores: list[float] = []
    for status in statuses:
        tp = fp = fn = 0
        for label in labels:
            pred = pred_by_id.get(label.case_id)
            pred_status = _get_answerability_status(pred.answerability) if pred else None
            expected_status = label.expected_answerability
            if expected_status == status and pred_status == status:
                tp += 1
            elif expected_status != status and pred_status == status:
                fp += 1
            elif expected_status == status and pred_status != status:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(citation_f1(precision, recall))
    return mean(f1_scores)


def no_answer_precision(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Precision for the 'no answer' class.

    Of cases predicted as no-answer, the fraction that were truly no-answer.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    tp = fp = 0
    for label in labels:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if _is_no_answer_prediction(pred):
            if label.expected_no_answer:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def no_answer_recall(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Recall for the 'no answer' class.

    Of true no-answer cases, the fraction predicted as no-answer.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    tp = fn = 0
    for label in labels:
        if not label.expected_no_answer:
            continue
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            fn += 1
            continue
        if _is_no_answer_prediction(pred):
            tp += 1
        else:
            fn += 1
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def no_answer_f1(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """F1 for the 'no answer' class."""
    return citation_f1(
        no_answer_precision(labels, predictions),
        no_answer_recall(labels, predictions),
    )


def unsupported_numeric_release_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of no-answer cases where a number was released.

    Returns 0.0 when there are no no-answer cases.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    no_answer_cases = [label for label in labels if label.expected_no_answer]
    if not no_answer_cases:
        return 0.0
    released = 0
    for label in no_answer_cases:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if _numbers_in_text(pred.answer):
            released += 1
    return released / len(no_answer_cases)


def invalid_citation_release_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of cases with at least one invalid citation.

    A citation is invalid when it does not match any expected source
    (and expected sources are defined).
    """
    pred_by_id = {p.case_id: p for p in predictions}
    if not labels:
        return 0.0
    invalid = 0
    for label in labels:
        pred = pred_by_id.get(label.case_id)
        if pred is None or not pred.sources:
            continue
        if label.expected_sources:
            has_invalid = any(
                not any(es.matches(src) for es in label.expected_sources)
                for src in pred.sources
            )
            if has_invalid:
                invalid += 1
    return invalid / len(labels)


def calculation_mismatch_release_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of calculation cases where a wrong value was released.

    Returns 0.0 when there are no calculation cases.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    calc_cases = [label for label in labels if label.expected_calculations]
    if not calc_cases:
        return 0.0
    mismatched = 0
    for label in calc_cases:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if not (pred.answer or "").strip():
            continue
        if _is_no_answer_prediction(pred):
            continue
        acc = calculation_accuracy(label.expected_calculations, pred.calculations)
        if acc < 1.0:
            mismatched += 1
    return mismatched / len(calc_cases)


def false_block_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of answerable cases that were wrongly blocked.

    Returns 0.0 when there are no answerable cases.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    answerable_cases = [
        label
        for label in labels
        if not label.expected_no_answer and label.expected_answerability != "no_answer"
    ]
    if not answerable_cases:
        return 0.0
    blocked = 0
    for label in answerable_cases:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if _is_blocked_prediction(pred):
            blocked += 1
    return blocked / len(answerable_cases)


def unsafe_answer_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Overall fraction of cases with any safety violation.

    Combines unsupported numeric release, invalid citations, forbidden
    terms, and false blocks.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    if not labels:
        return 0.0
    unsafe = 0
    for label in labels:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        is_unsafe = False
        if label.expected_no_answer and _numbers_in_text(pred.answer):
            is_unsafe = True
        if label.expected_sources and pred.sources:
            if any(
                not any(es.matches(src) for es in label.expected_sources)
                for src in pred.sources
            ):
                is_unsafe = True
        if label.forbidden_answer_terms:
            answer_lower = (pred.answer or "").lower()
            if any(term.lower() in answer_lower for term in label.forbidden_answer_terms):
                is_unsafe = True
        if not label.expected_no_answer and _is_blocked_prediction(pred):
            is_unsafe = True
        if is_unsafe:
            unsafe += 1
    return unsafe / len(labels)


def validator_fail_closed_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of cases where the validator failed closed (blocked)."""
    if not predictions:
        return 0.0
    failed_closed = sum(1 for pred in predictions if _is_blocked_prediction(pred))
    return failed_closed / len(predictions)


# ---------------------------------------------------------------------------
# Utility Metrics
# ---------------------------------------------------------------------------


def strict_case_pass(label: EvaluationLabel, prediction: EvaluationPrediction) -> bool:
    """Return True only if ALL strict conditions are met.

    Conditions: intent correct AND retrieval satisfied AND citation satisfied
    AND expected numbers correct AND calculation correct AND answerability
    correct AND validation status correct AND no forbidden content AND no
    system error.
    """
    # No system error
    if prediction.error_code is not None:
        return False
    # Intent correct
    if label.expected_intent and prediction.intent != label.expected_intent:
        return False
    # Retrieval satisfied
    if label.expected_sources:
        full_recall = recall_at_k(
            label.expected_sources,
            prediction.retrieved_chunks,
            max(1, len(prediction.retrieved_chunks)),
        )
        if full_recall < 1.0:
            return False
    # Citation satisfied
    if label.expected_sources:
        if citation_recall(label.expected_sources, prediction.sources) < 1.0:
            return False
    # Expected numbers correct
    if label.expected_numbers:
        if numeric_accuracy(prediction.answer, label.expected_numbers) < 1.0:
            return False
    # Calculation correct
    if label.expected_calculations:
        if (
            calculation_accuracy(label.expected_calculations, prediction.calculations)
            < 1.0
        ):
            return False
    # Answerability correct
    if label.expected_answerability:
        if (
            answerability_accuracy(label.expected_answerability, prediction.answerability)
            < 1.0
        ):
            return False
    # Validation status correct
    if label.expected_validation_status:
        if _get_validation_status(prediction.validation) != label.expected_validation_status:
            return False
    # No forbidden content
    if label.forbidden_answer_terms:
        answer_lower = (prediction.answer or "").lower()
        if any(term.lower() in answer_lower for term in label.forbidden_answer_terms):
            return False
    return True


def macro_strict_pass_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Average strict pass rate over slice tags (not cases).

    Groups cases by slice tag and averages the per-tag pass rate.
    Falls back to overall rate when no slice tags are present.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    slice_tags: set[str] = set()
    for label in labels:
        slice_tags.update(label.slice_tags)
    if not slice_tags:
        if not labels:
            return 0.0
        passes = sum(
            1
            for label in labels
            if (pred := pred_by_id.get(label.case_id)) is not None
            and strict_case_pass(label, pred)
        )
        return passes / len(labels)
    rates: list[float] = []
    for tag in sorted(slice_tags):
        tag_labels = [label for label in labels if tag in label.slice_tags]
        if not tag_labels:
            continue
        total = 0
        passes = 0
        for label in tag_labels:
            pred = pred_by_id.get(label.case_id)
            if pred is None:
                continue
            total += 1
            if strict_case_pass(label, pred):
                passes += 1
        if total > 0:
            rates.append(passes / total)
    return mean(rates) if rates else 0.0


def supported_answer_coverage(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of answerable cases that got a supported answer.

    A supported answer is non-empty, not a refusal, has no system error,
    and fully cites expected sources.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    answerable = [
        label
        for label in labels
        if not label.expected_no_answer and label.expected_answerability != "no_answer"
    ]
    if not answerable:
        return 0.0
    supported = 0
    for label in answerable:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if not (pred.answer or "").strip():
            continue
        if _is_no_answer_prediction(pred):
            continue
        if pred.error_code is not None:
            continue
        if label.expected_sources:
            if citation_recall(label.expected_sources, pred.sources) < 1.0:
                continue
        supported += 1
    return supported / len(answerable)


def partial_answer_utility(
    label: EvaluationLabel,
    prediction: EvaluationPrediction,
) -> float:
    """Weighted partial credit across all evaluation dimensions.

    Weights: intent 0.10, retrieval 0.20, citation 0.20, numbers 0.20,
    calculation 0.15, answerability 0.10, safety 0.05.
    """
    weights = {
        "intent": 0.10,
        "retrieval": 0.20,
        "citation": 0.20,
        "numbers": 0.20,
        "calculation": 0.15,
        "answerability": 0.10,
        "safety": 0.05,
    }
    score = 0.0
    # Intent
    if label.expected_intent:
        score += weights["intent"] * (
            1.0 if prediction.intent == label.expected_intent else 0.0
        )
    else:
        score += weights["intent"]
    # Retrieval
    if label.expected_sources:
        score += weights["retrieval"] * recall_at_k(
            label.expected_sources,
            prediction.retrieved_chunks,
            max(1, len(prediction.retrieved_chunks)),
        )
    else:
        score += weights["retrieval"]
    # Citation
    if label.expected_sources:
        score += weights["citation"] * citation_recall(
            label.expected_sources, prediction.sources
        )
    else:
        score += weights["citation"]
    # Numbers
    if label.expected_numbers:
        score += weights["numbers"] * numeric_accuracy(
            prediction.answer, label.expected_numbers
        )
    else:
        score += weights["numbers"]
    # Calculation
    if label.expected_calculations:
        score += weights["calculation"] * calculation_accuracy(
            label.expected_calculations, prediction.calculations
        )
    else:
        score += weights["calculation"]
    # Answerability
    if label.expected_answerability:
        score += weights["answerability"] * answerability_accuracy(
            label.expected_answerability, prediction.answerability
        )
    else:
        score += weights["answerability"]
    # Safety
    safety_score = 1.0
    if prediction.error_code is not None:
        safety_score = 0.0
    if label.forbidden_answer_terms:
        answer_lower = (prediction.answer or "").lower()
        if any(t.lower() in answer_lower for t in label.forbidden_answer_terms):
            safety_score = 0.0
    score += weights["safety"] * safety_score
    return score


def correct_refusal_rate(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> float:
    """Fraction of no-answer cases correctly refused.

    A correct refusal is a no-answer prediction without any numbers released.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    no_answer_cases = [label for label in labels if label.expected_no_answer]
    if not no_answer_cases:
        return 0.0
    correct = 0
    for label in no_answer_cases:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            continue
        if _is_no_answer_prediction(pred) and not _numbers_in_text(pred.answer):
            correct += 1
    return correct / len(no_answer_cases)


def answered_case_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of cases that produced a non-empty, non-refusal answer."""
    if not predictions:
        return 0.0
    answered = sum(
        1
        for pred in predictions
        if (pred.answer or "").strip() and not _is_no_answer_prediction(pred)
    )
    return answered / len(predictions)


# ---------------------------------------------------------------------------
# System Metrics
# ---------------------------------------------------------------------------


def p50_latency(predictions: list[EvaluationPrediction]) -> float:
    """Median (p50) latency in milliseconds."""
    latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
    return _percentile(latencies, 50)


def p95_latency(predictions: list[EvaluationPrediction]) -> float:
    """95th percentile latency in milliseconds."""
    latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
    return _percentile(latencies, 95)


def avg_retrieved_chunks(predictions: list[EvaluationPrediction]) -> float:
    """Average number of retrieved chunks per prediction."""
    if not predictions:
        return 0.0
    return mean(len(p.retrieved_chunks) for p in predictions)


def avg_context_tokens(predictions: list[EvaluationPrediction]) -> float:
    """Approximate average context tokens derived from answer length.

    Uses the heuristic of ~4 characters per token.
    """
    if not predictions:
        return 0.0
    return mean(len(p.answer or "") / 4 for p in predictions)


def avg_sources(predictions: list[EvaluationPrediction]) -> float:
    """Average number of cited sources per prediction."""
    if not predictions:
        return 0.0
    return mean(len(p.sources) for p in predictions)


def llm_call_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of cases that made an LLM call (from retrieval_debug)."""
    if not predictions:
        return 0.0
    llm_calls = 0
    for pred in predictions:
        debug = pred.retrieval_debug or {}
        if (
            debug.get("llm_called")
            or debug.get("used_llm_rewrite")
            or debug.get("llm_rewrite")
        ):
            llm_calls += 1
    return llm_calls / len(predictions)


def validation_block_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of cases blocked by validation."""
    if not predictions:
        return 0.0
    blocked = sum(1 for pred in predictions if _is_blocked_prediction(pred))
    return blocked / len(predictions)


def calculation_bypass_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of calculation-intent cases that bypassed calculation.

    Cases with intent 'financial_calculation' but no structured
    calculations are counted as bypassed.
    """
    if not predictions:
        return 0.0
    calc_cases = [p for p in predictions if p.intent == "financial_calculation"]
    if not calc_cases:
        return 0.0
    bypassed = sum(1 for p in calc_cases if not p.calculations)
    return bypassed / len(calc_cases)


def system_error_rate(predictions: list[EvaluationPrediction]) -> float:
    """Fraction of cases with a system error (error_code set)."""
    if not predictions:
        return 0.0
    errors = sum(1 for p in predictions if p.error_code is not None)
    return errors / len(predictions)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_all_metrics(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> dict[str, Any]:
    """Compute all metrics and return a flat dictionary.

    Aggregates retrieval, grounding, safety, utility, and system metrics
    over all paired label/prediction cases.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    paired: list[tuple[EvaluationLabel, EvaluationPrediction]] = [
        (label, pred_by_id[label.case_id])
        for label in labels
        if label.case_id in pred_by_id
    ]

    result: dict[str, Any] = {}

    # --- Retrieval metrics (averaged over cases with expected_sources) ---
    retrieval_pairs = [(label, p) for label, p in paired if label.expected_sources]
    if retrieval_pairs:
        result["recall_at_1"] = _avg(
            [recall_at_k(label.expected_sources, p.retrieved_chunks, 1) for label, p in retrieval_pairs]
        )
        result["recall_at_3"] = _avg(
            [recall_at_k(label.expected_sources, p.retrieved_chunks, 3) for label, p in retrieval_pairs]
        )
        result["recall_at_5"] = _avg(
            [recall_at_k(label.expected_sources, p.retrieved_chunks, 5) for label, p in retrieval_pairs]
        )
        result["precision_at_5"] = _avg(
            [
                precision_at_k(label.expected_sources, p.retrieved_chunks, 5)
                for label, p in retrieval_pairs
            ]
        )
        result["mrr"] = _avg(
            [mrr(label.expected_sources, p.retrieved_chunks) for label, p in retrieval_pairs]
        )
        result["ndcg_at_5"] = _avg(
            [ndcg_at_k(label.expected_sources, p.retrieved_chunks, 5) for label, p in retrieval_pairs]
        )
        result["document_coverage"] = _avg(
            [
                document_coverage(label.expected_sources, p.retrieved_chunks)
                for label, p in retrieval_pairs
            ]
        )
        result["expected_page_recall"] = _avg(
            [
                expected_page_recall(label.expected_sources, p.retrieved_chunks)
                for label, p in retrieval_pairs
            ]
        )
    else:
        for key in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "precision_at_5",
            "mrr",
            "ndcg_at_5",
            "document_coverage",
            "expected_page_recall",
        ):
            result[key] = 0.0

    # --- Grounding metrics ---
    citation_pairs = [(label, p) for label, p in paired if label.expected_sources]
    if citation_pairs:
        precisions = [
            citation_precision(label.expected_sources, p.sources) for label, p in citation_pairs
        ]
        recalls = [
            citation_recall(label.expected_sources, p.sources) for label, p in citation_pairs
        ]
        result["citation_precision"] = _avg(precisions)
        result["citation_recall"] = _avg(recalls)
        result["citation_f1"] = citation_f1(
            result["citation_precision"], result["citation_recall"]
        )
    else:
        result["citation_precision"] = 0.0
        result["citation_recall"] = 0.0
        result["citation_f1"] = 0.0

    number_pairs = [(label, p) for label, p in paired if label.expected_numbers]
    if number_pairs:
        result["numeric_accuracy"] = _avg(
            [numeric_accuracy(p.answer, label.expected_numbers) for label, p in number_pairs]
        )
    else:
        result["numeric_accuracy"] = 0.0

    calc_pairs = [(label, p) for label, p in paired if label.expected_calculations]
    if calc_pairs:
        result["metric_value_accuracy"] = _avg(
            [
                metric_value_accuracy(p.answer, label.expected_calculations)
                for label, p in calc_pairs
            ]
        )
        result["period_value_accuracy"] = _avg(
            [
                period_value_accuracy(p.answer, label.expected_calculations)
                for label, p in calc_pairs
            ]
        )
        result["unit_scale_accuracy"] = _avg(
            [
                unit_scale_accuracy(p.answer, label.expected_calculations)
                for label, p in calc_pairs
            ]
        )
        result["calculation_accuracy"] = _avg(
            [
                calculation_accuracy(label.expected_calculations, p.calculations)
                for label, p in calc_pairs
            ]
        )
        result["formula_version_accuracy"] = _avg(
            [
                formula_version_accuracy(label.expected_calculations, p.calculations)
                for label, p in calc_pairs
            ]
        )
    else:
        for key in (
            "metric_value_accuracy",
            "period_value_accuracy",
            "unit_scale_accuracy",
            "calculation_accuracy",
            "formula_version_accuracy",
        ):
            result[key] = 0.0

    consistency_pairs = [(label, p) for label, p in paired if p.calculations]
    if consistency_pairs:
        result["answer_calculation_consistency"] = _avg(
            [
                answer_calculation_consistency(p.answer, p.calculations)
                for label, p in consistency_pairs
            ]
        )
    else:
        result["answer_calculation_consistency"] = 0.0

    # --- Safety metrics ---
    result["answerability_macro_f1"] = answerability_macro_f1(labels, predictions)
    result["no_answer_precision"] = no_answer_precision(labels, predictions)
    result["no_answer_recall"] = no_answer_recall(labels, predictions)
    result["no_answer_f1"] = no_answer_f1(labels, predictions)
    result["unsupported_numeric_release_rate"] = unsupported_numeric_release_rate(
        labels, predictions
    )
    result["invalid_citation_release_rate"] = invalid_citation_release_rate(
        labels, predictions
    )
    result["calculation_mismatch_release_rate"] = calculation_mismatch_release_rate(
        labels, predictions
    )
    result["false_block_rate"] = false_block_rate(labels, predictions)
    result["unsafe_answer_rate"] = unsafe_answer_rate(labels, predictions)
    result["validator_fail_closed_rate"] = validator_fail_closed_rate(predictions)

    # --- Utility metrics ---
    result["macro_strict_pass_rate"] = macro_strict_pass_rate(labels, predictions)
    result["supported_answer_coverage"] = supported_answer_coverage(labels, predictions)
    result["correct_refusal_rate"] = correct_refusal_rate(labels, predictions)
    result["answered_case_rate"] = answered_case_rate(predictions)
    result["strict_pass_rate"] = _avg(
        [1.0 if strict_case_pass(label, p) else 0.0 for label, p in paired]
    )
    result["partial_answer_utility"] = _avg(
        [partial_answer_utility(label, p) for label, p in paired]
    )

    # --- System metrics ---
    result["p50_latency_ms"] = p50_latency(predictions)
    result["p95_latency_ms"] = p95_latency(predictions)
    result["avg_retrieved_chunks"] = avg_retrieved_chunks(predictions)
    result["avg_context_tokens"] = avg_context_tokens(predictions)
    result["avg_sources"] = avg_sources(predictions)
    result["llm_call_rate"] = llm_call_rate(predictions)
    result["validation_block_rate"] = validation_block_rate(predictions)
    result["calculation_bypass_rate"] = calculation_bypass_rate(predictions)
    result["system_error_rate"] = system_error_rate(predictions)

    # --- Sample counts ---
    result["total_cases"] = len(labels)
    result["scored_cases"] = len(paired)

    return result
