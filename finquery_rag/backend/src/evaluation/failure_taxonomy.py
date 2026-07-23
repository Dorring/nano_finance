"""Phase 5 failure classification for FinQuery RAG evaluation.

Classifies prediction failures into a fixed taxonomy with a strict priority
order. All operations are deterministic and offline.

Streaming Contract errors should be tested via a dedicated SSE test suite,
NOT inferred from the blind runner — the blind runner is non-streaming and
can only surface streaming-related warnings indirectly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.evaluation.metrics import (
    _get_answerability_status,
    _get_validation_status,
    _normalize_number,
    _numbers_in_text,
    _to_decimal,
    calculation_accuracy,
    citation_recall,
    recall_at_k,
    strict_case_pass,
)
from src.evaluation.schemas import EvaluationLabel, EvaluationPrediction

__all__ = [
    "FailureClassification",
    "SYSTEM_ERROR",
    "AUTH_OR_ENVIRONMENT",
    "INTENT_ERROR",
    "QUERY_REWRITE_ERROR",
    "RETRIEVAL_MISS",
    "RANKING_MISS",
    "DOCUMENT_SCOPE_ERROR",
    "CONTEXT_BUILD_ERROR",
    "ANSWERABILITY_FALSE_POSITIVE",
    "ANSWERABILITY_FALSE_NEGATIVE",
    "GENERATION_ERROR",
    "UNSUPPORTED_NUMERIC_RELEASE",
    "UNIT_PERIOD_ERROR",
    "CITATION_ERROR",
    "CALCULATION_ERROR",
    "VALIDATION_FALSE_PASS",
    "VALIDATION_FALSE_BLOCK",
    "ANSWER_QUALITY_TERM_VIOLATION",
    "STREAMING_CONTRACT_ERROR",
    "FAILURE_PRIORITY",
    "classify_failure",
    "classify_all_failures",
]


# ---------------------------------------------------------------------------
# Structured failure classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureClassification:
    """Structured classification of a single failure.

    Fields:
        observed_failure: The failure category constant that was observed.
        probable_root_cause: Most likely root cause or ``None`` if unknown.
        upstream_failures: Upstream contributing factors.
        downstream_failures: Downstream effects caused by this failure.
        confidence: Classification confidence — ``"high"``, ``"medium"``,
            or ``"low"``.
    """

    observed_failure: str
    probable_root_cause: str | None
    upstream_failures: tuple[str, ...]
    downstream_failures: tuple[str, ...]
    confidence: str


# ---------------------------------------------------------------------------
# Failure category constants (ordered by priority)
# ---------------------------------------------------------------------------

SYSTEM_ERROR = "system_error"
AUTH_OR_ENVIRONMENT = "auth_or_environment"
INTENT_ERROR = "intent_error"
QUERY_REWRITE_ERROR = "query_rewrite_error"
RETRIEVAL_MISS = "retrieval_miss"
RANKING_MISS = "ranking_miss"
DOCUMENT_SCOPE_ERROR = "document_scope_error"
CONTEXT_BUILD_ERROR = "context_build_error"
ANSWERABILITY_FALSE_POSITIVE = "answerability_false_positive"
ANSWERABILITY_FALSE_NEGATIVE = "answerability_false_negative"
GENERATION_ERROR = "generation_error"
UNSUPPORTED_NUMERIC_RELEASE = "unsupported_numeric_release"
UNIT_PERIOD_ERROR = "unit_period_error"
CITATION_ERROR = "citation_error"
CALCULATION_ERROR = "calculation_error"
VALIDATION_FALSE_PASS = "validation_false_pass"
VALIDATION_FALSE_BLOCK = "validation_false_block"
ANSWER_QUALITY_TERM_VIOLATION = "answer_quality_term_violation"
STREAMING_CONTRACT_ERROR = "streaming_contract_error"

# Fixed priority order (highest first).
FAILURE_PRIORITY: list[str] = [
    SYSTEM_ERROR,
    AUTH_OR_ENVIRONMENT,
    INTENT_ERROR,
    QUERY_REWRITE_ERROR,
    RETRIEVAL_MISS,
    RANKING_MISS,
    DOCUMENT_SCOPE_ERROR,
    CONTEXT_BUILD_ERROR,
    ANSWERABILITY_FALSE_POSITIVE,
    ANSWERABILITY_FALSE_NEGATIVE,
    GENERATION_ERROR,
    UNSUPPORTED_NUMERIC_RELEASE,
    UNIT_PERIOD_ERROR,
    CITATION_ERROR,
    CALCULATION_ERROR,
    VALIDATION_FALSE_PASS,
    VALIDATION_FALSE_BLOCK,
    ANSWER_QUALITY_TERM_VIOLATION,
    STREAMING_CONTRACT_ERROR,
]

_BLOCKED_STATUSES = frozenset({"blocked", "rejected", "failed", "error"})
_NO_ANSWER_ANSWERABILITY = frozenset({"no_answer", "not_answerable"})


def _normalize_text(text: str) -> str:
    """Normalize text for term comparison (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _is_auth_or_env_category(system_error_category: str | None) -> bool:
    """Check if the structured error category indicates auth/environment."""
    if not system_error_category:
        return False
    return system_error_category.startswith(
        "auth_"
    ) or system_error_category.startswith("env_")


def _supported_numbers(
    label: EvaluationLabel, prediction: EvaluationPrediction
) -> set[Any]:
    """Return the set of numbers supported by expected evidence.

    Includes expected numbers, expected calculation values, and actual
    calculation values produced by the model (so that a wrong but
    model-generated calculation is a CALCULATION_ERROR, not an
    UNSUPPORTED_NUMERIC_RELEASE).
    """
    supported: set[Any] = set()
    for n in label.expected_numbers:
        dec = _to_decimal(_normalize_number(n))
        if dec is not None:
            supported.add(dec)
    for calc in label.expected_calculations:
        dec = _to_decimal(_normalize_number(calc.expected_value))
        if dec is not None:
            supported.add(dec)
    for calc in prediction.calculations:
        val = calc.get("value") or calc.get("expected_value")
        if val is not None:
            dec = _to_decimal(_normalize_number(str(val)))
            if dec is not None:
                supported.add(dec)
    return supported


def _detect_failures(
    label: EvaluationLabel,
    prediction: EvaluationPrediction,
) -> list[str]:
    """Detect all failures for a single case, in priority order."""
    failures: list[str] = []

    # --- SYSTEM_ERROR / AUTH_OR_ENVIRONMENT ---
    # Classify by the structured ``system_error_category`` field rather
    # than matching exception class name prefixes.
    if (
        prediction.error_code is not None
        or prediction.system_error_category is not None
    ):
        if _is_auth_or_env_category(prediction.system_error_category):
            failures.append(AUTH_OR_ENVIRONMENT)
        else:
            failures.append(SYSTEM_ERROR)

    # --- INTENT_ERROR ---
    if label.expected_intent and prediction.intent != label.expected_intent:
        failures.append(INTENT_ERROR)

    # --- QUERY_REWRITE_ERROR ---
    debug = prediction.retrieval_debug or {}
    if debug.get("rewrite_error") or debug.get("query_rewrite_error"):
        failures.append(QUERY_REWRITE_ERROR)

    # --- RETRIEVAL_MISS / RANKING_MISS / DOCUMENT_SCOPE_ERROR ---
    if label.expected_sources and prediction.retrieved_chunks:
        full_recall = recall_at_k(
            label.expected_sources,
            prediction.retrieved_chunks,
            len(prediction.retrieved_chunks),
        )
        if full_recall <= 0:
            failures.append(RETRIEVAL_MISS)
        else:
            top_k_recall = recall_at_k(
                label.expected_sources,
                prediction.retrieved_chunks,
                min(5, len(prediction.retrieved_chunks)),
            )
            if top_k_recall < full_recall:
                failures.append(RANKING_MISS)

            expected_docs = {s.filename for s in label.expected_sources if s.filename}
            retrieved_docs = {
                c.get("filename") or c.get("doc_name")
                for c in prediction.retrieved_chunks
            }
            if expected_docs and not (expected_docs & retrieved_docs):
                failures.append(DOCUMENT_SCOPE_ERROR)
    elif label.expected_sources and not prediction.retrieved_chunks:
        failures.append(RETRIEVAL_MISS)

    # --- CONTEXT_BUILD_ERROR ---
    # Context insufficiency is correct behaviour for no-answer cases,
    # not an error.
    if prediction.context_sufficient is False:
        is_no_answer_case = (
            label.expected_no_answer
            or label.expected_answerability in _NO_ANSWER_ANSWERABILITY
        )
        if not is_no_answer_case:
            failures.append(CONTEXT_BUILD_ERROR)

    # --- ANSWERABILITY_FALSE_POSITIVE / ANSWERABILITY_FALSE_NEGATIVE ---
    # Uses ``not_answerable`` (production status) in addition to
    # ``no_answer``.
    if label.expected_answerability:
        pred_status = _get_answerability_status(prediction.answerability)
        if (
            label.expected_answerability in _NO_ANSWER_ANSWERABILITY
            and pred_status == "answerable"
        ):
            failures.append(ANSWERABILITY_FALSE_POSITIVE)
        elif label.expected_answerability == "answerable" and pred_status in (
            "no_answer",
            "not_answerable",
            "insufficient_context",
        ):
            failures.append(ANSWERABILITY_FALSE_NEGATIVE)

    # --- GENERATION_ERROR ---
    if not (prediction.answer or "").strip() and prediction.error_code is None:
        failures.append(GENERATION_ERROR)

    # --- UNSUPPORTED_NUMERIC_RELEASE ---
    # Applies to ALL cases: any number in the answer that is not
    # supported by expected evidence is an unsupported numeric release.
    answer_numbers = _numbers_in_text(prediction.answer)
    if answer_numbers:
        supported = _supported_numbers(label, prediction)
        if any(n not in supported for n in answer_numbers):
            failures.append(UNSUPPORTED_NUMERIC_RELEASE)

    # --- UNIT_PERIOD_ERROR ---
    if label.expected_calculations:
        answer_lower = (prediction.answer or "").lower()
        for calc in label.expected_calculations:
            if calc.unit and calc.unit.lower() not in answer_lower:
                failures.append(UNIT_PERIOD_ERROR)
                break

    # --- CITATION_ERROR ---
    if label.expected_sources:
        if citation_recall(label.expected_sources, prediction.sources) < 1.0:
            failures.append(CITATION_ERROR)

    # --- CALCULATION_ERROR ---
    if label.expected_calculations:
        if (
            calculation_accuracy(label.expected_calculations, prediction.calculations)
            < 1.0
        ):
            failures.append(CALCULATION_ERROR)

    # --- VALIDATION_FALSE_PASS / VALIDATION_FALSE_BLOCK ---
    val_status = _get_validation_status(prediction.validation)
    if label.expected_validation_status:
        if (
            label.expected_validation_status in _BLOCKED_STATUSES
            and val_status == "passed"
        ):
            failures.append(VALIDATION_FALSE_PASS)
        elif (
            label.expected_validation_status == "passed"
            and val_status in _BLOCKED_STATUSES
        ):
            failures.append(VALIDATION_FALSE_BLOCK)

    # --- ANSWER_QUALITY_TERM_VIOLATION ---
    # Required/Forbidden Terms failures must never become unclassified.
    answer_norm = _normalize_text(prediction.answer)
    if label.required_answer_terms:
        missing = [
            t
            for t in label.required_answer_terms
            if _normalize_text(t) not in answer_norm
        ]
        if missing:
            failures.append(ANSWER_QUALITY_TERM_VIOLATION)
    if label.forbidden_answer_terms:
        found = [
            t for t in label.forbidden_answer_terms if _normalize_text(t) in answer_norm
        ]
        if found:
            failures.append(ANSWER_QUALITY_TERM_VIOLATION)

    # --- STREAMING_CONTRACT_ERROR ---
    # NOTE: Streaming Contract errors should be tested via a dedicated
    # SSE test suite, NOT inferred from the blind runner. The blind
    # runner is non-streaming and can only surface indirect warnings.
    if any("stream" in w.lower() for w in prediction.warnings):
        failures.append(STREAMING_CONTRACT_ERROR)

    return failures


def classify_failure(
    label: EvaluationLabel,
    prediction: EvaluationPrediction,
) -> tuple[str | None, list[str]]:
    """Classify the primary and secondary failures for a case.

    Returns ``(primary_failure, secondary_failures)``. Returns
    ``(None, [])`` when the case passed strict evaluation.

    Classification priority is fixed: SYSTEM_ERROR > AUTH_OR_ENVIRONMENT >
    INTENT_ERROR > QUERY_REWRITE_ERROR > RETRIEVAL_MISS > RANKING_MISS >
    DOCUMENT_SCOPE_ERROR > CONTEXT_BUILD_ERROR > ANSWERABILITY_FALSE_POSITIVE
    > ANSWERABILITY_FALSE_NEGATIVE > GENERATION_ERROR >
    UNSUPPORTED_NUMERIC_RELEASE > UNIT_PERIOD_ERROR > CITATION_ERROR >
    CALCULATION_ERROR > VALIDATION_FALSE_PASS > VALIDATION_FALSE_BLOCK >
    ANSWER_QUALITY_TERM_VIOLATION > STREAMING_CONTRACT_ERROR.

    Note: Streaming Contract errors should be verified via a dedicated
    SSE test suite rather than inferred from blind runner output.
    """
    if strict_case_pass(label, prediction):
        return None, []

    failures = _detect_failures(label, prediction)
    if not failures:
        # Case didn't pass strict but no specific failure was identified.
        return "unclassified", []

    # Sort by fixed priority order.
    priority_map = {name: idx for idx, name in enumerate(FAILURE_PRIORITY)}
    failures.sort(key=lambda f: priority_map.get(f, len(FAILURE_PRIORITY)))
    return failures[0], failures[1:]


def classify_all_failures(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
) -> dict[str, Any]:
    """Classify failures for all paired label/prediction cases.

    Returns ``{case_id: {"primary_failure": str | None,
    "secondary_failures": list[str], "passed": bool}}``.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    result: dict[str, Any] = {}
    for label in labels:
        pred = pred_by_id.get(label.case_id)
        if pred is None:
            result[label.case_id] = {
                "primary_failure": SYSTEM_ERROR,
                "secondary_failures": [],
                "passed": False,
            }
            continue
        primary, secondary = classify_failure(label, pred)
        result[label.case_id] = {
            "primary_failure": primary,
            "secondary_failures": secondary,
            "passed": primary is None,
        }
    return result
