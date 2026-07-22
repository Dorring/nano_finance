"""Phase 5 failure classification for FinQuery RAG evaluation.

Classifies prediction failures into a fixed taxonomy with a strict priority
order. All operations are deterministic and offline.
"""
from __future__ import annotations

from typing import Any

from src.evaluation.metrics import (
    _get_answerability_status,
    _get_validation_status,
    _numbers_in_text,
    calculation_accuracy,
    citation_recall,
    recall_at_k,
    strict_case_pass,
)
from src.evaluation.schemas import EvaluationLabel, EvaluationPrediction

__all__ = [
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
    "STREAMING_CONTRACT_ERROR",
    "FAILURE_PRIORITY",
    "classify_failure",
    "classify_all_failures",
]

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
    STREAMING_CONTRACT_ERROR,
]

_BLOCKED_STATUSES = frozenset({"blocked", "rejected", "failed", "error"})


def _has_auth_error(error_code: str) -> bool:
    """Check if the error code indicates an auth/environment issue."""
    return error_code.startswith("auth_") or error_code.startswith("env_")


def _detect_failures(
    label: EvaluationLabel,
    prediction: EvaluationPrediction,
) -> list[str]:
    """Detect all failures for a single case, in priority order."""
    failures: list[str] = []

    # --- SYSTEM_ERROR / AUTH_OR_ENVIRONMENT ---
    if prediction.error_code is not None:
        if _has_auth_error(prediction.error_code):
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
    if prediction.context_sufficient is False:
        failures.append(CONTEXT_BUILD_ERROR)

    # --- ANSWERABILITY_FALSE_POSITIVE / ANSWERABILITY_FALSE_NEGATIVE ---
    if label.expected_answerability:
        pred_status = _get_answerability_status(prediction.answerability)
        if (
            label.expected_answerability == "no_answer"
            and pred_status == "answerable"
        ):
            failures.append(ANSWERABILITY_FALSE_POSITIVE)
        elif (
            label.expected_answerability == "answerable"
            and pred_status in ("no_answer", "not_answerable", "insufficient_context")
        ):
            failures.append(ANSWERABILITY_FALSE_NEGATIVE)

    # --- GENERATION_ERROR ---
    if not (prediction.answer or "").strip() and prediction.error_code is None:
        failures.append(GENERATION_ERROR)

    # --- UNSUPPORTED_NUMERIC_RELEASE ---
    if label.expected_no_answer and _numbers_in_text(prediction.answer):
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

    # --- STREAMING_CONTRACT_ERROR ---
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
    STREAMING_CONTRACT_ERROR.
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
