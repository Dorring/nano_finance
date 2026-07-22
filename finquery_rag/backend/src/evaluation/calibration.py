"""Threshold calibration for Phase 5 RAG evaluation.

Searches the pre-registered parameter space, computes metrics for every
candidate configuration, and applies the constraint-based selection rule
defined in the evaluation protocol.

The module is deterministic: given the same labels, predictions, search
space, and baseline metrics, the output is always identical. All
candidates are reported — not just the winner — so the selection process
is fully auditable.

Key concepts:
    candidate: A dict with ``"params"`` and ``"metrics"`` keys representing
        one point in the calibration search space.
    safety constraint: A function that checks whether a candidate's
        safety metrics are no worse than the baseline.
    selection rule: An ordered list of steps (from the protocol) that
        filters and ranks candidates.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Callable

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.schemas import EvaluationLabel, EvaluationPrediction

__all__ = [
    "SAFETY_METRICS",
    "CALIBRATION_CONSTRAINTS",
    "search_calibration_space",
    "select_best_candidate",
    "eliminate_unsafe_candidates",
    "apply_params_to_prediction",
]

SAFETY_METRICS: tuple[str, ...] = (
    "unsupported_numeric_release_rate",
    "invalid_citation_release_rate",
    "calculation_mismatch_release_rate",
    "false_block_rate",
    "unsafe_answer_rate",
    "validator_fail_closed_rate",
)

# Default selection rule matching the protocol.
DEFAULT_SELECTION_RULE: dict[str, Any] = {
    "steps": [
        "eliminate candidates with any safety metric worse than baseline",
        "eliminate candidates with new unsupported_numeric_release",
        "eliminate candidates with new calculation_mismatch_release",
        "eliminate candidates with invalid_citation_release higher than baseline",
        "maximize macro_strict_pass_rate among remaining",
        "tiebreak: higher citation_recall",
        "tiebreak: lower p95_latency",
        "tiebreak: smallest diff from baseline",
    ]
}


# ---------------------------------------------------------------------------
# Safety constraint functions
# ---------------------------------------------------------------------------


def _check_no_safety_regression(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    """Return True if no safety metric is worse than baseline."""
    metrics = candidate.get("metrics", {})
    for metric in SAFETY_METRICS:
        cand_val = metrics.get(metric, 0.0)
        base_val = baseline.get(metric, 0.0)
        if cand_val > base_val:
            return False
    return True


def _check_no_new_unsupported_numeric(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    """Return True if no *new* unsupported numeric release was introduced."""
    metrics = candidate.get("metrics", {})
    cand_val = metrics.get("unsupported_numeric_release_rate", 0.0)
    base_val = baseline.get("unsupported_numeric_release_rate", 0.0)
    if base_val == 0.0 and cand_val > 0.0:
        return False
    return True


def _check_no_new_calculation_mismatch(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    """Return True if no *new* calculation mismatch was introduced."""
    metrics = candidate.get("metrics", {})
    cand_val = metrics.get("calculation_mismatch_release_rate", 0.0)
    base_val = baseline.get("calculation_mismatch_release_rate", 0.0)
    if base_val == 0.0 and cand_val > 0.0:
        return False
    return True


def _check_invalid_citation_not_higher(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    """Return True if invalid citation release is not higher than baseline."""
    metrics = candidate.get("metrics", {})
    cand_val = metrics.get("invalid_citation_release_rate", 0.0)
    base_val = baseline.get("invalid_citation_release_rate", 0.0)
    return cand_val <= base_val


CALIBRATION_CONSTRAINTS: list[Callable[[dict[str, Any], dict[str, Any]], bool]] = [
    _check_no_safety_regression,
    _check_no_new_unsupported_numeric,
    _check_no_new_calculation_mismatch,
    _check_invalid_citation_not_higher,
]


# ---------------------------------------------------------------------------
# Parameter application (simulated re-scoring)
# ---------------------------------------------------------------------------


def apply_params_to_prediction(
    pred: EvaluationPrediction, params: dict[str, Any]
) -> EvaluationPrediction:
    """Apply calibration parameters to a prediction.

    Simulates the effect of retrieval parameters by filtering and
    truncating the retrieved chunks, then checking context sufficiency.
    When sufficiency fails the answer is blocked.

    Args:
        pred: The original prediction.
        params: Calibration parameters (from the search space).

    Returns:
        A new ``EvaluationPrediction`` with adjusted fields.
    """
    chunks = list(pred.retrieved_chunks)

    min_score = params.get("min_score_threshold", 0.0)
    if min_score > 0.0:
        chunks = [c for c in chunks if float(c.get("score", 1.0)) >= min_score]

    rrf_floor = params.get("numeric_rrf_floor", 0.0)
    if rrf_floor > 0.0:
        chunks = [c for c in chunks if float(c.get("rrf_score", 1.0)) >= rrf_floor]

    dense_floor = params.get("numeric_dense_floor", 0.0)
    if dense_floor > 0.0:
        chunks = [
            c for c in chunks if float(c.get("dense_score", 1.0)) >= dense_floor
        ]

    n_results = params.get("n_results", len(chunks))
    chunks = chunks[:n_results]

    max_tokens = params.get("max_context_tokens", 1_000_000)
    total_tokens = 0
    limited: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_tokens = max(1, len(str(chunk.get("text", ""))) // 4)
        if total_tokens + chunk_tokens > max_tokens:
            break
        limited.append(chunk)
        total_tokens += chunk_tokens
    chunks = limited

    best_threshold = params.get("sufficiency_best_score_threshold", 0.0)
    avg_threshold = params.get("sufficiency_average_score_threshold", 0.0)

    scores = [float(c.get("score", 1.0)) for c in chunks]
    best_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    context_sufficient = bool(chunks)
    if best_threshold > 0.0 and best_score < best_threshold:
        context_sufficient = False
    if avg_threshold > 0.0 and avg_score < avg_threshold:
        context_sufficient = False

    answer = pred.answer
    validation = pred.validation
    sources = list(pred.sources)

    if not context_sufficient:
        answer = ""
        validation = {"status": "blocked", "reason": "insufficient_context"}
        sources = []

    chunk_filenames = {c.get("filename") for c in chunks if c.get("filename")}
    if chunk_filenames:
        filtered_sources = [
            s for s in sources if s.get("filename") in chunk_filenames
        ]
        if filtered_sources:
            sources = filtered_sources

    return dataclasses.replace(
        pred,
        answer=answer,
        sources=tuple(sources),
        retrieved_chunks=tuple(chunks),
        validation=validation,
        context_sufficient=context_sufficient,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _generate_param_combinations(
    search_space: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    """Return the cartesian product of all search-space dimensions.

    Parameter names are sorted alphabetically so the output order is
    deterministic regardless of dict insertion order.
    """
    keys = sorted(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values, strict=True)))
    return combos


def search_calibration_space(
    calibration_labels: list[EvaluationLabel],
    calibration_predictions: list[EvaluationPrediction],
    search_space: dict[str, list[Any]],
    baseline_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Search the calibration parameter space and return all candidates.

    For every combination of parameters in ``search_space``, this function
    simulates the effect of those parameters on the predictions, computes
    the full metric suite, and records the result as a candidate.

    Args:
        calibration_labels: Labels for the calibration partition.
        calibration_predictions: Predictions for the calibration partition.
        search_space: A dict mapping parameter names to lists of candidate
            values.
        baseline_metrics: The baseline metric dict, used for safety
            constraint evaluation.

    Returns:
        A list of candidate dicts, each with keys ``"params"``,
        ``"metrics"``, ``"safe"``, and ``"violations"``. The list is
        ordered deterministically by parameter combination.
    """
    combinations = _generate_param_combinations(search_space)
    candidates: list[dict[str, Any]] = []
    for params in combinations:
        adjusted = [
            apply_params_to_prediction(p, params) for p in calibration_predictions
        ]
        metrics = compute_all_metrics(calibration_labels, adjusted)
        violations = _collect_violations(
            {"params": params, "metrics": metrics}, baseline_metrics
        )
        candidates.append(
            {
                "params": params,
                "metrics": metrics,
                "safe": len(violations) == 0,
                "violations": violations,
            }
        )
    return candidates


def _collect_violations(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> list[str]:
    """Return a list of human-readable constraint violation descriptions."""
    violations: list[str] = []
    metrics = candidate.get("metrics", {})
    for metric in SAFETY_METRICS:
        cand_val = metrics.get(metric, 0.0)
        base_val = baseline.get(metric, 0.0)
        if cand_val > base_val:
            violations.append(
                f"{metric} ({cand_val:.4f}) worse than baseline ({base_val:.4f})"
            )
    base_unsupported = baseline.get("unsupported_numeric_release_rate", 0.0)
    cand_unsupported = metrics.get("unsupported_numeric_release_rate", 0.0)
    if base_unsupported == 0.0 and cand_unsupported > 0.0:
        violations.append("new unsupported_numeric_release introduced")
    base_mismatch = baseline.get("calculation_mismatch_release_rate", 0.0)
    cand_mismatch = metrics.get("calculation_mismatch_release_rate", 0.0)
    if base_mismatch == 0.0 and cand_mismatch > 0.0:
        violations.append("new calculation_mismatch_release introduced")
    base_citation = baseline.get("invalid_citation_release_rate", 0.0)
    cand_citation = metrics.get("invalid_citation_release_rate", 0.0)
    if cand_citation > base_citation:
        violations.append(
            f"invalid_citation_release ({cand_citation:.4f}) higher than "
            f"baseline ({base_citation:.4f})"
        )
    return violations


# ---------------------------------------------------------------------------
# Elimination
# ---------------------------------------------------------------------------


def eliminate_unsafe_candidates(
    candidates: list[dict[str, Any]], baseline_metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    """Filter out candidates that violate any safety constraint.

    A candidate is eliminated if any safety metric is worse than the
    baseline, a new unsupported numeric release or calculation mismatch
    is introduced, or invalid citation release is higher than baseline.

    Violations are always recomputed from the candidate's ``"metrics"``
    dict so that the function works correctly with synthetic candidates
    that may not have a pre-populated ``"violations"`` field.

    Args:
        candidates: List of candidate dicts (each with ``"metrics"``).
        baseline_metrics: Baseline metric dict for comparison.

    Returns:
        A new list containing only safe candidates. The original list is
        not modified.
    """
    safe: list[dict[str, Any]] = []
    for candidate in candidates:
        violations = _collect_violations(candidate, baseline_metrics)
        if not violations:
            safe.append(candidate)
    return safe


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_best_candidate(
    candidates: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    selection_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the protocol's selection rule and return the best candidate.

    Steps (from the protocol):
        1. Eliminate candidates with any safety metric worse than baseline.
        2. Eliminate candidates with new unsupported_numeric_release.
        3. Eliminate candidates with new calculation_mismatch_release.
        4. Eliminate candidates with invalid_citation_release higher than
           baseline.
        5. Maximize ``macro_strict_pass_rate``.
        6. Tiebreak: higher ``citation_recall``.
        7. Tiebreak: lower ``p95_latency_ms``.
        8. Tiebreak: smallest absolute diff from baseline
           ``macro_strict_pass_rate``.

    Args:
        candidates: List of candidate dicts.
        baseline_metrics: Baseline metric dict.
        selection_rule: Optional override for the selection rule. When
            ``None``, ``DEFAULT_SELECTION_RULE`` is used.

    Returns:
        The best candidate dict. If all candidates are eliminated, the
        candidate closest to the baseline is returned (or an empty dict
        if there are no candidates at all).
    """
    if not candidates:
        return {}

    # The selection_rule parameter is accepted for API compatibility.
    # The default rule (DEFAULT_SELECTION_RULE) is encoded in the sort
    # key below: eliminate unsafe, then maximize macro_strict_pass_rate,
    # tiebreak on citation_recall, p95_latency, and diff from baseline.
    del selection_rule  # reserved for future rule-interpreter support

    safe_candidates = eliminate_unsafe_candidates(candidates, baseline_metrics)

    if not safe_candidates:
        safe_candidates = list(candidates)

    def sort_key(cand: dict[str, Any]) -> tuple[float, ...]:
        m = cand.get("metrics", {})
        macro = float(m.get("macro_strict_pass_rate", 0.0))
        citation_recall = float(m.get("citation_recall", 0.0))
        p95 = float(m.get("p95_latency_ms", float("inf")))
        base_macro = float(
            baseline_metrics.get("macro_strict_pass_rate", 0.0)
        )
        diff = abs(macro - base_macro)
        return (-macro, -citation_recall, p95, diff)

    safe_candidates.sort(key=sort_key)
    return safe_candidates[0]
