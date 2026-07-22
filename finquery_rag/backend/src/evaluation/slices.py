"""Phase 5 slice reporting for FinQuery RAG evaluation.

Groups evaluation cases by slice tags and computes per-slice metrics.
All operations are deterministic and offline.
"""
from __future__ import annotations

from typing import Any, Callable

from src.evaluation.schemas import EvaluationLabel, EvaluationPrediction

__all__ = [
    "SLICE_CATEGORIES",
    "compute_slice_metrics",
]

# Mapping of category names to the slice tag values that belong to them.
# Each slice tag is a string that may appear in EvaluationLabel.slice_tags.
SLICE_CATEGORIES: dict[str, list[str]] = {
    "Intent": [
        "document_qa",
        "financial_calculation",
        "document_summary",
        "multi_document_comparison",
        "front_matter",
        "conversation",
        "unknown",
    ],
    "Language": [
        "chinese",
        "english",
        "mixed",
    ],
    "SourceType": [
        "narrative_paragraph",
        "table",
        "front_matter",
        "multi_page",
        "multi_document",
    ],
    "Difficulty": [
        "direct",
        "paraphrased",
        "multi_hop",
        "ambiguous",
        "adversarial_no_answer",
    ],
    "Safety": [
        "expected_answer",
        "expected_no_answer",
        "calculation_blocked",
        "unsupported_numeric_trap",
        "wrong_period_trap",
        "wrong_unit_trap",
        "wrong_citation_trap",
    ],
}


def compute_slice_metrics(
    labels: list[EvaluationLabel],
    predictions: list[EvaluationPrediction],
    metrics_func: Callable[
        [list[EvaluationLabel], list[EvaluationPrediction]], dict[str, Any]
    ],
) -> dict[str, dict[str, Any]]:
    """Compute metrics for each slice.

    For each slice tag, the subset of cases bearing that tag is scored
    independently using *metrics_func*. The result is a nested dict:

        {category: {slice_name: {metric_name: value, "sample_count": n}}}

    Slices with 0 cases are reported with ``sample_count=0`` and no other
    metric keys.
    """
    pred_by_id = {p.case_id: p for p in predictions}
    result: dict[str, dict[str, Any]] = {}

    for category, slice_names in SLICE_CATEGORIES.items():
        result[category] = {}
        for slice_name in slice_names:
            slice_labels = [
                label for label in labels if slice_name in label.slice_tags
            ]
            sample_count = len(slice_labels)
            if sample_count == 0:
                result[category][slice_name] = {"sample_count": 0}
                continue
            slice_preds = [
                pred_by_id[label.case_id]
                for label in slice_labels
                if label.case_id in pred_by_id
            ]
            if slice_preds:
                metrics = metrics_func(slice_labels, slice_preds)
            else:
                metrics = {}
            metrics["sample_count"] = sample_count
            result[category][slice_name] = metrics

    return result
