"""Report builder for Phase 5 RAG evaluation.

Assembles the final evaluation report and run manifest. All outputs are
deterministic and JSON-serializable. The report builder never adds
timestamps, usernames, absolute paths, or any other non-deterministic
value to the report — only the caller-provided data appears in the
output.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = [
    "REQUIRED_REPORT_FIELDS",
    "build_final_report",
    "build_run_manifest_output",
    "validate_report_completeness",
]

REQUIRED_REPORT_FIELDS: tuple[str, ...] = (
    "metrics",
    "slices",
    "failures",
    "ablations",
    "calibration",
    "confidence_intervals",
    "manifest",
)


def build_final_report(
    *,
    metrics: dict[str, Any],
    slices: dict[str, Any],
    failures: dict[str, Any],
    ablations: dict[str, Any],
    calibration: dict[str, Any],
    confidence_intervals: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final evaluation report.

    All keyword arguments are required — there are no defaults. The
    report structure is:

    .. code-block:: json

        {
            "protocol_version": "1.0",
            "metrics": {...},
            "slices": {...},
            "failures": {...},
            "ablations": {...},
            "calibration": {...},
            "confidence_intervals": {...},
            "manifest": {...}
        }

    No timestamps, usernames, or absolute file paths are injected. The
    output is fully deterministic given the same inputs.

    Args:
        metrics: Aggregate metric values.
        slices: Per-slice metric breakdowns.
        failures: Failure taxonomy breakdown.
        ablations: Ablation comparison results.
        calibration: Calibration search results and selected candidate.
        confidence_intervals: CI bounds for key metrics.
        manifest: Run manifest (from ``build_run_manifest_output``).

    Returns:
        A JSON-serializable dict representing the final report.
    """
    return {
        "protocol_version": "1.0",
        "metrics": _ensure_serializable(metrics),
        "slices": _ensure_serializable(slices),
        "failures": _ensure_serializable(failures),
        "ablations": _ensure_serializable(ablations),
        "calibration": _ensure_serializable(calibration),
        "confidence_intervals": _ensure_serializable(confidence_intervals),
        "manifest": _ensure_serializable(manifest),
    }


def build_run_manifest_output(
    *,
    git_commit: str,
    git_dirty: bool,
    predictions_sha256: str,
    questions_sha256: str,
    labels_sha256: str,
    case_count: int,
    run_type: str,
    n_results: int = 0,
    random_seed: int = 0,
    config_hash: str | None = None,
    model_checkpoint_sha256: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable run manifest.

    The manifest captures the environment and input hashes needed to
    reproduce a sealed evaluation run. No timestamps or usernames are
    included.

    Args:
        git_commit: Git commit SHA of the run.
        git_dirty: Whether the working tree was dirty.
        predictions_sha256: SHA256 of the predictions JSONL.
        questions_sha256: SHA256 of the questions JSONL.
        labels_sha256: SHA256 of the labels JSONL.
        case_count: Number of evaluation cases.
        run_type: One of ``"baseline"``, ``"calibration"``, ``"sealed"``,
            ``"ablation"``.
        n_results: Retrieval ``n_results`` parameter.
        random_seed: Random seed used for the run.
        config_hash: SHA256 of the config file (optional).
        model_checkpoint_sha256: SHA256 of the model checkpoint (optional).
        embedding_model: Name of the embedding model (optional).
        reranker_model: Name of the reranker model (optional).
        python_version: Python version string (optional).

    Returns:
        A JSON-serializable manifest dict.
    """
    return {
        "git_commit": str(git_commit),
        "git_dirty": bool(git_dirty),
        "predictions_sha256": str(predictions_sha256),
        "questions_sha256": str(questions_sha256),
        "labels_sha256": str(labels_sha256),
        "case_count": int(case_count),
        "run_type": str(run_type),
        "n_results": int(n_results),
        "random_seed": int(random_seed),
        "config_hash": config_hash,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "python_version": python_version,
    }


def validate_report_completeness(report: dict[str, Any]) -> list[str]:
    """Return a list of missing required report fields.

    Args:
        report: The report dict to validate.

    Returns:
        A list of field names that are missing from ``report``. An
        empty list means the report is complete.
    """
    missing: list[str] = []
    for field in REQUIRED_REPORT_FIELDS:
        if field not in report:
            missing.append(field)
    return missing


def _ensure_serializable(obj: Any) -> Any:
    """Recursively convert a structure to JSON-serializable types."""
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {str(k): _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
