"""Phase 5 sealed scorer.

Scores sealed predictions against labels independently of the RAG engine.
The scorer is a pure, offline, deterministic function of its inputs:

1. Verify the predictions file SHA256 matches the run manifest (protocol).
2. Recompute and record the labels SHA256.
3. Verify a 1:1 ``case_id`` correspondence between predictions and labels
   (no missing, no extra).
4. Score each prediction against its label.
5. Write the report atomically.

The scorer never calls the RAG engine and never modifies the predictions
file. The same inputs always produce byte-identical output.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .case_scorer import score_case
from .dataset_loader import load_labels
from .manifests import compute_jsonl_sha256
from .schemas import EvaluationPrediction


def score_sealed_predictions(
    *,
    predictions_path: Path,
    labels_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score sealed predictions against labels.

    Args:
        predictions_path: Path to the sealed predictions JSONL file.
        labels_path: Path to the labels JSONL file.
        protocol_path: Path to the run manifest JSON (carries the expected
            ``predictions_sha256`` and ``labels_sha256``).
        output_path: Path to write the scoring report JSON.

    Returns:
        The scoring report dict (also written to ``output_path``).

    Raises:
        ValueError: when the predictions SHA256 does not match the manifest,
            the labels SHA256 does not match, or the ``case_id`` sets are
            not in 1:1 correspondence.
    """
    protocol = _load_protocol(protocol_path)
    run_id = str(protocol.get("run_id", "unknown"))
    expected_pred_sha = protocol.get("predictions_sha256")
    expected_label_sha = protocol.get("labels_sha256")

    actual_pred_sha = compute_jsonl_sha256(predictions_path)
    if expected_pred_sha is not None and expected_pred_sha != actual_pred_sha:
        raise ValueError(
            "predictions SHA256 mismatch: manifest="
            f"{expected_pred_sha} actual={actual_pred_sha}"
        )

    actual_label_sha = compute_jsonl_sha256(labels_path)
    if expected_label_sha is not None and expected_label_sha != actual_label_sha:
        raise ValueError(
            "labels SHA256 mismatch: manifest="
            f"{expected_label_sha} actual={actual_label_sha}"
        )

    predictions = _load_predictions(predictions_path)
    labels = {label.case_id: label for label in load_labels(labels_path)}

    pred_ids = set(predictions)
    label_ids = set(labels)
    missing = sorted(label_ids - pred_ids)
    extra = sorted(pred_ids - label_ids)
    if missing:
        raise ValueError(f"missing predictions for case_ids: {missing}")
    if extra:
        raise ValueError(f"extra predictions for case_ids: {extra}")

    case_reports: list[dict[str, Any]] = []
    passed_count = 0
    for case_id in sorted(label_ids):
        case_score = score_case(labels[case_id], predictions[case_id])
        case_reports.append(case_score.to_dict())
        if case_score.passed:
            passed_count += 1

    total = len(case_reports)
    report: dict[str, Any] = {
        "run_id": run_id,
        "predictions_sha256": actual_pred_sha,
        "labels_sha256": actual_label_sha,
        "case_count": total,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
        "cases": case_reports,
    }
    _write_json_atomic(output_path, report)
    return report


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_protocol(path: Path) -> dict[str, Any]:
    """Load the run manifest JSON as a plain dict."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"protocol manifest at {path} must be a JSON object")
    return data


def _load_predictions(path: Path) -> dict[str, EvaluationPrediction]:
    """Load predictions JSONL into a dict keyed by ``case_id``.

    Raises ``ValueError`` on duplicate ``case_id`` values.
    """
    predictions: dict[str, EvaluationPrediction] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            pred = EvaluationPrediction.from_dict(item)
            if pred.case_id in predictions:
                raise ValueError(
                    f"duplicate prediction case_id {pred.case_id!r} "
                    f"at {path}:{line_no}"
                )
            predictions[pred.case_id] = pred
    return predictions


# ---------------------------------------------------------------------------
# Atomic output
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object atomically with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(content)
        os.replace(tmp_name, path)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
