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
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .dataset_loader import load_labels
from .manifests import compute_jsonl_sha256
from .schemas import EvaluationLabel, EvaluationPrediction


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
_NO_ANSWER_MARKERS = (
    "couldn't find",
    "could not find",
    "sufficiently relevant",
    "no relevant",
    "not found",
    "cannot answer",
    "无法",
    "没有找到",
    "未找到",
    "不足以",
)


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
        case_report = _score_case(labels[case_id], predictions[case_id])
        case_reports.append(case_report)
        if case_report["passed"]:
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
# Scoring
# ---------------------------------------------------------------------------


def _score_case(label: EvaluationLabel, pred: EvaluationPrediction) -> dict[str, Any]:
    """Score one prediction against one label.

    A case passes when every applicable check passes. When no checks apply
    (the label carries no expected signal), the case passes by default.
    """
    answer = pred.answer or ""
    answer_norm = _normalize_text(answer)
    checks: list[dict[str, Any]] = []

    if label.required_answer_terms:
        ok = all(
            _normalize_text(term) in answer_norm
            for term in label.required_answer_terms
        )
        checks.append({"name": "required_terms", "passed": ok})

    if label.forbidden_answer_terms:
        ok = all(
            _normalize_text(term) not in answer_norm
            for term in label.forbidden_answer_terms
        )
        checks.append({"name": "forbidden_terms", "passed": ok})

    if label.expected_numbers:
        found = {_normalize_number(n) for n in _extract_numbers(answer)}
        ok = all(
            _normalize_number(n) in found for n in label.expected_numbers
        )
        checks.append({"name": "numbers", "passed": ok})

    if label.expected_no_answer:
        checks.append(
            {"name": "no_answer", "passed": _looks_like_no_answer(answer)}
        )

    if label.expected_sources:
        matched = sum(
            1
            for src in label.expected_sources
            if any(src.matches(c) for c in pred.sources)
        )
        recall = matched / len(label.expected_sources)
        checks.append(
            {"name": "sources", "passed": recall >= 1.0, "recall": recall}
        )

    if label.expected_calculations:
        checks.append(
            {"name": "calculations", "passed": _calculations_pass(label, pred)}
        )

    if label.expected_intent:
        checks.append(
            {
                "name": "intent",
                "passed": pred.intent == label.expected_intent,
            }
        )

    if label.expected_answerability:
        actual = (
            pred.answerability.get("status")
            if isinstance(pred.answerability, dict)
            else None
        )
        checks.append(
            {
                "name": "answerability",
                "passed": actual == label.expected_answerability,
            }
        )

    if label.expected_validation_status:
        actual = (
            pred.validation.get("status")
            if isinstance(pred.validation, dict)
            else None
        )
        checks.append(
            {
                "name": "validation",
                "passed": actual == label.expected_validation_status,
            }
        )

    passed = all(check["passed"] for check in checks) if checks else True
    return {
        "case_id": label.case_id,
        "passed": passed,
        "checks": checks,
    }


def _calculations_pass(
    label: EvaluationLabel, pred: EvaluationPrediction
) -> bool:
    """Return True when every expected calculation has a matching prediction.

    Matching is by ``calc_id`` (falling back to operation) with the
    predicted value within the expected tolerance.
    """
    pred_by_id: dict[str, dict[str, Any]] = {}
    for calc in pred.calculations:
        key = str(
            calc.get("id") or calc.get("calc_id") or calc.get("operation")
        )
        pred_by_id[key] = calc
    for expected in label.expected_calculations:
        candidate = pred_by_id.get(expected.calc_id)
        if candidate is None:
            return False
        try:
            pred_value = Decimal(str(candidate.get("value")))
            exp_value = Decimal(expected.expected_value)
            tolerance = Decimal(expected.tolerance or "0")
        except (InvalidOperation, ValueError, TypeError):
            return False
        if abs(pred_value - exp_value) > abs(tolerance):
            return False
    return True


# ---------------------------------------------------------------------------
# Text / number helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _normalize_number(value: str) -> str:
    return str(value).replace(",", "").strip().rstrip("%")


def _extract_numbers(text: str) -> list[str]:
    return [_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text)]


def _looks_like_no_answer(answer: str) -> bool:
    text = answer.lower()
    return any(marker in text for marker in _NO_ANSWER_MARKERS)


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
