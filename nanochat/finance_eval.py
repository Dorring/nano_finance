"""CPU-safe metrics for the lightweight financial evaluation suite."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Iterable


TASK_BY_SOURCE = {
    "finqa": "numeric_qa",
    "tatqa": "table_qa",
    "finer": "entity_extraction",
    "finred": "relation_extraction",
    "finsen": "sentiment",
    "fiqa": "sentiment",
    "ectsum": "summarization",
    "finance_r1": "instruction_following",
}

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:,\d{3})*\.\d+|\.\d+|\d+(?:,\d{3})*)(?:[eE][-+]?\d+)?%?"
)
_SENTIMENT_RE = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?\"'")


def extract_number(value: object) -> tuple[float, bool] | None:
    match = _NUMBER_RE.search(str(value))
    if not match:
        return None
    token = match.group(0).replace(",", "")
    is_percent = token.endswith("%")
    try:
        return float(token.rstrip("%")), is_percent
    except ValueError:
        return None


def numeric_match(reference: object, prediction: object, tolerance: float = 1e-3) -> bool:
    expected = extract_number(reference)
    actual = extract_number(prediction)
    if expected is None or actual is None:
        return False
    expected_value, expected_percent = expected
    actual_value, actual_percent = actual
    def close(left: float, right: float) -> bool:
        limit = max(tolerance, tolerance * abs(left))
        return math.isclose(left, right, abs_tol=limit, rel_tol=tolerance)

    if close(expected_value, actual_value):
        return True
    if expected_percent != actual_percent:
        if expected_percent:
            return close(expected_value, actual_value * 100)
        return close(expected_value * 100, actual_value)
    return False


def parse_json_list(value: object) -> list[dict] | None:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        return None
    return parsed


def extraction_items(value: object) -> set[tuple[str, ...]]:
    parsed = parse_json_list(value)
    if parsed is None:
        return set()
    return {
        tuple(f"{key}={normalize_text(item[key])}" for key in sorted(item))
        for item in parsed
    }


def rouge_l(reference: object, prediction: object) -> float:
    left = normalize_text(reference).split()
    right = normalize_text(prediction).split()
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(right)
    recall = lcs / len(left)
    return 2 * precision * recall / (precision + recall)


def _macro_f1(labels: list[str], predictions: list[str]) -> float:
    scores = []
    for label in sorted(set(labels) | set(predictions)):
        tp = sum(a == label and p == label for a, p in zip(labels, predictions))
        fp = sum(a != label and p == label for a, p in zip(labels, predictions))
        fn = sum(a == label and p != label for a, p in zip(labels, predictions))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def sentiment_label(value: object) -> str:
    match = _SENTIMENT_RE.search(str(value))
    return match.group(1).lower() if match else ""


def primary_metric(task: str) -> str:
    return {
        "numeric_qa": "numeric_accuracy",
        "table_qa": "exact_match",
        "entity_extraction": "micro_f1",
        "relation_extraction": "micro_f1",
        "sentiment": "macro_f1",
        "summarization": "rouge_l",
        "instruction_following": "non_empty_rate",
    }[task]


def _evaluate_tasks(rows: list[dict], tolerance: float) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        task = row.get("task_type") or TASK_BY_SOURCE.get(row.get("source"))
        if not task:
            raise ValueError(f"Unknown task for source {row.get('source')!r}")
        grouped[task].append(row)

    results = {}
    for task, task_rows in sorted(grouped.items()):
        references = [row["reference"] for row in task_rows]
        predictions = [row.get("prediction", "") for row in task_rows]
        exact = [
            normalize_text(reference) == normalize_text(prediction)
            for reference, prediction in zip(references, predictions)
        ]
        metrics: dict[str, float | int] = {
            "count": len(task_rows),
            "exact_match": sum(exact) / len(exact),
        }
        if task == "numeric_qa":
            matches = [
                numeric_match(reference, prediction, tolerance)
                for reference, prediction in zip(references, predictions)
            ]
            metrics["numeric_accuracy"] = sum(matches) / len(matches)
        elif task in {"entity_extraction", "relation_extraction"}:
            counts = Counter(tp=0, fp=0, fn=0)
            valid = 0
            for reference, prediction in zip(references, predictions):
                expected = extraction_items(reference)
                actual = extraction_items(prediction)
                valid += parse_json_list(prediction) is not None
                counts["tp"] += len(expected & actual)
                counts["fp"] += len(actual - expected)
                counts["fn"] += len(expected - actual)
            denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            metrics["json_valid_rate"] = valid / len(task_rows)
            metrics["micro_f1"] = 2 * counts["tp"] / denominator if denominator else 0.0
        elif task == "sentiment":
            labels = [sentiment_label(value) for value in references]
            predicted_labels = [sentiment_label(value) for value in predictions]
            metrics["accuracy"] = sum(a == p for a, p in zip(labels, predicted_labels)) / len(labels)
            metrics["macro_f1"] = _macro_f1(labels, predicted_labels)
        elif task == "summarization":
            scores = [rouge_l(a, p) for a, p in zip(references, predictions)]
            metrics["rouge_l"] = sum(scores) / len(scores)
        elif task == "instruction_following":
            metrics["non_empty_rate"] = sum(bool(normalize_text(p)) for p in predictions) / len(predictions)
        results[task] = metrics
    return results


def evaluate_records(records: Iterable[dict], tolerance: float = 1e-3) -> dict:
    """Evaluate rows containing source/task_type, reference, and prediction."""
    rows = list(records)
    tasks = _evaluate_tasks(rows, tolerance)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    sources = {}
    for source, source_rows in sorted(by_source.items()):
        source_tasks = _evaluate_tasks(source_rows, tolerance)
        sources[source] = {
            "count": len(source_rows),
            "tasks": source_tasks,
        }

    primary_scores = [
        metrics[primary_metric(task)]
        for task, metrics in tasks.items()
        if task != "instruction_following"
    ]
    empty_prediction_count = sum(
        row.get("prediction") is None
        or not str(row.get("prediction", "")).strip()
        for row in rows
    )
    contentless_prediction_count = sum(
        not normalize_text(row.get("prediction", ""))
        for row in rows
    )
    return {
        "count": len(rows),
        "empty_prediction_count": empty_prediction_count,
        "empty_prediction_rate": (
            empty_prediction_count / len(rows) if rows else 0.0
        ),
        "contentless_prediction_count": contentless_prediction_count,
        "contentless_prediction_rate": (
            contentless_prediction_count / len(rows) if rows else 0.0
        ),
        "macro_primary_score": (
            sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
        ),
        "tasks": tasks,
        "sources": sources,
    }
