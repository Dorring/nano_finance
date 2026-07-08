"""CPU-safe metrics for lightweight dialogue smoke/regression evaluation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable

from nanochat.finance_eval import normalize_text, rouge_l
from nanochat.chat_format import normalize_messages, validate_messages_for_generation


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z]+")
_SUSPICIOUS_QUESTION_RUN_RE = re.compile(r"\?{4,}")
_REFUSAL_RE = re.compile(
    r"(无法|不能|没有足够|不足以|未提供|无法确定|不知道|"
    r"can't|cannot|can not|not enough|insufficient|unable to determine)",
    re.IGNORECASE,
)


def _iter_quality_texts(row: dict) -> Iterable[tuple[str, str]]:
    for message_index, message in enumerate(row.get("messages", [])):
        yield f"messages[{message_index}].content", str(message.get("content", ""))
    for field in (
        "reference",
        "required_substrings",
        "forbidden_substrings",
    ):
        value = row.get(field)
        if value is None:
            continue
        for index, text in enumerate(_as_list(value)):
            yield f"{field}[{index}]", text


def validate_examples(examples: Iterable[dict], expected_split: str = "val") -> None:
    rows = list(examples)
    if not rows:
        raise ValueError("dialogue evaluation set is empty")
    ids = [row.get("id") for row in rows]
    if any(not identifier for identifier in ids):
        raise ValueError("dialogue evaluation IDs must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError("dialogue evaluation IDs must be unique")
    splits = {row.get("split") for row in rows}
    if splits != {expected_split}:
        raise ValueError(f"refusing non-{expected_split} data: splits={splits}")
    for row in rows:
        row_id = row.get("id")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{row_id}: messages must be a list")
        validate_messages_for_generation(normalize_messages(messages))
        if "reference" in row and not normalize_text(row.get("reference", "")):
            raise ValueError(f"{row_id}: reference is contentless")
        for field, text in _iter_quality_texts(row):
            if _SUSPICIOUS_QUESTION_RUN_RE.search(text):
                raise ValueError(
                    f"{row_id}: suspicious question-mark run in {field}: {text!r}"
                )


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"expected string/list value, got {type(value).__name__}")


def _contains_all(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def _contains_none(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return all(needle.lower() not in lowered for needle in needles)


def _json_valid(text: str) -> bool:
    try:
        json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _language_match(text: str, expected: str) -> bool:
    expected = expected.lower()
    if expected in {"", "any", "none"}:
        return True
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    if expected in {"zh", "zh-cn", "chinese"}:
        return cjk > 0 and cjk >= words
    if expected in {"en", "english"}:
        return words > 0 and cjk == 0
    raise ValueError(f"unknown expected_language={expected!r}")


def record_checks(row: dict, prediction: object | None = None) -> dict[str, float]:
    """Return per-record check scores in [0, 1].

    The evaluation schema is intentionally simple so hand-written smoke sets and
    data-thread generated regression sets can share one scorer.
    """
    text = str(row.get("prediction", "") if prediction is None else prediction)
    checks: dict[str, float] = {
        "non_empty": float(bool(normalize_text(text))),
    }
    required = _as_list(row.get("required_substrings"))
    if required:
        checks["required_substrings"] = float(_contains_all(text, required))
    forbidden = _as_list(row.get("forbidden_substrings"))
    if forbidden:
        checks["forbidden_substrings"] = float(_contains_none(text, forbidden))
    if "expect_refusal" in row:
        refused = bool(_REFUSAL_RE.search(text))
        checks["refusal"] = float(refused == bool(row["expect_refusal"]))
    if "expected_language" in row:
        checks["language"] = float(_language_match(text, str(row["expected_language"])))
    if "format" in row:
        expected_format = str(row["format"]).lower()
        if expected_format == "json":
            checks["json_valid"] = float(_json_valid(text))
        else:
            raise ValueError(f"unknown format={expected_format!r}")
    if "min_chars" in row:
        checks["min_chars"] = float(len(text.strip()) >= int(row["min_chars"]))
    if "max_chars" in row:
        checks["max_chars"] = float(len(text.strip()) <= int(row["max_chars"]))
    if "reference" in row:
        checks["rouge_l"] = rouge_l(row["reference"], text)
    return checks


def evaluate_records(records: Iterable[dict]) -> dict:
    rows = list(records)
    by_task: dict[str, list[dict]] = defaultdict(list)
    scored_rows = []
    for row in rows:
        checks = record_checks(row)
        score = sum(checks.values()) / len(checks) if checks else 0.0
        task = row.get("task_type", "dialogue")
        scored = {
            "id": row.get("id"),
            "task_type": task,
            "score": score,
            "checks": checks,
        }
        scored_rows.append(scored)
        by_task[task].append(scored)

    tasks = {}
    for task, task_rows in sorted(by_task.items()):
        check_names = sorted({
            name
            for row in task_rows
            for name in row["checks"]
        })
        metrics = {
            "count": len(task_rows),
            "score": sum(row["score"] for row in task_rows) / len(task_rows),
        }
        for name in check_names:
            values = [
                row["checks"][name]
                for row in task_rows
                if name in row["checks"]
            ]
            metrics[name] = sum(values) / len(values)
        tasks[task] = metrics

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
        "empty_prediction_rate": empty_prediction_count / len(rows) if rows else 0.0,
        "contentless_prediction_count": contentless_prediction_count,
        "contentless_prediction_rate": (
            contentless_prediction_count / len(rows) if rows else 0.0
        ),
        "macro_score": (
            sum(task["score"] for task in tasks.values()) / len(tasks)
            if tasks else 0.0
        ),
        "tasks": tasks,
        "records": scored_rows,
    }
