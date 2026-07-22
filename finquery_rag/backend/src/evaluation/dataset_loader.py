"""Phase 5 dataset loaders for separated questions and labels JSONL.

This module loads ``EvaluationQuery`` (questions only) and ``EvaluationLabel``
objects from JSONL files. The split enforces isolation: the blind runner
only ever sees ``EvaluationQuery`` objects, while the sealed scorer only
ever sees ``EvaluationLabel`` objects.

Duplicate ``case_id`` values within a single file raise ``ValueError``, as
do ``case_id`` set mismatches between the questions file and the labels
file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .schemas import EvaluationLabel, EvaluationQuery


def _read_jsonl_rows(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a JSONL file into a list of (line_no, row_dict) tuples.

    Blank lines and lines starting with ``#`` are skipped. Each non-skipped
    line must be a JSON object.
    """
    rows: list[tuple[int, dict[str, Any]]] = []
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
            if not isinstance(item, dict):
                raise ValueError(
                    f"invalid JSONL at {path}:{line_no}: row must be an object"
                )
            rows.append((line_no, item))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL rows atomically.

    The caller gets either the complete new file or the previous file
    remains untouched. Partial files are never left on disk because a
    failed write would silently poison later regression comparisons.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            for row_no, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {row_no} must be an object")
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_name, out)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_queries(path: str | Path) -> list[EvaluationQuery]:
    """Load ``EvaluationQuery`` objects from a questions-only JSONL file.

    Raises ``ValueError`` on duplicate ``case_id`` values or malformed rows.
    """
    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    for line_no, item in _read_jsonl_rows(path):
        try:
            query = EvaluationQuery.from_dict(item)
        except ValueError as exc:
            raise ValueError(
                f"invalid evaluation query at {path}:{line_no}: {exc}"
            ) from exc
        if query.case_id in seen:
            raise ValueError(
                f"duplicate evaluation query case_id {query.case_id!r} "
                f"at {path}:{line_no}"
            )
        seen.add(query.case_id)
        queries.append(query)
    return queries


def load_labels(path: str | Path) -> list[EvaluationLabel]:
    """Load ``EvaluationLabel`` objects from a labels JSONL file.

    Raises ``ValueError`` on duplicate ``case_id`` values or malformed rows.
    """
    labels: list[EvaluationLabel] = []
    seen: set[str] = set()
    for line_no, item in _read_jsonl_rows(path):
        try:
            label = EvaluationLabel.from_dict(item)
        except ValueError as exc:
            raise ValueError(
                f"invalid evaluation label at {path}:{line_no}: {exc}"
            ) from exc
        if label.case_id in seen:
            raise ValueError(
                f"duplicate evaluation label case_id {label.case_id!r} "
                f"at {path}:{line_no}"
            )
        seen.add(label.case_id)
        labels.append(label)
    return labels


def load_queries_and_labels(
    questions_path: str | Path,
    labels_path: str | Path,
) -> tuple[list[EvaluationQuery], list[EvaluationLabel]]:
    """Load questions and labels, verifying ``case_id`` sets match exactly.

    Raises ``ValueError`` when the set of ``case_id`` values in the
    questions file does not match the set in the labels file (missing or
    extra entries).
    """
    queries = load_queries(questions_path)
    labels = load_labels(labels_path)
    query_ids = {q.case_id for q in queries}
    label_ids = {label.case_id for label in labels}
    missing = sorted(label_ids - query_ids)
    extra = sorted(query_ids - label_ids)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing label case_ids: {missing}")
        if extra:
            parts.append(f"extra label case_ids: {extra}")
        raise ValueError(
            "case_id mismatch between questions and labels: " + "; ".join(parts)
        )
    return queries, labels
