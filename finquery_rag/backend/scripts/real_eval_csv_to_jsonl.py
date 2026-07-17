"""Convert a manually labeled FinQuery eval CSV into JSONL cases.

The CSV is easier for manual annotation. The generated JSONL is consumed by
`python -m src.eval_cli run/score/retrieval-eval-bundle`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PLACEHOLDER = "REPLACE_"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert FinQuery real-eval CSV labels to JSONL")
    parser.add_argument("--csv", required=True, help="Input labeling CSV")
    parser.add_argument("--out", required=True, help="Output JSONL")
    parser.add_argument("--allow-placeholders", action="store_true", help="Keep rows that still contain REPLACE_* placeholders")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(Path(args.csv).open("r", encoding="utf-8-sig", newline="")))
    cases = []
    skipped = []
    for idx, row in enumerate(rows, 2):
        if not args.allow_placeholders and _row_has_placeholder(row):
            skipped.append({"line": idx, "id": row.get("id"), "reason": "placeholder"})
            continue
        case = _row_to_case(row)
        if case is not None:
            cases.append(case)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(json.dumps({
        "input_rows": len(rows),
        "written_cases": len(cases),
        "skipped": skipped,
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _row_to_case(row: dict) -> dict | None:
    case_id = _clean(row.get("id"))
    question = _clean(row.get("question"))
    if not case_id or not question:
        return None

    expected_no_answer = _as_bool(row.get("expected_no_answer"))
    case = {
        "id": case_id,
        "question": question,
        "document_names": _split(row.get("document_names"), sep="|"),
        "expected_intent": _clean(row.get("expected_intent")) or "document_qa",
        "tags": _split(row.get("tags"), sep=","),
    }
    if expected_no_answer:
        case["expected_no_answer"] = True
    else:
        contains = _split(row.get("expected_answer_contains"), sep="|")
        numbers = _split(row.get("expected_numbers"), sep="|")
        sources = _parse_sources(row.get("expected_sources"))
        if contains:
            case["expected_answer_contains"] = contains
        if numbers:
            case["expected_numbers"] = numbers
        if sources:
            case["expected_sources"] = sources
    return {key: value for key, value in case.items() if value not in (None, "", [])}


def _parse_sources(value: str | None) -> list[dict]:
    sources = []
    for item in _split(value, sep="|"):
        if ":" not in item:
            continue
        filename, page = item.rsplit(":", 1)
        filename = _clean(filename)
        page = _clean(page)
        if not filename or not page:
            continue
        try:
            page_value: int | str = int(page)
        except ValueError:
            page_value = page
        sources.append({"filename": filename, "page": page_value})
    return sources


def _split(value: str | None, *, sep: str) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(sep) if part.strip()]


def _as_bool(value: str | None) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").strip().split())


def _row_has_placeholder(row: dict) -> bool:
    return any(PLACEHOLDER in str(value or "") for value in row.values())


if __name__ == "__main__":
    raise SystemExit(main())
