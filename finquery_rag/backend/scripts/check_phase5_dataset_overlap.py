#!/usr/bin/env python3
"""Dataset overlap checker for Phase 5 evaluation partitions.

Checks for data leakage between the ``dev`` / ``calibration`` / ``sealed``
partitions under ``eval_data/phase5/``. For each pair of partitions it
detects:

ERRORS (cause exit code 1):
    - Duplicate case_ids
    - Identical questions (exact match)
    - Normalized question duplicates (lowercase, strip, collapse whitespace)
    - Same document/page/metric/period combinations
    - Same expected number combinations
    - Cross-partition chunk_id overlaps

WARNINGS (do NOT fail):
    - High-similarity questions (Jaccard similarity on word sets >= 0.8)

Exit codes:
    0 = clean (or no phase5 data found)
    1 = overlap errors detected
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE5_DIR = BACKEND_DIR / "eval_data" / "phase5"
PARTITION_NAMES = ["dev", "calibration", "sealed"]


def load_partition_cases(partition_dir: Path) -> list[dict]:
    """Load questions from a partition directory.

    Reads ``questions.jsonl`` from the given directory. Each non-empty line
    must be a JSON object. Lines that fail to parse are skipped. Returns an
    empty list if the directory or file does not exist.
    """
    if not partition_dir.is_dir():
        return []
    qfile = partition_dir / "questions.jsonl"
    if not qfile.is_file():
        return []
    cases: list[dict] = []
    try:
        text = qfile.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            cases.append(obj)
    return cases


def _case_id(case: dict) -> str | None:
    """Return the case_id for a case, or None if absent."""
    cid = case.get("id") or case.get("case_id")
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    return None


def _question(case: dict) -> str | None:
    """Return the question text for a case, or None if absent."""
    q = case.get("question") or case.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return None


def _normalize_question(q: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return re.sub(r"\s+", " ", q.strip().lower())


def _doc_page_metric_period_key(case: dict) -> tuple | None:
    """Return a (document, page, metric, period) tuple if any field present."""
    document = case.get("document") or case.get("doc")
    page = case.get("page") or case.get("page_number")
    metric = case.get("metric")
    period = case.get("period")
    if document is None and page is None and metric is None and period is None:
        return None
    return (str(document), str(page), str(metric), str(period))


def _document_names(case: dict) -> set[str]:
    """Return the set of document names associated with a case.

    Handles both ``document_names`` (list) and ``document`` (string) fields.
    """
    out: set[str] = set()
    doc_names = case.get("document_names")
    if isinstance(doc_names, list):
        for d in doc_names:
            if isinstance(d, str) and d.strip():
                out.add(d.strip())
    elif isinstance(doc_names, str) and doc_names.strip():
        out.add(doc_names.strip())
    document = case.get("document") or case.get("doc")
    if isinstance(document, str) and document.strip():
        out.add(document.strip())
    return out


def check_document_name_overlap(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for document filename overlap between partitions.

    No document should appear in more than one partition — this is a hard
    data leakage violation.
    """
    docs_a: set[str] = set()
    for c in cases_a:
        docs_a |= _document_names(c)
    docs_b: set[str] = set()
    for c in cases_b:
        docs_b |= _document_names(c)
    common = docs_a & docs_b
    if not common:
        return []
    return [
        f"document name overlap {name_a} vs {name_b}: {sorted(common)}"
    ]


def _expected_numbers_key(case: dict) -> tuple | None:
    """Return a sorted tuple of expected numbers, or None."""
    nums = case.get("expected_numbers") or case.get("expected_number")
    if nums is None or isinstance(nums, bool):
        return None
    if isinstance(nums, (int, float)):
        return (float(nums),)
    if isinstance(nums, list):
        flat: list[float] = []
        for n in nums:
            if isinstance(n, bool):
                continue
            if isinstance(n, (int, float)):
                flat.append(float(n))
            elif isinstance(n, str):
                try:
                    flat.append(float(n))
                except ValueError:
                    continue
        if not flat:
            return None
        return tuple(sorted(flat))
    return None


def _chunk_ids(case: dict) -> set[str]:
    """Return the set of chunk_ids associated with a case."""
    ids = (
        case.get("chunk_ids")
        or case.get("chunks")
        or case.get("expected_chunk_ids")
    )
    out: set[str] = set()
    if isinstance(ids, list):
        for c in ids:
            if isinstance(c, str) and c.strip():
                out.add(c.strip())
            elif isinstance(c, dict):
                cid = c.get("chunk_id") or c.get("id")
                if isinstance(cid, str) and cid.strip():
                    out.add(cid.strip())
    elif isinstance(ids, str) and ids.strip():
        out.add(ids.strip())
    return out


def check_case_id_overlap(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for duplicate case_ids between partitions."""
    ids_a = {_case_id(c) for c in cases_a if _case_id(c)}
    ids_b = {_case_id(c) for c in cases_b if _case_id(c)}
    common = ids_a & ids_b
    if not common:
        return []
    return [
        f"case_id overlap {name_a} vs {name_b}: {sorted(common)}"
    ]


def check_question_overlap(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for identical/normalized question duplicates."""
    violations: list[str] = []
    qs_a = {_question(c) for c in cases_a if _question(c)}
    qs_b = {_question(c) for c in cases_b if _question(c)}
    common_exact = qs_a & qs_b
    for q in sorted(common_exact):
        violations.append(
            f"identical question {name_a} vs {name_b}: {q[:80]!r}"
        )
    norm_a = {_normalize_question(q) for q in qs_a}
    norm_b = {_normalize_question(q) for q in qs_b}
    common_norm = norm_a & norm_b
    exact_normalized = {_normalize_question(q) for q in common_exact}
    for q in sorted(common_norm):
        if q in exact_normalized:
            continue
        violations.append(
            f"normalized question overlap {name_a} vs {name_b}: {q[:80]!r}"
        )
    return violations


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def check_high_similarity(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str, threshold: float = 0.8,
) -> list[str]:
    """Check for high-similarity questions (warnings only).

    Compares each question in ``cases_a`` against each question in
    ``cases_b`` using Jaccard similarity over word sets. Returns a list of
    warning strings for pairs whose similarity is >= ``threshold``.
    Identical questions are skipped (they are reported as errors by
    :func:`check_question_overlap`).
    """
    warnings: list[str] = []
    qs_a = [q for q in (_question(c) for c in cases_a) if q]
    qs_b = [q for q in (_question(c) for c in cases_b) if q]
    words_a = [(q, set(re.findall(r"\w+", q.lower()))) for q in qs_a]
    words_b = [(q, set(re.findall(r"\w+", q.lower()))) for q in qs_b]
    seen: set[tuple[str, str]] = set()
    for qa, wa in words_a:
        for qb, wb in words_b:
            if qa == qb:
                continue
            sim = jaccard_similarity(wa, wb)
            if sim < threshold:
                continue
            key = (qa, qb) if qa < qb else (qb, qa)
            if key in seen:
                continue
            seen.add(key)
            warnings.append(
                f"high similarity {name_a} vs {name_b} "
                f"(jaccard={sim:.2f}): {qa[:60]!r} ~ {qb[:60]!r}"
            )
    return warnings


def check_doc_page_metric_period(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for same document/page/metric/period combinations."""
    keys_a: dict[tuple, list[dict]] = {}
    for c in cases_a:
        k = _doc_page_metric_period_key(c)
        if k is not None:
            keys_a.setdefault(k, []).append(c)
    keys_b: dict[tuple, list[dict]] = {}
    for c in cases_b:
        k = _doc_page_metric_period_key(c)
        if k is not None:
            keys_b.setdefault(k, []).append(c)
    common = set(keys_a) & set(keys_b)
    violations: list[str] = []
    for k in sorted(common, key=lambda x: tuple(str(s) for s in x)):
        violations.append(
            f"same doc/page/metric/period {name_a} vs {name_b}: {k}"
        )
    return violations


def check_expected_numbers(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for same expected number combinations."""
    keys_a: dict[tuple, list[dict]] = {}
    for c in cases_a:
        k = _expected_numbers_key(c)
        if k is not None:
            keys_a.setdefault(k, []).append(c)
    keys_b: dict[tuple, list[dict]] = {}
    for c in cases_b:
        k = _expected_numbers_key(c)
        if k is not None:
            keys_b.setdefault(k, []).append(c)
    common = set(keys_a) & set(keys_b)
    violations: list[str] = []
    for k in sorted(common):
        violations.append(
            f"same expected_numbers {name_a} vs {name_b}: {k}"
        )
    return violations


def check_chunk_id_overlap(
    cases_a: list[dict], cases_b: list[dict],
    name_a: str, name_b: str,
) -> list[str]:
    """Check for cross-partition chunk_id overlaps."""
    chunks_a: set[str] = set()
    for c in cases_a:
        chunks_a |= _chunk_ids(c)
    chunks_b: set[str] = set()
    for c in cases_b:
        chunks_b |= _chunk_ids(c)
    common = chunks_a & chunks_b
    if not common:
        return []
    return [
        f"chunk_id overlap {name_a} vs {name_b}: {sorted(common)}"
    ]


def main() -> int:
    """Run all overlap checks. Return 0 if clean, 1 if errors found."""
    if not PHASE5_DIR.is_dir():
        print("no phase5 data found", file=sys.stderr)
        print("PASS")
        return 0

    partitions: dict[str, list[dict]] = {}
    for name in PARTITION_NAMES:
        cases = load_partition_cases(PHASE5_DIR / name)
        if cases:
            partitions[name] = cases

    if not partitions:
        print("no phase5 data found", file=sys.stderr)
        print("PASS")
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    for a, b in combinations(sorted(partitions), 2):
        cases_a = partitions[a]
        cases_b = partitions[b]
        errors.extend(check_case_id_overlap(cases_a, cases_b, a, b))
        errors.extend(check_question_overlap(cases_a, cases_b, a, b))
        errors.extend(check_document_name_overlap(cases_a, cases_b, a, b))
        warnings.extend(check_high_similarity(cases_a, cases_b, a, b))
        errors.extend(check_doc_page_metric_period(cases_a, cases_b, a, b))
        errors.extend(check_expected_numbers(cases_a, cases_b, a, b))
        errors.extend(check_chunk_id_overlap(cases_a, cases_b, a, b))

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        print(
            f"FAIL: {len(errors)} overlap error(s) detected.", file=sys.stderr
        )
        return 1
    if warnings:
        print(f"PASS (with {len(warnings)} warning(s))")
    else:
        print("PASS: no overlap detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
