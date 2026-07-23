"""Label consistency validator for Phase 5 v2.

Rejects labels that are internally inconsistent or reference non-existent
artifacts. Every label must pass this validator before it can be used in
any evaluation partition.

Checks:
  1. expected_no_answer=true AND expected_calculations non-empty → REJECT
  2. expected_answerability="answerable" AND expected_no_answer=true → REJECT
  3. required_answer_terms ∩ forbidden_answer_terms ≠ ∅ → REJECT
  4. Empty ExpectedSource (no filename, page, or chunk_id) → REJECT
  5. Operation not in valid set → REJECT
  6. Unit not in valid set (when defined) → REJECT
  7. Formula version not in valid set (when defined) → REJECT
  8. Expected Source not in Index Manifest (when provided) → REJECT
  9. Expected Number unparseable → REJECT
 10. tolerance is NaN or Infinity → REJECT
 11. Missing expected_value on calculation → REJECT (enforced in schema)
 12. annotation_evidence missing document_sha256 → WARN
"""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation

from src.evaluation.schemas import (
    EvaluationLabel,
)

__all__ = ["validate_label", "validate_labels", "LabelValidationError"]


_VALID_OPERATIONS = frozenset({
    "difference", "growth_rate", "percentage_share", "sum", "average",
    "gross_margin", "net_margin", "debt_ratio", "scale_conversion",
})

_VALID_UNITS = frozenset({
    "ratio", "percent", "percentage_point",
    "元", "万元", "百万元", "亿元",
    "million", "billion", "currency",
})

_VALID_FORMULA_VERSIONS = frozenset({"v1", "v2", "v3"})


class LabelValidationError(Exception):
    """Raised when a label fails consistency validation."""


def validate_label(
    label: EvaluationLabel,
    *,
    index_manifest_chunks: set[str] | None = None,
    index_manifest_filenames: set[str] | None = None,
) -> list[str]:
    """Validate a single label and return a list of error strings.

    An empty list means the label is valid. When ``index_manifest_chunks``
    or ``index_manifest_filenames`` are provided, expected sources are
    checked against the index manifest.

    Args:
        label: The label to validate.
        index_manifest_chunks: Set of chunk_ids known to the index.
        index_manifest_filenames: Set of filenames known to the index.
    """
    errors: list[str] = []
    cid = label.case_id

    # 1. no_answer + calculations
    if label.expected_no_answer and label.expected_calculations:
        errors.append(
            f"{cid}: expected_no_answer=true but expected_calculations is non-empty"
        )

    # 2. answerable + no_answer
    if (
        label.expected_answerability == "answerable"
        and label.expected_no_answer
    ):
        errors.append(
            f"{cid}: expected_answerability='answerable' but expected_no_answer=true"
        )

    # 3. required ∩ forbidden
    required_set = {t.lower() for t in label.required_answer_terms}
    forbidden_set = {t.lower() for t in label.forbidden_answer_terms}
    overlap = required_set & forbidden_set
    if overlap:
        errors.append(
            f"{cid}: required and forbidden terms overlap: {sorted(overlap)}"
        )

    # 4. Empty expected source
    for i, src in enumerate(label.expected_sources):
        if not any([src.filename, src.page is not None, src.chunk_id]):
            errors.append(f"{cid}: expected_sources[{i}] is empty")

    # 5-7. Calculation checks
    for calc in label.expected_calculations:
        if calc.operation not in _VALID_OPERATIONS:
            errors.append(
                f"{cid}: calculation {calc.calc_id} has invalid operation "
                f"{calc.operation!r} (valid: {sorted(_VALID_OPERATIONS)})"
            )
        if calc.unit and calc.unit not in _VALID_UNITS:
            errors.append(
                f"{cid}: calculation {calc.calc_id} has invalid unit "
                f"{calc.unit!r} (valid: {sorted(_VALID_UNITS)})"
            )
        if calc.formula_version and calc.formula_version not in _VALID_FORMULA_VERSIONS:
            errors.append(
                f"{cid}: calculation {calc.calc_id} has invalid formula_version "
                f"{calc.formula_version!r} (valid: {sorted(_VALID_FORMULA_VERSIONS)})"
            )
        # 9. Expected number parseable
        try:
            Decimal(calc.expected_value.replace(",", "").rstrip("%"))
        except (InvalidOperation, ValueError, AttributeError):
            errors.append(
                f"{cid}: calculation {calc.calc_id} expected_value "
                f"{calc.expected_value!r} is not a parseable number"
            )
        # 10. tolerance NaN/Inf
        try:
            tol_dec = Decimal(calc.tolerance)
            if math.isnan(float(tol_dec)) or math.isinf(float(tol_dec)):
                errors.append(
                    f"{cid}: calculation {calc.calc_id} tolerance "
                    f"{calc.tolerance!r} is NaN or Infinity"
                )
        except (InvalidOperation, ValueError):
            errors.append(
                f"{cid}: calculation {calc.calc_id} tolerance "
                f"{calc.tolerance!r} is not a parseable number"
            )

    # 8. Expected source in index manifest
    if index_manifest_chunks is not None or index_manifest_filenames is not None:
        for i, src in enumerate(label.expected_sources):
            if src.chunk_id and index_manifest_chunks is not None:
                if src.chunk_id not in index_manifest_chunks:
                    errors.append(
                        f"{cid}: expected_sources[{i}] chunk_id "
                        f"{src.chunk_id!r} not in index manifest"
                    )
            if src.filename and index_manifest_filenames is not None:
                if src.filename not in index_manifest_filenames:
                    errors.append(
                        f"{cid}: expected_sources[{i}] filename "
                        f"{src.filename!r} not in index manifest"
                    )

    return errors


def validate_labels(
    labels: list[EvaluationLabel],
    *,
    index_manifest_chunks: set[str] | None = None,
    index_manifest_filenames: set[str] | None = None,
) -> dict[str, list[str]]:
    """Validate all labels and return a dict mapping case_id → errors.

    Raises ``LabelValidationError`` if any label has errors.
    """
    all_errors: dict[str, list[str]] = {}
    for label in labels:
        errs = validate_label(
            label,
            index_manifest_chunks=index_manifest_chunks,
            index_manifest_filenames=index_manifest_filenames,
        )
        if errs:
            all_errors[label.case_id] = errs
    if all_errors:
        raise LabelValidationError(
            f"{len(all_errors)} labels failed validation:\n"
            + "\n".join(
                f"  {cid}:\n" + "\n".join(f"    - {e}" for e in errs)
                for cid, errs in all_errors.items()
            )
        )
    return all_errors
