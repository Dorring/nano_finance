"""Unified case scorer — the single canonical implementation of case-level
pass/fail logic for Phase 5 evaluation.

This module replaces the two former parallel implementations:
    - ``metrics.strict_case_pass()``
    - ``sealed_scorer._score_case()``

Every consumer (sealed scorer, ``compute_all_metrics``, failure taxonomy,
slice metrics, calibration, ablation, final report) must call
``score_case()`` from this module. No other module may define its own
strict-pass logic.

The scorer performs 17 checks (when applicable):

 1. no_system_error
 2. intent_correct
 3. retrieval_at_fixed_k
 4. citation_recall_satisfied
 5. citation_precision_satisfied
 6. required_terms_present
 7. forbidden_terms_absent
 8. expected_numbers_correct
 9. no_answer_behaviour_correct
10. no_answer_no_unsupported_numeric
11. calculation_operation_correct
12. calculation_value_correct
13. calculation_unit_correct
14. formula_version_correct
15. answerability_correct
16. validation_status_correct
17. answer_calculation_consistency

Checks are only emitted when the label defines the relevant expected
signal, so a case is never penalised for a dimension it does not test.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.evaluation.schemas import (
    CaseCheck,
    CaseScore,
    EvaluationLabel,
    EvaluationPrediction,
    ExpectedCalculation,
)

__all__ = ["score_case", "case_passes", "RETRIEVAL_K"]


# Fixed K for retrieval evaluation (protocol-defined).
RETRIEVAL_K = 5

_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")

_NO_ANSWER_MARKERS = (
    "couldn't find", "could not find", "no relevant", "not found",
    "unable to find", "cannot answer", "no information", "insufficient",
    "无法", "没有找到", "未找到", "不足以", "无法回答",
)

_BLOCKED_STATUSES = frozenset({"blocked", "rejected", "failed", "error"})
_NO_ANSWER_STATUSES = frozenset({"no_answer", "not_answerable", "insufficient_context"})


# ---------------------------------------------------------------------------
# Helpers (self-contained — no import from metrics.py to avoid cycles)
# ---------------------------------------------------------------------------


def _normalize_number(value: str) -> str:
    return str(value).replace(",", "").strip().rstrip("%")


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _numbers_in_text(text: str) -> list[Decimal]:
    numbers: list[Decimal] = []
    for match in _NUMBER_RE.finditer(text or ""):
        d = _to_decimal(_normalize_number(match.group(0)))
        if d is not None:
            numbers.append(d)
    return numbers


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _looks_like_no_answer(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in _NO_ANSWER_MARKERS)


def _get_answerability_status(pred: EvaluationPrediction) -> str | None:
    if pred.answerability is None:
        return None
    return pred.answerability.get("status") or pred.answerability.get("answerability")


def _get_validation_status(pred: EvaluationPrediction) -> str | None:
    if pred.validation is None:
        return None
    return pred.validation.get("status") or pred.validation.get("result")


def _is_blocked(pred: EvaluationPrediction) -> bool:
    val = _get_validation_status(pred)
    if val in _BLOCKED_STATUSES:
        return True
    if not (pred.answer or "").strip() and pred.error_code is None:
        return True
    return False


def _is_no_answer(pred: EvaluationPrediction) -> bool:
    if not (pred.answer or "").strip():
        return True
    if _get_answerability_status(pred) in _NO_ANSWER_STATUSES:
        return True
    return _looks_like_no_answer(pred.answer)


def _source_matches(expected_filename: str | None, expected_page: Any,
                    expected_chunk_id: str | None, candidate: dict[str, Any]) -> bool:
    cand_id = candidate.get("chunk_id") or candidate.get("doc_id")
    if expected_chunk_id and cand_id != expected_chunk_id:
        return False
    if expected_filename:
        cand_fn = (candidate.get("filename")
                   or candidate.get("document_name")
                   or candidate.get("doc_name"))
        if cand_fn != expected_filename:
            return False
    if expected_page is not None:
        if str(candidate.get("page")) != str(expected_page):
            return False
    return True


def _recall_at_k(expected_sources, retrieved, k: int) -> float:
    if not expected_sources:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    matched = 0
    for es in expected_sources:
        if any(_source_matches(es.filename, es.page, es.chunk_id, c) for c in top_k):
            matched += 1
    return matched / len(expected_sources)


def _precision_at_k(expected_sources, retrieved, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    matched = sum(
        1 for c in top_k
        if any(_source_matches(es.filename, es.page, es.chunk_id, c) for es in expected_sources)
    )
    return matched / len(top_k)


def _citation_recall(expected_sources, sources) -> float:
    if not expected_sources:
        return 1.0
    matched = sum(
        1 for es in expected_sources
        if any(_source_matches(es.filename, es.page, es.chunk_id, s) for s in sources)
    )
    return matched / len(expected_sources)


def _citation_precision(expected_sources, sources) -> float:
    if not sources:
        return 1.0 if not expected_sources else 0.0
    matched = sum(
        1 for s in sources
        if any(_source_matches(es.filename, es.page, es.chunk_id, s) for es in expected_sources)
    )
    return matched / len(sources)


def _match_pred_calc(expected: ExpectedCalculation, pred_calcs) -> dict[str, Any] | None:
    pred_by_id = {
        str(p.get("id") or p.get("calc_id") or p.get("operation") or ""): p
        for p in pred_calcs
    }
    pred = pred_by_id.get(expected.calc_id)
    if pred is not None:
        return pred
    return next(
        (p for p in pred_calcs if p.get("operation") == expected.operation),
        None,
    )


# ---------------------------------------------------------------------------
# Canonical scorer
# ---------------------------------------------------------------------------


def score_case(label: EvaluationLabel, prediction: EvaluationPrediction) -> CaseScore:
    """Score one prediction against one label.

    Returns a ``CaseScore`` with all applicable checks. ``passed`` is
    True only when every applicable check passes.
    """
    checks: list[CaseCheck] = []
    answer = prediction.answer or ""
    answer_norm = _normalize_text(answer)

    # 1. no_system_error
    checks.append(CaseCheck(
        name="no_system_error",
        passed=prediction.error_code is None,
        detail=prediction.error_code,
    ))

    # 2. intent_correct
    if label.expected_intent:
        checks.append(CaseCheck(
            name="intent_correct",
            passed=prediction.intent == label.expected_intent,
            detail=f"expected={label.expected_intent} got={prediction.intent}",
        ))

    # 3. retrieval_at_fixed_k
    if label.expected_sources:
        rec = _recall_at_k(label.expected_sources, prediction.retrieved_chunks, RETRIEVAL_K)
        checks.append(CaseCheck(
            name="retrieval_at_fixed_k",
            passed=rec >= 1.0,
            detail=f"recall@{RETRIEVAL_K}={rec:.4f}",
        ))

    # 4. citation_recall_satisfied
    if label.expected_sources:
        cr = _citation_recall(label.expected_sources, prediction.sources)
        checks.append(CaseCheck(
            name="citation_recall_satisfied",
            passed=cr >= 1.0,
            detail=f"recall={cr:.4f}",
        ))

    # 5. citation_precision_satisfied
    if label.expected_sources and prediction.sources:
        cp = _citation_precision(label.expected_sources, prediction.sources)
        checks.append(CaseCheck(
            name="citation_precision_satisfied",
            passed=cp >= 1.0,
            detail=f"precision={cp:.4f}",
        ))

    # 6. required_terms_present
    if label.required_answer_terms:
        ok = all(_normalize_text(t) in answer_norm for t in label.required_answer_terms)
        missing = [t for t in label.required_answer_terms if _normalize_text(t) not in answer_norm]
        checks.append(CaseCheck(
            name="required_terms_present",
            passed=ok,
            detail=f"missing={missing}" if missing else None,
        ))

    # 7. forbidden_terms_absent
    if label.forbidden_answer_terms:
        found = [t for t in label.forbidden_answer_terms if _normalize_text(t) in answer_norm]
        checks.append(CaseCheck(
            name="forbidden_terms_absent",
            passed=not found,
            detail=f"found={found}" if found else None,
        ))

    # 8. expected_numbers_correct
    if label.expected_numbers:
        found_nums = {_normalize_number(n) for n in _extract_number_strings(answer)}
        missing_nums = [n for n in label.expected_numbers if _normalize_number(n) not in found_nums]
        checks.append(CaseCheck(
            name="expected_numbers_correct",
            passed=not missing_nums,
            detail=f"missing={missing_nums}" if missing_nums else None,
        ))

    # 9. no_answer_behaviour_correct
    if label.expected_no_answer:
        is_na = _is_no_answer(prediction)
        checks.append(CaseCheck(
            name="no_answer_behaviour_correct",
            passed=is_na,
            detail="expected no-answer but answer given" if not is_na else None,
        ))

    # 10. no_answer_no_unsupported_numeric
    if label.expected_no_answer:
        nums = _numbers_in_text(answer)
        checks.append(CaseCheck(
            name="no_answer_no_unsupported_numeric",
            passed=not nums,
            detail=f"released_numbers={[str(n) for n in nums]}" if nums else None,
        ))

    # 11-14. Calculation checks
    if label.expected_calculations:
        _add_calculation_checks(label, prediction, checks)

    # 15. answerability_correct
    if label.expected_answerability:
        actual = _get_answerability_status(prediction)
        checks.append(CaseCheck(
            name="answerability_correct",
            passed=actual == label.expected_answerability,
            detail=f"expected={label.expected_answerability} got={actual}",
        ))

    # 16. validation_status_correct
    if label.expected_validation_status:
        actual = _get_validation_status(prediction)
        checks.append(CaseCheck(
            name="validation_status_correct",
            passed=actual == label.expected_validation_status,
            detail=f"expected={label.expected_validation_status} got={actual}",
        ))

    # 17. answer_calculation_consistency
    if prediction.calculations:
        ans_nums = _numbers_in_text(answer)
        inconsistent = []
        for calc in prediction.calculations:
            val = _to_decimal(_normalize_number(str(calc.get("value", ""))))
            if val is None:
                continue
            if not any(abs(n - val) == 0 for n in ans_nums):
                inconsistent.append(str(calc.get("id", calc.get("operation", "?"))))
        checks.append(CaseCheck(
            name="answer_calculation_consistency",
            passed=not inconsistent,
            detail=f"calc_ids_not_in_answer={inconsistent}" if inconsistent else None,
        ))

    passed = all(c.passed for c in checks)
    failures = [c.name for c in checks if not c.passed]
    primary = failures[0] if failures else None
    secondary = tuple(failures[1:])
    return CaseScore(
        case_id=label.case_id,
        passed=passed,
        checks=tuple(checks),
        primary_failure=primary,
        secondary_failures=secondary,
    )


def case_passes(label: EvaluationLabel, prediction: EvaluationPrediction) -> bool:
    """Convenience wrapper: return only the pass/fail boolean."""
    return score_case(label, prediction).passed


# ---------------------------------------------------------------------------
# Calculation sub-checks
# ---------------------------------------------------------------------------


def _add_calculation_checks(
    label: EvaluationLabel,
    prediction: EvaluationPrediction,
    checks: list[CaseCheck],
) -> None:
    """Add the four calculation checks (11-14) to ``checks``."""
    pred_calcs = prediction.calculations

    # 11. calculation_operation_correct
    op_ok = True
    op_details: list[str] = []
    for expected in label.expected_calculations:
        pred = _match_pred_calc(expected, pred_calcs)
        if pred is None or pred.get("operation") != expected.operation:
            op_ok = False
            op_details.append(f"{expected.calc_id}: expected={expected.operation}")
    checks.append(CaseCheck(
        name="calculation_operation_correct",
        passed=op_ok,
        detail="; ".join(op_details) if op_details else None,
    ))

    # 12. calculation_value_correct
    val_ok = True
    val_details: list[str] = []
    for expected in label.expected_calculations:
        pred = _match_pred_calc(expected, pred_calcs)
        if pred is None:
            val_ok = False
            val_details.append(f"{expected.calc_id}: no matching prediction")
            continue
        pred_val = _to_decimal(pred.get("value"))
        exp_val = _to_decimal(_normalize_number(expected.expected_value))
        if pred_val is None or exp_val is None:
            val_ok = False
            val_details.append(f"{expected.calc_id}: unparseable value")
            continue
        tol = _to_decimal(expected.tolerance) or Decimal("0")
        if abs(pred_val - exp_val) > abs(tol):
            val_ok = False
            val_details.append(
                f"{expected.calc_id}: pred={pred_val} expected={exp_val} tol={tol}"
            )
    checks.append(CaseCheck(
        name="calculation_value_correct",
        passed=val_ok,
        detail="; ".join(val_details) if val_details else None,
    ))

    # 13. calculation_unit_correct
    unit_calcs = [c for c in label.expected_calculations if c.unit]
    if unit_calcs:
        unit_ok = True
        unit_details: list[str] = []
        for expected in unit_calcs:
            pred = _match_pred_calc(expected, pred_calcs)
            if pred is None:
                unit_ok = False
                unit_details.append(f"{expected.calc_id}: no prediction")
                continue
            pred_unit = pred.get("unit")
            if pred_unit and pred_unit != expected.unit:
                unit_ok = False
                unit_details.append(
                    f"{expected.calc_id}: pred={pred_unit} expected={expected.unit}"
                )
        checks.append(CaseCheck(
            name="calculation_unit_correct",
            passed=unit_ok,
            detail="; ".join(unit_details) if unit_details else None,
        ))

    # 14. formula_version_correct
    fv_calcs = [c for c in label.expected_calculations if c.formula_version]
    if fv_calcs:
        fv_ok = True
        fv_details: list[str] = []
        for expected in fv_calcs:
            pred = _match_pred_calc(expected, pred_calcs)
            if pred is None:
                fv_ok = False
                fv_details.append(f"{expected.calc_id}: no prediction")
                continue
            pred_fv = pred.get("formula_version")
            if pred_fv != expected.formula_version:
                fv_ok = False
                fv_details.append(
                    f"{expected.calc_id}: pred={pred_fv} expected={expected.formula_version}"
                )
        checks.append(CaseCheck(
            name="formula_version_correct",
            passed=fv_ok,
            detail="; ".join(fv_details) if fv_details else None,
        ))


def _extract_number_strings(text: str) -> list[str]:
    return [_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text)]
