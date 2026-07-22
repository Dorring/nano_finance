"""Comprehensive tests for the rewritten CalculationValidator.

Tests all 16 scenarios covering the 9 error codes and edge cases.

Run with: pytest test_calculation_validator_full.py -v
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from src.domain.calculation import CalculationStatus
from src.domain.validation import ValidationSeverity
from src.validation.calculation_validator import (
    CalculationValidator,
    CODE_CALCULATION_VALUE_MISSING,
    CODE_CALCULATION_VALUE_MISMATCH,
    CODE_CALCULATION_UNIT_MISMATCH,
    CODE_FORMULA_VERSION_MISMATCH,
    CODE_OPERAND_COUNT_MISMATCH,
    CODE_OPERAND_PROVENANCE_MISSING,
    CODE_CALCULATION_STATUS_MISMATCH,
    CODE_CALCULATION_PAYLOAD_MISMATCH,
    CODE_CALCULATION_EXTRA_NUMERIC_CLAIM,
    CODE_CALCULATION_MISMATCH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_claim(
    metric: str = "revenue",
    value: str | None = "100",
    claim_type: str = "amount",
    raw_text: str = "",
) -> SimpleNamespace:
    """Build an ExtractedClaim-like object for testing."""
    return SimpleNamespace(
        metric=metric,
        value=value,
        claim_type=claim_type,
        raw_text=raw_text,
    )


def make_operand(evidence_chunk_id: str | None = "chunk_001") -> SimpleNamespace:
    """Build an operand-like object for testing."""
    return SimpleNamespace(evidence_chunk_id=evidence_chunk_id)


def make_result(
    status: CalculationStatus = CalculationStatus.EXECUTED,
    value: Decimal | str | None = Decimal("100"),
    target_metric: str = "revenue",
    unit: str = "\u5143",
    operands: tuple | None = None,
    formula_version: str = "v1",
    error_code: str | None = None,
) -> SimpleNamespace:
    """Build a CalculationResult-like object for testing."""
    if operands is None:
        operands = (make_operand(),)
    return SimpleNamespace(
        status=status,
        value=value,
        target_metric=target_metric,
        unit=unit,
        operands=operands,
        formula_version=formula_version,
        error_code=error_code,
    )


def make_payload(
    status: str = "executed",
    value: Decimal | str | None = Decimal("100"),
    operands: list | None = None,
    formula_version: str = "v1",
) -> dict:
    """Build a public_payload dict for testing."""
    if operands is None:
        operands = [{"value": 100, "evidence_chunk_id": "chunk_001"}]
    return {
        "status": status,
        "value": value,
        "operands": operands,
        "formula_version": formula_version,
    }


def codes(issues) -> set[str]:
    """Extract issue codes from a tuple of ValidationIssue."""
    return {i.code for i in issues}


def all_critical(issues) -> bool:
    """Check that all issues have CRITICAL severity."""
    return all(i.severity is ValidationSeverity.CRITICAL for i in issues)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


class TestCalculationValidator:
    """All 16 test scenarios for CalculationValidator."""

    def setup_method(self):
        self.validator = CalculationValidator()

    # 1. Correct result -> no issues
    def test_01_correct_result_no_issues(self):
        result = make_result(value=Decimal("100"))
        claims = (make_claim(value="100", raw_text="Revenue is 100\u5143"),)
        payload = make_payload(value=Decimal("100"))
        issues = self.validator.validate(claims, result, payload)
        assert issues == ()

    # 2. Answer missing result value -> CALCULATION_VALUE_MISSING
    def test_02_value_missing(self):
        result = make_result(value=Decimal("100"), target_metric="revenue")
        claims = (make_claim(metric="profit", value="100"),)
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_VALUE_MISSING in codes(issues)
        assert all_critical(issues)

    # 3. Answer value wrong -> CALCULATION_VALUE_MISMATCH
    def test_03_value_mismatch(self):
        result = make_result(value=Decimal("100"))
        claims = (make_claim(value="200", raw_text="Revenue is 200\u5143"),)
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_VALUE_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 4. Unit wrong -> CALCULATION_UNIT_MISMATCH
    def test_04_unit_mismatch(self):
        result = make_result(value=Decimal("100"), unit="\u5143")
        claims = (
            make_claim(
                value="100",
                raw_text="Revenue is 100\u7f8e\u5143",
            ),
        )
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_UNIT_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 5. Formula version wrong -> FORMULA_VERSION_MISMATCH
    def test_05_formula_version_mismatch(self):
        result = make_result(formula_version="v2")
        payload = make_payload(formula_version="v1")
        issues = self.validator.validate((), result, payload)
        assert CODE_FORMULA_VERSION_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 6. Operand count wrong -> OPERAND_COUNT_MISMATCH
    def test_06_operand_count_mismatch(self):
        result = make_result(operands=(make_operand(), make_operand()))
        payload = make_payload(
            operands=[{"value": 1}, {"value": 2}, {"value": 3}],
        )
        issues = self.validator.validate((), result, payload)
        assert CODE_OPERAND_COUNT_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 7. Operand missing evidence -> OPERAND_PROVENANCE_MISSING
    def test_07_operand_provenance_missing(self):
        result = make_result(
            operands=(make_operand(evidence_chunk_id=None),),
        )
        issues = self.validator.validate((), result, None)
        assert CODE_OPERAND_PROVENANCE_MISSING in codes(issues)
        assert all_critical(issues)

    # 8. Payload value wrong -> CALCULATION_PAYLOAD_MISMATCH
    def test_08_payload_value_mismatch(self):
        result = make_result(value=Decimal("100"))
        payload = make_payload(value=Decimal("200"))
        issues = self.validator.validate((), result, payload)
        assert CODE_CALCULATION_PAYLOAD_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 9. Payload status mismatch (BLOCKED shown as EXECUTED)
    #    -> CALCULATION_STATUS_MISMATCH
    def test_09_status_mismatch_blocked_as_executed(self):
        result = make_result(
            status=CalculationStatus.BLOCKED,
            value=None,
            error_code="INSUFFICIENT_DATA",
        )
        payload = make_payload(status="executed")
        issues = self.validator.validate((), result, payload)
        assert CODE_CALCULATION_STATUS_MISMATCH in codes(issues)
        assert all_critical(issues)

    # 10. Extra numeric claim for same metric
    #     -> CALCULATION_EXTRA_NUMERIC_CLAIM
    def test_10_extra_numeric_claim(self):
        result = make_result(value=Decimal("100"))
        claims = (
            make_claim(value="100", raw_text="Revenue is 100\u5143"),
            make_claim(value="200", raw_text="Revenue is 200\u5143"),
        )
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_EXTRA_NUMERIC_CLAIM in codes(issues)
        assert all_critical(issues)

    # 11. Scale conversion (yi -> raw)
    def test_11_scale_conversion_yi(self):
        result = make_result(
            value=Decimal("100000000"),
            unit="\u5143",
        )
        claims = (
            make_claim(
                value="1",
                raw_text="Revenue is 1\u4ebf\u5143",
            ),
        )
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_VALUE_MISSING not in codes(issues)
        assert CODE_CALCULATION_VALUE_MISMATCH not in codes(issues)

    # 12. Ratio vs percent conversion
    def test_12_ratio_percent_conversion(self):
        result = make_result(
            value=Decimal("50"),
            unit="%",
            target_metric="margin",
        )
        claims = (
            make_claim(
                metric="margin",
                value="0.5",
                claim_type="ratio",
                raw_text="Margin is 0.5",
            ),
        )
        issues = self.validator.validate(claims, result, None)
        assert CODE_CALCULATION_VALUE_MISMATCH not in codes(issues)
        assert CODE_CALCULATION_VALUE_MISSING not in codes(issues)

    # 13. Negative value
    def test_13_negative_value(self):
        result = make_result(
            value=Decimal("-100"),
            target_metric="loss",
        )
        claims = (
            make_claim(
                metric="loss",
                value="-100",
                raw_text="Loss is -100\u5143",
            ),
        )
        issues = self.validator.validate(claims, result, None)
        assert issues == ()

    # 14. Zero value
    def test_14_zero_value(self):
        result = make_result(
            value=Decimal("0"),
            target_metric="balance",
        )
        claims = (
            make_claim(
                metric="balance",
                value="0",
                raw_text="Balance is 0\u5143",
            ),
        )
        issues = self.validator.validate(claims, result, None)
        assert issues == ()

    # 15. No calculation result (None) -> no issues
    def test_15_no_calculation_result(self):
        issues = self.validator.validate((), None, None)
        assert issues == ()

    # 16. BLOCKED result with no payload -> no issues
    def test_16_blocked_no_payload(self):
        result = make_result(
            status=CalculationStatus.BLOCKED,
            value=None,
            error_code="INSUFFICIENT_DATA",
        )
        issues = self.validator.validate((), result, None)
        assert issues == ()


# ---------------------------------------------------------------------------
# Compat alias test
# ---------------------------------------------------------------------------


class TestCompatAlias:
    """Verify the backward-compatibility alias is defined."""

    def test_compat_alias_exists(self):
        assert CODE_CALCULATION_MISMATCH == "CALCULATION_MISMATCH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
