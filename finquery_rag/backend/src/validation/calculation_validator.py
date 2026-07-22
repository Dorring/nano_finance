"""Validation of calculation results against extracted claims.

The ``CalculationValidator`` checks that the answer numeric claims are
consistent with the deterministic ``CalculationResult`` produced by the
calculation engine.

Calculation mismatches always produce ``CRITICAL`` issues because a wrong
calculated value must block the response -- the deterministic renderer is
the source of truth and any deviation is a hard error.

For ``EXECUTED`` results the validator verifies that:
    1. The answer includes the calculated value for the target metric
       (``CALCULATION_VALUE_MISSING``).
    2. The reported value matches within tolerance, accounting for common
       scale conversions (``CALCULATION_VALUE_MISMATCH``).
    3. No conflicting units are mentioned (``CALCULATION_UNIT_MISMATCH``).
    4. The public payload formula_version is consistent
       (``FORMULA_VERSION_MISMATCH``).
    5. The public payload operand count is consistent
       (``OPERAND_COUNT_MISMATCH``).
    6. All operands have evidence provenance
       (``OPERAND_PROVENANCE_MISSING``).
    7. No extra wrong numeric claims exist for the same metric
       (``CALCULATION_EXTRA_NUMERIC_CLAIM``).
    8. The public payload value is consistent
       (``CALCULATION_PAYLOAD_MISMATCH``).

For ``BLOCKED`` / ``FAILED`` results the validator verifies that the
public payload does not falsely report an ``EXECUTED`` status (and vice
versa) via ``CALCULATION_STATUS_MISMATCH``.

Layer dependency: ``domain <- validation``. Imports only from
``src.domain`` and stdlib.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from src.domain.calculation import (
    CalculationResult,
    CalculationStatus,
)
from src.domain.validation import (
    ExtractedClaim,
    ValidationIssue,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

CODE_CALCULATION_VALUE_MISMATCH = "CALCULATION_VALUE_MISMATCH"
CODE_CALCULATION_VALUE_MISSING = "CALCULATION_VALUE_MISSING"
CODE_CALCULATION_UNIT_MISMATCH = "CALCULATION_UNIT_MISMATCH"
CODE_FORMULA_VERSION_MISMATCH = "FORMULA_VERSION_MISMATCH"
CODE_OPERAND_COUNT_MISMATCH = "OPERAND_COUNT_MISMATCH"
CODE_OPERAND_PROVENANCE_MISSING = "OPERAND_PROVENANCE_MISSING"
CODE_CALCULATION_STATUS_MISMATCH = "CALCULATION_STATUS_MISMATCH"
CODE_CALCULATION_PAYLOAD_MISMATCH = "CALCULATION_PAYLOAD_MISMATCH"
CODE_CALCULATION_EXTRA_NUMERIC_CLAIM = "CALCULATION_EXTRA_NUMERIC_CLAIM"
CODE_CALCULATION_MISMATCH = "CALCULATION_MISMATCH"  # compat alias


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tolerance for numeric value comparisons.
_TOLERANCE = Decimal("0.01")

#: Scale factors tried by ``_values_match`` to handle common financial
#: unit conversions:
#:   1            -- same unit
#:   100          -- percent <-> ratio
#:   10000        -- wan (ten-thousand)
#:   100000000    -- yi (hundred-million)
_SCALE_FACTORS: tuple[Decimal, ...] = (
    Decimal(1),
    Decimal(100),
    Decimal(10000),
    Decimal(100000000),
)

#: Unit pattern -> group mapping, ordered by length (descending) so that
#: longer patterns are matched before their substrings.  Units within
#: the same group do not conflict with each other.
_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("\u4ebf\u7f8e\u5143", "usd"),
    ("\u4e07\u7f8e\u5143", "usd"),
    ("\u4e07\u4ebf\u5143", "cny"),
    ("\u4eba\u6c11\u5e01", "cny"),
    ("\u4ebf\u5143", "cny"),
    ("\u4e07\u5143", "cny"),
    ("\u7f8e\u5143", "usd"),
    ("\u767e\u5206\u6bd4", "proportion"),
    ("percent", "proportion"),
    ("ratio", "proportion"),
    ("CNY", "cny"),
    ("USD", "usd"),
    ("RMB", "cny"),
    ("\u5143", "cny"),
    ("%", "proportion"),
    ("\u500d", "multiple"),
)

#: Claim types that carry a numeric ``value``.
_NUMERIC_CLAIM_TYPES = frozenset({"amount", "percent", "ratio"})


class CalculationValidator:
    """Validate answer claims against a deterministic calculation result.

    All issues returned by this validator are ``CRITICAL`` -- calculation
    mismatches always block the response.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        calculation_result: CalculationResult | None,
        public_payload: dict | None = None,
    ) -> tuple[ValidationIssue, ...]:
        """Validate extracted claims against a calculation result.

        Args:
            claims: Claims extracted from the generated answer.
            calculation_result: The deterministic calculation result, or
                ``None`` if no calculation was performed.
            public_payload: The public-facing payload (typically the
                output of ``CalculationResult.to_public_dict()``).  When
                ``None``, payload-specific checks are skipped.

        Returns:
            A tuple of ``ValidationIssue`` objects.  Empty tuple means
            no issues were found.
        """
        if calculation_result is None:
            return ()

        issues: list[ValidationIssue] = []
        status = calculation_result.status

        # The status-mismatch check applies to all statuses whenever a
        # public payload is present (covers both directions: executed
        # shown as blocked/failed, and vice versa).
        if public_payload is not None:
            self._check_status_mismatch(issues, calculation_result, public_payload)

        if status is CalculationStatus.EXECUTED:
            self._check_executed(issues, claims, calculation_result, public_payload)
        # For BLOCKED / FAILED only the status-mismatch check applies
        # (already done above).  NOT_APPLICABLE and READY produce no
        # calculation-specific issues.

        return tuple(issues)

    # ------------------------------------------------------------------
    # EXECUTED checks
    # ------------------------------------------------------------------

    def _check_executed(
        self,
        issues: list[ValidationIssue],
        claims: tuple[ExtractedClaim, ...],
        result: CalculationResult,
        public_payload: dict | None,
    ) -> None:
        # 1, 2, 7 -- value consistency (missing / mismatch / extra claim)
        self._check_value_consistency(issues, claims, result)

        # 3 -- unit mismatch
        self._check_unit_mismatch(issues, claims, result)

        # 6 -- operand provenance
        self._check_operand_provenance(issues, result)

        # Payload-specific checks (4, 5, 8)
        if public_payload is not None:
            self._check_formula_version(issues, result, public_payload)
            self._check_operand_count(issues, result, public_payload)
            self._check_payload_value(issues, result, public_payload)

    def _check_value_consistency(
        self,
        issues: list[ValidationIssue],
        claims: tuple[ExtractedClaim, ...],
        result: CalculationResult,
    ) -> None:
        """Checks 1 (missing), 2 (mismatch), and 7 (extra numeric claim)."""
        if result.value is None:
            return

        target_metric = result.target_metric
        if not target_metric:
            return

        # Collect numeric claims matching the target metric.
        metric_claims = [
            c
            for c in claims
            if c.metric == target_metric
            and c.value is not None
            and c.claim_type in _NUMERIC_CLAIM_TYPES
        ]

        # 1. CALCULATION_VALUE_MISSING -- the deterministic renderer MUST
        #    include the result.
        if not metric_claims:
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_VALUE_MISSING,
                    f"Calculation result value '{result.value}' for metric "
                    f"'{target_metric}' is not mentioned in the answer.",
                )
            )
            return

        result_value = self._normalize_value(result.value)
        if result_value is None:
            return

        matching: list[ExtractedClaim] = []
        wrong: list[ExtractedClaim] = []
        for claim in metric_claims:
            claim_value = self._normalize_value(claim.value)
            if claim_value is not None and self._values_match(
                result_value, claim_value
            ):
                matching.append(claim)
            else:
                wrong.append(claim)

        # 2. CALCULATION_VALUE_MISMATCH -- all claims have wrong values.
        if wrong and not matching:
            claim = wrong[0]
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_VALUE_MISMATCH,
                    f"Answer value '{claim.value}' for metric "
                    f"'{target_metric}' does not match calculation "
                    f"result '{result.value}'.",
                    claim_text=claim.raw_text,
                )
            )

        # 7. CALCULATION_EXTRA_NUMERIC_CLAIM -- some match, some don't
        #    (extra wrong numbers for the same metric).
        if matching and wrong:
            claim = wrong[0]
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_EXTRA_NUMERIC_CLAIM,
                    f"Answer contains extra numeric claim '{claim.value}' "
                    f"for metric '{target_metric}' that differs from the "
                    f"calculated value '{result.value}'.",
                    claim_text=claim.raw_text,
                )
            )

    def _check_unit_mismatch(
        self,
        issues: list[ValidationIssue],
        claims: tuple[ExtractedClaim, ...],
        result: CalculationResult,
    ) -> None:
        """Check 3 -- answer mentions a unit conflicting with the result."""
        if not result.unit:
            return

        result_group = self._unit_group(result.unit)
        if result_group is None:
            return

        for claim in claims:
            if not claim.raw_text:
                continue
            mentioned_groups = self._find_unit_groups(claim.raw_text)
            conflicting = mentioned_groups - {result_group}
            if conflicting:
                for pattern, group in _UNIT_PATTERNS:
                    if group in conflicting and pattern in claim.raw_text:
                        issues.append(
                            self._make_issue(
                                CODE_CALCULATION_UNIT_MISMATCH,
                                f"Answer mentions unit '{pattern}' which "
                                f"conflicts with calculation unit "
                                f"'{result.unit}'.",
                                claim_text=claim.raw_text,
                            )
                        )
                        return

    def _check_operand_provenance(
        self,
        issues: list[ValidationIssue],
        result: CalculationResult,
    ) -> None:
        """Check 6 -- every operand must have ``evidence_chunk_id``."""
        for operand in result.operands:
            evidence_id = getattr(operand, "evidence_chunk_id", None)
            if not evidence_id:
                issues.append(
                    self._make_issue(
                        CODE_OPERAND_PROVENANCE_MISSING,
                        "Calculation operand is missing evidence_chunk_id "
                        "(provenance).",
                    )
                )
                return

    def _check_formula_version(
        self,
        issues: list[ValidationIssue],
        result: CalculationResult,
        public_payload: dict,
    ) -> None:
        """Check 4 -- public payload formula_version must match."""
        if "formula_version" not in public_payload:
            return

        payload_version = public_payload["formula_version"]
        result_version = result.formula_version

        if payload_version != result_version:
            issues.append(
                self._make_issue(
                    CODE_FORMULA_VERSION_MISMATCH,
                    f"Public payload formula_version '{payload_version}' "
                    f"differs from calculation result formula_version "
                    f"'{result_version}'.",
                )
            )

    def _check_operand_count(
        self,
        issues: list[ValidationIssue],
        result: CalculationResult,
        public_payload: dict,
    ) -> None:
        """Check 5 -- public payload operand count must match."""
        if "operands" not in public_payload:
            return

        payload_count = len(public_payload["operands"])
        result_count = len(result.operands)

        if payload_count != result_count:
            issues.append(
                self._make_issue(
                    CODE_OPERAND_COUNT_MISMATCH,
                    f"Public payload has {payload_count} operands but "
                    f"calculation result has {result_count}.",
                )
            )

    def _check_payload_value(
        self,
        issues: list[ValidationIssue],
        result: CalculationResult,
        public_payload: dict,
    ) -> None:
        """Check 8 -- public payload value must match (strict, no scaling)."""
        if "value" not in public_payload:
            return

        result_value = self._normalize_value(result.value)
        payload_value = self._normalize_value(public_payload["value"])

        if result_value is None and payload_value is None:
            return

        if result_value is None or payload_value is None:
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_PAYLOAD_MISMATCH,
                    f"Public payload value '{public_payload['value']}' "
                    f"differs from calculation result value "
                    f"'{result.value}'.",
                )
            )
            return

        if abs(result_value - payload_value) > _TOLERANCE:
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_PAYLOAD_MISMATCH,
                    f"Public payload value '{payload_value}' differs from "
                    f"calculation result value '{result_value}'.",
                )
            )

    # ------------------------------------------------------------------
    # Status mismatch check (applies to all statuses)
    # ------------------------------------------------------------------

    def _check_status_mismatch(
        self,
        issues: list[ValidationIssue],
        result: CalculationResult,
        public_payload: dict,
    ) -> None:
        """Check 9 -- public payload status must match result status.

        Fires when one side is ``EXECUTED`` and the other is ``BLOCKED``
        or ``FAILED`` (in either direction).  Other status combinations
        (e.g. ``BLOCKED`` vs ``FAILED``) are not flagged because both
        represent "no result produced" and the distinction is not
        user-visible.
        """
        payload_status = public_payload.get("status")
        if payload_status is None:
            return

        result_status = result.status.value
        payload_status_str = str(payload_status)

        executed = CalculationStatus.EXECUTED.value
        no_result_statuses = {
            CalculationStatus.BLOCKED.value,
            CalculationStatus.FAILED.value,
        }

        result_executed = result_status == executed
        payload_executed = payload_status_str == executed
        result_no_result = result_status in no_result_statuses
        payload_no_result = payload_status_str in no_result_statuses

        if (result_executed and payload_no_result) or (
            payload_executed and result_no_result
        ):
            issues.append(
                self._make_issue(
                    CODE_CALCULATION_STATUS_MISMATCH,
                    f"Public payload status '{payload_status_str}' "
                    f"differs from calculation result status "
                    f"'{result_status}'.",
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_value(self, value: Any) -> Decimal | None:
        """Normalize a numeric value to ``Decimal``.

        Accepts ``Decimal``, ``int``, ``float``, and numeric strings.
        Returns ``None`` if the value cannot be normalized.
        """
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return Decimal(text)
            except InvalidOperation:
                return None
        return None

    def _values_match(self, result_value: Decimal, claim_value: Decimal) -> bool:
        """Check if two values match within tolerance.

        Tries common scale conversions (percent <-> ratio, wan, yi) by
        multiplying and dividing the claim value by each scale factor.
        """
        for factor in _SCALE_FACTORS:
            scaled = claim_value * factor
            if abs(result_value - scaled) <= _TOLERANCE:
                return True
        for factor in _SCALE_FACTORS:
            if factor == 0:
                continue
            scaled = claim_value / factor
            if abs(result_value - scaled) <= _TOLERANCE:
                return True
        return False

    def _unit_group(self, unit: str) -> str | None:
        """Return the unit group for a unit string, or ``None``."""
        if not unit:
            return None
        for pattern, group in _UNIT_PATTERNS:
            if pattern in unit:
                return group
        return None

    def _find_unit_groups(self, text: str) -> set[str]:
        """Return the set of unit groups mentioned in ``text``."""
        groups: set[str] = set()
        for pattern, group in _UNIT_PATTERNS:
            if pattern in text:
                groups.add(group)
        return groups

    def _make_issue(
        self,
        code: str,
        message: str,
        claim_text: str = "",
    ) -> ValidationIssue:
        """Construct a ``CRITICAL`` ``ValidationIssue``."""
        return ValidationIssue(
            code=code,
            severity=ValidationSeverity.CRITICAL,
            message=message,
            claim_text=claim_text,
            evidence_ids=(),
            public_message="",
        )
