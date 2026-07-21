"""Calculation result consistency validation (Phase 4 Commit 6).

The ``CalculationValidator`` checks that numeric values in the generated
answer are consistent with the deterministic ``CalculationResult`` from
Phase 3. If the LLM mentions a different value for the same metric that
was calculated, the answer is flagged with ``CALCULATION_MISMATCH``.

This validator also marks numeric claims as "supported by calculation"
so the ``ResponseValidator`` (Commit 7) can suppress ``NUMERIC_UNGROUND``
issues for calculated values.

The validator is deterministic — no LLM, no retrieval.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

from decimal import Decimal

from src.domain.calculation import CalculationResult, CalculationStatus
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity


# Issue code.
CODE_CALCULATION_MISMATCH = "CALCULATION_MISMATCH"

# Tolerance for decimal comparison (handles rounding differences).
_TOLERANCE = Decimal("0.01")


class CalculationValidator:
    """Validates that answer values match the deterministic calculation.

    When a ``CalculationResult`` is EXECUTED, the validator finds numeric
    claims in the answer that reference the same target metric and checks
    that their values match the calculated value (with tolerance for
    format differences like ratio vs. percentage).
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        calculation_result: CalculationResult | None,
    ) -> tuple[ValidationIssue, ...]:
        """Validate that calculated values match the answer.

        Returns a tuple of ``ValidationIssue`` objects. If no calculation
        was executed, returns an empty tuple.
        """
        if calculation_result is None:
            return ()

        if calculation_result.status is not CalculationStatus.EXECUTED:
            return ()

        if calculation_result.value is None or calculation_result.target_metric is None:
            return ()

        calc_value = calculation_result.value
        calc_unit = calculation_result.unit
        target_metric = calculation_result.target_metric

        # Find numeric claims with the same metric.
        matching_claims = tuple(
            c for c in claims
            if c.claim_type in ("amount", "percent", "ratio")
            and c.metric is not None
            and c.metric == target_metric
        )

        if not matching_claims:
            # The answer doesn't mention the calculated metric — this is
            # not necessarily an error (the LLM may have rephrased).
            return ()

        issues: list[ValidationIssue] = []
        for claim in matching_claims:
            if claim.value is None:
                continue
            if not self._values_match(claim.value, claim.claim_type, calc_value, calc_unit):
                issues.append(
                    ValidationIssue(
                        code=CODE_CALCULATION_MISMATCH,
                        severity=ValidationSeverity.CRITICAL,
                        message=(
                            f"Answer value {claim.value} (type: {claim.claim_type}) "
                            f"for metric '{target_metric}' does not match "
                            f"calculated value {calc_value} (unit: {calc_unit})."
                        ),
                        claim_text=claim.raw_text,
                        evidence_ids=(),
                        public_message=(
                            "A calculated value in the answer does not match "
                            "the deterministic computation result."
                        ),
                    )
                )

        return tuple(issues)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _values_match(
        claim_value: Decimal,
        claim_type: str,
        calc_value: Decimal,
        calc_unit: str | None,
    ) -> bool:
        """Check if a claim value matches the calculated value.

        Handles format differences:
        - ``ratio`` (0.4) vs ``percent`` (40) — normalize to the same scale.
        - Rounding differences — use a tolerance of 0.01.
        """
        # Normalize both values to the same representation.
        claim_normalized = CalculationValidator._normalize_value(claim_value, claim_type)
        calc_normalized = CalculationValidator._normalize_value(
            calc_value, "ratio" if calc_unit == "ratio" else "base"
        )

        # If the calculation unit is "ratio", the calc value is already
        # in ratio form (e.g., 0.4). If the claim is a percentage (40%),
        # we need to divide by 100 before comparing.
        if calc_unit == "ratio" and claim_type == "percent":
            claim_normalized = claim_normalized / Decimal("100")
        elif calc_unit == "ratio" and claim_type == "ratio":
            pass  # both are ratios
        elif claim_type == "percent" and calc_unit != "ratio":
            # Calc is a plain number, claim is a percentage.
            pass  # compare directly
        elif claim_type == "ratio" and calc_unit != "ratio":
            # Calc is a plain number, claim is a ratio.
            pass  # compare directly

        diff = abs(claim_normalized - calc_normalized)
        return diff <= _TOLERANCE

    @staticmethod
    def _normalize_value(value: Decimal, claim_type: str) -> Decimal:
        """Normalize a value for comparison.

        Percentages are kept as-is (e.g., 40 for 40%). Ratios are kept
        as-is (e.g., 0.4). The normalization only strips trailing zeros.
        """
        return value.normalize() if value == value.to_integral_value() else value
