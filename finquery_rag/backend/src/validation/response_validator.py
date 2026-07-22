"""Grounded response validation aggregator (Phase 4 Commit 7).

The ``ResponseValidator`` is the central post-generation validator that
orchestrates all individual validators (numeric, unit/period, citation,
calculation, unsupported claim) and produces a single ``ValidationResult``.

The verdict is determined by the highest-severity issue:
- Any CRITICAL issue -> ``BLOCKED`` (answer must NOT be returned).
- Any ERROR issue -> ``BLOCKED`` (strict paths) or ``REPAIRABLE`` (lenient).
- Only WARNING/INFO -> ``PASSED``.
- No issues -> ``PASSED``.
- Validator raises an exception -> ``FAILED`` (fail-closed; answer must
  NOT be returned).

Claims supported by a deterministic calculation (``CalculationResult``
EXECUTED) are exempt from ``NUMERIC_UNGROUND`` issues — the calculation
IS the evidence.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""

from __future__ import annotations

from src.domain.calculation import CalculationResult, CalculationStatus
from src.domain.evidence import EvidenceItem
from src.domain.validation import (
    ExtractedClaim,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
)
from src.validation.calculation_validator import CalculationValidator
from src.validation.citation_validator import CitationValidator
from src.validation.claim_extractor import ClaimExtractor
from src.validation.numeric_claim_validator import (
    CODE_NUMERIC_UNGROUND,
    NumericClaimValidator,
)
from src.validation.unsupported_claim_validator import UnsupportedClaimValidator
from src.validation.unit_period_validator import UnitPeriodValidator
from src.validation.validation_policy import get_policy_for_intent


class ResponseValidator:
    """Aggregates all post-generation validators into a single verdict.

    The validator is deterministic and never calls the LLM. It runs the
    following validators in order:

    1. ``ClaimExtractor`` — extract claims from the answer.
    2. ``CalculationValidator`` — check calculation consistency.
    3. ``NumericClaimValidator`` — check numeric grounding (suppressed
       for claims supported by calculation).
    4. ``UnitPeriodValidator`` — check unit/period/currency consistency.
    5. ``CitationValidator`` — check citation presence and resolvability.
    6. ``UnsupportedClaimValidator`` — flag high-confidence unsupported
       claims.
    """

    def __init__(self) -> None:
        self._extractor = ClaimExtractor()
        self._calc_validator = CalculationValidator()
        self._numeric_validator = NumericClaimValidator()
        self._unit_period_validator = UnitPeriodValidator()
        self._citation_validator = CitationValidator()
        self._unsupported_validator = UnsupportedClaimValidator()

    def validate(
        self,
        *,
        answer: str,
        intent: str,
        evidence: tuple[EvidenceItem, ...],
        calculation_result: CalculationResult | None,
        sources: tuple[dict, ...] = (),
    ) -> ValidationResult:
        """Validate a generated answer against evidence and calculations.

        Returns a ``ValidationResult``. Never raises — if an internal
        error occurs, returns ``ValidationStatus.FAILED`` with a single
        sanitized CRITICAL issue (fail-closed).

        ``sources`` is the tuple of source objects returned to the API
        consumer. The CitationValidator uses it to verify chunk_id,
        document_name, and page consistency.
        """
        try:
            return self._validate_inner(
                answer=answer,
                intent=intent,
                evidence=evidence,
                calculation_result=calculation_result,
                sources=sources,
            )
        except Exception:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                issues=(
                    ValidationIssue(
                        code="VALIDATOR_ERROR",
                        severity=ValidationSeverity.CRITICAL,
                        message="Response validator encountered an internal error.",
                        public_message=(
                            "The answer could not be verified due to an "
                            "internal validation error."
                        ),
                    ),
                ),
            )

    # -----------------------------------------------------------------
    # Internal validation logic
    # -----------------------------------------------------------------

    def _validate_inner(
        self,
        *,
        answer: str,
        intent: str,
        evidence: tuple[EvidenceItem, ...],
        calculation_result: CalculationResult | None,
        sources: tuple[dict, ...] = (),
    ) -> ValidationResult:
        policy = get_policy_for_intent(intent)

        # Conversation / unsupported intents: no validation.
        if not policy.applies_any_validation:
            return ValidationResult(status=ValidationStatus.NOT_APPLICABLE)

        # Extract claims.
        claims = self._extractor.extract(answer)

        # Run calculation validator first (to identify supported claims).
        calc_issues = self._calc_validator.validate(claims, calculation_result)

        # Identify claims supported by calculation (to suppress NUMERIC_UNGROUND).
        calc_supported_claim_ids = self._calc_supported_claim_ids(
            claims, calculation_result
        )

        # Run numeric validator.
        numeric_issues = self._numeric_validator.validate(claims, evidence, policy)

        # Suppress NUMERIC_UNGROUND for claims supported by calculation.
        numeric_issues = tuple(
            i
            for i in numeric_issues
            if not (
                i.code == CODE_NUMERIC_UNGROUND
                and i.claim_text in calc_supported_claim_ids
            )
        )

        # Run unit/period validator.
        unit_period_issues = self._unit_period_validator.validate(
            claims, evidence, policy
        )

        # Run citation validator (with sources for chunk/page validation).
        citation_issues = self._citation_validator.validate(
            claims, evidence, policy, sources
        )

        # Run unsupported claim validator.
        unsupported_issues = self._unsupported_validator.validate(
            claims, evidence, calculation_result, policy
        )

        # Aggregate all issues.
        all_issues = (
            calc_issues
            + numeric_issues
            + unit_period_issues
            + citation_issues
            + unsupported_issues
        )

        # Count claims.
        numeric_claim_count = sum(
            1 for c in claims if c.claim_type in ("amount", "percent", "ratio")
        )
        supported_count = numeric_claim_count - sum(
            1 for i in numeric_issues if i.code == CODE_NUMERIC_UNGROUND
        )
        unsupported_count = numeric_claim_count - supported_count

        # Determine status.
        status = self._determine_status(all_issues, policy)

        return ValidationResult(
            status=status,
            issues=all_issues,
            checked_claim_count=numeric_claim_count,
            supported_claim_count=max(0, supported_count),
            unsupported_claim_count=max(0, unsupported_count),
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _calc_supported_claim_ids(
        claims: tuple[ExtractedClaim, ...],
        calculation_result: CalculationResult | None,
    ) -> set[str]:
        """Return the raw_text of claims supported by calculation.

        A claim is "supported by calculation" if:
        - The calculation is EXECUTED with a value and target_metric.
        - The claim's metric matches the calculation's target_metric.
        - The claim's value matches the calculation's value (with tolerance).
        """
        if calculation_result is None:
            return set()
        if calculation_result.status is not CalculationStatus.EXECUTED:
            return set()
        if calculation_result.value is None or calculation_result.target_metric is None:
            return set()

        calc_value = calculation_result.value
        target_metric = calculation_result.target_metric

        supported: set[str] = set()
        for claim in claims:
            if claim.claim_type not in ("amount", "percent", "ratio"):
                continue
            if claim.metric is None or claim.metric != target_metric:
                continue
            if claim.value is None:
                continue
            if CalculationValidator()._values_match(calc_value, claim.value):
                supported.add(claim.raw_text)

        return supported

    @staticmethod
    def _determine_status(
        issues: tuple[ValidationIssue, ...],
        policy,
    ) -> ValidationStatus:
        """Determine the validation status from the issues.

        - Any CRITICAL -> BLOCKED.
        - Any ERROR -> BLOCKED (strict) or REPAIRABLE (lenient).
        - Only WARNING/INFO -> PASSED.
        - No issues -> PASSED.
        """
        if not issues:
            return ValidationStatus.PASSED

        has_critical = any(i.severity is ValidationSeverity.CRITICAL for i in issues)
        has_error = any(i.severity is ValidationSeverity.ERROR for i in issues)

        if has_critical:
            return ValidationStatus.BLOCKED
        if has_error:
            if policy.strict_numeric_grounding:
                return ValidationStatus.BLOCKED
            return ValidationStatus.REPAIRABLE

        return ValidationStatus.PASSED
