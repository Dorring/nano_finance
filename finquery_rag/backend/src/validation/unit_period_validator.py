"""Unit and period consistency validator (Phase 4 Commit 5).

The ``UnitPeriodValidator`` checks that the units, scales, currencies, and
periods mentioned in the generated answer are consistent with the evidence.
Unlike the ``NumericClaimValidator`` which checks if a *value* is grounded,
this validator checks if the *context* (unit, period) matches.

Checks performed:
- **Period mismatch**: if the answer mentions "FY2025" but the evidence
  only discusses "FY2024", flag a ``PERIOD_MISMATCH`` issue.
- **Currency mismatch**: if the answer mentions "$" (USD) but the evidence
  uses "¥" (CNY), flag a ``CURRENCY_MISMATCH`` issue.

This validator is deterministic — no LLM, no retrieval.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

import re

from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity
from src.validation.validation_policy import ValidationPolicy


# Issue codes.
CODE_PERIOD_MISMATCH = "PERIOD_MISMATCH"
CODE_CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
CODE_UNIT_MISMATCH = "UNIT_MISMATCH"


class UnitPeriodValidator:
    """Validates unit, scale, currency, and period consistency.

    The validator inspects extracted claims and compares their
    periods/currencies against what appears in the evidence text.
    Mismatches are flagged as ERROR (blocking for strict intents).
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
    ) -> tuple[ValidationIssue, ...]:
        """Validate unit and period consistency against the evidence set.

        Returns a tuple of ``ValidationIssue`` objects. If both
        ``validate_units`` and ``validate_periods`` are False, returns
        an empty tuple.
        """
        if not policy.validate_units and not policy.validate_periods:
            return ()

        if not evidence or not claims:
            return ()

        issues: list[ValidationIssue] = []

        if policy.validate_periods:
            issues.extend(self._validate_periods(claims, evidence))

        if policy.validate_units:
            issues.extend(self._validate_currencies(claims, evidence))

        return tuple(issues)

    # -----------------------------------------------------------------
    # Period validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_periods(
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> list[ValidationIssue]:
        """Check that period claims are grounded in evidence.

        A period claim is grounded if the evidence text contains the same
        year or period reference. If the evidence discusses a *different*
        year, we flag a mismatch.
        """
        period_claims = tuple(c for c in claims if c.claim_type == "period")
        if not period_claims:
            return []

        # Extract all years mentioned in the evidence.
        evidence_years = UnitPeriodValidator._extract_years_from_evidence(evidence)
        if not evidence_years:
            # If evidence has no years at all, we cannot flag a mismatch;
            # the period claim is simply ungrounded (handled by numeric
            # validator if it's a numeric claim).
            return []

        issues: list[ValidationIssue] = []
        for claim in period_claims:
            claim_years = UnitPeriodValidator._extract_years_from_text(
                claim.period or claim.raw_text
            )
            if not claim_years:
                continue

            # If none of the claim's years appear in the evidence, flag it.
            if not claim_years.intersection(evidence_years):
                issues.append(
                    ValidationIssue(
                        code=CODE_PERIOD_MISMATCH,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Period '{claim.raw_text}' (years: {sorted(claim_years)}) "
                            f"not found in evidence (years: {sorted(evidence_years)})."
                        ),
                        claim_text=claim.raw_text,
                        evidence_ids=tuple(e.chunk_id for e in evidence),
                        public_message=(
                            "A time period in the answer does not match "
                            "the source documents."
                        ),
                    )
                )

        return issues

    # -----------------------------------------------------------------
    # Currency validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_currencies(
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> list[ValidationIssue]:
        """Check that currency claims match the evidence.

        If the answer mentions a specific currency (e.g., USD via "$") but
        the evidence only mentions a different currency (e.g., CNY via "¥"),
        flag a mismatch. If the evidence mentions multiple currencies or no
        currencies, no mismatch is flagged (to avoid false positives).
        """
        currency_claims = tuple(
            c for c in claims if c.claim_type == "amount" and c.currency is not None
        )
        if not currency_claims:
            return []

        evidence_currencies = UnitPeriodValidator._extract_currencies_from_evidence(evidence)
        if not evidence_currencies or len(evidence_currencies) > 1:
            # No currencies or mixed currencies in evidence — can't flag.
            return []

        evidence_currency = evidence_currencies.pop()
        issues: list[ValidationIssue] = []
        for claim in currency_claims:
            if claim.currency and claim.currency != evidence_currency:
                issues.append(
                    ValidationIssue(
                        code=CODE_CURRENCY_MISMATCH,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Currency '{claim.currency}' in claim "
                            f"'{claim.raw_text}' does not match evidence "
                            f"currency '{evidence_currency}'."
                        ),
                        claim_text=claim.raw_text,
                        evidence_ids=tuple(e.chunk_id for e in evidence),
                        public_message=(
                            "A currency in the answer does not match "
                            "the source documents."
                        ),
                    )
                )

        return issues

    # -----------------------------------------------------------------
    # Extraction helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_years_from_text(text: str) -> set[str]:
        """Extract 4-digit years from text."""
        return set(re.findall(r"\b(20[0-2]\d|19[89]\d)\b", text))

    @staticmethod
    def _extract_years_from_evidence(
        evidence: tuple[EvidenceItem, ...],
    ) -> set[str]:
        """Extract all years mentioned in the evidence set."""
        years: set[str] = set()
        for item in evidence:
            years.update(UnitPeriodValidator._extract_years_from_text(item.content or ""))
        return years

    @staticmethod
    def _extract_currencies_from_evidence(
        evidence: tuple[EvidenceItem, ...],
    ) -> set[str]:
        """Extract currency symbols from the evidence set."""
        _SYMBOL_TO_CODE = {"$": "USD", "¥": "CNY", "€": "EUR", "£": "GBP"}
        currencies: set[str] = set()
        for item in evidence:
            text = item.content or ""
            for symbol, code in _SYMBOL_TO_CODE.items():
                if symbol in text:
                    currencies.add(code)
        return currencies
