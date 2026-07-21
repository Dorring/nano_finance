"""Numeric claim grounding validator (Phase 4 Commit 5).

The ``NumericClaimValidator`` checks whether each numeric claim extracted
from the generated answer is grounded in the retrieved evidence. A numeric
claim is considered *grounded* if its value (in some textual form) appears
in at least one evidence chunk.

This validator does NOT use an LLM. It performs deterministic text matching
with multiple value representations (plain, comma-formatted, scale-suffixed)
to handle format variations between the LLM output and the source text.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

from decimal import Decimal

from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity
from src.validation.validation_policy import (
    ACTION_WARN,
    ValidationPolicy,
)


# Issue code for unsupported numeric claims.
CODE_NUMERIC_UNGROUND = "NUMERIC_UNGROUND"


class NumericClaimValidator:
    """Validates that numeric claims are grounded in evidence.

    The validator checks each ``amount``, ``percent``, and ``ratio`` claim
    to see if its value appears in the evidence text. Claims whose values
    cannot be found in any evidence chunk are flagged as unsupported.
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
    ) -> tuple[ValidationIssue, ...]:
        """Validate numeric claims against the evidence set.

        Returns a tuple of ``ValidationIssue`` objects. If the policy
        disables numeric claim validation, returns an empty tuple.
        """
        if not policy.validate_numeric_claims:
            return ()

        numeric_claims = tuple(
            c for c in claims if c.claim_type in ("amount", "percent", "ratio")
        )
        if not numeric_claims:
            return ()

        if not evidence:
            # No evidence means all numeric claims are unsupported.
            return tuple(
                self._build_unsupported_issue(c, policy, evidence_ids=())
                for c in numeric_claims
            )

        issues: list[ValidationIssue] = []
        for claim in numeric_claims:
            supporting_ids = self._find_supporting_evidence(claim, evidence)
            if not supporting_ids:
                issues.append(
                    self._build_unsupported_issue(
                        claim, policy, evidence_ids=supporting_ids
                    )
                )

        return tuple(issues)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _find_supporting_evidence(
        claim: ExtractedClaim,
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[str, ...]:
        """Find evidence chunk IDs that contain the claim's value.

        Returns a tuple of chunk IDs. An empty tuple means the value was
        not found in any evidence chunk.
        """
        if claim.value is None:
            return ()

        representations = NumericClaimValidator._value_representations(claim)
        supporting: list[str] = []
        for item in evidence:
            text = item.content or ""
            for rep in representations:
                if rep and rep in text:
                    supporting.append(item.chunk_id)
                    break
        return tuple(supporting)

    @staticmethod
    def _value_representations(claim: ExtractedClaim) -> tuple[str, ...]:
        """Generate multiple string representations of the claim value.

        The LLM may format numbers differently from the source text:
        - ``1000000`` vs ``1,000,000`` vs ``1 million``
        - ``40%`` vs ``40.00%`` vs ``0.40`` (for percentages)
        """
        if claim.value is None:
            return ()

        value = claim.value
        reps: list[str] = []

        if claim.claim_type == "percent":
            # Percentage: 40% could appear as "40%", "40.00%", "40 percent"
            reps.append(f"{value}%")
            reps.append(f"{value:.2f}%")
            # Also check if it appears as a decimal ratio (0.40)
            ratio = value / Decimal("100")
            reps.append(str(ratio))
            reps.append(f"{ratio:.2f}")
            reps.append(f"{ratio:.4f}")

        elif claim.claim_type == "ratio":
            # Ratio: 0.4 could appear as "0.4", "0.40", "40%"
            reps.append(str(value))
            reps.append(f"{value:.2f}")
            reps.append(f"{value:.4f}")
            # Also as percentage
            pct = value * Decimal("100")
            reps.append(f"{pct}%")
            reps.append(f"{pct:.2f}%")

        elif claim.claim_type == "amount":
            # Amount: try plain, comma-formatted, and scale-suffixed
            reps.append(str(value))
            reps.append(f"{value:,}")
            # Try without trailing zeros
            reps.append(str(value.normalize()))
            # Try with 2 decimal places
            reps.append(f"{value:.2f}")
            # Try scale-suffixed representations
            reps.extend(NumericClaimValidator._scale_representations(value))

        # Always include the integer representation if the value is whole.
        if value == value.to_integral_value():
            int_val = int(value)
            reps.append(str(int_val))
            reps.append(f"{int_val:,}")

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for r in reps:
            if r and r not in seen:
                seen.add(r)
                unique.append(r)
        return tuple(unique)

    @staticmethod
    def _scale_representations(value: Decimal) -> tuple[str, ...]:
        """Generate scale-suffixed representations of a value.

        E.g., 1000000 -> "1 million", "1M", "1.00 million"
        """
        reps: list[str] = []
        abs_value = abs(value)

        if abs_value >= Decimal("1000000000") and abs_value % Decimal("1000000000") == 0:
            scaled = value / Decimal("1000000000")
            reps.append(f"{scaled} billion")
            reps.append(f"{scaled}B")
            reps.append(f"{scaled:.2f} billion")
        elif abs_value >= Decimal("1000000") and abs_value % Decimal("1000000") == 0:
            scaled = value / Decimal("1000000")
            reps.append(f"{scaled} million")
            reps.append(f"{scaled}M")
            reps.append(f"{scaled:.2f} million")
        elif abs_value >= Decimal("1000") and abs_value % Decimal("1000") == 0:
            scaled = value / Decimal("1000")
            reps.append(f"{scaled} thousand")
            reps.append(f"{scaled}K")
            reps.append(f"{scaled:.2f} thousand")

        return tuple(reps)

    @staticmethod
    def _build_unsupported_issue(
        claim: ExtractedClaim,
        policy: ValidationPolicy,
        evidence_ids: tuple[str, ...],
    ) -> ValidationIssue:
        """Build a ValidationIssue for an unsupported numeric claim.

        Severity depends on the policy:
        - ``strict_numeric_grounding=True`` -> CRITICAL (always blocks).
        - ``strict_numeric_grounding=False`` -> ERROR (may be repairable).
        - ``unsupported_numeric_action="warn"`` -> WARNING (non-blocking).
        """
        if policy.unsupported_numeric_action == ACTION_WARN:
            severity = ValidationSeverity.WARNING
        elif policy.strict_numeric_grounding:
            severity = ValidationSeverity.CRITICAL
        else:
            severity = ValidationSeverity.ERROR

        metric_label = claim.metric or claim.claim_type
        return ValidationIssue(
            code=CODE_NUMERIC_UNGROUND,
            severity=severity,
            message=(
                f"Numeric claim '{claim.raw_text}' (metric: {metric_label}, "
                f"value: {claim.value}) not found in any evidence chunk."
            ),
            claim_text=claim.raw_text,
            evidence_ids=evidence_ids,
            public_message=(
                f"A numeric value in the answer could not be verified "
                f"against the source documents."
            ),
        )
