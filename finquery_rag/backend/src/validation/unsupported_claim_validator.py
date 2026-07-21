"""Unsupported claim validator (Phase 4 Commit 7).

The ``UnsupportedClaimValidator`` flags numeric claims whose *metric* does
not appear in any evidence chunk at all. This is a high-confidence signal
that the claim is hallucinated — not just a value mismatch (handled by
``NumericClaimValidator``) but a metric that has no grounding whatsoever.

Claims whose metric matches the calculation result's ``target_metric`` are
exempt — they are supported by the deterministic calculation, not by raw
evidence text.

The validator is deterministic — no LLM, no retrieval.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

from src.domain.calculation import CalculationResult, CalculationStatus
from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity
from src.validation.validation_policy import ValidationPolicy


# Issue code.
CODE_UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"


class UnsupportedClaimValidator:
    """Flags numeric claims whose metric is absent from all evidence.

    This is a stronger check than ``NumericClaimValidator``: it verifies
    that the *metric name* (e.g. "revenue", "gross_margin") appears in
    the evidence text at all. If the metric is not mentioned anywhere in
    the evidence and is not the target of a calculation, the claim is
    flagged as ``UNSUPPORTED_CLAIM``.
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        calculation_result: CalculationResult | None,
        policy: ValidationPolicy,
    ) -> tuple[ValidationIssue, ...]:
        """Validate that numeric claims have metric grounding in evidence.

        Returns a tuple of ``ValidationIssue`` objects. If the policy
        disables numeric claim validation, returns an empty tuple.
        """
        if not policy.validate_numeric_claims:
            return ()

        numeric_claims = tuple(
            c for c in claims
            if c.claim_type in ("amount", "percent", "ratio")
            and c.metric is not None
        )
        if not numeric_claims:
            return ()

        # Determine the calculation target metric (if any) for exemption.
        calc_metric: str | None = None
        if (
            calculation_result is not None
            and calculation_result.status is CalculationStatus.EXECUTED
        ):
            calc_metric = calculation_result.target_metric

        # Build a set of metric keywords present in the evidence text.
        evidence_metrics = self._extract_evidence_metrics(evidence)

        issues: list[ValidationIssue] = []
        for claim in numeric_claims:
            metric = claim.metric
            if metric is None:
                continue

            # Exempt claims matching the calculation target metric.
            if calc_metric is not None and metric == calc_metric:
                continue

            # Check if the metric or its readable form appears in evidence.
            if not self._metric_in_evidence(metric, evidence_metrics):
                issues.append(
                    ValidationIssue(
                        code=CODE_UNSUPPORTED_CLAIM,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Claim '{claim.raw_text}' references metric "
                            f"'{metric}' which does not appear in any "
                            f"evidence chunk."
                        ),
                        claim_text=claim.raw_text,
                        evidence_ids=tuple(e.chunk_id for e in evidence),
                        public_message=(
                            "The answer references a financial metric "
                            "not found in the source documents."
                        ),
                    )
                )

        return tuple(issues)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_evidence_metrics(
        evidence: tuple[EvidenceItem, ...],
    ) -> set[str]:
        """Extract a set of metric-related keywords from evidence text.

        Converts common metric keys (e.g. ``gross_margin``) to their
        readable forms (e.g. ``gross margin``) and checks if they appear
        in the evidence.
        """
        from src.validation.claim_extractor import _METRIC_CANONICAL

        # Build reverse map: canonical key -> list of readable keywords.
        readable_by_key: dict[str, list[str]] = {}
        for keyword, canonical in _METRIC_CANONICAL.items():
            readable_by_key.setdefault(canonical, []).append(keyword)

        found: set[str] = set()
        for item in evidence:
            text = (item.content or "").lower()
            for canonical, keywords in readable_by_key.items():
                for kw in keywords:
                    if kw in text:
                        found.add(canonical)
                        break
        return found

    @staticmethod
    def _metric_in_evidence(metric: str, evidence_metrics: set[str]) -> bool:
        """Check if a metric key is present in the evidence metrics set."""
        return metric in evidence_metrics
