"""Citation validation (Phase 4 Commit 6).

The ``CitationValidator`` checks two things:

1. **Citation presence**: when the policy requires citations
   (``require_citations=True``), numeric claims in the answer should be
   accompanied by at least one citation reference. If no citation
   references exist in the answer at all, each numeric claim is flagged
   with ``CITATION_MISSING``.

2. **Citation resolvability**: citation references like ``[1]`` or
   ``[doc.pdf, p.12]`` should resolve to an actual evidence source. If a
   citation cannot be resolved, it is flagged with
   ``CITATION_UNRESOLVED``.

The validator is deterministic — no LLM, no retrieval.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

import re

from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity
from src.validation.validation_policy import (
    ACTION_BLOCK,
    ValidationPolicy,
)


# Issue codes.
CODE_CITATION_MISSING = "CITATION_MISSING"
CODE_CITATION_UNRESOLVED = "CITATION_UNRESOLVED"

# Regex for detecting document-style citations: [doc.pdf, p.12] or [report.pdf p5]
_RE_DOC_CITATION = re.compile(
    r"([\w.\-]+\.(?:pdf|docx?|xlsx?|txt))"  # document name
    r"[,\s]*"
    r"(?:p\.?\s*(\d+))?",                    # optional page
    re.IGNORECASE,
)


class CitationValidator:
    """Validates that citation references are present and resolvable.

    The validator inspects extracted claims and evidence to ensure that
    numeric claims are backed by citations and that citations resolve to
    real evidence sources.
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
    ) -> tuple[ValidationIssue, ...]:
        """Validate citations against the evidence set.

        Returns a tuple of ``ValidationIssue`` objects. If the policy
        does not require citations, only resolvability is checked.
        """
        issues: list[ValidationIssue] = []

        citation_claims = tuple(c for c in claims if c.claim_type == "citation_ref")
        numeric_claims = tuple(
            c for c in claims if c.claim_type in ("amount", "percent", "ratio")
        )

        # --- 1. Check citation resolvability ---
        unresolved = self._check_resolvability(citation_claims, evidence)
        issues.extend(unresolved)

        # --- 2. Check citation presence for numeric claims ---
        if policy.require_citations and numeric_claims and not citation_claims:
            for claim in numeric_claims:
                issues.append(
                    self._build_missing_issue(claim, policy, evidence)
                )

        return tuple(issues)

    # -----------------------------------------------------------------
    # Resolvability checks
    # -----------------------------------------------------------------

    @staticmethod
    def _check_resolvability(
        citation_claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> list[ValidationIssue]:
        """Check if each citation reference resolves to an evidence source.

        Returns a list of ``ValidationIssue`` objects for unresolved
        citations.
        """
        if not citation_claims or not evidence:
            return []

        evidence_docs = {item.document_name for item in evidence if item.document_name}
        issues: list[ValidationIssue] = []

        for claim in citation_claims:
            resolved = False
            for ref in claim.citation_refs:
                if CitationValidator._resolve_citation(ref, evidence, evidence_docs):
                    resolved = True
                    break

            if not resolved:
                issues.append(
                    ValidationIssue(
                        code=CODE_CITATION_UNRESOLVED,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Citation '{claim.raw_text}' could not be resolved "
                            f"to any evidence source."
                        ),
                        claim_text=claim.raw_text,
                        evidence_ids=tuple(e.chunk_id for e in evidence),
                        public_message=(
                            "A citation in the answer could not be matched "
                            "to a source document."
                        ),
                    )
                )

        return issues

    @staticmethod
    def _resolve_citation(
        ref: str,
        evidence: tuple[EvidenceItem, ...],
        evidence_docs: set[str | None],
    ) -> bool:
        """Check if a single citation reference resolves to evidence.

        - Number citations (``"1"``, ``"2"``) resolve if the number is
          within the evidence count.
        - Document citations (``"doc.pdf, p.12"``) resolve if the
          document name appears in the evidence set.
        - Other citations resolve if the reference text appears in any
          evidence content.
        """
        ref = ref.strip()
        if not ref:
            return False

        # Number citation: [1], [2], etc.
        if ref.isdigit():
            idx = int(ref)
            return 1 <= idx <= len(evidence)

        # Document-style citation: [doc.pdf, p.12]
        doc_match = _RE_DOC_CITATION.search(ref)
        if doc_match:
            doc_name = doc_match.group(1)
            if doc_name in evidence_docs:
                return True

        # Fallback: check if the reference text appears in any evidence.
        for item in evidence:
            if ref in (item.content or ""):
                return True

        # Check if any evidence document name contains the reference.
        for doc in evidence_docs:
            if doc and doc in ref:
                return True

        return False

    # -----------------------------------------------------------------
    # Missing citation issue builder
    # -----------------------------------------------------------------

    @staticmethod
    def _build_missing_issue(
        claim: ExtractedClaim,
        policy: ValidationPolicy,
        evidence: tuple[EvidenceItem, ...],
    ) -> ValidationIssue:
        """Build a CITATION_MISSING issue for a numeric claim without citations."""
        if policy.missing_citation_action == ACTION_BLOCK:
            severity = ValidationSeverity.CRITICAL
        else:
            severity = ValidationSeverity.WARNING

        return ValidationIssue(
            code=CODE_CITATION_MISSING,
            severity=severity,
            message=(
                f"Numeric claim '{claim.raw_text}' (metric: {claim.metric or 'unknown'}) "
                f"is not accompanied by any citation reference."
            ),
            claim_text=claim.raw_text,
            evidence_ids=tuple(e.chunk_id for e in evidence),
            public_message=(
                "A numeric value in the answer lacks a citation to "
                "the source document."
            ),
        )
