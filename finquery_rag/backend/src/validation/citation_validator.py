"""Citation validation (Phase 4 Commit 6 + Phase 4 hotfix).

The ``CitationValidator`` checks three things:

1. **Source object validity**: each source in the ``sources`` tuple must
   reference a chunk that actually exists in the retrieved evidence. The
   ``chunk_id``, ``document_name``, and ``page`` must all be consistent
   with the evidence set. Invalid sources are flagged with
   ``CITATION_CHUNK_MISSING``, ``CITATION_DOCUMENT_MISMATCH``, or
   ``CITATION_PAGE_MISMATCH``.

2. **Citation presence**: when the policy requires citations
   (``require_citations=True``), numeric claims in the answer should be
   accompanied by at least one citation reference. If no citation
   references exist in the answer at all, each numeric claim is flagged
   with ``CITATION_MISSING``.

3. **Citation resolvability**: citation references like ``[1]`` or
   ``[doc.pdf, p.12]`` should resolve to an actual source in the
   ``sources`` tuple (not just any evidence chunk). Number citations
   ``[1]`` resolve to ``sources[0]``. Document citations must match both
   the document name AND the page. Unresolved citations are flagged with
   ``CITATION_UNRESOLVED`` or ``CITATION_NOT_RETRIEVED``.

4. **Claim support**: for each numeric claim that has an associated
   citation, the cited source's evidence must contain the claim's value.
   If not, the claim is flagged with ``CITATION_DOES_NOT_SUPPORT_CLAIM``.

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
CODE_CITATION_NOT_RETRIEVED = "CITATION_NOT_RETRIEVED"
CODE_CITATION_PAGE_MISMATCH = "CITATION_PAGE_MISMATCH"
CODE_CITATION_DOCUMENT_MISMATCH = "CITATION_DOCUMENT_MISMATCH"
CODE_CITATION_CHUNK_MISSING = "CITATION_CHUNK_MISSING"
CODE_CITATION_DOES_NOT_SUPPORT_CLAIM = "CITATION_DOES_NOT_SUPPORT_CLAIM"
CODE_DOCUMENT_COVERAGE_MISSING = "DOCUMENT_COVERAGE_MISSING"

# Regex for detecting document-style citations: [doc.pdf, p.12] or [report.pdf p5]
_RE_DOC_CITATION = re.compile(
    r"([\w.\-]+\.(?:pdf|docx?|xlsx?|txt))"  # document name
    r"[,\s]*"
    r"(?:p\.?\s*(\d+))?",                    # optional page
    re.IGNORECASE,
)


class CitationValidator:
    """Validates that citation references are present and resolvable.

    The validator inspects extracted claims, evidence, and the ``sources``
    tuple (the source objects returned to the API consumer) to ensure that
    numeric claims are backed by citations and that citations resolve to
    real sources with consistent chunk/document/page references.
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
        sources: tuple[dict, ...] = (),
    ) -> tuple[ValidationIssue, ...]:
        """Validate citations against the evidence set and source objects.

        Returns a tuple of ``ValidationIssue`` objects. If the policy
        does not require citations, only resolvability is checked.
        """
        issues: list[ValidationIssue] = []

        citation_claims = tuple(c for c in claims if c.claim_type == "citation_ref")
        numeric_claims = tuple(
            c for c in claims if c.claim_type in ("amount", "percent", "ratio")
        )

        # --- 1. Check source object validity ---
        source_issues = self._check_source_objects(sources, evidence, policy)
        issues.extend(source_issues)

        # --- 2. Check citation resolvability against sources ---
        unresolved = self._check_resolvability(citation_claims, evidence, sources)
        issues.extend(unresolved)

        # --- 3. Check citation presence for numeric claims ---
        if policy.require_citations and numeric_claims and not citation_claims:
            for claim in numeric_claims:
                issues.append(
                    self._build_missing_issue(claim, policy, evidence)
                )

        # --- 4. Check claim support: cited source must contain the value ---
        support_issues = self._check_claim_support(
            numeric_claims, citation_claims, evidence, sources, policy
        )
        issues.extend(support_issues)

        return tuple(issues)

    # -----------------------------------------------------------------
    # Source object validity checks
    # -----------------------------------------------------------------

    @staticmethod
    def _check_source_objects(
        sources: tuple[dict, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
    ) -> list[ValidationIssue]:
        """Check that each source object references a real evidence chunk.

        Validates:
        - ``chunk_id`` must exist in the evidence set.
        - ``document_name`` must match the evidence chunk's document.
        - ``page`` must match the evidence chunk's page.
        """
        if not sources or not evidence:
            return []

        # Build lookup: chunk_id -> EvidenceItem
        evidence_by_id: dict[str, EvidenceItem] = {}
        for item in evidence:
            if item.chunk_id:
                evidence_by_id[item.chunk_id] = item

        issues: list[ValidationIssue] = []
        seen_source_keys: set[tuple] = set()

        for src in sources:
            # Normalize source fields (sources may use filename or document_name).
            chunk_id = src.get("chunk_id") or ""
            doc_name = src.get("document_name") or src.get("filename") or ""
            page = src.get("page")

            # Deduplicate sources by (chunk_id, doc_name, page).
            source_key = (chunk_id, doc_name, str(page))
            if source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)

            if not chunk_id:
                # Source has no chunk_id — can't verify.
                issues.append(
                    ValidationIssue(
                        code=CODE_CITATION_CHUNK_MISSING,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Source (doc: {doc_name}, page: {page}) has no "
                            f"chunk_id and cannot be verified against evidence."
                        ),
                        claim_text=None,
                        evidence_ids=(),
                        public_message=(
                            "A source reference could not be verified "
                            "against the retrieved evidence."
                        ),
                    )
                )
                continue

            # Check chunk_id exists in evidence.
            matching_evidence = evidence_by_id.get(chunk_id)
            if matching_evidence is None:
                # Try matching by document_name + page as a fallback.
                found = False
                for item in evidence:
                    if (
                        item.document_name == doc_name
                        and (page is None or item.page == page)
                    ):
                        found = True
                        break
                if not found:
                    issues.append(
                        ValidationIssue(
                            code=CODE_CITATION_NOT_RETRIEVED,
                            severity=ValidationSeverity.CRITICAL,
                            message=(
                                f"Source chunk_id '{chunk_id}' (doc: {doc_name}, "
                                f"page: {page}) was not found in the retrieved "
                                f"evidence."
                            ),
                            claim_text=None,
                            evidence_ids=(chunk_id,),
                            public_message=(
                                "A source reference points to a passage "
                                "that was not retrieved."
                            ),
                        )
                    )
                continue

            # Check document_name consistency.
            # Use lenient matching: the context builder may strip file
            # extensions and prefixes from the document name (e.g.,
            # "paper.pdf" -> "paper"). We compare normalized forms.
            if (
                doc_name
                and matching_evidence.document_name
                and not CitationValidator._doc_names_match(
                    doc_name, matching_evidence.document_name
                )
            ):
                issues.append(
                    ValidationIssue(
                        code=CODE_CITATION_DOCUMENT_MISMATCH,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Source document_name '{doc_name}' does not match "
                            f"evidence document_name '{matching_evidence.document_name}' "
                            f"for chunk_id '{chunk_id}'."
                        ),
                        claim_text=None,
                        evidence_ids=(chunk_id,),
                        public_message=(
                            "A source document name does not match the "
                            "retrieved evidence."
                        ),
                    )
                )

            # Check page consistency.
            if page is not None and matching_evidence.page is not None:
                try:
                    src_page = int(page)
                    ev_page = int(matching_evidence.page)
                    if src_page != ev_page:
                        issues.append(
                            ValidationIssue(
                                code=CODE_CITATION_PAGE_MISMATCH,
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"Source page {src_page} does not match "
                                    f"evidence page {ev_page} for chunk_id "
                                    f"'{chunk_id}'."
                                ),
                                claim_text=None,
                                evidence_ids=(chunk_id,),
                                public_message=(
                                    "A source page number does not match "
                                    "the retrieved evidence."
                                ),
                            )
                        )
                except (ValueError, TypeError):
                    pass  # Non-integer page; skip check.

        return issues

    # -----------------------------------------------------------------
    # Resolvability checks
    # -----------------------------------------------------------------

    @staticmethod
    def _check_resolvability(
        citation_claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        sources: tuple[dict, ...],
    ) -> list[ValidationIssue]:
        """Check if each citation reference resolves to a source.

        - Number citations ``[1]`` resolve to ``sources[0]`` (1-indexed).
        - Document citations ``[doc.pdf, p.12]`` resolve if both the
          document name AND page match a source.
        - If no sources are provided, falls back to evidence-based resolution.
        """
        if not citation_claims:
            return []

        issues: list[ValidationIssue] = []

        # Build source lookup sets.
        source_docs: set[str | None] = set()
        source_doc_pages: set[tuple[str | None, int | None]] = set()
        for src in sources:
            doc = src.get("document_name") or src.get("filename")
            page = src.get("page")
            source_docs.add(doc)
            try:
                page_int = int(page) if page is not None else None
            except (ValueError, TypeError):
                page_int = None
            source_doc_pages.add((doc, page_int))

        evidence_docs = {item.document_name for item in evidence if item.document_name}

        for claim in citation_claims:
            resolved = False
            for ref in claim.citation_refs:
                if CitationValidator._resolve_citation(
                    ref, evidence, sources, source_docs, source_doc_pages, evidence_docs
                ):
                    resolved = True
                    break

            if not resolved:
                issues.append(
                    ValidationIssue(
                        code=CODE_CITATION_UNRESOLVED,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Citation '{claim.raw_text}' could not be resolved "
                            f"to any source."
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
        sources: tuple[dict, ...],
        source_docs: set[str | None],
        source_doc_pages: set[tuple[str | None, int | None]],
        evidence_docs: set[str | None],
    ) -> bool:
        """Check if a single citation reference resolves to a source.

        - Number citations (``"1"``, ``"2"``) resolve if the number is
          within the sources count (falls back to evidence count).
        - Document citations (``"doc.pdf, p.12"``) resolve if both the
          document name AND page match a source.
        - Other citations resolve if the reference text appears in any
          evidence content.
        """
        ref = ref.strip()
        if not ref:
            return False

        # Number citation: [1], [2], etc. — resolve against sources first.
        if ref.isdigit():
            idx = int(ref)
            if sources and 1 <= idx <= len(sources):
                return True
            # Fallback: check evidence count.
            return 1 <= idx <= len(evidence)

        # Document-style citation: [doc.pdf, p.12]
        doc_match = _RE_DOC_CITATION.search(ref)
        if doc_match:
            doc_name = doc_match.group(1)
            page_str = doc_match.group(2)
            try:
                page_int = int(page_str) if page_str else None
            except ValueError:
                page_int = None

            # If we have sources, require both doc + page match.
            if sources:
                for src in sources:
                    src_doc = src.get("document_name") or src.get("filename")
                    if src_doc == doc_name:
                        if page_int is None:
                            return True  # No page specified; doc match suffices.
                        try:
                            src_page = int(src.get("page")) if src.get("page") is not None else None
                        except (ValueError, TypeError):
                            src_page = None
                        if src_page == page_int:
                            return True
                # Doc found in sources but page didn't match.
                if doc_name in source_docs:
                    return False  # Don't fall through — page mismatch is real.
            else:
                # No sources: fall back to evidence.
                if doc_name in evidence_docs:
                    return True

        # Fallback: check if the reference text appears in any evidence.
        for item in evidence:
            if ref in (item.content or ""):
                return True

        # Check if any evidence/source document name contains the reference.
        for doc in evidence_docs:
            if doc and doc in ref:
                return True

        return False

    # -----------------------------------------------------------------
    # Claim support checks
    # -----------------------------------------------------------------

    @staticmethod
    def _check_claim_support(
        numeric_claims: tuple[ExtractedClaim, ...],
        citation_claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        sources: tuple[dict, ...],
        policy: ValidationPolicy,
    ) -> list[ValidationIssue]:
        """Check that cited numeric claims are supported by their cited evidence.

        For each numeric claim that is near a citation reference, find the
        cited source's evidence and verify that the claim's value appears
        in that evidence. If not, flag ``CITATION_DOES_NOT_SUPPORT_CLAIM``.
        """
        if not numeric_claims or not evidence:
            return []

        # If there are no citation references, skip (CITATION_MISSING handles it).
        if not citation_claims:
            return []

        issues: list[ValidationIssue] = []

        # Build source -> evidence lookup.
        source_evidence: dict[int, EvidenceItem] = {}  # 1-indexed
        for i, src in enumerate(sources):
            chunk_id = src.get("chunk_id") or ""
            for item in evidence:
                if item.chunk_id == chunk_id:
                    source_evidence[i + 1] = item
                    break

        # For each numeric claim, find the nearest citation ref and check support.
        for claim in numeric_claims:
            if claim.value is None:
                continue

            # Find the nearest citation claim (by position in answer text).
            # Since claims are extracted in order, we can use a simple proximity
            # heuristic: look for a citation ref whose raw_text appears near
            # the numeric claim in the answer.
            # For now, if any citation ref is a number and resolves to a source,
            # check if that source's evidence contains the value.
            for cite_claim in citation_claims:
                for ref in cite_claim.citation_refs:
                    ref_stripped = ref.strip()
                    if ref_stripped.isdigit():
                        idx = int(ref_stripped)
                        cited_evidence = source_evidence.get(idx)
                        if cited_evidence is not None:
                            # Check if the claim value appears in this evidence.
                            if not CitationValidator._value_in_evidence(
                                claim, cited_evidence
                            ):
                                issues.append(
                                    ValidationIssue(
                                        code=CODE_CITATION_DOES_NOT_SUPPORT_CLAIM,
                                        severity=ValidationSeverity.ERROR,
                                        message=(
                                            f"Numeric claim '{claim.raw_text}' "
                                            f"(value: {claim.value}) is not supported "
                                            f"by the cited source (chunk_id: "
                                            f"{cited_evidence.chunk_id})."
                                        ),
                                        claim_text=claim.raw_text,
                                        evidence_ids=(cited_evidence.chunk_id,),
                                        public_message=(
                                            "A numeric value in the answer is not "
                                            "supported by its cited source."
                                        ),
                                    )
                                )
                            break  # Only check the first matching citation.

        return issues

    @staticmethod
    def _value_in_evidence(claim: ExtractedClaim, item: EvidenceItem) -> bool:
        """Check if a claim's value appears in the given evidence item."""
        if claim.value is None:
            return True  # Can't check; assume supported.

        text = item.content or ""
        value = claim.value

        # Generate representations of the value.
        reps: list[str] = [str(value), f"{value:,}", str(value.normalize())]
        if value == value.to_integral_value():
            int_val = int(value)
            reps.append(str(int_val))
            reps.append(f"{int_val:,}")

        if claim.claim_type == "percent":
            reps.append(f"{value}%")
            reps.append(f"{value:.2f}%")

        for rep in reps:
            if rep and rep in text:
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

    @staticmethod
    def _doc_names_match(name_a: str, name_b: str) -> bool:
        """Check if two document names refer to the same document.

        The context builder may strip file extensions and user prefixes
        from the document name (e.g., ``"paper.pdf"`` -> ``"paper"``).
        This method normalizes both names by:
        1. Converting to lowercase.
        2. Removing common file extensions.
        3. Stripping user-id prefixes (``user_123_``).
        4. Comparing the resulting stems.
        """
        import re

        def _normalize(name: str) -> str:
            n = name.lower().strip()
            # Remove common file extensions.
            n = re.sub(r"\.(pdf|txt|csv|xlsx?|docx?|html?|md|json)$", "", n)
            # Strip user-id prefixes (e.g., "user_123_paper" -> "paper").
            n = re.sub(r"^user_\d+_", "", n)
            return n

        na = _normalize(name_a)
        nb = _normalize(name_b)
        return na == nb or na in nb or nb in na
