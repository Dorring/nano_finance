"""Tests for CitationValidator and CalculationValidator (Phase 4 Commit 6).

Verifies citation presence/resolvability checks and calculation result
consistency validation.
"""

from __future__ import annotations

import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.domain.evidence import EvidenceItem
from src.domain.validation import ValidationSeverity
from src.validation.calculation_validator import (
    CODE_CALCULATION_MISMATCH,
    CODE_CALCULATION_VALUE_MISMATCH,
    CalculationValidator,
)
from src.validation.citation_validator import (
    CODE_CITATION_MISSING,
    CODE_CITATION_UNRESOLVED,
    CitationValidator,
)
from src.validation.claim_extractor import ClaimExtractor
from src.validation.validation_policy import (
    ACTION_BLOCK,
    ACTION_WARN,
    ValidationPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence(
    content: str,
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content=content,
        document_name=document_name,
        page=12,
        content_type="text",
        score=0.9,
        rerank_score=None,
        metadata={},
    )


def _strict_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=True,
        require_citations=True,
        validate_numeric_claims=True,
        validate_units=True,
        validate_periods=True,
        strict_numeric_grounding=True,
        unsupported_numeric_action=ACTION_BLOCK,
        missing_citation_action=ACTION_BLOCK,
    )


def _warn_citation_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=True,
        require_citations=True,
        validate_numeric_claims=True,
        validate_units=True,
        validate_periods=True,
        strict_numeric_grounding=True,
        unsupported_numeric_action=ACTION_BLOCK,
        missing_citation_action=ACTION_WARN,
    )


def _no_citation_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=True,
        require_citations=False,
        validate_numeric_claims=True,
        validate_units=True,
        validate_periods=True,
        strict_numeric_grounding=True,
        unsupported_numeric_action=ACTION_BLOCK,
        missing_citation_action=ACTION_WARN,
    )


# ---------------------------------------------------------------------------
# CitationValidator tests
# ---------------------------------------------------------------------------


class TestCitationValidator:
    def test_citation_present_no_issue(self):
        """Numeric claims with citations should not produce CITATION_MISSING."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000 [1]."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) == 0

    def test_citation_missing_when_required(self):
        """Numeric claims without citations should produce CITATION_MISSING."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) >= 1
        assert missing[0].severity == ValidationSeverity.CRITICAL

    def test_citation_missing_warn_action(self):
        """When missing_citation_action is 'warn', severity is WARNING."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _warn_citation_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) >= 1
        assert missing[0].severity == ValidationSeverity.WARNING

    def test_no_citation_required_no_issue(self):
        """When require_citations is False, no CITATION_MISSING."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _no_citation_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) == 0

    def test_number_citation_resolves(self):
        """A number citation [1] should resolve if there is at least 1 evidence."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000 [1]."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        unresolved = [i for i in issues if i.code == CODE_CITATION_UNRESOLVED]
        assert len(unresolved) == 0

    def test_document_citation_resolves(self):
        """A document citation should resolve if the document is in evidence."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000 [annual_report.pdf, p.12]."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        unresolved = [i for i in issues if i.code == CODE_CITATION_UNRESOLVED]
        assert len(unresolved) == 0

    def test_unresolved_citation(self):
        """A citation to a non-existent document should be flagged."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000 [nonexistent_report.pdf, p.99]."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        unresolved = [i for i in issues if i.code == CODE_CITATION_UNRESOLVED]
        assert len(unresolved) == 1

    def test_no_numeric_claims_no_missing(self):
        """If there are no numeric claims, no CITATION_MISSING."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "The company performed well."
        evidence = (_evidence("The company performed well."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) == 0

    def test_public_message_no_internals(self):
        """Public message should not leak internal details."""
        extractor = ClaimExtractor()
        validator = CitationValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        missing = [i for i in issues if i.code == CODE_CITATION_MISSING]
        assert len(missing) >= 1
        public = missing[0].to_public_dict()
        assert "message" not in public
        assert "evidence_ids" not in public


# ---------------------------------------------------------------------------
# CalculationValidator tests
# ---------------------------------------------------------------------------


class TestCalculationValidator:
    def test_no_calculation_result(self):
        """When calculation_result is None, no issues."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 40%."
        claims = extractor.extract(answer)
        issues = validator.validate(claims, None)
        assert len(issues) == 0

    def test_calculation_not_executed(self):
        """When calculation is NOT_APPLICABLE, no issues."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 40%."
        claims = extractor.extract(answer)
        calc = CalculationResult(status=CalculationStatus.NOT_APPLICABLE)
        issues = validator.validate(claims, calc)
        assert len(issues) == 0

    def test_calculation_match_no_issue(self):
        """When the answer value matches the calculation, no issue."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 40%."
        claims = extractor.extract(answer)
        calc = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            target_metric="gross_margin",
        )
        issues = validator.validate(claims, calc)
        assert len(issues) == 0

    def test_calculation_mismatch(self):
        """When the answer value differs from the calculation, flag it."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 50%."
        claims = extractor.extract(answer)
        calc = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            target_metric="gross_margin",
        )
        issues = validator.validate(claims, calc)
        assert len(issues) == 1
        assert issues[0].code in (
            CODE_CALCULATION_MISMATCH,
            CODE_CALCULATION_VALUE_MISMATCH,
        )
        assert issues[0].severity == ValidationSeverity.CRITICAL

    def test_answer_does_not_mention_metric(self):
        """When the answer doesn't mention the calculated metric, no issue."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Revenue was $1,000,000."
        claims = extractor.extract(answer)
        calc = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            target_metric="gross_margin",
        )
        issues = validator.validate(claims, calc)
        # Phase 4 hotfix: answer must mention the calculated value;
        # missing metric claim -> CALCULATION_VALUE_MISSING
        from src.validation.calculation_validator import CODE_CALCULATION_VALUE_MISSING

        assert len(issues) == 1
        assert issues[0].code == CODE_CALCULATION_VALUE_MISSING

    def test_ratio_vs_percentage_match(self):
        """0.4 ratio should match 40% percentage."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 40%."
        claims = extractor.extract(answer)
        calc = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            target_metric="gross_margin",
        )
        issues = validator.validate(claims, calc)
        assert len(issues) == 0

    def test_public_message_no_internals(self):
        """Public message should not leak calculation internals."""
        extractor = ClaimExtractor()
        validator = CalculationValidator()
        answer = "Gross margin was 50%."
        claims = extractor.extract(answer)
        calc = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            target_metric="gross_margin",
        )
        issues = validator.validate(claims, calc)
        assert len(issues) == 1
        public = issues[0].to_public_dict()
        assert "message" not in public
        assert public["public_message"] is not None
