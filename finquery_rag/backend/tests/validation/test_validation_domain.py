"""Tests for Phase 4 validation domain objects (src/domain/validation.py)."""
from __future__ import annotations

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.validation import (  # noqa: E402
    AnswerabilityResult,
    AnswerabilityStatus,
    ExtractedClaim,
    GroundedResponseResult,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


class TestEnums:
    def test_answerability_status_values(self):
        assert AnswerabilityStatus.ANSWERABLE.value == "answerable"
        assert AnswerabilityStatus.NOT_ANSWERABLE.value == "not_answerable"
        assert AnswerabilityStatus.CALCULATION_BLOCKED.value == "calculation_blocked"
        assert AnswerabilityStatus.PARTIALLY_ANSWERABLE.value == "partially_answerable"

    def test_validation_status_values(self):
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.BLOCKED.value == "blocked"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.REPAIRABLE.value == "repairable"
        assert ValidationStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_severity_values(self):
        assert ValidationSeverity.CRITICAL.value == "critical"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"


class TestValidationIssue:
    def test_public_dict_omits_internal_message(self):
        issue = ValidationIssue(
            code="NUMERIC_VALUE_MISMATCH",
            severity=ValidationSeverity.CRITICAL,
            message="internal: expected 1250 but got 152 at chunk_001 p12",
            claim_text="152 million",
            evidence_ids=("chunk_001",),
            public_message="回答包含与证据不符的数值。",
        )
        pub = issue.to_public_dict()
        assert pub["code"] == "NUMERIC_VALUE_MISMATCH"
        assert pub["severity"] == "critical"
        assert pub["public_message"] == "回答包含与证据不符的数值。"
        # Internal message and evidence_ids must NOT leak.
        assert "message" not in pub
        assert "evidence_ids" not in pub
        assert "internal" not in str(pub)

    def test_trace_dict_includes_internal_message(self):
        issue = ValidationIssue(
            code="UNIT_MISMATCH",
            severity=ValidationSeverity.ERROR,
            message="expected million got billion",
            claim_text="125 billion",
        )
        trace = issue.to_trace_dict()
        # Phase 4 hotfix: trace redacts message to message_hash.
        assert "message_hash" in trace
        assert "message" not in trace
        assert trace["code"] == "UNIT_MISMATCH"
        # claim_text is redacted to claim_excerpt (max 80 chars).
        assert trace["claim_excerpt"] == "125 billion"
        assert "claim_text" not in trace


class TestAnswerabilityResult:
    def test_public_dict_omits_scores(self):
        result = AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason_codes=("no_evidence",),
            evidence_count=0,
            document_count=0,
            best_score=0.01,
            average_score=0.01,
            missing_requirements=("document: annual_report.pdf",),
        )
        pub = result.to_public_dict()
        assert pub["status"] == "not_answerable"
        assert pub["missing_requirements"] == ["document: annual_report.pdf"]
        assert "best_score" not in pub
        assert "average_score" not in pub

    def test_trace_dict_includes_scores(self):
        result = AnswerabilityResult(
            status=AnswerabilityStatus.ANSWERABLE,
            reason_codes=(),
            evidence_count=3,
            document_count=1,
            best_score=0.95,
            average_score=0.8,
        )
        trace = result.to_trace_dict()
        assert trace["best_score"] == 0.95
        assert trace["average_score"] == 0.8


class TestExtractedClaim:
    def test_decimal_preserved(self):
        claim = ExtractedClaim(
            claim_id="c1", claim_type="amount", raw_text="125.50",
            value=Decimal("125.50"), unit="base", scale="million",
        )
        assert claim.value == Decimal("125.50")

    def test_negative_value_preserved(self):
        claim = ExtractedClaim(
            claim_id="c2", claim_type="amount", raw_text="(50)",
            value=Decimal("-50"), unit="base",
        )
        assert claim.value == Decimal("-50")

    def test_trace_dict(self):
        claim = ExtractedClaim(
            claim_id="c3", claim_type="percent", raw_text="25%",
            value=Decimal("25"), unit="percent",
            citation_refs=("ref1",),
        )
        trace = claim.to_trace_dict()
        assert trace["value"] == "25"
        assert trace["citation_refs"] == ["ref1"]


class TestValidationResult:
    def test_public_dict_omits_repaired_answer(self):
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            issues=(),
            checked_claim_count=3,
            supported_claim_count=3,
            unsupported_claim_count=0,
        )
        pub = result.to_public_dict()
        assert pub["status"] == "passed"
        assert pub["checked_claim_count"] == 3
        assert "repaired_answer" not in pub

    def test_public_dict_includes_public_issues(self):
        issue = ValidationIssue(
            code="UNSUPPORTED_NUMERIC_CLAIM",
            severity=ValidationSeverity.ERROR,
            message="internal detail",
            public_message="回答包含无法由当前证据支持的数值。",
        )
        result = ValidationResult(
            status=ValidationStatus.BLOCKED,
            issues=(issue,),
            checked_claim_count=2,
            supported_claim_count=1,
            unsupported_claim_count=1,
        )
        pub = result.to_public_dict()
        assert pub["issues"][0]["public_message"] == "回答包含无法由当前证据支持的数值。"
        assert "internal detail" not in str(pub)

    def test_trace_dict_includes_issue_codes_and_repaired_flag(self):
        issue = ValidationIssue(
            code="SCALE_MISMATCH", severity=ValidationSeverity.ERROR,
            message="m", claim_text="125 billion",
        )
        result = ValidationResult(
            status=ValidationStatus.REPAIRABLE,
            issues=(issue,),
            checked_claim_count=1,
            supported_claim_count=0,
            unsupported_claim_count=1,
            repaired_answer="repaired text",
        )
        trace = result.to_trace_dict()
        assert trace["issue_codes"] == ["SCALE_MISMATCH"]
        assert trace["repaired"] is True
        # Full repaired answer text is NOT in trace (minimize leakage).
        assert "repaired text" not in str(trace)

    def test_failed_status_does_not_default_to_passed(self):
        result = ValidationResult(status=ValidationStatus.FAILED)
        assert result.status is ValidationStatus.FAILED
        assert result.status is not ValidationStatus.PASSED


class TestGroundedResponseResult:
    def test_public_dict_includes_answerability_and_validation(self):
        ar = AnswerabilityResult(
            status=AnswerabilityStatus.ANSWERABLE, reason_codes=(),
            evidence_count=2, document_count=1,
            best_score=0.9, average_score=0.8,
        )
        vr = ValidationResult(
            status=ValidationStatus.PASSED,
            checked_claim_count=1, supported_claim_count=1,
        )
        gr = GroundedResponseResult(
            answer="safe answer",
            sources=({"filename": "a.pdf", "page": 1},),
            answerability=ar,
            validation=vr,
            warnings=("partial coverage",),
        )
        pub = gr.to_public_dict()
        assert pub["answer"] == "safe answer"
        assert pub["answerability"]["status"] == "answerable"
        assert pub["validation"]["status"] == "passed"
        assert pub["warnings"] == ["partial coverage"]

    def test_public_dict_omits_none_sections(self):
        gr = GroundedResponseResult(
            answer="conv answer",
            sources=(),
            answerability=None,
            validation=None,
        )
        pub = gr.to_public_dict()
        assert "answerability" not in pub
        assert "validation" not in pub
        assert "warnings" not in pub
