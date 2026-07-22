"""Tests for ResponseValidator, UnsupportedClaimValidator, and
GroundedValidationPipeline (Phase 4 Commit 7).

Verifies:
- ``UnsupportedClaimValidator`` flags numeric claims whose metric is
  absent from all evidence (with calculation exemption).
- ``ResponseValidator`` aggregates all validators into a single verdict
  with fail-closed semantics.
- ``GroundedValidationPipeline`` combines answerability + validation
  into a ``GroundedResponseResult``.
- Public / trace dict separation.
- Determinism (same inputs -> same outputs).
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
from src.domain.validation import (
    AnswerabilityStatus,
    GroundedResponseResult,
    ValidationSeverity,
    ValidationStatus,
)
from src.retrieval.context_builder import SufficiencyResult
from src.validation.unsupported_claim_validator import (
    CODE_UNSUPPORTED_CLAIM,
    UnsupportedClaimValidator,
)
from src.validation.response_validator import ResponseValidator
from src.validation.validation_pipeline import GroundedValidationPipeline
from src.validation.validation_policy import (
    ACTION_WARN,
    ValidationPolicy,
    get_policy_for_intent,
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
    return get_policy_for_intent("financial_calculation")


def _executed_calc(
    value: Decimal = Decimal("0.4"),
    target_metric: str = "gross_margin",
) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.GROSS_MARGIN,
        value=value,
        unit="ratio",
        formula="gross_profit / revenue",
        formula_version="gross_margin.v1",
        target_metric=target_metric,
    )


# ---------------------------------------------------------------------------
# UnsupportedClaimValidator
# ---------------------------------------------------------------------------

class TestUnsupportedClaimValidator:
    """Tests for the UnsupportedClaimValidator."""

    def test_metric_present_in_evidence_passes(self):
        """Numeric claim whose metric appears in evidence is not flagged."""
        claims = _extract_claims("The gross margin was 40%.")
        evidence = (
            _evidence("The company reported gross margin of 38% in 2024."),
        )
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, _strict_policy())
        codes = [i.code for i in issues]
        assert CODE_UNSUPPORTED_CLAIM not in codes

    def test_metric_absent_from_evidence_is_flagged(self):
        """Numeric claim whose metric is absent from evidence is flagged."""
        claims = _extract_claims("The EBITDA was $5 million.")
        evidence = (
            _evidence("The company reported revenue of $100 million in 2024."),
        )
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, _strict_policy())
        codes = [i.code for i in issues]
        assert CODE_UNSUPPORTED_CLAIM in codes

    def test_calculation_target_metric_exempt(self):
        """Claims matching the calculation target_metric are not flagged."""
        claims = _extract_claims("The gross margin was 40%.")
        # Evidence does NOT mention "gross margin" — but the calculation
        # target_metric is gross_margin, so the claim is exempt.
        evidence = (
            _evidence("The company reported revenue of $100 million."),
        )
        calc = _executed_calc(target_metric="gross_margin")
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, calc, _strict_policy())
        codes = [i.code for i in issues]
        assert CODE_UNSUPPORTED_CLAIM not in codes

    def test_policy_disabled_returns_empty(self):
        """When validate_numeric_claims is False, no issues are returned."""
        claims = _extract_claims("The EBITDA was $5 million.")
        evidence = (
            _evidence("The company reported revenue of $100 million."),
        )
        policy = ValidationPolicy(
            require_evidence=True,
            require_citations=False,
            validate_numeric_claims=False,
            validate_units=False,
            validate_periods=False,
            strict_numeric_grounding=False,
            unsupported_numeric_action=ACTION_WARN,
            missing_citation_action=ACTION_WARN,
        )
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, policy)
        assert issues == ()

    def test_non_numeric_claims_not_checked(self):
        """Non-numeric claims (period, citation) are not flagged."""
        # We cannot easily inject non-numeric claims via the extractor;
        # this test verifies that an answer with no numeric claims
        # produces no unsupported-claim issues.
        claims = _extract_claims("The company performed well in 2024.")
        evidence = (_evidence("Some unrelated text."),)
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, _strict_policy())
        assert issues == ()

    def test_issue_severity_is_error(self):
        """Unsupported claim issues are ERROR severity."""
        claims = _extract_claims("The EBITDA was $5 million.")
        evidence = (
            _evidence("The company reported revenue of $100 million."),
        )
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, _strict_policy())
        unsupported = [i for i in issues if i.code == CODE_UNSUPPORTED_CLAIM]
        assert len(unsupported) >= 1
        assert all(
            i.severity is ValidationSeverity.ERROR for i in unsupported
        )

    def test_issue_has_public_message(self):
        """Issues carry a sanitized public_message."""
        claims = _extract_claims("The EBITDA was $5 million.")
        evidence = (
            _evidence("The company reported revenue of $100 million."),
        )
        validator = UnsupportedClaimValidator()
        issues = validator.validate(claims, evidence, None, _strict_policy())
        unsupported = [i for i in issues if i.code == CODE_UNSUPPORTED_CLAIM]
        assert len(unsupported) >= 1
        assert unsupported[0].public_message is not None
        # Public message must NOT contain the metric name (internal).
        assert "EBITDA" not in unsupported[0].public_message


# ---------------------------------------------------------------------------
# ResponseValidator
# ---------------------------------------------------------------------------

class TestResponseValidatorPass:
    """Tests where the answer passes validation."""

    def test_answer_with_grounded_numbers_passes(self):
        """Answer whose numbers appear in evidence passes validation."""
        answer = "The revenue was $1,000,000 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.status is ValidationStatus.PASSED

    def test_conversation_intent_not_applicable(self):
        """Conversation intent produces NOT_APPLICABLE."""
        answer = "Hello, how can I help you today?"
        evidence = ()
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="conversation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.status is ValidationStatus.NOT_APPLICABLE

    def test_calculation_supported_claim_not_flagged(self):
        """A claim matching an EXECUTED calculation is not flagged."""
        # Calculation says gross_margin = 0.4 (40%).
        calc = _executed_calc(value=Decimal("0.4"), target_metric="gross_margin")
        # Answer claims 40% gross margin — matches calculation.
        answer = "The gross margin was 40% in 2024 [1]."
        # Evidence has 2024 context but no 40% number.
        evidence = (
            _evidence("Fiscal year 2024 annual report context."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=calc,
        )
        # The 40% claim is supported by the calculation; no NUMERIC_UNGROUND.
        assert result.status is ValidationStatus.PASSED


class TestResponseValidatorBlock:
    """Tests where the answer must be blocked."""

    def test_ungrounded_numeric_blocks(self):
        """A numeric claim absent from evidence blocks (strict policy)."""
        answer = "The revenue was $999,999,999 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.status is ValidationStatus.BLOCKED

    def test_calculation_mismatch_blocks(self):
        """A claim that contradicts the calculation result blocks."""
        calc = _executed_calc(value=Decimal("0.4"), target_metric="gross_margin")
        # Answer claims 55% — does not match the 40% calculation.
        answer = "The gross margin was 55% in 2024 [1]."
        evidence = (
            _evidence("Fiscal year 2024 annual report context."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=calc,
        )
        assert result.status is ValidationStatus.BLOCKED

    def test_missing_citation_blocks_when_required(self):
        """financial_calculation requires citations; missing citation blocks."""
        # Number is grounded but no citation [n] present.
        answer = "The revenue was $1,000,000 in 2024."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.status is ValidationStatus.BLOCKED

    def test_period_mismatch_blocks(self):
        """A year in the answer not present in evidence blocks."""
        answer = "The revenue was $1,000,000 in 2025 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.status is ValidationStatus.BLOCKED


class TestResponseValidatorFailClosed:
    """Tests for fail-closed behavior on validator errors."""

    def test_none_evidence_does_not_crash(self):
        """Passing None as evidence does not crash the validator."""
        # The validator should handle None gracefully via fail-closed.
        validator = ResponseValidator()
        # We simulate an internal error by passing malformed input.
        # The ResponseValidator catches all exceptions and returns FAILED.
        result = validator.validate(
            answer="Some answer.",
            intent="financial_calculation",
            evidence=None,  # type: ignore[arg-type]
            calculation_result=None,
        )
        # Either NOT_APPLICABLE (if no claims extracted) or FAILED.
        assert result.status in (
            ValidationStatus.PASSED,
            ValidationStatus.FAILED,
            ValidationStatus.NOT_APPLICABLE,
        )

    def test_failed_status_has_critical_issue(self):
        """If the validator returns FAILED, it must have a CRITICAL issue."""
        # Force a failure by passing a non-iterable evidence.
        validator = ResponseValidator()
        # Use an object that will cause iteration to fail inside a validator.
        class _BadEvidence:
            def __iter__(self):
                raise RuntimeError("simulated internal error")

        result = validator.validate(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=_BadEvidence(),  # type: ignore[arg-type]
            calculation_result=None,
        )
        assert result.status is ValidationStatus.FAILED
        assert len(result.issues) >= 1
        assert all(
            i.severity is ValidationSeverity.CRITICAL for i in result.issues
        )


class TestResponseValidatorAggregation:
    """Tests for issue aggregation and status determination."""

    def test_only_warnings_passes(self):
        """An answer with only WARNING-level issues passes."""
        # document_summary does not require citations; missing citation is
        # a WARNING there. The numeric claim is grounded.
        answer = "The revenue was $1,000,000 in 2024."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="document_summary",
            evidence=evidence,
            calculation_result=None,
        )
        # document_summary: strict_numeric_grounding=True, missing_citation
        # action is WARN. Number is grounded, no citation -> WARNING only.
        assert result.status is ValidationStatus.PASSED

    def test_claim_counts_populated(self):
        """checked/supported/unsupported claim counts are populated."""
        answer = "The revenue was $1,000,000 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert result.checked_claim_count > 0
        assert result.supported_claim_count >= 0
        assert result.unsupported_claim_count >= 0
        assert (
            result.supported_claim_count + result.unsupported_claim_count
            <= result.checked_claim_count
        )


class TestResponseValidatorSerialization:
    """Tests for public/trace dict separation."""

    def test_public_dict_excludes_internal_messages(self):
        """to_public_dict excludes internal issue messages."""
        answer = "The revenue was $999,999,999 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        public = result.to_public_dict()
        assert public["status"] == result.status.value
        for issue in public["issues"]:
            assert "message" not in issue  # internal message excluded
            assert "public_message" in issue

    def test_trace_dict_includes_internal_messages(self):
        """to_trace_dict includes internal issue messages for debugging."""
        answer = "The revenue was $999,999,999 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        result = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        trace = result.to_trace_dict()
        assert "issues" in trace
        for issue in trace["issues"]:
            # Phase 4 hotfix: trace redacts message to message_hash.
            assert "message_hash" in issue  # hashed internal message
            assert "code" in issue

    def test_failed_public_dict_sanitized(self):
        """FAILED result's public_dict has no internal exception text."""
        validator = ResponseValidator()

        class _BadEvidence:
            def __iter__(self):
                raise RuntimeError("simulated internal error")

        result = validator.validate(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=_BadEvidence(),  # type: ignore[arg-type]
            calculation_result=None,
        )
        public = result.to_public_dict()
        assert public["status"] == "failed"
        for issue in public["issues"]:
            # Public message must NOT contain the exception text.
            assert "simulated internal error" not in (
                issue.get("public_message") or ""
            )


class TestResponseValidatorDeterminism:
    """Determinism: same inputs produce same outputs."""

    def test_same_inputs_same_output(self):
        answer = "The revenue was $1,000,000 in 2024 [1]."
        evidence = (
            _evidence("The revenue was $1,000,000 in fiscal year 2024."),
        )
        validator = ResponseValidator()
        r1 = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        r2 = validator.validate(
            answer=answer,
            intent="financial_calculation",
            evidence=evidence,
            calculation_result=None,
        )
        assert r1.status == r2.status
        assert len(r1.issues) == len(r2.issues)
        assert tuple(i.code for i in r1.issues) == tuple(
            i.code for i in r2.issues
        )


# ---------------------------------------------------------------------------
# GroundedValidationPipeline
# ---------------------------------------------------------------------------

class TestGroundedValidationPipelineAnswerability:
    """Tests for the pre-generation answerability gate."""

    def test_not_answerable_when_no_evidence(self):
        pipeline = GroundedValidationPipeline()
        result = pipeline.evaluate_answerability(
            question="What is the revenue?",
            intent="financial_calculation",
            evidence=(),
            sufficiency_result=SufficiencyResult(
                is_sufficient=False, best_score=0.0, average_score=0.0
            ),
            calculation_result=None,
            requested_documents=(),
        )
        assert result.status is AnswerabilityStatus.NOT_ANSWERABLE

    def test_answerable_when_evidence_sufficient(self):
        pipeline = GroundedValidationPipeline()
        result = pipeline.evaluate_answerability(
            question="What is the revenue?",
            intent="financial_calculation",
            evidence=(_evidence("Revenue was $1M."),),
            sufficiency_result=SufficiencyResult(
                is_sufficient=True, best_score=0.9, average_score=0.85
            ),
            calculation_result=None,
            requested_documents=(),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE

    def test_calculation_blocked_bypasses_llm(self):
        pipeline = GroundedValidationPipeline()
        calc = CalculationResult(
            status=CalculationStatus.BLOCKED,
            error_code="INSUFFICIENT_OPERANDS",
        )
        result = pipeline.evaluate_answerability(
            question="What is the gross margin?",
            intent="financial_calculation",
            evidence=(_evidence("Some text."),),
            sufficiency_result=SufficiencyResult(
                is_sufficient=True, best_score=0.9, average_score=0.85
            ),
            calculation_result=calc,
            requested_documents=(),
        )
        assert result.status is AnswerabilityStatus.CALCULATION_BLOCKED


class TestGroundedValidationPipelineValidation:
    """Tests for the post-generation validation gate."""

    def test_validate_response_passes(self):
        pipeline = GroundedValidationPipeline()
        result = pipeline.validate_response(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=(_evidence("The revenue was $1,000,000 in 2024."),),
            calculation_result=None,
        )
        assert result.status is ValidationStatus.PASSED

    def test_validate_response_blocks(self):
        pipeline = GroundedValidationPipeline()
        result = pipeline.validate_response(
            answer="The revenue was $999,999,999 in 2024 [1].",
            intent="financial_calculation",
            evidence=(_evidence("The revenue was $1,000,000 in 2024."),),
            calculation_result=None,
        )
        assert result.status is ValidationStatus.BLOCKED

    def test_validate_response_conversation_not_applicable(self):
        pipeline = GroundedValidationPipeline()
        result = pipeline.validate_response(
            answer="Hello!",
            intent="conversation",
            evidence=(),
            calculation_result=None,
        )
        assert result.status is ValidationStatus.NOT_APPLICABLE


class TestGroundedValidationPipelineAggregation:
    """Tests for build_grounded_response."""

    def test_build_grounded_response_basic(self):
        pipeline = GroundedValidationPipeline()
        answerability = pipeline.evaluate_answerability(
            question="What is the revenue?",
            intent="financial_calculation",
            evidence=(_evidence("Revenue was $1M."),),
            sufficiency_result=SufficiencyResult(
                is_sufficient=True, best_score=0.9, average_score=0.85
            ),
            calculation_result=None,
            requested_documents=(),
        )
        validation = pipeline.validate_response(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=(_evidence("The revenue was $1,000,000 in 2024."),),
            calculation_result=None,
        )
        sources = ({"filename": "annual_report.pdf", "page": 12},)
        grounded = pipeline.build_grounded_response(
            answer="The revenue was $1,000,000 in 2024 [1].",
            sources=sources,
            answerability=answerability,
            validation=validation,
        )
        assert isinstance(grounded, GroundedResponseResult)
        assert grounded.answerability is not None
        assert grounded.validation is not None
        assert grounded.sources == sources

    def test_build_grounded_response_with_warnings(self):
        pipeline = GroundedValidationPipeline()
        grounded = pipeline.build_grounded_response(
            answer="Partial answer.",
            sources=(),
            answerability=None,
            validation=None,
            warnings=("Some requested documents were not found.",),
        )
        assert grounded.warnings == ("Some requested documents were not found.",)

    def test_build_grounded_response_public_dict(self):
        pipeline = GroundedValidationPipeline()
        answerability = pipeline.evaluate_answerability(
            question="What is the revenue?",
            intent="financial_calculation",
            evidence=(_evidence("Revenue was $1M."),),
            sufficiency_result=SufficiencyResult(
                is_sufficient=True, best_score=0.9, average_score=0.85
            ),
            calculation_result=None,
            requested_documents=(),
        )
        validation = pipeline.validate_response(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=(_evidence("The revenue was $1,000,000 in 2024."),),
            calculation_result=None,
        )
        grounded = pipeline.build_grounded_response(
            answer="The revenue was $1,000,000 in 2024 [1].",
            sources=({"filename": "annual_report.pdf", "page": 12},),
            answerability=answerability,
            validation=validation,
        )
        public = grounded.to_public_dict()
        assert "answer" in public
        assert "sources" in public
        assert "answerability" in public
        assert "validation" in public
        # Answerability public dict excludes scores.
        assert "best_score" not in public["answerability"]
        assert "average_score" not in public["answerability"]
        # Validation public dict excludes internal messages.
        for issue in public["validation"].get("issues", []):
            assert "message" not in issue


class TestGroundedValidationPipelineDeterminism:
    """Determinism: same inputs produce same outputs."""

    def test_answerability_deterministic(self):
        pipeline = GroundedValidationPipeline()
        kwargs = dict(
            question="What is the revenue?",
            intent="financial_calculation",
            evidence=(_evidence("Revenue was $1M."),),
            sufficiency_result=SufficiencyResult(
                is_sufficient=True, best_score=0.9, average_score=0.85
            ),
            calculation_result=None,
            requested_documents=(),
        )
        r1 = pipeline.evaluate_answerability(**kwargs)
        r2 = pipeline.evaluate_answerability(**kwargs)
        assert r1.status == r2.status
        assert r1.reason_codes == r2.reason_codes
        assert r1.evidence_count == r2.evidence_count

    def test_validation_deterministic(self):
        pipeline = GroundedValidationPipeline()
        kwargs = dict(
            answer="The revenue was $1,000,000 in 2024 [1].",
            intent="financial_calculation",
            evidence=(_evidence("The revenue was $1,000,000 in 2024."),),
            calculation_result=None,
        )
        r1 = pipeline.validate_response(**kwargs)
        r2 = pipeline.validate_response(**kwargs)
        assert r1.status == r2.status
        assert len(r1.issues) == len(r2.issues)


# ---------------------------------------------------------------------------
# Helper: extract claims via the real ClaimExtractor
# ---------------------------------------------------------------------------

def _extract_claims(answer: str):
    """Extract claims using the real ClaimExtractor (integration)."""
    from src.validation.claim_extractor import ClaimExtractor

    return ClaimExtractor().extract(answer)
