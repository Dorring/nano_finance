"""Tests for the pre-generation AnswerabilityEvaluator (Phase 4 Commit 4).

Verifies all four answerability verdicts and their reason codes, the
public/trace serialization boundary, and the deterministic nature of the
evaluator (no LLM, no retrieval, no side effects).
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
from src.domain.validation import AnswerabilityStatus
from src.retrieval.context_builder import SufficiencyResult
from src.validation.answerability import (
    AnswerabilityEvaluator,
    REASON_CALCULATION_BLOCKED,
    REASON_CALCULATION_FAILED,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_MISSING_DOCUMENTS,
    REASON_NO_EVIDENCE,
    REASON_NO_RETRIEVAL_REQUIRED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
    page: int = 12,
    score: float = 0.9,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content="Revenue was $1,000,000 for FY2025.",
        document_name=document_name,
        page=page,
        content_type="text",
        score=score,
        rerank_score=None,
        metadata={},
    )


def _sufficient() -> SufficiencyResult:
    return SufficiencyResult(is_sufficient=True, best_score=0.9, average_score=0.85)


def _insufficient() -> SufficiencyResult:
    return SufficiencyResult(is_sufficient=False, best_score=0.01, average_score=0.005)


def _calc_result(status: CalculationStatus) -> CalculationResult:
    return CalculationResult(
        status=status,
        operation=CalculationOperation.GROSS_MARGIN,
        error_code="MISSING_OPERANDS" if status is CalculationStatus.BLOCKED else "INTERNAL_ERROR",
        error_message="internal detail" if status is CalculationStatus.FAILED else None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalculationBlocked:
    """CALCULATION_BLOCKED: LLM must not be invoked."""

    def test_blocked_calculation(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="Calculate gross margin",
            intent="financial_calculation",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=_calc_result(CalculationStatus.BLOCKED),
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.CALCULATION_BLOCKED
        assert REASON_CALCULATION_BLOCKED in result.reason_codes

    def test_failed_calculation(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="Calculate gross margin",
            intent="financial_calculation",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=_calc_result(CalculationStatus.FAILED),
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.CALCULATION_BLOCKED
        assert REASON_CALCULATION_FAILED in result.reason_codes

    def test_executed_calculation_does_not_block(self):
        """EXECUTED calculation should not force CALCULATION_BLOCKED."""
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="Calculate gross margin",
            intent="financial_calculation",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=CalculationResult(
                status=CalculationStatus.EXECUTED,
                operation=CalculationOperation.GROSS_MARGIN,
                value=Decimal("0.4"),
                unit="ratio",
            ),
            requested_documents=("annual_report.pdf",),
        )
        # EXECUTED does not trigger CALCULATION_BLOCKED; it falls through.
        assert result.status is not AnswerabilityStatus.CALCULATION_BLOCKED

    def test_not_applicable_calculation_does_not_block(self):
        """NOT_APPLICABLE calculation should not force CALCULATION_BLOCKED."""
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=CalculationResult(status=CalculationStatus.NOT_APPLICABLE),
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE


class TestConversationIntent:
    """Conversation intents require no evidence."""

    def test_conversation_answerable_without_evidence(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="Hello, what can you do?",
            intent="conversation",
            evidence=(),
            sufficiency_result=SufficiencyResult(is_sufficient=False, best_score=0.0, average_score=0.0),
            calculation_result=None,
            requested_documents=(),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE
        assert REASON_NO_RETRIEVAL_REQUIRED in result.reason_codes


class TestNotAnswerable:
    """NOT_ANSWERABLE: LLM must not be invoked."""

    def test_no_evidence(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(),
            sufficiency_result=SufficiencyResult(is_sufficient=False, best_score=0.0, average_score=0.0),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.NOT_ANSWERABLE
        assert REASON_NO_EVIDENCE in result.reason_codes
        assert result.evidence_count == 0
        assert result.document_count == 0
        assert len(result.missing_requirements) > 0

    def test_insufficient_evidence(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(score=0.001),),
            sufficiency_result=_insufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.NOT_ANSWERABLE
        assert REASON_INSUFFICIENT_EVIDENCE in result.reason_codes
        assert result.evidence_count == 1


class TestPartiallyAnswerable:
    """PARTIALLY_ANSWERABLE: some requested documents are missing."""

    def test_missing_documents(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="Compare revenue across reports",
            intent="multi_document_comparison",
            evidence=(_evidence(document_name="report_a.pdf"),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("report_a.pdf", "report_b.pdf"),
        )
        assert result.status is AnswerabilityStatus.PARTIALLY_ANSWERABLE
        assert REASON_MISSING_DOCUMENTS in result.reason_codes
        assert any("report_b.pdf" in m for m in result.missing_requirements)

    def test_all_documents_present_is_answerable(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(document_name="annual_report.pdf"),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE


class TestAnswerable:
    """ANSWERABLE: all checks pass."""

    def test_answerable_with_evidence(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(), _evidence(chunk_id="c2", page=13)),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE
        assert result.reason_codes == ()
        assert result.evidence_count == 2
        assert result.document_count == 1


class TestSerialization:
    """Public/trace dict separation."""

    def test_public_dict_excludes_scores(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        public = result.to_public_dict()
        assert "best_score" not in public
        assert "average_score" not in public
        assert public["status"] == "answerable"

    def test_trace_dict_includes_scores(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        trace = result.to_trace_dict()
        assert trace["best_score"] == 0.9
        assert trace["average_score"] == 0.85

    def test_public_dict_missing_requirements(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(),
            sufficiency_result=SufficiencyResult(is_sufficient=False, best_score=0.0, average_score=0.0),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        public = result.to_public_dict()
        assert "missing_requirements" in public
        assert any("annual_report.pdf" in m for m in public["missing_requirements"])


class TestDeterminism:
    """The evaluator must be deterministic — no LLM, no retrieval."""

    def test_same_inputs_same_output(self):
        ev = AnswerabilityEvaluator()
        kwargs = dict(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        r1 = ev.evaluate(**kwargs)
        r2 = ev.evaluate(**kwargs)
        assert r1 == r2

    def test_none_calculation_result_is_handled(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("annual_report.pdf",),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE

    def test_empty_requested_documents(self):
        """No requested documents means no PARTIALLY_ANSWERABLE."""
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(_evidence(),),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=(),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE


class TestDocumentCounting:
    """Document count should reflect unique document names."""

    def test_multiple_chunks_same_document(self):
        ev = AnswerabilityEvaluator()
        result = ev.evaluate(
            question="What is the revenue?",
            intent="document_qa",
            evidence=(
                _evidence(chunk_id="c1", document_name="a.pdf", page=1),
                _evidence(chunk_id="c2", document_name="a.pdf", page=2),
                _evidence(chunk_id="c3", document_name="b.pdf", page=1),
            ),
            sufficiency_result=_sufficient(),
            calculation_result=None,
            requested_documents=("a.pdf", "b.pdf"),
        )
        assert result.status is AnswerabilityStatus.ANSWERABLE
        assert result.document_count == 2
        assert result.evidence_count == 3
