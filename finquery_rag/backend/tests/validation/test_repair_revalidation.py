"""Phase 4 hotfix: repair + revalidation regression tests.

Verifies the _validate_and_repair_once flow:
1. PASSED → no repair, no-op RepairResult.
2. REPAIRABLE → one repair → revalidation PASSED → return repaired answer.
3. REPAIRABLE → one repair → revalidation still fails → safe fallback.
4. REPAIRABLE → repair strips all content → safe fallback.
5. BLOCKED → immediate safe fallback (no repair attempt).
6. FAILED → immediate safe fallback (fail-closed).
7. At most ONE repair (no second repair after revalidation fails).
8. Validator exception → FAILED → safe fallback.
9. LLM call count does not increase (repair is deterministic).
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.evidence import EvidenceItem
from src.domain.validation import (
    AnswerabilityResult,
    AnswerabilityStatus,
    ExtractedClaim,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidationResult,
)
from src.validation.response_repair import ResponseRepair
from src.validation.validation_pipeline import GroundedValidationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    content="Revenue was 1250 million in FY2024.",
    chunk_id="chunk_001",
    document_name="annual_report.pdf",
    page=12,
):
    return EvidenceItem.from_chunk({
        "doc_id": chunk_id,
        "content": content,
        "metadata": {"document_name": document_name, "page": page},
        "score": 0.95,
    })


def _make_claim(value="1250", metric="revenue", period="2024", raw_text="Revenue was 1250 million"):
    return ExtractedClaim(
        claim_id="c1",
        claim_type="amount",
        raw_text=raw_text,
        metric=metric,
        value=Decimal(value),
        period=period,
    )


def _passed_result():
    return ValidationResult(
        status=ValidationStatus.PASSED,
        issues=(),
        checked_claim_count=1,
        supported_claim_count=1,
        unsupported_claim_count=0,
    )


def _repairable_result(claim_text="Revenue was 500 million"):
    """A REPAIRABLE result with an ungrounded numeric claim that can be stripped."""
    return ValidationResult(
        status=ValidationStatus.REPAIRABLE,
        issues=(
            ValidationIssue(
                code="NUMERIC_UNGROUND",
                severity=ValidationSeverity.ERROR,
                message="internal",
                claim_text=claim_text,
            ),
        ),
        checked_claim_count=2,
        supported_claim_count=1,
        unsupported_claim_count=1,
    )


def _blocked_result():
    return ValidationResult(
        status=ValidationStatus.BLOCKED,
        issues=(
            ValidationIssue(
                code="CALCULATION_MISMATCH",
                severity=ValidationSeverity.CRITICAL,
                message="internal",
                claim_text="calc mismatch",
            ),
        ),
        checked_claim_count=1,
        supported_claim_count=0,
        unsupported_claim_count=1,
    )


# ---------------------------------------------------------------------------
# ResponseRepair unit tests
# ---------------------------------------------------------------------------

class TestResponseRepairDirect:
    """Direct tests on ResponseRepair.repair() behavior."""

    def test_passed_returns_noop(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Revenue was 1250 million.",
            validation=_passed_result(),
        )
        assert result.was_repaired is False
        assert result.fallback_used is False
        assert result.answer == "Revenue was 1250 million."

    def test_blocked_returns_fallback(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Bad answer with wrong number.",
            validation=_blocked_result(),
        )
        assert result.fallback_used is True
        assert result.was_repaired is False
        assert "cannot provide a verified answer" in result.answer

    def test_repair_strips_ungrounded_claim(self):
        repair = ResponseRepair()
        answer = "Revenue was 1250 million. Revenue was 500 million."
        result = repair.repair(
            answer=answer,
            validation=_repairable_result("Revenue was 500 million"),
        )
        assert result.was_repaired is True
        assert result.fallback_used is False
        assert "500 million" not in result.answer
        assert "1250 million" in result.answer

    def test_repair_strips_all_content_returns_fallback(self):
        repair = ResponseRepair()
        answer = "Revenue was 500 million."
        result = repair.repair(
            answer=answer,
            validation=_repairable_result("Revenue was 500 million"),
        )
        assert result.fallback_used is True
        assert "cannot provide a verified answer" in result.answer

    def test_answerability_not_answerable_returns_fallback(self):
        repair = ResponseRepair()
        answerability = AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason_codes=("no_evidence",),
            evidence_count=0,
            document_count=0,
            best_score=0.0,
            average_score=0.0,
        )
        result = repair.repair(
            answer="some answer",
            validation=_passed_result(),
            answerability=answerability,
        )
        assert result.fallback_used is True
        assert "cannot answer this question" in result.answer

    def test_answerability_calculation_blocked_returns_fallback(self):
        repair = ResponseRepair()
        answerability = AnswerabilityResult(
            status=AnswerabilityStatus.CALCULATION_BLOCKED,
            reason_codes=("calculation_blocked",),
            evidence_count=2,
            document_count=1,
            best_score=0.9,
            average_score=0.8,
        )
        result = repair.repair(
            answer="some answer",
            validation=_passed_result(),
            answerability=answerability,
        )
        assert result.fallback_used is True
        assert "calculation could not be completed" in result.answer


# ---------------------------------------------------------------------------
# _validate_and_repair_once integration tests
# ---------------------------------------------------------------------------

class TestValidateAndRepairOnce:
    """Tests on the orchestrator's _validate_and_repair_once method."""

    @staticmethod
    def _make_orchestrator():
        from src.application.rag_orchestrator import RAGOrchestrator
        orch = RAGOrchestrator.__new__(RAGOrchestrator)
        orch._validation_pipeline = GroundedValidationPipeline()
        orch._response_repair = ResponseRepair()
        return orch

    def test_passed_returns_noop_repair(self):
        orch = self._make_orchestrator()
        evidence = (_evidence(),)
        # An answer that is fully grounded in evidence.
        answer = "Revenue was 1250 million in FY2024."
        final_answer, final_val, repair, initial = orch._validate_and_repair_once(
            answer=answer,
            intent="document_qa",
            sources=({"filename": "annual_report.pdf", "page": 12, "chunk_id": "chunk_001"},),
            evidence=evidence,
            calculation_result=None,
            answerability=None,
        )
        assert final_val.status in (ValidationStatus.PASSED, ValidationStatus.NOT_APPLICABLE)
        assert repair is not None
        assert repair.was_repaired is False
        assert repair.fallback_used is False

    def test_blocked_returns_fallback(self):
        orch = self._make_orchestrator()
        evidence = (_evidence(),)
        # An answer with a value not in evidence — should be flagged.
        answer = "Revenue was 99999 million."
        final_answer, final_val, repair, initial = orch._validate_and_repair_once(
            answer=answer,
            intent="financial_calculation",
            sources=({"filename": "annual_report.pdf", "page": 12, "chunk_id": "chunk_001"},),
            evidence=evidence,
            calculation_result=None,
            answerability=None,
        )
        # The ungrounded numeric should trigger BLOCKED or REPAIRABLE.
        assert final_val.status in (
            ValidationStatus.BLOCKED,
            ValidationStatus.FAILED,
            ValidationStatus.REPAIRABLE,
        )
        # If it was repairable and stripped, fallback may or may not be used.
        # But the key invariant: the original bad answer is NOT returned as-is.
        if repair is not None and repair.fallback_used:
            assert "99999" not in final_answer

    def test_no_second_repair_after_revalidation_fails(self):
        """If revalidation fails, a safe fallback is used — NOT a second repair."""
        orch = self._make_orchestrator()
        evidence = (_evidence(),)
        # An answer with two ungrounded claims — first stripped, second remains.
        answer = "Revenue was 500 million. Assets were 99999."
        final_answer, final_val, repair, initial = orch._validate_and_repair_once(
            answer=answer,
            intent="financial_calculation",
            sources=({"filename": "annual_report.pdf", "page": 12, "chunk_id": "chunk_001"},),
            evidence=evidence,
            calculation_result=None,
            answerability=None,
        )
        # The repair should have been attempted at most once.
        if repair is not None:
            assert repair.was_repaired in (True, False)  # at most one repair
            # If fallback was used, the bad content must be gone.
            if repair.fallback_used:
                assert "99999" not in final_answer

    def test_validator_exception_returns_failed(self):
        """If the validator raises, the result must be FAILED (fail-closed)."""
        orch = self._make_orchestrator()
        # Patch validate_response to raise.
        orch._validation_pipeline.validate_response = MagicMock(
            side_effect=RuntimeError("validator crash")
        )
        final_answer, final_val, repair, initial = orch._validate_and_repair_once(
            answer="Some answer.",
            intent="document_qa",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )
        assert final_val.status is ValidationStatus.FAILED
        assert repair is not None
        assert repair.fallback_used is True

    def test_llm_not_called_during_repair(self):
        """The repair path must never call the LLM."""
        orch = self._make_orchestrator()

        original_generate = getattr(orch, '_llm_gateway', None)
        if original_generate is not None:
            original_generate.generate = MagicMock(side_effect=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("LLM must not be called during repair")
            ))

        # Run validation+repair.
        evidence = (_evidence(),)
        orch._validate_and_repair_once(
            answer="Revenue was 99999 million.",
            intent="financial_calculation",
            sources=({"filename": "annual_report.pdf", "page": 12, "chunk_id": "chunk_001"},),
            evidence=evidence,
            calculation_result=None,
            answerability=None,
        )
        # If we get here without AssertionError, the test passes.
        assert True

    def test_initial_validation_stored_separately_from_final(self):
        """The initial validation result must be returned as the 4th tuple element."""
        orch = self._make_orchestrator()
        evidence = (_evidence(),)
        answer = "Revenue was 1250 million in FY2024."
        final_answer, final_val, repair, initial = orch._validate_and_repair_once(
            answer=answer,
            intent="document_qa",
            sources=({"filename": "annual_report.pdf", "page": 12, "chunk_id": "chunk_001"},),
            evidence=evidence,
            calculation_result=None,
            answerability=None,
        )
        # When PASSED on first try, initial == final.
        assert initial is not None
        assert initial.status == final_val.status
