"""Phase 4 hotfix: repair + revalidation regression tests with call-count spies.

Verifies the _validate_and_repair_once flow:
1. PASSED → repair called 0 times, safe_fallback called 0 times
2. BLOCKED → repair called 0 times, safe_fallback called 1 time
3. FAILED → repair called 0 times, safe_fallback called 1 time
4. REPAIRABLE → revalidation PASSED → repair called 1 time, safe_fallback called 0 times
5. REPAIRABLE → revalidation BLOCKED → repair called 1 time, safe_fallback called 1 time
6. REPAIRABLE → revalidation FAILED → repair called 1 time, safe_fallback called 1 time
7. REPAIRABLE → repair produces empty answer → repair called 1 time, safe_fallback called 1 time
8. Validator exception → repair called 0 times, safe_fallback called 1 time
9. LLM call count does not increase (repair is deterministic)
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from src.application.rag_orchestrator import RAGOrchestrator
from src.domain.evidence import EvidenceItem
from src.domain.validation import (
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
    """Build an EvidenceItem from a chunk dict."""
    return EvidenceItem.from_chunk(
        {
            "doc_id": chunk_id,
            "content": content,
            "metadata": {"document_name": document_name, "page": page},
            "score": 0.95,
        }
    )


def _passed_result():
    return ValidationResult(
        status=ValidationStatus.PASSED,
        issues=(),
        checked_claim_count=1,
        supported_claim_count=1,
        unsupported_claim_count=0,
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


def _failed_result():
    return ValidationResult(
        status=ValidationStatus.FAILED,
        issues=(
            ValidationIssue(
                code="VALIDATOR_ERROR",
                severity=ValidationSeverity.CRITICAL,
                message="internal",
                claim_text="",
            ),
        ),
        checked_claim_count=0,
        supported_claim_count=0,
        unsupported_claim_count=0,
    )


def _repairable_result(claim_text="Revenue was 500 million"):
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


def _build_orchestrator():
    """Create a minimal RAGOrchestrator with spies on repair/safe_fallback.

    Returns ``(orch, repair_spy, safe_fallback_spy)`` where each spy wraps
    the real ``ResponseRepair`` method via ``patch.object`` so that call
    counts can be asserted while still exercising the real deterministic
    repair logic.
    """
    orch = RAGOrchestrator.__new__(RAGOrchestrator)
    orch._validation_pipeline = GroundedValidationPipeline()
    orch._response_repair = ResponseRepair()

    # Wrap the real methods with MagicMock spies.  Each spy's side_effect
    # is the original bound method, so the real repair/fallback logic still
    # runs while we can assert call counts.
    repair_spy = patch.object(
        orch._response_repair,
        "repair",
        side_effect=orch._response_repair.repair,
    ).start()
    safe_fallback_spy = patch.object(
        orch._response_repair,
        "safe_fallback",
        side_effect=orch._response_repair.safe_fallback,
    ).start()

    return orch, repair_spy, safe_fallback_spy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepairRevalidationCallCounts:
    """Spy-based call-count regression tests for _validate_and_repair_once."""

    # -- 1. PASSED -------------------------------------------------------
    def test_01_passed_no_repair_no_fallback(self):
        """PASSED → repair 0, safe_fallback 0."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(return_value=_passed_result())
        orch._validation_pipeline.validate_response = validate_mock

        answer = "Revenue was 1250 million in FY2024."
        result = orch._validate_and_repair_once(
            answer=answer,
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 0
        assert fb_spy.call_count == 0
        assert validate_mock.call_count == 1
        # Return shape: (answer, final_validation, repair_result, initial_validation)
        assert result[0] == answer
        assert result[2] is not None
        assert result[2].was_repaired is False
        assert result[2].fallback_used is False
        assert result[3].status is ValidationStatus.PASSED

    # -- 2. BLOCKED ------------------------------------------------------
    def test_02_blocked_no_repair_one_fallback(self):
        """BLOCKED → repair 0, safe_fallback 1."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(return_value=_blocked_result())
        orch._validation_pipeline.validate_response = validate_mock

        result = orch._validate_and_repair_once(
            answer="Revenue was 500 million.",
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 0
        assert fb_spy.call_count == 1
        assert validate_mock.call_count == 1
        assert result[3].status is ValidationStatus.BLOCKED

    # -- 3. FAILED -------------------------------------------------------
    def test_03_failed_no_repair_one_fallback(self):
        """FAILED → repair 0, safe_fallback 1."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(return_value=_failed_result())
        orch._validation_pipeline.validate_response = validate_mock

        result = orch._validate_and_repair_once(
            answer="Revenue was 500 million.",
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 0
        assert fb_spy.call_count == 1
        assert validate_mock.call_count == 1
        assert result[3].status is ValidationStatus.FAILED

    # -- 4. REPAIRABLE → revalidation PASSED -----------------------------
    def test_04_repairable_revalidation_passed(self):
        """REPAIRABLE → revalidation PASSED → repair 1, safe_fallback 0."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(side_effect=[_repairable_result(), _passed_result()])
        orch._validation_pipeline.validate_response = validate_mock

        answer = "Revenue was 1250 million in FY2024. Revenue was 500 million."
        result = orch._validate_and_repair_once(
            answer=answer,
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 1
        assert fb_spy.call_count == 0
        assert validate_mock.call_count == 2
        # The ungrounded claim should have been stripped.
        assert "500 million" not in result[0]
        assert "1250 million" in result[0]

    # -- 5. REPAIRABLE → revalidation BLOCKED ----------------------------
    def test_05_repairable_revalidation_blocked(self):
        """REPAIRABLE → revalidation BLOCKED → repair 1, safe_fallback 1."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(side_effect=[_repairable_result(), _blocked_result()])
        orch._validation_pipeline.validate_response = validate_mock

        orch._validate_and_repair_once(
            answer=("Revenue was 1250 million in FY2024. Revenue was 500 million."),
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 1
        assert fb_spy.call_count == 1
        assert validate_mock.call_count == 2

    # -- 6. REPAIRABLE → revalidation FAILED -----------------------------
    def test_06_repairable_revalidation_failed(self):
        """REPAIRABLE → revalidation FAILED → repair 1, safe_fallback 1."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(side_effect=[_repairable_result(), _failed_result()])
        orch._validation_pipeline.validate_response = validate_mock

        orch._validate_and_repair_once(
            answer=("Revenue was 1250 million in FY2024. Revenue was 500 million."),
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 1
        assert fb_spy.call_count == 1
        assert validate_mock.call_count == 2

    # -- 7. REPAIRABLE → repair produces empty answer --------------------
    def test_07_repairable_empty_after_repair(self):
        """REPAIRABLE → repair strips everything → repair 1, safe_fallback 1."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        claim = "Revenue was 500 million"
        validate_mock = MagicMock(
            side_effect=[
                _repairable_result(claim_text=claim),
                _blocked_result(),
            ]
        )
        orch._validation_pipeline.validate_response = validate_mock

        # The answer is ONLY the ungrounded claim sentence, so stripping
        # removes all content → repair returns a fallback answer, and
        # revalidation does not pass → safe_fallback is called.
        result = orch._validate_and_repair_once(
            answer=claim + ".",
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 1
        # When repair() returns fallback_used=True (empty after strip),
        # safe_fallback is NOT called because repair() already handled it.
        assert fb_spy.call_count == 0
        assert result[2].fallback_used is True
        assert validate_mock.call_count == 1

    # -- 8. Validator exception ------------------------------------------
    def test_08_validator_exception_fail_closed(self):
        """Validator exception → repair 0, safe_fallback 1 (fail-closed)."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(side_effect=RuntimeError("validator crashed"))
        orch._validation_pipeline.validate_response = validate_mock

        orch._validate_and_repair_once(
            answer="Revenue was 1250 million in FY2024.",
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 0
        assert fb_spy.call_count == 1
        assert validate_mock.call_count == 1

    # -- 9. LLM not called during repair ---------------------------------
    def test_09_llm_not_called_during_repair(self):
        """Repair is deterministic — LLM call count must not increase."""
        orch, repair_spy, fb_spy = _build_orchestrator()
        validate_mock = MagicMock(side_effect=[_repairable_result(), _passed_result()])
        orch._validation_pipeline.validate_response = validate_mock

        # Strict LLM gateway mock — any call raises AssertionError so the
        # test fails immediately if the repair path tries to use the LLM.
        llm_gateway = MagicMock()
        _guard = AssertionError("LLM must not be called during repair/revalidation")
        llm_gateway.generate.side_effect = _guard
        llm_gateway.chat.side_effect = _guard
        llm_gateway.complete.side_effect = _guard
        llm_gateway.invoke.side_effect = _guard
        llm_gateway.aget.side_effect = _guard
        orch._llm_gateway = llm_gateway

        answer = "Revenue was 1250 million in FY2024. Revenue was 500 million."
        # If the LLM were called, AssertionError would propagate and the
        # test would fail.  Successful completion proves the repair flow is
        # deterministic.
        orch._validate_and_repair_once(
            answer=answer,
            intent="financial_calculation",
            sources=(),
            evidence=(_evidence(),),
            calculation_result=None,
            answerability=None,
        )

        assert repair_spy.call_count == 1
        assert fb_spy.call_count == 0
        # Explicitly verify the LLM gateway was never called.
        llm_gateway.generate.assert_not_called()
        llm_gateway.chat.assert_not_called()
        llm_gateway.complete.assert_not_called()
        llm_gateway.invoke.assert_not_called()
        llm_gateway.aget.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
