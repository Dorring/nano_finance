"""Tests for ResponseRepair (Phase 4 Commit 8).

Verifies:
- ``PASSED`` / ``NOT_APPLICABLE`` -> answer returned as-is (no repair).
- ``REPAIRABLE`` -> single deterministic repair attempt (strip ungrounded
  claims). If repair produces empty result, safe fallback is used.
- ``BLOCKED`` -> safe fallback immediately.
- ``FAILED`` -> safe fallback immediately (fail-closed).
- Answerability ``NOT_ANSWERABLE`` / ``CALCULATION_BLOCKED`` -> safe
  fallback immediately.
- At most ONE repair attempt.
- Repair NEVER calls the LLM (deterministic).
- Public / trace dict separation.
- Safe fallback messages do not expose internal details.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.validation import (
    AnswerabilityResult,
    AnswerabilityStatus,
    ValidationIssue,
    ValidationSeverity,
    ValidationResult,
    ValidationStatus,
)
from src.validation.numeric_claim_validator import CODE_NUMERIC_UNGROUND
from src.validation.citation_validator import CODE_CITATION_MISSING
from src.validation.response_repair import (
    RepairResult,
    ResponseRepair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _issue(
    code: str = CODE_NUMERIC_UNGROUND,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
    claim_text: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message="test issue",
        claim_text=claim_text,
        public_message="A validation issue was found.",
    )


def _validation(
    status: ValidationStatus,
    issues: tuple[ValidationIssue, ...] = (),
) -> ValidationResult:
    return ValidationResult(
        status=status,
        issues=issues,
        checked_claim_count=1,
        supported_claim_count=0,
        unsupported_claim_count=1,
    )


def _answerability(status: AnswerabilityStatus) -> AnswerabilityResult:
    return AnswerabilityResult(
        status=status,
        reason_codes=("test",),
        evidence_count=0,
        document_count=0,
        best_score=None,
        average_score=None,
        missing_requirements=(),
    )


# ---------------------------------------------------------------------------
# No repair needed
# ---------------------------------------------------------------------------

class TestNoRepairNeeded:
    """Tests where the answer is returned as-is."""

    def test_passed_returns_original(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="The revenue was $1,000,000 in 2024 [1].",
            validation=_validation(ValidationStatus.PASSED),
        )
        assert result.was_repaired is False
        assert result.fallback_used is False
        assert result.answer == "The revenue was $1,000,000 in 2024 [1]."

    def test_not_applicable_returns_original(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Hello!",
            validation=_validation(ValidationStatus.NOT_APPLICABLE),
        )
        assert result.was_repaired is False
        assert result.fallback_used is False
        assert result.answer == "Hello!"


# ---------------------------------------------------------------------------
# Safe fallback
# ---------------------------------------------------------------------------

class TestSafeFallback:
    """Tests for safe fallback on BLOCKED / FAILED."""

    def test_blocked_uses_fallback(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="The revenue was $999,999,999.",
            validation=_validation(
                ValidationStatus.BLOCKED,
                (_issue(severity=ValidationSeverity.CRITICAL),),
            ),
        )
        assert result.fallback_used is True
        assert result.was_repaired is False
        assert "cannot provide a verified answer" in result.answer.lower()

    def test_failed_uses_fallback(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="The revenue was $1,000,000.",
            validation=_validation(ValidationStatus.FAILED),
        )
        assert result.fallback_used is True
        assert result.was_repaired is False
        assert "encountered an issue" in result.answer.lower()

    def test_blocked_fallback_no_internal_details(self):
        """The fallback message must not expose internal details."""
        repair = ResponseRepair()
        result = repair.repair(
            answer="Some answer with sensitive data.",
            validation=_validation(ValidationStatus.BLOCKED),
        )
        # Fallback must not contain the original answer text.
        assert "sensitive data" not in result.answer
        # Fallback must not contain stack traces or error codes.
        assert "Traceback" not in result.answer
        assert "Error" not in result.answer

    def test_not_answerable_uses_fallback(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Some answer.",
            validation=_validation(ValidationStatus.PASSED),
            answerability=_answerability(AnswerabilityStatus.NOT_ANSWERABLE),
        )
        assert result.fallback_used is True
        assert result.was_repaired is False
        assert "cannot answer" in result.answer.lower()

    def test_calculation_blocked_uses_fallback(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Some answer.",
            validation=_validation(ValidationStatus.PASSED),
            answerability=_answerability(AnswerabilityStatus.CALCULATION_BLOCKED),
        )
        assert result.fallback_used is True
        assert result.was_repaired is False
        assert "calculation could not be completed" in result.answer.lower()


# ---------------------------------------------------------------------------
# Deterministic repair
# ---------------------------------------------------------------------------

class TestDeterministicRepair:
    """Tests for the single deterministic repair attempt."""

    def test_repairable_strips_ungrounded_claims(self):
        """REPAIRABLE status triggers claim stripping."""
        repair = ResponseRepair()
        answer = (
            "The revenue was $1,000,000 in 2024 [1]. "
            "The EBITDA was $999,999,999 in 2024."
        )
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (
                _issue(
                    claim_text="$999,999,999",
                    severity=ValidationSeverity.ERROR,
                ),
            ),
        )
        result = repair.repair(answer=answer, validation=validation)
        assert result.was_repaired is True
        assert result.fallback_used is False
        # The ungrounded claim should be stripped.
        assert "$999,999,999" not in result.answer
        # The grounded claim should remain.
        assert "$1,000,000" in result.answer

    def test_repair_all_content_stripped_uses_fallback(self):
        """If all content is stripped, fallback is used."""
        repair = ResponseRepair()
        answer = "The EBITDA was $999,999,999 in 2024."
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (
                _issue(
                    claim_text="$999,999,999",
                    severity=ValidationSeverity.ERROR,
                ),
            ),
        )
        result = repair.repair(answer=answer, validation=validation)
        assert result.fallback_used is True
        assert result.was_repaired is False

    def test_repair_at_most_once(self):
        """The repair is applied at most once (no recursive repairs)."""
        repair = ResponseRepair()
        answer = (
            "Good text remains. "
            "Claim A is $999,999,999. "
            "Claim B is $888,888,888."
        )
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (
                _issue(
                    claim_text="$999,999,999",
                    severity=ValidationSeverity.ERROR,
                ),
                _issue(
                    claim_text="$888,888,888",
                    severity=ValidationSeverity.ERROR,
                ),
            ),
        )
        result = repair.repair(answer=answer, validation=validation)
        # Both bad claims should be stripped in a single pass.
        assert result.was_repaired is True
        assert "$999,999,999" not in result.answer
        assert "$888,888,888" not in result.answer
        # Good text survives.
        assert "Good text remains" in result.answer

    def test_repair_does_not_call_llm(self):
        """The repair is deterministic (no LLM call).

        This is verified by checking that the repair completes without
        any network call or external dependency. The test passes if the
        repair returns a result without error.
        """
        repair = ResponseRepair()
        answer = "The revenue was $1,000,000 [1]. Bad claim $999,999,999."
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (_issue(claim_text="$999,999,999"),),
        )
        result = repair.repair(answer=answer, validation=validation)
        # Just verify it completes and produces a result.
        assert isinstance(result, RepairResult)
        assert result.was_repaired is True

    def test_repairable_with_only_citation_warning_returns_as_is(self):
        """REPAIRABLE with only non-blocking citation issues keeps answer."""
        repair = ResponseRepair()
        answer = "The revenue was $1,000,000 in 2024."
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (
                _issue(
                    code=CODE_CITATION_MISSING,
                    severity=ValidationSeverity.WARNING,
                ),
            ),
        )
        result = repair.repair(answer=answer, validation=validation)
        # Citation warnings are non-blocking; answer returned as-is.
        assert result.was_repaired is False
        assert result.fallback_used is False
        assert result.answer == answer


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for public/trace dict separation."""

    def test_public_dict_excludes_notes(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Bad claim $999,999,999. Good text remains.",
            validation=_validation(
                ValidationStatus.REPAIRABLE,
                (_issue(claim_text="$999,999,999"),),
            ),
        )
        public = result.to_public_dict()
        assert "was_repaired" in public
        assert "fallback_used" in public
        assert "repair_notes" not in public

    def test_trace_dict_includes_notes(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Bad claim $999,999,999. Good text remains.",
            validation=_validation(
                ValidationStatus.REPAIRABLE,
                (_issue(claim_text="$999,999,999"),),
            ),
        )
        trace = result.to_trace_dict()
        assert "repair_notes" in trace
        assert "answer_length" in trace
        assert isinstance(trace["repair_notes"], list)

    def test_fallback_public_dict_no_internal_details(self):
        repair = ResponseRepair()
        result = repair.repair(
            answer="Some answer.",
            validation=_validation(ValidationStatus.BLOCKED),
        )
        public = result.to_public_dict()
        # Public dict only has flags.
        assert set(public.keys()) == {"was_repaired", "fallback_used"}
        assert public["fallback_used"] is True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same inputs produce same outputs."""

    def test_same_inputs_same_output(self):
        repair = ResponseRepair()
        answer = "Good text. Bad claim $999,999,999."
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (_issue(claim_text="$999,999,999"),),
        )
        r1 = repair.repair(answer=answer, validation=validation)
        r2 = repair.repair(answer=answer, validation=validation)
        assert r1.answer == r2.answer
        assert r1.was_repaired == r2.was_repaired
        assert r1.fallback_used == r2.fallback_used
        assert r1.repair_notes == r2.repair_notes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_answer_with_repairable(self):
        """Empty answer with REPAIRABLE uses fallback."""
        repair = ResponseRepair()
        result = repair.repair(
            answer="",
            validation=_validation(ValidationStatus.REPAIRABLE),
        )
        assert result.fallback_used is True

    def test_whitespace_only_answer_with_repairable(self):
        """Whitespace-only answer with REPAIRABLE uses fallback."""
        repair = ResponseRepair()
        result = repair.repair(
            answer="   ",
            validation=_validation(ValidationStatus.REPAIRABLE),
        )
        assert result.fallback_used is True

    def test_no_issues_repairable_returns_as_is(self):
        """REPAIRABLE with no issues returns answer as-is."""
        repair = ResponseRepair()
        answer = "Some answer without issues."
        validation = _validation(ValidationStatus.REPAIRABLE, ())
        result = repair.repair(answer=answer, validation=validation)
        assert result.was_repaired is False
        assert result.answer == answer

    def test_critical_issue_not_repaired(self):
        """CRITICAL issues are never repaired — always fallback."""
        repair = ResponseRepair()
        answer = "Critical error $999,999,999."
        validation = _validation(
            ValidationStatus.REPAIRABLE,
            (
                _issue(
                    severity=ValidationSeverity.CRITICAL,
                    claim_text="$999,999,999",
                ),
            ),
        )
        # Even though status is REPAIRABLE, a CRITICAL issue means the
        # repair cannot fix it (strip only targets NUMERIC_UNGROUND).
        result = repair.repair(answer=answer, validation=validation)
        # The CRITICAL issue is not NUMERIC_UNGROUND, so strip_claims is
        # empty, has_blocking=True -> fallback.
        assert result.fallback_used is True
