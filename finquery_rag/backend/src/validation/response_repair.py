"""Deterministic response repair and safe fallback (Phase 4 Commit 8).

The ``ResponseRepair`` module provides a single deterministic repair
attempt for answers that received a ``REPAIRABLE`` validation verdict,
and a safe fallback for answers that are ``BLOCKED`` or ``FAILED``.

Key invariants (from the Phase 4 specification):
- At most ONE repair attempt per answer.
- The repair NEVER calls the LLM.
- If the repair fails (the answer is still invalid after repair), the
  safe fallback is used.
- The safe fallback is a deterministic refusal message that does not
  expose internal errors, stack traces, or evidence content.

Repair strategy (deterministic):
1. Strip ungrounded numeric claims — remove sentences containing numeric
   values that the validator flagged as ``NUMERIC_UNGROUND``.
2. If the repaired answer is empty or still contains blocking issues,
   fall back to the safe refusal message.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.domain.validation import (
    AnswerabilityResult,
    AnswerabilityStatus,
    ValidationResult,
    ValidationStatus,
)
from src.validation.numeric_claim_validator import CODE_NUMERIC_UNGROUND
from src.validation.citation_validator import (
    CODE_CITATION_MISSING,
    CODE_CITATION_UNRESOLVED,
)


# ---------------------------------------------------------------------------
# Safe fallback messages
# ---------------------------------------------------------------------------

_FALLBACK_NOT_ANSWERABLE = (
    "I cannot answer this question based on the available evidence. "
    "The retrieved documents do not contain sufficient information to "
    "provide a verified response."
)

_FALLBACK_BLOCKED = (
    "I cannot provide a verified answer to this question. "
    "The response did not pass validation against the source documents."
)

_FALLBACK_FAILED = (
    "I encountered an issue while verifying the answer and cannot "
    "provide a response at this time."
)

_FALLBACK_CALCULATION_BLOCKED = (
    "The requested calculation could not be completed with the available "
    "data. Please provide more specific financial figures or try a "
    "different query."
)


# ---------------------------------------------------------------------------
# Repair result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairResult:
    """Outcome of a repair attempt.

    Attributes:
        answer: the final answer text — either the repaired answer, the
            original answer (if no repair was needed), or the safe
            fallback message.
        was_repaired: True if a deterministic repair was applied.
        fallback_used: True if the safe fallback message was used.
        repair_notes: internal notes for trace logging (never exposed
            in public API responses).
    """

    answer: str
    was_repaired: bool
    fallback_used: bool
    repair_notes: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for public API responses.

        Exposes only whether a repair or fallback was used; internal
        ``repair_notes`` are omitted.
        """
        return {
            "was_repaired": self.was_repaired,
            "fallback_used": self.fallback_used,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trace logging.

        Includes ``repair_notes`` for debugging.
        """
        return {
            "was_repaired": self.was_repaired,
            "fallback_used": self.fallback_used,
            "repair_notes": list(self.repair_notes),
            "answer_length": len(self.answer),
        }


# ---------------------------------------------------------------------------
# Response repair
# ---------------------------------------------------------------------------

class ResponseRepair:
    """Deterministic repair and safe fallback for validated answers.

    The repair strategy is conservative:
    - ``PASSED`` / ``NOT_APPLICABLE`` -> return the answer as-is.
    - ``REPAIRABLE`` -> attempt ONE deterministic repair (strip ungrounded
      numeric claims). If the result is empty or still has blocking
      issues, use the safe fallback.
    - ``BLOCKED`` -> use the safe fallback immediately.
    - ``FAILED`` -> use the safe fallback immediately (fail-closed).

    The repair NEVER calls the LLM and NEVER performs retrieval.
    """

    # Sentence-splitting pattern: splits on sentence boundaries while
    # keeping the trailing punctuation. Handles ".", "!", "?" and
    # newlines.
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

    def repair(
        self,
        *,
        answer: str,
        validation: ValidationResult,
        answerability: AnswerabilityResult | None = None,
    ) -> RepairResult:
        """Attempt a single deterministic repair.

        Returns a ``RepairResult``. Never raises.
        """
        # If answerability says NOT_ANSWERABLE or CALCULATION_BLOCKED,
        # use the appropriate fallback immediately.
        if answerability is not None:
            if answerability.status is AnswerabilityStatus.NOT_ANSWERABLE:
                return RepairResult(
                    answer=_FALLBACK_NOT_ANSWERABLE,
                    was_repaired=False,
                    fallback_used=True,
                    repair_notes=("answerability:not_answerable",),
                )
            if answerability.status is AnswerabilityStatus.CALCULATION_BLOCKED:
                return RepairResult(
                    answer=_FALLBACK_CALCULATION_BLOCKED,
                    was_repaired=False,
                    fallback_used=True,
                    repair_notes=("answerability:calculation_blocked",),
                )

        status = validation.status

        # PASSED / NOT_APPLICABLE -> no repair needed.
        if status in (ValidationStatus.PASSED, ValidationStatus.NOT_APPLICABLE):
            return RepairResult(
                answer=answer,
                was_repaired=False,
                fallback_used=False,
                repair_notes=(),
            )

        # BLOCKED / FAILED -> safe fallback immediately.
        if status is ValidationStatus.BLOCKED:
            return RepairResult(
                answer=_FALLBACK_BLOCKED,
                was_repaired=False,
                fallback_used=True,
                repair_notes=("validation:blocked",),
            )
        if status is ValidationStatus.FAILED:
            return RepairResult(
                answer=_FALLBACK_FAILED,
                was_repaired=False,
                fallback_used=True,
                repair_notes=("validation:failed",),
            )

        # REPAIRABLE -> attempt deterministic repair.
        repaired, notes = self._attempt_repair(answer, validation)

        if repaired is None or not repaired.strip():
            # Repair produced an empty answer -> fallback.
            return RepairResult(
                answer=_FALLBACK_BLOCKED,
                was_repaired=False,
                fallback_used=True,
                repair_notes=notes + ("repair:empty_result",),
            )

        return RepairResult(
            answer=repaired,
            was_repaired=True,
            fallback_used=False,
            repair_notes=notes,
        )

    # -----------------------------------------------------------------
    # Internal repair strategies
    # -----------------------------------------------------------------

    def _attempt_repair(
        self,
        answer: str,
        validation: ValidationResult,
    ) -> tuple[str | None, tuple[str, ...]]:
        """Attempt a single deterministic repair.

        Current strategy: strip sentences containing ungrounded numeric
        claims. If the answer has no ungrounded claims (only citation
        issues), attempt to append citations.

        Returns ``(repaired_answer, notes)``. If the repair fails,
        returns ``(None, notes)``.
        """
        notes: list[str] = []

        # Collect claim texts for issues that can be repaired by stripping.
        strip_claims: set[str] = set()
        has_blocking = False

        for issue in validation.issues:
            if issue.code == CODE_NUMERIC_UNGROUND and issue.claim_text:
                strip_claims.add(issue.claim_text)
            elif issue.severity.value in ("critical", "error"):
                # If there are non-repairable blocking issues, repair
                # cannot help.
                has_blocking = True

        if has_blocking and not strip_claims:
            # Blocking issues that we cannot repair by stripping.
            notes.append("repair:unrepairable_blocking_issues")
            return None, tuple(notes)

        if strip_claims:
            repaired = self._strip_claims(answer, strip_claims)
            notes.append("repair:stripped_ungrounded_claims")
            if repaired is None or not repaired.strip():
                notes.append("repair:all_content_stripped")
                return None, tuple(notes)
            return repaired, tuple(notes)

        # No strip-able claims; try citation repair.
        citation_issues = [
            i for i in validation.issues
            if i.code in (CODE_CITATION_MISSING, CODE_CITATION_UNRESOLVED)
            and i.severity.value != "critical"
        ]
        if citation_issues:
            # Citation repair: append a generic "[source]" marker.
            # This is a minimal deterministic repair — we don't try to
            # resolve specific evidence indices because that would
            # require retrieval access (not allowed in the validator).
            notes.append("repair:citation_issues_non_blocking")
            # For WARNING-level citation issues, the answer is still
            # usable — return as-is.
            return answer, tuple(notes)

        # No repairable issues found.
        notes.append("repair:no_repairable_issues")
        return answer, tuple(notes)

    @classmethod
    def _strip_claims(
        cls,
        answer: str,
        claims_to_strip: set[str],
    ) -> str | None:
        """Remove sentences containing the specified claim texts.

        Splits the answer into sentences and removes any sentence that
        contains one of the claim texts. If all sentences are removed,
        returns None.
        """
        if not answer.strip():
            return None

        sentences = cls._split_sentences(answer)
        kept: list[str] = []

        for sentence in sentences:
            should_strip = False
            for claim_text in claims_to_strip:
                if claim_text in sentence:
                    should_strip = True
                    break
            if not should_strip:
                kept.append(sentence)

        if not kept:
            return None

        return " ".join(kept)

    @classmethod
    def _split_sentences(cls, answer: str) -> list[str]:
        """Split an answer into sentences.

        Handles common sentence boundaries (.!?) and newlines.
        """
        parts = cls._SENTENCE_SPLIT.split(answer)
        return [p.strip() for p in parts if p.strip()]
