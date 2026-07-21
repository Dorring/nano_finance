"""Validation and answerability domain objects (Phase 4).

These types form the typed boundary for the Phase 4 grounded-answer
validation layer. They are deliberately dependency-free (stdlib +
``decimal`` only) so that the ``domain`` layer does not import from
``validation``, ``application``, or ``services``.

Dependency direction::

    domain
      ↑
    validation
      ↑
    application
      ↑
    services / api

Key invariants enforced by these types:
- ``ValidationIssue`` separates an internal ``message`` (for logs) from a
  ``public_message`` (for API responses). Internal exception text, full
  evidence, and stack traces must never reach the public payload.
- ``ValidationResult.to_public_dict`` omits internal diagnostics and only
  emits the status, claim counts, and public issue messages.
- ``ValidationResult.to_trace_dict`` omits full answer/evidence text but
  retains issue codes and claim counts for debugging.
- A validator that cannot complete MUST produce ``ValidationStatus.FAILED``
  — it must never default to PASSED.

Phase 4 scope note: these types support deterministic validation of
numeric values, units, periods, metrics, citations, and calculation
results. They do NOT claim to verify arbitrary natural-language facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnswerabilityStatus(str, Enum):
    """Pre-generation answerability verdict.

    - ``ANSWERABLE``          -> evidence is sufficient; generation allowed.
    - ``PARTIALLY_ANSWERABLE`` -> some sub-questions / documents covered;
      a limited answer is permitted but must not be framed as complete.
    - ``NOT_ANSWERABLE``      -> evidence is missing or irrelevant; the LLM
      must NOT be invoked and a deterministic refusal is returned.
    - ``CALCULATION_BLOCKED`` -> the deterministic calculation pipeline
      returned BLOCKED / FAILED; the LLM must NOT be invoked and the
      Phase 3 safe response is returned.
    """

    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    NOT_ANSWERABLE = "not_answerable"
    CALCULATION_BLOCKED = "calculation_blocked"


class ValidationStatus(str, Enum):
    """Post-generation validation verdict.

    - ``NOT_APPLICABLE`` -> validation does not apply (e.g. conversation).
    - ``PASSED``         -> no ERROR/CRITICAL issues; all strict numeric
      claims are supported and citations are valid.
    - ``REPAIRABLE``     -> only format/cosmetic issues; a single
      deterministic repair is permitted.
    - ``BLOCKED``        -> a core numeric/unit/period/citation/calculation
      error; the answer must NOT be returned to the user.
    - ``FAILED``         -> the validator itself could not complete; the
      answer must NOT be returned (fail-closed).
    """

    NOT_APPLICABLE = "not_applicable"
    PASSED = "passed"
    REPAIRABLE = "repairable"
    BLOCKED = "blocked"
    FAILED = "failed"


class ValidationSeverity(str, Enum):
    """Severity of a single validation issue.

    - ``INFO``     -> informational; never affects the verdict.
    - ``WARNING``  -> non-blocking; may be surfaced to the user.
    - ``ERROR``    -> blocking for strict paths; repairable for lenient ones.
    - ``CRITICAL`` -> always blocking; never repairable.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    """A single deterministic validation finding.

    ``message`` is for internal logs only and may contain evidence
    references; it MUST NOT be placed in a public API response. Public
    responses use ``public_message`` (or the frontend maps ``code`` to a
    localized string).
    """

    code: str
    severity: ValidationSeverity
    message: str
    claim_text: str | None = None
    evidence_ids: tuple[str, ...] = ()
    public_message: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for public API responses.

        Omits the internal ``message`` and ``evidence_ids`` (which may
        leak internal structure). Only ``code``, ``severity``, and
        ``public_message`` are exposed.
        """
        return {
            "code": self.code,
            "severity": self.severity.value,
            "public_message": self.public_message,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trace logging.

        Retains ``code``, ``severity``, ``claim_text`` (the offending
        snippet — not the full answer), and ``evidence_ids`` for debugging.
        The full internal ``message`` is included because trace storage is
        access-controlled and used for incident diagnosis.
        """
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "claim_text": self.claim_text,
            "evidence_ids": list(self.evidence_ids),
        }


# ---------------------------------------------------------------------------
# Answerability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerabilityResult:
    """Outcome of the pre-generation answerability evaluation.

    ``missing_requirements`` is a human-readable tuple of what is missing
    (e.g. ``("document: peer_report.pdf", "metric: net_income")``). It is
    used to build the safe-fallback refusal message and MUST NOT contain
    internal paths, stack traces, or prompt content.
    """

    status: AnswerabilityStatus
    reason_codes: tuple[str, ...]
    evidence_count: int
    document_count: int
    best_score: float | None
    average_score: float | None
    missing_requirements: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for public API responses.

        Exposes only the status, reason codes, counts, and missing
        requirements (used to explain the refusal to the user). Scores are
        included because they are already exposed via ``confidence`` in
        Phase 3; raw retrieval internals are not.
        """
        return {
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "evidence_count": self.evidence_count,
            "document_count": self.document_count,
            "missing_requirements": list(self.missing_requirements),
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trace logging.

        Includes ``best_score`` / ``average_score`` for debugging the
        answerability decision; these are omitted from public responses to
        avoid implying calibrated probabilities.
        """
        return {
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "evidence_count": self.evidence_count,
            "document_count": self.document_count,
            "best_score": self.best_score,
            "average_score": self.average_score,
            "missing_requirements": list(self.missing_requirements),
        }


# ---------------------------------------------------------------------------
# Extracted claim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedClaim:
    """A single deterministic claim extracted from a generated answer.

    Phase 4 only extracts claims that can be deterministically validated:
    numeric values (amounts, percentages, ratios), years/quarters, metric
    names, and citation references. Free-form natural-language propositions
    are NOT extracted as claims — their unverifiability is a documented
    limitation.

    ``value`` uses ``Decimal`` so precision and sign are preserved exactly.
    ``claim_type`` is one of: ``amount``, ``percent``, ``ratio``,
    ``period``, ``metric``, ``citation_ref``.
    """

    claim_id: str
    claim_type: str
    raw_text: str
    metric: str | None = None
    value: Decimal | None = None
    unit: str | None = None
    scale: str | None = None
    currency: str | None = None
    period: str | None = None
    citation_refs: tuple[str, ...] = ()

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trace logging.

        Includes the standardized ``value`` (as a string) and structural
        fields. ``raw_text`` is included because it is a short claim
        snippet, not the full answer — acceptable for trace storage.
        """
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "raw_text": self.raw_text,
            "metric": self.metric,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "scale": self.scale,
            "currency": self.currency,
            "period": self.period,
            "citation_refs": list(self.citation_refs),
        }


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of post-generation validation.

    - ``checked_claim_count``  -> how many claims were examined.
    - ``supported_claim_count`` -> how many were supported by evidence /
      calculation.
    - ``unsupported_claim_count`` -> how many lacked support.
    - ``repaired_answer`` -> if a single deterministic repair was applied,
      the repaired answer text (otherwise ``None``).

    A validator that raises must produce ``ValidationStatus.FAILED`` with a
    single CRITICAL issue — never PASSED.
    """

    status: ValidationStatus
    issues: tuple[ValidationIssue, ...] = ()
    checked_claim_count: int = 0
    supported_claim_count: int = 0
    unsupported_claim_count: int = 0
    repaired_answer: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for public API responses.

        Emits status, claim counts, and public issue payloads. Internal
        ``message`` text and full evidence are never included. If the
        validator FAILED, only the status and a single sanitized issue are
        emitted — the internal exception is not exposed.
        """
        public_issues = [i.to_public_dict() for i in self.issues]
        payload: dict[str, Any] = {
            "status": self.status.value,
            "checked_claim_count": self.checked_claim_count,
            "supported_claim_count": self.supported_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "issues": public_issues,
        }
        return payload

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize for trace logging.

        Includes the full issue payloads (with internal ``message``) and
        issue codes for debugging. ``repaired_answer`` is included as a
        flag (``bool``) rather than the full text to minimize answer
        leakage into trace storage.
        """
        return {
            "status": self.status.value,
            "checked_claim_count": self.checked_claim_count,
            "supported_claim_count": self.supported_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "issue_codes": [i.code for i in self.issues],
            "issues": [i.to_trace_dict() for i in self.issues],
            "repaired": self.repaired_answer is not None,
        }


# ---------------------------------------------------------------------------
# Aggregated grounded-response result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundedResponseResult:
    """The final output of the grounded validation pipeline.

    Combines the answerability verdict, the post-generation validation
    verdict, and the (possibly repaired / fallback) answer that is safe to
    return to the user.

    ``warnings`` are non-blocking, user-facing notices (e.g. partial
    coverage). They MUST NOT contain internal diagnostics.
    """

    answer: str
    sources: tuple[dict[str, Any], ...]
    answerability: AnswerabilityResult | None
    validation: ValidationResult | None
    warnings: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answer": self.answer,
            "sources": list(self.sources),
        }
        if self.answerability is not None:
            payload["answerability"] = self.answerability.to_public_dict()
        if self.validation is not None:
            payload["validation"] = self.validation.to_public_dict()
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload
