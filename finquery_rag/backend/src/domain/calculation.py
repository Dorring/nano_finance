"""Financial calculation domain objects.

These types form the typed boundary between the RAG orchestrator and the
deterministic calculation pipeline introduced in Phase 3. They are
deliberately dependency-free (stdlib + ``decimal`` only) so that the
``domain`` layer does not import from ``finance``, ``application``, or
``services``.

Dependency direction: ``domain -> finance -> application -> services``.

Key invariants enforced by these types:
- Every ``CalculationOperand`` MUST cite ``source_text`` and
  ``evidence_chunk_id`` so the calculation is auditable end-to-end.
- Every formula carries a ``formula_version`` string for traceability across
  releases.
- ``CalculationStatus`` drives the orchestrator's LLM-bypass decision:
  - ``EXECUTED``  -> skip LLM, return deterministic answer.
  - ``BLOCKED``   -> skip LLM, return deterministic refusal.
  - ``FAILED``    -> fall back to LLM.
  - ``NOT_APPLICABLE`` -> continue normal RAG flow (no calculation attempted).
  - ``READY``     -> transient state between plan builder and executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class CalculationOperation(str, Enum):
    """The set of deterministic financial operations supported in Phase 3 v1.

    ``ROE`` and ``CAGR`` are deliberately excluded from v1 because they
    require additional evidence disambiguation (average equity / multi-period
    compounding) that is out of scope for this phase.
    """

    DIFFERENCE = "difference"
    GROWTH_RATE = "growth_rate"
    PERCENTAGE_SHARE = "percentage_share"
    SUM = "sum"
    AVERAGE = "average"
    GROSS_MARGIN = "gross_margin"
    NET_MARGIN = "net_margin"
    DEBT_RATIO = "debt_ratio"
    SCALE_CONVERSION = "scale_conversion"


class CalculationStatus(str, Enum):
    """Lifecycle status of a calculation attempt.

    The orchestrator inspects this to decide whether to bypass the LLM.
    """

    NOT_APPLICABLE = "not_applicable"
    READY = "ready"
    EXECUTED = "executed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class CalculationOperand:
    """A single numeric input bound to retrieved evidence.

    Every operand MUST cite the exact ``source_text`` substring from a
    retrieved ``EvidenceItem`` so the calculation is auditable. The
    ``evidence_chunk_id`` ties back to the retrieval pipeline's evidence set;
    ``document_name`` and ``page`` are denormalized for display without
    requiring a re-lookup of the evidence item.
    """

    name: str
    value: Decimal
    unit: str | None = None
    scale: str | None = None
    source_text: str = ""
    evidence_chunk_id: str = ""
    document_name: str | None = None
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for trace / legacy API emission."""
        return {
            "name": self.name,
            "value": str(self.value),
            "unit": self.unit,
            "scale": self.scale,
            "source_text": self.source_text,
            "evidence_chunk_id": self.evidence_chunk_id,
            "document_name": self.document_name,
            "page": self.page,
        }


@dataclass(frozen=True)
class CalculationPlan:
    """An immutable plan describing a single deterministic calculation.

    The ``formula_version`` pins the exact formula used so results are
    reproducible and auditable across releases (e.g. ``"gross_margin.v1"``).
    ``precision`` controls the number of decimal places in the result.
    """

    operation: CalculationOperation
    operands: tuple[CalculationOperand, ...]
    formula_version: str
    target_metric: str
    precision: int = 4
    label: str | None = None
    status: CalculationStatus = CalculationStatus.READY
    block_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "operands": [op.to_dict() for op in self.operands],
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "precision": self.precision,
            "label": self.label,
            "status": self.status.value,
            "block_reason": self.block_reason,
        }


@dataclass(frozen=True)
class CalculationResult:
    """The outcome of executing (or attempting) a ``CalculationPlan``.

    - ``EXECUTED``  -> ``value`` is populated; orchestrator bypasses LLM.
    - ``BLOCKED``   -> plan could not be built or operands are insufficient;
      orchestrator bypasses LLM and returns a deterministic refusal.
    - ``FAILED``    -> plan was built but execution raised an error;
      orchestrator falls back to the LLM.
    - ``NOT_APPLICABLE`` -> question was not a calculation; orchestrator
      continues the normal RAG flow.
    - ``READY``     -> transient; only used between plan builder and executor.
    """

    status: CalculationStatus
    operation: CalculationOperation | None = None
    value: Decimal | None = None
    unit: str | None = None
    formula: str | None = None
    formula_version: str | None = None
    target_metric: str | None = None
    operands: tuple[CalculationOperand, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for ``AnswerResult.calculations``."""
        payload: dict[str, Any] = {
            "status": self.status.value,
            "operation": self.operation.value if self.operation else None,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula": self.formula,
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "operands": [op.to_dict() for op in self.operands],
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
        return payload


NOT_APPLICABLE_RESULT = CalculationResult(status=CalculationStatus.NOT_APPLICABLE)
"""Sentinel returned by the pipeline when the question is not a calculation."""
