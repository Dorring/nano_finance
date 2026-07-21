"""Operation router: decides whether a question should enter the calculation pipeline.

The router is deliberately *conservative*: it only routes to the deterministic
calculation pipeline when ALL of the following are true:

1. ``intent["intent"] == "financial_calculation"`` (first gate, from
   ``src.services.intent.classify_query_intent``).
2. A recognized metric alias or generic operation keyword is present in the
   question text.
3. An explicit calculation verb/pattern is present (e.g. "calculate",
   "compute", "计算", "根据...计算", "derive"). This prevents reported-metric
   lookups like "毛利率是多少" from entering the calculation pipeline even
   when the intent classifier labels them ``financial_calculation``.

If any gate fails, the router returns ``NOT_APPLICABLE`` and the orchestrator
continues the normal RAG flow (LLM / deterministic answer extractor).

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and ``src.finance.metric_lexicon``
(both allowed) and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.calculation import CalculationOperation, CalculationStatus
from src.finance.metric_lexicon import (
    GenericOperationEntry,
    MetricDefinition,
    find_generic_operation,
    find_metric_by_alias,
)


# Explicit calculation verbs/patterns. At least one must be present for the
# router to enter the calculation pipeline. This is the conservative gate
# that prevents reported-metric lookups from being routed to calculation.
EXPLICIT_CALCULATION_PATTERNS: tuple[str, ...] = (
    "calculate",
    "compute",
    "derive",
    "work out",
    "convert",
    "根据",
    "计算",
    "求",
    "换算",
)


@dataclass(frozen=True)
class RoutingDecision:
    """The outcome of routing a question through the calculation pipeline gate.

    - ``READY``: the question should enter the calculation pipeline. The
      router has identified the operation, metric, formula version, and
      expected operand roles.
    - ``NOT_APPLICABLE``: the question should continue the normal RAG flow.
      ``reason`` explains why routing was declined.
    """

    status: CalculationStatus
    operation: CalculationOperation | None = None
    metric: str | None = None
    formula_version: str | None = None
    operand_roles: tuple[str, ...] = ()
    formula_template: str | None = None
    unit: str | None = None
    reason: str = ""

    @classmethod
    def not_applicable(cls, reason: str) -> "RoutingDecision":
        return cls(status=CalculationStatus.NOT_APPLICABLE, reason=reason)

    @classmethod
    def from_metric(cls, definition: MetricDefinition) -> "RoutingDecision":
        return cls(
            status=CalculationStatus.READY,
            operation=definition.operation,
            metric=definition.metric,
            formula_version=definition.formula_version,
            operand_roles=definition.operand_roles,
            formula_template=definition.formula_template,
            unit=definition.unit,
            reason="named_metric_matched",
        )

    @classmethod
    def from_generic(cls, entry: GenericOperationEntry) -> "RoutingDecision":
        return cls(
            status=CalculationStatus.READY,
            operation=entry.operation,
            metric=entry.operation.value,
            formula_version=entry.formula_version,
            operand_roles=(),
            formula_template=entry.formula_template,
            unit=entry.unit,
            reason="generic_operation_matched",
        )


def _has_explicit_calculation_pattern(text: str) -> bool:
    """Check whether the question contains an explicit calculation verb."""
    lowered = text.lower()
    return any(
        pattern in lowered or pattern in text
        for pattern in EXPLICIT_CALCULATION_PATTERNS
    )


def route_calculation(question: str, intent: dict[str, Any]) -> RoutingDecision:
    """Route a question to the calculation pipeline or decline (NOT_APPLICABLE).

    Args:
        question: The user's (possibly rewritten) question text.
        intent: The intent dict returned by ``classify_query_intent``.

    Returns:
        A ``RoutingDecision`` with ``status`` READY or NOT_APPLICABLE.
    """
    # Gate 1: intent must be financial_calculation.
    if intent.get("intent") != "financial_calculation":
        return RoutingDecision.not_applicable("intent_not_financial_calculation")

    # Gate 2: an explicit calculation verb must be present.
    if not _has_explicit_calculation_pattern(question):
        return RoutingDecision.not_applicable("no_explicit_calculation_verb")

    # Gate 3a: check for a named metric (gross_margin, net_margin, etc.).
    metric_def = find_metric_by_alias(question)
    if metric_def is not None:
        return RoutingDecision.from_metric(metric_def)

    # Gate 3b: check for a generic operation keyword (sum, difference, etc.).
    generic_entry = find_generic_operation(question)
    if generic_entry is not None:
        return RoutingDecision.from_generic(generic_entry)

    # No metric or operation keyword matched — let the LLM handle it.
    return RoutingDecision.not_applicable("no_metric_or_operation_matched")
