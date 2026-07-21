"""Calculation executor: runs a CalculationPlan to produce a CalculationResult.

The executor is the final step in the deterministic calculation pipeline.
It takes a ``CalculationPlan`` (built by the plan builder from a routing
decision and extracted operands) and produces a ``CalculationResult``:

- If the plan is already ``BLOCKED`` or ``NOT_APPLICABLE``, the result
  mirrors that status without invoking any primitive.
- If the plan is ``READY``, the executor validates operand count, looks
  up the primitive function from ``CALCULATION_REGISTRY``, and calls it.
  - Success -> ``EXECUTED`` with the computed value.
  - Primitive returns ``ok=False`` -> ``BLOCKED`` (deterministic refusal;
    e.g. division by zero, insufficient operands from adapter).
  - Unexpected exception -> ``FAILED`` (fall back to LLM).

The executor NEVER calls the LLM and NEVER accesses retrieval. It is
purely deterministic and side-effect-free.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and
``src.finance.calculation_registry`` (both allowed) and stdlib only.
"""

from __future__ import annotations

import logging

from src.domain.calculation import (
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_registry import get_operation_entry

logger = logging.getLogger(__name__)


def execute_plan(plan: CalculationPlan) -> CalculationResult:
    """Execute a ``CalculationPlan`` and return a ``CalculationResult``.

    Args:
        plan: The calculation plan to execute. May be READY, BLOCKED, or
            NOT_APPLICABLE.

    Returns:
        A ``CalculationResult`` with status EXECUTED, BLOCKED, FAILED, or
        NOT_APPLICABLE.
    """
    # Pass-through for non-READY plans.
    if plan.status is CalculationStatus.NOT_APPLICABLE:
        return CalculationResult(status=CalculationStatus.NOT_APPLICABLE)

    if plan.status is CalculationStatus.BLOCKED:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PLAN_BLOCKED",
            error_message=plan.block_reason or "plan was blocked before execution",
        )

    if plan.status is not CalculationStatus.READY:
        # FAILED or EXECUTED plans should not reach the executor; mirror as-is.
        return CalculationResult(
            status=plan.status,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
        )

    # READY plan: look up the registry entry.
    entry = get_operation_entry(plan.operation)
    if entry is None:
        return CalculationResult(
            status=CalculationStatus.FAILED,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="UNKNOWN_OPERATION",
            error_message=f"no registry entry for operation {plan.operation!r}",
        )

    # Validate operand count.
    operand_count = len(plan.operands)
    if operand_count < entry.min_operands:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="INSUFFICIENT_OPERANDS",
            error_message=(
                f"{plan.operation.value} requires at least {entry.min_operands} "
                f"operands, got {operand_count}"
            ),
        )
    if operand_count > entry.max_operands:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="TOO_MANY_OPERANDS",
            error_message=(
                f"{plan.operation.value} accepts at most {entry.max_operands} "
                f"operands, got {operand_count}"
            ),
        )

    # Execute the primitive function.
    try:
        result = entry.func(plan.operands, plan.precision)
    except Exception as exc:  # noqa: BLE001 - executor must not raise
        logger.warning("calculation primitive raised: %s", exc, exc_info=True)
        return CalculationResult(
            status=CalculationStatus.FAILED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_EXCEPTION",
            error_message=str(exc),
        )

    if not result.ok or result.value is None:
        # Primitive declined (e.g. division by zero, missing scale params).
        # This is a deterministic refusal, so we BLOCK rather than FAILED
        # so the orchestrator bypasses the LLM with a deterministic refusal.
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_DECLINED",
            error_message=result.error or "primitive returned without a value",
        )

    # Success.
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=plan.operation,
        value=result.value,
        unit=entry.unit,
        formula=entry.formula,
        formula_version=entry.formula_version,
        target_metric=plan.target_metric,
        operands=plan.operands,
    )
