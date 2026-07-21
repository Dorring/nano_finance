"""Calculation registry: maps CalculationOperation to primitive functions.

The registry is the single source of truth for how each deterministic
financial operation is executed. It binds each ``CalculationOperation``
to:

- The primitive function (via an adapter) that performs the arithmetic.
- The human-readable formula string.
- The formula version string (for traceability).
- The result unit.
- Operand count constraints (min/max).
- The expected operand roles (for validation by the plan builder).

The executor consults this registry to look up and dispatch calculations.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and ``src.finance.primitive_tools``
(both allowed) and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.domain.calculation import CalculationOperand, CalculationOperation
from src.finance.primitive_tools import (
    ToolResult,
    average_values,
    debt_ratio,
    difference,
    gross_margin,
    growth_rate,
    net_margin,
    percentage_share,
    sum_values,
)


# Type alias for the adapter functions that wrap primitive_tools calls.
# Each adapter takes a tuple of operands and a precision int, returning
# a ``ToolResult``.
CalculationFunc = Callable[[tuple[CalculationOperand, ...], int], ToolResult]


@dataclass(frozen=True)
class OperationEntry:
    """Registry entry binding an operation to its execution metadata."""

    operation: CalculationOperation
    func: CalculationFunc
    formula: str
    formula_version: str
    unit: str
    min_operands: int
    max_operands: int
    operand_roles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------
# Each adapter extracts Decimal values from CalculationOperand instances
# and delegates to the corresponding primitive_tools function. Adapters
# never raise; they return ``ToolResult(ok=False, ...)`` on misuse so the
# executor can map the outcome to a BLOCKED result.


def _two_operand_adapter(
    primitive: Callable,
    precision: int,
    operands: tuple[CalculationOperand, ...],
    role_names: tuple[str, str],
) -> ToolResult:
    if len(operands) < 2:
        return ToolResult(
            False,
            error=f"requires 2 operands ({role_names[0]}, {role_names[1]}), "
            f"got {len(operands)}",
        )
    return primitive(operands[0].value, operands[1].value, precision=precision)


def _difference_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        difference, precision, operands, ("current", "previous")
    )


def _growth_rate_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        growth_rate, precision, operands, ("current", "previous")
    )


def _percentage_share_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        percentage_share, precision, operands, ("part", "total")
    )


def _gross_margin_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(gross_margin, precision, operands, ("revenue", "cogs"))


def _net_margin_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        net_margin, precision, operands, ("revenue", "net_income")
    )


def _debt_ratio_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        debt_ratio, precision, operands, ("total_liabilities", "total_assets")
    )


def _sum_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands:
        return ToolResult(False, error="sum requires at least 1 operand, got 0")
    values = [op.value for op in operands]
    return sum_values(values, precision=precision)


def _average_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands:
        return ToolResult(False, error="average requires at least 1 operand, got 0")
    values = [op.value for op in operands]
    return average_values(values, precision=precision)


def _scale_conversion_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    # Scale conversion requires explicit from_scale/to_scale parameters that
    # are not carried by CalculationOperand. For v1, this adapter exists for
    # registry completeness but always declines; the executor will map the
    # decline to a BLOCKED result.
    return ToolResult(
        False,
        error="scale_conversion requires explicit from/to scale parameters "
        "not available in the plan",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALCULATION_REGISTRY: dict[CalculationOperation, OperationEntry] = {
    CalculationOperation.DIFFERENCE: OperationEntry(
        operation=CalculationOperation.DIFFERENCE,
        func=_difference_adapter,
        formula="current - previous",
        formula_version="difference.v1",
        unit="base",
        min_operands=2,
        max_operands=2,
        operand_roles=("current", "previous"),
    ),
    CalculationOperation.GROWTH_RATE: OperationEntry(
        operation=CalculationOperation.GROWTH_RATE,
        func=_growth_rate_adapter,
        formula="(current - previous) / previous",
        formula_version="growth_rate.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("current", "previous"),
    ),
    CalculationOperation.PERCENTAGE_SHARE: OperationEntry(
        operation=CalculationOperation.PERCENTAGE_SHARE,
        func=_percentage_share_adapter,
        formula="part / total",
        formula_version="percentage_share.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("part", "total"),
    ),
    CalculationOperation.SUM: OperationEntry(
        operation=CalculationOperation.SUM,
        func=_sum_adapter,
        formula="sum(operands)",
        formula_version="sum.v1",
        unit="base",
        min_operands=1,
        max_operands=100,
        operand_roles=(),
    ),
    CalculationOperation.AVERAGE: OperationEntry(
        operation=CalculationOperation.AVERAGE,
        func=_average_adapter,
        formula="sum(operands) / count",
        formula_version="average.v1",
        unit="base",
        min_operands=1,
        max_operands=100,
        operand_roles=(),
    ),
    CalculationOperation.GROSS_MARGIN: OperationEntry(
        operation=CalculationOperation.GROSS_MARGIN,
        func=_gross_margin_adapter,
        formula="(revenue - cogs) / revenue",
        formula_version="gross_margin.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("revenue", "cogs"),
    ),
    CalculationOperation.NET_MARGIN: OperationEntry(
        operation=CalculationOperation.NET_MARGIN,
        func=_net_margin_adapter,
        formula="net_income / revenue",
        formula_version="net_margin.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("revenue", "net_income"),
    ),
    CalculationOperation.DEBT_RATIO: OperationEntry(
        operation=CalculationOperation.DEBT_RATIO,
        func=_debt_ratio_adapter,
        formula="total_liabilities / total_assets",
        formula_version="debt_ratio.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("total_liabilities", "total_assets"),
    ),
    CalculationOperation.SCALE_CONVERSION: OperationEntry(
        operation=CalculationOperation.SCALE_CONVERSION,
        func=_scale_conversion_adapter,
        formula="value * from_factor / to_factor",
        formula_version="scale_conversion.v1",
        unit="base",
        min_operands=1,
        max_operands=1,
        operand_roles=("value",),
    ),
}


def get_operation_entry(
    operation: CalculationOperation,
) -> OperationEntry | None:
    """Return the registry entry for ``operation``, or None if not registered."""
    return CALCULATION_REGISTRY.get(operation)
