"""Finance package: deterministic financial calculation primitives and pipeline.

Layer dependency: ``domain -> finance -> application -> services``. Modules in
this package must NOT import from ``src.services`` or ``src.application``.
"""

from src.finance.calculation_executor import execute_plan  # noqa: F401
from src.finance.calculation_renderer import render_calculation_result  # noqa: F401
from src.finance.calculation_registry import (  # noqa: F401
    CALCULATION_REGISTRY,
    OperationEntry,
    get_operation_entry,
)
from src.finance.primitive_tools import (  # noqa: F401
    ToolResult,
    average_values,
    convert_scale,
    debt_ratio,
    difference,
    format_ratio_percent,
    gross_margin,
    growth_rate,
    net_margin,
    parse_financial_number,
    percentage_share,
    sum_values,
    verify_sum,
)
