"""Financial metric lexicon: maps metric names to operations and operand roles.

Each ``MetricDefinition`` pins the formula version, operand roles, and
human-readable formula template for a named financial metric. The operation
router consults this lexicon to decide whether a question should enter the
deterministic calculation pipeline.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` (allowed) and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.calculation import CalculationOperation


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable definition of a named financial metric.

    ``operand_roles`` are the logical names of the inputs the executor
    expects (e.g. ``("revenue", "cogs")``). The evidence extractor is
    responsible for finding actual numeric values in retrieved evidence
    that satisfy these roles.
    """

    metric: str
    operation: CalculationOperation
    operand_roles: tuple[str, ...]
    formula_version: str
    aliases_en: tuple[str, ...]
    aliases_zh: tuple[str, ...]
    formula_template: str
    unit: str


# ---------------------------------------------------------------------------
# Named metrics: specific formulas with fixed operand roles
# ---------------------------------------------------------------------------

NAMED_METRICS: dict[str, MetricDefinition] = {
    "gross_margin": MetricDefinition(
        metric="gross_margin",
        operation=CalculationOperation.GROSS_MARGIN,
        operand_roles=("revenue", "cogs"),
        formula_version="gross_margin.v1",
        aliases_en=("gross margin", "gross profit margin", "gp margin"),
        aliases_zh=("毛利率",),
        formula_template="(revenue - cogs) / revenue",
        unit="ratio",
    ),
    "net_margin": MetricDefinition(
        metric="net_margin",
        operation=CalculationOperation.NET_MARGIN,
        operand_roles=("revenue", "net_income"),
        formula_version="net_margin.v1",
        aliases_en=("net margin", "net profit margin", "np margin"),
        aliases_zh=("净利率", "净利润率"),
        formula_template="net_income / revenue",
        unit="ratio",
    ),
    "debt_ratio": MetricDefinition(
        metric="debt_ratio",
        operation=CalculationOperation.DEBT_RATIO,
        operand_roles=("total_liabilities", "total_assets"),
        formula_version="debt_ratio.v1",
        aliases_en=(
            "debt ratio",
            "debt to asset ratio",
            "debt-to-asset",
            "liability ratio",
        ),
        aliases_zh=("资产负债率", "负债比率"),
        formula_template="total_liabilities / total_assets",
        unit="ratio",
    ),
    "growth_rate": MetricDefinition(
        metric="growth_rate",
        operation=CalculationOperation.GROWTH_RATE,
        operand_roles=("current", "previous"),
        formula_version="growth_rate.v1",
        aliases_en=(
            "growth rate",
            "yoy growth",
            "qoq growth",
            "year over year",
            "year-over-year",
            "year on year",
        ),
        aliases_zh=("增长率", "同比增长率", "环比增长率"),
        formula_template="(current - previous) / previous",
        unit="ratio",
    ),
    "percentage_share": MetricDefinition(
        metric="percentage_share",
        operation=CalculationOperation.PERCENTAGE_SHARE,
        operand_roles=("part", "total"),
        formula_version="percentage_share.v1",
        aliases_en=("percentage share", "share of", "proportion of", "percentage of"),
        aliases_zh=("占比", "比例"),
        formula_template="part / total",
        unit="ratio",
    ),
}


# ---------------------------------------------------------------------------
# Generic operation keywords: no fixed operand roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenericOperationEntry:
    """A generic operation triggered by keywords rather than a named metric."""

    operation: CalculationOperation
    formula_version: str
    keywords_en: tuple[str, ...]
    keywords_zh: tuple[str, ...]
    formula_template: str
    unit: str


GENERIC_OPERATIONS: tuple[GenericOperationEntry, ...] = (
    GenericOperationEntry(
        operation=CalculationOperation.SUM,
        formula_version="sum.v1",
        keywords_en=("sum of", "total of", "add up", "combined total"),
        keywords_zh=("求和", "总和", "合计"),
        formula_template="sum(operands)",
        unit="base",
    ),
    GenericOperationEntry(
        operation=CalculationOperation.DIFFERENCE,
        formula_version="difference.v1",
        keywords_en=("difference between", "subtract", "minus"),
        keywords_zh=("差额", "差值", "相减"),
        formula_template="current - previous",
        unit="base",
    ),
    GenericOperationEntry(
        operation=CalculationOperation.AVERAGE,
        formula_version="average.v1",
        keywords_en=("average of", "mean of", "arithmetic mean"),
        keywords_zh=("平均值", "均值", "平均数"),
        formula_template="sum(operands) / count",
        unit="base",
    ),
    GenericOperationEntry(
        operation=CalculationOperation.SCALE_CONVERSION,
        formula_version="scale_conversion.v1",
        keywords_en=("convert", "in terms of", "expressed in"),
        keywords_zh=("换算", "转换为"),
        formula_template="value * from_factor / to_factor",
        unit="base",
    ),
)


def find_metric_by_alias(text: str) -> MetricDefinition | None:
    """Return the first ``MetricDefinition`` whose alias appears in ``text``.

    Matching is case-insensitive for English aliases and substring-based
    for Chinese aliases. Longer aliases are checked first to avoid partial
    matches (e.g. "gross profit margin" before "gross margin").
    """
    lowered = text.lower()
    # Sort all aliases across all metrics by length (descending) so that
    # more specific phrases win over shorter substrings.
    candidates: list[tuple[str, MetricDefinition]] = []
    for definition in NAMED_METRICS.values():
        for alias in definition.aliases_en:
            candidates.append((alias, definition))
        for alias in definition.aliases_zh:
            candidates.append((alias, definition))
    candidates.sort(key=lambda pair: len(pair[0]), reverse=True)

    for alias, definition in candidates:
        if alias.lower() in lowered:
            return definition
    return None


def find_generic_operation(text: str) -> GenericOperationEntry | None:
    """Return the first generic operation whose keyword appears in ``text``."""
    lowered = text.lower()
    for entry in GENERIC_OPERATIONS:
        for kw in entry.keywords_en:
            if kw in lowered:
                return entry
        for kw in entry.keywords_zh:
            if kw in text:
                return entry
    return None
