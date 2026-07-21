"""Deterministic financial calculation primitives for FinQuery.

Phase 3 Commit 3: migrated from ``src.services.financial_tools`` to
``src.finance.primitive_tools`` so the finance layer is the canonical home
for pure calculation logic. The original module is kept as a thin
re-export shim for backward compatibility.

These helpers are intentionally pure and dependency-free. They do not call
LLMs or external systems; RAG answers can cite their structured outputs
later. All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP`` and rejects
``NaN``/``Infinity`` implicitly via ``Decimal`` construction.

New primitives added in Phase 3:
- ``difference`` — subtraction
- ``average_values`` — arithmetic mean
- ``gross_margin`` — (revenue - cogs) / revenue
- ``net_margin`` — net_income / revenue
- ``debt_ratio`` — total_liabilities / total_assets
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


__all__ = [
    "ToolResult",
    "parse_financial_number",
    "growth_rate",
    "percentage_share",
    "sum_values",
    "verify_sum",
    "convert_scale",
    "format_ratio_percent",
    "difference",
    "average_values",
    "gross_margin",
    "net_margin",
    "debt_ratio",
]


_NUMBER_RE = re.compile(
    r"^\s*(?P<prefix>[^\d\-\(]*)?"
    r"(?P<negative>\()?"
    r"(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<percent>%?)"
    r"(?P<suffix>[^\d\)]*)?"
    r"(?P<close>\))?\s*$"
)

_SCALE_FACTORS = {
    "": Decimal("1"),
    "ones": Decimal("1"),
    "unit": Decimal("1"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "百万": Decimal("1000000"),
    "百万元": Decimal("1000000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Decimal | None = None
    error: str | None = None
    unit: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": str(self.value) if self.value is not None else None,
            "error": self.error,
            "unit": self.unit,
            "details": self.details or {},
        }


def parse_financial_number(value: Any, scale: str | None = None) -> ToolResult:
    """Parse numeric financial text into a Decimal.

    Handles commas, percentages, surrounding currency text, accounting negatives
    like ``(1,234)``, and common English/Chinese scale words.
    """
    if isinstance(value, Decimal):
        return ToolResult(True, value=value, details={"scale": scale or ""})
    if isinstance(value, (int, float)):
        return ToolResult(
            True, value=Decimal(str(value)), details={"scale": scale or ""}
        )

    text = str(value).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return ToolResult(False, error=f"Cannot parse financial number: {value!r}")

    number_text = match.group("number").replace(",", "")
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return ToolResult(False, error=f"Invalid numeric value: {value!r}")

    if match.group("negative") == "(" and match.group("close") == ")":
        number = -abs(number)

    suffix = (match.group("suffix") or "").strip()
    inferred_scale = scale or _infer_scale(suffix)
    factor = _scale_factor(inferred_scale)
    if factor is None:
        return ToolResult(False, error=f"Unknown scale: {inferred_scale!r}")

    if match.group("percent") == "%":
        number = number / Decimal("100")

    return ToolResult(
        True,
        value=number * factor,
        unit="base",
        details={
            "input": value,
            "scale": inferred_scale or "",
            "is_percent": match.group("percent") == "%",
        },
    )


def growth_rate(current: Any, previous: Any, precision: int = 4) -> ToolResult:
    """Return (current - previous) / previous."""
    cur = _ensure_decimal(current)
    prev = _ensure_decimal(previous)
    if not cur.ok:
        return cur
    if not prev.ok:
        return prev
    if prev.value == 0:
        return ToolResult(
            False, error="Cannot compute growth rate with zero previous value"
        )
    value = (cur.value - prev.value) / prev.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="ratio",
        details={"current": str(cur.value), "previous": str(prev.value)},
    )


def percentage_share(part: Any, total: Any, precision: int = 4) -> ToolResult:
    """Return part / total."""
    p = _ensure_decimal(part)
    t = _ensure_decimal(total)
    if not p.ok:
        return p
    if not t.ok:
        return t
    if t.value == 0:
        return ToolResult(False, error="Cannot compute share with zero total")
    value = p.value / t.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="ratio",
        details={"part": str(p.value), "total": str(t.value)},
    )


def sum_values(values: list[Any], precision: int = 2) -> ToolResult:
    """Return sum(values)."""
    parsed = []
    for value in values:
        item = _ensure_decimal(value)
        if not item.ok:
            return item
        parsed.append(item.value)
    total = sum(parsed, Decimal("0"))
    return ToolResult(
        True,
        value=_quantize(total, precision),
        unit="base",
        details={"count": len(parsed), "values": [str(v) for v in parsed]},
    )


def verify_sum(
    components: list[Any],
    reported_total: Any,
    tolerance: Any = Decimal("0.01"),
    precision: int = 2,
) -> ToolResult:
    """Check whether sum(components) matches reported_total within tolerance."""
    summed = sum_values(components, precision=precision)
    if not summed.ok:
        return summed
    reported = _ensure_decimal(reported_total)
    tol = _ensure_decimal(tolerance)
    if not reported.ok:
        return reported
    if not tol.ok:
        return tol

    diff = abs(summed.value - reported.value)
    ok = diff <= abs(tol.value)
    return ToolResult(
        ok,
        value=_quantize(diff, precision),
        error=None if ok else "Reported total does not match component sum",
        unit="base",
        details={
            "computed_total": str(summed.value),
            "reported_total": str(reported.value),
            "tolerance": str(tol.value),
        },
    )


def convert_scale(
    value: Any, from_scale: str, to_scale: str, precision: int = 4
) -> ToolResult:
    """Convert a value from one scale to another, e.g. million -> billion."""
    val = _ensure_decimal(value)
    if not val.ok:
        return val
    from_factor = _scale_factor(from_scale)
    to_factor = _scale_factor(to_scale)
    if from_factor is None:
        return ToolResult(False, error=f"Unknown source scale: {from_scale!r}")
    if to_factor is None:
        return ToolResult(False, error=f"Unknown target scale: {to_scale!r}")
    converted = val.value * from_factor / to_factor
    return ToolResult(
        True,
        value=_quantize(converted, precision),
        unit=to_scale,
        details={"from_scale": from_scale, "to_scale": to_scale},
    )


def format_ratio_percent(value: Any, precision: int = 2) -> ToolResult:
    """Format a ratio Decimal as percent value, e.g. 0.125 -> 12.50."""
    val = _ensure_decimal(value)
    if not val.ok:
        return val
    return ToolResult(
        True,
        value=_quantize(val.value * Decimal("100"), precision),
        unit="percent",
    )


# ---------------------------------------------------------------------------
# New primitives added in Phase 3
# ---------------------------------------------------------------------------


def difference(current: Any, previous: Any, precision: int = 2) -> ToolResult:
    """Return current - previous.

    Formula version: ``difference.v1``
    """
    cur = _ensure_decimal(current)
    prev = _ensure_decimal(previous)
    if not cur.ok:
        return cur
    if not prev.ok:
        return prev
    value = cur.value - prev.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="base",
        details={"current": str(cur.value), "previous": str(prev.value)},
    )


def average_values(values: list[Any], precision: int = 4) -> ToolResult:
    """Return the arithmetic mean of ``values``.

    Formula version: ``average.v1``
    """
    parsed = []
    for value in values:
        item = _ensure_decimal(value)
        if not item.ok:
            return item
        parsed.append(item.value)
    if not parsed:
        return ToolResult(False, error="Cannot compute average of empty list")
    total = sum(parsed, Decimal("0"))
    avg = total / Decimal(len(parsed))
    return ToolResult(
        True,
        value=_quantize(avg, precision),
        unit="base",
        details={"count": len(parsed), "values": [str(v) for v in parsed]},
    )


def gross_margin(revenue: Any, cogs: Any, precision: int = 4) -> ToolResult:
    """Return (revenue - cogs) / revenue.

    Formula version: ``gross_margin.v1``
    """
    rev = _ensure_decimal(revenue)
    cog = _ensure_decimal(cogs)
    if not rev.ok:
        return rev
    if not cog.ok:
        return cog
    if rev.value == 0:
        return ToolResult(False, error="Cannot compute gross margin with zero revenue")
    value = (rev.value - cog.value) / rev.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="ratio",
        details={"revenue": str(rev.value), "cogs": str(cog.value)},
    )


def net_margin(revenue: Any, net_income: Any, precision: int = 4) -> ToolResult:
    """Return net_income / revenue.

    Formula version: ``net_margin.v1``
    """
    rev = _ensure_decimal(revenue)
    ni = _ensure_decimal(net_income)
    if not rev.ok:
        return rev
    if not ni.ok:
        return ni
    if rev.value == 0:
        return ToolResult(False, error="Cannot compute net margin with zero revenue")
    value = ni.value / rev.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="ratio",
        details={"revenue": str(rev.value), "net_income": str(ni.value)},
    )


def debt_ratio(
    total_liabilities: Any, total_assets: Any, precision: int = 4
) -> ToolResult:
    """Return total_liabilities / total_assets.

    Formula version: ``debt_ratio.v1``
    """
    tl = _ensure_decimal(total_liabilities)
    ta = _ensure_decimal(total_assets)
    if not tl.ok:
        return tl
    if not ta.ok:
        return ta
    if ta.value == 0:
        return ToolResult(
            False, error="Cannot compute debt ratio with zero total assets"
        )
    value = tl.value / ta.value
    return ToolResult(
        True,
        value=_quantize(value, precision),
        unit="ratio",
        details={"total_liabilities": str(tl.value), "total_assets": str(ta.value)},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_decimal(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    return parse_financial_number(value)


def _scale_factor(scale: str | None) -> Decimal | None:
    return _SCALE_FACTORS.get((scale or "").strip().lower())


def _infer_scale(text: str) -> str:
    lowered = text.lower()
    for key in sorted(_SCALE_FACTORS, key=len, reverse=True):
        if key and key in lowered:
            return key
    return ""


def _quantize(value: Decimal, precision: int) -> Decimal:
    exponent = Decimal("1").scaleb(-precision)
    return value.quantize(exponent, rounding=ROUND_HALF_UP)
