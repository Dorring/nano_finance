"""Post-answer validation helpers for financial calculations.

This module checks whether numeric calculation claims in a generated answer are
consistent with deterministic calculation outputs. It is pure and does not call
LLMs or mutate query behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any


_PERCENT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>%|percent|percentage points?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerCalculationValidation:
    ok: bool
    mentioned_values: list[str] = field(default_factory=list)
    expected_values: list[str] = field(default_factory=list)
    missing_calculations: list[str] = field(default_factory=list)
    unsupported_values: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mentioned_values": self.mentioned_values,
            "expected_values": self.expected_values,
            "missing_calculations": self.missing_calculations,
            "unsupported_values": self.unsupported_values,
            "warnings": self.warnings,
        }


def validate_answer_calculations(
    answer: str,
    calculations: list[dict[str, Any]],
    tolerance_percent_points: str | Decimal = "0.05",
) -> AnswerCalculationValidation:
    """Validate percentage claims against structured deterministic calculations.

    `calculations` should contain dictionaries with `id`, `value`, and `unit`.
    Ratio values are converted to percentage points, e.g. 0.2 -> 20.
    """
    mentioned = extract_percent_values(answer)
    expected = _expected_percent_calculations(calculations)
    tolerance = _decimal(tolerance_percent_points) or Decimal("0.05")

    missing = []
    for calc_id, expected_value in expected.items():
        if not any(abs(value - expected_value) <= tolerance for value in mentioned):
            missing.append(calc_id)

    unsupported = []
    for value in mentioned:
        if expected and not any(abs(value - exp) <= tolerance for exp in expected.values()):
            unsupported.append(_format_decimal(value))

    warnings = []
    if missing:
        warnings.append("Answer is missing one or more expected calculation values.")
    if unsupported:
        warnings.append("Answer contains percentage values not supported by calculations.")

    return AnswerCalculationValidation(
        ok=not missing and not unsupported,
        mentioned_values=[_format_decimal(v) for v in mentioned],
        expected_values=[_format_decimal(v) for v in expected.values()],
        missing_calculations=missing,
        unsupported_values=unsupported,
        warnings=warnings,
    )


def extract_percent_values(text: str) -> list[Decimal]:
    """Extract percent-like values as percentage points, not ratios."""
    values = []
    for match in _PERCENT_RE.finditer(text or ""):
        value = _decimal(match.group("value").replace(",", ""))
        if value is not None:
            values.append(value)
    return values


def _expected_percent_calculations(calculations: list[dict[str, Any]]) -> dict[str, Decimal]:
    expected: dict[str, Decimal] = {}
    for idx, calc in enumerate(calculations or []):
        raw_value = _decimal(calc.get("value"))
        if raw_value is None:
            continue
        unit = (calc.get("unit") or "").lower()
        calc_id = str(calc.get("id") or calc.get("calc_id") or calc.get("operation") or idx)
        if unit == "ratio":
            expected[calc_id] = raw_value * Decimal("100")
        elif unit == "percent":
            expected[calc_id] = raw_value
    return expected


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")
