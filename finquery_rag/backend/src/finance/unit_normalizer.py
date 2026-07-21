"""Unit normalization for financial calculation operands.

Detects the currency/unit of a text fragment and checks that all operands
in a calculation plan use consistent units. When units are inconsistent
(e.g. revenue in USD but cogs in CNY), the plan must be BLOCKED because
mixing currencies produces meaningless ratios.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and stdlib only.
"""

from __future__ import annotations

import re

from src.domain.calculation import CalculationOperand


# Currency / unit detection patterns. Checked in order; first match wins.
# Patterns are case-insensitive for ASCII; Chinese is matched as-is.
_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bUSD\b|\bUS\$\b|\$"), "USD"),
    (re.compile(r"\bEUR\b|€"), "EUR"),
    (re.compile(r"\bGBP\b|£"), "GBP"),
    (re.compile(r"\bJPY\b|¥"), "JPY"),
    (re.compile(r"\bCNY\b|\bRMB\b|人民币|元(?!币)"), "CNY"),
    (re.compile(r"%"), "percent"),
)

# Scale words that may appear alongside currency. These are NOT units;
# they are handled by parse_financial_number's scale factor.
_SCALE_WORDS: tuple[str, ...] = (
    "thousand",
    "million",
    "billion",
    "trillion",
    "k",
    "m",
    "bn",
    "万",
    "百万",
    "亿",
)


def detect_unit(text: str | None) -> str | None:
    """Detect the currency or unit implied by a text fragment.

    Returns ``None`` when no unit is detectable (bare numbers without
    currency symbols or percent signs).
    """
    if not text:
        return None
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(text):
            return unit
    return None


def normalize_operand_unit(operand: CalculationOperand) -> str | None:
    """Return the effective unit for an operand.

    If the operand already has ``unit`` set, use it. Otherwise infer the
    unit from ``source_text``.
    """
    if operand.unit:
        return operand.unit
    return detect_unit(operand.source_text)


def check_unit_consistency(operands: tuple[CalculationOperand, ...]) -> bool:
    """Return ``True`` if all operands use consistent units.

    Operands with no detectable unit are treated as compatible with any
    unit. If at least one operand has a unit and another has a different
    unit, they are inconsistent.
    """
    detected: list[str] = []
    for op in operands:
        u = normalize_operand_unit(op)
        if u is not None:
            detected.append(u)
    if not detected:
        return True
    # All detected units must be the same.
    first = detected[0]
    return all(u == first for u in detected[1:])


def detect_scale(text: str | None) -> str | None:
    """Detect the scale word in a text fragment (e.g. 'million', '亿').

    Returns the scale word as a string, or ``None`` if none is found.
    """
    if not text:
        return None
    lowered = text.lower()
    # Check longer words first to avoid partial matches (billion before b).
    for word in sorted(_SCALE_WORDS, key=len, reverse=True):
        if word in lowered or word in text:
            return word
    return None
