"""Phase 7C tests: answer calculation validation."""
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.answer_validation import (
    extract_percent_values,
    validate_answer_calculations,
)


def test_extract_percent_values_supports_percent_words():
    values = extract_percent_values("Revenue grew 20%, margin was 12.5 percent.")
    assert values == [Decimal("20"), Decimal("12.5")]


def test_validate_answer_calculations_passes_ratio_mentioned_as_percent():
    result = validate_answer_calculations(
        "Revenue grew 20%.",
        [{"id": "growth", "operation": "growth_rate", "value": "0.2000", "unit": "ratio"}],
    )
    assert result.ok is True
    assert result.mentioned_values == ["20"]
    assert result.expected_values == ["20"]
    assert result.warnings == []


def test_validate_answer_calculations_allows_small_tolerance():
    result = validate_answer_calculations(
        "Revenue grew 20.04%.",
        [{"id": "growth", "value": "0.2000", "unit": "ratio"}],
        tolerance_percent_points="0.05",
    )
    assert result.ok is True


def test_validate_answer_calculations_flags_missing_expected_value():
    result = validate_answer_calculations(
        "Revenue increased.",
        [{"id": "growth", "value": "0.2000", "unit": "ratio"}],
    )
    assert result.ok is False
    assert result.missing_calculations == ["growth"]
    assert "missing" in result.warnings[0]


def test_validate_answer_calculations_flags_unsupported_percent():
    result = validate_answer_calculations(
        "Revenue grew 30%.",
        [{"id": "growth", "value": "0.2000", "unit": "ratio"}],
    )
    assert result.ok is False
    assert result.unsupported_values == ["30"]
    assert any("not supported" in warning for warning in result.warnings)


def test_validate_answer_calculations_supports_percent_unit():
    result = validate_answer_calculations(
        "Margin was 12.5%.",
        [{"id": "margin", "value": "12.50", "unit": "percent"}],
    )
    assert result.ok is True


def test_validate_answer_calculations_ignores_non_percent_calculations():
    result = validate_answer_calculations(
        "Total revenue was 120.",
        [{"id": "sum", "value": "120", "unit": "base"}],
    )
    assert result.ok is True
    assert result.expected_values == []


def test_validation_result_to_dict():
    result = validate_answer_calculations(
        "Revenue grew 20%.",
        [{"id": "growth", "value": "0.2", "unit": "ratio"}],
    ).to_dict()
    assert result["ok"] is True
    assert result["mentioned_values"] == ["20"]
