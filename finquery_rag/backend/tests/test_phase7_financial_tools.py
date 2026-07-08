"""Phase 7A tests: deterministic financial calculation tools."""
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.financial_tools import (
    convert_scale,
    format_ratio_percent,
    growth_rate,
    parse_financial_number,
    percentage_share,
    sum_values,
    verify_sum,
)


def test_parse_plain_and_comma_number():
    assert parse_financial_number("1,234.50").value == Decimal("1234.50")


def test_parse_accounting_negative():
    assert parse_financial_number("(1,234)").value == Decimal("-1234")


def test_parse_percent_as_ratio():
    result = parse_financial_number("12.5%")
    assert result.ok is True
    assert result.value == Decimal("0.125")
    assert result.details["is_percent"] is True


def test_parse_scale_words():
    assert parse_financial_number("$1.2 million").value == Decimal("1200000.0")
    assert parse_financial_number("3亿元").value == Decimal("300000000")


def test_parse_unknown_returns_error():
    result = parse_financial_number("not a number")
    assert result.ok is False
    assert "Cannot parse" in result.error


def test_growth_rate():
    result = growth_rate("120", "100", precision=4)
    assert result.ok is True
    assert result.value == Decimal("0.2000")
    assert result.unit == "ratio"


def test_growth_rate_zero_previous_error():
    result = growth_rate("120", "0")
    assert result.ok is False
    assert "zero previous" in result.error


def test_percentage_share():
    result = percentage_share("25", "200", precision=4)
    assert result.value == Decimal("0.1250")


def test_sum_values_quantized():
    result = sum_values(["1.111", "2.222"], precision=2)
    assert result.value == Decimal("3.33")
    assert result.details["count"] == 2


def test_verify_sum_passes_with_tolerance():
    result = verify_sum(["1.00", "2.00"], "3.004", tolerance="0.01")
    assert result.ok is True
    assert result.details["computed_total"] == "3.00"


def test_verify_sum_fails_outside_tolerance():
    result = verify_sum(["1.00", "2.00"], "3.50", tolerance="0.01")
    assert result.ok is False
    assert "does not match" in result.error


def test_convert_scale_million_to_billion():
    result = convert_scale("1500", "million", "billion", precision=4)
    assert result.ok is True
    assert result.value == Decimal("1.5000")
    assert result.unit == "billion"


def test_convert_scale_unknown_error():
    result = convert_scale("1", "weird", "million")
    assert result.ok is False
    assert "Unknown source scale" in result.error


def test_format_ratio_percent():
    result = format_ratio_percent("0.125", precision=2)
    assert result.ok is True
    assert result.value == Decimal("12.50")
    assert result.unit == "percent"


def test_tool_result_to_dict_serializes_decimal():
    result = growth_rate("120", "100", precision=2).to_dict()
    assert result["value"] == "0.20"
    assert result["unit"] == "ratio"
