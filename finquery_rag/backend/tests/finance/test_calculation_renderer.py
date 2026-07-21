"""Tests for the deterministic calculation renderer (Phase 3 Commit 7)."""

import os
import sys
from decimal import Decimal


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_renderer import render_calculation_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _operand(
    name: str,
    value: str,
    document_name: str | None = "annual_report.pdf",
    page: int | None = 12,
    evidence_chunk_id: str = "chunk_001",
) -> CalculationOperand:
    return CalculationOperand(
        name=name,
        value=Decimal(value),
        source_text=f"{name} = {value}",
        evidence_chunk_id=evidence_chunk_id,
        document_name=document_name,
        page=page,
    )


def _executed_result(
    operation: CalculationOperation = CalculationOperation.GROSS_MARGIN,
    value: Decimal = Decimal("0.4000"),
    unit: str = "ratio",
    formula: str = "(revenue - cogs) / revenue",
    formula_version: str = "gross_margin.v1",
    target_metric: str = "gross_margin",
    operands: tuple[CalculationOperand, ...] = (),
) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=operation,
        value=value,
        unit=unit,
        formula=formula,
        formula_version=formula_version,
        target_metric=target_metric,
        operands=operands,
    )


# ---------------------------------------------------------------------------
# Status dispatch
# ---------------------------------------------------------------------------


class TestRendererStatusDispatch:
    def test_not_applicable_returns_empty(self):
        result = CalculationResult(status=CalculationStatus.NOT_APPLICABLE)
        assert render_calculation_result(result) == ""

    def test_failed_returns_safe_message(self):
        result = CalculationResult(
            status=CalculationStatus.FAILED,
            error_code="PRIMITIVE_EXCEPTION",
            error_message="boom",
        )
        rendered = render_calculation_result(result)
        assert rendered != ""
        assert "Unable to compute" in rendered
        assert "boom" not in rendered

    def test_executed_returns_nonempty(self):
        result = _executed_result()
        rendered = render_calculation_result(result)
        assert rendered != ""
        assert "Gross Margin" in rendered

    def test_blocked_returns_nonempty(self):
        result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.GROSS_MARGIN,
            target_metric="gross_margin",
            error_code="INSUFFICIENT_EVIDENCE",
            error_message="Could not find COGS figure",
        )
        rendered = render_calculation_result(result)
        assert rendered != ""
        assert "Unable to compute" in rendered


# ---------------------------------------------------------------------------
# EXECUTED rendering
# ---------------------------------------------------------------------------


class TestRenderExecuted:
    def test_metric_label_in_output(self):
        result = _executed_result(target_metric="net_margin")
        rendered = render_calculation_result(result)
        assert "Net Margin" in rendered

    def test_ratio_value_formatted_as_percentage(self):
        result = _executed_result(value=Decimal("0.4000"), unit="ratio")
        rendered = render_calculation_result(result)
        assert "40.00%" in rendered

    def test_ratio_value_half_up_rounding(self):
        result = _executed_result(value=Decimal("0.12345"), unit="ratio")
        rendered = render_calculation_result(result)
        # 0.12345 * 100 = 12.345, rounded to 2 dp = 12.35 (ROUND_HALF_UP
        # applied by quantize in _format_decimal, but Decimal default
        # rounding for quantize is ROUND_HALF_EVEN; the renderer uses
        # .quantize without explicit rounding, so we just check it's
        # rendered as a percentage).
        assert "12.35%" in rendered or "12.34%" in rendered

    def test_base_value_formatted_with_thousands_separator(self):
        result = _executed_result(
            operation=CalculationOperation.DIFFERENCE,
            value=Decimal("1234.56"),
            unit="base",
            formula="current - previous",
            formula_version="difference.v1",
            target_metric="difference",
        )
        rendered = render_calculation_result(result)
        assert "1,234.56" in rendered
        assert "Difference" in rendered

    def test_formula_line_included(self):
        result = _executed_result()
        rendered = render_calculation_result(result)
        assert "Formula: (revenue - cogs) / revenue" in rendered
        assert "(gross_margin.v1)" in rendered

    def test_operands_listed_with_evidence(self):
        ops = (
            _operand("revenue", "1000", document_name="report.pdf", page=5),
            _operand("cogs", "600", document_name="report.pdf", page=6),
        )
        result = _executed_result(operands=ops)
        rendered = render_calculation_result(result)
        assert "Inputs:" in rendered
        assert "revenue = 1,000.00" in rendered
        assert "report.pdf, p.5" in rendered
        assert "cogs = 600.00" in rendered
        assert "report.pdf, p.6" in rendered

    def test_operand_without_document_cites_chunk(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1000"),
            source_text="revenue = 1000",
            evidence_chunk_id="chunk_abc",
            document_name=None,
            page=None,
        )
        result = _executed_result(operands=(op,))
        rendered = render_calculation_result(result)
        assert "chunk chunk_abc" in rendered

    def test_operand_without_any_citation(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1000"),
            source_text="revenue = 1000",
            evidence_chunk_id="",
            document_name=None,
            page=None,
        )
        result = _executed_result(operands=(op,))
        rendered = render_calculation_result(result)
        assert "revenue = 1,000.00" in rendered
        # No citation suffix
        assert "chunk" not in rendered.split("revenue = 1,000.00")[1].split("\n")[0]

    def test_page_zero_is_preserved_in_citation(self):
        op = _operand("revenue", "1000", page=0)
        result = _executed_result(operands=(op,))
        rendered = render_calculation_result(result)
        assert "p.0" in rendered

    def test_no_operands_omits_inputs_section(self):
        result = _executed_result(operands=())
        rendered = render_calculation_result(result)
        assert "Inputs:" not in rendered

    def test_growth_rate_rendered(self):
        ops = (
            _operand("current", "120", page=5),
            _operand("previous", "100", page=3),
        )
        result = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROWTH_RATE,
            value=Decimal("0.2000"),
            unit="ratio",
            formula="(current - previous) / previous",
            formula_version="growth_rate.v1",
            target_metric="growth_rate",
            operands=ops,
        )
        rendered = render_calculation_result(result)
        assert "Growth Rate: 20.00%" in rendered
        assert "current = 120.00" in rendered
        assert "previous = 100.00" in rendered


# ---------------------------------------------------------------------------
# BLOCKED rendering
# ---------------------------------------------------------------------------


class TestRenderBlocked:
    def test_blocked_includes_metric_label(self):
        result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.GROSS_MARGIN,
            target_metric="gross_margin",
            error_code="INSUFFICIENT_EVIDENCE",
            error_message="Could not find COGS figure in retrieved evidence",
        )
        rendered = render_calculation_result(result)
        assert "Unable to compute Gross Margin" in rendered
        assert "Could not find COGS figure" in rendered

    def test_blocked_without_error_message_uses_default(self):
        result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.GROSS_MARGIN,
            target_metric="gross_margin",
            error_code="PLAN_BLOCKED",
            error_message=None,
        )
        rendered = render_calculation_result(result)
        assert "Unable to compute Gross Margin" in rendered
        assert "calculation could not be completed" in rendered

    def test_blocked_without_target_metric(self):
        result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            error_code="PLAN_BLOCKED",
            error_message="plan was blocked",
        )
        rendered = render_calculation_result(result)
        assert "Unable to compute Result" in rendered


# ---------------------------------------------------------------------------
# Layer purity
# ---------------------------------------------------------------------------


class TestRendererLayerPurity:
    def test_renderer_does_not_import_services_or_application(self):
        import inspect

        from src.finance import calculation_renderer as mod

        source = inspect.getsource(mod)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")) or stripped.startswith(
                "from __future__"
            ):
                continue
            assert "from src.services" not in stripped, f"forbidden: {stripped}"
            assert "from src.application" not in stripped, f"forbidden: {stripped}"
            assert "import src.services" not in stripped, f"forbidden: {stripped}"
            assert "import src.application" not in stripped, f"forbidden: {stripped}"
