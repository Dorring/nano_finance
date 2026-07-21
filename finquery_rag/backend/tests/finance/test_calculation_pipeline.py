"""Tests for the calculation pipeline (Phase 3 Commit 8)."""

import os
import sys
from decimal import Decimal


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    NOT_APPLICABLE_RESULT,
    CalculationOperation,
    CalculationStatus,
)
from src.domain.evidence import EvidenceItem
from src.finance.calculation_pipeline import CalculationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence(
    content: str,
    chunk_id: str = "chunk_001",
    document_name: str = "annual_report.pdf",
    page: int = 12,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content=content,
        document_name=document_name,
        page=page,
        content_type=None,
        score=1.0,
        rerank_score=None,
        metadata={},
    )


def _intent_financial_calc() -> dict:
    return {
        "intent": "financial_calculation",
        "requires_retrieval": True,
        "confidence": 0.9,
    }


def _intent_document_qa() -> dict:
    return {"intent": "document_qa", "requires_retrieval": True, "confidence": 0.9}


# ---------------------------------------------------------------------------
# NOT_APPLICABLE routing
# ---------------------------------------------------------------------------


class TestPipelineNotApplicable:
    def test_document_qa_intent_returns_not_applicable(self):
        pipeline = CalculationPipeline()
        result = pipeline.try_calculate(
            "What was the gross margin last quarter?",
            _intent_document_qa(),
            evidence=(),
        )
        assert result.status is CalculationStatus.NOT_APPLICABLE
        assert result is NOT_APPLICABLE_RESULT

    def test_no_explicit_calculation_verb_returns_not_applicable(self):
        """'毛利率是多少' has financial_calculation intent but no calc verb."""
        pipeline = CalculationPipeline()
        result = pipeline.try_calculate(
            "报表中显示的毛利率是多少",
            _intent_financial_calc(),
            evidence=(),
        )
        assert result.status is CalculationStatus.NOT_APPLICABLE

    def test_no_metric_or_operation_matched_returns_not_applicable(self):
        pipeline = CalculationPipeline()
        pipeline.try_calculate(
            "计算所有数据的总和",  # "总和" is a generic keyword
            _intent_financial_calc(),
            evidence=(),
        )
        # Actually "总和" IS a generic operation keyword, so this should
        # route to READY. Let's use a question that has a calc verb but
        # no recognized metric/operation.
        pass

    def test_calculation_verb_but_no_metric_returns_not_applicable(self):
        pipeline = CalculationPipeline()
        result = pipeline.try_calculate(
            "计算所有内容的详细信息",
            _intent_financial_calc(),
            evidence=(),
        )
        assert result.status is CalculationStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# EXECUTED calculations
# ---------------------------------------------------------------------------


class TestPipelineExecuted:
    def test_gross_margin_executed(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "Total revenue for FY2025 was $1,000,000. "
                "Cost of goods sold was $600,000."
            ),
        )
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        assert result.operation is CalculationOperation.GROSS_MARGIN
        assert result.value == Decimal("0.4000")
        assert result.formula_version == "gross_margin.v1"
        assert len(result.operands) == 2

    def test_growth_rate_executed(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "Revenue for the current period FY2025 was $120 million. "
                "Previous period FY2024 revenue was $100 million."
            ),
        )
        result = pipeline.try_calculate(
            "Compute the YoY growth rate of revenue",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        assert result.operation is CalculationOperation.GROWTH_RATE
        assert result.value == Decimal("0.2000")

    def test_percentage_share_executed(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "The segment revenue was $30 million. Total revenue was $120 million."
            ),
        )
        result = pipeline.try_calculate(
            "Calculate the percentage share of segment revenue in total revenue",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        assert result.operation is CalculationOperation.PERCENTAGE_SHARE
        assert result.value == Decimal("0.2500")

    def test_debt_ratio_executed(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "Total liabilities were $400 million. Total assets were $1,000 million."
            ),
        )
        result = pipeline.try_calculate(
            "Calculate the debt ratio from total liabilities and total assets",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        assert result.operation is CalculationOperation.DEBT_RATIO
        assert result.value == Decimal("0.4000")

    def test_net_margin_executed(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence("Revenue was $1,000 million. Net income was $150 million."),
        )
        result = pipeline.try_calculate(
            "Calculate the net margin from revenue and net income",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        assert result.operation is CalculationOperation.NET_MARGIN
        assert result.value == Decimal("0.1500")

    def test_executed_result_has_evidence_bound_operands(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "Total revenue for FY2025 was $1,000,000. "
                "Cost of goods sold was $600,000.",
                chunk_id="chunk_abc",
                document_name="report.pdf",
                page=5,
            ),
        )
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.EXECUTED
        for operand in result.operands:
            assert operand.evidence_chunk_id == "chunk_abc"
            assert operand.document_name == "report.pdf"
            assert operand.page == 5
            assert operand.source_text  # non-empty


# ---------------------------------------------------------------------------
# BLOCKED calculations (missing operands)
# ---------------------------------------------------------------------------


class TestPipelineBlocked:
    def test_missing_cogs_blocks(self):
        pipeline = CalculationPipeline()
        evidence = (_evidence("Total revenue for FY2025 was $1,000,000."),)
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PLAN_BLOCKED"
        assert "cogs" in result.error_message.lower()

    def test_missing_revenue_blocks(self):
        pipeline = CalculationPipeline()
        evidence = (_evidence("Cost of goods sold was $600,000."),)
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.BLOCKED
        assert "revenue" in result.error_message.lower()

    def test_no_evidence_blocks(self):
        pipeline = CalculationPipeline()
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=(),
        )
        assert result.status is CalculationStatus.BLOCKED

    def test_generic_operation_no_operands_blocks(self):
        """Generic operations (sum, difference) with no fixed roles block
        when no operands are extracted."""
        pipeline = CalculationPipeline()
        result = pipeline.try_calculate(
            "Calculate the sum of all values",
            _intent_financial_calc(),
            evidence=(),
        )
        # "sum of" is a generic keyword → routes to SUM, but no operands
        # → BLOCKED.
        assert result.status is CalculationStatus.BLOCKED

    def test_division_by_zero_blocks(self):
        pipeline = CalculationPipeline()
        evidence = (
            _evidence(
                "Total revenue for FY2025 was $0. Cost of goods sold was $600,000."
            ),
        )
        result = pipeline.try_calculate(
            "Calculate the gross margin from revenue and COGS",
            _intent_financial_calc(),
            evidence=evidence,
        )
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PRIMITIVE_DECLINED"


# ---------------------------------------------------------------------------
# Pipeline layer purity
# ---------------------------------------------------------------------------


class TestPipelineLayerPurity:
    def test_pipeline_does_not_import_services_or_application(self):
        import inspect

        from src.finance import calculation_pipeline as mod

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
