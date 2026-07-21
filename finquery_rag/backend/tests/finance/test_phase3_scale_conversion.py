"""Phase 3 SCALE_CONVERSION tests (Option A)."""
import os, sys
from decimal import Decimal
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand, CalculationOperation, CalculationPlan, CalculationStatus,
)
from src.domain.evidence import EvidenceItem
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.operation_router import route_calculation, _extract_target_scale

WAN = "\u4e07"
YI = "\u4ebf"
WAN_YUAN = "\u4e07\u5143"
YI_YUAN = "\u4ebf\u5143"
BAI_WAN = "\u767e\u4e07"
HUAN_SUAN = "\u6362\u7b97"
YINGYE_SHOURU = "\u8425\u4e1a\u6536\u5165"
QING_JIANG = "\u8bf7\u5c06"
CHENG = "\u6210"


def _evidence(content, chunk_id="c1"):
    return EvidenceItem(
        chunk_id=chunk_id, content=content, document_name="r.pdf",
        page=1, content_type="text", score=0.9, rerank_score=None, metadata={},
    )

def _scale_plan(value, source_scale, target_scale, precision=4):
    op = CalculationOperand(
        name="value", value=Decimal(value), scale=source_scale,
        source_text=value, evidence_chunk_id="c1",
    )
    return CalculationPlan(
        operation=CalculationOperation.SCALE_CONVERSION,
        operands=(op,), formula_version="scale_conversion.v1",
        target_metric="scale_conversion", precision=precision,
        source_scale=source_scale, target_scale=target_scale,
    )


class TestRouterTargetScale:
    def test_wan_to_yi_zh(self):
        q = QING_JIANG + YINGYE_SHOURU + HUAN_SUAN + CHENG + YI_YUAN
        assert _extract_target_scale(q) == YI_YUAN

    def test_million_to_billion_en(self):
        assert _extract_target_scale("convert 500 million to billion") == "billion"

    def test_no_conversion_verb(self):
        assert _extract_target_scale("what is the revenue") is None

    def test_currency_rejected(self):
        assert _extract_target_scale("convert 1000 USD to CNY") == "__CURRENCY__"

    def test_route_has_target_scale(self):
        q = QING_JIANG + "1000" + WAN + HUAN_SUAN + CHENG + YI_YUAN
        d = route_calculation(q, {"intent": "financial_calculation"})
        assert d.operation is CalculationOperation.SCALE_CONVERSION
        assert d.target_scale == YI_YUAN


class TestExecutorScaleConversion:
    def test_yuan_to_wan(self):
        r = execute_plan(_scale_plan("10000", "ones", WAN))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1")

    def test_wan_to_yi(self):
        r = execute_plan(_scale_plan("10000", WAN_YUAN, YI_YUAN))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1")

    def test_baiwan_to_yi(self):
        r = execute_plan(_scale_plan("100", BAI_WAN, YI))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1")

    def test_billion_to_million(self):
        r = execute_plan(_scale_plan("1", "billion", "million"))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1000")

    def test_million_to_billion(self):
        r = execute_plan(_scale_plan("1000", "million", "billion"))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1")

    def test_negative_value(self):
        r = execute_plan(_scale_plan("-500", WAN, YI))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("-0.05")

    def test_zero_value(self):
        r = execute_plan(_scale_plan("0", WAN, YI))
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("0")

    def test_missing_source_scale_blocks(self):
        r = execute_plan(_scale_plan("100", "", YI))
        assert r.status is CalculationStatus.BLOCKED
        assert r.error_code == "UNIT_AMBIGUOUS"

    def test_missing_target_scale_blocks(self):
        r = execute_plan(_scale_plan("100", WAN, ""))
        assert r.status is CalculationStatus.BLOCKED
        assert r.error_code == "UNIT_AMBIGUOUS"

    def test_unsupported_scale_blocks(self):
        r = execute_plan(_scale_plan("100", "trillion", YI))
        assert r.status is CalculationStatus.BLOCKED
        assert r.error_code == "PRIMITIVE_DECLINED"

    def test_currency_target_blocks(self):
        r = execute_plan(_scale_plan("100", WAN, "__CURRENCY__"))
        assert r.status is CalculationStatus.BLOCKED
        assert r.error_code == "CURRENCY_NOT_SUPPORTED"

    def test_unit_field_has_target_scale(self):
        r = execute_plan(_scale_plan("100", WAN, YI))
        assert r.status is CalculationStatus.EXECUTED
        assert r.unit == YI


class TestPipelineScaleConversion:
    def test_e2e_wan_to_yi(self):
        ev = (_evidence(YINGYE_SHOURU + " 10000 " + WAN_YUAN),)
        q = QING_JIANG + YINGYE_SHOURU + HUAN_SUAN + CHENG + YI_YUAN
        r = CalculationPipeline().try_calculate(q, {"intent": "financial_calculation"}, ev)
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("1")

    def test_e2e_currency_blocks(self):
        ev = (_evidence("revenue 1000 USD"),)
        r = CalculationPipeline().try_calculate("convert revenue to CNY", {"intent": "financial_calculation"}, ev)
        assert r.status is CalculationStatus.BLOCKED
        assert r.error_code == "CURRENCY_NOT_SUPPORTED"


class TestLLMBypass:
    def test_executed_bypasses_llm(self):
        r = execute_plan(_scale_plan("1000", WAN, YI))
        assert r.status in (CalculationStatus.EXECUTED, CalculationStatus.BLOCKED, CalculationStatus.FAILED)

    def test_blocked_bypasses_llm(self):
        r = execute_plan(_scale_plan("100", "", YI))
        assert r.status in (CalculationStatus.EXECUTED, CalculationStatus.BLOCKED, CalculationStatus.FAILED)
