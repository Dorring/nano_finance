"""Tests for unit normalizer and evidence extractor (Phase 3 Commit 5)."""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationStatus,
)
from src.domain.evidence import EvidenceItem
from src.finance.evidence_extractor import (
    ROLE_KEYWORDS,
    extract_operands,
)
from src.finance.operation_router import RoutingDecision
from src.finance.unit_normalizer import (
    check_unit_consistency,
    detect_scale,
    detect_unit,
    normalize_operand_unit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    content: str, chunk_id: str = "c1", doc: str = "report.pdf", page: int = 1
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content=content,
        document_name=doc,
        page=page,
        content_type="text",
        score=1.0,
        rerank_score=None,
        metadata={},
    )


def _make_routing(
    roles: tuple[str, ...], metric: str = "gross_margin"
) -> RoutingDecision:
    return RoutingDecision(
        status=CalculationStatus.READY,
        operation=CalculationOperation.GROSS_MARGIN,
        metric=metric,
        formula_version=f"{metric}.v1",
        operand_roles=roles,
        formula_template="test",
        unit="ratio",
    )


# ---------------------------------------------------------------------------
# Unit normalizer: detect_unit
# ---------------------------------------------------------------------------


class TestDetectUnit:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Revenue was $1.2 million", "USD"),
            ("Revenue was US$1.2 million", "USD"),
            ("Revenue was USD 1.2 million", "USD"),
            ("Revenue was €1.2 million", "EUR"),
            ("Revenue was EUR 1.2 million", "EUR"),
            ("Revenue was £1.2 million", "GBP"),
            ("Revenue was 1.2 million CNY", "CNY"),
            ("Revenue was 1.2 million RMB", "CNY"),
            ("收入为1.2亿元人民币", "CNY"),
            ("Margin was 12.5%", "percent"),
        ],
    )
    def test_detect_currency(self, text, expected):
        assert detect_unit(text) == expected

    def test_detect_unit_returns_none_for_bare_number(self):
        assert detect_unit("Revenue was 1.2 million") is None

    def test_detect_unit_returns_none_for_empty(self):
        assert detect_unit("") is None
        assert detect_unit(None) is None


# ---------------------------------------------------------------------------
# Unit normalizer: detect_scale
# ---------------------------------------------------------------------------


class TestDetectScale:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("$1.2 million", "million"),
            ("1.5 billion", "billion"),
            ("3.5 thousand", "thousand"),
            ("1.2亿", "亿"),
            ("3.5万", "万"),
        ],
    )
    def test_detect_scale(self, text, expected):
        assert detect_scale(text) == expected

    def test_detect_scale_returns_none_for_bare_number(self):
        assert detect_scale("Revenue was 1.2") is None


# ---------------------------------------------------------------------------
# Unit normalizer: check_unit_consistency
# ---------------------------------------------------------------------------


class TestCheckUnitConsistency:
    def test_all_same_currency_is_consistent(self):
        ops = (
            CalculationOperand(
                name="revenue", value=Decimal("1000"), source_text="$1,000"
            ),
            CalculationOperand(name="cogs", value=Decimal("600"), source_text="$600"),
        )
        assert check_unit_consistency(ops) is True

    def test_different_currencies_are_inconsistent(self):
        ops = (
            CalculationOperand(
                name="revenue", value=Decimal("1000"), source_text="$1,000"
            ),
            CalculationOperand(name="cogs", value=Decimal("600"), source_text="€600"),
        )
        assert check_unit_consistency(ops) is False

    def test_no_units_is_consistent(self):
        ops = (
            CalculationOperand(
                name="revenue", value=Decimal("1000"), source_text="1000"
            ),
            CalculationOperand(name="cogs", value=Decimal("600"), source_text="600"),
        )
        assert check_unit_consistency(ops) is True

    def test_mixed_unit_and_no_unit_is_consistent(self):
        ops = (
            CalculationOperand(
                name="revenue", value=Decimal("1000"), source_text="$1,000"
            ),
            CalculationOperand(name="cogs", value=Decimal("600"), source_text="600"),
        )
        assert check_unit_consistency(ops) is True

    def test_percent_and_currency_are_inconsistent(self):
        ops = (
            CalculationOperand(
                name="revenue", value=Decimal("1000"), source_text="$1,000"
            ),
            CalculationOperand(name="margin", value=Decimal("0.12"), source_text="12%"),
        )
        assert check_unit_consistency(ops) is False


# ---------------------------------------------------------------------------
# Unit normalizer: normalize_operand_unit
# ---------------------------------------------------------------------------


class TestNormalizeOperandUnit:
    def test_uses_explicit_unit_when_set(self):
        op = CalculationOperand(
            name="x", value=Decimal("1"), unit="USD", source_text="1000"
        )
        assert normalize_operand_unit(op) == "USD"

    def test_infers_from_source_text_when_unit_not_set(self):
        op = CalculationOperand(name="x", value=Decimal("1"), source_text="$1,000")
        assert normalize_operand_unit(op) == "USD"

    def test_returns_none_when_neither_set(self):
        op = CalculationOperand(name="x", value=Decimal("1"), source_text="1000")
        assert normalize_operand_unit(op) is None


# ---------------------------------------------------------------------------
# Evidence extractor: ROLE_KEYWORDS
# ---------------------------------------------------------------------------


class TestRoleKeywords:
    @pytest.mark.parametrize(
        "role",
        [
            "revenue",
            "cogs",
            "net_income",
            "total_liabilities",
            "total_assets",
            "current",
            "previous",
            "part",
            "total",
        ],
    )
    def test_role_has_keywords(self, role):
        assert role in ROLE_KEYWORDS
        kw = ROLE_KEYWORDS[role]
        assert "en" in kw or "zh" in kw

    def test_revenue_has_both_en_and_zh(self):
        assert "en" in ROLE_KEYWORDS["revenue"]
        assert "zh" in ROLE_KEYWORDS["revenue"]


# ---------------------------------------------------------------------------
# Evidence extractor: extract_operands for named metrics
# ---------------------------------------------------------------------------


class TestExtractOperandsNamedMetrics:
    def test_extract_revenue_and_cogs_for_gross_margin(self):
        evidence = (
            _make_evidence("Total revenue for FY2025 was $1,000,000.", chunk_id="c1"),
            _make_evidence("Cost of goods sold was $600,000 in FY2025.", chunk_id="c2"),
        )
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert len(result.operands) == 2
        assert result.operands[0].name == "revenue"
        assert result.operands[0].value == Decimal("1000000")
        assert result.operands[0].evidence_chunk_id == "c1"
        assert result.operands[1].name == "cogs"
        assert result.operands[1].value == Decimal("600000")
        assert result.operands[1].evidence_chunk_id == "c2"

    def test_extract_revenue_and_net_income_for_net_margin(self):
        evidence = (
            _make_evidence("Revenue was $2,000,000.", chunk_id="c1"),
            _make_evidence("Net income was $200,000.", chunk_id="c2"),
        )
        routing = _make_routing(("revenue", "net_income"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert result.operands[0].value == Decimal("2000000")
        assert result.operands[1].value == Decimal("200000")

    def test_extract_liabilities_and_assets_for_debt_ratio(self):
        evidence = (
            _make_evidence("Total liabilities were $500,000.", chunk_id="c1"),
            _make_evidence("Total assets were $1,000,000.", chunk_id="c2"),
        )
        routing = _make_routing(("total_liabilities", "total_assets"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert result.operands[0].value == Decimal("500000")
        assert result.operands[1].value == Decimal("1000000")

    def test_extract_with_scale_words(self):
        evidence = (
            _make_evidence("Revenue was $1.2 million.", chunk_id="c1"),
            _make_evidence("COGS was $0.6 million.", chunk_id="c2"),
        )
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert result.operands[0].value == Decimal("1200000")
        assert result.operands[1].value == Decimal("600000")

    def test_extract_with_chinese_keywords(self):
        evidence = (
            _make_evidence("营业收入为1,000,000元。", chunk_id="c1"),
            _make_evidence("营业成本为600,000元。", chunk_id="c2"),
        )
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert result.operands[0].value == Decimal("1000000")
        assert result.operands[1].value == Decimal("600000")

    def test_extract_with_chinese_scale(self):
        evidence = (
            _make_evidence("营业收入为1.2亿元。", chunk_id="c1"),
            _make_evidence("营业成本为6千万元。", chunk_id="c2"),
        )
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert result.operands[0].value == Decimal("120000000")
        assert result.operands[1].value == Decimal("60000000")


# ---------------------------------------------------------------------------
# Evidence extractor: missing operands
# ---------------------------------------------------------------------------


class TestExtractOperandsMissing:
    def test_missing_cogs_returns_partial_result(self):
        evidence = (_make_evidence("Total revenue was $1,000,000.", chunk_id="c1"),)
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert not result.all_found
        assert "cogs" in result.missing_roles
        assert "revenue" in result.found_roles
        assert len(result.operands) == 1

    def test_all_missing_returns_empty(self):
        evidence = (_make_evidence("The weather was sunny today.", chunk_id="c1"),)
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands(evidence, routing)

        assert not result.all_found
        assert len(result.operands) == 0
        assert set(result.missing_roles) == {"revenue", "cogs"}

    def test_empty_evidence_returns_empty(self):
        routing = _make_routing(("revenue", "cogs"))
        result = extract_operands((), routing)

        assert not result.all_found
        assert len(result.operands) == 0


# ---------------------------------------------------------------------------
# Evidence extractor: generic operations (no fixed roles)
# ---------------------------------------------------------------------------


class TestExtractOperandsGeneric:
    def test_empty_roles_returns_empty_result(self):
        evidence = (_make_evidence("Some content", chunk_id="c1"),)
        routing = RoutingDecision(
            status=CalculationStatus.READY,
            operation=CalculationOperation.SUM,
            metric="sum",
            formula_version="sum.v1",
            operand_roles=(),  # generic operation
        )
        result = extract_operands(evidence, routing)
        assert len(result.operands) == 0
        assert result.expected_roles == ()


# ---------------------------------------------------------------------------
# Evidence extractor: source_text binding
# ---------------------------------------------------------------------------


class TestSourceTextBinding:
    def test_source_text_contains_the_raw_number(self):
        evidence = (_make_evidence("Revenue was $1,234,567.", chunk_id="c1"),)
        routing = _make_routing(("revenue",))
        result = extract_operands(evidence, routing)

        assert result.all_found
        op = result.operands[0]
        assert "1,234,567" in op.source_text or "1234567" in op.source_text
        assert op.evidence_chunk_id == "c1"
        assert op.document_name == "report.pdf"
        assert op.page == 1

    def test_source_text_captures_scale_word(self):
        evidence = (_make_evidence("Revenue was $1.2 million.", chunk_id="c1"),)
        routing = _make_routing(("revenue",))
        result = extract_operands(evidence, routing)

        assert result.all_found
        op = result.operands[0]
        assert "million" in op.source_text.lower() or op.value == Decimal("1200000")


# ---------------------------------------------------------------------------
# Evidence extractor: ambiguity warnings
# ---------------------------------------------------------------------------


class TestAmbiguityWarnings:
    def test_warning_when_multiple_candidates_found(self):
        evidence = (
            _make_evidence("Revenue was $1,000,000.", chunk_id="c1"),
            _make_evidence("Revenue was $1,200,000.", chunk_id="c2"),
        )
        routing = _make_routing(("revenue",))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert any("2 candidate" in w for w in result.warnings)

    def test_warning_when_sentence_has_multiple_numbers(self):
        evidence = (
            _make_evidence(
                "Revenue was $1,000,000 and COGS was $600,000.", chunk_id="c1"
            ),
        )
        routing = _make_routing(("revenue",))
        result = extract_operands(evidence, routing)

        assert result.all_found
        assert any("2 numbers" in w for w in result.warnings)
