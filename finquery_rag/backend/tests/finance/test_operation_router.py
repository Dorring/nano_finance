"""Tests for metric lexicon and operation router (Phase 3 Commit 4)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import CalculationOperation, CalculationStatus
from src.finance.metric_lexicon import (
    GENERIC_OPERATIONS,
    NAMED_METRICS,
    find_generic_operation,
    find_metric_by_alias,
)
from src.finance.operation_router import (
    EXPLICIT_CALCULATION_PATTERNS,
    RoutingDecision,
    route_calculation,
)


# ---------------------------------------------------------------------------
# Metric lexicon: named metrics
# ---------------------------------------------------------------------------


class TestNamedMetrics:
    def test_has_five_named_metrics(self):
        assert len(NAMED_METRICS) == 5

    @pytest.mark.parametrize(
        "metric",
        ["gross_margin", "net_margin", "debt_ratio", "growth_rate", "percentage_share"],
    )
    def test_metric_definition_has_required_fields(self, metric):
        d = NAMED_METRICS[metric]
        assert d.metric == metric
        assert isinstance(d.operation, CalculationOperation)
        assert len(d.operand_roles) >= 2
        assert d.formula_version.endswith(".v1")
        assert len(d.aliases_en) > 0 or len(d.aliases_zh) > 0
        assert d.formula_template != ""
        assert d.unit == "ratio"

    @pytest.mark.parametrize(
        "text,expected_metric",
        [
            ("What is the gross margin?", "gross_margin"),
            ("Compute the gross profit margin", "gross_margin"),
            ("计算毛利率", "gross_margin"),
            ("Calculate the net margin", "net_margin"),
            ("计算净利率", "net_margin"),
            ("Compute the debt ratio", "debt_ratio"),
            ("计算资产负债率", "debt_ratio"),
            ("Calculate the growth rate", "growth_rate"),
            ("计算同比增长率", "growth_rate"),
            ("Compute the percentage share", "percentage_share"),
            ("计算占比", "percentage_share"),
        ],
    )
    def test_find_metric_by_alias(self, text, expected_metric):
        d = find_metric_by_alias(text)
        assert d is not None
        assert d.metric == expected_metric

    def test_find_metric_returns_none_for_no_match(self):
        assert find_metric_by_alias("What is the weather today?") is None

    def test_longer_alias_wins_over_shorter(self):
        """'gross profit margin' should match before 'gross margin'."""
        d = find_metric_by_alias("Calculate the gross profit margin")
        assert d.metric == "gross_margin"

    def test_operand_roles_are_consistent(self):
        assert NAMED_METRICS["gross_margin"].operand_roles == ("revenue", "cogs")
        assert NAMED_METRICS["net_margin"].operand_roles == ("revenue", "net_income")
        assert NAMED_METRICS["debt_ratio"].operand_roles == (
            "total_liabilities",
            "total_assets",
        )
        assert NAMED_METRICS["growth_rate"].operand_roles == ("current", "previous")
        assert NAMED_METRICS["percentage_share"].operand_roles == ("part", "total")


# ---------------------------------------------------------------------------
# Metric lexicon: generic operations
# ---------------------------------------------------------------------------


class TestGenericOperations:
    def test_has_four_generic_operations(self):
        assert len(GENERIC_OPERATIONS) == 4

    @pytest.mark.parametrize(
        "text,expected_op",
        [
            ("Calculate the sum of revenue and expenses", CalculationOperation.SUM),
            (
                "Compute the difference between Q1 and Q2",
                CalculationOperation.DIFFERENCE,
            ),
            ("Calculate the average of these numbers", CalculationOperation.AVERAGE),
            ("Convert 1.2 million to thousand", CalculationOperation.SCALE_CONVERSION),
        ],
    )
    def test_find_generic_operation(self, text, expected_op):
        entry = find_generic_operation(text)
        assert entry is not None
        assert entry.operation is expected_op

    def test_find_generic_returns_none_for_no_match(self):
        assert find_generic_operation("What is the gross margin?") is None


# ---------------------------------------------------------------------------
# Operation router: conservative gating
# ---------------------------------------------------------------------------


class TestRouteCalculationConservativeGating:
    """The router must be conservative — only route when all gates pass."""

    _CALC_INTENT = {"intent": "financial_calculation", "requires_retrieval": True}
    _DOC_QA_INTENT = {"intent": "document_qa", "requires_retrieval": True}
    _CONV_INTENT = {"intent": "conversation", "requires_retrieval": False}

    def test_non_calculation_intent_returns_not_applicable(self):
        decision = route_calculation("Calculate gross margin", self._DOC_QA_INTENT)
        assert decision.status is CalculationStatus.NOT_APPLICABLE
        assert "intent_not_financial_calculation" in decision.reason

    def test_conversational_intent_returns_not_applicable(self):
        decision = route_calculation("你好", self._CONV_INTENT)
        assert decision.status is CalculationStatus.NOT_APPLICABLE

    def test_calc_intent_without_explicit_verb_returns_not_applicable(self):
        """'毛利率是多少' has no explicit calculation verb → NOT_APPLICABLE."""
        decision = route_calculation("报表中显示的毛利率是多少", self._CALC_INTENT)
        assert decision.status is CalculationStatus.NOT_APPLICABLE
        assert "no_explicit_calculation_verb" in decision.reason

    def test_english_reported_metric_returns_not_applicable(self):
        """'What is the gross margin' → NOT_APPLICABLE (no calc verb)."""
        decision = route_calculation("What is the gross margin?", self._CALC_INTENT)
        assert decision.status is CalculationStatus.NOT_APPLICABLE
        assert "no_explicit_calculation_verb" in decision.reason

    def test_calc_intent_with_verb_but_no_metric_returns_not_applicable(self):
        """'Calculate the weather' has calc verb but no metric → NOT_APPLICABLE."""
        decision = route_calculation(
            "Calculate the weather forecast", self._CALC_INTENT
        )
        assert decision.status is CalculationStatus.NOT_APPLICABLE
        assert "no_metric_or_operation_matched" in decision.reason


# ---------------------------------------------------------------------------
# Operation router: successful routing
# ---------------------------------------------------------------------------


class TestRouteCalculationSuccessful:
    _CALC_INTENT = {"intent": "financial_calculation", "requires_retrieval": True}

    @pytest.mark.parametrize(
        "question,expected_op,expected_metric",
        [
            (
                "Calculate the gross margin from revenue and COGS",
                CalculationOperation.GROSS_MARGIN,
                "gross_margin",
            ),
            (
                "根据收入和营业成本计算毛利率",
                CalculationOperation.GROSS_MARGIN,
                "gross_margin",
            ),
            (
                "Compute the net margin from revenue and net income",
                CalculationOperation.NET_MARGIN,
                "net_margin",
            ),
            (
                "Calculate the debt ratio from liabilities and assets",
                CalculationOperation.DEBT_RATIO,
                "debt_ratio",
            ),
            (
                "计算同比增长率",
                CalculationOperation.GROWTH_RATE,
                "growth_rate",
            ),
            (
                "计算占比",
                CalculationOperation.PERCENTAGE_SHARE,
                "percentage_share",
            ),
        ],
    )
    def test_routes_named_metric(self, question, expected_op, expected_metric):
        decision = route_calculation(question, self._CALC_INTENT)
        assert decision.status is CalculationStatus.READY
        assert decision.operation is expected_op
        assert decision.metric == expected_metric
        assert decision.formula_version is not None
        assert decision.formula_version.endswith(".v1")
        assert len(decision.operand_roles) >= 2

    @pytest.mark.parametrize(
        "question,expected_op",
        [
            ("Calculate the sum of revenue and expenses", CalculationOperation.SUM),
            (
                "Compute the difference between Q1 and Q2 revenue",
                CalculationOperation.DIFFERENCE,
            ),
            (
                "Calculate the average of these three numbers",
                CalculationOperation.AVERAGE,
            ),
            ("Convert 1.2 million to billion", CalculationOperation.SCALE_CONVERSION),
        ],
    )
    def test_routes_generic_operation(self, question, expected_op):
        decision = route_calculation(question, self._CALC_INTENT)
        assert decision.status is CalculationStatus.READY
        assert decision.operation is expected_op
        # Generic operations have empty operand_roles (determined by evidence extractor)
        assert decision.operand_roles == ()

    def test_formula_version_matches_metric(self):
        decision = route_calculation("Calculate the gross margin", self._CALC_INTENT)
        assert decision.formula_version == "gross_margin.v1"

    def test_formula_template_is_human_readable(self):
        decision = route_calculation("Calculate the gross margin", self._CALC_INTENT)
        assert "revenue" in decision.formula_template
        assert "cogs" in decision.formula_template


# ---------------------------------------------------------------------------
# Operation router: RoutingDecision helpers
# ---------------------------------------------------------------------------


class TestRoutingDecisionHelpers:
    def test_not_applicable_factory(self):
        d = RoutingDecision.not_applicable("test_reason")
        assert d.status is CalculationStatus.NOT_APPLICABLE
        assert d.reason == "test_reason"
        assert d.operation is None

    def test_not_applicable_is_frozen(self):
        d = RoutingDecision.not_applicable("x")
        with pytest.raises(Exception):
            d.operation = CalculationOperation.SUM  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Explicit calculation patterns
# ---------------------------------------------------------------------------


class TestExplicitCalculationPatterns:
    @pytest.mark.parametrize(
        "pattern", ["calculate", "compute", "derive", "计算", "根据", "求", "换算"]
    )
    def test_pattern_exists(self, pattern):
        assert pattern in EXPLICIT_CALCULATION_PATTERNS

    def test_patterns_are_lowercased_for_english(self):
        """English patterns must be lowercase for case-insensitive matching."""
        for p in EXPLICIT_CALCULATION_PATTERNS:
            if p.isascii():
                assert p == p.lower()
