"""Tests for grounding metrics in src.evaluation.metrics."""
from __future__ import annotations

from src.evaluation.metrics import (
    answer_calculation_consistency,
    calculation_accuracy,
    citation_f1,
    citation_precision,
    citation_recall,
    formula_version_accuracy,
    numeric_accuracy,
)
from src.evaluation.schemas import ExpectedCalculation, ExpectedSource


def _src(filename: str, page: int | None = None) -> ExpectedSource:
    return ExpectedSource(filename=filename, page=page)


def _calc(
    calc_id: str = "c1",
    operation: str = "sum_values",
    expected_value: str = "300",
    tolerance: str = "0",
    unit: str | None = None,
) -> ExpectedCalculation:
    return ExpectedCalculation(
        calc_id=calc_id,
        operation=operation,
        args={},
        expected_value=expected_value,
        tolerance=tolerance,
        unit=unit,
    )


class TestCitationPrecision:
    def test_citation_precision_perfect(self) -> None:
        """All cited sources match expected sources."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        sources = [
            {"filename": "a.pdf", "page": 1},
            {"filename": "b.pdf", "page": 2},
        ]
        assert citation_precision(expected, sources) == 1.0

    def test_citation_precision_partial(self) -> None:
        expected = [_src("a.pdf", page=1)]
        sources = [
            {"filename": "a.pdf", "page": 1},
            {"filename": "c.pdf", "page": 3},
        ]
        assert citation_precision(expected, sources) == 0.5


class TestCitationRecall:
    def test_citation_recall_partial(self) -> None:
        """Half of expected sources are cited."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        sources = [{"filename": "a.pdf", "page": 1}]
        assert citation_recall(expected, sources) == 0.5

    def test_citation_recall_full(self) -> None:
        expected = [_src("a.pdf", page=1)]
        sources = [{"filename": "a.pdf", "page": 1}]
        assert citation_recall(expected, sources) == 1.0


class TestCitationF1:
    def test_f1_perfect(self) -> None:
        assert citation_f1(1.0, 1.0) == 1.0

    def test_f1_zero(self) -> None:
        assert citation_f1(0.0, 0.0) == 0.0

    def test_f1_half(self) -> None:
        assert citation_f1(1.0, 0.5) == 2 * 1.0 * 0.5 / 1.5


class TestNumericAccuracy:
    def test_numeric_accuracy_correct(self) -> None:
        """All expected numbers found in answer."""
        answer = "The revenue is 1,000 and profit is 500."
        expected = ["1000", "500"]
        assert numeric_accuracy(answer, expected) == 1.0

    def test_numeric_accuracy_wrong(self) -> None:
        """Expected numbers not in answer."""
        answer = "The revenue is 999."
        expected = ["1000", "500"]
        assert numeric_accuracy(answer, expected) == 0.0

    def test_numeric_accuracy_with_commas(self) -> None:
        answer = "Revenue: 1,234,567"
        assert numeric_accuracy(answer, ["1234567"]) == 1.0

    def test_numeric_accuracy_with_percent(self) -> None:
        answer = "Growth: 15.5%"
        assert numeric_accuracy(answer, ["15.5"]) == 1.0


class TestCalculationAccuracy:
    def test_calculation_accuracy_correct(self) -> None:
        """Prediction calculations match expected."""
        expected = [_calc("c1", "sum_values", "300")]
        predictions = [{"id": "c1", "operation": "sum_values", "value": "300"}]
        assert calculation_accuracy(expected, predictions) == 1.0

    def test_calculation_accuracy_wrong_value(self) -> None:
        expected = [_calc("c1", "sum_values", "300")]
        predictions = [{"id": "c1", "operation": "sum_values", "value": "999"}]
        assert calculation_accuracy(expected, predictions) == 0.0

    def test_calculation_accuracy_with_tolerance(self) -> None:
        expected = [_calc("c1", "sum_values", "300", tolerance="0.5")]
        predictions = [{"id": "c1", "operation": "sum_values", "value": "300.3"}]
        assert calculation_accuracy(expected, predictions) == 1.0

    def test_calculation_accuracy_no_predictions(self) -> None:
        expected = [_calc("c1", "sum_values", "300")]
        assert calculation_accuracy(expected, []) == 0.0


class TestAnswerCalculationConsistency:
    def test_answer_calculation_consistency(self) -> None:
        """Calculation values appear in the answer."""
        answer = "The total is 300 yuan."
        calcs = [{"id": "c1", "operation": "sum_values", "value": "300"}]
        assert answer_calculation_consistency(answer, calcs) == 1.0

    def test_consistency_value_missing(self) -> None:
        answer = "The total is 999."
        calcs = [{"id": "c1", "operation": "sum_values", "value": "300"}]
        assert answer_calculation_consistency(answer, calcs) == 0.0


class TestFormulaVersionAccuracy:
    def test_formula_version_accuracy(self) -> None:
        """Correct operations used in predictions."""
        expected = [
            _calc("c1", "sum_values", "300"),
            _calc("c2", "growth_rate", "0.15"),
        ]
        predictions = [
            {"id": "c1", "operation": "sum_values", "value": "300"},
            {"id": "c2", "operation": "growth_rate", "value": "0.15"},
        ]
        assert formula_version_accuracy(expected, predictions) == 1.0

    def test_formula_version_wrong_operation(self) -> None:
        expected = [_calc("c1", "sum_values", "300")]
        predictions = [{"id": "c1", "operation": "growth_rate", "value": "300"}]
        assert formula_version_accuracy(expected, predictions) == 0.0
