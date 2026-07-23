"""Tests for slice reporting in src.evaluation.slices."""
from __future__ import annotations

from src.evaluation.schemas import EvaluationLabel, EvaluationPrediction
from src.evaluation.slices import SLICE_CATEGORIES, compute_slice_metrics


def _label(case_id: str, slice_tags: tuple[str, ...]) -> EvaluationLabel:
    return EvaluationLabel(
        case_id=case_id,
        expected_sources=(),
        expected_numbers=("100",),
        expected_calculations=(),
        expected_intent=None,
        expected_answerability=None,
        expected_validation_status=None,
        expected_no_answer=False,
        required_answer_terms=(),
        forbidden_answer_terms=(),
        slice_tags=slice_tags,
    )


def _pred(case_id: str, answer: str = "100") -> EvaluationPrediction:
    return EvaluationPrediction(
        case_id=case_id,
        answer=answer,
        sources=(),
        retrieved_chunks=(),
        calculations=(),
        answerability=None,
        validation=None,
        warnings=(),
        intent=None,
        intent_confidence=None,
        context_sufficient=None,
        retrieval_debug={},
        trace_id=None,
        latency_ms=100.0,
        error_code=None,
    )


def _dummy_metrics_func(labels, predictions) -> dict:
    """Simple metrics function that returns a count-based metric."""
    return {"case_count": len(labels)}


class TestSliceSampleCount:
    def test_slice_sample_count_reported(self) -> None:
        """Every slice must report sample_count."""
        labels = [
            _label("c1", ("document_qa", "english")),
            _label("c2", ("financial_calculation", "chinese")),
        ]
        preds = [_pred("c1"), _pred("c2")]
        result = compute_slice_metrics(labels, preds, _dummy_metrics_func)
        for category, slices in result.items():
            for slice_name, metrics in slices.items():
                assert "sample_count" in metrics, (
                    f"slice {category}/{slice_name} missing sample_count"
                )


class TestSliceMetricsMatchSubset:
    def test_slice_metrics_match_subset(self) -> None:
        """Slice metrics should match manually computed subset."""
        labels = [
            _label("c1", ("document_qa",)),
            _label("c2", ("document_qa",)),
            _label("c3", ("financial_calculation",)),
        ]
        preds = [_pred("c1"), _pred("c2"), _pred("c3")]
        result = compute_slice_metrics(labels, preds, _dummy_metrics_func)
        assert result["Intent"]["document_qa"]["sample_count"] == 2
        assert result["Intent"]["document_qa"]["case_count"] == 2
        assert result["Intent"]["financial_calculation"]["sample_count"] == 1
        assert result["Intent"]["financial_calculation"]["case_count"] == 1


class TestEmptySlice:
    def test_empty_slice_has_zero_count(self) -> None:
        """Slices with no cases should have sample_count=0."""
        labels = [_label("c1", ("document_qa",))]
        preds = [_pred("c1")]
        result = compute_slice_metrics(labels, preds, _dummy_metrics_func)
        # 'conversation' has no cases
        assert result["Intent"]["conversation"]["sample_count"] == 0
        # 'multi_hop' has no cases
        assert result["Difficulty"]["multi_hop"]["sample_count"] == 0
        # Verify empty slice only has sample_count key
        assert set(result["Intent"]["conversation"].keys()) == {"sample_count"}


class TestSliceCategoriesComplete:
    def test_all_categories_present(self) -> None:
        """All SLICE_CATEGORIES must appear in the result."""
        labels = []
        preds = []
        result = compute_slice_metrics(labels, preds, _dummy_metrics_func)
        assert set(result.keys()) == set(SLICE_CATEGORIES.keys())
