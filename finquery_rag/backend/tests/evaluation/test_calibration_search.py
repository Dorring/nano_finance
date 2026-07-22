"""Tests for src.evaluation.calibration search and selection."""
from __future__ import annotations

import copy
import random

from src.evaluation.calibration import (
    eliminate_unsafe_candidates,
    search_calibration_space,
    select_best_candidate,
)
from src.evaluation.schemas import (
    EvaluationLabel,
    EvaluationPrediction,
    ExpectedSource,
)

_BASELINE_METRICS = {
    "macro_strict_pass_rate": 0.5,
    "unsupported_numeric_release_rate": 0.0,
    "invalid_citation_release_rate": 0.0,
    "calculation_mismatch_release_rate": 0.0,
    "false_block_rate": 0.0,
    "unsafe_answer_rate": 0.0,
    "validator_fail_closed_rate": 0.0,
    "citation_recall": 0.5,
    "p95_latency_ms": 100.0,
}


def _label(case_id: str = "c1") -> EvaluationLabel:
    return EvaluationLabel(
        case_id=case_id,
        expected_sources=(ExpectedSource(filename="a.pdf", page=1),),
        expected_numbers=(),
        expected_calculations=(),
        expected_intent=None,
        expected_answerability=None,
        expected_validation_status=None,
        expected_no_answer=False,
        required_answer_terms=(),
        forbidden_answer_terms=(),
        slice_tags=("direct",),
    )


def _pred(
    case_id: str = "c1",
    chunks: tuple[dict, ...] = (),
    answer: str = "Revenue was 1000.",
    sources: tuple[dict, ...] = (),
) -> EvaluationPrediction:
    return EvaluationPrediction(
        case_id=case_id,
        answer=answer,
        sources=sources,
        retrieved_chunks=chunks,
        calculations=(),
        answerability={"status": "answerable"},
        validation={"status": "passed"},
        warnings=(),
        intent="document_qa",
        intent_confidence=0.9,
        context_sufficient=True,
        retrieval_debug={},
        trace_id=None,
        latency_ms=50.0,
        error_code=None,
    )


def _make_labels_predictions() -> tuple[list, list]:
    """Create two labels and two predictions with scored chunks."""
    labels = [_label("c1"), _label("c2")]
    preds = [
        _pred(
            "c1",
            chunks=(
                {"filename": "a.pdf", "page": 1, "score": 0.9, "text": "Revenue 1000"},
                {"filename": "b.pdf", "page": 2, "score": 0.3, "text": "Other info"},
            ),
            sources=({"filename": "a.pdf", "page": 1},),
        ),
        _pred(
            "c2",
            chunks=(
                {"filename": "a.pdf", "page": 1, "score": 0.8, "text": "Revenue 2000"},
                {"filename": "c.pdf", "page": 3, "score": 0.2, "text": "Noise"},
            ),
            sources=({"filename": "a.pdf", "page": 1},),
        ),
    ]
    return labels, preds


_SMALL_SEARCH_SPACE = {
    "n_results": [1, 2],
    "min_score_threshold": [0.0, 0.5],
}


class TestSearchCalibrationSpace:
    def test_search_returns_all_candidates(self) -> None:
        """Number of candidates equals the product of search-space sizes."""
        labels, preds = _make_labels_predictions()
        candidates = search_calibration_space(
            labels, preds, _SMALL_SEARCH_SPACE, _BASELINE_METRICS
        )
        expected = 1
        for values in _SMALL_SEARCH_SPACE.values():
            expected *= len(values)
        assert len(candidates) == expected
        for cand in candidates:
            assert "params" in cand
            assert "metrics" in cand
            assert "safe" in cand

    def test_result_independent_of_case_order(self) -> None:
        """Shuffling labels/predictions must not change candidate params or metrics."""
        labels, preds = _make_labels_predictions()
        candidates_a = search_calibration_space(
            labels, preds, _SMALL_SEARCH_SPACE, _BASELINE_METRICS
        )
        shuffled_labels = list(reversed(labels))
        shuffled_preds = list(reversed(preds))
        candidates_b = search_calibration_space(
            shuffled_labels,
            shuffled_preds,
            _SMALL_SEARCH_SPACE,
            _BASELINE_METRICS,
        )
        assert len(candidates_a) == len(candidates_b)
        for ca, cb in zip(candidates_a, candidates_b, strict=True):
            assert ca["params"] == cb["params"]
            assert ca["metrics"] == cb["metrics"]


class TestEliminateUnsafeCandidates:
    def test_eliminate_unsafe_candidates(self) -> None:
        """Candidates with safety metrics worse than baseline are removed."""
        candidates = [
            {
                "params": {"n_results": 1},
                "metrics": {**_BASELINE_METRICS, "macro_strict_pass_rate": 0.6},
                "safe": True,
                "violations": [],
            },
            {
                "params": {"n_results": 2},
                "metrics": {
                    **_BASELINE_METRICS,
                    "unsupported_numeric_release_rate": 0.1,
                    "macro_strict_pass_rate": 0.9,
                },
                "safe": False,
                "violations": ["unsupported_numeric_release_rate worse"],
            },
        ]
        safe = eliminate_unsafe_candidates(candidates, _BASELINE_METRICS)
        assert len(safe) == 1
        assert safe[0]["params"] == {"n_results": 1}


class TestSelectBestCandidate:
    def test_select_maximizes_macro_strict_pass(self) -> None:
        """The candidate with the highest macro_strict_pass_rate is selected."""
        candidates = [
            {
                "params": {"n_results": 1},
                "metrics": {**_BASELINE_METRICS, "macro_strict_pass_rate": 0.6},
                "safe": True,
                "violations": [],
            },
            {
                "params": {"n_results": 2},
                "metrics": {**_BASELINE_METRICS, "macro_strict_pass_rate": 0.8},
                "safe": True,
                "violations": [],
            },
            {
                "params": {"n_results": 3},
                "metrics": {**_BASELINE_METRICS, "macro_strict_pass_rate": 0.7},
                "safe": True,
                "violations": [],
            },
        ]
        best = select_best_candidate(candidates, _BASELINE_METRICS)
        assert best["params"] == {"n_results": 2}

    def test_tiebreak_citation_recall(self) -> None:
        """When macro_strict_pass ties, higher citation_recall wins."""
        candidates = [
            {
                "params": {"n_results": 1},
                "metrics": {
                    **_BASELINE_METRICS,
                    "macro_strict_pass_rate": 0.8,
                    "citation_recall": 0.6,
                },
                "safe": True,
                "violations": [],
            },
            {
                "params": {"n_results": 2},
                "metrics": {
                    **_BASELINE_METRICS,
                    "macro_strict_pass_rate": 0.8,
                    "citation_recall": 0.9,
                },
                "safe": True,
                "violations": [],
            },
        ]
        best = select_best_candidate(candidates, _BASELINE_METRICS)
        assert best["params"] == {"n_results": 2}
