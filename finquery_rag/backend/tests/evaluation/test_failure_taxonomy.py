"""Tests for failure classification in src.evaluation.failure_taxonomy."""
from __future__ import annotations

from src.evaluation.failure_taxonomy import (
    CALCULATION_ERROR,
    CITATION_ERROR,
    INTENT_ERROR,
    RETRIEVAL_MISS,
    SYSTEM_ERROR,
    classify_all_failures,
    classify_failure,
)
from src.evaluation.schemas import (
    EvaluationLabel,
    EvaluationPrediction,
    ExpectedCalculation,
    ExpectedSource,
)


def _label(
    case_id: str = "c1",
    expected_sources: tuple[ExpectedSource, ...] = (),
    expected_numbers: tuple[str, ...] = (),
    expected_calculations: tuple[ExpectedCalculation, ...] = (),
    expected_intent: str | None = None,
    expected_answerability: str | None = None,
    expected_validation_status: str | None = None,
    expected_no_answer: bool = False,
    forbidden_answer_terms: tuple[str, ...] = (),
) -> EvaluationLabel:
    return EvaluationLabel(
        case_id=case_id,
        expected_sources=expected_sources,
        expected_numbers=expected_numbers,
        expected_calculations=expected_calculations,
        expected_intent=expected_intent,
        expected_answerability=expected_answerability,
        expected_validation_status=expected_validation_status,
        expected_no_answer=expected_no_answer,
        required_answer_terms=(),
        forbidden_answer_terms=forbidden_answer_terms,
        slice_tags=(),
    )


def _pred(
    case_id: str = "c1",
    answer: str = "The answer is 42.",
    sources: tuple[dict, ...] = (),
    retrieved_chunks: tuple[dict, ...] = (),
    calculations: tuple[dict, ...] = (),
    intent: str | None = None,
    answerability: dict | None = None,
    validation: dict | None = None,
    error_code: str | None = None,
    system_error_category: str | None = None,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        case_id=case_id,
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        calculations=calculations,
        answerability=answerability,
        validation=validation,
        warnings=(),
        intent=intent,
        intent_confidence=None,
        context_sufficient=None,
        retrieval_debug={},
        trace_id=None,
        latency_ms=100.0,
        error_code=error_code,
        system_error_category=system_error_category,
    )


def _passing_label_and_pred() -> tuple[EvaluationLabel, EvaluationPrediction]:
    """Build a label/prediction pair that passes strict evaluation."""
    src = (ExpectedSource(filename="a.pdf", page=1),)
    label = _label(
        "c1",
        expected_sources=src,
        expected_numbers=("42",),
        expected_intent="document_qa",
        expected_answerability="answerable",
        expected_validation_status="passed",
    )
    pred = _pred(
        "c1",
        answer="The answer is 42.",
        sources=({"filename": "a.pdf", "page": 1},),
        retrieved_chunks=({"filename": "a.pdf", "page": 1},),
        intent="document_qa",
        answerability={"status": "answerable"},
        validation={"status": "passed"},
    )
    return label, pred


class TestNoFailure:
    def test_no_failure_when_passed(self) -> None:
        """A passing case returns (None, [])."""
        label, pred = _passing_label_and_pred()
        primary, secondary = classify_failure(label, pred)
        assert primary is None
        assert secondary == []


class TestSystemError:
    def test_system_error_classification(self) -> None:
        """A system error is classified as SYSTEM_ERROR."""
        label = _label("c1")
        pred = _pred("c1", answer="ok", error_code="INTERNAL_ERROR")
        primary, secondary = classify_failure(label, pred)
        assert primary == SYSTEM_ERROR

    def test_auth_error_classification(self) -> None:
        label = _label("c1")
        pred = _pred(
            "c1",
            answer="",
            error_code="AuthError",
            system_error_category="auth_token_expired",
        )
        primary, _ = classify_failure(label, pred)
        from src.evaluation.failure_taxonomy import AUTH_OR_ENVIRONMENT

        assert primary == AUTH_OR_ENVIRONMENT


class TestRetrievalMiss:
    def test_retrieval_miss_classification(self) -> None:
        """Expected sources not in retrieved chunks → RETRIEVAL_MISS."""
        src = (ExpectedSource(filename="a.pdf", page=1),)
        label = _label("c1", expected_sources=src)
        pred = _pred(
            "c1",
            answer="No data.",
            sources=(),
            retrieved_chunks=({"filename": "b.pdf", "page": 2},),
        )
        primary, secondary = classify_failure(label, pred)
        assert primary == RETRIEVAL_MISS


class TestCitationError:
    def test_citation_error_classification(self) -> None:
        """Citation recall < 1.0 → CITATION_ERROR."""
        src = (
            ExpectedSource(filename="a.pdf", page=1),
            ExpectedSource(filename="b.pdf", page=2),
        )
        label = _label("c1", expected_sources=src)
        pred = _pred(
            "c1",
            answer="ok",
            sources=({"filename": "a.pdf", "page": 1},),
            retrieved_chunks=(
                {"filename": "a.pdf", "page": 1},
                {"filename": "b.pdf", "page": 2},
            ),
        )
        primary, secondary = classify_failure(label, pred)
        assert primary == CITATION_ERROR


class TestCalculationError:
    def test_calculation_error_classification(self) -> None:
        """Calculation accuracy < 1.0 → CALCULATION_ERROR."""
        calc = (
            ExpectedCalculation(
                calc_id="c1",
                operation="sum_values",
                args={},
                expected_value="300",
            ),
        )
        label = _label("c1", expected_calculations=calc)
        pred = _pred(
            "c1",
            answer="The total is 999.",
            calculations=({"id": "c1", "operation": "sum_values", "value": "999"},),
        )
        primary, secondary = classify_failure(label, pred)
        assert primary == CALCULATION_ERROR


class TestClassificationPriority:
    def test_classification_priority(self) -> None:
        """SYSTEM_ERROR takes priority over CITATION_ERROR."""
        src = (ExpectedSource(filename="a.pdf", page=1),)
        label = _label("c1", expected_sources=src)
        pred = _pred(
            "c1",
            answer="",
            sources=(),
            retrieved_chunks=({"filename": "b.pdf", "page": 2},),
            error_code="INTERNAL_ERROR",
        )
        primary, secondary = classify_failure(label, pred)
        assert primary == SYSTEM_ERROR
        # CITATION_ERROR and RETRIEVAL_MISS should be in secondary
        assert RETRIEVAL_MISS in secondary
        assert CITATION_ERROR in secondary

    def test_intent_error_before_retrieval(self) -> None:
        """INTENT_ERROR has higher priority than RETRIEVAL_MISS."""
        src = (ExpectedSource(filename="a.pdf", page=1),)
        label = _label("c1", expected_sources=src, expected_intent="document_qa")
        pred = _pred(
            "c1",
            answer="ok",
            intent="financial_calculation",
            retrieved_chunks=(),
        )
        primary, secondary = classify_failure(label, pred)
        assert primary == INTENT_ERROR
        assert RETRIEVAL_MISS in secondary


class TestClassifyAllFailures:
    def test_classify_all_failures(self) -> None:
        """classify_all_failures returns results for all cases."""
        src = (ExpectedSource(filename="a.pdf", page=1),)
        labels = [
            _label("c1", expected_sources=src, expected_numbers=("42",)),
            _label("c2"),
        ]
        preds = [
            _pred(
                "c1",
                answer="The answer is 42.",
                sources=({"filename": "a.pdf", "page": 1},),
                retrieved_chunks=({"filename": "a.pdf", "page": 1},),
            ),
            _pred("c2", answer="ok", error_code="INTERNAL_ERROR"),
        ]
        result = classify_all_failures(labels, preds)
        assert "c1" in result
        assert "c2" in result
        assert result["c1"]["passed"] is True
        assert result["c1"]["primary_failure"] is None
        assert result["c2"]["passed"] is False
        assert result["c2"]["primary_failure"] == SYSTEM_ERROR
