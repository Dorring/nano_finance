"""Tests for safety and utility metrics in src.evaluation.metrics."""
from __future__ import annotations

from src.evaluation.metrics import (
    false_block_rate,
    strict_case_pass,
    unsupported_numeric_release_rate,
    validator_fail_closed_rate,
)
from src.evaluation.schemas import (
    EvaluationLabel,
    EvaluationPrediction,
    ExpectedSource,
)


def _label(
    case_id: str = "c1",
    expected_sources: tuple[ExpectedSource, ...] = (),
    expected_numbers: tuple[str, ...] = (),
    expected_intent: str | None = None,
    expected_answerability: str | None = None,
    expected_validation_status: str | None = None,
    expected_no_answer: bool = False,
    forbidden_answer_terms: tuple[str, ...] = (),
    slice_tags: tuple[str, ...] = (),
) -> EvaluationLabel:
    return EvaluationLabel(
        case_id=case_id,
        expected_sources=expected_sources,
        expected_numbers=expected_numbers,
        expected_calculations=(),
        expected_intent=expected_intent,
        expected_answerability=expected_answerability,
        expected_validation_status=expected_validation_status,
        expected_no_answer=expected_no_answer,
        required_answer_terms=(),
        forbidden_answer_terms=forbidden_answer_terms,
        slice_tags=slice_tags,
    )


def _pred(
    case_id: str = "c1",
    answer: str = "The answer is 42.",
    sources: tuple[dict, ...] = (),
    retrieved_chunks: tuple[dict, ...] = (),
    intent: str | None = None,
    answerability: dict | None = None,
    validation: dict | None = None,
    error_code: str | None = None,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        case_id=case_id,
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        calculations=(),
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
    )


class TestUnsupportedNumericRelease:
    def test_unsupported_numeric_release_detected(self) -> None:
        """No-answer case where a number was released."""
        labels = [_label("c1", expected_no_answer=True)]
        preds = [_pred("c1", answer="The revenue is 42.")]
        assert unsupported_numeric_release_rate(labels, preds) == 1.0

    def test_no_release_when_refused(self) -> None:
        labels = [_label("c1", expected_no_answer=True)]
        preds = [_pred("c1", answer="Could not find relevant information.")]
        assert unsupported_numeric_release_rate(labels, preds) == 0.0


class TestInvalidCitationRelease:
    def test_invalid_citation_release_detected(self) -> None:
        """A cited source does not match any expected source."""
        from src.evaluation.metrics import invalid_citation_release_rate

        expected = (ExpectedSource(filename="a.pdf", page=1),)
        labels = [_label("c1", expected_sources=expected)]
        preds = [
            _pred(
                "c1",
                sources=(
                    {"filename": "a.pdf", "page": 1},
                    {"filename": "z.pdf", "page": 99},
                ),
            )
        ]
        rate = invalid_citation_release_rate(labels, preds)
        assert rate == 1.0

    def test_no_invalid_citation(self) -> None:
        from src.evaluation.metrics import invalid_citation_release_rate

        expected = (ExpectedSource(filename="a.pdf", page=1),)
        labels = [_label("c1", expected_sources=expected)]
        preds = [_pred("c1", sources=({"filename": "a.pdf", "page": 1},))]
        assert invalid_citation_release_rate(labels, preds) == 0.0


class TestFalseBlockRate:
    def test_false_block_rate(self) -> None:
        """Answerable case wrongly blocked."""
        labels = [
            _label("c1", expected_answerability="answerable"),
            _label("c2", expected_answerability="answerable"),
        ]
        preds = [
            _pred("c1", answer="", validation={"status": "blocked"}),
            _pred("c2", answer="OK", validation={"status": "passed"}),
        ]
        assert false_block_rate(labels, preds) == 0.5

    def test_no_false_blocks(self) -> None:
        labels = [_label("c1", expected_answerability="answerable")]
        preds = [_pred("c1", answer="OK", validation={"status": "passed"})]
        assert false_block_rate(labels, preds) == 0.0


class TestValidatorFailClosedRate:
    def test_validator_fail_closed_rate(self) -> None:
        preds = [
            _pred("c1", answer="", validation={"status": "blocked"}),
            _pred("c2", answer="OK", validation={"status": "passed"}),
            _pred("c3", answer="OK", validation={"status": "passed"}),
        ]
        assert validator_fail_closed_rate(preds) == 1.0 / 3.0


class TestStrictCasePass:
    def test_strict_case_pass_all_conditions(self) -> None:
        """All conditions satisfied → True."""
        expected_sources = (ExpectedSource(filename="a.pdf", page=1),)
        label = _label(
            "c1",
            expected_sources=expected_sources,
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
        assert strict_case_pass(label, pred) is True

    def test_strict_case_fail_on_wrong_number(self) -> None:
        """Wrong number in answer → False."""
        expected_sources = (ExpectedSource(filename="a.pdf", page=1),)
        label = _label(
            "c1",
            expected_sources=expected_sources,
            expected_numbers=("42",),
        )
        pred = _pred(
            "c1",
            answer="The answer is 99.",
            sources=({"filename": "a.pdf", "page": 1},),
            retrieved_chunks=({"filename": "a.pdf", "page": 1},),
        )
        assert strict_case_pass(label, pred) is False

    def test_strict_case_fail_on_forbidden_term(self) -> None:
        """Forbidden term in answer → False."""
        label = _label(
            "c1",
            forbidden_answer_terms=("confidential",),
        )
        pred = _pred("c1", answer="This is confidential data.")
        assert strict_case_pass(label, pred) is False

    def test_strict_case_fail_on_system_error(self) -> None:
        label = _label("c1")
        pred = _pred("c1", answer="ok", error_code="INTERNAL_ERROR")
        assert strict_case_pass(label, pred) is False
