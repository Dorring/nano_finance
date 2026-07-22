"""Tests for Phase 5 evaluation schemas (EvaluationQuery, EvaluationLabel, EvaluationPrediction).

Verifies label isolation: EvaluationQuery must reject expected_* fields.
"""
from __future__ import annotations

import pytest

from src.evaluation.schemas import (
    DatasetManifest,
    EvaluationLabel,
    EvaluationPrediction,
    EvaluationQuery,
    ExpectedCalculation,
    ExpectedSource,
)


class TestEvaluationQuery:
    def test_minimal_query(self) -> None:
        q = EvaluationQuery.from_dict({
            "case_id": "q1",
            "question": "What is the revenue?",
        })
        assert q.case_id == "q1"
        assert q.question == "What is the revenue?"
        assert q.document_names == ()
        assert q.tags == ()

    def test_full_query(self) -> None:
        q = EvaluationQuery.from_dict({
            "case_id": "q1",
            "question": "What is the revenue?",
            "document_names": ["report.pdf"],
            "tags": ["financial_calculation"],
            "metadata": {"source": "annual_report"},
        })
        assert q.document_names == ("report.pdf",)
        assert q.tags == ("financial_calculation",)

    def test_id_alias(self) -> None:
        q = EvaluationQuery.from_dict({"id": "q2", "question": "Q"})
        assert q.case_id == "q2"

    def test_missing_case_id_raises(self) -> None:
        with pytest.raises(ValueError, match="missing case_id"):
            EvaluationQuery.from_dict({"question": "Q"})

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValueError, match="missing question"):
            EvaluationQuery.from_dict({"case_id": "q1"})

    def test_rejects_expected_sources(self) -> None:
        with pytest.raises(ValueError, match="must not contain label field"):
            EvaluationQuery.from_dict({
                "case_id": "q1",
                "question": "Q",
                "expected_sources": [{"filename": "r.pdf"}],
            })

    def test_rejects_expected_numbers(self) -> None:
        with pytest.raises(ValueError, match="must not contain label field"):
            EvaluationQuery.from_dict({
                "case_id": "q1",
                "question": "Q",
                "expected_numbers": ["1000"],
            })

    def test_rejects_expected_no_answer(self) -> None:
        with pytest.raises(ValueError, match="must not contain label field"):
            EvaluationQuery.from_dict({
                "case_id": "q1",
                "question": "Q",
                "expected_no_answer": True,
            })

    def test_rejects_expected_calculations(self) -> None:
        with pytest.raises(ValueError, match="must not contain label field"):
            EvaluationQuery.from_dict({
                "case_id": "q1",
                "question": "Q",
                "expected_calculations": [],
            })

    def test_round_trip(self) -> None:
        q = EvaluationQuery.from_dict({
            "case_id": "q1",
            "question": "Q",
            "document_names": ["a.pdf", "b.pdf"],
            "tags": ["t1"],
            "metadata": {"k": "v"},
        })
        d = q.to_dict()
        q2 = EvaluationQuery.from_dict(d)
        assert q2 == q


class TestEvaluationLabel:
    def test_minimal_label(self) -> None:
        label = EvaluationLabel.from_dict({"case_id": "l1"})
        assert label.case_id == "l1"
        assert label.expected_sources == ()
        assert label.expected_no_answer is False
        assert label.slice_tags == ()

    def test_full_label(self) -> None:
        label = EvaluationLabel.from_dict({
            "case_id": "l1",
            "expected_sources": [{"filename": "r.pdf", "page": 1}],
            "expected_numbers": ["1000"],
            "expected_calculations": [{
                "id": "c1",
                "operation": "sum",
                "args": {"values": ["100", "200"]},
                "expected_value": "300",
            }],
            "expected_intent": "financial_calculation",
            "expected_answerability": "answerable",
            "expected_validation_status": "passed",
            "expected_no_answer": False,
            "required_answer_terms": ["revenue"],
            "forbidden_answer_terms": ["confidential"],
            "slice_tags": ["financial_calculation", "direct"],
        })
        assert len(label.expected_sources) == 1
        assert label.expected_numbers == ("1000",)
        assert label.expected_calculations[0].operation == "sum"
        assert label.required_answer_terms == ("revenue",)
        assert label.forbidden_answer_terms == ("confidential",)

    def test_round_trip(self) -> None:
        label = EvaluationLabel.from_dict({
            "case_id": "l1",
            "expected_sources": [{"filename": "r.pdf", "page": 1}],
            "expected_numbers": ["1000"],
            "slice_tags": ["direct"],
        })
        d = label.to_dict()
        label2 = EvaluationLabel.from_dict(d)
        assert label2 == label


class TestEvaluationPrediction:
    def test_minimal_prediction(self) -> None:
        pred = EvaluationPrediction.from_dict({"case_id": "p1", "answer": "hello"})
        assert pred.case_id == "p1"
        assert pred.answer == "hello"
        assert pred.calculations == ()
        assert pred.answerability is None
        assert pred.validation is None

    def test_full_prediction_with_phase3_4_fields(self) -> None:
        pred = EvaluationPrediction.from_dict({
            "case_id": "p1",
            "answer": "The revenue is 1000.",
            "sources": [{"filename": "r.pdf", "page": 1}],
            "retrieved_chunks": [{"filename": "r.pdf", "page": 1}],
            "calculations": [{"operation": "sum", "value": "1000"}],
            "answerability": {"status": "answerable"},
            "validation": {"status": "passed"},
            "warnings": ["minor"],
            "intent": "financial_calculation",
            "intent_confidence": 0.95,
            "context_sufficient": True,
            "retrieval_debug": {"method": "hybrid"},
            "trace_id": "trace-001",
            "latency_ms": 150.0,
            "error_code": None,
        })
        assert len(pred.calculations) == 1
        assert pred.answerability == {"status": "answerable"}
        assert pred.validation == {"status": "passed"}
        assert pred.warnings == ("minor",)
        assert pred.latency_ms == 150.0

    def test_round_trip(self) -> None:
        pred = EvaluationPrediction.from_dict({
            "case_id": "p1",
            "answer": "test",
            "calculations": [{"op": "sum"}],
            "answerability": {"status": "answerable"},
            "validation": {"status": "passed"},
        })
        d = pred.to_dict()
        pred2 = EvaluationPrediction.from_dict(d)
        assert pred2 == pred


class TestDatasetManifest:
    def test_valid_partition(self) -> None:
        m = DatasetManifest.from_dict({
            "partition": "dev",
            "case_count": 40,
            "questions_sha256": "abc123",
            "labels_sha256": "def456",
            "created_at": "2026-07-23T00:00:00Z",
            "slices": ["direct", "multi-hop"],
        })
        assert m.partition == "dev"
        assert m.case_count == 40

    def test_invalid_partition(self) -> None:
        with pytest.raises(ValueError, match="partition must be one of"):
            DatasetManifest.from_dict({"partition": "invalid", "case_count": 0})

    def test_sealed_manifest_can_have_null_labels_sha(self) -> None:
        m = DatasetManifest.from_dict({
            "partition": "sealed",
            "case_count": 120,
            "questions_sha256": "abc",
            "labels_sha256": None,
            "created_at": "",
            "slices": [],
        })
        assert m.labels_sha256 is None
