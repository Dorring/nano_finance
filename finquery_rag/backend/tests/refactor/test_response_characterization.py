"""Characterize current RAGEngine response rendering behavior.

These tests record the exact behavior of _validate_answer and response
construction so that extraction to ResponseRenderer preserves behavior.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    return RAGEngine(llm_client=None, model_name="test", use_hybrid=True)


class TestResponseCharacterization:
    """Record current _validate_answer behavior."""

    def test_empty_answer_returns_refusal(self):
        engine = _make_engine()
        result = engine._validate_answer("", [])
        assert "couldn't generate" in result.lower()

    def test_whitespace_only_answer_returns_refusal(self):
        engine = _make_engine()
        result = engine._validate_answer("   ", [])
        assert "couldn't generate" in result.lower()

    def test_short_answer_returns_refusal(self):
        engine = _make_engine()
        result = engine._validate_answer("Hi", [])
        assert "couldn't generate" in result.lower()

    def test_valid_answer_returned_unchanged(self):
        engine = _make_engine()
        answer = "The total revenue was $100 million for the fiscal year."
        result = engine._validate_answer(answer, [])
        assert result == answer

    def test_artifacts_stripped(self):
        engine = _make_engine()
        answer = "The revenue was $100 million.<|end|>"
        result = engine._validate_answer(answer, [])
        assert "<|end|>" not in result
        assert "revenue" in result

    def test_long_answer_truncated(self):
        engine = _make_engine()
        answer = "A" * 10000
        result = engine._validate_answer(answer, [])
        assert len(result) < 10000

    def test_sufficiency_check_empty_chunks(self):
        engine = _make_engine()
        is_sufficient, best, avg = engine._check_context_sufficiency([])
        assert is_sufficient is False
        assert best == 0.0
        assert avg == 0.0

    def test_sufficiency_check_with_high_score(self):
        engine = _make_engine()
        chunks = [{"score": 0.5}]
        is_sufficient, best, avg = engine._check_context_sufficiency(chunks)
        assert is_sufficient is True
        assert best == 0.5

    def test_sufficiency_check_with_low_rrf_score(self):
        engine = _make_engine()
        chunks = [{"score": 0.001}]
        is_sufficient, best, avg = engine._check_context_sufficiency(chunks)
        # 0.001 < rrf_sufficiency_threshold (0.025)
        assert is_sufficient is False

    def test_confidence_empty_chunks(self):
        engine = _make_engine()
        assert engine._compute_confidence([]) == 0.0

    def test_confidence_single_chunk(self):
        engine = _make_engine()
        chunks = [{"score": 0.5}]
        confidence = engine._compute_confidence(chunks)
        assert confidence == pytest.approx(0.5)

    def test_confidence_bounded_to_one(self):
        engine = _make_engine()
        chunks = [{"score": 1.5}]
        confidence = engine._compute_confidence(chunks)
        assert confidence <= 1.0
