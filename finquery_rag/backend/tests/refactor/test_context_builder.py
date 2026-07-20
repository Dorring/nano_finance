"""Unit tests for ContextBuilder and EvidenceSufficiencyEvaluator."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator, SufficiencyResult


class TestContextBuilder:
    def test_empty_chunks(self):
        cb = ContextBuilder()
        context, sources = cb.build([])
        assert context == ""
        assert sources == []

    def test_single_chunk(self):
        cb = ContextBuilder()
        chunks = [{
            "content": "Revenue was $100 million.",
            "doc_id": "report.pdf::page_1::chunk_0",
            "metadata": {"doc_name": "report.pdf", "page": 1},
            "score": 0.5,
        }]
        context, sources = cb.build(chunks)
        assert "Revenue was $100 million" in context
        assert len(sources) == 1

    def test_context_includes_source_citation(self):
        cb = ContextBuilder()
        chunks = [{
            "content": "Revenue was $100 million.",
            "doc_id": "report.pdf::page_1::chunk_0",
            "metadata": {"doc_name": "report.pdf", "page": 1},
            "score": 0.5,
        }]
        context, sources = cb.build(chunks)
        assert "report.pdf" in context

    def test_parent_context_key_with_excerpt(self):
        chunk = {"metadata": {"parent_id": "a::1", "parent_excerpt": "text"}}
        assert ContextBuilder._parent_context_key(chunk) == "a::1"

    def test_parent_context_key_without_excerpt(self):
        chunk = {"metadata": {"parent_id": "a::1"}}
        assert ContextBuilder._parent_context_key(chunk) is None

    def test_compact_child_snippet_short(self):
        assert ContextBuilder._compact_child_snippet("Hello") == "Hello"

    def test_compact_child_snippet_truncates(self):
        result = ContextBuilder._compact_child_snippet("A" * 1000, max_chars=100)
        assert result.endswith("[...]")

    def test_compose_parent_context_no_children(self):
        assert ContextBuilder._compose_parent_context("Parent", []) == "Parent"

    def test_compose_parent_context_with_children(self):
        result = ContextBuilder._compose_parent_context("Parent", ["Child 1"])
        assert "Child 1" in result
        assert "Parent" in result

    def test_min_score_threshold_filters(self):
        cb = ContextBuilder(min_score_threshold=0.3)
        chunks = [
            {"content": "Low", "doc_id": "a::1", "metadata": {"page": 1}, "score": 0.1},
            {"content": "High", "doc_id": "b::1", "metadata": {"page": 2}, "score": 0.5},
        ]
        context, sources = cb.build(chunks)
        assert len(sources) == 1
        assert "High" in context


class TestEvidenceSufficiencyEvaluator:
    def test_empty_chunks(self):
        ev = EvidenceSufficiencyEvaluator()
        result = ev.evaluate([])
        assert result.is_sufficient is False
        assert result.best_score == 0.0

    def test_high_dense_score_sufficient(self):
        ev = EvidenceSufficiencyEvaluator()
        result = ev.evaluate([{"score": 0.5}])
        assert result.is_sufficient is True

    def test_low_rrf_score_insufficient(self):
        ev = EvidenceSufficiencyEvaluator()
        result = ev.evaluate([{"score": 0.001}])
        assert result.is_sufficient is False

    def test_confidence_empty(self):
        ev = EvidenceSufficiencyEvaluator()
        assert ev.confidence([]) == 0.0

    def test_confidence_single_chunk(self):
        ev = EvidenceSufficiencyEvaluator()
        assert ev.confidence([{"score": 0.5}]) == pytest.approx(0.5)

    def test_sufficiency_result_dataclass(self):
        r = SufficiencyResult(is_sufficient=True, best_score=0.5, average_score=0.3)
        assert r.is_sufficient is True
        assert r.best_score == 0.5
