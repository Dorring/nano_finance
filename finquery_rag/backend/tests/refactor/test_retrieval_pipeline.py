"""Unit tests for RetrievalPipeline and CandidateFusion."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.retrieval.candidate_fusion import (
    normalize_scores,
    dedupe_chunks,
    chunk_doc_name,
    ensure_multi_doc_coverage,
    boost_front_matter_chunks,
    summarize_retrieved_chunks,
    source_from_chunk,
)


class TestNormalizeScores:
    def test_copies_fused_to_score(self):
        chunks = [{"fused_score": 0.03, "score": 0.5}]
        result = normalize_scores(chunks)
        assert result[0]["score"] == 0.03

    def test_preserves_non_fused(self):
        chunks = [{"score": 0.5}]
        result = normalize_scores(chunks)
        assert result[0]["score"] == 0.5

    def test_adds_zero_score(self):
        chunks = [{}]
        result = normalize_scores(chunks)
        assert result[0]["score"] == 0


class TestDedupeChunks:
    def test_removes_duplicates(self):
        chunks = [
            {"doc_id": "a::1", "content": "hello"},
            {"doc_id": "a::1", "content": "hello"},
            {"doc_id": "b::1", "content": "world"},
        ]
        result = dedupe_chunks(chunks)
        assert len(result) == 2

    def test_preserves_order(self):
        chunks = [
            {"doc_id": "a::1", "content": "first"},
            {"doc_id": "b::1", "content": "second"},
        ]
        result = dedupe_chunks(chunks)
        assert result[0]["doc_id"] == "a::1"


class TestChunkDocName:
    def test_from_metadata(self):
        assert chunk_doc_name({"metadata": {"doc_name": "report.pdf"}}) == "report.pdf"

    def test_from_doc_id(self):
        assert chunk_doc_name({"doc_id": "user_1_report.pdf::page_1::chunk_0"}) == "report.pdf"

    def test_returns_none(self):
        assert chunk_doc_name({}) is None


class TestEnsureMultiDocCoverage:
    def test_single_doc_passthrough(self):
        selected = [{"doc_id": "a::1", "score": 0.5}]
        result = ensure_multi_doc_coverage([], selected, ["doc_a"], 3)
        assert result == selected

    def test_adds_missing_doc(self):
        candidates = [
            {"doc_id": "a::1", "score": 0.5, "metadata": {"doc_name": "doc_a"}},
            {"doc_id": "b::1", "score": 0.3, "metadata": {"doc_name": "doc_b"}},
        ]
        selected = [candidates[0]]
        result = ensure_multi_doc_coverage(candidates, selected, ["doc_a", "doc_b"], 3)
        assert len(result) == 2


class TestBoostFrontMatterChunks:
    def test_no_boost_for_non_front_matter(self):
        chunks = [{"score": 0.5, "metadata": {"page": 1}}]
        result = boost_front_matter_chunks("What is the revenue?", chunks, is_front_matter_query_fn=lambda q: False)
        assert result[0]["score"] == 0.5

    def test_boosts_page_1_for_front_matter(self):
        chunks = [{"score": 0.5, "metadata": {"page": 1}}]
        result = boost_front_matter_chunks("What is the title?", chunks, is_front_matter_query_fn=lambda q: True)
        assert result[0]["score"] == pytest.approx(0.52)


class TestSummarizeRetrievedChunks:
    def test_excludes_content(self):
        chunks = [{"doc_id": "a::1", "content": "sensitive", "metadata": {"doc_name": "r.pdf"}, "score": 0.5}]
        result = summarize_retrieved_chunks(chunks)
        assert "content" not in result[0]


class TestSourceFromChunk:
    def test_extracts_filename_and_page(self):
        chunk = {"doc_id": "a::1", "metadata": {"doc_name": "report.pdf", "page": 3}}
        result = source_from_chunk(chunk)
        assert result["filename"] == "report.pdf"
        assert result["page"] == 3

    def test_fallback_to_doc_id(self):
        chunk = {"doc_id": "user_1_report.pdf::page_3::chunk_0", "metadata": {}}
        result = source_from_chunk(chunk)
        assert result["filename"] == "report.pdf"
