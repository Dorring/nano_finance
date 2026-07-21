"""Characterize current RAGEngine retrieval behavior.

These tests record the exact behavior of retrieval-related methods
so that extraction to RetrievalPipeline preserves behavior.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    return RAGEngine(llm_client=None, model_name="test", use_hybrid=True)


class TestRetrievalCharacterization:
    """Record current retrieval helper behavior."""

    def test_normalize_scores_copies_fused_to_score(self):
        engine = _make_engine()
        chunks = [
            {"fused_score": 0.03, "score": 0.5},
            {"fused_score": 0.02, "score": 0.3},
        ]
        result = engine._normalize_scores(chunks)
        assert result[0]["score"] == 0.03
        assert result[1]["score"] == 0.02

    def test_normalize_scores_preserves_non_fused(self):
        engine = _make_engine()
        chunks = [{"score": 0.5}]
        result = engine._normalize_scores(chunks)
        assert result[0]["score"] == 0.5

    def test_dedupe_chunks_removes_duplicates(self):
        chunks = [
            {"doc_id": "a::1", "content": "hello"},
            {"doc_id": "a::1", "content": "hello"},
            {"doc_id": "b::1", "content": "world"},
        ]
        result = RAGEngine._dedupe_chunks(chunks)
        assert len(result) == 2

    def test_dedupe_chunks_preserves_order(self):
        chunks = [
            {"doc_id": "a::1", "content": "first"},
            {"doc_id": "b::1", "content": "second"},
            {"doc_id": "a::1", "content": "first"},
        ]
        result = RAGEngine._dedupe_chunks(chunks)
        assert result[0]["doc_id"] == "a::1"
        assert result[1]["doc_id"] == "b::1"

    def test_chunk_doc_name_extracts_from_metadata(self):
        chunk = {"metadata": {"doc_name": "report.pdf"}}
        assert RAGEngine._chunk_doc_name(chunk) == "report.pdf"

    def test_chunk_doc_name_extracts_from_doc_id(self):
        chunk = {"doc_id": "user_1_report.pdf::page_1::chunk_0"}
        assert RAGEngine._chunk_doc_name(chunk) == "report.pdf"

    def test_chunk_doc_name_returns_none_for_empty(self):
        chunk = {}
        assert RAGEngine._chunk_doc_name(chunk) is None

    def test_make_retrieval_debug_structure(self):
        engine = _make_engine()
        debug = engine._make_retrieval_debug(10, 3)
        assert debug["candidate_count"] == 10
        assert debug["returned_count"] == 3

    def test_boost_front_matter_no_boost_for_non_front_matter(self):
        engine = _make_engine()
        chunks = [
            {"score": 0.5, "metadata": {"page": 1}},
        ]
        result = engine._boost_front_matter_chunks("What is the revenue?", chunks)
        assert result[0]["score"] == 0.5

    def test_boost_front_matter_boosts_page_1_for_front_matter(self):
        engine = _make_engine()
        chunks = [
            {"score": 0.5, "metadata": {"page": 1}},
        ]
        result = engine._boost_front_matter_chunks("What is the title?", chunks)
        assert result[0]["score"] == pytest.approx(0.52)
        assert result[0]["front_matter_boost"] == 0.02

    def test_source_from_chunk_extracts_filename_and_page(self):
        chunk = {
            "doc_id": "user_1_report.pdf::page_3::chunk_0",
            "metadata": {"doc_name": "report.pdf", "page": 3},
        }
        result = RAGEngine._source_from_chunk(chunk)
        assert result["filename"] == "report.pdf"
        assert result["page"] == 3

    def test_source_from_chunk_fallback_to_doc_id(self):
        chunk = {
            "doc_id": "user_1_report.pdf::page_3::chunk_0",
            "metadata": {},
        }
        result = RAGEngine._source_from_chunk(chunk)
        assert result["filename"] == "report.pdf"

    def test_summarize_retrieved_chunks_excludes_content(self):
        chunks = [
            {
                "doc_id": "a::1",
                "content": "sensitive content",
                "metadata": {"doc_name": "report.pdf", "page": 1},
                "score": 0.5,
            }
        ]
        result = RAGEngine._summarize_retrieved_chunks(chunks)
        assert len(result) == 1
        assert "content" not in result[0]
        assert result[0]["doc_id"] == "a::1"
