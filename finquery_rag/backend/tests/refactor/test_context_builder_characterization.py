"""Characterize current RAGEngine context building behavior.

These tests record the exact behavior of build_context and related methods
so that extraction to ContextBuilder preserves behavior.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    return RAGEngine(llm_client=None, model_name="test", use_hybrid=True)


class TestContextBuilderCharacterization:
    """Record current build_context behavior."""

    def test_empty_chunks_returns_empty(self):
        engine = _make_engine()
        context, sources = engine.build_context([])
        assert context == ""
        assert sources == []

    def test_single_chunk_builds_context(self):
        engine = _make_engine()
        chunks = [
            {
                "content": "Revenue was $100 million.",
                "doc_id": "report.pdf::page_1::chunk_0",
                "metadata": {"doc_name": "report.pdf", "page": 1},
                "score": 0.5,
            }
        ]
        context, sources = engine.build_context(chunks)
        assert "Revenue was $100 million" in context
        assert len(sources) == 1
        assert sources[0]["filename"] == "report.pdf"

    def test_context_includes_source_citation(self):
        engine = _make_engine()
        chunks = [
            {
                "content": "Revenue was $100 million.",
                "doc_id": "report.pdf::page_1::chunk_0",
                "metadata": {"doc_name": "report.pdf", "page": 1},
                "score": 0.5,
            }
        ]
        context, sources = engine.build_context(chunks)
        assert "report.pdf" in context

    def test_parent_context_key_extracts_key(self):
        chunk = {
            "metadata": {
                "parent_id": "report.pdf::page_5",
                "parent_excerpt": "Some excerpt text",
                "section_path": "section_1",
            }
        }
        key = RAGEngine._parent_context_key(chunk)
        assert key == "report.pdf::page_5"

    def test_parent_context_key_returns_none_without_parent(self):
        chunk = {"metadata": {}}
        key = RAGEngine._parent_context_key(chunk)
        assert key is None

    def test_compact_child_snippet_truncates(self):
        long_text = "A" * 1000
        result = RAGEngine._compact_child_snippet(long_text, max_chars=100)
        # Result is truncated + " [...]" suffix
        assert len(result) <= 110
        assert result.endswith("[...]")

    def test_compose_parent_context_combines(self):
        parent = "Parent excerpt text."
        children = ["Child 1 evidence.", "Child 2 evidence."]
        result = RAGEngine._compose_parent_context(parent, children)
        assert "Parent excerpt text" in result
        assert "Child 1 evidence" in result
